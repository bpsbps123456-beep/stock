"""mtfaudit 2: 미리보기 감사. 자료를 시각 T에서 자른 캐시로 zones_mtf.prepare를 다시 돌려, T까지 끝난 b봉의 s·flag가 전체 계산과 같은지.

- 5분봉: 시가 < T인 봉만(끝 ≤ T). 1시간봉: 시각 < T. 일봉: 그날 16:00 ET ≤ T인 날만.
- 비교 대상: 전체 계산에서 끝 ≤ T인 b봉(예열 봉 포함). 불일치는 'T에 걸친 미완성 z봉'(잘린 자료에서는 끝난 봉처럼 보임) 탓인지 분류한다.
  미완성 z봉 탓이면 전체 계산은 그 봉을 안 쓴 것(미리보기 없음)이고, 그 밖의 불일치는 미리보기 의심이다.
- 직접 점검: 전체 계산에서 고른 z봉 끝 ≤ b봉 끝, 같은 시각에 끝나는 z봉 비율, 16:00에 끝나는 b봉이 그날 일봉을 쓰는 수(awake면 체결 가능).

py code/mtfaudit_lookahead.py → results/mtfaudit_lookahead.csv, results/mtfaudit_lookahead.json
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine  # noqa: E402
import zones_mtf as Z  # noqa: E402

ET = "America/New_York"
SYMS = ["MU", "BE", "NBIS"]
TS = ["2026-07-17 05:10", "2026-07-17 09:30", "2026-07-17 16:00", "2026-08-05 11:20", "2026-08-20 16:05",
      "2026-09-04 20:00", "2026-09-08 04:05", "2026-09-24 15:55", "2026-10-09 09:35", "2026-10-22 13:40", "2026-10-22 20:00"]
OUT = engine.HERE / "results"
REAL_CACHE = engine.CACHE


def use_cache(p):
    engine.CACHE = p
    Z.CACHE = p


def write_trunc(sym, T, tmp):
    for base in ("5m", "1h"):
        d = pd.read_parquet(REAL_CACHE / f"{sym}_{base}.parquet")
        d[d.index < T].to_parquet(tmp / f"{sym}_{base}.parquet")
    d1 = pd.read_parquet(REAL_CACHE / f"{sym}_1d.parquet")
    close = pd.DatetimeIndex(pd.to_datetime(pd.DatetimeIndex(d1.index).date)).tz_localize(ET) + pd.Timedelta(hours=16)
    d1[np.asarray(close <= T)].to_parquet(tmp / f"{sym}_1d.parquet")


def partial_last(z_full, z_tr):
    """잘린 자료의 마지막 z봉이 전체 자료의 같은 (날, 시작) 봉보다 짧으면(미완성) 그 끝 시각을, 아니면 None."""
    if not len(z_tr):
        return None
    last = z_tr.iloc[-1]
    m = z_full[(z_full.day == last.day) & (z_full.start == last.start)]
    if len(m) and int(m.end.iloc[0]) != int(last.end):
        return int(last.end)
    return None


def direct_checks(sym):
    """전체 계산에서 z봉 대응을 직접 본다."""
    rows = []
    for b in Z.BASE_TFS:
        tb = Z.bars_with_warm(sym, b)
        be = tb.end.values.astype(np.int64)
        ev = (tb.day.values >= Z.START5)
        aw = Z.awake_mask(tb.start)
        for z in Z.zone_tfs(b)[1:]:
            if z == "D":
                ze = Z.daily_values(sym)["end"]
            else:
                ze = Z.bars_with_warm(sym, z).end.values.astype(np.int64)
            idx = np.searchsorted(ze, be, side="right") - 1
            ok = idx >= 0
            later = ok & (ze[np.clip(idx, 0, None)] > be)
            nxt_le = ok & (idx + 1 < len(ze)) & (ze[np.clip(idx + 1, 0, len(ze) - 1)] <= be)
            same = ok & (ze[np.clip(idx, 0, None)] == be)
            # 결정은 다음 b봉 시가에 체결된다. 다음 b봉이 awake일 때만 실제 판에서 바로 체결
            nxt_aw = np.r_[aw[1:], False]
            rows.append(dict(sym=sym, b=b, z=str(z), bars_eval=int(ev.sum()), z_end_after_b_end=int(later.sum()),
                             not_last_completed=int(nxt_le.sum()), same_close_share=float(same[ev].mean()),
                             same_close_next_awake=int((same & ev & nxt_aw).sum())))
    return rows


def main():
    tmp = Path(tempfile.mkdtemp(prefix="mtfaudit_"))
    rec, direct = [], []
    try:
        for sym in SYMS:
            use_cache(REAL_CACHE)
            full = {b: Z.prepare(sym, b) for b in Z.BASE_TFS}
            zfull = {z: Z.bars_with_warm(sym, z) for b in Z.BASE_TFS for z in Z.zone_tfs(b) if z != "D"}
            direct += direct_checks(sym)
            for ts in TS:
                T = pd.Timestamp(ts, tz=ET)
                write_trunc(sym, T, tmp)
                use_cache(tmp)
                ztr = {z: Z.bars_with_warm(sym, z) for z in zfull}
                for b in Z.BASE_TFS:
                    F = full[b]
                    tbf = F["tb"]
                    M = int((tbf.end.values <= T.value).sum())
                    assert (tbf.end.values[:M] <= T.value).all()
                    try:
                        Tt = Z.prepare(sym, b)
                    except Exception as e:  # 자료가 너무 짧으면
                        rec.append(dict(sym=sym, T=ts, b=b, flag="ERROR", compared=0, mismatch=-1, explained=0, note=repr(e)[:80]))
                        continue
                    tbt = Tt["tb"]
                    bar_same = len(tbt) >= M and np.allclose(tbt[["O", "H", "L", "C", "V"]].values[:M].astype(float),
                                                             tbf[["O", "H", "L", "C", "V"]].values[:M].astype(float), atol=1e-9, rtol=0) \
                        and (tbt.end.values[:M] == tbf.end.values[:M]).all()
                    s_mis = int((Tt["s"][:M] != F["s"][:M]).sum())
                    rec.append(dict(sym=sym, T=ts, b=b, flag="bars", compared=M, mismatch=int(not bar_same), explained=0, note=""))
                    rec.append(dict(sym=sym, T=ts, b=b, flag="s", compared=M, mismatch=s_mis, explained=0, note=""))
                    be = tbf.end.values[:M].astype(np.int64)
                    for nm in F["flags"]:
                        mis = Tt["flags"][nm][:M] != F["flags"][nm][:M]
                        zs = nm.split("_")[0][1:]
                        expl = np.zeros(M, bool)
                        if zs not in ("D", str(b)):
                            pe = partial_last(zfull[int(zs)], ztr[int(zs)])
                            if pe is not None:
                                expl = be >= pe
                        rec.append(dict(sym=sym, T=ts, b=b, flag=nm, compared=M, mismatch=int(mis.sum()),
                                        explained=int((mis & expl).sum()), note=""))
                use_cache(REAL_CACHE)
            print(sym, "완료", flush=True)
    finally:
        use_cache(REAL_CACHE)
        shutil.rmtree(tmp, ignore_errors=True)
    r = pd.DataFrame(rec)
    r["unexplained"] = r.mismatch - r.explained
    r.to_csv(OUT / "mtfaudit_lookahead.csv", index=False)
    dc = pd.DataFrame(direct)
    dc.to_csv(OUT / "mtfaudit_lookahead_direct.csv", index=False)
    fl = r[~r.flag.isin(["bars", "s", "ERROR"])]
    summ = dict(syms=SYMS, T=TS, n_T=len(TS), b=Z.BASE_TFS, prepares=int(len(r[r.flag == "s"])), errors=int((r.flag == "ERROR").sum()),
                bars_mismatch=int(r[r.flag == "bars"].mismatch.sum()), s_compared=int(r[r.flag == "s"].compared.sum()),
                s_mismatch=int(r[r.flag == "s"].mismatch.sum()), flag_values_compared=int(fl.compared.sum()),
                flag_mismatch=int(fl.mismatch.sum()), flag_mismatch_partial_z=int(fl.explained.sum()),
                flag_mismatch_unexplained=int(fl.unexplained.sum()),
                mismatch_by_T=fl.groupby("T").mismatch.sum().to_dict(),
                direct_z_end_after_b_end=int(dc.z_end_after_b_end.sum()), direct_not_last_completed=int(dc.not_last_completed.sum()),
                direct_same_close_share_median=float(dc.same_close_share.median()),
                daily_same_close_next_awake=int(dc[dc.z == "D"].same_close_next_awake.sum()))
    json.dump(summ, open(OUT / "mtfaudit_lookahead.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    bad = fl[fl.unexplained != 0]
    if len(bad):
        print(bad.head(40).to_string())


if __name__ == "__main__":
    main()
