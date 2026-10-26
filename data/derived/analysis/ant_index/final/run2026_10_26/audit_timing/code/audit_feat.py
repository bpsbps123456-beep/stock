"""판정 시점 감사 1: 포럼 특징 미래 교란 시험(부록 15 스냅샷, 2026-10-26 실행분).

동결 코드(interim/code loader·feat_activity·audit_leak)는 한 줄도 고치지 않고 불러 쓴다. 바꾸는 것은 입력 범위뿐이다.
  - 스냅샷: final/run2026_10_26/snapshot_manifest.json의 바이트만 읽는다(포럼 원본은 읽기만).
  - 메모리 때문에 마감 T마다 파일 부분집합 [T − 80일, T + 12일](자료 범위로 자름)만 쓴다. 한 글의 줄은 한 파일에만 있다
    (loader_summary resolve_stats.files_multi_no = 0) → 부분집합 안의 글은 줄이 모두 들어온다. 기준 실행과 교란 실행이 같은
    부분집합을 쓰므로 '마감 ≤ T 창이 T 뒤 정보로 바뀌는가'라는 차분 시험은 그대로다.
  - 교란: audit_leak.perturb 그대로(A: T 뒤 글·댓글 내용·작성자, 모든 수치 필드, 다시 받은 줄 제목·본문·ncomment, T 뒤 새 글 300개,
    T 앞 글 30%에 T 뒤 댓글만 든 새 줄 / B: A + T 뒤 글·댓글 20% 지우기 / B_keeproots: B인데 깊이 0 댓글은 안 지움).
  - 다시 돌리는 단계: loader.build → feat_activity.build(판정 4칸의 포럼 특징 A_rz8·A_saved_rz8·fight·ln_n_comments_onposts·
    trunc_share·missing_rate·outage가 모두 여기서 나온다). feat_text·feat_topic은 판정 4칸 입력이 아니라 이번에는 돌리지 않는다.
  - 비교: 마감 ≤ T인 행의 모든 공통 열(windows_kr·windows_us 전 변형, feat_activity 창·주말·H40 주).
  - 재현: 기준 실행의 판정 열이 저장 파일(results/feat_activity.parquet·windows_kr/us)과 같은지, T 앞 15일 창으로 본다
    (부분집합 앞쪽은 예열이 모자라 다를 수 있어 보지 않는다).
산출: audit_timing/results/audit_feat.json. 임시 파일은 audit_timing/tmp/ 아래에 만들고 끝나면 지운다.
실행: py -B audit_feat.py [--only T1,T2]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time as _time  # noqa: E402
from datetime import date, datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
AUD = HERE.parent
RUN = AUD.parent
ANT = RUN.parent.parent
INTERIM_CODE = ANT / "interim" / "code"
sys.path.insert(0, str(INTERIM_CODE))

import loader as LD  # noqa: E402
import feat_activity as FA  # noqa: E402
import audit_leak as AL  # noqa: E402

RES = RUN / "results"
OUT = AUD / "results"
TMP = AUD / "tmp"
MAN = RUN / "snapshot_manifest.json"
SEED = 20261026
JUDGE_COLS_KR = ["A_rz8", "A_saved_rz8", "fight", "ln_n_comments_onposts", "trunc_share", "NW90_rz8", "Bperp_rz8",
                 "WK_rz8", "U_rz8", "N_rz8"]
WIN_COLS = ["missing_rate", "outage", "over96", "trunc_share", "d_no", "n_posts", "n_comments", "complete"]

# 마감 T(창 마감 경계 = [open, close)의 close). KR 07:59 마감 → 08:00, US 08:29 ET 마감 → 21:30/22:30 KST
CUTS = {
    "T1": pd.Timestamp("2025-03-14 08:00"),   # 이번에 새로 들어온 2025 상반기(M0 적합·2025 보조)
    "T2": pd.Timestamp("2025-06-24 21:30"),   # 미국 창(서머타임) 마감
    "T3": pd.Timestamp("2025-08-20 08:00"),   # 새 2025 자료와 앞 스냅샷의 경계
    "T4": pd.Timestamp("2026-04-15 08:00"),   # 앞 구간
    "T5": pd.Timestamp("2026-08-14 08:00"),   # 뒤 구간(라벨 판정 기간)
    "T6": pd.Timestamp("2026-10-16 08:00"),   # 실시간 수집 기간(다시 받은 줄이 있는 구간), 자료 끝까지
    "T7": pd.Timestamp("2026-10-22 21:30"),   # 미국 창 끝 무렵, 자료 끝까지
}


def subset_manifest(man: dict, T: pd.Timestamp, back_days: int = 80, fwd_days: int = 12) -> dict:
    lo = (T - pd.Timedelta(days=back_days)).date()
    hi = (T + pd.Timedelta(days=fwd_days)).date()
    files = {k: v for k, v in man["files"].items() if lo.isoformat() <= k[:10] <= hi.isoformat()}
    cutoff = max(k[:10] for k in files)
    return {"cutoff": cutoff, "S": man["S"], "files": files, "first": min(k[:10] for k in files)}


def write_inputs(b: dict, tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    b["units"].to_parquet(tmp / "units.parquet", index=False)
    b["windows_kr"].to_parquet(tmp / "windows_kr.parquet", index=False)
    b["windows_us"].to_parquet(tmp / "windows_us.parquet", index=False)
    shutil.copy(RES / "loader_calendar.parquet", tmp / "loader_calendar.parquet")
    (tmp / "loader_summary.json").write_text(json.dumps({"snapshot": {"t_end": str(b["t_end"])}}), "utf-8")


def run_fa(b: dict, tmp: Path) -> dict[str, pd.DataFrame]:
    write_inputs(b, tmp)
    fb = FA.build(results=tmp)
    fa = pd.concat([fb["fk"], fb["fu"]], ignore_index=True, sort=False)
    fa = fa.drop(columns=["outage", "spam_wave"]).rename(columns={"outage_b": "outage", "spam_b": "spam_wave"})
    shutil.rmtree(tmp, ignore_errors=True)
    return {"fa": fa, "fa_wkend": fb["wkend"], "fa_h40": fb["h40"],
            "wk": b["windows_kr"].copy(), "wu": b["windows_us"].copy()}


KEYS = {"fa": ["win_id"], "fa_wkend": ["sat"], "fa_h40": ["market", "stock", "week_close_day"],
        "wk": ["win_id"], "wu": ["win_id"]}


def compare(base: dict, pert: dict, T: pd.Timestamp) -> dict:
    """audit_leak.compare와 같은 규칙(마감 ≤ T 행, 공통 열 전부, 수치 1e-9 상대 허용, 결측끼리 같음)."""
    rep = {}
    for name, keys in KEYS.items():
        a, b = base[name].copy(), pert[name].copy()
        for df in (a, b):
            for k in keys:
                if k in ("week_close_day", "sat"):
                    df[k] = pd.to_datetime(df[k])
        a = a[pd.to_datetime(a["close_kst"]) <= T]
        b = b[pd.to_datetime(b["close_kst"]) <= T]
        a = a.drop_duplicates(keys).set_index(keys).sort_index()
        b = b.drop_duplicates(keys).set_index(keys).sort_index()
        missing = sorted(map(str, set(a.index) ^ set(b.index)))
        idx = a.index.intersection(b.index)
        a, b = a.loc[idx], b.loc[idx]
        diff = {}
        for c in a.columns.intersection(b.columns):
            x, y = a[c], b[c]
            xn, yn = AL._num(x), AL._num(y)
            if xn is not None and yn is not None:
                bad = ~((np.isnan(xn) & np.isnan(yn)) | (np.abs(xn - yn) <= 1e-9 * np.maximum(1, np.abs(xn))))
            else:
                xm, ym = pd.isna(x).to_numpy(), pd.isna(y).to_numpy()
                xs = np.array([str(v) for v in x.to_numpy(object)], dtype=object)
                ys = np.array([str(v) for v in y.to_numpy(object)], dtype=object)
                bad = ~((xm & ym) | (~xm & ~ym & (xs == ys)))
            if bad.any():
                diff[c] = {"rows": int(bad.sum()), "first": str(idx[np.flatnonzero(bad)[0]]),
                           "rows_ids": [str(v) for v in idx[np.flatnonzero(bad)][:8]]}
        rep[name] = {"rows_compared": int(len(idx)), "n_key_mismatch": len(missing), "key_mismatch": missing[:5],
                     "cols_compared": int(len(a.columns.intersection(b.columns))), "diff": diff}
    return rep


def reproduce(f0: dict, T: pd.Timestamp) -> dict:
    """기준 실행(부분집합)의 판정 열이 저장 산출과 같은가: 마감이 (T − 15일, T]인 창."""
    lo = T - pd.Timedelta(days=15)
    saved_fa = pd.read_parquet(RES / "feat_activity.parquet")
    out = {}
    for mk, wname in (("kr", "wk"), ("us", "wu")):
        a = f0["fa"][f0["fa"]["market"] == mk].set_index("win_id")
        s = saved_fa[saved_fa["market"] == mk].set_index("win_id")
        cl = pd.to_datetime(a["close_kst"])
        ids = a.index[(cl > lo) & (cl <= T)].intersection(s.index)
        cols = [c for c in JUDGE_COLS_KR if c in a.columns and c in s.columns]
        d = {}
        for c in cols:
            x, y = a.loc[ids, c].astype(float).to_numpy(), s.loc[ids, c].astype(float).to_numpy()
            bad = ~((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-9 * np.maximum(1, np.abs(y))))
            d[c] = int(bad.sum())
        sw = pd.read_parquet(RES / f"windows_{mk}.parquet")
        sw = sw[sw["variant"] == "main"].set_index("win_id")
        aw = f0[wname][f0[wname]["variant"] == "main"].set_index("win_id")
        wids = aw.index[(pd.to_datetime(aw["close_kst"]) > lo) & (pd.to_datetime(aw["close_kst"]) <= T)].intersection(sw.index)
        for c in WIN_COLS:
            x = pd.to_numeric(aw.loc[wids, c].astype("Float64"), errors="coerce").to_numpy(float, na_value=np.nan)
            y = pd.to_numeric(sw.loc[wids, c].astype("Float64"), errors="coerce").to_numpy(float, na_value=np.nan)
            bad = ~((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-9 * np.maximum(1, np.abs(y))))
            d[f"win:{c}"] = int(bad.sum())
        out[mk] = {"windows": int(len(ids)), "win_rows": int(len(wids)), "n_diff_by_col": d}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(CUTS))
    a = ap.parse_args(argv)
    t0 = _time.time()
    man = json.loads(MAN.read_text("utf-8"))
    cal = LD.trading_calendars(LD.CAL_START, date.fromisoformat(man["cutoff"]), RES / "loader_calendar.parquet")
    S = datetime.fromisoformat(man["S"])
    high = man["state_high"]
    report = {"created": datetime.now().isoformat(timespec="seconds"), "manifest": str(MAN), "S": man["S"],
              "code": {"loader.py": AL.hashlib.sha256((INTERIM_CODE / "loader.py").read_bytes()).hexdigest(),
                       "feat_activity.py": AL.hashlib.sha256((INTERIM_CODE / "feat_activity.py").read_bytes()).hexdigest(),
                       "audit_leak.py": AL.hashlib.sha256((INTERIM_CODE / "audit_leak.py").read_bytes()).hexdigest()},
              "cuts": {}}
    outp = OUT / "audit_feat.json"
    if outp.exists():
        old = json.loads(outp.read_text("utf-8"))
        report["cuts"].update(old.get("cuts", {}))
    for k, key in enumerate(CUTS):
        if key not in a.only.split(","):
            continue
        T = CUTS[key]
        sm = subset_manifest(man, T)
        rows, man2 = LD.read_snapshot(LD.POSTS_DIR, sm["cutoff"], sm)
        man2["cutoff"] = sm["cutoff"]
        base = {"man": man2}
        b0 = LD.build(rows, man2, cal, {"high": high})
        f0 = run_fa(b0, TMP / f"{key}_base")
        rec = {"T": str(T), "files": [sm["first"], sm["cutoff"]], "n_files": len(sm["files"]), "n_rows": len(rows),
               "t_end": str(b0["t_end"]), "reproduce_saved": reproduce(f0, T), "modes": {}}
        print(key, T, "base done", round(_time.time() - t0), flush=True)
        modes = [("A", False, False), ("B", True, False)]
        for tag, delete, keep in modes:
            rows_p, st, max_no = AL.perturb(rows, b0["units"], T, S, SEED + k, delete=delete, keep_roots=keep)
            bp = LD.build(rows_p, man2, cal, {"high": max(high, max_no)})
            fp = run_fa(bp, TMP / f"{key}_{tag}")
            cmpr = compare(f0, fp, T)
            us = AL.unit_shift_check(b0["units"], bp["units"], T)
            nd = sum(len(v["diff"]) for v in cmpr.values())
            rec["modes"][tag] = {"perturb": st, "compare": cmpr, "units": us, "n_diff_cols": nd}
            print(key, tag, "diff cols", nd, round(_time.time() - t0), flush=True)
            if tag == "B" and nd:
                rows_p, st, max_no = AL.perturb(rows, b0["units"], T, S, SEED + k, delete=True, keep_roots=True)
                bp = LD.build(rows_p, man2, cal, {"high": max(high, max_no)})
                fp = run_fa(bp, TMP / f"{key}_Bk")
                cmpr = compare(f0, fp, T)
                nd2 = sum(len(v["diff"]) for v in cmpr.values())
                rec["modes"]["B_keeproots"] = {"perturb": st, "compare": cmpr,
                                               "units": AL.unit_shift_check(b0["units"], bp["units"], T),
                                               "n_diff_cols": nd2}
                print(key, "B_keeproots diff cols", nd2, round(_time.time() - t0), flush=True)
            del rows_p, bp, fp
        report["cuts"][key] = rec
        OUT.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), "utf-8")
        del rows, b0, f0
    shutil.rmtree(TMP, ignore_errors=True)
    report["seconds"] = round(_time.time() - t0)
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), "utf-8")
    print("wrote", outp, report["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
