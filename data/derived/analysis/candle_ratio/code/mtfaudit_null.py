"""mtfaudit 3: 귀무 이동(mtf_run.shifter) 점검.

- 달력: 144종목 5분봉 평가일 합집합을 따로 만들어 mtf_run.calendar·mtf_meta와 대조.
- 항등: k = 0, D, 2D에서 shifted(flags) == flags.
- 같은 시각 칸: 봉 번호를 flag처럼 넣어 이동 → 출처 봉이 (d − k) mod D 날의 같은 시각 칸인지, 빈 칸이면 바로 앞 봉 출처를 잇는지, 예열 봉은 그대로인지.
- 빈 칸 비율: 144종목 × b 9개 × k 40개, 평가 봉 중 출처 칸이 빈 비율.
- mtf_rows: 원안(BASE) 행이 k마다 같은지(flag와 무관해야 함), 모든 종목 k 집합이 같은지.

py code/mtfaudit_null.py → results/mtfaudit_null.json, results/mtfaudit_null_empty.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine  # noqa: E402
import mtf_run  # noqa: E402
import zones_mtf as Z  # noqa: E402

OUT = engine.HERE / "results"
SYMS3 = ["MU", "BE", "NBIS"]
SEL, CONF = engine.WIN["5m"]["sel"], engine.WIN["5m"]["conf"]


def my_calendar():
    days = set()
    for s in mtf_run.SYMS:
        d = engine.load(s, "5m")
        dd = pd.Series(d.index.date).unique()
        days.update(x for x in dd if SEL[0] <= x <= SEL[1] or CONF[0] <= x <= CONF[1])
    return sorted(days)


def probe(sym, cal, ks):
    D = len(cal)
    dix = {d: i for i, d in enumerate(cal)}
    out = []
    for b in Z.BASE_TFS:
        P = Z.prepare(sym, b)
        sh = mtf_run.shifter(P, cal)
        tb = P["tb"]
        n = len(tb)
        ev = np.flatnonzero(P["seg"] >= 0)
        ident = all(all((sh(P["flags"], k)[nm] == v).all() for nm, v in P["flags"].items()) for k in (0, D, 2 * D))
        st = pd.DatetimeIndex(tb.start)
        slot = (np.asarray(st.hour * 60 + st.minute) - 240) // b
        dayi = np.array([dix.get(d, -1) for d in tb.day.values])
        cell = {(dayi[j], slot[j]): j for j in ev}
        idx = {"idx": np.arange(1, n + 1, dtype=float)}
        bad_same = bad_ffill = warm_changed = 0
        empty = []
        flag_share = []
        for k in ks:
            got = sh(idx, k)["idx"]
            warm_changed += int((got[P["seg"] < 0] != idx["idx"][P["seg"] < 0]).sum())
            src = got[ev].astype(np.int64) - 1
            prev = -1
            ne = 0
            for t, j in enumerate(ev):
                want = cell.get(((dayi[j] - k) % D, slot[j]))
                if want is not None:
                    if src[t] != want or slot[src[t]] != slot[j] or st[src[t]].time() != st[j].time():
                        bad_same += 1
                    prev = src[t]
                else:
                    ne += 1
                    if src[t] != prev:
                        bad_ffill += 1
            empty.append(ne / len(ev))
            fl = sh(P["flags"], k)
            flag_share.append(np.mean([fl[nm][ev].mean() - P["flags"][nm][ev].mean() for nm in P["flags"]]))
        out.append(dict(sym=sym, b=b, identity_0_D_2D=bool(ident), same_slot_errors=bad_same, ffill_errors=bad_ffill,
                        warm_changed=warm_changed, empty_share_mean=float(np.mean(empty)), empty_share_max=float(np.max(empty)),
                        flag_share_shift_mean=float(np.mean(flag_share))))
        print(out[-1], flush=True)
    return out


def empty_all(cal, ks):
    """144종목 × b: 출처 칸이 빈 평가 봉 비율(k 40개 평균·최대)과 빈 칸이 몰린 시간대."""
    D = len(cal)
    dix = {d: i for i, d in enumerate(cal)}
    rows = []
    for s in mtf_run.SYMS:
        d = engine.load(s, "5m")
        for b in Z.BASE_TFS:
            tb = engine.tf_bars(d, b, "eth")
            day = tb.day.values
            m = np.array([x in dix for x in day])
            tb = tb[m]
            di = np.array([dix[x] for x in tb.day.values])
            st = pd.DatetimeIndex(tb.start)
            sl = (np.asarray(st.hour * 60 + st.minute) - 240) // b
            occ = np.zeros((D, sl.max() + 1), bool)
            occ[di, sl] = True
            e = np.array([(~occ[(di - k) % D, sl]).mean() for k in ks])
            rth = (sl * b + 240 >= 570) & (sl * b + 240 < 960)
            e_rth = np.array([(~occ[(di - k) % D, sl])[rth].mean() for k in ks])
            rows.append(dict(sym=s, b=b, bars=len(tb), days=len(set(di)), empty_mean=e.mean(), empty_max=e.max(),
                             empty_rth_mean=e_rth.mean()))
    return pd.DataFrame(rows)


def rows_checks():
    g = pd.read_parquet(OUT / "mtf_rows.parquet", columns=["sym", "b", "clf", "action", "board", "k", "ret_sel", "ret_conf"],
                        filters=[("clf", "==", "BASE")])
    base0 = g[g.k == 0].set_index(["sym", "b", "board"])[["ret_sel", "ret_conf"]]
    gk = g[g.k > 0].join(base0, on=["sym", "b", "board"], rsuffix="_0")
    diff = float(np.nanmax(np.abs(gk[["ret_sel", "ret_conf"]].values - gk[["ret_sel_0", "ret_conf_0"]].values)))
    return dict(base_rows_k_gt0=len(gk), base_max_diff_vs_k0=diff)


def main():
    meta = json.load(open(OUT / "mtf_meta.json", encoding="utf-8"))
    mine = my_calendar()
    used = mtf_run.calendar(mtf_run.SYMS)
    D = len(used)
    ks = list(range(mtf_run.GUARD, D - mtf_run.GUARD + 1))
    cal_info = dict(mine_D=len(mine), used_D=D, meta_D=meta["D"], same=mine == used, first=str(used[0]), last=str(used[-1]),
                    ks=[ks[0], ks[-1], len(ks)], meta_shifts=meta["shifts"])
    print(cal_info, flush=True)
    pr = []
    for s in SYMS3:
        pr += probe(s, used, ks)
    ea = empty_all(used, ks)
    ea.to_csv(OUT / "mtfaudit_null_empty.csv", index=False)
    byb = ea.groupby("b").agg(empty_median=("empty_mean", "median"), empty_p90=("empty_mean", lambda x: x.quantile(0.9)),
                               empty_max=("empty_max", "max"), empty_rth_max=("empty_rth_mean", "max"),
                               short_days=("days", lambda x: int((x < D).sum())))
    worst = ea.sort_values("empty_mean", ascending=False).head(8)[["sym", "b", "days", "empty_mean", "empty_max"]]
    rc = rows_checks()
    summ = dict(calendar=cal_info, probe=pr, empty_by_b=byb.round(4).reset_index().to_dict("records"),
                worst=worst.round(4).to_dict("records"), syms_missing_days=sorted(set(ea[ea.days < D].sym)), rows=rc)
    json.dump(summ, open(OUT / "mtfaudit_null.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(byb.round(4).to_string())
    print(worst.round(4).to_string())
    print(rc, summ["syms_missing_days"])


if __name__ == "__main__":
    main()
