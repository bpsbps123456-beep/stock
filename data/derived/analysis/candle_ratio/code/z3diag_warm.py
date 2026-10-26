"""z3 diag 갈래 점검: 5분봉 창은 예열 시작 = 고르기 시작(7/17)이라 첫 며칠은 긴 창 표지(HTF 160·320, FLIP K80, ER K40, ADX)가
미완성이다(zones3.prepare가 NaN을 거짓=애매로 바꾸고, HTF 상태는 예열 중 0). 고르기 구간을 7/23부터(첫 4거래일 제외)로 잘라 다시 잰다.
py code/z3diag_warm.py → results/z3diag_warm_sel.csv
"""
from __future__ import annotations

from datetime import date
from multiprocessing import Pool

import numpy as np
import pandas as pd

from z3diag_analyze import null_cmp, stock_stats
from z3diag_run import BOARDS, GUARD, OUT, SYMS, bar_sets, cands_for, shifted_src, work

START = date(2026, 7, 23)

if __name__ == "__main__":
    with Pool(3) as pool:
        data = sorted(pool.imap_unordered(work, [(s, "5m") for s in SYMS], chunksize=4), key=lambda R: R["sym"])
    cal = sorted(set().union(*[set(R["day"].tolist()) for R in data]))
    D = len(cal)
    dix = {d: i for i, d in enumerate(cal)}
    ks = [0] + list(range(GUARD, D - GUARD + 1))
    cands = cands_for("5m")
    RES = np.full((len(data), len(BOARDS), len(ks), len(cands), 2, 2), np.nan)
    for si, R in enumerate(data):
        n = len(R["O"])
        r = np.r_[R["O"][1:], R["C"][-1]] / R["O"] - 1.0
        day_i = np.array([dix[d] for d in R["day"]])
        grid = np.full((D, int(R["slot"].max()) + 1), -1, np.int64)
        grid[day_i, R["slot"]] = np.arange(n)
        comps = []
        for nm, awk, c in BOARDS:
            p = R["pos"][awk]
            comps.append((p * r - c * np.abs(np.diff(np.r_[0.0, p])), p, r))
        Fz = np.zeros((len(ks), len(cands), n), np.bool_)
        for ki, k in enumerate(ks):
            if k == 0:
                Fk = R["F"]
            else:
                src, _ = shifted_src(day_i, R["slot"], D, k, grid)
                Fk = np.where(src[None, :] >= 0, R["F"][:, np.maximum(src, 0)], False)
            Fz[ki, :, 1:] = Fk[:, :-1]
        wm = (R["seg"] == 0) & (np.array(R["day"]) >= START)
        res = bar_sets(Fz.reshape(-1, n), np.r_[False, np.ones(n - 1, np.bool_)], (wm,), comps)
        RES[si] = res[:, 0].reshape(len(BOARDS), len(ks), len(cands), 2, 2)
    rows = []
    for bi, (bn, _, _) in enumerate(BOARDS):
        S = stock_stats(RES[:, bi])
        q95, p = null_cmp(S["diff"][0], S["diff"][1:])
        for ci, c in enumerate(cands):
            rows.append(dict(board=bn, cand=c, trend_pb=S["trend"][0, ci], amb_pb=S["amb"][0, ci], diff_pb=S["diff"][0, ci],
                             diff_null_q95=q95[ci], diff_p=p[ci]))
    df = pd.DataFrame(rows)
    df["crit1"] = (df.diff_pb > df.diff_null_q95) & (df.amb_pb < 0) & (df.trend_pb > 0)
    df.to_csv(OUT / "z3diag_warm_sel.csv", index=False, encoding="utf-8-sig")
    full = pd.read_csv(OUT / "z3diag_bars_5m.csv")
    full = full[full.win == "sel"][["board", "cand", "diff_pb", "diff_p"]].rename(columns={"diff_pb": "diff_full", "diff_p": "p_full"})
    m = df.merge(full, on=["board", "cand"])
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print(m[m.board.isin(["real01", "ideal", "real0"])].round(5).to_string(index=False))
    print("diff > q95 개수(7/23~):", df.assign(ps=df.diff_pb > df.diff_null_q95).groupby("board").ps.sum().to_dict())
