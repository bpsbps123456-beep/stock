"""robust 갈래 보강: 원안의 편도 비용 민감도(0·0.03·0.05·0.1·0.2%), 이상적/실제 판, 144종목, 귀무 k=15~54.
실제 판 손해의 대부분이 비용으로 나와(robust_realloss), 스프레드가 작을 때 남는지 본다.
py code/robust_cost.py → results/robust_cost.csv
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from robust_common import ORIG, RES, SYMS, bars_for, null_shifts, pos_ret, ratio, window_masks, wmetrics

COSTS = [0.0, 0.0003, 0.0005, 0.001, 0.002]


def work(args):
    sym, ks = args
    B = bars_for(sym, "5m", ORIG["tf"], "eth", 0)
    p0 = ratio(B["O"], B["C"], B["V"], ORIG["N"], ORIG["kind"])
    masks = window_masks(B["seg"])
    arr = np.full((len(ks), 2, len(COSTS), 3, 8), np.nan)
    m = np.zeros(8)
    for ki, k in enumerate(ks):
        p = np.roll(p0, k * B["bpd"]) if k else p0
        for mi, aw in enumerate((B["ones"], B["awake"])):
            for ci, c in enumerate(COSTS):
                pos, r, rb = pos_ret(B["O"], B["C"], p, aw, ORIG["b"], ORIG["s"], True, c)
                for wi, wm in enumerate(masks):
                    wmetrics(r, rb, pos, wm, m); arr[ki, mi, ci, wi] = m
    return sym, arr


if __name__ == "__main__":
    D, sh = null_shifts("5m"); ks = [0] + sh
    A = np.full((len(SYMS), len(ks), 2, len(COSTS), 3, 8), np.nan)
    with Pool(3) as pool:
        for sym, a in pool.imap_unordered(work, [(s, ks) for s in SYMS]):
            A[SYMS.index(sym)] = a
    rows = []
    for mi, mode in enumerate(("ideal", "real")):
        for ci, c in enumerate(COSTS):
            for wi, wn in enumerate(("full", "sel", "conf")):
                a = A[:, :, mi, ci, wi]
                ret = np.expm1(a[..., 0]) * 100; st = a[..., 5] * 100
                ds = ret - st; dd = np.expm1(a[..., 1]) * 100 - a[..., 6] * 100
                med = np.median(ds, axis=0)
                r = dict(mode=mode, cost_pct=c * 100, window=wn, ret_med=np.median(ret[:, 0]), static_med=np.median(st[:, 0]),
                         d_static_med=med[0], d_static_pos=(ds[:, 0] > 0).mean(), d_static_null_med=np.median(med[1:]),
                         d_static_null_pct=(med[1:] < med[0]).mean(), ddred_med=np.median(dd[:, 0]), ddred_pos=(dd[:, 0] > 0).mean())
                for s in ("MU", "BE", "NBIS"):
                    r[f"{s}_ret"] = ret[SYMS.index(s), 0]; r[f"{s}_static"] = st[SYMS.index(s), 0]
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "robust_cost.csv", index=False)
    pd.set_option("display.width", 250)
    print(df[df.window != "sel"].round(2).to_string(index=False))
