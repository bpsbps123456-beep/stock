"""격자 실행: py code/run_grid.py MU 5m  → results/grid_MU_5m.parquet
비용은 0과 민감도(정규장 0.05%, 연장 0.1% 편도) 두 가지를 함께 잰다."""
import sys
import time

import numpy as np
import pandas as pd

from engine import *

COST_SENS = {"rth": 0.0005, "eth": 0.001}


def prepare(sym, base, sess, tf):
    d = load(sym, base)
    tb = tf_bars(d, tf, sess)
    tb = tb[tb.day >= WIN[base]["warm"]].reset_index(drop=True)
    return tb


def grid(sym, base, costs_fn=lambda sess: [0.0, COST_SENS[sess]], tfs=None, shift_days=0):
    tfs = tfs or (TF_5M if base == "5m" else TF_1H)
    d = load(sym, base)
    win = WIN[base]
    bs = np.array(BS); ss = np.array(SS)
    frames = []
    for sess in ("eth", "rth"):
        costs = np.array(costs_fn(sess))
        cfg = pd.DataFrame(config_rows(costs), columns=["cost", "dir", "dayflat", "b", "s"])
        for tf in tfs:
            tb = tf_bars(d, tf, sess)
            tb = tb[tb.day >= win["warm"]].reset_index(drop=True)
            if len(tb) < 40:
                continue
            O, C, V = tb.O.values.astype(np.float64), tb.C.values.astype(np.float64), tb.V.values.astype(np.float64)
            lod = tb.lod.values
            seg = segments(tb.day.values, win)
            awake = np.ones(len(tb), np.bool_)
            bh_sel, bh_conf = bh(O, C, seg)
            bpd = int(pd.Series(tb.day.values).value_counts().median())
            for n in NS:
                for kind in RATIOS:
                    p = ratio(O, C, V, n, kind)
                    if shift_days:
                        p = np.roll(p, shift_days * bpd)
                    res = np.zeros((len(cfg), 11))
                    grid_kernel(O, C, p, lod, seg, awake, bs, ss, costs, res)
                    f = cfg.copy()
                    f[COLS] = res
                    f["tf"] = tf; f["sess"] = sess; f["N"] = n; f["ratio"] = kind
                    f["bh_sel"] = bh_sel; f["bh_conf"] = bh_conf
                    frames.append(f)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    sym, base = sys.argv[1], sys.argv[2]
    t0 = time.time()
    g = grid(sym, base)
    for c in ["N", "tf", "ent_sel", "ent_conf", "hold_sel", "hold_conf", "bars_sel", "bars_conf"]:
        g[c] = g[c].astype(np.int32)
    g.to_parquet(HERE / "results" / f"grid_{sym}_{base}.parquet")
    print(sym, base, len(g), "칸", round(time.time() - t0, 1), "초")
