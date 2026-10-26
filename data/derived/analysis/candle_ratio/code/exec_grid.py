"""exec 갈래: 사용자가 깨어 있는 시간(한국시간 08~24시)에만 체결하는 판으로 격자를 다시 돈다.

run_grid.grid를 그대로 복사하고 awake만 engine.awake_mask(tb.start)로 바꿨다.
자는 동안 나온 신호는 다음 깨어 있는 봉 시가에 체결된다(engine.sim의 awake 인자).
정규장 dayflat(16:00 ET = 05:00 KST 종가 매도)은 자는 시간이라 feasible=False로 표시한다.
연장 dayflat(20:00 ET = 09:00 KST 마감)은 실행 가능으로 둔다.

사용: py code/exec_grid.py MU 5m   → results/exec_grid_MU_5m_awake.parquet
"""
import sys
import time

import numpy as np
import pandas as pd

from engine import *
from run_grid import COST_SENS


def grid_awake(sym, base, costs_fn=lambda sess: [0.0, COST_SENS[sess]], tfs=None, shift_days=0):
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
            awake = awake_mask(tb.start)  # 유일한 변경점
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
    g = pd.concat(frames, ignore_index=True)
    g["feasible"] = ~((g.sess == "rth") & g.dayflat)
    return g


if __name__ == "__main__":
    sym, base = sys.argv[1], sys.argv[2]
    t0 = time.time()
    g = grid_awake(sym, base)
    for c in ["N", "tf", "ent_sel", "ent_conf", "hold_sel", "hold_conf", "bars_sel", "bars_conf"]:
        g[c] = g[c].astype(np.int32)
    g.to_parquet(HERE / "results" / f"exec_grid_{sym}_{base}_awake.parquet")
    print(sym, base, "awake", len(g), "행", int(g.feasible.sum()), "실행 가능", round(time.time() - t0, 1), "초")
