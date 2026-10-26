"""초단타 1,600가지: 매수 자리 10 x 필터 4 x 목표 4 x 보유 2 x 프리장 매수 2 x 프리장 정리 2 (수수료 0%). 창 3개, 느슨/엄격 체결."""
import itertools
import numpy as np
import pandas as pd
from core import *
from grid_short import portfolio, window_metrics
from run_on import ctx, day_arrays, levels, LEVELS, FILTERS
from sim_on2 import sim_overnight2

TARGETS = [0.005, 0.01, 0.02, 0.03]


def run(win, margin):
    S, common, sel = ctx(win)
    for s in SYMS:
        day_arrays(S, s)
    LV = {s: levels(S, s, win) for s in SYMS}
    rows = []
    for lv, fi, tg, hd, pe, xp in itertools.product(LEVELS, FILTERS, TARGETS, [1, 2], [False, True], [False, True]):
        eqs, ntr, nw, sr = [], 0, 0, 0.0
        for s in SYMS:
            A = S[s]
            e, t, w, r = sim_overnight2(A["o"], A["h"], A["l"], A["c"], A["is_reg"], A["is_pre"], A["is_awake"], A["is_end"],
                                        A["first_reg"], A["day_idx"], A["j0"], LV[s][(lv, fi)], tg, hd, 0.0, margin, pe, xp)
            eqs.append(e); ntr += t; nw += w; sr += r
        rows.append({"win": win, "entry": lv, "filter": fi, "tgt": tg, "hold": hd, "pre_buy": pe, "pre_exit": xp,
                     "trades_per_day": ntr / len(common), "win_rate": nw / ntr if ntr else np.nan,
                     "avg_trade": sr / ntr if ntr else np.nan, **window_metrics(portfolio(eqs, S), sel)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for margin, tag in ((0.0, "loose"), (0.001, "strict")):
        out = pd.concat([run(w, margin) for w in ("long", "y2025", "short")], ignore_index=True)
        out.to_parquet(f"run_on2_{tag}.parquet")
        print(tag, out.groupby("win").size().to_dict())
