"""추세 지표(현재 240분) + 초단타(종가-2% 매수, +1% 매도, 최대 2세션) 자금 나누기. 체결 M1, 기본/비용 인상."""
import numpy as np
import pandas as pd
from core import *
from grid_short import portfolio, window_metrics
from run_on import ctx, day_arrays, levels
from run_v2 import exec_masks
from sim2 import simulate_v2
from sim_on import sim_overnight
from evalcfg import data, indicator

D, HALF = data()


def curves(win, mcost, lcost, margin):
    S, common, sel = ctx(win)
    for s in SYMS:
        day_arrays(S, s)
    tr, sc = [], []
    for s in SYMS:
        A = S[s]
        tb = tf_bars(D[(s, "1h")], 240, False, HALF)
        K, u, z = indicator(tb, 21, "rms", 50, 0.75, 1.75)
        ex, aw = exec_masks(S, s)["M1"]
        nn = len(tb)
        e1, *_ = simulate_v2(A["o"], A["h"], A["l"], A["c"], ex, aw, A["is_end"], A["j0"], map_completed(A["start_ns"], tb.end_ns.values),
                             z, K + 1.75 * u, K - 1.75 * u, True, True, mcost, np.full(len(A["o"]), -1, np.int64), np.zeros(nn, bool),
                             np.full(nn, 50.0), np.full(nn, np.nan), 0, 0.0, 0.01, 6, 0, 0, ex, lcost)
        lv = levels(S, s, win)[("종가-2%", "없음")]
        e2, *_ = sim_overnight(A["o"], A["h"], A["l"], A["c"], A["is_reg"], A["is_pre"], A["is_awake"], A["is_end"], A["first_reg"],
                               A["day_idx"], A["j0"], lv, 0.01, 2, mcost, lcost, margin, np.full(len(A["days"]), np.nan), 0.005, 0.03)
        tr.append(e1); sc.append(e2)
    return portfolio(tr, S), portfolio(sc, S), sel


F = lambda x: f"{x:+.0%}"
for tag, kw in (("기본", dict(mcost=0.001, lcost=0.0, margin=0.0)), ("비용 인상", dict(mcost=0.002, lcost=0.001, margin=0.001))):
    print(f"\n[{tag}]  수익 (낙폭)")
    for win, lab in (("long", "2026"), ("y2025", "2025"), ("short", "최근60일")):
        t, s, sel = curves(win, **kw)
        line = []
        for w in (1.0, 0.7, 0.5, 0.0):
            m = window_metrics(w * t + (1 - w) * s, sel)
            line.append(f"추세{int(w*100)}%/초단타{int((1-w)*100)}%: {F(m['all_ret'])} ({F(m['all_mdd'])}, 샤프 {m['all_sh']:.2f})")
        print(f"  {lab}: " + " | ".join(line))
