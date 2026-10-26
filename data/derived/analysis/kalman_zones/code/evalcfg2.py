"""추세 조건 변형 평가 (simulate2)."""
import numpy as np
from core import *
from evalcfg import data, prep, tf_source, indicator
from grid_short import portfolio, window_metrics

VARIANTS = {  # (stop_mode, buy_filter, dip_buy)
    "V0 원래 규칙(매도=손절)": (0, 0, 0),
    "V1 손절 없음": (1, 0, 0),
    "V2 칼만선 하락일 때만 손절": (2, 0, 0),
    "V3 V2 + 매수는 상승 때만": (2, 1, 0),
    "V4 V2 + 상승 중 매도구간 눌림매수": (2, 0, 1),
}

def evaluate2(win, tf, eth, L, norm, n, a, b, stop_mode, buy_filter, dip_buy, tp_half=True, panic_hold=True,
              cost=0.0005, rec=False):
    D, H = data()
    S, common, sel = prep(win)
    eqs, logs, per, ntr, exs, zinfo = [], {}, {}, 0, [], {}
    for s in SYMS:
        tb = tf_bars(D[(s, tf_source(win, tf, eth))], tf, eth, H)
        idx = map_completed(S[s]["start_ns"], tb.end_ns.values)
        K, u, z = indicator(tb, L, norm, n, a, b)
        tr = kalman_trend(K)
        eq, nt, log, ex = simulate2(S[s]["o"], S[s]["h"], S[s]["l"], S[s]["c"], S[s]["is_reg"], S[s]["is_awake"],
                                    S[s]["is_end"], idx, z, tr, K + b * u, K - b * u, S[s]["j0"], tp_half, panic_hold,
                                    cost, stop_mode, buy_filter, dip_buy, rec)
        eqs.append(eq); ntr += nt; exs.append(ex.mean()); per[s] = eq[-1] - 1; logs[s] = log; zinfo[s] = (tb, z, idx, tr)
    port = portfolio(eqs, S)
    return {"trades": ntr, "expo": float(np.mean(exs)), **window_metrics(port, sel)}, per, logs, zinfo, port, common, sel
