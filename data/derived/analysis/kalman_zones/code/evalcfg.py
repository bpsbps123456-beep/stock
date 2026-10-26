"""설정 하나를 두 창(2026년 1시간봉 기반, 최근 60일 5분봉 기반)에서 평가하고 종목·체결 기록까지 돌려준다."""
import numpy as np
import pandas as pd

from core import *
from grid_short import prepare, portfolio, window_metrics

WINDOWS = {
    "long": dict(base_key="1h", eval_start=pd.Timestamp("2026-01-02").date(), split=pd.Timestamp("2026-06-01").date()),
    "short": dict(base_key="5m", eval_start=pd.Timestamp("2026-07-16").date(), split=pd.Timestamp("2026-08-20").date()),
}
_D, _H, _P = None, None, {}


def data():
    global _D, _H
    if _D is None:
        _D = load()
        _H = set()
        for s in SYMS:
            _H |= half_days(_D[(s, "1h")])
    return _D, _H


def prep(win):
    if win not in _P:
        D, H = data()
        _P[win] = prepare(D, H, **WINDOWS[win])
    return _P[win]


def tf_source(win, tf, eth):
    if win == "long":
        return "1h"
    return "1h" if (not eth and tf % 60 == 0) else "5m"


def indicator(tb, L, norm, n, a, b):
    g = length_to_gain(L)
    K = kalman(tb.C.values, gain_to_q(g), 3.0)
    if norm == "atr":
        u = atr(tb.H.values, tb.L.values, tb.C.values, n) * band_scale(g)
    else:
        u = rms(tb.C.values - K, n)
    return K, u, zones(tb.C.values, K, u, a, b)


def evaluate(win, tf, eth, L=None, norm=None, n=None, a=None, b=None, tp_half=True, panic_hold=True,
             cost=0.0005, original=False, rec=False):
    D, H = data()
    S, common, sel = prep(win)
    eqs, logs, per, ntr, exs, zinfo = [], {}, {}, 0, [], {}
    for s in SYMS:
        tb = tf_bars(D[(s, tf_source(win, tf, eth))], tf, eth, H)
        idx = map_completed(S[s]["start_ns"], tb.end_ns.values)
        if original:
            K = kalman(tb.C.values, 0.01, 3.0)
            _, trd = ratchet(K, atr(tb.H.values, tb.L.values, tb.C.values, 5), 0.5)
            z = trd.copy(); z[z == 0] = -9
            tp = pb = np.full(len(tb), np.nan)
            use_lim = False
        else:
            K, u, z = indicator(tb, L, norm, n, a, b)
            tp, pb = K + b * u, K - b * u
            use_lim = True
        eq, nt, log, ex = simulate(S[s]["o"], S[s]["h"], S[s]["l"], S[s]["c"], S[s]["is_reg"], S[s]["is_awake"],
                                   S[s]["is_end"], idx, z, tp, pb, S[s]["j0"], tp_half, panic_hold, use_lim, cost, rec)
        eqs.append(eq); ntr += nt; exs.append(ex.mean())
        per[s] = eq[-1] - 1
        logs[s] = log
        zinfo[s] = (tb, z, idx)
    port = portfolio(eqs, S)
    out = {"trades": ntr, "expo": float(np.mean(exs)), **window_metrics(port, sel)}
    return out, per, logs, zinfo, port, common, sel
