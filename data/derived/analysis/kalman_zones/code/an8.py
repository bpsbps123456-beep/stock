"""추세 정의 3종(미리 정함)으로 '하락 추세일 때만 손절'·'상승 중 눌림 매수'를 시험."""
import numpy as np, pandas as pd
from core import *
from evalcfg import data, prep, tf_source, indicator
from grid_short import portfolio, window_metrics
F = lambda x: f"{x:+.1%}"
P = dict(tf=240, eth=False, L=21, norm="rms", n=50, a=0.75, b=1.75)

def trends(tb, K, L):
    c = tb.C.values
    out = {}
    for m in (5, 10):
        t = np.ones(len(K), np.int8); t[m:] = np.where(K[m:] > K[:-m], 1, -1); out[f"T1 칼만선 {m}봉 기울기"] = t
    line, col = ratchet(K, atr(tb.H.values, tb.L.values, c, 14), 1.0)
    col = col.copy(); col[col == 0] = 1; out["T2 계단선 색(ATR14x1.0)"] = col
    Kslow = kalman(c, gain_to_q(length_to_gain(3 * L)), 3.0)
    out["T3 가격 vs 3배 느린 칼만선"] = np.where(c > Kslow, 1, -1).astype(np.int8)
    return out

def run(win, tname, stop_mode, dip_buy):
    D, H = data(); S, common, sel = prep(win); eqs = []
    for s in SYMS:
        tb = tf_bars(D[(s, tf_source(win, P["tf"], P["eth"]))], P["tf"], P["eth"], H)
        idx = map_completed(S[s]["start_ns"], tb.end_ns.values)
        K, u, z = indicator(tb, P["L"], P["norm"], P["n"], P["a"], P["b"])
        tr = trends(tb, K, P["L"])[tname]
        eq, *_ = simulate2(S[s]["o"], S[s]["h"], S[s]["l"], S[s]["c"], S[s]["is_reg"], S[s]["is_awake"], S[s]["is_end"],
                           idx, z, tr, K + P["b"] * u, K - P["b"] * u, S[s]["j0"], True, True, 0.0005, stop_mode, 0, dip_buy, False)
        eqs.append(eq)
    return window_metrics(portfolio(eqs, S), sel)

D, H = data()
names = list(trends(tf_bars(D[("AMD", "1h")], 240, False, H).iloc[:200], np.linspace(1, 2, 200), 21).keys())
rows = []
for tname in names:
    for label, sm, db in [("하락일 때만 손절", 2, 0), ("+상승 중 눌림매수", 2, 1)]:
        ml, ms = run("long", tname, sm, db), run("short", tname, sm, db)
        rows.append({"trend": tname, "rule": label, "L_sel": ml["sel_ret"], "L_selsh": ml["sel_sh"], "L_val": ml["val_ret"],
                     "L_valmdd": ml["val_mdd"], "L_all": ml["all_ret"], "L_allmdd": ml["all_mdd"], "S_all": ms["all_ret"], "S_mdd": ms["all_mdd"]})
t = pd.DataFrame(rows)
print("기준 V0(매도=손절): 2026 선택 +149.4% 샤프 7.00 | 확인 +33.4% 낙폭 -17.8% | 전체 +232.6% 낙폭 -17.8% | 최근60일 +31.4% 낙폭 -15.9%")
print(t.to_string(index=False, formatters={c: F for c in t.columns if c not in ("trend", "rule", "L_selsh")} | {"L_selsh": "{:.2f}".format}))
