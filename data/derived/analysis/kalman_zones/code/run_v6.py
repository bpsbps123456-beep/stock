"""타점 많은 설정(60분 연장·짧은 평활) 검증: 비용, 지정가 체결 제한(정규장만), 프리장 체결 금지, 이웃 값."""
import numpy as np
import pandas as pd

from core import *
from evalcfg import data
from grid_short import portfolio, window_metrics
from run_v2 import exec_masks
from run_v3 import ctx
from run_v5 import eth_native
from sim2 import simulate_v2

D, HALF = data()


def run_cfg(win, kind, hk, a=0.75, b=1.75, mcost=0.0005, lcost=0.0, lim="exec", mode="M1"):
    S, common, sel = ctx(win)
    weeks = len(common) / 5
    eqs, trades = [], 0
    for s in SYMS:
        A = S[s]
        if kind == "60연장":
            tb, d = eth_native(s), 17
        else:
            tb, d = tf_bars(D[(s, "1h")], 240, False, HALF), 2
        L = max(3, int(round(hk * d)))
        K = kalman(tb.C.values, gain_to_q(length_to_gain(L)), 3.0)
        u = rms(tb.C.values - K, max(10, 25 * d))
        z = zones(tb.C.values, K, u, a, b)
        ex, aw = exec_masks(S, s)[mode]
        lim_ok = A["is_reg"] if lim == "reg" else ex
        nn = len(tb)
        eq, t, *_ = simulate_v2(A["o"], A["h"], A["l"], A["c"], ex, aw, A["is_end"], A["j0"],
                                map_completed(A["start_ns"], tb.end_ns.values), z, K + b * u, K - b * u,
                                True, True, mcost, np.full(len(A["o"]), -1, np.int64), np.zeros(nn, bool),
                                np.full(nn, 50.0), np.full(nn, np.nan), 0, 0.0, 0.01, 6, 0, 0, lim_ok, lcost)
        eqs.append(eq); trades += t
    m = window_metrics(portfolio(eqs, S), sel)
    return m["all_ret"], m["all_sh"], m["all_mdd"], trades / weeks / 7


SCEN = {"기본(시장가0.05%·지정가0)": {}, "비용 0.1%/0.05%": dict(mcost=0.001, lcost=0.0005),
        "비용 0.2%/0.1%": dict(mcost=0.002, lcost=0.001), "비용 0.3%/0.2%": dict(mcost=0.003, lcost=0.002),
        "지정가는 정규장에서만": dict(lim="reg"), "프리장 체결 없음(M0)": dict(mode="M0")}
F = lambda x: f"{x:+.0%}"
rows = []
for kind, hk in [("60연장", 0.25), ("60연장", 0.5), ("60연장", 0.75), ("60연장", 1), ("60연장", 1.5), ("240정규", 10.5)]:
    for sc, kw in SCEN.items():
        r26 = run_cfg("long", kind, hk, **kw)
        r25 = run_cfg("y2025", kind, hk, **kw)
        rows.append({"설정": f"{kind} 평활{hk:g}일", "조건": sc, "2026": F(r26[0]), "샤프26": f"{r26[1]:.2f}", "낙폭26": F(r26[2]),
                     "2025": F(r25[0]), "샤프25": f"{r25[1]:.2f}", "낙폭25": F(r25[2]), "체결/주/종목": f"{r26[3]:.1f}"})
t = pd.DataFrame(rows)
print("보유: 2026 +276% (샤프 2.96, 낙폭 -39%) / 2025 +948% (샤프 2.45, 낙폭 -44%)")
for k, g in t.groupby("설정", sort=False):
    print(f"\n[{k}]")
    print(g.drop(columns="설정").to_string(index=False))
print("\n[60연장 평활 0.5일 — a·b 이웃 (기본 비용)]")
for a, b in [(0.5, 1.5), (0.75, 1.5), (0.75, 1.75), (0.75, 2.0), (1.0, 2.0), (1.0, 1.75)]:
    r26, r25 = run_cfg("long", "60연장", 0.5, a, b), run_cfg("y2025", "60연장", 0.5, a, b)
    r26c, r25c = run_cfg("long", "60연장", 0.5, a, b, mcost=0.002, lcost=0.001), run_cfg("y2025", "60연장", 0.5, a, b, mcost=0.002, lcost=0.001)
    print(f"  a{a} b{b}: 2026 {F(r26[0])} (비용0.2%/0.1%면 {F(r26c[0])}) | 2025 {F(r25[0])} (비용 올리면 {F(r25c[0])})")
