"""중간 설정(타점 조금 더)도 같은 스트레스를 견디는가: 240분 정규 평활 3·5·10.5일, 60분 정규 평활 3·5·10.5일."""
import numpy as np
import pandas as pd
from core import *
from evalcfg import data
from grid_short import portfolio, window_metrics
from run_v2 import exec_masks
from run_v3 import ctx
from sim2 import simulate_v2

D, HALF = data()


def run_cfg(win, tf, hk, mcost=0.0005, lcost=0.0, lim="exec", mode="M1"):
    S, common, sel = ctx(win)
    weeks = len(common) / 5
    eqs, trades, flips = [], 0, 0
    d = int(np.ceil(390 / tf))
    for s in SYMS:
        A = S[s]
        tb = tf_bars(D[(s, "1h")], tf, False, HALF)
        L = max(3, int(round(hk * d)))
        K = kalman(tb.C.values, gain_to_q(length_to_gain(L)), 3.0)
        u = rms(tb.C.values - K, max(10, 25 * d))
        z = zones(tb.C.values, K, u, 0.75, 1.75)
        zw = z[(tb.end_ns.values >= A["start_ns"][A["j0"]]) & (z != -9)]
        flips += int((zw[1:] != zw[:-1]).sum())
        ex, aw = exec_masks(S, s)[mode]
        lim_ok = A["is_reg"] if lim == "reg" else ex
        nn = len(tb)
        eq, t, *_ = simulate_v2(A["o"], A["h"], A["l"], A["c"], ex, aw, A["is_end"], A["j0"],
                                map_completed(A["start_ns"], tb.end_ns.values), z, K + 1.75 * u, K - 1.75 * u,
                                True, True, mcost, np.full(len(A["o"]), -1, np.int64), np.zeros(nn, bool),
                                np.full(nn, 50.0), np.full(nn, np.nan), 0, 0.0, 0.01, 6, 0, 0, lim_ok, lcost)
        eqs.append(eq); trades += t
    m = window_metrics(portfolio(eqs, S), sel)
    return m, flips / weeks / 7, trades / weeks / 7


F = lambda x: f"{x:+.0%}"
SC = {"기본": {}, "비용0.2/0.1%": dict(mcost=0.002, lcost=0.001), "지정가 정규장만": dict(lim="reg"), "프리장 없음": dict(mode="M0")}
rows = []
for tf in (240, 60):
    for hk in (3, 5, 10.5):
        row = {"설정": f"{tf}분 정규 평활{hk:g}일"}
        for win, lab in (("long", "26"), ("y2025", "25")):
            for sc, kw in SC.items():
                m, fl, tr = run_cfg(win, tf, hk, **kw)
                row[f"{lab} {sc}"] = F(m["all_ret"])
                if sc == "기본":
                    row[f"{lab} 샤프·낙폭"] = f"{m['all_sh']:.2f}·{m['all_mdd']:.0%}"
                    if lab == "26":
                        row["타점/주"], row["체결/주"] = f"{fl:.1f}", f"{tr:.1f}"
        rows.append(row)
t = pd.DataFrame(rows).set_index("설정")
print("보유: 2026 +276% (샤프 2.96·-39%) / 2025 +948% (샤프 2.45·-44%). 타점·체결은 종목 하나 기준 주당 횟수")
print(t[["타점/주", "체결/주", "26 기본", "26 샤프·낙폭", "26 비용0.2/0.1%", "26 지정가 정규장만", "26 프리장 없음"]].to_string())
print()
print(t[["25 기본", "25 샤프·낙폭", "25 비용0.2/0.1%", "25 지정가 정규장만", "25 프리장 없음"]].to_string())
