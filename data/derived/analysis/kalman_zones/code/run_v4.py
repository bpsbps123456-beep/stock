"""타점 수 ↔ 성적: 칼만 평활 기간(거래일)을 줄여 신호를 늘리면 성적이 어떻게 되는가. 현재 지표 규칙, 체결 M1(프리장 포함)."""
import numpy as np
import pandas as pd

from core import *
from evalcfg import data, tf_source
from grid_short import portfolio, window_metrics
from run_v2 import exec_masks
from run_v3 import ctx
from datetime import date
from evalcfg import WINDOWS
WINDOWS['short2'] = dict(base_key='5m', eval_start=date(2026, 8, 7), split=date(2026, 9, 1))
from sim2 import simulate_v2

D, HALF = data()
HK = [0.5, 1, 2, 3, 5, 10.5]


def bpd(tf, eth):
    return int(np.ceil((960 if eth else 390) / tf))


def run(win, tf, eth, preopen=False):
    S, common, sel = ctx(win)
    d = bpd(tf, eth)
    weeks = len(common) / 5
    rows = []
    for hk in HK:
        L = max(3, int(round(hk * d)))
        g = length_to_gain(L)
        n = max(10, 25 * d)  # 폭 창 25거래일 (현재 지표 240분 RMS 50봉과 같음)
        eqs, trades, flips = [], 0, 0
        for s in SYMS:
            A = S[s]
            tb = tf_bars(D[(s, tf_source("long" if win == "long" else "short", tf, eth))], tf, eth, HALF)
            K = kalman(tb.C.values, gain_to_q(g), 3.0)
            u = rms(tb.C.values - K, n)
            z = zones(tb.C.values, K, u, 0.75, 1.75)
            t0 = A["start_ns"][A["j0"]]
            zw = z[(tb.end_ns.values >= t0) & (z != -9)]
            flips += int((zw[1:] != zw[:-1]).sum())
            ex, aw = exec_masks(S, s)["M1"]
            if preopen:
                aw = aw & ~A["is_reg"]
            bidx = map_completed(A["start_ns"], tb.end_ns.values)
            nn = len(tb)
            eq, t, *_ = simulate_v2(A["o"], A["h"], A["l"], A["c"], ex, aw, A["is_end"], A["j0"], bidx, z, K + 1.75 * u,
                                    K - 1.75 * u, True, True, 0.0005, np.full(len(A["o"]), -1, np.int64),
                                    np.zeros(nn, bool), np.full(nn, 50.0), np.full(nn, np.nan), 0, 0.0, 0.01, 6, 0, 0, ex, 0.0)
            eqs.append(eq); trades += t
        m = window_metrics(portfolio(eqs, S), sel)
        rows.append({"win": win, "tf": f"{tf}분 {'연장' if eth else '정규'}{' 개장전판단' if preopen else ''}", "평활(일)": hk, "L": L,
                     "타점/주/종목": flips / weeks / 7, "체결/주/종목": trades / weeks / 7, **m})
    return rows


if __name__ == "__main__":
    rows = []
    rows += run("long", 60, False) + run("long", 240, False)
    rows += run("short2", 40, False) + run("short2", 40, False, preopen=True) + run("short2", 40, True) + run("short2", 240, False)
    t = pd.DataFrame(rows)
    t.to_parquet("run_v4.parquet")
    F = lambda x: f"{x:+.1%}"
    for (win, tf), g in t.groupby(["win", "tf"], sort=False):
        print(f"\n[{'2026년 1~9월 (선택 1~5월/확인 6~9월)' if win == 'long' else '최근 (8/7~9/25, 40분 폭 25일 예열 뒤)'}] {tf}")
        print(g[["평활(일)", "L", "타점/주/종목", "체결/주/종목", "all_ret", "all_sh", "all_mdd", "sel_ret", "val_ret"]]
              .to_string(index=False, formatters={c: F for c in ["all_ret", "all_mdd", "sel_ret", "val_ret"]} |
                         {"타점/주/종목": "{:.1f}".format, "체결/주/종목": "{:.1f}".format, "all_sh": "{:.2f}".format}))
