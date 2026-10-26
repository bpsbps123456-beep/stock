"""타점 많은 설정이 2026 전체·2025에서도 버티는가: 60분 연장(yfinance 1시간봉 그대로) 짧은 평활, 240분 평활 5일 vs 10.5일."""
from datetime import date

import numpy as np
import pandas as pd

from core import *
from evalcfg import data, WINDOWS
from grid_short import portfolio, window_metrics
from run_v2 import exec_masks
from run_v3 import ctx
from sim2 import simulate_v2

D, HALF = data()
WINDOWS["y2025"] = dict(base_key="1h", eval_start=date(2025, 1, 2), split=date(2025, 7, 1))


def eth_native(s):
    """yfinance 1시간봉(연장 포함)을 그대로 분봉으로 쓴다. 끝 시각 = 다음 봉 시작(같은 날) 또는 +60분."""
    b = D[(s, "1h")]
    st = b.index.as_unit("ns").asi8
    nxt = np.r_[st[1:], st[-1] + 3600 * 10 ** 9]
    same_day = np.r_[np.asarray(b.index.date[1:]) == np.asarray(b.index.date[:-1]), False]
    end = np.where(same_day, np.minimum(nxt, st + 3600 * 10 ** 9), st + 3600 * 10 ** 9)
    return pd.DataFrame({"O": b.Open.values, "H": b.High.values, "L": b.Low.values, "C": b.Close.values, "end_ns": end})


def run(win, kind, hk, n_days=25):
    S, common, sel = ctx(win)
    weeks = len(common) / 5
    eqs, trades, flips = [], 0, 0
    for s in SYMS:
        A = S[s]
        if kind == "60연장":
            tb, d = eth_native(s), 17
        else:
            tf = int(kind[:-2])
            tb, d = tf_bars(D[(s, "1h")], tf, False, HALF), int(np.ceil(390 / tf))
        L = max(3, int(round(hk * d)))
        K = kalman(tb.C.values, gain_to_q(length_to_gain(L)), 3.0)
        u = rms(tb.C.values - K, max(10, n_days * d))
        z = zones(tb.C.values, K, u, 0.75, 1.75)
        zw = z[(tb.end_ns.values >= A["start_ns"][A["j0"]]) & (z != -9)]
        flips += int((zw[1:] != zw[:-1]).sum())
        ex, aw = exec_masks(S, s)["M1"]
        nn = len(tb)
        eq, t, *_ = simulate_v2(A["o"], A["h"], A["l"], A["c"], ex, aw, A["is_end"], A["j0"],
                                map_completed(A["start_ns"], tb.end_ns.values), z, K + 1.75 * u, K - 1.75 * u,
                                True, True, 0.0005, np.full(len(A["o"]), -1, np.int64), np.zeros(nn, bool),
                                np.full(nn, 50.0), np.full(nn, np.nan), 0, 0.0, 0.01, 6, 0, 0, ex, 0.0)
        eqs.append(eq); trades += t
    return {"win": win, "봉": kind, "평활(일)": hk, "L": L, "타점/주/종목": flips / weeks / 7, "체결/주/종목": trades / weeks / 7,
            **window_metrics(portfolio(eqs, S), sel)}


if __name__ == "__main__":
    rows = []
    for win in ("long", "y2025"):
        S, common, sel = ctx(win)
        bh = portfolio([buy_hold(S[s]["o"], S[s]["c"], S[s]["is_reg"], S[s]["is_end"], S[s]["j0"]) for s in SYMS], S)
        rows.append({"win": win, "봉": "보유", **window_metrics(bh, sel)})
        for hk in (0.5, 1, 2, 3, 5, 10.5):
            rows.append(run(win, "60연장", hk))
        for hk in (3, 5, 10.5):
            rows.append(run(win, "240정규", hk))
    t = pd.DataFrame(rows)
    t.to_parquet("run_v5.parquet")
    F = lambda x: f"{x:+.1%}" if pd.notna(x) else "-"
    for win, g in t.groupby("win", sort=False):
        print(f"\n===== {'2026년 (선택 1~5월 / 확인 6~9월)' if win == 'long' else '2025년 (상반기 / 하반기)'} =====")
        print(g[["봉", "평활(일)", "L", "타점/주/종목", "체결/주/종목", "all_ret", "all_sh", "all_mdd", "sel_ret", "val_ret"]]
              .to_string(index=False, formatters={c: F for c in ["all_ret", "all_mdd", "sel_ret", "val_ret"]} |
                         {"타점/주/종목": lambda v: f"{v:.1f}" if pd.notna(v) else "-", "체결/주/종목": lambda v: f"{v:.1f}" if pd.notna(v) else "-",
                          "all_sh": "{:.2f}".format, "평활(일)": lambda v: f"{v:g}" if pd.notna(v) else "-", "L": lambda v: f"{v:.0f}" if pd.notna(v) else "-"}))
