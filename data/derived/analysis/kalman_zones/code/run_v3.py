"""뼈대(현재 지표) vs 칼만 단계 매매 4가지(미리 정함). 2026년(선택 1~5월·확인 6~9월), 2025년, 최근 60일. 체결 M1."""
from datetime import date

import numpy as np
import pandas as pd

from core import *
from evalcfg import data, tf_source, indicator, WINDOWS, _P
from grid_short import prepare, portfolio, window_metrics
from osc import rsi_state
from run_v2 import exec_masks
from sim2 import simulate_v2
from sim3 import simulate_ladder

D, HALF = data()
WINDOWS["y2025"] = dict(base_key="1h", eval_start=date(2025, 1, 2), split=date(2025, 7, 1))
LADDERS = {  # keep_tp, buy_to, cut_to, recover_to, rsi_to
    "L1 단계(매수 2/3)": (1 / 3, 2 / 3, 1 / 3, 1 / 3, 0.0),
    "L2 단계(매수 전량)": (1 / 3, 1.0, 1 / 3, 1 / 3, 0.0),
    "L3 L1 + RSI 과매도 1/3": (1 / 3, 2 / 3, 1 / 3, 1 / 3, 1 / 3),
    "L4 L2 + RSI 과매도 1/3": (1 / 3, 1.0, 1 / 3, 1 / 3, 1 / 3),
}


def ctx(win):
    if win not in _P:
        _P[win] = prepare(D, HALF, **WINDOWS[win])
    return _P[win]


def run(win, sw_tf, mode="M1"):
    S, common, sel = ctx(win)
    rows = []
    bh = portfolio([buy_hold(S[s]["o"], S[s]["c"], S[s]["is_reg"], S[s]["is_end"], S[s]["j0"]) for s in SYMS], S)
    rows.append({"win": win, "rule": "보유", **window_metrics(bh, sel), "trades": 0})
    L = {}
    for s in SYMS:
        tb = tf_bars(D[(s, "1h")], 240, False, HALF)
        K, u, z = indicator(tb, 21, "rms", 50, 0.75, 1.75)
        sb = tf_bars(D[(s, tf_source("short" if win == "short" else "long", sw_tf, False))], sw_tf, False, HALF)
        r, _, _ = rsi_state(sb.C.values, 14)
        cond = np.nan_to_num((r < 30).astype(float)).astype(bool)
        L[s] = dict(bidx=map_completed(S[s]["start_ns"], tb.end_ns.values), z=z, tp=K + 1.75 * u, pb=K - 1.75 * u,
                    sidx=map_completed(S[s]["start_ns"], sb.end_ns.values), sig=cond & ~np.r_[False, cond[:-1]],
                    rsi=np.nan_to_num(r, nan=50.0))
    # 뼈대(현재 지표 규칙)
    eqs, nt = [], 0
    for s in SYMS:
        A, Y = S[s], L[s]
        ex, aw = exec_masks(S, s)[mode]
        eq, t, *_ = simulate_v2(A["o"], A["h"], A["l"], A["c"], ex, aw, A["is_end"], A["j0"], Y["bidx"], Y["z"], Y["tp"],
                                Y["pb"], True, True, 0.0005, Y["sidx"], np.zeros(len(Y["sig"]), bool), Y["rsi"],
                                np.full(len(Y["sig"]), np.nan), 0, 0.0, 0.01, 6, 0, 0, ex, 0.0)
        eqs.append(eq); nt += t
    rows.append({"win": win, "rule": "현재 지표(뼈대)", **window_metrics(portfolio(eqs, S), sel), "trades": nt})
    for name, (kt, bt, ct, rt, rs) in LADDERS.items():
        eqs, nt = [], 0
        for s in SYMS:
            A, Y = S[s], L[s]
            ex, aw = exec_masks(S, s)[mode]
            eq, t = simulate_ladder(A["o"], A["h"], A["l"], A["c"], ex, aw, A["is_end"], A["j0"], Y["bidx"], Y["z"],
                                    Y["tp"], Y["pb"], kt, bt, ct, rt, Y["sidx"], Y["sig"], rs, 0.0005)
            eqs.append(eq); nt += t
        rows.append({"win": win, "rule": name, **window_metrics(portfolio(eqs, S), sel), "trades": nt})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    F = lambda x: f"{x:+.1%}"
    out = pd.concat([run("long", 60), run("y2025", 60), run("short", 40)], ignore_index=True)
    out.to_parquet("run_v3.parquet")
    label = {"long": "2026년 (선택 1~5월 / 확인 6~9월)", "y2025": "2025년 (상반기 / 하반기)", "short": "최근 60일 (7/16~8/19 / 8/20~9/25)"}
    for win, g in out.groupby("win", sort=False):
        print(f"\n===== {label[win]} =====")
        print(g[["rule", "all_ret", "all_sh", "all_mdd", "sel_ret", "sel_sh", "val_ret", "val_sh", "val_mdd", "trades"]]
              .to_string(index=False, formatters={c: F for c in ["all_ret", "all_mdd", "sel_ret", "val_ret", "val_mdd"]} |
                         {c: "{:.2f}".format for c in ["all_sh", "sel_sh", "val_sh"]}))
