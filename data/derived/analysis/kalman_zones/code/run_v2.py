"""2026년 한정: 칼만 240분 뼈대 단독 vs 뼈대 + 과매도 스윙(분할매수→반등 분할매도).

체결 조건 M0 정규장만 / M1 정규장 + 깨어 있는 프리장(사용자 실제) / MF 밤새 대응(참고).
2026년 창: 1시간봉 체결, 스윙 60분 정규장, 선택 1~5월·확인 6~9월.
최근 창: 5분봉 체결, 스윙 40분 정규장·연장, 선택 7/16~8/19·확인 8/20~9/25.
"""
import itertools

import numpy as np
import pandas as pd

from core import *
from evalcfg import data, prep, tf_source, indicator
from grid_short import portfolio, window_metrics
from osc import rsi_state, stoch_rsi, rsi_price
from sim2 import simulate_v2

D, HALF = data()
SW_SIZE, SW_MAX, COST = 1 / 3, 6, 0.0005


def exec_masks(S, s):
    t = pd.to_datetime(S[s]["start_ns"], utc=True).tz_convert("America/New_York")
    mins = np.asarray(t.hour * 60 + t.minute)
    reg, awake = S[s]["is_reg"], S[s]["is_awake"]
    pre = (mins >= 240) & (mins < 570)
    post = (mins >= 960) & (mins < 1200)
    return {"M0": (reg, awake), "M1": (reg | (pre & awake), awake),
            "MF": (reg | pre | post, np.ones_like(awake))}


def layers(win, s, sw_tf, sw_eth):
    S = prep(win)[0]
    tb = tf_bars(D[(s, "1h")], 240, False, HALF)
    K, u, z = indicator(tb, 21, "rms", 50, 0.75, 1.75)
    bidx = map_completed(S[s]["start_ns"], tb.end_ns.values)
    sb = tf_bars(D[(s, tf_source(win, sw_tf, sw_eth))], sw_tf, sw_eth, HALF)
    r, ru, rd = rsi_state(sb.C.values, 14)
    k, _ = stoch_rsi(r, 14, 3, 3)
    sidx = map_completed(S[s]["start_ns"], sb.end_ns.values)
    dip = rsi_price(sb.C.values, ru, rd, 14, 30.0)
    sig = {}
    for name, cond in {"combo": (r < 30) & (k < 30), "rsi": r < 30}.items():
        cond = np.nan_to_num(cond.astype(float)).astype(bool)
        sig[name] = cond & ~np.r_[False, cond[:-1]]
    return dict(bidx=bidx, bz=z, btp=K + 1.75 * u, bpb=K - 1.75 * u, sidx=sidx, srsi=np.nan_to_num(r, nan=50.0),
                sdip=np.nan_to_num(dip, nan=np.nan), sig=sig)


def run(win, sw_tf, sw_eth):
    S, common, sel = prep(win)
    LY = {s: layers(win, s, sw_tf, sw_eth) for s in SYMS}
    MK = {s: exec_masks(S, s) for s in SYMS}
    rows = []
    variants = [("뼈대만", None, 0, 0.0, 0, 0)]
    variants += [(f"{e} 필터{f} 익절{int(tp * 100)}% 밤지정가{d} 떼어두기{r}", e, f, tp, d, r)
                 for e, f, tp, d, r in itertools.product(["combo", "rsi"], [0, 1], [0.01, 0.02], [0, 1], [0, 1])]
    for mode in ("M0", "M1", "MF"):
        for name, e, f, tp, d, r in variants:
            eqs, ntr, nst, nsw, spnl = [], 0, 0, 0, 0.0
            for s in SYMS:
                A, L = S[s], LY[s]
                ex, aw = MK[s][mode]
                sig = L["sig"][e] if e else np.zeros(len(L["srsi"]), bool)
                eq, t, st, sw, sp = simulate_v2(A["o"], A["h"], A["l"], A["c"], ex, aw, A["is_end"], A["j0"],
                                                L["bidx"], L["bz"], L["btp"], L["bpb"], True, True, COST,
                                                L["sidx"], sig, L["srsi"], L["sdip"], f, SW_SIZE if e else 0.0,
                                                tp, SW_MAX, d, r, ex, 0.0)
                eqs.append(eq); ntr += t; nst += st; nsw += sw; spnl += sp
            m = window_metrics(portfolio(eqs, S), sel)
            rows.append({"win": win, "swing": f"{sw_tf}{'연장' if sw_eth else '정규'}", "mode": mode, "variant": name,
                         "trades": ntr, "sw_trades": nst, "sw_win": nsw / nst if nst else np.nan,
                         "sw_avg": spnl / nst if nst else np.nan, **m})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = pd.concat([run("long", 60, False), run("short", 40, False), run("short", 40, True)], ignore_index=True)
    out.to_parquet("run_v2.parquet")
    F = lambda x: f"{x:+.1%}"
    for (win, sw), g in out.groupby(["win", "swing"], sort=False):
        print(f"\n===== {win} / 스윙 {sw} =====")
        base = g[g.variant == "뼈대만"].set_index("mode")
        for mode in ("M0", "M1", "MF"):
            b = base.loc[mode]
            print(f"[{mode}] 뼈대만: 전체 {F(b.all_ret)} 샤프 {b.all_sh:.2f} 낙폭 {F(b.all_mdd)} | 선택 {F(b.sel_ret)} (샤프 {b.sel_sh:.2f}) | 확인 {F(b.val_ret)} (샤프 {b.val_sh:.2f})")
        gg = g[(g["mode"] == "M1") & (g.variant != "뼈대만")].sort_values("sel_sh", ascending=False)
        print("  M1(사용자 조건) 스윙 추가 32가지 — 선택 구간 샤프 순 상위 8 / 하위 3")
        show = pd.concat([gg.head(8), gg.tail(3)])
        print(show[["variant", "sel_ret", "sel_sh", "val_ret", "val_sh", "all_ret", "all_mdd", "sw_trades", "sw_win", "sw_avg"]]
              .to_string(index=False, formatters={c: F for c in ["sel_ret", "val_ret", "all_ret", "all_mdd", "sw_avg"]} |
                         {c: "{:.2f}".format for c in ["sel_sh", "val_sh"]} | {"sw_win": "{:.0%}".format}))
        print(f"  스윙 추가 32가지 중 뼈대보다 선택 구간 샤프가 높은 것: {(gg.sel_sh > base.loc['M1'].sel_sh).sum()}개, "
              f"확인 구간 수익이 높은 것: {(gg.val_ret > base.loc['M1'].val_ret).sum()}개")
