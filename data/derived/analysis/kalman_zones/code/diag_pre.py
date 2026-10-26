"""프리장 효과가 진짜인지: (1) 프리장 시작 시각 제한 (2) 꼬리 대신 봉 종가로 체결. 종가-1%, 필터 없음, 1세션, 프리장 매수·정리."""
import numpy as np, pandas as pd
from numba import njit
from core import *
from grid_short import portfolio, window_metrics
from run_on import ctx, day_arrays, levels


@njit(cache=True)
def sim3(o, h, l, c, is_reg, is_pre, is_awake, is_end, first_reg, day, j0, lvl, tgt, hold, pre_entry, exit_pre, close_fill):
    n = len(o)
    cash, sh = 1.0, 0.0
    holding, tp_on = False, False
    entry, eday, tp = 0.0, -1, 0.0
    n_end = 0
    for j in range(j0, n):
        if is_end[j]:
            n_end += 1
    eq = np.empty(n_end)
    k = 0
    ntr, npre = 0, 0
    for j in range(j0, n):
        d = day[j]
        if holding and (not tp_on) and is_awake[j]:
            tp_on = True
        pre_ok = is_pre[j] and is_awake[j]
        if holding and (is_reg[j] or pre_ok):
            hit_tp = (c[j] >= tp) if close_fill else (h[j] >= tp)
            if d >= eday + hold and ((exit_pre and pre_ok) or (is_reg[j] and first_reg[j]) or (exit_pre and is_reg[j])):
                cash = sh * o[j]; sh = 0.0; holding = False; tp_on = False; ntr += 1
            elif tp_on and hit_tp:
                px = c[j] if close_fill else max(o[j], tp)
                cash = sh * px; sh = 0.0; holding = False; tp_on = False; ntr += 1
                if is_pre[j]:
                    npre += 1
        ok = is_reg[j] or (pre_entry and pre_ok)
        if (not holding) and ok and d >= 0 and not np.isnan(lvl[d]):
            hit = (c[j] <= lvl[d]) if close_fill else (l[j] <= lvl[d])
            if hit:
                px = c[j] if close_fill else min(o[j], lvl[d])
                sh = cash / px; cash = 0.0
                entry, eday, holding = px, d, True
                tp = entry * (1 + tgt)
                tp_on = is_awake[j]
        if is_end[j]:
            eq[k] = cash + sh * c[j]
            k += 1
    return eq, ntr, npre


F = lambda x: f"{x:+.0%}"
for win, lab in (("long", "2026"), ("y2025", "2025"), ("short", "최근60일")):
    S, common, sel = ctx(win)
    for s in SYMS:
        day_arrays(S, s)
    LV = {s: levels(S, s, win)[("종가-1%", "없음")] for s in SYMS}
    out = []
    for name, pre_from, pe, xp, cf in [("정규장만 매수", 240, False, False, False),
                                      ("프리장 04시~ 꼬리 체결", 240, True, True, False),
                                      ("프리장 07시~ 꼬리 체결", 420, True, True, False),
                                      ("프리장 08시~ 꼬리 체결", 480, True, True, False),
                                      ("정규장만 · 종가 체결", 240, False, False, True),
                                      ("프리장 04시~ · 종가 체결", 240, True, True, True),
                                      ("프리장 07시~ · 종가 체결", 420, True, True, True)]:
        for tgt in (0.005, 0.01):
            eqs, nt, npre = [], 0, 0
            for s in SYMS:
                A = S[s]
                t = pd.to_datetime(A["start_ns"], utc=True).tz_convert("America/New_York")
                mins = np.asarray(t.hour * 60 + t.minute)
                pre = (mins >= pre_from) & (mins < 570)
                e, n1, n2 = sim3(A["o"], A["h"], A["l"], A["c"], A["is_reg"], pre, A["is_awake"], A["is_end"], A["first_reg"],
                                 A["day_idx"], A["j0"], LV[s], tgt, 1, pe, xp, cf)
                eqs.append(e); nt += n1; npre += n2
            m = window_metrics(portfolio(eqs, S), sel)
            out.append(f"{name} 목표{tgt*100:g}%: {F(m['all_ret'])} (하루 {nt/len(common):.1f}회, 프리장 익절 비중 {npre/max(nt,1):.0%})")
    print(f"\n[{lab}]")
    for o_ in out:
        print("  " + o_)
