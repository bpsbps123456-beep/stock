"""초단타 v2: 프리장 매수(pre_entry)와 프리장 시간 정리(exit_pre)를 더한 sim_overnight."""
import numpy as np
from numba import njit


@njit(cache=True)
def sim_overnight2(o, h, l, c, is_reg, is_pre, is_awake, is_end, first_reg, day, j0, lvl, tgt, hold, cost, margin,
                   pre_entry, exit_pre):
    """lvl[d]: 세션 d(날짜)의 매수 지정가(전일 정규장 종가 등으로 계산). 프리장 봉도 같은 날짜 d.
    매수: 정규장, 또는 pre_entry면 깨어 있는 프리장에서 저가가 지정가에 닿으면 min(시가, 지정가).
    익절: 매수 다음 봉부터, 깨어 있는 프리장·정규장(자는 동안은 걸어 둔 지정가로 정규장)에서 목표가 도달 시.
    시간 정리: 세션 d >= 매수일+hold 의 첫 체결 가능 봉 시가 (exit_pre면 깨어 있는 프리장부터, 아니면 정규장 첫 봉).
    """
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
    ntr, nwin, sret = 0, 0, 0.0
    for j in range(j0, n):
        d = day[j]
        if holding and (not tp_on) and is_awake[j]:
            tp_on = True
        pre_ok = is_pre[j] and is_awake[j]
        can = is_reg[j] or pre_ok
        if holding and can:
            if d >= eday + hold and ((exit_pre and pre_ok) or (is_reg[j] and first_reg[j]) or (exit_pre and is_reg[j])):
                px = o[j]
                cash = sh * px * (1 - cost)
                r = px / entry - 1
                ntr += 1; nwin += 1 if r > 0 else 0; sret += r
                sh, holding, tp_on = 0.0, False, False
            elif tp_on and h[j] >= tp * (1 + margin):
                px = max(o[j], tp)
                cash = sh * px * (1 - cost)
                r = px / entry - 1
                ntr += 1; nwin += 1 if r > 0 else 0; sret += r
                sh, holding, tp_on = 0.0, False, False
        entry_ok = is_reg[j] or (pre_entry and pre_ok)
        if (not holding) and entry_ok and d >= 0 and not np.isnan(lvl[d]) and l[j] <= lvl[d] * (1 - margin):
            px = min(o[j], lvl[d])
            sh = cash * (1 - cost) / px
            cash = 0.0
            entry, eday, holding = px, d, True
            tp = entry * (1 + tgt)
            tp_on = is_awake[j]
        if is_end[j]:
            eq[k] = cash + sh * c[j]
            k += 1
    return eq, ntr, nwin, sret
