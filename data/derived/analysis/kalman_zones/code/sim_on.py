"""초단타(밤사이 반등) 모의매매: 개장 전에 정한 지지선 지정가로 정규장에서 사고, 다음 날 프리장에서 목표가에 판다.

- 매수: 세션 d의 지정가 lvl[d](개장 전 정보로 계산, 없으면 NaN). 정규장 봉 저가가 lvl*(1-margin) 이하이면 min(시가, lvl)에 체결.
- 목표가 매도: entry*(1+tgt). 체결 시점에 깨어 있으면 바로, 아니면 다음에 깨어 있을 때부터 걸린다.
  체결 가능 봉 = 정규장(자는 동안도 걸어 둔 지정가) + 깨어 있는 프리장. 고가가 목표가*(1+margin) 이상이면 max(시가, 목표가).
- 시간 청산: 진입 후 hold번째 세션의 정규장 첫 봉 시가(목표가 미체결 시).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def sim_overnight(o, h, l, c, is_reg, is_pre, is_awake, is_end, first_reg, day, j0, lvl, tgt, hold, mcost, lcost, margin,
                  tp_lvl, tp_lo, tp_hi):
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
    ntr, nwin, sret, hold_bars = 0, 0, 0.0, 0
    for j in range(j0, n):
        d = day[j]
        if holding and (not tp_on) and is_awake[j]:
            tp_on = True
        can = is_reg[j] or (is_pre[j] and is_awake[j])
        if holding and can:
            if is_reg[j] and first_reg[j] and d >= eday + hold:
                px = o[j]
                cash = sh * px * (1 - mcost)
                r = px / entry - 1
                ntr += 1; nwin += 1 if r > 0 else 0; sret += r
                sh, holding, tp_on = 0.0, False, False
            elif tp_on and tp > 0 and h[j] >= tp * (1 + margin):
                px = max(o[j], tp)
                cash = sh * px * (1 - lcost)
                r = px / entry - 1
                ntr += 1; nwin += 1 if r > 0 else 0; sret += r
                sh, holding, tp_on = 0.0, False, False
        if (not holding) and is_reg[j] and d >= 0 and not np.isnan(lvl[d]) and l[j] <= lvl[d] * (1 - margin):
            px = min(o[j], lvl[d])
            sh = cash * (1 - lcost) / px
            cash = 0.0
            entry, eday, holding = px, d, True
            if np.isnan(tp_lvl[d]):
                tp = entry * (1 + tgt) if tgt > 0 else 0.0
            else:
                tp = min(max(tp_lvl[d] * 0.998, entry * (1 + tp_lo)), entry * (1 + tp_hi))
            tp_on = is_awake[j]
        if holding:
            hold_bars += 1
        if is_end[j]:
            eq[k] = cash + sh * c[j]
            k += 1
    return eq, ntr, nwin, sret


@njit(cache=True)
def sim_close_to_open(o, c, is_reg, is_end, first_reg, j0, mcost):
    """참고 대조군(사용자는 실행 불가): 매일 정규장 종가에 사서 다음 날 시가에 판다."""
    n = len(o)
    cash, sh = 1.0, 0.0
    n_end = 0
    for j in range(j0, n):
        if is_end[j]:
            n_end += 1
    eq = np.empty(n_end)
    k = 0
    for j in range(j0, n):
        if sh > 0 and is_reg[j] and first_reg[j]:
            cash = sh * o[j] * (1 - mcost)
            sh = 0.0
        if is_end[j]:
            eq[k] = cash + sh * c[j]
            k += 1
            if cash > 0:
                sh = cash * (1 - mcost) / c[j]
                cash = 0.0
    return eq
