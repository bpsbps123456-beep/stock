"""칼만 240분 뼈대 + 짧은 봉 과매도 스윙(분할매수→반등 분할매도) 모의매매. 사용자 체결 조건 반영."""
import numpy as np
from numba import njit


@njit(cache=True)
def simulate_v2(o, h, l, c, exec_ok, is_awake, is_end, j0,
                bidx, bzone, btp, bpb, tp_half, panic_hold, cost,
                sidx, ssig, srsi, sdip, filt, sw_size, sw_tp, sw_max, use_dip, reserve, lim_ok, lim_cost, lmargin=0.0):
    """뼈대(b)는 simulate와 같은 규칙. 스윙(s)은 과매도 신호 봉에 sw_size만큼 사고, RSI>=50·보유 sw_max봉·익절 지정가(+sw_tp)에 판다.
    reserve=1 이면 뼈대 '전량'을 자본의 (1-sw_size)로 줄여 스윙 몫을 항상 남긴다. 0 이면 남는 현금만 쓴다.
    """
    n = len(o)
    cash, shb, shs = 1.0, 0.0, 0.0
    state, desired, etype = 0, 0, 0
    last_b, last_zone = -1, -99
    blim_buy, blim_sell = np.nan, np.nan
    last_s = -1
    s_want = 0            # 1 스윙 매수 원함, -1 스윙 청산 원함, 0 없음
    s_open = False
    s_entry, s_bars = 0.0, 0
    s_tp, s_dip = np.nan, np.nan
    full = 1.0 - sw_size if reserve else 1.0
    n_end = 0
    for j in range(j0, n):
        if is_end[j]:
            n_end += 1
    eq = np.empty(n_end)
    k = 0
    trades, s_trades, s_wins = 0, 0, 0
    s_pnl = 0.0
    for j in range(j0, n):
        px_mark = c[j]
        # ── 뼈대 판단 ──
        i = bidx[j]
        if is_awake[j] and i >= 0 and i != last_b and bzone[i] != -9:
            last_b = i
            z = int(bzone[i])
            if panic_hold and etype == 2 and z >= 0:
                etype = 1
            if z == 2:
                if last_zone != 2 and desired > 0:
                    desired = (desired - 1) if tp_half else 0
            elif z == 1:
                if desired < 2:
                    if desired == 0:
                        etype = 1
                    desired = 2
            elif z == -1:
                if desired > 0 and not (panic_hold and etype == 2):
                    desired = 0
            elif z == -2:
                if desired < 2:
                    if desired == 0:
                        etype = 2
                    desired = 2
            last_zone = z
            blim_buy = bpb[i] if desired < 2 else np.nan
            blim_sell = btp[i] if desired > 0 else np.nan
        # ── 스윙 판단 ──
        si = sidx[j]
        if is_awake[j] and si >= 0 and si != last_s:
            last_s = si
            bz = int(bzone[bidx[j]]) if bidx[j] >= 0 else -9
            ok = (filt == 0) or (bz >= 0)
            if s_open:
                s_bars += 1
                if srsi[si] >= 50.0 or s_bars >= sw_max:
                    s_want = -1
                if np.isnan(s_tp):
                    s_tp = s_entry * (1.0 + sw_tp)
                s_dip = np.nan
            else:
                if ssig[si] and ok:
                    s_want = 1
                s_dip = sdip[si] if (use_dip and ok) else np.nan
        # ── 체결(정규장, 또는 깨어 있을 때 프리장) ──
        if exec_ok[j]:
            eqv = cash + (shb + shs) * o[j]
            # 뼈대 시장가
            if desired != state:
                if desired > state:
                    target = full * eqv
                    have = (shb + shs) * o[j]
                    spend = min(cash, max(0.0, target - have))
                    shb += shs + spend * (1 - cost) / o[j]
                    shs = 0.0
                    s_open = False
                    s_tp = np.nan
                    cash -= spend
                elif desired == 1:
                    sell = 0.5 * shb
                    cash += sell * o[j] * (1 - cost)
                    shb -= sell
                else:
                    cash += (shb + shs) * o[j] * (1 - cost)
                    shb, shs = 0.0, 0.0
                    if s_open:
                        s_open = False
                        s_tp = np.nan
                state = desired
                trades += 1
            # 뼈대 지정가
            if lim_ok[j] and (not np.isnan(blim_buy)) and state < 2 and l[j] <= blim_buy * (1 - lmargin):
                p = min(o[j], blim_buy)
                if state == 0:
                    etype = 2
                eqv = cash + (shb + shs) * p
                spend = min(cash, max(0.0, full * eqv - (shb + shs) * p))
                shb += shs + spend * (1 - lim_cost) / p
                shs = 0.0
                s_open = False
                s_tp = np.nan
                cash -= spend
                state = 2
                desired = 2
                blim_buy = np.nan
                trades += 1
            elif lim_ok[j] and (not np.isnan(blim_sell)) and state > 0 and h[j] >= blim_sell * (1 + lmargin):
                p = max(o[j], blim_sell)
                if tp_half and state == 2:
                    sell = 0.5 * shb
                    cash += sell * p * (1 - lim_cost)
                    shb -= sell
                    state = 1
                else:
                    cash += shb * p * (1 - lim_cost)
                    shb = 0.0
                    state = 0
                desired = state
                blim_sell = np.nan
                trades += 1
            # 스윙 시장가
            if s_want == 1 and not s_open:
                eqv = cash + (shb + shs) * o[j]
                spend = min(cash, sw_size * eqv)
                if spend > 1e-9:
                    shs = spend * (1 - cost) / o[j]
                    cash -= spend
                    s_open, s_entry, s_bars, s_tp = True, o[j], 0, np.nan
                    trades += 1
                s_want = 0
            elif s_want == -1 and s_open:
                cash += shs * o[j] * (1 - cost)
                r = o[j] / s_entry - 1
                s_pnl += r; s_trades += 1; s_wins += 1 if r > 0 else 0
                shs = 0.0
                s_open, s_tp = False, np.nan
                s_want = 0
                trades += 1
            # 스윙 지정가
            if lim_ok[j] and (not s_open) and (not np.isnan(s_dip)) and l[j] <= s_dip and cash > 1e-9:
                p = min(o[j], s_dip)
                eqv = cash + (shb + shs) * p
                spend = min(cash, sw_size * eqv)
                shs = spend * (1 - lim_cost) / p
                cash -= spend
                s_open, s_entry, s_bars, s_tp = True, p, 0, np.nan
                s_dip = np.nan
                trades += 1
            elif lim_ok[j] and s_open and (not np.isnan(s_tp)) and h[j] >= s_tp:
                p = max(o[j], s_tp)
                cash += shs * p * (1 - lim_cost)
                r = p / s_entry - 1
                s_pnl += r; s_trades += 1; s_wins += 1 if r > 0 else 0
                shs = 0.0
                s_open, s_tp = False, np.nan
                s_want = 0
                trades += 1
        if is_end[j]:
            eq[k] = cash + (shb + shs) * px_mark
            k += 1
    return eq, trades, s_trades, s_wins, s_pnl
