"""칼만 구간 이동에 맞춘 단계(분할) 매매. 목표 비중 T를 구간이 바뀔 때마다 조정한다.

  → 과매수            : T = min(T, keep_tp)          [익절]
  평범·매도·패닉 → 매수 : T = max(T, buy_to)           [분할매수: 추세 따라붙기]
  과매수 → 매수        : T = max(T, buy_to)           [되사기]
  매수·과매수 → 평범    : T = min(T, cut_to)           [분할매도: 약해짐]
  매도·패닉 → 평범      : T = max(T, recover_to)       [분할매수: 회복 초입]
  → 매도              : T = 0 (패닉 매수분은 평범 회복 전까지 보유) [손절]
  → 패닉셀            : T = 1                        [패닉 매수]
  (선택) 평범에서 짧은 봉 RSI 과매도 시작 : T = max(T, rsi_to)
지정가: 패닉 매수(패닉 라인, T→1), 익절(과매수 라인, T→keep_tp).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def simulate_ladder(o, h, l, c, exec_ok, is_awake, is_end, j0, bidx, bzone, btp, bpb,
                    keep_tp, buy_to, cut_to, recover_to, sidx, ssig, rsi_to, cost):
    n = len(o)
    cash, sh = 1.0, 0.0
    T = 0.0
    panic_pos = False
    last_b, prev = -1, -99
    last_s = -1
    lim_buy, lim_sell = np.nan, np.nan
    n_end = 0
    for j in range(j0, n):
        if is_end[j]:
            n_end += 1
    eq = np.empty(n_end)
    k = 0
    trades = 0
    for j in range(j0, n):
        i = bidx[j]
        if is_awake[j] and i >= 0 and i != last_b and bzone[i] != -9:
            last_b = i
            z = int(bzone[i])
            if panic_pos and z >= 0:
                panic_pos = False
            if z != prev:
                if z == 2:
                    T = min(T, keep_tp)
                elif z == 1:
                    T = max(T, buy_to)
                elif z == 0:
                    if prev == 1 or prev == 2:
                        T = min(T, cut_to)
                    elif prev == -1 or prev == -2:
                        T = max(T, recover_to)
                elif z == -1:
                    if not panic_pos:
                        T = 0.0
                elif z == -2:
                    T = 1.0
                    panic_pos = True
            prev = z
            lim_buy = bpb[i] if T < 1.0 else np.nan
            lim_sell = btp[i] if T > keep_tp else np.nan
        si = sidx[j]
        if rsi_to > 0 and is_awake[j] and si >= 0 and si != last_s:
            last_s = si
            if ssig[si] and prev == 0:
                T = max(T, rsi_to)
        if exec_ok[j]:
            eqv = cash + sh * o[j]
            cur = sh * o[j] / eqv if eqv > 0 else 0.0
            if abs(cur - T) > 0.02:
                if T > cur:
                    spend = min(cash, (T - cur) * eqv)
                    sh += spend * (1 - cost) / o[j]
                    cash -= spend
                else:
                    sell = min(sh, (cur - T) * eqv / o[j])
                    cash += sell * o[j] * (1 - cost)
                    sh -= sell
                trades += 1
            if (not np.isnan(lim_buy)) and l[j] <= lim_buy:
                p = min(o[j], lim_buy)
                eqv = cash + sh * p
                spend = min(cash, max(0.0, eqv - sh * p))
                sh += spend / p
                cash -= spend
                T = 1.0
                panic_pos = True
                lim_buy = np.nan
                trades += 1
            elif (not np.isnan(lim_sell)) and h[j] >= lim_sell:
                p = max(o[j], lim_sell)
                eqv = cash + sh * p
                target_sh = keep_tp * eqv / p
                if sh > target_sh:
                    cash += (sh - target_sh) * p
                    sh = target_sh
                    trades += 1
                T = min(T, keep_tp)
                lim_sell = np.nan
        if is_end[j]:
            eq[k] = cash + sh * c[j]
            k += 1
    return eq, trades
