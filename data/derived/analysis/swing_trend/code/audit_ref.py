"""audit 갈래: swing.py를 쓰지 않고 BRIEF.md 정의만 보고 다시 짠 참조 계산(순수 파이썬·pandas, numba 없음).

- 판정 6개: ZZP_th0.1, ZZA_k3.0, CR_n10_b0.5_s0.5, EMA_f10_s30, DON_n20_m10, Z5
- Z5는 두 가지: z5_swing(시작값 seed, 예열 없음 — swing.py가 택한 방식) / z5_pine(trend_5zone.pine 그대로: RSI는 SMA seed RMA, bar_index < 129면 na)
- 체결: 날 d 종가 상태 → d+1 시가 체결. 매매 장부(진입가·청산가)로 수익·낙폭을 다시 센다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ---------- 보조 ----------
def wilder_atr(h, l, c, n=20):
    m = len(c)
    tr = [0.0] * m
    for i in range(m):
        if i == 0:
            tr[i] = h[i] - l[i]
        else:
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    out = [math.nan] * m
    if m >= n:
        a = sum(tr[:n]) / n
        out[n - 1] = a
        for i in range(n, m):
            a = (a * (n - 1) + tr[i]) / n
            out[i] = a
    return np.array(out)


def ema_seed_first(x, span):
    a = 2.0 / (span + 1.0)
    out = [0.0] * len(x)
    y = None
    for i, v in enumerate(x):
        y = v if y is None else a * v + (1 - a) * y
        out[i] = y
    return np.array(out)


# ---------- 판정 ----------
def zz_state(c, rule):
    """BRIEF 정의: 현금(하락)에서 시작. 하락 상태면 그동안의 최저 종가를 따라가다 최저 대비 문턱만큼 오르면 상승 확정,
    상승 상태면 최고 종가를 따라가다 최고 대비 문턱만큼 내리면 하락 확정. 확정 날부터 새 극점을 그 날 종가로 다시 센다.
    rule(i, ext, up) → 문턱 가격(없으면 None)."""
    st = []
    piv = []  # (극점 날, 확정 날, 종류)
    up = False
    ext, ext_i = c[0], 0
    for i, x in enumerate(c):
        if not up:
            if x < ext:
                ext, ext_i = x, i
            thr = rule(i, ext, False)
            if thr is not None and x >= thr:
                piv.append((ext_i, i, 1))
                up, ext, ext_i = True, x, i
        else:
            if x > ext:
                ext, ext_i = x, i
            thr = rule(i, ext, True)
            if thr is not None and x <= thr:
                piv.append((ext_i, i, -1))
                up, ext, ext_i = False, x, i
        st.append(1 if up else 0)
    return np.array(st, dtype=np.int64), piv


def zzp(c, th):
    return zz_state(list(c), lambda i, e, up: e * (1 - th) if up else e * (1 + th))


def zza(h, l, c, k, n=20):
    a = wilder_atr(list(h), list(l), list(c), n)

    def rule(i, e, up):
        if math.isnan(a[i]):
            return None
        return e - k * a[i] if up else e + k * a[i]

    return zz_state(list(c), rule)


def cr(o, c, n, b, s):
    o = pd.Series(o); c = pd.Series(c)
    up = (c > o).astype(float).rolling(n).sum()
    dn = (c < o).astype(float).rolling(n).sum()
    tot = up + dn
    p = (up / tot.where(tot > 0)).fillna(0.5).where(up.notna())
    st = np.zeros(len(c), np.int64)
    cur = 0
    for i, v in enumerate(p.values):
        if not np.isnan(v):
            if cur == 0 and v > b:
                cur = 1
            elif cur == 1 and v < s:
                cur = 0
        st[i] = cur
    return st


def ema_cross(c, f, s):
    st = (ema_seed_first(c, f) > ema_seed_first(c, s)).astype(np.int64)
    st[:s] = 0
    return st


def don(h, l, c, n, m):
    h = pd.Series(h); l = pd.Series(l)
    hi = h.rolling(n).max().shift(1).values
    lo = l.rolling(m).min().shift(1).values
    st = np.zeros(len(c), np.int64)
    cur = 0
    for i in range(len(c)):
        if i >= max(n, m):
            if cur == 0 and c[i] > hi[i]:
                cur = 1
            elif cur == 1 and c[i] < lo[i]:
                cur = 0
        st[i] = cur
    return st


def _zone_from(lc, rsi, n=20, th=0.5, valid_from=0):
    """lc: 로그 종가, rsi 배열. z = (lc − N일 평균) ÷ (N일 로그수익 표본표준편차 × k)."""
    m = len(lc)
    k = math.sqrt((n - 1.0) * (2.0 * n - 1.0) / (6.0 * n))
    lr = np.r_[np.nan, np.diff(lc)]
    zn = np.full(m, np.nan)
    for i in range(m):
        if i < n or i < valid_from:  # lr 창 n개(첫 lr은 1번 날) → i ≥ n부터
            continue
        w = lr[i - n + 1:i + 1]
        sd = float(np.std(w, ddof=1))
        mu = float(np.mean(lc[i - n + 1:i + 1]))
        if not (sd > 0) or np.isnan(rsi[i]):
            continue
        z = (lc[i] - mu) / (sd * k)
        r = rsi[i]
        zn[i] = 2 if r >= 70 else -2 if r <= 30 else 1 if z >= th else -1 if z <= -th else 0
    return zn


def _rsi(up_avg, dn_avg):
    out = np.full(len(up_avg), np.nan)
    for i, (u, d) in enumerate(zip(up_avg, dn_avg)):
        if np.isnan(u) or np.isnan(d):
            continue
        if d == 0:
            out[i] = 100.0 if u > 0 else np.nan
        else:
            out[i] = 100.0 - 100.0 / (1.0 + u / d)
    return out


def rsi_seed_first(c, n=14):
    """swing.py가 택한 방식(pandas ewm alpha=1/n, adjust=False)과 같은 정의: 첫 변화값에서 시작하는 Wilder 평균."""
    m = len(c)
    ua = np.full(m, np.nan); da = np.full(m, np.nan)
    a = 1.0 / n
    u = d = None
    for i in range(1, m):
        du = max(c[i] - c[i - 1], 0.0); dd = max(c[i - 1] - c[i], 0.0)
        if u is None:
            u, d = du, dd
        else:
            u = a * du + (1 - a) * u
            d = a * dd + (1 - a) * d
        ua[i], da[i] = u, d
    return _rsi(ua, da)


def rsi_pine(c, n=14):
    """Pine ta.rsi: u = max(Δ,0), d = max(−Δ,0), ta.rma(첫 값 = 처음 n개 단순평균, 이후 Wilder)."""
    m = len(c)
    ua = np.full(m, np.nan); da = np.full(m, np.nan)
    a = 1.0 / n
    if m > n:
        us = [max(c[i] - c[i - 1], 0.0) for i in range(1, m)]
        ds = [max(c[i - 1] - c[i], 0.0) for i in range(1, m)]
        u = sum(us[:n]) / n; d = sum(ds[:n]) / n
        ua[n], da[n] = u, d
        for i in range(n + 1, m):
            u = a * us[i - 1] + (1 - a) * u
            d = a * ds[i - 1] + (1 - a) * d
            ua[i], da[i] = u, d
    return _rsi(ua, da)


def z5_zone(c, pine=False):
    c = np.asarray(c, float)
    lc = np.log(c)
    if pine:
        return _zone_from(lc, rsi_pine(c), valid_from=129)
    return _zone_from(lc, rsi_seed_first(c))


def z5_hold(zn):
    st = np.zeros(len(zn), np.int64)
    cur = 0
    for i, z in enumerate(zn):
        if not np.isnan(z):
            if z >= 1:
                cur = 1
            elif z <= -1:
                cur = 0
        st[i] = cur
    return st


def six_states(g, z5_pine=False):
    O, H, L, C = (g[k].to_numpy(float) for k in ("O", "H", "L", "C"))
    return {
        "ZZP_th0.1": zzp(C, 0.10)[0],
        "ZZA_k3.0": zza(H, L, C, 3.0)[0],
        "CR_n10_b0.5_s0.5": cr(O, C, 10, 0.5, 0.5),
        "EMA_f10_s30": ema_cross(C, 10, 30),
        "DON_n20_m10": don(H, L, C, 20, 10),
        "Z5": z5_hold(z5_zone(C, pine=z5_pine)),
    }


# ---------- 회계(매매 장부) ----------
def ledger_sim(o, c, st, j0, j1, cost):
    """구간 [j0, j1]. 날 j 보유 = st[j−1](전날 종가 판정) → 날 j 시가에 체결.
    j0 전날 보유면 O[j0]에 비용 없이 들고 시작(진입 수에 안 셈), 구간 끝은 C[j1]로 평가(비용 없음).
    평가점: O[j0](=1), O[j0+1], …, O[j1], C[j1]. 반환: 수익, 최대 낙폭, 평균 보유, 진입 수, 매매 목록."""
    pos = [int(st[j - 1]) if j > 0 else 0 for j in range(j0, j1 + 1)]
    cash, sh = 1.0, 0.0
    if pos[0] == 1:
        sh, cash = 1.0 / o[j0], 0.0
    marks = [1.0]
    trades = []
    ent = 0
    entry = (j0, o[j0]) if pos[0] == 1 else None
    for t, j in enumerate(range(j0, j1 + 1)):
        if t > 0 and pos[t] != pos[t - 1]:
            if pos[t] == 1:
                cash *= 1 - cost
                sh, cash = cash / o[j], 0.0
                ent += 1
                entry = (j, o[j])
            else:
                cash, sh = sh * o[j] * (1 - cost), 0.0
                trades.append((entry[0], j, entry[1], o[j]))
                entry = None
        px_end = o[j + 1] if j < j1 else c[j1]
        marks.append(cash + sh * px_end)
    if entry is not None:
        trades.append((entry[0], j1, entry[1], c[j1]))
    marks = np.array(marks)
    mdd = float((marks / np.maximum.accumulate(marks) - 1).min())
    return float(marks[-1] - 1), mdd, float(np.mean(pos)), ent, trades


def static_mix(o, c, j0, j1, x):
    """같은 비중 그냥 보유: O[j0]에 x만 사서 끝까지(재조정 없음), 나머지 현금. 평가점은 ledger_sim과 같다."""
    px = np.array(list(o[j0:j1 + 1]) + [c[j1]], float)
    eq = (1 - x) + x * px / px[0]
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return float(eq[-1] - 1), mdd
