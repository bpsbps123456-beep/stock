"""RSI·스토캐스틱RSI(Pine과 같은 계산)와 신호 발생 뒤 움직임(이벤트 연구)."""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def rma_skipna(x, n):
    """Pine ta.rma: 앞쪽 na를 건너뛰고 처음 n개 평균으로 시작, 이후 (x + (n-1)*s)/n."""
    m = len(x)
    out = np.full(m, np.nan)
    start = -1
    for i in range(m):
        if not np.isnan(x[i]):
            start = i
            break
    if start < 0:
        return out
    s = np.nan
    for i in range(start, m):
        k = i - start
        if k == n - 1:
            acc = 0.0
            for j in range(start, i + 1):
                acc += x[j]
            s = acc / n
        elif k >= n:
            s = (x[i] + (n - 1) * s) / n
        out[i] = s
    return out


@njit(cache=True)
def rsi(c, n):
    m = len(c)
    up = np.full(m, np.nan)
    dn = np.full(m, np.nan)
    for i in range(1, m):
        ch = c[i] - c[i - 1]
        up[i] = ch if ch > 0 else 0.0
        dn[i] = -ch if ch < 0 else 0.0
    ru = rma_skipna(up, n)
    rd = rma_skipna(dn, n)
    out = np.full(m, np.nan)
    for i in range(m):
        if np.isnan(ru[i]) or np.isnan(rd[i]):
            continue
        if rd[i] == 0:
            out[i] = 100.0
        elif ru[i] == 0:
            out[i] = 0.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + ru[i] / rd[i])
    return out


@njit(cache=True)
def sma_strict(x, n):
    """Pine ta.sma: 창 안에 na가 있으면 na."""
    m = len(x)
    out = np.full(m, np.nan)
    for i in range(n - 1, m):
        acc = 0.0
        ok = True
        for j in range(i - n + 1, i + 1):
            if np.isnan(x[j]):
                ok = False
                break
            acc += x[j]
        if ok:
            out[i] = acc / n
    return out


@njit(cache=True)
def stoch_rsi(r, n_stoch, sk, sd):
    m = len(r)
    raw = np.full(m, np.nan)
    for i in range(n_stoch - 1, m):
        lo, hi, ok = 1e18, -1e18, True
        for j in range(i - n_stoch + 1, i + 1):
            if np.isnan(r[j]):
                ok = False
                break
            lo = min(lo, r[j])
            hi = max(hi, r[j])
        if ok and hi > lo:
            raw[i] = 100.0 * (r[i] - lo) / (hi - lo)
    k = sma_strict(raw, sk)
    d = sma_strict(k, sd)
    return k, d


@njit(cache=True)
def forward(c, h, l, H):
    """각 봉 종가 기준 H봉 뒤 수익률, 그 사이 최고가(MFE)·최저가(MAE)."""
    m = len(c)
    f = np.full(m, np.nan)
    up = np.full(m, np.nan)
    dn = np.full(m, np.nan)
    for i in range(m - H):
        hi, lo = -1e18, 1e18
        for j in range(i + 1, i + H + 1):
            hi = max(hi, h[j])
            lo = min(lo, l[j])
        f[i] = c[i + H] / c[i] - 1
        up[i] = hi / c[i] - 1
        dn[i] = lo / c[i] - 1
    return f, up, dn


@njit(cache=True)
def rsi_state(c, n):
    """RSI와 함께 상승·하락 RMA 상태를 돌려준다(다음 봉에서 RSI가 특정 값이 되는 가격 계산용)."""
    m = len(c)
    up = np.full(m, np.nan)
    dn = np.full(m, np.nan)
    for i in range(1, m):
        ch = c[i] - c[i - 1]
        up[i] = ch if ch > 0 else 0.0
        dn[i] = -ch if ch < 0 else 0.0
    ru = rma_skipna(up, n)
    rd = rma_skipna(dn, n)
    out = np.full(m, np.nan)
    for i in range(m):
        if np.isnan(ru[i]) or np.isnan(rd[i]):
            continue
        if rd[i] == 0:
            out[i] = 100.0
        elif ru[i] == 0:
            out[i] = 0.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + ru[i] / rd[i])
    return out, ru, rd


def rsi_price(c, ru, rd, n, thr):
    """다음 봉 종가가 이 가격이면 RSI(n)가 thr가 되는 가격(하락 방향). 이미 thr 아래면 현재가 바로 아래."""
    R = thr / (100.0 - thr)
    p = c - (n - 1) * (ru / R - rd)
    return np.minimum(p, c * 0.999)
