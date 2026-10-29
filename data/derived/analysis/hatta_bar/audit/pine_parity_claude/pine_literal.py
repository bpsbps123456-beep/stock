"""Literal bar-by-bar translation of hatta_bar.pine (Pine v6) for parity checks.

Translated only from ../../hatta_bar.pine. Pine semantics emulated:
- script body runs once per bar, bar_index 0..n-1; barstate.isconfirmed is true on every historical bar
- float series history x[i] is na (NaN) before the first bar; comparisons with na are false
- bool series history out of range is false; int history out of range is na
- ta.lowest(src, len) is na until len bars exist; ta.sma likewise; ta.atr = ta.rma(ta.tr(true), len)
  with rma seeded by ta.sma of the first len values
- each f_ema call site keeps its own `var` state
- `var` arrays persist across bars
- division by zero: Pine yields na (no +inf); switch DIV0_INF=True to test the alternative
"""
import math
import numpy as np
from numba import njit

M_TOUCH, M_LEVEL = 0, 1
T_OFF, T_NOW, T_PER = 0, 1, 2
R_WICK, R_CPOS = 0, 1
E_BASIC, E_BODY, E_FULL, E_RECL = 0, 1, 2, 3

DEFAULTS = dict(nBars=120, tolPct=0.05, pSw=3, kModeA=M_LEVEL, kMinA=2, kModeB=M_TOUCH, kMinB=1,
                trendMd=T_PER, dropPct=2.0, showA=True, recMd=R_WICK, recMin=0.5, rngMin=1.0,
                needAb=False, showB=True, needBl=True, engMd=E_BASIC, volMin=0.0, minNot=25000.0,
                minRng=0.2, minDens=15, noNight=False)

NAN = np.nan


@njit(cache=True)
def run(o, h, l, c, v, t, dayNY, hrNY,
        nBars, tolPct, pSw, kModeA, kMinA, kModeB, kMinB, trendMd, dropPct, showA, recMd, recMin,
        rngMin, needAb, showB, needBl, engMd, volMin, minNot, minRng, minDens, noNight, div0_inf):
    n = len(c)
    # series storage (every series referenced with [] gets an array)
    e10 = np.full(n, NAN); e25 = np.full(n, NAN); e60 = np.full(n, NAN)
    stack = np.zeros(n, np.bool_)
    s30 = np.full(n, NAN)
    persist = np.zeros(n, np.bool_)
    tr = np.full(n, NAN)
    atr14 = np.full(n, NAN)
    smaV = np.full(n, NAN)
    volR = np.full(n, NAN)
    hh120 = np.full(n, NAN)
    nIn30 = np.zeros(n, np.int64)
    S = np.full(n, NAN)
    thr = np.full(n, NAN)
    brk = np.zeros(n, np.bool_)
    rng = np.full(n, NAN)
    sane = np.zeros(n, np.bool_)
    densOk = np.zeros(n, np.bool_)
    dd = np.full(n, NAN)
    trendNow = np.zeros(n, np.bool_)
    dropOk = np.zeros(n, np.bool_)
    volOk = np.zeros(n, np.bool_)
    isSw = np.zeros(n, np.bool_)
    night = np.zeros(n, np.bool_)
    nTouch = np.full(n, NAN)
    nLevel = np.full(n, NAN)
    kCntA = np.full(n, NAN)
    kCntB = np.full(n, NAN)
    sigA = np.zeros(n, np.bool_)
    sigB = np.zeros(n, np.bool_)

    # var state
    ve10 = NAN; ve25 = NAN; ve60 = NAN
    a10 = 2.0 / (10 + 1); a25 = 2.0 / (25 + 1); a60 = 2.0 / (60 + 1)
    rma_prev = NAN
    swP = np.empty(n + 1, np.float64)  # var array<float> swP (deque emulated with head pointer)
    swB = np.empty(n + 1, np.int64)
    sw_head = 0
    sw_tail = 0
    tol = tolPct / 100

    for bi in range(n):
        # f_ema(close, 10/25/60): each call site has its own var e
        ve10 = c[bi] if math.isnan(ve10) else a10 * c[bi] + (1 - a10) * ve10
        ve25 = c[bi] if math.isnan(ve25) else a25 * c[bi] + (1 - a25) * ve25
        ve60 = c[bi] if math.isnan(ve60) else a60 * c[bi] + (1 - a60) * ve60
        e10[bi] = ve10; e25[bi] = ve25; e60[bi] = ve60
        stack[bi] = e10[bi] < e25[bi] and e25[bi] < e60[bi]
        # ta.sma(stack ? 1.0 : 0.0, 30)
        if bi >= 29:
            ssum = 0.0
            for j in range(bi - 29, bi + 1):
                ssum += 1.0 if stack[j] else 0.0
            s30[bi] = ssum / 30
        e60_30 = e60[bi - 30] if bi >= 30 else NAN
        persist[bi] = (s30[bi] >= 0.8) and (e60[bi] < e60_30)
        # ta.atr(14) = ta.rma(ta.tr(true), 14)
        if bi == 0:
            tr[bi] = h[bi] - l[bi]
        else:
            tr[bi] = max(max(h[bi] - l[bi], abs(h[bi] - c[bi - 1])), abs(l[bi] - c[bi - 1]))
        if math.isnan(rma_prev):
            if bi >= 13:
                ssum = 0.0
                for j in range(bi - 13, bi + 1):
                    ssum += tr[j]
                rma_prev = ssum / 14
        else:
            rma_prev = (1.0 / 14) * tr[bi] + (1 - 1.0 / 14) * rma_prev
        atr14[bi] = rma_prev
        # volR = volume / ta.sma(volume, 20)[1]
        if bi >= 19:
            ssum = 0.0
            for j in range(bi - 19, bi + 1):
                ssum += v[j]
            smaV[bi] = ssum / 20
        den = smaV[bi - 1] if bi >= 1 else NAN
        if math.isnan(den):
            volR[bi] = NAN
        elif den == 0.0:
            if div0_inf and v[bi] > 0:
                volR[bi] = np.inf
            else:
                volR[bi] = NAN
        else:
            volR[bi] = v[bi] / den

        # hh120 / nIn30 loop
        hhv = h[bi]
        cnt = 1
        for i in range(1, 481):
            if bi - i < 0 or t[bi - i] <= t[bi] - 120 * 60000:
                break
            hhv = max(hhv, h[bi - i])
            if t[bi - i] > t[bi] - 30 * 60000:
                cnt += 1
        hh120[bi] = hhv
        nIn30[bi] = cnt

        # S = ta.lowest(low, nBars - 1)[2]
        L = nBars - 1
        j0 = bi - 2
        if j0 >= 0 and j0 >= L - 1:
            mn = l[j0]
            for j in range(j0 - L + 1, j0 + 1):
                if l[j] < mn:
                    mn = l[j]
            S[bi] = mn
        thr[bi] = S[bi] * (1 - tol)
        low1 = l[bi - 1] if bi >= 1 else NAN
        brk[bi] = (low1 >= thr[bi]) and (l[bi] < thr[bi]) and (l[bi] < low1)
        rng[bi] = h[bi] - l[bi]
        single = o[bi] == h[bi] and h[bi] == l[bi] and l[bi] == c[bi]
        sane[bi] = (v[bi] * c[bi] >= minNot) and (rng[bi] >= minRng / 100 * c[bi]) and (not single)
        densOk[bi] = nIn30[bi] >= minDens
        dd[bi] = (hh120[bi] - l[bi]) / hh120[bi]
        if trendMd == T_OFF:
            trendNow[bi] = True
        elif trendMd == T_NOW:
            trendNow[bi] = stack[bi]
        else:
            trendNow[bi] = persist[bi]
        dropOk[bi] = dropPct <= 0 or dd[bi] >= dropPct / 100
        volOk[bi] = volMin <= 0 or volR[bi] >= volMin
        # isSw = low[pSw] < ta.lowest(low, pSw)[pSw + 1] and low[pSw] <= ta.lowest(low, pSw)
        lowp = l[bi - pSw] if bi - pSw >= 0 else NAN
        jb = bi - pSw - 1  # bar where ta.lowest(low,pSw) is read for [pSw+1]
        lo_before = NAN
        if jb >= 0 and jb >= pSw - 1:
            mn = l[jb]
            for j in range(jb - pSw + 1, jb + 1):
                if l[j] < mn:
                    mn = l[j]
            lo_before = mn
        lo_after = NAN
        if bi >= pSw - 1:
            mn = l[bi]
            for j in range(bi - pSw + 1, bi + 1):
                if l[j] < mn:
                    mn = l[j]
            lo_after = mn
        isSw[bi] = (lowp < lo_before) and (lowp <= lo_after)
        night[bi] = hrNY[bi] >= 20 or hrNY[bi] < 4

        # swing list (barstate.isconfirmed is always true on history)
        while (sw_tail - sw_head) > 0 and swB[sw_head] < bi - nBars:
            sw_head += 1
        if brk[bi]:
            nSw = sw_tail - sw_head
            pts = np.empty(nSw + 1, np.float64)
            npts = 0
            sIn = False
            if nSw > 0:
                for i in range(nSw):
                    if swB[sw_head + i] <= bi - 1 - pSw:
                        px = swP[sw_head + i]
                        pts[npts] = px
                        npts += 1
                        if px == S[bi]:
                            sIn = True
            if not sIn:
                pts[npts] = S[bi]
                npts += 1
            ps = np.sort(pts[:npts])
            tc = 0
            lc = 0
            prev = -1.0
            for i in range(npts):
                x = ps[i]
                if x <= S[bi] * (1 + 0.003):
                    tc += 1
                if x <= S[bi] * (1 + 0.01):
                    if lc == 0 or x > prev * (1 + 0.0015):
                        lc += 1
                    prev = x
            nTouch[bi] = tc
            nLevel[bi] = lc
        if bi >= 2 * pSw and isSw[bi]:
            swP[sw_tail] = l[bi - pSw]
            swB[sw_tail] = bi - pSw
            sw_tail += 1

        kCntA[bi] = nTouch[bi] if kModeA == M_TOUCH else nLevel[bi]
        kCntB[bi] = nTouch[bi] if kModeB == M_TOUCH else nLevel[bi]

        # ① tail reversal on the break bar itself
        if rng[bi] > 0:
            if recMd == R_WICK:
                rec = (min(o[bi], c[bi]) - l[bi]) / rng[bi]
            else:
                rec = (c[bi] - l[bi]) / rng[bi]
        else:
            rec = -1.0
        tn1 = trendNow[bi - 1] if bi >= 1 else False
        atr1 = atr14[bi - 1] if bi >= 1 else NAN
        sigA[bi] = (showA and brk[bi] and kCntA[bi] >= kMinA and tn1 and dropOk[bi] and volOk[bi]
                    and sane[bi] and densOk[bi]
                    and o[bi] >= thr[bi] and rec >= recMin and (rngMin <= 0 or rng[bi] >= rngMin * atr1)
                    and ((not needAb) or c[bi] >= S[bi])
                    and not (noNight and night[bi]))

        # ② engulfing green bar on the bar after the break bar
        if bi >= 1:
            if engMd == E_BASIC:
                engLvl = o[bi - 1] - 0.1 * (h[bi - 1] - l[bi - 1])
            elif engMd == E_BODY:
                engLvl = o[bi - 1]
            elif engMd == E_FULL:
                engLvl = h[bi - 1]
            else:
                engLvl = S[bi - 1]
            tn2 = trendNow[bi - 2] if bi >= 2 else False
            sigB[bi] = (showB and brk[bi - 1] and kCntB[bi - 1] >= kMinB and c[bi - 1] < o[bi - 1]
                        and dayNY[bi] == dayNY[bi - 1]
                        and ((not needBl) or c[bi - 1] < thr[bi - 1]) and tn2 and dropOk[bi - 1]
                        and volOk[bi - 1] and sane[bi - 1] and sane[bi] and densOk[bi]
                        and c[bi] > o[bi] and c[bi] >= engLvl and not (noNight and night[bi]))
    return sigA, sigB, S, nTouch, nLevel, brk
