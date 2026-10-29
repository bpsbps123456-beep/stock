"""hatta_bar.pine (현재 v3 소스, 204줄) 을 봉 하나씩 그대로 옮긴 참조 구현. 엔진 코드는 보지 않고 Pine만 읽고 옮겼다.

흉내 낸 Pine 의미
- 스크립트 본문은 봉마다 한 번(bar_index 0..n-1). 과거 봉에서 barstate.isconfirmed 는 항상 참.
- float 시리즈 x[i] 가 범위 밖이면 na(NaN). na 와의 비교는 거짓. bool 시리즈 범위 밖은 false(v6). int 범위 밖은 na.
- ta.lowest / ta.sma 는 length 봉이 쌓이기 전 na. ta.atr(14) = ta.rma(ta.tr(true), 14), rma 는 첫 14봉 sma 로 시작.
- f_ema 호출 자리마다 var 상태가 따로 있다.
- var 배열(swP, swB)은 봉을 넘어 유지.
- 모든 시리즈는 조건과 무관하게 매 봉 계산된다(ta.lowest(low,nBars)[1] 과 ta.lowest(low,nBars-1)[2] 둘 다).
줄 번호(Lnn)는 hatta_bar.pine 의 줄이다.
"""
import math
import numpy as np
from numba import njit

B_NEW, B_FIRST = 0, 1
M_TOUCH, M_LEVEL = 0, 1
T_OFF, T_NOW, T_PER = 0, 1, 2
R_WICK, R_CPOS = 0, 1
E_BASIC, E_BODY, E_FULL, E_RECL = 0, 1, 2, 3

# Pine 입력 기본값 (L18~L63)
DEFAULTS = dict(nBars=120, tolPct=0.05, pSw=2, brkMd=B_NEW, kModeA=M_LEVEL, kMinA=2, kModeB=M_TOUCH, kMinB=1,
                trendMd=T_PER, dropPct=2.0, showA=True, recMd=R_WICK, recMin=0.5, rngMin=0.0, needAb=False,
                showB=True, needBl=True, engMd=E_BASIC, volMin=0.0, minNot=25000.0, minRng=0.2, minDens=15,
                noNight=False, onlyTrd=False, dedup30=True)
ORDER = ["nBars", "tolPct", "pSw", "brkMd", "kModeA", "kMinA", "kModeB", "kMinB", "trendMd", "dropPct", "showA",
         "recMd", "recMin", "rngMin", "needAb", "showB", "needBl", "engMd", "volMin", "minNot", "minRng",
         "minDens", "noNight", "onlyTrd", "dedup30"]

NAN = np.nan


@njit(cache=True)
def _lowest_at(l, j, length):
    """ta.lowest(low, length) 시리즈의 봉 j 값. j < 0 이면 범위 밖(na), j < length-1 이면 아직 na."""
    if j < 0 or j < length - 1:
        return NAN
    mn = l[j]
    for q in range(j - length + 1, j + 1):
        if l[q] < mn:
            mn = l[q]
    return mn


@njit(cache=True)
def run(o, h, l, c, v, t, dayNY, hrNY, hrKST,
        nBars, tolPct, pSw, brkMd, kModeA, kMinA, kModeB, kMinB, trendMd, dropPct, showA, recMd, recMin,
        rngMin, needAb, showB, needBl, engMd, volMin, minNot, minRng, minDens, noNight, onlyTrd, dedup30):
    n = len(c)
    e60 = np.full(n, NAN)
    stack = np.zeros(n, np.bool_)
    persist = np.zeros(n, np.bool_)
    tr = np.full(n, NAN)
    atr14 = np.full(n, NAN)
    smaV = np.full(n, NAN)
    volR = np.full(n, NAN)
    S = np.full(n, NAN)
    thr = np.full(n, NAN)
    brk = np.zeros(n, np.bool_)
    sane = np.zeros(n, np.bool_)
    trendNow = np.zeros(n, np.bool_)
    dropOk = np.zeros(n, np.bool_)
    volOk = np.zeros(n, np.bool_)
    nTouch = np.full(n, NAN)
    nLevel = np.full(n, NAN)
    kCntB = np.full(n, NAN)
    keepA = np.zeros(n, np.bool_)
    keepB = np.zeros(n, np.bool_)
    sigA = np.zeros(n, np.bool_)
    sigB = np.zeros(n, np.bool_)

    # var 상태
    ve10 = NAN; ve25 = NAN; ve60 = NAN
    a10 = 2.0 / (10 + 1); a25 = 2.0 / (25 + 1); a60 = 2.0 / (60 + 1)
    rma_prev = NAN
    swP = np.empty(n + 1, np.float64)   # var array<float> swP — 앞에서 shift 하므로 머리 포인터로 흉내
    swB = np.empty(n + 1, np.int64)
    head = 0
    tail = 0
    lastA = -1; hasLastA = False         # var int lastA = na
    lastB = -1; hasLastB = False

    tol = tolPct / 100                   # L98

    for bi in range(n):
        # L72-80 f_ema: 첫 값은 src, 그 뒤 a*src + (1-a)*e
        ve10 = c[bi] if math.isnan(ve10) else a10 * c[bi] + (1 - a10) * ve10
        ve25 = c[bi] if math.isnan(ve25) else a25 * c[bi] + (1 - a25) * ve25
        ve60 = c[bi] if math.isnan(ve60) else a60 * c[bi] + (1 - a60) * ve60
        e60[bi] = ve60
        # L81
        stack[bi] = ve10 < ve25 and ve25 < ve60
        # L82 ta.sma(stack ? 1 : 0, 30)
        s30 = NAN
        if bi >= 29:
            ssum = 0.0
            for j in range(bi - 29, bi + 1):
                ssum += 1.0 if stack[j] else 0.0
            s30 = ssum / 30
        # L83
        e60_30 = e60[bi - 30] if bi >= 30 else NAN
        persist[bi] = s30 >= 0.8 and e60[bi] < e60_30
        # L84 ta.atr(14)
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
        # L85 vAvg = ta.sma(volume, 20)[1]
        if bi >= 19:
            ssum = 0.0
            for j in range(bi - 19, bi + 1):
                ssum += v[j]
            smaV[bi] = ssum / 20
        vAvg = smaV[bi - 1] if bi >= 1 else NAN
        # L86 volR = vAvg > 0 ? volume / vAvg : na
        volR[bi] = v[bi] / vAvg if vAvg > 0 else NAN

        # L89-96 최근 120분 최고가, 직전 30분 봉 수
        hh120 = h[bi]
        nIn30 = 1
        for i in range(1, 481):
            if bi - i < 0 or t[bi - i] <= t[bi] - 120 * 60000:
                break
            hh120 = max(hh120, h[bi - i])
            if t[bi - i] > t[bi] - 30 * 60000:
                nIn30 += 1

        # L99-103
        sNew = _lowest_at(l, bi - 1, nBars)          # ta.lowest(low, nBars)[1]
        sFirst = _lowest_at(l, bi - 2, nBars - 1)    # ta.lowest(low, nBars - 1)[2]
        S[bi] = sNew if brkMd == B_NEW else sFirst
        thr[bi] = S[bi] * (1 - tol)
        low1 = l[bi - 1] if bi >= 1 else NAN
        if brkMd == B_NEW:
            brk[bi] = l[bi] < thr[bi]
        else:
            brk[bi] = low1 >= thr[bi] and l[bi] < thr[bi] and l[bi] < low1
        # L104-107
        rng = h[bi] - l[bi]
        single = o[bi] == h[bi] and h[bi] == l[bi] and l[bi] == c[bi]
        sane[bi] = v[bi] * c[bi] >= minNot and rng >= minRng / 100 * c[bi] and not single
        densOk = nIn30 >= minDens
        # L108-111
        dd = (hh120 - l[bi]) / hh120
        if trendMd == T_OFF:
            trendNow[bi] = True
        elif trendMd == T_NOW:
            trendNow[bi] = stack[bi]
        else:
            trendNow[bi] = persist[bi]
        dropOk[bi] = dropPct <= 0 or dd >= dropPct / 100
        volOk[bi] = volMin <= 0 or volR[bi] >= volMin
        # L112-113 loR = ta.lowest(low, pSw); isSw = low[pSw] < loR[pSw+1] and low[pSw] <= loR
        lowp = l[bi - pSw] if bi - pSw >= 0 else NAN
        loR_prev = _lowest_at(l, bi - pSw - 1, pSw)
        loR_now = _lowest_at(l, bi, pSw)
        isSw = lowp < loR_prev and lowp <= loR_now
        # L114-117
        night = hrNY[bi] >= 20 or hrNY[bi] < 4
        trdOk = (not onlyTrd) or hrKST[bi] >= 8

        # L120-157 확정 스윙 저점 목록 (과거 봉은 항상 confirmed)
        while (tail - head) > 0 and swB[head] < bi - nBars:
            head += 1
        if brk[bi]:
            nSw = tail - head
            pts = np.empty(nSw + 1, np.float64)
            npts = 0
            sIn = False
            if nSw > 0:
                for i in range(nSw):
                    if swB[head + i] <= bi - 1 - pSw:
                        px = swP[head + i]
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
            anchor = -1.0
            for i in range(npts):
                x = ps[i]
                if x <= S[bi] * (1 + 0.003):
                    tc += 1
                if x <= S[bi] * (1 + 0.01):
                    if lc == 0 or x > anchor * (1 + 0.0015):
                        lc += 1
                        anchor = x
            nTouch[bi] = tc
            nLevel[bi] = lc
        if bi >= 2 * pSw and isSw:
            swP[tail] = l[bi - pSw]
            swB[tail] = bi - pSw
            tail += 1

        # L159-160
        kCntA = nTouch[bi] if kModeA == M_TOUCH else nLevel[bi]
        kCntB[bi] = nTouch[bi] if kModeB == M_TOUCH else nLevel[bi]

        # L163-166 ① 꼬리 반전
        if rng > 0:
            rec = (min(o[bi], c[bi]) - l[bi]) / rng if recMd == R_WICK else (c[bi] - l[bi]) / rng
        else:
            rec = -1.0
        tn1 = trendNow[bi - 1] if bi >= 1 else False
        atr1 = atr14[bi - 1] if bi >= 1 else NAN
        sigA[bi] = (showA and brk[bi] and kCntA >= kMinA and tn1 and dropOk[bi] and volOk[bi] and sane[bi]
                    and densOk and o[bi] >= thr[bi] and rec >= recMin and (rngMin <= 0 or rng >= rngMin * atr1)
                    and ((not needAb) or c[bi] >= S[bi]) and not (noNight and night) and trdOk)

        # L169-172 ② 장악 양봉
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
                        and ((not needBl) or c[bi - 1] < thr[bi - 1]) and tn2 and dropOk[bi - 1] and volOk[bi - 1]
                        and sane[bi - 1] and sane[bi] and densOk
                        and c[bi] > o[bi] and c[bi] >= engLvl and not (noNight and night) and trdOk)

        # L175-182 30분 안 같은 신호 숨김(표시 거르기)
        kA = sigA[bi] and ((not dedup30) or (not hasLastA) or t[bi] - lastA >= 30 * 60000)
        kB = sigB[bi] and ((not dedup30) or (not hasLastB) or t[bi] - lastB >= 30 * 60000)
        if kA:
            lastA = t[bi]; hasLastA = True
        if kB:
            lastB = t[bi]; hasLastB = True
        keepA[bi] = kA
        keepB[bi] = kB
    return keepA, keepB, S, nTouch, nLevel, brk
