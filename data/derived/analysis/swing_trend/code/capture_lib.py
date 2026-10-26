"""capture 갈래 계산(질문 2: 반전을 실시간으로 확인한 뒤에도 추세가 이어져 늦음을 갚는가).

- 확정 사건: 실시간 지그재그가 반전을 확정한 날(cf) 종가에서 보면
  이어진 몫 = 그 구간 사후 극점까지, 되돌림 = 극점에서 반대 확정까지, 먹은 몫 = 반대 확정 종가까지.
  집행판(captx)은 확정 다음 날 시가 → 반대 확정 다음 날 시가.
- 무작위 가격: 창(예열 60일 + 구간) 안에서 봉(전일 종가 대비 O·H·L·C 비율)을 예열·구간 따로 섞는다.
  구간의 일 수익 묶음과 총수익은 그대로, 순서(추세 지속·되돌림)만 지운다.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from swing import atr, zigzag

ZZ_CFG = [("ZZP_th0.03", 0.03, 0.0), ("ZZP_th0.05", 0.05, 0.0), ("ZZP_th0.07", 0.07, 0.0),
          ("ZZP_th0.1", 0.10, 0.0), ("ZZP_th0.15", 0.15, 0.0),
          ("ZZA_k1.5", 0.0, 1.5), ("ZZA_k2.0", 0.0, 2.0), ("ZZA_k3.0", 0.0, 3.0), ("ZZA_k4.0", 0.0, 4.0)]
ZZ_TH = np.array([c[1] for c in ZZ_CFG])
ZZ_K = np.array([c[2] for c in ZZ_CFG])


@njit(cache=True)
def events(O, C, piv_idx, piv_conf, piv_type, lo, hi, end, cfg, rep, buf, nb):
    """확정일 cf가 [lo, min(hi, end−1)]인 사건을 buf에 쌓는다. end 뒤 정보는 쓰지 않는다(그때까지 반대 확정이 없으면 end 종가로 자른다).
    buf 열: cfg, rep, typ, cf, capt, cont, retr, captx, hold, cens, gone(극점 → 확정 종가, 새 방향 부호)"""
    m = len(piv_conf)
    for i in range(m):
        cf = piv_conf[i]
        if cf < lo or cf > hi or cf >= end:
            continue
        t = piv_type[i]
        if i + 1 < m and piv_conf[i + 1] <= end:
            ex = piv_idx[i + 1]
            op = piv_conf[i + 1]
            cens = 0
        else:
            ex = cf
            for j in range(cf, end + 1):
                if t == 1 and C[j] > C[ex]:
                    ex = j
                elif t == -1 and C[j] < C[ex]:
                    ex = j
            op = end
            cens = 1
        px_in = O[cf + 1]
        px_out = O[op + 1] if op < end else C[end]
        if t == 1:
            cont = C[ex] / C[cf] - 1.0
            retr = 1.0 - C[op] / C[ex]
            capt = C[op] / C[cf] - 1.0
            captx = px_out / px_in - 1.0
        else:
            cont = 1.0 - C[ex] / C[cf]
            retr = C[op] / C[ex] - 1.0
            capt = 1.0 - C[op] / C[cf]
            captx = 1.0 - px_out / px_in
        buf[nb, 0] = cfg; buf[nb, 1] = rep; buf[nb, 2] = t; buf[nb, 3] = cf
        buf[nb, 4] = capt; buf[nb, 5] = cont; buf[nb, 6] = retr; buf[nb, 7] = captx
        buf[nb, 8] = op - cf; buf[nb, 9] = cens
        pv = piv_idx[i]
        buf[nb, 10] = (C[cf] / C[pv] - 1.0) if t == 1 else (1.0 - C[cf] / C[pv])
        nb += 1
    return nb


@njit(cache=True)
def run_block(O, H, L, C, p0, lo_eval, hi_eval, end_eval, nrep, seed, shuffle, ths, ks):
    """창 하나(O..C, 0이 창 첫날)에 대해 지그재그 9개 설정의 확정 사건.
    p0: 구간 첫날(창 좌표), lo_eval..hi_eval: 사건 집계 범위, end_eval: 절단 끝.
    shuffle=True면 rep마다 봉을 예열[1, p0)·구간[p0, n) 따로 섞는다. 같은 seed·같은 n이면 종목이 달라도 같은 순서로 섞인다(종목 간 같은 날 묶음 유지)."""
    n = len(C)
    ro = np.empty(n); rh = np.empty(n); rl = np.empty(n); rc = np.empty(n)
    for i in range(1, n):
        ro[i] = O[i] / C[i - 1]; rh[i] = H[i] / C[i - 1]; rl[i] = L[i] / C[i - 1]; rc[i] = C[i] / C[i - 1]
    ncfg = len(ths)
    buf = np.empty((nrep * ncfg * n, 11))
    nb = 0
    np.random.seed(seed)
    perm = np.arange(n)
    o = O.copy(); h = H.copy(); l = L.copy(); c = C.copy()
    empty = np.zeros(0)
    for r in range(nrep):
        if shuffle:
            perm = np.arange(n)
            w = perm[1:p0].copy(); np.random.shuffle(w); perm[1:p0] = w
            q = perm[p0:].copy(); np.random.shuffle(q); perm[p0:] = q
            c[0] = C[0]; o[0] = O[0]; h[0] = H[0]; l[0] = L[0]
            for i in range(1, n):
                k = perm[i]
                o[i] = c[i - 1] * ro[k]; h[i] = c[i - 1] * rh[k]; l[i] = c[i - 1] * rl[k]; c[i] = c[i - 1] * rc[k]
        a20 = atr(h, l, c, 20)
        for g in range(ncfg):
            if ths[g] > 0:
                st, pi, pc, pt = zigzag(c, ths[g], empty)
            else:
                st, pi, pc, pt = zigzag(c, 0.0, ks[g] * a20)
            nb = events(o, c, pi, pc, pt, lo_eval, hi_eval, end_eval, g, r, buf, nb)
    return buf[:nb]


# ---------------- 판정 28개 대 사후 진실 구간 ----------------

def truth_segments(C, piv_idx, piv_conf, piv_type):
    """사후 진실 구간 목록: (시작 극점, 끝 극점, 방향 +1 상승/−1 하락, 꼬리 여부).
    마지막 확정 극점 뒤는 그때까지의 극값으로 끝나는 꼬리 구간(끝 극점 미확정)."""
    segs = []
    m = len(piv_idx)
    for i in range(m - 1):
        segs.append((int(piv_idx[i]), int(piv_idx[i + 1]), int(piv_type[i]), False))
    if m:
        p, cf, t = int(piv_idx[-1]), int(piv_conf[-1]), int(piv_type[-1])
        tail = C[cf:]
        e = cf + int(np.argmax(tail) if t == 1 else np.argmin(tail))
        if e > p:
            segs.append((p, e, t, True))
    return segs


def runs(pos):
    """보유 연속 구간 (시작 날, 끝 날) 목록. pos[j]=1이면 j일 시가→j+1일 시가 보유."""
    d = np.diff(np.r_[0, pos, 0])
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1) - 1
    return s, e


@njit(cache=True)
def curve_block(O, H, L, C, p0, nrep, seed, shuffle, th, k, maxd):
    """진실 지그재그(th 비율 또는 k × ATR20) 상승 구간(L+1 ≥ p0, H+1 ≤ n−1)의 예지자 지연 곡선 합.
    반환 acc[rep, :]: 0 구간 수, 1 Σ예지자, 2 갑작 시작 수, 3 갑작 끝 수,
    4.. Σ진입 지연 d(0..maxd), 그다음 Σ청산 지연 d, 그다음 갑작 시작 구간의 Σ예지자·Σ진입 d, 갑작 끝 구간의 Σ예지자·Σ청산 d."""
    n = len(C)
    nd = maxd + 1
    ncol = 4 + 2 * nd + 2 * (1 + nd)
    acc = np.zeros((nrep, ncol))
    ro = np.empty(n); rh = np.empty(n); rl = np.empty(n); rc = np.empty(n)
    for i in range(1, n):
        ro[i] = O[i] / C[i - 1]; rh[i] = H[i] / C[i - 1]; rl[i] = L[i] / C[i - 1]; rc[i] = C[i] / C[i - 1]
    np.random.seed(seed)
    o = O.copy(); h = H.copy(); l = L.copy(); c = C.copy()
    empty = np.zeros(0)
    for r in range(nrep):
        if shuffle:
            perm = np.arange(n)
            w = perm[1:p0].copy(); np.random.shuffle(w); perm[1:p0] = w
            q = perm[p0:].copy(); np.random.shuffle(q); perm[p0:] = q
            for i in range(1, n):
                kk = perm[i]
                o[i] = c[i - 1] * ro[kk]; h[i] = c[i - 1] * rh[kk]; l[i] = c[i - 1] * rl[kk]; c[i] = c[i - 1] * rc[kk]
        a20 = atr(h, l, c, 20)
        if th > 0:
            st, pi, pc, pt = zigzag(c, th, empty)
        else:
            st, pi, pc, pt = zigzag(c, 0.0, k * a20)
        m = len(pi)
        for i in range(m - 1):
            if pt[i] != 1:
                continue
            Lp = pi[i]; Hp = pi[i + 1]
            if Lp + 1 < p0 or Hp + 1 > n - 1:
                continue
            nxt = pi[i + 2] + 1 if i + 2 < m else n - 1
            if nxt > n - 1:
                nxt = n - 1
            orc = np.log(o[Hp + 1] / o[Lp + 1])
            a = a20[Lp]
            abs_ = (not np.isnan(a)) and ((c[Lp + 1] - c[Lp] >= 2 * a) or (o[Lp + 1] - c[Lp] >= a))
            b = a20[Hp]
            abe = (not np.isnan(b)) and ((c[Hp] - c[Hp + 1] >= 2 * b) or (c[Hp] - o[Hp + 1] >= b))
            acc[r, 0] += 1; acc[r, 1] += orc
            if abs_:
                acc[r, 2] += 1
                acc[r, 4 + 2 * nd] += orc
            if abe:
                acc[r, 3] += 1
                acc[r, 4 + 2 * nd + 1 + nd] += orc
            for d in range(nd):
                ein = Lp + 1 + d
                if ein > Hp + 1:
                    ein = Hp + 1
                ev = np.log(o[Hp + 1] / o[ein])
                xo = Hp + 1 + d
                if xo > nxt:
                    xo = nxt
                xv = np.log(o[xo] / o[Hp + 1])
                acc[r, 4 + d] += ev
                acc[r, 4 + nd + d] += xv
                if abs_:
                    acc[r, 4 + 2 * nd + 1 + d] += ev
                if abe:
                    acc[r, 4 + 2 * nd + 1 + nd + 1 + d] += xv
    return acc
