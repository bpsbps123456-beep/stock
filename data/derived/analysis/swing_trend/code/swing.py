"""중단기 추세(1주~1달) 연구 공통 계산(BRIEF.md).

- 판정 상태 st[d]는 d일 종가까지의 정보로 정한다(1 보유, 0 현금). 체결은 d+1일 시가다.
- 날 j의 수익 = pos[j] × (O[j+1]/O[j] − 1), pos[j] = st[j−1]. 구간 마지막 날은 C/O로 끝낸다.
- 지그재그: 극점에서 θ(%) 또는 k × ATR20만큼 되돌면 반전 확정. 확정 시점이 실시간 판정, 극점이 사후 반전점이다.
"""
from __future__ import annotations

import pathlib
from datetime import date

import numpy as np
import pandas as pd
from numba import njit

HERE = pathlib.Path(__file__).resolve().parent.parent
PERIODS = {
    "P1": (date(2026, 1, 2), date(2026, 5, 29)),
    "P2": (date(2026, 6, 1), date(2026, 10, 22)),
    "Y25": (date(2025, 1, 2), date(2025, 12, 31)),
    "Y24": (date(2024, 1, 2), date(2024, 12, 31)),
    "Y23": (date(2023, 1, 3), date(2023, 12, 29)),
}
DETECTORS = (
    [("ZZP", dict(th=t)) for t in (0.03, 0.05, 0.07, 0.10, 0.15)]
    + [("ZZA", dict(k=k)) for k in (1.5, 2.0, 3.0, 4.0)]
    + [("CR", dict(n=n, b=b, s=s)) for n in (5, 10, 15, 20) for b, s in ((0.5, 0.5), (0.6, 0.4))]
    + [("EMA", dict(f=f, s=s)) for f, s in ((5, 20), (10, 30), (10, 50), (20, 50))]
    + [("DON", dict(n=n, m=m)) for n, m in ((20, 10), (10, 5), (55, 20))]
    + [("SLP", dict(n=n)) for n in (10, 20, 50)]
    + [("Z5", {})]
)
DEFAULT, DEFAULT2 = "ZZA_k3.0", "CR_n10_b0.5_s0.5"


def dname(kind, prm):
    return kind + "".join(f"_{k}{v}" for k, v in prm.items())


def load() -> dict:
    d = pd.read_parquet(HERE / "cache" / "daily.parquet")
    return {s: g.sort_values("date").reset_index(drop=True) for s, g in d.groupby("sym")}


@njit(cache=True)
def atr(h, l, c, n):
    m = len(c)
    out = np.full(m, np.nan)
    s = 0.0
    for i in range(m):
        tr = h[i] - l[i] if i == 0 else max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        if i < n:
            s += tr
            if i == n - 1:
                s /= n
                out[i] = s
        else:
            s = (s * (n - 1) + tr) / n
            out[i] = s
    return out


@njit(cache=True)
def zigzag(c, th, absval):
    """실시간 지그재그 확인 상태와 사후 반전점.
    absval이 비어 있지 않으면 문턱 = absval[i](가격 단위), 아니면 비율 th.
    반환: st(상태), piv_idx(반전점 날), piv_conf(확정 날), piv_type(+1 저점→상승 확정, −1 고점→하락 확정)."""
    n = len(c)
    st = np.zeros(n, np.int64)
    piv_idx = np.full(n, -1, np.int64)
    piv_conf = np.full(n, -1, np.int64)
    piv_type = np.zeros(n, np.int64)
    npv = 0
    state = 0
    ext = c[0]
    ext_i = 0
    use_abs = len(absval) == n
    for i in range(n):
        if use_abs and np.isnan(absval[i]):
            if state == 0 and c[i] < ext:
                ext = c[i]; ext_i = i
            elif state == 1 and c[i] > ext:
                ext = c[i]; ext_i = i
            st[i] = state
            continue
        if state == 0:
            if c[i] < ext:
                ext = c[i]; ext_i = i
            thr_up = ext + absval[i] if use_abs else ext * (1 + th)
            if c[i] >= thr_up:
                piv_idx[npv] = ext_i; piv_conf[npv] = i; piv_type[npv] = 1; npv += 1
                state = 1; ext = c[i]; ext_i = i
        else:
            if c[i] > ext:
                ext = c[i]; ext_i = i
            thr_dn = ext - absval[i] if use_abs else ext * (1 - th)
            if c[i] <= thr_dn:
                piv_idx[npv] = ext_i; piv_conf[npv] = i; piv_type[npv] = -1; npv += 1
                state = 0; ext = c[i]; ext_i = i
        st[i] = state
    return st, piv_idx[:npv], piv_conf[:npv], piv_type[:npv]


@njit(cache=True)
def ratio_state(o, c, n, b, s):
    m = len(c)
    st = np.zeros(m, np.int64)
    state = 0
    for i in range(m):
        if i >= n - 1:
            up = 0.0; dn = 0.0
            for j in range(i - n + 1, i + 1):
                if c[j] > o[j]:
                    up += 1
                elif c[j] < o[j]:
                    dn += 1
            p = up / (up + dn) if up + dn > 0 else 0.5
            if state == 0 and p > b:
                state = 1
            elif state == 1 and p < s:
                state = 0
        st[i] = state
    return st


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


@njit(cache=True)
def donchian_state(h, l, c, n, m):
    k = len(c)
    st = np.zeros(k, np.int64)
    state = 0
    for i in range(k):
        if i >= max(n, m):
            hi = h[i - n]
            for j in range(i - n, i):
                hi = max(hi, h[j])
            lo = l[i - m]
            for j in range(i - m, i):
                lo = min(lo, l[j])
            if state == 0 and c[i] > hi:
                state = 1
            elif state == 1 and c[i] < lo:
                state = 0
        st[i] = state
    return st


def zone5(c, n=20, th=0.5):
    c = pd.Series(c, dtype=float)
    lc = np.log(c)
    sd = lc.diff().rolling(n).std(ddof=1)
    k = np.sqrt((n - 1.0) * (2.0 * n - 1.0) / (6.0 * n))
    z = (lc - lc.rolling(n).mean()) / (sd * k)
    dl = c.diff()
    up = dl.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-dl.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    zn = np.where(rsi >= 70, 2, np.where(rsi <= 30, -2, np.where(z >= th, 1, np.where(z <= -th, -1, 0)))).astype(float)
    zn[(z.isna() | rsi.isna()).values] = np.nan
    return zn


def states(g: pd.DataFrame) -> dict:
    """판정 28개의 상태(종가 기준)."""
    O, H, L, C = (g[k].values.astype(np.float64) for k in ("O", "H", "L", "C"))
    a20 = atr(H, L, C, 20)
    empty = np.zeros(0)
    out = {}
    for kind, prm in DETECTORS:
        nm = dname(kind, prm)
        if kind == "ZZP":
            st = zigzag(C, prm["th"], empty)[0]
        elif kind == "ZZA":
            st = zigzag(C, 0.0, prm["k"] * a20)[0]
        elif kind == "CR":
            st = ratio_state(O, C, prm["n"], prm["b"], prm["s"])
        elif kind == "EMA":
            st = (ema(C, prm["f"]) > ema(C, prm["s"])).astype(np.int64)
            st[: prm["s"]] = 0
        elif kind == "DON":
            st = donchian_state(H, L, C, prm["n"], prm["m"])
        elif kind == "SLP":
            e = ema(C, prm["n"])
            st = ((C > e) & (np.r_[False, e[1:] > e[:-1]])).astype(np.int64)
            st[: prm["n"]] = 0
        elif kind == "Z5":
            zn = zone5(C)
            st = np.zeros(len(C), np.int64)
            cur = 0
            for i, z in enumerate(zn):
                if not np.isnan(z):
                    cur = 1 if z >= 1 else 0 if z <= -1 else cur
                st[i] = cur
        out[nm] = st
    return out


def period_index(g: pd.DataFrame, per: str):
    a, b = PERIODS[per]
    d = g.date.values
    idx = np.flatnonzero((d >= a) & (d <= b))
    return idx


@njit(cache=True)
def sim_period(o, c, st, j0, j1, cost):
    """j0..j1(포함) 구간. pos[j] = st[j−1]. 반환: 수익, 최대 낙폭, 평균 보유, 진입 수, 날별 자산."""
    n = j1 - j0 + 1
    eq = np.empty(n)
    v = 1.0; pk = 1.0; mdd = 0.0; hold = 0.0; ent = 0
    prev = st[j0 - 1] if j0 > 0 else 0
    for t in range(n):
        j = j0 + t
        pos = st[j - 1] if j > 0 else 0
        r = (o[j + 1] / o[j] - 1.0) if (t < n - 1) else (c[j] / o[j] - 1.0)
        if pos != prev and t > 0:
            v *= 1.0 - cost
            if pos == 1:
                ent += 1
        v *= 1.0 + pos * r
        prev = pos
        hold += pos
        pk = max(pk, v)
        mdd = min(mdd, v / pk - 1.0)
        eq[t] = v
    return v - 1.0, mdd, hold / n, ent, eq


def static_bh(o, c, j0, j1, x):
    px = np.r_[o[j0:j1 + 1], c[j1]]
    eq = 1 + x * (px / px[0] - 1)
    bh = px / px[0]
    mdd = lambda e: float((e / np.maximum.accumulate(e) - 1).min())
    return float(eq[-1] - 1), mdd(eq), float(bh[-1] - 1), mdd(bh)


def shift_state(st, j0, j1, k):
    """구간 [j0, j1] 안에서 상태를 거래일 k만큼 원형 이동(구간 밖은 그대로). pos가 st[j−1]이므로 j0−1부터 민다."""
    a = max(j0 - 1, 0)
    seg = st[a:j1 + 1]
    out = st.copy()
    out[a:j1 + 1] = np.roll(seg, k)
    return out
