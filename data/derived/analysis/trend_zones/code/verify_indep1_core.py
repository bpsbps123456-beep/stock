"""검증자 #1 공용 계산(2026-10-11, 연구용). zones.py를 import하지 않고 그 docstring 정의만으로 다시 구현한다.

구조를 일부러 다르게 했다: 종목별 1차원 numpy 재귀·창 계산 → 날짜×종목 배열로 되돌림.
seed 옵션: "first"(첫 값 시작, pandas ewm adjust=False와 같은 정의) / "sma"(Wilder·TradingView식 SMA 시작) — 민감도용.
"""
import os
from datetime import date, timedelta
from itertools import product

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 5))
SRC = os.path.join(ROOT, "data/derived/analysis/rsi_breakout/cache/1d_long.parquet")
SOXX_SRC = os.path.join(ROOT, "data/derived/analysis/stop_research/cache/ohlc.parquet")
RES = os.path.abspath(os.path.join(HERE, "../results"))
MINB = 130
PER = {"앞": ("2026-01-02", "2026-05-29", True), "뒤": ("2026-06-01", "2026-10-09", False),
       "2025": ("2025-01-02", "2025-12-31", False), "2024": ("2024-01-02", "2024-12-31", False)}
TK = ("T1", "T2", "T3")
ZN = (2, 1, 0, -1, -2)


# ------------------------------------------------------------------ 자료
def load(upto=None, drop=()):
    x = pd.read_parquet(SRC)
    x = x[(x.sym != "QQQ") & (~x.sym.isin(drop))].copy()
    x["d"] = x["ts"].dt.strftime("%Y-%m-%d")
    if upto:
        x = x[x.d <= upto]
    piv = {k: x.pivot(index="d", columns="sym", values=k).sort_index() for k in ("Open", "High", "Low", "Close")}
    syms = sorted(piv["Close"].columns)
    dates = list(piv["Close"].index)
    P = {k[0]: v.reindex(columns=syms).to_numpy(float) for k, v in piv.items()}
    return dates, syms, P


# ------------------------------------------------------------------ 1차원 도구
def rec(x, alpha, minp, seed="first"):
    """지수 재귀. 앞쪽 NaN은 건너뛰고 첫 유한값부터 시작. minp개 관측 전은 NaN."""
    n = len(x)
    y = np.full(n, np.nan)
    fin = np.flatnonzero(np.isfinite(x))
    if len(fin) == 0:
        return y
    s = fin[0]
    if seed == "first":
        v = x[s]
        y[s] = v
        for i in range(s + 1, n):
            v = alpha * x[i] + (1 - alpha) * v
            y[i] = v
        y[: s + minp - 1] = np.nan
    else:
        if n - s < minp:
            return y
        v = float(np.mean(x[s:s + minp]))
        y[s + minp - 1] = v
        for i in range(s + minp, n):
            v = alpha * x[i] + (1 - alpha) * v
            y[i] = v
    return y


def roll(x, w, fn, **kw):
    y = np.full(len(x), np.nan)
    if len(x) >= w:
        y[w - 1:] = fn(swv(x, w), axis=1, **kw)   # 창 안에 NaN이 있으면 NaN (min_periods=w)
    return y


def lag(x, k):
    y = np.full(len(x), np.nan)
    if k < len(x):
        y[k:] = x[:-k]
    return y


# ------------------------------------------------------------------ 지표(종목별)
EMAS = (10, 20, 30, 35, 50, 60, 150)


def sym_ind(o, h, l, c, seed="first", ddof=1):
    n = len(c)
    out = {}
    out["valid"] = np.arange(1, n + 1) >= MINB
    lr = np.r_[np.nan, np.diff(np.log(c))]
    em = {L: rec(c, 2 / (L + 1), min(L, MINB), "first" if seed in ("first", "rsi_sma") else seed) for L in EMAS}
    pc = lag(c, 1)
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    rs = "sma" if seed in ("sma", "rsi_sma") else "first"
    atr = rec(tr, 1 / 14, 14, rs)
    dc = np.r_[np.nan, np.diff(c)]
    au = rec(np.where(np.isfinite(dc), np.maximum(dc, 0), np.nan), 1 / 14, 14, rs)
    ad = rec(np.where(np.isfinite(dc), np.maximum(-dc, 0), np.nan), 1 / 14, 14, rs)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = np.where(ad > 0, 100 - 100 / (1 + au / ad), np.where(np.isfinite(ad), 100.0, np.nan))
        out["atr"] = atr
        out["rsi"] = rsi
        for L in (10, 20, 35):
            dev = c - em[L]
            for w in (20, 25, 30, 60):
                out[("z", L, f"rms{w}")] = dev / np.sqrt(roll(dev ** 2, w, np.mean))
            out[("z", L, "atr14")] = dev / atr
        for L, k in ((10, 3), (20, 5), (50, 10)):
            out[("slope", (L, k))] = (em[L] - lag(em[L], k)) / (k * atr)
        for a, b in ((10, 30), (20, 60), (50, 150)):
            out[("cross", (a, b))] = (em[a] - em[b]) / atr
        for N in (20, 60, 120):
            sd = roll(lr, N, np.std, ddof=ddof)
            out[("mz", N)] = np.log(c / lag(c, N)) / (sd * np.sqrt(N))
        for N in (20, 55, 120):
            hh, ll = roll(h, N, np.max), roll(l, N, np.min)
            out[("don", N)] = 2 * (c - ll) / (hh - ll) - 1
        out["xdev"] = out[("z", 20, "rms60")]
        out["xrsi"] = rsi
        sd60 = roll(lr, 60, np.std, ddof=ddof)
        out["xret"] = np.log(c / lag(c, 10)) / (sd60 * np.sqrt(10))
    return out


class Panel:
    def __init__(self, dates, syms, P, seed="first", ddof=1):
        self.dates, self.syms = dates, syms
        self.O, self.H, self.L, self.C = P["O"], P["H"], P["L"], P["C"]
        T, N = self.C.shape
        self.f = {}
        self.ind = {}
        start = []
        for j in range(N):
            fin = np.flatnonzero(np.isfinite(self.C[:, j]))
            s, e = fin[0], fin[-1]
            assert len(fin) == e - s + 1, "중간 결측"
            start.append((s, e))
            d = sym_ind(self.O[s:e + 1, j], self.H[s:e + 1, j], self.L[s:e + 1, j], self.C[s:e + 1, j], seed, ddof)
            for k, v in d.items():
                if k not in self.ind:
                    self.ind[k] = np.full((T, N), np.nan) if k != "valid" else np.zeros((T, N), bool)
                self.ind[k][s:e + 1, j] = v
        self.start = start
        self.valid = self.ind["valid"] & np.isfinite(self.C) & np.isfinite(self.O)

    def fwd(self, h):
        if h not in self.f:
            T, N = self.O.shape
            f = np.full((T, N), np.nan)
            for j, (s, e) in enumerate(self.start):
                o = self.O[s:e + 1, j]
                n = len(o)
                if n > 1 + h:
                    f[s:s + n - 1 - h, j] = o[1 + h:] / o[1:n - h] - 1
            self.f[h] = np.where(self.valid, f, np.nan)
        return self.f[h]


# ------------------------------------------------------------------ 격자·단계
def grid():
    G = []
    for L in (10, 20, 35):
        for w in ("rms20", "rms30", "rms60", "atr14"):
            for a in (0.5, 0.75, 1.0):
                for b in (1.5, 2.0, 2.5):
                    G.append(("dev", L, w, a, b))
    T = [("slope", (10, 3)), ("slope", (20, 5)), ("slope", (50, 10)), ("cross", (10, 30)), ("cross", (20, 60)),
         ("cross", (50, 150)), ("mz", 20), ("mz", 60), ("mz", 120), ("don", 20), ("don", 55), ("don", 120)]
    TAU = {"slope": (0.05, 0.1, 0.2), "cross": (0.5, 1.0, 2.0), "mz": (0.5, 1.0, 1.5), "don": (0.2, 0.4, 0.6)}
    X = [("xdev", v) for v in (1.5, 2.0, 2.5)] + [("xrsi", v) for v in (70, 75, 80)] + [("xret", v) for v in (1.5, 2.0, 2.5)]
    for (tk, tp) in T:
        for tau in TAU[tk]:
            for xk, xv in X:
                G.append(("tx", tk, tp, tau, xk, xv))
    return G


BASE = ("dev", 10, "rms25", 0.75, 1.75)
WIN = ("tx", "mz", 60, 0.5, "xrsi", 70)


def key(c):
    """zones.json cfg(dict)이나 내 튜플을 같은 문자열 키로."""
    if isinstance(c, dict):
        if c["fam"] == "dev":
            c = ("dev", c["L"], c["w"], c["a"], c["b"])
        else:
            tp = tuple(c["tp"]) if isinstance(c["tp"], list) else c["tp"]
            c = ("tx", c["t"], tp, c["tau"], c["x"], c["xv"])
    if c[0] == "dev":
        return f"dev|{c[1]}|{c[2]}|{float(c[3])}|{float(c[4])}"
    return f"tx|{c[1]}|{c[2]}|{float(c[3])}|{c[4]}|{float(c[5])}"


def zones(P: Panel, c):
    if c[0] == "dev":
        _, L, w, a, b = c
        z = P.ind[("z", L, w)]
        Z = np.full(z.shape, -9, np.int8)
        fin = np.isfinite(z)
        Z[fin] = -2
        Z[fin & (z > -b)] = -1
        Z[fin & (z > -a)] = 0
        Z[fin & (z >= a)] = 1
        Z[fin & (z >= b)] = 2
    else:
        _, tk, tp, tau, xk, xv = c
        t = P.ind[(tk, tp)]
        x = P.ind[xk]
        hi, lo = (xv, 100 - xv) if xk == "xrsi" else (xv, -xv)
        fin = np.isfinite(t) & np.isfinite(x)
        Z = np.full(t.shape, -9, np.int8)
        Z[fin] = 0
        Z[fin & (t <= -tau)] = -1
        Z[fin & (t >= tau)] = 1
        Z[fin & (x <= lo)] = -2
        Z[fin & (x >= hi)] = 2          # 우선순위: 과열 위 > 과열 아래 > 추세(마지막에 덮어쓴 것이 이김)
    Z[~P.valid] = -9
    return Z


# ------------------------------------------------------------------ 평가
def rows_of(dates, p, h):
    a, b, purge = PER[p]
    idx = [i for i, d in enumerate(dates) if a <= d <= b]
    if purge:
        last = idx[-1]
        idx = [i for i in idx if i + 1 + h <= last]
    return np.array(idx)


def excess(f):
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(f).sum(1, keepdims=True)
        mu = np.where(cnt > 0, np.nansum(f, 1, keepdims=True) / np.maximum(cnt, 1), np.nan)
    return f - mu


def zone_means(Z, EX, rows):
    z, e = Z[rows], EX[rows]
    have = np.isfinite(e) & (z != -9)
    m, share = {}, {}
    tot = have.sum()
    for k in ZN:
        M = have & (z == k)
        n = M.sum(1)
        m[k] = np.where(n > 0, np.where(M, e, 0.0).sum(1) / np.maximum(n, 1), np.nan)
        share[k] = M.sum() / tot if tot else np.nan
    return m, share


def score(Z, EX, rows):
    m, share = zone_means(Z, EX, rows)
    ser = {"T1": m[1] - m[-1], "T2": m[-2] - m[-1], "T3": m[1] - m[2]}
    T = {k: (float(np.nanmean(v)) if np.isfinite(v).any() else np.nan) for k, v in ser.items()}
    return {"share": {k: float(v) for k, v in share.items()}, "T": T, "S": float(sum(T.values())), "ser": ser,
            "zm": {k: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for k, v in m.items()}}


def score_pandas(Z, EX, rows, dates):
    """같은 값을 긴 표 groupby로 다시 낸다(교차 확인용)."""
    z, e = Z[rows], EX[rows]
    di = np.repeat(np.array(rows), z.shape[1])
    df = pd.DataFrame({"d": di, "z": z.ravel(), "e": e.ravel()})
    df = df[(df.z != -9) & np.isfinite(df.e)]
    g = df.groupby(["d", "z"]).e.mean().unstack()
    T = {"T1": (g[1] - g[-1]).mean(), "T2": (g[-2] - g[-1]).mean(), "T3": (g[1] - g[2]).mean()}
    sh = df.z.value_counts(normalize=True).to_dict()
    return {"T": {k: float(v) for k, v in T.items()}, "S": float(sum(T.values())), "share": {int(k): float(v) for k, v in sh.items()}}


def eligible(share):
    return 0.02 <= share[2] <= 0.20 and 0.02 <= share[-2] <= 0.20 and min(share[1], share[0], share[-1]) >= 0.10


def nw_t(x, L):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return np.nan
    u = x - x.mean()
    g0 = u @ u / n
    s = g0
    for k in range(1, min(L, n - 1) + 1):
        s += 2 * (1 - k / (L + 1)) * (u[k:] @ u[:-k]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


# ------------------------------------------------------------------ SOXX 주간 경고(독립 구현: 달력 월요일 기준)
def soxx_warn(dates):
    o = pd.read_parquet(SOXX_SRC, columns=["date", "sym", "Close"])
    s = o[o.sym == "SOXX"].copy()
    s["d"] = s["date"].dt.strftime("%Y-%m-%d")
    s = s.drop_duplicates("d").set_index("d")["Close"].astype(float).sort_index()
    sd = list(s.index)
    cl = s.to_numpy()
    sma = roll(cl, 20, np.mean)
    below = cl < sma                                    # NaN 비교는 False
    warn = np.zeros(len(dates), bool)
    src = [None] * len(dates)
    import bisect
    for i in range(len(dates) - 1):
        d1 = date.fromisoformat(dates[i + 1])
        mon = (d1 - timedelta(days=d1.weekday())).isoformat()
        k = bisect.bisect_left(sd, mon) - 1                 # 월요일 전 마지막 SOXX 거래일
        if k >= 0:
            warn[i] = bool(below[k])
            src[i] = sd[k]
    return warn, src
