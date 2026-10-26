"""검증자 #1 (2026-10-10, 연구용): zones.py·zones_regime.py를 import하지 않고 독자 구현으로 다시 계산한다.

1. 지표(EMA·ATR·RSI·mz·don·slope·cross·이격·xret)를 직접 점화식/창 계산으로 만들고 432개 격자 + 옛 칼만의
   단계 비중·T1/T2/T3·S를 앞/뒤/2025/2024(h=10, 초과, 날짜 평균)로 재계산 → zones.json과 비교(허용 0.05%p).
2. 자격(앞 ±2 각 2~20%, ±1·0 각 10% 이상) 재판정, 자격 통과 중 앞 S 1위 확인.
3. SOXX 주간 경고를 독자 구현(t+1의 ISO 주 시작 전 마지막 SOXX 종가 < SMA20) → zones_regime.json과 비교.
4. 반박 시도: 국면별 NW t, 경고 구간 묶음 수, 극상방 표본 수, h 5·20 견고성, 주 단위 순환 이동 위약(placebo),
   미래 정보 점검(단계는 t까지, 성과는 O[t+1]→O[t+1+h], 앞 purge).
실행: $env:PYTHONIOENCODING='utf-8'; py data/derived/analysis/trend_zones/code/verify_1_recompute.py
결과: results/verify_1_recompute.json, verify_1_recompute.txt
"""
import json, os, warnings
from bisect import bisect_left
from datetime import date, timedelta
from itertools import product

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 5))
SRC = os.path.join(ROOT, "data/derived/analysis/rsi_breakout/cache/1d_long.parquet")
SOXX = os.path.join(ROOT, "data/derived/analysis/stop_research/cache/ohlc.parquet")
RES = os.path.abspath(os.path.join(HERE, "../results"))
MINB = 130
PER = {"앞": ("2026-01-02", "2026-05-29", True), "뒤": ("2026-06-01", "2026-10-09", False),
       "2025": ("2025-01-02", "2025-12-31", False), "2024": ("2024-01-02", "2024-12-31", False)}
K = ("T1 상방-하방", "T2 극하방-하방", "T3 상방-극상방")
TOL = 0.0005
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ------------------------------------------------------------------ 자료
def load():
    x = pd.read_parquet(SRC)
    x = x[x.sym != "QQQ"].copy()
    x["d"] = x["ts"].dt.strftime("%Y-%m-%d")
    piv = {k: x.pivot(index="d", columns="sym", values=k).sort_index() for k in ("Open", "High", "Low", "Close")}
    dates = list(piv["Close"].index)
    syms = list(piv["Close"].columns)
    return dates, syms, *(piv[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))


# ------------------------------------------------------------------ 지표(독자 구현)
def rec_ema(X, alpha, minp):
    """열마다 첫 유효값에서 시작하는 y=αx+(1-α)y_prev, 유효 관측 수 < minp면 NaN."""
    T, N = X.shape
    out = np.full((T, N), np.nan)
    prev = np.full(N, np.nan)
    cnt = np.zeros(N)
    for i in range(T):
        x = X[i]
        ok = np.isfinite(x)
        have = np.isfinite(prev)
        prev = np.where(ok & ~have, x, np.where(ok & have, alpha * x + (1 - alpha) * prev, prev))
        cnt += ok
        out[i] = np.where(cnt >= minp, prev, np.nan)
    return out


def roll(X, w, fn):
    T, N = X.shape
    out = np.full((T, N), np.nan)
    if T >= w:
        win = swv(X, w, axis=0)                     # (T-w+1, N, w)
        with np.errstate(invalid="ignore"):
            v = fn(win)
        out[w - 1:] = v                              # NaN이 창에 있으면 결과 NaN(nan-aware 아님)
    return out


def lag(X, k):
    out = np.full_like(X, np.nan)
    out[k:] = X[:-k]
    return out


class Ind:
    def __init__(self, O, H, L, C):
        self.O, self.H, self.L, self.C = O, H, L, C
        self.valid = (np.cumsum(np.isfinite(C), axis=0) >= MINB) & np.isfinite(C) & np.isfinite(O)
        pc = lag(C, 1)
        tr = np.fmax(H - L, np.fmax(np.abs(H - pc), np.abs(L - pc)))   # 첫 봉은 H-L
        self.atr = rec_ema(tr, 1 / 14, 14)
        d = C - pc
        up = rec_ema(np.where(np.isfinite(d), np.maximum(d, 0), np.nan), 1 / 14, 14)
        dn = rec_ema(np.where(np.isfinite(d), np.maximum(-d, 0), np.nan), 1 / 14, 14)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.rsi = np.where(dn == 0, 100.0, 100 - 100 / (1 + up / dn))
        self.lr = np.log(C) - np.log(pc)
        self.c = {}

    def get(self, key, fn):
        if key not in self.c:
            self.c[key] = fn()
        return self.c[key]

    def ema(self, Lp):
        return self.get(("ema", Lp), lambda: rec_ema(self.C, 2 / (Lp + 1), min(Lp, MINB)))

    def sd(self, n):
        return self.get(("sd", n), lambda: roll(self.lr, n, lambda w: w.std(axis=-1, ddof=1)))

    def zdev(self, Lp, kind):
        def f():
            dev = self.C - self.ema(Lp)
            if kind == "atr14":
                wd = self.atr
            else:
                w = int(kind[3:])
                wd = np.sqrt(roll(dev ** 2, w, lambda a: a.mean(axis=-1)))
            with np.errstate(divide="ignore", invalid="ignore"):
                return dev / wd
        return self.get(("z", Lp, kind), f)

    def trend(self, kind, p):
        def f():
            with np.errstate(divide="ignore", invalid="ignore"):
                if kind == "slope":
                    Lp, k = p
                    e = self.ema(Lp)
                    return (e - lag(e, k)) / (k * self.atr)
                if kind == "cross":
                    a, b = p
                    return (self.ema(a) - self.ema(b)) / self.atr
                if kind == "mz":
                    return np.log(self.C / lag(self.C, p)) / (self.sd(p) * np.sqrt(p))
                if kind == "don":
                    hh = roll(self.H, p, lambda w: w.max(axis=-1))
                    ll = roll(self.L, p, lambda w: w.min(axis=-1))
                    return 2 * (self.C - ll) / (hh - ll) - 1
            raise ValueError(kind)
        return self.get(("t", kind, p), f)

    def stretch(self, kind):
        def f():
            if kind == "xdev":
                return self.zdev(20, "rms60")
            if kind == "xrsi":
                return self.rsi
            if kind == "xret":
                with np.errstate(divide="ignore", invalid="ignore"):
                    return np.log(self.C / lag(self.C, 10)) / (self.sd(60) * np.sqrt(10))
            raise ValueError(kind)
        return self.get(("x", kind), f)


def my_grid():
    g = []
    for Lp, w, a, b in product((10, 20, 35), ("rms20", "rms30", "rms60", "atr14"), (0.5, 0.75, 1.0), (1.5, 2.0, 2.5)):
        g.append({"fam": "dev", "L": Lp, "w": w, "a": a, "b": b})
    T = ([("slope", p, (0.05, 0.1, 0.2)) for p in ((10, 3), (20, 5), (50, 10))]
         + [("cross", p, (0.5, 1.0, 2.0)) for p in ((10, 30), (20, 60), (50, 150))]
         + [("mz", p, (0.5, 1.0, 1.5)) for p in (20, 60, 120)]
         + [("don", p, (0.2, 0.4, 0.6)) for p in (20, 55, 120)])
    X = [("xdev", v) for v in (1.5, 2.0, 2.5)] + [("xrsi", v) for v in (70, 75, 80)] + [("xret", v) for v in (1.5, 2.0, 2.5)]
    for (tk, tp, taus) in T:
        for tau in taus:
            for xk, xv in X:
                g.append({"fam": "tx", "t": tk, "tp": tp, "tau": tau, "x": xk, "xv": xv})
    return g


def ckey(c):
    tp = c.get("tp")
    if isinstance(tp, list):
        tp = tuple(tp)
    return tuple(sorted((k, (tp if k == "tp" else v)) for k, v in c.items() if k != "name"))


def nm(c):
    if c.get("name"):
        return c["name"]
    if c["fam"] == "dev":
        return f"이격 EMA{c['L']}·{c['w']}·a{c['a']}·b{c['b']}"
    tp = c["tp"] if not isinstance(c["tp"], (tuple, list)) else ",".join(map(str, c["tp"]))
    return f"{c['t']}({tp})≥{c['tau']} + {c['x']}≥{c['xv']}"


BASE = {"fam": "dev", "L": 10, "w": "rms25", "a": 0.75, "b": 1.75, "name": "옛 Kalman 5단계(일봉 환산)"}


def zone(I, c):
    if c["fam"] == "dev":
        z = I.zdev(c["L"], c["w"])
        a, b = c["a"], c["b"]
        with np.errstate(invalid="ignore"):
            Z = np.where(z >= b, 2, np.where(z >= a, 1, np.where(z > -a, 0, np.where(z > -b, -1, -2))))
        fin = np.isfinite(z)
    else:
        tp = tuple(c["tp"]) if isinstance(c["tp"], (list, tuple)) else c["tp"]
        t = I.trend(c["t"], tp)
        x = I.stretch(c["x"])
        hi = c["xv"]
        lo = 100 - c["xv"] if c["x"] == "xrsi" else -c["xv"]
        tau = c["tau"]
        with np.errstate(invalid="ignore"):
            Z = np.where(x >= hi, 2, np.where(x <= lo, -2, np.where(t >= tau, 1, np.where(t <= -tau, -1, 0))))
        fin = np.isfinite(x) & np.isfinite(t)
    return np.where(fin & I.valid, Z, -9).astype(np.int8)


# ------------------------------------------------------------------ 성과·채점
def fwd(I, h):
    O = I.O
    T = O.shape[0]
    f = np.full_like(O, np.nan)
    f[: T - 1 - h] = O[1 + h:] / O[1: T - h] - 1          # f[t] = O[t+1+h]/O[t+1]-1
    f = np.where(I.valid, f, np.nan)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(f, axis=1, keepdims=True)
    return f, f - mu


def daily(Z, ex, raw):
    """날짜별: 단계별 평균 초과(m), 평균 원수익(mr), 개수(cnt)."""
    have = np.isfinite(ex) & (Z != -9)
    m, mr, cnt = {}, {}, {}
    for z in (2, 1, 0, -1, -2):
        M = have & (Z == z)
        n = M.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m[z] = np.where(n > 0, np.where(M, ex, 0).sum(axis=1) / n, np.nan)
            mr[z] = np.where(n > 0, np.where(M, raw, 0).sum(axis=1) / n, np.nan)
        cnt[z] = n
    term = {K[0]: m[1] - m[-1], K[1]: m[-2] - m[-1], K[2]: m[1] - m[2]}
    return {"m": m, "mr": mr, "cnt": cnt, "tot": have.sum(axis=1), "term": term}


def nanmean(v):
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def agg(D, rows):
    tot = D["tot"][rows].sum()
    share = {z: float(D["cnt"][z][rows].sum() / tot) if tot else float("nan") for z in (2, 1, 0, -1, -2)}
    terms = {k: nanmean(D["term"][k][rows]) for k in K}
    return {"share": share, "terms": terms, "S": float(sum(terms.values()))}


def nw(x, L):
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    e = x - x.mean()
    g0 = e @ e / n
    v = g0 + sum(2 * (1 - k / (L + 1)) * (e[k:] @ e[:-k]) / n for k in range(1, min(L, n - 1) + 1))
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else float("nan")


def rows_of(dates, p, h):
    a, b, purge = PER[p]
    idx = [i for i, d in enumerate(dates) if a <= d <= b]
    if purge:
        e = max(idx)
        idx = [i for i in idx if i + 1 + h <= e]
    return np.array(idx)


def elig(s):
    return 0.02 <= s[2] <= 0.20 and 0.02 <= s[-2] <= 0.20 and min(s[1], s[-1], s[0]) >= 0.10


# ------------------------------------------------------------------ SOXX 경고(독자 구현)
def soxx_warn(dates):
    o = pd.read_parquet(SOXX, columns=["date", "sym", "Close"])
    s = o[o.sym == "SOXX"].sort_values("date")
    sd = list(s["date"].dt.strftime("%Y-%m-%d"))
    c = s["Close"].to_numpy(float)
    cs = np.cumsum(np.r_[0.0, c])
    sma = np.full(len(c), np.nan)
    sma[19:] = (cs[20:] - cs[:-20]) / 20
    below = np.where(np.isfinite(sma), c < sma, False)
    warn = np.zeros(len(dates), bool)
    wk = [None] * len(dates)
    src = [None] * len(dates)
    for i in range(len(dates) - 1):
        t1 = date.fromisoformat(dates[i + 1])
        mon = (t1 - timedelta(days=t1.weekday())).isoformat()   # ISO 주 시작(월)
        j = bisect_left(sd, mon) - 1                              # 그 주 시작 전 마지막 SOXX 거래일
        warn[i] = bool(below[j]) if j >= 0 else False
        wk[i] = mon
        src[i] = sd[j] if j >= 0 else None
    missing = [d for d in dates if d not in set(sd)]
    return warn, wk, src, missing


# ------------------------------------------------------------------ 실행
def main():
    out, lines = {}, []
    P = lambda v: f"{v * 100:+.2f}"
    dates, syms, O, H, L, C = load()
    I = Ind(O, H, L, C)
    lines.append(f"종목 {len(syms)}개, {dates[0]}~{dates[-1]}, 거래일 {len(dates)}")

    # ---------------- 1. 격자 재계산 + zones.json 비교
    ref = json.load(open(os.path.join(RES, "zones.json"), encoding="utf-8"))
    G = my_grid()
    refmap = {ckey(g["cfg"]): g for g in ref["grid"]}
    same_grid = set(map(ckey, G)) == set(refmap) and len(G) == len(ref["grid"]) == 432
    lines.append(f"격자: 독자 생성 {len(G)}개, zones.json {len(ref['grid'])}개, 설정 집합 일치 {same_grid}")
    F = {h: fwd(I, h) for h in (5, 10, 20)}
    R = {(p, h): rows_of(dates, p, h) for p in PER for h in (5, 10, 20)}
    allc = G + [BASE]
    Dh = {}
    res = {}
    for c in allc:
        Z = zone(I, c)
        for h in (5, 10, 20):
            Dh[(ckey(c), h)] = daily(Z, F[h][1], F[h][0])
        res[ckey(c)] = {p: agg(Dh[(ckey(c), 10)], R[p, 10]) for p in PER}
    maxdiff = {"share": 0.0, "terms": 0.0, "S": 0.0}
    worst = None
    for c in G:
        k = ckey(c)
        for p in ("앞", "뒤", "2025"):
            a, b = res[k][p], refmap[k][p]
            for z in (2, 1, 0, -1, -2):
                dlt = abs(a["share"][z] - b["share"][str(z)])
                maxdiff["share"] = max(maxdiff["share"], dlt)
            for t in K:
                x, y = a["terms"][t], b["terms"][t]
                if np.isfinite(x) or np.isfinite(y):
                    dlt = abs(x - y)
                    if dlt > maxdiff["terms"]:
                        maxdiff["terms"], worst = dlt, (nm(c), p, t, x, y)
            maxdiff["S"] = max(maxdiff["S"], abs(a["S"] - b["S"]))
    lines.append(f"zones.json 대비 최대 차이(432개×앞·뒤·2025): 비중 {maxdiff['share'] * 100:.5f}%p, "
                 f"항 {maxdiff['terms'] * 100:.5f}%p, S {maxdiff['S'] * 100:.5f}%p (허용 0.05%p) 최악 {worst}")
    out["격자_비교"] = {"설정집합_일치": same_grid, "최대차이_퍼센트포인트": {k: v * 100 for k, v in maxdiff.items()},
                     "허용_통과": all(v <= TOL for v in maxdiff.values())}

    # 자격·1위
    E = [c for c in G if elig(res[ckey(c)]["앞"]["share"])]
    ref_e = {ckey(g["cfg"]) for g in ref["grid"] if g["eligible"]}
    E.sort(key=lambda c: -res[ckey(c)]["앞"]["S"])
    win = E[0]
    lines.append(f"자격 통과 {len(E)}개(zones.json {len(ref_e)}개, 집합 일치 {set(map(ckey, E)) == ref_e}); "
                 f"앞 S 1위 {nm(win)} {P(res[ckey(win)]['앞']['S'])}, 2위 {nm(E[1])} {P(res[ckey(E[1])]['앞']['S'])}; "
                 f"zones.json best={ref['best']}")
    # 자격 없이 전체 432개 1위(참고)
    G_sorted = sorted(G, key=lambda c: -res[ckey(c)]["앞"]["S"])
    lines.append(f"(참고) 자격 무시 432개 앞 S 1위: {nm(G_sorted[0])} {P(res[ckey(G_sorted[0])]['앞']['S'])}")
    out["자격"] = {"개수": len(E), "zones_json_개수": len(ref_e), "집합일치": set(map(ckey, E)) == ref_e,
                 "1위": nm(win), "1위_앞S": res[ckey(win)]["앞"]["S"], "zones_json_best": ref["best"],
                 "자격무시_1위": nm(G_sorted[0])}

    # 상세: 1위·옛칼만·다른 계열 10개
    picks = [win, BASE,
             {"fam": "dev", "L": 20, "w": "rms60", "a": 0.75, "b": 2.0},
             {"fam": "dev", "L": 35, "w": "atr14", "a": 1.0, "b": 2.5},
             {"fam": "tx", "t": "slope", "tp": (10, 3), "tau": 0.1, "x": "xdev", "xv": 2.0},
             {"fam": "tx", "t": "slope", "tp": (50, 10), "tau": 0.05, "x": "xrsi", "xv": 70},
             {"fam": "tx", "t": "cross", "tp": (20, 60), "tau": 1.0, "x": "xrsi", "xv": 70},
             {"fam": "tx", "t": "cross", "tp": (50, 150), "tau": 2.0, "x": "xret", "xv": 2.5},
             {"fam": "tx", "t": "mz", "tp": 20, "tau": 0.5, "x": "xrsi", "xv": 70},
             {"fam": "tx", "t": "mz", "tp": 120, "tau": 1.5, "x": "xdev", "xv": 1.5},
             {"fam": "tx", "t": "don", "tp": (120), "tau": 0.4, "x": "xrsi", "xv": 70},
             {"fam": "tx", "t": "don", "tp": 20, "tau": 0.2, "x": "xret", "xv": 2.0}]
    lines += ["", "== 개별 설정 재계산(h10, 초과, 날짜 평균) — 내 값 / zones.json 값"]
    det = {}
    for c in picks:
        k = ckey(c)
        r = res[k] if k in res else None
        row = {}
        for p in PER:
            a = r[p]
            refv = refmap.get(k, {}).get(p) if p != "2024" else None
            tstat = {t: nw(Dh[(k, 10)]["term"][t][R[p, 10]], 10) for t in K}
            row[p] = {"S": a["S"], **a["terms"], "share": a["share"], "t": tstat,
                      "ref_S": (refv or {}).get("S")}
            lines.append(f"[{nm(c)} · {p}] S {P(a['S'])} (ref {P(refv['S']) if refv else '  -  '}) "
                         + " ".join(f"{t[:2]} {P(a['terms'][t])}(t{tstat[t]:+.1f})" for t in K)
                         + " | 비중 " + " ".join(f"{z:+d}:{a['share'][z] * 100:.1f}" for z in (2, 1, 0, -1, -2)))
        det[nm(c)] = row
    # 옛칼만·1위의 앞~2024 zones.txt 숫자와 대조(zones.json detail)
    for lab in ("선택", "옛칼만"):
        dd = ref["detail"][lab]
        c = dd["cfg"]
        k = ckey(c)
        for p in PER:
            e = dd["d"][f"{p}·h10"]
            mine = res[k][p]
            diff = max(abs(mine["terms"][t] - e["terms"][t]) for t in K)
            tdiff = max(abs(nw(Dh[(k, 10)]["term"][t][R[p, 10]], 10) - e["t"][t]) for t in K)
            lines.append(f"  detail {lab} {p}: 항 최대차 {diff * 100:.5f}%p, t 최대차 {tdiff:.4f}")
    out["개별"] = det

    # 격자 분포(K1·K2)
    lines += ["", "== 자격 통과 설정에서 항이 양수인 비율(h10)"]
    dist = {}
    for p in PER:
        dist[p] = {}
        for t in K:
            v = np.array([res[ckey(c)][p]["terms"][t] for c in E])
            v = v[np.isfinite(v)]
            dist[p][t] = {"양수비율": float((v > 0).mean()), "중앙": float(np.median(v)), "n": int(len(v))}
        lines.append(f"[{p}] " + " | ".join(f"{t} {dist[p][t]['양수비율'] * 100:.0f}% 중앙 {P(dist[p][t]['중앙'])}" for t in K))
    out["분포"] = dist

    # ---------------- 2. 미래 정보 점검
    lines += ["", "== 미래 정보·자료 점검"]
    aud = {}
    # (a) 단계가 t 이후 자료에 의존하지 않는지: t0 이후 자료를 NaN으로 지운 뒤 다시 계산해 t0까지 단계가 같은지
    cut = dates.index("2026-03-02")
    Ot, Ht, Lt, Ct = (A.copy() for A in (O, H, L, C))
    for A in (Ot, Ht, Lt, Ct):
        A[cut + 1:] = np.nan
    It = Ind(Ot, Ht, Lt, Ct)
    mism = 0
    for c in E + [BASE]:
        z1 = zone(I, c)[: cut + 1]
        z2 = zone(It, c)[: cut + 1]
        mism += int((z1 != z2).sum())
    aud["자르기_검사_불일치칸"] = mism
    lines.append(f"(a) 2026-03-02 이후 가격을 지운 뒤 재계산: 자격 {len(E)}개+옛칼만의 3/2까지 단계 불일치 {mism}칸")
    # (b) 성과 정의: f[t] = O[t+1+h]/O[t+1]-1 직접 확인(임의 종목·날짜)
    j = syms.index("NVDA") if "NVDA" in syms else 0
    i = dates.index("2026-02-02")
    f_chk = O[i + 11, j] / O[i + 1, j] - 1
    aud["성과정의"] = {"종목": syms[j], "t": dates[i], "t+1": dates[i + 1], "t+11": dates[i + 11],
                    "직접": float(f_chk), "배열": float(F[10][0][i, j])}
    lines.append(f"(b) {syms[j]} t={dates[i]}: O[{dates[i + 11]}]/O[{dates[i + 1]}]-1 = {f_chk * 100:+.3f}% vs 배열 {F[10][0][i, j] * 100:+.3f}%")
    # (c) 앞 purge
    ra = R["앞", 10]
    last_used = dates[ra.max() + 1 + 10]
    aud["앞_purge"] = {"마지막_신호일": dates[ra.max()], "마지막_사용_시가일": last_used, "신호일수": int(len(ra))}
    lines.append(f"(c) 앞 마지막 신호일 {dates[ra.max()]} → 마지막 사용 시가 {last_used} (≤ 2026-05-29), 신호일 {len(ra)}일")
    # 2025 끝 신호가 2026 1월 시가를 쓰는지(보조 구간 겹침)
    r25 = R["2025", 10]
    aud["2025_끝_겹침"] = dates[r25.max() + 1 + 10] if r25.max() + 11 < len(dates) else None
    lines.append(f"    참고: 2025 마지막 신호 {dates[r25.max()]}의 성과는 {aud['2025_끝_겹침']} 시가까지(앞 구간 초입과 9거래일 겹침, 보조 구간이라 선택엔 영향 없음)")
    # (d) 횡단 평균이 같은 날 종목만 쓰는지 + 초과/원수익 무관성: T항은 같은 날 두 단계 평균의 차라 시장 평균이 상쇄
    kw = ckey(win)
    rawD = daily(zone(I, win), F[10][0], F[10][0])
    d_raw = max(abs(agg(rawD, R[p, 10])["terms"][t] - res[kw][p]["terms"][t]) for p in PER for t in K)
    aud["초과_vs_원수익_T항_최대차"] = d_raw
    lines.append(f"(d) 1위 설정 T항을 원수익으로 계산해도 최대차 {d_raw * 100:.6f}%p → '초과'는 T항에 영향 없음(같은 날 평균이 상쇄)")
    # (e) MIN_BARS·상장 직후
    late = {s: dates[int(np.argmax(np.isfinite(C[:, k])))] for k, s in enumerate(syms) if not np.isfinite(C[0, k])}
    firstvalid = {s: dates[int(np.argmax(I.valid[:, syms.index(s)]))] if I.valid[:, syms.index(s)].any() else None for s in late}
    aud["늦은_상장"] = {s: {"첫봉": late[s], "첫유효": firstvalid[s]} for s in late}
    lines.append(f"(e) 2023-01-03 이후 시작 {len(late)}종목, 130봉 뒤 유효: " + ", ".join(f"{s} {firstvalid[s]}" for s in late))
    v_all = int(I.valid.sum())
    rsi_nan_valid = int((I.valid & ~np.isfinite(I.rsi)).sum())
    aud["유효칸_RSI_NaN"] = rsi_nan_valid
    # EMA150 초기값 잔존 가중(2023-01-03 시작 데이터의 한계)
    w150 = (1 - 2 / 151) ** np.array([130, 250, 500])
    aud["EMA150_초기값_잔존가중_130_250_500봉"] = [float(x) for x in w150]
    lines.append(f"    유효칸 {v_all}개 중 RSI NaN {rsi_nan_valid}개. EMA150 첫 값 잔존 가중 130봉 {w150[0]:.2f}, 250봉 {w150[1]:.3f}, 500봉 {w150[2]:.4f}"
                 " (2023-01 시작 자료라 2023~24 초 cross(50,150) 값에 초기값 영향)")

    # ---------------- 3. SOXX 경고 + 국면 분할
    warn, wk, src, missing = soxx_warn(dates)
    lines += ["", f"== SOXX 주간 경고(독자 구현). 주가 날짜 중 SOXX에 없는 날 {len(missing)}개"]
    # 미래 정보: 경고 판단일 src < t+1
    bad = [i for i in range(len(dates) - 1) if src[i] is None or src[i] >= dates[i + 1]]
    late_src = [i for i in range(len(dates) - 1) if src[i] is not None and src[i] > dates[i]]
    aud["경고_판단일_t+1_이전"] = len(bad) == 0
    aud["경고_판단일_t_이후인_경우"] = len(late_src)
    lines.append(f"경고 판단 종가일이 t+1보다 늦은 경우 {len(bad)}건, t보다 늦은 경우 {len(late_src)}건(0이어야 t 종가 시점에 앎)")
    regref = json.load(open(os.path.join(RES, "zones_regime.json"), encoding="utf-8"))
    split = {}
    for p in PER:
        r = R[p, 10]
        split[p] = {"경고": r[warn[r]], "평상": r[~warn[r]]}
        # 경고 묶음(연속 경고 주) 수
        wks = sorted({wk[i] for i in r[warn[r]]})
        runs = 0
        prev = None
        for w_ in wks:
            if prev is None or (date.fromisoformat(w_) - date.fromisoformat(prev)).days > 7:
                runs += 1
            prev = w_
        split[p]["묶음"] = runs
        split[p]["경고주"] = len(wks)
        lines.append(f"  {p}: 경고 {warn[r].mean() * 100:.0f}% ({warn[r].sum()}일, {len(wks)}주, 연속 묶음 {runs}개)")
    reg = {}
    md = 0.0
    for c in E + [win, BASE]:
        k = ckey(c)
        reg[k] = {p: {s: {t: nanmean(Dh[(k, 10)]["term"][t][split[p][s]]) for t in K} for s in ("경고", "평상")} for p in PER}
        rr = regref.get(nm(c))
        if rr:
            for p in PER:
                for s in ("경고", "평상"):
                    for t in K:
                        x, y = reg[k][p][s][t], rr[p][s][t]
                        if np.isfinite(x) and y is not None and np.isfinite(y):
                            md = max(md, abs(x - y))
    lines.append(f"zones_regime.json 대비 최대 차이 {md * 100:.5f}%p")
    out["국면_비교_최대차"] = md * 100
    lines += ["", "== 자격 설정 중 항이 양수인 비율 / 중앙값(국면별, h10)"]
    regdist = {}
    for p in PER:
        for s in ("평상", "경고"):
            parts, d_ = [], {}
            for t in K:
                v = np.array([reg[ckey(c)][p][s][t] for c in E])
                v = v[np.isfinite(v)]
                d_[t] = {"양수비율": float((v > 0).mean()), "중앙": float(np.median(v))}
                parts.append(f"{t} {d_[t]['양수비율'] * 100:3.0f}% {P(d_[t]['중앙'])}")
            regdist[f"{p}·{s}"] = d_
            lines.append(f"[{p}·{s}] " + " | ".join(parts))
    out["국면_분포"] = regdist

    # 1위 설정 국면별 T항 + NW t + 극상방 표본
    lines += ["", f"== 사전 등록 1위 {nm(win)}: 국면별 T항(t, 극상방·상방 동시 존재 날짜 수, 극상방 하루 평균 종목 수, 극상방 초과·원수익)"]
    D = Dh[(kw, 10)]
    wdet = {}
    for p in PER:
        for s in ("평상", "경고"):
            r = split[p][s]
            both = int(np.isfinite(D["term"][K[2]][r]).sum())
            n2 = float(D["cnt"][2][r].mean()) if len(r) else float("nan")
            ex2 = nanmean(D["m"][2][r])
            raw2 = nanmean(D["mr"][2][r])
            rawall = nanmean(np.nanmean(np.where(np.isfinite(F[10][0][r]), F[10][0][r], np.nan), axis=1)) if len(r) else float("nan")
            ts = {t: nw(D["term"][t][r], 10) for t in K}
            wdet[f"{p}·{s}"] = {**{t: reg[kw][p][s][t] for t in K}, "t": ts, "T3_날짜수": both, "극상방_하루평균": n2,
                                "극상방_초과": ex2, "극상방_원수익": raw2, "전체_원수익": rawall}
            lines.append(f"[{p}·{s}] " + "  ".join(f"{t[:2]} {P(reg[kw][p][s][t])}(t{ts[t]:+.1f})" for t in K)
                         + f" | T3 날짜 {both}, 극상방 {n2:.1f}종목/일, 극상방 초과 {P(ex2)} 원 {P(raw2)} (전체 원 {P(rawall)})")
    out["1위_국면"] = wdet

    # 전 기간 합친 경고/평상 T3 (1위, 자격 중앙)
    allrows = np.concatenate([R[p, 10] for p in ("앞", "뒤", "2025", "2024")])
    aw, an = allrows[warn[allrows]], allrows[~warn[allrows]]
    t3w = nanmean(D["term"][K[2]][aw]); t3n = nanmean(D["term"][K[2]][an])
    lines.append(f"4구간 합침: 1위 T3 경고 {P(t3w)} (t {nw(D['term'][K[2]][aw], 10):+.1f}) / 평상 {P(t3n)} (t {nw(D['term'][K[2]][an], 10):+.1f})")
    # 경고-평상 차이 자체의 t: 날짜별 T3를 경고 표시 회귀(블록 붓스트랩 대신 주 단위 위약으로 대체, 아래)
    out["1위_합침"] = {"T3_경고": t3w, "T3_평상": t3n}

    # ---------------- 4. 견고성: h 5·20
    lines += ["", "== h 5·20에서 T3>0 비율(자격 설정, 국면별)"]
    hrob = {}
    for h in (5, 20):
        for p in PER:
            r = R[p, h]
            sp = {"경고": r[warn[r]], "평상": r[~warn[r]]}
            seg = []
            for s in ("평상", "경고"):
                v = np.array([nanmean(Dh[(ckey(c), h)]["term"][K[2]][sp[s]]) for c in E])
                v = v[np.isfinite(v)]
                hrob[f"h{h}·{p}·{s}"] = float((v > 0).mean())
                seg.append(f"{s} {np.mean(v > 0) * 100:3.0f}% 중앙 {P(np.median(v))}")
            lines.append(f"[h{h}·{p}] " + " | ".join(seg))
    out["h_견고성_T3양수비율"] = hrob

    # ---------------- 5. 위약: 주 단위 경고 표시를 순환 이동
    weeks = sorted({w_ for w_ in wk if w_ is not None})
    wl = {}
    for i in range(len(dates) - 1):
        wl[wk[i]] = warn[i]
    lab = np.array([wl[w_] for w_ in weeks])
    widx = np.array([weeks.index(w_) if w_ is not None else -1 for w_ in wk])
    T3 = np.array([Dh[(ckey(c), 10)]["term"][K[2]] for c in E])        # (설정, 날짜)
    T3w = Dh[(kw, 10)]["term"][K[2]]
    prow = {p: R[p, 10] for p in PER}

    def stat(wv):
        a, b, per = 0.0, 0.0, {}
        for p in PER:
            r = prow[p]
            m_ = wv[r]
            if m_.sum() == 0 or (~m_).sum() == 0:
                return None
            with np.errstate(invalid="ignore"):
                sw = np.nanmean(np.where(np.isfinite(T3[:, r[m_]]), T3[:, r[m_]], np.nan), axis=1)
                sn = np.nanmean(np.where(np.isfinite(T3[:, r[~m_]]), T3[:, r[~m_]], np.nan), axis=1)
            d_ = np.mean(sw > 0) - np.mean(sn > 0)
            a += d_ / 4
            per[p] = d_
            b += (nanmean(T3w[r[m_]]) - nanmean(T3w[r[~m_]])) / 4
        return a, b, per

    def warn_from(labels):
        wv = np.zeros(len(dates), bool)
        ok = widx >= 0
        wv[ok] = labels[widx[ok]]
        return wv

    obs = stat(warn)
    sims_a, sims_b, sims_min = [], [], []
    nW = len(weeks)
    for sh in range(4, nW - 3):
        s_ = stat(warn_from(np.roll(lab, sh)))
        if s_ is None:
            continue
        sims_a.append(s_[0]); sims_b.append(s_[1]); sims_min.append(min(s_[2].values()))
    sims_a, sims_b, sims_min = map(np.array, (sims_a, sims_b, sims_min))
    pa = float(np.mean(sims_a >= obs[0])); pb = float(np.mean(sims_b >= obs[1]))
    pm = float(np.mean(sims_min >= min(obs[2].values())))
    lines += ["", f"== 위약(주 단위 경고 표시 순환 이동 {len(sims_a)}회, 묶음 구조 유지)",
              f"관측: 경고-평상 T3 양수 비율 차 평균 {obs[0] * 100:+.1f}%p (구간별 " + ", ".join(f"{p} {v * 100:+.0f}" for p, v in obs[2].items())
              + f"), 1위 T3 차 평균 {P(obs[1])}%p",
              f"위약 분포: 비율 차 평균 중앙 {np.median(sims_a) * 100:+.1f}%p, 95% {np.quantile(sims_a, .95) * 100:+.1f}%p → p={pa:.3f}; "
              f"4구간 최소 차 ≥ 관측 p={pm:.3f}; 1위 T3 차 p={pb:.3f}"]
    out["위약"] = {"회수": int(len(sims_a)), "관측_비율차_평균": obs[0], "관측_구간별": obs[2], "관측_1위차": obs[1],
                 "p_비율차": pa, "p_구간최소": pm, "p_1위": pb,
                 "위약_비율차_중앙": float(np.median(sims_a)), "위약_비율차_95": float(np.quantile(sims_a, .95))}

    # ---------------- 6. 다른 '시장 하락' 정의로도 같은가(참고): 유니버스 동일가중 지수 < SMA20 (주 단위, 같은 시점 규칙)
    ew = np.nanmean(C / np.where(np.isfinite(lag(C, 1)), lag(C, 1), np.nan), axis=1)
    ew = np.cumprod(np.where(np.isfinite(ew), ew, 1.0))
    sma = pd.Series(ew).rolling(20).mean().to_numpy()
    below = ew < sma
    wv2 = np.zeros(len(dates), bool)
    for i in range(len(dates) - 1):
        t1 = date.fromisoformat(dates[i + 1])
        mon = (t1 - timedelta(days=t1.weekday())).isoformat()
        j = bisect_left(dates, mon) - 1
        wv2[i] = bool(below[j]) if j >= 0 and np.isfinite(sma[j]) else False
    agree = float((wv2 == warn)[allrows].mean())
    s2 = stat(wv2)
    lines.append(f"(참고) 유니버스 동일가중 < SMA20 주 경고: SOXX 경고와 일치율 {agree * 100:.0f}%, 경고-평상 T3 양수비율 차 "
                 + ", ".join(f"{p} {v * 100:+.0f}" for p, v in s2[2].items()))
    out["대체경고_동일가중"] = {"일치율": agree, "구간별": s2[2]}

    out["점검"] = aud
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(RES, "verify_1_recompute.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(RES, "verify_1_recompute.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))


if __name__ == "__main__":
    main()
