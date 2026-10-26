"""추세 5단계 띠 연구 (2026-10-10, 연구용). desk/·gbrc/·docs/·watchlist.py는 고치지 않는다.

사용자: 가격 흐름만으로 개별주의 지금 주 추세가 상방인지 하방인지 애매한지 5단계로 보고 싶다(차트 맨 아래 색 띠).
        극상방은 '이제 팔고 나올 때', 극하방은 '사 봐도 될 때'라고 본다. 어느 종목에나 맞는 건 없으니 경우의 수를
        최대한 조합해 확률을 높이는 기준을 찾아 달라. 단계는 +2 극상방, +1 상방, 0 애매, -1 하방, -2 극하방.

자료: rsi_breakout/cache/1d_long.parquet — TIME·KoAct 2026 미국 보유 150종목(QQQ 제외) 일봉, 2023-01-03~2026-10-09,
      yfinance auto_adjust=False(분할만 반영, 트레이딩뷰 기본 차트와 같음).
시점: t일 종가까지로 단계를 정하고, 성과는 t+1일 시가부터 잰다: fwd_h = O[t+1+h] / O[t+1] - 1.
      초과 = 같은 날 신호가 있는 전 종목 평균을 뺀 값(시장 전체 움직임 제거). 종목은 일봉 130개가 쌓인 뒤부터 쓴다.

사전 등록 격자(결과를 본 뒤 바꾸지 않음) — 432개
  A. 한 축(이격) 108개: z = (C - EMA_L) / 폭.  z >= b → +2, >= a → +1, > -a → 0, > -b → -1, 그 아래 -2.
     L ∈ {10, 20, 35}, 폭 ∈ {RMS20, RMS30, RMS60(C-EMA_L의 이동 제곱평균근), ATR14}, a ∈ {0.5, 0.75, 1.0}, b ∈ {1.5, 2.0, 2.5}.
  B. 두 축(추세 T + 과열 X) 324개: X가 위 문턱 이상이면 +2, 아래 문턱 이하면 -2. 아니면 T >= τ → +1, T <= -τ → -1, 그 사이 0.
     T 12개 × τ 3개:
       기울기 slope(L,k) = (EMA_L[t]-EMA_L[t-k]) / (k·ATR14), (L,k) ∈ {(10,3),(20,5),(50,10)}, τ ∈ {0.05, 0.1, 0.2}
       교차 cross(f,s) = (EMA_f - EMA_s) / ATR14, (f,s) ∈ {(10,30),(20,60),(50,150)}, τ ∈ {0.5, 1.0, 2.0}
       모멘텀 mz(N) = ln(C/C[N]) / (일 로그수익 표준편차(N일) · √N), N ∈ {20, 60, 120}, τ ∈ {0.5, 1.0, 1.5}
       채널 don(N) = 2(C-LL_N)/(HH_N-LL_N) - 1, N ∈ {20, 55, 120}, τ ∈ {0.2, 0.4, 0.6}
     X 3종류 × 문턱 3개: xdev = (C-EMA20)/RMS60 ≥ {1.5, 2.0, 2.5} | xrsi = RSI14 ≥ {70, 75, 80}(아래는 100-x) |
       xret = ln(C/C[10]) / (60일 표준편차·√10) ≥ {1.5, 2.0, 2.5}
  참고(선택 대상 아님): 옛 Kalman 5단계(240분 21봉·RMS 50봉·a 0.75·b 1.75)를 일봉으로 옮긴 EMA10·RMS25·0.75·1.75.
채점(사용자 해석 그대로, h = 10거래일, 초과 수익, 날짜마다 먼저 평균):
  T1 = 상방 - 하방(추세가 이어지나), T2 = 극하방 - 하방(극하방이 살 자리인가), T3 = 상방 - 극상방(극상방이 팔 자리인가).
  각 항은 같은 날 두 단계 평균의 차를 날짜로 평균한 값이다. 점수 S = T1 + T2 + T3.
  자격: 앞 구간에서 ±2 각각 2~20%, ±1·0 각각 10% 이상.
선택: 앞 2026-01-02~05-29(10일 뒤 성과가 5/29 안에 끝나는 신호만) S 1위. 확인: 뒤 2026-06-01~10-09. 보조: 2025, 2024.
함께 보고: 단계별 비중·10일 수익·10일 상승 확률, h 5·20, 이웃 설정, 반도체/비반도체, 단계대로 사고판 결과(보조).
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/trend_zones/code/zones.py
결과: data/derived/analysis/trend_zones/results/zones.json, zones.txt
"""
import json, os, sys
from itertools import product

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 5))
sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "data/derived/analysis/rsi_breakout/cache/1d_long.parquet")
OUT = os.path.join(HERE, "../results")
MIN_BARS = 130
H = 10
PERIODS = {"앞": ("2026-01-02", "2026-05-29", True), "뒤": ("2026-06-01", "2026-10-09", False),
           "2025": ("2025-01-02", "2025-12-31", False), "2024": ("2024-01-02", "2024-12-31", False)}
NAMES = {2: "극상방", 1: "상방", 0: "애매", -1: "하방", -2: "극하방"}


# ---------------------------------------------------------------- 자료·지표
def load():
    x = pd.read_parquet(SRC)
    x = x[x.sym != "QQQ"].copy()
    x["d"] = pd.to_datetime(x["ts"]).dt.strftime("%Y-%m-%d")
    f = {k: x.pivot_table(index="d", columns="sym", values=k).sort_index() for k in ("Open", "High", "Low", "Close")}
    return f["Open"], f["High"], f["Low"], f["Close"]


class Ind:
    def __init__(self, O, Hh, L, C):
        self.O, self.H, self.L, self.C = O, Hh, L, C
        self.dates = list(C.index)
        self.syms = list(C.columns)
        self.valid = (C.notna().cumsum() >= MIN_BARS) & C.notna() & O.notna()
        pc = C.shift(1)
        tr = pd.concat([(Hh - L), (Hh - pc).abs(), (L - pc).abs()]).groupby(level=0).max().reindex(C.index)
        self.atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        d = C.diff()
        up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        dn = (-d).clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        self.rsi = (100 - 100 / (1 + up / dn)).where(dn > 0, 100.0)
        self.lr = np.log(C).diff()
        self._ema, self._c = {}, {}

    def ema(self, L):
        if L not in self._ema:
            self._ema[L] = self.C.ewm(span=L, adjust=False, min_periods=min(L, MIN_BARS)).mean()
        return self._ema[L]

    def cached(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def width(self, L, kind):
        def f():
            if kind == "atr14":
                return self.atr
            w = int(kind[3:])
            dev = self.C - self.ema(L)
            return np.sqrt((dev ** 2).rolling(w, min_periods=w).mean())
        return self.cached(("w", L, kind), f)

    def zdev(self, L, kind):
        return self.cached(("z", L, kind), lambda: (self.C - self.ema(L)) / self.width(L, kind))

    def trend(self, kind, p):
        def f():
            if kind == "slope":
                L, k = p
                return (self.ema(L) - self.ema(L).shift(k)) / (k * self.atr)
            if kind == "cross":
                a, b = p
                return (self.ema(a) - self.ema(b)) / self.atr
            if kind == "mz":
                N = p
                sd = self.lr.rolling(N, min_periods=N).std()
                return np.log(self.C / self.C.shift(N)) / (sd * np.sqrt(N))
            if kind == "don":
                N = p
                hh = self.H.rolling(N, min_periods=N).max()
                ll = self.L.rolling(N, min_periods=N).min()
                return 2 * (self.C - ll) / (hh - ll) - 1
            raise ValueError(kind)
        return self.cached(("t", kind, p), f)

    def stretch(self, kind):
        def f():
            if kind == "xdev":
                return self.zdev(20, "rms60")
            if kind == "xrsi":
                return self.rsi
            if kind == "xret":
                sd = self.lr.rolling(60, min_periods=60).std()
                return np.log(self.C / self.C.shift(10)) / (sd * np.sqrt(10))
            raise ValueError(kind)
        return self.cached(("x", kind), f)


# ---------------------------------------------------------------- 설정 → 단계
def grid():
    out = []
    for L, w, a, b in product((10, 20, 35), ("rms20", "rms30", "rms60", "atr14"), (0.5, 0.75, 1.0), (1.5, 2.0, 2.5)):
        out.append({"fam": "dev", "L": L, "w": w, "a": a, "b": b})
    tlist = ([("slope", p, (0.05, 0.1, 0.2)) for p in ((10, 3), (20, 5), (50, 10))]
             + [("cross", p, (0.5, 1.0, 2.0)) for p in ((10, 30), (20, 60), (50, 150))]
             + [("mz", p, (0.5, 1.0, 1.5)) for p in (20, 60, 120)]
             + [("don", p, (0.2, 0.4, 0.6)) for p in (20, 55, 120)])
    xlist = ([("xdev", x) for x in (1.5, 2.0, 2.5)] + [("xrsi", x) for x in (70, 75, 80)]
             + [("xret", x) for x in (1.5, 2.0, 2.5)])
    for (tk, tp, taus), tau, (xk, xv) in ((t, tau, x) for t in tlist for tau in t[2] for x in xlist):
        out.append({"fam": "tx", "t": tk, "tp": tp, "tau": tau, "x": xk, "xv": xv})
    return out


BASELINE = {"fam": "dev", "L": 10, "w": "rms25", "a": 0.75, "b": 1.75, "name": "옛 Kalman 5단계(일봉 환산)"}


def cname(c):
    if c.get("name"):
        return c["name"]
    if c["fam"] == "dev":
        return f"이격 EMA{c['L']}·{c['w']}·a{c['a']}·b{c['b']}"
    tp = c["tp"] if not isinstance(c["tp"], (tuple, list)) else ",".join(map(str, c["tp"]))
    return f"{c['t']}({tp})≥{c['tau']} + {c['x']}≥{c['xv']}"


def zones(I: Ind, c) -> np.ndarray:
    """dates × syms 정수 배열(-2..2), 계산 불가는 -9."""
    if c["fam"] == "dev":
        z = I.zdev(c["L"], c["w"]).to_numpy()
        a, b = c["a"], c["b"]
        out = np.select([z >= b, z >= a, z > -a, z > -b], [2, 1, 0, -1], -2)
    else:
        t = I.trend(c["t"], tuple(c["tp"]) if isinstance(c["tp"], list) else c["tp"]).to_numpy()
        x = I.stretch(c["x"]).to_numpy()
        z = t
        hi, lo = (c["xv"], 100 - c["xv"]) if c["x"] == "xrsi" else (c["xv"], -c["xv"])
        tau = c["tau"]
        out = np.select([x >= hi, x <= lo, t >= tau, t <= -tau], [2, -2, 1, -1], 0)
        z = np.where(np.isfinite(x) & np.isfinite(t), 0.0, np.nan)
    ok = np.isfinite(z) & I.valid.to_numpy()
    return np.where(ok, out, -9).astype(np.int8)


# ---------------------------------------------------------------- 평가
class Fwd:
    def __init__(self, I: Ind, h):
        O = I.O
        f = (O.shift(-(1 + h)) / O.shift(-1) - 1).to_numpy()
        f = np.where(I.valid.to_numpy(), f, np.nan)
        self.raw = f
        mu = np.nanmean(np.where(np.isfinite(f), f, np.nan), axis=1, keepdims=True)
        self.ex = f - mu
        self.h = h


def period_rows(I: Ind, name, h):
    a, b, purge = PERIODS[name]
    idx = [i for i, d in enumerate(I.dates) if a <= d <= b]
    if purge:
        end = max(idx)
        idx = [i for i in idx if i + 1 + h <= end]
    return np.array(idx)


def nw_t(x, lag):
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else float("nan")


def evaluate(Z, F: Fwd, rows, full=False):
    Zr, ex, raw = Z[rows], F.ex[rows], F.raw[rows]
    have = np.isfinite(ex) & (Zr != -9)
    tot = have.sum()
    m, mr, hit, share = {}, {}, {}, {}
    for z in (2, 1, 0, -1, -2):
        M = have & (Zr == z)
        n = M.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m[z] = np.where(n > 0, np.where(M, ex, 0).sum(axis=1) / n, np.nan)
            mr[z] = np.where(n > 0, np.where(M, raw, 0).sum(axis=1) / n, np.nan)
            hit[z] = np.where(n > 0, (M & (raw > 0)).sum(axis=1) / n, np.nan)
        share[z] = float(M.sum() / tot) if tot else float("nan")
    terms = {"T1 상방-하방": m[1] - m[-1], "T2 극하방-하방": m[-2] - m[-1], "T3 상방-극상방": m[1] - m[2]}
    res = {"share": share, "terms": {k: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
                                     for k, v in terms.items()}}
    res["S"] = float(sum(res["terms"].values()))
    if full:
        res["t"] = {k: nw_t(v, F.h) for k, v in terms.items()}
        res["zone"] = {NAMES[z]: {"비중": share[z], "초과": float(np.nanmean(m[z])), "수익": float(np.nanmean(mr[z])),
                                  "상승확률": float(np.nanmean(hit[z])), "날짜수": int(np.isfinite(m[z]).sum())}
                       for z in (2, 1, 0, -1, -2)}
    return res


def eligible(r):
    s = r["share"]
    return 0.02 <= s[2] <= 0.20 and 0.02 <= s[-2] <= 0.20 and min(s[1], s[-1], s[0]) >= 0.10


# ---------------------------------------------------------------- 보조: 단계대로 사고팔기
def follow(I: Ind, Z, a, b):
    """+1·-2면 산다, -1·+2면 판다, 0이면 그대로. t일 종가 단계 → t+1 시가부터 t+2 시가까지 보유. 종목 같은 비중."""
    O = I.O.to_numpy()
    r = O[2:] / O[1:-1] - 1                                  # r[t] = t+1 시가 → t+2 시가
    rows = {i for i, d in enumerate(I.dates) if a <= d <= b and i + 2 < len(I.dates)}
    port, hold = [], []
    state = np.zeros(Z.shape[1])
    for i in range(len(I.dates) - 2):
        z = Z[i]
        state = np.where(z == -9, 0, np.where((z == 1) | (z == -2), 1, np.where((z == -1) | (z == 2), 0, state)))
        if i in rows:
            ok = np.isfinite(r[i]) & (z != -9)
            if ok.any():
                port.append(np.nanmean(np.where(ok, state * r[i], np.nan)))
                hold.append(np.nanmean(np.where(ok, r[i], np.nan)))
    def stat(x):
        v = np.cumprod(1 + np.array(x))
        return {"수익": float(v[-1] - 1), "낙폭": float((v / np.maximum.accumulate(np.r_[1, v])[1:] - 1).min())}
    return {"단계대로": stat(port), "계속 보유": stat(hold)}


# ---------------------------------------------------------------- 실행
def main():
    os.makedirs(OUT, exist_ok=True)
    I = Ind(*load())
    print(f"종목 {len(I.syms)}개, {I.dates[0]}~{I.dates[-1]}")
    F = {h: Fwd(I, h) for h in (5, 10, 20)}
    rows = {(p, h): period_rows(I, p, h) for p in PERIODS for h in (5, 10, 20)}
    G = grid()
    res = []
    for c in G:
        Z = zones(I, c)
        r = {"cfg": c, "name": cname(c)}
        for p in ("앞", "뒤", "2025"):
            r[p] = evaluate(Z, F[H], rows[p, H])
        r["eligible"] = eligible(r["앞"])
        res.append(r)
    ok = [r for r in res if r["eligible"]]
    ok.sort(key=lambda r: -r["앞"]["S"])
    lines = [f"격자 {len(G)}개, 자격 통과 {len(ok)}개 (h={H}, 초과 수익, 날짜 평균)"]
    hdr = f"{'설정':<44} {'앞 S':>7} {'T1':>6} {'T2':>6} {'T3':>6} | {'뒤 S':>7} {'T1':>6} {'T2':>6} {'T3':>6} | {'2025 S':>7}"
    lines += ["", "== 앞 구간 상위 15", hdr]
    pct = lambda v: f"{v * 100:+6.2f}"
    for r in ok[:15]:
        a, b, c = r["앞"], r["뒤"], r["2025"]
        lines.append(f"{r['name']:<44} {pct(a['S'])} {pct(a['terms']['T1 상방-하방'])} {pct(a['terms']['T2 극하방-하방'])} "
                     f"{pct(a['terms']['T3 상방-극상방'])} | {pct(b['S'])} {pct(b['terms']['T1 상방-하방'])} "
                     f"{pct(b['terms']['T2 극하방-하방'])} {pct(b['terms']['T3 상방-극상방'])} | {pct(c['S'])}")
    # 전체 격자의 항별 분포(사용자 가설이 격자 전반에서 성립하나)
    for p in ("앞", "뒤", "2025"):
        for k in ("T1 상방-하방", "T2 극하방-하방", "T3 상방-극상방"):
            v = np.array([r[p]["terms"][k] for r in ok])
            lines.append(f"[{p}] {k}: 양수 {np.mean(v > 0) * 100:4.0f}%  중앙 {np.median(v) * 100:+.2f}%  (자격 {len(v)}개)")
    best = ok[0]
    Zb = zones(I, best["cfg"])
    Z0 = zones(I, BASELINE)
    detail = {}
    for label, c, Z in (("선택", best["cfg"], Zb), ("옛칼만", BASELINE, Z0)):
        d = {}
        for p in PERIODS:
            for h in (5, 10, 20):
                d[f"{p}·h{h}"] = evaluate(Z, F[h], rows[p, h], full=True)
            a, b, _ = PERIODS[p]
            d[f"{p}·따라하기"] = follow(I, Z, a, b)
        detail[label] = {"name": cname(c), "cfg": c, "d": d}
    for label in ("선택", "옛칼만"):
        D = detail[label]
        lines += ["", f"== {label}: {D['name']}"]
        for p in PERIODS:
            e = D["d"][f"{p}·h10"]
            lines.append(f"[{p}] S {pct(e['S'])}  " + "  ".join(f"{k} {pct(v)} (t {e['t'][k]:+.1f})" for k, v in e["terms"].items()))
            lines.append("      " + "  ".join(f"{n} {z['비중'] * 100:4.1f}%·초과{pct(z['초과'])}·수익{pct(z['수익'])}·상승{z['상승확률'] * 100:3.0f}%"
                                       for n, z in e["zone"].items()))
            for h in (5, 20):
                e2 = D["d"][f"{p}·h{h}"]
                lines.append(f"      h{h}: S {pct(e2['S'])} " + " ".join(f"{k[:2]} {pct(v)}" for k, v in e2["terms"].items()))
            fl = D["d"][f"{p}·따라하기"]
            lines.append(f"      단계대로 사고팔기 {pct(fl['단계대로']['수익'])} (낙폭 {pct(fl['단계대로']['낙폭'])}) vs 계속 보유 "
                         f"{pct(fl['계속 보유']['수익'])} (낙폭 {pct(fl['계속 보유']['낙폭'])})")
    # 이웃(선택 설정에서 한 값만 한 칸 바꾼 것)
    def neighbors(c):
        out = []
        for g in G:
            if g["fam"] != c["fam"]:
                continue
            diff = [k for k in g if k != "fam" and g[k] != c[k]]
            if len(diff) == 1:
                out.append(g)
        return out
    nb = [r for r in res if r["cfg"] in neighbors(best["cfg"])]
    lines += ["", f"== 이웃 {len(nb)}개 (한 값만 다름)"]
    for r in nb:
        lines.append(f"{r['name']:<44} 앞 {pct(r['앞']['S'])} 뒤 {pct(r['뒤']['S'])} 2025 {pct(r['2025']['S'])} 자격 {r['eligible']}")
    # 반도체/비반도체
    from desk import config
    from desk.strategies import sector_of
    semis = np.array([sector_of(s, s) in config.SEMIS_SECTORS for s in I.syms])
    lines += ["", "== 선택 설정, 반도체 / 비반도체 (h10)"]
    for p in ("앞", "뒤", "2025"):
        out = []
        for lab, mask in (("반도체", semis), ("비반도체", ~semis)):
            Zm = np.where(mask[None, :], Zb, -9).astype(np.int8)
            e = evaluate(Zm, F[H], rows[p, H])
            out.append(f"{lab} S {pct(e['S'])} " + " ".join(f"{k[:2]} {pct(v)}" for k, v in e["terms"].items()))
        lines.append(f"[{p}] " + " | ".join(out))
    # 오늘 단계(참고)
    last = {I.syms[j]: NAMES.get(int(Zb[-1, j])) for j in range(len(I.syms)) if Zb[-1, j] != -9}
    cnt = {n: sum(1 for v in last.values() if v == n) for n in NAMES.values()}
    lines += ["", f"== {I.dates[-1]} 종가 단계 분포(선택 설정): {cnt}"]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "zones.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(OUT, "zones.json"), "w", encoding="utf-8") as f:
        json.dump({"grid": [{k: r[k] for k in ("name", "cfg", "eligible", "앞", "뒤", "2025")} for r in res],
                   "best": best["name"], "detail": detail, "last": last},
                  f, ensure_ascii=False, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())


if __name__ == "__main__":
    main()
