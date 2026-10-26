"""7단계 반박 검증 V_STKBUY(2026-10-11): 갈래 STKBUY 대표 후보 G1을 깨려고 시도한다.

후보 G1: 종목 종가가 하루 −15% 이하인 날 = 사건(같은 종목은 20거래일 이상 띄움). 1거래일 더 기다리고
        그 사이 또 −15%가 나오면 그날부터 다시 셈(C1). 다음 날 시가에 사서 63거래일 보유, 진입가 −25%에 장중 손절
        (시가가 선 아래면 시가).
주장(STKBUY): 사전 기준(같은 종목·같은 창 아무 날 대비) 2026 앞 +21.6%p(t 1.6, 149건) → 2026 뒤 +4.8%p(64건)
        · 2025 +7.5%p · 미끄러짐 0.5% 뒤 +4.9%p. d_adj 앞 +12.5 → 뒤 −14.1%p · 2025 +6.0%p. 2026 뒤 평균 −8.4%.

독립 재계산: 공유 엔진(lev5·trade_lab·stock_lab·harness)을 import하지 않는다. cache/p4_stocks.parquet만 읽어
  종목별 배열을 새로 만들고, 사건·진입·거래 수익·기준을 이 파일에서 다시 짠다. 표본 거래는 하루씩 도는
  순수 파이썬 루프로 한 번 더 대조한다.
1987·2000~02·2008·2011 시대: p4_stocks가 2014년부터라, 같은 종목표 중 2014-01 이전 상장 1,600종목의 1985~2014
  일봉을 yfinance로 새로 받아(cache/p7_V_STKBUY_old.parquet, 이 갈래 전용 새 파일) 2014 겹치는 구간으로 이어 붙였다.
  지금 살아 있는 종목만이라 생존편향은 2014 이후보다 더 크다(상한으로 읽는다).

사전 고정(확인 구간을 본 뒤 바꾸지 않음)
  - 이웃: 파라미터마다 G1 값 앞뒤 2칸, 하나씩만 바꿈(OAT)
      하루 하락 x 10·12.5·15·17.5·20% / 대기 N 0·1·2·3·5거래일 / 조건 C0·C1·C2 / 보유 H 21·42·63·84·105거래일
      / 손절 없음·15·20·25·30·35% / 사건 간격 10·15·20·25·30거래일  → 이웃 23개(G1 제외, H105는 2026 창에 거래 0)
  - 시점: 하루 늦춘 판(진입 e+1 시가, 같은 63일), 사건일 종가 다음 날 시가 대신 '대기 끝난 날 종가' 체결 판.
  - 미끄러짐 0.5%: (가) 원 규약(진입·손절 청산에만) (나) 왕복(진입·모든 청산).
  - '한 달' 정의: 보유 63거래일 vs 달력 3개월, 마삼룰식 대기 21거래일 vs 달력 1개월 + 1일(C1).
  - 위약: 같은 거래 수를 (가) 같은 종목·같은 창 무작위 날 (나) 같은 날·무작위 종목에 둔 판, 시드 20개.
  - 시대: 1987 · 2000~02 · 2008 · 2011 · 2018 4분기 · 2020 · 2022 · 2025 · 2026 앞 · 2026 뒤(진입일 기준).
  - 판정: 아래 judge()에 적은 문턱을 결과를 보기 전에 정했다.

실행: py p7_V_STKBUY.py            (전체 → results/p7_V_STKBUY.json; md는 json 숫자로 손으로 씀)
      py p7_V_STKBUY.py --grid     (보고용 x × N 격자를 json에 더함, 전체 실행 뒤)
      py p7_V_STKBUY.py --cal      (달력 포트폴리오 연율·낙폭을 json에 더함, 전체 실행 뒤)
      py p7_V_STKBUY.py --fetch    (1985~2014 일봉 받기, 한 번만)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
OLD = CACHE / "p7_V_STKBUY_old.parquet"
OUT_JSON = RESULTS / "p7_V_STKBUY.json"
OUT_MD = RESULTS / "p7_V_STKBUY.md"

# ============================================================================ 사전 고정
G1 = {"x": 0.15, "N": 1, "C": "C1", "H": 63, "stop": 0.25, "gap": 20}
NB = {"x": (0.10, 0.125, 0.15, 0.175, 0.20), "N": (0, 1, 2, 3, 5), "C": ("C0", "C1", "C2"),
      "H": (21, 42, 63, 84, 105), "stop": (None, 0.15, 0.20, 0.25, 0.30, 0.35), "gap": (10, 15, 20, 25, 30)}
SLIP = 0.005
SEEDS = 20
HIST = 252                     # 사건 판단에 필요한 이력(s ≥ 252), 기준 진입은 e ≥ 253
CLAIM = {"앞_d": 0.216, "앞_t": 1.6, "앞_n": 149, "뒤_d": 0.048, "뒤_n": 64, "y25_d": 0.075, "slip_뒤_d": 0.049,
         "앞_adj": 0.125, "뒤_adj": -0.141, "y25_adj": 0.060, "뒤_mean": -0.084}
# 창: 이름 → (진입 시작, 진입 끝, 청산 마감 또는 None)
WIN = {
    "2026앞": ("2026-01-02", "2026-05-29", "2026-05-29"),
    "2026뒤": ("2026-06-01", "2026-10-09", None),
    "2025": ("2025-01-01", "2025-12-31", None),
    "2022": ("2022-01-01", "2022-12-31", None),
    "2020": ("2020-01-01", "2020-12-31", None),
    "2015~2019": ("2015-01-01", "2019-12-31", "2019-12-31"),
    "2015~2025": ("2015-01-01", "2025-12-31", None),
}
ERAS = {
    "1987": ("1987-01-01", "1987-12-31"),
    "1987 10~12월": ("1987-10-01", "1987-12-31"),
    "2000~02": ("2000-01-01", "2002-12-31"),
    "2008": ("2008-01-01", "2008-12-31"),
    "2008 9월~09 3월": ("2008-09-01", "2009-03-31"),
    "2011": ("2011-01-01", "2011-12-31"),
    "2018 4분기": ("2018-10-01", "2018-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
    "2026 앞": ("2026-01-02", "2026-03-03"),
    "2026 뒤": ("2026-06-01", "2026-10-09"),
    "긴 역사 앞 1986~2005": ("1986-01-01", "2005-12-31"),
    "긴 역사 뒤 2006~2025": ("2006-01-01", "2025-12-31"),
}


# ============================================================================ 자료
class Stk:
    __slots__ = ("sym", "days", "o", "h", "l", "c")

    def __init__(self, sym, days, o, h, l, c):
        self.sym, self.days, self.o, self.h, self.l, self.c = sym, days, o, h, l, c

    def pos(self, day, side="left"):
        return int(np.searchsorted(self.days, day, side=side))


def _arrays(df: pd.DataFrame) -> dict:
    out = {}
    df = df.sort_values(["sym", "date"])
    for sym, g in df.groupby("sym", sort=False):
        g = g.dropna(subset=["Open", "High", "Low", "Close"])
        g = g[(g.Open > 0) & (g.High > 0) & (g.Low > 0) & (g.Close > 0)]
        if len(g) < 260:
            continue
        f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy()
        out[sym] = Stk(sym, g.date.to_numpy().astype(str), g.Open.to_numpy() * f, g.High.to_numpy() * f,
                       g.Low.to_numpy() * f, g.Close.to_numpy() * f)
    return out


def suspicious(c) -> bool:
    r = c[1:] / c[:-1] - 1
    return bool((r > 1.5).any() or (r < -0.7).any())


def load_new():
    raw = _arrays(pd.read_parquet(CACHE / "p4_stocks.parquet"))
    syms = sorted(raw)
    bad = [s for s in syms if suspicious(raw[s].c)]
    good = [raw[s] for s in syms if s not in set(bad)]
    return good, [raw[s] for s in bad], len(syms)


def load_spliced(new_list):
    """1985~2013(옛 자료) + 2014~(p4) 이어 붙이기. 겹치는 2014년 종가 비율 중앙값으로 옛 자료를 맞춘다.
    옛 구간에 자료 오류 의심(하루 +150%/−70%)이 있거나 겹침 비율이 흔들리면(사분위 폭 > 1%) 이어 붙이지 않는다."""
    if not OLD.exists():
        return None, {}
    old = _arrays(pd.read_parquet(OLD))
    out, info = [], {"spliced": 0, "old_bad": 0, "old_mismatch": 0, "new_only": 0}
    for s in new_list:
        g = old.get(s.sym)
        if g is None:
            out.append(s); info["new_only"] += 1
            continue
        ov = np.intersect1d(g.days, s.days)
        ov = ov[ov < "2015-01-01"]
        if len(ov) < 100:
            out.append(s); info["new_only"] += 1
            continue
        ig, isn = np.searchsorted(g.days, ov), np.searchsorted(s.days, ov)
        ratio = s.c[isn] / g.c[ig]
        q1, med, q3 = np.quantile(ratio, [0.25, 0.5, 0.75])
        if (q3 - q1) / med > 0.01:
            out.append(s); info["old_mismatch"] += 1
            continue
        k = g.days < s.days[0]
        if suspicious(g.c[k] * med) or k.sum() < 1:
            out.append(s); info["old_bad"] += 1 if k.sum() >= 1 else 0
            if k.sum() < 1:
                info["new_only"] += 1
            continue
        # 이음매(옛 마지막 날 → 새 첫날) 하루 변화도 자료 오류 문턱 안이어야 한다
        cc = np.concatenate([g.c[k] * med, s.c])
        if suspicious(cc):
            out.append(s); info["old_bad"] += 1
            continue
        out.append(Stk(s.sym, np.concatenate([g.days[k], s.days]), np.concatenate([g.o[k] * med, s.o]),
                       np.concatenate([g.h[k] * med, s.h]), np.concatenate([g.l[k] * med, s.l]), cc))
        info["spliced"] += 1
    return out, info


# ============================================================================ 거래 수익
def ret_all(s: Stk, H: int, stop, slip=0.0, rt=False):
    """모든 진입일 e의 H거래일 거래 수익(e일 시가 진입). e + H − 1 ≥ n이면 NaN.
    stop: 진입가 대비 장중 손절(시가 ≤ 선이면 시가, 저가 ≤ 선이면 선). slip: 진입가 ×(1+slip), 손절 청산 ×(1−slip).
    rt=True면 기간 끝 종가 청산에도 ×(1−slip)(왕복)."""
    o, l, c = s.o, s.l, s.c
    n = len(c)
    out = np.full(n, np.nan)
    m = n - H + 1
    if m <= 0:
        return out
    entry = o[:m] * (1 + slip)
    cend = c[H - 1:H - 1 + m] * ((1 - slip) if rt else 1.0)
    if stop is None:
        out[:m] = cend / entry - 1
        return out
    L = entry * (1 - stop)
    ow, lw = swv(o, H)[:m], swv(l, H)[:m]
    hit = (ow <= L[:, None]) | (lw <= L[:, None])
    anyh = hit.any(1)
    t = hit.argmax(1)
    ot = ow[np.arange(m), t]
    px = np.where(ot <= L, ot, L) * (1 - slip)
    out[:m] = np.where(anyh, px, cend) / entry - 1
    return out


def mae_all(s: Stk, H: int):
    n = len(s.c)
    out = np.full(n, np.nan)
    m = n - H + 1
    if m > 0:
        out[:m] = swv(s.l, H)[:m].min(1) / s.o[:m] - 1
    return out


def loop_trade(s: Stk, e: int, H: int, stop, slip=0.0):
    """하루씩 도는 순수 파이썬 대조용."""
    if e + H - 1 >= len(s.c):
        return float("nan")
    entry = float(s.o[e]) * (1 + slip)
    if stop is not None:
        L = entry * (1 - stop)
        for k in range(e, e + H):
            if s.o[k] <= L:
                return float(s.o[k]) * (1 - slip) / entry - 1
            if s.l[k] <= L:
                return L * (1 - slip) / entry - 1
    return float(s.c[e + H - 1]) / entry - 1


# ============================================================================ 사건 → 진입
def raw_events(s: Stk, x: float):
    c = s.c
    n = len(c)
    r = np.full(n, np.nan)
    r[1:] = c[1:] / c[:-1] - 1
    idx = np.arange(n)
    return np.flatnonzero((idx >= HIST) & (idx <= n - 2) & (r <= -x))


def first_of_episodes(R, gap):
    P, last = [], None
    for v in R:
        if last is None or v - last >= gap:
            P.append(int(v))
            last = v
    return P


def entries(s: Stk, x, N, C, gap, cal_wait=False):
    """(s0, e): s0 = 사건 1번의 첫 사건일, e = 진입일(시가). 하루씩 앞으로 보며 판단(미래 정보 없음).
    cal_wait=True: 대기 = 달력 1개월 + 1일(마삼룰), C1 재발도 달력 1개월 안에서 센다."""
    R = raw_events(s, x)
    if not len(R):
        return np.array([], int), np.array([], int)
    n = len(s.c)
    israw = np.zeros(n, bool)
    israw[R] = True
    P = first_of_episodes(R, gap)
    s0l, el = [], []
    seen = set()
    for p in P:
        if cal_wait:
            t = p
            while True:
                lim = (pd.Timestamp(str(s.days[t])) + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
                k = t + 1
                nxt = None
                while k < n and s.days[k] <= lim:
                    if C == "C1" and israw[k]:
                        nxt = k
                        break
                    k += 1
                if nxt is None:
                    break
                t = nxt
            # 마지막 사건일 + 1개월 + 1일 이후 첫 거래일 시가
            tgt = (pd.Timestamp(str(s.days[t])) + pd.DateOffset(months=1) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            e = s.pos(tgt)
        elif N == 0 or C == "C0":
            e = p + 1 + N
        elif C == "C1":
            t, k = p, p + 1
            while k <= t + N and k < n:
                if israw[k]:
                    t = k
                k += 1
            e = t + 1 + N
        elif C == "C2":
            if p + N >= n or not (s.c[p + N] <= s.c[p]):
                continue
            e = p + 1 + N
        else:
            raise ValueError(C)
        if e > n - 1 or e in seen:
            continue
        seen.add(e)
        s0l.append(p)
        el.append(e)
    return np.array(s0l, int), np.array(el, int)


# ============================================================================ 기준·통계
def win_bounds(s: Stk, w, H):
    d0, d1, ex = WIN[w]
    a = max(s.pos(d0), HIST + 1)
    z = s.pos(d1, "right")
    if ex is not None:
        z = min(z, s.pos(ex, "right") - H + 1)
    return a, z


def tstat_month(d, months):
    if len(d) < 3:
        return float("nan")
    u, inv = np.unique(months, return_inverse=True)
    if len(u) < 3:
        return float("nan")
    g = np.bincount(inv, d) / np.bincount(inv)
    sd = g.std(ddof=1)
    return float(g.mean() / (sd / np.sqrt(len(g)))) if sd > 0 else float("nan")


class Ctx:
    """한 (H, stop, slip, rt) 조합의 종목별 수익, 같은 날 전 종목 평균(xs), 종목 평소 초과(alpha)."""

    def __init__(self, stocks, H, stop, slip=0.0, rt=False, alpha_from="2015-01-01", calH=False):
        self.stocks, self.H = stocks, H
        t0 = time.time()
        if calH:
            self.R = [cal_ret(s, stop) for s in stocks]
        else:
            self.R = [ret_all(s, H, stop, slip, rt) for s in stocks]
        alld = np.unique(np.concatenate([s.days for s in stocks]))
        self.G = alld
        self.gi = [np.searchsorted(alld, s.days) for s in stocks]
        S = np.zeros(len(alld)); Cn = np.zeros(len(alld))
        for R, gi in zip(self.R, self.gi):
            v = R.copy()
            v[:HIST + 1] = np.nan
            k = np.isfinite(v)
            S += np.bincount(gi[k], v[k], minlength=len(alld)); Cn += np.bincount(gi[k], minlength=len(alld))
        self.xs = np.where(Cn > 0, S / np.maximum(Cn, 1), np.nan)
        self.alpha = np.full(len(stocks), np.nan)
        for i, (s, R, gi) in enumerate(zip(stocks, self.R, self.gi)):
            lo = max(s.pos(alpha_from), HIST + 1)
            v = R[lo:] - self.xs[gi[lo:]]
            v = v[np.isfinite(v)]
            if len(v):
                self.alpha[i] = v.mean()
        self.sec = time.time() - t0

    def base(self, i, w):
        a, z = win_bounds(self.stocks[i], w, self.H)
        if z <= a:
            return float("nan")
        v = self.R[i][a:z]
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")

    def pooled(self, w):
        tot = cnt = 0.0
        for i, s in enumerate(self.stocks):
            a, z = win_bounds(s, w, self.H)
            if z <= a:
                continue
            v = self.R[i][a:z]
            v = v[np.isfinite(v)]
            tot += v.sum(); cnt += len(v)
        return tot / cnt if cnt else float("nan")


def cal_ret(s: Stk, stop):
    """달력 3개월 보유: e일 시가 진입 → (e일 + 3개월) 전 마지막 거래일 종가. 손절은 같은 규칙."""
    n = len(s.c)
    tgt = (pd.to_datetime(s.days) + pd.DateOffset(months=3)).strftime("%Y-%m-%d").to_numpy()
    last = np.searchsorted(s.days, tgt, side="left") - 1          # 목표일 전 마지막 거래일
    Hs = last - np.arange(n) + 1
    out = np.full(n, np.nan)
    ok = last < n - 0                                              # 목표일이 자료 끝을 넘으면 NaN
    ok &= np.searchsorted(s.days, tgt, side="left") < n
    for H in np.unique(Hs[ok]):
        if H < 2:
            continue
        RH = ret_all(s, int(H), stop)
        k = ok & (Hs == H)
        out[k] = RH[k]
    return out


def trades_in(ctx: Ctx, ent, w, shift=0):
    """창 w 안 거래: (종목 i, s0, e) 목록 → 배열들. shift: 진입을 shift거래일 늦춤."""
    d0, d1, ex = WIN[w]
    rows = []
    for i, (s0s, es) in ent.items():
        s = ctx.stocks[i]
        for s0, e in zip(s0s, es):
            e = e + shift
            if e > len(s.c) - 1:
                continue
            z = e + ctx.H - 1
            if not np.isfinite(ctx.R[i][e]):
                continue
            dd = s.days[e]
            if not (d0 <= dd <= d1):
                continue
            if ex is not None and s.days[min(z, len(s.c) - 1)] > ex:
                continue
            rows.append((i, s0, e))
    return rows


def cell(ctx: Ctx, rows, w, mae=None, imm_ctx=None):
    if not rows:
        return {"n": 0}
    i = np.array([r[0] for r in rows]); s0 = np.array([r[1] for r in rows]); e = np.array([r[2] for r in rows])
    ret = np.array([ctx.R[a][b] for a, b in zip(i, e)])
    base = np.array([ctx.base(a, w) for a in i])
    gd = np.array([ctx.gi[a][b] for a, b in zip(i, e)])
    xs = ctx.xs[gd]
    al = ctx.alpha[i]
    imm = np.array([ctx.R[a][b + 1] for a, b in zip(i, s0)])
    months = np.array([int(ctx.G[g][:4]) * 12 + int(ctx.G[g][5:7]) for g in gd])
    d = ret - base
    adj = ret - xs - al
    ka = np.isfinite(adj)
    ki = np.isfinite(imm)
    out = {"n": int(len(ret)), "nsym": int(len(np.unique(i))), "ndate": int(len(np.unique(gd))),
           "mean": float(ret.mean()), "med": float(np.median(ret)), "win": float((ret > 0).mean()),
           "base": float(np.nanmean(base)), "d_any": float(np.nanmean(d)), "t_any": tstat_month(d, months),
           "d_adj": float(adj[ka].mean()) if ka.any() else float("nan"),
           "t_adj": tstat_month(adj[ka], months[ka]) if ka.any() else float("nan"),
           "d_imm": float((ret - imm)[ki].mean()) if ki.any() else float("nan"),
           "pooled": ctx.pooled(w) if w in WIN else float("nan")}
    if mae is not None:
        mv = np.array([mae[a][b] for a, b in zip(i, e)])
        out["p20"] = float((mv <= -0.2).mean())
    return out


# ============================================================================ 실행 조각
def build_entries(stocks, x, N, C, gap, cal_wait=False):
    ent = {}
    for i, s in enumerate(stocks):
        s0, e = entries(s, x, N, C, gap, cal_wait)
        if len(e):
            ent[i] = (s0, e)
    return ent


def era_window_add(name, d0, d1):
    WIN[name] = (d0, d1, None)


def placebo(ctx: Ctx, rows, w, seeds=SEEDS):
    """(가) 같은 종목·같은 창 무작위 날 (나) 같은 날·무작위 종목. 반환: 실제 평균, 위약 평균 목록 둘."""
    if not rows:
        return None
    actual = float(np.mean([ctx.R[i][e] for i, _, e in rows]))
    # (가)
    pools = {}
    for i, _, _ in rows:
        if i in pools:
            continue
        a, z = win_bounds(ctx.stocks[i], w, ctx.H)
        cand = np.arange(a, max(z, a))
        cand = cand[np.isfinite(ctx.R[i][cand])] if len(cand) else cand
        pools[i] = cand
    # (나) 날짜별 후보(종목 i, e)
    want = {ctx.gi[i][e] for i, _, e in rows}
    bydate = {g: [] for g in want}
    for k, (s, R, gi) in enumerate(zip(ctx.stocks, ctx.R, ctx.gi)):
        m = np.isin(gi, list(want))
        m[:HIST + 1] = False
        for p in np.flatnonzero(m & np.isfinite(R)):
            bydate[gi[p]].append(R[p])
    bydate = {g: np.array(v) for g, v in bydate.items()}
    A, B = [], []
    for sd in range(seeds):
        rng = np.random.default_rng(1000 + sd)
        va = [ctx.R[i][rng.choice(pools[i])] for i, _, _ in rows if len(pools[i])]
        vb = [rng.choice(bydate[ctx.gi[i][e]]) for i, _, e in rows if len(bydate[ctx.gi[i][e]])]
        A.append(float(np.mean(va))); B.append(float(np.mean(vb)))
    return {"actual": actual, "same_stock": A, "same_date": B,
            "beat_same_stock": int(sum(actual > v for v in A)), "beat_same_date": int(sum(actual > v for v in B))}


def run():
    t00 = time.time()
    J = {"claim": CLAIM, "g1": G1, "nb": {k: [str(v) for v in vs] for k, vs in NB.items()}}
    stocks, badst, nraw = load_new()
    J["universe"] = {"loaded_260": nraw, "bad": len(badst), "used": len(stocks)}
    print("종목", nraw, "자료 오류 의심", len(badst), "사용", len(stocks), flush=True)

    # ---- 1) 재현
    ctx = Ctx(stocks, 63, 0.25)
    mae = [mae_all(s, 63) for s in stocks]
    ent = build_entries(stocks, 0.15, 1, "C1", 20)
    rep = {}
    for w in ("2026앞", "2026뒤", "2025", "2022", "2020", "2015~2019", "2015~2025"):
        rep[w] = cell(ctx, trades_in(ctx, ent, w), w, mae)
    ctx_s = Ctx(stocks, 63, 0.25, SLIP)
    ctx_rt = Ctx(stocks, 63, 0.25, SLIP, rt=True)
    rep["slip_2026뒤"] = cell(ctx_s, trades_in(ctx_s, ent, "2026뒤"), "2026뒤")
    rep["slip_2026앞"] = cell(ctx_s, trades_in(ctx_s, ent, "2026앞"), "2026앞")
    rep["slip_2025"] = cell(ctx_s, trades_in(ctx_s, ent, "2025"), "2025")
    rep["rt_2026뒤"] = cell(ctx_rt, trades_in(ctx_rt, ent, "2026뒤"), "2026뒤")
    rep["rt_2026앞"] = cell(ctx_rt, trades_in(ctx_rt, ent, "2026앞"), "2026앞")
    rep["rt_2025"] = cell(ctx_rt, trades_in(ctx_rt, ent, "2025"), "2025")
    J["repro"] = rep
    print("재현", {w: (rep[w].get("n"), round(rep[w].get("d_any", np.nan) * 100, 2), round(rep[w].get("d_adj", np.nan) * 100, 2),
                     round(rep[w].get("mean", np.nan) * 100, 2)) for w in rep}, flush=True)

    # 하루씩 루프 대조(표본 300건)
    rng = np.random.default_rng(3)
    allrows = [(i, s0, e) for i, (s0s, es) in ent.items() for s0, e in zip(s0s, es) if np.isfinite(ctx.R[i][e])]
    pick = rng.choice(len(allrows), min(300, len(allrows)), replace=False)
    err = max(abs(loop_trade(stocks[allrows[k][0]], allrows[k][2], 63, 0.25) - ctx.R[allrows[k][0]][allrows[k][2]]) for k in pick)
    J["loop_check_maxerr"] = float(err)
    print("루프 대조 최대 오차", err, flush=True)

    # 2026 앞·뒤 거래 몰림(날짜·종목)
    conc = {}
    for w in ("2026앞", "2026뒤"):
        rows = trades_in(ctx, ent, w)
        d = pd.Series([stocks[i].days[e] for i, _, e in rows])
        ev = pd.Series([stocks[i].days[s0] for i, s0, _ in rows])
        r = pd.Series([ctx.R[i][e] for i, _, e in rows])
        top = ev.value_counts().head(5)
        bymon = pd.DataFrame({"m": d.str[:7], "r": r}).groupby("m").r.agg(["count", "mean"])
        conc[w] = {"event_dates": int(ev.nunique()), "top_event_dates": {k: int(v) for k, v in top.items()},
                   "top5_share": float(top.sum() / len(ev)), "by_entry_month": {k: [int(a), float(b)] for k, (a, b) in bymon.iterrows()},
                   "best5": [(stocks[i].sym, stocks[i].days[e], float(ctx.R[i][e])) for i, _, e in sorted(rows, key=lambda q: -ctx.R[q[0]][q[2]])[:5]],
                   "mean_ex_top5": float(np.sort(r.to_numpy())[:-5].mean()) if len(r) > 5 else float("nan")}
    J["concentration"] = conc
    # 표본 견고성: 최대 기여 1종목 빼기, 5% 절사, 사건일 묶음 부트스트랩, d_any = 0 비율(기준이 손절 −25% 자체), 손절 비율
    ex = {}
    for w in ("2026앞", "2026뒤"):
        rows = trades_in(ctx, ent, w)
        ret = np.array([ctx.R[i][e] for i, _, e in rows])
        base = np.array([ctx.base(i, w) for i, _, _ in rows])
        gd = np.array([ctx.gi[i][e] for i, _, e in rows])
        adj = ret - ctx.xs[gd] - ctx.alpha[[i for i, _, _ in rows]]
        d = ret - base
        top = int(np.argmax(ret))
        keep = np.arange(len(ret)) != top
        q = np.sort(d); k = int(len(q) * 0.05)
        ev = np.array([stocks[i].days[s0] for i, s0, _ in rows]); u = np.unique(ev)
        rb = np.random.default_rng(0)
        Bd, Ba = [], []
        for _ in range(2000):
            pk = rb.choice(u, len(u))
            Bd.append(np.concatenate([d[ev == x] for x in pk]).mean())
            Ba.append(np.nanmean(np.concatenate([adj[ev == x] for x in pk])))
        ex[w] = {"top_sym": stocks[rows[top][0]].sym, "top_ret": float(ret[top]), "d_any_ex_top": float(d[keep].mean()),
                 "d_adj_ex_top": float(np.nanmean(adj[keep])), "d_any_trim5": float(q[k:len(q) - k].mean()),
                 "d_any_median": float(np.median(d)), "d_adj_median": float(np.nanmedian(adj)),
                 "boot_d_any_5_50_95": [float(v) for v in np.percentile(Bd, [5, 50, 95])],
                 "boot_d_adj_5_50_95": [float(v) for v in np.percentile(Ba, [5, 50, 95])],
                 "share_d_zero": float((np.abs(d) < 1e-9).mean()), "share_stop": float((ret <= -0.2499).mean()),
                 "share_base_le_m20": float((base <= -0.20).mean())}
    J["sample_checks"] = ex

    # ---- 2) 시점·비용·'한 달'
    tim = {}
    for w in ("2026앞", "2026뒤", "2025", "2015~2025"):
        tim[f"delay1_{w}"] = cell(ctx, trades_in(ctx, ent, w, shift=1), w)
    # 대기 끝난 날 종가 체결 = e−1일 종가 진입: 수익 = (청산가)/(c[e−1]) − 1. 63일 보유는 e−1..e+61.
    ctx_c = CtxClose(stocks, 63, 0.25)
    for w in ("2026앞", "2026뒤", "2025", "2015~2025"):
        tim[f"closefill_{w}"] = cell(ctx_c, trades_in(ctx_c, {i: (s0, e - 1) for i, (s0, e) in ent.items()}, w), w)
    # 자료 오류 의심 71종목 포함(미래 정보로 거른 것 되돌리기)
    st_all = stocks + badst
    ctx_all = Ctx(st_all, 63, 0.25)
    ent_all = build_entries(st_all, 0.15, 1, "C1", 20)
    for w in ("2026앞", "2026뒤", "2025", "2015~2025"):
        tim[f"withbad_{w}"] = cell(ctx_all, trades_in(ctx_all, ent_all, w), w)
    del ctx_all
    # 달력 3개월 보유
    ctx_cal = Ctx(stocks, 63, 0.25, calH=True)
    for w in ("2026앞", "2026뒤", "2025", "2015~2025"):
        tim[f"cal3m_{w}"] = cell(ctx_cal, trades_in(ctx_cal, ent, w), w)
    del ctx_cal
    # 마삼룰식 대기: 21거래일 vs 달력 1개월 + 1일 (C1), 63일 · 고정 −25%
    ent21 = build_entries(stocks, 0.15, 21, "C1", 20)
    entcal = build_entries(stocks, 0.15, 21, "C1", 20, cal_wait=True)
    for w in ("2026앞", "2026뒤", "2025", "2015~2025"):
        tim[f"wait21_{w}"] = cell(ctx, trades_in(ctx, ent21, w), w)
        tim[f"waitcal_{w}"] = cell(ctx, trades_in(ctx, entcal, w), w)
    J["timing"] = tim
    print("시점·비용 끝", round(time.time() - t00), flush=True)

    # ---- 3) 이웃(OAT)
    nb = []
    ctx_cache = {(63, 0.25): ctx}

    def get_ctx(H, stop):
        if (H, stop) not in ctx_cache:
            ctx_cache[(H, stop)] = Ctx(stocks, H, stop)
        return ctx_cache[(H, stop)]

    for p, vals in NB.items():
        for v in vals:
            if v == G1[p]:
                continue
            q = dict(G1); q[p] = v
            cx = get_ctx(q["H"], q["stop"])
            en = ent if all(q[k] == G1[k] for k in ("x", "N", "C", "gap")) else build_entries(stocks, q["x"], q["N"], q["C"], q["gap"])
            row = {"param": p, "value": str(v)}
            for w in ("2026앞", "2026뒤", "2025", "2015~2025"):
                row[w] = cell(cx, trades_in(cx, en, w), w)
            nb.append(row)
            print("이웃", p, v, {w: round(row[w].get("d_any", np.nan) * 100, 1) for w in ("2026앞", "2026뒤", "2025")}, flush=True)
            if (q["H"], q["stop"]) != (63, 0.25):
                del ctx_cache[(q["H"], q["stop"])]
    J["neighbors"] = nb

    # ---- 4) 위약
    pl = {}
    for w in ("2026앞", "2026뒤", "2025", "2022", "2020", "2015~2025"):
        pl[w] = placebo(ctx, trades_in(ctx, ent, w), w)
        print("위약", w, pl[w]["beat_same_stock"], pl[w]["beat_same_date"], flush=True)
    J["placebo"] = pl
    del ctx_s, ctx_rt, ctx_cache, ctx, mae

    # ---- 5) 시대(이어 붙인 1985~ 자료)
    sp, info = load_spliced(stocks)
    J["splice"] = info
    if sp is not None:
        for name, (d0, d1) in ERAS.items():
            WIN["E:" + name] = (d0, d1, None)
        ctxL = Ctx(sp, 63, 0.25, alpha_from="1986-01-01")
        ctxL0 = Ctx(sp, 63, None, alpha_from="1986-01-01")
        maeL = [mae_all(s, 63) for s in sp]
        entL = build_entries(sp, 0.15, 1, "C1", 20)
        entL0 = build_entries(sp, 0.15, 0, "C0", 20)
        er = {}
        for name in ERAS:
            w = "E:" + name
            d0, d1 = ERAS[name]
            er[name] = {"g1": cell(ctxL, trades_in(ctxL, entL, w), w, maeL),
                        "imm": cell(ctxL, trades_in(ctxL, entL0, w), w, maeL),
                        "g1_hold": cell(ctxL0, trades_in(ctxL0, entL, w), w),
                        "nstocks": int(sum(1 for s in sp if s.days[0] <= d1 and s.days[-1] >= d0 and s.pos(d1, "right") - max(s.pos(d0), HIST + 1) > 0)),
                        "base_rate_pooled": ctxL.pooled(w)}
            print("시대", name, er[name]["g1"].get("n"), round(er[name]["g1"].get("mean", np.nan) * 100, 1),
                  round(er[name]["g1"].get("d_any", np.nan) * 100, 1), flush=True)
        J["eras"] = er
        pl2 = {}
        for name in ("1987", "1987 10~12월", "2000~02", "2008", "2008 9월~09 3월", "2011", "2018 4분기", "긴 역사 앞 1986~2005", "긴 역사 뒤 2006~2025"):
            w = "E:" + name
            rows = trades_in(ctxL, entL, w)
            if len(rows) >= 10:
                pl2[name] = placebo(ctxL, rows, w)
        J["placebo_eras"] = pl2
    J["sec"] = round(time.time() - t00)
    J["verdict"] = judge(J)
    OUT_JSON.write_text(json.dumps(J, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_md(J)
    print("끝", J["sec"], "초", J["verdict"], flush=True)


class CtxClose(Ctx):
    """대기 끝난 날(e−1) 종가 진입 판: e' = e−1을 진입일로 쓰되 진입가 = c[e'] (시가 대신 종가)."""

    def __init__(self, stocks, H, stop):
        self.stocks, self.H = stocks, H
        self.R = []
        for s in stocks:
            n = len(s.c)
            out = np.full(n, np.nan)
            m = n - H
            if m > 0:
                entry = s.c[:m]
                L = entry * (1 - stop)
                ow, lw = swv(s.o[1:], H)[:m], swv(s.l[1:], H)[:m]       # 다음 날부터 H일
                hit = (ow <= L[:, None]) | (lw <= L[:, None])
                anyh = hit.any(1); t = hit.argmax(1)
                ot = ow[np.arange(m), t]
                px = np.where(ot <= L, ot, L)
                cend = s.c[H:H + m]
                out[:m] = np.where(anyh, px, cend) / entry - 1
            self.R.append(out)
        alld = np.unique(np.concatenate([s.days for s in stocks]))
        self.G = alld
        self.gi = [np.searchsorted(alld, s.days) for s in stocks]
        S = np.zeros(len(alld)); Cn = np.zeros(len(alld))
        for R, gi in zip(self.R, self.gi):
            v = R.copy(); v[:HIST + 1] = np.nan
            k = np.isfinite(v)
            S += np.bincount(gi[k], v[k], minlength=len(alld)); Cn += np.bincount(gi[k], minlength=len(alld))
        self.xs = np.where(Cn > 0, S / np.maximum(Cn, 1), np.nan)
        self.alpha = np.full(len(stocks), np.nan)
        for i, (s, R, gi) in enumerate(zip(stocks, self.R, self.gi)):
            lo = max(s.pos("2015-01-01"), HIST + 1)
            v = R[lo:] - self.xs[gi[lo:]]
            v = v[np.isfinite(v)]
            if len(v):
                self.alpha[i] = v.mean()

    def base(self, i, w):
        d0, d1, ex = WIN[w]
        s = self.stocks[i]
        a = max(s.pos(d0), HIST + 1)
        z = s.pos(d1, "right")
        if ex is not None:
            z = min(z, s.pos(ex, "right") - self.H)
        if z <= a:
            return float("nan")
        v = self.R[i][a:z]
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")

    def pooled(self, w):
        return float("nan")


# ============================================================================ 판정(사전 고정)
def judge(J):
    """사전 문턱.
    refuted  : (A) 2026 뒤 평균 수익 < 같은 창 무조건 기저율 이고 d_adj < 0, 그리고
               (B) 위약 '같은 날·무작위 종목'을 2026 뒤에서 20개 중 15개 이상 이기지 못함
               → 사전 기준 +는 기준 구조(같은 창에 폭락 자체를 품음)에서 나온 것.
    robust   : 재현 0.5%p 안 + 2026 뒤·2025 d_any > 0이 하루 늦춤·왕복 미끄러짐에서도 유지
               + 이웃 23개 중 70% 이상이 2026 뒤·2025 둘 다 d_any > 0 + 위약 두 종류를 2026 뒤에서 18/20 이상
               + 2026 뒤 d_adj ≥ 0.
    fragile  : 그 밖."""
    r = J["repro"]
    w = r["2026뒤"]
    pl = J["placebo"]["2026뒤"]
    A = (w["mean"] < w["pooled"]) and (w["d_adj"] < 0)
    B = pl["beat_same_date"] < 15
    if A and B:
        return "refuted"
    nb = J["neighbors"]
    ok = sum(1 for x in nb if x["2026뒤"].get("d_any", -1) > 0 and x["2025"].get("d_any", -1) > 0)
    t = J["timing"]
    robust = (abs(w["d_any"] - CLAIM["뒤_d"]) < 0.005 and t["delay1_2026뒤"].get("d_any", -1) > 0
              and t["delay1_2025"].get("d_any", -1) > 0 and r["rt_2026뒤"].get("d_any", -1) > 0
              and ok / len(nb) >= 0.7 and pl["beat_same_stock"] >= 18 and pl["beat_same_date"] >= 18 and w["d_adj"] >= 0)
    return "confirmed_robust" if robust else "confirmed_fragile"


# ============================================================================ 보고서
def pct(x, d=1, sign=True):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def pp(x, d=1):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:+.{d}f}"


def write_md(J):
    pass  # 아래 md는 결과를 본 뒤 손으로 쓴다(숫자는 json에서 옮김).


def fetch_old():
    """p4 종목표 중 2014-01-03 이전부터 자료가 있는 종목의 1985~2014 일봉(분할 수정, Adj Close 배당 수정)."""
    import yfinance as yf
    st = pd.read_parquet(CACHE / "p4_stocks.parquet", columns=["date", "sym"])
    first = st.groupby("sym").date.min()
    syms = sorted(first[first <= "2014-01-03"].index)
    parts = []
    for i in range(0, len(syms), 150):
        chunk = syms[i:i + 150]
        for _ in range(3):
            try:
                f = yf.download(chunk, start="1985-01-01", end="2015-01-01", auto_adjust=False, progress=False,
                                threads=True, group_by="column")
                break
            except Exception as e:
                print("재시도", i, type(e).__name__, e, flush=True)
                time.sleep(5)
        else:
            continue
        cols = [f[c].stack(future_stack=True).rename(c) for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume")]
        df = pd.concat(cols, axis=1).reset_index()
        df.columns = ["date", "sym", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        parts.append(df.dropna(subset=["Close"]))
        print(f"{i + len(chunk)}/{len(syms)}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_parquet(OLD)
    print("저장", OLD, out.shape, out.sym.nunique())


GRID_X = (0.10, 0.125, 0.15, 0.175, 0.20)
GRID_N = (0, 1, 2, 3, 5, 21)


def grid_run():
    """보고용 2차원 격자(판정에 쓰지 않음): 하루 하락 x × 대기 N(C1), 63일 · 고정 −25%. json에 'grid'로 더한다."""
    J = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    stocks, _, _ = load_new()
    ctx = Ctx(stocks, 63, 0.25)
    out = {}
    for x in GRID_X:
        for N in GRID_N:
            en = build_entries(stocks, x, N, "C1", 20)
            out[f"{x}|{N}"] = {w: cell(ctx, trades_in(ctx, en, w), w) for w in ("2026앞", "2026뒤", "2025", "2015~2025")}
            print("격자", x, N, {w: round(out[f"{x}|{N}"][w].get("mean", np.nan) * 100, 1) for w in ("2026앞", "2026뒤")}, flush=True)
    J["grid"] = out
    OUT_JSON.write_text(json.dumps(J, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


CAL_SPANS = {"2015~2025": ("2015-01-02", "2025-12-31"), "1986~2005": ("1986-01-02", "2005-12-31"),
             "2006~2025": ("2006-01-02", "2025-12-31")}


def calendar_pf(stocks, ent, H, stop, d0, d1, cash=0.0):
    """그날 열린 거래를 같은 돈으로 나눠 든 달력 포트폴리오(매일 재조정 근사). 열린 거래가 없으면 현금(cash 연이율)."""
    G = np.unique(np.concatenate([s.days for s in stocks]))
    S = np.zeros(len(G)); C = np.zeros(len(G))
    rc = cash / 252
    ntr = 0
    for i, (s0s, es) in ent.items():
        s = stocks[i]
        gi = np.searchsorted(G, s.days)
        for e in es:
            if e + H - 1 >= len(s.c) or not (d0 <= s.days[e] <= d1):
                continue
            entry = s.o[e]
            path = s.c[e:e + H] / entry
            kx = None
            if stop is not None:
                L = entry * (1 - stop)
                hit = np.flatnonzero((s.o[e:e + H] <= L) | (s.l[e:e + H] <= L))
                if len(hit):
                    kx = hit[0]
                    path = path.copy()
                    path[kx:] = min(s.o[e + kx], L) / entry if s.o[e + kx] <= L else L / entry
            r = path / np.concatenate([[1.0], path[:-1]]) - 1
            if kx is not None and rc:
                r[kx + 1:] = rc
            S[gi[e:e + H]] += r; C[gi[e:e + H]] += 1
            ntr += 1
    span = (G >= d0) & (G <= d1)
    r = np.where(C > 0, S / np.maximum(C, 1), rc)[span]
    eq = np.concatenate([[1.0], np.cumprod(1 + r)])
    yrs = len(r) / 252
    return {"cagr": float(eq[-1] ** (1 / yrs) - 1), "mdd": float((eq / np.maximum.accumulate(eq) - 1).min()),
            "per_year": ntr / (int(d1[:4]) - int(d0[:4]) + 1), "expo": float((C[span] > 0).mean())}


def cal_run():
    """G1·바로 매수·보유만의 달력 포트폴리오(1986~2005·2006~2025는 이어 붙인 자료, 2015~2025는 p4 자료로 원 숫자 대조)."""
    J = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    stocks, _, _ = load_new()
    sp, _ = load_spliced(stocks)
    out = {}
    for name, (d0, d1) in CAL_SPANS.items():
        st = stocks if name == "2015~2025" else sp
        for tag, (N, C, stop) in {"G1": (1, "C1", 0.25), "N0": (0, "C0", 0.25), "G1보유만": (1, "C1", None)}.items():
            en = build_entries(st, 0.15, N, C, 20)
            out[f"{name}|{tag}"] = {"c0": calendar_pf(st, en, 63, stop, d0, d1),
                                    "c4": calendar_pf(st, en, 63, stop, d0, d1, cash=0.04)}
            print("달력", name, tag, {k: round(v, 3) for k, v in out[f"{name}|{tag}"]["c0"].items()}, flush=True)
        # 기준: 같은 종목표 동일가중 보유(이력 252일 뒤 종목)
        G = np.unique(np.concatenate([s.days for s in st]))
        S = np.zeros(len(G)); Cn = np.zeros(len(G))
        for s in st:
            gi = np.searchsorted(G, s.days)
            r = np.full(len(s.c), np.nan); r[1:] = s.c[1:] / s.c[:-1] - 1; r[:HIST + 1] = np.nan
            k = np.isfinite(r)
            S += np.bincount(gi[k], r[k], minlength=len(G)); Cn += np.bincount(gi[k], minlength=len(G))
        span = (G >= d0) & (G <= d1)
        r = np.where(Cn > 0, S / np.maximum(Cn, 1), 0.0)[span]
        eq = np.concatenate([[1.0], np.cumprod(1 + r)])
        out[f"{name}|EW"] = {"c0": {"cagr": float(eq[-1] ** (252 / len(r)) - 1), "mdd": float((eq / np.maximum.accumulate(eq) - 1).min())}}
        print("달력", name, "EW", out[f"{name}|EW"], flush=True)
    J["calendar"] = out
    OUT_JSON.write_text(json.dumps(J, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch_old()
    elif "--grid" in sys.argv:
        grid_run()
    elif "--cal" in sys.argv:
        cal_run()
    else:
        run()
