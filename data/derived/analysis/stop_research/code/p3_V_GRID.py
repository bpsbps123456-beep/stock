"""반박 검증 V_GRID(2026-10-11): 3단계 갈래 GRID의 대표 후보 G1 = recover_q100_x10_d20.

후보: 전량 · 고점 −10% 장중 손절 → 고점 −20% 지정가 재매수 · 못 닿으면 종가 ≥ 판 가격×1.03 다음 날 시가 재매수.
  고점 = 창 시작 직전 종가부터 잰 추적 최고 종가(t−1까지). 깊이 되산 몫은 종가가 고점×0.90 위로 돌아올 때까지 손절 해제.
목적: '앞 구간 효율선초과 +14.5%'와 확인 숫자를 깨 보기. tranche_lab·stock_lab을 import하지 않고 cache/p2_long.parquet만으로
엔진을 새로 짠다(배당 수정 = Adj Close/Close 비율을 시가·고가·저가·종가에 곱함). 정의 문장대로 짜고 숫자를 맞춰 본다.

할 일 1) 독립 재계산 2) 시점 감사·하루 늦춘 판·미끄러짐 3) 이웃 4) 위약(노출 경로 원형 이동 · 같은 평균 노출 고정)
      5) 장세 6) 판정 → results/p3_V_GRID.md·json
위약 설계(사전 고정)
  P-a 종목별 원형 이동: 종목마다 '그날 들고 있던 비율' 경로를 창 안에서 k일(20 ≤ k ≤ N−20, 종목마다 따로) 원형 이동. 시드 20.
  P-b 공통 원형 이동: 같은 k를 모든 종목에(포트 단위 동시 매도 타이밍은 유지). 시드 20.
  두 위약 모두 종목별 평균 노출을 정확히 보존한다. 수익은 '전날 종가 노출 × 그날 종가 수익'(종가 근사)으로 잰다.
  같은 근사로 실제 경로(k=0)도 재서 '장중 체결 효과'(정확 − 근사)와 '날짜 맞춤 효과'(근사 k=0 − 위약)를 나눈다.
  위약은 종목별 사후 평균 노출을 공짜로 받아 손절 규칙에 불리하게 치우칠 수 있다 → 귀무 보정(N0): 창 안 날짜 순서를
  모든 종목에 같은 순열로 섞은 자료(시계열 예측력 0, 종목 간 동시 움직임·종목별 총수익 보존)에서 같은 차이를 잰다. 시드 20.
  N0 자료에서 GRID 240개를 다시 돌려 '앞 효율선초과 1등' 값의 귀무 분포도 낸다(선택편향 보정).

실행: cd code && PYTHONIOENCODING=utf-8 py p3_V_GRID.py        (전체, json+md)
      cd code && PYTHONIOENCODING=utf-8 py p3_V_GRID.py md     (json으로 md만)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ID = "p3_V_GRID"

# ============================================================================ 사전 고정(결과를 본 뒤 바꾸지 않음)
CAND = {"q": 1.0, "x": 0.10, "d": 0.20, "fb": "recover", "b": 0.03, "n": 20}
NEIGH_X = (0.06, 0.08, 0.10, 0.12, 0.14)        # 손절 깊이: 후보 ±2칸(2%p)
NEIGH_D = (0.16, 0.18, 0.20, 0.22, 0.24)        # 재매수 깊이: 후보 ±2칸(2%p)
NEIGH_Q = (0.50, 0.75, 1.00)                    # 판매 비율: 후보(100%)에서 아래로 2칸(25%p). 위는 없음
NEIGH_B = (0.01, 0.02, 0.03, 0.04, 0.05)        # 되돌림 여유: 후보 ±2칸(1%p)
N_NEIGH = len(NEIGH_X) * len(NEIGH_D) * len(NEIGH_Q) * len(NEIGH_B)    # 375(후보 포함)
Q_USER = 0.25                                   # 사용자 예시(25%만 매도). 1차원 표에만, 이웃 비율에는 넣지 않음
SLIP = 0.005
N_SEEDS = 20
MIN_SHIFT = 20
N_NULL = 20
MIN_HIST = 252
EFF_STEP = 0.005
WICK = 0.10                                     # 자료 점검: 깊은 재매수 날 아래꼬리(min(시가,종가) 대비 저가)가 10% 넘는 날
WIN = {
    "2026": ("2026-01-02", "2026-10-09"), "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"),
    "2025": ("2025-01-02", "2025-12-31"), "2024": ("2024-01-02", "2024-12-31"), "2023": ("2023-01-03", "2023-12-29"),
    "2022": ("2022-01-03", "2022-12-30"), "2021": ("2021-01-04", "2021-12-31"), "2020": ("2020-01-02", "2020-12-31"),
    "2019": ("2019-01-02", "2019-12-31"), "2018": ("2018-01-02", "2018-12-31"),
}
MAIN = ("2026", "2025")
REGIME = ("2026앞", "2026뒤", "2025", "2022", "2018")
CTX = ("2024", "2023", "2022", "2021", "2020", "2019", "2018")
SLICES = {"2026": ("2026", "2026앞", "2026뒤")}
EXCLUDE = {"2018": {"QUBT"}, "2019": {"QUBT"}}
FB = {"recover": 0, "time": 1, "none": 2}
# GRID 240개(선택편향 귀무용, p3_GRID.py 상수를 그대로 옮겨 적음)
G_QS = (0.25, 0.50, 1.00)
G_XS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)
G_OFFS = (0.05, 0.10, 0.15, 0.20, 0.30, None)
# 갈래 GRID가 주장한 숫자(results/p3_GRID.md, 과제 문)
CLAIM = {
    ("2026앞", "효율선초과"): 0.145, ("2026앞", "포트_수익"): 0.536, ("2026앞", "포트_낙폭"): -0.080,
    ("2026앞", "포트BH_수익"): 0.654, ("2026앞", "포트BH_낙폭"): -0.124,
    ("2026뒤", "포트_수익"): -0.021, ("2026뒤", "효율선초과"): -0.021, ("2026뒤", "포트BH_수익"): -0.041,
    ("2025", "효율선초과"): 0.054, ("2025", "포트_수익"): 0.665, ("2025", "포트_낙폭"): -0.289,
    ("2026", "효율선초과"): 0.022, ("2026", "포트_수익"): 0.503, ("2026", "포트_낙폭"): -0.216, ("2026", "평균노출"): 0.81,
    ("2026", "타이밍값"): 0.039,
    ("2026앞", "승률"): 0.30, ("2026앞", "손익비"): 1.93, ("2026앞", "기대값"): -0.007, ("2026앞", "손절주기"): 763,
    ("2026뒤", "승률"): 0.39, ("2026뒤", "손익비"): 2.25, ("2026뒤", "기대값"): 0.014, ("2026뒤", "손절주기"): 411,
    ("2025", "승률"): 0.35, ("2025", "손익비"): 1.93, ("2025", "기대값"): 0.002, ("2025", "손절주기"): 1023,
}
CLAIM_SLIP = {"2026앞": -0.031, "2026뒤": -0.052, "2025": -0.102, "2026": -0.148}   # 미끄러짐 0.5% 효율선초과


# ============================================================================ 자료(독립 적재)
def load_data() -> dict:
    df = pd.read_parquet(CACHE / "p2_long.parquet")
    tags = json.loads((CACHE / "p2_universe.json").read_text(encoding="utf-8"))
    etf = {s for s, v in tags.items() if "ETF" in v}
    out = {}
    for sym, g in df.groupby("sym", sort=True):
        if sym in etf:
            continue
        g = g.sort_values("date")
        ok = np.isfinite(g[["Open", "High", "Low", "Close"]].to_numpy()).all(axis=1)
        ok &= (g.Close.to_numpy() > 0) & (g.Open.to_numpy() > 0) & (g.Low.to_numpy() > 0)
        g = g[ok]
        f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy()
        out[sym] = {"d": g.date.to_numpy().astype(str), "o": g.Open.to_numpy() * f, "h": g.High.to_numpy() * f,
                    "l": g.Low.to_numpy() * f, "c": g.Close.to_numpy() * f}
    return out


# ============================================================================ 엔진(정의 문장대로)
@njit(cache=False)
def sim(o, h, l, c, a, z, x, d, fb, b, n, slip, delay, phantom):
    """a..z 거래일, a−1 종가에 전량 보유(가치 1). d ≤ 0이면 깊은 재매수 없음. fb 0=recover 1=time 2=none.
    delay 0: 손절·깊은 재매수는 t일 장중(시가가 선 아래면 시가), 되돌림은 c[t−1] 신호 → t일 시가.
    delay 1(하루 늦춘 판): 모든 체결을 하루 늦춤 — 장중 닿음(t) → t+1 시가, 되돌림 c[t−1] 신호 → t+1 시가.
    phantom 1: 되돌림 재매수 시가가 이미 손절선 아래면 정의대로 사서 같은 시가에 바로 판다(미끄러짐 두 번).
    phantom 0: 그 왕복을 하지 않는다(판 가격·판 날만 그 시가로 옮김). 미끄러짐 0이면 둘이 같다.
    주기 = 처음 판 가격 / 마지막에 되산 가격 − 1(같은 시가 왕복은 한 주기로 이어 붙임).
    반환: vals(N+1), expo(N, 종가 시점 보유 비율), cyc(K×6: 판 t, 판 가격, 산 t(−1=열림), 산 가격, 종류 1깊은 2되돌림 0열림, 왕복수), K."""
    N = z - a + 1
    vals = np.empty(N + 1)
    expo = np.empty(N)
    cyc = np.zeros((N + 2, 6))
    k = 0
    sh = 1.0 / c[a - 1]
    cash = 0.0
    held = True
    armed = True
    sold_px = 0.0
    sold_t = -1
    ch_px = 0.0
    ch_t = -1
    nph = 0
    peak = c[a - 1]
    vals[0] = 1.0
    pend = 0          # 하루 늦춘 판: 1 매도, 2 깊은 재매수, 3 되돌림 재매수
    pend_t = -1
    for t in range(a, z + 1):
        L = peak * (1.0 - x)
        if delay == 0:
            # 1) 시가: 되돌림 재매수(c[t-1]까지 정보)
            if not held:
                go = False
                if fb == 0 and c[t - 1] >= sold_px * (1.0 + b):
                    go = True
                elif fb == 1 and t - sold_t >= n:
                    go = True
                if go:
                    if o[t] <= L:
                        if phantom == 1:
                            cash = cash * (1.0 - slip) / (1.0 + slip)
                        sold_px = o[t] * (1.0 - slip)
                        sold_t = t
                        nph += 1
                    else:
                        px = o[t] * (1.0 + slip)
                        cyc[k, 0] = ch_t; cyc[k, 1] = ch_px; cyc[k, 2] = t; cyc[k, 3] = px; cyc[k, 4] = 2; cyc[k, 5] = nph
                        k += 1
                        sh = cash / px
                        cash = 0.0
                        held = True
                        armed = True
            # 2) 장중 손절
            if held and armed:
                px = -1.0
                if o[t] <= L:
                    px = o[t]
                elif l[t] <= L:
                    px = L
                if px > 0:
                    px = px * (1.0 - slip)
                    cash = sh * px
                    sh = 0.0
                    held = False
                    sold_px = px
                    sold_t = t
                    ch_px = px
                    ch_t = t
                    nph = 0
            # 3) 장중 깊은 재매수
            if (not held) and d > 0:
                R = peak * (1.0 - d)
                if R < sold_px and l[t] <= R:
                    px = min(o[t], R) * (1.0 + slip)
                    cyc[k, 0] = ch_t; cyc[k, 1] = ch_px; cyc[k, 2] = t; cyc[k, 3] = px; cyc[k, 4] = 1; cyc[k, 5] = nph
                    k += 1
                    sh = cash / px
                    cash = 0.0
                    held = True
                    armed = False
        else:
            # 시가: 어제 정한 주문 체결
            if pend != 0 and pend_t == t:
                if pend == 1 and held:
                    px = o[t] * (1.0 - slip)
                    cash = sh * px
                    sh = 0.0
                    held = False
                    sold_px = px
                    sold_t = t
                    ch_px = px
                    ch_t = t
                    nph = 0
                elif pend >= 2 and not held:
                    px = o[t] * (1.0 + slip)
                    cyc[k, 0] = ch_t; cyc[k, 1] = ch_px; cyc[k, 2] = t; cyc[k, 3] = px
                    cyc[k, 4] = 1 if pend == 2 else 2
                    cyc[k, 5] = 0
                    k += 1
                    sh = cash / px
                    cash = 0.0
                    held = True
                    armed = pend == 3
                pend = 0
            if (not held) and pend == 0:
                go = False
                if fb == 0 and c[t - 1] >= sold_px * (1.0 + b):
                    go = True
                elif fb == 1 and t - sold_t >= n:
                    go = True
                if go:
                    pend = 3
                    pend_t = t + 1
            if held and armed and pend == 0 and l[t] <= L:
                pend = 1
                pend_t = t + 1
            if (not held) and d > 0 and pend == 0:
                R = peak * (1.0 - d)
                if R < sold_px and l[t] <= R:
                    pend = 2
                    pend_t = t + 1
        # 4) 종가
        eq = sh * c[t] + cash
        vals[t - a + 1] = eq
        expo[t - a] = sh * c[t] / eq if eq > 0 else 0.0
        if c[t] > peak:
            peak = c[t]
        if held and (not armed) and c[t] >= peak * (1.0 - x):
            armed = True
    if not held:
        cyc[k, 0] = ch_t; cyc[k, 1] = ch_px; cyc[k, 2] = -1; cyc[k, 3] = c[z]; cyc[k, 4] = 0; cyc[k, 5] = nph
        k += 1
    return vals, expo, cyc, k


# ============================================================================ 실행 단위
class Run:
    """한 실행 창의 대상 종목(창 시작 전 252거래일 이상)과 공통 달력. items[s] = (days, o, h, l, c, a, z)."""

    def __init__(self, key, items):
        self.key = key
        self.items = items
        self.syms = sorted(items)
        cal = sorted(set().union(*[set(items[s][0][items[s][5]:items[s][6] + 1]) for s in self.syms]))
        self.cal = np.array(cal)
        self.pos = {s: np.searchsorted(self.cal, items[s][0][items[s][5]:items[s][6] + 1]) for s in self.syms}
        T, S = len(self.cal), len(self.syms)
        self.B = np.full((T, S), np.nan)
        for j, s in enumerate(self.syms):
            d, o, h, l, c, a, z = items[s]
            self.B[self.pos[s], j] = c[a:z + 1] / c[a - 1:z] - 1
        self.miss = int(np.isnan(self.B).sum())

    def mask(self, w):
        d0, d1 = WIN[w]
        return (self.cal >= d0) & (self.cal <= d1)


def make_run(data, key, syms=None):
    d0, d1 = WIN[key]
    items = {}
    for s in sorted(data):
        if syms is not None and s not in syms:
            continue
        if s in EXCLUDE.get(key, ()):
            continue
        x = data[s]
        a = int(np.searchsorted(x["d"], d0, "left"))
        z = int(np.searchsorted(x["d"], d1, "right")) - 1
        if a < MIN_HIST or z <= a:
            continue
        items[s] = (x["d"], x["o"], x["h"], x["l"], x["c"], a, z)
    return Run(key, items)


def run_rule(run: Run, x, d, fb="recover", b=0.03, n=20, slip=0.0, delay=0, phantom=1):
    out = {}
    for s in run.syms:
        dd, o, h, l, c, a, z = run.items[s]
        v, e, cy, k = sim(o, h, l, c, a, z, x, -1.0 if d is None else d, FB[fb], b, n, slip, delay, phantom)
        out[s] = (v, e, cy[:k].copy())
    return out


# ============================================================================ 평가
def curve(daily):
    return np.concatenate([[1.0], np.cumprod(1 + np.asarray(daily, float))])


def mdd(v):
    return float((v / np.maximum.accumulate(v) - 1).min())


_EG = np.arange(0.0, 1.0001, EFF_STEP)


def eff_excess(pr, bh_daily):
    """같은 최대 낙폭 고정 노출 효율선 대비 초과 수익(노출 0~1, 0.005 간격)."""
    cur = np.concatenate([np.ones((len(_EG), 1)), np.cumprod(1 + _EG[:, None] * bh_daily[None, :], axis=1)], axis=1)
    md = (cur / np.maximum.accumulate(cur, axis=1) - 1).min(axis=1)
    ok = md >= mdd(pr) - 1e-12
    i = int(np.argmax(np.where(ok, cur[:, -1], -np.inf)))
    return float(pr[-1] - cur[i, -1]), float(_EG[i])


def daily_matrix(run: Run, res, q=1.0, field="r"):
    """종목 일수익(q 몫만 규칙, 1−q는 보유) 또는 종가 노출 행렬(T×S)."""
    T, S = len(run.cal), len(run.syms)
    M = np.full((T, S), np.nan)
    for j, s in enumerate(run.syms):
        dd, o, h, l, c, a, z = run.items[s]
        v, e, _ = res[s]
        bh = c[a - 1:z + 1] / c[a - 1]
        vq = q * v + (1 - q) * bh
        if field == "r":
            M[run.pos[s], j] = vq[1:] / vq[:-1] - 1
        else:
            M[run.pos[s], j] = (q * v[1:] * e + (1 - q) * bh[1:]) / vq[1:]
    return M


def cycle_list(run: Run, res, w):
    """창 w에서 판 주기의 손익(창 끝 시점 정보: 창 끝 뒤에 되산 것·열린 것은 창 끝 종가로)."""
    d0, d1 = WIN[w]
    g, kind, nph, wick = [], [], 0, 0
    for s in run.syms:
        dd, o, h, l, c, a, z = run.items[s]
        cy = res[s][2]
        if not len(cy):
            continue
        iz = int(np.searchsorted(dd, d1, "right")) - 1
        for et, ep, rt, rp, kd, ph in cy:
            et, rt = int(et), int(rt)
            if not (d0 <= dd[et] <= d1):
                continue
            if rt >= 0 and rt <= iz:
                g.append(ep / rp - 1)
                kind.append(int(kd))
                if kd == 1 and l[rt] < (1 - WICK) * min(o[rt], c[rt]):
                    wick += 1
            else:
                g.append(ep / c[iz] - 1)
                kind.append(0)
            nph += int(ph)
    return np.array(g), np.array(kind), nph, wick


def cyc_stats(g, kind):
    n = len(g)
    if not n:
        return {"손절주기": 0, "승률": np.nan, "평균이익": np.nan, "평균손실": np.nan, "손익비": np.nan, "기대값": np.nan,
                "깊은재매수비율": np.nan}
    w, lo = g[g > 0], g[g <= 0]
    return {"손절주기": int(n), "승률": float(len(w) / n), "평균이익": float(w.mean()) if len(w) else np.nan,
            "평균손실": float(lo.mean()) if len(lo) else np.nan,
            "손익비": float(w.mean() / -lo.mean()) if len(w) and len(lo) and lo.mean() < 0 else np.nan,
            "기대값": float(g.mean()), "깊은재매수비율": float((kind == 1).mean())}


def wstats(run: Run, res, w, q=1.0, cycles=True, M=None, E=None):
    m = run.mask(w)
    if M is None:
        M = daily_matrix(run, res, q)
    if E is None:
        E = daily_matrix(run, res, q, "e")
    Rw, Bw, Ew = M[m], run.B[m], E[m]
    rp = np.nanmean(Rw, axis=1)
    bp = np.nanmean(Bw, axis=1)
    pr, pb = curve(rp), curve(bp)
    exc, e_eq = eff_excess(pr, bp)
    ebar = float(np.nanmean(np.nanmean(Ew, axis=0)))
    fx = curve(ebar * bp)
    sr = np.nanprod(1 + Rw, axis=0) - 1
    sb = np.nanprod(1 + Bw, axis=0) - 1
    out = {"종목수": int(Rw.shape[1]), "포트_수익": float(pr[-1] - 1), "포트_낙폭": mdd(pr), "포트BH_수익": float(pb[-1] - 1),
           "포트BH_낙폭": mdd(pb), "효율선초과": exc, "효율선_e": e_eq, "평균노출": ebar,
           "고정평균노출_수익": float(fx[-1] - 1), "타이밍값": float(pr[-1] - fx[-1]),
           "수익_나은종목비율": float((sr > sb + 1e-12).mean())}
    if cycles:
        g, kind, nph, wick = cycle_list(run, res, w)
        out.update(cyc_stats(g, kind))
        out["같은시가왕복"] = nph
        out["큰아래꼬리_깊은재매수"] = wick
        out["종목당_손절"] = out["손절주기"] / out["종목수"]
    return out


def all_w(run: Run, res, q=1.0, cycles=True):
    M, E = daily_matrix(run, res, q), daily_matrix(run, res, q, "e")
    return {w: wstats(run, res, w, q, cycles, M, E) for w in SLICES.get(run.key, (run.key,))}


# ============================================================================ 1) 독립 재계산 + 엔진 점검
def engine_checks(data, run26):
    out = {}
    # 손절 없는 규칙(x=10 → 아주 깊게) = 매수·보유
    res = run_rule(run26, 0.999, None, "none")
    M = daily_matrix(run26, res)
    out["손절없음_vs_보유_최대차"] = float(np.nanmax(np.abs(M - run26.B)))
    # 미래 교란: T 이후 가격을 바꿔도 T까지 가치가 같아야 한다
    worst = 0.0
    rng = np.random.default_rng(7)
    for s in run26.syms[:40]:
        dd, o, h, l, c, a, z = run26.items[s]
        T = a + 70
        v1, _, _, _ = sim(o, h, l, c, a, z, 0.10, 0.20, 0, 0.03, 20, 0.0, 0, 1)
        o2, h2, l2, c2 = o.copy(), h.copy(), l.copy(), c.copy()
        for arr in (o2, h2, l2, c2):
            arr[T + 1:] *= 1 + rng.normal(0, 0.05, len(arr) - T - 1)
        h2[T + 1:] = np.maximum.reduce([h2[T + 1:], o2[T + 1:], c2[T + 1:]])
        l2[T + 1:] = np.minimum.reduce([l2[T + 1:], o2[T + 1:], c2[T + 1:]])
        for dl in (0, 1):
            v1, _, _, _ = sim(o, h, l, c, a, z, 0.10, 0.20, 0, 0.03, 20, 0.0, dl, 1)
            v2, _, _, _ = sim(o2, h2, l2, c2, a, z, 0.10, 0.20, 0, 0.03, 20, 0.0, dl, 1)
            worst = max(worst, float(np.max(np.abs(v1[:T - a + 2] - v2[:T - a + 2]))))
    out["미래교란_최대차"] = worst
    # phantom 0/1은 미끄러짐 0에서 같아야 한다
    r1 = run_rule(run26, 0.10, 0.20, "recover", 0.03, phantom=1)
    r0 = run_rule(run26, 0.10, 0.20, "recover", 0.03, phantom=0)
    out["왕복유무_미끄러짐0_최대차"] = float(max(np.max(np.abs(r1[s][0] - r0[s][0])) for s in run26.syms))
    # 일부 매도 = 섞음(정의상): q=0.25의 종목 가치 = 0.25 × 규칙 + 0.75 × 보유
    return out


def compare_claim(stats):
    rows = []
    for (w, k), v in CLAIM.items():
        got = stats[w][k]
        unit = 1.0 if k in ("손익비", "손절주기", "평균노출") else 100.0
        diff = (got - v) * (1 if k in ("손익비", "손절주기") else 100)
        tol = {"손익비": 0.05, "손절주기": 5}.get(k, 0.5)
        # 주장 숫자는 반올림 값이라 반올림 폭 안이면 일치로 본다
        rnd = {"손익비": 0.005, "손절주기": 0.5}.get(k, 0.05 if k not in ("승률",) else 0.5)
        rows.append({"창": w, "항목": k, "주장": v, "재계산": got, "차": diff, "일치": bool(abs(diff) <= tol + rnd)})
    return rows


# ============================================================================ 3) 이웃
def neighbors(runs, slip=0.0, phantom=0):
    out = {}
    for x in NEIGH_X:
        for d in NEIGH_D:
            for b in NEIGH_B:
                res = {k: run_rule(runs[k], x, d, "recover", b, slip=slip, phantom=phantom) for k in MAIN}
                for q in NEIGH_Q + (Q_USER,):
                    st = {}
                    for k in MAIN:
                        st.update(all_w(runs[k], res[k], q, cycles=(q == 1.0)))
                    key = f"x{x:.2f}_d{d:.2f}_q{q:.2f}_b{b:.2f}"
                    out[key] = {w: {kk: st[w][kk] for kk in ("포트_수익", "포트_낙폭", "효율선초과", "타이밍값", "평균노출",
                                                              "승률", "손익비", "기대값", "손절주기") if kk in st[w]}
                                for w in st}
    return out


# ============================================================================ 4) 위약
def placebo(run: Run, res, w, seeds=N_SEEDS):
    """원형 이동 위약. x_t = 전날 종가 노출(창 첫날 = 그 전날 노출, 실행 첫날은 1)."""
    m = run.mask(w)
    idx = np.where(m)[0]
    E = daily_matrix(run, res, 1.0, "e")
    T, S = len(run.cal), len(run.syms)
    # 전날 노출 행렬: 종목 자기 날짜 기준으로 한 칸 민다
    X = np.full((T, S), np.nan)
    for j, s in enumerate(run.syms):
        e = res[s][1]
        X[run.pos[s], j] = np.concatenate([[1.0], e[:-1]])
    Xw, Bw = X[idx], run.B[idx]
    bp = np.nanmean(Bw, axis=1)
    exact = curve(np.nanmean(daily_matrix(run, res)[idx], axis=1))

    def ev(Xs):
        pr = curve(np.nanmean(Xs * Bw, axis=1))
        exc, _ = eff_excess(pr, bp)
        return float(pr[-1] - 1), mdd(pr), exc

    base = ev(Xw)
    ebar = float(np.nanmean(np.nanmean(E[idx], axis=0)))
    ebar_x = float(np.nanmean(np.nanmean(Xw, axis=0)))
    fx = curve(ebar_x * bp)
    out = {"정확_수익": float(exact[-1] - 1), "정확_낙폭": mdd(exact), "정확_효율선초과": eff_excess(exact, bp)[0],
           "근사_수익": base[0], "근사_낙폭": base[1], "근사_효율선초과": base[2],
           "고정평균노출_수익": float(fx[-1] - 1), "고정평균노출_낙폭": mdd(fx), "평균노출_x": ebar_x, "평균노출_e": ebar}
    for tag, common in (("Pa", False), ("Pb", True)):
        rs, ms, xs = [], [], []
        for sd in range(seeds):
            rng = np.random.default_rng(1000 + sd)
            Xs = np.full_like(Xw, np.nan)
            kc = None
            for j in range(S):
                col = Xw[:, j]
                ok = np.isfinite(col)
                v = col[ok]
                Nn = len(v)
                if Nn <= 2 * MIN_SHIFT:
                    Xs[ok, j] = v
                    continue
                if common:
                    if kc is None:
                        kc = int(rng.integers(MIN_SHIFT, len(idx) - MIN_SHIFT + 1))
                    k = kc % Nn
                else:
                    k = int(rng.integers(MIN_SHIFT, Nn - MIN_SHIFT + 1))
                Xs[ok, j] = np.roll(v, k)
            r_, m_, x_ = ev(Xs)
            rs.append(r_)
            ms.append(m_)
            xs.append(x_)
        rs, ms, xs = np.array(rs), np.array(ms), np.array(xs)
        out[tag] = {"수익_평균": float(rs.mean()), "수익_최소": float(rs.min()), "수익_최대": float(rs.max()),
                    "낙폭_평균": float(ms.mean()), "효율선초과_평균": float(xs.mean()), "효율선초과_최대": float(xs.max()),
                    "근사가_위약보다_큰_비율": float((base[0] > rs).mean()),
                    "근사_효율선초과가_위약보다_큰_비율": float((base[2] > xs).mean()),
                    "날짜맞춤효과": float(base[0] - rs.mean())}
    out["장중체결효과"] = out["정확_수익"] - out["근사_수익"]
    out["종목배분효과_Pa"] = out["Pa"]["수익_평균"] - out["고정평균노출_수익"]
    return out


# ============================================================================ 귀무 자료(날짜 순서 섞기)
def null_run(run: Run, rng, segs):
    """실행의 대상 종목을 공통 달력에 맞추고(빠진 날은 평평한 봉), 창 안 날짜 순서를 모든 종목에 같은 순열로 섞는다.
    segs = 섞는 구간들(달력 마스크). 하루의 (밤사이, 시가→고가, 시가→저가, 시가→종가) 로그값 묶음을 통째로 옮긴다."""
    T = len(run.cal)
    perm = np.arange(T)
    for sm in segs:
        ii = np.where(sm)[0]
        perm[ii] = rng.permutation(ii)
    items = {}
    for s in run.syms:
        dd, o, h, l, c, a, z = run.items[s]
        pos = run.pos[s]
        c0 = c[a - 1]
        go = np.zeros(T); gh = np.zeros(T); gl = np.zeros(T); gc = np.zeros(T)
        pc = c[a - 1:z]
        go[pos] = np.log(o[a:z + 1] / pc)
        gh[pos] = np.log(h[a:z + 1] / o[a:z + 1])
        gl[pos] = np.log(l[a:z + 1] / o[a:z + 1])
        gc[pos] = np.log(c[a:z + 1] / o[a:z + 1])
        go, gh, gl, gc = go[perm], gh[perm], gl[perm], gc[perm]
        cc = c0 * np.exp(np.cumsum(go + gc))
        prev = np.concatenate([[c0], cc[:-1]])
        oo = prev * np.exp(go)
        hh = oo * np.exp(gh)
        ll = oo * np.exp(gl)
        days = np.concatenate([["0000-00-00"], run.cal])
        items[s] = (days, np.concatenate([[c0], oo]), np.concatenate([[c0], hh]), np.concatenate([[c0], ll]),
                    np.concatenate([[c0], cc]), 1, T)
    r = Run(run.key, items)
    return r


def grid240():
    out = []
    for fb in ("recover", "time", "none"):
        for q in (G_QS if fb == "recover" else (1.0,)):
            for x in G_XS:
                for off in G_OFFS:
                    d = None if off is None else round(x + off, 4)
                    out.append((fb, q, x, d))
    assert len(out) == 240
    return out


def null_calibration(runs):
    """귀무(N0): 날짜 순서 섞은 자료에서 G1의 효율선초과·타이밍값·위약 차이, 그리고 240개 중 앞 1등 값."""
    G = grid240()
    res_out = {"G1": [], "max240_앞": [], "pos240_앞": [], "pa_gap": {}, "pb_gap": {}}
    for sd in range(N_NULL):
        rng = np.random.default_rng(500 + sd)
        r26 = runs["2026"]
        nr26 = null_run(r26, rng, [r26.mask("2026앞"), r26.mask("2026뒤")])
        nr25 = null_run(runs["2025"], rng, [np.ones(len(runs["2025"].cal), bool)])
        st = {}
        base = {}
        for k, nr in (("2026", nr26), ("2025", nr25)):
            res = run_rule(nr, 0.10, 0.20, "recover", 0.03)
            base[k] = (nr, res)
            st.update(all_w(nr, res))
        row = {w: {kk: st[w][kk] for kk in ("포트_수익", "포트_낙폭", "포트BH_수익", "포트BH_낙폭", "효율선초과", "타이밍값",
                                             "평균노출", "승률", "손익비", "기대값")} for w in st}
        for w in ("2026앞", "2026뒤", "2025"):
            nr, res = base["2026" if w.startswith("2026") else "2025"]
            pl = placebo(nr, res, w, seeds=3)
            row[w]["Pa_날짜맞춤"] = pl["Pa"]["날짜맞춤효과"]
            row[w]["Pb_날짜맞춤"] = pl["Pb"]["날짜맞춤효과"]
            row[w]["장중체결"] = pl["장중체결효과"]
            row[w]["종목배분_Pa"] = pl["종목배분효과_Pa"]
        res_out["G1"].append(row)
        # 240개 앞 효율선초과
        cache = {}
        vals = []
        for fb, q, x, d in G:
            kk = (fb, x, d)
            if kk not in cache:
                cache[kk] = run_rule(nr26, x, d, fb, 0.03, 20)
            M = daily_matrix(nr26, cache[kk], q)
            m = nr26.mask("2026앞")
            pr = curve(np.nanmean(M[m], axis=1))
            vals.append(eff_excess(pr, np.nanmean(nr26.B[m], axis=1))[0])
        vals = np.array(vals)
        res_out["max240_앞"].append(float(vals.max()))
        res_out["pos240_앞"].append(float((vals > 0).mean()))
        print(f"  귀무 {sd + 1}/{N_NULL}: G1 앞 효율선초과 {row['2026앞']['효율선초과']:+.3f}, 240개 최대 {vals.max():+.3f}", flush=True)
    return res_out


def fill_diag(runs):
    """장중 체결가가 다음 날 시가·그날 종가와 얼마나 다른가(하루 늦춘 판의 차이를 설명하려고)."""
    out = {}
    for key in MAIN:
        run = runs[key]
        res = run_rule(run, 0.10, 0.20, "recover", 0.03)
        sn, sc, dn, dc, gap = [], [], [], [], 0
        for s in run.syms:
            dd, o, h, l, c, a, z = run.items[s]
            for et, ep, rt, rp, kd, ph in res[s][2]:
                et, rt = int(et), int(rt)
                if et + 1 <= z:
                    if ep < o[et] - 1e-12:
                        sn.append(o[et + 1] / ep - 1)
                        sc.append(c[et] / ep - 1)
                    else:
                        gap += 1
                if kd == 1 and rt + 1 <= z:
                    dn.append(o[rt + 1] / rp - 1)
                    dc.append(c[rt] / rp - 1)
        sn, sc, dn, dc = map(np.array, (sn, sc, dn, dc))
        out[key] = {"장중손절_n": len(sn), "시가손절_n": gap, "손절_다음시가_평균": float(sn.mean()),
                    "손절_그날종가_평균": float(sc.mean()), "손절_종가가_위_비율": float((sc > 0).mean()),
                    "깊은_n": len(dn), "깊은_다음시가_평균": float(dn.mean()), "깊은_그날종가_평균": float(dc.mean())}
    return out


def null_regime(runs):
    """장세 표용 귀무(첫 실행 뒤 더함, 후보·격자는 그대로): 2022·2018에서 G1만, 같은 날짜 섞기 시드 20."""
    out = {}
    for key in ("2022", "2018"):
        rows = []
        for sd in range(N_NULL):
            rng = np.random.default_rng(900 + sd)
            nr = null_run(runs[key], rng, [np.ones(len(runs[key].cal), bool)])
            res = run_rule(nr, 0.10, 0.20, "recover", 0.03)
            st = all_w(nr, res)[key]
            pl = placebo(nr, res, key, seeds=3)
            rows.append({kk: st[kk] for kk in ("포트_수익", "포트_낙폭", "포트BH_낙폭", "효율선초과", "타이밍값", "승률", "손익비", "기대값")}
                        | {"Pa_날짜맞춤": pl["Pa"]["날짜맞춤효과"], "종목배분_Pa": pl["종목배분효과_Pa"]})
        out[key] = rows
    return out


def actual240(runs):
    """실제 자료에서 240개 앞 효율선초과(내 엔진) — 1등 재현과 귀무 비교용."""
    G = grid240()
    r = runs["2026"]
    m = r.mask("2026앞")
    cache, vals = {}, []
    for fb, q, x, d in G:
        kk = (fb, x, d)
        if kk not in cache:
            cache[kk] = run_rule(r, x, d, fb, 0.03, 20)
        M = daily_matrix(r, cache[kk], q)
        pr = curve(np.nanmean(M[m], axis=1))
        vals.append(eff_excess(pr, np.nanmean(r.B[m], axis=1))[0])
    vals = np.array(vals)
    i = int(np.argmax(vals))
    return {"max": float(vals.max()), "argmax": list(map(str, G[i])), "pos_share": float((vals > 0).mean()),
            "n_pos": int((vals > 0).sum())}


# ============================================================================ 메인
def compute():
    t0 = time.time()
    data = load_data()
    runs = {k: make_run(data, k) for k in MAIN + CTX}
    print("종목 수", {k: len(r.syms) for k, r in runs.items()}, "빠진 칸", {k: r.miss for k, r in runs.items()}, flush=True)
    P = {"id": ID, "made": time.strftime("%Y-%m-%d %H:%M"),
         "consts": {"CAND": CAND, "NEIGH_X": NEIGH_X, "NEIGH_D": NEIGH_D, "NEIGH_Q": NEIGH_Q, "NEIGH_B": NEIGH_B,
                    "N_NEIGH": N_NEIGH, "SLIP": SLIP, "N_SEEDS": N_SEEDS, "MIN_SHIFT": MIN_SHIFT, "N_NULL": N_NULL,
                    "WICK": WICK}}
    P["checks"] = engine_checks(data, runs["2026"])
    print("엔진 점검", P["checks"], flush=True)

    # 1) 후보 재계산(미끄러짐 0, 정의대로 = 같은 시가 왕복 포함 / 왕복 없음 동일)
    cand = {}
    for k in MAIN + CTX:
        res = run_rule(runs[k], 0.10, 0.20, "recover", 0.03)
        cand.update(all_w(runs[k], res))
    P["cand"] = cand
    P["claim_cmp"] = compare_claim(cand)
    print("재현", [(r["창"], r["항목"], round(r["주장"], 4), round(r["재계산"], 4), r["일치"]) for r in P["claim_cmp"]], flush=True)

    # 2) 미끄러짐·하루 늦춘 판
    var = {}
    for tag, kw in (("slip_왕복없음", dict(slip=SLIP, phantom=0)), ("slip_왕복포함", dict(slip=SLIP, phantom=1)),
                    ("delay", dict(delay=1)), ("delay_slip", dict(delay=1, slip=SLIP))):
        st = {}
        for k in MAIN + ("2022", "2018"):
            st.update(all_w(runs[k], run_rule(runs[k], 0.10, 0.20, "recover", 0.03, **kw)))
        var[tag] = st
    P["var"] = var
    P["fill"] = fill_diag(runs)
    print("미끄러짐·늦춤", {t: {w: round(v[w]["효율선초과"], 4) for w in ("2026앞", "2026뒤", "2025", "2026")} for t, v in var.items()},
          flush=True)

    # 3) 이웃
    P["neigh"] = neighbors(runs, 0.0, 0)
    P["neigh_slip"] = neighbors(runs, SLIP, 0)
    print(f"이웃 끝 {time.time() - t0:.0f}초", flush=True)

    # 4) 위약
    pl = {}
    for k, ws in (("2026", ("2026앞", "2026뒤", "2026")), ("2025", ("2025",)), ("2022", ("2022",)), ("2018", ("2018",))):
        res = run_rule(runs[k], 0.10, 0.20, "recover", 0.03)
        for w in ws:
            pl[w] = placebo(runs[k], res, w)
    P["placebo"] = pl
    print("위약", {w: (round(v["근사_수익"], 4), round(v["Pa"]["수익_평균"], 4), round(v["고정평균노출_수익"], 4)) for w, v in pl.items()},
          flush=True)
    # 4b) 귀무 보정 + 선택편향
    P["actual240"] = actual240(runs)
    print("실제 240 앞 1등", P["actual240"], flush=True)
    P["null"] = null_calibration(runs)
    P["null_regime"] = null_regime(runs)
    print(f"끝 {time.time() - t0:.0f}초", flush=True)
    p = RESULTS / f"{ID}.json"
    p.write_text(json.dumps(P, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return P


# ============================================================================ 보고서
def _nan(v):
    return v is None or (isinstance(v, float) and not math.isfinite(v))


def pct(v, d=1):
    return "—" if _nan(v) else f"{v * 100:+.{d}f}%".replace("-", "−")


def upct(v, d=0):
    return "—" if _nan(v) else f"{v * 100:.{d}f}%".replace("-", "−")


def pp(v, d=1):
    return "—" if _nan(v) else f"{v * 100:+.{d}f}%p".replace("-", "−")


def f2(v):
    return "—" if _nan(v) else f"{v:.2f}"


def nkey(x, d, q, b):
    return f"x{x:.2f}_d{d:.2f}_q{q:.2f}_b{b:.2f}"


def report(P):
    C, V, N, PL, NU = P["cand"], P["var"], P["neigh"], P["placebo"], P["null"]
    NS = P["neigh_slip"]
    ck = P["checks"]
    L = []
    a = L.append

    def nul(w, k):
        return np.array([r[w][k] for r in NU["G1"]])

    def ge(arr, v):
        return float(np.mean(np.asarray(arr) >= v - 1e-12))

    # ---------- 숫자 미리
    cmp = P["claim_cmp"]
    n_ok = sum(r["일치"] for r in cmp)
    worst = max(abs(r["차"]) for r in cmp if r["항목"] not in ("손익비", "손절주기"))
    mx = np.array(NU["max240_앞"])
    ps = np.array(NU["pos240_앞"])
    act_max = P["actual240"]["max"]
    keys = [k for k in N if f"_q{Q_USER:.2f}_" not in k]
    A = np.array([[N[k][w]["효율선초과"] for w in ("2026앞", "2026뒤", "2025", "2026")] for k in keys])
    AS = np.array([[NS[k][w]["효율선초과"] for w in ("2026앞", "2026뒤", "2025", "2026")] for k in keys])
    all3 = float((A[:, :3] > 0).all(axis=1).mean())
    all3s = float((AS[:, :3] > 0).all(axis=1).mean())
    gapa = {w: PL[w]["Pa"]["날짜맞춤효과"] for w in ("2026앞", "2026뒤", "2025")}
    ngap = {w: nul(w, "Pa_날짜맞춤") for w in ("2026앞", "2026뒤", "2025")}
    sh4 = [ge(nul(w, k), C[w][k]) for w in ("2026뒤", "2025") for k in ("효율선초과", "기대값")]
    g22 = np.array([r["Pa_날짜맞춤"] for r in P["null_regime"]["2022"]])

    a("# V_GRID: 대표 후보 G1 반박 검증 — 전량 · 고점 −10% 손절 → 고점 −20% 지정가 재매수 · 되돌림 +3% (2026-10-11)")
    a("")
    a("> 대상 주장: `results/p3_GRID.md`의 G1(`recover_q100_x10_d20`). 코드 `code/p3_V_GRID.py`, 숫자 `results/p3_V_GRID.json`.")
    a("> `tranche_lab.py`·`stock_lab.py`를 import하지 않고 `cache/p2_long.parquet`만으로 엔진을 새로 짰다(배당 수정 = Adj Close/Close 비율을 시가·고가·저가·종가에 곱함).")
    a("> 대상: AI 종목 전체를 창 시작 직전 종가에 다 들고 시작(2026 138종목, 2025 133종목, 2022 112종목, 2018 89종목 · QUBT 제외). 동일가중 매일 재조정. 수수료·현금 이자 0.")
    a("")
    a("## 결론")
    a("")
    a(f"1. **숫자는 그대로 재현됐다.** 주장 숫자 {len(cmp)}개 중 {n_ok}개가 반올림 폭 안에서 같았다(가장 큰 차 {worst:.2f}%p, 주장 승률이 정수 %로 반올림된 탓). "
      f"앞 효율선초과 {pct(C['2026앞']['효율선초과'])}, 뒤 {pct(C['2026뒤']['포트_수익'])}, 2025 {pct(C['2025']['효율선초과'])}, 2026 전체 {pct(C['2026']['효율선초과'])}. "
      f"미끄러짐 0.5% 숫자(앞 {pct(V['slip_왕복없음']['2026앞']['효율선초과'])}, 뒤 {pct(V['slip_왕복없음']['2026뒤']['효율선초과'])}, 2025 {pct(V['slip_왕복없음']['2025']['효율선초과'])}, 2026 {pct(V['slip_왕복없음']['2026']['효율선초과'])})도 같았다.")
    a(f"2. **'앞 구간 1위 +14.5%'는 우연으로 설명된다.** 날짜 순서를 섞어 예측력을 없앤 가짜 자료 20개에서 같은 240개를 돌리면 앞 1등 값이 "
      f"중앙값 {pct(float(np.median(mx)))}({pct(mx.min())}~{pct(mx.max())})였다. 실제 1등 {pct(act_max)}는 그 중앙값보다 낮다(가짜 자료 {ge(mx, act_max) * 20:.0f}/20개가 같거나 높음). "
      f"G1 자체의 뒤·2025 효율선초과와 손절 1회 기대값도 가짜 자료와 구별되지 않았다(가짜 ≥ 실제가 20개 중 {min(sh4) * 20:.0f}~{max(sh4) * 20:.0f}개).")
    a(f"3. **선 이탈 때 판 값이 남지 않는다.** 이웃 {len(keys)}개 중 앞·뒤·2025 효율선초과가 모두 플러스인 조합은 {all3 * 100:.0f}%였다. 미끄러짐 0.5%면 뒤·2025·2026 전체에서 {len(keys)}개 모두 마이너스다. "
      f"위약에서 실제 노출 경로가 날짜만 원형 이동한 경로보다 높았지만(앞 {pp(gapa['2026앞'])}, 뒤 {pp(gapa['2026뒤'])}, 2025 {pp(gapa['2025'])}), 가짜 자료에서도 평균 {pp(ngap['2026앞'].mean())}·{pp(ngap['2026뒤'].mean())}·{pp(ngap['2025'].mean())}가 나왔다. "
      f"뒤·2025에서는 실제가 그 범위 안이고, 가짜보다 큰 곳(가짜 20개 중 1개만 같거나 높음)은 고르는 데 쓴 앞 구간과 맥락 해 2022(긴 하락장)뿐이다. **판정: 반박(refuted).**")
    a("")
    a("## 사용자 질문과의 관계")
    a("")
    q25 = N[nkey(0.10, 0.20, Q_USER, 0.03)]
    q25s = NS[nkey(0.10, 0.20, Q_USER, 0.03)]
    a(f"- 같은 선(고점 −10% 손절 → −20% 재매수)에서 25%만 파는 판은 비용 0에서 앞 {pct(q25['2026앞']['효율선초과'])}, 뒤 {pct(q25['2026뒤']['효율선초과'])}, "
      f"2025 {pct(q25['2025']['효율선초과'])}, 2026 전체 {pct(q25['2026']['효율선초과'])}(같은 평균 노출 대비 {pp(q25['2026']['타이밍값'])})였다. "
      f"미끄러짐 0.5%면 {pct(q25s['2026앞']['효율선초과'])} / {pct(q25s['2026뒤']['효율선초과'])} / {pct(q25s['2025']['효율선초과'])} / {pct(q25s['2026']['효율선초과'])}다.")
    a("- 일부 매도는 정의상 '전량 규칙 q + 그냥 보유 1−q'라서 전량 판의 값과 비용을 q만큼 줄인다. 전량 판에 남는 값이 없으니 25% 판에도 없다. 갈래 GRID의 결론 3과 같다.")
    a("")
    a("## 시험 범위 (사전 고정, 스크립트 맨 위 상수)")
    a("")
    a("| 검증 | 개수·간격 | 비고 |")
    a("|---|---|---|")
    a(f"| 독립 재계산 | 후보 1개, 주장 숫자 {len(cmp)}개 + 미끄러짐 4개 | 0.5%p 넘게 다르면 원인 규명 |")
    a("| 시점 감사 | 미래 교란(40종목 × 정상·하루 늦춘 판), 손절 없음 = 보유, 같은 시가 왕복 유무 | 엔진 코드도 읽음 |")
    a("| 체결 변형 | 하루 늦춘 판, 미끄러짐 0.5%(왕복 없음 · 정의대로 왕복 포함), 둘 다 | 2026·2025·2022·2018 |")
    a(f"| 이웃 | 손절 X {{6, 8, 10, 12, 14%}} × 재매수 D {{16, 18, 20, 22, 24%}} × 판매 비율 q {{50, 75, 100%}} × 되돌림 여유 b {{1, 2, 3, 4, 5%}} = **{N_NEIGH}개** | 후보 ±2칸(X·D 2%p, b 1%p, q는 아래로 25%p). 비용 0과 0.5% 두 번. q 25%는 사용자 예시로 따로 |")
    a(f"| 위약 | P-a 종목별 원형 이동 · P-b 공통 원형 이동, 각 시드 {N_SEEDS}개, 이동 {MIN_SHIFT}거래일 이상 | 같은 평균 노출 고정과 나란히 |")
    a(f"| 귀무 보정 | 날짜 순서 섞은 자료 {N_NULL}개(2026은 앞·뒤 따로 섞음, 2025 · 2022 · 2018) | 위약 치우침 · 240개 1등 선택편향 |")
    a("")
    a("## 변수 시점")
    a("")
    a("| 변수 | 언제 아는가 | 체결 |")
    a("|---|---|---|")
    a("| 고점 = 창 시작 직전 종가부터 잰 최고 종가 | t−1 종가 | — |")
    a("| 손절가 = 고점 × 0.90 | t−1 종가 | t일 장중 스탑(시가가 그 아래면 시가) · 하루 늦춘 판은 닿은 다음 날 시가 |")
    a("| 깊은 재매수가 = 고점 × 0.80 | t−1 종가 | t일 장중 지정가 min(시가, 재매수가) · 하루 늦춘 판은 다음 날 시가 |")
    a("| 되돌림: 종가 ≥ 판 가격 × 1.03 | t−1 종가 | t일 시가 · 하루 늦춘 판은 t+1일 시가 |")
    a("| 재무장: 깊이 되산 뒤 종가 ≥ 고점 × 0.90 | t 종가 | t+1일부터 손절 다시 걸림 |")
    a("| 평균 노출 · 효율선 노출 · 위약 경로 · 귀무 자료 | 사후 | 평가용. 매매에 쓰지 않음 |")
    a("")
    # ---------- 1
    a("## 1) 독립 재계산")
    a("")
    a("| 창 | 항목 | 주장 | 재계산 | 일치 |")
    a("|---|---|---|---|---|")
    for r in cmp:
        k = r["항목"]
        if k in ("손익비",):
            s1, s2 = f2(r["주장"]), f2(r["재계산"])
        elif k == "손절주기":
            s1, s2 = str(int(r["주장"])), str(int(r["재계산"]))
        elif k == "평균노출":
            s1, s2 = f2(r["주장"]), f2(r["재계산"])
        elif k == "승률":
            s1, s2 = upct(r["주장"]), upct(r["재계산"], 1)
        else:
            s1, s2 = pct(r["주장"]), pct(r["재계산"], 2)
        a(f"| {r['창']} | {k} | {s1} | {s2} | {'예' if r['일치'] else '아니오'} |")
    for w, v in CLAIM_SLIP.items():
        g = V["slip_왕복없음"][w]["효율선초과"]
        a(f"| {w} | 미끄러짐 0.5% 효율선초과 | {pct(v)} | {pct(g, 2)} | {'예' if abs(g - v) <= 0.001 else '아니오'} |")
    a("")
    a(f"- 엔진 점검: 손절 없는 규칙 = 매수·보유(최대 차 {ck['손절없음_vs_보유_최대차']:.0e}), 미래 교란 뒤 그날까지 가치 차 {ck['미래교란_최대차']:.0e}(정상·하루 늦춘 판 모두), "
      f"같은 시가 왕복을 하든 안 하든 비용 0이면 같음(차 {ck['왕복유무_미끄러짐0_최대차']:.0e}).")
    a(f"- 손절 1회 성적은 '되돌림으로 되산 시가가 이미 손절선 아래라 같은 시가에 다시 판' 왕복을 한 주기로 이었다(2026 {C['2026']['같은시가왕복']}번). 갈래 GRID의 '실_' 규약과 같다.")
    a(f"- 자료 점검: 깊은 재매수 날 아래꼬리가 10% 넘는 날(잘못 찍힌 저가 의심)은 2026 {C['2026']['큰아래꼬리_깊은재매수']}번, 2025 {C['2025']['큰아래꼬리_깊은재매수']}번뿐이라 결과를 바꾸지 않는다.")
    a("")
    # ---------- 2
    a("## 2) 시점 감사 · 하루 늦춘 판 · 미끄러짐")
    a("")
    a("엔진(`tranche_lab.simulate`)을 읽고 확인한 것: 고점은 그날 종가 뒤에 갱신돼 다음 날 손절가·재매수가에만 쓰인다. 되돌림 신호는 전날 종가다. "
      "같은 날 손절 뒤 재매수는 재매수가 < 손절가 < 시가라서 가격이 위에서 아래로 내려오며 둘 다 지나야 하므로 일봉으로도 가능한 순서다. "
      "시가가 재매수가 아래로 갭 하락하면 '재매수가 < 판 가격' 조건 때문에 같은 가격 왕복이 생기지 않는다. 미래 정보를 쓰는 곳은 찾지 못했다.")
    a("")
    a("| 판 | 앞 수익 / 낙폭 / 효율선초과 | 뒤 수익 / 낙폭 | 2025 수익 / 낙폭 / 효율선초과 | 2026 수익 / 낙폭 / 효율선초과 | 2026 노출 / 타이밍 값 | 2026 손절 횟수 |")
    a("|---|---|---|---|---|---|---|")

    def vrow(name, s):
        a(f"| {name} | {pct(s['2026앞']['포트_수익'])} / {pct(s['2026앞']['포트_낙폭'])} / {pct(s['2026앞']['효율선초과'])} | "
          f"{pct(s['2026뒤']['포트_수익'])} / {pct(s['2026뒤']['포트_낙폭'])} | "
          f"{pct(s['2025']['포트_수익'])} / {pct(s['2025']['포트_낙폭'])} / {pct(s['2025']['효율선초과'])} | "
          f"{pct(s['2026']['포트_수익'])} / {pct(s['2026']['포트_낙폭'])} / {pct(s['2026']['효율선초과'])} | "
          f"{f2(s['2026']['평균노출'])} / {pp(s['2026']['타이밍값'])} | {s['2026']['손절주기']} |")

    vrow("G1 그대로(장중 체결, 비용 0)", C)
    vrow("미끄러짐 0.5%(같은 시가 왕복 안 함)", V["slip_왕복없음"])
    vrow("미끄러짐 0.5%(정의대로 왕복까지)", V["slip_왕복포함"])
    vrow("하루 늦춘 판(닿은 다음 날 시가)", V["delay"])
    vrow("하루 늦춘 판 + 미끄러짐 0.5%", V["delay_slip"])
    a(f"| 매수·보유 | {pct(C['2026앞']['포트BH_수익'])} / {pct(C['2026앞']['포트BH_낙폭'])} / 0 | {pct(C['2026뒤']['포트BH_수익'])} / {pct(C['2026뒤']['포트BH_낙폭'])} | "
      f"{pct(C['2025']['포트BH_수익'])} / {pct(C['2025']['포트BH_낙폭'])} / 0 | {pct(C['2026']['포트BH_수익'])} / {pct(C['2026']['포트BH_낙폭'])} / 0 | 1.00 / 0 | — |")
    a("")
    fd = P["fill"]
    a(f"- 하루 늦춘 판이 비용 0에서 오히려 나았다(2026 효율선초과 {pct(C['2026']['효율선초과'])} → {pct(V['delay']['2026']['효율선초과'])}). "
      f"장중 손절가에 판 뒤 다음 날 시가가 평균 {pct(fd['2026']['손절_다음시가_평균'])}(2025 {pct(fd['2025']['손절_다음시가_평균'])}) 높았다. 스탑이 장중 꼬리 끝에서 팔리는 일이 많다는 뜻이다"
      f"(손절한 날 종가가 손절가 위로 돌아온 비율 {upct(fd['2026']['손절_종가가_위_비율'])}).")
    a(f"- 수익 차이는 작고(2026 {pct(C['2026']['포트_수익'])} → {pct(V['delay']['2026']['포트_수익'])}) 효율선초과가 크게 움직인 것은 2026 최대 낙폭이 {pct(C['2026']['포트_낙폭'])} → {pct(V['delay']['2026']['포트_낙폭'])}로 바뀌어 비교할 효율선이 달라졌기 때문이다. "
      "체결을 하루 바꾼 것만으로 효율선초과가 10%p 흔들린다. 이 지표 하나로 규칙을 고르기에는 잡음이 크다.")
    a("- 하루 늦춰도, 장중 그대로여도 미끄러짐 0.5%를 넣으면 2025·2026 전체 효율선초과가 마이너스다. 2026 종목당 손절이 8~9번이라 왕복 비용(약 1%)이 쌓인다.")
    a("")
    # ---------- 3
    a("## 3) 이웃 (사전 고정 ±2칸)")
    a("")
    a("### 핵심 격자: 손절 X × 재매수 D (전량, 되돌림 +3%)")
    a("")
    a("칸 = 앞 효율선초과 / 뒤 수익(= 뒤 효율선초과, 뒤는 효율선 최선이 현금) / 2025 효율선초과. 굵은 칸 = G1.")
    a("")
    a("**비용 0**")
    a("")
    a("| 손절 X \\ 재매수 D | " + " | ".join(f"−{d * 100:.0f}%" for d in NEIGH_D) + " |")
    a("|---|" + "---|" * len(NEIGH_D))
    for x in NEIGH_X:
        cells = []
        for d in NEIGH_D:
            s = N[nkey(x, d, 1.0, 0.03)]
            t = f"{pct(s['2026앞']['효율선초과'])} / {pct(s['2026뒤']['효율선초과'])} / {pct(s['2025']['효율선초과'])}"
            cells.append(f"**{t}**" if (x, d) == (0.10, 0.20) else t)
        a(f"| −{x * 100:.0f}% | " + " | ".join(cells) + " |")
    a("")
    a("**미끄러짐 0.5%** (칸 = 앞 / 뒤 / 2025 효율선초과)")
    a("")
    a("| 손절 X \\ 재매수 D | " + " | ".join(f"−{d * 100:.0f}%" for d in NEIGH_D) + " |")
    a("|---|" + "---|" * len(NEIGH_D))
    for x in NEIGH_X:
        cells = []
        for d in NEIGH_D:
            s = NS[nkey(x, d, 1.0, 0.03)]
            t = f"{pct(s['2026앞']['효율선초과'])} / {pct(s['2026뒤']['효율선초과'])} / {pct(s['2025']['효율선초과'])}"
            cells.append(f"**{t}**" if (x, d) == (0.10, 0.20) else t)
        a(f"| −{x * 100:.0f}% | " + " | ".join(cells) + " |")
    a("")
    a("### 한 축씩 움직이기 (나머지는 G1 그대로)")
    a("")
    a("| 축 | 값 | 앞 | 뒤 | 2025 | 2026 전체 | 2026 타이밍 값 | 미끄러짐 0.5%: 2026 전체 |")
    a("|---|---|---|---|---|---|---|---|")
    axes = (("손절 X", "x", NEIGH_X), ("재매수 D", "d", NEIGH_D), ("판매 비율 q", "q", (Q_USER,) + NEIGH_Q), ("되돌림 여유 b", "b", NEIGH_B))
    for name, p_, vals in axes:
        for v in vals:
            kk = {"x": 0.10, "d": 0.20, "q": 1.0, "b": 0.03}
            kk[p_] = v
            s, ss = N[nkey(**kk)], NS[nkey(**kk)]
            lab = f"−{v * 100:.0f}%" if p_ in ("x", "d") else (f"{v * 100:.0f}%" if p_ == "q" else f"+{v * 100:.0f}%")
            if p_ == "q" and v == Q_USER:
                lab += "(사용자 예시)"
            star = " ←G1" if kk == {"x": 0.10, "d": 0.20, "q": 1.0, "b": 0.03} else ""
            a(f"| {name} | {lab}{star} | {pct(s['2026앞']['효율선초과'])} | {pct(s['2026뒤']['효율선초과'])} | {pct(s['2025']['효율선초과'])} | "
              f"{pct(s['2026']['효율선초과'])} | {pp(s['2026']['타이밍값'])} | {pct(ss['2026']['효율선초과'])} |")
    a("")
    a("### 이웃 전체에서 효율선초과 > 0 비율")
    a("")
    a("| 묶음 | 개수 | 앞 | 뒤 | 2025 | 앞·뒤·2025 모두 | 앞·2025 둘 다 | 2026 전체 |")
    a("|---|---|---|---|---|---|---|---|")

    def share_row(name, M):
        a(f"| {name} | {len(M)} | {upct((M[:, 0] > 0).mean())} | {upct((M[:, 1] > 0).mean())} | {upct((M[:, 2] > 0).mean())} | "
          f"{upct((M[:, :3] > 0).all(axis=1).mean())} | {upct(((M[:, 0] > 0) & (M[:, 2] > 0)).mean())} | {upct((M[:, 3] > 0).mean())} |")

    share_row("비용 0 · 전체", A)
    for q in sorted(NEIGH_Q, reverse=True):
        idx = [i for i, k in enumerate(keys) if f"_q{q:.2f}_" in k]
        share_row(f"비용 0 · 판매 비율 {q * 100:.0f}%", A[idx])
    share_row("미끄러짐 0.5% · 전체", AS)
    a("")
    sel = A[:, 0] > 0
    a(f"- 앞에서 효율선을 이긴 이웃({int(sel.sum())}개)이 2025에도 이긴 비율 {upct((A[sel, 2] > 0).mean())}, 고르지 않은 전체 {upct((A[:, 2] > 0).mean())}. "
      f"뒤에서 이긴 비율은 둘 다 {upct((A[sel, 1] > 0).mean())}·{upct((A[:, 1] > 0).mean())}다.")
    a(f"- 앞 효율선초과는 손절 −10%에서 뾰족하다(−8% {pct(N[nkey(0.08, 0.20, 1.0, 0.03)]['2026앞']['효율선초과'])}, −10% {pct(C['2026앞']['효율선초과'])}, −12% {pct(N[nkey(0.12, 0.20, 1.0, 0.03)]['2026앞']['효율선초과'])}). "
      f"이웃 평균은 {pct(A[:, 0].mean())}다. 1등 값의 상당 부분이 그 한 칸에만 있다.")
    a("- 뒤 구간은 매수·보유가 −4.1%라 효율선 최선이 현금(0%)이다. 노출이 0.8 안팎인 손절 규칙이 뒤에서 플러스를 내려면 하락을 거의 다 피해야 하는데 이웃 어느 것도 못 했다.")
    a("")
    # ---------- 4
    a("## 4) 위약: '선 이탈 때 판 것'인가 '덜 들고 있던 것'인가")
    a("")
    a("같은 종목의 '그날 들고 있던 비율' 경로를 날짜만 원형 이동했다(P-a 종목마다 다른 이동, P-b 모든 종목 같은 이동, 시드 20). 종목별 평균 노출은 그대로다. "
      "수익은 '전날 종가 보유 비율 × 그날 종가 수익'으로 쟀고, 같은 방식으로 실제 경로도 쟀다(종가 근사).")
    a("")
    a("| 창 | 매수·보유 | G1 실제(장중 체결) | 실제 경로 · 종가 근사 | 위약 P-a 평균 (최소~최대) | 위약 P-b 평균 | 같은 평균 노출 고정 | 실제 경로 − P-a | 가짜 자료의 같은 차(평균 ± 표준편차) | 가짜 ≥ 실제 |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    bh = {w: C[w]["포트BH_수익"] for w in ("2026앞", "2026뒤", "2025", "2026", "2022", "2018")}
    nreg = P.get("null_regime", {})
    for w in ("2026앞", "2026뒤", "2025", "2026", "2022", "2018"):
        p = PL[w]
        if w in ngap:
            g = ngap[w]
            nt = f"{pp(g.mean())} ± {g.std() * 100:.1f}%p"
            nge = f"{ge(g, p['Pa']['날짜맞춤효과']) * 20:.0f}/20"
        elif w in nreg:
            g = np.array([r["Pa_날짜맞춤"] for r in nreg[w]])
            nt = f"{pp(g.mean())} ± {g.std() * 100:.1f}%p"
            nge = f"{ge(g, p['Pa']['날짜맞춤효과']) * 20:.0f}/20"
        else:
            nt, nge = "—", "—"
        a(f"| {w} | {pct(bh[w])} | {pct(p['정확_수익'])} | {pct(p['근사_수익'])} | {pct(p['Pa']['수익_평균'])} ({pct(p['Pa']['수익_최소'])}~{pct(p['Pa']['수익_최대'])}) | "
          f"{pct(p['Pb']['수익_평균'])} | {pct(p['고정평균노출_수익'])} | {pp(p['Pa']['날짜맞춤효과'])} | {nt} | {nge} |")
    a("")
    a("**나눠 보기** (G1 실제 − 같은 평균 노출 고정 = 장중 체결 효과 + 날짜 맞춤 효과 + 종목 배분 효과)")
    a("")
    a("| 창 | G1 − 고정(갈래의 '타이밍 값') | 장중 체결(실제 − 종가 근사) | 날짜 맞춤(종가 근사 − P-a) | 종목 배분(P-a − 고정) | 가짜 자료 평균: 타이밍 값 / 날짜 맞춤 / 종목 배분 |")
    a("|---|---|---|---|---|---|")
    for w in ("2026앞", "2026뒤", "2025", "2026", "2022", "2018"):
        p = PL[w]
        tot = p["정확_수익"] - p["고정평균노출_수익"]
        if w in ngap:
            nn = f"{pp(nul(w, '타이밍값').mean())} / {pp(nul(w, 'Pa_날짜맞춤').mean())} / {pp(nul(w, '종목배분_Pa').mean())}"
        elif w in nreg:
            nn = (f"{pp(np.mean([r['타이밍값'] for r in nreg[w]]))} / {pp(np.mean([r['Pa_날짜맞춤'] for r in nreg[w]]))} / "
                  f"{pp(np.mean([r.get('종목배분_Pa', float('nan')) for r in nreg[w]]))}")
        else:
            nn = "—"
        a(f"| {w} | {pp(tot)} | {pp(p['장중체결효과'])} | {pp(p['Pa']['날짜맞춤효과'])} | {pp(p['종목배분효과_Pa'])} | {nn} |")
    a("")
    a("- 나눠 보기의 '고정'은 전날 종가 보유 비율의 평균(위약과 같은 평균)으로 맞췄다. 그래서 갈래 표의 타이밍 값(종가 시점 보유 비율 평균)과 0.1~0.2%p 다르다.")
    a(f"- 위약 효율선초과: P-a 평균은 앞{pct(PL['2026앞']['Pa']['효율선초과_평균'])}, 2025 {pct(PL['2025']['Pa']['효율선초과_평균'])}, 2026 {pct(PL['2026']['Pa']['효율선초과_평균'])}로 실제보다 낮았다. "
      "원형 이동은 낙폭 구간의 노출 감소를 다른 날로 옮기므로 낙폭이 깊어진다.")
    a("- 그런데 이 위약은 손절 규칙 쪽으로 치우쳐 있다. 예측력이 전혀 없는 가짜 자료(날짜 순서를 모든 종목에 같은 순열로 섞음)에서도 '실제 경로 − 원형 이동'이 평균 플러스로 나왔다. "
      "원형 이동은 창 뒤쪽의 노출을 창 앞쪽 날짜에 붙인다. 창 뒤쪽 노출은 창 앞쪽 상승일이 만든 고점 때문에 낮아지기 쉬워, 위약이 앞쪽 상승일에 덜 들고 있게 되는 것으로 추정한다(확인하지 않음).")
    a(f"- 그래서 가짜 자료의 같은 차와 견줘야 한다. 뒤({pp(gapa['2026뒤'])} 대 가짜 평균 {pp(ngap['2026뒤'].mean())})와 2025({pp(gapa['2025'])} 대 {pp(ngap['2025'].mean())})는 가짜 자료 범위 안이다. "
      f"앞({pp(gapa['2026앞'])} 대 {pp(ngap['2026앞'].mean())})은 높지만 240개 중 1등을 고른 구간이라 이 차이에 선택편향이 그대로 들어 있다. "
      f"고르지 않은 맥락 해 가운데 1년 내내 빠진 2022({pp(PL['2022']['Pa']['날짜맞춤효과'])} 대 {pp(g22.mean())}, 가짜 ≥ 실제 {ge(g22, PL['2022']['Pa']['날짜맞춤효과']) * 20:.0f}/20)만 가짜보다 컸다. "
      f"그해에도 포트 수익은 {pct(C['2022']['포트_수익'])}로 현금(효율선)을 못 이겼다.")
    a("- 같은 평균 노출 고정 대비 '타이밍 값'(갈래 표)도 비용 0에서 플러스였지만, 미끄러짐 0.5%면 2026 전체 "
      f"{pp(V['slip_왕복없음']['2026']['타이밍값'])}, 2025 {pp(V['slip_왕복없음']['2025']['타이밍값'])}로 뒤집힌다.")
    a("")
    # ---------- 4b
    a("### 귀무 보정 · 선택편향 (날짜 섞은 가짜 자료 20개)")
    a("")
    a("| 지표 | 실제 | 가짜 자료 평균 ± 표준편차 (최소~최대) | 가짜 ≥ 실제 |")
    a("|---|---|---|---|")

    def nrow(name, act, arr, fmt=pct):
        arr = np.asarray(arr, float)
        a(f"| {name} | {fmt(act)} | {fmt(arr.mean())} ± {arr.std() * 100:.1f}%p ({fmt(arr.min())}~{fmt(arr.max())}) | {ge(arr, act) * 20:.0f}/20 |")

    nrow("240개 중 앞 효율선초과 1등 값", act_max, mx)
    nrow("240개 중 앞 효율선초과 > 0 비율", P["actual240"]["pos_share"], ps, upct)
    for w in ("2026앞", "2026뒤", "2025"):
        nrow(f"G1 {w} 효율선초과", C[w]["효율선초과"], nul(w, "효율선초과"))
    for w in ("2026앞", "2026뒤", "2025"):
        nrow(f"G1 {w} 타이밍 값(같은 평균 노출 대비)", C[w]["타이밍값"], nul(w, "타이밍값"))
    for w in ("2026앞", "2026뒤", "2025"):
        nrow(f"G1 {w} 손절 1회 기대값", C[w]["기대값"], nul(w, "기대값"))
    a("")
    a("- 가짜 자료는 그 창의 종목별 하루 (밤사이, 시가→고가, 시가→저가, 시가→종가) 묶음을 날짜만 섞었다. 종목별 창 수익·하루 움직임 크기·종목 간 동시 움직임은 남고, 추세·되돌림 같은 시계열 성질만 사라진다. "
      "매수·보유 포트 수익은 실제와 같고(앞 +65.4%, 뒤 −4.1%, 2025 +74.3%) 낙폭만 달라진다.")
    a(f"- 가짜 자료에서도 G1 앞 효율선초과는 {pct(nul('2026앞', '효율선초과').min())}~{pct(nul('2026앞', '효율선초과').max())}로 넓게 흔들렸다. 실제 G1의 {pct(C['2026앞']['효율선초과'])}는 그 위쪽 끝이지만, "
      f"240개 가운데 가장 좋은 것을 고르면 가짜 자료에서도 1등이 중앙값 {pct(float(np.median(mx)))}에 이른다.")
    a("")
    # ---------- 5
    a("## 5) 장세별")
    a("")
    a("| 창 | 매수·보유 수익 / 낙폭 | G1 수익 / 낙폭 | 효율선초과 | 타이밍 값 (가짜 평균) | 손절 1회 횟수 · 승률 · 손익비 · 기대값 | 가짜 자료 기대값 평균 | 미끄러짐 0.5%: 효율선초과 / 타이밍 값 |")
    a("|---|---|---|---|---|---|---|---|")
    for w in REGIME:
        s = C[w]
        sl = V["slip_왕복없음"][w]
        if w in ("2026앞", "2026뒤", "2025"):
            nt, ne = pp(nul(w, "타이밍값").mean()), pct(nul(w, "기대값").mean())
        elif w in nreg:
            nt = pp(np.mean([r["타이밍값"] for r in nreg[w]]))
            ne = pct(np.mean([r["기대값"] for r in nreg[w]]))
        else:
            nt, ne = "—", "—"
        lab = w + (" (QUBT 제외)" if w == "2018" else "") + (" (5/29까지 정보)" if w == "2026앞" else "")
        a(f"| {lab} | {pct(s['포트BH_수익'])} / {pct(s['포트BH_낙폭'])} | {pct(s['포트_수익'])} / {pct(s['포트_낙폭'])} | {pct(s['효율선초과'])} | "
          f"{pp(s['타이밍값'])} ({nt}) | {s['손절주기']} · {upct(s['승률'])} · {f2(s['손익비'])} · {pct(s['기대값'])} | {ne} | {pct(sl['효율선초과'])} / {pp(sl['타이밍값'])} |")
    a("")
    a("- 하락장(2026 뒤, 2022, 2018)은 매수·보유가 마이너스라 효율선 최선이 현금이고, G1도 마이너스라 효율선을 못 넘었다. 2022에는 손절 1회 기대값이 +3.5%로 좋았지만 "
      "포트 수익은 −20.1%(매수·보유 −30.6%)로 '덜 잃음'에 그쳤다.")
    ctx = "; ".join(f"{w} {pct(C[w]['포트_수익'])}(매수·보유 {pct(C[w]['포트BH_수익'])}, 효율선초과 {pct(C[w]['효율선초과'])})"
                    for w in ("2024", "2023", "2021", "2020", "2019"))
    a(f"- 맥락(한두 줄): {ctx}. 2021에는 효율선초과 {pct(C['2021']['효율선초과'])}로 크게 졌다(장세 전환 위험).")
    a("")
    a("## 판정")
    a("")
    a("**반박(refuted).** 숫자 재현은 정확했지만, 후보가 '선 이탈 때 팔아서' 얻는 값은 확인되지 않았다.")
    a("")
    a("| 확인 | 결과 |")
    a("|---|---|")
    a(f"| 재현 | 주장 {len(cmp)}개 모두 반올림 안 일치, 미끄러짐 0.5% 숫자 일치 |")
    a(f"| 뒤 구간 확인 | 뒤 효율선초과 {pct(C['2026뒤']['효율선초과'])} (효율선 = 현금) → 실패 |")
    a(f"| 비용 | 미끄러짐 0.5%면 앞 {pct(V['slip_왕복없음']['2026앞']['효율선초과'])}, 뒤 {pct(V['slip_왕복없음']['2026뒤']['효율선초과'])}, 2025 {pct(V['slip_왕복없음']['2025']['효율선초과'])} → 모두 마이너스 |")
    a(f"| 하루 늦춘 판 | 비용 0에서는 오히려 나음(2026 {pct(V['delay']['2026']['효율선초과'])}), 비용 0.5%면 2025 {pct(V['delay_slip']['2025']['효율선초과'])} → 지표가 체결 하루에 10%p 흔들림 |")
    a(f"| 이웃 {len(keys)}개 | 앞·뒤·2025 모두 > 0 {upct(all3)}, 비용 0.5%면 {upct(all3s)} |")
    a(f"| 위약 | 원형 이동보다 높지만 가짜 자료에서도 같은 크기(뒤·2025는 범위 안) |")
    a(f"| 선택편향 | 240개 1등 {pct(act_max)} < 가짜 자료 1등 중앙값 {pct(float(np.median(mx)))} |")
    a("")
    a("## 선택편향·한계")
    a("")
    a("- 가짜 자료(날짜 섞기)는 하루 단위 묶음을 통째로 옮겨 변동성 뭉침(큰 날이 몰려 오는 성질)도 없앤다. 손절 규칙이 변동성 뭉침에서 값을 낸다면 가짜 자료가 그 값을 0으로 본다. "
      "다만 그 경우에도 실제 G1이 뒤·2025에서 가짜 범위 안이라는 결과는 그대로다.")
    a("- 2022·2018 귀무는 장세 표를 채우려고 첫 실행 뒤 더했다(후보·이웃 격자는 바꾸지 않음). 귀무 시드는 20개라 '가짜 ≥ 실제' 비율의 해상도가 5%다.")
    a("- 위약 수익은 종가 근사라 실제(장중 체결)와 다르다. 그 차이를 '장중 체결 효과'로 따로 적었다(2026 앞 "
      f"{pp(PL['2026앞']['장중체결효과'])}, 2025 {pp(PL['2025']['장중체결효과'])}).")
    a("- 이웃 격자에서 X·D는 2%p 간격, b는 1%p 간격만 봤다. 되돌림 방식(20거래일 뒤·없음)은 갈래 GRID가 이미 96개로 봤다.")
    a("- 생존편향: 종목 목록이 2026-10 AI_UNIVERSE와 TIME 보유 이력이다. 상장폐지 종목이 없어 깊이 빠진 뒤 되사는 규칙이 실제보다 좋게 나오고, 손절이 막았을 큰 손실은 빠져 있다.")
    a("- 장중 스탑·지정가는 가격에 닿기만 하면 그 가격에 체결된다고 봤다. 미끄러짐은 0과 0.5%만 봤다. 수수료·현금 이자는 0이다.")
    a("")
    a("## 다시 돌리기")
    a("")
    a("```bash")
    a("cd data/derived/analysis/stop_research/code")
    a("PYTHONIOENCODING=utf-8 py p3_V_GRID.py         # 전체(약 1분) → results/p3_V_GRID.json·md")
    a("PYTHONIOENCODING=utf-8 py p3_V_GRID.py md      # json에서 보고서만")
    a("PYTHONIOENCODING=utf-8 py p3_V_GRID.py repro   # 재현 숫자만")
    a("```")
    out = RESULTS / f"{ID}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("보고서", out)
    return out


def repro_only():
    data = load_data()
    runs = {k: make_run(data, k) for k in MAIN}
    print("종목 수", {k: len(r.syms) for k, r in runs.items()}, "빠진 칸", {k: r.miss for k, r in runs.items()}, flush=True)
    print("엔진 점검", engine_checks(data, runs["2026"]), flush=True)
    cand = {}
    for k in MAIN:
        cand.update(all_w(runs[k], run_rule(runs[k], 0.10, 0.20, "recover", 0.03)))
    for r in compare_claim(cand):
        print(r["창"], r["항목"], round(r["주장"], 4), round(r["재계산"], 4), "일치" if r["일치"] else "불일치")
    for w, s in cand.items():
        print(w, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "md":
        report(json.loads((RESULTS / f"{ID}.json").read_text(encoding="utf-8")))
    elif arg == "repro":
        repro_only()
    else:
        compute()
        report(json.loads((RESULTS / f"{ID}.json").read_text(encoding="utf-8")))
