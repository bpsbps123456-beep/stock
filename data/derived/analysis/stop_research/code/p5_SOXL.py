"""5단계 갈래 SOXL(2026-10-11): SOXL 같은 3배 반도체 상품에 손절·추세·변동성 규칙을 걸면 '덜 담기'를 이기나.

사용자: "SOXL이나 KORU 같은 건 손절 규칙 어떻게 잡아?" (미국 상장, 수수료 0, 스탑 주문 가능 — docs/desk_system.md §1)

자료(사전 고정, 결과를 본 뒤 바꾸지 않는다)
  - 기초 = lev5.splice('^SOX', 'SOXX', '2001-07-16')(1994-05~). 상품 = lev5.synth(기초, 3, 보수 0.95%, 추가 비용 1.5%)
    (추가 비용은 실제 SOXL과 겹치는 2010-03~2026-10에서 잰 연 로그 비용 차). 같은 규칙을 실제 SOXL(2010-03~)에 다시 돌린다.
  - 실행: 1995-05-01부터 2026-10-09까지 한 번 이어서 돌린다(상태 이어짐, 앞 1년은 지표 준비). 창은 이 곡선을 잘라 잰다.
    실제 SOXL은 2010-03-12부터(합성도 같은 날 시작한 판을 따로 돌려 나란히 비교).
  - 시점: 비율 w는 t일 종가까지의 기초 지표로 정해 t+1일 시가에 맞춘다(엔진에서는 on_open(t)이 [t−1]을 읽음).
    m(변동성 중앙값)은 σ20의 t−1까지 확장 중앙값. 추적 손절선은 t−1 종가까지의 산 뒤 최고 종가로 정해 t일 장중
    (시가가 선 아래면 시가, 저가가 닿으면 그 값). 현금 이자 0%(연 4% 판은 한 줄).
  - 매일 맞춤 = 목표 비율이 정해진 날마다 시가에 비율을 다시 맞춘다(오르내린 만큼 사고팖). 주 1회 = 주 첫 거래일만.

규칙 묶음(사전 고정, 21개 + 그냥 보유 + 덜 담기 3개)
  R0 그냥 보유(w=1)
  F  덜 담기 w ∈ {0.33, 0.5, 0.67} 매일 맞춤(효율선 재료, 비교 기준)
  T  기초 종가 < SMA n → w=0, 위면 1. n ∈ {50,100,150,200} × 띠 {0, 3%}(이탈 SMA×0.97, 회복 SMA×1.03) = 8,
     + SMA200 이탈 시 w=0.5 = 1
  V  σ20 = 기초 20일 실현변동성(연). w = clip(m/σ20), clip(0.75·m/σ20), clip(σ*/(3σ20)) σ* ∈ {0.45, 0.60} 매일 4개
     + 주 1회 m/σ20, σ*0.60 = 6
  VS σ20 > {1.5, 2.0}×m → 0, 아니면 1 = 2
  TV 기초 > SMA200이고 σ20 < 1.5m → 1, 하나만 → 0.5, 둘 다 아니면 0 = 1
  S  상품 산 뒤 최고 종가 대비 −{30,40,50}% 장중 → 기초 > SMA50(종가) 다음 시가 재진입 = 3

평가(사전 고정)
  - 효율선초과 = 규칙 − (규칙의 최대 낙폭보다 깊지 않은 '덜 담기' w(0~1, 0.01 간격, 매일 맞춤) 가운데 최고 수익).
    짧은 창은 기간 수익(%p), 긴 창은 연율(CAGR %p). 같은노출초과 = 규칙 − 같은 평균 노출 w의 덜 담기.
  - 고르기(후보 21개, R0·F 제외): G1 2026 앞 효율선초과 최대, G2 긴 역사(1995-05~2025-12) 효율선초과(연) 최대,
    G3 2026 앞 칼마(연율/최대 낙폭) 최대. 각 뒤·2025·긴 역사 확인. '앞 > 0' 집합의 뒤·2025 성공 비율 vs 무조건.
  - 비용: 수수료 0, 미끄러짐 0.5%(|Δw|에 곱함, 효율선도 같은 조건) 민감도. 연 회전(Σ|Δw|/년)과 연 매매(|Δw| ≥ 0.1인 날/년).
  - 전멸 = 최대 낙폭 −90% 이하. 최악 1년 = 굴린 252거래일 최저 수익. 전고점 회복 = 물 밑에 있던 가장 긴 기간.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lev5
import stock_lab as S

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
ID = "p5_SOXL"

# ============================================================================ 사전 고정 상수
FEE, EXTRA = 0.0095, 0.015
RUN0, END = "1995-05-01", "2026-10-09"
REAL0 = "2010-03-12"
F_W = (0.33, 0.5, 0.67)
T_N, T_BAND = (50, 100, 150, 200), (0.0, 0.03)
V_SPECS = (("m", 1.0, False), ("m", 0.75, False), ("t", 0.45, False), ("t", 0.60, False), ("m", 1.0, True), ("t", 0.60, True))
VS_K = (1.5, 2.0)
S_X = (0.30, 0.40, 0.50)
SLIP, CASH_ALT = 0.005, 0.04
W_GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
EPS = 1e-6
TRADE_MIN = 0.1

WIN = {  # [첫날, 끝날]: 첫날 직전 종가부터 끝날 종가까지
    "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"), "2025": ("2025-01-02", "2025-12-31"),
    "긴역사_앞이전": (RUN0, "2025-12-31"), "긴역사": (RUN0, END),
}
CRASH = {  # 폭락·반등 구간(맥락, 고르기에 쓰지 않음)
    "2000~02": ("2000-03-01", "2002-10-31"), "2008": ("2007-10-01", "2009-03-31"), "2009반등": ("2009-04-01", "2009-12-31"),
    "2011": ("2011-01-03", "2011-12-30"), "2015~16": ("2015-05-01", "2016-02-29"), "2018Q4": ("2018-10-01", "2018-12-31"),
    "2020급락": ("2020-02-01", "2020-03-31"), "2020반등": ("2020-04-01", "2020-12-31"), "2022": ("2022-01-03", "2022-12-30"),
    "2025관세": ("2025-02-20", "2025-04-08"), "2025반등": ("2025-04-09", "2025-07-31"), "2026_7월": ("2026-06-23", "2026-07-29"),
}
REAL_WIN = {"2026앞": WIN["2026앞"], "2026뒤": WIN["2026뒤"], "2025": WIN["2025"], "2022": CRASH["2022"],
            "2020급락": CRASH["2020급락"], "2020반등": CRASH["2020반등"], "2018Q4": CRASH["2018Q4"], "2011": CRASH["2011"],
            "2015~16": CRASH["2015~16"], "2025관세": CRASH["2025관세"], "2026_7월": CRASH["2026_7월"],
            "실제전체": (REAL0, END)}
LONG = {"긴역사_앞이전", "긴역사", "실제전체"}
PICK, CHECK, AUX, LONGPICK = "2026앞", "2026뒤", "2025", "긴역사_앞이전"


# ============================================================================ 규칙
def rule_list():
    out = [{"id": "R0", "fam": "R0", "label": "그냥 보유(3배 전부)"}]
    for w in F_W:
        out.append({"id": f"F{int(round(w * 100))}", "fam": "F", "w": w, "label": f"덜 담기 {w:.0%}(나머지 현금, 매일 맞춤)"})
    for n in T_N:
        for bd in T_BAND:
            out.append({"id": f"T{n}" + ("b3" if bd else ""), "fam": "T", "n": n, "band": bd,
                        "label": f"기초<{n}일선{' (띠 3%)' if bd else ''} → 현금"})
    out.append({"id": "T200h", "fam": "T", "n": 200, "band": 0.0, "low": 0.5, "label": "기초<200일선 → 절반"})
    for kind, k, wk in V_SPECS:
        if kind == "m":
            rid = f"Vm{int(k * 100)}" + ("w" if wk else "")
            lab = f"변동성 맞춤 w={'' if k == 1 else f'{k:g}·'}m/σ20"
        else:
            rid = f"Vt{int(k * 100)}" + ("w" if wk else "")
            lab = f"목표 변동성 {k:.0%}: w={k:g}/(3σ20)"
        out.append({"id": rid, "fam": "V", "kind": kind, "k": k, "weekly": wk, "label": lab + (" (주 1회)" if wk else " (매일)")})
    for k in VS_K:
        out.append({"id": f"VS{int(k * 10)}", "fam": "VS", "k": k, "label": f"σ20 > {k:g}×m → 현금"})
    out.append({"id": "TV", "fam": "TV", "label": "추세+변동성 합산: 200일선 위·σ20<1.5m 둘 다 1, 하나 0.5, 없음 0"})
    for x in S_X:
        out.append({"id": f"S{int(x * 100)}", "fam": "S", "x": x, "label": f"추적 −{x:.0%} 장중 → 기초>50일선 재진입"})
    return out


def candidates(rules):
    return [r for r in rules if r["fam"] not in ("R0", "F")]


# ============================================================================ 지표(기초 인덱스, [t] = t일 종가까지)
class Ind:
    def __init__(self, u: S.Bars):
        self.u = u
        c = pd.Series(u.c)
        self.c = u.c
        self.sma = {n: c.rolling(n, min_periods=n).mean().to_numpy() for n in (50, 100, 150, 200)}
        r = c.pct_change()
        self.sig = (r.rolling(20, min_periods=20).std() * np.sqrt(252)).to_numpy()
        self.med = pd.Series(self.sig).expanding(min_periods=20).median().shift(1).to_numpy()   # t−1까지
        wk = pd.to_datetime(pd.Series(u.days)).dt.isocalendar()
        self.week = (wk.year.astype(int) * 100 + wk.week.astype(int)).to_numpy()

    def weights(self, sp) -> np.ndarray:
        """t일 종가에 정한 목표 비율 W[t](다음 시가 체결). S는 따로."""
        f, c, n_ = sp["fam"], self.c, len(self.c)
        if f == "R0":
            return np.ones(n_)
        if f == "F":
            return np.full(n_, sp["w"])
        if f == "T":
            m = self.sma[sp["n"]]
            bd, low = sp["band"], sp.get("low", 0.0)
            W = np.ones(n_)
            inn = True
            for t in range(n_):
                if not np.isfinite(m[t]):
                    W[t] = 1.0
                    continue
                if inn and c[t] < m[t] * (1 - bd):
                    inn = False
                elif not inn and c[t] > m[t] * (1 + bd):
                    inn = True
                W[t] = 1.0 if inn else low
            return W
        sig, med = self.sig, self.med
        if f == "V":
            if sp["kind"] == "m":
                W = sp["k"] * med / sig
            else:
                W = sp["k"] / (3 * sig)
            return np.clip(np.nan_to_num(W, nan=1.0), 0, 1)
        if f == "VS":
            return np.where(np.nan_to_num(sig) > sp["k"] * np.nan_to_num(med, nan=np.inf), 0.0, 1.0)
        if f == "TV":
            a = c > np.nan_to_num(self.sma[200], nan=-np.inf)
            b = np.nan_to_num(sig) < 1.5 * np.nan_to_num(med, nan=np.inf)
            return (a.astype(float) + b.astype(float)) / 2
        raise ValueError(f)


# ============================================================================ 엔진
def _lists(P: S.Bars):
    if "lists" not in P._cache:
        P._cache["lists"] = (P.o.tolist(), P.h.tolist(), P.l.tolist(), P.c.tolist())
    return P._cache["lists"]


def simulate(P: S.Bars, a: int, z: int, tgt=None, stop_x=None, reent=None, slip=0.0, cash_rate=0.0):
    """a..z 거래일. a−1 종가에 전량 보유(가치 1). tgt[t] = t일 시가 목표 비율(nan = 그대로).
    stop_x: 추적 손절 비율(산 뒤 최고 종가, t−1까지), reent[t]: t일 시가 재진입 허용(t−1 정보).
    반환 E(종가 가치, 길이 n+1), W(종가 비율), TO(그날 |Δw|)."""
    o, h, l, c = _lists(P)
    n = z - a + 1
    E = np.empty(n + 1); W = np.empty(n); TO = np.zeros(n)
    E[0] = 1.0
    cash, sh = 0.0, 1.0 / c[a - 1]
    peak, out_t = c[a - 1], None
    di = cash_rate / 252
    for k in range(n):
        t = a + k
        ot = o[t]
        eq = cash + sh * ot
        tg = float("nan")
        if stop_x is None:
            tg = tgt[t]
        elif sh <= 0 and out_t is not None and t - 1 >= out_t and reent[t]:
            tg = 1.0
        if tg == tg:
            cur = sh * ot / eq if eq > 0 else 0.0
            d = tg - cur
            if abs(d) > 1e-9:
                if d < 0:
                    sell = -d * eq / ot
                    cash += sell * ot * (1 - slip)
                    sh = 0.0 if tg <= 1e-9 else sh - sell
                else:
                    buy = min(d * eq, cash)
                    sh += buy / (ot * (1 + slip))
                    cash -= buy
                    if stop_x is not None:
                        peak = ot * (1 + slip)
                TO[k] += abs(d)
        if stop_x is not None and sh > 0:
            L = peak * (1 - stop_x)
            px = ot if ot <= L else (L if l[t] <= L else None)
            if px is not None:
                eq = cash + sh * px
                TO[k] += sh * px / eq
                cash += sh * px * (1 - slip)
                sh = 0.0
                out_t = t
        cash *= 1 + di
        ct = c[t]
        E[k + 1] = cash + sh * ct
        W[k] = sh * ct / E[k + 1] if E[k + 1] > 0 else 0.0
        if sh > 0 and ct > peak:
            peak = ct
    return {"E": E, "W": W, "TO": TO, "a": a, "z": z}


def make_inputs(P: S.Bars, ind: Ind, sp: dict, lag: int = 0):
    """상품 날짜에 맞춘 tgt/reent. 상품 날짜는 기초 날짜의 부분집합이어야 한다."""
    ui = np.searchsorted(ind.u.days, P.days)
    assert np.all(ind.u.days[ui] == P.days)
    if sp["fam"] == "S":
        j = np.clip(ui - 1 - lag, 0, None)
        m = ind.sma[50][j]
        reent = (np.isfinite(m) & (ind.c[j] > m)).tolist()
        return {"stop_x": sp["x"], "reent": reent}
    Wd = ind.weights(sp)
    j = np.clip(ui - 1 - lag, 0, None)
    tg = Wd[j].copy()
    if sp.get("weekly"):
        wk = ind.week[ui]
        first = np.concatenate([[True], wk[1:] != wk[:-1]])
        tg = np.where(first, tg, np.nan)
    return {"tgt": tg.tolist()}


def run(P, ind, sp, a, z, slip=0.0, cash_rate=0.0, lag=0, tg_override=None):
    if tg_override is not None:
        return simulate(P, a, z, tgt=tg_override, slip=slip, cash_rate=cash_rate)
    return simulate(P, a, z, slip=slip, cash_rate=cash_rate, **make_inputs(P, ind, sp, lag))


# ============================================================================ 창 통계
def mdd(v):
    return float((v / np.maximum.accumulate(v) - 1).min())


def underwater(v):
    pk = np.maximum.accumulate(v)
    ar = np.arange(len(v))
    last = np.maximum.accumulate(np.where(v >= pk * (1 - 1e-12), ar, 0))
    dur = ar - last
    return float(dur.max() / 252), bool(dur[-1] > 0), float(dur[-1] / 252)


def wstats(res, P, d0, d1, full=False):
    a = res["a"]
    i0, i1 = P.idx(d0), P.idx(d1, "right") - 1
    k0, k1 = i0 - a, i1 - a
    if k0 < 0 or k1 < k0:
        return None
    E = res["E"][k0:k1 + 2]
    n = k1 - k0 + 1
    g = E[-1] / E[0]
    to = res["TO"][k0:k1 + 1]
    yrs = n / 252
    out = {"수익": float(g - 1), "낙폭": mdd(E), "연율": float(g ** (1 / yrs) - 1) if g > 0 else -1.0,
           "평균노출": float(res["W"][k0:k1 + 1].mean()), "연회전": float(to.sum() / yrs), "연매매": float((to >= TRADE_MIN).sum() / yrs),
           "일수": int(n)}
    out["칼마"] = float(out["연율"] / -out["낙폭"]) if out["낙폭"] < -1e-9 else float("nan")
    if full:
        r252 = E[252:] / E[:-252] - 1
        out["최악1년"] = float(r252.min()) if len(r252) else float("nan")
        mx, under, cur = underwater(E)
        out["최장물밑_년"], out["끝에물밑"], out["지금물밑_년"] = mx, under, cur
        out["전멸"] = bool(out["낙폭"] <= -0.90)
    return out


def years_of(res, P, y0=1996, y1=2026):
    out = {}
    for y in range(y0, y1 + 1):
        s = wstats(res, P, f"{y}-01-01", f"{y}-12-31" if y < 2026 else END)
        if s:
            out[str(y)] = s["수익"]
    return out


class Frontier:
    """창별 '덜 담기' w 곡선의 (수익, 연율, 낙폭)."""

    def __init__(self, P, a, z, wins, slip, cash_rate=0.0):
        self.tab = {}
        runs = {w: simulate(P, a, z, tgt=[w] * len(P.c), slip=slip, cash_rate=cash_rate) for w in W_GRID}
        self.runs = runs
        for wn, (d0, d1) in wins.items():
            rows = [wstats(runs[w], P, d0, d1) for w in W_GRID]
            if rows[0] is None:
                continue
            self.tab[wn] = (np.array([r["수익"] for r in rows]), np.array([r["연율"] for r in rows]),
                            np.array([r["낙폭"] for r in rows]))

    def excess(self, wn, st):
        rets, cagrs, mdds = self.tab[wn]
        key = cagrs if wn in LONG else rets
        val = st["연율"] if wn in LONG else st["수익"]
        ok = mdds >= st["낙폭"] - 1e-12
        i = int(np.argmax(np.where(ok, key, -np.inf)))
        j = int(round(min(max(st["평균노출"], 0), 1) * 100))
        return {"효율선초과": float(val - key[i]), "효율선_w": float(W_GRID[i]), "효율선_값": float(key[i]),
                "같은노출초과": float(val - key[j]), "같은노출_w": float(W_GRID[j])}


def evaluate(P, ind, rules, a, z, wins, slip, cash_rate=0.0, full_keys=()):
    fr = Frontier(P, a, z, wins, slip, cash_rate)
    out = {}
    for sp in rules:
        res = run(P, ind, sp, a, z, slip, cash_rate)
        row = {}
        for wn, (d0, d1) in wins.items():
            st = wstats(res, P, d0, d1, full=wn in full_keys)
            if st is None:
                continue
            if wn in fr.tab:
                st.update(fr.excess(wn, st))
            row[wn] = st
        out[sp["id"]] = {"w": row, "res": res}
    return out, fr


# ============================================================================ 점검
def checks(u, P, ind, rules):
    msgs = []
    a, z = P.idx("2015-01-02"), P.idx(END, "right") - 1

    # 1) 엔진을 stock_lab.simulate와 대조(R0, T200b3, S30, Vm100, 미끄러짐 0.5%)
    class Pol(S.Policy):
        def __init__(self, sp):
            self.sp = sp
            self.inp = make_inputs(P, ind, sp)

        def on_open(self, b, t, st):
            if self.sp["fam"] == "S":
                if st["pos"] > 1e-9 or st["exit_t"] is None or st["days_out"] < 1:
                    return None
                return 1.0 if self.inp["reent"][t] else None
            v = self.inp["tgt"][t]
            return None if v != v else v

        def stop(self, b, t, st):
            return st["peak"] * (1 - self.sp["x"]) if self.sp["fam"] == "S" else None

    rl = {r["id"]: r for r in rules}
    for rid in ("R0", "T200b3", "S30", "Vm100", "Vt60w", "TV"):
        v1, _, _ = S.simulate(P, a, z, Pol(rl[rid]), SLIP)
        v2 = run(P, ind, rl[rid], a, z, SLIP)["E"]
        msgs.append(f"엔진 대조 {rid}: stock_lab.simulate와 가치 최대 상대차 {np.max(np.abs(v1 / v2 - 1)):.1e}")

    # 2) 합성 검증(추가 비용 1.5% 넣은 뒤): 실제 SOXL과의 연 로그 비용 차는 0 근처여야 한다
    msgs.append("합성(추가 1.5%) vs 실제 SOXL: " + json.dumps({k: (round(x, 4) if isinstance(x, float) else x)
                                                       for k, x in lev5.validate("SOXL", P).items()}, ensure_ascii=False))

    # 3) 미래 교란: T 이후 기초 가격을 흔들고 합성·지표를 다시 만든 뒤 T까지 가치가 같아야 한다
    T = u.idx("2020-03-02")
    rng = np.random.default_rng(7)
    u2 = S.Bars(u.sym, u.days.copy(), u.o.copy(), u.h.copy(), u.l.copy(), u.c.copy(), u.v.copy())
    for arr in (u2.o, u2.h, u2.l, u2.c):
        arr[T + 1:] *= 1 + rng.normal(0, 0.05, len(arr) - T - 1)
    u2.h[T + 1:] = np.maximum.reduce([u2.h[T + 1:], u2.o[T + 1:], u2.c[T + 1:]])
    u2.l[T + 1:] = np.minimum.reduce([u2.l[T + 1:], u2.o[T + 1:], u2.c[T + 1:]])
    P2, ind2 = lev5.synth(u2, 3, FEE, EXTRA), Ind(u2)
    a0, z0 = P.idx(RUN0), P.idx(END, "right") - 1
    worst = 0.0
    for sp in rules:
        r1 = run(P, ind, sp, a0, z0)["E"]
        r2 = run(P2, ind2, sp, a0, z0)["E"]
        k = T - a0 + 2
        worst = max(worst, float(np.max(np.abs(r1[:k] - r2[:k]))))
    msgs.append(f"미래 교란(2020-03-02 뒤 기초 ±5% 잡음) 규칙 {len(rules)}개: 교란일까지 가치 최대 차 {worst:.1e}(0이어야 함)")
    return msgs


# ============================================================================ 본 실행
def pct(x, d=1):
    return "—" if x is None or x != x else f"{x * 100:+.{d}f}%"


def pp(x, d=1):
    return "—" if x is None or x != x else f"{x * 100:+.{d}f}%p"


def rm(st, cagr=False):
    if st is None:
        return "—"
    v = st["연율"] if cagr else st["수익"]
    return f"{'연 ' if cagr else ''}{v * 100:+.1f}%/{st['낙폭'] * 100:.{1 if cagr else 0}f}%"


def main():
    t0 = time.time()
    u = lev5.splice("^SOX", "SOXX", "2001-07-16")
    ind = Ind(u)
    P = lev5.synth(u, 3, FEE, EXTRA)
    R = lev5.bars("SOXL")
    rules = rule_list()
    cands = candidates(rules)
    msgs = checks(u, P, ind, rules)
    for m in msgs:
        print(m)

    a, z = P.idx(RUN0), P.idx(END, "right") - 1
    allwins = {**WIN, **{k: v for k, v in CRASH.items()}}
    full = ("긴역사", "긴역사_앞이전")
    print("합성 실행 …", round(time.time() - t0, 1))
    syn0, fr0 = evaluate(P, ind, rules, a, z, allwins, 0.0, full_keys=full)
    syn5, fr5 = evaluate(P, ind, rules, a, z, allwins, SLIP, full_keys=full)
    synC, frC = evaluate(P, ind, rules, a, z, {"긴역사": WIN["긴역사"], "긴역사_앞이전": WIN["긴역사_앞이전"]}, 0.0,
                         cash_rate=CASH_ALT, full_keys=full)
    print("실제 SOXL 실행 …", round(time.time() - t0, 1))
    ra, rz = R.idx(REAL0), R.idx(END, "right") - 1
    pa, pz = P.idx(REAL0), P.idx(END, "right") - 1
    real0, _ = evaluate(R, ind, rules, ra, rz, REAL_WIN, 0.0, full_keys=("실제전체",))
    real5, _ = evaluate(R, ind, rules, ra, rz, REAL_WIN, SLIP, full_keys=("실제전체",))
    synR, _ = evaluate(P, ind, rules, pa, pz, REAL_WIN, 0.0, full_keys=("실제전체",))

    # 합성·실제 같은 규칙 일수익 상관
    corr = {}
    for sp in rules:
        e1 = real0[sp["id"]]["res"]["E"]; e2 = synR[sp["id"]]["res"]["E"]
        r1, r2 = e1[1:] / e1[:-1] - 1, e2[1:] / e2[:-1] - 1
        corr[sp["id"]] = float(np.corrcoef(r1, r2)[0, 1]) if np.std(r1) > 0 and np.std(r2) > 0 else float("nan")

    # ------------------------------------------------------------------ 고르기
    def ex(tab, rid, wn, key="효율선초과"):
        return tab[rid]["w"][wn][key]

    g1 = max(cands, key=lambda r: ex(syn0, r["id"], PICK))
    g2 = max(cands, key=lambda r: ex(syn0, r["id"], LONGPICK))
    g3 = max(cands, key=lambda r: np.nan_to_num(syn0[r["id"]]["w"][PICK]["칼마"], nan=-np.inf))
    picks = {"G1": g1, "G2": g2, "G3": g3}

    # 하루 지연(신호를 하루 늦게 읽음) 점검: 고른 규칙 + 긴 역사에서 효율선을 이긴 규칙 + V 전부(사후 점검, 규칙은 그대로)
    lag_ids = []
    for sp in [*picks.values(), *[r for r in cands if ex(syn0, r["id"], LONGPICK) > EPS], *[r for r in cands if r["fam"] == "V"]]:
        if sp["id"] not in lag_ids:
            lag_ids.append(sp["id"])
    rl = {r["id"]: r for r in rules}
    lagged = {}
    for rid in lag_ids:
        res = run(P, ind, rl[rid], a, z, 0.0, 0.0, lag=1)
        row = {}
        for wn in (PICK, CHECK, AUX, "긴역사_앞이전", "긴역사"):
            st = wstats(res, P, *allwins[wn], full=wn in full)
            st.update(fr0.excess(wn, st))
            row[wn] = st
        lagged[rid] = row

    def succ(tab, base_set, wn):
        if not base_set:
            return (0, 0)
        return (sum(1 for r in base_set if ex(tab, r["id"], wn) > EPS), len(base_set))

    sets = {}
    for tabname, tab in (("미끄러짐0", syn0), ("미끄러짐0.5%", syn5)):
        A = [r for r in cands if ex(tab, r["id"], PICK) > EPS]
        Lg = [r for r in cands if ex(tab, r["id"], LONGPICK) > EPS]
        sets[tabname] = {
            "앞>0": {"n": len(A), "ids": [r["id"] for r in A], "뒤": succ(tab, A, CHECK), "2025": succ(tab, A, AUX),
                     "긴역사": succ(tab, A, "긴역사")},
            "무조건": {"n": len(cands), "앞": succ(tab, cands, PICK), "뒤": succ(tab, cands, CHECK), "2025": succ(tab, cands, AUX),
                     "긴역사_앞이전": succ(tab, cands, LONGPICK), "긴역사": succ(tab, cands, "긴역사")},
            "긴역사>0": {"n": len(Lg), "ids": [r["id"] for r in Lg], "앞": succ(tab, Lg, PICK), "뒤": succ(tab, Lg, CHECK),
                       "2025": succ(tab, Lg, AUX)},
        }

    # ------------------------------------------------------------------ 저장(json)
    def strip(tab):
        return {rid: v["w"] for rid, v in tab.items()}

    years = {sp["id"]: years_of(syn0[sp["id"]]["res"], P) for sp in rules}
    payload = {
        "설정": {"기초": "^SOX(~2001-07-13) + SOXX(2001-07-16~)", "합성": f"3배 일일 재설정, 보수 {FEE}, 추가 비용 {EXTRA}, 금융비용 ^IRX",
               "실행": [RUN0, END], "실제": [REAL0, END], "미끄러짐": SLIP, "현금이자_대안": CASH_ALT,
               "효율선_w": "0~1, 0.01 간격, 매일 맞춤", "규칙수": {"후보": len(cands), "R0": 1, "F": len(F_W)}},
        "점검": msgs, "규칙": rules,
        "합성": strip(syn0), "합성_미끄러짐": strip(syn5), "합성_현금4%": strip(synC),
        "실제SOXL": strip(real0), "실제SOXL_미끄러짐": strip(real5), "합성_2010시작": strip(synR), "합성실제_일수익상관": corr,
        "해별": years, "고르기": {k: v["id"] for k, v in picks.items()}, "하루지연": lagged, "성공비율": sets,
        "효율선표": {wn: {"수익": fr0.tab[wn][0].tolist(), "연율": fr0.tab[wn][1].tolist(), "낙폭": fr0.tab[wn][2].tolist()}
                  for wn in (PICK, CHECK, AUX, "긴역사_앞이전", "긴역사")},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{ID}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    # ------------------------------------------------------------------ 보고서(md)
    lab = {r["id"]: r["label"] for r in rules}
    S0 = lambda rid, wn: syn0[rid]["w"].get(wn)
    lines = []
    A = lines.append
    A(f"# 5단계 SOXL: 3배 반도체에 손절·추세·변동성 규칙 (2026-10-11)")
    A("")
    A("> 사용자: \"SOXL이나 KORU 같은 건 손절 규칙 어떻게 잡아?\"")
    A(f"> 기초 = ^SOX(1994-05~2001-07) + SOXX(2001-07~). 상품 = 기초 3배 합성(보수 0.95% + 실제 SOXL과 맞춘 추가 비용 연 1.5%, 금융비용 ^IRX), "
      f"{RUN0}~{END} 한 번 이어서 실행. 같은 규칙을 실제 SOXL(2010-03~)에도 돌렸다. 스크립트 `code/{ID}.py`, 숫자 `results/{ID}.json`.")
    A(f"> 사전 고정 후보 {len(cands)}개(T 9·V 6·VS 2·TV 1·S 3) + 그냥 보유 + 덜 담기 3개. 비율 w는 t일 종가 지표 → t+1일 시가 체결, 현금 이자 0%, 수수료 0.")
    A("")
    lines_concl_idx = len(lines)
    A("## 결론 3줄")
    A("")
    W_ = lambda rid, wn: syn0[rid]["w"][wn]
    sx = ["S30", "S40", "S50"]
    rng_ = lambda vals: f"{min(vals) * 100:+.1f}~{max(vals) * 100:+.1f}"
    A(f"1. **SOXL에 가격 손절(산 뒤 고점 대비 −30·−40·−50% 장중)은 맞지 않았다.** 1995~2026 합성 연 "
      f"{', '.join(pct(W_(r, '긴역사')['연율']) for r in sx)}, 최대 낙폭 모두 −100%(2000~02 "
      f"{rng_([W_(r, '2000~02')['수익'] for r in sx])}%)로 전멸을 못 막았다. 같은 낙폭의 '덜 담기'보다 1995~2025 연 "
      f"{rng_([W_(r, LONGPICK)['효율선초과'] for r in sx])}%p, 실제 SOXL 2010~2026 연 "
      f"{rng_([real0[r]['w']['실제전체']['효율선초과'] for r in sx])}%p였고, 2026 뒤(6~10월)는 "
      f"{rng_([W_(r, CHECK)['수익'] for r in sx])}%로 그냥 보유({pct(W_('R0', CHECK)['수익'])})보다 나빴다.")
    s0_ = sets["미끄러짐0"]
    A(f"2. **2026 기준 합격 규칙은 없다.** 2026 앞으로 고른 G1({lab[g1['id']]})은 앞 {pp(W_(g1['id'], PICK)['효율선초과'])}였지만 "
      f"뒤 {pp(W_(g1['id'], CHECK)['효율선초과'])}, 2025 {pp(W_(g1['id'], AUX)['효율선초과'])}로 무너졌다"
      f"(앞>0 {s0_['앞>0']['n']}/{len(cands)}개 중 뒤 성공 {s0_['앞>0']['뒤'][0]}개, 고르지 않아도 {s0_['무조건']['뒤'][0]}/{len(cands)}). "
      f"긴 역사로 고른 G2({lab[g2['id']]})는 1995~2025 연 {pp(W_(g2['id'], LONGPICK)['효율선초과'])}였지만 2026 앞 "
      f"{pp(W_(g2['id'], PICK)['효율선초과'])}, 2025 {pp(W_(g2['id'], AUX)['효율선초과'])}였다.")
    vt = "Vt45"
    vids = [r["id"] for r in cands if r["fam"] == "V"]
    A(f"3. **전멸을 막은 것은 비중이었다.** 1995~2026에 −90%를 넘지 않은 것은 덜 담기 33%({rm(W_('F33', '긴역사'), True)})와 "
      f"목표 변동성 45%(w = 0.45 ÷ (3 × 기초 20일 변동성), {rm(W_(vt, '긴역사'), True)}, 문턱 바로 위)뿐이다(그냥 보유 {rm(W_('R0', '긴역사'), True)}, "
      f"2000년 고점을 {W_('R0', '긴역사')['지금물밑_년']:.1f}년째 못 되찾음). 변동성 맞춤(V)은 2026 6~10월 고변동 횡보를 줄였고"
      f"(그냥 보유 {pct(W_('R0', CHECK)['수익'])} → 변동성 맞춤 m/σ20 {pct(W_('Vm100', CHECK)['수익'])}, 목표 변동성 45% {pct(W_(vt, CHECK)['수익'])}, "
      f"같은 평균 노출의 덜 담기보다 {rng_([W_(r, CHECK)['같은노출초과'] for r in vids])}%p) 실제 SOXL 2010~2026에서 6개 모두 효율선을 이겼지만"
      f"({rng_([real0[r]['w']['실제전체']['효율선초과'] for r in vids])}%p/연), 2026 앞 1~5월 상승장에서 "
      f"{rng_([W_(r, PICK)['효율선초과'] for r in vids])}%p 뒤져 2026 기준으로는 관찰에 그친다.")
    A("")

    # 사용자 표
    user_rows = ["R0", "F67", "F50", "F33", "T200", "Vm100", "Vt45", "TV", "S30", "S40", "S50"]
    for gk in ("G1", "G2", "G3"):
        if picks[gk]["id"] not in user_rows:
            user_rows.append(picks[gk]["id"])
    A("## 사용자 표 (합성 SOXL, 수익/최대 낙폭, 미끄러짐 0)")
    A("")
    A("| 규칙 | 2026 앞(고르기) | 2026 뒤(확인) | 2025 | 2022 | 2020 2~3월 | 2007.10~09.03 | 2000.03~02.10 | 1995~2026 연율/최대 낙폭 | 전멸(≤−90%) | 연 매매 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for rid in user_rows:
        tag = "".join(f" ({gk})" for gk, sp in picks.items() if sp["id"] == rid)
        L = S0(rid, "긴역사")
        A(f"| {lab[rid]}{tag} | {rm(S0(rid, '2026앞'))} | {rm(S0(rid, '2026뒤'))} | {rm(S0(rid, '2025'))} | {rm(S0(rid, '2022'))} | "
          f"{rm(S0(rid, '2020급락'))} | {rm(S0(rid, '2008'))} | {rm(S0(rid, '2000~02'))} | {rm(L, True)} | "
          f"{'예' if L['전멸'] else '아니오'} | {L['연매매']:.1f} |")
    A("")

    # 효율선 표 (전체 후보)
    A("## 효율선 표: 같은 최대 낙폭의 '덜 담기'보다 더 번 몫")
    A("")
    A("짧은 창은 기간 수익 %p, 긴 역사는 연율 %p. 괄호 = 그 낙폭에서 효율선이 고른 덜 담기 w. 미끄러짐 0.5%는 규칙과 덜 담기 모두에 적용.")
    A("")
    A("| 규칙 | 2026 앞 | 2026 뒤 | 2025 | 1995~2025 (G2 기준) | 1995~2026 | 실제 SOXL 2010~2026(연) | 같은 노출 덜 담기 대비(앞·뒤·2025·1995~2025) | 미끄러짐 0.5%: 앞·뒤·2025·1995~2025 | 평균 노출(1995~2026) | 연 회전 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for sp in rules:
        if sp["fam"] == "F":
            continue
        rid = sp["id"]
        w = syn0[rid]["w"]; w5 = syn5[rid]["w"]
        cell = lambda wn: f"{pp(w[wn]['효율선초과'])} ({w[wn]['효율선_w']:.2f})"
        tag = "".join(f" **{gk}**" for gk, s in picks.items() if s["id"] == rid)
        A(f"| {lab[rid]}{tag} | {cell('2026앞')} | {cell('2026뒤')} | {cell('2025')} | {cell('긴역사_앞이전')} | {cell('긴역사')} | "
          f"{pp(real0[rid]['w']['실제전체']['효율선초과'])} | "
          f"{' · '.join(pp(w[k]['같은노출초과']) for k in ('2026앞', '2026뒤', '2025', '긴역사_앞이전'))} | "
          f"{' · '.join(pp(w5[k]['효율선초과']) for k in ('2026앞', '2026뒤', '2025', '긴역사_앞이전'))} | "
          f"{w['긴역사']['평균노출']:.2f} | {w['긴역사']['연회전']:.1f} |")
    A("")

    # 갈래 요약
    A("갈래별로 효율선을 이긴 규칙 수(미끄러짐 0, > 0):")
    A("")
    A("| 갈래(개수) | 2026 앞 | 2026 뒤 | 2025 | 1995~2025 | 실제 SOXL 2010~2026 | 1995~2026 전멸(≤−90%) 피함 |")
    A("|---|---|---|---|---|---|---|")
    fam_name = {"T": "T 기초 추세", "V": "V 변동성 맞춤", "VS": "VS 변동성 스위치", "TV": "TV 추세+변동성", "S": "S 넓은 추적 손절"}
    for fm, nm in fam_name.items():
        grp = [r["id"] for r in cands if r["fam"] == fm]
        cnt = lambda tab, wn: sum(1 for rid in grp if tab[rid]["w"][wn]["효율선초과"] > EPS)
        A(f"| {nm}({len(grp)}) | {cnt(syn0, PICK)} | {cnt(syn0, CHECK)} | {cnt(syn0, AUX)} | {cnt(syn0, LONGPICK)} | "
          f"{cnt(real0, '실제전체')} | {sum(1 for rid in grp if not syn0[rid]['w']['긴역사']['전멸'])} |")
    A(f"| 덜 담기 F(3) | — | — | — | — | — | {sum(1 for r in rules if r['fam'] == 'F' and not syn0[r['id']]['w']['긴역사']['전멸'])} |")
    A("")

    # 고르기 표
    A("## 고르기 표")
    A("")
    A("| 방법 | 고른 규칙 | 2026 앞 | 2026 뒤 | 2025 | 1995~2025 | 1995~2026 | 하루 늦게 읽으면(앞·뒤·2025·1995~2025 효율선초과) |")
    A("|---|---|---|---|---|---|---|---|")
    desc = {"G1": "2026 앞 효율선초과 최대", "G2": "1995~2025 효율선초과(연) 최대", "G3": "2026 앞 칼마 최대"}
    for gk, sp in picks.items():
        rid = sp["id"]; w = syn0[rid]["w"]; lg = lagged[rid]
        c = lambda wn: f"{rm(w[wn], wn in LONG)} ({pp(w[wn]['효율선초과'])})"
        A(f"| {gk} {desc[gk]} | {lab[rid]} | {c('2026앞')} | {c('2026뒤')} | {c('2025')} | {c('긴역사_앞이전')} | {c('긴역사')} | "
          f"{' · '.join(pp(lg[k]['효율선초과']) for k in ('2026앞', '2026뒤', '2025', '긴역사_앞이전'))} |")
    A("")
    s0 = sets["미끄러짐0"]; s5 = sets["미끄러짐0.5%"]
    fr_ = lambda t: f"{t[0]}/{t[1]}" + (f"({t[0] / t[1]:.0%})" if t[1] else "")
    A("'앞 > 0' 집합이 뒤·2025에도 > 0인 비율 (무조건 = 후보 전부):")
    A("")
    A("| 기준 | 앞 통과 | 뒤 성공 | 2025 성공 | 1995~2026 성공 |")
    A("|---|---|---|---|---|")
    for nm, s in (("미끄러짐 0", s0), ("미끄러짐 0.5%", s5)):
        A(f"| {nm} · 앞>0 | {s['앞>0']['n']}/{len(cands)} | {fr_(s['앞>0']['뒤'])} | {fr_(s['앞>0']['2025'])} | {fr_(s['앞>0']['긴역사'])} |")
        A(f"| {nm} · 무조건 | — | {fr_(s['무조건']['뒤'])} | {fr_(s['무조건']['2025'])} | {fr_(s['무조건']['긴역사'])} |")
        A(f"| {nm} · 1995~2025>0 집합 | {s['긴역사>0']['n']}/{len(cands)} | {fr_(s['긴역사>0']['뒤'])} | {fr_(s['긴역사>0']['2025'])} | 앞 {fr_(s['긴역사>0']['앞'])} |")
    A("")
    A(f"앞>0 규칙(미끄러짐 0): {', '.join(s0['앞>0']['ids']) or '없음'}. 1995~2025>0 규칙: {', '.join(s0['긴역사>0']['ids']) or '없음'}.")
    A("")

    A("하루 늦게 읽기(신호를 t−1 대신 t−2 종가로, 사후 점검·규칙은 그대로) 효율선초과:")
    A("")
    A("| 규칙 | 원래 앞·뒤·2025·1995~2025 | 하루 늦게 앞·뒤·2025·1995~2025 |")
    A("|---|---|---|")
    for rid, lg in lagged.items():
        w = syn0[rid]["w"]
        A(f"| {lab[rid]} | {' · '.join(pp(w[k]['효율선초과']) for k in (PICK, CHECK, AUX, LONGPICK))} | "
          f"{' · '.join(pp(lg[k]['효율선초과']) for k in (PICK, CHECK, AUX, LONGPICK))} |")
    A("")

    # 2026 6~10월 고변동 횡보: V가 줄였나
    A("## 2026 6~10월 고변동 횡보(SOXX는 200일선 위): 변동성 맞춤이 줄였나")
    A("")
    A("| 규칙 | 합성 2026 뒤 | 실제 SOXL 2026 뒤 | 2026 뒤 평균 w | 같은 노출 덜 담기 대비 | 효율선초과 | 2026 7월 AI 쇼크(6/23~7/29) |")
    A("|---|---|---|---|---|---|---|")
    for rid in ["R0", "F50", "T200", "TV", "VS15", "VS20"] + [r["id"] for r in rules if r["fam"] == "V"]:
        w = syn0[rid]["w"]["2026뒤"]; rw = real0[rid]["w"]["2026뒤"]
        A(f"| {lab[rid]} | {rm(w)} | {rm(rw)} | {w['평균노출']:.2f} | {pp(w['같은노출초과'])} | {pp(w['효율선초과'])} | {rm(syn0[rid]['w']['2026_7월'])} |")
    A("")

    # 폭락 구간 표
    A("## 폭락·반등 구간 (합성, 수익/최대 낙폭)")
    A("")
    ck = list(CRASH)
    A("| 규칙 | " + " | ".join(ck) + " | 최악 1년(굴린 252일) | 최장 전고점 회복 | 2026-10 현재 |")
    A("|---|" + "---|" * (len(ck) + 3))
    for sp in rules:
        rid = sp["id"]; w = syn0[rid]["w"]; L = w["긴역사"]
        A(f"| {lab[rid]} | " + " | ".join(rm(w[k]) for k in ck) +
          f" | {pct(L['최악1년'])} | {L['최장물밑_년']:.1f}년 | {'물밑 ' + format(L['지금물밑_년'], '.1f') + '년' if L['끝에물밑'] else '전고점'} |")
    A("")

    # 합성 vs 실제
    A("## 합성 vs 실제 SOXL (같은 날 2010-03-12 시작, 수익/최대 낙폭)")
    A("")
    rk = ["2026앞", "2026뒤", "2025", "2022", "2020급락", "2011", "실제전체"]
    A("| 규칙 | " + " | ".join(f"{k} 합성 / 실제" for k in rk) + " | 일수익 상관 | 실제 효율선초과 앞·뒤·2025 | 실제 미끄러짐 0.5% 효율선초과 앞·뒤·2025 |")
    A("|---|" + "---|" * (len(rk) + 3))
    for sp in rules:
        rid = sp["id"]
        ws, wr, wr5 = synR[rid]["w"], real0[rid]["w"], real5[rid]["w"]
        A(f"| {lab[rid]} | " + " | ".join(f"{rm(ws[k], k in LONG)} / {rm(wr[k], k in LONG)}" for k in rk) +
          f" | {corr[rid]:.3f} | {' · '.join(pp(wr[k].get('효율선초과')) for k in ('2026앞', '2026뒤', '2025'))} | "
          f"{' · '.join(pp(wr5[k].get('효율선초과')) for k in ('2026앞', '2026뒤', '2025'))} |")
    A("")

    # 현금 4%
    A("## 현금 이자 연 4% 판 (1995~2026 연율/최대 낙폭, 효율선초과)")
    A("")
    A("| 규칙 | 이자 0% | 이자 4% | 효율선초과(4%) |")
    A("|---|---|---|---|")
    for rid in user_rows:
        A(f"| {lab[rid]} | {rm(S0(rid, '긴역사'), True)} | {rm(synC[rid]['w']['긴역사'], True)} | {pp(synC[rid]['w']['긴역사']['효율선초과'])} |")
    A("")

    # 해별
    A("## 해별 수익 (합성, 1996~2026.10)")
    A("")
    yr = list(years["R0"])
    A("| 해 | " + " | ".join(lab[r] for r in user_rows) + " |")
    A("|---|" + "---|" * len(user_rows))
    for y in yr:
        A(f"| {y} | " + " | ".join(pct(years[r][y], 0) for r in user_rows) + " |")
    A("")
    A("## 한계")
    A("")
    A("- 시험 범위: 후보 21개(T 9·V 6·VS 2·TV 1·S 3) + 그냥 보유 + 덜 담기 3개, 효율선 w 101칸(0~1, 0.01), 창 5개 + 폭락·반등 12구간, "
      "미끄러짐 2단계, 현금 이자 2단계, 상품 2개(합성·실제). 고르기 3종. 고른 규칙이 21개 중 최대값이라 앞 구간 값은 선택편향으로 부풀어 있다.")
    A("- ^SOX 일봉의 시가는 1994·1997~2001년에 85~96%가 전일 종가와 같다(시가 자료 없음). 그 구간은 '다음 날 시가 체결'이 사실상 "
      "'신호 낸 종가 체결'이 되고, 손절이 갭으로 뚫리는 손실도 못 잰다. 하루 늦게 읽기 점검으로 보완했다.")
    A("- 2001-07 전 기초는 배당 없는 ^SOX 가격 지수다. 추가 비용 1.5%는 배당이 들어간 SOXX 시기로 맞춘 값이라 그 전 구간 합성은 조금 보수적이다. "
      "SOXL은 2021년부터 ICE 반도체 지수를 따른다(같은 종목군).")
    A("- 덜 담기(효율선)는 매일 비율을 맞춘다고 보았다. 미끄러짐 0.5%판에서는 이 매일 맞춤 비용(연 회전 약 3)이 효율선을 낮춰 규칙에 조금 유리하다. "
      "현금 이자 0%가 기본이라 현금을 오래 드는 규칙에 불리하다(연 4%판 표 참고).")
    A("- 2026 뒤 효율선은 거의 w=0(수익 0 근처)이라, 그 창의 효율선초과는 사실상 '규칙 수익 그 자체'다.")
    A("- 2026 7월 AI 쇼크, 2025 관세 등 폭락 구간 날짜는 사후에 정했다(고르기에 쓰지 않음). KORU는 이 갈래에서 재지 않았다.")
    A("")
    A("## 점검")
    A("")
    for m in msgs:
        A(f"- {m}")
    A("")
    (RESULTS / f"{ID}.md").write_text("\n".join(lines), encoding="utf-8")
    print("끝", round(time.time() - t0, 1), "초")
    print("고르기", {k: v["id"] for k, v in picks.items()})
    return payload


if __name__ == "__main__":
    main()
