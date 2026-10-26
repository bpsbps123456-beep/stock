"""4단계 갈래 LEVIDX(2026-10-11): 지수 레버리지 ETF에 손절·추세 필터를 얹으면 '같은 낙폭의 더 낮은 배수로 그냥 들고 있기'를 이기나.

사용자: "일반적인 주식도 한다. 특히 레버리지 주식도 함!" → 지수 레버리지(3배 QQQ·SOXX·SPY·XLK·IWM, 2배 QQQ·SOXX).

설정(사전 고정, 아래 상수. 뒤 구간·2025를 본 뒤 바꾸지 않는다)
  - 상품: 합성(lev_lab.synth, 2014-01-02부터 일일 재설정, 금융비용 ^IRX, 보수 3배 0.95%·2배 1.05%) 7개.
    같은 규칙을 실제 상품(TQQQ·SOXL·UPRO·TECL·TNA·QLD·USD, 모두 2014년 이전 상장)에 다시 돌려 재현을 확인한다.
  - 실행: 2015-01-02 ~ 2026-10-09 한 번 이어서 돌린다(상태 이어짐, 2014년은 지표 준비). 창은 이 곡선을 잘라 잰다.
    시작은 '전량 보유'(2014-12-31 종가). 현금 이자 0%.
  - 시점: 기초 지표(종가·SMA·20일 변동성)와 레버리지 고점은 t−1 종가까지 → t일 시가 체결(stock_lab.simulate 규약).
    추적 손절은 t−1까지의 고점으로 정한 선에 t일 장중(시가가 선 아래면 시가). 지정가 재매수도 t−1 정보로 정한 가격.
  - 비교 기준
      효율선초과 = 규칙 수익 − (규칙보다 깊지 않은 최대 낙폭을 내는 고정 배수 k 들고 있기의 최대 수익).
        k = 0~L, 0.05 간격. k ≤ 1은 기초 일수익 × k(나머지 현금 0%), k > 1은 lev_lab.synth(기초, k)(금융비용·보수 포함).
        실제 상품은 '같은 상품을 f = k/L 비중만 들고 나머지 현금(매일 비중 유지)'으로 효율선을 만든다.
      같은노출초과 = 규칙 수익 − (그 창 평균 노출 × L 배수를 고정으로 들고 있기).
  - 고르기(2026 앞 1/2~5/29만 사용): G1 효율선초과 최대(대표), G2 손절 주기 기대값 최대(규칙 단위로 7상품 합산 30건 이상),
    G3 칼마(연율 수익/최대 낙폭) 최대. 주기는 창 끝까지 못 되샀으면 창 끝 종가로 평가(앞 구간 고르기에 뒤 가격을 쓰지 않음).
  - 비용: 수수료 0, 미끄러짐 0.5%(사고팔 때 각각) 민감도.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lev_lab as LL
import stock_lab as S
import trade_lab as TL

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
ID = "p4_LEVIDX"

# ============================================================================ 사전 고정 격자
PRODUCTS = [  # (실제 상품, 기초, 배수)
    ("TQQQ", "QQQ", 3), ("SOXL", "SOXX", 3), ("UPRO", "SPY", 3), ("TECL", "XLK", 3), ("TNA", "IWM", 3),
    ("QLD", "QQQ", 2), ("USD", "SOXX", 2),
]
T_SMA = (20, 50, 100, 150, 200)          # T 기초 추세: 종가 < SMA → 현금
T_BAND = (0.0, 0.03)                     #   띠(이탈은 SMA×(1−띠) 아래, 회복은 SMA×(1+띠) 위)
S_X = (0.15, 0.20, 0.30, 0.40, 0.50)     # S 레버리지 가격 추적 손절(산 뒤 최고 종가 대비)
S_RE = ("sma50", "brk20", "wait20")      #   재진입: 기초 > SMA50 | 레버리지 종가 > 직전 20일 최고 고가 | 판 뒤 20거래일
V_THR = (0.20, 0.25, 0.35)               # V 기초 20일 실현변동성(연) > 문턱 → 현금
P_SMA = (50, 200)                        # P T(띠 0)의 SMA50·200을 노출 0.5로
P_LOW = 0.5
D_X = (0.30, 0.40)                       # D 추적 손절 뒤
D_D = (0.50, 0.60, 0.70)                 #   고점 대비 −d% 지정가 재매수, 못 닿으면 기초 > SMA50
K_STEP = 0.05                            # 효율선 배수 간격
SLIP = 0.005                             # 미끄러짐 민감도(한 번 거래마다)
RUN = ("2015-01-02", "2026-10-09")
G2_MIN = 30                              # G2 최소 주기 수(7상품 합산)
REF_K = {3: (2.0, 1.5), 2: (1.5, 1.0)}   # 참고 행: 더 낮은 고정 배수
EPS = 1e-6                               # '> 0' 판단 여유(같은 곡선의 부동소수 차이를 이김으로 세지 않음)

WIN = {  # 창 [첫날, 끝날] — 첫날 직전 종가부터 끝날 종가까지
    "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"), "2026": ("2026-01-02", "2026-10-09"),
    "2025": ("2025-01-02", "2025-12-31"), "2024": ("2024-01-02", "2024-12-31"), "2023": ("2023-01-03", "2023-12-29"),
    "2022": ("2022-01-03", "2022-12-30"), "2021": ("2021-01-04", "2021-12-31"), "2020": ("2020-01-02", "2020-12-31"),
    "2019": ("2019-01-02", "2019-12-31"), "2018": ("2018-01-02", "2018-12-31"),
    # 폭락 구간(맥락, 사후에 정한 날짜라 고르기에 쓰지 않음)
    "2018Q4": ("2018-10-01", "2018-12-31"), "2020급락": ("2020-02-20", "2020-03-23"), "2020반등": ("2020-03-24", "2020-08-31"),
    "2025관세": ("2025-02-20", "2025-04-08"), "2025반등": ("2025-04-09", "2025-07-31"), "AI쇼크": ("2026-06-23", "2026-07-29"),
    "전체": RUN,
}
PICK, CHECK, AUX = "2026앞", "2026뒤", "2025"


# ============================================================================ 규칙
def rule_list():
    out = []
    for n in T_SMA:
        for bd in T_BAND:
            out.append({"id": f"T{n}" + (f"b{int(bd * 100)}" if bd else ""), "fam": "T", "n": n, "band": bd,
                        "label": f"기초<SMA{n}{f' (띠 {bd:.0%})' if bd else ''} → 현금"})
    re_lab = {"sma50": "기초>SMA50", "brk20": "20일 고가 돌파", "wait20": "20거래일 뒤"}
    for x in S_X:
        for re in S_RE:
            out.append({"id": f"S{int(x * 100)}_{re}", "fam": "S", "x": x, "re": re,
                        "label": f"추적 −{x:.0%} → 재진입 {re_lab[re]}"})
    for v in V_THR:
        out.append({"id": f"V{int(v * 100)}", "fam": "V", "thr": v, "label": f"기초 20일 변동성 > {v:.0%} → 현금"})
    for n in P_SMA:
        out.append({"id": f"P{n}", "fam": "P", "n": n, "label": f"기초<SMA{n} → 절반(0.5)"})
    for x in D_X:
        for d in D_D:
            out.append({"id": f"D{int(x * 100)}_{int(d * 100)}", "fam": "D", "x": x, "d": d,
                        "label": f"추적 −{x:.0%} → 고점 −{d:.0%} 지정가 재매수(못 닿으면 기초>SMA50)"})
    return out


class Rule(S.Policy):
    """u = 같은 날짜의 기초 Bars. 노출은 0/1(P는 0.5/1). 내려간 날과 절반 주기를 스스로 기록한다."""

    def __init__(self, spec: dict, u: S.Bars, slip: float = 0.0):
        self.sp, self.u, self.slip = spec, u, slip
        self.inn = True
        self.down_days = []         # 노출을 줄인 날 인덱스(시가 판단분)
        self.half = []              # P 절반 주기 (exit_t, exit_px, re_t, re_px)
        self._open_half = None

    # --- 시가 판단(t−1 종가까지)
    def on_open(self, b, t, st):
        sp, u = self.sp, self.u
        f = sp["fam"]
        if f in ("T", "P"):
            m = u.sma(sp["n"])[t - 1]
            if not np.isfinite(m):
                return None
            band = sp.get("band", 0.0)
            low = 0.0 if f == "T" else P_LOW
            cu = u.c[t - 1]
            if self.inn and cu < m * (1 - band):
                self.inn = False
                self.down_days.append(t)
                if f == "P":
                    self._open_half = (t, b.o[t] * (1 - self.slip))
                return low
            if not self.inn and cu > m * (1 + band):
                self.inn = True
                if f == "P" and self._open_half:
                    self.half.append((*self._open_half, t, b.o[t] * (1 + self.slip)))
                    self._open_half = None
                return 1.0
            return None
        if f == "V":
            rv = u.rvol(20)[t - 1]
            if not np.isfinite(rv):
                return None
            if self.inn and rv > sp["thr"]:
                self.inn = False
                self.down_days.append(t)
                return 0.0
            if not self.inn and rv <= sp["thr"]:
                self.inn = True
                return 1.0
            return None
        # S·D: 판 뒤 재진입
        if st["pos"] > 1e-9 or st["exit_t"] is None or st["days_out"] < 1:
            return None
        re = sp.get("re", "sma50")
        if re == "sma50":
            m = u.sma(50)[t - 1]
            return 1.0 if np.isfinite(m) and u.c[t - 1] > m else None
        if re == "brk20":
            hh = b.high_n(20)[t - 2]
            return 1.0 if np.isfinite(hh) and b.c[t - 1] > hh else None
        if re == "wait20":
            return 1.0 if t - st["exit_t"] >= 20 else None
        raise ValueError(re)

    def stop(self, b, t, st):
        if self.sp["fam"] in ("S", "D"):
            return st["peak"] * (1 - self.sp["x"])
        return None

    def limit_buys(self, b, t, st):
        if self.sp["fam"] == "D" and st["pos"] <= 1e-9 and st["exit_t"] is not None:
            return [(st["peak"] * (1 - self.sp["d"]), 1.0)]
        return ()


class Hold(S.Policy):
    pass


# ============================================================================ 자료·고정 배수
_RF = None


def rf_arr(days):
    global _RF
    if _RF is None:
        _RF = LL.rf_series()
    return _RF.reindex(days).ffill().fillna(0.0).to_numpy()


def k_daily(u: S.Bars, k: float) -> np.ndarray:
    """고정 k배(일일 재설정) 종가 일수익 [t](t−1→t), [0] = nan. lev_lab.synth의 종가 식과 같다(k ≤ 1은 현금 0%)."""
    ru = np.concatenate([[np.nan], u.c[1:] / u.c[:-1] - 1])
    if k <= 1 + 1e-12:
        return k * ru
    fee = LL.FEE.get(int(k), 0.01)
    r = k * ru - (k - 1) * rf_arr(u.days) / 252 - fee / 252
    return np.maximum(r, -0.99)


def mdd(v):
    return float((v / np.maximum.accumulate(v) - 1).min())


def curve(r_daily, wa, wz):
    return np.concatenate([[1.0], np.cumprod(1 + np.nan_to_num(r_daily[wa:wz + 1]))])


class Frontier:
    """창별 고정 배수 효율선 (k, 수익, 낙폭)."""

    def __init__(self, daily_by_k: dict, wins: dict):
        self.tab = {}
        for w, (wa, wz) in wins.items():
            ks = sorted(daily_by_k)
            rets, mdds = [], []
            for k in ks:
                v = curve(daily_by_k[k], wa, wz)
                rets.append(v[-1] - 1)
                mdds.append(mdd(v))
            self.tab[w] = (np.array(ks), np.array(rets), np.array(mdds))

    def excess(self, w, ret, dd):
        """(효율선초과, 그 k, 그 수익, 같은낙폭 k(낙폭이 더 깊지 않은 가장 큰 k), 그 수익)."""
        ks, rets, mdds = self.tab[w]
        ok = mdds >= dd - 1e-12
        i = int(np.argmax(np.where(ok, rets, -np.inf)))
        j = int(np.max(np.flatnonzero(ok)))
        return float(ret - rets[i]), float(ks[i]), float(rets[i]), float(ks[j]), float(rets[j])


# ============================================================================ 한 상품 × 규칙
def run_one(b: S.Bars, u: S.Bars, spec: dict | None, slip: float):
    a, z = b.idx(RUN[0]), b.idx(RUN[1], "right") - 1
    pol = Rule(spec, u, slip) if spec else Hold()
    vals, cycles, events = S.simulate(b, a, z, pol, slip)
    expo = S.simulate.last_pos.copy()
    sells = {int(np.searchsorted(b.days, e[0])) for e in events if e[1] == "sell"}
    downs = sorted(sells | set(getattr(pol, "down_days", [])))
    cyc = []
    for cy in cycles:
        cyc.append((int(np.searchsorted(b.days, cy["exit_day"])), cy["exit_px"],
                    None if cy["re_day"] is None else int(np.searchsorted(b.days, cy["re_day"])), cy["re_px"], 1.0))
    for (et, ep, rt, rp) in getattr(pol, "half", []):
        cyc.append((et, ep, rt, rp, 1 - P_LOW))
    if spec and spec["fam"] == "P" and pol._open_half:
        cyc.append((pol._open_half[0], pol._open_half[1], None, None, 1 - P_LOW))
    return {"a": a, "z": z, "vals": vals, "expo": expo, "downs": downs, "cyc": cyc}


def win_idx(b: S.Bars):
    out = {}
    for w, (d0, d1) in WIN.items():
        out[w] = (b.idx(d0), b.idx(d1, "right") - 1)
    return out


def metrics(res, b: S.Bars, wins, L, front: Frontier, eqexp_daily) -> dict:
    a = res["a"]
    out = {}
    for w, (wa, wz) in wins.items():
        v = res["vals"][wa - a: wz - a + 2]
        v = v / v[0]
        n = wz - wa + 1
        ret, dd = float(v[-1] - 1), mdd(v)
        ex = res["expo"][wa - a: wz - a + 1]
        e_mean = float(ex.mean())
        fx, fk, fret, mk, mret = front.excess(w, ret, dd)
        eq = curve(eqexp_daily(e_mean), wa, wz)
        ann = lambda x: (1 + x) ** (252 / n) - 1 if x > -1 else -1.0
        cagr = ann(ret)
        downs = [t for t in res["downs"] if wa <= t <= wz]
        g, wts = [], []
        for (et, ep, rt, rp, wt) in res["cyc"]:
            if not (wa <= et <= wz):
                continue
            px = rp if (rt is not None and rt <= wz) else b.c[wz]
            g.append(ep / px - 1)
            wts.append(wt)
        g = np.array(g)
        win, loss = g[g > 0], g[g <= 0]
        out[w] = {
            "수익": ret, "낙폭": dd, "칼마": float(cagr / -dd) if dd < -1e-9 else float("nan"), "연율": float(cagr),
            "효율선초과": fx, "효율선_k": fk, "효율선_수익": fret, "효율선초과_연": float(cagr - ann(fret)),
            "같은낙폭초과": float(ret - mret), "같은낙폭_k": mk, "같은낙폭_수익": mret,
            "평균노출": e_mean, "평균배수": e_mean * L, "같은노출_수익": float(eq[-1] - 1), "같은노출초과": float(ret - (eq[-1] - 1)),
            "같은노출초과_연": float(cagr - ann(eq[-1] - 1)), "보유대비": float("nan"),
            "줄임횟수": len(downs), "연줄임": float(len(downs) / (n / 252)),
            "주기": int(len(g)), "승률": float(len(win) / len(g)) if len(g) else float("nan"),
            "평균이익": float(win.mean()) if len(win) else float("nan"), "평균손실": float(loss.mean()) if len(loss) else float("nan"),
            "손익비": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
            "기대값": float(g.mean()) if len(g) else float("nan"), "_g": g.tolist(),
        }
    # 전체 구간 꼬리(굴린 창)
    v = res["vals"]
    r60 = v[60:] / v[:-60] - 1
    r252 = v[252:] / v[:-252] - 1
    out["전체"]["최악1년"] = float(r252.min())
    out["전체"]["P(60일≤−30%)"] = float((r60 <= -0.30).mean())
    out["전체"]["60일하위5%"] = float(np.quantile(r60, 0.05))
    return out


# ============================================================================ 점검
def checks(us, synths):
    msgs = []
    u = us["QQQ"]
    for L in (2, 3):
        s = LL.synth("QQQ", L)
        rs = np.concatenate([[np.nan], s.c[1:] / s.c[:-1] - 1])
        d = np.nanmax(np.abs(rs - k_daily(u, L)))
        msgs.append(f"고정 배수 식 vs lev_lab.synth 종가 일수익 최대 차 L={L}: {d:.2e}")
    b = synths["TQQQ"]
    h = run_one(b, u, None, 0.0)
    a, z = h["a"], h["z"]
    bh = np.concatenate([[1.0], b.c[a:z + 1] / b.c[a - 1]])
    msgs.append(f"보유 정책 vs 합성 매수·보유 최대 차(0이어야 함): {np.max(np.abs(h['vals'] - bh)):.2e}")
    # 미래 교란: T200과 S30 — T 이후 가격을 바꿔도 T까지 가치가 같아야 한다
    rl = {r["id"]: r for r in rule_list()}
    for rid in ("T200b3", "S30_sma50", "D30_50", "V25", "P50"):
        r1 = run_one(b, u, rl[rid], 0.0)
        T = b.idx("2020-03-02")
        rng = np.random.default_rng(7)
        b2 = S.Bars(b.sym, b.days, b.o.copy(), b.h.copy(), b.l.copy(), b.c.copy(), b.v)
        u2 = S.Bars(u.sym, u.days, u.o.copy(), u.h.copy(), u.l.copy(), u.c.copy(), u.v)
        for bb in (b2, u2):
            for arr in (bb.o, bb.h, bb.l, bb.c):
                arr[T + 1:] *= 1 + rng.normal(0, 0.05, len(arr) - T - 1)
            bb.h[T + 1:] = np.maximum.reduce([bb.h[T + 1:], bb.o[T + 1:], bb.c[T + 1:]])
            bb.l[T + 1:] = np.minimum.reduce([bb.l[T + 1:], bb.o[T + 1:], bb.c[T + 1:]])
        r2 = run_one(b2, u2, rl[rid], 0.0)
        k = T - r1["a"] + 2
        msgs.append(f"미래 교란 {rid}: 교란일까지 가치 최대 차(0이어야 함) {np.max(np.abs(r1['vals'][:k] - r2['vals'][:k])):.2e}")
    return msgs


# ============================================================================ 본 실행
def main():
    t0 = time.time()
    lev = LL.load()
    rules = rule_list()
    unders = sorted({u for _, u, _ in PRODUCTS})
    us = {u: lev[u] for u in unders}
    synths = {p: LL.synth(u, L) for p, u, L in PRODUCTS}
    reals = {p: lev[p] for p, _, _ in PRODUCTS}
    excluded = [p for p in reals if TL.bad_bars(reals[p])]
    grid_n = len(rules) * len(PRODUCTS)
    print(f"규칙 {len(rules)}개(T {len(T_SMA) * len(T_BAND)}, S {len(S_X) * len(S_RE)}, V {len(V_THR)}, P {len(P_SMA)}, "
          f"D {len(D_X) * len(D_D)}) × 상품 {len(PRODUCTS)} = {grid_n} 조합 · 자료 오류 의심 상품 {excluded}", flush=True)
    chk = checks(us, synths)
    for m in chk:
        print(" ", m, flush=True)

    # 효율선(합성: k배 고정, 실제: 같은 상품 f=k/L 비중)
    fronts, eqfun = {}, {}
    for p, u, L in PRODUCTS:
        ks = [round(k, 2) for k in np.arange(0, L + 1e-9, K_STEP)]
        wins = win_idx(synths[p])
        fronts[("synth", p)] = Frontier({k: k_daily(us[u], k) for k in ks}, wins)
        rr = np.concatenate([[np.nan], reals[p].c[1:] / reals[p].c[:-1] - 1])
        fronts[("real", p)] = Frontier({k: (k / L) * rr for k in ks}, win_idx(reals[p]))
        eqfun[("synth", p)] = (lambda e, uu=us[u], LL_=L: k_daily(uu, e * LL_))
        eqfun[("real", p)] = (lambda e, rr_=rr: e * rr_)
    print(f"효율선 준비 {time.time() - t0:.0f}s", flush=True)

    rows = []
    for kind, bars in (("synth", synths), ("real", reals)):
        for slip in (0.0, SLIP):
            if kind == "real" and slip > 0:
                continue
            for p, u, L in PRODUCTS:
                b = bars[p]
                wins = win_idx(b)
                for spec in [None] + rules:
                    res = run_one(b, us[u], spec, slip)
                    m = metrics(res, b, wins, L, fronts[(kind, p)], eqfun[(kind, p)])
                    rows.append({"kind": kind, "slip": slip, "product": p, "under": u, "L": L,
                                 "rule": spec["id"] if spec else "보유", "fam": spec["fam"] if spec else "보유",
                                 "label": spec["label"] if spec else f"{p} 들고만 있기", "m": m})
                # 참고: 더 낮은 고정 배수로 그냥 들고 있기(합성 = k배 일일 재설정, 실제 = 같은 상품 k/L 비중 + 현금)
                if slip == 0.0:
                    for k in REF_K[L]:
                        dly = k_daily(us[u], k) if kind == "synth" else eqfun[("real", p)](k / L)
                        a, z = b.idx(RUN[0]), b.idx(RUN[1], "right") - 1
                        res = {"a": a, "z": z, "vals": curve(dly, a, z), "expo": np.full(z - a + 1, k / L), "downs": [], "cyc": []}
                        m = metrics(res, b, wins, L, fronts[(kind, p)], eqfun[(kind, p)])
                        lab_k = f"고정 {k:g}배 들고만 있기" + ("" if kind == "synth" else f"({p} {k / L:.0%} + 현금)")
                        rows.append({"kind": kind, "slip": slip, "product": p, "under": u, "L": L, "rule": f"K{k:g}",
                                     "fam": "고정", "label": lab_k, "m": m})
            print(f"{kind} slip {slip} 끝 {time.time() - t0:.0f}s", flush=True)
    analyse(rules, rows, chk, grid_n, excluded)
    print(f"총 {time.time() - t0:.0f}s")



# ============================================================================ 분석·보고
def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def pct0(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


FAM_NAME = {"T": "T 기초 추세", "S": "S 추적 손절", "V": "V 변동성", "P": "P 절반", "D": "D 깊은 재매수"}
CRASH = ["2018Q4", "2020급락", "2020반등", "2022", "2025관세", "2025반등", "AI쇼크"]
DOWN4 = ["2018Q4", "2020급락", "2022", "2025관세"]
CRASH_NAME = {"2018Q4": "2018 4분기", "2020급락": "2020 급락", "2020반등": "2020 반등", "2022": "2022", "2025관세": "2025 관세",
              "2025반등": "2025 반등", "AI쇼크": "2026 AI 쇼크"}


def analyse(rules, rows, chk, grid_n, excluded):
    R = {(r["kind"], r["slip"], r["product"], r["rule"]): r for r in rows}
    prods = [p for p, _, _ in PRODUCTS]
    rids = [r["id"] for r in rules]
    lab = {r["id"]: r["label"] for r in rules}
    lab["보유"] = "들고만 있기"
    fam = {r["id"]: r["fam"] for r in rules}

    def M(kind, p, rid, w, key, slip=0.0):
        return R[(kind, slip, p, rid)]["m"][w][key]

    def L_of(kind, p, rid):
        r = R[(kind, 0.0, p, rid)]
        return r["label"] if r["fam"] == "고정" else lab[rid]

    combos = [(p, rid) for p in prods for rid in rids]
    pos = lambda x: np.isfinite(x) and x > EPS

    # ---------------- 고르기(2026 앞만)
    g1 = max(combos, key=lambda c: M("synth", *c, PICK, "효율선초과"))
    g3 = max(combos, key=lambda c: np.nan_to_num(M("synth", *c, PICK, "칼마"), nan=-9))
    per_g1 = {p: max(rids, key=lambda rid: M("synth", p, rid, PICK, "효율선초과")) for p in prods}

    def pooled_win(rid, w):
        g = sum((M("synth", p, rid, w, "_g") for p in prods), [])
        return len(g), (float(np.mean(g)) if g else float("nan"))

    pooled = {rid: pooled_win(rid, PICK) for rid in rids}
    g2_min = G2_MIN
    cand = [rid for rid in rids if pooled[rid][0] >= g2_min]
    if not cand:
        g2_min = 10
        cand = [rid for rid in rids if pooled[rid][0] >= g2_min]
    g2 = max(cand, key=lambda r: pooled[r][1]) if cand else None

    # ---------------- '앞 > 0' 집합 성공률
    def pass_set(key, name, base=None):
        thr = (lambda c, w: np.nan_to_num(M("synth", c[0], "보유", w, key), nan=-9)) if base else (lambda c, w: 0.0)
        val = lambda c, w: np.nan_to_num(M("synth", *c, w, key), nan=-9)
        sel = [c for c in combos if val(c, PICK) > thr(c, PICK) + EPS]
        out = {"기준": name, "앞통과": len(sel), "조합수": len(combos)}
        for w in (CHECK, AUX, "전체"):
            succ = [c for c in sel if val(c, w) > thr(c, w) + EPS]
            allc = [c for c in combos if val(c, w) > thr(c, w) + EPS]
            out[w] = [len(succ) / len(sel) if sel else float("nan"), len(allc) / len(combos)]
        return out

    ps = [pass_set("효율선초과", "효율선초과 > 0"), pass_set("같은낙폭초과", "같은 낙폭 배수 대비 > 0"),
          pass_set("같은노출초과", "같은 평균 노출 배수 대비 > 0"), pass_set("칼마", "칼마 > 그 상품 들고만 있기", base=True),
          pass_set("수익", "수익 > 그 상품 들고만 있기", base=True)]
    sel = [rid for rid in rids if pooled[rid][0] >= 5 and pooled[rid][1] > 0]
    ev_ps = {"기준": "손절 주기 기대값 > 0(규칙 단위 7상품 합산, 주기 5건 이상)", "앞통과": len(sel), "조합수": len(rids)}
    for w in (CHECK, AUX, "전체"):
        pw = {rid: pooled_win(rid, w) for rid in rids}
        succ = [rid for rid in sel if pw[rid][0] and pw[rid][1] > 0]
        allr = [rid for rid in rids if pw[rid][0] and pw[rid][1] > 0]
        ev_ps[w] = [len(succ) / len(sel) if sel else float("nan"), len(allr) / len(rids)]
    all3 = [c for c in combos if all(pos(M("synth", *c, w, "효율선초과")) for w in (PICK, CHECK, AUX))]

    # ---------------- 계열 요약(합성, 미끄러짐 0)
    famA, famB = [], []
    for f in FAM_NAME:
        cs = [c for c in combos if fam[c[1]] == f]
        row = {"계열": f, "조합": len(cs)}
        for w in (PICK, CHECK, AUX, "2024", "2023", "2021", "2019"):
            xs = np.array([M("synth", *c, w, "효율선초과") for c in cs])
            row[w] = [float(np.median(xs)), float(np.mean(xs > EPS))]
        xs = np.array([M("synth", *c, "전체", "효율선초과_연") for c in cs])
        row["전체_연"] = [float(np.median(xs)), float(np.mean(xs > EPS))]
        xs = np.array([M("synth", *c, "전체", "효율선초과_연", SLIP) for c in cs])
        row["전체_연_슬립"] = [float(np.median(xs)), float(np.mean(xs > EPS))]
        row["연줄임"] = float(np.median([M("synth", *c, "전체", "연줄임") for c in cs]))
        famA.append(row)
        rb = {"계열": f}
        for w in CRASH:
            d = np.array([M("synth", *c, w, "수익") - M("synth", c[0], "보유", w, "수익") for c in cs])
            x = np.array([M("synth", *c, w, "같은낙폭초과") for c in cs])
            rb[w] = [float(np.median(d)), float(np.median(x)), float(np.mean(x > EPS))]
        famB.append(rb)

    # ---------------- 실제 상품 재현
    rep = {}
    for w in (PICK, CHECK, AUX, "2022", "2020", "전체"):
        key = "연율" if w == "전체" else "수익"
        a = np.array([M("synth", *c, w, key) for c in combos])
        bb = np.array([M("real", *c, w, key) for c in combos])
        da = np.array([M("synth", *c, w, key) - M("synth", c[0], "보유", w, key) for c in combos])
        db = np.array([M("real", *c, w, key) - M("real", c[0], "보유", w, key) for c in combos])
        ea = np.array([M("synth", *c, w, "효율선초과") for c in combos])
        eb = np.array([M("real", *c, w, "효율선초과") for c in combos])
        rep[w] = {"상관": float(np.corrcoef(a, bb)[0, 1]), "차_중앙": float(np.median(bb - a)),
                  "보유대비_부호일치": float(np.mean((da > EPS) == (db > EPS))),
                  "효율선_부호일치": float(np.mean((ea > EPS) == (eb > EPS))),
                  "효율선>0_합성": float(np.mean(ea > EPS)), "효율선>0_실제": float(np.mean(eb > EPS))}

    # ---------------- 미끄러짐
    slip_rows = []
    for w in (PICK, CHECK, AUX, "전체"):
        key = "연율" if w == "전체" else "수익"
        d = np.array([M("synth", *c, w, key, SLIP) - M("synth", *c, w, key) for c in combos])
        ex0 = np.array([M("synth", *c, w, "효율선초과") for c in combos])
        ex1 = np.array([M("synth", *c, w, "효율선초과", SLIP) for c in combos])
        slip_rows.append({"창": w, "단위": key, "변화_중앙": float(np.median(d)), "변화_하위10%": float(np.quantile(d, 0.10)),
                          "효율선>0_전": float(np.mean(ex0 > EPS)), "효율선>0_후": float(np.mean(ex1 > EPS))})
    xs = np.array([M("synth", *c, "전체", "연줄임") for c in combos])
    ys = np.array([M("synth", *c, "전체", "연율", SLIP) - M("synth", *c, "전체", "연율") for c in combos])
    slope = float(np.polyfit(xs, ys, 1)[0])

    # ============================================================ 문서
    def rr(kind, p, rid, w, slip=0.0):
        m = R[(kind, slip, p, rid)]["m"][w]
        return f"{pct(m['수익'])}/{pct0(m['낙폭'], 0)}"

    def full(kind, p, rid, slip=0.0):
        m = R[(kind, slip, p, rid)]["m"]["전체"]
        return f"연 {pct(m['연율'])}/{pct0(m['낙폭'], 0)}"

    def ex3(kind, p, rid):
        if rid == "보유":
            return "기준"
        return " · ".join(pct(M(kind, p, rid, w, "효율선초과")) for w in (PICK, CHECK, AUX))

    lines = []
    A = lines.append
    A("# 4단계 LEVIDX: 지수 레버리지 ETF에 손절·추세 필터 (2026-10-11)")
    A("")
    A("> 사용자: \"7종목만 거래하는 게 아니라 일반적인 주식도 한다. 특히 레버리지 주식도 함!\"")
    A("> 대상: 3배 QQQ·SOXX·SPY·XLK·IWM, 2배 QQQ·SOXX. 합성(2014~)으로 고르고 실제 TQQQ·SOXL·UPRO·TECL·TNA·QLD·USD로 같은 규칙을 다시 돌렸다. "
      "스크립트 `code/p4_LEVIDX.py`, 숫자 `results/p4_LEVIDX.json`.")
    A("> 핵심 비교: 규칙 vs '같은 최대 낙폭이 나오는 더 낮은 고정 배수로 그냥 들고 있기'(효율선). 규칙이 낙폭을 줄인 만큼만 수익을 잃으면 0, 더 벌면 +.")
    A("")
    A("## 결론")
    A("")
    A("@@CONCLUSION@@")
    A("")
    # ---- 핵심 표(실제 상품)
    A("## 핵심 표: 실제 TQQQ·SOXL (수익/최대 낙폭, 수수료 0, 미끄러짐 0, 현금 이자 0)")
    A("")
    A("효율선초과 = 같은 최대 낙폭이 나오는 '같은 상품을 덜 들고(나머지 현금) 그냥 있기'보다 더 번 몫(앞·뒤·2025). 규칙은 합성 2026 앞으로 골랐다.")
    A("")
    A("| 상품 | 규칙 | 2026 앞(고르기) | 2026 뒤(확인) | 2025 | 2022 | 2020 급락 | 2018 4분기 | 2026 AI 쇼크 | 2015~2026 | 효율선초과 앞·뒤·2025 | 연 들락 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    core = {}
    for p in ("TQQQ", "SOXL"):
        lst = []
        for rid in ["보유", "K2", "T200", "T200b3", per_g1[p], "V25", "P200"]:
            if rid not in lst:
                lst.append(rid)
        core[p] = lst
        for rid in lst:
            tag = " (이 상품 G1)" if rid == per_g1[p] else ""
            tag += " (전체 G1)" if (p, rid) == g1 else ""
            A(f"| {p} | {L_of('real', p, rid)}{tag} | " + " | ".join(rr("real", p, rid, w) for w in (PICK, CHECK, AUX, "2022", "2020급락", "2018Q4", "AI쇼크"))
              + f" | {full('real', p, rid)} | {ex3('real', p, rid)} | {num(M('real', p, rid, '전체', '연줄임'), 1)} |")
    A("")
    A("효율선 기준점 예(합성, 기초<SMA200 → 현금): 효율선 = 낙폭이 더 깊지 않은 고정 배수 중 최대 수익, 같은 낙폭 = 낙폭이 더 깊지 않은 가장 큰 배수.")
    A("")
    for p in ("TQQQ", "SOXL"):
        parts = []
        for w in (PICK, CHECK, AUX, "2022", "전체"):
            m = R[("synth", 0.0, p, "T200")]["m"][w]
            if w == "전체":
                parts.append(f"2015~2026 연 {pct(m['연율'])}/{pct0(m['낙폭'], 0)} ↔ 고정 {m['효율선_k']:.2f}배 연 {pct(m['연율'] - m['효율선초과_연'])}")
            else:
                parts.append(f"{w} {pct(m['수익'])}/{pct0(m['낙폭'], 0)} ↔ 효율선 {m['효율선_k']:.2f}배 {pct(m['효율선_수익'])}, "
                             f"같은 낙폭 {m['같은낙폭_k']:.2f}배 {pct(m['같은낙폭_수익'])}")
        A(f"- {p}: " + "; ".join(parts))
    A("")
    # ---- 고르기
    A("## 고르기 (2026 앞만 보고 골라 뒤·2025 확인, 합성 252조합)")
    A("")
    A("| 방법 | 고른 것 | 앞 | 뒤 | 2025 | 2015~2026 |")
    A("|---|---|---|---|---|---|")

    def sel_row(name, p, rid, key):
        cells = []
        for w in (PICK, CHECK, AUX):
            if key == "칼마":
                cells.append(f"{rr('synth', p, rid, w)} (칼마 {num(M('synth', p, rid, w, '칼마'))}, 들고만 {num(M('synth', p, '보유', w, '칼마'))})")
            else:
                cells.append(f"{rr('synth', p, rid, w)} (효율선초과 {pct(M('synth', p, rid, w, '효율선초과'))})")
        A(f"| {name} | {p} · {lab[rid]} | " + " | ".join(cells) + f" | {full('synth', p, rid)}, 들고만 {full('synth', p, '보유')} "
          f"(효율선초과 연 {pct(M('synth', p, rid, '전체', '효율선초과_연'))}) |")

    sel_row("G1 효율선초과 최대(대표)", *g1, "효율선초과")
    if g2:
        cells = []
        for w in (PICK, CHECK, AUX):
            nn, ee = pooled_win(g2, w)
            ex = np.mean([M("synth", p, g2, w, "효율선초과") for p in prods])
            cells.append(f"주기 {nn}건 기대 {pct(ee)} · 7상품 효율선초과 평균 {pct(ex)}")
        A(f"| G2 주기 기대값 최대(7상품 합산 {g2_min}건 이상) | 7상품 · {lab[g2]} | " + " | ".join(cells)
          + f" | 7상품 효율선초과 연 평균 {pct(np.mean([M('synth', p, g2, '전체', '효율선초과_연') for p in prods]))} |")
    sel_row("G3 칼마 최대", *g3, "칼마")
    A("")
    A("상품별 G1(그 상품 안에서 앞 효율선초과 최대):")
    A("")
    A("| 상품 | G1 규칙 | 앞 | 뒤 | 2025 | 2022 수익(들고만) | 2015~2026 효율선초과(연) |")
    A("|---|---|---|---|---|---|---|")
    for p in prods:
        rid = per_g1[p]
        A(f"| {p} | {lab[rid]} | " + " | ".join(pct(M("synth", p, rid, w, "효율선초과")) for w in (PICK, CHECK, AUX))
          + f" | {pct(M('synth', p, rid, '2022', '수익'))} ({pct(M('synth', p, '보유', '2022', '수익'))}) | {pct(M('synth', p, rid, '전체', '효율선초과_연'))} |")
    A("")
    A("'앞 > 0' 집합이 뒤·2025에도 > 0인 비율 (괄호 = 고르지 않고 252개 전부의 비율):")
    A("")
    A("| 기준 | 앞 통과 | 뒤 성공 | 2025 성공 | 2015~2026 성공 |")
    A("|---|---|---|---|---|")
    for x in ps + [ev_ps]:
        A(f"| {x['기준']} | {x['앞통과']}/{x['조합수']} | " + " | ".join(f"{pct0(x[w][0], 0)} ({pct0(x[w][1], 0)})" for w in (CHECK, AUX, "전체")) + " |")
    A("")
    A(f"앞·뒤·2025 모두 효율선초과 > 0: {len(all3)}/{len(combos)}조합" + (": " + ", ".join(
        f"{p} {lab[rid]}({' · '.join(pct(M('synth', p, rid, w, '효율선초과')) for w in (PICK, CHECK, AUX))}; 2015~2026 연 {pct(M('synth', p, rid, '전체', '효율선초과_연'))})"
        for p, rid in all3) if all3 else ""))
    A("")
    # ---- 계열
    A("## 계열별: 효율선초과 (합성 252조합, 중앙값 · 이긴 조합 비율)")
    A("")
    A("| 계열 | 조합 | 2026 앞 | 2026 뒤 | 2025 | 2024 | 2023 | 2021 | 2019 | 2015~2026(연율 차) | 같은 기간, 미끄러짐 0.5% | 연 들락(중앙) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in famA:
        A(f"| {FAM_NAME[row['계열']]} | {row['조합']} | " + " | ".join(f"{pct(row[w][0])} · {pct0(row[w][1], 0)}" for w in (PICK, CHECK, AUX, "2024", "2023", "2021", "2019"))
          + f" | {pct(row['전체_연'][0])} · {pct0(row['전체_연'][1], 0)} | {pct(row['전체_연_슬립'][0])} · {pct0(row['전체_연_슬립'][1], 0)} | {num(row['연줄임'], 1)} |")
    A("")
    beat_check = [c for c in combos if pos(M("synth", *c, CHECK, "효율선초과"))]
    A(f"2026 뒤에 효율선을 이긴 조합 {len(beat_check)}개: " + ", ".join(f"{p} {lab[rid]}({pct(M('synth', p, rid, CHECK, '효율선초과'))}, 앞 {pct(M('synth', p, rid, PICK, '효율선초과'))})" for p, rid in beat_check) + ".")
    A("")
    A("### 변동성 필터 사후 관찰 (고르기 규약 밖 — 2015~2026 긴 기간 표에서 눈에 띈 것, 매매 근거 아님)")
    A("")
    A("| 상품 | 규칙 | 앞 | 뒤 | 2026 전체 | 2025 | 2015~2026 연율(들고만) | 효율선초과 연 0% · 0.5% | 평균 노출 | 연 들락 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for p in prods:
        for rid in ("V20", "V25", "V35"):
            m, m5, b0 = R[("synth", 0.0, p, rid)]["m"], R[("synth", SLIP, p, rid)]["m"], R[("synth", 0.0, p, "보유")]["m"]
            A(f"| {p} | {lab[rid]} | " + " | ".join(pct(m[w]["효율선초과"]) for w in (PICK, CHECK, "2026", AUX))
              + f" | {pct(m['전체']['연율'])} ({pct(b0['전체']['연율'])}) | {pct(m['전체']['효율선초과_연'])} · {pct(m5['전체']['효율선초과_연'])} | "
              f"{num(m['전체']['평균노출'])} | {num(m['전체']['연줄임'], 1)} |")
    A("")
    A("- 문턱이 기초마다 같아 SOXX(평소 변동성 30%대) 기초에서는 거의 늘 현금이 된다. QQQ·SPY 기초에서만 긴 기간 효율선을 앞섰고, 2025는 모두 졌다(관세 폭락 뒤 늦게 복귀).")
    A("")
    A("폭락·반등 구간: 하락장에서는 효율선이 '현금(0배)'이 되므로 대신 두 가지를 본다 — 들고만 있기 대비 수익 차(중앙) / 같은 낙폭 배수 대비 초과(중앙) · 이긴 비율.")
    A("")
    A("| 계열 | " + " | ".join(CRASH_NAME[w] for w in CRASH) + " |")
    A("|---|" + "---|" * len(CRASH))
    for rb in famB:
        A(f"| {FAM_NAME[rb['계열']]} | " + " | ".join(f"{pct(rb[w][0])} / {pct(rb[w][1])} · {pct0(rb[w][2], 0)}" for w in CRASH) + " |")
    A("")
    # ---- 승률·손익비·꼬리
    A("## 승률·손익비·꼬리 (합성 2015~2026, 손절 주기 = 판 가격/되산 가격 − 1, +면 싸게 되삼)")
    A("")
    A("| 상품 | 규칙 | 주기 | 승률 | 평균 이익 | 평균 손실 | 손익비 | 기대값 | 최악 1년 | 60일 −30% 이하 확률 | 60일 하위 5% | 최대 낙폭 | 연율 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in ("TQQQ", "SOXL"):
        lst = []
        for rid in ["보유", "K2", "K1.5", "T200", "T200b3", "T50", "T20", "P200", "V25", "V35", "S30_sma50", "S30_wait20", "D30_50", per_g1[p], g1[1]]:
            if rid not in lst:
                lst.append(rid)
        for rid in lst:
            m = R[("synth", 0.0, p, rid)]["m"]["전체"]
            A(f"| {p} | {L_of('synth', p, rid)} | {m['주기']} | {pct0(m['승률'], 0)} | {pct(m['평균이익'])} | {pct(m['평균손실'])} | {num(m['손익비'])} | "
              f"{pct(m['기대값'])} | {pct(m['최악1년'])} | {pct0(m['P(60일≤−30%)'])} | {pct(m['60일하위5%'])} | {pct0(m['낙폭'], 0)} | {pct(m['연율'])} |")
    A("")
    # ---- 맥락
    A("## 맥락: 폭락·반등 구간 (합성, 수익/최대 낙폭) — 날짜는 사후에 정했으므로 고르기에 쓰지 않음")
    A("")
    ctx = CRASH + ["2024", "2023", "2021", "2019"]
    A("| 상품 | 규칙 | " + " | ".join(CRASH_NAME.get(w, w) for w in ctx) + " |")
    A("|---|---|" + "---|" * len(ctx))
    for p in ("TQQQ", "SOXL", "UPRO", "TNA"):
        seen = []
        for rid in ["보유", "K2", "T200", "T200b3", "T50", "V25", "S30_sma50", "D30_50", per_g1[p]]:
            if rid in seen:
                continue
            seen.append(rid)
            A(f"| {p} | {L_of('synth', p, rid)} | " + " | ".join(rr("synth", p, rid, w) for w in ctx) + " |")
    A("")
    A("구간: 2018 4분기 10/1~12/31, 2020 급락 2/20~3/23, 2020 반등 3/24~8/31, 2025 관세 2/20~4/8, 2025 반등 4/9~7/31, 2026 AI 쇼크 6/23~7/29(6/22 종가 기준).")
    A("")
    # ---- 들락·미끄러짐
    A("## 들락거림과 미끄러짐 0.5% (사고팔 때 각각)")
    A("")
    A("| 창 | 변화(중앙) | 변화(하위 10%) | 효율선초과 > 0 비율 0% → 0.5% |")
    A("|---|---|---|---|")
    for s_ in slip_rows:
        unit = "연율" if s_["단위"] == "연율" else "수익"
        A(f"| {s_['창']} ({unit}) | {pct(s_['변화_중앙'], 2)} | {pct(s_['변화_하위10%'], 2)} | {pct0(s_['효율선>0_전'], 0)} → {pct0(s_['효율선>0_후'], 0)} |")
    A("")
    A(f"- 2015~2026: 연 줄임(들락) 1번당 연율 {pct(slope, 2)} (252조합 회귀). 왕복 1번 ≈ 0.5% × 2.")
    A("")
    A("| 상품 | 규칙 | 연 들락 | 연율 0% → 0.5% | 2026 앞 0% → 0.5% | 2026 뒤 0% → 0.5% | 2025 0% → 0.5% |")
    A("|---|---|---|---|---|---|---|")
    for p in ("TQQQ", "SOXL"):
        seen = []
        for rid in ["T20", "T20b3", "T50", "T200", "T200b3", "S15_sma50", "S30_sma50", "V25", "P50", per_g1[p]]:
            if rid in seen:
                continue
            seen.append(rid)
            m0, m1 = R[("synth", 0.0, p, rid)]["m"], R[("synth", SLIP, p, rid)]["m"]
            A(f"| {p} | {lab[rid]} | {num(m0['전체']['연줄임'], 1)} | {pct(m0['전체']['연율'])} → {pct(m1['전체']['연율'])} | "
              + " | ".join(f"{pct(m0[w]['수익'])} → {pct(m1[w]['수익'])}" for w in (PICK, CHECK, AUX)) + " |")
    A("")
    # ---- 재현
    A("## 실제 상품 재현 (같은 36규칙을 실제 7상품에)")
    A("")
    A("| 창 | 규칙 수익 상관(합성↔실제) | 실제 − 합성(중앙) | 들고만 대비 +/− 일치 | 효율선초과 +/− 일치 | 효율선초과 > 0 비율 합성 → 실제 |")
    A("|---|---|---|---|---|---|")
    for w, x in rep.items():
        A(f"| {w}{' (연율)' if w == '전체' else ''} | {num(x['상관'], 3)} | {pct(x['차_중앙'])} | {pct0(x['보유대비_부호일치'], 0)} | {pct0(x['효율선_부호일치'], 0)} | "
          f"{pct0(x['효율선>0_합성'], 0)} → {pct0(x['효율선>0_실제'], 0)} |")
    A("")
    A("| 상품 | 규칙 | 합성 앞 · 뒤 · 2025 | 실제 앞 · 뒤 · 2025 | 합성 2015~2026 | 실제 2015~2026 |")
    A("|---|---|---|---|---|---|")
    for p in ("TQQQ", "SOXL"):
        for rid in core[p]:
            A(f"| {p} | {L_of('synth', p, rid)} | " + " · ".join(pct(M("synth", p, rid, w, "수익")) for w in (PICK, CHECK, AUX)) + " | "
              + " · ".join(pct(M("real", p, rid, w, "수익")) for w in (PICK, CHECK, AUX)) + f" | {full('synth', p, rid)} | {full('real', p, rid)} |")
    A("")
    A("실제 상품 효율선 = 같은 상품을 k/L 비중만 들고 나머지 현금(매일 비중 유지). USD(2배 반도체)는 다른 반도체 지수를 따라가 합성과 일수익 상관 0.95.")
    A("")
    # ---- 부록
    A("## 부록: TQQQ·SOXL 합성 36규칙 전부")
    A("")
    A("| 상품 | 규칙 | 앞 효율선초과 | 뒤 | 2025 | 2022 수익 | 2015~2026 | 효율선초과(연) | 같은 노출 대비(연) | 평균 배수 | 연 들락 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in ("TQQQ", "SOXL"):
        for rid in ["보유", "K2", "K1.5"] + rids:
            m = R[("synth", 0.0, p, rid)]["m"]
            A(f"| {p} | {L_of('synth', p, rid)} | " + " | ".join(pct(m[w]["효율선초과"]) for w in (PICK, CHECK, AUX))
              + f" | {pct(m['2022']['수익'])} | {full('synth', p, rid)} | {pct(m['전체']['효율선초과_연'])} | {pct(m['전체']['같은노출초과_연'])} | "
              f"{num(m['전체']['평균배수'])} | {num(m['전체']['연줄임'], 1)} |")
    A("")
    # ---- 격자·시점·점검·한계
    A("## 격자 (사전 고정, 스크립트 맨 위 상수)")
    A("")
    A(f"- T 기초 추세: SMA {list(T_SMA)} × 띠 {{0%, 3%}} = {len(T_SMA) * len(T_BAND)}. 이탈(종가 < SMA×(1−띠)) → 다음 시가 전량 현금, 회복(종가 > SMA×(1+띠)) → 다음 시가 복귀.")
    A(f"- S 추적 손절(산 뒤 최고 종가 대비, 장중): {{15, 20, 30, 40, 50}}% × 재진입 {{기초 > SMA50, 레버리지 종가 > 직전 20일 최고 고가, 판 뒤 20거래일}} = {len(S_X) * len(S_RE)}. "
      "기초 > SMA50 재진입은 판 날 종가부터 본다(이미 위면 다음 날 되삼 — 들락으로 나타남).")
    A(f"- V 변동성: 기초 20일 실현변동성(연) > {{20, 25, 35}}% → 현금, 문턱 이하 → 복귀 = {len(V_THR)}.")
    A(f"- P 절반: 기초 < SMA{{50, 200}}(띠 0) → 노출 0.5 = {len(P_SMA)}.")
    A(f"- D 깊은 재매수: 추적 {{30, 40}}% 손절 뒤 판 직전 고점 대비 {{50, 60, 70}}% 지정가 재매수, 먼저 기초 > SMA50이 되면 그때 시가 = {len(D_X) * len(D_D)}.")
    A(f"- 규칙 {len(rules)}개 × 상품 7 = **{grid_n}조합**. 실행: 합성 미끄러짐 0%·0.5%, 실제 0% → {grid_n * 3}회 + 들고만 있기·고정 배수 참고.")
    A(f"- 효율선: 고정 배수 k = 0~L, {K_STEP} 간격(3배 61점, 2배 41점), 창마다 따로 잰다. 참고 행 '고정 k배' = 3배 상품 k {{2, 1.5}}, 2배 상품 k {{1.5, 1}}.")
    A(f"- 자료 오류 의심(trade_lab.bad_bars) 상품·기초: {len(excluded)}개 (뺀 수 0).")
    A("- 같은 날짜 경로 하나로 2015~2026을 이어서 돌리고 창은 잘라 쟀다(상태 이어짐). 시작은 2014-12-31 종가에 전량 보유.")
    A("")
    A("## 변수 시점")
    A("")
    A("| 변수 | 언제 아나 | 쓰는 때 |")
    A("|---|---|---|")
    A("| 기초 종가, SMA20~200 | t−1 종가 | t일 시가 체결 |")
    A("| 기초 20일 실현변동성 | t−1 종가(t−20..t−1 일수익) | t일 시가 |")
    A("| 레버리지 산 뒤 최고 종가 | t−1 종가 | t일 장중 손절선(시가가 선 아래면 시가) |")
    A("| 레버리지 직전 20일 최고 고가 | t−2까지 고가, t−1 종가와 비교 | t일 시가 재진입 |")
    A("| D 지정가 = 판 직전 고점 × (1−d) | 판 날(t−1까지 고점) | 이후 장중 저가 ≤ 가격이면 min(시가, 가격) |")
    A("| 합성 금융비용 ^IRX·보수 | 그날 공시 금리 | 합성 종가 계산(실제 상품은 가격에 내장) |")
    A("| 효율선·같은 낙폭·같은 노출 기준 | 그 창 전체(사후) | 평가 기준일 뿐 매매 신호가 아님 |")
    A("| 폭락 구간 날짜 | 사후 | 맥락 표만 |")
    A("")
    A("## 점검")
    A("")
    for m_ in chk:
        A(f"- {m_}")
    A("")
    A("## 선택편향·한계")
    A("")
    A(f"- {grid_n}조합에서 앞 구간 최대를 고르면 운이 섞인다. 7상품은 서로 닮아(QQQ·XLK·SPY 3배, SOXX 2·3배) 독립 표본은 7보다 훨씬 적다.")
    A("- 2026 앞 5개월·뒤 4개월, 상품마다 경로 하나. 추세 규칙은 1년에 몇 번만 들락거려 앞 구간 손절 주기 수가 적다(G2는 7상품 합산).")
    A("- 생존편향: 지금 남은 지수 상품만 본다. 지수 3배는 상장폐지가 드물지만, 망해서 사라진 테마 레버리지는 빠져 있다(손절에 불리한 방향).")
    A("- 현금 이자 0%: 현금에 머문 기간만큼 단기 금리(^IRX)를 받지 않았다. 오래 현금에 머무는 규칙(T·V)에 불리한 가정이며, 효율선의 k<1 쪽도 같은 가정이다.")
    A("- 합성은 실제보다 연 약 2% 더 벌었다(3배 스왑 비용). 효율선 k>1 보수는 lev_lab.synth 기본값(1~2배 미만 1%, 2배 1.05%, 3배 0.95%)이라 k=1 바로 위에 작은 계단이 있다.")
    A("- 하락 창의 효율선은 정의상 0배(현금)가 되기 쉽다. 그래서 폭락 구간은 '들고만 대비'와 '같은 낙폭 배수 대비'를 따로 보였다.")
    A("- 미끄러짐 0.5%(사고팔 때 각각)는 사전에 정한 스트레스 값이다. 대형 레버리지 ETF의 평소 호가 차는 이보다 작을 것으로 보지만 이번에 재지 않았다. 반대로 갭 하락 날의 손절 체결은 시가로만 반영했다.")
    A("- 세금·환전 비용 없음. 장중 손절은 선 가격에 체결된다고 가정(갭 하락은 시가). 합성 상품의 장중 가격은 기초 시가·고가·저가로 만들었다.")
    A("")
    md = "\n".join(lines)

    summary = {"G1": {"product": g1[0], "rule": g1[1], "label": lab[g1[1]]}, "G2": {"rule": g2, "label": lab.get(g2), "min": g2_min,
               "pooled_앞": pooled.get(g2)}, "G3": {"product": g3[0], "rule": g3[1], "label": lab[g3[1]]}, "per_product_G1": per_g1,
               "pass_sets": ps + [ev_ps], "all3": all3, "families_A": famA, "families_B": famB, "replication": rep,
               "slippage": slip_rows, "slip_slope": slope}
    concl = conclusion(summary, M, lab, R)
    md = md.replace("@@CONCLUSION@@", concl)
    payload = {
        "grid": {"rules": rules, "products": PRODUCTS, "n_combos": grid_n, "K_STEP": K_STEP, "slip": SLIP, "windows": WIN,
                 "REF_K": {str(k): v for k, v in REF_K.items()}},
        "checks": chk, "excluded": excluded, "summary": summary, "conclusion": concl,
        "rows": [{k: v for k, v in r.items() if k != "m"} | {"m": {w: {kk: vv for kk, vv in mm.items() if kk != "_g"}
                                                                    for w, mm in r["m"].items()}} for r in rows],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{ID}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (RESULTS / f"{ID}.md").write_text(md, encoding="utf-8")
    print("저장", RESULTS / f"{ID}.md")
    print("G1", g1, lab[g1[1]], {w: round(M("synth", *g1, w, "효율선초과"), 4) for w in (PICK, CHECK, AUX)})
    print("G2", g2, pooled.get(g2), {w: pooled_win(g2, w) for w in (CHECK, AUX)} if g2 else None)
    print("G3", g3, lab[g3[1]])
    print("상품별 G1", per_g1)
    for x in ps + [ev_ps]:
        print("pass", x)
    print("all3", all3)
    for row in famA:
        print("famA", row)
    for row in famB:
        print("famB", row)
    for w, x in rep.items():
        print("rep", w, x)
    for s_ in slip_rows:
        print("slip", s_)
    print("slope", slope)


def conclusion(sm, M, lab, R):
    g1p, g1r = sm["G1"]["product"], sm["G1"]["rule"]
    ef = sm["pass_sets"][0]
    fa = {r["계열"]: r for r in sm["families_A"]}
    fb = {r["계열"]: r for r in sm["families_B"]}
    g = lambda p, rid, w, k, slip=0.0: M("synth", p, rid, w, k, slip)
    rl = lambda p, rid, w, k: M("real", p, rid, w, k)
    n_check = round(ef[CHECK][1] * ef["조합수"])
    sl = {x["창"]: x for x in sm["slippage"]}
    tq_v = R[("synth", 0.0, "TQQQ", "V25")]["m"]
    tq_v5 = R[("synth", SLIP, "TQQQ", "V25")]["m"]
    l1 = (f"1. **2026 기준으로는 기각.** {ef['조합수']}조합 중 2026 앞에서 '같은 낙폭의 더 낮은 고정 배수로 그냥 들고 있기'(효율선)를 이긴 {ef['앞통과']}개 가운데 "
          f"뒤에서도 이긴 것은 {round(ef[CHECK][0] * ef['앞통과'])}개다(고르지 않아도 {n_check}개, {pct0(ef[CHECK][1], 0)}). 대표 G1({g1p}, {lab[g1r]})은 "
          f"앞 {pct(g(g1p, g1r, PICK, '효율선초과'))}p였지만 뒤 {pct(g(g1p, g1r, CHECK, '효율선초과'))}p, 2015~2026 연 {pct(g(g1p, g1r, '전체', '연율'))}"
          f"(들고만 {pct(g(g1p, '보유', '전체', '연율'))}, 효율선보다 연 {pct(g(g1p, g1r, '전체', '효율선초과_연'))}p). "
          f"2026 뒤(반도체발 AI 쇼크)에는 추세·추적·절반·깊은 재매수 계열 {sum(fa[f]['조합'] for f in 'TSPD')}조합이 하나도 효율선을 못 이겼다.")
    l2 = (f"2. **폭락 해에 '망하는 것'은 줄였지만 공짜가 아니었다.** 실제 TQQQ 2022 {pct(rl('TQQQ', '보유', '2022', '수익'))} → QQQ<200일선 규칙 {pct(rl('TQQQ', 'T200', '2022', '수익'))}, "
          f"2020 급락 {pct(rl('TQQQ', '보유', '2020급락', '수익'))} → {pct(rl('TQQQ', 'T200', '2020급락', '수익'))}. "
          f"SOXL 2022 {pct(rl('SOXL', '보유', '2022', '수익'))} → {pct(rl('SOXL', 'T200', '2022', '수익'))}. "
          f"추세 계열은 하락 구간 4곳(2018 4분기·2020 급락·2022·2025 관세)에서 들고만보다 중앙 {pct(min(fb['T'][w][0] for w in DOWN4))}~{pct(max(fb['T'][w][0] for w in DOWN4))}p 덜 잃었지만, "
          f"같은 낙폭이 나오는 낮은 고정 배수와 비교하면 중앙 {pct(min(fb['T'][w][1] for w in DOWN4))}~{pct(max(fb['T'][w][1] for w in DOWN4))}p로 같거나 조금 나빴고 "
          f"반등(2020 3~8월 중앙 {pct(fb['T']['2020반등'][0])}p, 2025 4~7월 {pct(fb['T']['2025반등'][0])}p)을 크게 놓쳤다. "
          f"2015~2026(합성)로 보면 200일선 TQQQ 연 {pct(g('TQQQ', 'T200', '전체', '연율'))}/{pct0(g('TQQQ', 'T200', '전체', '낙폭'), 0)} ≈ 고정 {g('TQQQ', 'T200', '전체', '효율선_k'):.2f}배 "
          f"연 {pct(g('TQQQ', 'T200', '전체', '연율') - g('TQQQ', 'T200', '전체', '효율선초과_연'))}. "
          f"2026 AI 쇼크(6/23~7/29)는 QQQ·SOXX가 200일선 위라 200일선 규칙이 켜지지 않았다(실제 SOXL {pct(rl('SOXL', 'T200', 'AI쇼크', '수익'))}).")
    l3 = (f"3. **들락과 미끄러짐:** 연 들락 중앙 추세 {num(fa['T']['연줄임'], 1)}·추적 {num(fa['S']['연줄임'], 1)}·변동성 {num(fa['V']['연줄임'], 1)}·절반 {num(fa['P']['연줄임'], 1)}·깊은 재매수 {num(fa['D']['연줄임'], 1)}번. "
          f"미끄러짐 0.5%면 들락 1번당 연 {pct(sm['slip_slope'], 1)}p, 2015~2026 효율선을 이긴 비율 {pct0(sl['전체']['효율선>0_전'], 0)} → {pct0(sl['전체']['효율선>0_후'], 0)}. "
          f"긴 기간에 앞선 것은 변동성 필터(QQQ 20일 변동성 > 25% → 현금, TQQQ 효율선보다 연 {pct(tq_v['전체']['효율선초과_연'])}p, 0.5%에서도 {pct(tq_v5['전체']['효율선초과_연'])}p)뿐인데 "
          f"2026 앞에는 한 번도 켜지지 않았고 2025는 {pct(tq_v[AUX]['효율선초과'])}p라 사후 관찰로만 남긴다. 낙폭을 줄이는 확실한 방법은 처음부터 덜 드는 것(3배를 2/3만 = 고정 2배)이고, 손절·추세 필터는 그 값을 넘지 못했다.")
    return "\n".join([l1, l2, l3])


if __name__ == "__main__":
    main()
