"""2단계 갈래 DEEP(2026-10-10): '개별주 최대 낙폭 −50% 가정, −40%쯤 기계적으로 다시 사기'(사용자 아이디어 3).

1부 낙폭 분포(2014~2026, 긴 역사가 이 갈래의 핵심 근거라 쓰되 '맥락'으로 표시)
  국면 = 252일 최고 종가인 날 종가 P에서 시작해, 종가가 P를 넘는 날 끝난다(고점 고정, 신고가→다음 신고가).
  국면 최대 낙폭 = 국면 안 최저 종가/P − 1(종가 기준). 2014-01-02부터 자료가 있는 종목은 252거래일 뒤부터 센다(그 전 고점은 진짜 고점이 아닐 수 있음).
  끝나지 않은 국면(2026-10-09까지 회복 못 함)은 '진행 중'으로 표시하고 지금까지 낙폭을 쓴다(낙폭은 하한, 바닥 유지 비율은 상한).
2부 첫 도달 뒤 전방 수익: 같은 국면에서 종가가 P 대비 −30·−40·−50·−60%에 처음 닿은 날 t(한 번) → t+1 시가 매수 → t+N 종가(N=20·60·120·250).
  추가 −20% = t+1..t+N 종가 최저가 매수가의 0.8 이하. 비교용 '아무 날' = 같은 시대 모든 종목·날의 같은 계산.
3부 규칙(stock_lab 엔진, 시점 규칙 그대로: 지표는 [t-1], 손절가·지정가는 t-1까지 정보, 같은 날 순서 시가→손절→지정가).
  고점 H = 직전 거래일까지 252거래일 최고 종가(close_high(252)[t-1]). 지정가 매수는 직전 종가보다 낮을 때만 건다(아래에 걸어 두는 주문).
  손절한 날 같은 날 매수는 손절 체결가보다 낮은 지정가만 인정한다(물리 순서).
  D1 보유 중 추적 손절(진입 뒤 최고 종가 × (1−s), 장중) → H×(1−d) 지정가 기계 재매수, 못 닿으면 20일 최고 고가 돌파(종가) 다음 시가 재진입.
     재매수 뒤: trail = 같은 추적 손절 다시 켬 / floor = '가정 바닥' H_재매수×(1−d−10%p) 이탈 때만 손절, 종가가 H_재매수의 90% 회복하면 추적 손절로.
     통제(d 없음) = 추적 손절 + 20일 돌파 재진입만(깊은 지정가 없음) — 깊은 재매수가 보태는 것이 있는지 보려고 둔다.
  D2 같은 손절 → H×(1−r1·r2·r3)에 1/3씩 사다리(사다리 중에는 손절 없음), 남은 몫은 20일 돌파에 한꺼번에. 다 채운 뒤 trail/floor(마지막 칸−10%p).
  D3 손절 없음. 창 시작 노출 0.5 → H×(1−d) 지정가에 1.0(남은 현금 전부, '같은 돈 추가 매수'). end = 끝까지 / trim = 종가가 H_추가×0.9 회복하면 0.5로.
  D4 창 시작 현금. H×(1−d) 지정가에 1.0 매수 → 60거래일 뒤 시가 매도, 또는 종가가 H_매수×0.9 회복하면 다음 시가 매도(rec10). 손절 없음.
  창 시작: D1·D2는 엔진대로 전량 보유. D3·D4·고정50은 창 전날 시가에 0.5·0으로 맞춰 두고 창 첫날부터 잰다(엔진은 전량 보유로 시작하므로 하루 앞서 돌림).
  주기(손절 1회): 판 값 E, 다시 산 칸 i의 비중 w_i·가격 px_i → 주기 손익 = Σ w_i·E/px_i − 1(되산 주식 수 / 판 주식 수 − 1, +면 싸게 되삼).
     구간 끝까지 못 채운 몫은 구간 끝 종가로 잰다(2026 앞 구간은 5/29 종가, 미래 정보 없음). D3·D4는 매수 1회 = 판 값/산 값 − 1.
고르기 3종(사전 고정): G1 앞 효율선초과 최대 / G2 앞 주기 기대값 최대(주기 30개 이상) / G3 앞 칼마(수익/|낙폭|) 최대.
대표 후보: D1(깊은 재매수 30개)에서 G1, 앞 효율선초과가 0 이하면 G3.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stock_lab as L  # noqa: E402

# ============================================================================ 사전 격자(뒤 구간·2025를 보기 전에 고정)
STOPS = (0.15, 0.20, 0.25)                          # 추적 손절
D1_DEPTHS = (0.30, 0.35, 0.40, 0.45, 0.50)          # 재매수 깊이(252일 고점 대비)
AFTER = ("trail", "floor")                          # 재매수 뒤 손절 방식
FLOOR_GAP = 0.10                                    # 가정 바닥 = 재매수 깊이 + 10%p
D2_LADDERS = ((0.20, 0.30, 0.40), (0.30, 0.40, 0.50), (0.40, 0.50, 0.60))
D3_DEPTHS = D1_DEPTHS
D3_EXITS = ("end", "trim")
D3_START = 0.5
D4_DEPTHS = D1_DEPTHS
D4_HOLDS = (60, "rec10")
REC = 0.90                                          # 이전 고점의 −10%까지 회복
BRK_N = 20                                          # 돌파 재진입 창
SLIP_SENS = 0.005                                   # 대표 후보 체결 미끄러짐 민감도
MIN_CYC_G2 = 30
# 1·2부
P1_LEVELS = (0.50, 0.60, 0.70)
P2_LEVELS = (0.30, 0.40, 0.50, 0.60)
P2_HORIZONS = (20, 60, 120, 250)
ERAS = (("~2017", "2014-01-01", "2017-12-31"), ("2018~19", "2018-01-01", "2019-12-31"), ("2020", "2020-01-01", "2020-12-31"),
        ("2021~22", "2021-01-01", "2022-12-31"), ("2023~24", "2023-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"),
        ("2026", "2026-01-01", "2026-12-31"))
# 실행 창: 평가(2026 전체·앞·뒤, 2025 보조) + 맥락(2022, 2018 4분기, 2020)
L.WINDOWS["2018Q4"] = ("2018-10-01", "2018-12-31")
RUNS_ALL = {"2026": L.RUNS["2026"], "2025": L.RUNS["2025"], "2022": L.RUNS["2022"], "2018Q4": L.WINDOWS["2018Q4"],
            "2020": L.RUNS["2020"]}


def grid() -> list[dict]:
    g = []
    for s in STOPS:
        for d in D1_DEPTHS:
            for a in AFTER:
                g.append({"id": f"D1_s{int(s*100)}_d{int(d*100)}_{a}", "fam": "D1", "stop": s, "depth": d, "after": a, "start": "hold"})
        g.append({"id": f"D1ctl_s{int(s*100)}", "fam": "D1ctl", "stop": s, "depth": None, "after": "trail", "start": "hold"})
    for s in STOPS:
        for lad in D2_LADDERS:
            for a in AFTER:
                g.append({"id": f"D2_s{int(s*100)}_{'-'.join(str(int(r*100)) for r in lad)}_{a}", "fam": "D2", "stop": s,
                          "rungs": lad, "after": a, "start": "hold"})
    for d in D3_DEPTHS:
        for e in D3_EXITS:
            g.append({"id": f"D3_d{int(d*100)}_{e}", "fam": "D3", "depth": d, "exit": e, "start": "shift"})
    for d in D4_DEPTHS:
        for h in D4_HOLDS:
            g.append({"id": f"D4_d{int(d*100)}_{h}", "fam": "D4", "depth": d, "hold": h, "start": "shift"})
    return g


REFS = [{"id": "BH", "fam": "ref", "start": "hold"}, {"id": "FIX50", "fam": "ref", "start": "shift"}]


def describe(sp: dict) -> str:
    f = sp["fam"]
    if f == "D1":
        aft = "재매수 뒤 같은 추적 손절" if sp["after"] == "trail" else f"재매수 뒤 가정 바닥 −{int((sp['depth']+FLOOR_GAP)*100)}% 이탈만 손절"
        return f"추적 −{int(sp['stop']*100)}% 손절 → 252일 고점 −{int(sp['depth']*100)}% 지정가 재매수(못 닿으면 20일 돌파), {aft}"
    if f == "D1ctl":
        return f"추적 −{int(sp['stop']*100)}% 손절 → 20일 돌파 재진입만(깊은 지정가 없음)"
    if f == "D2":
        lad = "·".join(f"−{int(r*100)}%" for r in sp["rungs"])
        aft = "다 채운 뒤 추적 손절" if sp["after"] == "trail" else f"다 채운 뒤 −{int((sp['rungs'][-1]+FLOOR_GAP)*100)}% 이탈만 손절"
        return f"추적 −{int(sp['stop']*100)}% 손절 → {lad} 1/3씩 사다리(남은 몫 20일 돌파), {aft}"
    if f == "D3":
        ex = "끝까지 보유" if sp["exit"] == "end" else "고점 −10% 회복 시 0.5로"
        return f"노출 0.5로 시작, 252일 고점 −{int(sp['depth']*100)}%에 1.0(물타기), {ex}, 손절 없음"
    if f == "D4":
        h = "60거래일 보유" if sp["hold"] == 60 else "고점 −10% 회복까지 보유"
        return f"현금으로 시작, 252일 고점 −{int(sp['depth']*100)}% 지정가 매수, {h}, 손절 없음"
    return {"BH": "매수·보유", "FIX50": "고정 노출 0.5"}[sp["id"]]


# ============================================================================ 정책
def _H(b, t):
    return b.close_high(252)[t - 1]


def _brk(b, t):
    if t < 2:
        return False
    hn = b.high_n(BRK_N)[t - 2]
    return bool(np.isfinite(hn) and b.c[t - 1] > hn)


class _Logged(L.Policy):
    """엔진 체결 규칙(시가·손절가·min(시가, 지정가), 미끄러짐)을 그대로 따라 주기·매매를 기록한다."""

    def __init__(self, slip):
        self.slip = slip
        self.trades = []
        self._nex = 0
        self._open = None
        self._stopL = None
        self._offer = []
        self.cyc = None

    def _day(self):
        self._open, self._stopL, self._offer = None, None, []

    def _stop_fill(self, b, t):
        return (b.o[t] if b.o[t] <= self._stopL else self._stopL) * (1 - self.slip)

    def _lim_fill(self, b, t, lvl):
        return min(b.o[t], lvl) * (1 + self.slip)

    def _new_cyc(self, b, t, why):
        self.cyc = {"type": "cyc", "sym": b.sym, "start": b.days[t], "E": self._stop_fill(b, t), "legs": [], "why": why}
        self.trades.append(self.cyc)

    def _leg(self, day, px, w, kind):
        if self.cyc is not None:
            self.cyc["legs"].append((day, float(px), float(w), kind))

    def _close_cyc(self, day, px, kind):
        if self.cyc is not None:
            self._leg(day, px, 1.0 - sum(x[2] for x in self.cyc["legs"]), kind)
            self.cyc = None


class D1(_Logged):
    def __init__(self, stop, depth, after, slip=0.0):
        super().__init__(slip)
        self.x, self.d, self.after = stop, depth, after
        self.mode, self.floor, self.Href = "hold", None, None

    def on_open(self, b, t, st):
        self._day()
        if st["pos"] <= 1e-9 and st["days_out"] >= 1 and _brk(b, t):
            self._open = 1.0
            return 1.0
        return None

    def stop(self, b, t, st):
        self._stopL = self.floor if self.mode == "floor" else st["peak"] * (1 - self.x)
        return self._stopL

    def limit_buys(self, b, t, st):
        if self.d is None or st["pos"] > 1e-9:
            return ()
        lvl = (1 - self.d) * _H(b, t)
        if not lvl < b.c[t - 1]:
            return ()
        if st["exit_t"] == t and not lvl < st["exit_px"]:
            return ()
        self._offer = [(lvl, 1.0)]
        return self._offer

    def after_close(self, b, t, st):
        if self._open is not None:
            self._close_cyc(b.days[t], b.o[t] * (1 + self.slip), "brk")
            self.mode = "hold"
        if st["n_exits"] > self._nex:
            self._nex = st["n_exits"]
            self._new_cyc(b, t, self.mode)
            self.mode = "out"
        if self._offer and b.l[t] <= self._offer[0][0]:
            lvl = self._offer[0][0]
            self._close_cyc(b.days[t], self._lim_fill(b, t, lvl), "deep")
            self.Href = lvl / (1 - self.d)
            if self.after == "floor":
                self.mode, self.floor = "floor", self.Href * (1 - self.d - FLOOR_GAP)
            else:
                self.mode = "hold"
        if self.mode == "floor" and b.c[t] >= REC * self.Href:
            self.mode = "hold"


class D2(_Logged):
    def __init__(self, stop, rungs, after, slip=0.0):
        super().__init__(slip)
        self.x, self.rungs, self.after = stop, tuple(rungs), after
        self.mode, self.k, self.filled, self.pk, self.floor, self.Href, self.stop_t = "hold", 0, [False] * 3, None, None, None, None

    def on_open(self, b, t, st):
        self._day()
        if self.mode == "ladder" and t > self.stop_t and st["pos"] < 1 - 1e-6 and _brk(b, t):
            self._open = 1.0
            return 1.0
        return None

    def stop(self, b, t, st):
        if self.mode == "ladder":
            return None
        if self.mode == "floor":
            self._stopL = self.floor
        else:
            self._stopL = (self.pk if self.pk is not None else st["peak"]) * (1 - self.x)
        return self._stopL

    def limit_buys(self, b, t, st):
        stopped_today = st["exit_t"] == t and st["pos"] <= 1e-9
        if not (self.mode == "ladder" or stopped_today) or st["pos"] >= 1 - 1e-6 or self._open is not None:
            return ()
        H, c1 = _H(b, t), b.c[t - 1]
        cap = st["exit_px"] if stopped_today else np.inf
        k = 0 if stopped_today else self.k
        filled = [False] * 3 if stopped_today else self.filled
        offer = []
        for i, r in enumerate(self.rungs):
            if filled[i]:
                continue
            lvl = (1 - r) * H
            if lvl < c1 and lvl < cap:
                offer.append((lvl, i))
        offer.sort(key=lambda x: -x[0])
        self._offer = [(lvl, (k + j + 1) / 3 - 1e-6, i) for j, (lvl, i) in enumerate(offer)]
        return [(lvl, tgt) for lvl, tgt, _ in self._offer]

    def _finish(self, b, t, how):
        self.cyc = None                      # 칸·돌파 다리는 이미 기록됨
        if how == "rungs" and self.after == "floor":
            self.Href = _H(b, t)
            self.mode, self.floor = "floor", self.Href * (1 - self.rungs[-1] - FLOOR_GAP)
        else:
            self.mode = "hold"
        self.pk = b.c[t]

    def after_close(self, b, t, st):
        if self._open is not None:
            self._leg(b.days[t], b.o[t] * (1 + self.slip), 1.0 - self.k / 3, "brk")
            self._finish(b, t, "brk")
        if st["n_exits"] > self._nex:
            self._nex = st["n_exits"]
            self._new_cyc(b, t, self.mode)
            self.mode, self.k, self.filled, self.stop_t = "ladder", 0, [False] * 3, t
        for lvl, _, i in self._offer:
            if b.l[t] <= lvl and not self.filled[i]:
                self._leg(b.days[t], self._lim_fill(b, t, lvl), 1 / 3, "rung")
                self.filled[i] = True
                self.k += 1
        if self.mode == "ladder" and self.k == 3:
            self._finish(b, t, "rungs")
        if self.mode == "floor" and b.c[t] >= REC * self.Href:
            self.mode, self.pk = "hold", b.c[t]
        if self.mode == "hold":
            self.pk = max(self.pk if self.pk is not None else st["peak"], b.c[t])


class D3(_Logged):
    def __init__(self, depth, exit, slip=0.0):
        super().__init__(slip)
        self.d, self.exit = depth, exit
        self.mode, self.Href, self.tr = "base", None, None

    def on_open(self, b, t, st):
        self._day()
        if t == st["start_t"]:
            return D3_START
        if self.mode == "added" and self.exit == "trim" and b.c[t - 1] >= REC * self.Href:
            self._open = D3_START
            return D3_START
        return None

    def limit_buys(self, b, t, st):
        if t == st["start_t"] or self.mode != "base" or self._open is not None:
            return ()
        lvl = (1 - self.d) * _H(b, t)
        if not lvl < b.c[t - 1]:
            return ()
        self._offer = [(lvl, 1.0)]
        return self._offer

    def after_close(self, b, t, st):
        if self._open is not None and self.tr is not None:
            self.tr["exit"] = (b.days[t], float(b.o[t] * (1 - self.slip)))
            self.tr, self.mode = None, "base"
        if self._offer and b.l[t] <= self._offer[0][0]:
            lvl = self._offer[0][0]
            self.tr = {"type": "long", "sym": b.sym, "start": b.days[t], "buy": float(self._lim_fill(b, t, lvl)), "exit": None}
            self.trades.append(self.tr)
            self.mode, self.Href = "added", lvl / (1 - self.d)


class D4(_Logged):
    def __init__(self, depth, hold, slip=0.0):
        super().__init__(slip)
        self.d, self.hold = depth, hold
        self.Href, self.tr, self.buy_t = None, None, None

    def on_open(self, b, t, st):
        self._day()
        if t == st["start_t"]:
            return 0.0
        if st["pos"] > 1e-9 and self.tr is not None:
            if (self.hold == 60 and t - self.buy_t >= 60) or (self.hold == "rec10" and b.c[t - 1] >= REC * self.Href):
                self._open = 0.0
                return 0.0
        return None

    def limit_buys(self, b, t, st):
        if t == st["start_t"] or st["pos"] > 1e-9:
            return ()
        lvl = (1 - self.d) * _H(b, t)
        if not lvl < b.c[t - 1]:
            return ()
        self._offer = [(lvl, 1.0)]
        return self._offer

    def after_close(self, b, t, st):
        if self._open is not None and self.tr is not None:
            self.tr["exit"] = (b.days[t], float(b.o[t] * (1 - self.slip)))
            self.tr = None
        if self._offer and b.l[t] <= self._offer[0][0]:
            lvl = self._offer[0][0]
            self.tr = {"type": "long", "sym": b.sym, "start": b.days[t], "buy": float(self._lim_fill(b, t, lvl)), "exit": None}
            self.trades.append(self.tr)
            self.buy_t, self.Href = t, lvl / (1 - self.d)


class Fix(L.Policy):
    def __init__(self, e):
        self.e = e

    def on_open(self, b, t, st):
        return self.e if t == st["start_t"] else None


def make_policy(sp: dict, slip: float):
    f = sp["fam"]
    if f in ("D1", "D1ctl"):
        return D1(sp["stop"], sp["depth"], sp["after"], slip)
    if f == "D2":
        return D2(sp["stop"], sp["rungs"], sp["after"], slip)
    if f == "D3":
        return D3(sp["depth"], sp["exit"], slip)
    if f == "D4":
        return D4(sp["depth"], sp["hold"], slip)
    return L.Policy() if sp["id"] == "BH" else Fix(0.5)


# ============================================================================ 주기 통계
def _close_at(sym, d1):
    b = L.load()[sym]
    return float(b.c[b.idx(d1, "right") - 1])


def trade_gain(tr, d1):
    if tr["type"] == "cyc":
        E, got, g = tr["E"], 0.0, 0.0
        for day, px, w, _ in tr["legs"]:
            if day <= d1:
                g += w * E / px
                got += w
        if got < 1 - 1e-9:
            g += (1 - got) * E / _close_at(tr["sym"], d1)
        return g - 1, got >= 1 - 1e-9
    ex = tr["exit"]
    px = ex[1] if ex is not None and ex[0] <= d1 else _close_at(tr["sym"], d1)
    return px / tr["buy"] - 1, ex is not None and ex[0] <= d1


def trade_stats(trades, d0, d1) -> dict:
    sel = [tr for tr in trades if d0 <= tr["start"] <= d1]
    if not sel:
        return {"손절주기": 0, "승률": float("nan"), "평균이익": float("nan"), "평균손실": float("nan"), "손익비": float("nan"),
                "기대값": float("nan"), "완료비율": float("nan"), "깊은재매수비율": float("nan")}
    gs = [trade_gain(tr, d1) for tr in sel]
    g = np.array([x[0] for x in gs])
    win, loss = g[g > 0], g[g <= 0]
    legs = [lg for tr in sel if tr["type"] == "cyc" for lg in tr["legs"] if lg[0] <= d1]
    deep_w = sum(lg[2] for lg in legs if lg[3] in ("deep", "rung"))
    all_w = sum(lg[2] for lg in legs)
    return {"손절주기": int(len(g)), "승률": float(len(win) / len(g)), "평균이익": float(win.mean()) if len(win) else float("nan"),
            "평균손실": float(loss.mean()) if len(loss) else float("nan"),
            "손익비": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
            "기대값": float(g.mean()), "완료비율": float(np.mean([x[1] for x in gs])),
            "깊은재매수비율": float(deep_w / all_w) if all_w > 0 else float("nan")}


_CAL = None


def prev_day(d0: str) -> str:
    global _CAL
    if _CAL is None:
        _CAL = L.load()["SPY"].days
    i = int(np.searchsorted(_CAL, d0))
    return str(_CAL[i - 1])


def run_variant(sp: dict, slip: float = 0.0, runs: dict | None = None):
    runs = runs or RUNS_ALL
    out = {}
    for rname, (d0, d1) in runs.items():
        pols = []

        def fac():
            p = make_policy(sp, slip)
            pols.append(p)
            return p

        r0 = prev_day(d0) if sp["start"] == "shift" else d0
        res = L.run_universe(fac, None, {rname: (r0, d1)}, slip)
        ws = L.window_stats(res)
        trades = [tr for p in pols for tr in getattr(p, "trades", [])]
        for w in ws:
            ws[w].update(trade_stats(trades, *L.WINDOWS[w]))
        out.update(ws)
    return sp["id"], out


def _worker(args):
    sp, slip = args
    return run_variant(sp, slip)


# ============================================================================ 1·2부
def _era(day):
    for name, a, z in ERAS:
        if a <= day <= z:
            return name
    return None


def groups() -> dict:
    data, src = L.load(), L.sources()
    ai = [s for s, v in src.items() if "AI_UNIVERSE" in v and s in data]
    dv = {}
    for s in ai:
        b = data[s]
        m = (b.days >= "2026-01-02") & (b.days <= "2026-10-09")
        dv[s] = float(np.nanmean(b.c[m] * b.v[m])) if m.any() else 0.0
    rank = sorted(ai, key=lambda s: -dv[s])
    big = set(rank[: len(rank) // 3])
    time_hist = {s for s, v in src.items() if ("base_hold" in v or "base_2026" in v) and s in data}
    return {"AI대형": big, "AI나머지": set(ai) - big, "TIME보유이력": time_hist}


def p1_episodes() -> list[dict]:
    data = L.load()
    rows = []
    for s, b in data.items():
        if s in L.ETFS:
            continue
        c, n = b.c, len(b.c)
        s0 = 251 if b.days[0] <= "2014-01-10" else 0
        ch = b.close_high(252)
        st = next((t for t in range(s0, n) if c[t] >= ch[t] - 1e-12), None)
        if st is None:
            continue
        M = np.maximum.accumulate(c[st:])
        newhigh = np.where(c[st + 1:] > M[:-1])[0] + st + 1
        starts = np.concatenate([[st], newhigh]).astype(int)
        ends = np.concatenate([newhigh, [n]]).astype(int)
        for p, e in zip(starts, ends):
            seg = c[p:e] / c[p] - 1
            if seg.min() > -0.30:
                continue
            hits = {}
            for X in (0.30, 0.40, 0.50, 0.60, 0.70):
                w = np.where(seg <= -X + 1e-12)[0]
                if len(w):
                    hits[X] = int(p + w[0])
            rows.append({"sym": s, "p": int(p), "peak_day": b.days[p], "mdd": float(seg.min()), "trough": int(p + seg.argmin()),
                         "end": int(e) if e < n else None, "n": n, "hits": hits})
    return rows


def p1_table(rows, keyf) -> dict:
    r = [x for x in rows if 0.40 in x["hits"]]
    out = {}
    for key in sorted({k for x in r for k in keyf(x)}, key=lambda k: [e[0] for e in ERAS].index(k) if k in [e[0] for e in ERAS] else -1):
        g = [x for x in r if key in keyf(x)]
        if not g:
            continue
        mdd = np.array([x["mdd"] for x in g])
        done = np.array([x["end"] is not None for x in g])
        h40 = np.array([x["hits"][0.40] for x in g])
        ends = np.array([x["end"] if x["end"] is not None else 10 ** 9 for x in g])
        nn = np.array([x["n"] for x in g])
        rec_days = (ends - h40)[done]
        d = {"국면": len(g), "종목": len({x["sym"] for x in g}), "진행중": int((~done).sum()),
             "중앙_최대낙폭": float(np.median(mdd)), "바닥50유지": float((mdd > -0.50).mean())}
        for X in P1_LEVELS:
            d[f"P{int(X*100)}"] = float((mdd <= -X + 1e-12).mean())
            d[f"P{int(X*100)}_완료만"] = float((mdd[done] <= -X + 1e-12).mean()) if done.any() else float("nan")
        for H, lab in ((250, "1년"), (500, "2년")):
            ok = h40 + H < nn
            d[f"회복_{lab}"] = float(((ends - h40) <= H)[ok].mean()) if ok.any() else float("nan")
            d[f"회복_{lab}_n"] = int(ok.sum())
        d["회복_전체"] = float(done.mean())
        d["회복_중앙일"] = float(np.median(rec_days)) if len(rec_days) else float("nan")
        d["40뒤_바닥까지_중앙일"] = float(np.median([x["trough"] - x["hits"][0.40] for x in g]))
        out[key] = d
    return out


def p2_events(rows) -> list[dict]:
    data = L.load()
    ev = []
    for x in rows:
        b = data[x["sym"]]
        n = len(b.c)
        for X in P2_LEVELS:
            t = x["hits"].get(X)
            if t is None or t + 1 >= n:
                continue
            e = b.o[t + 1]
            rec = {"sym": x["sym"], "X": X, "day": b.days[t], "era": _era(b.days[t])}
            for N in P2_HORIZONS:
                if t + N < n:
                    rec[f"r{N}"] = float(b.c[t + N] / e - 1)
                    rec[f"m{N}"] = bool(b.c[t + 1:t + N + 1].min() / e - 1 <= -0.20 + 1e-12)
            ev.append(rec)
    return ev


def p2_table(ev, X) -> dict:
    out = {}
    for era in ["전체"] + [e[0] for e in ERAS]:
        g = [x for x in ev if x["X"] == X and (era == "전체" or x["era"] == era)]
        if not g:
            continue
        d = {"사건": len(g), "종목": len({x["sym"] for x in g})}
        for N in P2_HORIZONS:
            r = np.array([x[f"r{N}"] for x in g if f"r{N}" in x])
            m = np.array([x[f"m{N}"] for x in g if f"m{N}" in x])
            d[f"n{N}"] = int(len(r))
            if len(r):
                d[f"평균{N}"] = float(r.mean())
                d[f"중앙{N}"] = float(np.median(r))
                d[f"플러스{N}"] = float((r > 0).mean())
                d[f"추가20_{N}"] = float(m.mean())
        out[era] = d
    return out


def p2_anyday() -> dict:
    """비교용: 같은 시대 모든 종목·날에 t+1 시가 매수 → t+N 종가."""
    data = L.load()
    acc = {(era, N): [] for era in ["전체"] + [e[0] for e in ERAS] for N in (60, 250)}
    for s, b in data.items():
        if s in L.ETFS:
            continue
        n = len(b.c)
        s0 = 251 if b.days[0] <= "2014-01-10" else 0
        eras = np.array([_era(d) for d in b.days])
        for N in (60, 250):
            if n <= s0 + N + 1:
                continue
            t = np.arange(s0, n - N)
            r = b.c[t + N] / b.o[t + 1] - 1
            for era in [e[0] for e in ERAS]:
                m = eras[t] == era
                acc[(era, N)].append(r[m])
            acc[("전체", N)].append(r)
    out = {}
    for (era, N), parts in acc.items():
        r = np.concatenate(parts) if parts else np.array([])
        if len(r):
            out.setdefault(era, {})[f"평균{N}"] = float(r.mean())
            out[era][f"중앙{N}"] = float(np.median(r))
            out[era][f"플러스{N}"] = float((r > 0).mean())
    return out


# ============================================================================ 시점 점검
def check():
    data = L.load()
    syms = L.universe(252, "2026-01-02")
    worst = 0.0
    for sp in grid()[::7]:
        for s in syms[:12]:
            b = data[s]
            a, z = b.idx("2026-01-02"), b.idx("2026-10-09", "right") - 1
            v1, _, _ = L.simulate(b, a, z, make_policy(sp, 0.0))
            T = a + 70
            b2 = L.Bars(b.sym, b.days, b.o.copy(), b.h.copy(), b.l.copy(), b.c.copy(), b.v)
            rng = np.random.default_rng(1)
            for arr in (b2.o, b2.h, b2.l, b2.c):
                arr[T + 1:] *= 1 + rng.normal(0, 0.08, len(arr) - T - 1)
            b2.h[T + 1:] = np.maximum.reduce([b2.h[T + 1:], b2.o[T + 1:], b2.c[T + 1:]])
            b2.l[T + 1:] = np.minimum.reduce([b2.l[T + 1:], b2.o[T + 1:], b2.c[T + 1:]])
            v2, _, _ = L.simulate(b2, a, z, make_policy(sp, 0.0))
            worst = max(worst, float(np.max(np.abs(v1[:T - a + 2] - v2[:T - a + 2]))))
    print("미래 교란 뒤 t까지 가치 최대 차(0이어야 함)", worst)
    # 주기 기록 대조: D1 단일 칸 주기 손익 = 엔진 주기 손익
    sp = {"fam": "D1", "stop": 0.15, "depth": 0.30, "after": "trail", "start": "hold"}
    diffs = []
    for s in syms[:40]:
        b = data[s]
        a, z = b.idx("2026-01-02"), b.idx("2026-10-09", "right") - 1
        p = make_policy(sp, 0.0)
        _, cyc, _ = L.simulate(b, a, z, p)
        mine = [trade_gain(tr, "2026-10-09")[0] for tr in p.trades]
        eng = [cy["gain"] for cy in cyc]
        if len(mine) != len(eng):
            diffs.append(9.9)
        else:
            diffs += [abs(x - y) for x, y in zip(mine, eng)]
    print("D1 주기 손익 기록 vs 엔진 최대 차(0이어야 함)", max(diffs) if diffs else 0.0, "주기", len(diffs))


# ============================================================================ 본체
KEYW = ("2026", "2026앞", "2026뒤", "2025", "2022", "2018Q4", "2020")


def main():
    t0 = time.time()
    G = grid()
    print("격자", len(G), {f: sum(1 for g in G if g["fam"] == f) for f in ("D1", "D1ctl", "D2", "D3", "D4")})
    # ---------------- 1·2부
    rows = p1_episodes()
    grp = groups()
    eraf = lambda x: [_era(L.load()[x["sym"]].days[x["hits"][0.40]])]  # noqa: E731
    grpf = lambda x: ["전체"] + [k for k, v in grp.items() if x["sym"] in v]  # noqa: E731
    t_era = p1_table(rows, eraf)
    t_all = p1_table(rows, lambda x: ["전체"])
    t_grp = p1_table(rows, grpf)
    t_grp_era = {k: p1_table([x for x in rows if x["sym"] in grp[k]], eraf) for k in grp}
    ev = p2_events(rows)
    t2 = {f"{int(X*100)}": p2_table(ev, X) for X in P2_LEVELS}
    anyday = p2_anyday()
    print(f"1부 국면(−30% 이상) {len(rows)}, −40% 도달 {sum(1 for x in rows if 0.40 in x['hits'])}; 2부 사건 {len(ev)}  ({time.time()-t0:.0f}s)")
    # ---------------- 3부
    specs = G + REFS
    res = {}
    with ProcessPoolExecutor(max_workers=10) as ex:
        for sid, out in ex.map(_worker, [(sp, 0.0) for sp in specs]):
            res[sid] = out
    print(f"3부 {len(specs)}개 실행 ({time.time()-t0:.0f}s)")

    def g(sid, w, k):
        return res[sid].get(w, {}).get(k, float("nan"))

    ids = [sp["id"] for sp in G]
    calmar = {i: g(i, "2026앞", "포트_수익") / abs(g(i, "2026앞", "포트_낙폭")) if g(i, "2026앞", "포트_낙폭") < 0 else float("nan") for i in ids}
    G1 = max(ids, key=lambda i: g(i, "2026앞", "효율선초과"))
    elig = [i for i in ids if g(i, "2026앞", "손절주기") >= MIN_CYC_G2 and np.isfinite(g(i, "2026앞", "기대값"))]
    G2 = max(elig, key=lambda i: g(i, "2026앞", "기대값")) if elig else None
    G3 = max([i for i in ids if np.isfinite(calmar[i])], key=lambda i: calmar[i])
    d1 = [sp["id"] for sp in G if sp["fam"] == "D1"]
    rep = max(d1, key=lambda i: g(i, "2026앞", "효율선초과"))
    rep_rule = "D1 G1"
    if not g(rep, "2026앞", "효율선초과") > 0:
        rep = max(d1, key=lambda i: calmar[i] if np.isfinite(calmar[i]) else -9)
        rep_rule = "D1 G3(G1 앞 효율선초과 ≤ 0)"
    fam_g1 = {f: max([sp["id"] for sp in G if sp["fam"] == f], key=lambda i: g(i, "2026앞", "효율선초과"))
              for f in ("D1", "D1ctl", "D2", "D3", "D4")}
    # 고르기 정보: 앞 효율선초과 > 0 집합
    S = [i for i in ids if g(i, "2026앞", "효율선초과") > 0]
    pos = lambda lst, w: float(np.mean([g(i, w, "효율선초과") > 0 for i in lst])) if lst else float("nan")  # noqa: E731
    both = lambda lst: float(np.mean([(g(i, "2026뒤", "효율선초과") > 0) and (g(i, "2025", "효율선초과") > 0) for i in lst])) if lst else float("nan")  # noqa: E731
    passset = {"앞>0_개수": len(S), "전체": len(ids),
               "뒤>0|앞>0": pos(S, "2026뒤"), "뒤>0_무조건": pos(ids, "2026뒤"),
               "2025>0|앞>0": pos(S, "2025"), "2025>0_무조건": pos(ids, "2025"),
               "둘다>0|앞>0": both(S), "둘다>0_무조건": both(ids)}
    # 가족별 집합
    passfam = {}
    for f in ("D1", "D1ctl", "D2", "D3", "D4"):
        fi = [sp["id"] for sp in G if sp["fam"] == f]
        Sf = [i for i in fi if i in S]
        passfam[f] = {"n": len(fi), "앞>0": len(Sf), "뒤>0|앞>0": pos(Sf, "2026뒤"), "뒤>0": pos(fi, "2026뒤"),
                      "2025>0|앞>0": pos(Sf, "2025"), "2025>0": pos(fi, "2025"), "2022>0": pos(fi, "2022"), "2018Q4>0": pos(fi, "2018Q4")}
    # 미끄러짐 민감도
    spmap = {sp["id"]: sp for sp in specs}
    with ProcessPoolExecutor(max_workers=4) as ex:
        sl = dict(ex.map(_worker, [(spmap[i], SLIP_SENS) for i in sorted({rep, G1, G3} | ({G2} if G2 else set()))]))
    print(f"민감도 끝 ({time.time()-t0:.0f}s)")

    payload = {"grid": {"STOPS": STOPS, "D1_DEPTHS": D1_DEPTHS, "AFTER": AFTER, "FLOOR_GAP": FLOOR_GAP, "D2_LADDERS": D2_LADDERS,
                        "D3_DEPTHS": D3_DEPTHS, "D3_EXITS": D3_EXITS, "D4_DEPTHS": D4_DEPTHS, "D4_HOLDS": D4_HOLDS, "REC": REC,
                        "BRK_N": BRK_N, "n": len(G)},
               "p1": {"전체": t_all, "시대": t_era, "묶음": t_grp, "묶음×시대": t_grp_era,
                      "묶음크기": {k: len(v) for k, v in grp.items()}, "AI대형": sorted(grp["AI대형"])},
               "p2": t2, "p2_아무날": anyday,
               "p3": {"specs": {sp["id"]: {**{k: v for k, v in sp.items()}, "rule": describe(sp)} for sp in specs},
                      "results": res, "picks": {"G1": G1, "G2": G2, "G3": G3, "대표": rep, "대표규칙": rep_rule, "가족G1": fam_g1},
                      "calmar_앞": calmar, "passset": passset, "passfam": passfam, "slip": {k: v for k, v in sl.items()}}}
    p = L.save("p2_DEEP", payload)
    print("저장", p, f"({time.time()-t0:.0f}s)")
    # ---------------- 요약 출력
    print("\n[1부] −40% 도달 국면", json.dumps(t_all, ensure_ascii=False))
    for k, v in t_era.items():
        print(" 시대", k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    for k, v in t_grp.items():
        print(" 묶음", k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    print("\n[2부]")
    for X, tb in t2.items():
        for era, d in tb.items():
            print(f" −{X}% {era}", {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in d.items()})
    print(" 아무 날", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in anyday.items()})
    print("\n[3부] 고르기", payload["p3"]["picks"])
    for i in [G1, G2, G3, rep, "BH", "FIX50"] + list(fam_g1.values()):
        if i:
            print(i, "|", L.fmt(res[i], KEYW))
    print("passset", passset)
    print("passfam", json.dumps(passfam, ensure_ascii=False))
    for i, v in sl.items():
        print("slip0.5%", i, "|", L.fmt(v, ("2026", "2026앞", "2026뒤", "2025")))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
    else:
        main()
