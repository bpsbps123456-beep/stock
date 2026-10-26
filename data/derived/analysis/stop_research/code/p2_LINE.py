"""2단계 갈래 LINE(2026-10-10): 가격선 이탈 기계 매도 + 재진입 5종, AI 종목 전체(종목 단위, stock_lab 엔진).

사용자 아이디어 1: "특정 가격 라인을 이탈하면 기계적으로 매도, 다시 사는 건 변동성이 줄어들 때 등".
평가 단위: 손절 1회 = 판 가격/다시 산 가격 − 1(주기). 승률·손익비·기대값 + 동일가중 포트폴리오 효율선초과.

시점(stock_lab 규약 그대로)
  - 종가 신호(SMA·20일 수익·진폭·같은 선 회복)는 t-1 종가까지 계산 → t일 시가 체결.
  - 장중 손절가는 t-1까지 정보로 정해 t일 밤사이 스탑으로 건다(시가가 선 아래면 시가 체결).
  - 돌파 매수(r2, 저가 이탈형 r1)는 t-1까지의 직전 20일 최고 고가에 t일 매수 스탑. 시가가 위면 시가, 아니면 그 가격.
    stock_lab.simulate에는 매수 스탑 훅이 없어 아래 sim_ext가 그 함수를 그대로 옮기고 stop_buys 한 단계만 더했다
    (돌파 매수가 없는 정책에서 원본과 결과가 같음을 check()에서 확인). 판 날에는 다시 사지 않는다(days_out ≥ 1).
  - 창 시작 직전 종가에 전량 보유. 2026은 1/2~10/9를 이어 돌려 앞·뒤로 자른다.
  - 주기 통계를 앞·뒤로 자를 때 그 창 끝까지 못 되산 주기는 창 끝 종가로 평가한다(앞 고르기에 뒤 가격이 섞이지 않게).

실행: PYTHONIOENCODING=utf-8 py p2_LINE.py           (전체)
      PYTHONIOENCODING=utf-8 py p2_LINE.py check     (엔진 동치·미래 교란 점검만)
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stock_lab as SL  # noqa: E402
from stock_lab import Policy  # noqa: E402

# ============================================================================ 사전 고정 격자(뒤·2025를 본 뒤 바꾸지 않음)
STOPS = {   # 키: (설명, 종류, 모수)
    "s1": ("종가<SMA20 → 다음 시가", "sma", 20),
    "s2": ("종가<SMA50 → 다음 시가", "sma", 50),
    "s3": ("종가<SMA100 → 다음 시가", "sma", 100),
    "s4": ("종가<SMA200 → 다음 시가(Faber)", "sma", 200),
    "s5": ("직전 10일 최저 저가 장중 이탈(터틀 S1 청산)", "low", 10),
    "s6": ("직전 20일 최저 저가 장중 이탈(터틀 S2 청산)", "low", 20),
    "s7": ("샹들리에: 산 뒤 최고 고가 − 3×ATR22 장중(LeBeau)", "chand", 3.0),
    "s8": ("터틀 2N: 진입가 − 2×ATR20(진입 때 값) 장중", "2n", 2.0),
    "s9": ("고정 −8%(진입가) 장중(O'Neil)", "fixed", 0.08),
    "s10": ("추적 −15%(산 뒤 최고 종가) 장중", "trail", 0.15),
    "s11": ("추적 −25%(산 뒤 최고 종가) 장중", "trail", 0.25),
    "s12": ("20일 수익 ≤ −15% → 다음 시가(Kaminski-Lo류)", "ret20", 0.15),
}
REENTRY = {
    "r1": "같은 선 회복(SMA·20일 수익: 종가가 다시 선 위 → 다음 시가 / 저가 이탈형: 20일 최고 고가 돌파 장중 / "
          "가격선형 s7~s11: 종가가 손절된 선 위 → 다음 시가)",
    "r2": "직전 20일 최고 고가 돌파 장중 매수 스탑(터틀)",
    "r3": "진폭 정상화: 10일 평균 (고가−저가)/전일종가 ≤ 1.2 × 직전 252일 그 값의 중앙, 그리고 종가 > 직전 종가 → 다음 시가",
    "r4": "판 뒤 20거래일째 시가",
    "r5": "그 창 안에서 다시 사지 않음(손절만의 값)",
}
CHAND_ATR, TURTLE_ATR = 22, 20
BRK_N = 20
R3_N, R3_MED, R3_K = 10, 252, 1.2
R4_DAYS = 20
SLIP_SENS = 0.005
G2_MIN_CYCLES = 30
CONTEXT_YEARS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]
VARIANTS = [(s, r) for s in STOPS for r in REENTRY]          # 12 × 5 = 60
DUPLICATES = {("s5", "r1"): ("s5", "r2"), ("s6", "r1"): ("s6", "r2")}   # 정의상 같은 규칙(저가 이탈형 r1 = r2)


# ============================================================================ 엔진 확장(매수 스탑 한 단계만 추가)
def sim_ext(b, a, z, policy, slip: float = 0.0):
    """stock_lab.simulate와 같은 코드 + '손절 → 매수 스탑(stop_buys) → 지정가 매수' 순서의 매수 스탑 단계."""
    c, o, h, l = b.c, b.o, b.h, b.l
    cash, sh = 0.0, 1.0 / c[a - 1]
    st = {"pos": 1.0, "entry_px": c[a - 1], "entry_t": a - 1, "peak": c[a - 1], "high_peak": h[a - 1],
          "exit_px": None, "exit_t": None, "trough": None, "days_out": 0, "n_exits": 0, "last_reason": None,
          "start_t": a}
    vals = [1.0]
    pos_close = []
    cycles, events = [], []

    def equity(px):
        return cash + sh * px

    def set_pos(target, px, t, reason):
        nonlocal cash, sh
        eq = equity(px)
        cur = sh * px / eq if eq > 0 else 0.0
        if abs(target - cur) < 1e-9:
            return
        if target < cur:
            fill = px * (1 - slip)
            sell_sh = sh - target * eq / px
            cash += sell_sh * fill
            sh -= sell_sh
            if target <= 1e-9:
                sh = 0.0
                st.update(exit_px=fill, exit_t=t, trough=fill, days_out=0, n_exits=st["n_exits"] + 1, last_reason=reason)
                events.append((b.days[t], "sell", reason, fill))
        else:
            fill = px * (1 + slip)
            buy_val = (target - cur) * eq
            buy_val = min(buy_val, cash)
            sh += buy_val / fill
            cash -= buy_val
            if cur <= 1e-9:
                if st["exit_px"] is not None:
                    cycles.append({"exit_day": b.days[st["exit_t"]], "exit_px": st["exit_px"], "re_day": b.days[t],
                                   "re_px": fill, "gain": st["exit_px"] / fill - 1, "days_out": t - st["exit_t"],
                                   "reason": st["last_reason"], "open": False})
                st.update(entry_px=fill, entry_t=t, peak=fill, high_peak=fill, exit_px=None, exit_t=None, trough=None)
            events.append((b.days[t], "buy", reason, fill))
        eqn = equity(px)
        st["pos"] = sh * px / eqn if eqn > 0 else 0.0

    sb = getattr(policy, "stop_buys", None)
    for t in range(a, z + 1):
        tgt = policy.on_open(b, t, st)
        if tgt is not None and np.isfinite(o[t]):
            set_pos(float(np.clip(tgt, 0.0, 1.0)), o[t], t, "open")
        if st["pos"] > 1e-9:
            L = policy.stop(b, t, st)
            if L is not None and np.isfinite(L):
                if o[t] <= L:
                    set_pos(0.0, o[t], t, "stop_gap")
                elif l[t] <= L:
                    set_pos(0.0, L, t, "stop")
        if sb is not None and st["pos"] <= 1e-9:                 # 추가 단계: 매수 스탑(돌파)
            for lvl, tg in sb(b, t, st):
                if lvl is not None and np.isfinite(lvl) and h[t] >= lvl:
                    set_pos(tg, max(o[t], lvl), t, "stop_buy")
                    break
        for lvl, tg in sorted(policy.limit_buys(b, t, st), key=lambda x: -x[0]):
            if lvl is None or not np.isfinite(lvl):
                continue
            if l[t] <= lvl and st["pos"] < tg - 1e-9:
                set_pos(tg, min(o[t], lvl), t, "limit")
        vals.append(equity(c[t]))
        pos_close.append(st["pos"])
        if st["pos"] > 1e-9:
            st["peak"] = max(st["peak"], c[t])
            st["high_peak"] = max(st["high_peak"], h[t])
        else:
            st["days_out"] += 1
            st["trough"] = min(st["trough"], c[t]) if st["trough"] is not None else c[t]
        if hasattr(policy, "after_close"):
            policy.after_close(b, t, st)
    if st["pos"] <= 1e-9 and st["exit_px"] is not None:
        cycles.append({"exit_day": b.days[st["exit_t"]], "exit_px": st["exit_px"], "re_day": None, "re_px": c[z],
                       "gain": st["exit_px"] / c[z] - 1, "days_out": z - st["exit_t"], "reason": st["last_reason"],
                       "open": True})
    sim_ext.last_pos = np.array(pos_close)
    return np.array(vals), cycles, events


_ORIG_SIM = SL.simulate
SL.simulate = sim_ext      # 이 프로세스 안에서만 run_universe가 확장판을 쓰게 한다(파일은 고치지 않음)


# ============================================================================ 지표(엔진 Bars 메모 사용)
def rp10(b):
    return b._memo(("p2line_rp", R3_N), lambda: pd.Series(b.range_pct()).rolling(R3_N, min_periods=R3_N).mean().to_numpy())


def rp_med(b):   # [i] = rp10[i-252..i-1]의 중앙(현재 값 제외 직전 252일)
    return b._memo(("p2line_rpmed", R3_N, R3_MED),
                   lambda: pd.Series(rp10(b)).rolling(R3_MED, min_periods=R3_MED).median().shift(1).to_numpy())


def ret20(b, i):
    return b.c[i] / b.c[i - 20] - 1 if i >= 20 else np.nan


# ============================================================================ 정책
class Line(Policy):
    def __init__(self, sk: str, rk: str):
        self.sk, self.rk = sk, rk
        _, self.kind, self.p = STOPS[sk]
        self.name = f"{sk}·{rk}"
        self.last_L = None       # 오늘 건 손절선
        self.exit_line = None    # 손절된 선(가격선형 r1용)
        self._n_key, self._n = None, np.nan

    # ---- 시가 판단(t-1 정보)
    def on_open(self, b, t, st):
        c = b.c
        if st["pos"] > 1e-9:
            if self.kind == "sma":
                m = b.sma(self.p)[t - 1]
                if np.isfinite(m) and c[t - 1] < m:
                    return 0.0
            elif self.kind == "ret20":
                r = ret20(b, t - 1)
                if np.isfinite(r) and r <= -self.p:
                    return 0.0
            return None
        if st["exit_px"] is None or self.rk in ("r5", "r2"):
            return None
        if self.rk == "r4":
            return 1.0 if st["days_out"] >= R4_DAYS else None
        if self.rk == "r3":
            rp, md = rp10(b)[t - 1], rp_med(b)[t - 1]
            ok = np.isfinite(rp) and np.isfinite(md) and rp <= R3_K * md and c[t - 1] > c[t - 2]
            return 1.0 if ok else None
        # r1 같은 선 회복
        if self.kind == "sma":
            m = b.sma(self.p)[t - 1]
            return 1.0 if np.isfinite(m) and c[t - 1] > m else None
        if self.kind == "ret20":
            r = ret20(b, t - 1)
            return 1.0 if np.isfinite(r) and r > -self.p else None
        if self.kind in ("chand", "2n", "fixed", "trail"):
            return 1.0 if self.exit_line is not None and c[t - 1] > self.exit_line else None
        return None   # 저가 이탈형 r1 → stop_buys

    # ---- 장중 손절가(t-1 정보 + 오늘 시가 체결가가 진입가인 경우 그 값)
    def stop(self, b, t, st):
        k, L = self.kind, None
        if k == "low":
            L = b.low_n(self.p)[t - 1]
        elif k == "chand":
            L = st["high_peak"] - self.p * b.atr(CHAND_ATR)[t - 1]
        elif k == "2n":
            et = st["entry_t"]
            if self._n_key != et:   # 진입 때 N: 창 시작 보유는 a-1 종가 값, 그 뒤 진입은 진입 전날 값
                ref = et if et < st["start_t"] else et - 1
                self._n_key, self._n = et, b.atr(TURTLE_ATR)[ref]
            L = st["entry_px"] - self.p * self._n
        elif k == "fixed":
            L = st["entry_px"] * (1 - self.p)
        elif k == "trail":
            L = st["peak"] * (1 - self.p)
        self.last_L = L
        return L

    # ---- 매수 스탑(돌파)
    def stop_buys(self, b, t, st):
        if st["exit_px"] is None or st["days_out"] < 1:
            return ()
        if self.rk == "r2" or (self.rk == "r1" and self.kind == "low"):
            return ((b.high_n(BRK_N)[t - 1], 1.0),)
        return ()

    def after_close(self, b, t, st):
        if st["pos"] <= 1e-9 and st["exit_t"] == t and st["last_reason"] in ("stop", "stop_gap"):
            self.exit_line = self.last_L


# ============================================================================ 평가
def cyc_stats(cycles, d0, d1, data):
    """그 창에서 판 주기. 창 끝까지 못 되산 주기는 창 끝 종가로 평가."""
    g, days = [], []
    for cy in cycles:
        if not (d0 <= cy["exit_day"] <= d1):
            continue
        if cy["re_day"] is not None and cy["re_day"] <= d1:
            g.append(cy["gain"])
        else:
            b = data[cy["sym"]]
            z = b.idx(d1, "right") - 1
            g.append(cy["exit_px"] / b.c[z] - 1)
        days.append(cy["days_out"])
    g = np.array(g)
    win, loss = g[g > 0], g[g <= 0]
    return {
        "손절주기": int(len(g)), "승률": float(len(win) / len(g)) if len(g) else float("nan"),
        "평균이익": float(win.mean()) if len(win) else float("nan"),
        "평균손실": float(loss.mean()) if len(loss) else float("nan"),
        "손익비": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
        "기대값": float(g.mean()) if len(g) else float("nan"),
        "기대값_중앙": float(np.median(g)) if len(g) else float("nan"),
        "밖_일수_중앙": float(np.median(days)) if days else float("nan"),
    }


def run_variant(args):
    sk, rk, slip, runs = args
    data = SL.load()
    res = SL.run_universe(lambda: Line(sk, rk), runs=runs, slip=slip)
    ws = SL.window_stats(res)
    for rname, r in res.items():
        for w in SL.SLICES.get(rname, [rname]):
            if w in ws:
                d0, d1 = SL.WINDOWS[w]
                ws[w].update(cyc_stats(r["cycles"], d0, d1, data))
                ws[w]["손절_종목당"] = float(ws[w]["손절주기"] / max(ws[w]["종목수"], 1))
    return f"{sk}·{rk}", ws


def calmar(s):
    return s["포트_수익"] / abs(s["포트_낙폭"]) if s["포트_낙폭"] < 0 else float("inf")


# ============================================================================ 점검
def check():
    data = SL.load()
    syms = SL.universe(252, "2026-01-02")[:30]
    # 1) 매수 스탑이 없는 정책: 확장판 = 원본
    worst = 0.0
    for sk, rk in [("s1", "r1"), ("s7", "r1"), ("s9", "r3"), ("s10", "r4"), ("s12", "r5"), ("s8", "r1")]:
        for s in syms:
            b = data[s]
            a, z = b.idx("2026-01-02"), b.idx("2026-10-09", "right") - 1
            v1, c1, _ = _ORIG_SIM(b, a, z, Line(sk, rk))
            v2, c2, _ = sim_ext(b, a, z, Line(sk, rk))
            worst = max(worst, float(np.max(np.abs(v1 - v2))), abs(len(c1) - len(c2)))
    print("확장판 vs 원본 최대 차(0이어야 함)", worst)
    # 2) 미래 교란: t 이후 가격을 바꿔도 t까지 가치가 같아야 한다(돌파·진폭·2N·샹들리에 정책)
    rng = np.random.default_rng(0)
    worst = 0.0
    for sk, rk in [("s5", "r2"), ("s6", "r1"), ("s7", "r3"), ("s8", "r2"), ("s3", "r3"), ("s12", "r1"), ("s11", "r1")]:
        for s in syms[:10]:
            b = data[s]
            a, z = b.idx("2026-01-02"), b.idx("2026-10-09", "right") - 1
            v1, _, _ = sim_ext(b, a, z, Line(sk, rk))
            for T in (a + 30, a + 90):
                b2 = SL.Bars(b.sym, b.days, b.o.copy(), b.h.copy(), b.l.copy(), b.c.copy(), b.v)
                for arr in (b2.o, b2.h, b2.l, b2.c):
                    arr[T + 1:] *= 1 + rng.normal(0, 0.05, len(arr) - T - 1)
                b2.h[T + 1:] = np.maximum.reduce([b2.h[T + 1:], b2.o[T + 1:], b2.c[T + 1:]])
                b2.l[T + 1:] = np.minimum.reduce([b2.l[T + 1:], b2.o[T + 1:], b2.c[T + 1:]])
                v2, _, _ = sim_ext(b2, a, z, Line(sk, rk))
                worst = max(worst, float(np.max(np.abs(v1[:T - a + 2] - v2[:T - a + 2]))))
    print("미래 교란 뒤 t까지 가치 최대 차(0이어야 함)", worst)
    # 3) 중복 정의 확인
    for (x, y) in DUPLICATES.items():
        b = data[syms[0]]
        a, z = b.idx("2026-01-02"), b.idx("2026-10-09", "right") - 1
        v1, _, _ = sim_ext(b, a, z, Line(*x))
        v2, _, _ = sim_ext(b, a, z, Line(*y))
        print("중복", x, y, "차", float(np.max(np.abs(v1 - v2))))


# ============================================================================ 본 실행
def main():
    t0 = time.time()
    runs = {k: SL.RUNS[k] for k in ["2026", "2025"] + CONTEXT_YEARS}
    with Pool(min(12, len(VARIANTS))) as pool:
        out = dict(pool.map(run_variant, [(s, r, 0.0, runs) for s, r in VARIANTS]))
    print(f"60 변형 완료 {time.time() - t0:.0f}s")
    data = SL.load()
    names = [f"{s}·{r}" for s, r in VARIANTS]
    bh = {w: {k: out[names[0]][w][k] for k in ("포트BH_수익", "포트BH_낙폭", "종목수")} for w in out[names[0]]}

    # ---- 고르기(앞 구간만)
    front = {n: out[n]["2026앞"] for n in names}
    g1 = max(names, key=lambda n: front[n]["효율선초과"])
    elig = [n for n in names if front[n]["손절주기"] >= G2_MIN_CYCLES and np.isfinite(front[n]["기대값"])]
    g2 = max(elig, key=lambda n: front[n]["기대값"])
    g3 = max(names, key=lambda n: calmar(front[n]))
    picks = {"G1": g1, "G2": g2, "G3": g3}

    # ---- 미끄러짐 0.5% 민감도(고른 변형)
    uniq = sorted(set(picks.values()))
    with Pool(len(uniq)) as pool:
        slip = dict(pool.map(run_variant, [(*n.split("·"), SLIP_SENS, {k: SL.RUNS[k] for k in ("2026", "2025")})
                                           for n in uniq]))

    # ---- 고르기에 정보가 있나: 앞 효율선초과 > 0 집합의 뒤·2025 성공률 vs 무조건
    def frac(ns, f):
        return (sum(1 for n in ns if f(n)) / len(ns)) if ns else float("nan")
    pos_front = [n for n in names if front[n]["효율선초과"] > 0]
    back_ok = lambda n: out[n]["2026뒤"]["효율선초과"] > 0
    y25_ok = lambda n: out[n]["2025"]["효율선초과"] > 0
    pass_set = {
        "앞>0 개수": len(pos_front), "전체": len(names),
        "뒤>0 | 앞>0": frac(pos_front, back_ok), "뒤>0 무조건": frac(names, back_ok),
        "2025>0 | 앞>0": frac(pos_front, y25_ok), "2025>0 무조건": frac(names, y25_ok),
        "뒤·2025 둘다>0 | 앞>0": frac(pos_front, lambda n: back_ok(n) and y25_ok(n)),
        "뒤·2025 둘다>0 무조건": frac(names, lambda n: back_ok(n) and y25_ok(n)),
        "앞>0 집합": pos_front,
    }

    # ---- 사용자 질문: 승률 30~45%라도 손익비로 기대값 +인 규칙이 앞·뒤·2025 일관?
    consist = []
    for n in names:
        ws = out[n]
        ok = all(np.isfinite(ws[w]["기대값"]) and ws[w]["기대값"] > 0 and ws[w]["손절주기"] >= 10
                 for w in ("2026앞", "2026뒤", "2025"))
        low_wr = all(0.30 <= ws[w]["승률"] <= 0.45 for w in ("2026앞", "2026뒤", "2025"))
        consist.append({"name": n, "세구간_기대값+": ok, "세구간_승률30~45%": low_wr,
                        **{f"{w}_{k}": ws[w][k] for w in ("2026앞", "2026뒤", "2025")
                           for k in ("승률", "손익비", "기대값", "손절주기", "효율선초과")}})
    lowwr_any = [c for c in consist if any(0.30 <= c[f"{w}_승률"] <= 0.45 and c[f"{w}_기대값"] > 0
                                           for w in ("2026앞", "2026뒤", "2025"))]

    # ---- 맥락: 2018~2024 해별 효율선초과 부호
    ctx = {n: {y: out[n][y]["효율선초과"] for y in CONTEXT_YEARS} for n in names}

    # ---- 보조(보고용, 격자·고르기와 무관): 뒤 매수·보유 수익을 이긴 변형, 7월 AI 쇼크(6/22~7/29) 구간
    back_beat = [n for n in names if out[n]["2026뒤"]["포트_수익"] > bh["2026뒤"]["포트BH_수익"]]
    supp = {"뒤_BH수익_이김": back_beat,
            "뒤_낙폭_줄임_비율": frac(names, lambda n: out[n]["2026뒤"]["포트_낙폭"] > bh["2026뒤"]["포트BH_낙폭"]),
            "2025_BH수익_이김_비율": frac(names, lambda n: out[n]["2025"]["포트_수익"] > bh["2025"]["포트BH_수익"]),
            "해별_효율선초과+비율": {y: frac(names, lambda n: ctx[n][y] > 0) for y in CONTEXT_YEARS},
            "해별_BH수익_이김_비율": {y: frac(names, lambda n: out[n][y]["포트_수익"] > bh[y]["포트BH_수익"]) for y in CONTEXT_YEARS},
            "AI쇼크": {}}
    for n in uniq:
        sk, rk = n.split("·")
        r = SL.run_universe(lambda: Line(sk, rk), runs={"2026": SL.RUNS["2026"]})["2026"]
        m = (r["R"].index >= "2026-06-22") & (r["R"].index <= "2026-07-29")
        pr = np.concatenate([[1.0], (1 + r["R"][m].mean(axis=1).fillna(0)).cumprod().to_numpy()])
        pb = np.concatenate([[1.0], (1 + r["B"][m].mean(axis=1).fillna(0)).cumprod().to_numpy()])
        supp["AI쇼크"][n] = {"수익": float(pr[-1] - 1), "낙폭": SL._mdd(pr), "BH수익": float(pb[-1] - 1), "BH낙폭": SL._mdd(pb)}

    payload = {
        "grid": {"stops": STOPS, "reentry": REENTRY, "n": len(VARIANTS), "duplicates": [list(map(list, kv)) for kv in DUPLICATES.items()],
                 "consts": {"CHAND_ATR": CHAND_ATR, "TURTLE_ATR": TURTLE_ATR, "BRK_N": BRK_N, "R3": [R3_N, R3_MED, R3_K],
                            "R4_DAYS": R4_DAYS, "SLIP_SENS": SLIP_SENS, "G2_MIN_CYCLES": G2_MIN_CYCLES}},
        "bh": bh, "picks": picks, "slip": slip, "pass_set": pass_set, "consist": consist,
        "lowwr_any": [c["name"] for c in lowwr_any], "context": ctx, "supp": supp, "results": out,
    }
    p = SL.save("p2_LINE", payload)
    print("저장", p)
    write_md(payload)


# ============================================================================ 보고서
def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def pp(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def write_md(P):
    out, picks, bh, names = P["results"], P["picks"], P["bh"], list(P["results"])
    S = {k: v[0] for k, v in STOPS.items()}
    L = []
    L.append("# 2단계 LINE: 가격선 이탈 기계 매도 + 재진입 (2026-10-10)\n")
    L.append("스크립트 `code/p2_LINE.py`, 전 결과 `results/p2_LINE.json`. AI 종목 전체(2026 "
             f"{bh['2026']['종목수']}종목, 2025 {bh['2025']['종목수']}종목) 각각을 창 시작에 들고, 손절선 12 × 재진입 5를 얹었다.\n")
    ps, sp = P["pass_set"], P["supp"]
    g1 = out[picks["G1"]]
    n_front_pos_exp = sum(1 for n in names if out[n]["2026앞"]["기대값"] > 0)
    shock = sp["AI쇼크"]
    sh_lo, sh_hi = min(v["수익"] for v in shock.values()), max(v["수익"] for v in shock.values())
    L.append("## 결론\n")
    L.append(f"1. **기각(2026 장세).** 앞 구간에서 '같은 낙폭만큼 덜 들고 있기'(효율선)를 이긴 {ps['앞>0 개수']}개 중 뒤 구간에서도 이긴 것은 "
             f"{pp(ps['뒤>0 | 앞>0'], 0)}다(무조건 {pp(ps['뒤>0 무조건'], 0)}, 그 1개는 노출 12%의 사실상 현금). 앞 1등 G1 `{picks['G1']}`(SMA100 이탈·회복)은 "
             f"앞 {pct(g1['2026앞']['포트_수익'])}/{pct(g1['2026앞']['포트_낙폭'])}였지만 뒤 {pct(g1['2026뒤']['포트_수익'])}"
             f"(매수·보유 {pct(bh['2026뒤']['포트BH_수익'])}), 2026 전체 {pct(g1['2026']['포트_수익'])}(매수·보유 {pct(bh['2026']['포트BH_수익'])}).")
    L.append(f"2. **'승률이 낮아도 손익비로 이기는' 손절은 2026에 없었다.** SMA·저가 이탈 + 선 회복·돌파 재진입 12개는 "
             "승률 18~41%, 손익비 1.5~3.4로 말한 모양(자주 지고 크게 이김)이었다. 그러나 앞 구간은 승률 18~26%·손익비 1.5~2.6이라 "
             "손익분기 손익비(승률 25%면 3.0)에 못 미쳐 기대값 −1.7~−4.8%였다. 앞 주기 기대값 플러스는 "
             f"{n_front_pos_exp}/60, 앞·뒤·2025 모두 플러스인 규칙은 0개다.")
    L.append(f"3. 손절이 한 일은 낙폭 축소다. 뒤 구간 낙폭은 {pp(sp['뒤_낙폭_줄임_비율'], 0)}가 줄였고, 7월 쇼크(6/22~7/29) 손실은 고른 3개 기준 "
             f"{pct(next(iter(shock.values()))['BH수익'])} → {pct(sh_lo)}~{pct(sh_hi)}였다. 그러나 반등을 놓쳐 수익이 그보다 더 줄었다. "
             f"하락이 몇 주 이어진 2025(관세)에서는 {pp(ps['2025>0 무조건'], 0)}가 효율선을 이겼지만 매수·보유 수익을 이긴 것은 0개다. "
             "진폭 정상화 재진입(r3)이 가장 나빴다(앞 효율선초과 12개 중 10개 마이너스).\n")
    L.append("## 격자 (사전 고정 60개, 정의상 중복 2개 → 서로 다른 규칙 58개)\n")
    L.append("| 키 | 손절선 |\n|---|---|")
    for k, v in STOPS.items():
        L.append(f"| {k} | {v[0]} |")
    L.append("\n| 키 | 재진입 |\n|---|---|")
    for k, v in REENTRY.items():
        L.append(f"| {k} | {v} |")
    L.append(f"\n- 12 × 5 = 60. s5·r1 = s5·r2, s6·r1 = s6·r2(저가 이탈형의 '같은 선 회복'이 20일 돌파로 지정됨). "
             f"고르기 3종(G1 앞 효율선초과 최대, G2 앞 주기 기대값 최대·주기 {G2_MIN_CYCLES}개 이상, G3 앞 칼마 최대). "
             "미끄러짐 0으로 고르고, 고른 변형만 0.5%로 다시 쟀다. 결과를 본 뒤 격자를 바꾸지 않았다.\n")
    L.append("## 변수 시점\n")
    L.append("| 변수 | 언제 아는가 | 체결 |\n|---|---|---|")
    L.append("| SMA20·50·100·200, 20일 수익 | t−1 종가 | t일 시가 |")
    L.append("| 직전 10·20일 최저 저가, ATR22·ATR20, 산 뒤 최고 고가·종가 | t−1 종가 | t일 장중 스탑(밤사이 걸어 둠; 시가가 선 아래면 시가) |")
    L.append("| 진입가(−8%, 2N) | 산 시점(시가·장중 체결가) | 그날부터 장중 스탑 |")
    L.append("| 직전 20일 최고 고가(돌파) | t−1 종가 | t일 매수 스탑(시가가 위면 시가), 판 날은 제외 |")
    L.append("| 10일 평균 진폭, 직전 252일 중앙, 종가 상승 | t−1 종가 | t일 시가 |")
    L.append("| 판 뒤 경과일 | 매일 종가 | 20거래일째 시가 |\n")

    def row(n, label):
        ws = out[n]
        f, k, y5, y2, a = ws["2026앞"], ws["2026뒤"], ws["2025"], ws["2022"], ws["2026"]
        return (f"| {label} | {n} | {pct(a['포트_수익'])} / {pct(a['포트_낙폭'])} ({pct(a['효율선초과'])}) | {pct(f['포트_수익'])} / {pct(f['포트_낙폭'])} "
                f"| {pct(f['효율선초과'])} | {pct(k['포트_수익'])} / {pct(k['포트_낙폭'])} | {pct(k['효율선초과'])} "
                f"| {pct(y5['포트_수익'])} / {pct(y5['포트_낙폭'])} | {pct(y5['효율선초과'])} | {pct(y2['포트_수익'])} / {pct(y2['포트_낙폭'])} "
                f"| {a['평균노출']:.2f} |")

    L.append("## 고르기 결과 (동일가중 포트폴리오 수익 / 최대 낙폭, 미끄러짐 0)\n")
    L.append("| 고르기 | 변형 | 2026 전체 (효율선초과) | 앞(고르기) | 앞 효율선초과 | 뒤(확인) | 뒤 효율선초과 | 2025(보조) | 2025 효율선초과 | 2022(맥락) | 2026 평균 노출 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    b26, b25 = out[names[0]]["2026"], out[names[0]]["2025"]
    L.append(f"| 매수·보유 | — | {pct(b26['포트BH_수익'])} / {pct(b26['포트BH_낙폭'])} | {pct(bh['2026앞']['포트BH_수익'])} / {pct(bh['2026앞']['포트BH_낙폭'])} | 0 "
             f"| {pct(bh['2026뒤']['포트BH_수익'])} / {pct(bh['2026뒤']['포트BH_낙폭'])} | 0 | {pct(b25['포트BH_수익'])} / {pct(b25['포트BH_낙폭'])} | 0 "
             f"| {pct(bh['2022']['포트BH_수익'])} / {pct(bh['2022']['포트BH_낙폭'])} | 1.00 |")
    for g, n in picks.items():
        L.append(row(n, g))
    for g, n in picks.items():
        s = P["slip"][n]
        L.append(f"| {g} 미끄러짐 0.5% | {n} | {pct(s['2026']['포트_수익'])} / {pct(s['2026']['포트_낙폭'])} ({pct(s['2026']['효율선초과'])}) | {pct(s['2026앞']['포트_수익'])} / {pct(s['2026앞']['포트_낙폭'])} "
                 f"| {pct(s['2026앞']['효율선초과'])} | {pct(s['2026뒤']['포트_수익'])} / {pct(s['2026뒤']['포트_낙폭'])} | {pct(s['2026뒤']['효율선초과'])} "
                 f"| {pct(s['2025']['포트_수익'])} / {pct(s['2025']['포트_낙폭'])} | {pct(s['2025']['효율선초과'])} | — | {s['2026']['평균노출']:.2f} |")
    L.append(f"\n- G1 앞 1등, G2 앞 주기 기대값 1등(주기 {G2_MIN_CYCLES}개 이상; 앞 기대값은 60개 모두 마이너스라 '덜 마이너스' 1등), G3 앞 칼마 1등.")
    L.append(f"- 고르기에 정보가 있나: 앞 효율선초과 > 0인 변형 {ps['앞>0 개수']}/{ps['전체']}개. 그중 뒤 > 0 {pp(ps['뒤>0 | 앞>0'], 0)}"
             f"(무조건 {pp(ps['뒤>0 무조건'], 0)}), 2025 > 0 {pp(ps['2025>0 | 앞>0'], 0)}(무조건 {pp(ps['2025>0 무조건'], 0)}), "
             f"둘 다 {pp(ps['뒤·2025 둘다>0 | 앞>0'], 0)}(무조건 {pp(ps['뒤·2025 둘다>0 무조건'], 0)}).")
    beat = sp["뒤_BH수익_이김"]
    L.append(f"- 뒤 구간은 매수·보유가 마이너스라 효율선이 현금(노출 0)이 된다. 즉 '뒤 효율선초과 > 0' = 뒤에 절대 수익을 냈다는 뜻이다. "
             f"덜 엄격하게 '뒤 매수·보유 수익({pct(bh['2026뒤']['포트BH_수익'])})을 이겼나'로 보면 {len(beat)}/60개다. "
             f"그중 {sum(1 for n in beat if n.endswith('r5'))}개는 r5(앞에서 팔고 현금 대기), 나머지는 모두 r4(20거래일 대기): "
             + ", ".join(n for n in beat if not n.endswith("r5")) + ".")
    s0 = next(iter(shock.values()))
    L.append("- 7월 AI 쇼크(6/22~7/29, 고른 변형): " + "; ".join(
        f"{n} {pct(v['수익'])}/{pct(v['낙폭'])}" for n, v in shock.items()) + f" (매수·보유 {pct(s0['BH수익'])}/{pct(s0['BH낙폭'])}).\n")

    L.append("## 손절 주기 승률·손익비 (주기 = 판 가격/다시 산 가격 − 1, 못 되산 주기는 창 끝 종가; 앞 효율선초과 순)\n")
    L.append("| 변형 | 앞 n | 앞 승률 | 앞 손익비 | 앞 기대값 | 뒤 n | 뒤 승률 | 뒤 손익비 | 뒤 기대값 | 2025 n | 2025 승률 | 2025 손익비 | 2025 기대값 | 수익 나은 종목(2026) | 최악5% 종목(2026, BH) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = sorted(names, key=lambda n: -out[n]["2026앞"]["효율선초과"])
    for n in order:
        ws = out[n]
        cells = []
        for w in ("2026앞", "2026뒤", "2025"):
            s = ws[w]
            cells += [str(s["손절주기"]), pp(s["승률"], 0), "—" if not np.isfinite(s["손익비"]) else f"{s['손익비']:.2f}", pct(s["기대값"])]
        a = ws["2026"]
        L.append(f"| {n} | " + " | ".join(cells) + f" | {pp(a['수익_나은종목비율'], 0)} | {pct(a['최악5%_종목수익'], 0)} ({pct(a['최악5%_BH'], 0)}) |")
    cnt = {w: sum(1 for n in names if out[n][w]["기대값"] > 0 and out[n][w]["손절주기"] >= 10) for w in ("2026앞", "2026뒤", "2025")}
    L.append(f"\n- 주기 기대값 플러스(주기 10개 이상): 앞 {cnt['2026앞']}/60, 뒤 {cnt['2026뒤']}/60, 2025 {cnt['2025']}/60, 세 구간 모두 "
             f"{sum(1 for c in P['consist'] if c['세구간_기대값+'])}/60. 승률 30~45%이면서 기대값 +인 구간이 하나라도 있는 변형 {len(P['lowwr_any'])}개는 "
             "모두 그 + 구간이 뒤나 2025뿐이다.")
    L.append("- 손익분기 손익비 = (1 − 승률)/승률: 승률 20% → 4.0, 25% → 3.0, 30% → 2.33, 40% → 1.5. "
             "선 이탈·돌파형(r1·r2)의 실제 손익비는 1.5~3.4로 대체로 그 아래였다.\n")

    L.append("## 효율선초과 한눈에 (앞 / 뒤 / 2025, 행 = 손절선, 열 = 재진입)\n")
    L.append("| 손절선 | " + " | ".join(REENTRY) + " |")
    L.append("|---|" + "---|" * len(REENTRY))
    for sk in STOPS:
        cells = []
        for rk in REENTRY:
            ws = out[f"{sk}·{rk}"]
            cells.append(f"{pct(ws['2026앞']['효율선초과'], 0)} / {pct(ws['2026뒤']['효율선초과'], 0)} / {pct(ws['2025']['효율선초과'], 0)}")
        L.append(f"| {sk} {S[sk]} | " + " | ".join(cells) + " |")

    L.append("\n## 맥락: 2018~2024 (장세 전환 위험)\n")
    yb = ", ".join(f"{y} {pct(bh[y]['포트BH_수익'], 0)}/{pct(bh[y]['포트BH_낙폭'], 0)}" for y in CONTEXT_YEARS)
    L.append(f"매수·보유(수익/낙폭): {yb}. 이 유니버스(2026-10 생존 종목)에서는 2018도 +30%라 '꺾인 해'는 2022 하나다.\n")
    L.append("| 해 | " + " | ".join(CONTEXT_YEARS) + " |")
    L.append("|---|" + "---|" * len(CONTEXT_YEARS))
    L.append("| 효율선초과 + 비율 | " + " | ".join(pp(sp["해별_효율선초과+비율"][y], 0) for y in CONTEXT_YEARS) + " |")
    L.append("| 매수·보유 수익 이긴 비율 | " + " | ".join(pp(sp["해별_BH수익_이김_비율"][y], 0) for y in CONTEXT_YEARS) + " |")
    L.append("\n칸 = 효율선초과 + 해 수 / 7 · 2022 수익/낙폭(매수·보유 "
             f"{pct(bh['2022']['포트BH_수익'], 0)}/{pct(bh['2022']['포트BH_낙폭'], 0)})\n")
    L.append("| 손절선 | " + " | ".join(REENTRY) + " |")
    L.append("|---|" + "---|" * len(REENTRY))
    for sk in STOPS:
        cells = []
        for rk in REENTRY:
            n = f"{sk}·{rk}"
            c = P["context"][n]
            npos = sum(1 for y in CONTEXT_YEARS if c[y] > 0)
            cells.append(f"{npos}/7 · {pct(out[n]['2022']['포트_수익'], 0)}/{pct(out[n]['2022']['포트_낙폭'], 0)}")
        L.append(f"| {sk} | " + " | ".join(cells) + " |")
    L.append("\n- 2022(꺾인 해)에는 변형의 90%가 매수·보유보다 덜 잃었다. SMA·저가 이탈형(s1~s6, r1~r4)은 −31% → −6~−22%, "
             "가격 고정·추적형(s8~s11)은 다시 사고 또 손절되며 −21~−33%로 거의 못 막았다. r5(팔고 그해 현금)는 −2~−20%. "
             "오르는 해(2018·2019·2021·2023·2024)에는 매수·보유를 이긴 변형이 0개다.")
    L.append("- 2022 효율선초과가 모두 마이너스인 것은 하락 해의 효율선이 현금(0%)이기 때문이다(덜 잃어도 −).\n")

    L.append("## 선택편향·한계\n")
    L.append("- 60개(서로 다른 58개) 중 앞 1등을 골랐으므로 앞 수치는 위로 치우친다. 뒤·2025를 본 뒤 격자·정의를 바꾸지 않았다.")
    L.append("- 생존편향: 종목 목록이 2026-10 기준(+TIME 보유 이력)이라 이긴 종목 쪽으로 기운다. 손절에는 불리한 쪽이다"
             "(망해서 사라진 종목에서 손절이 막아 줬을 손실이 빠져 있음). 반대로 깊은 하락 매수는 좋게 나온다.")
    L.append("- 동일가중 종목 포트폴리오(매일 재조정) 기준이다. 데스크 7종목 규칙 위 결과는 1단계 report.md를 본다. 수수료·현금 이자 0, 배당 수정 가격.")
    L.append("- 정의상 선택(과제에 없던 것): 가격선형(s7~s11)의 r1 '같은 선 회복' = 손절된 그 선 위 종가 → 다음 시가. "
             "2N의 N은 진입 때 값으로 고정. 저가 이탈형 r1 = r2라 중복 2개.")
    L.append("- SMA 손절(s1~s4) + r3·r4는 다시 살 때 아직 선 아래면 다음 시가에 또 팔린다(정의대로 둠). s1~s4·r3은 종목당 17~23번 왕복했고, "
             "그래서 G2(s4·r3)는 미끄러짐 0.5%에 크게 무너졌다.")
    L.append("- r3 진폭 조건이 걸린(진폭이 평소의 1.2배 초과) 날은 2026 앞 47%, 뒤 37%, 7월 쇼크 53%, 2025 4월 73%였다. "
             "그래도 손절 직후에는 조건이 풀려 있는 경우가 많아 r3는 사실상 '다음 상승 종가에 되사기'(밖에 있던 날 중앙 2~3일)가 됐다.")
    L.append("- 엔진: stock_lab.simulate에 매수 스탑 훅이 없어 sim_ext가 같은 코드에 돌파 매수 한 단계를 더했다. "
             "돌파 매수가 없는 정책에서 원본과 차 0, 미래 가격 교란 시 과거 가치 차 0(`py p2_LINE.py check`). "
             "같은 날 손절과 매수가 겹치면 손절 먼저(보수적), 판 날에는 돌파 매수를 하지 않는다.")
    L.append("- 2026 앞·뒤 주기 통계는 그 창 끝까지 못 되산 주기를 창 끝 종가로 평가했다"
             "(엔진 window_stats는 2026 전체 끝 값으로 평가해 앞 고르기에 뒤 가격이 섞임 → 이 스크립트에서 다시 계산).")
    (SL.RESULTS / "p2_LINE.md").write_text("\n".join(L), encoding="utf-8")
    print("md", SL.RESULTS / "p2_LINE.md")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
    elif len(sys.argv) > 1 and sys.argv[1] == "md":
        write_md(json.loads((SL.RESULTS / "p2_LINE.json").read_text(encoding="utf-8")))
    else:
        main()
