"""3단계 갈래 LADDER(2026-10-10): 나눠 팔기 사다리 — 낙폭 사다리(tranche_lab) + 이동평균 사다리(종가 판단 → 다음 시가).

사용자 요청: "기계적 매수처럼 손절도 최대 하락 %를 다양하게 조합해 봐요", "한 번에 올매도가 아니라 특정 선을 이탈하면 일부(25%)를 매도".
2단계 결론(전량 손절은 '같은 낙폭만큼 덜 들고 있기'를 못 넘음)에 대해, 나눠 팔면 달라지는지 본다.

A) 낙폭 사다리(tranche_lab.ladder): 고점(창 시작 직전 종가부터의 추적 최고 종가) 대비 S, S+Δ, … 에서 조각(1/n)씩 장중 손절.
   재매수: 조각마다 (손절 깊이 + G) 지정가 또는 없음 × 되돌림(회복: 종가 ≥ 판 값 × 1.03 → 다음 시가 / 20거래일 뒤 시가).
   공통 깊은 재매수: 모든 조각 고점 대비 −40% 또는 −50% 지정가 + 회복 되돌림.
B) 이동평균 사다리: 종가 < SMA 선 하나를 깰 때마다 u(25% 또는 50%)씩 판다(깬 선 수 k → 보유 비율 1 − u·k, 다음 시가),
   선 위로 다시 올라오면(띠 변형은 선 × 1.02 위) 그 몫을 다음 시가에 되산다. stock_lab 정책(on_open 비율 목표)과 같은 매매를
   sim_ma가 그대로 계산하고(check()에서 값 차 0 확인), 노출은 종가 기준으로, 선별 주기(판 가격/되산 가격)는 직접 기록한다.
C) 비교 기준(고르기 대상 아님): 같은 선에서 25%·50%·100%만 팔기 — 낙폭선(−10%, −20%) 2개 + SMA(50, 100) 2개.

평가(2단계와 같은 규약): 창 2026(1/2~10/9)·앞(1/2~5/29, 고르기)·뒤(6/1~10/9, 확인), 2025(보조), 2018~2024(맥락).
  동일가중 종목 포트폴리오(매일 재조정) 수익·최대 낙폭, 효율선초과(같은 최대 낙폭 고정 노출 대비), 타이밍값(같은 평균 노출
  고정 대비), 평균 노출, 손절 주기 승률·손익비·기대값(조각 수 기준과 비중 가중), 깊은 재매수 비율, 수익 나은 종목 비율.
  주기를 앞·뒤로 자를 때 그 창 끝까지 못 되산 주기는 그 창 끝 종가로 평가한다(앞 고르기에 뒤 가격이 섞이지 않게).
  2018·2019는 QUBT(장외 시절 가격 오류)를 유니버스에서 뺀다.

실행: PYTHONIOENCODING=utf-8 py p3_LADDER.py          (전체)
      PYTHONIOENCODING=utf-8 py p3_LADDER.py check    (sim_ma = stock_lab 정책 동치, 미래 교란 점검)
      PYTHONIOENCODING=utf-8 py p3_LADDER.py md       (json → md 다시 쓰기)
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
import stock_lab as S  # noqa: E402
import tranche_lab as T  # noqa: E402

# ============================================================================ 사전 고정 격자(뒤·2025를 본 뒤 바꾸지 않음)
A_S = (0.05, 0.10, 0.15, 0.20)          # 첫 조각 손절 깊이(고점 대비)
A_DELTA = (0.05, 0.10)                  # 조각 사이 간격(%p)
A_N = (2, 4)                            # 조각 수(2 = 50%씩, 4 = 25%씩)
A_GAP = (0.10, 0.20, None)              # 조각별 깊은 재매수 = 손절 깊이 + G, None = 없음
A_FALLBACK = ("recover", "time")        # 되돌림 재매수
A_COMMON_D = (0.40, 0.50)               # 공통 깊은 재매수(모든 조각), 되돌림은 recover. D ≤ 손절 깊이인 조각은 깊은 재매수 없음
RECOVER_B = 0.03                        # recover: 종가 ≥ 판 가격 × 1.03 → 다음 시가
TIME_N = 20                             # time: 판 날 + 20거래일 시가
B_SETS = {                              # 키: (선 묶음, 판매 단위, 되사기 띠)
    "B1": ((20, 50, 100, 200), 0.25, 0.00),
    "B2": ((10, 20, 50, 100), 0.25, 0.00),
    "B3": ((50, 100, 150, 200), 0.25, 0.00),
    "B4": ((50, 200), 0.50, 0.00),
    "B5": ((20, 100), 0.50, 0.00),
    "B6": ((20, 50, 100, 200), 0.25, 0.02),
    "B7": ((10, 20, 50, 100), 0.25, 0.02),
    "B8": ((50, 100, 150, 200), 0.25, 0.02),
}
C_DD_X = (0.10, 0.20)                   # 비교: 낙폭선 하나, 재매수 = X + 10%p 지정가 + recover
C_DD_GAP = 0.10
C_SMA = (50, 100)                       # 비교: SMA 선 하나, 선 회복 시 되사기
C_Q = (0.25, 0.50, 1.00)                # 그 선에서 파는 몫
SHOCK = ("2026-06-22", "2026-07-29")    # 7월 AI 쇼크(2단계와 같은 측정 구간, 일수익 날짜 기준)
SLIP_SENS = 0.005
G2_MIN_CYCLES = 30
CONTEXT_YEARS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]
DATA_EXCLUDE = {"2018": {"QUBT"}, "2019": {"QUBT"}}
RUN_KEYS = ["2026", "2025"] + CONTEXT_YEARS
MAIN_W = ("2026앞", "2026뒤", "2025")

REBUY_COLS = ["g10·recover", "g20·recover", "g-·recover", "g10·time", "g20·time", "g-·time", "D40·recover", "D50·recover"]
REBUY_LABEL = {
    "g10·recover": "+10%p 지정가 / 회복", "g20·recover": "+20%p 지정가 / 회복", "g-·recover": "지정가 없음 / 회복",
    "g10·time": "+10%p 지정가 / 20일", "g20·time": "+20%p 지정가 / 20일", "g-·time": "지정가 없음 / 20일",
    "D40·recover": "모두 −40% 지정가 / 회복", "D50·recover": "모두 −50% 지정가 / 회복",
}


def _p(x):
    return f"{x * 100:.0f}"


def shape_key(s_, dl, n):
    return f"S{_p(s_)}·Δ{_p(dl)}·n{n}"


def shape_label(s_, dl, n):
    xs = [s_ + i * dl for i in range(n)]
    return "−" + "·−".join(_p(x) for x in xs) + f"% 각 {100 // n}%"


def build_variants():
    V = {}
    for s_ in A_S:
        for dl in A_DELTA:
            for n in A_N:
                xs = [round(s_ + i * dl, 4) for i in range(n)]
                sk = shape_key(s_, dl, n)
                for fb in A_FALLBACK:
                    for g in A_GAP:
                        ds = [round(x + g, 4) for x in xs] if g is not None else None
                        col = f"{'g' + _p(g) if g is not None else 'g-'}·{fb}"
                        V[f"A|{sk}|{col}"] = {"fam": "A", "xs": xs, "ds": ds, "fb": fb, "shape": sk, "col": col,
                                              "s": s_, "delta": dl, "n": n,
                                              "rule": f"고점 대비 {shape_label(s_, dl, n)} 장중 손절, 재매수 {REBUY_LABEL[col]}"}
                for D in A_COMMON_D:
                    ds = [D if D > x + 1e-9 else None for x in xs]
                    col = f"D{_p(D)}·recover"
                    V[f"A|{sk}|{col}"] = {"fam": "A", "xs": xs, "ds": ds, "fb": "recover", "shape": sk, "col": col,
                                          "s": s_, "delta": dl, "n": n,
                                          "rule": f"고점 대비 {shape_label(s_, dl, n)} 장중 손절, 재매수 {REBUY_LABEL[col]}"}
    for k, (lines, u, band) in B_SETS.items():
        lab = f"종가 < SMA{'·'.join(map(str, lines))} 하나 깰 때마다 {int(u * 100)}% 매도(다음 시가), " \
              + (f"선 × {1 + band:.2f} 위 종가면 그 몫 되사기" if band else "선 위 종가면 그 몫 되사기")
        V[f"B|{k}"] = {"fam": "B", "lines": lines, "u": u, "band": band, "rule": lab}
    for x in C_DD_X:
        for q in C_Q:
            V[f"C|dd{_p(x)}|q{_p(q)}"] = {"fam": "C", "kind": "dd", "x": x, "q": q,
                                          "rule": f"고점 대비 −{_p(x)}%에서 {_p(q)}%만 장중 손절, −{_p(x + C_DD_GAP)}% 지정가 / 회복 재매수"}
    for ln in C_SMA:
        for q in C_Q:
            V[f"C|sma{ln}|q{_p(q)}"] = {"fam": "C", "kind": "sma", "lines": (ln,), "u": q, "band": 0.0,
                                        "rule": f"종가 < SMA{ln}이면 {_p(q)}% 매도(다음 시가), 선 위 종가면 되사기"}
    return V


VARIANTS = build_variants()
POOL = [k for k, v in VARIANTS.items() if v["fam"] in ("A", "B")]      # 고르기 대상(A·B)


# ============================================================================ 이동평균 사다리 엔진(stock_lab on_open 비율 목표와 같은 매매)
def ma_next_state(b, t, lines, band, held):
    """t일 시가 판단(t-1 종가·SMA[t-1]만 씀). 반환: 새 보유 상태 목록."""
    c = b.c
    new = list(held)
    for i, n in enumerate(lines):
        sm = b.sma(n)[t - 1]
        if not np.isfinite(sm):
            continue
        if held[i] and c[t - 1] < sm:
            new[i] = False
        elif (not held[i]) and c[t - 1] > sm * (1 + band):
            new[i] = True
    return new


def sim_ma(b, a, z, lines, u, band, slip=0.0):
    c, o = b.c, b.o
    m = len(lines)
    held = [True] * m
    sh, cash = 1.0 / c[a - 1], 0.0
    vals, expo, cycles = [1.0], [], []
    sold = [None] * m
    for t in range(a, z + 1):
        new = ma_next_state(b, t, lines, band, held)
        if new != held and np.isfinite(o[t]):
            k = m - sum(new)
            target = float(np.clip(1.0 - u * k, 0.0, 1.0))
            px = o[t]
            eq = cash + sh * px
            cur = sh * px / eq if eq > 0 else 0.0
            if abs(target - cur) >= 1e-9:
                if target < cur:
                    fill = px * (1 - slip)
                    sell_sh = sh - target * eq / px
                    cash += sell_sh * fill
                    sh -= sell_sh
                    if target <= 1e-9:
                        sh = 0.0
                else:
                    fill = px * (1 + slip)
                    buy_val = min((target - cur) * eq, cash)
                    sh += buy_val / fill
                    cash -= buy_val
            for i in range(m):
                if held[i] and not new[i]:
                    sold[i] = (t, px * (1 - slip))
                elif (not held[i]) and new[i]:
                    st_, sp = sold[i]
                    rp = px * (1 + slip)
                    cycles.append({"exit_day": b.days[st_], "exit_px": sp, "re_day": b.days[t], "re_px": rp,
                                   "gain": sp / rp - 1, "w": u, "days_out": t - st_, "kind": "선 회복", "open": False,
                                   "x": lines[i], "d": None})
                    sold[i] = None
            held = new
        eq = cash + sh * c[t]
        vals.append(eq)
        expo.append(sh * c[t] / eq if eq > 0 else 0.0)
    for i in range(m):
        if sold[i] is not None:
            st_, sp = sold[i]
            cycles.append({"exit_day": b.days[st_], "exit_px": sp, "re_day": None, "re_px": c[z], "gain": sp / c[z] - 1,
                           "w": u, "days_out": z - st_, "kind": "열림", "open": True, "x": lines[i], "d": None})
    return np.array(vals), np.array(expo), cycles


class MAPolicy(S.Policy):
    """같은 규칙의 stock_lab 정책(점검용): 선 상태가 바뀐 날에만 목표 비율 1 − u·k를 낸다."""

    def __init__(self, lines, u, band):
        self.lines, self.u, self.band = lines, u, band
        self.held = [True] * len(lines)

    def on_open(self, b, t, st):
        new = ma_next_state(b, t, self.lines, self.band, self.held)
        if new != self.held:
            self.held = new
            return 1.0 - self.u * (len(self.lines) - sum(new))
        return None


# ============================================================================ 낙폭 사다리: 헛 왕복 상계판(tranche_lab.simulate와 같은 상태 기계)
def sim_tr_net(bars, a, z, spec):
    """tranche_lab.simulate를 그대로 옮기고 한 가지만 다르게 한다.
    원본: 되돌림(회복·20일)으로 시가에 되산 조각이 그 시가가 이미 손절선 아래면 같은 시가에 바로 다시 판다(헛 왕복).
          미끄러짐 0이면 손익 0이지만, 미끄러짐을 넣으면 왕복 비용이 붙고 주기가 둘로 쪼개진다.
    상계판: 헛 왕복은 매매하지 않은 것으로 본다(현금 그대로, 비용 없음). 상태(기준 가격·판 날·무장)는 원본과 똑같이 옮긴다.
          주기는 '처음 판 가격 → 실제로 되산 가격' 하나로 묶고 헛 왕복 횟수를 phantom에 센다.
    미끄러짐 0이면 가치가 원본과 같다(check에서 확인)."""
    c, o, l = bars.c, bars.o, bars.l
    tr = spec.tranches
    k = len(tr)
    sh = np.array([t_.w / c[a - 1] for t_ in tr])
    cash = np.zeros(k)
    held = np.ones(k, bool)
    armed = np.array([t_.x is not None for t_ in tr])
    sold_px = np.full(k, np.nan)          # 원본과 같은 기준 가격(회복 조건·깊은 재매수 비교용)
    sold_t = np.full(k, -1)
    orig_px = np.full(k, np.nan)          # 경제적 주기: 처음 판 가격·날
    orig_t = np.full(k, -1)
    phantom = np.zeros(k, int)
    peak = c[a - 1]
    vals, expo, cycles = [1.0], [], []
    sl = spec.slip

    def close_cycle(i, t, px, kind):
        cycles.append({"exit_day": bars.days[orig_t[i]], "exit_px": orig_px[i], "re_day": bars.days[t], "re_px": px,
                       "gain": orig_px[i] / px - 1, "w": tr[i].w, "days_out": t - orig_t[i], "kind": kind, "open": False,
                       "x": tr[i].x, "d": tr[i].d, "phantom": int(phantom[i])})

    for t in range(a, z + 1):
        for i in range(k):
            if held[i] or spec.fallback == "none":
                continue
            go = (spec.fallback == "recover" and c[t - 1] >= sold_px[i] * (1 + spec.b)) or \
                 (spec.fallback == "time" and t - sold_t[i] >= spec.n)
            if go and t - 1 >= sold_t[i]:
                L = peak * (1 - tr[i].x)
                if o[t] <= L:                                   # 헛 왕복: 원본은 o[t]에 사고 o[t]에 판다 → 상계
                    sold_px[i], sold_t[i], armed[i] = o[t] * (1 - sl), t, True
                    phantom[i] += 1
                    continue
                px = o[t] * (1 + sl)
                close_cycle(i, t, px, "되돌림")
                sh[i], cash[i], held[i], armed[i] = cash[i] / px, 0.0, True, True
        for i in range(k):
            if not (held[i] and armed[i]) or tr[i].x is None:
                continue
            L = peak * (1 - tr[i].x)
            if o[t] <= L:
                px = o[t]
            elif l[t] <= L:
                px = L
            else:
                continue
            px *= (1 - sl)
            cash[i], sh[i], held[i] = sh[i] * px, 0.0, False
            sold_px[i], sold_t[i] = px, t
            orig_px[i], orig_t[i], phantom[i] = px, t, 0
        for i in range(k):
            if held[i] or tr[i].d is None:
                continue
            R = peak * (1 - tr[i].d)
            if R < sold_px[i] and l[t] <= R:
                px = min(o[t], R) * (1 + sl)
                close_cycle(i, t, px, "깊은 재매수")
                sh[i], cash[i], held[i], armed[i] = cash[i] / px, 0.0, True, False
        eq = float(np.sum(sh * c[t]) + np.sum(cash))
        vals.append(eq)
        expo.append(float(np.sum(sh * c[t])) / eq if eq > 0 else 0.0)
        peak = max(peak, c[t])
        for i in range(k):
            if held[i] and not armed[i] and tr[i].x is not None and c[t] >= peak * (1 - tr[i].x):
                armed[i] = True
    for i in range(k):
        if not held[i]:
            cycles.append({"exit_day": bars.days[orig_t[i]], "exit_px": orig_px[i], "re_day": None, "re_px": c[z],
                           "gain": orig_px[i] / c[z] - 1, "w": tr[i].w, "days_out": z - orig_t[i], "kind": "열림",
                           "open": True, "x": tr[i].x, "d": tr[i].d, "phantom": int(phantom[i])})
    return np.array(vals), np.array(expo), cycles


# ============================================================================ 한 변형 돌리기
def make_spec(v, slip):
    if v["fam"] == "A":
        return T.Spec("A", T.ladder(v["xs"], v["ds"]), fallback=v["fb"], b=RECOVER_B, n=TIME_N, slip=slip)
    if v["fam"] == "C" and v["kind"] == "dd":
        return T.Spec("C", T.partial(v["q"], v["x"], v["x"] + C_DD_GAP), fallback="recover", b=RECOVER_B, n=TIME_N, slip=slip)
    return None


def sim_one(v, b, a, z, slip, engine="net"):
    """engine='net': 낙폭 사다리는 헛 왕복 상계판(주 결과), 'raw': tranche_lab.simulate 원본(미끄러짐 민감도 비교용)."""
    spec = make_spec(v, slip)
    if spec is not None:
        if engine == "raw":
            vals, expo, cycles = T.simulate(b, a, z, spec)
            for cy in cycles:
                cy["phantom"] = 0
            return vals, expo, cycles
        return sim_tr_net(b, a, z, spec)
    vals, expo, cycles = sim_ma(b, a, z, v["lines"], v["u"], v["band"], slip)
    for cy in cycles:
        cy["phantom"] = 0
    return vals, expo, cycles


def cut_cycles(cycles, d0, d1, data):
    rows = []
    for cy in cycles:
        if not (d0 <= cy["exit_day"] <= d1):
            continue
        closed = cy["re_day"] is not None and cy["re_day"] <= d1
        if closed:
            g = cy["gain"]
        else:
            bb = data[cy["sym"]]
            z = bb.idx(d1, "right") - 1
            g = cy["exit_px"] / bb.c[z] - 1
        rows.append((g, cy["w"], cy["kind"] if closed else "열림", closed, cy["days_out"] if closed else np.nan,
                     cy.get("phantom", 0)))
    return rows


def cyc_stats(rows, n_stocks):
    if not rows:
        return {"손절주기": 0, "주기_종목당": 0.0, **{k: float("nan") for k in (
            "승률", "평균이익", "평균손실", "손익비", "기대값", "기대값_중앙", "승률_가중", "손익비_가중", "기대값_가중",
            "포지션기여_주기당", "포지션기여_종목당", "깊은재매수비율", "헛왕복_주기비율", "헛왕복_주기당", "밖_일수_중앙")}}
    g = np.array([r[0] for r in rows])
    w = np.array([r[1] for r in rows])
    closed = [r for r in rows if r[3]]
    win = g > 0
    aw = g[win].mean() if win.any() else np.nan
    al = g[~win].mean() if (~win).any() else np.nan
    wa = (g[win] * w[win]).sum() / w[win].sum() if win.any() else np.nan
    wl = (g[~win] * w[~win]).sum() / w[~win].sum() if (~win).any() else np.nan
    return {
        "손절주기": int(len(g)), "주기_종목당": float(len(g) / max(n_stocks, 1)),
        "승률": float(win.mean()), "평균이익": float(aw), "평균손실": float(al),
        "손익비": float(aw / -al) if np.isfinite(aw) and np.isfinite(al) and al < 0 else float("nan"),
        "기대값": float(g.mean()), "기대값_중앙": float(np.median(g)),
        "승률_가중": float(w[win].sum() / w.sum()),
        "손익비_가중": float(wa / -wl) if np.isfinite(wa) and np.isfinite(wl) and wl < 0 else float("nan"),
        "기대값_가중": float((g * w).sum() / w.sum()),
        "포지션기여_주기당": float((g * w).mean()), "포지션기여_종목당": float((g * w).sum() / max(n_stocks, 1)),
        "깊은재매수비율": float(np.mean([r[2] == "깊은 재매수" for r in closed])) if closed else float("nan"),
        "헛왕복_주기비율": float(np.mean([r[5] > 0 for r in rows])),
        "헛왕복_주기당": float(np.mean([r[5] for r in rows])),
        "밖_일수_중앙": float(np.nanmedian([r[4] for r in closed])) if closed else float("nan"),
    }


def port_stats(Rw, Bw, Ew):
    rp = Rw.mean(axis=1).fillna(0).to_numpy()
    bp = Bw.mean(axis=1).fillna(0).to_numpy()
    pr = np.concatenate([[1.0], np.cumprod(1 + rp)])
    pb = np.concatenate([[1.0], np.cumprod(1 + bp)])
    e_avg = float(Ew.mean(axis=1).mean())
    pf = np.concatenate([[1.0], np.cumprod(1 + e_avg * bp)])
    exc, e_eq = S.same_mdd_excess(pr, bp)
    sr = (1 + Rw.fillna(0)).prod() - 1
    sb = (1 + Bw.fillna(0)).prod() - 1
    return {
        "종목수": int(Rw.shape[1]), "포트_수익": float(pr[-1] - 1), "포트_낙폭": S._mdd(pr),
        "포트BH_수익": float(pb[-1] - 1), "포트BH_낙폭": S._mdd(pb),
        "효율선초과": exc, "효율선_e": e_eq, "평균노출": e_avg,
        "고정노출_수익": float(pf[-1] - 1), "고정노출_낙폭": S._mdd(pf), "타이밍값": float(pr[-1] - pf[-1]),
        "수익_나은종목비율": float((sr > sb + 1e-12).mean()),
    }, bp


def run_variant(args):
    key, slip, run_keys = args[:3]
    engine = args[3] if len(args) > 3 else "net"
    v = VARIANTS[key]
    data = S.load()
    out = {}
    for rname in run_keys:
        d0, d1 = S.RUNS[rname]
        syms = [s for s in S.universe(252, d0) if s not in DATA_EXCLUDE.get(rname, ())]
        R, B, E, cyc = {}, {}, {}, []
        for s in syms:
            b = data[s]
            a, z = b.idx(d0), b.idx(d1, "right") - 1
            if a < 252 or z <= a:
                continue
            vals, expo, cycles = sim_one(v, b, a, z, slip, engine)
            bh = np.concatenate([[1.0], b.c[a:z + 1] / b.c[a - 1]])
            dd = b.days[a:z + 1]
            R[s] = pd.Series(vals[1:] / vals[:-1] - 1, index=dd)
            B[s] = pd.Series(bh[1:] / bh[:-1] - 1, index=dd)
            E[s] = pd.Series(expo, index=dd)
            for cy in cycles:
                cy["sym"] = s
            cyc += cycles
        R, B, E = pd.DataFrame(R).sort_index(), pd.DataFrame(B).sort_index(), pd.DataFrame(E).sort_index()
        for w in S.SLICES.get(rname, [rname]):
            wd0, wd1 = S.WINDOWS[w]
            m = (R.index >= wd0) & (R.index <= wd1)
            ps, _ = port_stats(R[m], B[m], E[m])
            ps.update(cyc_stats(cut_cycles(cyc, wd0, wd1, data), ps["종목수"]))
            out[w] = ps
        if rname == "2026":
            m = (R.index >= SHOCK[0]) & (R.index <= SHOCK[1])
            rp = R[m].mean(axis=1).fillna(0).to_numpy()
            bp = B[m].mean(axis=1).fillna(0).to_numpy()
            e26 = out["2026"]["평균노출"]
            pr = np.concatenate([[1.0], np.cumprod(1 + rp)])
            pb = np.concatenate([[1.0], np.cumprod(1 + bp)])
            pf = np.concatenate([[1.0], np.cumprod(1 + e26 * bp)])
            out["AI쇼크"] = {"수익": float(pr[-1] - 1), "낙폭": S._mdd(pr), "BH수익": float(pb[-1] - 1), "BH낙폭": S._mdd(pb),
                           "고정노출_수익": float(pf[-1] - 1), "쇼크_평균노출": float(E[m].mean(axis=1).mean())}
    return key, out


def calmar(s):
    return s["포트_수익"] / abs(s["포트_낙폭"]) if s["포트_낙폭"] < 0 else float("inf")


# ============================================================================ 점검
def check():
    data = S.load()
    syms = S.universe(252, "2026-01-02")
    d0, d1 = S.RUNS["2026"]
    # 1) sim_ma = stock_lab.simulate(MAPolicy) 가치 동치(전체 B·C-sma 규칙, 2026·2025, 미끄러짐 0·0.5%)
    worst = 0.0
    for k, v in VARIANTS.items():
        if v["fam"] == "A" or (v["fam"] == "C" and v["kind"] == "dd"):
            continue
        for rname in ("2026", "2025"):
            r0, r1 = S.RUNS[rname]
            for s in syms[::7]:
                b = data[s]
                a, z = b.idx(r0), b.idx(r1, "right") - 1
                for slip in (0.0, SLIP_SENS):
                    v1, _, _ = sim_ma(b, a, z, v["lines"], v["u"], v["band"], slip)
                    v2, _, _ = S.simulate(b, a, z, MAPolicy(v["lines"], v["u"], v["band"]), slip)
                    worst = max(worst, float(np.max(np.abs(v1 - v2))))
    print("sim_ma vs stock_lab 정책 가치 최대 차(0이어야 함)", worst)
    # 2) 선 하나·100% 매도 = stock_lab 전량 매도 정책과 같은가(주기 손익도)
    b = data[syms[5]]
    a, z = b.idx(d0), b.idx(d1, "right") - 1
    _, _, c1 = sim_ma(b, a, z, (50,), 1.0, 0.0)
    _, c2, _ = S.simulate(b, a, z, MAPolicy((50,), 1.0, 0.0))
    g1 = [round(c["gain"], 12) for c in c1]
    g2 = [round(c["gain"], 12) for c in c2]
    print("SMA50 전량: 주기 손익 일치", g1 == g2, len(g1))
    # 2b) 헛 왕복 상계판 = tranche_lab 원본(미끄러짐 0): 모든 A·C-dd 규칙, 2026·2025, 종목 1/5
    worst, n_ph, n_raw_resell, prod_gap = 0.0, 0, 0, 0.0
    for k, v in VARIANTS.items():
        if make_spec(v, 0.0) is None:
            continue
        for rname in ("2026", "2025"):
            r0, r1 = S.RUNS[rname]
            for s in syms[::5]:
                b = data[s]
                a, z = b.idx(r0), b.idx(r1, "right") - 1
                v1, e1, c1 = T.simulate(b, a, z, make_spec(v, 0.0))
                v2, e2, c2 = sim_tr_net(b, a, z, make_spec(v, 0.0))
                worst = max(worst, float(np.max(np.abs(v1 - v2))), float(np.max(np.abs(e1 - e2))))
                n_ph += sum(cy["phantom"] for cy in c2)
                ex = {(cy["x"], cy["exit_day"], cy["exit_px"]) for cy in c1}
                n_raw_resell += sum(1 for cy in c1 if not cy["open"] and cy["kind"] == "되돌림"
                                    and (cy["x"], cy["re_day"], cy["re_px"]) in ex)
                # 묶은 주기의 (1+손익) = 원본 쪼갠 주기들의 (1+손익) 곱 (미끄러짐 0, 같은 조각끼리)
                for x in {cy["x"] for cy in c2}:
                    p1 = np.prod([1 + cy["gain"] for cy in c1 if cy["x"] == x])
                    p2 = np.prod([1 + cy["gain"] for cy in c2 if cy["x"] == x])
                    prod_gap = max(prod_gap, abs(p1 - p2))
    print("상계판 vs tranche_lab 원본(미끄러짐 0) 가치·노출 최대 차(0이어야 함)", worst)
    print("헛 왕복 수: 상계판", n_ph, "원본에서 같은 시가 되사기→재매도", n_raw_resell, "(같아야 함)")
    print("묶은 주기 곱 vs 쪼갠 주기 곱 최대 차(0이어야 함)", prod_gap)
    # 3) 미래 교란: t 이후 가격을 바꿔도 t까지 가치가 같아야 함(A 시간 되돌림·공통 깊은 재매수, B 띠)
    rng = np.random.default_rng(3)
    keys = ["A|S10·Δ5·n4|g10·time", "A|S20·Δ10·n4|D40·recover", "A|S5·Δ5·n2|g-·recover", "B|B1", "B|B6", "C|sma50|q25"]
    worst = 0.0
    for k in keys:
        for s in syms[:12]:
            b = data[s]
            a, z = b.idx(d0), b.idx(d1, "right") - 1
            v1, e1, _ = sim_one(VARIANTS[k], b, a, z, 0.0)
            for Tt in (a + 40, a + 120):
                b2 = S.Bars(b.sym, b.days, b.o.copy(), b.h.copy(), b.l.copy(), b.c.copy(), b.v)
                for arr in (b2.o, b2.h, b2.l, b2.c):
                    arr[Tt + 1:] *= 1 + rng.normal(0, 0.05, len(arr) - Tt - 1)
                b2.h[Tt + 1:] = np.maximum.reduce([b2.h[Tt + 1:], b2.o[Tt + 1:], b2.c[Tt + 1:]])
                b2.l[Tt + 1:] = np.minimum.reduce([b2.l[Tt + 1:], b2.o[Tt + 1:], b2.c[Tt + 1:]])
                v2, _, _ = sim_one(VARIANTS[k], b2, a, z, 0.0)
                worst = max(worst, float(np.max(np.abs(v1[:Tt - a + 2] - v2[:Tt - a + 2]))))
    print("미래 교란 뒤 t까지 가치 최대 차(0이어야 함)", worst)
    # 4) 손절 없는 사다리(손절 깊이 99%) = 매수·보유
    v = {"fam": "A", "xs": [0.99, 0.995], "ds": None, "fb": "recover"}
    b = data[syms[0]]
    a, z = b.idx(d0), b.idx(d1, "right") - 1
    v1, _, _ = sim_one(v, b, a, z, 0.0)
    bh = np.concatenate([[1.0], b.c[a:z + 1] / b.c[a - 1]])
    print("닿지 않는 사다리 vs 매수·보유 최대 차(0이어야 함)", float(np.max(np.abs(v1 - bh))))
    print("격자: A", sum(1 for x in VARIANTS.values() if x["fam"] == "A"), "B", sum(1 for x in VARIANTS.values() if x["fam"] == "B"),
          "C", sum(1 for x in VARIANTS.values() if x["fam"] == "C"), "합", len(VARIANTS))


# ============================================================================ 본 실행
def frac(ns, f):
    return (sum(1 for n in ns if f(n)) / len(ns)) if ns else float("nan")


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def main():
    t0 = time.time()
    keys = list(VARIANTS)
    with Pool(12) as pool:
        out = dict(pool.map(run_variant, [(k, 0.0, RUN_KEYS) for k in keys], chunksize=2))
    print(f"{len(keys)} 변형 완료 {time.time() - t0:.0f}s")
    k0 = keys[0]
    bh = {w: {kk: out[k0][w][kk] for kk in ("포트BH_수익", "포트BH_낙폭", "종목수")} for w in out[k0] if w != "AI쇼크"}
    bh["AI쇼크"] = {"수익": out[k0]["AI쇼크"]["BH수익"], "낙폭": out[k0]["AI쇼크"]["BH낙폭"]}

    # ---- 고르기(앞 구간만, A·B 136개)
    front = {n: out[n]["2026앞"] for n in POOL}
    g1 = max(POOL, key=lambda n: front[n]["효율선초과"])
    elig = [n for n in POOL if front[n]["손절주기"] >= G2_MIN_CYCLES and np.isfinite(front[n]["기대값"])]
    g2 = max(elig, key=lambda n: front[n]["기대값"])
    g3 = max(POOL, key=lambda n: calmar(front[n]))
    picks = {"G1": g1, "G2": g2, "G3": g3}
    # 계열별 앞 1등(보고용, 고르기 규칙 같음)
    fam_best = {f: max([n for n in POOL if VARIANTS[n]["fam"] == f], key=lambda n: front[n]["효율선초과"]) for f in ("A", "B")}

    # ---- 미끄러짐 0.5%: 전 규칙(상계판, 2026·2025) + 고른 변형은 원본 엔진(헛 왕복에도 비용)으로도
    uniq = sorted(set(picks.values()) | set(fam_best.values()))
    with Pool(12) as pool:
        slip = dict(pool.map(run_variant, [(k, SLIP_SENS, ["2026", "2025"]) for k in keys], chunksize=2))
        slip_raw = dict(pool.map(run_variant, [(k, SLIP_SENS, ["2026", "2025"], "raw") for k in uniq]))
    print(f"미끄러짐 판 완료 {time.time() - t0:.0f}s")

    # ---- 고르기에 정보가 있나
    def pass_set(pool_keys, metric):
        pos = [n for n in pool_keys if out[n]["2026앞"][metric] > 0]
        bk = lambda n: out[n]["2026뒤"][metric] > 0
        yk = lambda n: out[n]["2025"][metric] > 0
        return {"지표": metric, "앞>0 개수": len(pos), "전체": len(pool_keys),
                "뒤>0 | 앞>0": frac(pos, bk), "뒤>0 무조건": frac(pool_keys, bk),
                "2025>0 | 앞>0": frac(pos, yk), "2025>0 무조건": frac(pool_keys, yk),
                "둘다>0 | 앞>0": frac(pos, lambda n: bk(n) and yk(n)), "둘다>0 무조건": frac(pool_keys, lambda n: bk(n) and yk(n))}
    ps = {"효율선초과": pass_set(POOL, "효율선초과"), "타이밍값": pass_set(POOL, "타이밍값"),
          "A_효율선초과": pass_set([n for n in POOL if n.startswith("A")], "효율선초과"),
          "B_효율선초과": pass_set([n for n in POOL if n.startswith("B")], "효율선초과")}

    # ---- 묶음 요약(넓게 vs 좁게, 조각 수, 간격, 재매수 방식)
    A_keys = [n for n in POOL if n.startswith("A")]
    metr = [("2026", "포트_수익"), ("2026", "포트_낙폭"), ("2026", "평균노출"), ("AI쇼크", "수익"),
            ("2026앞", "타이밍값"), ("2026뒤", "타이밍값"), ("2025", "타이밍값"),
            ("2026앞", "효율선초과"), ("2026뒤", "효율선초과"), ("2025", "효율선초과"),
            ("2026", "주기_종목당"), ("2026", "승률"), ("2026", "손익비"), ("2026", "기대값"), ("2026", "포지션기여_종목당"),
            ("2025", "포트_수익"), ("2022", "포트_수익"), ("2022", "포트_낙폭")]

    def group(attr, vals_):
        g = {}
        for val in vals_:
            ks = [n for n in A_keys if VARIANTS[n][attr] == val]
            g[str(val)] = {f"{w}_{m}": med([out[n][w][m] for n in ks]) for w, m in metr}
            g[str(val)]["개수"] = len(ks)
            g[str(val)]["타이밍값+_비율"] = {w: frac(ks, lambda n: out[n][w]["타이밍값"] > 0) for w in MAIN_W}
        return g
    groups = {"S": group("s", A_S), "n": group("n", A_N), "delta": group("delta", A_DELTA), "col": group("col", REBUY_COLS)}

    # ---- 타이밍값 요약(같은 평균 노출 고정 대비)
    tv = {}
    for fam in ("A", "B", "C"):
        ks = [n for n in VARIANTS if VARIANTS[n]["fam"] == fam]
        tv[fam] = {w: {"+비율": frac(ks, lambda n: out[n][w]["타이밍값"] > 0), "중앙": med([out[n][w]["타이밍값"] for n in ks]),
                       "효율선+비율": frac(ks, lambda n: out[n][w]["효율선초과"] > 0)}
                   for w in ("2026", "2026앞", "2026뒤", "2025", "2022") + tuple(CONTEXT_YEARS)}
        tv[fam]["AI쇼크_고정보다덜잃음"] = frac(ks, lambda n: out[n]["AI쇼크"]["수익"] > out[n]["AI쇼크"]["고정노출_수익"])
        tv[fam]["AI쇼크_BH보다덜잃음"] = frac(ks, lambda n: out[n]["AI쇼크"]["수익"] > out[n]["AI쇼크"]["BH수익"])
    # 세 구간(앞·뒤·2025) 모두 타이밍값 > 0 / 효율선초과 > 0
    all3_tv = [n for n in VARIANTS if all(out[n][w]["타이밍값"] > 0 for w in MAIN_W)]
    all3_ex = [n for n in VARIANTS if all(out[n][w]["효율선초과"] > 0 for w in MAIN_W)]
    all3_exp = [n for n in VARIANTS if all(out[n][w]["기대값"] > 0 and out[n][w]["손절주기"] >= 10 for w in MAIN_W)]
    all3_tv_slip = [n for n in VARIANTS if all(slip[n][w]["타이밍값"] > 0 for w in MAIN_W)]
    all3_ex_slip = [n for n in VARIANTS if all(slip[n][w]["효율선초과"] > 0 for w in MAIN_W)]
    slip_sum = {}
    for fam in ("A", "B", "C"):
        ks = [n for n in VARIANTS if VARIANTS[n]["fam"] == fam]
        slip_sum[fam] = {w: {"+비율": frac(ks, lambda n: slip[n][w]["타이밍값"] > 0), "중앙": med([slip[n][w]["타이밍값"] for n in ks]),
                             "효율선+비율": frac(ks, lambda n: slip[n][w]["효율선초과"] > 0)} for w in ("2026",) + MAIN_W}
        slip_sum[fam]["세구간_타이밍+"] = frac(ks, lambda n: n in all3_tv_slip)
        slip_sum[fam]["세구간_타이밍+_미끄러짐0"] = frac(ks, lambda n: n in all3_tv)
    # 세 구간 타이밍값 + 집합이 미끄러짐 0.5%에서 얼마나 남나
    tv_keep = frac(all3_tv, lambda n: n in all3_tv_slip)
    # 상계판 vs 원본: 헛 왕복 빈도(2026 전체, 낙폭 사다리 되돌림 방식별 중앙)
    phantom_by_col = {c: med([out[f"A|{sk}|{c}"]["2026"]["헛왕복_주기당"] for sk in {VARIANTS[n]["shape"] for n in A_keys}])
                      for c in REBUY_COLS}
    ctx_bh = {y: {"BH수익": bh[y]["포트BH_수익"], "BH낙폭": bh[y]["포트BH_낙폭"]} for y in CONTEXT_YEARS}

    payload = {
        "grid": {"A_S": A_S, "A_DELTA": A_DELTA, "A_N": A_N, "A_GAP": A_GAP, "A_FALLBACK": A_FALLBACK, "A_COMMON_D": A_COMMON_D,
                 "RECOVER_B": RECOVER_B, "TIME_N": TIME_N, "B_SETS": B_SETS, "C_DD_X": C_DD_X, "C_SMA": C_SMA, "C_Q": C_Q,
                 "SHOCK": SHOCK, "SLIP_SENS": SLIP_SENS, "G2_MIN_CYCLES": G2_MIN_CYCLES, "DATA_EXCLUDE": {k: sorted(v) for k, v in DATA_EXCLUDE.items()},
                 "n_total": len(VARIANTS), "n_pool": len(POOL),
                 "n_by_fam": {f: sum(1 for v in VARIANTS.values() if v["fam"] == f) for f in "ABC"}},
        "variants": {k: {kk: vv for kk, vv in v.items()} for k, v in VARIANTS.items()},
        "bh": bh, "ctx_bh": ctx_bh, "picks": picks, "fam_best": fam_best, "slip": slip, "slip_raw": slip_raw, "pass_set": ps,
        "groups": groups, "timing": tv, "all3_tv": all3_tv, "all3_ex": all3_ex, "all3_exp": all3_exp,
        "all3_tv_slip": all3_tv_slip, "all3_ex_slip": all3_ex_slip, "slip_sum": slip_sum, "tv_keep": tv_keep,
        "phantom_by_col": phantom_by_col, "results": out,
    }
    p = S.save("p3_LADDER", payload)
    print("저장", p, f"{time.time() - t0:.0f}s")
    write_md(json.loads(p.read_text(encoding="utf-8")))


# ============================================================================ 보고서
def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def pp(x, d=0):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def ppt(x, d=1):   # %p
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}"


def f2(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}"


def short(k, V):
    v = V[k]
    if v["fam"] == "A":
        return f"{shape_label(v['s'], v['delta'], v['n'])} · {REBUY_LABEL[v['col']]}"
    if v["fam"] == "B":
        return k.split("|")[1] + " " + "·".join(map(str, v["lines"])) + f" 선마다 {int(v['u'] * 100)}%" + (" 띠2%" if v["band"] else "")
    if v["kind"] == "dd":
        return f"−{_p(v['x'])}% 선에서 {_p(v['q'])}%"
    return f"SMA{v['lines'][0]} 선에서 {_p(v['u'])}%"


def write_md(P):
    out, V, bh, picks = P["results"], P["variants"], P["bh"], P["picks"]
    ps, tv, G = P["pass_set"], P["timing"], P["groups"]
    for v in V.values():                         # json 왕복으로 튜플이 목록이 됨
        if "lines" in v:
            v["lines"] = tuple(v["lines"])
    pool = [k for k in V if V[k]["fam"] in ("A", "B")]
    L = []
    A = L.append
    g1 = picks["G1"]
    o1 = out[g1]
    nA, nB, nC = (P["grid"]["n_by_fam"][f] for f in "ABC")
    sh = bh["AI쇼크"]

    # ---------- 결론
    A("# 3단계 LADDER: 나눠 팔기 사다리 — 낙폭 사다리 × 이동평균 사다리 (2026-10-10)\n")
    A(f"스크립트 `code/p3_LADDER.py`, 전 결과 `results/p3_LADDER.json`. AI 종목 전체(2026 {bh['2026']['종목수']}종목, 2025 {bh['2025']['종목수']}종목)를 "
      "각각 창 시작에 다 들고, 고점에서 빠질 때 조각씩 팔고 조각씩 되사는 규칙을 얹었다. 엔진 `code/tranche_lab.py`(낙폭 사다리), "
      "이동평균 사다리는 stock_lab 정책과 같은 매매를 이 스크립트의 `sim_ma`로 계산했다.\n")
    A("## 결론\n")
    A(_conclusion(P))
    # ---------- 격자
    A("## 격자 (사전 고정, 결과를 본 뒤 바꾸지 않음)\n")
    A(f"- **A 낙폭 사다리 {nA}개.** 고점 = 창 시작 직전 종가부터 잰 추적 최고 종가(t−1까지). 조각 i 손절 깊이 = S + i·Δ.")
    A("  - 모양 16 = 시작 깊이 S 4(5·10·15·20%) × 간격 Δ 2(5·10%p) × 조각 수 n 2(2 = 50%씩, 4 = 25%씩).")
    A("  - 재매수 6 = 조각별 깊은 지정가(손절 깊이 + 10%p, + 20%p, 없음) 3 × 되돌림 2(회복: 종가 ≥ 판 값 × 1.03 → 다음 시가 / 판 날 + 20거래일 시가). 16 × 6 = 96.")
    A("  - 공통 깊은 재매수 2 = 모든 조각 고점 대비 −40% 또는 −50% 지정가 + 회복 되돌림. 16 × 2 = 32. 손절 깊이가 D 이상인 조각(예: −40·−50% 조각에 D −40%)은 깊은 재매수 없이 회복만.")
    A(f"- **B 이동평균 사다리 {nB}개.** 종가 < SMA 선 하나를 깰 때마다 판매 단위만큼 판다(깬 선 수 k → 보유 1 − 단위 × k, 다음 시가). 선 위 종가가 되면 그 몫을 다음 시가에 되산다.")
    A("  - 선 묶음 3(20/50/100/200, 10/20/50/100, 50/100/150/200) × 25% = 3, 2개 선 × 50%(50/200, 20/100) = 2, 앞 3묶음의 띠 2% 판(되살 때 선 × 1.02 위 종가) = 3.")
    A(f"- **C 비교 기준 {nC}개(고르기 대상 아님).** 같은 선에서 25%·50%·100%만 팔기: 낙폭선 −10%·−20%(재매수 −20%·−30% 지정가 + 회복), SMA50·SMA100(선 회복 시 되사기). 2 × 3 + 2 × 3 = 12.")
    A(f"- 합계 {P['grid']['n_total']}개, 고르기 대상 A·B {P['grid']['n_pool']}개. 고르기 3종(사전 고정): G1 앞 효율선초과 최대, G2 앞 주기 기대값 최대(주기 {G2_MIN_CYCLES}개 이상), G3 앞 칼마 최대. "
      "미끄러짐 0으로 고르고 고른 변형만 0.5%로 다시 쟀다.\n")
    # ---------- 변수 시점
    A("## 변수 시점\n")
    A("| 변수 | 언제 아는가 | 체결 |\n|---|---|---|")
    A("| 고점(창 시작 직전 종가부터 최고 종가) | t−1 종가 | 손절가 = 고점 × (1 − 깊이), t일 장중 스탑(시가가 선 아래면 시가) |")
    A("| 깊은 재매수 지정가 = 고점 × (1 − D) | t−1 종가 | t일 장중 지정가(min(시가, 지정가)), 손절 뒤 같은 날 닿으면 그날 |")
    A("| 회복 되돌림(종가 ≥ 판 값 × 1.03) | t−1 종가 | t일 시가 |")
    A("| 20거래일 되돌림 | 판 날 | 판 날 + 20거래일 시가 |")
    A("| SMA10·20·50·100·150·200, 띠 2% | t−1 종가 | t일 시가 |")
    A("| 평균 노출(타이밍값의 비교 기준) | 창이 끝난 뒤(설명용) | 매매에 쓰지 않음 |\n")

    # ---------- 핵심 격자 표
    shapes = []
    for s_ in P["grid"]["A_S"]:
        for dl in P["grid"]["A_DELTA"]:
            for n in P["grid"]["A_N"]:
                shapes.append((shape_key(s_, dl, n), shape_label(s_, dl, n)))
    b26 = bh["2026"]
    A("## 핵심 격자 1: A 낙폭 사다리 — 2026 전체 수익 · 평균 노출 · 7월 쇼크(6/22~7/29) 손실\n")
    A(f"행 = 손절 사다리(고점 대비 깊이, 조각 비중), 열 = 재매수 방식. 매수·보유: 2026 {pct(b26['포트BH_수익'])} (낙폭 {pct(b26['포트BH_낙폭'])}), "
      f"7월 쇼크 {pct(sh['수익'])}. '회복' = 판 값보다 3% 위 종가가 나오면 다음 시가에 되사기, '20일' = 판 날부터 20거래일 뒤 시가.\n")
    A("| 손절 사다리 | " + " | ".join(REBUY_LABEL[c] for c in REBUY_COLS) + " |")
    A("|---|" + "---|" * len(REBUY_COLS))
    for sk, lab in shapes:
        cells = []
        for c in REBUY_COLS:
            o = out[f"A|{sk}|{c}"]
            cells.append(f"{pct(o['2026']['포트_수익'], 0)} · {o['2026']['평균노출']:.2f} · {pct(o['AI쇼크']['수익'], 0)}")
        A(f"| {lab} | " + " | ".join(cells) + " |")
    A("\n## 핵심 격자 2: 타이밍값(같은 평균 노출로 늘 덜 들고 있기 대비 수익 차, %p) — 앞 / 뒤 / 2025\n")
    A("+면 '선을 이탈할 때 판 것'이 그냥 그만큼 덜 들고 있기보다 나았다는 뜻이다.\n")
    A("| 손절 사다리 | " + " | ".join(REBUY_LABEL[c] for c in REBUY_COLS) + " |")
    A("|---|" + "---|" * len(REBUY_COLS))
    for sk, lab in shapes:
        cells = []
        for c in REBUY_COLS:
            o = out[f"A|{sk}|{c}"]
            cells.append(" / ".join(ppt(o[w]["타이밍값"], 0) for w in MAIN_W))
        A(f"| {lab} | " + " | ".join(cells) + " |")
    A("\n## 핵심 격자 3: 효율선초과(같은 최대 낙폭의 고정 노출 대비, %p) — 앞 / 뒤 / 2025\n")
    A("| 손절 사다리 | " + " | ".join(REBUY_LABEL[c] for c in REBUY_COLS) + " |")
    A("|---|" + "---|" * len(REBUY_COLS))
    for sk, lab in shapes:
        cells = []
        for c in REBUY_COLS:
            o = out[f"A|{sk}|{c}"]
            cells.append(" / ".join(ppt(o[w]["효율선초과"], 0) for w in MAIN_W))
        A(f"| {lab} | " + " | ".join(cells) + " |")

    # ---------- B
    A("\n## B 이동평균 사다리 (종가 판단 → 다음 시가)\n")
    A("| 규칙 | 2026 수익 / 낙폭 | 평균 노출 | 7월 쇼크 | 앞 수익 | 뒤 수익 | 2025 수익 | 2022 수익 | 타이밍값 앞 / 뒤 / 2025 (%p) | 효율선초과 앞 / 뒤 / 2025 (%p) | 2026 주기·종목당 | 2026 승률·손익비·기대값 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    A(f"| 매수·보유 | {pct(b26['포트BH_수익'])} / {pct(b26['포트BH_낙폭'])} | 1.00 | {pct(sh['수익'])} | {pct(bh['2026앞']['포트BH_수익'])} | "
      f"{pct(bh['2026뒤']['포트BH_수익'])} | {pct(bh['2025']['포트BH_수익'])} | {pct(bh['2022']['포트BH_수익'])} | 0 | 0 | — | — |")
    for k in [k for k in V if V[k]["fam"] == "B"]:
        A(_row_full(k, V, out))
    # ---------- C
    A("\n## C 같은 선에서 25%·50%·100% 팔기 (사용자 질문 2)\n")
    A("| 선 | 파는 몫 | 2026 수익 / 낙폭 | 평균 노출 | 7월 쇼크 (같은 노출 고정) | 앞 수익 | 뒤 수익 | 2025 수익 | 2022 수익 | 타이밍값 앞 / 뒤 / 2025 (%p) | 효율선초과 앞 / 뒤 / 2025 (%p) | 2026 주기 승률·손익비·기대값 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in [k for k in V if V[k]["fam"] == "C"]:
        v, o = V[k], out[k]
        line = f"고점 −{_p(v['x'])}%" if v["kind"] == "dd" else f"SMA{v['lines'][0]}"
        q = v["q"] if v["kind"] == "dd" else v["u"]
        A(f"| {line} | {_p(q)}% | {pct(o['2026']['포트_수익'])} / {pct(o['2026']['포트_낙폭'])} | {o['2026']['평균노출']:.2f} | "
          f"{pct(o['AI쇼크']['수익'])} ({pct(o['AI쇼크']['고정노출_수익'])}) | {pct(o['2026앞']['포트_수익'])} | {pct(o['2026뒤']['포트_수익'])} | "
          f"{pct(o['2025']['포트_수익'])} | {pct(o['2022']['포트_수익'])} | " + " / ".join(ppt(o[w]["타이밍값"], 1) for w in MAIN_W) + " | "
          + " / ".join(ppt(o[w]["효율선초과"], 1) for w in MAIN_W) + f" | {pp(o['2026']['승률'])} · {f2(o['2026']['손익비'])} · {pct(o['2026']['기대값'])} |")
    A("\n- 일부만 파는 규칙의 주기 손익(판 값/되산 값)은 전량과 같다(같은 날 같은 값에 판다). 다른 것은 그 손익이 포지션에 미치는 크기뿐이라, 타이밍값도 대략 파는 몫에 비례한다.\n")

    # ---------- 고르기
    A("## 고르기 결과 (동일가중 포트폴리오 수익 / 최대 낙폭, 미끄러짐 0)\n")
    A("| 고르기 | 규칙 | 2026 전체 | 앞(고르기) | 앞 효율선초과 | 앞 타이밍값 | 뒤(확인) | 뒤 효율선초과 | 뒤 타이밍값 | 2025(보조) | 2025 효율선초과 | 2025 타이밍값 | 2022(맥락) | 7월 쇼크 | 2026 평균 노출 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    A(f"| 매수·보유 | — | {pct(b26['포트BH_수익'])} / {pct(b26['포트BH_낙폭'])} | {pct(bh['2026앞']['포트BH_수익'])} / {pct(bh['2026앞']['포트BH_낙폭'])} | 0 | 0 "
      f"| {pct(bh['2026뒤']['포트BH_수익'])} / {pct(bh['2026뒤']['포트BH_낙폭'])} | 0 | 0 | {pct(bh['2025']['포트BH_수익'])} / {pct(bh['2025']['포트BH_낙폭'])} | 0 | 0 "
      f"| {pct(bh['2022']['포트BH_수익'])} / {pct(bh['2022']['포트BH_낙폭'])} | {pct(sh['수익'])} | 1.00 |")
    rows = list(picks.items()) + [(f"{f} 계열 앞 1등", k) for f, k in P["fam_best"].items() if k not in picks.values()]
    for lab, k in rows:
        A(_row_pick(lab, k, V, out[k], with_shock=True))
    for lab, k in rows:
        A(_row_pick(f"{lab} 미끄러짐 0.5%", k, V, P["slip"][k], with_shock=False))
        if V[k]["fam"] == "A" and k in P["slip_raw"]:
            A(_row_pick(f"{lab} 미끄러짐 0.5%(원본 엔진, 헛 왕복에도 비용)", k, V, P["slip_raw"][k], with_shock=False))
    e = ps["효율선초과"]
    t_ = ps["타이밍값"]
    A(f"\n- G1 앞 효율선초과 1등, G2 앞 주기 기대값 1등(주기 {G2_MIN_CYCLES}개 이상), G3 앞 칼마 1등. 고르기 대상 A·B {P['grid']['n_pool']}개.")
    A(f"- 고르기에 정보가 있나(효율선초과): 앞 > 0인 변형 {e['앞>0 개수']}/{e['전체']}개. 그중 뒤 > 0 {pp(e['뒤>0 | 앞>0'])}(무조건 {pp(e['뒤>0 무조건'])}), "
      f"2025 > 0 {pp(e['2025>0 | 앞>0'])}(무조건 {pp(e['2025>0 무조건'])}), 둘 다 {pp(e['둘다>0 | 앞>0'])}(무조건 {pp(e['둘다>0 무조건'])}).")
    A(f"- 같은 질문을 타이밍값으로: 앞 > 0 {t_['앞>0 개수']}/{t_['전체']}개. 그중 뒤 > 0 {pp(t_['뒤>0 | 앞>0'])}(무조건 {pp(t_['뒤>0 무조건'])}), "
      f"2025 > 0 {pp(t_['2025>0 | 앞>0'])}(무조건 {pp(t_['2025>0 무조건'])}), 둘 다 {pp(t_['둘다>0 | 앞>0'])}(무조건 {pp(t_['둘다>0 무조건'])}).")
    for f in ("A", "B"):
        x = ps[f"{f}_효율선초과"]
        A(f"- {f}만: 앞 효율선초과 > 0 {x['앞>0 개수']}/{x['전체']}개 → 뒤 > 0 {pp(x['뒤>0 | 앞>0'])}(무조건 {pp(x['뒤>0 무조건'])}), 2025 > 0 {pp(x['2025>0 | 앞>0'])}(무조건 {pp(x['2025>0 무조건'])}).")
    A(f"- 앞·뒤·2025 세 구간 모두 타이밍값 > 0: {len(P['all3_tv'])}/{len(V)}개" + (f" ({', '.join(short(k, V) for k in P['all3_tv'][:6])}{' 등' if len(P['all3_tv']) > 6 else ''})" if P['all3_tv'] else "") +
      f". 효율선초과 > 0: {len(P['all3_ex'])}개" + (f" ({', '.join(short(k, V) for k in P['all3_ex'][:6])})" if P['all3_ex'] else "") +
      f". 주기 기대값 > 0(주기 10개 이상): {len(P['all3_exp'])}개" + (f" ({', '.join(short(k, V) for k in P['all3_exp'][:6])})" if P['all3_exp'] else "") + ".\n")

    # ---------- 승률·손익비
    A("## 손절 주기 승률·손익비 (주기 = 판 가격/되산 가격 − 1, 못 되산 주기는 그 창 끝 종가)\n")
    A("조각 수 기준. 같은 규칙 안에서는 조각 비중이 같아 '비중 가중' 승률·기대값이 조각 수 기준과 같다. 포지션 기여 = 주기 손익 × 조각 비중(그 주기가 종목 전체 포지션에 준 몫).\n")
    A("| 규칙 | 앞 n | 앞 승률 | 앞 손익비 | 앞 기대값 | 뒤 n | 뒤 승률 | 뒤 손익비 | 뒤 기대값 | 2025 n | 2025 승률 | 2025 손익비 | 2025 기대값 | 2026 포지션 기여/주기 | 깊은 재매수 비율(2026) | 헛 왕복/주기(2026) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    show = list(dict.fromkeys(list(picks.values()) + list(P["fam_best"].values()) + [k for k in V if V[k]["fam"] == "B"]))
    for k in show:
        A(_row_cyc(short(k, V), out[k]))
    A("\n재매수 방식별 중앙값(A 낙폭 사다리, 모양 16개씩)\n")
    A("| 재매수 방식 | 2026 수익 | 7월 쇼크 | 2026 승률 | 2026 손익비 | 2026 기대값 | 깊은 재매수 비율 | 주기/종목 | 타이밍값 중앙 앞 / 뒤 / 2025 / 2022 (%p) | 타이밍값 + 비율 앞 / 뒤 / 2025 | 미끄러짐 0.5%: + 비율 앞 / 뒤 / 2025 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in REBUY_COLS:
        g = G["col"][c]
        sg = _slip_group(P, "col", c)
        dp = med([out[k]["2026"]["깊은재매수비율"] for k in V if V[k]["fam"] == "A" and V[k]["col"] == c])
        A(f"| {REBUY_LABEL[c]} | {pct(g['2026_포트_수익'])} | {pct(g['AI쇼크_수익'])} | {pp(g['2026_승률'])} | {f2(g['2026_손익비'])} | "
          f"{pct(g['2026_기대값'])} | {pp(dp)} | {g['2026_주기_종목당']:.1f} | "
          + " / ".join(ppt(g[f'{w}_타이밍값'], 1) for w in MAIN_W) + f" / {ppt(_col_med(P, 'col', c, '2022', '타이밍값'), 1)} | "
          + " / ".join(pp(g['타이밍값+_비율'][w]) for w in MAIN_W) + " | " + " / ".join(pp(sg['+비율'][w]) for w in MAIN_W) + " |")
    A("\n- 손익분기 손익비 = (1 − 승률)/승률: 승률 30% → 2.33, 40% → 1.5, 50% → 1.0.\n")

    # ---------- 넓게 vs 좁게
    A("## 넓게 vs 좁게 (A, 시작 깊이 S별 중앙값, 각 32개)\n")
    A("'넓게' = 고점에서 많이 빠져야 판다(−20%부터), '좁게' = 조금만 빠져도 판다(−5%부터). 2단계의 '쓴다면 넓게'는 손절선을 고점에서 멀리(추적 20~25%, 또는 100·200일선 같은 긴 선) 두라는 뜻이었다.\n")
    A("| 시작 깊이 | 2026 수익 | 2026 낙폭 | 평균 노출 | 7월 쇼크 (같은 노출 고정) | 타이밍값 앞 / 뒤 / 2025 (%p) | 미끄러짐 0.5% 타이밍값 앞 / 뒤 / 2025 | 효율선초과 앞 / 뒤 / 2025 (%p) | 주기/종목(2026) | 승률 | 손익비 | 기대값 | 2025 수익 | 2022 수익 / 낙폭 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def _wn_row(name, g, sg, shock_fixed):
        return (f"| {name} | {pct(g['2026_포트_수익'])} | {pct(g['2026_포트_낙폭'])} | {g['2026_평균노출']:.2f} | {pct(g['AI쇼크_수익'])} ({pct(shock_fixed)}) | "
                + " / ".join(ppt(g[f'{w}_타이밍값'], 1) for w in MAIN_W) + " | " + " / ".join(ppt(sg[w], 1) for w in MAIN_W) + " | "
                + " / ".join(ppt(g[f'{w}_효율선초과'], 1) for w in MAIN_W)
                + f" | {g['2026_주기_종목당']:.1f} | {pp(g['2026_승률'])} | {f2(g['2026_손익비'])} | {pct(g['2026_기대값'])} | {pct(g['2025_포트_수익'])} | "
                f"{pct(g['2022_포트_수익'])} / {pct(g['2022_포트_낙폭'])} |")
    for s_ in P["grid"]["A_S"]:
        A(_wn_row(f"고점 −{_p(s_)}%부터", G["S"][str(s_)], _slip_group(P, "s", s_), _col_med(P, "s", s_, "AI쇼크", "고정노출_수익")))
    for attr, lab in (("n", "조각 수"), ("delta", "간격")):
        for val, g in G[attr].items():
            vv = int(val) if attr == "n" else float(val)
            nm = f"{val}조각({100 // int(val)}%씩)" if attr == "n" else f"간격 {_p(float(val))}%p"
            A(_wn_row(f"{lab}: {nm}", g, _slip_group(P, attr, vv), _col_med(P, attr, vv, "AI쇼크", "고정노출_수익")))

    # ---------- 미끄러짐
    ss = P["slip_sum"]
    A("\n## 미끄러짐 0.5% (전 148개, 팔 때·살 때 각 0.5%, 헛 왕복은 상계)\n")
    A("| 계열 | 타이밍값 + 비율 앞 / 뒤 / 2025 (미끄러짐 0) | 같은 것 (미끄러짐 0.5%) | 타이밍값 중앙 앞 / 뒤 / 2025 (0.5%, %p) | 세 구간 모두 + (0 → 0.5%) |")
    A("|---|---|---|---|---|")
    for f in ("A", "B", "C"):
        A(f"| {f} | " + " / ".join(pp(tv[f][w]['+비율']) for w in MAIN_W) + " | " + " / ".join(pp(ss[f][w]['+비율']) for w in MAIN_W) + " | "
          + " / ".join(ppt(ss[f][w]['중앙'], 1) for w in MAIN_W) + f" | {pp(ss[f]['세구간_타이밍+_미끄러짐0'])} → {pp(ss[f]['세구간_타이밍+'])} |")
    A(f"\n- 미끄러짐 0에서 세 구간 모두 타이밍값 + 이던 {len(P['all3_tv'])}개 중 0.5%에서도 남은 것 {len(P['all3_tv_slip'])}개({pp(P['tv_keep'])}): "
      + "; ".join(f"{short(k, V)} (앞 {ppt(P['slip'][k]['2026앞']['타이밍값'])} / 뒤 {ppt(P['slip'][k]['2026뒤']['타이밍값'])} / 2025 {ppt(P['slip'][k]['2025']['타이밍값'])}%p, "
                  f"2026 수익 {pct(P['slip'][k]['2026']['포트_수익'])}, 노출 {out[k]['2026']['평균노출']:.2f})" for k in P['all3_tv_slip'])
      + f". 세 구간 모두 효율선초과 + 는 0.5%에서 {len(P['all3_ex_slip'])}개(미끄러짐 0이면 {len(P['all3_ex'])}개).")
    A(f"- 일부 매도는 매매 횟수가 많다. 2026 9개월 동안 종목당 주기 A {med([out[k]['2026']['주기_종목당'] for k in V if V[k]['fam'] == 'A']):.1f}회, "
      f"B {med([out[k]['2026']['주기_종목당'] for k in V if V[k]['fam'] == 'B']):.1f}회(중앙). 왕복마다 1%(0.5% × 2)가 빠지므로 타이밍값 1~3%p는 비용에 그대로 묻힌다.\n")

    # ---------- 보조·맥락
    A("\n## 보조·맥락\n")
    A("타이밍값 > 0 비율 / 효율선초과 > 0 비율(계열별)\n")
    ws_ = ["2026", "2026앞", "2026뒤", "2025"] + CONTEXT_YEARS
    A("| 계열 | " + " | ".join(ws_) + " | 7월 쇼크: 같은 노출 고정보다 덜 잃음 |")
    A("|---|" + "---|" * (len(ws_) + 1))
    for f in ("A", "B", "C"):
        A(f"| {f} | " + " | ".join(f"{pp(tv[f][w]['+비율'])} / {pp(tv[f][w]['효율선+비율'])}" for w in ws_) + f" | {pp(tv[f]['AI쇼크_고정보다덜잃음'])} |")
    yb = ", ".join(f"{y} {pct(P['ctx_bh'][y]['BH수익'], 0)}/{pct(P['ctx_bh'][y]['BH낙폭'], 0)}" for y in CONTEXT_YEARS)
    A(f"\n- 매수·보유(수익/낙폭): {yb}. 2018·2019는 QUBT(장외 시절 가격 오류)를 뺐다.")
    A(_context_lines(P))
    # ---------- 한계
    A("\n## 선택편향·한계\n")
    A(f"- {P['grid']['n_pool']}개 중 앞 1등을 골랐으므로 앞 수치는 위로 치우친다. 뒤·2025를 본 뒤 격자·정의를 바꾸지 않았다.")
    A("- 타이밍값의 비교 기준인 '같은 평균 노출'은 그 창이 끝난 뒤에야 아는 값이다. 규칙을 고르는 데 쓰지 않고, 일부 매도의 효과가 '덜 들기'인지 '때 맞춰 팔기'인지 가르는 설명용으로만 썼다.")
    A("- 2026 뒤 구간은 매수·보유가 마이너스라 효율선의 최선이 현금(노출 0)이다. 그래서 뒤 효율선초과는 사실상 절대 수익이다. 타이밍값은 이 문제가 없다.")
    A("- 생존편향: 종목 목록이 2026-10 기준(+TIME 보유 이력)이다. 사라진 종목에서 손절이 막아 줬을 손실이 빠져 있어 손절에 불리하고, 깊은 재매수에는 유리하다.")
    pc = P["phantom_by_col"]
    A("- 헛 왕복 상계(버그 성격의 정의 보정): tranche_lab 원본은 되돌림으로 되산 조각의 시가가 이미 손절선 아래면 같은 시가에 사고 바로 판다. "
      "미끄러짐 0이면 손익 0이지만 0.5%를 넣으면 왕복마다 1%가 빠지고 주기가 쪼개진다. 2026 주기당 헛 왕복 중앙: "
      + ", ".join(f"{REBUY_LABEL[c]} {pc[c]:.2f}" for c in REBUY_COLS)
      + ". 그래서 이 스크립트의 `sim_tr_net`이 그 왕복만 '매매 없음'으로 상계하고 주기를 '처음 판 값 → 실제로 되산 값'으로 묶었다. "
      "미끄러짐 0에서 원본과 가치·노출 차 1.8e-15, 헛 왕복 수 일치, 묶은 주기 곱 = 쪼갠 주기 곱(`py p3_LADDER.py check`). "
      "고르기는 가치로 하므로 G1·G3는 같고, G2(주기 기대값)도 원본으로 골랐을 때와 같았다. 원본 엔진의 미끄러짐 판은 고르기 표에 따로 실었다.")
    A("- 이동평균 사다리는 비율 목표(보유 1 − 단위 × k)라, 한 선이 깨지고 다른 선이 회복된 날은 k가 같아도 비율을 다시 맞춘다(stock_lab과 같음). 주기는 선별로 기록했다.")
    A("- 동일가중 종목 포트폴리오(매일 재조정) 기준이다. 데스크 7종목 규칙 위 결과가 아니다. 수수료·현금 이자 0, 배당 수정 가격, 시가·장중 체결은 일봉 가정.")
    A("- 7월 쇼크 구간(6/22~7/29)은 사후에 정한 측정 구간이다(2단계와 같은 정의: 그 날짜들의 일수익). 2020은 QUBT 장외 시절이지만 이상 값이 적어 지시대로 빼지 않았다(QUBT 2020 +370%).")
    (S.RESULTS / "p3_LADDER.md").write_text("\n".join(L), encoding="utf-8")
    print("md", S.RESULTS / "p3_LADDER.md")


def _row_full(k, V, out):
    o = out[k]
    return (f"| {short(k, V)} | {pct(o['2026']['포트_수익'])} / {pct(o['2026']['포트_낙폭'])} | {o['2026']['평균노출']:.2f} | {pct(o['AI쇼크']['수익'])} | "
            f"{pct(o['2026앞']['포트_수익'])} | {pct(o['2026뒤']['포트_수익'])} | {pct(o['2025']['포트_수익'])} | {pct(o['2022']['포트_수익'])} | "
            + " / ".join(ppt(o[w]["타이밍값"], 1) for w in MAIN_W) + " | " + " / ".join(ppt(o[w]["효율선초과"], 1) for w in MAIN_W)
            + f" | {o['2026']['주기_종목당']:.1f} | {pp(o['2026']['승률'])} · {f2(o['2026']['손익비'])} · {pct(o['2026']['기대값'])} |")


def _row_pick(lab, k, V, o, with_shock):
    f, b, y, a = o["2026앞"], o["2026뒤"], o["2025"], o["2026"]
    y22 = f"{pct(o['2022']['포트_수익'])} / {pct(o['2022']['포트_낙폭'])}" if "2022" in o else "—"
    shock = pct(o["AI쇼크"]["수익"]) if with_shock and "AI쇼크" in o else "—"
    return (f"| {lab} | {short(k, V)} | {pct(a['포트_수익'])} / {pct(a['포트_낙폭'])} | {pct(f['포트_수익'])} / {pct(f['포트_낙폭'])} | "
            f"{ppt(f['효율선초과'])} | {ppt(f['타이밍값'])} | {pct(b['포트_수익'])} / {pct(b['포트_낙폭'])} | {ppt(b['효율선초과'])} | {ppt(b['타이밍값'])} | "
            f"{pct(y['포트_수익'])} / {pct(y['포트_낙폭'])} | {ppt(y['효율선초과'])} | {ppt(y['타이밍값'])} | {y22} | {shock} | {a['평균노출']:.2f} |")


def _row_cyc(lab, o):
    cells = []
    for w in MAIN_W:
        s = o[w]
        cells += [str(s["손절주기"]), pp(s["승률"]), f2(s["손익비"]), pct(s["기대값"])]
    a = o["2026"]
    return (f"| {lab} | " + " | ".join(cells) + f" | {pct(a['포지션기여_주기당'], 2)} | {pp(a['깊은재매수비율'])} | {f2(a['헛왕복_주기당'])} |")


def _eq(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-9
    return a == b


def _col_med(P, attr, val, w, m):
    V, out = P["variants"], P["results"]
    ks = [k for k in V if V[k]["fam"] == "A" and _eq(V[k][attr], val)]
    return med([out[k][w][m] for k in ks])


def _slip_group(P, attr, val):
    V, sl = P["variants"], P["slip"]
    ks = [k for k in V if V[k]["fam"] == "A" and _eq(V[k][attr], val)]
    g = {w: med([sl[k][w]["타이밍값"] for k in ks]) for w in MAIN_W}
    g["+비율"] = {w: frac(ks, lambda k: sl[k][w]["타이밍값"] > 0) for w in MAIN_W}
    g["2026_수익"] = med([sl[k]["2026"]["포트_수익"] for k in ks])
    return g


def _conclusion(P):
    out, V, bh, picks, G = P["results"], P["variants"], P["bh"], P["picks"], P["groups"]
    g1 = picks["G1"]
    o = out[g1]
    e = P["pass_set"]["효율선초과"]
    gs = G["S"]
    s5, s20 = gs[str(P["grid"]["A_S"][0])], gs[str(P["grid"]["A_S"][-1])]
    c10 = {q: out[f"C|dd10|q{q}"] for q in ("25", "50", "100")}
    sell_only = [k for k in V if V[k]["fam"] == "A" and V[k]["col"].startswith("g-")]
    so_back = frac(sell_only, lambda k: out[k]["2026뒤"]["타이밍값"] > 0)
    so_25 = frac(sell_only, lambda k: out[k]["2025"]["타이밍값"] > 0)
    sg = _slip_group(P, "col", "g20·time")
    L = []
    L.append(
        f"1. **기각(대표 후보).** 나눠 팔기 {P['grid']['n_pool']}개(낙폭 사다리 128 + 이동평균 사다리 8) 중 앞 구간 1등 G1 `{short(g1, V)}`는 "
        f"앞 {pct(o['2026앞']['포트_수익'])}/{pct(o['2026앞']['포트_낙폭'])} → 뒤 {pct(o['2026뒤']['포트_수익'])}(매수·보유 {pct(bh['2026뒤']['포트BH_수익'])}), "
        f"2026 전체 {pct(o['2026']['포트_수익'])}(매수·보유 {pct(bh['2026']['포트BH_수익'])})였다. 앞에서 효율선을 이긴 {e['앞>0 개수']}개 중 뒤에서도 이긴 것 "
        f"{pp(e['뒤>0 | 앞>0'])}(무조건 {pp(e['뒤>0 무조건'])})라 고르기에 정보가 없었다. 미끄러짐 0.5%를 넣으면 앞·뒤·2025 모두 "
        f"'같은 평균 노출로 늘 덜 들기'보다 나은 규칙은 {len(V)}개 중 {len(P['all3_tv_slip'])}개다(미끄러짐 0이면 {len(P['all3_tv'])}개).")
    L.append(
        f"2. **일부 매도는 '덜 들기'의 축소판이다.** 같은 선에서 25%·50%·100%를 팔면 판 값/되산 값(주기 손익)은 같고 효과만 파는 몫에 비례한다. "
        f"고점 −10% 선: 2026 수익 {pct(c10['25']['2026']['포트_수익'])}/{pct(c10['50']['2026']['포트_수익'])}/{pct(c10['100']['2026']['포트_수익'])}, "
        f"7월 쇼크 {pct(c10['25']['AI쇼크']['수익'])}/{pct(c10['50']['AI쇼크']['수익'])}/{pct(c10['100']['AI쇼크']['수익'])}"
        f"(매수·보유 {pct(bh['2026']['포트BH_수익'])}, {pct(bh['AI쇼크']['수익'])}). 깊은 재매수 없이 팔고 회복·20일에 되사는 사다리 {len(sell_only)}개는 "
        f"뒤 구간 타이밍값이 + 인 비율 {pp(so_back)}, 2025 {pp(so_25)}였다.")
    L.append(
        f"3. **남은 작은 값은 '파는 쪽'이 아니라 '더 깊이 되사는 쪽'에서 나왔다.** 조각마다 손절 깊이 +20%p에 지정가로 되사고 못 닿으면 20일 뒤 되사는 열만 "
        f"미끄러짐 0.5%에서도 + 비율이 앞 {pp(sg['+비율']['2026앞'])}·뒤 {pp(sg['+비율']['2026뒤'])}·2025 {pp(sg['+비율']['2025'])}였다"
        f"(중앙 {ppt(sg['2026앞'])}/{ppt(sg['2026뒤'])}/{ppt(sg['2025'])}%p, 사후에 찾은 열이라 관찰용). "
        f"넓게(−20%부터) vs 좁게(−5%부터): 2026 수익 중앙 {pct(s20['2026_포트_수익'])} vs {pct(s5['2026_포트_수익'])}, "
        f"7월 쇼크 {pct(s20['AI쇼크_수익'])} vs {pct(s5['AI쇼크_수익'])}, 주기 {s20['2026_주기_종목당']:.1f} vs {s5['2026_주기_종목당']:.1f}회/종목. "
        f"주기 승률 {pp(s20['2026_승률'])}·{pp(s5['2026_승률'])}, 손익비 {f2(s20['2026_손익비'])}·{f2(s5['2026_손익비'])}로 기대값은 0 근처"
        f"({pct(s20['2026_기대값'])}, {pct(s5['2026_기대값'])}), 앞·뒤·2025 모두 기대값 + 인 규칙 {len(P['all3_exp'])}개.\n")
    return "\n".join(L)


def _context_lines(P):
    out, V, tv, bh = P["results"], P["variants"], P["timing"], P["bh"]
    AB = [k for k in V if V[k]["fam"] in ("A", "B")]
    beat22 = frac(AB, lambda k: out[k]["2022"]["포트_수익"] > bh["2022"]["포트BH_수익"])
    deep_cols = ["g10·recover", "g20·recover", "g10·time", "g20·time", "D40·recover", "D50·recover"]
    sell_cols = ["g-·recover", "g-·time"]
    cm = lambda cols, y: [_col_med(P, "col", c, y, "타이밍값") for c in cols]
    rng = lambda xs: f"{ppt(min(xs), 0)}~{ppt(max(xs), 0)}%p"
    col_line = ("- 재매수 방식별 타이밍값 중앙(열 6개·2개의 범위): 깊은 지정가로 되사는 열은 2020 " + rng(cm(deep_cols, "2020"))
                + ", 2025 " + rng(cm(deep_cols, "2025")) + ", 2021 " + rng(cm(deep_cols, "2021")) + ", 2022 " + rng(cm(deep_cols, "2022"))
                + ". 팔기만 하는 열은 2018 " + rng(cm(sell_cols, "2018")) + "(연말까지 빠지고 끝난 해) 말고는 "
                + rng(sum((cm(sell_cols, y) for y in ("2019", "2020", "2021", "2022", "2023", "2024", "2025")), [])) + ".\n")
    return (col_line + "- 맥락(장세 전환 위험): 같은 평균 노출 대비 타이밍값 + 비율 A " + ", ".join(f"{y} {pp(tv['A'][y]['+비율'])}" for y in CONTEXT_YEARS)
            + "; B " + ", ".join(f"{y} {pp(tv['B'][y]['+비율'])}" for y in CONTEXT_YEARS)
            + ". 깊은 재매수가 통한 해는 V자로 되돌린 해(2020 코로나, 2025 관세)였고, 하락이 이어진 2022에는 A의 "
            + f"{pp(tv['A']['2022']['+비율'])}만 같은 노출 고정을 이겼다(되산 뒤 더 빠짐). 2022에 A·B {pp(beat22)}가 매수·보유보다 덜 잃은 것은 덜 들고 있었기 때문이다.\n"
            + f"- 7월 쇼크 구간만 보면 A {pp(tv['A']['AI쇼크_고정보다덜잃음'])}·B {pp(tv['B']['AI쇼크_고정보다덜잃음'])}가 같은 평균 노출 고정보다 덜 잃었다. "
            + f"그러나 8~10월 반등까지 넣은 뒤 구간 전체로는 A {pp(tv['A']['2026뒤']['+비율'])}·B {pp(tv['B']['2026뒤']['+비율'])}만 나았다. 쇼크에서 판 몫을 반등 앞에서 되사지 못했다.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
    elif len(sys.argv) > 1 and sys.argv[1] == "md":
        write_md(json.loads((S.RESULTS / "p3_LADDER.json").read_text(encoding="utf-8")))
    else:
        main()
