"""3단계 엔진(2026-10-10): 분할 손절 × 분할 재매수. 한 종목을 조각(트랜치)으로 나눠 조각마다 따로 팔고 따로 되산다.

사용자 요청: "손절도 최대 하락 %를 다양하게 조합", "한 번에 다 팔지 말고 선을 이탈하면 25%처럼 일부만".

기준선: 고점 = 창 시작 직전 종가부터 잰 최고 종가(추적 고점, t-1까지). 모든 조각이 같은 고점을 본다.
조각 i: 비중 w_i, 손절 깊이 X_i(고점 대비), 재매수 깊이 D_i(> X_i, None이면 깊은 재매수 없음).
하루 순서(t일)
  1) 시가: t-1 종가로 정한 '되돌림 재매수'(fallback)를 체결한다.
       recover : 판 조각은 t-1 종가 ≥ 판 가격 × (1+b)이면 t 시가에 되산다(깊이 안 빠지고 돌아온 경우).
       time    : 판 날 + N거래일 시가에 되산다.
       none    : 되돌림 재매수 없음(깊은 재매수만).
  2) 장중 손절: 들고 있고 '무장된' 조각의 손절가 L = 고점[t-1] × (1-X_i). 시가 ≤ L이면 시가, 저가 ≤ L이면 L에 판다.
  3) 장중 깊은 재매수: 판 조각의 지정가 R = 고점[t-1] × (1-D_i). R < 판 가격이고 저가 ≤ R이면 min(시가, R)에 산다.
     같은 날 손절 뒤 더 빠져 R에 닿으면 그날 되살 수 있다(위에서 아래로 차례로 지나간 것으로 본다).
  4) 종가: 고점 갱신. 깊이 되산 조각은 종가 ≥ 고점 × (1-X_i)가 될 때까지 무장 해제(이미 손절선 아래라 바로 다시 팔리는 것을 막음).
     되돌림으로 되산 조각은 바로 무장(다시 빠지면 또 판다).
돈: 조각마다 자기 주식·현금 통을 따로 둔다. 되살 때는 그 조각이 판 돈 전부로 산다. 그래서 주기 손익 = 판 가격/되산 가격 − 1이 정확하다.
주기: 조각 하나의 '판 → 되산' 한 번. 창 끝까지 못 되샀으면 판 가격/창 끝 종가 − 1. 승률·손익비·기대값은 조각 수 기준과 비중 가중을 함께 낸다.
시점: 손절가·재매수가·되돌림 조건은 모두 t-1 종가까지의 고점·가격으로 정한다(stock_lab과 같은 규약).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import stock_lab as S


@dataclass(frozen=True)
class Tranche:
    w: float               # 비중(합 1)
    x: float | None        # 손절 깊이(0.10 = 고점 대비 −10%). None이면 이 조각은 늘 보유
    d: float | None        # 깊은 재매수 깊이(> x). None이면 없음


@dataclass(frozen=True)
class Spec:
    name: str
    tranches: tuple
    fallback: str = "recover"     # recover | time | none
    b: float = 0.03               # recover 여유(판 가격 × (1+b) 이상 종가)
    n: int = 20                   # time: 판 날 + n거래일
    slip: float = 0.0             # 체결 미끄러짐(판 때 −, 살 때 +)


def simulate(bars: S.Bars, a: int, z: int, spec: Spec):
    c, o, h, l = bars.c, bars.o, bars.h, bars.l
    tr = spec.tranches
    k = len(tr)
    sh = np.array([t.w / c[a - 1] for t in tr])           # 조각별 주식 수
    cash = np.zeros(k)                                    # 조각별 현금
    held = np.ones(k, bool)
    armed = np.array([t.x is not None for t in tr])
    sold_px = np.full(k, np.nan)
    sold_t = np.full(k, -1)
    peak = c[a - 1]
    vals, expo, cycles = [1.0], [], []
    sl = spec.slip

    def close_cycle(i, t, px, kind):
        cycles.append({"exit_day": bars.days[sold_t[i]], "exit_px": sold_px[i], "re_day": bars.days[t], "re_px": px,
                       "gain": sold_px[i] / px - 1, "w": tr[i].w, "days_out": t - sold_t[i], "kind": kind, "open": False,
                       "x": tr[i].x, "d": tr[i].d})

    for t in range(a, z + 1):
        # 1) 시가: 되돌림 재매수(t-1 종가 기준)
        for i in range(k):
            if held[i] or spec.fallback == "none":
                continue
            go = (spec.fallback == "recover" and c[t - 1] >= sold_px[i] * (1 + spec.b)) or \
                 (spec.fallback == "time" and t - sold_t[i] >= spec.n)
            if go and t - 1 >= sold_t[i]:
                px = o[t] * (1 + sl)
                close_cycle(i, t, px, "되돌림")
                sh[i], cash[i], held[i], armed[i] = cash[i] / px, 0.0, True, True
        # 2) 장중 손절
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
        # 3) 장중 깊은 재매수
        for i in range(k):
            if held[i] or tr[i].d is None:
                continue
            R = peak * (1 - tr[i].d)
            if R < sold_px[i] and l[t] <= R:
                px = min(o[t], R) * (1 + sl)
                close_cycle(i, t, px, "깊은 재매수")
                sh[i], cash[i], held[i], armed[i] = cash[i] / px, 0.0, True, False
        # 4) 종가
        eq = float(np.sum(sh * c[t]) + np.sum(cash))
        vals.append(eq)
        expo.append(float(np.sum(sh * c[t])) / eq if eq > 0 else 0.0)
        peak = max(peak, c[t])
        for i in range(k):
            if held[i] and not armed[i] and tr[i].x is not None and c[t] >= peak * (1 - tr[i].x):
                armed[i] = True
    for i in range(k):
        if not held[i]:
            cycles.append({"exit_day": bars.days[sold_t[i]], "exit_px": sold_px[i], "re_day": None, "re_px": c[z],
                           "gain": sold_px[i] / c[z] - 1, "w": tr[i].w, "days_out": z - sold_t[i], "kind": "열림",
                           "open": True, "x": tr[i].x, "d": tr[i].d})
    return np.array(vals), np.array(expo), cycles


def run_universe(spec: Spec, syms=None, runs=None):
    data = S.load()
    runs = runs or S.RUNS
    out = {}
    for rname, (d0, d1) in runs.items():
        cyc, R, B, E = [], {}, {}, {}
        for s in (syms or S.universe(252, d0)):
            b = data[s]
            a, z = b.idx(d0), b.idx(d1, "right") - 1
            if a < 252 or z <= a:
                continue
            v, e, cycles = simulate(b, a, z, spec)
            bh = np.concatenate([[1.0], b.c[a:z + 1] / b.c[a - 1]])
            dd = b.days[a:z + 1]
            R[s] = pd.Series(v[1:] / v[:-1] - 1, index=dd)
            B[s] = pd.Series(bh[1:] / bh[:-1] - 1, index=dd)
            E[s] = pd.Series(e, index=dd)
            for cy in cycles:
                cy["sym"] = s
            cyc += cycles
        out[rname] = {"R": pd.DataFrame(R).sort_index(), "B": pd.DataFrame(B).sort_index(),
                      "E": pd.DataFrame(E).sort_index(), "cycles": cyc}
    return out


def window_stats(res: dict) -> dict:
    """stock_lab.window_stats와 같은 칸 + 비중 가중 주기 기대값·깊은 재매수 비율."""
    out = {}
    for rname, r in res.items():
        for w in S.SLICES.get(rname, [rname]):
            d0, d1 = S.WINDOWS[w]
            Rw = r["R"][(r["R"].index >= d0) & (r["R"].index <= d1)]
            Bw = r["B"][(r["B"].index >= d0) & (r["B"].index <= d1)]
            Ew = r["E"][(r["E"].index >= d0) & (r["E"].index <= d1)]
            if not len(Rw):
                continue
            pr = np.concatenate([[1.0], (1 + Rw.mean(axis=1).fillna(0)).cumprod().to_numpy()])
            pb = np.concatenate([[1.0], (1 + Bw.mean(axis=1).fillna(0)).cumprod().to_numpy()])
            exc, e_eq = S.same_mdd_excess(pr, Bw.mean(axis=1).fillna(0).to_numpy())
            cyc = [cy for cy in r["cycles"] if d0 <= cy["exit_day"] <= d1]
            g = np.array([cy["gain"] for cy in cyc]) if cyc else np.array([])
            wt = np.array([cy["w"] for cy in cyc]) if cyc else np.array([])
            win, loss = g[g > 0], g[g <= 0]
            sr = (1 + Rw.fillna(0)).prod() - 1
            sb = (1 + Bw.fillna(0)).prod() - 1
            out[w] = {
                "종목수": int(Rw.shape[1]), "포트_수익": float(pr[-1] - 1), "포트_낙폭": S._mdd(pr),
                "포트BH_수익": float(pb[-1] - 1), "포트BH_낙폭": S._mdd(pb), "효율선초과": exc, "효율선_e": e_eq,
                "평균노출": float(Ew.mean().mean()), "수익_나은종목비율": float((sr > sb + 1e-12).mean()),
                "손절주기": int(len(g)), "승률": float(len(win) / len(g)) if len(g) else float("nan"),
                "평균이익": float(win.mean()) if len(win) else float("nan"),
                "평균손실": float(loss.mean()) if len(loss) else float("nan"),
                "손익비": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
                "기대값": float(g.mean()) if len(g) else float("nan"),
                "기대값_가중": float((g * wt).sum() / wt.sum()) if len(g) else float("nan"),
                "깊은재매수비율": float(np.mean([cy["kind"] == "깊은 재매수" for cy in cyc])) if cyc else float("nan"),
            }
    return out


def fmt(ws, keys=("2026앞", "2026뒤", "2025", "2022")):
    p = []
    for k in keys:
        if k in ws:
            s = ws[k]
            p.append(f"{k} {s['포트_수익']:+.1%}/{s['포트_낙폭']:.1%}(BH {s['포트BH_수익']:+.1%}/{s['포트BH_낙폭']:.1%}) "
                     f"초과 {s['효율선초과']:+.1%} 노출 {s['평균노출']:.2f} 승률 {s['승률']:.0%} 손익비 {s['손익비']:.2f} "
                     f"기대 {s['기대값']:+.1%} 깊은 {s['깊은재매수비율']:.0%} n{s['손절주기']}")
    return " | ".join(p)


def ladder(xs, ds=None, w=None):
    """사다리: 깊이 xs마다 같은 비중(또는 w) 조각. ds는 조각별 재매수 깊이(같은 길이) 또는 None."""
    n = len(xs)
    w = w or [1.0 / n] * n
    ds = ds or [None] * n
    return tuple(Tranche(wi, x, d) for wi, x, d in zip(w, xs, ds))


def partial(q, x, d):
    """한 선에서 q만 판다(나머지 1−q는 늘 보유)."""
    out = [Tranche(q, x, d)]
    if q < 1:
        out.append(Tranche(1 - q, None, None))
    return tuple(out)


def check():
    data = S.load()
    syms = S.universe(252, "2026-01-02")
    # 1) 손절 없는 조각 = 매수·보유
    hold = Spec("보유", (Tranche(1.0, None, None),))
    r = run_universe(hold, syms[:30], {"2026": S.RUNS["2026"]})
    d = float((r["2026"]["R"] - r["2026"]["B"]).abs().max().max())
    print("손절 없는 조각 vs 매수·보유 최대 차(0이어야 함)", d)
    # 2) 조각 4개로 나눠도 손절이 같으면 한 조각과 같아야 함
    one = Spec("한 조각", partial(1.0, 0.15, 0.30))
    four = Spec("네 조각 같은 선", ladder([0.15] * 4, [0.30] * 4))
    r1 = run_universe(one, syms[:30], {"2026": S.RUNS["2026"]})["2026"]["R"]
    r4 = run_universe(four, syms[:30], {"2026": S.RUNS["2026"]})["2026"]["R"]
    print("같은 선 1조각 vs 4조각 최대 차(0이어야 함)", float((r1 - r4).abs().max().max()))
    # 3) 미래 교란
    b = data[syms[3]]
    a, z = b.idx("2026-01-02"), b.idx("2026-10-09", "right") - 1
    sp = Spec("사다리", ladder([0.10, 0.20, 0.30, 0.40], [0.20, 0.30, 0.40, 0.50]))
    v1, _, _ = simulate(b, a, z, sp)
    T = a + 80
    b2 = S.Bars(b.sym, b.days, b.o.copy(), b.h.copy(), b.l.copy(), b.c.copy(), b.v)
    rng = np.random.default_rng(1)
    for arr in (b2.o, b2.h, b2.l, b2.c):
        arr[T + 1:] *= 1 + rng.normal(0, 0.05, len(arr) - T - 1)
    b2.h[T + 1:] = np.maximum.reduce([b2.h[T + 1:], b2.o[T + 1:], b2.c[T + 1:]])
    b2.l[T + 1:] = np.minimum.reduce([b2.l[T + 1:], b2.o[T + 1:], b2.c[T + 1:]])
    v2, _, _ = simulate(b2, a, z, sp)
    print("미래 교란 뒤 t까지 가치 최대 차(0이어야 함)", float(np.max(np.abs(v1[:T - a + 2] - v2[:T - a + 2]))))
    # 4) 예시
    for spc in (Spec("전량 −10%→−20% 재매수", partial(1.0, 0.10, 0.20)),
                Spec("25%만 −10%→−20%", partial(0.25, 0.10, 0.20)),
                Spec("사다리 −10/−20/−30/−40 각 25%, 재매수 +10%p", ladder([0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]))):
        ws = window_stats(run_universe(spc, syms, {"2026": S.RUNS["2026"], "2025": S.RUNS["2025"]}))
        print(spc.name, "::", fmt(ws))


if __name__ == "__main__":
    check()
