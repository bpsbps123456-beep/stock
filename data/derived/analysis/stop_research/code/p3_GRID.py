"""3단계 갈래 GRID(2026-10-10): 손절 % × 재매수 % × 판매 비율.

사용자 질문: "'쓴다면 넓게'가 무슨 말이야? 기계적 매수처럼 손절도 최대 하락 %를 다양하게 조합해 봐요."
             "한 번에 다 팔지 말고 특정 선을 이탈하면 일부(예: 25%)만 파는 거지."
엔진: tranche_lab.partial(q, x, d) + Spec. AI 종목 전체(stock_lab.universe)를 창 시작 직전 종가에 다 들고 시작한다.
  - 한 종목을 두 조각으로 나눈다: q 조각은 손절·재매수 규칙을 따르고, 1−q 조각은 늘 보유한다.
  - 고점 = 창 시작 직전 종가부터 잰 추적 최고 종가(t−1까지). 손절가 L = 고점×(1−X), 장중(시가가 L 아래면 시가).
  - 깊은 재매수: 고점×(1−D) 지정가(장중, min(시가, R)). 되돌림(fallback): recover = t−1 종가 ≥ 판 가격×1.03이면 t 시가,
    time = 판 날 + 20거래일 시가, none = 없음(깊은 재매수에 못 닿으면 창 끝까지 현금).
평가(2단계 규약 그대로): 2026 전체를 이어 돌려 앞(1/2~5/29, 고르기)·뒤(6/1~10/9, 확인)로 자른다. 2025 보조, 2018~2024 맥락.
  효율선초과 = 같은 최대 낙폭의 고정 노출 대비 초과 수익. 타이밍 값 = 같은 평균 노출 고정 대비 초과 수익(이 갈래에서 새로 냄).
고르기(사전 고정): G1 앞 효율선초과 최대 / G2 앞 주기 기대값 최대(주기 30개 이상, 같으면 앞 효율선초과) / G3 앞 칼마 최대(낙폭 −0.1% 이상만).
실행: PYTHONIOENCODING=utf-8 py p3_GRID.py check   (엔진: 일부 = 전량 규칙과 보유의 섞음인지, 주기 동일성)
      PYTHONIOENCODING=utf-8 py p3_GRID.py         (전체 계산 → results/p3_GRID.json·md)
      PYTHONIOENCODING=utf-8 py p3_GRID.py md      (json에서 보고서만 다시)
"""
from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stock_lab as S  # noqa: E402
import tranche_lab as T  # noqa: E402

ID = "p3_GRID"
RESULTS = HERE.parent / "results"

# ============================================================================ 사전 고정 격자(뒤 구간·2025를 본 뒤 바꾸지 않음)
QS = (0.25, 0.50, 1.00)                                   # 선 하나를 이탈하면 파는 비율(나머지는 늘 보유)
XS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)     # 손절 깊이(고점 대비, 장중)
D_OFFS = (0.05, 0.10, 0.15, 0.20, 0.30, None)             # 재매수 깊이 = X + 이 값(None = 깊은 재매수 없음)
D_CAP = 0.60                                              # 재매수 깊이 상한(X 30% + 30%p = 60%라 실제로 걸리지 않음)
RECOVER_B = 0.03                                          # recover: 종가 ≥ 판 가격 × 1.03 → 다음 시가
TIME_N = 20                                               # time: 판 날 + 20거래일 시가
EXTRA_FALLBACKS = ("time", "none")                        # q = 100% 행만
G2_MIN_CYCLES = 30
CALMAR_MIN_MDD = -0.001                                   # 칼마는 앞 낙폭이 −0.1%보다 깊을 때만(0/0 무한대 방지, LITTEST 교훈)
SLIP_SENS = 0.005
FIXED_REF = (0.75, 0.50)                                  # 참고용 고정 노출
MAIN_RUNS = ("2026", "2025")
CTX_RUNS = ("2024", "2023", "2022", "2021", "2020", "2019", "2018")
DATA_EXCLUDE = {"2018": {"QUBT"}, "2019": {"QUBT"}}       # QUBT 2018~19 장외 자료(하루 +2,033% 등)가 동일가중 매수·보유를 부풀림
REP_COMBOS = ((0.05, 0.05), (0.10, 0.10), (0.15, 0.15), (0.20, 0.20), (0.25, 0.30))   # 일부 vs 전량 표의 대표 X·(D−X)
WORKERS = 10


def grid() -> list[dict]:
    out = []
    for fb in ("recover",) + EXTRA_FALLBACKS:
        for q in (QS if fb == "recover" else (1.0,)):
            for x in XS:
                for off in D_OFFS:
                    d = None if off is None else min(round(x + off, 4), D_CAP)
                    vid = f"{fb}_q{int(q * 100)}_x{int(round(x * 100))}_d{'n' if d is None else int(round(d * 100))}"
                    out.append({"id": vid, "fb": fb, "q": q, "x": x, "off": off, "d": d})
    return out


GRID = grid()
assert len(GRID) == 3 * 8 * 6 + 8 * 6 * 2 == 240, len(GRID)
BY_ID = {v["id"]: v for v in GRID}


def spec_of(v: dict, slip: float = 0.0) -> T.Spec:
    return T.Spec(v["id"], T.partial(v["q"], v["x"], v["d"]), fallback=v["fb"], b=RECOVER_B, n=TIME_N, slip=slip)


def stocks_all() -> list[str]:
    return sorted(s for s in S.load() if s not in S.ETFS)


# ============================================================================ 계산
def merge_phantoms(cycles: list[dict], slip: float) -> list[dict]:
    """엔진의 '되돌림 재매수 → 같은 시가에 바로 손절' 왕복(시가가 이미 손절선 아래라 되산 즉시 다시 판 것)을 앞뒤 주기와 합친다.
    돈은 그대로 현금이라 경제적으로는 한 번의 '밖에 있던 기간'이다. 합친 주기 손익 = 처음 판 가격 / 마지막에 되산 가격 − 1(곱)."""
    by = {}
    for cy in cycles:
        by.setdefault(cy["sym"], []).append(cy)
    out = []
    ratio = (1 - slip) / (1 + slip)
    for sym, cs in by.items():
        cs = sorted(cs, key=lambda c: (c["exit_day"], c["re_day"] or "9999"))
        cur = None
        for cy in cs:
            if (cur is not None and not cur["open"] and cur["kind"] == "되돌림" and cur["re_day"] == cy["exit_day"]
                    and abs(cy["exit_px"] - cur["re_px"] * ratio) <= 1e-9 * cur["re_px"]):
                cur = dict(cur, re_day=cy["re_day"], re_px=cy["re_px"], gain=(1 + cur["gain"]) * (1 + cy["gain"]) - 1,
                           kind=cy["kind"], open=cy["open"], days_out=cur["days_out"] + cy["days_out"], n_seg=cur["n_seg"] + 1)
            else:
                if cur is not None:
                    out.append(cur)
                cur = dict(cy, n_seg=1)
        if cur is not None:
            out.append(cur)
    return out


def cycle_block(cyc: list[dict], gains: np.ndarray, s: dict, pre: str):
    n = len(gains)
    s[pre + "주기"] = n
    if not n:
        for k in ("승률", "평균이익", "평균손실", "손익비", "기대값", "깊은비율", "깊은_평균", "되돌림_평균", "열림_평균"):
            s[pre + k] = float("nan")
        s[pre + "열림_n"] = s[pre + "깊은_n"] = s[pre + "되돌림_n"] = 0
        return
    win, loss = gains[gains > 0], gains[gains <= 0]
    s[pre + "승률"] = float(len(win) / n)
    s[pre + "평균이익"] = float(win.mean()) if len(win) else float("nan")
    s[pre + "평균손실"] = float(loss.mean()) if len(loss) else float("nan")
    s[pre + "손익비"] = float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan")
    s[pre + "기대값"] = float(gains.mean())
    kinds = np.array([cy["kind"] for cy in cyc])
    for kind, key in (("깊은 재매수", "깊은"), ("되돌림", "되돌림"), ("열림", "열림")):
        m = kinds == kind
        s[f"{pre}{key}_n"] = int(m.sum())
        s[f"{pre}{key}_평균"] = float(gains[m].mean()) if m.any() else float("nan")
    s[pre + "깊은비율"] = float((kinds == "깊은 재매수").mean())


def summarize(res: dict, slip: float = 0.0) -> dict:
    data = S.load()
    ws = T.window_stats(res)
    for rname, r in res.items():
        for w in S.SLICES.get(rname, [rname]):
            if w not in ws:
                continue
            d0, d1 = S.WINDOWS[w]
            s = ws[w]
            m = (r["B"].index >= d0) & (r["B"].index <= d1)
            bm = r["B"][m].mean(axis=1).fillna(0).to_numpy()
            e = s["평균노출"]
            fc = np.concatenate([[1.0], np.cumprod(1 + e * bm)])
            s["고정평균노출_수익"] = float(fc[-1] - 1)
            s["고정평균노출_낙폭"] = S._mdd(fc)
            s["노출효과"] = s["고정평균노출_수익"] - s["포트BH_수익"]
            s["타이밍값"] = s["포트_수익"] - s["고정평균노출_수익"]
            s["칼마"] = s["포트_수익"] / abs(s["포트_낙폭"]) if s["포트_낙폭"] < CALMAR_MIN_MDD else float("nan")
            cyc = [cy for cy in r["cycles"] if d0 <= cy["exit_day"] <= d1]
            for kind, key in (("깊은 재매수", "깊은"), ("되돌림", "되돌림"), ("열림", "열림")):
                g = [cy["gain"] for cy in cyc if cy["kind"] == kind]
                s[f"{key}_n"] = len(g)
                s[f"{key}_평균"] = float(np.mean(g)) if g else float("nan")
            nd, nf = s["깊은_n"], s["되돌림_n"]
            s["깊은체결_끝난주기중"] = nd / (nd + nf) if nd + nf else float("nan")
            s["종목당_손절"] = s["손절주기"] / s["종목수"] if s["종목수"] else float("nan")
            s["밖_평균일"] = float(np.mean([cy["days_out"] for cy in cyc])) if cyc else float("nan")
            # 앞 구간에 판 주기 가운데 5/29 뒤에 끝났거나(되산 날) 창 끝까지 열린 것(고르기 지표에 섞인 뒤 정보)
            s["창밖에서_끝난주기"] = int(sum(1 for cy in cyc if cy["open"] or (cy["re_day"] is not None and cy["re_day"] > d1)))
            if cyc:
                g = np.array([cy["gain"] for cy in cyc])
                wt = np.array([cy["w"] for cy in cyc])
                win = g > 0
                s["승률_가중"] = float(wt[win].sum() / wt.sum())
                aw = (g[win] * wt[win]).sum() / wt[win].sum() if win.any() else float("nan")
                al = (g[~win] * wt[~win]).sum() / wt[~win].sum() if (~win).any() else float("nan")
                s["손익비_가중"] = float(aw / -al) if np.isfinite(aw) and np.isfinite(al) and al < 0 else float("nan")
            else:
                s["승률_가중"] = s["손익비_가중"] = float("nan")
            # 합친 주기(같은 시가 왕복을 합침)와, 창 끝(d1) 시점까지의 정보만 쓴 합친 주기 손익(앞 구간 고르기용)
            mc = [cy for cy in merge_phantoms(r["cycles"], slip) if d0 <= cy["exit_day"] <= d1]
            s["같은시가왕복"] = int(sum(cy["n_seg"] - 1 for cy in mc))
            cycle_block(mc, np.array([cy["gain"] for cy in mc]), s, "실_")
            pit, pit_kind = [], []
            for cy in mc:
                if not cy["open"] and cy["re_day"] <= d1:
                    pit.append(cy["gain"])
                    pit_kind.append(cy)
                else:
                    b = data[cy["sym"]]
                    pit.append(cy["exit_px"] / b.c[b.idx(d1, "right") - 1] - 1)
                    pit_kind.append(dict(cy, kind="열림"))
            cycle_block(pit_kind, np.array(pit), s, "실시점_")
            s["실_밖_평균일"] = float(np.mean([cy["days_out"] for cy in mc])) if mc else float("nan")
    return ws


def simulate_np(bars: S.Bars, a: int, z: int, spec: T.Spec):
    """tranche_lab.simulate와 같은 거래 순서·체결. 하나만 다르다: 되돌림 재매수 날 시가가 이미 손절선 아래면 엔진은 되사고
    같은 시가에 바로 다시 판다(현금 그대로, 미끄러짐만 두 번 냄). 여기서는 그 왕복을 거래하지 않고 판 가격·판 날만 그 시가로 옮긴다.
    미끄러짐 0이면 엔진과 가치·노출이 같다(check). 미끄러짐 민감도에서 실제로는 하지 않을 왕복 비용을 빼려고만 쓴다."""
    c, o, l = bars.c, bars.o, bars.l
    tr = spec.tranches
    k = len(tr)
    sh = np.array([t.w / c[a - 1] for t in tr])
    cash = np.zeros(k)
    held = np.ones(k, bool)
    armed = np.array([t.x is not None for t in tr])
    sold_px, sold_t = np.full(k, np.nan), np.full(k, -1)
    ch_px, ch_t = np.full(k, np.nan), np.full(k, -1)          # 주기 시작(실제로 판 가격·날)
    peak = c[a - 1]
    vals, expo, cycles = [1.0], [], []
    sl = spec.slip

    def close_cycle(i, t, px, kind):
        cycles.append({"exit_day": bars.days[ch_t[i]], "exit_px": ch_px[i], "re_day": bars.days[t], "re_px": px,
                       "gain": ch_px[i] / px - 1, "w": tr[i].w, "days_out": t - ch_t[i], "kind": kind, "open": False,
                       "x": tr[i].x, "d": tr[i].d})

    for t in range(a, z + 1):
        for i in range(k):
            if held[i] or spec.fallback == "none":
                continue
            go = (spec.fallback == "recover" and c[t - 1] >= sold_px[i] * (1 + spec.b)) or \
                 (spec.fallback == "time" and t - sold_t[i] >= spec.n)
            if go and t - 1 >= sold_t[i]:
                if tr[i].x is not None and o[t] <= peak * (1 - tr[i].x):
                    sold_px[i], sold_t[i] = o[t] * (1 - sl), t        # 엔진이라면 o(1+sl)에 사서 o(1−sl)에 판다
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
            ch_px[i], ch_t[i] = px, t
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
            cycles.append({"exit_day": bars.days[ch_t[i]], "exit_px": ch_px[i], "re_day": None, "re_px": c[z],
                           "gain": ch_px[i] / c[z] - 1, "w": tr[i].w, "days_out": z - ch_t[i], "kind": "열림",
                           "open": True, "x": tr[i].x, "d": tr[i].d})
    return np.array(vals), np.array(expo), cycles


def run_variant(args):
    vid, slip, ctx = args[:3]
    nophantom = len(args) > 3 and args[3]
    v = BY_ID[vid]
    sp = spec_of(v, slip)
    orig = T.simulate
    if nophantom:
        T.simulate = simulate_np          # 이 작업 프로세스 안에서만, 끝나면 되돌린다(공유 파일은 그대로)
    try:
        out = summarize(T.run_universe(sp, None, {k: S.RUNS[k] for k in MAIN_RUNS}), slip)
        if ctx:
            normal = {k: S.RUNS[k] for k in CTX_RUNS if k not in DATA_EXCLUDE}
            out.update(summarize(T.run_universe(sp, None, normal), slip))
            for k, ex in DATA_EXCLUDE.items():
                syms = [s for s in stocks_all() if s not in ex]
                out.update(summarize(T.run_universe(sp, syms, {k: S.RUNS[k]}), slip))
    finally:
        T.simulate = orig
    return vid, slip, out


def reference() -> dict:
    """매수·보유와 고정 노출 참고값(같은 종목·같은 창)."""
    hold = T.Spec("보유", (T.Tranche(1.0, None, None),))
    res = T.run_universe(hold, None, {k: S.RUNS[k] for k in MAIN_RUNS + tuple(k for k in CTX_RUNS if k not in DATA_EXCLUDE)})
    for k, ex in DATA_EXCLUDE.items():
        res.update(T.run_universe(hold, [s for s in stocks_all() if s not in ex], {k: S.RUNS[k]}))
    out = {}
    for rname, r in res.items():
        for w in S.SLICES.get(rname, [rname]):
            d0, d1 = S.WINDOWS[w]
            m = (r["B"].index >= d0) & (r["B"].index <= d1)
            bm = r["B"][m].mean(axis=1).fillna(0).to_numpy()
            row = {"종목수": int(r["B"].shape[1])}
            for e in (1.0,) + FIXED_REF:
                c = np.concatenate([[1.0], np.cumprod(1 + e * bm)])
                row[f"고정{e:.2f}_수익"], row[f"고정{e:.2f}_낙폭"] = float(c[-1] - 1), S._mdd(c)
            out[w] = row
    return out


def blend_check(n_syms: int = 40) -> dict:
    """일부 매도(q)는 '전량 규칙 q + 매수·보유 1−q'를 창 시작에 나눠 담은 것과 같아야 한다(조각이 같은 고점을 보므로).
    그리고 손절 주기(손익·종류·날짜)는 q와 상관없이 같아야 한다."""
    data = S.load()
    syms = S.universe(252, "2026-01-02")[:n_syms]
    worst_v, worst_c = 0.0, 0.0
    for x, d, fb in ((0.10, 0.20, "recover"), (0.20, None, "recover"), (0.05, 0.35, "recover"), (0.15, 0.30, "time")):
        for q in (0.25, 0.50):
            for s in syms:
                b = data[s]
                a, z = b.idx("2026-01-02"), b.idx("2026-10-09", "right") - 1
                v1, _, c1 = T.simulate(b, a, z, T.Spec("f", T.partial(1.0, x, d), fallback=fb, b=RECOVER_B, n=TIME_N))
                vq, _, cq = T.simulate(b, a, z, T.Spec("p", T.partial(q, x, d), fallback=fb, b=RECOVER_B, n=TIME_N))
                bh = np.concatenate([[1.0], b.c[a:z + 1] / b.c[a - 1]])
                worst_v = max(worst_v, float(np.max(np.abs(vq - (q * v1 + (1 - q) * bh)))))
                if len(c1) != len(cq):
                    worst_c = float("inf")
                else:
                    for u, w in zip(c1, cq):
                        if u["exit_day"] != w["exit_day"] or u["kind"] != w["kind"]:
                            worst_c = float("inf")
                        worst_c = max(worst_c, abs(u["gain"] - w["gain"]))
    # simulate_np(같은 시가 왕복을 거래하지 않음)는 미끄러짐 0에서 엔진과 가치가 같고, 엔진 주기를 합친 것과 주기가 같아야 한다
    worst_np, worst_np_c, n_ph = 0.0, 0.0, 0
    for x, d, fb in ((0.08, None, "time"), (0.10, 0.20, "recover"), (0.25, 0.40, "recover"), (0.05, None, "none")):
        for s in syms:
            b = data[s]
            a, z = b.idx("2026-01-02"), b.idx("2026-10-09", "right") - 1
            sp = T.Spec("e", T.partial(1.0, x, d), fallback=fb, b=RECOVER_B, n=TIME_N)
            v1, e1, c1 = T.simulate(b, a, z, sp)
            v2, e2, c2 = simulate_np(b, a, z, sp)
            worst_np = max(worst_np, float(np.max(np.abs(v1 - v2))), float(np.max(np.abs(e1 - e2))))
            for cy in c1:
                cy["sym"] = s
            m1 = merge_phantoms(c1, 0.0)
            n_ph += len(c1) - len(m1)
            if len(m1) != len(c2):
                worst_np_c = float("inf")
            else:
                for u, w in zip(sorted(m1, key=lambda c: c["exit_day"]), sorted(c2, key=lambda c: c["exit_day"])):
                    worst_np_c = max(worst_np_c, abs(u["gain"] - w["gain"]))
    return {"섞음_최대차": worst_v, "주기손익_최대차": worst_c, "왕복없는시뮬_가치최대차": worst_np,
            "왕복없는시뮬_합친주기최대차": worst_np_c, "점검중_같은시가왕복수": n_ph, "종목수": len(syms)}


# ============================================================================ 고르기
def pick(R: dict) -> dict:
    ids = [v["id"] for v in GRID]
    f = {i: R[i]["2026앞"] for i in ids}
    g1 = max(ids, key=lambda i: f[i]["효율선초과"])
    ok2 = [i for i in ids if f[i]["손절주기"] >= G2_MIN_CYCLES and np.isfinite(f[i]["기대값"])]
    g2 = max(ok2, key=lambda i: (round(f[i]["기대값"], 12), f[i]["효율선초과"]))
    ok3 = [i for i in ids if np.isfinite(f[i]["칼마"])]
    g3 = max(ok3, key=lambda i: f[i]["칼마"])
    # G2′(버그 수정판): 앞 주기 기대값을 5/29 종가까지의 정보로만(뒤에 끝난 주기는 5/29 종가로 평가), 같은 시가 왕복은 합침
    ok2p = [i for i in ids if f[i]["실시점_주기"] >= G2_MIN_CYCLES and np.isfinite(f[i]["실시점_기대값"])]
    g2p = max(ok2p, key=lambda i: (round(f[i]["실시점_기대값"], 12), f[i]["효율선초과"]))
    passset = [i for i in ids if f[i]["효율선초과"] > 0]

    def share(sub, fn):
        return float(np.mean([fn(i) for i in sub])) if sub else float("nan")

    tests = {
        "뒤 효율선초과>0": lambda i: R[i]["2026뒤"]["효율선초과"] > 0,
        "2025 효율선초과>0": lambda i: R[i]["2025"]["효율선초과"] > 0,
        "둘 다>0": lambda i: R[i]["2026뒤"]["효율선초과"] > 0 and R[i]["2025"]["효율선초과"] > 0,
        "뒤 수익>매수·보유": lambda i: R[i]["2026뒤"]["포트_수익"] > R[i]["2026뒤"]["포트BH_수익"],
        "뒤 타이밍값>0": lambda i: R[i]["2026뒤"]["타이밍값"] > 0,
        "2025 타이밍값>0": lambda i: R[i]["2025"]["타이밍값"] > 0,
    }
    ps = {k: {"앞>0 집합": share(passset, fn), "전체": share(ids, fn)} for k, fn in tests.items()}
    by_q = {}
    for q in QS:
        sub = [i for i in ids if BY_ID[i]["q"] == q and BY_ID[i]["fb"] == "recover"]
        sp = [i for i in sub if i in passset]
        by_q[f"{int(q * 100)}"] = {"앞>0": len(sp), "전체": len(sub),
                                    "뒤>0|앞>0": share(sp, tests["뒤 효율선초과>0"]), "뒤>0 전체": share(sub, tests["뒤 효율선초과>0"]),
                                    "2025>0|앞>0": share(sp, tests["2025 효율선초과>0"]), "2025>0 전체": share(sub, tests["2025 효율선초과>0"])}
    return {"G1": g1, "G2": g2, "G2p": g2p, "G3": g3, "passset_n": len(passset), "n": len(ids), "pass": ps, "by_q": by_q}


# ============================================================================ 메인
def compute():
    t0 = time.time()
    chk = blend_check()
    print("섞음 점검", chk, flush=True)
    ref = reference()
    print("참고값", {k: (round(v["고정1.00_수익"], 4), round(v["고정1.00_낙폭"], 4)) for k, v in ref.items()}, flush=True)
    R = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for vid, _, out in ex.map(run_variant, [(v["id"], 0.0, True) for v in GRID], chunksize=4):
            R[vid] = out
    print(f"격자 {len(R)}개 {time.time() - t0:.0f}초", flush=True)
    P = pick(R)
    reps = sorted({P["G1"], P["G2"], P["G2p"], P["G3"]})
    jobs = [(i, SLIP_SENS, True, False) for i in reps] + [(i, SLIP_SENS, True, True) for i in reps]
    slip = {}
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(jobs))) as ex:
        for (vid, _, out), job in zip(ex.map(run_variant, jobs), jobs):
            slip[("왕복없음:" if job[3] else "엔진:") + vid] = out
    print("고르기", P["G1"], P["G2"], P["G2p"], P["G3"], f"{time.time() - t0:.0f}초", flush=True)
    payload = {"id": ID, "made": time.strftime("%Y-%m-%d %H:%M"), "grid": GRID, "grid_n": len(GRID),
               "consts": {"QS": QS, "XS": XS, "D_OFFS": D_OFFS, "D_CAP": D_CAP, "RECOVER_B": RECOVER_B, "TIME_N": TIME_N,
                          "G2_MIN_CYCLES": G2_MIN_CYCLES, "SLIP_SENS": SLIP_SENS, "DATA_EXCLUDE": {k: sorted(v) for k, v in DATA_EXCLUDE.items()}},
               "check": chk, "ref": ref, "res": R, "pick": P, "slip": slip}
    payload["slip_grid"] = slip_grid_run()
    p = S.save(ID, payload)
    print("저장", p, flush=True)
    return payload


def slip_grid_run() -> dict:
    """격자 전체를 미끄러짐 0.5%로 한 번 더(2026·2025만, 같은 시가 왕복은 거래하지 않는 simulate_np). 넓게·좁게 비교의 비용 쪽."""
    t0 = time.time()
    keep = ("포트_수익", "포트_낙폭", "효율선초과", "타이밍값", "평균노출", "실_주기", "실_기대값", "실_승률", "실_손익비")
    out = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for vid, _, res in ex.map(run_variant, [(v["id"], SLIP_SENS, False, True) for v in GRID], chunksize=4):
            out[vid] = {w: {k: res[w][k] for k in keep} for w in res}
    print(f"미끄러짐 격자 {len(out)}개 {time.time() - t0:.0f}초", flush=True)
    return out


def load_payload():
    return json.loads((RESULTS / f"{ID}.json").read_text(encoding="utf-8"))


# ============================================================================ 보고서
def _nan(v):
    return v is None or (isinstance(v, float) and not math.isfinite(v))


def pct(v, d=1, sign=True):
    if _nan(v):
        return "—"
    return (f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%").replace("-", "−")


def pp(v, d=1):
    return "—" if _nan(v) else f"{v * 100:+.{d}f}%p".replace("-", "−")


def f2(v):
    return "—" if _nan(v) else f"{v:.2f}"


def vid(fb, q, x, off):
    d = None if off is None else min(round(x + off, 4), D_CAP)
    return f"{fb}_q{int(q * 100)}_x{int(round(x * 100))}_d{'n' if d is None else int(round(d * 100))}"


FB_TXT = {"recover": f"못 닿으면 종가가 판 가격 +{RECOVER_B:.0%} 위로 돌아온 다음 날 시가에 재매수",
          "time": f"못 닿으면 판 날 +{TIME_N}거래일 시가에 재매수(그 시가가 손절선 아래면 다시 판 것과 같아 계속 현금)",
          "none": "못 닿으면 창 끝까지 현금(되돌림 재매수 없음)"}


def desc(v: dict) -> str:
    q = "전량" if v["q"] == 1 else f"{v['q']:.0%}만"
    d = f"고점 −{v['d']:.0%} 지정가 재매수, " if v["d"] is not None else "깊은 재매수 없음, "
    return f"고점 대비 −{v['x']:.0%} 이탈 시 {q} 매도 → {d}{FB_TXT[v['fb']]}"


def short(v: dict) -> str:
    q = "전량" if v["q"] == 1 else f"{v['q']:.0%}"
    d = f"−{v['d']:.0%}" if v["d"] is not None else "없음"
    fb = {"recover": "되돌림+3%", "time": f"{TIME_N}일 뒤", "none": "되돌림 없음"}[v["fb"]]
    return f"{q} · 손절 −{v['x']:.0%} · 재매수 {d} · {fb}"


def report(P: dict):
    R, ref, pk, SG, SL = P["res"], P["ref"], P["pick"], P["slip_grid"], P["slip"]
    G = {v["id"]: v for v in P["grid"]}
    ids = list(G)
    m = lambda a: float(np.nanmean(a)) if len(a) else float("nan")  # noqa: E731
    OFFS = D_OFFS
    hdr = "| 손절 X \\ 재매수 D | " + " | ".join("없음" if o is None else f"X+{o * 100:.0f}%p" for o in OFFS) + " |"
    sep = "|---|" + "---|" * len(OFFS)
    g1, g2, g2p, g3 = pk["G1"], pk["G2"], pk["G2p"], pk["G3"]

    def xd(fn, fb="recover", q=1.0, bold=()):
        rows = [hdr, sep]
        for x in XS:
            cells = []
            for o in OFFS:
                i = vid(fb, q, x, o)
                t = fn(i)
                cells.append(f"**{t}**" if i in bold else t)
            rows.append(f"| −{x * 100:.0f}% | " + " | ".join(cells) + " |")
        return "\n".join(rows)

    L = []
    A = L.append
    bh = {w: (ref[w]["고정1.00_수익"], ref[w]["고정1.00_낙폭"]) for w in ref}
    s1 = R[g1]
    np1 = SL["왕복없음:" + g1]
    en1 = SL["엔진:" + g1]
    # ---- 집계(본문·결론에 쓰는 숫자)
    xavg = {}
    for x in XS:
        i18 = [vid("recover", q, x, o) for q in QS for o in OFFS]
        i6 = [vid("recover", 1.0, x, o) for o in OFFS]
        xavg[x] = {
            "손절수": m([R[i]["2026"]["실_주기"] / R[i]["2026"]["종목수"] for i in i6]),
            "밖일": m([R[i]["2026"]["실_밖_평균일"] for i in i6]),
            # 앞 구간 손절 1회 성적은 격자 표와 같게 5/29 종가까지의 정보로(실시점_), 뒤·2025는 끝난 값(실_)
            **{f"{w}_{k}": m([R[i][w][("실시점_" if w == "2026앞" else "실_") + k] for i in i6]) for w in ("2026앞", "2026뒤", "2025") for k in ("승률", "손익비", "기대값")},
            **{f"{w}_초과": m([R[i][w]["효율선초과"] for i in i18]) for w in ("2026앞", "2026뒤", "2025", "2026")},
            **{f"{w}_slip초과": m([SG[i][w]["효율선초과"] for i in i18]) for w in ("2026앞", "2026뒤", "2025", "2026")},
            "2022_타이밍": m([R[i]["2022"]["타이밍값"] for i in i18]),
        }
    qavg = {}
    for q in QS:
        i48 = [vid("recover", q, x, o) for x in XS for o in OFFS]
        qavg[q] = {f"{w}_{k}": m([R[i][w][k] for i in i48]) for w in ("2026앞", "2026뒤", "2025", "2026")
                   for k in ("평균노출", "효율선초과", "노출효과", "타이밍값", "포트_수익")}
        for w in ("2026앞", "2026뒤", "2025", "2026"):
            qavg[q][f"{w}_타>0"] = float(np.mean([R[i][w]["타이밍값"] > 0 for i in i48]))
            qavg[q][f"{w}_slip초과"] = m([SG[i][w]["효율선초과"] for i in i48])
            qavg[q][f"{w}_slip타"] = m([SG[i][w]["타이밍값"] for i in i48])
    slope = {}
    for w in ("2026앞", "2026뒤", "2025", "2026"):
        t1 = np.array([R[vid("recover", 1.0, x, o)][w]["타이밍값"] for x in XS for o in OFFS])
        for q in (0.25, 0.5):
            tq = np.array([R[vid("recover", q, x, o)][w]["타이밍값"] for x in XS for o in OFFS])
            slope[(w, q)] = (float((tq * t1).sum() / (t1 * t1).sum()), float(np.mean(np.sign(tq) == np.sign(t1))))
    davg = {}
    for o in OFFS:
        i8 = [vid("recover", 1.0, x, o) for x in XS]
        davg[o] = {f"{w}_{k}": m([R[i][w]["실_" + k] for i in i8]) for w in ("2026앞", "2026뒤", "2025", "2022")
                   for k in ("깊은비율", "깊은_평균", "되돌림_평균", "기대값", "승률", "손익비")}
    P["agg"] = {"xavg": {str(k): v for k, v in xavg.items()}, "qavg": {str(k): v for k, v in qavg.items()},
                "slope": {f"{k[0]}_{k[1]}": v for k, v in slope.items()}, "davg": {str(k): v for k, v in davg.items()}}
    slip_pos = {w: float(np.mean([SG[i][w]["효율선초과"] > 0 for i in ids])) for w in ("2026앞", "2026뒤", "2025", "2026")}
    P["agg"]["slip_pos"] = slip_pos

    # ---- 머리
    A("# GRID: 손절 % × 재매수 % × 판매 비율 (2026-10-10)\n")
    A("> 사용자 질문: \"'쓴다면 넓게'가 무슨 말이야? 기계적 매수처럼 손절도 최대 하락 %를 다양하게 조합해 봐요.\" "
      "\"한 번에 다 팔지 말고, 특정 선을 이탈하면 일부(예: 25%)만 파는 거지.\"")
    A(f"> 코드 `code/p3_GRID.py`(`py p3_GRID.py check`로 엔진 점검), 숫자 `results/p3_GRID.json`. 엔진 `code/tranche_lab.py`(partial·Spec). "
      f"대상: AI 종목 전체, 창 시작 직전 종가에 다 들고 시작(2026 {bh_n(ref, '2026')}종목, 2025 {bh_n(ref, '2025')}종목). 수수료 0, 현금 이자 0, 미끄러짐은 0과 0.5% 두 가지.\n")

    A("## 결론\n")
    A(f"1. **240개 조합에서 2026 앞 구간 1등(전량, 고점 −10% 손절 → −20% 지정가 재매수)은 뒤 구간에서 효율선을 넘지 못했다.** "
      f"뒤 {pct(s1['2026뒤']['포트_수익'])}(매수·보유 {pct(bh['2026뒤'][0])}, 효율선 최선 = 현금 0%), 2025 효율선초과 {pct(s1['2025']['효율선초과'])}였다. "
      f"미끄러짐 0.5%를 넣으면 앞 {pct(np1['2026앞']['효율선초과'])}, 뒤 {pct(np1['2026뒤']['효율선초과'])}, 2025 {pct(np1['2025']['효율선초과'])}로 모두 마이너스가 되고, "
      f"2026 전체는 {pct(s1['2026']['효율선초과'])} → {pct(np1['2026']['효율선초과'])}로 뒤집혔다. "
      f"앞에서 효율선을 이긴 {pk['passset_n']}개가 뒤에서도 이긴 비율은 {pct(pk['pass']['뒤 효율선초과>0']['앞>0 집합'], 0, False)}로, 고르지 않은 전체 {pct(pk['pass']['뒤 효율선초과>0']['전체'], 0, False)}와 같았다.")
    x5, x30 = xavg[0.05], xavg[0.30]
    xs_ = list(XS)
    rng_ = lambda k: (min(xavg[x][k] for x in xs_), max(xavg[x][k] for x in xs_))  # noqa: E731
    A(f"2. **'좁게 vs 넓게'의 차이는 손절 1회 성적보다 거래 횟수에서 크게 났다.** 손절 1회 기대값은 앞 구간에서 모든 X가 마이너스({pct(rng_('2026앞_기대값')[0])}~{pct(rng_('2026앞_기대값')[1])}), "
      f"2025는 0 근처({pct(rng_('2025_기대값')[0])}~{pct(rng_('2025_기대값')[1])})였다. 하락이 이어진 뒤 구간만 좁을수록 조금 나았다(−5% {pct(x5['2026뒤_기대값'])}, −30% {pct(x30['2026뒤_기대값'])}). "
      f"반면 2026 종목당 손절 횟수는 {x5['손절수']:.1f}번 대 {x30['손절수']:.1f}번으로 크게 갈렸다. 그래서 미끄러짐 0.5%를 넣으면 2026 효율선초과 평균이 "
      f"좁게(−5%) {pct(x5['2026_slip초과'])}, 넓게(−30%) {pct(x30['2026_slip초과'])}로 벌어진다. '쓴다면 넓게'는 이 뜻이다.")
    q25 = qavg[0.25]
    A(f"3. **25%만 파는 것은 '전량 규칙 25% + 그냥 보유 75%'와 수학적으로 똑같다(차이 {P['check']['섞음_최대차']:.0e}).** 손절 1회 성적은 전량과 같다. "
      f"선 이탈 타이밍의 값은 전량의 약 1/4로 줄 뿐 부호는 그대로다(기울기 {slope[('2026', 0.25)][0]:.2f}, 부호 일치 {slope[('2026', 0.25)][1]:.0%}). "
      f"25% 매도의 2026 수익이 매수·보유와 다른 몫은 덜 들고 있던 효과(평균 노출 {q25['2026_평균노출']:.2f}, {pp(q25['2026_노출효과'])})가 거의 전부다. "
      f"선 이탈 때 판 타이밍 값은 {pp(q25['2026_타이밍값'])}(미끄러짐 0.5%면 {pp(q25['2026_slip타'])})였다.\n")

    A("## '쓴다면 넓게'가 무슨 뜻이었나\n")
    A("- 2단계 결론의 '쓴다면 넓게'는 손절선을 고점에서 멀리(예: 고점 대비 −20~25%) 두라는 뜻이었다. 고점 −5~8%처럼 가까이 두면 AI 종목의 보통 흔들림(하루 3~5%, 몇 주 사이 10~15% 눌림)에도 손절이 걸린다. 그때마다 팔고 다시 사느라 비용을 내고 반등을 놓친다.")
    A(f"- 이번 격자에서 숫자로 보면 이렇다. 손절 1회의 승률·손익비는 주로 재매수 깊이(D−X)가 정한다. X는 '몇 번 거래하느냐'를 정한다. 비용이 0이면 좁은 손절과 넓은 손절의 우열이 구간마다 바뀐다. 비용이 들어가면 좁은 쪽이 2026 전체에서 일관되게 더 잃는다(아래 '넓게 vs 좁게' 표).\n")

    A("## 시험 범위 (격자는 실행 전 스크립트 맨 위 상수로 고정, 뒤 구간·2025를 본 뒤 바꾸지 않음)\n")
    A("| 묶음 | 개수 | 조합 |")
    A("|---|---|---|")
    A(f"| 판매 비율 × 손절 × 재매수, 되돌림 = 판 가격 +3% 회복 | 144 | q {{25, 50, 100%}} × X {{5, 8, 10, 12, 15, 20, 25, 30%}} × D {{X+5, +10, +15, +20, +30%p, 없음}} |")
    A(f"| 전량(q = 100%) × 되돌림 바꾸기 | 96 | 되돌림 {{판 날 +20거래일, 없음}} × X 8개 × D 6개 |")
    A(f"| 합계 | **240** | 고르기 3종(G1·G2·G3)은 240개 전체에서. 대표 = G1 |")
    A("")
    A("- 간격: X는 5~12%를 촘촘히(2~3%p), 15~30%는 5%p. D는 X에서 5~30%p 더 깊은 곳(60% 상한은 실제로 걸리지 않음).")
    A("- 판매 비율 q: 선 하나를 이탈하면 그 종목의 q만 판다. 나머지 1−q는 끝까지 보유한다. 판 몫은 깊은 지정가나 되돌림으로만 되산다.")
    A(f"- 민감도: 240개 전체를 미끄러짐 0.5%(살 때 +0.5%, 팔 때 −0.5%)로 한 번 더 돌렸다. 고른 규칙 4개는 2018~2024 맥락까지 돌렸다.\n")

    A("## 변수 시점\n")
    A("| 변수 | 언제 아는가 | 체결 |")
    A("|---|---|---|")
    A("| 고점 = 창 시작 직전 종가부터 잰 최고 종가 | t−1 종가 | — |")
    A("| 손절가 L = 고점 × (1 − X) | t−1 종가 | t일 장중 스탑(시가가 L 아래면 시가에 판다) |")
    A("| 깊은 재매수가 R = 고점 × (1 − D) | t−1 종가 | t일 장중 지정가, min(시가, R). 판 가격보다 낮을 때만 |")
    A("| 되돌림(+3%): 종가 ≥ 판 가격 × 1.03 | t−1 종가 | t일 시가 |")
    A("| 되돌림(20일): 판 날 + 20거래일 | 판 날 | 그날 시가(시가가 손절선 아래면 바로 다시 팔림 → 20일 뒤 다시 봄) |")
    A("| 재무장: 깊이 되산 뒤 종가가 손절선 위로 회복 | t 종가 | t+1부터 다시 손절 걸림 |")
    A("| 평균 노출, 효율선 노출, 같은 노출 고정 수익 | 사후 | 평가용. 매매에 쓰지 않음 |")
    A("| 앞 구간 손절 1회 기대값(G2) | 2단계 규약은 5/29 뒤에 끝난 주기를 끝난 값으로 셈 → 미래 정보 | G2′는 5/29 종가까지로 다시 잼(한계 참조) |\n")

    A("## 핵심 격자 표\n")
    A(f"숫자 읽는 법: 행 = 손절 깊이(고점 대비), 열 = 재매수 깊이(손절보다 몇 %p 더 깊은 곳). 예: 행 −10%, 열 X+10%p = 고점 −10%에서 팔고 −20%에 되산다. "
      f"되돌림은 '판 가격 +3% 회복'이다. 굵은 칸 = G1.\n")
    A(f"- 기준: 매수·보유 앞 {pct(bh['2026앞'][0])}/{pct(bh['2026앞'][1])}, 뒤 {pct(bh['2026뒤'][0])}/{pct(bh['2026뒤'][1])}, "
      f"2025 {pct(bh['2025'][0])}/{pct(bh['2025'][1])}, 2026 전체 {pct(bh['2026'][0])}/{pct(bh['2026'][1])} (수익/최대 낙폭).")
    A("- 효율선초과 = 같은 최대 낙폭이 되도록 그냥 덜 들고 있었을 때보다 더 번 수익. 뒤 구간은 매수·보유가 마이너스라 효율선 최선이 현금(0%)이어서 뒤 수익 = 뒤 효율선초과다.\n")
    for q in QS:
        qt = "전량(100%)" if q == 1 else f"{q:.0%}만"
        A(f"### 판매 비율 {qt}\n")
        A(f"**앞 구간(1/2~5/29) 효율선초과** — 이 표로 고른다\n")
        A(xd(lambda i: pct(R[i]["2026앞"]["효율선초과"]), q=q, bold=(g1,)))
        A(f"\n**뒤 구간(6/1~10/9) 수익** — 매수·보유 {pct(bh['2026뒤'][0])}\n")
        A(xd(lambda i: pct(R[i]["2026뒤"]["포트_수익"]), q=q, bold=(g1,)))
        A(f"\n**2025 효율선초과** (보조)\n")
        A(xd(lambda i: pct(R[i]["2025"]["효율선초과"]), q=q, bold=(g1,)))
        A("")
    A("### 손절 1회 성적 (판매 비율과 무관: 25%·50%·100% 모두 같은 값)\n")
    A("손절 1회 = 판 가격 ÷ 다시 산 가격 − 1(+면 싸게 되산 것). 같은 시가에 되사자마자 다시 판 왕복은 한 번으로 합쳤다(한계 참조).\n")
    A("**뒤 구간 손절 1회 기대값**\n")
    A(xd(lambda i: pct(R[i]["2026뒤"]["실_기대값"]), bold=(g1,)))
    A("\n**뒤 구간 승률 / 손익비**\n")
    A(xd(lambda i: f"{R[i]['2026뒤']['실_승률']:.0%} / {f2(R[i]['2026뒤']['실_손익비'])}", bold=(g1,)))
    A("\n**앞 구간 · 2025 손절 1회 기대값** (앞은 5/29 종가까지 정보로 잼)\n")
    A(xd(lambda i: f"{pct(R[i]['2026앞']['실시점_기대값'])} · {pct(R[i]['2025']['실_기대값'])}", bold=(g1,)))
    A("")
    A("### 되돌림 방식을 바꾼 전량 규칙 (칸 = 앞 효율선초과 / 뒤 수익 / 2025 효율선초과)\n")
    A("이 두 표의 굵은 칸은 G3(20거래일 표)와 G2(되돌림 없음 표)다.\n")
    for fb in EXTRA_FALLBACKS:
        A(f"**{ {'time': f'판 날 +{TIME_N}거래일 시가에 되사기', 'none': '되돌림 없음(깊은 재매수만)'}[fb] }**\n")
        A(xd(lambda i: f"{pct(R[i]['2026앞']['효율선초과'])} / {pct(R[i]['2026뒤']['포트_수익'])} / {pct(R[i]['2025']['효율선초과'])}", fb=fb, bold=(g3, g2)))
        A("")
    A(f"- '20거래일 뒤' 되사기는 그날 시가가 아직 손절선 아래면 엔진이 되산 즉시 같은 시가에 다시 판다. 그래서 실제로는 '20일마다 시가가 손절선 위인지 보고 위면 다시 든다'가 되고, 평균 노출이 낮다(−8% 손절은 2026 앞 {f2(R[vid('time', 1.0, 0.08, None)]['2026앞']['평균노출'])}).")
    A("- '되돌림 없음 + 깊은 재매수 없음'(열 '없음')은 한 번 팔면 그해 끝까지 현금이다.\n")
    A("### 미끄러짐 0.5%를 넣은 2026 전체 효율선초과 (전량, 되돌림 +3%)\n")
    A(xd(lambda i: f"{pct(R[i]['2026']['효율선초과'])} → {pct(SG[i]['2026']['효율선초과'])}", bold=(g1,)))
    A(f"\n- 칸 = 비용 0 → 비용 0.5%. 240개 중 비용 0.5%에서 효율선초과가 플러스인 비율: 앞 {slip_pos['2026앞']:.0%}, 뒤 {slip_pos['2026뒤']:.0%}, 2025 {slip_pos['2025']:.0%}, 2026 전체 {slip_pos['2026']:.0%}.\n")

    A("## 넓게 vs 좁게 (손절 깊이 X별 평균, 되돌림 +3%)\n")
    A("손절 1회 성적은 전량 6개(D) 평균, 효율선초과는 18개(D 6 × q 3) 평균이다.\n")
    A("| 손절 X | 2026 종목당 손절 횟수 | 판 뒤 밖에 있던 날(평균) | 앞(5/29까지 정보) 승률 / 손익비 / 기대값 | 뒤 승률 / 손익비 / 기대값 | 2025 승률 / 손익비 / 기대값 | 효율선초과 앞 / 뒤 / 2025 / 2026 | 미끄러짐 0.5%: 2026 효율선초과 | 2022 타이밍 값(맥락) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for x in XS:
        a = xavg[x]
        A(f"| −{x * 100:.0f}% | {a['손절수']:.1f} | {a['밖일']:.0f}일 | {a['2026앞_승률']:.0%} / {f2(a['2026앞_손익비'])} / {pct(a['2026앞_기대값'])} | "
          f"{a['2026뒤_승률']:.0%} / {f2(a['2026뒤_손익비'])} / {pct(a['2026뒤_기대값'])} | {a['2025_승률']:.0%} / {f2(a['2025_손익비'])} / {pct(a['2025_기대값'])} | "
          f"{pct(a['2026앞_초과'])} / {pct(a['2026뒤_초과'])} / {pct(a['2025_초과'])} / {pct(a['2026_초과'])} | {pct(a['2026_slip초과'])} | {pp(a['2022_타이밍'])} |")
    A("")
    lo_x = [x for x in XS if x <= 0.15]
    hi_x = [x for x in XS if x >= 0.20]
    A(f"- 손절 1회 승률·손익비는 X보다 재매수 깊이(D−X)를 따라 움직인다(다음 절). 판 뒤 밖에 있던 날도 X와 관계없이 평균 {xavg[0.05]['밖일']:.0f}일 안팎이다. "
      f"X의 영향은 뒤 구간에서만 보였다(좁을수록 1회 기대값이 조금 높음).")
    A(f"- X가 크게 바꾸는 것은 거래 횟수와 그에 따른 비용이다. 2026년(1/2~10/9)에 −5%는 종목당 {xavg[0.05]['손절수']:.1f}번, −30%는 {xavg[0.30]['손절수']:.1f}번 손절했다. "
      f"미끄러짐 0.5%면 왕복마다 약 1%를 낸다.")
    A(f"- 비용 0에서도 2026 전체 효율선초과는 넓은 쪽(−20~−30%: {pct(min(xavg[x]['2026_초과'] for x in hi_x))}~{pct(max(xavg[x]['2026_초과'] for x in hi_x))})이 "
      f"좁은 쪽(−5~−15%: {pct(min(xavg[x]['2026_초과'] for x in lo_x))}~{pct(max(xavg[x]['2026_초과'] for x in lo_x))})보다 나았다. 앞 구간만 보면 반대로 −10%({pct(xavg[0.10]['2026앞_초과'])})가 가장 좋았다.")
    A("- 2022년처럼 1년 내내 빠진 장에서는 좁은 손절의 타이밍 값이 플러스, 넓은 손절이 마이너스였다(맥락, 마지막 열). 하락이 길게 이어지는 해에만 좁은 손절이 값을 했다.\n")

    A("## 일부 vs 전량 (같은 X·D에서 판매 비율만 바꿈)\n")
    A("타이밍 값 = 규칙 수익 − 같은 평균 노출로 그냥 덜 들고 있었을 때 수익. 노출 효과 = 그 고정 노출 수익 − 매수·보유 수익. 그래서 규칙 수익 − 매수·보유 = 노출 효과 + 타이밍 값.\n")
    A("**48개(X 8 × D 6, 되돌림 +3%) 평균**\n")
    A("| 판매 비율 | 구간 | 평균 노출 | 수익 | 노출 효과 | 타이밍 값 | 타이밍 값 > 0 비율 | 효율선초과 | 미끄러짐 0.5%: 타이밍 값 / 효율선초과 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for q in QS:
        a = qavg[q]
        for w, wn in (("2026앞", "앞"), ("2026뒤", "뒤"), ("2025", "2025"), ("2026", "2026 전체")):
            A(f"| {q:.0%} | {wn} | {a[w + '_평균노출']:.2f} | {pct(a[w + '_포트_수익'])} | {pp(a[w + '_노출효과'])} | {pp(a[w + '_타이밍값'])} | {a[w + '_타>0']:.0%} | "
              f"{pct(a[w + '_효율선초과'])} | {pp(a[w + '_slip타'])} / {pct(a[w + '_slip초과'])} |")
    A("")
    A("**25%·50% 매도의 타이밍 값은 전량의 몇 배인가** (48개, 원점 지나는 기울기 / 부호가 같은 비율)\n")
    A("| 구간 | 25% | 50% |")
    A("|---|---|---|")
    for w, wn in (("2026앞", "앞"), ("2026뒤", "뒤"), ("2025", "2025"), ("2026", "2026 전체")):
        A(f"| {wn} | {slope[(w, 0.25)][0]:.2f} / {slope[(w, 0.25)][1]:.0%} | {slope[(w, 0.5)][0]:.2f} / {slope[(w, 0.5)][1]:.0%} |")
    A("")
    A("**대표 조합에서 판매 비율별로** (2026 전체 · 2025)\n")
    A("| 손절 → 재매수 | 판매 비율 | 2026 수익 / 낙폭 | 노출 | 같은 노출 고정 수익 | 타이밍 값 | 효율선초과 | 2025 타이밍 값 | 2025 효율선초과 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for x, off in REP_COMBOS:
        for q in QS:
            s = R[vid("recover", q, x, off)]
            A(f"| −{x:.0%} → −{x + off:.0%} | {q:.0%} | {pct(s['2026']['포트_수익'])} / {pct(s['2026']['포트_낙폭'])} | {s['2026']['평균노출']:.2f} | "
              f"{pct(s['2026']['고정평균노출_수익'])} | {pp(s['2026']['타이밍값'])} | {pct(s['2026']['효율선초과'])} | {pp(s['2025']['타이밍값'])} | {pct(s['2025']['효율선초과'])} |")
    A(f"| 매수·보유 | — | {pct(bh['2026'][0])} / {pct(bh['2026'][1])} | 1.00 | — | — | 0 | — | 0 |")
    A("")
    A("- 일부 매도에서 판 조각은 전량 규칙과 같은 날 같은 가격에 팔고 되산다(모든 조각이 같은 고점을 본다). 그래서 손절 1회 성적은 판매 비율과 무관하고, 종목 가치는 정확히 'q × 전량 규칙 + (1−q) × 매수·보유'다.")
    A("- 결과: 일부 매도는 전량 매도의 효과(좋든 나쁘든)와 비용을 q만큼 줄인 것이다. 전량에서 마이너스인 선은 25%로 나눠 팔아도 마이너스다. 나눠 판다고 새로운 값이 생기지 않는다.")
    fr = {q: [R[vid("recover", q, x, o)]["2026앞"]["효율선초과"] for x in XS for o in OFFS] for q in QS}
    A(f"- 앞 구간 평균 효율선초과는 25%({pct(qavg[0.25]['2026앞_효율선초과'])})가 전량({pct(qavg[1.0]['2026앞_효율선초과'])})보다 높았다. "
      f"전량은 조합에 따라 {pct(min(fr[1.0]))}~{pct(max(fr[1.0]))}로 크게 퍼지고, 25%는 그 폭이 {pct(min(fr[0.25]))}~{pct(max(fr[0.25]))}로 줄었을 뿐이다. 25%가 선을 더 잘 고른 것이 아니다.")
    tg = lambda q, x, o: R[vid("recover", q, x, o)]["2026"]["타이밍값"]  # noqa: E731
    A(f"- 같은 X·D에서 일부 매도는 타이밍 값의 부호를 바꾸지 못한다. '선 이탈 때 판 것'이 값을 낸 조합은 전량에서 더 크게 냈고"
      f"(−20% → −40%: 2026 타이밍 전량 {pp(tg(1.0, 0.20, 0.20))}, 25% {pp(tg(0.25, 0.20, 0.20))}), "
      f"손해 본 조합은 25%에서도 손해였다(−15% → −30%: 전량 {pp(tg(1.0, 0.15, 0.15))}, 25% {pp(tg(0.25, 0.15, 0.15))}).\n")

    A("## 깊은 재매수는 실제로 얼마나 체결됐나 (전량, 되돌림 +3%, X 8개 평균)\n")
    A("앞 구간 줄은 '앞에 판 주기가 결국 어떻게 끝났나'(5/29 뒤에 끝난 것 포함)를 보인다. 설명용이며 고르기에 쓰지 않는다.\n")
    A("| 재매수 깊이 | 구간 | 깊은 재매수 체결 비율(=승률 근사) | 체결된 손절 1회 평균 이익 | 되돌림으로 되산 손절 1회 평균 손실 | 본전 체결 비율 | 손절 1회 기대값 |")
    A("|---|---|---|---|---|---|---|")
    for o in OFFS[:-1]:
        a = davg[o]
        for w, wn in (("2026앞", "앞"), ("2026뒤", "뒤"), ("2025", "2025")):
            gw, gl = a[w + "_깊은_평균"], a[w + "_되돌림_평균"]
            be = -gl / (gw - gl) if np.isfinite(gw) and np.isfinite(gl) and gw - gl > 0 else float("nan")
            A(f"| X+{o * 100:.0f}%p | {wn} | {a[w + '_깊은비율']:.0%} | {pct(gw)} | {pct(gl)} | {be:.0%} | {pct(a[w + '_기대값'])} |")
    A("")
    A("- 재매수를 깊게 걸수록 체결은 드물어지고(X+5%p 45~53% → X+30%p 6~11%) 체결됐을 때 이익은 커진다(+6% → +56%). 되돌림으로 되산 손절은 깊이와 관계없이 평균 −6% 안팎을 잃는다.")
    wr = [davg[o][w + "_승률"] for o in (0.15, 0.20) for w in ("2026앞", "2026뒤", "2025")]
    po = [davg[o][w + "_손익비"] for o in (0.15, 0.20) for w in ("2026앞", "2026뒤", "2025")]
    A(f"- 그래서 '승률 30%라도 손익비로 이긴다'는 모양은 실제로 나온다(X+15~20%p: 승률 {min(wr):.0%}~{max(wr):.0%}, 손익비 {min(po):.1f}~{max(po):.1f}). "
      "그런데 실제 체결 비율이 본전 체결 비율에 거의 붙어 있어 기대값이 0 근처다. 앞 구간(V자 반등)은 다섯 줄 모두 본전 아래, 뒤 구간(하락)은 모두 위, 2025는 본전과 거의 같았다.")
    A("- 손익비는 재매수를 깊게 걸면 얼마든지 키울 수 있다. 대신 그만큼 승률이 내려가 기대값은 그대로다. 손익비만 보고 고르면 안 되는 이유다.\n")

    A("## 고르기 (2026 앞에서 고르고 뒤에서 확인, 2025 보조)\n")
    A("| 고르기 | 규칙 | 앞 수익 / 낙폭 / 효율선초과 | 뒤 수익 / 낙폭 | 2025 수익 / 낙폭 / 효율선초과 | 2026 수익 / 낙폭 / 효율선초과 | 2026 평균 노출 / 타이밍 값 | 수익 나은 종목(2026) |")
    A("|---|---|---|---|---|---|---|---|")

    def prow(name, s, rule):
        return (f"| {name} | {rule} | {pct(s['2026앞']['포트_수익'])} / {pct(s['2026앞']['포트_낙폭'])} / {pct(s['2026앞']['효율선초과'])} | "
                f"{pct(s['2026뒤']['포트_수익'])} / {pct(s['2026뒤']['포트_낙폭'])} | {pct(s['2025']['포트_수익'])} / {pct(s['2025']['포트_낙폭'])} / {pct(s['2025']['효율선초과'])} | "
                f"{pct(s['2026']['포트_수익'])} / {pct(s['2026']['포트_낙폭'])} / {pct(s['2026']['효율선초과'])} | {s['2026']['평균노출']:.2f} / {pp(s['2026']['타이밍값'])} | "
                f"{s['2026'].get('수익_나은종목비율', float('nan')):.0%} |")
    A(prow("**G1 = 대표** (앞 효율선초과 1위)", R[g1], short(G[g1])))
    A(prow("G1, 미끄러짐 0.5%", np1, "같은 규칙(같은 시가 왕복은 거래 안 함)"))
    A(prow("G1, 미끄러짐 0.5%(엔진 그대로)", en1, "같은 규칙(같은 시가 왕복 비용까지)"))
    A(prow("G2 (앞 손절 1회 기대값 1위, 2단계 규약)", R[g2], short(G[g2])))
    A(prow("G2′ (같은 지표를 5/29까지 정보로)", R[g2p], short(G[g2p])))
    A(prow("G3 (앞 칼마 1위)", R[g3], short(G[g3])))
    for e in (1.0,) + FIXED_REF:
        k = f"고정{e:.2f}"
        A(f"| {'매수·보유' if e == 1 else f'고정 노출 {e:.2f}'} | — | {pct(ref['2026앞'][k + '_수익'])} / {pct(ref['2026앞'][k + '_낙폭'])} / 0 | "
          f"{pct(ref['2026뒤'][k + '_수익'])} / {pct(ref['2026뒤'][k + '_낙폭'])} | {pct(ref['2025'][k + '_수익'])} / {pct(ref['2025'][k + '_낙폭'])} / 0 | "
          f"{pct(ref['2026'][k + '_수익'])} / {pct(ref['2026'][k + '_낙폭'])} / 0 | {e:.2f} / 0 | — |")
    A("")
    A(f"- G1 규칙: {desc(G[g1])}. 앞 구간 1등은 240개 중 뒤 효율선초과 {rank(R, ids, g1, '2026뒤')}위, 2025 {rank(R, ids, g1, '2025')}위, 2026 전체 {rank(R, ids, g1, '2026')}위였다.")
    A(f"- G1의 2026 종목당 손절은 {R[g1]['2026']['실_주기'] / R[g1]['2026']['종목수']:.1f}번이다. 미끄러짐 0.5%면 앞 효율선초과 {pct(R[g1]['2026앞']['효율선초과'])} → {pct(np1['2026앞']['효율선초과'])}, "
      f"2025 {pct(R[g1]['2025']['효율선초과'])} → {pct(np1['2025']['효율선초과'])}가 된다.")
    A(f"- G2는 '되돌림 없음, −30% 손절, 깊은 재매수 없음'(한 번 팔면 그해 현금)이 뽑혔다. 앞 기대값 {pct(R[g2]['2026앞']['기대값'])}는 앞에 판 종목을 10/9 종가로 잰 값(미래 정보)이다. 5/29 종가로 다시 재면 {pct(R[g2]['2026앞']['실시점_기대값'])}다. 그래서 같은 지표를 5/29까지로 잰 G2′도 함께 적었다.")
    A(f"- G3는 노출이 낮아(2026 {R[g3]['2026']['평균노출']:.2f}) 낙폭이 작은 규칙이 뽑혔다. 사실상 '거의 현금'이다.\n")
    A("**앞 구간 고르기에 정보가 있었나** (240개)\n")
    A("| 확인 | 앞 효율선초과 > 0 집합 | 전체(무조건) |")
    A("|---|---|---|")
    A(f"| 개수 | {pk['passset_n']} | {pk['n']} |")
    for k, v in pk["pass"].items():
        A(f"| {k} | {v['앞>0 집합']:.0%} | {v['전체']:.0%} |")
    A("")
    A("| 판매 비율(되돌림 +3%) | 앞 > 0 개수 / 48 | 뒤 > 0: 앞 > 0 집합 / 전체 | 2025 > 0: 앞 > 0 집합 / 전체 |")
    A("|---|---|---|---|")
    for q in QS:
        b = pk["by_q"][str(int(q * 100))]
        A(f"| {q:.0%} | {b['앞>0']} | {pct(b['뒤>0|앞>0'], 0, False)} / {pct(b['뒤>0 전체'], 0, False)} | {pct(b['2025>0|앞>0'], 0, False)} / {pct(b['2025>0 전체'], 0, False)} |")
    A("")
    A("- 25%·50% 매도는 뒤 구간에서 96개 모두 효율선을 넘지 못했다. 뒤 구간 효율선 최선이 현금(0%)인데 노출이 0.85~0.92로 남아 매수·보유의 −4%를 거의 그대로 받았기 때문이다.\n")

    A("## 손절 1회 승률·손익비 (고른 규칙)\n")
    A("조각이 하나라 조각 수 기준 = 비중 가중이다. 괄호는 엔진 그대로(같은 시가 왕복을 따로 센) 값이다.\n")
    A("| 규칙 | 구간 | 손절 횟수 | 승률 | 평균 이익 / 평균 손실 | 손익비 | 기대값 | 깊은 재매수 비율 |")
    A("|---|---|---|---|---|---|---|---|")
    for name, i in (("G1", g1), ("G2′", g2p), ("G3", g3)):
        for w, wn in (("2026앞", "앞(5/29까지)"), ("2026뒤", "뒤"), ("2025", "2025"), ("2022", "2022(맥락)")):
            s = R[i][w]
            pre = "실시점_" if w == "2026앞" else "실_"
            A(f"| {name} | {wn} | {s[pre + '주기']} ({s['손절주기']}) | {s[pre + '승률']:.0%} ({s['승률']:.0%}) | {pct(s[pre + '평균이익'])} / {pct(s[pre + '평균손실'])} | "
              f"{f2(s[pre + '손익비'])} ({f2(s['손익비'])}) | {pct(s[pre + '기대값'])} ({pct(s['기대값'])}) | {s[pre + '깊은비율']:.0%} |")
    A("")
    A(f"- G1(−10% → −20%)은 승률 30% 안팎, 손익비 2 안팎이다. 본전 승률이 약 1/(1+2) = 33%라 기대값이 0 근처에서 구간마다 부호가 바뀐다(앞 {pct(R[g1]['2026앞']['실시점_기대값'])}, 뒤 {pct(R[g1]['2026뒤']['실_기대값'])}, 2025 {pct(R[g1]['2025']['실_기대값'])}).\n")

    A("## 보조·맥락\n")
    ctx = []
    for y in ("2018", "2019", "2020", "2021", "2022", "2023", "2024"):
        s = R[g1][y]
        ctx.append(f"{y} {pct(s['포트_수익'])}(매수·보유 {pct(ref[y]['고정1.00_수익'])}, 타이밍 {pp(s['타이밍값'])})")
    A(f"- G1 맥락(2018~2019는 QUBT 제외): " + ", ".join(ctx) + ".")
    tpos = {y: float(np.mean([R[i][y]["타이밍값"] > 0 for i in ids])) for y in ("2018", "2019", "2020", "2021", "2022", "2023", "2024")}
    A(f"- 240개 중 타이밍 값이 플러스인 비율은 해마다 {min(tpos.values()):.0%}~{max(tpos.values()):.0%}로 흔들렸다(2021 {tpos['2021']:.0%}, 2022 {tpos['2022']:.0%}). 같은 규칙이 장세에 따라 뒤집힌다(장세 전환 위험).")
    A(f"- 2025 보조: 앞 > 0 집합의 2025 성공 {pk['pass']['2025 효율선초과>0']['앞>0 집합']:.0%}는 전체 {pk['pass']['2025 효율선초과>0']['전체']:.0%}보다 조금 높았다. 2025는 대부분이 플러스라 확인에 쓸 정보가 적다.\n")

    A("## 선택편향·한계\n")
    A(f"- 240개에서 앞 구간 1등을 골랐다. 1등의 앞 효율선초과 {pct(R[g1]['2026앞']['효율선초과'])}에는 선택편향이 들어 있다. 실제로 1등은 뒤 구간에서 240개 중 {rank(R, ids, g1, '2026뒤')}위였다.")
    best26 = sorted([i for i in ids if G[i]["fb"] == "recover"], key=lambda i: -R[i]["2026"]["효율선초과"])[0]
    A(f"- 사후에 보면 2026 전체·2025가 모두 좋은 조합이 있다(예: {short(G[best26])}: 2026 효율선초과 {pct(R[best26]['2026']['효율선초과'])}, "
      f"2025 {pct(R[best26]['2025']['효율선초과'])}, 미끄러짐 0.5%면 2026 {pct(SG[best26]['2026']['효율선초과'])}). 그러나 앞 구간에서는 {pct(R[best26]['2026앞']['효율선초과'])}여서 미리 고를 방법이 없었다. 근거로 쓰지 않는다.")
    A(f"- 버그 수정 1(첫 실행 뒤, 격자 그대로): 2단계 규약의 '앞 구간 손절 1회 기대값'은 5/29에 아직 안 끝난 주기를 10/9 값으로 센다. 그래서 G2가 미래 정보로 뽑혔다. "
      f"5/29 종가로 다시 잰 G2′를 함께 적었다. 대표(G1)는 이 지표를 쓰지 않아 영향이 없다.")
    A(f"- 버그 수정 2(측정): 엔진은 되돌림 재매수 날 시가가 이미 손절선 아래면 되사고 같은 시가에 바로 다시 판다. 돈은 현금 그대로라 수익·노출은 맞다(점검 차 {P['check']['왕복없는시뮬_가치최대차']:.0e}). "
      f"다만 손절 횟수가 부풀고 미끄러짐을 넣으면 실제로는 하지 않을 왕복 비용이 붙는다. 그래서 손절 1회 성적은 그 왕복을 합쳐 셌다. 미끄러짐 민감도는 그 왕복을 거래하지 않는 같은 엔진 사본(`simulate_np`, 비용 0에서 엔진과 차이 0)으로 쟀다. "
      f"되돌림 +3% 규칙은 이 차이가 작다(G1 2026 손절 {R[g1]['2026']['손절주기']}번 → {R[g1]['2026']['실_주기']}번). '20거래일 뒤' 규칙은 크다(G3 {R[g3]['2026']['손절주기']}번 → {R[g3]['2026']['실_주기']}번).")
    A("- 보고서 수정(숫자 재계산 없음): '넓게 vs 좁게' 표의 앞 구간 승률·손익비·기대값을 격자 표와 같은 5/29 종가 기준으로 바꿨다(처음 판은 5/29 뒤에 끝난 주기를 끝난 값으로 셌다). 앞 기대값이 X별 −2.2%~−1.5%에서 −2.3%~−1.8%로 바뀌었고 결론은 같다.")
    A("- 2026 뒤 구간은 매수·보유가 −4.1%여서 효율선 최선이 현금이다. 뒤 효율선초과는 규칙의 수익과 같다. 노출을 남기는 25%·50% 매도는 구조적으로 불리하다.")
    A("- 생존편향: 종목 목록이 2026-10 AI_UNIVERSE와 TIME 보유 이력이다. 상장폐지 종목이 없다. 깊이 빠진 뒤 되사는 규칙은 실제보다 좋게 나오고, 손절이 막아 줬을 큰 손실은 빠져 있다.")
    A("- 고점은 창 시작 직전 종가부터 잰다(252일 고점이 아님). 그래서 해마다 첫날 손절선이 '직전 종가 −X%'에서 다시 시작한다.")
    A("- QUBT 2018~2019 장외 자료(하루 +2,033% 등)는 2018·2019 맥락에서 뺐다. 수수료·현금 이자 0이다. 미끄러짐은 0과 0.5%만 봤다.")
    A("- 같은 날 손절 뒤 더 빠져 재매수가에 닿으면 그날 되산다고 봤다(위에서 아래로 차례로 지나갔다고 가정). 일봉으로는 순서를 알 수 없다.\n")

    A("## 다시 돌리기\n")
    A("```bash")
    A("cd data/derived/analysis/stop_research/code")
    A("PYTHONIOENCODING=utf-8 py p3_GRID.py check   # 일부 = 섞음, 주기 동일, 왕복 없는 사본 = 엔진")
    A("PYTHONIOENCODING=utf-8 py p3_GRID.py         # 240개 + 맥락 + 미끄러짐(약 5분, 10프로세스)")
    A("PYTHONIOENCODING=utf-8 py p3_GRID.py md      # json에서 보고서만")
    A("```")
    (RESULTS / f"{ID}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    S.save(ID, P)
    print("보고서", RESULTS / f"{ID}.md")


def bh_n(ref, w):
    return ref[w]["종목수"]


def rank(R, ids, i, w):
    order = sorted(ids, key=lambda j: -R[j][w]["효율선초과"])
    return order.index(i) + 1


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "check":
        print(blend_check())
    elif arg == "slipgrid":                 # 이미 있는 json에 미끄러짐 격자만 더한다
        pl = load_payload()
        pl["slip_grid"] = slip_grid_run()
        S.save(ID, pl)
    elif arg == "md":
        report(load_payload())
    else:
        report(compute())
