"""deskx_expand 결과의 견고성·쏠림 감사(2026-10-10, 연구용 스크래치, 적대적 검토).

렌즈: (1) 앞 기준 1위(T/off/m=1)와 이웃(m 축 T/off/m=2, 대리 축 T/on/m=1, 후보군 축 TK/off/m=1)을 다시 돌려
      LIVE(T/off/m=none = 데스크 반도체 50%)와 월별로 비교.
      (2) 수익이 종목 1~2개·날짜 1~2일에 기대는가(종목 기여 분해, 1위 종목을 현금으로 바꾼 회계식 제거,
      후보에서 빼고 다시 돌리는 구조적 제거, 상위 2일 제거, LIVE 대비 차이를 만든 날·종목).
      (3) m 한도가 실제로 분산시키는가(세부 섹터 수, 최대 세부 섹터 비중, 투자 비중)와 MDD를 낮추는가
          (현금 비중 차이를 뺀 '투자분 수익'으로도 비교).
      (4) 한 축만 바꾼 쌍으로 각 아이디어(대리 on, TK, m)의 한계 효과.
      덧붙임: 시작 주 바꾸기, lookback·buffer 흔들기, 대리 가격 점검, 빌더 주장(3/30 BE 밀림, 10/12 미리보기) 확인.
파일은 이 스크립트 하나만 쓴다(결과는 화면 출력). deskx_expand.py를 import만 하고 고치지 않는다.
실행: cd data/derived/analysis/rsi_breakout/code && PYTHONIOENCODING=utf-8 py verify_expand_robust.py
"""
import json, os, sys
from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deskx  # noqa: E402
import deskx_expand as X  # noqa: E402
from deskx import PERIODS  # noqa: E402
from desk.strategies import run_model, in_semis_budget, UNKNOWN  # noqa: E402

FRONT, BACK, SEP, FULL = (p[0] for p in PERIODS)
PER = {lab: (a, b) for lab, a, b in PERIODS}
MONTHS = [f"2026-{m:02d}" for m in range(1, 11)]
G = dict(X.GRID)
LIVE = X.LIVE_NAME
CHOSEN = "T/off/m=1"
NEIGH = ["T/off/m=2", "T/on/m=1", "TK/off/m=1"]
SAVED = os.path.join(HERE, "../results/deskx/expand.json")


class RobustEngine(X.ExpandEngine):
    exclude: frozenset = frozenset()

    def candidates(self, p, session):
        base = super().candidates(p, session)
        if self.exclude:
            base = {k: v for k, v in base.items() if self.ticker[k] not in self.exclude}
        return base


def run(E, p, exclude=()):
    E.exclude = frozenset(exclude)
    E.log = []
    try:
        v, rows = E.run(p)
        log = E.log
    finally:
        E.log = None
        E.exclude = frozenset()
    return v, rows, log


def decompose(E, v, rows):
    """일별 수익 r(일 d 시가→다음 시가, evaluate와 같은 색인)과 종목별 일별 기여(w×수익) 행렬."""
    k0 = E.days.index(rows[0]["day"])
    idx = [row["day"] for row in rows[:-1]]
    r = np.zeros(len(idx))
    c = defaultdict(lambda: np.zeros(len(idx)))
    inv = np.zeros(len(idx))
    for i in range(len(idx)):
        k = k0 + i + 1
        for key, wt in rows[i]["w"].items():
            o = E.col("opn", key)
            ret = (o[k] / o[k - 1] - 1) if o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) \
                and o[k - 1] > 0 else 0.0
            c[E.ticker[key]][i] += wt * ret
            r[i] += wt * ret
        inv[i] = sum(rows[i]["w"].values())
    vv = v.to_numpy()
    assert np.max(np.abs(vv[1:] / vv[:-1] - 1 - r)) < 1e-12
    return pd.Series(r, index=idx), {t: pd.Series(x, index=idx) for t, x in c.items()}, pd.Series(inv, index=idx)


def pstat(r, a, b):
    x = r[(r.index >= a) & (r.index <= b)]
    if not len(x):
        return np.nan, np.nan
    curve = (1 + x).cumprod()
    return float(curve.iloc[-1] - 1), float((curve / np.maximum.accumulate(curve) - 1).min())


def pstat0(r, a, b):
    """시작점 1.0을 고점에 포함한 낙폭(첫날 하락도 잡는다)."""
    x = r[(r.index >= a) & (r.index <= b)]
    curve = np.concatenate([[1.0], (1 + x).cumprod().to_numpy()])
    return float(curve[-1] - 1), float((curve / np.maximum.accumulate(curve) - 1).min())


def summary(r):
    return {lab: pstat(r, a, b) for lab, (a, b) in PER.items()}


def monthly(r):
    return {m: float((1 + r[r.index.str.startswith(m)]).prod() - 1) for m in MONTHS}


def value_contrib(r, c, a, b):
    """기간 [a,b] 시작 1의 가치 가중 기여(합 = 기간 수익)."""
    mask = (r.index >= a) & (r.index <= b)
    x = r[mask]
    val = np.concatenate([[1.0], (1 + x).cumprod().to_numpy()[:-1]])
    out = {t: float((val * s[mask].to_numpy()).sum()) for t, s in c.items()}
    tot = float((1 + x).prod() - 1)
    assert abs(sum(out.values()) - tot) < 1e-9 * max(1, abs(tot))
    return out, tot


def fmt(x, nd=3):
    return "   nan" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def groups_stats(E, rows):
    nsub, maxsub, inv, effn = [], [], [], []
    for row in rows:
        g = defaultdict(float)
        w = {k: x for k, x in row["w"].items() if x > 1e-12}
        for k, x in w.items():
            g[E.group(k)] += x
        tot = sum(w.values())
        nsub.append(len(g))
        maxsub.append(max(g.values()) if g else 0.0)
        inv.append(tot)
        effn.append(tot ** 2 / sum(x * x for x in w.values()) if w else 0.0)
    return {"nsub": np.mean(nsub), "maxsub_avg": np.mean(maxsub), "maxsub_max": np.max(maxsub),
            "invested": np.mean(inv), "effN": np.mean(effn)}


def main():
    E = RobustEngine()
    saved = json.load(open(SAVED, encoding="utf-8"))
    sv = {r["name"]: r for r in saved["variants"]}

    # ------------------------------------------------------------------ 0. 재현
    print("=" * 100)
    print("0. 재현: 12개 변형을 다시 돌려 expand.json과 비교, T/off/m=none = desk run_model(반도체 50%)")
    R = {}
    worst = 0.0
    for name, p in X.GRID:
        v, rows, log = run(E, p)
        r, c, inv = decompose(E, v, rows)
        R[name] = dict(v=v, rows=rows, log=log, r=r, c=c, inv=inv, p=p)
        s = summary(r)
        for lab in (FRONT, BACK, SEP, FULL):
            worst = max(worst, abs(s[lab][0] - sv[name][lab]["수익"]), abs(s[lab][1] - sv[name][lab]["낙폭"]))
    ref = np.asarray(run_model(E.book, start=deskx.START).value)
    d_live = float(np.max(np.abs(R[LIVE]["v"].to_numpy() - ref)))
    print(f"  expand.json과 최대 차 {worst:.2e}, LIVE vs run_model 최대 차 {d_live:.2e}")
    assert worst < 1e-9 and d_live < 1e-12

    # ------------------------------------------------------------------ 1. 이웃·월별
    print("=" * 100)
    print("1. 앞 기준 1위와 이웃(한 축만 바꿈) vs LIVE — 기간·월별 수익")
    show = [LIVE, CHOSEN] + NEIGH
    print(f"{'변형':<14}" + "".join(f"{lab[:6]:>9}" for lab in (FRONT, BACK, SEP, FULL)) + f"{'MDD':>8}"
          + "".join(f"{m[-2:]:>7}" for m in MONTHS))
    for n in show:
        s, mo = summary(R[n]["r"]), monthly(R[n]["r"])
        print(f"{n:<14}" + "".join(f"{s[lab][0]:>+9.3f}" for lab in (FRONT, BACK, SEP, FULL)) + f"{s[FULL][1]:>+8.3f}"
              + "".join(f"{mo[m]:>+7.3f}" for m in MONTHS))
    ml, mc = monthly(R[LIVE]["r"]), monthly(R[CHOSEN]["r"])
    wins = [m[-2:] for m in MONTHS if mc[m] > ml[m] + 1e-9]
    ties = [m[-2:] for m in MONTHS if abs(mc[m] - ml[m]) <= 1e-9]
    print(f"  {CHOSEN}가 LIVE를 이긴 달: {wins} ({len(wins)}/10), 같은 달: {ties}")
    print("  월별 차(1위 - LIVE):", {m[-2:]: round(mc[m] - ml[m], 3) for m in MONTHS})
    # 보유가 LIVE와 같은 주
    for n in [CHOSEN] + NEIGH:
        live_keep = {e["session"]: set(e["keep"]) for e in R[LIVE]["log"]}
        diff = [e["session"] for e in R[n]["log"] if set(e["keep"]) != live_keep[e["session"]]]
        print(f"  {n}: LIVE와 종목이 다른 주 {len(diff)}/{len(R[n]['log'])}, 마지막으로 다른 주 {diff[-1] if diff else '-'}"
              f", 6/1 이후 다른 주 {sum(d >= '2026-06-01' for d in diff)}")

    # ------------------------------------------------------------------ 2. 쏠림
    print("=" * 100)
    print("2. 쏠림: 종목 기여(가치 가중, 기간 시작 1) — 상위 5와 상위 3 비중")
    for n in [LIVE, CHOSEN, "T/off/m=2"]:
        for lab in (FRONT, BACK, FULL):
            a, b = PER[lab]
            vc, tot = value_contrib(R[n]["r"], R[n]["c"], a, b)
            top = sorted(vc.items(), key=lambda x: -x[1])
            share = sum(x for _, x in top[:3]) / tot if tot > 0 else float("nan")
            print(f"  {n:<13}{lab[:6]:<7} 합 {tot:+.3f} 상위3 비중 {share:5.2f} | "
                  + ", ".join(f"{t} {x:+.3f}" for t, x in top[:5]) + " | 하위: "
                  + ", ".join(f"{t} {x:+.3f}" for t, x in top[-3:]))

    print("\n2b. 1위 대비 차이를 만든 종목(앞 구간, 가치 가중 기여 차 = 1위 - LIVE)")
    a, b = PER[FRONT]
    vc_c, tot_c = value_contrib(R[CHOSEN]["r"], R[CHOSEN]["c"], a, b)
    vc_l, tot_l = value_contrib(R[LIVE]["r"], R[LIVE]["c"], a, b)
    dd = {t: vc_c.get(t, 0) - vc_l.get(t, 0) for t in set(vc_c) | set(vc_l)}
    top = sorted(dd.items(), key=lambda x: -x[1])
    print(f"  앞 차이 {tot_c - tot_l:+.3f} = 기여 차 합 {sum(dd.values()):+.3f}")
    print("  +쪽:", ", ".join(f"{t} {x:+.3f}" for t, x in top[:6]))
    print("  -쪽:", ", ".join(f"{t} {x:+.3f}" for t, x in top[-6:]))
    # 뒤 구간도
    a, b = PER[BACK]
    vc_c2, tot_c2 = value_contrib(R[CHOSEN]["r"], R[CHOSEN]["c"], a, b)
    vc_l2, tot_l2 = value_contrib(R[LIVE]["r"], R[LIVE]["c"], a, b)
    dd2 = {t: vc_c2.get(t, 0) - vc_l2.get(t, 0) for t in set(vc_c2) | set(vc_l2)}
    top2 = sorted(dd2.items(), key=lambda x: -x[1])
    print(f"  뒤 차이 {tot_c2 - tot_l2:+.3f}: +쪽", ", ".join(f"{t} {x:+.3f}" for t, x in top2[:4]),
          "| -쪽", ", ".join(f"{t} {x:+.3f}" for t, x in top2[-4:]))

    print("\n2c. 회계식 제거: 그 종목 몫을 현금(수익 0)으로 바꿈 — 앞/뒤/전체 수익")
    for n in [LIVE, CHOSEN, "T/off/m=2"]:
        r, c = R[n]["r"], R[n]["c"]
        vc, _ = value_contrib(r, c, *PER[FULL])
        top = [t for t, _ in sorted(vc.items(), key=lambda x: -x[1])]
        for drop in ([top[0]], top[:2], ["BE"]):
            rr = r.copy()
            for t in drop:
                if t in c:
                    rr = rr - c[t]
            s = summary(rr)
            print(f"  {n:<13} -{'+'.join(drop):<11} 앞 {s[FRONT][0]:+.3f} 뒤 {s[BACK][0]:+.3f} 전체 {s[FULL][0]:+.3f}"
                  f" MDD {s[FULL][1]:+.3f}")

    print("\n2d. 구조적 제거: 그 종목을 후보에서 빼고 다시 돌림(빈자리는 규칙대로 다음 종목) — 1위 vs LIVE")
    print(f"  {'제외':<12}{'변형':<13}{'앞':>8}{'뒤':>8}{'전체':>8}{'MDD':>8}   1위-LIVE: 앞 / 뒤 / 전체 / MDD")
    for excl in ([], ["SNDK"], ["BE"], ["DELL"], ["WDC"], ["CIEN"], ["SNDK", "DELL"], ["BE", "CIEN"]):
        res = {}
        for n in (LIVE, CHOSEN, "T/off/m=2"):
            v, rows, _ = run(E, G[n], exclude=excl)
            r, _, _ = decompose(E, v, rows)
            res[n] = summary(r)
        for n in (LIVE, CHOSEN, "T/off/m=2"):
            s = res[n]
            tail = ""
            if n != LIVE:
                L = res[LIVE]
                tail = (f"   {s[FRONT][0] - L[FRONT][0]:+.3f} / {s[BACK][0] - L[BACK][0]:+.3f} / "
                        f"{s[FULL][0] - L[FULL][0]:+.3f} / {s[FULL][1] - L[FULL][1]:+.3f}")
            print(f"  {('-' + '+'.join(excl)) if excl else '(없음)':<12}{n:<13}{s[FRONT][0]:>+8.3f}{s[BACK][0]:>+8.3f}"
                  f"{s[FULL][0]:>+8.3f}{s[FULL][1]:>+8.3f}{tail}")

    print("\n2e. 날짜 쏠림")
    for n in [LIVE, CHOSEN]:
        r = R[n]["r"]
        for lab in (FRONT, FULL):
            a, b = PER[lab]
            x = r[(r.index >= a) & (r.index <= b)]
            top2 = x.nlargest(2)
            y = x.copy()
            y[top2.index] = 0.0
            print(f"  {n:<13}{lab[:6]:<7} 원래 {float((1 + x).prod() - 1):+.3f} → 상위 2일({', '.join(f'{d} {v:+.3f}' for d, v in top2.items())}) 0으로 "
                  f"{float((1 + y).prod() - 1):+.3f}")
    rc, rl = R[CHOSEN]["r"], R[LIVE]["r"]
    for lab in (FRONT, FULL):
        a, b = PER[lab]
        m = (rc.index >= a) & (rc.index <= b)
        d = (rc - rl)[m]
        base = float((1 + rc[m]).prod() - (1 + rl[m]).prod())
        out = [f"원래 차 {base:+.3f}"]
        for k in (1, 2, 3, 5):
            idx = d.nlargest(k).index
            y = rc[m].copy()
            y[idx] = rl[m][idx]
            out.append(f"차 상위 {k}일을 LIVE로 바꾸면 {float((1 + y).prod() - (1 + rl[m]).prod()):+.3f}")
        print(f"  {lab[:6]} 1위-LIVE: " + " | ".join(out))
        print(f"         차가 큰 날: " + ", ".join(f"{dd_} {x:+.3f}" for dd_, x in d.nlargest(5).items())
              + f" | 차가 0이 아닌 날 {int((d.abs() > 1e-12).sum())}/{len(d)}")
        if lab == FRONT:
            cc, cl = R[CHOSEN]["c"], R[LIVE]["c"]
            for day in d.nlargest(3).index:
                nd = {t: float(cc.get(t, pd.Series(0.0, index=rc.index))[day] - cl.get(t, pd.Series(0.0, index=rc.index))[day])
                      for t in set(cc) | set(cl)}
                big = sorted(nd.items(), key=lambda x: -abs(x[1]))[:3]
                print(f"         {day}(시가→다음 시가) 종목별 차: " + ", ".join(f"{t} {x:+.3f}" for t, x in big))
    # 월별 앞 구간 차가 어느 달에서 났는가
    a, b = PER[FRONT]
    print("  앞 구간 1위-LIVE 차의 월별 분해(그 달 수익 차):",
          {m[-2:]: round(mc[m] - ml[m], 3) for m in MONTHS[:5]})

    # ------------------------------------------------------------------ 3. 분산
    print("=" * 100)
    print("3. m 한도가 분산시키는가, MDD를 낮추는가 — 12개 변형")
    print(f"{'변형':<14}{'세부수':>7}{'최대세부':>9}{'최대세부max':>11}{'유효N':>7}{'투자':>7}{'일변동':>8}"
          f"{'앞MDD':>8}{'뒤MDD':>8}{'전체MDD':>8}{'전체MDD0':>9} | 투자분만: {'앞':>7}{'뒤':>7}{'전체':>7}{'MDD':>7}")
    D = {}
    for name, _ in X.GRID:
        gs = groups_stats(E, R[name]["rows"])
        r, inv = R[name]["r"], R[name]["inv"]
        s = summary(r)
        _, mdd0 = pstat0(r, *PER[FULL])
        ri = r / inv.where(inv > 1e-9, np.nan)
        ri = ri.fillna(0.0)
        si = summary(ri)
        vol = float(r[(r.index >= deskx.START)].std() * np.sqrt(252))
        D[name] = dict(gs=gs, s=s, si=si, vol=vol)
        print(f"{name:<14}{gs['nsub']:>7.2f}{gs['maxsub_avg']:>9.3f}{gs['maxsub_max']:>11.3f}{gs['effN']:>7.2f}"
              f"{gs['invested']:>7.3f}{vol:>8.3f}{s[FRONT][1]:>+8.3f}{s[BACK][1]:>+8.3f}{s[FULL][1]:>+8.3f}{mdd0:>+9.3f}"
              f" | {'':>8}{si[FRONT][0]:>+7.3f}{si[BACK][0]:>+7.3f}{si[FULL][0]:>+7.3f}{si[FULL][1]:>+7.3f}")
    print("  m=1 vs m=none (같은 univ·proxies): 전체 MDD 차, 일변동 차")
    for u in ("T", "TK"):
        for px in ("off", "on"):
            a, b = D[f"{u}/{px}/m=1"], D[f"{u}/{px}/m=none"]
            print(f"   {u}/{px}: MDD {a['s'][FULL][1] - b['s'][FULL][1]:+.3f}, 앞MDD {a['s'][FRONT][1] - b['s'][FRONT][1]:+.3f},"
                  f" 뒤MDD {a['s'][BACK][1] - b['s'][BACK][1]:+.3f}, 변동 {a['vol'] - b['vol']:+.3f}, 투자 "
                  f"{a['gs']['invested'] - b['gs']['invested']:+.3f}, 최대세부 {a['gs']['maxsub_avg'] - b['gs']['maxsub_avg']:+.3f}")
    # MDD 시점
    for n in (LIVE, CHOSEN):
        r = R[n]["r"]
        curve = (1 + r[r.index >= deskx.START]).cumprod()
        dd_ = curve / curve.cummax() - 1
        trough = dd_.idxmin()
        peak = curve[:trough].idxmax()
        print(f"  {n}: 전체 MDD {dd_.min():+.3f} 고점 {peak} → 저점 {trough}")

    # ------------------------------------------------------------------ 4. 한계 효과
    print("=" * 100)
    print("4. 한 축만 바꾼 쌍의 한계 효과(앞 / 뒤 / 9/1~ / 전체 / 전체 MDD)")

    def eff(pairs, lab):
        rows_ = []
        for a, b in pairs:
            sa, sb = D[a]["s"], D[b]["s"]
            rows_.append([sa[FRONT][0] - sb[FRONT][0], sa[BACK][0] - sb[BACK][0], sa[SEP][0] - sb[SEP][0],
                          sa[FULL][0] - sb[FULL][0], sa[FULL][1] - sb[FULL][1]])
            print(f"   {a:<14} - {b:<14} " + " / ".join(f"{x:+.3f}" for x in rows_[-1]))
        arr = np.array(rows_)
        pos = (arr > 1e-9).sum(axis=0)
        print(f"  [{lab}] 평균 " + " / ".join(f"{x:+.3f}" for x in arr.mean(axis=0))
              + f" | 양수 개수(앞,뒤,9/1~,전체,MDD) {tuple(int(x) for x in pos)} / {len(pairs)}")

    ms = ("1", "2", "none")
    eff([(f"{u}/on/m={m}", f"{u}/off/m={m}") for u in ("T", "TK") for m in ms], "대리 on - off")
    eff([(f"TK/{px}/m={m}", f"T/{px}/m={m}") for px in ("off", "on") for m in ms], "TK - T")
    eff([(f"{u}/{px}/m=1", f"{u}/{px}/m=none") for u in ("T", "TK") for px in ("off", "on")], "m=1 - none")
    eff([(f"{u}/{px}/m=2", f"{u}/{px}/m=none") for u in ("T", "TK") for px in ("off", "on")], "m=2 - none")

    # ------------------------------------------------------------------ 5. 시작 주
    print("=" * 100)
    print("5. 시작 주 바꾸기(첫 매수 주만 다름): 공통 구간 3/2~5/29, 6/1~10/9, 3/2~10/9 — 1위·m=2 vs LIVE")
    starts = ["2026-01-02", "2026-01-12", "2026-01-20", "2026-01-26", "2026-02-02", "2026-02-09", "2026-02-17",
              "2026-02-23", "2026-03-02"]
    orig = deskx.START
    tally = Counter()
    try:
        for st in starts:
            deskx.START = st
            res = {}
            for n in (LIVE, CHOSEN, "T/off/m=2"):
                v, rows, _ = run(E, G[n])
                r, _, _ = decompose(E, v, rows)
                res[n] = (pstat(r, "2026-03-02", "2026-05-29")[0], pstat(r, "2026-06-01", "2026-10-09")[0],
                          pstat(r, "2026-03-02", "2026-10-09"))
            L, C, M2 = res[LIVE], res[CHOSEN], res["T/off/m=2"]
            tally["front"] += C[0] > L[0]
            tally["back"] += C[1] > L[1]
            tally["mdd"] += C[2][1] > L[2][1]
            print(f"  시작 {st}: LIVE 앞' {L[0]:+.3f} 뒤 {L[1]:+.3f} 3/2~ {L[2][0]:+.3f} MDD {L[2][1]:+.3f} | "
                  f"1위 차 {C[0] - L[0]:+.3f} / {C[1] - L[1]:+.3f} / {C[2][0] - L[2][0]:+.3f} / MDD {C[2][1] - L[2][1]:+.3f} | "
                  f"m=2 차 {M2[0] - L[0]:+.3f} / {M2[1] - L[1]:+.3f}")
    finally:
        deskx.START = orig
    print(f"  1위가 LIVE를 이긴 시작 수: 앞' {tally['front']}/{len(starts)}, 뒤 {tally['back']}/{len(starts)}, "
          f"MDD(3/2~) {tally['mdd']}/{len(starts)}")

    # ------------------------------------------------------------------ 6. 모수 흔들기
    print("=" * 100)
    print("6. lookback × buffer 흔들기(T/off): m=1 - m=none, 같은 모수끼리")
    t6 = Counter()
    cells = 0
    dfront, dback, dfull, dmdd = [], [], [], []
    for lb in (100, 110, 120, 130, 140):
        for bf in (3, 4, 5, 6, 7):
            a = run(E, replace(G[CHOSEN], lookback=lb, buffer=bf))
            b = run(E, replace(G[LIVE], lookback=lb, buffer=bf))
            sa, sb = summary(decompose(E, a[0], a[1])[0]), summary(decompose(E, b[0], b[1])[0])
            cells += 1
            dfront.append(sa[FRONT][0] - sb[FRONT][0])
            dback.append(sa[BACK][0] - sb[BACK][0])
            dfull.append(sa[FULL][0] - sb[FULL][0])
            dmdd.append(sa[FULL][1] - sb[FULL][1])
            t6["front"] += sa[FRONT][0] > sb[FRONT][0]
            t6["back"] += sa[BACK][0] > sb[BACK][0]
            t6["full"] += sa[FULL][0] > sb[FULL][0]
            t6["mdd"] += sa[FULL][1] > sb[FULL][1]
            t6["both"] += (sa[FRONT][0] > sb[FRONT][0]) and (sa[BACK][0] > sb[BACK][0])
            print(f"  lb={lb} buf={bf}: none 앞 {sb[FRONT][0]:+.3f} 뒤 {sb[BACK][0]:+.3f} MDD {sb[FULL][1]:+.3f} | "
                  f"m=1 차 앞 {sa[FRONT][0] - sb[FRONT][0]:+.3f} 뒤 {sa[BACK][0] - sb[BACK][0]:+.3f} "
                  f"전체 {sa[FULL][0] - sb[FULL][0]:+.3f} MDD {sa[FULL][1] - sb[FULL][1]:+.3f}")
    print(f"  m=1이 이긴 칸(/{cells}): 앞 {t6['front']}, 뒤 {t6['back']}, 앞·뒤 모두 {t6['both']}, 전체 {t6['full']},"
          f" MDD 얕음 {t6['mdd']}")
    for lab_, arr in (("앞", dfront), ("뒤", dback), ("전체", dfull), ("MDD", dmdd)):
        arr = np.array(arr)
        print(f"  m=1-none {lab_}: 평균 {arr.mean():+.3f} 중앙 {np.median(arr):+.3f} 최소 {arr.min():+.3f} 최대 {arr.max():+.3f}")
    chosen_front = dfront[[(lb, bf) for lb in (100, 110, 120, 130, 140) for bf in (3, 4, 5, 6, 7)].index((120, 5))]
    print(f"  채택 칸(lb=120, buf=5)의 앞 차 {chosen_front:+.3f}는 25칸 중 {int((np.array(dfront) > chosen_front).sum()) + 1}위")
    # 7f 0 비중 자리(선택됐지만 상한 때문에 비중 0) — 대리·KoAct 변형 포함
    zero = {}
    for name, _ in X.GRID:
        z = sum(1 for e in R[name]["log"] for x in e["keep"] if e["tgt"].get(x, 0.0) <= 1e-9)
        zero[name] = z
    print("  선택됐지만 비중 0인 자리(종목-주):", zero)

    # ------------------------------------------------------------------ 7. 구현 점검
    print("=" * 100)
    print("7. 구현 점검")
    # 7a 대리 가격
    for t in E.fetched:
        key = E.proxy_key[t]
        o = E.col("opn", key)
        a = E.col("adj", key)
        lo = E.days.index("2025-08-01")
        seg = o[lo:]
        rr = seg[1:] / seg[:-1] - 1
        print(f"  {t}: 2025-08~ 시가 NaN {int(np.isnan(seg).sum())}, 수정종가 NaN {int(np.isnan(a[lo:]).sum())},"
              f" 최대 |시가→시가| {np.nanmax(np.abs(rr)):.3f}, 섹터 {E.sector.get(key)} (반도체 예산 {in_semis_budget(E.sector.get(key, UNKNOWN))})")
    # 7b 대리 후보 시점
    bad = 0
    for e in R["T/on/m=1"]["log"]:
        pdf = deskx.latest_before(E.book.pdfs, e["session"]) if hasattr(deskx, "latest_before") else None
        if pdf is not None and pdf >= e["session"]:
            bad += 1
    print(f"  대리 후보 PDF 시점 위반 {bad}건(latest_before 사용)")
    # 7c 빌더 주장: 2026-03-30 BE 밀림
    k = E.days.index("2026-03-30")
    for n in ("T/off/m=1", "T/on/m=1"):
        prev = [e for e in R[n]["log"] if e["session"] < "2026-03-30"][-1]
        ranked = E.ranking(G[n], k, "2026-03-30")
        rank = {E.ticker[x["key"]]: i + 1 for i, x in enumerate(ranked)}
        cur = [e for e in R[n]["log"] if e["session"] == "2026-03-30"][0]
        print(f"  {n} 2026-03-30: BE 순위 {rank.get('BE')}, MU {rank.get('MU')}, TTMI {rank.get('TTMI')}; 지난주 BE 보유 "
              f"{'BE' in {E.ticker[x] for x in prev['keep']}}; 이번 주 보유 {[E.ticker[x] for x in cur['keep']]}")
    # BE 기여(두 변형)
    for n in ("T/off/m=1", "T/on/m=1", LIVE):
        vc, _ = value_contrib(R[n]["r"], R[n]["c"], *PER[FULL])
        print(f"  {n} BE 전체 기여 {vc.get('BE', 0):+.3f}, MU {vc.get('MU', 0):+.3f}, TTMI {vc.get('TTMI', 0):+.3f}")
    # 7d 미리보기
    pv = X.preview(E, G[CHOSEN], R[CHOSEN]["log"])
    tot = sum(h["weight"] for h in pv["holdings"])
    print(f"  10/12 미리보기({CHOSEN}): " + ", ".join(f"{h['name']} {h['weight']:.3f}" for h in pv["holdings"])
          + f" | 합 {tot:.3f} 현금 {pv['cash']:.3f} 반도체예산 {pv['semis_budget_weight']:.3f} 지난주 {pv['held_last_week']}")
    pl = X.preview(E, G[LIVE], R[LIVE]["log"])
    print(f"  10/12 미리보기(LIVE): " + ", ".join(f"{h['name']} {h['weight']:.3f}" for h in pl["holdings"]))
    # 7e m 한도로 쓰인 그룹 라벨 출처: AI_UNIVERSE 세부 섹터인가, watchlist 섹터 대체인가, 미분류인가
    lab = Counter()
    for e in R[CHOSEN]["log"]:
        for x in e["keep"]:
            g = E.group(x)
            src = "미분류" if g.startswith("미분류") else ("AI_UNIVERSE" if g in X.TOP_OF else "watchlist 섹터")
            lab[src] += 1
    print(f"  {CHOSEN} 보유 종목-주의 그룹 라벨 출처: {dict(lab)}")


if __name__ == "__main__":
    main()
