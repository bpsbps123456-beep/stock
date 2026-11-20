"""갈래 composite — 합성 점수 모멘텀 (2026-11-20).

점수 계열(모두 세션 k 행 = k-1 종가까지만 사용):
  a RA_<lbs>   : 여러 기간 수익 순위 평균(그 세션 종목군 안 백분위 순위)
  b RV<lb>     : 수익(lb) / 일간 로그수익 표준편차(lb)
  c RES<lb>    : 잔차 모멘텀 = Σlog r_i − β·Σlog r_QQQ (β = 같은 lb 창 일간 로그수익 회귀)
  d HI252      : AC[k-1] / max(AC[k-252..k-1])
  e ACC<s>v<l> : 순위(수익 s) − 순위(수익 l) (종목군 안 백분위)
  f TQ<lb>     : 수익(lb) × |Σlog r| / Σ|log r|
  g UP<lb>     : lb 창의 상승일 비율
격자: 점수 15 × 종목군 6 × n 3 × weight 2 = 540 (caps=True, buffer=5, freq='W', 2026-01-02~LAST).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code"))
import engine as E  # noqa: E402

UNIVERSES = ["TIME", "FUNDS", "AI", "WIDE200", "WIDETECH50", "ALL50"]
NS = [5, 7, 10]
WEIGHTS = ["equal", "ivol"]
FUND_U = {"TIME", "KOACT", "FUNDS"}
K_START_DAY = "2025-07-01"  # 점수 행을 만드는 첫 세션(2025 하반기 확인 포함)


# ---------------------------------------------------------------- 점수 원재료 (k-1 종가까지)
def lagged_rolling(e, frame: pd.DataFrame) -> np.ndarray:
    """frame 행 i = 종가 i+1에서 끝나는 창의 값(일간 수익 기준) → 행 k = 종가 k-1까지."""
    s = frame.to_numpy(float)
    out = np.full_like(e.AC, np.nan)
    out[2:] = s[:-1]
    return out


def daily_lr(e) -> pd.DataFrame:
    return pd.DataFrame(np.log(e.AC[1:] / e.AC[:-1]))


def risk_adj(e, lb):
    return e.ret(lb) / e.vol(lb)


def residual(e, lb):
    X = daily_lr(e)
    q = e.bench_ac["QQQ"].to_numpy(float)
    y = np.log(q[1:] / q[:-1])
    Y = pd.DataFrame(np.where(X.notna().to_numpy(), y[:, None], np.nan))
    mp = int(lb * 0.9)
    r = lambda F: F.rolling(lb, min_periods=mp)  # noqa: E731
    mx, my = r(X).mean(), r(Y).mean()
    mxy, myy = r(X * Y).mean(), r(Y * Y).mean()
    cnt = r(X).count()
    beta = (mxy - mx * my) / (myy - my * my)
    res = (mx - beta * my) * cnt  # Σx − β·Σy over valid days
    return lagged_rolling(e, res)


def high_prox(e, lb=252):
    A = pd.DataFrame(e.AC)
    hi = A.rolling(lb, min_periods=200).max().to_numpy()
    out = np.full_like(e.AC, np.nan)
    out[1:] = e.AC[:-1] / hi[:-1]  # 행 k: AC[k-1] / max(AC[k-lb..k-1])
    return out


def trend_quality(e, lb):
    X = daily_lr(e)
    mp = int(lb * 0.9)
    eff = X.rolling(lb, min_periods=mp).sum().abs() / X.abs().rolling(lb, min_periods=mp).sum()
    return e.ret(lb) * lagged_rolling(e, eff)


def up_ratio(e, lb):
    X = daily_lr(e)
    up = (X > 0).astype(float).where(X.notna())
    return lagged_rolling(e, up.rolling(lb, min_periods=int(lb * 0.9)).mean())


# ---------------------------------------------------------------- 종목군 안 순위 점수
def universe_masks(e, u, k0):
    K = len(e.days)
    M = np.zeros_like(e.AC, dtype=bool)
    for k in range(k0, K):
        M[k] = e.universe(u, k)
    return M


def rank_combo(comps, M, k0, weights):
    """comps: [행렬], weights: 부호/가중. 행 k의 종목군∩전 성분 유한 종목 안 백분위 순위의 가중합."""
    ok = M.copy()
    for C in comps:
        ok &= np.isfinite(C)
    out = np.full(M.shape, np.nan)
    acc = np.zeros(M.shape)
    for C, w in zip(comps, weights):
        R = pd.DataFrame(np.where(ok, C, np.nan)[k0:]).rank(axis=1, pct=True).to_numpy()
        acc[k0:] += w * np.nan_to_num(R)
    out[k0:] = np.where(ok[k0:], acc[k0:], np.nan)
    return out


def build_scores(e):
    """{이름: (계열, 정의문, 종목군→행렬 함수)}"""
    k0 = e.pos[K_START_DAY]
    R = {lb: e.ret(lb) for lb in (5, 20, 60, 120, 250)}
    plain = {
        "RV60": ("b", "ret(60)/vol(60)", risk_adj(e, 60)),
        "RV120": ("b", "ret(120)/vol(120)", risk_adj(e, 120)),
        "RES60": ("c", "Σlr−β·Σlr_QQQ, 60세션 회귀", residual(e, 60)),
        "RES120": ("c", "Σlr−β·Σlr_QQQ, 120세션 회귀", residual(e, 120)),
        "HI252": ("d", "AC[k-1]/max(AC[k-252..k-1])", high_prox(e)),
        "TQ60": ("f", "ret(60)×|Σlr|/Σ|lr| (60)", trend_quality(e, 60)),
        "TQ120": ("f", "ret(120)×|Σlr|/Σ|lr| (120)", trend_quality(e, 120)),
        "UP60": ("g", "상승일 비율(60)", up_ratio(e, 60)),
        "UP120": ("g", "상승일 비율(120)", up_ratio(e, 120)),
    }
    ranked = {
        "RA_20_60_120": ("a", [20, 60, 120], [1, 1, 1]),
        "RA_60_120_250": ("a", [60, 120, 250], [1, 1, 1]),
        "RA_20_120": ("a", [20, 120], [1, 1]),
        "RA_5_20_60": ("a", [5, 20, 60], [1, 1, 1]),
        "ACC20v120": ("e", [20, 120], [1, -1]),
        "ACC20v60": ("e", [20, 60], [1, -1]),
    }
    return k0, R, plain, ranked


def main():
    t0 = time.time()
    e = E.Engine()
    k0, R, plain, ranked = build_scores(e)
    masks = {u: universe_masks(e, u, k0) for u in UNIVERSES}
    print("scores ready", round(time.time() - t0, 1))

    base = e.run(e.ret(120), universe="TIME", n=7, buffer=5, weight="time", caps=True)
    bm = e.metrics(base)
    bm["H1_MDD"] = e.mdd(base, "2026-01-02", "2026-06-30")
    elig = bm["H1_MDD"] - 0.05

    rows, results = [], {}
    defs = {}
    for sname, (fam, desc, S) in plain.items():
        defs[sname] = (fam, desc)
    for sname, (fam, lbs, ws) in ranked.items():
        desc = ("순위평균 ret(" + ",".join(map(str, lbs)) + ")") if fam == "a" else f"pct순위 ret({lbs[0]}) − pct순위 ret({lbs[1]})"
        defs[sname] = (fam, desc)

    def score_for(sname, u):
        if sname in plain:
            return plain[sname][2]
        fam, lbs, ws = ranked[sname]
        return rank_combo([R[lb] for lb in lbs], masks[u], k0, ws)

    cache = {}
    for sname in defs:
        for u in UNIVERSES:
            S = score_for(sname, u)
            cache[(sname, u)] = S
            for n in NS:
                for w in WEIGHTS:
                    cid = f"{sname}|{u}|n{n}|{w}"
                    res = e.run(S, universe=u, n=n, buffer=5, weight=w, caps=True)
                    m = e.metrics(res)
                    m["H1_MDD"] = e.mdd(res, "2026-01-02", "2026-06-30")
                    m["calmar_H1"] = m["H1"] / abs(m["H1_MDD"]) if m["H1_MDD"] < 0 else float("inf")
                    m["eligible"] = bool(m["H1_MDD"] >= elig)
                    rows.append({"id": cid, "score": sname, "family": defs[sname][0], "universe": u, "n": n,
                                 "weight": w, **m})
                    results[cid] = res
    print("grid", len(rows), round(time.time() - t0, 1))

    # ---- 상반기만으로 최종 후보(계열 다르게 최대 3)
    elig_rows = sorted([r for r in rows if r["eligible"]], key=lambda r: -r["calmar_H1"])
    finals, fams = [], set()
    for r in elig_rows:
        if r["family"] in fams:
            continue
        finals.append(r)
        fams.add(r["family"])
        if len(finals) == 3:
            break

    fin_out = []
    for r in finals:
        S = cache[(r["score"], r["universe"])]
        kw = dict(universe=r["universe"], n=r["n"], buffer=5, weight=r["weight"], caps=True)
        d1 = e.metrics(e.run(S, delay=1, **kw))
        c10 = e.metrics(e.run(S, cost_bps=10, **kw))
        y25 = None
        if r["universe"] not in FUND_U:
            y25 = e.metrics_2025(e.run(S, start="2025-07-01", end="2025-12-31", **kw))
        fin_out.append({**r, "definition": defs[r["score"]][1], "delay1": d1, "cost10": c10, "y2025h2": y25,
                        "holdings_last": list(results[r["id"]].holdings.items())[-1]})

    # ---- 이동 전방 검증(격자 전체 540)
    wf = e.walk_forward(results, lookback=63)
    wf_ret = e.span(wf["result"], "2026-03-02", E.LAST)
    wf_mdd = e.mdd(wf["result"])
    # 대표 부분집합: 종목군별
    wf_by_u = {}
    for u in UNIVERSES:
        sub = {k: v for k, v in results.items() if k.split("|")[1] == u}
        w_ = e.walk_forward(sub, lookback=63)
        wf_by_u[u] = {"ret": e.span(w_["result"], "2026-03-02", E.LAST), "n_switch": w_["n_switch"],
                      "picks": w_["picks_by_month"]}
    base_mar = e.span(base, "2026-03-02", E.LAST)

    out = {
        "branch": "composite",
        "grid": {"scores": list(defs), "families": {k: v for k, v in defs.items()}, "universes": UNIVERSES,
                 "n": NS, "weights": WEIGHTS, "caps": True, "buffer": 5, "freq": "W", "n_configs": len(rows)},
        "baseline": {**bm, "Mar_Nov": base_mar},
        "eligibility_H1_MDD_floor": elig,
        "finalists": fin_out,
        "walk_forward_all": {"Mar_Nov_return": wf_ret, "MDD": wf_mdd, "n_switch": wf["n_switch"],
                             "picks_by_month": wf["picks_by_month"], "n_configs": len(results)},
        "walk_forward_by_universe": wf_by_u,
        "top20_H1_calmar_eligible": [r["id"] for r in elig_rows[:20]],
        "rows": rows,
    }
    (HERE / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("saved", round(time.time() - t0, 1))
    for r in fin_out:
        print(json.dumps({k: r[k] for k in ("id", "H1", "H1_MDD", "calmar_H1", "Y2026", "H2", "RECENT", "MDD2026",
                                            "MDD_H2", "turnover")}, default=float))
        print("  delay1 H2", r["delay1"]["H2"], "cost10 H2", r["cost10"]["H2"], "2025", r["y2025h2"])
    print("WF", wf_ret, wf["n_switch"], wf["picks_by_month"])
    print("base", bm, base_mar)


if __name__ == "__main__":
    main()
