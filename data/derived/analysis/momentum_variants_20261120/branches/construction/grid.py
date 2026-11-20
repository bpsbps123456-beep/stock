"""갈래 construction: 같은 모멘텀 점수(ret lb)에서 구성·타이밍(freq·buffer·n·weight·caps·종목군)만 바꾼다.

격자(사전 고정, 결과 보기 전):
  universe ∈ {TIME, FUNDS, ALL50, WIDETECH50}, lb ∈ {60, 120}, freq ∈ {D, W, 2W, M},
  buffer ∈ {0, 3, 5, 10}, n ∈ {3, 5, 7, 10, 15, 20}, weight ∈ {equal, ivol, time(TIME만)}, caps ∈ {True, False}
  → lb 2 × freq 4 × buffer 4 × n 6 × weight × caps 2:
     TIME 2·4·4·6·3·2=1,152, 나머지 3종목군 × 2·4·4·6·2·2=768 → 3,456 설정. 점수 비례 비중은 생략.
각 설정은 cost 0·delay 0(본 평가), cost 10bp, delay 1로 세 번 돈다(같은 설정의 민감도, 새 설정 아님).
후보 선택(H1만): 칼마 = H1 / |H1_MDD|, 자격 H1_MDD >= 기준 H1_MDD - 0.05. 계열 = 종목군(사전 고정),
  계열마다 1위 → 그중 칼마 상위 3개 계열.
이동 전방 검증: 전체(본 평가)와 cost10 전체, 종목군별·lb별 부분집합.
실행: py data/derived/analysis/momentum_variants_20261120/branches/construction/grid.py
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code"))
import engine as E  # noqa: E402

UNIS = ["TIME", "FUNDS", "ALL50", "WIDETECH50"]
LBS = [60, 120]
FREQS = ["D", "W", "2W", "M"]
BUFS = [0, 3, 5, 10]
NS = [3, 5, 7, 10, 15, 20]
CAPS = [True, False]
KEYS = ("universe", "lb", "freq", "buffer", "n", "weight", "caps")


def grid():
    for u in UNIS:
        ws = ["equal", "ivol", "time"] if u == "TIME" else ["equal", "ivol"]
        for lb, fq, b, n, w, c in itertools.product(LBS, FREQS, BUFS, NS, ws, CAPS):
            yield dict(universe=u, lb=lb, freq=fq, buffer=b, n=n, weight=w, caps=c)


def cid(p):
    return f"{p['universe']}|lb{p['lb']}|{p['freq']}|n{p['n']}|b{p['buffer']}|{p['weight']}|{'cap' if p['caps'] else 'nocap'}"


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def main():
    t0 = time.time()
    e = E.Engine()
    S = {lb: e.ret(lb) for lb in LBS}

    def run(p, **kw):
        q = {k: v for k, v in p.items() if k != "lb"}
        return e.run(S[p["lb"]], **q, **kw)

    def met(r):
        m = e.metrics(r)
        m["H1_MDD"] = e.mdd(r, "2026-01-02", "2026-06-30")
        return m

    base_p = dict(universe="TIME", lb=120, freq="W", buffer=5, n=7, weight="time", caps=True)
    base = met(run(base_p))
    elig = base["H1_MDD"] - 0.05

    rows, R0, R10 = [], {}, {}
    for p in grid():
        i = cid(p)
        r0 = run(p)
        r10 = run(p, cost_bps=10)
        r1 = run(p, delay=1)
        m = met(r0)
        m10, m1 = e.metrics(r10), e.metrics(r1)
        m["calmar_H1"] = m["H1"] / abs(m["H1_MDD"]) if m["H1_MDD"] < 0 else float("inf")
        m["eligible"] = bool(m["H1_MDD"] >= elig)
        m["cost10"] = {k: m10[k] for k in ("Y2026", "H1", "H2", "RECENT", "MDD_H2")}
        m["delay1"] = {k: m1[k] for k in ("Y2026", "H1", "H2", "RECENT", "MDD_H2")}
        m["Mar_Nov"] = E.Engine.span(r0, "2026-03-02", E.LAST)
        rows.append({"id": i, **p, **m})
        R0[i], R10[i] = r0, r10
    print("grid", len(rows), "configs", round(time.time() - t0, 1), "s", flush=True)

    # ---------------- finalists (H1 only)
    best_by_fam = {}
    for row in rows:
        if not row["eligible"]:
            continue
        f = row["universe"]
        if f not in best_by_fam or row["calmar_H1"] > best_by_fam[f]["calmar_H1"]:
            best_by_fam[f] = row
    fam_rank = sorted(best_by_fam.values(), key=lambda r: -r["calmar_H1"])
    finalists = []
    for row in fam_rank[:3]:
        p = {k: row[k] for k in KEYS}
        f = dict(row)
        f["cost_sens"] = {}
        for c in (0, 10, 25):
            for d in (0, 1):
                mm = e.metrics(run(p, cost_bps=c, delay=d))
                f["cost_sens"][f"c{c}_d{d}"] = {k: mm[k] for k in ("Y2026", "H1", "H2", "RECENT", "MDD_H2")}
        if p["universe"] in ("TIME", "KOACT", "FUNDS"):
            f["Y2025H2"] = None
        else:
            r25 = run(p, start="2025-07-01", end="2025-12-31")
            f.update(e.metrics_2025(r25))
        f["expo_ctrl_H2"] = None  # 노출 1 (노출 규칙 없음)
        finalists.append(f)

    # ---------------- walk forward
    def wf(sub, R):
        out = e.walk_forward({k: R[k] for k in sub})
        res = out["result"]
        return {"n_configs": len(sub), "Mar_Nov_return": float(res.v_close[-1] / res.v_open[0] - 1),
                "MDD": e.mdd(res), "n_switch": out["n_switch"], "picks_by_month": out["picks_by_month"]}

    ids = [r["id"] for r in rows]
    WF = {"ALL_cost0": wf(ids, R0), "ALL_cost10_noswitchcost": wf(ids, R10)}
    for u in UNIS:
        WF[f"{u}_cost0"] = wf([r["id"] for r in rows if r["universe"] == u], R0)
    for lb in LBS:
        WF[f"lb{lb}_cost0"] = wf([r["id"] for r in rows if r["lb"] == lb], R0)
    for fq in FREQS:
        WF[f"freq{fq}_cost10"] = wf([r["id"] for r in rows if r["freq"] == fq], R10)

    # ---------------- 요약
    MS = ("Y2026", "H1", "H2", "RECENT", "MDD2026", "MDD_H2", "H1_MDD", "turnover", "Mar_Nov")

    def agg(g):
        return {"count": len(g),
                **{f"med_{m}": float(np.median([x[m] for x in g])) for m in MS},
                "med_H2_cost10": float(np.median([x["cost10"]["H2"] for x in g])),
                "med_Y2026_cost10": float(np.median([x["cost10"]["Y2026"] for x in g])),
                "med_H2_delay1": float(np.median([x["delay1"]["H2"] for x in g])),
                "share_H2_gt_base": float(np.mean([x["H2"] > base["H2"] for x in g]))}

    def by(key, sub):
        return {str(v): agg([r for r in sub if r[key] == v]) for v in sorted({r[key] for r in sub}, key=str)}

    summary = {k: by(k, rows) for k in ("n", "freq", "buffer", "weight", "caps", "universe", "lb")}
    summary_by_universe = {u: {k: by(k, [r for r in rows if r["universe"] == u]) for k in ("n", "freq", "buffer", "weight")} for u in UNIS}

    one_axis = {}
    for key, vals in (("n", NS), ("freq", FREQS), ("buffer", BUFS), ("weight", ["equal", "ivol", "time"]),
                      ("caps", CAPS), ("lb", LBS), ("universe", ["TIME", "FUNDS"])):
        for v in vals:
            p = dict(base_p)
            p[key] = v
            if p["universe"] != "TIME" and p["weight"] == "time":
                p["weight"] = "equal"
            r = next(x for x in rows if x["id"] == cid(p))
            one_axis[f"{key}={v}"] = {"id": r["id"], **{m: r[m] for m in ("Y2026", "H1", "H1_MDD", "H2", "RECENT", "MDD_H2", "turnover")},
                                      "H2_cost10": r["cost10"]["H2"], "H2_delay1": r["delay1"]["H2"]}

    elig_rows = sorted([r for r in rows if r["eligible"]], key=lambda r: -r["calmar_H1"])
    bh2 = max(rows, key=lambda r: r["H2"])
    pick = ("id", "calmar_H1", "H1", "H1_MDD", "H2", "RECENT", "Y2026", "MDD_H2")
    out = {
        "branch": "construction",
        "tested_configs": len(rows),
        "runs_per_config": "3 (cost0/delay0 본 평가, cost10, delay1)",
        "baseline": {"id": cid(base_p), **base},
        "eligibility_H1_MDD_min": elig,
        "family_rule": "계열 = 종목군(사전 고정). 계열별 자격 통과 칼마 1위 → 칼마 상위 3계열",
        "family_best": [{k: r[k] for k in pick} for r in fam_rank],
        "finalists": finalists,
        "walk_forward": WF,
        "summary": summary,
        "summary_by_universe": summary_by_universe,
        "one_axis_from_baseline": one_axis,
        "top20_H1_calmar": [{k: r[k] for k in pick} for r in elig_rows[:20]],
        "n_eligible": len(elig_rows),
        "share_all_H2_gt_base": float(np.mean([r["H2"] > base["H2"] for r in rows])),
        "best_H2_reference": {k: bh2[k] for k in pick},
        "rows": rows,
        "elapsed_s": time.time() - t0,
    }
    (HERE / "results.json").write_text(json.dumps(clean(out), ensure_ascii=False, indent=1), encoding="utf-8")
    print("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
