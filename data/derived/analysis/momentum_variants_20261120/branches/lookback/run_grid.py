"""갈래 lookback: 단일 기간 수익 순위(모멘텀 기간 짧게·길게) 전수 격자.

격자(사전 고정): lb∈{5,10,20,40,60,90,120,180,250} × skip∈{0,5,21}(skip<lb → 23쌍)
 × 종목군∈{TIME,FUNDS,AI,WIDE200,WIDETECH50,ALL50} × n∈{5,7,10,15} × weight∈{equal,ivol}(TIME은 time 추가)
 × caps∈{True,False} × buffer∈{0,5}, freq='W'. 총 23×4×2×2×(5×2+3)=4,784 설정.
계열(사전 고정): 기간 묶음 short{5,10,20}·mid{40,60,90}·long{120,180,250}. 후보는 묶음마다 상반기 칼마 1위(최대 3개).
자격: 상반기 MDD ≥ 기준 상반기 MDD − 5%p. H2는 고른 뒤 적기만 한다.
실행: py data/derived/analysis/momentum_variants_20261120/branches/lookback/run_grid.py
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

LBS = [5, 10, 20, 40, 60, 90, 120, 180, 250]
SKIPS = [0, 5, 21]
UNIS = ["TIME", "FUNDS", "AI", "WIDE200", "WIDETECH50", "ALL50"]
NS = [5, 7, 10, 15]
CAPS = [True, False]
BUFS = [0, 5]
FUND_UNIS = {"TIME", "KOACT", "FUNDS"}
H1A, H1B = "2026-01-02", "2026-06-30"
FAMILY = {5: "short", 10: "short", 20: "short", 40: "mid", 60: "mid", 90: "mid", 120: "long", 180: "long", 250: "long"}


def cid(c):
    return f"{c['universe']}|lb{c['lb']}s{c['skip']}|n{c['n']}b{c['buffer']}|{c['weight']}|caps{int(c['caps'])}"


def grid():
    out = []
    for lb, sk in ((l, s) for l in LBS for s in SKIPS if s < l):
        for u in UNIS:
            ws = ["equal", "ivol"] + (["time"] if u == "TIME" else [])
            for n, w, cp, b in itertools.product(NS, ws, CAPS, BUFS):
                out.append(dict(lb=lb, skip=sk, universe=u, n=n, weight=w, caps=cp, buffer=b))
    return out


def r6(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def main():
    t0 = time.time()
    e = E.Engine()
    scores = {(l, s): e.ret(l, s) for l in LBS for s in SKIPS if s < l}
    cfgs = grid()
    print("configs", len(cfgs), flush=True)

    base = e.run(e.ret(120), universe="TIME", n=7, buffer=5, weight="time", caps=True)
    bm = e.metrics(base)
    bm["H1_MDD"] = e.mdd(base, H1A, H1B)
    bm["Y2025H2"] = None
    print("baseline", {k: round(v, 4) for k, v in bm.items() if v is not None}, flush=True)

    rows, paths = [], {}
    for i, c in enumerate(cfgs):
        kw = dict(universe=c["universe"], n=c["n"], buffer=c["buffer"], weight=c["weight"], caps=c["caps"], freq="W")
        res = e.run(scores[(c["lb"], c["skip"])], **kw)
        m = e.metrics(res)
        m["H1_MDD"] = e.mdd(res, H1A, H1B)
        if c["universe"] not in FUND_UNIS:
            r25 = e.run(scores[(c["lb"], c["skip"])], start="2025-07-01", end="2025-12-31", **kw)
            m.update(e.metrics_2025(r25))
        else:
            m["Y2025H2"] = None
        m["calmar_H1"] = m["H1"] / abs(m["H1_MDD"]) if m["H1_MDD"] < 0 else float("inf")
        k = cid(c)
        rows.append({"id": k, "family": FAMILY[c["lb"]], **c, **{a: r6(b) for a, b in m.items()}})
        paths[k] = E.Result(res.days, res.v_open, res.v_close, res.expo, res.r_oo, res.r_oc, res.turnover, {})
        if i % 500 == 0:
            print(i, round(time.time() - t0, 1), flush=True)

    # ---- 고르기(상반기만)
    lim = bm["H1_MDD"] - 0.05
    elig = [r for r in rows if r["H1_MDD"] is not None and r["H1_MDD"] >= lim]
    finals = []
    for fam in ["short", "mid", "long"]:
        cand = sorted([r for r in elig if r["family"] == fam], key=lambda r: -r["calmar_H1"])
        if cand:
            finals.append(cand[0])
    finals.sort(key=lambda r: -r["calmar_H1"])

    fin_out = []
    for r in finals:
        kw = dict(universe=r["universe"], n=r["n"], buffer=r["buffer"], weight=r["weight"], caps=r["caps"], freq="W")
        S = scores[(r["lb"], r["skip"])]
        d1 = e.metrics(e.run(S, delay=1, **kw))
        c10 = e.metrics(e.run(S, cost_bps=10, **kw))
        fo = dict(r)
        fo["delay1_H2"] = r6(d1["H2"])
        fo["delay1_RECENT"] = r6(d1["RECENT"])
        fo["delay1_MDD_H2"] = r6(d1["MDD_H2"])
        fo["cost10_H2"] = r6(c10["H2"])
        fo["cost10_RECENT"] = r6(c10["RECENT"])
        fo["cost10_MDD_H2"] = r6(c10["MDD_H2"])
        fo["expo_ctrl_H2"] = None  # 노출 1(현금 규칙 없음)이라 대조 불필요
        # 순위 안 위치(상반기 칼마 기준, 같은 계열/전체)
        fo["rank_H1_calmar_all_elig"] = 1 + sum(1 for x in elig if x["calmar_H1"] > r["calmar_H1"])
        fin_out.append(fo)

    # ---- 이동 전방 검증
    wf = {}
    def do_wf(name, keys):
        out = E.Engine.walk_forward({k: paths[k] for k in keys}, lookback=63)
        res = out["result"]
        wf[name] = {"n_configs": len(keys), "Mar_Nov": r6(res.v_close[-1] / res.v_open[0] - 1),
                    "H2": r6(E.Engine.span(res, "2026-07-01", E.LAST)), "RECENT": r6(E.Engine.span(res, "2026-09-01", E.LAST)),
                    "MDD": r6(E.Engine.mdd(res)), "n_switch": out["n_switch"], "picks_by_month": out["picks_by_month"]}
    do_wf("full_grid", list(paths))
    for u in UNIS:
        do_wf(f"universe_{u}", [r["id"] for r in rows if r["universe"] == u])
    # 기간만 바꾸는 부분집합(n=7·caps·equal·buffer5·skip0) — 종목군별
    for u in ["FUNDS", "ALL50"]:
        do_wf(f"lb_only_{u}", [r["id"] for r in rows if r["universe"] == u and r["n"] == 7 and r["caps"] and r["weight"] == "equal" and r["buffer"] == 5 and r["skip"] == 0])
    # 비교: 같은 3~11월 구간의 기준 수익
    wf["baseline_Mar_Nov"] = r6(E.Engine.span(base, "2026-03-02", E.LAST))

    # ---- 기간별 모양 표 (n=7, caps, equal, buffer 5, skip 0; FUNDS·ALL50)
    shape = []
    for u in ["FUNDS", "ALL50"]:
        for lb in LBS:
            r = next(x for x in rows if x["universe"] == u and x["lb"] == lb and x["skip"] == 0 and x["n"] == 7
                     and x["caps"] and x["weight"] == "equal" and x["buffer"] == 5)
            shape.append({k: r[k] for k in ["universe", "lb", "H1", "H1_MDD", "H2", "RECENT", "MDD_H2", "Y2026", "Y2025H2", "turnover"]})
    # 기간별 평균(격자 전체, 종목군별) — 모양의 강건성 확인
    shape_avg = []
    for u in UNIS:
        for lb in LBS:
            g = [x for x in rows if x["universe"] == u and x["lb"] == lb]
            shape_avg.append({"universe": u, "lb": lb, "n_configs": len(g),
                              **{f"median_{k}": r6(float(np.nanmedian([x[k] for x in g if x[k] is not None])))
                                 for k in ["H1", "H2", "RECENT", "MDD_H2"]}})

    out = {"branch": "lookback", "n_configs": len(rows), "baseline": {k: r6(v) for k, v in bm.items()},
           "eligibility_H1_MDD_floor": r6(lim), "n_eligible": len(elig), "family_rule": FAMILY,
           "finalists": fin_out, "walk_forward": wf, "shape_table": shape, "shape_median_by_universe": shape_avg,
           "rows": rows, "runtime_sec": round(time.time() - t0, 1)}
    (HERE / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda x: None if x == float("inf") else x), encoding="utf-8")
    print("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
