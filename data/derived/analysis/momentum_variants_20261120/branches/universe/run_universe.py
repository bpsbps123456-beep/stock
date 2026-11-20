"""갈래 universe: 종목군을 넓히면 나아지나(사용자 가설 '종목은 더 넓게').

사전 고정(결과 보기 전):
- 점수 3종 고정: ret120, ret60, ret250 skip21. 종목군만 바꾼다.
- 격자(본 격자): 13 종목군 × 3 점수 × n{7,10,15} × caps{T,F} × weight equal(TIME만 time 추가), buffer 5, 주간(W).
  = 12×18 + 1×36 = 252 설정.
- 생존 편향 점검 격자: WIDE·WIDETECH·ALL 9개 종목군에 시점 마스크를 씌운 변형
  PIT0630 = 2025-06-30 종가까지 20일 평균 거래대금(adv20[2025-07-01 세션])이 그 종목군 문턱 이상이던 종목만.
  PIT1231 = 같은 방식, 2025-12-31 종가까지(2026 시작 시점, 2025 하반기 실행에는 쓰지 않음).
  마스크는 pick 함수로 구현(엔진 기본 버퍼 선택과 같은 규칙을 마스크 통과 종목에만 적용).
  = 9×18×2 = 324 설정(2026 실행).
- 2025 하반기 보조 실행(start 2025-07-01, end 2025-12-31): 펀드 종목군 제외 10 종목군×18 + PIT0630 9×18 = 342 실행.
- 최종 후보: 본 격자 252개만 후보 풀. 계열 = 종목군 묶음 {펀드(TIME·KOACT·FUNDS), AI, WIDE, WIDETECH, ALL}.
  자격: 상반기 MDD ≥ 기준 상반기 MDD − 0.05. 순위: H1 / |H1 MDD|. 계열마다 1위, 위에서 3개.
- 이동 전방 검증: (a) 본 격자 252개, (b) 본+PIT 576개, (c) PIT0630 + 펀드·AI(시점 편향을 줄인 풀).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code"))
import engine as E  # noqa: E402

UNIVERSES = ["TIME", "KOACT", "FUNDS", "AI", "WIDE50", "WIDE100", "WIDE200", "WIDE500", "WIDE1000",
             "WIDETECH50", "WIDETECH200", "ALL50", "ALL200"]
WIDE_LIKE = [u for u in UNIVERSES if u.startswith(("WIDE", "ALL"))]
FUND_U = {"TIME", "KOACT", "FUNDS"}
SCORES = {"r120": (120, 0), "r60": (60, 0), "r250s21": (250, 21)}
NS = [7, 10, 15]
CAPS = [True, False]
BUFFER = 5


def family(u: str) -> str:
    if u in FUND_U:
        return "FUND"
    if u == "AI":
        return "AI"
    if u.startswith("WIDETECH"):
        return "WIDETECH"
    if u.startswith("WIDE"):
        return "WIDE"
    return "ALL"


def threshold(u: str) -> float:
    for p in ("WIDETECH", "WIDE", "ALL"):
        if u.startswith(p):
            return float(u[len(p):]) * 1e6
    raise ValueError(u)


def make_pick(mask: np.ndarray, n: int, buffer: int):
    """엔진 기본 버퍼 선택과 같은 규칙을 mask 통과 종목에만 적용."""
    def pick(k, order, score_k, prev):
        o = [c for c in order if mask[c]]
        top = o[: n + buffer]
        chosen = [c for c in top if c in prev][:n]
        for c in o:
            if len(chosen) >= n:
                break
            if c not in chosen:
                chosen.append(c)
        return chosen
    return pick


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if math.isnan(x) else x
    if isinstance(x, np.integer):
        return int(x)
    return x


def main():
    t0 = time.time()
    e = E.Engine()
    S = {nm: e.ret(lb, sk) for nm, (lb, sk) in SCORES.items()}
    pit = {}
    for tag, day in (("PIT0630", "2025-07-01"), ("PIT1231", "2026-01-02")):
        k = e.pos[day]  # adv20[k] = 20일 평균 거래대금, k-1 종가까지
        adv = np.nan_to_num(e.adv20[k], nan=0.0)
        pit[tag] = {u: adv >= threshold(u) for u in WIDE_LIKE}

    def cfg_list():
        out = []
        for u in UNIVERSES:
            ws = ["equal", "time"] if u == "TIME" else ["equal"]
            for sc in SCORES:
                for n in NS:
                    for cp in CAPS:
                        for w in ws:
                            out.append(dict(universe=u, score=sc, n=n, caps=cp, weight=w, mask=None))
        for tag in ("PIT0630", "PIT1231"):
            for u in WIDE_LIKE:
                for sc in SCORES:
                    for n in NS:
                        for cp in CAPS:
                            out.append(dict(universe=u, score=sc, n=n, caps=cp, weight="equal", mask=tag))
        return out

    def cid(c):
        m = f"|{c['mask']}" if c["mask"] else ""
        return f"{c['universe']}{m}|{c['score']}|n{c['n']}|{'cap' if c['caps'] else 'nocap'}|{c['weight']}"

    def run(c, **kw):
        pk = make_pick(pit[c["mask"]][c["universe"]], c["n"], BUFFER) if c["mask"] else None
        return e.run(S[c["score"]], universe=c["universe"], n=c["n"], buffer=BUFFER, weight=c["weight"],
                     caps=c["caps"], freq="W", pick=pk, **kw)

    cfgs = cfg_list()
    rows, res26 = {}, {}
    for c in cfgs:
        r = run(c)
        m = e.metrics(r)
        m["H1_MDD"] = e.mdd(r, "2026-01-02", "2026-06-30")
        m["calmar_H1"] = m["H1"] / abs(m["H1_MDD"]) if m["H1_MDD"] < 0 else float("nan")
        m["avg_names"] = float(np.mean([len(h) for h in r.holdings.values()])) if r.holdings else 0.0
        rows[cid(c)] = {**c, "id": cid(c), "family": family(c["universe"]), **m}
        res26[cid(c)] = r
    n_main = sum(1 for c in cfgs if c["mask"] is None)
    n_pit = len(cfgs) - n_main
    # 2025 하반기 보조
    n25 = 0
    for c in cfgs:
        if c["universe"] in FUND_U or c["mask"] == "PIT1231":
            rows[cid(c)]["Y2025H2"] = None
            rows[cid(c)]["MDD2025H2"] = None
            continue
        r = run(c, start="2025-07-01", end="2025-12-31")
        m = e.metrics_2025(r)
        rows[cid(c)].update(m)
        n25 += 1
    print("runs done", round(time.time() - t0, 1), "s", n_main, n_pit, n25)

    base_id = "TIME|r120|n7|cap|time"
    base = rows[base_id]
    elig = [r for r in rows.values() if r["mask"] is None and r["H1_MDD"] >= base["H1_MDD"] - 0.05
            and np.isfinite(r["calmar_H1"])]
    elig.sort(key=lambda r: -r["calmar_H1"])
    finalists, fams = [], set()
    for r in elig:
        if r["family"] in fams:
            continue
        fams.add(r["family"])
        finalists.append(r["id"])
        if len(finalists) == 3:
            break
    cfg_by_id = {cid(c): c for c in cfgs}
    fin = []
    for fid in finalists:
        c = cfg_by_id[fid]
        d1 = e.metrics(run(c, delay=1))
        c10 = e.metrics(run(c, cost_bps=10))
        row = dict(rows[fid])
        row["delay1_H2"] = d1["H2"]
        row["delay1_RECENT"] = d1["RECENT"]
        row["delay1_MDD_H2"] = d1["MDD_H2"]
        row["cost10_H2"] = c10["H2"]
        row["cost10_RECENT"] = c10["RECENT"]
        row["cost10_MDD_H2"] = c10["MDD_H2"]
        row["expo_ctrl_H2"] = None
        if c["universe"] in WIDE_LIKE:
            for tag in ("PIT0630", "PIT1231"):
                pid = cid({**c, "mask": tag})
                row[f"{tag}_check"] = {k: rows[pid].get(k) for k in ("H1", "H1_MDD", "Y2026", "H2", "RECENT", "MDD_H2", "Y2025H2")}
        fin.append(row)
    # 기준 자체의 지연·비용
    bc = cfg_by_id[base_id]
    base_extra = {"delay1_H2": e.metrics(run(bc, delay=1))["H2"], "cost10_H2": e.metrics(run(bc, cost_bps=10))["H2"]}

    # 이동 전방 검증
    def wf(ids, label):
        out = E.Engine.walk_forward({i: res26[i] for i in ids})
        r = out["result"]
        ret = float(r.v_close[-1] / r.v_open[0] - 1)
        mdd = E.Engine.mdd(r)
        return {"label": label, "n_configs": len(ids), "Mar_Nov_return": ret, "MDD": mdd,
                "n_switch": out["n_switch"], "picks_by_month": out["picks_by_month"],
                "H2_part": E.Engine.span(r, "2026-07-01", E.LAST), "RECENT_part": E.Engine.span(r, "2026-09-01", E.LAST)}
    main_ids = [i for i, r in rows.items() if r["mask"] is None]
    all_ids = list(rows)
    pitfa_ids = [i for i, r in rows.items() if r["mask"] == "PIT0630" or (r["mask"] is None and r["universe"] in FUND_U | {"AI"})]
    fund_ids = [i for i, r in rows.items() if r["mask"] is None and r["universe"] in FUND_U]
    wfs = [wf(main_ids, "main_252"), wf(all_ids, "main+PIT_576"), wf(pitfa_ids, "PIT0630+FUND+AI"),
           wf(fund_ids, "FUND_only")]
    # 같은 기간 기준·QQQ
    bres = res26[base_id]
    base_MarNov = E.Engine.span(bres, "2026-03-02", E.LAST)
    qqq = e.bench("QQQ")
    bench = {"QQQ": e.metrics(qqq), "QQQ_MarNov": E.Engine.span(qqq, "2026-03-02", E.LAST),
             "QQQ_2025H2": e.metrics_2025(e.bench("QQQ", "2025-07-01", "2025-12-31"))}

    # 종목군별 요약(18 설정 중앙값, equal만)
    def med(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return float(np.median(v)) if v else None
    summary = {}
    for u in UNIVERSES:
        for tag in (None, "PIT0630", "PIT1231"):
            if tag and u not in WIDE_LIKE:
                continue
            rs = [r for r in rows.values() if r["universe"] == u and r["mask"] == tag and r["weight"] == "equal"]
            key = u + (f"|{tag}" if tag else "")
            summary[key] = {k: med([r.get(k) for r in rs]) for k in
                            ("H1", "H1_MDD", "Y2026", "H2", "RECENT", "MDD_H2", "Y2025H2", "avg_names")}
            summary[key]["n_cfg"] = len(rs)
            summary[key]["universe_size_0102"] = int(e.universe(u, e.pos["2026-01-02"]).sum()) if u not in FUND_U else None
            summary[key]["universe_size_0901"] = int(e.universe(u, e.pos["2026-09-01"]).sum())
            if tag:
                summary[key]["universe_size_0102"] = int((e.universe(u, e.pos["2026-01-02"]) & pit[tag][u]).sum())
                summary[key]["universe_size_0901"] = int((e.universe(u, e.pos["2026-09-01"]) & pit[tag][u]).sum())
    # 생존 편향 크기: 짝지은 설정별 (원래 − PIT) 중앙값
    bias = {}
    for u in WIDE_LIKE:
        for tag in ("PIT0630", "PIT1231"):
            diffs = {k: [] for k in ("H1", "H2", "RECENT", "Y2026", "Y2025H2")}
            for r in rows.values():
                if r["universe"] == u and r["mask"] is None:
                    p = rows[cid({**cfg_by_id[r["id"]], "mask": tag})]
                    for k in diffs:
                        if r.get(k) is not None and p.get(k) is not None:
                            diffs[k].append(r[k] - p[k])
            bias[f"{u}|{tag}"] = {k: (float(np.median(v)) if v else None) for k, v in diffs.items()}
    # 점수별 × 종목군 대표(n=10, caps=True)
    rep = {}
    for u in UNIVERSES:
        for sc in SCORES:
            r = rows[cid(dict(universe=u, score=sc, n=10, caps=True, weight="equal", mask=None))]
            rep[f"{u}|{sc}"] = {k: r.get(k) for k in ("H1", "H1_MDD", "H2", "RECENT", "MDD_H2", "Y2025H2")}

    out = {
        "branch": "universe",
        "grid": {"universes": UNIVERSES, "scores": SCORES, "n": NS, "caps": CAPS, "buffer": BUFFER, "freq": "W",
                 "weight": "equal (+time for TIME)", "n_main": n_main, "n_pit": n_pit, "n_2025H2_runs": n25,
                 "n_total_2026_configs": len(cfgs)},
        "baseline": {**base, **base_extra, "MarNov": base_MarNov},
        "bench": bench,
        "selection_rule": "pool=main 252; H1_MDD >= base_H1_MDD-0.05; rank H1/|H1_MDD|; 1 per family FUND/AI/WIDE/WIDETECH/ALL; top 3",
        "n_eligible": len(elig),
        "finalists": fin,
        "walk_forward": wfs,
        "summary_by_universe": summary,
        "survivorship_bias_median_diff": bias,
        "rep_n10_cap": rep,
        "rows": list(rows.values()),
        "runtime_s": time.time() - t0,
    }
    (HERE / "results.json").write_text(json.dumps(clean(out), ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
