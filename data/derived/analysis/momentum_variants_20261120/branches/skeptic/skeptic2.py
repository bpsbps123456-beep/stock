"""회의적 감사 2: AI 종목군 사후 선택 크기, 격자 선택 편향(상반기 칼마 ↔ 하반기 순위 상관, 후보 H2 백분위).

실행: py data/derived/analysis/momentum_variants_20261120/branches/skeptic/skeptic2.py → results2.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code"))
sys.path.insert(0, str(HERE.parent / "adaptive"))
sys.path.insert(0, str(HERE.parent / "composite"))
import engine as E  # noqa: E402
import adaptive as AD  # noqa: E402
import composite as CP  # noqa: E402


def met(e, r):
    m = e.metrics(r)
    m["H1_MDD"] = e.mdd(r, "2026-01-02", "2026-06-30")
    m["JulAug"] = e.span(r, "2026-07-01", "2026-08-31")
    m["OctNov"] = e.span(r, "2026-10-01", E.LAST)
    return {k: round(float(v), 4) for k, v in m.items() if k in ("H1", "H1_MDD", "H2", "RECENT", "MDD_H2", "JulAug", "OctNov")}


def main():
    out = {}
    e = E.Engine()
    orig = e.universe
    adv0630 = np.nan_to_num(e.adv20[e.pos["2025-07-01"]], nan=0.0)
    first = e.snap_dates[0]
    fund0 = e._mask(set(e.held.get((first, "TIME"), {})) | set(e.held.get((first, "KoAct"), {})))
    noaxti = np.ones(len(e.tickers), bool)
    noaxti[e.col["AXTI"]] = False

    def uni(name, k):
        if name == "AI_P0630":
            return orig("AI", k) & (adv0630 >= 50e6)
        if name == "AI_LIQ":
            return orig("AI", k) & (e.adv20[k] >= 50e6)
        if name == "AI_FUND0":
            return orig("AI", k) & fund0
        if name == "AI_NOAXTI":
            return orig("AI", k) & noaxti
        if name == "ALL50_NOAI":   # 다이얼 종목군에서 사후 AI 목록·펀드 사후 보유 빼기: WIDE만
            return orig("WIDE50", k)
        return orig(name, k)
    e.universe = uni
    B = AD.Builder(e)
    R60 = CP.residual(e, 60)
    out["sizes_2026_01_02"] = {u: int(e.universe(u, e.pos["2026-01-05"]).sum()) for u in ("AI", "AI_P0630", "AI_LIQ", "AI_FUND0", "AI_NOAXTI")}
    runs = {}
    for u in ("AI", "AI_P0630", "AI_LIQ", "AI_FUND0", "AI_NOAXTI"):
        runs[f"RES60_n5@{u}"] = e.run(R60, universe=u, n=5, buffer=5, weight="equal", caps=True)
        runs[f"F1_dialALL50@{u}"] = B.icdial(u, 7, 60, 20, 10, "rev5", "ALL50")
    runs["F1_dialWIDE50@AI"] = B.icdial("AI", 7, 60, 20, 10, "rev5", "ALL50_NOAI")
    for u in ("WIDETECH50",):
        runs[f"WT_M_n3_ivol_noAXTI"] = None
    e.universe = orig
    # WIDETECH50 후보에서 AXTI 하나만 빼기
    def uni2(name, k):
        if name.endswith("_NOAXTI"):
            return orig(name[:-7], k) & noaxti
        return orig(name, k)
    e.universe = uni2
    runs["WT_M_n3_ivol_noAXTI"] = e.run(e.ret(60), universe="WIDETECH50_NOAXTI", n=3, buffer=5, weight="ivol", caps=False, freq="M")
    runs["WT_W_n5b0_noAXTI"] = e.run(e.ret(60), universe="WIDETECH50_NOAXTI", n=5, buffer=0, weight="equal", caps=False, freq="W")
    e.universe = orig
    out["ai_hindsight_variants"] = {k: met(e, r) for k, r in runs.items()}

    # 선택 편향: 갈래별 상반기 칼마 순위 ↔ 하반기 순위 상관, 후보 H2 백분위
    br = {}
    cands = {"construction": ["WIDETECH50|lb60|M|n3|b5|ivol|nocap", "TIME|lb120|W|n3|b5|time|nocap", "FUNDS|lb60|W|n3|b10|equal|nocap"],
             "lookback": ["WIDETECH50|lb60s0|n5b0|equal|caps0", "TIME|lb120s5|n5b5|time|caps0"],
             "adaptive": ["F1:u=AI,n=7,lb=60,h=20,K=10,alt=rev5,du=ALL50"],
             "composite": ["RES60|AI|n5|equal"],
             "universe": ["WIDETECH50|r60|n7|nocap|equal"]}
    for b, ids in cands.items():
        d = json.loads((HERE.parent / b / "results.json").read_text(encoding="utf-8"))
        rows = d["rows"]
        if b == "universe":
            rows = [r for r in rows if r.get("mask") is None]
        df = pd.DataFrame([{k: r.get(k) for k in ("id", "H1", "H1_MDD", "H2", "RECENT", "MDD_H2")} for r in rows]).dropna()
        df["calmar"] = df.H1 / df.H1_MDD.abs()
        rho = float(df.calmar.rank().corr(df.H2.rank()))
        rho_r = float(df.calmar.rank().corr(df.RECENT.rank()))
        top = df.sort_values("calmar", ascending=False)
        dec = int(max(1, len(df) // 10))
        info = {"n_rows": len(df), "spearman_calmarH1_H2": round(rho, 3), "spearman_calmarH1_RECENT": round(rho_r, 3),
                "H2_median_all": round(float(df.H2.median()), 4), "H2_median_top_decile_calmar": round(float(top.H2.iloc[:dec].median()), 4),
                "H2_median_bottom_decile_calmar": round(float(top.H2.iloc[-dec:].median()), 4),
                "H2_best": round(float(df.H2.max()), 4)}
        for i in ids:
            row = df[df.id == i]
            if len(row):
                h2 = float(row.H2.iloc[0])
                info[i] = {"H2": round(h2, 4), "H2_pctile": round(float((df.H2 < h2).mean()), 3),
                           "calmar_rank": int((df.calmar > float(row.calmar.iloc[0])).sum()) + 1}
        br[b] = info
    out["selection_bias"] = br
    (HERE / "results2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
