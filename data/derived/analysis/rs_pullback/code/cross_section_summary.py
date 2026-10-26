"""cross_section 갈래 요약 (2026-10-13). 새 조합을 고르지 않는다.

- 36조합 평균으로 본 '강함 안 SP_hi − 강함 전체', 'SP_lo − 강함 전체' 날짜 계열(NW t).
- cross_section_rankers.json(51개 평가)과 cross_section_extra.json(점검)을 짧게 묶어 cross_section.json에 넣는다.

실행: PYTHONIOENCODING=utf-8 py cross_section_summary.py   (cross_section.py sort·rank, cross_section_extra.py 뒤)
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cross_section as X  # noqa: E402

H = X.H


def legs_vs_strong() -> dict:
    P = H.load_panel()
    U = H.universe_mask(P)
    E = H.eligible()
    feat = X.features(P)
    out = {}
    gs = {X.combo_name(L, p): X.groups(feat["S"][L], feat["P"][p], U) for L in X.LS for p in X.P_NAMES}
    for h in X.HS:
        for w in ("front", "back", "y2025", "y2024"):
            dates = H.window_dates(w, h)
            cols = X.window_cols(P, w)
            hi_s, lo_s = {}, {}
            for n, g in gs.items():
                hi, _ = X.daily_excess(g["SP_hi"], h, dates, cols, E)
                lo, _ = X.daily_excess(g["SP_lo"], h, dates, cols, E)
                st, _ = X.daily_excess(g["strong"], h, dates, cols, E)
                hi_s[n], lo_s[n] = hi - st, lo - st
            out[f"h{h}_{w}"] = {"SP_hi_minus_strong_all36": X.stat(pd.concat(hi_s, axis=1).mean(axis=1), h),
                                "SP_lo_minus_strong_all36": X.stat(pd.concat(lo_s, axis=1).mean(axis=1), h)}
    return out


def rankers_compact() -> list:
    rk = json.load(open(os.path.join(X.RESULTS, "cross_section_rankers.json"), encoding="utf-8"))
    rows = []
    for r in rk["rows"]:
        R = r["result"]
        row = {"name": r["name"], "spec": r["spec"], "no_lookahead": r["no_lookahead"]}
        for w in ("front", "back", "y2025", "y2024", "late"):
            x = R.get(w, {})
            row[w] = {f"ic{h}": x.get(f"h{h}", {}).get("ic") for h in X.HS}
            row[w].update({f"ic{h}_t": x.get(f"h{h}", {}).get("ic_t") for h in X.HS})
            row[w]["top20_h10"] = x.get("h10", {}).get("top20")
            row[w]["top20_h20"] = x.get("h20", {}).get("top20")
            row[w]["port_S"] = x.get("port_S", {}).get("ret")
            row[w]["port_SA"] = x.get("port_SA", {}).get("ret")
        rows.append(row)
    return rows


if __name__ == "__main__":
    path = os.path.join(X.RESULTS, "cross_section.json")
    res = json.load(open(path, encoding="utf-8"))
    res["legs_vs_strong"] = legs_vs_strong()
    rows = rankers_compact()
    res["rankers_compact"] = rows
    res["rankers_counts"] = {
        "n": len(rows), "no_lookahead_all_true": all(r["no_lookahead"] for r in rows),
        "ic10_front_pos": sum(r["front"]["ic10"] > 0 for r in rows),
        "ic10_back_pos": sum(r["back"]["ic10"] > 0 for r in rows),
        "ic10_both_pos": sum(r["front"]["ic10"] > 0 and r["back"]["ic10"] > 0 for r in rows),
        "ic20_both_pos": sum(r["front"]["ic20"] > 0 and r["back"]["ic20"] > 0 for r in rows),
        "top20_h10_both_pos": sum(r["front"]["top20_h10"] > 0 and r["back"]["top20_h10"] > 0 for r in rows),
        "front_best_ic20": max(rows, key=lambda r: r["front"]["ic20"])["name"],
        "front_best_ic10": max(rows, key=lambda r: r["front"]["ic10"])["name"],
    }
    fb10 = next(r for r in rows if r["name"] == res["rankers_counts"]["front_best_ic10"])
    fb20 = next(r for r in rows if r["name"] == res["rankers_counts"]["front_best_ic20"])
    res["rankers_counts"]["front_best_ic10_back"] = fb10["back"]["ic10"]
    res["rankers_counts"]["front_best_ic20_back"] = fb20["back"]["ic20"]
    ex = json.load(open(os.path.join(X.RESULTS, "cross_section_extra.json"), encoding="utf-8"))
    res["extra_files"] = {"rankers_full": "cross_section_rankers.json", "ranker_checks_and_beta": "cross_section_extra.json"}
    res["beta_check_all36_h10"] = ex["beta_check_all36_h10"]
    res["ranker_candidates_ic10_both_pos"] = ex["rankers"]["candidates_ic10_both_pos"]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res["rankers_counts"], ensure_ascii=False))
    for k, v in res["legs_vs_strong"].items():
        print(k, {a: (b["mean"], b["t"]) for a, b in v.items()})
