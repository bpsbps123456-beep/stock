"""실적 후 흐름 예측 틀 — 통과 규칙을 겹치는 것끼리 묶고, 가짜 사건(보통 날)과 비교해 '실적 고유'인지 가른다.

실행: py data/derived/analysis/earnings_trend/code/framework.py  → results/framework.csv, results/framework.json

1) rules_passed.csv의 규칙을 같은 진입·기간·방향 안에서 사건이 많이 겹치는 것(자카드 ≥ 0.4)끼리 묶는다.
   대표 = 2026A·2026B 기준 대비 차이 중 작은 쪽이 가장 큰 규칙.
2) 대표 규칙을 가짜 사건(실적과 15거래일 넘게 떨어진 날, 같은 특징 계산)에 그대로 적용한다.
   실적 사건의 기준 대비 차이 − 가짜 사건의 기준 대비 차이 ≥ 8%p면 '실적 고유', 아니면 '2026 장세·업종 흐름'.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import window  # noqa: E402
from fetch_sec import RESULTS  # noqa: E402
from fetch_wide import W  # noqa: E402
from rules import predicates  # noqa: E402

JACCARD = 0.4
SPECIFIC = 0.08


def mask_of(df: pd.DataFrame, rule: str, e5: bool) -> np.ndarray:
    P = predicates(df, e5=e5)
    return np.logical_and.reduce([P[n][2] for n in rule.split(" & ")])


def win_stats(df: pd.DataFrame, m: np.ndarray, ycol: str, sel: np.ndarray) -> tuple[int, float, float, float]:
    y = pd.to_numeric(df[ycol], errors="coerce").values
    ok = sel & ~np.isnan(y)
    base = float((y[ok] > 0).mean()) if ok.any() else np.nan
    s = ok & m
    return int(s.sum()), (float((y[s] > 0).mean()) if s.any() else np.nan), base, (float(np.median(y[s])) if s.any() else np.nan)


def main() -> None:
    ev = pd.read_csv(RESULTS / "events_wide.csv")
    ev["window"] = ev["reaction_date"].apply(window).fillna("other")
    pl = pd.read_parquet(W / "placebo_wide.parquet")
    pl["window"] = pl["reaction_date"].apply(window).fillna("other")
    rules = pd.read_csv(RESULTS / "rules_passed.csv")
    rules["minlift"] = np.minimum(rules["lift_A"].abs(), rules["lift_B"].abs())
    reps = []
    for (entry, target, side), g in rules.groupby(["entry", "target", "side"]):
        e5 = entry == "E5"
        g = g.sort_values(["minlift", "n_B"], ascending=False)
        masks = {r: mask_of(ev, r, e5) for r in g["rule"]}
        clusters: list[dict] = []
        for _, r in g.iterrows():
            m = masks[r["rule"]]
            for c in clusters:
                inter = (m & c["mask"]).sum()
                if inter / max((m | c["mask"]).sum(), 1) >= JACCARD:
                    c["members"].append(r["rule"])
                    break
            else:
                clusters.append({"rep": r, "mask": m, "members": [r["rule"]]})
        for c in clusters:
            r = c["rep"]
            row = {k: r[k] for k in ["entry", "target", "side", "rule", "n_A", "win_A", "lift_A", "n_B", "win_B", "lift_B",
                                     "mean_B", "median_B", "n_P", "win_P", "lift_P"]}
            row["n_variants"] = len(c["members"])
            row["variants"] = " | ".join(c["members"][1:6])
            for alt in [x for x in r.index if x.startswith("B_") and x.endswith("_win") and pd.notna(r[x])]:
                row[alt] = r[alt]
                row[alt.replace("_win", "_base")] = r[alt.replace("_win", "_base")]
            row["ai_2026_n"], row["ai_2026_win"] = r.get("ai_2026_n"), r.get("ai_2026_win")
            # 가짜 사건 비교(2026 전체)
            pm = mask_of(pl, r["rule"], e5)
            sel26_e = ev["window"].str.startswith("2026").values
            sel26_p = pl["window"].str.startswith("2026").values
            n_e, w_e, b_e, _ = win_stats(ev, c["mask"], r["target"], sel26_e)
            n_p, w_p, b_p, med_p = win_stats(pl, pm, r["target"], sel26_p)
            row.update({"n_2026": n_e, "win_2026": w_e, "base_2026": b_e,
                        "placebo_n_2026": n_p, "placebo_win_2026": w_p, "placebo_base_2026": b_p, "placebo_median": med_p})
            lift_e = (w_e - b_e) if pd.notna(w_e) else np.nan
            lift_p = (w_p - b_p) if pd.notna(w_p) and n_p >= 30 else np.nan
            row["earnings_specific"] = (bool(abs(lift_e) - (abs(lift_p) if np.sign(lift_p) == np.sign(lift_e) else -abs(lift_p)) >= SPECIFIC)
                                        if pd.notna(lift_p) else None)
            reps.append(row)
    out = pd.DataFrame(reps).sort_values(["entry", "target", "side", "lift_B"], key=lambda s: s if s.name != "lift_B" else -s.abs())
    out.to_csv(RESULTS / "framework.csv", index=False)
    summ = out.groupby(["entry", "target", "side"]).agg(clusters=("rule", "size"),
                                                         specific=("earnings_specific", lambda s: int((s == True).sum())))  # noqa: E712
    (RESULTS / "framework.json").write_text(json.dumps({"clusters": summ.reset_index().to_dict("records"),
                                                        "jaccard": JACCARD, "specific_gap": SPECIFIC}, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print(summ.to_string())


if __name__ == "__main__":
    main()
