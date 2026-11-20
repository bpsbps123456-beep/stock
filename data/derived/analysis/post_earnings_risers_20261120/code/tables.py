"""보고서 표 두 개(결과 본 뒤 탐색): 한국 개인 매수 전·후, 업종 추세 × 한국 개인 등장 2×2.

results/kr_before_after.csv, results/theme_kr_2x2.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]


def main() -> None:
    ev = pd.read_csv(BASE / "results/events.csv")
    ft = pd.read_csv(BASE / "results/features.csv")
    fx = pd.read_csv(BASE / "results/features_extra.csv")
    d = ev.merge(ft, on=["ticker", "reaction_date"]).merge(fx, on=["ticker", "reaction_date"])
    d = d[d.d20_x.notna()].copy()
    d["half"] = np.where(d.season.isin(["2026Q1", "2026Q2"]), "H1", "H2")
    d["tile"] = pd.cut(d.groupby("season").d20_x.rank(pct=True), [0, 1 / 3, 2 / 3, 1], labels=["하위", "중간", "상위"])
    rows = []
    for name, g in (("2026 보유 전체", d), ("보유 당시", d[d.held_any])):
        g = g.assign(pre=g.kr_days_pre > 0, post=g.post_kr_buy_m > 0)
        both = g[(g.kr_buy_pre_m > 0) & g.post_kr_buy_m.notna()]
        ratio = np.log((both.post_kr_buy_m + 0.5) / (both.kr_buy_pre_m + 0.5))
        for t in ("하위", "중간", "상위"):
            gt = g[g.tile == t]
            rows.append({"sample": name, "tile": t, "n": len(gt), "pre_any_pct": round(100 * gt.pre.mean(), 1),
                         "post_any_pct": round(100 * gt.post.mean(), 1),
                         "post_over_pre_median_x": round(float(np.exp(ratio[both.tile == t].median())), 2),
                         "n_ratio": int((both.tile == t).sum())})
    pd.DataFrame(rows).to_csv(BASE / "results/kr_before_after.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    d["theme"] = np.where(d.groupby("season").t_grp_pre20.rank(pct=True) > 0.5, "업종 강", "업종 약")
    d["kr"] = np.where(d.kr_days_pre > 0, "한국 개인 상위50 등장", "등장 없음")
    d["pct"] = d.groupby("season").d20_x.rank(pct=True) * 100
    t2 = (d.groupby(["theme", "kr", "half"]).agg(n=("ticker", "size"), pct_mean=("pct", "mean"), d20_median=("d20_x", "median"),
                                                 win_pct=("d20_x", lambda s: 100 * (s > 0).mean())).round(1).reset_index())
    t2.to_csv(BASE / "results/theme_kr_2x2.csv", index=False)
    print(t2.to_string(index=False))


if __name__ == "__main__":
    main()
