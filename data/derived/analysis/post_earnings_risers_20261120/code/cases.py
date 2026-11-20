"""사례 조사 표본: 시즌마다 실적 뒤(다음 날 시가 → +20) 가장 많이 오른 종목과, 같은 시즌·비슷한 반응일·비슷한 크기인데
하위 1/3에 머문 짝. 짝은 결과를 보고 고르지만 고르는 규칙은 기계적이다(반응일 QQQ 대비·시총 순위 거리 최소, 중복 없음).

결과: results/cases.csv (조사자에게는 ticker·발표일·반응일만 보낸다: cases_blind.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
TOP = {"2026Q1": 8, "2026Q2": 8, "2026Q3": 8, "2026Q4": 4}


def main() -> None:
    ev = pd.read_csv(BASE / "results/events.csv")
    ft = pd.read_csv(BASE / "results/features.csv")
    d = ev.merge(ft, on=["ticker", "reaction_date"])
    d = d[d.d20_x.notna()].copy()
    d["pct"] = d.groupby("season").d20_x.rank(pct=True)
    d["r0_rk"] = d.groupby("season").r0_x.rank(pct=True)
    d["mc_rk"] = d.groupby("season").px_mcap_b.rank(pct=True)
    rows = []
    for s, n in TOP.items():
        g = d[d.season == s]
        top = g.sort_values("d20_x", ascending=False).head(n)
        pool = g[g.pct <= 1 / 3].copy()
        used = set()
        for t in top.itertuples():
            cand = pool[~pool.index.isin(used) & (pool.ticker != t.ticker)]
            dist = (cand.r0_rk - t.r0_rk).abs() + 0.5 * (cand.mc_rk - t.mc_rk).abs()
            c = cand.loc[dist.idxmin()]
            used.add(c.name)
            for role, r in (("riser", t._asdict()), ("control", c.to_dict())):
                rows.append({"pair": f"{s}-{t.ticker}", "role": role, "ticker": r["ticker"], "ann_date": r["ann_date"],
                             "reaction_date": r["reaction_date"], "season": s, "group": r["group"], "held_any": r["held_any"],
                             "r0_x": round(r["r0_x"], 2), "d20_x": round(r["d20_x"], 2), "mcap_b": round(r["px_mcap_b"], 1)})
    cs = pd.DataFrame(rows)
    cs.to_csv(BASE / "results/cases.csv", index=False)
    blind = cs[["ticker", "ann_date", "reaction_date"]].sample(frac=1, random_state=7).reset_index(drop=True)
    blind.insert(0, "case_id", [f"C{i + 1:02d}" for i in range(len(blind))])
    blind.to_csv(BASE / "astra/cases_blind.csv", index=False)
    print(cs.groupby(["season", "role"]).size().unstack(), len(blind))


if __name__ == "__main__":
    main()
