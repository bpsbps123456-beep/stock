"""요인 × 결과 검정(시즌 안 순위), 상반기 고르기 / 하반기 확인, 오른 종목 단면.

결과: results/ic.csv(요인 × 결과 × 반기 IC), results/profile.csv(시즌 안 상위·하위 1/3 중앙값), results/analysis.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUTS = ["d20_x", "d5_x", "r0_x", "t20_x", "nx_x", "d40_x"]
H1, H2 = {"2026Q1", "2026Q2"}, {"2026Q3", "2026Q4"}


def load(extra: list[Path] | None = None, broad: bool = False) -> pd.DataFrame:
    ev = pd.read_csv(BASE / "results/events.csv")
    ft = pd.read_csv(BASE / "results/features.csv")
    df = ev.merge(ft, on=["ticker", "reaction_date"], how="left")
    for p in extra or []:
        if p.exists():
            df = df.merge(pd.read_csv(p), on=["ticker", "reaction_date"], how="left")
    df = df.copy() if broad else df[df.held_any].copy()
    df["half"] = np.where(df.season.isin(H1), "H1", np.where(df.season.isin(H2), "H2", None))
    return df


def season_ic(df: pd.DataFrame, x: str, y: str, min_n: int = 12):
    """시즌별 스피어만 IC의 건수 가중 평균과 시즌별 값."""
    vals, ns = [], []
    per = {}
    for s, g in df.groupby("season"):
        g = g[[x, y]].dropna()
        if len(g) < min_n or g[x].nunique() < 3:
            continue
        ic = g[x].rank().corr(g[y].rank())
        per[s] = (round(ic, 3), len(g))
        vals.append(ic)
        ns.append(len(g))
    if not ns:
        return np.nan, 0, per
    return float(np.average(vals, weights=ns)), int(sum(ns)), per


def bool_gap(df: pd.DataFrame, x: str, y: str):
    """참/거짓 요인: 시즌 안 백분위 순위(0~100) 평균 차(참 − 거짓)와 건수."""
    d = df[[x, y, "season"]].dropna()
    if d.empty:
        return np.nan, 0, 0
    d = d.assign(p=d.groupby("season")[y].rank(pct=True) * 100)
    t, f = d[d[x].astype(bool)], d[~d[x].astype(bool)]
    if len(t) < 5 or len(f) < 5:
        return np.nan, len(t), len(f)
    return float(t.p.mean() - f.p.mean()), len(t), len(f)


def main() -> None:
    broad = "--broad" in sys.argv  # 2026 중 한 번이라도 보유한 종목 전부(보조, 사후 편입 편향 있음)
    extra = [Path(p) for p in sys.argv[1:] if p != "--broad"]
    df = load(extra, broad)
    tag = "_broad" if broad else ""
    skip = {"ticker", "reaction_date", "ann_date", "timing", "time_src", "k", "eps_est", "eps_rep", "snap", "season",
            "half", "group", "nx_end_k", "nx_complete", "last_k", "held_TIME", "held_KoAct", "held_any", "w_TIME", "w_KoAct",
            "eps_surprise"} | set(OUTS) | {c for c in df.columns if c in ("r0", "d5", "d20", "d40", "t5", "t20", "t40", "nx", "d5_x", "t5_x", "t40_x", "gap")}
    feats = [c for c in df.columns if c not in skip]
    rows = []
    for x in feats:
        isbool = df[x].dropna().isin([True, False, 0, 1]).all() and df[x].nunique() <= 2
        for y in OUTS:
            for half in ("H1", "H2"):
                d = df[df.half == half]
                if y == "nx_x":
                    d = d[d.nx_complete]
                if isbool:
                    gap, nt, nf = bool_gap(d, x, y)
                    rows.append({"feature": x, "outcome": y, "half": half, "kind": "bool", "stat": gap, "n": nt + nf, "n_true": nt})
                else:
                    ic, nn, per = season_ic(d, x, y)
                    rows.append({"feature": x, "outcome": y, "half": half, "kind": "ic", "stat": ic, "n": nn,
                                 "per_season": json.dumps(per)})
    ic = pd.DataFrame(rows)
    ic.to_csv(BASE / f"results/ic{tag}.csv", index=False)
    wide = ic.pivot_table(index=["feature", "kind"], columns=["outcome", "half"], values="stat").round(3)
    print(wide[[("d20_x", "H1"), ("d20_x", "H2"), ("d5_x", "H1"), ("d5_x", "H2"), ("r0_x", "H1"), ("r0_x", "H2"), ("t20_x", "H1"), ("t20_x", "H2")]].to_string())
    # 상위·하위 1/3 단면(주 결과)
    d = df[df.d20_x.notna()].copy()
    d["tile"] = d.groupby("season").d20_x.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["하", "중", "상"]))
    num = [c for c in feats if pd.api.types.is_numeric_dtype(d[c]) and d[c].nunique() > 2]
    prof = d.groupby("tile", observed=True)[num + ["d20_x", "r0_x"]].median().T
    boo = [c for c in feats if c not in num]
    prof_b = d.groupby("tile", observed=True)[boo].mean().T if boo else pd.DataFrame()
    pd.concat([prof, prof_b]).to_csv(BASE / f"results/profile{tag}.csv")
    print(pd.concat([prof, prof_b]).round(3).to_string())
    summary = {"events_held": int(len(df)), "by_season": df.groupby("season").size().to_dict(),
               "d20_complete": df.groupby("season").d20_x.count().to_dict(), "features": len(feats), "outcomes": OUTS}
    (BASE / f"results/analysis{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
