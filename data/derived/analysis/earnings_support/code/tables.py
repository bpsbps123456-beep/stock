"""보고서 표: 주 설정(Z1~Z6 하단·창 B) × 구간, N2 귀무 분위, 체결 비율이 가장 가까운 N1, N0.

실행: py data/derived/analysis/earnings_support/code/tables.py [split]
출력: results/main_table.csv, results/robust_table.csv, 화면 요약
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lib import RES  # noqa: E402

PERIODS = ["2025", "2026H1", "2026H2"]


def pct_rank(actual: float, null: np.ndarray) -> float:
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(actual):
        return np.nan
    return float(((null < actual).sum() + 0.5 * (null == actual).sum()) / len(null))


def build(split: str = "all", window: str = "B", side: str = "bot") -> pd.DataFrame:
    s = pd.read_csv(RES / "summary_all.csv")
    n2 = pd.read_csv(RES / "null_n2.csv")
    s = s[s.split == split]
    n2 = n2[n2.split == split]
    rows = []
    for k in range(1, 7):
        for p in PERIODS:
            row = {"level": f"Z{k}", "period": p}
            for x in ("x1", "x2", "x4", "x5"):
                a = s[(s.strategy == f"Z{k}_{side}") & (s.window == window) & (s.period == p) & (s.exit == x)]
                if a.empty:
                    continue
                a = a.iloc[0]
                if x == "x1":
                    row.update(n=int(a.n_events), fill=a.fill_rate, win_x1=a.win, mean_x1=a["mean"],
                               med_x1=a["median"], t_x1=a.t, p5_x1=a.p5, ev_x1=a.per_event, ev_t_x1=a.per_event_t)
                    nn = n2[(n2.k == k) & (n2.window == window) & (n2.period == p) & (n2.exit == "x1")]
                    row.update(n2_mean_pct=pct_rank(a["mean"], nn["mean"].to_numpy()),
                               n2_win_pct=pct_rank(a.win, nn["win"].to_numpy()),
                               n2_ev_pct=pct_rank(a.per_event, nn["per_event"].to_numpy()),
                               n2_mean_med=nn["mean"].median(), n2_win_med=nn["win"].median())
                    m = s[s.strategy.str.startswith("N1_") & (s.window == window) & (s.period == p) & (s.exit == "x1")]
                    if not m.empty:
                        j = (m.fill_rate - a.fill_rate).abs().idxmin()
                        row.update(n1_match=m.at[j, "strategy"], n1_fill=m.at[j, "fill_rate"], n1_win=m.at[j, "win"],
                                   n1_mean=m.at[j, "mean"])
                else:
                    row.update({f"win_{x}": a.win, f"mean_{x}": a["mean"], f"t_{x}": a.t, f"ev_{x}": a.per_event})
            rows.append(row)
    return pd.DataFrame(rows)


def n0(split: str = "all") -> pd.DataFrame:
    s = pd.read_csv(RES / "summary_all.csv")
    s = s[(s.split == split) & (s.strategy == "N0")]
    return s[s.period.isin(PERIODS + ["2026"])][["period", "exit", "n_events", "win", "mean", "median", "t", "p5"]]


def main(split: str = "all") -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    t = build(split)
    tag = "" if split == "all" else "_" + split.replace(">=", "ge")
    t.to_csv(RES / f"main_table{tag}.csv", index=False)
    cols = ["level", "period", "n", "fill", "win_x1", "mean_x1", "t_x1", "ev_x1", "n2_mean_pct", "n2_win_pct",
            "n1_match", "n1_win", "n1_mean", "win_x2", "mean_x2", "win_x4", "mean_x4", "mean_x5"]
    print(t[[c for c in cols if c in t]].round(4).to_string())
    print(n0(split).round(4).to_string())
    rob = []
    for w in ("A", "B", "C"):
        for side in ("bot", "top"):
            b = build(split, w, side)
            b.insert(0, "window", w)
            b.insert(1, "side", side)
            rob.append(b)
    pd.concat(rob).to_csv(RES / f"robust_table{tag}.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
