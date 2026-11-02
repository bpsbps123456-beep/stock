"""Pass 5 (exploratory): is the cost model's half-spread representative for H2-type moments?

Uses only the cached quote sample (cache/cost_quotes.parquet, 2026-07/08 signal moments) and cached
Jul-Oct 2026 1-minute bars (cache/alp1m).  For each quoted signal minute, recompute the 04:00 day-VWAP gap
of that bar and compare half-spread / (tier x tod cell median) for VWAP gap < -3 % versus the rest.
Also reports the break-even cost multiplier for my rebuilt H2.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar/audit/h3-recompute")
import h3r_core as C  # noqa: E402


def main():
    q = pd.read_parquet(C.BASE + "/cache/cost_quotes.parquet")
    cm = C.load_cost()
    C.CACHE = C.BASE + "/cache/alp1m"
    vg, rng_pct, ntr = [], [], []
    for sym, g in q.groupby("sym"):
        d = C.load_bars(sym)
        vw = C.day_vwap(d)
        pos = np.searchsorted(d["ts"], g["ts"].to_numpy())
        for i, p in zip(g.index, pos):
            if p < len(d["ts"]) and d["ts"][p] == q.at[i, "ts"]:
                vg.append((i, d["c"][p] / vw[p] - 1, (d["h"][p] - d["l"][p]) / d["c"][p], d["n"][p]))
    v = pd.DataFrame(vg, columns=["i", "vgap", "rng", "trades"]).set_index("i")
    q = q.join(v)
    q["cell"] = q["tier"] + "|" + q["tod"]
    q["cell_med"] = q["cell"].map(lambda k: cm["table"][k]["half"] if k in cm["table"] else np.nan)
    q["ratio"] = q["half"] / q["cell_med"]
    res = dict(n_quotes=int(len(q)), matched_bar=int(q["vgap"].notna().sum()))
    for lab, g in [("vgap<-3%", q[q.vgap < -0.03]), ("vgap>=-3%", q[q.vgap >= -0.03]),
                   ("vgap<-3%_tod_am", q[(q.vgap < -0.03) & (q.tod == "am")]),
                   ("vgap>=-3%_tod_am", q[(q.vgap >= -0.03) & (q.tod == "am")])]:
        res[lab] = dict(n=int(len(g)), ratio_median=float(g.ratio.median()), ratio_mean=float(g.ratio.mean()),
                        half_median=float(g.half.median()),
                        tier_mix=g["tier"].value_counts().to_dict(), tod_mix=g["tod"].value_counts().to_dict())
    # within-cell comparison (pooled over cells): median of (ratio | vgap<-3%) per cell
    rows = []
    for cell, g in q.groupby("cell"):
        a, b = g[g.vgap < -0.03].ratio, g[g.vgap >= -0.03].ratio
        if len(a) >= 3:
            rows.append(dict(cell=cell, n_lo=len(a), med_lo=float(a.median()), n_hi=len(b), med_hi=float(b.median())))
    res["by_cell_with_ge3_lowvgap"] = rows
    # Spearman-like check: rank correlation of ratio with vgap
    res["spearman_ratio_vs_vgap"] = float(q[["ratio", "vgap"]].rank().corr().iloc[0, 1])
    # break-even multiplier for my H2
    h2 = pd.read_parquet(C.OUT + "/h2_mine.parquet")
    charged = h2.c_in + np.where(h2.hit, 0.0, h2.c_out)
    res["H2_mean_r1"] = float(h2.r1.mean())
    res["H2_mean_cost_charged"] = float(charged.mean())
    res["H2_breakeven_cost_multiplier"] = float(h2.r1.mean() / charged.mean())
    for mult in (1.5, 2.0):
        res[f"H2_net_if_cost_x{mult}"] = float((h2.r1 - mult * charged).mean())
    with open(C.OUT + "/s5_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
