"""Pass 7 (exploratory): cost sensitivity of H2 (2026H1 main, 2025H1 aux).

The pre-registered cost model uses the MEDIAN half-spread per (cap tier x time-of-day) cell.  Expected cost is
the MEAN.  Recompute H2 net R1 with (a) per-cell mean half-spread from the same quote sample, (b) uniform
multipliers on the median model, and report day-cluster bootstrap p (one-sided, B=2000).
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar/audit/h3-recompute")
import h3r_core as C  # noqa: E402
from s2_compare import day_boot, dedup  # noqa: E402


def load_h2(ds):
    if ds == "2026h1":
        return pd.read_parquet(C.OUT + "/h2_mine.parquet")
    sig = pd.read_parquet(C.OUT + "/sig1_2025h1.parquet")
    sig = sig[(sig.day >= 20250102) & (sig.day <= 20250623) & C.in_window(sig["mins"].to_numpy())]
    return dedup(sig[sig.vgap < -0.03])


def main():
    q = pd.read_parquet(C.BASE + "/cache/cost_quotes.parquet")
    cm = C.load_cost()
    q["cell"] = q["tier"] + "|" + q["tod"]
    cell_mean = q.groupby("cell")["half"].mean().to_dict()
    tod_mean = q.groupby("tod")["half"].mean().to_dict()
    cell_med = q.groupby("cell")["half"].median().to_dict()
    res = dict(check_median_matches_model=all(np.isclose(cell_med[k], v["half"]) for k, v in cm["table"].items()
                                              if k in cell_med and v["n"] >= 5),
               cell_mean_over_median={k: round(cell_mean[k] / cm["table"][k]["half"], 3) for k in cm["table"]
                                      if k in cell_mean})
    rng = np.random.default_rng(11)
    # bootstrap CI of mean ratio half/cell-median for quotes at VWAP gap < -3 % (vgap from s5)
    s5 = None
    try:
        with open(C.OUT + "/s5_result.json", encoding="utf-8") as f:
            s5 = json.load(f)
    except OSError:
        pass
    res["s5_ratio_mean_lowvgap"] = s5["vgap<-3%"]["ratio_mean"] if s5 else None

    def cost_mean(tier, tod):
        k = f"{tier}|{tod}"
        if tier is not None and k in cm["table"] and k in cell_mean and cm["table"][k]["n"] >= 5:
            return cell_mean[k]
        return tod_mean.get(tod, cm["by_tod"][tod])

    for ds in ("2026h1", "2025h1"):
        h2 = load_h2(ds).copy()
        charged = h2.c_in + np.where(h2.hit, 0.0, h2.c_out)
        r = {}
        r["median_model"] = day_boot(h2.assign(x=h2.r1 - charged), "x")
        cin = np.array([cost_mean(t, C.tod_in(m)) for t, m in zip(h2.tier, h2.mins)])
        cout = np.array([cost_mean(t, C.tod_out(m)) for t, m in zip(h2.tier, h2.z_mins)])
        r["mean_model"] = day_boot(h2.assign(x=h2.r1 - cin - np.where(h2.hit, 0.0, cout)), "x")
        for mult in (1.1, 1.27, 1.45, 2.0):
            r[f"median_x{mult}"] = day_boot(h2.assign(x=h2.r1 - mult * charged), "x")
        r["breakeven_multiplier"] = float(h2.r1.mean() / charged.mean())
        res[ds] = r
    with open(C.OUT + "/s7_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
