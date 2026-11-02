"""Pass 8 (exploratory): symbol-cluster bootstrap, strict touch x mean-cost combination for my rebuilt H2."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar/audit/h3-recompute")
import h3r_core as C  # noqa: E402
from s2_compare import day_boot  # noqa: E402


def cluster_boot(df, col, key, B=2000, seed=5):
    g = df.groupby(key)[col].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(B, len(s)))
    bm = s[idx].sum(1) / n[idx].sum(1)
    return dict(mean=float(s.sum() / n.sum()), se=float(bm.std(ddof=1)), p=float((np.sum(bm <= 0) + 1) / (B + 1)))


def main():
    h2 = pd.read_parquet(C.OUT + "/h2_mine_feat.parquet")
    q = pd.read_parquet(C.BASE + "/cache/cost_quotes.parquet")
    q["cell"] = q["tier"] + "|" + q["tod"]
    cm = C.load_cost()
    cell_mean = q.groupby("cell")["half"].mean().to_dict()
    tod_mean = q.groupby("tod")["half"].mean().to_dict()

    def cmean(tier, tod):
        k = f"{tier}|{tod}"
        if k in cm["table"] and cm["table"][k]["n"] >= 5 and k in cell_mean:
            return cell_mean[k]
        return tod_mean[tod]

    cin = np.array([cmean(t, C.tod_in(m)) for t, m in zip(h2.tier, h2.mins)])
    cout = np.array([cmean(t, C.tod_out(m)) for t, m in zip(h2.tier, h2.z_mins)])
    res = {}
    res["symbol_cluster_boot_r1n"] = cluster_boot(h2, "r1n", "sym")
    res["day_cluster_boot_r1n"] = day_boot(h2, "r1n")
    h2["x_strict_mean"] = h2.r1_strict - cin - np.where(h2.hit_strict, 0.0, cout)
    res["strict_touch_and_mean_cost_dayboot"] = day_boot(h2, "x_strict_mean")
    h2["x_strict_med"] = h2.r1n_strict
    res["strict_touch_median_cost_dayboot"] = day_boot(h2, "x_strict_med")
    res["strict_touch_median_cost_symboot"] = cluster_boot(h2, "x_strict_med", "sym")
    with open(C.OUT + "/s8_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
