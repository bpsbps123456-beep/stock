"""Pass 6 (exploratory): universe-hindsight and same-symbol same-day drift checks on my rebuilt H2 (2026H1).

1. Universe was chosen by June-2026 trading value (BRIEF2/BRIEF3).  Split H2 by each symbol's own
   2026-01-02 -> 06-23 return (hindsight variable, explanation only, never a trading rule).
2. Same-symbol same-day baseline: mean E2 R1 of every valid bar of the same symbol and day inside the
   judgment window (04:00-10:59, 19:00-19:29).  vs_sd = R1(signal) - baseline.  Controls for symbol-day drift.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar/audit/h3-recompute")
import h3r_core as C  # noqa: E402
from s2_compare import day_boot, summ  # noqa: E402
from s4_samemin import light  # noqa: E402

N_SLICES = 0


def S(df, label, out, col="r1n"):
    global N_SLICES
    N_SLICES += 1
    r = summ(df, col)
    if len(df) >= 10 and df["day"].nunique() >= 3:
        b = day_boot(df, col)
        r["se"], r["p_one_sided"] = b["se"], b["p"]
    out[label] = r


def main():
    h2 = pd.read_parquet(C.OUT + "/h2_vsbt.parquet")
    out = {}
    # ---- 1. symbol H1 return (hindsight) ----
    daily = pd.read_parquet(C.BASE + "/cache/daily_2026h1.parquet")
    daily = daily[(daily.date >= "2026-01-02") & (daily.date <= "2026-06-23")].sort_values(["sym", "date"])
    g = daily.groupby("sym")["c"].agg(["first", "last"])
    ret = (g["last"] / g["first"] - 1).rename("h1ret")
    h2 = h2.merge(ret, left_on="sym", right_index=True, how="left")
    uni_ret = ret.reindex(sorted(C.load_universe()))
    out["universe_h1ret_median"] = float(uni_ret.median())
    out["H2_signal_weighted_h1ret_median"] = float(h2.h1ret.median())
    q1, q2 = uni_ret.quantile([1 / 3, 2 / 3])
    out["universe_tercile_cuts"] = [float(q1), float(q2)]
    h2["ret_terc"] = np.where(h2.h1ret <= q1, "T1_low", np.where(h2.h1ret <= q2, "T2_mid", "T3_high"))
    out["H2_share_by_universe_tercile"] = h2["ret_terc"].value_counts(normalize=True).round(4).to_dict()
    s1 = {}
    for k, gg in h2.groupby("ret_terc"):
        S(gg, k, s1)
    out["by_symbol_h1_return_tercile"] = s1
    bysym = h2.groupby("sym").agg(pnl=("r1n", "sum"), h1ret=("h1ret", "first"))
    out["spearman_sym_pnl_vs_h1ret"] = float(bysym.rank().corr().iloc[0, 1])
    s_ai = {}
    uni = C.load_universe()
    h2["ai"] = h2.sym.map(lambda s: uni[s]["ai"])
    for k, gg in h2.groupby("ai"):
        S(gg, f"ai={k}", s_ai)
    out["by_ai"] = s_ai

    # ---- 2. same-symbol same-day baseline ----
    rows = []
    for sym, gs in h2.groupby("sym"):
        d = C.load_bars(sym)
        nxt, e2ok, z = light(d)
        days = set(gs.day)
        for day in days:
            idx = np.nonzero((d["day"] == day) & C.in_window(d["mins"]) & nxt & e2ok)[0]
            r = [C.e2_outcome(d, int(t), int(z[t]))[2] for t in idx]
            rows.append((sym, day, float(np.mean(r)) if r else np.nan, len(r)))
    sd = pd.DataFrame(rows, columns=["sym", "day", "sd_bt", "sd_n"])
    h2 = h2.merge(sd, on=["sym", "day"], how="left")
    h2["vs_sd"] = h2["r1"] - h2["sd_bt"]
    ok = h2.vs_sd.notna()
    out["same_symbol_day"] = dict(n=int(ok.sum()), sd_bt_mean=float(h2.loc[ok, "sd_bt"].mean()),
                                  sd_n_median=float(h2.loc[ok, "sd_n"].median()),
                                  vs_sd_mean=float(h2.loc[ok, "vs_sd"].mean()),
                                  boot=day_boot(h2[ok], "vs_sd"))
    out["n_exploratory_slices_evaluated"] = N_SLICES
    with open(C.OUT + "/s6_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
