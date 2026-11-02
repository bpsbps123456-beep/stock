"""Pass 4: same-minute other-symbol baseline (criterion 2) for my rebuilt H2.

Baseline for a signal (sym, ts) = mean E2 R1 (pre-cost) over every OTHER universe symbol that has a bar
starting exactly at ts which is a valid E2 bar (next bar within 3 min same date, >=1 bar after the entry
bar inside the 30-min window).  vs = R1(signal) - baseline.  (My reading of BRIEF.md comparison 3 + v2.1.)
Usage: py s4_samemin.py [2026h1|2025h1]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar/audit/h3-recompute")
import h3r_core as C  # noqa: E402
from s2_compare import day_boot, dedup  # noqa: E402


def light(d):
    ts, day = d["ts"], d["day"]
    n = len(ts)
    nxt = np.zeros(n, dtype=bool)
    nxt[:-1] = ((ts[1:] - ts[:-1]) <= 3) & (day[1:] == day[:-1])
    z = np.searchsorted(ts, ts + 30, side="right") - 1
    e2ok = (z >= np.arange(n) + 2) & (day[np.clip(z, 0, n - 1)] == day)
    return nxt, e2ok, z


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "2026h1"
    C.CACHE = C.BASE + "/cache/alp1m_" + ds
    suffix = "" if ds == "2026h1" else "_" + ds
    if ds == "2026h1":
        h2 = pd.read_parquet(C.OUT + "/h2_mine.parquet")
    else:
        sig = pd.read_parquet(C.OUT + f"/sig1{suffix}.parquet")
        sig = sig[(sig.day >= 20250102) & (sig.day <= 20250623) & C.in_window(sig["mins"].to_numpy())]
        h2 = dedup(sig[sig.vgap < -0.03])
    Ms = np.unique(h2["ts"].to_numpy())
    uni = C.load_universe()
    have = set(f[:-8] for f in os.listdir(C.CACHE))
    rows = []
    for k, sym in enumerate(sorted(uni)):
        if sym not in have:
            continue
        d = C.load_bars(sym)
        nxt, e2ok, z = light(d)
        n = len(d["ts"])
        pos = np.searchsorted(d["ts"], Ms)
        ok = pos < n
        pos_ok = pos[ok]
        match = d["ts"][pos_ok] == Ms[ok]
        for t in pos_ok[match]:
            if nxt[t] and e2ok[t]:
                entry, hit, r1, _ = C.e2_outcome(d, int(t), int(z[t]))
                rows.append((sym, int(d["ts"][t]), r1, hit))
        if k % 100 == 0:
            print(k, sym, len(rows), flush=True)
    bt = pd.DataFrame(rows, columns=["sym", "ts", "r1", "hit"])
    bt.to_parquet(C.OUT + f"/samemin_rows{suffix}.parquet")
    g = bt.groupby("ts")["r1"].agg(["sum", "count"])
    h2 = h2.copy()
    h2 = h2.merge(g, left_on="ts", right_index=True, how="left")
    own = bt.set_index(["sym", "ts"])["r1"]
    own_r1 = [own.get((s, t), np.nan) for s, t in zip(h2.sym, h2.ts)]
    h2["own_in_bt"] = ~np.isnan(own_r1)
    h2["own_r1"] = own_r1
    s_oth = h2["sum"] - np.where(h2.own_in_bt, h2.own_r1, 0.0)
    n_oth = h2["count"] - h2.own_in_bt.astype(int)
    h2["bt"] = s_oth / n_oth
    h2["n_oth"] = n_oth
    h2["vs_bt"] = h2["r1"] - h2["bt"]
    ok = h2["vs_bt"].notna()
    res = dict(n=int(len(h2)), n_with_baseline=int(ok.sum()),
               own_bar_matches_signal_r1=int(np.sum(np.isclose(h2.loc[h2.own_in_bt, "own_r1"],
                                                                h2.loc[h2.own_in_bt, "r1"]))),
               n_oth_median=float(h2["n_oth"].median()), n_oth_min=int(h2["n_oth"].min()),
               baseline_mean=float(h2.loc[ok, "bt"].mean()), r1_mean=float(h2.loc[ok, "r1"].mean()),
               vs_bt_mean=float(h2.loc[ok, "vs_bt"].mean()),
               vs_bt_boot=day_boot(h2[ok], "vs_bt"))
    # by time-of-day (exploratory)
    h2["tod4"] = h2["mins"].map(C.tod_in)
    res["by_tod"] = {k: dict(n=int(len(g_)), vs_bt=float(g_.vs_bt.mean()), bt=float(g_.bt.mean()))
                     for k, g_ in h2[ok].groupby("tod4")}
    with open(C.OUT + f"/s4_result{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    h2.to_parquet(C.OUT + f"/h2_vsbt{suffix}.parquet")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
