"""Probe which bar-validity definition reproduces the tables' n_same_min (count of valid bars at that minute)."""
import os, sys, glob
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute as RC

per = {}
for sid, sym in enumerate(RC.SYMS):
    a = RC.arrays(RC.load_raw(sym))
    res = RC.bar_out(a["ts"], a["day"], a["sess"], a["o"], a["h"], a["l"], a["c"], RC.COST)
    a["v0"], a["v2"], a["lastw"] = res[0], res[1], res[10]
    n = len(a["ts"])
    nxt = np.full(n, 10**9); nxt[:-1] = a["ts"][1:] - a["ts"][:-1]
    nxt2 = np.full(n, 10**9); nxt2[:-2] = a["ts"][2:] - a["ts"][:-2]
    a["gap1"] = nxt; a["gap2"] = nxt2
    st = pd.Series(a["h"], index=pd.to_datetime(a["ts"] * 60, unit="s"))
    a["dens"] = st.rolling("30min").count().to_numpy()
    nbar = np.where(a["lastw"] >= 0, a["lastw"] - np.arange(n), 0)
    a["nwin"] = nbar
    per[sym] = a

defs = {
    "v0&v2": lambda a: a["v0"] & a["v2"],
    "v0&v2&gap1<=2": lambda a: a["v0"] & a["v2"] & (a["gap1"] <= 2),
    "v0&v2&gap1<=3": lambda a: a["v0"] & a["v2"] & (a["gap1"] <= 3),
    "v0&v2&gap1<=1": lambda a: a["v0"] & a["v2"] & (a["gap1"] <= 1),
    "v0&v2&dens>=15": lambda a: a["v0"] & a["v2"] & (a["dens"] >= 15),
    "v0&v2&gap2<=3": lambda a: a["v0"] & a["v2"] & (a["gap2"] <= 3),
    "v0&v2&gap2<=4": lambda a: a["v0"] & a["v2"] & (a["gap2"] <= 4),
}
counts = {}
for name, fn in defs.items():
    rows = []
    for sid, sym in enumerate(RC.SYMS):
        a = per[sym]
        rows.append(pd.Series(a["ts"][fn(a)]))
    counts[name] = pd.concat(rows).value_counts()

tabs = sorted(glob.glob(os.path.join(RC.CAND, "A_*.parquet")) + glob.glob(os.path.join(RC.CAND, "B_*.parquet")))
X = pd.concat([pd.read_parquet(f, columns=["sid", "t", "ts", "n_same_min"]) for f in tabs], ignore_index=True)
X = X.drop_duplicates(["sid", "t"])
print("distinct candidate (sid,t):", len(X))
for name, cnt in counts.items():
    mine = cnt.reindex(X.ts.to_numpy()).fillna(0).to_numpy()
    print(f"{name:20s} n_same_min mismatches: {(mine != X.n_same_min.to_numpy()).sum()}  (mine>table {(mine > X.n_same_min.to_numpy()).sum()})")
