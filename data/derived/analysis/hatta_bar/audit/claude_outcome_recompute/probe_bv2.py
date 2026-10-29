"""Brute-force tercile base sets for bv on all 400 sampled rows."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute as RC

OUT = os.path.join(RC.HERE, "out")
S = pd.read_csv(os.path.join(OUT, "sample.csv"))
cache = {}
def arr(sym):
    if sym in cache: return cache[sym]
    a = RC.arrays(RC.load_raw(sym))
    res = RC.bar_out(a["ts"], a["day"], a["sess"], a["o"], a["h"], a["l"], a["c"], RC.COST)
    for k, v in zip(["v0", "v2", "r1_e0", "hit_e0", "race_e0", "r1_e2", "hit_e2", "race_e2", "hits_e2", "r1net_e2", "lastw"], res):
        a[k] = v
    gap1 = np.full(len(a["ts"]), 10**9); gap1[:-1] = a["ts"][1:] - a["ts"][:-1]
    a["vg"] = gap1 <= 3
    a["atr"] = RC.atr14(a["h"], a["l"], a["c"]); a["atrp"] = a["atr"] / a["c"]
    rng = a["h"] - a["l"]
    a["onept"] = (a["o"] == a["h"]) & (a["h"] == a["l"]) & (a["l"] == a["c"])
    vol = RC.load_raw(sym)["Volume"].to_numpy(float)
    a["sane"] = ~a["onept"] & (vol * a["c"] >= 25000) & (rng >= 0.002 * a["c"])
    cache[sym] = a
    return a

bases = {
    "all": lambda a: np.ones(len(a["ts"]), bool),
    "v0": lambda a: a["v0"], "v2": lambda a: a["v2"], "vg": lambda a: a["vg"],
    "v0vg": lambda a: a["v0"] & a["vg"], "v2vg": lambda a: a["v2"] & a["vg"],
    "notonept": lambda a: ~a["onept"],
}
res = {k: 0 for k in bases}
bad = {k: [] for k in bases}
for i, r in S.iterrows():
    sym = RC.SYMS[int(r.sid)]; a = arr(sym); t = int(r.t)
    g0 = (a["day"] == a["day"][t]) & (a["sess"] == a["sess"][t]) & ~np.isnan(a["atrp"])
    for k, fn in bases.items():
        g = np.where(g0 & fn(a))[0]
        if t not in set(g.tolist()):
            res[k] += 1; bad[k].append(i); continue
        pct = pd.Series(a["atrp"][g]).rank(pct=True).to_numpy()
        terc = np.minimum(np.floor(pct * 3), 2)
        sel = g[terc == terc[np.searchsorted(g, t)]]
        vm = a["v2"][sel] & a["vg"][sel]
        mine = np.nanmean(a["r1_e2"][sel][vm])
        if not abs(mine - r.bv_r1_e2) <= 1e-9:
            res[k] += 1; bad[k].append(i)
for k in bases:
    print(k, res[k], bad[k][:12])
