"""Test extra member-exclusion hypotheses for bv over all 400 sampled rows."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute as RC
from probe_bv2 import arr, S

def extra(a):
    n = len(a["ts"]); idx = np.arange(n)
    # last bar index of each day
    day = a["day"]; last_of_day = np.zeros(n, np.int64)
    ends = np.r_[np.where(day[1:] != day[:-1])[0], n - 1]
    starts = np.r_[0, ends[:-1] + 1]
    for s, e in zip(starts, ends): last_of_day[s:e + 1] = e
    lw = a["lastw"]
    nwin = np.where(lw >= 0, lw - idx, 0)
    win_end = a["mins"] + 30
    last_min = a["mins"][last_of_day]
    sessw = np.where(lw >= 0, a["sess"][np.maximum(lw, 0)], -1)
    return dict(
        none=np.ones(n, bool),
        win_complete_vs_daylast=win_end <= last_min,
        win_before_2000=win_end <= 1199,
        mins_lt_1930=a["mins"] < 1170,
        same_sess_window=sessw == a["sess"],
        nwin_ge3=nwin >= 3, nwin_ge5=nwin >= 5,
        lastbar_not_daylast=lw != last_of_day,
    )

cnt = {}; bad = {}
for i, r in S.iterrows():
    sym = RC.SYMS[int(r.sid)]; a = arr(sym); t = int(r.t)
    if "ex" not in a: a["ex"] = extra(a)
    gall = np.where((a["day"] == a["day"][t]) & (a["sess"] == a["sess"][t]))[0]
    g = gall[~np.isnan(a["atrp"][gall])]
    pct = pd.Series(a["atrp"][g]).rank(pct=True).to_numpy()
    terc = np.minimum(np.floor(pct * 3), 2)
    sel = g[terc == terc[np.searchsorted(g, t)]]
    for name, ok in a["ex"].items():
        vm = a["v2"][sel] & a["vg"][sel] & ok[sel]
        res = []
        for f in ("r1_e2", "hit_e2", "race_e2"):
            v = a[f][sel][vm]
            mine = np.nanmean(v) if np.isfinite(v).any() else np.nan
            tab = r[f"bv_{f}"]
            res.append((np.isnan(mine) and np.isnan(tab)) or abs(mine - tab) <= 1e-9)
        good = all(res)
        cnt[name] = cnt.get(name, 0) + (not good)
        if not good: bad.setdefault(name, []).append(i)
for k, v in sorted(cnt.items(), key=lambda kv: kv[1]):
    print(f"{k:26s} {v:4d} {bad.get(k, [])[:12]}")
