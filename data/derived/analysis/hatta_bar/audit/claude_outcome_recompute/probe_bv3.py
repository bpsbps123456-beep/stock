"""Try alternative tercile assignment rules for bv over all 400 sampled rows (group base = all bars with ATR)."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute as RC
from probe_bv2 import arr, S

def rules(x, n_incl_nan):
    s = pd.Series(x)
    r_avg = s.rank().to_numpy(); r_first = s.rank(method="first").to_numpy(); n = len(x)
    out = {
        "floor(pct*3)": np.minimum(np.floor(r_avg / n * 3), 2),
        "floor((r-1)*3/n)": np.floor((r_avg - 1) * 3 / n),
        "floor((r-1)/(n-1)*3)": np.minimum(np.floor((r_avg - 1) / max(n - 1, 1) * 3), 2),
        "qcut(values)": pd.qcut(s, 3, labels=False, duplicates="drop").to_numpy().astype(float),
        "qcut(rank first)": pd.qcut(pd.Series(r_first), 3, labels=False).to_numpy().astype(float),
        "qcut(rank avg)": pd.qcut(pd.Series(r_avg), 3, labels=False, duplicates="drop").to_numpy().astype(float),
        "floor(r*3/(n+1))": np.floor(r_avg * 3 / (n + 1)),
        "floor(pct*3) n incl NaN": np.minimum(np.floor(r_avg / n_incl_nan * 3), 2),
        "quantile edges <": (x > np.quantile(x, 1/3)).astype(float) + (x > np.quantile(x, 2/3)).astype(float),
        "quantile edges <=": (x >= np.quantile(x, 1/3)).astype(float) + (x >= np.quantile(x, 2/3)).astype(float),
        "digitize nanpercentile": np.digitize(x, np.nanpercentile(x, [100/3, 200/3])).astype(float),
    }
    return out

cnt = {}; bad = {}
for i, r in S.iterrows():
    sym = RC.SYMS[int(r.sid)]; a = arr(sym); t = int(r.t)
    gall = np.where((a["day"] == a["day"][t]) & (a["sess"] == a["sess"][t]))[0]
    g = gall[~np.isnan(a["atrp"][gall])]
    for name, terc in rules(a["atrp"][g], len(gall)).items():
        sel = g[terc == terc[np.searchsorted(g, t)]]
        vm = a["v2"][sel] & a["vg"][sel]
        mine = np.nanmean(a["r1_e2"][sel][vm])
        ok = abs(mine - r.bv_r1_e2) <= 1e-9
        cnt[name] = cnt.get(name, 0) + (not ok)
        if not ok: bad.setdefault(name, []).append(i)
for k, v in sorted(cnt.items(), key=lambda kv: kv[1]):
    print(f"{k:28s} {v:4d} {bad.get(k, [])[:10]}")
