"""For the residual bv mismatches, search single-member add/remove that reproduces the table bv (r1_e2 and hit_e2 together)."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute as RC
from probe_bv2 import arr, S

def hhmm(m): return f"{m//60:02d}:{m%60:02d}"
for i in [53, 69, 74, 95, 202, 208, 280, 316, 363]:
    r = S.loc[i]; sym = RC.SYMS[int(r.sid)]; a = arr(sym); t = int(r.t)
    gall = np.where((a["day"] == a["day"][t]) & (a["sess"] == a["sess"][t]))[0]
    g = gall[~np.isnan(a["atrp"][gall])]
    pct = pd.Series(a["atrp"][g]).rank(pct=True).to_numpy()
    terc = np.minimum(np.floor(pct * 3), 2)
    mt = terc[np.searchsorted(g, t)]
    sel = g[terc == mt]
    mem = sel[a["v2"][sel] & a["vg"][sel]]
    base_r1 = a["r1_e2"][mem]; base_hit = a["hit_e2"][mem]
    found = []
    for j in mem:   # remove one
        keep = mem[mem != j]
        if abs(np.nanmean(a["r1_e2"][keep]) - r.bv_r1_e2) <= 1e-9 and abs(np.nanmean(a["hit_e2"][keep]) - r.bv_hit_e2) <= 1e-9:
            found.append(("remove", j))
    others = g[(terc == mt) & ~(a["v2"][g] & a["vg"][g])]
    for j in others:   # add an invalid same-tercile bar (if it has an e2 value)
        if np.isnan(a["r1_e2"][j]): continue
        ad = np.r_[mem, j]
        if abs(np.nanmean(a["r1_e2"][ad]) - r.bv_r1_e2) <= 1e-9:
            found.append(("add_invalid_same_terc", j))
    for j in g[(terc != mt) & a["v2"][g] & a["vg"][g]]:
        ad = np.r_[mem, j]
        if abs(np.nanmean(a["r1_e2"][ad]) - r.bv_r1_e2) <= 1e-9:
            found.append(("add_other_terc", j))
    # one member's r1 changed: implied value
    implied = r.bv_r1_e2 * len(mem) - (base_r1.sum())
    desc = []
    for kind, j in found:
        desc.append(f"{kind} j={j} {hhmm(a['mins'][j])} gap_next={a['ts'][j+1]-a['ts'][j] if j+1 < len(a['ts']) else None} "
                    f"lastw-j={a['lastw'][j]-j} t+30 end={hhmm(a['mins'][j]+30)}")
    print(f"#{i} {sym} day={a['day'][t]} sess={a['sess'][t]} t@{hhmm(a['mins'][t])} members={len(mem)} "
          f"day bars={len(gall)} last bar of day @{hhmm(a['mins'][gall[-1]])}; implied single-r1 delta={implied:.3e}")
    for d in desc: print("    ", d)
    if not found:
        print("     no single add/remove explains it")
