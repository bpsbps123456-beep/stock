"""Count how many signals hinge on BRIEF ambiguities (no outcome data used)."""
import os

import numpy as np
import pandas as pd

import detect as D

OUT = D.OUT
eng = pd.read_csv(os.path.join(D.ROOT, "cache", "engine_signals_yf1m.csv"))
diag = pd.read_csv(os.path.join(OUT, "diag.csv"))
cellp = {c["name"]: c for c in D.CELLS}

res = {}
# (c) volume-ratio denominator zero (vr = inf) among v>=2 cells
for cname in ("alt_A1", "alt_B1"):
    sub = diag[(diag.cell == cname) & diag.pass_c]
    res[f"vr_inf_{cname}"] = int(np.isinf(sub.vr).sum())
    sub_all = diag[(diag.cell == cname)]
    res[f"vr_nan_cand_{cname}"] = int(sub_all.vr.isna().sum())

# (d) BRIEF hygiene 'next bar starts within 2 min after signal close' on engine dump signals
gap_fail = {c: 0 for c in cellp}
float_edge = {c: 0 for c in cellp}
pos_rule_diff = {c: 0 for c in cellp}
early = {c: 0 for c in cellp}
for sym, g in eng.groupby("sym"):
    d = D.load(sym)
    idx = d.index
    l = d["Low"].to_numpy(float)
    n = len(d)
    for _, r in g.iterrows():
        cp = cellp[r.cell]
        k = int(r.k)
        t = k if cp["pat"] == "A" else k + 1
        if t + 1 >= n or (idx[t + 1] - idx[t]) > pd.Timedelta(minutes=3):
            gap_fail[r.cell] += 1
        if k < 400:
            early[r.cell] += 1

# (b) 'S position' reading: add S unless the most recent argmin bar is itself a confirmed swing
# (f) float edges: break-test operands within 1e-9 relative of threshold
for sym in sorted(set(diag.sym)):
    d = D.load(sym)
    l = d["Low"].to_numpy(float)
    sws = {}
    sub = diag[(diag.sym == sym) & diag.pass_c]
    for _, r in sub.iterrows():
        cp = cellp[r.cell]
        k, N, p, tol = int(r.k), cp["N"], cp["p"], cp["tol"]
        S = l[k - N:k - 1].min()
        thr = S * (1 - tol)
        if min(abs(l[k - 1] - thr), abs(l[k] - thr)) <= 1e-9 * S:
            float_edge[r.cell] += 1
        if cp["kmode"] != "touch":
            continue
        if p not in sws:
            sws[p] = D.swing_mask(l, p)
        sw = sws[p]
        w = np.arange(k - N, k - 1)
        last_arg = w[np.nonzero(l[w] == S)[0][-1]]
        js = np.nonzero(sw[k - N:min(k - 2, k - 1 - p) + 1])[0] + (k - N)
        pts = list(l[js])
        if last_arg not in set(js):
            pts.append(S)
        touch_pos = int(np.sum(np.array(pts) <= S * 1.003))
        if (touch_pos >= cp["K"]) != (r.touch >= cp["K"]) or touch_pos != r.touch:
            pos_rule_diff[r.cell] += 1

print("vol-ratio:", res)
print("next-bar >2min after close (BRIEF hygiene, not in dump def):", gap_fail)
print("float-edge break tests:", float_edge)
print("touch value differs under 'last argmin position' reading (touch cells):", pos_rule_diff)
print("signals with k<400 (warm-up region):", early)
