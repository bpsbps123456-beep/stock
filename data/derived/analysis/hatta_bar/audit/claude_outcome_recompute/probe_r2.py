"""Diagnose r2_e2 mismatches: test stop-rule variants on all 400 sampled rows (prints agreement counts only)."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute as RC
from probe_bv2 import arr, S

def r2(a, t, stop, j0, cmp_le=False, stop_first=True, tgt_close=False, gap_fill=True, entry="c1", cost=0.0, k_stop=None):
    last = a["lastw"][t]
    o, h, l, c = a["o"], a["h"], a["l"], a["c"]
    e = c[t + 1] if entry == "c1" else o[t + 1]
    tgt = e * 1.01
    for j in range(j0, last + 1):
        hit_s = (l[j] <= stop) if cmp_le else (l[j] < stop)
        hit_t = h[j] >= tgt
        if hit_s and (stop_first or not hit_t):
            fill = (o[j] if (gap_fill and o[j] < stop) else stop)
            return fill / e - 1.0
        if hit_t:
            return 0.01
    return c[last] / e - 1.0

variants = {
    "spec": dict(),
    "stop_le": dict(cmp_le=True),
    "no_gapfill": dict(gap_fill=False),
    "from_t+1": dict(j0off=1),
}
res = {}
rows = []
for i, r in S.iterrows():
    sym = RC.SYMS[int(r.sid)]; a = arr(sym); t = int(r.t)
    A = r.table.startswith("A")
    stops = {"lowt": a["l"][t], "minlow_k_t": min(a["l"][t - 1], a["l"][t]), "lowk": a["l"][t - 1],
             "minlow_t_t1": min(a["l"][t], a["l"][t + 1]), "minlow_k_t_t1": min(a["l"][t - 1], a["l"][t], a["l"][t + 1])}
    spec_stop = stops["lowt"] if A else stops["minlow_k_t"]
    rec = {"i": i, "kind": "A" if A else "B", "tab": r.r2_e2}
    for sname, sv in stops.items():
        for vname, kw in variants.items():
            kw = dict(kw); off = kw.pop("j0off", 2)
            rec[f"{sname}|{vname}"] = r2(a, t, sv, t + off, **kw)
    # entry below stop?
    rec["entry_below_spec_stop"] = a["c"][t + 1] < spec_stop
    rec["t1_low_below_stop"] = a["l"][t + 1] < spec_stop
    rows.append(rec)
D = pd.DataFrame(rows)
cols = [c for c in D.columns if "|" in c]
for kind in ("A", "B"):
    sub = D[D.kind == kind]
    out = {c: int((~((sub[c] - sub.tab).abs() <= 1e-9)).sum()) for c in cols}
    print(kind, len(sub), sorted(out.items(), key=lambda kv: kv[1])[:8])
spec_col = np.where(D.kind == "A", D["lowt|spec"], D["minlow_k_t|spec"])
mm = ~(np.abs(spec_col - D.tab) <= 1e-9)
print("spec mismatches", int(mm.sum()), "of which t+1 low below stop:", int((mm & D.t1_low_below_stop).sum()),
      "; t+1 low below stop overall:", int(D.t1_low_below_stop.sum()), "entry below stop:", int(D.entry_below_spec_stop.sum()))
D.to_csv(os.path.join(RC.HERE, "out", "r2_probe.csv"), index=False)
