"""Per-field mismatch counts on the 400 sampled rows (out/sample.csv from recompute.py), LITERAL vs RESOLVED rules.
Agreement counts only."""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute as RC
import full_check as FC

S = pd.read_csv(os.path.join(RC.HERE, "out", "sample.csv"))
per = {}
mm = {"lit": [], "res": []}
for sid, sym in enumerate(RC.SYMS):
    a = FC.sym_arrays(sym)
    a["bv_lit"] = FC.bv_maps(a, "lit"); a["bv_res"] = FC.bv_maps(a, "res")
    per[sym] = a
    for mode, vb in (("lit", a["v0"] & a["v2"]), ("res", a["v0"] & a["v2"] & a["vg"] & a["late_ok"])):
        mm[mode].append(pd.DataFrame({"sid": sid, "ts": a["ts"][vb], **{f: a[f][vb] for f in FC.F_BT}}))
M = {k: pd.concat(v, ignore_index=True) for k, v in mm.items()}
fields = ["r1_e0", "hit_e0", "r1_e2", "r1net_e2", "hit_e2", "hits_e2", "race_e2",
          "bv_r1_e2", "bv_hit_e2", "bv_race_e2", "bv_r1net_e2", "bt_r1_e2", "bt_r1net_e2", "r2_e2"]
out = {f: {"lit": 0, "res": 0} for f in fields}
causes = {f: {} for f in fields}
for i, r in S.iterrows():
    sym = RC.SYMS[int(r.sid)]; a = per[sym]; t = int(r.t); kind = r.table[0]
    assert a["ts"][t] == int(r.ts)
    for mode in ("lit", "res"):
        val0 = a["v0"] & a["v2"] if mode == "lit" else a["v0"] & a["v2"] & a["vg"] & a["late_ok"]
        o_ = M[mode][(M[mode].ts == int(r.ts)) & (M[mode].sid != int(r.sid))]
        mine = {f: a[f][t] for f in ["r1_e0", "hit_e0", "r1_e2", "r1net_e2", "hit_e2", "hits_e2", "race_e2"]}
        for f in ["r1_e2", "hit_e2", "race_e2", "r1net_e2"]:
            mine[f"bv_{f}"] = a[f"bv_{mode}"][f"bv_{f}"][t]
        mine["bt_r1_e2"] = o_.r1_e2.mean(); mine["bt_r1net_e2"] = o_.r1net_e2.mean()
        mine["r2_e2"] = FC.r2_all(a, np.array([t]), kind, mode)[0]
        for f in fields:
            th, mv = float(r[f]), float(mine[f])
            if not ((np.isnan(th) and np.isnan(mv)) or abs(th - mv) <= RC.TOL):
                out[f][mode] += 1
                if mode == "lit":
                    # attribute literal mismatch to the inferred rule(s)
                    tags = []
                    if f == "r2_e2" and a["c"][t + 1] < (a["l"][t] if kind == "A" else min(a["l"][t - 1], a["l"][t])):
                        tags.append("entry<stop->0")
                    if f.startswith("bt_"):
                        tags.append("gap>3min invalid")
                    if f.startswith("bv_"):
                        g = (a["day"] == a["day"][t]) & (a["sess"] == a["sess"][t])
                        if (g & a["v2"] & ~a["vg"]).any(): tags.append("gap>3min invalid")
                        if (g & ~a["late_ok"]).any(): tags.append(">=19:30 invalid")
                        if (g & np.isnan(a["atrp"])).any(): tags.append("NaN-ATR->tercile0")
                    key = "+".join(tags) or "unattributed"
                    causes[f][key] = causes[f].get(key, 0) + 1
print(f"{'field':14s} {'literal':>8s} {'resolved':>8s}  literal-mismatch attribution")
for f in fields:
    print(f"{f:14s} {out[f]['lit']:8d} {out[f]['res']:8d}  {causes[f]}")
json.dump({"counts": out, "causes": causes}, open(os.path.join(RC.HERE, "out", "sample_check.json"), "w"), indent=1)
