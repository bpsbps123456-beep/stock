"""Full-population agreement check of every A_*/B_* candidate row (yf1m) under two rule sets.

LITERAL  = audit-task definitions as written (valid = next bar & t+2 inside the 30-min same-date window;
           tercile over bars with ATR; R2 stop checked from bar t+2 even if entry is already below the stop).
RESOLVED = LITERAL + three rules inferred from the tables during the 400-row audit:
           (a) valid also needs next bar start <= signal start + 3 min (BRIEF hygiene "next bar within 2 min of close"),
           (b) bars starting >= 19:30 ET are not valid (30-min window would run past 20:00),
           (c) bars with NaN ATR (first 13 bars of the series) fall into tercile 0 instead of being dropped,
           (d) R2 = 0 when the E2 entry (close of t+1) is already below the stop.
Reports agreement counts only.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute as RC

F_E2 = ["r1_e2", "r1net_e2", "hit_e2", "race_e2"]
F_E0 = ["r1_e0", "hit_e0"]
F_BT = ["r1_e2", "r1net_e2", "race_e2", "hit_e2", "r1_e0", "hit_e0"]


def sym_arrays(sym):
    a = RC.arrays(RC.load_raw(sym))
    res = RC.bar_out(a["ts"], a["day"], a["sess"], a["o"], a["h"], a["l"], a["c"], RC.COST)
    for k, v in zip(["v0", "v2", "r1_e0", "hit_e0", "race_e0", "r1_e2", "hit_e2", "race_e2", "hits_e2", "r1net_e2", "lastw"], res):
        a[k] = v
    k = RC.COST[a["sess"]]
    a["r1net_e0"] = np.where(a["hit_e0"] == 1.0, a["r1_e0"] - k, a["r1_e0"] - 2 * k)
    n = len(a["ts"])
    gap1 = np.full(n, 10**9); gap1[:-1] = a["ts"][1:] - a["ts"][:-1]
    a["vg"] = gap1 <= 3
    a["late_ok"] = a["mins"] < 1170
    a["atrp"] = RC.atr14(a["h"], a["l"], a["c"]) / a["c"]
    df = pd.DataFrame({"day": a["day"], "sess": a["sess"], "atrp": a["atrp"]})
    pct = df.groupby(["day", "sess"])["atrp"].rank(pct=True).to_numpy()
    terc = np.minimum(np.floor(pct * 3), 2)
    a["terc_lit"] = terc                              # NaN ATR -> no tercile
    a["terc_res"] = np.where(np.isnan(terc), 0.0, terc)
    return a


def bv_maps(a, mode):
    terc = a["terc_lit"] if mode == "lit" else a["terc_res"]
    if mode == "lit":
        val0, val2 = a["v0"], a["v2"]
    else:
        val0 = a["v0"] & a["vg"] & a["late_ok"]; val2 = a["v2"] & a["vg"] & a["late_ok"]
    df = pd.DataFrame({"day": a["day"], "sess": a["sess"], "terc": terc})
    out = {}
    for f, vv in [(f, val2) for f in F_E2] + [(f, val0) for f in F_E0]:
        x = pd.DataFrame({"day": a["day"], "sess": a["sess"], "terc": terc, "v": np.where(vv, a[f], np.nan)})
        m = x.groupby(["day", "sess", "terc"])["v"].transform("mean").to_numpy()
        m = np.where(np.isnan(terc), np.nan, m)
        out[f"bv_{f}"] = m
    return out


def r2_all(a, ts_idx, kind, mode):
    out = np.full(len(ts_idx), np.nan)
    for n_, t in enumerate(ts_idx):
        stop = a["l"][t] if kind == "A" else min(a["l"][t - 1], a["l"][t])
        if mode == "res" and a["c"][t + 1] < stop:
            out[n_] = 0.0
        else:
            out[n_] = RC.r2_stop(t, stop, a, a["lastw"])
    return out


def main():
    per = {}
    mm_lit, mm_res = [], []
    for sid, sym in enumerate(RC.SYMS):
        a = sym_arrays(sym)
        a["bv_lit"] = bv_maps(a, "lit"); a["bv_res"] = bv_maps(a, "res")
        per[sym] = a
        for lst, vb in ((mm_lit, a["v0"] & a["v2"]), (mm_res, a["v0"] & a["v2"] & a["vg"] & a["late_ok"])):
            lst.append(pd.DataFrame({"sid": sid, "ts": a["ts"][vb], **{f: a[f][vb] for f in F_BT}}))
    aggs = {}
    for mode, lst in (("lit", mm_lit), ("res", mm_res)):
        M = pd.concat(lst, ignore_index=True); g = M.groupby("ts")
        ag = pd.DataFrame({"cnt": g.size()})
        for f in F_BT:
            ag[f"s_{f}"] = g[f].sum(); ag[f"n_{f}"] = g[f].count()
        aggs[mode] = ag

    report = {"tables": {}, "totals": {}}
    tot = {}
    for path in sorted(glob.glob(os.path.join(RC.CAND, "A_*.parquet")) + glob.glob(os.path.join(RC.CAND, "B_*.parquet"))):
        key = os.path.basename(path)[:-8]; kind = key[0]
        X = pd.read_parquet(path)
        res_tab = {"rows": len(X)}
        mine = {"lit": {}, "res": {}}
        for f in ["r1_e0", "hit_e0", "race_e0", "r1net_e0", "r1_e2", "r1net_e2", "hit_e2", "hits_e2", "race_e2"]:
            col = np.empty(len(X))
            for sid, grp in X.groupby("sid"):
                a = per[RC.SYMS[sid]]; col[grp.index.to_numpy()] = a[f][grp.t.to_numpy()]
            mine["lit"][f] = col; mine["res"][f] = col
        for mode in ("lit", "res"):
            for f in [f"bv_{x}" for x in F_E2 + F_E0]:
                col = np.empty(len(X))
                for sid, grp in X.groupby("sid"):
                    a = per[RC.SYMS[sid]]; col[grp.index.to_numpy()] = a[f"bv_{mode}"][f][grp.t.to_numpy()]
                mine[mode][f] = col
            col = np.empty(len(X))
            for sid, grp in X.groupby("sid"):
                a = per[RC.SYMS[sid]]; col[grp.index.to_numpy()] = r2_all(a, grp.t.to_numpy(), kind, mode)
            mine[mode]["r2_e2"] = col
            ag = aggs[mode].reindex(X.ts.to_numpy())
            ownvalid = np.empty(len(X), bool)
            for sid, grp in X.groupby("sid"):
                a = per[RC.SYMS[sid]]; tt = grp.t.to_numpy()
                ownvalid[grp.index.to_numpy()] = (a["v0"][tt] & a["v2"][tt]) if mode == "lit" else \
                    (a["v0"][tt] & a["v2"][tt] & a["vg"][tt] & a["late_ok"][tt])
            for f in F_BT:
                own = mine[mode][f]
                has = ownvalid & ~np.isnan(own)
                s = ag[f"s_{f}"].to_numpy() - np.where(has, own, 0.0)
                n = ag[f"n_{f}"].to_numpy() - has
                with np.errstate(invalid="ignore", divide="ignore"):
                    mine[mode][f"bt_{f}"] = np.where(n > 0, s / n, np.nan)
            mine[mode]["n_same_min"] = ag["cnt"].fillna(0).to_numpy()
        for mode in ("lit", "res"):
            for f, v in mine[mode].items():
                if f not in X.columns:
                    continue
                th = X[f].to_numpy(float); mv = np.asarray(v, float)
                bad = ~((np.isnan(th) & np.isnan(mv)) | (np.abs(th - mv) <= RC.TOL))
                res_tab[f"{mode}:{f}"] = int(bad.sum())
                tot[f"{mode}:{f}"] = tot.get(f"{mode}:{f}", 0) + int(bad.sum())
        # counts for root-cause sizing (no outcome values)
        stop = []
        for sid, grp in X.groupby("sid"):
            a = per[RC.SYMS[sid]]; tt = grp.t.to_numpy()
            st = a["l"][tt] if kind == "A" else np.minimum(a["l"][tt - 1], a["l"][tt])
            stop.append(pd.Series(a["c"][tt + 1] < st, index=grp.index))
        eb = pd.concat(stop).sort_index().to_numpy()
        res_tab["entry_below_stop_rows"] = int(eb.sum())
        tot["entry_below_stop_rows"] = tot.get("entry_below_stop_rows", 0) + int(eb.sum())
        tot["rows"] = tot.get("rows", 0) + len(X)
        report["tables"][key] = res_tab
        print(key, {k: v for k, v in res_tab.items() if (isinstance(v, int) and v and k != "rows") or k == "rows"}, flush=True)
    report["totals"] = tot
    json.dump(report, open(os.path.join(RC.HERE, "out", "full_check.json"), "w"), indent=1)
    print("TOTALS")
    for k, v in sorted(tot.items()):
        print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
