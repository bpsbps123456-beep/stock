"""Inspect remaining bv mismatches: which group members / tercile rule would reproduce the table value."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute as RC

OUT = os.path.join(RC.HERE, "out")
R = pd.read_csv(os.path.join(OUT, "mine.csv")); S = pd.read_csv(os.path.join(OUT, "sample.csv"))
best = "vg_own|all|floor|average"
bad = set()
for f in ("r1_e2", "hit_e2", "race_e2", "r1net_e2", "r1_e0", "hit_e0"):
    m = (R[f"bv_{f}|{best}"] - S[f"bv_{f}"]).abs() > RC.TOL
    m &= ~(R[f"bv_{f}|{best}"].isna() & S[f"bv_{f}"].isna())
    print(f, int(m.sum()), sorted(R.index[m].tolist()))
    bad |= set(R.index[m].tolist())

for i in sorted(bad):
    r = S.loc[i]; sym = RC.SYMS[int(r.sid)]
    a = RC.arrays(RC.load_raw(sym))
    res = RC.bar_out(a["ts"], a["day"], a["sess"], a["o"], a["h"], a["l"], a["c"], RC.COST)
    v0, v2 = res[0], res[1]
    r1_2, hit_2, race_2 = res[5], res[6], res[7]
    gap1 = np.full(len(a["ts"]), 10**9); gap1[:-1] = a["ts"][1:] - a["ts"][:-1]
    vg = gap1 <= 3
    atrp = RC.atr14(a["h"], a["l"], a["c"]) / a["c"]
    t = int(r.t)
    g = np.where((a["day"] == a["day"][t]) & (a["sess"] == a["sess"][t]) & ~np.isnan(atrp))[0]
    gall = np.where((a["day"] == a["day"][t]) & (a["sess"] == a["sess"][t]))[0]
    pct = pd.Series(atrp[g]).rank(pct=True).to_numpy()
    terc = np.minimum(np.floor(pct * 3), 2)
    my = terc[np.searchsorted(g, t)]
    sel = g[terc == my]
    vm = v2[sel] & vg[sel]
    print(f"\n#{i} {sym} {r.table} t={t} day={a['day'][t]} sess={a['sess'][t]} group={len(g)} (incl NaN-ATR {len(gall)}) "
          f"terc={my} sel={len(sel)} valid_in_sel={vm.sum()} pct_t={pct[np.searchsorted(g, t)]:.6f}")
    mine = R.loc[i, f"bv_r1_e2|{best}"]; tab = r.bv_r1_e2
    print(f"   diff r1 = {mine - tab:.3e}; hit diff = {R.loc[i, f'bv_hit_e2|{best}'] - r.bv_hit_e2:.3e}; "
          f"race diff = {R.loc[i, f'bv_race_e2|{best}'] - r.bv_race_e2:.3e}")
    # try: boundary members moved between terciles (pct exactly at k/3 boundaries)
    x = pct * 3
    near = np.abs(x - np.round(x)) < 1e-9
    print("   members exactly on a tercile boundary:", int(near.sum()), "pct*3 values:", np.unique(x[near]).tolist()[:6])
    # alternative: ceil(pct*3)-1
    t2 = np.ceil(pct * 3) - 1
    my2 = t2[np.searchsorted(g, t)]
    sel2 = g[t2 == my2]; vm2 = v2[sel2] & vg[sel2]
    print(f"   ceil-1 rule: r1 mean diff vs table = {np.nanmean(r1_2[sel2][vm2]) - tab:.3e}")
    # alternative: rank by atr (not atr%) or by atr[t-1]
    for lab, key in (("atr abs", RC.atr14(a["h"], a["l"], a["c"])), ("atr%[t-1]", np.r_[np.nan, atrp[:-1]])):
        gg = np.where((a["day"] == a["day"][t]) & (a["sess"] == a["sess"][t]) & ~np.isnan(key))[0]
        if t not in set(gg.tolist()):
            print("  ", lab, "t not in group"); continue
        p3 = pd.Series(key[gg]).rank(pct=True).to_numpy()
        tt = np.minimum(np.floor(p3 * 3), 2)
        s3 = gg[tt == tt[np.searchsorted(gg, t)]]
        v3 = v2[s3] & vg[s3]
        print(f"   {lab}: r1 mean diff vs table = {np.nanmean(r1_2[s3][v3]) - tab:.3e}")
    # fp: compute floor((rank)*3/n) using integer ranks
    rk = pd.Series(atrp[g]).rank().to_numpy(); n = len(g)
    t4 = np.minimum(np.floor(rk * 3 / n), 2)
    s4 = g[t4 == t4[np.searchsorted(g, t)]]; v4 = v2[s4] & vg[s4]
    print(f"   floor(rank*3/n): r1 mean diff vs table = {np.nanmean(r1_2[s4][v4]) - tab:.3e}")
