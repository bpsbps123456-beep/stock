"""Independent recomputation of grid cell statistics vs code/grid.py grid_table (system under test).
Prints only cell counts and mismatch counts. Does not write into results/.
Usage: py grid_check.py yf1m 20261014 [sample_per_file]"""
import os, sys, json, itertools, time
import numpy as np
import pandas as pd
import numba as nb

BASE = "C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar"
sys.path.insert(0, f"{BASE}/code")
import grid as G  # system under test

NS, TOLS, PS = (60, 120, 240, 390), (0.0, 0.0005, 0.001), (2, 3, 5)
STATS = ["n", "days", "syms", "share", "r1", "r1net", "lift", "t", "hit", "bvhit", "race", "r1e0", "hite0", "liftnet", "tnet"]

# BRIEF grid axes, enumerated independently
K_AX = [("k1", None, 1, 0)] + [(f"{m}{p}_{k}", m, k, p) for m in ("touch", "level") for p in PS for k in (2, 3)]
EMA_AX = ("none", "now", "persist")
D_AX = (0.0, 0.01, 0.02, 0.03)
V_AX = (0.0, 2.0)
A_SH = list(itertools.product(("wick", "cpos"), (0.5, 0.65), (0.0, 1.0, 2.0), (False, True)))
B_SH = list(itertools.product((True, False), ("basic", "body", "full", "reclaim")))


@nb.njit
def keep_mask(sid, ts):
    """rows sorted by (sid, ts): keep a row unless it starts < 30 min after the last KEPT row of the same symbol."""
    n = len(sid)
    keep = np.zeros(n, np.bool_)
    for q in range(n):
        ok = True
        for r in range(q - 1, -1, -1):          # search backwards for last kept row of same symbol
            if sid[r] != sid[q]:
                break
            if keep[r]:
                ok = ts[q] - ts[r] >= 30
                break
        keep[q] = ok
    return keep


def cl_t(x, day):
    s = pd.Series(x).groupby(day)
    n = len(x)
    m = x.mean()
    g = s.agg(["sum", "count"])
    Gc = len(g)
    if Gc < 2:
        return np.nan
    v = ((g["sum"] - m * g["count"]) ** 2).sum()
    if v <= 0:
        return np.nan
    se = np.sqrt(Gc / (Gc - 1) * v) / n
    return m / se


def stats(Y):
    if len(Y) == 0:
        return [0] + [np.nan] * 14
    x = (Y.r1_e2 - Y.bv_r1_e2).to_numpy()
    xn = (Y.r1net_e2 - Y.bv_r1net_e2).to_numpy()
    dc = Y.groupby("day").size()
    race = Y.race_e2.dropna()
    return [len(Y), len(dc), Y.sid.nunique(), dc.max() / len(Y), Y.r1_e2.mean(), Y.r1net_e2.mean(), x.mean(),
            cl_t(x, Y.day.to_numpy()), Y.hit_e2.mean(), Y.bv_hit_e2.mean(), race.mean() if len(race) else np.nan,
            Y.r1_e0.mean(), Y.hit_e0.mean(), xn.mean(), cl_t(xn, Y.day.to_numpy())]


def my_mask(X, pat, kopt, e, D, v, sh):
    _, mode, k, p = kopt
    m = np.ones(len(X), bool)
    if mode is not None:
        m &= X[f"{mode}{p}"].to_numpy() >= k
    if e == "now":
        m &= X["stack"].to_numpy().astype(bool)
    elif e == "persist":
        m &= X["persist"].to_numpy().astype(bool)
    m &= X.dd.to_numpy() >= D
    if v > 0:
        m &= X.volr.to_numpy() >= v
    if pat == "A":
        rm, r, mm, ab = sh
        m &= X.openok.to_numpy() & (X[rm].to_numpy() >= r) & (X.rngatr.to_numpy() >= mm)
        if ab:
            m &= X.above.to_numpy()
    else:
        bl, eng = sh
        m &= X[f"eng_{eng}"].to_numpy()
        if bl:
            m &= X.below.to_numpy()
    return m


def main(src, split, sample):
    rng = np.random.default_rng(7)
    rep = {}
    tot_cells = tot_bad = 0
    bad_by_stat = {}
    t0 = time.time()
    for pat in ("A", "B"):
        shapes = A_SH if pat == "A" else B_SH
        for N in NS:
            for tol in TOLS:
                path = f"{BASE}/cache/cand_{src}/{pat}_{N}_{tol}.parquet"
                g = G.grid_table(path, pat, N, tol, split)
                X = pd.read_parquet(path)
                X = X[X.tradable & X.sane].sort_values(["sid", "ts"], kind="stable").reset_index(drop=True)
                cells = list(itertools.product(K_AX, EMA_AX, D_AX, V_AX, shapes))
                assert len(cells) == len(g), (len(cells), len(g))
                pick = np.arange(len(cells)) if sample <= 0 else rng.choice(len(cells), min(sample, len(cells)), replace=False)
                nb_bad = 0
                for ci in pick:
                    kopt, e, D, v, sh = cells[ci]
                    row = g.iloc[ci]
                    # check the grid row is the same cell
                    assert row.K == kopt[0] and row.ema == e and row.D == D and row.v == v
                    Y = X[my_mask(X, pat, kopt, e, D, v, sh)]
                    Y = Y[keep_mask(Y.sid.to_numpy().astype(np.int64), Y.ts.to_numpy().astype(np.int64))]
                    front = Y[Y.day <= split]
                    back = Y[Y.day > split]
                    mine = stats(front) + stats(back)
                    theirs = [row[f"f_{s}"] for s in STATS] + [row[f"b_{s}"] for s in STATS]
                    bad = False
                    for j, (a_, b_) in enumerate(zip(mine, theirs)):
                        a_ = float(a_); b_ = float(b_)
                        if (np.isnan(a_) and np.isnan(b_)) or abs(a_ - b_) <= 1e-9 * max(1, abs(a_)):
                            continue
                        # grid reports n=0 with NaN elsewhere; mine reports 0 + NaN — identical; any other diff is bad
                        nm = ("f_" if j < 15 else "b_") + STATS[j % 15]
                        bad_by_stat[nm] = bad_by_stat.get(nm, 0) + 1
                        bad = True
                    nb_bad += bad
                tot_cells += len(pick)
                tot_bad += nb_bad
                rep[f"{pat}_{N}_{tol}"] = dict(cells=len(g), checked=int(len(pick)), mismatched=int(nb_bad))
                print(pat, N, tol, "cells", len(g), "checked", len(pick), "mismatched", nb_bad, f"{time.time()-t0:.0f}s", flush=True)
    print("TOTAL checked", tot_cells, "mismatched", tot_bad, "by stat", bad_by_stat)
    json.dump(dict(rep=rep, total=tot_cells, bad=tot_bad, by_stat=bad_by_stat),
              open(f"{BASE}/audit/claude_skeptic/grid_check_{src}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 400)
