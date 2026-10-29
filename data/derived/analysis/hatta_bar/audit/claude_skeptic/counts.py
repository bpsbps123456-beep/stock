"""Counts only (no performance numbers):
1) dump_signals.py CELLS (cache/engine_signals_yf1m.csv) vs auditor reference selection.
2) plain (simple N-bar new low) candidate row sets vs independent rule.
3) structural-issue counts on tradable & sane candidate rows (engine arrays)."""
import sys, json
import numpy as np
import pandas as pd

BASE = "C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar"
sys.path.insert(0, f"{BASE}/code")
sys.path.insert(0, f"{BASE}/audit/claude_skeptic")
import engine as E
import ref as R
from build import SYMS

CELLS = [  # copied from BRIEF-described cells in dump_signals.py (cell definitions are inputs, not logic)
    ("orig_A", "A", 120, 0.0005, "level", 3, 2, "persist", 0.02, 0.0, ("wick", 0.5, 1.0, False)),
    ("orig_B", "B", 120, 0.0005, None, 3, 1, "persist", 0.02, 0.0, (True, "basic")),
    ("alt_A1", "A", 60, 0.0, "touch", 2, 3, "now", 0.0, 2.0, ("cpos", 0.65, 0.0, True)),
    ("alt_A2", "A", 390, 0.001, "level", 5, 3, "none", 0.03, 0.0, ("wick", 0.65, 2.0, False)),
    ("alt_B1", "B", 240, 0.001, "level", 2, 2, "now", 0.01, 2.0, (False, "full")),
    ("alt_B2", "B", 60, 0.0, "touch", 5, 2, "none", 0.0, 0.0, (False, "reclaim")),
]
LO, HI = 20260930, 20261028


def sel(X, pat, kmode, p, K, ema, D, v, sh):
    m = X.sane.to_numpy().astype(bool)
    if kmode:
        m &= X[f"{kmode}{p}"].to_numpy() >= K
    if ema != "none":
        m &= X["stack" if ema == "now" else "persist"].to_numpy().astype(bool)
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
    return X[m]


def main():
    dump = pd.read_csv(f"{BASE}/cache/engine_signals_yf1m.csv")
    mine_rows = []
    plain_mine = {}
    struct = dict(rows=0, e0_ok_e2_bad=0, gap_or_dayend=0, after1929=0, cost_sess_cross_entry=0,
                  cost_sess_cross_exit=0, sig_1058_1059=0, first_day=0, prepost_sane=0)
    drop_any = dict(tradsane_breaks=0, kept_valid=0, e2_only_invalid=0, next_gap=0)
    for sid, sym in enumerate(SYMS):
        d = R.load("yf1m", sym)
        if d is None or len(d) < 500:
            continue
        r = R.per_symbol(d)
        a = E.prep(d)
        out = E.bar_outcomes(a)
        n = len(d)
        # (1) dump cells
        cache = {}
        for name, pat, N, tol, kmode, p, K, ema, D, v, sh in CELLS:
            if (N, tol) not in cache:
                cache[(N, tol)] = R.tables(sym, sid, r, N, tol, 0, 0, filt=False)
            A, B = cache[(N, tol)]
            X = A if pat == "A" else B
            if X is None or len(X) == 0:
                continue
            Y = sel(X, pat, kmode, p, K, ema, D, v, sh)
            for _, row in Y.iterrows():
                ts = pd.Timestamp(int(row.ts) * 60, unit="s", tz="UTC").tz_convert("America/New_York")
                mine_rows.append((name, sym, ts.strftime("%Y-%m-%d %H:%M"), pat, int(row.k), float(row.S),
                                  int(row[f"touch{p}"]), int(row[f"level{p}"])))
        # (2) plain: S = min(low[t-N..t-1]), break = low < S
        low = r["l"]
        for N in E.NS:
            S = np.full(n, np.nan)
            for t in range(N, n):
                S[t] = low[t - N:t].min()
            with np.errstate(invalid="ignore"):
                k = np.nonzero(low < S)[0]
            k = k[(k >= 31) & (k < n - 3)]
            vA = (r["valid"][0] & r["valid"][1])[k] & (r["day"][k] >= LO) & (r["day"][k] <= HI)
            plain_mine.setdefault(f"plain_A_{N}", set()).update((sid, int(x)) for x in k[vA])
            t = k + 1
            red = (r["c"][k] < r["o"][k]) & (r["day"][t] == r["day"][k])
            vB = red & (r["valid"][0] & r["valid"][1])[t] & (r["day"][t] >= LO) & (r["day"][t] <= HI)
            plain_mine.setdefault(f"plain_B_{N}", set()).update((sid, int(x)) for x in t[vB])
        # (3) structural counts on A and B breaks of (120, 0.0005), tradable & sane, in period, before valid filter
        A, B = cache[(120, 0.0005)]
        for X in (A, B):
            X = X[X.tradable & X.sane & (X.day >= LO) & (X.day <= HI)]
            tt = X.t.to_numpy().astype(np.int64)
            drop_any["tradsane_breaks"] += len(tt)
            v0, v2 = out["valid_e0"][tt], out["valid_e2"][tt]
            drop_any["kept_valid"] += int((v0 & v2).sum())
            drop_any["e2_only_invalid"] += int((v0 & ~v2).sum())
            drop_any["next_gap"] += int((~v0).sum())
            Xv = X[v0 & v2]
            tv = Xv.t.to_numpy().astype(np.int64)
            struct["rows"] += len(tv)
            zz = out["zz"][tv]
            struct["cost_sess_cross_entry"] += int((a["sess"][tv + 1] != a["sess"][tv]).sum())
            struct["cost_sess_cross_exit"] += int((a["sess"][zz] != a["sess"][tv]).sum())
            struct["sig_1058_1059"] += int(((a["mins"][tv] >= 658) & (a["mins"][tv] < 660)).sum())
            struct["first_day"] += int((a["day"][tv] == LO).sum())
            struct["prepost_sane"] += int(np.isin(a["sess"][tv], (0, 3)).sum())
    mine = pd.DataFrame(mine_rows, columns=dump.columns)
    res = {}
    for name, *_ in CELLS:
        de = dump[dump.cell == name]
        dm = mine[mine.cell == name]
        ke = set(zip(de.sym, de.ts_et))
        km = set(zip(dm.sym, dm.ts_et))
        M = de.merge(dm, on=["cell", "sym", "ts_et"], suffixes=("_e", "_m"))
        attr_bad = int(((M.k_e != M.k_m) | (np.abs(M.S_e - M.S_m) > 1e-9) | (M.touch_e != M.touch_m)
                        | (M.level_e != M.level_m)).sum())
        res[name] = dict(engine=len(de), mine=len(dm), matched=len(ke & km), only_e=len(ke - km), only_m=len(km - ke),
                         attr_mismatch=attr_bad)
        print(name, res[name])
    pl = {}
    for key, s in plain_mine.items():
        P = pd.read_parquet(f"{BASE}/cache/cand_yf1m/{key}.parquet")
        se = set(zip(P.sid, P.t))
        pl[key] = dict(engine=len(se), mine=len(s), matched=len(se & s))
        print(key, pl[key])
    print("drop counts (A+B, N120 tol0.05%, tradable&sane, in period):", drop_any)
    print("structural counts on kept rows:", struct)
    json.dump(dict(dump=res, plain=pl, drop=drop_any, struct=struct),
              open(f"{BASE}/audit/claude_skeptic/counts_yf1m.json", "w"), indent=1)


if __name__ == "__main__":
    main()
