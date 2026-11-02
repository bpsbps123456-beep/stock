"""Pass 1: detect every v3 original pattern-(1) signal (all hours) for the 400 symbols.

Output: out/cand.parquet (candidates that pass all bar-local conditions, with D and level count)
        out/sig1.parquet  (full pattern-(1) signals = candidates with D>=2% and level>=2, plus outcome)
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar/audit/h3-recompute")
import h3r_core as C  # noqa: E402


def prev_close_map(daily):
    out = {}
    for sym, g in daily.groupby("sym", sort=False):
        g = g.sort_values("date")
        dd = (g["date"].dt.year * 10000 + g["date"].dt.month * 100 + g["date"].dt.day).to_numpy(np.int64)
        out[sym] = (dd, g["c"].to_numpy(np.float64))
    return out


def main():
    t0 = time.time()
    ds = sys.argv[1] if len(sys.argv) > 1 else "2026h1"
    C.CACHE = C.BASE + "/cache/alp1m_" + ds
    suffix = "" if ds == "2026h1" else "_" + ds
    uni = C.load_universe()
    cm = C.load_cost()
    daily = pd.read_parquet(C.BASE + "/cache/daily_" + ds + ".parquet")
    pcm = prev_close_map(daily)
    cand_rows, sig_rows = [], []
    import os
    have = set(f[:-8] for f in os.listdir(C.CACHE))
    for k, sym in enumerate(sorted(uni)):
        if sym not in have:
            print("no bars", sym)
            continue
        d = C.load_bars(sym)
        A = C.base_arrays(d)
        vwap = C.day_vwap(d)
        rvwap = C.day_vwap(d, start_min=570)
        cand = A["brk"] & A["openok"] & (A["wick"] >= C.R_MIN) & A["san"] & A["tr"] & A["nxt"] & A["e2ok"]
        idx = np.nonzero(cand)[0]
        sw = C.swings(d["l"])
        shares = uni[sym]["shares"]
        dd, cc = pcm.get(sym, (np.array([], np.int64), np.array([])))
        for t in idx:
            S = A["S"][t]
            D, mh = C.drop120(d, t)
            lev, npts = C.level_count(d["l"], sw, t, S)
            ok = (D >= C.D_MIN) and (lev >= C.K_MIN)
            row = dict(sym=sym, t=int(t), ts=int(d["ts"][t]), day=int(d["day"][t]), mins=int(d["mins"][t]),
                       S=S, D=D, lev=lev, npts=npts, ok=ok)
            cand_rows.append(dict(row))
            if not ok:
                continue
            z = int(A["z"][t])
            entry, hit, r1, tb = C.e2_outcome(d, t, z)
            _, hit_s, r1_s, _ = C.e2_outcome(d, t, z, tgt_mult=1.011)
            r1_strict = 0.01 if hit_s else (d["c"][z] / entry - 1.0)
            # prev official close
            j = np.searchsorted(dd, row["day"], side="left") - 1
            pc = cc[j] if j >= 0 else np.nan
            cap_b = shares * pc / 1e9 if np.isfinite(pc) else np.nan
            tr_ = C.tier(cap_b)
            c_in = C.cost(cm, tr_, C.tod_in(row["mins"]))
            c_out = C.cost(cm, tr_, C.tod_out(int(d["mins"][z])))
            r1n = r1 - c_in - (0.0 if hit else c_out)
            # features for adversarial checks
            o, h, l, c, v, nn, vw = (d[x][t] for x in ("o", "h", "l", "c", "v", "n", "vw"))
            lm1 = d["l"][t - 1]
            lp1 = d["l"][t + 1]
            mx = np.max(d["h"][t + 2:z + 1])
            row.update(
                vgap=c / vwap[t] - 1.0,
                rvgap=(c / rvwap[t] - 1.0) if (570 <= row["mins"] < 960 and np.isfinite(rvwap[t])) else np.nan,
                o=o, h=h, l=l, c=c, v=v, trades=nn, bvw=vw, l_m1=lm1, l_p1=lp1,
                n_m1=d["n"][t - 1], n_p1=d["n"][t + 1],
                entry=entry, z=z, z_mins=int(d["mins"][z]), z_ts=int(d["ts"][z]), hit=hit, r1=r1,
                hit_strict=hit_s, r1_strict=r1_strict,
                tb=int(tb), tb_h=(d["h"][tb] if tb >= 0 else np.nan), tb_c=(d["c"][tb] if tb >= 0 else np.nan),
                tb_n=(d["n"][tb] if tb >= 0 else np.nan), tb_vw=(d["vw"][tb] if tb >= 0 else np.nan),
                tb_v=(d["v"][tb] if tb >= 0 else np.nan),
                n_touch=int(np.sum(d["h"][t + 2:z + 1] >= entry * 1.01)),
                mfe=mx / entry - 1.0, mae=np.min(d["l"][t + 2:z + 1]) / entry - 1.0,
                next_gap=int(d["ts"][t + 1] - d["ts"][t]),
                prevc=pc, cap_b=cap_b, tier=tr_, c_in=c_in, c_out=c_out, r1n=r1n,
                dayret_low=(l / pc - 1.0) if np.isfinite(pc) else np.nan,
                dayret_close=(c / pc - 1.0) if np.isfinite(pc) else np.nan,
                maxh120=mh,
            )
            sig_rows.append(row)
        if k % 50 == 0:
            print(k, sym, len(idx), len(sig_rows), f"{time.time() - t0:.0f}s", flush=True)
    cand = pd.DataFrame(cand_rows)
    sig = pd.DataFrame([r for r in sig_rows])
    cand.to_parquet(C.OUT + "/cand" + suffix + ".parquet")
    sig.to_parquet(C.OUT + "/sig1" + suffix + ".parquet")
    print("done", len(cand), len(sig), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
