"""Randomized differential test: engine.py (system under test) vs auditor ref.py on synthetic 1-minute data with
tick-rounded prices (ties), random minute gaps (incl. exactly 2/3/4-minute gaps), day rollovers, 19:29/19:30 bars,
zero volumes and a Trades column. Prints only mismatch counts. Usage: py fuzz.py [trials]"""
import sys
import numpy as np
import pandas as pd

BASE = "C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar"
sys.path.insert(0, f"{BASE}/code")
sys.path.insert(0, f"{BASE}/audit/claude_skeptic")
import engine as E   # system under test
import ref as R      # auditor
from compare import cmp_col


def synth(rng):
    ndays = rng.integers(3, 6)
    rows = []
    px = float(rng.uniform(4, 12))
    tick = 0.01
    day0 = pd.Timestamp("2026-08-03")
    for dd in range(ndays):
        date = day0 + pd.Timedelta(days=int(dd))
        if date.weekday() >= 5:
            continue
        m = 240
        while m < 1200:
            reg = 570 <= m < 960
            drift = rng.normal(0, 0.0015 if reg else 0.001)
            if rng.random() < 0.01:
                drift -= rng.uniform(0.005, 0.02)          # sharp drops -> breaks
            if rng.random() < 0.01:
                drift += rng.uniform(0.005, 0.015)
            o = round(px / tick) * tick
            c = round(px * (1 + drift) / tick) * tick
            hi = max(o, c) + tick * rng.integers(0, 4)
            lo = min(o, c) - tick * rng.integers(0, 6)
            if rng.random() < 0.05:                       # long lower wick
                lo -= tick * rng.integers(3, 15)
            lo = max(lo, tick)
            vol = 0.0 if rng.random() < 0.05 else float(rng.integers(100, 50000))
            trades = float(rng.integers(0, 20))
            rows.append((date + pd.Timedelta(minutes=int(m)), o, hi, lo, c, vol, trades))
            px = c
            # gaps: mostly 1 minute; sometimes exactly 2,3,4,5 minutes
            u = rng.random()
            step = 1 if u < (0.9 if reg else 0.7) else int(rng.choice([2, 3, 4, 5, 7]))
            m += step
    d = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close", "Volume", "Trades"]).set_index("t")
    d.index = d.index.tz_localize("America/New_York")
    d = d[(d.High >= d.Low) & (d.Low > 0)]
    return d


def main(trials):
    rng = np.random.default_rng(12345)
    tot = dict(bars=0, out_mism=0, zz_mism=0, rows_e=0, rows_r=0, only_e=0, only_r=0, col_mism={}, gap_counts={})
    for tr in range(trials):
        d = synth(rng)
        if len(d) < 500:
            continue
        a = E.prep(d)
        out = E.bar_outcomes(a)
        r = R.per_symbol(d)
        tot["bars"] += len(d)
        gaps = np.diff(a["ts"])
        for g in (1, 2, 3, 4):
            tot["gap_counts"][g] = tot["gap_counts"].get(g, 0) + int((gaps == g).sum())
        # all-bar outcomes
        for m, tag in ((0, "e0"), (1, "e2")):
            if (out[f"valid_{tag}"] != r["valid"][m]).any():
                tot["out_mism"] += int((out[f"valid_{tag}"] != r["valid"][m]).sum())
            for k, name in enumerate(R.FIELDS):
                nb_, _, _ = cmp_col(out[f"{name}_{tag}"], r["f"][k, m])
                tot["out_mism"] += nb_
            for name in ("r1", "r1net", "hit", "race"):
                nb_, _, _ = cmp_col(out[f"bv_{name}_{tag}"], r["bv"][f"bv_{name}_{tag}"])
                tot["out_mism"] += nb_
        zz_e = out["zz"]
        zz_r = np.where(r["valid"][0], r["last"], -1)
        tot["zz_mism"] += int((zz_e != zz_r).sum())
        for N in (60, 120):
            for tol in (0.0, 0.0005, 0.001):
                A, B = E.candidates("X", a, out, N, tol)
                A2, B2 = R.tables("X", 0, r, N, tol, 0, 99999999)
                for X, Y in ((A, A2), (B, B2)):
                    if X is None:
                        X = pd.DataFrame(columns=["t"])
                    else:
                        X = X[X.valid]
                    if Y is None:
                        Y = pd.DataFrame(columns=["t"])
                    tot["rows_e"] += len(X)
                    tot["rows_r"] += len(Y)
                    se, sr = set(X.t), set(Y.t)
                    tot["only_e"] += len(se - sr)
                    tot["only_r"] += len(sr - se)
                    if len(X) == 0 or len(Y) == 0:
                        continue
                    M = X.merge(Y, on="t", suffixes=("_e", "_r"))
                    for c in X.columns:
                        if c in ("t", "sym", "sid") or c not in Y.columns:
                            continue
                        nb_, _, _ = cmp_col(M[c + "_e"], M[c + "_r"])
                        if nb_:
                            tot["col_mism"][c] = tot["col_mism"].get(c, 0) + nb_
    print(tot)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
