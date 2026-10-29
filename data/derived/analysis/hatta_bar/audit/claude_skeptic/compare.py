"""Compare engine/build candidate tables with the auditor reference. Prints ONLY counts / mismatch counts / max abs diffs.
Usage: py compare.py yf1m"""
import sys, json
import numpy as np
import pandas as pd

BASE = "C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar"
NS, TOLS = (60, 120, 240, 390), (0.0, 0.0005, 0.001)


def cmp_col(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype == bool or b.dtype == bool or a.dtype.kind in "iu":
        bad = a.astype(float) != b.astype(float)
        return int(bad.sum()), float("nan"), bad
    a = a.astype(float)
    b = b.astype(float)
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(invalid="ignore"):
        diff = np.abs(a - b)
        tol = 1e-9 * np.maximum(1.0, np.abs(a))
        bad = ~both_nan & ~(diff <= tol)
        # inf==inf
        bad &= ~(np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b)))
    md = float(np.nanmax(np.where(both_nan | np.isinf(diff), np.nan, diff))) if (~both_nan).any() else 0.0
    return int(bad.sum()), md, bad


def main(src):
    rep = {}
    for pat in ("A", "B"):
        for N in NS:
            for tol in TOLS:
                key = f"{pat}_{N}_{tol}"
                E = pd.read_parquet(f"{BASE}/cache/cand_{src}/{key}.parquet")
                R = pd.read_parquet(f"{BASE}/audit/claude_skeptic/ref_{src}/{key}.parquet")
                ek = set(zip(E.sid, E.t))
                rk = set(zip(R.sid, R.t))
                M = E.merge(R, on=["sid", "t"], suffixes=("_e", "_r"))
                cols = [c for c in E.columns if c not in ("sid", "t", "sym", "n_same_min") and c in R.columns]
                mism = {}
                for c in cols:
                    nb, md, _ = cmp_col(M[c + "_e"], M[c + "_r"])
                    if nb:
                        mism[c] = (nb, md)
                rep[key] = dict(engine=len(E), ref=len(R), common=len(M), only_engine=len(ek - rk),
                                only_ref=len(rk - ek), mismatched_cols=mism,
                                cols_missing_in_ref=[c for c in E.columns if c not in R.columns])
                print(key, "engine", len(E), "ref", len(R), "common", len(M), "onlyE", len(ek - rk),
                      "onlyR", len(rk - ek), "mism", mism, flush=True)
    json.dump(rep, open(f"{BASE}/audit/claude_skeptic/compare_{src}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1])
