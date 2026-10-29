import json, numpy as np, pandas as pd
from compare import cmp_col, BASE
rep = json.load(open(f"{BASE}/audit/claude_skeptic/compare_yf1m.json"))
print("missing cols per key:", {k: v["cols_missing_in_ref"] for k, v in rep.items() if v["cols_missing_in_ref"]})
E = pd.read_parquet(f"{BASE}/cache/cand_yf1m/A_120_0.0005.parquet")
R = pd.read_parquet(f"{BASE}/audit/claude_skeptic/ref_yf1m/A_120_0.0005.parquet")
cols = [c for c in E.columns if c not in ("sid","t","sym","n_same_min") and c in R.columns]
print("n compared cols", len(cols))
M = E.merge(R, on=["sid","t"], suffixes=("_e","_r"))
# perturbation test: shift one value per column type -> must be detected
for c in ["r1_e2","touch3","persist","bt_r1_e2","bv_r1_e2","r2_e2","dd"]:
    x = M[c+"_r"].to_numpy().copy()
    if x.dtype == bool: x[5] = ~x[5]
    else: x = x.astype(float); x[5] = x[5] + 1e-6 if not np.isnan(x[5]) else 1.0
    print(c, "detected", cmp_col(M[c+"_e"], x)[0])
# non-trivial coverage: how many rows have non-null/true in key fields (counts only)
print("valid all True:", bool(M.valid_e.all()), "persist true count", int(M.persist_e.sum()), "touch3>=2", int((M.touch3_e>=2).sum()), "level3>=2", int((M.level3_e>=2).sum()), "r2 non-nan", int(M.r2_e2_e.notna().sum()), "amb", int(M.amb_e2_e.sum()), "race nan", int(M.race_e2_e.isna().sum()), "bt nan", int(M.bt_r1_e2_e.isna().sum()))
print("volr inf count", int(np.isinf(M.volr_e).sum()), "volr==0", int((M.volr_e==0).sum()))
