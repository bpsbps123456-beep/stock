import sys
import numpy as np, pandas as pd
from scipy import stats
import warnings; warnings.filterwarnings("ignore")
sys.argv = ["recheck.py", "--tag", "dbg"] + sys.argv[1:]
src = open("recheck.py", encoding="utf-8").read()
cut = src.index("def verdict(")
g = {"__name__": "dbg"}
exec(compile(src[:cut], "recheck_head", "exec"), g)
df = g["df"]; base_ok = g["base_ok"]; in_rng = g["in_rng"]; ols = g["ols"]; nw = g["nw_cov"]; CT = g["H66_CTRL"]; rankz = g["rankz"]
back = base_ok & in_rng(df, g["BACK_LABEL"]) & ~g["gap_excl"]("H66")
xh, xp, _ = g["m0_expanding"]("P_bar")
d = df[back].copy(); d["xp"] = xp[back]; d = d.dropna(subset=["xp", "lnR5", "R5"] + CT).sort_values("D")
W = np.column_stack([np.ones(len(d))] + [d[c].values for c in CT] + [rankz(d.xp.values)])
y = d.lnR5.values
b, u = ols(y, W)
for L in (0, 4, 5, 8):
    V = nw(W, u, L) if L else np.linalg.pinv(W.T @ W) @ (W * u[:, None]).T @ (W * u[:, None]) @ np.linalg.pinv(W.T @ W)
    t = b[-1] / np.sqrt(V[-1, -1]); print(f"NW lag {L}: t={t:+.3f} p1={stats.t.sf(-t, len(d)-W.shape[1]):.4f}")
t, k = g["hodrick_t"](d, rankz(d.xp.values)); print(f"Hodrick 1B: t={t:+.3f} p1={stats.t.sf(-t, len(d)-k):.4f}  n={len(d)}")
