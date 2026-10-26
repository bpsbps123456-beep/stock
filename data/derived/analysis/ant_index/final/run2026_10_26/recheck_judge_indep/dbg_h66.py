import sys, json
import numpy as np, pandas as pd
from scipy import stats
sys.argv = ["recheck.py", "--tag", "dbg", "--harness_sample", "--m0_extra_cal"] + sys.argv[1:]
src = open("recheck.py", encoding="utf-8").read()
cut = src.index("def verdict(")
g = {"__name__": "dbg"}
import warnings; warnings.filterwarnings("ignore")
exec(compile(src[:cut], "recheck_head", "exec"), g)
df = g["df"]; M0 = g["M0"]; ic_test = g["ic_test"]; base_ok = g["base_ok"]; in_rng = g["in_rng"]
back = base_ok & in_rng(df, g["BACK_LABEL"]) & ~g["gap_excl"]("H66")
xhat, xperp, info = g["m0_expanding"]("P_bar")
d = df[back].copy(); d["xhat"] = xhat[back]; d["xperp"] = xperp[back]
d = d.dropna(subset=["xhat", "R5", "P_bar"] + g["H66_CTRL"])
print("n", len(d), d.D_str.iloc[0], d.D_str.iloc[-1])
print("corr(X,Xhat)", np.corrcoef(d.P_bar, d.xhat)[0, 1], "spearman X̂-R5", stats.spearmanr(d.xhat, d.R5)[0],
      "X-R5", stats.spearmanr(d.P_bar, d.R5)[0], "X⊥-R5", stats.spearmanr(d.xperp, d.R5)[0])
for c in M0:
    if d[c].std() > 0:
        print(f"  {c:14s} IC(R5)={stats.spearmanr(d[c], d.R5)[0]:+.3f}  corr(X)={stats.spearmanr(d[c], d.P_bar)[0]:+.3f}")
# 전 표본 적합(표본 안) X̂
fit = df[df.judge_ok].dropna(subset=["P_bar"] + M0)
Xf = np.column_stack([np.ones(len(fit))] + [fit[c].values for c in M0]); b, *_ = np.linalg.lstsq(Xf, fit.P_bar.values, rcond=None)
Xt = np.column_stack([np.ones(len(d))] + [d[c].values for c in M0])
print("in-sample full-fit X̂ IC", stats.spearmanr(Xt @ b, d.R5)[0])
coef = dict(zip(["const"] + M0, b)); print({k: round(v, 4) for k, v in coef.items()})
