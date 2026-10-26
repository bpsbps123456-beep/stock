import sys, json
import numpy as np, pandas as pd
from scipy import stats
import warnings; warnings.filterwarnings("ignore")
extra = sys.argv[1:]
sys.argv = ["recheck.py", "--tag", "dbg", "--harness_sample"] + extra
src = open("recheck.py", encoding="utf-8").read()
cut = src.index("def verdict(")
g = {"__name__": "dbg"}
exec(compile(src[:cut], "recheck_head", "exec"), g)
df = g["df"]; M0 = g["M0"]; base_ok = g["base_ok"]; in_rng = g["in_rng"]; ols = g["ols"]; CT = g["H66_CTRL"]
back = base_ok & in_rng(df, g["BACK_LABEL"]) & ~g["gap_excl"]("H66")
xh, xp, _ = g["m0_expanding"]("P_bar")
d = df[back].copy(); d["xh"] = xh[back]; d["xp"] = xp[back]; d = d.dropna(subset=["xh", "xp", "R5", "P_bar"] + CT)
Z = np.column_stack([np.ones(len(d))] + [d[c].values for c in CT])
ZM = np.column_stack([Z] + [d[c].values for c in M0 if d[c].std() > 0])
def rz(v): r = stats.rankdata(v); return (r - r.mean()) / r.std()
y = d.R5.values
defs = {
 "rank-partial(mine)": lambda x: np.corrcoef(ols(rz(x), Z)[1], ols(rz(y), Z)[1])[0, 1],
 "spearman(x, R5resid)": lambda x: stats.spearmanr(x, ols(y, Z)[1])[0],
 "spearman(xresid, R5resid)": lambda x: stats.spearmanr(ols(x, Z)[1], ols(y, Z)[1])[0],
 "pearson(xresid,R5resid)": lambda x: np.corrcoef(ols(x, Z)[1], ols(y, Z)[1])[0, 1],
 "plain spearman": lambda x: stats.spearmanr(x, y)[0],
 "rank-partial +M0 in-sample": lambda x: np.corrcoef(ols(rz(x), ZM)[1], ols(rz(y), ZM)[1])[0, 1],
 "spearman(x, R5resid|ctrl+M0)": lambda x: stats.spearmanr(x, ols(y, ZM)[1])[0],
}
print("harness: X⊥ -0.1075, X̂ -0.2471, X -0.2345 (n 69)  n=", len(d))
for k, f in defs.items():
    print(f"{k:30s} X⊥ {f(d.xp.values):+.4f}  X̂ {f(d.xh.values):+.4f}  X {f(d.P_bar.values):+.4f}")
print("--- 전 표본(표본 안) M0 적합의 X̂·X⊥")
for lab, trm in [("full", df.judge_ok), ("thru_0930", df.judge_ok & (df.D < "2026-10-01")), ("2026_only", df.judge_ok & (df.D >= "2026-01-01")), ("back_only", back)]:
    fit = df[trm].dropna(subset=["P_bar"] + M0)
    Xf = np.column_stack([np.ones(len(fit))] + [fit[c].values for c in M0]); b, *_ = np.linalg.lstsq(Xf, fit.P_bar.values, rcond=None)
    res = fit.P_bar.values - Xf @ b; sd = res.std()
    Xt = np.column_stack([np.ones(len(d))] + [d[c].values for c in M0]); xh2 = Xt @ b; xp2 = (d.P_bar.values - xh2) / sd
    print(lab)
    for k, f in defs.items():
        print(f"   {k:30s} X⊥ {f(xp2):+.4f}  X̂ {f(xh2):+.4f}  X {f(d.P_bar.values):+.4f}")
