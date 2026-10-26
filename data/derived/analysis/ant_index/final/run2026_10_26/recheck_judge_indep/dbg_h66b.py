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
df = g["df"]; M0 = g["M0"]; base_ok = g["base_ok"]; in_rng = g["in_rng"]; ic_stat = g["ic_stat"]
back = base_ok & in_rng(df, g["BACK_LABEL"]) & ~g["gap_excl"]("H66")
def fit_pred(train_mask, x="P_bar"):
    fit = df[train_mask].dropna(subset=[x] + M0)
    Xf = np.column_stack([np.ones(len(fit))] + [fit[c].values for c in M0]); b, *_ = np.linalg.lstsq(Xf, fit[x].values, rcond=None)
    res = fit[x].values - Xf @ b; sd = res.std(ddof=Xf.shape[1])
    ok = df[M0].notna().all(axis=1)
    Xa = np.column_stack([np.ones(ok.sum())] + [df.loc[ok, c].values for c in M0])
    xh = pd.Series(np.nan, index=df.index); xh[ok] = Xa @ b
    return xh, (df[x] - xh) / sd
variants = {}
variants["expanding"] = g["m0_expanding"]("P_bar")[:2]
variants["full_insample"] = fit_pred(df.judge_ok)
variants["fit_2025_only"] = fit_pred(df.judge_ok & (df.D.dt.year == 2025))
variants["fit_thru_0630"] = fit_pred(df.judge_ok & (df.D < "2026-07-01"))
variants["fit_back_only"] = fit_pred(back)
for k, (xh, xp) in variants.items():
    d = df[back].copy(); d["xh"] = xh[back]; d["xp"] = xp[back]
    d = d.dropna(subset=["xh", "xp", "R5", "P_bar"] + g["H66_CTRL"])
    r = {}
    for mode in ("partial", "plain"):
        r[mode] = [round(ic_stat(d, c, mode)[0], 4) for c in ("xp", "xh", "P_bar")]
    print(f"{k:15s} n={len(d)} [X⊥, X̂, X] partial={r['partial']} plain={r['plain']}")
print("harness: X⊥ -0.1075, X̂ -0.2471, X -0.2345 (n 69)")
print("--- 추가 가설")
def expanding_shift(lag_months=0, inclusive=False):
    xh = pd.Series(np.nan, index=df.index); xp = pd.Series(np.nan, index=df.index)
    for mth in sorted(df.D.dt.to_period("M").unique()):
        end = (mth + lag_months).to_timestamp()
        tr = df.judge_ok & ((df.D < end) if not inclusive else (df.D < (mth + 1).to_timestamp()))
        fit = df[tr].dropna(subset=["P_bar"] + M0)
        if len(fit) < 120: continue
        Xf = np.column_stack([np.ones(len(fit))] + [fit[c].values for c in M0]); b, *_ = np.linalg.lstsq(Xf, fit.P_bar.values, rcond=None)
        res = fit.P_bar.values - Xf @ b; sd = np.sqrt(res @ res / (len(fit) - np.linalg.matrix_rank(Xf)))
        tg = df[(df.D.dt.to_period("M") == mth)].dropna(subset=M0)
        Xt = np.column_stack([np.ones(len(tg))] + [tg[c].values for c in M0])
        xh[tg.index] = Xt @ b; xp[tg.index] = (tg.P_bar.values - Xt @ b) / sd
    return xh, xp
v2 = {"in_month(fit incl. month m)": expanding_shift(inclusive=True), "lead1(fit to m+1 start)": expanding_shift(lag_months=1),
      "fit_thru_0930(last fit, all)": fit_pred(df.judge_ok & (df.D < "2026-10-01"))}
for k, (xh, xp) in v2.items():
    d = df[back].copy(); d["xh"] = xh[back]; d["xp"] = xp[back]
    d = d.dropna(subset=["xh", "xp", "R5", "P_bar"] + g["H66_CTRL"])
    r = {mode: [round(ic_stat(d, c, mode)[0], 4) for c in ("xp", "xh", "P_bar")] for mode in ("partial", "plain")}
    print(f"{k:30s} n={len(d)} [X⊥, X̂, X] partial={r['partial']} plain={r['plain']}")
