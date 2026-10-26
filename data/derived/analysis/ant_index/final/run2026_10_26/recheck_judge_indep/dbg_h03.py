import sys, json
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
extra = sys.argv[1:]
sys.argv = ["recheck.py", "--tag", "dbg", "--harness_sample"] + extra
src = open("recheck.py", encoding="utf-8").read()
cut = src.index("def verdict(")
g = {"__name__": "dbg"}
exec(compile(src[:cut], "recheck_head", "exec"), g)
df = g["df"]; M0 = g["M0"]; base_ok = g["base_ok"]; in_rng = g["in_rng"]; hark = g["hark"]; HARK = g["HARK"]
def fit_pred(train_mask, x):
    fit = df[train_mask].dropna(subset=[x] + M0)
    Xf = np.column_stack([np.ones(len(fit))] + [fit[c].values for c in M0]); b, *_ = np.linalg.lstsq(Xf, fit[x].values, rcond=None)
    res = fit[x].values - Xf @ b; sd = res.std(ddof=Xf.shape[1])
    ok = df[M0].notna().all(axis=1)
    Xa = np.column_stack([np.ones(ok.sum())] + [df.loc[ok, c].values for c in M0])
    xh = pd.Series(np.nan, index=df.index); xh[ok] = Xa @ b
    return xh, (df[x] - xh) / sd
H = json.load(open(g["RES"] + "/judgment.json", encoding="utf-8"))["judgment_cells"]
for cid, x, extra_c, lab in [("H03", "A_rz8", [], False), ("H08", "fight", ["A_rz8", "ln_n_comments_onposts", "trunc_share"], False), ("H70", "P_share_rz8", [], True)]:
    ctrls = HARK[:9] + extra_c + HARK[9:]
    back = base_ok & in_rng(df, g["BACK_LABEL"] if lab else g["BACK"]) & ~g["gap_excl"](cid)
    hj = H[cid]
    print(f"== {cid} harness: est {hj['est']:.4f} X̂ {hj['placebo_est']:.4f} X(pl) {hj['placebo_x_est']:.4f} X⊥ {hj['xperp_est']:.4f} n_pl {hj['n_placebo']}")
    for name, (xh, xp) in {"expanding": g["m0_expanding"](x)[:2], "full_insample": fit_pred(df.judge_ok, x),
                           "fit_thru_0930": fit_pred(df.judge_ok & (df.D < "2026-10-01"), x)}.items():
        m = back & xh.notna()
        a = hark(xh, "lnHL", ctrls, m); b_ = hark(df[x].astype(float), "lnHL", ctrls, m); c = hark(xp, "lnHL", ctrls, m)
        print(f"   {name:14s} n={a['n']} X̂ {a['est']:.4f}  X {b_['est']:.4f}  X⊥ {c['est']:.4f}  ratio {a['est']/b_['est']:.3f}")
