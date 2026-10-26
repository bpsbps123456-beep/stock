import sys, json
import numpy as np, pandas as pd
sys.argv = ["recheck.py", "--tag", "dbg"] + sys.argv[1:]
src = open("recheck.py", encoding="utf-8").read()
cut = src.index("# ---------------------------------------------------------------- 5.")
g = {"__name__": "dbg"}
exec(compile(src[:cut], "recheck_head", "exec"), g)
df = g["df"]; M0 = g["M0"]
H = json.load(open(g["RES"] + "/judge_real.json", encoding="utf-8"))["cells"]
for cid, x in [("H66", "P_bar"), ("H03", "A_rz8"), ("H70", "P_share_rz8"), ("H08", "fight")]:
    hm = {m["month"]: (m["n_fit"], m.get("R2_obs")) for m in H[cid]["info"]["M0"]["months"]}
    out = []
    for mth in sorted(df["D"].dt.to_period("M").unique()):
        fit = df[(df["D"] < mth.to_timestamp()) & df["judge_ok"]].dropna(subset=[x] + M0)
        if len(fit) < 100: continue
        Xf = np.column_stack([np.ones(len(fit))] + [fit[c].values for c in M0]); y = fit[x].values
        b, *_ = np.linalg.lstsq(Xf, y, rcond=None); res = y - Xf @ b
        out.append((str(mth), len(fit), round(1 - res.var() / y.var(), 4), hm.get(str(mth))))
    print(cid, out)
