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
df = g["df"]; base_ok = g["base_ok"]; in_rng = g["in_rng"]; ic_test = g["ic_test"]
back = base_ok & in_rng(df, g["BACK_LABEL"]) & ~g["gap_excl"]("H66")
xh, xp, _ = g["m0_expanding"]("P_bar")
df["__xh"] = xh
xh2, xp2, _ = g["m0_expanding"]("__xh")
for nm, s in [("X⊥", xp), ("X̂", xh), ("X̂⊥(X̂를 다시 M0 잔차화)", xp2), ("X raw", df.P_bar)]:
    for mode in ("partial", "plain"):
        r = ic_test(s, back & xp.notna(), -1, with_t=False, mode=mode)
        print(f"{nm:28s} {mode:8s} n={r['n']} IC={r['est']:+.4f}")
