import sys, json, itertools
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
sys.argv = ["recheck.py", "--tag", "dbg", "--harness_sample"]
src = open("recheck.py", encoding="utf-8").read()
cut = src.index("def verdict(")
g = {"__name__": "dbg"}
exec(compile(src[:cut], "recheck_head", "exec"), g)
df = g["df"]; hark = g["hark"]; base_ok = g["base_ok"]; in_rng = g["in_rng"]
ks = g["ks"]; C = g["C"]; O = g["O"]; H = g["H"]; L = g["L"]; V = g["V"]; lr_s = g["lr_s"]; r_s = g["r_s"]
iD = df["iD"].astype(int).values; iP = iD - 1
# 변형 열 만들기
df["lnHL_plain"] = np.log(np.log(H[iD] / L[iD]))
with np.errstate(all="ignore"):
    df["HL_raw"] = np.log(H[iD] / L[iD])
    df.loc[~np.isfinite(df["HL_raw"]) | (df["HL_raw"] <= 0), "HL_raw"] = np.nan
def rvk(k, kind):
    out = []
    for p in iP:
        x = lr_s[p - k + 1:p + 1]
        if p - k + 1 < 1 or np.any(~np.isfinite(x)): out.append(np.nan); continue
        out.append(np.log(np.std(x, ddof=1)) if kind == "std" else np.log(np.sqrt(np.mean(x ** 2))))
    return np.array(out)
for k in (5, 22):
    df[f"lnRV{k}_std"] = rvk(k, "std")
vb = []
for p in iP:
    w = V[p - 19:p + 1]; w = w[np.isfinite(w) & (w > 0)]
    vb.append(np.log(V[p] / w.mean()) if len(w) >= 15 and np.isfinite(V[p]) and V[p] > 0 else np.nan)
df["vol_anom_incl"] = vb
# 같은 날 일정 대응
cal_cols = [c for c in df.columns if c.startswith("cal_")]
H_ = json.load(open(g["RES"] + "/judgment.json", encoding="utf-8"))["judgment_cells"]
target = {c: H_[c]["est"] for c in ("H03", "H08", "H70")}
base = g["HARK"]
def run(yc, rv, va, cal):
    res = {}
    for cid, x, ex, lab in [("H03", "A_rz8", [], False), ("H08", "fight", ["A_rz8", "ln_n_comments_onposts", "trunc_share"], False), ("H70", "P_share_rz8", [], True)]:
        ctr = [c for c in base if not c.startswith("cal_")]
        ctr = [("lnRV5_std" if c == "lnRV5" and rv == "std" else "lnRV22_std" if c == "lnRV22" and rv == "std" else c) for c in ctr]
        ctr = [("vol_anom_incl" if c == "vol_anom" and va == "incl" else c) for c in ctr]
        ctr = ctr[:9] + ex + ctr[9:] + (cal_cols if cal else [])
        back = base_ok & in_rng(df, g["BACK_LABEL"] if lab else g["BACK"]) & ~g["gap_excl"](cid)
        r = hark(df[x].astype(float), yc, ctr, back)
        res[cid] = (round(r["est"], 4), r["n"])
    return res
print("harness", target)
for yc, rv, va, cal in itertools.product(["lnHL", "HL_raw"], ["rms", "std"], ["excl", "incl"], [True, False]):
    print(yc, rv, va, "cal" if cal else "nocal", run(yc, rv, va, cal))
