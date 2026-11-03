"""Headline z_hit shares (2026 = 2026h1 + 2026, before dedup) and the Z1/Z2 dedup'd sets vs results/z4_signals.csv."""
import json

import numpy as np
import pandas as pd

from common import CACHE, OUT, RESULTS

KEY = ["period", "sym", "ts", "pat"]
zf = pd.read_parquet(CACHE / "zone_flags.parquet")
mine = pd.read_parquet(OUT / "mine_all.parquet")
m = zf.merge(mine[KEY + ["z_hit", "mins"]], on=KEY, suffixes=("_e", "_m"))
m["y"] = np.where(m["period"].str.startswith("2026"), "2026", "2025")
m["day"] = pd.to_datetime(m["ts"] * 60, unit="s", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y%m%d").astype(int)
res = {}

# ---- 1. shares before dedup (all sig rows are inside the order window 04:00-10:59 / 19:00-19:29 ET) ----
win = ((m["mins"] >= 240) & (m["mins"] <= 659)) | ((m["mins"] >= 1140) & (m["mins"] <= 1169))
res["all_sig_rows_in_order_window"] = bool(win.all())
sh = {}
for (y, pat), d in m.groupby(["y", "pat"]):
    sh[f"y{y}_{pat}"] = {"n": len(d), "hit_mine": int(d["z_hit_m"].sum()), "hit_engine": int(d["z_hit_e"].sum()),
                        "share_mine": round(float(d["z_hit_m"].mean()), 4), "share_engine": round(float(d["z_hit_e"].mean()), 4)}
for (per, pat), d in m.groupby(["period", "pat"]):
    sh[f"{per}_{pat}"] = {"n": len(d), "share_mine": round(float(d["z_hit_m"].mean()), 4),
                          "share_engine": round(float(d["z_hit_e"].mean()), 4)}
res["shares_before_dedup"] = sh


# ---- 2. dedup variants ----
def dedup(d: pd.DataFrame, gap: int = 30, rule: str = "ge") -> pd.DataFrame:
    """rule ge: keep if ts - last kept >= gap; gt: > gap; chain: drop if within gap of the previous signal (kept or not)."""
    d = d.sort_values(["sym", "ts"])
    keep = np.zeros(len(d), bool)
    sym, ts = d["sym"].to_numpy(), d["ts"].to_numpy(np.int64)
    last = prev = None
    cur = None
    for i in range(len(d)):
        if sym[i] != cur:
            cur, last, prev = sym[i], None, None
        if rule == "chain":
            ok = prev is None or ts[i] - prev >= gap
            prev = ts[i]
        elif rule == "gt":
            ok = last is None or ts[i] - last > gap
        else:
            ok = last is None or ts[i] - last >= gap
        if ok:
            keep[i] = True
            last = ts[i]
    return d[keep]


z4 = pd.read_csv(RESULTS / "z4_signals.csv")
prim = json.loads((RESULTS / "z4_primary.json").read_text(encoding="utf-8"))
HYP = {"Z1": "A", "Z2": "B"}


def keyset(d):
    return set(zip(d["period"], d["sym"], d["ts"].astype(np.int64)))


dd = {}
for hyp, pat in HYP.items():
    ref = keyset(z4[z4["hyp"] == hyp])
    for flag in ("z_hit_e", "z_hit_m"):
        for rule in ("ge", "gt", "chain"):
            for order in ("filter_then_dedup", "dedup_then_filter"):
                base = m[(m["y"] == "2026") & (m["pat"] == pat)]
                if order == "filter_then_dedup":
                    s = dedup(base[base[flag]], rule=rule)
                else:
                    s = dedup(base, rule=rule)
                    s = s[s[flag]]
                ks = keyset(s)
                dd[f"{hyp}|{flag}|{rule}|{order}"] = {"n": len(ks), "n_z4": len(ref), "exact": ks == ref,
                                                      "only_mine": len(ks - ref), "only_z4": len(ref - ks)}
res["dedup_vs_z4_signals"] = dd

# ---- 3. counts vs z4_primary.json under the matching rule (ge, filter then dedup) ----
chk = {}
for hyp, pat in HYP.items():
    for y in ("2026", "2025"):
        for zone, label in ((True, f"y{y}"), (False, f"no_zone_{y}")):
            base = m[(m["y"] == y) & (m["pat"] == pat)]
            s = dedup(base[base["z_hit_m"] == zone])
            e = prim[hyp][label]
            chk[f"{hyp}_{label}"] = {"n": (len(s), e["n"]), "days": (int(s["day"].nunique()), e["days"]),
                                     "syms": (int(s["sym"].nunique()), e["syms"])}
    for per in ("2025h1", "2025", "2026h1", "2026"):
        base = m[(m["period"] == per) & (m["pat"] == pat)]
        s = dedup(base[base["z_hit_m"]])
        e = prim[hyp]["by_period"][per]
        chk[f"{hyp}_{per}"] = {"n": (len(s), e["n"]), "days": (int(s["day"].nunique()), e["days"]),
                               "syms": (int(s["sym"].nunique()), e["syms"])}
res["counts_mine_vs_primary(mine, engine)"] = chk
res["all_counts_equal"] = all(v[0] == v[1] for c in chk.values() for v in c.values())

# ---- 4. z4_signals.csv fields that tie back to cond (t, day, outcome) ----
cs = []
for per in ("2026h1", "2026"):
    c = pd.read_parquet(CACHE / f"cond_{per}.parquet", columns=["arm", "pat", "sym", "ts", "t", "day", "r1_e2", "hit_e2"])
    c = c[c["arm"] == "sig"].assign(period=per)
    cs.append(c)
cs = pd.concat(cs)
j = z4.assign(pat=z4["hyp"].map(HYP)).merge(cs, on=["period", "sym", "ts", "pat"], suffixes=("", "_c"), how="left")
res["z4_rows_tie_to_cond"] = {"rows": len(j), "found": int(j["t_c"].notna().sum()),
                              "t_equal": int((j["t"] == j["t_c"]).sum()), "day_equal": int((j["day"] == j["day_c"]).sum()),
                              "r1_e2_equal": int(np.isclose(j["r1_e2"], j["r1_e2_c"]).sum()),
                              "hit_e2_equal": int((j["hit_e2"] == j["hit_e2_c"]).sum())}

(OUT / "s3_headline.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(json.dumps(res, indent=1, default=str))
