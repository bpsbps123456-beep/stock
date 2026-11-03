"""Side checks: (1) effect of the half-day post-13:00 10-minute bars on the 2026 Z1/Z2 sets,
(2) where the engine's z_sweep (aux flag, definition not in BRIEF4) differs from a plain guess."""
import json

import numpy as np
import pandas as pd

from common import CACHE, OUT, ny_from_unix_min

KEY = ["period", "sym", "ts", "pat"]
mine = pd.read_parquet(OUT / "mine_all.parquet")
zf = pd.read_parquet(CACHE / "zone_flags.parquet")
m = zf.merge(mine, on=KEY, suffixes=("_e", "_m"))


def dedup(d):
    d = d.sort_values(["sym", "ts"])
    keep, last, cur = [], None, None
    for s, t in zip(d["sym"], d["ts"]):
        if s != cur:
            cur, last = s, None
        ok = last is None or t - last >= 30
        keep.append(ok)
        if ok:
            last = t
    return d[np.array(keep, bool)]


res = {}
flip = m[m["z_hit_m"] != m["v_trim_hit"]]
res["halfday_trim_flips_by_period"] = {f"{a}_{b}": int(v) for (a, b), v in flip.groupby(["period", "pat"]).size().items()}
res["halfday_trim_flip_months"] = (pd.Series(ny_from_unix_min(flip["ts"]).strftime("%Y-%m")).value_counts()
                                   .sort_index().astype(int).to_dict())
y26 = m[m["period"].isin(["2026h1", "2026"])]
for hyp, pat in (("Z1", "A"), ("Z2", "B")):
    b = y26[y26["pat"] == pat]
    s0, s1 = dedup(b[b["z_hit_m"]]), dedup(b[b["v_trim_hit"]])
    k0, k1 = set(zip(s0.sym, s0.ts)), set(zip(s1.sym, s1.ts))
    res[f"{hyp}_2026_n_asis_vs_trim"] = {"asis": len(k0), "trim": len(k1), "only_asis": len(k0 - k1), "only_trim": len(k1 - k0)}

# z_sweep: compare engine with guess 'any zone: L < zlo*0.995 and C >= zlo*0.999'
d = m[m["z_sweep"] != m["z_sweep_guess"]]
res["z_sweep_guess_mismatch"] = {"n": len(d), "engine_true_guess_false": int((d["z_sweep"] & ~d["z_sweep_guess"]).sum()),
                                 "engine_false_guess_true": int((~d["z_sweep"] & d["z_sweep_guess"]).sum()),
                                 "by_z_hit": {str(a): int(v) for a, v in d.groupby("z_hit_e").size().items()}}
(OUT / "s5_sensitivity.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(json.dumps(res, indent=1, default=str))
