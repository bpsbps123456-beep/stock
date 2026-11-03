"""Optional: which simple definition reproduces the engine's aux flag z_sweep (BRIEF4: '구역 바닥을 0.5% 넘게 뚫은 꼬리').

Only rows where the broad guess is True are re-examined (the engine is a strict subset of the broad guess)."""
import json

import numpy as np
import pandas as pd

from common import CACHE, OUT, SymZones, bars160, load_10m

KEY = ["period", "sym", "ts", "pat"]
mine = pd.read_parquet(OUT / "mine_all.parquet")
zf = pd.read_parquet(CACHE / "zone_flags.parquet")
m = zf.merge(mine, on=KEY, suffixes=("_e", "_m"))
cand = m[m["z_sweep_guess"]]

V = {
    "any_zone": lambda zl, sup, L, C, hit: any(L < z[0] * .995 and C >= z[0] * .999 for z in zl),
    "supports_only": lambda zl, sup, L, C, hit: any(L < z[0] * .995 and C >= z[0] * .999 for z in sup),
    "no_touch_and_any": lambda zl, sup, L, C, hit: (not hit) and any(L < z[0] * .995 and C >= z[0] * .999 for z in zl),
    "nearest_zone_above_low": lambda zl, sup, L, C, hit: (lambda a: bool(a) and C >= min(a, key=lambda z: z[0])[0] * .999)(
        [z for z in zl if L < z[0] * .995]),
    "close_ge_zlo": lambda zl, sup, L, C, hit: any(L < z[0] * .995 and C >= z[0] for z in zl),
    "depth_le_2pct": lambda zl, sup, L, C, hit: any(z[0] * .98 <= L < z[0] * .995 and C >= z[0] * .999 for z in zl),
    "depth_le_1pct": lambda zl, sup, L, C, hit: any(z[0] * .99 <= L < z[0] * .995 and C >= z[0] * .999 for z in zl),
}
rows = []
for sym, g in cand.groupby("sym"):
    sz = SymZones(bars160(load_10m(sym), "exact"))
    for r in g.itertuples(index=False):
        sup, res, _ = sz.at(sz.decision(int(r.ts)))
        zl = list(sup) + list(res)
        hit = any(z[0] * .995 <= r.L <= z[1] * 1.001 and r.C >= z[0] * .999 for z in zl)
        rows.append({"eng": bool(r.z_sweep), **{k: bool(f(zl, sup, r.L, r.C, hit)) for k, f in V.items()}})
d = pd.DataFrame(rows)
out = {k: {"agree": int((d[k] == d["eng"]).sum()), "n": len(d)} for k in V}
(OUT / "s6_sweep_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
