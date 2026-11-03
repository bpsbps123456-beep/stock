"""Three random z_hit examples (seed 20261103) for eyeballing, plus a tie-out of z4_primary means to z4_signals.csv."""
import json

import numpy as np
import pandas as pd

from common import CACHE, OUT, RESULTS, SEED, SymZones, bars160, load_10m, ny_from_unix_min
from desk.zones import KEEP, PIV_N

mine = pd.read_parquet(OUT / "mine_all.parquet")
zf = pd.read_parquet(CACHE / "zone_flags.parquet")
hits = mine[mine["z_hit"]].reset_index(drop=True)
rng = np.random.default_rng(SEED)
pick = hits.iloc[np.sort(rng.choice(len(hits), size=3, replace=False))]
SLOT = {0: "09:30-12:10", 1: "12:10-14:50", 2: "14:50-16:00"}
ex = []
for r in pick.itertuples(index=False):
    sz = SymZones(bars160(load_10m(r.sym), "exact"))
    j = sz.decision(int(r.ts))
    b = sz.b
    conf = sz.piv[sz.piv[:, 0] <= j - PIV_N][-KEEP:]
    mem = conf[(conf[:, 2] >= r.zlo) & (conf[:, 2] <= r.zhi)]
    swings = [f"{b['day'].iloc[int(i)]:%Y-%m-%d} {SLOT[int(b['g'].iloc[int(i)])]} {'low' if k == 0 else 'high'} {p:g}"
              for i, k, p in mem]
    e = zf[(zf.period == r.period) & (zf.sym == r.sym) & (zf.ts == r.ts) & (zf.pat == r.pat)].iloc[0]
    ex.append({
        "period": r.period, "sym": r.sym, "pat": r.pat,
        "signal_bar_NY": f"{ny_from_unix_min([r.ts])[0]:%Y-%m-%d %H:%M}",
        "signal_OHLC": [r.O, r.H, r.Lt, r.C],
        "break_bar_NY_low": (f"{ny_from_unix_min([r.k_ts])[0]:%Y-%m-%d %H:%M}", r.L) if r.pat == "B" else None,
        "L_used": r.L, "C_used": r.C,
        "zone_lo_hi": [r.zlo, r.zhi], "zone_touches": int(r.z_str),
        "touch_band": [round(r.zlo * 0.995, 4), round(r.zhi * 1.001, 4)], "close_floor": round(r.zlo * 0.999, 4),
        "zone_swings(160m bar, kind, price)": swings,
        "decision_bar": f"{b['day'].iloc[j]:%Y-%m-%d} {SLOT[int(b['g'].iloc[j])]}", "ref_close": float(b["close"].iloc[j]),
        "nearest_support": bool(r.z_near), "rank_in_supports": int(r.z_rank),
        "prior_day_low": (f"{pd.Timestamp(r.pdl_day):%Y-%m-%d}", r.pdl), "pdl_hit": bool(r.pdl_hit),
        "engine": {"z_hit": bool(e.z_hit), "z_str": int(e.z_str), "z_near": bool(e.z_near), "pdl_hit": bool(e.pdl_hit)},
    })

# tie-out: z4_primary y2026 means vs z4_signals.csv rows
z4 = pd.read_csv(RESULTS / "z4_signals.csv")
prim = json.loads((RESULTS / "z4_primary.json").read_text(encoding="utf-8"))
tie = {}
for hyp in ("Z1", "Z2"):
    d = z4[z4.hyp == hyp]
    p = prim[hyp]["y2026"]
    tie[hyp] = {"n": (len(d), p["n"]), "hit": (float(d.hit_e2.mean()), p["hit"]), "r1": (float(d.r1_e2.mean()), p["r1"]),
                "r1n": (float(d.r1n.mean()), p["r1n"]), "days": (int(d.day.nunique()), p["days"]),
                "syms": (int(d.sym.nunique()), p["syms"])}
out = {"examples": ex, "z4_primary_vs_signals_csv": tie}
(OUT / "s4_examples.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, indent=1, default=str))
