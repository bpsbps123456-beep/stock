"""Probe data conventions (1-min duplicates/invalid rows, ts <-> bar index, mins, engine flag relations)."""
import numpy as np
import pandas as pd
from pathlib import Path

HB = Path(__file__).resolve().parents[2]
C = HB / "cache"
DIRS = {"2026": "alp1m", "2026h1": "alp1m_2026h1", "2025": "alp1m_2025", "2025h1": "alp1m_2025h1"}

zf = pd.read_parquet(C / "zone_flags.parquet")
print("engine z_near vs z_rank:")
print(pd.crosstab(zf.z_near, zf.z_rank.clip(-1, 3)))
print("z_hit vs z_rank>=0:")
print(pd.crosstab(zf.z_hit, zf.z_rank >= 0))
print("z_hit vs z_sweep:")
print(pd.crosstab(zf.z_hit, zf.z_sweep))
print("z_hit vs z_str>0:")
print(pd.crosstab(zf.z_hit, zf.z_str > 0))

for per, d in DIRS.items():
    cond = pd.read_parquet(C / f"cond_{per}.parquet", columns=["arm", "pat", "sym", "ts", "t", "k", "day", "mins", "sess"])
    s = cond[cond.arm == "sig"]
    for sym in ["AA", "NVDA", "RGTI"]:
        f = C / d / f"{sym}.parquet"
        if not f.exists():
            continue
        x = pd.read_parquet(f)
        ndup = int(x.index.duplicated().sum())
        bad = int(((x.High < x.Low) | (x.Low <= 0)).sum())
        mono = bool(x.index.is_monotonic_increasing)
        y = x[~x.index.duplicated(keep="first")]
        y = y[~((y.High < y.Low) | (y.Low <= 0))]
        tsm = (y.index.as_unit("ns").asi8 // 60_000_000_000).astype(np.int64)
        ss = s[s.sym == sym]
        ok_t = (tsm[ss.t.to_numpy()] == ss.ts.to_numpy()).mean() if len(ss) else np.nan
        # mins vs NY minute-of-day
        loc = pd.to_datetime(ss.ts.to_numpy() * 60, unit="s", utc=True).tz_convert("America/New_York")
        mod = loc.hour * 60 + loc.minute
        ok_m = (mod == ss.mins.to_numpy()).mean() if len(ss) else np.nan
        ok_day = (loc.strftime("%Y%m%d").astype(int) == ss.day.to_numpy()).mean() if len(ss) else np.nan
        print(per, sym, "rows", len(x), "dup", ndup, "bad", bad, "mono", mono, "n_sig", len(ss),
              "idx[t]==ts", ok_t, "mins==NYmod", ok_m, "day==NYday", ok_day)
    # sess vs mins
    print(per, s.groupby("sess").mins.agg(["min", "max", "count"]).to_string())
