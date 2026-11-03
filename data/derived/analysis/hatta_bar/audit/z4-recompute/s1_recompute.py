"""Recompute zone flags for every signal (all four periods), one symbol at a time -> out/mine_all.parquet.

Main definition (BRIEF4): exact 160-minute bars from the 10-minute bars as given; decision bar = last
160-minute bar with end <= signal bar start; zones = desk.zones.zones(pivots, decision, ref=decision close);
pattern A: L=low[t], C=close[t]; pattern B: L=low[k], C=close[t]; prior-day low = previous trading day's
09:30-16:00 low from the 10-minute bars.
Variants (power checks / sensitivity, not the definition): 'trim' (half-day post-13:00 10m bars dropped),
'approx' (12:30/14:30 bar boundaries), 'bt' (pattern B with L=low[t]), PDL same-day for 19:00 signals.
"""
import time

import numpy as np
import pandas as pd

from common import (OUT, SymZones, bars160, day_lows, flags_for, load_10m, load_1m, load_signals,
                    ny_from_unix_min, pdl_touch)


def prev_low(dl: pd.Series, day: pd.Timestamp, same_day: bool = False) -> tuple[float, object]:
    days = dl.index
    i = days.searchsorted(day, side="right" if same_day else "left") - 1
    if i < 0:
        return np.nan, None
    return float(dl.iloc[i]), days[i]


def main():
    sig = load_signals()
    rows = []
    t0 = time.time()
    syms = sorted(sig["sym"].unique())
    for n, sym in enumerate(syms):
        g = sig[sig["sym"] == sym]
        ten = load_10m(sym)
        szs = {m: SymZones(bars160(ten, m)) for m in ("exact", "trim", "approx")}
        dl, dl_t = day_lows(ten), day_lows(ten, trim=True)
        bex = szs["exact"].b
        for per, gp in g.groupby("period"):
            ts1, o1, h1, l1, c1 = load_1m(per, sym)
            for r in gp.itertuples(index=False):
                t, k = int(r.t), int(r.k)
                t_ok = 0 <= t < len(ts1) and ts1[t] == r.ts
                if not t_ok:  # fall back to timestamp match
                    t = int(np.searchsorted(ts1, r.ts))
                    if not (t < len(ts1) and ts1[t] == r.ts):
                        rows.append({"period": per, "sym": sym, "ts": r.ts, "pat": r.pat, "err": "ts_not_found"})
                        continue
                L = float(l1[k] if r.pat == "B" else l1[t])
                C = float(c1[t])
                f = flags_for(szs["exact"], int(r.ts), L, C)
                ft = flags_for(szs["trim"], int(r.ts), L, C)
                fa = flags_for(szs["approx"], int(r.ts), L, C)
                fb = flags_for(szs["exact"], int(r.ts), float(l1[t]), C) if r.pat == "B" else f
                loc = ny_from_unix_min([r.ts])[0]
                day = pd.Timestamp(loc.tz_localize(None).normalize())
                pdl, pdl_day = prev_low(dl, day)
                pdl_t, _ = prev_low(dl_t, day)
                pdl_s, _ = prev_low(dl, day, same_day=True)
                j = f["dec"]
                rows.append({
                    "period": per, "sym": sym, "ts": int(r.ts), "pat": r.pat, "t": int(r.t), "k": k,
                    "t_idx_ok": bool(t_ok), "mins": int(r.mins), "k_ts": int(ts1[k]),
                    "O": float(o1[t]), "H": float(h1[t]), "Lt": float(l1[t]), "C": C, "L": L,
                    "dec_day": bex["day"].iloc[j] if j >= 0 else pd.NaT, "dec_g": int(bex["g"].iloc[j]) if j >= 0 else -1,
                    "ref": float(bex["close"].iloc[j]) if j >= 0 else np.nan,
                    "z_hit": f["z_hit"], "z_str": f["z_str"], "z_near": f["z_near"], "z_rank": f["z_rank"],
                    "n_zones": f["n_zones"], "n_hit": f["n_hit"], "swings": f["swings"], "zlo": f["zlo"], "zhi": f["zhi"],
                    "z_sweep_guess": f["z_sweep_guess"],
                    "pdl": pdl, "pdl_day": pdl_day, "pdl_hit": pdl_touch(pdl, L, C),
                    "v_trim_hit": ft["z_hit"], "v_trim_str": ft["z_str"], "v_trim_near": ft["z_near"],
                    "v_approx_hit": fa["z_hit"], "v_approx_str": fa["z_str"], "v_approx_near": fa["z_near"],
                    "v_bt_hit": fb["z_hit"],
                    "v_pdl_trim": pdl_touch(pdl_t, L, C), "v_pdl_same": pdl_touch(pdl_s, L, C),
                })
        if n % 50 == 0:
            print(n, sym, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "mine_all.parquet", index=False)
    print("done", len(df), "errors", int(df.get("err", pd.Series(dtype=object)).notna().sum()), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
