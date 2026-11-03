"""Z4 audit: independent zone-touch recompute (BRIEF4.md). Shared loaders and per-symbol flag computation.

Own code: exact 160-minute bars from the 10-minute bars, decision bar, touch test, prior-day low.
Canon called as allowed: desk.zones.pivots / desk.zones.zones (the user's indicator rules).
Does not read or import hatta_bar/code/zone_eval.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HB = HERE.parents[1]            # .../hatta_bar
ROOT = HERE.parents[5]          # repo root (desk package)
sys.path.insert(0, str(ROOT))
from desk.zones import pivots, zones  # noqa: E402  (canon)

CACHE = HB / "cache"
RESULTS = HB / "results"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
DIR1M = {"2026": "alp1m", "2026h1": "alp1m_2026h1", "2025": "alp1m_2025", "2025h1": "alp1m_2025h1"}
PERIODS = ["2025h1", "2025", "2026h1", "2026"]
NY = "America/New_York"
SEED = 20261103

# NYSE early closes (13:00) inside the 10-minute data range 2024-09-03..2026-10-28.
HALF_DAYS = pd.to_datetime(["2024-11-29", "2024-12-24", "2025-07-03", "2025-11-28", "2025-12-24"])


def unix_min(idx: pd.DatetimeIndex) -> np.ndarray:
    """Unix minutes of a tz-aware index (index unit may be us or ns)."""
    return (idx.as_unit("ns").asi8 // 60_000_000_000).astype(np.int64)


def ny_from_unix_min(ts) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(ts, dtype=np.int64) * 60, unit="s", utc=True).tz_convert(NY)


def load_signals() -> pd.DataFrame:
    out = []
    for per in PERIODS:
        c = pd.read_parquet(CACHE / f"cond_{per}.parquet", columns=["arm", "pat", "sym", "ts", "t", "k", "day", "mins"])
        c = c[c.arm == "sig"].drop(columns="arm")
        c.insert(0, "period", per)
        out.append(c)
    return pd.concat(out, ignore_index=True)


def load_10m(sym: str) -> pd.DataFrame:
    x = pd.read_parquet(CACHE / "alp10m" / f"{sym}.parquet")
    x = x.tz_convert(NY) if x.index.tz is not None else x.tz_localize(NY)
    return x.sort_index()


def bars160(ten: pd.DataFrame, mode: str = "exact") -> pd.DataFrame:
    """Regular 160-minute bars built from 10-minute bars.

    mode 'exact'   : 09:30-12:10, 12:10-14:50, 14:50-16:00 (BRIEF4 rule), bars as given.
    mode 'trim'    : same, but drop 10-minute bars starting >= 13:00 on the five NYSE half days
                     (sensitivity only: the cached 'regular' 10-minute files keep post-13:00 trades there).
    mode 'approx'  : 09:30-12:30, 12:30-14:30, 14:30-16:00 (desk.zones default, wrong for BRIEF4; power check).
    Columns: day, g, open, high, low, close, nbar, end_ts (unix minutes of the nominal bar end).
    """
    m = ten.index.hour * 60 + ten.index.minute
    keep = (m >= 570) & (m < 960)
    x, m = ten[keep], np.asarray(m[keep])
    day = x.index.tz_localize(None).normalize()
    if mode == "trim":
        k2 = ~(day.isin(HALF_DAYS) & (m >= 780))
        x, m, day = x[k2], m[k2], day[k2]
    if mode == "approx":
        g = np.where(m < 750, 0, np.where(m < 870, 1, 2))
        ends = np.array([750, 870, 960])
    else:
        g = (m - 570) // 160
        ends = np.array([730, 890, 960])
    df = pd.DataFrame({"day": day, "g": g, "o": x["Open"].to_numpy(float), "h": x["High"].to_numpy(float),
                       "l": x["Low"].to_numpy(float), "c": x["Close"].to_numpy(float)})
    b = (df.groupby(["day", "g"], sort=True)
           .agg(open=("o", "first"), high=("h", "max"), low=("l", "min"), close=("c", "last"), nbar=("o", "size"))
           .reset_index())
    end_local = (b["day"] + pd.to_timedelta(ends[b["g"].to_numpy()], unit="min")).dt.tz_localize(NY)
    b["end_ts"] = unix_min(pd.DatetimeIndex(end_local))
    return b


def day_lows(ten: pd.DataFrame, trim: bool = False) -> pd.Series:
    m = np.asarray(ten.index.hour * 60 + ten.index.minute)
    day = ten.index.tz_localize(None).normalize()
    keep = (m >= 570) & (m < 960)
    if trim:
        keep &= ~(day.isin(HALF_DAYS) & (m >= 780))
    return ten["Low"][keep].groupby(day[keep]).min().sort_index()


def load_1m(per: str, sym: str):
    """1-minute bars as the engine indexes them: duplicates removed, High<Low or Low<=0 dropped."""
    f = CACHE / DIR1M[per] / f"{sym}.parquet"
    x = pd.read_parquet(f, columns=["Open", "High", "Low", "Close"])
    x = x[~x.index.duplicated(keep="first")]
    x = x[~((x["High"] < x["Low"]) | (x["Low"] <= 0))]
    return (unix_min(x.index), x["Open"].to_numpy(float), x["High"].to_numpy(float),
            x["Low"].to_numpy(float), x["Close"].to_numpy(float))


def touched(zlist, L: float, C: float):
    """Zones touched per BRIEF4: zlo*0.995 <= L <= zhi*1.001 and C >= zlo*0.999."""
    return [z for z in zlist if z[0] * 0.995 <= L <= z[1] * 1.001 and C >= z[0] * 0.999]


def swept(zlist, L: float, C: float):
    """Guess for engine z_sweep (not required): tail more than 0.5% below a zone low, close back >= zlo*0.999."""
    return [z for z in zlist if L < z[0] * 0.995 and C >= z[0] * 0.999]


class SymZones:
    """Point-in-time zones for one symbol and one bar construction."""

    def __init__(self, b: pd.DataFrame):
        self.b = b
        self.end = b["end_ts"].to_numpy(np.int64)
        self.close = b["close"].to_numpy(float)
        self.piv = pivots(b)
        self._cache: dict[int, tuple] = {}

    def decision(self, sig_start: int) -> int:
        """Index of the last 160-minute bar whose end time <= signal bar start (-1 if none)."""
        return int(np.searchsorted(self.end, sig_start, side="right")) - 1

    def at(self, j: int):
        if j not in self._cache:
            if j < 0:
                self._cache[j] = ([], [], 0)
            else:
                self._cache[j] = zones(self.piv, j, float(self.close[j]))
        return self._cache[j]


def flags_for(sz: SymZones, sig_start: int, L: float, C: float) -> dict:
    j = sz.decision(sig_start)
    sup, res, total = sz.at(j)
    zl = list(sup) + list(res)
    hit = touched(zl, L, C)
    out = {"dec": j, "n_zones": len(zl), "n_hit": len(hit), "swings": total}
    if hit:
        top = max(hit, key=lambda z: (z[0] + z[1]) / 2)
        rank = next((i + 1 for i, z in enumerate(sup) if z is top), -1)
        out.update(z_hit=True, z_str=int(top[2]), z_near=bool(sup) and (sup[0] is top), z_rank=rank,
                   zlo=float(top[0]), zhi=float(top[1]))
    else:
        out.update(z_hit=False, z_str=0, z_near=False, z_rank=-1, zlo=np.nan, zhi=np.nan)
    out["z_sweep_guess"] = bool(swept(zl, L, C))
    return out


def pdl_touch(pdl: float, L: float, C: float) -> bool:
    if not np.isfinite(pdl):
        return False
    return bool(pdl * 0.995 <= L <= pdl * 1.001 and C >= pdl * 0.999)
