"""Support/resistance zones of the user's TradingView indicator on 160-minute bars (desk_system.md §14.2).

Python canon of the zone part of `data/derived/analysis/kalman_zones/dip_scalper.pine` ('이격매수매도'):
- 160-minute bars, regular session only. Built from Yahoo 1-hour bars as [09:30,12:30) [12:30,14:30)
  [14:30,16:00) — the approximation of `rsi_breakout` (true bars start 09:30·12:10·14:50; `exact=True`
  builds those from 5-minute bars).
- Pivots = ta.pivotlow/pivothigh(5, 5): the middle bar is the window extreme and the 5 bars on its left
  are strictly beyond it. A pivot is known only when its 5 right bars have closed.
- Zones: the latest 60 confirmed swings sorted by price; a swing within 0.5% of a zone's low joins it.
  Stronger zones (more touches, then more recent) knock out zones whose middles sit within 1.2%.
  A zone whose middle is below the reference close is support; Z1 is the nearest support.
The earnings backtest `data/derived/analysis/earnings_support/` used the same rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

PIV_N, KEEP, TOL, GAP = 5, 60, 0.005, 0.012
REG0, REG1 = 570, 960  # 09:30, 16:00 in minutes


def bars160(h1: pd.DataFrame, exact: bool = False) -> pd.DataFrame:
    """Regular-session 160-minute bars (columns day, g, open, high, low, close) in time order."""
    if h1 is None or h1.empty or not isinstance(h1.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=["day", "g", "open", "high", "low", "close"])
    x = h1.tz_convert("America/New_York") if h1.index.tz is not None else h1.tz_localize("America/New_York")
    m = np.asarray(x.index.hour * 60 + x.index.minute)
    x = x[(m >= REG0) & (m < REG1)]
    m = np.asarray(x.index.hour * 60 + x.index.minute)
    g = (m - REG0) // 160 if exact else np.where(m < 750, 0, np.where(m < 870, 1, 2))
    day = x.index.tz_localize(None).normalize()
    b = x.groupby([day, g]).agg(open=("Open", "first"), high=("High", "max"), low=("Low", "min"), close=("Close", "last"))
    b.index.names = ["day", "g"]
    return b.reset_index()


def pivots(b: pd.DataFrame) -> np.ndarray:
    """Confirmed pivots as rows (bar, 0=low/1=high, price), in the order Pine adds them."""
    lo, hi = b["low"].to_numpy(float), b["high"].to_numpy(float)
    n, k = len(b), PIV_N
    if n < 2 * k + 1:
        return np.empty((0, 3))
    wl, wh = sliding_window_view(lo, 2 * k + 1), sliding_window_view(hi, 2 * k + 1)
    cl, ch = lo[k:n - k], hi[k:n - k]
    is_lo = (cl == wl.min(axis=1)) & (wl[:, :k] > cl[:, None]).all(axis=1)
    is_hi = (ch == wh.max(axis=1)) & (wh[:, :k] < ch[:, None]).all(axis=1)
    rows = []
    for j in np.flatnonzero(is_lo | is_hi):
        i = j + k
        if is_lo[j]:
            rows.append((i, 0, lo[i]))
        if is_hi[j]:
            rows.append((i, 1, hi[i]))
    return np.array(rows, dtype=float) if rows else np.empty((0, 3))


def zones(piv: np.ndarray, decision: int, ref: float) -> tuple[list, list, int]:
    """Zones known after bar `decision` closed: (supports nearest first, resistances nearest first, swings).

    A zone is [low, high, touches, last swing bar]."""
    conf = piv[piv[:, 0] <= decision - PIV_N] if len(piv) else piv
    total = len(conf)
    conf = conf[-KEEP:]
    if not len(conf):
        return [], [], 0
    px, bi = conf[:, 2], conf[:, 0].astype(int)
    merged: list[list] = []
    for i in np.argsort(px, kind="stable"):
        p, q = float(px[i]), int(bi[i])
        if merged and (p - merged[-1][0]) / merged[-1][0] <= TOL:
            merged[-1][1] = p
            merged[-1][2] += 1
            merged[-1][3] = max(merged[-1][3], q)
        else:
            merged.append([p, p, 1, q])
    keep: list[list] = []
    for z in sorted(merged, key=lambda z: z[2] * 1e7 + z[3], reverse=True):
        mid = (z[0] + z[1]) / 2
        if all(abs(mid - (k[0] + k[1]) / 2) / ((k[0] + k[1]) / 2) >= GAP for k in keep):
            keep.append(z)
    keep.sort(key=lambda z: (z[0] + z[1]) / 2)
    sup = [z for z in keep if (z[0] + z[1]) / 2 < ref][::-1]
    res = [z for z in keep if (z[0] + z[1]) / 2 >= ref]
    return sup, res, total


ENDS = {0: 750, 1: 870, 2: 960}  # approximate bars end 12:30·14:30·16:00


def current(h1: pd.DataFrame, ref: float | None = None, now: pd.Timestamp | None = None) -> dict:
    """Zones after the last finished 160-minute bar; `ref` defaults to that bar's close.

    A bar still forming at `now` is dropped, so the zones match what the chart showed then."""
    b = bars160(h1)
    now = pd.Timestamp.now(tz="America/New_York") if now is None else now.tz_convert("America/New_York")
    ends = [pd.Timestamp(d).tz_localize("America/New_York") + pd.Timedelta(minutes=ENDS[int(g)])
            for d, g in zip(b["day"], b["g"])]
    b = b[[e <= now for e in ends]].reset_index(drop=True)
    if b.empty:
        return {"supports": [], "resistances": [], "swings": 0, "asof": None}
    last = len(b) - 1
    ref = float(b["close"].iloc[-1]) if ref is None else ref
    sup, res, total = zones(pivots(b), last, ref)
    when = lambda i: f"{pd.Timestamp(b['day'].iloc[i]):%Y-%m-%d}·{int(b['g'].iloc[i]) + 1}봉"
    as_dict = lambda z, tag, i: {"name": f"{tag}{i + 1}", "lo": round(z[0], 4), "hi": round(z[1], 4),
                                 "touches": int(z[2]), "last_swing": when(int(z[3]))}
    return {"supports": [as_dict(z, "Z", i) for i, z in enumerate(sup)],
            "resistances": [as_dict(z, "R", i) for i, z in enumerate(res)],
            "swings": total, "ref": ref, "asof": f"{pd.Timestamp(b['day'].iloc[-1]):%Y-%m-%d}"}
