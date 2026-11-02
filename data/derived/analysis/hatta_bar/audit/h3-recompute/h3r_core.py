"""Independent re-implementation of hatta_bar v3 original pattern (1) for the H3 audit.

Written only from BRIEF.md (v3), BRIEF2.md, BRIEF3.md and the audit task text.
Does NOT import or read hatta_bar/code/*.  Reads only cached files (no network).
"""
import json
import os

import numpy as np
import pandas as pd

BASE = r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar"
CACHE = BASE + "/cache/alp1m_2026h1"
OUT = BASE + "/audit/h3-recompute/out"

# v3 original pattern (1) parameters (BRIEF.md "원안")
N = 120          # window bars for S'
TOL = 0.0005     # 0.05 %
P = 2            # swing half-width
LEV_BAND = 0.01  # points with price <= S' * 1.01
LEV_GAP = 0.0015 # new level when > first point of current level * (1 + 0.15 %)
K_MIN = 2        # level count >= 2
TR_W = 30        # trend window b-29..b
TR_MIN = 24      # 80 % of 30 bars
D_MIN = 0.02     # 120-minute drop >= 2 %
R_MIN = 0.5      # lower wick ratio >= 0.5
TRADES_MIN = 5
DOLLAR_MIN = 25_000.0
RANGE_MIN = 0.002
CNT30_MIN = 15


def load_universe():
    with open(BASE + "/results/universe2.json", encoding="utf-8") as f:
        return json.load(f)


def load_cost():
    with open(BASE + "/results/cost_model.json", encoding="utf-8") as f:
        return json.load(f)


def load_bars(sym):
    df = pd.read_parquet(f"{CACHE}/{sym}.parquet")
    idx = df.index
    assert idx.is_monotonic_increasing and not idx.has_duplicates
    ts = idx.as_unit("s").asi8 // 60  # unix minutes (UTC based)
    day = (idx.year * 10000 + idx.month * 100 + idx.day).to_numpy().astype(np.int64)
    mins = (idx.hour * 60 + idx.minute).to_numpy().astype(np.int64)
    d = dict(
        ts=np.asarray(ts, dtype=np.int64),
        day=day,
        mins=mins,
        o=df["Open"].to_numpy(np.float64),
        h=df["High"].to_numpy(np.float64),
        l=df["Low"].to_numpy(np.float64),
        c=df["Close"].to_numpy(np.float64),
        v=df["Volume"].to_numpy(np.float64),
        n=df["Trades"].to_numpy(np.float64),
        vw=df["VWAP"].to_numpy(np.float64),
    )
    return d


def ema(x, n):
    return pd.Series(x).ewm(alpha=2.0 / (n + 1.0), adjust=False).mean().to_numpy()


def day_vwap(d, start_min=None):
    """Cumulative VWAP from the first bar of the NY date (or from first bar with mins>=start_min)."""
    vw = d["vw"].copy()
    bad = ~np.isfinite(vw) | (vw <= 0)
    vw[bad] = (d["h"][bad] + d["l"][bad] + d["c"][bad]) / 3.0
    v = d["v"].copy()
    pv = vw * v
    if start_min is not None:
        keep = d["mins"] >= start_min
        pv = np.where(keep, pv, 0.0)
        v = np.where(keep, v, 0.0)
    s = pd.DataFrame({"day": d["day"], "pv": pv, "v": v})
    g = s.groupby("day", sort=False)
    cpv = g["pv"].cumsum().to_numpy()
    cv = g["v"].cumsum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = cpv / cv
    if start_min is not None:
        out = np.where(d["mins"] >= start_min, out, np.nan)
    return out


def swings(l):
    n = len(l)
    sw = np.zeros(n, dtype=bool)
    if n >= 5:
        j = np.arange(2, n - 2)
        sw[j] = (l[j - 1] > l[j]) & (l[j - 2] > l[j]) & (l[j + 1] >= l[j]) & (l[j + 2] >= l[j])
    return sw


def level_count(l, sw, t, S):
    """Level count for bar t (swings j in [t-N, t-3], i.e. j+P <= t-1 and j <= t-2)."""
    lo = max(t - N, 0)
    hi = t - P - 1  # inclusive
    js = np.nonzero(sw[lo:hi + 1])[0] + lo
    pts = l[js]
    pts = pts[pts <= S * (1 + LEV_BAND)]
    if not np.any(pts == S):
        pts = np.append(pts, S)
    pts = np.sort(pts)
    lev = 1
    first = pts[0]
    for p in pts[1:]:
        if p > first * (1 + LEV_GAP):
            lev += 1
            first = p
    return lev, len(pts)


def base_arrays(d):
    """Vectorised pieces of pattern (1) for every bar."""
    n = len(d["ts"])
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    ts = d["ts"]
    S = pd.Series(l).rolling(N, min_periods=N).min().shift(1).to_numpy()  # min(l[t-N..t-1])
    e10, e25, e60 = ema(c, 10), ema(c, 25), ema(c, 60)
    al = ((e10 < e25) & (e25 < e60)).astype(np.float64)
    alsum = pd.Series(al).rolling(TR_W, min_periods=TR_W).sum().to_numpy()
    tr = np.zeros(n, dtype=bool)
    b = np.arange(n) - 1
    okb = b >= TR_W  # need b-30 >= 0
    tr[okb] = (alsum[b[okb]] >= TR_MIN) & (e60[b[okb]] < e60[b[okb] - 30])
    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        wick = np.where(rng > 0, (np.minimum(o, c) - l) / rng, np.nan)
    cnt30 = np.arange(n) - np.searchsorted(ts, ts - 30, side="right") + 1
    brk = l < S * (1 - TOL)
    openok = o >= S * (1 - TOL)
    san = (d["n"] >= TRADES_MIN) & (d["v"] * c >= DOLLAR_MIN) & (rng >= RANGE_MIN * c) & (cnt30 >= CNT30_MIN)
    nxt = np.zeros(n, dtype=bool)
    nxt[:-1] = ((ts[1:] - ts[:-1]) <= 3) & (d["day"][1:] == d["day"][:-1])
    z = np.searchsorted(ts, ts + 30, side="right") - 1
    zday_ok = d["day"][np.clip(z, 0, n - 1)] == d["day"]
    e2ok = (z >= np.arange(n) + 2) & zday_ok
    return dict(S=S, e10=e10, e25=e25, e60=e60, alsum=alsum, tr=tr, wick=wick, cnt30=cnt30,
                brk=brk, openok=openok, san=san, nxt=nxt, z=z, e2ok=e2ok)


def drop120(d, t):
    ts = d["ts"]
    i0 = np.searchsorted(ts, ts[t] - 120, side="right")
    mh = d["h"][i0:t + 1].max()
    return 1.0 - d["l"][t] / mh, mh


def e2_outcome(d, t, z, tgt_mult=1.01):
    entry = d["c"][t + 1]
    tgt = entry * tgt_mult
    hh = d["h"][t + 2:z + 1]
    touch = np.nonzero(hh >= tgt)[0]
    if len(touch):
        return entry, True, 0.01, t + 2 + touch[0]
    return entry, False, d["c"][z] / entry - 1.0, -1


def tod_in(m):
    if m < 570:
        return "pre"
    if m < 585:
        return "open15"
    if m < 660:
        return "am"
    return "post"


def tod_out(m):
    if m < 570:
        return "pre"
    if m < 960:
        return "am"
    return "post"


def tier(cap_b):
    if not np.isfinite(cap_b):
        return None
    if cap_b < 5:
        return "<5B"
    if cap_b < 20:
        return "5-20B"
    if cap_b < 100:
        return "20-100B"
    return "100B+"


def cost(cm, tr, tod):
    key = f"{tr}|{tod}"
    if tr is not None and key in cm["table"]:
        return cm["table"][key]["half"]
    return cm["by_tod"][tod]


def in_window(m):
    return ((m >= 240) & (m < 660)) | ((m >= 1140) & (m < 1170))
