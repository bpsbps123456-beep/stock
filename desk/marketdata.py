"""Daily prices (cached) and live quotes from Yahoo via yfinance.

Daily history: split-adjusted Open/Close plus dividend-adjusted Adj Close. Live quotes
include pre- and post-market minutes so the evening KST view shows a current price.
"""

from __future__ import annotations

import json
import time as _time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import config

FIELDS = ("Adj Close", "Close", "Open")


def _path(field: str):
    return config.CACHE / f"prices_{field.replace(' ', '_').lower()}.csv"


def history(tickers: list[str], force: bool = False) -> dict[str, pd.DataFrame]:
    """Daily bars from PRICE_START through today, reusing a fresh enough cache."""
    config.CACHE.mkdir(parents=True, exist_ok=True)
    meta_path = config.CACHE / "prices_meta.json"
    wanted = sorted(set(tickers))
    if not force and meta_path.exists() and all(_path(f).exists() for f in FIELDS):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fresh = _time.time() - meta.get("fetched_epoch", 0) < config.PRICE_MAX_AGE_MIN * 60
        if fresh and set(wanted) <= set(meta.get("tickers", [])):
            return {f: pd.read_csv(_path(f), index_col=0, parse_dates=True) for f in FIELDS}
    import yfinance as yf

    end = (date.today() + timedelta(days=1)).isoformat()
    frame = yf.download(wanted, start=config.PRICE_START, end=end, auto_adjust=False,
                        progress=False, threads=True)
    out = {f: frame[f].dropna(axis=1, how="all") for f in FIELDS}
    for f, df in out.items():
        df.to_csv(_path(f))
    meta_path.write_text(json.dumps({"fetched_epoch": _time.time(), "tickers": wanted,
                                     "fetched_at": datetime.now(config.KST).isoformat(timespec="seconds")}),
                         encoding="utf-8")
    return out


def _session(ts: pd.Timestamp) -> str:
    t = ts.tz_convert(config.NY).time() if ts.tzinfo else ts.time()
    if t < datetime.strptime("09:30", "%H:%M").time():
        return "장전"
    if t < datetime.strptime("16:00", "%H:%M").time():
        return "장중"
    return "장후"


def live_quotes(tickers: list[str]) -> dict[str, dict]:
    """Last 1-minute print per ticker (incl. extended hours), with its NY-time session."""
    tickers = sorted({t for t in tickers if t})
    if not tickers:
        return {}
    import yfinance as yf

    try:
        frame = yf.download(tickers, period="5d", interval="1m", prepost=True, auto_adjust=False,
                            progress=False, threads=True)
    except Exception as exc:  # network hiccups must not break the page
        return {"_error": {"message": f"{type(exc).__name__}: {exc}"}}
    closes = frame["Close"] if "Close" in frame else pd.DataFrame()
    out = {}
    for t in tickers:
        if t not in closes.columns:
            continue
        s = closes[t].dropna()
        if s.empty:
            continue
        ts = s.index[-1]
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        out[t] = {"price": float(s.iloc[-1]), "session": _session(ts),
                  "time_kst": ts.tz_convert(config.KST).strftime("%m-%d %H:%M"),
                  "time_ny": ts.tz_convert(config.NY).strftime("%m-%d %H:%M"), "epoch": ts.timestamp()}
    return out


def ffill_on(series: pd.Series, days: list[str]) -> np.ndarray:
    s = series.copy()
    s.index = s.index.strftime("%Y-%m-%d")
    return s.reindex(sorted(set(s.index) | set(days))).ffill().reindex(days).to_numpy(float)
