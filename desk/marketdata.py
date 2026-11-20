"""Daily prices (cached) and live quotes from Yahoo via yfinance.

Daily history: split-adjusted Open/Close plus dividend-adjusted Adj Close. Live quotes
include pre- and post-market minutes so the evening KST view shows a current price.
"""

from __future__ import annotations

import json
import os
import time as _time
from datetime import date, datetime, time, timedelta

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


def _sip_basket_bars(symbols, first, last, cutoff, *, full=False):
    """One batch of split-adjusted (dividends excluded) daily bars, market data only."""
    import requests
    env = dict(os.environ)
    dotenv = config.ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip("\"'"))
    key = env.get("ALPACA_KEY_ID") or env.get("APCA_API_KEY_ID")
    secret = env.get("ALPACA_SECRET_KEY") or env.get("APCA_API_SECRET_KEY")
    if not (key and secret):
        return {}
    vendor = {s.replace("-", "."):s for s in symbols}
    # A free data key can read SIP past the 15-minute delay. Limit the end to
    # the observed clock even when the final session closed earlier today.
    end = min(datetime.combine(date.fromisoformat(last) + timedelta(days=1), time.min, config.NY), cutoff)
    response = requests.get("https://data.alpaca.markets/v2/stocks/bars",
                            headers={"APCA-API-KEY-ID":key, "APCA-API-SECRET-KEY":secret},
                            params={"symbols":",".join(vendor), "timeframe":"1Day", "start":first,
                                    "end":end.isoformat(), "adjustment":"split", "asof":"-",
                                    "feed":"sip", "limit":10000}, timeout=12)
    response.raise_for_status()
    data = response.json()
    if data.get("next_page_token"):
        return {}  # Callers batch requests below the page size. Partial results are unusable.
    result = {}
    for name, bars in data["bars"].items():
        if name not in vendor:
            continue
        out = {}
        for r in bars:
            day = datetime.fromisoformat(r["t"].replace("Z","+00:00")).astimezone(config.NY).date().isoformat()
            if (first <= day <= last if full else day in (first, last)) and r["v"] > 0:
                out[day] = float(r["o"]), float(r["c"])
        result[vendor[name]] = out
    return result


def repair_basket_bars(book, basket: dict | None, completed: str, *, now=None) -> dict:
    """Fill missing month-open / latest-close values in this book. Return provenance of filled cells.

    Both endpoints must be present in the same SIP split-adjusted series. Any available
    Yahoo endpoint must agree with that basis. A data outage leaves blanks for a retry.
    """
    import requests
    from .news import market_calendar
    if not basket or not completed:
        return {}
    first = basket["effective_date"]
    if (first[:7] != completed[:7] or first > completed or first not in book.opn.index
            or completed not in book.cls.index):
        return {}
    def value(frame, day, symbol):
        return frame.at[day,symbol] if symbol in frame and day in frame.index else np.nan

    def valid(v):
        return np.isfinite(v) and v > 0

    wanted = [s for s in basket["symbols"] if not valid(value(book.opn,first,s))
              or not valid(value(book.cls,completed,s))]
    if not wanted:
        return {}
    calendar = market_calendar()
    if not calendar.is_session(completed):
        return {}
    cutoff = (now or datetime.now(config.NY)) - timedelta(minutes=16)
    if calendar.session_close(completed).to_pydatetime() > cutoff:
        return {}
    try:
        bars = _sip_basket_bars(wanted,first,completed,cutoff)
    except (requests.RequestException, OSError, KeyError, ValueError, TypeError, AttributeError):
        return {}
    report = {}
    for s, data in bars.items():
        if s not in wanted or first not in data or completed not in data:
            continue
        # Check all available corresponding Yahoo quotes at those endpoints.
        # No conversion is inferred from a large disagreement (e.g. a split).
        matches = True
        for day in {first,completed}:
            for frame, quote in zip((book.opn,book.cls), data[day]):
                current = value(frame,day,s)
                if not valid(quote) or (valid(current) and not np.isclose(current,quote,rtol=.002,atol=.01)):
                    matches = False
        if not matches:
            continue
        notes = {"source":"Alpaca SIP, split-adjusted daily bars"}
        for frame,day,quote,tag in [(book.opn,first,data[first][0],"open_date"),
                                    (book.cls,completed,data[completed][1],"close_date")]:
            if not valid(value(frame,day,s)):
                frame.at[day,s] = quote
                notes[tag] = day
        report[s] = notes
    return report
