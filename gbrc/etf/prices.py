"""Unadjusted close history, close-to-close returns, and the KRX session calendar.

Call `prices.load_close_history` through the module (not a from-import) so tests can
replace the network download in one place.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from gbrc.config import performance_funds

from . import market
from .storage import NORMALIZED_HOLDINGS_DIR, snapshot_dates


def load_close_history(tickers, start_date: str, end_date: str) -> dict[str, Any]:
    """{ticker: pandas Series of unadjusted closes} — the NAV's close-to-close basis.

    A failed download yields {} (cells then read N/A), never a guessed price.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {}

    tickers = sorted({t for t in tickers if t})
    if not tickers:
        return {}

    try:
        data = yf.download(tickers, start=start_date, end=end_date,
                           auto_adjust=False, progress=False)
    except Exception:
        return {}

    if data is None or data.empty or "Close" not in data:
        return {}

    close = data["Close"]
    history = {}
    if hasattr(close, "columns"):
        for ticker in tickers:
            if ticker not in close.columns:
                continue
            series = close[ticker].dropna()
            if not series.empty:
                history[ticker] = series
    elif len(tickers) == 1:
        series = close.dropna()
        if not series.empty:
            history[tickers[0]] = series
    return history


def close_point_on_or_before(series, date_str: str | None):
    """(date, close) of the last close at or before date_str, else None."""
    if series is None or series.empty or not date_str:
        return None
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None
    eligible = series[series.index.date <= target]
    if eligible.empty:
        return None
    try:
        return eligible.index[-1].date(), float(eligible.iloc[-1])
    except (TypeError, ValueError):
        return None


def close_return_pct(close_history, ticker, base_date, target_date) -> float | None:
    if not ticker or not base_date or not target_date:
        return None
    series = close_history.get(ticker)
    base_point = close_point_on_or_before(series, base_date)
    target_point = close_point_on_or_before(series, target_date)
    if not base_point or not target_point:
        return None
    base_actual_date, base = base_point
    target_actual_date, target = target_point
    if target_actual_date <= base_actual_date:
        return None
    if base is None or target is None or base <= 0:
        return None
    return (target / base - 1.0) * 100.0


_SESSION_CACHE: dict[tuple[str, str], list[str]] = {}


def krx_sessions(through_date: str, root: Path = NORMALIZED_HOLDINGS_DIR) -> list[str]:
    """KRX sessions up to through_date. D-n and price dates count sessions, not weekdays.

    The calendar is the union of both funds' validated snapshot dates (all KRX
    sessions) and both ETFs' market-price trading days, so a failed price download
    still leaves the snapshot dates.
    """
    cache_key = (str(root), through_date)
    if cache_key in _SESSION_CACHE:
        return _SESSION_CACHE[cache_key]
    funds = performance_funds()
    dates = [d for fund in funds for d in snapshot_dates(fund["snapshot_ticker"], root)]
    through = datetime.strptime(through_date, "%Y-%m-%d").date()
    history = load_close_history(
        [fund["price_ticker"] for fund in funds],
        (through - timedelta(days=120)).strftime("%Y-%m-%d"),
        (through + timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    price_dates = [idx.strftime("%Y-%m-%d") for series in history.values() for idx in series.index]
    sessions = [d for d in market.merge_sessions(dates, price_dates) if d <= through_date]
    _SESSION_CACHE[cache_key] = sessions
    return sessions
