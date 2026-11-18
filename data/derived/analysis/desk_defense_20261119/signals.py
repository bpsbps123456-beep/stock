"""2026 PIT replay of US ETF-holding trend grades and Giant individual recovery grades."""
from __future__ import annotations

import json
import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

import watchlist
from desk import giant, strategies, trend
from gbrc.etf.market import is_cash_holding, listing_market, yahoo_ticker
from gbrc.etf.storage import NORMALIZED_HOLDINGS_DIR

START = "2026-01-02"
END = "2026-11-17"
GRADE = {"S": 4, "A": 3, "B": 2, "C": 1, "D": 0}
SAVED = ("trend_grades.csv", "giant_grades.csv")


def holdings_history():
    """Separate as-of per fund; a Korean holiday never erases the previous basket."""
    by_fund = {}
    all_symbols = set()
    for fund, _ in trend.FUNDS:
        history = {}
        for path in sorted(NORMALIZED_HOLDINGS_DIR.glob(f"*_{fund}.json")):
            day = path.name[:10]
            if day > END:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = {}
            for row in data["holdings"]:
                code = row.get("code")
                if listing_market(code) != "US" or is_cash_holding(row):
                    continue
                key = strategies.research_key(code)
                symbol = yahoo_ticker(code, day)
                # Yahoo preserves the original trading history under these renamed symbols.
                if code in ("SATS US EQUITY", "SKHYV US EQUITY"):
                    symbol = yahoo_ticker(key)
                if symbol:
                    rows.setdefault(key, symbol)
            history[day] = rows
            all_symbols.update(rows.values())
        by_fund[fund] = (sorted(history), history)
    return by_fund, all_symbols


def bars_for(symbols):
    cache = HERE / "cache" / "grade_bars"
    cache.mkdir(exist_ok=True, parents=True)
    missing = sorted(s for s in symbols if not (cache / f"{s}.parquet").exists())
    for start in range(0, len(missing), 24):
        batch = missing[start:start + 24]
        frame = yf.download(batch, start="2010-01-01", end="2026-11-18", auto_adjust=False,
                            threads=6, progress=False, group_by="ticker", multi_level_index=True)
        for sym in batch:
            if sym not in frame.columns.get_level_values(0):
                continue
            bars = frame[sym][["High", "Close"]].dropna().sort_index()
            bars = bars[(bars.High > 0) & (bars.Close > 0)]
            bars.index = pd.DatetimeIndex(bars.index).tz_localize(None).normalize()
            if not bars.empty:
                bars.to_parquet(cache / f"{sym}.parquet")
    return {s: pd.read_parquet(cache / f"{s}.parquet") for s in sorted(symbols)
            if (cache / f"{s}.parquet").exists()}


def tickers_as_of(histories, day):
    rows = {}
    for days, snapshots in histories.values():
        i = bisect_right(days, day) - 1
        if i >= 0:
            for key, symbol in snapshots[days[i]].items():
                rows.setdefault(key, symbol)
    return rows


def grade_trend(book, bars, histories, days):
    # The tab takes the current basket, compares its members for ten closes and smooths.
    # At each replay date the whole lookback must therefore use that date's public roster.
    cal = bars["QQQ"].index
    cal = cal[(cal >= "2025-12-01") & (cal <= END)]
    ind = {s: trend.indicators(b[b.index >= "2024-01-01"]) for s, b in bars.items()}
    panels = {k: trend._wide(ind, k, cal) for k in [*trend.PARTS, "bars", "up", "down"]}
    out = pd.DataFrame(np.nan, index=days, columns=sorted(set(book.ticker.values())))
    ref_counts = []
    missing_observations = {}
    for day in days:
        t = pd.Timestamp(day)
        rows = tickers_as_of(histories, day)
        used = {}
        proxies = set()
        for key, sym in rows.items():
            proxy = watchlist.AI_UNIVERSE_ALIASES.get(sym)
            if proxy and "." in proxy and proxy in bars and (
                    sym not in bars or len(bars[sym].loc[:t]) < trend.MIN_BARS):
                used[key] = proxy
                proxies.add(proxy)
            elif sym in bars:
                used[key] = sym
            else:
                missing_observations[sym] = missing_observations.get(sym, 0) + 1
        if not used or len(rows) - len(used) > trend.MAX_MISSING * len(rows):
            raise ValueError(f"Insufficient signal-day price coverage on {day}")
        selected = list(dict.fromkeys(used.values()))
        ref = set(selected) - proxies
        ref_counts.append(len(ref))
        pos = cal.get_loc(t)
        recent = cal[max(0, pos - trend.SMOOTH + 1):pos + 1]
        mask = panels["bars"].loc[recent, selected] >= trend.MIN_BARS
        component = []
        for part in trend.PARTS:
            values = panels[part].loc[recent, selected].where(mask)
            component.append(trend.rank_vs(values, ref))
        weighted = sum(p.fillna(0) * w for p, w in zip(component, trend.PARTS.values()))
        total = sum(p.notna() * w for p, w in zip(component, trend.PARTS.values()))
        scores = weighted / total.where(total > 0)
        smooth = scores.mean().where(scores.iloc[-1].notna())
        pct = trend.rank_vs(smooth.to_frame().T, ref).iloc[0]
        for key, sym in used.items():
            traded = book.ticker.get(key)
            if traded is not None:
                grade = trend.gate(trend.grade_of(pct[sym]), panels["up"].at[t, sym], panels["down"].at[t, sym])
                out.at[day, traded] = GRADE.get(grade, np.nan)
    return out, {"min_us_reference": min(ref_counts), "max_us_reference": max(ref_counts),
                 "missing_price_by_held_days": missing_observations}


def grade_giant(symbols, bars, days):
    universe = set(watchlist.AI_SUBSECTOR)
    names = {s: watchlist.AI_UNIVERSE_ALIASES.get(s, s) for s in symbols}
    names = {s: sym for s, sym in names.items() if sym in universe and sym in bars}
    out = pd.DataFrame(np.nan, index=days, columns=symbols)
    for day in days:
        t = pd.Timestamp(day)
        anchor = giant.find_anchor(bars[giant.ANCHOR].loc[:t, "Close"])
        for s, sym in names.items():
            history = bars[sym].loc[:t]
            if len(history) and (t - history.index[-1]).days <= 7:
                grade = giant.grade_of(giant.own_recovery(history, anchor))
                out.at[day, s] = GRADE.get(grade, np.nan)
    return out


def build_signals(book, rebuild=False):
    """Frozen replay inputs; both outputs indexed by close date strings and traded symbols."""
    days = [d for d in book.days if START <= d <= END]
    symbols = sorted(set(book.ticker.values()))
    if not rebuild and all((HERE / name).exists() for name in SAVED):
        grades = [pd.read_csv(HERE / name, index_col=0) for name in SAVED]
        if all(g.index.tolist() == days and g.columns.tolist() == symbols for g in grades):
            return tuple(grades)
    histories, held = holdings_history()
    prices = bars_for(held | set(symbols) | set(watchlist.AI_UNIVERSE_ALIASES.values()) | {"QQQ", giant.ANCHOR})
    trend_df, stats = grade_trend(book, prices, histories, days)
    giant_df = grade_giant(symbols, prices, days)
    for frame, name in zip((trend_df, giant_df), SAVED):
        frame.to_csv(HERE / name, index_label="close_date", float_format="%.0f")
    stats.update({"first_close": days[0], "last_close": days[-1], "sessions": len(days),
                  "traded_symbols": len(symbols), "trend_nonempty": int(trend_df.count().sum()),
                  "giant_nonempty": int(giant_df.count().sum()), "download_success": len(prices),
                  "giant_scope": "individual recovery grade, current AI_UNIVERSE map, no grade if no qualifying crash"})
    (HERE / "signal_coverage.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return trend_df, giant_df
