import csv
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from desk import config, finviz_basket as basket, finviz_trend as trend, marketdata


SIGNAL = "2026-11-30"
CAPTURE = datetime(2026, 11, 30, 16, 45, tzinfo=config.NY)


def screen(url, symbols, first, total):
    rows = [{"text": {"ticker": symbol, "country": "USA"},
             "numbers": {"order": rank, "price": 42, "curRatio": 2., "debtEq": debt}}
            for rank, (symbol, debt) in enumerate(symbols, start=first)]
    page = {"url": "/screener/stocks?" + urlsplit(url).query, "origin": "https://finviz.com",
            "screenerType": "stocks", "today": CAPTURE.isoformat(),
            "data": {"rows": rows, "firstRow": first, "pageSize": 20, "totalCount": total}}
    return '<script type="application/json" data-ssr-props="screener">' + json.dumps(page) + '</script>'


class TestTrendPaper(unittest.TestCase):
    def test_pagination_exact_thresholds_and_independent_month_freeze(self):
        first_page = [(f"A{n}", 1. if n % 2 else -0.1) for n in range(19)] + [("LAST", 0.)]
        second_page = [(f"B{n}", .99) for n in range(8)]

        def bars(symbols, first, last, cutoff, *, full):
            days = basket.market_calendar().sessions_in_range(first, last).strftime("%Y-%m-%d")
            self.assertTrue(full)
            self.assertEqual(len(days), 175)
            self.assertLess(cutoff, CAPTURE)
            return {symbol: {day: (100. + n, 100. + n) for n, day in enumerate(days)}
                    for symbol in symbols}

        with TemporaryDirectory() as tmp, patch.object(basket, "_get_screen", side_effect=[
            screen(trend.SCREEN_URL, first_page, 0, 28),
            screen(trend.SCREEN_URL + "&r=21", second_page, 20, 28),
        ]) as get, patch.object(basket, "_distinct_issuers", side_effect=lambda names: names[:-1]), \
                patch.object(marketdata, "_sip_basket_bars", side_effect=bars):
            path = Path(tmp)
            record = trend.capture_due(CAPTURE, path)
            self.assertEqual(record["ranked_symbols"], ["LAST", *[f"B{n}" for n in range(6)]])
            self.assertEqual(record["cash_slots"], 1)
            self.assertEqual(record["source"], trend.SOURCE)
            self.assertEqual(trend.capture_due(CAPTURE, path), record)
            self.assertEqual(get.call_count, 2)
            with self.assertRaises(basket.BasketError):
                basket.load_basket("2026-12-01", path)

    def test_does_not_freeze_partial_or_stale_sip_history(self):
        flat = {str(day): (100., 100.) for day in range(175)}
        days = list(flat)
        self.assertFalse(trend._trend_passes(flat, days))
        self.assertFalse(trend._trend_passes(dict(list(flat.items())[1:]), days))
        for missing in ("100", "174"):
            with self.assertRaises(basket.BasketError):
                trend._trend_passes({**{k: v for k, v in flat.items() if k != missing}}, days)
        with TemporaryDirectory() as tmp, patch.object(basket, "_get_screen", return_value=screen(
            trend.SCREEN_URL, [("LAST", .1)], 0, 1,
        )), patch.object(marketdata, "_sip_basket_bars", return_value={}):
            with self.assertRaises(basket.BasketError):
                trend.capture_due(CAPTURE, Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_reconstruction_matches_frozen_research_inputs(self):
        seed = basket.load_basket("2026-11-20", trend.DIRECTORY,
                                  screen_url=trend.SCREEN_URL, live_source=trend.SOURCE)
        with (config.ROOT / seed["source_ref"]).open(encoding="utf-8") as source:
            same_day = [row for row in csv.DictReader(source)
                        if row["trade_open"] == seed["effective_date"] and row["signal_close"] == seed["signal_date"]]
        columns = ("current_ratio", "debt_equity", "rv", "rsi", "close", "sma30", "sma40", "sma175")
        passing = []
        for row in same_day:
            current, debt, rv, rsi, close, short, mid, long = (float(row[key] or "nan") for key in columns)
            if current > 1.5 and 0 <= debt < 1 and rv > 1 and rsi > 40 and close > mid and close > long and short > mid:
                passing.append((-rv, row["symbol"]))
        self.assertEqual(seed["ranked_symbols"], [symbol for _, symbol in sorted(passing)[:7]])

    def test_candidate_repairs_only_its_own_book_and_records_first_preopen(self):
        completed, next_open = "2026-11-19", "2026-11-20"
        cal = basket.market_calendar()
        days = cal.sessions_in_range(cal.session_offset(completed, -100), completed).strftime("%Y-%m-%d").tolist()
        close = pd.DataFrame({"SOXX": 100.}, index=days)
        close.at[completed, "SOXX"] = 80
        opn = close.copy()
        book = SimpleNamespace(days=days, cls=close, opn=opn, ticker={})
        weekly = {"target": {}, "ranked": [], "rebalance_day": False}
        frozen = {"symbols": ["CAND"], "source": trend.SOURCE, "effective_date": "2026-11-02",
                  "signal_date": "2026-10-30"}
        sip = {"CAND": {"2026-11-02": (10., 10.), completed: (11., 11.)}}
        pre = datetime(2026, 11, 20, 8, tzinfo=config.NY)
        with TemporaryDirectory() as tmp, patch.object(config, "CACHE", Path(tmp)), \
                patch.object(basket, "load_basket", return_value=frozen), \
                patch.object(marketdata, "_sip_basket_bars", return_value=sip):
            paper = trend.Paper()
            route = paper.observe(book, weekly, next_open, completed, pre, {})
            self.assertEqual(route["mode"], "finviz")
            self.assertIn("CAND", route["bar_sources"])
            self.assertNotIn("CAND", close)
            self.assertNotIn("CAND", opn)
            record = trend.observation_dir() / f"{next_open}.json"
            first = json.loads(record.read_text(encoding="utf-8"))
            self.assertEqual(first["variant"], trend.VARIANT)
            self.assertNotIn("applied", first)
            self.assertFalse((record.parent.parent / record.name).exists())
            close.at[completed, "SOXX"] = np.nan
            paper.observe(book, weekly, next_open, completed, pre.replace(hour=9), {})
            self.assertEqual(json.loads(record.read_text(encoding="utf-8")), first)
