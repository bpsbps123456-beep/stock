import csv
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

import pandas as pd
import numpy as np

from desk import finviz_basket as basket, marketdata


def now(stamp):
    return datetime.fromisoformat(stamp).replace(tzinfo=basket.config.NY)


def screen(ranked, stamp="2026-11-30T16:45:00", *, total=None, query=None):
    data = {"today": stamp, "origin": "https://finviz.com", "screenerType": "stocks",
            "url": "/screener/stocks?" + (query or urlsplit(basket.SCREEN_URL).query),
            "data": {"firstRow": 0, "pageSize": 20,
                     "totalCount": len(ranked) if total is None else total,
                     "rows": [{"text": {"ticker": s, "country": "USA"}, "numbers": {"order": n, "price": 42}}
                              for n,s in enumerate(ranked)]}}
    return '<script type="application/json" data-ssr-props="screener">'+json.dumps(data)+'</script>'


class TestFinvizBasket(unittest.TestCase):
    def test_months_calendar_holidays_and_no_out_of_window_fetch(self):
        self.assertEqual(basket.month_dates("2027-01-20"), ("2026-12-31", "2027-01-04"))
        with TemporaryDirectory() as temp, patch.object(basket, "_fetch") as fetch:
            for timepoint in ("2026-11-30T16:10", "2026-11-27T16:50", "2026-12-01T04:00",
                              "2026-12-01T09:29", "2026-11-20T02:00", "2026-11-01T00:01"):
                self.assertIsNone(basket.capture_due(now(timepoint), Path(temp)))
            fetch.assert_not_called()
        start, _ = basket._window("2026-11-27")  # early close after Thanksgiving
        self.assertEqual(start.astimezone(basket.config.NY).strftime("%H:%M"), "13:25")

    def test_freeze_conflicting_retries_and_expire(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)
            with patch.object(basket, "_fetch", return_value=(["ABC"], ["ABC"])) as fetch:
                first = basket.capture_due(now("2026-12-01T02:00"), path)
                self.assertEqual(first["signal_date"], "2026-11-30")
                self.assertEqual(first["cash_slots"], 6)
                self.assertEqual(basket.capture_due(now("2026-12-01T03:00"), path), first)
                self.assertEqual(fetch.call_count, 1)
            second = dict(first, ranked_symbols=["XYZ"], symbols=["XYZ"])
            self.assertEqual(basket._freeze(second,path), first)
            self.assertEqual(basket.load_basket("2026-12-15", path), first)
            self.assertIsNone(basket.load_basket("2027-01-07", path))
            self.assertIsNone(basket.load_basket("2026-11-30", path))
            (path/"2026-12.json").write_text("{}",encoding="utf-8")
            with self.assertRaises(basket.BasketError):
                basket.load_basket("2026-12-15", path)

    def test_screen_parser_requires_the_exact_query_fresh_page_and_complete_rows(self):
        clock, signal = now("2026-11-30T16:45"), "2026-11-30"
        self.assertEqual(basket._parse_screen(screen([]), clock, signal), [])
        tickers = ["A","B","C","D","E","F","G","H"]
        self.assertEqual(basket._parse_screen(screen(tickers),clock,signal), tickers[:7])
        for bad in ("<html>Sign in</html>",screen(["A"],total=9),screen(["A"],query="v=111"),
                    screen(["A"], stamp="2026-11-30T15:46:00"),screen(["A","A"])):
            with self.assertRaises(basket.BasketError):
                basket._parse_screen(bad,clock,signal)

    def test_separate_shares_of_one_issuer_keep_the_second_slot_in_cash(self):
        first = Mock(text=screen(["GOOG","GOOGL","A"]))
        sec = Mock()
        sec.json.return_value = {str(i):{"ticker": s, "cik_str": i//2+1}
                                 for i,s in enumerate(["GOOG","GOOGL","A"])}
        with patch.object(basket.requests, "get", side_effect=[first,sec]):
            raw, actual = basket._fetch(now("2026-11-30T16:45"), "2026-11-30")
        self.assertEqual((raw, actual), (["GOOG","GOOGL","A"], ["GOOG","A"]))
        with TemporaryDirectory() as temp, patch.object(basket, "_fetch", side_effect=basket.BasketError("실패")):
            with self.assertRaises(basket.BasketError):
                basket.capture_due(now("2026-11-30T16:45"),Path(temp))
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_november_seed_reconciles_both_frozen_sources(self):
        seed = basket.load_basket("2026-11-20")
        self.assertEqual(seed["source"], "research_reconstruction")
        self.assertGreater(datetime.fromisoformat(seed["captured_at"]).date().isoformat(), seed["effective_date"])
        meta = json.loads((basket.config.ROOT / seed["source_ref"]).with_name("screens.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["balance"]["url"], basket.SCREEN_URL)
        self.assertEqual((seed["signal_date"],seed["effective_date"]),("2026-10-30","2026-11-02"))
        with (basket.config.ROOT / seed["source_ref"]).open(encoding="utf-8") as source:
            ranked = [r for r in csv.DictReader(source) if r["signal_close"] == seed["signal_date"]
                      and r["trade_open"] == seed["effective_date"] and r["tight"] == "True"
                      and float(r["rv"]) > 1 and float(r["rsi"]) > 40]
        ranked.sort(key=lambda r:(-float(r["rv"]),r["symbol"]))
        self.assertEqual(seed["symbols"], [r["symbol"] for r in ranked[:7]])
        with (basket.config.ROOT / seed["ledger_ref"]).open(encoding="utf-8") as ledger:
            first = next(r for r in csv.DictReader(ledger) if r["date"] == seed["effective_date"]
                         and r["key"] == "balance/monthly1")
        self.assertEqual(set(seed["symbols"]), set(first["holdings"].split("|")))


class TestBasketBarRepair(unittest.TestCase):
    first, last = "2026-11-02", "2026-11-19"
    basket = {"effective_date":first, "symbols":["CART","WST"]}

    def book(self):
        opn = pd.DataFrame({"CART":[56.44,57.00], "WST":[400.,402.], "QQQ":[600.,590.]},
                           index=[self.first, self.last])
        cls = pd.DataFrame({"CART":[56.30,np.nan], "WST":[405.,404.], "QQQ":[601.,589.]},
                           index=[self.first, self.last])
        return SimpleNamespace(opn=opn,cls=cls)

    def test_scoped_backfill_then_idempotent_no_request(self):
        book = self.book()
        sip = {"CART":{self.first:(56.44,56.30), self.last:(57.00,57.84)}}
        end = now("2026-11-20T02:00")
        with patch.object(marketdata, "_sip_basket_bars", return_value=sip) as fetch:
            result = marketdata.repair_basket_bars(book,self.basket,self.last,now=end)
            self.assertEqual(list(result), ["CART"])
            self.assertEqual(result["CART"]["close_date"],self.last)
            self.assertNotIn("open_date",result["CART"])
            self.assertEqual(book.cls.at[self.last,"CART"],57.84)
            self.assertEqual(book.cls.at[self.last,"QQQ"],589.)
            self.assertEqual(marketdata.repair_basket_bars(book,self.basket,self.last,now=end), {})
            self.assertEqual(fetch.call_count,1)
            self.assertEqual(fetch.call_args.args[0],["CART"])

    def test_split_conflict_missing_other_endpoint_and_unclosed_day_do_not_fill(self):
        for rows in ({self.first:(28.22,28.15),self.last:(57.,57.84)}, {self.last:(57.,57.84)}):
            book=self.book()
            with patch.object(marketdata, "_sip_basket_bars",return_value={"CART":rows}):
                self.assertEqual(marketdata.repair_basket_bars(book,self.basket,self.last,now=now("2026-11-20T02:00")), {})
                self.assertTrue(np.isnan(book.cls.at[self.last,"CART"]))
        with patch.object(marketdata,"_sip_basket_bars") as fetch:
            self.assertEqual(marketdata.repair_basket_bars(self.book(),self.basket,self.last,now=now("2026-11-19T15:59")), {})
            fetch.assert_not_called()

    def test_entirely_absent_symbol_uses_two_consistent_split_only_bars(self):
        book=self.book()
        for frame in (book.opn,book.cls):
            del frame["CART"]
        with patch.object(marketdata,"_sip_basket_bars",
                          return_value={"CART":{self.first:(56.44,56.30),self.last:(57.,57.84)}}):
            report=marketdata.repair_basket_bars(book,self.basket,self.last,now=now("2026-11-20T02:00"))
        self.assertEqual((report["CART"]["open_date"],report["CART"]["close_date"]),(self.first,self.last))
        self.assertEqual(book.opn.at[self.first,"CART"],56.44)
        self.assertTrue(np.isnan(book.cls.at[self.first,"CART"]))
