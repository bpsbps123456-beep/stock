"""Desk robustness fixes of 2026-11-21 (no network): shared file helpers, the XNYS calendar, test-server
record folders, journal gaps and input checks, price-cache meta, news price expiry, §6 validate
without the #774 input, digest timing, the briefing lock, the season residual line, the bottom
outlook caveat, and regenerable caches that are missing or broken."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from desk import common, config, journal

ROOT = Path(__file__).resolve().parents[1]

# The hand-written NYSE closures config.py used until 2026-11-21 (2026–2027). XNYS must agree on them.
OLD_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


class CommonFilesTest(unittest.TestCase):
    def test_write_atomic_keeps_the_old_file_when_the_new_data_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "a.json"
            common.write_atomic(path, {"x": 1})
            with self.assertRaises(ValueError):
                common.write_atomic(path, {"x": math.nan})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"x": 1})
            self.assertEqual([p.name for p in path.parent.iterdir()], ["a.json"])  # no temp left behind

    def test_write_once_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock.json"
            self.assertTrue(common.write_once(path, {"first": True}))
            self.assertFalse(common.write_once(path, {"first": False}))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"first": True})

    def test_read_cache_treats_broken_files_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            self.assertIsNone(common.read_cache(path))
            path.write_text("{\"a\": 1", encoding="utf-8")  # e.g. a crash mid-write or a merge marker
            self.assertIsNone(common.read_cache(path))
            path.write_text("[1, 2]", encoding="utf-8")
            self.assertIsNone(common.read_cache(path))
            path.write_text("{\"a\": 1}", encoding="utf-8")
            self.assertEqual(common.read_cache(path), {"a": 1})

    def test_state_and_news_reexport_the_single_helpers(self):
        from desk import news, state
        self.assertIs(state.clean, common.clean)
        self.assertIs(news.write_atomic, common.write_atomic)
        self.assertIs(news.now_iso, common.now_iso)
        self.assertIs(news.market_calendar, config.market_calendar)


class XnysCalendarTest(unittest.TestCase):
    def test_same_sessions_as_the_old_hand_list_in_2026_and_2027(self):
        day, diff = date(2026, 1, 1), []
        while day <= date(2027, 12, 31):
            old = day.weekday() < 5 and day.isoformat() not in OLD_HOLIDAYS
            if config.is_us_session(day) != old:
                diff.append(day.isoformat())
            day += timedelta(days=1)
        self.assertEqual(diff, [])

    def test_2028_holidays_no_longer_run_out(self):
        self.assertFalse(config.is_us_session(date(2028, 1, 17)))   # MLK day 2028
        self.assertFalse(config.is_us_session(date(2028, 11, 23)))  # Thanksgiving 2028
        self.assertTrue(config.is_us_session(date(2028, 1, 18)))
        self.assertEqual(config.next_rebalance("2028-01-14"), "2028-01-18")
        ny = datetime(2028, 1, 17, 8, tzinfo=config.NY)
        self.assertEqual(config.next_us_session(ny)[0], "2028-01-18")


class RecordFolderTest(unittest.TestCase):
    def test_normal_run_writes_real_folders(self):
        with mock.patch.object(config, "RECORDS_SCRATCH", None):
            real = ROOT / "data" / "news_evidence" / "bottom_outlooks"
            self.assertEqual(config.record_dir(real), real)
            self.assertEqual(config.real_record(real / "x.json"), real / "x.json")

    def test_scratch_mirror_and_read_only_fallback(self):
        from desk import finviz_basket, finviz_switch, finviz_trend
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(config, "RECORDS_SCRATCH", Path(tmp)):
            real = config.ROOT / "data" / "normalized" / "desk_finviz"
            scratch = config.record_dir(real)
            self.assertEqual(scratch, Path(tmp) / "data" / "normalized" / "desk_finviz")
            self.assertEqual(config.real_record(scratch / "2026-11.json"), real / "2026-11.json")
            self.assertTrue(str(finviz_switch.observation_dir()).startswith(tmp))
            self.assertTrue(str(finviz_trend.observation_dir()).startswith(tmp))
            if (real / "2026-11.json").exists():  # a frozen real month is read, never copied or rewritten
                basket = finviz_basket.load_basket("2026-11-20", scratch)
                self.assertEqual(basket["effective_date"], "2026-11-02")
                self.assertFalse(scratch.exists())

    def test_test_server_environment_moves_every_one_time_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = ("import json; from desk import config, finviz_basket, finviz_switch, finviz_trend, bottom_paper, season;"
                    "print(json.dumps([str(p) for p in (finviz_basket.DIRECTORY, finviz_switch.observation_dir(),"
                    " finviz_trend.observation_dir(), finviz_trend.DIRECTORY, bottom_paper.PAPER_DIR, season.PAPER)]))")
            env = dict(os.environ, DESK_JOURNAL_DIR=tmp)
            out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                                 encoding="utf-8", timeout=120)
            self.assertEqual(out.returncode, 0, out.stderr)
            paths = json.loads(out.stdout.strip().splitlines()[-1])
            scratch = str(Path(tmp).resolve() / "records")
            for p in paths:
                self.assertTrue(str(Path(p).resolve()).startswith(scratch), p)


class JournalGapTest(unittest.TestCase):
    def test_missing_close_uses_the_previous_close_and_is_listed(self):
        days = ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"]
        closes = pd.DataFrame({"X": [100.0, 101.0, np.nan, 103.0]})
        entries = [{"date": "2026-11-02", "type": "deposit", "amount": 1000.0},
                   {"date": "2026-11-02", "type": "buy", "ticker": "X", "qty": 10.0, "price": 100.0}]
        out = journal.history(entries, closes, days, np.ones(4))
        self.assertTrue(all(math.isfinite(v) and v > 0 for v in out["twr"]))
        self.assertAlmostEqual(out["twr"][2], 1.01)   # valued at the 11-03 close, not at 0
        self.assertAlmostEqual(out["twr"][3], 1.03)
        self.assertEqual([(f["date"], f["ticker"], f["basis"]) for f in out["filled"]],
                         [("2026-11-04", "X", "직전 종가")])

    def test_a_name_without_any_price_column_is_valued_at_its_trade_price(self):
        days = ["2026-11-02", "2026-11-03"]
        closes = pd.DataFrame({"QQQ": [1.0, 1.0]})
        entries = [{"date": "2026-11-02", "type": "deposit", "amount": 1000.0},
                   {"date": "2026-11-02", "type": "buy", "ticker": "NEW", "qty": 2.0, "price": 50.0}]
        out = journal.history(entries, closes, days, np.ones(2))
        self.assertEqual(out["usd"], [1000.0, 1000.0])
        self.assertEqual(out["twr"], [1.0, 1.0])
        self.assertEqual({f["basis"] for f in out["filled"]}, {"체결가"})


class JournalInputTest(unittest.TestCase):
    def test_nan_inf_and_text_are_400_errors_and_nothing_is_saved(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(config, "TRADES", Path(tmp) / "t.json"):
            journal.add_entry({"date": "2026-11-02", "type": "deposit", "amount": 100})
            bad = [{"type": "buy", "ticker": "X", "qty": math.nan, "price": 10},
                   {"type": "buy", "ticker": "X", "qty": 1, "price": math.inf},
                   {"type": "buy", "ticker": "X", "qty": "abc", "price": 10},
                   {"type": "buy", "ticker": "X", "qty": 1, "price": 10, "fee": "x"},
                   {"type": "deposit", "amount": "-inf"},
                   {"type": "withdraw", "amount": True}]
            for payload in bad:
                with self.subTest(payload=payload), self.assertRaises(journal.JournalError):
                    journal.add_entry({"date": "2026-11-03", **payload})
            self.assertEqual(len(journal.load_entries()), 1)
            journal.add_entry({"date": "2026-11-03", "type": "buy", "ticker": "x", "qty": "2", "price": "10.5", "fee": ""})
            self.assertEqual(journal.load_entries()[-1]["qty"], 2.0)

    def test_settings_tolerance_must_be_a_finite_number(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(config, "SETTINGS", Path(tmp) / "s.json"):
            for value in ("abc", math.nan, math.inf, ""):  # 빈 값이 조용히 0%가 되면 안 된다
                with self.subTest(value=value), self.assertRaises(journal.JournalError):
                    journal.save_settings({"tolerance": value})
            self.assertEqual(journal.save_settings({"tolerance": "0.05"})["tolerance"], 0.05)


def _frame(tickers, days=3):
    idx = pd.date_range("2026-11-02", periods=days, freq="B")
    cols = pd.MultiIndex.from_product([("Adj Close", "Close", "Open"), tickers])
    return pd.DataFrame(1.0, index=idx, columns=cols)


class PriceMetaTest(unittest.TestCase):
    def test_meta_records_received_tickers_and_retries_the_missing_ones(self):
        from desk import marketdata
        calls = []

        def download(tickers, **_):
            calls.append(list(tickers))
            frame = _frame(tickers)
            for field in ("Adj Close", "Close", "Open"):
                frame[(field, "GONE")] = np.nan  # Yahoo returned nothing for this one
            return frame

        fake = SimpleNamespace(download=download)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(config, "CACHE", Path(tmp)), \
                mock.patch.dict(sys.modules, {"yfinance": fake}):
            out = marketdata.history(["AAA", "GONE"])
            self.assertNotIn("GONE", out["Close"].columns)
            meta = json.loads((Path(tmp) / "prices_meta.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["tickers"], meta["missing"]), (["AAA"], ["GONE"]))
            self.assertEqual(marketdata.missing_tickers(), ["GONE"])
            marketdata.history(["AAA", "GONE"])   # within MISSING_RETRY_MIN: served from the cache
            self.assertEqual(len(calls), 1)
            meta["fetched_epoch"] = time.time() - (marketdata.MISSING_RETRY_MIN + 1) * 60
            (Path(tmp) / "prices_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            marketdata.history(["AAA"])           # everything wanted is present: still the cache
            self.assertEqual(len(calls), 1)
            marketdata.history(["AAA", "GONE"])   # the lost name is asked for again
            self.assertEqual(len(calls), 2)
            self.assertFalse(marketdata.retry_due())  # still lost after its one retry: no more 5-minute refetches
            meta = json.loads((Path(tmp) / "prices_meta.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["retried"])
            meta["fetched_epoch"] = time.time() - (marketdata.MISSING_RETRY_MIN + 1) * 60
            (Path(tmp) / "prices_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            marketdata.history(["AAA", "GONE"])   # inside PRICE_MAX_AGE_MIN the retried set is served from cache
            self.assertEqual(len(calls), 2)

    def test_a_broken_meta_file_downloads_again(self):
        from desk import marketdata
        calls = []
        fake = SimpleNamespace(download=lambda tickers, **_: calls.append(1) or _frame(tickers))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(config, "CACHE", Path(tmp)), \
                mock.patch.dict(sys.modules, {"yfinance": fake}):
            marketdata.history(["AAA"])
            (Path(tmp) / "prices_meta.json").write_text("{broken", encoding="utf-8")
            marketdata.history(["AAA"])
            self.assertEqual(len(calls), 2)


class NewsPriceExpiryTest(unittest.TestCase):
    def setUp(self):
        from desk import news
        self.news = news
        news._LAST_AUTO["at"] = None

    def test_stale_until_every_benchmark_has_the_last_completed_close(self):
        now = datetime(2026, 11, 20, 23, 0, tzinfo=timezone.utc)  # after the 11-20 close
        fresh = {"series": {"SOXX": {"2026-11-20": 1.0}, "QQQ": {"2026-11-20": 1.0}}}
        old = {"series": {"SOXX": {"2026-10-23": 1.0}, "QQQ": {"2026-11-20": 1.0}}}
        self.assertFalse(self.news.prices_stale(fresh, now))
        self.assertTrue(self.news.prices_stale(old, now))
        self.assertTrue(self.news.prices_stale({"series": {}}, now))

    def test_live_view_refreshes_a_stale_snapshot_once_and_warns_on_failure(self):
        now = datetime(2026, 11, 20, 23, 0, tzinfo=timezone.utc)
        old = {"series": {"SOXX": {"2026-10-23": 1.0}}, "fetched_at": "2026-10-25T05:01:00+00:00"}
        new = {"series": {"SOXX": {"2026-11-20": 2.0}}, "fetched_at": "2026-11-20T23:00:00+00:00"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news_prices.json"
            path.write_text(json.dumps(old), encoding="utf-8")
            with mock.patch.object(self.news, "PRICES", path), \
                    mock.patch.object(self.news, "refresh_prices", return_value=new) as refresh:
                self.assertEqual(self.news._auto_prices(now), (new, None))
                refresh.assert_called_once_with(download=False)
            self.news._LAST_AUTO["at"] = None
            with mock.patch.object(self.news, "PRICES", path), \
                    mock.patch.object(self.news, "refresh_prices", side_effect=OSError("offline")) as refresh:
                prices, warning = self.news._auto_prices(now)
                self.assertEqual(prices["fetched_at"], old["fetched_at"])
                self.assertIn("자동 갱신 실패", warning)
                self.assertEqual(self.news._auto_prices(now), (old, None))  # throttled: no retry storm
                self.assertEqual(refresh.call_count, 1)

    def test_fresh_snapshot_is_not_refreshed(self):
        now = datetime(2026, 11, 20, 23, 0, tzinfo=timezone.utc)
        fresh = {"series": {"SOXX": {"2026-11-20": 2.0}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news_prices.json"
            path.write_text(json.dumps(fresh), encoding="utf-8")
            with mock.patch.object(self.news, "PRICES", path), \
                    mock.patch.object(self.news, "refresh_prices") as refresh:
                self.assertEqual(self.news._auto_prices(now), (fresh, None))
                refresh.assert_not_called()


class ValidateGateTest(unittest.TestCase):
    def test_missing_774_input_fails_unless_skipped_on_purpose(self):
        from desk import validate
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(validate, "NPZ", Path(tmp) / "none.npz"), \
                mock.patch.object(validate.config, "ROOT", Path(tmp)), mock.patch("builtins.print"):
            self.assertEqual(validate.check_774(None, None), (False, "FAIL"))
            self.assertEqual(validate.check_774(None, None, skip=True), (True, "미검증"))

    def test_unknown_option_is_rejected_before_any_work(self):
        from desk import validate
        with mock.patch.object(validate.S, "load_book", side_effect=AssertionError("no work")), \
                mock.patch("builtins.print"):
            self.assertEqual(validate.main(["--skip774"]), 2)


class DigestTimingTest(unittest.TestCase):
    def test_digest_reads_completed_closes_only(self):
        from desk import digest

        class Stop(Exception):
            pass

        with mock.patch.object(digest.S, "load_book", side_effect=Stop) as load, \
                mock.patch.object(digest, "last_completed_session", return_value="2026-11-20"):
            with self.assertRaises(Stop):
                digest.briefing_block(engine=object())
        self.assertEqual(load.call_args.kwargs.get("through"), "2026-11-20")


class BriefingLockTest(unittest.TestCase):
    def test_briefing_runs_outside_the_desk_lock_and_only_once(self):
        from desk import state
        desk = state.Desk()
        seen = {}

        def run(*_, **__):
            got = []
            t = threading.Thread(target=lambda: got.append(desk.lock.acquire(timeout=1) and (desk.lock.release() or True)))
            t.start()
            t.join()
            seen["other_thread_got_lock"] = got == [True]
            seen["busy"] = desk.busy
            seen["second"] = desk.run_briefing()
            return SimpleNamespace(returncode=0, stdout="done", stderr="")

        with mock.patch.object(state.subprocess, "run", side_effect=run), mock.patch.object(desk, "ensure") as ensure:
            out = desk.run_briefing()
        self.assertTrue(out["ok"])
        self.assertTrue(seen["other_thread_got_lock"])
        self.assertTrue(seen["busy"])
        self.assertFalse(seen["second"]["ok"])
        ensure.assert_called_once_with(force=True)
        self.assertIsNone(desk.busy)


class SeasonResidualLineTest(unittest.TestCase):
    def test_momentum_ic_can_score_the_live_rule_field(self):
        from desk import season
        n_days, names = 30, [f"K{i}" for i in range(9)]
        growth = np.linspace(0.0, 0.02, len(names))  # K8 rises fastest
        adj = pd.DataFrame({k: 100 * np.cumprod(np.full(n_days, 1 + g)) for k, g in zip(names, growth)})
        book = SimpleNamespace(days=[f"d{i:02d}" for i in range(n_days)], adj=adj, ticker={k: k for k in names})
        ranked = [{"key": k, "score": float(i), "mom120": float(-i)} for i, k in enumerate(names)]
        days = book.days[5:20]
        rankings = [ranked] * len(days)
        by_score = season.momentum_ic(book, days, rankings, 5, field="score")
        by_ret = season.momentum_ic(book, days, rankings, 5)
        self.assertTrue(np.allclose(by_score[np.isfinite(by_score)], 1.0))
        self.assertTrue(np.allclose(by_ret[np.isfinite(by_ret)], -1.0))

    def test_unreadable_paper_ledger_is_an_error_not_a_reset(self):
        from desk import season
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "season_paper.json"
            path.write_text("{\"records\": [", encoding="utf-8")
            with mock.patch.object(season, "PAPER", path):
                with self.assertRaises(season.SeasonError):
                    season._load_paper()
            self.assertEqual(path.read_text(encoding="utf-8"), "{\"records\": [")


class BottomOutlookCaveatTest(unittest.TestCase):
    def test_caveat_states_the_drawdown_lower_breaks(self):
        from desk import bottom_outlook
        text = bottom_outlook.drawdown_caveat({"n": 9, "below": 4, "above": 0})
        self.assertIn("4/4", text)
        self.assertIn("하단 이탈", text)
        self.assertIsNone(bottom_outlook.drawdown_caveat({"n": 9, "below": 0, "above": 2}))
        self.assertIsNone(bottom_outlook.drawdown_caveat({"n": 0, "below": 0, "above": 0}))


class RegenerableCacheTest(unittest.TestCase):
    """Untracked desk caches (data/derived/desk) are rebuilt when missing or broken."""

    def _check(self, module, compute_name="compute", extra=()):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            patches = [mock.patch.object(module, "CACHE_PATH", path),
                       mock.patch.object(module, compute_name, return_value={"v": 1}), *extra]
            for p in patches:
                p.start()
            try:
                module.payload()
                self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["v"], 1)
                path.write_text("<<<<<<< merge marker", encoding="utf-8")
                module.payload()
                self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["v"], 1)
                self.assertEqual(getattr(module, compute_name).call_count, 2)
            finally:
                for p in reversed(patches):
                    p.stop()

    def test_value(self):
        from desk import value
        self._check(value)

    def test_dip(self):
        from desk import dip
        self._check(dip, extra=[mock.patch.object(dip, "research", return_value={})])

    def test_giant(self):
        from desk import giant
        self._check(giant, extra=[mock.patch.object(giant, "add_comparison_views", side_effect=lambda d: d)])


if __name__ == "__main__":
    unittest.main()
