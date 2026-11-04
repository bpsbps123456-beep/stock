"""BRIEF8 실제호가 검증: 잘못된 호가 배제와 분할 단위 불변성."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("premarket_quotes", ROOT / "data/derived/analysis/sox_open_switch/code/premarket_quotes.py")
quotes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quotes)


class PremarketQuotesTest(unittest.TestCase):
    def record(self, target="2026-07-08T09:01:00-04:00", stamp="2026-07-08T09:00:59-04:00", **change):
        q = {"t": stamp, "bp": 99.0, "ap": 101.0, "bs": 10, "as": 20}
        q.update(change)
        return quotes.quote_record("SOXS", target, q)

    def test_recent_quote_preserves_actual_timestamp_and_size(self):
        q = self.record()
        self.assertTrue(q["valid"])
        self.assertEqual(q["age_seconds"], 1)
        self.assertEqual(q["ask_size"], 20)
        self.assertEqual(q["quote_timestamp"], "2026-07-08T13:00:59+00:00")

    def test_stale_future_crossed_and_empty_size_rejected(self):
        self.assertEqual(self.record(stamp="2026-07-08T09:00:29-04:00")["reason"], "stale_quote")
        self.assertTrue(self.record(stamp="2026-07-08T09:00:30-04:00")["valid"])
        self.assertEqual(self.record(stamp="2026-07-08T09:01:01-04:00")["reason"], "future_quote")
        self.assertEqual(self.record(bp=102)["reason"], "crossed_quote")
        self.assertEqual(self.record(**{"as": 0})["reason"], "empty_size")
        self.assertEqual(self.record(bs=0)["reason"], "empty_size")

    def test_missing_and_nonfinite_quote_rejected(self):
        self.assertEqual(quotes.quote_record("SOXS", "2026-07-08T09:01:00-04:00", None)["reason"], "missing_quote")
        self.assertEqual(self.record(bp=float("nan"))["reason"], "nonfinite_quote")

    def test_same_day_raw_return_and_double_spread(self):
        entry = self.record()
        exit_ = self.record(target="2026-07-08T09:59:00-04:00", stamp="2026-07-08T09:58:59-04:00", bp=109, ap=111)
        a = quotes.audit_trade(entry, exit_, 10, 11)
        b = quotes.audit_trade(entry, exit_, 100, 110)
        self.assertTrue(a["quote_valid"])
        self.assertAlmostEqual(a["quote_return_pct"], (109 / 101 - 1) * 100)
        self.assertAlmostEqual(a["quote_return_cost2_pct"], (108 / 102 - 1) * 100)
        self.assertAlmostEqual(a["quote_return_pct"], b["quote_return_pct"])
        self.assertLess(a["quote_return_cost2_pct"], a["quote_return_pct"])
        self.assertAlmostEqual(a["entry_mid_raw_to_adjusted"], 10)

    def test_changed_split_units_flagged_without_selectively_removing_return(self):
        entry = self.record()
        exit_ = self.record(target="2026-07-08T09:59:00-04:00", stamp="2026-07-08T09:58:59-04:00", bp=109, ap=111)
        result = quotes.audit_trade(entry, exit_, 10, 110)
        self.assertTrue(result["quote_valid"])
        self.assertTrue(result["ratio_flag"])
        self.assertEqual(result["ratio_reason"], "raw_adjusted_ratio_changed")
        self.assertAlmostEqual(result["quote_return_pct"], (109 / 101 - 1) * 100)
        exit_ = self.record(target="2026-07-09T09:59:00-04:00", stamp="2026-07-09T09:58:59-04:00")
        self.assertEqual(quotes.audit_trade(entry, exit_, 100, 100)["quote_reason"], "not_same_day_ordered_trade")

    def test_naive_quote_time_is_not_silently_localized(self):
        with self.assertRaises(ValueError):
            quotes.aware_timestamp("2026-07-08 09:01")

    def test_summary_keeps_flagged_loss_and_compares_only_matched_model_rows(self):
        frame = quotes.pd.DataFrame([
            {"strategy_id": "g5_exit0959", "day": "2026-07-08", "quote_valid": True,
             "quote_reason": "ok", "ratio_flag": False, "quote_return_pct": 2,
             "quote_return_cost2_pct": 1.8, "return_pct": 3},
            {"strategy_id": "g5_exit0959", "day": "2026-07-09", "quote_valid": True,
             "quote_reason": "ok", "ratio_flag": True, "quote_return_pct": -30,
             "quote_return_cost2_pct": -30.2, "return_pct": -29},
            {"strategy_id": "g5_exit0959", "day": "2026-07-10", "quote_valid": False,
             "quote_reason": "entry:stale_quote", "ratio_flag": False, "quote_return_pct": float("nan"),
             "quote_return_cost2_pct": float("nan"), "return_pct": 90}])
        row = quotes.summarize(frame)["rows"][0]
        self.assertEqual(row["n_valid"], 2)
        self.assertEqual(row["n_excluded"], 1)
        self.assertEqual(row["mean_trade_return_pct"], -14)
        self.assertEqual(row["mean_trade_return_without_ratio_flags_pct"], 2)
        self.assertEqual(row["mean_model_return_on_same_trades_pct"], -13)

    def daily_inputs(self):
        daily = quotes.pd.DataFrame([
            {"strategy_id": "g5_exit0959", "day": "2026-07-08", "signal": True},
            {"strategy_id": "g5_exit0959", "day": "2026-07-09", "signal": False},
            {"strategy_id": "g5_exit0959", "day": "2026-07-10", "signal": True}])
        trades = quotes.pd.DataFrame([
            {"strategy_id": "g5_exit0959", "day": "2026-07-08", "quote_valid": True,
             "ratio_flag": False, "quote_return_pct": 10.0, "quote_return_cost2_pct": 9.0},
            {"strategy_id": "g5_exit0959", "day": "2026-07-10", "quote_valid": True,
             "ratio_flag": True, "quote_return_pct": -10.0, "quote_return_cost2_pct": -11.0}])
        return daily, trades

    def test_daily_stats_use_full_calendar_and_retain_flagged_loss(self):
        daily, trades = self.daily_inputs()
        row = quotes.daily_quote_statistics(daily, trades)["rows"][0]["periods"]["2026_back"]
        self.assertEqual(row["days"], 3)
        self.assertEqual(row["signals"], 2)
        self.assertEqual(row["per_day_pct"], 0)
        self.assertEqual(row["n_ratio_flagged_valid"], 1)
        self.assertAlmostEqual(row["cost2_per_day_pct"], -2 / 3)
        self.assertAlmostEqual(row["compound_pct"], -1)
        self.assertAlmostEqual(row["max_dd_pct"], -10)
        self.assertEqual(row["drop_top2_per_day_pct"], -10)

    def test_daily_stats_missing_invalid_duplicate_and_extra_quotes_fail(self):
        daily, trades = self.daily_inputs()
        for bad in (trades.iloc[:1], quotes.pd.concat([trades, trades.iloc[:1]]),
                    trades.assign(quote_valid=False), trades.assign(quote_return_pct=float("nan"))):
            with self.assertRaises(ValueError):
                quotes.daily_quote_statistics(daily, bad)
        extra = trades.iloc[:1].assign(day="2026-07-09")
        with self.assertRaises(ValueError):
            quotes.daily_quote_statistics(daily, quotes.pd.concat([trades, extra]))
        with self.assertRaises(ValueError):
            quotes.daily_quote_statistics(daily, trades, expected_counts={"2026_all": 4})


if __name__ == "__main__":
    unittest.main()
