"""Causal timing, raw-bar integrity, and calendar-matched placebo checks."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

FILE = Path(__file__).resolve().parents[1] / "data/derived/analysis/sox_open_switch/code/premarket_fade.py"
SPEC = importlib.util.spec_from_file_location("premarket_fade_tested", FILE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def synthetic():
    class Data:
        days = [pd.Timestamp("2026-07-02").date(), pd.Timestamp("2026-07-06").date()]
        rth_end = np.array([M.LIB.idx("13:00"), M.LIB.idx("16:00")])
    d = Data()
    d.A, d.has = {}, {}
    for sym in ("SOXL", "SOXS"):
        d.A[sym] = {f: np.full((2, 960), 100.0 if f != "Volume" else 1000.0)
                    for f in ("Open", "High", "Low", "Close", "Volume")}
        d.has[sym] = np.ones((2, 960), dtype=bool)
    d.A["SOXL"]["Close"][0, M.LIB.idx("12:59")] = 100
    d.A["SOXL"]["Close"][0, M.LIB.idx("15:59")] = 999
    d.A["SOXL"]["Close"][1, M.SIGNAL_BAR] = 106
    d.A["SOXS"]["Close"][1, M.ENTRY_BAR] = 101
    d.A["SOXS"]["Close"][1, M.LIB.idx("09:58")] = 104
    d.A["SOXS"]["Close"][1, M.LIB.idx("09:44")] = 103
    return d


class PremarketFadeTest(unittest.TestCase):
    def test_prior_halfday_close_and_no_current_future_price(self):
        d = synthetic()
        self.assertAlmostEqual(M.gap_for_day(d, 1), 6)
        d.A["SOXL"]["Close"][1, M.SIGNAL_BAR + 1:] = 999999
        self.assertAlmostEqual(M.gap_for_day(d, 1), 6)

    def test_missing_or_zero_volume_bar_not_executable(self):
        d = synthetic()
        d.has["SOXS"][1, M.ENTRY_BAR] = False
        self.assertFalse(M.valid_bar(d, "SOXS", 1, M.ENTRY_BAR))
        d.has["SOXS"][1, M.ENTRY_BAR] = True
        d.A["SOXS"]["Volume"][1, M.ENTRY_BAR] = 0
        self.assertFalse(M.valid_bar(d, "SOXS", 1, M.ENTRY_BAR))

    def test_engine_fills_only_specified_minutes(self):
        d = synthetic()
        costs = {s: {k: 0.0 for k in ("pre", "open5", "early", "late")} for s in ("SOXL", "SOXS")}
        for actual, bar, px in (("09:59", "09:58", 104), ("09:45", "09:44", 103)):
            r, trades = M.LIB.Engine(d, costs=costs, scheme="std").run_day(M.FixedWindow(-1, actual), 1)
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0][2:4], ("09:00", bar))
            self.assertAlmostEqual(r, px / 101 - 1)
            # Prices after the pre-registered exit cannot affect the return.
            e = synthetic()
            e.A["SOXS"]["Close"][1, M.LIB.idx(actual):] = 90000
            r2, _ = M.LIB.Engine(e, costs=costs, scheme="std").run_day(M.FixedWindow(-1, actual), 1)
            self.assertEqual(r, r2)

    def test_null_preserves_each_month_counts_and_valid_days(self):
        dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-02-02", "2026-02-03"])
        sig = np.array([True, False, False, False, True])
        eligible = np.array([True, True, False, True, True])
        x = M.month_matched_null(dates, sig, np.ones(5), eligible, nboot=100)
        np.testing.assert_array_equal(x[:, :3].sum(1), np.ones(100))
        np.testing.assert_array_equal(x[:, 3:].sum(1), np.ones(100))
        self.assertTrue((x[:, 2] == 0).all())


if __name__ == "__main__":
    unittest.main()
