"""Critical execution boundaries for the SOXL closing-entry research model."""
import importlib.util
import unittest
from pathlib import Path

import exchange_calendars as xc
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / "data/derived/analysis/soxl_close_bracket_20261111/study.py"
SPEC = importlib.util.spec_from_file_location("close_bracket_test_subject", PATH)
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


class CloseBracketTests(unittest.TestCase):
    def single(self, op, hi, lo):
        return pd.DataFrame({"Open": [op], "High": [hi], "Low": [lo]}, index=pd.DatetimeIndex(["2026-01-05 09:30"], tz=STUDY.ET))

    def data(self):
        schedule = xc.get_calendar("XNYS").schedule.loc["2026-01-02":"2026-01-05"]
        chunks = []
        for _, s in schedule.iterrows():
            idx = pd.date_range(s["open"], s["close"]-pd.Timedelta(minutes=1), freq="min").tz_convert(STUDY.ET)
            chunks.append(pd.DataFrame({"Open": 100., "High": 100., "Low": 100., "Close": 100., "Volume": 100}, index=idx))
        return pd.concat(chunks), pd.Series({"2026-01-02": 100., "2026-01-05": 100.}), schedule

    def test_opening_price_precedes_extremes(self):
        down = STUDY.first_exit(self.single(95, 104, 94), 100)
        up = STUDY.first_exit(self.single(104, 106, 90), 100)
        self.assertEqual((down["reason"], down["exit_price"]), ("stop_gap", 95))
        self.assertEqual((up["reason"], up["exit_price"]), ("take_profit_gap", 104))
        self.assertFalse(up["ambiguity"])
        self.assertEqual(STUDY.first_exit(self.single(104, 106, 90), 100, gap_tp="target")["exit_price"], 102)

    def test_same_minute_order_is_explicit(self):
        f = self.single(100, 104, 96)
        conservative = STUDY.first_exit(f, 100)
        optimistic = STUDY.first_exit(f, 100, "target_first")
        self.assertTrue(conservative["ambiguity"])
        self.assertEqual(conservative["exit_price"], 97)
        self.assertEqual(optimistic["exit_price"], 102)

    def test_prices_before_closing_entry_do_not_trigger_exit(self):
        f, c, s = self.data()
        f.loc["2026-01-02", "Low"] = 80
        t, _, _ = STUDY.simulate(f, c, s)
        self.assertEqual(len(t), 1)
        self.assertEqual(t.iloc[0].reason, "open_censored")

    def test_late_exit_defers_closing_reentry(self):
        f, c, s = self.data()
        f.loc[pd.Timestamp("2026-01-05 15:50", tz=STUDY.ET), "Low"] = 96
        t, e, extra = STUDY.simulate(f, c, s)
        self.assertEqual(len(t), 1)
        self.assertEqual(t.iloc[0].reason, "stop")
        self.assertEqual(int(e.iloc[-1].position), 0)
        self.assertEqual(extra["late_exit_reentry_deferred_days"], 1)

    def test_entry_closing_minute_and_terminal_afterhours_are_excluded(self):
        f, c, s = self.data()
        for ts in ("2026-01-02 16:00", "2026-01-05 16:01"):
            f.loc[pd.Timestamp(ts, tz=STUDY.ET)] = [100, 100, 80, 100, 100]
        t, _, _ = STUDY.simulate(f.sort_index(), c, s, "EXT")
        self.assertEqual(len(t), 1)
        self.assertEqual(t.iloc[0].reason, "open_censored")

    def test_no_overlapping_position_and_costs_are_separate(self):
        f, c, s = self.data()
        f.loc[pd.Timestamp("2026-01-05 09:30", tz=STUDY.ET)] = [95, 95, 95, 95, 100]
        t, e, _ = STUDY.simulate(f, c, s)
        self.assertEqual(len(t), 2)
        self.assertEqual(t.iloc[0].exit_price, 95)
        self.assertEqual(t.iloc[1].reason, "open_censored")
        self.assertAlmostEqual(t.iloc[0].net_10bp_pct, (95*.999/(100*1.001)-1)*100)
        self.assertAlmostEqual(e.iloc[-1].nav_10bp, 100/1.001*.95*.999/1.001)


if __name__ == "__main__":
    unittest.main()
