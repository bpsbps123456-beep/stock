"""Small deterministic cases for the entry-time threshold and delayed sell."""
import unittest

import numpy as np
import pandas as pd

from replay import dead_times, session_days, sim


class ReplayTests(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2026-11-16 04:00", periods=66, freq="3min", tz="America/New_York")
        self.bars = pd.DataFrame(dict(o=100., l=100., c=100., r=50., ema=100.), index=idx)
        self.bars.loc[idx[61], ["l", "r"]] = [97., 65.]
        self.blocked = np.zeros(len(idx), dtype=bool)

    def test_entry_threshold_and_pending_sell(self):
        threshold = np.full(len(self.bars), 70.)
        threshold[61] = 60.
        allowed = ~self.blocked.copy()
        allowed[62:64] = False
        self.bars.loc[self.bars.index[64], "o"] = 94.
        trades = sim(self.bars, dead_times(self.bars), self.blocked, x=threshold, exit_allowed=allowed)
        self.assertEqual(trades, [(61, 64, 98., 94., "RSI")])

    def test_terminal_mark_and_overnight_day(self):
        self.bars.loc[self.bars.index[-1], "c"] = 96.
        self.assertEqual(sim(self.bars, dead_times(self.bars), self.blocked), [])
        self.assertEqual(sim(self.bars, dead_times(self.bars), self.blocked, mark_end=True),
                         [(61, 65, 98., 96., "평가")])
        idx = pd.DatetimeIndex(["2026-10-30 19:57", "2026-11-01 20:00", "2026-11-02 04:00"],
                               tz="America/New_York")
        self.assertEqual(session_days(idx).strftime("%Y-%m-%d").tolist(),
                         ["2026-10-30", "2026-11-02", "2026-11-02"])


if __name__ == "__main__":
    unittest.main()
