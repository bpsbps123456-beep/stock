"""Current SOXS publication completeness and hypothetical USD thresholds."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

FILE = Path(__file__).resolve().parents[1] / "data/derived/analysis/seibro_bottoms/code/soxs_current.py"
SPEC = importlib.util.spec_from_file_location("seibro_soxs_current_tests", FILE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class SoxsCurrentTest(unittest.TestCase):
    def records(self, day, count):
        return [{"kind": "daily", "dtype": dtype, "start": day, "end": day,
                 "n": count, "rows": [{}] * count} for dtype in "1234"]

    def test_empty_or_partial_newer_date_is_not_a_zero_flow_observation(self):
        rows = self.records("2026-11-04", 1) + self.records("2026-11-05", 0)
        rows += self.records("2026-11-06", 1)[:3]
        cutoff, availability = M.complete_daily_cutoff({"requests": rows})
        self.assertEqual(cutoff, pd.Timestamp("2026-11-04"))
        self.assertEqual(availability["complete_dates"], ["2026-11-04"])
        self.assertEqual(len(availability["excluded"]), 2)

    def test_threshold_accounts_for_current_inclusion_and_ties(self):
        # Replacing 250 with 200 gives two observations equal to 200, exactly P80.
        self.assertEqual(M.required_usd_for_p80(np.arange(1., 251.)), 200.)
        # Equal observations all rank P50; one extra dollar exceeds all past ones.
        self.assertEqual(M.required_usd_for_p80(np.full(120, 10.)), 11.)
        self.assertEqual(M.required_usd_for_p80(np.arange(-250., 0.)), -51.)

    def test_threshold_requires_valid_current_value_and_rank_warmup(self):
        self.assertIsNone(M.required_usd_for_p80(np.arange(119.)))
        values = np.arange(150.)
        values[-1] = np.nan
        self.assertIsNone(M.required_usd_for_p80(values))
        self.assertEqual(M.required_usd_for_p80(np.zeros(120), lower_bound=0), 1.)


if __name__ == "__main__":
    unittest.main()
