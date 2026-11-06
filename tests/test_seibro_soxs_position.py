"""Causal price-position display and exact-session alignment contracts."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / "data/derived/analysis/seibro_bottoms/code/soxs_position.py"
spec = importlib.util.spec_from_file_location("test_soxs_position_module", PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class PositionTests(unittest.TestCase):
    def test_range_uses_no_future_and_requires_full_window(self):
        pf = pd.DataFrame({"SOXX_Close": np.arange(1., 81.)})
        original = mod.price_locations(pf)
        pf.loc[65:, "SOXX_Close"] = .01
        changed = mod.price_locations(pf)
        pd.testing.assert_series_equal(original.range60.iloc[:65], changed.range60.iloc[:65])
        self.assertTrue(original.range60.iloc[:59].isna().all())
        self.assertEqual(original.range60.iloc[59], 100.)
        self.assertEqual(changed.range60.iloc[65], 0.)

    def test_previous_session_does_not_substitute_last_available(self):
        sessions = pd.bdate_range("2026-01-01", periods=5)
        pf = pd.DataFrame({"SOXX_Close": [100.], "dd60": [-.1], "range60": [40.]}, index=sessions[:1])
        exact = mod.preceding_location(sessions[1], sessions, pf)
        self.assertEqual(exact["close"], 100.)
        self.assertEqual(exact["dd60"], -10.)
        self.assertIsNone(mod.preceding_location(sessions[2], sessions, pf)["close"])

    def test_missing_statuses_and_empty_on_are_not_zero_observations(self):
        rows = [{"status": s, "dd60": d, "range60": 0.}
                for s, d in [("ON", -5.), ("ON", -10.), ("ON", -20.),
                             ("UNKNOWN", -30.), ("UNPUBLISHED", -40.), ("NO_NEW_SOURCE", 0.)]]
        result = mod.summarize(rows, "ON")
        self.assertEqual(result["n"], 3)
        self.assertEqual([b["n"] for b in result["bins"]], [0, 1, 1, 1])
        empty = mod.ledger.clean(mod.summarize(rows, "OFF"))
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["median_dd60"])


if __name__ == "__main__":
    unittest.main()
