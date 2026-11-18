"""Check the bottom forecast against fixed data and historical information boundaries."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from desk import bottom_outlook as model


class BottomOutlookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (Path(__file__).resolve().parents[1] / "data/derived/analysis"
                / "news_bottom_outlook_20261119/soxx_close.csv")
        cls.close = pd.read_csv(path, parse_dates=["date"], index_col="date")["SOXX"]
        cls.outlook, cls.checks = model.replay_outlook(cls.close)

    def test_replay_uses_the_same_forecast_and_only_mature_labels(self):
        case = self.checks[2]
        history = self.close.loc[:case["date"]]
        actual = model.build_outlook(history)
        expected = history.iloc[-1] * (1 + case["bands"][actual["method"]])
        np.testing.assert_allclose([actual[k] for k in ("lower", "center", "upper")], expected)
        panel = model._panel(self.close)
        pos = panel.index.get_loc(case["date"])
        # Labels after pos - horizon are incomplete at the prediction close.
        panel.loc[panel.index[pos - model.HORIZON + 1]:, "target"] = 42.0
        _, shocked = model._bands(panel, pos)
        np.testing.assert_allclose(case["bands"][model.BASE], shocked[model.BASE])
        np.testing.assert_allclose(case["bands"][model.VOL], shocked[model.VOL])

    def test_target_excludes_today_and_keeps_missing_sessions(self):
        panel = model._panel(self.close)
        pos = 50
        origin = panel.index[pos]
        expected = panel["close"].iloc[pos+1:pos+11].min() / panel["close"].iat[pos] - 1
        self.assertAlmostEqual(panel.at[origin, "target"], expected)
        missing_day = panel.index[pos+3]
        panel_with_gap = model._panel(self.close.drop(missing_day))
        self.assertTrue(np.isnan(panel_with_gap.at[origin, "target"]))
        self.assertTrue(np.isnan(panel_with_gap.at[missing_day, "close"]))

    def test_disjoint_forward_check_and_frozen_metrics(self):
        out = self.outlook
        self.assertEqual(out["as_of"], "2026-11-17")
        stats = out["validation"]
        self.assertEqual((stats["n"], stats["hits"], stats["below"], stats["above"]), (9, 5, 4, 0))
        self.assertEqual(stats["drawdown"]["n"], 9)
        self.assertEqual(out["method"], model.BASE)
        self.assertAlmostEqual(out["center"], 135.26, places=2)
        for earlier, later in zip(self.checks, self.checks[1:]):
            self.assertLessEqual(earlier["end"], later["date"])

    def test_short_or_missing_recent_history_withholds_forecast(self):
        for close in [self.close.iloc[:0], self.close.loc[:"2026-05-29"],
                      self.close.drop(self.close.index[-2])]:
            out = model.build_outlook(close)
            self.assertFalse(out["available"])
            self.assertIsNone(out["lower"])
            self.assertTrue(out["reason"])
