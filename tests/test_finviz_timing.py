"""Execution schedules, shadow baskets and financing regression checks."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("finviz_timing", ROOT / "data/derived/analysis/finviz_recipes_20261120/timing.py")
TIMING = module_from_spec(SPEC)
SPEC.loader.exec_module(TIMING)


def source(days, opens, closes, issuers=None):
    o, c = np.asarray(opens, float), np.asarray(closes, float)
    names = [f"S{i}" for i in range(o.shape[1])]
    return SimpleNamespace(days=days, symbols=names, opn=o, close=c, fx=np.ones(len(days)),
                           tradable=np.ones_like(o, bool), settlements=np.zeros_like(o, bool),
                           issuers=np.tile(issuers if issuers is not None else names, (len(days), 1)),
                           sectors=np.full_like(o, "Sector", dtype=object))


class FinvizTimingTests(unittest.TestCase):
    def test_calendar_holiday_and_month_ordinals_are_prefix_invariant(self):
        days = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
                "2026-01-09", "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15",
                "2026-01-16", "2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23",
                "2026-02-02", "2026-02-03"]
        gate = [True] * len(days)
        expected = {"weekly3": [3, 8, 13], "monthly6": [5], "monthly11": [10],
                    "monthly1_11": [0, 10, 15], "biweekly": [1, 11, 15]}
        for name in TIMING.SCHEDULES:
            full = TIMING.review_days(days, gate, name)
            if name in expected:
                self.assertEqual(full, expected[name])
            for end in range(1, len(days)):
                self.assertEqual(TIMING.review_days(days[:end], gate[:end], name),
                                 [i for i in full if i < end])

    def test_recovery_restarts_risk_off_refresh_interval(self):
        days = [f"2026-01-{i:02d}" for i in range(1, 15)]
        gate = [True, False, False, False, False, False, False, True,
                False, False, False, False, False, False]
        self.assertEqual(TIMING.review_days(days, gate, "exit"), [1, 8])
        self.assertEqual(TIMING.review_days(days, gate, "exit5"), [1, 6, 8, 13])

    def test_shadow_drift_midmonth_and_calendar_start_cash(self):
        days = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
        s = source(days, [[1, 1], [1, 1], [3, 1], [3, 1]], [[1, 1], [2, 1], [3, 1], [3, 1]])
        gate = np.array([True, True, False, True])
        engine = TIMING.TimingEngine(s, gate, {0: np.array([0., 1.])})
        ranks = [[0], [1], [1], [1]]
        out = engine.evaluate(ranks, "monthly1", detail=True, stress=True)
        # The January F shadow grew from 1/7 to 3/(3+6), despite the newer S1 rank.
        self.assertAlmostEqual(out["switch"].exposure.iloc[2], 1/3)
        self.assertEqual(out["switch"].holdings.iloc[2], "S0")
        self.assertEqual(out["switch"].holdings.iloc[3], "S1")
        self.assertAlmostEqual(out["always"].attrs["final_weights"]["S0"], 1/3)
        self.assertEqual(out["always"].attrs["final_weights"], out["always"].attrs["final_weights_10bp"])
        self.assertTrue((engine.evaluate(ranks, "monthly6")["always"].active == 0).all())
        exit_path = engine.evaluate(ranks, "exit", detail=True)
        self.assertEqual(set(exit_path), {"switch"})
        self.assertEqual(exit_path["switch"].holdings.iloc[2], "S1")

    def test_cost_amounts_use_actual_net_orders_and_end_weights(self):
        days = ["2026-01-05", "2026-01-06"]
        s = source(days, np.ones((2, 14)), np.ones((2, 14)))
        engine = TIMING.TimingEngine(s, np.array([False, False]), {})
        f = engine.evaluate([list(range(7)), list(range(7, 14))], "daily",
                            detail=True, stress=True)["always"]
        for bps, col in [(.001, "krw10"), (.0025, "krw25")]:
            self.assertAlmostEqual(f[col].iloc[0], 1/(1+bps))
            self.assertAlmostEqual(f[col].iloc[1], (1-bps)/(1+bps)**2)
        self.assertAlmostEqual(f.attrs["final_weights"]["CASH"], 0.)
        self.assertAlmostEqual(f.attrs["final_weights_10bp"]["S9"], 1/7)

    def test_merger_and_duplicate_share_classes_leave_stated_vacancies(self):
        days = ["2026-01-05", "2026-01-06"]
        s = source(days, np.ones((2, 9)), np.ones((2, 9)), ["A","A","B","C","D","E","F","G","H"])
        s.opn[1, 2] = s.close[1, 2] = 2.
        s.settlements[1, 2] = True
        s.tradable[1, 2] = False
        engine = TIMING.TimingEngine(s, np.array([False, False]), {})
        ranks = [list(range(9))] * len(days)
        path = engine.evaluate(ranks, "daily", detail=True, stress=True)["always"]
        self.assertEqual(path.active.tolist(), [6, 5])
        self.assertNotIn("S7", path.holdings.iloc[-1])
        self.assertNotIn("S8", path.holdings.iloc[-1])
        self.assertAlmostEqual(path.usd.iloc[-1], 8/7)
        self.assertAlmostEqual(path.attrs["final_weights"]["CASH"], 2/7)
