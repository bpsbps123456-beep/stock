"""Small share-and-cash examples for the seven-slot research ledger."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("seven_sleeve_slots", ROOT / "data/derived/analysis/seven_sleeves_20261120/slots.py")
SLOTS = module_from_spec(SPEC)
SPEC.loader.exec_module(SLOTS)
Market, replay = SLOTS.Market, SLOTS.replay


def slots_for(*groups):
    return [*groups, *[("empty", "monthly")] * (7 - len(groups))]


class SevenSlotTests(unittest.TestCase):
    def test_gaps_apply_to_overnight_owner_and_fees_are_self_financing(self):
        days = ["2026-01-05", "2026-01-06"]
        m = Market(days, ["X", "Y"], [[10, 20], [25, 30]], [[20, 20], [5, 33]],
                   [1400, 1400, 1540], ["x", "y"])
        ranks = {"D": [[0], [1]], "empty": [[], []]}
        curve = replay(m, ranks, slots_for(("D", "daily")), detail=True)
        np.testing.assert_allclose(curve.usd, [(6 + 2) / 7, (6 + 2.5 * 1.1) / 7])
        np.testing.assert_allclose(curve.turn, [1 / 7, 5 / 8.5])
        net = (6 + 2.5 * (1 - .001) * 1.1 / (1 + .001)**2) / 7
        self.assertAlmostEqual(curve.krw10.iloc[-1], net * 1.1)
        self.assertEqual(curve.holdings.iloc[-1], "Y")

    def test_frequency_held_quantities_drift_until_monthly_reset(self):
        days = ["2026-01-29", "2026-01-30", "2026-02-02", "2026-02-03"]
        m = Market(days, ["X", "Y", "Z"], [[1, 1, 1], [2, 1, 1], [2, 1, 1], [2, 2, 1]],
                   [[2, 1, 1], [2, 1, 1], [2, 2, 1], [2, 2, 1]], [1]*4, ["x", "y", "z"])
        ranks = {"D": [[0], [2], [0], [2]], "M": [[1], [2], [1], [2]], "empty": [[]]*4}
        c = replay(m, ranks, slots_for(("D", "monthly"), ("M", "weekly")), detail=True)
        self.assertEqual(c.holdings.tolist(), ["X|Y"] * 4)
        self.assertAlmostEqual(c.slot1_value.iloc[1], 2/7)
        self.assertAlmostEqual(c.slot1_value.iloc[2], 8/49)
        self.assertAlmostEqual(c.slot2_value.iloc[2], 16/49)
        self.assertAlmostEqual(c.turn.iloc[2], (6+1)/56)
        self.assertEqual(c.turn.iloc[-1], 0)

    def test_stable_incumbents_duplicate_issuer_and_portfolio_cap(self):
        days = ["2026-01-05", "2026-01-06", "2026-01-07"]
        m = Market(days, ["GOOG", "GOOGL", "A", "B", "C", "D"],
                   np.ones((3, 6)), np.ones((3, 6)), [1]*3, ["tech"]*4 + ["other"]*2)
        ranks = {"D": [[0], [5], [5]], "E": [[1, 2, 3, 4, 5], [4, 2, 1, 3, 5], [1, 3, 5]],
                 "empty": [[]]*3}
        c = replay(m, ranks, slots_for(("D", "monthly"), ("E", "daily"), ("E", "daily")),
                   sector_cap=2, detail=True)
        self.assertEqual(c.holdings.tolist(), ["GOOG|A|C", "GOOG|A|C", "GOOG|B|D"])
        self.assertAlmostEqual(c.turn.iloc[1], 0)
        self.assertTrue((c.active <= 7).all())

    def test_missing_auction_and_cash_settlement(self):
        days = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        op = [[1, 1], [np.nan, 1], [np.nan, 1], [1, 1]]
        m = Market(days, ["X", "Y"], op, op, [1]*4, ["x", "y"],
                   settlements={(days[2], "X"): 3})
        ranks = {"D": [[0], [1], [0, 1], [0, 1]], "empty": [[]]*4}
        c = replay(m, ranks, slots_for(("D", "daily")), detail=True)
        self.assertEqual(c.slot1.tolist(), ["X", "X", "", "Y"])
        self.assertAlmostEqual(c.turn.iloc[2], 0)
        self.assertAlmostEqual(c.slot1_value.iloc[2], 3/7)
        self.assertAlmostEqual(c.krw10.iloc[2], (6 + 3/1.001)/7)

    def test_cap_three_admits_third_and_replaces_fourth_same_sector_name(self):
        days = ["2026-01-05"]
        names = ["A", "B", "C", "D", "E"]
        prices = np.ones((1, 5))
        m = Market(days, names, prices, prices, [1], ["IT"] * 4 + ["Energy"])
        ranks = {"M": [list(range(5))], "empty": [[]]}
        c = replay(m, ranks, slots_for(*[("M", "monthly")] * 4), sector_cap=3, detail=True)
        self.assertEqual(c.holdings.iloc[0], "A|B|C|E")
        self.assertEqual(c.sectors.iloc[0], 2)
        self.assertAlmostEqual(c.max_sector_weight.iloc[0], 3/7)

    def test_schedule_first_available_and_net_cross_slot_transfer(self):
        days = ["2026-01-05", "2026-01-06", "2026-01-07"]
        m = Market(days, ["X", "Y"], np.ones((3, 2)), np.ones((3, 2)),
                   [1]*3, ["x", "y"])
        ranks = {"D": [[], [0], [1]], "E": [[0], [1], [1]], "empty": [[]]*3}
        c = replay(m, ranks, slots_for(("D", "monthly"), ("E", "daily")), detail=True)
        self.assertEqual(c.slot1.tolist(), ["", "X", "X"])
        self.assertEqual(c.slot2.tolist(), ["X", "Y", "Y"])
        self.assertAlmostEqual(c.turn.iloc[1], 1/7)
        # Two fills in total: opening X, then one incremental unit of Y. With
        # costs, a small residual X order arises because the old slot paid a fee.
        self.assertAlmostEqual(c.krw10.iloc[1], (5 + 2 / 1.001) / 7, places=6)

    def test_extra_gate_reviews_exit_and_reenter_only_the_desk_sleeve(self):
        days = ["2026-01-05", "2026-01-06", "2026-01-07"]
        m = Market(days, ["X", "Y"], np.ones((3, 2)), np.ones((3, 2)), [1]*3, ["x", "y"])
        ranks = {"D": [[0], [], [0]], "E": [[1], [], []], "empty": [[]]*3}
        layout = slots_for(("D", "monthly"), ("E", "monthly"))
        normal = replay(m, ranks, layout, detail=True)
        gated = replay(m, ranks, layout, detail=True, extra_reviews={"D": {1, 2}})
        self.assertEqual(normal.slot1.tolist(), ["X", "X", "X"])
        self.assertEqual(gated.slot1.tolist(), ["X", "", "X"])
        self.assertEqual(gated.slot2.tolist(), ["Y", "Y", "Y"])


if __name__ == "__main__":
    unittest.main()
