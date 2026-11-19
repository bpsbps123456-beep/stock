"""Regression boundaries for screen order sizing, merger settlement and sleeve blending."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

FOLDER = Path(__file__).resolve().parents[1] / "data/derived/analysis/finviz_rotation_20261119"
with patch.object(sys, "path", [str(FOLDER), *sys.path]):
    import compare


class FinvizRotationTests(unittest.TestCase):
    def test_sparse_slots_remain_cash_with_stable_ties(self):
        target = compare.relative_target(["B", "A", "C"], {"A": 1.8, "B": 1.8, "C": np.nan})
        self.assertEqual(list(target), ["A", "B"])
        self.assertAlmostEqual(sum(target.values()), 2/7)

    def test_liquidation_preserves_other_positions_and_blocks_reentry(self):
        o = np.array([[10., 10.], [20., 10.], [20., 15.]])
        w = np.array([.5, .5])
        valid = np.array([[True, True], [False, True], [False, True]])
        targets = compare.close_terminated({0: w, 2: w}, o, valid)
        np.testing.assert_allclose(targets[1], [0, 1/3])
        np.testing.assert_allclose(targets[2], [0, .5])
        values = compare.replay(range(3), o, o, np.ones(3), targets, np.ones(3, bool))
        np.testing.assert_allclose(values.usd, [1, 1.5, 1.75])

    def test_mixed_sleeves_include_drift_when_only_one_rebalances(self):
        o = np.array([[1., 1.], [3., 1.]])
        a = {0: np.array([.5, .0])}
        b = {0: np.array([0., 1.]), 1: np.array([0., 1.])}
        out = compare.blend(range(2), o, a, b)
        np.testing.assert_allclose(out[1], [.375, .5])
