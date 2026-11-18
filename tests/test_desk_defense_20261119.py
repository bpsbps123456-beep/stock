"""Execution boundaries for the three-panel research replay."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parents[1] / "data/derived/analysis/desk_defense_20261119"
with patch.object(sys, "path", [str(FOLDER), *sys.path]):
    import study


class DeskDefenseTests(unittest.TestCase):
    def test_exit_keeps_overnight_gap_and_reenters_after_recovery_gap(self):
        days = ["2026-07-06", "2026-07-07", "2026-07-08"]
        o = np.array([[100.], [70.], [90.]])
        c = np.array([[110.], [90.], [99.]])
        result = study.replay(days, o, c, np.ones(3), {0: np.ones(1)}, [True, False, True])
        np.testing.assert_allclose(result.usd, [1.10, .70, .77])
        np.testing.assert_allclose(result.turn, [1., 1., 1.])
        np.testing.assert_allclose(result.krw10, result.usd * (.999 ** np.arange(1, 4)))

    def test_third_recovery_close_only_trades_at_following_open(self):
        days = list(pd.bdate_range("2025-01-01", periods=116).strftime("%Y-%m-%d"))
        prices = np.ones(116)*100
        prices[-6:] = [50, 120, 120, 120, 120, 120]
        book = SimpleNamespace(cls=pd.DataFrame({"QQQ": prices, "SOXX": prices}, index=days))
        gates = study.market_gates(book, days[-6:])
        np.testing.assert_equal(gates["SOXX20r3"], [True, False, False, False, True, True])
        np.testing.assert_equal(gates["SOXX20r1"], [True, False, True, True, True, True])

    def test_interweek_switch_receives_current_drifted_basket(self):
        base = {0: np.array([.5, 0.])}
        alternate = {0: np.array([0., .5])}
        prices = np.array([[1., 1.], [2., 4.], [1., 1.]])
        switch = study.switched_targets(range(3), prices, base, alternate, [True, False, True])
        np.testing.assert_allclose(switch[1], [0., .8])
        np.testing.assert_allclose(switch[2], [.5, 0.])
