"""BRIEF3 causal windows, publication selection, and incomplete support checks."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

FILE = Path(__file__).resolve().parents[1] / "data/derived/analysis/seibro_bottoms/code/soxs_focus.py"
SPEC = importlib.util.spec_from_file_location("seibro_soxs_focus_tests", FILE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class SoxsFocusTest(unittest.TestCase):
    def flows(self, n=160):
        dates = pd.bdate_range("2025-01-02", periods=n)
        x = pd.DataFrame({"SOXS_buy": np.arange(n) + 100.0, "SOXS_sell": 100.0,
                          "SOXS_net": np.arange(n).astype(float), "cens_SOXS": False,
                          "part_SOXS": False}, index=dates)
        x.index.name = "date"
        return x

    def test_censor_and_partial_missing_invalidate_entire_five_session_window(self):
        x = self.flows()
        x.loc[x.index[125], "cens_SOXS"] = True
        x.loc[x.index[135], "part_SOXS"] = True
        f = M.features(x, x.index)
        for i in (125, 135):
            self.assertTrue(f.gross_usd_L5.iloc[i:i + 5].isna().all())
            self.assertTrue(f.buy_share_L5.iloc[i:i + 5].isna().all())
        i = 145
        self.assertAlmostEqual(f.buy_share_L5.iloc[i], x.SOXS_buy.iloc[i-4:i+1].sum()
                               / (x.SOXS_buy.iloc[i-4:i+1].sum() + x.SOXS_sell.iloc[i-4:i+1].sum()))

    def test_rank_uses_only_2025_onward_and_is_unchanged_by_future(self):
        x = self.flows()
        old = x.iloc[:10].copy()
        old.index = pd.bdate_range("2024-01-02", periods=10)
        old.SOXS_buy = 1e20
        f = M.features(pd.concat([old, x]), old.index.append(x.index))
        self.assertEqual(f.index.min().year, 2025)
        self.assertTrue(f.gross_usd_L1_P.iloc[:119].isna().all())
        y = x.copy()
        y.loc[y.index[150]:, "SOXS_buy"] = 1e15
        g = M.features(y, y.index)
        np.testing.assert_allclose(f.gross_usd_L1_P.iloc[:150], g.gross_usd_L1_P.iloc[:150], equal_nan=True)

    def test_latest_source_is_selected_before_measurement_validity(self):
        x = pd.DataFrame({"entry_settle": pd.to_datetime(["2026-07-06", "2026-07-06"]),
                          "cens_SOXS": [False, True], "P": [99, np.nan]},
                         index=pd.to_datetime(["2026-06-30", "2026-07-01"]))
        x.index.name = "date"
        selected = M.latest_sources(x)
        self.assertEqual(selected.iloc[0].source_date, pd.Timestamp("2026-07-01"))
        self.assertTrue(selected.iloc[0].cens_SOXS)
        self.assertTrue(pd.isna(selected.iloc[0].P))

    def test_state_uses_previous_close_and_twenty_sessions_must_be_complete(self):
        dates = pd.bdate_range("2026-07-01", periods=25)
        p = pd.DataFrame({"SOXX_Open": 100., "SOXX_Close": 100., "dd60": 0.}, index=dates)
        p.loc[dates[0], "dd60"] = -0.15
        p.loc[dates[1], "dd60"] = 0.5
        p.loc[dates[20], "SOXX_Close"] = 96.9
        r = M.reaction(p, dates[1])
        self.assertTrue(r["state"])
        self.assertTrue(r["complete20"])
        self.assertFalse(r["support20"])
        late = M.reaction(p, dates[10])
        self.assertFalse(late["complete20"])
        self.assertTrue(pd.isna(late["support20"]))

    def test_exact_shift_keeps_identity_and_cooldown_counts_market_sessions(self):
        p = M.shift_reference(np.array([True, False, False]), np.array([10., 0., 0.]))
        self.assertAlmostEqual(p["upper_tail"], 1/3)
        self.assertEqual(M.shift_reference([True, True], [1., 2.])["upper_tail"], 1.)
        self.assertEqual(M.cooldown_positions([0, 1, 9, 10, 19, 20]), [0, 10, 20])


if __name__ == "__main__":
    unittest.main()
