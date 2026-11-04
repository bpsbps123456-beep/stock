"""BRIEF4 warmup, full reaction windows, composition, and legacy regression."""
import importlib.util
import io
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "data/derived/analysis/seibro_bottoms"
SPEC = importlib.util.spec_from_file_location("seibro_soxs_multiyear_tests", STUDY / "code/soxs_multiyear.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class SoxsMultiyearTest(unittest.TestCase):
    def test_rank_warmup_starts_in_2023_and_remains_causal(self):
        dates = pd.bdate_range("2023-01-03", periods=300)
        md = pd.DataFrame({"SOXS_buy": np.arange(300) + 10., "SOXS_sell": 10.,
                           "SOXS_net": np.arange(300), "cens_SOXS": False,
                           "part_SOXS": False}, index=dates)
        old = md.iloc[:20].copy()
        old.index = pd.bdate_range("2022-01-03", periods=20)
        old.SOXS_buy = 1e20
        f = M.COMMON.features(pd.concat([old, md]), old.index.append(dates), rank_start=M.RANK_START)
        self.assertEqual(f.index.min(), pd.Timestamp("2023-01-03"))
        self.assertTrue(f.gross_usd_L1_P.iloc[:119].isna().all())
        self.assertEqual(f.gross_usd_L1_rank_n.iloc[250], 250)
        changed = md.copy()
        changed.loc[dates[270]:, "SOXS_buy"] = 1e30
        g = M.COMMON.features(changed, dates, rank_start=M.RANK_START)
        np.testing.assert_allclose(f.gross_usd_L1_P.iloc[:270], g.gross_usd_L1_P.iloc[:270], equal_nan=True)

    def test_missing_interior_close_invalidates_only_affected_horizons(self):
        dates = pd.bdate_range("2024-12-02", periods=25)
        pf = pd.DataFrame({"SOXX_Open": 100., "SOXX_Close": 105., "dd60": -.12}, index=dates)
        pf.loc[dates[8], "SOXX_Close"] = np.nan
        r = M.COMMON.reaction(pf, dates[1])
        self.assertTrue(r["R5_complete"])
        self.assertAlmostEqual(r["R5_pct"], 5.)
        self.assertEqual(r["MAE5_pct"], 0.)
        for h in (10, 20):
            self.assertFalse(r[f"R{h}_complete"])
            self.assertTrue(np.isnan(r[f"R{h}_pct"]))
            self.assertTrue(np.isnan(r[f"MAE{h}_pct"]))
        pf.loc[dates[8], "SOXX_Close"] = 90.
        pf.loc[dates[19], "SOXX_Close"] = 120.
        pf.loc[dates[20], "SOXX_Close"] = 110.
        r = M.COMMON.reaction(pf, dates[1])
        self.assertTrue(r["R20_complete"])
        self.assertAlmostEqual(r["R20_pct"], 10.)
        self.assertAlmostEqual(r["MAE20_pct"], -10.)
        self.assertAlmostEqual(r["MFE20_pct"], 20.)
        pf.SOXX_Close = 90.
        self.assertEqual(M.COMMON.reaction(pf, dates[1])["MFE20_pct"], 0.)

    def test_event_cooldowns_continue_across_year_boundary(self):
        dates = pd.bdate_range("2024-12-16", periods=35)
        base = pd.DataFrame({"path_id": "gross_usd_L1", "measure": "gross_usd",
                             "delay_sessions": 0, "state": True, "value_valid": True,
                             "rank_high": True, "entry_index": np.arange(35), "entry": dates})
        panel = M.context_panel(base, "allprice")
        for h in M.HORIZONS:
            self.assertEqual(panel.loc[panel[f"event{h}"], "entry_index"].tolist(), list(range(0, 35, h)))
        first_2025 = panel[panel.entry.dt.year.eq(2025)].iloc[0]
        self.assertFalse(first_2025.event10)

    def test_year_weights_and_invalid_year_exclusion_match_mean_and_win_tests(self):
        x = pd.DataFrame({"entry": pd.to_datetime(["2024-01-02"] * 4 + ["2025-01-02"] * 2
                                                   + ["2026-01-02"] * 3),
                          "on": [True, False, False, False, True, True, True, True, False],
                          "R10_pct": [10., 0., 0., -5., 50., -50., -1., 1., 10.],
                          "eligible": True, "evaluated_entry": True, "R10_complete": True})
        for invalid_on in (True, False):
            with self.subTest(invalid_on=invalid_on):
                x.loc[x.entry.dt.year.eq(2025), "on"] = invalid_on
                matched = M.year_matched_comparison(x, 10)
                tests = M.primary_shift_tests(x, draws=73, seed=17)
                self.assertEqual(matched["n_on_included"], 3)
                self.assertEqual(matched["excluded_years"][0]["reason"], "no_off" if invalid_on else "no_on")
                self.assertAlmostEqual(matched["on_minus_off_mean_pct"], -25 / 9)
                self.assertAlmostEqual(matched["on_minus_off_positive_rate_pp"], 0.)
                for test, value in zip(tests, (matched["on_minus_off_mean_pct"],
                                              matched["on_minus_off_positive_rate_pp"])):
                    self.assertAlmostEqual(test["observed"], value)
                    self.assertEqual([y["year"] for y in test["included_years"]], [2024, 2026])
                    self.assertEqual(test["n_on_excluded"], 2 if invalid_on else 0)
                    self.assertEqual(test["n_off_excluded"], 0 if invalid_on else 2)
                    self.assertAlmostEqual(test["upper_tail"], (1 + test["exceedances"]) / 74)
                self.assertEqual(tests, M.primary_shift_tests(x, draws=73, seed=17))

    def test_all_existing_2026_daily_columns_reproduce_without_writing_outputs(self):
        import exchange_calendars as xc
        md = pd.read_parquet(STUDY / "derived/measures_daily.parquet").set_index("date").sort_index()
        md = md.loc["2025-01-01":]
        pf = pd.read_parquet(STUDY / "derived/price_feats.parquet").set_index("date").sort_index().loc[:M.END]
        sessions = pd.DatetimeIndex(xc.get_calendar("XNYS", start="2024-12-01", end="2027-01-01").sessions).tz_localize(None)
        feat = M.COMMON.features(md, sessions[sessions <= md.index.max()])
        configs = [(measure, length, 0) for measure in M.COMMON.MEASURES for length in (1, 5)]
        configs += [("gross_usd", 5, 1)]
        current = pd.concat([M.COMMON.attach_events(M.COMMON.path_panel(md, feat, pf, sessions, *c))
                             for c in configs], ignore_index=True)
        original = pd.read_csv(STUDY / "results/soxs_focus/daily.csv")
        # CSV roundtrip normalizes legacy dates, booleans, optional columns and NA.
        actual = pd.read_csv(io.StringIO(current[original.columns].to_csv(index=False)))
        pd.testing.assert_frame_equal(actual, original, check_dtype=False, check_exact=False,
                                      atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
