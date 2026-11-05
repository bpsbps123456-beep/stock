"""Source/publication separation, masked money and unfinished reactions."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

FILE = Path(__file__).resolve().parents[1] / "data/derived/analysis/seibro_bottoms/code/soxs_ledger.py"
SPEC = importlib.util.spec_from_file_location("seibro_soxs_ledger_tests", FILE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class SoxsLedgerTest(unittest.TestCase):
    def fixture(self, n=135):
        sessions = M.build.nyse_sessions()
        dates = sessions[sessions >= "2023-01-03"][:n + 8]
        md = pd.DataFrame({"SOXS_buy": np.arange(n) + 100., "SOXS_sell": 100.,
                           "SOXS_net": np.arange(n).astype(float), "SOXS_tot": np.arange(n) + 200.,
                           "cens_SOXS": False, "part_SOXS": False, "unit_id": np.arange(n),
                           "unit_n_trades": 1}, index=dates[:n])
        for col in ("unit_first", "unit_last", "seibro_date"):
            md[col] = dates[2:n + 2]
        md["known_settle"], md["entry_settle"] = dates[3:n + 3], dates[3:n + 3]
        md.index.name = "date"
        pf = pd.DataFrame({"SOXX_Open": 100., "SOXX_Close": 100., "dd60": -0.12,
                           "SOXX_daily_return_pct": 0.}, index=dates)
        return md, pf, sessions

    def test_censored_money_and_unpublished_money_never_become_actual_zero(self):
        md, pf, sessions = self.fixture()
        date = md.index[125]
        md.loc[date, ["SOXS_buy", "SOXS_sell", "SOXS_net", "SOXS_tot"]] = 0.
        md.loc[date, "cens_SOXS"] = True
        actual_zero = md.index[124]
        md.loc[actual_zero, "SOXS_buy"] = 0.
        feat = M.focus.features(md, sessions, rank_start="2023-01-03")
        source = M.source_ledger(md, feat, pf, sessions)
        self.assertTrue(pd.isna(source.loc[date, "SOXS_buy"]))
        self.assertTrue(source.loc[date, "source_observed"])
        self.assertEqual(source.loc[date, "gross_usd_L1_status"], "UNKNOWN")
        self.assertEqual(source.loc[actual_zero, "SOXS_buy"], 0.)
        self.assertEqual(source.loc[actual_zero, "gross_usd_L1_status"], "OFF")
        last = source.iloc[-1]
        self.assertFalse(last.source_observed)
        self.assertTrue(pd.isna(last.SOXS_buy))
        self.assertEqual(last.gross_usd_L1_status, "UNPUBLISHED")
        self.assertTrue(pd.isna(last.entry_previous_close))
        self.assertTrue(pd.isna(last.entry_previous_dd60_pct))

    def test_missing_day_invalidates_entire_five_session_window(self):
        md, pf, sessions = self.fixture()
        md.loc[md.index[125], "part_SOXS"] = True
        feat = M.focus.features(md, sessions, rank_start="2023-01-03")
        source = M.source_ledger(md, feat, pf, sessions)
        self.assertTrue(source.loc[md.index[125:130], "gross_usd_L5"].isna().all())
        self.assertTrue(source.loc[md.index[125:130], "gross_usd_L5_status"].eq("UNKNOWN").all())
        self.assertTrue(np.isfinite(source.loc[md.index[130], "gross_usd_L5"]))

    def test_latest_published_off_source_does_not_restore_older_on(self):
        md, pf, sessions = self.fixture()
        older, newer = md.index[125:127]
        entry_date = md.loc[newer, "entry_settle"]
        md.loc[older, ["entry_settle", "known_settle"]] = entry_date
        feat = M.focus.features(md, sessions, rank_start="2023-01-03")
        for key in M.KEYS:
            feat.loc[older, key + "_P"] = 99.
            feat.loc[newer, key + "_P"] = 10.
        source = M.source_ledger(md, feat, pf, sessions)
        self.assertEqual(source.loc[older, "gross_usd_L1_status"], "ON")
        self.assertFalse(source.loc[older, "selected_for_entry"])
        self.assertTrue(source.loc[newer, "selected_for_entry"])
        entry = M.entry_ledger(md, feat, pf, sessions, start=entry_date, end=entry_date)
        self.assertEqual(entry.iloc[0].source_date, newer)
        self.assertEqual(entry.iloc[0].gross_usd_L1_status, "OFF")

    def test_incomplete_future_prices_leave_reactions_and_excursions_missing(self):
        md, pf, sessions = self.fixture()
        feat = M.focus.features(md, sessions, rank_start="2023-01-03")
        entry_date = md.iloc[-1].entry_settle
        truncated = pf.loc[:entry_date]
        entry = M.entry_ledger(md, feat, truncated, sessions, start=entry_date, end=entry_date)
        for h in (5, 10, 20):
            self.assertFalse(entry.iloc[0][f"R{h}_complete"])
            for col in (f"R{h}_pct", f"MAE{h}_pct", f"MFE{h}_pct"):
                self.assertTrue(pd.isna(entry.iloc[0][col]))


if __name__ == "__main__":
    unittest.main()
