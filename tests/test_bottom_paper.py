import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from desk.bottom_paper import latest_completed, record_and_score


class PaperTests(unittest.TestCase):
    def test_first_issue_wins_and_scores_across_split(self):
        forecast = dict(available=True, as_of="2026-11-16", close=200, horizon=2,
                        center=190, lower=180, upper=195, method="test")
        close = pd.Series([100, 96, 92], index=pd.to_datetime(["2026-11-16", "2026-11-17", "2026-11-18"]))
        now = datetime.fromisoformat("2026-11-16T22:00:00+00:00")
        end = datetime.fromisoformat("2026-11-18T22:00:00+00:00")
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            record_and_score(forecast, close, now, folder)
            stats = record_and_score(dict(forecast, lower=199), close, now, folder)
            self.assertEqual(stats["pending"], 1)
            row = json.loads((folder / "2026-11-16.json").read_text(encoding="utf-8"))
            self.assertEqual(row["lower"], 180)
            self.assertEqual(record_and_score(forecast, close, end, folder)["hits"], 1)
            self.assertEqual(record_and_score(forecast, close.iloc[::2], end, folder)["unscored"], 1)

    def test_no_retroactive_issue_and_early_close_calendar(self):
        with tempfile.TemporaryDirectory() as temp:
            record_and_score({"available": True, "as_of": "2026-11-17"}, pd.Series(dtype=float),
                             datetime.fromisoformat("2026-11-18T15:00:00+00:00"), Path(temp))
            self.assertEqual(list(Path(temp).iterdir()), [])
        self.assertEqual(latest_completed(datetime.fromisoformat("2026-11-27T18:01:00+00:00")), "2026-11-27")
