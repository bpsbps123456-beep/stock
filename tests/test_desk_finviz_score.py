"""§3.2 30·40·175 자동 채점(desk/finviz_score.py, 2026-11-22 사용자 "자동 채점해요")."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from desk import finviz_score as F


def frame(rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows).T  # {day: {ticker: value}} -> index days, columns tickers


class BasketReturnTest(unittest.TestCase):
    def test_seven_slots_and_cash(self):
        full = [f"T{i}" for i in range(7)]
        opens = dict.fromkeys(full, 10.0)
        closes = {**dict.fromkeys(full, 10.0), "T0": 17.0}
        self.assertAlmostEqual(F.basket_return(full, opens, closes), 0.1)  # one of seven up 70%
        five = full[:5]  # two empty slots stay in dollar cash (0%)
        self.assertAlmostEqual(F.basket_return(five, opens, closes), 0.1)
        self.assertEqual(F.basket_return([], {}, {}), 0.0)

    def test_missing_price_waits(self):
        self.assertIsNone(F.basket_return(["A"], {"A": 10.0}, {}))
        self.assertIsNone(F.basket_return(["A"], {"A": float("nan")}, {"A": 11.0}))
        with self.assertRaises(ValueError):
            F.basket_return([f"T{i}" for i in range(8)], {}, {})


class CalendarTest(unittest.TestCase):
    def test_month_sessions(self):
        nov = F.month_sessions("2026-11")
        self.assertEqual((nov[0], nov[-1]), ("2026-11-02", "2026-11-30"))
        self.assertNotIn("2026-11-26", nov)  # Thanksgiving
        self.assertEqual(F.month_sessions("2026-12")[0], "2026-12-01")
        self.assertEqual(F.FORWARD_FROM, "2026-12")


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for day, mode in (("2026-12-01", "finviz"), ("2026-12-02", "desk")):
            (self.tmp / f"{day}.json").write_text(json.dumps({"mode": mode}), encoding="utf-8")
        self.months = [
            dict(month="2026-11", forward=False, effective="2026-11-02", baskets={"current": ["A"], "sma30_40_175": ["B"]}),
            dict(month="2026-12", forward=True, effective="2026-12-01", baskets={"current": ["A"], "sma30_40_175": ["B"]}),
            dict(month="2027-01", forward=True, effective="2027-01-04", baskets={"current": ["A"], "sma30_40_175": ["B"]}),
        ]
        self.prices = {
            "Open": frame({"2026-11-02": {"A": 10, "B": 10}, "2026-12-01": {"A": 10, "B": 10}, "2027-01-04": {"A": 10, "B": 10}}),
            "Close": frame({"2026-11-30": {"A": 20, "B": 5}, "2026-12-31": {"A": 10.7, "B": 11.4},
                            "2027-01-05": {"A": 10, "B": 10}}),
        }

    def test_reference_month_never_counts(self):
        out = F.score(self.months, self.prices, "2027-01-05", observations=self.tmp)
        nov, dec, jan = out["months"]
        self.assertEqual((nov["status"], nov["forward"]), ("닫힘", False))
        self.assertEqual((dec["status"], jan["status"]), ("닫힘", "진행 중"))
        self.assertEqual(out["closed"], 1)  # only the closed forward month
        self.assertAlmostEqual(out["total"]["current"], 0.07 / 7)       # one name +7%, six slots in cash
        self.assertAlmostEqual(out["total"]["sma30_40_175"], 0.14 / 7)
        self.assertEqual(out["wins"], 1)
        self.assertEqual(out["evacuation_months"], 1)  # 12/1 was a FINVIZ session
        self.assertEqual((dec["evacuation_days"], dec["recorded_days"]), (1, 2))
        self.assertEqual(out["verdict"], f"쌓는 중 1/{F.REVIEW_AFTER}")

    def test_before_forward_months(self):
        out = F.score(self.months[:1], self.prices, "2026-11-20", observations=self.tmp)
        self.assertEqual(out["closed"], 0)
        self.assertIsNone(out["total"]["current"])
        self.assertIn("2026-12 월초 바스켓부터", out["verdict"])

    def test_review_after_three_closed_months_and_price_wait(self):
        months = [dict(m, forward=True) for m in self.months[:2]]
        months.append(dict(month="2026-10", forward=True, effective="2026-10-01", baskets={"current": ["A"], "sma30_40_175": ["B"]}))
        prices = {"Open": pd.concat([self.prices["Open"], frame({"2026-10-01": {"A": 10, "B": 10}})]),
                  "Close": pd.concat([self.prices["Close"], frame({"2026-10-30": {"A": 10, "B": 10}})])}
        out = F.score(months, prices, "2027-01-05", observations=self.tmp)
        self.assertEqual(out["closed"], 3)
        self.assertIn("교체 재검토", out["verdict"])
        gap = F.score([dict(self.months[1], baskets={"current": ["A"], "sma30_40_175": ["Z"]})], self.prices, "2027-01-05",
                      observations=self.tmp)
        self.assertEqual(gap["months"][0]["status"], "가격 대기")
        self.assertEqual(gap["closed"], 0)


class PayloadTest(unittest.TestCase):
    def test_failure_does_not_raise_and_cache_reads_none(self):
        path = Path(tempfile.mkdtemp()) / "score.json"
        with mock.patch.object(F, "load_months", side_effect=RuntimeError("boom")):
            out = F.payload(force=True, path=path)
        self.assertIn("boom", out["error"])
        self.assertFalse(path.exists())
        self.assertIsNone(F.cached(path))


if __name__ == "__main__":
    unittest.main()
