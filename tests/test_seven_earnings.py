"""Standalone quarter and earnings-only eligibility regression examples."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

FOLDER = Path(__file__).resolve().parents[1] / "data/derived/analysis/seven_sleeves_20261120"
with patch.object(sys, "path", [str(FOLDER), *sys.path]):
    import earnings as E


class FiscalQuarterTests(unittest.TestCase):
    def test_ytd_difference_requires_identical_fiscal_start(self):
        table = E.quarter_table({
            (E.ordinal("2025-01-01"), E.ordinal("2025-09-30")): (70., "a"),
            (E.ordinal("2025-01-01"), E.ordinal("2025-12-31")): (100., "b"),
            (E.ordinal("2024-12-31"), E.ordinal("2025-09-30")): (99., "x"),
        })
        q = table[E.ordinal("2025-12-31")]
        self.assertEqual((q.start, q.end, q.value, q.method, q.source),
                         (E.ordinal("2025-10-01"), E.ordinal("2025-12-31"), 30., "ytd_difference", "b/a"))

    def test_profitable_gate_keeps_raw_growth_available(self):
        periods = [("2024-07-01", "2024-09-30", 100.), ("2024-10-01", "2024-12-31", 100.),
                   ("2025-07-01", "2025-09-30", 120.), ("2025-10-01", "2025-12-31", 130.)]
        state = {E.SALES[0]: {(E.ordinal(s), E.ordinal(e)): (v, "a") for s, e, v in periods}}
        end = E.ordinal("2025-12-31")
        missing = E.income_signal(state, end)
        self.assertEqual(missing["status"], "unavailable_operating_income")
        self.assertAlmostEqual(missing["raw_score"], .35)
        state[E.OPERATING] = {(E.ordinal("2025-10-01"), end): (-1., "b")}
        self.assertEqual(E.income_signal(state, end)["status"], "nonpositive_operating_income")
        state[E.OPERATING] = {(E.ordinal("2025-10-01"), end): (1., "b")}
        self.assertAlmostEqual(E.income_signal(state, end)["score"], .35)
