"""2026-11-21 연구 보강: SOXX 100일선 대피 회차 분해(desk_etf_validation)와 5일저가선 전방 검증의 매도 규칙."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data/derived/analysis"


def load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvacuationSpellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compare = load("desk_etf_validation_compare", ANALYSIS / "desk_etf_validation_20261120/compare.py")
        cls.rows = cls.compare.spells()

    def test_five_spells_with_the_last_still_open(self):
        spans = [(r["start"], r["end"], r["sessions"], r["open"]) for r in self.rows]
        self.assertEqual(spans, [("2026-03-30", "2026-03-31", 2, False), ("2026-07-30", "2026-07-30", 1, False),
                                 ("2026-08-25", "2026-08-27", 3, False), ("2026-08-31", "2026-09-21", 15, False),
                                 ("2026-10-13", "2026-11-18", 27, True)])

    def test_closed_spells_lose_and_the_open_one_carries_the_gain(self):
        closed = [r for r in self.rows if not r["open"]]
        cum = {k: np.prod([1 + r[k] for r in closed]) - 1 for k in self.compare.SPELL_KEYS}
        self.assertAlmostEqual(cum["original/monthly1"], -0.1428, places=4)
        self.assertAlmostEqual(cum["exit"], -0.1985, places=4)
        self.assertAlmostEqual(self.rows[-1]["original/monthly1"], 0.3241, places=4)
        self.assertGreater(self.rows[-1]["exit"], self.rows[-1]["original/monthly1"])   # 열린 구간은 달러 현금이 더 나았다

    def test_metrics_note_marks_the_old_weekly_rule(self):
        self.assertIn("120일 수익률 주간 규칙", self.compare.NOTE)


class LineFiveDayExitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ANALYSIS / "dip_reason/code"))
        cls.fwd = load("box_forward_line5d", ANALYSIS / "box_trade/code/forward_line5d.py")

    def test_target_only_on_regular_bars_before_the_morning_close(self):
        o = np.array([100.0, 100.0, 101.0, 101.0, 99.0])
        h = np.array([100.0, 101.0, 103.0, 104.0, 99.0])
        reg = np.array([True, True, False, True, False])
        x8 = np.array([4, 4, 4, 4, -1])
        # 봉 2는 장외라 목표 102에 닿아도 무시, 봉 3(정규장)에서 max(시가 101, 목표 102) = 102
        self.assertEqual(self.fwd.tp2w(o, h, reg, x8, 0, 100.0, True), 102.0)
        # 11:00 뒤 체결이면 목표 없이 08시 봉 시가
        self.assertEqual(self.fwd.tp2w(o, h, reg, x8, 0, 100.0, False), 99.0)
        # 08시 봉이 없으면 거래 미완
        self.assertTrue(np.isnan(self.fwd.tp2w(o, h, reg, x8, 4, 99.0, True)))


if __name__ == "__main__":
    unittest.main()
