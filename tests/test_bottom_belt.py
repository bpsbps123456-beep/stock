"""바닥 벨트 2차 탐색(bottom_zone_20261111/belt)의 표시·시점 계약."""
import importlib.util
import json
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "data/derived/analysis/bottom_zone_20261111/belt/code"
EVENTS = ROOT / "data/derived/analysis/seibro_bottoms/derived/events.json"


def load_module():
    spec = importlib.util.spec_from_file_location("belt_search_under_test", CODE / "belt_search.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BottomBeltTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bs = load_module()
        cls.panel, cls.wide, cls.cnn, cls.news = cls.bs.load()
        cls.atoms = cls.bs.build_atoms(cls.panel, cls.wide, cls.cnn, cls.news)
        cls.lab = cls.bs.labels(cls.panel, cls.wide)

    def test_search_space_is_the_registered_one(self):
        self.assertEqual(self.atoms.shape[1], 49)
        self.assertEqual(sum(1 for _ in self.bs.combos(49)), 19649)

    def test_swing_lows_are_b10_plus_november(self):
        b10 = json.loads(EVENTS.read_text(encoding="utf-8"))["B10"]
        want = [d for d in b10 if d >= "2024"] + ["2026-11-05"]
        got = sorted({a for a in self.lab["anchor"] if a >= "2024"})
        self.assertEqual(got, want)
        for day in want:                       # 바닥 날 자체는 벨트 안이어야 한다
            self.assertEqual(self.lab.loc[day, "belt10"], 1.0, day)

    def test_future_values_do_not_change_past_conditions(self):
        cut = pd.Timestamp("2026-06-30")
        panel = self.panel.copy()
        numeric = panel.select_dtypes("number").columns
        panel.loc[panel.index > cut, numeric] = panel.loc[panel.index > cut, numeric] * 1.7 + 3
        wide = {t: g.copy() for t, g in self.wide.items()}
        for g in wide.values():
            cols = g.select_dtypes("number").columns
            g.loc[g.index > cut, cols] = g.loc[g.index > cut, cols] * 1.9
        cnn = json.loads(json.dumps(self.cnn))
        limit = cut.value // 10**6 + 86_400_000
        for block in cnn.values():
            if isinstance(block, dict) and "data" in block:
                for row in block["data"]:
                    if row["x"] > limit:
                        row["y"] = row["y"] * 2 + 5
        news = self.news.copy()
        news.loc[news["date"] > cut, "family"] = "DN"
        news.loc[news["date"] > cut, "house"] = [f"h{i}" for i in range(int((news["date"] > cut).sum()))]
        changed = self.bs.build_atoms(panel, wide, cnn, news)
        pd.testing.assert_frame_equal(self.atoms.loc[:cut], changed.loc[:cut])

    def test_bb1_registry_values(self):
        spec = importlib.util.spec_from_file_location("belt_now_under_test", CODE / "now_update.py")
        now = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(now)
        self.assertEqual(now.MAIN, ("낙폭20", "SOXL비중상위20"))
        self.assertEqual([now.bucket(n) for n in (0, 1, 2, 3, 5, 6, 10, 11, 54)],
                         ["0", "1-2", "1-2", "3-5", "3-5", "6-10", "6-10", "11+", "11+"])
        bb1 = now.history_rates()["BB-1"]
        self.assertEqual((bb1["days"], round(bb1["belt10"] * bb1["days"]), bb1["lows"]), (36, 31, 6))

    def test_frozen_collectors_keep_their_dates(self):
        zone = CODE.parents[1] / "code"
        self.assertIn("def market(end='2026-11-12'):", (zone / "collect_A.py").read_text(encoding="utf-8"))
        self.assertIn("def raw_panel(source_end='2026-11-11'):", (zone / "panel_A.py").read_text(encoding="utf-8"))

    def test_closest_combination_counts(self):
        full = pd.concat([self.panel[["down"]], self.lab, self.atoms], axis=1)
        on = full[(full.index >= "2024-01-02") & (full["down"] == True)  # noqa: E712
                  & full["belt10"].notna() & full["낙폭20"] & full["SOXL비중상위20"]]
        self.assertEqual((len(on), int(on["belt10"].sum()), on["anchor"].nunique()), (36, 31, 6))


if __name__ == "__main__":
    unittest.main()
