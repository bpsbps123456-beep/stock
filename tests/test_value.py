"""저평가 표시(desk/value.py, docs/desk_system.md §8.4): 분야는 '전망 대비 주가'의 30일 변화를 AI 전체와
비교한 하위 1/3 중 전망↑·PER↓, 회사는 그런 분야가 대표 분야이고 자기 전망↑·PER↓. 네트워크 없이 잰다."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from desk import tour, value


def row(R, P, analysts=10):
    return {"analysts": analysts, "R": R, "P": P, "V": (1 + P) / (1 + R) - 1}


class MeasureTest(unittest.TestCase):
    def test_v_is_the_forward_pe_change(self):
        r = value.measure({"+1y": {"current": 12.0, "30daysAgo": 10.0}, "analysts": 5}, 110.0, 100.0)
        self.assertAlmostEqual(r["R"], 0.2)
        self.assertAlmostEqual(r["P"], 0.1)
        self.assertAlmostEqual(r["V"], 1.1 / 1.2 - 1)

    def test_losses_are_not_judged_by_pe(self):
        r = value.measure({"+1y": {"current": -1.0, "30daysAgo": 0.5}, "analysts": 9}, 10.0, 9.0)
        self.assertTrue(r["loss"])
        self.assertNotIn("V", r)
        self.assertFalse(value.usable(r))

    def test_missing_estimates_leave_no_v(self):
        r = value.measure({}, 10.0, 9.0)
        self.assertNotIn("V", r)
        self.assertFalse(value.usable(r))


class JudgeTest(unittest.TestCase):
    def setUp(self):
        # six zones of three companies; zone k's companies all have price change P_k and estimate change R_k
        spec = {"a": (0.05, -0.10), "b": (-0.02, -0.08), "c": (0.03, 0.00), "d": (0.02, 0.10), "e": (0.01, 0.12), "f": (0.0, 0.20)}
        self.rows, self.zones, self.main = {}, {}, {}
        for z, (R, P) in spec.items():
            self.zones[z] = [f"{z}{i}" for i in range(3)]
            for t in self.zones[z]:
                self.rows[t] = row(R, P)
                self.main[t] = z

    def test_bottom_third_with_rising_estimates_and_falling_pe(self):
        out = value.judge(self.rows, self.zones, self.main)
        flagged = sorted(z for z, r in out["zones"].items() if r["flag"])
        # a and b are the two cheapest of six; b's estimates fell, so only a qualifies
        self.assertEqual(flagged, ["a"])
        self.assertEqual(out["zones"]["a"]["rank"], 1)
        self.assertEqual(out["zones"]["a"]["of"], 6)

    def test_a_zone_whose_pe_rose_is_not_called_cheap(self):
        rows = {t: row(0.05, 0.10) for t in self.zones["a"]}          # zone a: PER +4.8%
        rows.update({t: row(0.0, 0.30) for z in "bcdef" for t in self.zones[z]})
        out = value.judge(rows, self.zones, self.main)
        self.assertEqual(out["zones"]["a"]["rank"], 1)
        self.assertFalse(out["zones"]["a"]["flag"])

    def test_companies_need_their_own_rising_estimates_and_falling_pe(self):
        self.rows["a0"] = row(-0.01, -0.10)     # estimates cut
        self.rows["a1"] = row(0.05, 0.08)       # PER rose
        out = value.judge(self.rows, self.zones, self.main)
        c = out["companies"]
        self.assertTrue(c["a2"]["flag"])
        self.assertFalse(c["a0"]["flag"])
        self.assertFalse(c["a1"]["flag"])
        self.assertFalse(c["b0"]["flag"])       # cheap zone but estimates fell → zone not flagged

    def test_only_the_main_zone_counts(self):
        self.zones["a"].append("x")
        self.rows["x"] = row(0.05, -0.10)
        self.main["x"] = "f"                    # listed in a, but its representative zone is f
        out = value.judge(self.rows, self.zones, self.main)
        self.assertFalse(out["companies"]["x"]["flag"])

    def test_thin_coverage_and_small_zones(self):
        self.rows["a0"] = row(0.05, -0.10, analysts=2)
        out = value.judge(self.rows, self.zones, self.main)
        self.assertEqual(out["companies"]["a0"]["skip"], "thin")
        self.assertFalse(out["companies"]["a0"]["flag"])
        self.assertNotIn("a", out["zones"])     # two measurable companies left: no zone verdict

    def test_memory_zones_carry_the_cycle_note(self):
        self.zones["hbm"] = self.zones.pop("a")
        for t in self.zones["hbm"]:
            self.main[t] = "hbm"
        out = value.judge(self.rows, self.zones, self.main)
        self.assertTrue(out["zones"]["hbm"]["cycle"])
        self.assertTrue(out["zones"]["hbm"]["flag"])
        self.assertFalse(out["zones"]["c"]["cycle"])


class MapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = tour.load_map()
        cls.sym = value.map_symbols(cls.m)

    def test_symbols_for_every_listed_company(self):
        self.assertEqual(self.sym["005930"], ["005930.KS"])             # universe symbol as is
        self.assertEqual(self.sym["NEE"], ["NEE"])                       # US outside the universe
        self.assertEqual(self.sym["700"], ["0700.HK"])
        self.assertEqual(self.sym["006400"], ["006400.KS", "006400.KQ"])
        private = [t for t, c in self.m["companies"].items() if c["market"] in ("private", "fund")]
        self.assertFalse(set(private) & set(self.sym))

    def test_main_zone_matches_the_page(self):
        zones, main = value.zone_index(self.m)
        self.assertNotIn("axes", zones)
        for st, z in tour.zones(self.m):
            for mem in z.get("members", []):
                if mem.get("main"):
                    self.assertEqual(main[mem["t"]], z["id"], mem["t"])


class PayloadTest(unittest.TestCase):
    def test_rules_and_check_come_from_the_code_not_the_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "value.json"
            cache.write_text(json.dumps({"_epoch": time.time(), "zones": {}, "companies": {}, "universe": {},
                                         "check": {"old": True}}), encoding="utf-8")
            with mock.patch.object(value, "CACHE_PATH", cache), mock.patch.object(value, "compute") as compute:
                out = value.payload()
            compute.assert_not_called()
        self.assertEqual(out["check"], value.CHECK)
        self.assertEqual(out["cycle_zones"], sorted(value.CYCLE_ZONES))


if __name__ == "__main__":
    unittest.main()
