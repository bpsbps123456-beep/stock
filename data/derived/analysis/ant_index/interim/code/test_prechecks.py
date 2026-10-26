"""가격 전 점검표(prechecks*.py) 점검. 실행: py -m unittest test_prechecks (이 폴더에서)."""
from __future__ import annotations

import json
import math
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import prechecks as P
import prechecks_price as PP

RESULTS = Path(__file__).resolve().parents[1] / "results"


class TestNEff(unittest.TestCase):
    def test_matches_doc_mde_table(self):
        # HYPOTHESES.md §2.7: n 96, 1일 약 95 / 일 z(ρ 0.5) 5·10·21일 약 41·33·27 / ρ 0.95·10일 약 10
        self.assertAlmostEqual(P.n_eff(96, 1, 0.5), 95, delta=0.5)
        self.assertAlmostEqual(P.n_eff(96, 5, 0.5), 41, delta=1)
        self.assertAlmostEqual(P.n_eff(96, 10, 0.5), 33, delta=1)
        self.assertAlmostEqual(P.n_eff(96, 21, 0.5), 27, delta=1)
        self.assertAlmostEqual(P.n_eff(96, 10, 0.95), 10, delta=1)
        self.assertAlmostEqual(2.49 / math.sqrt(P.n_eff(96, 1, 0.0)), 0.26, delta=0.01)

    def test_edge(self):
        self.assertEqual(P.n_eff(3, 5, 0.5), 0.0)
        self.assertEqual(P.n_eff(50, 1, float("nan")), 49)
        self.assertEqual(P.n_eff(50, 1, -0.4), 49)          # 음의 ρ는 0으로 자른다(보수)


class TestBlocks(unittest.TestCase):
    def test_greedy(self):
        self.assertEqual(P.greedy_blocks(range(10), 1), 10)
        self.assertEqual(P.greedy_blocks(range(10), 2), 5)
        self.assertEqual(P.greedy_blocks([0, 1, 5, 6, 11], 5), 3)
        # 뒤 구간 98거래일, 10일 목표: 엄한 간격 11 → 8묶음
        self.assertEqual(P.greedy_blocks(range(98 - 10), 11), 8)


class TestAC1(unittest.TestCase):
    def test_gaps_only_adjacent_pairs(self):
        x = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9], float)
        r, n = P.ac1(x)
        self.assertEqual(n, 6)
        self.assertGreater(r, 0.99)

    def test_too_few(self):
        r, n = P.ac1(np.array([1, np.nan, 2, np.nan, 3], float))
        self.assertTrue(r != r)


class TestM9(unittest.TestCase):
    def test_sign_and_exclusion(self):
        self.assertEqual(P.m9_claims("계좌 -35% 손절했어요"), [(-1, 35.0, "%")])
        self.assertEqual(P.m9_claims("금리 3% 올랐"), [])                  # 제외어
        self.assertEqual(P.m9_claims("오늘 120% 날씨"), [])                 # 맥락어 없음
        self.assertEqual(P.m9_claims("익절 2배 성공"), [(1, 2.0, "배")])
        self.assertEqual(P.m9_claims("수익률 30%"), [])                    # '수익률'만으로는 부호 없음

    def test_first_person(self):
        self.assertTrue(P.M10.search("전 그냥 버팁니다"))
        self.assertFalse(P.M10.search("전날 올랐다"))


class TestPriceFirewall(unittest.TestCase):
    def test_last_day(self):
        self.assertEqual(PP.LAST_DAY, date(2026, 10, 14))
        j = json.loads((RESULTS / "prechecks_price.json").read_text("utf-8"))
        for d in j["last_dates"].values():
            self.assertLessEqual(date.fromisoformat(d), PP.LAST_DAY)

    def test_no_forward_fields(self):
        txt = (RESULTS / "prechecks_price.json").read_text("utf-8").lower()
        for bad in ("fwd", "forward", "mdd", "\"r5\"", "\"r10\"", "\"r21\"", "open", "high", "low"):
            self.assertNotIn(bad, txt)

    def test_crash_starts_synthetic(self):
        idx = [date(2026, 1, 1 + i) for i in range(20)]
        v = [100, 101, 102, 103, 104, 100, 96, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106]
        s = pd.Series(v, index=idx, dtype=float)
        st = PP.crash_starts(s)
        self.assertEqual(st, [idx[5]])          # 5일 고점(104, idx 4) 다음 날

    def test_cluster(self):
        cal = [date(2026, 1, 1 + i) for i in range(30)]
        days = [cal[0], cal[3], cal[8], cal[20]]
        self.assertEqual(PP.cluster_events(days, cal), [cal[0], cal[20]])   # 0→3→8은 5일 안으로 이어짐


class TestOutputs(unittest.TestCase):
    def test_summary(self):
        o = json.loads((RESULTS / "prechecks_summary.json").read_text("utf-8"))
        self.assertTrue(all(o["frozen_ok"].values()))
        self.assertTrue(all(o["lex_v02_ok"].values()))
        ids = {r["id"] for r in o["features"]}
        for h in ("H03", "H06", "H10", "H31", "H35", "H46", "H49", "H51"):
            self.assertIn(h, ids)
        for r in o["features"]:
            self.assertGreaterEqual(r["miss_all"], 0)
            self.assertLessEqual(r["miss_all"], 1)

    def test_md_written(self):
        md = (RESULTS / "prechecks.md").read_text("utf-8")
        for sec in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.", "## 8.", "## 9.", "## 10.", "## 11."):
            self.assertIn(sec, md)


if __name__ == "__main__":
    unittest.main()
