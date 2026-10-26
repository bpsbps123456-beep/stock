"""feat_activity.py 점검: 표준화·싸움 판정·재표집 기대값·심야 칸·주 창, 그리고 '마감 뒤 자료를 쓰지 않는다'(실제 스냅샷).

실행: py -m unittest test_feat_activity   (이 폴더에서)
"""
from __future__ import annotations

import sys
import unittest
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feat_activity as F  # noqa: E402

RES = F.RESULTS


class TestStandardize(unittest.TestCase):
    def test_rz8_needs_eight_and_excludes_self(self):
        vals = [1, 2, 3, 4, 5, 6, 7, 8, 100]
        wd = [0] * 9
        ok = [True] * 9
        z = F.rz8(vals, wd, ok, min_base=8)
        self.assertTrue(np.all(np.isnan(z[:8])))           # 직전 8개가 모이기 전에는 결측
        med, mad = 4.5, 2.0
        self.assertAlmostEqual(z[8], min(6.0, (100 - med) / (1.4826 * mad)))
        self.assertEqual(z[8], 6.0)                          # ±6에서 자른다

    def test_rz8_same_weekday_and_holiday_excluded(self):
        vals = [10, 50, 12, 50, 14, 999, 11]
        wd = [0, 1, 0, 1, 0, 0, 0]
        ok = [True, True, True, True, True, False, True]     # 999는 연휴 뒤 창: 기준에서 뺀다
        z = F.rz8(vals, wd, ok, min_base=3)
        self.assertEqual(z[5], 6.0)                          # 자기 값은 계산된다(잘림)
        # 마지막 창 기준 = 요일 0의 [10, 12, 14](999 제외, 요일 1 제외) → (11 − 12) / (1.4826 × 2)
        self.assertAlmostEqual(z[6], (11 - 12) / (1.4826 * 2))
        vals[3] = 0                                          # 다른 요일 값은 영향 없음
        self.assertAlmostEqual(F.rz8(vals, wd, ok, min_base=3)[6], z[6])

    def test_z_trailing_min(self):
        x = np.arange(50, dtype=float)
        z = F.z_trailing(x, [True] * 50, n=60, min_n=40)
        self.assertTrue(np.all(np.isnan(z[:40])))
        self.assertTrue(np.isfinite(z[40]))

    def test_expanding_resid_uses_past_only(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=40)
        y = 2 * x + rng.normal(size=40)
        r1, _ = F.expanding_resid(y, x, [True] * 40, min_n=20)
        y2 = y.copy()
        y2[30:] += 100                                       # 미래 값을 바꿔도
        r2, _ = F.expanding_resid(y2, x, [True] * 40, min_n=20)
        np.testing.assert_allclose(r1[:30], r2[:30])         # 앞 창 잔차는 그대로
        self.assertAlmostEqual(r2[30] - r1[30], 100)         # 30번째 계수도 앞 창으로만 적합(자기 값만 다름)
        self.assertTrue(np.all(np.isnan(r1[:20])))


class TestFight(unittest.TestCase):
    def test_pair_alternation(self):
        self.assertEqual(F.alt_len_pair(list("XYX")), 3)
        self.assertEqual(F.alt_len_pair(list("XXY")), 2)
        self.assertEqual(F.alt_len_pair(list("XYZX")), 3)    # 끼어든 사람이 있어도 쌍판은 셈
        self.assertEqual(F.alt_len_pair(list("XYZ")), 2)
        self.assertEqual(F.alt_len_pair(list("XXYYX")), 3)   # 연속 중복은 합친다
        self.assertEqual(F.alt_len_pair(list("XYXY")), 4)

    def test_adjacent(self):
        self.assertTrue(F.alt_adjacent(list("XYX")))
        self.assertFalse(F.alt_adjacent(list("XYZX")))
        self.assertFalse(F.alt_adjacent(list("XXY")))

    def test_fight_by_post_threads_separate(self):
        cm = pd.DataFrame({"no": [1] * 6, "root_cid": [0, 0, 3, 3, 3, 5], "cid": range(6),
                           "dt_kst": pd.date_range("2026-08-01", periods=6, freq="min"),
                           "a": ["X", "Y", "X", "Z", "W", "X"]})
        fb = F.fight_by_post(cm)                             # 줄기 0: X,Y / 줄기 3: X,Z,W → 싸움 없음
        self.assertFalse(bool(fb["fight"].iloc[0]) if len(fb) else False)
        cm.loc[4, "a"] = "X"                                 # 줄기 3: X,Z,X → 싸움
        fb = F.fight_by_post(cm)
        self.assertTrue(bool(fb.set_index("no").loc[1, "fight"]))


class TestResample(unittest.TestCase):
    def test_expected_unique_matches_simulation(self):
        counts = np.array([10, 5, 3, 1, 1, 1, 1])
        pool = np.repeat(np.arange(len(counts)), counts)
        rng = np.random.default_rng(1)
        n = 6
        sim = np.mean([len(set(rng.choice(pool, n, replace=False))) for _ in range(20000)])
        self.assertAlmostEqual(F.expected_unique(counts, n), sim, delta=0.03)
        self.assertAlmostEqual(F.expected_unique(counts, 1), 1.0, places=9)
        self.assertEqual(F.expected_unique(counts, 100), float(len(counts)))


class TestCalendarBits(unittest.TestCase):
    def test_night_blocks_dst(self):
        nb = F.night_blocks(datetime(2026, 9, 14, 21, 30), datetime(2026, 9, 15, 21, 30), {date(2026, 9, 14)})
        self.assertEqual([(s.hour, e.hour, ss) for s, e, ss in nb], [(1, 5, True)])
        nb = F.night_blocks(datetime(2026, 11, 9, 22, 30), datetime(2026, 11, 10, 22, 30), set())
        self.assertEqual([(s.hour, e.hour, ss) for s, e, ss in nb], [(2, 6, False)])   # 서머타임 해제

    def test_week_window_monday_holiday(self):
        days = [date(2026, 9, 28), date(2026, 9, 29), date(2026, 10, 2), date(2026, 10, 6), date(2026, 10, 7)]
        ww = F.week_windows(days, "kr")
        self.assertEqual(ww[0][0], date(2026, 10, 6))       # 10/5 휴장 → 화요일 07:59 마감
        self.assertEqual(ww[0][1], datetime(2026, 9, 28, 8))
        self.assertEqual(ww[0][2], datetime(2026, 10, 6, 8))


@unittest.skipUnless((RES / "units.parquet").exists(), "로더 산출이 없다")
class TestNoFuture(unittest.TestCase):
    """마감 뒤 단위를 지워도 그 마감 전 창의 특징이 그대로인지(실제 스냅샷)."""

    @classmethod
    def setUpClass(cls):
        inp = F.load_inputs()
        cls.inp = inp
        u = inp["u"]
        # 사전 결과는 단위마다 독립이라, 이 점검에서는 결정적 가짜 값으로 바꿔도 된다
        rng = np.random.default_rng(7)
        cls.lex = pd.DataFrame({"score": pd.array(np.where(u["utype"] == "post", rng.integers(-2, 3, len(u)), 0),
                                                  dtype="Int64"),
                                "stocks": [()] * len(u), "politics_only": rng.random(len(u)) < 0.01,
                                "lex_excluded": ""}, index=u.index)
        cls.c0 = pd.Timestamp(u.loc[u["utype"] == "post", "dt_kst"].min())

    def _kr(self, u, lex):
        f = F.activity_features(F.window_frame(self.inp["wk"]), u, "win_kr", "pwin_kr", "pre_close_kr")
        f = F.newcomer_features(f, u, lex, "win_kr", self.c0)
        return F.regular_conversion(f, u, lex, self.c0)

    def test_kr_features_unchanged_by_later_units(self):
        u, lex = self.inp["u"], self.lex
        full = self._kr(u, lex)
        cut = full["close_kst"].iloc[45]
        keep = u["dt_kst"] < cut
        part = self._kr(u[keep], lex[keep])
        rows = full["close_kst"] <= cut
        cols = ["saved_cap5", "U", "N", "A_rz8m4", "U_rz8m4", "N_rz8m4", "Bperp_rz8m4", "fight", "fight_all",
                "fight_adj", "fight_z60", "n_comments_onposts", "NW30", "NEW30_per_h", "NW30_rz8m4",
                "h37_den_bear", "h37_bull_conv_rate", "h37_bear_conv_rate"]
        self.assertEqual(len(full), len(part))
        self.assertGreater(int(rows.sum()), 40)
        pd.testing.assert_frame_equal(full.loc[rows, cols], part.loc[rows, cols], check_dtype=False)

    def test_us_night_share_unchanged_by_later_units(self):
        u = self.inp["u"]
        nyse = set(self.inp["cal"].loc[self.inp["cal"]["nyse"], "date"].dt.date)
        fw = F.window_frame(self.inp["wu"])
        full = F.night_share(fw, u, nyse)
        cut = full["close_kst"].iloc[40]
        part = F.night_share(fw, u[u["dt_kst"] < cut], nyse)
        rows = full["close_kst"] <= cut
        cols = ["ns_tot_no_interp", "ns_tot_comments_conf", "ns_night_no_interp", "NS_share", "NS_share_sess"]
        pd.testing.assert_frame_equal(full.loc[rows, cols], part.loc[rows, cols])


if __name__ == "__main__":
    unittest.main()
