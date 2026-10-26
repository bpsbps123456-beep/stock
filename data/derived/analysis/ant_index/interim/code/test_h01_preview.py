"""H01 미리보기 테스트: 가격 특징이 지난 값만 쓰는지, 창 ↔ 세션 짝, 방화벽, HAC, 급락 창 고르기, R² 분해.

실행: py -m unittest test_h01_preview   (이 폴더에서)   네트워크를 쓰지 않는다.
"""

from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h01_preview as H  # noqa: E402


def closes(n: int = 120, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    days, d = [], date(2026, 1, 2)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=days)


class TestPastOnly(unittest.TestCase):
    def test_future_closes_do_not_change_past_features(self):
        c = closes()
        v = pd.Series(np.random.default_rng(1).integers(1, 1000, len(c)).astype(float), index=c.index)
        a = H.session_features(c, v)
        c2, v2 = c.copy(), v.copy()
        cut = c.index[90]
        c2[c2.index > cut] *= 1.5                      # 뒤 종가를 바꿔도
        v2[v2.index > cut] *= 9
        b = H.session_features(c2, v2)
        pd.testing.assert_frame_equal(a.loc[:cut], b.loc[:cut])   # 앞 세션 값은 그대로

    def test_definitions(self):
        c = closes()
        f = H.session_features(c)
        d = c.index[70]
        i = list(c.index).index(d)
        self.assertAlmostEqual(f.at[d, "r"], c.iloc[i] / c.iloc[i - 1] - 1)
        self.assertAlmostEqual(f.at[d, "r20"], c.iloc[i] / c.iloc[i - 20] - 1)
        self.assertAlmostEqual(f.at[d, "DD20"], c.iloc[i] / c.iloc[i - 19:i + 1].max() - 1)
        self.assertAlmostEqual(f.at[d, "rpos"] + f.at[d, "rneg"], f.at[d, "r"])
        self.assertAlmostEqual(f.at[d, "r_lag3"], c.iloc[i - 3] / c.iloc[i - 4] - 1)

    def test_zero_volume_is_missing(self):
        c = closes()
        v = pd.Series(100.0, index=c.index)
        v.iloc[50] = 0
        f = H.session_features(c, v)
        self.assertTrue(np.isnan(f["ln_vol_anom"].iloc[50]))
        self.assertTrue(np.isfinite(f["ln_vol_anom"].iloc[51]))   # 기준 평균에서는 빠지고 계산은 된다


class TestFirewallAndMapping(unittest.TestCase):
    def test_guard_past(self):
        H.guard_past([date(2026, 10, 14)])
        with self.assertRaises(RuntimeError):
            H.guard_past([date(2026, 10, 14), date(2026, 10, 15)])

    def test_last_us_session(self):
        us = [date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 8)]   # 9/7 노동절 휴장
        self.assertEqual(H.last_us_session(date(2026, 9, 7), us), date(2026, 9, 4))    # 월요일 창 → 금요일
        self.assertEqual(H.last_us_session(date(2026, 9, 8), us), date(2026, 9, 4))    # 휴장 다음 날 창
        self.assertEqual(H.last_us_session(date(2026, 9, 9), us), date(2026, 9, 8))
        for D in (date(2026, 9, 7), date(2026, 9, 9)):
            self.assertLessEqual(H.last_us_session(D, us), D - timedelta(days=1))

    def test_in_window_and_opex(self):
        self.assertTrue(H.in_window(date(2026, 9, 18), date(2026, 9, 18), date(2026, 9, 21)))
        self.assertTrue(H.in_window(date(2026, 9, 19), date(2026, 9, 18), date(2026, 9, 21)))
        self.assertFalse(H.in_window(date(2026, 9, 21), date(2026, 9, 18), date(2026, 9, 21)))
        self.assertEqual(H.nth_weekday(2026, 10, 4, 3), date(2026, 10, 16))
        cal = {date(2026, 10, 14), date(2026, 10, 15)}
        # 달력 끝 뒤 만기일은 옮기지 않는다(10/14로 끌려오면 마지막 창에 잘못 든다)
        self.assertEqual(H.to_prev_session(date(2026, 10, 16), cal, date(2026, 10, 15)), date(2026, 10, 16))
        self.assertEqual(H.to_prev_session(date(2026, 10, 15), {date(2026, 10, 14)}, date(2026, 10, 15)), date(2026, 10, 14))

    def test_crash_overlap_union(self):
        D = [date(2026, 7, 28), date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31), date(2026, 8, 3), date(2026, 8, 4)]
        P = [date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31), date(2026, 8, 3)]
        o = pd.Series([pd.Timestamp(p) + pd.Timedelta(hours=8) for p in P])
        c = pd.Series([pd.Timestamp(d) + pd.Timedelta(hours=8) for d in D])
        m = H.overlaps_days(o, c, [date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31)])
        self.assertEqual(m.tolist(), [False, True, True, True, True, False])


class TestStats(unittest.TestCase):
    def test_hac_L0_is_white(self):
        rng = np.random.default_rng(4)
        n = 50
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = 0.3 * X[:, 1] + rng.normal(size=n) * (1 + np.abs(X[:, 1]))
        b, e = H.ols(y, X)
        inv = np.linalg.inv(X.T @ X)
        white = inv @ (X.T * e ** 2) @ X @ inv
        np.testing.assert_allclose(H.hac_cov(X, e, np.arange(n), L=0, small=False), white)

    def test_hac_skips_gaps(self):
        rng = np.random.default_rng(5)
        n = 40
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        e = rng.normal(size=n)
        far = np.arange(n) * 10                          # 모든 행이 10 넘게 떨어져 지연 짝이 없다
        np.testing.assert_allclose(H.hac_cov(X, e, far, L=5, small=False), H.hac_cov(X, e, far, L=0, small=False))
        self.assertFalse(np.allclose(H.hac_cov(X, e, np.arange(n), L=5, small=False),
                                     H.hac_cov(X, e, np.arange(n), L=0, small=False)))

    def test_corr_test_slope_equals_pearson(self):
        rng = np.random.default_rng(6)
        r = rng.normal(size=60)
        s = 0.4 * r + rng.normal(size=60)
        c = H.corr_test(s, r, np.arange(60))
        self.assertAlmostEqual(c["rho"], np.corrcoef(s, r)[0, 1])
        b, _ = H.ols(H.zs(s), np.column_stack([np.ones(60), H.zs(r)]))
        self.assertAlmostEqual(b[1], c["rho"])

    def test_shapley_sums_to_r2(self):
        rng = np.random.default_rng(7)
        n = 80
        df = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
        y = df["a"] + 0.5 * df["c"] + 0.3 * df["e"] + rng.normal(size=n)
        t = H.block_table(y.to_numpy(), df, {"x": ["a", "b"], "y": ["c"], "z": ["d", "e"]})
        self.assertAlmostEqual(sum(v["shapley"] for v in t["blocks"].values()), t["r2"])
        for v in t["blocks"].values():
            self.assertGreaterEqual(v["partial_r2"], v["delta_r2"] - 1e-12)

    def test_circ_shift_range(self):
        self.assertEqual(H.circ_shifts(60), list(range(21, 40)))
        self.assertEqual(H.circ_shifts(40), [])


if __name__ == "__main__":
    unittest.main()
