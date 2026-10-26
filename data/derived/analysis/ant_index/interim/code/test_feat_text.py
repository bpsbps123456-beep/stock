"""텍스트 특징 테스트: 정규화가 직전 창만 쓰는지(자기 제외·최소 창 수), 작성자 1회 셈, 창 경계, 뉴스 분류, 사건 σ60.

실행: py -m unittest test_feat_text   (이 폴더에서)   네트워크를 쓰지 않는다.
"""

from __future__ import annotations

import sys
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feat_text as T  # noqa: E402


class TestNormalize(unittest.TestCase):
    def test_z60_self_excluded_and_min40(self):
        x = np.arange(70, dtype=float)
        v = np.ones(70, bool)
        z = T.z_prev(x, v)
        self.assertTrue(np.all(np.isnan(z[:40])))
        h = x[0:40]
        self.assertAlmostEqual(z[40], (40 - h.mean()) / h.std(ddof=1))
        h = x[5:65]                                   # 직전 60창만
        self.assertAlmostEqual(z[65], (65 - h.mean()) / h.std(ddof=1))

    def test_z60_future_does_not_change_past(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=80)
        v = np.ones(80, bool)
        a = T.z_prev(x, v)
        x2 = x.copy()
        x2[60:] += 100                                # 뒤 값을 바꿔도 앞 z는 그대로
        b = T.z_prev(x2, v)
        np.testing.assert_allclose(a[:60], b[:60])

    def test_invalid_window_not_in_baseline(self):
        rng = np.random.default_rng(2)
        x = np.r_[rng.normal(size=40), 1000.0, rng.normal(size=5)]
        v = np.ones(46, bool)
        v[40] = False
        z = T.z_prev(x, v)
        self.assertTrue(np.isnan(z[40]))
        h = x[:40]                                     # 1000(무효 창)은 기준에 안 들어간다
        self.assertAlmostEqual(z[41], (x[41] - h.mean()) / h.std(ddof=1))

    def test_pct250_min120(self):
        x = np.arange(130, dtype=float)
        p = T.pct_prev(x, np.ones(130, bool))
        self.assertTrue(np.all(np.isnan(p[:120])))
        self.assertEqual(p[120], 1.0)
        self.assertEqual(T.stage_of(np.array([0.0, 0.05, 0.95, 1.0, np.nan]), 10).tolist()[:4], [1, 1, 10, 10])

    def test_rz8_same_weekday_holiday_excluded(self):
        n = 5 * 12
        wd = np.tile(np.arange(5), 12)
        x = np.random.default_rng(3).normal(size=n) + np.where(wd == 0, 10.0, 0.0)
        hol = np.zeros(n, bool)
        hol[5] = True                                  # 둘째 월요일은 연휴 뒤 창
        r = T.rz8(x, np.ones(n, bool), wd, hol)
        mon = np.flatnonzero(wd == 0)
        self.assertTrue(np.all(np.isnan(r[mon[:9]])))  # 연휴 창이 빠져 기준 8개는 10번째 월요일에야 찬다
        self.assertTrue(np.isfinite(r[mon[9]]))
        self.assertLessEqual(np.nanmax(np.abs(r)), 6.0)


class TestAggregate(unittest.TestCase):
    def test_author_ratio_counts_author_once(self):
        df = pd.DataFrame({"w": ["A"] * 4 + ["B"] * 2, "a": ["x", "x", "y", "z", "x", "y"],
                           "h_E": [True, True, False, False, False, False]})
        r, n, cnt = T.author_ratio(df, "w", ["h_E"], pd.Index(["A", "B", "C"]))
        self.assertAlmostEqual(r.loc["A", "h_E"], 1 / 3)
        self.assertEqual(r.loc["B", "h_E"], 0.0)
        self.assertTrue(np.isnan(r.loc["C", "h_E"]))
        self.assertEqual(int(n.loc["A"]), 3)
        self.assertEqual(int(cnt.loc["A", "h_E"]), 1)

    def test_assign_ranges_half_open(self):
        o = np.array(["2026-10-13T09:00", "2026-10-14T09:00"], dtype="datetime64[s]")
        c = np.array(["2026-10-14T08:00", "2026-10-15T08:00"], dtype="datetime64[s]")
        t = pd.Series(pd.to_datetime(["2026-10-13 08:59:00", "2026-10-13 09:00:00", "2026-10-14 07:59:59",
                                      "2026-10-14 08:00:00", "2026-10-14 09:00:00"]))
        out = T.assign_ranges(t, o, c, np.array(["d1", "d2"], dtype=object))
        self.assertEqual(list(out), [None, "d1", "d1", None, "d2"])

    def test_top_mask_ceil(self):
        att = pd.Series([1.0, 5.0, 3.0, 2.0, 4.0, 0.5, 0.1, 9.0, 7.0, 6.0, 8.0])
        m = T.top_mask(att, pd.Series([0] * 11), pd.Series(range(11)))
        self.assertEqual(int(m.sum()), 2)               # ⌈1.1⌉
        self.assertEqual(set(att[m]), {9.0, 8.0})

    def test_attention_uses_previous_days_only(self):
        rows = []
        start = date(2026, 8, 1)
        for k in range(25):
            d = start + timedelta(days=k)
            rows.append({"dt_kst": pd.Timestamp(datetime(d.year, d.month, d.day, 10)), "c": 4.0, "ok": True})
        p = pd.DataFrame(rows)
        p["day"], p["hour"] = p["dt_kst"].dt.date, p["dt_kst"].dt.hour
        a = T.attention(p, "c", "ok", start)
        self.assertTrue(np.all(np.isnan(a.iloc[:20])))  # 직전 20일이 다 자료 안이어야 한다
        self.assertAlmostEqual(a.iloc[20], 1.0)
        p2 = p.copy()
        p2.loc[20, "c"] = 40.0                          # 같은 날 값은 기준에 안 들어간다
        self.assertAlmostEqual(T.attention(p2, "c", "ok", start).iloc[20], 10.0)


class TestMisc(unittest.TestCase):
    def test_classify_news(self):
        self.assertEqual(T.classify_news("[시황] 장 마감", ""), "price")
        self.assertEqual(T.classify_news("[속보] 코스피 2% 하락", ""), "price")
        self.assertEqual(T.classify_news("[단독] 삼성 수주 공시", "본문"), "event")
        self.assertEqual(T.classify_news("[펌] 아무 이야기", "본문\n둘째 줄 실적"), "other")  # 본문 첫 줄만 본다

    def test_shock_z_excludes_today(self):
        idx = [date(2026, 1, 1) + timedelta(days=k) for k in range(70)]
        r = np.r_[0.0, np.tile([0.01, -0.01], 34), -0.10]
        close = pd.Series(100 * np.cumprod(1 + r), index=idx)
        z = T.shock_z(close)
        sig = pd.Series(r[1:-1][-60:]).std(ddof=1)
        self.assertAlmostEqual(z.iloc[-1], -0.10 / sig, places=6)
        self.assertTrue(np.isnan(z.iloc[59]))           # 직전 일수익 60개가 차기 전


if __name__ == "__main__":
    unittest.main()
