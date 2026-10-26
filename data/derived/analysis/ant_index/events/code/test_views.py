"""조회·추천 특징 코드 테스트(가짜 자료만, 네트워크·원자료 없음).

실행: py -m unittest test_views   (이 폴더에서)   또는   py test_views.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feat_views as FV  # noqa: E402
import views_fetch as VF  # noqa: E402


def ts(s: str) -> np.datetime64:
    return np.datetime64(pd.Timestamp(s).to_datetime64(), "s")


class TestStats(unittest.TestCase):
    def test_gini(self):
        self.assertAlmostEqual(FV.gini(np.ones(10)), 0.0)
        x = np.zeros(100)
        x[-1] = 5
        self.assertAlmostEqual(FV.gini(x), 0.99, places=6)          # 한 글이 전부 = (n−1)/n
        self.assertTrue(np.isnan(FV.gini(np.array([3.0]))))

    def test_top_share(self):
        x = np.arange(1, 21, dtype=float)                             # 1..20, 위 10개 합 155 / 210
        self.assertAlmostEqual(FV.top_share(x), 155 / 210)
        self.assertAlmostEqual(FV.top_share(x, 2), 39 / 210)

    def test_pav(self):
        out = FV.pav(np.array([1.0, 3.0, 2.0, 4.0]), np.ones(4))
        np.testing.assert_allclose(out, [1.0, 2.5, 2.5, 4.0])
        self.assertTrue((np.diff(FV.pav(np.random.default_rng(1).normal(size=50), np.ones(50))) >= -1e-12).all())


class TestCurve(unittest.TestCase):
    def test_fit_recovers_known_curve(self):
        # 참 곡선 log g(a) = −0.3·log(24/a) (a ≤ 24h), 0 그 뒤. 둘째 줄 24~28h
        rng = np.random.default_rng(7)
        a1 = np.exp(rng.uniform(np.log(0.2), np.log(20), 3000))
        a2 = rng.uniform(24, 28, 3000)
        true = lambda a: np.where(a < 24, -0.3 * np.log(24 / a), 0.0)
        y = true(a2) - true(a1) + rng.normal(0, 0.02, 3000)
        fit = FV.fit_short_curve(a1, a2, y)
        for a in (0.5, 1, 3, 12):
            got = float(FV.eval_curve(np.array([a]), fit["knots_h"], fit["h"])[0])
            self.assertAlmostEqual(got, float(true(np.array([a]))[0]), delta=0.03)
        self.assertEqual(fit["h"][fit["knots_h"].index(24.0)], 0.0)
        self.assertTrue((np.diff(fit["h"]) >= -1e-9).all())


class TestFetch(unittest.TestCase):
    def test_propagate(self):
        L = np.array([ts("2026-10-14 10:00"), ts("2026-10-14 09:00"), ts("2026-10-14 11:00")])
        U = np.array([ts("2026-10-14 12:00"), ts("2026-10-14 10:30"), ts("2026-10-14 13:00")])
        Lp, Up = VF.propagate(L, U, [np.array([0, 1, 2])])
        self.assertEqual(Lp[1], ts("2026-10-14 10:00"))   # 앞줄 하한이 뒤로
        self.assertEqual(Up[0], ts("2026-10-14 10:30"))   # 뒷줄 상한이 앞으로

    def test_stab_events(self):
        L = np.array([ts("2026-10-14 10:00"), ts("2026-10-14 10:05"), ts("2026-10-14 13:00")])
        U = np.array([ts("2026-10-14 10:20"), ts("2026-10-14 10:30"), ts("2026-10-14 13:10")])
        ev, out = VF.stab_events(L, U)
        self.assertEqual(ev.tolist(), [0, 0, 1])
        self.assertEqual(out[0], (ts("2026-10-14 10:05"), ts("2026-10-14 10:20")))

    def test_schedule_fills_gaps_and_wide(self):
        anchors = np.array([ts("2026-10-14 00:00"), ts("2026-10-14 12:00")])
        active = [(ts("2026-10-13 23:00"), ts("2026-10-14 18:00"))]
        wide = [(ts("2026-10-14 05:30"), ts("2026-10-14 07:00"))]
        sched, virt = VF.schedule(anchors, wide, active, 3 * 3600, 4.2 * 3600, ts("2026-10-15 00:00"))
        # 0시~12시 간격 12시간 → 가상 3개(3·6·9시), 6시는 넓은 사건 범위 안이라 그대로, 12시 뒤 15시 가상 1개(18시 미만)
        self.assertEqual(int(virt.sum()), 4)
        self.assertIn(ts("2026-10-14 06:00"), sched)
        self.assertIn(ts("2026-10-14 15:00"), sched)

    def test_backward_time(self):
        log = pd.DataFrame({"t": [pd.Timestamp("2026-10-14 01:00"), pd.Timestamp("2026-10-14 02:00"),
                                  pd.Timestamp("2026-10-14 05:00")],
                            "low": [9000, 8000, 7000], "saved": [900, 1800, 100], "run": [0, 0, 1]})
        model = {"rate_s_per_no": 3.6, "runs": []}
        runs = []
        for r, g in log.groupby("run"):
            runs.append({"run": r, "L0": int(g["low"].iloc[0]) + 1000, "first_t": g["t"].iloc[0], "last_t": g["t"].iloc[-1],
                         "first_low": int(g["low"].iloc[0]), "last_low": int(g["low"].iloc[-1]),
                         "t": g["t"].tolist(), "low": g["low"].tolist(),
                         "start_t_est": g["t"].iloc[0] - pd.Timedelta(hours=1)})
        runs[0]["end_low"], runs[1]["end_low"] = runs[1]["L0"], -1
        runs[0]["prev_last_t"], runs[1]["prev_last_t"] = None, runs[0]["last_t"]
        runs[0]["next_start_t"], runs[1]["next_start_t"] = runs[1]["start_t_est"], None
        model["runs"] = runs
        no = np.array([8500, 9500, 7900])
        T, lo, hi, run = VF.backward_time(no, model, pd.Timestamp("2026-10-15"))
        self.assertEqual(pd.Timestamp(T[0]), pd.Timestamp("2026-10-14 01:30:30"))   # 9000(01:00)과 8000(02:00) 사이 가운데
        self.assertEqual(pd.Timestamp(T[1]), pd.Timestamp("2026-10-14 00:30:30"))   # 첫 로그 위: 3.6초 × 500 앞
        self.assertEqual(run.tolist(), [0, 0, 1])
        self.assertLessEqual(pd.Timestamp(lo[0]), pd.Timestamp(T[0]))
        self.assertGreaterEqual(pd.Timestamp(hi[0]), pd.Timestamp(T[0]))

    def test_parse_log(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "collect.log"
            p.write_text("2026-10-14 00:10  번호 440,523 (2026-10-11)  저장 938  없음 62  실패 0\n"
                         "2026-10-14 01:33  번호 439,523 (2026-10-05)  저장 1,851  없음 149  실패 0\n"
                         "2026-10-16 09:41  번호 405,523 (2026-07-17)  저장 1,032  없음 107  실패 0\n", "utf-8")
            lg = VF.parse_log(p)
            self.assertEqual(lg["low"].tolist(), [440523, 439523, 405523])
            self.assertEqual(lg["run"].tolist(), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
