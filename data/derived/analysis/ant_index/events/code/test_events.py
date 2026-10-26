# -*- coding: utf-8 -*-
"""events.py 단위 테스트. 네트워크와 실제 가격을 쓰지 않는다.

실행: py -m unittest test_events   (이 폴더에서)
"""
from __future__ import annotations

import random
import sys
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import events as E  # noqa: E402


def kinds(pivots):
    return [(p.kind, p.idx, p.confirm_idx) for p in pivots]


class TestZigzagBasics(unittest.TestCase):
    def test_boundary_inclusive(self):
        pv, _ = E.zigzag([100, 92], 0.08)            # 정확히 -8% → 고점 확정
        self.assertEqual(kinds(pv), [("high", 0, 1)])
        pv, _ = E.zigzag([100, 92.01], 0.08)
        self.assertEqual(pv, [])
        pv, _ = E.zigzag([100, 108], 0.08)           # 정확히 +8% → 저점 확정
        self.assertEqual(kinds(pv), [("low", 0, 1)])
        pv, _ = E.zigzag([100, 107.99], 0.08)
        self.assertEqual(pv, [])
        pv, _ = E.zigzag([100, 88], 0.12)            # 강건판 경계
        self.assertEqual(kinds(pv), [("high", 0, 1)])
        pv, _ = E.zigzag([100, 88.01], 0.12)
        self.assertEqual(pv, [])

    def test_downtrend_then_rebound(self):
        c = [100, 110, 120, 115, 110.4, 105, 100, 103, 108, 104]
        pv, tail = E.zigzag(c, 0.08)
        # 첫 확정: 100에서 +8%(108 이상)는 1번째 날 110 → 0번째 날 100이 저점
        # 120(2) → 110.4(4)는 -8% → 고점 2 확정, 100(6) → 108(8)은 +8% → 저점 6 확정
        self.assertEqual(kinds(pv), [("low", 0, 1), ("high", 2, 4), ("low", 6, 8)])
        self.assertEqual((tail.kind, tail.idx), ("high", 8))

    def test_low_can_be_breached_without_new_pivot(self):
        # 100 저점이 108로 확정된 뒤 99.5(108 대비 -7.87%)는 고점 확정이 아니라 새 꺾임점이 생기지 않는다
        dates = [f"2025-09-{d:02d}" for d in range(1, 6)]
        c = [100, 108, 99.5, 120, 110]
        pv, _ = E.zigzag(c, 0.08)
        self.assertEqual(kinds(pv), [("low", 0, 1), ("high", 3, 4)])
        old = E.EVENT_START
        E.EVENT_START = "2025-09-01"
        try:
            ev = E.asset_events(dates, c, 0.08, {})
        finally:
            E.EVENT_START = old
        self.assertEqual(ev["pivots"][0]["breached_after_confirm"], ["2025-09-03"])
        self.assertEqual(ev["pivots"][1]["breached_after_confirm"], [])

    def test_invalid_close(self):
        with self.assertRaises(ValueError):
            E.zigzag([100, float("nan")], 0.08)
        with self.assertRaises(ValueError):
            E.zigzag([100, 0], 0.08)

    def test_empty_and_flat(self):
        self.assertEqual(E.zigzag([], 0.08), ([], E.Tail(None, None, None)))
        pv, tail = E.zigzag([100] * 10, 0.08)
        self.assertEqual(pv, [])
        self.assertIsNone(tail.kind)
        self.assertEqual((tail.hi_idx, tail.lo_idx), (0, 0))   # 같은 값이면 첫 날


class TestUndeterminedStart(unittest.TestCase):
    """처음 방향이 정해지지 않은 구간."""

    def test_low_first(self):
        c = [100, 97, 95, 99, 102.6, 101]
        pv, tail = E.zigzag(c, 0.08)
        # 100→95는 -5%라 시작일은 고점이 아니다. 95→102.6이 정확히 +8% → 2번째 날 저점, 4번째 날 확정
        self.assertEqual(kinds(pv), [("low", 2, 4)])
        self.assertEqual((tail.kind, tail.idx), ("high", 4))

    def test_high_first(self):
        c = [100, 104, 103, 95.68]
        pv, tail = E.zigzag(c, 0.08)
        self.assertEqual(kinds(pv), [("high", 1, 3)])
        self.assertEqual((tail.kind, tail.idx), ("low", 3))

    def test_undetermined_tail_keeps_both(self):
        pv, tail = E.zigzag([100, 105, 97, 104, 98], 0.08)
        self.assertEqual(pv, [])
        self.assertEqual((tail.kind, tail.hi_idx, tail.lo_idx), (None, 1, 2))

    def test_high_after_first_low_is_searched_from_the_low(self):
        # 방향 미정 구간의 최고(107, 1번째 날)가 저점(99, 2번째 날)보다 앞이다.
        # 저점 확정 뒤 고점 후보는 저점 이후에서만 찾아야 한다 → 3번째 날 106.92가 고점(1번째 날 107이 아님).
        c = [100, 107, 99, 106.92, 106.92 * 0.92]
        pv, _ = E.zigzag(c, 0.08)
        self.assertEqual(kinds(pv), [("low", 2, 3), ("high", 3, 4)])
        for a, b in zip(pv, pv[1:]):
            self.assertLess(a.idx, b.idx)

    def test_min_before_max_resolves_to_high(self):
        # 방향 미정 최저(95, 1번째 날)가 최고(100.4, 2번째 날)보다 앞이고, 100.4에서 -8% 닿음 → 고점만 확정.
        # 95는 꺾임점이 아니고, 새 저점 후보는 확정일(3번째 날)부터 찾는다.
        c = [100, 95, 100.4, 100.4 * 0.92, 95.5]
        pv, tail = E.zigzag(c, 0.08)
        self.assertEqual(kinds(pv), [("high", 2, 3)])
        self.assertEqual((tail.kind, tail.idx), ("low", 3))

    def test_start_day_can_be_pivot(self):
        pv, _ = E.zigzag([100, 95, 91.9, 99.5], 0.08)
        self.assertEqual(kinds(pv)[0], ("high", 0, 2))


class TestSameDay(unittest.TestCase):
    """같은 날 양방향."""

    def test_confirm_day_becomes_next_extreme(self):
        # 2번째 날(100)이 120 고점을 확정하는 날이면서, 곧바로 저점이 된다(3번째 날 115로 +15%).
        pv, tail = E.zigzag([100, 120, 100, 115], 0.08)
        self.assertEqual(kinds(pv), [("low", 0, 1), ("high", 1, 2), ("low", 2, 3)])
        self.assertEqual((tail.kind, tail.idx), ("high", 3))

    def test_gap_day_in_undetermined_confirms_one_side_only(self):
        # 방향 미정에서 한 날에 크게 뛰어 최고도 새로 쓰고 최저 대비 +8%도 닿는 경우 → 저점 하나만 확정, 그날은 고점 후보
        pv, tail = E.zigzag([100, 96, 99, 120], 0.08)
        self.assertEqual(kinds(pv), [("low", 1, 3)])
        self.assertEqual((tail.kind, tail.idx), ("high", 3))
        # 반대로 크게 빠지는 날
        pv, tail = E.zigzag([100, 104, 101, 80], 0.08)
        self.assertEqual(kinds(pv), [("high", 1, 3)])
        self.assertEqual((tail.kind, tail.idx), ("low", 3))

    def test_both_sides_never_on_one_day_random(self):
        """변동이 큰 무작위 종가 4,000개 묶음에서: 단언이 터지지 않고, 하루에 확정은 하나, 고·저가 번갈아 나온다,
        진폭 >= X, 극값은 이웃 꺾임점 사이의 진짜 최고·최저, 확정일은 처음 X%에 닿은 날."""
        rng = np.random.default_rng(20261022)
        for trial in range(4000):
            n = int(rng.integers(2, 160))
            vol = float(rng.choice([0.01, 0.03, 0.06, 0.12]))
            c = list(100 * np.exp(np.cumsum(rng.normal(0, vol, n))))
            if trial % 7 == 0:   # 같은 값 이어짐도 섞는다
                for i in range(1, n):
                    if rng.random() < 0.2:
                        c[i] = c[i - 1]
            x = float(rng.choice([0.08, 0.12]))
            pv, tail = E.zigzag(c, x)
            conf = [p.confirm_idx for p in pv]
            self.assertEqual(len(conf), len(set(conf)))
            self.assertEqual(conf, sorted(conf))
            for a, b in zip(pv, pv[1:]):
                self.assertNotEqual(a.kind, b.kind)
                self.assertLess(a.idx, b.idx)
                self.assertLessEqual(a.confirm_idx, b.idx)      # 다음 극값은 앞 확정일 이후
                if a.kind == "low":
                    self.assertGreaterEqual(b.value / a.value - 1, x - 1e-9)
                else:
                    self.assertLessEqual(b.value / a.value - 1, -x + 1e-9)
            for k, p in enumerate(pv):
                # 극값은 '앞 꺾임점 확정일(처음이면 0) ~ 이 꺾임점 확정일' 구간의 진짜 최고·최저(같으면 처음 날)
                s = pv[k - 1].confirm_idx if k else 0
                seg = c[s:p.confirm_idx + 1]
                target = max(seg) if p.kind == "high" else min(seg)
                self.assertEqual(p.value, target)
                self.assertEqual(p.idx, s + seg.index(target))
            for p in pv:
                self.assertLess(p.idx, p.confirm_idx)
                between = c[p.idx + 1:p.confirm_idx]
                if p.kind == "high":
                    self.assertTrue(all(v / p.value - 1 > -x + E.TOL for v in between))
                    self.assertLessEqual(c[p.confirm_idx] / p.value - 1, -x + E.TOL)
                    self.assertTrue(all(v <= p.value for v in c[p.idx:p.confirm_idx + 1]))
                else:
                    self.assertTrue(all(v / p.value - 1 < x - E.TOL for v in between))
                    self.assertGreaterEqual(c[p.confirm_idx] / p.value - 1, x - E.TOL)
                    self.assertTrue(all(v >= p.value for v in c[p.idx:p.confirm_idx + 1]))
            if pv and tail.kind is not None:
                self.assertNotEqual(tail.kind, pv[-1].kind)
                self.assertGreaterEqual(tail.idx, pv[-1].confirm_idx)
                rest = c[pv[-1].confirm_idx:]
                self.assertEqual(tail.value, max(rest) if tail.kind == "high" else min(rest))

    def test_brute_force_reference(self):
        """다른 방식(매일 전 구간을 다시 훑는 느린 구현)과 같은 결과."""
        def slow(c, x):
            piv, last_kind, last_conf = [], None, 0
            t = 1
            while t < len(c):
                found = None
                if last_kind in (None, "high"):   # 저점 찾기
                    seg = range(last_conf, t)
                    lo = min(seg, key=lambda i: (c[i], i))
                    if c[t] / c[lo] - 1 >= x - E.TOL:
                        found = ("low", lo)
                if found is None and last_kind in (None, "low"):
                    seg = range(last_conf, t)
                    hi = min(seg, key=lambda i: (-c[i], i))
                    if c[t] / c[hi] - 1 <= -x + E.TOL:
                        found = ("high", hi)
                if found:
                    piv.append((found[0], found[1], t))
                    last_kind, last_conf = found[0], t
                t += 1
            return piv
        rng = np.random.default_rng(7)
        for _ in range(1500):
            n = int(rng.integers(2, 120))
            c = list(np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.05, n))), 2))
            for x in (0.08, 0.12):
                self.assertEqual(kinds(E.zigzag(c, x)[0]), slow(c, x))


class TestFakeEvents(unittest.TestCase):
    def test_fake_bottom_thresholds(self):
        c = [100, 95, 95, 95, 95, 95, 95, 94.99, 94.99 * 0.9501]
        got = {r["idx"]: r["trigger"] for r in E.fake_bottom_candidates(c)}
        self.assertEqual(got, {1: "1d"})              # 정확히 -5%는 넣고, 마지막 날(-4.99%, 5일 -5%)은 뺀다
        c2 = [100, 99, 98, 97, 96, 92]                 # 5일 전(100) 대비 정확히 -8%, 하루 -4.17%
        got2 = {r["idx"]: r["trigger"] for r in E.fake_bottom_candidates(c2)}
        self.assertEqual(got2, {5: "5d"})
        c3 = [100, 99, 98, 97, 96, 92.01]
        self.assertEqual(E.fake_bottom_candidates(c3), [])
        c4 = [100, 100, 100, 100, 100, 92]             # 두 조건 모두
        self.assertEqual(E.fake_bottom_candidates(c4)[0]["trigger"], "1d+5d")

    def test_fake_top_20day(self):
        c = list(range(100, 119)) + [118, 118.5, 117]
        got = [r["idx"] for r in E.fake_top_candidates(c)]
        # 0~18번째 날은 오늘 포함 20일이 안 된다. 19번째 날(118)은 앞 19일 최고 118과 같아 최고로 본다.
        # 20번째 날 118.5도 최고. 21번째 날 117은 아님
        self.assertEqual(got, [19, 20])
        self.assertEqual(E.fake_top_candidates(list(range(100, 119))), [])   # 19일뿐이면 없음

    def test_exclusion_radius(self):
        cands = [{"idx": i} for i in (6, 7, 10, 13, 14)]
        fakes, held, dropped = E.split_fakes(cands, confirmed=[10], tail_idx=None)
        self.assertEqual([r["idx"] for r in fakes], [6, 14])        # 거리 4는 남고
        self.assertEqual([r["idx"] for r in dropped], [7, 10, 13])  # 거리 3 이내는 뺀다
        self.assertEqual(held, [])
        fakes, held, dropped = E.split_fakes(cands, confirmed=[], tail_idx=13)
        self.assertEqual([r["idx"] for r in held], [10, 13, 14])    # 미확정 끝 극값 ±3 안은 보류
        self.assertEqual([r["idx"] for r in fakes], [6, 7])
        self.assertEqual(dropped, [])
        fakes, held, dropped = E.split_fakes(cands, confirmed=[7], tail_idx=13)   # 확정 근처가 먼저
        self.assertEqual([r["idx"] for r in dropped], [6, 7, 10])
        self.assertEqual([r["idx"] for r in held], [13, 14])
        self.assertEqual(fakes, [])

    def test_runs(self):
        self.assertEqual(E.label_runs([1, 2, 5, 9, 13, 20]), [0, 0, 0, 1, 2, 3])   # 간격 3까지 같은 묶음, 4부터 새 묶음

    def test_asset_events_end_to_end(self):
        dates = [f"2025-08-{d:02d}" for d in range(1, 31)]
        # 8/21 이후 사건만 쓰는지: 8/05 저점(기간 밖), 8/22 고점, 8/27 저점
        c = [100] * 4 + [90, 95, 100, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
                         118, 120, 115, 112, 109, 108, 110, 112, 117, 119]
        old_start, old_end = E.EVENT_START, E.EVENT_END
        E.EVENT_START, E.EVENT_END = "2025-08-21", "2025-08-30"
        try:
            ev = E.asset_events(dates, c, 0.08, {})
        finally:
            E.EVENT_START, E.EVENT_END = old_start, old_end
        self.assertEqual(ev["tops"], ["2025-08-22"])
        self.assertEqual(ev["bottoms"], ["2025-08-26"])
        self.assertEqual([p["date"] for p in ev["warmup_pivots"]], ["2025-08-01", "2025-08-05"])
        top = ev["pivots"][0]
        self.assertAlmostEqual(top["amp_in"], 120 / 90 - 1, places=6)
        self.assertEqual(top["confirm_date"], "2025-08-25")      # 120*0.92=110.4 → 109(8/25)
        self.assertEqual(top["confirm_lag_td"], 3)
        self.assertEqual(ev["tail_unconfirmed"]["kind"], "high")


class TestRejudge(unittest.TestCase):
    D = ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16", "2026-10-19",
         "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-26"]

    def test_pending_before_deadline(self):
        c = [120, 110, 105, 100, 102, 104, 103, 105]       # 10/21까지, +8% 반등 없음
        r = E.rejudge_tail(self.D[:8], c, 0.08)
        self.assertEqual(r["focus"]["status"], "미확정")
        self.assertTrue(r["focus"]["is_tail_low"])
        self.assertFalse(r["deadline_close_in"])
        self.assertTrue(r["stage"].startswith("판정 대기"))
        self.assertEqual(r["tail"]["status"], "미확정")
        self.assertAlmostEqual(r["tail"]["confirm_level"], 108.0)

    def test_confirmed_by_deadline(self):
        c = [120, 110, 105, 100, 102, 104, 103, 105, 107, 108]
        r = E.rejudge_tail(self.D[:10], c, 0.08)
        self.assertEqual(r["focus"]["status"], "확정")
        self.assertEqual(r["focus"]["confirm_date"], "2026-10-23")
        self.assertEqual(r["tail"]["kind"], "high")

    def test_not_confirmed_at_deadline_stays_unconfirmed(self):
        c = [120, 110, 105, 100, 102, 104, 103, 105, 107, 107.9, 115]   # 10/26 반등은 마감 뒤라 버린다
        r = E.rejudge_tail(self.D, c, 0.08)
        self.assertTrue(r["deadline_close_in"])
        self.assertEqual(r["data_through"], "2026-10-23")
        self.assertEqual(r["focus"]["status"], "미확정")
        self.assertEqual(r["stage"], "마감 뒤 판정 끝")

    def test_focus_undercut_moves_tail(self):
        c = [120, 110, 105, 100, 102, 99.9, 103, 101]
        r = E.rejudge_tail(self.D[:8], c, 0.08)
        self.assertEqual(r["focus"]["status"], "미확정")
        self.assertFalse(r["focus"]["is_tail_low"])
        self.assertEqual(r["focus"]["undercut_on"], ["2026-10-19"])
        self.assertEqual(r["tail"]["date"], "2026-10-19")

    def test_tie_keeps_first_and_reports_later_equal(self):
        c = [120, 110, 105, 100, 102, 100, 103]
        r = E.rejudge_tail(self.D[:7], c, 0.08)
        self.assertEqual(r["tail"]["date"], "2026-10-15")
        self.assertEqual(r["tail"]["same_close_later"], ["2026-10-19"])


class TestReconcile(unittest.TestCase):
    def _nv(self, rows):
        return E.add_ref_next(pd.DataFrame(rows, columns=["date", "close", "chg"]))

    def test_close_verify_fills_missing_yf(self):
        cfg = dict(E.ASSETS["KOSPI"])
        days = ["2026-10-19", "2026-10-20", "2026-10-21"]
        yfd = pd.DataFrame({"date": days, "yf_close": [5768.009766, 5552.740234, np.nan], "yf_adj": [5768.009766, 5552.740234, np.nan]})
        nv = self._nv([("2026-10-19", 5768.01, 100.0), ("2026-10-20", 5552.74, -215.27), ("2026-10-21", 5735.66, 182.92)])
        df, rep = E.reconcile("KOSPI", cfg, days, yfd, nv)
        self.assertEqual(df["close"].tolist(), [5768.01, 5552.74, 5735.66])
        self.assertEqual(df["source"].tolist(), ["yfinance", "yfinance", "naver_close"])
        self.assertEqual(len(rep), 1)

    def test_ref_next_repairs_wrong_and_missing(self):
        cfg = dict(E.ASSETS["SKHYNIX"])
        days = ["2025-09-17", "2025-09-18", "2025-09-19", "2025-09-22"]
        yfd = pd.DataFrame({"date": ["2025-09-17", "2025-09-18", "2025-09-22"],
                            "yf_close": [333500.0, 355500.0, 351000.0], "yf_adj": [333500.0, 355500.0, 351000.0]})
        nv = self._nv([("2025-09-17", 333500.0, -14500.0), ("2025-09-18", 353000.0, 19500.0),
                       ("2025-09-19", 353000.0, 0.0), ("2025-09-22", 351000.0, -2000.0)])
        df, rep = E.reconcile("SKHYNIX", cfg, days, yfd, nv)
        self.assertEqual(df["close"].tolist(), [333500.0, 353000.0, 353000.0, 351000.0])
        self.assertEqual(df["source"].tolist(), ["yfinance", "naver_ref_next", "naver_ref_next", "yfinance_unverified"])
        self.assertEqual({r["date"]: r["reason"] for r in rep}, {"2025-09-18": "불일치", "2025-09-19": "yfinance 없음"})

    def test_ref_next_ignored_across_naver_gap(self):
        cfg = dict(E.ASSETS["SKHYNIX"])
        days = ["2025-09-18", "2025-09-19", "2025-09-22"]
        yfd = pd.DataFrame({"date": days, "yf_close": [355500.0, 353000.0, 351000.0], "yf_adj": [355500.0, 353000.0, 351000.0]})
        nv = self._nv([("2025-09-18", 353000.0, 19500.0), ("2025-09-22", 351000.0, -2000.0)])   # 네이버에 9/19 없음
        df, rep = E.reconcile("SKHYNIX", cfg, days, yfd, nv)
        self.assertEqual(df.loc[0, "source"], "yfinance_unverified")   # 9/22 기준가(9/19 종가)로 9/18을 고치지 않는다
        self.assertEqual(rep, [])

    def test_missing_everywhere(self):
        cfg = dict(E.ASSETS["SOXX"])
        yfd = pd.DataFrame({"date": ["2026-10-20"], "yf_close": [500.01], "yf_adj": [500.01]})
        nv = self._nv([("2026-10-20", 500.01, -1.92)])
        with self.assertRaises(ValueError):
            E.reconcile("SOXX", cfg, ["2026-10-19", "2026-10-20"], yfd, nv)
        df, rep = E.reconcile("SOXX", cfg, ["2026-10-20", "2026-10-22"], yfd, nv)   # 마감 뒤 빈 날은 비워 둔다
        self.assertEqual(df["date"].tolist(), ["2026-10-20"])
        self.assertEqual(rep[0]["date"], "2026-10-22")

    def test_finality_clock(self):
        kst = ZoneInfo("Asia/Seoul")
        now = datetime(2026, 10, 22, 22, 44, tzinfo=kst)
        self.assertTrue(E.is_final("krx", "2026-10-22", now))
        self.assertFalse(E.is_final("nyse", "2026-10-22", now))      # 미국 장중
        self.assertFalse(E.is_final("krx", "2026-10-23", now))
        later = datetime(2026, 10, 24, 6, 30, tzinfo=kst)             # 10/23 17:30 ET
        self.assertTrue(E.is_final("nyse", "2026-10-23", later))

    def test_expected_dates_post_cutoff(self):
        cal = pd.DataFrame({"date": pd.to_datetime(["2026-10-20", "2026-10-21"]), "krx": [True, True], "nyse": [True, True]})
        now = datetime(2026, 10, 22, 22, 44, tzinfo=ZoneInfo("Asia/Seoul"))
        days, post = E.expected_dates("krx", cal, {"2026-10-22"}, now)
        self.assertEqual(days, ["2026-10-20", "2026-10-21", "2026-10-22"])
        self.assertEqual(post["2026-10-23"], "마감 전(비워 둠)")
        days, post = E.expected_dates("nyse", cal, {"2026-10-22"}, now)
        self.assertEqual(days, ["2026-10-20", "2026-10-21"])


if __name__ == "__main__":
    unittest.main()
