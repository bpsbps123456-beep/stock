"""급락 후 담기 contract: 세션별 기준 종가, 오류 체결 거르기, 연구와 같은 시각별 조건·정규장 규칙,
입력 검증과 캐시가 docs/desk_system.md §10대로 동작해야 한다. 네트워크 없이 합성 분봉으로 잰다."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from desk import config, dip

NY = config.NY


def minute_bars(day: str, segments, prev_close=100.0, prev_ah=None):
    """segments: [(시작 'HH:MM', 끝 'HH:MM', 시작가, 끝가)] — 그 사이를 1분봉 직선으로 잇는다.
    전 거래일 정규장(종가 prev_close)과 애프터(prev_ah, 없으면 prev_close)도 함께 만든다."""
    rows = []
    prev_day = (pd.Timestamp(day) - pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
    for d, segs in ((prev_day, [("09:30", "15:59", prev_close, prev_close),
                                ("16:00", "19:59", prev_close, prev_ah if prev_ah is not None else prev_close)]),
                    (day, segments)):
        for a, b, p0, p1 in segs:
            idx = pd.date_range(f"{d} {a}", f"{d} {b}", freq="1min", tz=NY)
            px = np.linspace(p0, p1, len(idx))
            for ts, p in zip(idx, px):
                rows.append((ts, p, p * 1.002, p * 0.998, p, 1000.0))
    df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"]).set_index("ts")
    return df[~df.index.duplicated()]


class QuoteTest(unittest.TestCase):
    def test_premarket_compares_with_the_last_close(self):
        q = {"symbol": "BULL", "marketState": "PRE", "regularMarketPrice": 7.285, "regularMarketPreviousClose": 7.0,
             "regularMarketTime": 100, "preMarketPrice": 5.0, "preMarketTime": 200}
        r = dip.classify_quote(q, now_epoch=1000)
        self.assertEqual(r["session"], "장전")
        self.assertEqual(r["ref"], 7.285)
        self.assertAlmostEqual(r["chg"], -31.37, places=2)

    def test_regular_and_after_hours_compare_with_the_previous_close(self):
        reg = dip.classify_quote({"symbol": "X", "marketState": "REGULAR", "regularMarketPrice": 90,
                                  "regularMarketPreviousClose": 100}, now_epoch=1)
        self.assertEqual((reg["session"], reg["chg"], reg["day_chg"]), ("장중", -10.0, -10.0))
        post = dip.classify_quote({"symbol": "X", "marketState": "POST", "regularMarketPrice": 90,
                                   "regularMarketPreviousClose": 100, "regularMarketTime": 10,
                                   "postMarketPrice": 81, "postMarketTime": 20}, now_epoch=1)
        self.assertEqual(post["session"], "장후")
        self.assertEqual(post["chg"], -19.0)         # 정규장 -10%에 장후 -10%를 더한 전전일 대비
        self.assertEqual(post["post_chg"], -10.0)
        self.assertIsNone(dip.classify_quote({"symbol": "X"}))

    def test_earnings_flag_only_for_the_last_36_hours(self):
        base = {"symbol": "X", "marketState": "REGULAR", "regularMarketPrice": 90, "regularMarketPreviousClose": 100}
        now = 1_800_000_000
        self.assertTrue(dip.classify_quote({**base, "earningsTimestamp": now - 12 * 3600}, now)["earnings"])
        self.assertFalse(dip.classify_quote({**base, "earningsTimestamp": now - 50 * 3600}, now)["earnings"])
        self.assertFalse(dip.classify_quote({**base, "earningsTimestamp": now + 30 * 86400}, now)["earnings"])


class PremarketTest(unittest.TestCase):
    def test_single_bad_print_does_not_become_the_premarket_low(self):
        bars = minute_bars("2026-10-02", [("04:00", "09:29", 99, 98)])
        bad = pd.Timestamp("2026-10-02 06:10", tz=NY)
        bars.loc[bad, ["Low", "Close"]] = [60.0, 60.0]   # 한 번 찍힌 -40%, 그 5분봉 종가는 정상
        s = dip.split_sessions(bars, date(2026, 10, 2))
        lo, _ = dip.pm_low(s["pm"], 100.0)
        self.assertIsNone(lo)                              # 5분봉 종가가 -7% 이하인 봉이 없다

    def test_real_drop_and_rebound(self):
        bars = minute_bars("2026-10-07", [("04:00", "06:59", 100, 99), ("07:00", "08:19", 99, 67.3),
                                          ("08:20", "09:29", 67.3, 80.2)])
        f = dip.features(bars, date(2026, 10, 7), 100.0, earnings=False)
        self.assertAlmostEqual(f["pm_low"], -32.8, delta=0.3)
        self.assertAlmostEqual(f["reb"], 19.4, delta=0.6)
        self.assertFalse(f["news_ah"])

    def test_hourly_condition_needs_two_confirmed_bars(self):
        # 전날 애프터 -12%(SMCI형) → 04시 봉부터 -15% 이하면 첫 봉에서 조건이 맞는다
        bars = minute_bars("2026-03-20", [("04:00", "09:29", 84, 80)], prev_ah=88.0)
        hours = dip.hourly_conditions(*[dip.split_sessions(bars, date(2026, 3, 20))[k] for k in ("pm", "prev_ah")], 100.0)
        self.assertEqual([h["at"] for h in hours], ["05:00", "06:00", "07:00", "08:00", "09:00", "09:29"])
        self.assertTrue(hours[0]["cond"])
        # 전날이 평온하고 07시에 처음 빠지면 그 봉만으로는 조건이 맞지 않는다(BULL형: 09:00부터)
        bars = minute_bars("2026-10-07", [("04:00", "06:59", 100, 99), ("07:00", "07:59", 99, 78), ("08:00", "09:29", 78, 74)])
        s = dip.split_sessions(bars, date(2026, 10, 7))
        h = {x["at"]: x["cond"] for x in dip.hourly_conditions(s["pm"], s["prev_ah"], 100.0)}
        self.assertFalse(h["08:00"])
        self.assertTrue(h["09:00"])

    def test_flat_carried_bars_do_not_confirm(self):
        bars = minute_bars("2026-10-07", [("04:00", "09:29", 80, 80)], prev_ah=88.0)
        bars["High"] = bars["Low"] = bars["Close"]          # 거래 없이 이어 찍힌 봉
        s = dip.split_sessions(bars, date(2026, 10, 7))
        self.assertFalse(any(h["cond"] for h in dip.hourly_conditions(s["pm"], s["prev_ah"], 100.0)))

    def test_flags_wait_until_their_time_and_follow_the_research_bands(self):
        day = "2026-10-07"
        bars = minute_bars(day, [("04:00", "06:30", 84, 70), ("06:31", "09:29", 70, 80)], prev_ah=99.0)
        early = datetime(2026, 10, 7, 7, 30, tzinfo=NY)
        f = dip.features(bars, date(2026, 10, 7), 100.0, earnings=False, now_et=early)
        self.assertIsNone(f["flag08"])
        self.assertIsNone(f["flag0929"])
        f = dip.features(bars, date(2026, 10, 7), 100.0, earnings=False)
        self.assertTrue(f["flag08"])                         # 08:00에 -15% 이하·저점 대비 5% 이상 반등
        self.assertTrue(f["flag0929"])                       # 당일 뉴스·실적 외·반등 10~20%
        self.assertFalse(dip.features(bars, date(2026, 10, 7), 100.0, earnings=True)["flag0929"])


class IntradayRuleTest(unittest.TestCase):
    def bars3(self):
        bars = minute_bars("2026-10-02", [("09:30", "09:35", 100, 95), ("09:36", "10:30", 95, 106), ("10:31", "15:59", 106, 104)])
        s = dip.split_sessions(bars, date(2026, 10, 2))
        return dip.resample(s["reg"], 3)

    def test_rules_enter_on_bar_closes_inside_the_awake_window(self):
        rows = {r["rule"]: r for r in dip.intraday_rules(self.bars3(), 3, next_open=105.0)}
        self.assertEqual(rows["open"]["at"], "09:30")
        self.assertEqual(rows["t1000"]["at"], "10:00")
        self.assertIsNotNone(rows["rev2"]["at"])
        self.assertLess(rows["rev2"]["entry_vs_open"], 0)    # 저점 반등 매수는 시가보다 싸게
        self.assertIsNotNone(rows["open"]["tp5_at"])          # 106까지 올랐다
        for r in rows.values():
            if r["at"]:
                self.assertLessEqual(r["at"], "11:00")

    def test_stop_signal_only_until_eleven(self):
        # 10:30까지 92로 밀리면 시가 매수의 -5%(95) 아래 마감 시각이 손절 신호다
        bars = minute_bars("2026-10-06", [("09:30", "09:40", 100, 103), ("09:41", "10:30", 103, 92), ("10:31", "15:59", 92, 92)])
        s = dip.split_sessions(bars, date(2026, 10, 6))
        rows = {r["rule"]: r for r in dip.intraday_rules(dip.resample(s["reg"], 3), 3, None)}
        self.assertIsNotNone(rows["open"]["stop_at"])
        self.assertLessEqual(rows["open"]["stop_at"], "11:00")
        self.assertIsNone(rows["open"]["tp5_at"])
        # 같은 낙폭이 11:00 뒤에 오면 사용자는 자고 있으므로 손절 신호가 없다
        bars = minute_bars("2026-10-06", [("09:30", "09:40", 100, 103), ("09:41", "11:10", 103, 101), ("11:11", "15:59", 101, 90)])
        s = dip.split_sessions(bars, date(2026, 10, 6))
        rows = {r["rule"]: r for r in dip.intraday_rules(dip.resample(s["reg"], 3), 3, None)}
        self.assertIsNone(rows["open"]["stop_at"])


class BottomHoldTest(unittest.TestCase):
    """바닥 15분 유지(연구 lowfind NL15_2): 저가 15분 무갱신 + 저가 +2% 안 + 전일 대비 -7% 이하, 09:45~11:00."""

    def reg3(self, segs, day="2026-10-02"):
        bars = minute_bars(day, segs)
        s = dip.split_sessions(bars, date.fromisoformat(day))
        return dip.resample(s["reg"], 3)

    def test_fires_after_fifteen_flat_minutes_near_the_low(self):
        # 09:30~09:50 100 → 88(-12%)로 빠진 뒤 88.5 근처에서 평평
        b = self.reg3([("09:30", "09:50", 100, 88), ("09:51", "15:59", 88.3, 88.6)])
        h = dip.bottom_hold(b, 100.0, 3)
        self.assertIsNotNone(h["signal_i"])
        self.assertGreaterEqual(h["signal_at"], "10:05")      # 저가 뒤 15분
        self.assertLessEqual(h["signal_at"], "11:00")
        rows = {r["rule"]: r for r in dip.intraday_rules(b, 3, None, 100.0)}
        self.assertEqual(rows["hold15"]["at"], h["signal_at"])

    def test_no_signal_when_far_from_low_shallow_or_after_eleven(self):
        far = self.reg3([("09:30", "09:40", 100, 88), ("09:41", "10:00", 88, 92), ("10:01", "15:59", 92, 93)])
        self.assertIsNone(dip.bottom_hold(far, 100.0, 3)["signal_i"])       # 저가 +4% 위
        shallow = self.reg3([("09:30", "09:50", 100, 96), ("09:51", "15:59", 96.2, 96.4)])
        self.assertIsNone(dip.bottom_hold(shallow, 100.0, 3)["signal_i"])   # -4%뿐
        late = self.reg3([("09:30", "11:20", 100, 99), ("11:21", "11:40", 99, 88), ("11:41", "15:59", 88.2, 88.4)])
        self.assertIsNone(dip.bottom_hold(late, 100.0, 3)["signal_i"])      # 11:00 뒤에야 바닥
        self.assertIsNone(dip.intraday_rules(late, 3, None, 100.0)[-1]["at"])

    def test_live_state_ignores_the_bar_still_forming(self):
        b = self.reg3([("09:30", "09:50", 100, 88), ("09:51", "15:59", 88.3, 88.6)])
        self.assertEqual(len(dip.closed_bars(b, 3, 10 * 60 + 44)), len(b[b.index.strftime("%H:%M") <= "10:41"]))
        f = dip.features(minute_bars("2026-10-02", [("09:30", "09:50", 100, 88), ("09:51", "10:04", 88.3, 88.6)]),
                         date(2026, 10, 2), 100.0, earnings=False, now_et=datetime(2026, 10, 2, 10, 4, tzinfo=NY))
        self.assertIsNone(f["flag_hold"])                     # 10:03에 끝난 봉까지는 저가 뒤 15분이 안 됐다

    def test_survival_lookup_follows_the_table(self):
        self.assertEqual(dip.survival_of(5), 12)
        self.assertEqual(dip.survival_of(20), 31)
        self.assertEqual(dip.survival_of(90), 50)
        self.assertEqual(dip.survival_of(300), 59)
        self.assertIsNone(dip.survival_of(None))


class GuardTest(unittest.TestCase):
    def test_symbol_and_date_are_checked_before_any_download(self):
        with mock.patch.object(dip, "minute_bars", side_effect=AssertionError("network")):
            with self.assertRaises(dip.DipError):
                dip.detail("12;", None)
            with self.assertRaises(dip.DipError):
                dip.detail("STX", "2020-01-02")
            with self.assertRaises(dip.DipError):
                dip.detail("STX", "10/02")

    def test_kst_shift_follows_us_daylight_saving(self):
        self.assertEqual(dip.kst_shift(date(2026, 10, 2)), 13)
        self.assertEqual(dip.kst_shift(date(2026, 12, 2)), 14)

    def test_common_stock_filter_and_news_links(self):
        self.assertTrue(dip.NOT_COMMON.search("NewAmsterdam Pharma Company N.V. Warrant"))
        self.assertTrue(dip.NOT_COMMON.search("NGL Energy Common Units representing"))
        self.assertFalse(dip.NOT_COMMON.search("BrightSpring Health Services"))
        rows = dip.news_rows({"data": {"rows": [{"title": "t", "url": "/articles/x", "publisher": "p", "created": "Oct 7"}]}})
        self.assertEqual(rows[0]["url"], "https://www.nasdaq.com/articles/x")

    def test_research_constants_cover_the_page(self):
        self.assertEqual([r[0] for r in dip.TIME_STATS], ["05:00", "06:00", "07:00", "08:00", "09:00", "09:29", "시가"])
        self.assertIn("10~20%", [b for b, _ in dip.REBOUND_STATS])
        self.assertIn(("STX", "2026-10-02"), [(c["t"], c["day"]) for c in dip.CASES])
        json.dumps(dip.research(), ensure_ascii=False)       # 화면으로 그대로 나간다


class CacheTest(unittest.TestCase):
    def test_scan_is_cached_for_two_minutes(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(dip, "CACHE_PATH", Path(tmp) / "dip.json"), \
                mock.patch.object(config, "CACHE", Path(tmp)), \
                mock.patch.object(dip, "compute", return_value={"rows": [], "now_et": "x"}) as compute:
            dip.payload()
            dip.payload()
            self.assertEqual(compute.call_count, 1)
            dip.payload(force=True)
            self.assertEqual(compute.call_count, 2)


if __name__ == "__main__":
    unittest.main()
