"""Regression tests for earnings date/EPS collection in earnings_prefill (no network)."""

from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

import earnings_prefill as ep

ET = ZoneInfo("America/New_York")
TODAY = date(2026, 9, 27)


class FakeTicker:
    def __init__(self, frame=None, error=None):
        self.frame = frame
        self.error = error

    def get_earnings_dates(self, limit=12):
        if self.error:
            raise self.error
        return self.frame


def earnings_frame(rows):
    index = pd.DatetimeIndex([ts for ts, *_ in rows], name="Earnings Date")
    return pd.DataFrame(
        [values for _, *values in rows],
        index=index,
        columns=["EPS Estimate", "Reported EPS", "Surprise(%)"],
    )


def event(days_from_today: int, reported: bool) -> dict:
    return {
        "date": TODAY + timedelta(days=days_from_today),
        "time_et": "16:00",
        "session": "장후",
        "eps_estimate": 1.0,
        "eps_reported": 1.1 if reported else None,
        "eps_surprise_pct": 10.0 if reported else None,
    }


class ClassifySessionTests(unittest.TestCase):
    def test_sessions_follow_us_market_hours(self) -> None:
        cases = {
            (16, 0): "장후",  # MU 등 장 마감 후 발표
            (15, 0): "장후",  # 11월 미확정 일정: 20:00Z가 EST로 15:00
            (8, 0): "장전",  # TSM
            (2, 0): "장전",  # TSM 과거 이벤트
            (10, 0): "장중",
        }
        for (hour, minute), expected in cases.items():
            ts = datetime(2026, 9, 30, hour, minute, tzinfo=ET)
            self.assertEqual(ep.classify_session(ts), expected, (hour, minute))


class FetchEarningsEventsTests(unittest.TestCase):
    def test_events_are_newest_first_with_nan_as_none(self) -> None:
        frame = earnings_frame([
            (datetime(2026, 6, 24, 16, 0, tzinfo=ET), 20.69, 25.11, 21.39),
            (datetime(2026, 9, 30, 16, 0, tzinfo=ET), 31.59, float("nan"), float("nan")),
            (datetime(2026, 3, 18, 16, 0, tzinfo=ET), 9.16, 12.20, 33.21),
        ])
        events = ep.fetch_earnings_events(FakeTicker(frame))

        self.assertEqual([e["date"] for e in events],
                         [date(2026, 9, 30), date(2026, 6, 24), date(2026, 3, 18)])
        upcoming, last = events[0], events[1]
        self.assertEqual(upcoming["time_et"], "16:00")
        self.assertEqual(upcoming["session"], "장후")
        self.assertIsNone(upcoming["eps_reported"])
        self.assertIsNone(upcoming["eps_surprise_pct"])
        self.assertEqual((last["eps_estimate"], last["eps_reported"]), (20.69, 25.11))

    def test_fetch_failure_or_empty_returns_empty_list(self) -> None:
        self.assertEqual(ep.fetch_earnings_events(FakeTicker(error=ValueError("404"))), [])
        self.assertEqual(ep.fetch_earnings_events(FakeTicker(frame=None)), [])
        self.assertEqual(ep.fetch_earnings_events(FakeTicker(frame=earnings_frame([]))), [])


class SelectEarningsEventContractTests(unittest.TestCase):
    """날짜 선택 규칙의 기준값과 무관하게 반드시 성립해야 하는 경우."""

    def test_day_after_report_picks_just_reported_quarter(self) -> None:
        # 2026-04 RMBS·GOOGL·META·STX: 발표 다음 날 실행했더니 다음 분기 날짜가 저장됨
        events = [event(+90, reported=False), event(-1, reported=True)]
        self.assertEqual(ep.select_earnings_event(events, TODAY), events[1])

    def test_between_quarters_picks_upcoming_report(self) -> None:
        events = [event(+10, reported=False), event(-80, reported=True)]
        self.assertEqual(ep.select_earnings_event(events, TODAY), events[0])

    def test_no_events_returns_none(self) -> None:
        self.assertIsNone(ep.select_earnings_event([], TODAY))


if __name__ == "__main__":
    unittest.main()
