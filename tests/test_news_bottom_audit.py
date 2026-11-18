"""뉴스 공개·기록 시점, 겹치는 날 및 미성숙 결과 집계를 검증한다."""
import unittest

import pandas as pd

from desk import news
from desk.news_bottom_audit import _after_date, _counts, _ledger_dates


class NewsBottomAuditTest(unittest.TestCase):
    def setUp(self):
        self.days = news.market_calendar().sessions_in_range("2026-06-15", "2026-08-31")
        self.frame = pd.DataFrame({"in_dd": True, "low": 95.0, "close": 100.0}, index=self.days)

    def count(self, dates, cutoff, frame=None):
        return _counts(self.frame if frame is None else frame, dates, self.days[0], self.days[-1], cutoff, True)

    def test_outcome_window_must_mature_even_when_loss_is_known(self):
        day = self.days[4]
        self.frame.loc[self.days[7], "close"] = 94.0
        before = self.count([day], self.days[6])
        self.assertEqual((before["n"], before["pending"], before["early_breaks"]), (0, 1, []))
        pending = self.count([day], self.days[7])
        self.assertEqual((pending["n"], pending["pending"]), (0, 1))
        self.assertEqual(pending["early_breaks"][0]["break"], self.days[7].date().isoformat())
        final = self.count([day], self.days[24])
        self.assertEqual((final["hits"], final["n"], final["pending"]), (0, 1, 0))

    def test_event_windows_do_not_overlap_and_missing_outcomes_stay_missing(self):
        dates = self.days[[0, 1, 19, 20, 21]]
        out = self.count(dates, self.days[40])
        self.assertEqual((out["hits"], out["n"]), (2, 2))
        missing = self.count(dates, self.days[40], self.frame.drop(index=self.days[38]))
        self.assertEqual((missing["hits"], missing["n"], missing["missing"]), (1, 1, 1))

    def test_public_date_only_uses_session_after_publication(self):
        self.assertEqual(_after_date("2026-07-02"), pd.Timestamp("2026-07-06"))

    def test_ledger_uses_first_known_close_and_one_source(self):
        def event(eid, public, recorded):
            return dict(id=eid, episode="e", family="R", source_group="bank-note", time_precision="exact",
                        available_at=public, recorded_at=recorded)

        events = [event("note", "2026-11-02T18:00:00Z", "2026-11-02T22:00:00Z"),
                  event("repeat", "2026-11-04T16:00:00Z", "2026-11-04T17:00:00Z")]
        ledger = dict(episodes=[dict(id="e", benchmark="SOXX")], events=events)
        self.assertEqual(_ledger_dates(ledger, pd.Timestamp("2026-11-02T21:00:00Z")), set())
        self.assertEqual(_ledger_dates(ledger, pd.Timestamp("2026-11-05T21:00:00Z")), {pd.Timestamp("2026-11-03")})
