"""Network-free tests for holding identity, ticker mapping, and price dates."""

from __future__ import annotations

import unittest

from gbrc.etf.market import (
    company_key,
    holding_key,
    listing_market,
    previous_session,
    price_reference_date,
    return_between,
    select_base_date,
    yahoo_ticker,
)

# KRX sessions around real 2026 holidays: 9/24-9/26 추석, 10/3 개천절(토).
SESSIONS = [
    "2026-09-04", "2026-09-07", "2026-09-08", "2026-09-09",
    "2026-09-17", "2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23",
    "2026-09-28", "2026-09-29",
]


class HoldingIdentityTests(unittest.TestCase):
    def test_case_and_whitespace_differences_are_the_same_holding(self) -> None:
        self.assertEqual(holding_key("NQM6 Index"), holding_key("NQM6  INDEX"))

    def test_futures_roll_keeps_one_identity(self) -> None:
        self.assertEqual(holding_key("NQU6 INDEX"), holding_key("NQZ6 INDEX"))
        self.assertEqual(holding_key("NQZ6 INDEX"), "NQ FUTURES")

    def test_ticker_change_alias_keeps_continuity(self) -> None:
        self.assertEqual(holding_key("SKHYV US EQUITY"), "SKHY US EQUITY")

    def test_share_classes_merge_only_for_cross_fund_matching(self) -> None:
        self.assertNotEqual(holding_key("GOOGL US EQUITY"), holding_key("GOOG US EQUITY"))
        self.assertEqual(company_key("GOOGL US EQUITY"), company_key("GOOG US EQUITY"))


class TickerMappingTests(unittest.TestCase):
    def test_every_listing_market_in_the_two_funds_is_priceable(self) -> None:
        cases = {
            "INTC US EQUITY": ("US", "INTC"),
            "MOG/A US EQUITY": ("US", "MOG-A"),
            "009150": ("KR", "009150.KS"),
            "0043Y0": ("KR", "0043Y0.KS"),
            "285A JP EQUITY": ("JP", "285A.T"),
            "100 HK EQUITY": ("HK", "0100.HK"),
            "300502 CH EQUITY": ("CN", "300502.SZ"),
            "601138 CH EQUITY": ("CN", "601138.SS"),
            "NQZ6 INDEX": ("FUT", "NQZ26.CME"),
        }
        for code, (expected_market, expected_ticker) in cases.items():
            with self.subTest(code=code):
                self.assertEqual(listing_market(code), expected_market)
                self.assertEqual(yahoo_ticker(code, "2026-09-23"), expected_ticker)

    def test_single_digit_futures_year_resolves_to_nearest_decade(self) -> None:
        self.assertEqual(yahoo_ticker("NQH0 INDEX", "2029-12-01"), "NQH30.CME")
        self.assertEqual(yahoo_ticker("NQZ9 INDEX", "2030-01-05"), "NQZ29.CME")

    def test_unknown_codes_and_cash_are_not_priced(self) -> None:
        self.assertIsNone(yahoo_ticker(""))
        self.assertIsNone(yahoo_ticker("KRD010010001"))


class PriceReferenceDateTests(unittest.TestCase):
    def test_domestic_uses_previous_session_and_foreign_the_close_before_it(self) -> None:
        # PDF 9/23 = 9/22 KRX 세션 마감 기준.
        self.assertEqual(price_reference_date("009150", "2026-09-23", SESSIONS), "2026-09-22")
        # 해외는 9/22보다 앞선 마지막 현지 종가 → on-or-before 9/21.
        self.assertEqual(
            price_reference_date("INTC US EQUITY", "2026-09-23", SESSIONS), "2026-09-21"
        )

    def test_monday_snapshot_looks_through_the_weekend(self) -> None:
        # PDF 9/21(월) = 9/18(금) 세션 → 해외는 9/17 이전 종가.
        self.assertEqual(
            price_reference_date("INTC US EQUITY", "2026-09-21", SESSIONS), "2026-09-17"
        )

    def test_after_korean_holiday_the_last_session_is_before_the_holiday(self) -> None:
        # 추석 연휴 뒤 첫 PDF 9/28 = 9/23 세션 마감 기준.
        self.assertEqual(price_reference_date("009150", "2026-09-28", SESSIONS), "2026-09-23")
        self.assertEqual(
            price_reference_date("INTC US EQUITY", "2026-09-28", SESSIONS), "2026-09-22"
        )

    def test_weekday_fallback_without_sessions(self) -> None:
        self.assertEqual(previous_session("2026-09-21", 1), "2026-09-18")


class SessionCountingTests(unittest.TestCase):
    def test_d_n_counts_krx_sessions_not_weekdays(self) -> None:
        # 9/28 기준 D-1은 연휴 전 9/23, 평일 계산이면 9/25(휴장)가 나온다.
        self.assertEqual(previous_session("2026-09-28", 1, SESSIONS), "2026-09-23")
        self.assertEqual(previous_session("2026-09-28", 3, SESSIONS), "2026-09-21")

    def test_missing_snapshot_falls_back_to_the_latest_earlier_one(self) -> None:
        snapshots = ["2026-09-17", "2026-09-18", "2026-09-22", "2026-09-23"]
        intended, used = select_base_date(snapshots, "2026-09-23", 2, SESSIONS)
        self.assertEqual(intended, "2026-09-21")
        self.assertEqual(used, "2026-09-18")

    def test_short_history_uses_the_earliest_snapshot(self) -> None:
        intended, used = select_base_date(["2026-09-22", "2026-09-23"], "2026-09-23", 5)
        self.assertEqual(used, "2026-09-22")
        self.assertIsNotNone(intended)


class ReturnTests(unittest.TestCase):
    def test_return_needs_two_distinct_closes(self) -> None:
        series = {"2026-09-18": 100.0, "2026-09-21": 110.0}
        self.assertAlmostEqual(return_between(series, "2026-09-18", "2026-09-21"), 10.0)
        # 휴장으로 두 기준일이 같은 종가를 가리키면 N/A(None)다.
        self.assertIsNone(return_between(series, "2026-09-19", "2026-09-20"))
        self.assertIsNone(return_between({}, "2026-09-18", "2026-09-21"))


if __name__ == "__main__":
    unittest.main()
