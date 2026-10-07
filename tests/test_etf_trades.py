"""Network-free tests for the real-trade timing analysis (gbrc etf trades)."""

from __future__ import annotations

import unittest
from datetime import date, timedelta

from gbrc.etf import market, trades

NAMES = ["A", "B", "C", "D", "E", "F"]
FX = 1000.0


def weekdays(start: str, end: str) -> list[str]:
    out, day = [], date.fromisoformat(start)
    while day <= date.fromisoformat(end):
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


SESSIONS = weekdays("2026-08-24", "2026-09-11")  # KRX 세션 = 스냅샷 날짜
US_DAYS = weekdays("2026-08-17", "2026-10-16")


def us_price(name: str, day: str) -> float:
    # A는 8/31 미국장에서 +10% 점프: 매니저 평가 종가(8/28)와 추종 매수일(9/1) 사이.
    if name == "A" and day >= "2026-08-31":
        return 110.0
    return 100.0


PRICES = {
    **{name: {d: us_price(name, d) for d in US_DAYS} for name in NAMES},
    "QQQ": {d: 400.0 for d in US_DAYS},
    "KRW=X": {d: FX for d in US_DAYS},
    "456600.KS": {d: 10_000.0 + i for i, d in enumerate(SESSIONS)},
}


def build_snapshots() -> dict[str, dict]:
    snapshots = {}
    for as_of in SESSIONS:
        holdings = []
        for name in NAMES:
            code = f"{name} US EQUITY"
            qty = 150.0 if (name == "A" and as_of >= "2026-09-01") else 100.0
            ref = market.price_reference_date(code, as_of, SESSIONS)
            close = market.close_on_or_before(PRICES[name], ref)[1]
            holdings.append({"code": code, "name": name, "asset_type": "security",
                             "qty": qty, "value_krw": qty * close * FX})
        nav = sum(h["value_krw"] for h in holdings)
        for h in holdings:
            h["weight"] = h["value_krw"] / nav * 100.0
        snapshots[as_of] = {"date": as_of, "as_of_date": as_of, "holdings": holdings}
    return snapshots


class TradeAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshots = build_snapshots()

    def test_buy_timing_is_measured_from_manager_close_and_follower_entry(self) -> None:
        events = trades.collect_events(self.snapshots, PRICES, SESSIONS)
        buys = [e for e in events if e["kind"] == "매수"]
        self.assertEqual(len(buys), 1)
        buy = buys[0]
        self.assertEqual((buy["date"], buy["key"]), ("2026-09-01", "A US EQUITY"))
        self.assertAlmostEqual(buy["mgr5_excess"], 0.10)  # 평가 종가 8/28 → 점프 포함
        self.assertAlmostEqual(buy["fol5_excess"], 0.0)  # 9/1 매수 → 점프 이후

        stats = {(row["side"], row["window"]): row for row in trades.event_stats(events)}
        row = stats[("매수·신규", 5)]
        self.assertAlmostEqual(row["mgr_mean_pct"], 10.0)
        self.assertEqual(row["mgr_hit_pct"], 100.0)
        self.assertEqual(row["fol_hit_pct"], 0.0)

    def test_buying_before_the_jump_shows_positive_trade_effect(self) -> None:
        segments = trades.build_segments(self.snapshots, PRICES, SESSIONS)
        months = {row["month"]: row for row in trades.monthly_breakdown(
            self.snapshots, segments, PRICES, SESSIONS)}
        september = months["2026-09"]
        self.assertAlmostEqual(september["static_pct"], 10.0 / 6, places=4)
        self.assertGreater(september["actual_pct"], september["static_pct"])
        self.assertGreater(september["trade_effect_pp"], 0.5)
        self.assertEqual(sum(len(s.missing) for s in segments), 0)

    def test_top_n_backtest_and_etf_reference(self) -> None:
        result = trades.backtest_top_n(self.snapshots, PRICES, top_n=1, every=20,
                                       weighting="fund", end_date="2026-09-11")
        self.assertAlmostEqual(result["return_pct"], 10.0)  # 동률 비중이면 키 순서로 A
        self.assertEqual(result["max_drawdown_pct"], 0.0)
        refs = trades.reference_returns(self.snapshots, PRICES, "456600.KS", "2026-09-11")
        self.assertGreater(refs["etf"]["return_pct"], 0)
        self.assertAlmostEqual(refs["qqq_krw"]["return_pct"], 0.0)

    def test_local_holiday_between_valuations_is_zero_not_missing(self) -> None:
        prices = {"A": {"2026-09-03": 100.0, "2026-09-08": 105.0}, "KRW=X": {"2026-09-01": FX}}
        # 9/7(월) 미국 휴장처럼 두 PDF(9/8, 9/9)가 같은 9/3 종가를 가리키는 구간.
        self.assertEqual(
            trades.market_return("A US EQUITY", "2026-09-08", "2026-09-09", prices, SESSIONS),
            0.0,
        )

    def test_report_renders_every_section(self) -> None:
        report, summary = trades.analyze("456600", "TIME fixture", "456600.KS", "2026-09-11",
                                         3, snapshots=self.snapshots, prices=PRICES)
        for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5."):
            self.assertIn(heading, report)
        self.assertIn("주문표가 아닙니다", report)
        self.assertEqual(summary["missing_segments"], 0)


if __name__ == "__main__":
    unittest.main()
