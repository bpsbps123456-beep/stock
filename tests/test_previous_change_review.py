"""Network-free tests for the prior-day interpretation review."""

from __future__ import annotations

import unittest
from unittest import mock

import etf_crawler


def snapshot(
    date: str,
    *,
    cash: float | None,
    security_weights: dict[str, float],
) -> dict[str, object]:
    holdings = [
        {
            "code": f"{ticker} US EQUITY",
            "name": ticker,
            "weight": weight,
        }
        for ticker, weight in security_weights.items()
    ]
    if cash is not None:
        holdings.append(
            {
                "code": "",
                "name": "Cash",
                "asset_type": "cash",
                "weight": cash,
            }
        )
    return {
        "date": date,
        "holdings_count": len(holdings),
        "holdings": holdings,
    }


def d1_diff(previous_date: str, current_date: str) -> dict[str, object]:
    return {
        "prev_date": previous_date,
        "curr_date": current_date,
        "increased": [
            {
                "code": "NVDA US EQUITY",
                "delta": 5.0,
                "relative_pct": 10.0,
            }
        ],
        "decreased": [
            {
                "code": "MU US EQUITY",
                "delta": -9.0,
                "relative_pct": -60.0,
            }
        ],
        "new_in": [],
        "new_out": [],
    }


class PreviousChangeReviewTests(unittest.TestCase):
    def test_recomputes_cash_sector_and_top5_then_scores_next_day(self) -> None:
        previous_d5_date = "2026-07-02"
        previous_date = "2026-07-09"
        current_date = "2026-07-10"
        dates = [
            previous_d5_date,
            "2026-07-03",
            "2026-07-06",
            "2026-07-07",
            "2026-07-08",
            previous_date,
            current_date,
        ]

        previous_d5 = snapshot(
            previous_d5_date,
            cash=10.0,
            security_weights={"NVDA": 40.0, "MSFT": 40.0, "MU": 10.0},
        )
        previous = snapshot(
            previous_date,
            cash=5.0,
            security_weights={"NVDA": 50.0, "MSFT": 30.0, "MU": 15.0},
        )
        current = snapshot(
            current_date,
            cash=4.0,
            security_weights={"NVDA": 55.0, "MSFT": 35.0, "MU": 6.0},
        )
        snapshots = {
            previous_d5_date: previous_d5,
            "2026-07-03": snapshot(
                "2026-07-03", cash=9.0, security_weights={"NVDA": 91.0}
            ),
            "2026-07-06": snapshot(
                "2026-07-06", cash=8.0, security_weights={"NVDA": 92.0}
            ),
            "2026-07-07": snapshot(
                "2026-07-07", cash=7.0, security_weights={"NVDA": 93.0}
            ),
            "2026-07-08": snapshot(
                "2026-07-08", cash=6.0, security_weights={"NVDA": 94.0}
            ),
            previous_date: previous,
            current_date: current,
        }

        with mock.patch.object(
            etf_crawler,
            "load_snapshot",
            side_effect=snapshots.get,
        ) as load_snapshot:
            review = etf_crawler.format_previous_change_review(
                current,
                dates,
                {1: (previous_date, previous)},
                d1_diff(previous_date, current_date),
            )

        self.assertGreaterEqual(load_snapshot.call_count, 7)
        self.assertIn("리스크온 → 다음 날 -1.00%p (일치)", review)
        self.assertIn("AI 반도체 +10.00%p→+5.00%p 일치", review)
        self.assertIn("빅테크 -10.00%p→+5.00%p 불일치", review)
        self.assertIn(
            "전일 집중도 해석 검증: D-5 +5.00%p → 다음 날 +1.00%p (일치)",
            review,
        )

    def test_missing_d1_or_previous_snapshot_returns_comparison_message(self) -> None:
        current = snapshot(
            "2026-07-10",
            cash=4.0,
            security_weights={"NVDA": 96.0},
        )

        with mock.patch.object(etf_crawler, "load_snapshot") as load_snapshot:
            no_diff = etf_crawler.format_previous_change_review(
                current,
                ["2026-07-10"],
                {},
                None,
            )
            no_previous = etf_crawler.format_previous_change_review(
                current,
                ["2026-07-09", "2026-07-10"],
                {1: (None, None)},
                d1_diff("2026-07-09", "2026-07-10"),
            )

        load_snapshot.assert_not_called()
        self.assertIn("D-1 비교 스냅샷 없음", no_diff)
        self.assertIn("전일 해석 검증: 비교 스냅샷 없음", no_previous)

    def test_missing_cash_and_d5_snapshot_returns_na_safely(self) -> None:
        previous_date = "2026-07-09"
        current_date = "2026-07-10"
        previous = snapshot(
            previous_date,
            cash=None,
            security_weights={"NVDA": 50.0},
        )
        current = snapshot(
            current_date,
            cash=None,
            security_weights={"NVDA": 55.0},
        )
        snapshots = {previous_date: previous, current_date: current}

        with mock.patch.object(
            etf_crawler,
            "load_snapshot",
            side_effect=snapshots.get,
        ):
            review = etf_crawler.format_previous_change_review(
                current,
                [previous_date, current_date],
                {1: (previous_date, previous)},
                d1_diff(previous_date, current_date),
            )

        self.assertIn("전일 현금 해석 검증: N/A", review)
        self.assertIn("전일 섹터 해석 검증: 유의미한 D-5 선행 방향 없음", review)
        self.assertIn("전일 집중도 해석 검증: N/A", review)


if __name__ == "__main__":
    unittest.main()
