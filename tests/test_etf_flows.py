"""Network-free tests for quantity-based real-trade detection."""

from __future__ import annotations

import unittest

from gbrc.etf.flows import (
    cross_fund_consensus,
    format_flow_label,
    quantity_flows,
    split_factor,
    us_listed_top,
)


def snapshot(date: str, rows: dict[str, tuple[float, float]], cash: float = 0.0) -> dict:
    """rows: code -> (qty, price). Weights use one NAV denominator incl. cash."""
    values = {code: qty * price for code, (qty, price) in rows.items()}
    nav = sum(values.values()) + cash
    holdings = [
        {
            "code": code,
            "name": code,
            "asset_type": "security",
            "qty": qty,
            "value_krw": values[code],
            "weight": values[code] / nav * 100.0,
        }
        for code, (qty, _price) in rows.items()
    ]
    if cash:
        holdings.append({"code": "", "name": "Cash", "asset_type": "cash",
                         "qty": cash, "value_krw": cash, "weight": cash / nav * 100.0})
    return {"date": date, "as_of_date": date, "holdings": holdings}


BASE = {f"S{i} US EQUITY": (100.0, 10.0) for i in range(8)}


class CommonFactorTests(unittest.TestCase):
    def test_creation_scaling_is_not_a_trade_but_a_selective_add_is(self) -> None:
        after = {code: (qty * 1.10, price) for code, (qty, price) in BASE.items()}
        after["S0 US EQUITY"] = (100.0 * 1.10 * 1.30, 10.0)  # 공통 +10% 위에 30% 추가 매수
        result = quantity_flows([snapshot("2026-09-22", BASE), snapshot("2026-09-23", after)])

        self.assertAlmostEqual(result.common_factor, 1.10)
        self.assertEqual(result.flows["S0 US EQUITY"].kind, "매수")
        self.assertAlmostEqual(result.flows["S0 US EQUITY"].selective_pct, 30.0)
        self.assertEqual(result.flows["S1 US EQUITY"].kind, "유지")
        self.assertEqual(format_flow_label(result.flows["S1 US EQUITY"]), "가격효과")

    def test_price_rally_raises_weight_without_a_trade(self) -> None:
        after = dict(BASE)
        after["S2 US EQUITY"] = (100.0, 20.0)  # 가격만 두 배
        result = quantity_flows([snapshot("2026-09-22", BASE), snapshot("2026-09-23", after)])
        flow = result.flows["S2 US EQUITY"]
        self.assertGreater(flow.weight, flow.prev_weight)
        self.assertEqual(flow.kind, "유지")

    def test_common_factor_chains_over_every_day_in_the_window(self) -> None:
        day2 = {code: (qty * 1.05, price) for code, (qty, price) in BASE.items()}
        day3 = {code: (qty * 1.05 * 0.90, price) for code, (qty, price) in BASE.items()}
        day3["S3 US EQUITY"] = (100.0 * 1.05 * 0.90 * 0.70, 10.0)  # 선별 30% 매도
        path = [snapshot("2026-09-21", BASE), snapshot("2026-09-22", day2),
                snapshot("2026-09-23", day3)]
        result = quantity_flows(path)
        self.assertAlmostEqual(result.common_factor, 1.05 * 0.90)
        self.assertEqual(result.flows["S3 US EQUITY"].kind, "매도")
        self.assertAlmostEqual(result.flows["S3 US EQUITY"].selective_pct, -30.0)

    def test_too_few_holdings_leave_the_factor_unestimated(self) -> None:
        small = {"A US EQUITY": (10.0, 1.0), "B US EQUITY": (10.0, 1.0)}
        bigger = {"A US EQUITY": (20.0, 1.0), "B US EQUITY": (10.0, 1.0)}
        result = quantity_flows([snapshot("2026-09-22", small), snapshot("2026-09-23", bigger)])
        self.assertFalse(result.common_estimated)
        self.assertEqual(result.common_factor, 1.0)
        self.assertEqual(result.flows["A US EQUITY"].kind, "매수")


class SplitTests(unittest.TestCase):
    def test_split_with_a_big_same_day_move_is_detected(self) -> None:
        # KLAC 2026-06-15: 수량 x9.955, 가격 x0.112 (10:1 분할 + 당일 +12%)
        prev = {"qty": 2.24, "value_krw": 7_314_003}
        curr = {"qty": 22.3, "value_krw": 8_173_419}
        self.assertEqual(split_factor(prev, curr), 10.0)

    def test_split_is_not_reported_as_a_buy(self) -> None:
        after = dict(BASE)
        after["S4 US EQUITY"] = (400.0, 2.5)  # 4:1 분할
        result = quantity_flows([snapshot("2026-09-22", BASE), snapshot("2026-09-23", after)])
        flow = result.flows["S4 US EQUITY"]
        self.assertEqual(flow.split_factor, 4.0)
        self.assertEqual(flow.kind, "유지")
        self.assertEqual(format_flow_label(flow), "가격효과·분할")

    def test_real_quadrupling_without_price_drop_is_a_trade(self) -> None:
        prev = {"qty": 100.0, "value_krw": 1000.0}
        curr = {"qty": 400.0, "value_krw": 4000.0}
        self.assertEqual(split_factor(prev, curr), 1.0)


class EntryExitTests(unittest.TestCase):
    def test_new_exit_and_reentry(self) -> None:
        day1 = dict(BASE)
        day2 = {k: v for k, v in BASE.items() if k != "S5 US EQUITY"}
        day3 = dict(BASE)
        day3.pop("S6 US EQUITY")
        day3["NEW US EQUITY"] = (50.0, 10.0)
        result = quantity_flows([snapshot("2026-09-21", day1), snapshot("2026-09-22", day2),
                                 snapshot("2026-09-23", day3)])
        self.assertEqual(result.flows["NEW US EQUITY"].kind, "신규")
        self.assertEqual(result.flows["S6 US EQUITY"].kind, "편출")
        self.assertLess(result.flows["S6 US EQUITY"].trade_pp, 0)
        self.assertEqual(result.flows["S5 US EQUITY"].kind, "신규")  # 편출 후 재편입

    def test_futures_roll_is_neither_new_nor_exit(self) -> None:
        snap_before = snapshot("2026-09-17", BASE)
        snap_before["holdings"].append({"code": "NQU6 INDEX", "name": "NQ", "qty": 0,
                                        "value_krw": 50.0, "weight": 5.0})
        snap_after = snapshot("2026-09-18", BASE)
        snap_after["holdings"].append({"code": "NQZ6 INDEX", "name": "NQ", "qty": 0,
                                       "value_krw": 50.0, "weight": 5.0})
        result = quantity_flows([snap_before, snap_after])
        flow = result.flows["NQ FUTURES"]
        self.assertEqual(flow.status, "held")
        self.assertEqual(flow.kind, "N/A")  # 선물은 수량이 없어 실매매 판정에서 제외


class CrossFundTests(unittest.TestCase):
    def test_consensus_and_divergence_with_share_class_merge(self) -> None:
        a_after = dict(BASE)
        a_after["S0 US EQUITY"] = (150.0, 10.0)
        a_after["S1 US EQUITY"] = (50.0, 10.0)
        a_after["GOOG US EQUITY"] = (100.0, 10.0)
        a_before = dict(BASE, **{"GOOG US EQUITY": (80.0, 10.0)})
        b_before = dict(BASE, **{"GOOGL US EQUITY": (100.0, 10.0)})
        b_after = dict(BASE)
        b_after["S0 US EQUITY"] = (130.0, 10.0)
        b_after["S1 US EQUITY"] = (130.0, 10.0)
        b_after["GOOGL US EQUITY"] = (150.0, 10.0)
        a = quantity_flows([snapshot("2026-09-22", a_before), snapshot("2026-09-23", a_after)])
        b = quantity_flows([snapshot("2026-09-22", b_before), snapshot("2026-09-23", b_after)])
        buckets = cross_fund_consensus(a, b)
        common_buy = [company for company, _, _ in buckets["common_buy"]]
        divergence = [company for company, _, _ in buckets["divergence"]]
        self.assertIn("S0 US EQUITY", common_buy)
        self.assertIn("GOOG US EQUITY", common_buy)
        self.assertIn("S1 US EQUITY", divergence)

    def test_watch_list_is_us_listed_only(self) -> None:
        rows = dict(BASE, **{"009150": (1000.0, 10.0), "285A JP EQUITY": (900.0, 10.0)})
        snap = snapshot("2026-09-23", rows)
        snap["holdings"].append({"code": "NQZ6 INDEX", "name": "NQ", "qty": 0,
                                 "value_krw": 9_000.0, "weight": 50.0})
        top = us_listed_top(snap, 3)
        self.assertEqual(len(top), 3)
        self.assertTrue(all(row["code"].endswith(" US EQUITY") for row in top))


if __name__ == "__main__":
    unittest.main()
