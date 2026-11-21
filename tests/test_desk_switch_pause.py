"""§5 전환 경로 멈춤 기준(2026-11-21 사용자 결정 '기본값 유지 + 멈춤 기준 고정')."""

import unittest

import numpy as np

from desk import alerts
from desk.state import account_gap


def ctx(**kw):
    base = {"account_drawdown": None, "model_drawdown": -0.10, "model_vs_etf_20d": 0.0, "model_vs_etf_120d": 0.05}
    base.update(kw)
    return base


class SwitchPauseRuleTest(unittest.TestCase):
    def test_frozen_values(self):
        self.assertEqual(alerts.BACKTEST_SWITCH_2026, {"mdd": -0.282, "gap20": -0.110})
        self.assertEqual(alerts.BACKTEST_2026, {"mdd": -0.303, "gap20": -0.081})  # 주간 기준은 그대로

    def test_account_uses_switch_limit_only_when_switch_orders_apply(self):
        # -34%: 전환 경로 기준(-33.2%)은 넘고, 주간(#774) 기준(-35.3%)은 넘지 않는다
        weekly = alerts.my_pause_rule(ctx(account_drawdown=-0.34))
        switch = alerts.my_pause_rule(ctx(account_drawdown=-0.34, switch_applied=True))
        self.assertFalse([r for r in weekly if r["level"] == "serious"])
        serious = [r for r in switch if r["level"] == "serious"]
        self.assertEqual(len(serious), 1)
        self.assertIn("내 계좌", serious[0]["title"])
        self.assertIn(alerts.SWITCH_BASIS, serious[0]["body"])

    def test_model_keeps_the_weekly_limit(self):
        out = alerts.my_pause_rule(ctx(model_drawdown=-0.36, switch_applied=True))
        self.assertTrue(any("7종목 전략(모의)" in r["title"] for r in out if r["level"] == "serious"))
        self.assertFalse(alerts.my_pause_rule(ctx(model_drawdown=-0.34, switch_applied=True)))

    def test_switch_gap_warning(self):
        hit = alerts.my_pause_rule(ctx(switch_applied=True, account_vs_etf_20d=-0.17))
        self.assertTrue(any("전환 경로 검증 범위 밖" in r["title"] for r in hit))
        self.assertFalse(alerts.my_pause_rule(ctx(switch_applied=True, account_vs_etf_20d=-0.15)))
        self.assertFalse(alerts.my_pause_rule(ctx(account_vs_etf_20d=-0.20)))  # 주간 주문이면 이 칸은 보지 않는다

    def test_describe_adds_switch_line(self):
        self.assertEqual(len(alerts.describe()), 3)
        lines = alerts.describe(switch_applied=True)
        self.assertEqual(len(lines), 4)
        self.assertIn("-33.2%", lines[-1])
        self.assertIn("-16.0%p", lines[-1])


class AccountGapTest(unittest.TestCase):
    def test_gap_on_last_day_both_priced(self):
        twr = np.linspace(1.0, 1.21, 22)
        etf = np.r_[np.linspace(100, 110, 21), np.nan]  # ETF not yet priced on the last US day
        got = account_gap(twr, etf, 20)
        want = (twr[20] / twr[0] - 1) - (etf[20] / etf[0] - 1)
        self.assertAlmostEqual(got, want)

    def test_short_history_is_none(self):
        self.assertIsNone(account_gap(np.ones(10), np.ones(10) * 100, 20))
        self.assertIsNone(account_gap([], [], 20))


if __name__ == "__main__":
    unittest.main()
