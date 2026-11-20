"""데스크 7종목 순위 점수(2026-11-21 사용자 결정): 60일 잔차 모멘텀. docs/desk_system.md §3."""

import unittest

import numpy as np

from desk import config
from desk import strategies as S


class ResidualMomentumTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.q = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
        self.qr = np.r_[np.nan, np.diff(np.log(self.q))]

    def stock(self, beta, alpha):
        r = beta * np.nan_to_num(self.qr) + alpha
        return 50 * np.exp(np.cumsum(r))

    def test_removes_the_market_part(self):
        k = 150
        pure = S.residual_momentum(self.stock(2.0, 0.0), self.q, k, 60)
        drift = S.residual_momentum(self.stock(2.0, 0.001), self.q, k, 60)
        self.assertAlmostEqual(pure, 0.0, places=9)        # 시장을 2배로 따라간 몫은 0
        self.assertAlmostEqual(drift, 0.06, places=6)      # 하루 0.1% 고유 상승 × 60일

    def test_uses_only_closes_through_k_minus_1(self):
        px = self.stock(1.0, 0.002)
        k = 150
        base = S.residual_momentum(px, self.q, k, 60)
        later = px.copy()
        later[k:] *= 3.0                                    # k 종가부터 바꿔도 결과는 같아야 한다
        self.assertEqual(base, S.residual_momentum(later, self.q, k, 60))

    def test_needs_ninety_percent_of_the_window(self):
        px = self.stock(1.0, 0.001)
        k = 150
        holes = px.copy()
        holes[k - 20:k - 13] = np.nan                       # 7개 가격이 빠지면 수익 8개가 빠져 52/60 < 54
        self.assertTrue(np.isnan(S.residual_momentum(holes, self.q, k, 60)))
        self.assertTrue(np.isnan(S.residual_momentum(px, self.q, 30, 60)))  # 이력 부족

    def test_rule_uses_residual_and_774_keeps_return(self):
        self.assertEqual((config.RULE["score"], config.RULE["lookback"]), ("res", 60))
        self.assertNotIn("score", config.CODEX_774_RULE)   # #774 재생은 120일 수익률
        self.assertEqual(config.CODEX_774_RULE["lookback"], 120)


if __name__ == "__main__":
    unittest.main()
