"""Synthetic accounting and causality checks; no prices are downloaded here."""

from __future__ import annotations

import unittest

import numpy as np

from gbrc.etf.research_engine import (
    FAMILIES, GATES, MIXES, SIZES, FREQUENCIES, WEIGHTINGS, REGIMES,
    _simulate_one, build_rankings, config_grid, price_features, simulate_detail,
    simulation_inputs,
)


def bundle(days=35, securities=12):
    dates = np.busday_offset('2026-01-02', np.arange(days)).astype(str)
    rng = np.random.default_rng(7401)
    weights = rng.uniform(0.01, 0.1, (days, 2, securities))
    weights *= 0.9 / weights.sum(axis=2, keepdims=True)
    flow = rng.normal(size=(days, 6, 2, securities))
    kinds = np.sign(flow)
    flow[0] = np.nan
    kinds[0] = np.nan
    prices = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, (days, securities)), axis=0)
    return {
        'dates': dates, 'snapshot_dates': dates.copy(), 'weights': weights,
        'flow_pp': flow, 'flow_kind': kinds, 'cash': np.full((days, 2), 0.1),
        'prices': prices, 'tradable': np.ones_like(prices, dtype=bool),
        'price_age_days': np.zeros_like(prices), 'fx_age_days': np.zeros_like(prices),
        'benchmarks': np.repeat(prices[:, :1], 3, axis=1),
        'qqq_usd': prices[:, 0].copy(),
    }


def simulate_synthetic(order, returns=None, tradable=None, stale=None, *, size=1,
                       frequency=0, weighting=0, exposure=None, schedule=None,
                       conviction=None, inverse_vol=None, cost=0.0):
    """Manually prescribed ranks keep accounting tests independent of signal code."""
    order = np.asarray(order, dtype=np.int16)
    days = len(order)
    count = max(1, int(order.max()) + 1)
    ranks = np.full((1, days, 10), -1, dtype=np.int16)
    ranks[0, :, :order.shape[1]] = order
    returns = np.zeros((days, count)) if returns is None else np.asarray(returns, dtype=float)
    tradable = np.ones_like(returns, dtype=bool) if tradable is None else tradable
    stale = np.zeros_like(returns, dtype=bool) if stale is None else stale
    exposure = np.ones((days, 4)) if exposure is None else exposure
    schedule = np.ones((days, 3), dtype=bool) if schedule is None else schedule
    conviction = np.ones_like(ranks, dtype=float) if conviction is None else conviction
    inverse_vol = np.ones_like(ranks, dtype=float) if inverse_vol is None else inverse_vol
    return _simulate_one(
        np.array([0, size, frequency, weighting, 0], dtype=np.int32), ranks,
        conviction, inverse_vol, exposure, schedule, returns, tradable, stale,
        cost, 0, days, True,
    )


class ResearchCausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arrays = bundle()
        cls.ranking = build_rankings(cls.arrays)

    def test_snapshots_on_or_after_execution_date_never_change_earlier_signals(self):
        cutoff = 23
        changed = {key: value.copy() for key, value in self.arrays.items()}
        changed['weights'][cutoff:] = changed['weights'][cutoff:, :, ::-1]
        changed['flow_pp'][cutoff:] *= -100
        changed['flow_kind'][cutoff:] *= -1
        changed['cash'][cutoff:] = 0.95
        altered = build_rankings(changed)
        for key in ('ranks', 'conviction', 'inverse_vol'):
            np.testing.assert_array_equal(
                self.ranking[key][:, :cutoff + 1], altered[key][:, :cutoff + 1])
        np.testing.assert_array_equal(self.ranking['exposure'][:cutoff + 1],
                                      altered['exposure'][:cutoff + 1])
        self.assertEqual(self.ranking['signal_snapshot_index'][cutoff], cutoff - 1)

    def test_future_prices_cannot_change_earlier_ranks_or_portfolio_returns(self):
        cutoff = 24
        changed = {key: value.copy() for key, value in self.arrays.items()}
        changed['prices'][cutoff:] *= np.arange(1, 13)[None, :] * 50
        changed['qqq_usd'][cutoff:] *= 0.01
        altered = build_rankings(changed)
        for key in ('ranks', 'conviction', 'inverse_vol'):
            np.testing.assert_array_equal(
                self.ranking[key][:, :cutoff + 1], altered[key][:, :cutoff + 1])
        np.testing.assert_array_equal(self.ranking['exposure'][:cutoff + 1],
                                      altered['exposure'][:cutoff + 1])
        params = [0, 3, 0, 0, 0]
        base = simulate_detail(params, self.arrays, self.ranking)
        other = simulate_detail(params, changed, altered)
        np.testing.assert_array_equal(base['returns'][:cutoff], other['returns'][:cutoff])

    def test_future_invalid_price_does_not_erase_valid_past_returns(self):
        cutoff = 24
        changed = {key: value.copy() for key, value in self.arrays.items()}
        changed['prices'][cutoff:] = np.nan
        altered = build_rankings(changed)
        params = [0, 3, 0, 0, 0]
        base = simulate_detail(params, self.arrays, self.ranking)
        other = simulate_detail(params, changed, altered)
        np.testing.assert_array_equal(base['returns'][:cutoff], other['returns'][:cutoff])
        self.assertTrue(np.isnan(other['returns'][cutoff:]).all())

    def test_fewer_than_ten_universe_names_are_safe_and_never_duplicate(self):
        arrays = bundle(days=4, securities=2)
        ranking = build_rankings(arrays)
        result = simulate_detail([0, 10, 0, 0, 0], arrays, ranking, cost=0)
        self.assertTrue(np.isfinite(result['returns']).all())
        np.testing.assert_array_equal(result['counts'], [0, 2, 2, 2])
        for holdings in result['holdings']:
            active = holdings[holdings >= 0]
            self.assertEqual(len(active), len(set(active.tolist())))
            self.assertTrue(np.all(active < 2))

    def test_extra_delay_uses_strictly_earlier_snapshot_and_price_features(self):
        delayed = build_rankings(self.arrays, extra_lag=2)
        np.testing.assert_array_equal(delayed['signal_snapshot_index'][2:],
                                      self.ranking['signal_snapshot_index'][:-2])
        np.testing.assert_array_equal(delayed['ranks'][:, 2:], self.ranking['ranks'][:, :-2])
        np.testing.assert_array_equal(delayed['exposure'][2:], self.ranking['exposure'][:-2])

    def test_indicators_wait_for_real_history_and_exclude_execution_close(self):
        prices = np.arange(1.0, 26.0)[:, None]
        mom, ma, vol = price_features(prices)
        for values in (mom, ma, vol):
            self.assertTrue(np.isnan(values[:21]).all())
        self.assertEqual(mom[21, 0], 20.0)
        self.assertAlmostEqual(ma[21, 0], 21.0 / np.mean(np.arange(2, 22)) - 1)
        changed = prices.copy()
        changed[21:] *= 1000
        for old, new in zip((mom, ma, vol), price_features(changed)):
            np.testing.assert_array_equal(old[:22], new[:22])

    def test_single_fund_signal_does_not_require_unused_fund_data(self):
        arrays = bundle(days=4)
        arrays['flow_pp'][:] = np.nan
        arrays['flow_kind'][:] = np.nan
        arrays['flow_pp'][:, :, 0] = 1
        arrays['flow_kind'][:, :, 0] = 1
        result = build_rankings(arrays)
        rid = next(i for i, cfg in enumerate(result['rank_configs']) if cfg == {
            'family': 'buy_only', 'lookback': 1, 'time_mix': 1.0, 'gate': 'none'})
        self.assertTrue(np.all(result['ranks'][rid, 2] >= 0))
        joint = next(i for i, cfg in enumerate(result['rank_configs']) if cfg == {
            'family': 'buy_only', 'lookback': 1, 'time_mix': 0.5, 'gate': 'none'})
        self.assertTrue(np.all(result['ranks'][joint, 2] == -1))

    def test_calendar_week_month_and_iso_year_boundaries(self):
        arrays = bundle(days=7)
        arrays['dates'] = np.array(['2025-12-29', '2025-12-30', '2026-01-02',
                                   '2026-01-05', '2026-01-30', '2026-02-02', '2026-02-03'])
        arrays['snapshot_dates'] = arrays['dates'].copy()
        result = build_rankings(arrays)
        self.assertTrue(result['schedule'][:, 0].all())
        np.testing.assert_array_equal(result['schedule'][:, 1],
                                      [True, False, False, True, True, True, False])
        np.testing.assert_array_equal(result['schedule'][:, 2],
                                      [True, False, True, False, False, True, False])

    def test_config_grid_exhaustive_declared_dimensions_and_maximum_ten(self):
        expected = 6 * len(MIXES) * len(FAMILIES) * len(GATES)
        self.assertEqual(len(self.ranking['rank_configs']), expected)
        params, configs = config_grid(self.ranking['rank_configs'])
        self.assertEqual(len(params), 252720)
        self.assertEqual(len(configs), expected * len(SIZES) * len(FREQUENCIES)
                         * len(WEIGHTINGS) * len(REGIMES))
        self.assertLessEqual(params[:, 1].max(), 10)


class ResearchAccountingTests(unittest.TestCase):
    def test_buy_at_close_gets_no_entry_day_return(self):
        r, *_ = simulate_synthetic([[0], [0], [0]], [[0.50], [0.10], [0.20]])
        np.testing.assert_allclose(r, [0, 0.10, 0.20])

    def test_entry_and_full_switch_charge_both_legs_without_borrowing(self):
        fee = 0.015
        r, turn, counts, cash, holdings, allocations = simulate_synthetic(
            [[0], [1]], cost=fee)
        np.testing.assert_allclose(r, [1 / (1 + fee) - 1, (1 - fee) / (1 + fee) - 1])
        np.testing.assert_allclose(turn, [1 / (1 + fee), 2 / (1 + fee)])
        np.testing.assert_allclose(cash, 0, atol=1e-14)
        np.testing.assert_allclose(allocations.sum(axis=1), 1)
        self.assertTrue(np.all(cash >= 0))
        np.testing.assert_array_equal(counts, [1, 1])
        np.testing.assert_array_equal(holdings[:, 0], [0, 1])

    def test_rebalance_costs_use_drifted_dollars(self):
        fee = 0.01
        r, turnover, _, cash, _, allocation = simulate_synthetic(
            [[0, 1], [0, 1]], [[0, 0], [1, 0]], size=2, cost=fee)
        self.assertAlmostEqual(r[1], 0.5 - 0.5 * fee)
        self.assertAlmostEqual(turnover[1], 1 / 3)
        np.testing.assert_allclose(allocation[1, :2], [0.5, 0.5])
        self.assertGreaterEqual(cash.min(), 0)

    def test_nine_frozen_holdings_leave_exactly_one_new_slot(self):
        tradable = np.ones((2, 20), dtype=bool)
        tradable[1, :9] = False
        r, _, counts, cash, holdings, allocations = simulate_synthetic(
            [list(range(10)), list(range(10, 20))], tradable=tradable, size=10,
            cost=0.0015)
        self.assertTrue(np.isfinite(r).all())
        np.testing.assert_array_equal(counts, [10, 10])
        self.assertEqual(set(holdings[1]), set(range(9)) | {10})
        self.assertGreaterEqual(cash.min(), 0)
        np.testing.assert_allclose(allocations.sum(axis=1) + cash, 1)

    def test_closed_market_holding_cannot_be_liquidated_by_cash_rule(self):
        exposure = np.ones((2, 4))
        exposure[1] = 0
        tradable = np.array([[True], [False]])
        r, turnover, counts, cash, holdings, _ = simulate_synthetic(
            [[0], [0]], tradable=tradable, exposure=exposure)
        np.testing.assert_allclose(r, 0)
        self.assertEqual(turnover[1], 0)
        self.assertEqual(counts[1], 1)
        self.assertEqual(holdings[1, 0], 0)
        self.assertEqual(cash[1], 0)

    def test_weekly_portfolio_does_not_follow_midweek_rank_changes(self):
        schedule = np.zeros((4, 3), dtype=bool)
        schedule[[0, 3], 1] = True
        r, turnover, _, _, holdings, _ = simulate_synthetic(
            [[0], [1], [1], [1]], schedule=schedule, frequency=1)
        np.testing.assert_array_equal(holdings[:, 0], [0, 0, 0, 1])
        np.testing.assert_array_equal(turnover, [1, 0, 0, 2])

    def test_first_eligible_signal_can_initialize_between_calendar_boundaries(self):
        schedule = np.zeros((3, 3), dtype=bool)
        schedule[0] = True
        _, _, counts, _, holdings, _ = simulate_synthetic(
            [[-1], [0], [1]], frequency=2, schedule=schedule)
        np.testing.assert_array_equal(counts, [0, 1, 1])
        np.testing.assert_array_equal(holdings[:, 0], [-1, 0, 0])

    def test_stale_held_mark_invalidates_strategy_but_unused_stale_name_does_not(self):
        stale = np.array([[False, True], [True, True]])
        bad, *_ = simulate_synthetic([[0], [0]], returns=np.zeros((2, 2)), stale=stale)
        self.assertEqual(bad[0], 0)
        self.assertTrue(np.isnan(bad[1:]).all())
        stale[:, 0] = False
        good, *_ = simulate_synthetic([[0], [0]], returns=np.zeros((2, 2)), stale=stale)
        self.assertTrue(np.isfinite(good).all())

    def test_nan_mark_or_age_cannot_be_silently_treated_as_zero_return(self):
        arrays = bundle(days=3)
        ranking = build_rankings(arrays)
        arrays['prices'][2, 0] = np.nan
        arrays['price_age_days'][2, 0] = np.nan
        stale = simulation_inputs(arrays, ranking)[-1]
        self.assertTrue(stale[2, 0])


if __name__ == '__main__':
    unittest.main()
