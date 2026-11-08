"""Independent price-path checks for the fixed-rule cross-stock transport study."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'ma_cross_stock', ROOT / 'data/derived/analysis/ma_entry_filter_20261108/cross_stock_review.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class SingleStockTransportTests(unittest.TestCase):
    def make_book(self):
        days = ['2025-12-31', '2026-01-02', '2026-01-05', '2026-01-06',
                '2026-01-07', '2026-01-08', '2026-01-09', '2026-01-12']
        opn = pd.DataFrame({'AAA': [100., 100., 100., 50., 200., 200., 200., 200.]}, index=days)
        cls = pd.DataFrame({'AAA': [100., 100., 80., 200., 200., 200., 200., 300.]}, index=days)
        book = SimpleNamespace(days=days, ticker={'AAA': 'AAA'}, opn=opn,
                               cls=cls, adj=cls.copy(), fx=np.ones(len(days)))
        averages = pd.DataFrame({'AAA': [100., 100., 90., 200., 200., 200., 200., 300.]}, index=days)
        return book, averages

    def test_buy_hold_is_fixed_quantity_price_over_initial_open(self):
        book, average = self.make_book()
        path, entries, exits, unknown = C.run_sleeve(book, 'AAA', 'buy_hold')
        # Three shares' worth of final price for every original dollar, without
        # any daily or weekly rescaling of the share count.
        np.testing.assert_allclose(path.close, [.8, 2., 2., 2., 2., 3.], rtol=0, atol=1e-12)
        self.assertAlmostEqual(path.value[path.days.index('2026-01-06')], .5)
        self.assertEqual(sum(e['side'] == 'buy' for e in path.events), 1)
        self.assertEqual(sum(e['side'] == 'sell' for e in path.events), 0)
        self.assertEqual(exits, [])
        self.assertEqual(unknown, [])

    def test_exit_pays_the_gap_then_waits_for_next_week_to_reenter(self):
        book, average = self.make_book()
        path, entries, exits, unknown = C.run_sleeve(book, 'AAA', 'exit', 2, average)
        np.testing.assert_allclose(path.close, [.8, .5, .5, .5, .5, .75], rtol=0, atol=1e-12)
        self.assertEqual([(e['signal_date'], e['session'], e['close'], e['ma'], e['open']) for e in exits],
                         [('2026-01-05', '2026-01-06', 80., 90., 50.)])
        self.assertEqual([e['date'] for e in path.events if e['side'] == 'buy'],
                         ['2026-01-05', '2026-01-12'])
        self.assertTrue(all(not path.weights[i] for i in range(1, 5)))
        self.assertEqual(unknown, [])

    def test_entry_only_does_not_sell_an_already_held_stock_below_ma(self):
        book, average = self.make_book()
        path, entries, exits, unknown = C.run_sleeve(book, 'AAA', 'entry_only', 2, average)
        np.testing.assert_allclose(path.close, [.8, 2., 2., 2., 2., 3.], rtol=0, atol=1e-12)
        self.assertEqual(exits, [])
        self.assertEqual(sum(e['side'] == 'sell' for e in path.events), 0)

    def test_equality_and_one_day_ma_are_no_op_controls(self):
        book, average = self.make_book()
        baseline = C.run_sleeve(book, 'AAA', 'buy_hold')[0]
        equal, entries, exits, unknown = C.run_sleeve(book, 'AAA', 'exit', 2, book.cls.copy())
        identity, e2, x2, u2 = C.run_sleeve(book, 'AAA', 'exit', 1)
        for path in [equal, identity]:
            np.testing.assert_allclose(path.close, baseline.close, rtol=0, atol=1e-12)
            self.assertEqual(path.weights, baseline.weights)
        self.assertEqual(exits, [])
        self.assertEqual(x2, [])

    def test_unknown_held_ma_is_reported_without_an_invented_sale(self):
        book, average = self.make_book()
        average.loc['2026-01-05', 'AAA'] = np.nan
        path, entries, exits, unknown = C.run_sleeve(book, 'AAA', 'exit', 2, average)
        self.assertEqual(exits, [])
        self.assertEqual([e['session'] for e in unknown], ['2026-01-06'])
        self.assertTrue(path.weights[path.days.index('2026-01-06')])
        self.assertAlmostEqual(path.close[-1], 3.)

    def test_future_closes_cannot_change_earlier_entry_or_exit_decisions(self):
        book, average = self.make_book()
        changed = copy.copy(book)
        changed.cls = book.cls.copy()
        changed.cls.loc[changed.cls.index >= '2026-01-06'] *= .01
        original, entries, exits, unknown = C.run_sleeve(book, 'AAA', 'exit', 2)
        altered, e2, x2, u2 = C.run_sleeve(changed, 'AAA', 'exit', 2)
        past = lambda events: [e for e in events if e['session'] <= '2026-01-06']
        self.assertEqual(past(entries), past(e2))
        self.assertEqual(past(exits), past(x2))
        holdings = lambda path: [w for day, w in zip(path.days, path.weights) if day <= '2026-01-06']
        self.assertEqual(holdings(original), holdings(altered))


class InitialCapitalBasketTests(unittest.TestCase):
    def test_matched_control_uses_actual_strategy_capital_at_previous_endpoint(self):
        days = ['2026-05-28', '2026-05-29', '2026-06-01', '2026-06-02']
        periods = {'front': [days[0], days[1]], 'back': [days[2], days[3]]}
        aggregate = [
            {'group': 'g', 'strategy': 'exit_SMA126', 'period': 'front', 'usd_return_pct': 11.},
            {'group': 'g', 'strategy': 'exit_SMA126', 'period': 'back', 'usd_return_pct': 8.},
            {'group': 'g', 'strategy': 'buy_hold', 'period': 'back', 'usd_return_pct': 100.},
        ]
        stock = [
            {'ticker': 'AAA', 'strategy': 'exit_SMA126', 'period': 'front', 'usd_matched_exposure_return_pct': 20.},
            {'ticker': 'BBB', 'strategy': 'exit_SMA126', 'period': 'front', 'usd_matched_exposure_return_pct': 0.},
            {'ticker': 'AAA', 'strategy': 'exit_SMA126', 'period': 'back', 'usd_matched_exposure_return_pct': 10.},
            {'ticker': 'BBB', 'strategy': 'exit_SMA126', 'period': 'back', 'usd_matched_exposure_return_pct': -10.},
            {'ticker': 'AAA', 'strategy': 'entry_only_SMA126', 'period': 'back', 'usd_matched_exposure_return_pct': 999.},
        ]
        curves = {'AAA__exit_SMA126': np.array([2., 4., 100., 100.]),
                  'BBB__exit_SMA126': np.array([1., 1., 1000., 1000.]),
                  'AAA__buy_hold': np.array([9., 1., 1., 1.]),
                  'BBB__buy_hold': np.array([1., 99., 99., 99.])}
        result = C.add_exposure_diagnostics(aggregate, stock, {'g': ['BBB', 'AAA']},
                                            curves, days, periods)
        # Front starts with one original dollar each: (20+0)/2=10. Back
        # inherits actual strategy May29 capital4:1: (4*10-10)/5=6.
        # Current June NAV, buyhold NAV and equal back weights all differ.
        self.assertAlmostEqual(result.loc[0, 'usd_matched_exposure_return_pct'], 10.)
        self.assertAlmostEqual(result.loc[0, 'usd_vs_matched_exposure_return_pp'], 1.)
        self.assertAlmostEqual(result.loc[1, 'usd_matched_exposure_return_pct'], 6.)
        self.assertAlmostEqual(result.loc[1, 'usd_vs_matched_exposure_return_pp'], 2.)
        self.assertTrue(np.isnan(result.loc[2, 'usd_matched_exposure_return_pct']))
        self.assertTrue(np.isnan(result.loc[2, 'usd_vs_matched_exposure_return_pp']))

    def test_exposure_uses_open_nav_without_altering_close_return(self):
        days = ['2026-01-05', '2026-01-06']
        nav = np.array([[2., 1.], [4., 1.]])
        open_nav = np.array([[1., 1.], [2., 1.]])
        exposure = np.array([[1., 0.], [1., 0.]])
        stats = C.basket_stats(nav, exposure, days, days[0], days[-1], open_nav=open_nav)
        self.assertAlmostEqual(stats['return_pct'], 150.)
        self.assertAlmostEqual(stats['mean_open_exposure_pct'], (.5 + 2/3)/2*100)

    def test_nav_averaging_does_not_rebalance_capital_across_sleeves(self):
        days = ['2026-01-05', '2026-01-06']
        nav = np.array([[2., 1.], [4., 1.]])
        exposure = np.array([[1., 0.], [1., 0.]])
        stats = C.basket_stats(nav, exposure, days, days[0], days[-1])
        # Initial equal dollars become (4+1)/2=2.5; redistributing capital
        # halfway would instead end at2.25. The cash sleeve remains one dollar.
        self.assertAlmostEqual(stats['return_pct'], 150.)
        self.assertAlmostEqual(stats['mdd_pct'], 0.)
        self.assertAlmostEqual(stats['mean_open_exposure_pct'], (2/3 + 4/5)/2*100)

    def test_back_period_uses_prior_endpoint_and_carries_the_same_sleeves(self):
        days = ['2026-05-29', '2026-06-01', '2026-06-02']
        nav = np.array([[2., 1.], [1., 1.], [.5, 1.]])
        exposure = np.array([[1., 0.], [1., 0.], [1., 0.]])
        stats = C.basket_stats(nav, exposure, days, '2026-06-01', '2026-06-02')
        self.assertAlmostEqual(stats['return_pct'], -50.)
        self.assertAlmostEqual(stats['mdd_pct'], -50.)
        self.assertAlmostEqual(stats['mean_open_exposure_pct'], (.5 + 1/3)/2*100)


if __name__ == '__main__':
    unittest.main()
