"""Full requested grid and independent selection/cache/EMA behavior checks."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'ma_cross_stock_grid', ROOT / 'data/derived/analysis/ma_entry_filter_20261108/cross_stock_grid.py')
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)
C = G.C


class FullGridTests(unittest.TestCase):
    def test_requested_240_settings_are_complete_unique_and_include_identity_controls(self):
        specs = G.candidate_specs()
        self.assertEqual(len(specs), 241)
        self.assertEqual(specs[0], ('buy_hold', None, 0))  # matrix controls use baseline column0
        candidates = [(kind, length) for name, kind, length in specs if length]
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertEqual(set(candidates), {(kind, n) for kind in ['SMA', 'EMA'] for n in range(1, 121)})
        self.assertEqual(sum(length == 0 for name, kind, length in specs), 1)

    def test_front_selection_ignores_back_results_and_preserves_constrained_objective(self):
        base = {'strategy': 'buy_hold', 'kind': 'BASE', 'length': 0,
                'return_pct': 10., 'mdd_pct': -20., 'back_return_pct': 10.}
        rows = [base,
            {'strategy': 'exit_SMA1', 'kind': 'SMA', 'length': 1,
             'return_pct': 10., 'mdd_pct': -20., 'back_return_pct': 10.},
            {'strategy': 'exit_SMA12', 'kind': 'SMA', 'length': 12,
             'return_pct': 15., 'mdd_pct': -25., 'back_return_pct': -99.},
            {'strategy': 'exit_EMA18', 'kind': 'EMA', 'length': 18,
             'return_pct': 14.+5e-10, 'mdd_pct': -10., 'back_return_pct': 999.},
            {'strategy': 'exit_SMA18', 'kind': 'SMA', 'length': 18,
             'return_pct': 14., 'mdd_pct': -15., 'back_return_pct': -999.},
            {'strategy': 'exit_SMA7', 'kind': 'SMA', 'length': 7,
             'return_pct': 14., 'mdd_pct': -20., 'back_return_pct': -9999.}]
        self.assertEqual(G.select_front(rows)['strategy'], 'exit_SMA12')
        self.assertEqual(G.select_front(rows, baseline=base, require_mdd=True)['strategy'], 'exit_SMA7')
        changed = copy.deepcopy(rows)
        for i, row in enumerate(changed):
            row['back_return_pct'] = i*1e90
            row['back_mdd_pct'] = -i*1e90
            row['full_return_pct'] = -i*1e99
        self.assertEqual(G.select_front(changed)['strategy'], 'exit_SMA12')
        self.assertEqual(G.select_front(changed, baseline=base, require_mdd=True)['strategy'], 'exit_SMA7')

    def test_additional_external_group_is_disjoint_from_selection_stocks(self):
        universe = pd.DataFrame([
            {'ticker': 'A', 'in_time_jan2': True, 'in_koact_jan2': False,
             'in_ai_current': True, 'discovery_overlap': False},
            {'ticker': 'B', 'in_time_jan2': False, 'in_koact_jan2': True,
             'in_ai_current': False, 'discovery_overlap': False},
            {'ticker': 'C', 'in_time_jan2': True, 'in_koact_jan2': False,
             'in_ai_current': True, 'discovery_overlap': True},
            {'ticker': 'D', 'in_time_jan2': False, 'in_koact_jan2': False,
             'in_ai_current': True, 'discovery_overlap': False},
            {'ticker': 'E', 'in_time_jan2': False, 'in_koact_jan2': False,
             'in_ai_current': True, 'discovery_overlap': True},
        ])
        groups = G.study_groups(universe, ['A','B','C','D','E'])
        self.assertEqual(set(groups['additional_external']), {'D'})
        self.assertTrue(set(groups['additional_external']).isdisjoint(groups['jan2_external']))
        self.assertEqual(set(groups['additional_external']) | set(groups['jan2_external']),
                         set(groups['broad_external']))


class CachedColumnAndKindTests(unittest.TestCase):
    def make_book(self):
        days = pd.bdate_range('2025-05-01', '2026-01-16').strftime('%Y-%m-%d').tolist()
        close = np.full(len(days), 100.)
        close[days.index('2026-01-05')] = 200.
        close[days.index('2026-01-06'):] = 109.
        opn = np.r_[100., close[:-1]]
        cls = pd.DataFrame({'AAA': close}, index=days)
        return SimpleNamespace(days=days, ticker={'AAA':'AAA'}, cls=cls,
                               opn=pd.DataFrame({'AAA': opn}, index=days), adj=cls.copy(),
                               fx=np.ones(len(days)))

    def test_cached_and_uncached_all_path_state_match_at_grid_edges_and_middle(self):
        book = self.make_book()
        original_col = C.S._col
        for kind in ['SMA', 'EMA']:
            for length in [1, 18, 120]:
                with self.subTest(kind=kind, length=length):
                    plain = C.run_sleeve(book, 'AAA', 'exit', length, kind=kind)
                    cached = C.run_sleeve(book, 'AAA', 'exit', length, kind=kind, cache_columns=True)
                    p, q = plain[0], cached[0]
                    np.testing.assert_array_equal(q.value, p.value)
                    np.testing.assert_array_equal(q.close, p.close)
                    self.assertEqual(q.weights, p.weights)
                    self.assertEqual(q.events, p.events)
                    self.assertEqual(q.rebalances, p.rebalances)
                    self.assertEqual(q.selected, p.selected)
                    self.assertEqual(q.last, p.last)
                    self.assertEqual(cached[1:], plain[1:])
                    self.assertIs(C.S._col, original_col)

    def test_numeric_equality_tolerance_handles_entry_exit_and_material_breaks(self):
        book = self.make_book()
        price = 125.27999877929688  # ARM/SMA3 equality that exposed summation rounding
        book.cls.loc[:, 'AAA'] = price
        book.opn.loc[:, 'AAA'] = price
        book.adj = book.cls.copy()
        average = book.cls.copy()
        next_above = np.nextafter(price, np.inf)
        next_below = np.nextafter(price, -np.inf)
        average.loc['2026-01-02', 'AAA'] = next_above  # first-entry rounding equality
        average.loc['2026-01-05', 'AAA'] = next_above  # held-position rounding equality
        average.loc['2026-01-06', 'AAA'] = price*(1.+1e-8)  # economically larger break
        average.loc['2026-01-09', 'AAA'] = next_below  # mirrored side of equality: reentry
        average.loc['2026-01-12', 'AAA'] = next_above
        path, entries, exits, unknown = C.run_sleeve(
            book, 'AAA', 'exit', 3, average, comparison_rtol=1e-12)
        self.assertEqual([e['date'] for e in path.events if e['side'] == 'buy'],
                         ['2026-01-05', '2026-01-12'])
        self.assertEqual([e['session'] for e in exits], ['2026-01-07'])
        self.assertTrue(path.weights[path.days.index('2026-01-06')])
        self.assertEqual(path.weights[path.days.index('2026-01-07')], {})
        self.assertEqual(unknown, [])
        first = next(e for e in entries if e['session'] == '2026-01-05')
        self.assertLess(first['close'], first['ma'])
        self.assertTrue(first['allowed'])
        # Earlier studies retain exact comparisons unless they opt into the
        # rounding-equivalence tolerance used by this grid.
        exact = C.run_sleeve(book, 'AAA', 'exit', 3, average)[0]
        self.assertFalse(exact.weights[exact.days.index('2026-01-05')])

    def test_ema_factory_changes_the_actual_exit_decision(self):
        book = self.make_book()
        sma, se, sx, _ = C.run_sleeve(book, 'AAA', 'exit', 18, kind='SMA', cache_columns=True)
        ema, ee, ex, _ = C.run_sleeve(book, 'AAA', 'exit', 18, kind='EMA', cache_columns=True)
        # After a long100 seed, closes200 then109 yield SMA106.0556 but
        # EMA110.3657. Only EMA must sell at the next Jan7 open109.
        self.assertFalse(any(e['session'] == '2026-01-07' for e in sx))
        jan7 = [e for e in ex if e['session'] == '2026-01-07']
        self.assertEqual(len(jan7), 1)
        self.assertAlmostEqual(jan7[0]['open'], 109.)
        self.assertAlmostEqual(jan7[0]['ma'], 110.36565096952909)
        i = ema.days.index('2026-01-07')
        self.assertTrue(sma.weights[i])
        self.assertEqual(ema.weights[i], {})

    def test_one_day_sma_and_ema_match_unfiltered_hold(self):
        book = self.make_book()
        hold = C.run_sleeve(book, 'AAA', 'buy_hold')[0]
        for kind in ['SMA', 'EMA']:
            path, entries, exits, unknown = C.run_sleeve(
                book, 'AAA', 'exit', 1, kind=kind, cache_columns=True)
            np.testing.assert_array_equal(path.close, hold.close)
            self.assertEqual(path.weights, hold.weights)
            self.assertEqual(exits, [])
            self.assertEqual(unknown, [])

    def test_column_patch_restores_even_on_failure(self):
        book = self.make_book()
        original = C.S._col
        with self.assertRaises(RuntimeError):
            with C.cached_columns(book):
                np.testing.assert_array_equal(C.S._col(book, book.cls, 'AAA'), book.cls.AAA)
                raise RuntimeError('research failure')
        self.assertIs(C.S._col, original)

    def test_cached_ema_cannot_use_current_or_future_closes_for_open_decisions(self):
        book = self.make_book()
        altered = copy.copy(book)
        altered.cls = book.cls.copy()
        cutoff = '2026-01-06'
        altered.cls.loc[altered.cls.index >= cutoff] *= .01
        a, entries, exits, _ = C.run_sleeve(book, 'AAA', 'exit', 18, kind='EMA', cache_columns=True)
        b, e2, x2, _ = C.run_sleeve(altered, 'AAA', 'exit', 18, kind='EMA', cache_columns=True)
        past = lambda events: [e for e in events if e['session'] <= cutoff]
        self.assertEqual(past(entries), past(e2))
        self.assertEqual(past(exits), past(x2))
        before = lambda path: [w for d, w in zip(path.days, path.weights) if d <= cutoff]
        self.assertEqual(before(a), before(b))


if __name__ == '__main__':
    unittest.main()
