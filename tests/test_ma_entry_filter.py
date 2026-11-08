"""Protect the research gate's timing, portfolio scope and isolation."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'ma_entry_filter', ROOT / 'data/derived/analysis/ma_entry_filter_20261108/review.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class EntryFilterTests(unittest.TestCase):
    def test_sma_and_ema_full_history_and_manual_seed(self):
        price = pd.Series([10., 8., 12., 14.])
        sma = M.moving_average(price, 'SMA', 3)
        ema = M.moving_average(price, 'EMA', 3)
        self.assertTrue(sma.iloc[:2].isna().all())
        self.assertTrue(ema.iloc[:2].isna().all())
        self.assertAlmostEqual(sma.iloc[2], 10.)
        self.assertAlmostEqual(sma.iloc[3], 34./3)
        self.assertAlmostEqual(ema.iloc[2], 10.5)  # 10 -> 9 -> 10.5
        self.assertAlmostEqual(ema.iloc[3], 12.25)
        self.assertTrue(M.eligible(10., 10.))
        self.assertFalse(M.eligible(9., 10.))
        self.assertFalse(M.eligible(np.nan, 10.))
        self.assertFalse(M.eligible(10., np.nan))
        with self.assertRaises(ValueError):
            M.moving_average(price, 'invalid', 3)

    def test_held_exempt_and_blocked_allocation_is_cash(self):
        decision = {'selected': ['held', 'blocked', 'new'],
                    'target': {'held': .25, 'blocked': .20, 'new': .15},
                    'raw': {'held': .4, 'blocked': .3, 'new': .3}, 'ranked': ['held', 'blocked', 'new']}
        original = copy.deepcopy(decision)
        result, blocked = M.apply_gate(decision, {'held'},
            {'held': (5., 10.), 'blocked': (9., 10.), 'new': (10., 10.)})
        self.assertEqual(blocked, {'blocked'})
        self.assertEqual(result['selected'], ['held', 'new'])
        self.assertEqual(result['target'], {'held': .25, 'new': .15})
        self.assertAlmostEqual(sum(result['target'].values()), .4)
        self.assertEqual(decision, original)

    def test_prior_close_only_and_restore_on_exception(self):
        book = SimpleNamespace(
            cls=pd.DataFrame({'NEW': [10., 6., 1000.]}),
            ticker={'new': 'NEW'}, days=['2026-01-07', '2026-01-08', '2026-01-09'])
        decision = {'ranked': ['new'], 'selected': ['new'], 'target': {'new': .25}}
        original = M.S.weekly_target
        events = []
        with patch.object(M.S, 'weekly_target', return_value=decision):
            with self.assertRaises(RuntimeError):
                with M.entry_gate(book, 'SMA', 2, events):
                    filtered = M.S.weekly_target(book, 2, book.days[2], set())
                    self.assertEqual(filtered['target'], {})
                    self.assertEqual(events[0]['signal_date'], book.days[1])
                    self.assertEqual(events[0]['close'], 6.)
                    self.assertEqual(events[0]['ma'], 8.)
                    raise RuntimeError('research failure')
        self.assertIs(M.S.weekly_target, original)


class EntryFilterEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.book = M.R.load_book(M.BASE / 'inputs', M.END)
        cls.baseline = M.S.run_model(cls.book)

    def test_always_allowed_reproduces_operating_path(self):
        events = []
        with patch.object(M, 'eligible', return_value=True):
            with M.entry_gate(self.book, 'SMA', 20, events):
                path = M.S.run_model(self.book)
        np.testing.assert_array_equal(path.close, self.baseline.close)
        self.assertEqual(path.rebalances, self.baseline.rebalances)
        self.assertEqual(path.selected, self.baseline.selected)
        self.assertTrue(events)
        self.assertTrue(all(e['allowed'] for e in events))

    def test_future_prices_cannot_change_earlier_decisions(self):
        changed = copy.copy(self.book)
        changed.cls = self.book.cls.copy()
        cutoff = '2026-07-01'
        changed.cls.loc[changed.cls.index >= cutoff] *= .1
        original_events, changed_events = [], []
        with M.entry_gate(self.book, 'EMA', 20, original_events):
            original_path = M.S.run_model(self.book)
        with M.entry_gate(changed, 'EMA', 20, changed_events):
            changed_path = M.S.run_model(changed)
        before = lambda events: [e for e in events if e['session'] <= cutoff]
        # A July 1 open decision, if any, also cannot use the July 1 close.
        self.assertEqual(before(original_events), before(changed_events))
        prior_targets = lambda path: {d: w for d, w in path.rebalances.items() if d <= cutoff}
        self.assertEqual(prior_targets(original_path), prior_targets(changed_path))
        self.assertEqual(M.validate(self.book, original_path, original_events), 'passed')


if __name__ == '__main__':
    unittest.main()
