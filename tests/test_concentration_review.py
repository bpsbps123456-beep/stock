"""Protect research isolation and the cash comparator's actual meaning."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('concentration_review', ROOT / 'data/derived/analysis/concentration_review_20261105/review.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


class ConcentrationReviewTests(unittest.TestCase):
    def test_cash_comparator_and_activation(self):
        raw = {'a': .8, 'b': .1, 'c': .1}
        sectors = {'a': 'one', 'b': 'two', 'c': 'three'}
        original = R.S.project_caps
        rule = dict(R.config.RULE)
        def weekly(book, k, session, current):
            return R.S.project_caps(raw, sectors)
        with patch.object(R.S, 'weekly_target', weekly):
            with R.variant({'stock_cap': .25, 'raw_ceiling': True}, '2026-11-09'):
                before = R.S.weekly_target(None, 0, '2026-11-02', set())
                after = R.S.weekly_target(None, 0, '2026-11-09', set())
        self.assertEqual(before, original(raw, sectors))
        self.assertAlmostEqual(after['a'], .25)
        self.assertAlmostEqual(after['b'], .1)
        self.assertAlmostEqual(sum(after.values()), .45)
        self.assertEqual(R.config.RULE, rule)
        self.assertIs(R.S.project_caps, original)

    def test_exception_restores_engine_and_initial_peak_counts(self):
        original = R.S.project_caps
        with self.assertRaises(RuntimeError):
            with R.variant({'stock_cap': .15, 'raw_ceiling': False}):
                raise RuntimeError('research failure')
        self.assertIs(R.S.project_caps, original)
        result = R.metrics([100, 90, 95])
        self.assertAlmostEqual(result['mdd_pct'], -10)
        self.assertAlmostEqual(result['return_pct'], -5)

    def test_forward_uses_only_first_sixty_and_preserves_operating_rule(self):
        from types import SimpleNamespace
        days = R.pd.bdate_range('2026-11-09', periods=65).strftime('%Y-%m-%d').tolist()
        paths = {'baseline': SimpleNamespace(days=days)}
        result = {'rows': {'baseline': {'return_pct': 1., 'mdd_pct': -5.},
                           'cap20': {'return_pct': 1., 'mdd_pct': -4.}}}
        with patch.object(R, 'compare', return_value=result) as compare:
            decision = R.forward_decision(None, paths, days[-1])
        self.assertEqual(compare.call_args.args[-1], days[59])
        self.assertEqual(decision['status'], 'candidate_only')
        self.assertFalse(decision['operating_rule_changed'])
        pending = R.forward_decision(None, paths, days[19])
        self.assertEqual(pending['status'], 'observation_only')
