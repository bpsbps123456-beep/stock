"""Independent execution-order and state tests for the optional research exit."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from desk import strategies as S


class ExitAtOpenHookTests(unittest.TestCase):
    def make_book(self):
        # Dec 31 is signal warmup. Jan 6 is a Tuesday: a sale must not reset
        # the weekly schedule and cause a Jan 7 purchase.
        days = ['2025-12-31', '2026-01-02', '2026-01-05', '2026-01-06',
                '2026-01-07', '2026-01-08', '2026-01-09', '2026-01-12']
        opn = pd.DataFrame({'AAA':[100.,100.,100.,50.,200.,200.,200.,200.],
                            'BBB':[100.]*8},index=days)
        cls = opn.copy()
        cls.loc['2026-01-05','AAA'] = 80.
        return SimpleNamespace(days=days,ticker={'a':'AAA','b':'BBB','unheld':'CCC'},
                               opn=opn,cls=cls)

    def decision(self, selected=('a','b'), target=None):
        return {'pdf':'test','ranked':[{'key':'a'},{'key':'b'}],
                'selected':list(selected),'raw':{'a':.2,'b':.2},
                'target':dict(target if target is not None else {'a':.2,'b':.2})}

    def sell_tuesday(self, book, k, day, holdings):
        return {'a'} if day == '2026-01-06' else set()

    def test_overnight_loss_paid_before_exit_and_survivor_not_rescaled(self):
        book = self.make_book()
        with patch.object(S,'weekly_target',side_effect=lambda *args:self.decision()):
            path = S.run_model(book,exit_at_open=self.sell_tuesday)
        tuesday = path.days.index('2026-01-06')
        # Portfolio began with 20% AAA. Its 100 -> 50 move must cost 10%
        # before the sale; liquidation cannot avoid an already-opened gap.
        self.assertAlmostEqual(path.value[tuesday],.9)
        self.assertAlmostEqual(path.close[tuesday],.9)
        self.assertNotIn('a',path.weights[tuesday])
        self.assertAlmostEqual(path.weights[tuesday]['b'],.2/.9)
        self.assertAlmostEqual(1-sum(path.weights[tuesday].values()),.7/.9)
        exit_events = [e for e in path.events if e.get('reason') == 'exit_rule']
        self.assertEqual(exit_events,[{'date':'2026-01-06','key':'a','side':'sell',
                                       'price':50.,'reason':'exit_rule'}])

    def test_actual_holdings_buffer_removed_and_no_midweek_reentry(self):
        book, calls = self.make_book(), []
        def weekly(b,k,day,current):
            calls.append((day,set(current)))
            return self.decision()
        with patch.object(S,'weekly_target',side_effect=weekly):
            path = S.run_model(book,exit_at_open=self.sell_tuesday)
        self.assertEqual([day for day,current in calls],
                         ['2026-01-02','2026-01-05','2026-01-12'])
        self.assertEqual(calls[-1][1],{'b'})
        for day in ['2026-01-06','2026-01-07','2026-01-08','2026-01-09']:
            self.assertNotIn('a',path.weights[path.days.index(day)])
        # A recovery from 50 to 200 while in cash earns nothing. Reentry can
        # occur at the next weekly open, according to that week's target.
        self.assertAlmostEqual(path.close[path.days.index('2026-01-09')],.9)
        self.assertIn('a',path.weights[path.days.index('2026-01-12')])
        self.assertEqual(path.last['date'],'2026-01-12')

    def test_weekly_day_exit_happens_before_buffer_is_passed_to_target(self):
        book, calls = self.make_book(), []
        def weekly(b,k,day,current):
            calls.append((day,set(current)))
            return self.decision(selected=('b',),target={'b':.2}) if day >= '2026-01-05' else self.decision()
        def exit_monday(b,k,day,holdings):
            return {'a'} if day == '2026-01-05' else set()
        with patch.object(S,'weekly_target',side_effect=weekly):
            path = S.run_model(book,exit_at_open=exit_monday)
        self.assertEqual(dict(calls)['2026-01-05'],{'b'})
        monday_sales = [e for e in path.events if e['date']=='2026-01-05' and e['side']=='sell']
        self.assertEqual(len(monday_sales),1)
        self.assertEqual(monday_sales[0]['reason'],'exit_rule')

    def test_missing_or_nonpositive_open_cannot_invent_a_sale(self):
        for invalid in [np.nan,0.,-1.]:
            with self.subTest(open=invalid):
                book = self.make_book()
                book.opn.loc['2026-01-06','AAA'] = invalid
                with patch.object(S,'weekly_target',side_effect=lambda *args:self.decision()):
                    with self.assertRaises(ValueError):
                        S.run_model(book,exit_at_open=self.sell_tuesday)

    def test_callback_cannot_sell_an_unheld_symbol(self):
        def unheld(b,k,day,holdings):
            return {'unheld'} if day == '2026-01-06' else set()
        with patch.object(S,'weekly_target',side_effect=lambda *args:self.decision()):
            with self.assertRaises(ValueError):
                S.run_model(self.make_book(),exit_at_open=unheld)

    def test_default_none_and_empty_exit_hook_match_full_operating_path(self):
        with patch.object(S,'weekly_target',side_effect=lambda *args:self.decision()):
            base = S.run_model(self.make_book())
            none = S.run_model(self.make_book(),exit_at_open=None)
            empty = S.run_model(self.make_book(),exit_at_open=lambda *args:())
        for candidate in [none,empty]:
            np.testing.assert_array_equal(candidate.value,base.value)
            np.testing.assert_array_equal(candidate.close,base.close)
            self.assertEqual(candidate.weights,base.weights)
            self.assertEqual(candidate.events,base.events)
            self.assertEqual(candidate.rebalances,base.rebalances)
            self.assertEqual(candidate.selected,base.selected)
            self.assertEqual(candidate.last,base.last)


if __name__ == '__main__':
    unittest.main()
