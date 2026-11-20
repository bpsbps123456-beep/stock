"""The fiscal-year and holdout columns cannot affect the H1 screen and trading-calendar choice."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]/'data/derived/analysis/finviz_recipes_20261120'
sys.path.insert(0,str(STUDY))
from research import select_train, spells
from timing import CALENDAR


class FinvizSelectionTests(unittest.TestCase):
    def test_selection_uses_matched_schedules_exposure_and_drawdown(self):
        rows = []
        for recipe, extra in [('original',0.), ('eligible',.1), ('cash_heavy',.2)]:
            for j, schedule in enumerate(CALENDAR):
                rows.append(dict(key=f'{recipe}/{schedule}', recipe=recipe, schedule=schedule,
                                 h1_ret=j/50+extra, h1_dd=-.3 if j==7 else -.1,
                                 h1_exposure=.1 if recipe=='cash_heavy' else .9, oct_ret=3*j))
        data = pd.DataFrame(rows).set_index('key')
        selected, _ = select_train(data)
        self.assertEqual(selected,'eligible/'+CALENDAR[6])
        data['oct_ret'] = -data.oct_ret
        data['year_ret'] = np.arange(len(data))*100
        self.assertEqual(select_train(data)[0],selected)

    def test_episode_attributes_gap_to_the_exiting_basket(self):
        dates = [f'2026-01-{n:02d}' for n in range(1,8)]
        gate = np.array([True,False,True,True,True,False,False])
        constant = pd.Series([1.]*7, index=dates)
        relative = pd.Series([1.,1.1,1.2,1.2,1.2,1.15,1.08], index=dates)
        detail, scores = spells(dates,gate,{'original/monthly1':constant,'test':relative})
        sub = detail[detail.key.eq('test')]
        np.testing.assert_allclose(sub.relative_return, [.2,-.1])
        self.assertEqual(list(sub.end_mark),[dates[2],dates[-1]])
        self.assertEqual(scores.at['test','episodes_ahead'],1)
        self.assertAlmostEqual(scores.at['test','worst_excluding_one'],-.1)
