"""Exploratory three-names-per-sector sensitivity of the 175 preregistered mixed layouts."""
from pathlib import Path

import numpy as np
import pandas as pd

from experiment import grid, prepare
from evaluate import summarize
from slots import replay

HERE = Path(__file__).resolve().parent
UNCAPPED = 'D2M3E2-m-em-c0'
CRITERIA = ['full_return_10bp', 'h2_return_10bp', 'h2_mdd_10bp']


def run():
    src, market, ranks, _ = prepare()
    original = pd.read_csv(HERE / 'results.csv', index_col=0)
    mixed = original.query("kind == 'grid' and D > 0 and E > 0 and (M > 0 or F > 0)")
    two = mixed[mixed.cap.eq(2)]
    unlimited = mixed[mixed.cap.eq(0)]
    base, switch = original.loc['base'], original.loc['switch']
    old_wins = unlimited[unlimited[CRITERIA].gt(base[CRITERIA].astype(float)).all(axis=1)]
    assert set(old_wins.index) == {UNCAPPED} and len(two) == 175
    configs = grid()
    plans = {key[:-1] + '3': (configs[key][0], 3) for key in two.index}
    plans[UNCAPPED] = configs[UNCAPPED]
    # Starting at the first actual D selection isolates fully invested paths from the initial unfilled D slots.
    first_d = next(j for j, names in enumerate(src.ranks['D']) if names)
    rows, panels = {}, {}
    for key, (layout, cap) in plans.items():
        curve = replay(market, ranks[2 if cap else None], layout, cap, detail=True)
        panels[key] = curve
        stats = summarize(curve)
        stats.update(cap=cap or 0, mean_active=curve.active.mean(), full_seven_days=int(curve.active.eq(7).sum()),
                     full_after_first_d=bool(curve.active.iloc[first_d:].eq(7).all()),
                     full_h2_days=int(curve.loc['2026-07-01':, 'active'].eq(7).sum()),
                     min_sectors=int(curve.sectors.min()), mean_sectors=curve.sectors.mean(),
                     max_sector_weight=curve.max_sector_weight.max())
        rows[key] = stats
    result = pd.DataFrame.from_dict(rows, orient='index').rename_axis('strategy')
    np.testing.assert_allclose(result.loc[UNCAPPED, CRITERIA].astype(float),
                               original.loc[UNCAPPED, CRITERIA].astype(float), atol=1e-9)
    result['beats_base_three'] = result[CRITERIA].gt(base[CRITERIA].astype(float)).all(axis=1)
    result['beats_switch_three'] = result[CRITERIA].gt(switch[CRITERIA].astype(float)).all(axis=1)
    capped = result[result.cap.eq(3)]
    h1 = capped[capped.h1_mdd_10bp.ge(base.h1_mdd_10bp - 1e-12)]
    wins = capped[capped.beats_base_three]
    def top(subset, by):
        return subset.sort_index().sort_values(by, ascending=False, kind='stable').index[0]
    forward = top(h1 if len(h1) else capped, 'h1_return_10bp')
    year = top(wins if len(wins) else capped, 'full_return_10bp')
    result['h1_selected'] = result.index == forward
    result['year_selected'] = result.index == year
    result.to_csv(HERE / 'robustness.csv', float_format='%.11g')
    keys = dict.fromkeys([forward, year, UNCAPPED])
    pd.concat({s: panels[s] for s in keys}, names=['strategy', 'date']).to_csv(HERE / 'robustness_daily.csv', float_format='%.11g')
    metrics = ['h1_return_10bp', *CRITERIA, 'full_mdd_10bp', 'full_seven_days', 'full_after_first_d',
               'mean_active', 'min_sectors', 'mean_sectors', 'max_sector_weight']
    print('cap3 count / H1 eligible / better base & switch:', len(capped),len(h1),len(wins),int(capped.beats_switch_three.sum()))
    fully = capped[capped.full_after_first_d]
    print('seven every day since first D / better base & switch:',len(fully),int(fully.beats_base_three.sum()),int(fully.beats_switch_three.sum()))
    show = [*keys, UNCAPPED[:-1]+'3', top(capped, 'full_return_10bp')]
    if len(fully):
        show.append(top(fully, 'full_return_10bp'))
    print(result.loc[list(dict.fromkeys(show)), metrics].to_string())


if __name__ == '__main__':
    run()
