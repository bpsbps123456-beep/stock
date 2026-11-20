"""Compare point-in-time screen recipes and trading calendars against the original monthly switch."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'seven_sleeves_20261120'
sys.path[:0] = [str(PREVIOUS), str(HERE.parents[3])]
from market import load_inputs
from baselines import market_gate
from timing import TimingEngine, SCHEDULES
from recipes import recipes, metadata

ORIGINAL = 'original/monthly1'
PERIODS = {'year': ('2026-01-01', '2026-11-18'), 'h1': ('2026-01-01', '2026-06-30'),
           'h2': ('2026-07-01', '2026-11-18'), 'q3': ('2026-07-01', '2026-09-30'),
           'oct': ('2026-10-01', '2026-11-18')}


def describe(curve):
    result = {}
    for p, (first, last) in PERIODS.items():
        ix = np.flatnonzero((curve.index >= first) & (curve.index <= last))
        start, end = ix[0], ix[-1]
        view = curve.iloc[start:end+1]
        for col, prefix in [('krw10', ''), ('krw', 'gross_'), ('krw25', 'stress_')]:
            if col not in curve:
                continue
            initial = curve[col].iat[start-1] if start else 1.
            wealth = np.r_[initial, view[col]]
            result[f'{p}_{prefix}ret'] = wealth[-1]/wealth[0]-1
            result[f'{p}_{prefix}dd'] = min(wealth/np.maximum.accumulate(wealth)-1)
        result[f'{p}_exposure'] = view.exposure.mean()
        result[f'{p}_turn'] = view.turn.sum()
    return result


def rank_value(frame, field):
    return frame.sort_index().sort_values(field, ascending=False, kind='stable').index[0]


def select_train(always):
    base = always.loc[ORIGINAL]
    same = always[always.recipe.eq('original')].set_index('schedule').h1_ret
    train = always[['recipe', 'schedule', 'h1_ret', 'h1_dd', 'h1_exposure']].copy()
    train['excess'] = train.h1_ret - train.schedule.map(same)
    grouped = train.groupby('recipe').agg(median_excess=('excess', 'median'),
               sufficiently_invested=('h1_exposure', lambda a: int(a.ge(.6).sum())))
    viable = grouped[grouped.sufficiently_invested.ge(5) & grouped.median_excess.gt(1e-12)]
    best = rank_value(viable, 'median_excess') if len(viable) else 'original'
    pool = train[train.recipe.eq(best) & train.h1_exposure.ge(.6) & train.h1_dd.ge(base.h1_dd-1e-12)]
    choice = rank_value(pool, 'h1_ret') if len(pool) else ORIGINAL
    return choice, grouped


def spells(days, gate, paths):
    """Measured until the close after restoring the desk, including the actual opening gap and fees."""
    starts = np.flatnonzero(~gate & np.r_[True, gate[:-1]])
    ends = np.flatnonzero(~gate & np.r_[gate[1:], True])
    data = []
    for n, (first, end) in enumerate(zip(starts, ends), 1):
        mark = min(end+1, len(days)-1)
        moved = {}
        for k, curve in paths.items():
            previous = curve.iat[first-1] if first else 1.
            moved[k] = curve.iat[mark] / previous
        for key, ratio in moved.items():
            data.append({'key': key, 'episode': n, 'exit': days[first], 'last_off': days[end],
                         'end_mark': days[mark], 'sessions_off': end-first+1,
                         'portfolio_return': ratio-1, 'relative_return': ratio/moved[ORIGINAL]-1})
    data = pd.DataFrame(data)
    wide = data.pivot(index='key', columns='episode', values='relative_return').add(1)
    total = wide.prod(axis=1)
    for k, end_ratio in total.items():
        np.testing.assert_allclose(end_ratio, paths[k].iat[-1]/paths[ORIGINAL].iat[-1], atol=5e-10)
    summary = pd.DataFrame({'episodes_ahead': (wide > 1 + 1e-10).sum(axis=1),
                            'worst_excluding_one': wide.rdiv(total, axis=0).min(axis=1)-1})
    return data, summary


def run():
    src = load_inputs()
    gate = market_gate(src)
    engine = TimingEngine(src, gate)
    lookup = recipes(src)
    info = metadata()
    assert set(info) == set(lookup) and len(info) == 17
    engine.verify_original(lookup['original'])
    switched, always, nav = {}, {}, {}
    for name, orders in lookup.items():
        assert len(orders) == len(src.days)
        for schedule in SCHEDULES:
            key = f'{name}/{schedule}'
            paths = engine.evaluate(orders, schedule)
            for mode, frame in paths.items():
                target = switched if mode == 'switch' else always
                review = engine.reviews[schedule]
                counts = np.array([len(orders[j]) for j in review])
                target[key] = dict(recipe=name, schedule=schedule, reviews=len(review),
                                   mean_eligible_at_review=counts.mean(), reviews_below_seven=int((counts<7).sum()),
                                   off_exposure=frame.exposure.to_numpy()[~gate].mean(),
                                   off_held=frame.active.to_numpy()[~gate].mean(), **describe(frame))
            nav[key] = paths['switch'].krw10
    def table(rows):
        return pd.DataFrame.from_dict(rows, orient='index').rename_axis('key')
    sw, al = table(switched), table(always)
    assert (len(sw), len(al)) == (170, 136)
    recommended, training = select_train(al)
    challenger = sw.loc[recommended]
    ref = sw.loc[ORIGINAL]
    passed = bool(challenger.q3_ret >= ref.q3_ret-1e-12 and challenger.q3_dd >= ref.q3_dd-1e-12)
    screened = sw[(sw.year_ret > ref.year_ret+1e-10) & (sw.h2_ret > ref.h2_ret+1e-10)
                  & (sw.h2_dd > ref.h2_dd+1e-10)]
    diag = rank_value(sw, 'year_ret')
    controlled = rank_value(screened, 'year_ret') if len(screened) else ORIGINAL
    episodes, analysis = spells(src.days, gate, nav)
    sw = sw.join(analysis)
    pick = dict(train_pick=recommended, q3_pass=passed, october_pick=recommended if passed else ORIGINAL,
                hindsight=diag, superior_hindsight=controlled, candidates=len(sw), superior=len(screened),
                four_or_more_of_five=int((sw.episodes_ahead >= 4).sum()),
                improved_after_removing_any_episode=int((sw.worst_excluding_one > 1e-10).sum()),
                off_sessions=int((~gate).sum()), h1_off=int((~gate[:sum(x <= '2026-06-30' for x in src.days)]).sum()))
    keys = list(dict.fromkeys([ORIGINAL, recommended, diag, controlled,
                              f'{recommended.split("/")[0]}/monthly1', 'original/'+recommended.split('/')[1],
                              f'{diag.split("/")[0]}/monthly1', 'original/'+diag.split('/')[1]]))
    curves = engine.controls(detail=True, stress=True)
    for key in keys:
        name, schedule = key.split('/')
        curves[key] = engine.evaluate(lookup[name], schedule, detail=True, stress=True)['switch']
    details = table({k:describe(v) for k,v in curves.items()})
    # Save the choice before reading the extra day's valuations.
    (HERE/'selection.json').write_text(json.dumps(pick, indent=2)+'\n',encoding='utf-8')
    (HERE/'screens.json').write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    training.to_csv(HERE/'training.csv', float_format='%.11g')
    sw.to_csv(HERE/'switch_results.csv', float_format='%.11g')
    al.to_csv(HERE/'always_results.csv', float_format='%.11g')
    episodes.to_csv(HERE/'episodes.csv', index=False, float_format='%.11g')
    details.to_csv(HERE/'selected_results.csv', float_format='%.11g')
    pd.concat(curves, names=['key','date']).to_csv(HERE/'selected_daily.csv', float_format='%.11g')
    weights = {k:{s:float(w) for s,w in v.attrs['final_weights_10bp'].items()} for k,v in curves.items()}
    (HERE/'end_weights.json').write_text(json.dumps(weights, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(pick))


if __name__ == '__main__':
    run()
