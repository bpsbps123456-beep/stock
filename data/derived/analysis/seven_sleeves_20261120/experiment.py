"""Run the preregistered seven-slot grid and select continuous, auditable account paths."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
import market as inputs
from earnings import earnings_scores
from slots import Market, replay
from evaluate import summarize


SCHEDULES = [('weekly', 'weekly'), ('weekly', 'monthly'), ('daily', 'monthly'),
             ('monthly', 'monthly'), ('monthly', 'weekly')]
FOUR = [(3, 1, 1, 2), (2, 2, 1, 2), (2, 1, 2, 2), (3, 1, 2, 1), (1, 2, 2, 2)]


def configuration(counts, tf, ef, cap):
    frequency = {'D': 'weekly', 'M': tf, 'F': tf, 'E': ef}
    layout = tuple((label, frequency[label]) for j in range(max(counts))
                   for label, n in zip('DMFE', counts) if n > j)
    key = ''.join(f'{s}{n}' for s, n in zip('DMFE', counts) if n)
    if counts[1] or counts[2]:
        key += '-' + tf[0]
    if counts[3]:
        key += '-e' + ef[0]
    key += '-c' + str(cap or 0)
    return key, (layout, cap)


def grid():
    counts = set(FOUR)
    for d in range(8):
        for t in range(8-d):
            e = 7 - d - t
            counts.update([(d, t, 0, e), (d, 0, t, e)])
    return dict(configuration(c, tf, ef, cap) for c in sorted(counts)
                for tf, ef in SCHEDULES for cap in (None, 2))


def prepare():
    from baselines import market_gate
    src = inputs.load_inputs()
    market = Market(src.days, src.symbols, src.opn, src.close, src.fx, src.sectors,
                    src.issuers, src.settlements, src.tradable)
    # Raw share identifiers change over time; resolve the contemporaneous roster label per cell.
    raw = np.unique(src.raw_symbols)
    fiscal = earnings_scores(src.signals, raw)
    positions = np.searchsorted(raw, src.raw_symbols)
    values = np.take_along_axis(fiscal.to_numpy(), positions, axis=1)
    ranks = dict(src.ranks, E=[inputs.locations(row) for row in
                               np.where(src.membership, values, np.nan)])
    known = {s: [[i for i in rank if src.sector_known[j, i]] for j, rank in enumerate(days)]
             for s, days in ranks.items()}
    return src, market, {None: ranks, 2: known}, market_gate(src)


def configs_for_report():
    return [configuration(counts, 'weekly', 'monthly', 2)[0]
            for counts in ((3, 2, 0, 2), (3, 0, 2, 2))]


def overlays(src, eligible, gate):
    ranks = eligible[2]
    trans = np.flatnonzero(np.r_[False, gate[1:] != gate[:-1]])
    extras = {'SOXX100': (dict(ranks, D=[x if gate[j] else []
                                          for j, x in enumerate(ranks['D'])]), {'D': trans})}
    grades = []
    for kind in ('trend', 'giant'):
        data = pd.read_csv(inputs.DESK / f'{kind}_grades.csv', index_col=0)
        data = data.rename(columns=inputs.canon).reindex(index=src.signals, columns=src.symbols)
        grades.append(data.ge(3).to_numpy())
    for name, mask in zip(('trendA', 'giantA', 'bothA'), [*grades, grades[0] & grades[1]]):
        extras[name] = (dict(ranks, D=[[i for i in order if mask[j, i]]
                                       for j, order in enumerate(ranks['D'])]), {})
    return extras


def choose(result, configs):
    candidates = result.loc[list(configs)]
    mixed = candidates.query('D > 0 and E > 0 and (M > 0 or F > 0) and cap == 2')
    base = result.loc['base']
    h1 = mixed[mixed.h1_mdd_10bp >= base.h1_mdd_10bp - 1e-12]
    def rank(frame, col):
        return frame.sort_index().sort_values(col, ascending=False, kind='stable').index[0]
    wins = mixed[(mixed.full_return_10bp > base.full_return_10bp) &
                 (mixed.h2_return_10bp > base.h2_return_10bp) & (mixed.h2_mdd_10bp > base.h2_mdd_10bp)]
    return dict(candidates=len(candidates), mixed=len(mixed), h1_candidates=len(h1),
                better_all=len(wins), forward=rank(h1 if len(h1) else mixed, 'h1_return_10bp'),
                year=rank(wins if len(wins) else mixed, 'full_return_10bp'),
                highest_mixed=rank(mixed, 'full_return_10bp'),
                total=rank(candidates, 'full_return_10bp'), total_zero=rank(candidates, 'full_return'))


def contribution(curve, market):
    """Additive USD P&L per starting dollar; monthly budget transfers carry zero P&L."""
    loc = {s: i for i, s in enumerate(market.symbols)}
    profit = {}
    previous = [('', 1/7)] * 7
    for j, row in enumerate(curve.itertuples()):
        nxt = []
        for k, (old, amount) in enumerate(previous, 1):
            if old:
                profit[old] = profit.get(old, 0.) + amount * (market.overnight[j, loc[old]] - 1)
            name = getattr(row, f'slot{k}')
            value = getattr(row, f'slot{k}_value')
            if name:
                growth = market.intraday[j, loc[name]]
                profit[name] = profit.get(name, 0.) + value - value/growth
            nxt.append((name, value))
        previous = nxt
    np.testing.assert_allclose(sum(profit.values()), curve.usd.iloc[-1] - 1, atol=1e-10)
    return profit


def run():
    from baselines import baselines
    src, market, eligible, gate = prepare()
    controls = baselines(src, gate)
    configs = grid()
    extras = overlays(src, eligible, gate)
    jobs = {k: (layout, cap, eligible[cap], {}) for k, (layout, cap) in configs.items()}
    representatives = configs_for_report()
    for original in representatives:
        for suffix, (signal, changes) in extras.items():
            jobs[f'{original}/{suffix}'] = (*configs[original], signal, changes)
    rows = {k: dict(kind='control', **summarize(v)) for k, v in controls.items()}
    for k, (layout, cap, ranks, changes) in jobs.items():
        curve = replay(market, ranks, layout, cap, extra_reviews=changes)
        meta = {label: sum(s == label for s, _ in layout) for label in 'DMFE'}
        meta.update(cap=cap or 0, tech_frequency=next((f for s, f in layout if s in 'MF'), ''),
                    earnings_frequency=next((f for s, f in layout if s == 'E'), ''),
                    mean_active=curve.active.mean())
        rows[k] = dict(kind='grid' if k in configs else 'overlay', **meta, **summarize(curve))
    result = pd.DataFrame.from_dict(rows, orient='index').rename_axis('strategy')
    pick = choose(result, configs)
    keys = list(dict.fromkeys([*controls, *representatives, pick['forward'], pick['year'],
                               pick['highest_mixed'], pick['total'], pick['total_zero'],
                               *(key for key in jobs if '/' in key)]))
    detail = {}
    attribution = []
    for k in keys:
        if k in controls:
            curve = controls[k]
        else:
            layout, cap, ranks, changes = jobs[k]
            curve = replay(market, ranks, layout, cap, detail=True, extra_reviews=changes)
            sums = contribution(curve, market)
            attribution.extend((k, s, pnl) for s, pnl in sorted(sums.items(), key=lambda x: -x[1]))
        result.loc[k, 'mean_active'] = curve.active.mean()
        result.loc[k, 'mean_sectors'] = curve.sectors.mean()
        result.loc[k, 'full_seven_days'] = curve.active.eq(7).sum()
        result.loc[k, 'max_sector_weight'] = curve.max_sector_weight.max()
        detail[k] = curve
    assert all((v.krw10 > 0).all() and len(v) == len(src.days) for v in detail.values())
    for suffix in ('', '_10bp', '_usd'):
        np.testing.assert_allclose(1 + result['full_return' + suffix],
                                   (1 + result['h1_return' + suffix]) * (1 + result['h2_return' + suffix]))
    result.to_csv(HERE / 'results.csv', float_format='%.10g')
    pd.concat(detail, names=['strategy', 'date']).to_csv(HERE / 'selected_daily.csv', float_format='%.11g')
    pd.DataFrame(attribution, columns=['strategy', 'symbol', 'usd_profit_per_initial_dollar']).to_csv(
        HERE / 'attribution.csv', index=False, float_format='%.10g')
    for s in 'DMFE':
        lengths = [len(x) for x in eligible[None][s]]
        pick[f'{s}_eligible_range'] = [min(lengths), max(lengths)]
    pick['dates'] = [src.days[0], src.days[-1]]
    pick['sessions'] = len(src.days)
    pick['market_off_sessions'] = int((~gate).sum())
    (HERE / 'selection.json').write_text(json.dumps(pick, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(pick))


if __name__ == '__main__':
    run()
