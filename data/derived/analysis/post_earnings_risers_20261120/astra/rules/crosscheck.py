"""Read-only recount of the frozen Astra and Claude candidates from source inputs."""
import hashlib
import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from panel import CUTOFF, KEY, OUT, STUDY, load_panel


def source_panel(panel):
    path = STUDY / 'results'
    combined = pd.read_csv(path / 'events.csv')
    for filename in ('features.csv', 'features_extra.csv'):
        combined = combined.merge(pd.read_csv(path / filename), on=KEY, validate='one_to_one')
    combined = combined.merge(panel[KEY + ['E1', 'E6']], on=KEY, validate='one_to_one')
    combined['bmo'] = combined.timing.eq('bmo')
    return combined


def recount_ours(panel, result):
    checked = 0
    for candidate in result['confirmation']:
        entry, direction = candidate['family'].split('_')
        spec = candidate['spec']
        eligible = panel.timing.ne('unk') if entry == 'E1' else pd.Series(True, index=panel.index)
        train = eligible & panel.season.isin(['2026Q1', '2026Q2'])
        def historical_rank(name):
            ref = panel.loc[train, name].dropna()
            def midrank(x):
                return ((ref < x).sum() + .5 * (ref == x).sum()) / len(ref)
            return panel[name].map(midrank, na_action='ignore')
        a, b = historical_rank(spec['a']), historical_rank(spec['b'])
        score = (a + (b if spec['sign'] == 1 else 1-b)) / 2
        def threshold(series, op, locked):
            cut = series[train].quantile(.33 if op == 'le' else .67)
            assert np.isclose(cut, locked, rtol=0, atol=1e-12)
            return series.le(cut) if op == 'le' else series.ge(cut)
        mask = eligible & threshold(score, spec['score_op'], spec['score_cut'])
        if spec['gate']:
            mask &= threshold(panel[spec['gate']], spec['gate_op'], spec['gate_cut'])
        y = panel[entry]
        within = panel.reaction_date.le(CUTOFF)
        buckets = {f'Q{i}': panel.season.eq(f'2026Q{i}') & within for i in range(1, 5)}
        buckets.update(H1=buckets['Q1'] | buckets['Q2'], H2=buckets['Q3'] | buckets['Q4'], early=~within)
        for band, flags in buckets.items():
            stored = candidate['stats'][band]
            events = panel.loc[mask & flags & y.notna()]
            values = events[entry]
            wins = values.gt(0) if direction == 'up' else values.lt(0)
            assert (len(events), int(wins.sum()), events.ticker.nunique()) == (stored['n'], stored['wins'], stored['tickers'])
            if len(events):
                np.testing.assert_allclose([values.mean(), values.median(), events.ticker.value_counts(normalize=True).max()],
                                           [stored['mean_x'], stored['median_x'], stored['max_ticker_share']], rtol=0, atol=.0001)
            checked += 1
    return checked


def recount_claude(panel):
    frame = source_panel(panel)
    h1 = frame.season.isin(['2026Q1', '2026Q2'])
    h2 = ~h1 & frame.reaction_date.le(CUTOFF)

    @lru_cache(None)
    def condition(token):
        field, operator, value = re.fullmatch(r'(.+?)([≤≥=])(.+)', token).groups()
        x = frame[field]
        if operator == '=':
            target = {'참': True, '거짓': False}.get(value, value)
            return x.eq(target).to_numpy()
        # Restore the full cut from the reference sample; CSV names round it to 4 significant figures.
        p = .33333 if operator == '≤' else .66667
        cut = x[h1].quantile(p)
        assert f'{cut:.4g}' == value, (token, cut)
        return (x.le(cut) if operator == '≤' else x.ge(cut)).to_numpy()

    summaries = {}
    for entry in ('e1', 'e6'):
        y = frame[entry.upper()]
        for side in ('up', 'down'):
            path = STUDY / 'results' / f'rules_{entry}_{side}.csv'
            other = pd.read_csv(path)
            hits = y.gt(0) if side == 'up' else y.lt(0)
            audit = {'h1_rows': len(other), 'h2_pass': 0, 'h2_pass_strict': 0,
                     'pass_with_next_day_wiki': 0, 'pass_with_reconstructed_size_or_turn': 0,
                     'pass_with_either_issue_for_entry': 0,
                     'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            votes = pd.Series(0, index=frame.index)
            for row in other.itertuples():
                mask = np.logical_and.reduce([condition(token) for token in row.rule.split(' & ')])
                votes += mask
                stored_members = np.fromstring(row.members, sep=',', dtype=int)
                assert np.array_equal(np.flatnonzero(mask), stored_members), row.rule
                for name, partition in [('h1', h1), ('q1', frame.season.eq('2026Q1')),
                                        ('q2', frame.season.eq('2026Q2')), ('h2', h2)]:
                    sel = mask & partition & y.notna()
                    n, rate = int(sel.sum()), hits[sel].mean()
                    assert n == getattr(row, f'n_{name}'), (row.rule, name, n)
                    assert np.isclose(rate, getattr(row, f'hit_{name}'), atol=.00051, equal_nan=True), (row.rule, rate)
                counts = frame.loc[sel, 'ticker'].value_counts()
                assert len(counts) == row.tickers_h2
                assert np.isclose(y[sel].mean(), row.mean_h2, atol=.00501, equal_nan=True)
                passed = n >= 15 and rate >= .7
                assert passed == row.pass_h2
                if passed:
                    audit['h2_pass'] += 1
                    aligned = y[sel].mean() > 0 if side == 'up' else y[sel].mean() < 0
                    audit['h2_pass_strict'] += int(len(counts) >= 8 and counts.max()/n <= .25 and aligned)
                    wiki = 'a_h07_wiki_post' in row.rule
                    approx_size = 'px_mcap_b' in row.rule or 'px_turn' in row.rule
                    audit['pass_with_next_day_wiki'] += int(wiki)
                    audit['pass_with_reconstructed_size_or_turn'] += int(approx_size)
                    audit['pass_with_either_issue_for_entry'] += int(approx_size or (wiki and entry == 'e1'))
            vote_threshold = max(votes.loc[h1].quantile(.8), 1e-9)
            vote_events = votes.ge(vote_threshold) & h2 & y.notna()
            stored_votes = pd.read_csv(path.with_name(f'rules_{entry}_{side}_vote_h2.csv'))
            assert len(stored_votes) == vote_events.sum()
            assert frame.loc[vote_events, KEY].reset_index(drop=True).equals(stored_votes[KEY])
            np.testing.assert_allclose(stored_votes.votes, votes[vote_events])
            np.testing.assert_allclose(stored_votes.y, y[vote_events], rtol=0, atol=.0001)
            audit.update(vote_n=int(vote_events.sum()), vote_wins=int(hits[vote_events].sum()))
            summaries[f'{entry}_{side}'] = audit
    return summaries


def unknown_time_sensitivity(panel):
    """Repeat the fake-only pipeline with the ambiguous E1 label outside its shuffle pool."""
    from search import RUNS, Selector, candidates, groups, permute
    from threadpoolctl import threadpool_limits

    adjusted = panel.copy()
    adjusted.loc[adjusted.timing.eq('unk'), 'E1'] = np.nan
    simulations = permute(adjusted)
    h2 = (adjusted.season.isin(['2026Q3', '2026Q4']) & adjusted.reaction_date.le(CUTOFF)).to_numpy()
    passed = [[] for _ in range(RUNS)]
    with threadpool_limits(limits=2):
        for col, entry in enumerate(('E1', 'E6')):
            _, masks = candidates(adjusted, entry)
            selector = Selector(adjusted, entry, masks)
            for side in ('up', 'down'):
                draws = simulations[:, col, :]
                for trial, selected in enumerate(selector.choose(draws, side)):
                    for rule in selected['ids']:
                        mask = masks[rule] & h2 & np.isfinite(draws[:, trial])
                        y = draws[mask, trial]
                        sign = 1 if side == 'up' else -1
                        if len(y) < 15 or (y*sign > 0).mean() < .7 or sign*y.mean() <= 0:
                            continue
                        counts = adjusted.loc[mask, 'ticker'].value_counts()
                        if len(counts) >= 8 and counts.max()/len(y) <= .25:
                            passed[trial].append(mask)
    cluster_counts = np.array([groups(masks) for masks in passed])
    return {'runs': RUNS, 'trials_ge_1': int((cluster_counts >= 1).sum()),
            'trials_ge_3': int((cluster_counts >= 3).sum()), 'p95': float(np.quantile(cluster_counts, .95))}


if __name__ == '__main__':
    assert (OUT / 'REPORT.md').is_file(), 'Write the frozen-candidate report before opening peer outputs.'
    panel = load_panel()
    result = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    output = dict(own_candidate_buckets_checked=recount_ours(panel, result), peer=recount_claude(panel),
                  unknown_time_sensitivity=unknown_time_sensitivity(panel))
    print(json.dumps(output, indent=2, ensure_ascii=False))

