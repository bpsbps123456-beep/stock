"""Independent compound-return, drawdown and frozen-grid selection audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'gbrc.py').exists())


def read(path):
    return Path(path).read_text(encoding='utf-8')


def main():
    protocol = json.loads(read(HERE / 'protocol.json'))
    configs = json.loads(read(HERE / 'configurations.json'))
    published = json.loads(read(HERE / 'candidate_metrics.json'))
    selection = json.loads(read(HERE / 'selection.json'))
    bundle = np.load(HERE.parent / 'us_only_experiment/bundle.npz')
    dates = bundle['dates'].astype(str)
    assert len(configs) == protocol['candidate_count'] == 801
    keys = [json.dumps({k: v for k, v in c.items() if k not in ('index', 'stages')},
                       sort_keys=True) for c in configs]
    assert len(keys) == len(set(keys)), 'Duplicate definitions'
    paths = {}
    all_ret = []
    all_dd = []
    all_turnover = []
    for lag in range(3):
        name = f'publication_lag{lag}'
        r = np.load(HERE / f'{name}_returns.npy').astype(float)
        stats = np.load(HERE / f'{name}_stats.npy')
        assert r.shape == (801, len(dates))
        assert np.isfinite(r).all() and (r > -1).all()
        assert np.isfinite(stats).all() and np.max(stats[:, 0]) <= 10
        assert np.sum(stats[:, 18:20]) == 0, 'Invalid target allocation or cap'
        nav = np.concatenate([np.ones((801, 1)), np.cumprod(1 + r, axis=1)], axis=1)
        ret = (nav[:, -1] - 1) * 100
        dd = (nav / np.maximum.accumulate(nav, axis=1) - 1).min(axis=1) * 100
        np.testing.assert_allclose(ret, [x['scenarios'][name]['return_pct'] for x in published],
                                   atol=1e-8, rtol=0)
        np.testing.assert_allclose(dd, [x['scenarios'][name]['max_drawdown_pct'] for x in published],
                                   atol=1e-8, rtol=0)
        all_ret.append(ret)
        all_dd.append(dd)
        # Stable contract: index 12 is gross buy+sell notional divided by NAV.
        all_turnover.append(stats[:, 12])
        paths[name] = {'paths': len(r), 'return_range_pct': [float(ret.min()), float(ret.max())],
                       'max_holdings': int(stats[:, 0].max()),
                       'missing_held_close_marks': float(stats[:, 14].sum()),
                       'frozen_infeasible_cap_days': float(stats[:, 15].sum())}
    wr = np.min(all_ret, axis=0)
    wd = np.min(all_dd, axis=0)
    wt = np.max(all_turnover, axis=0)
    threshold = float(bundle['benchmarks'][-1, 0] / bundle['benchmarks'][0, 0] - 1) * 200
    cap_eligible = [i for i, c in enumerate(configs)
                    if c['stock_cap'] <= .25 and c['detail_sector_cap'] <= .6
                    and c['semiconductor_plus_unknown_cap'] <= .6]
    eligible = [i for i in cap_eligible if wr[i] >= threshold]
    eligible.sort(key=lambda i: (-wd[i], wt[i], i))
    assert selection['selected_index'] == (eligible[0] if eligible else None)
    published_rank = selection['meeting_2x_ranked_indices']
    assert set(published_rank) == set(eligible)
    rank_tie_error = max((abs(wd[a] - wd[b]) for a, b in zip(published_rank, eligible)), default=0)
    # cumprod versus log accumulation can order numerically equal lower ranks
    # differently. The selected rule must still match exactly.
    assert rank_tie_error < 1e-10
    assert selection['eligible_target_count'] == len(cap_eligible)
    shortlist = [{'index': i, 'worst_return_pct': float(wr[i]),
                  'worst_drawdown_pct': float(wd[i]), 'worst_turnover_nav': float(wt[i])}
                 for i in eligible[:20]]
    hash_checks = {}
    for name, expected in protocol['source_sha256'].items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_checks[name] = got == expected
    assert all(hash_checks.values()), 'Source mutated after protocol freeze'
    result = {'method': 'Independent NumPy cumprod including initial NAV=1; no engine metric functions',
              'paths_verified': 801 * 3, 'candidate_definitions_unique': True,
              'all_published_returns_and_drawdowns_match': True,
              'balanced_membership_and_selected_rule_match': True,
              'lower_ranking_matches_with_numeric_ties': True,
              'largest_drawdown_difference_for_reordered_ties_pp': float(rank_tie_error),
              'reference_return_pct': threshold / 2, 'required_return_pct': threshold,
              'cap_eligible_definitions': len(cap_eligible),
              'balanced_eligible_definitions': len(eligible),
              'selected_index': eligible[0] if eligible else None,
              'balanced_top20': shortlist, 'scenario_checks': paths,
              'source_hashes_unchanged': hash_checks}
    (HERE / 'result_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                           encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('source_hashes_unchanged', 'balanced_top20')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
