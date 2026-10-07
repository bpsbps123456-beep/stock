"""Adaptive second-stage ETF research across execution-cost and delay scenarios.

This experiment was designed AFTER the initial retrospective holdout was viewed.
It is exploratory robustness research, NOT independent or untouched validation.
Selection itself uses January-May training and June-July validation only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import ceil
from pathlib import Path

import numpy as np
from numba import set_num_threads

from .research import OUT, benchmarks
from .research_engine import build_rankings, config_grid, simulate_detail, simulate_grid, simulation_inputs
from .research_validation import (
    BENCHMARK_NAMES, TRAIN_END, VALIDATION_END, HOLDOUT_END,
    _batch_metrics, _metric_row, _one_metrics,
)
from .storage import atomic_write_json

STATUS = (
    'Adaptive exploration designed after the initial August-September holdout was viewed. '
    'Not independent validation; reported holdout is reused historical data. '
    'No candidate is optimized on holdout returns in this second-stage selection.'
)
SCENARIOS = {
    'base_15bps': {'cost': 0.0015, 'extra_global_session_lag': 0},
    'delayed_30bps': {'cost': 0.003, 'extra_global_session_lag': 1},
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_robust(train_metrics, validation_metrics, scope):
    """Freeze a candidate using ONLY the two supplied pre-holdout metric sets."""
    eligible_mask = np.ones(len(scope), dtype=bool)
    for values in train_metrics.values():
        eligible_mask &= np.isfinite(values['objective'][scope])
        eligible_mask &= values['max_drawdown'][scope] > -0.5
    eligible = scope[eligible_mask]
    training_objective = np.minimum.reduce([v['objective'] for v in train_metrics.values()])
    validation_objective = np.minimum.reduce([v['objective'] for v in validation_metrics.values()])
    train_order = eligible[np.lexsort((eligible, -training_objective[eligible]))]
    shortlist = train_order[:ceil(len(eligible) * 0.01)]
    finite_validation = shortlist[np.isfinite(validation_objective[shortlist])]
    final_order = finite_validation[np.lexsort((finite_validation,
                                                -validation_objective[finite_validation]))]
    return {
        'scope_count': int(len(scope)), 'eligible_training_count': int(len(eligible)),
        'shortlist_count': int(len(shortlist)), 'shortlist_indices': shortlist.tolist(),
        'validation_ranked_shortlist_indices': final_order.tolist(),
        'selected_index': int(final_order[0]) if len(final_order) else None,
        'selected_worst_training_objective': (
            float(training_objective[final_order[0]]) if len(final_order) else None),
        'selected_worst_validation_objective': (
            float(validation_objective[final_order[0]]) if len(final_order) else None),
    }


def run(out=OUT, reuse=False, threads=8):
    out = Path(out)
    print('Robustness stage: adaptive exploration, not an independent holdout', flush=True)
    arrays = dict(np.load(out / 'bundle.npz', allow_pickle=False))
    dates = arrays['dates'].astype(str)
    if dates[-1] > HOLDOUT_END:
        raise ValueError('Research bundle extends beyond the original sample')
    raw = json.loads((out / 'raw_prices.json').read_text(encoding='utf-8'))
    last, marks = np.nan, []
    for day in dates:
        last = raw['QQQ'].get(str(day), last)
        marks.append(last)
    arrays['qqq_usd'] = np.asarray(marks)
    fingerprints = {name: _sha(out / name) for name in (
        'bundle.npz', 'raw_prices.json', 'grid_returns.npy', 'configurations.json')}
    fingerprints['research_engine.py'] = _sha(Path(__file__).with_name('research_engine.py'))
    previous = {}
    result_path = out / 'robust_research.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding='utf-8'))
    protocol = {
        'schema_version': 1, 'research_status': STATUS,
        'created_before_second_stage_grid_execution': datetime.now(timezone.utc).isoformat(),
        'period': [str(dates[0]), str(dates[-1])], 'candidate_count': 252720,
        'scenarios': SCENARIOS,
        'selection_rule': {
            'train_before': TRAIN_END, 'validation_before': VALIDATION_END,
            'objective_in_each_scenario': 'compound log return minus 0.5 * absolute max drawdown',
            'robust_objective': 'minimum objective across the two execution scenarios',
            'eligible': 'training max drawdown strictly above -50% in EACH scenario',
            'shortlist': 'top 1% of training-eligible parameter rows within each scope, ceiling rounding',
            'selection': 'highest worst-scenario validation objective within the frozen shortlist',
            'tie_break': 'lower global candidate index',
            'scopes': ['all', 'mixed_max_stocks_3_to_10', 'mixed_max_stocks_5_to_10'],
            'mixed_definition': '0 < TIME signal mixture < 1; does not guarantee holdings sourced from both funds',
        },
        'benchmark_policy': (
            'Both scenarios compare to the original direct ETF buy-and-hold path, '
            '15 bps initial entry cost and no extra delay. The stressed strategy receives '
            'no easier delayed or higher-cost benchmark. Full-period 2x/3x counts use '
            'the stronger of the two original ETF cumulative profit rates, positive only.'
        ),
        'fresh_holdout_policy': (
            'Start cash on the first available August valuation date, earn no stock return '
            'before same-date close entry, use each scenario rule and cost. Fresh ETF '
            'benchmarks enter at the same date close with 15 bps cost and no delay.'
        ),
        'parameter_count_note': 'Correlated parameter rows with duplicate trajectories; not independent trials',
        'assumptions': 'Fees/slippage allowances are scenario assumptions; taxes and dividends are excluded',
        'input_fingerprints': fingerprints,
        'state': 'running',
    }
    if reuse and previous.get('input_fingerprints') != fingerprints:
        raise ValueError('Cannot reuse stressed grid: input/engine fingerprints differ')
    atomic_write_json(result_path, protocol)
    print('Building delayed causal rankings (+1 global valuation session)', flush=True)
    delayed_ranking = build_rankings(arrays, extra_lag=1)
    params, configs = config_grid(delayed_ranking['rank_configs'])
    original_configs = json.loads((out / 'configurations.json').read_text(encoding='utf-8'))
    if len(params) != 252720 or configs != original_configs:
        raise ValueError('Delayed grid does not match original global configuration IDs')
    base = np.load(out / 'grid_returns.npy', mmap_mode='r')
    set_num_threads(threads)
    if reuse:
        delayed = np.load(out / 'delayed_grid_returns.npy', mmap_mode='r')
        delayed_stats = np.load(out / 'delayed_grid_stats.npy')
    else:
        print(f'Simulating {len(params):,} configurations at 30 bps with extra delay', flush=True)
        delayed, delayed_stats = simulate_grid(
            params, *simulation_inputs(arrays, delayed_ranking), SCENARIOS['delayed_30bps']['cost'])
        np.save(out / 'delayed_grid_returns.npy', delayed)
        np.save(out / 'delayed_grid_stats.npy', delayed_stats)
    if delayed.shape != base.shape or base.shape != (len(params), len(dates)):
        raise ValueError('Scenario return matrices do not align')
    scenario_returns = {'base_15bps': base, 'delayed_30bps': delayed}
    columns = {
        'train': np.flatnonzero(dates < TRAIN_END),
        'validation': np.flatnonzero((dates >= TRAIN_END) & (dates < VALIDATION_END)),
        'holdout': np.flatnonzero(dates >= VALIDATION_END),
        'full': np.arange(len(dates)),
    }
    if any(len(c) == 0 for c in columns.values()):
        raise ValueError('Every original temporal segment must be present')
    print('Selecting with worst-scenario training and validation metrics only', flush=True)
    train = {name: _batch_metrics(v, columns['train']) for name, v in scenario_returns.items()}
    validation = {name: _batch_metrics(v, columns['validation']) for name, v in scenario_returns.items()}
    scopes = {
        'all': np.arange(len(configs)),
        'mixed_max_stocks_3_to_10': np.array([
            i for i, c in enumerate(configs) if 0 < c['time_mix'] < 1 and c['max_stocks'] >= 3]),
        'mixed_max_stocks_5_to_10': np.array([
            i for i, c in enumerate(configs) if 0 < c['time_mix'] < 1 and c['max_stocks'] >= 5]),
    }
    selections = {name: select_robust(train, validation, scope) for name, scope in scopes.items()}
    # Candidate IDs above are frozen before computing full or holdout performance.
    frozen = {name: selection['selected_index'] for name, selection in selections.items()}
    print(f'Frozen robust selection IDs: {frozen}', flush=True)
    metrics = {
        name: {'train': train[name], 'validation': validation[name],
               'holdout': _batch_metrics(v, columns['holdout']),
               'full': _batch_metrics(v, columns['full'])}
        for name, v in scenario_returns.items()
    }
    benchmark_returns, benchmark_entry = benchmarks(arrays, SCENARIOS['base_15bps']['cost'])
    benchmark_metrics = {segment: _batch_metrics(benchmark_returns.T, index)
                         for segment, index in columns.items()}
    benchmark_profiles = {
        segment: {name: _metric_row(values, i) for i, name in enumerate(BENCHMARK_NAMES)}
        for segment, values in benchmark_metrics.items()
    }
    stronger = int(np.argmax(benchmark_metrics['full']['return'][:2]))
    threshold = float(benchmark_metrics['full']['return'][stronger])
    holdout_start = int(columns['holdout'][0])
    base_ranking = build_rankings(arrays)
    fresh_bench = np.zeros((len(dates) - holdout_start, 3))
    fee = SCENARIOS['base_15bps']['cost']
    fresh_bench[0] = -fee / (1 + fee)
    bp = arrays['benchmarks'][holdout_start:]
    fresh_bench[1:] = bp[1:] / bp[:-1] - 1
    fresh_benchmark_profiles = {
        name: _one_metrics(fresh_bench[:, i]) for i, name in enumerate(BENCHMARK_NAMES)}

    def profile(index, fresh=False):
        if index is None:
            return None
        result = {'global_candidate_index': int(index), 'config': configs[index], 'scenarios': {}}
        for name, values in metrics.items():
            scenario = {'segments': {segment: _metric_row(metric, index)
                                     for segment, metric in values.items()}}
            full_return = values['full']['return'][index]
            scenario['full_return_ratio_to_stronger_original_etf'] = (
                float(full_return / threshold) if threshold > 0 and np.isfinite(full_return) else None)
            if fresh:
                ranking = base_ranking if name == 'base_15bps' else delayed_ranking
                detail = simulate_detail(params[index], arrays, ranking,
                                         cost=SCENARIOS[name]['cost'], start=holdout_start)
                scenario['fresh_cash_holdout'] = {
                    **_one_metrics(detail['returns']), 'entry_eligible_from': str(dates[holdout_start]),
                    'max_observed_holdings': int(detail['counts'].max()),
                    'average_cash_fraction': float(detail['cash'].mean()),
                    'total_traded_value_over_nav': float(detail['turnover'].sum()),
                }
            result['scenarios'][name] = scenario
        return result

    for name, selection in selections.items():
        scope = scopes[name]
        finite = np.ones(len(scope), dtype=bool)
        for values in metrics.values():
            finite &= np.isfinite(values['full']['return'][scope])
        counts = {'scope_parameter_rows': int(len(scope)), 'finite_both_scenarios': int(finite.sum())}
        for multiple in (1, 2, 3):
            passed = finite.copy()
            for values in metrics.values():
                passed &= values['full']['return'][scope] >= multiple * threshold
            counts[f'at_least_{multiple}x_stronger_etf_in_both_scenarios'] = (
                int(passed.sum()) if threshold > 0 else None)
        selection['full_period_joint_target_counts'] = counts
        selection['selected'] = profile(selection['selected_index'], fresh=True)
        selection['top5_by_preholdout_robust_validation'] = [
            profile(index) for index in selection['validation_ranked_shortlist_indices'][:5]]
    result = {
        **protocol, 'state': 'complete', 'completed_at': datetime.now(timezone.utc).isoformat(),
        'benchmarks': benchmark_profiles, 'benchmark_entry_date': str(dates[benchmark_entry]),
        'stronger_full_period_original_etf': BENCHMARK_NAMES[stronger],
        'stronger_full_period_original_etf_return_pct': threshold * 100,
        'fresh_cash_holdout_benchmarks': fresh_benchmark_profiles,
        'scopes': selections,
        'delayed_grid_audit': {
            'invalid_full_path_count': int(np.count_nonzero(~np.isfinite(delayed).all(axis=1))),
            'maximum_observed_holdings': int(delayed_stats[:, 1].max()),
            'minimum_average_cash_fraction': float(delayed_stats[:, 2].min()),
            'maximum_average_cash_fraction': float(delayed_stats[:, 2].max()),
        },
        'segment_basis': 'Continuing original portfolios; fresh-cash holdout is separately simulated',
        'limitations': [
            STATUS, 'Scenario robustness is not protection against researcher selection bias.',
            'Neither scenario establishes historical publication time of backfilled disclosures.',
            'Eight-plus-month sample does not establish performance over a full bull/bear cycle.',
            'Global daily local/FX closes are not synchronized executable quotations.',
            'Maximum stock count is a cap, not a guarantee of diversification or minimum holdings.',
        ],
    }
    atomic_write_json(result_path, result)
    print(json.dumps({'selected_indices': frozen, 'audit': result['delayed_grid_audit']},
                     ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--reuse-grid', action='store_true')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    run(args.out, args.reuse_grid, args.threads)


if __name__ == '__main__':
    main()
