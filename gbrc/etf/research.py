"""Reproducible 252,720-configuration ETF disclosure strategy experiment.

Run: py -m gbrc.etf.research
This writes research artifacts only, never orders or automated trading rules.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from numba import set_num_threads

from .research_data import prepare_bundle
from .research_engine import build_rankings, config_grid, simulate_grid, simulation_inputs, simulate_detail
from .storage import PROJECT_ROOT, atomic_write_json

OUT = PROJECT_ROOT / 'data/derived/analysis/2026-09-23_strategy_research'


def metrics(r):
    r = np.asarray(r, dtype=float)
    if not np.all(np.isfinite(r)):
        return {'return_pct': None, 'max_drawdown_pct': None}
    wealth = np.r_[1.0, np.cumprod(1+r)]
    return {'return_pct': float((wealth[-1]-1)*100),
            'max_drawdown_pct': float(np.min(wealth/np.maximum.accumulate(wealth)-1)*100)}


def benchmarks(arrays, fee):
    p, dates = arrays['benchmarks'], arrays['dates']
    first = int(np.searchsorted(dates, arrays['snapshot_dates'][0], side='right'))
    if not np.all(np.isfinite(p[first:])):
        raise ValueError('Missing benchmark prices; cannot rank strategies fairly')
    r = np.zeros_like(p)
    r[first] = -fee/(1+fee)
    r[first+1:] = p[first+1:]/p[first:-1]-1
    return r, first


def write_protocol(out, arrays, fee):
    protocol = {
        'created_before_grid_execution': datetime.now(timezone.utc).isoformat(),
        'period': [str(arrays['dates'][0]), str(arrays['dates'][-1])],
        'candidate_count': 252720,
        'search_space': {'hypotheses': 13, 'flow_windows_krx': [1,3,5,10,20,40],
            'time_mix': [0,0.25,0.5,0.75,1], 'price_gates': 3,
            'max_stocks': [1,2,3,5,7,10], 'rebalance': ['daily','calendar_week','calendar_month'],
            'weighting': ['equal','conviction','inverse_volatility'], 'risk_rules': 4},
        'selection': 'Jan-May log wealth minus half absolute max drawdown; reject train drawdown below -50%; '
                     'top 1% shortlisted, Jun-Jul same-objective winner; Aug-Sep retrospective holdout inspected after selection',
        'regime': 'Prior 20 observed valuation sessions QQQ USD return >3% and above MA20 = up; '
                  '<-3% and below MA20 = down; otherwise range; first 21 sessions unknown.',
        'price_basis': 'split-adjusted Close, dividends excluded; all values KRW; same-date local closes not synchronized',
        'disclosure_rule': 'Latest snapshot as_of STRICTLY BEFORE local execution calendar date',
        'execution': 'Earn existing positions close-to-close return then rebalance at current local close; '
                     'freeze closed-market positions, no leverage, cash in KRW with zero interest',
        'fees': {'one_way_fraction_of_traded_value': fee, 'includes': 'assumed commission + slippage + currency exchange allowance',
                 'sensitivity_bps': [0,15,30,50], 'tax': 'not deducted; account/jurisdiction dependent',
                 'terminal': 'mark to market, no forced sale fee'},
        'first_entry': 'All accounts begin cash; first disclosure-following date benchmark entry; '
                       'strategy waits in cash until its first eligible signal; subsequent rebalances calendar-based',
        'targets': 'Strategy cumulative percentage return >=2x or3x EACH ETF cumulative percentage return; '
                   'ratios defined only for positive ETF returns. This is not multiplying ending wealth.',
        'stress': ['same selected rule at 0/15/30/50bps', 'one extra global valuation-session delay',
                   'fresh cash start in holdout', 'selected daily path by prior-information regime',
                   'chronological monthly walk-forward with switch surcharge'],
        'limitations': ['retrospectively retrieved holdings; historical publication timestamp not verified',
            '252720 parameter rows are correlated and include duplicates; count unique rounded return paths separately',
            'holdout is retrospective, not future/live validation; no proof of 2x/3x future returns',
            'eight-plus-month history cannot validate an entire bull/bear cycle',
            'prices/features use only this sample; full lookback required, no external warmup'],
    }
    atomic_write_json(out/'protocol.json', protocol)
    return protocol


def run(out=OUT, refresh=False, reuse=False, threads=8):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print('Loading bounded historical dataset', flush=True)
    if not refresh and (out/'bundle.npz').exists():
        arrays = dict(np.load(out/'bundle.npz', allow_pickle=False))
        meta = json.loads((out/'metadata.json').read_text(encoding='utf-8'))
        for item in meta['inputs']:
            if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Snapshot changed; rerun --refresh-data')
    else:
        bundle = prepare_bundle(out, refresh)
        arrays, meta = bundle['arrays'], bundle['metadata']
    raw = json.loads((out/'raw_prices.json').read_text(encoding='utf-8'))
    # Construct causal raw-USD QQQ marks for market-state signals.
    last = np.nan
    qqq = []
    for day in arrays['dates']:
        last = raw['QQQ'].get(str(day), last)
        qqq.append(last)
    arrays['qqq_usd'] = np.asarray(qqq)
    fee = 0.0015
    protocol = write_protocol(out, arrays, fee) if not reuse else json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    print('Building causal ranks: 13 hypotheses x6 windows x5 fund blends x3 gates', flush=True)
    ranking = build_rankings(arrays)
    params, configs = config_grid(ranking['rank_configs'])
    if len(params) != protocol['candidate_count']:
        raise ValueError('Candidate grid does not match preregistered protocol')
    set_num_threads(threads)
    if reuse and (out/'grid_returns.npy').exists():
        daily = np.load(out/'grid_returns.npy', mmap_mode='r')
        stats = np.load(out/'grid_stats.npy')
    else:
        print(f'Simulating {len(params):,} parameter combinations', flush=True)
        daily, stats = simulate_grid(params, *simulation_inputs(arrays, ranking), fee)
        np.save(out/'grid_returns.npy', daily)
        np.save(out/'grid_stats.npy', stats)
        atomic_write_json(out/'configurations.json', configs)
    print('Temporal selection, retrospective holdout, and monthly walk-forward', flush=True)
    from .research_validation import evaluate_candidates
    bench, entry = benchmarks(arrays, fee)
    evaluation = evaluate_candidates(daily, arrays['dates'].tolist(), bench, configs)
    hashes = {hashlib.blake2b(np.round(row, 7).tobytes(), digest_size=16).digest()
              for row in daily if np.isfinite(row).all()}
    evaluation['unique_return_paths_rounded_1e_7'] = len(hashes)
    evaluation['benchmark_entry_date'] = str(arrays['dates'][entry])
    evaluation['source_data_summary'] = {k:meta[k] for k in ('sample_start','sample_end','snapshot_count',
                  'common_snapshot_count','security_count','missing_complete_price_series')}
    atomic_write_json(out/'evaluation.json', evaluation)
    np.save(out/'benchmark_returns.npy', bench)
    np.save(out/'regime_state.npy', ranking['regime_state'])
    print(json.dumps({'evaluation_keys': list(evaluation), 'unique_paths':len(hashes)}, ensure_ascii=False), flush=True)
    return arrays, ranking, params, configs, daily, stats, evaluation, bench


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--refresh-data', action='store_true')
    ap.add_argument('--reuse-grid', action='store_true')
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    run(a.out, a.refresh_data, a.reuse_grid, a.threads)


if __name__ == '__main__':
    main()
