"""Descriptive stress checks of frozen primary and subgroup selections."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from .research_engine import build_rankings, config_grid, simulation_inputs, simulate_detail
from .research import OUT, metrics
from .storage import atomic_write_json


def run(out=OUT):
    out = Path(out)
    arrays = dict(np.load(out/'bundle.npz', allow_pickle=False))
    raw = json.loads((out/'raw_prices.json').read_text(encoding='utf-8'))
    last, q = np.nan, []
    for d in arrays['dates']:
        last = raw['QQQ'].get(str(d), last)
        q.append(last)
    arrays['qqq_usd'] = np.array(q)
    ranking = build_rankings(arrays)
    params, configs = config_grid(ranking['rank_configs'])
    delayed = build_rankings(arrays, extra_lag=1)
    evaluation = json.loads((out/'evaluation.json').read_text(encoding='utf-8'))
    labels = {'primary_selected': evaluation['selected_index'], 'hindsight_best': evaluation['best_index']}
    for mix,label in [(1.0,'TIME_top10_weekly'),(0.0,'KoAct_top10_weekly')]:
        labels[label] = next(i for i,c in enumerate(configs) if c == {
            'family':'conviction','lookback':1,'time_mix':mix,'gate':'none',
            'max_stocks':10,'rebalance':'weekly','weighting':'conviction','risk_rule':'always'})
    subgroup_path = out/'subgroup_evaluation.json'
    subgroups = json.loads(subgroup_path.read_text(encoding='utf-8')) if subgroup_path.exists() else {}
    # Consume agent subgroup format without altering selection logic.
    groups = subgroups.get('groups', subgroups.get('subgroups', {}))
    if isinstance(groups, list):
        groups = {x['name']:x for x in groups}
    for label, row in groups.items():
        index = row.get('selected_global_index', row.get('selected_index', row.get('global_index')))
        if index is not None:
            labels[label] = index
    robust_path = out/'robust_research.json'
    if robust_path.exists():
        robust = json.loads(robust_path.read_text(encoding='utf-8'))
        for label,row in robust.get('scopes',{}).items():
            index = row.get('selected_global_index',row.get('selected_index'))
            if index is not None:
                labels['robust_'+label] = index
    dates = arrays['dates'].astype(str)
    hold = dates >= '2026-08-01'
    hstart = int(np.flatnonzero(hold)[0])
    bench = np.load(out/'benchmark_returns.npy')
    grid = np.load(out/'grid_returns.npy', mmap_mode='r')
    returns = simulation_inputs(arrays, ranking)[5]
    curves = {'dates': dates.tolist(), 'series': {}}
    result = {'assumptions': {'one_way_bps': [0,15,30,50],
             'delay': 'one extra global valuation-session, all disclosure and market signals delayed',
             'regime': 'QQQ USD past 20 observed valuation dates, +/-3% plus MA20 confirmation; unknown during warmup',
             'contributions': 'sum of prior-NAV-weighted KRW holding P&L divided by initial NAV; fees separate',
             'remove_top_contributor': 'post-selection dependence diagnostic: removes the ex-post top contributing stock from the cached top10, refills only from remaining cached ranks; ranks11+ are not reconsidered, so this is not a fully reranked exclusion strategy',
             'fresh_start': 'cash on first holdout date, execute its close; cannot earn first-day pre-entry return'},
              'candidates': {}, 'regime_counts': {name:int(np.sum(ranking['regime_state']==i))
                for i,name in enumerate(['unknown','up','down','range'])}}
    for label, index in labels.items():
        index = int(index)
        detail = simulate_detail(params[index], arrays, ranking)
        r = detail['returns']
        if not np.allclose(r, grid[index], atol=1e-7):
            raise ValueError(f'Grid/detail mismatch: {label}')
        row = {'index': index, 'config': configs[index], 'full': metrics(r), 'holdout':metrics(r[hold]),
               'max_observed_stocks': int(detail['counts'].max()),
               'average_observed_stocks':float(detail['counts'].mean()),
               'cash_days':int(np.sum(detail['cash'] > .99999)),
               'average_cash_pct':float(detail['cash'].mean()*100),
               'traded_nav_multiple':float(detail['turnover'].sum()),
               'rebalance_trade_days':int(np.sum(detail['turnover'] > 1e-8)), 'costs':[]}
        for bps in [0,15,30,50]:
            stress = simulate_detail(params[index], arrays, ranking, cost=bps/10000)['returns']
            row['costs'].append({'one_way_bps':bps, 'full':metrics(stress), 'holdout':metrics(stress[hold])})
        lagged = simulate_detail(params[index], arrays, delayed)['returns']
        row['extra_lag'] = {'full':metrics(lagged), 'holdout':metrics(lagged[hold])}
        fresh = simulate_detail(params[index], arrays, ranking, start=hstart)['returns']
        row['fresh_holdout'] = metrics(fresh)
        contribution = np.zeros(len(arrays['symbols']))
        wealth = np.r_[1.0, np.cumprod(1+r)]
        for t in range(1,len(dates)):
            for k,s in enumerate(detail['holdings'][t-1]):
                if s >= 0:
                    contribution[s] += wealth[t]*detail['allocations'][t-1,k]*returns[t,s]
        order = np.argsort(-contribution)
        row['contributions_pp'] = [{'symbol':str(arrays['symbols'][s]), 'pnl_per_initial_nav_pp':float(contribution[s]*100)}
                                   for s in order if abs(contribution[s]) > 1e-10]
        row['fee_pnl_pp'] = float((wealth[-1]-1 - contribution.sum())*100)
        first_s = int(order[0])
        reduced = dict(ranking)
        reduced['ranks'] = ranking['ranks'].copy()
        reduced['conviction'] = ranking['conviction'].copy()
        reduced['inverse_vol'] = ranking['inverse_vol'].copy()
        rid = params[index,0]
        for t in range(len(dates)):
            positions = np.flatnonzero((ranking['ranks'][rid,t] >= 0) & (ranking['ranks'][rid,t] != first_s))
            reduced['ranks'][rid,t] = -1
            reduced['conviction'][rid,t] = 0
            reduced['inverse_vol'][rid,t] = 0
            for k,p in enumerate(positions):
                reduced['ranks'][rid,t,k] = ranking['ranks'][rid,t,p]
                reduced['conviction'][rid,t,k] = ranking['conviction'][rid,t,p]
                reduced['inverse_vol'][rid,t,k] = ranking['inverse_vol'][rid,t,p]
        without = simulate_detail(params[index], arrays, reduced)['returns']
        row['without_top_contributor'] = {'symbol': str(arrays['symbols'][first_s]),
                                          'full':metrics(without), 'holdout':metrics(without[hold])}
        row['by_prior_regime'] = {name:{'observations':int(np.sum(ranking['regime_state']==i)),
                'candidate':metrics(r[ranking['regime_state']==i]),
                'benchmarks':{name2:metrics(bench[ranking['regime_state']==i,j]) for j,name2 in enumerate(['time','koact','qqq_krw'])}}
                for i,name in enumerate(['unknown','up','down','range']) if np.any(ranking['regime_state']==i)}
        row['monthly'] = {month:metrics(r[np.char.startswith(dates,month)]) for month in sorted({d[:7] for d in dates})}
        result['candidates'][label] = row
        curves['series'][label] = np.cumprod(1+r).tolist()
        if label == 'primary_selected' or label == 'hindsight_best' or 'mixed' in label or label in ['max_stocks_5','max_stocks_10','weekly_ge3']:
            with (out/f'{label}_historical_ledger.csv').open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(['date','signal_as_of','return_pct','nav','turnover_nav','cash_pct','holdings_after_close'])
                for t,d in enumerate(dates):
                    si = ranking['signal_snapshot_index'][t]
                    holdings = '; '.join(f'{arrays["symbols"][s]}:{detail["allocations"][t,k]*100:.4f}%'
                                         for k,s in enumerate(detail['holdings'][t]) if s >= 0)
                    writer.writerow([d, arrays['snapshot_dates'][si] if si>=0 else '', r[t]*100, wealth[t+1],
                                     detail['turnover'][t],detail['cash'][t]*100,holdings])
    for j,name in enumerate(['TIME ETF','KoAct ETF','QQQ KRW']):
        curves['series'][name] = np.cumprod(1+bench[:,j]).tolist()
    result['fresh_holdout_benchmarks'] = {name:metrics(np.r_[-.0015/1.0015, arrays['benchmarks'][hstart+1:,j]/arrays['benchmarks'][hstart:-1,j]-1])
                                        for j,name in enumerate(['time','koact','qqq_krw'])}
    atomic_write_json(out/'stress.json',result)
    atomic_write_json(out/'equity_curves.json',curves)
    # Full result catalog makes every trial inspectable, not only survivors.
    with (out/'all_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        keys=list(configs[0])
        writer.writerow(['id',*keys,'full_return_pct','full_max_drawdown_pct','holdout_return_pct','holdout_max_drawdown_pct'])
        from .research_validation import _batch_metrics
        full=_batch_metrics(grid,np.arange(len(dates)))
        held=_batch_metrics(grid,np.flatnonzero(hold))
        for i,config in enumerate(configs):
            writer.writerow([i,*[config[k] for k in keys],full['return'][i]*100,full['max_drawdown'][i]*100,
                             held['return'][i]*100,held['max_drawdown'][i]*100])
    print(json.dumps({label:{'full':row['full'], 'holdout':row['holdout'], 'lag':row['extra_lag'],
                           'without_top':row['without_top_contributor']} for label,row in result['candidates'].items()},indent=2))
    return result


if __name__ == '__main__':
    run()
