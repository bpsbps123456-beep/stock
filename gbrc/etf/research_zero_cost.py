"""US-only fixed-grid research with explicit trading-cost deductions set to zero.

Run: py -m gbrc.etf.research_zero_cost [--reuse-grid] [--threads 8]
Reads the existing US experiment; writes only its zero_cost/ child directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from numba import set_num_threads
from .research import OUT, benchmarks
from .research_engine import build_rankings, config_grid, simulate_grid, simulation_inputs, simulate_detail
from .research_robust import select_robust
from .research_us_only import _hash
from .research_us_experiment import _rule_ko
from .research_validation import evaluate_candidates, _batch_metrics, _metric_row, _one_metrics
from .storage import atomic_write_json, atomic_write_text

STATUS = 'Additional adaptive research after earlier results; August-September is reused historical evaluation, not independent validation.'
SOURCE_FILES = ('bundle.npz', 'metadata.json', 'raw_prices.json', 'protocol.json',
                'configurations.json', 'evaluation.json', 'summary.json',
                'grid_returns.npy', 'delayed_grid_returns.npy')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(source=OUT/'us_only_experiment', reuse=False, threads=8):
    source = Path(source).resolve()
    out = (source/'zero_cost').resolve()
    if out.parent != source:
        raise ValueError('Output must be isolated inside source/zero_cost')
    out.mkdir(parents=True, exist_ok=True)
    hashes = {name: _hash(source/name) for name in SOURCE_FILES}
    arrays = dict(np.load(source/'bundle.npz', allow_pickle=False))
    meta, raw, saved_configs = [load(source/name) for name in
                               ('metadata.json', 'raw_prices.json', 'configurations.json')]
    dates = arrays['dates'].astype(str)
    if not all(row['market'] == 'US' for row in meta['securities']):
        raise ValueError('Source includes non-US securities')
    if not all(str(day) in raw['QQQ'] for day in dates):
        raise ValueError('Source date axis includes non-US sessions')
    if hashes['bundle.npz'] != meta['bundle_sha256']:
        raise ValueError('Source bundle checksum mismatch')
    arrays['qqq_usd'] = np.asarray([raw['QQQ'][str(day)] for day in dates])
    engine_hash = _hash(Path(__file__).with_name('research_engine.py'))
    if (out/'grid_returns.npy').exists():
        previous = load(out/'protocol.json')
        if not reuse or previous['source_sha256'] != hashes or previous['engine_sha256'] != engine_hash:
            raise ValueError('Existing grid requires --reuse-grid and identical inputs/engine')
    protocol = {
        'created_before_grid_execution': datetime.now(timezone.utc).isoformat(),
        'research_status': STATUS, 'period': [str(dates[0]), str(dates[-1])],
        'securities': len(arrays['symbols']), 'us_sessions': len(dates),
        'candidate_count_per_scenario': 252720, 'explicit_cost_deduction': 0.0,
        'scenarios': {'base_zero': {'cost': 0.0, 'extra_us_session_lag': 0},
                      'lag1_zero': {'cost': 0.0, 'extra_us_session_lag': 1}},
        'selection': 'Jan-May training MDD > -50%; top1% by log return minus half absolute MDD; June-July validation winner; no August-September selection',
        'robust_selection': 'Same selection using worse objective across both zero-cost delay scenarios',
        'scopes': ['all', 'exact_max3', 'exact_max5', 'mixed_max3_to10', 'mixed_max5_to10'],
        'benchmark_policy': 'Same zero explicit-cost TIME/KoAct/QQQ paths; fresh August cash starts also deduct zero',
        'flow_policy': 'Original full-fund CU and KRX snapshot signals and NAV weights retained; US-only filtering does not renormalize inputs',
        'cost_interpretation': 'Zero means no explicit buy/sell cost deduction. It does not verify zero real slippage, bid-ask spread, currency costs or taxes.',
        'execution': 'Assumed regular-session Close fills and fractional positions, KRW daily FX marks, no leverage; taxes/dividends excluded',
        'source_directory': str(source), 'configurations_source': str(source/'configurations.json'),
        'source_sha256': hashes, 'engine_sha256': engine_hash,
    }
    atomic_write_json(out/'protocol.json', protocol)
    print('Zero-cost US experiment: protocol saved before grid; 150 listings /182 US sessions', flush=True)
    set_num_threads(threads)
    rankings, grids, stats = {}, {}, {}
    for name, lag, prefix in [('base_zero', 0, ''), ('lag1_zero', 1, 'delayed_')]:
        print(f'Building {name} causal rankings', flush=True)
        rankings[name] = build_rankings(arrays, extra_lag=lag)
        params, configs = config_grid(rankings[name]['rank_configs'])
        if len(params) != 252720 or configs != saved_configs:
            raise ValueError('Configuration IDs differ from original US experiment')
        path = out/f'{prefix}grid_returns.npy'
        if reuse and path.exists():
            grids[name] = np.load(path, mmap_mode='r')
            stats[name] = np.load(out/f'{prefix}grid_stats.npy')
        else:
            print(f'Simulating {name}: {len(params):,} configurations, explicit fee0', flush=True)
            grids[name], stats[name] = simulate_grid(params, *simulation_inputs(arrays, rankings[name]), 0.0)
            np.save(path, grids[name]); np.save(out/f'{prefix}grid_stats.npy', stats[name])
    bench, entry = benchmarks(arrays, 0.0)
    evaluation = evaluate_candidates(grids['base_zero'], dates.tolist(), bench, configs)
    # The legacy helper adds fixed 15/30bp walk-forward switching surcharges.
    # Its grid/selection/benchmark/fresh paths are unaffected; omit that diagnostic.
    evaluation.pop('walk_forward', None)
    evaluation['excluded_diagnostics'] = {'walk_forward':
        'Legacy hardcoded switching surcharges; excluded from zero-cost experiment'}
    evaluation.update({'research_status': STATUS, 'explicit_cost_deduction': 0.0})
    atomic_write_json(out/'evaluation.json', evaluation)
    np.save(out/'benchmark_returns.npy', bench)
    columns = {'train': np.flatnonzero(dates < '2026-06-01'),
               'validation': np.flatnonzero((dates >= '2026-06-01') & (dates < '2026-08-01')),
               'holdout': np.flatnonzero(dates >= '2026-08-01'), 'full': np.arange(len(dates))}
    train = {name: _batch_metrics(grid, columns['train']) for name, grid in grids.items()}
    valid = {name: _batch_metrics(grid, columns['validation']) for name, grid in grids.items()}
    scopes = {'all': np.arange(len(configs)),
              'exact_max3': np.array([i for i,c in enumerate(configs) if c['max_stocks'] == 3]),
              'exact_max5': np.array([i for i,c in enumerate(configs) if c['max_stocks'] == 5]),
              'mixed_max3_to10': np.array([i for i,c in enumerate(configs) if c['max_stocks'] >= 3 and 0<c['time_mix']<1]),
              'mixed_max5_to10': np.array([i for i,c in enumerate(configs) if c['max_stocks'] >= 5 and 0<c['time_mix']<1])}
    selections = {scope: {'base': select_robust({'base_zero': train['base_zero']},
                    {'base_zero': valid['base_zero']}, indices),
                    'robust': select_robust(train, valid, indices)} for scope, indices in scopes.items()}
    assert selections['all']['base']['selected_index'] == evaluation['selected_index']
    print('Frozen selections before subgroup holdout: '+json.dumps({k:{mode:s['selected_index'] for mode,s in v.items()} for k,v in selections.items()}), flush=True)
    metric_sets = {name: {'train': train[name], 'validation': valid[name],
                          'holdout': _batch_metrics(grid, columns['holdout']),
                          'full': _batch_metrics(grid, columns['full'])} for name,grid in grids.items()}
    hstart = int(columns['holdout'][0])
    fresh_bench = np.zeros((len(dates)-hstart, 3))
    bp = arrays['benchmarks'][hstart:]
    fresh_bench[1:] = bp[1:]/bp[:-1]-1
    benchmark_profiles = {name: {'segments': {key:_one_metrics(bench[ix,j]) for key,ix in columns.items()},
                                'fresh_holdout': _one_metrics(fresh_bench[:,j])}
                          for j,name in enumerate(('time','koact','qqq_krw'))}
    fixed = {label: row['index'] for label,row in load(source/'summary.json')['candidates'].items()}
    selected_ids = {s['selected_index'] for v in selections.values() for s in v.values()}
    selected_ids.update(fixed.values())
    profiles = {}
    for index in sorted(selected_ids):
        profile = {'index': index, 'config': configs[index], 'rule_korean': _rule_ko(configs[index]), 'scenarios': {}}
        for name in grids:
            detail = simulate_detail(params[index], arrays, rankings[name], cost=0.0)
            if not np.allclose(detail['returns'], grids[name][index], atol=1e-7):
                raise ValueError('Grid/detail mismatch')
            fresh = simulate_detail(params[index], arrays, rankings[name], cost=0.0, start=hstart)
            profile['scenarios'][name] = {
                'segments': {key:_metric_row(metric,index) for key,metric in metric_sets[name].items()},
                'fresh_holdout': _one_metrics(fresh['returns']),
                'actual_max_stocks': int(detail['counts'].max()),
                'actual_mean_stocks': float(detail['counts'].mean()),
                'mean_cash_pct': float(detail['cash'].mean()*100),
            }
        profiles[str(index)] = profile
    stronger = max(benchmark_profiles[name]['segments']['full']['return_pct'] for name in ('time','koact'))/100
    counts = {f'at_least_{multiple}x_both_zero_cost_etfs_in_both_scenarios':
              int(np.sum(np.minimum(metric_sets['base_zero']['full']['return'],
                                    metric_sets['lag1_zero']['full']['return']) >= multiple*stronger))
              if stronger>0 else None for multiple in (1,2,3)}
    results = {'protocol': protocol, 'selections': selections, 'profiles': profiles,
               'fixed_previous_us_candidates': fixed, 'benchmarks': benchmark_profiles,
               'full_period_target_counts': counts, 'benchmark_entry_date': str(dates[entry]),
               'fresh_holdout_start': str(dates[hstart]), 'hindsight_best': evaluation['hindsight_best'],
               'candidate_count': len(configs), 'total_grid_paths': len(configs)*2}
    atomic_write_json(out/'results.json', results)
    audit = {'source_files_unchanged': {name:_hash(source/name) for name in SOURCE_FILES} == hashes,
             'explicit_cost_zero_in_both_scenarios_and_benchmarks': True,
             'legacy_charged_walk_forward_excluded': True,
             'us_listings_only': True, 'us_sessions_only': True, 'selection_uses_holdout': False,
             'max_observed_holdings': max(int(s[:,1].max()) for s in stats.values()),
             'invalid_paths': {name:int(np.sum(~np.isfinite(grid).all(axis=1))) for name,grid in grids.items()},
             'all_selected_grid_detail_paths_match': True}
    if not audit['source_files_unchanged'] or audit['max_observed_holdings']>10 or any(audit['invalid_paths'].values()):
        raise ValueError('Zero-cost artifact validation failed')
    atomic_write_json(out/'validation.json', audit)
    write_report(out, results)
    print(json.dumps({'out':str(out),'audit':audit,'targets':counts},ensure_ascii=False), flush=True)
    return results


def write_report(out, result):
    def pct(x): return f'{x:+.2f}%'
    scope_names = {'all':'전체','exact_max3':'최대3종목','exact_max5':'최대5종목',
                   'mixed_max3_to10':'두 펀드 혼합·상한3~10','mixed_max5_to10':'두 펀드 혼합·상한5~10'}
    lines = ['# 미국 상장·매수매도 비용 차감0 연구', '',
             '사용자가 지정한 매수매도 비용0원을 적용해 미국150종목·182거래일에서252,720개 규칙을 다시 계산했다. '
             '기본 실행과1미국거래일 추가 지연 모두 비용 차감0으로, 총505,440개 경로다. ETF도 같은 무비용 기준이다.', '',
             '1~5월 훈련·6~7월 검증으로 후보를 선택했다. 8~9월은 앞선 연구에서 본 과거 자료이므로 독립 검증이 아니다.', '',
             '| 선택 방식 / 집단 | 번호 | 실제 최대/평균 종목 | 전체 | 최대낙폭 | 8월 현금 출발 | +1미국일 전체 | +1미국일 8월 현금 출발 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,b in result['benchmarks'].items():
        lines.append(f"| {name.upper()} | — | — | {pct(b['segments']['full']['return_pct'])} | {pct(b['segments']['full']['max_drawdown_pct'])} | {pct(b['fresh_holdout']['return_pct'])} | — | — |")
    compact = {}
    def row(label,index):
        p=result['profiles'][str(index)]; b=p['scenarios']['base_zero']; d=p['scenarios']['lag1_zero']
        lines.append(f"| {label} | {index} | {b['actual_max_stocks']}/{b['actual_mean_stocks']:.2f} | {pct(b['segments']['full']['return_pct'])} | {pct(b['segments']['full']['max_drawdown_pct'])} | {pct(b['fresh_holdout']['return_pct'])} | {pct(d['segments']['full']['return_pct'])} | {pct(d['fresh_holdout']['return_pct'])} |")
        compact[label]=p
    for scope,selection in result['selections'].items():
        for mode,label in [('base','기본조건 선택'),('robust','두 실행조건 강건 선택')]:
            row(label+' / '+scope_names[scope],selection[mode]['selected_index'])
    lines += ['', '강건 선택은 기본·하루지연 중 나쁜 훈련·검증 점수를 기준으로 한다. 현금 출발은8월 첫 미국거래일에 '
              '현금으로 재시작해 그날 진입 전 수익을 받지 않는다.', '', '## 이전 미국 전용 후보를 고정한 무비용 비교', '',
              '| 기존 후보 | 번호 | 실제 최대/평균 종목 | 전체 | 최대낙폭 | 8월 현금 출발 | +1미국일 전체 | +1미국일 8월 현금 출발 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label,index in result['fixed_previous_us_candidates'].items(): row(label,index)
    lines += ['', '## 선택 규칙', '']
    selected = sorted({s['selected_index'] for modes in result['selections'].values() for s in modes.values()})
    lines.extend(f"- **#{index}**: {result['profiles'][str(index)]['rule_korean']}." for index in selected)
    h=result['hindsight_best']; lines += ['',f"참고용 사후 최고값: #{h['index']}, {pct(h['segments']['full']['return_pct'])}. 전체 결과를 보고 고른 값이며 미래 추천이 아니다.", '',
        '**비용0의 뜻은 계산에서 명시적 매수·매도 비용을 차감하지 않는다는 것이다.** 실제 슬리피지·호가 차이·환전 비용·세금까지 '
        '모두0임을 검증한 것은 아니다. 정규장 종가·소수 단위 체결과 원화 환율 평가를 가정하며 세금·배당은 제외했다. '
        '최대10종목·차입 없음·현금 원화 조건과 원래 펀드 수량 신호를 유지했다. 실제 주문 수익률이나 미래ETF2~3배 수익을 보장하지 않는다.', '',
        '원래 비용 있는 결과는 보존했다. `results.json`은 모든 선택 근거·고정후보·구간 성과, `validation.json`은 검증 결과다.', '']
    atomic_write_text(out/'report.md','\n'.join(lines))
    atomic_write_json(out/'summary.json',{'research_status':STATUS,'explicit_cost_deduction':0,
        'benchmarks':result['benchmarks'],'candidates':compact,'targets':result['full_period_target_counts']})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-grid',action='store_true'); parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args(); run(reuse=args.reuse_grid,threads=args.threads)
