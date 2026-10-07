"""Frozen-rule US-listing restriction diagnostic; never reselects candidates.

Run: py -m gbrc.etf.research_us_only

Reads the original global experiment without changing it, retains its 189-date
calendar and full-fund quantity-flow normalization, filters the security axis to
metadata market == 'US', then rebuilds ranks. This is not a simulation of orders
submitted during 08:00-23:00 KST: same-close fractional fills remain assumptions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .research import OUT, benchmarks, metrics
from .research_engine import build_rankings, config_grid, simulate_detail
from .storage import atomic_write_json, atomic_write_text


FIXED = {
    'weekly_cap3': {'index': 51282, 'config': {
        'family': 'net_flow', 'lookback': 3, 'time_mix': 0.25, 'gate': 'none',
        'max_stocks': 3, 'rebalance': 'weekly', 'weighting': 'conviction', 'risk_rule': 'trend_cash'}},
    'weekly_cap5': {'index': 59742, 'config': {
        'family': 'net_flow', 'lookback': 3, 'time_mix': 0.5, 'gate': 'none',
        'max_stocks': 5, 'rebalance': 'weekly', 'weighting': 'conviction', 'risk_rule': 'trend_cash'}},
    'primary': {'index': 163958, 'config': {
        'family': 'repeated_buy', 'lookback': 10, 'time_mix': 1.0, 'gate': 'none',
        'max_stocks': 1, 'rebalance': 'weekly', 'weighting': 'equal', 'risk_rule': 'trend_cash'}},
    'time_top10_weekly': {'index': 33892, 'config': {
        'family': 'conviction', 'lookback': 1, 'time_mix': 1.0, 'gate': 'none',
        'max_stocks': 10, 'rebalance': 'weekly', 'weighting': 'conviction', 'risk_rule': 'always'}},
    'koact_top10_weekly': {'index': 196, 'config': {
        'family': 'conviction', 'lookback': 1, 'time_mix': 0.0, 'gate': 'none',
        'max_stocks': 10, 'rebalance': 'weekly', 'weighting': 'conviction', 'risk_rule': 'always'}},
}
SCENARIOS = {
    'base_15bps': {'cost': 0.0015, 'extra_global_session_lag': 0},
    'cost_30bps': {'cost': 0.003, 'extra_global_session_lag': 0},
    'lag1_15bps': {'cost': 0.0015, 'extra_global_session_lag': 1},
    'lag1_30bps': {'cost': 0.003, 'extra_global_session_lag': 1},
}
LABELS = {'weekly_cap3': '고정 주간·최대 3종목', 'weekly_cap5': '고정 주간·최대 5종목',
          'primary': '기존 주 후보·최대 1종목', 'time_top10_weekly': 'TIME 상위 10·주간',
          'koact_top10_weekly': 'KoAct 상위 10·주간'}
SCENARIO_LABELS = {'base_15bps': '기본·15bp', 'cost_30bps': '비용만 30bp',
                   'lag1_15bps': '+1관측일·15bp', 'lag1_30bps': '+1관측일·30bp'}
SECURITY_AXIS = ('symbols', 'prices', 'tradable', 'price_age_days', 'fx_age_days',
                 'weights', 'flow_pp', 'flow_kind')


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _filter_us(arrays: dict, metadata: dict) -> tuple[dict, list[dict], np.ndarray]:
    securities = metadata['securities']
    if arrays['symbols'].astype(str).tolist() != [item['symbol'] for item in securities]:
        raise ValueError('Metadata security order does not match bundle symbols')
    mask = np.array([item['market'] == 'US' for item in securities])
    if not np.any(mask):
        raise ValueError('No US listings in historical security universe')
    filtered = dict(arrays)
    for key in SECURITY_AXIS:
        if arrays[key].shape[-1] != len(mask):
            raise ValueError(f'Unexpected security axis for {key}')
        filtered[key] = arrays[key][..., mask].copy()
        np.testing.assert_equal(filtered[key], arrays[key][..., mask])
    for key in set(arrays) - set(SECURITY_AXIS):
        np.testing.assert_equal(filtered[key], arrays[key])
    selected = [item for item in securities if item['market'] == 'US']
    return filtered, selected, mask


def _summary(detail: dict, arrays: dict, security_metadata: list[dict], cap: int) -> dict:
    if not np.isfinite(detail['returns']).all():
        raise ValueError('Nonfinite simulated returns')
    if np.max(detail['counts']) > min(cap, 10):
        raise ValueError('Portfolio exceeded fixed stock cap')
    if np.min(detail['cash']) < -1e-9:
        raise ValueError('Borrowing detected')
    if not np.allclose(np.sum(detail['allocations'], axis=1) + detail['cash'], 1, atol=1e-7):
        raise ValueError('Stock and cash allocation does not sum to NAV')
    held_indices = sorted({int(n) for n in detail['holdings'].ravel() if n >= 0})
    held = [str(arrays['symbols'][n]) for n in held_indices]
    markets = sorted({security_metadata[n]['market'] for n in held_indices})
    return {
        **metrics(detail['returns']), 'max_observed_stocks': int(np.max(detail['counts'])),
        'average_observed_stocks': float(np.mean(detail['counts'])),
        'average_cash_pct': float(np.mean(detail['cash']) * 100),
        'cash_days': int(np.sum(detail['cash'] > .99999)),
        'rebalance_trade_days': int(np.sum(detail['turnover'] > 1e-8)),
        'traded_nav_multiple': float(np.sum(detail['turnover'])),
        'historically_held_symbols': held, 'held_listing_markets': markets,
        'all_held_listings_are_us': all(market == 'US' for market in markets),
        'nonnegative_cash': True, 'cap_respected': True,
    }


def run(source: Path = OUT, out: Path | None = None) -> dict:
    source = Path(source).resolve()
    out = (Path(out) if out is not None else source / 'us_only_feasibility').resolve()
    allowed = source / 'us_only_feasibility'
    if out != allowed and allowed not in out.parents:
        raise ValueError('Output must remain in original experiment/us_only_feasibility/')
    out.mkdir(parents=True, exist_ok=True)
    source_names = ('bundle.npz', 'metadata.json', 'raw_prices.json', 'configurations.json',
                    'grid_returns.npy', 'stress.json', 'evaluation.json', 'protocol.json')
    hashes = {name: _hash(source / name) for name in source_names}
    metadata = _load(source / 'metadata.json')
    if hashes['bundle.npz'] != metadata['bundle_sha256']:
        raise ValueError('Bundle does not match original metadata checksum')
    if hashes['raw_prices.json'] != metadata['raw_prices_sha256']:
        raise ValueError('Raw price cache does not match original metadata checksum')
    for item in metadata['inputs']:
        if _hash(Path(item['path'])) != item['sha256']:
            raise ValueError('Normalized source snapshot checksum changed: ' + item['path'])
    arrays = dict(np.load(source / 'bundle.npz', allow_pickle=False))
    dates = arrays['dates'].astype(str)
    if str(dates[0]) != metadata['sample_start'] or str(dates[-1]) != metadata['sample_end']:
        raise ValueError('Sample bounds disagree with original metadata')
    raw = _load(source / 'raw_prices.json')
    last, qqq = np.nan, []
    for day in dates:
        last = raw['QQQ'].get(str(day), last)
        qqq.append(last)
    arrays['qqq_usd'] = np.asarray(qqq)
    us_arrays, us_securities, mask = _filter_us(arrays, metadata)
    current_configs = _load(source / 'configurations.json')
    for label, fixed in FIXED.items():
        if current_configs[fixed['index']] != fixed['config']:
            raise ValueError(f'Saved fixed candidate index/config mismatch: {label}')
    hold = dates >= '2026-08-01'
    hold_start = int(np.flatnonzero(hold)[0])
    baseline = np.load(source / 'grid_returns.npy', mmap_mode='r')
    bench, entry = benchmarks(arrays, .0015)
    segments = {'train': dates < '2026-06-01',
                'validation': (dates >= '2026-06-01') & (dates < '2026-08-01'),
                'holdout': hold, 'full': np.ones(len(dates), dtype=bool)}
    retained = us_arrays['weights'].sum(axis=2)
    dropped_weight = arrays['weights'][..., ~mask].sum(axis=2)
    universe = {
        'listing_rule': "metadata.securities[].market == 'US'",
        'global_security_count': len(mask), 'us_security_count': int(mask.sum()),
        'removed_security_count': int((~mask).sum()), 'us_securities': us_securities,
        'removed_securities': [item for item in metadata['securities'] if item['market'] != 'US'],
        'original_us_nav_weight_pct': {name: {'mean': float(np.mean(retained[:, i])*100),
            'min': float(np.min(retained[:, i])*100), 'max': float(np.max(retained[:, i])*100)}
            for i, name in enumerate(('time', 'koact'))},
        'removed_non_us_nav_weight_pct': {name: {'mean': float(np.mean(dropped_weight[:, i])*100),
            'max': float(np.max(dropped_weight[:, i])*100)} for i, name in enumerate(('time', 'koact'))},
    }
    protocol = {
        'created_before_execution': datetime.now(timezone.utc).isoformat(),
        'task': 'Fixed global rules restricted to US-listed securities; no candidate reselection',
        'period': [str(dates[0]), str(dates[-1])], 'fixed_candidates': FIXED,
        'scenarios': SCENARIOS, 'candidate_count': len(FIXED),
        'global_calendar_preserved': dates.tolist(), 'flow_policy': 'Slice original full-fund flow_pp and '
        'flow_kind unchanged; preserve original CU/common factors, actual cash, and NAV weights; do not renormalize inputs',
        'ranking_policy': 'Recompute eligible rankings and price-feature percentiles on US security subset',
        'benchmarks': 'Original direct TIME/KoAct ETFs and QQQ in KRW, 15bp initial buy, unchanged across stresses',
        'fresh_holdout': 'Cash start at first August date; can enter that local close only; prior-day gain not earned',
        'not_execution_validation': 'Universe restriction diagnostic on the original global calendar; '
        'not a replay of orders at 08:00-23:00 KST; assumes same-close fractional fills and daily FX marks',
        'calendar_limitation': 'First global date of a week may be a US holiday. Closed holdings are frozen; '
        'weekly rules do not automatically retry next US session. The global 20-observation features also remain unchanged in time.',
        'selection_policy': 'Never choose a new candidate from US-only realised returns; frozen IDs remain in declared order',
        'cost_cash_assumptions': '15/30bp per side; zero interest on KRW cash; tax/dividends excluded; no leverage',
        'listing_vs_nationality': 'US listing rule includes foreign-company US ADRs/listings such as SKHY and TSM; '
        'Korean/Japanese/Hong Kong/Chinese local-exchange listings are removed.',
        'point_in_time_limitation': metadata['backfill_temporal_limitation'],
        'input_sha256': hashes,
        'code_sha256': {name: _hash(Path(__file__).with_name(name))
                        for name in ('research_us_only.py', 'research_engine.py', 'research.py')},
    }
    atomic_write_json(out / 'protocol.json', protocol)
    results = {'schema_version': 1, 'research_status': 'fixed-rule universe restriction diagnostic; '
               'retrospective and not executable order validation', 'protocol': protocol,
               'universe': universe, 'candidates': {}, 'benchmarks': {}, 'validation': {}}
    curves = {'dates': dates.tolist(), 'series': {}}
    difference = {}
    params = None
    rank_configs = None
    for mode, inputs, securities in [('global', arrays, metadata['securities']),
                                      ('us_only', us_arrays, us_securities)]:
        print(f'Rebuilding {mode} ranks; security count={len(inputs["symbols"])}', flush=True)
        rankings = {lag: build_rankings(inputs, extra_lag=lag) for lag in (0, 1)}
        if params is None:
            params, generated = config_grid(rankings[0]['rank_configs'])
            rank_configs = rankings[0]['rank_configs']
            if generated != current_configs:
                raise ValueError('Current engine configuration grid differs from saved experiment')
        for lag in (0, 1):
            if rankings[lag]['rank_configs'] != rank_configs:
                raise ValueError('Security filtering changed parameter index interpretation')
            np.testing.assert_equal(rankings[lag]['schedule'], rankings[0]['schedule'])
        for label, fixed in FIXED.items():
            row = results['candidates'].setdefault(label, {'index': fixed['index'], 'config': fixed['config']})
            scenarios = {}
            for scenario_name, scenario in SCENARIOS.items():
                lag, cost = scenario['extra_global_session_lag'], scenario['cost']
                detail = simulate_detail(params[fixed['index']], inputs, rankings[lag], cost=cost)
                full = _summary(detail, inputs, securities, fixed['config']['max_stocks'])
                if mode == 'us_only' and not full['all_held_listings_are_us']:
                    raise ValueError('Non-US holding escaped universe restriction')
                fresh = simulate_detail(params[fixed['index']], inputs, rankings[lag], cost=cost, start=hold_start)
                fresh_summary = _summary(fresh, inputs, securities, fixed['config']['max_stocks'])
                if mode == 'us_only' and not fresh_summary['all_held_listings_are_us']:
                    raise ValueError('Non-US fresh-start holding escaped restriction')
                scenarios[scenario_name] = {
                    'segments': {key: metrics(detail['returns'][selected]) for key, selected in segments.items()},
                    'portfolio': full, 'fresh_holdout': fresh_summary,
                }
                curves['series'][f'{mode}/{label}/{scenario_name}'] = np.cumprod(1 + detail['returns']).tolist()
                if mode == 'global' and scenario_name == 'base_15bps':
                    error = float(np.max(np.abs(detail['returns'] - baseline[fixed['index']])))
                    if error > 1e-7:
                        raise ValueError(f'Original global reproduction failed: {label}, {error}')
                    difference[label] = error
            row[mode] = scenarios
        del rankings
    for j, name in enumerate(('time', 'koact', 'qqq_krw')):
        fresh = np.r_[-.0015 / 1.0015,
                      arrays['benchmarks'][hold_start+1:, j] / arrays['benchmarks'][hold_start:-1, j] - 1]
        results['benchmarks'][name] = {
            'segments': {key: metrics(bench[selected, j]) for key, selected in segments.items()},
            'fresh_holdout': metrics(fresh), 'initial_entry': str(dates[entry]),
            'fresh_entry': str(dates[hold_start]), 'one_way_entry_cost_bps': 15,
        }
        curves['series'][f'benchmark/{name}'] = np.cumprod(1 + bench[:, j]).tolist()
    after = {name: _hash(source / name) for name in source_names}
    if after != hashes:
        raise ValueError('Original experiment source files changed during diagnostic')
    results['validation'] = {
        'source_files_unchanged': True, 'normalized_snapshot_checksums_verified': len(metadata['inputs']),
        'saved_and_current_code_configurations_match': True,
        'fixed_candidate_indices_verified': {key: value['index'] for key, value in FIXED.items()},
        'security_metadata_order_verified': True, 'original_flow_values_preserved': True,
        'original_nav_input_weights_not_renormalized': True,
        'non_security_axes_and_global_calendar_preserved': True,
        'original_global_daily_returns_max_abs_errors': difference,
        'all_us_runs_only_us_listings': True, 'all_runs_cap_respected': True,
        'all_runs_nonnegative_cash_and_nav_partition_verified': True,
        'simulation_runs': len(FIXED)*2*len(SCENARIOS)*2,
        'new_candidate_selection_performed': False,
    }
    atomic_write_json(out / 'results.json', results)
    atomic_write_json(out / 'validation.json', results['validation'])
    atomic_write_json(out / 'equity_curves.json', curves)
    atomic_write_text(out / 'report.md', _report(results))
    print(json.dumps({'out': str(out), 'us_security_count': len(us_securities),
                      'validation': results['validation']}, ensure_ascii=False, indent=2), flush=True)
    return results


def _pct(value: float) -> str:
    return f'{value:+.2f}%'


def _table(columns: list[str], rows: list[list[Any]]) -> str:
    return ('| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join('---' for _ in columns) + ' |\n'
            + '\n'.join('| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows) + '\n')


def _report(result: dict) -> str:
    u = result['universe']
    text = ['# 미국 상장 종목만 허용한 고정 규칙 재계산\n',
            f"기존 2026-01-02~09-23 연구의 {u['global_security_count']}종목에서 미국 상장 "
            f"{u['us_security_count']}종목만 남겼다. **기존 후보 다섯 개의 규칙과 번호를 고정하고 재계산했으며, "
            '새 수익률을 보고 후보를 다시 고르지 않았다.**\n',
            '원래 글로벌 연구는 그대로 보존했다. 이 결과는 미국 종목 제한의 영향만 분리한 진단이며, '
            '**08~23시 KST 실제 주문이나 실계좌 수익률을 재현하지 않는다.**\n',
            '## 기본 비용 15bp: 글로벌과 미국 전용 비교\n']
    rows = []
    for label, candidate in result['candidates'].items():
        g, us = [candidate[mode]['base_15bps'] for mode in ('global', 'us_only')]
        rows.append([LABELS[label], candidate['index'], _pct(g['segments']['full']['return_pct']),
                     _pct(us['segments']['full']['return_pct']),
                     _pct(us['segments']['full']['max_drawdown_pct']),
                     _pct(g['fresh_holdout']['return_pct']), _pct(us['fresh_holdout']['return_pct']),
                     us['portfolio']['max_observed_stocks']])
    text.append(_table(['고정 규칙', '번호', '글로벌 전체', '미국 전용 전체', '미국 전용 최대낙폭',
                        '글로벌 8월 현금 출발', '미국 전용 8월 현금 출발', '미국 전용 실제 최대종목'], rows))
    text += ['## 미국 전용 비용·지연 민감도\n',
             '추가 지연은 미국 거래일이 아닌 기존 글로벌 관측일 한 칸이다. 8~9월 기존 보유 성과는 '
             '1월부터 이어진 계좌의 구간 손익이고, 현금 출발은 8월 3일 이전 상승분을 받지 않는다.\n']
    rows = []
    for label, candidate in result['candidates'].items():
        for name, scenario in candidate['us_only'].items():
            rows.append([LABELS[label], SCENARIO_LABELS[name], _pct(scenario['segments']['full']['return_pct']),
                         _pct(scenario['segments']['full']['max_drawdown_pct']),
                         _pct(scenario['segments']['holdout']['return_pct']),
                         _pct(scenario['fresh_holdout']['return_pct']),
                         scenario['portfolio']['max_observed_stocks']])
    text.append(_table(['고정 규칙', '조건', '전체', '전체 최대낙폭', '기존 보유 8~9월', '8월 현금 출발',
                        '실제 최대종목'], rows))
    text.append('## 같은 기준의 비교 ETF\n')
    text.append(_table(['비교 대상', '전체', '기존 보유 8~9월', '8월 현금 출발'], [
        [name, _pct(row['segments']['full']['return_pct']), _pct(row['segments']['holdout']['return_pct']),
         _pct(row['fresh_holdout']['return_pct'])] for name, row in result['benchmarks'].items()]))
    text += ['비교 ETF는 모든 민감도에서 기존 15bp 진입 경로를 유지했다. 불리한 전략 조건에 맞춰 ETF '
             '수익률을 낮추지 않았다. ETF 수익률의 2배는 누적 이익률의 배수이며 최종 자산 배수가 아니다.\n',
             '## 규칙·데이터 검증과 해석\n']
    text.append(f"원래 펀드 NAV에서 미국 상장 종목의 평균 비중은 TIME "
                f"{u['original_us_nav_weight_pct']['time']['mean']:.2f}%, KoAct "
                f"{u['original_us_nav_weight_pct']['koact']['mean']:.2f}%였다. 입력 비중은 그대로 유지했지만 "
                '실제 전략 배분은 허용된 미국 종목으로 집중되므로, 원래 펀드의 복제가 아닌 미국 보유 부분의 '
                '집중 전략으로 성격이 바뀐다.\n')
    rules = [
        '전체 펀드에서 이미 계산한 수량 기반 실매매·설정/환매 공통 배율을 보존했다. 종목을 줄인 뒤 CU '
        '배율을 다시 추정하거나 미국 종목 입력 NAV 비중을 100%로 재정규화하지 않았다.',
        '종목축만 미국 상장으로 잘라 순위·가격 특성 순위를 다시 계산했다. 날짜·원래 현금·관찰기간·신호 '
        '공개 지연·ETF 벤치마크는 유지했다.',
        '미국 상장 기준이므로 SKHY·TSM 같은 외국 기업의 미국 상장 주식/ADR은 포함한다. 한국·일본·홍콩·중국 '
        '현지 거래소 종목은 제외했다.',
        '주간 교체일은 기존 글로벌 날짜 합집합의 주 첫 관측일이다. 미국 휴장일이면 기존 보유를 유지하며 '
        '다음 미국 거래일에 자동 재시도하지 않는다. 미국 전용 실제 운영 달력은 아직 구현하지 않았다.',
        '직전까지 알려진 신호로 같은 날 종가 체결·소수점 주식·일별 환율 평가를 가정했다. 실제 주문 시각, '
        '정수 수량, 환전·결제 자금·호가·미체결을 재현하지 않는다.',
        '원화 현금 이자는 0%, 편도 비용은 15/30bp 가정이다. 세금·배당은 제외했고 레버리지는 사용하지 않았다. '
        '최대 보유 수는 상한이며 최소 분산 수가 아니다.',
        '과거 스냅샷은 사후 수집 자료다. 8~9월도 이미 관측한 과거 검증이며 미래 수익·실행 가능성이 입증된 것은 아니다.',
        f"원본 파일 해시 불변, 스냅샷 {result['validation']['normalized_snapshot_checksums_verified']}개 해시, "
        '저장된 설정과 현재 코드의 인덱스·설정 일치, 글로벌 기본 경로 재현, 모든 미국 전용 보유의 US성, '
        '최대 10종목·후보별 상한, 현금 비음수·NAV 합계를 확인했다.',
    ]
    text.append('\n'.join('- ' + rule for rule in rules) + '\n')
    text += ['## 고정된 설정\n']
    text.append(_table(['번호', '규칙', '관찰기간', 'TIME/KoAct 신호', '배분', '방어'], [
        [row['index'], row['config']['family'], f"D-{row['config']['lookback']}",
         f"{row['config']['time_mix']:.0%}/{1-row['config']['time_mix']:.0%}",
         row['config']['weighting'], row['config']['risk_rule']] for row in result['candidates'].values()]))
    text += ['모두 주간 교체이며 가격 필터는 없다. trend_cash는 교체일에 하락 신호일 때 현금 전환을 '
             '목표로 하고, always는 유효 종목이 있으면 주식 노출을 유지한다.\n',
             '## 재실행과 원자료\n', '`py -m gbrc.etf.research_us_only`\n',
             '- [전체 결과와 글로벌 비교](results.json)\n- [실행 전 고정 규약](protocol.json)\n'
             '- [검증 결과](validation.json)\n- [같은 날짜 축의 자산곡선](equity_curves.json)\n',
             '### ✅ 자기검증 (Self-Check)\n',
             '- 트리거 매칭: [Path-F ✅]\n- Gate B 통과: [ETF 정본 및 기존 연구 코드 Read + 고정 규칙 재계산 Y]\n'
             '- 가정으로 채운 숫자: [있음 — 편도 비용 15/30bp, 현금 이자 0%, 종가 소수점 체결; 원천 수량·가격·환율 대체 없음]\n'
             '- 돌발 호재 확인 (Path-O/V 한정): [불필요 ✅]\n- 규칙 위반: [없음 ✅]\n']
    return '\n'.join(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=OUT)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    run(args.source, args.out)


if __name__ == '__main__':
    main()
