"""Read-only reproduction and return-attribution audit of Claude section 16.

Writes only this directory's contribution_audit.json and contribution_audit.md.
No reselection, changed execution rules, price downloads or original-file writes.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ['NUMBA_DISABLE_JIT'] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
SRC = HERE.parent.parent / 'us_only_experiment'
LAB = REPO / 'data/derived/analysis/strategy_lab'
sys.path.insert(0, str(REPO))

import numpy as np
from gbrc.etf.research_engine import build_rankings, config_grid, simulate_detail


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def period_audit(dates, returns, owners, start, end):
    ix = np.flatnonzero((dates >= start) & (dates <= end))
    r = returns[ix]
    nav = np.cumprod(1 + r)
    prev = np.r_[1., nav[:-1]]
    dollars = prev * r
    total_log_growth = float(np.log1p(r).sum())
    rows = []
    for symbol in sorted(set(owners[ix])):
        sel = owners[ix] == symbol
        rows.append({'symbol': symbol, 'return_sessions': int(sel.sum()),
                     'simple_daily_sum_pct': float(r[sel].sum() * 100),
                     'capital_contribution_pp': float(dollars[sel].sum() * 100),
                     'fraction_of_total_log_growth': float(np.log1p(r[sel]).sum() / total_log_growth) if total_log_growth else None,
                     'compounded_held_sessions_return_pct': float((np.prod(1 + r[sel]) - 1) * 100)})
    rows.sort(key=lambda row: -row['capital_contribution_pp'])
    total = float((nav[-1] - 1) * 100)
    summed = sum(row['capital_contribution_pp'] for row in rows)
    np.testing.assert_allclose(summed, total, atol=1e-10)
    return {'start': str(dates[ix[0]]), 'end': str(dates[ix[-1]]), 'sessions': len(ix),
            'total_return_pct': total, 'simple_daily_sum_pct': float(r.sum() * 100),
            'capital_contribution_sum_pp': summed, 'attribution_residual_pp': summed - total,
            'attribution_base': '1 unit of NAV at the previous close before this period', 'by_symbol': rows}


def holding_runs(dates, returns, post_close_names):
    nav = np.cumprod(1 + returns)
    prev = np.r_[1., nav[:-1]]
    starts = np.r_[0, np.flatnonzero(post_close_names[1:] != post_close_names[:-1]) + 1]
    rows = []
    for i, start in enumerate(starts):
        next_start = int(starts[i + 1]) if i + 1 < len(starts) else len(dates)
        # Buy at close[start], earn starting next date; exit-date return belongs
        # to the old holding, not to the position bought at that day's close.
        ix = np.arange(start + 1, min(next_start + 1, len(dates)))
        rr = returns[ix]
        rows.append({'symbol': str(post_close_names[start]), 'entry_close_date': str(dates[start]),
                     'last_post_close_holding_date': str(dates[next_start - 1]),
                     'exit_close_date': str(dates[next_start]) if next_start < len(dates) else None,
                     'valuation_date_if_unclosed': str(dates[-1]) if next_start == len(dates) else None,
                     'first_earned_return_date': str(dates[ix[0]]) if len(ix) else None,
                     'last_earned_return_date': str(dates[ix[-1]]) if len(ix) else None,
                     'simple_daily_sum_pct': float(rr.sum() * 100),
                     'compounded_run_return_pct': float((np.prod(1 + rr) - 1) * 100),
                     'capital_contribution_pp_of_original_nav': float((prev[ix] * rr).sum() * 100)})
    return rows


def main():
    inputs = [SRC / 'bundle.npz', SRC / 'raw_prices.json', SRC / 'configurations.json',
              SRC / 'zero_cost/results.json', LAB / 'code/compare_codex.py', LAB / 'logs/compare_codex.txt',
              LAB / '2026-09-23_strategy_lab.md', REPO / 'gbrc/etf/research_engine.py']
    before = {str(p.relative_to(REPO)): sha(p) for p in inputs}
    arrays = dict(np.load(SRC / 'bundle.npz', allow_pickle=False))
    dates = arrays['dates'].astype(str)
    raw = read(SRC / 'raw_prices.json')
    arrays['qqq_usd'] = np.array([raw['QQQ'][str(day)] for day in dates])
    rankings = build_rankings(arrays)
    params, configs = config_grid(rankings['rank_configs'])
    cid = 239120
    detail = simulate_detail(params[cid], arrays, rankings, cost=0.0)
    saved = np.load(SRC / 'zero_cost/grid_returns.npy', mmap_mode='r')[cid]
    np.testing.assert_allclose(detail['returns'], saved, atol=1e-6)
    returns = detail['returns']
    symbols = arrays['symbols'].astype(str)
    assert np.max(np.sum(detail['holdings'] >= 0, axis=1)) == 1
    allocated = detail['allocations'].sum(axis=1)
    np.testing.assert_allclose(allocated[allocated > 0], 1., atol=1e-10)
    post_names = np.array([symbols[j] if j >= 0 else 'CASH' for j in detail['holdings'][:, 0]])
    owners = np.concatenate([np.array(['CASH']), post_names[:-1]])
    assert np.allclose(returns[owners == 'CASH'], 0.)
    periods = {
        'section16_table_through_sep22': period_audit(dates, returns, owners, '2026-01-02', '2026-09-22'),
        'section16_contribution_log_through_sep23': period_audit(dates, returns, owners, '2026-01-02', '2026-09-23'),
        'june_july_selection_period': period_audit(dates, returns, owners, '2026-06-01', '2026-07-31'),
    }
    keep = dates <= '2026-09-22'
    runs = holding_runs(dates[keep], returns[keep], post_names[keep])
    sr = read(SRC / 'zero_cost/results.json')['selections']['all']['base']
    selection = {k: sr[k] for k in ('scope_count', 'eligible_training_count', 'shortlist_count', 'selected_index')}
    selection.update({'selected_is_first_validation_rank': sr['validation_ranked_shortlist_indices'][0] == cid,
                      'objective': 'log compounded return minus half absolute maximum drawdown',
                      'train': 'January-May', 'selection': 'June-July',
                      'interpretation': 'Selection bias is a valid concern; this does not measure its numerical size or establish that the whole validation gain is spurious.'})
    stock_runs = [r for r in runs if r['symbol'] != 'CASH']
    by_symbol = periods['section16_table_through_sep22']['by_symbol']
    lite_arm = sum(r['capital_contribution_pp'] for r in by_symbol if r['symbol'] in ('LITE', 'ARM'))
    top_runs = sorted(stock_runs, key=lambda r: -r['capital_contribution_pp_of_original_nav'])[:2]
    concentration = {'stock_holding_runs': len(stock_runs), 'distinct_stocks': len(set(r['symbol'] for r in stock_runs)),
                     'state_transitions_including_cash_to_first_stock': len(runs) - 1,
                     'stock_to_stock_transitions': len(stock_runs) - 1,
                     'lite_plus_arm_capital_contribution_pp': lite_arm,
                     'lite_plus_arm_fraction_of_total_profit': lite_arm / periods['section16_table_through_sep22']['total_return_pct'],
                     'lite_plus_arm_fraction_of_log_growth': sum(r['fraction_of_total_log_growth'] for r in by_symbol if r['symbol'] in ('LITE', 'ARM')),
                     'lite_plus_arm_fraction_of_simple_daily_sum': sum(r['simple_daily_sum_pct'] for r in by_symbol if r['symbol'] in ('LITE', 'ARM')) / periods['section16_table_through_sep22']['simple_daily_sum_pct'],
                     'top_two_runs': top_runs,
                     'top_two_runs_fraction_of_total_profit': sum(r['capital_contribution_pp_of_original_nav'] for r in top_runs) / periods['section16_table_through_sep22']['total_return_pct']}
    first_t = int(np.flatnonzero(post_names != 'CASH')[0])
    first_si = int(rankings['signal_snapshot_index'][first_t])
    rid, _, _, wi, _ = params[cid]
    eligible_before = np.sum(rankings['ranks'][rid, :first_t] >= 0, axis=1)
    initial_cash = {'first_cash_date': str(dates[0]), 'last_post_close_cash_date': str(dates[first_t - 1]),
                    'post_close_cash_us_sessions': int(np.sum(post_names == 'CASH')),
                    'first_buy_close_date': str(dates[first_t]), 'first_gain_date': str(dates[first_t + 1]),
                    'latest_snapshot_used_on_first_entry': str(arrays['snapshot_dates'][first_si]),
                    'snapshot_index_used_on_first_entry': first_si,
                    'first_entry_flow_base_dates': arrays['flow_base_dates'][first_si, 5].astype(str).tolist(),
                    'eligible_rank_rows_before_first_entry': int(np.sum(eligible_before > 0)),
                    'interpretation': 'The initial cash span is verified; the 40-day flow lookback uses original fund snapshot/flow-base dates, not a 43-US-session market-timing decision.'}
    after = {str(p.relative_to(REPO)): sha(p) for p in inputs}
    assert before == after
    result = {'candidate_id': cid, 'configuration': configs[cid],
              'scope': 'Original zero-cost US-only close-execution base path; NOT the new common open-execution comparison',
              'stored_grid_reproduction_max_abs_error': float(np.max(np.abs(returns - saved))),
              'attribution_formula': 'contribution[t] = NAV[t-1] / NAV[period_start-1] * return[t]; assign to previous close holding. All contributions sum exactly to the compounded period return.',
              'simple_sum_warning': 'sum(return[t]) by stock is not initial-capital P&L; do not add such values to explain a compounded total.',
              'periods': periods, 'holding_runs_through_sep22': runs, 'concentration': concentration,
              'selection': selection, 'initial_cash': initial_cash,
              'source_sha256': before, 'original_inputs_unchanged': True}
    (HERE / 'contribution_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    full = periods['section16_table_through_sep22']; validation = periods['june_july_selection_period']
    lines = ['# §16 수익 기여·선정 주장 감사', '',
             '원본 #239120의 비용 0원·미국 전용·종가 base 경로만 재현했다. 새 공통 시가 엔진 결과와 구분한다. 원본 입력은 수정하지 않았다.', '',
             '## 핵심 판정', '',
             '- LITE +56%p·ARM +45%p는 코드가 명시한 **일별 수익률 단순합**이다. 초기자본 대비 수익 기여가 아니다. 원본 기여 로그는 9/23까지, 비교표는 9/22까지여서 종료일도 다르다.',
             '- 6~7월 +37%는 전체 포트폴리오의 복리수익이다. 이를 SNOW 한 종목의 수익 또는 단독 기여로 설명하면 부정확하다. 아래 표에서 종목별 정상 자본 기여를 대조한다.',
             '- 2,519개 후보 중 6~7월 목적함수로 최종 선택했다는 지적은 맞다. 이 구간은 선택에 사용됐으므로 독립 검증이 아니다. 다만 이 사실만으로 성과가 얼마나 부풀려졌는지 수치로 확정할 수는 없다.',
             f'- 1/2~3/5 현금, 3/6 종가 첫 매수는 재현된다. 14는 현금→최초 주식 진입 1회와 이후 주식 간 교체 13회의 합계다. 고유 종목은 {concentration["distinct_stocks"]}개다.', '',
             '## 1/2~9/22 원본 종가 경로의 기여', '',
             f'총 복리수익 **{full["total_return_pct"]:+.4f}%**. 정상 기여는 `직전 NAV × 당일 수익률`을 전일 종가 보유종목에 배분한다. 아래 기여 합계는 총 수익과 일치한다.', '',
             '| 종목 | 일별 수익률 단순합 | 초기자본 대비 기여 |', '|---|---:|---:|']
    lines += [f'| {r["symbol"]} | {r["simple_daily_sum_pct"]:+.4f}% | {r["capital_contribution_pp"]:+.4f}%p |' for r in by_symbol]
    lines += ['', f'LITE+ARM의 정상 자본 기여는 **{lite_arm:+.4f}%p**, 전체 이익의 **{concentration["lite_plus_arm_fraction_of_total_profit"]*100:.2f}%**다. 두 회사는 여러 번에 나눠 보유했으므로 “두 번의 거래”와도 다르다.',
              f'실제 보유 구간 중 초기자본 기여가 가장 큰 두 구간의 합계는 전체 이익의 **{concentration["top_two_runs_fraction_of_total_profit"]*100:.2f}%**다. “두 번이 대부분”이라는 표현은 무엇을 두 번으로 세었는지와 기여 계산을 제시해야 한다.', '',
              f'다른 기여 정의에서는 결론의 강도가 달라진다. LITE+ARM은 일별 수익률 단순합의 **{concentration["lite_plus_arm_fraction_of_simple_daily_sum"]*100:.2f}%**, 가산 가능한 로그 성장의 **{concentration["lite_plus_arm_fraction_of_log_growth"]*100:.2f}%**를 차지한다. 따라서 두 회사가 성장에 크게 기여했고 집중 위험이 있다는 지적은 타당하다. 초기자본 기여는 앞선 이익을 재투자한 실제 경로의 손익회계 분해이며, 유일한 인과적 공로 배분법은 아니다.', '',
              '## 6~7월 선택 구간', '',
              f'구간 시작 자본을 1로 다시 표시하면 전체 복리수익은 **{validation["total_return_pct"]:+.4f}%**다. 이는 신규 현금 매수가 아니라 기존 보유를 이어가는 구간 수익이다.', '',
              '| 종목 | 일별 수익률 단순합 | 6월 직전 자본 대비 기여 |', '|---|---:|---:|']
    lines += [f'| {r["symbol"]} | {r["simple_daily_sum_pct"]:+.4f}% | {r["capital_contribution_pp"]:+.4f}%p |' for r in validation['by_symbol']]
    lines += ['', '## 개별 보유 구간: 단순합과 복리수익', '',
              '진입일 종가에 매수하므로 그날 수익은 받지 않는다. 교체일 종가까지의 수익은 매도하는 이전 종목에 귀속된다. 마지막 미청산 보유는 9/22 종가로 평가한다.', '',
              '| 종목 | 매수 종가일 | 매도 종가일/말일 평가 | 일별 단순합 | 보유 구간 복리 | 초기자본 기여 |',
              '|---|---|---|---:|---:|---:|']
    lines += [f'| {r["symbol"]} | {r["entry_close_date"]} | {r["exit_close_date"] or r["valuation_date_if_unclosed"]} | {r["simple_daily_sum_pct"]:+.3f}% | {r["compounded_run_return_pct"]:+.3f}% | {r["capital_contribution_pp_of_original_nav"]:+.3f}%p |' for r in stock_runs]
    lines += ['', '## 선발 근거와 해석 범위', '',
              f'- 전체 {selection["scope_count"]:,}개 중 1~5월 낙폭·유효성 조건을 통과한 {selection["eligible_training_count"]:,}개의 상위 1%를 올림하여 {selection["shortlist_count"]:,}개를 만들었다. 그 안에서 6~7월 `로그 누적수익 − 0.5 × 최대낙폭 절댓값` 1위가 #239120이다. 단순 누적수익 1위와는 다른 정의다.',
              '- 8~9월을 직접 선발에 쓰지는 않았지만, 이전 실험에서 이미 본 기간을 재사용한 회고적 평가다. 독립적인 미래 검증으로 표현하지 않는다.',
              '- 초기 현금 기간은 신호 준비·자격 조건에 따른 결과이며 하락장을 성공적으로 예측했다는 증거가 아니다. 40일 창은 펀드 스냅샷·원본 flow-base 달력을 따른다.',
              '- 원본 종가 경로의 특정 종목 집중과 선택 편향 우려는 타당하다. 단순합을 자본 기여로 부르거나 전체 6~7월 수익을 SNOW 단독 성과로 귀속하는 표현은 정정해야 한다.',
              '- 정규화한 자본 기여는 실제 지나간 순서와 운용 자본에 의존한다. 특정 종목을 제거했을 때의 대체 전략 수익이나 독립적 인과 효과를 뜻하지 않는다.', '',
              '입력 해시·전체 소수점·보유 구간·초기 현금·선정 근거는 [contribution_audit.json](contribution_audit.json)에 저장했다. 재현 코드는 [contribution_audit.py](contribution_audit.py)다.', '']
    (HERE / 'contribution_audit.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'periods': periods, 'concentration': concentration, 'initial_cash': initial_cash,
                      'selection': selection, 'original_inputs_unchanged': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
