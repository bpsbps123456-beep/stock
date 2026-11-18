"""Render the Korean report from the fixed protocol and completed result ledgers."""
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
NAMES = {
    'baseline': '필터 없음', 'sector_sma20': 'SOXX 20일선 위',
    'above_vwap': 'SOXL 당일 VWAP 이상', 'late_up': '최근 30분 상승',
    'rsi_turn': '1분 RSI 30 이하 반등', 'down5': 'SOXL 당일 5% 이상 하락',
    'bb1': 'BB-1 바닥 신호',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pct(value):
    return '—' if pd.isna(value) else f'{value:+.2f}%'


def table(headers, rows):
    lines = [headers, ['---'] * len(headers), *rows]
    assert all(len(row) == len(headers) for row in lines)
    return '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in lines)


def render():
    spec = read_json(HERE / 'protocol.json')
    summary = read_json(OUT / 'summary.json')
    audit = read_json(OUT / 'validation.json')
    inputs = read_json(OUT / 'input_audit.json')
    assert audit['pass'] and audit['new_last_day'] == spec['end']
    metrics = pd.read_csv(OUT / 'metrics.csv').set_index(['period', 'window', 'gap_tp', 'filter']).sort_index()
    monthly = pd.read_csv(OUT / 'monthly.csv').set_index(['filter', 'month'])
    early = pd.read_csv(OUT / 'earlier_cutoff_metrics.csv').set_index('filter')
    assert all(frame.index.is_unique for frame in (metrics, monthly, early))
    trades = pd.read_csv(OUT / 'trades.csv')
    belt_trades = trades[(trades['filter'] == 'bb1') & (trades.run_start == '2026-07-01') & (trades.window == 'RTH') & (trades.gap_tp == 'open')]
    gaps = belt_trades[belt_trades.reason.eq('take_profit_gap')]
    ids = list(spec['filters'])
    assert set(ids) == set(NAMES)

    def block(period, window='RTH', gap='open'):
        return metrics.loc[(period, window, gap), :].loc[ids]

    h1, summer, h2, sept = map(block, ['h1', 'jul_aug', 'h2', 'sep_on'])
    capped, extended = block('h2', gap='target'), block('h2', window='EXT')

    def chosen(train, test, key):
        pick = summary[key]
        if pick is None:
            return '자격을 충족한 후보가 없어 현금 유지', '거래 0건, 수익 0.00%'
        return (f'**{NAMES[pick]}** · {pct(train.at[pick, "net10"])}',
                f'**{pct(test.at[pick, "net10"])}** · 청산 {test.at[pick, "closed"]}건')

    h1_choice, h2_result = chosen(h1, h2, 'h1_pick')
    summer_choice, sept_result = chosen(summer, sept, 'summer_pick')
    top = h2.drop(index='baseline').net10.idxmax()
    component = pd.read_csv(OUT / 'belt_components.csv').set_index(['filter', 'gap_tp'])
    deep = pd.read_csv(OUT / 'drawdown_only.csv').set_index(['start', 'window', 'gap_tp'])
    july = deep.loc[('2026-07-01', 'RTH', 'open')]
    flags = pd.read_csv(OUT / 'features.csv', index_col='day')
    h1_deep = int(flags.loc[:'2026-06-30', 'bb1_prior_soxx_drawdown60'].le(-.20).sum())
    top_row = h2.loc[top]
    baseline = h2.loc['baseline']
    positive = int((h2.drop(index='baseline').net10 > 0).sum())
    selections = audit['selection']
    forward_gains = sum(s['forward_return_pct'] > 0 for s in selections)

    sections = [
        '# SOXL 종가 매수에 조건 하나를 더하면 7월 이후 수익이 남았나?',
        f'미국 거래일 2026-01-02~{spec["end"]} · 7월 1일 현금 출발 {int(baseline.sessions)}거래일 · '
        '정규장 종가 매수, 매수가 대비 −3% 손절·+2% 익절',
        f'7월 1일 이후 매일 조건 없이 시도한 전략은 **{pct(baseline.net10)}**, 같은 기간 SOXL 단순 보유는 '
        f'**{pct(summary["buy_hold_h2"])}**였다. 단일 필터 6개 중 {positive}개가 수익을 냈다. '
        f'하반기 결과를 본 뒤 고른 최고는 **{NAMES[top]} {pct(top_row.net10)}**였다. '
        f'이 거래들 중 가장 큰 이익 한 건을 빼면 {pct(top_row.without_best10)}다.',
        f'**앞 구간에서 고른 조건이 다음 구간에서도 이익을 낸 경우는 {len(selections)}번 중 {forward_gains}번이었다.** BB-1은 청산 {h2.at["bb1", "closed"]}건 중 {len(gaps)}건이 '
        f'목표가를 넘어선 상승 갭에서 팔린 것으로 계산됐고, 그 거래들의 비용 후 평균 이익은 {pct(gaps.net_10bp_pct.mean())}였다. '
        f'익절 체결가를 +2%에 제한하면 BB-1 누적은 **{pct(capped.at["bb1", "net10"])}**, 연장시간에도 청산하면 {pct(extended.at["bb1", "net10"])}다. '
        f'BB-1의 {len(belt_trades)}건은 모두 11월 12일 신호 등록 전에 청산됐다. 마지막 거래는 {belt_trades.exit_day.max()}에 끝났다. 1분 RSI 반등도 {h2.at["rsi_turn", "closed"]}건으로 '
        f'{pct(h2.at["rsi_turn", "net10"])}였지만 최고 이익 한 건 제외 시 {pct(h2.at["rsi_turn", "without_best10"])}다.',
        '## 반도체가 깊게 하락한 날만 고르면',
        f'BB-1 수익을 확인한 뒤 두 조건의 역할을 분해했다. 수급 확인이 가능한 7/1~11/11의 같은 {component.iloc[0].flow_known_days:.0f}거래일을 대상으로 '
        '전일 SOXX가 직전 60거래일 최고 종가보다 20% 이상 하락한 날과 여기에 한국인 SOXL 매수 쏠림까지 확인한 날을 비교했다.',
        table(['7/1~11/11 조건', '신호일', '청산', '상승 갭 시가 체결', '상승 갭 +2% 상한'], [
            [label, int(component.at[(key, 'open'), 'signals']), int(component.at[(key, 'open'), 'closed']),
             pct(component.at[(key, 'open'), 'net10']), pct(component.at[(key, 'target'), 'net10'])]
            for key, label in [('drawdown_only', '낙폭 20%만'), ('bb1', '낙폭 20%와 수급 쏠림')]
        ]),
        f'수급자료가 없는 날에도 전일 SOXX 가격은 알 수 있다. 이를 포함한 7/1~11/17 전체에서 낙폭만 쓰면 '
        f'**{pct(july.net10)}**(청산 {int(july.closed)}건·미청산 {int(july.open_positions)}건), 최대낙폭 {pct(july.mdd10)}였다. '
        f'9/1 현금 출발은 {pct(deep.at[("2026-09-01","RTH","open"),"net10"])}, 7/1 출발에 상승 갭 +2% 상한을 적용하면 '
        f'{pct(deep.at[("2026-07-01","RTH","target"),"net10"])}, 장전·장후 청산 포함 시 {pct(deep.at[("2026-07-01","EXT","open"),"net10"])}였다. '
        f'상반기에 이 낙폭 조건이 켜진 날은 {h1_deep}일이었다. 이 분해는 6개 필터의 시간순 선정 뒤에 한 사후 확인이다. [같은 날짜의 수급 추가 효과](results/belt_components.csv) · [낙폭만 적용한 전체 기간](results/drawdown_only.csv)',
        '## 앞 구간에서 골랐을 때',
        table(['선정 구간', '당시 성과로 고른 조건 · 선정 구간 수익', '다음 평가 구간', '다음 구간 수익 · 청산'], [
            ['1/2~6/30', h1_choice, '7/1~11/17', h2_result],
            ['7/1~8/31', summer_choice, '9/1~11/17', sept_result],
        ]),
        '일반 필터 5개 중 상반기에는 청산 20건 이상, 7~8월에는 10건 이상이며, 편도 10bp 비용 후 수익이 양수인 후보의 1위를 골랐다. '
        '무필터는 비교 기준이고 BB-1은 11월 12일 등록되어 선정 대상에서 제외했다. 각 검증 구간은 첫날 현금 100으로 시작한다. '
        '2026년 하반기 무필터 성과를 이미 본 상태에서 추가 조건을 설계했으므로 두 단계 모두 과거 자료 안에서의 시간순 검증이다.',
        '## 익절·손절을 고정한 7개 결과',
        table(['조건', '1~6월 수익 · 청산', '7~8월 수익 · 청산', '7/1~11/17 수익 · 청산', '7월 이후 최대낙폭', '9/1~11/17 수익 · 청산'], [
            [NAMES[f], *(f'{pct(rows.at[f, "net10"])} · {rows.at[f, "closed"]}건' for rows in (h1, summer, h2)),
             pct(h2.at[f, 'mdd10']), f'{pct(sept.at[f, "net10"])} · {sept.at[f, "closed"]}건'] for f in ids
        ]),
        '수익은 종가로 평가한 계좌의 누적 복리 수익률이다. 표의 거래 수와 승률은 청산 건만 세며, 최종 보유분은 마지막 종가로 평가한다. '
        '최대낙폭은 일별 종가로 평가한 자산이 고점에서 저점까지 하락한 비율이다. 현금 수익률은 0%로 두었다.',
        '## 7월 이후 비용과 체결 가정',
        table(['조건', '비용 0', '편도 5bp', '편도 10bp', '편도 20bp', '상승 갭 익절 +2% 상한¹', '04~20시 청산¹'], [
            [NAMES[f], *(pct(h2.at[f, col]) for col in ('net0', 'net5', 'net10', 'net20')),
             pct(capped.at[f, 'net10']), pct(extended.at[f, 'net10'])] for f in ids
        ]),
        '¹ 두 체결 감도에는 편도 10bp를 적용했다. 10bp는 매수·매도 각 0.10%의 가상 비용이다. 사용자 수수료는 0원이며 실제 호가 비용은 측정하지 않았다. '
        '기본 계산에서는 정규장 첫 관측 시가가 경계를 넘으면 상승·하락 모두 그 시가에 청산한다. 익절 상한 열은 상승 갭을 +2%에서 제한하고 '
        '하락 갭은 시가대로 반영한다. 04~20시 열은 미국 동부시간 장전·장후에도 조건 주문이 체결된다고 가정한다.',
        table(['조건', '신호일', '진입', '최종 보유', '청산 승률', '최악 거래', '최대 이익 거래 제외 후 누적'], [
            [NAMES[f], h2.at[f, 'signals'], h2.at[f, 'entries'], h2.at[f, 'open_positions'],
             f'{h2.at[f, "win_pct"]:.1f}%', pct(h2.at[f, 'worst10']), pct(h2.at[f, 'without_best10'])] for f in ids
        ]),
        '## 7월 이후 월별 수익',
    ]
    months = sorted(m for m in monthly.index.unique('month') if '2026-07' <= m <= '2026-11')
    assert len(months) == 5
    sections.extend([
        table(['조건', *[m[5:] + '월' for m in months]], [
            [NAMES[f], *(f'{pct(monthly.at[(f, m), "net10"])} · {monthly.at[(f, m), "closed"]}건' for m in months)] for f in ids
        ]),
        '7월 1일 시작 계좌를 매월 이어서 계산한 편도 10bp 수익과 그 달의 청산 수다. 11월은 17일까지다.',
        '## 종가 주문을 위해 신호를 5분 일찍 확정하면',
        table(['장중 조건', '15:50 판단 · 수익 · 청산', '15:45 판단 · 수익 · 청산'], [
            [NAMES[f], f'{pct(h2.at[f, "net10"])} · {h2.at[f, "closed"]}건', f'{pct(early.at[f, "net10"])} · {early.at[f, "closed"]}건'] for f in early.index
        ]),
        '미국 동부시간 기준으로 신호와 현금 여부를 모두 5분 먼저 확정해 주문 제출 시간을 남겨 보았다. 두 열은 모두 7월 현금 출발, 정규장 청산, 편도 10bp다. '
        '15:45 결과는 후보 선정에 사용하지 않았다. [추가 체결 가정 결과](results/earlier_cutoff_metrics.csv)',
        '## 같은 횟수의 신호를 다른 날에 옮겼을 때',
    ])
    rotations = audit['rotations']
    sections.extend([
        table(['조건', '원래 날짜 수익', '다른 날짜 중앙값', '원래 수익 이상인 이동 비율'], [
            [NAMES[f], pct(rotations[f]['actual_return_pct']), pct(rotations[f]['nonzero_shift_median_pct']),
             f'{100 * rotations[f]["fraction_shifts_at_least_actual"]:.1f}%'] for f in ids if f != 'baseline'
        ]),
        f'7월 이후 신호일 배열을 한 거래일씩 순환 이동해 {rotations[top]["nonzero_shifts"]}개 다른 날짜 배열과 비교했다. '
        '신호 빈도는 같고 보유 중 건너뛰는 날이 달라 실제 거래 수는 달라질 수 있다. 이동 결과끼리 같은 가격 경로를 공유하므로 비율은 통계적 유의확률이 아니다.',
        '## 신호 시점과 실행 조건',
        table(['조건', '종가 매수 전에 확인하는 값'], [[NAMES[f], spec['filters'][f]] for f in ids]),
        'SOXL의 당일 조건은 종가 주문을 결정하는 시점(통상 15:50 ET)에 마지막으로 완성된 분봉까지만 쓴다. '
        '조건이 맞고 그때 현금이면 종가에 하나의 포지션을 열며, 청산되기 전에는 추가 매수하지 않는다. '
        '신호의 1분 RSI(14)는 30 이하에서 상승한 날의 종가 진입을 뜻한다. 이번 추가 탐색 범위는 무필터 1개와 단일 조건 6개다. '
        'BB-1을 제외한 각 조건은 사전에 고른 문턱 한 가지로 계산했고, 필터끼리 결합하거나 문턱을 조정하지 않았다.',
        'BB-1은 전일 SOXX 종가가 60거래일 최고 종가보다 20% 이상 낮고, 한국 투자자 미국 주식 매수 상위 50개 중 '
        'SOXL이 차지한 최근 5세션 금액 비중이 250세션 분포의 80백분위 이상일 때 켜진다. 수급표는 다음 예탁결제원 영업일 09:00 KST에 알 수 있다고 가정했다. '
        'BB-1을 발굴한 19,649개 조합 탐색에는 같은 2026년 자료가 포함됐다. '
        f'BB-1의 7월 이후 수급·신호 미확인일 {sum(d >= "2026-07-01" for d in inputs["unknown_bb1_dates"])}일은 신규 진입을 건너뛰었다. '
        f'마지막 수급 신호 관찰일은 {inputs["bb1"]["latest_source_observation"]}이다. 특히 11/13·16·17은 가격 조건을 충족했으므로 '
        '당시 수급을 확인할 수 있었다면 BB-1의 진입 여부와 수익이 달라질 수 있다. [신호 원자료와 결측일](results/input_audit.json)',
        '정규장 매수·청산 모형은 Alpaca SIP 분할조정 일봉 종가와 분봉 체결가를 사용한다. 같은 분봉 안에서 두 선에 모두 닿으면 손절로 처리한다. '
        '데이터의 종가와 실제 경매 체결가 사이의 차이, 지연·정정된 분봉, 호가 대기는 모형에 포함되지 않는다. [Alpaca 역사 시세 API](https://docs.alpaca.markets/us/reference/stockbars) · [분봉 집계 방식](https://docs.alpaca.markets/us/docs/market-data-faq)',
        '미국 정규장 15:50 ET는 한국 다음 날 04:50(서머타임) 또는 05:50이다. 사용자 직접 주문 시간(08~24시 KST)과 겹치지 않는다. '
        '장중 필터는 새벽에 신호를 처리해 종가 주문을 내는 기능이 필요하다. 종가 체결 직후 손절·익절 연동과 시간외 스탑의 증권사 지원도 확인되지 않았다. '
        '전일 SOXX 20일선의 가격 조건은 한국 주문 가능 시간 전에 이미 확정된다. [사용자 매매 조건과 BB-1 산식](../../../../../docs/desk_system.md)',
        f'독립적으로 가격을 시간순 재생해 {len(audit["models"])}개 원장의 거래와 자산 평가를 대조했다. 2025년 자료는 지표 준비에만 썼다. '
        '[탐색·선정 기준](protocol.json) · [전체 성과](results/metrics.csv) · [거래별 원장](results/trades.csv) · [검증 기록](results/validation.json)',
        '이 연구 폴더에서 `py features.py`, `py backtest.py`, `py backtest.py --buffer`, `py backtest.py --components`, `py audit_filters.py`, `py report.py` 순으로 실행해 재생성한다.',
    ])
    return '\n\n'.join(sections) + '\n'


if __name__ == '__main__':
    (HERE / 'REPORT.md').write_text(render(), encoding='utf-8')
