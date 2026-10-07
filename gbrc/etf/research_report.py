"""Korean research report and offline interactive equity chart from saved results.

No strategy is selected or recalculated here. ``write_report(Path(...))`` reads
the experiment's saved evaluation, subgroup, stress, protocol and data audits.
"""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
import re
from typing import Any

from .storage import atomic_write_text


FAMILIES = {
    'conviction': '펀드 보유 비중', 'net_flow': '수량 기준 순매수 강도',
    'buy_only': '실매수 종목만', 'joint_buy': '두 펀드 동반 실매수',
    'joint_hold': '두 펀드 공통 보유', 'new_entry': '신규 편입',
    'repeated_buy': '반복 실매수', 'avoid_sell': '실매도 회피',
    'divergence': '두 펀드 매매 방향 차이', 'buy_momentum': '실매수와 가격 상승',
    'buy_dip': '실매수와 상대적 가격 부진', 'holding_momentum': '보유 비중과 가격 상승',
    'sold_reversal': '실매도 후 반등 가설',
}
FREQUENCIES = {'daily': '매일', 'weekly': '주간', 'monthly': '월간'}
WEIGHTS = {'equal': '동일 비중', 'conviction': '펀드 보유 비중 비례',
           'inverse_volatility': '최근 변동성 역수 비례'}
GATES = {'none': '가격 필터 없음', 'positive_momentum': '최근 20관측일 상승 필터',
         'above_ma20': '20관측일 평균 위 필터'}
RISKS = {'always': '상시 주식 노출', 'trend_half': '하락 상태에서 목표 노출 50%',
         'trend_cash': '하락 상태에서 현금 전환', 'manager_cash': '두 펀드 평균 현금 비중 반영'}
BENCHMARK_NAMES = {'time': 'TIME', 'koact': 'KoAct', 'qqq_krw': 'QQQ·원화'}
SEGMENTS = {'full': '전체 표본', 'train': '선발 1~5월', 'validation': '선택 6~7월',
            'holdout': '사후 홀드아웃 8~9월'}
SOURCE_OVERFIT = 'https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf'
FAMILY_DEFINITIONS = {
    'conviction': '두 펀드의 혼합 NAV 보유 비중 순위',
    'net_flow': '수량 조정 후 혼합 매매 %p 순위; 순매도가 상대적으로 작아도 선택 가능',
    'buy_only': '실매수·신규이며 혼합 순매수 %p가 양수인 종목',
    'joint_buy': '두 펀드 모두 실매수·신규인 종목; 매매 강도에 보유 비중 보조',
    'joint_hold': '두 펀드가 함께 보유하는 종목의 혼합 비중 순위',
    'new_entry': '신호에 참여하는 펀드 중 하나 이상에서 신규인 종목',
    'repeated_buy': '기간 중 일별 실매수·신규가 반복되고 누적 순매수도 양수',
    'avoid_sell': '신호에 참여하는 펀드에서 실매도·편출 판정이 없는 보유 종목',
    'divergence': '한 펀드는 매수하고 다른 펀드는 매도하는 종목의 매매 강도',
    'buy_momentum': '실매수 종목에서 매매 강도와 최근 20관측일 상승 순위를 반반 결합',
    'buy_dip': '실매수 종목에서 매매 강도와 상대적으로 낮은 최근 상승률 순위를 반반 결합',
    'holding_momentum': '현재 보유 비중 순위와 최근 20관측일 상승 순위를 반반 결합',
    'sold_reversal': '아직 보유 중인 실매도 종목을 비중과 상승 순위로 평가',
}


def pct(value: Any, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return 'N/A'
    return f'{float(value):+.{digits}f}%'


def num(value: Any, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return 'N/A'
    return f'{float(value):,.{digits}f}'


def ratio(value: Any) -> str:
    return 'N/A' if value is None else f'{float(value):.2f}배'


def _inline(text: str) -> str:
    text = escape(str(text))
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)


class Document:
    def __init__(self):
        self.md: list[str] = []
        self.html: list[str] = []

    def heading(self, title: str, level: int = 2) -> None:
        self.md.append('#' * level + ' ' + title + '\n')
        self.html.append(f'<h{level}>{escape(title)}</h{level}>')

    def p(self, text: str, css: str = '') -> None:
        self.md.append(text + '\n')
        self.html.append(f'<p class="{css}">{_inline(text)}</p>')

    def bullets(self, values: list[str]) -> None:
        self.md.append('\n'.join('- ' + value for value in values) + '\n')
        self.html.append('<ul>' + ''.join('<li>' + _inline(value) + '</li>' for value in values) + '</ul>')

    def table(self, columns: list[str], rows: list[list[Any]], caption: str = '') -> None:
        safe = [[str(cell) for cell in row] for row in rows]
        self.md.append('| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join('---' for _ in columns)
                       + ' |\n' + '\n'.join('| ' + ' | '.join(row) + ' |' for row in safe) + '\n')
        self.html.append('<div class="table-scroll"><table>' +
                         (f'<caption>{escape(caption)}</caption>' if caption else '') +
                         '<thead><tr>' + ''.join(f'<th scope="col">{escape(c)}</th>' for c in columns) +
                         '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(
                             f'<{("th scope=" + chr(34) + "row" + chr(34)) if i == 0 else "td"}>'
                             + _inline(value) + ('</th>' if i == 0 else '</td>')
                             for i, value in enumerate(row)) + '</tr>' for row in safe) +
                         '</tbody></table></div>')


def config_text(config: dict) -> str:
    mix = float(config['time_mix'])
    return (f"{FAMILIES.get(config['family'], config['family'])} · D-{config['lookback']} · "
            f"TIME {mix:.0%}/KoAct {1-mix:.0%} · {FREQUENCIES[config['rebalance']]} · "
            f"상한 {config['max_stocks']}종목 · {WEIGHTS[config['weighting']]} · "
            f"{RISKS[config['risk_rule']]} · {GATES[config['gate']]}")


def _profile_row(label: str, profile: dict) -> list:
    segments = profile['segments']
    return [label, *[pct(segments[key]['return_pct']) for key in ('train', 'validation', 'holdout', 'full')],
            pct(segments['full']['max_drawdown_pct']),
            ratio(segments['holdout']['benchmarks']['time']['return_ratio']),
            ratio(segments['holdout']['benchmarks']['koact']['return_ratio'])]


def _load(out: Path, name: str) -> dict:
    return json.loads((out / name).read_text(encoding='utf-8'))


def _main_report(evaluation: dict, subgroups: dict, stress: dict, protocol: dict,
                 metadata: dict, out: Path) -> Document:
    doc = Document()
    selected = evaluation['selected']
    h = selected['segments']['holdout']
    cfg = selected['config']
    sf = selected['segments']['full']
    fresh = stress['candidates']['primary_selected']['fresh_holdout']['return_pct']
    fresh_time = stress['fresh_holdout_benchmarks']['time']['return_pct']
    fresh_koact = stress['fresh_holdout_benchmarks']['koact']['return_pct']
    robust = _load(out, 'robust_research.json') if (out / 'robust_research.json').exists() else None
    doc.heading('TIME × KoAct 매매기록 조합 연구', 1)
    doc.p(f"관측 범위 {metadata['sample_start']}~{metadata['sample_end']} · "
          f"검증 스냅샷 {metadata['snapshot_count']:,}개 · 역사적 직접 보유 {metadata['security_count']}종목 · "
          f"시도 {evaluation['candidate_count']:,}개", 'eyebrow')
    doc.p(f"**2~3배의 과거 수익률을 보인 규칙은 찾았지만, 실행 조건을 바꿔도 유지되는 2배 초과수익은 입증하지 못했다.** "
          f"1~5월 선발과 6~7월 선택을 거친 주 후보는 전체 표본 {pct(sf['return_pct'])}, "
          f"사후 홀드아웃 {pct(h['return_pct'])}였다. 홀드아웃에서 TIME 수익률의 "
          f"{ratio(h['benchmarks']['time']['return_ratio'])}, KoAct의 "
          f"{ratio(h['benchmarks']['koact']['return_ratio'])}이며, 선택된 규칙의 종목 상한은 "
          f"{cfg['max_stocks']}개다.", 'lead')
    doc.p(f"**8월에 현금으로 새로 출발하면 주 후보는 {pct(fresh)}, TIME은 {pct(fresh_time)}, "
          f"KoAct는 {pct(fresh_koact)}였다. 이 조건에서는 2배 목표를 충족하지 못했다.** "
          '5종목 상한의 혼합 후보도 기본 성과는 높았지만 신호를 한 관측일 더 늦추자 전체 수익률이 크게 줄었다. '
          '현재 결과는 실행 시점과 소수 종목에 민감한 연구 후보로 보는 것이 적절하다.', 'callout')
    if robust:
        robust_all = robust['scopes']['all']['selected']['scenarios']
        doc.p(f"비용과 지연을 함께 견디는 규칙을 찾기 위해 같은 설정을 추가로 전수 검사했다. "
              f"두 실행 조건 중 나쁜 쪽의 과거 점수로 선발해도, 전체 집단 선택 후보의 8~9월 수익률은 "
              f"기본 {pct(robust_all['base_15bps']['segments']['holdout']['return_pct'])}, "
              f"지연·비용 강화 {pct(robust_all['delayed_30bps']['segments']['holdout']['return_pct'])}였다. "
              '**불리한 실행 조건까지 고려한 재선발에서도 목표를 달성하는 선택 절차는 확인하지 못했다.**')
    doc.p('전체기간 최고 수익 규칙은 마지막 날짜까지의 결과를 보고 골랐다. 주 후보는 7월까지의 자료로 '
          '골랐지만, 연구 자체는 9월에 수집된 과거 자료로 수행했다. 8~9월도 실제 미래 실험이 아닌 '
          '**사후 시간 분리 검증**이며, 그 결과를 보고 다시 고른 하위집단은 추가 진단이다.', 'callout')
    doc.p('여기서 2배는 누적 수익률의 배수다. ETF가 +20%이면 전략의 +40%가 2배다. '
          '원금 포함 최종 자산을 두 배로 만든다는 의미가 아니다. 비교 ETF 수익률이 0 이하인 구간은 '
          '배수 대신 수익률 차이를 본다.')

    doc.heading('같은 조건에서 비교한 성과')
    rows = []
    for label, profile in [('주 후보·선발/선택', selected),
                           ('전체기간 수익 최고·사후 선택', evaluation['hindsight_best']),
                           ('1~5월만 보고 선택', evaluation['train_only'])]:
        rows.append(_profile_row(label, profile))
    for name in evaluation['benchmark_order']:
        rows.append([BENCHMARK_NAMES.get(name, name),
                     *[pct(evaluation['segments'][seg]['benchmarks'][name]['return_pct'])
                       for seg in ('train', 'validation', 'holdout', 'full')],
                     pct(evaluation['segments']['full']['benchmarks'][name]['max_drawdown_pct']), '—', '—'])
    doc.table(['구분', '1~5월', '6~7월', '8~9월', '전체', '전체 최대낙폭', '8~9월/TIME', '8~9월/KoAct'], rows,
              '누적 원화 가격 수익률. 편도 가정 비용 차감, 세금·배당 제외.')
    doc.p(f"최대낙폭은 구간 내 이전 최고자산 대비 최저 하락률이다. 기본 비용은 거래금액의 편도 "
          f"{protocol['fees']['one_way_fraction_of_traded_value']*100:.2f}%로, 매도·매수를 모두 하면 양쪽에 각각 든다. "
          f"벤치마크 최초 매수일은 {evaluation['benchmark_entry_date']} 종가이고 전략은 첫 유효 신호가 나올 때까지 "
          '현금으로 기다린다. 표의 구간 수익률은 1월부터 이어진 포트폴리오를 구간별로 다시 1로 환산한 값이며, '
          '8월 현금 신규 출발과 구분한다.')
    doc.md.append('[대화형 자산곡선 열기](report.html) — 기간 선택 및 범례로 곡선을 켜고 끌 수 있다.\n')
    doc.html.append('<!--CURVE_CHART-->')

    doc.heading('어떤 가설을 얼마나 조합했나')
    doc.table(['조합 축', '시험한 값'], [
        ['수량·보유 신호 13종', '보유 비중, 순매수, 매수만, 동반매수, 공통보유, 신규, 반복매수, '
         '매도회피, 방향차이, 매수+상승, 매수+부진, 보유+상승, 매도 후 반등'],
        ['거래 관찰 기간', 'KRX 1·3·5·10·20·40 거래 세션'],
        ['TIME 신호 비중', '0·25·50·75·100%; 나머지는 KoAct'],
        ['가격 필터', '없음 / 최근 20관측일 상승 / 20관측일 평균 위'],
        ['종목 상한', '1·2·3·5·7·10'],
        ['교체 주기', '일간 / 달력 주간 / 달력 월간'],
        ['배분', '동일 비중 / 펀드 보유 비중 / 변동성 역수'],
        ['방어', '상시 / 하락 상태 50% / 하락 상태 현금 / 펀드 현금 추종'],
    ])
    doc.p(f"총 {evaluation['candidate_count']:,}개 설정을 실행했고, 일간 수익률을 1e-7로 반올림해 구분한 "
          f"서로 다른 기본 실행 경로는 {evaluation['unique_return_paths_rounded_1e_7']:,}개다. 동일 신호·보유로 귀결되는 "
          '설정이 있어 시도 횟수와 독립적인 전략의 수는 다르다. 이 숫자는 독립 표본 수나 통계적 유의성의 증거가 아니다.')
    doc.p(f"선발 점수는 누적 로그수익에서 최대낙폭 절대값의 절반을 차감했다. 1~5월 최대낙폭이 -50% 이하인 "
          f"설정을 제외한 {evaluation['eligible_train_count']:,}개 중 상위 1%인 "
          f"{evaluation['shortlist_count']:,}개를 고정하고, 6~7월 점수가 가장 높은 하나를 선택했다. "
          '8~9월 성과는 이 선택 점수에 넣지 않았다. 실험 규약과 설정 목록은 아래 원자료에 저장했다.')

    doc.heading('주 후보의 규칙')
    doc.p(f"후보 #{selected['index']}: **{config_text(cfg)}**.")
    doc.p('반복 실매수는 일별 수량 조정 후 매수·신규 판정이 관찰 기간에 두 번 이상이고 누적 순매수 신호도 '
          '양수인 종목을 고른다. 펀드의 설정·환매에 따른 공통 수량 배율과 주식분할은 제거한다. '
          '주 후보의 TIME 100%는 신호 조합에서 TIME만 사용했다는 뜻이다. 두 펀드를 섞은 모든 설정이 '
          '단일 펀드 신호보다 낫다는 결과는 나오지 않았다.')
    doc.p('거래일보다 앞선 날짜의 스냅샷과 직전까지의 가격만 의사결정에 쓰고, 종가로 보유 손익을 반영한 뒤 '
          '교체한다. 휴장 시장의 보유는 거래할 수 있을 때까지 유지한다. 종목 상한은 보유 수의 최대값이며, '
          '신호 부족이나 현금 전환으로 더 적게 보유할 수 있다. 최대 10종목 규칙은 매일 10종목 분산을 보장하지 않는다.')

    if 'TIME_top10_weekly' in stress['candidates']:
        doc.heading('단순히 펀드 상위 10종목만 좁혀 담는 경우')
        doc.p('복잡한 매매 신호가 없어도 단순 집중만으로 성과가 바뀔 수 있으므로, 각 펀드의 NAV 비중 '
              '상위 10종목을 주간 교체하고 비중 비례로 보유하는 규칙을 함께 비교했다. 가격 필터와 장세 방어는 '
              '사용하지 않고, 다른 후보와 같은 공시 지연·비용·휴장 처리·현금 제약을 적용했다.')
        doc.table(['단순 비교 규칙', '전체 수익', '전체 최대낙폭', '+1일 지연 전체',
                   '기존 보유 8~9월', '8월 현금 출발'], [
            [label, pct(row['full']['return_pct']), pct(row['full']['max_drawdown_pct']),
             pct(row['extra_lag']['full']['return_pct']), pct(row['holdout']['return_pct']),
             pct(row['fresh_holdout']['return_pct'])]
            for key, label in [('TIME_top10_weekly', 'TIME 상위 10·주간'),
                               ('KoAct_top10_weekly', 'KoAct 상위 10·주간')]
            for row in [stress['candidates'][key]]])
        simple = stress['candidates']['KoAct_top10_weekly']['fresh_holdout']['return_pct']
        doc.p(f"KoAct 단순 상위 10의 8월 현금 출발 수익률 {pct(simple)}는 주 후보의 {pct(fresh)}보다 높았다. "
              '주 후보가 전체기간에 크게 앞섰다는 사실만으로 반복 매수 신호가 단순 집중보다 꾸준히 우월하다고 '
              '확정할 수 없다. 단순 집중 규칙도 원래 ETF보다 최대낙폭이 커질 수 있다.')

    doc.heading('3·5·7·10종목 상한을 따로 본 추가 진단')
    subgroup_rows = []
    for size in (3, 5, 7, 10):
        group = subgroups['subgroups'][f'max_stocks_{size}']
        profile = group['selected']
        segments = profile['segments']
        subgroup_rows.append([f'상한 {size}종목', f"#{profile['global_index']}",
                              FAMILIES[profile['config']['family']],
                              pct(segments['full']['return_pct']),
                              pct(segments['full']['max_drawdown_pct']),
                              pct(segments['holdout']['return_pct']),
                              ratio(segments['holdout']['benchmarks']['time']['return_ratio']),
                              ratio(segments['holdout']['benchmarks']['koact']['return_ratio'])])
    doc.table(['집단', '선택 번호', '신호', '전체 수익', '최대낙폭', '8~9월', '8~9월/TIME', '8~9월/KoAct'], subgroup_rows)
    doc.p('각 상한 집단 안에서 같은 1~5월 선발 → 6~7월 선택 규칙을 다시 적용했다. 이 비교는 주 결과 확인 후 '
          '추가한 진단으로, 유리한 집단을 골라 새로운 독립 검증처럼 해석하면 안 된다. 3·5·7·10은 각각 '
          '종목 수의 상한이며 최소 보유 수 제약은 없다.')
    for size in (3, 5, 7, 10):
        profile = subgroups['subgroups'][f'max_stocks_{size}']['selected']
        doc.p(f"상한 {size}: {config_text(profile['config'])}.")
    extras = []
    for key, label in [('both_funds_mixed_ge3', '두 펀드 혼합·상한 3개 이상'),
                       ('daily_ge3', '일간·상한 3개 이상'), ('weekly_ge3', '주간·상한 3개 이상'),
                       ('monthly_ge3', '월간·상한 3개 이상')]:
        profile = subgroups['subgroups'][key]['selected']
        extras.append([label, f"#{profile['global_index']}",
                       pct(profile['segments']['full']['return_pct']),
                       pct(profile['segments']['holdout']['return_pct'])])
    doc.table(['추가 집단', '선택 번호', '전체 수익률', '8~9월 수익률'], extras)

    _stress_report(doc, stress, evaluation)
    if robust:
        _robust_report(doc, robust)
    _walk_forward_report(doc, evaluation)

    doc.heading('부록 · 13개 가설별 추가 진단')
    doc.p('가설별로 같은 선발·선택 절차를 적용한 결과다. 주 결과를 확인한 뒤 비교한 하위집단 진단이므로 '
          '여기서 성과가 가장 높은 가설을 새 검증 우승자로 바꾸지는 않는다. 관찰 기간·혼합 비중·가격 필터·'
          '종목 상한·교체 주기·배분·방어 조건은 각 가설 안에서도 달라진다.')
    family_rows = []
    for family, label in FAMILIES.items():
        group = subgroups['families'][family]
        profile = group['selected']
        family_rows.append([label, f"{group['candidate_count']:,}", f"#{profile['global_index']}",
                            pct(profile['segments']['full']['return_pct']),
                            pct(profile['segments']['holdout']['return_pct']),
                            profile['config']['max_stocks'],
                            FREQUENCIES[profile['config']['rebalance']]])
    doc.table(['가설', '설정 수', '선택 번호', '전체 수익', '8~9월 수익', '종목 상한', '주기'], family_rows)
    doc.table(['가설', '구체적 신호 정의'], [[label, FAMILY_DEFINITIONS[family]]
              for family, label in FAMILIES.items()])

    doc.heading('이번 결과로 판단할 수 있는 범위')
    boot = evaluation['holdout_bootstrap']
    ci = boot.get('relative_wealth_return_ci95_pct')
    if ci:
        doc.p(f"홀드아웃은 {boot['observations']}관측일이다. 선택 후보와 TIME의 상대자산 수익에 대해 "
              f"5일 블록 재표집 {boot['samples']:,}회를 적용한 조건부 95% 구간은 "
              f"{pct(ci[0])}~{pct(ci[1])}로 0을 포함한다. 이것은 이미 고른 규칙에 대한 짧은 표본의 "
              '불확실성 진단이며, 수십만 번의 탐색 편향을 교정한 성공 확률이 아니다.')
    doc.p('많은 설정을 같은 자료에 반복 적용하면 우연히 뛰어난 백테스트가 나타날 가능성이 커진다. '
          '시간을 나눴다는 사실만으로 이 문제를 모두 해결하지는 못한다. '
          f'[Bailey 외, The Probability of Backtest Overfitting]({SOURCE_OVERFIT})')
    doc.bullets([
        '자료 범위는 2026-01-02~2026-09-23뿐이다. 그 이전 워밍업 가격, 이후 수익률, 아직 거래되지 않은 종목의 '
        '미래 가격을 넣지 않았다. 모든 역사적 보유 종목을 포함해 현재 생존 종목만 고르는 편향을 줄였다.',
        '스냅샷 대부분은 9월 27일 과거 일자로 내려받았다. 날짜별 실제 최초 공개 시각을 검증한 실시간 기록이 없어, '
        '엄격한 다음 날짜 체결과 추가 지연 검사를 해도 과거 정보 접근 가능성이 완전히 증명되지는 않는다.',
        '미국·한국·일본·홍콩·중국 종가를 같은 날짜 축으로 표시했다. 각 시장 종가와 당일 외환 종가는 동시 호가가 '
        '아니므로 실제 체결에는 시간대·호가·환전 차이가 생길 수 있다.',
        '가격은 분할을 반영한 Yahoo Close다. 배당·분배금·세금은 제외했다. 현금 이자는 0%, 편도 수수료·슬리피지·'
        '환전 여유는 거래금액의 0.15%를 기본 가정으로 두었다. 매수 자금은 현금뿐이며 차입·공매도·레버리지는 없다.',
        '현금, 지수선물과 두 ETF 재간접 보유는 직접 종목 후보에서 제외했고 원래 NAV 비중은 감사 자료에 남겼다. '
        'GOOG/GOOGL은 GOOG 가격으로 합쳤고, SKHYV/SKHY 및 SATS/ECHO의 티커 연속성은 별도로 기록했다.',
        '이 보고서는 규칙 후보와 그 한계를 평가한다. 현재 시점 종목 추천·목표 비중·주문 수량을 생성하지 않았다. '
        '다음 단계는 규칙을 동결한 새 관측 구간의 모의운용과 추가 기간 검증이다.',
    ])

    doc.heading('재현 자료')
    links = [('report.html', '기간을 바꿔 보는 대화형 보고서'), ('report.md', 'Markdown 보고서'),
             ('protocol.json', '실험 규약'), ('configurations.json', '전체 252,720개 설정'),
             ('all_results.csv', '전체 설정별 결과와 설정 CSV'), ('evaluation.json', '주 평가 결과'),
             ('subgroup_evaluation.json', '상한·가설별 추가 진단'), ('stress.json', '비용·지연·신규 출발 검사'),
             ('robust_research.json', '두 실행 조건을 함께 고려한 추가 전수 탐색'),
             ('stress.csv', '민감도 결과 CSV'), ('equity_curves.json', '자산곡선 원자료'),
             ('metadata.json', '입력 해시·환율·제외 종목·가격 감사'), ('raw_prices.json', '다운로드한 가격'),
             ('bundle.npz', '정규화 연구 입력 배열')]
    links.extend((path.name, path.stem.replace('_historical_ledger', '') + ' 과거 보유·교체 기록 CSV')
                 for path in sorted(out.glob('*_historical_ledger.csv')))
    doc.bullets([f'[{label}]({name})' for name, label in links if (out / name).exists() or name in ('report.html', 'report.md')])
    doc.p('구성종목 출처: [TIME 운용사 공시](https://timeetf.co.kr/m11_view.php?idx=6), '
          '[KoAct 운용사 공시](https://www.samsungactive.co.kr/etf/view.do?id=2ETFQ1). '
          '실험에는 작업 폴더의 검증 스냅샷과 위 기간 안의 Yahoo 가격·관측 환율만 사용했다.')
    doc.heading('✅ 자기검증 (Self-Check)', 3)
    doc.bullets([
        '트리거 매칭: [Path-F ✅]',
        'Gate B 통과: [ETF 정본 Read + 과거 데이터 브리핑·실매매 분석·전략 실험 Y]',
        '가정으로 채운 숫자: [있음 — 편도 비용 0.15%, 민감도 0/15/30/50bp, 현금 이자 0%; '
        '원천 가격·보유 수량·환율을 가정으로 대체하지 않음]',
        '돌발 호재 확인 (Path-O/V 한정): [불필요 ✅]',
        '규칙 위반: [없음 ✅]',
    ])
    return doc


def _robust_report(doc: Document, robust: dict) -> None:
    doc.heading('추가 전수 검사 · 비용과 지연을 함께 견디는가')
    doc.p(f"처음 결과를 확인한 뒤 같은 {robust['candidate_count']:,}개 설정에 편도 30bp와 "
          f"추가 1관측일 지연을 적용했다. 기본 15bp 실행과 합하면 두 시나리오의 격자 실행은 "
          f"{2*robust['candidate_count']:,}회지만, 서로 다른 규칙 수는 여전히 {robust['candidate_count']:,}개다. "
          '두 조건 중 나쁜 쪽의 점수로 1~5월 상위 1%를 뽑고, 그 안에서 6~7월 점수가 가장 높은 후보를 선택했다. '
          '비교 ETF는 원래의 15bp 매수·보유 기준을 유지해 불리한 시나리오에서 벤치마크를 낮추지 않았다.')
    doc.p('이 재탐색 설계는 첫 홀드아웃 결과를 본 후 추가했다. 점수에 8~9월 수익률을 직접 넣지는 않았지만 '
          '이미 본 기간을 다시 활용했으므로 독립 검증이 될 수 없다.', 'callout')
    labels = {'all': '전체 설정', 'mixed_max_stocks_3_to_10': '두 펀드 혼합·상한 3~10',
              'mixed_max_stocks_5_to_10': '두 펀드 혼합·상한 5~10'}
    rows = []
    for scope, group in robust['scopes'].items():
        chosen = group['selected']
        for scenario, sc in chosen['scenarios'].items():
            rows.append([labels[scope], f"#{chosen['global_candidate_index']}",
                         '기본·15bp' if scenario == 'base_15bps' else '+1일·30bp',
                         pct(sc['segments']['full']['return_pct']),
                         pct(sc['segments']['holdout']['return_pct']),
                         pct(sc['fresh_cash_holdout']['return_pct']),
                         pct(sc['fresh_cash_holdout']['max_drawdown_pct'])])
    doc.table(['선발 범위', '선택 번호', '실행 조건', '전체 수익', '기존 보유 8~9월', '8월 현금 출발',
               '현금 출발 최대낙폭'], rows)
    counts = robust['scopes']['all']['full_period_joint_target_counts']
    doc.p(f"전체기간 결과를 사후에 보면 두 실행 조건 모두에서 두 ETF 수익률의 2배 이상인 설정은 "
          f"{counts['at_least_2x_stronger_etf_in_both_scenarios']:,}개, 3배 이상은 "
          f"{counts['at_least_3x_stronger_etf_in_both_scenarios']:,}개 존재했다. "
          '그러나 과거 앞부분으로 선택한 위 후보들의 이후 성과는 그 목표에 미치지 못했다. '
          '뒤까지 보고 성공한 조합을 찾는 일과, 앞부분만 보고 성공할 조합을 골라내는 일의 차이가 드러났다.')
    for scope, group in robust['scopes'].items():
        chosen = group['selected']
        doc.p(f"{labels[scope]}의 선택 규칙: {config_text(chosen['config'])}.")
    doc.p('혼합은 TIME 신호 비중이 0%와 100% 사이인 설정을 뜻한다. 항상 두 펀드에서 각각 종목을 '
          '보유한다는 뜻은 아니며, 상한 5~10 집단도 최소 5종목을 보장하지 않는다.')


def _walk_forward_report(doc: Document, evaluation: dict) -> None:
    wf = evaluation['walk_forward']
    doc.heading('매달 이전 자료만으로 다시 고르면')
    doc.p(f"6월부터 직전 월까지의 점수로 다시 선택하는 순차 검증의 누적 수익률은 "
          f"{pct(wf['with_extra_surcharge']['return_pct'])}, 최대낙폭은 "
          f"{pct(wf['with_extra_surcharge']['max_drawdown_pct'])}였다. 같은 기간 TIME "
          f"{pct(wf['benchmarks']['time']['return_pct'])}, KoAct "
          f"{pct(wf['benchmarks']['koact']['return_pct'])}와 비교한다.")
    doc.table(['평가 월', '직전까지의 선택 번호', '사용한 과거 종료일', '월간 수익률', 'TIME', 'KoAct'], [
        [row['month'], f"#{row['candidate_index']}", row['training_end'],
         pct(row['with_extra_surcharge']['return_pct']),
         pct(row['benchmarks']['time']['return_pct']), pct(row['benchmarks']['koact']['return_pct'])]
        for row in wf['picks']])
    doc.p('이 검사는 각 후보의 기존 보유 손익 경로를 월별로 연결하고 최초 15bp, 규칙 변경 시 30bp를 '
          '추가 차감했다. 실제 교체 계좌를 주문 단위로 다시 시뮬레이션한 결과는 아니므로 보조 진단으로 본다. '
          '한 번 찾은 전체기간 최고 규칙의 수익률을 매달 미리 알 수 있었다는 뜻도 아니다.')


def _stress_report(doc: Document, stress: dict, evaluation: dict) -> None:
    candidates = stress['candidates']
    labels = {'primary_selected': '주 후보', 'hindsight_best': '전체기간 최고·사후',
              'TIME_top10_weekly': 'TIME 상위 10·주간', 'KoAct_top10_weekly': 'KoAct 상위 10·주간',
              'diversified_ge3': '상한 3개 이상 전체', 'both_funds_mixed_ge3': '두 펀드 혼합·상한 3개 이상',
              'weekly_ge3': '주간·상한 3개 이상', 'daily_ge3': '일간·상한 3개 이상',
              'monthly_ge3': '월간·상한 3개 이상', 'robust_all': '재선발·전체',
              'robust_mixed_max_stocks_3_to_10': '재선발·혼합 상한 3~10',
              'robust_mixed_max_stocks_5_to_10': '재선발·혼합 상한 5~10'}
    labels.update({f'max_stocks_{size}': f'상한 {size}종목 집단' for size in (1, 2, 3, 5, 7, 10)})
    doc.heading('비용·신호 지연·신규 출발을 바꾸면')
    doc.p('다음 표의 기본과 추가 지연 열은 모두 편도 15bp를 적용했다. 이후 추가 전수 검사에서는 '
          '지연과 편도 30bp를 동시에 적용한 경우를 별도로 제시한다.')
    doc.table(['규칙', '기본 전체', '+1관측일 지연 전체', '기존 보유 8~9월', '8월 현금 출발',
               '실제 최대 종목 수', '평균 종목 수'], [
        [labels.get(label, label), pct(row['full']['return_pct']), pct(row['extra_lag']['full']['return_pct']),
         pct(row['holdout']['return_pct']), pct(row['fresh_holdout']['return_pct']),
         row['max_observed_stocks'], num(row['average_observed_stocks'])]
        for label, row in candidates.items()])
    fresh = stress['fresh_holdout_benchmarks']
    doc.p(f"8월 현금 출발의 같은 기준 비교는 TIME {pct(fresh['time']['return_pct'])}, "
          f"KoAct {pct(fresh['koact']['return_pct'])}, QQQ·원화 {pct(fresh['qqq_krw']['return_pct'])}다. "
          '8월 첫 관측일 종가에 진입하므로 그날 진입 전 주가 상승은 얻지 않는다. 신호 지연 검사는 '
          '펀드 공시와 가격 신호 모두에 한 번 더 관측일 지연을 주며, 각 시장의 한 거래일과 반드시 일치하지는 않는다.')
    c5 = candidates['max_stocks_5']
    hs = candidates['hindsight_best']
    doc.p(f"5종목 상한 후보의 전체 수익률은 {pct(c5['full']['return_pct'])}에서 추가 지연 시 "
          f"{pct(c5['extra_lag']['full']['return_pct'])}로 바뀌어 기본 TIME "
          f"{pct(evaluation['segments']['full']['benchmarks']['time']['return_pct'])}보다 낮아졌다. "
          f"전체기간 최고 후보도 {pct(hs['full']['return_pct'])}에서 "
          f"{pct(hs['extra_lag']['full']['return_pct'])}로 낮아졌다. "
          '좋은 결과만 남기지 않고 이 실패를 함께 기록한다.', 'callout')
    costs = []
    for key in ('primary_selected', 'max_stocks_5', 'weekly_ge3'):
        row = candidates[key]
        for cost in row['costs']:
            costs.append([labels[key], f"{cost['one_way_bps']}bp", pct(cost['full']['return_pct']),
                          pct(cost['full']['max_drawdown_pct']), pct(cost['holdout']['return_pct'])])
    doc.table(['규칙', '편도 비용', '전체 수익', '전체 최대낙폭', '기존 보유 8~9월'], costs)
    primary = candidates['primary_selected']
    doc.p(f"주 후보는 {primary['rebalance_trade_days']}일에 거래했고, 누적 매수·매도 금액을 당시 NAV로 "
          f"나눈 회전량 합계는 {num(primary['traded_nav_multiple'])}배다. 관측일 평균 현금 비중은 "
          f"{num(primary['average_cash_pct'])}%, 전액 현금 관측일은 {primary['cash_days']}일이다. "
          '기본 비용과 별도로 세금은 빼지 않았고, 마지막 날짜에는 평가만 하며 강제 청산 비용을 추가하지 않았다.')
    max10 = candidates['max_stocks_10']
    doc.p(f"상한 10종목 집단에서 선택된 규칙도 실제 최대 보유는 {max10['max_observed_stocks']}종목, "
          f"평균 {num(max10['average_observed_stocks'])}종목, 전액 현금 {max10['cash_days']}관측일이었다. "
          '따라서 이 규칙을 10종목 분산 전략이라고 부르지 않는다.')

    doc.heading('상승·하락·횡보 상태는 과거 정보로 분류')
    doc.p('각 의사결정 전 QQQ 달러 가격의 최근 20관측일 수익률이 +3%를 넘고 20일 평균 위면 상승, '
          '-3% 미만이면서 평균 아래면 하락, 그 밖은 횡보로 정의했다. 과거값이 부족한 처음 21관측일은 '
          '미정이다. 이 구분은 거래 전 신호이며 장세에 대한 정답 라벨이 아니다. 방어 노출 변경은 각 후보의 '
          '교체 일정에 적용되므로 하락으로 분류된 모든 날이 전액 현금인 것은 아니다.')
    regimes = primary['by_prior_regime']
    regime_labels = {'up': '상승', 'down': '하락', 'range': '횡보', 'unknown': '미정'}
    doc.table(['직전 정보 기준 상태', '관측일 수', '주 후보', 'TIME', 'KoAct'], [
        [regime_labels[key], row['observations'], pct(row['candidate']['return_pct']),
         pct(row['benchmarks']['time']['return_pct']), pct(row['benchmarks']['koact']['return_pct'])]
        for key in ('up', 'down', 'range', 'unknown') for row in [regimes[key]]])
    doc.p('상태별 수익률은 떨어져 있는 해당 상태의 일간 손익만 골라 복리 연결한 진단이다. '
          '상태 진입마다 별도 계좌를 운용한 수익률이 아니며 상태별 값을 단순 합산할 수 없다. '
          '특히 하락 상태 표본만으로 장기 하락장을 견디는 규칙이라고 결론 낼 수 없다.')

    doc.heading('수익을 만든 종목과 집중 의존도')
    doc.table(['주 후보 과거 기여 상위 6종목', '초기 원금 대비 손익 기여'], [
        [row['symbol'], f"{row['pnl_per_initial_nav_pp']:+.2f}%p"]
        for row in primary['contributions_pp'][:6]])
    top = primary['without_top_contributor']
    ht = hs['without_top_contributor']
    doc.p('기여도는 매일 직전 보유의 원화 손익을 초기 원금으로 나눠 누적한 값이다. '
          '현재 매수 후보 목록이 아니며, 수익률이 쌓인 뒤의 큰 원금 효과도 포함한다. '
          f"수수료 기여 {primary['fee_pnl_pp']:+.2f}%p는 별도 항목이다.")
    doc.p(f"주 후보의 사후 최대 기여 종목 {top['symbol']}을 기존 상위 10개 후보 목록에서 제거하면 "
          f"전체 {pct(top['full']['return_pct'])}였다. 전체기간 최고 후보는 최대 기여 "
          f"{ht['symbol']} 제거 후 {pct(ht['full']['return_pct'])}였다. "
          '이 검사는 11위 이하 후보를 새로 채우지 않으므로 공석과 현금 증가도 반영하는 의존성 진단이다. '
          '전체 종목군을 다시 순위 매긴 대체 전략의 성과나 새로 선발한 투자 규칙으로 해석하지 않는다.')


CSS = r'''
:root{color-scheme:light;--ink:#172a37;--muted:#5c6b73;--teal:#087c81;--line:#d9e3e8;--paper:#fff;--wash:#f3f7f9}
*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:16px/1.8 "Malgun Gothic","Segoe UI",sans-serif}
main{max-width:1240px;margin:0 auto;padding:52px 60px 72px;background:var(--paper);min-height:100vh;border-top:7px solid var(--teal)}
h1{font-size:36px;line-height:1.3;letter-spacing:-1.5px;margin:0 0 12px}h2{font-size:23px;line-height:1.4;letter-spacing:-.7px;margin:48px 0 16px;padding-top:22px;border-top:1px solid var(--line)}h3{font-size:18px;margin-top:38px}
p{margin:15px 0}.eyebrow{color:var(--muted);font-size:13px;letter-spacing:.2px}.lead{font-size:19px;line-height:1.9;border-left:4px solid var(--teal);padding:10px 0 10px 22px;margin:25px 0}.callout{background:#edf6f7;border:1px solid #d4e9eb;padding:18px 22px;border-radius:8px}a{color:#076d89;text-underline-offset:3px}strong{font-weight:750}li{margin:9px 0}ul{padding-left:23px}.table-scroll{overflow-x:auto;margin:20px 0;border:1px solid var(--line);border-radius:7px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.65}caption{text-align:left;padding:12px 16px;color:var(--muted);font-size:12px;background:var(--wash)}th,td{padding:12px 13px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}thead th{background:#183b4b;color:#fff;font-size:12px;font-weight:600}th:first-child,td:first-child{text-align:left}tbody th{font-weight:600}tbody tr:nth-child(even){background:#f6f9fa}tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}.table-scroll td:nth-child(2){white-space:normal}button{font:inherit;cursor:pointer;border:1px solid #bfcfd6;border-radius:5px;background:white;color:var(--ink);padding:7px 13px}button:focus-visible,input:focus-visible{outline:3px solid #df9526;outline-offset:3px}button[aria-pressed=true]{background:var(--teal);border-color:var(--teal);color:#fff}.chart{padding:25px 20px 17px;border:1px solid var(--line);border-radius:9px;background:linear-gradient(#fff,#f9fcfd);margin-top:30px}.chart h3{margin:0;font-size:18px}.chart-subtitle{font-size:12px;color:var(--muted);margin:5px 0 15px}.chart-buttons{display:flex;gap:8px;flex-wrap:wrap}.legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin:18px 0 4px}.legend label{font-size:12px;display:flex;align-items:center;gap:5px;cursor:pointer}.swatch{width:20px;height:3px;display:inline-block}.chart svg{display:block;width:100%;height:auto;min-height:250px}.chart-status{color:var(--muted);font-size:12px;min-height:44px}.chart-help{font-size:12px;color:var(--muted);margin-bottom:0}.chart text{font:12px "Malgun Gothic","Segoe UI",sans-serif}footer{margin-top:40px;color:var(--muted);font-size:12px}
@media(max-width:720px){main{padding:30px 20px 42px}h1{font-size:28px}.lead{font-size:17px;padding-left:14px}h2{font-size:21px}.chart{padding:16px 8px}.chart svg{min-height:220px}th,td{padding:10px;font-size:12px}}
@media print{body,main{background:white}main{max-width:none;padding:0;border:0}h2{break-after:avoid}table,.chart{break-inside:avoid}.chart-buttons,.legend{display:none}a{color:inherit}body{font-size:11pt}.table-scroll{overflow:visible}th,td{white-space:normal;font-size:9pt}.lead{font-size:13pt}}
'''

CHART_HTML = '''
<section class="chart" aria-labelledby="curve-heading">
<h3 id="curve-heading">같은 원금에서 출발한 자산곡선</h3>
<p class="chart-subtitle">구간 시작 직전 자산 = 100 · 원화 · 비용 차감 · 배당·세금 제외</p>
<div class="chart-buttons" role="group" aria-label="자산곡선 기간">
<button data-period="full" aria-pressed="true">전체</button>
<button data-period="train" aria-pressed="false">1~5월 선발</button>
<button data-period="validation" aria-pressed="false">6~7월 선택</button>
<button data-period="holdout" aria-pressed="false">8~9월 사후 홀드아웃</button>
</div><div class="legend" id="legend" role="group" aria-label="표시할 곡선"></div>
<svg id="curves" viewBox="0 0 1080 480" role="img" aria-labelledby="curve-heading curve-description">
<desc id="curve-description">선택 규칙 및 비교 ETF의 누적 자산곡선. 정확한 수익률은 위 표와 아래 구간 요약에서 확인할 수 있습니다.</desc>
</svg><p class="chart-status" id="chart-status" aria-live="polite"></p>
<p class="chart-help">범례를 클릭해 비교 대상을 고르고, 그래프 위에 포인터를 올리면 날짜별 자산을 확인할 수 있다. 구간별 값은 진행 중 계좌를 다시 100으로 환산했다.</p>
<noscript><p>자바스크립트가 꺼져 있다. 동일한 수익률은 위 성과표에서 확인할 수 있다.</p></noscript>
</section>
'''

CHART_JS = r'''
(()=>{'use strict';const data=JSON.parse(document.getElementById('curve-data').textContent);
const svg=document.getElementById('curves'),legend=document.getElementById('legend'),status=document.getElementById('chart-status');
const NS='http://www.w3.org/2000/svg',colors=['#087c81','#db783b','#29578b','#9470ac','#737c26','#c64e73','#1d9171','#674c91'];
const specialColors={'주 후보':'#087c81','TIME':'#db783b','KoAct':'#29578b','추가진단·상한 5종목':'#9470ac','QQQ·원화':'#737c26'};
const colorFor=(name,i)=>specialColors[name]||colors[i%colors.length];
const names=Object.keys(data.series),visible=new Set(data.default_visible||names);let period='full',range=[],drawn=[],hover;
const make=(tag,attrs={},text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n};
names.forEach((name,i)=>{const lab=document.createElement('label'),cb=document.createElement('input'),sw=document.createElement('span'),tx=document.createElement('span');cb.type='checkbox';cb.checked=visible.has(name);cb.setAttribute('aria-label',name+' 표시');sw.className='swatch';sw.style.background=colorFor(name,i);tx.textContent=name;lab.append(cb,sw,tx);legend.append(lab);cb.addEventListener('change',()=>{cb.checked?visible.add(name):visible.delete(name);render()})});
document.querySelectorAll('[data-period]').forEach(b=>b.addEventListener('click',()=>{period=b.dataset.period;document.querySelectorAll('[data-period]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));render()}));
const percent=n=>(n>=0?'+':'')+n.toFixed(2)+'%';
function render(){while(svg.lastChild)svg.removeChild(svg.lastChild);const desc=make('desc',{id:'curve-description'},'구간 시작 직전 자산을 100으로 환산한 누적 자산곡선.');svg.append(desc);
range=data.dates.map((d,i)=>({d,i})).filter(({d})=>period==='full'||(period==='train'?d<'2026-06-01':period==='validation'?d>='2026-06-01'&&d<'2026-08-01':d>='2026-08-01'));
if(!range.length){status.textContent='이 구간의 관측값이 없습니다.';return}const first=range[0].i,last=range[range.length-1].i;
drawn=names.filter(n=>visible.has(n)).map(name=>{const base=first?data.series[name][first-1]:1;return{name,color:colorFor(name,names.indexOf(name)),values:range.map(({i})=>100*data.series[name][i]/base)}});
const all=[100,...drawn.flatMap(x=>x.values).filter(Number.isFinite)];let lo=Math.min(...all),hi=Math.max(...all);let pad=Math.max((hi-lo)*.08,3);lo=Math.max(0,lo-pad);hi+=pad;
const X=70,Y=26,W=990,H=390,x=i=>X+i*W/Math.max(1,range.length-1),y=v=>Y+H-(v-lo)/(hi-lo)*H;
for(let i=0;i<6;i++){const v=lo+(hi-lo)*i/5,yy=y(v);svg.append(make('line',{x1:X,x2:X+W,y1:yy,y2:yy,stroke:'#e1e9ed'}));svg.append(make('text',{x:X-12,y:yy+4,'text-anchor':'end',fill:'#5c6b73'},v.toFixed(0)))}
svg.append(make('line',{x1:X,x2:X+W,y1:y(100),y2:y(100),stroke:'#b1c2cb','stroke-dasharray':'4 4'}));
if(period==='full'){for(const[day,label]of[['2026-06-01','선택 구간'],['2026-08-01','사후 홀드아웃']]){const j=range.findIndex(r=>r.d>=day);if(j>=0){svg.append(make('line',{x1:x(j),x2:x(j),y1:Y,y2:Y+H,stroke:'#93a9b5','stroke-dasharray':'5 5'}));svg.append(make('text',{x:x(j)+7,y:Y+14,fill:'#5c6b73'},label))}}}
const tickCount=Math.min(5,range.length);for(let i=0;i<tickCount;i++){const j=Math.round(i*(range.length-1)/Math.max(1,tickCount-1));svg.append(make('text',{x:x(j),y:Y+H+28,'text-anchor':i===0?'start':i===tickCount-1?'end':'middle',fill:'#5c6b73'},range[j].d))}
drawn.forEach(s=>{const segments=[];let segment=[];s.values.forEach((v,i)=>{if(Number.isFinite(v)){segment.push(x(i)+','+y(v))}else if(segment.length){segments.push(segment);segment=[]}});if(segment.length)segments.push(segment);segments.forEach(points=>svg.append(make('polyline',{points:points.join(' '),fill:'none',stroke:s.color,'stroke-width':2.8,'stroke-linejoin':'round','stroke-linecap':'round'})))});
hover=make('line',{x1:X,x2:X,y1:Y,y2:Y+H,stroke:'#7a919e','stroke-dasharray':'3 3',visibility:'hidden'});svg.append(hover);
const summary=range[0].d+' ~ '+range[range.length-1].d+' · '+drawn.map(s=>s.name+' '+percent(s.values[s.values.length-1]-100)).join(' / ');status.textContent=summary;
svg.onpointermove=e=>{const rect=svg.getBoundingClientRect(),px=(e.clientX-rect.left)/rect.width*1080,j=Math.max(0,Math.min(range.length-1,Math.round((px-X)/W*(range.length-1))));hover.setAttribute('x1',x(j));hover.setAttribute('x2',x(j));hover.setAttribute('visibility','visible');status.textContent=range[j].d+' · '+drawn.map(s=>s.name+' '+s.values[j].toFixed(2)).join(' / ')};
svg.onpointerleave=()=>{hover.setAttribute('visibility','hidden');status.textContent=summary};}
render()})();
'''


def write_report(out: Path) -> None:
    out = Path(out)
    evaluation, subgroups, stress, protocol, metadata, curves = [
        _load(out, name) for name in ('evaluation.json', 'subgroup_evaluation.json', 'stress.json',
                                      'protocol.json', 'metadata.json', 'equity_curves.json')]
    doc = _main_report(evaluation, subgroups, stress, protocol, metadata, out)
    if not curves.get('dates') or not curves.get('series'):
        raise ValueError('Equity curves require nonempty dates and series')
    for label, series in curves['series'].items():
        if len(series) != len(curves['dates']):
            raise ValueError(f'Equity curve length mismatch: {label}')
    curve_labels = {'primary_selected': '주 후보', 'hindsight_best': '전체기간 최고·사후 선택',
                    'TIME_top10_weekly': 'TIME 상위 10·주간', 'KoAct_top10_weekly': 'KoAct 상위 10·주간',
                    'diversified_ge3': '추가진단·상한 3개 이상',
                    'both_funds_mixed_ge3': '추가진단·두 펀드 혼합',
                    'weekly_ge3': '추가진단·주간', 'daily_ge3': '추가진단·일간',
                    'monthly_ge3': '추가진단·월간', 'TIME ETF': 'TIME', 'KoAct ETF': 'KoAct',
                    'QQQ KRW': 'QQQ·원화', 'robust_all': '재선발·전체(기본)',
                    'robust_mixed_max_stocks_3_to_10': '재선발·혼합 상한 3~10(기본)',
                    'robust_mixed_max_stocks_5_to_10': '재선발·혼합 상한 5~10(기본)'}
    curve_labels.update({f'max_stocks_{size}': f'추가진단·상한 {size}종목' for size in (1, 2, 3, 5, 7, 10)})
    curves = {**curves, 'series': {curve_labels.get(label, label): series
                                  for label, series in curves['series'].items()},
              'default_visible': ['주 후보', 'TIME', 'KoAct', '추가진단·상한 5종목']}
    chart_data = json.dumps(curves, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    body = '\n'.join(doc.html).replace('<!--CURVE_CHART-->', CHART_HTML)
    html = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>TIME × KoAct 매매기록 조합 연구</title><style>' + CSS + '</style></head><body><main>'
            + body + '</main><script id="curve-data" type="application/json">' + chart_data
            + '</script><script>' + CHART_JS + '</script></body></html>')
    atomic_write_text(out / 'report.md', '\n'.join(doc.md))
    atomic_write_text(out / 'report.html', html)


def main() -> None:
    from .research import OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    write_report(args.out)


if __name__ == '__main__':
    main()
