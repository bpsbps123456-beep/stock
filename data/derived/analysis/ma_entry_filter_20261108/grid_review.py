"""Exhaustive 1..200 MA-length study; reuse the entry gate and portfolio engine.

Run fetch_warmup.py first, then this file. Never refresh the operating desk cache.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'grid_1_200'
spec = importlib.util.spec_from_file_location('ma_entry_review_grid', HERE / 'review.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
TIE = 1e-9  # percentage points; deterministic numerical ties


def candidate_grid(protocol):
    grid = protocol['grid']
    return [(kind, n) for kind in grid['type']
            for n in range(grid['min_length'], grid['max_length'] + 1, grid['step'])]


def select_front(rows, kind=None):
    """Read front returns only. Neither back returns nor MDD can select the winner."""
    candidates = [r for r in rows if r['name'] != 'baseline' and (kind is None or r['kind'] == kind)]
    best = max(r['front_return_pct'] for r in candidates)
    tied = [r for r in candidates if best - r['front_return_pct'] <= TIE]
    return min(tied, key=lambda r: (0 if r['kind'] == 'SMA' else 1, r['length']))['name']


def hindsight_best(rows, field):
    candidates = [r for r in rows if r['name'] != 'baseline']
    best = max(r[field] for r in candidates)
    return min((r for r in candidates if best - r[field] <= TIE),
               key=lambda r: (0 if r['kind'] == 'SMA' else 1, r['length']))['name']


def signal_history(book):
    """Prepend signal-only warmup; existing ranking/trading/FX inputs stay frozen."""
    warm = pd.read_csv(OUT / 'inputs/close_warmup.csv', index_col=0, parse_dates=True)
    warm.index = warm.index.strftime('%Y-%m-%d')
    if warm.empty or warm.index.max() >= book.days[0]:
        raise ValueError('Warmup must be strictly earlier than the frozen book.')
    if warm.index.min() < '2025-01-01':
        raise ValueError('Only 2025 warmup is authorised for this study.')
    combined = pd.concat([warm.reindex(columns=book.cls.columns), book.cls])
    if combined.index.has_duplicates or not combined.index.is_monotonic_increasing:
        raise ValueError('Signal history must have unique increasing sessions.')
    pd.testing.assert_frame_equal(combined.loc[book.cls.index], book.cls)
    return combined


def run_variant(book, signals, kind, length):
    average = M.moving_average(signals, kind, length).reindex(book.cls.index)
    events = []
    with M.entry_gate(book, kind, length, events, averages=average):
        path = M.S.run_model(book)
    M.validate(book, path, events)
    return path, events


def annotate_events(events, kind, length, valid_counts):
    output = []
    for e in events:
        count = int(valid_counts.at[e['signal_date'], e['ticker']])
        if e['allowed']:
            reason = 'allowed'
        elif not np.isfinite(e['close']):
            reason = 'missing_close'
        elif e['ma'] is None:
            reason = 'insufficient_history' if count < length else 'missing_recent_history'
        else:
            reason = 'below_ma'
        output.append({'name': f'{kind}{length}', 'kind': kind, 'length': length,
                       **e, 'history_count': count, 'reason': reason})
    return output


def frozen_input_check():
    old = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    for name, expected in old['input_sha256'].items():
        if hashlib.sha256((M.ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen input changed: {name}')
    return old


def flatten(name, kind, length, periods, events):
    row = {'name': name, 'kind': kind, 'length': length}
    for label, metrics in periods.items():
        for field in ['return_pct', 'mdd_pct', 'mean_open_exposure_pct']:
            row[f'{label}_{field}'] = metrics[field]
    blocked = [e for e in events if not e['allowed']]
    row.update(blocked_attempts=len(blocked),
               below_ma_attempts=sum(e['reason'] == 'below_ma' for e in blocked),
               missing_history_attempts=sum(e['reason'] != 'below_ma' for e in blocked),
               unique_blocked_tickers=len({e['ticker'] for e in blocked}),
               front_blocked=sum(e['session'] <= '2026-05-29' for e in blocked),
               back_blocked=sum(e['session'] >= '2026-06-01' for e in blocked))
    return row


def markdown_table(names, indexed):
    lines = ['| 설정 | 1~5월 수익 | 6~11/4 수익 | 뒤 최대낙폭 | 전체 수익 | 뒤 평균투자비중 | 보류 시도 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        r = indexed[name]
        lines.append(f"| {name} | {r['front_return_pct']:+.2f}% | {r['back_return_pct']:+.2f}% | "
                     f"{r['back_mdd_pct']:.2f}% | {r['full_return_pct']:+.2f}% | "
                     f"{r['back_mean_open_exposure_pct']:.1f}% | {r['blocked_attempts']} |")
    return lines


def write_report(result, rows):
    ix = {r['name']: r for r in rows}
    selected = result['selected_by_front_only']
    base, chosen = ix['baseline'], ix[selected]
    main_names = list(dict.fromkeys(['baseline', selected, *result['selected_by_type_front_only'].values()]))
    lines = ['# 신규 편입 이평선 필터: 1~200일 전수 비교', '',
             '사용자 요청대로 SMA·EMA 각각 1~200일을 1일 간격으로 모두 계산했다. 이번 400후보+현행1이며, 기존6개를 포함한 고유 누적400후보다(새 조합394개). 종목마다 다른 기간을 고르거나 손절을 붙이는 조합은 시험하지 않았다.', '',
             '## 먼저 고르고 뒤에서 확인', '',
             '2026년 1/2~5/29 수익률로만 고른 후보를 6/1~11/4에서 확인했다. 기존에 본 2026 자료를 더 촘촘히 탐색했으므로 독립 새 검증은 아니다. 비용 전 원화·정규장 시가 가정 체결이며 실계좌 성과가 아니다.', '']
    lines += markdown_table(main_names, ix)
    best_front = max(r['front_return_pct'] for r in rows if r['name'] != 'baseline')
    ties = [r for r in rows if r['name'] != 'baseline' and best_front-r['front_return_pct'] <= TIE]
    lines += ['', f"앞 구간 최고 수익은 {best_front:+.2f}%로 **{len(ties)}개가 동률**이다. "
              '사전에 고정한 동률 처리(종류 SMA 우선·짧은 기간 우선)로 1일선이 선택됐다. '
              '1일선은 종가 자신과 비교하므로 필터 효과가 없는 대조다. 앞 구간만으로 23일선을 특정할 근거는 없다.']
    lines += ['', f"앞 구간 선택은 **{selected}**. 뒤 구간 현행 대비 수익 차이는 "
              f"**{chosen['back_return_pct']-base['back_return_pct']:+.2f}%p**, 최대낙폭 차이는 "
              f"**{chosen['back_mdd_pct']-base['back_mdd_pct']:+.2f}%p**다.", '',
              f"400개 중 뒤 구간 수익이 현행보다 높은 후보는 **{result['counts']['back_return_better']}/400**, "
              f"뒤 최대낙폭도 함께 나아진 후보는 **{result['counts']['back_return_and_mdd_better']}/400**, "
              f"앞·뒤 수익이 모두 현행보다 높은 후보는 **{result['counts']['front_and_back_return_better']}/400**이다.", '',
              '## 결과를 보고 고른 최고값 — 사후 발견', '',
              '아래는 뒤 구간과 전체 구간 결과를 보고 고른 값이다. 실제로 그 전에 선택했을 성적이나 검증된 최적 기간으로 해석하지 않는다.', '']
    hindsight = list(dict.fromkeys(['baseline', result['best_back_hindsight'], result['best_full_hindsight']]))
    lines += markdown_table(hindsight, ix)
    best_back = ix[result['best_back_hindsight']]
    lines += ['', f"사후 최고인 {result['best_back_hindsight']}도 뒤 수익 개선은 "
              f"{best_back['back_return_pct']-base['back_return_pct']:+.4f}%p, "
              f"뒤 최대낙폭 개선은 {best_back['back_mdd_pct']-base['back_mdd_pct']:+.4f}%p다. "
              '이번 표본에서 약간 나은 기간은 있지만, 큰 하락을 줄이는 개선책으로 볼 정도의 차이는 확인되지 않았다.']
    for label, field in [('뒤 구간 수익 상위10', 'back_return_pct'), ('전체 구간 수익 상위10', 'full_return_pct')]:
        top = sorted((r for r in rows if r['name'] != 'baseline'), key=lambda r: (-r[field], r['kind'], r['length']))[:10]
        lines += ['', f'### {label} (사후)', ''] + markdown_table([r['name'] for r in top], ix)
    winner = ix[result['best_back_hindsight']]
    neighbors = [f"{winner['kind']}{n}" for n in range(max(1, winner['length']-5), min(200, winner['length']+5)+1)]
    lines += ['', '## 사후 뒤 구간 1위 주변 ±5일', '',
              '하루 차이의 인접 기간은 같은 거래를 많이 공유한다. 최고값의 주변이 같이 좋은지 보는 민감도이며 추가 독립 검증은 아니다.', '']
    lines += markdown_table(neighbors, ix)
    lines += ['', '## 해석·재현', '',
              '- 개별 종목의 신규 편입만 전일 확정 종가로 판정한다. 기존 보유는 면제하며 보류 배분은 현금, 대체·재정규화 없음. 주중 회복해도 다음 주간 재조정까지 기다린다.',
              '- 기존 매매·순위·환율·PDF 입력은 그대로다. MA 계산에만 2025년 초 Close를 보완했다. 200일선의 가짜 이력 부족을 피하고, 실제 상장 이력 부족·결측은 차단 사유로 따로 센다. 2025년 수익은 평가하지 않았다.',
              '- EMA는 첫 유효 가격 시드여서 더 긴 예열로 이전6후보와 초기값 영향이 달라질 수 있다. 결과 JSON의 legacy_comparison에 차이를 공개한다. 같은 입력·같은 길이의 SMA20·60·120과 현행 곡선은 이전 결과를 재현해야 한다.',
              '- 차단 시도에는 다음 주 같은 종목 재심사가 포함된다. 고유 사건 수가 아니다. 수익 개선 개수도 독립400개의 증거가 아니다.',
              '- 400개에서 최고값을 고르는 선택편향과 기존 7종목 모델의 선택편향이 겹친다. 운용 설정·주문·Pine 기본값은 변경하지 않는다.',
              '- 사용자 수수료0. 호가·환전·세금·현금이자 제외. 기존 엔진의 원화 환산·현금 정의도 유지한다.',
              '- 모든 기간은 summary.csv, 월별은 monthly.csv, 일별 원화 곡선은 curves_krw.csv, 차단 원장은 entry_decisions.csv, 예열 출처와 겹침 대조는 inputs/warmup_meta.json에 있다.', '',
              '독립 검산: `py data/derived/analysis/ma_entry_filter_20261108/audit_grid.py` → `audit.json`. '
              '400개 전체 수익·최대낙폭·월별 산술·차단 원장과 주요 후보의 별도 이평선 계산·곡선 재생을 확인한다. '
              'Python 단위검증은 `py -m unittest`로 실행한다.', '',
              '![기간별 수익과 낙폭](length_comparison.png)', '',
              '재현: 연구 선택 의존성(`pip install -e ".[research]"`, matplotlib 포함)을 설치한 뒤 '
              '`py data/derived/analysis/ma_entry_filter_20261108/fetch_warmup.py` → '
              '`py data/derived/analysis/ma_entry_filter_20261108/grid_review.py`. '
              '동결 예열을 그대로 쓸 때는 첫 명령을 생략한다. 예열 재수집은 새 파일·시각·지문으로 취급한다.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def plot_grid(rows, selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(rows)
    base = frame.loc[frame.name == 'baseline'].iloc[0]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    panels = [('front_return_pct', 'Jan-May 2026: selection return (%)'),
              ('back_return_pct', 'Jun-Nov 4 2026: confirmation return (%)'),
              ('back_mdd_pct', 'Jun-Nov 4 2026: maximum drawdown (%)')]
    for ax, (field, title) in zip(axes, panels):
        for kind, color in [('SMA', '#167d9a'), ('EMA', '#bd6930')]:
            part = frame.loc[frame.kind == kind]
            ax.plot(part.length, part[field], label=kind, color=color, linewidth=1.5)
        ax.axhline(base[field], color='#555555', linestyle='--', linewidth=1, label='No filter')
        chosen = frame.loc[frame.name == selected].iloc[0]
        ax.scatter([chosen.length], [chosen[field]], color='black', marker='*', s=110,
                   zorder=4, label='Selected using Jan-May only')
        ax.set_title(title, loc='left', fontsize=11)
        ax.grid(alpha=.18)
    axes[0].legend(loc='best', fontsize=9)
    axes[-1].set(xlabel='Moving-average length (sessions)', xlim=(1, 200))
    fig.suptitle('New entries only | 400 settings | KRW, before spreads/tax', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT/'length_comparison.png', dpi=150)
    plt.close(fig)


def main():
    started = time.perf_counter()
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    old = frozen_input_check()
    book = M.R.load_book(M.BASE/'inputs', M.END)
    baseline = M.S.run_model(book)
    signals = signal_history(book)
    counts = signals.notna().cumsum().reindex(book.cls.index)
    legacy_curves = pd.read_csv(HERE/'curves_krw.csv', index_col=0)
    np.testing.assert_allclose(M.S.model_index(book, baseline), legacy_curves.baseline, rtol=0, atol=1e-9)
    specs = protocol['periods']
    metrics = {}; rows = []; curves = {}; entries = []; monthly = []
    legacy = {}; validations = {}; grid = candidate_grid(protocol)
    for i, (kind, length) in enumerate([('BASE', 0)] + grid):
        name = 'baseline' if not length else f'{kind}{length}'
        path, events = (baseline, []) if not length else run_variant(book, signals, kind, length)
        events = annotate_events(events, kind, length, counts)
        periods = {label: M.period(book, path, a, z, baseline) for label, (a, z) in specs.items()}
        metrics[name] = periods
        rows.append(flatten(name, kind, length, periods, events))
        curves[name] = M.S.model_index(book, path)
        entries.extend(events)
        for month in range(1, 12):
            label = f'2026-{month:02d}'
            monthly.append({'name': name, 'month': label,
                            **M.period(book, path, label+'-01', label+'-31', baseline)})
        validations[name] = M.validate(book, path, events)
        if length == 1:
            np.testing.assert_array_equal(path.close, baseline.close)
            assert path.rebalances == baseline.rebalances
        if name in old['periods']['full']:
            legacy[name] = {'full_return_difference_pp': periods['full']['return_pct']-old['periods']['full'][name]['return_pct']}
            if kind == 'SMA' or not length:
                np.testing.assert_allclose(curves[name], legacy_curves[name], rtol=0, atol=1e-8)
                legacy[name]['same_curve_verified'] = True
        if i and i % 25 == 0:
            print(f'{i}/400 complete ({time.perf_counter()-started:.1f}s)', flush=True)
    frame = pd.DataFrame(rows)
    base = rows[0]
    candidates = frame.loc[frame.name != 'baseline']
    rb = candidates.back_return_pct > base['back_return_pct'] + TIE
    mb = candidates.back_mdd_pct > base['back_mdd_pct'] + TIE
    fb = candidates.front_return_pct > base['front_return_pct'] + TIE
    selected = select_front(rows)
    result = {'end_us': M.END, 'protocol_sha256': hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
              'selected_by_front_only': selected,
              'selected_by_type_front_only': {kind: select_front(rows, kind) for kind in protocol['grid']['type']},
              'best_back_hindsight': hindsight_best(rows, 'back_return_pct'),
              'best_full_hindsight': hindsight_best(rows, 'full_return_pct'),
              'counts': {'candidates': len(candidates), 'back_return_better': int(rb.sum()),
                         'back_return_and_mdd_better': int((rb & mb).sum()),
                         'front_and_back_return_better': int((fb & rb).sum())},
              'periods': metrics, 'validation': validations, 'legacy_comparison': legacy,
              'input_sha256': {**old['input_sha256'],
                  **{str(p.relative_to(M.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((OUT/'inputs').glob('*')) if p.is_file()}}}
    assert len(grid) == 400 and len(set(grid)) == 400
    (OUT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    frame.to_csv(OUT/'summary.csv', index=False)
    pd.DataFrame(curves, index=baseline.days).to_csv(OUT/'curves_krw.csv', index_label='us_session')
    pd.DataFrame(entries).to_csv(OUT/'entry_decisions.csv', index=False)
    pd.DataFrame(monthly).drop(columns=['usd', 'usd_mean_exposure_control']).to_csv(OUT/'monthly.csv', index=False)
    plot_grid(rows, selected)
    write_report(result, rows)
    print(json.dumps({k: result[k] for k in ['selected_by_front_only', 'selected_by_type_front_only',
                     'best_back_hindsight', 'best_full_hindsight', 'counts', 'legacy_comparison']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
