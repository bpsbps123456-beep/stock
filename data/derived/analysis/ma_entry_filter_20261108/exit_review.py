"""Daily individual MA exits, using the same weekly selection and wealth engine."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'exit_1_200'
spec = importlib.util.spec_from_file_location('ma_entry_grid_for_exits', HERE / 'grid_review.py')
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
M, S = G.M, G.M.S


def below_ma(close, average):
    return bool(np.isfinite(close) and np.isfinite(average) and close < average)


def make_exit_callback(book, averages, name, exits, unknown):
    """Shared prior-close predicate for both the seven-stock and cross-stock studies."""
    closes = {key: book.cls[tk].to_numpy(float) for key, tk in book.ticker.items()}
    means = {key: averages[tk].to_numpy(float) for key, tk in book.ticker.items()}

    def at_open(b, k, day, weights):
        requested = []
        for key, weight in weights.items():
            c, a = closes[key][k-1], means[key][k-1]
            event = {'name': name, 'session': day, 'signal_date': b.days[k-1],
                     'key': key, 'ticker': b.ticker[key], 'close': float(c),
                     'ma': float(a) if np.isfinite(a) else None, 'weight': weight}
            if not (np.isfinite(c) and np.isfinite(a)):
                unknown.append(event)
            elif below_ma(c, a):
                requested.append(key)
                exits.append({**event, 'open': S._price_at(b, key, k)})
        return requested
    return at_open


def run_variant(book, signals, kind, length, averages=None):
    average = (M.moving_average(signals, kind, length).reindex(book.cls.index)
               if averages is None else averages)
    entries, exits, unknown = [], [], []
    at_open = make_exit_callback(book, average, f'{kind}{length}', exits, unknown)

    with M.entry_gate(book, kind, length, entries, averages=average):
        path = S.run_model(book, exit_at_open=at_open)
    M.validate(book, path, entries)
    # Every requested exit executes and cannot be repurchased below the MA that open.
    holdings = dict(zip(path.days, path.weights))
    executed = {(e['date'], e['key']): e for e in path.events if e.get('reason') == 'exit_rule'}
    assert len(executed) == len(exits)
    for e in exits:
        assert e['signal_date'] < e['session']
        assert below_ma(e['close'], e['ma'])
        assert e['open'] is not None and e['open'] > 0
        assert executed[e['session'], e['key']]['price'] == e['open']
        assert e['key'] not in holdings[e['session']]
    for e in path.events:
        assert e['price'] is not None and e['price'] > 0
        if e['side'] == 'buy':
            assert e['date'] in path.rebalances
    return path, entries, exits, unknown


def turnover(book, path):
    """All weight changes, including partial weekly rebalances; no alternate NAV engine."""
    rows = []
    previous = {}
    for day, final in zip(path.days, path.weights):
        k = book.days.index(day)
        before = S._drift(previous, S._open_returns(book, previous, k))[0] if previous else {}
        keys = set(before) | set(final)
        buys = sum(max(0., final.get(key, 0.) - before.get(key, 0.)) for key in keys)
        sells = sum(max(0., before.get(key, 0.) - final.get(key, 0.)) for key in keys)
        rows.append({'session': day, 'buy_weight': buys, 'sell_weight': sells,
                     'one_way_turnover': buys+sells})
        previous = final
    return rows


def friction_metrics(curve, daily, specs, days):
    output = {}
    t = np.array([r['one_way_turnover'] for r in daily])
    for bps in (5, 10):
        adjusted = curve * np.cumprod(1. - t * bps / 10000.)
        for label, (a, z) in specs.items():
            idx = [i for i, day in enumerate(days) if a <= day <= z]
            lo, hi = idx[0], idx[-1]
            metrics = M.R.metrics(np.r_[adjusted[lo-1] if lo else 100., adjusted[lo:hi+1]])
            output[f'{label}_return_{bps}bp_pct'] = metrics['return_pct']
    return output


def table(names, ix):
    lines = ['| 설정 | 앞1~5월 수익 | 뒤6~11/4 수익 | 뒤 최대낙폭 | 전체 수익 | 뒤 투자비중 | MA 전량매도 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in dict.fromkeys(names):
        r = ix[name]
        lines.append(f"| {name} | {r['front_return_pct']:+.2f}% | {r['back_return_pct']:+.2f}% | "
                     f"{r['back_mdd_pct']:.2f}% | {r['full_return_pct']:+.2f}% | "
                     f"{r['back_mean_open_exposure_pct']:.1f}% | {r['ma_exits']} |")
    return lines


def report(result, rows, entry):
    ix = {r['name']: r for r in rows}
    selected = result['selected_by_front_only']
    base, chosen = ix['baseline'], ix[selected]
    lines = ['# 종목별 이평선 아래 전량매도: 1~200일 전수 비교', '',
             '2026-11-09 사용자 추가 가설. 종목이 이평선 아래이면 **그 종목 보유분 전체**를 판다. '
             '한 종목 이탈로 계좌 전체를 파는 조건은 아니다. 이번 SMA·EMA 각200개=400후보+현행1, '
             '이전 신규 편입 보류400개까지 누적800개 규칙·기간 조합이다.', '',
             '## 앞 구간으로 고르고 뒤 구간 확인', '',
             '1/2~5/29 원화 수익으로만 고른 후보를 6/1~11/4에서 확인한다. 이미 본2026자료 재사용이므로 '
             '독립 새 검증은 아니다. 전일 완성 종가로 판단하고 다음 정규장 시가에 가상 체결한다.', '']
    lines += table(['baseline', selected, *result['selected_by_type_front_only'].values()], ix)
    lines += ['', f"앞 구간 선택 **{selected}**의 뒤 구간 수익 개선은 "
              f"{chosen['back_return_pct']-base['back_return_pct']:+.2f}%p, 최대낙폭 개선은 "
              f"{chosen['back_mdd_pct']-base['back_mdd_pct']:+.2f}%p다. "
              f"앞 구간 수익 1위 동률 {result['front_tie_count']}개. 뒤 구간 수익과 최대낙폭 모두 현행보다 "
              f"좋은 후보는 {result['counts']['back_return_and_mdd_better']}/400개다.", '',
              '## 같은 이평선으로 신규 편입만 보류했을 때와 비교', '',
              '| 설정 | 신규보류만: 뒤 수익 | 전량매도: 뒤 수익 | 신규보류만: 뒤 최대낙폭 | 전량매도: 뒤 최대낙폭 |',
              '|---|---:|---:|---:|---:|']
    names = list(dict.fromkeys([selected, result['best_back_hindsight'], result['best_full_hindsight'],
                               'SMA23', 'SMA18', 'SMA51']))
    for name in names:
        r, e = ix[name], entry.loc[name]
        lines.append(f"| {name} | {e.back_return_pct:+.2f}% | {r['back_return_pct']:+.2f}% | "
                     f"{e.back_mdd_pct:.2f}% | {r['back_mdd_pct']:.2f}% |")
    lines += ['', '## 사후 최고값과 이웃 기간', '',
              '뒤 구간·전체 구간을 보고 고른 최고값은 사후 발견이다. 사전에 선택한 성적이나 최적값으로 쓰지 않는다.', '']
    lines += table(['baseline', result['best_back_hindsight'], result['best_full_hindsight'],
                    result['best_back_mdd_hindsight']], ix)
    best = ix[result['best_back_hindsight']]
    lines += ['', '뒤 수익 사후1위 주변 ±5일:', '']
    lines += table([f"{best['kind']}{n}" for n in range(max(1, best['length']-5), min(200, best['length']+5)+1)], ix)
    whole = ix[result['best_full_hindsight']]
    lines += ['', '전체 수익 사후1위 주변 ±5일. 투자비중을 줄인 효과와 기간 경계의 민감도를 같이 본다:', '']
    lines += table([f"{whole['kind']}{n}" for n in range(max(1, whole['length']-5), min(200, whole['length']+5)+1)], ix)
    lines += ['', '## 거래 빈도와 가상 호가비용 민감도', '',
              '수수료는 사용자 조건인0. 아래5·10bp는 실제 호가 추정이 아닌 가상 조건이다. 매일 시가 전후 비중 차이의 '
              '절댓값 합(부분 재조정 포함)에 비용률을 곱해 자산을 비례 차감한다. 매수·매도 양방향을 포함하며 '
              '동일 회전율 계산을 현행에도 적용한다. 비중의 비용 피드백·실제 체결 슬리피지를 재현하는 시뮬레이터는 아니다.', '',
              '| 설정 | 전체 신규편입 | 전체 청산 | 그중 MA청산 | 전체 단방향 회전율 | 뒤 수익 5bp | 뒤 수익 10bp |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name in dict.fromkeys(['baseline', selected, result['best_back_hindsight'], result['best_full_hindsight']]):
        r = ix[name]
        lines.append(f"| {name} | {r['new_entries']} | {r['full_exits']} | {r['ma_exits']} | "
                     f"{r['full_turnover']:.2f}배 | {r['back_return_5bp_pct']:+.2f}% | {r['back_return_10bp_pct']:+.2f}% |")
    lines += ['', '## 계산 경계·검증', '',
              '- 기존 7종목 순위·PDF 시점·상한·원화 환산과 현금 정의를 유지한다. MA 예열만 2025년을 쓰고 성과는2026만 평가한다. 입력 SHA를 확인해 앞선 신규보류 연구와 같은 자료로 비교한다.',
              '- 밤사이 갭 손익을 반영한 다음 매도한다. 종가나 이평선 가격으로 팔았다고 가정하지 않는다. 종가<MA가 조건이며 같으면 유지한다.',
              '- 매도분은 현금으로 두고 주중 즉시 재매수하지 않는다. 팔린 이름은 유지 버퍼에서 제거하며 청산 당일 임의 대체·재배분하지 않는다. 정기 주간에는 기존 규칙으로 종목을 재선택하며 회복 종목도 이때만 재편입할 수 있다.',
              '- MA·종가 결측은 매도 근거 부족으로 유지하되 신규 편입은 보류한다. 실제 보유 신호 결측은 '
              f"{result['unknown_held_signals']}건이다. 모든 의도한 체결의 양수 시가를 확인했다.",
              '- 수익·낙폭 감소와 평균 투자비중 감소를 함께 본다. 결과 JSON의 USD 동일 평균노출 대조는 사후 설명용으로, 미리 알 수 있는 매매 규칙이 아니다.',
              '- 400설정 선택편향·인접 기간 동일 거래·기존 모델 선택편향이 겹친다. 이미 본 자료를 앞뒤로 나누었으므로 앞으로 종이 기록으로 확인해야 한다. 운용·주문·계좌·Pine 기본값은 바꾸지 않았다.',
              '- summary.csv 전체401행, monthly.csv 월별, curves_krw.csv 일별, entry_decisions.csv 신규 판단, exits.csv 전량매도, turnover.csv 회전율을 보존한다. unknown_held_signals.csv는 결측 판단 기록이며 0건이면 헤더만 있다.',
              '- audit_exit.py는 다른 MA 구현으로 모든 매도 시점·값과 주요 경로를 재생한다. audit.json에 검산 결과, 전체 unittest와 desk validate는 저장소 검증에서 확인한다.', '',
              '![기간별 성과](length_comparison.png)', '',
              '재현: `py data/derived/analysis/ma_entry_filter_20261108/exit_review.py` → '
              '`py data/derived/analysis/ma_entry_filter_20261108/audit_exit.py`. '
              '기존 grid_1_200의 예열 입력을 사용하며 가격을 재수집하지 않는다.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    started = time.perf_counter()
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    G.frozen_input_check()
    previous = json.loads((G.OUT/'results.json').read_text(encoding='utf-8'))
    for name, expected in previous['input_sha256'].items():
        assert hashlib.sha256((M.ROOT/name).read_bytes()).hexdigest() == expected, name
    book = M.R.load_book(M.BASE/'inputs', M.END)
    signals = G.signal_history(book)
    counts = signals.notna().cumsum().reindex(book.cls.index)
    baseline = S.run_model(book)
    reference = pd.read_csv(G.OUT/'curves_krw.csv', index_col=0)
    np.testing.assert_allclose(S.model_index(book, baseline), reference.baseline, rtol=0, atol=1e-9)
    entry = pd.read_csv(G.OUT/'summary.csv').set_index('name')
    grid = G.candidate_grid(protocol)
    assert len(grid) == len(set(grid)) == 400
    rows, curves, metrics, entries, exits, unknown, turns, monthly = [], {}, {}, [], [], [], [], []
    specs = protocol['periods']
    for i, (kind, length) in enumerate([('BASE', 0)] + grid):
        name = 'baseline' if not length else f'{kind}{length}'
        path, ev, ex, un = (baseline, [], [], []) if not length else run_variant(book, signals, kind, length)
        ev = G.annotate_events(ev, kind, length, counts)
        periods = {label: M.period(book, path, a, z, baseline) for label, (a, z) in specs.items()}
        row = G.flatten(name, kind, length, periods, ev)
        row.update(ma_exits=len(ex), unknown_held_signals=len(un),
                   new_entries=sum(e['side'] == 'buy' for e in path.events),
                   full_exits=sum(e['side'] == 'sell' for e in path.events),
                   back_ma_exits=sum(e['session'] >= '2026-06-01' for e in ex))
        daily = turnover(book, path)
        for label, (a, z) in specs.items():
            row[f'{label}_turnover'] = sum(r['one_way_turnover'] for r in daily if a <= r['session'] <= z)
        curve = S.model_index(book, path)
        row.update(friction_metrics(curve, daily, specs, path.days))
        for field in ['back_return_pct', 'back_mdd_pct', 'full_return_pct']:
            row['vs_entry_'+field+'_pp'] = row[field] - float(entry.loc[name, field])
        rows.append(row); metrics[name] = periods; curves[name] = curve
        entries.extend(ev); exits.extend(ex); unknown.extend(un)
        turns.extend({'name': name, **r} for r in daily)
        for month in range(1, 12):
            label = f'2026-{month:02d}'
            monthly.append({'name': name, 'month': label,
                            **M.period(book, path, label+'-01', label+'-31')})
        if length == 1:
            np.testing.assert_array_equal(path.close, baseline.close)
            assert path.rebalances == baseline.rebalances and not ex
        if i and i % 25 == 0:
            print(f'{i}/400 complete ({time.perf_counter()-started:.1f}s)', flush=True)
    frame = pd.DataFrame(rows)
    base, c = rows[0], frame.loc[frame.name != 'baseline']
    selected = G.select_front(rows)
    best_front = c.front_return_pct.max()
    rb = c.back_return_pct > base['back_return_pct'] + G.TIE
    mb = c.back_mdd_pct > base['back_mdd_pct'] + G.TIE
    result = {'end_us': M.END, 'protocol_sha256': hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
              'selected_by_front_only': selected,
              'selected_by_type_front_only': {kind: G.select_front(rows, kind) for kind in protocol['grid']['type']},
              'front_tie_count': int((best_front-c.front_return_pct <= G.TIE).sum()),
              'best_back_hindsight': G.hindsight_best(rows, 'back_return_pct'),
              'best_full_hindsight': G.hindsight_best(rows, 'full_return_pct'),
              'best_back_mdd_hindsight': G.hindsight_best(rows, 'back_mdd_pct'),
              'counts': {'candidates': len(c), 'back_return_better': int(rb.sum()),
                         'back_mdd_better': int(mb.sum()), 'back_return_and_mdd_better': int((rb & mb).sum()),
                         'front_and_back_return_better': int(((c.front_return_pct > base['front_return_pct']+G.TIE) & rb).sum()),
                         'back_both_better_than_entry_same_n': int(((c.vs_entry_back_return_pct_pp > G.TIE) &
                                                                  (c.vs_entry_back_mdd_pct_pp > G.TIE)).sum())},
              'unknown_held_signals': len(unknown), 'periods': metrics,
              'input_sha256': previous['input_sha256']}
    (OUT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    frame.to_csv(OUT/'summary.csv', index=False)
    pd.DataFrame(curves, index=baseline.days).to_csv(OUT/'curves_krw.csv', index_label='us_session')
    pd.DataFrame(entries).to_csv(OUT/'entry_decisions.csv', index=False)
    pd.DataFrame(exits).to_csv(OUT/'exits.csv', index=False)
    pd.DataFrame(unknown, columns=['name','session','signal_date','key','ticker','close','ma','weight']).to_csv(
        OUT/'unknown_held_signals.csv', index=False)
    pd.DataFrame(turns).to_csv(OUT/'turnover.csv', index=False)
    pd.DataFrame(monthly).drop(columns=['usd']).to_csv(OUT/'monthly.csv', index=False)
    # Reuse the plotting layout, with a label identifying the new rule.
    old_out = G.OUT
    try:
        G.OUT = OUT
        G.plot_grid(rows, selected, title='Daily full exits below MA | 400 settings | KRW, before spreads/tax')
    finally:
        G.OUT = old_out
    report(result, rows, entry)
    print(json.dumps({k: v for k, v in result.items() if k not in ['periods', 'input_sha256']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
