"""Fixed SMA126 transport study across frozen, explicitly listed US stock cohorts.

Each stock receives one separate initial dollar. All wealth calculations reuse the
desk engine. No 7-stock ranking or allocation is claimed for this sleeve experiment.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE/'cross_stock_20261109'
SPEC = importlib.util.spec_from_file_location('ma_exit_for_cross_stock', HERE/'exit_review.py')
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)
M, S = E.M, E.S


@contextmanager
def cached_columns(book):
    """Immutable research-book column access; leave all wealth math unchanged."""
    original = S._col
    frames = [(getattr(book, role), {}) for role in ('opn', 'cls', 'adj')]
    for frame, columns in frames:
        columns.update({t: frame[t].to_numpy(float) for t in book.ticker.values() if t in frame})

    def column(b, frame, key):
        for source, columns in frames:
            if frame is source:
                return columns.get(b.ticker[key])
        return original(b, frame, key)

    with patch.object(S, '_col', column):
        yield


def run_sleeve(book, ticker, mode, length=126, averages=None, start='2026-01-05', *,
               kind='SMA', cache_columns=False, comparison_rtol=0.):
    """One security's capital; weekly eligibility only, daily exits when requested."""
    if mode not in {'buy_hold', 'entry_only', 'exit'}:
        raise ValueError(mode)
    b = copy.copy(book)
    b.ticker = {ticker: ticker}
    b.cls, b.opn = book.cls[[ticker]], book.opn[[ticker]]
    average = (M.moving_average(b.cls, kind, length) if averages is None else averages[[ticker]])
    entries, exits, unknown = [], [], []

    def single_stock_target(b, k, day, current):
        return {'pdf': 'fixed_cross_stock_universe', 'ranked': [{'key': ticker}],
                'selected': [ticker], 'raw': {ticker: 1.}, 'target': {ticker: 1.}}

    with patch.object(S, 'weekly_target', single_stock_target), (cached_columns(b) if cache_columns else nullcontext()):
        if mode == 'buy_hold':
            path = S.run_model(b, start=start)
        else:
            callback = E.make_exit_callback(b, average, f'{kind}{length}', exits, unknown,
                          comparison_rtol=comparison_rtol) if mode == 'exit' else None
            with M.entry_gate(b, kind, length, entries, averages=average, comparison_rtol=comparison_rtol):
                path = S.run_model(b, start=start, exit_at_open=callback)
    holdings = dict(zip(path.days, path.weights))
    for weights in path.weights:
        assert set(weights) <= {ticker}
        assert not weights or abs(weights[ticker]-1.) <= 1e-10
    for event in path.events:
        assert event['price'] is not None and event['price'] > 0
        if event['side'] == 'buy':
            assert event['date'] in path.rebalances
    for event in exits:
        assert event['signal_date'] < event['session']
        assert E.below_ma(event['close'], event['ma'], comparison_rtol=comparison_rtol)
        assert ticker not in holdings[event['session']]
    for event in entries:
        assert event['signal_date'] < event['session']
        assert event['allowed'] == M.eligible(event['close'], event['ma'] if event['ma'] is not None else np.nan,
                                             comparison_rtol=comparison_rtol)
    return path, entries, exits, unknown


def interval(days, begin, end):
    indices = [i for i, day in enumerate(days) if begin <= day <= end]
    return indices[0], indices[-1]


def curve_stats(curve, days, begin, end):
    lo, hi = interval(days, begin, end)
    return M.R.metrics(np.r_[curve[lo-1] if lo else 1., curve[lo:hi+1]])


def basket_stats(curves, exposures, days, begin, end, *, open_nav=None):
    """Equal ORIGINAL capital, no transfers: mean NAV, not mean percentage returns."""
    curves, exposures = np.asarray(curves), np.asarray(exposures)
    if curves.ndim != 2 or curves.shape != exposures.shape or not curves.shape[1]:
        raise ValueError('Need sessions x stocks NAV and exposure arrays.')
    stats = curve_stats(curves.mean(axis=1), days, begin, end)
    lo, hi = interval(days, begin, end)
    # Callers with close NAV supply actual open NAV separately for exposure weights.
    weights = curves if open_nav is None else np.asarray(open_nav)
    stats['mean_open_exposure_pct'] = float(np.mean(
        (weights*exposures).sum(axis=1)[lo:hi+1]/weights.sum(axis=1)[lo:hi+1])*100)
    return stats


def exposure_control(base_curve, exposure, days, begin, end):
    """Ex-post constant-fraction USD control: explanatory, not known at period start."""
    lo, hi = interval(days, begin, end)
    base = np.r_[base_curve[lo-1] if lo else 1., base_curve[lo:hi+1]]
    fraction = float(np.mean(exposure[lo:hi+1]))
    return M.R.metrics(np.r_[1., np.cumprod(1.+(base[1:]/base[:-1]-1.)*fraction)])


def load_inputs():
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    universe = pd.read_csv(OUT/'inputs/universe.csv')
    meta = json.loads((OUT/'inputs/input_meta.json').read_text(encoding='utf-8'))
    tickers = sorted(universe.loc[universe.eligible.astype(str).str.lower().isin(['true','1']), 'ticker'])
    frozen = M.R.load_book(M.BASE/'inputs', M.END)
    frames = {}
    for kind in ('open', 'close', 'adj_close'):
        frame = pd.read_csv(OUT/f'inputs/prices_{kind}.csv', index_col=0, parse_dates=True)
        frame.index = frame.index.strftime('%Y-%m-%d')
        frames[kind] = frame
    book = copy.copy(frozen)
    book.opn, book.cls, book.adj = (frames[k].reindex(frozen.days) for k in ('open','close','adj_close'))
    book.keys, book.ticker = tickers, {t: t for t in tickers}
    book.sector = dict(zip(universe.ticker, universe.sector))
    # FX and calendar are the same frozen references as the original 7-stock study.
    return protocol, universe, meta, tickers, book, frames['close']


def group_members(universe, tickers):
    frame = universe.set_index('ticker').loc[tickers]
    boolean = lambda col: frame[col].astype(str).str.lower().isin(['true','1'])
    jan = boolean('in_time_jan2') | boolean('in_koact_jan2')
    ai, overlap = boolean('in_ai_current'), boolean('discovery_overlap')
    masks = {'jan2_all': jan, 'jan2_external': jan & ~overlap, 'jan2_overlap': jan & overlap,
             'ai_all': ai, 'ai_external': ai & ~overlap,
             'broad_all': pd.Series(True, index=frame.index), 'broad_external': ~overlap,
             'broad_overlap': overlap}
    return {name: list(frame.index[mask]) for name, mask in masks.items() if mask.any()}


def comparison_table(rows, groups):
    frame = pd.DataFrame(rows)
    lines = ['| 집단 | 종목수 | 매수보유: 뒤 수익 | SMA126: 뒤 수익 | 매수보유: 뒤 낙폭 | SMA126: 뒤 낙폭 | 종목별 수익차 중앙값 | 수익·낙폭 동시개선 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group in groups:
        part = frame.loc[(frame.group == group) & (frame.period == 'back')].set_index('strategy')
        base, rule = part.loc['buy_hold'], part.loc['exit_SMA126']
        lines.append(f"| {group} | {int(rule.stocks)} | {base.krw_return_pct:+.2f}% | {rule.krw_return_pct:+.2f}% | "
                     f"{base.krw_mdd_pct:.2f}% | {rule.krw_mdd_pct:.2f}% | {rule.median_return_delta_pp:+.2f}%p | "
                     f"{int(rule.both_better_count)}/{int(rule.stocks)} |")
    return lines


def add_exposure_diagnostics(aggregate, per_stock, groups, primary_curves, days, periods):
    """Same beginning capital weights; per-stock ex-post mean-exposure controls."""
    frame = pd.DataFrame(aggregate).copy()
    stock = pd.DataFrame(per_stock)
    frame['usd_matched_exposure_return_pct'] = np.nan
    frame['usd_vs_matched_exposure_return_pp'] = np.nan
    for i, row in frame.loc[frame.strategy == 'exit_SMA126'].iterrows():
        members = groups[row.group]
        lo, _ = interval(days, *periods[row.period])
        initial = [primary_curves[t+'__exit_SMA126'][lo-1] if lo else 1. for t in members]
        part = stock.loc[(stock.strategy == 'exit_SMA126') & (stock.period == row.period)].set_index('ticker')
        control = float(np.average(part.loc[members, 'usd_matched_exposure_return_pct'], weights=initial))
        frame.at[i, 'usd_matched_exposure_return_pct'] = control
        frame.at[i, 'usd_vs_matched_exposure_return_pp'] = row.usd_return_pct-control
    return frame


def write_report(result, aggregate, per_stock, neighbors):
    lines = ['# 고정126일선, 다른 미국 종목의2026년 적용성 재검증', '',
             f"원래7종목 포트폴리오를 넘어 **{result['eligible_stocks']}종목**을 같은 조건으로 비교했다. "
             f"종목당15설정(매수보유·신규보류126·전량매도SMA120~132), 총{result['path_count']}경로. "
             '126일은 이전 결과에서 발견한 값을 고정했으며 이번 성적에 맞춰 바꾸지 않았다. 이웃13기간은 민감도 확인이다.', '',
             '## 주 비교: 6/1~11/4', '',
             '시초집단은1/2 TIME·KoAct 미국종목 합집합, 최초 거래는1/5. external은 이전400설정에서 실제 편입된 종목을 모두 제외한 군, '
             'overlap은 그 중복군이다. ai는 현재 AI목록이고 broad는 두 목록 합집합이다. '
             '아래 수익·낙폭은 같은 초기금액으로 나눈 종목별 자산의 합(원화)이며, 주간에 종목 간 자금을 재배분하지 않는다.', '']
    lines += comparison_table(aggregate, result['groups'])
    ag = pd.DataFrame(aggregate)
    lines += ['', '## 상승 구간의 대가·전체 결과', '',
              '| 집단 | 매수보유: 앞 구간 | 126일: 앞 구간 | 매수보유: 전체 | 126일: 전체 | 126일: 뒤 평균투자비중 |',
              '|---|---:|---:|---:|---:|---:|']
    for group in ('jan2_all','jan2_external','broad_all','broad_external'):
        part = ag.loc[ag.group == group].set_index(['strategy','period'])
        lines.append(f"| {group} | {part.at[('buy_hold','front'),'krw_return_pct']:+.2f}% | "
                     f"{part.at[('exit_SMA126','front'),'krw_return_pct']:+.2f}% | "
                     f"{part.at[('buy_hold','full'),'krw_return_pct']:+.2f}% | "
                     f"{part.at[('exit_SMA126','full'),'krw_return_pct']:+.2f}% | "
                     f"{part.at[('exit_SMA126','back'),'mean_open_exposure_pct']:.1f}% |")
    lines += ['', '## 같은 평균 투자비중 대조: 사후 진단', '',
              '각 종목에서 그 기간의 전략 평균 투자비중만큼 매수보유 일별수익에 일정하게 노출한 USD 대조다. '
              '바스켓 대조의 시작 종목별 자금은 실제 전략의 해당 구간 시작자산과 같게 맞췄다. '
              '그 기간 평균 비중은 사후에만 알 수 있으므로 실행전략·확정 가능한 대안이 아니다. '
              '현금 증가의 효과와 매도 시점 효과를 구분하는 설명용 비교다.', '',
              '| 집단 | 구간 | 126일 USD수익 | 같은 평균비중 대조 USD수익 | 전략−대조 |',
              '|---|---|---:|---:|---:|']
    for group in ('jan2_external','broad_external','broad_all'):
        for label in ('front','back','full'):
            row = ag.loc[(ag.group == group) & (ag.strategy == 'exit_SMA126') & (ag.period == label)].iloc[0]
            lines.append(f"| {group} | {label} | {row.usd_return_pct:+.2f}% | "
                         f"{row.usd_matched_exposure_return_pct:+.2f}% | {row.usd_vs_matched_exposure_return_pp:+.2f}%p |")
    lines += ['', '## 126일 고정과 이웃 기간', '',
              '| 기간 | 시초 외부군: 뒤 수익 | 뒤 낙폭 | 확대 외부군: 뒤 수익 | 뒤 낙폭 |',
              '|---|---:|---:|---:|---:|']
    for row in neighbors:
        lines.append(f"| SMA{row['length']} | {row['jan2_external_return_pct']:+.2f}% | "
                     f"{row['jan2_external_mdd_pct']:.2f}% | {row['broad_external_return_pct']:+.2f}% | "
                     f"{row['broad_external_mdd_pct']:.2f}% |")
    lines += ['', '## 126일 전량매도 종목별 차이', '',
              '뒤 구간 수익 차이순 상·하위 각10종목(전체는per_stock.csv). 이 표의 종목은 사후 설명용으로 골랐다.', '',
              '| 종목 | 섹터 | 발견 중복 | 매수보유 수익 | 126일 수익 | 수익 차이 | 낙폭 차이 | 뒤 투자비중 | 전체 매도 |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    stock = pd.DataFrame(per_stock)
    stock = stock.loc[(stock.strategy == 'exit_SMA126') & (stock.period == 'back')].sort_values('return_delta_pp')
    for _, r in pd.concat([stock.tail(10).iloc[::-1], stock.head(10)]).drop_duplicates('ticker').iterrows():
        lines.append(f"| {r.ticker} | {r.sector} | {bool(r.discovery_overlap)} | {r.base_krw_return_pct:+.2f}% | "
                     f"{r.krw_return_pct:+.2f}% | {r.return_delta_pp:+.2f}%p | {r.mdd_delta_pp:+.2f}%p | "
                     f"{r.mean_open_exposure_pct:.1f}% | {int(r.exits)} |")
    lines += ['', '## 해석·재현', '',
              '- 원래7종목의순위·버퍼·상한을 제거하고 종목당 동일 초기자금으로 이탈/회복 조건 자체를 비교한 실험이다. 이 결과를7종목 포트폴리오 수익이라고 부르지 않는다.',
              '- 전일 확정 종가<MA이면 다음 정규장 시가에 그 종목 슬리브 전체를 매도한다. 밤사이 갭을 먼저 부담한다. 재편입은 주간 첫미국세션에서 전일Close>=MA일 때만 가능하다. 주중 즉시 반등 재진입은 시험하지 않았다.',
              '- 매수보유는공통첫날시가100%편입. 신규보류126은 최초편입조건만제한하고 편입뒤MA이탈로팔지않는다. 슬리브별 현금은USD다.',
              '- 매수보유와 전략의5월말상태를그대로이어 뒤구간수익을 계산한다. 바스켓은원금동일 NAV평균으로 계산하므로 뒤구간은5월말의종목별자산차이를물려받는다. per_stock.csv의중앙값·개선비율은종목별동일가중이고 바스켓과다른통계다.',
              '- 평균투자비중은 시가NAV로가중한다. basket 곡선은종가NAV로표시한다. CSV의USD동일평균노출대조는그기간평균투자비중을사후에알고줄인설명용대조이며 실행가능한사전규칙이아니다.',
              '- 2026 앞/뒤구간을 나눠고정126의성적을모두보이며, 이번자료로다른기간을다시고르지않는다. 현재AI목록·후일수집한과거PDF·상장생존및연중미보유조건에선택편향이남는다. 시장상관이있어여러종목을독립실험으로세지않는다.',
              '- 전체시초집단·중복군·외부군·섹터별·섹터하나씩뺀값을같이공개해대상선택과큰기여종목의영향을확인한다. 누락/신규상장은inputs/universe.csv의제외사유를보존한다.',
              '- contributions.csv는전체구간USD기준 원금동일 슬리브의기말자산차 기여이며 뒤구간수익률이나원화기여가아니다. 같은기말FX계수를곱하면원화기말자산차로바꿀수있다.',
              '- 수수료0, 배당현금 미반영 가격수익. 호가·환전·세금·현금이자제외. 5/10bp 비용표는회전율비례차감의가상민감도이며실제호가추정이아니다. 원화와USD를동시에보존한다.',
              '- 전체표aggregate.csv·per_stock.csv, 섹터sector.csv·leave_one_sector_out.csv, 민감도neighbors.csv, 일별basket_curves.csv·primary_stock_curves.csv, 실제판단signals.csv를보존한다. 입력은inputs/input_meta.json의출처·SHA로확인한다.', '',
              '![고정126일선 외부 종목 비교](comparison.png)', '',
              '재현: `py data/derived/analysis/ma_entry_filter_20261108/prepare_cross_stock.py` → '
              '`py data/derived/analysis/ma_entry_filter_20261108/cross_stock_review.py` → 독립검산. '
              '이미동결한입력이있으면첫단계생략. 운용·주문·Pine는변경하지않았다.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def plot(curves, days, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, group in zip(axes, ['jan2_external', 'broad_external']):
        for strategy, label, color in [('buy_hold','Buy and hold','#666666'),
                                        ('entry_only_SMA126','Entry gate only','#be9038'),
                                        ('exit_SMA126','Sell below SMA126','#147f8e')]:
            ax.plot(pd.to_datetime(days), curves[group+'__'+strategy]*100, label=label, color=color, lw=1.6)
        ax.set_title(f'{group}: {len(groups[group])} stocks, equal initial capital', loc='left')
        ax.axvline(pd.Timestamp('2026-06-01'), color='#888888', ls=':', lw=1)
        ax.set_ylabel('KRW wealth, initial=100')
        ax.grid(alpha=.2)
    axes[0].legend(loc='best')
    fig.suptitle('Fixed SMA126 | Next-open exits, weekly-only reentry | 2026', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT/'comparison.png', dpi=150)
    plt.close(fig)


def main():
    started = time.perf_counter()
    protocol, universe, meta, tickers, book, signals = load_inputs()
    groups = group_members(universe, tickers)
    info = universe.set_index('ticker')
    lengths = range(120, 133)
    averages = {n: M.moving_average(signals[tickers], 'SMA', n).reindex(book.days) for n in lengths}
    all_specs = [('buy_hold', 'buy_hold', 126), ('entry_only_SMA126', 'entry_only', 126)] + [
        (f'exit_SMA{n}', 'exit', n) for n in lengths]
    paths, curves, open_nav, exposures, costs, trades = {}, {}, {}, {}, {}, []
    per_stock = []
    for j, ticker in enumerate(tickers):
        for name, mode, n in all_specs:
            path, entries, exits, unknown = run_sleeve(book, ticker, mode, n, averages[n])
            assert not unknown, (ticker, name, unknown)
            paths[ticker, name] = path
            curves[ticker, name] = path.close
            open_nav[ticker, name] = path.value
            exposures[ticker, name] = np.array([sum(w.values()) for w in path.weights])
            daily = E.turnover(book, path)
            turnover = np.array([r['one_way_turnover'] for r in daily])
            costs[ticker, name] = {bps: path.close*np.cumprod(1.-turnover*bps/10000.) for bps in (5,10)}
            if name in {'buy_hold','entry_only_SMA126','exit_SMA126'}:
                trades.extend({'ticker': ticker, 'strategy': name, 'stage': 'entry', **e} for e in entries)
                trades.extend({'ticker': ticker, 'strategy': name, 'stage': 'exit', **e} for e in exits)
        if (j+1) % 10 == 0:
            print(f'{j+1}/{len(tickers)} stocks, {(j+1)*15} paths ({time.perf_counter()-started:.1f}s)', flush=True)
    days = next(iter(paths.values())).days
    k0 = book.days.index(days[0])
    fx = book.fx[k0:k0+len(days)]/book.fx[k0-1]
    # Each stock carries the same initial dollar across all periods; never restart in June.
    for ticker in tickers:
        base = curves[ticker, 'buy_hold']
        for name, _, n in all_specs:
            path = paths[ticker, name]
            for label, (begin, end) in protocol['periods'].items():
                usd = curve_stats(curves[ticker,name], days, begin, end)
                krw = curve_stats(curves[ticker,name]*fx, days, begin, end)
                bkrw = curve_stats(base*fx, days, begin, end)
                lo, hi = interval(days, begin, end)
                control = exposure_control(base, exposures[ticker,name], days, begin, end)
                row = {'ticker': ticker, 'sector': info.at[ticker,'sector'],
                       'discovery_overlap': bool(info.at[ticker,'discovery_overlap']),
                       'strategy': name, 'length': n, 'period': label,
                       **{'usd_'+key: value for key,value in usd.items()},
                       **{'krw_'+key: value for key,value in krw.items()},
                       **{'usd_matched_exposure_'+key: value for key,value in control.items()},
                       'base_krw_return_pct': bkrw['return_pct'], 'base_krw_mdd_pct': bkrw['mdd_pct'],
                       'return_delta_pp': krw['return_pct']-bkrw['return_pct'],
                       'mdd_delta_pp': krw['mdd_pct']-bkrw['mdd_pct'],
                       'mean_open_exposure_pct': float(exposures[ticker,name][lo:hi+1].mean()*100),
                       'exits': sum(e.get('reason') == 'exit_rule' for e in path.events)}
                per_stock.append(row)
    ps = pd.DataFrame(per_stock)
    aggregate, basket_curves, sectors, leave_out = [], {}, [], []

    def aggregate_one(group, members, name):
        usd = np.column_stack([curves[t,name] for t in members])
        opn = np.column_stack([open_nav[t,name] for t in members])
        ex = np.column_stack([exposures[t,name] for t in members])
        outputs = []
        for label, (begin, end) in protocol['periods'].items():
            um = basket_stats(usd, ex, days, begin, end, open_nav=opn)
            km = curve_stats(usd.mean(axis=1)*fx, days, begin, end)
            lo, hi = interval(days, begin, end)
            weighted_exposure = um['mean_open_exposure_pct']
            part = ps.loc[(ps.ticker.isin(members)) & (ps.strategy == name) & (ps.period == label)]
            row = {'group': group, 'stocks': len(members), 'strategy': name, 'period': label,
                   **{'usd_'+key: value for key,value in um.items() if key != 'mean_open_exposure_pct'},
                   **{'krw_'+key: value for key,value in km.items()},
                   'mean_open_exposure_pct': weighted_exposure,
                   'median_return_delta_pp': float(part.return_delta_pp.median()),
                   'return_better_count': int((part.return_delta_pp > 1e-9).sum()),
                   'mdd_better_count': int((part.mdd_delta_pp > 1e-9).sum()),
                   'both_better_count': int(((part.return_delta_pp > 1e-9) & (part.mdd_delta_pp > 1e-9)).sum())}
            for bps in (5,10):
                cost_curve = np.mean([costs[t,name][bps] for t in members], axis=0)*fx
                row[f'krw_return_{bps}bp_pct'] = curve_stats(cost_curve, days, begin, end)['return_pct']
            outputs.append(row)
        return outputs

    for group, members in groups.items():
        for name, _, n in all_specs:
            aggregate.extend(aggregate_one(group, members, name))
            basket_curves[group+'__'+name] = np.mean([curves[t,name] for t in members], axis=0)*fx
        for sector in sorted({info.at[t,'sector'] for t in members}):
            inside = [t for t in members if info.at[t,'sector'] == sector]
            outside = [t for t in members if info.at[t,'sector'] != sector]
            for name in ('buy_hold','exit_SMA126'):
                sectors.extend({'sector': sector, **r} for r in aggregate_one(group, inside, name))
                if outside:
                    leave_out.extend({'excluded_sector': sector, **r} for r in aggregate_one(group, outside, name))
    primary_curves = {ticker+'__'+name: curves[ticker,name] for ticker in tickers
                      for name in ('buy_hold','entry_only_SMA126','exit_SMA126')}
    ag = add_exposure_diagnostics(aggregate, ps, groups, primary_curves, days, protocol['periods'])
    aggregate = ag.to_dict('records')
    neighbors = []
    for n in lengths:
        row = {'length': n}
        for group in ('jan2_external','broad_external'):
            record = ag.loc[(ag.group == group) & (ag.strategy == f'exit_SMA{n}') & (ag.period == 'back')].iloc[0]
            row[group+'_return_pct'], row[group+'_mdd_pct'] = record.krw_return_pct, record.krw_mdd_pct
        neighbors.append(row)
    contributions = []
    for group, members in groups.items():
        base = np.mean([curves[t,'buy_hold'] for t in members], axis=0)
        rule = np.mean([curves[t,'exit_SMA126'] for t in members], axis=0)
        delta = {t: float((curves[t,'exit_SMA126'][-1]-curves[t,'buy_hold'][-1])/len(members)*100) for t in members}
        contributions.extend({'group': group, 'ticker': t, 'full_terminal_wealth_delta_pp': value}
                             for t,value in sorted(delta.items(), key=lambda p: -p[1]))
        assert abs(sum(delta.values())-(rule[-1]-base[-1])*100) <= 1e-8
    hashes = {str(p.relative_to(M.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((OUT/'inputs').glob('*')) if p.is_file()}
    frozen_hashes = E.G.frozen_input_check()['input_sha256']
    result = {'eligible_stocks': len(tickers), 'path_count': len(paths), 'settings_per_stock': 15,
              'primary_fixed_length': 126, 'sensitivity_lengths': list(lengths), 'groups': groups,
              'protocol_sha256': hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
              'input_sha256': hashes, 'original_frozen_input_sha256': frozen_hashes,
              'end_us': M.END, 'periods': protocol['periods'], 'validation': 'passed',
              'contributions_units': 'Full-period USD terminal wealth difference per equal initial dollar; not back-period return attribution',
              'elapsed_seconds': time.perf_counter()-started}
    (OUT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    ps.to_csv(OUT/'per_stock.csv', index=False)
    ag.to_csv(OUT/'aggregate.csv', index=False)
    pd.DataFrame(neighbors).to_csv(OUT/'neighbors.csv', index=False)
    pd.DataFrame(sectors).to_csv(OUT/'sector.csv', index=False)
    pd.DataFrame(leave_out).to_csv(OUT/'leave_one_sector_out.csv', index=False)
    pd.DataFrame(contributions).to_csv(OUT/'contributions.csv', index=False)
    pd.DataFrame(basket_curves, index=days).to_csv(OUT/'basket_curves.csv', index_label='us_session')
    pd.DataFrame(primary_curves, index=days).to_csv(OUT/'primary_stock_curves.csv', index_label='us_session')
    pd.DataFrame(trades).to_csv(OUT/'signals.csv', index=False)
    plot(basket_curves, days, groups)
    write_report(result, aggregate, per_stock, neighbors)
    print('\n'.join(comparison_table(aggregate, groups)))


if __name__ == '__main__':
    main()
