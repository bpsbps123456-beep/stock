"""Seventeen fixed FINVIZ screens, reconstructed from information available before each auction."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from urllib.parse import urlencode
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
SEVEN = HERE.parent / 'seven_sleeves_20261120'
sys.path.insert(0, str(SEVEN))
import market
from fundamentals import load_fund_pass

BASE = dict(cap='midover', fa_curratio='o1', fa_debteq='u1', geo='usa', sh_avgvol='o500',
            sh_instown='o40', sh_price='o10', sh_relvol='o1', ta_rsi='nos40', ta_sma20='sa50',
            ta_sma200='pa', ta_sma50='pa')
RV = '-relativevolume'
MONTH = '-perf4w'
QUARTER = '-perf13w'
# These deltas, including the sorting key and exact thresholds, were frozen before viewing returns.
DESIGNS = {
    'original': ('원본 검색식', '원본 12개 조건, 상대거래량 내림차순', {}, RV),
    'vol150': ('상대거래량 1.5배', '상대거래량 > 1.5', {'sh_relvol': 'o1.5'}, RV),
    'vol200': ('상대거래량 2배', '상대거래량 > 2', {'sh_relvol': 'o2'}, RV),
    'rsi50': ('RSI 50 초과', 'RSI(14) > 50', {'ta_rsi': 'nos50'}, RV),
    'liquid': ('하루 평균 100만 주', '3개월 평균 거래량 > 100만 주', {'sh_avgvol': 'o1000'}, RV),
    'balance': ('높은 유동비율·낮은 부채', '유동비율 > 1.5, 부채/자본 < 0.5', {'fa_curratio': 'o1.5', 'fa_debteq': 'u0.5'}, RV),
    'ownership60': ('기관 지분 60% 초과', '기관 지분 > 60%', {'sh_instown': 'o60'}, RV),
    'growth10': ('분기 매출 10% 성장', '최근 분기 매출 전년 동기 대비 > 10%', {'fa_salesqoq': 'o10'}, RV),
    'growth25': ('분기 매출 25% 성장', '최근 분기 매출 전년 동기 대비 > 25%', {'fa_salesqoq': 'o25'}, RV),
    'rank_month': ('1개월 수익률 순', '원본 조건을 통과한 종목 중 21거래일 수익률 내림차순', {}, MONTH),
    'rank_quarter': ('3개월 수익률 순', '원본 조건을 통과한 종목 중 63거래일 수익률 내림차순', {}, QUARTER),
    'month_momentum': ('1개월 10% 상승', '상대거래량 조건 해제, 21거래일 수익률 > 10%, 같은 수익률 내림차순', {'sh_relvol': None, 'ta_perf': '4w10o'}, MONTH),
    'quarter_momentum': ('3개월 10% 상승', '상대거래량 조건 해제, 63거래일 수익률 > 10%, 같은 수익률 내림차순', {'sh_relvol': None, 'ta_perf': '13w10o'}, QUARTER),
    'pullback': ('상승 추세의 RSI 눌림', 'RSI(14) < 50, 상대거래량 조건 해제, 63거래일 수익률 내림차순', {'sh_relvol': None, 'ta_rsi': 'nob50'}, QUARTER),
    'near_high': ('52주 고가 5% 이내', '종가가 최근 52주 장중 최고가의 95% 이상', {'ta_highlow52w': 'b0to5h'}, RV),
    'relaxed_volume': ('상대거래량 문턱 해제', '상대거래량 하한 해제, 상대거래량 내림차순', {'sh_relvol': None}, RV),
    'quiet_growth': ('거래량 문턱 없는 매출 성장', '상대거래량 하한 해제, 분기 매출 전년 동기 대비 > 10%, 63거래일 수익률 내림차순', {'sh_relvol': None, 'fa_salesqoq': 'o10'}, QUARTER),
}


def metadata():
    out = {}
    for key, (name, changes, modifications, order) in DESIGNS.items():
        filters = [f'{k}_{v}' for k, v in {**BASE, **modifications}.items() if v is not None]
        url = 'https://finviz.com/screener?' + urlencode(dict(v=111, f=','.join(filters), ft=4, ta=0, o=order))
        out[key] = dict(name=name, changes=changes, url=url, order=order, filters=filters,
                        historical_proxy='실제 FINVIZ의 설정 링크. 과거 순위는 당시 대형·중형 표본, SEC 공시·13F, 정규장 종가로 재구성.')
    return out


def historic_highs(symbols):
    """Fetch true SIP daily highs for the names whose previous SIP freeze contains close/open only."""
    path = market.PRIOR.parent / 'hatta_bar/code/fetch_alpaca.py'
    spec = importlib.util.spec_from_file_location('past_sip_bars', path)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    auth = api.keys()
    params = dict(symbols=','.join(sorted(symbols)), start='2025-01-01', end='2026-11-17T23:59:59Z',
                  timeframe='1Day', adjustment='raw', asof='-', feed='sip', limit=10000)
    values = {s: {} for s in symbols}
    while True:
        response = requests.get(api.URL, params=params, headers=auth, timeout=50)
        response.raise_for_status()
        data = response.json()
        for s, rows in data['bars'].items():
            for r in rows:
                day = pd.Timestamp(r['t']).tz_convert('America/New_York').tz_localize(None).normalize()
                values[s][day] = r['h']
        if not data.get('next_page_token'):
            break
        params['page_token'] = data['next_page_token']
    return {s: pd.Series(v, dtype=float) for s, v in values.items()}


def technical_features(src):
    dates = pd.DatetimeIndex(src.signals)
    last_independent = {s: dates[dates < pd.Timestamp(terms[0])][-1].date().isoformat()
                        for s, terms in market.mid.DEALS.items()}
    fields = ['price', 'ma', 'rsi', 'avg', 'rv', 'p21', 'p63', 'high_frac']
    panel = {s: np.full((len(dates), len(src.symbols)), np.nan) for s in fields}
    with patch.dict(market.tech.SIP_END, last_independent):
        highs = historic_highs(market.tech.SIP_END)
        for i, symbol in enumerate(src.symbols):
            market.tech.fetch(symbol)
            b = market.tech.read_bars(symbol)
            c, v = b.close, b.volume
            # SIP's current split-adjusted history can include splits AFTER the old freeze. Use raw highs
            # divided by the original freeze's own price multiplier to keep both series in the same units.
            high = highs[symbol].reindex(c.index)/b.split_mult if symbol in highs else b.high
            if high.loc[:dates[-1]].isna().any():
                raise ValueError(f'Missing high in an otherwise valid bar: {symbol}')
            if (high.loc[:dates[-1]] + .0001 < c.loc[:dates[-1]]).any():
                raise ValueError(f'Source high below close for {symbol}')
            avg = v.shift(1).rolling(63).mean()
            sma20, sma50, sma200 = (c.rolling(n).mean() for n in [20, 50, 200])
            data = {'price': c*b.split_mult, 'ma': (sma20>sma50) & (c>sma50) & (c>sma200),
                    'rsi': market.tech.rsi(c), 'avg': avg/b.split_mult, 'rv': v/avg.where(avg>0),
                    'p21': c.pct_change(21, fill_method=None), 'p63': c.pct_change(63, fill_method=None),
                    # 52 calendar weeks ending at this signal's completed session, inclusive of its own high.
                    'high_frac': c/high.rolling('364D').max()}
            for key, series in data.items():
                panel[key][:, i] = series.astype(float).reindex(dates).to_numpy()
    return panel


def dated_statements(table, dates, column):
    rows = table.sort_values(['filed', 'period', 'accession']).drop_duplicates('filed', keep='last')
    rows.index = pd.to_datetime(rows.filed) + pd.Timedelta(days=1)
    value = rows[column].reindex(dates, method='ffill')
    return value.to_numpy()


def fundamentals(src):
    """Both cohorts retain their own financial/ownership proxy on the original roster date."""
    dates = pd.DatetimeIndex(src.signals)
    classes = pd.read_csv(SEVEN / 'classifications.csv', dtype=str).fillna('')
    monthly = dict(tuple(classes[classes.pool.eq('large')].groupby('month')))
    changing = {t: s.set_index('month').cik.to_dict() for t, s in classes[classes.pool.eq('large')].groupby('raw')
                if s.cik.nunique() > 1}
    signal_month = [max(s[:7], '2026-01') for s in src.signals]
    out = {key: np.full(src.membership.shape, np.nan) for key in ['quality', 'tight', 'own']}
    for pool, path in [('large', market.PRIOR), ('mid', market.mid.OUT)]:
        price = market.frozen_field(path / 'technical_inputs', 'price')
        quality = load_fund_pass(price, midcaps=pool=='mid')
        ratios = pd.read_csv(path / 'fundamental_inputs/as_filed.csv')
        strict = pd.DataFrame(False, index=src.signals, columns=price.columns)
        for (s, cik), group in ratios.groupby(['ticker', 'cik']):
            if s not in strict:
                continue
            group = group.assign(tight=(group.current_ratio.gt(1.5) & group.debt_equity.ge(0) & group.debt_equity.lt(.5)))
            matched = pd.Series(dated_statements(group, dates, 'tight')).eq(True).to_numpy()
            if s in changing and pool=='large':
                matched = matched & np.array([changing[s].get(month) == str(cik) for month in signal_month])
            strict[s] |= matched
        values = dict(quality=quality, tight=strict, own=market.frozen_field(path, 'ownership_proxy'))
        for month, big in monthly.items():
            cohort = big if pool=='large' else classes[classes.pool.eq('mid') & ~classes.symbol.isin(big.symbol)]
            rows = np.flatnonzero([day[:7] == month for day in src.days])
            cols = [src.loc[s] for s in cohort.symbol]
            for field, frame in values.items():
                out[field][np.ix_(rows, cols)] = frame.loc[np.array(src.signals)[rows], cohort.raw].to_numpy(float)
    return out


def quarterly_sales(src):
    """Latest comparable year-on-year quarterly revenue, from the as-filed earnings archive."""
    dates = pd.DatetimeIndex(src.signals)
    filings = pd.read_csv(SEVEN / 'earnings_filings.csv', usecols=['cik', 'filed', 'period', 'sales_yoy'])
    assert not filings.duplicated(['cik', 'filed']).any()
    # The date of information is the signal close. At month boundaries an effective new name/CIK
    # in the execution roster has no access to a prior registrant's income statement.
    growth = np.full(src.membership.shape, np.nan)
    by_cik = {}
    for cik, docs in filings.groupby('cik'):
        docs = docs.sort_values('filed')
        docs.index = pd.to_datetime(docs.filed) + pd.Timedelta(days=1)
        visible = docs.reindex(dates, method='ffill')
        fresh = (dates - pd.to_datetime(visible.period)).dt.days.between(0, 180)
        by_cik[str(cik)] = visible.sales_yoy.where(fresh).to_numpy()
    # Identity matrix includes all share classes and actual monthly changes of registrant.
    for i, symbol in enumerate(src.symbols):
        for cik in np.unique(src.issuers[:, i]):
            if cik in by_cik:
                use = src.issuers[:, i] == cik
                growth[use, i] = by_cik[cik][use]
    return growth


def freeze(src):
    tech = technical_features(src)
    funds = fundamentals(src)
    group = (src.membership & (funds['quality'] == 1) & (funds['own'] > .4) & (tech['ma'] == 1)
             & (tech['price'] > 10) & (tech['avg'] > 500_000))
    row, col = np.nonzero(group)
    # Sparse issuer/signal rows contain every security satisfying the gates shared by all 17 recipes.
    data = {key: tech[key][row, col] for key in ['avg', 'rv', 'rsi', 'p21', 'p63', 'high_frac']}
    data.update(tight=funds['tight'][row, col].astype(bool), own=funds['own'][row, col],
                sales=quarterly_sales(src)[row, col], row=row.astype(np.int16), col=col.astype(np.int16),
                days=np.array(src.days), signals=np.array(src.signals), symbols=np.array(src.symbols))
    _rank(src, data)  # Reject any disagreement before replacing the previously verified snapshot.
    fields = {key: data[key] for key in ['avg', 'rv', 'rsi', 'p21', 'p63', 'high_frac', 'tight', 'own', 'sales']}
    table = pd.DataFrame(dict(trade_open=data['days'][row], signal_close=data['signals'][row],
                              symbol=data['symbols'][col], **fields))
    for part, block in [('h1', table[table.trade_open <= '2026-06-30']), ('h2', table[table.trade_open > '2026-06-30'])]:
        target = HERE / f'recipe_inputs_{part}.csv'
        block.to_csv(target, index=False, float_format='%.17g')
        assert target.stat().st_size < 10_000_000


def _rank(src, data):
    row, col = data['row'], data['col']
    n = len(src.days)
    for key in ['days', 'signals', 'symbols']:
        np.testing.assert_array_equal(data[key], getattr(src, key))
    rv, month, quarter = data['rv'], data['p21'], data['p63']
    conditions = {'sh_relvol': {'o1': rv>1, 'o1.5': rv>1.5, 'o2': rv>2},
                  'ta_rsi': {'nos40': data['rsi']>40, 'nos50': data['rsi']>50, 'nob50': data['rsi']<50},
                  'sh_avgvol': {'o1000': data['avg']>1_000_000}, 'fa_curratio': {'o1.5': data['tight']},
                  'sh_instown': {'o60': data['own']>.6}, 'fa_salesqoq': {'o10': data['sales']>.10, 'o25': data['sales']>.25},
                  'ta_perf': {'4w10o': month>.10, '13w10o': quarter>.10},
                  'ta_highlow52w': {'b0to5h': data['high_frac']>=.95}}
    output = {}
    for key, (_, _, modifications, order) in DESIGNS.items():
        codes = {**BASE, **modifications}
        scores = {RV: rv, MONTH: month, QUARTER: quarter}[order]
        keep = np.isfinite(scores)
        for field, options in conditions.items():
            token = codes.get(field)
            if token in options:
                keep &= options[token]
        r, c = row[keep], col[keep]
        order = np.lexsort((c, -scores[keep], r))
        cuts = np.cumsum(np.bincount(r, minlength=n))[:-1]
        output[key] = [block.tolist() for block in np.split(c[order], cuts)]
    for j, (actual, expected) in enumerate(zip(output['original'], src.ranks['F'])):
        if actual != expected:
            missing = [src.symbols[i] for i in expected if i not in actual]
            extra = [src.symbols[i] for i in actual if i not in expected]
            raise AssertionError(f'Reference ranking mismatch {src.days[j]}: missing={missing}, extra={extra}')
    return output


def recipes(src):
    files = [HERE / f'recipe_inputs_{part}.csv' for part in ['h1', 'h2']]
    tab = pd.concat([pd.read_csv(path, float_precision='round_trip') for path in files], ignore_index=True)
    row = pd.Index(src.days).get_indexer(tab.trade_open)
    col = pd.Index(src.symbols).get_indexer(tab.symbol)
    assert (row >= 0).all() and (col >= 0).all() and not tab.duplicated(['trade_open', 'symbol']).any()
    np.testing.assert_array_equal(np.array(src.signals)[row], tab.signal_close)
    data = dict(row=row, col=col, days=src.days, symbols=src.symbols, signals=src.signals)
    data.update({k: tab[k].to_numpy() for k in ['avg', 'rv', 'rsi', 'p21', 'p63', 'high_frac', 'tight', 'own', 'sales']})
    return _rank(src, data)


if __name__ == '__main__':
    source = market.load_inputs()
    if '--refresh' in sys.argv:
        freeze(source)
    ranks = recipes(source)
    for name, table in ranks.items():
        print(name, min(map(len, table)), max(map(len, table)))
