"""Dated market, issuer, and price-only/Finviz rankings for the seven-slot experiment."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'finviz_rotation_20261119'
DESK = HERE.parent / 'desk_defense_20261119'
sys.path[:0] = [str(PRIOR), str(DESK)]
import add_midcaps as mid
import technicals as tech
from universe import API, HEADERS
from inputs import load_book
from desk import strategies as desk
from fundamentals import load_fund_pass, ticker_map

ALIASES = mid.ALIASES
END = '2026-11-18'
SECTORS = {'Communication Services', 'Consumer Discretionary', 'Consumer Staples', 'Energy',
           'Financials', 'Health Care', 'Industrials', 'Information Technology', 'Materials',
           'Real Estate', 'Utilities'}


def canon(symbol):
    return ALIASES.get(symbol, symbol)


def frozen_field(path, field):
    return pd.read_csv(path / f'{field}.csv', index_col=0)


def combine(big, small):
    full = pd.concat([big, small], axis=1).rename(columns=canon)
    return full.loc[:, ~full.columns.duplicated()].sort_index(axis=1)


def freeze_strength():
    """Use 200 actual company closes, including all pre-January warmup observations."""
    reference = combine(frozen_field(tech.FROZEN, 'ma_pass'), frozen_field(mid.FROZEN, 'ma_pass')).eq(1)
    dates = pd.DatetimeIndex(reference.index)
    ends = {s: dates[dates < pd.Timestamp(terms[0])][-1].date().isoformat() for s, terms in mid.DEALS.items()}
    columns = {}
    with patch.dict(tech.SIP_END, ends):
        for symbol in reference:
            tech.fetch(symbol)
            bars = tech.read_bars(symbol)
            if bars is None:
                raise ValueError(f'Historical observations missing for {symbol}')
            close = bars.close
            sma20, sma50, sma200 = [close.rolling(n).mean() for n in (20, 50, 200)]
            passed = (sma20 > sma50) & (close > sma50) & (close > sma200)
            columns[symbol] = (close / sma200 - 1).where(passed).reindex(dates).to_numpy()
    score = pd.DataFrame(columns, index=reference.index)
    np.testing.assert_array_equal(score.notna(), reference)
    score.to_csv(HERE / 'ma_strength.csv', index_label='signal_close', float_format='%.14g')


def revision_sectors(revision):
    params = {'action': 'query', 'prop': 'revisions', 'revids': revision, 'rvslots': 'main',
              'rvprop': 'ids|content', 'formatversion': 2, 'format': 'json'}
    response = requests.get(API, params=params, headers=HEADERS, timeout=45)
    response.raise_for_status()
    rev = response.json()['query']['pages'][0]['revisions'][0]
    assert rev['revid'] == revision
    table = rev['slots']['main']['content'].split('id="constituents"', 1)[1].split('\n|}', 1)[0]
    values = {}
    for row in re.split(r'\n\|-+[^\n]*\n', table)[2:]:
        symbol = re.search(r'\{\{(?:nysesymbol|nasdaqsymbol|bzx link)\s*\|\s*([^|}]+)', row, re.I)
        fields = [cell.strip().strip('|').strip() for cell in re.split(r'\|\||\n\|', row)]
        sectors = [cell for cell in fields if cell in SECTORS]
        if symbol is None or len(sectors) != 1:
            raise ValueError(f'Missing GICS sector in revision {revision}: {row[:120]}')
        values[symbol.group(1).strip().replace('.', '-')] = sectors[0]
    return values


def time_sector(symbol, industry):
    """Broad-sector proxy for out-of-index TIME holdings, using the desk's industry/business labels."""
    if industry in {'AI 반도체', '메모리', '반도체 소재', '반도체 장비', '반도체 장비(적자)', '반도체(적자)',
                    '엔터프라이즈 SW', '네트워크/광통신', '양자컴퓨팅'}:
        return 'Information Technology'
    if industry in {'헬스케어', '헬스케어 AI'}:
        return 'Health Care'
    # For broader desk groups distinguish the business. Ambiguous or unmapped issuers stay unknown.
    # 2025-10-31 Fidelity MSCI Industrials holdings identify BE, EOSE, FLNC and SMR:
    # https://www.sec.gov/Archives/edgar/data/945908/000141036825040132/NPORT_PBR5_40562781_1025.htm
    business = {'Industrials': {'PL', 'RKLB', 'SPCX', 'BE', 'EOSE', 'FLNC', 'SMR', 'SERV'},
                'Information Technology': {'QNT', 'NBIS'}, 'Energy': {'CCJ', 'LEU', 'UEC', 'UUUU'},
                'Communication Services': {'ASTS', 'BIDU'}, 'Consumer Discretionary': {'BABA', 'MELI', 'PDD'},
                'Financials': {'CRCL'}}
    return next((sector for sector, names in business.items() if symbol in names), '')


def freeze_classifications():
    sources = json.loads((PRIOR / 'broad_sources.json').read_text(encoding='utf-8'))
    revisions = [mid.REVISION, *(s['revision'] for s in sources)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        sector_map = dict(zip(revisions, pool.map(revision_sectors, revisions)))
    # Dec 5 S&P DJI press release "CRH, Carvana and Comfort Systems USA Set to Join S&P 500".
    sector_map[sources[0]['revision']].update(CRH='Materials', CVNA='Consumer Discretionary', FIX='Industrials')
    rows = []
    broad = pd.read_csv(PRIOR / 'broad_membership.csv')
    for src in sources:
        for r in broad[broad.month.eq(src['month'])].itertuples():
            rows.append((src['month'], 'large', r.symbol, canon(r.symbol), str(r.cik),
                         sector_map[src['revision']][r.symbol], 'index'))
    mids = pd.read_csv(mid.OUT / 'membership.csv')
    ident = {ticker: str(cik) for ticker, cik in ticker_map().items()}
    # A single known identity is used only to merge economically identical shares.
    # XOM's change in registrant is covered by the actual month rows above.
    for root in (mid.OUT, PRIOR):
        filers = pd.read_csv(root / 'fundamental_inputs/as_filed.csv', usecols=['ticker', 'cik'])
        for sym, group in filers.groupby('ticker'):
            if group.cik.nunique() == 1:
                ident[sym] = str(group.cik.iat[0])
    for r in mids.itertuples():
        cik = str(int(r.cik)) if pd.notna(r.cik) else ident.get(r.symbol, ident.get(canon(r.symbol), ''))
        rows.append(('*', 'mid', r.symbol, canon(r.symbol), cik, sector_map[mid.REVISION][r.symbol], 'index'))
    book = load_book()
    for key in book.keys:
        s = book.ticker[key]
        rows.append(('*', 'TIME', s, canon(s), ident.get(s, ''), time_sector(s, book.sector[key]), 'desk_proxy'))
    pd.DataFrame(rows, columns=['month', 'pool', 'raw', 'symbol', 'cik', 'sector', 'source']).to_csv(
        HERE / 'classifications.csv', index=False)


def screen_score(panels, path, supplement):
    financial = load_fund_pass(panels['price'], midcaps=supplement)
    own = pd.read_csv(path / 'ownership_proxy.csv', index_col=0)
    return panels['relvol'].where(panels['technical_pass'] & financial.eq(True) & own.gt(.4))


def locations(values):
    valid = np.flatnonzero(np.isfinite(values))
    # Symbol indices are alphabetic; stable sort resolves exact ties in ticker order.
    return valid[np.argsort(-values[valid], kind='stable')].tolist()


def freeze_finviz():
    large = tech.load_panels()
    small = {field: frozen_field(mid.FROZEN, field) for field in ('price', 'technical_pass', 'relvol')}
    small['technical_pass'] = small['technical_pass'].eq(1)
    ranks = [screen_score(large, PRIOR, False), screen_score(small, mid.OUT, True)]
    dates = list(large['close'].index)
    days = [*dates[1:], END]
    classes = pd.read_csv(HERE / 'classifications.csv', dtype=str).fillna('')
    old_log = pd.read_csv(mid.OUT / 'decisions.csv').query("frequency == 'daily' and filter == 'ownership'").set_index('trade_open')
    symbols = list(combine(large['close'], small['price']).columns)
    loc = {s: i for i, s in enumerate(symbols)}
    full = np.full((len(days), len(symbols)), np.nan)
    for month, group in classes[classes.pool.eq('large')].groupby('month'):
        mid_by = classes[classes.pool.eq('mid') & ~classes.symbol.isin(group.symbol)]
        period = np.array([day[:7] == month for day in days])
        for comp, selection in zip(ranks, [group, mid_by]):
            full[np.ix_(period, [loc[s] for s in selection.symbol])] = comp.loc[np.array(dates)[period], selection.raw].to_numpy()
    for i, day in enumerate(days[:-1]):
        observed = old_log.loc[day]
        ix = locations(full[i])
        assert len(ix) == observed.large_eligible + observed.additional_mid_eligible, day
        expected = set(str(observed.selected_before_terminations).split('|')) if pd.notna(observed.selected_before_terminations) else set()
        assert {symbols[x] for x in ix[:7]} == expected, day
    pd.DataFrame(full, index=days, columns=symbols).to_csv(HERE / 'finviz_scores.csv', index_label='trade_open', float_format='%.15g')


def universe_arrays(days, symbols):
    classes = pd.read_csv(HERE / 'classifications.csv', dtype=str).fillna('')
    loc = {s: i for i, s in enumerate(symbols)}
    mids = classes[classes.pool.eq('mid')]
    previous = [(f'Unknown/{s}', f'Ticker/{s}') for s in symbols]
    for row in classes[classes.pool.eq('TIME')].itertuples():
        previous[loc[row.symbol]] = (row.sector or f'Unknown/{row.symbol}', row.cik or f'Ticker/{row.symbol}')
    months = {}
    for month, big in classes[classes.pool.eq('large')].groupby('month'):
        member = np.zeros(len(symbols), bool)
        raw = list(symbols)
        source = pd.concat([mids, big]).drop_duplicates('symbol', keep='last')
        for row in source.itertuples():
            col = loc[row.symbol]
            member[col] = True
            raw[col] = row.raw
            previous[col] = (row.sector, row.cik or f'Ticker/{row.symbol}')
        sector, issuer = zip(*previous)
        months[month] = (member, raw, sector, issuer)
    return tuple(np.asarray([months[d[:7]][i] for d in days]) for i in range(4))


def load_inputs():
    """Return execution-session arrays; every score already incorporates the previous close lag."""
    book = load_book()
    ref = desk.run_model(book)
    days = [*ref.days, END]
    signals = [tech.FIRST, *ref.days]
    symbols = list(frozen_field(HERE, 'ma_strength').columns)
    loc = {s: i for i, s in enumerate(symbols)}
    tail = pd.read_csv(HERE / 'tail_prices.csv').set_index('symbol')
    opn, close = [], []
    for field, original, output in [('open', book.opn, opn), ('close', book.cls, close)]:
        joined = combine(frozen_field(tech.FROZEN, field), frozen_field(mid.FROZEN, field))
        np.testing.assert_array_equal(joined.columns, symbols)
        # The original operational series is the execution source for its shared names.
        old = original.rename(columns=canon).reindex(index=ref.days, columns=symbols)
        panel = old.combine_first(joined.loc[ref.days]).reindex(columns=symbols)
        output.extend(panel.to_numpy())
        output.append(tail[field].reindex(symbols).to_numpy())
    opn, close = np.array(opn), np.array(close)
    allowed = combine(frozen_field(tech.FROZEN, 'tradable'), frozen_field(mid.FROZEN, 'tradable')).eq(1)
    tradable = np.concatenate([allowed.loc[ref.days].to_numpy(), allowed.iloc[[-1]].to_numpy()])
    if np.any(tradable[-1] & ~(np.isfinite(opn[-1]) & np.isfinite(close[-1]))):
        raise ValueError(f'Uncovered active final-session quotes: {np.array(symbols)[tradable[-1] & ~np.isfinite(close[-1])]}')
    settlements = np.zeros_like(tradable)
    for s, terms in {**tech.MERGERS, **mid.DEALS}.items():
        col, row = loc[canon(s)], days.index(terms[0])
        assert not tradable[row:, col].any() and np.isfinite(opn[row, col])
        settlements[row, col] = True
    membership, raw, sectors, issuers = universe_arrays(days, symbols)
    for pair in [('GOOG', 'GOOGL'), ('NWS', 'NWSA'), ('FOXA', 'FOX'), ('UA', 'UAA')]:
        np.testing.assert_array_equal(issuers[:, loc[pair[0]]], issuers[:, loc[pair[1]]])
    strength = frozen_field(HERE, 'ma_strength').loc[signals, symbols].to_numpy()
    f = frozen_field(HERE, 'finviz_scores').loc[days, symbols].to_numpy()
    assert not np.isfinite(f[~membership]).any()
    ranks = {'M': [locations(row) for row in np.where(membership, strength, np.nan)],
             'F': [locations(row) for row in f], 'D': []}
    selection = []
    for day in days:
        if day in ref.rebalances:
            pdf, ordered = desk.momentum_ranking(book, book.days.index(day), day)
            assert pdf < day
            operational = set(ref.rebalances[day])
            selection = [loc[canon(r['ticker'])] for r in ordered if r['key'] in operational]
            assert len(selection) == len(operational) <= 7
        ranks['D'].append(selection.copy())
    fx_before = book.fx[book.days.index(tech.FIRST)]
    fx = np.r_[book.fx[-len(ref.days):], tail.at['KRW=X', 'close']] / fx_before
    return SimpleNamespace(days=days, signals=signals, symbols=symbols, loc=loc, aliases=ALIASES,
                           opn=opn, close=close, fx=fx, tradable=tradable, settlements=settlements,
                           membership=membership, raw_symbols=raw, sectors=sectors, issuers=issuers,
                           sector_known=~np.char.startswith(sectors, 'Unknown/'), ma_score=strength,
                           ranks=ranks, book=book, desk_path=ref)


load_market = load_inputs


def audit():
    data = load_inputs()
    ref, book = data.desk_path, data.book
    units, cash = {}, 1.
    for j, day in enumerate(ref.days):
        nav = cash + sum(q * data.opn[j, i] for i, q in units.items())
        if day in ref.rebalances:
            weights = {data.loc[canon(book.ticker[k])]: w for k, w in ref.rebalances[day].items()}
            units = {i: w * nav / data.opn[j, i] for i, w in weights.items()}
            cash = nav * (1 - sum(weights.values()))
        actual = cash + sum(q * data.close[j, i] for i, q in units.items())
        np.testing.assert_allclose(actual, ref.close[j], rtol=1e-12, atol=1e-12)
    for k in ['D', 'M', 'F']:
        print(k, len(data.ranks[k]), min(map(len, data.ranks[k])), max(map(len, data.ranks[k])))
    print('symbols, sessions, settlement opens:', len(data.symbols), len(data.days), data.settlements.sum())
    unknown = {data.symbols[i] for j, row in enumerate(data.ranks['D']) for i in row if not data.sector_known[j, i]}
    print('unknown sectors among operational D ranks:', sorted(unknown))


if __name__ == '__main__':
    if '--refresh' in sys.argv:
        freeze_strength()
        freeze_classifications()
        freeze_finviz()
    audit()
