"""Value a November 18 closing portfolio through November 19, with no new decisions.

SIP daily prices are raw as traded. Two adjacent split-adjustment ratios express
the November 19 raw close in November 18 share units. Ordinary dividends are
excluded, consistent with the main price-return study. USD cash earns zero.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'seven_sleeves_20261120'
FILE = HERE / 'extension_quotes.csv'
PREVIOUS, NEXT = '2026-11-18', '2026-11-19'


def sip_api():
    spec = importlib.util.spec_from_file_location('sip_client', HERE.parent / 'hatta_bar/code/fetch_alpaca.py')
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api


def fetch_sip(session, names):
    api = sip_api()
    headers = api.keys()
    rows = {}
    for k in range(0, len(names), 95):
        batch = {s.replace('-', '.'): s for s in names[k:k+95]}
        fields = dict(symbols=','.join(batch), start=PREVIOUS + 'T05:00:00Z', end=NEXT + 'T23:59:59Z',
                      timeframe='1Day', asof='-', feed='sip', limit=10000)
        both = {}
        for adjusted in ('raw', 'split'):
            res = session.get(api.URL, params={**fields, 'adjustment': adjusted}, headers=headers, timeout=30)
            res.raise_for_status()
            payload = res.json()
            assert not payload.get('next_page_token')
            both[adjusted] = {s: {r['t'][:10]: r for r in history}
                              for s, history in payload['bars'].items()}
        for vendor, symbol in batch.items():
            if vendor not in both['raw']:
                continue
            raw, split = both['raw'][vendor], both['split'][vendor]
            if {PREVIOUS, NEXT} - set(raw):
                continue
            # Split-adjusted close is rounded to cents, distorted for SNDK whose
            # 8:1 split begins trading Nov20. Its adjusted volume is exactly 8x
            # raw on BOTH dates. Share volume preserves that exact ratio.
            ratio = (split[PREVIOUS]['v'] / raw[PREVIOUS]['v']) / (split[NEXT]['v'] / raw[NEXT]['v'])
            price_ratio = (raw[PREVIOUS]['c'] / split[PREVIOUS]['c']) / (raw[NEXT]['c'] / split[NEXT]['c'])
            np.testing.assert_allclose(ratio, price_ratio, rtol=.0005)
            rows[symbol] = dict(source='Alpaca SIP', prior_open=raw[PREVIOUS]['o'],
                                prior_close=raw[PREVIOUS]['c'], prior_volume=raw[PREVIOUS]['v'],
                                next_open=raw[NEXT]['o'], next_close=raw[NEXT]['c'],
                                next_volume=raw[NEXT]['v'], shares_per_prior_share=ratio)
    return rows


def fetch_yahoo(session, symbol):
    start = int(pd.Timestamp(PREVIOUS, tz='UTC').timestamp()) - 86400
    end = int(pd.Timestamp('2026-11-20', tz='UTC').timestamp())
    response = session.get('https://query1.finance.yahoo.com/v8/finance/chart/' + symbol,
                           params=dict(period1=start, period2=end, interval='1d', events='splits'),
                           headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
    response.raise_for_status()
    data = response.json()['chart']['result'][0]
    tz = data['meta']['exchangeTimezoneName']
    days = pd.to_datetime(data['timestamp'], unit='s', utc=True).tz_convert(tz).strftime('%Y-%m-%d')
    bars = pd.DataFrame(data['indicators']['quote'][0], index=days)
    ratio = 1.
    for event in data.get('events', {}).get('splits', {}).values():
        if pd.Timestamp(event['date'], unit='s', tz='UTC').tz_convert(tz).strftime('%Y-%m-%d') == NEXT:
            ratio *= event['numerator'] / event['denominator']
    before, after = bars.loc[PREVIOUS], bars.loc[NEXT]
    # Both Yahoo historical bars use the latest share units in the downloaded range.
    return dict(source='Yahoo chart', prior_open=before.open * ratio, prior_close=before.close * ratio,
                prior_volume=before.volume / ratio, next_open=after.open, next_close=after.close,
                next_volume=after.volume, shares_per_prior_share=ratio)


def freeze():
    before = pd.read_csv(PRIOR / 'tail_prices.csv', index_col='symbol')
    names = sorted(set(before.index) - {'KRW=X'})
    with requests.Session() as session:
        session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=.4,
                                                    status_forcelist=(429, 500, 502, 503, 504))))
        rows = fetch_sip(session, names)
        for s in sorted(set(before.index) - set(rows)):
            rows[s] = fetch_yahoo(session, s)
    table = pd.DataFrame.from_dict(rows, orient='index').sort_index()
    table['frozen_prior_close'] = before.close
    table['next_close_prior_units'] = table.next_close * table.shares_per_prior_share
    table['close_to_close_factor'] = table.next_close_prior_units / table.frozen_prior_close
    assert set(before.index) == set(table.index) and np.isfinite(table.drop(columns='source')).all().all()
    assert (table[['prior_open', 'prior_close', 'next_open', 'next_close', 'shares_per_prior_share']] > 0).all().all()
    assert table.loc[table.index != 'KRW=X', ['prior_volume', 'next_volume']].gt(0).all().all()
    overlap = (table.prior_close / table.frozen_prior_close - 1).abs()
    if overlap.max() > .002:
        raise ValueError(f'Investigate previous-session source discrepancy: {overlap.nlargest().to_dict()}')
    table.to_csv(FILE, index_label='symbol', float_format='%.15g')
    print('source counts', table.source.value_counts().to_dict(), 'maximum prior quote gap', overlap.max(),
          'new split events', table.shares_per_prior_share[~np.isclose(table.shares_per_prior_share, 1)].to_dict())


def get_forward(src):
    """Return Nov19 close in prior units, FX normalized to main NAV base, and coverage metadata."""
    if src.days[-1] != PREVIOUS:
        raise ValueError('The forward mark requires the original November 18 endpoint')
    table = pd.read_csv(FILE, index_col='symbol')
    assert table.index.is_unique
    shares = table.reindex(src.symbols)
    quotes = shares.next_close_prior_units.to_numpy()
    live = np.asarray(src.tradable[-1], bool)
    np.testing.assert_array_equal(live, np.isfinite(quotes))
    np.testing.assert_allclose(shares.frozen_prior_close.to_numpy()[live], src.close[-1, live], rtol=1e-12)
    fx18, fx19 = table.loc['KRW=X', ['frozen_prior_close', 'next_close']]
    summary = dict(prior=PREVIOUS, next=NEXT, priced=int(live.sum()), previous_settlements=int((~live).sum()),
                   usdkrw_prior=float(fx18), usdkrw_next=float(fx19), fx_ratio=float(fx19 / fx18),
                   split_events=shares.index[live & ~np.isclose(shares.shares_per_prior_share, 1)].tolist(),
                   max_prior_source_gap=float((shares.prior_close / shares.frozen_prior_close - 1).abs().max()))
    return quotes, float(src.fx[-1] * fx19 / fx18), summary


def hold_return(weights, src, forward):
    """One-day KRW return of November 18 closing stock weights and residual USD cash."""
    quotes, fx, _ = forward
    w = np.asarray(weights, dtype=float)
    if w.shape != quotes.shape or not np.isfinite(w).all() or w.min() < 0 or w.sum() > 1+1e-9:
        raise ValueError('Supply nonnegative November 18 closing weights in src.symbols order')
    held = w > 0
    if np.any(held & ~np.isfinite(quotes)):
        raise ValueError('Missing November 19 quote for held stock')
    usd_return = w[held] @ (quotes[held] / src.close[-1, held] - 1)
    return (1 + usd_return) * fx / src.fx[-1] - 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    if parser.parse_args().refresh:
        freeze()
    import sys
    sys.path.insert(0, str(PRIOR))
    sys.path.insert(0, str(HERE.parents[3]))
    from market import load_inputs
    source = load_inputs()
    forward = get_forward(source)
    print(forward[2])
