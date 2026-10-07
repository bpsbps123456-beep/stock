"""Auditable, bounded input bundle for historical ETF strategy research.

The bundle never downloads holdings or prices before/after the observed holdings
sample. Yahoo ``Close`` (``auto_adjust=False``) is split adjusted, but excludes
cash distributions. Holdings disclosures were backfilled: this is a historical
disclosure simulation, not evidence that a live collector had them at the time.

``prepare_bundle(out_dir)`` returns ``{'arrays': ..., 'metadata': ...}`` and
atomically writes bundle.npz, metadata.json and raw_prices.json. Array axes:
T = union local close dates, S = common disclosure dates, N = company universe,
fund axis = time, koact; lookbacks = 1,3,5,10,20,40 KRX sessions. Weights/cash
are NAV fractions; flow_pp is percentage POINTS, flow_kind is -2,-1,0,1,2.
Unavailable full windows/common factors are NaN, not artificial zero flows.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping
import warnings

import numpy as np

from .flows import quantity_flows
from .market import canonical_code, company_key, is_cash_holding, listing_market, yahoo_ticker
from .storage import NORMALIZED_HOLDINGS_DIR, atomic_write_json, load_normalized_snapshot

VERSION = 1
FUNDS = ('time', 'koact')
FUND_IDS = ('456600', '2ETFQ1')
LOOKBACKS = (1, 3, 5, 10, 20, 40)
BENCHMARKS = ('456600.KS', '0015B0.KS', 'QQQ')
FX_TICKERS = ('KRW=X', 'JPYKRW=X', 'HKDKRW=X', 'CNY=X')
CURRENCY = {'KR': 'KRW', 'US': 'USD', 'JP': 'JPY', 'HK': 'HKD', 'CN': 'CNY'}
KIND = {'편출': -2, '매도': -1, '유지': 0, '매수': 1, '신규': 2}
CORPORATE_ALIASES = {'SATS US EQUITY': 'ECHO US EQUITY'}
CORPORATE_ALIAS_SOURCES = [{
    'old': 'SATS US EQUITY', 'new': 'ECHO US EQUITY', 'effective_date': '2026-06-24',
    'source': 'https://ir.echostar.com/node/33601/pdf',
    'reason': 'Issuer-confirmed ticker rename of the same security; Yahoo ECHO includes prior history',
}]
# Explicit product-name rules, rather than a blacklist selected on performance.
PRODUCT_NAME = re.compile(
    r'\b(?:ETF|ETN|FUND|ISHARES|PROSHARES|DIREXION|LEVERAGED|ULTRAPRO|'
    r'ULTRASHORT|[23]X|INDEX)\b|(?:^TIME\s|^KoAct\s|^KODEX\s|^TIGER\s)|'
    r'레버리지|인버스|액티브|상장지수|투자신탁', re.IGNORECASE)


def exclusion_reason(holding: Mapping[str, Any]) -> str | None:
    if is_cash_holding(holding):
        return 'cash: tracked separately at actual NAV fraction'
    if listing_market(holding.get('code')) == 'FUT':
        return 'futures: direct unlevered securities universe only'
    if holding.get('asset_type') in {'fund', 'etf', 'etn', 'derivative'}:
        return 'fund/derivative asset_type'
    if PRODUCT_NAME.search(str(holding.get('name', ''))):
        return 'fund/ETF/leveraged product: explicit name rule'
    if listing_market(holding.get('code')) not in CURRENCY:
        return 'unsupported listing identity: cannot map an executable security'
    return None


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _research_key(code: Any) -> str:
    key = company_key(code)
    return CORPORATE_ALIASES.get(key, key)


def _flow_view(snapshot: dict) -> dict:
    """Apply verified same-security rename before canonical quantity calculations.

    Share classes remain separate until after flow calculation, as in flows.py.
    The original normalized snapshot is never changed.
    """
    return {**snapshot, 'holdings': [
        {**row, 'code': CORPORATE_ALIASES.get(canonical_code(row.get('code')), row.get('code'))}
        for row in snapshot['holdings']]}


def _load_inputs() -> tuple[list[dict[str, dict]], list[dict]]:
    funds, inputs = [], []
    for fund_id in FUND_IDS:
        snapshots = {}
        for path in sorted(NORMALIZED_HOLDINGS_DIR.glob(f'*_{fund_id}.json')):
            snapshot = load_normalized_snapshot(path)
            snapshots[snapshot['as_of_date']] = snapshot
            inputs.append({'path': str(path), 'sha256': _hash(path),
                           'as_of_date': snapshot['as_of_date'],
                           'retrieved_at': snapshot.get('retrieved_at')})
        if not snapshots:
            raise ValueError(f'No normalized snapshots for {fund_id}')
        funds.append(snapshots)
    return funds, inputs


def _universe(funds: list[dict[str, dict]]) -> tuple[list[dict], list[dict]]:
    securities, excluded = {}, {}
    for fund, snapshots in zip(FUNDS, funds):
        for day, snapshot in sorted(snapshots.items()):
            for row in snapshot['holdings']:
                code = canonical_code(row.get('code'))
                reason = exclusion_reason(row)
                if reason:
                    item = excluded.setdefault(code, {'code': code, 'name': row.get('name'),
                                                      'reason': reason, 'observations': 0,
                                                      'max_nav_weight_pct': 0.0})
                    item['observations'] += 1
                    item['max_nav_weight_pct'] = max(item['max_nav_weight_pct'], row['weight'])
                    continue
                key = _research_key(code)
                market = listing_market(key)
                item = securities.setdefault(key, {
                    'company_key': key, 'symbol': yahoo_ticker(key), 'name': row.get('name'),
                    'market': market, 'currency': CURRENCY[market], 'aliases': set(),
                    'first_held': day, 'last_held': day, 'funds': set(), 'held_dates': set(),
                })
                item['aliases'].add(code)
                item['funds'].add(fund)
                item['held_dates'].add(day)
                item['first_held'] = min(item['first_held'], day)
                item['last_held'] = max(item['last_held'], day)
    ordered = sorted(securities.values(), key=lambda x: x['symbol'])
    for item in ordered:
        for key in ('aliases', 'funds', 'held_dates'):
            item[key] = sorted(item[key])
    if len({item['symbol'] for item in ordered}) != len(ordered):
        raise ValueError('Two company keys resolved to one Yahoo symbol')
    return ordered, sorted(excluded.values(), key=lambda x: x['code'])


def _download(tickers: list[str], start: str, end: str) -> tuple[dict, list[str]]:
    """Batch downloads with one retry; request Yahoo's exclusive end next day."""
    import yfinance as yf

    prices, notes = {ticker: {} for ticker in tickers}, []
    exclusive = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
    pending = list(tickers)
    for attempt in range(2):
        for offset in range(0, len(pending), 40):
            batch = pending[offset:offset + 40]
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    frame = yf.download(batch, start=start, end=exclusive, auto_adjust=False,
                                        actions=False, progress=False, threads=8,
                                        timeout=20, group_by='column')
                    notes.extend(str(item.message) for item in captured)
                if frame.empty or 'Close' not in frame:
                    continue
                closes = frame['Close']
                for ticker in batch:
                    series = closes[ticker] if getattr(closes, 'ndim', 1) == 2 else closes
                    for index, value in series.dropna().items():
                        day = index.strftime('%Y-%m-%d')
                        if start <= day <= end and math.isfinite(float(value)) and value > 0:
                            prices[ticker][day] = float(value)
            except Exception as exc:
                notes.append(f'Yahoo batch attempt {attempt + 1}: {type(exc).__name__}: {exc}')
        pending = [ticker for ticker in tickers if not prices[ticker]]
        if not pending:
            break
    notes.extend(f'No Yahoo Close observations: {ticker}; retained as missing universe member'
                 for ticker in pending)
    return prices, sorted(set(notes))


def _ffill(series: Mapping[str, float], dates: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.full(len(dates), np.nan)
    ages = np.full(len(dates), np.nan)
    observed = np.zeros(len(dates), dtype=bool)
    last_day, last_value = None, np.nan
    for index, day in enumerate(dates):
        value = series.get(day)
        if value is not None and math.isfinite(value) and value > 0:
            last_day, last_value = day, value
            observed[index] = True
        values[index] = last_value
        if last_day:
            ages[index] = (date.fromisoformat(day) - date.fromisoformat(last_day)).days
    return values, observed, ages


def _fx_arrays(raw: Mapping[str, Mapping[str, float]], dates: list[str]) -> dict:
    result = {'KRW': (np.ones(len(dates)), np.zeros(len(dates)))}
    for currency, ticker in (('USD', 'KRW=X'), ('JPY', 'JPYKRW=X'), ('HKD', 'HKDKRW=X')):
        values, _, ages = _ffill(raw.get(ticker, {}), dates)
        result[currency] = (values, ages)
    cny, _, cny_age = _ffill(raw.get('CNY=X', {}), dates)
    # Observed USD/KRW divided by observed USD/CNY; no synthetic fixed FX rate.
    result['CNY'] = (result['USD'][0] / cny, np.maximum(result['USD'][1], cny_age))
    return result


def _snapshot_arrays(funds: list[dict[str, dict]], dates: list[str], securities: list[dict],
                     krx_sessions: list[str]) -> tuple[dict, list[dict]]:
    funds = [{day: _flow_view(snapshot) for day, snapshot in fund.items()} for fund in funds]
    index = {item['company_key']: i for i, item in enumerate(securities)}
    shape = (len(dates), 2, len(securities))
    weights = np.zeros(shape)
    cash = np.zeros((len(dates), 2))
    excluded_weight = np.zeros((len(dates), 2))
    pp = np.full((len(dates), len(LOOKBACKS), 2, len(securities)), np.nan)
    kinds = np.full(pp.shape, np.nan)
    bases = np.full((len(dates), len(LOOKBACKS), 2), '', dtype='<U10')
    failed = []
    for s, day in enumerate(dates):
        for f, snapshots in enumerate(funds):
            snapshot = snapshots[day]
            for row in snapshot['holdings']:
                key = _research_key(row.get('code'))
                if is_cash_holding(row):
                    cash[s, f] += row['weight'] / 100
                elif exclusion_reason(row) is None and key in index:
                    weights[s, f, index[key]] += row['weight'] / 100
                else:
                    excluded_weight[s, f] += row['weight'] / 100
            sessions_before = bisect_left(krx_sessions, day)
            available = sorted(snapshots)
            for lookback_index, lookback in enumerate(LOOKBACKS):
                if sessions_before < lookback:
                    continue
                intended = krx_sessions[sessions_before - lookback]
                base_index = bisect_right(available, intended) - 1
                if base_index < 0:
                    continue  # Never substitute a shorter-than-requested window.
                base = available[base_index]
                path = [snapshots[d] for d in available if base <= d <= day]
                if len(path) < 2:
                    continue
                result = quantity_flows(path)
                bases[s, lookback_index, f] = base
                if not result.common_estimated:
                    failed.append({'date': day, 'fund': FUNDS[f], 'lookback': lookback,
                                   'base_date': base, 'reason': 'common factor unavailable'})
                    continue
                pp[s, lookback_index, f] = 0
                kinds[s, lookback_index, f] = 0
                for flow in result.flows.values():
                    key = _research_key(flow.key)
                    if key not in index:
                        continue
                    n = index[key]
                    if flow.kind == 'N/A':
                        pp[s, lookback_index, f, n] = np.nan
                        kinds[s, lookback_index, f, n] = np.nan
                    elif flow.kind != '유지':
                        pp[s, lookback_index, f, n] += float(flow.trade_pp)
                        previous = kinds[s, lookback_index, f, n]
                        current = KIND[flow.kind]
                        # Share-class flows are combined only after quantity calculation.
                        if previous and np.sign(previous) != np.sign(current):
                            net = pp[s, lookback_index, f, n]
                            kinds[s, lookback_index, f, n] = np.sign(net)
                        else:
                            kinds[s, lookback_index, f, n] = current
    return {'weights': weights, 'cash': cash, 'excluded_weight': excluded_weight,
            'flow_pp': pp, 'flow_kind': kinds, 'flow_base_dates': bases}, failed


def _atomic_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=path.parent, prefix=f'.{path.name}.',
                                         suffix='.tmp', delete=False) as handle:
            temp = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        temp = None
    finally:
        if temp and temp.exists():
            temp.unlink()


def prepare_bundle(out_dir: Path, refresh: bool = False) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    funds, inputs = _load_inputs()
    snapshot_dates = sorted(set(funds[0]) & set(funds[1]))
    if not snapshot_dates:
        raise ValueError('Funds have no common snapshot period')
    first = min(min(fund) for fund in funds)
    last = max(max(fund) for fund in funds)
    securities, exclusions = _universe(funds)
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    tickers = sorted({item['symbol'] for item in securities} | set(BENCHMARKS) | set(FX_TICKERS))
    raw_path, metadata_path = out_dir / 'raw_prices.json', out_dir / 'metadata.json'
    prior = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    if (not refresh and raw_path.exists() and prior.get('sample_start') == first
            and prior.get('sample_end') == last
            and set(tickers) <= set(prior.get('requested_tickers', []))):
        raw = json.loads(raw_path.read_text(encoding='utf-8'))
        notes = prior.get('download_warnings', [])
    else:
        raw, notes = _download(tickers, first, last)
        atomic_write_json(raw_path, raw)
    raw = {ticker: {d: v for d, v in raw.get(ticker, {}).items() if first <= d <= last}
           for ticker in tickers}
    # FX-only dates are not trading dates, but must participate in causal FX ffill.
    trading_dates = sorted({day for ticker in tickers if ticker not in FX_TICKERS
                            for day in raw[ticker] if first <= day <= last})
    full_dates = sorted(set(trading_dates) | {day for ticker in FX_TICKERS for day in raw[ticker]})
    if not trading_dates:
        raise ValueError('No observed market Close dates inside snapshot sample')
    take = np.array([bisect_left(full_dates, day) for day in trading_dates])
    fx = _fx_arrays(raw, full_dates)
    prices = np.full((len(trading_dates), len(securities)), np.nan)
    tradable = np.zeros(prices.shape, dtype=bool)
    price_age = np.full(prices.shape, np.nan)
    fx_age = np.full(prices.shape, np.nan)
    gaps, missing, coverage = [], [], []
    for n, item in enumerate(securities):
        local, observed, ages = _ffill(raw[item['symbol']], full_dates)
        conversion, conversion_age = fx[item['currency']]
        prices[:, n] = (local * conversion)[take]
        price_age[:, n], fx_age[:, n] = ages[take], conversion_age[take]
        tradable[:, n] = (observed & np.isfinite(conversion) & (conversion_age <= 5))[take]
        if not raw[item['symbol']]:
            missing.append(item['symbol'])
        held = set(item['held_dates'])
        bad = []
        for i, day in enumerate(trading_dates):
            if item['first_held'] <= day <= item['last_held'] and (
                    not np.isfinite(prices[i, n]) or price_age[i, n] > 5 or fx_age[i, n] > 5):
                bad.append(day)
        if bad:
            gaps.append({'symbol': item['symbol'], 'first_problem': bad[0], 'last_problem': bad[-1],
                         'dates': bad, 'count': len(bad),
                         'held_disclosure_dates': sorted(held & set(bad)),
                         'reason': 'no close/FX yet, or last observation more than 5 calendar days old'})
        observations = sorted(raw[item['symbol']])
        coverage.append({'symbol': item['symbol'], 'observations': len(observations),
                         'first_close': observations[0] if observations else None,
                         'last_close': observations[-1] if observations else None})
    benchmarks = np.full((len(trading_dates), 3), np.nan)
    for n, ticker in enumerate(BENCHMARKS):
        values, _, _ = _ffill(raw[ticker], full_dates)
        if ticker == 'QQQ':
            values *= fx['USD'][0]
        benchmarks[:, n] = values[take]
    krx_sessions = sorted(set().union(*(set(fund) for fund in funds),
                                     set(raw[BENCHMARKS[0]]), set(raw[BENCHMARKS[1]])))
    snapshot_arrays, failed = _snapshot_arrays(funds, snapshot_dates, securities, krx_sessions)
    arrays = {'dates': np.asarray(trading_dates), 'symbols': np.asarray([s['symbol'] for s in securities]),
              'prices': prices, 'tradable': tradable, 'price_age_days': price_age,
              'fx_age_days': fx_age, 'benchmarks': benchmarks,
              'benchmark_names': np.asarray(['time', 'koact', 'QQQ']),
              'snapshot_dates': np.asarray(snapshot_dates), 'fund_names': np.asarray(FUNDS),
              'lookbacks': np.asarray(LOOKBACKS), **snapshot_arrays}
    nav_error = float(np.max(np.abs(arrays['weights'].sum(axis=2) + arrays['cash']
                                    + arrays['excluded_weight'] - 1)))
    if nav_error > 1e-6:
        raise ValueError(f'NAV partition does not sum to one: {nav_error}')
    metadata = {
        'version': VERSION, 'generated_at': datetime.now(timezone.utc).isoformat(),
        'sample_start': first, 'sample_end': last, 'input_fingerprint': fingerprint,
        'inputs': inputs, 'snapshot_count': len(inputs), 'common_snapshot_count': len(snapshot_dates),
        'trading_date_count': len(trading_dates), 'security_count': len(securities),
        'fund_order': list(FUNDS), 'lookbacks_krx_sessions': list(LOOKBACKS),
        'securities': securities, 'exclusions': exclusions, 'requested_tickers': tickers,
        'currency_conversion': {'KRW': '1', 'USD': 'KRW=X', 'JPY': 'JPYKRW=X',
                                'HKD': 'HKDKRW=X', 'CNY': 'KRW=X / CNY=X'},
        'currency_notes': 'CNYKRW=X returned no observations; CNY uses two observed USD legs. '
                          'FX forward fill is causal, never backfilled; trade requires FX age <= 5 calendar days.',
        'price_basis': 'Yahoo Close, auto_adjust=False: splits reflected, dividends excluded; KRW conversion',
        'price_download_bounds': {'start_inclusive': first, 'last_inclusive': last,
                                  'yahoo_end_exclusive': (date.fromisoformat(last) + timedelta(days=1)).isoformat()},
        'price_coverage': coverage, 'missing_complete_price_series': missing,
        'held_span_price_gaps_over_5_calendar_days': gaps, 'download_warnings': notes,
        'failed_common_factor_windows': failed, 'nav_partition_max_abs_error': nav_error,
        'raw_prices_sha256': _hash(raw_path),
        'identity_audit': [s for s in securities if len(s['aliases']) > 1 or s['company_key'] not in s['aliases']],
        'corporate_action_aliases': CORPORATE_ALIAS_SOURCES,
        'backfill_temporal_limitation': 'The source snapshots were retrieved retrospectively, mainly '
            '2026-09-27. as_of_date is a disclosure label, not an archived publication timestamp. '
            'Execution must impose a disclosure delay; historical point-in-time availability is not proven.',
        'calendar_and_execution_limitation': 'Local close dates from several time zones share a date axis. '
            'Same-date foreign stock and FX closes are daily marks, not synchronized executable quotes. '
            'Use strict following-date local execution, slippage/costs and extra lag sensitivity.',
        'share_class_note': 'GOOG and GOOGL are combined at company level; GOOG is the execution proxy. '
            'Canonical quantity flows are calculated on each original share class before combination. '
            'SKHYV is canonicalized to SKHY, while Korean 000660 remains a distinct listing.',
        'array_contract': {'weights': '[S,2,N] NAV fractions', 'cash': '[S,2] actual NAV cash fractions',
            'excluded_weight': '[S,2] excluded noncash product fractions',
            'flow_pp': '[S,6,2,N] signed percentage points; only selective >=5%/new/exit',
            'flow_kind': '[S,6,2,N] -2 exit/-1 sell/0 hold/1 buy/2 new; NaN unavailable',
            'prices': '[T,N] causal local Close x causal FX; NaN before first observation',
            'tradable': '[T,N] actual observed local Close and valid FX',
            'price_age_days': '[T,N] days since real local Close; NaN before first',
            'fx_age_days': '[T,N] maximum days since required observed FX leg',
            'benchmarks': '[T,3] time/koact/QQQ KRW closes',
            'flow_base_dates': '[S,6,2] actual base snapshot date, blank if unavailable'},
    }
    _atomic_npz(out_dir / 'bundle.npz', arrays)
    metadata['bundle_sha256'] = _hash(out_dir / 'bundle.npz')
    atomic_write_json(metadata_path, metadata)
    return {'arrays': arrays, 'metadata': metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    result = prepare_bundle(args.out, args.refresh)
    meta = result['metadata']
    print(json.dumps({key: meta[key] for key in ('sample_start', 'sample_end', 'snapshot_count',
                      'common_snapshot_count', 'trading_date_count', 'security_count',
                      'missing_complete_price_series')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
