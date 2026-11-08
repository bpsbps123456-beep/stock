"""Freeze 2025-only MA warmup without changing any portfolio-engine input.

Prefer an existing unadjusted-Close cache. Validate the June overlap in old
split units; fetch missing active symbols via Yahoo with auto_adjust=False.
Do not manufacture history for symbols with no frozen input history or IPOs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / 'grid_1_200' / 'inputs'
FROZEN = HERE.parent / 'concentration_review_20261105' / 'inputs' / 'prices_close.csv'
CACHE = HERE.parent / 'strategy_lab' / 'holdout' / 'prices_close.csv'
BEGIN, CUT, OVERLAP_END = '2025-01-01', '2025-06-02', '2025-06-14'
TOLERANCE = 1e-5


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def align_overlap(old, new):
    """Return the constant unit factor and residual; never accept a varying fit."""
    pair = pd.concat([old.rename('old'), new.rename('new')], axis=1).dropna()
    pair = pair[(pair.old > 0) & (pair.new > 0)]
    if len(pair) < 3:
        return {'overlap_n': len(pair), 'factor_old_per_new': None,
                'max_relative_residual': None, 'accepted': False}
    factor = float((pair.old / pair.new).median())
    residual = float(((pair.new * factor / pair.old) - 1).abs().max())
    return {'overlap_n': len(pair), 'factor_old_per_new': factor,
            'max_relative_residual': residual, 'accepted': residual <= TOLERANCE}


def main():
    old = pd.read_csv(FROZEN, index_col=0, parse_dates=True).sort_index()
    cache = pd.read_csv(CACHE, index_col=0, parse_dates=True).sort_index()
    # The frozen history supplies the overlap calendar, not any future price.
    old_overlap = old.loc[(old.index >= CUT) & (old.index < OVERLAP_END)]
    calendar = cache.loc[(cache.index >= BEGIN) & (cache.index < CUT), 'QQQ'].dropna().index
    warmup = pd.DataFrame(index=calendar, columns=old.columns, dtype=float)
    records, fetch = {}, []
    for ticker in old.columns:
        if not old[ticker].notna().any():
            records[ticker] = {'source': None, 'status': 'no_legacy_signal_prices',
                               'overlap_n': 0, 'factor_old_per_new': None,
                               'max_relative_residual': None}
            continue
        if not old_overlap[ticker].notna().any():
            # No 2025 overlap exists for these later-starting frozen series.
            # Do not invent a unit factor or fill them from another listing.
            records[ticker] = {'source': None, 'status': 'no_legacy_2025_overlap',
                               'first_legacy_date': str(old[ticker].first_valid_index().date()),
                               'overlap_n': 0, 'factor_old_per_new': None,
                               'max_relative_residual': None}
            continue
        if ticker in cache:
            match = align_overlap(old_overlap[ticker], cache[ticker].reindex(old_overlap.index))
            if match['accepted']:
                warmup[ticker] = cache[ticker].reindex(calendar) * match['factor_old_per_new']
                records[ticker] = {'source': str(CACHE.relative_to(ROOT)).replace('\\','/'),
                                   'status': 'aligned', **match}
                continue
        fetch.append(ticker)

    fetched_at = None
    downloaded_hash = None
    if fetch:
        import yfinance as yf
        downloaded = yf.download(fetch, start=BEGIN, end=OVERLAP_END, auto_adjust=False,
                                 progress=False, threads=True, timeout=20)
        fetched_at = datetime.now(timezone.utc).isoformat()
        close = downloaded['Close']
        if isinstance(close, pd.Series):
            close = close.to_frame(fetch[0])
        close = close.sort_index()
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        downloaded_hash = hashlib.sha256(close.to_csv().encode('utf-8')).hexdigest()
        for ticker in fetch:
            series = close[ticker] if ticker in close else pd.Series(dtype=float)
            match = align_overlap(old_overlap[ticker], series.reindex(old_overlap.index))
            records[ticker] = {'source': 'Yahoo/yfinance auto_adjust=False Close',
                               'status': 'aligned' if match['accepted'] else 'overlap_unresolved', **match}
            if match['accepted']:
                warmup[ticker] = series.reindex(calendar) * match['factor_old_per_new']

    before_2026 = old.loc[(old.index >= CUT) & (old.index < '2026-01-02')]
    for ticker in old.columns:
        prefix = warmup[ticker].dropna()
        records[ticker].update({
            'warmup_rows': len(prefix),
            'first_warmup_date': str(prefix.index[0].date()) if len(prefix) else None,
            'last_warmup_date': str(prefix.index[-1].date()) if len(prefix) else None,
            'legacy_pre2026_valid_rows': int(before_2026[ticker].notna().sum()),
            'combined_pre2026_valid_rows': int(len(prefix) + before_2026[ticker].notna().sum())})
    assert warmup.index.min() >= pd.Timestamp(BEGIN)
    assert warmup.index.max() < pd.Timestamp(CUT)
    assert not warmup.index.duplicated().any()
    unresolved = sorted(t for t,r in records.items() if r['status'] == 'overlap_unresolved')
    OUT.mkdir(parents=True, exist_ok=True)
    warmup.to_csv(OUT/'close_warmup.csv', index_label='us_session')
    metadata = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'MA signal warmup only; old ranking, open, close valuation, FX and PDF inputs unchanged',
        'period': {'begin_inclusive': BEGIN, 'end_exclusive': CUT},
        'overlap': {'begin_inclusive': CUT, 'end_exclusive': OVERLAP_END,
                    'minimum_pairs': 3, 'max_relative_residual_allowed': TOLERANCE},
        'price_definition': 'unadjusted-for-dividends Close in frozen input split units; constant overlap unit ratio only',
        'calendar_source': str(CACHE.relative_to(ROOT)).replace('\\','/') + ' QQQ actual nonmissing 2025 sessions',
        'frozen_close_sha256': fingerprint(FROZEN),
        'existing_cache_sha256': fingerprint(CACHE),
        'downloaded_tickers': fetch, 'downloaded_at_utc': fetched_at,
        'downloaded_close_sha256': downloaded_hash,
        'output_sha256': fingerprint(OUT/'close_warmup.csv'),
        'unresolved_active_tickers': unresolved,
        'ticker_count': len(records), 'calendar_sessions': len(warmup),
        'ticker': records,
        'limitations': [
            'Existing cache is read only and restricted to 2025 for this study; no earlier-year returns are evaluated.',
            'Constant unit alignment must agree on every valid overlap row; rejected histories remain missing.',
            'Valid-row counts below 200 may reflect a newer IPO or unresolved history, and are not filled.',
            'EMA initialization changes with the longer 2025 seed history; grid results need not exactly equal the earlier short-warmup study.'
        ]}
    (OUT/'warmup_meta.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'warmup_sessions': len(warmup), 'downloaded_tickers': fetch,
                      'unresolved': unresolved,
                      'below_200': {t:r['combined_pre2026_valid_rows'] for t,r in records.items()
                                    if r['combined_pre2026_valid_rows'] < 200}}, ensure_ascii=False))
    if unresolved:
        raise SystemExit('Unresolved overlap; inspect warmup_meta.json before running grid.')


if __name__ == '__main__':
    main()
