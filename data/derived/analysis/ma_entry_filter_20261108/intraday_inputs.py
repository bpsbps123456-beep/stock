"""Freeze SIP 30-minute inputs for the 4..16-hour MA exit study.

The operating desk cache, daily anchors and point-in-time PDF selection are read-only.
Raw bars are split-adjusted, start-labelled and stored in the ignored cache directory.
No bars are forward-filled; absent trades and missing sessions remain visible.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE/'intraday_4_16h_20261111'
CACHE = OUT/'cache'/'alp30m'
BEGIN, END = '2025-06-02', '2026-11-04'
ET = 'America/New_York'
COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Trades', 'VWAP']


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _module('ma_daily_for_intraday_inputs', HERE/'review.py')
A = _module('alpaca_for_intraday_inputs', HERE.parent/'hatta_bar/code/fetch_alpaca.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_book():
    """Original 83-ticker frozen daily/PDF book, without refreshing shared prices."""
    return M.R.load_book(M.BASE/'inputs', END)


def load_bars(ticker, cache=CACHE):
    """NY-time, start-labelled 30-minute OHLCV; empty sessions are not filled."""
    return pd.read_parquet(Path(cache)/f'{ticker}.parquet')


def normalise_bars(rows, begin=BEGIN, end=END):
    if not rows:
        return pd.DataFrame(columns=COLUMNS,
                            index=pd.DatetimeIndex([], tz=ET, name='timestamp'))
    data = pd.DataFrame(rows)
    data.index = pd.to_datetime(data.pop('t'), utc=True).dt.tz_convert(ET)
    data.index.name = 'timestamp'
    data = data.rename(columns={'o':'Open', 'h':'High', 'l':'Low', 'c':'Close',
                                'v':'Volume', 'n':'Trades', 'vw':'VWAP'})
    data = data.reindex(columns=COLUMNS).sort_index()
    if data.index.has_duplicates:
        raise ValueError('Duplicate provider timestamps')
    minutes = data.index.hour*60+data.index.minute
    dates = data.index.strftime('%Y-%m-%d')
    data = data[(dates >= begin)&(dates <= end)&(minutes >= 240)&(minutes < 1200)]
    if len(data):
        if ((data.index.minute % 30 != 0)|(data.index.second != 0)).any():
            raise ValueError('Provider 30-minute bar is not aligned to :00/:30')
        ohlc = data[['Open','High','Low','Close']].to_numpy(float)
        if not np.isfinite(ohlc).all() or (ohlc <= 0).any():
            raise ValueError('Non-finite/non-positive OHLC')
        if (data.High+1e-10 < data[['Open','Close','Low']].max(axis=1)).any():
            raise ValueError('High below another OHLC field')
        if (data.Low-1e-10 > data[['Open','Close','High']].min(axis=1)).any():
            raise ValueError('Low above another OHLC field')
        if (data.Volume < 0).any():
            raise ValueError('Negative volume')
    return data


def fetch_one(ticker, headers, pace, *, begin=BEGIN, end=END, cache=CACHE,
              refresh=False):
    """Paginate the existing Alpaca SIP endpoint, sharing its request rate limiter."""
    path = Path(cache)/f'{ticker}.parquet'
    if path.exists() and not refresh:
        return {'ticker':ticker, 'fetch_status':'cached', 'error':''}
    params = {'symbols':ticker, 'timeframe':'30Min',
              'start':pd.Timestamp(begin,tz=ET).tz_convert('UTC').isoformat(),
              'end':(pd.Timestamp(end)+pd.Timedelta(days=1)).tz_localize(ET).tz_convert('UTC').isoformat(),
              'limit':10000, 'adjustment':'split', 'feed':'sip', 'sort':'asc'}
    rows, tokens = [], set()
    while True:
        for attempt in range(6):
            pace.wait()
            try:
                response = requests.get(A.URL,headers=headers,params=params,timeout=45)
            except requests.exceptions.RequestException:
                if attempt == 5:
                    raise RuntimeError('Provider connection failed after six attempts') from None
                time.sleep(min(2+attempt*3, 17))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 5:
                    raise RuntimeError(f'Provider HTTP {response.status_code} after six attempts')
                time.sleep(min(2+attempt*3, 17))
                continue
            if response.status_code != 200:
                # Print only the provider's public message, never headers/credentials.
                try:
                    message = str(response.json().get('message',''))[:180]
                except ValueError:
                    message = ''
                raise RuntimeError(f'Provider HTTP {response.status_code}: {message}')
            payload = response.json()
            break
        rows.extend((payload.get('bars') or {}).get(ticker, []))
        token = payload.get('next_page_token')
        if not token:
            break
        if token in tokens:
            raise RuntimeError('Provider pagination token repeated')
        tokens.add(token)
        params['page_token'] = token
    data = normalise_bars(rows, begin, end)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp.parquet')
    data.to_parquet(temporary)
    temporary.replace(path)
    return {'ticker':ticker, 'fetch_status':'downloaded' if len(data) else 'empty', 'error':''}


def model_requirements(book):
    """Top-12 union bounds every possible buffered 7-name weekly selection."""
    by_ticker = {ticker:{'weekly_ranked_n':0,'weekly_top12_n':0,
                         'baseline_held_sessions':0} for ticker in book.ticker.values()}
    last_week = None
    for k, day in enumerate(book.days):
        week = M.S.iso_week(day)
        if day < '2026-01-02' or week == last_week:
            continue
        last_week = week
        _, rows = M.S.momentum_ranking(book,k,day)
        for row in rows:
            info = by_ticker[row['ticker']]
            info['weekly_ranked_n'] += 1
            info['weekly_top12_n'] += int(row['rank'] <= 12)
    baseline = M.S.run_model(book)
    for weights in baseline.weights:
        for key, weight in weights.items():
            if weight > 0:
                by_ticker[book.ticker[key]]['baseline_held_sessions'] += 1
    return by_ticker


def coverage_row(ticker, book, requirements, *, cache=CACHE, begin=BEGIN, end=END):
    """Measure actual sessions against frozen daily anchors, including early closes."""
    import exchange_calendars as xcals
    info = {'ticker':ticker, **requirements.get(ticker,{}), 'rows':0}
    path = Path(cache)/f'{ticker}.parquet'
    if not path.exists():
        return {**info,'data_status':'missing'}
    bars = load_bars(ticker,cache)
    info.update(rows=len(bars), sha256=sha(path),
                first_timestamp=str(bars.index.min()) if len(bars) else '',
                last_timestamp=str(bars.index.max()) if len(bars) else '',
                duplicate_timestamps=int(bars.index.duplicated().sum()))
    if not len(bars):
        return {**info,'data_status':'empty'}
    minutes = bars.index.hour*60+bars.index.minute
    bar_dates = bars.index.strftime('%Y-%m-%d')
    cal = xcals.get_calendar('XNYS')
    daily_close = book.cls.get(ticker,pd.Series(dtype=float))
    daily_open = book.opn.get(ticker,pd.Series(dtype=float))
    expected, missing_days, missing_slots, observed_days = 0, [], 0, 0
    open_errors, close_errors, bad_close_days = [], [], []
    for day in book.days:
        if not begin <= day <= end or not np.isfinite(daily_close.get(day,np.nan)):
            continue
        expected += 1
        session = pd.Timestamp(day)
        close = cal.session_close(session).tz_convert(ET)
        end_minute = close.hour*60+close.minute
        selected = (bar_dates == day)&(minutes >= 570)&(minutes < end_minute)
        rth = bars.loc[selected]
        expected_slots = set(range(570,end_minute,30))
        actual_slots = set(minutes[selected])
        missing_slots += len(expected_slots-actual_slots)
        if not len(rth):
            missing_days.append(day)
            continue
        observed_days += 1
        if rth.index[0].hour*60+rth.index[0].minute == 570:
            anchor = daily_open.get(day,np.nan)
            if np.isfinite(anchor) and anchor > 0:
                open_errors.append(abs(float(rth.Open.iloc[0])/anchor-1))
        if rth.index[-1].hour*60+rth.index[-1].minute == end_minute-30:
            error = abs(float(rth.Close.iloc[-1])/float(daily_close[day])-1)
            close_errors.append(error)
            if error > .01:
                bad_close_days.append(day)
    info.update(data_status='available', expected_anchor_sessions=expected,
                observed_rth_anchor_sessions=observed_days,
                missing_rth_days_n=len(missing_days), missing_rth_days=';'.join(missing_days),
                missing_rth_slots=missing_slots,
                daily_open_abs_relerr_median=float(np.median(open_errors)) if open_errors else None,
                daily_open_abs_relerr_max=max(open_errors,default=None),
                daily_close_abs_relerr_median=float(np.median(close_errors)) if close_errors else None,
                daily_close_abs_relerr_max=max(close_errors,default=None),
                close_over_1pct_days_n=len(bad_close_days),close_over_1pct_days=';'.join(bad_close_days))
    return info


def frozen_hashes(book):
    paths = list((M.BASE/'inputs').glob('prices_*.csv'))
    # This book consumes only TIME PDFs; hash exactly the snapshots it actually read.
    paths.extend(ROOT/'data/normalized/etf_holdings'/f'{day}_456600.json' for day in book.pdfs)
    return {path.relative_to(ROOT).as_posix():sha(path) for path in paths if path.exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tickers',nargs='*',help='Default: original frozen 83-ticker book')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--inventory-only',action='store_true')
    args = parser.parse_args()
    book = load_book()
    tickers = sorted(set(args.tickers or book.ticker.values()))
    requirements = model_requirements(book)
    OUT.mkdir(parents=True,exist_ok=True)
    results = []
    if not args.inventory_only:
        headers, pace = A.keys(), A.Pace(3.)
        print(f'SIP 30Min: {len(tickers)} symbols, {BEGIN}..{END}, {args.workers} workers',flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(fetch_one,ticker,headers,pace,refresh=args.refresh):ticker
                       for ticker in tickers}
            for done in as_completed(pending):
                ticker = pending[done]
                try:
                    result = done.result()
                except Exception as exc:
                    result = {'ticker':ticker,'fetch_status':'error','error':str(exc)}
                results.append(result)
                print(f'{len(results)}/{len(tickers)} {ticker}: {result["fetch_status"]}'
                      +(f' {result["error"]}' if result['error'] else ''),flush=True)
    by_ticker = {row['ticker']:row for row in results}
    rows = [{**coverage_row(ticker,book,requirements),**by_ticker.get(ticker,{})}
            for ticker in tickers]
    pd.DataFrame(rows).to_csv(OUT/'coverage.csv',index=False)
    metadata = {'created_at_utc':datetime.now(timezone.utc).isoformat(),
                'provider':'Alpaca historical stock bars','feed':'sip','adjustment':'split',
                'timeframe':'30Min','timezone':ET,'timestamp_label':'bar start',
                'requested_begin_inclusive':BEGIN,'requested_end_inclusive':END,
                'session_filter':'04:00 <= NY bar-start < 20:00',
                'rth_calendar':'XNYS, including early closes',
                'missing_bars_policy':'No OHLC forward-fill or synthetic trades',
                'tickers':tickers,'ticker_count':len(tickers),
                'cache_relative_to_root':CACHE.relative_to(ROOT).as_posix(),
                'columns':COLUMNS,'frozen_daily_and_pdf_sha256':frozen_hashes(book),
                'coverage_sha256':sha(OUT/'coverage.csv'),
                'source_sha256':sha(__file__),
                'downloaded_raw_files_ignored_by_git':True,
                'operating_rule_changed':False}
    (OUT/'input_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Coverage saved: {OUT.relative_to(ROOT)}/coverage.csv',flush=True)


if __name__ == '__main__':
    main()
