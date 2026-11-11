"""Independent scalar MA, cash/share ledger and output audit for hourly exits.

The research event book supplies an event calendar and valuation marks, each of
which is checked against raw sources here. This auditor never calls run_model,
run_variant, apply_gate, signal_matrix, moving_average or the runner's metrics.
Only the canonical weekly_target allocation decision is reused.
"""
from __future__ import annotations

from datetime import date
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import exchange_calendars as xc
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE/'intraday_4_16h_20261111'
ET = 'America/New_York'
ATOL, RTOL, TIE = 1e-7, 1e-12, 1e-9
MININT = np.iinfo(np.int64).min


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def ns(index):
    return pd.DatetimeIndex(index).as_unit('ns').asi8


def calendar(days, mode):
    """Own session boundaries and trading clock, without runner aggregation."""
    cal = xc.get_calendar('XNYS', start=pd.Timestamp(min(days))-pd.Timedelta(days=7),
                          end=pd.Timestamp(max(days))+pd.Timedelta(days=7))
    result = {}
    for day in days:
        row = cal.schedule.loc[day]
        first, last = row['open'].tz_convert(ET), row['close'].tz_convert(ET)
        if mode == 'extended':
            first = pd.Timestamp(day+' 04:00', tz=ET)
            last = pd.Timestamp(day+(' 17:00' if last.hour == 13 else ' 20:00'), tz=ET)
        result[day] = (first, last)
    return result


def raw_candles(raw, windows, hours):
    rows, u = [], 0.
    for day, (first, last) in windows.items():
        start = first
        while start < last:
            end = min(start+pd.Timedelta(hours=hours), last)
            length = (end-start).total_seconds()/3600
            rows.append((start, end, u, u+length))
            u += length
            start = end
    frame = pd.DataFrame(rows, columns=['start', 'end', 'u_start', 'u_end'])
    frame.index = pd.DatetimeIndex(frame.end)
    left_edge, right_edge = ns(frame.start), ns(frame.end)
    output = {}
    for ticker, bars in raw.items():
        stamps = ns(bars.index)
        assert np.all(stamps[1:] > stamps[:-1]), ticker
        first = np.searchsorted(stamps, left_edge, side='left')
        last = np.searchsorted(stamps, right_edge, side='left')
        values = np.full(len(frame), np.nan)
        valid = last > first
        values[valid] = bars.Close.to_numpy(float)[last[valid]-1]
        output[ticker] = values
    return pd.DataFrame(output, index=frame.index), frame


def scalar_signals(closes, kind, length, events):
    price = np.full((len(events), len(closes.columns)), np.nan)
    mean = price.copy()
    end = np.full(price.shape, MININT, dtype=np.int64)
    ev = ns(events)
    for j, ticker in enumerate(closes):
        present = closes[ticker].dropna()
        values = present.to_numpy(float)
        stamps = ns(present.index)
        averages = np.full(len(values), np.nan)
        if kind == 'SMA':
            for k in range(length-1, len(values)):
                averages[k] = math.fsum(values[k-length+1:k+1])/length
        elif kind == 'EMA':
            alpha, state = 2./(length+1.), None
            for k, value in enumerate(values):
                state = float(value) if state is None else alpha*float(value)+(1-alpha)*state
                if k+1 >= length:
                    averages[k] = state
        else:
            raise ValueError(kind)
        positions = np.searchsorted(stamps, ev, side='right')-1
        valid = positions >= 0
        price[valid, j] = values[positions[valid]]
        mean[valid, j] = averages[positions[valid]]
        end[valid, j] = stamps[positions[valid]]
    return price, mean, end


def all_clock_statistics(closes, meta, start):
    """Independent actual-bar clock kernels for all 240 MA specifications.

    The half-hour clock has exactly representable integer/half-integer values;
    cumulative clock sums therefore give exact SMA window sums here. Price SMA
    calculations separately use math.fsum. EMA clock states follow a direct
    vector recurrence for all lengths, with a first-actual-bar seed per symbol.
    """
    lengths = np.arange(1, 121)
    alpha = 2./(lengths+1.)
    sma_lags = [[] for _ in lengths]
    ema_lags = [[] for _ in lengths]
    spans = [[] for _ in lengths]
    cut = pd.Timestamp(start, tz=ET).value
    for ticker in closes:
        present = closes[ticker].notna().to_numpy()
        u = meta.u_end.to_numpy(float)[present]
        a = meta.u_start.to_numpy(float)[present]
        stamps = ns(meta.index)[present]
        count = len(u)
        if not count:
            continue
        cumulative = np.r_[0., np.cumsum(u)]
        lag_matrix = np.empty((count, 120))
        state = np.full(120, u[0])
        lag_matrix[0] = 0.
        for k in range(1, count):
            state = alpha*u[k]+(1.-alpha)*state
            lag_matrix[k] = u[k]-state
        for n in lengths:
            if count < n:
                continue
            valid = stamps[n-1:] >= cut
            clock_mean = (cumulative[n:]-cumulative[:-n])/n
            sma_lags[n-1].append((u[n-1:]-clock_mean)[valid])
            ema_lags[n-1].append(lag_matrix[n-1:, n-1][valid])
            spans[n-1].append((u[n-1:]-a[:count-n+1])[valid])
    result = {}
    for kind, collections in [('SMA', sma_lags), ('EMA', ema_lags)]:
        for n in lengths:
            lag = np.concatenate(collections[n-1])
            window = np.concatenate(spans[n-1])
            result[kind, int(n)] = {'mean_lag_hours_median': float(np.median(lag)),
                                    'mean_lag_hours_min': float(np.min(lag)),
                                    'mean_lag_hours_max': float(np.max(lag)),
                                    'window_hours_median': float(np.median(window))}
    return result


def allowed(close, average):
    return (math.isfinite(close) and math.isfinite(average) and
            close-average >= -RTOL*max(abs(close), abs(average)))


def metrics(curve, days, begin, end, initial=100.):
    positions = np.flatnonzero((np.asarray(days) >= begin)&(np.asarray(days) <= end))
    a, z = int(positions[0]), int(positions[-1])
    sample = np.r_[curve[a-1] if a else initial, curve[a:z+1]]
    return {'return_pct': float((sample[-1]/sample[0]-1)*100),
            'mdd_pct': float(np.min(sample/np.maximum.accumulate(sample)-1)*100)}


def event_book_check(daily, raw, book, meta, trade_ok, mode, start, end):
    """Check every event quote against its proper raw/frozen source."""
    di = {day: k for k, day in enumerate(daily.days)}
    px = book.opn.to_numpy(float)
    stamps = pd.DatetimeIndex(meta.time)
    keys = list(book.opn)
    expected_ok = np.zeros(px.shape, bool)
    event_rows = list(meta.itertuples(index=False))
    for j, ticker in enumerate(keys):
        opening = raw[ticker].Open.reindex(stamps.floor('30min')).to_numpy(float)
        last = np.nan
        expected_marks = np.full(len(meta), np.nan)
        for k, row in enumerate(event_rows):
            if row.role in ('seed', 'close'):
                value = float(daily.cls[ticker].iloc[di[row.session]])
            elif row.role == 'open':
                value = float(daily.opn[ticker].iloc[di[row.session]])
            else:
                value = opening[k]
            if row.role in ('open', 'trade'):
                expected_ok[k, j] = np.isfinite(value) and value > 0
            if np.isfinite(value):
                last = value
            expected_marks[k] = last
        np.testing.assert_allclose(px[:, j], expected_marks, rtol=0, atol=0, equal_nan=True,
                                   err_msg=f'{mode} {ticker} raw/frozen past-only marks')
    np.testing.assert_array_equal(trade_ok, expected_ok)
    expected_awake = [8 <= stamp.tz_convert('Asia/Seoul').hour < 24 for stamp in stamps]
    np.testing.assert_array_equal(meta.awake, expected_awake)
    assert not stamps.has_duplicates
    assert stamps.is_monotonic_increasing
    windows = calendar([d for d in daily.days if start <= d <= end], mode)
    regular = calendar(list(windows), 'regular')
    prior_week = None
    durations = np.zeros(len(meta))
    for day, (first, last) in windows.items():
        if day == end:
            last = regular[day][1]
        selected = np.flatnonzero(meta.session.eq(day).to_numpy())
        assert len(selected)
        expected = pd.date_range(first, last, freq='30min', inclusive='left')
        assert set(ns(expected)) <= set(ns(stamps[selected])), day
        week = date.fromisoformat(day).isocalendar()[:2]
        weekly = [k for k in selected if meta.weekly_open.iloc[k]]
        assert len(weekly) == int(week != prior_week), (day, weekly)
        if weekly:
            assert stamps[weekly[0]] == regular[day][0]
        prior_week = week
        for pos, k in enumerate(selected):
            next_time = stamps[selected[pos+1]] if pos+1 < len(selected) else last
            durations[k] = max(0., (min(next_time, last)-max(stamps[k], first)).total_seconds()/3600)
        np.testing.assert_allclose(sum(durations[selected]), (last-first).total_seconds()/3600,
                                   rtol=0, atol=1e-12)
    if 'duration_hours' in meta:
        np.testing.assert_allclose(meta.duration_hours, durations, rtol=0, atol=1e-12)
    return durations


def cash_share_path(I, daily, book, meta, ok, durations, closes, kind, length,
                    policy, decision_cache):
    """Actual shares and USD cash; no canonical/research wealth loop is used."""
    di = {day: k for k, day in enumerate(daily.days)}
    ticker_pos = {ticker: j for j, ticker in enumerate(book.opn)}
    marks = book.opn.to_numpy(float)
    price, mean, ends = ((None, None, None) if length == 0 else
                         scalar_signals(closes, kind, length, meta.time))
    if length:
        assert list(closes.columns) == list(book.opn)
    cash, shares, selected = 1., {}, set()
    rows, exits, entries, snapshots = [], [], [], []
    unknown, new_entries = 0, 0
    for k, row in enumerate(meta.itertuples(index=False)):
        if row.session < I.START:
            continue
        p = marks[k]
        old_values = {key: quantity*p[ticker_pos[daily.ticker[key]]]
                      for key, quantity in shares.items()}
        assert all(math.isfinite(value) and value > 0 for value in old_values.values())
        before = cash+math.fsum(old_values.values())
        assert before > 0
        active = row.role in ('open', 'trade') and (policy != 'awake' or
                  8 <= row.time.tz_convert('Asia/Seoul').hour < 24)
        if length and active:
            for key in list(shares):
                j = ticker_pos[daily.ticker[key]]
                c, a = float(price[k, j]), float(mean[k, j])
                if not (math.isfinite(c) and math.isfinite(a)):
                    unknown += 1
                    continue
                if not allowed(c, a) and ok[k, j]:
                    assert ends[k, j] <= row.time.value
                    cash += shares.pop(key)*p[j]
                    selected.discard(key)
                    exits.append({'event': row.event, 'session': row.session,
                                  'ticker': daily.ticker[key], 'key': key,
                                  'close': c, 'ma': a, 'signal_ns': int(ends[k, j]),
                                  'price': float(p[j])})
        if row.weekly_open:
            cache_key = (row.session, tuple(sorted(selected)))
            if cache_key not in decision_cache:
                decision_cache[cache_key] = I.S.weekly_target(
                    daily, di[row.session], row.session, selected)
            decision = decision_cache[cache_key]
            if decision['ranked']:
                blocked = set()
                for key in decision['selected']:
                    if key not in selected and length:
                        j = ticker_pos[daily.ticker[key]]
                        c, a = float(price[k, j]), float(mean[k, j])
                        passed = allowed(c, a)
                        if not passed:
                            blocked.add(key)
                        assert ends[k, j] <= row.time.value or ends[k, j] == MININT
                        entries.append({'event': row.event, 'session': row.session,
                                        'ticker': daily.ticker[key], 'key': key,
                                        'close': c, 'ma': a, 'signal_ns': int(ends[k, j]),
                                        'allowed': passed})
                target = {key: weight for key, weight in decision['target'].items()
                          if key not in blocked}
                wealth = cash+math.fsum(quantity*p[ticker_pos[daily.ticker[key]]]
                                         for key, quantity in shares.items())
                held_values = {key: quantity*p[ticker_pos[daily.ticker[key]]]
                               for key, quantity in shares.items()}
                for key in set(target)|set(shares):
                    dollars = wealth*target.get(key, 0.)-held_values.get(key, 0.)
                    if abs(dollars) > 1e-12:
                        assert ok[k, ticker_pos[daily.ticker[key]]], (row.event, key, 'weekly fill missing')
                new_entries += len(set(target)-set(shares))
                updated = {}
                for key, weight in target.items():
                    execution = p[ticker_pos[daily.ticker[key]]]
                    assert math.isfinite(execution) and execution > 0
                    updated[key] = wealth*weight/execution
                cash = wealth-math.fsum(quantity*p[ticker_pos[daily.ticker[key]]]
                                       for key, quantity in updated.items())
                shares = updated
                selected = set(decision['selected'])-blocked
        new_values = {key: quantity*p[ticker_pos[daily.ticker[key]]]
                      for key, quantity in shares.items()}
        wealth = cash+math.fsum(new_values.values())
        assert abs(wealth-before) < 1e-10, (row.event, wealth, before)
        turnover = math.fsum(abs(new_values.get(key, 0.)-old_values.get(key, 0.))
                             for key in set(new_values)|set(old_values))/wealth
        exposure = math.fsum(new_values.values())/wealth
        assert cash > -1e-10 and -1e-10 <= exposure <= 1+1e-10
        rows.append({'session': row.session, 'turnover': turnover, 'exposure': exposure,
                     'duration': float(durations[k]), 'usd': wealth})
        if row.role == 'close':
            snapshots.append((len(rows)-1, row.session, wealth))
    events = pd.DataFrame(rows)
    positions = np.asarray([x[0] for x in snapshots])
    days = [x[1] for x in snapshots]
    usd = np.asarray([x[2] for x in snapshots])
    base_fx = daily.fx[di[I.START]-1]
    krw = usd*np.asarray([daily.fx[di[d]] for d in days])/base_fx*100
    summary = {'ma_exits': len(exits), 'unknown_held': unknown,
               'blocked_entries': sum(not x['allowed'] for x in entries),
               'new_entries': new_entries}
    for label, (begin, end) in I.PERIODS.items():
        summary.update({f'{label}_{key}': value for key, value in metrics(krw, days, begin, end).items()})
        mask = (events.session >= begin)&(events.session <= end)
        w = events.loc[mask, 'duration'].to_numpy(float)
        summary[f'{label}_exposure_pct'] = float(np.dot(events.loc[mask, 'exposure'], w)/w.sum()*100)
        summary[f'{label}_turnover'] = float(events.loc[mask, 'turnover'].sum())
        summary[f'{label}_ma_exits'] = sum(begin <= x['session'] <= end for x in exits)
    for bp in (5, 10):
        adjusted = krw*np.cumprod(1.-events.turnover.to_numpy(float)*bp/10000)[positions]
        for label, (begin, end) in I.PERIODS.items():
            summary[f'{label}_return_{bp}bp_pct'] = metrics(adjusted, days, begin, end)['return_pct']
    return summary, krw, days, entries, exits


def choose(rows, risk=False):
    base = next(row for row in rows if int(row['length']) == 0)
    candidates = [row for row in rows if int(row['length']) > 0 and
                  (not risk or row['front_mdd_pct'] >= base['front_mdd_pct']-TIE)]
    best = max(row['front_return_pct'] for row in candidates)
    return min((row for row in candidates if best-row['front_return_pct'] <= TIE),
               key=lambda row: (row.get('mode') != 'regular', row['kind'] != 'SMA',
                                int(row['hours']), int(row['length'])))['name']


def read_ledger(path):
    try:
        return pd.read_csv(path).to_dict('records')
    except pd.errors.EmptyDataError:
        return []


def compare_ledger(expected, path, field):
    stored = read_ledger(path)
    assert len(stored) == len(expected), (path.name, len(stored), len(expected))
    for old, new in zip(stored, expected):
        for key in ('event', 'session', 'ticker', 'key', 'signal_ns'):
            assert old[key] == new[key], (path.name, key, old[key], new[key])
        if field == 'entries':
            assert bool(old['allowed']) == new['allowed']
        for key in ('close', 'ma')+ (('price',) if field == 'exits' else ()):
            np.testing.assert_allclose(old[key], new[key], rtol=1e-12, atol=1e-9, equal_nan=True)


def main():
    began = time.perf_counter()
    I = module('intraday_runner_for_independent_audit', HERE/'intraday_review.py')
    result = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    source = json.loads((OUT/'input_metadata.json').read_text(encoding='utf-8'))
    assert digest(OUT/'protocol.json') == result['protocol_sha256']
    assert digest(OUT/'input_metadata.json') == result['input_metadata_sha256']
    for path, expected in source['frozen_daily_and_pdf_sha256'].items():
        assert digest(ROOT/path) == expected, path
    prior_study = json.loads((HERE/'exit_1_200'/'results.json').read_text(encoding='utf-8'))
    old_hashes = {path.replace('\\', '/'): value for path, value in prior_study['input_sha256'].items()}
    new_hashes = {path.replace('\\', '/'): value
                  for path, value in source['frozen_daily_and_pdf_sha256'].items()}
    shared_inputs = set(old_hashes)&set(new_hashes)
    assert len(shared_inputs) >= 3
    for path in shared_inputs:
        assert old_hashes[path] == new_hashes[path], ('changed frozen prior-study input', path)
    assert digest(OUT/'coverage.csv') == source['coverage_sha256']
    coverage = pd.read_csv(OUT/'coverage.csv')
    for row in coverage.itertuples(index=False):
        if pd.notna(getattr(row, 'sha256', None)):
            assert digest(OUT/'cache'/'alp30m'/f'{row.ticker}.parquet') == row.sha256
    summary = pd.read_csv(OUT/'summary_unique.csv')
    assert not summary.name.duplicated().any()
    expected_nonbaseline = (4+13)*2*120*2
    assert len(summary[summary.length > 0]) == expected_nonbaseline
    assert len(summary[summary.length == 0]) == 4
    aliases = pd.read_csv(OUT/'candidate_aliases.csv')
    assert len(aliases) == 2*2*13*2*120
    assert not aliases.requested.duplicated().any()
    assert set(aliases.source) <= set(summary.name)
    for row in aliases.itertuples(index=False):
        mode, policy, h, kind_n = row.requested.split('_')
        expected_h = min(int(h[:-1]), 7) if mode == 'regular' else int(h[:-1])
        assert row.source == f'{mode}_{policy}_{expected_h}h_{kind_n}'
    daily, raw = I.load_inputs()
    curves, groups = {}, {}
    replay = set(result['selected_front_only'].values())
    replay.update(result.get('selected_front_pooled', {}).values())
    for field in ('entries', 'exits'):
        suffix = f'_{field}.csv'
        replay.update(path.name[:-len(suffix)] for path in OUT.glob(f'*{suffix}'))
    front_selections = {}
    errors = {'all_curve_period_metrics': 0., 'selected_curve_index': 0.,
              'selected_metrics': 0., 'all_actual_bar_time_units': 0.}
    checked_fields = 0
    for mode in ('regular', 'extended'):
        for policy in ('awake', 'unrestricted'):
            name = f'{mode}_{policy}'
            frame = pd.read_csv(OUT/f'curves_{name}.csv', index_col=0)
            rows = summary[(summary['mode'] == mode)&(summary.policy == policy)].to_dict('records')
            assert set(frame) == {row['name'] for row in rows}
            groups[name] = (frame, rows)
            curves.update({column: frame[column].to_numpy(float) for column in frame})
            for row in rows:
                for label, (begin, end) in I.PERIODS.items():
                    own = metrics(curves[row['name']], frame.index.to_numpy(str), begin, end)
                    for field, value in own.items():
                        error = abs(value-row[f'{label}_{field}'])
                        errors['all_curve_period_metrics'] = max(errors['all_curve_period_metrics'], error)
                        assert error < ATOL, (row['name'], label, field, error)
                        checked_fields += 1
            for risk in (False, True):
                key = name+('_risk_floor' if risk else '')
                front_selections[key] = choose(rows, risk)
                assert result['selected_front_only'][key] == front_selections[key]
                mutated = [dict(row, back_return_pct=1e60*(i+1), full_return_pct=-1e80*(i+1),
                                back_mdd_pct=-100., full_mdd_pct=-100.) for i, row in enumerate(rows)]
                assert choose(mutated, risk) == front_selections[key]
            nonbaseline = [row for row in rows if row['length'] > 0]
            for field in ('back_return_pct', 'full_return_pct'):
                replay.add(max(nonbaseline, key=lambda row: row[field])['name'])
            replay.add(f'{mode}_{policy}_4h_SMA1')
            replay.add(f'{mode}_{policy}_baseline')
            # Fixed boundary checks prevent validation from following only winners.
            for hours in (4, 7 if mode == 'regular' else 16):
                for kind in ('SMA', 'EMA'):
                    replay.add(f'{mode}_{policy}_{hours}h_{kind}120')
    pooled_selections = {}
    for policy in ('awake', 'unrestricted'):
        rows = summary[summary.policy == policy].to_dict('records')
        for risk in (False, True):
            key = policy+('_risk_floor' if risk else '')
            pooled_selections[key] = choose(rows, risk)
            assert result['selected_front_pooled'][key] == pooled_selections[key]
            changed = [dict(row, back_return_pct=1e60*(i+1), full_return_pct=-1e80*(i+1),
                            back_mdd_pct=-100., full_mdd_pct=-100.) for i, row in enumerate(rows)]
            assert choose(changed, risk) == pooled_selections[key]
    old_baseline = pd.read_csv(HERE/'exit_1_200'/'curves_krw.csv', index_col=0)['baseline']
    for name, values in curves.items():
        if name.endswith('_baseline') or name.endswith('_SMA1') or name.endswith('_EMA1'):
            np.testing.assert_allclose(values, old_baseline.to_numpy(float), rtol=0, atol=ATOL,
                                       err_msg=name)
    # Independently reconstruct every requested nonredundant candle, all symbols.
    candle_cache, unit_meta, event_cache = {}, {}, {}
    candle_values, unit_fields = 0, 0
    for mode in ('regular', 'extended'):
        windows = calendar(daily.days, mode)
        for hours in (range(4, 8) if mode == 'regular' else range(4, 17)):
            close, own_meta = raw_candles(raw, windows, hours)
            stored = pd.read_parquet(OUT/'cache'/f'closes_{mode}_{hours}h.parquet')
            np.testing.assert_array_equal(ns(close.index), ns(stored.index))
            assert list(close) == list(stored)
            np.testing.assert_allclose(close, stored, rtol=0, atol=0, equal_nan=True,
                                       err_msg=f'{mode} {hours}h aggregation')
            candle_cache[mode, hours] = close
            unit_meta[mode, hours] = own_meta
            candle_values += close.size
            own_units = all_clock_statistics(close, own_meta, I.START)
            for row in summary[(summary['mode'] == mode)&(summary.hours == hours)].to_dict('records'):
                for field, value in own_units[row['kind'], int(row['length'])].items():
                    error = abs(value-float(row[field]))
                    errors['all_actual_bar_time_units'] = max(errors['all_actual_bar_time_units'], error)
                    assert error < ATOL, (row['name'], field, value, row[field])
                    unit_fields += 1
            print(f'Independent all-symbol {mode} {hours}h candles and 240 clock kernels passed', flush=True)
        book, meta, ok = I.event_book(daily, raw, mode)
        durations = event_book_check(daily, raw, book, meta, ok, mode, I.START, I.END)
        event_cache[mode] = book, meta, ok, durations
    output_paths, decision_cache, ledgers, ledger_files = [], {}, 0, 0
    indexed = summary.set_index('name')
    for number, name in enumerate(sorted(replay), 1):
        row = indexed.loc[name]
        mode, policy, hours = row['mode'], row.policy, int(row.hours)
        kind, length = row.kind, int(row.length)
        book, meta, ok, durations = event_cache[mode]
        close = candle_cache[mode, hours] if length else None
        own, curve, days, entries, exits = cash_share_path(
            I, daily, book, meta, ok, durations, close, kind, length, policy, decision_cache)
        frame, _ = groups[f'{mode}_{policy}']
        assert days == frame.index.tolist()
        curve_error = float(np.max(np.abs(curve-curves[name])))
        errors['selected_curve_index'] = max(errors['selected_curve_index'], curve_error)
        assert curve_error < ATOL, (name, curve_error)
        for key, value in own.items():
            error = abs(value-float(row[key]))
            errors['selected_metrics'] = max(errors['selected_metrics'], error)
            assert error < ATOL, (name, key, value, row[key], error)
        for field, events in [('entries', entries), ('exits', exits)]:
            path = OUT/f'{name}_{field}.csv'
            if path.exists():
                compare_ledger(events, path, field)
                ledgers += len(events)
                ledger_files += 1
        output_paths.append({'name': name, 'max_curve_error': curve_error,
                             'ma_exits': len(exits), 'entry_decisions': len(entries),
                             'full_return_pct': own['full_return_pct'],
                             'full_turnover': own['full_turnover']})
        print(f'Independent cash/share audit {number}/{len(replay)}: {name}', flush=True)
    audit = {'status': 'passed', 'elapsed_seconds': time.perf_counter()-began,
             'source_audit_sha256': digest(__file__), 'runner_sha256': digest(HERE/'intraday_review.py'),
             'results_sha256': digest(OUT/'results.json'), 'summary_sha256': digest(OUT/'summary_unique.csv'),
             'protocol_sha256': result['protocol_sha256'],
             'raw_files_hashed': int(coverage.sha256.notna().sum()),
             'prior_daily_study_shared_input_hashes_unchanged': len(shared_inputs),
             'nominal_candidates_per_policy': 6240, 'timeframe_deduplicated_candidates_per_policy': 4080,
             'summary_rows_checked': len(summary), 'all_curve_period_metric_fields': checked_fields,
             'all_actual_bar_time_unit_fields': unit_fields,
             'independent_raw_aggregation_values': candle_values,
             'cash_share_paths_checked': len(output_paths), 'ledger_rows_checked': ledgers,
             'ledger_files_checked': ledger_files,
             'front_only_selection_and_future_mutation': front_selections,
             'pooled_front_only_selection_and_future_mutation': pooled_selections,
             'N1_all_modes_and_policies_equal_frozen_original': True,
             'errors': errors, 'paths': output_paths,
             'scope': '2026 frozen original seven-stock model; not independent unseen validation',
             'limitations': ['Execution within each 30-minute source bar and actual spread are unobserved.',
                             'MDD uses regular-session daily close marks, not the deepest intraday loss.',
                             'SIP signal closes and June-2025 EMA seeds differ from the prior daily-MA study.',
                             'Weekly targets reuse the canonical allocator; wealth and MA are independent.',
                             'Repeated searches of already-seen 2026 data retain selection bias.']}
    (OUT/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: audit[key] for key in ['status', 'cash_share_paths_checked', 'errors', 'elapsed_seconds']},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
