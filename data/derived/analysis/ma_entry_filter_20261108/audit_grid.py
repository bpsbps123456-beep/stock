"""Independent MA arithmetic, grid metrics and selected-path replay audit.

Does not call moving_average, period, annotate_events or flatten from the grid
runner. Reuses only the explicitly required existing portfolio/gate engine.
Writes grid_1_200/audit.json; no operating inputs or settings are modified.
"""
from __future__ import annotations

from collections import deque
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE/'grid_1_200'
SPEC = importlib.util.spec_from_file_location('grid_under_audit', HERE/'grid_review.py')
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)
M = G.M
TOL = 1e-9


def independent_ma(prices, kind, length):
    output = np.full(prices.shape, np.nan, dtype=float)
    for j in range(prices.shape[1]):
        values = prices.iloc[:, j].to_numpy(float)
        if kind == 'SMA':
            window = deque(maxlen=length)
            for i, value in enumerate(values):
                window.append(value)
                if len(window) == length and all(math.isfinite(x) for x in window):
                    output[i,j] = math.fsum(window)/length
        else:
            alpha, count, state = 2./(length+1), 0, None
            for i, value in enumerate(values):
                if math.isfinite(value):
                    state = value if state is None else (1-alpha)*state+alpha*value
                    count += 1
                if count >= length:
                    output[i,j] = state
    return pd.DataFrame(output, index=prices.index, columns=prices.columns)


def direct_metrics(curves, start, finish):
    positions = np.flatnonzero((curves.index >= start) & (curves.index <= finish))
    lo, hi = int(positions[0]), int(positions[-1])
    prior = curves.iloc[lo-1].to_numpy() if lo else np.full(len(curves.columns), 100.)
    values = np.vstack([prior, curves.iloc[lo:hi+1].to_numpy()])
    returns = (values[-1]/values[0]-1)*100
    drawdown = (values/np.maximum.accumulate(values, axis=0)-1).min(axis=0)*100
    return dict(zip(curves.columns, returns)), dict(zip(curves.columns, drawdown))


def own_winner(rows, field, kind=None):
    subset = [r for r in rows if r['name'] != 'baseline' and (kind is None or r['kind'] == kind)]
    maximum = max(r[field] for r in subset)
    candidates = [r for r in subset if maximum-r[field] <= TOL]
    candidates.sort(key=lambda r: (r['kind'] != 'SMA', r['length']))
    return candidates[0]['name']


def bool_value(value):
    return value.lower() == 'true' if isinstance(value, str) else bool(value)


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_warmup(warmup, signals):
    meta = json.loads((OUT/'inputs/warmup_meta.json').read_text(encoding='utf-8'))
    assert hash_file(OUT/'inputs/close_warmup.csv') == meta['output_sha256']
    cache_path = M.ROOT/'data/derived/analysis/strategy_lab/holdout/prices_close.csv'
    assert hash_file(cache_path) == meta['existing_cache_sha256']
    assert hash_file(M.BASE/'inputs/prices_close.csv') == meta['frozen_close_sha256']
    assert not meta['unresolved_active_tickers']
    old = pd.read_csv(M.BASE/'inputs/prices_close.csv', index_col=0, parse_dates=True)
    cache = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    overlap = old.loc['2025-06-02':'2025-06-13']
    local_verified, downloaded_verified = 0, 0
    max_residual = 0.
    # Re-download only the 17 earlier-missing sources to independently verify
    # their overlap and actual frozen prefix. No CSV/cache is overwritten.
    tickers = meta['downloaded_tickers']
    downloaded = None
    if tickers:
        import yfinance as yf
        data = yf.download(tickers, start='2025-01-01', end='2025-06-14',
                           auto_adjust=False, progress=False, threads=True, timeout=20)
        downloaded = data['Close'].sort_index()
        if downloaded.index.tz is not None:
            downloaded.index = downloaded.index.tz_localize(None)
    for ticker, details in meta['ticker'].items():
        if details['status'] != 'aligned':
            assert warmup[ticker].isna().all()
            continue
        source = downloaded if ticker in tickers else cache
        pair = pd.DataFrame({'frozen': overlap[ticker], 'source': source[ticker].reindex(overlap.index)}).dropna()
        ratio = (pair.frozen/pair.source).to_numpy()
        assert len(ratio) == details['overlap_n']
        factor = float(np.median(ratio))
        residual = float(np.max(np.abs(pair.source.to_numpy()*factor/pair.frozen.to_numpy()-1)))
        assert abs(factor-details['factor_old_per_new']) <= 1e-10
        assert residual <= meta['overlap']['max_relative_residual_allowed']
        expected = source[ticker].copy()
        expected.index = expected.index.strftime('%Y-%m-%d')
        np.testing.assert_allclose(warmup[ticker], expected.reindex(warmup.index)*factor,
                                   rtol=1e-10, atol=1e-8, equal_nan=True)
        max_residual = max(max_residual, residual)
        if ticker in tickers:
            downloaded_verified += 1
        else:
            local_verified += 1
    coverage = {t:int(signals.loc[signals.index < '2026-01-02',t].notna().sum()) for t in signals}
    # Raw source reference rows include some KRX/FX dates outside US sessions.
    # Stock MA coverage must agree; reference-only calendar differences are
    # disclosed rather than treated as stock history or silently filled.
    reference_calendar_differences = {}
    for ticker,details in meta['ticker'].items():
        if coverage[ticker] != details['combined_pre2026_valid_rows']:
            assert ticker in ['456600.KS','KRW=X']
            reference_calendar_differences[ticker] = {
                'actual_us_signal_sessions': coverage[ticker],
                'raw_source_valid_rows_in_metadata': details['combined_pre2026_valid_rows']}
    return {'cache_columns_independently_verified': local_verified,
            'yahoo_columns_independently_redownloaded': downloaded_verified,
            'maximum_overlap_relative_residual': max_residual,
            'reference_only_calendar_differences': reference_calendar_differences,
            'below_200_pre2026': {t:n for t,n in coverage.items() if n<200}}


def main():
    results = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    assert hash_file(OUT/'protocol.json') == results['protocol_sha256']
    for filename, expected in results['input_sha256'].items():
        assert hash_file(M.ROOT/filename) == expected, filename
    summary = pd.read_csv(OUT/'summary.csv')
    curves = pd.read_csv(OUT/'curves_krw.csv', index_col=0)
    entries = pd.read_csv(OUT/'entry_decisions.csv')
    monthly = pd.read_csv(OUT/'monthly.csv')
    rows = summary.to_dict('records')
    names = {f'{kind}{n}' for kind in ['SMA','EMA'] for n in range(1,201)}
    assert set(summary.name) == names|{'baseline'}
    assert set(curves.columns) == names|{'baseline'}
    assert len(summary) == 401 and not summary.name.duplicated().any()
    assert set(entries.name) == names
    assert np.array_equal(curves.SMA1, curves.baseline)
    assert np.array_equal(curves.EMA1, curves.baseline)
    old_curves = pd.read_csv(HERE/'curves_krw.csv', index_col=0)
    np.testing.assert_allclose(curves.baseline, old_curves.baseline, rtol=0, atol=1e-9)
    max_metric_error = 0.
    indexed = summary.set_index('name')
    for label, (start, finish) in protocol['periods'].items():
        returns, mdd = direct_metrics(curves, start, finish)
        for name in curves:
            for suffix, values in [('return_pct', returns), ('mdd_pct', mdd)]:
                error = abs(float(indexed.at[name,f'{label}_{suffix}'])-float(values[name]))
                max_metric_error = max(max_metric_error, error)
                assert error <= 1e-8, (name, label, suffix, error)
    for month in range(1,12):
        label = f'2026-{month:02d}'
        returns, mdd = direct_metrics(curves, label+'-01', label+'-31')
        part = monthly.loc[monthly.month == label].set_index('name')
        assert set(part.index) == set(curves.columns)
        for name in curves:
            assert abs(float(part.at[name,'return_pct'])-returns[name]) <= 1e-8
            assert abs(float(part.at[name,'mdd_pct'])-mdd[name]) <= 1e-8
    selected = own_winner(rows, 'front_return_pct')
    assert selected == results['selected_by_front_only']
    by_kind = {kind: own_winner(rows, 'front_return_pct', kind) for kind in ['SMA','EMA']}
    assert by_kind == results['selected_by_type_front_only']
    assert own_winner(rows,'back_return_pct') == results['best_back_hindsight']
    assert own_winner(rows,'full_return_pct') == results['best_full_hindsight']
    changed = copy.deepcopy(rows)
    for i,row in enumerate(changed):
        row.update(back_return_pct=(-1)**i*1e8, back_mdd_pct=-i*1e5,
                   full_return_pct=i*1e6, recent_return_pct=-i*1e6)
    assert G.select_front(changed) == selected
    assert all(G.select_front(changed,kind) == by_kind[kind] for kind in by_kind)
    base = indexed.loc['baseline']
    candidate = summary.loc[summary.name != 'baseline']
    rb = candidate.back_return_pct > base.back_return_pct+TOL
    mb = candidate.back_mdd_pct > base.back_mdd_pct+TOL
    fb = candidate.front_return_pct > base.front_return_pct+TOL
    assert results['counts'] == {'candidates':400,'back_return_better':int(rb.sum()),
                                 'back_return_and_mdd_better':int((rb&mb).sum()),
                                 'front_and_back_return_better':int((fb&rb).sum())}
    book = M.R.load_book(M.BASE/'inputs',M.END)
    warmup = pd.read_csv(OUT/'inputs/close_warmup.csv', index_col=0)
    assert warmup.index.min() >= '2025-01-01' and warmup.index.max() < book.days[0]
    signals = pd.concat([warmup.reindex(columns=book.cls.columns), book.cls])
    assert not signals.index.duplicated().any() and signals.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(signals.loc[book.days],book.cls,check_names=False)
    source_check = audit_warmup(warmup, signals)
    reason_counts = {}
    for event in entries.to_dict('records'):
        count = int(signals.loc[:event['signal_date'],event['ticker']].notna().sum())
        assert count == event['history_count']
        assert event['signal_date'] == book.days[book.days.index(event['session'])-1]
        close_finite, ma_finite = np.isfinite(event['close']), np.isfinite(event['ma'])
        allowed = bool_value(event['allowed'])
        assert allowed == (close_finite and ma_finite and event['close'] >= event['ma'])
        reason = ('allowed' if allowed else 'missing_close' if not close_finite else
                  'below_ma' if ma_finite else 'insufficient_history' if count < event['length'] else
                  'missing_recent_history')
        assert event['reason'] == reason
        reason_counts[reason] = reason_counts.get(reason,0)+1
    representatives = list(dict.fromkeys([selected,*by_kind.values(),results['best_back_hindsight'],
                       results['best_full_hindsight'],'SMA18','EMA51','SMA200','EMA200','SMA1','EMA1']))
    replay, replayed_paths = {}, {}
    for name in representatives:
        row = indexed.loc[name]
        average = independent_ma(signals,row.kind,int(row.length)).reindex(book.cls.index)
        events = []
        with M.entry_gate(book,row.kind,int(row.length),events,averages=average):
            path = M.S.run_model(book)
        rebuilt = M.S.model_index(book,path)
        curve_error = float(np.max(np.abs(rebuilt-curves[name].to_numpy())))
        np.testing.assert_allclose(rebuilt, curves[name],rtol=0,atol=1e-8)
        stored = entries.loc[entries.name == name].to_dict('records')
        assert len(stored) == len(events)
        ma_error = 0.
        for actual, expected in zip(events,stored):
            assert actual['ticker'] == expected['ticker']
            assert actual['session'] == expected['session']
            assert actual['allowed'] == bool_value(expected['allowed'])
            if actual['ma'] is None:
                assert pd.isna(expected['ma'])
            else:
                error = abs(actual['ma']-expected['ma'])
                ma_error = max(ma_error,error)
                assert error <= 1e-8
        for session,weights in path.rebalances.items():
            assert sum(weights.values()) <= 1+1e-10
            assert max(weights.values(),default=0.) <= .25+1e-10
            group = {}
            semis = 0.
            for key,weight in weights.items():
                group[book.sector[key]] = group.get(book.sector[key],0.)+weight
                if M.S.in_semis_budget(book.sector[key]):
                    semis += weight
            assert max(group.values(),default=0.) <= .6+1e-10
            assert semis <= .5+1e-10
        replay[name] = {'manual_ma_max_error':ma_error,'full_curve_max_error':curve_error,
                        'event_count':len(events),'rebalance_count':len(path.rebalances),
                        'entry_verdicts_match':True,'target_caps_passed':True}
        replayed_paths[name] = path
    baseline = M.S.run_model(book)
    winner = replayed_paths[results['best_back_hindsight']]
    schedule = sorted(baseline.rebalances)
    weekly_differences = []
    for i,day in enumerate(schedule):
        original = baseline.rebalances[day]
        filtered = winner.rebalances[day]
        changed_keys = [key for key in set(original)|set(filtered)
                        if abs(original.get(key,0.)-filtered.get(key,0.)) > 1e-10]
        if not changed_keys:
            continue
        lo = baseline.days.index(day)
        next_day = schedule[i+1] if i+1<len(schedule) else None
        hi = baseline.days.index(next_day) if next_day else None
        original_end = baseline.value[hi] if next_day else baseline.close[-1]
        filtered_end = winner.value[hi] if next_day else winner.close[-1]
        original_return = (original_end/baseline.value[lo]-1)*100
        filtered_return = (filtered_end/winner.value[lo]-1)*100
        weekly_differences.append({
            'from_session_open':day,
            'to_session_open':next_day,
            'to_last_close_if_no_next_open':baseline.days[-1] if not next_day else None,
            'baseline_usd_interval_return_pct':float(original_return),
            'filtered_usd_interval_return_pct':float(filtered_return),
            'local_usd_return_difference_pp':float(filtered_return-original_return),
            'changed_targets': {book.ticker[key]: {'baseline_weight':original.get(key,0.),
                                                  'filtered_weight':filtered.get(key,0.)}
                                for key in sorted(changed_keys)}})
    best_front = max(row['front_return_pct'] for row in rows if row['name'] != 'baseline')
    front_ties = [row for row in rows if row['name'] != 'baseline'
                  and best_front-row['front_return_pct'] <= TOL]
    output = {'status':'passed','grid_candidates':400,'stored_curve_count':401,
              'metric_cells_checked':401*len(protocol['periods'])*2,
              'monthly_metric_cells_checked':401*11*2,
              'max_stored_metric_error':max_metric_error,'one_day_noop_verified':True,
              'frozen_baseline_verified':True,'front_selection_verified':selected,
              'front_selection_unchanged_after_back_score_mutation':True,
              'front_maximum_ties': {'all':len(front_ties),
                                     **{kind:sum(r['kind']==kind for r in front_ties) for kind in ['SMA','EMA']}},
              'all_entry_rows_checked':len(entries),'all_entry_reason_counts':reason_counts,
              'warmup':source_check,'manual_replays':replay,
              'hindsight_best_changed_rebalance_intervals':weekly_differences,
              'weekly_attribution_note':'Each interval compares the actual target differences at the same opens, in USD. These local percentage-point differences do not add to the compounded KRW total. This explains the hindsight winner, not an ex-ante rule.',
              'limit':'Manual MA and target-cap replay covers listed representative paths; all-grid arithmetic metrics and all entry reason/timing rows are checked.'}
    (OUT/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
