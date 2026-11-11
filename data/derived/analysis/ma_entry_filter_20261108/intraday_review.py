"""Session anchored 4..16h exits; scheduling adapter to the canonical wealth engine.

Raw prices come from intraday_inputs.py, never from TradingView. The existing frozen
daily book remains authoritative for weekly ranks, allocations, opens and NAV marks.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from unittest.mock import patch

import exchange_calendars as xc
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'intraday_4_16h_20261111'
SPEC = importlib.util.spec_from_file_location('intraday_cross_stock_helpers', HERE/'cross_stock_review.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
M, S = C.M, C.S
ET = 'America/New_York'
START, END = '2026-01-02', '2026-11-04'
PERIODS = {'front': (START, '2026-05-29'), 'back': ('2026-06-01', END), 'full': (START, END)}
RTOL, TIE = 1e-12, 1e-9


class IndexedDays(list):
    """Immutable session labels with O(1) lookup for the existing turnover helper."""
    def __init__(self,values):
        super().__init__(values)
        self.positions = {v:i for i,v in enumerate(values)}

    def index(self,value,*args):
        return super().index(value,*args) if args else self.positions[value]


def session_windows(days, mode):
    if mode not in ('regular', 'extended'):
        raise ValueError(mode)
    cal = xc.get_calendar('XNYS', start=pd.Timestamp(min(days))-pd.Timedelta(days=7),
                         end=pd.Timestamp(max(days))+pd.Timedelta(days=7))
    schedule = cal.schedule.loc[min(days):max(days)]
    rows = []
    for day in days:
        r = schedule.loc[day]
        o, c = r['open'].tz_convert(ET), r['close'].tz_convert(ET)
        if mode == 'extended':
            o = o.normalize()+pd.Timedelta(hours=4)
            c = c.normalize()+pd.Timedelta(hours=17 if c.hour == 13 else 20)
        rows.append({'session': day, 'open': o, 'close': c})
    return pd.DataFrame(rows).set_index('session')


def bar_windows(windows, hours):
    if not isinstance(hours, int) or hours < 1:
        raise ValueError(hours)
    rows, elapsed = [], 0.
    for day, r in windows.iterrows():
        a, stop = r['open'], r['close']
        while a < stop:
            z = min(a+pd.Timedelta(hours=hours), stop)
            length = (z-a).total_seconds()/3600
            rows.append({'session': day, 'start': a, 'end': z,
                         'u_start': elapsed, 'u_end': elapsed+length})
            elapsed += length
            a = z
    result = pd.DataFrame(rows)
    return result.set_index('end', drop=False)


def aggregate_closes(raw, windows, hours):
    """Use only prints inside [start,end); an empty bar is absent for that symbol."""
    meta = bar_windows(windows, hours)
    starts, ends = pd.DatetimeIndex(meta.start).as_unit('ns').asi8, meta.index.as_unit('ns').asi8
    result = {}
    for ticker, bars in raw.items():
        t = bars.index.as_unit('ns').asi8
        pos = np.searchsorted(ends, t, side='right')
        safe = np.minimum(pos, len(meta)-1)
        valid = (pos < len(meta)) & (t >= starts[safe]) & (t < ends[safe])
        s = pd.Series(bars.Close.to_numpy(float)[valid], index=pos[valid])
        result[ticker] = s.groupby(level=0).last().reindex(range(len(meta))).to_numpy(float)
    return pd.DataFrame(result, index=meta.index), meta


def awake(ts):
    return 8 <= pd.Timestamp(ts).tz_convert('Asia/Seoul').hour < 24


def signal_matrix(closes, kind, length, events):
    """Latest *completed actual* bar per symbol, never a forming/empty candle."""
    ev = pd.DatetimeIndex(events).as_unit('ns').asi8
    prices = np.full((len(events), len(closes.columns)), np.nan)
    means = prices.copy()
    stamp = np.full(prices.shape, np.iinfo(np.int64).min, dtype=np.int64)
    for j, ticker in enumerate(closes):
        s = closes[ticker].dropna()
        if s.empty:
            continue
        average = M.moving_average(s, kind, length)
        source_ns = s.index.as_unit('ns').asi8
        ix = np.searchsorted(source_ns, ev, side='right')-1
        valid = ix >= 0
        prices[valid,j] = s.to_numpy(float)[ix[valid]]
        means[valid,j] = average.to_numpy(float)[ix[valid]]
        stamp[valid,j] = source_ns[ix[valid]]
    return prices, means, stamp


def timing_stats(meta, kind, length):
    """Smoothing expressed in elapsed session hours, including missing-print time."""
    u = pd.Series(meta.u_end.to_numpy(float), index=meta.index)
    av = M.moving_average(u, kind, length)
    lag = u-av
    span = u-pd.Series(meta.u_start.to_numpy(float), index=meta.index).shift(length-1)
    valid = (meta.index.strftime('%Y-%m-%d') >= START) & lag.notna()
    values = lag.loc[valid]
    return {'mean_lag_hours_median': float(values.median()),
            'mean_lag_hours_min': float(values.min()), 'mean_lag_hours_max': float(values.max()),
            'window_hours_median': float(span.loc[valid].median())}


def timing_samples(closes,meta):
    u = meta.u_end.to_numpy(float)
    starts = meta.u_start.to_numpy(float)
    evaluation = meta.index.as_unit('ns').asi8 >= pd.Timestamp(START,tz=ET).value
    return [(pd.Series(u[valid]),starts[valid],evaluation[valid])
            for valid in (closes[t].notna().to_numpy() for t in closes) if valid.any()]


def actual_timing_stats(closes,meta,kind,length,prepared=None):
    """Pool actual symbol bars; absent prints lengthen time, never compress the clock."""
    lags,spans = [],[]
    for u,starts,evaluation in (timing_samples(closes,meta) if prepared is None else prepared):
        if len(u) < length:
            continue
        lag = u.to_numpy(float)-M.moving_average(u,kind,length).to_numpy(float)
        mask = evaluation & np.isfinite(lag)
        lags.append(lag[mask])
        spans.append((u.to_numpy(float)[length-1:]-starts[:len(u)-length+1])[mask[length-1:]])
    lag,span = np.concatenate(lags),np.concatenate(spans)
    return {'mean_lag_hours_median':float(np.median(lag)),
            'mean_lag_hours_min':float(np.min(lag)), 'mean_lag_hours_max':float(np.max(lag)),
            'window_hours_median':float(np.median(span))}


def load_inputs():
    spec = importlib.util.spec_from_file_location('ma_intraday_inputs', HERE/'intraday_inputs.py')
    inputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inputs)
    daily = M.R.load_book(M.BASE/'inputs', END)
    raw = {ticker: inputs.load_bars(ticker) for ticker in sorted(set(daily.ticker.values()))}
    return daily, raw


def event_book(daily, raw, mode):
    """Insert price events without changing the canonical open-to-open arithmetic.

    16:00 close marks precede any after-hours order (by 1ms). Past marks may carry
    forward for NAV only; trade_ok separately requires a real executable opening print.
    """
    days = [d for d in daily.days if START <= d <= END]
    windows = session_windows(days, mode)
    regular = session_windows(days, 'regular')
    event_rows, prior_week = [], None
    old_week = S.iso_week
    for day in days:
        w = old_week(day)
        first = w != prior_week
        ro, rc = regular.loc[day, ['open','close']]
        points = {ro: 'open', rc: 'close'}
        # All 30m opens are needed: the first open after a signal can be absent
        # for one symbol, which must then wait for its next actual opening print.
        a, end = windows.loc[day, ['open','close']]
        while a < end:
            points.setdefault(a, 'trade')
            a += pd.Timedelta(minutes=30)
        if mode == 'extended':
            points[rc+pd.Timedelta(milliseconds=1)] = 'trade'
        for ts, role in sorted(points.items()):
            if day == END and ts > rc:
                continue
            week = prior_week if first and ts < ro and prior_week is not None else w
            event_rows.append({'time': ts, 'session': day, 'role': role,
                               'week': week, 'weekly_open': first and role == 'open'})
        prior_week = w
    meta = pd.DataFrame(event_rows)
    # Seed one previous mark, allowing the same FX base and past-only mark carry.
    previous_day = daily.days[daily.days.index(days[0])-1]
    previous_ts = pd.Timestamp(previous_day+' 16:00', tz=ET)
    seed = pd.DataFrame([{'time': previous_ts, 'session': previous_day, 'role': 'seed',
                          'week': old_week(previous_day), 'weekly_open': False}])
    meta = pd.concat([seed,meta], ignore_index=True)
    ix = pd.DatetimeIndex(meta.time)
    matrix = pd.DataFrame(np.nan, index=ix, columns=sorted(set(daily.ticker.values())))
    trade_ok = np.zeros(matrix.shape, bool)
    di = {d: i for i,d in enumerate(daily.days)}
    daily_rows = np.asarray([di[d] for d in meta.session])
    mark_rows = meta.role.isin(['close','seed']).to_numpy()
    open_rows = meta.role.eq('open').to_numpy()
    trade_rows = meta.role.isin(['open','trade']).to_numpy()
    for j,ticker in enumerate(matrix):
        bars = raw[ticker]
        # Our event points always coincide with 30m boundaries, apart from close+1ms.
        query = ix.floor('30min')
        vals = bars.Open.reindex(query).to_numpy(float,copy=True)
        vals[mark_rows] = daily.cls[ticker].to_numpy(float)[daily_rows[mark_rows]]
        vals[open_rows] = daily.opn[ticker].to_numpy(float)[daily_rows[open_rows]]
        trade_ok[:,j] = np.isfinite(vals) & (vals > 0) & trade_rows
        matrix[ticker] = vals
    # A stale mark can value a held position, but cannot pass trade_ok.
    matrix = matrix.ffill()
    b = copy.copy(daily)
    b.days = IndexedDays([t.isoformat() for t in ix])
    matrix.index = b.days
    b.opn = matrix
    b.cls = matrix
    b.adj = matrix
    b.fx = np.asarray([daily.fx[di[d]] for d in meta.session])
    meta['event'] = b.days
    meta['awake'] = [awake(t) for t in ix]
    meta['trade_event'] = meta.role.isin(['open','trade'])
    # Integrate post-decision holdings through the selected session, never overnight.
    durations = np.zeros(len(meta))
    times_ns = ix.as_unit('ns').asi8
    for k in range(len(meta)):
        day = meta.session.iloc[k]
        if k+1 < len(meta) and day == meta.session.iloc[k+1]:
            durations[k] = (times_ns[k+1]-times_ns[k])/3.6e12
        elif day != END and meta.role.iloc[k] == 'trade':
            stop = windows.loc[day,'close']
            durations[k] = max(0.,(stop-meta.time.iloc[k]).total_seconds()/3600)
    meta['duration_hours'] = durations
    b._intraday_records = meta.to_dict('records')
    b._intraday_rank_cache = {}
    with C.cached_columns(daily):
        for r in b._intraday_records:
            if r['weekly_open']:
                d = r['session']
                b._intraday_rank_cache[d] = S.momentum_ranking(daily,di[d],d)
    return b, meta, trade_ok


@contextmanager
def schedule_adapter(daily, book, meta, p, a, stamps, policy, trade_ok, logs):
    original_target = S.weekly_target
    di = {d:i for i,d in enumerate(daily.days)}
    ticker_col = {t:j for j,t in enumerate(book.opn)}
    row = getattr(book,'_intraday_records',None)
    if row is None:
        row = meta.to_dict('records')
    week = dict(zip(book.days,meta.week))
    cache = {}
    original_intraday = S._intraday
    original_ranking = S.momentum_ranking

    def ranking(b,k,session):
        if b is daily and session in book._intraday_rank_cache:
            return book._intraday_rank_cache[session]
        return original_ranking(b,k,session)

    def target(b,k,event,current):
        r = row[k]
        if not r['weekly_open']:
            return {'pdf':None, 'ranked':[], 'selected':[], 'raw':{}, 'target':{}}
        d = r['session']
        key = (d,frozenset(current))
        if key not in cache:
            cache[key] = original_target(daily,di[d],d,current)
        decision = cache[key]
        if p is None:
            for item in set(current) | set(decision['target']):
                if not trade_ok[k,ticker_col[b.ticker[item]]]:
                    raise ValueError(f'Weekly rebalance has no executable open: {event} {item}')
            return decision
        values = {key: (p[k,ticker_col[b.ticker[key]]], a[k,ticker_col[b.ticker[key]]])
                  for key in decision['selected']}
        gated, blocked = M.apply_gate(decision,current,values,comparison_rtol=RTOL)
        for item in set(current) | set(gated['target']):
            if not trade_ok[k,ticker_col[b.ticker[item]]]:
                raise ValueError(f'Weekly rebalance has no executable open: {event} {item}')
        for key in decision['selected']:
            if key not in current:
                j = ticker_col[b.ticker[key]]
                logs['entries'].append({'event':event, 'session':d, 'ticker':b.ticker[key],
                    'key':key, 'close':float(p[k,j]), 'ma':float(a[k,j]),
                    'signal_ns':int(stamps[k,j]), 'allowed':key not in blocked})
        return gated

    def exits(b,k,event,weights):
        if p is None or not row[k]['trade_event'] or (policy == 'awake' and not row[k]['awake']):
            return []
        result = []
        for key in weights:
            j = ticker_col[b.ticker[key]]
            c,av = p[k,j], a[k,j]
            if not (np.isfinite(c) and np.isfinite(av)):
                logs['unknown'] += 1
            elif C.E.below_ma(c,av,comparison_rtol=RTOL) and trade_ok[k,j]:
                assert stamps[k,j] <= pd.Timestamp(event).value
                result.append(key)
                logs['exits'].append({'event':event, 'session':row[k]['session'],
                    'ticker':b.ticker[key], 'key':key, 'close':float(c), 'ma':float(av),
                    'signal_ns':int(stamps[k,j]), 'price':S._price_at(b,key,k)})
        return result

    def intraday(b,keys,k):
        # Every inserted event is an instantaneous mark, so open and close are
        # literally the same frame. Let the canonical drift process an empty move.
        return {} if b.opn is b.cls else original_intraday(b,keys,k)

    with patch.object(S,'weekly_target',target), patch.object(S,'iso_week',lambda d:week[d]), \
         patch.object(S,'momentum_ranking',ranking), patch.object(S,'_intraday',intraday), C.cached_columns(book):
        yield exits


def run_variant(daily,book,meta,trade_ok,closes,kind,length,policy,signals=None):
    p,a,stamp = (signals if signals is not None else signal_matrix(closes,kind,length,meta.time)) if length else (None,None,None)
    logs = {'entries':[], 'exits':[], 'unknown':0}
    with schedule_adapter(daily,book,meta,p,a,stamp,policy,trade_ok,logs) as callback:
        path = S.run_model(book,start=START,exit_at_open=callback if length else None)
    return path,logs


def summarize(daily,book,meta,path,logs):
    k0 = book.days.index(path.days[0])
    ev = meta.iloc[k0:].reset_index(drop=True)
    close_pos = np.flatnonzero(ev.role.eq('close'))
    days = ev.session.iloc[close_pos].tolist()
    # At the close marker the canonical event open == daily closing price.
    usd = path.value[close_pos]
    base_fx = daily.fx[daily.days.index(START)-1]
    fx = np.asarray([daily.fx[daily.days.index(d)] for d in days])
    krw = usd*fx/base_fx*100
    # All changes, including within-target partial weekly rebalances.
    with C.cached_columns(book):
        turns = C.E.turnover(book,path)
    tv = np.asarray([r['one_way_turnover'] for r in turns])
    exp = np.asarray([sum(w.values()) for w in path.weights])
    durations = ev.duration_hours.to_numpy(float)
    row = {'ma_exits':len(logs['exits']), 'unknown_held':logs['unknown'],
           'blocked_entries':sum(not e['allowed'] for e in logs['entries']),
           'new_entries':sum(e['side']=='buy' for e in path.events)}
    for label,(lo,hi) in PERIODS.items():
        i,z = C.interval(days,lo,hi)
        stats = M.R.metrics(np.r_[krw[i-1] if i else 100., krw[i:z+1]])
        row.update({f'{label}_{key}':value for key,value in stats.items()})
        mask = (ev.session >= lo) & (ev.session <= hi)
        dur = durations[mask]
        row[f'{label}_exposure_pct'] = float(np.dot(exp[mask],dur)/dur.sum()*100)
        row[f'{label}_turnover'] = float(tv[mask].sum())
        row[f'{label}_ma_exits'] = sum(lo <= e['session'] <= hi for e in logs['exits'])
    for bp in (5,10):
        adjusted = krw*np.cumprod(1.-tv*bp/10000)[close_pos]
        for label,(lo,hi) in PERIODS.items():
            i,z = C.interval(days,lo,hi)
            row[f'{label}_return_{bp}bp_pct'] = (adjusted[z]/(adjusted[i-1] if i else 100.)-1)*100
    return row,krw,usd,days


def choose(rows,require_mdd=False):
    base = next(r for r in rows if r['length'] == 0)
    candidates = [r for r in rows if r['length'] and
                  (not require_mdd or r['front_mdd_pct'] >= base['front_mdd_pct']-TIE)]
    best = max(r['front_return_pct'] for r in candidates)
    return min((r for r in candidates if best-r['front_return_pct'] <= TIE),
               key=lambda r:(r.get('mode')!='regular',r['kind']!='SMA',r['hours'],r['length']))


def name(mode,policy,hours,kind,length):
    return f'{mode}_{policy}_{hours}h_{kind}{length}' if length else f'{mode}_{policy}_baseline'


def worker(task):
    mode,hours = task
    began = time.perf_counter()
    daily,raw = load_inputs()
    windows = session_windows(daily.days,mode)
    closes,bars = aggregate_closes(raw,windows,hours)
    book,meta,trade_ok = event_book(daily,raw,mode)
    OUT.mkdir(exist_ok=True)
    parts = OUT/'parts'
    parts.mkdir(exist_ok=True)
    # Stable hashes and binary cache make subsequent selected-path audit independent.
    closes.to_parquet(OUT/'cache'/f'closes_{mode}_{hours}h.parquet')
    rows,curves = [],{}
    expected = S.model_index(daily,S.run_model(daily))
    clock_inputs = timing_samples(closes,bars)
    for kind in ('SMA','EMA'):
        for length in range(1,121):
            lag = actual_timing_stats(closes,bars,kind,length,prepared=clock_inputs)
            signals = signal_matrix(closes,kind,length,meta.time)
            for policy in ('awake','unrestricted'):
                path,logs = run_variant(daily,book,meta,trade_ok,closes,kind,length,policy,signals=signals)
                row,krw,usd,days = summarize(daily,book,meta,path,logs)
                if length == 1:
                    np.testing.assert_allclose(krw,expected,rtol=0,atol=1e-8)
                    assert not logs['exits'] and not row['blocked_entries']
                key = name(mode,policy,hours,kind,length)
                rows.append({'name':key,'mode':mode,'policy':policy,'hours':hours,'kind':kind,
                             'length':length,**lag,**row})
                curves[key] = krw
            if length % 30 == 0:
                print(f'{mode} {hours}h {kind} {length}/120 complete',flush=True)
        print(f'{mode} {hours}h {kind}: 120 lengths x2 execution policies complete',flush=True)
    pd.DataFrame(rows).to_csv(parts/f'{mode}_{hours}h_summary.csv',index=False)
    pd.DataFrame(curves,index=days).to_csv(parts/f'{mode}_{hours}h_curves.csv',index_label='session',float_format='%.10g')
    print(f'{mode} {hours}h finished in {time.perf_counter()-began:.1f}s',flush=True)
    return mode,hours


def combine():
    daily,raw = load_inputs()
    rows,curves = [],{}
    books = {mode:event_book(daily,raw,mode) for mode in ('regular','extended')}
    for mode in ('regular','extended'):
        for policy in ('awake','unrestricted'):
            book,meta,ok = books[mode]
            path,logs = run_variant(daily,book,meta,ok,None,'BASE',0,policy)
            row,krw,usd,days = summarize(daily,book,meta,path,logs)
            original = S.model_index(daily,S.run_model(daily))
            np.testing.assert_allclose(krw,original,rtol=0,atol=1e-8)
            key = name(mode,policy,0,'BASE',0)
            rows.append({'name':key,'mode':mode,'policy':policy,'hours':0,'kind':'BASE','length':0,**row})
            curves[key] = krw
        hours_list = range(4,8) if mode == 'regular' else range(4,17)
        for hours in hours_list:
            part = pd.read_csv(OUT/'parts'/f'{mode}_{hours}h_summary.csv')
            rows.extend(part.to_dict('records'))
            cv = pd.read_csv(OUT/'parts'/f'{mode}_{hours}h_curves.csv',index_col=0)
            curves.update({key:cv[key].to_numpy(float) for key in cv})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT/'summary_unique.csv',index=False)
    # Explicitly retain all requested 6240 nominal aliases for each policy.
    aliases = []
    ix = summary.set_index('name')
    for mode in ('regular','extended'):
        for policy in ('awake','unrestricted'):
            for h in range(4,17):
                h0 = min(h,7) if mode == 'regular' else h
                for kind in ('SMA','EMA'):
                    for n in range(1,121):
                        source = name(mode,policy,h0,kind,n)
                        aliases.append({'requested':name(mode,policy,h,kind,n),'source':source})
    pd.DataFrame(aliases).to_csv(OUT/'candidate_aliases.csv',index=False)
    selections = {}
    for mode in ('regular','extended'):
        for policy in ('awake','unrestricted'):
            group = summary[(summary['mode']==mode)&(summary.policy==policy)].to_dict('records')
            for constrained in (False,True):
                k = f'{mode}_{policy}'+('_risk_floor' if constrained else '')
                selections[k] = choose(group,constrained)['name']
    pooled = {}
    for policy in ('awake','unrestricted'):
        group = summary[summary.policy==policy].to_dict('records')
        for constrained in (False,True):
            pooled[policy+('_risk_floor' if constrained else '')] = choose(group,constrained)['name']
    # Store curves in small session/policy partitions rather than a >10MB file.
    for mode in ('regular','extended'):
        for policy in ('awake','unrestricted'):
            part = {k:v for k,v in curves.items() if k.startswith(f'{mode}_{policy}_')}
            pd.DataFrame(part,index=days).to_csv(OUT/f'curves_{mode}_{policy}.csv',index_label='session',float_format='%.10g')
    selected_names = set(selections.values())
    for mode in ('regular','extended'):
        for policy in ('awake','unrestricted'):
            group = summary[(summary['mode']==mode)&(summary.policy==policy)&(summary.length>0)]
            selected_names.update(group.sort_values(field,ascending=False).iloc[0]['name']
                                  for field in ('back_return_pct','full_return_pct'))
    # Detailed execution ledgers for front-selected and explicitly hindsight paths.
    for key in sorted(selected_names):
        r = ix.loc[key]
        mode,policy,h,kind,n = r['mode'],r.policy,int(r.hours),r.kind,int(r.length)
        book,meta,ok = books[mode]
        closes = pd.read_parquet(OUT/'cache'/f'closes_{mode}_{h}h.parquet')
        path,logs = run_variant(daily,book,meta,ok,closes,kind,n,policy)
        for field in ('exits','entries'):
            pd.DataFrame(logs[field]).to_csv(OUT/f'{key}_{field}.csv',index=False)
    result = {'periods':PERIODS,'selected_front_only':selections,'selected_front_pooled':pooled,
              'unique_candidates_per_execution_policy':4080,
              'nominal_candidates_per_execution_policy':6240,
              'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
              'input_metadata_sha256':hashlib.sha256((OUT/'input_metadata.json').read_bytes()).hexdigest(),
              'baseline_expansion_check':'four expanded daily-close curves match frozen original within 1e-8 index point'}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--combine-only',action='store_true')
    ap.add_argument('--task',nargs=2,metavar=('MODE','HOURS'))
    args = ap.parse_args()
    if args.combine_only:
        combine()
    elif args.task:
        worker((args.task[0],int(args.task[1])))
    else:
        tasks = [('regular',h) for h in range(4,8)]+[('extended',h) for h in range(4,17)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker,tasks))
        combine()


if __name__ == '__main__':
    main()
