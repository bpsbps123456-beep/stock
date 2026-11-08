"""Freeze a broad stock panel without calculating or selecting performance.

Primary membership is the Jan 2 TIME/KoAct US-security union. Current AI map is
a separately flagged retrospective supplement. GOOG is collapsed into GOOGL.
Only new research inputs are written; operating caches and settings are read-only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE/'cross_stock_20261109'/'inputs'
sys.path.insert(0,str(ROOT))
import watchlist

FIELDS = {'open':'Open','close':'Close','adj_close':'Adj Close'}
BEGIN, START, SIGNAL, END = '2025-01-01','2026-01-05','2026-01-02','2026-11-04'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(ticker):
    return 'GOOGL' if ticker == 'GOOG' else ticker.replace('/','-')


def load_universe():
    members, snapshots = {}, {}
    for fund in ['456600','2ETFQ1']:
        path = ROOT/'data/normalized/etf_holdings'/f'2026-01-02_{fund}.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        assert data['as_of_date'] == SIGNAL
        members[fund] = {canonical(h['code'].split()[0]) for h in data['holdings']
                         if h['asset_type']=='security' and h['code'].endswith(' US EQUITY')}
        snapshots[fund] = {'path':str(path.relative_to(ROOT)),'sha256':sha(path),
                           'as_of_date':data['as_of_date'],'retrieved_at':data['retrieved_at'],
                           'historical_reconstruction':True,
                           'us_security_count':len(members[fund])}
    ai = {canonical(ticker):(sector,subsector)
          for sector,subs in watchlist.AI_UNIVERSE.items()
          for subsector,tickers in subs.items() for ticker in tickers if '.' not in ticker}
    names = sorted(set(ai)|members['456600']|members['2ETFQ1'])
    assert len(names)==140 and len(set.union(*members.values()))==63 and len(ai)==115
    return names,members,ai,snapshots


def discovery_holdings():
    spec = importlib.util.spec_from_file_location('cross_stock_original_model',HERE/'grid_review.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    book = module.M.R.load_book(module.M.BASE/'inputs',module.M.END)
    path = module.M.S.run_model(book)
    baseline = {canonical(book.ticker[key]) for weights in path.weights
                for key,weight in weights.items() if weight>0}
    entry_path = HERE/'exit_1_200/entry_decisions.csv'
    entries = pd.read_csv(entry_path)
    all_candidates = baseline|{canonical(x) for x in entries.loc[
        entries.allowed&(entries.proposed_weight>0),'ticker']}
    assert len(baseline)==27 and len(all_candidates)==32
    return baseline,all_candidates,{'path':str(entry_path.relative_to(ROOT)),
                                  'sha256':sha(entry_path),
                                  'definition':'union of baseline actual holdings and all 400 exit-grid allowed positive-weight new entries'}


def calendar():
    paths = [ROOT/'data/derived/analysis/strategy_lab/holdout/prices_close.csv',
             ROOT/'data/derived/analysis/concentration_review_20261105/inputs/prices_close.csv']
    days = set()
    sources = []
    for path in paths:
        data = pd.read_csv(path,index_col=0,usecols=['Date','QQQ'])
        days.update(data.loc[(data.index>=BEGIN)&(data.index<=END)&data.QQQ.notna()].index)
        sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),
                        'calendar_only_column':'QQQ'})
    index = pd.Index(sorted(days),name='Date')
    assert len(index[(index>=START)&(index<=END)]) == 211
    return index,sources


def extract(raw,tickers):
    result = {}
    for suffix,label in FIELDS.items():
        if isinstance(raw.columns,pd.MultiIndex):
            try:
                frame = raw.xs(label,level=0,axis=1)
            except KeyError:
                frame = pd.DataFrame(index=raw.index)
        else:
            frame = raw[[label]].rename(columns={label:tickers[0]}) if label in raw else pd.DataFrame(index=raw.index)
        frame = frame.copy()
        frame.index = pd.DatetimeIndex(frame.index)
        if frame.index.tz is not None:
            frame.index = frame.index.tz_localize(None)
        frame.index = frame.index.strftime('%Y-%m-%d')
        frame.index.name = 'Date'
        result[suffix] = frame.reindex(columns=tickers).sort_index()
    return result


def download(tickers):
    kwargs = {'start':BEGIN,'end':'2026-11-05','auto_adjust':False,'actions':False,
              'repair':False,'keepna':True,'progress':False,'threads':8,'timeout':30}
    attempts = []
    try:
        raw = yf.download(tickers,**kwargs)
        frames = extract(raw,tickers)
        attempts.append({'tickers':tickers,'exception':None})
    except Exception as error:
        attempts.append({'tickers':tickers,'exception':repr(error)})
        frames = {field:pd.DataFrame(columns=tickers,dtype=float) for field in FIELDS}
    missing = [ticker for ticker in tickers if any(frames[field][ticker].dropna().empty for field in FIELDS)]
    # A single sequential retry records vendor failures separately from an IPO's
    # genuine incomplete history. No price is forward/backward filled.
    for ticker in missing:
        try:
            raw = yf.download([ticker],**{**kwargs,'threads':False})
            retry = extract(raw,[ticker])
            combined_index = frames['close'].index.union(retry['close'].index).sort_values()
            for field in FIELDS:
                frames[field] = frames[field].reindex(combined_index)
                frames[field][ticker] = retry[field][ticker].reindex(combined_index)
            attempts.append({'tickers':[ticker],'exception':None})
        except Exception as error:
            attempts.append({'tickers':[ticker],'exception':repr(error)})
    return frames,{'provider':'Yahoo Finance via yfinance','yfinance_version':yf.__version__,
                   'kwargs':kwargs,'attempts':attempts,
                   'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
                   'definition':'split-adjusted, dividend-unadjusted Open/Close; Adj Close separately; explicit auto_adjust=False'}


def validity(frames,ticker,index):
    annual = index[(index>=START)&(index<=END)]
    signal_index = index[index<=SIGNAL]
    close = frames['close'][ticker].reindex(signal_index)
    valid = np.isfinite(close)&(close>0)
    recent = valid.iloc[-132:]
    count = int(valid.sum())
    missing = {}
    for field in FIELDS:
        values = frames[field][ticker].reindex(annual)
        missing[field] = int((~np.isfinite(values)|(values<=0)).sum())
    return count,len(recent)==132 and bool(recent.all()),missing


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tickers,members,ai,snapshots = load_universe()
    baseline,discovery,discovery_meta = discovery_holdings()
    index,calendar_sources = calendar()
    fresh,download_meta = download(tickers)
    source_files = {}
    for field,frame in fresh.items():
        path = OUT/f'source_{field}.csv'
        frame.to_csv(path,index_label='Date')
        source_files[field] = {'path':path.name,'sha256':sha(path),'rows':len(frame),'columns':len(frame.columns)}
    # Full-ticker fallback uses the old frozen fields together, in one split
    # unit system. Never mix a fresh prefix with a stale later segment.
    frozen_dir = ROOT/'data/derived/analysis/concentration_review_20261105/inputs'
    frozen = {field:pd.read_csv(frozen_dir/f'prices_{field}.csv',index_col=0) for field in FIELDS}
    chosen = {field:frame.reindex(index=index,columns=tickers) for field,frame in fresh.items()}
    fallback,rows = [],[]
    for ticker in tickers:
        count,recent,missing = validity(chosen,ticker,index)
        source = 'fresh_yfinance'
        # Fallback is allowed only when it provides a completely usable common
        # panel that the fresh source did not; exclusions remain explicit.
        if (not recent or missing['open'] or missing['close']) and all(ticker in f.columns for f in frozen.values()):
            old = {field:frame.reindex(index=index,columns=[ticker]) for field,frame in frozen.items()}
            old_count,old_recent,old_missing = validity(old,ticker,index)
            if old_recent and not old_missing['open'] and not old_missing['close']:
                for field in FIELDS:
                    chosen[field][ticker] = old[field][ticker]
                fallback.append(ticker)
                source = 'whole_ticker_frozen_fallback'
                count,recent,missing = old_count,old_recent,old_missing
        reasons = []
        if count<132:
            reasons.append('fewer_than_132_valid_closes_by_2026_01_02')
        elif not recent:
            reasons.append('missing_or_nonpositive_close_in_last_132_sessions')
        if missing['open']:
            reasons.append('incomplete_or_nonpositive_execution_open_2026')
        if missing['close']:
            reasons.append('incomplete_or_nonpositive_valuation_close_2026')
        origin = ai.get(ticker,('',''))
        rows.append({'ticker':ticker,'in_time_jan2':ticker in members['456600'],
                     'in_koact_jan2':ticker in members['2ETFQ1'],
                     'in_primary_jan2':ticker in members['456600']|members['2ETFQ1'],
                     'in_ai_current':ticker in ai,'sector':watchlist.resolve_sector(ticker),
                     'ai_sector':origin[0],'ai_subsector':origin[1],
                     'baseline_overlap':ticker in baseline,'discovery_overlap':ticker in discovery,
                     'eligible':not reasons,'exclusion_reasons':';'.join(reasons),
                     'valid_pre_start_closes':count,'last_132_close_sessions_valid':recent,
                     'missing_open_2026':missing['open'],'missing_close_2026':missing['close'],
                     'missing_adj_close_2026':missing['adj_close'],'data_source':source})
    universe = pd.DataFrame(rows)
    universe_path = OUT/'universe.csv'
    universe.to_csv(universe_path,index=False)
    selected_files = {}
    for field,frame in chosen.items():
        for reference in ['QQQ','KRW=X']:
            frame[reference] = frozen[field][reference].reindex(index)
        path = OUT/f'prices_{field}.csv'
        frame.to_csv(path,index_label='Date')
        selected_files[field] = {'path':path.name,'sha256':sha(path),'rows':len(frame),'columns':len(frame.columns)}
    calendar_path = OUT/'calendar.csv'
    pd.DataFrame({'Date':index,'evaluation_session':index>=START}).to_csv(calendar_path,index=False)
    counts = {'all':len(universe),'eligible_all':int(universe.eligible.sum()),
              'primary':int(universe.in_primary_jan2.sum()),
              'primary_eligible':int((universe.in_primary_jan2&universe.eligible).sum()),
              'ai_current':int(universe.in_ai_current.sum()),
              'ai_current_eligible':int((universe.in_ai_current&universe.eligible).sum()),
              'eligible_discovery_overlap':int((universe.eligible&universe.discovery_overlap).sum()),
              'eligible_external_to_all400':int((universe.eligible&~universe.discovery_overlap).sum()),
              'primary_eligible_external_to_all400':int((universe.eligible&universe.in_primary_jan2&~universe.discovery_overlap).sum())}
    meta = {'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'purpose':'price/universe preparation only; no performance-based eligibility or period selection',
            'period':{'source_begin':BEGIN,'signal_cutoff':SIGNAL,'first_trade_session':START,'last_valuation_session':END,
                      'evaluation_us_sessions':211,'minimum_valid_pre_start_closes':132},
            'canonical_aliases':{'GOOG':'GOOGL'},'universe_definition':{
                'primary':'Jan 2 TIME/KoAct US EQUITY security union; GOOG collapsed into GOOGL',
                'secondary':'all current watchlist.AI_UNIVERSE tickers without non-US suffixes; retrospective membership',
                'eligibility':'132 positive finite closing prices through Jan2, every last132 US close valid, complete positive Open/Close on every Jan5-Nov4 US session; Adj Close missingness recorded separately',
                'completeness_limitation':'Full-period availability is a retrospective data-completeness condition; delistings, unavailable histories and exclusions are listed, not treated as zero-return survivors.'},
            'watchlist':{'path':'watchlist.py','sha256':sha(ROOT/'watchlist.py')},
            'membership_snapshots':snapshots,'discovery_overlap':discovery_meta,
            'calendar_sources':calendar_sources,'calendar_sha256':sha(calendar_path),
            'reference_columns':{'tickers':['QQQ','KRW=X'],
                'definition':'original frozen concentration values reindexed to the wider signal calendar; earlier unavailable reference values remain missing',
                'source_sha256':{str((frozen_dir/f'prices_{field}.csv').relative_to(ROOT)):
                                sha(frozen_dir/f'prices_{field}.csv') for field in FIELDS}},
            'download':download_meta,'fresh_source_files':source_files,
            'actual_input_files':selected_files,'universe_sha256':sha(universe_path),
            'fallback_tickers':fallback,
            'fallback_sources':{str((frozen_dir/f'prices_{field}.csv').relative_to(ROOT)):
                                sha(frozen_dir/f'prices_{field}.csv') for field in FIELDS} if fallback else {},
            'counts':counts,'exclusions':universe.loc[~universe.eligible].to_dict('records'),
            'no_fill':True,'operating_caches_modified':False}
    (OUT/'input_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert all(p.stat().st_size<10_000_000 for p in OUT.iterdir() if p.is_file())
    print(json.dumps({'counts':counts,'fallback_tickers':fallback,'excluded':meta['exclusions'],
                      'files':selected_files},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
