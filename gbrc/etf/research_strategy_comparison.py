"""Fixed strategies, common US open execution / same-day close marks. No search.

Run: py -m gbrc.etf.research_strategy_comparison
Only writes strategy_comparison/ beside the existing US experiment.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .research import OUT
from .research_engine import percentile, price_features, build_rankings, config_grid
from .research_us_only import _hash
from .research_validation import _one_metrics
from .storage import atomic_write_json, atomic_write_text

IDS = (201609, 246798, 201608, 201610, 239120, 240024)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def disclosure_indices(dates, snapshot_dates, delay):
    """PDF first visible on first US session on/after its date, plus delay sessions."""
    available = np.searchsorted(dates, snapshot_dates, side='left') + delay
    return np.searchsorted(available, np.arange(len(dates)), side='right') - 1


def schedules(dates, snapshot_index=None):
    daily = np.ones(len(dates), dtype=bool)
    if snapshot_index is not None:
        daily = (snapshot_index >= 0) & (snapshot_index != np.r_[-1, snapshot_index[:-1]])
    weekly = np.zeros(len(dates), dtype=bool)
    previous = None
    for t, day in enumerate(dates):
        week = date.fromisoformat(str(day)).isocalendar()[:2]
        if daily[t] and week != previous:
            weekly[t] = True
            previous = week
    return {'daily': daily, 'weekly': weekly}


def momentum120(adjusted_close, execution_indices):
    """Use only closes ending strictly before each execution index."""
    if np.min(execution_indices) < 121: raise ValueError('Missing MOM120 warmup')
    with np.errstate(divide='ignore',invalid='ignore'):
        return adjusted_close[execution_indices-1]/adjusted_close[execution_indices-121]-1


def our_signals(arrays, config, si):
    """The four frozen signal families, using existing percentile/features helpers.

    Snapshot availability alone changes; price features always end before today.
    Full-fund flow values and original KRX flow-base dates remain untouched.
    """
    prices = arrays['prices']; T, N = prices.shape
    sd = arrays['snapshot_dates'].astype(str)
    wfund = arrays['weights'][np.maximum(si, 0)].copy(); wfund[si < 0] = 0
    lb = config['lookback']; li = list(arrays['lookbacks']).index(lb)
    ff = arrays['flow_pp'][np.maximum(si, 0), li].copy(); ff[si < 0] = np.nan
    kk = arrays['flow_kind'][np.maximum(si, 0), li].copy(); kk[si < 0] = np.nan
    mix = config['time_mix']; w = mix*wfund[:, 0] + (1-mix)*wfund[:, 1]
    prior = np.vstack([np.full(N, np.nan), prices[:-1]])
    priced = np.isfinite(prior) & (prior > 0); held = (w > 0) & priced
    mom, ma, vol = price_features(prices)
    mr = percentile(mom, (wfund[:, 0]+wfund[:, 1] > 0) & priced)
    wr = percentile(w, held)
    flow = ff[:, 0] if mix == 1 else ff[:, 1] if mix == 0 else mix*ff[:, 0]+(1-mix)*ff[:, 1]
    valid = np.isfinite(ff[:, 0]) if mix == 1 else np.isfinite(ff[:, 1]) if mix == 0 else np.isfinite(ff).all(axis=1)
    fr = percentile(flow, held & valid)
    buy = ((kk[:, 0] > 0) & (mix > 0)) | ((kk[:, 1] > 0) & (mix < 1))
    family = config['family']
    if family == 'sold_reversal':
        sell = ((kk[:, 0] < 0) & (mix > 0)) | ((kk[:, 1] < 0) & (mix < 1))
        score, eligible = wr+.1*mr, held & valid & sell & np.isfinite(mom)
    elif family == 'joint_buy':
        score, eligible = fr+.1*wr, held & valid & (kk[:, 0] > 0) & (kk[:, 1] > 0)
    elif family == 'new_entry':
        new = ((kk[:, 0] == 2) & (mix > 0)) | ((kk[:, 1] == 2) & (mix < 1))
        score, eligible = wr+.1*fr, held & valid & new
    elif family == 'repeated_buy':
        count = np.zeros((T, N))
        for t, s in enumerate(si):
            if s >= lb:
                bd = arrays['flow_base_dates'][s, li, 0]
                base = int(np.searchsorted(sd, bd)) if bd else s
                dk = arrays['flow_kind'][base+1:s+1, 0]
                count[t] = mix*np.sum(dk[:, 0] > 0, axis=0)+(1-mix)*np.sum(dk[:, 1] > 0, axis=0)
        score, eligible = percentile(count, held)+.1*fr, held & valid & (count >= 2) & (flow > 0)
    else:
        raise ValueError('Only predeclared frozen families are supported')
    if config['gate'] == 'positive_momentum': eligible &= mom > 0
    if config['gate'] == 'above_ma20': eligible &= ma > 0
    order = np.argsort(-np.where(eligible, score, -np.inf), axis=1, kind='stable')[:, :10]
    ok = np.take_along_axis(eligible, order, axis=1)
    basis = np.ones((T, 10))
    if config['weighting'] == 'conviction': basis = np.take_along_axis(w, order, axis=1)
    if config['weighting'] == 'inverse_volatility':
        iv = np.divide(1, np.maximum(vol, .001), out=np.zeros_like(vol), where=np.isfinite(vol))
        basis = np.take_along_axis(iv, order, axis=1)
    qm, qa, _ = price_features(arrays['qqq_usd'].reshape(-1, 1))
    down = (qm[:, 0] < -.03) & (qa[:, 0] < 0)
    exposure = np.ones(T)
    if config['risk_rule'] == 'trend_half': exposure[down] = .5
    if config['risk_rule'] == 'trend_cash': exposure[down] = 0
    return {'ranks': np.where(ok, order, -1), 'basis': np.where(ok, basis, 0),
            'exposure': exposure, 'schedule': schedules(arrays['dates'])[config['rebalance']],
            'max_stocks': config['max_stocks'], 'buffer': 0}


def claude_signals(arrays, si, momentum, label):
    T, N = arrays['prices'].shape
    w = arrays['weights'][np.maximum(si, 0), 0].copy(); w[si < 0] = 0
    prior = arrays['prior_close']
    # Previous-price eligibility avoids using today's close/missingness at the open.
    eligible = (w > 0) & np.isfinite(prior) & (prior > 0)
    score = w if label == 'TIME_clone' else momentum
    eligible &= np.isfinite(score)
    order = np.argsort(-np.where(eligible, score, -np.inf), axis=1, kind='stable')
    ok = np.take_along_axis(eligible, order, axis=1)
    basis = np.ones((T, N)) if label == 'H6e' else np.take_along_axis(w, order, axis=1)
    return {'ranks': np.where(ok, order, -1), 'basis': np.where(ok, basis, 0),
            'exposure': np.ones(T), 'schedule': schedules(arrays['dates'], si)['weekly' if label == 'H6e' else 'daily'],
            'max_stocks': N if label == 'TIME_clone' else 5, 'buffer': 0 if label == 'TIME_clone' else 5}


def simulate_open(opens, closes, fx_open, fx_close, signal, start=0):
    """Cash-funded fractional USD shares. Overnight -> rebalance -> intraday close.

    Missing Open freezes an existing position and prevents new execution. A missing
    Close carries the latest known USD mark and is counted, never backfilled.
    """
    T, N = opens.shape
    shares = np.zeros(N); cash = 1.; last_mark = np.full(N, np.nan)
    if start: last_mark = pd.DataFrame(closes[:start]).ffill().iloc[-1].to_numpy()
    nav_before = 1.; initialized = False
    rets=[]; navs=[]; counts=[]; cash_pct=[]; max_weights=[]; turnovers=[]; holdings=[]
    missing_marks = 0; entries = 0; last_open_weights = np.zeros(N)
    for t in range(start, T):
        observed_open = np.isfinite(opens[t]) & (opens[t] > 0)
        open_mark = np.where(observed_open, opens[t], last_mark)
        if np.any((shares > 0) & ~np.isfinite(open_mark)):
            raise ValueError('Held position has no known open mark')
        values = np.where(shares > 0, shares*np.nan_to_num(open_mark)*fx_open[t], 0.)
        nav_open = cash+values.sum(); turnover = 0.
        if signal['schedule'][t] or not initialized:
            frozen = (shares > 0) & ~observed_open
            available = max(0, signal['max_stocks']-int(frozen.sum()))
            ranks = signal['ranks'][t]; basis = signal['basis'][t]
            picks=[]
            if signal['buffer']:
                for rank in ranks[:signal['max_stocks']+signal['buffer']]:
                    if rank >= 0 and shares[rank] > 0 and observed_open[rank] and len(picks)<available:
                        picks.append(int(rank))
            for k, rank in enumerate(ranks):
                if len(picks) >= available: break
                if rank >= 0 and observed_open[rank] and rank not in picks and basis[k] > 0:
                    picks.append(int(rank))
            wanted = np.where(frozen, values, 0.)
            if picks:
                b = np.array([basis[np.flatnonzero(ranks == j)[0]] for j in picks])
                budget = max(0, nav_open*signal['exposure'][t]-wanted.sum())
                wanted[picks] = budget*b/b.sum()
                initialized = True
            entries += int(np.sum((wanted > 1e-12) & (values <= 1e-12)))
            turnover = np.abs(wanted-values).sum()/nav_open
            shares[observed_open] = wanted[observed_open]/(opens[t, observed_open]*fx_open[t])
            cash = nav_open-wanted.sum()
            if cash < -1e-10: raise ValueError('Borrowed cash')
            cash = max(0., cash)
            last_open_weights = wanted/nav_open
        observed_close = np.isfinite(closes[t]) & (closes[t] > 0)
        last_mark = np.where(observed_open, opens[t], last_mark)
        last_mark = np.where(observed_close, closes[t], last_mark)
        missing_marks += int(np.sum((shares > 0) & ~observed_close))
        values_close = np.where(shares > 0, shares*np.nan_to_num(last_mark)*fx_close[t], 0.)
        nav = cash+values_close.sum()
        if not np.isfinite(nav) or nav <= 0: raise ValueError('Invalid NAV')
        rets.append(nav/nav_before-1); navs.append(nav); nav_before=nav
        counts.append(int((shares > 1e-14).sum())); cash_pct.append(cash/nav*100)
        max_weights.append(float(values_close.max()/nav*100)); turnovers.append(turnover)
        holdings.append([[int(j),float(values_close[j]/nav)] for j in np.flatnonzero(shares > 1e-14)])
    return {'returns': np.array(rets), 'nav': navs, 'counts': counts, 'cash_pct': cash_pct,
            'max_weight_pct': max_weights, 'turnovers': turnovers, 'holdings': holdings,
            'missing_held_close_marks': missing_marks, 'position_entries': entries}


def describe(path, dates):
    r = path['returns']; dates=np.asarray(dates)
    slices={'full': np.ones(len(dates), bool), 'train': dates < '2026-06-01',
            'validation': (dates >= '2026-06-01') & (dates < '2026-08-01'),
            'later': dates >= '2026-08-01'}
    return {'segments': {k:_one_metrics(r[m]) for k,m in slices.items() if m.any()},
            'monthly': {month:_one_metrics(r[np.char.startswith(dates, month)]) for month in sorted(set(d[:7] for d in dates))},
            'actual_max_stocks': max(path['counts']), 'actual_mean_stocks': float(np.mean(path['counts'])),
            'max_close_weight_pct': max(path['max_weight_pct']), 'mean_cash_pct': float(np.mean(path['cash_pct'])),
            'trading_days': int(np.count_nonzero(np.asarray(path['turnovers']) > 1e-10)),
            'one_way_turnover_nav': float(np.sum(path['turnovers'])), 'position_entries': path['position_entries'],
            'missing_held_close_marks': path['missing_held_close_marks'], 'daily_returns': r.tolist(),
            'daily_nav': path['nav'], 'daily_holdings': path['holdings']}


def self_tests():
    sig={'ranks':np.array([[0],[0],[0]]),'basis':np.ones((3,1)), 'exposure':np.ones(3),
         'schedule':np.array([True,False,False]),'max_stocks':1,'buffer':0}
    p=simulate_open(np.array([[10.],[20.],[30.]]),np.array([[15.],[25.],[24.]]),np.ones(3),np.ones(3),sig)
    np.testing.assert_allclose(p['nav'], [1.5,2.5,2.4])
    fresh=simulate_open(np.array([[10.],[20.],[30.]]),np.array([[15.],[25.],[24.]]),np.ones(3),np.ones(3),sig,start=1)
    np.testing.assert_allclose(fresh['nav'], [1.25,1.2])
    # Rotation sells the overnight gap at open, then receives only new-stock intraday.
    rot={**sig,'ranks':np.array([[0],[1],[1]]),'schedule':np.ones(3,bool)}
    q=simulate_open(np.array([[10.,10.],[20.,10.],[20.,20.]]),np.array([[10.,10.],[20.,20.],[20.,20.]]),np.ones(3),np.ones(3),rot)
    np.testing.assert_allclose(q['nav'], [1.,4.,4.])
    cashsig={**sig,'exposure':np.zeros(3)}
    q=simulate_open(np.ones((3,1)),np.ones((3,1)),np.array([1.,2.,3.]),np.array([2.,3.,4.]),cashsig)
    np.testing.assert_allclose(q['nav'],1.)
    d=np.array(['2026-01-02','2026-01-05','2026-01-06']); s=np.array(['2026-01-02','2026-01-03','2026-01-06'])
    np.testing.assert_array_equal(disclosure_indices(d,s,0),[0,1,2])
    np.testing.assert_array_equal(disclosure_indices(d,s,1),[-1,0,1])
    return {'overnight_and_intraday_accounting':True,'fresh_entry_no_preentry_gap':True,
            'rotation_at_open':True,'krw_cash_has_no_fx_gain':True,'publication_us_session_delay':True}


def run():
    source=OUT/'us_only_experiment'; lab=OUT.parent/'strategy_lab'; out=OUT/'strategy_comparison'
    out.mkdir(parents=True,exist_ok=True)
    files=[source/'bundle.npz',source/'metadata.json',source/'raw_prices.json',source/'configurations.json',source/'zero_cost/results.json',
           lab/'lab_panels.json',*[lab/f'prices_{field}.csv' for field in ('open','close','adj_close')]]
    hashes={str(p):_hash(p) for p in files}
    a=dict(np.load(source/'bundle.npz',allow_pickle=False)); meta=load(source/'metadata.json')
    dates=a['dates'].astype(str); configs=load(source/'configurations.json'); panels=load(lab/'lab_panels.json')
    companies=[s['company_key'] for s in meta['securities']]
    tickers=[panels['ticker'][c] for c in companies]
    frames={f:pd.read_csv(lab/f'prices_{f}.csv',index_col=0) for f in ('open','close','adj_close')}
    all_days=frames['close']['QQQ'].dropna().index.to_numpy(str)
    assert np.array_equal(all_days[(all_days>=dates[0]) & (all_days<=dates[-1])],dates)
    close_all=frames['close'].reindex(all_days)
    fx_all=close_all['KRW=X'].ffill()
    at=np.searchsorted(all_days,dates)
    fx_close=fx_all.iloc[at].to_numpy(); fx_open=fx_all.iloc[at-1].to_numpy()
    op=frames['open'].reindex(dates)[tickers].to_numpy(float)
    cl=frames['close'].reindex(dates)[tickers].to_numpy(float)
    prior_close=close_all[tickers].ffill().iloc[at-1].to_numpy(float)
    a['prior_close']=prior_close
    a['prices']=close_all[tickers].ffill().iloc[at].to_numpy(float)*fx_close[:,None]
    a['qqq_usd']=close_all['QQQ'].iloc[at].to_numpy(float)
    adj=frames['adj_close'].reindex(all_days)[tickers].to_numpy(float)
    mom=momentum120(adj,at)
    assert all(s['market']=='US' for s in meta['securities'])
    # No hidden disagreement in fund weights between the two cached datasets.
    for k,day in enumerate(a['snapshot_dates']):
        for fi,tag in enumerate('TK'):
            np.testing.assert_allclose(a['weights'][k,fi], [panels['panels'][tag]['weights'][day].get(c,0)/100 for c in companies],atol=1e-12)
    protocol={
        'created_before_comparison_execution':datetime.now(timezone.utc).isoformat(),
        'period':[str(dates[0]),str(dates[-1])],'us_sessions':len(dates),'fixed_candidate_ids':list(IDS),
        'claude_fixed':['H6e','H6','TIME_clone'],'selection':'None; all candidates fixed before common-engine results',
        'research_status':'Retrospective comparison after both studies were examined; no independent untouched validation',
        'delay_definition':'PDF D first available at US session on/after D; shift that availability by 0/1/2 US sessions. All price features end at previous US close, unshifted.',
        'zero_delay_assumption':'Same-date PDF is available before US open; historical publication timestamps are not verified.',
        'native_delay_difference':'Claude original apply_delay shifts decision rows and their old scores; this comparison delays disclosures only by actual US sessions.',
        'execution':'Prior close signals; old shares earn overnight gap; rebalance fractional dollar allocations at observed regular-session Open; mark same-day Close; explicit trading cost0.',
        'fx':'KRW cash. Open conversions use previous US-session daily KRW=X Close as known-price proxy, not verified contemporaneous open FX. Close valuations use same-day daily FX Close.',
        'price_basis':'All P&L uses matching Yahoo split-reflected raw Open/Close; dividends and taxes excluded. No AdjClose/rawClose dividend factor in execution prices.',
        'drawdown':'Daily end-of-US-session KRW NAV drawdown including initial NAV1; not intraday or open-auction peak drawdown.',
        'signal_warmup':'Claude MOM120 retains AdjClose signal data from2025-05; our fixed20-day indicators retain2026-only warmup. Different information histories are disclosed.',
        'schedule':'Claude daily/weekly trades only on new available PDF; weekly first new-PDF session of ISO week. Our calendar daily/weekly schedule preserved. Cash starts enter at first eligible signal even midweek.',
        'cash_start':'Fresh June/August runs restart shares/cash but preserve past information; no pre-entry overnight gain. Continuous subperiod metrics retain inherited holdings.',
        'position_limit':'Fixed H6/H6e and our candidates <=10; TIME_clone full-US normalized fundweight is a reference and exempt from10-stock limit; not ETF share performance.',
        'proxy_changes':[{'company':companies[i],'old_symbol':str(a['symbols'][i]),'common_ticker':t} for i,t in enumerate(tickers) if t!=a['symbols'][i]],
        'share_class_note':'GOOG and GOOGL are distinct share classes. Both strategies use GOOGL as the merged-company proxy here; earlier our simulation used GOOG.',
        'original_endpoint_difference':'Claude original2026 runs endSep22; this common comparison endsSep23. Native reproduction stored separately.',
        'execution_limit':'Same-open dollar allocation uses aggregate opening prices to compute NAV and fractional shares; not exact previously submitted quantities or guaranteed actual fills. Actual slippage/spread/FX costs are unverified, not subtracted.',
        'source_sha256':hashes,
        'comparison_code_sha256':_hash(Path(__file__)),
    }
    atomic_write_json(out/'protocol.json',protocol)
    print('Common open comparison protocol saved; fixed candidates only',flush=True)
    audit=self_tests()
    raw=load(source/'raw_prices.json'); compared=0
    for j,ticker in enumerate(tickers):
        if ticker not in raw: continue
        for t,day in enumerate(dates):
            if str(day) in raw[ticker] and np.isfinite(cl[t,j]):
                np.testing.assert_allclose(cl[t,j],raw[ticker][str(day)],rtol=1e-7)
                compared+=1
    audit['matching_split_close_observations_across_caches']=compared
    # Verify the fixed-family extraction matches the original engine exactly under
    # its original strict-before disclosure contract, including repeated-buy bases.
    strict=np.searchsorted(a['snapshot_dates'].astype(str),dates,side='left')-1
    native=build_rankings(a); params,_=config_grid(native['rank_configs'])
    for index in IDS:
        sig=our_signals(a,configs[index],strict); rid,_,_,wi,ri=params[index]
        np.testing.assert_array_equal(sig['ranks'],native['ranks'][rid])
        if wi: np.testing.assert_allclose(sig['basis'],native['conviction' if wi==1 else 'inverse_vol'][rid],rtol=1e-6)
        np.testing.assert_array_equal(sig['exposure'],native['exposure'][:,ri])
    audit['fixed_family_signals_match_original_engine']=True
    # Mutating current/future Close cannot change current/predecessor signals.
    t=80; changed={**a,'prices':a['prices'].copy(),'qqq_usd':a['qqq_usd'].copy()}
    changed['prices'][t:]*=7; changed['qqq_usd'][t:]*=.1
    si=disclosure_indices(dates,a['snapshot_dates'].astype(str),0)
    for index in IDS:
        s1=our_signals(a,configs[index],si); s2=our_signals(changed,configs[index],si)
        for key in ('ranks','basis','exposure'): np.testing.assert_allclose(s1[key][:t+1],s2[key][:t+1])
    audit['future_close_does_not_change_prior_or_current_signals']=True
    future={**a,'weights':a['weights'].copy(),'flow_pp':a['flow_pp'].copy(),'flow_kind':a['flow_kind'].copy()}
    for key in ('weights','flow_pp','flow_kind'): future[key][si[t]+1:]*=-17
    for index in IDS:
        s1=our_signals(a,configs[index],si); s2=our_signals(future,configs[index],si)
        for key in ('ranks','basis','exposure'): np.testing.assert_allclose(s1[key][:t+1],s2[key][:t+1])
    changed_adj=adj.copy();changed_adj[at[t]:]*=31
    np.testing.assert_allclose(mom[:t+1],momentum120(changed_adj,at)[:t+1])
    for label in ('H6e','H6','TIME_clone'):
        c1=claude_signals(a,si,mom,label);c2=claude_signals(future,si,momentum120(changed_adj,at),label)
        for key in ('ranks','basis','exposure'): np.testing.assert_allclose(c1[key][:t+1],c2[key][:t+1])
    audit['future_pdf_and_momentum_closes_do_not_change_prior_or_current_signals']=True
    # Existing ETF references use KRX Close on the shared calendar labels, not
    # simultaneous US Open/Close executions. Preserve this distinction explicitly.
    original_bench=load(source/'zero_cost/results.json')['benchmarks']
    results={'protocol':protocol,'dates':dates.tolist(),'tickers':tickers,'strategies':{},
             'reference_etfs':{k:v for k,v in original_bench.items() if k in ('time','koact')},
             'reference_etf_timing':'KRX Close references from the original zero-cost study; intraday valuation/execution times differ from US-share paths.'}
    for lag in (0,1,2):
        si=disclosure_indices(dates,a['snapshot_dates'].astype(str),lag)
        signals={str(i):our_signals(a,configs[i],si) for i in IDS}
        signals.update({name:claude_signals(a,si,mom,name) for name in ('H6e','H6','TIME_clone')})
        for name,sig in signals.items():
            path=simulate_open(op,cl,fx_open,fx_close,sig)
            row=describe(path,dates)
            row['fresh_starts']={}
            for label,day in [('june','2026-06-01'),('august','2026-08-01')]:
                start=int(np.searchsorted(dates,day)); fresh=simulate_open(op,cl,fx_open,fx_close,sig,start)
                row['fresh_starts'][label]={'date':str(dates[start]),**describe(fresh,dates[start:])}
            row['latest_snapshot_dates']=[str(a['snapshot_dates'][s]) if s>=0 else None for s in si]
            if name!='TIME_clone' and row['actual_max_stocks']>10: raise ValueError('Stock limit exceeded')
            results['strategies'].setdefault(name,{'config':configs[int(name)] if name.isdigit() else {'name':name,'max_stocks':sig['max_stocks'],'buffer':sig['buffer']},'scenarios':{}})['scenarios'][f'publication_lag{lag}']=row
            print(f'{name} disclosure+{lag}: full {row["segments"]["full"]["return_pct"]:.3f}% freshAug {row["fresh_starts"]["august"]["segments"]["full"]["return_pct"]:.3f}%',flush=True)
    audit.update({'original_files_unchanged':{str(p):_hash(p) for p in files}==hashes,
                  'weights_match_claude_cached_panels':True,'us_only':True,'explicit_cost_zero':True,
                  'candidates_reselected':False,'candidate_max_stocks_le10':True,
                  'missing_held_close_marks':sum(s['missing_held_close_marks'] for p in results['strategies'].values() for s in p['scenarios'].values())})
    assert audit['original_files_unchanged']
    atomic_write_json(out/'results.json',results); atomic_write_json(out/'validation.json',audit)
    lines=['# 고정 전략 공통 시가 비교','','2026-01-02~09-23,182미국거래일. 매매비용 차감0,원화현금,시가체결 후당일종가평가. 새후보선택없음.',
           '공시지연0/1/2는 미국거래일 지연이다. Claude 원래 결정행지연/시가간회계와다르다. 당일PDF공개시각은미검증이다.','',
           '| 전략 | 공시지연 | 전체 | MDD | 6~7월 지속 | 8~9월 지속 | 6월 현금출발 | 8월 현금출발 | 실제최대/평균종목 | 최대비중 | 평균현금 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,p in results['strategies'].items():
        for scenario,r in p['scenarios'].items():
            m=r['segments'];fresh=r['fresh_starts']
            values=[m['full']['return_pct'],m['full']['max_drawdown_pct'],m['validation']['return_pct'],m['later']['return_pct'],fresh['june']['segments']['full']['return_pct'],fresh['august']['segments']['full']['return_pct']]
            lines.append(f'| {name} | {scenario[-1]} | '+' | '.join(f'{v:+.2f}%' for v in values)+f' | {r["actual_max_stocks"]}/{r["actual_mean_stocks"]:.2f} | {r["max_close_weight_pct"]:.2f}% | {r["mean_cash_pct"]:.2f}% |')
    lines+=['','TIME_clone은미국전체복제참조이며ETF직접보유수익률이나10종목제한후보가아니다. #239120/#240024는각1/2종목집중전략이다.',
            'Claude MOM120은2025년준비가격,우리20일지표는2026년자료만사용한다. GOOGL을회사대표가격으로통일했으며GOOG와다른주식이다.',
            '시가환전은직전US일FX종가를근사사용한다. 같은시가의NAV금액배분·소수수량체결이므로전날확정주문수량의실제계좌재현은아니다. 세금·배당제외,실제슬리피지·호가차이·환전비용미검증.',
            'MDD는시작NAV1을포함한미국일종가최대낙폭이며장중최대낙폭이아니다. ETF참조는기존KRX종가원화수익으로미국주식과일중평가시각이다르다.',
            '세부원천해시·가정은protocol.json,월별/지속구간/현금출발/일별보유는results.json,합성검증은validation.json. 원래연구재현은native_reproduction.json/md로별도구분한다.','']
    atomic_write_text(out/'common_report.md','\n'.join(lines))
    print(json.dumps(audit,ensure_ascii=False),flush=True)
    return results


if __name__=='__main__':
    run()
