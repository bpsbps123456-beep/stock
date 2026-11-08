"""Independent raw-price cash/quantity audit of all cross-stock research paths.

The stock simulator, scalar SMA sums, metric arithmetic and aggregation below do
not call the portfolio engine or runner. The engine is reused only to verify the
older seven-stock curves after the shared-callback refactor. No input is changed.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE/'cross_stock_20261109'
sys.path.insert(0,str(ROOT))
import watchlist


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(ticker):
    return 'GOOGL' if ticker=='GOOG' else ticker.replace('/','-')


def stats(curve,lo,hi):
    curve = np.asarray(curve)
    first = curve[lo-1] if lo else np.ones(curve.shape[1:])
    values = np.concatenate([np.asarray(first)[None],curve[lo:hi+1]],axis=0)
    return ((values[-1]/values[0]-1)*100,
            np.min(values/np.maximum.accumulate(values,axis=0)-1,axis=0)*100)


def raw_paths(opens,closes,prior_closes,means,weekly,specs,days,tickers):
    """Actual cash and share quantities; the only trades are all-in/all-out."""
    shape = (len(days),len(tickers),len(specs))
    close_nav,open_nav,exposure,turnover = [np.empty(shape) for _ in range(4)]
    signals,exit_counts = [],np.zeros((len(tickers),len(specs)),dtype=int)
    for s,(strategy,mode,length) in enumerate(specs):
        cash,quantity = np.ones(len(tickers)),np.zeros(len(tickers))
        entry_events,exit_events = [],[]
        for i,day in enumerate(days):
            open_nav[i,:,s] = cash+quantity*opens[i]
            was_held = quantity>0
            mean = means[length][i]
            sold = was_held&(prior_closes[i]<mean) if mode=='exit' else np.zeros(len(tickers),dtype=bool)
            if sold.any():
                cash[sold] += quantity[sold]*opens[i,sold]
                quantity[sold] = 0.
                exit_counts[sold,s] += 1
                if strategy=='exit_SMA126':
                    for j in np.flatnonzero(sold):
                        exit_events.append({'ticker':tickers[j],'strategy':strategy,'stage':'exit',
                            'session':day,'signal_date':days[i-1] if i else '2026-01-02',
                            'key':tickers[j],'close':prior_closes[i,j],'ma':mean[j],
                            'weight':1.,'open':opens[i,j]})
            available = quantity==0
            bought = np.zeros(len(tickers),dtype=bool)
            if weekly[i]:
                eligible = np.ones(len(tickers),dtype=bool) if mode=='buy_hold' else prior_closes[i]>=mean
                bought = available&eligible
                if strategy in ['entry_only_SMA126','exit_SMA126']:
                    for j in np.flatnonzero(available):
                        entry_events.append({'ticker':tickers[j],'strategy':strategy,'stage':'entry',
                            'session':day,'signal_date':days[i-1] if i else '2026-01-02',
                            'key':tickers[j],'close':prior_closes[i,j],'ma':mean[j],
                            'allowed':bool(eligible[j]),'proposed_weight':1.})
                quantity[bought] = cash[bought]/opens[i,bought]
                cash[bought] = 0.
            assert not (sold&bought).any()
            exposure[i,:,s] = quantity>0
            turnover[i,:,s] = sold.astype(int)+bought.astype(int)
            close_nav[i,:,s] = cash+quantity*closes[i]
        signals.extend(entry_events)
        signals.extend(exit_events)
    return close_nav,open_nav,exposure,turnover,exit_counts,signals


def main():
    started = time.perf_counter()
    result = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    meta = json.loads((OUT/'inputs/input_meta.json').read_text(encoding='utf-8'))
    assert digest(OUT/'protocol.json')==result['protocol_sha256']
    for name,sha in result['input_sha256'].items():
        assert digest(ROOT/name)==sha,name
    for name,sha in result['original_frozen_input_sha256'].items():
        assert digest(ROOT/name)==sha,name
    assert digest(ROOT/'watchlist.py')==meta['watchlist']['sha256']
    assert not meta['fallback_tickers'] and meta['no_fill']
    frames = {field:pd.read_csv(OUT/f'inputs/prices_{field}.csv',index_col=0)
              for field in ['open','close','adj_close']}
    for field in frames:
        source = pd.read_csv(OUT/f'inputs/source_{field}.csv',index_col=0)
        assert digest(OUT/f'inputs/source_{field}.csv')==meta['fresh_source_files'][field]['sha256']
        assert digest(OUT/f'inputs/prices_{field}.csv')==meta['actual_input_files'][field]['sha256']
        pd.testing.assert_frame_equal(frames[field][source.columns],source.reindex(frames[field].index))
        frozen_path = ROOT/f'data/derived/analysis/concentration_review_20261105/inputs/prices_{field}.csv'
        frozen = pd.read_csv(frozen_path,index_col=0)
        pd.testing.assert_frame_equal(frames[field][['QQQ','KRW=X']],
                                      frozen[['QQQ','KRW=X']].reindex(frames[field].index))
    universe = pd.read_csv(OUT/'inputs/universe.csv').set_index('ticker')
    assert len(universe)==140 and universe.index.is_unique
    funds = {}
    for fund in ['456600','2ETFQ1']:
        details = meta['membership_snapshots'][fund]
        assert digest(ROOT/details['path'])==details['sha256']
        snapshot = json.loads((ROOT/details['path']).read_text(encoding='utf-8'))
        assert snapshot['as_of_date']=='2026-01-02'
        funds[fund] = {canonical(h['code'].split()[0]) for h in snapshot['holdings']
                       if h['asset_type']=='security' and h['code'].endswith(' US EQUITY')}
    ai = {canonical(t) for subs in watchlist.AI_UNIVERSE.values() for syms in subs.values()
          for t in syms if '.' not in t}
    assert set(universe.index)==funds['456600']|funds['2ETFQ1']|ai
    for ticker,row in universe.iterrows():
        assert row.in_time_jan2==(ticker in funds['456600'])
        assert row.in_koact_jan2==(ticker in funds['2ETFQ1'])
        assert row.in_ai_current==(ticker in ai)
        assert row.sector==watchlist.resolve_sector(ticker)
    tickers = sorted(universe.index[universe.eligible])
    assert len(tickers)==138
    assert set(universe.index[~universe.eligible])=={'CBRS','XE'}
    dates = frames['close'].index
    assert dates.is_unique and dates.is_monotonic_increasing and dates.min()>='2025-01-01'
    days = list(dates[(dates>='2026-01-05')&(dates<='2026-11-04')])
    assert len(days)==211
    for ticker,row in universe.iterrows():
        pre = frames['close'].loc[:'2026-01-02',ticker]
        positive = np.isfinite(pre)&(pre>0)
        count = int(positive.sum())
        recent = len(positive.iloc[-132:])==132 and bool(positive.iloc[-132:].all())
        assert count==row.valid_pre_start_closes and recent==row.last_132_close_sessions_valid
        missing = {}
        for field in frames:
            values = frames[field].loc[days,ticker]
            missing[field] = int((~np.isfinite(values)|(values<=0)).sum())
            assert missing[field]==row[f'missing_{field}_2026']
        assert row.eligible==(count>=132 and recent and missing['open']==0 and missing['close']==0)
    overlap_path = ROOT/meta['discovery_overlap']['path']
    assert digest(overlap_path)==meta['discovery_overlap']['sha256']
    original_entries = pd.read_csv(overlap_path)
    discovered = {canonical(t) for t in original_entries.loc[
        original_entries.allowed&(original_entries.proposed_weight>0),'ticker']}
    # SMA1 is the complete frozen baseline path, so its positive entry ledger
    # includes every baseline name and provides an independent overlap check.
    assert len(discovered)==32
    assert all(row.discovery_overlap==(ticker in discovered) for ticker,row in universe.iterrows())
    groups = {}
    primary = funds['456600']|funds['2ETFQ1']
    definitions = {'jan2_all':primary,'jan2_external':primary-discovered,'jan2_overlap':primary&discovered,
                   'ai_all':ai,'ai_external':ai-discovered,'broad_all':set(universe.index),
                   'broad_external':set(universe.index)-discovered,'broad_overlap':set(universe.index)&discovered}
    for group,members in definitions.items():
        groups[group] = [t for t in tickers if t in members]
    assert groups==result['groups']
    assert result['eligible_stocks']==138 and result['settings_per_stock']==15 and result['path_count']==2070
    assert result['sensitivity_lengths']==list(range(120,133)) and result['primary_fixed_length']==126

    opens = frames['open'].loc[days,tickers].to_numpy(float)
    closes = frames['close'].loc[days,tickers].to_numpy(float)
    raw_closes = frames['close'][tickers].to_numpy(float)
    positions = [dates.get_loc(day)-1 for day in days]
    prior_closes = raw_closes[positions]
    means = {length:np.array([[math.fsum(raw_closes[k-length+1:k+1,j])/length
                               for j in range(len(tickers))] for k in positions])
             for length in range(120,133)}
    assert all(np.isfinite(mean).all() for mean in means.values())
    iso = [date.fromisoformat(day).isocalendar()[:2] for day in days]
    weekly = np.array([i==0 or iso[i]!=iso[i-1] for i in range(len(days))])
    specs = [('buy_hold','buy_hold',126),('entry_only_SMA126','entry_only',126)]+[
        (f'exit_SMA{length}','exit',length) for length in range(120,133)]
    names = [r[0] for r in specs]
    strategy_pos,ticker_pos = {s:i for i,s in enumerate(names)},{t:i for i,t in enumerate(tickers)}
    nav,opn,exposure,turnover,exit_counts,signals = raw_paths(
        opens,closes,prior_closes,means,weekly,specs,days,tickers)
    np.testing.assert_allclose(nav[:,:,0],closes/opens[0],rtol=0,atol=1e-12)
    fx = frames['close'].loc[days,'KRW=X'].to_numpy()/float(frames['close'].at['2026-01-02','KRW=X'])
    primary_curves = pd.read_csv(OUT/'primary_stock_curves.csv',index_col=0)
    assert list(primary_curves.index)==days and len(primary_curves.columns)==138*3
    curve_error = 0.
    for ticker in tickers:
        for strategy in ['buy_hold','entry_only_SMA126','exit_SMA126']:
            actual = nav[:,ticker_pos[ticker],strategy_pos[strategy]]
            error = float(np.max(np.abs(actual-primary_curves[ticker+'__'+strategy])))
            curve_error = max(curve_error,error)
            assert error<=1e-9,(ticker,strategy,error)
    expected_signals = pd.DataFrame(signals).sort_values(['ticker','strategy','stage','session']).reset_index(drop=True)
    stored_signals = pd.read_csv(OUT/'signals.csv').sort_values(['ticker','strategy','stage','session']).reset_index(drop=True)
    assert len(expected_signals)==len(stored_signals)
    signal_error = 0.
    for col in ['ticker','strategy','stage','session','signal_date','key']:
        assert expected_signals[col].equals(stored_signals[col]),col
    for col in ['close','ma','weight','open','proposed_weight']:
        left,right = expected_signals[col].to_numpy(float),stored_signals[col].to_numpy(float)
        np.testing.assert_allclose(left,right,rtol=0,atol=1e-8,equal_nan=True)
        signal_error = max(signal_error,float(np.nanmax(np.abs(left-right))))
    part = expected_signals.stage=='entry'
    assert all(bool(a)==(str(b).lower()=='true') for a,b in zip(
        expected_signals.loc[part,'allowed'],stored_signals.loc[part,'allowed']))
    assert (stored_signals.signal_date<stored_signals.session).all()
    assert all(weekly[days.index(d)] for d in stored_signals.loc[part,'session'])
    periods = {label:(days.index(next(d for d in days if d>=start)),
                       max(i for i,d in enumerate(days) if d<=finish))
               for label,(start,finish) in protocol['periods'].items()}
    ps = pd.read_csv(OUT/'per_stock.csv').set_index(['ticker','strategy','period'])
    assert len(ps)==138*15*3 and ps.index.is_unique
    metric_error,metric_cells = 0.,0
    def check(value,expected):
        nonlocal metric_error,metric_cells
        delta = abs(float(value)-float(expected))
        metric_error = max(metric_error,delta)
        metric_cells += 1
        assert delta<=1e-7,(value,expected,delta)
    direct,matched_direct = {},{}
    for label,(lo,hi) in periods.items():
        usd_ret,usd_dd = stats(nav,lo,hi)
        krw_ret,krw_dd = stats(nav*fx[:,None,None],lo,hi)
        fractions = exposure[lo:hi+1].mean(axis=0)
        base = np.concatenate([nav[lo-1,:,0][None] if lo else np.ones((1,len(tickers))),nav[lo:hi+1,:,0]],axis=0)
        daily = base[1:]/base[:-1]-1
        matched = np.concatenate([np.ones((1,len(tickers),15)),
                    np.cumprod(1.+daily[:,:,None]*fractions[None],axis=0)],axis=0)
        ctl_ret,ctl_dd = stats(matched,0,len(matched)-1)
        direct[label] = (krw_ret,krw_dd)
        matched_direct[label] = ctl_ret
        for j,ticker in enumerate(tickers):
            for s,strategy in enumerate(names):
                row = ps.loc[ticker,strategy,label]
                for field,values in [('usd_return_pct',usd_ret),('usd_mdd_pct',usd_dd),
                    ('krw_return_pct',krw_ret),('krw_mdd_pct',krw_dd),
                    ('usd_matched_exposure_return_pct',ctl_ret),('usd_matched_exposure_mdd_pct',ctl_dd)]:
                    check(row[field],values[j,s])
                check(row.base_krw_return_pct,krw_ret[j,0]);check(row.base_krw_mdd_pct,krw_dd[j,0])
                check(row.return_delta_pp,krw_ret[j,s]-krw_ret[j,0]);check(row.mdd_delta_pp,krw_dd[j,s]-krw_dd[j,0])
                check(row.mean_open_exposure_pct,fractions[j,s]*100)
                assert int(row.exits)==exit_counts[j,s]
    baskets = pd.read_csv(OUT/'basket_curves.csv',index_col=0)
    aggregate = pd.read_csv(OUT/'aggregate.csv')
    sector = pd.read_csv(OUT/'sector.csv')
    leave_out = pd.read_csv(OUT/'leave_one_sector_out.csv')
    cost_nav = {bps:nav*np.cumprod(1.-turnover*bps/10000.,axis=0) for bps in [5,10]}
    def check_aggregate(row,members):
        j = [ticker_pos[t] for t in members]
        s = strategy_pos[row.strategy]
        lo,hi = periods[row.period]
        u = nav[:,j,s].mean(axis=1)
        ur,ud = stats(u,lo,hi);kr,kd = stats(u*fx,lo,hi)
        assert int(row.stocks)==len(j)
        for field,value in [('usd_return_pct',ur),('usd_mdd_pct',ud),('krw_return_pct',kr),('krw_mdd_pct',kd)]:
            check(row[field],value)
        actual_exposure = (opn[:,j,s]*exposure[:,j,s]).sum(axis=1)/opn[:,j,s].sum(axis=1)
        check(row.mean_open_exposure_pct,actual_exposure[lo:hi+1].mean()*100)
        krw_ret,krw_dd = direct[row.period]
        dr,dd = krw_ret[j,s]-krw_ret[j,0],krw_dd[j,s]-krw_dd[j,0]
        check(row.median_return_delta_pp,np.median(dr))
        assert int(row.return_better_count)==int((dr>1e-9).sum())
        assert int(row.mdd_better_count)==int((dd>1e-9).sum())
        assert int(row.both_better_count)==int(((dr>1e-9)&(dd>1e-9)).sum())
        for bps in [5,10]:
            check(row[f'krw_return_{bps}bp_pct'],stats(cost_nav[bps][:,j,s].mean(axis=1)*fx,lo,hi)[0])
    assert len(aggregate)==8*15*3
    for _,row in aggregate.iterrows():
        check_aggregate(row,groups[row.group])
        if row.strategy=='exit_SMA126':
            j = [ticker_pos[t] for t in groups[row.group]]
            s = strategy_pos[row.strategy]
            lo,_ = periods[row.period]
            initial = nav[lo-1,j,s] if lo else np.ones(len(j))
            control = np.average(matched_direct[row.period][j,s],weights=initial)
            check(row.usd_matched_exposure_return_pct,control)
            check(row.usd_vs_matched_exposure_return_pp,row.usd_return_pct-control)
        else:
            assert pd.isna(row.usd_matched_exposure_return_pct)
            assert pd.isna(row.usd_vs_matched_exposure_return_pp)
    basket_error = 0.
    assert len(baskets.columns)==8*15 and list(baskets.index)==days
    for group,members in groups.items():
        j = [ticker_pos[t] for t in members]
        for s,strategy in enumerate(names):
            error = float(np.max(np.abs(nav[:,j,s].mean(axis=1)*fx-baskets[group+'__'+strategy])))
            basket_error = max(basket_error,error)
            assert error<=1e-9
    for _,row in sector.iterrows():
        members = [t for t in groups[row.group] if universe.at[t,'sector']==row.sector]
        check_aggregate(row,members)
    for _,row in leave_out.iterrows():
        members = [t for t in groups[row.group] if universe.at[t,'sector']!=row.excluded_sector]
        check_aggregate(row,members)
    neighbors = pd.read_csv(OUT/'neighbors.csv')
    assert list(neighbors.length)==list(range(120,133))
    for _,row in neighbors.iterrows():
        for group in ['jan2_external','broad_external']:
            expected = aggregate.loc[(aggregate.group==group)&(aggregate.strategy==f'exit_SMA{int(row.length)}')&
                                     (aggregate.period=='back')].iloc[0]
            check(row[group+'_return_pct'],expected.krw_return_pct)
            check(row[group+'_mdd_pct'],expected.krw_mdd_pct)
    contributions = pd.read_csv(OUT/'contributions.csv')
    assert len(contributions)==sum(len(x) for x in groups.values())
    contribution_error = 0.
    for group,members in groups.items():
        stored = contributions.loc[contributions.group==group].set_index('ticker')
        assert set(stored.index)==set(members)
        delta = [(nav[-1,ticker_pos[t],strategy_pos['exit_SMA126']]-nav[-1,ticker_pos[t],0])/len(members)*100 for t in members]
        np.testing.assert_allclose(stored.loc[members,'full_terminal_wealth_delta_pp'],delta,rtol=0,atol=1e-8)
        contribution_error = max(contribution_error,float(np.max(np.abs(stored.loc[members,'full_terminal_wealth_delta_pp']-delta))))
        j = [ticker_pos[t] for t in members]
        assert abs(sum(delta)-(nav[-1,j,strategy_pos['exit_SMA126']].mean()-nav[-1,j,0].mean())*100)<=1e-8
    print(f'Independent raw-share simulation:2070 paths,6210 stock rows,{len(signals)} signals and all aggregate tables passed.',flush=True)

    # Verify that extracting make_exit_callback preserved the previous results.
    spec = importlib.util.spec_from_file_location('prior_study_audit_helpers',HERE/'audit_grid.py')
    old = importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    spec = importlib.util.spec_from_file_location('shared_exit_regression',HERE/'exit_review.py')
    E = importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
    book = E.M.R.load_book(E.M.BASE/'inputs',E.M.END)
    old_signals = E.G.signal_history(book)
    old_curves = pd.read_csv(HERE/'exit_1_200/curves_krw.csv',index_col=0)
    baseline = E.S.run_model(book)
    regression = {'baseline':float(np.max(np.abs(E.S.model_index(book,baseline)-old_curves.baseline)))}
    for name in ['SMA126','EMA65','SMA2']:
        averages = old.independent_ma(old_signals,name[:3],int(name[3:])).reindex(book.cls.index)
        path,entries,exits,unknown = E.run_variant(book,old_signals,name[:3],int(name[3:]),averages=averages)
        regression[name] = float(np.max(np.abs(E.S.model_index(book,path)-old_curves[name])))
        assert not unknown
    assert all(error<=1e-8 for error in regression.values())
    output = {'status':'passed','eligible_stock_count':138,'excluded':['CBRS','XE'],
              'independent_raw_cash_quantity_paths':2070,'settings_per_stock':15,
              'scalar_sma_lengths_checked':list(range(120,133)),
              'scalar_sma_signal_cells':13*211*138,'daily_primary_stock_curves_checked':138*3,
              'max_daily_stock_curve_error':curve_error,'all_signal_rows_checked':len(signals),
              'max_signal_value_error':signal_error,'per_stock_period_rows_checked':len(ps),
              'aggregate_period_rows_checked':len(aggregate),'sector_rows_checked':len(sector),
              'leave_one_sector_out_rows_checked':len(leave_out),'basket_curves_checked':len(baskets.columns),
              'max_basket_curve_error':basket_error,'numeric_metric_cells_checked':metric_cells,
              'max_numeric_metric_error':metric_error,'contribution_rows_checked':len(contributions),
              'max_contribution_error':contribution_error,'original_seven_stock_refactor_curve_errors':regression,
              'group_counts':{g:len(v) for g,v in groups.items()},
              'front_back_carry_verified':True,'equal_initial_capital_basket_verified':True,
              'actual_open_nav_weighted_exposure_verified':True,'fx_and_cash_currency_verified':True,
              'five_and_ten_bp_weight_drag_verified':True,'source_and_protocol_hashes_verified':True,
              'matched_exposure_group_period_rows_checked':8*3,
              'limitations':[
                'This is transport to new securities using already observed2026 prices, not a new untouched time holdout. CurrentAI membership, historical reconstruction, full-period data availability and within-year discovery exclusion remain retrospective conditions.',
                'Stocks and the eight overlapping cohorts share market exposures; their improvement counts are descriptive and cannot be treated as138 independent trials.',
                'The experiment gives each stock a separate initial dollar, removes seven-stock selection/caps and allows only weekly reentry; it does not estimate the original portfolio or immediate daily reentry.',
                'Back-period baskets inherit all May-end stock NAV differences. Paired per-stock medians use equal stock weights and therefore differ from a basket return difference.',
                'Contribution CSV values are USD full-period terminal-wealth percentage points per equal initial dollar. KRW terminal differences require multiplying by the shared final FX factor; these are not back-period timing contributions.',
                'Five/ten-bp scenarios multiply sleeve wealth by traded-weight drag, including the initial trade; they are sensitivity approximations, not measured fills, FX conversion costs or tax/dividend simulations.',
                'No significance test or after-the-fact selection of a better neighboring length is established by this arithmetic audit.'
              ],'elapsed_seconds':time.perf_counter()-started}
    (OUT/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='limitations'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
