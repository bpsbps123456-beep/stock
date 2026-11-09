"""Independent cash/share and MA audit of the broad 1..120 SMA/EMA grid.

No portfolio, signal, metric or allocation function from the grid runner is used
to rebuild paths. Only frozen raw inputs and the predeclared groups are read.
Writes cross_stock_grid_1_120_20261109/audit.json, never an operating input.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'cross_stock_20261109'
OUT = HERE/'cross_stock_grid_1_120_20261109'
ROOT = HERE.parents[3]
TIE = 1e-9
ATOL = 1e-7


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(curve,lo,hi):
    first = curve[lo-1] if lo else np.ones(curve.shape[1:])
    values = np.concatenate([np.asarray(first)[None],curve[lo:hi+1]],axis=0)
    return ((values[-1]/values[0]-1)*100,
            np.min(values/np.maximum.accumulate(values,axis=0)-1,axis=0)*100)


def own_averages(raw,positions,kind):
    """SMA direct scalar fsum; EMA first-finite seed and ignore-NA recurrence."""
    lengths = np.arange(1,121)
    output = np.empty((len(positions),raw.shape[1],120))
    if kind=='SMA':
        # All eligible names have the last132 prior US sessions and every2026
        # session, so no invented missing values enter these direct sums.
        for i,k in enumerate(positions):
            for j in range(raw.shape[1]):
                output[i,j,0] = raw[k,j]
                for n in range(2,121):
                    sample = raw[k-n+1:k+1,j]
                    assert np.isfinite(sample).all()
                    output[i,j,n-1] = math.fsum(sample)/n
    else:
        state = np.full((raw.shape[1],120),np.nan)
        count = np.zeros(raw.shape[1],dtype=int)
        alpha = 2./(lengths+1.)
        output_positions = {k:i for i,k in enumerate(positions)}
        for k,values in enumerate(raw):
            valid = np.isfinite(values)
            first = valid[:,None]&~np.isfinite(state)
            update = (1.-alpha)*state+alpha*values[:,None]
            state = np.where(first,values[:,None],np.where(valid[:,None],update,state))
            count += valid
            if k in output_positions:
                output[output_positions[k]] = np.where(count[:,None]>=lengths,state,np.nan)
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(output[:,:,0],raw[positions])
    return output


def simulate(opens,closes,prior,means,weekly,comparison_rtol=1e-12):
    """Separate actual cash and stock quantities, all-in/out, no transfers."""
    shape = means.shape
    nav,open_nav = np.empty(shape),np.empty(shape)
    held,bought,sold = [np.empty(shape,dtype=bool) for _ in range(3)]
    cash,quantity = np.ones(shape[1:]),np.zeros(shape[1:])
    for i in range(len(opens)):
        o,c = opens[i,:,None],closes[i,:,None]
        open_nav[i] = cash+quantity*o
        delta = prior[i,:,None]-means[i]
        equal = np.abs(delta)<=comparison_rtol*np.maximum(np.abs(prior[i,:,None]),np.abs(means[i]))
        sell = (quantity>0)&(delta<0)&~equal
        cash = np.where(sell,cash+quantity*o,cash)
        quantity = np.where(sell,0.,quantity)
        buy = ((quantity==0)&((delta>=0)|equal) if weekly[i]
               else np.zeros(shape[1:],dtype=bool))
        assert not (sell&buy).any()
        quantity = np.where(buy,cash/o,quantity)
        cash = np.where(buy,0.,cash)
        nav[i] = cash+quantity*c
        held[i],bought[i],sold[i] = quantity>0,buy,sell
    return nav,open_nav,held,bought,sold


def own_picker(front,group,floor=False):
    subset = [row for row in front if row['group']==group and row['strategy']!='buy_hold']
    if floor:
        base = next(row for row in front if row['group']==group and row['strategy']=='buy_hold')
        subset = [row for row in subset if row['mdd_pct']>=base['mdd_pct']-TIE]
    best = max(row['return_pct'] for row in subset)
    tied = [row for row in subset if best-row['return_pct']<=TIE]
    return min(tied,key=lambda r:(r['kind']!='SMA',r['length']))['strategy'],len(tied)


def main():
    started = time.perf_counter()
    result = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    assert digest(OUT/'protocol.json')==result['protocol_sha256']
    assert protocol.get('numerical_equality'), 'Corrected numerical-equality policy must be recorded before results.'
    assert protocol['grid']=={'kinds':['SMA','EMA'],'minimum':1,'maximum':120,'step':1,
                              'candidates':240,'stocks':138,'candidate_paths':33120,'control_paths':138}
    assert digest(SOURCE/'inputs/input_meta.json')==protocol['frozen_input_meta_sha256']
    for name,sha in result['input_sha256'].items():
        assert digest(ROOT/name)==sha,name
    expected_comparison_sources = {
        str((SOURCE/'aggregate.csv').relative_to(ROOT)),
        str((HERE/'exit_1_200/summary.csv').relative_to(ROOT))}
    assert set(result['comparison_source_sha256'])==expected_comparison_sources
    for name,sha in result['comparison_source_sha256'].items():
        assert digest(ROOT/name)==sha,name
    prior_seven = pd.read_csv(HERE/'exit_1_200/summary.csv').set_index('name')
    seven_range = prior_seven.loc[(prior_seven.length>=2)&(prior_seven.length<=120)]
    seven_baseline = float(prior_seven.at['baseline','full_return_pct'])
    assert len(seven_range)==238
    assert (seven_range.full_return_pct<seven_baseline-TIE).all()
    seven_best = seven_range.sort_values(['full_return_pct','kind','length'],ascending=[False,True,True]).iloc[0]
    assert seven_best.name=='EMA90'
    seven_context = {
        'scope':'Previously observed original seven-stock portfolio study, Jan2-Nov4; separate from the broad independent-sleeve experiment and not a new grid holdout.',
        'source':'exit_1_200/summary.csv','length_range':[2,120],'candidate_count':238,
        'full_return_better_than_baseline_count':int((seven_range.full_return_pct>seven_baseline+TIE).sum()),
        'full_return_strictly_worse_than_baseline_count':int((seven_range.full_return_pct<seven_baseline-TIE).sum()),
        'baseline_full_return_pct':seven_baseline,
        'EMA6_full_return_pct':float(prior_seven.at['EMA6','full_return_pct']),
        'EMA6_back_return_pct':float(prior_seven.at['EMA6','back_return_pct']),
        'best_nonidentity_strategy_in_2_120':str(seven_best.name),
        'best_nonidentity_full_return_pct':float(seven_best.full_return_pct)}
    fixed_126 = pd.read_csv(SOURCE/'aggregate.csv')
    fixed_context = fixed_126.loc[(fixed_126.group.isin(['jan2_external','broad_external']))&
        (fixed_126.strategy.isin(['buy_hold','exit_SMA126']))&
        (fixed_126.period.isin(['back','full'])),
        ['group','strategy','period','krw_return_pct','krw_mdd_pct']].to_dict('records')
    assert len(fixed_context)==8
    fixed_result = json.loads((SOURCE/'results.json').read_text(encoding='utf-8'))
    fixed_audit = json.loads((SOURCE/'audit.json').read_text(encoding='utf-8'))
    assert fixed_audit['status']=='passed'
    for name,sha in fixed_result['input_sha256'].items():
        assert digest(ROOT/name)==sha,name
    universe = pd.read_csv(SOURCE/'inputs/universe.csv').set_index('ticker')
    tickers = sorted(universe.index[universe.eligible])
    assert len(universe)==140 and len(tickers)==138
    assert set(universe.index[~universe.eligible])=={'CBRS','XE'}
    groups = {group:list(members) for group,members in fixed_result['groups'].items()}
    groups['additional_external'] = sorted(set(groups['broad_external'])-set(groups['jan2_external']))
    assert len(groups['additional_external'])==61
    if 'groups' in result:
        assert result['groups']==groups
    frames = {field:pd.read_csv(SOURCE/f'inputs/prices_{field}.csv',index_col=0)
              for field in ['open','close']}
    dates = frames['close'].index
    days = list(dates[(dates>='2026-01-05')&(dates<='2026-11-04')])
    assert len(days)==211 and dates.is_unique and dates.is_monotonic_increasing
    opens = frames['open'].loc[days,tickers].to_numpy(float)
    closes = frames['close'].loc[days,tickers].to_numpy(float)
    assert np.isfinite(opens).all() and np.isfinite(closes).all() and (opens>0).all() and (closes>0).all()
    positions = [dates.get_loc(day)-1 for day in days]
    raw = frames['close'][tickers].to_numpy(float)
    assert np.isfinite(raw[positions[0]-131:positions[0]+1]).all()
    prior = raw[positions]
    iso = [date.fromisoformat(day).isocalendar()[:2] for day in days]
    weekly = np.array([i==0 or iso[i]!=iso[i-1] for i in range(len(days))])
    fx = frames['close'].loc[days,'KRW=X'].to_numpy()/float(frames['close'].at['2026-01-02','KRW=X'])
    periods = {label:(next(i for i,d in enumerate(days) if d>=start),
                     max(i for i,d in enumerate(days) if d<=finish))
               for label,(start,finish) in protocol['periods'].items()}
    assert set(periods)=={'front','back','full'}
    member_indices = {g:[tickers.index(t) for t in members] for g,members in groups.items()}
    ag = pd.read_csv(OUT/'aggregate.csv').set_index(['group','strategy','period'])
    assert len(ag)==9*241*3 and ag.index.is_unique
    baseline_ps = pd.read_csv(OUT/'per_stock_baseline.csv')
    assert len(baseline_ps)==138*3
    metric_error,curve_error,checked_cells = 0.,0.,0
    front_rows,selection_rows = [],[]
    def compare(actual,expected,label):
        nonlocal metric_error,checked_cells
        actual,expected = np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
        assert actual.shape==expected.shape,(label,actual.shape,expected.shape)
        error = float(np.max(np.abs(actual-expected))) if actual.size else 0.
        metric_error = max(metric_error,error)
        checked_cells += actual.size
        assert np.isfinite(actual).all() and error<=ATOL,(label,error)
    baseline_nav = (closes/opens[0])[:,:,None]
    baseline_open = (opens/opens[0])[:,:,None]
    baseline_hold = np.ones(baseline_nav.shape,dtype=bool)
    baseline_buy = np.zeros(baseline_nav.shape,dtype=bool);baseline_buy[0]=True
    baseline_sell = np.zeros(baseline_nav.shape,dtype=bool)
    base_stats = {label:stats(baseline_nav*fx[:,None,None],lo,hi) for label,(lo,hi) in periods.items()}
    old_ps = pd.read_csv(SOURCE/'per_stock.csv').set_index(['ticker','strategy','period'])
    sma120_regression_error = 0.
    primary_per_stock_rows = 0
    basket_columns = 0
    ma_seconds = {}
    for kind in ['BASE','SMA','EMA']:
        if kind=='BASE':
            names,lengths = ['buy_hold'],[0]
            nav,opn,held,buys,sells = baseline_nav,baseline_open,baseline_hold,baseline_buy,baseline_sell
            stored_curves = None
        else:
            names,lengths = [f'exit_{kind}{n}' for n in range(1,121)],list(range(1,121))
            t = time.perf_counter()
            means = own_averages(raw,positions,kind)
            ma_seconds[kind] = time.perf_counter()-t
            nav,opn,held,buys,sells = simulate(opens,closes,prior,means,weekly)
            del means
            np.testing.assert_allclose(nav[:,:,0],baseline_nav[:,:,0],rtol=0,atol=1e-12)
            assert held[:,:,0].all() and not sells[:,:,0].any()
            assert buys[0,:,0].all() and not buys[1:,:,0].any()
            stored_curves = pd.read_csv(OUT/f'basket_curves_{kind}.csv',index_col=0)
            assert list(stored_curves.index)==days
            assert len(stored_curves.columns)==len(groups)*121
            basket_columns += len(stored_curves.columns)
        turn = buys.astype(float)+sells.astype(float)
        cost_nav = {bps:nav*np.cumprod(1.-turn*bps/10000.,axis=0) for bps in [5,10]}
        for label,(lo,hi) in periods.items():
            if kind=='BASE':
                part = baseline_ps.loc[baseline_ps.period==label].copy()
            else:
                part = pd.read_csv(OUT/f'per_stock_{kind}_{label}.csv')
                primary_per_stock_rows += len(part)
            assert len(part)==len(tickers)*len(names)
            assert (part.period==label).all() and not part.duplicated(['ticker','strategy']).any()
            indexed = part.set_index(['ticker','strategy']).reindex(pd.MultiIndex.from_product([tickers,names]))
            assert indexed.length.tolist()==lengths*len(tickers)
            if kind!='BASE':
                assert (indexed.kind==kind).all()
            ur,ud = stats(nav,lo,hi);kr,kd = stats(nav*fx[:,None,None],lo,hi)
            expected = {'return_pct':kr,'mdd_pct':kd,'usd_return_pct':ur,'usd_mdd_pct':ud,
                        'mean_open_exposure_pct':held[lo:hi+1].mean(axis=0)*100,
                        'buys':buys[lo:hi+1].sum(axis=0),'sells':sells[lo:hi+1].sum(axis=0),
                        'ma_exits':sells[lo:hi+1].sum(axis=0),'one_way_turnover':turn[lo:hi+1].sum(axis=0)}
            for bps in [5,10]:
                expected[f'return_{bps}bp_pct'] = stats(cost_nav[bps]*fx[:,None,None],lo,hi)[0]
            for field,values in expected.items():
                compare(indexed[field].to_numpy(),values.ravel(),f'{kind}:{label}:{field}')
            # Independent constant-fraction daily-USD control for each stock,
            # initialized with actual strategy capital at the period boundary.
            base = np.concatenate([baseline_nav[lo-1,:,0][None] if lo else np.ones((1,len(tickers))),
                                   baseline_nav[lo:hi+1,:,0]],axis=0)
            fraction = held[lo:hi+1].mean(axis=0)
            controls = np.prod(1.+(base[1:]/base[:-1]-1.)[:,:,None]*fraction[None],axis=0)
            if 'usd_matched_exposure_return_pct' in indexed:
                compare(indexed.usd_matched_exposure_return_pct.to_numpy(),
                        ((controls-1.)*100).ravel(),f'{kind}:{label}:stock exposure control')
            for group,j in member_indices.items():
                mean_nav = nav[:,j,:].mean(axis=1)
                br,bm = stats(mean_nav*fx[:,None],lo,hi)
                bur,bum = stats(mean_nav,lo,hi)
                exposure = (opn[:,j,:]*held[:,j,:]).sum(axis=1)/opn[:,j,:].sum(axis=1)
                initial = nav[lo-1,j,:] if lo else np.ones((len(j),len(names)))
                control_return = ((controls[j]-1.)*initial).sum(axis=0)/initial.sum(axis=0)*100
                for s,strategy in enumerate(names):
                    row = ag.loc[group,strategy,label]
                    assert int(row.n)==len(j)
                    checks = {'return_pct':br[s],'mdd_pct':bm[s],'usd_return_pct':bur[s],
                              'usd_mdd_pct':bum[s],'mean_open_exposure_pct':exposure[lo:hi+1,s].mean()*100}
                    for bps in [5,10]:
                        checks[f'return_{bps}bp_pct'] = stats(cost_nav[bps][:,j,s].mean(axis=1)*fx,lo,hi)[0]
                    for field,value in checks.items():
                        compare(np.array(row[field]),np.array(value),f'aggregate:{group}:{strategy}:{label}:{field}')
                    dr = kr[j,s]-base_stats[label][0][j,0]
                    dd = kd[j,s]-base_stats[label][1][j,0]
                    for field,value in [('return_better_count',int((dr>TIE).sum())),
                                        ('mdd_better_count',int((dd>TIE).sum())),
                                        ('both_better_count',int(((dr>TIE)&(dd>TIE)).sum()))]:
                        if field in ag:
                            assert int(row[field])==value,(group,strategy,label,field,row[field],value)
                    if 'usd_matched_exposure_return_pct' in ag.columns:
                        if kind!='BASE' or np.isfinite(row.usd_matched_exposure_return_pct):
                            compare(np.array(row.usd_matched_exposure_return_pct),np.array(control_return[s]),'group exposure control')
                            compare(np.array(row.usd_vs_matched_exposure_return_pp),np.array(bur[s]-control_return[s]),'group timing delta')
                    record = {'group':group,'strategy':strategy,'kind':kind,'length':lengths[s],
                              'period':label,'return_pct':float(br[s]),'mdd_pct':float(bm[s]),
                              'usd_return_pct':float(bur[s]),'mean_open_exposure_pct':float(exposure[lo:hi+1,s].mean()*100)}
                    selection_rows.append(record)
                    if label=='front':
                        front_rows.append(record)
            if kind=='SMA':
                s = names.index('exit_SMA120')
                old = old_ps.loc[pd.IndexSlice[tickers,'exit_SMA120',label],:]
                for old_field,new_field in [('krw_return_pct','return_pct'),('krw_mdd_pct','mdd_pct'),
                                             ('usd_return_pct','usd_return_pct'),('usd_mdd_pct','usd_mdd_pct'),
                                             ('mean_open_exposure_pct','mean_open_exposure_pct')]:
                    error = float(np.max(np.abs(old[old_field].to_numpy()-expected[new_field][:,s])))
                    sma120_regression_error = max(sma120_regression_error,error)
                    assert error<=ATOL
        if kind!='BASE':
            for group,j in member_indices.items():
                for s,strategy in enumerate(names):
                    field = group+'::'+strategy
                    rebuilt = nav[:,j,s].mean(axis=1)*fx
                    error = float(np.max(np.abs(rebuilt-stored_curves[field])))
                    curve_error = max(curve_error,error)
                    assert error<=1e-9,(field,error)
                rebuilt_base = baseline_nav[:,j,0].mean(axis=1)*fx
                compare(stored_curves[group+'::buy_hold'].to_numpy(),rebuilt_base,'basket baseline')
            print(f'{kind}:120 settings x138 stocks independently verified.',flush=True)
    assert primary_per_stock_rows==138*240*3
    actual_selectors = json.loads((OUT/'selectors.json').read_text(encoding='utf-8'))
    selectors,ties = {},{}
    for key,group,floor in [('primary','jan2_external',False),('secondary','jan2_external',True),
                             ('broad','broad_external',False)]:
        selectors[key],ties[key] = own_picker(front_rows,group,floor)
        assert actual_selectors[key]==selectors[key],(key,actual_selectors,selectors)
    # Feed only front statistics to selection. Mutating every later score cannot
    # affect this independent reconstruction of the saved common parameters.
    changed = [dict(r,return_pct=(-1)**i*1e9,mdd_pct=-i*1e6) if r['period']!='front' else dict(r)
               for i,r in enumerate(selection_rows)]
    future_mutation = {key:own_picker([r for r in changed if r['period']=='front'],group,floor)[0]
                       for key,group,floor in [('primary','jan2_external',False),
                            ('secondary','jan2_external',True),('broad','broad_external',False)]}
    assert future_mutation==selectors
    assert result['selectors']==selectors and result['stocks']==138 and result['candidate_count']==240
    diagnostics = {}
    for group in groups:
        diagnostics[group] = {}
        for label in ['back','full']:
            rows = [r for r in selection_rows if r['group']==group and r['period']==label]
            base = next(r for r in rows if r['strategy']=='buy_hold')
            candidates = [r for r in rows if r['strategy']!='buy_hold']
            diagnostics[group][label] = {'hindsight_best':own_picker(rows,group)[0],
                'return_better':sum(r['return_pct']>base['return_pct']+TIE for r in candidates),
                'mdd_better':sum(r['mdd_pct']>base['mdd_pct']+TIE for r in candidates),
                'both_better':sum(r['return_pct']>base['return_pct']+TIE and
                                   r['mdd_pct']>base['mdd_pct']+TIE for r in candidates)}
    assert result['descriptive']==diagnostics
    neighbor_names = set()
    for strategy in selectors.values():
        kind,length = strategy[5:8],int(strategy[8:])
        neighbor_names.update(f'exit_{kind}{n}' for n in range(max(1,length-5),min(120,length+5)+1))
    neighbors = pd.read_csv(OUT/'neighbors.csv').set_index(['group','strategy','period'])
    assert set(neighbors.index)=={key for key in ag.index if key[1] in neighbor_names}
    for field in ['return_pct','mdd_pct','mean_open_exposure_pct','return_5bp_pct','return_10bp_pct']:
        compare(neighbors[field].to_numpy(),ag.loc[neighbors.index,field].to_numpy(),'selected neighbors:'+field)
    assert all(path.stat().st_size<10_000_000 for path in OUT.glob('*.csv'))
    for key,expected in [('eligible_stocks',138),('settings_per_stock',241),('path_count',33258)]:
        if key in result:
            assert result[key]==expected,(key,result[key],expected)
    selected_rows = [row for row in selection_rows if row['strategy'] in set(selectors.values())
                     and row['group'] in ['jan2_external','broad_external','additional_external']]
    output = {'status':'passed','eligible_stock_count':138,'candidate_settings':240,
              'candidate_cash_quantity_paths':33120,'baseline_cash_quantity_paths':138,
              'scalar_ma_signal_cells':2*120*138*211,'per_stock_period_rows_checked':99360+414,
              'aggregate_period_rows_checked':len(ag),'stored_basket_curves_checked':basket_columns,
              'numeric_cells_checked':int(checked_cells),'max_numeric_error':metric_error,
              'max_basket_curve_error':curve_error,'no_op_SMA1_EMA1_verified':True,
              'comparison_relative_equality_tolerance':1e-12,
              'provisional_strict_float_roundoff_discrepancy':{
                  'signal':'ARM/SMA3/2026-02-11',
                  'close':125.27999877929688,'pandas_average':125.27999877929689,
                  'direct_fsum_average':125.27999877929688,
                  'resolution':'Explicit relative1e-12 equality for both entry and exit; provisional strict-float grid replaced, no metric-error tolerance widened.'},
              'SMA120_prior_fixed_study_max_metric_error':sma120_regression_error,
              'front_only_selectors_verified':selectors,'front_maximum_ties':ties,
              'front_selection_unchanged_after_back_score_mutation':True,
              'all_paired_improvement_counts_verified':True,'hindsight_descriptive_counts_verified':True,
              'selected_neighbor_rows_checked':len(neighbors),
              'group_counts':{g:len(v) for g,v in groups.items()},'selected_group_period_results':selected_rows,
              'front_back_carry_verified':True,'actual_open_nav_exposure_verified':True,
              'all_five_ten_bp_sensitivities_verified':True,'source_and_protocol_hashes_verified':True,
              'comparison_source_hashes_verified':True,
              'prior_seven_stock_portfolio_context':seven_context,
              'prior_fixed126_sleeve_context':fixed_context,
              'ma_seconds':ma_seconds,'elapsed_seconds':time.perf_counter()-started,
              'limitations':[
                'The audit independently rebuilds every raw-price cash/share path and all retained metrics and baskets; no daily per-stock grid curve file is assumed.',
                'Chronological front selection uses one common parameter for all stocks;2026 prices, currentAI membership and discovery/external conditioning were already observed, so this is not a new untouched holdout.',
                'The primary and broad front pickers and the front-MDD-constrained secondary picker are separate prespecified procedures. None establishes a per-stock optimal MA.',
                'The nine cohorts overlap and share market factors;33,120 paths and many adjacent lengths do not constitute independent trials.',
                'SMA/EMA1 are no-op controls. Weekly-only reentry and daily prior-close exits are the tested timing; immediate daily reentry is not included.',
                'Cost and matched-exposure diagnostics are proportional-drag and ex-post controls respectively, not observed fills or executable allocations.'
              ]}
    (OUT/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ['selected_group_period_results','limitations']},
                     ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
