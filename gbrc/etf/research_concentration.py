"""Predeclared concentration and mixed-signal research; no live orders.

Run: py -m gbrc.etf.research_concentration [--reuse] [--threads 8]
"""
from __future__ import annotations
import argparse
import hashlib
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import json
import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads
from .research import OUT
from .research_strategy_comparison import disclosure_indices, our_signals, momentum120, schedules, simulate_open
from .research_engine import price_features
from .research_validation import _batch_metrics, _one_metrics
from .research_us_only import _hash
from .storage import atomic_write_json, atomic_write_text

DEST=OUT/'concentration_research'
STATS=('max_stocks','mean_stocks','max_stock_weight','max_detail_weight','max_semiconductor_plus_unknown_weight',
       'max_ai_hardware_weight','max_unknown_weight','stock_violation_days','detail_violation_days',
       'semiconductor_plus_unknown_violation_days','any_violation_days','trading_days','one_way_turnover',
       'mean_cash','missing_held_close_marks','frozen_constraint_violation_days','trim_days','max_known_semiconductor_weight',
       'target_overallocation_days','target_cap_violation_without_frozen_exception_days')


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def candidate_grid():
    """Canonical A/B/C/D union: 801 fixed hypotheses, before seeing results."""
    rows={}
    def add(stage,anchor,mix,n,weight,freq,stock=1.,sector=1.,broad=1.,overflow=0,trim=0):
        if mix==0: anchor=0
        if mix==1: n,weight=0,0
        if stock==sector==broad==1: overflow=0
        key=(anchor,mix,n,weight,freq,stock,sector,broad,overflow,trim)
        if key not in rows:
            rows[key]={'anchor':('momentum','239120','201609')[anchor], 'our_mix':mix,
                'momentum_stocks':n,'momentum_weighting':('equal','fund','inverse_volatility')[weight],
                'rebalance':('daily','weekly','monthly')[freq], 'stock_cap':stock,'detail_sector_cap':sector,
                'semiconductor_plus_unknown_cap':broad,'overflow':('cash','redistribute')[overflow],
                'next_open_trim':bool(trim),'stages':[]}
        if stage not in rows[key]['stages']: rows[key]['stages'].append(stage)
    for n,w,f in product((5,7,10),range(3),range(3)): add('A',0,0,n,w,f)
    combinations=[(0,0.)]+[(a,m) for a,m in product((1,2),(.25,.5,.75,1.))]
    for (a,m),sc,se,f,o in product(combinations,(.2,.25,.33,1.),(.4,.6,1.),range(3),range(2)):
        add('B',a,m,5,0,f,sc,se,1.,o)
    for (a,m),f,o,tr in product(combinations,range(3),range(2),range(2)):
        add('C_trim' if tr else 'C',a,m,5,0,f,.25,.6,.6,o,tr)
    for n,w,f,o in product((5,7,10),range(3),range(3),range(2)):
        add('D',0,0,n,w,f,.25,.6,.6,o)
    configs=[]
    for index,config in enumerate(rows.values()): configs.append({'index':index,**config})
    if len(configs)!=801: raise AssertionError(f'Unexpected grid count {len(configs)}')
    return np.asarray(list(rows),dtype=np.float64),configs


@njit(cache=True)
def project_caps(raw, frozen, sectors, semis, stock_cap, sector_cap, broad_cap, overflow):
    """Order-independent proportional progressive fill, with fixed frozen assets.

    Cash mode never exceeds a selected name's original target. Redistribution
    only fills the originally selected tradable names. Cash is not renormalized.
    """
    n=len(raw); ng=int(np.max(sectors))+1
    result=frozen.copy(); group=np.zeros(ng); broad=0.; locked=0.
    for i in range(n):
        locked+=frozen[i]; group[sectors[i]]+=frozen[i]
        if semis[i]: broad+=frozen[i]
    # Unavailable/frozen target dollars are not an excuse to add unnamed stocks.
    flex=np.zeros(n); cap=np.zeros(n); active=np.zeros(n,dtype=np.bool_)
    total_raw=np.sum(raw); flex_raw=0.
    for i in range(n):
        if frozen[i]<=0 and raw[i]>0:
            flex_raw+=raw[i];active[i]=True
            cap[i]=min(stock_cap,raw[i]) if overflow==0 else stock_cap
    budget=min(flex_raw,max(0.,total_raw-locked))
    used=0.
    for _ in range(3*n+ng+10):
        if budget-used<=1e-12: break
        total=0.; mass=np.zeros(ng); bm=0.
        for i in range(n):
            if active[i] and (cap[i]-flex[i]<=1e-12 or sector_cap-group[sectors[i]]<=1e-12 or (semis[i] and broad_cap-broad<=1e-12)):
                active[i]=False
            if active[i]:
                total+=raw[i];mass[sectors[i]]+=raw[i]
                if semis[i]: bm+=raw[i]
        if total<=1e-15: break
        scale=(budget-used)/total
        for i in range(n):
            if active[i]: scale=min(scale,max(0.,cap[i]-flex[i])/raw[i])
        for g in range(ng):
            if mass[g]>0: scale=min(scale,max(0.,sector_cap-group[g])/mass[g])
        if bm>0: scale=min(scale,max(0.,broad_cap-broad)/bm)
        if scale<=1e-15: break
        for i in range(n):
            if active[i]:
                amount=scale*raw[i];flex[i]+=amount;group[sectors[i]]+=amount;used+=amount
                if semis[i]: broad+=amount
    return result+flex


@njit(cache=True)
def _simulate_one(param, opens, closes, prior_close, fx_open, fx_close, mom_order, mom_basis,
                  our_ranks, our_basis, our_exposure, schedule, sectors, semis, ai, unknown,
                  start=0, detail=False):
    anchor=int(param[0]);mix=param[1];number=int(param[2]);weighting=int(param[3]);freq=int(param[4])
    stockcap,sectorcap,broadcap=param[5],param[6],param[7];overflow=int(param[8]);trim=param[9]>0
    T,N=opens.shape;ng=int(np.max(sectors))+1
    shares=np.zeros(N);cash=1.;before=1.;last_mark=prior_close[start].copy()
    previous_mom=np.zeros(N,dtype=np.bool_);initialized=False;previous_breach=False
    returns=np.zeros(T-start);stats=np.zeros(len(STATS))
    weights=np.zeros((T-start,N)) if detail else np.zeros((1,N))
    raw_events=np.zeros((T-start,N)) if detail else np.zeros((1,N))
    target_events=np.zeros((T-start,N)) if detail else np.zeros((1,N))
    event_flags=np.zeros(T-start,dtype=np.bool_) if detail else np.zeros(1,dtype=np.bool_)
    for t in range(start,T):
        observed=np.isfinite(opens[t]) & (opens[t]>0)
        marked=np.where(observed,opens[t],last_mark)
        val=shares*np.nan_to_num(marked)*fx_open[t]
        nav_open=cash+np.sum(val)
        regular=schedule[t,freq] or not initialized
        do_trim=trim and previous_breach and not regular
        if regular or do_trim:
            raw=np.zeros(N)
            if do_trim:
                raw=val/nav_open
            else:
                if mix<1:
                    selected=np.zeros(N,dtype=np.bool_);count=0
                    for k in range(min(N,number+5)):
                        j=mom_order[t,k]
                        if j>=0 and previous_mom[j] and observed[j] and count<number and mom_basis[weighting,t,j]>0:
                            selected[j]=True;count+=1
                    for k in range(N):
                        if count>=number: break
                        j=mom_order[t,k]
                        if j>=0 and observed[j] and not selected[j] and mom_basis[weighting,t,j]>0:
                            selected[j]=True;count+=1
                    denom=0.
                    for j in range(N):
                        if selected[j]: denom+=mom_basis[weighting,t,j]
                    if denom>0:
                        for j in range(N):
                            if selected[j]: raw[j]+=(1-mix)*mom_basis[weighting,t,j]/denom
                    previous_mom=selected
                if mix>0:
                    ai_source=anchor-1;our_n=1 if anchor==1 else 3
                    denom=0.;selected_our=np.zeros(N)
                    count=0
                    for k in range(10):
                        if count>=our_n: break
                        j=our_ranks[ai_source,t,k]
                        if j>=0 and observed[j] and our_basis[ai_source,t,k]>0:
                            selected_our[j]=our_basis[ai_source,t,k];denom+=selected_our[j];count+=1
                    if denom>0: raw+=mix*our_exposure[ai_source,t]*selected_our/denom
            frozen=np.zeros(N);nf=0
            for j in range(N):
                if shares[j]>0 and not observed[j]: frozen[j]=val[j]/nav_open;nf+=1
            # The maximum includes frozen old positions as well as new targets.
            chosen=0
            order=np.argsort(-raw)
            for k in range(N):
                j=order[k]
                if raw[j]>0 and frozen[j]<=0:
                    if chosen>=10-nf: raw[j]=0.
                    else: chosen+=1
            desired=project_caps(raw,frozen,sectors,semis,stockcap,sectorcap,broadcap,0 if do_trim else overflow)
            if np.sum(raw)>0: initialized=True
            traded=0.
            for j in range(N):
                if observed[j]:
                    wanted=desired[j]*nav_open;traded+=abs(wanted-val[j])
                    shares[j]=wanted/(opens[t,j]*fx_open[t])
            cash=max(0.,nav_open*(1-np.sum(desired)))
            if traded>1e-10: stats[11]+=1
            stats[12]+=traded/nav_open
            if do_trim: stats[16]+=1
            fg=np.zeros(ng);fb=0.;fviolate=False
            for j in range(N):
                fg[sectors[j]]+=frozen[j]
                if semis[j]: fb+=frozen[j]
                if frozen[j]>stockcap+1e-8: fviolate=True
            if np.max(fg)>sectorcap+1e-8 or fb>broadcap+1e-8: fviolate=True
            if fviolate: stats[15]+=1
            tg=np.zeros(ng);tb=0.;bad=False
            for j in range(N):
                tg[sectors[j]]+=desired[j]
                if semis[j]: tb+=desired[j]
                if desired[j]>stockcap+1e-8 and frozen[j]<=stockcap+1e-8: bad=True
            for g in range(ng):
                if tg[g]>sectorcap+1e-8 and fg[g]<=sectorcap+1e-8: bad=True
            if tb>broadcap+1e-8 and fb<=broadcap+1e-8: bad=True
            if np.sum(desired)>1+1e-8: stats[18]+=1
            if bad: stats[19]+=1
            if detail:
                raw_events[t-start]=raw;target_events[t-start]=desired;event_flags[t-start]=True
        for j in range(N):
            if observed[j]: last_mark[j]=opens[t,j]
            if np.isfinite(closes[t,j]) and closes[t,j]>0: last_mark[j]=closes[t,j]
            elif shares[j]>0: stats[14]+=1
        value=shares*np.nan_to_num(last_mark)*fx_close[t];nav=cash+np.sum(value)
        returns[t-start]=nav/before-1.;before=nav
        weight=value/nav;count=np.sum(shares>1e-14)
        stats[0]=max(stats[0],count);stats[1]+=count
        group=np.zeros(ng);sem=0.;aigroup=0.;unk=0.;knownsem=0.
        for j in range(N):
            group[sectors[j]]+=weight[j]
            if semis[j]: sem+=weight[j]
            if semis[j] and sectors[j]!=unknown: knownsem+=weight[j]
            if ai[j]: aigroup+=weight[j]
            if sectors[j]==unknown: unk+=weight[j]
        mx=np.max(weight);mg=np.max(group)
        stats[2]=max(stats[2],mx);stats[3]=max(stats[3],mg);stats[4]=max(stats[4],sem)
        stats[5]=max(stats[5],aigroup);stats[6]=max(stats[6],unk)
        stats[17]=max(stats[17],knownsem)
        sv=mx>stockcap+1e-8;dv=mg>sectorcap+1e-8;bv=sem>broadcap+1e-8
        stats[7]+=sv;stats[8]+=dv;stats[9]+=bv;previous_breach=sv or dv or bv
        stats[10]+=previous_breach;stats[13]+=cash/nav
        if detail: weights[t-start]=weight
    stats[1]/=T-start;stats[13]/=T-start
    return returns,stats,weights,raw_events,target_events,event_flags


@njit(cache=True,parallel=True)
def simulate_grid(params, opens, closes, prior_close, fx_open, fx_close, mom_order, mom_basis,
                  our_ranks, our_basis, our_exposure, schedule, sectors, semis, ai, unknown):
    M=len(params);T=len(opens)
    returns=np.empty((M,T),dtype=np.float32);stats=np.empty((M,len(STATS)))
    for m in prange(M):
        r,s,_,_,_,_=_simulate_one(params[m],opens,closes,prior_close,fx_open,fx_close,mom_order,mom_basis,
                                 our_ranks,our_basis,our_exposure,schedule,sectors,semis,ai,unknown)
        returns[m]=r;stats[m]=s
    return returns,stats


def load_inputs():
    source=OUT/'us_only_experiment';lab=OUT.parent/'strategy_lab'
    a=dict(np.load(source/'bundle.npz',allow_pickle=False));meta=load(source/'metadata.json')
    panels=load(lab/'lab_panels.json');dates=a['dates'].astype(str)
    tickers=[panels['ticker'][s['company_key']] for s in meta['securities']]
    frames={f:pd.read_csv(lab/f'prices_{f}.csv',index_col=0) for f in ('open','close','adj_close')}
    all_days=frames['close']['QQQ'].dropna().index.to_numpy(str);at=np.searchsorted(all_days,dates)
    close_all=frames['close'].reindex(all_days);fx=close_all['KRW=X'].ffill().to_numpy()
    opens=frames['open'].reindex(dates)[tickers].to_numpy(float)
    closes=frames['close'].reindex(dates)[tickers].to_numpy(float)
    prior=close_all[tickers].ffill().iloc[at-1].to_numpy(float)
    fx_open,fx_close=fx[at-1],fx[at]
    a['prices']=close_all[tickers].ffill().iloc[at].to_numpy(float)*fx_close[:,None]
    a['prior_close']=prior;a['qqq_usd']=close_all['QQQ'].iloc[at].to_numpy(float)
    adjusted=frames['adj_close'].reindex(all_days)[tickers].to_numpy(float)
    momentum=momentum120(adjusted,at);iv=np.zeros_like(momentum)
    for t,i in enumerate(at):
        with np.errstate(divide='ignore',invalid='ignore'):
            history=np.log(adjusted[i-20:i]/adjusted[i-21:i-1])
        counts=np.isfinite(history).sum(axis=0)
        with np.errstate(invalid='ignore'):
            vol=np.nanstd(history,axis=0)
        iv[t]=np.divide(1.,vol,out=np.zeros_like(vol),where=(counts>=15)&np.isfinite(vol)&(vol>0))
    sector=load(DEST/'sector_map.json')
    assert sector['ordered_tickers']==tickers
    sectors=np.asarray(sector['detail_ids'],dtype=np.int64)
    semis=np.asarray(sector['risk_group_masks']['broad_semiconductor'],dtype=np.bool_) | (sectors==int(sector['unknown_id']))
    ai=np.asarray(sector['risk_group_masks']['ai_hardware_infrastructure'],dtype=np.bool_)
    schedule=np.zeros((len(dates),3),dtype=np.bool_)
    cal=schedules(dates);schedule[:,0]=cal['daily'];schedule[:,1]=cal['weekly']
    schedule[:,2]=np.r_[True,np.array([day[:7] for day in dates[1:]])!=np.array([day[:7] for day in dates[:-1]])]
    fixed=load(source/'configurations.json')
    per_scenario={}
    for lag in range(3):
        si=disclosure_indices(dates,a['snapshot_dates'].astype(str),lag)
        w=a['weights'][np.maximum(si,0),0].copy();w[si<0]=0
        eligible=(w>0)&np.isfinite(prior)&(prior>0)&np.isfinite(momentum)
        order=np.argsort(-np.where(eligible,momentum,-np.inf),axis=1,kind='stable')
        order=np.where(np.take_along_axis(eligible,order,axis=1),order,-1).astype(np.int64)
        mb=np.array([np.ones_like(w),w,iv])
        sig=[our_signals(a,fixed[index],si) for index in (239120,201609)]
        per_scenario[f'publication_lag{lag}']=(opens,closes,prior,fx_open,fx_close,order,mb,
            np.array([s['ranks'] for s in sig],dtype=np.int64),np.array([s['basis'] for s in sig]),
            np.array([s['exposure'] for s in sig]),schedule,sectors,semis,ai,int(sector['unknown_id']))
    paths=[source/'bundle.npz',source/'metadata.json',source/'configurations.json',source/'zero_cost/results.json',
           OUT/'strategy_comparison/results.json',DEST/'sector_map.json',lab/'lab_panels.json',
           *[lab/f'prices_{field}.csv' for field in ('open','close','adj_close')],
           Path(__file__).with_name('research_strategy_comparison.py'),Path(__file__).with_name('research_engine.py')]
    return a,tickers,sector,per_scenario,paths


def run(reuse=False,threads=8):
    DEST.mkdir(parents=True,exist_ok=True)
    params,configs=candidate_grid()
    # Input loading constructs only price/signal arrays and does not score paths.
    arrays,tickers,sector,inputs,paths=load_inputs();dates=arrays['dates'].astype(str)
    hashes={str(path):_hash(path) for path in paths}
    time_return=float(arrays['benchmarks'][-1,0]/arrays['benchmarks'][0,0]-1)
    old_bench=load(OUT/'us_only_experiment/zero_cost/results.json')['benchmarks']
    protocol={
        'created_before_simulation':datetime.now(timezone.utc).isoformat(),'candidate_count':len(configs),
        'scenario_count':3,'grid_paths':len(configs)*3,'period':[str(dates[0]),str(dates[-1])],
        'stage_counts_after_union':{stage:sum(stage in c['stages'] for c in configs) for stage in ('A','B','C','C_trim','D')},
        'canonical_dedup':'Pure-momentum and pure-our endpoints canonicalized; unrestricted cash/redistribution merged; identical A/B/C/D definitions merged.',
        'selection':'Retrospective full-period balanced comparison after previous data inspection. Among stockcap<=25%,detailcap<=60%,semiconductor-plus-unknown-cap<=60% targets and worst-lag return>=2*TIME Jan2-close reference, minimize absolute worst daily-close MDD; tie-break lower worst turnover. No relaxation if empty; report Pareto set.',
        'target_reference':{'time_jan2_close_return':time_return,'required_two_times_return':2*time_return,
            'original_later_entry_time_return':old_bench['time']['segments']['full']['return_pct']/100,
            'timing':'ETF KRX Close reference on same date labels; not simultaneous US Open execution.'},
        'no_untouched_holdout':'All dates were already observed in prior research; August is a reused historical diagnostic, never independent validation.',
        'portfolio':'Single KRWcash account. On actual US calendar daily/weekly/monthly event, refresh both signal targets with previous closes/latest eligible disclosures, combine identical stocks, then constrain targets. No inter-event capital reset.',
        'momentum':'TIME-held MOM120 prior AdjClose,2025 warmup; keep incumbent momentum names in topN+5, then fill topN. Independent momentum membership resets on cash restart. Equal/fund/inverse20log-return-vol weights.',
        'our_signals':'Frozen239120(new-entry40,max1) or201609(sold-reversal20,max3,inverse20vol,trend-half) signal templates. Their recomputation on chosen portfolio schedule is a new hypothesis, not native fixed-strategy account replay.',
        'publication_delay':'0/1/2 actual US-session disclosure delay; previous-close price features unchanged. Same-date publication before Open is assumed, historical timestamps unverified.',
        'projection':'Proportional progressive filling under simultaneous NAV stock/detail/known-semiconductor-OR-unknown limits after same-stock aggregation and fixed untradable holdings. Cash mode cannot exceed original name target; redistribution stays within selected names; never normalize capped portfolio to100%.',
        'missing_quotes':'Missing Open freezes existing shares and prevents new trade. Frozen old names count toward10. Frozen infeasible caps remain and are flagged; missing Close carries known mark and is counted.',
        'slot_overflow':'If frozen old names plus selected new names exceed10, retain the largest raw new targets within remaining slots. Removed target allocation becomes cash; frozen holdings consume the remaining investment budget. Redistribution only redistributes cap overflow within the surviving selected names.',
        'cap_meaning':'Targets at rebalance, not continuous hard limits. Close violation counts and maximum realized weights reported. Gaps/drift can exceed caps.',
        'trim':'If prior Close violates any enabled cap, next Open projects current holdings with cash overflow (sells only). No immediate redistribution. If regular rebalance coincides, compute its final target and trade once.',
        'unknown_sectors':'All40 unclassified stocks share one unknown detail-sector cap AND consume the combined known-semiconductor-plus-unknown60% capacity. This is conservative uncertainty treatment, not sector reclassification. Known-semiconductor and unknown realized weights reported separately.',
        'execution':'Common split-reflected raw Open/Close; old shares earn overnight gap, fractional NAV allocations at Open, mark same-day Close. Explicit costs0, dividends/taxes excluded. FXOpen=previousUSdailyFXClose proxy; FXClose=todaydailyFXClose. CashKRW.',
        'execution_limitation':'Aggregate opening-price dollar allocation is not exact pre-submitted share quantities; actual slippage/spread/FX costs are unverified.',
        'fresh_starts':'Reset shares, cash, and momentum-buffer membership; retain historical prices/disclosures.',
        'exposure_matched_control':'For selected profiles, normalize each recorded pre-cap target to the cap-constrained investment fraction determined at that same Open, then replay an unconstrained account on the same event dates. No subsequent realized-cash backscaling. This is an explanatory counterfactual conditioned on constrained-account event/target information, not an independently selected tradable strategy.',
        'source_sha256':hashes,'code_sha256':_hash(Path(__file__)),
        'configurations_sha256':hashlib.sha256(json.dumps(configs,sort_keys=True).encode()).hexdigest(),
    }
    if (DEST/'protocol.json').exists():
        previous=load(DEST/'protocol.json')
        for key in ('candidate_count','selection','configurations_sha256'):
            if protocol[key]!=previous.get(key): raise ValueError('Frozen candidate/selection protocol changed')
        if reuse and any(protocol[key]!=previous.get(key) for key in ('source_sha256','code_sha256')):
            raise ValueError('Cannot reuse grids with changed input/code hashes')
        protocol['created_before_simulation']=previous['created_before_simulation']
        protocol['recalculations']=previous.get('recalculations',[])
        if not reuse:
            protocol['recalculations'].append({'at':datetime.now(timezone.utc).isoformat(),
                'previous_code_sha256':previous['code_sha256'],'new_code_sha256':protocol['code_sha256'],
                'reason':'Explicit fresh rerun; hypothesis grid and selection policy unchanged'})
    atomic_write_json(DEST/'protocol.json',protocol);atomic_write_json(DEST/'configurations.json',configs)
    print(f'Protocol frozen: {len(configs)} definitions x3 disclosure scenarios',flush=True)
    set_num_threads(threads);grids={};stats={}
    for name,arguments in inputs.items():
        gp=DEST/f'{name}_returns.npy';sp=DEST/f'{name}_stats.npy'
        if reuse and gp.exists():
            grids[name]=np.load(gp);stats[name]=np.load(sp)
        else:
            print('Simulating '+name,flush=True)
            grids[name],stats[name]=simulate_grid(params,*arguments)
            np.save(gp,grids[name]);np.save(sp,stats[name])
    context={'protocol':protocol,'configs':configs,'params':params,'arrays':arrays,
             'tickers':tickers,'sector':sector,'inputs':inputs,'grids':grids,'stats':stats,'hashes':hashes,'paths':paths}
    finish(context)
    return context


def segment_metrics(returns,dates):
    dates=np.asarray(dates,dtype=str)
    masks={'full':np.ones(len(dates),dtype=bool),'train':dates<'2026-06-01',
           'june_july':(dates>='2026-06-01')&(dates<'2026-08-01'),'august_september':dates>='2026-08-01'}
    return {name:_one_metrics(np.asarray(returns)[mask]) for name,mask in masks.items() if mask.any()}


def exposure_matched(detail,arguments,start):
    """Causal comparison using only investment fraction decided at each Open."""
    _,_,_,raw,target,events=detail
    T,N=raw.shape
    order=np.argsort(-raw,axis=1,kind='stable')
    basis=np.take_along_axis(raw,order,axis=1)
    sig={'ranks':np.where(basis>0,order,-1),'basis':basis,'schedule':events,
         'exposure':target.sum(axis=1),'max_stocks':10,'buffer':0}
    op,cl,_,fo,fc=arguments[:5]
    return simulate_open(op[start:],cl[start:],fo[start:],fc[start:],sig)


def finish(c):
    dates=c['arrays']['dates'].astype(str);configs=c['configs'];params=c['params'];grids=c['grids'];stats=c['stats']
    metrics={name:_batch_metrics(r,np.arange(len(dates))) for name,r in grids.items()}
    worst_return=np.minimum.reduce([m['return'] for m in metrics.values()])
    worst_mdd=np.minimum.reduce([m['max_drawdown'] for m in metrics.values()])
    worst_turnover=np.maximum.reduce([s[:,12] for s in stats.values()])
    target=(params[:,5]<=.25)&(params[:,6]<=.6)&(params[:,7]<=.6)
    time_return=c['protocol']['target_reference']['time_jan2_close_return']
    meets=target&(worst_return>=2*time_return)
    candidates=np.flatnonzero(meets)
    ordered=sorted(candidates.tolist(),key=lambda i:(-worst_mdd[i],worst_turnover[i],i))
    selected=ordered[0] if ordered else None
    pareto=[];best_risk=-np.inf
    for i in sorted(np.flatnonzero(target).tolist(),key=lambda i:(-worst_return[i],-worst_mdd[i],worst_turnover[i],i)):
        if worst_mdd[i]>best_risk+1e-12:
            pareto.append(i);best_risk=worst_mdd[i]
    def find(anchor,mix,n,weight,freq):
        for i,p in enumerate(params):
            if tuple(p)==(anchor,mix,n,weight,freq,1.,1.,1.,0.,0.): return i
        raise ValueError('Reference rule not present')
    references={'calendar_momentum5_equal_weekly':find(0,0,5,0,1),
                'calendar_momentum5_fund_daily':find(0,0,5,1,0),
                'calendar_239120_daily':find(1,1,0,0,0),'calendar_201609_daily':find(2,1,0,0,0)}
    representatives=set(references.values())
    if selected is not None: representatives.add(selected)
    if pareto: representatives.update((pareto[0],pareto[-1],pareto[len(pareto)//2]))
    selection={'selected_index':selected,'selection_policy':c['protocol']['selection'],
        'eligible_target_count':int(target.sum()),'meeting_2x_count':int(meets.sum()),
        'meeting_3x_count':int(np.sum(target&(worst_return>=3*time_return))),
        'meeting_2x_ranked_indices':ordered,'pareto_indices':pareto,
        'reference_indices':references,'representative_indices':sorted(representatives),
        'worst_return':worst_return.tolist(),'worst_max_drawdown':worst_mdd.tolist(),
        'worst_one_way_turnover':worst_turnover.tolist(),'required_2x_return':2*time_return,
        'interpretation':'Full-period retrospective selection with predeclared targets. No independent holdout. If selected_index is null, no constrained rule met the2x target in every scenario.'}
    atomic_write_json(DEST/'selection.json',selection)
    print('Selection frozen '+json.dumps({k:selection[k] for k in ('selected_index','eligible_target_count','meeting_2x_count','meeting_3x_count','representative_indices')}),flush=True)
    rows=[]
    for i,config in enumerate(configs):
        rows.append({'index':i,'worst_return_pct':float(worst_return[i]*100),
            'worst_max_drawdown_pct':float(worst_mdd[i]*100),'worst_turnover':float(worst_turnover[i]),
            'balanced_target_eligible':bool(target[i]),'passes_2x_in_all_scenarios':bool(meets[i]),
            'scenarios':{name:{'return_pct':float(metrics[name]['return'][i]*100),
                'max_drawdown_pct':float(metrics[name]['max_drawdown'][i]*100),
                'statistics':{key:float(value) for key,value in zip(STATS,stats[name][i])}} for name in grids}})
    atomic_write_json(DEST/'candidate_metrics.json',rows)
    profiles={};archives={};hstart=int(np.searchsorted(dates,'2026-08-01'))
    detail_matches=True
    for i in sorted(representatives):
        profile={'index':i,'config':configs[i],'worst_return_pct':float(worst_return[i]*100),
                 'worst_max_drawdown_pct':float(worst_mdd[i]*100),'scenarios':{}}
        for name,arguments in c['inputs'].items():
            detail=_simulate_one(params[i],*arguments,0,True)
            np.testing.assert_allclose(detail[0],grids[name][i],atol=1e-7)
            np.testing.assert_allclose(detail[1],stats[name][i],rtol=1e-10,atol=1e-10)
            fresh=_simulate_one(params[i],*arguments,hstart,True)
            control=exposure_matched(detail,arguments,0)
            fresh_control=exposure_matched(fresh,arguments,hstart)
            def pack(path,days):
                r,s,w,raw,targets,events=path
                return {'segments':segment_metrics(r,days),
                    'statistics':{key:float(value) for key,value in zip(STATS,s)},
                    'monthly':{month:_one_metrics(r[np.char.startswith(np.asarray(days),month)]) for month in sorted(set(day[:7] for day in days))},
                    'daily_returns':r.tolist(),'daily_nav':np.cumprod(1+r).tolist(),
                    'daily_holdings':[[[int(j),float(row[j])] for j in np.flatnonzero(row>1e-12)] for row in w]}
            row=pack(detail,dates)
            row['fresh_august']={'start':str(dates[hstart]),**pack(fresh,dates[hstart:])}
            row['exposure_matched_uncapped_control']={
                'interpretation':c['protocol']['exposure_matched_control'],
                'segments':segment_metrics(control['returns'],dates),
                'fresh_august_segments':segment_metrics(fresh_control['returns'],dates[hstart:]),
                'daily_returns':control['returns'].tolist(),'daily_nav':control['nav']}
            profile['scenarios'][name]=row
            for suffix,path in [('full',detail),('fresh_august',fresh)]:
                prefix=f'candidate{i}_{name}_{suffix}'
                for key,value in zip(('returns','stats','weights','precap_targets','projected_targets','events'),path):
                    archives[prefix+'_'+key]=value
        profiles[str(i)]=profile
    np.savez_compressed(DEST/'representative_paths.npz',**archives)
    source_still={str(path):_hash(path) for path in c['paths']}
    validation={'source_files_unchanged':source_still==c['hashes'],'candidate_count':len(configs),
        'distinct_parameter_rows':len({tuple(row) for row in params}),'scenario_paths':sum(len(x) for x in grids.values()),
        'all_returns_finite':all(np.isfinite(x).all() for x in grids.values()),
        'maximum_observed_holdings':max(float(x[:,0].max()) for x in stats.values()),
        'missing_held_close_marks':sum(float(x[:,14].sum()) for x in stats.values()),
        'frozen_constraint_violation_days':sum(float(x[:,15].sum()) for x in stats.values()),
        'target_overallocation_days':sum(float(x[:,18].sum()) for x in stats.values()),
        'target_cap_violation_without_frozen_exception_days':sum(float(x[:,19].sum()) for x in stats.values()),
        'representative_grid_detail_matches':detail_matches,'stats_columns':list(STATS),
        'conservative_broad_mask_includes_all_unknown':True,
        'actual_close_caps_can_be_exceeded_by_drift':True,
        'cash_start_resets_momentum_membership':True,
        'selection_uses_reused_full_history':True,'independent_holdout_claimed':False}
    assert validation['source_files_unchanged'] and validation['all_returns_finite']
    assert validation['maximum_observed_holdings']<=10 and validation['target_overallocation_days']==0
    assert validation['target_cap_violation_without_frozen_exception_days']==0
    benchmarks={name:{'same_calendar_close_return_pct':float((c['arrays']['benchmarks'][-1,j]/c['arrays']['benchmarks'][0,j]-1)*100),
                      'valuation':'KRX Close forTIME/KoAct; not simultaneousUSOpen execution'} for j,name in enumerate(('time','koact','qqq_krw'))}
    results={'protocol':c['protocol'],'selection':selection,'dates':dates.tolist(),'tickers':c['tickers'],
        'benchmarks':benchmarks,'profiles':profiles,'statistics_columns':list(STATS),
        'prior_fixed_common_engine_references':load(OUT/'strategy_comparison/results.json')['strategies'],
        'artifact_note':'candidate_metrics.json contains all801 scenario metrics. representative_paths.npz preserves pre-cap/projected targets and event flags for exposure-control audit.'}
    atomic_write_json(DEST/'results.json',results);atomic_write_json(DEST/'validation.json',validation)
    atomic_write_json(DEST/'summary.json',{'selection':selection,'benchmarks':benchmarks,'profiles':profiles,'validation':validation})
    print(json.dumps({'out':str(DEST),'validation':validation},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reuse',action='store_true');p.add_argument('--threads',type=int,default=8)
    args=p.parse_args();run(args.reuse,args.threads)
