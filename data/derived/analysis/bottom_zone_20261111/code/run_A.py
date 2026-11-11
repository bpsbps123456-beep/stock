"""Two-stage locked research; stage2 requires committed freeze_A.json."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from panel_A import ROOT, FAMILIES, SIGNALS, assemble, inputs, dump, fingerprint

PERIODS = {'2024':('2024-01-02','2024-12-31'),'2025':('2025-01-01','2025-12-31'),
           '2026H1':('2026-01-01','2026-06-30'),'2026H2':('2026-07-01','2026-12-31'),
           '2026':('2026-01-01','2026-12-31'),'all':('2024-01-02','2026-12-31')}


def outcomes(px):
    out = pd.DataFrame(index=px.index)
    for i,d in enumerate(px.index):
        if d<pd.Timestamp('2024-01-02'):
            continue  # 2023 is rank preparation, never an outcome sample.
        o = px.open.iloc[i]
        w = px.close.iloc[i:i+20]
        full = len(w)==20 and w.notna().all()
        mn,mx = w.min()/o-1,w.max()/o-1
        out.loc[d,'min_seen'] = mn
        out.loc[d,'full20'] = full
        out.loc[d,'min20'] = mn if full else np.nan
        out.loc[d,'max20'] = mx if full else np.nan
        for suffix,limit in [('Z',.05),('Z3',.03),('Z8',.08)]:
            out.loc[d,suffix] = 0 if mn < -limit-1e-12 else (1 if full else np.nan)
        out.loc[d,'H'] = 0 if mx>.05+1e-12 else (1 if full else np.nan)
        for n in [10,20]:
            out.loc[d,'R'+str(n)] = px.close.iloc[i+n-1]/o-1 if i+n<=len(px) else np.nan
    return out


def combo(panel,reps):
    x = panel.copy()
    vals = pd.DataFrame({f:x[k] if k else np.nan for f,k in reps.items()},index=x.index)
    x['K_min'] = vals.eq(1).sum(axis=1)
    x['K_max'] = x.K_min+vals.isna().sum(axis=1)
    x['K'] = x.K_min.where(vals.notna().all(axis=1))
    for n in [2,3,4]:
        x['K'+str(n)] = np.where(x.K_min>=n,1,np.where(x.K_max<n,0,np.nan))
    for fam in ['X','T','G','M','N']:
        a,c = vals.S,vals[fam]
        x['S_AND_'+fam] = np.where(a.eq(0)|c.eq(0),0,np.where(a.eq(1)&c.eq(1),1,np.nan))
    return x


def event_indices(frame,key,state='down',on=1):
    occurred=frame.open.notna() if 'open' in frame else True
    positions = np.flatnonzero((frame[key].eq(on)&frame[state]&frame.index.to_series().ge('2024-01-02')&occurred).to_numpy())
    result=[]
    for i in positions:
        if not result or i-result[-1]>=10: result.append(i)
    return np.asarray(result,dtype=int)


def matching(frame,on,off):
    """Chronological greedy standardized price matching, no outcome lookups."""
    pairs=[]
    for year in sorted(set(frame.index[on].year)):
        oni=[i for i in on if frame.index[i].year==year]
        pool=[i for i in off if frame.index[i].year==year]
        mask=(frame.index.year==year)&frame['active_state']
        raw=frame[['dd60','ret5','ret1']].to_numpy(float)
        sd=np.nanstd(raw[mask],axis=0,ddof=1)
        sd=np.where(sd>0,sd,1)
        z=raw/sd
        for i in oni:
            candidates=[j for j in pool if abs(j-i)>2 and np.isfinite(z[j]).all()]
            if not candidates: continue
            dist=np.sum((z[candidates]-z[i])**2,axis=1)
            j=candidates[int(np.argmin(dist))]
            pool.remove(j)
            pairs.append((i,j))
    return pairs


def stats(frame,indices,target='Z'):
    z=frame.iloc[indices]
    out={'n':len(z),'scored':int(z[target].notna().sum()),'rate':z[target].mean(),
         'pending':int(z[target].isna().sum()),'complete20':int(z.full20.eq(True).sum())}
    for col in ['Z3','Z8','R10','R20','min20']:
        out[col]=z[col].mean()
        out[col+'_n']=int(z[col].notna().sum())
    for col in ['R10','R20']:
        vals=z[col].dropna()
        out[col+'_win']=float((vals>0).mean()) if len(vals) else np.nan
    return out


def evaluate(frame,key,state='down',event=False):
    fr=frame.copy()
    fr['active_state']=fr[state]
    if event:
        all_on=event_indices(fr,key,state,1)
        all_off=event_indices(fr,key,state,0)
    else:
        all_on=np.flatnonzero((fr[key].eq(1)&fr[state]).to_numpy())
        all_off=np.flatnonzero((fr[key].eq(0)&fr[state]).to_numpy())
    target='Z' if state=='down' else 'H'
    result={}
    pairs_out=[]
    for name,(lo,hi) in PERIODS.items():
        mask=np.asarray((fr.index>=lo)&(fr.index<=hi))
        on=all_on[mask[all_on]]
        off=all_off[mask[all_off]]
        # Outcome availability is symmetric between ON and control candidates.
        scored=fr[target].notna().to_numpy()
        pairs=matching(fr,on[scored[on]],off[scored[off]])
        mi=np.asarray([i for i,j in pairs],int)
        mj=np.asarray([j for i,j in pairs],int)
        result[name]={'on':stats(fr,on,target),'off':stats(fr,off,target),
                      'matched_on':stats(fr,mi,target),'match':stats(fr,mj,target),
                      'unmatched':int(sum(scored[on])-len(pairs))}
        result[name]['delta_off']=result[name]['on']['rate']-result[name]['off']['rate']
        result[name]['delta_match']=result[name]['matched_on']['rate']-result[name]['match']['rate']
        for i,j in pairs:
            pairs_out.append(dict(key=key,event=event,period=name,on_date=str(fr.index[i].date()),off_date=str(fr.index[j].date())))
    return result,pairs_out


def circular(frame,key,state='down',n=10000):
    """Vectorized 10,000 yearly circular shifts of full-year tri-state signal."""
    rng=np.random.default_rng(20261111)
    result={}
    target='Z' if state=='down' else 'H'
    for year in [2024,2025,2026]:
        sub=frame[frame.index.year==year]
        a=sub[key].to_numpy(float)
        y=sub[target].to_numpy(float)
        valid=sub[state].to_numpy(bool)&np.isfinite(y)
        obs_on=valid&(a==1); obs_off=valid&(a==0)
        obs=y[obs_on].mean()-y[obs_off].mean() if obs_on.any() and obs_off.any() else np.nan
        if len(a):
            offsets=rng.integers(0,len(a),size=n)
            ar=a[(np.arange(len(a))[None,:]-offsets[:,None])%len(a)]
            on=(ar==1)&valid; off=(ar==0)&valid
            non,nof=on.sum(1),off.sum(1)
            yy=np.nan_to_num(y)
            delta=(on*yy).sum(1)/np.maximum(non,1)-(off*yy).sum(1)/np.maximum(nof,1)
            good=(non>0)&(nof>0)
        else:
            delta=np.array([]);good=np.array([],bool)
        p=(1+int(np.sum(delta[good]>=obs-1e-12)))/(1+int(good.sum())) if np.isfinite(obs) and good.any() else np.nan
        result[str(year)]=dict(observed=obs,p=p,valid_null=int(good.sum()),attempted=n)
    return result


def holm(p):
    names=list(p)
    vals=np.array([p[k] if pd.notna(p[k]) else 1 for k in names])
    ix=np.argsort(vals,kind='stable')
    adj=np.maximum.accumulate(vals[ix]*(len(vals)-np.arange(len(vals))))
    return {names[k]:min(1,float(adj[j])) for j,k in enumerate(ix)}


def bottom_tables(frame,keys,pairs):
    ev=json.loads((ROOT.parent/'seibro_bottoms/derived/events.json').read_text(encoding='utf-8'))
    bottoms=[pd.Timestamp(d) for d in ev['B10'] if d>='2024-01-01']
    zones={}
    for d in bottoms:
        i=frame.index.get_loc(d)
        zones[str(d.date())]=((np.arange(len(frame))>=i-10)&(np.arange(len(frame))<=i+5)&
                            (frame.open<=frame.loc[d,'close']*1.05)&frame.down).to_numpy()
    rows=[]; summary={}
    for key in keys:
        event=event_indices(frame,key)
        matched=pairs[(pairs.key==key)&pairs.event.eq(True)&pairs.period.eq('all')] if len(pairs) else pd.DataFrame()
        mi=frame.index.get_indexer(pd.to_datetime(matched.off_date)) if len(matched) else np.array([],int)
        hit=np.zeros(len(frame),bool)
        mhit=np.zeros(len(frame),bool)
        caught,mcaught=0,0
        for bottom,zone in zones.items():
            indices=np.flatnonzero(zone&frame[key].eq(1).to_numpy())
            e=event[zone[event]]
            m=mi[zone[mi]]
            caught+=int(len(e)>0);mcaught+=int(len(m)>0)
            hit|=zone;mhit|=zone
            first=frame.index[indices[0]] if len(indices) else pd.NaT
            mf=frame.index[m.min()] if len(m) else pd.NaT
            bi=frame.index.get_loc(pd.Timestamp(bottom))
            rows.append(dict(key=key,bottom=bottom,first=str(first.date()) if pd.notna(first) else None,
                sessions=int(frame.index.get_loc(first)-bi) if pd.notna(first) else None,
                pct=(frame.loc[first,'open']/frame.loc[bottom,'close']-1)*100 if pd.notna(first) else None,
                event_hit=bool(len(e)),daily_hit=bool(len(indices)),match_hit=bool(len(m)),
                match_first=str(mf.date()) if pd.notna(mf) else None,
                match_sessions=int(frame.index.get_loc(mf)-bi) if pd.notna(mf) else None))
        summary[key]=dict(bottom_n=10,daily_caught=sum(r['daily_hit'] for r in rows if r['key']==key),
            caught=caught,recall=caught/10,event_n=len(event),zone_events=int(hit[event].sum()),
            precision=hit[event].mean() if len(event) else np.nan,
            match_caught=mcaught,match_recall=mcaught/10,match_event_n=len(mi),
            match_zone_events=int(mhit[mi].sum()),match_precision=mhit[mi].mean() if len(mi) else np.nan)
    return pd.DataFrame(rows),summary


def exhaustion(src,cust,px,frame):
    ev=json.loads((ROOT.parent/'seibro_bottoms/derived/events.json').read_text(encoding='utf-8'))
    rows=[];daily=[];crash=[];timing=[]
    buy5=src.SOXL_buy.reindex(px.index).rolling(5,min_periods=5).sum()
    custody=cust.sort_values('custody_date').drop_duplicates('source_date',keep='last').set_index('source_date')
    for item in ev['B10_events']:
        d=pd.Timestamp(item['bottom'])
        if d.year<2024:continue
        top=pd.Timestamp(item['top10'])
        end=px.index[min(px.index.get_loc(d)+10,len(px)-1)]
        x=src.loc[top:end].join(px[['close']]).copy()
        v=buy5.loc[top:d]
        mx=v.max(); peak=v.idxmax()
        peakbuy=x.SOXL_buy.loc[:d].max()
        ratio=buy5.loc[d]/mx if mx>0 else np.nan
        rows.append(dict(bottom=str(d.date()),top=str(top.date()),buy5_peak=str(peak.date()),
            peak_to_bottom=px.index.get_loc(d)-px.index.get_loc(peak),bottom_buy5_ratio=ratio,
            bottom_day_buy_vs_max=src.loc[d,'SOXL_buy']/peakbuy if peakbuy>0 else np.nan,
            bottom_net=src.loc[d,'SOXL_net']))
        sr=src.loc[d]
        entry=sr.entry
        timing.append(dict(bottom=str(d.date()),entry=str(entry.date()),delay=px.index.get_loc(entry)-px.index.get_loc(d),
            gain_pct=100*(px.loc[entry,'open']/px.loc[d,'close']-1),published=bool(sr.published)))
    # Description includes ongoing decline; endpoints are descriptive, never signal inputs.
    eps=[(r['bottom'],r['top']) for r in rows]+[('current','2026-10-05')]
    for label,top in eps:
        end=px.index[-1] if label=='current' else px.index[min(px.index.get_loc(pd.Timestamp(label))+10,len(px)-1)]
        x=src.reindex(px.index).loc[top:end].copy()
        x['cum_SOXL_net']=x.SOXL_net.cumsum()
        x['SOXS_share']=x.SOXS_buy/(x.SOXS_buy+x.SOXL_buy)
        x['SOXL_shares']=custody.shares.reindex(x.index,method='ffill')
        x['SOXX_ret1']=px.close.pct_change(fill_method=None).reindex(x.index)
        x['episode']=label
        daily.append(x.reset_index(names='date'))
        z=x[x.SOXX_ret1<=-.03].copy()
        z['net_change_from_previous_crash']=z.SOXL_net.diff()
        crash.append(z.reset_index(names='date'))
    pd.concat(daily).to_csv(ROOT/'results/exhaustion_daily.csv',index=False)
    pd.concat(crash).to_csv(ROOT/'results/crash_response.csv',index=False)
    pd.DataFrame(rows).to_csv(ROOT/'results/exhaustion_bottoms.csv',index=False)
    pd.DataFrame(timing).to_csv(ROOT/'results/publication_lag.csv',index=False)
    return rows,timing


def stage1(news_series=None):
    args=inputs(news_series)
    panel=assemble(*args)
    px=args[2].loc[:'2025-12-31']
    pre=panel.loc[:'2025-12-31'].join(outcomes(px))
    pre=pre.loc['2024-01-02':]
    reps={};selection={}
    for fam,keys in FAMILIES.items():
        candidates=[]
        for key in keys:
            on=pre.loc[pre.down&pre[key].eq(1),'Z'].dropna()
            off=pre.loc[pre.down&pre[key].eq(0),'Z'].dropna()
            delta=on.mean()-off.mean()
            candidates.append(dict(key=key,on_n=len(on),off_n=len(off),on=on.mean(),off=off.mean(),delta=delta,
                eligible=len(on)>=5 and len(off)>0))
        eligible=[r for r in candidates if r['eligible']]
        winner=max(eligible,key=lambda r:r['delta']) if eligible else None
        reps[fam]=winner['key'] if winner else None
        selection[fam]=candidates
    paths=[ROOT/'BRIEF_A.md',ROOT/'TASK_A.md',ROOT/'data/source_flows.csv',ROOT/'data/custody.csv',
           ROOT/'data/prices.csv',ROOT/'data/cnn.json']+sorted((ROOT/'code').glob('*.py'))
    news=Path(news_series) if news_series else ROOT.parents[3]/'research/pb_flow_census_20261026/census.csv'
    paths.append(news)
    dump(ROOT/'freeze_A.json',dict(stage='selected_using_outcomes_through_2025_only',
        created_utc=pd.Timestamp.now(tz='UTC'),cutoff='2025-12-31',representatives=reps,
        selection=selection,fingerprints=fingerprint(paths),
        raw_sha256_manifest=fingerprint(sorted((ROOT/'data/seibro').glob('*/*.xml.gz'))),
        statement='2026 is previously seen historical data, not a new holdout'))
    pre.to_csv(ROOT/'results/selection_panel_A.csv',index_label='date')
    print(json.dumps(reps))


def stage2(news_series=None):
    freeze=ROOT/'freeze_A.json'
    rel=freeze.relative_to(ROOT.parents[3]).as_posix()
    committed=subprocess.check_output(['git','show','HEAD:'+rel])
    if committed.replace(b'\r\n',b'\n')!=freeze.read_bytes().replace(b'\r\n',b'\n'):
        raise ValueError('freeze must be committed before stage2')
    fixed=json.loads(freeze.read_text(encoding='utf-8'))
    for name,sha in fixed['fingerprints'].items():
        if name.startswith('data/'):
            path=ROOT/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
                raise ValueError('Input changed after freeze: '+name)
    reps=fixed['representatives']
    args=inputs(news_series)
    frame=combo(assemble(*args),reps).join(outcomes(args[2]))
    frame.to_csv(ROOT/'results/daily_A.csv',index_label='date')
    keys=SIGNALS+['K2','K3','K4']+['S_AND_'+f for f in ['X','T','G','M','N']]
    summaries={};pairs=[]
    for key in keys:
        summaries[key]={}
        for mode in ['daily','event']:
            tab,pa=evaluate(frame,key,event=mode=='event')
            summaries[key][mode]=tab
            pairs+=pa
        summaries[key]['circular']=circular(frame,key)
    primary=[v for v in reps.values() if v]+['K3']
    rawp={key:summaries[key]['circular']['2026']['p'] for key in primary}
    # Always seven Holm slots, including untestable/missing families.
    for fam,key in reps.items():
        if not key: rawp['missing_'+fam]=np.nan
    adjusted=holm(rawp)
    dump(ROOT/'results/summary_A.json',dict(settings=34,representatives=reps,primary=primary,
        summaries=summaries,holm=adjusted,rawp=rawp))
    pdf=pd.DataFrame(pairs)
    pdf.to_csv(ROOT/'results/matches_A.csv',index=False)
    bottom,bs=bottom_tables(frame,keys,pdf)
    bottom.to_csv(ROOT/'results/bottom_hits_A.csv',index=False)
    dump(ROOT/'results/bottom_summary_A.json',bs)
    top={}
    for key in ['S1H','S1L','S5H','S5L','T5H','CH','G75']:
        top[key]={mode:evaluate(frame,key,state='high',event=mode=='event')[0] for mode in ['daily','event']}
    dump(ROOT/'results/high_aux_A.json',top)
    sens={key:{mode:evaluate(frame,key,event=mode=='event')[0] for mode in ['daily','event']} for key in ['X1_30','X1_70']}
    dump(ROOT/'results/sensitivity_A.json',sens)
    exhaustion(args[0],args[1],args[2],frame)
    events=[]
    for key in keys:
        z=frame.iloc[event_indices(frame,key)].copy()
        z['key']=key
        events.append(z.reset_index(names='date'))
    pd.concat(events).to_csv(ROOT/'results/events_A.csv',index=False)
    latest=frame.iloc[-1]
    dump(ROOT/'results/current_A.json',dict(date=frame.index[-1],down=latest.down,
        source_date=latest.source_date,dd60=latest.dd60,source_known=latest.source_known,
        custody_date=latest.custody_date,news_coverage_end=str(args[4].date.max()),
        signals={key:{'on':latest[key],'status':latest[key+'_status']} for key in SIGNALS},K=latest.K,K_min=latest.K_min,K_max=latest.K_max,
        values=latest.to_dict()))
    print('stage2 done',reps)


if __name__=='__main__':
    from pathlib import Path
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['select','evaluate'])
    ap.add_argument('--news-series')
    a=ap.parse_args()
    (stage1 if a.stage=='select' else stage2)(a.news_series)
