"""Independent outcome, match, null, Holm and bottom census recomputation.

Does not import run_A or panel_A. Signals are audited by causal unit tests;
this verifier independently recomputes the seven primary result paths.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    f=pd.read_csv(ROOT/'results/daily_A.csv',parse_dates=['date']).set_index('date')
    px=pd.read_csv(ROOT/'data/prices.csv',parse_dates=['date'])
    px=px[px.ticker=='SOXX'].set_index('date')
    summary=json.loads((ROOT/'results/summary_A.json').read_text(encoding='utf-8'))
    bottom_summary=json.loads((ROOT/'results/bottom_summary_A.json').read_text(encoding='utf-8'))
    detail=pd.read_csv(ROOT/'results/bottom_hits_A.csv')
    mismatch=[];checks=0
    def check(label,a,c):
        nonlocal checks
        checks+=1
        if a is None:a=np.nan
        if c is None:c=np.nan
        if isinstance(a,(int,float,np.number)) or isinstance(c,(int,float,np.number)):
            ok=np.isclose(a,c,atol=1e-10,rtol=1e-9,equal_nan=True)
        else:ok=a==c
        if not ok:mismatch.append(dict(label=label,actual=str(a),expected=str(c)))
    z=[]
    for day in f.index:
        if day not in px.index or day<pd.Timestamp('2024-01-02'):
            z.append(np.nan);continue
        i=px.index.get_loc(day)
        prices=px.close.iloc[i:i+20]
        low=prices.min()/px.loc[day,'open']-1
        z.append(0. if low<-.05-1e-12 else (1. if len(prices)==20 else np.nan))
    z=np.array(z)
    for i,v in enumerate(z):check('daily_Z_'+str(i),v,f.Z.iloc[i])
    def events(key,on=1):
        out=[]
        for i,row in enumerate(f.itertuples()):
            if row.Index>=pd.Timestamp('2024-01-02') and row.down and pd.notna(row.open) and f[key].iloc[i]==on:
                if not out or i-out[-1]>=10:out.append(i)
        return out
    def match(key,on,off):
        out=[]
        for yr in sorted(set(f.index[on].year)):
            ix=(f.index.year==yr)&f.down
            scale=f.loc[ix,['dd60','ret5','ret1']].std(ddof=1).replace(0,1).to_numpy()
            pts=f[['dd60','ret5','ret1']].to_numpy()/scale
            pool=[j for j in off if f.index[j].year==yr]
            for i in on:
                if f.index[i].year!=yr:continue
                distances=sorted((float(np.dot(pts[i]-pts[j],pts[i]-pts[j])),j) for j in pool if abs(i-j)>2)
                if distances:
                    j=distances[0][1];out.append((i,j));pool.remove(j)
        return out
    rawp={}
    saved_pairs=pd.read_csv(ROOT/'results/matches_A.csv')
    for key in summary['primary']:
        ix=(f.index.year==2026)&f.down
        on=np.flatnonzero(ix&f[key].eq(1))
        off=np.flatnonzero(ix&f[key].eq(0))
        ons=on[np.isfinite(z[on])];offs=off[np.isfinite(z[off])]
        pairs=match(key,ons,offs)
        exp=summary['summaries'][key]['daily']['2026']
        for role,indices in [('on',on),('off',off),('matched_on',[i for i,j in pairs]),('match',[j for i,j in pairs])]:
            zz=z[indices]
            check(key+'_'+role+'_n',len(indices),exp[role]['n'])
            check(key+'_'+role+'_scored',np.isfinite(zz).sum(),exp[role]['scored'])
            check(key+'_'+role+'_rate',np.nanmean(zz) if np.isfinite(zz).any() else np.nan,exp[role]['rate'])
        periods={'2024':('2024-01-02','2024-12-31'),'2025':('2025-01-01','2025-12-31'),
                 '2026H1':('2026-01-01','2026-06-30'),'2026H2':('2026-07-01','2026-12-31'),
                 '2026':('2026-01-01','2026-12-31'),'all':('2024-01-02','2026-12-31')}
        for mode in ['daily','event']:
            allon=events(key) if mode=='event' else np.flatnonzero(f.down&f[key].eq(1)).tolist()
            alloff=events(key,0) if mode=='event' else np.flatnonzero(f.down&f[key].eq(0)).tolist()
            for label,(lo,hi) in periods.items():
                oni=[i for i in allon if lo<=str(f.index[i].date())<=hi]
                ofi=[i for i in alloff if lo<=str(f.index[i].date())<=hi]
                pp=match(key,[i for i in oni if np.isfinite(z[i])],[i for i in ofi if np.isfinite(z[i])])
                expected=summary['summaries'][key][mode][label]
                for role,indices in [('on',oni),('off',ofi),('matched_on',[i for i,j in pp]),('match',[j for i,j in pp])]:
                    zz=z[indices]
                    check(f'{key}_{mode}_{label}_{role}_n',len(indices),expected[role]['n'])
                    check(f'{key}_{mode}_{label}_{role}_scored',np.isfinite(zz).sum(),expected[role]['scored'])
                    check(f'{key}_{mode}_{label}_{role}_rate',np.nanmean(zz) if np.isfinite(zz).any() else np.nan,expected[role]['rate'])
                sp=saved_pairs[(saved_pairs.key==key)&saved_pairs.event.eq(mode=='event')&saved_pairs.period.eq(label)]
                actual_dates=[(str(f.index[i].date()),str(f.index[j].date())) for i,j in pp]
                check(f'{key}_{mode}_{label}_pair_dates',str(actual_dates),str(list(zip(sp.on_date,sp.off_date))))
        rng=np.random.default_rng(20261111)
        for yr in [2024,2025,2026]:
            sub=f[f.index.year==yr]
            arr=sub[key].to_numpy()
            yy=z[f.index.year==yr]
            valid=sub.down.to_numpy()&np.isfinite(yy)
            offs_draw=rng.integers(0,len(arr),10000)
            hist=np.bincount(offs_draw,minlength=len(arr))
            rates=[];weights=[]
            for offset in np.flatnonzero(hist):
                shifted=np.roll(arr,offset)
                a=yy[valid&(shifted==1)];b=yy[valid&(shifted==0)]
                if len(a) and len(b):rates.append(a.mean()-b.mean());weights.append(hist[offset])
            a=yy[valid&(arr==1)];b=yy[valid&(arr==0)]
            obs=a.mean()-b.mean() if len(a) and len(b) else np.nan
            n=sum(weights)
            p=(1+sum(w for v,w in zip(rates,weights) if v>=obs-1e-12))/(1+n) if np.isfinite(obs) and n else np.nan
            check(f'{key}_{yr}_p',p,summary['summaries'][key]['circular'][str(yr)]['p'])
            if yr==2026:rawp[key]=p
    for key in summary['rawp']:
        rawp.setdefault(key,np.nan)
    ordered=sorted(rawp,key=lambda k:rawp[k] if np.isfinite(rawp[k]) else 1.)
    highest=0
    for pos,key in enumerate(ordered):
        p=rawp[key] if np.isfinite(rawp[key]) else 1.
        highest=max(highest,p*(7-pos))
        check(key+'_Holm',min(1,highest),summary['holm'][key])
    ev=json.loads((ROOT.parent/'seibro_bottoms/derived/events.json').read_text(encoding='utf-8'))
    bottoms=[pd.Timestamp(d) for d in ev['B10'] if d>='2024-01-01']
    for key in bottom_summary:
        e=events(key)
        on=[i for i in e if np.isfinite(z[i])]
        off=[i for i in events(key,0) if np.isfinite(z[i])]
        pairs=match(key,on,off)
        mi=[j for i,j in pairs]
        union=set();caught=mcaught=dailycaught=0
        for bottom in bottoms:
            bi=f.index.get_loc(bottom)
            zone={i for i in range(max(0,bi-10),min(len(f),bi+6)) if f.down.iloc[i] and f.open.iloc[i]<=px.loc[bottom,'close']*1.05}
            union|=zone
            hits=sorted(i for i in zone if f[key].iloc[i]==1)
            eh=sorted(set(e)&zone);mh=sorted(set(mi)&zone)
            caught+=bool(eh);mcaught+=bool(mh);dailycaught+=bool(hits)
            row=detail[(detail.key==key)&detail.bottom.eq(str(bottom.date()))].iloc[0]
            check(key+str(bottom)+'_event',bool(eh),bool(row.event_hit))
            check(key+str(bottom)+'_match',bool(mh),bool(row.match_hit))
            first=str(f.index[hits[0]].date()) if hits else None
            expected=None if pd.isna(row['first']) else row['first']
            check(key+str(bottom)+'_first',first,expected)
            mfirst=str(f.index[mh[0]].date()) if mh else None
            check(key+str(bottom)+'_match_first',mfirst,None if pd.isna(row.match_first) else row.match_first)
            if mh:check(key+str(bottom)+'_match_sessions',mh[0]-bi,row.match_sessions)
            if hits:
                check(key+str(bottom)+'_sessions',hits[0]-bi,row.sessions)
                check(key+str(bottom)+'_pct',100*(f.open.iloc[hits[0]]/px.loc[bottom,'close']-1),row.pct)
        out=bottom_summary[key]
        for label,val in dict(caught=caught,daily_caught=dailycaught,match_caught=mcaught,event_n=len(e),
                zone_events=len(set(e)&union),match_event_n=len(mi),match_zone_events=len(set(mi)&union)).items():
            check(key+'_'+label,val,out[label])
        for label,val in dict(recall=caught/10,match_recall=mcaught/10,
                precision=len(set(e)&union)/len(e) if e else np.nan,
                match_precision=len(set(mi)&union)/len(mi) if mi else np.nan).items():
            check(key+'_'+label,val,out[label])
    result=dict(passed=not mismatch,comparisons=checks,mismatches=mismatch,primary_paths=len(summary['primary']),
        bottom_settings=len(bottom_summary),method='No target imports; raw-price outcome, scalar greedy matching, unique-offset weighted null, independent bottom census')
    (ROOT/'results/verify_A.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))
    if mismatch:raise SystemExit(1)


if __name__=='__main__':main()
