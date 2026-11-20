"""Independent equal-weight rank-pair search with H1-only model selection."""
from itertools import combinations
import hashlib
import json
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from panel import BASE,CLOSE,CUTOFF,GATES,LATE,OUT,load_panel

RUNS=200
SEED=20261120
ENTRIES=('E1','E6')
SIDES=('up','down')

def ecdf(values,train):
    ref=np.sort(values[train & np.isfinite(values)])
    if not len(ref): return np.full(len(values),np.nan)
    # Midrank to avoid assigning tied zeros to the maximum rank.
    out=(np.searchsorted(ref,values,side='left')+np.searchsorted(ref,values,side='right'))/(2*len(ref))
    out[~np.isfinite(values)]=np.nan
    return out

def quantiles(values,train):
    sample=values[train & np.isfinite(values)]
    if len(sample)==0: return (np.nan,np.nan)
    return np.quantile(sample,[.33,.67])

def universe(df,entry):
    return df.timing.ne('unk').to_numpy() if entry=='E1' else np.ones(len(df),bool)

def candidates(df,entry):
    eligible=universe(df,entry)
    train=df.season.isin(['2026Q1','2026Q2']).to_numpy() & eligible
    factors=BASE+(LATE if entry=='E6' else [])
    gates=GATES+(['d5_x'] if entry=='E6' else [])
    ranks={f:ecdf(df[f].to_numpy(float),train) for f in factors}
    gate_masks=[(None,None,None,np.ones(len(df),bool))]
    for f in gates:
        data=df[f].to_numpy(float)
        low,high=quantiles(data,train)
        gate_masks.extend([(f,'le',float(low),data<=low),(f,'ge',float(high),data>=high)])
    specs,masks=[],[]
    for a,b in combinations(factors,2):
        for sign in [1,-1]:
            score=(ranks[a]+(ranks[b] if sign==1 else 1-ranks[b]))/2
            lo,hi=quantiles(score,train)
            for side,cut,mask in [('le',lo,score<=lo),('ge',hi,score>=hi)]:
                for gate,gside,gcut,gmask in gate_masks:
                    if gate in (a,b): continue
                    specs.append(dict(a=a,b=b,sign=sign,score_op=side,score_cut=float(cut),gate=gate,gate_op=gside,gate_cut=gcut))
                    masks.append(eligible & mask & gmask)
    return specs,np.asarray(masks)

def wilson_lower(x,n):
    # z=1 gives a deterministic small-sample penalty.
    p=x/n
    return (p+1/(2*n)-np.sqrt(p*(1-p)/n+1/(4*n*n)))/(1+1/n)

class Selector:
    def __init__(self,df,entry,masks):
        self.masks=masks
        y=df[entry].to_numpy()
        train=df.season.isin(['2026Q1','2026Q2']).to_numpy() & np.isfinite(y)
        self.q=[train & df.season.eq(f'2026Q{q}').to_numpy() for q in (1,2)]
        self.train=train
        n1,n2=[masks[:,idx].sum(axis=1) for idx in self.q]
        self.ids=np.flatnonzero((n1>=8)&(n2>=8)&(n1+n2>=25))
        self.n1=n1[self.ids]
        self.n2=n2[self.ids]
        self.n=self.n1+self.n2
        self.hmask=masks[self.ids][:,train]
        self.signal=[np.ascontiguousarray(masks[self.ids][:,idx],dtype='float32') for idx in self.q]

    def choose(self,outcomes,side,diagnostic=False):
        wins=outcomes>0 if side=='up' else outcomes<0
        w1,w2=[arr @ np.ascontiguousarray(wins[idx],dtype='float32') for arr,idx in zip(self.signal,self.q)]
        n1,n2,n=self.n1[:,None],self.n2[:,None],self.n[:,None]
        rates=(w1/n1,w2/n2,(w1+w2)/n)
        passmask=(rates[0]>=.6)&(rates[1]>=.6)&(rates[2]>=.7)
        balanced=np.minimum(wilson_lower(w1,n1),wilson_lower(w2,n2))
        pooled=wilson_lower(w1+w2,n)
        penalties=np.maximum(0,.60-rates[0])+np.maximum(0,.60-rates[1])+np.maximum(0,.70-rates[2])
        chosen=[]
        for it in range(wins.shape[1]):
            ids=np.flatnonzero(passmask[:,it])
            is_example=not len(ids) and diagnostic
            if is_example: ids=np.flatnonzero(np.isclose(penalties[:,it],penalties[:,it].min(),atol=1e-12,rtol=0))
            order=ids[np.lexsort((self.ids[ids],-pooled[ids,it],-balanced[ids,it]))]
            picked=[]
            for j in order:
                if any(2*np.count_nonzero(self.hmask[j]&self.hmask[p])>=min(self.n[j],self.n[p]) for p in picked):
                    continue
                picked.append(j)
                if len(picked)==(1 if is_example else 3): break
            chosen.append({'ids':[int(self.ids[p]) for p in picked],'diagnostic':bool(is_example),
                           'passed_h1_before_cap':int(passmask[:,it].sum())})
        return chosen

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def select():
    df=load_panel(train_only=True)
    lock={}
    for entry in ENTRIES:
        specs,masks=candidates(df,entry)
        model=Selector(df,entry,masks)
        for side in SIDES:
            sel=model.choose(df[entry].to_numpy()[:,None],side,diagnostic=True)[0]
            sel['specs']=[specs[i] for i in sel['ids']]
            sel['settings']=len(specs)
            sel['support_qualified']=len(model.ids)
            lock[f'{entry}_{side}']=sel
    path=OUT/'results.json'
    if path.exists():
        assert json.loads(path.read_bytes())['lock_hash']==fingerprint(lock),'H1 lock changed'
        return
    obj=dict(locked_at=datetime.now().astimezone().isoformat(),lock_hash=fingerprint(lock),selection=lock)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    for k,v in lock.items(): print(k,v['settings'],v['passed_h1_before_cap'],v['ids'],v['diagnostic'])

def permute(df):
    """One joint permutation per season/maturity stratum preserves the E1/E6 relationship."""
    rng=np.random.default_rng(SEED)
    y=df[list(ENTRIES)].to_numpy()
    simulated=np.repeat(y[:,:,None],RUNS,axis=2)
    flag=np.isfinite(y[:,0]).astype(int)+2*np.isfinite(y[:,1])
    # Early observations are never part of the permutation pool.
    grouped=df.assign(flag=flag).loc[df.reaction_date<=CUTOFF].groupby(['season','flag']).groups
    for i in range(RUNS):
        for idx in grouped.values():
            ix=np.asarray(idx)
            simulated[ix,:,i]=y[rng.permutation(ix),:]
    return simulated

def stats(df,mask,y,side):
    good=mask & np.isfinite(y)
    values=y[good]
    n=len(values)
    counts=df.ticker[good].value_counts()
    wins=int((values>0 if side=='up' else values<0).sum())
    return dict(n=n,wins=wins,rate=wins/n if n else None,tickers=len(counts),
                max_ticker_share=float(counts.max()/n) if n else None,
                mean_x=float(values.mean()) if n else None,median_x=float(np.median(values)) if n else None)

def buckets(df):
    return dict(H1=df.season.isin(['2026Q1','2026Q2']).to_numpy(),
                H2=(df.season.isin(['2026Q3','2026Q4'])&(df.reaction_date<=CUTOFF)).to_numpy(),
                **{f'Q{q}':(df.season.eq(f'2026Q{q}') & (df.reaction_date<=CUTOFF)).to_numpy() for q in range(1,5)},
                early=(df.reaction_date>CUTOFF).to_numpy())

def strict_check(s,side):
    return (s['n']>=15 and s['rate']>=.7 and s['tickers']>=8 and s['max_ticker_share']<=.25
            and (s['mean_x']>0 if side=='up' else s['mean_x']<0))

def groups(masks):
    # Connected components prevent overlaps from counting as independent successes.
    components=[]
    for m in masks:
        intersect=[i for i,component in enumerate(components) if any(
            2*np.count_nonzero(m & other)>=min(m.sum(),other.sum()) for other in component)]
        merged=[m]
        for i in reversed(intersect): merged+=components.pop(i)
        components.append(merged)
    return len(components)

def evaluate():
    path=OUT/'results.json'
    obj=json.loads(path.read_bytes())
    assert 'confirmation' not in obj,'Confirmation already materialized'
    df=load_panel()
    assert len(df)==600
    bands=buckets(df)
    simulations=permute(df)
    detailed=[]
    base={}
    real={k:[] for k in ('hit70','strict')}
    fake={k:[[] for _ in range(RUNS)] for k in real}
    family_null={}
    for eix,entry in enumerate(ENTRIES):
        specs,masks=candidates(df,entry)
        model=Selector(df,entry,masks)
        y=df[entry].to_numpy()
        sims=simulations[:,eix,:]
        for side in SIDES:
            family=f'{entry}_{side}'
            lock=obj['selection'][family]
            assert lock['specs']==[specs[j] for j in lock['ids']]
            base[family]={part:stats(df,universe(df,entry)&mask,y,side) for part,mask in bands.items()}
            for rank,rid in enumerate(lock['ids'],1):
                mask=masks[rid]
                all_stats={p:stats(df,mask&bm,y,side) for p,bm in bands.items()}
                h2=all_stats['H2']
                hit=not lock['diagnostic'] and h2['n']>=15 and h2['rate']>=.70
                strong=hit and strict_check(h2,side)
                valid=mask&bands['H2']&np.isfinite(y)
                if hit:real['hit70'].append(valid)
                if strong:real['strict'].append(valid)
                rec={'family':family,'rank':rank,'id':rid,'diagnostic':lock['diagnostic'],
                     'spec':specs[rid],'stats':all_stats,'hit70':bool(hit),'strict':bool(strong),
                     'early_triggered':int((mask&bands['early']).sum()),
                     'early_ready':int((mask&bands['early']&df[entry+'_ready']).sum()),
                     'early_entered':int((mask&bands['early']&df[entry+'_entered']).sum())}
                detailed.append(rec)
            selections=model.choose(sims,side)
            family_null[family]={'hit70':[],'strict':[],'h1_picked':[]}
            for it,sel in enumerate(selections):
                counters=dict(hit70=0,strict=0)
                for rid in sel['ids']:
                    valid=masks[rid]&bands['H2']&np.isfinite(y)
                    v=sims[valid,it]
                    if len(v)<15 or (v>0 if side=='up' else v<0).mean()<.7: continue
                    counters['hit70']+=1
                    fake['hit70'][it].append(valid)
                    s=stats(df,valid,sims[:,it],side)
                    if strict_check(s,side):
                        counters['strict']+=1
                        fake['strict'][it].append(valid)
                for k in counters: family_null[family][k].append(counters[k])
                family_null[family]['h1_picked'].append(len(sel['ids']))
    def rollup(pools):
        return [groups(x) for x in pools]
    null={}
    for k in real:
        raw=[len(x) for x in fake[k]]
        merged=rollup(fake[k])
        null[k]=dict(raw=raw,clusters=merged,observed_raw=len(real[k]),observed_clusters=groups(real[k]),
                     trials_with_any=sum(v>0 for v in raw),p95=float(np.quantile(merged,.95)),
                     empirical_p=(1+sum(v>=groups(real[k]) for v in merged))/(RUNS+1))
    obj.update(confirmation=detailed,baseline=base,null=null,null_by_family=family_null,
               repetitions=RUNS,seed=SEED,study_rows=len(df),evaluated_at=datetime.now().astimezone().isoformat())
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    flat=[]
    for c in detailed:
        row={k:v for k,v in c.items() if k not in ['spec','stats']}
        row.update(c['spec'])
        for band,stats_ in c['stats'].items(): row.update({band+'_'+k:v for k,v in stats_.items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(OUT/'candidates.csv',index=False)

if __name__=='__main__':
    with threadpool_limits(limits=2):
        {'select':select,'evaluate':evaluate}[sys.argv[1]]()

