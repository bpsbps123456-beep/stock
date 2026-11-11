"""Post-freeze prices, conservative entries and prespecified comparisons."""
import csv,json,hashlib,subprocess
from collections import Counter
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from collect_v2 import NEWS,V2,save,csvwrite

OUT=V2/'results';PRICE=V2/'prices'

def verify_freeze():
 f=json.loads((NEWS/'freeze_B_v2.json').read_text(encoding='utf-8'))
 for p,h in f['input_hashes'].items():
  if hashlib.sha256((NEWS/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h:raise ValueError('Frozen input changed: '+p)
 committed=subprocess.check_output(['git','log','-1','--format=%H','--',str(NEWS/'freeze_B_v2.json')],text=True).strip()
 if not committed:raise ValueError('Freeze not committed')
 return committed

def fetch():
 freeze=verify_freeze();PRICE.mkdir(exist_ok=True)
 import yfinance as yf
 meta=dict(fetched_at=datetime.now(timezone.utc).isoformat(),freeze_commit=freeze,provider='Yahoo Finance via yfinance; fresh full history, repair=False',tickers={})
 for ticker in ['SOXX','QQQ']:
  p=PRICE/(ticker+'_full.csv')
  if p.exists():continue
  d=yf.Ticker(ticker).history(period='max',auto_adjust=False,actions=True,repair=False)
  if d.empty:raise ValueError('Empty prices '+ticker)
  d.index=d.index.tz_localize(None).normalize();d.index.name='date';d.to_csv(p)
  meta['tickers'][ticker]=dict(rows=len(d),start=str(d.index.min().date()),end=str(d.index.max().date()),splits={str(k.date()):v for k,v in d['Stock Splits'].items() if v})
 save(PRICE/'fetch_metadata.json',meta)

def load(ticker):
 d=pd.read_csv(PRICE/(ticker+'_full.csv'),index_col='date',parse_dates=True)
 d=d.loc[d.index<='2026-11-10'].copy()
 ratio=d['Adj Close']/d['Close']
 for c in ['Open','High','Low','Close']:d[c]=d[c]*ratio
 d=d.dropna(subset=['Open','Close']);return d

def features(d):
 z=d.copy();prev=z['Close'].shift(1)
 z['dd60']=prev/prev.rolling(60).max()-1;z['lag5']=prev/z['Close'].shift(6)-1
 z['state']=z['dd60']<=-.10
 for h in [5,10,20]:z['r'+str(h)]=z['Close'].shift(-(h-1))/z['Open']-1
 z['min20']=z['Close'][::-1].rolling(20,min_periods=20).min()[::-1]/z['Open']-1
 z['near']=np.where(z['min20'].notna(),(z['min20']>=-.05).astype(float),np.nan)
 return z

def entry_index(index,published_date,published_at=''):
 if not published_date:return None
 day=pd.Timestamp(published_date)
 # Evidence version uses dates only. Clock-qualified sources would use
 # same open only when known public before 09:30 America/New_York.
 side='right'
 if published_at:
  t=pd.Timestamp(published_at).tz_convert('America/New_York')
  day=t.tz_localize(None).normalize()
  if (t.hour,t.minute)<(9,30):side='left'
 i=index.searchsorted(day,side=side)
 return int(i) if i<len(index) else None

def matched_indices(z,indices):
 matched=[]
 for i in indices:
  r=z.iloc[i]
  if pd.isna(r['near']):continue
  pool=z[(z.index.year==z.index[i].year)&(z['state']==r['state'])&z['near'].notna()].dropna(subset=['dd60','lag5'])
  pool=pool.loc[pool.index!=z.index[i]]
  if pool.empty:continue
  scale=z.loc[(z.index.year==z.index[i].year)&z['near'].notna(),['dd60','lag5']].std(ddof=0).replace(0,1)
  dist=(((pool[['dd60','lag5']]-r[['dd60','lag5']])/scale)**2).sum(axis=1)
  matched.append(int(z.index.get_loc(dist.idxmin())))
 return matched

def summary(z,indices):
 sub=z.iloc[indices]
 valid=sub['near'].dropna()
 return dict(n=len(sub),complete20=len(valid),near_count=int(valid.sum()),near_fraction=float(valid.mean()) if len(valid) else None,returns={str(h):dict(n=int(sub['r'+str(h)].count()),mean=float(sub['r'+str(h)].mean()) if sub['r'+str(h)].count() else None) for h in [5,10,20]})

def cluster_indices(z,calls):
 cuts={}
 for r in calls:
  if r['action'] not in {'downgrade','target_cut'}:continue
  i=entry_index(z.index,r['published_date'],r['published_at'])
  if i is not None:cuts.setdefault(i,set()).add(r['house'])
 out=[];block=-1
 for i in range(len(z)):
  houses=set().union(*(cuts.get(k,set()) for k in range(max(0,i-4),i+1)))
  if len(houses)>=2 and i>block:
   out.append(i);block=i+4
 return out

def primary_test(z,indices,rng):
 yearidx=np.flatnonzero(z.index.year==2026);sub=z.iloc[yearidx]
 signal=np.zeros(len(sub),dtype=int)
 for i in indices:
  j=np.searchsorted(yearidx,i)
  if j<len(yearidx) and yearidx[j]==i:signal[j]+=1
 eligible=(sub['state']&sub['near'].notna()).to_numpy();y=sub['near'].fillna(0).to_numpy()
 base=float(y[eligible].mean()) if eligible.any() else np.nan
 w=signal*eligible;n=int(w.sum());effect=float((w@y)/n-base) if n else None
 null=[]
 for shift in rng.integers(0,len(signal),10000):
  shifted=np.roll(signal,int(shift))*eligible
  if shifted.sum():null.append(float(shifted@y/shifted.sum()-base))
 p=(1+sum(v>=effect-1e-12 for v in null))/(1+len(null)) if effect is not None else None
 return dict(n=n,near_count=int(w@y),base_count=int(y[eligible].sum()),base_n=int(eligible.sum()),effect=effect,p_raw=p,valid_permutations=len(null),distinct_shifts=len(signal),signal_days=int((signal>0).sum()))

def analyze():
 freeze=verify_freeze();OUT.mkdir(exist_ok=True)
 calls=list(csv.DictReader((V2/'calls_census.csv').open(encoding='utf-8-sig')))
 results={};joined=[]
 for ticker in ['SOXX','QQQ']:
  d=load(ticker);z=features(d);z.to_csv(OUT/(ticker+'_sessions.csv'))
  if ticker=='QQQ':
   primary=features(load('SOXX'))
   for column in ['dd60','lag5','state']:z[column]=primary[column].reindex(z.index)
   z.to_csv(OUT/(ticker+'_sessions.csv'))
  if ticker=='SOXX':
   raw=pd.read_csv(PRICE/'SOXX_full.csv',index_col='date',parse_dates=True)
   split=float(raw.loc['2026-11-05','Stock Splits']) if pd.Timestamp('2026-11-05') in raw.index else None
   around=d.loc['2026-11-04':'2026-11-06','Close'];ratios=around.pct_change().dropna()
   check=dict(reported_split_20261105=split,adjusted_closes={str(k.date()):v for k,v in around.items()},largest_abs_adj_change=float(ratios.abs().max()) if len(ratios) else None,pass_continuity=bool(len(ratios) and ratios.abs().max()<.5),event_matches_three_to_one=split==3)
   save(OUT/'split_check.json',check)
   if not check['pass_continuity']:raise ValueError('Split continuity check failed')
  mapping={r['id']:entry_index(z.index,r['published_date'],r['published_at']) for r in calls}
  grid=[]
  for year in (2024,2025,2026):
   for action in ['upgrade','downgrade','target_raise','target_cut']:
    for state in [False,True]:
     idx=[mapping[r['id']] for r in calls if r['action']==action and mapping[r['id']] is not None and z.index[mapping[r['id']]].year==year and bool(z.iloc[mapping[r['id']]]['state'])==state]
     controls=np.flatnonzero((z.index.year==year)&(z['state']==state)&z['near'].notna()).tolist();matched=matched_indices(z,idx)
     grid.append(dict(year=year,action=action,decline_state=state,signals=summary(z,idx),all_days=summary(z,controls),matched=summary(z,matched),matched_unique=len(set(matched)),matched_reuses=len(matched)-len(set(matched))))
  upgrades=[mapping[r['id']] for r in calls if r['action']=='upgrade' and mapping[r['id']] is not None]
  clusters=cluster_indices(z,calls)
  rng=np.random.default_rng(20261111)
  hu=primary_test(z,upgrades,rng);hd=primary_test(z,clusters,rng)
  valid=[(i,t) for i,t in enumerate([hu,hd]) if t['p_raw'] is not None];valid.sort(key=lambda x:x[1]['p_raw'])
  prev=0
  for rank,(i,t) in enumerate(valid):prev=max(prev,min(1,t['p_raw']*(2-rank)));t['p_holm']=prev
  for t in [hu,hd]:t.setdefault('p_holm',None)
  hyp=[]
  for year in (2024,2025,2026):
   for name,idx in [('H-U',upgrades),('H-D',clusters)]:
    chosen=[i for i in idx if z.index[i].year==year and z.iloc[i]['state']]
    matched=matched_indices(z,chosen)
    hyp.append(dict(year=year,hypothesis=name,signals=summary(z,chosen),matched=summary(z,matched),dates=[str(z.index[i].date()) for i in chosen]))
  results[ticker]=dict(grid=grid,hypotheses=hyp,H_U_2026=hu,H_D_2026=hd,cluster_dates=[str(z.index[i].date()) for i in clusters if z.index[i].year>=2024])
  if ticker=='SOXX':
   anchors=json.loads((NEWS/'protocol_B_v2.json').read_text(encoding='utf-8'))['search_B2']['anchors']['real']
   for r in calls:
    out=dict(r);i=mapping[r['id']]
    if r['published_date']:
     p=int(z.index.searchsorted(pd.Timestamp(r['published_date']),side='left'));b=min(anchors,key=lambda a:abs(p-z.index.searchsorted(pd.Timestamp(a))))
     out.update(nearest_real_bottom=b,bottom_signed_sessions=p-int(z.index.searchsorted(pd.Timestamp(b))))
    if i is not None:
     out.update(entry_date=str(z.index[i].date()),dd60=z.iloc[i]['dd60'],lag5=z.iloc[i]['lag5'],decline_state=bool(z.iloc[i]['state']),near20='' if pd.isna(z.iloc[i]['near']) else int(z.iloc[i]['near']),min20=z.iloc[i]['min20'],return5=z.iloc[i]['r5'],return10=z.iloc[i]['r10'],return20=z.iloc[i]['r20'])
    joined.append(out)
   current=d.loc['2026-10-05':,'Close'];save(OUT/'current_anchor.json',dict(high_start='2026-10-05',lowest_close_date=str(current.idxmin().date()),asof=str(d.index.max().date()),recent10=[str(x.date()) for x in d.index[-10:]]))
 results['freeze_commit']=freeze;results['undefined_dates']=sum(not r['published_date'] for r in calls)
 save(OUT/'analysis_B1.json',results);csvwrite(OUT/'calls_with_prices.csv',joined)
 print(json.dumps({t:{k:results[t][k] for k in ['H_U_2026','H_D_2026']} for t in ['SOXX','QQQ']},indent=2))

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('command',choices=['fetch','analyze']);a=p.parse_args();fetch() if a.command=='fetch' else analyze()
