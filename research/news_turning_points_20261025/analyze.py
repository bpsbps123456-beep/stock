"""Fixed-definition retrospective news event study. No live trading signals.

Run from repository root. Cached Yahoo OHLC input lives in --data-dir.
All dates are NYSE sessions. Events/definitions must match freeze.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parents[1] / 'data/derived/analysis/news_turning_points_20261025'
HORIZONS = (1, 3, 5, 10, 20)


def pivots(values: np.ndarray, threshold: float) -> list[dict]:
    """Return confirmed close-to-close directional-change extremes only.

    A peak is the highest close before a threshold drawdown; a trough is
    the lowest close before a threshold rebound. confirm_i is explicitly
    later than i. The terminal candidate is never returned.
    """
    result = []
    direction = 0
    lo_i = hi_i = 0
    for i in range(1, len(values)):
        x = values[i]
        if direction == 0:
            if x < values[lo_i]: lo_i = i
            if x > values[hi_i]: hi_i = i
            if x >= values[lo_i] * (1 + threshold):
                result.append(dict(kind='bottom', i=lo_i, confirm_i=i))
                direction, hi_i = 1, i
            elif x <= values[hi_i] * (1 - threshold):
                result.append(dict(kind='top', i=hi_i, confirm_i=i))
                direction, lo_i = -1, i
        elif direction == 1:
            if x > values[hi_i]: hi_i = i
            if x <= values[hi_i] * (1 - threshold):
                result.append(dict(kind='top', i=hi_i, confirm_i=i))
                direction, lo_i = -1, i
        else:
            if x < values[lo_i]: lo_i = i
            if x >= values[lo_i] * (1 + threshold):
                result.append(dict(kind='bottom', i=lo_i, confirm_i=i))
                direction, hi_i = 1, i
    return result


def wilson(k: int, n: int) -> list[float] | None:
    if not n: return None
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n)) / den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [float(mid-half), float(mid+half)]


def run(data: Path) -> dict:
    data.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((HERE/'freeze.json').read_text(encoding='utf-8'))
    for name, expected in freeze['files'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == expected, name
    # A sharp new low must replace an earlier candidate; an unfinished low
    # must not become a historical success at the data boundary.
    assert pivots(np.array([100., 120., 108., 90., 100., 95.]), .08) == [
        {'kind':'bottom','i':0,'confirm_i':1},
        {'kind':'top','i':1,'confirm_i':2},
        {'kind':'bottom','i':3,'confirm_i':4},
    ]
    cal = xcals.get_calendar('XNYS', start='2023-12-01', end='2026-11-30')
    sessions = cal.sessions_in_range('2023-12-01', '2026-10-23').tz_localize(None)
    price_meta=json.loads((HERE/'prices_meta.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((HERE/'prices.csv').read_bytes()).hexdigest()==price_meta['sha256']
    price_snapshot=pd.read_csv(HERE/'prices.csv',header=[0,1],index_col=0,parse_dates=True)
    frames = {name:price_snapshot[name].reindex(sessions) for name in ['open','close','low','high']}
    for name, frame in frames.items():
        assert frame[['SOXX','QQQ','SPY']].notna().all().all(), f'missing {name}'
        assert (frame[['SOXX','QQQ','SPY']] > 0).all().all()
    op, cl, low, high = (frames[name] for name in ['open','close','low','high'])
    assert (low[['SOXX','QQQ','SPY']] <= op[['SOXX','QQQ','SPY']]+.001).all().all()
    assert (high[['SOXX','QQQ','SPY']] >= cl[['SOXX','QQQ','SPY']]-.001).all().all()
    n = len(sessions)
    piv = {}
    for pct in [8,12]:
        p = pd.DataFrame(pivots(cl.SOXX.to_numpy(), pct/100))
        p['date'] = [sessions[i].date().isoformat() for i in p.i]
        p['confirmed_date'] = [sessions[i].date().isoformat() for i in p.confirm_i]
        p['close'] = [float(cl.SOXX.iloc[i]) for i in p.i]
        p['threshold'] = pct
        piv[pct] = p
        p.to_csv(data/f'pivots_{pct}.csv',index=False,encoding='utf-8-sig')
    daily = pd.DataFrame(index=sessions)
    daily['i'] = np.arange(n)
    daily['year'] = sessions.year
    # All covariates are known at yesterday's close, before today's open.
    daily['dd20'] = (cl.SOXX/cl.SOXX.rolling(20,min_periods=20).max()-1).shift(1)
    daily['bucket'] = pd.cut(-daily.dd20,[-np.inf,.05,.10,np.inf],right=False,labels=[0,1,2]).astype('Int64')
    fall = cl.SOXX.pct_change(5,fill_method=None) < 0
    yield_fall = cl['^TNX'].diff(5) < 0
    daily['M_at_entry'] = (fall & yield_fall).shift(1,fill_value=False)
    for asset in ['SOXX','QQQ','SPY']:
        for h in HORIZONS:
            daily[f'{asset}_r{h}'] = cl[asset].shift(-(h-1))/op[asset]-1
    # Opening auction to fifth session opening auction: both are within
    # user's 08-24 KST window, but still excludes spread/slippage.
    daily['SOXX_open5'] = op.SOXX.shift(-4)/op.SOXX-1
    for i in range(n):
        if i+5 <= n:
            daily.loc[sessions[i],'mae5'] = min(0,float(low.SOXX.iloc[i:i+5].min()/op.SOXX.iloc[i]-1))
            daily.loc[sessions[i],'mfe5'] = max(0,float(high.SOXX.iloc[i:i+5].max()/op.SOXX.iloc[i]-1))
    for pct, p in piv.items():
        for kind in ['bottom','top']:
            subset = p[p.kind==kind]
            for i in range(n):
                near = subset.iloc[np.argmin(abs(subset.i.to_numpy()-i))]
                delta = i-int(near.i)
                daily.loc[sessions[i],f'{kind}_distance{pct}'] = delta
                daily.loc[sessions[i],f'{kind}_date{pct}'] = near.date
                # Latest five-session windows are not scored.
                daily.loc[sessions[i],f'{kind}_near{pct}'] = float(abs(delta)<=5) if i+5 < n else np.nan
    base = daily[(daily.year>=2024)&(daily.year<=2026)].copy()
    events = json.loads((HERE/'events.json').read_text(encoding='utf-8'))
    rows = []
    previous = {}
    for event in sorted(events,key=lambda e:(e['date'],e['id'])):
        i = int(sessions.searchsorted(pd.Timestamp(event['date']), side='right'))
        if i >= n: continue
        fam = event['family']
        clustered = event['include'] and fam in previous and i-previous[fam] <= 5
        keep = event['include'] and not clustered
        if keep: previous[fam] = i
        r = dict(event, entry=sessions[i].date().isoformat(), clustered=clustered, keep=keep,
                 **daily.iloc[i].to_dict())
        r['direction'] = -1 if fam=='T' else 1
        rows.append(r)
    news = pd.DataFrame(rows)
    news.to_csv(data/'event_outcomes.csv',index=False,encoding='utf-8-sig')
    macro_rows=[]
    for fam, condition in [('M',fall&yield_fall),('C',fall)]:
        onset=condition & ~condition.shift(1,fill_value=False)
        prev=-100
        for k in range(1,n-1):
            i=k+1
            if sessions[k].year < 2024 or not onset.iloc[k] or i-prev < 5: continue
            prev=i
            macro_rows.append(dict(id=f'{fam}-{sessions[k].date()}',family=fam,date=str(sessions[k].date()),
                entry=str(sessions[i].date()),direction=1,keep=True,**daily.iloc[i].to_dict()))
    macro=pd.DataFrame(macro_rows)
    macro.to_csv(data/'macro_outcomes.csv',index=False,encoding='utf-8-sig')

    def metrics(group: pd.DataFrame, direction: int) -> dict:
        valid = group.dropna(subset=['SOXX_r5'])
        count=len(valid)
        success=int((valid.SOXX_r5*direction>0).sum())
        result=dict(n=len(group),n5=count,wins5=success,hit5=success/count if count else None,
                    hit5_wilson95=wilson(success,count),
                    mean5=float(valid.SOXX_r5.mean()) if count else None,
                    median5=float(valid.SOXX_r5.median()) if count else None,
                    mean_mae5=float(valid.mae5.mean()) if count else None,
                    worst_mae5=float(valid.mae5.min()) if count else None,
                    open5_hit=float((valid.SOXX_open5*direction>0).mean()) if count else None,
                    open5_mean=float(valid.SOXX_open5.mean()) if count else None)
        matched_hit=[]; matched_ret=[]; all_hit=[]; all_ret=[]; near_base=[]
        kind='top' if direction==-1 else 'bottom'
        for _,r in valid.iterrows():
            yy=base[(base.year==r.year)&base.SOXX_r5.notna()]
            bb=yy[yy.bucket==r.bucket]
            assert len(bb)>0
            all_hit.append((yy.SOXX_r5*direction>0).mean()); all_ret.append(yy.SOXX_r5.mean())
            matched_hit.append((bb.SOXX_r5*direction>0).mean()); matched_ret.append(bb.SOXX_r5.mean())
            near_base.append(bb[f'{kind}_near8'].mean())
        for name, vals in [('matched_hit5',matched_hit),('matched_mean5',matched_ret),
                           ('year_all_hit5',all_hit),('year_all_mean5',all_ret),('matched_near8',near_base)]:
            result[name]=float(np.mean(vals)) if vals else None
        for pct in [8,12]:
            values=group[f'{kind}_near{pct}'].dropna()
            near=group[group[f'{kind}_near{pct}']==1]
            result[f'near{pct}_n']=len(values)
            result[f'near{pct}_hits']=int(values.sum())
            result[f'near{pct}_rate']=float(values.mean()) if len(values) else None
            result[f'near{pct}_early']=int((near[f'{kind}_distance{pct}']<=0).sum())
            result[f'near{pct}_late']=int((near[f'{kind}_distance{pct}']>0).sum())
        result['horizons']={str(h):dict(n=int(group[f'SOXX_r{h}'].notna().sum()),
            hit=float((group[f'SOXX_r{h}'].dropna()*direction>0).mean()),
            mean=float(group[f'SOXX_r{h}'].mean())) for h in HORIZONS}
        result['controls']={a:dict(n=int(group[f'{a}_r5'].notna().sum()),
            hit=float((group[f'{a}_r5'].dropna()*direction>0).mean()),mean=float(group[f'{a}_r5'].mean()))
            for a in ['QQQ','SPY']}
        return result

    result={'data':{'start':'2024-01-02','end':'2026-10-23',
        'sessions':len(base),'sessions_by_year':{str(y):len(g) for y,g in base.groupby('year')},
        'missing_tnx_sessions':int(cl['^TNX'].isna().sum()),'raw_news':len(news),
        'nonduplicate_news':int(news.include.sum()),'clustered_kept_news':int(news.keep.sum()),
        'events_sha256':freeze['files']['events.json']},'families':{},'by_year':{},'macro':{},'coverage':{}}
    for fam,g in news[news.keep].groupby('family'):
        result['families'][fam]=metrics(g,-1 if fam=='T' else 1)
    for year,g in news[news.keep].groupby('year'):
        result['by_year'][str(year)]={fam:metrics(gg,-1 if fam=='T' else 1) for fam,gg in g.groupby('family')}
    for fam,g in macro.groupby('family'):
        result['macro'][fam]={'all':metrics(g,1),
            'by_year':{str(y):metrics(gg,1) for y,gg in g.groupby('year')}}
    for pct,p in piv.items():
        for year in [2024,2025,2026]:
            pp=p[p.date.str.startswith(str(year))]
            for kind in ['bottom','top']:
                pp_kind=pp[pp.kind==kind]
                g=news[news.keep & ((news.family=='T') if kind=='top' else (news.family!='T'))]
                covered=sum(bool((abs(g.i-int(row.i))<=5).any()) for _,row in pp_kind.iterrows())
                before=sum(bool(((g.i<=int(row.i))&(g.i>=int(row.i)-5)).any()) for _,row in pp_kind.iterrows())
                result['coverage'][f'{year}_{pct}_{kind}']=dict(total=len(pp_kind),covered=covered,before_or_same=before)
    # Descriptive subset of already-registered families, not a newly selected rule.
    result['descriptive_D_S_with_M']=metrics(news[news.keep & news.family.isin(['D','S']) & news.M_at_entry],1)
    result['raw_without_clustering']={fam:metrics(g,-1 if fam=='T' else 1) for fam,g in news[news.include].groupby('family')}
    # Emit every pivot, including uncovered ones, rather than only good stories.
    details=[]
    for _,p in piv[8].iterrows():
        if p.date<'2024-01-01': continue
        g=news[news.keep & ((news.family=='T') if p.kind=='top' else (news.family!='T'))]
        near=g[abs(g.i-int(p.i))<=5]
        details.append(dict(date=p.date,kind=p.kind,close=p.close,confirmed_date=p.confirmed_date,
             news_ids=','.join(near.id),early_ids=','.join(near[near.i<=p.i].id)))
    pd.DataFrame(details).to_csv(data/'pivot_coverage.csv',index=False,encoding='utf-8-sig')
    base.to_csv(data/'daily_baseline.csv',encoding='utf-8-sig')
    (data/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['data','families','coverage']},ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    run(parser.parse_args().data_dir)
