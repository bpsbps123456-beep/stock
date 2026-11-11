"""Publication-aware fixed 19-signal panel. No forward outcomes here."""
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent/'seibro_bottoms'


def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


focus = load('zone_focus', PRIOR/'code/soxs_focus.py')
b = sys.modules['soxs_focus_build']
FAMILIES = dict(S=['S1H','S1L','S5H','S5L'], X=['X1','X2','X3'],
                T=['T5H','T5L','CH','CL'], G=['G25'], M=['M50','M100','M200'], N=['ND','NR','NS','NT'])
SIGNALS = sum(FAMILIES.values(), [])


def dump(path, value):
    def clean(v):
        if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v, (list,tuple)): return [clean(x) for x in v]
        if isinstance(v, (np.bool_,bool)): return bool(v)
        if isinstance(v, (np.integer,int)): return int(v)
        if isinstance(v, (float,np.floating)): return float(v) if np.isfinite(v) else None
        if isinstance(v, (pd.Timestamp,Path)): return str(v)
        return v
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')


def fingerprint(paths):
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def records():
    c = load('zone_collect', ROOT.parent/'seibro_flows/code/collect.py')
    out = []
    for p in sorted((ROOT/'data/seibro').glob('*/*.xml.gz')):
        kind,dtype,day = p.name.split('.')[0].split('_')
        cache = ROOT/'data/cache'/p.name.replace('.xml.gz','.json')
        if cache.exists():
            rec = json.loads(cache.read_text(encoding='utf-8'))
        else:
            rows = c.parse(gzip.decompress(p.read_bytes()).decode('utf-8'), kind)
            rec = dict(start=str(pd.Timestamp(day).date()), kind=kind, dtype=None if dtype=='amt' else dtype, n=len(rows), rows=rows)
        out.append(rec)
    return out


def raw_panel():
    recs = records()
    names = {r['isin']:r['name'] for x in recs for r in x['rows']}
    seen = pd.DataFrame(dict(isin=list(names),first='',last='',n_dates=0,lists='',tot_sum=0))
    cls = b.load_classes(pd.Series(names), seen, {})
    products = dict(zip(cls['isin'],cls['product']))
    nyse,ksd = b.nyse_sessions(),b.ksd_sessions()
    _,settle = b.cm.calendars()
    source_days = nyse[(nyse >= '2023-01-03') & (nyse <= '2026-11-11')]
    mapping = {t:pd.Timestamp(b.cm.seibro_of(t.date(),settle)) for t in source_days}
    counts = pd.Series(mapping).value_counts()
    drec = {(pd.Timestamp(x['start']),str(x['dtype'])):x for x in recs if x['kind']=='daily'}
    rows = []
    for t,d in mapping.items():
        lists = [drec.get((d,v)) for v in '1234']
        union = {}
        complete = all(x is not None and x['n']>0 for x in lists)
        for x in lists:
            if x:
                for row in x['rows']:
                    if row['isin'] != b.DUMMY:
                        if row['isin'] in union:
                            for col in ['buy','sell','net','tot']:
                                if abs(union[row['isin']][col]-row[col])>1: raise ValueError('Inconsistent rank-list money')
                        union[row['isin']] = row
        n = counts[d]
        out = dict(date=t, settle_date=d, known_date=b.next_after(ksd,d),
                   entry=b.on_or_after(nyse,b.next_after(ksd,d)), unit_n=int(n), published=complete)
        for product in ['SOXL','SOXS']:
            vals = [r for isin,r in union.items() if products.get(isin)==product]
            for col in ['buy','sell','net']:
                out[product+'_'+col] = sum(r[col] for r in vals)/n if vals and complete else np.nan
        out['total_buy'] = sum(r['buy'] for r in lists[0]['rows'] if r['isin']!=b.DUMMY)/n if complete else np.nan
        out['total_net'] = sum(r['net'] for r in union.values())/n if complete else np.nan
        rows.append(out)
    src = pd.DataFrame(rows).set_index('date')
    # Latest expected source is selected before checking whether published.
    # Exclude dates whose publication convention is still in the future later.
    src.to_csv(ROOT/'data/source_flows.csv', index_label='date')
    frozen=pd.read_parquet(PRIOR/'derived/measures_daily.parquet').set_index('date')
    common=src[src.published].index.intersection(frozen.index)
    audit={col:float((src.loc[common,col]-frozen.loc[common,col]).abs().max())
           for col in ['SOXL_buy','SOXS_buy','SOXL_net','SOXS_net']}
    dump(ROOT/'data/frozen_overlap_audit.json',dict(common_rows=len(common),max_error_usd=audit,
         note='Fresh inputs only; historical frozen files read but never overwritten'))
    cu = []
    px = pd.read_csv(ROOT/'data/prices.csv', parse_dates=['date'])
    xl = px[px.ticker=='SOXL'].set_index('date')['close']
    for x in recs:
        if x['kind']!='custody': continue
        d = pd.Timestamp(x['start'])
        rows = [r for r in x['rows'] if r['isin']!=b.DUMMY]
        amount = [r['amt'] for r in rows if products.get(r['isin'])=='SOXL']
        raw_trade = max((t for t,s in mapping.items() if s<=d), default=pd.NaT)
        known = b.next_after(ksd,b.next_after(ksd,d))
        price = xl.loc[:d].iloc[-1] if len(xl.loc[:d]) else np.nan
        cu.append(dict(custody_date=d, source_date=raw_trade, entry=b.on_or_after(nyse,known), known_date=known,
                       amount=sum(amount) if amount else np.nan,
                       total=sum(r['amt'] for r in rows) if rows else np.nan,
                       shares=sum(amount)/price if amount else np.nan))
    cust = pd.DataFrame(cu).sort_values('custody_date')
    cust.to_csv(ROOT/'data/custody.csv', index=False)
    dump(ROOT/'data/input_qa.json',dict(requests=len(recs), empty=sum(x['n']==0 for x in recs),
        first=min(x['start'] for x in recs),last=max(x['start'] for x in recs),
        source_complete=int(src.published.sum()), missing_SOXS=src[src.published & src.SOXS_buy.isna()].index.astype(str).tolist(),
        shared_units=src[src.unit_n>1].reset_index()[['date','settle_date','unit_n']].astype(str).to_dict('records')))
    return src,cust


def price_features(px):
    c = px['close']
    f = pd.DataFrame(index=px.index)
    f['dd60'] = c/c.rolling(60,min_periods=60).max()-1
    f['ret5'] = c/c.shift(5)-1
    f['ret1'] = c/c.shift(1)-1
    f['low_ratio'] = np.nan
    f['peak_pos'] = np.nan
    for i in range(59,len(c)):
        w = c.iloc[i-59:i+1].to_numpy()
        peak = i-int(np.argmax(w[::-1]))
        f.iloc[i,f.columns.get_loc('peak_pos')] = peak
        f.iloc[i,f.columns.get_loc('low_ratio')] = c.iloc[i]/c.iloc[peak:i+1].min()-1
    for n in [50,100,200]: f['ma'+str(n)] = c.rolling(n,min_periods=n).mean()
    return f


def flag(v, test):
    return pd.Series(np.where(v.notna(),test.astype(float),np.nan),index=v.index)


def source_signals(src,px):
    s = src.reindex(px.index).copy()
    f = price_features(px)
    v = pd.DataFrame(index=px.index)
    for n in [1,5]:
        xs = s.SOXS_buy.rolling(n,min_periods=n).sum()
        xl = s.SOXL_buy.rolling(n,min_periods=n).sum()
        key = 'S'+str(n)
        v[key] = xs/(xs+xl).where(xs+xl>0)
    v['T5'] = s.SOXL_buy.rolling(5,min_periods=5).sum()/s.total_buy.rolling(5,min_periods=5).sum().replace(0,np.nan)
    v['buy5'] = s.SOXL_buy.rolling(5,min_periods=5).sum()
    v['X2_value'] = s.total_net.rolling(5,min_periods=5).sum()
    v['X1_value'] = np.nan
    for i in range(59,len(v)):
        peak = int(f.peak_pos.iloc[i])
        hi = v.buy5.iloc[peak:i+1].max()
        if hi>0: v.iloc[i,v.columns.get_loc('X1_value')] = v.buy5.iloc[i]/hi
    for key in ['S1','S5','T5']:
        v[key+'_P'],v[key+'_n'] = b.pct_rank(v[key].to_numpy(),250,120)
    for key in ['S1','S5','T5']:
        v[key+'H'] = flag(v[key+'_P'],v[key+'_P']>=80)
        v[key+'L'] = flag(v[key+'_P'],v[key+'_P']<=20)
    near = f.low_ratio<=.03
    v['X1'] = flag(v.X1_value, (v.X1_value<=.5)&near)
    v['X1_30'] = flag(v.X1_value, (v.X1_value<=.3)&near)
    v['X1_70'] = flag(v.X1_value, (v.X1_value<=.7)&near)
    v['X2'] = flag(v.X2_value,v.X2_value<0)
    return v


def assemble(src,cust,px,cnn,notes):
    v = source_signals(src,px)
    f = price_features(px)
    # Include the next session's observable setup; its open/outcome stay missing.
    entry_index = px.index.append(pd.DatetimeIndex([b.next_after(b.nyse_sessions(),px.index[-1])]))
    panel = pd.DataFrame(index=entry_index)
    panel['open'] = px.open
    panel['close'] = px.close
    panel[['dd60','ret5','ret1']] = f[['dd60','ret5','ret1']].reindex(entry_index).shift(1)
    panel['down'] = panel.dd60<=-.1
    panel['high'] = panel.dd60>=-.03
    selected = src.reset_index().rename(columns={'index':'date'}).sort_values(['entry','date']).drop_duplicates('entry',keep='last').set_index('entry')
    panel['source_date'] = selected.date.reindex(panel.index)
    panel['source_known'] = selected.known_date.reindex(panel.index)
    panel['source_published'] = selected.published.reindex(panel.index)
    # Missing source never silently becomes OFF.
    for col in v:
        panel[col] = panel.source_date.map(v[col]).to_numpy()
        panel.loc[panel.source_published.ne(True),col] = np.nan
    cp = cust.sort_values(['entry','custody_date']).drop_duplicates('entry',keep='last').set_index('entry')
    aligned = cp.reindex(panel.index.union(cp.index)).sort_index().ffill().reindex(panel.index)
    # ffill columnwise would skip newly missing observations; select row indices instead.
    ix = cp.index.searchsorted(panel.index,side='right')-1
    aligned = cp.iloc[np.maximum(ix,0)].copy()
    aligned.index = panel.index
    aligned.loc[ix<0,:] = np.nan
    panel['custody_date'] = aligned.custody_date
    panel['custody_source_date'] = aligned.source_date
    panel['custody_known'] = aligned.known_date
    panel['custody_age'] = (panel.index.to_series()-pd.to_datetime(aligned.custody_date)).dt.days
    panel['C'] = aligned.amount/aligned.total.replace(0,np.nan)
    panel['C_P'],panel['C_n'] = b.pct_rank(panel.C.to_numpy(),250,120)
    panel['CH'] = flag(panel.C_P,panel.C_P>=80)
    panel['CL'] = flag(panel.C_P,panel.C_P<=20)
    panel['X3_value'] = aligned.shares-aligned.shares.shift(5)
    low = aligned.source_date.map(f.low_ratio)
    panel['X3'] = flag(panel.X3_value.where(low.notna()),(panel.X3_value<=0)&(low<=.03))
    panel['G'] = cnn.reindex(panel.index).shift(1)
    panel['G25'] = flag(panel.G,panel.G<=25)
    panel['G75'] = flag(panel.G,panel.G>=75)
    for n in [50,100,200]:
        val = px.close.reindex(entry_index).shift(1)-f['ma'+str(n)].reindex(entry_index).shift(1)
        panel['M'+str(n)+'_value'] = val
        panel['M'+str(n)] = flag(val,val<0)
    last = pd.Timestamp(notes.date.max())
    for fam in 'DRST':
        hits = pd.Series(0.,index=panel.index)
        for d in pd.to_datetime(notes.loc[notes.family==fam,'date']).unique():
            k = panel.index.searchsorted(d,side='right')
            hits.iloc[k:k+5] = 1
        hits.loc[panel.index>last] = np.nan
        panel['N'+fam] = hits
    flowkeys=['S1H','S1L','S5H','S5L','X1','X2','T5H','T5L']
    for key in SIGNALS:
        panel[key+'_status']=np.where(panel[key].isna(),'UNKNOWN',np.where(panel[key].eq(1),'ON','OFF'))
        if key in flowkeys:
            panel.loc[panel.source_published.eq(False),key+'_status']='UNPUBLISHED'
            panel.loc[panel.source_date.isna(),key+'_status']='NO_NEW_SOURCE'
    return panel


def inputs(news_series=None):
    src = pd.read_csv(ROOT/'data/source_flows.csv',parse_dates=['date','settle_date','known_date','entry']).set_index('date')
    cust = pd.read_csv(ROOT/'data/custody.csv',parse_dates=['custody_date','source_date','entry','known_date'])
    prices = pd.read_csv(ROOT/'data/prices.csv',parse_dates=['date'])
    px = prices[prices.ticker=='SOXX'].set_index('date').sort_index()
    sessions=b.nyse_sessions()
    expected=sessions[(sessions>=px.index[0])&(sessions<=px.index[-1])]
    if not expected.equals(px.index):raise ValueError('Missing or duplicated SOXX sessions')
    # Single fresh response: invariant reference dates test split-adjustment consistency.
    anchors = {'2026-07-29':465/3,'2026-10-05':589.51/3,'2026-11-05':131.90}
    err = {k:float(abs(px.loc[k,'close']-v)) for k,v in anchors.items()}
    if max(err.values())>.02: raise ValueError(f'SOXX split basis mismatch {err}')
    if (px.close.pct_change(fill_method=None).abs()>.45).any(): raise ValueError('Unexpected split discontinuity')
    dump(ROOT/'data/split_check.json',dict(anchors=anchors,abs_error=err,passed=True,
        method='Single fresh Yahoo response; anchors already at 3:1 basis; no redivision or splice'))
    hist = json.loads((ROOT/'data/cnn.json').read_text())['fear_and_greed_historical']['data']
    # CNN daily historical x is the UTC-midnight DATE LABEL, not a US-close instant.
    cnn = pd.Series({pd.Timestamp(x['x'],unit='ms',tz='UTC').tz_localize(None).normalize():x['y'] for x in hist})
    path = Path(news_series) if news_series else ROOT.parents[3]/'research/pb_flow_census_20261026/census.csv'
    notes = pd.read_csv(path)
    if 'date' not in notes: notes['date'] = notes.published_at.str[:10]
    notes = notes[['date','family']].drop_duplicates()
    return src,cust,px,cnn,notes


if __name__=='__main__':
    raw_panel()
    args = inputs()
    out = assemble(*args)
    out.to_csv(ROOT/'data/signal_panel.csv',index_label='date')
    print('signals',len(out),out.index.max())
