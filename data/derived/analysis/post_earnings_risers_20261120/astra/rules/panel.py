"""Point-in-time inputs for rank-pair rules. Dates use the study's QQQ session index."""
from pathlib import Path
import csv
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
STUDY = OUT.parents[1]
KEY = ['ticker', 'reaction_date']
BASE = [
    'px_pre20_x', 'px_pre60_x', 'px_vol60', 'px_vsurge5', 'px_dvol_m',
    'px_clv0', 'px_o2c0', 'px_vol0', 'px_r0_sig', 'occ_call_ab',
    'occ_cust_callshare_ch', 'kr_buy_share_pre', 'a_h01_wiki', 'a_h08_dtc',
    'a_h09_si_ch', 'a_h18_vcomp', 'overnight20', 'intraday20', 'efficiency20',
    'beta60', 'qqq_ret20',
]
LATE = ['d5_x', 'breadth5', 'strength5', 'volfade5']
GATES = ['px_vol60', 'px_pre20_x', 'qqq_ret20', 'px_vol0', 'a_h01_wiki', 'a_h08_dtc']
CUTOFF = '2026-10-21'
CLOSE = '2026-11-19'

def load_panel(train_only=False):
    with (STUDY/'results/events.csv').open(encoding='utf-8-sig',newline='') as f:
        records = [
            {k:row[k] for k in [*KEY,'season','k','timing','d20_x','d5_x']}
            for row in csv.DictReader(f)
            if not train_only or row['season'] in ('2026Q1','2026Q2')
        ]
    frame = pd.DataFrame(records)
    assert not frame.duplicated(KEY).any()
    for k in ['k','d5_x','d20_x']:
        frame[k] = pd.to_numeric(frame[k],errors='raise')
    needed = set(KEY+BASE)
    for f in ['features','features_extra']:
        path = STUDY/'results'/f'{f}.csv'
        subset = pd.read_csv(path,usecols=lambda k:k in needed)
        frame = frame.merge(subset,how='left',on=KEY,validate='one_to_one',indicator=True)
        assert frame.pop('_merge').eq('both').all()
    cache=STUDY/'cache'
    raw={k:pd.read_parquet(cache/f'px_{k}.parquet') for k in ['open','close','adjclose','high','low','volume']}
    idx=raw['close']['QQQ'].dropna().index
    idx=idx[idx<=pd.Timestamp(CLOSE)]
    for k in raw:
        raw[k]=raw[k].reindex(idx)
    adj=raw['adjclose']
    factor=adj/raw['close']
    op,hi,lo=[raw[k]*factor for k in ['open','high','low']]
    volume=raw['volume']
    market='QQQ'
    def excess(tk,i,j):
        return 100*(adj[tk].iat[j]/op[tk].iat[i]-adj[market].iat[j]/op[market].iat[i]) if j<len(idx) else np.nan
    rows=[]
    for e in frame.itertuples():
        k,t=int(e.k),e.ticker
        assert str(idx[k].date())==e.reaction_date
        close=adj[t].to_numpy()
        q=adj[market].to_numpy()
        prev20=slice(k-21,k-1)
        last20=slice(k-20,k)
        with np.errstate(divide='ignore',invalid='ignore'):
            overnight=np.log(op[t].to_numpy()[last20]/close[prev20])-np.log(op[market].to_numpy()[last20]/q[prev20])
            intraday=np.log(close[last20]/op[t].to_numpy()[last20])-np.log(q[last20]/op[market].to_numpy()[last20])
            lr=np.diff(np.log(close[k-61:k]))
            qr=np.diff(np.log(q[k-61:k]))
            spread=np.abs(lr[-20:]).sum()
            beta=np.cov(lr,qr,ddof=1)[0,1]/np.var(qr,ddof=1)
            item=dict(overnight20=100*overnight.sum(),intraday20=100*intraday.sum(),
                      efficiency20=abs(lr[-20:].sum())/spread if spread>0 else np.nan,
                      beta60=beta,qqq_ret20=100*(q[k-1]/q[k-21]-1),E6=excess(t,k+6,k+25),
                      E6_ready=k+5<len(idx),E1_ready=True,
                      E6_entered=k+6<len(idx) and np.isfinite(op[t].iat[k+6]),
                      E1_entered=k+1<len(idx) and np.isfinite(op[t].iat[k+1]))
            # Refreshed adjusted bars may differ below 0.0001 percentage points.
            assert np.isclose(excess(t,k+1,k+20),e.d20_x,rtol=0,atol=.0001,equal_nan=True)
            assert np.isclose(excess(t,k+1,k+5),e.d5_x,rtol=0,atol=.0001,equal_nan=True)
            if k+5<len(idx):
                first=slice(k+1,k+6)
                daily=close[first]/op[t].to_numpy()[first]-q[first]/op[market].to_numpy()[first]
                top=np.max(hi[t].to_numpy()[first])
                bottom=np.min(lo[t].to_numpy()[first])
                item.update(breadth5=np.sum(daily>0) if np.isfinite(daily).all() else np.nan,
                            strength5=(close[k+5]-bottom)/(top-bottom) if top>bottom else np.nan,
                            volfade5=np.log(np.median(volume[t].to_numpy()[first])/volume[t].iat[k]))
        rows.append(item)
    return pd.concat([frame,pd.DataFrame(rows)],axis=1).rename(columns={'d20_x':'E1'}).replace([np.inf,-np.inf],np.nan)

