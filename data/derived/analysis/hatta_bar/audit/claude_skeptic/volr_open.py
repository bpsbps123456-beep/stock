import numpy as np, pandas as pd
X = pd.read_parquet("cache/cand_yf1m/A_120_0.0005.parquet")
ts = pd.to_datetime(X.ts*60, unit="s", utc=True).dt.tz_convert("America/New_York")
m = ts.dt.hour*60+ts.dt.minute
op = X[(m>=570)&(m<590)]
print("rows 09:30-09:49:", len(op), "volr inf:", int(np.isinf(op.volr).sum()), "volr>=2:", int((op.volr>=2).sum()), "volr>=1e3:", int((op.volr>=1e3).sum()))
d = pd.read_parquet("cache/yf1m/NVDA.parquet"); mm = d.index.hour*60+d.index.minute
print("NVDA premarket bars with volume>0:", int(((mm<570)&(d.Volume>0)).sum()), "of", int((mm<570).sum()))
for mn in (570, 571, 575, 585, 590):
    s = X[m==mn]
    print(mn, "rows", len(s), "inf", int(np.isinf(s.volr).sum()), "zero", int((s.volr==0).sum()), ">=1e3", int((s.volr>=1e3).sum()))
d = pd.read_parquet("cache/yf1m/NVDA.parquet"); mm = d.index.hour*60+d.index.minute
print("NVDA bars at 09:30:", int((mm==570).sum()), "09:29:", int((mm==569).sum()), "vol>0 at 09:30:", int(((mm==570)&(d.Volume>0)).sum()))
