import pandas as pd, numpy as np, json
d = pd.read_parquet("cache/yf1m/NVDA.parquet")
print(d.index.dtype, d.columns.tolist(), d.index[0], d.index[-1], len(d))
m = d.index.hour*60+d.index.minute
pre = d[(m<570)]; reg = d[(m>=570)&(m<960)]; post = d[m>=960]
print("pre bars", len(pre), "vol0 share", (pre.Volume==0).mean().round(3), "reg vol0", (reg.Volume==0).mean().round(3), "post vol0", (post.Volume==0).mean().round(3))
print("open/close outside HL:", ((d.Open>d.High)|(d.Open<d.Low)|(d.Close>d.High)|(d.Close<d.Low)).sum())
print("NaN rows:", d.isna().any(axis=1).sum())
meta = json.load(open("cache/cand_yf1m/meta.json"))
print(meta["src"], meta["from"], meta["to"], len(meta["syms"]))
x = pd.read_parquet("cache/cand_yf1m/A_120_0.0005.parquet")
print(x.columns.tolist()); print(len(x), "tradable", int(x.tradable.sum()), "sane", int(x.sane.sum()), "trad&sane", int((x.tradable&x.sane).sum()))
print(x.groupby("sess").size())
