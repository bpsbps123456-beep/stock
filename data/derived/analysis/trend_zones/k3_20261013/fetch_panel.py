import yfinance as yf, pandas as pd
src = pd.read_parquet(r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/rsi_breakout/cache/1d_long.parquet")
syms = sorted(set(src.sym) - {"QQQ"})
etfs = ["EWY","KORU","SOXX","SOXL","QQQ","SPY","SMH","IWM","EWJ","EWT","EEM","TQQQ"]
allsyms = syms + [e for e in etfs if e not in syms]
df = yf.download(allsyms, start="2023-01-01", end="2026-10-13", interval="1d", auto_adjust=False, progress=False, group_by="ticker", threads=True)
rows = []
for s in allsyms:
    try:
        d = df[s][["Open","High","Low","Close","Volume"]].dropna(subset=["Close"]).copy()
    except KeyError:
        print("missing", s); continue
    d["sym"] = s; d["ts"] = d.index; rows.append(d.reset_index(drop=True))
out = pd.concat(rows)
out["is_etf"] = out.sym.isin(etfs)
out.to_parquet("panel_1d_20261012.parquet")
print(out.sym.nunique(), out.ts.min(), out.ts.max())
# compare with cached closes for overlap
m = out.merge(src, on=["sym","ts"], suffixes=("","_c"))
diff = (m.Close/m.Close_c-1).abs()
print("overlap rows", len(m), "max rel diff", diff.max(), "share >0.1%", (diff>1e-3).mean())
print(m.loc[diff>1e-3, ["sym","ts","Close","Close_c"]].groupby("sym").size().sort_values().tail(10))
