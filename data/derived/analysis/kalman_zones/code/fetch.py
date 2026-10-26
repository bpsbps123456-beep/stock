import yfinance as yf, pandas as pd, os
SYMS = ["AMD","ARM","DELL","INTC","SNDK","STX","WDC"]
out = {}
for iv, per in [("5m","60d"),("1h","730d")]:
    df = yf.download(SYMS, interval=iv, period=per, progress=False, auto_adjust=False, prepost=True, group_by="ticker", threads=True)
    for s in SYMS:
        d = df[s][["Open","High","Low","Close","Volume"]].dropna(subset=["Open","High","Low","Close"]).copy()
        d.index = d.index.tz_convert("America/New_York")
        out[(s, iv)] = d
        t = d.index
        print(s, iv, len(d), t[0].date(), "->", t[-1].date(), "sessions", len(set(t.date)))
os.makedirs("data", exist_ok=True)
for (s, iv), d in out.items(): d.to_parquet(f"data/{s}_{iv}.parquet")
# 1h 봉 시각 구성 확인
d = out[("AMD","1h")]
print(sorted(set(d.index.strftime("%H:%M"))))
d5 = out[("AMD","5m")]
per_day = d5.groupby(d5.index.date).size()
print("5m bars/day min/median/max", per_day.min(), per_day.median(), per_day.max())
