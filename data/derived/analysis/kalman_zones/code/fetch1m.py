"""yfinance 1분봉(프리·애프터 포함, 최근 30일 한도)을 8일씩 나눠 받아 종목별로 이어 붙인다. 3분봉 이격 매수 검증용."""
import os, sys, datetime as dt
import pandas as pd, yfinance as yf
SYMS = ["AMD", "ARM", "DELL", "INTC", "SNDK", "STX", "WDC"]
OUT = sys.argv[1]
end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
start = end - pd.Timedelta(days=29)
for s in SYMS:
    parts = []
    a = start
    while a < end:
        b = min(a + pd.Timedelta(days=7), end)
        d = yf.download(s, interval="1m", start=a.strftime("%Y-%m-%d"), end=b.strftime("%Y-%m-%d"),
                        prepost=True, progress=False, auto_adjust=False)
        if len(d):
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            parts.append(d[["Open", "High", "Low", "Close", "Volume"]])
        a = b
    d = pd.concat(parts).sort_index()
    d = d[~d.index.duplicated()].dropna(subset=["Open", "High", "Low", "Close"])
    d.index = d.index.tz_convert("America/New_York")
    d.to_parquet(os.path.join(OUT, f"{s}_1m.parquet"))
    print(s, len(d), d.index[0], "->", d.index[-1], "days", len(set(d.index.date)))
