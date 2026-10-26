"""추세 5단계(일봉) 비교용 일봉: 144종목 + SOXX, 2025-10-01~ (5단계 예열 130봉 이상). → cache/<SYM>_1d.parquet"""
import pathlib
import pandas as pd
import yfinance as yf

CACHE = pathlib.Path(__file__).resolve().parent.parent / "cache"
SYMS = sorted(p.stem[:-3] for p in CACHE.glob("*_5m.parquet")) + ["SOXX"]
SYMS = sorted(set(SYMS))
d = yf.download(SYMS, interval="1d", start="2025-10-01", end="2026-10-24", auto_adjust=False, progress=False, group_by="ticker", threads=True)
n = 0
for s in SYMS:
    x = d[s][["Open", "High", "Low", "Close", "Adj Close", "Volume"]].dropna(subset=["Close"])
    if len(x):
        x.to_parquet(CACHE / f"{s}_1d.parquet"); n += 1
print("일봉", n, "종목", d.index[0].date(), "->", d.index[-1].date())
