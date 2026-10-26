"""중단기 추세 연구 일봉: candle_ratio 144종목 + SPY·QQQ·SOXX, 2022-06-01~2026-10-23, 수정주가(auto_adjust) OHLC.
→ cache/daily.parquet (열: sym, date, O, H, L, C, V)"""
import pathlib
import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).resolve().parent.parent
CR = HERE.parent / "candle_ratio" / "cache"
SYMS = sorted(set(p.stem[:-3] for p in CR.glob("*_5m.parquet")) | {"SPY", "QQQ", "SOXX"})
d = yf.download(SYMS, start="2022-06-01", end="2026-10-23", interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True)
rows = []
for s in SYMS:
    x = d[s][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
    x.columns = ["O", "H", "L", "C", "V"]
    x["sym"] = s
    x["date"] = pd.DatetimeIndex(x.index).date
    rows.append(x.reset_index(drop=True))
out = pd.concat(rows, ignore_index=True)
out.to_parquet(HERE / "cache" / "daily.parquet")
print(len(SYMS), "종목", len(out), "행", out.date.min(), "->", out.date.max())
print(out.groupby("sym").date.min().sort_values().tail(12).to_string())
