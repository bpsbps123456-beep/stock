"""N3 지그재그용 긴 이력: SOXX·^SOX, 1994-01-01~2026-10-23 일봉.
Codex(news_turning_points_20261025)와 같이 auto_adjust=False(분할 반영·배당 미포함).
-> cache/n3_raw.parquet (열: sym, date, O, H, L, C)"""
import pathlib
import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).resolve().parent.parent
SYMS = ["SOXX", "^SOX"]
d = yf.download(SYMS, start="1994-01-01", end="2026-10-24", interval="1d", auto_adjust=False,
                progress=False, group_by="ticker", threads=True, actions=False)
rows = []
for s in SYMS:
    x = d[s][["Open", "High", "Low", "Close"]].dropna(subset=["Close"]).copy()
    x.columns = ["O", "H", "L", "C"]
    x["sym"] = s
    x["date"] = pd.DatetimeIndex(x.index).tz_localize(None).normalize()
    rows.append(x.reset_index(drop=True))
out = pd.concat(rows, ignore_index=True)
(HERE / "cache").mkdir(exist_ok=True)
out.to_parquet(HERE / "cache" / "n3_raw.parquet")
print(len(out), "행")
print(out.groupby("sym").date.agg(["min", "max", "count"]).to_string())
