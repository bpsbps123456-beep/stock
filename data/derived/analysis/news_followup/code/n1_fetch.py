"""N1 긴 이력 가격: ^SOX·SOXX·QQQ·SPY·^TNX, 1994-01-01~2026-10-23 일봉.
Codex(news_turning_points_20261025)와 같이 auto_adjust=False(분할 반영·배당 미포함)를 주 자료로,
auto_adjust=True(배당 반영)를 민감도용으로 함께 저장한다.
-> cache/n1_raw.parquet, cache/n1_adj.parquet (열: sym, date, O, H, L, C)"""
import pathlib
import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).resolve().parent.parent
SYMS = ["^SOX", "SOXX", "QQQ", "SPY", "^TNX"]
for adj, name in [(False, "n1_raw"), (True, "n1_adj")]:
    d = yf.download(SYMS, start="1994-01-01", end="2026-10-24", interval="1d", auto_adjust=adj,
                    progress=False, group_by="ticker", threads=True, actions=False)
    rows = []
    for s in SYMS:
        x = d[s][["Open", "High", "Low", "Close"]].dropna(subset=["Close"]).copy()
        x.columns = ["O", "H", "L", "C"]
        x["sym"] = s
        x["date"] = pd.DatetimeIndex(x.index).tz_localize(None).normalize()
        rows.append(x.reset_index(drop=True))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(HERE / "cache" / f"{name}.parquet")
    print(name, len(out), "행")
    print(out.groupby("sym").date.agg(["min", "max", "count"]).to_string())
