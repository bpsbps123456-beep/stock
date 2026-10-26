"""연구용 일봉(yfinance). Close는 분할 조정, Adj Close는 분할+배당 조정이다.

보관금액 ÷ Close로 보유 좌수를 잴 때는 Close(분할 조정, 배당 미조정)를 쓴다.
수익률은 Open·Adj Close가 아니라 같은 기준(분할 조정 Open·Close)으로 잰다. 배당 차이는 작아 무시한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ["SOXX", "SMH", "^SOX", "SOXL", "SOXS", "QQQ", "TQQQ", "SQQQ", "SPY", "KORU", "EWY", "^KS11",
           "NVDA", "TSLA", "TSLL", "NVDL", "USD", "^VIX"]


def main() -> None:
    frames = []
    for t in TICKERS:
        df = yf.download(t, start="2022-06-01", end="2026-10-25", auto_adjust=False, progress=False,
                         actions=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={"Date": "date", "Adj Close": "AdjClose"})
        df["ticker"] = t
        frames.append(df[["date", "ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"]])
        print(t, len(df), df["date"].min().date(), df["date"].max().date())
    out = pd.concat(frames)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    (ROOT / "raw").mkdir(exist_ok=True)
    out.to_csv(ROOT / "raw" / "prices.csv", index=False)
    meta = {"source": "yfinance", "auto_adjust": False, "tickers": TICKERS,
            "rows": {t: int((out.ticker == t).sum()) for t in TICKERS}}
    (ROOT / "raw" / "prices_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
