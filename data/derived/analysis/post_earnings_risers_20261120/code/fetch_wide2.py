"""넓은 표본 가격(PREREG3): 넓은 연구 사건표 종목 + 펀드 종목 + 기준 ETF·VIX, 2025-01-02~2026-11-19 일봉.
결과(Git 제외): cache/w_{open,high,low,close,adjclose,volume}.parquet
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parents[1]
CACHE = BASE / "cache"
BENCH = ["QQQ", "SPY", "SOXX", "XLY", "XLF", "XLI", "XLK", "XLV", "XLRE", "XLU", "XLE", "XLP", "XLB", "XLC", "^VIX"]


def main() -> None:
    ev = pd.read_csv(BASE.parent / "earnings_trend/results/events_wide.csv", usecols=["ticker", "reaction_date"])
    tk = set(ev[ev.reaction_date >= "2025-12-01"].ticker) | set(pd.read_csv(BASE / "results/universe.csv").ticker)
    tk = sorted(tk) + BENCH
    parts = {c: [] for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume")}
    for i in range(0, len(tk), 200):
        raw = yf.download(tk[i:i + 200], start="2025-01-01", end="2026-11-20", auto_adjust=False, actions=False,
                          group_by="column", threads=True, progress=False)
        for c in parts:
            parts[c].append(raw[c])
        print(i, raw["Close"].shape, flush=True)
    for c, nm in (("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"), ("Adj Close", "adjclose"), ("Volume", "volume")):
        df = pd.concat(parts[c], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_parquet(CACHE / f"w_{nm}.parquet")
    c = pd.read_parquet(CACHE / "w_close.parquet")
    print("done", c.shape, c.index.min().date(), c.index.max().date(), "last-row NaN", int(c.iloc[-1].isna().sum()))


if __name__ == "__main__":
    main()
