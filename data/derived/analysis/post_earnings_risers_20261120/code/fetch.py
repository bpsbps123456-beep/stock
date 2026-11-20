"""가격(일봉 OHLCV, 수정·비수정)과 실적 발표 일시를 yfinance에서 받는다.

결과(Git 제외 캐시): cache/px_{open,high,low,close,adjclose,volume}.parquet, cache/earn_dates.csv, cache/info.csv
"""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parents[1]
CACHE = BASE / "cache"
BENCH = ["QQQ", "SPY", "SOXX", "IGV", "XLK", "XLU", "XLI", "XLE", "URA", "IWM", "ARKK"]
START, END = "2025-01-01", "2026-11-20"


def prices(tickers: list[str]) -> None:
    raw = yf.download(tickers, start=START, end=END, auto_adjust=False, actions=False,
                      group_by="column", threads=True, progress=False)
    for col, name in [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"),
                      ("Adj Close", "adjclose"), ("Volume", "volume")]:
        raw[col].to_parquet(CACHE / f"px_{name}.parquet")
    print("prices", raw["Close"].shape, raw.index.min().date(), raw.index.max().date())


def earn_dates(tickers: list[str]) -> None:
    rows, info = [], []
    for i, tk in enumerate(tickers):
        for attempt in range(3):
            try:
                t = yf.Ticker(tk)
                e = t.get_earnings_dates(limit=16)
                if e is not None and len(e):
                    for ts, r in e.iterrows():
                        rows.append({"ticker": tk, "ts": ts.isoformat(), "eps_est": r.get("EPS Estimate"),
                                     "eps_rep": r.get("Reported EPS"), "surprise_pct": r.get("Surprise(%)")})
                fi = t.fast_info
                info.append({"ticker": tk, "shares": getattr(fi, "shares", None),
                             "mcap_now": getattr(fi, "market_cap", None)})
                break
            except Exception as ex:  # 일시 오류는 다시
                if attempt == 2:
                    print("fail", tk, ex)
                time.sleep(2)
        time.sleep(0.3)
    pd.DataFrame(rows).to_csv(CACHE / "earn_dates.csv", index=False)
    pd.DataFrame(info).to_csv(CACHE / "info.csv", index=False)
    print("earn rows", len(rows))


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    uni = pd.read_csv(BASE / "results/universe.csv")
    tickers = sorted(uni.ticker)
    prices(tickers + BENCH)
    earn_dates(tickers)


if __name__ == "__main__":
    main()
