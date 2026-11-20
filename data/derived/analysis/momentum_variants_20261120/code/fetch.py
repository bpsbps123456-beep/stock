"""가격 받기: 넓은 종목(앞선 넓은 연구 2025~2026 실적 종목) + 두 펀드 2026 보유 + watchlist AI_UNIVERSE 미국 + 기준 ETF.
2024-06-01~2026-11-19 일봉(수정 계수 포함). 결과(Git 제외): cache/px_{open,close,adjclose,volume}.parquet, cache/meta.csv
"""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
import watchlist as w  # noqa: E402

BENCH = ["QQQ", "SPY", "SOXX", "IWM", "XLK", "^VIX"]


def main() -> None:
    ev = pd.read_csv(ROOT / "data/derived/analysis/earnings_trend/results/events_wide.csv",
                     usecols=["ticker", "reaction_date", "sector", "etf"])
    wide = set(ev[ev.reaction_date >= "2025-06-01"].ticker)
    fund = set(pd.read_csv(ROOT / "data/derived/analysis/post_earnings_risers_20261120/results/held_days.csv").ticker)
    ai = {s for g in w.AI_UNIVERSE.values() for v in g.values() for s in v if "." not in s and not s[0].isdigit()}
    tickers = sorted(wide | fund | ai) + BENCH
    parts = {c: [] for c in ("Open", "Close", "Adj Close", "Volume")}
    for i in range(0, len(tickers), 200):
        raw = yf.download(tickers[i:i + 200], start="2024-06-01", end="2026-11-20", auto_adjust=False, actions=False,
                          group_by="column", threads=True, progress=False)
        for c in parts:
            parts[c].append(raw[c])
        print(i, raw["Close"].shape, flush=True)
    for c, nm in (("Open", "open"), ("Close", "close"), ("Adj Close", "adjclose"), ("Volume", "volume")):
        df = pd.concat(parts[c], axis=1, sort=True)
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_parquet(BASE / f"cache/px_{nm}.parquet")
    sec = ev.drop_duplicates("ticker", keep="last").set_index("ticker")[["sector", "etf"]]
    meta = pd.DataFrame(index=sorted(wide | fund | ai))
    meta["sector"] = sec.sector.reindex(meta.index)
    meta["etf"] = sec.etf.reindex(meta.index)
    meta["in_wide"] = meta.index.isin(wide)
    meta["in_fund"] = meta.index.isin(fund)
    meta["in_ai"] = meta.index.isin(ai)
    meta.to_csv(BASE / "cache/meta.csv")
    c = pd.read_parquet(BASE / "cache/px_close.parquet")
    print("done", c.shape, c.index.min().date(), c.index.max().date(), "last-row NaN", int(c.iloc[-1].isna().sum()))


if __name__ == "__main__":
    main()
