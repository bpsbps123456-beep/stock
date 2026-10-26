"""양봉 비율 연구 자료: yfinance 5분봉(프리·애프터 포함, 최근 60일)과 1시간봉(2026-05-01~)을 받아
trend_zones/cache/dip_raw_1010의 7/17~10/09 5분봉과 이어 붙인다. 겹치는 시각은 새로 받은 값을 쓴다.
사용: py code/fetch.py  (candle_ratio 폴더에서)
"""
import pathlib
import time

import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).resolve().parent.parent
OLD = HERE.parent / "trend_zones" / "cache" / "dip_raw_1010"
CACHE = HERE / "cache"
SYMS = sorted(p.stem[3:] for p in OLD.glob("5m_*.parquet"))


def norm(d):
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    d.index = d.index.tz_convert("America/New_York")
    return d


def get(sym, **kw):
    for k in range(3):
        try:
            return norm(yf.download(sym, progress=False, auto_adjust=False, prepost=True, threads=False, **kw))
        except Exception as e:  # 잠깐 막히면 쉬었다 다시
            print(sym, "다시 시도", type(e).__name__, flush=True)
            time.sleep(5 * (k + 1))
    return None


for s in SYMS:
    new5 = get(s, interval="5m", period="60d")
    old5 = norm(pd.read_parquet(OLD / f"5m_{s}.parquet"))
    d5 = pd.concat([old5, new5]) if new5 is not None else old5
    d5 = d5[~d5.index.duplicated(keep="last")].sort_index()
    d5.to_parquet(CACHE / f"{s}_5m.parquet")
    h1 = get(s, interval="1h", start="2026-05-01")
    if h1 is not None:
        h1.to_parquet(CACHE / f"{s}_1h.parquet")
    print(s, "5m", len(d5), d5.index[0].date(), "->", d5.index[-1], "| 1h", 0 if h1 is None else len(h1), flush=True)
