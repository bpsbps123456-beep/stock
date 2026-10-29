"""yfinance 1분봉(프리·애프터 포함)을 7일씩 나눠 받아 종목별로 이어 붙인다. 야후는 최근 30일만 연다.
대상: watchlist.AI_SUBSECTOR의 미국 종목 + SPY·QQQ. 결과: cache/yf1m/<TICKER>.parquet (ET 시각, Git 제외).
사용: py fetch_yf.py [시작일 YYYY-MM-DD]"""
import os, sys, warnings
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, yfinance as yf

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../../.."))
OUT = os.path.join(HERE, "../cache/yf1m")
sys.path.insert(0, ROOT)
import watchlist  # noqa: E402

SYMS = sorted(s for s in watchlist.AI_SUBSECTOR if "." not in s) + ["SPY", "QQQ"]


def one(sym, start, end):
    parts, a = [], start
    while a < end:
        b = min(a + pd.Timedelta(days=7), end)
        d = yf.download(sym, interval="1m", start=a.strftime("%Y-%m-%d"), end=b.strftime("%Y-%m-%d"),
                        prepost=True, progress=False, auto_adjust=False, threads=False)
        if len(d):
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            parts.append(d[["Open", "High", "Low", "Close", "Volume"]])
        a = b
    if not parts:
        return sym, 0
    d = pd.concat(parts).sort_index()
    d = d[~d.index.duplicated()].dropna(subset=["Open", "High", "Low", "Close"])
    d.index = d.index.tz_convert("America/New_York")
    d.to_parquet(os.path.join(OUT, f"{sym}.parquet"))
    return sym, len(d)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    end = pd.Timestamp.now(tz="America/New_York").normalize()          # 오늘(장중) 봉은 뺀다
    start = pd.Timestamp(sys.argv[1], tz="America/New_York") if len(sys.argv) > 1 else end - pd.Timedelta(days=28)
    with ThreadPoolExecutor(6) as ex:
        for sym, n in ex.map(lambda s: one(s, start, end), SYMS):
            print(sym, n, flush=True)
