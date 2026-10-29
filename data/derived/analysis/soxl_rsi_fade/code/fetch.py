"""SOXL·SOXS·SOXX 장중 봉을 야후에서 받는다(프리·애프터 포함). 야후 한도: 1분봉 최근 30일(7일씩), 2분·5분봉 최근 60일.
결과: cache/<간격>/<TICKER>.parquet (ET 시각, Git 제외). 사용: py fetch.py"""
import os, warnings
import pandas as pd, yfinance as yf

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "../cache")
SYMS = ["SOXL", "SOXS", "SOXX", "QQQ"]
PLAN = {"1m": (29, 7), "2m": (59, 59), "5m": (59, 59)}   # 간격: (며칠 전부터, 한 번에 받는 날 수)


def one(sym, iv):
    back, step = PLAN[iv]
    end = pd.Timestamp.now(tz="America/New_York").normalize()          # 오늘(장중) 봉은 뺀다
    a, parts = end - pd.Timedelta(days=back), []
    while a < end:
        b = min(a + pd.Timedelta(days=step), end)
        d = yf.download(sym, interval=iv, start=a.strftime("%Y-%m-%d"), end=b.strftime("%Y-%m-%d"),
                        prepost=True, progress=False, auto_adjust=False, threads=False)
        if len(d):
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            parts.append(d[["Open", "High", "Low", "Close", "Volume"]])
        a = b
    d = pd.concat(parts).sort_index()
    d = d[~d.index.duplicated()].dropna(subset=["Open", "High", "Low", "Close"])
    d.index = d.index.tz_convert("America/New_York")
    os.makedirs(os.path.join(CACHE, iv), exist_ok=True)
    d.to_parquet(os.path.join(CACHE, iv, f"{sym}.parquet"))
    return len(d), d.index[0], d.index[-1]


if __name__ == "__main__":
    for iv in PLAN:
        for s in SYMS:
            print(iv, s, *one(s, iv), flush=True)
