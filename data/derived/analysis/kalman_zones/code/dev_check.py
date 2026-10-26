"""SNDK 9/28 3분봉: 보라선(EMA 60) 대비 저가 이격 확인 (yfinance 1분봉 → 3분봉, 프리·애프터 포함, 야간 세션 없음)."""
import sys
import numpy as np, pandas as pd
from osc import rsi
D1 = sys.argv[1] if len(sys.argv) > 1 and __name__ == "__main__" else r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/e846c1a2-c78d-40ed-885f-777d23bba52f/scratchpad/kalman/data1m"


def bars3(s):
    d = pd.read_parquet(f"{D1}/{s}_1m.parquet")
    r = d.resample("3min", label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Open"])
    return r


if __name__ == "__main__":
    b = bars3("SNDK")
    b["ema60"] = b.Close.ewm(span=60, adjust=False).mean()
    b["rsi"] = rsi(b.Close.values, 14)
    b["gap_low"] = 1 - b.Low / b.ema60                    # 이 봉 보라선 대비 저가
    b["gap_prev"] = 1 - b.Low / b.ema60.shift(1)           # 직전 봉 보라선 대비 (지정가 기준)
    k = b.index.tz_convert("Asia/Seoul")
    b.index = k
    w = b.loc["2026-09-28 22:24":"2026-09-29 00:30"]
    pd.set_option("display.width", 200)
    print(w[["Open", "High", "Low", "Close", "ema60", "gap_low", "gap_prev", "rsi"]].round(
        {"Open": 2, "High": 2, "Low": 2, "Close": 2, "ema60": 2, "gap_low": 4, "gap_prev": 4, "rsi": 1}).to_string())
