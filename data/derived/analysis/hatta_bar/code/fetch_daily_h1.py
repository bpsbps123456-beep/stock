"""3차: 넓힌 400종목의 1~6월 확인용 일봉(전일 종가·20일 거래대금·5일 수익률·ATR 계산용).
사용: py fetch_daily_h1.py   → cache/daily_2026h1.parquet, cache/daily_2025h1.parquet"""
import os, json
from universe2 import daily, CACHE, RES
from fetch_alpaca import keys, Pace

if __name__ == "__main__":
    syms = sorted(json.load(open(os.path.join(RES, "universe2.json"), encoding="utf-8")))
    head, pace = keys(), Pace(3.0)
    for tag, a, b in (("2026h1", "2025-11-01T00:00:00Z", "2026-07-01T00:00:00Z"),
                      ("2025h1", "2024-11-01T00:00:00Z", "2025-07-01T00:00:00Z")):
        d = daily(syms, a, b, head, pace)
        d.to_parquet(os.path.join(CACHE, f"daily_{tag}.parquet"))
        print(tag, len(d), d.sym.nunique())
