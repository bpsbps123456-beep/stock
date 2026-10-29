"""Alpaca SIP 1분봉(04:00~20:00 ET, 분할 반영) — SOXL·SOXS(매매), SOXX·QQQ·NVDA(설명·조건용).
키는 저장소 루트 .env(Git 제외). 결과: cache/alp1m/<TICKER>.parquet (ET 시각 인덱스). 사용: py fetch.py [시작 끝]"""
import os, sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../hatta_bar/code"))
from fetch_alpaca import keys, Pace, one  # noqa: E402  같은 받기 함수를 다시 쓴다

SYMS = ["SOXL", "SOXS", "SOXX", "QQQ", "NVDA"]
ET = "America/New_York"

if __name__ == "__main__":
    a, b = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2024-12-01", "2026-10-29")
    out = os.path.join(HERE, "../cache/alp1m")
    os.makedirs(out, exist_ok=True)
    start = pd.Timestamp(a, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    end = pd.Timestamp(b, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    head, pace = keys(), Pace()
    with ThreadPoolExecutor(3) as ex:
        for sym, n in ex.map(lambda s: one(s, start, end, head, pace, out), SYMS):
            print(sym, n, flush=True)
