"""SOXL 1분봉(Alpaca SIP, 연장시간 04:00~20:00 ET, 분할 반영)을 받는다.
무료 키는 15분 지난 자료만 주므로 끝 시각은 지금-16분으로 자른다.
결과: cache/SOXL_1m.parquet (ET 시각 인덱스, Open·High·Low·Close·Volume·Trades·VWAP)
사용: py fetch.py [시작=2026-08-03]
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../alpaca_sip"))
from fetch_alpaca import keys, Pace, one  # noqa: E402  같은 받기 함수를 재사용

ET = "America/New_York"

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2026-08-03"
    out = os.path.join(HERE, "../cache")
    os.makedirs(out, exist_ok=True)
    s = pd.Timestamp(start, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    e = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)).floor("min").strftime("%Y-%m-%dT%H:%M:%SZ")
    sym, n = one("SOXL", s, e, keys(), Pace(), out)
    src = os.path.join(out, "SOXL.parquet")
    dst = os.path.join(out, "SOXL_1m.parquet")
    os.replace(src, dst)
    d = pd.read_parquet(dst)
    print(sym, n, d.index.min(), d.index.max())
