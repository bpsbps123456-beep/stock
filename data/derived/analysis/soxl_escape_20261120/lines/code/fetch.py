"""SOXL·SOXX 1분봉(Alpaca SIP, 04:00~20:00 ET, 분할 반영) 2025-01-02~지금-16분. 2025년은 느린 선 예열과 보조 확인용.
결과: lines/cache/<SYM>_1m.parquet   사용: py fetch.py [종목 …] (기본 SOXL SOXX)"""
import os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../hatta_bar/code"))
from fetch_alpaca import keys, Pace, one  # noqa: E402
ET = "America/New_York"
if __name__ == "__main__":
    out = os.path.join(HERE, "../cache"); os.makedirs(out, exist_ok=True)
    s = pd.Timestamp("2025-01-02", tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    e = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)).floor("min").strftime("%Y-%m-%dT%H:%M:%SZ")
    head, pace = keys(), Pace()
    for sym in (sys.argv[1:] or ["SOXL", "SOXX"]):
        _, n = one(sym, s, e, head, pace, out)
        os.replace(os.path.join(out, f"{sym}.parquet"), os.path.join(out, f"{sym}_1m.parquet"))
        d = pd.read_parquet(os.path.join(out, f"{sym}_1m.parquet"))
        print(sym, n, d.index.min(), d.index.max(), flush=True)
