"""넓은 종목군 10분봉(Alpaca SIP, 04:00~20:00 ET, 분할 반영) 2025-01-02~지금-16분. 공용 받기 함수 alpaca_sip/fetch_alpaca.one 사용.
종목: watchlist.AI_SUBSECTOR 미국 상장 115개 + ETF 12개 + 3배 레버리지 2개. 결과: cache/10m/<종목>.parquet (Git 제외)
사용: py fetch10.py   (이미 받은 종목은 건너뜀)"""
import os, sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../alpaca_sip"))
from fetch_alpaca import keys, Pace, one, ET  # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../../../..")))
import watchlist  # noqa: E402

AI = sorted(s for s in watchlist.AI_SUBSECTOR if "." not in s)
ETF = ["SOXX", "SMH", "QQQ", "SPY", "IWM", "DIA", "XLK", "IGV", "XLF", "XLE", "XLV", "XLI"]
LEV = ["SOXL", "TQQQ"]
SYMS = AI + [s for s in ETF + LEV if s not in AI]

if __name__ == "__main__":
    out = os.path.join(HERE, "../cache/10m")
    os.makedirs(out, exist_ok=True)
    s = pd.Timestamp("2025-01-02", tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    e = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)).floor("min").strftime("%Y-%m-%dT%H:%M:%SZ")
    head, pace = keys(), Pace()
    todo = [x for x in SYMS if not os.path.exists(os.path.join(out, f"{x}.parquet"))]
    print(len(SYMS), "종목 중", len(todo), "받음", s, "→", e, flush=True)
    with ThreadPoolExecutor(8) as ex:
        for sym, n in ex.map(lambda x: one(x, s, e, head, pace, out, "10Min"), todo):
            print(sym, n, flush=True)
