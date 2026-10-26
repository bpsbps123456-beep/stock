"""QQQ 먼 만기(약 1년) 옵션 가격 이력(2026-10-14). 사용자 가설 '다소 먼 미래의 풋이 쌀 때'를 가격으로 재기 위한 자료.

Yahoo는 아직 상장된 계약의 일봉 이력만 준다(만기 지난 계약은 404). 그래서 2026년 안에서 1년 만기를 앞뒤로 감싸는
만기 7개(2026-12 ~ 2028-01)만 쓴다. 2025년에는 1년 만기를 감쌀 계약이 이미 만기돼 이 지표가 없다.
종가는 그날 마지막 체결가라 체결 시각이 QQQ 종가와 다를 수 있다(분석에서 체결 있는 날만 쓰고 여러 행사가를 묶어 줄인다).

실행: py fetch_leaps.py   (받은 계약은 건너뛴다. 결과 cache/leaps/QQQ_<만기>.parquet)
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "cache" / "leaps"
UNDERLYING = "QQQ"
EXPIRIES = ["2026-12-18", "2027-01-15", "2027-03-19", "2027-06-17", "2027-09-17", "2027-12-17", "2028-01-21"]
PUT_K = (450, 760)   # 2026 QQQ 종가 558~760의 0.80~1.00배
CALL_K = (560, 910)  # 같은 범위의 1.00~1.20배
STEP = 10
SINCE = "2025-06-01"


def contracts(expiry: str) -> list[tuple[str, str, float]]:
    ch = yf.Ticker(UNDERLYING).option_chain(expiry)
    out = []
    for kind, df, (lo, hi) in (("P", ch.puts, PUT_K), ("C", ch.calls, CALL_K)):
        for sym, k in zip(df["contractSymbol"], df["strike"]):
            if lo <= k <= hi and k % STEP == 0:
                out.append((sym, kind, float(k)))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for e in EXPIRIES:
        f = OUT / f"{UNDERLYING}_{e}.parquet"
        if f.exists():
            continue
        rows = []
        for sym, kind, k in contracts(e):
            h = yf.Ticker(sym).history(start=SINCE, auto_adjust=False)
            if len(h):
                rows.append(pd.DataFrame({"date": h.index.strftime("%Y-%m-%d"), "kind": kind, "strike": k,
                                          "close": h["Close"].to_numpy(), "volume": h["Volume"].to_numpy()}))
            time.sleep(0.25)
        df = pd.concat(rows, ignore_index=True)
        df.to_parquet(f, index=False)
        print(e, len(rows), "계약", len(df), "행", flush=True)


if __name__ == "__main__":
    main()
