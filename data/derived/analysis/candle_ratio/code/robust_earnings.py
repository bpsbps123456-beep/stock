"""robust 갈래: 144종목 실적 발표 시각을 yfinance get_earnings_dates로 받아 둔다.
py code/robust_earnings.py → results/robust_earnings_dates.csv (sym, ts_et, date, ok)
받기 실패한 종목은 ok=False 한 줄만 남긴다(분해 단계에서 큰 갭 날로 대신한다).
"""
import time

import pandas as pd
import yfinance as yf

from engine import CACHE, ET, HERE

OUT = HERE / "results" / "robust_earnings_dates.csv"
SYMS = sorted(p.stem[:-3] for p in CACHE.glob("*_5m.parquet"))
LO, HI = pd.Timestamp("2026-04-15", tz=ET), pd.Timestamp("2026-10-23", tz=ET)


def get(sym):
    for k in range(3):
        try:
            e = yf.Ticker(sym).get_earnings_dates(limit=12)
            if e is None or len(e) == 0:
                return []
            ts = pd.DatetimeIndex(e.index).tz_convert(ET)
            return [t for t in ts if LO <= t <= HI]
        except Exception as ex:  # 잠깐 막히면 쉬었다 다시
            print(sym, "다시 시도", type(ex).__name__, flush=True)
            time.sleep(4 * (k + 1))
    return None


rows = []
for i, s in enumerate(SYMS, 1):
    ts = get(s)
    if ts is None:
        rows.append(dict(sym=s, ts_et=None, date=None, ok=False))
    elif not ts:
        rows.append(dict(sym=s, ts_et=None, date=None, ok=True))
    else:
        for t in ts:
            rows.append(dict(sym=s, ts_et=t.isoformat(), date=t.date().isoformat(), ok=True))
    print(i, s, "실패" if ts is None else [t.strftime("%m-%d %H:%M") for t in ts], flush=True)
    time.sleep(0.3)
pd.DataFrame(rows).to_csv(OUT, index=False)
print("저장", OUT, "실패 종목", sum(1 for r in rows if not r["ok"]))
