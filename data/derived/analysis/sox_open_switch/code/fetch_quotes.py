"""실측 호가 간격 표본: 2025-01~2026-10에서 고른 30일 × SOXL·SOXS × 하루 59개 시각의 직전 NBBO(Alpaca SIP).
시각: 프리장 07:00~09:20 10분 간격·09:25·09:29, 정규장 09:31~10:00 매분, 10:05~11:00 5분 간격(ET).
결과: cache/quotes.parquet (sym, t, ask, bid). 사용: py fetch_quotes.py"""
import os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../hatta_bar/code"))
from fetch_alpaca import keys, Pace  # noqa: E402

URL = "https://data.alpaca.markets/v2/stocks/{sym}/quotes"
OUT = os.path.join(HERE, "../cache/quotes.parquet")


def times_for(day):
    hm = [(7, m) for m in range(0, 60, 10)] + [(8, m) for m in range(0, 60, 10)] + [(9, 0), (9, 10), (9, 20), (9, 25), (9, 29)]
    hm += [(9, m) for m in range(31, 60)] + [(10, 0)] + [(10, m) for m in range(5, 60, 5)] + [(11, 0)]
    return [pd.Timestamp(f"{day} {h:02d}:{m:02d}", tz="America/New_York") for h, m in hm]


def last_quote(sym, t, head, pace):
    end = t.tz_convert("UTC")
    params = {"start": (end - pd.Timedelta(seconds=120)).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 1, "sort": "desc", "feed": "sip"}
    for _ in range(6):
        pace.wait()
        r = requests.get(URL.format(sym=sym), headers=head, params=params, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            continue
        r.raise_for_status()
        q = r.json().get("quotes") or []
        return (sym, t, q[0]["ap"], q[0]["bp"]) if q else (sym, t, np.nan, np.nan)
    return sym, t, np.nan, np.nan


if __name__ == "__main__":
    d = pd.read_parquet(os.path.join(HERE, "../cache/alp1m/SOXL.parquet"))
    days = sorted({x for x in d.index.date if x >= pd.Timestamp("2025-01-02").date()})
    pick = [days[i] for i in np.linspace(0, len(days) - 1, 30).round().astype(int)]
    todo = [(s, t) for day in pick for t in times_for(day) for s in ["SOXL", "SOXS"]]
    print(len(pick), "일", len(todo), "요청", flush=True)
    head, pace = keys(), Pace(3.0)
    with ThreadPoolExecutor(3) as ex:
        rows = list(ex.map(lambda x: last_quote(x[0], x[1], head, pace), todo))
    pd.DataFrame(rows, columns=["sym", "t", "ask", "bid"]).to_parquet(OUT)
    print("done", len(rows), flush=True)
