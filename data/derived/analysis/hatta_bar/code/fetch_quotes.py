"""E1 실측 비용: 주 가설 칸 신호마다 Alpaca SIP 호가 두 번(신호 마감 + 10초의 매도호가, 30분 창 끝의 매수호가)을 받는다.
사용: py fetch_quotes.py [칸 이름…]   → cache/quotes.parquet (기본 원안 두 칸, 뒤 구간만) (sym, ts, ask_in, bid_in, ask_out, bid_out)"""
import os, glob
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests
import engine as E
from build import SYMS
from fetch_alpaca import keys, Pace

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(E.CACHE, "quotes.parquet")
URL = "https://data.alpaca.markets/v2/stocks/{sym}/quotes"


def last_quote(sym, end, head, pace, look=300):
    """end 시각(UTC) 직전 마지막 NBBO."""
    params = {"start": (end - pd.Timedelta(seconds=look)).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
              "end": end.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "limit": 1, "sort": "desc", "feed": "sip"}
    for attempt in range(6):
        pace.wait()
        r = requests.get(URL.format(sym=sym), headers=head, params=params, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            continue
        r.raise_for_status()
        q = r.json().get("quotes") or []
        return (q[0]["ap"], q[0]["bp"]) if q else (float("nan"), float("nan"))
    return float("nan"), float("nan")


def one(row, head, pace):
    sym, ts = row
    t = pd.Timestamp(int(ts) * 60, unit="s", tz="UTC")
    ask_in, bid_in = last_quote(sym, t + pd.Timedelta(seconds=70), head, pace)      # 봉 마감(+60초) + 10초
    ask_out, bid_out = last_quote(sym, t + pd.Timedelta(minutes=31), head, pace)    # 창 마지막 봉(+30분 시작) 마감
    return sym, int(ts), ask_in, bid_in, ask_out, bid_out


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or ["orig_A", "orig_B"]          # 기본: 원안 두 칸의 뒤 구간(9/1~) 신호
    need = set()
    for nm in names:
        s = pd.read_csv(os.path.join(E.CACHE, f"../results/signals_{nm}.csv"), usecols=["sid", "ts", "day"])
        s = s[s.day > 20260831]
        need |= {(SYMS[i], t) for i, t in zip(s.sid, s.ts)}
    done = pd.read_parquet(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["sym", "ts"])
    have = set(zip(done.sym, done.ts.astype(int)))
    todo = sorted(need - have)
    print(len(need), "신호,", len(todo), "받을 것", flush=True)
    head, pace = keys(), Pace(3.0)
    rows = []
    with ThreadPoolExecutor(3) as ex:
        for i, r in enumerate(ex.map(lambda x: one(x, head, pace), todo)):
            rows.append(r)
            if i % 200 == 0:
                print(i, flush=True)
    new = pd.DataFrame(rows, columns=["sym", "ts", "ask_in", "bid_in", "ask_out", "bid_out"])
    pd.concat([done, new], ignore_index=True).to_parquet(OUT)
    print("done", len(new))
