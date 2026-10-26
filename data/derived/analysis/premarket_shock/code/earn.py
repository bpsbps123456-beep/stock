"""사건 종목의 실적 발표 일시(yfinance get_earnings_dates). 저장: results/earnings_dates.json {티커: [ISO 일시]}.
이미 받은 종목은 건너뛴다. 사용: py earn.py [사건 csv 이름, 기본 events_raw.csv]"""
import json, os, sys, time
import pandas as pd, yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "../results/earnings_dates.json")

if __name__ == "__main__":
    ev = pd.read_csv(os.path.join(HERE, "../results", sys.argv[1] if len(sys.argv) > 1 else "events_raw.csv"))
    done = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    todo = sorted(set(ev[ev.cap >= 1e9].ticker) - set(done))
    print("받을 종목", len(todo), flush=True)

    def one(t):
        for attempt in range(3):
            try:
                e = yf.Ticker(t).get_earnings_dates(limit=24)
                return t, [] if e is None else [i.tz_convert("America/New_York").isoformat() for i in e.index]
            except Exception as ex:
                if "No earnings dates" in str(ex) or "not found" in str(ex).lower():
                    return t, []
                time.sleep(5 * (attempt + 1))
        return t, None                                    # 실패는 저장하지 않아 다음 실행에서 다시 받는다

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:
        for n, (t, v) in enumerate(pool.map(one, todo), 1):
            if v is not None:
                done[t] = v
            if n % 50 == 0:
                json.dump(done, open(OUT, "w", encoding="utf-8"))
                print(n, flush=True)
    json.dump(done, open(OUT, "w", encoding="utf-8"))
    print("완료", len(done))
