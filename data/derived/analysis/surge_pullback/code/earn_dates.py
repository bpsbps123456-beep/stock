"""급등 종목의 실적 발표 일시(yfinance get_earnings_dates, 최근 24분기). 실적 급등과 그 밖의 급등을 나누는 데만 쓴다.
premarket_shock/results/earnings_dates.json에 이미 있는 종목은 그대로 가져오고(읽기만), 나머지만 받는다.
저장: ../results/earnings_dates.json {티커: [ISO 일시]}. 실패한 종목은 저장하지 않아 다시 실행하면 이어 받는다.
사용: py earn_dates.py <종목 json 목록>"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "../results/earnings_dates.json")
OLD = os.path.join(HERE, "../../premarket_shock/results/earnings_dates.json")


def one(t):
    for attempt in range(3):
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=24)
            return t, [] if e is None else [i.tz_convert("America/New_York").isoformat() for i in e.index]
        except Exception as ex:
            if "No earnings dates" in str(ex) or "not found" in str(ex).lower():
                return t, []
            time.sleep(5 * (attempt + 1))
    return t, None


def main():
    syms = json.load(open(sys.argv[1]))
    done = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    old = json.load(open(OLD, encoding="utf-8")) if os.path.exists(OLD) else {}
    for t in syms:
        if t not in done and old.get(t):
            done[t] = old[t]
    todo = sorted(set(syms) - set(done))
    print("받을 종목", len(todo), "(이미 있음", len(done), ")", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for n, (t, v) in enumerate(pool.map(one, todo), 1):
            if v is not None:
                done[t] = v
            if n % 100 == 0:
                json.dump(done, open(OUT, "w", encoding="utf-8"))
                print(n, flush=True)
    json.dump(done, open(OUT, "w", encoding="utf-8"))
    print("완료", len(done))


if __name__ == "__main__":
    main()
