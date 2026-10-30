"""Alpaca SIP 1분봉(연장시간 04:00~20:00 ET, 분할 반영)을 종목별로 받는다. 무료 키는 15분 지난 SIP 이력을 준다.
키: 저장소 루트 .env(Git 제외)의 ALPACA_KEY_ID·ALPACA_SECRET_KEY(또는 APCA_API_KEY_ID·APCA_API_SECRET_KEY).
결과: cache/alp1m<접미사>/<TICKER>.parquet — ET 시각 인덱스, Open·High·Low·Close·Volume·Trades·VWAP.
사용: py fetch_alpaca.py 2026-06-24 2026-10-29           (끝 날짜는 포함 안 함)
      py fetch_alpaca.py 2025-06-24 2025-11-01 _2025
      py fetch_alpaca.py 2026-06-24 2026-10-29 - u2        (2차 넓힌 400종목, 없는 파일만)"""
import os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../../.."))
sys.path.insert(0, ROOT)
import watchlist  # noqa: E402

SYMS = sorted(s for s in watchlist.AI_SUBSECTOR if "." not in s) + ["SPY", "QQQ"]
URL = "https://data.alpaca.markets/v2/stocks/bars"
ET = "America/New_York"


def keys():
    env = dict(os.environ)
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8-sig"):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    kid = env.get("ALPACA_KEY_ID") or env.get("APCA_API_KEY_ID")
    sec = env.get("ALPACA_SECRET_KEY") or env.get("APCA_API_SECRET_KEY")
    if not kid or not sec:
        sys.exit(".env에 ALPACA_KEY_ID·ALPACA_SECRET_KEY가 없다")
    return {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}


class Pace:
    """분당 200회 한도 아래로(초당 3회) 요청 간격을 맞춘다."""
    def __init__(self, per_sec=3.0):
        self.gap, self.next, self.lock = 1.0 / per_sec, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - now))


def one(sym, start, end, head, pace, out):
    rows, token = [], None
    params = {"symbols": sym, "timeframe": "1Min", "start": start, "end": end, "limit": 10000,
              "adjustment": "split", "feed": "sip", "sort": "asc"}
    while True:
        if token:
            params["page_token"] = token
        for attempt in range(6):
            pace.wait()
            r = requests.get(URL, headers=head, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 + 3 * attempt)
                continue
            r.raise_for_status()
            break
        else:
            raise RuntimeError(f"{sym}: 재시도 초과 {r.status_code}")
        j = r.json()
        rows += (j.get("bars") or {}).get(sym, [])
        token = j.get("next_page_token")
        if not token:
            break
    if not rows:
        return sym, 0
    d = pd.DataFrame(rows)
    d.index = pd.to_datetime(d.pop("t"), utc=True).dt.tz_convert(ET)
    d = d.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "n": "Trades", "vw": "VWAP"})
    d = d[["Open", "High", "Low", "Close", "Volume", "Trades", "VWAP"]].sort_index()
    d = d[~d.index.duplicated()]
    mins = d.index.hour * 60 + d.index.minute
    d = d[(mins >= 4 * 60) & (mins < 20 * 60)]
    d.to_parquet(os.path.join(out, f"{sym}.parquet"))
    return sym, len(d)


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    suffix = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else ""
    if len(sys.argv) > 4 and sys.argv[4] == "u2":           # 2차: results/universe2.json 400종목
        import json
        SYMS = sorted(json.load(open(os.path.join(HERE, "../results/universe2.json"), encoding="utf-8"))) + ["SPY", "QQQ"]
    out = os.path.join(HERE, f"../cache/alp1m{suffix}")
    os.makedirs(out, exist_ok=True)
    start = pd.Timestamp(a, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    end = pd.Timestamp(b, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    head, pace = keys(), Pace()
    todo = [s for s in SYMS if not os.path.exists(os.path.join(out, f"{s}.parquet"))]
    print(f"{len(todo)}종목 {start} → {end}", flush=True)
    with ThreadPoolExecutor(8) as ex:                      # 응답이 느려 동시 요청을 늘린다(속도 한도는 Pace가 지킨다)
        for sym, n in ex.map(lambda s: one(s, start, end, head, pace, out), todo):
            print(sym, n, flush=True)
