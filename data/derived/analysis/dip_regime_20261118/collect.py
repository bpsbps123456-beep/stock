"""Cache split-adjusted SIP and Blue Ocean bars for the 2026 dip study."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "alpaca_sip"))
sys.path.insert(0, str(HERE.parent / "dip_reason/code"))
from fetch import user_universe
from fetch_alpaca import ET, URL, Pace, keys

CACHE = HERE / "cache"
END = "2026-11-18T00:59:59Z"  # 11/17 19:59:59 ET, the last completed session
SYMBOLS = user_universe()["합집합"]


def download(sym, feed, tf, headers, pace, *, follow=False):
    path = (CACHE / "next" if follow else CACHE) / f"{sym}_{feed}_{tf}.parquet"
    if path.exists():
        return
    params = dict(symbols=sym, feed=feed, timeframe=tf, limit=10000, adjustment="split",
                  asof="2026-11-17", sort="asc", end=END,
                  start="2026-05-01T04:00:00Z" if tf == "1Day" else "2026-06-21T23:00:00Z")
    if follow:
        params.update(start="2026-11-18T01:00:00Z", end="2026-11-18T14:29:59Z")
    rows = []
    with requests.Session() as session:
        while True:
            for attempt in range(6):
                pace.wait()
                try:
                    response = session.get(URL, headers=headers, params=params, timeout=45)
                    if response.status_code != 429 and response.status_code < 500:
                        response.raise_for_status()
                        break
                except (requests.ConnectionError, requests.Timeout):
                    if attempt == 5:
                        raise
                time.sleep(2 ** attempt)
            else:
                response.raise_for_status()
            page = response.json()
            rows.extend((page.get("bars") or {}).get(sym, []))
            token = page.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
    bars = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    bars.index = pd.to_datetime(bars.pop("t"), utc=True).dt.tz_convert(ET)
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError(f"Unordered or duplicate timestamps in {path.name}")
    bars.columns = ["Open", "High", "Low", "Close", "Volume"]
    tmp = path.with_suffix(".part")
    bars.to_parquet(tmp)
    tmp.replace(path)


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)
    headers, pace = keys(), Pace()
    follow = "--settle" in sys.argv[1:]
    if follow:
        (CACHE / "next").mkdir(exist_ok=True)
        trades = pd.read_parquet(CACHE / "trades.parquet", columns=["sym", "model", "reason"])
        syms = trades.loc[(trades.model != "chart") & (trades.reason == "평가"), "sym"].unique()
    else:
        syms = [*SYMBOLS, "QQQ"]
    tasks = [(s, f, "3Min") for s in syms for f in ("sip", "boats")]
    if not follow:
        tasks.append(("QQQ", "sip", "1Day"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, *task, headers, pace, follow=follow) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 10 == 0 or count == len(tasks):
                print(f"{count}/{len(tasks)}", flush=True)
