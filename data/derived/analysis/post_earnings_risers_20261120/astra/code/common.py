from pathlib import Path
import time
import threading
import requests
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CACHE = BASE / 'cache'
DATA = BASE / 'data'
UA = {"User-Agent": "GBRC-research personal-noncommercial research@example.com", "Accept-Encoding": "gzip, deflate"}
TICKERS = sorted(set(pd.read_csv(BASE.parent / 'results/universe.csv').ticker) | {'SPY','QQQ','SOXX'})
START, END = '2025-09-01', '2026-11-19'
_sec_lock = threading.Lock()
_sec_last = 0.0
_clients = threading.local()

def fetch(url, cache, sec=False, params=None):
    path = CACHE / cache
    if path.exists():
        return path
    for attempt in range(3):
        if sec:
            global _sec_last
            with _sec_lock:
                time.sleep(max(0,.3-(time.monotonic()-_sec_last)))
                _sec_last=time.monotonic()
        try:
            if not hasattr(_clients,'session'):
                _clients.session=requests.Session()
            r = _clients.session.get(url, headers=UA, timeout=45, params=params)
            r.raise_for_status()
            path.parent.mkdir(exist_ok=True, parents=True)
            tmp = path.with_suffix(path.suffix+'.tmp')
            tmp.write_bytes(r.content)
            tmp.replace(path)
            return path
        except requests.RequestException as exc:
            if getattr(exc.response, 'status_code', None) in (400,403,404,429) or attempt == 2:
                raise
            time.sleep(2 ** attempt)

