"""Fresh SEIBRO full history, raw CNN and single-response Yahoo histories."""
import concurrent.futures as cf
import datetime as dt
import gzip
import importlib.util
import json
import threading
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
spec = importlib.util.spec_from_file_location('prior_collect', ROOT.parent/'seibro_flows/code/collect.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
LOCAL = threading.local()


def fetch(job):
    day, kind, dtype = job
    tag = f'{kind}_{dtype or "amt"}_{day:%Y%m%d}'
    raw = ROOT/'data/seibro'/f'{day.year}'/(tag+'.xml.gz')
    cache = ROOT/'data/cache'/(tag+'.json')
    if cache.exists() and raw.exists():
        return json.loads(cache.read_text(encoding='utf-8'))
    if not hasattr(LOCAL, 's'):
        LOCAL.s = requests.Session()
        LOCAL.s.headers.update({'User-Agent': c.UA, 'Referer': c.PAGE})
        LOCAL.s.get(c.PAGE, timeout=40)
        LOCAL.last = 0
    error = None
    for attempt in range(3):
        time.sleep(max(0, .85-(time.monotonic()-LOCAL.last)))
        try:
            r = LOCAL.s.post(c.URL, data=c.body(kind, f'{day:%Y%m%d}', f'{day:%Y%m%d}', dtype).encode(),
                             headers={'Content-Type':'application/xml; charset=UTF-8'}, timeout=60)
            LOCAL.last = time.monotonic()
            r.raise_for_status()
            rows = c.parse(r.content.decode('utf-8', 'replace'), kind)
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(gzip.compress(r.content, mtime=0))
            out = dict(start=str(day.date()), end=str(day.date()), kind=kind, dtype=dtype,
                       fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(), n=len(rows), rows=rows)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
            return out
        except Exception as e:
            error = str(e)
            time.sleep(2*(attempt+1))
    raise RuntimeError(f'{tag}: {error}')


def market():
    folder = ROOT/'data'
    folder.mkdir(parents=True, exist_ok=True)
    headers = {'User-Agent':c.UA, 'Origin':'https://edition.cnn.com', 'Referer':'https://edition.cnn.com/'}
    r = requests.get('https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2023-01-01', headers=headers, timeout=60)
    r.raise_for_status()
    (folder/'cnn.json').write_bytes(r.content)
    frames = []
    from curl_cffi import requests as cr
    for ticker in ['SOXX','SOXL','SOXS']:
        r = cr.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}',
                   params={'period1':int(pd.Timestamp('2022-06-01', tz='UTC').timestamp()),
                           'period2':int(pd.Timestamp('2026-11-12', tz='UTC').timestamp()),
                           'interval':'1d','events':'splits,div'}, impersonate='chrome', timeout=60)
        r.raise_for_status()
        (folder/f'yahoo_{ticker}.json').write_bytes(r.content)
        v = r.json()['chart']['result'][0]
        q = v['indicators']['quote'][0]
        x = pd.DataFrame(q)
        x['date'] = pd.to_datetime(v['timestamp'], unit='s', utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
        x['ticker'] = ticker
        x = x.dropna(subset=['open','close'])
        # Never use a still-open US daily candle.
        cutoff = pd.Timestamp.now(tz='America/New_York')
        x = x[x.date < cutoff.tz_localize(None).normalize()] if cutoff.hour < 16 else x
        frames.append(x)
    pd.concat(frames).to_csv(folder/'prices.csv', index=False)
    (folder/'retrieved.json').write_text(json.dumps({'at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'source':'fresh single full response per ticker; no splice'}, indent=2), encoding='utf-8')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--market-only', action='store_true')
    a = ap.parse_args()
    market()
    if a.market_only:
        return
    days = pd.bdate_range('2023-01-02', '2026-11-11')
    jobs = [(d,k,t) for d in days for k,t in [('daily',v) for v in '1234']+[('custody',None)]]
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for i, result in enumerate(ex.map(fetch, jobs), 1):
            if i%100 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} last={result["start"]} {result["kind"]}', flush=True)
    print('Fresh collection complete', flush=True)


if __name__ == '__main__':
    main()
