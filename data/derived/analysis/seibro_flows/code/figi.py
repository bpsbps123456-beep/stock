"""ISIN → 티커(OpenFIGI v3 mapping, 키 없음: 분당 25요청 × 10건). 결과는 derived/figi.json에 쌓는다."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "derived"
CACHE = DER / "figi.json"
URL = "https://api.openfigi.com/v3/mapping"


def pick(data: list[dict]) -> dict:
    us = [d for d in data if d.get("exchCode") == "US"] or data
    d = us[0]
    return {"ticker": d.get("ticker"), "name": d.get("name"), "sectype": d.get("securityType"),
            "sectype2": d.get("securityType2"), "exch": d.get("exchCode")}


def main() -> None:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    old = DER / "figi_partial.json"
    if old.exists():  # 검토 단계에서 받은 것(티커·이름만)
        for k, v in json.loads(old.read_text(encoding="utf-8")).items():
            if isinstance(v, dict):
                cache.setdefault(k, v)
    isins = pd.read_csv(DER / "isins.csv")["isin"].dropna().tolist()
    todo = [i for i in isins if i not in cache or "sectype" not in cache[i]]
    print("ISIN", len(isins), "받을 것", len(todo))
    s = requests.Session()
    for k in range(0, len(todo), 10):
        chunk = todo[k:k + 10]
        for attempt in range(5):
            r = s.post(URL, json=[{"idType": "ID_ISIN", "idValue": i} for i in chunk], timeout=60)
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        for isin, res in zip(chunk, r.json()):
            cache[isin] = pick(res["data"]) if res.get("data") else {"ticker": None, "warning": res.get("warning")}
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        time.sleep(2.6)
    hit = sum(1 for i in isins if cache.get(i, {}).get("ticker"))
    print("티커 찾음", hit, "/", len(isins))


if __name__ == "__main__":
    main()
