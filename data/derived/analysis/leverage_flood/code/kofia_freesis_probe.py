"""Probe KOFIA freesis for the daily forced-sale (반대매매) series and dump raw columns.

Object STATSCU0100000060BO is the one the parent task verified (10/12 row TMPV6=2708, TMPV7=2.7).
Usage: py kofia_freesis_probe.py 20260601 20261013
"""
import json
import sys

import requests

URL = "https://freesis.kofia.or.kr/meta/getMetaDataList.do"


def fetch(obj, start, end):
    payload = {
        "dmSearch": {
            "tmpV40": "10000000",
            "tmpV41": "1",
            "tmpV1": "D",
            "tmpV45": start,
            "tmpV46": end,
            "OBJ_NM": obj,
        }
    }
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://freesis.kofia.or.kr/",
    }
    r = requests.post(URL, data=json.dumps(payload), headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "20260601"
    end = sys.argv[2] if len(sys.argv) > 2 else "20261013"
    objs = sys.argv[3:] or ["STATSCU0100000060BO"]
    for obj in objs:
        try:
            data = fetch(obj, start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"[{obj}] error: {exc}")
            continue
        keys = list(data.keys())
        print(f"[{obj}] top keys: {keys}")
        for k in keys:
            v = data[k]
            if isinstance(v, list):
                print(f"  {k}: {len(v)} rows")
                for row in v[:3]:
                    print("   ", json.dumps(row, ensure_ascii=False)[:600])
                if len(v) > 3:
                    print("    ...")
                    print("   ", json.dumps(v[-1], ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
