"""Save a KOFIA freesis daily series to CSV with header names, and probe neighbouring object names.

Usage: py kofia_freesis_save.py 20260101 20261013 STATSCU0100000060BO
"""
import csv
import json
import os
import sys

import requests

URL = "https://freesis.kofia.or.kr/meta/getMetaDataList.do"
RESULTS = r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis\leverage_flood\results"


def fetch(obj, start, end):
    payload = {"dmSearch": {"tmpV40": "10000000", "tmpV41": "1", "tmpV1": "D",
                            "tmpV45": start, "tmpV46": end, "OBJ_NM": obj}}
    headers = {"Content-Type": "application/json; charset=UTF-8", "User-Agent": "Mozilla/5.0",
               "Referer": "https://freesis.kofia.or.kr/"}
    r = requests.post(URL, data=json.dumps(payload), headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def save(obj, start, end):
    data = fetch(obj, start, end)
    rows = data.get("ds1", [])
    header = data.get("dsmHeader")
    unit = data.get("unit")
    print(f"[{obj}] unit={unit} rows={len(rows)}")
    print(f"[{obj}] header={json.dumps(header, ensure_ascii=False)[:1500]}")
    if not rows:
        return
    cols = list(rows[0].keys())
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"kofia_{obj}_{start}_{end}.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in sorted(rows, key=lambda r: r.get("TMPV1", "")):
            w.writerow(row)
    print(f"[{obj}] saved -> {out}")
    for row in sorted(rows, key=lambda r: r.get("TMPV1", "")):
        d = row.get("TMPV1", "")
        if d in {"20260619", "20260622", "20260623", "20260624", "20260728", "20260729", "20260730", "20260731", "20260803",
                 "20261002", "20261006", "20261007", "20261008", "20261012", "20261013"}:
            print("   ", json.dumps(row, ensure_ascii=False))


def main():
    start, end = sys.argv[1], sys.argv[2]
    objs = sys.argv[3:]
    for obj in objs:
        try:
            save(obj, start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"[{obj}] error: {exc}")


if __name__ == "__main__":
    main()
