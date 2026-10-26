# -*- coding: utf-8 -*-
"""KRX 공매도 잔고(종목별 일별) 수집 — 하이닉스(000660)·삼성전자(005930).

data.krx.co.kr generate.cmd/getJsonData.cmd 비공식 엔드포인트.
공매도 잔고는 보고 의무자가 T+2 보고 → KRX가 T+2 장 마감 후 공표(발표 시차 T+2).
결과: results/krx_short_balance_<code>.csv
"""
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)

URL = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020301",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
}

ISU = {"000660": "KR7000660001", "005930": "KR7005930003"}
NAMES = {"000660": "SK하이닉스", "005930": "삼성전자"}


def post(params):
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(URL, data=data, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_balance(code, strt, end):
    # MDCSTAT30502: 개별종목 공매도 잔고 추이 (일별)
    params = {
        "bld": "dbms/MDC/STAT/srt/MDCSTAT30502",
        "locale": "ko_KR",
        "isuCd": ISU[code],
        "strtDd": strt,
        "endDd": end,
        "share": "1",
        "money": "1",
        "csvxls_isNo": "false",
    }
    return post(params)


def main():
    strt = sys.argv[1] if len(sys.argv) > 1 else "20260501"
    end = sys.argv[2] if len(sys.argv) > 2 else "20261013"
    for code in ISU:
        try:
            j = fetch_balance(code, strt, end)
        except Exception as e:  # noqa: BLE001
            print(code, "ERROR", e)
            continue
        key = next((k for k in j if isinstance(j[k], list)), None)
        rows = j.get(key, []) if key else []
        print(code, NAMES[code], "rows:", len(rows), "key:", key)
        if rows:
            print("columns:", list(rows[0].keys()))
            print("first:", rows[-1])
            print("last:", rows[0])
            path = os.path.join(OUT, f"krx_short_balance_{code}.csv")
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in reversed(rows):
                    w.writerow(r)
            print("saved", path)
        time.sleep(1)


if __name__ == "__main__":
    main()
