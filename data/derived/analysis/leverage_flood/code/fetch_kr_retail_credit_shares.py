"""신용거래 체결주수 추이(잔고 주수) 긴 이력 수집 — 금융투자협회 freesis STATSCU0100000080BO.

출처: https://freesis.kofia.or.kr  POST /meta/getMetaDataList.do
본문: {"dmSearch":{"tmpV40":"1","tmpV41":"1","tmpV1":"D","tmpV45":"YYYYMMDD","tmpV46":"YYYYMMDD",
        "OBJ_NM":"STATSCU0100000080BO"}}
  - tmpV40="1" 이면 주 단위 그대로(나눗수 없음). 10/12 TMPV2=931,216,729 주.
  - 열: TMPV1 일자, TMPV2 신용거래융자 합계 주수, TMPV3 유가, TMPV4 코스닥,
        TMPV5 신용거래대주 합계 주수, TMPV6 유가, TMPV7 코스닥 (TMPV8·9 는 빈 열).
  - 이름은 '체결주수'지만 하루 9~15억 주라 일별 신규 체결이 아니라 잔고 주수다(fetch_kr_retail.py 와 같은 판단).
  - 왜 긴 이력인가: 250거래일 백분위를 2025-10 부터 온전히 재려면 2024-06 부터 필요하다.
    기존 kr_retail_credit_shares.csv(2025-06~)는 건드리지 않고 새 파일에 쓴다.

실행: py fetch_kr_retail_credit_shares.py [--start 20240601] [--end 20261014]
출력: ../results/kr_retail_credit_shares_long.csv  (date + 주수 열, 단위 주)
      ../results/kr_retail_credit_shares_long_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://freesis.kofia.or.kr"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://freesis.kofia.or.kr/stat/FreeSIS.do",
}
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OBJ = "STATSCU0100000080BO"
COLMAP = {
    "TMPV2": "credit_loan_shares_total",
    "TMPV3": "credit_loan_shares_kospi",
    "TMPV4": "credit_loan_shares_kosdaq",
    "TMPV5": "stock_lending_shares_total",
    "TMPV6": "stock_lending_shares_kospi",
    "TMPV7": "stock_lending_shares_kosdaq",
}


def fetch(start: str, end: str, retries: int = 3) -> dict:
    body = {"dmSearch": {"tmpV40": "1", "tmpV41": "1", "tmpV1": "D",
                         "tmpV45": start, "tmpV46": end, "OBJ_NM": OBJ}}
    last = None
    for i in range(retries):
        try:
            r = requests.post(BASE + "/meta/getMetaDataList.do", json=body, headers=HEADERS, timeout=120)
            return r.json()
        except Exception as ex:  # noqa: BLE001
            last = ex
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{OBJ}: {last}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20240601")
    ap.add_argument("--end", default="20261014")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 한 번에 긴 구간을 요청해도 되지만, 서버가 자르는 경우를 대비해 반기 단위로 나눠 받는다.
    bounds = pd.date_range(pd.Timestamp(args.start), pd.Timestamp(args.end), freq="6MS").tolist()
    if not bounds or bounds[0] > pd.Timestamp(args.start):
        bounds = [pd.Timestamp(args.start)] + bounds
    bounds.append(pd.Timestamp(args.end) + pd.Timedelta(days=1))
    rows: list[dict] = []
    header = None
    for a, b in zip(bounds[:-1], bounds[1:]):
        s, e = a.strftime("%Y%m%d"), (b - pd.Timedelta(days=1)).strftime("%Y%m%d")
        j = fetch(s, e)
        header = header or j.get("dsmHeader")
        part = j.get("ds1", [])
        rows.extend(part)
        print(f"{s}~{e}: {len(part)} rows")
        time.sleep(0.5)

    df = pd.DataFrame(rows).rename(columns={"TMPV1": "date", **COLMAP})
    df = df[["date"] + list(COLMAP.values())].copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in COLMAP.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    out = RESULTS / "kr_retail_credit_shares_long.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved {len(df)} rows {df['date'].min().date()}~{df['date'].max().date()} -> {out}")

    # 기존 파일(2025-06~)과 겹치는 구간이 같은지 확인한다.
    old_p = RESULTS / "kr_retail_credit_shares.csv"
    check = None
    if old_p.exists():
        old = pd.read_csv(old_p, parse_dates=["date"])
        m = old.merge(df, on="date", suffixes=("_old", "_new"))
        diff = int((m["credit_loan_shares_total_old"] != m["credit_loan_shares_total_new"]).sum())
        check = {"overlap_rows": int(len(m)), "total_mismatch_rows": diff}
        print("overlap check:", check)

    meta = {
        "source": BASE + "/meta/getMetaDataList.do",
        "obj_nm": OBJ,
        "fetched_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "start": args.start, "end": args.end,
        "rows": int(len(df)),
        "first_date": df["date"].min().strftime("%Y-%m-%d"),
        "last_date": df["date"].max().strftime("%Y-%m-%d"),
        "unit": "주(tmpV40=1)",
        "columns": COLMAP,
        "dsmHeader": header,
        "overlap_check_vs_kr_retail_credit_shares.csv": check,
    }
    (RESULTS / "kr_retail_credit_shares_long_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
