"""반대매매 금액·비중(미수금 대비) 수집기 — 금융투자협회 freesis 증시자금추이(STATSCU0100000060).

출처: https://freesis.kofia.or.kr  POST /meta/getMetaDataList.do
요청 본문: {"dmSearch":{"tmpV40":"1000000","tmpV41":"1","tmpV1":"D",
            "tmpV45":"YYYYMMDD","tmpV46":"YYYYMMDD","OBJ_NM":"STATSCU0100000060BO"}}
열(2026-10-14 fetch_kr_retail.py 에서 HEADER_NM 으로 확인):
  TMPV1 날짜, TMPV2 투자자예탁금, TMPV3 장내파생 예수금, TMPV4 대고객 RP,
  TMPV5 위탁매매 미수금, TMPV6 미수금 대비 실제 반대매매금액, TMPV7 반대매매비중(%)
단위: tmpV40=1000000 → 백만원. 저장은 억원(/100)으로 바꿔 기존 kr_retail_deposit_misu.csv 와 맞춘다. 비중은 그대로.
비중 = 당일 반대매매 / 전일 미수금 ×100 (2025-06~2026-10 상관 0.9998).

왜 다시 받나: 기존 kr_retail_deposit_misu.csv 는 2025-06-02 시작이라 2025-10 시점 250일 백분위 이력이 80일뿐.
2024-06-01 부터 받아 2025-10 평가 구간 첫날부터 250일 창(최소 120일)을 채운다.
같이 받는 것: STATSCU0100000020(유가증권시장) 의 KOSPI 지수 — 한국 거래일 달력·급락 판정용.

공표 시차 확인용으로 받은 시각과 마지막 행 날짜를 kr_forced_sale_fetch_meta.json 에 남긴다.

실행: py fetch_kr_forced_sale.py [--start 20240601] [--end 20261014]
출력: ../results/kr_forced_sale.csv (date, misu_balance, forced_sale_amount, forced_sale_ratio_pct, kospi_index)
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


def fetch_table(service_id: str, start: str, end: str, unit: str = "1000000", retries: int = 3) -> list[dict]:
    body = {"dmSearch": {"tmpV40": unit, "tmpV41": "1", "tmpV1": "D",
                         "tmpV45": start, "tmpV46": end, "OBJ_NM": service_id + "BO"}}
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(BASE + "/meta/getMetaDataList.do", json=body, headers=HEADERS, timeout=120)
            j = r.json()
            return j.get("ds1", [])
        except Exception as ex:  # noqa: BLE001
            last_err = ex
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{service_id}: {last_err}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20240601")
    ap.add_argument("--end", default="20261014")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    fetched_at = pd.Timestamp.now().isoformat(timespec="seconds")

    rows = fetch_table("STATSCU0100000060", args.start, args.end)
    fs = pd.DataFrame(rows).rename(columns={
        "TMPV1": "date", "TMPV5": "misu_balance", "TMPV6": "forced_sale_amount", "TMPV7": "forced_sale_ratio_pct"})
    fs = fs[["date", "misu_balance", "forced_sale_amount", "forced_sale_ratio_pct"]].copy()
    fs["date"] = pd.to_datetime(fs["date"], format="%Y%m%d")
    for c in ["misu_balance", "forced_sale_amount", "forced_sale_ratio_pct"]:
        fs[c] = pd.to_numeric(fs[c], errors="coerce")
    fs["misu_balance"] = fs["misu_balance"] / 100.0        # 백만원 -> 억원
    fs["forced_sale_amount"] = fs["forced_sale_amount"] / 100.0
    time.sleep(0.5)

    rows = fetch_table("STATSCU0100000020", args.start, args.end)
    ks = pd.DataFrame(rows).rename(columns={"TMPV1": "date", "TMPV2": "kospi_index"})[["date", "kospi_index"]].copy()
    ks["date"] = pd.to_datetime(ks["date"], format="%Y%m%d")
    ks["kospi_index"] = pd.to_numeric(ks["kospi_index"], errors="coerce")

    df = fs.merge(ks, on="date", how="outer").sort_values("date").reset_index(drop=True)
    out = RESULTS / "kr_forced_sale.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    # 기존 CSV 와 겹치는 구간 대조
    old_path = RESULTS / "kr_retail_deposit_misu.csv"
    check = {}
    if old_path.exists():
        old = pd.read_csv(old_path, parse_dates=["date"])
        m = old.merge(df, on="date", suffixes=("_old", "_new"))
        diff = (m["forced_sale_amount_old"] - m["forced_sale_amount_new"]).abs().max()
        diff2 = (m["forced_sale_ratio_pct_old"] - m["forced_sale_ratio_pct_new"]).abs().max()
        check = {"overlap_rows": int(len(m)), "max_abs_diff_amount_eok": float(diff), "max_abs_diff_ratio_pct": float(diff2)}

    last_fs = fs["date"].max().date().isoformat()
    meta = {
        "source": BASE + " STATSCU0100000060BO (TMPV5/6/7) + STATSCU0100000020BO (TMPV2)",
        "fetched_at": fetched_at, "start": args.start, "end": args.end,
        "rows": int(len(df)), "forced_sale_rows": int(fs["forced_sale_amount"].notna().sum()),
        "first_date": df["date"].min().date().isoformat(), "last_forced_sale_date": last_fs,
        "unit": "금액 억원(백만원/100), 비중 %, 지수 포인트",
        "overlap_check_vs_kr_retail_deposit_misu": check,
    }
    (RESULTS / "kr_forced_sale_fetch_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(df.tail(8).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
