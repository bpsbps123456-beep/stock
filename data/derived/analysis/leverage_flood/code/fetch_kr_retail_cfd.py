"""CFD 명목잔고(증거금포함·증거금차감) 긴 이력 수집 — 금융투자협회 freesis.

출처: https://freesis.kofia.or.kr  POST /meta/getMetaDataList.do
  파생 > 차액결제거래(CFD) > STATDRV0100000320 증거금포함 CFD명목잔고 / STATDRV0100000310 증거금차감 CFD명목잔고
  TMPV1 일자, TMPV2~6 매수(국내 유가·코스닥·기타, 해외 주식·기타), TMPV7~11 매도(같은 순서), TMPV12 합계.
  저장 단위 백만원 → tmpV40=1000000 으로 받아 /100 해서 억원으로 저장한다(fetch_kr_retail.py 와 같은 규약).

왜 따로 받나: 기존 kr_retail_cfd_gross.csv 는 2025-06-02 부터라 "직전 250거래일 백분위"가 2025-10 시점에 85일밖에
  못 본다. 2024-06-01 부터 받아 두면 2025-10-01 이후 모든 날에 250일 창이 꽉 찬다. 겹치는 구간은 기존 파일과 값이
  같은지 검사해 출력한다(다르면 경고).

공표 시차(2026-10-14 직접 확인): 10/14 13:28·14:04 조회 시 CFD 두 표 모두 10/13 행까지 있고 10/14 행은 없음.
  같은 시각 신용공여(STATSCU0100000070)·증시자금(0060)은 10/12 행까지만 있었다. 즉 CFD 는 T 의 잔고를 T+1 낮
  (13:30 이전)에 볼 수 있다 → 시차 1영업일(T+1 종가·T+2 시가에 행동 가능).

실행: py fetch_kr_retail_cfd.py [--start 20240601] [--end 20261014]
출력: ../results/kr_retail_cfd_gross_long.csv, ../results/kr_retail_cfd_net_of_margin_long.csv,
      ../results/kr_retail_cfd_fetch_meta.json
"""
from __future__ import annotations

import argparse
import datetime as dt
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

COLMAP = {
    "TMPV2": "cfd_buy_dom_kospi",
    "TMPV3": "cfd_buy_dom_kosdaq",
    "TMPV4": "cfd_buy_dom_other",
    "TMPV5": "cfd_buy_foreign_stock",
    "TMPV6": "cfd_buy_foreign_other",
    "TMPV7": "cfd_sell_dom_kospi",
    "TMPV8": "cfd_sell_dom_kosdaq",
    "TMPV9": "cfd_sell_dom_other",
    "TMPV10": "cfd_sell_foreign_stock",
    "TMPV11": "cfd_sell_foreign_other",
    "TMPV12": "cfd_total",
}
TABLES = {
    "STATDRV0100000320": ("kr_retail_cfd_gross_long", "kr_retail_cfd_gross"),
    "STATDRV0100000310": ("kr_retail_cfd_net_of_margin_long", "kr_retail_cfd_net_of_margin"),
}


def fetch_table(service_id: str, start: str, end: str, retries: int = 3) -> list[dict]:
    body = {"dmSearch": {"tmpV40": "1000000", "tmpV41": "1", "tmpV1": "D",
                         "tmpV45": start, "tmpV46": end, "OBJ_NM": service_id + "BO"}}
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(BASE + "/meta/getMetaDataList.do", json=body, headers=HEADERS, timeout=120)
            return r.json().get("ds1", [])
        except Exception as ex:  # noqa: BLE001
            last_err = ex
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{service_id}: {last_err}")


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={"TMPV1": "date", **COLMAP})
    keep = ["date"] + [c for c in COLMAP.values() if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for c in keep[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0  # 백만원 -> 억원
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20240601")
    ap.add_argument("--end", default="20261014")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    meta = {"source": BASE + "/meta/getMetaDataList.do", "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
            "start": args.start, "end": args.end, "unit": "억원(백만원/100)", "tables": {}}
    for sid, (out_name, old_name) in TABLES.items():
        rows = fetch_table(sid, args.start, args.end)
        df = to_frame(rows)
        if df.empty:
            print(f"[{sid}] 못 받음: 행 0", file=sys.stderr)
            meta["tables"][sid] = {"rows": 0}
            continue
        path = RESULTS / f"{out_name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        info = {"file": path.name, "rows": int(len(df)), "first": str(df["date"].iloc[0].date()),
                "last": str(df["date"].iloc[-1].date())}
        # 기존 짧은 파일과 겹치는 구간 대조
        old_path = RESULTS / f"{old_name}.csv"
        if old_path.exists():
            old = pd.read_csv(old_path, parse_dates=["date"])
            m = old.merge(df, on="date", suffixes=("_old", "_new"))
            diff = (m["cfd_total_old"] - m["cfd_total_new"]).abs().max() if len(m) else None
            info["overlap_rows"] = int(len(m))
            info["overlap_max_abs_diff_cfd_total"] = None if diff is None else float(diff)
            if diff is not None and diff > 0.01:
                print(f"[{sid}] 경고: 기존 파일과 합계가 다름 max|diff|={diff}", file=sys.stderr)
        meta["tables"][sid] = info
        print(sid, info)
    (RESULTS / "kr_retail_cfd_fetch_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
