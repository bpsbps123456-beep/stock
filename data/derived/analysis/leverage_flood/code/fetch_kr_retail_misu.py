"""위탁매매 미수금(한국 개인 초단기 레버리지) 수집 — 금융투자협회 freesis 증시자금추이.

출처: https://freesis.kofia.or.kr  POST /meta/getMetaDataList.do
  {"dmSearch":{"tmpV40":"1000000","tmpV41":"1","tmpV1":"D","tmpV45":"YYYYMMDD","tmpV46":"YYYYMMDD",
               "OBJ_NM":"STATSCU0100000060BO"}}
  열(2026-10-14 getSrvData.do HEADER_NM 으로 확인, fetch_kr_retail.py 참조):
    TMPV1 일자, TMPV2 투자자예탁금, TMPV3 장내파생상품 거래예수금, TMPV4 대고객 RP 매도잔고,
    TMPV5 위탁매매 미수금, TMPV6 미수금 대비 실제 반대매매금액, TMPV7 미수금 대비 반대매매비중(%)
  단위: tmpV40=1000000 → 백만원 그대로. 저장 때 억원(/100)으로 바꾼다.

이미 있는 ../results/kr_retail_deposit_misu.csv(2025-06-02~) 는 그대로 두고, 250거래일 백분위 창을
채우기 위해 2024-06-01 부터 다시 받아 ../results/kr_retail_misu_history.csv 로 따로 저장한다.
두 파일의 겹치는 구간 값이 같은지 검사해 출력한다.

가격: yfinance ^KS11(코스피), ^SOX(필라델피아 반도체) 종가 → ../results/kr_retail_misu_prices.csv

실행: py fetch_kr_retail_misu.py [--start 20240601] [--end 20261014]
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
OBJ = "STATSCU0100000060BO"
COLMAP = {
    "TMPV2": "investor_deposit",
    "TMPV3": "derivatives_deposit",
    "TMPV4": "customer_rp_balance",
    "TMPV5": "misu_balance",
    "TMPV6": "forced_sale_amount",
    "TMPV7": "forced_sale_ratio_pct",
}
NO_CONVERT = {"forced_sale_ratio_pct"}


def fetch_kofia(start: str, end: str, retries: int = 3) -> list[dict]:
    body = {"dmSearch": {"tmpV40": "1000000", "tmpV41": "1", "tmpV1": "D",
                         "tmpV45": start, "tmpV46": end, "OBJ_NM": OBJ}}
    last = None
    for i in range(retries):
        try:
            r = requests.post(BASE + "/meta/getMetaDataList.do", json=body, headers=HEADERS, timeout=120)
            return r.json().get("ds1", [])
        except Exception as ex:  # noqa: BLE001
            last = ex
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"freesis {OBJ}: {last}")


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={"TMPV1": "date", **COLMAP})
    keep = ["date"] + [c for c in COLMAP.values() if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    for c in keep[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if c not in NO_CONVERT:
            df[c] = df[c] / 100.0  # 백만원 -> 억원
    return df


def fetch_prices(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    s = pd.Timestamp(start).strftime("%Y-%m-%d")
    e = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out = None
    for tk, name in [("^KS11", "kospi"), ("^SOX", "sox")]:
        h = yf.Ticker(tk).history(start=s, end=e, auto_adjust=False)
        if h.empty:
            raise RuntimeError(f"yfinance {tk}: empty")
        ser = h["Close"].copy()
        ser.index = pd.to_datetime(ser.index.tz_localize(None).date)
        ser.name = name
        out = ser.to_frame() if out is None else out.join(ser, how="outer")
    out.index.name = "date"
    return out.sort_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20240601")
    ap.add_argument("--end", default="20261014")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)

    t0 = pd.Timestamp.now()
    rows = fetch_kofia(args.start, args.end)
    df = to_frame(rows)
    out = RESULTS / "kr_retail_misu_history.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"freesis {OBJ}: {len(df)} rows {df['date'].min().date()}..{df['date'].max().date()} -> {out}")
    print(f"조회 시각 {t0.isoformat(timespec='seconds')}  마지막 행 {df['date'].max().date()}")

    # 기존 CSV 와 대조
    old_p = RESULTS / "kr_retail_deposit_misu.csv"
    if old_p.exists():
        old = pd.read_csv(old_p, parse_dates=["date"])
        m = old.merge(df, on="date", suffixes=("_old", "_new"))
        diff = (m["misu_balance_old"] - m["misu_balance_new"]).abs().max()
        print(f"기존 CSV 겹침 {len(m)}행, misu_balance 최대 차이 {diff}")

    px = fetch_prices(args.start, args.end)
    outp = RESULTS / "kr_retail_misu_prices.csv"
    px.to_csv(outp, encoding="utf-8-sig")
    print(f"yfinance: {len(px)} rows {px.index.min().date()}..{px.index.max().date()} -> {outp}")
    print(px.tail(8).to_string())

    meta = {
        "source": BASE + " " + OBJ,
        "fetched_at": t0.isoformat(timespec="seconds"),
        "start": args.start, "end": args.end,
        "rows": int(len(df)), "last_row": str(df["date"].max().date()),
        "unit": "금액 억원(백만원/100), 반대매매비중 %",
        "prices": "yfinance ^KS11, ^SOX Close",
    }
    (RESULTS / "kr_retail_misu_fetch_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
