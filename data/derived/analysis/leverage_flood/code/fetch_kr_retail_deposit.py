"""투자자예탁금·신용융자 긴 이력 수집기 (갈래 kr_retail, 지표 #2 신용/예탁금 비율).

왜 따로 받나: fetch_kr_retail.py 의 kr_retail_deposit_misu.csv 는 2025-06-02 부터라
250거래일 백분위를 2025-10 부터 온전히 재려면 2024-06 이전 값이 더 필요하다.
freesis 는 2024-01-02 부터 같은 열로 내려준다(2026-10-14 확인, 676행).

출처: 금융투자협회 freesis  POST https://freesis.kofia.or.kr/meta/getMetaDataList.do
  STATSCU0100000060BO 증시자금추이 — TMPV1 일자, TMPV2 투자자예탁금(장내파생 예수금 제외),
      TMPV3 장내파생상품 거래예수금, TMPV4 대고객 RP 매도잔고, TMPV5 위탁매매 미수금,
      TMPV6 반대매매금액, TMPV7 반대매매비중(%)
  STATSCU0100000070BO 신용공여 잔고 추이 — TMPV2 신용거래융자 전체(유가+코스닥)
  단위: tmpV40=1000000(백만원)으로 받아 /100 → 억원.
공표 시차: 2026-10-14 14:04 조회 때 마지막 행이 10/12(월). 10/13 행은 아직 없음.
  10/12 값은 10/13(화) 중 보도됨 → T일 값은 T+1 영업일 오후·저녁에 보인다(영업일 시차 1).
가격: yfinance ^KS11(코스피)·^SOX(필라델피아 반도체) 종가 — 급락 판정용(사후값).

실행: py fetch_kr_retail_deposit.py [--start 20240101] [--end 20261014]
출력: ../results/kr_retail_deposit_ratio.csv  (date, investor_deposit, derivatives_deposit,
        customer_rp_balance, misu_balance, credit_loan_total, credit_to_deposit)
      ../results/kr_retail_deposit_prices.csv (date, kospi_close, sox_close — 각자 거래일, outer join)
      ../results/kr_retail_deposit_fetch_meta.json
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

DEPOSIT_COLS = {
    "TMPV2": "investor_deposit",
    "TMPV3": "derivatives_deposit",
    "TMPV4": "customer_rp_balance",
    "TMPV5": "misu_balance",
    "TMPV6": "forced_sale_amount",
    "TMPV7": "forced_sale_ratio_pct",
}
CREDIT_COLS = {"TMPV2": "credit_loan_total"}
NO_CONVERT = {"forced_sale_ratio_pct"}


def fetch_table(service_id: str, start: str, end: str, unit: str = "1000000", retries: int = 3) -> list[dict]:
    body = {"dmSearch": {"tmpV40": unit, "tmpV41": "1", "tmpV1": "D",
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


def to_frame(rows: list[dict], colmap: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("빈 응답")
    df = df.rename(columns={"TMPV1": "date", **colmap})
    keep = ["date"] + [c for c in colmap.values() if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    for c in keep[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if c not in NO_CONVERT:
            df[c] = df[c] / 100.0  # 백만원 -> 억원
    return df


def fetch_prices(start: str) -> pd.DataFrame:
    import yfinance as yf

    out = None
    for tk, col in [("^KS11", "kospi_close"), ("^SOX", "sox_close")]:
        h = yf.Ticker(tk).history(start=pd.Timestamp(start).strftime("%Y-%m-%d"), auto_adjust=False)
        if h.empty:
            raise RuntimeError(f"yfinance {tk} 빈 응답")
        s = h["Close"].copy()
        s.index = pd.to_datetime(s.index.tz_localize(None).date)
        s = s.rename(col).rename_axis("date").reset_index()
        out = s if out is None else out.merge(s, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20240101")
    ap.add_argument("--end", default="20261014")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)

    dep = to_frame(fetch_table("STATSCU0100000060", args.start, args.end), DEPOSIT_COLS)
    time.sleep(0.5)
    cre = to_frame(fetch_table("STATSCU0100000070", args.start, args.end), CREDIT_COLS)
    df = dep.merge(cre, on="date", how="outer").sort_values("date").reset_index(drop=True)
    df["credit_to_deposit"] = df["credit_loan_total"] / df["investor_deposit"]
    out = RESULTS / "kr_retail_deposit_ratio.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"deposit/credit: {len(df)} rows {df['date'].min().date()}~{df['date'].max().date()} -> {out}")

    px = fetch_prices(args.start)
    out_px = RESULTS / "kr_retail_deposit_prices.csv"
    px.to_csv(out_px, index=False, encoding="utf-8-sig")
    print(f"prices: {len(px)} rows {px['date'].min().date()}~{px['date'].max().date()} -> {out_px}")

    meta = {
        "source": BASE + " STATSCU0100000060BO(TMPV2~7), STATSCU0100000070BO(TMPV2) + yfinance ^KS11 ^SOX",
        "fetched_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "start": args.start, "end": args.end,
        "rows_deposit": int(len(df)),
        "last_deposit_date": str(df["date"].max().date()),
        "last_kospi_date": str(px.dropna(subset=["kospi_close"])["date"].max().date()),
        "last_sox_date": str(px.dropna(subset=["sox_close"])["date"].max().date()),
        "unit": "금액 억원(백만원/100), 반대매매비중 %, credit_to_deposit 배",
        "publish_lag_business_days": 1,
        "publish_lag_note": "T일 값이 T+1 영업일 오후·저녁 공표(10/14 14:04 조회 시 10/13 행 없음, 10/12 값은 10/13 보도)",
    }
    (RESULTS / "kr_retail_deposit_fetch_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
