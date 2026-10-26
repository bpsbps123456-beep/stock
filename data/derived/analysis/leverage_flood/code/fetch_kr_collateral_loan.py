"""예탁증권담보융자 잔고(증권사 담보대출) 일별 이력 수집 — 금융투자협회 freesis.

출처: https://freesis.kofia.or.kr  POST /meta/getMetaDataList.do
      OBJ_NM=STATSCU0100000070BO (신용공여 잔고 추이), TMPV9 = 예탁증권담보융자(저장 단위 백만원)
      열 이름은 fetch_kr_retail.py 가 /meta/getSrvData.do 의 dsGrid.HEADER_NM 으로 확인한 것을 따른다(2026-10-14).
      tmpV40=1000000 으로 받으면 백만원 원본 정밀도 그대로 온다.

왜 다시 받나: 기존 results/kr_retail_margin_balance.csv 는 2025-06-02 부터라 250거래일 백분위를 2026-06 이전에는
완전한 창으로 못 잰다. 2024-06-03 부터 받아 2025-10 이후 전 구간을 완전한 창으로 평가한다.
기존 CSV 와 겹치는 구간은 값이 같은지 대조해 meta 에 적는다.

가격: yfinance ^KS11(코스피), ^SOX(필라델피아 반도체) 종가. 급락(5일 내 −8%) 판정용.

실행: py fetch_kr_collateral_loan.py [--start 20240601] [--end 20261014]
출력: ../results/kr_collateral_loan.csv            (date, collateral_loan_100m_krw 억원)
      ../results/kr_collateral_loan_prices.csv     (date, kospi_close, sox_close)
      ../results/kr_collateral_loan_fetch_meta.json
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
OBJ = "STATSCU0100000070BO"
COL = "collateral_loan_100m_krw"


def fetch_rows(start: str, end: str, retries: int = 3) -> list[dict]:
    body = {"dmSearch": {"tmpV40": "1000000", "tmpV41": "1", "tmpV1": "D",
                         "tmpV45": start, "tmpV46": end, "OBJ_NM": OBJ}}
    last = None
    for i in range(retries):
        try:
            r = requests.post(BASE + "/meta/getMetaDataList.do", json=body, headers=HEADERS, timeout=120)
            r.raise_for_status()
            return r.json().get("ds1", [])
        except Exception as ex:  # noqa: BLE001
            last = ex
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"freesis {OBJ}: {last}")


def fetch_prices(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    s = pd.Timestamp(start).strftime("%Y-%m-%d")
    e = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out = None
    for tk, name in [("^KS11", "kospi_close"), ("^SOX", "sox_close")]:
        h = yf.Ticker(tk).history(start=s, end=e, auto_adjust=False)
        if h.empty:
            raise RuntimeError(f"yfinance {tk}: empty")
        c = h["Close"].copy()
        c.index = pd.to_datetime(c.index.tz_localize(None).date)
        c.name = name
        out = c.to_frame() if out is None else out.join(c, how="outer")
    out.index.name = "date"
    return out.sort_index().reset_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20240601")
    ap.add_argument("--end", default="20261014")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # freesis 는 한 요청에 2년 넘게도 주지만 안전하게 1년 단위로 나눠 받는다.
    rows: list[dict] = []
    s = pd.Timestamp(args.start)
    e = pd.Timestamp(args.end)
    cur = s
    while cur <= e:
        nxt = min(cur + pd.DateOffset(years=1) - pd.Timedelta(days=1), e)
        rows += fetch_rows(cur.strftime("%Y%m%d"), nxt.strftime("%Y%m%d"))
        cur = nxt + pd.Timedelta(days=1)
        time.sleep(0.5)
    df = pd.DataFrame(rows)
    if df.empty or "TMPV9" not in df.columns:
        raise RuntimeError("freesis 응답에 TMPV9 없음")
    df = df.rename(columns={"TMPV1": "date", "TMPV9": COL})[["date", COL]]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df[COL] = pd.to_numeric(df[COL], errors="coerce") / 100.0  # 백만원 -> 억원
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    out = RESULTS / "kr_collateral_loan.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{OBJ} TMPV9: {len(df)} rows {df['date'].min().date()}~{df['date'].max().date()} -> {out}")

    # 기존 CSV 와 대조
    cmp = {}
    old_path = RESULTS / "kr_retail_margin_balance.csv"
    if old_path.exists():
        old = pd.read_csv(old_path, parse_dates=["date"])[["date", "securities_collateral_loan"]]
        m = df.merge(old, on="date", how="inner")
        diff = (m[COL] - m["securities_collateral_loan"]).abs()
        cmp = {"overlap_rows": int(len(m)), "max_abs_diff_100m_krw": float(diff.max()) if len(m) else None,
               "rows_in_old_not_new": int(len(set(old["date"]) - set(df["date"]))),
               "rows_in_new_not_old_within_old_range": int(len([d for d in set(df["date"]) - set(old["date"])
                                                                  if old["date"].min() <= d <= old["date"].max()]))}
        print("compare with kr_retail_margin_balance.csv:", cmp)

    prices = fetch_prices(args.start, args.end)
    pout = RESULTS / "kr_collateral_loan_prices.csv"
    prices.to_csv(pout, index=False, encoding="utf-8-sig")
    print(f"prices: {len(prices)} rows {prices['date'].min().date()}~{prices['date'].max().date()} -> {pout}")

    meta = {
        "source": BASE + " " + OBJ + " TMPV9 (예탁증권담보융자, 백만원 -> 억원)",
        "fetched_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "start": args.start, "end": args.end,
        "rows": int(len(df)), "first": str(df["date"].min().date()), "last": str(df["date"].max().date()),
        "compare_with_existing_csv": cmp,
        "prices": {"source": "yfinance ^KS11, ^SOX Close", "rows": int(len(prices)),
                   "last": str(prices["date"].max().date())},
    }
    (RESULTS / "kr_collateral_loan_fetch_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
