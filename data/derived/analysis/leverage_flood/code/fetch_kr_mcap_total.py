"""시가총액(유가+코스닥 합산)·KOSPI 지수 — 수집·병합 (지표 #8, 갈래 kr_retail).

출처: 금융투자협회 freesis 주식 > 증시동향
  STATSCU0100000020BO 유가증권시장 / STATSCU0100000030BO 코스닥시장
  열(2026-10-14 HEADER_NM 확인): TMPV1 일자, TMPV2 지수, TMPV3 거래량(주), TMPV4 거래대금,
  TMPV5 시가총액, TMPV6 외국인 시가총액, TMPV7 외국인 비중(%)
  tmpV40=1000000 으로 받으면 금액은 백만원 원본 그대로. 저장은 억원(백만원/100)으로 통일.
  대조: 2026-10-13 유가 TMPV5 = 4,978,832,650 백만원 = 4,978.8조원(지수 6,035.35) — 2025-06-02 2,211.7조(지수 2,698.97)
  와 지수 배율(2.24배)·시총 배율(2.25배)이 맞음.

공표 시차(실측):
  - 2026-10-14 13:28 조회 시 10/13 행이 이미 있었음(같은 시각 신용공여 0070 표에는 10/13 행 없음).
  - 2026-10-14 14:04(장중) 조회 시 10/14 행 없음(마감 전이라 당연).
  - 지수는 장중 실시간, 종가 시총은 KRX 정보데이터시스템이 당일 마감 후 공개 → 시차 0 영업일로 둔다.
    freesis 표만 쓰면 늦어도 T+1 오전.

실행: py fetch_kr_mcap_total.py [--start 20250601] [--end 20261014] [--refetch]
  기본은 fetch_kr_retail.py 가 이미 저장한 ../results/kr_retail_kospi.csv·kr_retail_kosdaq.csv 를 읽어 병합한다.
  --refetch 를 주면 freesis 에서 다시 받는다. yfinance ^KS11·^SOX 는 항상 받는다(검증·급락 판정용).
출력: ../results/kr_mcap_total.csv     (date, kospi_index, kospi_mktcap, kosdaq_index, kosdaq_mktcap, mcap_total; 억원)
      ../results/kr_mcap_total_yf.csv  (date, ks11_close, sox_close; yfinance)
"""
from __future__ import annotations

import argparse
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

TABLES = {
    "STATSCU0100000020": ("kospi", {"TMPV2": "kospi_index", "TMPV5": "kospi_mktcap"}),
    "STATSCU0100000030": ("kosdaq", {"TMPV2": "kosdaq_index", "TMPV5": "kosdaq_mktcap"}),
}


def fetch_table(service_id: str, start: str, end: str, retries: int = 3) -> list[dict]:
    body = {"dmSearch": {"tmpV40": "1000000", "tmpV41": "1", "tmpV1": "D",
                         "tmpV45": start, "tmpV46": end, "OBJ_NM": service_id + "BO"}}
    last = None
    for i in range(retries):
        try:
            r = requests.post(BASE + "/meta/getMetaDataList.do", json=body, headers=HEADERS, timeout=120)
            return r.json().get("ds1", [])
        except Exception as ex:  # noqa: BLE001
            last = ex
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{service_id}: {last}")


def from_freesis(start: str, end: str) -> dict[str, pd.DataFrame]:
    out = {}
    for sid, (name, colmap) in TABLES.items():
        df = pd.DataFrame(fetch_table(sid, start, end))
        df = df.rename(columns={"TMPV1": "date", **colmap})[["date", *colmap.values()]]
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        for c in colmap.values():
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[f"{name}_mktcap"] = df[f"{name}_mktcap"] / 100.0  # 백만원 -> 억원
        out[name] = df.sort_values("date").reset_index(drop=True)
        print(f"freesis {sid} {name}: {len(df)} rows, last={df['date'].max().date()}")
        time.sleep(0.5)
    return out


def from_saved() -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("kospi", "kosdaq"):
        p = RESULTS / f"kr_retail_{name}.csv"
        df = pd.read_csv(p, parse_dates=["date"])[["date", f"{name}_index", f"{name}_mktcap"]]
        out[name] = df
        print(f"saved {p.name}: {len(df)} rows, last={df['date'].max().date()}")
    return out


def fetch_yf(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    s = pd.Timestamp(start).strftime("%Y-%m-%d")
    e = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(["^KS11", "^SOX"], start=s, end=e, auto_adjust=False, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    df = close.rename(columns={"^KS11": "ks11_close", "^SOX": "sox_close"}).reset_index()
    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df[["date", "ks11_close", "sox_close"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20250601")
    ap.add_argument("--end", default="20261014")
    ap.add_argument("--refetch", action="store_true")
    a = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)

    frames = from_freesis(a.start, a.end) if a.refetch else from_saved()
    m = frames["kospi"].merge(frames["kosdaq"], on="date", how="outer").sort_values("date").reset_index(drop=True)
    # KOFIA 는 12/31(휴장) 등 지수 없는 행을 넣기도 한다. 둘 다 있는 날만 합산한다.
    m = m.dropna(subset=["kospi_index", "kosdaq_index"]).reset_index(drop=True)
    m["mcap_total"] = m["kospi_mktcap"] + m["kosdaq_mktcap"]
    out = RESULTS / "kr_mcap_total.csv"
    m.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"merged: {len(m)} rows {m['date'].min().date()}~{m['date'].max().date()} -> {out}")

    y = fetch_yf(a.start, a.end)
    out2 = RESULTS / "kr_mcap_total_yf.csv"
    y.to_csv(out2, index=False, encoding="utf-8-sig")
    print(f"yfinance: {len(y)} rows, ks11 last={y.dropna(subset=['ks11_close'])['date'].max().date()}, "
          f"sox last={y.dropna(subset=['sox_close'])['date'].max().date()} -> {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
