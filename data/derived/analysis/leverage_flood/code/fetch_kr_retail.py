"""한국 개인 레버리지 지표 수집기 — 금융투자협회 freesis(종합통계 포털).

출처: https://freesis.kofia.or.kr  (POST /meta/getMetaDataList.do)
요청 본문: {"dmSearch":{"tmpV40":"<단위 나눗수>","tmpV41":"1","tmpV1":"D",
            "tmpV45":"YYYYMMDD","tmpV46":"YYYYMMDD","OBJ_NM":"<서비스ID>BO"}}
  - tmpV40 은 표시 단위 나눗수(T2050 코드: 1=원, 1000000=백만원, 10000000=천만원, 100000000=억원).
    저장 단위(BASIC_UNIT T2050^06)는 백만원이라 1000000 으로 받으면 원본 정밀도 그대로다.
  - 열 이름은 /meta/getSrvData.do 의 dsGrid.HEADER_NM 으로 확인했다(2026-10-14).
  - tmpV1(D/W/M)은 무시되고 항상 일별이 온다(2026-10-14 확인).

서비스ID(OBJ_NM 은 뒤에 BO 를 붙인다. 목록은 https://freesis.kofia.or.kr/stats/siteMap.do 의
clickLink('<parentDivId>','<serviceId>','<menuId>') 에서 뽑았다, 2026-10-14):
  주식 > 증시동향   STATSCU0100000010 일자별 주요 증시 현황 / 0020 유가증권시장 / 0030 코스닥시장
  주식 > 증시자금   STATSCU0100000060 증시자금추이(투자자예탁금·파생예수금·RP·위탁매매미수금·반대매매)
  주식 > 신용공여현황 STATSCU0100000070 신용공여 잔고 추이(신용융자·신용대주·청약자금대출·예탁증권담보융자)
                   STATSCU0100000080 신용거래 체결주수 추이(잔고 주수. 하루 13억 주라 일별 신규 체결이 아니라 잔고다)
  주식 > 대차거래   STATSCU0100000130 대차거래내역 / 0140 대차거래추이 / 0150 참여자별 / 0180 상위 10종목
  파생 > CFD       STATDRV0100000320 증거금포함 CFD명목잔고,  STATDRV0100000310 증거금차감 CFD명목잔고
  ※ 신용융자 '신규·상환' 일별 흐름은 freesis 메뉴에 없다(잔고 일별 증감 = 신규−상환 순액으로만 본다).
  ※ 반대매매비중(TMPV7) = 당일 반대매매금액 / 전일 미수금 ×100 (2025-06~2026-10 상관 0.9998, 당일 미수금 기준은 0.981).

확인(공표값 대조):
  2026-10-12 행 TMPV2 34,288,699백만원 = 34조2886억, 반대매매 27,083백만원 = 270억, 비중 2.7% — 10/13 보도와 일치.
  2026-06-24 반대매매 1,107억·7.5% — 6/25 16:37 보도와 일치. 6/15 는 raw 1,008(단위 1천만원)=100.8억인데
  일부 기사가 1,008억으로 적었다(단위 착오). 숫자는 freesis 원본을 따른다.
공표 시차: 10/14 13:25 조회 시 10/13 행이 아직 없고 10/12 행은 10/13 중 보도됐다 → T+1 오후(13:30 이후) 공표.
  CFD 도 금투협 안내대로 '매 영업일 장종료 후 전일 기준'(T+1).

실행: py fetch_kr_retail.py [--start 20250601] [--end 20261013]
출력: ../results/kr_retail_*.csv (금액 단위 억원, 지수는 포인트, 주수는 주)
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

# 서비스ID -> (파일 접미사, {TMPVn: 열 이름}, 금액열인지)
TABLES = {
    "STATSCU0100000070": (
        "margin_balance",
        {
            "TMPV2": "credit_loan_total",      # 신용거래융자 전체
            "TMPV3": "credit_loan_kospi",      # 신용거래융자 유가증권
            "TMPV4": "credit_loan_kosdaq",     # 신용거래융자 코스닥
            "TMPV5": "stock_lending_total",    # 신용거래대주 전체
            "TMPV6": "stock_lending_kospi",
            "TMPV7": "stock_lending_kosdaq",
            "TMPV8": "ipo_subscription_loan",  # 청약자금 대출
            "TMPV9": "securities_collateral_loan",  # 예탁증권담보융자
        },
        True,
    ),
    "STATSCU0100000060": (
        "deposit_misu",
        {
            "TMPV2": "investor_deposit",       # 투자자예탁금(장내파생상품 거래예수금 제외)
            "TMPV3": "derivatives_deposit",    # 장내파생상품 거래예수금
            "TMPV4": "customer_rp_balance",    # 대고객 RP 매도잔고
            "TMPV5": "misu_balance",           # 위탁매매 미수금
            "TMPV6": "forced_sale_amount",     # 위탁매매 미수금 대비 실제 반대매매금액
            "TMPV7": "forced_sale_ratio_pct",  # 미수금 대비 반대매매비중(%) — 비율, 단위 변환 안 함
        },
        True,
    ),
    "STATSCU0100000080": (
        "credit_shares",
        {
            "TMPV2": "credit_loan_shares_total",
            "TMPV3": "credit_loan_shares_kospi",
            "TMPV4": "credit_loan_shares_kosdaq",
            "TMPV5": "stock_lending_shares_total",
            "TMPV6": "stock_lending_shares_kospi",
            "TMPV7": "stock_lending_shares_kosdaq",
        },
        False,  # 단위 주
    ),
    "STATSCU0100000020": (
        "kospi",
        {
            "TMPV2": "kospi_index",
            "TMPV3": "kospi_volume_shares",
            "TMPV4": "kospi_value",            # 거래대금
            "TMPV5": "kospi_mktcap",           # 시가총액
            "TMPV6": "kospi_foreign_mktcap",
            "TMPV7": "kospi_foreign_pct",
        },
        True,
    ),
    "STATSCU0100000030": (
        "kosdaq",
        {
            "TMPV2": "kosdaq_index",
            "TMPV3": "kosdaq_volume_shares",
            "TMPV4": "kosdaq_value",
            "TMPV5": "kosdaq_mktcap",
            "TMPV6": "kosdaq_foreign_mktcap",
            "TMPV7": "kosdaq_foreign_pct",
        },
        True,
    ),
    "STATDRV0100000320": (
        "cfd_gross",
        {
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
        },
        True,
    ),
    "STATDRV0100000310": (
        "cfd_net_of_margin",
        {
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
        },
        True,
    ),
}

# 금액열이지만 단위 변환하지 않는 열(비율·지수·주수)
NO_CONVERT = {
    "forced_sale_ratio_pct", "kospi_index", "kospi_volume_shares", "kospi_foreign_pct",
    "kosdaq_index", "kosdaq_volume_shares", "kosdaq_foreign_pct",
}


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


def to_frame(rows: list[dict], colmap: dict[str, str], money: bool) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={"TMPV1": "date", **colmap})
    keep = ["date"] + [c for c in colmap.values() if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    for c in keep[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if money and c not in NO_CONVERT:
            df[c] = df[c] / 100.0  # 백만원 -> 억원
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20250601")
    ap.add_argument("--end", default="20261013")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    for sid, (suffix, colmap, money) in TABLES.items():
        unit = "1000000" if money else "1"  # 주수는 원 단위 그대로
        rows = fetch_table(sid, args.start, args.end, unit=unit)
        df = to_frame(rows, colmap, money)
        out = RESULTS / f"kr_retail_{suffix}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        frames[suffix] = df
        last = df["date"].max().date() if not df.empty else None
        print(f"{sid} {suffix}: {len(df)} rows, last={last} -> {out}")
        time.sleep(0.5)

    # 합본: 날짜 기준 outer join
    merged = None
    for suffix in ["margin_balance", "deposit_misu", "credit_shares", "kospi", "kosdaq"]:
        df = frames[suffix]
        merged = df if merged is None else merged.merge(df, on="date", how="outer")
    cfd = frames["cfd_gross"][["date", "cfd_total", "cfd_buy_dom_kospi", "cfd_buy_dom_kosdaq", "cfd_buy_foreign_stock"]]
    cfd = cfd.rename(columns={
        "cfd_total": "cfd_gross_total", "cfd_buy_dom_kospi": "cfd_gross_buy_dom_kospi",
        "cfd_buy_dom_kosdaq": "cfd_gross_buy_dom_kosdaq", "cfd_buy_foreign_stock": "cfd_gross_buy_foreign_stock"})
    merged = merged.merge(cfd, on="date", how="outer")
    cfd2 = frames["cfd_net_of_margin"][["date", "cfd_total"]].rename(columns={"cfd_total": "cfd_net_of_margin_total"})
    merged = merged.merge(cfd2, on="date", how="outer").sort_values("date").reset_index(drop=True)
    out = RESULTS / "kr_retail_daily.csv"
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"merged: {len(merged)} rows -> {out}")
    meta = {
        "source": BASE,
        "fetched_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "start": args.start, "end": args.end,
        "unit": "금액 억원(백만원/100), 비율 %, 지수 포인트, 주수 주",
        "tables": {sid: v[0] for sid, v in TABLES.items()},
    }
    (RESULTS / "kr_retail_fetch_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
