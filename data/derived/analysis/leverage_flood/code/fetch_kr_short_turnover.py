"""공매도 거래대금(전 시장 합계)·대차잔고 일별 — 치킨스탁 공매도 브리핑 페이지 스크랩 (KRX 로그인 대체, 최근 날짜만 존재).

사용: py fetch_kr_short_turnover.py [YYYY-MM-DD 시작] [YYYY-MM-DD 끝]
출력: ../results/kr_short_turnover_chickstock.csv
페이지: https://www.chickstockfi.com/short/<YYYY-MM-DD>  (없는 날짜는 404 → 건너뜀)
본문 문장 "공매도 거래대금 19,191 억원 … 직전 20 관측일 평균 17,827 억원 … 전 시장 대차잔고 157.3 조원" 에서 숫자를 뽑는다.
출처 자체가 거래소·금투협 일별 통계를 재게시한 것이라 KRX 공표 시차(공매도 거래대금은 당일 장 마감 후, 대차잔고는 전 관측일)를 따른다.
"""
from __future__ import annotations

import csv
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}


def parse(html: str) -> dict | None:
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t)
    m = re.search(r"공매도 거래대금 ([\d,]+) 억원 (\d{4}\.\d{2}\.\d{2}) 기준 직전 (\d+) 관측일 평균 ([\d,]+) 억원", t)
    if not m:
        return None
    rec = {"short_turnover_100m_krw": int(m.group(1).replace(",", "")), "stat_date": m.group(2).replace(".", ""),
           "avg_window_days": int(m.group(3)), "short_turnover_avg_100m_krw": int(m.group(4).replace(",", ""))}
    m2 = re.search(r"전 시장 대차잔고 ([\d.]+) 조원 전 관측일 대비 ([+\-]?[\d.]+) 조원 (\d{4}\.\d{2}\.\d{2}) 관측 기준", t)
    if m2:
        rec.update({"lending_balance_tril_krw": float(m2.group(1)), "lending_balance_chg_tril_krw": float(m2.group(2)), "lending_balance_date": m2.group(3).replace(".", "")})
    m3 = re.search(r"공매도 거래대금 10억원 이상인 코스피·코스닥 종목 중 (\S+) 의 거래비중이 ([\d.]+)%로 가장 높았습니다", t)
    if m3:
        rec.update({"top_short_share_stock": m3.group(1), "top_short_share_pct": float(m3.group(2))})
    # 코스피 상위 5 종목 비중·대금
    tops = re.findall(r"(\d) (\S+) (\d{6}) ([\d.]+) % (\d+)", t)
    rec["kospi_top5"] = "; ".join(f"{n}:{s}({c}) {p}% {a}억" for n, s, c, p, a in tops[:5])
    return rec


def main():
    s = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2026, 6, 1)
    e = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2026, 10, 13)
    rows = []
    d = s
    while d <= e:
        if d.weekday() < 5:
            url = f"https://www.chickstockfi.com/short/{d.isoformat()}"
            try:
                r = requests.get(url, headers=UA, timeout=60)
            except Exception as ex:  # noqa: BLE001
                print(d, "error", ex, file=sys.stderr)
                d += timedelta(days=1)
                continue
            if r.status_code == 200:
                rec = parse(r.text)
                if rec:
                    rec["date"] = d.strftime("%Y%m%d")
                    rec["url"] = url
                    rows.append(rec)
                    print(d, rec["short_turnover_100m_krw"], rec.get("lending_balance_tril_krw"), file=sys.stderr)
                else:
                    print(d, "200 but no match", file=sys.stderr)
            else:
                print(d, r.status_code, file=sys.stderr)
            time.sleep(0.5)
        d += timedelta(days=1)
    if not rows:
        print("못 받음: 한 날짜도 파싱되지 않음", file=sys.stderr)
        return
    fields = ["date", "stat_date", "short_turnover_100m_krw", "avg_window_days", "short_turnover_avg_100m_krw", "lending_balance_tril_krw", "lending_balance_chg_tril_krw",
              "lending_balance_date", "top_short_share_stock", "top_short_share_pct", "kospi_top5", "url"]
    with (RESULTS / "kr_short_turnover_chickstock.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"saved {RESULTS / 'kr_short_turnover_chickstock.csv'} rows={len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
