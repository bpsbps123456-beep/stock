"""실적 추세 연구 1단계 — SEC 원자료 받기(실적 보도자료 + XBRL 재무값).

실행: py data/derived/analysis/earnings_trend/code/fetch_sec.py [--only NVDA,AMD]

- 8-K 중 Item 2.02(실적 발표)만 고른다. 접수 시각(acceptanceDateTime)이 발표 시점의 정본이다.
  미 동부 16:00 이전 접수면 그날이 반응일, 이후면 다음 거래일이 반응일이다(backtest.py).
- 첨부 EX-99.*(보도자료·CFO 해설·주주서한)를 표 구조를 살린 글자로 바꿔 cache/pr/<티커>/에 둔다.
  에이전트는 이 파일만 읽는다(웹 검색 0회).
- XBRL companyfacts 원본은 cache/facts/<티커>.json. 분기값 정리는 fundamentals.py가 한다.
- SEC 요구대로 용도를 밝힌 User-Agent, 요청 간격 0.2초. www.sec.gov는 연락처 형식이 없으면 403이라
  예약 도메인(example.com, RFC 2606) 주소를 쓴다(개인 이메일 없음). 403·429면 바로 멈춘다.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
UA = {"User-Agent": "GBRC-research personal-noncommercial research@example.com", "Accept-Encoding": "gzip, deflate"}
START = "2024-07-01"  # 2025년 첫 발표의 '직전 분기 가이던스'까지 읽으려고 반년 앞에서 시작

# 연구 표본 32종목. 고른 기준(결과를 보기 전에 정함):
#  ① 미국 국내 제출 회사(10-Q·8-K·XBRL) ② AI 공정 단계마다 1~3개 + 비AI 5개
#  ③ 2025년 초 시총이 메가~소형까지 퍼지게 ④ 2024년 이전 상장(전년 동기 비교 가능)
# 분류 이름은 watchlist.AI_SUBSECTOR가 정본이다(여기서는 표본 목록만 둔다).
UNIVERSE = [
    "NVDA", "AMD", "AVGO", "MRVL", "INTC", "CRDO", "ALAB", "RMBS", "MU",
    "STX", "LRCX",
    "ANET", "CIEN", "LITE",
    "DELL",
    "VRT", "VST", "BE",
    "MSFT", "GOOGL", "META", "AMZN",
    "PLTR", "SNOW", "DDOG", "CRWD", "NOW",
    "LLY", "UNH", "NFLX", "UBER", "SOFI",
]


def broad_universe() -> list[str]:
    """기계 계산 요인만 넓혀 보는 확대 표본: watchlist.AI_UNIVERSE의 미국 상장(접미사 없는) 종목 − 핵심 32.
    보도자료는 읽지 않는다(가이던스·내러티브 없음). XBRL·8-K가 없는 해외 발행사는 자동으로 빠진다."""
    sys.path.insert(0, str(ROOT.parents[3]))
    import watchlist  # noqa: E402
    us = [s for subs in watchlist.AI_UNIVERSE.values() for syms in subs.values() for s in syms if "." not in s]
    return [s for s in dict.fromkeys(us) if s not in UNIVERSE]


def suffix(which: str) -> str:
    return "" if which == "core" else f"_{which}"


class Sec:
    def __init__(self, gap: float = 0.2):
        self.gap, self.last = gap, 0.0

    def get(self, url: str, timeout: int = 60) -> bytes:
        wait = self.gap - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                raise SystemExit(f"SEC가 HTTP {e.code}로 거절했다. 멈춘다(재시도 안 함): {url}")
            raise
        body = r.read()
        enc = (r.headers.get("Content-Encoding") or "").lower()
        if enc == "gzip":
            body = gzip.decompress(body)
        elif enc == "deflate":
            body = zlib.decompress(body)
        return body

    def json(self, url: str):
        return json.loads(self.get(url))


def html_to_text(html: bytes) -> str:
    """보도자료 HTML → 읽을 글자. 표는 한 행을 '칸 | 칸' 한 줄로 바꿔 숫자 열이 흩어지지 않게 한다."""
    soup = BeautifulSoup(html, "lxml")
    for bad in soup(["script", "style"]):
        bad.decompose()
    for table in soup.find_all("table"):
        lines = []
        for tr in table.find_all("tr"):
            cells = [re.sub(r"\s+", " ", td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c and c not in ("$", "%", ")")]
            if cells:
                # '(1,234' 과 ')'가 다른 칸이면 괄호 음수가 깨진다 → 여는 괄호만 남은 칸은 닫아 준다
                cells = [c + ")" if c.startswith("(") and not c.endswith(")") else c for c in cells]
                lines.append(" | ".join(cells))
        table.replace_with(soup.new_string("\n[TABLE]\n" + "\n".join(lines) + "\n[/TABLE]\n"))
    for br in soup.find_all(["br", "p", "div", "li", "h1", "h2", "h3", "h4"]):
        br.insert_after("\n")
    text = soup.get_text()
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def cik_map(sec: Sec) -> dict[str, int]:
    data = sec.json("https://www.sec.gov/files/company_tickers.json")
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def earnings_filings(sec: Sec, cik: int) -> tuple[dict, list[dict]]:
    sub = sec.json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    pages = [sub["filings"]["recent"]]
    if min(pages[0]["filingDate"]) > START:
        for f in sub["filings"].get("files", []):
            if f.get("filingTo", "9999") >= START:
                pages.append(sec.json(f"https://data.sec.gov/submissions/{f['name']}"))
    rows = []
    for page in pages:
        for i, form in enumerate(page["form"]):
            if form != "8-K" or "2.02" not in (page["items"][i] or ""):
                continue
            if page["filingDate"][i] < START:
                continue
            rows.append({
                "accession": page["accessionNumber"][i],
                "filing_date": page["filingDate"][i],
                "acceptance_utc": page["acceptanceDateTime"][i],
                "items": page["items"][i],
                "primary": page["primaryDocument"][i],
            })
    meta = {"name": sub["name"], "sic": sub.get("sic"), "sic_desc": sub.get("sicDescription"),
            "fiscal_year_end": sub.get("fiscalYearEnd")}
    rows = list({r["accession"]: r for r in rows}.values())
    return meta, sorted(rows, key=lambda r: r["acceptance_utc"])


def exhibits(sec: Sec, cik: int, accession: str) -> list[tuple[str, str]]:
    """[(문서종류 EX-99.1, 파일명)] — 접수 색인 페이지의 표에서 읽는다."""
    acc = accession.replace("-", "")
    html = sec.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{accession}-index.htm")
    soup = BeautifulSoup(html, "lxml")
    out = []
    for tr in soup.select("table.tableFile tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        kind = tds[3].get_text(strip=True).upper()
        link = tds[2].find("a")
        if kind.startswith("EX-99") and link:
            name = link.get_text(strip=True)
            if name.lower().endswith((".htm", ".html", ".txt")):
                out.append((kind, name))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--refresh", action="store_true", help="이미 받은 보도자료도 다시 받는다")
    ap.add_argument("--set", default="core", choices=["core", "broad"], help="broad = 확대 표본(보도자료 안 받음)")
    args = ap.parse_args()
    base = UNIVERSE if args.set == "core" else broad_universe()
    tickers = [t.strip().upper() for t in args.only.split(",") if t.strip()] or base

    sec = Sec()
    ciks = cik_map(sec)
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS / f"sec_events{suffix(args.set)}.csv"
    old = pd.read_csv(manifest_path, dtype=str) if manifest_path.exists() else pd.DataFrame()
    keep = old[~old["ticker"].isin(tickers)] if len(old) else old
    rows, metas = [], {}
    for t in tickers:
        cik = ciks.get(t)
        if cik is None:
            print(f"{t}: CIK 없음 — 건너뜀", file=sys.stderr)
            continue
        try:
            meta, filings = earnings_filings(sec, cik)
        except urllib.error.HTTPError as e:
            print(f"{t}: 제출 목록 HTTP {e.code} — 건너뜀", file=sys.stderr)
            continue
        if not filings:
            print(f"{t}: 실적 8-K 없음(해외 발행사 등) — 건너뜀", file=sys.stderr)
            continue
        metas[t] = {"cik": cik, **meta}
        outdir = CACHE / "pr" / t
        outdir.mkdir(parents=True, exist_ok=True)
        for f in filings:
            et = pd.Timestamp(f["acceptance_utc"]).tz_convert("America/New_York")
            files = []
            for kind, name in (exhibits(sec, cik, f["accession"]) if args.set == "core" else []):
                out = outdir / f"{f['filing_date']}_{f['accession']}_{kind.replace('EX-', 'ex')}.txt"
                if args.refresh or not out.exists():
                    raw = sec.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{f['accession'].replace('-', '')}/{name}")
                    out.write_text(html_to_text(raw), encoding="utf-8")
                files.append(out.relative_to(ROOT).as_posix())
            rows.append({"ticker": t, "cik": cik, "accession": f["accession"], "filing_date": f["filing_date"],
                         "acceptance_et": et.strftime("%Y-%m-%d %H:%M"), "items": f["items"],
                         "files": ";".join(files)})
        facts = CACHE / "facts" / f"{t}.json"
        facts.parent.mkdir(parents=True, exist_ok=True)
        if args.refresh or not facts.exists():
            try:
                facts.write_bytes(sec.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", timeout=180))
            except urllib.error.HTTPError as e:
                print(f"{t}: XBRL HTTP {e.code} — 건너뜀", file=sys.stderr)
        print(f"{t}: 실적 8-K {len(filings)}건", flush=True)

    new = pd.DataFrame(rows, dtype=str)
    out = pd.concat([keep, new], ignore_index=True).sort_values(["ticker", "acceptance_et"])
    out.to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)
    meta_path = RESULTS / f"companies{suffix(args.set)}.json"
    allmeta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    allmeta.update(metas)
    meta_path.write_text(json.dumps(allmeta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {manifest_path} ({len(out)}건)")


if __name__ == "__main__":
    main()
