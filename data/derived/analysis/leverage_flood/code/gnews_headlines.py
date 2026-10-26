# -*- coding: utf-8 -*-
"""Google News RSS로 헤지펀드 레버리지 관련 보도 제목·날짜를 모은다 (WebSearch 대신 보조 수단).
결과: results/gnews_hf_leverage_headlines.csv (query, pubDate, title, source, link)
"""
import csv, sys, time, re
from pathlib import Path
from urllib.parse import quote
import requests
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

RESULTS = Path(__file__).resolve().parent.parent / "results"
QUERIES = [
    ("en", "hedge funds leverage Goldman Sachs prime brokerage"),
    ("en", "hedge funds de-grossing Goldman"),
    ("en", "hedge funds margin calls"),
    ("en", "hedge funds Korea stocks Goldman prime"),
    ("en", "hedge funds net sellers Goldman prime book"),
    ("en", "hedge fund gross leverage record"),
    ("ko", "헤지펀드 레버리지"),
    ("ko", "헤지펀드 순매도 코스피"),
    ("ko", "헤지펀드 마진콜"),
    ("ko", "골드만삭스 프라임 헤지펀드"),
]
UA = {"User-Agent": "Mozilla/5.0"}
rows = []
for lang, q in QUERIES:
    if lang == "ko":
        url = f"https://news.google.com/rss/search?q={quote(q)}&hl=ko&gl=KR&ceid=KR:ko"
    else:
        url = f"https://news.google.com/rss/search?q={quote(q)}&hl=en-US&gl=US&ceid=US:en"
    try:
        r = requests.get(url, headers=UA, timeout=60)
        root = ET.fromstring(r.content)
        n = 0
        for it in root.iter("item"):
            title = it.findtext("title") or ""
            pub = it.findtext("pubDate") or ""
            link = it.findtext("link") or ""
            src = it.find("source")
            src = src.text if src is not None else ""
            try:
                d = parsedate_to_datetime(pub).strftime("%Y-%m-%d")
            except Exception:
                d = pub
            rows.append({"query": q, "pubDate": d, "title": title, "source": src, "link": link})
            n += 1
        print(q, r.status_code, n, file=sys.stderr)
    except Exception as e:
        print(q, "ERR", e, file=sys.stderr)
    time.sleep(1)
rows.sort(key=lambda r: r["pubDate"], reverse=True)
with (RESULTS / "gnews_hf_leverage_headlines.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["query", "pubDate", "title", "source", "link"])
    w.writeheader(); w.writerows(rows)
for r in rows:
    if r["pubDate"] >= "2026-05-01":
        print(r["pubDate"], "|", r["source"], "|", r["title"][:110])
