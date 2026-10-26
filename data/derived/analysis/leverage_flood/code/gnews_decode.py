# -*- coding: utf-8 -*-
"""Google News RSS 링크(news.google.com/rss/articles/<base64>)를 실제 기사 URL로 푼다.
1차: base64 안에 들어 있는 URL 추출(구형 인코딩). 2차: 실패하면 Google의 batchexecute 디코더(DotsSplashUi) 호출.
결과: results/gnews_hf_leverage_headlines.csv 에 resolved_url 열 추가
"""
import base64, csv, json, re, sys, time
from pathlib import Path
import requests

RESULTS = Path(__file__).resolve().parent.parent / "results"
P = RESULTS / "gnews_hf_leverage_headlines.csv"
UA = {"User-Agent": "Mozilla/5.0"}


def decode_b64(link: str):
    m = re.search(r"/articles/([^?]+)", link)
    if not m:
        return ""
    s = m.group(1)
    s += "=" * (-len(s) % 4)
    try:
        raw = base64.urlsafe_b64decode(s)
    except Exception:
        return ""
    # 구형: 바이트열 안에 http URL이 그대로 들어 있음
    mm = re.search(rb"(https?://[^\x00-\x1f\x7f-\xff]+)", raw)
    return mm.group(1).decode("utf-8", "ignore") if mm else ""


def decode_batchexecute(link: str):
    """신형(AU_yqL...) 인코딩: Google batchexecute로 복호화."""
    m = re.search(r"/articles/([^?]+)", link)
    if not m:
        return ""
    art_id = m.group(1)
    try:
        r = requests.get(f"https://news.google.com/rss/articles/{art_id}", headers=UA, timeout=30)
        sig = re.search(r'data-n-a-sg="([^"]+)"', r.text)
        ts = re.search(r'data-n-a-ts="([^"]+)"', r.text)
        if not (sig and ts):
            return ""
        payload = [["Fbv4je", f'["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,null,null,null,null,null,0,1],"X","X",1,[1,1,1],1,1,null,0,0,null,0],"{art_id}",{ts.group(1)},"{sig.group(1)}"]', None, "generic"]]
        data = {"f.req": json.dumps([payload])}
        r2 = requests.post("https://news.google.com/_/DotsSplashUi/data/batchexecute", headers={**UA, "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"}, data=data, timeout=30)
        txt = r2.text
        mm = re.search(r'\\"(https?://[^\\"]+)\\"', txt) or re.search(r'"(https?://[^"]+)"', txt.split("garturlres")[1] if "garturlres" in txt else txt)
        return mm.group(1).replace("\u003d", "=").replace("\u0026", "&") if mm else ""
    except Exception as e:  # noqa: BLE001
        print("batchexecute ERR", e, file=sys.stderr)
        return ""


rows = list(csv.DictReader(P.open(encoding="utf-8-sig")))
only_recent = [r for r in rows if r["pubDate"] >= "2026-05-01"]
cache = {}
for r in rows:
    r.setdefault("resolved_url", "")
for r in only_recent:
    if r["link"] in cache:
        r["resolved_url"] = cache[r["link"]]
        continue
    u = decode_b64(r["link"])
    if not u:
        u = decode_batchexecute(r["link"])
        time.sleep(0.5)
    cache[r["link"]] = u
    r["resolved_url"] = u
with P.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["query", "pubDate", "title", "source", "link", "resolved_url"])
    w.writeheader(); w.writerows(rows)
ok = sum(1 for r in only_recent if r["resolved_url"])
print(f"resolved {ok}/{len(only_recent)}", file=sys.stderr)
seen = set()
for r in only_recent:
    if r["resolved_url"] and r["resolved_url"] not in seen:
        seen.add(r["resolved_url"])
        print(r["pubDate"], "|", r["source"], "|", r["title"][:70], "|", r["resolved_url"])
