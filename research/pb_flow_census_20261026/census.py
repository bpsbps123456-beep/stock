"""Price-free collection and audit utilities for this research folder."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dateutil import parser as date_parser

ROOT = Path(__file__).resolve().parent
START, END = date(2023, 1, 1), date(2026, 10, 23)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze():
    digest = hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
    target = ROOT / "freeze.json"
    if target.exists():
        assert read_json(target)["protocol_sha256"] == digest, "Frozen protocol changed"
    else:
        write_json(target, {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                            "protocol_sha256": digest, "coding_started": False,
                            "price_data_accessed": False,
                            "legacy_protocol_sha256": hashlib.sha256((ROOT.parent / "news_turning_points_20261025/protocol.json").read_bytes()).hexdigest()})
    print("protocol_sha256=" + digest)


def windows(month_step):
    current = START
    while current <= END:
        serial = current.year * 12 + current.month - 1 + month_step
        following = min(date(serial // 12, serial % 12 + 1, 1), END + timedelta(days=1))
        yield current, following
        current = following


def query_grid(bank):
    p = read_json(ROOT / "protocol.json")["search"]
    for kind, step in [("monthly", 1), ("quarterly", 3), ("yearly_supplement", 12)]:
        templates = p[kind][bank]
        if isinstance(templates, str):
            templates = [templates]
        for a, b in windows(step):
            for n, template in enumerate(templates):
                yield {"query_id": f"{bank}_{kind}_{a}_{n}", "bank": bank, "kind": kind,
                       "start": str(a), "end_exclusive": str(b),
                       "query": f"{template} after:{a - timedelta(days=1)} before:{b}",
                       "template": template}


def fetch_query(q):
    path = ROOT / "search_runs" / (q["query_id"] + ".json")
    if path.exists():
        return read_json(path)
    result = dict(q, fetched_at_utc=datetime.now(timezone.utc).isoformat(), results=[], error="")
    try:
        response = requests.get("https://news.google.com/rss/search", params={"q": q["query"], "hl": "en-US", "gl": "US", "ceid": "US:en"}, timeout=35)
        response.raise_for_status()
        result["request_url"] = response.url
        root = ET.fromstring(response.content)
        for i, item in enumerate(root.findall("./channel/item")):
            raw_date = item.findtext("pubDate", "")
            pub = parsedate_to_datetime(raw_date).astimezone(ZoneInfo("America/New_York")) if raw_date else None
            result["results"].append({"rank": i + 1, "title": item.findtext("title", ""),
                "published_at": pub.date().isoformat() if pub else "", "feed_timestamp": raw_date,
                "url": item.findtext("link", ""), "source": item.findtext("source", ""),
                "description": item.findtext("description", "")})
    except Exception as e:
        result["error"] = type(e).__name__ + ": " + str(e)
    write_json(path, result)
    return result


def collect(bank):
    freeze()
    (ROOT / "search_runs").mkdir(exist_ok=True)
    queue = list(query_grid(bank))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        runs = list(pool.map(fetch_query, queue))
    # Split saturated query windows once, then recursively if necessary.
    while True:
        extra = []
        for run in runs:
            if len(run["results"]) < 100 or run.get("split"):
                continue
            a, b = date.fromisoformat(run["start"]), date.fromisoformat(run["end_exclusive"])
            if (b - a).days <= 1:
                continue
            mid = a + timedelta(days=(b-a).days // 2)
            for n, (left, right) in enumerate([(a, mid), (mid, b)]):
                extra.append(dict(query_id=run["query_id"]+f"_split{n}", bank=bank, kind="split", start=str(left),
                    end_exclusive=str(right), template=run["template"],
                    query=f"{run['template']} after:{left - timedelta(days=1)} before:{right}"))
            run["split"] = True
        if not extra:
            break
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            runs.extend(pool.map(fetch_query, extra))
    print(json.dumps({"bank": bank, "queries": len(runs), "results": sum(len(r["results"]) for r in runs),
                      "errors": sum(bool(r["error"]) for r in runs)}))


def inventory():
    records = {}
    for p in sorted((ROOT / "search_runs").glob("*.json")):
        run = read_json(p)
        for r in run["results"]:
            key = hashlib.sha256((r["title"].lower() + "|" + r["published_at"]).encode()).hexdigest()[:12]
            if key not in records:
                records[key] = dict(r, candidate_id=key, queried_banks=[], query_ids=[])
            records[key]["queried_banks"].append(run["bank"])
            records[key]["query_ids"].append(run["query_id"])
    rows = sorted(records.values(), key=lambda x:(x["published_at"], x["title"]))
    write_json(ROOT / "candidates.json", rows)
    print("unique candidates=" + str(len(rows)))


def supplement(bank):
    templates = {
        "GS": ['Goldman "hedge funds" stocks', 'Goldman "hedge funds" buying selling'],
        "MS": ['"Morgan Stanley" "hedge funds" stocks'],
        "JPM": ['JPMorgan "hedge funds" stocks'],
        "BofA": ['"BofA" "clients" stocks', '"Bank of America" "client flow"'],
    }[bank]
    queue = []
    for a, b in windows(1):
        for n, template in enumerate(templates):
            queue.append(dict(query_id=f"{bank}_supplement_{a}_{n}", bank=bank,
                kind="monthly_supplement", start=str(a), end_exclusive=str(b), template=template,
                query=f"{template} after:{a - timedelta(days=1)} before:{b}"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        runs = list(pool.map(fetch_query, queue))
    print(json.dumps({"bank": bank, "supplement_queries": len(runs), "results": sum(len(r["results"]) for r in runs), "errors": sum(bool(r["error"]) for r in runs)}))


def web_blocks(text):
    pattern = re.compile(r"^(.+) \((https?://[^\n]+)\)\n", re.M)
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        body = text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)]
        yield {"title": m[1].strip(), "url": m[2], "evidence": body.strip()}


def evidence_date(r):
    """Only explicit publication dates; no relative age -> date interpolation."""
    body = r.get("evidence", "")
    month = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
    patterns = [r"Published(?: on)?\s+(\d{2}/\d{2}/\d{4})", rf"Published(?: on)?\s+({month}\s+\d{{1,2}},?\s+20\d{{2}})",
                rf"(?:^|; |\n)({month}\s+\d{{1,2}},?\s+20\d{{2}})(?:\s|\|)",
                r"Date: \w+ (\w+ \d+ \d+:\d+:\d+ 20\d{2}) UTC"]
    for pattern in patterns:
        m = re.search(pattern, body, re.M | re.I)
        if m:
            try:
                dayfirst = r["url"].startswith(("https://uk.", "https://au.", "https://in."))
                return date_parser.parse(m[1], dayfirst=dayfirst).date().isoformat(), "date_article"
            except (ValueError, OverflowError):
                pass
    m = re.search(r"/(20\d{2})[/-](\d{1,2})[/-](\d{1,2})(?:/|[-?])", r["url"])
    if m:
        try:
            return date(*map(int, m.groups())).isoformat(), "date_url"
        except ValueError:
            pass
    return "", "unknown"


def normalized_title(title):
    title = re.sub(r"^(?:HEDGE ?FLOW|Focus:|Exclusive\s*\|)\s*", "", title, flags=re.I)
    title = re.split(r"\s[-|]\s|\sBy (?:Reuters|Bloomberg)|\s- Articles", title, maxsplit=1)[0]
    return re.sub(r"[^a-z0-9]", "", title.lower())


def prepare():
    freeze()
    sources = {}
    appearances = []
    queries = []
    for path in sorted((ROOT / "web_runs").glob("*.json")) + [ROOT / "web_search_01.json"]:
        run = read_json(path)
        qid = run.get("id", path.stem)
        query = run.get("query", " || ".join(run.get("queries", [])))
        blocks = list(web_blocks(run["result"]))
        queries.append(dict(query_id=qid, bank=qid.split("_")[0] if qid.startswith(("GS_", "MS_", "JPM_", "BofA_")) else "GS", engine="WebSearch", query=query, result_count=len(blocks), error="", start="", end_exclusive=""))
        for i, r in enumerate(blocks):
            sid = hashlib.sha256(r["url"].encode()).hexdigest()[:12]
            d, precision = evidence_date(r)
            r.update(source_id=sid, published_at=d, time_precision=precision, source=r["url"].split("/")[2])
            if sid not in sources or len(r["evidence"]) > len(sources[sid]["evidence"]):
                sources[sid] = r
            appearances.append(dict(query_id=qid, source_id=sid, rank=i+1, title=r["title"], published_at=d, url=r["url"], source=r["source"]))
    # RSS links remain valid discovery URLs even when direct publisher URL is unavailable.
    title_map = {}
    for sid, r in sources.items():
        title_map.setdefault(normalized_title(r["title"]), []).append(sid)
    for path in sorted((ROOT / "search_runs").glob("*.json")):
        run = read_json(path)
        queries.append(dict(query_id=run["query_id"], bank=run["bank"], engine="Google News RSS", query=run["query"], result_count=len(run["results"]), error=run["error"], start=run["start"], end_exclusive=run["end_exclusive"]))
        for r in run["results"]:
            ids = title_map.get(normalized_title(r["title"]), [])
            matches = [sid for sid in ids if not sources[sid]["published_at"] or abs((date.fromisoformat(sources[sid]["published_at"]) - date.fromisoformat(r["published_at"])).days) <= 3]
            if len(matches) == 1:
                sid = matches[0]
                if not sources[sid]["published_at"]:
                    sources[sid]["published_at"] = r["published_at"]
                    sources[sid]["time_precision"] = "date_feed"
            else:
                sid = hashlib.sha256((r["title"] + "|" + r["published_at"]).encode()).hexdigest()[:12]
                sources.setdefault(sid, dict(source_id=sid, title=r["title"], url=r["url"], published_at=r["published_at"],
                    time_precision="date_feed", source=r["source"], evidence=r["title"]))
            appearances.append(dict(query_id=run["query_id"], source_id=sid, rank=r["rank"], title=r["title"],
                published_at=r["published_at"], url=r["url"], source=r["source"]))
    write_json(ROOT / "sources_work.json", sorted(sources.values(), key=lambda r:(r["published_at"], r["title"])))
    write_json(ROOT / "appearances_work.json", appearances)
    write_json(ROOT / "queries_work.json", queries)
    print(json.dumps({"queries":len(queries), "appearances":len(appearances), "sources":len(sources)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "collect", "supplement", "inventory", "prepare"])
    parser.add_argument("--bank", choices=["GS", "MS", "JPM", "BofA"], default="GS")
    args = parser.parse_args()
    {"freeze": freeze, "collect": lambda: collect(args.bank), "supplement": lambda: supplement(args.bank), "inventory": inventory, "prepare": prepare}[args.action]()
