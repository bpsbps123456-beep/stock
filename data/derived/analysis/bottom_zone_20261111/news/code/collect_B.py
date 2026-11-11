"""Resumable public-news search inventory. Never imports or reads price data."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def b1_queries(protocol):
    queries = []
    for house in protocol["houses"]:
        for year in (2024, 2025, 2026):
            for quarter in range(1, 5):
                start = f"{year}-{quarter * 3 - 2:02d}-01"
                end = f"{year + 1}-01-01" if quarter == 4 else f"{year}-{quarter * 3 + 1:02d}-01"
                end = min(end, "2026-11-11")
                for template, query in protocol["search_B1"]["templates"].items():
                    q = query.format(house=house, house_ko=protocol["houses"][house])
                    q += f" after:{start} before:{end}"
                    queries.append(dict(id=f"B1-{len(queries)+1:04d}", house=house,
                                        quarter=f"{year}Q{quarter}", template=template,
                                        start=start, end=end, query=q))
    return queries


def fetch(query, stage):
    path = ROOT / "raw" / stage / (query["id"] + ".json")
    prior_attempts = []
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if not cached.get("error"):
            return cached
        prior_attempts = cached.get("prior_attempts", []) + [{k: v for k, v in cached.items() if k != "prior_attempts"}]
    result = dict(query, fetched_at=datetime.now(timezone.utc).isoformat(), results=[])
    if prior_attempts:
        result["prior_attempts"] = prior_attempts
    try:
        ko = query["template"].startswith("ko")
        params = {"q": query["query"], "hl": "ko" if ko else "en-US",
                  "gl": "KR" if ko else "US", "ceid": "KR:ko" if ko else "US:en"}
        response = requests.get("https://news.google.com/rss/search", params=params, timeout=45)
        result["request_url"] = response.url
        response.raise_for_status()
        for item in ET.fromstring(response.content).findall("./channel/item"):
            rawdate = item.findtext("pubDate", "")
            d = parsedate_to_datetime(rawdate).date().isoformat() if rawdate else ""
            source = item.find("source")
            title, url = item.findtext("title", ""), item.findtext("link", "")
            result["results"].append(dict(
                candidate_id=hashlib.sha256(url.encode()).hexdigest()[:16], title=title,
                published_date=d, feed_timestamp=rawdate, url=url,
                source=source.text if source is not None else "",
                source_home=source.get("url", "") if source is not None else ""))
        result["truncated"] = len(result["results"]) >= 100
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    save_json(path, result)
    return result


def collect(stage, limit=None):
    if (ROOT / "freeze_B.json").exists():
        raise SystemExit("Evidence is frozen; start a new evidence version before further collection.")
    protocol = json.loads((ROOT / "protocol_B.json").read_text(encoding="utf-8"))
    queries = b1_queries(protocol)
    if limit:
        queries = queries[:limit]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch, q, stage): q for q in queries}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if i % 25 == 0 or result.get("error"):
                print(f"{stage} {i}/{len(queries)} {result['id']} n={len(result['results'])} error={result.get('error','')} elapsed={time.monotonic()-started:.0f}s", flush=True)
    inventory(stage)


def inventory(stage):
    if (ROOT / "freeze_B.json").exists():
        raise SystemExit("Evidence is frozen; do not overwrite its search log or candidate inventory.")
    rows, candidates = [], {}
    for path in sorted((ROOT / "raw" / stage).glob("*.json")):
        query = json.loads(path.read_text(encoding="utf-8"))
        common = {k: query.get(k, "") for k in ("id", "house", "quarter", "template", "query", "fetched_at", "error", "anchor_date", "anchor_type")}
        common["query_id"] = common.pop("id")
        for i, r in enumerate(query["results"]):
            row = dict(common, rank=i+1, **r, decision="pending_review", reason="not_yet_coded")
            rows.append(row)
            candidates.setdefault(r["candidate_id"], dict(r, appearances=[]))["appearances"].append(common["query_id"])
        if not query["results"]:
            rows.append(dict(common, decision="search_error" if query.get("error") else "zero_results", reason=query.get("error", "not_found_not_proof_of_no_report")))
    fields = ["query_id", "house", "quarter", "template", "anchor_date", "anchor_type", "query", "fetched_at", "error", "rank", "candidate_id", "title", "published_date", "feed_timestamp", "url", "source", "source_home", "decision", "reason"]
    write_csv(ROOT / f"search_log_{stage}.csv", rows, fields)
    save_json(ROOT / f"candidates_{stage}.json", list(candidates.values()))
    print(json.dumps(dict(stage=stage, queries=len({r["query_id"] for r in rows}), appearances=sum(bool(r.get("url")) for r in rows), candidates=len(candidates)), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["B1"])
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    inventory(args.stage) if args.inventory else collect(args.stage, args.limit)
