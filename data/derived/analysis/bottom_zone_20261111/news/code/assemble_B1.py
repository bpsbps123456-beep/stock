"""Assemble reviewed evidence and explicit pending inventory; no price access.

This is intentionally an evidence assembler, not an automatic news classifier.
Pending rows cannot support an assertion of a complete institutional census.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from collect_B import ROOT, b1_queries, save_json, write_csv

CALL_FIELDS = ["id", "house", "strategist", "published_date", "published_at", "time_precision",
               "asset_scope", "action", "from_rating", "to_rating", "direction", "after_selloff",
               "rationale_tag", "quote", "url", "source", "verification", "source_group",
               "candidate_id", "coding_notes", "status"]
LOG_FIELDS = ["query_id", "request_id", "queries_in_request", "engine", "house", "quarter",
              "template", "query", "fetched_at", "rank", "candidate_id", "title", "published_date",
              "feed_timestamp", "publication_label", "raw_file", "url", "source", "source_home", "decision", "reason", "error"]


def digest(path):
    # The shared Windows checkout may apply autocrlf; hash text with LF line
    # endings so the evidence identity survives a fresh Git checkout.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_inventory():
    queries, appearances, sources = [], [], {}
    for path in sorted((ROOT / "raw" / "B1").glob("*.json")):
        current = json.loads(path.read_text(encoding="utf-8"))
        for attempt, query in enumerate(current.get("prior_attempts", []) + [current], 1):
            request_id = f"{query['id']}-attempt{attempt}"
            common = dict(query_id=query["id"], request_id=request_id, engine="Google News RSS",
                          queries_in_request=1, raw_file=path.relative_to(ROOT).as_posix(),
                          **{k: query.get(k, "") for k in
                          ("house", "quarter", "template", "query", "fetched_at", "error")})
            queries.append(common)
            for rank, result in enumerate(query["results"], 1):
                appearances.append(dict(common, rank=rank, publication_label=result.get("feed_timestamp", ""), **result))
                sources[result["candidate_id"]] = result
            if not query["results"]:
                appearances.append(dict(common, decision="search_error" if query.get("error") else "zero_results",
                                        reason=query.get("error", "not_found_not_proof_of_no_report")))
    # web.run returns a combined result stream for multi-query requests. Keep
    # that request grouping, rather than fabricate per-query result attribution.
    for path in sorted((ROOT / "raw").glob("web_B1_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        query_id = path.stem.replace("web_", "").replace("_", "-")
        common = dict(query_id=query_id, request_id=query_id, engine="web.run",
                      queries_in_request=len(data["queries"]), query=json.dumps(data["queries"], ensure_ascii=False),
                      template="supplement", fetched_at="2026-11-11", raw_file=path.relative_to(ROOT).as_posix())
        queries.append(common)
        result_text = data["result"]
        matches = list(re.finditer(r"(?m)^([^\n]+) \((https?://[^\s]+)\)\s*$", result_text))
        for rank, match in enumerate(matches, 1):
            url = match[2]
            candidate_id = hashlib.sha256(url.encode()).hexdigest()[:16]
            end = matches[rank].start() if rank < len(matches) else len(result_text)
            metadata = result_text[match.end():end]
            label = re.search(r"Published: ([^;\n]+)", metadata)
            result = dict(candidate_id=candidate_id, title=match[1], url=url,
                          published_date="", publication_label=label[1] if label else "",
                          source=url.split("/")[2])
            sources[candidate_id] = result
            appearances.append(dict(common, rank=rank, **result))
        if not matches:
            appearances.append(dict(common, decision="parse_requires_review", reason="raw_result_retained"))
    return queries, appearances, sources


def build_calls(sources):
    reviewed = json.loads((ROOT / "reviewed_calls_B1.json").read_text(encoding="utf-8"))
    calls = []
    for row in reviewed["title_reviewed"]:
        source = sources[row["candidate_id"]]
        record = {field: "" for field in CALL_FIELDS}
        record.update(row)
        record.update(published_date=source["published_date"], url=source["url"], source=source["source"],
                      quote="[요약] 제목에서 기관의 " + row["action"] + " 변경을 확인. 세부 근거는 본문 검토 전.")
        calls.append(record)
    for row in reviewed["body_reviewed"]:
        record = {field: "" for field in CALL_FIELDS}
        record.update({k: v for k, v in row.items() if k in CALL_FIELDS})
        read = json.loads((ROOT / "raw" / row["read_file"]).read_text(encoding="utf-8"))
        record.update(url=read["url"], source=read["url"].split("/")[2],
                      candidate_id=hashlib.sha256(read["url"].encode()).hexdigest()[:16])
        calls.append(record)
    for row in calls:
        row["published_at"] = ""
        row["time_precision"] = "date"
        row["direction"] = "+1" if row["action"] in ("upgrade", "target_raise") else "-1"
        row["source_group"] = "B1-" + hashlib.sha256(
            "|".join(row[k] for k in ("house", "published_date", "asset_scope", "action")).encode()).hexdigest()[:12]
        row["status"] = "provisional_partial_census"
    calls.sort(key=lambda row: (row["published_date"], row["house"], row["asset_scope"], row["action"]))
    for i, row in enumerate(calls, 1):
        row["id"] = f"BC{i:04d}"
    return calls


def assemble():
    if (ROOT / "freeze_B.json").exists():
        raise SystemExit("Frozen evidence exists. Create a new evidence version; do not overwrite this freeze/input set.")
    protocol = json.loads((ROOT / "protocol_B.json").read_text(encoding="utf-8"))
    queries, log, sources = read_inventory()
    calls = build_calls(sources)
    by_candidate = {}
    for row in calls:
        by_candidate.setdefault(row["candidate_id"], []).append(row["id"])
    access_audit = []
    duplicate_candidates = {}
    audit = json.loads((ROOT / "article_access_B1.json").read_text(encoding="utf-8"))
    for entry in audit["reads"]:
        read = json.loads((ROOT / "raw" / entry["file"]).read_text(encoding="utf-8"))
        candidate_id = hashlib.sha256(read["url"].encode()).hexdigest()[:16]
        linked = by_candidate.get(candidate_id, [])
        if entry.get("same_call_as"):
            same = json.loads((ROOT / "raw" / entry["same_call_as"]).read_text(encoding="utf-8"))
            linked = by_candidate.get(hashlib.sha256(same["url"].encode()).hexdigest()[:16], [])
            duplicate_candidates[candidate_id] = ",".join(linked)
        access_audit.append(dict(entry, url=read["url"], candidate_id=candidate_id,
                                 linked_calls=",".join(linked)))
    for row in log:
        candidate = row.get("candidate_id")
        if not candidate:
            continue
        if candidate in by_candidate:
            row.update(decision="adopt_provisional", reason="reviewed_call:" + ",".join(by_candidate[candidate]))
        elif candidate in duplicate_candidates:
            row.update(decision="exclude", reason="duplicate_source_group:" + duplicate_candidates[candidate])
        elif row.get("published_date") and not "2024-01-01" <= row["published_date"] <= "2026-11-10":
            row.update(decision="exclude", reason="outside_window")
        else:
            row.update(decision="pending_review", reason="unreviewed_not_an_exclusion")
    write_csv(ROOT / "calls_census.csv", calls, CALL_FIELDS)
    write_csv(ROOT / "search_log_B1.csv", log, LOG_FIELDS)
    write_csv(ROOT / "article_access_B1.csv", access_audit,
              ["file", "url", "candidate_id", "access", "decision", "same_call_as", "linked_calls", "notes"])
    # Retain all raw candidates (including likely false positives), rather than
    # keyword-excluding institutional changes a human has not checked.
    pending = [dict(row, decision="pending_review") for candidate, row in sources.items()
               if candidate not in by_candidate and candidate not in duplicate_candidates and
               (not row.get("published_date") or "2024-01-01" <= row["published_date"] <= "2026-11-10")]
    write_csv(ROOT / "pending_review_B1.csv", pending,
              ["candidate_id", "title", "published_date", "url", "source", "decision"])
    coverage = []
    for house in protocol["houses"]:
        for year in (2024, 2025, 2026):
            for quarter in range(1, 5):
                period = f"{year}Q{quarter}"
                completed = {q["query_id"] for q in queries if q.get("house") == house and q.get("quarter") == period and not q.get("error")}
                found = [r for r in calls if r["house"] == house and int(r["published_date"][:4]) == year and (int(r["published_date"][5:7])-1)//3+1 == quarter]
                coverage.append(dict(house=house, quarter=period, planned_queries=3,
                                     completed_queries=len(completed), reviewed_calls=len(found),
                                     status="search_complete_review_incomplete" if len(completed)==3 else "search_incomplete"))
    write_csv(ROOT / "coverage_B1.csv", coverage,
              ["house", "quarter", "planned_queries", "completed_queries", "reviewed_calls", "status"])
    families = dict(upgrade="U", downgrade="DN", target_raise="TU", target_cut="TC")
    series = [dict(date=r["published_date"], family=families[r["action"]], asset_scope=r["asset_scope"],
                   call_id=r["id"], source_group=r["source_group"], status=r["status"]) for r in calls]
    write_csv(ROOT / "news_series_B.csv", series,
              ["date", "family", "asset_scope", "call_id", "source_group", "status"])
    adopted_candidates = {r["candidate_id"] for r in calls}
    summary = dict(
        status="partial_collection_and_coding_no_price_analysis", fixed_grid=len(b1_queries(protocol)),
        fixed_query_requests=sum(q["engine"]=="Google News RSS" for q in queries),
        web_query_requests=sum(q["queries_in_request"] for q in queries if q["engine"]=="web.run"),
        search_requests=sum(q["queries_in_request"] for q in queries),
        result_appearances=sum(bool(r.get("url")) for r in log), unique_candidates=len(sources),
        adopted_calls=len(calls), adopted_source_candidates=len(adopted_candidates),
        pending_candidates=len(pending), by_house=dict(Counter(r["house"] for r in calls)),
        by_action=dict(Counter(r["action"] for r in calls)),
        by_year=dict(Counter(r["published_date"][:4] for r in calls)),
        by_verification=dict(Counter(r["verification"] for r in calls)),
        original_read_fraction=sum(r["verification"]=="원문 열람" for r in calls)/len(calls),
        body_read_fraction=sum(r["verification"] in ("원문 열람", "재보도") for r in calls)/len(calls),
        fixed_coverage_cells=len(coverage), fixed_coverage_complete=sum(r["completed_queries"]==3 for r in coverage),
        B2_status="not_started", price_access=False, H_U_2026=None, H_D_2026=None)
    save_json(ROOT / "results" / "collection_summary_B1.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def freeze():
    if (ROOT / "freeze_B.json").exists():
        raise SystemExit("Freeze exists; never overwrite.")
    tracked = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts
               and p.name not in ("freeze_B.json", "unittest_output.txt", "PROGRESS_B.md")]
    save_json(ROOT / "freeze_B.json", dict(
        status="partial_B1_only_not_completed_census", frozen_at=datetime.now(timezone.utc).isoformat(),
        price_access_before_freeze=False, B2_status="not_started_no_bottom_news_or_B2_search_log",
        hash_algorithm="SHA-256 of UTF-8 files with CRLF normalized to LF; BOM retained",
        reason="Resource-limited continuation checkpoint; all pending evidence is retained. No H-U/H-D outcomes calculated.",
        input_hashes={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(tracked)}))
    print("Partial B1 freeze written; COMMIT this before any later price access. Full review requires a new version/freeze.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["assemble", "freeze"])
    args = parser.parse_args()
    assemble() if args.command == "assemble" else freeze()
