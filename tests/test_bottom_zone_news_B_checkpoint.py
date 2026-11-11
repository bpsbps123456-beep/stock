"""Evidence audit for the preregistered Task B collection checkpoint.

These checks do not fetch news or prices. They protect the lossless search log,
access-barrier labels, and the immutable evidence snapshot used for handoff.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
NEWS = ROOT / "data/derived/analysis/bottom_zone_20261111/news"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class NewsBCheckpointTests(unittest.TestCase):
    def test_fixed_grid_has_every_house_quarter_template(self):
        protocol = read_json(NEWS / "protocol_B.json")
        raw = [read_json(p) for p in (NEWS / "raw/B1").glob("*.json")]
        expected = {(h, f"{y}Q{q}", t) for h in protocol["houses"]
                    for y in (2024, 2025, 2026) for q in range(1, 5)
                    for t in protocol["search_B1"]["templates"]}
        self.assertEqual({(r["house"], r["quarter"], r["template"]) for r in raw}, expected)
        self.assertEqual(len(raw), len(expected))
        self.assertTrue(all(not r.get("error") for r in raw))
        self.assertTrue(all(not r.get("truncated") for r in raw))
        self.assertTrue(all(r["end"] <= "2026-11-11" for r in raw))

    def test_each_returned_rss_result_and_failed_attempt_is_logged(self):
        log = read_csv(NEWS / "search_log_B1.csv")
        logged = Counter((r["request_id"], r["url"]) for r in log if r["engine"] == "Google News RSS")
        expected = Counter()
        for path in (NEWS / "raw/B1").glob("*.json"):
            raw = read_json(path)
            for index, attempt in enumerate(raw.get("prior_attempts", []) + [raw], 1):
                request_id = f"{attempt['id']}-attempt{index}"
                for result in attempt["results"]:
                    expected[(request_id, result["url"])] += 1
                if not attempt["results"]:
                    expected[(request_id, "")] += 1
                    rows = [r for r in log if r["request_id"] == request_id]
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["decision"], "search_error" if attempt.get("error") else "zero_results")
        self.assertEqual(logged, expected)
        for row in log:
            self.assertTrue((NEWS / row["raw_file"]).is_file())
            self.assertTrue(row["query"])
            self.assertTrue(row["decision"])
            self.assertTrue(row["reason"])

    def test_supplement_queries_are_counted_without_fake_attribution(self):
        log = read_csv(NEWS / "search_log_B1.csv")
        requests = {}
        for row in log:
            requests.setdefault(row["request_id"], row)
        web = [r for r in requests.values() if r["engine"] == "web.run"]
        raw_files = list((NEWS / "raw").glob("web_B1_*.json"))
        self.assertEqual(len(web), len(raw_files))
        self.assertEqual(sum(int(r["queries_in_request"]) for r in web),
                         sum(len(read_json(p)["queries"]) for p in raw_files))
        summary = read_json(NEWS / "results/collection_summary_B1.json")
        self.assertEqual(summary["search_requests"], sum(int(r["queries_in_request"]) for r in requests.values()))

    def test_calls_keep_unknown_clock_blank_and_series_not_final(self):
        calls = read_csv(NEWS / "calls_census.csv")
        series = read_csv(NEWS / "news_series_B.csv")
        self.assertEqual(len({r["id"] for r in calls}), len(calls))
        self.assertEqual({r["call_id"] for r in series}, {r["id"] for r in calls})
        self.assertEqual(len(series), len(calls))
        action_family = {"upgrade":"U", "downgrade":"DN", "target_raise":"TU", "target_cut":"TC"}
        by_id = {r["id"]: r for r in calls}
        for row in series:
            call = by_id[row["call_id"]]
            self.assertEqual(row["date"], call["published_date"])
            self.assertEqual(row["family"], action_family[call["action"]])
            self.assertEqual(row["status"], "provisional_partial_census")
        for call in calls:
            self.assertTrue("2024-01-01" <= call["published_date"] <= "2026-11-10")
            self.assertEqual(call["published_at"], "")
            self.assertEqual(call["time_precision"], "date")
            self.assertEqual(call["direction"], "+1" if call["action"] in ("upgrade", "target_raise") else "-1")

    def test_blocked_article_is_not_counted_as_read(self):
        audit = read_csv(NEWS / "article_access_B1.csv")
        opened = {p.name for p in (NEWS / "raw").glob("read_B1_*.json")}
        self.assertEqual({r["file"] for r in audit}, opened)
        calls = read_csv(NEWS / "calls_census.csv")
        blocked = [r for r in audit if r["access"] == "blocked_login_not_read"]
        self.assertTrue(blocked)
        for row in blocked:
            self.assertEqual(row["decision"], "title_only")
            self.assertFalse(any(c["url"] == row["url"] and c["verification"] != "제목·요약만" for c in calls))
        for row in audit:
            if row["decision"] == "duplicate_source_group":
                self.assertTrue(row["linked_calls"])
                self.assertFalse(any(c["url"] == row["url"] for c in calls))

    def test_unreviewed_queue_is_explicitly_pending(self):
        pending = read_csv(NEWS / "pending_review_B1.csv")
        log = read_csv(NEWS / "search_log_B1.csv")
        self.assertTrue(pending)
        self.assertTrue(all(r["decision"] == "pending_review" for r in pending))
        logged_pending = {r["candidate_id"] for r in log if r["decision"] == "pending_review"}
        self.assertEqual({r["candidate_id"] for r in pending}, logged_pending)
        summary = read_json(NEWS / "results/collection_summary_B1.json")
        self.assertEqual(summary["pending_candidates"], len(pending))
        self.assertFalse(summary["price_access"])
        self.assertIsNone(summary["H_U_2026"])
        self.assertIsNone(summary["H_D_2026"])

    def test_freeze_hashes_and_mutation_guards(self):
        freeze = read_json(NEWS / "freeze_B.json")
        self.assertFalse(freeze["price_access_before_freeze"])
        self.assertEqual(freeze["status"], "partial_B1_only_not_completed_census")
        for relative, expected in freeze["input_hashes"].items():
            data = (NEWS / relative).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected, relative)
        for command in (("assemble_B1.py", "assemble"), ("collect_B.py", "B1", "--inventory"),
                        ("collect_B.py", "B1")):
            result = subprocess.run([sys.executable, str(NEWS / "code" / command[0]), *command[1:]],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("rozen", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
