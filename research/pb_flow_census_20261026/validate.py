"""Check the assembled PB-flow census against protocol.json validation list.

Run after assemble.py:  py validate.py [--unittest]
Writes validation.json and exits non-zero when any check fails. Price-free: reads only
this folder, the git-excluded claude_work folder and the frozen legacy events/protocol.
"""
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LEGACY_DIR = REPO / "research/news_turning_points_20261025"
START, END = "2023-01-01", "2026-10-23"
COVER_START, COVER_END = date(2022, 12, 26), date(2026, 10, 19)
BANKS = {"GS", "MS", "JPM", "BofA"}
EXCLUDE_CODES = {"unrelated", "13f_or_manager_holdings", "fund_flows_not_bank_client", "bofa_fms_without_cft",
                 "forecast_not_observed", "other_bank", "insufficient_attribution", "junk_page", "out_of_window"}
DECISIONS = {"codex_coded", "coded_eligible", "coded_unverified", "coded_no", "screen_exclude",
             "query_zero_results", "query_error"}
CATEGORIES = {
    "bank": BANKS,
    "flow_direction": {"net_buy", "net_sell", "mixed", ""},
    "long_flow": {"buy", "sell", "na"},
    "short_flow": {"add", "cover", "na"},
    "family": {"D", "S", "R", "T", "N"},
    "verification": {"original_article_read", "syndicated_article_read", "headline_summary_only", "prior_verified_fact"},
    "time_precision": {"exact", "date_article", "date_feed", "date_url", "unknown"},
    "note_date_basis": {"explicit", "publication_proxy"},
}
REQUIRED = ["group_id", "id", "bank", "family", "url", "source", "verification", "source_group", "n_sources", "coder"]
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")

results = []


def check(name, ok, detail=""):
    results.append({"check": name, "ok": bool(ok), "detail": detail})


def read_csv(name):
    raw = (ROOT / name).read_bytes()
    check(f"{name}: UTF-8 without BOM", not raw.startswith(b"\xef\xbb\xbf"))
    return list(csv.DictReader(raw.decode("utf-8").splitlines()))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")


def verbatim_words(t):
    """Source-language words: none for a "[요약]" paraphrase, Hangul tokens excluded (same rule as assemble.py)."""
    t = (t or "").strip()
    if t.startswith("[요약]"):
        return 0
    return sum(1 for w in t.rstrip(" …").split() if not HANGUL.search(w))


def main():
    # 1. frozen protocol fingerprints
    freeze = load(ROOT / "freeze.json")
    sha = hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
    check("protocol.json SHA-256 unchanged", sha == freeze["protocol_sha256"], sha)
    lsha = hashlib.sha256((LEGACY_DIR / "protocol.json").read_bytes()).hexdigest()
    check("legacy protocol SHA-256 unchanged", lsha == freeze["legacy_protocol_sha256"], lsha)
    check("freeze.json records no price access", freeze.get("price_data_accessed") is False)

    census = read_csv("census.csv")
    gsrc = read_csv("group_sources.csv")
    unver = read_csv("census_unverified.csv")
    log = read_csv("search_log.csv")
    cross = read_csv("legacy_crosswalk.csv")
    cov = read_csv("coverage.csv")
    gids = {g["group_id"] for g in census}

    # 2. every query and every result has a decision and a reason
    queries = load(ROOT / "queries_work.json")
    appearances = load(ROOT / "appearances_work.json")
    candidates = load(ROOT / "candidates.json")
    bad = [r for r in log if r["decision"] not in DECISIONS or not r["reason"].strip()]
    check("search_log: every row has a valid decision and a reason", not bad, f"bad={len(bad)}")
    bad = [r for r in log if r["decision"] in ("screen_exclude", "coded_no") and r["exclusion_code"] not in EXCLUDE_CODES]
    check("search_log: exclusions use protocol exclusion codes", not bad, f"bad={len(bad)}")
    bad = [r for r in log if r["decision"] not in ("screen_exclude", "coded_no") and r["exclusion_code"]]
    check("search_log: exclusion code only on excluded rows", not bad, f"bad={len(bad)}")
    per_q = Counter(r["query_id"] for r in log if r["rank"] != "")
    app_q = Counter(a["query_id"] for a in appearances)
    check("search_log: every appearance logged once", per_q == app_q, f"log_results={sum(per_q.values())} appearances={len(appearances)}")
    zero = {r["query_id"] for r in log if r["decision"] == "query_zero_results"}
    missing_q = [q["query_id"] for q in queries if q["query_id"] not in app_q and q["query_id"] not in zero]
    check("search_log: every query has a row (zero-result/error included)", not missing_q and
          {r["query_id"] for r in log} == {q["query_id"] for q in queries}, f"queries={len(queries)} zero_rows={len(zero)}")
    errs = [q for q in queries if q["error"]]
    check("search_log: failed queries logged", all(any(r["query_id"] == q["query_id"] and r["decision"] == "query_error" for r in log) for q in errs),
          f"failed_queries={len(errs)}")
    logged_c = {r["candidate_id"] for r in log if r["candidate_id"]}
    miss_c = [c["candidate_id"] for c in candidates if c["candidate_id"] not in logged_c]
    check("search_log: every RSS candidate logged", not miss_c, f"candidates={len(candidates)} missing={len(miss_c)}")
    gid_bad = [r for r in log if r["group_ids"] and not set(r["group_ids"].split(";")) <= gids]
    check("search_log: group ids exist in census", not gid_bad, f"bad={len(gid_bad)}")

    # 3. legacy reconciliation
    events = load(LEGACY_DIR / "events.json")
    cmap = {r["legacy_id"]: r for r in cross}
    fam = {g["group_id"]: g["family"] for g in census}
    problems = []
    for e in events:
        r = cmap.get(e["id"])
        if not r:
            problems.append(f"{e['id']} missing")
            continue
        if r["legacy_family"] != e["family"] or r["legacy_include"] != str(e["include"]) or r["legacy_exclusion"] != e["exclusion"]:
            problems.append(f"{e['id']} original fields altered")
        if r["status"] == "linked":
            gl = r["group_ids"].split(";")
            if not gl or not set(gl) <= gids:
                problems.append(f"{e['id']} unknown group")
            cur = {fam[g] for g in gl if g in fam}
            if r["current_family"] != ";".join(sorted(cur)) or (r["family_changed"] == "yes") != (e["family"] not in cur):
                problems.append(f"{e['id']} family comparison inconsistent")
        elif r["status"] == "excluded":
            if not r["current_exclusion"].strip() or r["group_ids"]:
                problems.append(f"{e['id']} exclusion without reason")
        else:
            problems.append(f"{e['id']} bad status")
    check("legacy: all records reconciled (linked group or explicit exclusion)",
          not problems and len(cross) == len(events), f"events={len(events)} rows={len(cross)} problems={problems}")
    back = defaultdict(set)
    for g in census:
        for lid in filter(None, g["legacy_ids"].split(";")):
            back[lid].add(g["group_id"])
    check("legacy: census legacy_ids match crosswalk",
          all(back.get(r["legacy_id"], set()) == set(filter(None, r["group_ids"].split(";"))) for r in cross))

    # 4. one row per source_group
    check("census: group_id unique", len(gids) == len(census), f"rows={len(census)}")
    check("census: source_group equals group_id", all(g["source_group"] == g["group_id"] for g in census))
    members = defaultdict(list)
    for s in gsrc:
        members[s["group_id"]].append(s)
    bad = [g["group_id"] for g in census if len(members[g["group_id"]]) != int(g["n_sources"])
           or [s["source_id"] for s in members[g["group_id"]] if s["is_representative"] == "yes"] != [g["id"]]]
    check("census: n_sources and single representative match group_sources", not bad and set(members) == gids, f"bad={bad[:10]}")
    dup_pairs = Counter((s["group_id"], s["source_id"]) for s in gsrc)
    check("group_sources: no repeated group-source pair", max(dup_pairs.values()) == 1)
    rep_first = []
    for g in census:
        dates = [s["published_at"][:10] for s in members[g["group_id"]] if s["published_at"]]
        if dates and g["published_at"][:10] != min(dates):
            rep_first.append(g["group_id"])
    check("census: representative is earliest dated public source", not rep_first, f"bad={rep_first[:10]}")
    unv_ids = {r["id"] for r in unver}
    census_src = {s["source_id"] for s in gsrc}
    check("unverified rows kept out of census", not (unv_ids & census_src), f"overlap={sorted(unv_ids & census_src)[:10]}")

    # 5. publication window
    out = [g["group_id"] for g in census if g["published_at"] and not (START <= g["published_at"][:10] <= END)]
    check("census: no out-of-window publication", not out, f"bad={out}")
    out_u = [r["id"] for r in unver if r["published_at"] and not (START <= r["published_at"][:10] <= END)]
    check("census_unverified: no out-of-window publication", not out_u, f"bad={out_u}")
    out_s = [s["source_id"] for s in gsrc if s["published_at"] and not (START <= s["published_at"][:10] <= END)]
    check("group_sources: no out-of-window publication", not out_s, f"bad={out_s}")

    # 6. categorical values and explicit missingness
    for rows, label in ((census, "census"), (unver, "census_unverified")):
        for col, allowed in CATEGORIES.items():
            if label == "census_unverified" and col == "bank":
                allowed = BANKS | {""}
            if label == "census_unverified" and col == "note_date_basis":
                allowed = allowed | {""}
            vals = Counter(r[col] for r in rows if r[col] not in allowed)
            check(f"{label}: valid {col}", not vals, f"invalid={dict(vals)}")
    miss = {c: sum(not g[c] for g in census) for c in REQUIRED}
    check("census: required fields present", not any(miss.values()), f"missing={miss}")
    bad = [g["group_id"] for g in census if (g["note_date"] and not ISO.match(g["note_date"]))
           or (g["published_at"] and not ISO.match(g["published_at"]))
           or (not g["note_date"] and g["published_at"]) or (g["note_date"] and not g["note_date_basis"])
           or (not g["published_at"] and g["time_precision"] != "unknown")]
    check("census: dates ISO, note_date present when published_at known, basis explicit", not bad, f"bad={bad[:10]}")
    bad = [g["group_id"] for g in census if g["lookback_months"] and not re.match(r"^\d+(\.\d+)?$", g["lookback_months"])]
    check("census: lookback_months numeric or blank", not bad, f"bad={bad}")
    verbatim = lambda g: sum(verbatim_words(g[f]) for f in ("quote", "magnitude_text", "short_crowding"))
    bad = [g["group_id"] for g in census if verbatim(g) > 25]
    check("census: verbatim quote budget <= 25 words", not bad, f"bad={bad[:10]}")
    bad = [g["group_id"] for g in census if g["quote"].endswith("…") and verbatim(g) != 25]
    check("census: truncated quotes use the whole 25-word budget (paraphrase not counted)", not bad, f"bad={bad[:10]}")
    bad = [g["group_id"] for g in census if g["coder"].split("(")[0] not in ("claude", "codex")]
    check("census: coder is claude/codex", not bad)
    bad = [g["group_id"] for g in census if g["uncertain_dup"] and not
           set(g["uncertain_dup"].removeprefix("possible_dup_of:").split(";")) <= gids]
    check("census: uncertain_dup references existing groups", not bad)
    blanks = {c: sum(not g[c] for g in census) for c in ("note_date", "published_at", "scope", "flow_direction",
              "magnitude_text", "lookback_months", "gross_leverage", "net_leverage", "short_crowding", "quote")}
    check("census: blank counts reported (blank = not stated, never zero/mixed)", True, json.dumps(blanks))

    # 7. weekly counts reconcile
    weeks, wk = [], COVER_START
    while wk <= COVER_END:
        weeks.append(wk.isoformat())
        wk += timedelta(days=7)
    check("coverage: every Monday week x bank present once",
          sorted((r["week_start"], r["bank"]) for r in cov) == sorted((w, b) for w in weeks for b in BANKS),
          f"weeks={len(weeks)} rows={len(cov)}")
    monday = lambda d: (date.fromisoformat(d[:10]) - timedelta(days=date.fromisoformat(d[:10]).weekday())).isoformat()
    exp_note, exp_conf, exp_pub = Counter(), Counter(), Counter()
    exp_ids = defaultdict(set)
    for g in census:
        if g["note_date"]:
            k = (monday(g["note_date"]), g["bank"])
            exp_note[k] += 1
            exp_conf[k] += g["note_date_basis"] == "explicit"
            exp_ids[k].add(g["group_id"])
        if g["published_at"]:
            exp_pub[(monday(g["published_at"]), g["bank"])] += 1
    bad = []
    for r in cov:
        k = (r["week_start"], r["bank"])
        n, c, p = int(r["groups_note_date"]), int(r["groups_note_confirmed"]), int(r["groups_published"])
        if n != exp_note[k] or c != exp_conf[k] or int(r["groups_note_proxy"]) != n - c or p != exp_pub[k] \
                or set(filter(None, r["group_ids"].split(";"))) != exp_ids[k]:
            bad.append(k)
        want = "found" if n else ("unresolved_candidate" if int(r["unverified_candidates"]) else "search_not_found")
        if r["status"] != want:
            bad.append(k + ("status",))
        if int(r["groups_note_confirmed_excl_uncertain"]) != n - int(r["groups_uncertain_dup"]):
            bad.append(k + ("uncertain",))
    check("coverage: weekly counts reconcile to census rows", not bad, f"bad={bad[:10]}")
    for col, exp in (("groups_note_date", exp_note), ("groups_published", exp_pub)):
        tot = sum(int(r[col]) for r in cov)
        check(f"coverage: {col} total equals in-range census rows", tot == sum(exp.values()), f"total={tot}")
    outside = [g["group_id"] for g in census if g["note_date"] and not (COVER_START.isoformat() <= monday(g["note_date"]) <= COVER_END.isoformat())]
    check("coverage: every dated group falls in a coverage week", not outside, f"outside={outside}")

    # 8. no price access
    banned = ["yfin" + "ance", "yahoo" + "query", "ohl" + "cv", "prices" + ".csv", "prices" + "_meta", "event" + "_outcomes",
              "macro" + "_outcomes", "trading" + "view", "pivot" + "_coverage", "metrics" + ".json", "st" + "ooq"]
    hits = []
    for p in list(ROOT.glob("*.py")) + list(ROOT.glob("*.js")):
        if p.name == Path(__file__).name:
            continue
        text = p.read_text(encoding="utf-8").lower()
        hits += [f"{p.name}:{b}" for b in banned if b in text]
    check("no price APIs/files referenced by folder code", not hits, f"hits={hits}")
    legacy_refs = set()
    for p in ROOT.glob("*.py"):
        legacy_refs |= set(re.findall(r"news_turning_points_20261025/([\w.]+)", p.read_text(encoding="utf-8")))
    check("legacy folder read only for events.json/protocol.json", legacy_refs <= {"events.json", "protocol.json"}, f"refs={sorted(legacy_refs)}")
    price_cols = [c for f in ("census.csv", "census_unverified.csv", "coverage.csv") for c in read_csv(f)[0] if re.search(r"price|return|outcome", c)]
    check("no price/outcome columns in outputs", not price_cols, f"cols={price_cols}")

    # 9. writes only inside task directory (git-visible changes)
    st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    paths = [line[3:].strip().strip('"') for line in st.stdout.splitlines()]
    foreign = [p for p in paths if not p.startswith("research/pb_flow_census_20261026/")]
    protected = [p for p in foreign if p.startswith(("docs/", "AGENTS.md", "research/news_turning_points_20261025/"))]
    check("no changes to docs/, AGENTS.md or the frozen legacy study", not protected, f"protected={protected}")
    check("git-visible changes only inside task directory", not foreign, f"other_paths={foreign[:20]}")

    # 10. repository unittest (optional, slow)
    if "--unittest" in sys.argv:
        ut = subprocess.run([sys.executable, "-m", "unittest", "-q"], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
        tail = (ut.stderr or ut.stdout).strip().splitlines()[-3:]
        check("repository unittest passes", ut.returncode == 0, " / ".join(tail))

    summary = dict(
        ok=all(r["ok"] for r in results), n_checks=len(results), n_failed=sum(not r["ok"] for r in results),
        counts=dict(census_groups=len(census), by_bank=dict(Counter(g["bank"] for g in census)),
                    group_sources=len(gsrc), unverified=len(unver), search_log_rows=len(log),
                    legacy=dict(Counter(r["status"] for r in cross))),
        checks=results)
    (ROOT / "validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for r in results:
        print(("PASS " if r["ok"] else "FAIL ") + r["check"] + ("" if r["ok"] else "  " + r["detail"][:300]))
    print("ok" if summary["ok"] else "FAILED", f"{summary['n_failed']}/{summary['n_checks']} failed")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
