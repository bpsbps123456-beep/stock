"""Assemble the PB-flow census from Codex collection + Claude screening/coding.

Price-free: reads only this folder, the git-excluded claude_work folder and the frozen
legacy events.json (not its price/outcome files). Writes census.csv, group_sources.csv,
census_unverified.csv, search_log.csv, legacy_crosswalk.csv, coverage.csv, summary.json.
"""
import csv
import glob
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
WORK = REPO / "data/derived/analysis/pb_flow_census_20261026/claude_work"
LEGACY = REPO / "research/news_turning_points_20261025/events.json"
START, END = date(2023, 1, 1), date(2026, 10, 23)
COVER_START, COVER_END = date(2022, 12, 26), date(2026, 10, 19)

COLUMNS = ["group_id", "id", "bank", "note_date", "note_date_basis", "published_at", "time_precision",
           "scope", "flow_direction", "long_flow", "short_flow", "magnitude_text", "lookback_months",
           "gross_leverage", "net_leverage", "short_crowding", "family", "quote", "url", "source",
           "verification", "source_group", "n_sources", "coder", "uncertain_dup", "legacy_ids",
           "legacy_family", "coding_notes"]
EXCLUDE_CODES = ["unrelated", "13f_or_manager_holdings", "fund_flows_not_bank_client", "bofa_fms_without_cft",
                 "forecast_not_observed", "other_bank", "insufficient_attribution", "junk_page", "out_of_window"]
VER_RANK = {"original_article_read": 3, "syndicated_article_read": 2, "prior_verified_fact": 1,
            "headline_summary_only": 0, "": -1}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(name, rows, columns):
    with open(ROOT / name, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})


def norm_title(title):
    title = re.sub(r"^(?:UPDATE \d+-|CORRECTED-|HEDGE ?FLOW\s*-?|Focus:|Exclusive\s*\|)\s*", "", title or "", flags=re.I)
    title = re.sub(r"^(?:HEDGE ?FLOW\s*-?)\s*", "", title, flags=re.I)
    title = re.split(r"\s[-|–]\s|\s\(|\sBy (?:Reuters|Bloomberg|Investing\.com)|\s- Articles", title, maxsplit=1)[0]
    return re.sub(r"[^a-z0-9]", "", title.lower())


def normalize_exclusion(*texts):
    """Map free-text screening/coding reasons to protocol exclusion codes."""
    joined = " ".join(t for t in texts if t)
    low = joined.lower()
    for code in EXCLUDE_CODES:
        if re.search(r"(?<![a-z_])" + re.escape(code) + r"(?![a-z_])", low):
            return code
    rules = [
        ("out_of_window", ["outside publication window", "out of window", "기간 밖", "기간 외"]),
        ("junk_page", ["junk", "깨진", "시각요소"]),
        ("bofa_fms_without_cft", ["fund manager survey", "fms"]),
        ("13f_or_manager_holdings", ["13f", "13 f", "holdings", "보유 종목", "보유 내역", "지분", "vip", "대량보유", "manager holdings"]),
        ("fund_flows_not_bank_client", ["epfr", "mutual-fund", "mutual fund", "펀드 흐름", "mmf", "fund flow", "fund/trust", "etf aggregate", "superannuation"]),
        ("forecast_not_observed", ["forecast", "predict", "전망", "목표가", "등급", "outlook", "아웃룩", "모형 추정", "cta forecast", "경고 발언"]),
        ("other_bank", ["other bank", "barclays", "ubs", "citi", "hazeltree", "ortex", "nomura", "deutsche"]),
        ("insufficient_attribution", ["insufficient", "귀속", "출처 미확인", "은행 미상"]),
    ]
    for code, keys in rules:
        if any(k in low for k in keys):
            return code
    return "unrelated"


HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")


def verbatim_words(text):
    """Words quoted from the source: none for a "[요약]" paraphrase, Hangul tokens excluded."""
    text = (text or "").strip()
    if text.startswith("[요약]"):
        return 0
    return sum(1 for w in text.rstrip(" …").split() if not HANGUL.search(w))


def week_monday(d):
    return d - timedelta(days=d.weekday())


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build():
    sources = {r["source_id"]: r for r in load(ROOT / "sources_work.json")}
    appearances = load(ROOT / "appearances_work.json")
    queries = load(ROOT / "queries_work.json")
    candidates = load(ROOT / "candidates.json")
    reviewed = load(ROOT / "reviewed_notes.json")
    screen = load(WORK / "screen_decisions_all.json")
    auto_ids = set(load(WORK / "auto_excluded_ids.json"))
    overrides = load(ROOT / "dedup_overrides.json")
    reads = {}
    for p in (ROOT / "article_reads").glob("*.json"):
        r = load(p)
        if r.get("body_found"):
            reads[r["source_id"]] = r
    # Body re-read (2026-10-28): the ego-browser permission prompt left the reading queue at
    # headline_summary_only; bodies fetched later are logged in reread_20261028.csv. Only
    # status=read upgrades verification; codes change only through dedup_overrides fixes.
    reread = {}
    if (ROOT / "reread_20261028.csv").exists():
        with open(ROOT / "reread_20261028.csv", encoding="utf-8") as f:
            reread = {r["source_id"]: r for r in csv.DictReader(f) if r["status"] == "read"}
    for sid, r in reread.items():
        reads.setdefault(sid, {"source_id": sid, "url": r["url"], "body_found": True})

    def read_verification(url):
        dom = url.split("/")[2]
        return "original_article_read" if any(x in dom for x in ("reuters.com", "bloomberg.com")) else "syndicated_article_read"

    # RSS candidate id (inventory key) -> source_id (prepare key)
    cand_key = lambda t, d: hashlib.sha256((t.lower() + "|" + d).encode()).hexdigest()[:12]
    cand_to_src = {}
    for a in appearances:
        cand_to_src.setdefault(cand_key(a["title"], a["published_at"]), a["source_id"])
    cand_ids = {c["candidate_id"] for c in candidates}
    to_src = lambda i: i if i in sources else cand_to_src.get(i, i)

    # ---- coded rows -------------------------------------------------------
    rows = []
    for path in sorted(WORK.glob("coded_*.json")):
        for n, r in enumerate(load(path)["rows"]):
            r = dict(r)
            r["source_id"] = to_src(r["id"])
            r["coder"] = "claude"
            r["row_key"] = f"claude:{path.stem}:{n}"
            r["coded_file"] = path.name
            if r["verification"] == "headline_summary_only" and r["source_id"] in reread:
                r["verification"] = read_verification(reread[r["source_id"]]["url"])
            rows.append(r)
    for n, r in enumerate(reviewed):
        ids = [r["source_id"]] + r.get("also_sources", [])
        dated = sorted((sources[i]["published_at"], i) for i in ids if sources[i]["published_at"])
        first = dated[0][1] if dated else r["source_id"]
        s = sources[first]
        read_ids = [i for i in ids if i in reads]
        if read_ids:
            ver = read_verification(reads[read_ids[0]]["url"])
        else:
            ver = "headline_summary_only"
        pub = r.get("published_at") or s["published_at"]
        rows.append(dict(
            id=r["source_id"], source_id=r["source_id"], eligible="yes", exclusion_reason="", bank=r["bank"],
            note_date=r.get("note_date") or pub, note_date_basis="explicit" if r.get("note_date") else "publication_proxy",
            published_at=pub, time_precision=s["time_precision"] if not r.get("published_at") else "date_article",
            scope=r.get("scope", ""), flow_direction=r.get("flow_direction", ""), long_flow=r.get("long_flow", "na"),
            short_flow=r.get("short_flow", "na"), magnitude_text=r.get("magnitude_text", ""),
            lookback_months=r.get("lookback_months"), gross_leverage=r.get("gross_leverage", ""),
            net_leverage=r.get("net_leverage", ""), short_crowding=r.get("short_crowding", ""), family=r["family"],
            quote="[요약] " + r["coding_notes"].split(". ")[0].rstrip("."), verification=ver,
            coding_notes=r["coding_notes"], same_note_key="", url=s["url"], source=s["source"], coder="codex",
            row_key=f"codex:{n}", note_variant=r.get("note_variant", ""), member_ids=ids))

    # Legacy events re-coded from facts verified in the 10/25 study (prior_verified_fact).
    # They replace the bank-less unverified Claude row for the same source.
    legacy_src = set()
    for n, lr in enumerate(overrides.get("legacy_rows", [])):
        s = sources[lr["source_id"]]
        rows.append(dict(lr, id=lr["source_id"], url=s["url"], source=s["source"], coder="claude",
                         row_key=f"legacy:{lr['legacy_id']}:{lr['bank']}:{n}", coded_file="dedup_overrides.json"))
        legacy_src.add(lr["source_id"])
    rows = [r for r in rows if not (r["source_id"] in legacy_src and r["eligible"] == "unverified")]

    # fix: {source_id: {field: value, reason, _coder?, _bank?}}. _coder defaults to claude;
    # _bank narrows a fix to one bank's row when an article carries several banks' notes.
    for r in rows:
        fix = overrides.get("fix", {}).get(r["source_id"])
        if fix and r["coder"] == fix.get("_coder", "claude") and fix.get("_bank", r["bank"]) == r["bank"]:
            r.update({k: v for k, v in fix.items() if k != "reason" and not k.startswith("_")})
            r["coding_notes"] = r.get("coding_notes", "") + " [조립 수정] " + fix["reason"]

    yes = [r for r in rows if r["eligible"] == "yes"]
    unver = [r for r in rows if r["eligible"] == "unverified"]
    no_rows = [r for r in rows if r["eligible"] == "no"]

    # ---- grouping ----------------------------------------------------------
    dsu = DSU()
    for r in yes:
        dsu.find(r["row_key"])
    by = defaultdict(list)
    for r in yes:
        if r["same_note_key"]:
            by[("key", r["same_note_key"])].append(r)
        by[("src", r["source_id"], r["bank"], r["note_date"], r.get("note_variant", ""))].append(r)
        t = norm_title(sources.get(r["source_id"], {}).get("title", ""))
        if len(t) > 25:
            by[("title", t, r["bank"])].append(r)
        if r["note_date_basis"] == "explicit" and r["note_date"]:
            by[("explicit", r["bank"], r["note_date"], r.get("note_variant", ""))].append(r)
    for key, rs in by.items():
        for r in rs[1:]:
            dsu.union(rs[0]["row_key"], r["row_key"])
    # Codex also_sources: link to Claude rows of the same bank coded from those articles.
    by_src_bank = defaultdict(list)
    for r in yes:
        if r["coder"] == "claude":
            by_src_bank[(r["source_id"], r["bank"])].append(r)
    for r in yes:
        if r["coder"] == "codex":
            for i in r["member_ids"]:
                for c in by_src_bank.get((i, r["bank"]), []):
                    dsu.union(r["row_key"], c["row_key"])
    row_by_key = {r["row_key"]: r for r in yes}
    for bank, a, b, _reason in overrides.get("merge", []):
        ra = [r for r in yes if r["bank"] == bank and a in r.get("member_ids", [r["source_id"]])]
        rb = [r for r in yes if r["bank"] == bank and b in r.get("member_ids", [r["source_id"]])]
        assert ra and rb, (bank, a, b)
        for x in ra + rb:
            dsu.union(ra[0]["row_key"], x["row_key"])

    groups = defaultdict(list)
    for r in yes:
        groups[dsu.find(r["row_key"])].append(r)

    def pub_key(r):
        return (r["published_at"][:10] or "9999", r["row_key"])

    census = []
    group_sources = []
    for members in groups.values():
        member_sources = {}
        for r in members:
            for i in r.get("member_ids", [r["source_id"]]):
                member_sources.setdefault(i, r)
        dated = []
        for i, r in member_sources.items():
            s = sources.get(i, {})
            p = (r["published_at"] if i == r["source_id"] and r["published_at"] else s.get("published_at", ""))
            dated.append((p[:10] if p else "9999", p, i))
        dated.sort()
        rep_src = dated[0][2]
        primary = sorted(members, key=lambda r: (-VER_RANK.get(r["verification"], -1), 0 if r["coder"] == "codex" else 1, pub_key(r)))[0]
        explicit = Counter(r["note_date"] for r in members if r["note_date_basis"] == "explicit" and r["note_date"])
        if explicit:
            note_date, basis = sorted(explicit.items(), key=lambda kv: (-kv[1], kv[0]))[0][0], "explicit"
        else:
            note_date, basis = "", "publication_proxy"
        pub_full = dated[0][1] if dated[0][0] != "9999" else ""
        rep_row = member_sources[rep_src]
        if pub_full and rep_src == rep_row["source_id"] and rep_row["published_at"]:
            precision = rep_row["time_precision"]
        elif pub_full:
            precision = sources.get(rep_src, {}).get("time_precision", "date_feed")
        else:
            precision = "unknown"
        if basis == "publication_proxy":
            note_date = pub_full[:10]
        g = {k: primary.get(k, "") for k in COLUMNS}
        g.update(note_date=note_date, note_date_basis=basis if note_date else "", published_at=pub_full,
                 time_precision=precision, id=rep_src,
                 url=sources.get(rep_src, {}).get("url", rep_row.get("url", "")),
                 source=sources.get(rep_src, {}).get("source", rep_row.get("source", "")),
                 n_sources=len(member_sources))
        coders = sorted({r["coder"] for r in members})
        g["coder"] = primary["coder"] if len(coders) == 1 else primary["coder"] + "(+" + "+".join(c for c in coders if c != primary["coder"]) + ")"
        notes = [f"[{primary['coder']} 주 부호] " + primary.get("coding_notes", "")]
        diffs = []
        for r in members:
            if r is primary:
                continue
            d = {f: r.get(f) for f in ("family", "flow_direction", "long_flow", "short_flow", "note_date", "scope")
                 if (r.get(f) or "") != (primary.get(f) or "")}
            if d and (r["coder"] != primary["coder"] or "family" in d or "flow_direction" in d):
                diffs.append(f"[{r['coder']} {r['source_id']}] " + ", ".join(f"{k}={v or '빈칸'}" for k, v in d.items()))
        seen = set()
        for x in diffs:
            core = x.split("] ", 1)[1]
            if (x.split()[0], core) not in seen:
                seen.add((x.split()[0], core))
                notes.append(x)
        other_coder = [r for r in members if r["coder"] != primary["coder"]]
        if other_coder:
            notes.append(f"[{other_coder[0]['coder']} 원 메모] " + other_coder[0].get("coding_notes", "")[:600])
        g["coding_notes"] = " | ".join(n for n in notes if n)
        g["_members"] = members
        g["_member_sources"] = member_sources
        g["lookback_months"] = primary.get("lookback_months")
        census.append(g)

    # ---- quote budget (<=25 verbatim words per row) -----------------------
    # Only source-language words count: a "[요약]" field or a Hangul token is the coder's
    # paraphrase, not a quotation (counting them truncated quotes such as G0142 to one word).
    for g in census:
        verb = verbatim_words
        q = g["quote"] or ""
        budget = 25 - sum(verb(g[f]) for f in ("magnitude_text", "short_crowding"))
        if not q.startswith("[요약]") and verb(q) > max(budget, 0):
            words = q.split()[:max(budget, 0)]
            g["quote"] = (" ".join(words) + " …") if words else "[요약] 인용 예산 초과로 원문 인용 생략"
            g["coding_notes"] += " | [조립] 인용 합계 25단어 예산에 맞춰 quote를 잘라냄."

    census.sort(key=lambda g: (g["published_at"][:10] or g["note_date"] or "9999", g["bank"], g["id"]))
    for n, g in enumerate(census, 1):
        g["group_id"] = f"G{n:04d}"
        g["source_group"] = g["group_id"]

    # ---- uncertain duplicates: same bank, close dates, similar titles ------
    def tokens(g):
        t = " ".join(sources.get(i, {}).get("title", "") for i in g["_member_sources"])
        stop = {"hedge", "funds", "fund", "goldman", "sachs", "says", "the", "to", "in", "of", "and", "a", "by",
                "reuters", "bloomberg", "stocks", "stock", "at", "on", "for", "since", "investing", "com", "-", "|"}
        return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in stop and len(w) > 2}
    tok = {g["group_id"]: tokens(g) for g in census}
    dkey = lambda g: date.fromisoformat((g["note_date"] or g["published_at"][:10]))
    # A pair is uncertain when the same bank's groups sit within 4 days and either
    # (a) a coder explicitly named the other group's source as a possible same note,
    # (b) titles overlap (token Jaccard >= 0.3), or (c) one side only has a publication-proxy
    # date, so the covered week cannot be told apart. Explicit, different note dates with
    # dissimilar titles are treated as distinct notes.
    gid_of_src = defaultdict(set)
    for g in census:
        for i in g["_member_sources"]:
            gid_of_src[i].add(g["group_id"])
    named = defaultdict(set)
    for g in census:
        text = " ".join(r.get("coding_notes", "") for r in g["_members"])
        for i in set(re.findall(r"\b[0-9a-f]{12}\b", text)):
            for other in gid_of_src.get(i, ()):
                if other != g["group_id"]:
                    named[g["group_id"]].add(other)
    by_gid = {g["group_id"]: g for g in census}
    uncertain = defaultdict(set)
    by_bank = defaultdict(list)
    for g in census:
        if g["note_date"] or g["published_at"]:
            by_bank[g["bank"]].append(g)
    for bank, gs in by_bank.items():
        gs.sort(key=dkey)
        for i, a in enumerate(gs):
            for b in gs[i + 1:]:
                if (dkey(b) - dkey(a)).days > 4:
                    break
                ta, tb = tok[a["group_id"]], tok[b["group_id"]]
                jac = len(ta & tb) / max(1, len(ta | tb))
                coder_named = b["group_id"] in named[a["group_id"]] or a["group_id"] in named[b["group_id"]]
                proxy = "publication_proxy" in (a["note_date_basis"], b["note_date_basis"])
                if coder_named or jac >= 0.3 or proxy:
                    later = b if (b["published_at"] or "9999", b["group_id"]) >= (a["published_at"] or "9999", a["group_id"]) else a
                    other = a if later is b else b
                    uncertain[later["group_id"]].add(other["group_id"])
    for g in census:
        g["uncertain_dup"] = ("possible_dup_of:" + ";".join(sorted(uncertain[g["group_id"]]))) if uncertain.get(g["group_id"]) else ""

    # ---- group_sources ------------------------------------------------------
    for g in census:
        for i, r in g["_member_sources"].items():
            s = sources.get(i, {})
            group_sources.append(dict(group_id=g["group_id"], source_id=i, is_representative="yes" if i == g["id"] else "no",
                                      title=s.get("title", ""),
                                      published_at=(r["published_at"] if i == r["source_id"] and r["published_at"] else s.get("published_at", "")),
                                      source=s.get("source", ""), url=s.get("url", r.get("url", "")),
                                      coder=r["coder"], verification=r["verification"] if i == r["source_id"] else "",
                                      coded_row=r["row_key"]))
    src_group = {}
    for gs in group_sources:
        src_group.setdefault(gs["source_id"], set()).add(gs["group_id"])

    # ---- unverified ---------------------------------------------------------
    unv_rows = []
    for r in unver:
        s = sources.get(r["source_id"], {})
        unv_rows.append(dict(r, id=r["source_id"], title=s.get("title", ""), group_id="", n_sources=1,
                             uncertain_dup="", legacy_ids="", legacy_family="", source_group=""))
    unv_rows.sort(key=lambda r: (r["published_at"][:10] or r["note_date"] or "9999", r["id"]))
    unv_by_src = defaultdict(list)
    for r in unver:
        unv_by_src[r["source_id"]].append(r)
    no_by_src = defaultdict(list)
    for r in no_rows:
        no_by_src[r["source_id"]].append(r)

    # ---- search log ----------------------------------------------------------
    screen_by_src = {}
    for i, d in screen.items():
        screen_by_src.setdefault(to_src(i), d)
    codex_src = {}
    for r in yes:
        if r["coder"] == "codex":
            for i in r["member_ids"]:
                codex_src.setdefault(i, r)
    claude_yes_src = defaultdict(list)
    for r in yes:
        if r["coder"] == "claude":
            claude_yes_src[r["source_id"]].append(r)

    def decide(sid):
        groups_ = ";".join(sorted(src_group.get(sid, [])))
        if sid in codex_src:
            return "codex_coded", "", groups_, "Codex 검토 노트 " + codex_src[sid]["bank"] + " 부호화(reviewed_notes.json)"
        if sid in claude_yes_src:
            return "coded_eligible", "", groups_, "Claude 부호화 eligible=yes: " + "; ".join(r["same_note_key"] for r in claude_yes_src[sid])
        if sid in unv_by_src:
            return "coded_unverified", "", "", "Claude 부호화 eligible=unverified(은행·관측 사실 미확인): " + unv_by_src[sid][0]["coding_notes"][:160]
        if sid in no_by_src:
            r = no_by_src[sid][0]
            return "coded_no", normalize_exclusion(r["exclusion_reason"], r["coding_notes"] if r["exclusion_reason"] in ("", "Unrelated or insufficiently attributable result") else ""), "", "Claude 부호화 eligible=no: " + (r["exclusion_reason"] + " / " + r["coding_notes"])[:200]
        if sid in screen_by_src:
            d = screen_by_src[sid]
            if d["decision"] == "exclude":
                return "screen_exclude", normalize_exclusion(d["reason"], d.get("note", "")), "", ("자동 키워드 제외군 LLM 재선별: " if sid in auto_ids else "LLM 선별: ") + (d["reason"] + " / " + d.get("note", "")).strip(" /")[:200]
            return "screen_" + d["decision"] + "_uncoded", "", "", d["reason"][:200]
        return "undecided", "", "", ""

    log = []
    qinfo = {q["query_id"]: q for q in queries}
    app_by_q = defaultdict(list)
    for a in appearances:
        app_by_q[a["query_id"]].append(a)
    for q in queries:
        base = dict(query_id=q["query_id"], engine=q["engine"], bank_query=q["bank"], query=q["query"],
                    window_start=q["start"], window_end_exclusive=q["end_exclusive"], query_result_count=q["result_count"],
                    query_error=q["error"])
        if q["error"]:
            log.append(dict(base, rank="", decision="query_error", exclusion_code="", reason=q["error"]))
        if not app_by_q[q["query_id"]]:
            log.append(dict(base, rank="", decision="query_zero_results", exclusion_code="",
                            reason="결과 0건(색인 없음은 노트 부재의 증거가 아님)"))
        for a in sorted(app_by_q[q["query_id"]], key=lambda a: a["rank"]):
            sid = a["source_id"]
            decision, code, gids, reason = decide(sid)
            pub = a["published_at"]
            if decision == "screen_exclude" and code != "out_of_window" and pub and not (START.isoformat() <= pub <= END.isoformat()):
                reason += " (공개일도 기간 밖)"
            ck = cand_key(a["title"], a["published_at"]) if q["engine"] == "Google News RSS" else ""
            log.append(dict(base, rank=a["rank"], source_id=sid, candidate_id=ck if ck in cand_ids else "",
                            title=a["title"], published_at=pub, source=a["source"], url=a["url"],
                            decision=decision, exclusion_code=code, group_ids=gids, reason=reason))

    # ---- legacy crosswalk -------------------------------------------------
    legacy = load(LEGACY)
    url_to_src = defaultdict(set)
    for i, s in sources.items():
        url_to_src[s["url"].split("?")[0].rstrip("/")].add(i)
    lmap = overrides.get("legacy", {})
    cross = []
    legacy_by_group = defaultdict(list)
    for e in legacy:
        u = e["url"].split("?")[0].rstrip("/")
        sids = sorted(url_to_src.get(u, set()))
        auto_groups = sorted({gid for i in sids for gid in src_group.get(i, [])})
        manual = lmap.get(e["id"], {})
        gids = manual.get("groups", auto_groups) if manual else auto_groups
        # manual entries may reference group by representative source id
        resolved = []
        for x in gids:
            if x.startswith("G"):
                resolved.append(x)
            else:
                resolved.extend(sorted(src_group.get(x, [])))
        resolved = sorted(set(resolved))
        row = dict(legacy_id=e["id"], legacy_date=e["date"], legacy_family=e["family"], legacy_include=e["include"],
                   legacy_exclusion=e["exclusion"], legacy_title=e["title"], legacy_url=e["url"],
                   url_matched_sources=";".join(sids), group_ids=";".join(resolved))
        fams = sorted({next(g for g in census if g["group_id"] == x)["family"] for x in resolved})
        row["current_family"] = ";".join(fams)
        row["match_method"] = manual.get("method", "url_exact" if auto_groups else "")
        if resolved:
            row["status"] = "linked"
            row["family_changed"] = "yes" if e["family"] not in fams else "no"
        else:
            row["status"] = "excluded"
            row["family_changed"] = ""
        row["current_exclusion"] = manual.get("exclusion", "" if resolved else "not_found_in_current_search")
        row["note"] = manual.get("note", "")
        cross.append(row)
        for x in resolved:
            legacy_by_group[x].append(e)
    for g in census:
        es = legacy_by_group.get(g["group_id"], [])
        g["legacy_ids"] = ";".join(e["id"] for e in es)
        g["legacy_family"] = ";".join(e["family"] for e in es)

    # ---- coverage ---------------------------------------------------------
    cov = []
    banks = ["GS", "MS", "JPM", "BofA"]
    # Unverified rows without a bank are attributed to the banks whose queries returned them.
    q_bank = {q["query_id"]: q["bank"] for q in queries}
    banks_of_src = defaultdict(set)
    for a in appearances:
        banks_of_src[a["source_id"]].add(q_bank[a["query_id"]])
    unv_dates = defaultdict(list)
    for r in unver:
        d = r["note_date"] or r["published_at"][:10]
        if d:
            for b in ([r["bank"]] if r["bank"] in ("GS", "MS", "JPM", "BofA") else sorted(banks_of_src[r["source_id"]])):
                unv_dates[(week_monday(date.fromisoformat(d)), b)].append(r)
    wk = COVER_START
    while wk <= COVER_END:
        for bank in banks:
            gs = [g for g in census if g["bank"] == bank]
            note_conf = [g for g in gs if g["note_date"] and g["note_date_basis"] == "explicit" and week_monday(date.fromisoformat(g["note_date"])) == wk]
            note_proxy = [g for g in gs if g["note_date"] and g["note_date_basis"] == "publication_proxy" and week_monday(date.fromisoformat(g["note_date"])) == wk]
            pub = [g for g in gs if g["published_at"] and week_monday(date.fromisoformat(g["published_at"][:10])) == wk]
            all_note = note_conf + note_proxy
            unc = [g for g in all_note if g["uncertain_dup"]]
            u = unv_dates.get((wk, bank), [])
            if all_note:
                status = "found"
            elif u:
                status = "unresolved_candidate"
            else:
                status = "search_not_found"
            cov.append(dict(week_start=wk.isoformat(), bank=bank, groups_note_date=len(all_note),
                            groups_note_confirmed=len(note_conf), groups_note_proxy=len(note_proxy),
                            groups_note_confirmed_excl_uncertain=len(all_note) - len(unc),
                            groups_uncertain_dup=len(unc), groups_published=len(pub),
                            unverified_candidates=len(u), status=status,
                            found_confirmed="yes" if len(all_note) - len(unc) > 0 else "no",
                            group_ids=";".join(g["group_id"] for g in all_note)))
        wk += timedelta(days=7)

    # ---- write --------------------------------------------------------------
    for g in census:
        lm = g.get("lookback_months")
        g["lookback_months"] = "" if lm in (None, "") else (int(lm) if float(lm).is_integer() else round(float(lm), 2))
        for f in ("long_flow", "short_flow"):
            g[f] = g[f] or "na"
    write_csv("census.csv", census, COLUMNS)
    write_csv("group_sources.csv", group_sources, ["group_id", "source_id", "is_representative", "title", "published_at",
                                                   "source", "url", "coder", "verification", "coded_row"])
    ucols = ["id", "title", "bank", "note_date", "note_date_basis", "published_at", "time_precision", "scope",
             "flow_direction", "long_flow", "short_flow", "magnitude_text", "lookback_months", "family", "quote",
             "url", "source", "verification", "coder", "coding_notes"]
    for r in unv_rows:
        lm = r.get("lookback_months")
        r["lookback_months"] = "" if lm in (None, "") else lm
    write_csv("census_unverified.csv", unv_rows, ucols)
    write_csv("search_log.csv", log, ["query_id", "engine", "bank_query", "query", "window_start", "window_end_exclusive",
                                      "query_result_count", "query_error", "rank", "source_id", "candidate_id", "title",
                                      "published_at", "source", "url", "decision", "exclusion_code", "group_ids", "reason"])
    write_csv("legacy_crosswalk.csv", cross, ["legacy_id", "legacy_date", "legacy_family", "legacy_include", "legacy_exclusion",
                                              "legacy_title", "status", "group_ids", "current_family", "family_changed",
                                              "match_method", "current_exclusion", "url_matched_sources", "legacy_url", "note"])
    write_csv("coverage.csv", cov, ["week_start", "bank", "groups_note_date", "groups_note_confirmed", "groups_note_proxy",
                                    "groups_note_confirmed_excl_uncertain", "groups_uncertain_dup", "groups_published",
                                    "unverified_candidates", "status", "found_confirmed", "group_ids"])
    summary = dict(
        census_groups=len(census), by_bank=dict(Counter(g["bank"] for g in census)),
        coded_rows=len(rows), claude_rows=sum(r["coder"] == "claude" for r in rows), codex_rows=sum(r["coder"] == "codex" for r in rows),
        eligible_rows=len(yes), unverified_rows=len(unver), no_rows=len(no_rows),
        search_log_rows=len(log), decisions=dict(Counter(r["decision"] for r in log)),
        family=dict(Counter(g["family"] for g in census)), uncertain_groups=sum(bool(g["uncertain_dup"]) for g in census))
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return census, cov, cross, log


if __name__ == "__main__":
    build()
