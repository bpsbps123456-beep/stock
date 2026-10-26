"""Compare blind recode against existing Claude/Codex codes for the 40 sampled rows."""
import collections
import glob
import json
import os
from datetime import date

ROOT = r"C:\Users\geunt\OneDrive\Desktop\주식"
W = os.path.join(ROOT, r"data\derived\analysis\pb_flow_census_20261026\claude_work")
R = os.path.join(ROOT, r"research\pb_flow_census_20261026")
HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["bank", "flow_direction", "long_flow", "short_flow", "family", "lookback_months", "note_date"]

sample = json.load(open(os.path.join(HERE, "sample_keys.json"), encoding="utf-8"))["sample"]
recode = json.load(open(os.path.join(HERE, "recode.json"), encoding="utf-8"))["articles"]
packets = json.load(open(os.path.join(HERE, "packets.json"), encoding="utf-8"))

pubs = {}
for r in json.load(open(os.path.join(W, "code_reading_items.json"), encoding="utf-8")):
    pubs[r["id"]] = r.get("published_at") or ""
for r in json.load(open(os.path.join(W, "code_screen_items.json"), encoding="utf-8")):
    pubs.setdefault(r["id"], r.get("published_at") or "")
for r in json.load(open(os.path.join(R, "reading_queue.json"), encoding="utf-8")):
    pubs.setdefault(r["source_id"], r.get("published_at") or "")
sw = json.load(open(os.path.join(R, "sources_work.json"), encoding="utf-8"))
sw_rows = sw if isinstance(sw, list) else list(sw.values()) if isinstance(sw, dict) and not sw.get("sources") else sw.get("sources", [])
for s in sw_rows:
    if isinstance(s, dict):
        sid = s.get("source_id") or s.get("id")
        if sid and s.get("published_at"):
            pubs.setdefault(sid, s["published_at"])


def norm_cat(v, na="na"):
    if v in (None, "", "na", "unknown"):
        return na
    return str(v)


def norm(field, v):
    if field in ("long_flow", "short_flow"):
        return norm_cat(v, "na")
    if field in ("flow_direction", "bank"):
        return norm_cat(v, "")
    if field == "lookback_months":
        return None if v in (None, "") else round(float(v), 2)
    if field == "note_date":
        return (v or "")[:10]
    return v


existing = {}
occ = {}
for f in sorted(glob.glob(os.path.join(W, "coded_*.json"))):
    for r in json.load(open(f, encoding="utf-8"))["rows"]:
        k = occ.get(r["id"], 0)
        occ[r["id"]] = k + 1
        existing[("claude", f"{r['id']}#{k}")] = dict(r, _file=os.path.basename(f))
occ2 = {}
for r in json.load(open(os.path.join(R, "reviewed_notes.json"), encoding="utf-8")):
    k = occ2.get(r["source_id"], 0)
    occ2[r["source_id"]] = k + 1
    rr = dict(r)
    if not rr.get("note_date"):
        ids = [r["source_id"]] + r.get("also_sources", [])
        dated = sorted(pubs[i] for i in ids if pubs.get(i))
        rr["note_date"] = r.get("published_at") or (dated[0] if dated else "")
        rr["_note_date_filled"] = "publication_proxy(assemble.py rule)"
    existing[("codex", f"{r['source_id']}#{k}")] = rr


def days(a, b):
    try:
        return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)
    except ValueError:
        return 9999


out = []
for s in sample:
    ex = existing[(s["coder"], s["key"])]
    sid = s["key"].split("#")[0]
    notes = recode[sid]["notes"]
    n_existing_rows = (occ if s["coder"] == "claude" else occ2)[sid]
    if len(notes) == 1:
        mine, how = notes[0], "single_note"
    else:
        same_bank = [n for n in notes if n["bank"] == ex.get("bank")]
        cand = same_bank or notes
        mine = sorted(cand, key=lambda n: days(n["note_date"], norm("note_date", ex.get("note_date"))))[0]
        how = "matched_on_bank" if same_bank else "no_bank_match_nearest_date"
    row = {"sample_no": s["sample_no"], "coder": s["coder"], "key": s["key"], "id": sid,
           "title": packets[sid]["title"], "input_file": packets[sid]["input_file"],
           "n_existing_rows_same_id": n_existing_rows, "n_recode_notes": len(notes),
           "match": how, "recode_variant": mine["v"], "recode_confidence": mine.get("confidence", ""),
           "webfetch": mine.get("webfetch", ""), "info": mine.get("info", ""),
           "existing_scope": ex.get("scope", ""), "recode_scope": mine.get("scope", ""),
           "existing_basis": ex.get("note_date_basis", ex.get("_note_date_filled", "")),
           "recode_basis": mine.get("note_date_basis", "")}
    for f in FIELDS:
        a, b = norm(f, ex.get(f)), norm(f, mine.get(f))
        if f == "lookback_months":
            ag = (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 1)
            row[f + "_exact"] = int(a == b)
        elif f == "note_date":
            ag = a == b
            row["note_date_within3d"] = int(days(a, b) <= 3)
        else:
            ag = a == b
        row[f + "_existing"] = "" if a is None else a
        row[f + "_recode"] = "" if b is None else b
        row[f + "_agree"] = int(ag)
        io = mine.get("input_only", {})
        if f in io:
            bi = norm(f, io[f])
            row[f + "_recode_input_only"] = "" if bi is None else bi
    out.append(row)

json.dump(out, open(os.path.join(HERE, "compare_rows.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


for grp in ["all", "claude", "codex"]:
    rows = [r for r in out if grp == "all" or r["coder"] == grp]
    print(f"== {grp} n={len(rows)}")
    for f in FIELDS:
        ag = sum(r[f + "_agree"] for r in rows)
        extra = ""
        if f not in ("lookback_months", "note_date"):
            po, k = kappa([(r[f + "_existing"], r[f + "_recode"]) for r in rows])
            extra = f" kappa={k:.2f}"
        if f == "lookback_months":
            extra = f" exact={sum(r['lookback_months_exact'] for r in rows)}"
        if f == "note_date":
            extra = f" within3d={sum(r['note_date_within3d'] for r in rows)}"
        print(f"  {f}: {ag}/{len(rows)} = {ag/len(rows):.0%}{extra}")
print()
for r in out:
    dis = [f for f in FIELDS if not r[f + "_agree"]]
    if dis:
        print(r["sample_no"], r["coder"], r["key"], r["match"], r["n_existing_rows_same_id"], r["n_recode_notes"], "|", r["title"][:70])
        print("    scope ex:", r["existing_scope"][:80], "| mine:", r["recode_scope"], "| basis ex:", r["existing_basis"], "mine:", r["recode_basis"])
        for f in dis:
            print(f"    {f}: existing={r[f+'_existing']!r} recode={r[f+'_recode']!r}" + (f" input_only={r[f+'_recode_input_only']!r}" if f + '_recode_input_only' in r else ""))
