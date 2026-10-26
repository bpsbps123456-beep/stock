"""Draw the reliability sample and write blind input packets (no existing codes)."""
import glob
import json
import os
import random

ROOT = r"C:\Users\geunt\OneDrive\Desktop\주식"
W = os.path.join(ROOT, r"data\derived\analysis\pb_flow_census_20261026\claude_work")
R = os.path.join(ROOT, r"research\pb_flow_census_20261026")
OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 20261026

# Claude pool: eligible=yes rows; key = id#occurrence (file name order, row order)
claude_keys = []
occ = {}
for f in sorted(glob.glob(os.path.join(W, "coded_*.json"))):
    for r in json.load(open(f, encoding="utf-8"))["rows"]:
        k = occ.get(r["id"], 0)
        occ[r["id"]] = k + 1
        if r["eligible"] == "yes":
            claude_keys.append(f"{r['id']}#{k}")

codex_keys = []
occ2 = {}
for r in json.load(open(os.path.join(R, "reviewed_notes.json"), encoding="utf-8")):
    k = occ2.get(r["source_id"], 0)
    occ2[r["source_id"]] = k + 1
    codex_keys.append(f"{r['source_id']}#{k}")

rng = random.Random(SEED)
codex_s = rng.sample(sorted(codex_keys), 12)
claude_s = rng.sample(sorted(claude_keys), 28)
sample = [{"coder": "codex", "key": k} for k in codex_s] + [{"coder": "claude", "key": k} for k in claude_s]
for i, s in enumerate(sample, 1):
    s["sample_no"] = i
json.dump({"seed": SEED, "pool_claude_yes": len(claude_keys), "pool_codex": len(codex_keys),
           "method": "random.Random(seed); sample(sorted(codex_keys),12) then sample(sorted(claude_keys),28)",
           "sample": sample}, open(os.path.join(OUT, "sample_keys.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# Blind packets: input fields only
ri = {r["id"]: r for r in json.load(open(os.path.join(W, "code_reading_items.json"), encoding="utf-8"))}
si = {r["id"]: r for r in json.load(open(os.path.join(W, "code_screen_items.json"), encoding="utf-8"))}
rq = {r["source_id"]: r for r in json.load(open(os.path.join(R, "reading_queue.json"), encoding="utf-8"))}
KEEP_RI = ["title", "source", "url", "published_at", "time_precision", "article_read", "article_dates", "article_flow_paragraphs", "evidence"]
KEEP_SI = ["title", "source", "url", "published_at", "snippet"]
KEEP_RQ = ["title", "source", "url", "published_at", "time_precision", "evidence"]
packets = {}
for s in sample:
    sid = s["key"].split("#")[0]
    if sid in packets:
        continue
    p = {"id": sid}
    if sid in ri:
        p["input_file"] = "code_reading_items"; p.update({k: ri[sid].get(k) for k in KEEP_RI})
    elif sid in rq:
        p["input_file"] = "reading_queue"; p.update({k: rq[sid].get(k) for k in KEEP_RQ})
    elif sid in si:
        p["input_file"] = "code_screen_items"; p.update({k: si[sid].get(k) for k in KEEP_SI})
    ar = os.path.join(R, "article_reads", sid + ".json")
    p["has_article_read_file"] = os.path.exists(ar)
    packets[sid] = p
json.dump(packets, open(os.path.join(OUT, "packets.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("pool claude yes", len(claude_keys), "pool codex", len(codex_keys), "unique articles", len(packets))
print({k: sum(1 for p in packets.values() if p.get("input_file") == k) for k in ["code_reading_items", "reading_queue", "code_screen_items"]})
print(sum(p["has_article_read_file"] for p in packets.values()), "with article_reads file")
