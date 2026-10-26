"""List every US-listed TIME holding (2023-05 weekly PDFs + 2026 daily snapshots) and its desk sector.

Read-only on tracked files. Output: holdings_all.json, unclassified.json, koact_unclassified.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from desk.strategies import UNKNOWN, research_key, sector_of, subsector_of, us_weights  # noqa: E402
from gbrc.etf.market import company_key, holding_key, is_cash_holding, listing_market, yahoo_ticker  # noqa: E402
from gbrc.etf.research_data import exclusion_reason  # noqa: E402
from gbrc.etf.trades import load_fund_snapshots  # noqa: E402

HERE = Path(__file__).resolve().parent
WEEKLY = ROOT / "data/derived/analysis/strategy_lab/holdout/time_weekly"
# holdout_isin_audit.md: the weekly PDFs write these three US names by ISIN.
ISIN_TO_CODE = {
    "US86800U1043": "SMCI US EQUITY",
    "US12430A1025": "BZFD US EQUITY",
    "US0404131064": "ANET US EQUITY",
}

rows: dict[str, dict] = {}


def note(key, name, day, weight, source):
    r = rows.setdefault(key, {"key": key, "ticker": yahoo_ticker(key), "name": name,
                              "first": day, "last": day, "max_weight": 0.0, "max_weight_date": day,
                              "n_weekly": 0, "n_daily": 0, "names": set(), "isin": False})
    r["first"] = min(r["first"], day)
    r["last"] = max(r["last"], day)
    if weight > r["max_weight"]:
        r["max_weight"], r["max_weight_date"] = weight, day
    r[source] += 1
    r["names"].add(name)


# (b) 2023-05 ~ 2025-12 weekly PDFs
isin_hits = {}
for p in sorted(WEEKLY.glob("*.json")):
    s = json.loads(p.read_text(encoding="utf-8"))
    day = s["actual_date"]
    seen = {}
    for h in s["holdings"]:
        code = h.get("code") or ""
        if code in ISIN_TO_CODE:
            isin_hits[code] = isin_hits.get(code, 0) + 1
            h = {**h, "code": ISIN_TO_CODE[code]}
        if is_cash_holding(h) or exclusion_reason(h):
            continue
        key = research_key(h["code"])
        if listing_market(key) != "US":
            continue
        w, nm = seen.get(key, (0.0, h.get("name", "")))
        seen[key] = (w + float(h.get("weight") or 0.0) / 100.0, nm)
    for key, (w, nm) in seen.items():
        note(key, nm, day, w, "n_weekly")

# (a) 2026 daily snapshots (desk path)
snaps = load_fund_snapshots("456600", "9999-12-31")
for day, snap in sorted(snaps.items()):
    names = {research_key(r.get("code")): r.get("name", "") for r in snap["holdings"]}
    for key, w in us_weights(snap).items():
        note(key, names.get(key, ""), day, w, "n_daily")

# Unpriceable / unmapped US-looking codes in weekly PDFs (sanity: anything ISIN-like left?)
leftover = {}
for p in sorted(WEEKLY.glob("*.json")):
    s = json.loads(p.read_text(encoding="utf-8"))
    for h in s["holdings"]:
        code = (h.get("code") or "").strip()
        if not code or code in ISIN_TO_CODE or is_cash_holding(h):
            continue
        if listing_market(code) is None:
            leftover.setdefault(code, h.get("name"))

out = []
for key, r in rows.items():
    r["sector"] = sector_of(key, r["ticker"])
    r["subsector"] = subsector_of(key, r["ticker"])
    r["names"] = sorted(r["names"])
    r["max_weight"] = round(r["max_weight"] * 100, 3)
    out.append(r)
out.sort(key=lambda r: -r["max_weight"])
unc = [r for r in out if r["sector"] == UNKNOWN]

# KoAct US holdings (2026 daily) under the desk resolver
ko = {}
for day, snap in sorted(load_fund_snapshots("2ETFQ1", "9999-12-31").items()):
    for r in snap["holdings"]:
        if is_cash_holding(r) or exclusion_reason(r):
            continue
        key = research_key(r.get("code"))
        if listing_market(key) == "US":
            k = ko.setdefault(key, {"key": key, "ticker": yahoo_ticker(key), "name": r.get("name", ""),
                                    "first": day, "last": day, "max_weight": 0.0})
            k["last"] = day
            k["max_weight"] = max(k["max_weight"], float(r["weight"]))
ko_unc = [dict(v, sector=sector_of(v["key"], v["ticker"])) for v in ko.values()]
ko_unc = [v for v in ko_unc if v["sector"] == UNKNOWN]

(HERE / "holdings_all.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
(HERE / "unclassified.json").write_text(json.dumps(unc, ensure_ascii=False, indent=1), encoding="utf-8")
(HERE / "koact_us_all.json").write_text(json.dumps(sorted(ko.values(), key=lambda v: v["key"]), ensure_ascii=False, indent=1), encoding="utf-8")
(HERE / "koact_unclassified_before.json").write_text(json.dumps(ko_unc, ensure_ascii=False, indent=1), encoding="utf-8")
print("weekly PDFs:", len(list(WEEKLY.glob('*.json'))), "daily snaps:", len(snaps))
print("ISIN rows recovered:", isin_hits)
print("unmapped non-cash codes in weekly PDFs:", leftover)
print("TIME US companies:", len(out), "unclassified:", len(unc))
print("KoAct US companies:", len(ko), "unclassified:", len(ko_unc))
for r in unc:
    print(f"{r['ticker']:8s} {r['max_weight']:6.2f}% {r['first']}~{r['last']} w{r['n_weekly']:3d} d{r['n_daily']:3d} sub={r['subsector']!s:20s} {r['names']}")
print("--- KoAct unclassified")
for v in sorted(ko_unc, key=lambda v: v['key']):
    print(v["ticker"], v["name"], v["first"], v["last"], v["max_weight"])
