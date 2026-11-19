"""Published S&P 500 table revisions captured at the prior calendar month's end."""
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "broad_membership.csv"
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "GBRC-Research/1.0 (https://github.com/bpsbps123456-beep/stock; historical research)"}


def snapshot(month):
    cutoff = date.fromisoformat(month + "-01") - timedelta(days=1)
    params = {"action": "query", "prop": "revisions", "titles": "List of S&P 500 companies",
              "rvstart": f"{cutoff}T23:59:59Z", "rvlimit": 1, "rvslots": "main",
              "rvprop": "ids|timestamp|content", "formatversion": 2, "format": "json"}
    response = requests.get(API, headers=HEADERS, params=params, timeout=40)
    response.raise_for_status()
    rev = response.json()["query"]["pages"][0]["revisions"][0]
    assert rev["timestamp"][:10] <= str(cutoff)
    content = rev["slots"]["main"]["content"].split('id="constituents"', 1)[1].split("\n|}", 1)[0]
    rows = []
    for line in re.split(r"\n\|-+[^\n]*\n", content)[2:]:
        match = re.search(r"\{\{(?:nysesymbol|nasdaqsymbol|bzx link)\s*\|\s*([^|}]+)", line, re.I)
        chunks = re.split(r"\|\||\n\|", line)
        cik = re.search(r"\b(\d{7,10})\b", chunks[-2])
        if not match or not cik:
            raise ValueError(f"Cannot parse {month}: {line[:250]}")
        ticker = match.group(1).strip().replace(".", "-")
        hq = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", chunks[-4].strip())
        rows.append((month, ticker, str(int(cik.group(1))), hq))
    assert 495 <= len(rows) <= 512 and len({r[1] for r in rows}) == len(rows), (month, len(rows))
    return rows, {"month": month, "prior_month_end": str(cutoff), "revision": rev["revid"],
                  "timestamp_utc": rev["timestamp"], "securities": len(rows),
                  "url": f"https://en.wikipedia.org/w/index.php?oldid={rev['revid']}"}


def rebuild():
    months = [f"2026-{m:02}" for m in range(1, 12)]
    with ThreadPoolExecutor(max_workers=3) as workers:
        results = list(workers.map(snapshot, months))
    # The December 23 revision recorded this effective change in its change log,
    # but omitted the three additions from the displayed roster and retained SOLS.
    # Source issuer identity fields from February; membership is established by the
    # December 5 index-provider notice, before any performance measurement.
    january, note = results[0]
    january = [r for r in january if r[1] not in {"LKQ", "SOLS", "MHK"}]
    january.extend((months[0], *r[1:]) for r in results[1][0] if r[1] in {"CRH", "CVNA", "FIX"})
    assert len(january) == len({r[1] for r in january}) == 503
    note.update(securities=len(january), index_notice="https://press.spglobal.com/2025-12-05-CRH%2C-Carvana-and-Comfort-Systems-USA-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
                correction="12/22/2025 additions CRH, CVNA, FIX; remove leftover SOLS")
    results[0] = (january, note)
    with FROZEN.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["month", "symbol", "cik", "headquarters"])
        for rows, _ in results:
            writer.writerows(rows)
    (HERE / "broad_sources.json").write_text(json.dumps([s for _, s in results], indent=2) + "\n", encoding="utf-8")


@lru_cache
def _membership():
    by_month = {}
    with FROZEN.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            by_month.setdefault(row["month"], set()).add(row["symbol"])
    return by_month


def broad_members(day: str) -> set[str]:
    return _membership()[day[:7]].copy()


def broad_symbols() -> set[str]:
    return set.union(*_membership().values())


if __name__ == "__main__":
    rebuild()
