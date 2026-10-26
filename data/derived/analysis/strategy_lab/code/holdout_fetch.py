"""Fetch one TIME PDF per week (first KRX session of each ISO week) for the
untouched holdout period 2023-05-22 .. 2025-12-31. Research copy only: nothing is
written to data/normalized or data/etf_holdings. Resumable and paced."""

import datetime as dt
import json
import sys
import time
import urllib.error

from data import CACHE, REPO

sys.path.insert(0, str(REPO))
from gbrc.config import resolve_fund  # noqa: E402
from gbrc.etf.providers import fetch_holdings  # noqa: E402

TIME = resolve_fund("time")

OUT = CACHE / "holdout" / "time_weekly"
START, END = dt.date(2023, 5, 22), dt.date(2025, 12, 31)
PAUSE = 2.0


def fetch(day):
    for attempt in range(5):
        try:
            return fetch_holdings(TIME, day.isoformat())
        except (urllib.error.URLError, TimeoutError) as exc:
            wait = 30 * (attempt + 1)
            print(f"  {day} {type(exc).__name__}: retry in {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"giving up at {day}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in OUT.glob("*.json")}
    monday = START
    while monday <= END:
        week_days = [monday + dt.timedelta(days=i) for i in range(5)]
        if any(p[:10] >= monday.isoformat() and p[:10] <= week_days[-1].isoformat() for p in have):
            monday += dt.timedelta(days=7)
            continue
        for day in week_days:  # first KRX session of the week = first exact-date PDF
            snap = fetch(day)
            time.sleep(PAUSE)
            if snap and snap.get("actual_date") == day.isoformat() and snap["holdings"]:
                keep = {k: snap[k] for k in ("date", "actual_date", "source_url", "fetched_at", "holdings_count")}
                keep["holdings"] = [{k: h[k] for k in ("code", "name", "qty", "value_krw", "weight")}
                                    for h in snap["holdings"]]
                (OUT / f"{day.isoformat()}.json").write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
                have.add(day.isoformat())
                print(f"{day} ok holdings={snap['holdings_count']}", flush=True)
                break
        else:
            print(f"{monday} week: no exact-date PDF (holiday week)", flush=True)
        monday += dt.timedelta(days=7)
    print(f"done: {len(list(OUT.glob('*.json')))} weekly PDFs in {OUT}")


if __name__ == "__main__":
    main()
