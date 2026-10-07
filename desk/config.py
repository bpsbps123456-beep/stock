"""Paths, frozen rule parameters, and the US market calendar the desk needs."""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "derived" / "desk"          # regenerable (prices, state)
# User records are never regenerated; DESK_JOURNAL_DIR points tests at a scratch folder.
JOURNAL_DIR = Path(os.environ.get("DESK_JOURNAL_DIR") or ROOT / "data" / "journal")
TRADES = JOURNAL_DIR / "trades.json"
SETTINGS = JOURNAL_DIR / "settings.json"
STATIC = Path(__file__).resolve().parent / "static"

# Windows (Hyper-V/WSL) reserves shifting port ranges; the server tries these in order.
HOST, PORTS = "127.0.0.1", (8321, 7421, 5731, 0)
MODEL_START = "2026-01-02"        # first validated TIME snapshot
PRICE_START = "2025-06-01"        # warm-up for the 120-session momentum window
PRICE_MAX_AGE_MIN = 30            # reuse the daily price cache for this long
QUOTE_MAX_AGE_SEC = 90

# The one strategy (docs/desk_system.md §3): Codex concentration study #774, chosen by
# the user on 2026-09-27.
STRATEGY_NAME = "7종목 · 120일 수익률 · TIME 비중 · 주간 조정"
RULE = {"lookback": 120, "n": 7, "buffer": 5, "stock_cap": 0.25, "sector_cap": 0.60, "semis_cap": 0.60}
# watchlist.py labels that share the semiconductor budget (unclassified names count too).
SEMIS_SECTORS = {"AI 반도체", "메모리", "반도체 장비", "반도체 장비(적자)", "반도체(적자)", "반도체 소재"}
DEFAULT_SETTINGS = {"tolerance": 0.02}

KST = ZoneInfo("Asia/Seoul")
NY = ZoneInfo("America/New_York")
REFERENCES = ["QQQ", "KRW=X", "456600.KS"]

# NYSE full-day closures (future sessions only; past sessions come from QQQ prices).
NYSE_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


def is_us_session(day: date) -> bool:
    return day.weekday() < 5 and day.isoformat() not in NYSE_HOLIDAYS


def next_us_session(now: datetime | None = None) -> tuple[str, datetime]:
    """The next regular session whose open has not passed, and that open in KST."""
    now_ny = (now or datetime.now(NY)).astimezone(NY)
    day = now_ny.date()
    if not (is_us_session(day) and now_ny.time() < time(9, 30)):
        day += timedelta(days=1)
        while not is_us_session(day):
            day += timedelta(days=1)
    open_ny = datetime.combine(day, time(9, 30), tzinfo=NY)
    return day.isoformat(), open_ny.astimezone(KST)


def next_rebalance(session: str) -> str:
    """First US session of the ISO week after `session`'s week."""
    day = date.fromisoformat(session)
    week = day.isocalendar()[:2]
    while not (is_us_session(day) and day.isocalendar()[:2] != week):
        day += timedelta(days=1)
    return day.isoformat()
