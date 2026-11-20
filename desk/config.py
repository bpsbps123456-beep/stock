"""Paths, frozen rule parameters, and the US market calendar the desk needs."""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "derived" / "desk"          # regenerable (prices, state)
# User records are never regenerated; DESK_JOURNAL_DIR points tests at a scratch folder.
JOURNAL_DIR = Path(os.environ.get("DESK_JOURNAL_DIR") or ROOT / "data" / "journal")
TRADES = JOURNAL_DIR / "trades.json"
SETTINGS = JOURNAL_DIR / "settings.json"
STATIC = Path(__file__).resolve().parent / "static"
# One-time records (pre-open observations, month baskets, published outlooks, season paper) are
# locked once and never rewritten. A test server (DESK_JOURNAL_DIR set, docs/desk_system.md §2) must
# not lock them in the real folders, so it writes them under <DESK_JOURNAL_DIR>/records/.
RECORDS_SCRATCH = JOURNAL_DIR / "records" if os.environ.get("DESK_JOURNAL_DIR") else None


def record_dir(real: Path) -> Path:
    """Where a one-time record is written: `real`, or its mirror under the test server's scratch folder."""
    if RECORDS_SCRATCH is None:
        return real
    real = Path(real)
    try:
        return RECORDS_SCRATCH / real.resolve().relative_to(ROOT)
    except ValueError:
        return RECORDS_SCRATCH / real.name


def real_record(path: Path) -> Path:
    """The real location of a scratch record path (read-only fallback for immutable inputs such as
    an already frozen month basket); other paths are returned unchanged."""
    path = Path(path)
    if RECORDS_SCRATCH is not None:
        try:
            return ROOT / path.relative_to(RECORDS_SCRATCH)
        except ValueError:
            pass
    return path


# Windows (Hyper-V/WSL) reserves shifting port ranges; the server tries these in order.
HOST, PORTS = "127.0.0.1", (8321, 7421, 5731, 0)
MODEL_START = "2026-01-02"        # first validated TIME snapshot
PRICE_START = "2025-06-01"        # warm-up for the 120-session momentum window
PRICE_MAX_AGE_MIN = 30            # reuse the daily price cache for this long
QUOTE_MAX_AGE_SEC = 90

# The one strategy (docs/desk_system.md §3): Codex concentration study #774, chosen by
# the user on 2026-09-27. The semiconductor budget was lowered from #774's 60% to 50% on
# 2026-10-10 (user decision; evidence in docs/operations_changelog.md).
STRATEGY_NAME = "7종목 · 60일 잔차 모멘텀 · TIME 비중 · 주간 조정"
# score "res": 60거래일 잔차 모멘텀(종목 일간 로그수익 합 - 베타 x QQQ 합). 2026-11-21 사용자 결정으로 120일 수익률("ret")에서
# 바꿈(docs/desk_system.md §3, data/derived/analysis/momentum_variants_20261120/report.md). CODEX_774_RULE은 score 없이 120일 수익률.
RULE = {"score": "res", "lookback": 60, "n": 7, "buffer": 5, "stock_cap": 0.25, "sector_cap": 0.60, "semis_cap": 0.50}
# #774 exactly as Codex tested it (a literal, so later RULE edits cannot drift it);
# `desk validate` replays the engine with these caps.
CODEX_774_RULE = {"lookback": 120, "n": 7, "buffer": 5, "stock_cap": 0.25, "sector_cap": 0.60, "semis_cap": 0.60}
# watchlist.py labels that share the semiconductor budget (unclassified names count too).
SEMIS_SECTORS = {"AI 반도체", "메모리", "반도체 장비", "반도체 장비(적자)", "반도체(적자)", "반도체 소재"}
DEFAULT_SETTINGS = {"tolerance": 0.02, "finviz_orders": True}

KST = ZoneInfo("Asia/Seoul")
NY = ZoneInfo("America/New_York")
REFERENCES = ["QQQ", "KRW=X", "456600.KS"]

CALENDAR_YEARS_AHEAD = 3   # exchange_calendars stops about a year ahead by default


@lru_cache(maxsize=1)
def market_calendar():
    """The one US market calendar (exchange_calendars XNYS) for every desk module.

    Holidays come from XNYS's rules, so they never run out (the hand list used before ended in 2027)."""
    import exchange_calendars
    end = date.today().replace(month=12, day=31)
    return exchange_calendars.get_calendar("XNYS", end=end.replace(year=end.year + CALENDAR_YEARS_AHEAD).isoformat())


def is_us_session(day: date) -> bool:
    """Regular NYSE session (full or early close). Past the calendar's end only weekdays count."""
    cal = market_calendar()
    stamp = day.isoformat()
    if cal.first_session.date() <= day <= cal.last_session.date():
        return bool(cal.is_session(stamp))
    return day.weekday() < 5


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
