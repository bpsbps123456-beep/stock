"""다음 정규장 전에 발행한 SOXX 예상 구간을 날짜별로 한 번만 보존하고 채점한다."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .common import write_once
from .config import market_calendar


# A test server (DESK_JOURNAL_DIR) locks its outlooks under its scratch folder (config.record_dir).
PAPER_DIR = config.record_dir(config.ROOT / "data" / "news_evidence" / "bottom_outlooks")


def latest_completed(now: datetime) -> str:
    calendar = market_calendar()
    day = calendar.date_to_session(now.astimezone(config.NY).date(), direction="previous")
    if calendar.session_close(day) > now:
        day = calendar.previous_session(day)
    return day.date().isoformat()


def _lock_forecast(folder: Path, outlook: dict, now: datetime) -> None:
    day = outlook["as_of"]
    path = folder / f"{day}.json"
    if path.exists():
        return
    row = {k: outlook[k] for k in ("as_of", "close", "horizon", "lower", "center", "upper", "method")}
    row["recorded_at"] = now.isoformat(timespec="seconds")
    # Only publish a complete file. Two servers refreshing at once keep the first lock.
    write_once(path, row)


def record_and_score(outlook: dict, close: pd.Series, now: datetime, folder: Path = PAPER_DIR) -> dict:
    cal = market_calendar()
    as_of = latest_completed(now)
    if outlook.get("available") and outlook["as_of"] == as_of:
        if now < cal.session_open(cal.next_session(as_of)):
            _lock_forecast(folder, outlook, now)
    result = {"hits": 0, "n": 0, "pending": 0, "unscored": 0}
    for path in sorted(folder.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        days = cal.sessions_window(row["as_of"], row["horizon"] + 1)
        if days[-1].date().isoformat() > as_of:
            result["pending"] += 1
            continue
        prices = close.reindex(days.tz_localize(None))
        if not np.isfinite(prices).all() or (prices <= 0).any():
            result["unscored"] += 1
            continue
        # Anchor-relative pricing preserves the recorded units across subsequent splits.
        low = prices.iloc[1:].min() * row["close"] / prices.iloc[0]
        result["hits"] += int(row["lower"] <= low <= row["upper"])
        result["n"] += 1
    return result
