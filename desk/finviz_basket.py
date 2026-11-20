"""Immutable month-opening FINVIZ baskets used by the SOXX defensive strategy."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.parse import parse_qs, urlsplit

import requests

from . import config
from .news import market_calendar

DIRECTORY = config.ROOT / "data" / "normalized" / "desk_finviz"
SCREEN_URL = ("https://finviz.com/screener?v=111&f=cap_midover%2Cfa_curratio_o1.5%2Cfa_debteq_u0.5"
              "%2Cgeo_usa%2Csh_avgvol_o500%2Csh_instown_o40%2Csh_price_o10%2Csh_relvol_o1"
              "%2Cta_rsi_nos40%2Cta_sma20_sa50%2Cta_sma200_pa%2Cta_sma50_pa&ft=4&ta=0&o=-relativevolume")
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SYMBOL = re.compile(r"[A-Z][A-Z0-9]{0,8}(?:-[A-Z0-9]{1,3})?")
SSR = re.compile(r'<script\b[^>]*\bdata-ssr-props=[^>]+>(.*?)</script>', re.I | re.S)


class BasketError(ValueError):
    pass


def month_dates(day: str | date) -> tuple[str, str]:
    """Return last month's signal session and this month's first XNYS session."""
    day = date.fromisoformat(day) if isinstance(day, str) else day
    calendar = market_calendar()
    first = calendar.date_to_session(day.replace(day=1).isoformat(), direction="next")
    return calendar.previous_session(first).date().isoformat(), first.date().isoformat()


def _window(signal: str) -> tuple[datetime, datetime]:
    day = date.fromisoformat(signal)
    # Public FINVIZ regular-market quotes are delayed. Keep the capture to the
    # close and its overnight window; a later weekend filing can change the scan.
    close = market_calendar().session_close(signal).to_pydatetime()
    return close + timedelta(minutes=25), datetime.combine(day + timedelta(days=1), time(4), config.NY)


def _check(record: dict, signal: str, effective: str) -> dict:
    try:
        symbols = record["symbols"]
        source = record["source"]
        captured = datetime.fromisoformat(record["captured_at"])
        valid = (source in ("research_reconstruction", "finviz_close") and captured.tzinfo is not None
                 and record["signal_date"] == signal and record["effective_date"] == effective
                 and record["screen_url"] == SCREEN_URL and isinstance(symbols, list)
                 and len(symbols) <= 7 and all(isinstance(s, str) and SYMBOL.fullmatch(s) for s in symbols)
                 and len(set(symbols)) == len(symbols) and record["cash_slots"] == 7 - len(symbols))
        if source == "finviz_close":
            start, end = _window(signal)
            valid = valid and start <= captured < end
    except (KeyError, ValueError, TypeError) as exc:
        raise BasketError(f"{effective} FINVIZ 월초 기록을 해석할 수 없습니다") from exc
    if not valid:
        raise BasketError(f"{effective} FINVIZ 월초 기록의 기준일·종목·출처를 확인해야 합니다")
    return record


def load_basket(execution_day: str | date, directory: Path | None = None) -> dict | None:
    day = date.fromisoformat(execution_day) if isinstance(execution_day, str) else execution_day
    signal, effective = month_dates(day)
    path = (directory or DIRECTORY) / (effective[:7] + ".json")
    if day.isoformat() < effective:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise BasketError(f"{effective} FINVIZ 월초 기록을 읽을 수 없습니다") from exc
    return _check(data, signal, effective)


def _parse_screen(html: str, now: datetime, signal: str) -> list[str]:
    # Parse the site's own ranked first-page data, including explicit empty
    # results. A login, rate limit, or HTML change is never a zero-stock month.
    for match in SSR.finditer(html):
        page = json.loads(match[1])
        if page.get("screenerType") == "stocks":
            break
    else:
        raise BasketError("FINVIZ 종목 검색 결과를 확인할 수 없습니다")
    query = parse_qs(urlsplit(page["url"]).query)
    if query != parse_qs(urlsplit(SCREEN_URL).query) or page.get("origin") != "https://finviz.com":
        raise BasketError("FINVIZ 검색 조건 또는 정렬이 다릅니다")
    info_time = datetime.fromisoformat(page["today"])
    if info_time.tzinfo is None:
        info_time = info_time.replace(tzinfo=config.NY)
    start, end = _window(signal)
    if not start <= info_time < end or abs(info_time - now) > timedelta(minutes=10):
        raise BasketError("FINVIZ 응답 시각이 월말 종가 수집 시간과 다릅니다")
    data = page["data"]
    total, rows = data["totalCount"], data["rows"]
    if (type(total) is not int or total < 0 or data["firstRow"] != 0 or data["pageSize"] < 7
            or not isinstance(rows, list) or len(rows) != min(total, data["pageSize"])):
        raise BasketError("FINVIZ 상위 7개 순위를 온전히 확인할 수 없습니다")
    symbols = []
    for rank, row in enumerate(rows[:7]):
        symbol = row["text"]["ticker"].upper().replace(".", "-")
        if (row["numbers"]["order"] != rank or not SYMBOL.fullmatch(symbol) or symbol in symbols
                or row["numbers"]["price"] <= 0 or row["text"]["country"] != "USA"):
            raise BasketError("FINVIZ 종목과 상대거래량 정렬을 확인해야 합니다")
        symbols.append(symbol)
    return symbols


def _fetch(now: datetime, signal: str) -> tuple[list[str], list[str]]:
    from .erules import SEC_HEADERS
    try:
        response = requests.get(SCREEN_URL, headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"},
                                timeout=12)
        response.raise_for_status()
        ranked = _parse_screen(response.text, now, signal)
        if not ranked:
            return [], []
        response = requests.get(SEC_TICKERS, headers=SEC_HEADERS, timeout=12)
        response.raise_for_status()
        issuers = {r["ticker"].replace(".", "-"): int(r["cik_str"]) for r in response.json().values()}
        seen = set()
        held = []
        for s in ranked:
            cik = issuers[s]
            if cik <= 0:
                raise BasketError("SEC 발행사 식별자가 유효하지 않습니다")
            if cik not in seen:
                held.append(s)
                seen.add(cik)
        return ranked, held
    except BasketError:
        raise
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise BasketError("FINVIZ 검색 결과 또는 SEC 발행사 확인에 실패했습니다") from exc


def _freeze(record: dict, directory: Path) -> dict:
    signal, effective = month_dates(record["effective_date"])
    _check(record, signal, effective)
    path = directory / (effective[:7] + ".json")
    temp = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as file:
            temp = Path(file.name)
            json.dump(record, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        try:
            # Publish the entire file once. A concurrent desk process can finish
            # the same capture first; subsequent refreshes keep that exact record.
            os.link(temp, path)
        except FileExistsError:
            return load_basket(effective, directory)
        return record
    except OSError as exc:
        raise BasketError("FINVIZ 월초 명단을 저장하지 못했습니다") from exc
    finally:
        if temp:
            temp.unlink(missing_ok=True)


def capture_due(now: datetime | None = None, directory: Path | None = None) -> dict | None:
    """Capture only inside the final session's closing window. Never backfill a missed scan."""
    now = now or datetime.now(config.NY)
    if now.tzinfo is None:
        raise BasketError("FINVIZ 수집 시각에 시간대가 필요합니다")
    local = now.astimezone(config.NY)
    day = local.date()
    if local.time() < time(4):
        day -= timedelta(days=1)
    first_next_month = (day.replace(day=28) + timedelta(days=4)).replace(day=1)
    signal, effective = month_dates(first_next_month)
    if signal != day.isoformat():
        return None
    start, end = _window(signal)
    if not start <= now < end:
        return None
    directory = directory or DIRECTORY
    saved = load_basket(effective, directory)
    if saved is not None:
        return saved
    ranked, held = _fetch(now, signal)
    return _freeze({"source": "finviz_close", "signal_date": signal, "effective_date": effective,
                    "captured_at": now.isoformat(timespec="seconds"), "screen_url": SCREEN_URL,
                    "ranked_symbols": ranked, "symbols": held, "cash_slots": 7 - len(held)}, directory)


if __name__ == "__main__":
    try:
        record = capture_due()
    except BasketError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(record, ensure_ascii=False, indent=2) if record is not None
          else "월말 미국장 마감 후 FINVIZ 수집 가능 시간이 아닙니다.")
