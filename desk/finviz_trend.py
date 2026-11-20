"""Paper-only 30/40/175 FINVIZ variant registered on 2026-11-20."""
from __future__ import annotations

from datetime import timedelta
import math
from statistics import fmean
import time
from types import SimpleNamespace

import requests

from . import config, finviz_basket as basket_store, finviz_switch, marketdata

SCREEN_URL = ("https://finviz.com/screener?v=161&f=cap_midover%2Cfa_curratio_o1.5%2Cfa_debteq_u1"
              "%2Cgeo_usa%2Csh_avgvol_o500%2Csh_instown_o40%2Csh_price_o10%2Csh_relvol_o1"
              "%2Cta_rsi_nos40&ft=4&ta=0&o=-relativevolume")
SOURCE = "finviz_sip_close"
VARIANT = "sma30_40_175"
DIRECTORY = basket_store.DIRECTORY / VARIANT
SHORT, MEDIUM, LONG = 30, 40, 175


def observation_dir():
    return config.CACHE / "finviz_observations" / VARIANT


def _trend_passes(bars, sessions):
    if not bars or sessions[-1] not in bars:
        raise basket_store.BasketError("월말 SIP 종가가 누락되었습니다")
    days = [day for day in sessions if day in bars]
    if days != sessions[-len(days):]:
        raise basket_store.BasketError("이동평균에 필요한 SIP 종가가 중간에 누락되었습니다")
    if len(days) < LONG:
        return False
    closes = [bars[day][1] for day in days]
    if not all(math.isfinite(close) and close > 0 for close in closes):
        raise basket_store.BasketError("SIP 종가가 유효하지 않습니다")
    mid = fmean(closes[-MEDIUM:])
    return fmean(closes[-SHORT:]) > mid and closes[-1] > mid and closes[-1] > fmean(closes)


def _fetch(now, signal):
    cal = basket_store.market_calendar()
    start = cal.session_offset(signal, 1 - LONG)
    sessions = cal.sessions_in_range(start, signal).strftime("%Y-%m-%d").tolist()
    cutoff = now - timedelta(minutes=16)
    ranked, seen = [], set()
    first, total = 0, None
    try:
        while total is None or first < total:
            url = f"{SCREEN_URL}&r={first + 1}" if first else SCREEN_URL
            page = basket_store._screen_page(basket_store._get_screen(url), now, signal, url, first)
            if total is not None and total != page["totalCount"]:
                raise basket_store.BasketError("수집 중 FINVIZ 후보 수가 바뀌었습니다")
            total = page["totalCount"]
            rows = page["rows"]
            symbols = basket_store._screen_symbols(rows, first)
            if seen.intersection(symbols):
                raise basket_store.BasketError("수집 중 FINVIZ 순위가 바뀌었습니다")
            seen.update(symbols)
            # The financial view exposes numeric ratios for the study's strict boundaries.
            eligible = [symbol for symbol, row in zip(symbols, rows)
                        if row["numbers"]["curRatio"] > 1.5 and 0 <= row["numbers"]["debtEq"] < 1]
            # 20 symbols × 175 daily bars fits on one SIP response page.
            bars = marketdata._sip_basket_bars(eligible, sessions[0], signal, cutoff, full=True) if eligible else {}
            for symbol in eligible:
                if _trend_passes(bars.get(symbol), sessions):
                    ranked.append(symbol)
                if len(ranked) == 7:
                    return ranked, basket_store._distinct_issuers(ranked)
            first += len(rows)
        return ranked, basket_store._distinct_issuers(ranked)
    except basket_store.BasketError:
        raise
    except (requests.RequestException, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise basket_store.BasketError("30·40·175일선 FINVIZ, SIP 또는 SEC 자료를 확인할 수 없습니다") from exc


def capture_due(now=None, directory=None):
    return basket_store.capture_due(now, directory or DIRECTORY, screen_url=SCREEN_URL,
                                    live_source=SOURCE, fetch=_fetch)


class Paper:
    def __init__(self):
        self.capture_issue = None
        self.repair_key = None
        self.repair_at = None
        self.repaired = None

    def capture(self, now):
        try:
            capture_due(now)
            self.capture_issue = None
        except basket_store.BasketError as exc:
            self.capture_issue = str(exc)

    def observe(self, book, weekly, session, completed, now, weekly_targets):
        issue = self.capture_issue
        try:
            basket = basket_store.load_basket(session, DIRECTORY, screen_url=SCREEN_URL, live_source=SOURCE)
        except basket_store.BasketError as exc:
            basket, issue = None, str(exc)
        route = finviz_switch.decision(book, weekly, session, completed, basket, issue)
        if (route["mode"] == "blocked" and route["close"] is not None
                and route["close"] < route["sma100"] and basket):
            # Keep the candidate's extra price data outside the order engine's book.
            key = (basket["effective_date"], completed, id(book))
            stamp = time.monotonic()
            if key != self.repair_key or stamp - self.repair_at >= 300:
                columns = [*basket["symbols"], "SOXX"]
                copy = SimpleNamespace(days=book.days, ticker=book.ticker,
                                       cls=book.cls.reindex(columns=columns),
                                       opn=book.opn.reindex(columns=columns))
                sources = marketdata.repair_basket_bars(copy, basket, completed, now=now)
                self.repair_key, self.repair_at, self.repaired = key, stamp, (copy, sources)
            copy, sources = self.repaired
            if sources:
                route = finviz_switch.decision(copy, weekly, session, completed, basket, issue)
                route["bar_sources"] = sources
        if basket is None:
            route["basket_issue"] = issue or "이달에 고정한 관찰용 월초 명단이 없습니다."
        route["variant"] = VARIANT
        try:
            finviz_switch.record_decision(route, weekly_targets, session, now, observation_dir())
        except OSError:
            route["record_error"] = "이번 신호를 모의 관찰 기록에 저장하지 못했습니다."
        return route
