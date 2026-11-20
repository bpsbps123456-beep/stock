"""Next-open SOXX regime and the month-first FINVIZ shadow portfolio."""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from . import config
from .news import market_calendar


def decision(book, weekly, session, completed, basket, basket_issue=None):
    """Only completed split-adjusted closes. Cash is one original slot per missing name."""
    cal = market_calendar()
    previous = cal.previous_session(session)
    signal = previous.date().isoformat()
    result = dict(mode="blocked", order_due=False, signal_date=signal, close=None, sma100=None,
                  basket_month=session[:7], effective_date=None, source=None, reason="",
                  new_targets=[], new_cash_target=None, basket_symbols=[])
    if basket:
        result.update(effective_date=basket["effective_date"], source=basket["source"],
                      basket_symbols=basket["symbols"], basket_signal_date=basket["signal_date"])
    if completed != signal or book.days[-1] != signal or "SOXX" not in book.cls:
        result["reason"] = "직전 미국 정규장 종가가 확정된 뒤 시장 신호를 계산합니다."
        return result
    # Reindex against actual exchange sessions. A missing bar cannot silently shorten the window.
    start = cal.session_offset(previous, -100)
    window = cal.sessions_in_range(start, previous).strftime("%Y-%m-%d")
    closes = book.cls["SOXX"].reindex(window).to_numpy(float)
    if not (np.isfinite(closes) & (closes > 0)).all():
        result["reason"] = "SOXX 종가가 누락되어 100일선과 전환 여부를 확인할 수 없습니다."
        return result
    close, mean = float(closes[-1]), float(closes[1:].mean())
    on = close >= mean
    switched = on != (closes[-2] >= closes[:-1].mean())
    result.update(close=close, sma100=mean)
    if on:
        if not weekly["target"] and not weekly["ranked"]:
            result["reason"] = "기존 주간 전략의 목표를 확인할 수 없습니다."
            return result
        result.update(mode="desk", order_due=bool(switched or weekly["rebalance_day"]),
                      reason="100일선 회복 · 주간 전략으로 복귀" if switched else
                      ("주간 전략 정기 교체" if weekly["rebalance_day"] else "주간 전략 보유"))
        weights = {book.ticker[k]: v for k, v in weekly["target"].items()}
    else:
        if not basket:
            result["reason"] = basket_issue or "이달 월초에 고정한 FINVIZ 명단이 없습니다."
            return result
        symbols, first = basket["symbols"], basket["effective_date"]
        weight = 1 / 7
        weights = dict.fromkeys(symbols, weight)
        if session != first and symbols:
            # A switch between reviews uses the basket's drift since the first auction.
            # That upcoming open is still unknown. Show the prior close estimate in the draft.
            if first not in book.opn.index or not set(symbols).issubset(book.opn.columns.intersection(book.cls.columns)):
                result["reason"] = "월초 시가 또는 종가가 없어 FINVIZ 목표 비중 산출을 보류합니다."
                return result
            opens = book.opn.loc[first, symbols].to_numpy(float)
            marks = book.cls.loc[signal, symbols].to_numpy(float)
            if not (np.isfinite([opens, marks]) & (np.array([opens, marks]) > 0)).all():
                result["reason"] = "월초 시가 또는 직전 종가가 누락된 FINVIZ 종목이 있습니다."
                return result
            values = marks / opens
            total = 7 - len(symbols) + values.sum()
            weights = dict(zip(symbols, values / total))
        result.update(mode="finviz", order_due=bool(switched or session == first),
                      reason="100일선 이탈 · 월초 FINVIZ로 전환" if switched else
                      ("월 첫 거래일 · FINVIZ 명단 갱신" if session == first else "월초 FINVIZ 보유"))
    result.update(new_targets=[dict(ticker=tk, weight=w) for tk, w in weights.items()],
                  new_cash_target=max(0., 1. - sum(weights.values())))
    return result


def record_decision(signal, weekly_targets, session, now, directory=None):
    """Keep the first available pre-open observation independent of personal account settings."""
    if signal["mode"] == "blocked" or now >= market_calendar().session_open(session):
        return
    folder = Path(directory) if directory is not None else config.CACHE / "finviz_observations"
    dest = folder / (session + ".json")
    if dest.exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    observation = {k: v for k, v in signal.items() if k != "applied"}
    observation.update(execution_session=session, observed_at=now.isoformat(timespec="seconds"),
                       weekly_targets=weekly_targets)
    with TemporaryDirectory(dir=folder) as tmp:
        draft = Path(tmp) / "observation.json"
        draft.write_text(json.dumps(observation, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        try:
            if os.name == "nt":
                draft.rename(dest)
            else:
                os.link(draft, dest)
        except FileExistsError:
            pass


def research_comparison():
    path = config.ROOT / "data/derived/analysis/desk_etf_validation_20261120/metrics.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)
