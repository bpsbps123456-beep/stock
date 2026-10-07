"""The user's own trade journal: entries, positions (average cost), cash, and history.

Entries live in data/journal/trades.json (git-ignored, OneDrive-synced). Amounts are USD.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from datetime import date, datetime

import numpy as np

from . import config

TYPES = {"buy": "매수", "sell": "매도", "deposit": "입금", "withdraw": "출금"}


def _read(path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_entries() -> list[dict]:
    return _read(config.TRADES, {"version": 1, "entries": []})["entries"]


def load_settings() -> dict:
    """Known settings only; keys left over from retired strategies are ignored."""
    stored = _read(config.SETTINGS, {})
    return {key: stored.get(key, default) for key, default in config.DEFAULT_SETTINGS.items()}


def save_settings(changes: dict) -> dict:
    current = load_settings()
    if "tolerance" in changes:
        current["tolerance"] = min(0.2, max(0.0, float(changes["tolerance"])))
    _write(config.SETTINGS, current)
    return current


class JournalError(ValueError):
    pass


def add_entry(payload: dict) -> dict:
    kind = payload.get("type")
    if kind not in TYPES:
        raise JournalError("기록 종류를 매수·매도·입금·출금 중에서 선택하세요.")
    try:
        day = date.fromisoformat(str(payload.get("date"))).isoformat()
    except ValueError as exc:
        raise JournalError("날짜를 연-월-일 형식(YYYY-MM-DD)으로 입력하세요.") from exc
    entry = {"id": uuid.uuid4().hex[:10], "date": day, "type": kind,
             "note": str(payload.get("note", ""))[:200],
             "created_at": datetime.now(config.KST).isoformat(timespec="seconds")}
    if kind in ("buy", "sell"):
        ticker = str(payload.get("ticker", "")).strip().upper()
        qty, price = float(payload.get("qty") or 0), float(payload.get("price") or 0)
        fee = float(payload.get("fee") or 0)
        if not ticker or qty <= 0 or price <= 0 or fee < 0:
            raise JournalError("종목 코드를 입력하고, 수량과 단가가 0보다 큰지 확인하세요.")
        entry.update({"ticker": ticker, "qty": qty, "price": price, "fee": fee})
    else:
        amount = float(payload.get("amount") or 0)
        if amount <= 0:
            raise JournalError("입금·출금 금액은 0보다 커야 합니다.")
        entry["amount"] = amount
    entries = load_entries()
    trial = entries + [entry]
    compute(trial, {})  # raises if a sell exceeds the held quantity
    _write(config.TRADES, {"version": 1, "entries": trial})
    return entry


def delete_entry(entry_id: str) -> bool:
    entries = load_entries()
    kept = [e for e in entries if e["id"] != entry_id]
    if len(kept) == len(entries):
        return False
    compute(kept, {})
    _write(config.TRADES, {"version": 1, "entries": kept})
    return True


def _ordered(entries):
    return sorted(entries, key=lambda e: (e["date"], e.get("created_at", "")))


def compute(entries: list[dict], prices: dict[str, float]) -> dict:
    """Average-cost positions, realized P&L, cash, and closed round trips."""
    pos, cash, realized, contributed, trips = {}, 0.0, 0.0, 0.0, []
    for e in _ordered(entries):
        if e["type"] == "deposit":
            cash += e["amount"]
            contributed += e["amount"]
        elif e["type"] == "withdraw":
            cash -= e["amount"]
            contributed -= e["amount"]
        elif e["type"] == "buy":
            p = pos.setdefault(e["ticker"], {"qty": 0.0, "cost": 0.0, "opened": e["date"], "buys": 0})
            if p["qty"] <= 1e-12:
                p.update({"qty": 0.0, "cost": 0.0, "opened": e["date"], "buys": 0})
            p["qty"] += e["qty"]
            p["cost"] += e["qty"] * e["price"] + e.get("fee", 0.0)
            p["buys"] += 1
            cash -= e["qty"] * e["price"] + e.get("fee", 0.0)
        elif e["type"] == "sell":
            p = pos.get(e["ticker"])
            if not p or e["qty"] > p["qty"] + 1e-9:
                raise JournalError(f"{e['date']} {e['ticker']} 매도 수량이 보유 수량보다 많습니다")
            avg = p["cost"] / p["qty"]
            gain = e["qty"] * (e["price"] - avg) - e.get("fee", 0.0)
            realized += gain
            p["cost"] -= avg * e["qty"]
            p["qty"] -= e["qty"]
            cash += e["qty"] * e["price"] - e.get("fee", 0.0)
            if p["qty"] <= 1e-9:
                trips.append({"ticker": e["ticker"], "opened": p["opened"], "closed": e["date"],
                              "gain": gain})
                p["qty"], p["cost"] = 0.0, 0.0
    positions = []
    market_value = 0.0
    for ticker, p in pos.items():
        if p["qty"] <= 1e-9:
            continue
        price = prices.get(ticker)
        value = p["qty"] * price if price else math.nan
        market_value += value if np.isfinite(value) else 0.0
        avg = p["cost"] / p["qty"]
        positions.append({"ticker": ticker, "qty": p["qty"], "avg_cost": avg, "price": price,
                          "value": value, "cost": p["cost"],
                          "pnl": value - p["cost"] if np.isfinite(value) else math.nan,
                          "pnl_pct": value / p["cost"] - 1 if np.isfinite(value) and p["cost"] else math.nan,
                          "opened": p["opened"], "buys": p["buys"]})
    # Without deposit records cash goes negative; value the account as stocks + positive cash.
    equity = market_value + max(cash, 0.0)
    for row in positions:
        row["weight"] = row["value"] / equity if equity > 0 and np.isfinite(row["value"]) else math.nan
    positions.sort(key=lambda r: -(r["value"] if np.isfinite(r["value"]) else 0))
    unrealized = sum(r["pnl"] for r in positions if np.isfinite(r["pnl"]))
    return {"positions": positions, "cash": cash, "market_value": market_value, "equity": equity,
            "realized": realized, "unrealized": unrealized, "contributed": contributed,
            "trips": trips[::-1], "funded": contributed > 0}


def history(entries: list[dict], closes, days: list[str], fx: np.ndarray) -> dict:
    """Daily end-of-day account value (USD and KRW) from the first entry onward.

    `twr` is a time-weighted index: deposits and withdrawals count at the start of their
    day, so moving money never shows up as a gain or a drawdown. It is NaN from the first
    day the money base is unknown (a buy before any deposit).
    """
    if not entries:
        return {"dates": [], "usd": [], "krw": [], "twr": []}
    ordered = _ordered(entries)
    first = ordered[0]["date"]
    out_d, out_u, out_k, out_t = [], [], [], []
    qty: dict[str, float] = {}
    cash, i = 0.0, 0
    prev, index = 0.0, 1.0
    for k, day in enumerate(days):
        if day < first:
            continue
        flow = 0.0
        while i < len(ordered) and ordered[i]["date"] <= day:
            e = ordered[i]
            if e["type"] == "deposit":
                cash += e["amount"]
                flow += e["amount"]
            elif e["type"] == "withdraw":
                cash -= e["amount"]
                flow -= e["amount"]
            elif e["type"] == "buy":
                qty[e["ticker"]] = qty.get(e["ticker"], 0.0) + e["qty"]
                cash -= e["qty"] * e["price"] + e.get("fee", 0.0)
            elif e["type"] == "sell":
                qty[e["ticker"]] = qty.get(e["ticker"], 0.0) - e["qty"]
                cash += e["qty"] * e["price"] - e.get("fee", 0.0)
            i += 1
        value = cash
        for ticker, q in qty.items():
            if abs(q) < 1e-9:
                continue
            px = closes[ticker].iloc[k] if ticker in closes.columns else np.nan
            value += q * px if np.isfinite(px) else 0.0
        base = prev + flow
        index = index * value / base if base > 0 and math.isfinite(index) else math.nan
        prev = value
        out_d.append(day)
        out_u.append(value)
        out_k.append(value * fx[k] if np.isfinite(fx[k]) else math.nan)
        out_t.append(index)
    return {"dates": out_d, "usd": out_u, "krw": out_k, "twr": out_t}
