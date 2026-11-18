"""Frozen, offline inputs and a share-and-cash audit of the live desk model."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from desk import strategies as desk

END = "2026-11-17"
FROZEN = HERE / "frozen"
HOLDINGS = ROOT / "data/normalized/etf_holdings"
FIELDS = {"Adj Close": "prices_adj_close.csv", "Open": "prices_open.csv", "Close": "prices_close.csv"}


def _digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_book():
    """Use the desk's canonical selection with the captured price/snapshot versions."""
    sources = json.loads((FROZEN / "sources.json").read_text(encoding="utf-8"))
    for name, digest in sources["files"].items():
        if _digest(FROZEN / name) != digest:
            raise ValueError(f"Frozen price input changed: {name}")
    for fund, hashes in sources["holdings"].items():
        paths = sorted(p for p in HOLDINGS.glob(f"*_{fund}.json") if p.name[:10] <= END)
        if {p.name: _digest(p) for p in paths} != hashes:
            raise ValueError(f"Snapshot set changed for {fund}; restore the archived Git revision")
    frames = {field: pd.read_csv(FROZEN / name, index_col=0, parse_dates=True)
              for field, name in FIELDS.items()}
    snapshots = desk.load_fund_snapshots(desk.TIME_FUND, END)
    with patch.object(desk.marketdata, "history", return_value=frames), \
            patch.object(desk, "load_fund_snapshots", return_value=snapshots):
        book = desk.load_book()
    if book.days[-1] != END or book.cls.loc[:, ["QQQ", "SOXX"]].isna().any().any():
        raise ValueError("Incomplete market data")
    return book


def audit():
    """Independently price weekly targets with actual shares, USD cash and daily marks."""
    book = load_book()
    desk_path = desk.run_model(book)
    shares, cash, closes = {}, 1.0, []
    for day in desk_path.days:
        opens, prices = book.opn.loc[day], book.cls.loc[day]
        nav = cash + sum(qty * opens[symbol] for symbol, qty in shares.items())
        if day in desk_path.rebalances:
            weights = desk_path.rebalances[day]
            shares = {book.ticker[key]: nav * weight / opens[book.ticker[key]] for key, weight in weights.items()}
            cash = nav * (1 - sum(weights.values()))
        closes.append(cash + sum(qty * prices[symbol] for symbol, qty in shares.items()))
    np.testing.assert_allclose(closes, desk_path.close, atol=1e-12, rtol=1e-12)
    usd = np.array([1., *closes])
    krw = np.r_[1., desk.model_index(book, desk_path) / 100]
    split = desk_path.days.index("2026-06-30") + 1
    metrics = {}
    for label, curve in [("USD", usd), ("KRW", krw)]:
        metrics[label] = {}
        for name, segment in [("H1", curve[:split+1]), ("July_on", curve[split:]), ("YTD", curve)]:
            metrics[label][name] = {"return_pct": float((segment[-1] / segment[0] - 1) * 100),
                                    "mdd_pct": float((segment / np.maximum.accumulate(segment) - 1).min() * 100)}
    return {"first_session": desk_path.days[0], "last_session": desk_path.days[-1],
            "first_rebalance": min(desk_path.rebalances), "sessions": len(desk_path.days),
            "weekly_targets": len(desk_path.rebalances), "max_abs_usd_error": float(np.max(abs(closes - desk_path.close))),
            "baseline": metrics}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
