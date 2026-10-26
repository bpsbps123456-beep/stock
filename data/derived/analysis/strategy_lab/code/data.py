"""Strategy-lab data layer: both funds' US-listed holdings + prices on one calendar.

Timing contract (no look-ahead):
- PDF dated D is published ~21:45 KST on D, before the US session of US date D.
- A follower decides on US day t with the latest PDF dated <= t and closes up to t-1,
  and trades at the US close of t. Positions earn returns from close t to close t+1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[5]  # <repo>/data/derived/analysis/strategy_lab/code
sys.path.insert(0, str(REPO))

from gbrc.etf import flows, market  # noqa: E402
from gbrc.etf.trades import load_fund_snapshots  # noqa: E402

FUNDS = {"T": "456600", "K": "2ETFQ1"}
ETF_TICKERS = {"T": "456600.KS", "K": "0015B0.KS"}
REF_TICKERS = ["QQQ", "SPY", "SMH", "^VIX", "KRW=X", "456600.KS", "0015B0.KS"]
FLOW_WINDOWS = (1, 3, 5, 10, 20)
PRICE_START = "2025-05-01"
PRICE_END = "2026-09-27"
LAST_SNAPSHOT = "2026-09-23"
CACHE = REPO / "data" / "derived" / "analysis" / "strategy_lab"
PRICE_FIELDS = ("Adj Close", "Close", "Open")
# Same company under two source tickers: TIME used ECHO (Jan-Feb), KoAct used the old
# SATS (May-Jun). Yahoo only serves ECHO, so both map to one company and price.
COMPANY_ALIASES = {"SATS US EQUITY": "ECHO US EQUITY"}


def _company_rows(snapshot):
    """US-listed rows keyed by company (GOOG=GOOGL), weights summed."""
    out = {}
    for row in flows.keyed_securities(snapshot).values():
        if not market.is_us_listed(row["code"]):
            continue
        company = market.company_key(row["key"])
        company = COMPANY_ALIASES.get(company, company)
        cur = out.setdefault(company, {"weight": 0.0, "codes": set()})
        cur["weight"] += row["weight"]
        cur["codes"].add(row["code"])
    return out


def build_fund_panels(snapshots_by_fund):
    """Per fund: weights, windowed flows (trade %p), new-entry flags per PDF date."""
    panels = {}
    for tag, snaps in snapshots_by_fund.items():
        dates = sorted(snaps)
        rows = {d: _company_rows(snaps[d]) for d in dates}
        weights = {d: {c: r["weight"] for c, r in rows[d].items()} for d in dates}
        codes = {}
        for d in dates:
            for c, r in rows[d].items():
                codes.setdefault(c, set()).update(r["codes"])
        flow_pp = {L: {} for L in FLOW_WINDOWS}
        new_in = {L: {} for L in FLOW_WINDOWS}
        for i, d in enumerate(dates):
            for L in FLOW_WINDOWS:
                j = max(0, i - L)
                if j == i:
                    flow_pp[L][d], new_in[L][d] = {}, []
                    continue
                result = flows.quantity_flows([snaps[x] for x in dates[j:i + 1]])
                pp, new = {}, []
                for flow in result.flows.values():
                    if not market.is_us_listed(flow.code):
                        continue
                    company = market.company_key(flow.key)
                    company = COMPANY_ALIASES.get(company, company)
                    if flow.trade_pp is not None:
                        pp[company] = pp.get(company, 0.0) + flow.trade_pp
                    if flow.kind == "신규":
                        new.append(company)
                flow_pp[L][d], new_in[L][d] = pp, sorted(set(new))
        metrics = {d: snaps[d].get("metrics", {}) for d in dates}
        panels[tag] = {"dates": dates, "weights": weights,
                       "flow_pp": {str(L): v for L, v in flow_pp.items()},
                       "new_in": {str(L): v for L, v in new_in.items()},
                       "codes": {c: sorted(v) for c, v in codes.items()},
                       "metrics": metrics}
    return panels


def price_ticker(codes):
    """One Yahoo symbol per company; prefer GOOGL for the merged share class."""
    symbols = sorted({market.yahoo_ticker(c) for c in codes if market.yahoo_ticker(c)})
    if "GOOGL" in symbols:
        return "GOOGL"
    if "ECHO" in symbols:
        return "ECHO"
    return symbols[0] if symbols else None


def download(tickers):
    import yfinance as yf

    frame = yf.download(sorted(set(tickers)), start=PRICE_START, end=PRICE_END,
                        auto_adjust=False, progress=False, threads=True)
    return {field: frame[field].dropna(axis=1, how="all") for field in PRICE_FIELDS}


def _price_path(field):
    return CACHE / f"prices_{field.replace(' ', '_').lower()}.csv"


def load(refresh=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE / "lab_panels.json"
    if meta_path.exists() and not refresh:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        data["prices"] = {f: pd.read_csv(_price_path(f), index_col=0, parse_dates=True)
                          for f in PRICE_FIELDS}
        return data

    snaps = {tag: load_fund_snapshots(fid, LAST_SNAPSHOT) for tag, fid in FUNDS.items()}
    assert sorted(snaps["T"]) == sorted(snaps["K"]), "fund snapshot dates differ"
    panels = build_fund_panels(snaps)

    companies = sorted(set(panels["T"]["codes"]) | set(panels["K"]["codes"]))
    tick = {}
    for c in companies:
        codes = set(panels["T"]["codes"].get(c, [])) | set(panels["K"]["codes"].get(c, []))
        tick[c] = price_ticker(codes)
    prices = download([t for t in tick.values() if t] + REF_TICKERS)
    for field, frame in prices.items():
        frame.to_csv(_price_path(field))

    data = {"panels": panels, "companies": companies, "ticker": tick,
            "pdf_dates": sorted(snaps["T"])}
    meta_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    data["prices"] = prices
    return data


if __name__ == "__main__":
    d = load(refresh="--refresh" in sys.argv)
    adj = d["prices"]["Adj Close"]
    missing = [c for c, t in d["ticker"].items() if t not in adj.columns]
    print("companies", len(d["companies"]), "priced", len(d["companies"]) - len(missing))
    print("missing price:", [(c, d["ticker"][c]) for c in missing])
    print("price range", adj.index.min().date(), adj.index.max().date(), adj.shape)
