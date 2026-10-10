"""The desk's one strategy: TIME momentum top 7 (Codex concentration study #774).

Point in time: the decision for US session t uses TIME PDFs dated strictly before t and
closes through the previous US session; positions change at t's open (Codex's
"publication lag 1"). Model paths run open-to-open on split-adjusted opens (USD), are
marked at each close and converted to KRW with KRW=X. The TIME ETF trades in Seoul, so
US session t is compared with the first KRX close after it (the one that prices day t).

Rule (docs/desk_system.md §3): on the first US session of each ISO week rank TIME's US
holdings by 120-session adjusted-close return, keep last week's names while they stay in
the top 12 and fill to 7 from the top, weight them by TIME's NAV weights, then cap each
name at 25%, each watchlist sector at 60% and known semiconductors + unclassified at 60%.
Capped-out weight is re-spread over the chosen names; what still does not fit stays cash.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from gbrc.etf.market import company_key, is_cash_holding, listing_market, yahoo_ticker
from gbrc.etf.research_data import CORPORATE_ALIASES, exclusion_reason
from gbrc.etf.trades import load_fund_snapshots
from watchlist import resolve_sector, resolve_subsector

from . import config, marketdata

TIME_FUND = "456600"
UNKNOWN = "미분류"


def research_key(code) -> str:
    key = company_key(code)
    return CORPORATE_ALIASES.get(key, key)


def us_weights(snapshot) -> dict[str, float]:
    """NAV fractions of directly held US securities (not renormalised)."""
    out: dict[str, float] = {}
    for row in snapshot["holdings"]:
        if is_cash_holding(row) or exclusion_reason(row):
            continue
        key = research_key(row.get("code"))
        if listing_market(key) == "US":
            out[key] = out.get(key, 0.0) + float(row["weight"]) / 100.0
    return out


def sector_of(key: str, ticker: str) -> str:
    """watchlist.py label via the Yahoo symbol, the source code, then recorded renames."""
    renamed = [old for old, new in CORPORATE_ALIASES.items() if new == key]
    for symbol in (ticker, key.split()[0], *(old.split()[0] for old in renamed)):
        label = resolve_sector(symbol, default=UNKNOWN)
        if label != UNKNOWN:
            return label
    return UNKNOWN


def subsector_of(key: str, ticker: str) -> str:
    """Finer display label: the AI_UNIVERSE sub-sector, else the cap sector. Never used for caps."""
    renamed = [old for old, new in CORPORATE_ALIASES.items() if new == key]
    for symbol in (ticker, key.split()[0], *(old.split()[0] for old in renamed)):
        label = resolve_subsector(symbol)
        if label:
            return label
    return sector_of(key, ticker)


def in_semis_budget(sector: str) -> bool:
    return sector in config.SEMIS_SECTORS or sector == UNKNOWN


def iso_week(day: str):
    return date.fromisoformat(day).isocalendar()[:2]


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

@dataclass
class Book:
    days: list[str]              # US sessions (QQQ closes)
    adj: pd.DataFrame
    opn: pd.DataFrame
    cls: pd.DataFrame
    fx: np.ndarray
    snaps: dict                  # TIME snapshots by PDF date
    pdfs: list[str]
    keys: list[str]
    ticker: dict
    names: dict
    sector: dict
    etf: pd.Series               # TIME ETF 456600.KS, adjusted close (KRX dates)
    _w: dict = field(default_factory=dict)

    def weights(self, pdf: str) -> dict[str, float]:
        if pdf not in self._w:
            self._w[pdf] = us_weights(self.snaps[pdf])
        return self._w[pdf]


def load_book(force_prices: bool = False, extra: list[str] | None = None) -> Book:
    snaps = load_fund_snapshots(TIME_FUND, "9999-12-31")
    keys, names = set(), {}
    for snap in snaps.values():
        for row in snap["holdings"]:
            if is_cash_holding(row) or exclusion_reason(row):
                continue
            key = research_key(row.get("code"))
            if listing_market(key) == "US":
                keys.add(key)
                names.setdefault(key, row.get("name", ""))
    ticker = {k: yahoo_ticker(k) for k in keys}
    prices = marketdata.history(sorted(set(ticker.values()) | set(extra or [])) + config.REFERENCES,
                                force=force_prices)
    adj = prices["Adj Close"]
    days = [d.strftime("%Y-%m-%d") for d in adj["QQQ"].dropna().index]

    def on_days(df):
        out = df.copy()
        out.index = out.index.strftime("%Y-%m-%d")
        return out.reindex(days)

    return Book(days=days, adj=on_days(adj), opn=on_days(prices["Open"]), cls=on_days(prices["Close"]),
                fx=marketdata.ffill_on(adj["KRW=X"], days), snaps=snaps, pdfs=sorted(snaps),
                keys=sorted(keys, key=lambda k: ticker[k]), ticker=ticker, names=names,
                sector={k: sector_of(k, ticker[k]) for k in keys}, etf=adj["456600.KS"].dropna())


def latest_before(dates: list[str], day: str) -> str | None:
    i = bisect_left(dates, day) - 1
    return dates[i] if i >= 0 else None


def _col(book: Book, frame: pd.DataFrame, key: str) -> np.ndarray | None:
    tk = book.ticker[key]
    return frame[tk].to_numpy(float) if tk in frame.columns else None


# ---------------------------------------------------------------------------
# the rule (decision for the session at index k; k == len(days) means the next session)
# ---------------------------------------------------------------------------

def momentum_ranking(book: Book, k: int, session: str) -> tuple[str | None, list[dict]]:
    lookback = config.RULE["lookback"]
    pdf = latest_before(book.pdfs, session)
    if pdf is None or k - 1 - lookback < 0:
        return pdf, []
    w = book.weights(pdf)
    rows = []
    for key in sorted(w):
        if w[key] <= 0:
            continue
        px = _col(book, book.adj, key)
        if px is None:
            continue
        p1, p0 = px[k - 1], px[k - 1 - lookback]
        if not (np.isfinite(p1) and np.isfinite(p0) and p0 > 0):
            continue
        rows.append({"key": key, "ticker": book.ticker[key], "mom120": p1 / p0 - 1,
                     "time_weight": w[key], "sector": book.sector[key]})
    rows.sort(key=lambda r: -r["mom120"])  # stable: equal momentum keeps key order
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
    return pdf, rows


def select(ranked: list[dict], current: set[str]) -> list[str]:
    """Keep last week's names inside the top n + buffer, then fill to n from the top."""
    n, buf = config.RULE["n"], config.RULE["buffer"]
    order = [r["key"] for r in ranked]
    keep = [k for k in order[:n + buf] if k in current][:n]
    for k in order:
        if len(keep) >= n:
            break
        if k not in keep:
            keep.append(k)
    return keep


def project_caps(raw: dict[str, float], sector: dict[str, str]) -> dict[str, float]:
    """Proportional progressive fill under the stock, sector and semiconductor caps.

    Every name grows in proportion to its raw weight; a name stops when it, its sector
    or the semiconductor budget is full, and the others keep growing until the raw total
    is invested. Weight that fits nowhere stays cash (never renormalised to 100%).
    """
    stock, detail, semis = (config.RULE[k] for k in ("stock_cap", "sector_cap", "semis_cap"))
    names = [k for k, v in raw.items() if v > 0]
    group = {k: sector.get(k, UNKNOWN) for k in names}
    budgeted = {k for k in names if in_semis_budget(group[k])}
    alloc = dict.fromkeys(names, 0.0)
    target = sum(raw[k] for k in names)
    active = set(names)
    for _ in range(4 * len(names) + 8):
        used = sum(alloc.values())
        if target - used <= 1e-12:
            break
        in_group: dict[str, float] = {}
        for k in names:
            in_group[group[k]] = in_group.get(group[k], 0.0) + alloc[k]
        in_semis = sum(alloc[k] for k in budgeted)
        active = {k for k in active if stock - alloc[k] > 1e-12 and detail - in_group[group[k]] > 1e-12
                  and not (k in budgeted and semis - in_semis <= 1e-12)}
        if not active:
            break
        steps = [(target - used) / sum(raw[k] for k in active)]
        steps += [(stock - alloc[k]) / raw[k] for k in active]
        for g in {group[k] for k in active}:
            steps.append((detail - in_group[g]) / sum(raw[k] for k in active if group[k] == g))
        semis_rate = sum(raw[k] for k in active if k in budgeted)
        if semis_rate > 0:
            steps.append((semis - in_semis) / semis_rate)
        step = max(0.0, min(steps))
        for k in active:
            alloc[k] += step * raw[k]
    return {k: v for k, v in alloc.items() if v > 1e-12}


def weekly_target(book: Book, k: int, session: str, current: set[str]) -> dict:
    pdf, ranked = momentum_ranking(book, k, session)
    if not ranked:
        return {"pdf": pdf, "ranked": [], "selected": [], "raw": {}, "target": {}}
    sel = select(ranked, current)
    tw = {r["key"]: r["time_weight"] for r in ranked}
    total = sum(tw[key] for key in sel)
    raw = {key: tw[key] / total for key in sel}
    return {"pdf": pdf, "ranked": ranked, "selected": sel, "raw": raw, "target": project_caps(raw, book.sector)}


# ---------------------------------------------------------------------------
# model path
# ---------------------------------------------------------------------------

@dataclass
class Path:
    days: list[str]
    value: np.ndarray              # USD value at each session open, start 1.0
    close: np.ndarray              # USD value at each session close
    weights: list[dict]            # key -> weight after that open's trades
    events: list[dict]             # entries and exits
    rebalances: dict               # day -> target after that week's rebalance
    selected: set                  # names chosen at the last rebalance (the buffer state)
    last: dict                     # the last rebalance decision


def _open_returns(book: Book, keys, k: int) -> dict[str, float]:
    out = {}
    for key in keys:
        o = _col(book, book.opn, key)
        if o is None or not (np.isfinite(o[k]) and np.isfinite(o[k - 1]) and o[k - 1] > 0):
            out[key] = 0.0
        else:
            out[key] = o[k] / o[k - 1] - 1
    return out


def _drift(weights: dict[str, float], rets: dict[str, float]) -> tuple[dict[str, float], float]:
    gross = sum(w * rets.get(key, 0.0) for key, w in weights.items())
    cash = 1.0 - sum(weights.values())
    total = sum(w * (1 + rets.get(key, 0.0)) for key, w in weights.items()) + cash
    new = {key: w * (1 + rets.get(key, 0.0)) / total for key, w in weights.items()}
    return new, gross


def _price_at(book: Book, key: str, k: int, frame="opn") -> float | None:
    px = _col(book, getattr(book, frame), key)
    if px is None or k >= len(px) or not np.isfinite(px[k]):
        return None
    return float(px[k])


def run_model(book: Book, start: str = config.MODEL_START) -> Path:
    days = [d for d in book.days if d >= start]
    k0 = book.days.index(days[0])
    value, close, weights, events, rebalances = [1.0], [], [], [], {}
    w: dict[str, float] = {}
    selected: set[str] = set()
    last: dict = {}
    for k in range(k0, len(book.days)):
        day = book.days[k]
        if k > k0:
            w, gross = _drift(w, _open_returns(book, w, k))
            value.append(value[-1] * (1 + gross))
        if not last or iso_week(day) != iso_week(book.days[k - 1]):
            decision = weekly_target(book, k, day, selected)
            if decision["ranked"]:
                target = decision["target"]
                for key in set(w) - set(target):
                    events.append({"date": day, "key": key, "side": "sell", "price": _price_at(book, key, k)})
                for key in set(target) - set(w):
                    events.append({"date": day, "key": key, "side": "buy", "price": _price_at(book, key, k)})
                w, selected = dict(target), set(decision["selected"])
                rebalances[day] = dict(target)
                last = {"date": day, **decision}
        weights.append(dict(w))
        close.append(value[-1] * (1 + _drift(w, _intraday(book, w, k))[1]))
    return Path(days, np.array(value), np.array(close), weights, events, rebalances, selected, last)


def _intraday(book: Book, keys, k: int) -> dict[str, float]:
    out = {}
    for key in keys:
        o, c = _price_at(book, key, k, "opn"), _price_at(book, key, k, "cls")
        out[key] = (c / o - 1) if o and c else 0.0
    return out


def model_index(book: Book, path: Path) -> np.ndarray:
    """KRW value of the model at each close, 100 = the money before the first session."""
    k0 = book.days.index(path.days[0])
    fx = book.fx[k0 - 1:k0 + len(path.days)]
    return path.close * fx[1:] / fx[0] * 100


def etf_index(book: Book, days: list[str]) -> np.ndarray:
    """TIME ETF at the first KRX close after each US session (NaN until that close exists),
    100 at the first such close."""
    krx = book.etf.index.strftime("%Y-%m-%d").to_numpy()
    pos = np.searchsorted(krx, np.asarray(days), side="right")
    px = np.array([book.etf.iloc[p] if p < len(krx) else np.nan for p in pos], float)
    return px / px[np.flatnonzero(np.isfinite(px))[0]] * 100


def model_status(book: Book, path: Path, long_window: int) -> dict:
    """Model drawdown and its gaps to the TIME ETF on the last session the ETF has priced."""
    model, etf = model_index(book, path), etf_index(book, path.days)
    j = int(np.flatnonzero(np.isfinite(etf))[-1])
    curve = np.concatenate([[100.0], model])
    dd = curve / np.maximum.accumulate(curve) - 1

    def gap(n: int):
        return None if j < n else (model[j] / model[j - n] - 1) - (etf[j] / etf[j - n] - 1)

    return {"model": model, "etf": etf, "etf_last": j, "mdd": float(dd.min()), "dd_now": float(dd[-1]),
            "vs_etf_20d": gap(20), "vs_etf_long": gap(long_window)}


def next_target(book: Book, path: Path, session: str) -> dict:
    """Target for the next session: a new weekly target on rebalance days, otherwise the
    model's current holdings marked to the latest close."""
    k = len(book.days)
    rebalance = iso_week(session) != iso_week(book.days[-1])
    current, _ = _drift(path.weights[-1], _intraday(book, path.weights[-1], k - 1))
    decision = weekly_target(book, k, session, path.selected)
    if rebalance and decision["ranked"]:
        return {"session": session, "rebalance_day": True, **decision, "current": current}
    return {"session": session, "rebalance_day": False, "pdf": path.last.get("pdf"),
            "ranked": decision["ranked"], "selected": sorted(path.selected), "raw": {},
            "target": current, "current": current}
