"""Seven independently scheduled capital slots, valued at US opens and closes."""
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import date

import numpy as np
import pandas as pd


def _classifications(values, shape):
    arr = np.asarray(values, dtype=object)
    if arr.shape == (shape[1],):
        arr = np.broadcast_to(arr, shape)
    if arr.shape != shape or pd.isna(arr).any():
        raise ValueError("Issuer and sector labels must cover all symbols and sessions")
    return arr


class Market:
    """Aligned daily arrays. Ranks and all membership data are supplied as of the open."""

    def __init__(self, days, symbols, opens, closes, fx, sectors, issuers=None,
                 settlements=None, tradable=None):
        self.days = [str(d)[:10] for d in days]
        self.symbols = tuple(symbols)
        if not self.days or self.days != sorted(set(self.days)) or len(symbols) != len(set(symbols)):
            raise ValueError("Sorted unique sessions and distinct traded symbols are required")
        shape = len(self.days), len(symbols)
        opn = np.asarray(opens, dtype=float)
        cls = np.asarray(closes, dtype=float)
        if opn.shape != shape or cls.shape != shape:
            raise ValueError("Quote panel shape must be session by symbol")
        self.sectors = _classifications(sectors, shape)
        if issuers is None:
            # SEC CIKs supplied by the main study cover other dual-class issuers.
            aliases = {"GOOG": "GOOGL", "NWSA": "NWS", "FOXA": "FOX", "SATS": "ECHO"}
            issuers = [aliases.get(s, s) for s in symbols]
        self.issuers = _classifications(issuers, shape)
        fx = np.asarray(fx, dtype=float)
        self.fx = fx[1:] / fx[0] if len(fx) == shape[0] + 1 else fx
        if self.fx.shape != (shape[0],) or not (np.isfinite(self.fx) & (self.fx > 0)).all():
            raise ValueError("Missing positive FX conversion index")
        allow = np.ones(shape, dtype=bool) if tradable is None else np.asarray(tradable, dtype=bool)
        if allow.shape != shape:
            raise ValueError("Invalid tradability dimensions")
        self.executable = allow & np.isfinite(opn) & (opn > 0)
        self.payout = np.full(shape, np.nan)
        if isinstance(settlements, Mapping):
            for (day, symbol), amount in settlements.items():
                if not np.isfinite(amount) or amount < 0:
                    raise ValueError(f"Invalid cash settlement for {symbol} on {day}")
                self.payout[self.days.index(str(day)[:10]), self.symbols.index(symbol)] = amount
        elif settlements is not None:
            mask = np.asarray(settlements, dtype=bool)
            if mask.shape != shape or not (np.isfinite(opn[mask]) & (opn[mask] >= 0)).all():
                raise ValueError("Missing settlement value in opening marks")
            self.payout[mask] = opn[mask]
        self.settled = np.isfinite(self.payout)
        self.executable &= ~np.maximum.accumulate(self.settled, axis=0)
        self.overnight = np.ones(shape)
        self.intraday = np.ones(shape)
        last = np.full(shape[1], np.nan)
        for j in range(shape[0]):
            # An auction with no print cannot fill a sale. Carry the last observed
            # mark for an existing position until a new mark or a cash settlement.
            o = np.where(np.isfinite(opn[j]) & (opn[j] > 0), opn[j], last)
            o = np.where(self.settled[j], self.payout[j], o)
            c = np.where(np.isfinite(cls[j]) & (cls[j] > 0), cls[j], o)
            np.divide(o, last, out=self.overnight[j], where=np.isfinite(last) & (last > 0))
            np.divide(c, o, out=self.intraday[j], where=np.isfinite(o) & (o > 0))
            last = c
        dates = [date.fromisoformat(d) for d in self.days]
        self.weekly = [i == 0 or d.isocalendar()[:2] != dates[i-1].isocalendar()[:2]
                       for i, d in enumerate(dates)]
        self.monthly = [i == 0 or (d.year, d.month) != (dates[i-1].year, dates[i-1].month)
                        for i, d in enumerate(dates)]


def _pick(ranks, old, due, groups, issuer, sector, executable, cap):
    targets = old.copy()
    reserved = {issuer[s] for s, review in zip(old, due) if not review and s >= 0}
    counts = Counter(sector[s] for s, review in zip(old, due) if not review and s >= 0)
    for i, (label, _) in enumerate(groups):
        if not due[i]:
            continue
        targets[i] = -1
        for s in ranks[label]:
            if s < 0 or s >= len(executable):
                raise ValueError(f"Ranked symbol index outside market panel for {label}: {s}")
            if executable[s] and issuer[s] not in reserved and (cap is None or counts[sector[s]] < cap):
                targets[i] = s
                reserved.add(issuer[s])
                counts[sector[s]] += 1
                break
    # Rank determines the basket. Keep a selected incumbent in its existing slot
    # within each sleeve/frequency group, so rank-only permutations do not trade.
    review = defaultdict(list)
    for i, group in enumerate(groups):
        if due[i]:
            review[group].append(i)
    for positions in review.values():
        picked = [targets[i] for i in positions]
        destinations = []
        for i in positions:
            if old[i] >= 0 and old[i] in picked:
                targets[i] = old[i]
                picked.remove(old[i])
            else:
                destinations.append(i)
        for i, s in zip(destinations, picked):
            targets[i] = s
    return targets


def _trade(values, budget, old, new):
    """Net same-symbol transfers across slots; solve executable 10bp fees from cash."""
    sells = {s: i for i, s in enumerate(old) if s >= 0}
    buys = {s: i for i, s in enumerate(new) if s >= 0}
    legs = [(sells.get(s), buys.get(s)) for s in sorted(sells.keys() | buys.keys())]
    turnover = sum(abs((budget[0, b] if b is not None else 0) -
                       (values[0, a] if a is not None else 0)) for a, b in legs)
    # Costs belong to the account slot submitting the residual buy or sale.
    # Charging a buy reduces the shares affordable, including when rotating out
    # of another name. A fixed point enforces both sides against the net order.
    fees = np.zeros(7)
    for _ in range(8):
        result = np.zeros(7)
        for a, b in legs:
            order = ((budget[1, b] - fees[b] if b is not None else 0)
                     - (values[1, a] if a is not None else 0))
            if order > 0:
                result[b] += order * .001
            elif order < 0:
                result[a] -= order * .001
        if np.allclose(result, fees, rtol=1e-13, atol=1e-15):
            break
        fees = result
    budget[1] -= result
    return turnover


def replay(market, ranks, layout, sector_cap=None, detail=False, extra_reviews=None,
           open_targets=None):
    """Slot capitals start at 1/7 and reset at the first US open of each month.

    Each ranking is a sequence indexed by execution day and contains integer
    symbol indices already ordered by data available before that session.
    extra_reviews maps each sleeve to additional execution-session indices,
    for example the two transitions of a pre-open market gate.
    open_targets, when provided, receives sparse pre-intraday target vectors for
    reuse of this exact shadow account in market-regime switches.
    """
    groups = tuple(tuple(slot) for slot in layout)
    if len(groups) != 7 or sector_cap not in (None, 2, 3):
        raise ValueError("Exactly seven slots and a sector cap of None, 2 or 3 are required")
    labels = list(dict.fromkeys(label for label, _ in groups))
    extra = {label: set((extra_reviews or {}).get(label, ())) for label in labels}
    for label, frequency in groups:
        if frequency not in {"daily", "weekly", "monthly"} or len(ranks[label]) != len(market.days):
            raise ValueError(f"Invalid schedule or ranking history for {label}")
    old = np.full(7, -1, dtype=int)
    values = np.full((2, 7), 1/7)  # zero-cost and 10bp ledgers, cash or market value
    activated = [False] * 7
    rows = []
    for j, day in enumerate(market.days):
        rank = {label: ranks[label][j] for label in labels}
        held = old >= 0
        values[:, held] *= market.overnight[j, old[held]]
        open_nav = values[0].sum()
        settled = held & market.settled[j, old]
        old[settled] = -1  # cash payout with no brokerage trade
        fixed = (old >= 0) & ~market.executable[j, old]
        due = [not fixed[i] and not settled[i] and (
            frequency == "daily" or market.monthly[j] and frequency == "monthly"
            or market.weekly[j] and frequency == "weekly" or j in extra[label]
            or not activated[i] and len(rank[label]) > 0)
            for i, (label, frequency) in enumerate(groups)]
        for i, (label, _) in enumerate(groups):
            if due[i] and len(rank[label]) > 0:
                activated[i] = True
        sector = market.sectors[j]
        new = (_pick(rank, old, due, groups, market.issuers[j], sector,
                     market.executable[j], sector_cap) if any(due) else old.copy())
        turn = 0.
        if market.monthly[j] or not np.array_equal(old, new):
            budget = values.copy()
            if market.monthly[j] and not fixed.all():
                budget[:, ~fixed] = values[:, ~fixed].mean(axis=1, keepdims=True)
            turn = _trade(values, budget, old, new) / open_nav
            values = budget
        held = new >= 0
        exposure = values[0, held].sum() / open_nav
        if open_targets is not None and (market.monthly[j] or settled.any() or not np.array_equal(old, new)):
            w = np.zeros(len(market.symbols))
            w[new[held]] = values[0, held] / open_nav
            open_targets[j] = w
        values[:, held] *= market.intraday[j, new[held]]
        nav, net_nav = values.sum(axis=1)
        row = dict(usd=nav, krw=nav * market.fx[j], krw10=net_nav * market.fx[j],
                   turn=turn, exposure=exposure, active=int(held.sum()))
        if detail:
            weights = Counter()
            names = []
            for i, s in enumerate(new):
                symbol = market.symbols[s] if s >= 0 else ""
                if s >= 0:
                    weights[sector[s]] += values[0, i] / nav
                    names.append(symbol)
                row[f"slot{i+1}"] = symbol
                row[f"slot{i+1}_value"] = values[0, i]
            row.update(holdings="|".join(names), sectors=len(weights),
                       max_sector_weight=max(weights.values(), default=0.))
        rows.append(row)
        old = new
    return pd.DataFrame(rows, index=pd.Index(market.days, name="date"))
