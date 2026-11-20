"""Rebuild pre-trend candidates from public filings and completed price bars.

The base universe is the historical roster passing the original non-technical
fundamentals, 13F ownership proxy, USD 10 price and prior 63-day volume gates.
No RSI, relative-volume or moving-average gate is applied to the saved rows.
Rolling prices and the signal close use the share units prevailing on the signal
date. Short histories retain missing indicators until their full window exists.
"""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "finviz_recipes_20261120"
sys.path[:0] = [str(PRIOR), str(HERE.parents[3])]
import recipes
from timing import review_days

PERIODS = (10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 100, 125, 150, 175, 200)
PARTS = ("h1", "h2")


def review_indices(src):
    schedules = ("monthly1", "monthly1_11", "weekly1")
    # Calendar-only schedules are independent of the market-on flag.
    days = set()
    for schedule in schedules:
        days.update(review_days(src.days, np.ones(len(src.days), dtype=bool), schedule))
    return sorted(days)


def source_candidates(src, reviews):
    funds = recipes.fundamentals(src)
    base = src.membership & (funds["quality"] == 1) & (funds["own"] > .4)
    local_row, col = np.nonzero(base[reviews])
    row = np.array(reviews)[local_row]
    classes = pd.read_csv(recipes.SEVEN / "classifications.csv", dtype=str)
    large = set(zip(classes.month[classes.pool == "large"], classes.symbol[classes.pool == "large"]))
    table = pd.DataFrame(dict(trade_open=np.asarray(src.days)[row],
                              signal_close=np.asarray(src.signals)[row],
                              symbol=np.asarray(src.symbols)[col], col=col,
                              raw=src.raw_symbols[row, col], cik=src.issuers[row, col],
                              own=funds["own"][row, col],
                              sales_yoy=recipes.quarterly_sales(src)[row, col]))
    table["pool"] = ["large" if (day[:7], symbol) in large else "mid"
                     for day, symbol in zip(table.trade_open, table.symbol)]
    assert (table.signal_close < table.trade_open).all()
    return table


def add_technical(table, src):
    market = recipes.market
    sessions = pd.DatetimeIndex(src.signals)
    last_independent = {s: sessions[sessions < pd.Timestamp(term[0])][-1].date().isoformat()
                        for s, term in market.mid.DEALS.items()}
    pieces = []
    with patch.dict(market.tech.SIP_END, last_independent):
        for symbol, group in table.groupby("symbol", sort=False):
            dates = pd.DatetimeIndex(group.signal_close)
            market.tech.fetch(symbol)
            bars = market.tech.read_bars(symbol)
            if bars is None:
                raise ValueError(f"Historical bars missing for candidate {symbol}")
            c, v, unit = bars.close, bars.volume, bars.split_mult
            prior_volume = v.shift(1).rolling(63).mean()
            columns = dict(close=c * unit, avg=prior_volume / unit, rv=v / prior_volume.where(prior_volume > 0),
                           rsi=market.tech.rsi(c), p21=c.pct_change(21, fill_method=None),
                           p63=c.pct_change(63, fill_method=None))
            for n in PERIODS:
                sma = c.rolling(n).mean()
                columns[f"sma{n}"] = sma * unit
                if n in (25, 50, 75, 200):
                    columns[f"slope{n}"] = sma.pct_change(5, fill_method=None)
            matched = pd.DataFrame(columns).reindex(dates)
            matched.index = group.index
            pieces.append(matched)
    full = table.join(pd.concat(pieces), validate="one_to_one")
    return full[full.close.gt(10) & full.avg.gt(500_000)].copy()


def add_filing_ratios(table):
    tables = []
    roots = {"large": recipes.market.PRIOR, "mid": recipes.market.mid.OUT}
    for pool, root in roots.items():
        facts = pd.read_csv(root / "fundamental_inputs/as_filed.csv", dtype={"cik": str})
        facts = facts.sort_values(["filed", "period", "accession"])
        by_issuer = dict(tuple(facts.groupby(["ticker", "cik"])))
        for key, group in table[table.pool.eq(pool)].groupby(["raw", "cik"], sort=False):
            filings = by_issuer[key]
            ix = filings.filed.searchsorted(group.signal_close, side="left") - 1
            assert (ix >= 0).all()
            matched = filings.iloc[ix][["filed", "period", "current_ratio", "debt_equity"]].copy()
            matched.index = group.index
            assert (matched.filed < group.signal_close).all()
            age = pd.to_datetime(group.signal_close) - pd.to_datetime(matched.period)
            assert age.dt.days.between(0, 170).all()
            tables.append(matched)
    full = table.join(pd.concat(tables), validate="one_to_one")
    assert (full.current_ratio > 1).all() and full.debt_equity.between(0, 1, inclusive="left").all()
    return full.drop(columns=["raw", "cik", "pool", "period"])


def audit(src, table, reviews):
    ranks = recipes.recipes(src)
    original_ma = (table.sma20 > table.sma50) & (table.close > table.sma50) & (table.close > table.sma200)
    non_ma = (table.rsi > 40) & (table.rv > 1)
    selected = table[original_ma & non_ma].sort_values(["trade_open", "rv", "col"], ascending=[True, False, True])
    n_compared = 0
    for kind, kept in [("original", selected), ("balance", selected[(selected.current_ratio > 1.5)
                                                                     & (selected.debt_equity < .5)])]:
        days = kept.groupby("trade_open").col.agg(list).to_dict()
        for j in reviews:
            assert days.get(src.days[j], []) == ranks[kind][j], (kind, src.days[j])
            n_compared += len(days.get(src.days[j], []))
    alternate = (table.sma25 > table.sma75) & (table.close > table.sma75) & (table.close > table.sma150)
    added = table[non_ma & alternate & ~original_ma]
    assert len(added) > 0 and not table.duplicated(["trade_open", "symbol"]).any()
    assert table.trade_open.nunique() == len(reviews)
    print(f"{len(table)} pre-trend rows on {len(reviews)} reviews, {table.symbol.nunique()} companies/share classes. "
          f"Verified {n_compared} historical rank entries. 25/75/150 newly admits "
          f"{len(added)} signal-security observations ({added.symbol.nunique()} securities).")


def freeze():
    src = recipes.market.load_inputs()
    reviews = review_indices(src)
    base = source_candidates(src, reviews)
    table = add_filing_ratios(add_technical(base, src)).sort_values(["trade_open", "col"])
    audit(src, table, reviews)
    for part, block in zip(PARTS, [table[table.trade_open <= "2026-06-30"], table[table.trade_open > "2026-06-30"]]):
        path = HERE / f"features_{part}.csv"
        block.to_csv(path, index=False, float_format="%.17g")
        assert path.stat().st_size < 10_000_000
    audit(src, load_features(), reviews)


def load_features():
    return pd.concat([pd.read_csv(HERE / f"features_{part}.csv", float_precision="round_trip")
                      for part in PARTS], ignore_index=True)


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        freeze()
    else:
        source = recipes.market.load_inputs()
        audit(source, load_features(), review_indices(source))
