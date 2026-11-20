"""Month-opening candidate ranks, financial ratios and point-in-time capitalization.

The original screen already applied an estimated cap > USD 2bn within monthly
S&P 500 and fixed December 2025 S&P 400 rosters. The pool column identifies
which roster supplied prices and filings. USD size buckets use cap_usd.
The quarterly/annual SEC filing must predate the signal session. Close is the
then-traded USD price from the roster's technical_inputs/price.csv snapshot.
Different voting classes, intervening issuance or buybacks can affect the estimate.
For DOCS the June 2026 balance sheet supplies combined Class A+B shares and
the exchange close values both at the Class A price (SEC accession in CSV).
"""
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROTATION = HERE.parent / "finviz_rotation_20261119"
SEVEN = HERE.parent / "seven_sleeves_20261120"
RECIPES = HERE.parent / "finviz_recipes_20261120"
sys.path.insert(0, str(RECIPES))
import recipes


def monthly_ranks():
    src = recipes.market.load_inputs()
    ranks = recipes.recipes(src)
    rows = []
    for i, day in enumerate(src.days):
        if i and src.days[i - 1][:7] == day[:7]:
            continue
        assert src.signals[i] < day
        balance = {symbol: rank for rank, symbol in enumerate(ranks["balance"][i], 1)}
        for rank, col in enumerate(ranks["original"][i], 1):
            rows.append((day, src.signals[i], src.symbols[col], rank, balance.get(col),
                         src.raw_symbols[i, col], src.issuers[i, col]))
    return pd.DataFrame(rows, columns=["trade_open", "signal_date", "ticker", "rank_original",
                                       "rank_balance", "raw", "cik"])


def freeze():
    candidates = monthly_ranks()
    classes = pd.read_csv(SEVEN / "classifications.csv", dtype=str)
    pools = {}
    for pool, root in [("large", ROTATION), ("mid", ROTATION / "midcaps")]:
        facts = pd.read_csv(root / "fundamental_inputs/as_filed.csv", dtype={"cik": str})
        facts = facts.sort_values(["filed", "period", "accession"])
        events = pd.read_csv(root / "technical_inputs/split_events.csv")
        prices = pd.read_csv(root / "technical_inputs/price.csv", index_col=0)
        pools[pool] = (dict(tuple(facts.groupby(["ticker", "cik"]))), events, prices)
    measured = []
    for day, group in candidates.groupby("trade_open", sort=False):
        large = set(classes.loc[classes.month.eq(day[:7]) & classes.pool.eq("large"), "symbol"])
        for r in group.itertuples():
            pool = "large" if r.ticker in large else "mid"
            facts, events, prices = pools[pool]
            issuer = facts[r.raw, r.cik]
            filing = issuer.loc[issuer.filed.lt(r.signal_date)].iloc[-1]
            assert filing.share_date <= filing.filed < r.signal_date < r.trade_open
            assert 0 <= (pd.Timestamp(r.signal_date) - pd.Timestamp(filing.period)).days <= 170
            splits = events[events.ticker.eq(r.raw) & events.date.gt(filing.share_date)
                            & events.date.le(r.signal_date)]
            # The only intervening vendor event in this cohort is a distribution
            # of WAT shares to BDX holders. It did not multiply the issuer's shares.
            # https://www.sec.gov/Archives/edgar/data/10795/000119312526042827/d104936dex991.htm
            distribution = splits.ticker.eq("BDX") & splits.date.eq("2026-02-10")
            factor = splits.loc[~distribution, "ratio"].prod()
            px = prices.at[r.signal_date, r.raw]
            cap = px * filing.shares * factor
            assert px > 10 and cap > 2e9
            assert filing.current_ratio > 1 and 0 <= filing.debt_equity < 1
            assert (filing.current_ratio > 1.5 and filing.debt_equity < .5) == pd.notna(r.rank_balance)
            measured.append(dict(pool=pool, cap_usd=cap, current_ratio=filing.current_ratio,
                                 debt_equity=filing.debt_equity, filing_date=filing.filed,
                                 period_end=filing.period, accession=filing.accession,
                                 share_date=filing.share_date, filed_shares=filing.shares,
                                 share_factor=factor, close_usd=px,
                                 cap_status="ok_ex_distribution" if distribution.any() else "ok"))
    out = pd.concat([candidates, pd.DataFrame(measured)], axis=1)
    assert out.trade_open.nunique() == 11 and not out.duplicated(["trade_open", "ticker"]).any()
    assert out.drop(columns="rank_balance").notna().all(axis=None)
    out["rank_balance"] = out.rank_balance.astype("Int64")
    out.to_csv(HERE / "size_inputs.csv", index=False, float_format="%.17g")
    return out


if __name__ == "__main__":
    data = freeze()
    print(f"{len(data)} candidate-months, {data.ticker.nunique()} tickers, "
          f"{data.rank_balance.notna().sum()} balance candidate-months")
