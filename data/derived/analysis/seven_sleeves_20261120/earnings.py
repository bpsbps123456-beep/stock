"""Point-in-time quarterly growth. Run from the repository root with py <this file>."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "finviz_rotation_20261119"
sys.path.insert(0, str(PRIOR))
from fundamentals import get  # noqa: E402

CACHE = HERE / "cache" / "filing_flows"
END = "2026-11-17"
SALES = (
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
COLS = ["ticker", "cik", "filed", "accession", "period"]


def filings():
    # The balance-sheet archive identifies the actual fiscal period for each accession.
    docs = pd.concat(
        [
            pd.read_csv(d / "fundamental_inputs/as_filed.csv", usecols=COLS)
            for d in (PRIOR, PRIOR / "midcaps")
        ]
    )
    docs = docs.drop_duplicates().sort_values(["cik", "filed", "accession", "ticker"])
    names = set(pd.read_csv(PRIOR / "broad_membership.csv").symbol) | set(
        pd.read_csv(PRIOR / "midcaps/membership.csv").symbol
    )
    return docs[docs.ticker.isin(names)]


def download(cik):
    """Retain raw flow facts by accession and by observed unit, never the latest restated value alone."""
    path = CACHE / f"{cik}.json"
    if path.exists():
        return
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
    body = get(url)
    data = json.loads(body)["facts"].get("us-gaap", {}) if body else {}
    rows = []
    for tag in SALES:
        for obs in data.get(tag, {}).get("units", {}).get("USD", []):
            if (
                "start" in obs
                and obs["form"] in {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "20-F/A"}
                and obs["end"] >= "2023-12-01"
                and obs["filed"] <= END
                and obs["end"] <= obs["filed"]
            ):
                rows.append([obs["filed"], obs["accn"], tag, obs["start"], obs["end"], obs["val"]])
    path.write_text(json.dumps(sorted(rows), separators=(",", ":")), encoding="utf-8")


OPERATING = "OperatingIncomeLoss"


def download_operating(cik):
    path = CACHE / f"op_{cik}.json"
    if path.exists():
        return
    url = (
        f"https://data.sec.gov/api/xbrl/companyconcept/CIK{int(cik):010d}/us-gaap/{OPERATING}.json"
    )
    body = get(url)
    units = json.loads(body).get("units", {}) if body else {}
    rows = []
    for obs in units.get("USD", []):
        if (
            "start" in obs
            and obs["form"] in {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "20-F/A"}
            and obs["end"] >= "2023-12-01"
            and obs["filed"] <= END
            and obs["end"] <= obs["filed"]
        ):
            rows.append(
                [obs["filed"], obs["accn"], OPERATING, obs["start"], obs["end"], obs["val"]]
            )
    path.write_text(json.dumps(sorted(rows), separators=(",", ":")), encoding="utf-8")


def fetch_operating():
    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(download_operating, sorted(filings().cik.unique())))


def fetch_all():
    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(5) as pool:
        list(pool.map(download, sorted(filings().cik.unique())))


@dataclass(frozen=True)
class Quarter:
    start: int
    end: int
    value: float
    method: str
    source: str


@lru_cache(maxsize=None)
def ordinal(value):
    return date.fromisoformat(value).toordinal()


def quarter_table(spans):
    """Additive revenue or operating profit only. Prefer filed 3-month numbers, otherwise difference like fiscal YTDs."""
    quarters = {}
    fiscal_starts = defaultdict(list)
    for (start, end), (value, source) in sorted(spans.items()):
        duration = end - start + 1
        if 80 <= duration <= 100:
            quarters[end] = Quarter(start, end, value, "reported", source)
        if 80 <= duration <= 380:
            fiscal_starts[start].append((end, value, source))
    for start, periods in fiscal_starts.items():
        for j, (end, value, source) in enumerate(periods):
            if end in quarters:
                continue
            duration = end - start + 1
            # Accept only YTD6−Q1, YTD9−YTD6 or FY−YTD9 from the SAME fiscal start and tag.
            earlier_span = next(
                (
                    prior
                    for low, high, prior in (
                        (165, 200, (80, 100)),
                        (250, 295, (165, 200)),
                        (350, 380, (250, 295)),
                    )
                    if low <= duration <= high
                ),
                None,
            )
            if earlier_span is None:
                continue
            previous = [
                (e, v, src)
                for e, v, src in periods[:j]
                if earlier_span[0] <= e - start + 1 <= earlier_span[1] and 80 <= end - e <= 100
            ]
            if len(previous) != 1:
                continue
            prior_end, prior_value, prior_source = previous[0]
            quarters[end] = Quarter(
                prior_end + 1,
                end,
                value - prior_value,
                "ytd_difference",
                source + "/" + prior_source,
            )
    return quarters


def four_comparable(quarters, end):
    latest = quarters.get(end)
    if latest is None:
        return None
    prev = quarters.get(latest.start - 1)
    prior_candidates = [
        q
        for q in quarters.values()
        if 354 <= end - q.end <= 376
        and 354 <= latest.start - q.start <= 376
        and abs((latest.end - latest.start) - (q.end - q.start)) <= 8
    ]
    if prev is None or len(prior_candidates) != 1:
        return None
    last_year = prior_candidates[0]
    prev_year = quarters.get(last_year.start - 1)
    if (
        prev_year is None
        or not 354 <= prev.start - prev_year.start <= 376
        or abs((prev.end - prev.start) - (prev_year.end - prev_year.start)) > 8
    ):
        return None
    return latest, last_year, prev, prev_year


def income_signal(state, end):
    for tag in SALES:
        quarters = quarter_table(state.get(tag, {}))
        matched = four_comparable(quarters, end)
        if matched is None:
            continue
        current, year_ago, previous, previous_year = matched
        row = {"tag": tag}
        for key, q in zip(("current", "year_ago", "previous", "previous_year"), matched):
            row.update(
                {
                    key + "_start": date.fromordinal(q.start).isoformat(),
                    key + "_end": date.fromordinal(q.end).isoformat(),
                    key + "_sales": q.value,
                    key + "_method": q.method,
                    key + "_sources": q.source,
                }
            )
        if min(year_ago.value, previous_year.value) <= 0 or min(current.value, previous.value) < 0:
            return {**row, "status": "invalid_sales"}
        yoy = current.value / year_ago.value - 1
        previous_yoy = previous.value / previous_year.value - 1
        score = yoy + 0.5 * (yoy - previous_yoy)
        operating = quarter_table(state.get(OPERATING, {})).get(end)
        comparable = operating is not None and operating.start == current.start
        if not comparable:
            status = "unavailable_operating_income"
        elif operating.value <= 0:
            status = "nonpositive_operating_income"
        elif yoy <= 0:
            status = "nonpositive_growth"
        else:
            status = "eligible"
        return {
            **row,
            "operating_income": operating.value if comparable else np.nan,
            "operating_method": operating.method if comparable else None,
            "operating_sources": operating.source if comparable else None,
            "sales_yoy": yoy,
            "previous_yoy": previous_yoy,
            "acceleration": yoy - previous_yoy,
            "raw_score": score,
            "score": score if status == "eligible" else np.nan,
            "status": status,
        }
    return {"status": "unavailable_comparables"}


def history(cik, docs):
    rows = json.loads((CACHE / f"{cik}.json").read_text())
    rows.extend(json.loads((CACHE / f"op_{cik}.json").read_text()))
    updates = defaultdict(list)
    for filed, accession, tag, start, end, value in rows:
        if tag in SALES or tag == OPERATING:
            updates[filed].append((accession, tag, ordinal(start), ordinal(end), value))
    official = {}
    for day, group in docs.groupby("filed"):
        official[day] = [
            (ordinal(period), acc) for period, acc in zip(group.period, group.accession)
        ]
    state = defaultdict(dict)
    last_period = None
    out = []
    for day in sorted(set(updates) | set(official)):
        for accession, tag, start, end, value in sorted(updates[day]):
            state[tag][(start, end)] = (value, accession)
        if day in official:
            latest = max(official[day])
            last_period = max(last_period, latest[0]) if last_period is not None else latest[0]
        if last_period is not None and day >= "2025-06-01":
            out.append(
                {
                    "cik": cik,
                    "filed": day,
                    "period": date.fromordinal(last_period).isoformat(),
                    **income_signal(state, last_period),
                }
            )
    return out


def freeze():
    docs = filings()
    frames = []
    for cik, group in docs.groupby("cik"):
        frames.extend(history(cik, group.drop_duplicates(["filed", "period", "accession"])))
    table = pd.DataFrame(frames).sort_values(["cik", "filed"])
    table.to_csv(HERE / "earnings_filings.csv", index=False)
    tickers = docs[["cik", "ticker"]].drop_duplicates().sort_values(["ticker", "cik"])
    tickers.to_csv(HERE / "earnings_issuers.csv", index=False)
    return table


def earnings_scores(days, symbols):
    """Ungraded names are NaN. Returned raw scores can be ranked in any historical universe."""
    idx = pd.DatetimeIndex(pd.to_datetime(days))
    symbols = list(symbols)
    result = pd.DataFrame(np.nan, index=days, columns=symbols)
    table = pd.read_csv(HERE / "earnings_filings.csv")
    tickers = pd.read_csv(HERE / "earnings_issuers.csv")
    rosters = pd.read_csv(PRIOR / "broad_membership.csv")
    changed = {
        t: g.set_index("month").cik.to_dict()
        for t, g in rosters.groupby("symbol")
        if g.cik.nunique() > 1
    }
    month = idx.strftime("%Y-%m").map(lambda m: max(m, "2026-01"))
    groups = tickers[tickers.ticker.isin(symbols)].groupby("cik").ticker.agg(list)
    for cik, frame in table.groupby("cik"):
        if cik not in groups:
            continue
        frame.index = pd.to_datetime(frame.filed) + pd.Timedelta(days=1)
        known = frame.reindex(idx, method="ffill")
        fresh = (idx - pd.to_datetime(known.period)).dt.days.between(0, 180)
        vals = known.score.where(fresh).to_numpy()
        for ticker in groups[cik]:
            if ticker in changed:
                usable = np.array([changed[ticker].get(m) == cik for m in month])
                result.loc[usable, ticker] = vals[usable]
            else:
                result[ticker] = vals
    return result


def audit():
    """Independently verify date cutoffs, source availability and sample raw amounts."""
    frame = pd.read_csv(HERE / "earnings_filings.csv")
    issuer_map = pd.read_csv(HERE / "earnings_issuers.csv")
    scores = frame[frame.score.notna()]
    np.testing.assert_allclose(
        scores.score, 1.5 * scores.sales_yoy - 0.5 * scores.previous_yoy, rtol=1e-12
    )
    assert (scores.operating_income > 0).all() and (scores.sales_yoy > 0).all()
    assert (scores.current_end == scores.period).all() and (frame.period <= frame.filed).all()
    sources_checked = 0
    for cik, group in frame[frame.raw_score.notna()].groupby("cik"):
        observations = json.loads((CACHE / f"{cik}.json").read_text())
        observations.extend(json.loads((CACHE / f"op_{cik}.json").read_text()))
        source_dates = {acc: filed for filed, acc, *unused in observations}
        for row in group.itertuples():
            for col in (
                "current_sources",
                "previous_sources",
                "year_ago_sources",
                "previous_year_sources",
                "operating_sources",
            ):
                accs = getattr(row, col)
                if pd.isna(accs):
                    continue
                for acc in accs.split("/"):
                    assert source_dates[acc] <= row.filed, (cik, row.filed, acc)
                    sources_checked += 1
    days = pd.read_csv(PRIOR / "technical_inputs/price.csv", usecols=["date"]).date.tolist()
    symbols = sorted(
        set(pd.read_csv(PRIOR / "broad_membership.csv").symbol)
        | set(pd.read_csv(PRIOR / "midcaps/membership.csv").symbol)
    )
    wide = earnings_scores(days, symbols)
    sec = issuer_map[issuer_map.ticker == "AAPL"].cik.iat[0]
    sample = frame[frame.cik == sec].sort_values("filed").iloc[-1]
    event = sample.filed
    before = frame[(frame.cik == sec) & (frame.filed < event)].iloc[-1]
    ts = pd.Timestamp(event)
    lookup = earnings_scores([event, str((ts + pd.Timedelta(days=1)).date())], ["AAPL"])
    np.testing.assert_allclose(lookup.AAPL, [before.score, sample.score], equal_nan=True)
    # Read the fiscal Q4 example directly from annual and 9-month values in the raw XBRL source.
    raw = json.loads((CACHE / f"{sec}.json").read_text())
    raw.extend(json.loads((CACHE / f"op_{sec}.json").read_text()))
    for tag, sources_col, result_col in [
        (sample.tag, "current_sources", "current_sales"),
        (OPERATING, "operating_sources", "operating_income"),
    ]:
        annual_acc, nine_month_acc = sample[sources_col].split("/")
        key = (sample.period, sample.previous_end)
        current = [
            (start, val)
            for filed, acc, t, start, end, val in raw
            if acc == annual_acc
            and t == tag
            and end == key[0]
            and 350 <= (ordinal(end) - ordinal(start) + 1) <= 380
        ]
        assert len(current) == 1, (tag, key)
        parts = [
            val
            for filed, acc, t, start, end, val in raw
            if acc == nine_month_acc and t == tag and end == key[1] and start == current[0][0]
        ]
        assert len(parts) == 1 and current[0][1] - parts[0] == sample[result_col]
    stats = {
        "requested_symbols": len(symbols),
        "issuers_in_source": int(issuer_map.cik.nunique()),
        "mapped_symbols": int(issuer_map.ticker.nunique()),
        "missing_source_symbols": sorted(set(symbols) - set(issuer_map.ticker)),
        "events": len(frame),
        "events_by_status": frame.status.value_counts().to_dict(),
        "calendar_first": days[0],
        "calendar_last": days[-1],
        "calendar_closes": len(days),
        "score_available_ticker_closes_before_universe_and_price_filters": int(wide.count().sum()),
        "min_and_max_scored_names_before_roster_per_day": [
            int(wide.count(axis=1).min()),
            int(wide.count(axis=1).max()),
        ],
        "Q4_sample": sample.fillna("").to_dict(),
        "verified_accession_cutoffs": sources_checked,
        "freshness": "strict filing calendar date < signal close date; period age <=180 days. Score evaluated at next open.",
        "universe": "Raw historical S&P500 monthly/401-starting-midcap input labels. Fund arrays reconcile aliases and corporate actions. Rank within then-active pool.",
        "formula": "require current standalone sales YoY >0, previous/current year-ago denominators >0, current and prior-quarter sales >=0, and current standalone operating income >0. Score = current sales YoY + 0.5*(current YoY - previous quarter YoY). Filed operating income used as gate only. No earnings beat, adjusted EPS, valuation or price signal.",
        "quarter": "USD US GAAP quarterly sales/OperatingIncomeLoss. Single 80–100-day periods preferred; same-start cumulative 6m−3m, 9m−6m, FY−9m fallback (80–100 days difference). Consecutive quarter boundaries must touch; YoY start and end 354–376 days apart and durations within 8 days.",
        "timing": "Companyfacts per-accession originals and later revisions enter state only on their file date; no first-vintage facts overwritten backwards. Current SEC issuer directory needed only to reproduce prior roster mapping.",
        "limitations": "No comparable USD US GAAP business-revenue tag or operating income means unknown, especially banks and non-US filers. Recognized revenue may exclude other income. Large sales growth may include acquisitions, small denominators and 53-week fiscal years.",
        "sources": [
            "https://www.sec.gov/edgar/sec-api-documentation",
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        ],
    }
    (HERE / "earnings_coverage.json").write_text(
        json.dumps(stats, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return stats


if __name__ == "__main__":
    fetch_all()
    fetch_operating()
    freeze()
    audit()
