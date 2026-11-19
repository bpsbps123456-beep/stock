"""Filed SEC balance-sheet and business-address proxies for the historical screen."""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "fundamental_inputs"
END = "2026-11-17"
UA = {"User-Agent": "GBRC-research personal-noncommercial research@example.com"}
QUARTERS = ["2025q3", "2025q4", "2026q1", "2026q2", "2026q3"]
PERIODIC = {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "20-F/A"}
TAGS = {
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "Assets",
    "Liabilities",
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
    "LongTermDebt",
    "LongTermDebtCurrent",
    "LongTermDebtNoncurrent",
    "ShortTermBorrowings",
    "DebtCurrent",
    "CommercialPaper",
    "ShortTermBankLoansAndNotesPayable",
    "FinanceLeaseLiability",
    "LongTermDebtAndCapitalLeaseObligations",
    "LongTermDebtAndCapitalLeaseObligationsCurrent",
    "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
    "DebtLongtermAndShorttermCombinedAmount",
}
# Delisted/renamed securities absent from the current ticker directory. The
# December 2025 constituent names identify these unique 2025 SEC SUB issuers.
MIDCAP_FORMER = {
    "ASGN": 890564,
    "BLD": 1633931,
    "CIVI": 1509589,
    "FYBR": 20520,
    "GTLS": 892553,
    "JHG": 1274173,
    "MASI": 937556,
    "NSA": 1618563,
    "PCH": 1338749,
    "SNV": 18349,
    "TMHC": 1562476,
    "WBS": 801337,
}
LOCK = threading.Lock()
last_request = 0.0


def get(url):
    """At most six requests/sec, capped concurrency, pause for transient errors."""
    global last_request
    for attempt in range(4):
        with LOCK:
            wait = 0.18 - (time.monotonic() - last_request)
            if wait > 0:
                time.sleep(wait)
            last_request = time.monotonic()
        try:
            r = requests.get(url, headers=UA, timeout=45)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def ticker_map():
    path = CACHE / "sec_tickers.json"
    if not path.exists():
        path.write_bytes(get("https://www.sec.gov/files/company_tickers.json"))
    return {r["ticker"]: r["cik_str"] for r in json.loads(path.read_bytes()).values()}


def country_rows():
    frames = []
    for quarter in QUARTERS:
        p = CACHE / (quarter + ".zip")
        if not p.exists():
            p.write_bytes(
                get("https://www.sec.gov/files/dera/data/financial-statement-data-sets/" + p.name)
            )
        with zipfile.ZipFile(p) as z:
            t = pd.read_csv(
                z.open("sub.txt"),
                sep="\t",
                usecols=["cik", "form", "countryba", "filed"],
                dtype=str,
            )
        frames.append(t[t.form.isin(PERIODIC)])
    d = pd.concat(frames, ignore_index=True).dropna()
    d["filed"] = pd.to_datetime(d.filed, format="%Y%m%d").dt.strftime("%Y-%m-%d")
    return {int(k): sorted(set(zip(g.filed, g.countryba))) for k, g in d.groupby("cik")}


def collect_cik(cik):
    path = CACHE / "statements" / f"{cik}.json"
    if path.exists():
        return
    raw = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
    data = json.loads(raw)["facts"] if raw else {}
    periods = {}
    for source in ("dei", "us-gaap"):
        for tag, concept in data.get(source, {}).items():
            if tag not in TAGS:
                continue
            unit = "shares" if "SharesOutstanding" in tag else "USD"
            for obs in concept["units"].get(unit, []):
                if (
                    obs["form"] not in PERIODIC
                    or "start" in obs
                    or obs["end"] < "2025-02"
                    or not "2025-06" <= obs["filed"] <= END
                    or obs["end"] > obs["filed"]
                ):
                    continue
                acc = (obs["filed"], obs["accn"])
                periods.setdefault(acc, {}).setdefault(obs["end"], {})[tag] = obs["val"]
    rows = []
    for (filed, acc), times in sorted(periods.items()):
        # Only values for this filing's latest balance-sheet period; no older comparative rows.
        bs_end = max(
            (d for d, v in times.items() if "Assets" in v or "AssetsCurrent" in v), default=None
        )
        if not bs_end or (pd.Timestamp(filed) - pd.Timestamp(bs_end)).days > 170:
            continue
        x = dict(times[bs_end])
        share_date = max(
            (
                d
                for d, v in times.items()
                if d >= bs_end and "EntityCommonStockSharesOutstanding" in v
            ),
            default=None,
        )
        share_count = (
            times[share_date]["EntityCommonStockSharesOutstanding"]
            if share_date
            else x.get("CommonStockSharesOutstanding")
        )
        rows.append(
            {
                "filed": filed,
                "accession": acc,
                "period": bs_end,
                "share_date": share_date or bs_end,
                "shares": share_count,
                "facts": x,
            }
        )
    path.write_text(json.dumps(rows), encoding="utf-8")


def all_names():
    from technicals import time_symbols
    from universe import broad_symbols

    return broad_symbols() | set(time_symbols())


def historical_map():
    mapping = ticker_map()
    roster = pd.read_csv(HERE / "broad_membership.csv", dtype={"cik": int})
    by_name = {k: {v} for k, v in mapping.items()}
    by_name.update({name: set(group.cik) for name, group in roster.groupby("symbol")})
    return by_name


def debt_ratio(facts):
    """Reconstruct available disclosed borrowings. Do not infer no debt from missing tags."""
    eq = facts.get("StockholdersEquity")
    if not eq or eq <= 0:
        return None, "missing_equity"
    combined = facts.get("DebtLongtermAndShorttermCombinedAmount")
    tagged = [facts.get("LongTermDebtAndCapitalLeaseObligations"), facts.get("LongTermDebt")]
    total_lt = next((v for v in tagged if v is not None), None)
    noncurrent = next(
        (
            facts[k]
            for k in ("LongTermDebtAndCapitalLeaseObligationsNoncurrent", "LongTermDebtNoncurrent")
            if k in facts
        ),
        None,
    )
    due = next(
        (
            facts[k]
            for k in ("LongTermDebtAndCapitalLeaseObligationsCurrent", "LongTermDebtCurrent")
            if k in facts
        ),
        None,
    )
    # Generic DebtCurrent includes the short term and the current portion of long-term debt.
    st_parts = [facts.get(k) for k in ("CommercialPaper", "ShortTermBankLoansAndNotesPayable")]
    short = facts.get("ShortTermBorrowings", sum(x for x in st_parts if x is not None))
    if combined is not None:
        debt = combined
    elif total_lt is not None:
        debt = total_lt + max(short, facts.get("DebtCurrent", 0) - (due or 0))
    elif noncurrent is not None:
        debt = noncurrent + max((due or 0) + short, facts.get("DebtCurrent", 0))
    else:
        # Even all liabilities together may not reach the screen threshold.
        liab = facts.get("Liabilities")
        if liab is not None and 0 <= liab < eq:
            return liab / eq, "liabilities_upper_bound"
        return None, "missing_debt"
    if not any(k.startswith("LongTermDebtAndCapitalLeaseObligations") for k in facts):
        debt += facts.get("FinanceLeaseLiability", 0)
    return debt / eq, "reported_borrowings"


def rebuild(*, midcaps=False):
    CACHE.mkdir(exist_ok=True, parents=True)
    root = HERE / "midcaps" if midcaps else HERE
    out = root / "fundamental_inputs"
    out.mkdir(exist_ok=True, parents=True)
    if midcaps:
        roster = pd.read_csv(root / "membership.csv")
        names = sorted(roster.symbol.unique())
        mapping = {name: {cik} for name, cik in ticker_map().items()}
        mapping.update({name: {cik} for name, cik in MIDCAP_FORMER.items()})
        mapping.update({r.symbol: {int(r.cik)} for r in roster.dropna(subset=["cik"]).itertuples()})
    else:
        names = sorted(all_names())
        mapping = historical_map()
    ciks = sorted({cik for t in names for cik in mapping.get(t, ())})
    (CACHE / "statements").mkdir(exist_ok=True, parents=True)
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        list(pool.map(collect_cik, ciks))
    countries = country_rows()
    filing_rows, geo_rows = [], []
    pairs = ((name, cik) for name in names for cik in sorted(mapping.get(name, ())))
    for name, cik in pairs:
        prev = None
        for filed, country in countries.get(cik, []):
            if country != prev:
                geo_rows.append((name, cik, filed, country))
                prev = country
        for r in json.loads((CACHE / "statements" / f"{cik}.json").read_text()):
            v = r["facts"]
            eq = v.get("StockholdersEquity")
            ca, cl = v.get("AssetsCurrent"), v.get("LiabilitiesCurrent")
            ratio, method = debt_ratio(v)
            filing_rows.append(
                {
                    "ticker": name,
                    "cik": cik,
                    "filed": r["filed"],
                    "accession": r["accession"],
                    "period": r["period"],
                    "share_date": r["share_date"],
                    "shares": r["shares"],
                    "current_ratio": ca / cl if ca is not None and cl and cl > 0 else None,
                    "debt_equity": ratio,
                    "debt_method": method,
                    "help_current_liab_equity": cl / eq
                    if cl is not None and eq and eq > 0
                    else None,
                }
            )
    pd.DataFrame(filing_rows).to_csv(out / "as_filed.csv", index=False)
    pd.DataFrame(geo_rows, columns=["ticker", "cik", "filed", "country"]).to_csv(
        out / "business_country.csv", index=False
    )
    meta = {
        "sources": [
            "https://www.sec.gov/edgar/sec-api-documentation",
            "https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets",
            "https://finviz.com/help/screener",
            "https://finviz.com/stock?t=AAPL",
        ],
        "ciks_from": "point-in-time broad_membership.csv, else SEC company_tickers.json (current lookup)",
        "filing_rule": "strictly earlier filing calendar date; signal as-of close; max 170 days after reported balance sheet",
        "country_rule": "Last disclosed business-address country from SEC 2025Q3..2026Q3 periodic filing archive; carry until next change. Q4 2026 location changes unobservable.",
        "debt_rule": "positive US GAAP parent equity; reconstruct reported short/long borrowings + finance lease when separate. If debt unavailable, use total liabilities as conservative upper bound only if below equity. Lease/debt tags and vendor's adjustments may diverge.",
        "vendor_check": "AAPL Finviz 2026-11-18 current ratio 1.02 and debt/equity 0.77. SEC Sep-26 2026 10-K current assets 173857m,current liabilities 169915m,equity 112093m. Help formula yields 1.516, opposite eligibility from displayed 0.77. Alternative literal documented gate accessible with definition='help'.",
        "market_cap": "as-traded close × latest public common share count, split-adjusted since count date. Class-price differences, buybacks, new issuance after disclosure introduce uncertainty. Share-class-specific DEI tags may be absent.",
        "missing": "Unknown facts fail. Banks without classified balance sheets, non-US GAAP reports and negative parent equity fail.",
        "institutional": "This gate excludes ownership. Any 13F study proxy is separate.",
        "symbols": len(names),
        "ciks_found": len(ciks),
        "statements": len(filing_rows),
        "countries_n_symbols": len({r[0] for r in geo_rows}),
        "no_ticker_mapping": sorted(set(names) - set(mapping)),
        "no_statement_symbols": sorted(set(names) - {r["ticker"] for r in filing_rows}),
    }
    if midcaps:
        meta["ciks_from"] = (
            "Dec 2025 fixed midcaps/membership.csv, 12 former ticker links verified by issuer in 2025q3/2025q4 SEC per-filing SUB tables, otherwise SEC company_tickers.json (current lookup). Unmatched CADE has no US GAAP issuer matched here."
        )
    (out / "method.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def load_fund_pass(price, definition="debt", return_coverage=False, *, midcaps=False):
    """Boolean signal-close US domicile, cap, current ratio and D/E gates; ownership separate."""
    if definition not in {"debt", "help"}:
        raise ValueError(definition)
    root = HERE / "midcaps" if midcaps else HERE
    snapshots = pd.read_csv(root / "fundamental_inputs/as_filed.csv")
    geo = pd.read_csv(root / "fundamental_inputs/business_country.csv")
    splits = pd.read_csv(root / "technical_inputs/split_events.csv")
    ratio = "debt_equity" if definition == "debt" else "help_current_liab_equity"
    report = pd.DataFrame(False, index=price.index, columns=price.columns)
    counts = dict.fromkeys(
        (
            "price",
            "country_known",
            "us",
            "recent",
            "ratio_known",
            "debt_known",
            "cap_known",
            "all_known",
            "pass",
        ),
        0,
    )
    dates = pd.to_datetime(price.index)
    roster = pd.read_csv(HERE / "broad_membership.csv")
    changed = {
        name: group.set_index("month").cik.to_dict()
        for name, group in roster.groupby("symbol")
        if not midcaps and group.cik.nunique() > 1
    }
    counts["price"] = int(np.isfinite(price.to_numpy()).sum())
    geo_tickers = set(geo.ticker)
    for (t, cik), g in snapshots.groupby(["ticker", "cik"]):
        if t not in price or t not in geo_tickers:
            continue
        g = (
            g.sort_values(["filed", "period", "accession"])
            .drop_duplicates("filed", keep="last")
            .set_index("filed")
        )
        # Shift source date forward one calendar day, then as-of join; late Friday maps to Monday.
        g.index = pd.to_datetime(g.index) + pd.Timedelta(days=1)
        panel = g.reindex(dates, method="ffill")
        countries = (
            geo.loc[geo.ticker.eq(t) & geo.cik.eq(cik)]
            .drop_duplicates("filed", keep="last")
            .set_index("filed")
            .country
        )
        countries.index = pd.to_datetime(countries.index) + pd.Timedelta(days=1)
        loc = countries.reindex(dates, method="ffill")
        current = (dates - pd.to_datetime(panel.period)).dt.days.between(0, 170)
        count = panel.shares.copy()
        for ev in splits.loc[splits.ticker.eq(t)].itertuples():
            date = pd.Timestamp(ev.date)
            count = count.where(
                ~((pd.to_datetime(panel.share_date) < date) & (dates >= date)), count * ev.ratio
            )
        px = price[t].to_numpy()
        cap = count * px
        known = (
            loc.notna() & current & panel.current_ratio.notna() & panel[ratio].notna() & cap.notna()
        )
        ok = (
            known
            & loc.eq("US")
            & (panel.current_ratio > 1)
            & (panel[ratio] >= 0)
            & (panel[ratio] < 1)
            & (cap > 2e9)
        )
        active = np.isfinite(px)
        if t in changed:
            date_ciks = [changed[t].get(max("2026-01", d.strftime("%Y-%m"))) for d in dates]
            active &= np.array(date_ciks) == cik
            ok &= active
        report[t] |= ok.to_numpy()
        for label, condition in [
            ("country_known", loc.notna()),
            ("us", loc.eq("US")),
            ("recent", current),
            ("ratio_known", panel.current_ratio.notna()),
            ("debt_known", panel[ratio].notna()),
            ("cap_known", cap.notna()),
            ("all_known", known),
            ("pass", ok),
        ]:
            counts[label] += int((condition & active).sum())
    return (report, counts) if return_coverage else report


def audit(*, midcaps=False):
    """Independent scalar lookup on a fixed sample, including missing/foreign filers."""
    from technicals import load_panels

    root = HERE / "midcaps" if midcaps else HERE
    out = root / "fundamental_inputs"
    if midcaps:
        price = pd.read_csv(root / "technical_inputs/price.csv", index_col=0)
        audit_names = "AA AAL AGCO ASGN BLD CIVI CADE GTLS JHG MASI NSA PCH SNV TMHC WBS"
    else:
        price = load_panels()["price"]
        audit_names = "AAPL NVDA PLTR MSFT AMZN WMT SPGI IBM MU LRCX TSM META JPM SNDK INTC"
    result, counts = load_fund_pass(price, return_coverage=True, midcaps=midcaps)
    literal = load_fund_pass(price, definition="help", midcaps=midcaps)
    facts = pd.read_csv(out / "as_filed.csv")
    addresses = pd.read_csv(out / "business_country.csv")
    events = pd.read_csv(root / "technical_inputs/split_events.csv")
    sample = [symbol for symbol in audit_names.split() if symbol in price]
    n = 0
    for name in sample:
        for date in price.index[::8]:
            rows = facts[(facts.ticker == name) & (facts.filed < date)].sort_values(
                ["filed", "period", "accession"]
            )
            loc = addresses[(addresses.ticker == name) & (addresses.filed < date)].sort_values(
                "filed"
            )
            expected = False
            if len(rows) and len(loc):
                last = rows.iloc[-1]
                relevant = events[
                    (events.ticker == name)
                    & (events.date > last.share_date)
                    & (events.date <= date)
                ]
                cap = last.shares * relevant.ratio.prod() * price.at[date, name]
                age = (pd.Timestamp(date) - pd.Timestamp(last.period)).days
                expected = (
                    loc.country.iloc[-1] == "US"
                    and 0 <= age <= 170
                    and 0 <= last.debt_equity < 1
                    and last.current_ratio > 1
                    and cap > 2e9
                )
            if bool(expected) != result.at[date, name]:
                raise ValueError(f"Independent gate mismatch on {date}: {name}")
            n += 1
    stats = {
        "observations": counts,
        "unit": "ticker x signal close. All quotes include SPY and Dec31 warmup; each individual field coverage count requires linked issuer. Missing fails.",
        "help_vs_reported_debt_different_cells": int((literal != result).sum().sum()),
        "independent_scalar_checks": n,
        "sample": sample,
        "audit_dates": "Every eighth available close, including warmup.",
    }
    if midcaps:
        stats["unit"] = (
            "Fixed December midcap cohort, ticker x signal close incl Dec31 warmup. Each count requires linked issuer; missing fails."
        )
    (out / "coverage.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--midcaps", action="store_true")
    args = parser.parse_args()
    if args.audit:
        print(audit(midcaps=args.midcaps))
    else:
        rebuild(midcaps=args.midcaps)
