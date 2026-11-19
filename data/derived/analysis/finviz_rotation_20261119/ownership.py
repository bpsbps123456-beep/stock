"""Lagged institutional common-share proxy from public 13F and N-PORT bulk filings."""
from pathlib import Path
from zipfile import ZipFile
import json
import re
import pandas as pd
import requests
from fundamentals import UA

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
PERIODS = [
    ("2025-09-30", "2025-11-30", "2025q3", "01sep2025-30nov2025"),
    ("2025-12-31", "2026-02-28", "2025q4", "01dec2025-28feb2026"),
    ("2026-03-31", "2026-05-31", "2026q1", "01mar2026-31may2026"),
    ("2026-06-30", "2026-08-31", "2026q2", "01jun2026-31aug2026"),
]


def archive(name, directory):
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        url = f"https://www.sec.gov/files/{directory}/{name}"
        response = requests.get(url, headers=UA, timeout=60, stream=True)
        response.raise_for_status()
        tmp = path.with_suffix(".part")
        with tmp.open("wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
        tmp.replace(path)
    return ZipFile(path)


def read(z, table, columns, chunk=False):
    return pd.read_csv(z.open(table+".tsv"), sep="\t", usecols=columns.split(), dtype=str,
                       keep_default_na=False, chunksize=150000 if chunk else None,
                       encoding="utf-8", encoding_errors="replace")


def cusips(z, names):
    # Distinct accession votes for each ticker/cusip. Require a clear majority among funds.
    ids = {}
    for df in read(z, "IDENTIFIERS", "HOLDING_ID IDENTIFIER_TICKER", True):
        ticks = df.IDENTIFIER_TICKER.str.replace(".", "-", regex=False)
        valid = ticks.isin(names)
        ids.update(zip(df.HOLDING_ID[valid], ticks[valid]))
    pieces = []
    for df in read(z, "FUND_REPORTED_HOLDING", "ACCESSION_NUMBER HOLDING_ID ISSUER_CUSIP ASSET_CAT CURRENCY_CODE", True):
        take = df.HOLDING_ID.isin(ids) & df.ASSET_CAT.eq("EC") & df.CURRENCY_CODE.eq("USD") & df.ISSUER_CUSIP.str.fullmatch(r"[A-Z0-9]{8}\d")
        keep = df.loc[take, ["ACCESSION_NUMBER","HOLDING_ID","ISSUER_CUSIP"]].copy()
        keep["ticker"] = keep.HOLDING_ID.map(ids)
        pieces.append(keep[["ACCESSION_NUMBER", "ticker", "ISSUER_CUSIP"]])
    votes = pd.concat(pieces).drop_duplicates().groupby(["ticker", "ISSUER_CUSIP"]).size().rename("votes").reset_index()
    votes["share"] = votes.votes / votes.groupby("ticker").votes.transform("sum")
    chosen = votes[(votes.share >= .85) & (votes.votes >= 3)].copy()
    # A CUSIP must not be mapped to multiple target trading symbols.
    return chosen[~chosen.ISSUER_CUSIP.duplicated(False)]


def latest_filings(z, period):
    sub = read(z, "SUBMISSION", "ACCESSION_NUMBER FILING_DATE SUBMISSIONTYPE CIK PERIODOFREPORT")
    cover = read(z, "COVERPAGE", "ACCESSION_NUMBER AMENDMENTTYPE FILINGMANAGER_NAME FORM13FFILENUMBER")
    info = sub.merge(cover, on="ACCESSION_NUMBER")
    date = pd.Timestamp(period).strftime("%d-%b-%Y").upper()
    info = info[info.PERIODOFREPORT.eq(date) & info.SUBMISSIONTYPE.isin(["13F-HR", "13F-HR/A"])].copy()
    info["filed"] = pd.to_datetime(info.FILING_DATE, format="%d-%b-%Y")
    cutoff = next(p[1] for p in PERIODS if p[0] == period)
    if info.filed.max() > pd.Timestamp(cutoff):
        raise ValueError(f"13F archive contains filings after {cutoff}")
    info.sort_values(["filed", "ACCESSION_NUMBER"], inplace=True)
    # Last original/restatement replaces earlier rows. Subsequent additive amendments
    # supply only new holdings. Unrecognized amendments cannot replace the original.
    replace = info.SUBMISSIONTYPE.eq("13F-HR") | info.AMENDMENTTYPE.eq("RESTATEMENT")
    bases = info[replace].groupby("CIK").tail(1)
    info = info.join(bases.set_index("CIK").ACCESSION_NUMBER.rename("base"), on="CIK")
    added = info.AMENDMENTTYPE.eq("NEW HOLDINGS")
    order = dict(zip(info.ACCESSION_NUMBER, range(len(info))))
    keep = (info.ACCESSION_NUMBER == info.base) | (added & (info.ACCESSION_NUMBER.map(order)>info.base.map(order)))
    return info[keep].copy()


def count_shares(z, period, mapping):
    filers = latest_filings(z, period)
    links = read(z, "OTHERMANAGER2", "ACCESSION_NUMBER SEQUENCENUMBER CIK FORM13FFILENUMBER NAME")
    managers = filers.drop_duplicates("CIK").set_index("CIK")
    by_file = dict(zip(managers.FORM13FFILENUMBER, managers.index))
    by_name = dict(zip(managers.FILINGMANAGER_NAME.str.upper(), managers.index))
    linked_cik = {}
    for r in links.itertuples():
        cik = r.CIK or by_file.get(r.FORM13FFILENUMBER) or by_name.get(r.NAME.upper())
        if cik:
            linked_cik[(r.ACCESSION_NUMBER, r.SEQUENCENUMBER)] = cik
    codes = set(mapping.ISSUER_CUSIP)
    by_acc = filers.set_index("ACCESSION_NUMBER").CIK.to_dict()
    pieces = []
    for df in read(z, "INFOTABLE", "ACCESSION_NUMBER CUSIP SSHPRNAMT SSHPRNAMTTYPE PUTCALL VALUE OTHERMANAGER", True):
        take = df.ACCESSION_NUMBER.isin(by_acc) & df.CUSIP.isin(codes) & df.PUTCALL.eq("") & df.SSHPRNAMTTYPE.eq("SH")
        pieces.append(df.loc[take, ["ACCESSION_NUMBER", "CUSIP", "SSHPRNAMT", "VALUE", "OTHERMANAGER"]].copy())
    holdings = pd.concat(pieces, ignore_index=True)
    holdings["shares"] = pd.to_numeric(holdings.SSHPRNAMT)
    holdings["px"] = pd.to_numeric(holdings.VALUE)/holdings.shares
    med = holdings.groupby("CUSIP").px.transform("median")
    holdings = holdings[(holdings.shares > 0) & (holdings.px/med).between(.75,1.25)].copy()
    holdings["manager"] = holdings.ACCESSION_NUMBER.map(by_acc)
    held = set(zip(holdings.manager, holdings.CUSIP))
    overlap = []
    for r in holdings[holdings.OTHERMANAGER.ne("")].itertuples():
        others = [linked_cik.get((r.ACCESSION_NUMBER, n)) for n in re.findall(r"\d+", r.OTHERMANAGER)]
        if any(cik != r.manager and (cik, r.CUSIP) in held for cik in others):
            overlap.append(r.Index)
    holdings["lower"] = holdings.shares
    holdings.loc[overlap, "lower"] = 0
    counts = holdings.groupby("CUSIP")[["shares", "lower"]].sum()
    return mapping.join(counts, on="ISSUER_CUSIP")


def run(folder=HERE):
    names = (set(pd.read_csv(folder/"membership.csv").symbol) if folder != HERE else
             set(pd.read_csv(HERE/"technical_inputs/price.csv", index_col=0, nrows=0).columns))
    rows = []
    for period, cutoff, nport, form in PERIODS:
        # An ignored cache permits refreshing the denominator without reading 2GB again.
        tag = "mid" if folder != HERE else "large"
        cache = CACHE / f"ownership_{tag}_{period}.csv"
        if not cache.exists():
            print("Mapping", period, flush=True)
            with archive(nport+"_nport.zip", "dera/data/form-n-port-data-sets") as z:
                mapping = cusips(z, names)
            print("Counting", period, len(mapping), flush=True)
            with archive(form+"_form13f.zip", "datastandardsinnovation/data/form-13f-data-sets") as z:
                count_shares(z, period, mapping).to_csv(cache,index=False)
    prices = pd.read_csv(HERE/"technical_inputs/price.csv", index_col=0)
    prices = prices.reindex(columns=sorted(names))
    facts = pd.read_csv(folder/"fundamental_inputs/as_filed.csv")
    actions = pd.read_csv(folder/"technical_inputs/split_events.csv")
    for period, cutoff, nport, form in PERIODS:
        counts = pd.read_csv(CACHE/f"ownership_{tag}_{period}.csv",dtype={"ISSUER_CUSIP":str})
        known = facts[(facts.filed <= cutoff) & (facts.share_date <= period) & (facts.shares > 0)].copy()
        known = known.sort_values(["share_date", "filed", "accession"]).groupby("ticker").tail(1).set_index("ticker")
        counts = counts.join(known[["share_date", "filed", "shares"]].rename(columns={"shares":"denominator"}), on="ticker")
        for i, r in counts.dropna(subset=["denominator"]).iterrows():
            events = actions[(actions.ticker == r.ticker) & (actions.date > r.share_date) & (actions.date <= period)]
            counts.at[i, "denominator"] *= events.ratio.prod()
        counts["quarter"], counts["filings_through"] = period, cutoff
        counts["lower_fraction"] = counts.lower/counts.denominator
        counts["upper_fraction"] = counts.shares/counts.denominator
        age = pd.Timestamp(period)-pd.to_datetime(counts.share_date)
        # Firm-wide or unresolved multi-class denominators and ambiguous overlap near
        # 40% stay unknown. Extremely implausible totals are never evidence for passing.
        multi = {"GOOG","GOOGL","FOX","FOXA","NWS","NWSA","BRK-B","BF-B","HEI","HEI-A"}
        valid = ~counts.ticker.isin(multi) & age.dt.days.between(0,180) & counts.upper_fraction.between(0,1.25)
        valid &= (counts.lower_fraction > .4) | (counts.upper_fraction <= .4)
        counts["ratio"] = counts.lower_fraction.where(valid)
        rows.append(counts)
    publish(pd.concat(rows, ignore_index=True), prices, folder)


def publish(audit, prices, folder=HERE):
    # SEC issuer-wide common-share counts cannot resolve multiple classes. For
    # well-known unlisted voting share classes, leave the symbol out explicitly.
    uncertain_class = {"ABNB", "DASH", "PLTR", "SNAP", "META", "RDDT", "TTD", "DELL", "MOG-A", "MOG-B", "UA", "UAA"}
    audit.loc[audit.ticker.isin(uncertain_class), "ratio"] = float("nan")
    populated = audit.dropna(subset=["ratio"])
    assert (populated.filed <= populated.filings_through).all()
    assert (populated.share_date <= populated.quarter).all()
    assert (populated.lower_fraction <= populated.upper_fraction).all()
    assert populated.ratio.between(0,1.25).all()
    audit.to_csv(folder/"ownership_quarters.csv", index=False)
    days = prices.index[(prices.index >= "2025-12-31") & (prices.index <= "2026-11-17")]
    result = pd.DataFrame(index=days, columns=prices.columns, dtype=float)
    for cutoff, table in audit.groupby("filings_through"):
        result.loc[result.index > cutoff] = table.set_index("ticker").ratio.reindex(result.columns).to_numpy()
    result.to_csv(folder/"ownership_proxy.csv", index_label="signal_close")
    meta = {"unit":"fraction of point-in-time shares outstanding", "first_signal_close":str(days[0]),
            "archives":PERIODS, "usable_by_quarter":audit.groupby("quarter").ratio.count().to_dict(),
            "source_13f":"https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets",
            "source_nport":"https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets",
            "denominator":"Latest public share observation at or before quarter end, no older than 180 days, with filing date no later than cutoff. Split-adjusted to the same quarter end. Counts can be stale and shares by class may be unavailable.",
            "filters":"Keep positions priced within +/-25% of the same CUSIP median reported per-share value. Exclude unresolved total >125%, unresolved cross-manager threshold crossings, known multi-class denominators. Missing means unverified and fails the research gate.",
            "method":"Quarter-end SEC 13F original/restatement plus incremental holdings, filed through end of second month after reference quarter; next calendar date for closing signal. Q2 held through Nov17. Common equity CUSIP only, shares only, options excluded. SEC N-PORT separate source quarter, ticker/CUSIP votes 85%+ and three filings minimum. Aggregate lower estimate removes overlapping explicitly named other reporting managers, unresolved overlaps are possible. If overlap ambiguity straddles 40%, report missing. No exact Finviz data."}
    (folder/"ownership_period.json").write_text(json.dumps(meta, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--midcaps", action="store_true")
    run(HERE/"midcaps" if cmd.parse_args().midcaps else HERE)
