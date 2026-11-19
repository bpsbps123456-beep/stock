"""Supplement the daily, weekly and monthly screen with a fixed pre-2026 midcap roster."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

import technicals as T
from universe import API, HEADERS, broad_members

HERE = Path(__file__).resolve().parent
OUT = HERE / "midcaps"
FROZEN = OUT / "technical_inputs"
REVISION = 1329819820
ALIASES = {"ASGN": "EFOR", "BKH": "HZN", "PSTG": "P", "TCBI": "TXCP", **T.RENAMES}
# First open without independent trading and default consideration per share.
SEC = "https://www.sec.gov/Archives/edgar/data/"
DEALS = {
    "BLD": ("2026-07-01", 0., "QXO", 20.2, SEC+"1236275/000162828026073568/R10.htm"),
    "CADE": ("2026-02-02", 0., "HBAN", 2.475, SEC+"49196/000004919626000090/hban-20260930.htm"),
    "CIVI": ("2026-01-30", 0., "SM", 1.45, SEC+"1509589/000110465926008521/tm264358d6_8k.htm"),
    "CMA": ("2026-02-02", 0., "FITB", 1.8663, SEC+"35527/000003552726000243/fitb-20260930.htm"),
    "FYBR": ("2026-01-20", 38.5, "", 0., SEC+"732712/000119312526016059/d198618d8k.htm"),
    "GTLS": ("2026-07-16", 210., "", 0., SEC+"1701605/000170160526000035/bkr-20260930.htm"),
    "JHG": ("2026-07-01", 52., "", 0., SEC+"1274173/000110465926079401/tm2619303d2_ex99-1.htm"),
    "MASI": ("2026-06-10", 180., "", 0., SEC+"313616/000031361626000199/dhr-20260925.htm"),
    "NSA": ("2026-07-22", 0., "PSA", .14, SEC+"1393311/000162828026067286/psa-20260930.htm"),
    "NWE": ("2026-11-16", 0., "HZN", .98, SEC+"1993004/000199300426000072/nwe-20261113.htm"),
    "PCH": ("2026-02-02", .61, "RYN", 1.8185, "https://nasdaqtrader.com/TraderNews.aspx?id=ECA2026-47"),
    "SNV": ("2026-01-02", 0., "PNFP", .5237, SEC+"1605855/000160585526000001/wk-form4_1767387574.xml"),
    "TMHC": ("2026-07-24", 72.5, "", 0., "https://www.berkshirehathaway.com/news/jul2426.pdf"),
    "WBS": ("2026-08-20", 48.75, "SAN", 2.0548, SEC+"801337/000087666126000696/ruleprovisionnotice.htm"),
}


def freeze_members():
    OUT.mkdir(exist_ok=True)
    params = {"action": "query", "prop": "revisions", "revids": REVISION, "rvslots": "main",
              "rvprop": "ids|timestamp|content", "formatversion": 2, "format": "json"}
    response = requests.get(API, params=params, headers=HEADERS, timeout=40)
    response.raise_for_status()
    rev = response.json()["query"]["pages"][0]["revisions"][0]
    assert rev["timestamp"] < "2026-01-01"
    table = rev["slots"]["main"]["content"].split('id="constituents"', 1)[1].split("\n|}", 1)[0]
    rows = []
    for para in re.split(r"\n\|-+[^\n]*\n", table)[2:]:
        match = re.search(r"\{\{(?:nysesymbol|nasdaqsymbol|bzx link)\s*\|\s*([^|}]+)", para, re.I)
        fields = para.split("\n| ")
        if not match or len(fields) < 5:
            raise ValueError(f"Cannot parse midcap constituent: {para[:150]}")
        sym = match.group(1).strip().replace(".", "-")
        cik = re.search(r"[?&]CIK=(\d+)", para, re.I)
        def plain(text):
            text = re.sub(r'^style="[^"]*"\s*\|\s*', '', text)
            return re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", text).strip()
        rows.append((sym, cik.group(1) if cik else "", plain(fields[1]), plain(fields[-2])))
    result = pd.DataFrame(rows, columns=["symbol","cik","company","headquarters"])
    assert 395 <= len(result) <= 405 and result.symbol.is_unique
    result.to_csv(OUT / "membership.csv", index=False)
    (OUT / "membership_source.json").write_text(json.dumps({
        "revision": rev["revid"], "timestamp_utc": rev["timestamp"],
        "url": f"https://en.wikipedia.org/w/index.php?oldid={REVISION}",
        "roster": "Fixed December 2025 membership; retained until actual corporate termination.",
        "securities": len(result)}, indent=2), encoding="utf-8")


def prefetch():
    if not (OUT / "membership.csv").exists():
        freeze_members()
    symbols = pd.read_csv(OUT / "membership.csv").symbol
    extra = {v[2] for v in DEALS.values()} - {""}
    fetch = {ALIASES.get(s,s) for s in symbols if s not in DEALS}
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(T.fetch, sorted(fetch | extra)))
    return list(symbols)


def freeze():
    symbols = prefetch()
    existing = T.load_panels()
    dates = pd.DatetimeIndex(existing["close"].index)
    last_quote = {s: dates[dates < pd.Timestamp(terms[0])][-1].date().isoformat() for s, terms in DEALS.items()}
    components, events = {}, []
    FROZEN.mkdir(exist_ok=True)
    with patch.dict(T.SIP_END, last_quote):
        for s in symbols:
            underlying = ALIASES.get(s,s)
            b = T.read_bars(underlying)
            if b is None:
                raise ValueError(f"No historical data for {s}")
            f = T.indicators(b).reindex(dates)
            if s in DEALS:
                effective, cash, delivered, fraction, _ = DEALS[s]
                assert b.index[-1].date().isoformat() == last_quote[s], s
                f.loc[effective:, "tradable"] = 0
                settle = pd.Series({"open": cash, "close": cash})
                if delivered:
                    quote = T.read_bars(delivered).loc[effective]
                    settle += quote[["open","close"]] * quote.split_mult * fraction
                f.loc[effective, ["open","close"]] = settle
            elif f["close"].isna().any():
                raise ValueError(f"Missing required midcap historical quote for {s}: {f.index[f.close.isna()].tolist()[:3]}")
            diff = (b.split_mult.shift()/b.split_mult).dropna()
            events.extend((i.date().isoformat(), s, ratio) for i,ratio in diff.items() if ratio != 1)
            components[s] = f
    for field in T.FIELDS:
        wide = pd.DataFrame({s: f[field] for s, f in components.items()})
        wide.to_csv(FROZEN/(field+".csv"), index_label="date", float_format="%.15g")
    pd.DataFrame(sorted(events), columns=["date","ticker","ratio"]).to_csv(FROZEN/"split_events.csv", index=False)
    pd.DataFrame.from_dict(DEALS, orient="index", columns=["from_date","cash_usd","delivered","shares","source"]).to_csv(
        FROZEN/"terminal_events.csv", index_label="ticker")


def run():
    from compare import (blend, close_terminated, desk, load_book, market_gates, relative_target,
                         replay, scheduled, summarize, switched_targets)
    from fundamentals import load_fund_pass
    book = load_book()
    large = T.load_panels()
    mid = {field: pd.read_csv(FROZEN/(field+".csv"), index_col=0) for field in T.FIELDS}
    for field in ("ma_pass","technical_pass","non_ma_pass","tradable"):
        mid[field] = mid[field].eq(1)
    np.testing.assert_array_equal(mid["technical_pass"],mid["ma_pass"] & mid["non_ma_pass"])
    dates = large["close"].index.tolist()
    assert dates[0] == T.FIRST and dates[-1] == T.LAST and dates == mid["close"].index.tolist()
    prior, days = dates[:-1], dates[1:]

    def masks(panels, path, supplement):
        financial = load_fund_pass(panels["price"], midcaps=supplement)
        eligible = {"ma": panels["ma_pass"], "technical": panels["technical_pass"],
                    "quality": panels["technical_pass"] & financial.eq(True)}
        own = pd.read_csv(path/"ownership_proxy.csv", index_col=0)
        eligible["ownership"] = eligible["quality"] & (own > .4)
        eligible["without_ma"] = panels["non_ma_pass"] & financial.eq(True) & (own > .4)
        return eligible

    big_screen = masks(large, HERE, False)
    mid_screen = masks(mid, OUT, True)
    symbols = sorted({ALIASES.get(s,s) for s in set(large["close"]) | set(mid["close"])})
    loc = {s: i for i,s in enumerate(symbols)}
    def combined(field):
        # Multiple labels of the same issuer map to one position even when both cohorts hold it.
        all_columns = pd.concat([large[field], mid[field]], axis=1)
        all_columns.columns = [ALIASES.get(s,s) for s in all_columns]
        return all_columns.loc[:,~all_columns.columns.duplicated()].reindex(columns=symbols)
    def prices(field, ref):
        return ref.reindex(index=days,columns=symbols).combine_first(combined(field).loc[days]).to_numpy()
    opn = prices("open",book.opn)
    close = prices("close",book.cls)
    allowed = combined("tradable").loc[days].to_numpy(bool)
    fx = book.fx[-len(days):] / book.fx[-len(days)-1]
    ref = desk.run_model(book)
    market = market_gates(book, days)["SOXX100r1"]
    base = {}
    for day,weights in ref.rebalances.items():
        w = np.zeros(len(symbols))
        for key,weight in weights.items():
            s = book.ticker[key]
            w[loc[ALIASES.get(s,s)]] = weight
        base[days.index(day)] = w
    allow_all = np.ones(len(days), dtype=bool)
    ref_curve = replay(days,opn,close,fx,base,allow_all)
    np.testing.assert_allclose(ref_curve.krw, desk.model_index(book,ref)/100, rtol=1e-10)
    rv_large = large["relvol"].to_dict(orient="index")
    rv_mid = mid["relvol"].to_dict(orient="index")
    mid_universe = set(mid["close"])
    broad_scope = [broad_members(d) for d in days]
    # Within an overlapping issuer-date the point-in-time broad roster takes precedence.
    mid_scope = []
    for members in broad_scope:
        broad_issuers = {ALIASES.get(s,s) for s in members}
        mid_scope.append({s for s in mid_universe if ALIASES.get(s,s) not in broad_issuers})
    log, results = [], {}
    for screen in big_screen:
        big_pass = [set(big_screen[screen].columns[big_screen[screen].loc[p].fillna(False)])
                    for p in prior]
        mid_pass = [set(mid_screen[screen].columns[mid_screen[screen].loc[p].fillna(False)]) for p in prior]
        for freq in ["daily","weekly","monthly"]:
            plan = {}
            for j,(day,p) in enumerate(zip(days,prior)):
                if not scheduled(days,j,freq):
                    continue
                big = big_pass[j] & broad_scope[j]
                new = mid_pass[j] & mid_scope[j]
                ranks = {ALIASES.get(s,s): rv_large[p][s] for s in big}
                ranks.update({ALIASES.get(s,s): rv_mid[p][s] for s in new})
                chosen = relative_target(set(ranks),ranks)
                w = np.zeros(len(symbols))
                for s,weight in chosen.items():
                    w[loc[s]] = weight
                plan[j] = w
                log.append((screen,freq,day,p,len(big),len(new),"|".join(sorted(chosen))))
            plan = close_terminated(plan,opn,allowed)
            frame = replay(days,opn,close,fx,plan,allow_all)
            results[f"broad_{screen}_{freq}"] = summarize(frame)
            if screen == "ownership":
                for label, schedule in [("blend",blend(days,opn,base,plan)),
                                        ("switch",switched_targets(days,opn,base,plan,market))]:
                    combined_curve = replay(days,opn,close,fx,schedule,allow_all)
                    results[f"{label}_ownership_{freq}"] = summarize(combined_curve)
    pd.DataFrame(results).T.rename_axis("strategy").to_csv(OUT/"results.csv")
    pd.DataFrame(log,columns=["filter","frequency","trade_open","signal_close","large_eligible",
                              "additional_mid_eligible","selected_before_terminations"]).to_csv(OUT/"decisions.csv",index=False)
    size = [len(b) + len(m) for b,m in zip(broad_scope,mid_scope)]
    active_size = [sum(bool(allowed[i,loc[ALIASES.get(s,s)]]) for s in broad_scope[i] | mid_scope[i])
                   for i in range(len(days))]
    (OUT/"coverage.json").write_text(json.dumps({"fixed_midcap_members":len(mid_universe),"primary_members_range":
        [min(map(len,broad_scope)),max(map(len,broad_scope))],"unique_combined_min":min(size),"unique_combined_max":max(size),
        "live_combined_min":min(active_size),"live_combined_max":max(active_size),
        "midcap_acquired":len(DEALS),"midcap_price_missing":0,"screen_paths":15,"mixed_scope_checks":6,
        "eligibility": "Monthly archived large cap plus a fixed pre-2026 midcap roster. Every name remains in scope through conversion. "
        "Published roster label aliases merged by issuer. Merged cash and stock amounts realized at the first public conversion open. "
        "Large cap proxy and supplemental midcap use the same disclosed EOD, 13F and SEC reporting approximations.",
        "prices": "SIP historical split and raw bars for 14 terminated labels, Yahoo for continuing midcap labels. "
        "BLD default delivery assumes no election, yielding 20.2 QXO shares on July 1."},indent=2),encoding="utf-8")


if __name__ == "__main__":
    import sys
    if "--refresh" in sys.argv or not (FROZEN/"split_events.csv").exists():
        freeze()
    if "--inputs-only" not in sys.argv:
        run()
