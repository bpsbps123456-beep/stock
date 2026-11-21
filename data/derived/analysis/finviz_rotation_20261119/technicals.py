"""Replay the Finviz URL's technical conditions at the US regular close."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "desk_defense_20261119"
CACHE = HERE / "cache" / "yahoo"
FROZEN = HERE / "technical_inputs"
FIRST, LAST = "2025-12-31", "2026-11-17"
FIELDS = ("open", "close", "price", "ma_pass", "non_ma_pass", "technical_pass", "relvol", "split_mult", "tradable")
URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
RENAMES = {"MMC": "MRSH", "BK": "BNY", "SATS": "ECHO", "PSKY": "SKYD"}
SIP_END = {"CTRA": "2026-05-06", "HOLX": "2026-04-06", "DAY": "2026-02-03", "SNDK": LAST,
           "AVB": "2026-08-14", "EA": "2026-08-04", "KVUE": "2026-10-30", "WBD": "2026-10-05"}
RENAMES["EQR"] = "VMRK"  # Continuing unit, documented 1:1; prior EQR label until the August merger.
# Last independent quote, first settlement open, guaranteed USD cash and delivery of successor shares.
MERGERS = {
    "CTRA": ("2026-05-07", 0., "DVN", .70, "https://www.sec.gov/Archives/edgar/data/1090012/000119312526451216/R10.htm"),
    "HOLX": ("2026-04-07", 76., "", 0., "https://www.hologic.com/about/press-release/blackstone-and-tpg-complete-acquisition-hologic"),
    "DAY": ("2026-02-04", 70., "", 0., "https://www.thomabravo.com/press-releases/thoma-bravo-completes-acquisition-of-dayforce"),
    "AVB": ("2026-08-17", 0., "VMRK", 2.793, "https://investors.vivmarkresidential.com/sec-filings/all-sec-filings/content/0001140361-26-033377/ef20080318_8k.htm"),
    "EA": ("2026-08-05", 210., "", 0., "https://www.sec.gov/Archives/edgar/data/712515/000114036126031157/ef20079099_8k.htm"),
    "KVUE": ("2026-11-02", 3.50, "KMB", .14625, "https://www.sec.gov/Archives/edgar/data/55785/000110465926120990/tm2628636d1_8k.htm"),
    "WBD": ("2026-10-06", 31.01666668, "", 0., "https://www.sec.gov/Archives/edgar/data/2041610/000110465926113913/tm2626659d7_ex99-1.htm"),
}


def time_symbols():
    names = pd.read_csv(PREVIOUS / "frozen" / "prices_close.csv", nrows=0).columns
    return sorted(set(names) - {"Date", "QQQ", "SOXX", "KRW=X", "456600.KS"})


def fetch(symbol):
    """Download Yahoo's split-adjusted quoted prices, share volume and split events."""
    CACHE.mkdir(exist_ok=True, parents=True)
    if symbol in SIP_END:
        return
    symbol = RENAMES.get(symbol, symbol)
    path = CACHE / f"{symbol}.json"
    if path.exists():
        return
    params = {"period1": int(pd.Timestamp("2025-01-01", tz="UTC").timestamp()),
              "period2": int(pd.Timestamp("2026-11-19", tz="UTC").timestamp()),
              "interval": "1d", "events": "splits"}
    for host in (URL, URL.replace("query1", "query2"), URL):
        try:
            result = requests.get(host + quote(symbol), params=params,
                                  headers={"User-Agent": "Mozilla/5.0"}, timeout=18)
            if result.status_code == 404:
                return
            result.raise_for_status()
            payload = result.json()["chart"]["result"][0]
            if payload.get("timestamp"):
                path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
                return
        except (requests.RequestException, KeyError, IndexError, TypeError):
            continue


def sip_bars(symbol):
    """Retain acquired issuers even when Yahoo no longer publishes their symbol."""
    path = CACHE.parent / f"sip_{symbol}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    script = HERE.parent / "alpaca_sip" / "fetch_alpaca.py"
    spec = importlib.util.spec_from_file_location("historical_sip", script)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    params = {"symbols": symbol, "start": "2025-01-01", "end": SIP_END[symbol] + "T23:59:59Z",
              "asof": "-", "limit": 10000, "timeframe": "1Day", "feed": "sip"}
    bars = {}
    for adjustment in ("raw", "split"):
        response = requests.get(api.URL, params={**params, "adjustment": adjustment},
                                headers=api.keys(), timeout=30)
        response.raise_for_status()
        data = response.json()
        assert not data.get("next_page_token")
        bar = pd.DataFrame(data["bars"][symbol])
        bar.index = pd.to_datetime(bar.t, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        bars[adjustment] = bar.rename(columns={"o": "open", "c": "close", "v": "volume"})
    traded = bars["split"][["open", "close", "volume"]].copy()
    traded["split_mult"] = bars["raw"].close / traded.close
    if not traded.split_mult.eq(1).all():
        raise ValueError(f"Inspect split dates for SIP issuer {symbol}")
    traded.to_csv(path)
    return traded


def read_bars(symbol):
    if symbol in SIP_END:
        return sip_bars(symbol)
    path = CACHE / f"{RENAMES.get(symbol, symbol)}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    date = pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert("America/New_York")
    bars = pd.DataFrame(data["indicators"]["quote"][0], index=date.normalize().tz_localize(None))
    bars = bars.dropna(subset=["open", "close", "volume"])
    if bars.empty:
        return None
    if bars.index.has_duplicates or (bars[["open", "close"]] <= 0).any().any():
        raise ValueError(f"Duplicate or invalid bar for {symbol}")
    factor = pd.Series(1., index=bars.index)
    for event in data.get("events", {}).get("splits", {}).values():
        day = pd.Timestamp(event["date"], unit="s", tz="UTC").tz_convert("America/New_York").date()
        factor.loc[factor.index < pd.Timestamp(day)] *= event["numerator"] / event["denominator"]
    bars["split_mult"] = factor
    return bars


def rsi(close, n=14):
    """Wilder smoothing, seeded by the first n changes."""
    change = close.diff().dropna()
    if len(change) < n:
        return pd.Series(np.nan, index=close.index)
    gain, loss = change.clip(lower=0), (-change).clip(lower=0)
    avgs = []
    for series in (gain, loss):
        series = series.copy()
        series.iloc[n-1] = series.iloc[:n].mean()
        avgs.append(series.iloc[n-1:].ewm(alpha=1/n, adjust=False).mean())
    avg_gain, avg_loss = avgs
    return (100 * avg_gain / (avg_gain + avg_loss)).reindex(close.index)


def indicators(bars):
    c, v, factor = bars["close"], bars["volume"], bars["split_mult"]
    a20, a50, a200 = [c.rolling(n).mean() for n in (20, 50, 200)]
    ma = (a20 > a50) & (c > a50) & (c > a200)
    # Compare historic volumes in the share units that existed at the signal close.
    typical = v.shift(1).rolling(63).mean()
    ratio = v / typical.where(typical > 0)
    price = c * factor
    non_ma = (rsi(c) > 40) & (price > 10) & (typical/factor > 500_000) & (ratio > 1)
    full = ma & non_ma
    return pd.DataFrame({"open": bars.open, "close": c, "price": price,
                         "ma_pass": ma.astype(int), "non_ma_pass": non_ma.astype(int),
                         "technical_pass": full.astype(int),
                         "relvol": ratio, "split_mult": factor, "tradable": 1})


def freeze(symbols):
    symbols = sorted(set(symbols))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, symbols))
    FROZEN.mkdir(exist_ok=True)
    prices = pd.read_csv(PREVIOUS / "frozen" / "prices_close.csv", index_col=0).QQQ.dropna()
    dates = pd.to_datetime(prices.loc[FIRST:LAST].index)
    panel, split_events = {}, []
    missing, early = [], {}
    for s in symbols:
        b = read_bars(s)
        if b is None:
            missing.append(s)
            continue
        frame = indicators(b).reindex(dates)
        if frame.close.notna().sum() == 0:
            missing.append(s)
            continue
        last = frame.close.last_valid_index().date().isoformat()
        if last < LAST:
            early[s] = last
        # Mark known forced conversions at settlement. Preserve missing ordinary quotes,
        # so a missing observation cannot become a backdated delist or invented fill.
        frame["tradable"] = 1
        if s in MERGERS:
            effective, cash, underlying, ratio, _ = MERGERS[s]
            frame.loc[effective:, "tradable"] = 0
            values = pd.Series({"open": cash, "close": cash})
            if underlying:
                conversion = read_bars(underlying).loc[effective]
                values += conversion[["open", "close"]] * conversion.split_mult * ratio
            frame.loc[effective, ["open", "close"]] = values
        panel[s] = frame
        factor = b.split_mult.copy()
        changed = (factor.shift() / factor).dropna()
        split_events.extend((d.date().isoformat(), s, x) for d, x in changed.items() if x != 1)
    digests = {}
    for field in FIELDS:
        p = FROZEN / (field + ".csv")
        wide = pd.DataFrame({s: frame[field] for s, frame in panel.items()}, index=dates)
        wide.to_csv(p, float_format="%.15g", index_label="date", lineterminator="\n")
        digests[p.name] = digest(p)
    split_path = FROZEN / "split_events.csv"
    pd.DataFrame(sorted(split_events), columns=["date", "ticker", "ratio"]).to_csv(split_path, index=False)
    digests[split_path.name] = digest(split_path)
    events_path = FROZEN / "terminal_events.csv"
    pd.DataFrame.from_dict(MERGERS, orient="index", columns=["from_date","cash_usd","successor",
                           "shares","source"]).to_csv(events_path, index_label="ticker")
    digests[events_path.name] = digest(events_path)
    meta = {"first_close": FIRST, "last_close": LAST, "requested": len(symbols), "downloaded": len(panel),
            "absent": missing, "end_before_cutoff": early,
            "technical": "SMA20>SMA50, Close>SMA50, Close>SMA200, Wilder RSI14>40, then-as-traded price>$10, "
                         "prior 63 sessions' mean volume>500000 shares in signal-date units and daily volume/mean>1. "
                         "Completed US regular closes. 63 prior sessions approximates three months; intraday ordering can differ.",
            "source": URL, "definitions": "https://finviz.com/help/screener",
            "sip": {"endpoint": "https://docs.alpaca.markets/us/reference/stockbars", "symbols": SIP_END},
            "renamed_equivalent_shares": RENAMES, "corporate_action_limitations": [
                "HOLX unlisted contingent right, future payment up to $3, is unmarked; $76 guaranteed cash only.",
                "PSKY/SKYD October warrants and ordinary dividends/spin distributions are not separately credited.",
                "AVB delivery uses 2.793 times same continuing EQR quote August 17 under VMRK adjusted history.",
                "Forced cash/stock consideration is treated as available at the first conversion open."],
            "data": "2025-01-01 through 2026-11-18 included for split events; price signals stop at Nov 17.",
            "files": digests}
    (FROZEN / "sources.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_panels():
    sources = json.loads((FROZEN / "sources.json").read_text(encoding="utf-8"))
    frames = {}
    for field in FIELDS:
        p = FROZEN / f"{field}.csv"
        if digest(p) != sources["files"][p.name]:
            raise ValueError(f"Frozen technical input changed: {field}")
        frame = pd.read_csv(p, index_col=0)
        frames[field] = frame.eq(1) if field in {"ma_pass", "non_ma_pass", "technical_pass", "tradable"} else frame
    np.testing.assert_array_equal(frames["technical_pass"], frames["ma_pass"] & frames["non_ma_pass"])
    return frames


if __name__ == "__main__":
    from universe import broad_symbols
    result = freeze([*time_symbols(), *broad_symbols(), "SPY"])
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
