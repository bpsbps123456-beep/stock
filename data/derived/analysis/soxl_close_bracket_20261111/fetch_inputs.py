"""Freeze SOXL 2026 SIP bars through Nov 10 for the fixed close bracket study.

Credentials and minute fetching reuse hatta_bar/code/fetch_alpaca.py. No prior
study cache is changed. Repeat runs reuse this study's frozen caches by default.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
ET = "America/New_York"
START = "2026-01-01"
END_EXCLUSIVE = "2026-11-11"
LAST_DAY = "2026-11-10"
API = "https://data.alpaca.markets/v2/stocks/bars"
DOCS = {
    "historical_bars": "https://docs.alpaca.markets/reference/stockbars",
    "bar_construction": "https://docs.alpaca.markets/docs/market-data-faq",
    "minute_timestamps": "https://alpaca.markets/learn/stock-minute-bars",
}


def fetcher():
    path = ANALYSIS / "hatta_bar/code/fetch_alpaca.py"
    spec = importlib.util.spec_from_file_location("close_bracket_shared_fetch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_bound(day):
    return pd.Timestamp(day, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def daily_request(headers, pace):
    params = {
        "symbols": "SOXL", "timeframe": "1Day", "start": utc_bound(START),
        "end": utc_bound(END_EXCLUSIVE), "limit": 10000,
        "adjustment": "split", "feed": "sip", "sort": "asc",
    }
    records = []
    while True:
        for attempt in range(6):
            pace.wait()
            try:
                response = requests.get(API, params=params, headers=headers, timeout=60)
                if response.status_code == 429 or response.status_code >= 500:
                    time.sleep(2 + attempt * 2)
                    continue
                response.raise_for_status()
                break
            except requests.exceptions.RequestException:
                if attempt == 5:
                    raise
                time.sleep(2 + attempt * 2)
        else:
            raise RuntimeError("SOXL daily SIP request retries exhausted")
        body = response.json()
        records.extend((body.get("bars") or {}).get("SOXL", []))
        token = body.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    if not records:
        raise RuntimeError("SOXL daily SIP request returned no bars")
    frame = pd.DataFrame(records)
    frame.index = pd.to_datetime(frame.pop("t"), utc=True).dt.tz_convert(ET)
    frame = frame.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close",
                                  "v": "Volume", "n": "Trades", "vw": "VWAP"})
    return frame[["Open", "High", "Low", "Close", "Volume", "Trades", "VWAP"]].sort_index()


def validate_ohlc(frame, label):
    assert not frame.index.has_duplicates, f"{label}: duplicate timestamps"
    assert str(frame.index.tz) == ET, f"{label}: timezone mismatch"
    values = frame[["Open", "High", "Low", "Close"]].to_numpy()
    assert np.isfinite(values).all(), f"{label}: nonfinite OHLC"
    assert (frame.Low > 0).all(), f"{label}: nonpositive price"
    assert (frame.High >= frame[["Open", "Close"]].max(axis=1)).all(), f"{label}: high"
    assert (frame.Low <= frame[["Open", "Close"]].min(axis=1)).all(), f"{label}: low"


def main():
    cache = HERE / "cache"
    out = HERE / "results"
    recent_dir = cache / "recent"
    for path in (cache, out, recent_dir):
        path.mkdir(parents=True, exist_ok=True)
    mod = fetcher()
    pace = mod.Pace()
    daily_path = cache / "daily.parquet"
    recent_path = recent_dir / "SOXL.parquet"
    provenance_path = cache / "fetch_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    if not daily_path.exists() or not recent_path.exists():
        headers = mod.keys()
        if not daily_path.exists():
            daily_request(headers, pace).to_parquet(daily_path)
            provenance["daily_fetched_utc"] = datetime.now(timezone.utc).isoformat()
        if not recent_path.exists():
            mod.one("SOXL", utc_bound("2026-11-03"), utc_bound(END_EXCLUSIVE),
                    headers, pace, str(recent_dir))
            provenance["recent_fetched_utc"] = datetime.now(timezone.utc).isoformat()
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    oldbase = ANALYSIS / "sox_open_switch/cache"
    sources = [oldbase / "alp1m/SOXL.parquet", oldbase / "alp1m_recent/SOXL.parquet", recent_path]
    frames = [pd.read_parquet(p).loc[START:LAST_DAY].sort_index() for p in sources]
    comparisons = []
    for i, frame in enumerate(frames):
        validate_ohlc(frame, f"minute source {i}")
        for j in range(i):
            common = frame.index.intersection(frames[j].index)
            if not len(common):
                continue
            x = frame.loc[common, ["Open", "High", "Low", "Close"]].to_numpy()
            y = frames[j].loc[common, ["Open", "High", "Low", "Close"]].to_numpy()
            relative = float(np.max(np.abs(x / y - 1)))
            assert np.allclose(x, y, rtol=1e-10, atol=1e-10), f"minute overlaps disagree: {j}/{i}: {relative}"
            comparisons.append({"source_a": j, "source_b": i, "bars": len(common),
                                "max_relative_ohlc_error": relative})
    minute = pd.concat(frames).sort_index()
    minute = minute[~minute.index.duplicated(keep="last")]
    validate_ohlc(minute, "minute union")
    daily = pd.read_parquet(daily_path).loc[START:LAST_DAY].sort_index()
    validate_ohlc(daily, "daily")
    assert (daily.index.hour == 0).all() and (daily.index.minute == 0).all(), "daily timestamp not ET midnight"
    schedule = xc.get_calendar("XNYS").schedule.loc[START:LAST_DAY]
    expected_days = [str(d.date()) for d in schedule.index]
    actual_days = [str(d.date()) for d in daily.index]
    assert expected_days == actual_days, "daily dates differ from expected XNYS calendar"
    assert len(daily) == 216, f"expected216daily bars, got{len(daily)}"
    coverage, difference = [], []
    for day, row in schedule.iterrows():
        op, cl = row["open"].tz_convert(ET), row["close"].tz_convert(ET)
        regular = minute[(minute.index >= op) & (minute.index < cl)]
        expected = pd.date_range(op, cl - pd.Timedelta(minutes=1), freq="min")
        missing = expected.difference(regular.index)
        assert len(missing) == 0, f"missing RTH minutes on{day.date()}: {len(missing)}"
        assert (regular.Volume > 0).all(), f"nonpositive RTH volume on{day.date()}"
        drow = daily.loc[daily.index.date == day.date()].iloc[0]
        last_rth = float(regular.Close.iloc[-1])
        atclose = float(minute.loc[cl, "Open"]) if cl in minute.index else None
        coverage.append({"day": str(day.date()), "open_et": op.isoformat(), "close_et": cl.isoformat(),
                         "expected_rth_minutes": len(expected), "observed_rth_minutes": len(regular),
                         "missing_rth_minutes": len(missing), "close_timestamp_minute_available": cl in minute.index})
        difference.append({"day": str(day.date()), "daily_close": float(drow.Close),
                           "last_rth_minute_close": last_rth, "atclose_minute_open": atclose,
                           "last_rth_minus_daily_bp": (last_rth / float(drow.Close) - 1) * 10000,
                           "atclose_minus_daily_bp": None if atclose is None else (atclose / float(drow.Close) - 1) * 10000})
    minute_path = cache / "minute.parquet"
    minute.to_parquet(minute_path)
    daily.index.name = "timestamp"
    daily.to_csv(out / "daily_prices.csv")
    diff = pd.DataFrame(difference)
    diff.to_csv(out / "daily_close_vs_minute.csv", index=False)
    minute_minutes = minute.index.hour * 60 + minute.index.minute
    assert ((minute_minutes >= 240) & (minute_minutes < 1200)).all()
    audit = {
        "pass": True, "symbol": "SOXL", "start": str(daily.index[0].date()), "end": LAST_DAY,
        "sessions": len(daily), "timezone": ET, "feed": "sip", "adjustment": "split",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "fetch": provenance,
        "daily_api_parameters": {"timeframe": "1Day", "start": START, "end_exclusive": END_EXCLUSIVE,
                                 "feed": "sip", "adjustment": "split"},
        "minute_sources": [{"path": str(p.relative_to(ANALYSIS)), "sha256": digest(p), "used_bars": len(f)}
                           for p, f in zip(sources, frames)],
        "overlap_checks": comparisons,
        "minute_output": {"path": "cache/minute.parquet", "sha256": digest(minute_path),
                          "bars": len(minute), "first": minute.index[0].isoformat(), "last": minute.index[-1].isoformat()},
        "daily_output": {"path": "cache/daily.parquet", "sha256": digest(daily_path), "bars": len(daily)},
        "coverage": coverage,
        "close_comparison_bp": {key: {"median_absolute_bp": float(diff[key].abs().median()),
                                       "maximum_absolute_bp": float(diff[key].abs().max())}
                                for key in ["last_rth_minus_daily_bp", "atclose_minus_daily_bp"]},
        "daily_close_interpretation": "Split-adjusted SIP 1Day Close; provider construction excludes extended-hours T-condition trades from daily OHLC. Treat as daily regular close benchmark, not guaranteed user's closing-auction fill.",
        "minute_interpretation": "SIP split-adjusted trade OHLC for04:00<=ET<20:00. Missing extended-hour minutes are not interpolated; no20:00-04:00/ATS data.",
        "official_docs": DOCS,
    }
    (out / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"pass": True, "sessions": len(daily), "minute_bars": len(minute),
                      "end": LAST_DAY, "overlap": comparisons, "paths": [str(minute_path), str(daily_path)]}, indent=2))


if __name__ == "__main__":
    main()
