"""Freeze SIP price inputs and compute the six pre-specified, pre-auction flags."""
from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent.parent
CACHE, OUT = HERE / "cache", HERE / "results"
WARM, START, END, END_EXCLUSIVE = "2025-12-01", "2026-01-02", "2026-11-17", "2026-11-18"
TZ = "America/New_York"
FLAGS = ("sector_sma20", "above_vwap", "late_up", "rsi_turn", "down5", "bb1")
MINUTE = pd.Timedelta(minutes=1)
FIELDS = ["Open", "High", "Low", "Close", "Volume", "Trades", "VWAP"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("fixed_bracket_inputs", HERE.parent / "fetch_inputs.py")
htf = load_module("existing_wilder", ANALYSIS / "dip_reason/code/htf.py")


def source(path):
    return {"path": path.relative_to(ANALYSIS).as_posix(), "sha256": base.digest(path)}


def snapshot_prices():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    shared = base.fetcher()
    pace = shared.Pace()
    fetch_log = CACHE / "retrieved.json"
    retrieved = json.loads(fetch_log.read_text(encoding="utf-8")) if fetch_log.exists() else {}

    def record(label):
        retrieved[label] = datetime.now(timezone.utc).isoformat()
        fetch_log.write_text(json.dumps(retrieved, indent=2), encoding="utf-8")

    def minute_snapshot(label, lo, hi):
        directory = CACHE / label
        directory.mkdir(exist_ok=True)
        path = directory / "SOXL.parquet"
        if not path.exists():
            _, n = shared.one("SOXL", base.utc_bound(lo), base.utc_bound(hi),
                              shared.keys(), pace, str(directory))
            assert n > 0, label
            record(label)
        return path

    daily_paths = {sym: CACHE / f"{sym}_warm_daily.parquet" for sym in ("SOXL", "SOXX")}
    if any(not p.exists() for p in daily_paths.values()):
        params = dict(symbols="SOXL,SOXX", timeframe="1Day", start=base.utc_bound(WARM),
                      end=base.utc_bound(END_EXCLUSIVE), feed="sip", adjustment="split",
                      limit=10000, sort="asc")
        retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=retry))
            response = session.get(base.API, headers=shared.keys(), params=params, timeout=60)
            response.raise_for_status()
        body = response.json()
        assert not body.get("next_page_token"), "unexpected daily pagination"
        for sym, path in daily_paths.items():
            df = pd.DataFrame(body["bars"][sym])
            df.index = pd.to_datetime(df.pop("t"), utc=True).dt.tz_convert(TZ)
            df = df.rename(columns=dict(zip(("o", "h", "l", "c", "v", "n", "vw"), FIELDS)))
            df[FIELDS].sort_index().loc[WARM:END].to_parquet(path)
        record("daily_SOXL_SOXX")

    daily, sector = (pd.read_parquet(p) for p in daily_paths.values())
    schedule = xc.get_calendar("XNYS").schedule.loc[WARM:END]
    for sym, df in (("SOXL", daily), ("SOXX", sector)):
        base.validate_ohlc(df, sym + " daily")
        assert list(df.index.date) == list(schedule.index.date), sym + " daily coverage"

    old_path = HERE.parent / "cache/minute.parquet"
    warm_path = ANALYSIS / "sox_open_switch/cache/alp1m/SOXL.parquet"
    if not warm_path.exists():
        warm_path = minute_snapshot("warm", WARM, START)
    warm = pd.read_parquet(warm_path).loc[WARM:"2025-12-31"]
    assert len(warm) and str(warm.index[0].date()) == WARM
    if not old_path.exists():
        old_path = minute_snapshot("base", START, "2026-11-11")
    newer_path = minute_snapshot("extension", "2026-11-10", END_EXCLUSIVE)
    old, newer = pd.read_parquet(old_path).loc[START:"2026-11-10"], pd.read_parquet(newer_path).loc[:END]
    common = old.index.intersection(newer.index)
    assert len(common) >= 390 and str(common[0].date()) == "2026-11-10"
    maximum = float(np.max(np.abs(old.loc[common, FIELDS] / newer.loc[common, FIELDS] - 1)))
    assert np.allclose(old.loc[common, FIELDS], newer.loc[common, FIELDS], rtol=1e-10, atol=1e-10)
    minute = pd.concat([old, newer.loc["2026-11-11":]]).sort_index()
    base.validate_ohlc(minute, "SOXL minute")
    daily_out, minute_out = CACHE / "daily.parquet", CACHE / "minute.parquet"
    minute.to_parquet(minute_out)
    daily.loc[START:END].to_parquet(daily_out)
    old_daily = HERE.parent / "cache/daily.parquet"
    daily_diff = None
    if old_daily.exists():
        prior = pd.read_parquet(old_daily)
        old_cols = ["Open", "High", "Low", "Close"]
        daily_diff = float(np.abs(daily.loc[prior.index, old_cols] / prior[old_cols] - 1).to_numpy().max())
        assert daily_diff < 1e-10, "frozen daily overlap disagrees"
    audit = {"as_of_last_complete_us_session": END, "time_zone": TZ, "feed": "sip", "adjustment": "split",
             "retrieved_utc": retrieved, "alpaca_end_exclusive": END_EXCLUSIVE,
             "source_docs": base.DOCS, "minute_sources": list(map(source, (old_path, newer_path, warm_path))),
             "daily_sources": [source(p) for p in daily_paths.values()],
             "overlap_bars": len(common), "overlap_all_fields_max_relative_error": maximum,
             "previous_daily_ohlc_max_relative_error": daily_diff,
             "daily_output": source(daily_out), "minute_output": source(minute_out),
             "minute_rows": len(minute), "minute_first": minute.index[0].isoformat(),
             "minute_last": minute.index[-1].isoformat(), "sessions": len(daily.loc[START:END]),
             "warmup_only": [WARM, "2025-12-31"]}
    return warm, minute, daily, sector, schedule, audit


def frozen_csv(path, oid, **kwargs):
    """Use the archived Git blob even when the live dashboard has moved on."""
    payload = path.read_bytes().replace(b"\r\n", b"\n") if path.exists() else b""
    def blob_id(data):
        return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if blob_id(payload) != oid:
        payload = subprocess.check_output(["git", "cat-file", "blob", oid], cwd=HERE)
    assert blob_id(payload) == oid
    meta = {"path": path.relative_to(ANALYSIS).as_posix(), "git_blob": oid,
            "sha256_lf": hashlib.sha256(payload).hexdigest()}
    return pd.read_csv(io.BytesIO(payload), **kwargs), meta


def late_flow_availability():
    """Audit later official responses separately; none is an as-of entry signal."""
    prior_path = OUT / "input_audit.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        fixed = prior.get("bb1", {}).get("later_official_statement_checks")
        if fixed is not None:
            return fixed
    collector = load_module("registered_official_collector", ANALYSIS / "bottom_zone_20261111/code/collect_A.py")
    collector.ROOT = CACHE / "seibro_check"
    checks = []
    for date in ("2026-11-12", "2026-11-13", "2026-11-16"):
        records = [collector.fetch((pd.Timestamp(date), "daily", dtype)) for dtype in "1234"]
        xmls = [collector.ROOT / "data/seibro/2026" / f"daily_{k}_{date.replace('-', '')}.xml.gz" for k in "1234"]
        checks.append({"settlement_statement_day": date, "rank_lists": ["1", "2", "3", "4"],
                       "rows": [r["n"] for r in records], "retrieval_times_utc": [r["fetched_at"] for r in records],
                       "raw_xml_gzip_sha256": [base.digest(path) for path in xmls]})
    return {"source": collector.c.PAGE, "after_the_fact_only": True, "checks": checks,
            "interpretation": "These dated official responses are available at the recorded retrieval times. Their content contains no independently verified first-publication or revision timestamp on the decision date. They do not fill the primary flags."}


def belt_flags(sector):
    belt = ANALYSIS / "bottom_zone_20261111"
    atoms_path = belt / "belt/results/daily_belt.csv"
    panel_path = belt / "data/signal_panel.csv"
    update_path = belt / "belt/now/now_table.csv"
    # Read just the two registered atoms. The same file also contains future outcome labels.
    atoms, atoms_source = frozen_csv(atoms_path, "02a7072bff0705a2d717d84b845a6989b07dc87a",
                                     usecols=["date", "낙폭20", "SOXL비중상위20"], index_col="date")
    atoms = atoms.loc[START:"2026-11-11"]
    source_cols = ["date", "dd60", "source_date", "source_known", "source_published", "T5_P"]
    panel, panel_source = frozen_csv(panel_path, "782af73ab0c49b9d6793b748204287523664bf81",
                                     usecols=source_cols, index_col="date")
    panel = panel.loc[atoms.index]
    known = panel.source_published.eq(True) & panel.T5_P.notna() & panel.dd60.notna()
    assert panel.loc[known, "source_known"].le(panel.index[known]).all()
    assert panel.loc[known, "source_date"].lt(panel.index[known]).all()
    assert atoms.loc[known, "낙폭20"].equals(panel.loc[known, "dd60"].le(-.20))
    assert atoms.loc[known, "SOXL비중상위20"].equals(panel.loc[known, "T5_P"].ge(80))
    series = (atoms["낙폭20"] & atoms["SOXL비중상위20"]).astype("Int64").where(known)
    extension, ext_source = frozen_csv(update_path, "42eddc3c1a920f1695dc94ae9e4a002327d6844a",
                                      usecols=["관찰일", "수급원거래일", "수급공개", "낙폭60", "SOXL비중백분위", "BB1"], index_col="관찰일")
    extension = extension.loc[: "2026-11-12"]
    ext_known = extension["수급공개"].eq(True) & extension["수급원거래일"].lt(extension.index)
    extra = extension.BB1.astype("Int64").where(ext_known)
    overlap = series.index.intersection(extra.index)
    assert series.loc[overlap].tolist() == extra.loc[overlap].tolist()
    assert "2026-11-12" in extension.index
    diagnostics = panel[["dd60", "T5_P", "source_date", "source_known"]].copy()
    diagnostics.columns = ["bb1_prior_soxx_drawdown60", "bb1_flow_percentile", "bb1_source_trade_day", "bb1_flow_known_day"]
    diagnostics["bb1_availability"] = np.where(known, "frozen_historical_panel", "no_new_source")
    diagnostics.loc["2026-11-12"] = [extension.loc["2026-11-12", "낙폭60"], extension.loc["2026-11-12", "SOXL비중백분위"],
                                     extension.loc["2026-11-12", "수급원거래일"], "2026-11-12", "registered_snapshot"]
    dd = (sector.Close / sector.Close.rolling(60).max() - 1).shift()
    # The calendar assigns the joint 10/11 Nov US trade settlement statement
    # to the 16th and the 12 Nov US trade to the 17th (panel_A.raw_panel).
    for day, trade in [("2026-11-13", None), ("2026-11-16", "2026-11-11"), ("2026-11-17", "2026-11-12")]:
        status = "no_new_source" if trade is None else "no_contemporaneous_snapshot"
        diagnostics.loc[day] = [float(dd.loc[day]), np.nan, trade, day if trade else None, status]
    provenance = {"sources": [atoms_source, panel_source, ext_source],
                  "latest_source_observation": "2026-11-12",
                  "update_published_et": "2026-11-12T00:48:00-05:00",
                  "overlap_verified": len(overlap),
                  "later_official_statement_checks": late_flow_availability(),
                  "late_missing_days": diagnostics.loc["2026-11-13":, ["bb1_prior_soxx_drawdown60", "bb1_availability"]].to_dict("index"),
                  "late_missing_drawdown_source": "SIP SOXX last completed daily close relative to highest close in its trailing 60 sessions. The three November 13/16/17 price legs all pass -20%. November 13 has no new source under the registered KSD calendar. The Nov 16 and Nov 17 flow dates reflect the registered publication-lag convention; they lack a contemporaneous observation.",
                  "logic": "dd60 <= -0.20 and published T5_P >= 80 on the registered observation date. No flow carried forward, no outcome labels read. Missing flow or snapshot -> unknown; skipped."}
    return pd.concat([series, extra.loc[extra.index > series.index.max()]]), diagnostics, provenance


def compute(warm, minute, daily, sector, schedule, bb1, minutes_before_close=10):
    all_minutes = pd.concat([warm, minute]).sort_index()
    sessions = []
    for day, row in schedule.iterrows():
        op, cl = (row[k].tz_convert(TZ) for k in ("open", "close"))
        expected = pd.date_range(op, cl, inclusive="left", freq="min")
        reg = all_minutes.loc[op:cl-MINUTE]
        assert reg.index.equals(expected), f"regular minute gap: {day}"
        assert reg.Volume.gt(0).all() and reg.VWAP.gt(0).all(), f"bad VWAP inputs: {day}"
        sessions.append(reg)
    regular = pd.concat(sessions)
    # Matches the earlier study: RTH bars only, carried across sessions with the
    # overnight price change included. Time index denotes completion of each bar.
    rr = pd.Series(htf.rma_state(regular.Close.to_numpy(), 14)[2], index=regular.index + MINUTE)
    rising = rr.le(30) & rr.gt(rr.shift(1))
    previous_soxl = daily.Close.shift()
    previous_soxx = sector.Close.shift()
    previous_average = sector.Close.rolling(20, min_periods=20).mean().shift()
    rows = []
    for day, row in schedule.loc[START:END].iterrows():
        label = str(day.date())
        op, cl = (row[k].tz_convert(TZ) for k in ("open", "close"))
        decision = cl - minutes_before_close * MINUTE
        before = regular.loc[op:decision-MINUTE]
        close = float(before.Close.iloc[-1])
        last30 = float(before.at[decision - 31 * MINUTE, "Close"])
        vwap = float((before.VWAP * before.Volume).sum() / before.Volume.sum())
        prev = float(previous_soxl.at[op.normalize()])
        sector_close = float(previous_soxx.at[op.normalize()])
        sector_ma = float(previous_average.at[op.normalize()])
        assert np.isfinite([close, last30, vwap, prev, sector_close, sector_ma]).all()
        # For an early close the absolute 15:30 observation window is empty.
        window = rising.loc[op.normalize()+pd.Timedelta(hours=15, minutes=30):decision]
        matches = window.index[window]
        rows.append(dict(day=label, sector_sma20=int(sector_close >= sector_ma),
                         above_vwap=int(close >= vwap), late_up=int(close > last30),
                         rsi_turn=int(bool(len(matches))), down5=int(close <= .95 * prev),
                         bb1=bb1.get(label, pd.NA), decision_et=decision.isoformat(),
                         latest_bar_start_et=before.index[-1].isoformat(),
                         price_before_cutoff=close, regular_vwap_before_cutoff=vwap,
                         price_30min_earlier=last30, previous_soxl_close=prev,
                         previous_soxx_close=sector_close, previous_soxx_sma20=sector_ma,
                         first_rsi_turn_et=matches[0].isoformat() if len(matches) else None))
    result = pd.DataFrame(rows)
    result["bb1"] = result.bb1.astype("Int64")
    return result, len(regular)


def main():
    warm, minute, daily, sector, schedule, audit = snapshot_prices()
    # The last minute close and the daily close coincide in this frozen warm-up.
    warm_close = warm.loc["2025-12-31"].between_time("15:59", "15:59").Close.iloc[0]
    assert np.isclose(warm_close, daily.loc["2025-12-31", "Close"], rtol=1e-4)
    bb1, diagnostics, belt_audit = belt_flags(sector)
    features, n_regular = compute(warm, minute, daily, sector, schedule, bb1)
    features = features.join(diagnostics, on="day", validate="1:1")
    assert len(features) == 221 and features.day.is_unique
    earlier, _ = compute(warm, minute, daily, sector, schedule, bb1, minutes_before_close=15)
    intraday_flags = ["above_vwap", "late_up", "rsi_turn", "down5"]
    earlier_target = OUT / "earlier_cutoff_features.csv"
    earlier[["day", *intraday_flags]].to_csv(earlier_target, index=False)
    prior_path = ANALYSIS / "soxl_close_rsi/results/rebound_signals.csv"
    old = pd.read_csv(prior_path, usecols=["day", "tf", "threshold", "rule", "decision", "decision_minute"])
    matched = old[(old.tf == "1m") & (old.threshold == 30) & (old.rule == "rising") & (old.decision_minute <= 950)]
    overlap = features[features.day <= "2026-11-03"]
    assert overlap.loc[overlap.rsi_turn == 1, "first_rsi_turn_et"].tolist() == matched.sort_values("day").decision.tolist()
    target = OUT / "features.csv"
    features.to_csv(target, index=False)
    audit.update({"pass": True, "feature_output": source(target), "regular_minutes_including_warmup": n_regular,
                  "earlier_cutoff_output": source(earlier_target),
                  "earlier_cutoff": {"decision_minutes_before_close": 15, "latest_regular_bar_normal_day": "15:44",
                                     "selection_use": False, "signals": {k: int(earlier[k].sum()) for k in intraday_flags}},
                  "rsi_prior_study_overlap": {"source": source(prior_path), "through": "2026-11-03",
                                              "sessions": len(overlap), "matched_first_signal_timestamps": len(matched)},
                  "asof": {"sector_sma20": "SOXX through previous regular day, 20 inclusive closes",
                           "above_vwap": "RTH from open through cutoff - 1 minute (last completed bar)",
                           "late_up": "SOXL 15:49 minute close > 15:19 minute close (normal close)",
                           "rsi_turn": "RTH-only continuous Wilder RSI14, any completed bar from 15:30 through 15:50 inclusive whose RSI <=30 and > prior bar",
                           "down5": "last completed SOXL minute close <= 95% of preceding regular daily close",
                           "bb1": "registered cutoff-safe BB-1 only; separate post-hoc comparison"},
                  "bb1": belt_audit,
                  "bb1_availability_counts": features.bb1_availability.value_counts().to_dict(),
                  "known_rows": {name: int(features[name].notna().sum()) for name in FLAGS},
                  "signal_rows": {name: int(features[name].sum()) for name in FLAGS},
                  "unknown_bb1_dates": features.loc[features.bb1.isna(), "day"].tolist()})
    (OUT / "input_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("pass", "sessions", "signal_rows", "known_rows", "rsi_prior_study_overlap")}, indent=2))


if __name__ == "__main__":
    main()
