"""Independent ledger audit artifact, never an execution authority.

Reconstructs SOXS money, calendar, ranks, latest-source choices and close reactions
without importing any project build, current, ledger, feature or strategy module.
Run after source_daily.csv and entry_daily.csv exist in this directory.
"""
from __future__ import annotations

import bisect
import datetime as dt
import hashlib
import json
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
CURRENT = BASE / "seibro_bottoms/results/soxs_current_20261105"
KEYS = [f"{m}_L{h}" for m in ("gross_usd", "net_usd", "buy_share") for h in (1, 5)]
AMOUNTS = ("buy", "sell", "net", "tot")
START = pd.Timestamp("2024-01-02")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calendars():
    nyse = pd.DatetimeIndex(xc.get_calendar("XNYS", start="2022-06-01", end="2027-03-01").sessions).tz_localize(None)
    korea = pd.DatetimeIndex(xc.get_calendar("XKRX", start="2022-12-01", end="2027-03-01").sessions).tz_localize(None)
    korea = korea[korea != "2026-06-03"]
    closed = set()
    for year in range(2022, 2028):
        october = [dt.date(year, 10, d) for d in range(1, 32) if dt.date(year, 10, d).weekday() == 0]
        closed.add(october[1])
        veterans = dt.date(year, 11, 11)
        if veterans.weekday() == 6:
            veterans += dt.timedelta(days=1)
        if veterans.weekday() < 5:
            closed.add(veterans)
    settle = sorted((set(nyse.date) | {dt.date(2025, 1, 9)}) - closed)

    def seibro(source):
        day = pd.Timestamp(source).date()
        delay = 1 if day < dt.date(2024, 5, 28) else 0
        settlement = settle[bisect.bisect_right(settle, day) + delay]
        result = settlement + dt.timedelta(days=1)
        while result.weekday() > 4:
            result += dt.timedelta(days=1)
        return pd.Timestamp(result)

    def timing(source):
        report = seibro(source)
        known = korea[korea.searchsorted(report, side="right")]
        entry = nyse[nyse.searchsorted(known, side="left")]
        return report, known, entry

    return nyse, timing


def raw_panel(timing):
    snapshot = json.loads((CURRENT / "raw_snapshot.json").read_text(encoding="utf-8"))
    fresh = [r for r in snapshot["requests"] if r["kind"] == "daily"]
    published_end = max(pd.Timestamp(r["start"]) for r in fresh if len(r["rows"]))
    records, manifest = {}, []
    for folder in sorted((BASE / "seibro_flows/raw/daily").glob("d[1234]")):
        for path in sorted(folder.glob("*.json")):
            rec = json.loads(path.read_text(encoding="utf-8"))
            if pd.Timestamp(rec["start"]) > published_end:
                continue
            assert rec["start"] == rec["end"] and len(rec["rows"]) == rec["n"]
            records[(pd.Timestamp(rec["start"]), str(rec["dtype"]))] = rec
            manifest.append((str(path.relative_to(BASE)), digest(path)))
    for rec in fresh:
        if pd.Timestamp(rec["start"]) <= published_end:
            records[(pd.Timestamp(rec["start"]), str(rec["dtype"]))] = rec
    classes = pd.read_csv(BASE / "seibro_flows/classes.csv")
    isins = set(classes.loc[classes["product"].eq("SOXS"), "isin"])
    dates = sorted({key[0] for key in records})
    amounts, all_rows, deduplicated = {}, 0, 0
    for day in dates:
        union = {}
        for dtype in "1234":
            assert (day, dtype) in records, (day, dtype)
            for row in records[(day, dtype)]["rows"]:
                if row["isin"] not in isins:
                    continue
                value = np.array([row[c] for c in AMOUNTS], dtype=float)
                assert row["isin"] not in union or np.array_equal(union[row["isin"]], value), (day, row["isin"])
                union[row["isin"]] = value
                all_rows += 1
        amounts[day] = (np.sum(list(union.values()), axis=0) if union else np.zeros(4), bool(union))
        deduplicated += len(union)
    nyse, _ = calendars()
    assigned = {}
    for source in nyse[(nyse >= "2022-12-26") & (nyse <= published_end)]:
        report, known, entry = timing(source)
        if report in amounts:
            assigned.setdefault(report, []).append((source, known, entry))
    rows = []
    for uid, report in enumerate(dates):
        trades = assigned.get(report, [])
        if not trades:
            continue
        raw_amount, observed = amounts[report]
        for source, known, entry in trades:
            rows.append({"source_date": source, "seibro_date": report, "known": known,
                         "entry_settle": entry, "unit_id": uid, "unit_n_trades": len(trades),
                         "cens_SOXS": not observed, "part_SOXS": False,
                         **{f"SOXS_{c}": float(v / len(trades)) for c, v in zip(AMOUNTS, raw_amount)}})
    result = pd.DataFrame(rows).set_index("source_date").sort_index()
    info = {"published_seibro_end": str(published_end.date()), "files": len(manifest),
            "file_manifest_sha256": hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
            "raw_calendar_days": len(dates), "soxs_cross_list_rows": all_rows,
            "soxs_unique_date_isins": deduplicated, "cross_list_amount_mismatches": 0,
            "source_rows": len(result), "shared_unit_source_rows": int(result.unit_n_trades.gt(1).sum()),
            "unpublished_snapshot_dates": sorted({r["start"] for r in fresh if not len(r["rows"])})}
    return result, info


def features(full):
    x = full.loc["2023-01-03":]
    valid = (~x.cens_SOXS.fillna(True).astype(bool) & ~x.part_SOXS.fillna(True).astype(bool)
             & np.isfinite(x[["SOXS_buy", "SOXS_sell", "SOXS_net"]]).all(axis=1))
    values = x[["SOXS_buy", "SOXS_sell", "SOXS_net"]].where(valid, axis=0).to_numpy(float)
    out = pd.DataFrame(index=x.index)
    for h in (1, 5):
        sums = np.full_like(values, np.nan)
        for i in range(h - 1, len(values)):
            past = values[i - h + 1:i + 1]
            if np.isfinite(past).all():
                sums[i] = past.sum(axis=0)
        share = np.divide(sums[:, 0], sums[:, 0] + sums[:, 1], out=np.full(len(x), np.nan),
                          where=(sums[:, 0] + sums[:, 1]) > 0)
        for measure, array in (("gross_usd", sums[:, 0]), ("net_usd", sums[:, 2]), ("buy_share", share)):
            key = f"{measure}_L{h}"
            p, n = np.full(len(x), np.nan), np.zeros(len(x))
            for i, value in enumerate(array):
                w = array[max(0, i - 249):i + 1]
                w = w[np.isfinite(w)]
                if np.isfinite(value) and len(w) >= 120:
                    n[i] = len(w)
                    p[i] = 100 * ((w < value).sum() + .5 * (w == value).sum()) / len(w)
            out[key], out[key + "_P"], out[key + "_rank_n"] = array, p, n
    return out


def source_expected(published, nyse, timing, price_end):
    grid = nyse[(nyse >= "2023-01-03") & (nyse <= price_end)]
    x = published.reindex(grid).copy()
    x["source_observed"] = x.index.isin(published.index)
    x = x.join(features(x))
    for source in x.index[~x.source_observed]:
        report, known, entry = timing(source)
        x.loc[source, ["seibro_date", "known", "entry_settle"]] = [report, known, entry]
    latest = published.reset_index().sort_values(["entry_settle", "source_date"]).drop_duplicates("entry_settle", keep="last")
    latest = latest.set_index("entry_settle").source_date
    x["selected_for_entry"] = [observed and source == latest.get(entry)
                                 for source, observed, entry in zip(x.index, x.source_observed, x.entry_settle)]
    for key in KEYS:
        p = x[key + "_P"]
        x[key + "_status"] = np.where(~x.source_observed, "UNPUBLISHED",
                                       np.where(p.isna(), "UNKNOWN", np.where(p >= 80, "ON", "OFF")))
        x[key + "_on"] = x[key + "_status"].eq("ON").where(p.notna())
        x.loc[~x.source_observed, key + "_rank_n"] = np.nan
    x.loc[x.cens_SOXS.eq(True) | x.part_SOXS.eq(True), ["SOXS_" + c for c in AMOUNTS]] = np.nan
    x["selected_source_for_same_entry"] = x.entry_settle.map(latest).where(x.source_observed)
    x["unit_shared_source"] = x.unit_n_trades.gt(1)
    x["timing_basis"] = np.where(x.source_observed, "canonical_published_unit", "calendar_expected_unpublished")
    x["selection_reason"] = np.where(~x.source_observed, "unpublished_source_no_flow_values",
                                     np.where(x.selected_for_entry, "latest_source_selected_by_research_model",
                                              "shadowed_by_later_source_same_entry_research_model"))
    return x, latest


def price_reactions(prices, nyse, entries):
    p = prices.reindex(nyse[(nyse >= prices.index.min()) & (nyse <= prices.index.max())]).copy()
    p["dd60_pct"] = (p.close / p.close.rolling(60).max() - 1) * 100
    rows = []
    for entry in entries:
        before = p.index[p.index < entry]
        previous = p.loc[before[-1]]
        row = {"entry": entry, "previous_close": float(previous.close),
               "previous_dd60_pct": float(previous.dd60_pct), "state": bool(previous.dd60_pct <= -10)}
        i = nyse.get_loc(entry)
        op = p.open.get(entry, np.nan)
        for h in (5, 10, 20):
            window = p.close.reindex(nyse[i:i + h])
            complete = len(window) == h and np.isfinite(window).all() and window.gt(0).all() and np.isfinite(op) and op > 0
            row.update({f"R{h}_complete": bool(complete), f"R{h}_pct": np.nan,
                        f"MAE{h}_pct": np.nan, f"MFE{h}_pct": np.nan})
            if complete:
                row.update({f"R{h}_pct": float((window.iloc[-1] / op - 1) * 100),
                            f"MAE{h}_pct": float(min(0, (window.min() / op - 1) * 100)),
                            f"MFE{h}_pct": float(max(0, (window.max() / op - 1) * 100))})
        rows.append(row)
    return pd.DataFrame(rows).set_index("entry")


def entry_expected(source, latest, reactions):
    x = reactions.copy()
    x["source_date"] = latest.reindex(x.index)
    for col in ["known", "cens_SOXS", "part_SOXS"] + [k + suffix for k in KEYS for suffix in ("", "_P", "_rank_n", "_status", "_on")]:
        x[col] = x.source_date.map(source[col])
    for key in KEYS:
        x[key + "_status"] = x[key + "_status"].fillna("NO_NEW_SOURCE")
        status = x[key + "_status"]
        valid = status.isin(["ON", "OFF"])
        x[key + "_rank_n"] = x[key + "_rank_n"].fillna(0)
        x[key + "_on"] = status.eq("ON").where(valid)
        x[key + "_dd10_on"] = status.eq("ON") & x.state
        x[key + "_dd10_eligible"] = valid & x.state
        x[key + "_dd10_status"] = np.where(x.state, status, "OUTSIDE_DD10")
        previous, run, streak, transitions, runs, streaks = None, 0, 0, [], [], []
        for current in status:
            transitions.append("START" if previous is None else "UNCHANGED" if current == previous else previous + "->" + current)
            run = run + 1 if current == previous else 1
            streak = streak + 1 if current == "ON" else 0
            runs.append(run)
            streaks.append(streak)
            previous = current
        x[key + "_transition"], x[key + "_status_run_sessions"], x[key + "_on_streak_sessions"] = transitions, runs, streaks
        for h in (5, 10, 20):
            chosen, last = [], -1000
            for i in np.flatnonzero(status.eq("ON")):
                if i - last >= h:
                    chosen.append(i)
                    last = i
            event = np.zeros(len(x), dtype=bool)
            event[chosen] = True
            x[key + f"_event{h}"] = event
    return x


def monthly_expected(source, entry):
    rows = []
    for month in sorted(set(source.index.to_period("M")) | set(entry.index.to_period("M"))):
        s = source[source.index.to_period("M") == month]
        e = entry[entry.index.to_period("M") == month]
        for key in KEYS:
            ss, es = s[key + "_status"], e[key + "_status"]
            row = {"month": str(month), "key": key, "source_sessions": len(s), "entry_sessions": len(e)}
            for status in ("ON", "OFF", "UNKNOWN", "UNPUBLISHED"):
                row["source_" + status.lower()] = int(ss.eq(status).sum())
            row["source_shadowed_on"] = int((ss.eq("ON") & ~s.selected_for_entry).sum())
            for status, label in (("ON", "latest_on"), ("OFF", "latest_off"),
                                  ("UNKNOWN", "unknown"), ("NO_NEW_SOURCE", "no_new_source")):
                row["entry_" + label] = int(es.eq(status).sum())
            row["dd10_on"] = int(e[key + "_dd10_on"].sum())
            row["entry_on_price_pending"] = int((es.eq("ON") & ~e.price_session_complete).sum())
            rows.append(row)
    return pd.DataFrame(rows).set_index(["month", "key"]).sort_index()


def compare(expected, actual, columns, label, results):
    assert expected.index.equals(actual.index), (label, "date coverage/index mismatch")
    checks, mismatches = 0, []
    for col in columns:
        assert col in actual and col in expected, (label, col)
        a, b = expected[col], actual[col]
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
            same = np.isclose(pd.to_numeric(a), pd.to_numeric(b), equal_nan=True, atol=1e-8, rtol=1e-10)
        elif pd.api.types.is_datetime64_any_dtype(a):
            a, b = pd.to_datetime(a), pd.to_datetime(b)
            same = a.eq(b) | (a.isna() & b.isna())
        else:
            same = a.fillna("<NA>").astype(str).eq(b.fillna("<NA>").astype(str))
        checks += len(a)
        bad = np.flatnonzero(~np.asarray(same))
        for i in bad[:10]:
            mismatches.append({"column": col, "date": str(a.index[i]),
                               "expected": str(a.iloc[i]), "actual": str(b.iloc[i])})
    results[label] = {"rows": len(expected), "columns": len(columns), "comparisons": checks,
                      "mismatch_examples": mismatches, "passed": not mismatches}


def main():
    frozen_path = BASE / "seibro_bottoms/derived/measures_daily.parquet"
    price_path = CURRENT / "market.prices.csv"
    input_paths = [frozen_path, CURRENT / "raw_snapshot.json", price_path,
                   BASE / "seibro_flows/classes.csv", HERE / "source_daily.csv", HERE / "entry_daily.csv",
                   HERE / "monthly.csv", HERE / "warmup_daily.csv", HERE / "coverage.json"]
    hashes = {str(p.relative_to(BASE)): digest(p) for p in input_paths}
    nyse, timing = calendars()
    published, raw_info = raw_panel(timing)
    prices = pd.read_csv(price_path, parse_dates=["date"]).set_index("date").sort_index()
    prices = prices.loc[:"2026-11-04"]
    assert prices.index.max() == pd.Timestamp("2026-11-04")
    frozen = pd.read_parquet(frozen_path).set_index("date").sort_index()
    checks = {}
    columns = ["SOXS_" + c for c in AMOUNTS] + ["cens_SOXS", "part_SOXS", "seibro_date", "entry_settle", "unit_id", "unit_n_trades"]
    compare(published.reindex(frozen.index), frozen, columns, "raw_to_all_frozen_source_rows", checks)
    source, latest = source_expected(published, nyse, timing, prices.index.max())
    px = prices.copy()
    px["dd60_pct"] = (px.close / px.close.rolling(60).max() - 1) * 100
    source["SOXX_Open"], source["SOXX_Close"] = px.open.reindex(source.index), px.close.reindex(source.index)
    source["SOXX_daily_return_pct"] = px.close.pct_change(fill_method=None).reindex(source.index) * 100
    source["dd60_pct"] = px.dd60_pct.reindex(source.index)
    prior_dates = pd.DatetimeIndex([nyse[nyse.get_loc(e) - 1] for e in source.entry_settle])
    source["entry_previous_close"] = px.close.reindex(prior_dates).to_numpy()
    source["entry_previous_dd60_pct"] = px.dd60_pct.reindex(prior_dates).to_numpy()
    source["entry_previous_price_date"] = prior_dates.where(prior_dates.isin(px.index))
    source["source_to_entry_sessions"] = [nyse.get_loc(e) - nyse.get_loc(s) for s, e in zip(source.index, source.entry_settle)]
    source["source_close_to_entry_open_pct"] = (px.open.reindex(source.entry_settle).to_numpy() / source.SOXX_Close - 1) * 100
    end_entry = nyse[nyse.searchsorted(prices.index.max(), side="right")]
    entries = nyse[(nyse >= START) & (nyse <= end_entry)]
    entry = entry_expected(source, latest, price_reactions(prices, nyse, entries))
    entry["new_source_available"] = entry.source_date.notna()
    entry["price_session_complete"] = entry.index.isin(prices.index)
    entry["evaluated_entry"] = entry.price_session_complete
    entry["pending_entry"] = ~entry.price_session_complete
    actual_source = pd.read_csv(HERE / "source_daily.csv", parse_dates=["source_date", "entry_settle", "seibro_date"]).set_index("source_date").sort_index()
    actual_entry = pd.read_csv(HERE / "entry_daily.csv", parse_dates=["entry", "source_date"]).set_index("entry").sort_index()
    expected_source = source.loc[START:].copy()
    source_columns = columns + ["source_observed", "selected_for_entry", "selected_source_for_same_entry", "unit_shared_source",
        "timing_basis", "SOXX_Open", "SOXX_Close", "SOXX_daily_return_pct", "dd60_pct", "source_to_entry_sessions",
        "entry_previous_close", "entry_previous_dd60_pct", "source_close_to_entry_open_pct"]
    if "entry_previous_price_date" in actual_source:
        source_columns.append("entry_previous_price_date")
    source_columns += [k + suffix for k in KEYS for suffix in ("", "_P", "_rank_n", "_status", "_on")]
    compare(expected_source, actual_source, source_columns, "every_source_day", checks)
    entry_columns = ["source_date", "previous_close", "previous_dd60_pct", "state", "cens_SOXS", "part_SOXS"]
    entry_columns += [f"{prefix}{h}{suffix}" for h in (5, 10, 20) for prefix, suffix in (("R", "_pct"), ("R", "_complete"), ("MAE", "_pct"), ("MFE", "_pct"))]
    entry_columns += ["new_source_available", "price_session_complete", "evaluated_entry", "pending_entry"]
    entry_columns += [k + suffix for k in KEYS for suffix in ("", "_P", "_rank_n", "_status", "_on", "_dd10_on",
        "_dd10_eligible", "_dd10_status", "_event5", "_event10", "_event20", "_transition", "_status_run_sessions", "_on_streak_sessions")]
    compare(entry, actual_entry, entry_columns, "every_entry_day", checks)
    monthly = monthly_expected(expected_source, entry)
    actual_monthly = pd.read_csv(HERE / "monthly.csv").set_index(["month", "key"]).sort_index()
    compare(monthly, actual_monthly, list(monthly.columns), "every_month_all_six_definitions", checks)
    warmup = pd.read_csv(HERE / "warmup_daily.csv", parse_dates=["source_date", "entry_settle", "seibro_date"]).set_index("source_date").sort_index()
    compare(source.loc["2023-01-03":"2023-12-29"], warmup, source_columns, "all_2023_warmup_rows", checks)
    coverage = json.loads((HERE / "coverage.json").read_text(encoding="utf-8"))
    coverage_expected = {"source_rows": len(expected_source), "entry_rows": len(entry), "warmup_rows": len(warmup),
        "source_first": expected_source.index.min().isoformat(), "source_last": expected_source.index.max().isoformat(),
        "entry_first": entry.index.min().isoformat(), "entry_last": entry.index.max().isoformat(),
        "latest_completed_price_session": prices.index.max().isoformat(),
        "latest_published_source": published.index.max().isoformat(),
        "interior_missing_source_dates": [d.isoformat() for d in expected_source.index[(expected_source.index <= published.index.max()) & ~expected_source.source_observed]],
        "unpublished_source_dates": [d.isoformat() for d in expected_source.index[~expected_source.source_observed]],
        "entry_no_new_source_dates": [d.isoformat() for d in entry.index[entry.source_date.isna()]],
        "censored_source_dates": [d.isoformat() for d in expected_source.index[expected_source.cens_SOXS.eq(True)]],
        "partial_source_dates": [d.isoformat() for d in expected_source.index[expected_source.part_SOXS.eq(True)]],
        "shadowed_source_rows": int((expected_source.source_observed & ~expected_source.selected_for_entry).sum()),
        "source_year_counts": {str(y): int(n) for y, n in expected_source.groupby(expected_source.index.year).size().items()},
        "entry_year_counts": {str(y): int(n) for y, n in entry.groupby(entry.index.year).size().items()},
        "frozen_overlap_rows": len(frozen), "new_signal_definitions": 0, "thresholds_tested": [80]}
    bad_coverage = [key for key, value in coverage_expected.items() if coverage.get(key) != value]
    checks["coverage_metadata"] = {"fields": len(coverage_expected), "mismatch_fields": bad_coverage, "passed": not bad_coverage}
    known_source = pd.to_datetime(actual_source.known_settle_kst, utc=True)
    known_entry = pd.to_datetime(actual_entry.known_settle_kst, utc=True)
    expected_known_source = expected_source.known.dt.tz_localize("Asia/Seoul") + pd.Timedelta(hours=9)
    expected_known_entry = entry.known.dt.tz_localize("Asia/Seoul") + pd.Timedelta(hours=9)
    checks["publication_timestamps"] = {"source_mismatches": int((known_source != expected_known_source.dt.tz_convert("UTC")).sum()),
                                         "entry_mismatches": int((~(known_entry.eq(expected_known_entry.dt.tz_convert("UTC")) | (known_entry.isna() & expected_known_entry.isna()))).sum())}
    days = [pd.Timestamp(x) for x in ("2026-09-21", "2026-09-22", "2026-09-23")]
    holiday = [{"source": str(d.date()), "P": float(source.at[d, "gross_usd_L1_P"]),
                "status": source.at[d, "gross_usd_L1_status"], "entry": str(source.at[d, "entry_settle"].date()),
                "selected": bool(source.at[d, "selected_for_entry"])} for d in days]
    findings = {"harvest_holiday_same_entry": holiday,
                "chosen_2026_09_28_source": str(entry.at[pd.Timestamp("2026-09-28"), "source_date"].date()),
                "chosen_2026_09_28_status": entry.at[pd.Timestamp("2026-09-28"), "gross_usd_L1_status"],
                "gross1_by_year": [], "recent_ON_entries": []}
    for year, y in entry.groupby(entry.index.year):
        on = y[y.gross_usd_L1_status.eq("ON")]
        complete = on[on.R10_complete]
        off = y[y.gross_usd_L1_status.eq("OFF") & y.R10_complete]
        findings["gross1_by_year"].append({"year": int(year), "ON_days": len(on), "R10_complete_ON": len(complete),
            "R10_positive_ON": int(complete.R10_pct.gt(0).sum()), "R10_nonpositive_ON": int(complete.R10_pct.le(0).sum()),
            "R10_mean_ON": float(complete.R10_pct.mean()), "R10_mean_OFF": float(off.R10_pct.mean()),
            "unscored_ON": int((~on.R10_complete).sum())})
    for d, row in entry[(entry.index >= "2026-09-01") & entry.gross_usd_L1_status.eq("ON")].iterrows():
        findings["recent_ON_entries"].append({"entry": str(d.date()), "source": str(row.source_date.date()),
            "P": float(row.gross_usd_L1_P), "dd10": bool(row.state), "R10_complete": bool(row.R10_complete),
            "R10_pct": float(row.R10_pct) if row.R10_complete else None})
    checks["publication_timestamps"]["passed"] = not (checks["publication_timestamps"]["source_mismatches"] or checks["publication_timestamps"]["entry_mismatches"])
    passed = all(v["passed"] for v in checks.values())
    output = {"scope": "full independent raw/calendar/rank/latest-source/price-window audit; no production module imported",
              "code_sha256": digest(Path(__file__)), "input_sha256": hashes, "raw": raw_info,
              "causality": "2023 warmup; trailing at most250 NYSE source observations including current only; strict uncensored full5-session windows; unpublished values stay missing",
              "missing_semantics": "Top50-censored dollar placeholders are displayed as NA; UNKNOWN/UNPUBLISHED/NO_NEW_SOURCE ON flags are NA, not False; rank_n=0 records no usable rank for a published/entry row, never zero money or a zero outcome.",
              "calendar_limit": "The fixed cal model has only same-SEIBRO-date pred2 groups; audit reconstructs these without centered future price/volume decisions.",
              "checks": checks, "findings": findings, "passed": passed}
    assert hashes == {str(p.relative_to(BASE)): digest(p) for p in input_paths}, "inputs changed during audit"
    (HERE / "independent_audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "checks": checks, "findings": findings}, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
