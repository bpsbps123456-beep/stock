"""SOXL frozen close-RSI signal-day SIP quote audit; no strategy reselection.

Reuse the existing strict last-NBBO validation and same-day raw/adjusted audit.
Run: py .../code/quotes.py [--offline]. Raw quote cache is Git-ignored.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT.parent
OUT = ROOT / "results"
CACHE = ROOT / "cache/quotes.parquet"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


QUOTE = module("soxl_close_shared_quotes", ANALYSIS / "sox_open_switch/code/premarket_quotes.py")
STUDY = module("soxl_close_quote_stats", ROOT / "code/study.py")


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def summarize(frame, chosen, quote_records):
    periods = {"H1": ("2026-01-01", "2026-06-30"),
               "H2": ("2026-07-01", STUDY.END), "all": ("2026-01-01", STUDY.END)}
    cells = []
    for tf in STUDY.TF:
        for ex in ("16:00", *STUDY.EXITS):
            for period, (start, end) in periods.items():
                x = frame[(frame.tf == tf) & (frame.exit_hm == ex) & frame.day.between(start, end)]
                valid = x[x.quote_valid]
                clean = valid[~valid.ratio_flag]
                cells.append({"tf": tf, "exit_hm": ex, "period": period,
                              "n_signal_trades": len(x), "n_valid": len(valid),
                              "n_ratio_flagged": int(valid.ratio_flag.sum()),
                              "excluded_reasons": x.loc[~x.quote_valid, "quote_reason"].value_counts().to_dict(),
                              "quote": STUDY.stats(valid.quote_return_pct),
                              "quote_cost2": STUDY.stats(valid.quote_return_cost2_pct),
                              "quote_without_ratio_flags": STUDY.stats(clean.quote_return_pct),
                              "model_gross_matched": STUDY.stats(valid.gross_pct),
                              "model_10bp_matched": STUDY.stats(valid.net_10bp_pct),
                              "mean_entry_half_spread_bp": valid.entry_half_spread_bp.mean(),
                              "mean_exit_half_spread_bp": valid.exit_half_spread_bp.mean(),
                              "median_exit_half_spread_bp": valid.exit_half_spread_bp.median(),
                              "drop_best2_quote_mean_pct": valid.quote_return_pct.sort_values().iloc[:-2].mean() if len(valid) > 2 else None})
    wf = []
    for selected in chosen:
        tf, ex = selected["tf"], selected["selected_exit"]
        wf.append({"tf": tf, "selected_exit": ex,
                   "selection": "unchanged H1 model mean at 10bp one-way cost; quote data do not reselect",
                   "H1": next((x for x in cells if x["tf"] == tf and x["exit_hm"] == ex and x["period"] == "H1"), None),
                   "H2": next((x for x in cells if x["tf"] == tf and x["exit_hm"] == ex and x["period"] == "H2"), None)})
    valid = quote_records[quote_records.valid]
    spreads = []
    if len(valid):
        q = valid.copy()
        q["hm"] = pd.to_datetime(q.target, utc=True).dt.tz_convert(STUDY.ET).dt.strftime("%H:%M")
        q["half_spread_bp"] = (q.ask - q.bid) / (q.ask + q.bid) * 10000
        for hm, group in q.groupby("hm"):
            spreads.append({"hm": hm, "n": len(group), "mean_half_spread_bp": group.half_spread_bp.mean(),
                            "median_half_spread_bp": group.half_spread_bp.median(),
                            "max_half_spread_bp": group.half_spread_bp.max(),
                            "p95_half_spread_bp": group.half_spread_bp.quantile(.95)})
    return clean_json({"scope": "SOXL signal-day last SIP NBBO at/before entry15:31 and fixed exits; raw ask->bid",
                       "max_age_seconds": QUOTE.MAX_AGE_SECONDS, "ratio_change_flag_pct": QUOTE.MAX_RATIO_CHANGE * 100,
                       "limitations": "Displayed quote size is not guaranteed execution. Additional slippage, impact, exchange rate and tax excluded. Missing/invalid quotes excluded without zero fill; compare matched model returns. Ratio flags can include within-minute price movement, not only split errors.",
                       "quote_validation": "Last quote only; positive bid/ask and sizes, bid<=ask, finite, age0~30sec. No back-selection of an earlier valid quote.",
                       "n_unique_requests": len(quote_records), "n_unique_valid": len(valid),
                       "invalid_quote_reasons": quote_records.loc[~quote_records.valid, "reason"].value_counts().to_dict(),
                       "spread_by_time": spreads, "cells": cells, "walk_forward": wf})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    want = sorted({QUOTE.aware_timestamp(t).isoformat() for t in json.loads((OUT / "quote_targets.json").read_text(encoding="utf-8"))})
    records = pd.read_parquet(CACHE).to_dict("records") if CACHE.exists() else []
    have = {QUOTE.aware_timestamp(r["target"]).isoformat() for r in records}
    todo = [t for t in want if t not in have]
    if todo and not args.offline:
        sys.path.insert(0, str(ANALYSIS / "hatta_bar/code"))
        from fetch_alpaca import keys, Pace
        head, pace = keys(), Pace(3.0)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"SOXL quote requests {len(todo)}, cached {len(have)}", flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            for i, record in enumerate(pool.map(lambda t: QUOTE.last_quote("SOXL", t, head, pace), todo), 1):
                records.append(record)
                if i % 30 == 0:
                    pd.DataFrame(records).to_parquet(CACHE, index=False)
                    print(f"quotes {i}/{len(todo)} elapsed {time.monotonic() - started:.1f}s", flush=True)
        pd.DataFrame(records).to_parquet(CACHE, index=False)
    lookup = {QUOTE.aware_timestamp(r["target"]).isoformat(): r for r in records}
    for target in want:
        lookup.setdefault(target, QUOTE.quote_record("SOXL", target, None))
    unique = pd.DataFrame([lookup[t] for t in want])
    unique.to_csv(OUT / "quotes_targets.csv", index=False)
    trades = pd.read_csv(OUT / "trades.csv")
    trades = trades[trades.signal & trades.delay.eq(1)]
    rows = []
    for t in trades.to_dict("records"):
        en, ex = (lookup[QUOTE.aware_timestamp(t[field]).isoformat()] for field in ("entry", "exit"))
        row = {**t, **QUOTE.audit_trade(en, ex, t["entry_price"], t["exit_price"])}
        for prefix, quote in (("entry", en), ("exit", ex)):
            for k in ("quote_timestamp", "bid", "ask", "bid_size", "ask_size", "age_seconds", "reason"):
                row[f"{prefix}_{k}"] = quote.get(k)
            row[f"{prefix}_half_spread_bp"] = ((quote["ask"] - quote["bid"]) / (quote["ask"] + quote["bid"]) * 10000) if quote.get("valid") else np.nan
        rows.append(row)
    audited = pd.DataFrame(rows)
    audited.to_csv(OUT / "quotes_trades.csv", index=False)
    selected = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))["walk_forward"]
    summary = summarize(audited, selected, unique)
    summary["runtime_seconds"] = time.monotonic() - started
    (OUT / "quotes_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"requests": len(want), "valid_quotes": int(unique.valid.sum()), "trades": len(audited),
                      "valid_trades": int(audited.quote_valid.sum()), "elapsed_seconds": summary["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
