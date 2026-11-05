"""Independent lower-RSI threshold audit from raw minute bars and cached quotes.

Run after thresholds.py. This audit does not use threshold-study calculations
to choose entries or compute price/quote returns.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
ET = "America/New_York"
EXITS = ("16:30", "17:00", "18:00", "19:00")
PERIODS = {"H1": ("2026-01-01", "2026-06-30"),
           "H2": ("2026-07-01", "2026-11-03"),
           "all": ("2026-01-01", "2026-11-03")}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def independent_inputs():
    audit = module("threshold_raw_rsi_audit", ROOT / "code/audit.py")
    raw = audit.raw_data()
    schedule = xc.get_calendar("XNYS").schedule.loc["2025-01-01":"2026-11-03"]
    one, three = [], []
    for _, row in schedule.iterrows():
        op, cl = row["open"].tz_convert(ET), row["close"].tz_convert(ET)
        bars = raw.loc[op:cl - pd.Timedelta(minutes=1), "Close"]
        one.append(bars)
        bucket = ((bars.index - op) / pd.Timedelta(minutes=3)).astype(int)
        c = bars.groupby(bucket).last()
        c.index = op + pd.to_timedelta(c.index * 3, unit="min")
        three.append(c)
    by_tf = {"1m": pd.concat(one), "3m": pd.concat(three)}
    rows = []
    for tf, close in by_tf.items():
        rsi = pd.Series(audit.loop_rsi(close.to_numpy(), 14), index=close.index)
        step = 1 if tf == "1m" else 3
        for day, row in schedule.loc["2026-01-01":].iterrows():
            cl = row["close"].tz_convert(ET)
            if cl.strftime("%H:%M") != "16:00":
                continue
            decision = cl - pd.Timedelta(minutes=30)
            source = decision - pd.Timedelta(minutes=step)
            rows.append({"tf": tf, "day": str(day.date()), "rsi": float(rsi.at[source]),
                         "entry": decision + pd.Timedelta(minutes=1)})
    return raw, pd.DataFrame(rows)


def quote_valid(record):
    try:
        age = (pd.Timestamp(record["target"]) - pd.Timestamp(record["quote_timestamp"])).total_seconds()
        vals = [float(record[k]) for k in ("bid", "ask", "bid_size", "ask_size")]
        bid, ask, bs, ass = vals
        return bool(np.isfinite(vals).all() and 0 <= age <= 30 and 0 < bid <= ask and bs > 0 and ass > 0)
    except (KeyError, TypeError, ValueError):
        return False


def basic_stats(values):
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return {"n": 0}
    rng = np.random.default_rng(20261105)
    boot = rng.choice(a, size=(10000, len(a)), replace=True).mean(1)
    equity = np.r_[1., np.cumprod(1 + a / 100)]
    return {"n": len(a), "mean_pct": float(a.mean()), "median_pct": float(np.median(a)),
            "win_rate_pct": float((a > 0).mean() * 100), "worst_pct": float(a.min()),
            "best_pct": float(a.max()), "mean_ci95_pct": np.quantile(boot, [.025, .975]).tolist(),
            "compound_pct": float((equity[-1] - 1) * 100),
            "closed_trade_mdd_pct": float((equity / np.maximum.accumulate(equity) - 1).min() * 100)}


def independent_events(raw, signals):
    records = pd.read_csv(OUT / "quotes_targets.csv").to_dict("records")
    quotes = {pd.Timestamp(r["target"]): r for r in records}
    rows = []
    for signal in signals[signals.rsi <= 30].to_dict("records"):
        en = signal["entry"]
        entry_quote = quotes[en.tz_convert("UTC")]
        for hm in EXITS:
            ex = pd.Timestamp(f'{signal["day"]} {hm}', tz=ET)
            ep, xp = float(raw.at[en, "Open"]), float(raw.at[ex, "Open"])
            exit_quote = quotes[ex.tz_convert("UTC")]
            valid = quote_valid(entry_quote) and quote_valid(exit_quote)
            em = (float(entry_quote["ask"]) + float(entry_quote["bid"])) / 2
            xm = (float(exit_quote["ask"]) + float(exit_quote["bid"])) / 2
            eh = (float(entry_quote["ask"]) - float(entry_quote["bid"])) / 2
            xh = (float(exit_quote["ask"]) - float(exit_quote["bid"])) / 2
            rows.append({**signal, "exit_hm": hm, "gross_pct": (xp / ep - 1) * 100,
                         "net_10bp_pct": (xp * .999 / (ep * 1.001) - 1) * 100,
                         "quote_valid": valid,
                         "mae_pct": (float(raw.loc[en:ex - pd.Timedelta(minutes=1), "Low"].min()) / ep - 1) * 100,
                         "quote_cost2_pct": (max(0., xm - 2 * xh) / (em + 2 * eh) - 1) * 100 if valid else np.nan,
                         "quote_return_pct": (float(exit_quote["bid"]) / float(entry_quote["ask"]) - 1) * 100 if valid else np.nan})
    return pd.DataFrame(rows)


def expect_cells(signals, events):
    cells, selection = {}, {}
    for tf in ("1m", "3m"):
        last_days = set()
        for level in range(1, 31):
            days = set(signals[signals.tf.eq(tf) & signals.rsi.le(level)].day)
            assert last_days <= days, "threshold populations must be nested"
            last_days = days
            for hm in EXITS:
                for period, (start, end) in PERIODS.items():
                    df = events[events.tf.eq(tf) & events.rsi.le(level) & events.exit_hm.eq(hm)
                                & events.day.between(start, end)].sort_values("day")
                    q = df[df.quote_valid]
                    cells[tf, level, hm, period] = {"signal_days": df.day.tolist(),
                                                   "valid_quote_days": q.day.tolist(),
                                                   "gross": basic_stats(df.gross_pct),
                                                   "model_10bp": basic_stats(df.net_10bp_pct),
                                                   "quote": basic_stats(q.quote_return_pct),
                                                   "quote_cost2": basic_stats(q.quote_cost2_pct),
                                                   "mae_worst_pct": float(df.mae_pct.min()) if len(df) else None}
            candidates = [(hm, cells[tf, level, hm, "H1"]["model_10bp"]) for hm in EXITS]
            available = [(hm, st["mean_pct"]) for hm, st in candidates if st["n"] > 0]
            selection[tf, level] = max(available, key=lambda pair: pair[1])[0] if available else None
    return cells, selection


def same_stat(actual, expected):
    assert actual["n"] == expected["n"]
    if expected["n"] == 0:
        # A no-trade return must not be represented as observed0% performance.
        assert actual.get("mean_pct") is None
    for name, value in expected.items():
        if name == "n":
            continue
        assert np.allclose(actual[name], value, atol=1e-10, rtol=0), (name, actual[name], value)


def main():
    raw, signals = independent_inputs()
    recorded = pd.read_csv(OUT / "signals.csv")
    merged = signals.merge(recorded[["day", "tf", "rsi"]], on=["day", "tf"], suffixes=("_ind", "_prod"))
    assert len(merged) == 422
    rsi_error = float(np.max(np.abs(merged.rsi_ind - merged.rsi_prod)))
    assert rsi_error < 1e-10
    events = independent_events(raw, signals)
    expected, selected = expect_cells(signals, events)
    # Inclusive boundary and a wholly empty threshold are explicit contracts.
    fixture = pd.Series([9.999, 10., 10.001, np.nan])
    assert fixture.le(10).tolist() == [True, True, False, False]
    assert not signals.rsi.le(1).any()
    for tf in ("1m", "3m"):
        assert selected[tf, 1] is None

    # Compare independently calculated cells with the production schema.
    summary_path = OUT / "thresholds_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = check_summary(summary, expected, selected)
    production_checks = check_production_contracts(fixture)
    report = {"pass": True, "raw_signal_rows": len(signals), "thresholds": list(range(1, 31)),
              "timeframes": ["1m", "3m"], "candidate_cells": 240, "period_cells": len(expected),
              "rsi_max_absolute_error": rsi_error, "raw_event_horizon_rows": len(events),
              "nested_populations_checked": True, "inclusive_boundary_fixture_pass": True,
              "empty_threshold_fixture_pass": True, **checks, **production_checks,
              "counts_at_display_levels": {tf: {str(n): int((signals.tf.eq(tf) & signals.rsi.le(n)).sum())
                                                  for n in (30, 25, 20, 15, 10, 5)} for tf in ("1m", "3m")}}
    (OUT / "thresholds_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def check_summary(summary, expected, selected):
    # Production schema adapter is intentionally separate from independent math.
    cells = summary["cells"]
    assert len(cells) == len(expected) == 720
    seen = set()
    for actual in cells:
        key = actual["tf"], actual["threshold"], actual["exit_hm"], actual["period"]
        assert key not in seen
        seen.add(key)
        exp = expected[key]
        assert actual["signal_days"] == exp["signal_days"]
        assert actual["valid_quote_days"] == exp["valid_quote_days"]
        assert actual["n_signal"] == exp["gross"]["n"]
        assert actual["n_valid_quote"] == exp["quote"]["n"]
        same_stat(actual["model_gross"], exp["gross"])
        same_stat(actual["model_net10bp"], exp["model_10bp"])
        same_stat(actual["quote"], exp["quote"])
        same_stat(actual["quote_cost2"], exp["quote_cost2"])
        if exp["mae_worst_pct"] is None:
            assert actual["mae_worst_pct"] is None
        else:
            assert abs(actual["mae_worst_pct"] - exp["mae_worst_pct"]) < 1e-10
    assert seen == expected.keys()
    assert summary["comparison_fixed_exits"] == {"1m": "19:00", "3m": "18:00"}
    wf = summary["walk_forward"]
    assert len(wf) == 60
    for result in wf:
        key = result["tf"], result["threshold"]
        assert result["selected_exit"] == selected[key]
        if selected[key] is None:
            assert result["H1"] is None and result["H2"] is None
        else:
            for period in ("H1", "H2"):
                assert result[period]["period"] == period
                assert result[period]["exit_hm"] == selected[key]
                exp = expected[*key, selected[key], period]
                same_stat(result[period]["model_net10bp"], exp["model_10bp"])
                same_stat(result[period]["quote"], exp["quote"])
        assert not result["passes_frozen_gate"]  # All H2 populations have <20 signals.
    return {"period_cells_checked": len(cells), "walk_forward_rows_checked": len(wf),
            "all_statistics_match": True, "H1_selection_matches_independent_max": True,
            "no_H2_reselection": True, "empty_means_preserved_as_missing": True}


def check_production_contracts(boundary_fixture):
    production = module("threshold_production_contract_audit", ROOT / "code/thresholds.py")
    assert production.eligible(boundary_fixture, 10).tolist() == [True, True, False, False]
    events = production.load_events()
    _, original = production.analyze(events)
    changed = events.copy()
    h2 = changed.day.between(*PERIODS["H2"])
    for field in ("gross_pct", "net_10bp_pct", "quote_return_pct", "quote_return_cost2_pct"):
        changed.loc[h2, field] = changed.loc[h2, field] * 3 + 50
    _, perturbed = production.analyze(changed)
    changed_h2_cells = 0
    for before, after in zip(original, perturbed, strict=True):
        assert (before["tf"], before["threshold"]) == (after["tf"], after["threshold"])
        assert before["selected_exit"] == after["selected_exit"], "H2 returns influenced exit selection"
        assert before["H1"] == after["H1"], "H2 mutation altered H1 calculations"
        if before["H2"] and before["H2"]["n_signal"]:
            assert before["H2"]["model_net10bp"] != after["H2"]["model_net10bp"]
            changed_h2_cells += 1
    assert changed_h2_cells > 0
    tied = events.copy()
    tied.loc[tied.day.between(*PERIODS["H1"]), "net_10bp_pct"] = 0.
    _, tied_choices = production.analyze(tied)
    for result in tied_choices:
        assert result["selected_exit"] in (None, EXITS[0]), "tied H1 returns must choose the earliest exit"
    return {"production_inclusive_boundary_fixture_pass": True,
            "production_earliest_exit_tie_fixture_pass": True,
            "H2_return_mutation_selection_invariant": True,
            "H2_return_mutation_H1_invariant": True,
            "H2_return_mutation_affected_populated_rows": changed_h2_cells}


if __name__ == "__main__":
    main()
