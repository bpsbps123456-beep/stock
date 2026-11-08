"""Independent raw-bar audit of the closing-window RSI rebound comparison.

Run after rebound_study.py. Inputs, first signals, fill arithmetic and statistics
are rebuilt independently; production helpers are used only for contract checks.
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
PERIODS = {"all": ("2026-01-01", "2026-11-03"),
           "H1": ("2026-01-01", "2026-06-30"),
           "H2": ("2026-07-01", "2026-11-03")}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def independent_inputs():
    source = ROOT.parent / "sox_open_switch/cache"
    pieces = [pd.read_parquet(source / p / "SOXL.parquet").loc["2025-01-01":"2026-11-03"]
              for p in ("alp1m", "alp1m_recent")]
    raw = pd.concat(pieces).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    audit = module("independent_rebound_scalar_rsi", ROOT / "code/audit.py")
    schedule = xc.get_calendar("XNYS").schedule.loc["2025-01-01":"2026-11-03"]
    chunks = {"1m": [], "3m": []}
    for _, row in schedule.iterrows():
        op, cl = row["open"].tz_convert(ET), row["close"].tz_convert(ET)
        close = raw.loc[op:cl - pd.Timedelta(minutes=1), "Close"]
        chunks["1m"].append(close)
        bucket = ((close.index - op) / pd.Timedelta(minutes=3)).astype(int)
        three = close.groupby(bucket).last()
        three.index = op + pd.to_timedelta(three.index * 3, unit="min")
        chunks["3m"].append(three)
    events = []
    full_series = {}
    for tf, step in (("1m", 1), ("3m", 3)):
        close = pd.concat(chunks[tf])
        rsi = pd.Series(audit.loop_rsi(close.to_numpy(), 14),
                        index=close.index + pd.Timedelta(minutes=step))
        frame = pd.DataFrame({"rsi": rsi, "prev_rsi": rsi.shift(1)})
        frame["day"] = frame.index.strftime("%Y-%m-%d")
        frame["minute"] = frame.index.hour * 60 + frame.index.minute
        frame["tf"] = tf
        full_series[tf] = frame
        events.append(frame[frame.day.ge("2026-01-01") & frame.minute.between(930, 958)])
    return raw, pd.concat(events).sort_index(), full_series


def independent_signals(events):
    rows = []
    for tf in ("1m", "3m"):
        for level in range(1, 31):
            subset = events[events.tf.eq(tf) & events.rsi.le(level)]
            for rule in ("level", "rising"):
                eligible = subset if rule == "level" else subset[subset.rsi.gt(subset.prev_rsi)]
                for at, row in eligible.groupby("day", sort=True).head(1).iterrows():
                    rows.append({"day": row.day, "tf": tf, "threshold": level, "rule": rule,
                                 "decision": at, "rsi": row.rsi, "prev_rsi": row.prev_rsi,
                                 "entry": at + pd.Timedelta(minutes=1)})
    return pd.DataFrame(rows)


def independent_trades(raw, signals):
    rows = []
    missing = []
    for s in signals.to_dict("records"):
        for hm in EXITS:
            en = s["entry"]
            ex = pd.Timestamp(f'{s["day"]} {hm}', tz=ET)
            if en not in raw.index or ex not in raw.index or min(raw.at[en, "Volume"], raw.at[ex, "Volume"]) <= 0:
                missing.append({"day": s["day"], "tf": s["tf"], "threshold": s["threshold"],
                                "rule": s["rule"], "exit_hm": hm})
                continue
            ep, xp = float(raw.at[en, "Open"]), float(raw.at[ex, "Open"])
            path = raw.loc[en:ex - pd.Timedelta(minutes=1)]
            rows.append({**s, "exit_hm": hm, "exit": ex, "entry_price": ep, "exit_price": xp,
                         "gross_pct": (xp / ep - 1) * 100,
                         **{f"net_{bp}bp_pct": (xp * (1 - bp / 10000) / (ep * (1 + bp / 10000)) - 1) * 100
                            for bp in (0, 5, 10, 20)},
                         "mae_pct": (float(path.Low.min()) / ep - 1) * 100,
                         "mfe_pct": (float(path.High.max()) / ep - 1) * 100})
    return pd.DataFrame(rows), missing


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0}
    rng = np.random.default_rng(20261105)
    boot = rng.choice(a, (10000, len(a)), replace=True).mean(axis=1)
    equity = np.r_[1., np.cumprod(1 + a / 100)]
    return {"n": len(a), "mean_pct": float(a.mean()), "median_pct": float(np.median(a)),
            "win_rate_pct": float((a > 0).mean() * 100), "worst_pct": float(a.min()),
            "best_pct": float(a.max()), "mean_ci95_pct": np.quantile(boot, [.025, .975]).tolist(),
            "compound_pct": float((equity[-1] - 1) * 100),
            "closed_trade_mdd_pct": float((equity / np.maximum.accumulate(equity) - 1).min() * 100)}


def check_contracts():
    production = module("rebound_first_signal_contracts", ROOT / "code/visualize.py")
    def day(values):
        return {"times": {"1m": [929, 930, 931, 932]}, "r": {"1m": values}}
    assert production.first_signal(day([35., 29., 28., 31.]), "1m", 30) == 930
    assert production.first_signal(day([35., 29., 28., 31.]), "1m", 30, rising=True) is None
    assert production.first_signal(day([30., 30., 30., 30.]), "1m", 30) == 930
    assert production.first_signal(day([30., 30., 30., 30.]), "1m", 30, rising=True) is None
    assert production.first_signal(day([29., 30., 31., 35.]), "1m", 30, rising=True) == 930
    assert production.first_signal(day([29., 28., 29., 35.]), "1m", 30, rising=True) == 931
    assert production.first_signal(day([29., 28., 31., 35.]), "1m", 30, rising=True) is None
    values = [35., 29., 28., 29.]
    original = day(values)
    for cutoff in (930, 931):
        altered = day([v if t <= cutoff else 1000. - 1000. * (i % 2)
                       for i, (t, v) in enumerate(zip(original["times"]["1m"], values, strict=True))])
        for rising in (False, True):
            assert production.first_signal(original, "1m", 30, latest=cutoff, rising=rising) == \
                   production.first_signal(altered, "1m", 30, latest=cutoff, rising=rising)
    return {"inclusive_boundary_pass": True, "equal_rsi_is_not_rising": True,
            "cross_above_threshold_is_not_entry": True, "positive_change_below_threshold_pass": True,
            "future_rsi_event_mutation_invariant": True}


def same_stats(actual, expected):
    assert actual["n"] == expected["n"]
    if not expected["n"]:
        assert actual.get("mean_pct") is None
    for key, value in expected.items():
        assert np.allclose(actual[key], value, atol=1e-10, rtol=0), (key, actual[key], value)


def key_signals(df):
    return {(r["tf"], int(r["threshold"]), r["rule"], r["day"]): r for r in df.to_dict("records")}


def check_signal_output(signals):
    recorded = pd.read_csv(OUT / "rebound_signals.csv")
    expected, actual = key_signals(signals), key_signals(recorded)
    assert len(expected) == len(signals) == len(recorded) == len(actual)
    assert actual.keys() == expected.keys()
    error = 0.
    for key, exp in expected.items():
        row = actual[key]
        assert pd.Timestamp(row["decision"]) == exp["decision"]
        assert row["decision_minute"] == exp["decision"].hour * 60 + exp["decision"].minute
        assert exp["entry"] < pd.Timestamp(f'{exp["day"]} 16:00', tz=ET)
        for actual_name, exp_name in (("rsi", "rsi"), ("previous_rsi", "prev_rsi")):
            error = max(error, abs(row[actual_name] - exp[exp_name]))
        assert row["signal"]
    assert error < 1e-10
    return {"all_first_signal_dates_and_times_match": True, "max_rsi_absolute_error": error,
            "signal_rows_checked": len(signals)}


def key_trades(df):
    return {(r["tf"], int(r["threshold"]), r["rule"], r["day"], r["exit_hm"]): r
            for r in df.to_dict("records")}


def check_trade_output(trades, signals):
    recorded = pd.read_csv(OUT / "rebound_trades.csv")
    expected = key_trades(trades)
    actual = key_trades(recorded)
    assert len(recorded) == len(actual) == len(signals) * 4
    valid = {k: r for k, r in actual.items() if r["valid_model_price"]}
    assert valid.keys() == expected.keys()
    error = 0.
    for key, exp in expected.items():
        row = valid[key]
        for field in ("entry", "exit"):
            assert pd.Timestamp(row[field]) == exp[field]
        for field in ("entry_price", "exit_price", "gross_pct", "mae_pct", "mfe_pct",
                      "net_0bp_pct", "net_5bp_pct", "net_10bp_pct", "net_20bp_pct"):
            error = max(error, abs(row[field] - exp[field]))
    assert error < 1e-10
    return {"all_raw_entry_exit_prices_and_returns_match": True, "trade_rows_checked": len(expected),
            "trade_rows_including_missing": len(actual), "max_price_return_absolute_error": error}


def expected_periods(signals, trades):
    cells = {}
    pairs = {}
    for tf in ("1m", "3m"):
        for threshold in range(1, 31):
            for hm in EXITS:
                for period, (start, finish) in PERIODS.items():
                    sample_signals = signals[signals.tf.eq(tf) & signals.threshold.eq(threshold)
                                             & signals.day.between(start, finish)].sort_values("day")
                    sample_trades = trades[trades.tf.eq(tf) & trades.threshold.eq(threshold)
                                           & trades.exit_hm.eq(hm) & trades.day.between(start, finish)].sort_values("day")
                    for rule in ("level", "rising"):
                        sig = sample_signals[sample_signals.rule.eq(rule)]
                        valid = sample_trades[sample_trades.rule.eq(rule)]
                        cells[tf, threshold, rule, hm, period] = {
                            "n_signal": len(sig), "n_valid_model": len(valid),
                            "signal_days": sig.day.tolist(), "valid_days": valid.day.tolist(),
                            "missing_price_days": sorted(set(sig.day) - set(valid.day)),
                            "gross": stats(valid.gross_pct),
                            "costs": {str(bp): stats(valid[f"net_{bp}bp_pct"]) for bp in (0, 5, 10, 20)},
                            "mae_worst_pct": float(valid.mae_pct.min()) if len(valid) else None}
                    a, b = (sample_trades[sample_trades.rule.eq(rule)].set_index("day") for rule in ("level", "rising"))
                    common = sorted(set(a.index) & set(b.index))
                    aa, bb = a.loc[common], b.loc[common]
                    rise_days = set(sample_signals.loc[sample_signals.rule.eq("rising"), "day"])
                    dropped = a.loc[~a.index.isin(rise_days)]
                    delta = stats(bb.net_10bp_pct - aa.net_10bp_pct)
                    ds = {"n": 0} if not delta["n"] else {
                        "n": delta["n"], "mean_pp": delta["mean_pct"], "median_pp": delta["median_pct"],
                        "positive_share_pct": delta["win_rate_pct"], "mean_ci95_pp": delta["mean_ci95_pct"],
                        "worst_pp": delta["worst_pct"], "best_pp": delta["best_pct"]}
                    wait = [(b.at[d, "decision"] - a.at[d, "decision"]).total_seconds() / 60 for d in common]
                    rise_signals = sample_signals[sample_signals.rule.eq("rising")]
                    pairs[tf, threshold, hm, period] = {
                        "common_days": common, "n_common_valid": len(common),
                        "same_day_level_net10bp": stats(aa.net_10bp_pct),
                        "same_day_rising_net10bp": stats(bb.net_10bp_pct), "rising_minus_level_pp": ds,
                        "mean_entry_wait_minutes": float(np.mean(wait)) if wait else None,
                        "baseline_only_days": dropped.index.tolist(),
                        "baseline_only_net10bp": stats(dropped.net_10bp_pct),
                        "rising_signal_but_missing_price_days": sorted(set(rise_signals.day) - set(b.index))}
    return cells, pairs


def check_cells_and_pairs(summary, expected_cells, expected_pairs):
    actual_cells = summary["cells"]
    assert len(actual_cells) == len(expected_cells) == 1440
    seen = set()
    for actual in actual_cells:
        key = actual["tf"], actual["threshold"], actual["rule"], actual["exit_hm"], actual["period"]
        assert key not in seen
        seen.add(key)
        exp = expected_cells[key]
        for field in ("n_signal", "n_valid_model", "signal_days", "valid_days", "missing_price_days"):
            assert actual[field] == exp[field], (key, field)
        same_stats(actual["gross"], exp["gross"])
        for bp in ("0", "5", "10", "20"):
            same_stats(actual["costs"][bp], exp["costs"][bp])
        if exp["mae_worst_pct"] is None:
            assert actual["mae_worst_pct"] is None
        else:
            assert abs(actual["mae_worst_pct"] - exp["mae_worst_pct"]) < 1e-10
    assert seen == expected_cells.keys()
    actual_pairs = summary["paired"]
    assert len(actual_pairs) == len(expected_pairs) == 720
    seen = set()
    for actual in actual_pairs:
        key = actual["tf"], actual["threshold"], actual["exit_hm"], actual["period"]
        assert key not in seen
        seen.add(key)
        exp = expected_pairs[key]
        for field in ("common_days", "n_common_valid", "baseline_only_days", "rising_signal_but_missing_price_days"):
            assert actual[field] == exp[field], (key, field)
        for field in ("same_day_level_net10bp", "same_day_rising_net10bp", "baseline_only_net10bp"):
            same_stats(actual[field], exp[field])
        for field, value in exp["rising_minus_level_pp"].items():
            assert np.allclose(actual["rising_minus_level_pp"][field], value, atol=1e-10, rtol=0), (key, field)
        if exp["mean_entry_wait_minutes"] is None:
            assert actual["mean_entry_wait_minutes"] is None
        else:
            assert abs(actual["mean_entry_wait_minutes"] - exp["mean_entry_wait_minutes"]) < 1e-10
    assert seen == expected_pairs.keys()
    return {"period_cells_checked": len(actual_cells), "paired_period_cells_checked": len(actual_pairs),
            "all_statistics_and_paired_differences_match": True}


def check_walk_forward(summary, cells, pairs):
    rows = summary["walk_forward"]
    assert len(rows) == 60
    seen = set()
    for row in rows:
        key = row["tf"], row["threshold"]
        assert key not in seen
        seen.add(key)
        choices = [(hm, cells[*key, "level", hm, "H1"]) for hm in EXITS]
        choices = [(hm, c) for hm, c in choices if c["n_valid_model"]]
        hm = max(choices, key=lambda item: item[1]["costs"]["10"]["mean_pct"])[0] if choices else None
        assert row["selected_exit"] == hm
        assert row["selection"] == "baseline_H1_net10bp_mean_only"
        if hm is None:
            assert row["H1"] is None and row["H2"] is None
        else:
            for period in ("H1", "H2"):
                for rule in ("level", "rising"):
                    cell = row[period][rule]
                    assert cell["exit_hm"] == hm and cell["period"] == period
                    same_stats(cell["costs"]["10"], cells[*key, rule, hm, period]["costs"]["10"])
                assert row[period]["paired"]["common_days"] == pairs[*key, hm, period]["common_days"]
    return {"walk_forward_rows_checked": len(rows), "baseline_H1_only_exit_selection_matches": True}


def check_h2_selection_invariance():
    production = module("rebound_selection_contracts", ROOT / "code/rebound_study.py")
    study = module("rebound_selection_shared_constants", ROOT / "code/study.py")
    trades = pd.read_csv(OUT / "rebound_trades.csv", float_precision="round_trip")
    _, _, before = production.analyze({"study": study}, trades)
    altered = trades.copy()
    h2 = altered.day.ge("2026-07-01")
    for field in ("gross_pct", "net_0bp_pct", "net_5bp_pct", "net_10bp_pct", "net_20bp_pct"):
        altered.loc[h2, field] = altered.loc[h2, field] * -3 + 50
    _, _, after = production.analyze({"study": study}, altered)
    populated_h2 = 0
    for a, b in zip(before, after, strict=True):
        assert (a["tf"], a["threshold"]) == (b["tf"], b["threshold"])
        assert a["selected_exit"] == b["selected_exit"], "H2 returns influenced exit selection"
        assert a["H1"] == b["H1"], "H2 mutation altered H1 results"
        if a["H2"] and a["H2"]["level"]["n_valid_model"]:
            assert a["H2"]["level"]["costs"]["10"] != b["H2"]["level"]["costs"]["10"]
            populated_h2 += 1
    assert populated_h2 > 0
    return {"H2_return_mutation_exit_selection_invariant": True,
            "H2_return_mutation_H1_invariant": True,
            "H2_return_mutation_affected_populated_rows": populated_h2}


def main():
    raw, events, _ = independent_inputs()
    signals = independent_signals(events)
    trades, missing = independent_trades(raw, signals)
    assert len(events.day.unique()) == 211
    counts = {tf: {str(c): {rule: int(len(signals[signals.tf.eq(tf) & signals.threshold.eq(c) & signals.rule.eq(rule)]))
                           for rule in ("level", "rising")} for c in (30, 20, 10)}
              for tf in ("1m", "3m")}
    assert counts["1m"]["30"] == {"level": 64, "rising": 23}
    assert counts["3m"]["30"] == {"level": 36, "rising": 11}
    assert counts["3m"]["20"]["rising"] == 0
    summary = json.loads((OUT / "rebound_summary.json").read_text(encoding="utf-8"))
    cells, pairs = expected_periods(signals, trades)
    checks = {**check_signal_output(signals), **check_trade_output(trades, signals),
              **check_cells_and_pairs(summary, cells, pairs), **check_walk_forward(summary, cells, pairs),
              **check_h2_selection_invariance()}
    report = {"pass": True, "sessions": 211, "independent_signal_rows": len(signals),
              "independent_trade_rows": len(trades), "missing_trade_rows": missing,
              "counts_at_display_levels": counts, **checks, **check_contracts()}
    (OUT / "rebound_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
