"""Render the confirmed last-30-minute RSI rule using cached actual prices.

This describes signals and order timing, without calculating strategy returns.
Run: py data/derived/analysis/soxl_close_rsi/code/visualize.py
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FILE = "soxl-rsi-rebound-20261108-a6cf2e.html"
FRAMES = {"1m": 1, "3m": 3}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def first_signal(day, tf, level, latest=958, point=False, rising=False):
    values = day["r"][tf]
    for i, (t, value) in enumerate(zip(day["times"][tf], values, strict=True)):
        if value is None or not np.isfinite(value) or not 930 <= t <= latest or (point and t != 930):
            continue
        if value <= level and (not rising or (i > 0 and values[i - 1] is not None
                                              and np.isfinite(values[i - 1]) and value > values[i - 1])):
            return t
    return None


def build_inputs():
    study = module("window_visual_study", ROOT / "code/study.py")
    independent = module("window_visual_independent", ROOT / "code/audit.py")
    full, _ = study.load_bars()
    reg, schedule, _ = study.regular_bars(full)
    rma = study.shared_rma()
    prices = {"1m": reg.Close,
              "3m": reg.Close.resample("3min", origin="start_day", offset="9h30min",
                                       closed="left", label="left").last().dropna()}
    rsi = {}
    error = {}
    for tf, step in FRAMES.items():
        values = rma(prices[tf].to_numpy(), 14)[2]
        independent_values = independent.loop_rsi(prices[tf].to_numpy(), 14)
        error[tf] = float(np.nanmax(np.abs(values - independent_values)))
        assert error[tf] < 1e-10
        # A bar's RSI only becomes observable at its completion time.
        rsi[tf] = pd.Series(values, index=prices[tf].index + pd.Timedelta(minutes=step))
    p_times = list(range(900, 961)) + list(range(965, 1141, 5))
    r_times = {tf: list(range(900, 960, step)) for tf, step in FRAMES.items()}
    days = []
    exact_days = []
    for day, row in schedule.loc["2026-01-01":].iterrows():
        if row["close"].tz_convert(study.ET).strftime("%H:%M") != "16:00":
            continue
        date = str(day.date())
        base = pd.Timestamp(date, tz=study.ET)
        pi = base + pd.to_timedelta(p_times, unit="min")
        p = full.Open.reindex(pi)
        # Empty minutes stay missing; do not manufacture a price or tradable bar.
        p = p.where(full.Volume.reindex(pi).gt(0))
        rs = {tf: rsi[tf].reindex(base + pd.to_timedelta(times, unit="min")).tolist()
              for tf, times in r_times.items()}
        assert all(np.isfinite(v).all() for v in rs.values())
        exact = {"date": date, "times": r_times, "r": rs}
        rounded = {"date": date, "p": [round(float(x), 6) if pd.notna(x) else None for x in p],
                   "r": {tf: [round(float(x), 6) for x in values] for tf, values in rs.items()}}
        # Quantization for a compact offline chart must never change a first signal.
        for tf in FRAMES:
            for level in range(1, 31):
                for rising in (False, True):
                    assert first_signal(exact, tf, level, rising=rising) == first_signal(rounded | {"times": r_times}, tf, level, rising=rising)
        exact_days.append(exact)
        days.append(rounded)
    assert len(days) == 211
    old = pd.read_csv(OUT / "signals.csv")
    point_error = 0.
    for d in exact_days:
        for tf in FRAMES:
            old_rsi = old.loc[old.day.eq(d["date"]) & old.tf.eq(tf), "rsi"].item()
            actual = d["r"][tf][d["times"][tf].index(930)]
            point_error = max(point_error, abs(actual - old_rsi))
    assert point_error < 1e-10
    return {"study": study, "full": full, "days": days, "exact_days": exact_days,
            "p_times": p_times, "r_times": r_times, "point_error": point_error, "error": error}


def main():
    inputs = build_inputs()
    days, exact_days = inputs["days"], inputs["exact_days"]
    p_times, r_times = inputs["p_times"], inputs["r_times"]
    summary = {"start": days[0]["date"], "end": days[-1]["date"], "sessions": len(days),
               "signal_rule": "first completed-bar RSI <= threshold from15:30 through15:58 ET",
               "entry_delay_minutes": 1, "last_entry": "15:59 ET", "max_trades_per_day": 1,
               "returns_calculated": False, "counts": {}}
    for tf in FRAMES:
        summary["counts"][tf] = {
            str(level): {"point": sum(first_signal(d, tf, level, point=True) is not None for d in exact_days),
                         "window": sum(first_signal(d, tf, level) is not None for d in exact_days),
                         "through_1559": sum(first_signal(d, tf, level, latest=959) is not None for d in exact_days),
                         "rising": sum(first_signal(d, tf, level, rising=True) is not None for d in exact_days)}
            for level in range(1, 31)}
    assert summary["counts"]["1m"]["30"] == {"point": 12, "window": 64, "through_1559": 66, "rising": 23}
    assert summary["counts"]["3m"]["30"] == {"point": 12, "window": 36, "through_1559": 36, "rising": 11}
    summary["rising_rule"] = "current completed RSI <= threshold AND current RSI > previous completed RSI"
    sample = next(d for d in exact_days if d["date"] == "2026-01-15")
    assert first_signal(sample, "1m", 30) == 932
    assert first_signal(sample, "3m", 30) == 939
    summary["example_2026-01-15"] = {
        tf: {"rsi_1530": sample["r"][tf][sample["times"][tf].index(930)],
             "decision_minute": first_signal(sample, tf, 30),
             "entry_minute": first_signal(sample, tf, 30) + 1}
        for tf in FRAMES}
    for tf in FRAMES:
        decision = first_signal(sample, tf, 30, rising=True)
        pos = sample["times"][tf].index(decision)
        summary["example_2026-01-15"][tf] |= {"rising_decision_minute": decision, "rising_entry_minute": decision + 1,
                                            "rsi_before_rise": sample["r"][tf][pos - 1], "rsi_on_rise": sample["r"][tf][pos]}
    assert summary["example_2026-01-15"]["1m"]["rising_decision_minute"] == 953
    assert summary["example_2026-01-15"]["3m"]["rising_decision_minute"] == 954
    meta = {k: summary[k] for k in ("start", "end", "sessions")}
    data = {"meta": meta | {"default_day": "2026-01-15"}, "p_times": p_times, "r_times": r_times, "days": days}
    template = (ROOT / "code/strategy_visual_template.html").read_text(encoding="utf-8")
    assert template.count("__SOXL_VISUAL_DATA__") == 1
    html = template.replace("__SOXL_VISUAL_DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    assert len(html.encode("utf-8")) < 512 * 1024
    (ROOT / "strategy_visual.html").write_text(html, encoding="utf-8")
    scratch = os.getenv("ORCA_CHAT_SCRATCH_DIR")
    if scratch:
        (Path(scratch) / FILE).write_text(html, encoding="utf-8")
    (OUT / "window_visual_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {"pass": True, "regular_sessions": len(days), "max_rsi_error_independent_loop": inputs["error"],
             "prior_point_rsi_max_error": inputs["point_error"], "rounding_first_signal_invariant": True,
             "last_signal_1m": "15:58", "last_signal_3m": "15:57", "first_signal_state_inclusive": True,
             "sample_first_1m": "15:32", "sample_first_3m": "15:39", "html_bytes": len(html.encode("utf-8")),
             "sample_rising_1m": "15:53", "sample_rising_3m": "15:54", "rising_requires_positive_completed_bar_change": True,
             "returns_calculated": False}
    (OUT / "window_visual_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"audit": audit, "display_counts": {tf: {n: summary["counts"][tf][n] for n in ("30", "20", "10")} for tf in FRAMES},
                      "file": FILE}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
