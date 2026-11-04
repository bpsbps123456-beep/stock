"""Independent audit of frozen close RSI study; does not select parameters.

Run: py data/derived/analysis/soxl_close_rsi/code/audit.py
Uses its own loop RSI, data loading and return arithmetic. It also checks the
production signal function after truncating and mutating future input.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT.parent
ET = "America/New_York"


def loop_rsi(values, n=14):
    """Independent scalar Wilder recurrence, including initialization."""
    values = list(map(float, values))
    ans = np.full(len(values), np.nan)
    if len(values) <= n:
        return ans
    first = [values[i] - values[i - 1] for i in range(1, n + 1)]
    up = sum(max(x, 0) for x in first) / n
    down = sum(max(-x, 0) for x in first) / n
    for i in range(n, len(values)):
        if i > n:
            delta = values[i] - values[i - 1]
            up = ((n - 1) * up + max(delta, 0)) / n
            down = ((n - 1) * down + max(-delta, 0)) / n
        ans[i] = 100 if down == 0 else 100 * up / (up + down)
    return ans


def raw_data():
    source = ANALYSIS / "sox_open_switch/cache"
    parts = [pd.read_parquet(source / p / "SOXL.parquet").loc["2025-01-01":"2026-11-03"]
             for p in ("alp1m", "alp1m_recent")]
    data = pd.concat(parts).sort_index()
    return data[~data.index.duplicated(keep="last")]


def independent_signals(data):
    schedule = xc.get_calendar("XNYS").schedule.loc["2025-01-01":"2026-11-03"]
    minute_chunks, three_chunks, five_chunks, daily_closes = [], [], [], []
    for day, row in schedule.iterrows():
        start = row["open"].tz_convert(ET)
        finish = row["close"].tz_convert(ET)
        chunk = data.loc[start:finish - pd.Timedelta(minutes=1), "Close"]
        minute_chunks.append(chunk)
        # Anchor explicitly at the regular-session open. Each value belongs to
        # the three-minute interval [start + 3k, start + 3(k+1)).
        bucket = (chunk.index - start) // pd.Timedelta(minutes=3)
        three = chunk.groupby(bucket).last()
        three.index = start + pd.to_timedelta(three.index.to_numpy() * 3, unit="min")
        three_chunks.append(three)
        # Group by exact five-minute bucket, rather than global resample.
        five_chunks.append(chunk.groupby(chunk.index.floor("5min")).last())
        daily_closes.append((day.date(), float(chunk.iloc[-1])))
    minute = pd.concat(minute_chunks)
    three = pd.concat(three_chunks)
    five = pd.concat(five_chunks)
    minute_rsi = pd.Series(loop_rsi(minute.to_numpy()), index=minute.index)
    minute42_rsi = pd.Series(loop_rsi(minute.to_numpy(), n=42), index=minute.index)
    three_rsi = pd.Series(loop_rsi(three.to_numpy()), index=three.index)
    five_rsi = pd.Series(loop_rsi(five.to_numpy()), index=five.index)
    daily = pd.Series(dict(daily_closes))
    rows = []
    for day, row in schedule.loc["2026-01-01":].iterrows():
        close = row["close"].tz_convert(ET)
        if close.hour != 16:
            continue
        decision = close - pd.Timedelta(minutes=30)
        latest = decision - pd.Timedelta(minutes=1)
        # Rebuild the entire daily prefix through the forming value, preventing
        # any use of today's eventual close or future daily indicator states.
        prev = daily.loc[daily.index < day.date()].to_numpy()
        forming = np.r_[prev, minute.loc[latest]]
        vals = {"1m": minute_rsi.loc[latest],
                "3m": three_rsi.loc[decision - pd.Timedelta(minutes=3)],
                "1m42": minute42_rsi.loc[latest],
                "5m": five_rsi.loc[decision - pd.Timedelta(minutes=5)],
                "daily": loop_rsi(forming)[-1]}
        for tf, value in vals.items():
            rows.append({"day": str(day.date()), "tf": tf, "rsi": value,
                         "signal": value <= 30})
    return pd.DataFrame(rows)


def main():
    out = ROOT / "results"
    data = raw_data()
    expected = independent_signals(data).set_index(["day", "tf"]).sort_index()
    actual = pd.read_csv(out / "signals.csv").set_index(["day", "tf"]).sort_index()
    assert actual.index.equals(expected.index)
    error = float(np.max(np.abs(actual.rsi - expected.rsi)))
    assert np.allclose(actual.rsi, expected.rsi, rtol=0, atol=1e-10)
    assert np.array_equal(actual.signal.to_numpy(), expected.signal.to_numpy())
    # The observed 3m candle must start at15:27 and end at the15:30
    # decision. A15:30-start candle would include prices unavailable then.
    three_actual = actual.xs("3m", level="tf")
    source_start = pd.to_datetime(three_actual.signal_bar_start, utc=True).dt.tz_convert(ET)
    decision_time = pd.to_datetime(three_actual.decision, utc=True).dt.tz_convert(ET)
    assert (source_start.dt.strftime("%H:%M") == "15:27").all()
    assert ((decision_time - source_start).dt.total_seconds() == 180).all()

    trades = pd.read_csv(out / "trades.csv")
    entry = pd.to_datetime(trades.entry, utc=True).dt.tz_convert(ET)
    exit_ = pd.to_datetime(trades.exit, utc=True).dt.tz_convert(ET)
    ep = data.Open.reindex(pd.DatetimeIndex(entry)).to_numpy()
    xp = data.Open.reindex(pd.DatetimeIndex(exit_)).to_numpy()
    assert np.allclose(ep, trades.entry_price, rtol=0, atol=1e-10)
    assert np.allclose(xp, trades.exit_price, rtol=0, atol=1e-10)
    ret_error = float(np.max(np.abs((xp / ep - 1) * 100 - trades.gross_pct)))
    assert ret_error < 1e-10
    for bp in (0, 5, 10, 20):
        c = bp / 10000
        assert np.allclose((xp * (1 - c) / (ep * (1 + c)) - 1) * 100,
                           trades[f"net_{bp}bp_pct"], rtol=0, atol=1e-10)

    # Verify all chosen entries use information completed before order entry.
    observed = pd.to_datetime(trades.last_observed_1m, utc=True)
    assert ((entry.dt.tz_convert("UTC") - observed).dt.total_seconds() >= 60).all()
    assert (exit_ > entry).all()
    assert np.array_equal(entry.dt.date.to_numpy(), exit_.dt.date.to_numpy())

    spec = importlib.util.spec_from_file_location("production_close_rsi_audit", ROOT / "code/study.py")
    production = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(production)
    prefix_checks = []
    # Production functions must give identical prior decisions when the future
    # is absent or its prices are changed dramatically.
    for day in ("2026-01-02", "2026-06-30", "2026-10-30"):
        cutoff = pd.Timestamp(day + " 15:30", tz=ET)
        got, _ = production.signals(data[data.index < cutoff])
        subset = got[got.day <= day].set_index(["day", "tf"]).sort_index()
        reference = actual.loc[actual.index.get_level_values("day") <= day]
        assert subset.index.equals(reference.index)
        assert np.allclose(subset.rsi, reference.rsi, rtol=0, atol=1e-10)
        prefix_checks.append({"cutoff": str(cutoff), "rows": len(subset), "pass": True})
    cutoff = pd.Timestamp("2026-06-30 15:30", tz=ET)
    changed = data.copy()
    mask = changed.index >= cutoff
    changed.loc[mask, ["Open", "High", "Low", "Close", "VWAP"]] *= 7
    changed_signals, _ = production.signals(changed)
    subset = changed_signals[changed_signals.day <= "2026-06-30"].set_index(["day", "tf"]).sort_index()
    reference = actual.loc[actual.index.get_level_values("day") <= "2026-06-30"]
    assert np.allclose(subset.rsi, reference.rsi, rtol=0, atol=1e-10)
    assert np.array_equal(subset.signal.to_numpy(), reference.signal.to_numpy())

    # Known-direction and oscillating fixtures check the independent recurrence.
    assert loop_rsi(np.arange(50, dtype=float))[-1] == 100
    assert loop_rsi(-np.arange(50, dtype=float))[-1] == 0
    test = np.r_[np.arange(20), np.arange(20)[::-1], np.arange(20)]
    for n in (15, 25, 45):
        assert np.allclose(loop_rsi(test)[:n], loop_rsi(test[:n]), equal_nan=True)

    report = {"pass": True, "signal_rows": len(actual), "independent_signal_mismatches": 0,
              "max_rsi_absolute_error": error, "trade_rows": len(trades),
              "max_return_absolute_error_pct": ret_error, "cost_formulas_checked": [0, 5, 10, 20],
              "prefix_invariance_checks": prefix_checks,
              "future_price_mutation_factor": 7, "future_mutation_cutoff": str(cutoff),
              "future_mutation_prior_signal_mismatches": 0,
              "three_minute_source_bar": "15:27~15:30 ET; anchored09:30",
              "time_matched_auxiliary": "1m42 Wilder42, nominal42min compared with3m14",
              "signal_counts_by_timeframe": actual[actual.signal].reset_index().groupby("tf").size().to_dict(),
              "notes": ["2025 used only to initialize indicator state.",
                        "Last observed minute15:29 is complete at15:30; 3m source15:27 and5m source15:25.",
                        "No test establishes executability from bar OHLC; quote audit is separate."]}
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
