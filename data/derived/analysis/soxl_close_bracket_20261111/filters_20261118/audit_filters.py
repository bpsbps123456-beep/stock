"""Cross-check filter ledgers against an independent chronological first-hit search.

All allowed closing entries use the same bracket. Compute the hypothetical exit
of each possible entry from the price tape, then replay cash and shares on the
filtered entry dates. The production simulator supplies the comparison ledger
only for secondary component checks whose stored reports contain summaries.
"""
from __future__ import annotations

import bisect
import importlib.util
import io
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = HERE / "results"
COSTS = (0, 5, 10, 20)
FILTERS = ("baseline", "sector_sma20", "above_vwap", "late_up", "rsi_turn", "down5", "bb1")
JAN, JUL, SEP = "2026-01-02", "2026-07-01", "2026-09-01"
TZ = "America/New_York"
CHECKS = importlib.util.spec_from_file_location("bracket_reference", BASE / "audit.py")
REFERENCE = importlib.util.module_from_spec(CHECKS)
CHECKS.loader.exec_module(REFERENCE)


@dataclass(frozen=True)
class Session:
    day: str
    close_at: pd.Timestamp
    close_price: float


@dataclass(frozen=True)
class Exit:
    # A (timestamp, phase) pair orders the closing auction before a bar labelled 16:00.
    key: tuple
    price: float
    reason: str


def build_lattice(minute, daily, start, end, window, gap_tp):
    """Compute hypothetical first exits, independent of any filter identities."""
    days = xc.get_calendar("XNYS").schedule.loc[start:end]
    keys, observations, sessions = [], [], []
    for d, session in days.iterrows():
        day = d.strftime("%Y-%m-%d")
        opened, closed = (session[col].tz_convert(TZ) for col in ("open", "close"))
        price = float(daily.loc[day, "Close"])
        sessions.append(Session(day, closed, price))
        eligible = minute.loc[day]
        regular = eligible[(eligible.index >= opened) & (eligible.index < closed)]
        assert regular.index.equals(pd.date_range(opened, closed, inclusive="left", freq="min"))
        start_time = opened if window == "RTH" else opened.normalize() + pd.Timedelta(hours=4)
        pre_close = eligible[(eligible.index >= start_time) & (eligible.index < closed)]
        post_close = eligible[(eligible.index >= closed) & (eligible.index.hour < 20)]
        pieces = [pre_close, None]
        if window == "EXT" and day != end:
            pieces.append(post_close)
        for piece in pieces:
            if piece is None:
                keys.append((closed, 0))
                observations.append((price, price, price))
            else:
                keys.extend((ts, 1) for ts in piece.index)
                observations.extend(piece[["Open", "High", "Low"]].itertuples(index=False, name=None))
    assert all(a < b for a, b in zip(keys, keys[1:]))
    lattice = []
    for session in sessions:
        stop, target = .97 * session.close_price, 1.02 * session.close_price
        # A new position cannot observe any intraminute price from the auction minute.
        begin = bisect.bisect_right(keys, (session.close_at, 1))
        first = None
        for i in range(begin, len(keys)):
            hit = REFERENCE.independent_touch(*observations[i], stop, target, "stop", gap_tp)
            if hit is not None:
                reason, xp, _ = hit
                first = Exit(keys[i], xp, reason)
                break
        lattice.append(first)
    return sessions, lattice


def replay(sessions, lattice, permissions, costs=COSTS, cutoff_minutes=10):
    """Track prior open position even on dates whose entry filter is false."""
    rates = np.array(costs) / 10000.
    cash = np.full(len(rates), 100.)
    shares = np.zeros(len(rates))
    position = None
    trades, equity, held = [], [], []
    for i, (day, allow) in enumerate(zip(sessions, permissions, strict=True)):
        target = lattice[position] if position is not None else None
        cutoff = day.close_at - pd.Timedelta(minutes=cutoff_minutes)
        free_at_cutoff = position is None or (target is not None and target.key[0] < cutoff)
        if target is not None and target.key <= (day.close_at, 0):
            cash = shares * target.price * (1 - rates)
            shares = np.zeros_like(rates)
            trades.append((sessions[position], target))
            position = None
        if allow and free_at_cutoff:
            assert position is None
            position = i
            shares = cash / (day.close_price * (1 + rates))
            cash = np.zeros_like(rates)
        equity.append(cash + shares * day.close_price)
        held.append(position is not None)
    # Prior-session after-hours sales settle before the next snapshot above.
    # The last date ends at the regular close, so censored trades have no exit.
    return np.asarray(equity), np.asarray(held), trades, sessions[position] if position is not None else None


def check_close(a, b, description):
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-9, err_msg=description)


def check_run(model, flags, observed_t, observed_e):
    sessions, lattice = model
    equity, held, trades, tail = replay(sessions, lattice, flags)
    assert observed_e.day.tolist() == [x.day for x in sessions]
    np.testing.assert_array_equal(held, observed_e.position.astype(bool))
    for k, bp in enumerate(COSTS):
        check_close(equity[:, k], observed_e[f"nav_{bp}bp"], f"NAV {bp}bp")
    t = observed_t[observed_t.reason.ne("open_censored")]
    assert len(t) == len(trades)
    for ref, (_, obs) in zip(trades, t.iterrows(), strict=True):
        entry, exit_ = ref
        assert pd.Timestamp(obs.entry_time) == entry.close_at
        assert pd.Timestamp(obs.exit_time) == exit_.key[0]
        check_close((entry.close_price, exit_.price), (obs.entry_price, obs.exit_price), "fills")
        assert ("target" in exit_.reason) == obs.reason.startswith("take_profit")
        for bp in COSTS:
            rate = bp / 10000.
            check_close(obs[f"net_{bp}bp_pct"], 100*(exit_.price*(1-rate)/(entry.close_price*(1+rate))-1), "trade return")
    open_trades = observed_t[observed_t.reason.eq("open_censored")]
    assert len(open_trades) == int(tail is not None)
    if tail:
        assert pd.Timestamp(open_trades.iloc[0].entry_time) == tail.close_at
        check_close(open_trades.iloc[0].mark_price, sessions[-1].close_price, "terminal mark")
    return {"closed": len(trades), "marked_open": int(tail is not None), "daily_marks": len(held),
            "end_nav_10bp": float(equity[-1, COSTS.index(10)])}


def timing_diagnostic(model, signal):
    """Apply every temporal rotation to one entry mask on the same July+ tape.

    The rotated masks have equal eligible-day counts. Realized exposure can
    vary when a trade is still open at the next closing-order cutoff.
    """
    sessions, exits = model
    returns, trade_counts = [], []
    for k in range(len(sessions)):
        nav, _, trades, _ = replay(sessions, exits, np.roll(signal, k), (10,))
        returns.append(float(nav[-1, 0]-100.))
        trade_counts.append(len(trades))
    alternatives = np.asarray(returns[1:])
    tolerance = 1e-8
    return {"entry_mask_days": int(np.sum(signal)), "closed": trade_counts[0],
            "actual_return_pct": returns[0], "nonzero_shifts": len(alternatives),
            "nonzero_shift_median_pct": float(np.median(alternatives)),
            "nonzero_shift_closed_range": [min(trade_counts[1:]), max(trade_counts[1:])],
            "fraction_shifts_at_least_actual": float(np.mean(alternatives >= returns[0]-tolerance)),
            "rotation_scores": returns}


def load_flags(frame, column, sessions):
    if column == "baseline":
        if column in frame:
            assert frame[column].eq(True).all()
        return np.ones(len(sessions), dtype=bool)
    raw = frame.loc[[s.day for s in sessions], column]
    if column == "bb1":
        raw = raw.fillna(False)
    assert set(raw.unique()) <= {True, False, 1, 0}, (column, raw.unique())
    return raw.to_numpy(dtype=bool)


def check_features(features, earlier, minute):
    """Reconstruct each trading-day decision using only timestamp-eligible inputs."""
    cache = HERE / "cache"
    provenance = json.loads((OUT / "input_audit.json").read_text(encoding="utf-8"))
    warm_path = BASE.parent / provenance["minute_sources"][2]["path"]
    schedule = xc.get_calendar("XNYS").schedule.loc[provenance["warmup_only"][0]:features.index[-1]]
    warm = pd.read_parquet(warm_path, columns=["Close"]).loc[schedule.index[0].date().isoformat():"2025-12-31"]
    closes = pd.concat([warm.Close, minute.Close])
    # Use an independently seeded exponentially weighted series over RTH closes.
    regular = pd.concat([closes.loc[row.open:row.close-pd.Timedelta(minutes=1)]
                         for _, row in schedule.iterrows()])
    delta = regular.diff().dropna()
    def smooth(values):
        seeded = values.iloc[13:].copy()
        seeded.iloc[0] = values.iloc[:14].mean()
        return seeded.ewm(alpha=1/14, adjust=False).mean()
    gain, loss = smooth(delta.clip(lower=0)), smooth((-delta).clip(lower=0))
    rsi = (100 - 100 / (1 + gain/loss)).where(loss.gt(0), 100)
    rising = rsi.le(30) & rsi.gt(rsi.shift())
    daily = pd.read_parquet(cache / "SOXL_warm_daily.parquet").Close
    sector = pd.read_parquet(cache / "SOXX_warm_daily.parquet").Close
    def intraday_inputs(opened, cutoff):
        tape = minute.loc[opened:cutoff-pd.Timedelta(minutes=1)]
        price, previous_price = float(tape.Close.iloc[-1]), float(tape.Close.iloc[-31])
        vwap = float(np.average(tape.VWAP, weights=tape.Volume))
        prior = daily[daily.index < opened.normalize()].iloc[-1]
        # A 15:29-labelled candle is complete at 15:30, the first decision time.
        first_label = cutoff.normalize()+pd.Timedelta(hours=15, minutes=29)
        events = rising.loc[first_label:tape.index[-1]]
        first = events[events].first_valid_index()
        return (price, vwap, previous_price, prior), first

    def intraday_flags(inputs, first):
        price, vwap, previous_price, prior = inputs
        return [price >= vwap, price > previous_price, first is not None, price <= prior*.95]

    assert earlier.index.equals(features.index)
    for day, data in features.iterrows():
        opened = schedule.loc[day, "open"].tz_convert(TZ)
        cutoff = schedule.loc[day, "close"].tz_convert(TZ)-pd.Timedelta(minutes=10)
        assert pd.Timestamp(data.decision_et) == cutoff
        assert pd.Timestamp(data.latest_bar_start_et) == cutoff-pd.Timedelta(minutes=1)
        values, first = intraday_inputs(opened, cutoff)
        early_values, early_first = intraday_inputs(opened, cutoff-pd.Timedelta(minutes=5))
        early_flags = intraday_flags(early_values, early_first)
        np.testing.assert_array_equal(earlier.loc[day, list(FILTERS[2:-1])].to_numpy(), early_flags)
        recent_sector = sector[sector.index < opened.normalize()].tail(20)
        assert len(recent_sector) == 20
        if first is None:
            assert pd.isna(data.first_rsi_turn_et)
        else:
            assert pd.Timestamp(data.first_rsi_turn_et) == first + pd.Timedelta(minutes=1)
        expected = [recent_sector.iloc[-1] >= recent_sector.mean(), *intraday_flags(values, first)]
        np.testing.assert_array_equal(data[list(FILTERS[1:-1])].to_numpy(dtype=bool), expected)
        check_close([*values, recent_sector.iloc[-1], recent_sector.mean()],
                    data[["price_before_cutoff", "regular_vwap_before_cutoff", "price_30min_earlier",
                          "previous_soxl_close", "previous_soxx_close", "previous_soxx_sma20"]].astype(float),
                    "inputs "+day)
    # The historical signal panel records its market/publication lags. The
    # Nov 12 published panel supplies the extra observation with both raw legs.
    def published_csv(name, columns):
        source = next(s for s in provenance["bb1"]["sources"] if Path(s["path"]).name == name)
        raw = subprocess.check_output(["git", "cat-file", "blob", source["git_blob"]], cwd=HERE)
        return pd.read_csv(io.BytesIO(raw), index_col=columns[0], usecols=columns)

    history = published_csv("signal_panel.csv", ["date", "source_date", "source_known",
                                                "source_published", "dd60", "T5_P"]).reindex(features.index)
    known = history.source_published.eq(True) & history.source_known.le(history.index.to_series())
    known &= history.source_date.lt(history.index.to_series()) & history.dd60.notna() & history.T5_P.notna()
    bb = (history.dd60.le(-.20) & history.T5_P.ge(80)).astype(float).where(known)
    last = "2026-11-12"
    extra = published_csv("now_table.csv", ["관찰일", "수급공개", "수급원거래일",
                                           "낙폭60", "SOXL비중백분위"]).loc[last]
    assert extra["수급공개"] and extra["수급원거래일"] < last
    bb.loc[last] = float(extra["낙폭60"] <= -.20 and extra["SOXL비중백분위"] >= 80)
    pd.testing.assert_series_equal(bb, features.bb1, check_names=False)
    return {"sessions": len(features), "decision_inputs_checked_per_day": 6,
            "candidate_flags_checked": len(FILTERS)-1, "bb1_unavailable": int(bb.isna().sum()),
            "bb1_unavailable_since_july": int(bb.loc[JUL:].isna().sum())}


def check_selection(marks, trades):
    outcomes = []
    for start, end, forward, minimum in (
            (JAN, "2026-06-30", JUL, 20), (JUL, "2026-08-31", SEP, 10)):
        train = marks[marks.run_start.eq(start) & marks.day.eq(end)]
        test = marks[marks.run_start.eq(forward) & marks.day.eq(marks.day.max())]
        closed = trades[trades.run_start.eq(start) & trades.exit_day.le(end)]
        n = closed.groupby("filter").size().to_dict()
        training_returns = train.set_index("filter").nav_10bp - 100
        eligible = [c for c in FILTERS[1:-1] if n.get(c, 0) >= minimum and training_returns[c] > 0]
        best = max(eligible, key=training_returns.get) if eligible else None
        test_returns = test.set_index("filter").nav_10bp - 100
        outcomes.append({"fit_start": start, "fit_end": end, "minimum_closed": minimum,
                         "selected": best or "cash",
                         "eligible_candidates": eligible,
                         "forward_start": forward,
                         "forward_return_pct": float(test_returns[best]) if best else 0.,
                         "forward_unfiltered_pct": float(test_returns.baseline)})
    return outcomes


def check_earlier_cutoff(model, flags):
    metrics = pd.read_csv(OUT / "earlier_cutoff_metrics.csv").set_index("filter")
    sessions, lattice = model
    verified = {}
    for name in FILTERS[2:-1]:
        observed = metrics.loc[name]
        allowed = load_flags(flags, name, sessions)
        assert observed.cutoff_minutes == 15
        verified[name] = check_summary(model, allowed, observed, cutoff_minutes=15)
    return {"flags_verified_sessions": len(flags), "cutoff_minutes": 15,
            "model_sessions": len(sessions), "models": verified}


def check_summary(model, allowed, observed, cutoff_minutes=10):
    sessions, lattice = model
    equity, held, trades, tail = replay(sessions, lattice, allowed, cutoff_minutes=cutoff_minutes)
    returns = np.array([100*(exit_.price*.999/(entry.close_price*1.001)-1)
                        for entry, exit_ in trades])
    assert observed.sessions == len(sessions)
    assert observed.signals == allowed.sum() and observed.closed == len(trades)
    assert observed.open_positions == held[-1] == (tail is not None)
    assert observed.entries == len(trades) + int(held[-1])
    for k, bp in enumerate(COSTS):
        check_close(equity[-1, k]-100, observed[f"net{bp}"], "ending NAV")
    path = np.r_[100., equity[:, COSTS.index(10)]]
    drawdown = 100*(path/np.maximum.accumulate(path)-1).min()
    no_best = path[-1]/(1+returns.max()/100)-100
    check_close([returns.min(), returns.max(), 100*np.mean(returns > 0), drawdown, no_best],
                observed[["worst10", "best10", "win_pct", "mdd10", "without_best10"]].astype(float),
                "summary metrics")
    return {"closed": len(trades), "end_nav_10bp": float(path[-1])}


def check_components(features, minute, daily):
    matched = pd.read_csv(OUT / "belt_components.csv").assign(window="RTH")
    full = pd.read_csv(OUT / "drawdown_only.csv").assign(filter="drawdown_only")
    matched_end = "2026-11-11"
    assert set(zip(matched.start, matched.end, matched["filter"], matched.gap_tp)) == {
        (JUL, matched_end, name, gap)
        for name in ("drawdown_only", "bb1") for gap in ("open", "target")}
    assert set(zip(full.start, full.end, full.window, full.gap_tp)) == {
        (start, features.index[-1], window, gap) for start, window, gap in
        ((JUL, "RTH", "open"), (SEP, "RTH", "open"), (JUL, "EXT", "open"), (JUL, "RTH", "target"))}
    assert len(matched) == len(full) == 4
    dates = features.index[features.index >= JUL]
    sector = pd.read_parquet(HERE / "cache/SOXX_warm_daily.parquet").Close
    sector.index = sector.index.strftime("%Y-%m-%d")
    asset_drawdown = {}
    for day in dates:
        previous = sector[sector.index < day].tail(60)
        assert len(previous) == 60
        asset_drawdown[day] = previous.iloc[-1] / max(previous) - 1
    deep = pd.Series(asset_drawdown).le(-.20)
    reported_deep = features.loc[dates, "bb1_prior_soxx_drawdown60"].le(-.20)
    np.testing.assert_array_equal(deep, reported_deep)
    known = features.bb1.notna()  # Checked against the original publication fields by check_features.
    # Compare stored percentiles to the contemporaneous source instead of using
    # the drawdown-and-flow signal to infer the flow leg.
    provenance = json.loads((OUT / "input_audit.json").read_text(encoding="utf-8"))
    panel_blob = next(s["git_blob"] for s in provenance["bb1"]["sources"]
                      if Path(s["path"]).name == "signal_panel.csv")
    raw = subprocess.check_output(["git", "cat-file", "blob", panel_blob], cwd=HERE)
    source = pd.read_csv(io.BytesIO(raw), usecols=["date", "T5_P"], index_col="date").loc[JUL:matched_end]
    check_close(source.T5_P, features.loc[source.index, "bb1_flow_percentile"], "flow percentile")
    liquid = source.T5_P.ge(80)
    schedule = xc.get_calendar("XNYS").schedule
    prices = daily.Close.copy()
    prices.index = prices.index.strftime("%Y-%m-%d")
    # These component reports have no stored per-trade ledger. Recreate the
    # comparison table using the production engine and verify all fills against
    # the independent brute-force tape, settlement, cash and position replay.
    spec = importlib.util.spec_from_file_location("model_under_audit", BASE / "study.py")
    production = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(production)
    result = []
    for _, observed in pd.concat([matched, full], ignore_index=True).iterrows():
        start, end, window, gap = (observed[k] for k in ("start", "end", "window", "gap_tp"))
        model = build_lattice(minute, daily, start, end, window, gap)
        flags = deep.loc[start:end].copy()
        if end == matched_end:
            assert observed.flow_known_days == known.loc[start:end].sum()
            flags &= known.loc[start:end]
        if observed["filter"] == "bb1":
            flags &= liquid.loc[start:end]
            np.testing.assert_array_equal(flags, features.loc[start:end, "bb1"].eq(1))
        allowed = flags.to_numpy()
        summary = check_summary(model, allowed, observed)
        trades, equity, _ = production.simulate(minute, prices, schedule, run_start=start, run_end=end,
                                                window=window, gap_tp=gap, entry_days=set(flags.index[allowed]))
        fills = check_run(model, allowed, trades, equity)
        result.append({"filter": observed["filter"], "start": start, "end": end, "window": window,
                       "gap_tp": gap, "eligible": int(flags.sum()), **summary,
                       "daily_marks": fills["daily_marks"], "marked_open": fills["marked_open"]})
    return result


def main():
    minute = pd.read_parquet(HERE / "cache/minute.parquet")
    daily = pd.read_parquet(HERE / "cache/daily.parquet")
    marks = pd.read_csv(OUT / "equity.csv")
    trades = pd.read_csv(OUT / "trades.csv")
    features = pd.read_csv(OUT / "features.csv").set_index("day")
    earlier = pd.read_csv(OUT / "earlier_cutoff_features.csv", index_col="day")
    end = marks.day.max()
    result = {"new_last_day": end, "models": [], "rotations": {}, "old_baseline": [],
              "pre_cutoff_inputs": check_features(features, earlier, minute)}
    result["posthoc_belt_components"] = check_components(features, minute, daily)
    keys = ["window", "gap_tp", "run_start"]
    for (window, gap, start), group in marks.groupby(keys):
        model = build_lattice(minute, daily, start, end, window, gap)
        if (start, window, gap) == (JUL, "RTH", "open"):
            result["earlier_cutoff"] = check_earlier_cutoff(model, earlier)
        subset = trades[trades.window.eq(window) & trades.gap_tp.eq(gap) & trades.run_start.eq(start)]
        for candidate, observed in group.groupby("filter"):
            flags = load_flags(features, candidate, model[0])
            s = check_run(model, flags, subset[subset["filter"].eq(candidate)], observed)
            result["models"].append({"window":window,"gap_tp":gap,"run_start":start,"filter":candidate,**s})
            if window == "RTH" and gap == "open" and start == JUL and candidate != "baseline":
                result["rotations"][candidate] = timing_diagnostic(model, flags)
    old_equity = pd.read_csv(BASE / "results/equity.csv")
    old_trades = pd.read_csv(BASE / "results/trades.csv")
    for start in (JAN, JUL):
        args = dict(window="RTH", policy="stop_first", gap_tp="open", run_start=start)
        def select(frame):
            return frame[np.logical_and.reduce([frame[field].eq(value) for field, value in args.items()])]
        # Legacy files use the split-adjusted scale observed when they were frozen.
        # Compare only return/NAV if a later split changed pure price units.
        old_prices = pd.read_csv(BASE / "results/daily_prices.csv", index_col=0)["Close"]
        old_prices.index = old_prices.index.str[:10]
        scale = float(daily.loc[start,"Close"]) / float(old_prices.loc[start])
        t = select(old_trades).copy()
        for field in ("entry_price", "exit_price", "mark_price"):
            t[field] *= scale
        model = build_lattice(minute, daily, start, "2026-11-10", "RTH", "open")
        result["old_baseline"].append({"start":start,"price_scale":scale,
                                      **check_run(model, np.ones(len(model[0]),bool), t, select(old_equity))})
    expected = {(start, window, gap, candidate)
                for start, window, gap in ((JAN, "RTH", "open"), (JUL, "RTH", "open"),
                                           (SEP, "RTH", "open"), (JUL, "EXT", "open"),
                                           (JUL, "RTH", "target")) for candidate in FILTERS}
    actual = {(m["run_start"], m["window"], m["gap_tp"], m["filter"]) for m in result["models"]}
    assert expected == actual
    result["all_costs"] = list(COSTS)
    rth_marks = marks[marks.window.eq("RTH") & marks.gap_tp.eq("open")]
    rth_trades = trades[trades.window.eq("RTH") & trades.gap_tp.eq("open")]
    result["selection"] = check_selection(rth_marks, rth_trades)
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    for row, field in zip(result["selection"], ["h1_pick", "summer_pick"], strict=True):
        assert row["selected"] == (summary[field] or "cash")
    result["close_bracket_boundary_checks"] = REFERENCE.synthetic_checks()
    result["rotation_interpretation"] = "Descriptive timing diagnostic. One nonzero shift per session on the July fresh-cash path. Each shift preserves the number of allowed entry dates. Realized exposure and trade counts may differ. This is not an independent forward test."
    result["pass"] = True
    (OUT / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("Passed", len(result["models"]), "filtered models", flush=True)


if __name__ == "__main__":
    main()
