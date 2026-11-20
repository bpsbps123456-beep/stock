"""SOXX market gates and fixed-basket / cash controls on the 2026 desk ledger."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RECIPES = HERE.parent / "finviz_recipes_20261120"
HISTORY = HERE.parent / "bottom_zone_20261111/news/v2/prices/SOXX_full.csv"
sys.path.insert(0, str(RECIPES))
from research import TimingEngine, describe, load_inputs, market_gate, recipes
from compare import close_terminated, switched_targets


PERIODS = (25, 50, 75, 100, 125, 150, 200)
CONFIRM_PERIODS = (75, 100, 125)


def soxx_closes(src):
    """Use the split-adjusted Close (no dividend adjustment) of the frozen desk.

    Its June 2025 start is insufficient for SMA200 at the January 2026 auction.
    Extend the left edge using a committed price history in exactly the same units.
    """
    desk = src.book.cls.SOXX.loc[:src.signals[-1]]
    warmup = pd.read_csv(HISTORY, usecols=["date", "Close"], index_col="date").Close.loc["2025-01-01":]
    overlap = desk.index.intersection(warmup.index)
    if len(overlap) < 200:
        raise ValueError("Insufficient overlapping SOXX history to validate units")
    np.testing.assert_array_equal(desk.loc[overlap], warmup.loc[overlap])
    close = pd.concat([warmup.loc[warmup.index < desk.index[0]], desk])
    if close.index.has_duplicates or not close.index.is_monotonic_increasing or not close.gt(0).all():
        raise ValueError("Invalid SOXX close history")
    return close


def load_gates(src, lag=0):
    """Execution-day masks, True for the desk. lag=1 uses one extra trading session."""
    if lag not in (0, 1):
        raise ValueError("Supported gate lags are zero or one extra trading session")
    close = soxx_closes(src)
    rows = close.index.searchsorted(src.days, side="left") - 1
    np.testing.assert_array_equal(close.index[rows], src.signals)
    if rows[0] - lag < max(PERIODS) - 1:
        raise ValueError("Insufficient SOXX warmup")
    signals = {}
    for n in PERIODS:
        line = close.rolling(n, min_periods=n).mean()
        valid = line.notna()
        above = close.ge(line) & valid
        below = close.lt(line) & valid
        signals[f"sma{n}"] = above.where(valid)
        if n in CONFIRM_PERIODS:
            # Unresolved states stay missing until an actual trigger in warmup.
            for label, on, off in [
                (f"confirm2_{n}", above & above.shift(fill_value=False), below & below.shift(fill_value=False)),
                (f"band1_{n}", close.ge(line * 1.01), close.lt(line * .99)),
            ]:
                state = pd.Series(np.nan, index=close.index)
                state.loc[on] = 1.
                state.loc[off] = 0.
                signals[label] = state.ffill()
    result = {}
    for label, state in signals.items():
        known = state.iloc[rows - lag]
        if known.isna().any():
            raise ValueError(f"Unresolved SOXX gate in the study period: {label}")
        result[label] = known.to_numpy(dtype=bool)
    if not lag:
        np.testing.assert_array_equal(result["sma100"], market_gate(src))
    return result


def gate_counts(src, gate):
    dates = np.array(src.days)
    exits = ~gate & np.r_[True, gate[:-1]]
    return dict(off_sessions=int((~gate).sum()), off_spells=int(exits.sum()),
                off_h1=int((~gate & (dates < "2026-07-01")).sum()),
                off_h2=int((~gate & (dates >= "2026-07-01")).sum()),
                off_q4=int((~gate & (dates >= "2026-10-01")).sum()))


def gate_forward(src, engine, gates, monthly, normal):
    """Hold basket rules fixed and select only the market rule with the locked criterion."""
    folds = [("H1", "H2", "2026-01-01", "2026-06-30", "2026-11-18"),
             ("Q1", "Q2", "2026-01-01", "2026-03-31", "2026-06-30"),
             ("Q2", "Q3", "2026-04-01", "2026-06-30", "2026-09-30"),
             ("Q3", "Q4_partial", "2026-07-01", "2026-09-30", "2026-11-18")]

    def metrics(path, begin, end):
        wealth = path.krw10.to_numpy()
        anchor = wealth[begin - 1] if begin else 1.
        values = np.r_[anchor, wealth[begin:end + 1]]
        return values[-1] / anchor - 1, np.min(values / np.maximum.accumulate(values)) - 1

    days = pd.Index(src.days)
    baseline = normal["sma100"]
    order = ["sma100"] + [k for k in gates if k != "sma100"]
    chain = gates["sma100"].copy()
    rows = []
    for train, test, first, training_end, test_end in folds:
        start = days.searchsorted(first)
        cut = days.searchsorted(training_end, side="right")
        end = days.searchsorted(test_end, side="right") - 1
        _, base_dd = metrics(baseline, start, cut - 1)
        chosen, best = "sma100", -np.inf
        for name in order:
            score, dd = metrics(normal[name], start, cut - 1)
            if dd >= base_dd - .02 - 1e-12 and score > best + 1e-12:
                chosen, best = name, score
        gate = gates["sma100"].copy()
        gate[cut:end + 1] = gates[chosen][cut:end + 1]
        plan = switched_targets(src.days, src.opn, engine.desk, monthly, gate)
        tested = engine._price(plan, detail=True, stress=False)
        np.testing.assert_allclose(tested.krw10.iloc[:cut], baseline.krw10.iloc[:cut], atol=1e-12)
        actual_ret, actual_dd = metrics(tested, cut, end)
        base_ret, base_dd = metrics(baseline, cut, end)
        if train != "H1":
            chain[cut:end + 1] = gates[chosen][cut:end + 1]
        rows.append(dict(train=train, test=test, selected=chosen, test_first=src.days[cut],
                         test_last=src.days[end], selected_ret=actual_ret, base_ret=base_ret,
                         selected_dd=actual_dd, base_dd=base_dd,
                         off_sessions=int((~gate[cut:end + 1]).sum())))
    chain_plan = switched_targets(src.days, src.opn, engine.desk, monthly, chain)
    chained = engine._price(chain_plan, detail=True, stress=True)
    metrics_table = pd.DataFrame(rows)
    metrics_table.to_csv(HERE / "gate_forward.csv", index=False, float_format="%.12g")
    columns = ["krw", "krw10", "krw25", "turn", "exposure"]
    daily = pd.concat({"quarterly_gates": chained[columns], "sma100": baseline[columns]},
                      names=["rule", "date"])
    daily.to_csv(HERE / "gate_forward_daily.csv", float_format="%.12g")


def run():
    src = load_inputs()
    gates = load_gates(src)
    stale = load_gates(src, lag=1)
    for key, values in gates.items():
        np.testing.assert_array_equal(values[:-1], stale[key][1:])
    close = soxx_closes(src)
    daily = pd.DataFrame({"trade_open": src.days, "signal_close": src.signals,
                          "soxx_close": close.loc[src.signals].to_numpy()})
    for n in PERIODS:
        daily[f"line{n}"] = close.rolling(n).mean().loc[src.signals].to_numpy()
    for key, values in gates.items():
        daily[key] = values.astype(int)
    daily.to_csv(HERE / "gate_signals.csv", index=False, float_format="%.15g")

    engine = TimingEngine(src, gates["sma100"])
    current = recipes(src)["balance"]
    monthly = engine.targets(current, "monthly1")
    shifted = {i + 1: monthly[i] for i in engine.reviews["monthly1"] if i + 1 < len(src.days)}
    delayed = close_terminated(shifted, src.opn, src.tradable)
    rows, fixed = [], {}
    prior = pd.read_csv(RECIPES / "selected_daily.csv", index_col=["key", "date"])
    for name in gates:
        for timing, gate, plan, allocation in [
            ("normal", gates[name], monthly, "balance"),
            ("gate_only_lag1", stale[name], monthly, "balance"),
            ("gate_and_basket_lag1", stale[name], delayed, "balance"),
            ("normal", gates[name], {}, "USD_cash"),
            ("gate_only_lag1", stale[name], {}, "USD_cash"),
        ]:
            combined = switched_targets(src.days, src.opn, engine.desk, plan, gate)
            frame = engine._price(combined, detail=False, stress=True)
            if timing == "normal" and allocation == "balance":
                fixed[name] = frame
            if name == "sma100" and timing == "normal" and allocation == "balance":
                reference = prior.loc["balance/monthly1"]
                assert reference.index.equals(frame.index)
                np.testing.assert_allclose(frame[["krw", "krw10", "krw25", "turn"]],
                                           reference[["krw", "krw10", "krw25", "turn"]], rtol=2e-10)
            off_exposure = frame.exposure[~gate]
            rows.append(dict(gate=name, off_allocation=allocation, timing=timing,
                             **gate_counts(src, gate),
                             off_mean_exposure=off_exposure.mean(), **describe(frame)))
    results = pd.DataFrame(rows)
    results.to_csv(HERE / "gate_results.csv", index=False, float_format="%.12g")
    gate_forward(src, engine, gates, monthly, fixed)
    return results


if __name__ == "__main__":
    metrics = run()
    normal = metrics[(metrics.timing == "normal") & (metrics.off_allocation == "balance")]
    print(normal[["gate", "off_sessions", "off_spells", "year_ret", "h2_ret", "h2_dd", "oct_ret"]]
          .to_string(index=False))
