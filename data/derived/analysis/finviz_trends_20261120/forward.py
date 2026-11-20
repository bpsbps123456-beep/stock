"""Reprice selected rules in one continuous share-and-cash ledger across selection dates."""
from pathlib import Path

import numpy as np
import pandas as pd

from gates import HERE, RECIPES, TimingEngine, close_terminated, load_gates, switched_targets


WINDOWS = {"YTD": ("2026-01-01", "2026-11-18"), "H1": ("2026-01-01", "2026-06-30"),
           "H2": ("2026-07-01", "2026-11-18"), "Q1": ("2026-01-01", "2026-03-31"),
           "Q2": ("2026-04-01", "2026-06-30"), "Q3": ("2026-07-01", "2026-09-30"),
           "Q4": ("2026-10-01", "2026-11-18")}


def selection_paths(src, grid, choices, fixed):
    base = (grid.baseline, "sma100")
    if fixed.get("baseline") != base:
        raise ValueError("The baseline must be the existing monthly FINVIZ and SOXX100")
    paths = {name: [(0, pair)] for name, pair in fixed.items()}
    starts = {name: src.days.index(WINDOWS[name][0]) for name in ("Q2", "Q3", "Q4")}
    folds = [(starts["Q2"], choices["q1"]), (starts["Q3"], choices["q2"]), (starts["Q4"], choices["q3"])]
    for label, changes in (("anchored", [(starts["Q3"], choices["h1"])]), ("rolling", folds)):
        for option in ("joint", "screen_only", "gate_only"):
            name = (label if label == "rolling" else "anchored_h1") if option == "joint" else f"{label}_{option}"
            if name in paths:
                raise ValueError(f"Reserved walk-forward label was supplied as a fixed comparison: {name}")
            selected = [(0, base)]
            for first, (screen, gate) in changes:
                pair = (base[0] if option == "gate_only" else screen, base[1] if option == "screen_only" else gate)
                if pair != selected[-1][1]:
                    selected.append((first, pair))
            paths[name] = selected
    return paths


class SelectedReplay:
    def __init__(self, src, grid):
        self.src = src
        self.grid = grid
        self.gates = [load_gates(src), load_gates(src, lag=1)]
        self.engine = TimingEngine(src, self.gates[0]["sma100"])
        self.screens = {}

    def screen_plan(self, changes, lag):
        """Choose on the dated review, deduplicate the issuer, then shift the fill if requested."""
        key = (tuple(changes), lag)
        if key in self.screens:
            return self.screens[key]
        days = len(self.src.days)
        ranks = [[] for _ in range(days)]
        reviews = []
        for k, (first, screen) in enumerate(changes):
            end = changes[k + 1][0] if k + 1 < len(changes) else days
            schedule = self.grid.screens.at[screen, "schedule"]
            # Adopt the new stock rule at its first available auction even if it is midweek.
            dates = sorted({first, *(j for j in self.grid.reviews[schedule] if first <= j < end)})
            published = self.grid.ranks(screen)
            for j in dates:
                line = published[j]
                ranks[j] = line[line >= 0].tolist()
            reviews.extend(dates)
        self.engine.reviews["selected_walk"] = reviews
        on_time = self.engine.targets(ranks, "selected_walk")
        if lag:
            arrivals = {j + 1: on_time[j] for j in reviews if j + 1 < days}
            plan = close_terminated(arrivals, self.src.opn, self.src.tradable)
        else:
            plan = on_time
        self.screens[key] = plan
        return plan

    def evaluate(self, changes, lag):
        """At a selection boundary, the prior actual holdings earn the next opening gap."""
        screens = []
        gate = np.empty(len(self.src.days), dtype=bool)
        for k, (first, (screen, name)) in enumerate(changes):
            if not screens or screen != screens[-1][1]:
                screens.append((first, screen))
            end = changes[k + 1][0] + lag if k + 1 < len(changes) else len(gate)
            # When decisions fill one auction later, yesterday's selected rule and
            # that rule's already-published signal still govern the opening boundary.
            start = first + lag if first else 0
            gate[start:end] = self.gates[lag][name][start:end]
        screen = self.screen_plan(screens, lag)
        plan = switched_targets(self.src.days, self.src.opn, self.engine.desk, screen, gate)
        curve = self.engine._price(plan, detail=True, stress=not lag)
        curve["gate_on"] = gate
        return curve


def summaries(path):
    output = []
    wealth = path.nav_krw.to_numpy()
    dates = path.date.to_numpy()
    for period, (start, end) in WINDOWS.items():
        active = (dates >= start) & (dates <= end)
        locs = np.flatnonzero(active)
        anchor = wealth[locs[0] - 1] if locs[0] else 1.
        values = np.r_[anchor, wealth[active]]
        output.append(dict(period=period, start=dates[locs[0]], end=dates[locs[-1]],
                           mdd=(values / np.maximum.accumulate(values)).min() - 1,
                           turnover=path.turnover[active].sum(), exposure=path.exposure[active].mean(),
                           off_days=(~path.gate_on[active]).sum(), **{"return": values[-1] / anchor - 1}))
    return output


def evaluate_walks(src, grid, choices, fixed, output=None):
    """Evaluate static rules, H1 lock, rolling Q1→Q4 and their stock/gate attributions."""
    root = Path(output) if output is not None else HERE
    paths = selection_paths(src, grid, choices, fixed)
    engine = SelectedReplay(src, grid)
    cache, curves, rows = {}, [], []
    base = list(paths["baseline"])
    reference = pd.read_csv(RECIPES / "selected_daily.csv", index_col=["key", "date"]).loc["balance/monthly1"]
    for lag, tests in [(0, [("timely_10bp", "krw10"), ("timely_25bp", "krw25"), ("timely_0bp", "krw")]),
                       (1, [("lag1_10bp", "krw10")])]:
        baseline = engine.evaluate(base, lag)
        cache[tuple(base), lag] = baseline
        if not lag:
            cols = ["krw", "krw10", "krw25", "turn", "exposure"]
            assert reference.index.equals(baseline.index)
            np.testing.assert_allclose(baseline[cols], reference[cols], rtol=2e-10, atol=1e-12)
            assert baseline.holdings.fillna("").equals(reference.holdings.fillna(""))
        for label, changes in paths.items():
            key = (tuple(changes), lag)
            if key not in cache:
                cache[key] = engine.evaluate(changes, lag)
            path = cache[key]
            if changes[0] == base[0] and len(changes) > 1:
                first = changes[1][0] + lag
                np.testing.assert_allclose(path.krw10.iloc[:first], baseline.krw10.iloc[:first], rtol=0, atol=1e-12)
                assert path.holdings.iloc[:first].equals(baseline.holdings.iloc[:first])
            for execution, column in tests:
                daily = pd.DataFrame(dict(date=src.days, label=label, execution=execution,
                                          nav_krw=path[column].to_numpy(), turnover=path.turn.to_numpy(),
                                          exposure=path.exposure.to_numpy(), gate_on=path.gate_on.to_numpy()))
                curves.append(daily)
                rows.extend(dict(label=label, execution=execution, **entry) for entry in summaries(daily))
    daily = pd.concat(curves, ignore_index=True)
    result = pd.DataFrame(rows)
    daily.to_csv(root / "selected_curves.csv", index=False, float_format="%.12g")
    result.to_csv(root / "selected_results.csv", index=False, float_format="%.12g")
    return daily, result
