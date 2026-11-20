"""Post-hoc diagnostic: require five off days and two new exits inside each gate training window."""
import json
import sys

import numpy as np
import pandas as pd

from forward import SelectedReplay, WINDOWS, selection_paths
from gates import HERE, load_gates, load_inputs
from ranking import ScreenGrid
# The adjacent studies also expose a top-level experiment.py.
sys.path.insert(0, str(HERE))
from experiment import read_grid


def run():
    src = load_inputs()
    grid = ScreenGrid(src)
    gates = load_gates(src)
    lagged = load_gates(src, lag=1)
    dates = np.array(src.days)
    gate_order = ["sma100", *(g for g in gates if g != "sma100")]
    fit = read_grid().set_index(["key", "gate"])
    info = json.loads((HERE / "grid_summary.json").read_text(encoding="utf-8"))
    choices = {k: tuple(v) for k, v in info["train_choices"].items()}
    fixed = {"baseline": (grid.baseline, "sma100")}
    limited, audit = {}, {}
    for train, (screen, unrestricted) in choices.items():
        start, end = WINDOWS[train.upper()]
        mask = (dates >= start) & (dates <= end)
        counts = {}
        dd_floor = fit.at[fixed["baseline"], train + "_dd"] - .02
        evidence, finalists, original_pool = [], [], []
        for name in gate_order:
            # Only exits at this open with the previous auction risk-on count.
            # A spell continuing across the quarter start contributes days but no new exit.
            off = ~gates[name]
            exits = off & lagged[name]
            counts[name] = int((mask & off).sum()), int((mask & exits).sum())
            sufficient = counts[name][0] >= 5 and counts[name][1] >= 2
            if sufficient:
                evidence.append(name)
            if fit.at[(screen, name), train + "_dd"] >= dd_floor - 1e-12:
                original_pool.append(name)
                if sufficient:
                    finalists.append(name)

        def winner(pool):
            if not pool:
                return "sma100"
            best = max(fit.at[(screen, g), train + "_ret"] for g in pool)
            return next(g for g in pool if fit.at[(screen, g), train + "_ret"] >= best - 1e-12)

        assert winner(original_pool) == unrestricted
        chosen = winner(finalists)
        limited[train] = (screen, chosen)
        audit[train] = dict(screen=screen, original_gate=unrestricted, selected_gate=chosen,
                            gate_count=13, sufficient_event_gates=len(evidence), passed_gates=len(finalists),
                            fallback=not finalists, original_train_off_days=counts[unrestricted][0],
                            original_train_new_exits=counts[unrestricted][1],
                            selected_train_off_days=counts[chosen][0], selected_train_new_exits=counts[chosen][1])
    replay = SelectedReplay(src, grid)
    paths = selection_paths(src, grid, limited, fixed)
    previous = pd.read_csv(HERE / "selected_results.csv").set_index(["label", "execution", "period"])
    rows = []
    for path, intervals in (("anchored_h1", [("h1", "H2"), ("h1", "Q4")]),
                            ("rolling", [("q1", "Q2"), ("q2", "Q3"), ("q3", "Q4"),
                                         ("q2+q3", "H2")])):
        for lag, execution in [(0, "timely_10bp"), (1, "lag1_10bp")]:
            actual = replay.evaluate(paths[path], lag).krw10
            for train, test in intervals:
                start, end = WINDOWS[test]
                begin = dates.searchsorted(start)
                stop = dates.searchsorted(end, side="right")
                anchor = actual.iat[begin - 1] if begin else 1.
                values = np.r_[anchor, actual.iloc[begin:stop]]
                ret, dd = values[-1] / anchor - 1, (values / np.maximum.accumulate(values)).min() - 1
                baseline = previous.loc[("baseline", execution, test)]
                original = previous.loc[(path, execution, test)]
                stats = audit.get(train, {})
                rows.append(dict(design="post_hoc_5_off_days_2_new_exits", path=path, train=train,
                                 test=test, execution=execution, **stats,
                                 tested_ret=ret, tested_dd=dd, live_ret=baseline["return"], live_dd=baseline.mdd,
                                 unrestricted_ret=original["return"], unrestricted_dd=original.mdd,
                                 return_delta_live=ret-baseline["return"], dd_delta_live=dd-baseline.mdd,
                                 return_delta_unrestricted=ret-original["return"], dd_delta_unrestricted=dd-original.mdd))
    result = pd.DataFrame(rows)
    result.to_csv(HERE / "gate_evidence.csv", index=False, float_format="%.12g")
    return result


if __name__ == "__main__":
    print(run()[["path", "train", "test", "execution", "selected_gate", "passed_gates",
                 "tested_ret", "tested_dd", "live_ret", "unrestricted_ret"]].to_string(index=False))
