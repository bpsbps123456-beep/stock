"""Mark selected Nov 18 closing portfolios at Nov 19 prices using frozen quotes and FX."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from forward import SelectedReplay, selection_paths
from gates import HERE, RECIPES, load_gates, load_inputs
from ranking import ScreenGrid
from extension import NEXT, PREVIOUS, get_forward, hold_return
from timing import review_days


LABELS = ["baseline", "short25_only", "hindsight", "anchored_h1", "rolling", "rolling_screen_only"]


def next_gates(src):
    """Add only the Nov 18 closes to the information available for the Nov 19 opening."""
    tickers = ["QQQ", "SOXX"]
    tail = pd.read_csv(HERE.parent / "seven_sleeves_20261120/tail_prices.csv", index_col="symbol")
    extra = tail.loc[tickers, "close"].rename(PREVIOUS)
    closes = pd.concat([src.book.cls[tickers], extra.to_frame().T])
    assert closes.index.is_unique and closes.index[-2] == src.signals[-1]
    dates = [*src.days, NEXT]
    market = SimpleNamespace(days=dates, signals=[*src.signals, PREVIOUS],
                             book=SimpleNamespace(cls=closes), desk_path=SimpleNamespace(days=src.days))
    return dates, load_gates(market)


def run():
    src = load_inputs()
    grid = ScreenGrid(src)
    configuration = json.loads((HERE / "grid_summary.json").read_text(encoding="utf-8"))
    choices = {k: tuple(v) for k, v in configuration["train_choices"].items()}
    fixed = {k: tuple(v) for k, v in configuration["fixed"].items()}
    paths = selection_paths(src, grid, choices, fixed)
    replay = SelectedReplay(src, grid)
    dates, gates = next_gates(src)
    tomorrow = get_forward(src)
    prior = pd.read_csv(RECIPES / "forward_mark.csv", index_col="key").loc["balance/monthly1"]
    rows = []
    for label in LABELS:
        curve = replay.evaluate(paths[label], lag=0)
        at_close = curve.attrs["final_weights_10bp"]
        weights = np.array([at_close.get(symbol, 0.) for symbol in src.symbols])
        np.testing.assert_allclose(weights.sum() + at_close["CASH"], 1., atol=1e-12)
        daily_return = hold_return(weights, src, tomorrow)
        ytd = curve.krw10.iat[-1] * (1 + daily_return) - 1
        _, (screen, name) = paths[label][-1]
        gate = gates[name]
        assert gate[-2] == curve.gate_on.iat[-1]
        gate_changed = gate[-1] != gate[-2]
        screen_due = len(src.days) in review_days(dates, gate, grid.screens.at[screen, "schedule"])
        desk_due = len(src.days) in review_days(dates, gate, "weekly1")
        scheduled_hold = not gate_changed and not (desk_due if gate[-1] else screen_due)
        if label == "baseline":
            np.testing.assert_allclose([daily_return, ytd], [prior.mark_return, prior.marked_ytd], atol=1e-10, rtol=0)
            assert scheduled_hold == prior.scheduled_hold
        rows.append(dict(label=label, holdings_date=PREVIOUS, mark_date=NEXT, marked_return=daily_return,
                         marked_ytd=ytd, gate=name, gate_on_before=gate[-2], gate_on_after=gate[-1],
                         gate_changed=gate_changed, screen_due=screen_due, desk_due=desk_due,
                         scheduled_hold=scheduled_hold, stock_weight=weights.sum()))
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "mark_next.csv", index=False, float_format="%.12g")
    return table


if __name__ == "__main__":
    print(run().to_string(index=False))
