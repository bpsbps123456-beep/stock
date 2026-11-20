"""Post-selection checks for full-period finalists and matched 20- versus 25-session averages."""
from pathlib import Path

import numpy as np
import pandas as pd

from experiment import FastEngine, ScreenGrid, WINDOWS, load_gates, load_inputs, read_grid, summaries

HERE = Path(__file__).resolve().parent
FIVE = ["year_ret", "h2_ret", "h2_dd", "q4_ret", "q4_dd"]
TOL = 1e-12


def paired_shorts(data, screens):
    dimensions = ["medium", "long", "alignment", "profile", "order", "schedule", "gate"]
    fields = ["year_ret", "h2_ret", "q4_ret"]
    detail = data.reset_index().merge(screens.reset_index(), on="key", validate="many_to_one")
    pair = {n: detail.loc[detail.short.eq(n)].set_index(dimensions).sort_index() for n in (20, 25)}
    assert pair[20].index.is_unique and pair[20].index.equals(pair[25].index) and len(pair[25]) == 14976
    deltas = pair[25][fields] - pair[20][fields]
    peak = pair[25].sort_values("year_ret", ascending=False, kind="stable").iloc[0]
    rows = []
    for field in fields:
        delta = deltas[field]
        rows.append(dict(metric=field, matched_pairs=len(delta), wins_25=(delta > TOL).sum(),
                         ties=delta.abs().le(TOL).sum(), losses_25=(delta < -TOL).sum(),
                         median_delta_25_minus_20=delta.median(), hindsight_25_screen=peak.key,
                         hindsight_25_gate=peak.name[-1], hindsight_25_value=peak[field],
                         paired_20_value=pair[20].at[peak.name, field]))
    result = pd.DataFrame(rows)
    result.to_csv(HERE / "paired_short.csv", index=False, float_format="%.12g")
    return result


def run():
    data = read_grid().set_index(["key", "gate"])
    screens = pd.read_csv(HERE / "screen_results.csv", index_col="key",
                          usecols=["key", "short", "medium", "long", "alignment", "profile", "order", "schedule"])
    assert len(data) == 89856 and data.index.is_unique and screens.index.is_unique
    paired = paired_shorts(data, screens)
    src = load_inputs()
    grid = ScreenGrid(src)
    baseline = (grid.baseline, "sma100")
    winners = (data[FIVE] > data.loc[baseline, FIVE] + TOL).all(axis=1)
    assert winners.sum() == 8
    engine = FastEngine(src)
    gates = [load_gates(src), load_gates(src, lag=1)]
    days = pd.Index(src.days)
    spans = [(days.searchsorted(WINDOWS[w][0]), days.searchsorted(WINDOWS[w][1], side="right"))
             for w in ["year", "h2", "q4"]]
    outputs = []
    ordered = sorted(winners[winners].index, key=lambda pair: (screens.at[pair[0], "schedule"], *pair))
    for key, gate in [baseline, *ordered]:
        neighbours = [(other, gate) for other in grid.neighbors(key)]
        for case, lag, fee in [("timely_10bp", 0, .001), ("timely_25bp", 0, .0025), ("lag1_10bp", 1, .001)]:
            path = engine.evaluate(grid.ranks(key), screens.at[key, "schedule"], gates[lag][gate], fee=fee, delay=lag)
            values = summaries(path.nav, path.exposure, spans)[0]
            stats = {f"{w}_{field}": values[i, j] for i, w in enumerate(["year", "h2", "q4"])
                     for j, field in enumerate(["ret", "dd"])}
            if case == "timely_10bp":
                np.testing.assert_allclose(list(stats.values()), data.loc[(key, gate), list(stats)], atol=1e-10, rtol=0)
            outputs.append(dict(key=key, gate=gate, test=case, hindsight_finalist=(key, gate) != baseline,
                                n_neighbors_incl_self=len(neighbours), n_beating_five_timely_10bp_incl_self=int(
                                    winners.loc[neighbours].sum()), **stats))
    result = pd.DataFrame(outputs)
    control = result[~result.hindsight_finalist].set_index("test")
    deltas = result[FIVE].to_numpy() - control.loc[result.test, FIVE].to_numpy()
    result["beats_all_five"] = (deltas > TOL).all(axis=1)
    result.to_csv(HERE / "finalists.csv", index=False, float_format="%.12g")
    return result, paired


if __name__ == "__main__":
    finalists, paired = run()
    print(finalists.to_string(index=False))
    print(paired.to_string(index=False))
