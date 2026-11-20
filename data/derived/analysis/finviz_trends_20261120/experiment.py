"""Price every published trend combination and apply the locked chronological selection."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from ranking import ScreenGrid, screen_key
from engine import FastEngine
from gates import load_gates, load_inputs

WINDOWS = {"year": ("2026-01-01", "2026-11-18"), "h1": ("2026-01-01", "2026-06-30"),
           "h2": ("2026-07-01", "2026-11-18"), "q1": ("2026-01-01", "2026-03-31"),
           "q2": ("2026-04-01", "2026-06-30"), "q3": ("2026-07-01", "2026-09-30"),
           "q4": ("2026-10-01", "2026-11-18")}
FITS = ("h1", "q1", "q2", "q3")


def summaries(nav, exposure, spans):
    """Include the previous close in both return and drawdown of each window."""
    nav, exposure = np.atleast_2d(nav), np.atleast_2d(exposure)
    out = np.empty((len(nav), len(spans), 3))
    for p, (start, end) in enumerate(spans):
        initial = nav[:, start-1] if start else np.ones(len(nav))
        curve = nav[:, start:end]
        peaks = np.maximum(initial[:, None], np.maximum.accumulate(curve, axis=1))
        out[:, p, 0] = curve[:, -1] / initial - 1
        out[:, p, 1] = np.minimum(0, (curve / peaks - 1).min(axis=1))
        out[:, p, 2] = exposure[:, start:end].mean(axis=1)
    return out


def run_grid(src, grid):
    gate_dict = load_gates(src)
    gate_names = list(gate_dict)
    gates = np.stack(list(gate_dict.values()))
    days = pd.Index(src.days)
    spans = [(days.searchsorted(start), days.searchsorted(end, side="right")) for start, end in WINDOWS.values()]
    engine = FastEngine(src)
    metadata = grid.screens.copy()
    n = len(metadata)
    hybrid = np.empty((n, len(gates), len(spans), 3))
    always = np.empty((n, len(spans), 3))
    first_seen = {}
    for i, (key, schedule) in enumerate(metadata.schedule.items()):
        reviews = grid.reviews[schedule]
        ranks = grid.ranks(key)[reviews]
        signature = schedule, ranks.tobytes()
        j = first_seen.setdefault(signature, i)
        if i != j:
            hybrid[i], always[i] = hybrid[j], always[j]
            continue
        paths = engine.evaluate(ranks, reviews, gates)
        hybrid[i] = summaries(paths.nav, paths.exposure, spans)
        always[i] = summaries(paths.always_nav, paths.always_exposure, spans)[0]
    metadata["basket_id"] = [first_seen[(schedule, grid.ranks(key)[grid.reviews[schedule]].tobytes())]
                             for key, schedule in metadata.schedule.items()]
    # Store all gated returns and drawdowns at enough precision to check the train picks.
    columns = [f"{w}_{field}" for w in WINDOWS for field in ("ret", "dd", "exposure")]
    standalone = pd.DataFrame(always.reshape(n, -1), index=metadata.index, columns=columns)
    metadata.join(standalone).to_csv(HERE / "screen_results.csv", float_format="%.12g")
    directory = HERE / "market_grid"
    directory.mkdir(exist_ok=True)
    unique = np.flatnonzero(~metadata.basket_id.duplicated())
    np.testing.assert_array_equal(metadata.basket_id.iloc[unique], unique)
    per_gate = [f"{w}_{field}" for w in WINDOWS for field in ("ret", "dd")]
    for i, name in enumerate(gate_names):
        path = directory / f"{name}.csv"
        results = pd.DataFrame(hybrid[unique, i, :, :2].reshape(len(unique), -1), index=unique, columns=per_gate)
        results.to_csv(path, index_label="basket_id", float_format="%.12g")
        assert path.stat().st_size < 10_000_000
    return metadata, always, hybrid, gate_names


def read_grid():
    """Expand distinct baskets back to every named rule and market gate."""
    keys = pd.read_csv(HERE / "screen_results.csv", usecols=["key", "basket_id"])
    tables = []
    for path in sorted((HERE / "market_grid").glob("*.csv")):
        metrics = pd.read_csv(path).assign(gate=path.stem)
        expanded = keys.merge(metrics, on="basket_id", validate="many_to_one")
        assert len(expanded) == len(keys)
        tables.append(expanded)
    if len(tables) != 13:
        raise ValueError("Run experiment.py to generate all 13 market grids")
    return pd.concat(tables, ignore_index=True)


def select(grid, meta, always, hybrid, gate_names):
    base = meta.index.get_loc(grid.baseline)
    current_gate = gate_names.index("sma100")
    distance = (meta.short-20).abs() + (meta.medium-50).abs() + (meta.long-200).abs()
    neighbors = [meta.index.get_indexer(grid.neighbors(key)) for key in meta.index]
    selections, audit = {}, []
    for fit in FITS:
        w = list(WINDOWS).index(fit)
        observed = always[:, w]
        viable = (observed[:, 2] >= .6) & (observed[:, 1] >= observed[base, 1]-.02-1e-12)
        neighborhood = np.array([np.median(observed[idx, 0]) for idx in neighbors])
        viable &= neighborhood > observed[base, 0] + 1e-12
        eligible = np.flatnonzero(viable)
        stock = base
        if len(eligible):
            best = neighborhood[eligible].max()
            tied = eligible[neighborhood[eligible] >= best - 1e-12]
            stock = min(tied, key=lambda i:(i != base, distance.iat[i], meta.index[i]))
        stock_key = meta.index[stock]
        gated = hybrid[stock, :, w]
        eligible_gates = [i for i in [current_gate, *(k for k in range(len(gate_names)) if k != current_gate)]
                          if gated[i, 1] >= hybrid[base, current_gate, w, 1]-.02-1e-12]
        gate = current_gate
        if eligible_gates:
            best = gated[eligible_gates, 0].max()
            gate = next(i for i in eligible_gates if gated[i, 0] >= best-1e-12)
        selections[fit] = stock_key, gate_names[gate]
        audit.append(dict(train=fit, screen=stock_key, gate=gate_names[gate],
                          eligible_screens=len(eligible), eligible_gates=len(eligible_gates),
                          neighborhood_count=len(neighbors[stock]), neighbor_median=neighborhood[stock],
                          standalone_ret=observed[stock, 0], standalone_dd=observed[stock, 1],
                          standalone_exposure=observed[stock, 2], baseline_standalone_ret=observed[base, 0],
                          hybrid_train_ret=gated[gate, 0], hybrid_train_dd=gated[gate, 1],
                          baseline_train_ret=hybrid[base, current_gate, w, 0],
                          baseline_train_dd=hybrid[base, current_gate, w, 1]))
    pd.DataFrame(audit).to_csv(HERE / "selection.csv", index=False, float_format="%.12g")
    return selections


def fixed_checks(meta, grid, gate_names, hybrid):
    base, market = meta.index.get_loc(grid.baseline), gate_names.index("sma100")
    y, h2, q4 = [list(WINDOWS).index(w) for w in ("year", "h2", "q4")]
    annual = hybrid[:, :, y, 0]
    # Exact ties usually mean identical baskets or gates. Keep the first published pair.
    peak = annual.max()
    best_s, best_g = np.argwhere(annual >= peak-1e-12)[0]
    best_stock = np.flatnonzero(annual[:, market] >= annual[:, market].max()-1e-12)[0]
    best_gate = np.flatnonzero(annual[base] >= annual[base].max()-1e-12)[0]
    labels = {"baseline": (grid.baseline, "sma100"), "hindsight": (meta.index[best_s], gate_names[best_g]),
              "hindsight_stock_only": (meta.index[best_stock], "sma100"),
              "hindsight_gate_only": (grid.baseline, gate_names[best_gate])}
    for short in (10, 15, 25, 30, 35):
        labels[f"short{short}_only"] = (screen_key(short, 50, 200, "base", "balance", "rv", "monthly1"), "sma100")
    better = np.ones(annual.shape, bool)
    for p, column in [(y, 0), (h2, 0), (h2, 1), (q4, 0), (q4, 1)]:
        better &= hybrid[:, :, p, column] > hybrid[base, market, p, column]+1e-12
    basket_paths = meta.basket_id.nunique()
    info = dict(configurations=int(annual.size), stock_schedules=len(meta), unique_scheduled_rankings=basket_paths,
                beats_baseline_ytd=int((annual > annual[base, market]+1e-12).sum()),
                beats_all_five=int(better.sum()), tied_ytd_winners=int((annual >= peak-1e-12).sum()),
                fixed=labels)
    return labels, info


def run():
    src = load_inputs()
    grid = ScreenGrid(src)
    metadata, always, hybrid, gates = run_grid(src, grid)
    choices = select(grid, metadata, always, hybrid, gates)
    fixed, info = fixed_checks(metadata, grid, gates, hybrid)
    assert info["configurations"] == 89_856
    info["train_choices"] = choices
    (HERE / "grid_summary.json").write_text(json.dumps(info, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return src, grid, choices, fixed


if __name__ == "__main__":
    src, grid, choices, fixed = run()
    if "--grid-only" not in sys.argv:
        # Prior research helpers also include a module named forward.
        sys.path.insert(0, str(HERE))
        from forward import evaluate_walks
        evaluate_walks(src, grid, choices, fixed)
