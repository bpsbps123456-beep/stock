"""Reproduce secondary parameter-stratum diagnostics from the saved research grid.

    py -m gbrc.etf.research_subgroups [--out EXISTING_EXPERIMENT_DIRECTORY]

These strata were examined after the primary analysis.  Their train/validation
selection does not create a newly untouched holdout or a corrected significance
test.  This module reads existing artifacts and writes subgroup_evaluation.json;
it does not fetch prices or rerun portfolio simulations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

from . import research_validation as validation
from .storage import PROJECT_ROOT, atomic_write_json


DEFAULT_OUT = PROJECT_ROOT / "data/derived/analysis/2026-09-23_strategy_research"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(out: Path = DEFAULT_OUT) -> dict[str, Any]:
    """Use the frozen primary protocol within each explicitly named subgroup."""
    out = Path(out)
    configs = json.loads((out / "configurations.json").read_text(encoding="utf-8"))
    primary = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    returns = np.load(out / "grid_returns.npy", mmap_mode="r", allow_pickle=False)
    benchmarks = np.load(out / "benchmark_returns.npy", allow_pickle=False)
    with np.load(out / "bundle.npz", allow_pickle=False) as bundle:
        dates = bundle["dates"]
    if returns.shape != (len(configs), len(dates)) or benchmarks.shape != (len(dates), 3):
        raise ValueError("Saved configurations, daily grid, benchmarks and dates do not align")
    columns = {
        "train": np.flatnonzero(dates < "2026-06-01"),
        "validation": np.flatnonzero((dates >= "2026-06-01") & (dates < "2026-08-01")),
        "holdout": np.flatnonzero(dates >= "2026-08-01"),
        "full": np.arange(len(dates)),
    }
    metrics = {name: validation._batch_metrics(returns, columns[name])
               for name in ("train", "validation")}
    benchmark_metrics = {name: validation._batch_metrics(benchmarks.T, index)
                         for name, index in columns.items()}
    maximum = np.asarray([config["max_stocks"] for config in configs])
    mixes = np.asarray([config["time_mix"] for config in configs])
    frequency = np.asarray([config["rebalance"] for config in configs])
    families = np.asarray([config["family"] for config in configs])
    eligible_mask = np.zeros(len(configs), dtype=bool)
    eligible_mask[validation._eligible(metrics["train"])] = True
    masks = {f"max_stocks_{n}": maximum == n for n in (1, 2, 3, 5, 7, 10)}
    masks.update({
        "diversified_ge3": maximum >= 3,
        "both_funds_mixed_ge3": (maximum >= 3) & (mixes > 0) & (mixes < 1),
        "weekly_ge3": (maximum >= 3) & (frequency == "weekly"),
        "daily_ge3": (maximum >= 3) & (frequency == "daily"),
        "monthly_ge3": (maximum >= 3) & (frequency == "monthly"),
    })
    family_masks = {str(name): families == name for name in sorted(set(families))}

    def select(mask: np.ndarray) -> dict[str, Any]:
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError("Saved grid lacks one of the declared diagnostic strata")
        eligible = indices[eligible_mask[indices]]
        ranked = validation._rank(metrics["train"], eligible)
        shortlist = ranked[:ceil(len(eligible) * 0.01)]
        validated = validation._rank(metrics["validation"], shortlist)
        return {
            "candidate_count": int(len(indices)),
            "eligible_training_count": int(len(eligible)),
            "shortlist_count": int(len(shortlist)),
            "shortlist_global_indices": [int(index) for index in shortlist],
            "selected_global_index": int(validated[0]) if len(validated) else None,
            "train_only_global_index": int(ranked[0]) if len(ranked) else None,
        }

    selections = {name: select(mask) for name, mask in masks.items()}
    family_selections = {name: select(mask) for name, mask in family_masks.items()}
    # All diagnostic picks are frozen before inspecting subgroup holdout metrics.
    metrics.update({name: validation._batch_metrics(returns, columns[name])
                    for name in ("holdout", "full")})

    def profile(index: int | None, label: str) -> dict[str, Any] | None:
        if index is None:
            return None
        return {
            "label": label, "global_index": int(index), "config": configs[index],
            "segments": {
                name: {
                    **validation._metric_row(metric, index),
                    "benchmarks": {
                        benchmark_name: validation._comparison(
                            metric["return"][index], benchmark_metrics[name]["return"][j])
                        for j, benchmark_name in enumerate(validation.BENCHMARK_NAMES)},
                } for name, metric in metrics.items()},
        }

    def finish(mask: np.ndarray, selection: dict[str, Any]) -> dict[str, Any]:
        indices = np.flatnonzero(mask)
        best = int(validation._rank(metrics["full"], indices, "log_return")[0])
        return {
            **selection,
            "selected": profile(selection["selected_global_index"],
                "post-primary parameter-stratum diagnostic selected by train/validation; "
                "not new untouched holdout"),
            "full_hindsight_best": profile(best,
                "subgroup full-period hindsight return winner; not validated"),
            "distribution": {
                name: {
                    "median_return_pct": float(np.median(metric["return"][indices]) * 100),
                    "positive_return_pct": float(np.mean(metric["return"][indices] > 0) * 100),
                    "percentile10_return_pct": float(np.quantile(metric["return"][indices], .1) * 100),
                    "percentile90_return_pct": float(np.quantile(metric["return"][indices], .9) * 100),
                    "benchmarks": {
                        benchmark_name: validation._counts(
                            metric["return"][indices], benchmark_metrics[name]["return"][j])
                        for j, benchmark_name in enumerate(validation.BENCHMARK_NAMES)},
                } for name, metric in metrics.items()},
        }

    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_status": (
            "secondary parameter-stratum diagnostics after primary results; "
            "not new untouched validation"),
        "selection_rule": primary["selection_rule"],
        "holdout_status": primary["holdout_status"],
        "portfolio_basis": primary["segment_basis"],
        "global_index_basis": "zero-based row index into configurations.json and grid_returns.npy",
        "primary_selected_global_index": primary["selected_index"],
        "segments": {
            name: {
                "start": str(dates[index[0]]), "end": str(dates[index[-1]]),
                "observations": int(len(index)),
                "benchmarks": {
                    benchmark_name: validation._metric_row(benchmark_metrics[name], j)
                    for j, benchmark_name in enumerate(validation.BENCHMARK_NAMES)},
            } for name, index in columns.items()},
        "subgroups": {name: finish(mask, selections[name]) for name, mask in masks.items()},
        "families": {name: finish(mask, family_selections[name])
                     for name, mask in family_masks.items()},
        "limitations": [
            "Subgroups are examined after primary results; favorable subgroup results cannot "
            "create a newly untouched holdout.",
            "Each subgroup uses the same training top 1 percent and validation objective "
            "without selecting on holdout.",
            "Additional subgroup comparisons compound multiplicity; no adjusted significance claim.",
            "max_stocks is a cap, not a guarantee of at least that many held positions.",
            "Returns continue January-start portfolios; they are not fresh cash-start holdout results.",
        ],
        "source_files": {name: _sha256(out / name) for name in (
            "configurations.json", "grid_returns.npy", "benchmark_returns.npy", "evaluation.json")},
    }
    atomic_write_json(out / "subgroup_evaluation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="existing experiment directory containing grid artifacts")
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({
        "output": str(args.out / "subgroup_evaluation.json"),
        "subgroups": len(result["subgroups"]), "families": len(result["families"]),
        "selected_global_indices": {name: row["selected_global_index"]
                                    for name, row in result["subgroups"].items()},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
