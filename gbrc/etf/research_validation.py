"""Retrospective temporal validation of a large ETF-following strategy search.

The caller supplies daily net returns, including its own trading costs.  All
segments continue the caller's existing positions and simply rebase NAV to one;
they do not pretend to be fresh investments.  Selection uses only training and
validation dates.  The August/September evaluation is a retrospective holdout,
not a claim that the historical period was previously unseen by the researcher.
"""

from __future__ import annotations

from datetime import date
from math import ceil
from typing import Any

import numpy as np


BENCHMARK_NAMES = ("time", "koact", "qqq_krw")
TRAIN_END = "2026-06-01"  # exclusive
VALIDATION_END = "2026-08-01"  # exclusive
HOLDOUT_END = "2026-09-23"  # inclusive
CHUNK_SIZE = 2048
BOOTSTRAP_SEED = 20260927
BOOTSTRAP_SAMPLES = 2000


def _number(value: float) -> float | None:
    """Produce strict JSON numbers; infinities are not JSON numeric values."""
    return float(value) if np.isfinite(value) else None


def _batch_metrics(values: np.ndarray, columns: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics with bounded temporary memory and an explicit initial NAV of 1."""
    size = values.shape[0]
    result = {key: np.empty(size, dtype=np.float64)
              for key in ("return", "max_drawdown", "log_return", "objective")}
    if not len(columns):
        raise ValueError("Cannot evaluate an empty date segment")
    for start in range(0, size, CHUNK_SIZE):
        stop = min(size, start + CHUNK_SIZE)
        block = np.array(values[start:stop, columns], dtype=np.float64, copy=True)
        with np.errstate(divide="ignore"):
            np.log1p(block, out=block)
        np.cumsum(block, axis=1, out=block)
        log_return = block[:, -1].copy()
        peaks = np.maximum.accumulate(block, axis=1)
        # Including t=0 prevents a first-day loss from disappearing from MDD.
        np.maximum(peaks, 0.0, out=peaks)
        np.subtract(block, peaks, out=block)
        np.expm1(block, out=block)
        drawdown = np.minimum(block.min(axis=1), 0.0)
        with np.errstate(over="ignore"):
            result["return"][start:stop] = np.expm1(log_return)
        result["max_drawdown"][start:stop] = drawdown
        result["log_return"][start:stop] = log_return
        result["objective"][start:stop] = log_return - 0.5 * np.abs(drawdown)
    return result


def _one_metrics(values: np.ndarray) -> dict[str, float | None]:
    metrics = _batch_metrics(values.reshape(1, -1), np.arange(len(values)))
    return _metric_row(metrics, 0)


def _metric_row(metrics: dict[str, np.ndarray], index: int) -> dict[str, float | None]:
    return {
        "return_pct": _number(metrics["return"][index] * 100),
        "max_drawdown_pct": _number(metrics["max_drawdown"][index] * 100),
        "compound_log_return": _number(metrics["log_return"][index]),
        "selection_objective": _number(metrics["objective"][index]),
    }


def _eligible(metrics: dict[str, np.ndarray]) -> np.ndarray:
    # A -50% drawdown is excluded, not merely drawdowns worse than -50%.
    return np.flatnonzero((metrics["max_drawdown"] > -0.5)
                         & np.isfinite(metrics["objective"]))


def _rank(metrics: dict[str, np.ndarray], indices: np.ndarray,
          key: str = "objective") -> np.ndarray:
    """Deterministic lower candidate index breaks exact score ties."""
    return indices[np.lexsort((indices, -metrics[key][indices]))]


def _counts(candidate_returns: np.ndarray, benchmark_return: float) -> dict[str, Any]:
    positive = bool(benchmark_return > 0)
    return {
        "candidate_count": int(len(candidate_returns)),
        "beating_benchmark": int(np.count_nonzero(candidate_returns > benchmark_return)),
        "beating_2x_return": (int(np.count_nonzero(candidate_returns > 2 * benchmark_return))
                              if positive else None),
        "beating_3x_return": (int(np.count_nonzero(candidate_returns > 3 * benchmark_return))
                              if positive else None),
        "multiples_applicable": positive,
        "multiples_basis": "cumulative_profit_rate, not ending_wealth",
    }


def _comparison(candidate_return: float, benchmark_return: float) -> dict[str, Any]:
    return {
        "excess_return_pp": _number((candidate_return - benchmark_return) * 100),
        "return_ratio": (_number(candidate_return / benchmark_return)
                         if benchmark_return > 0 else None),
        "ratio_applicable": bool(benchmark_return > 0),
    }


def _bootstrap_excess(candidate: np.ndarray, benchmark: np.ndarray,
                      benchmark_name: str, benchmark_train: dict[str, Any]) -> dict[str, Any]:
    """Overlapping moving-block bootstrap of holdout daily log excess returns."""
    observations, block_length = len(candidate), 5
    out: dict[str, Any] = {
        "method": "overlapping_moving_block_bootstrap",
        "block_length": block_length,
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "observations": observations,
        "benchmark": benchmark_name,
        "benchmark_selection": "highest training-period cumulative return among three benchmarks",
        "benchmark_training_metrics": benchmark_train,
        "interpretation": "descriptive conditional uncertainty; not a multiple-testing correction",
        "holdout_status": "retrospective holdout; historical data were available to researcher",
    }
    if observations < block_length:
        return {**out, "status": "unavailable", "reason": "fewer than five holdout returns"}
    with np.errstate(divide="ignore", invalid="ignore"):
        excess = np.log1p(candidate.astype(np.float64)) - np.log1p(benchmark.astype(np.float64))
    if not np.isfinite(excess).all():
        return {**out, "status": "unavailable", "reason": "nonfinite log excess after total loss"}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks_needed = ceil(observations / block_length)
    starts = rng.integers(0, observations - block_length + 1,
                         size=(BOOTSTRAP_SAMPLES, blocks_needed))
    indices = (starts[:, :, None] + np.arange(block_length)).reshape(BOOTSTRAP_SAMPLES, -1)
    sums = excess[indices[:, :observations]].sum(axis=1)
    lower, upper = np.quantile(sums, [0.025, 0.975])
    return {
        **out,
        "status": "ok",
        "observed_compound_log_excess": _number(excess.sum()),
        "observed_relative_wealth_return_pct": _number(np.expm1(excess.sum()) * 100),
        "compound_log_excess_ci95": [_number(lower), _number(upper)],
        "relative_wealth_return_ci95_pct": [
            _number(np.expm1(lower) * 100), _number(np.expm1(upper) * 100)],
        "positive_resample_pct": float(np.mean(sums > 0) * 100),
    }


def _walk_forward(values: np.ndarray, dates: list[str], benchmarks: np.ndarray,
                  configs: list[dict]) -> dict[str, Any]:
    """Expand the training window at each calendar-month start from June."""
    months = sorted({day[:7] for day in dates if TRAIN_END <= day <= HOLDOUT_END})
    date_array = np.asarray(dates)
    picks, skipped, gross_parts, charged_parts, realised_columns = [], [], [], [], []
    previous_index: int | None = None
    for month in months:
        month_start = month + "-01"
        history = np.flatnonzero(date_array < month_start)
        month_columns = np.flatnonzero(np.char.startswith(date_array, month))
        if len(history) < 50:
            skipped.append({"month": month, "reason": "fewer than 50 prior daily returns",
                            "prior_observations": int(len(history))})
            continue
        metrics = _batch_metrics(values, history)
        eligible = _eligible(metrics)
        if not len(eligible):
            skipped.append({"month": month, "reason": "no candidate passes prior drawdown filter",
                            "prior_observations": int(len(history))})
            continue
        index = int(_rank(metrics, eligible)[0])
        month_returns = np.asarray(values[index, month_columns], dtype=np.float64).copy()
        changed = previous_index is not None and index != previous_index
        # The supplied June returns are a continuation of the January portfolio.
        # A June cash start needs its own entry allowance, then two-way switches.
        surcharge = 0.0015 if previous_index is None else (0.003 if changed else 0.0)
        charged = month_returns.copy()
        charged[0] = (1 + charged[0]) * (1 - surcharge) - 1
        picks.append({
            "month": month,
            "candidate_index": index,
            "config": configs[index],
            "training_start": dates[int(history[0])],
            "training_end": dates[int(history[-1])],
            "training_observations": int(len(history)),
            "training_metrics": _metric_row(metrics, index),
            "eligible_candidates": int(len(eligible)),
            "realised_start": dates[int(month_columns[0])],
            "realised_end": dates[int(month_columns[-1])],
            "realised_observations": int(len(month_columns)),
            "changed_candidate": changed,
            "extra_surcharge_bps": surcharge * 10000,
            "without_extra_surcharge": _one_metrics(month_returns),
            "with_extra_surcharge": _one_metrics(charged),
            "benchmarks": {name: _one_metrics(benchmarks[month_columns, i])
                           for i, name in enumerate(BENCHMARK_NAMES)},
        })
        gross_parts.append(month_returns)
        charged_parts.append(charged)
        realised_columns.extend(int(column) for column in month_columns)
        previous_index = index
    result: dict[str, Any] = {
        "selection_rule": "best prior objective with at least 50 prior returns and MDD > -50%",
        "selection_objective": "compound log return minus 0.5 times absolute maximum drawdown",
        "picks": picks,
        "skipped_months": skipped,
        "switch_count": sum(row["changed_candidate"] for row in picks),
        "extra_cost_assumption": (
            "Approximate NAV haircut before first realised return: 15 bps initial cash entry; "
            "30 bps on subsequent candidate changes; 0 bps when unchanged. Candidate returns "
            "already include their own rebalances. Stitched continuing portfolios are not "
            "a fully simulated switching portfolio."
        ),
        "portfolio_basis": "stitched ongoing portfolios; extra initial/switch cost approximation",
    }
    if picks:
        result.update({
            "start": dates[realised_columns[0]], "end": dates[realised_columns[-1]],
            "observations": len(realised_columns),
            "without_extra_surcharge": _one_metrics(np.concatenate(gross_parts)),
            "with_extra_surcharge": _one_metrics(np.concatenate(charged_parts)),
            "benchmarks": {name: _one_metrics(benchmarks[realised_columns, i])
                           for i, name in enumerate(BENCHMARK_NAMES)},
        })
    return result


def evaluate_candidates(returns: np.ndarray, dates: list[str],
                        benchmark_returns: np.ndarray, configs: list[dict]) -> dict[str, Any]:
    """Evaluate [candidate, date] returns against [date, TIME/KoAct/QQQ] returns.

    All values must be finite, daily simple returns no smaller than -1.  Dates
    must be unique chronological ISO dates ending no later than 2026-09-23.
    Training is before June, validation June/July, retrospective holdout August
    through September 23.  Ties always resolve to the lower candidate index.
    The entire result is JSON serializable, without returning the large matrix.
    """
    values = np.asarray(returns)
    benchmarks = np.asarray(benchmark_returns)
    if values.ndim != 2 or not values.shape[0]:
        raise ValueError("returns must have shape [nonempty candidates, dates]")
    if values.shape[1] != len(dates) or benchmarks.shape != (len(dates), 3):
        raise ValueError("returns/dates/benchmark shapes do not match")
    if len(configs) != values.shape[0] or any(not isinstance(c, dict) for c in configs):
        raise ValueError("configs must contain one dictionary for every candidate")
    if not dates or dates != sorted(set(dates)):
        raise ValueError("dates must be nonempty, unique and chronological")
    if any(date.fromisoformat(day).isoformat() != day for day in dates):
        raise ValueError("dates must use YYYY-MM-DD")
    if dates[-1] > HOLDOUT_END:
        raise ValueError("data extend beyond the predeclared holdout end")
    for start in range(0, len(values), CHUNK_SIZE):
        chunk = values[start:start + CHUNK_SIZE]
        if not np.isfinite(chunk).all() or np.any(chunk < -1):
            raise ValueError("candidate returns must be finite and at least -1")
    if not np.isfinite(benchmarks).all() or np.any(benchmarks < -1):
        raise ValueError("benchmark returns must be finite and at least -1")

    date_array = np.asarray(dates)
    columns = {
        "train": np.flatnonzero(date_array < TRAIN_END),
        "validation": np.flatnonzero((date_array >= TRAIN_END) & (date_array < VALIDATION_END)),
        "holdout": np.flatnonzero(date_array >= VALIDATION_END),
        "full": np.arange(len(dates)),
    }
    if any(not len(index) for index in columns.values()):
        raise ValueError("training, validation and holdout each need at least one date")

    # Freeze the selected index before computing any holdout or full-period metrics.
    train = _batch_metrics(values, columns["train"])
    eligible = _eligible(train)
    train_ranking = _rank(train, eligible)
    shortlist_size = ceil(len(eligible) * 0.01)
    shortlist = train_ranking[:shortlist_size]
    validation = _batch_metrics(values, columns["validation"])
    validation_ranking = _rank(validation, shortlist)
    selected_index = int(validation_ranking[0]) if len(validation_ranking) else None
    train_only_index = int(train_ranking[0]) if len(train_ranking) else None

    metrics = {"train": train, "validation": validation,
               "holdout": _batch_metrics(values, columns["holdout"]),
               "full": _batch_metrics(values, columns["full"])}
    benchmark_metrics = {name: _batch_metrics(benchmarks.T, index)
                         for name, index in columns.items()}
    all_indices = np.arange(len(values))
    hindsight_ranking = _rank(metrics["full"], all_indices, "log_return")
    best_index = int(hindsight_ranking[0])
    full_objective_best = int(_rank(metrics["full"], all_indices)[0])
    stronger_benchmark = int(np.argmax(benchmark_metrics["train"]["log_return"]))

    segments = {}
    for name, index in columns.items():
        segment_benchmarks = {}
        for i, benchmark_name in enumerate(BENCHMARK_NAMES):
            benchmark_return = benchmark_metrics[name]["return"][i]
            segment_benchmarks[benchmark_name] = {
                **_metric_row(benchmark_metrics[name], i),
                "counts_all_candidates": _counts(metrics[name]["return"], benchmark_return),
                "counts_train_shortlist": _counts(metrics[name]["return"][shortlist],
                                                   benchmark_return),
            }
        segments[name] = {
            "start": dates[int(index[0])], "end": dates[int(index[-1])],
            "observations": int(len(index)),
            "portfolio_basis": "ongoing holdings rebased to NAV 1; no added segment entry/exit cost",
            "benchmarks": segment_benchmarks,
        }

    def profile(index: int | None, label: str) -> dict[str, Any] | None:
        if index is None:
            return None
        return {
            "label": label, "index": index, "config": configs[index],
            "segments": {
                name: {
                    **_metric_row(metric, index),
                    "benchmarks": {
                        benchmark_name: _comparison(metric["return"][index],
                                                    benchmark_metrics[name]["return"][i])
                        for i, benchmark_name in enumerate(BENCHMARK_NAMES)},
                } for name, metric in metrics.items()},
        }

    result: dict[str, Any] = {
        "schema_version": 1,
        "candidate_count": int(len(values)),
        "daily_observations": len(dates),
        "benchmark_order": list(BENCHMARK_NAMES),
        "selection_status": ("selected" if selected_index is not None
                             else "no_candidate_passes_training_drawdown_filter"),
        "selection_rule": {
            "train_before": TRAIN_END,
            "validation_from": TRAIN_END,
            "validation_before": VALIDATION_END,
            "holdout_from": VALIDATION_END,
            "holdout_through": HOLDOUT_END,
            "objective": "compound_log_return - 0.5 * abs(max_drawdown_fraction)",
            "exclude_training_max_drawdown_pct_at_or_below": -50.0,
            "shortlist_fraction_of_eligible": 0.01,
            "shortlist_rounding": "ceiling",
            "tie_break": "lower candidate index",
            "final_selection": "highest validation objective within frozen training shortlist",
            "duplicate_paths": "not deduplicated; deterministic index ordering",
        },
        "holdout_status": "retrospective holdout; not a claim of historically unseen research data",
        "segment_basis": "ongoing portfolios, NAV rebased to 1 including first daily return",
        "eligible_train_count": int(len(eligible)),
        "shortlist_count": int(len(shortlist)),
        "shortlist_indices": [int(i) for i in shortlist],
        "validation_ranked_shortlist_indices": [int(i) for i in validation_ranking],
        "selected_index": selected_index,
        "train_only_index": train_only_index,
        "best_index": best_index,
        "hindsight_return_best_index": best_index,
        "hindsight_objective_best_index": full_objective_best,
        "hindsight_top20_indices": [int(i) for i in hindsight_ranking[:20]],
        "segments": segments,
        "selected": profile(selected_index, "training-shortlist then validation selected"),
        "train_only": profile(train_only_index, "training-only objective winner"),
        "hindsight_best": profile(best_index, "full-period hindsight highest return; not validated"),
        "hindsight_objective_best": profile(full_objective_best,
                                             "full-period hindsight objective winner; not validated"),
        "walk_forward": _walk_forward(values, dates, benchmarks, configs),
        "limitations": [
            "Historical temporal holdout is retrospective, not previously unseen live data.",
            "A large parameter search creates selection bias; bootstrap is not its correction.",
            "Segment results continue existing positions and are not fresh-start account simulations.",
            "Return multiples are undefined when the benchmark cumulative return is nonpositive.",
            "All results are observed-period returns without annualized projections.",
        ],
    }
    if selected_index is not None:
        selected = np.asarray(values[selected_index], dtype=np.float64)
        wealth = np.cumprod(1 + selected)
        result["selected_initial_nav"] = 1.0
        result["selected_daily_wealth"] = [
            {"date": day, "nav": _number(nav)} for day, nav in zip(dates, wealth)]
        result["holdout_bootstrap"] = _bootstrap_excess(
            selected[columns["holdout"]],
            benchmarks[columns["holdout"], stronger_benchmark],
            BENCHMARK_NAMES[stronger_benchmark],
            _metric_row(benchmark_metrics["train"], stronger_benchmark),
        )
    else:
        result["selected_daily_wealth"] = []
        result["holdout_bootstrap"] = {"status": "unavailable", "reason": "no selected candidate"}
    return result
