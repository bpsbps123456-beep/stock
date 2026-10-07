"""Network-free validation tests: chronology, costs, MDD and selection bias labels."""

from __future__ import annotations

from datetime import date, timedelta
import json
import unittest

import numpy as np

from gbrc.etf import research_validation as validation


def weekdays(start: str, end: str) -> list[str]:
    current, stop = date.fromisoformat(start), date.fromisoformat(end)
    days = []
    while current <= stop:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    return days


class ResearchValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = weekdays("2026-01-02", "2026-09-23")
        self.days = np.asarray(self.dates)
        self.train = self.days < "2026-06-01"
        self.valid = (self.days >= "2026-06-01") & (self.days < "2026-08-01")
        self.holdout = self.days >= "2026-08-01"
        self.benchmarks = np.zeros((len(self.dates), 3))

    def fixture(self) -> tuple[np.ndarray, list[dict]]:
        returns = np.zeros((200, len(self.dates)), dtype=np.float32)
        returns[0, self.train] = 0.003
        returns[1, self.train] = 0.0029
        returns[2, self.train] = 0.001
        returns[0, self.valid] = -0.001
        returns[1, self.valid] = 0.004
        returns[2, self.holdout] = 0.025
        returns[1, self.holdout] = 0.0005
        return returns, [{"candidate": i} for i in range(len(returns))]

    def test_holdout_cannot_change_selection_or_training_shortlist(self) -> None:
        returns, configs = self.fixture()
        result = validation.evaluate_candidates(returns, self.dates, self.benchmarks, configs)
        self.assertEqual(result["shortlist_indices"], [0, 1])
        self.assertEqual(result["train_only_index"], 0)
        self.assertEqual(result["selected_index"], 1)
        self.assertEqual(result["best_index"], 2)
        changed = returns.copy()
        changed[:, self.holdout] = -0.01
        changed[0, self.holdout] = 0.05
        later = validation.evaluate_candidates(changed, self.dates, self.benchmarks, configs)
        self.assertEqual(later["selected_index"], result["selected_index"])
        self.assertEqual(later["shortlist_indices"], result["shortlist_indices"])
        self.assertEqual(later["best_index"], 0)
        self.assertIn("retrospective", later["holdout_status"])
        json.dumps(result, allow_nan=False)

    def test_drawdown_includes_initial_nav_and_excludes_exact_minus_fifty(self) -> None:
        returns = np.zeros((3, len(self.dates)))
        returns[0, 0] = -0.5
        returns[1, 0] = -0.51
        returns[2, 0] = -0.1
        result = validation.evaluate_candidates(returns, self.dates, self.benchmarks,
                                                [{"id": i} for i in range(3)])
        self.assertEqual(result["eligible_train_count"], 1)
        self.assertEqual(result["selected_index"], 2)
        self.assertAlmostEqual(result["selected"]["segments"]["train"]["max_drawdown_pct"], -10)
        self.assertAlmostEqual(result["selected_daily_wealth"][0]["nav"], 0.9)
        self.assertEqual(result["selected_initial_nav"], 1)

    def test_segment_boundaries_are_exact_and_no_ratio_for_zero_or_negative_benchmark(self) -> None:
        returns = np.zeros((1, len(self.dates)))
        for day, value in (("2026-05-29", 0.01), ("2026-06-01", 0.02),
                           ("2026-07-31", 0.03), ("2026-08-03", 0.04)):
            returns[0, self.dates.index(day)] = value
        self.benchmarks[:, 1] = -0.001
        self.benchmarks[:, 2] = 0.00001
        result = validation.evaluate_candidates(returns, self.dates, self.benchmarks, [{}])
        segments = result["selected"]["segments"]
        self.assertAlmostEqual(segments["train"]["return_pct"], 1)
        self.assertAlmostEqual(segments["validation"]["return_pct"], (1.02 * 1.03 - 1) * 100)
        self.assertAlmostEqual(segments["holdout"]["return_pct"], 4)
        self.assertIsNone(segments["full"]["benchmarks"]["time"]["return_ratio"])
        self.assertIsNone(segments["full"]["benchmarks"]["koact"]["return_ratio"])
        counts = result["segments"]["full"]["benchmarks"]
        self.assertIsNone(counts["koact"]["counts_all_candidates"]["beating_2x_return"])
        self.assertEqual(counts["qqq_krw"]["counts_all_candidates"]["beating_3x_return"], 1)

    def test_walkforward_uses_only_prior_data_and_applies_entry_then_switch_cost(self) -> None:
        returns = np.zeros((2, len(self.dates)))
        returns[0, self.train] = 0.002
        returns[1, self.train] = 0.0019
        returns[1, np.char.startswith(self.days, "2026-06")] = 0.05
        result = validation.evaluate_candidates(returns, self.dates, self.benchmarks, [{}, {}])
        wf = result["walk_forward"]
        self.assertEqual([p["candidate_index"] for p in wf["picks"]], [0, 1, 1, 1])
        self.assertEqual([p["extra_surcharge_bps"] for p in wf["picks"]], [15, 30, 0, 0])
        for pick in wf["picks"]:
            self.assertLess(pick["training_end"], pick["month"] + "-01")
        # June executes candidate 0, later months execute candidate 1, all are flat.
        self.assertAlmostEqual(wf["without_extra_surcharge"]["return_pct"], 0)
        self.assertAlmostEqual(wf["with_extra_surcharge"]["return_pct"],
                               ((1 - 0.0015) * (1 - 0.003) - 1) * 100)
        changed = returns.copy()
        changed[0, self.days >= "2026-07-01"] = 0.1
        later = validation.evaluate_candidates(changed, self.dates, self.benchmarks, [{}, {}])
        self.assertEqual([p["candidate_index"] for p in later["walk_forward"]["picks"][:2]],
                         [0, 1])

    def test_bootstrap_benchmark_is_frozen_by_training_performance(self) -> None:
        returns, configs = self.fixture()
        self.benchmarks[self.train] = [0.0004, 0.0002, 0.0003]
        self.benchmarks[self.holdout] = [-0.001, 0.015, 0.005]
        result = validation.evaluate_candidates(returns, self.dates, self.benchmarks, configs)
        bootstrap = result["holdout_bootstrap"]
        self.assertEqual(bootstrap["benchmark"], "time")
        self.assertEqual(bootstrap["samples"], 2000)
        self.assertEqual(bootstrap["block_length"], 5)
        self.assertEqual(bootstrap["positive_resample_pct"], 100)
        lo, hi = bootstrap["compound_log_excess_ci95"]
        self.assertAlmostEqual(lo, hi)
        self.assertGreater(lo, 0)
        self.assertIn("not a multiple-testing correction", bootstrap["interpretation"])

    def test_no_eligible_candidates_is_explicit(self) -> None:
        returns = np.zeros((2, len(self.dates)))
        returns[:, 0] = -0.5
        result = validation.evaluate_candidates(returns, self.dates, self.benchmarks, [{}, {}])
        self.assertIsNone(result["selected_index"])
        self.assertEqual(result["shortlist_count"], 0)
        self.assertEqual(result["holdout_bootstrap"]["status"], "unavailable")
        json.dumps(result, allow_nan=False)

    def test_ties_are_deterministic_and_chunking_preserves_metrics(self) -> None:
        returns = np.full((201, len(self.dates)), 0.0001, dtype=np.float32)
        result = validation.evaluate_candidates(returns, self.dates, self.benchmarks,
                                                [{} for _ in range(201)])
        self.assertEqual(result["shortlist_indices"], [0, 1, 2])
        self.assertEqual(result["selected_index"], 0)
        previous = validation.CHUNK_SIZE
        try:
            validation.CHUNK_SIZE = 17
            chunked = validation.evaluate_candidates(returns, self.dates, self.benchmarks,
                                                     [{} for _ in range(201)])
        finally:
            validation.CHUNK_SIZE = previous
        self.assertEqual(chunked["selected"], result["selected"])

    def test_invalid_dates_or_missing_price_returns_fail_explicitly(self) -> None:
        returns = np.zeros((1, len(self.dates)))
        returns[0, 10] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validation.evaluate_candidates(returns, self.dates, self.benchmarks, [{}])
        returns[0, 10] = 0
        dates = self.dates.copy()
        dates[0] = dates[1]
        with self.assertRaisesRegex(ValueError, "unique"):
            validation.evaluate_candidates(returns, dates, self.benchmarks, [{}])
        dates = self.dates.copy()
        dates[-1] = "2026-09-24"
        with self.assertRaisesRegex(ValueError, "holdout end"):
            validation.evaluate_candidates(returns, dates, self.benchmarks, [{}])


if __name__ == "__main__":
    unittest.main()
