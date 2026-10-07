"""Regression tests for the canonical ETF snapshot foundation.

The fixtures are deliberately small and synthetic.  Every filesystem test uses
``TemporaryDirectory`` so the legacy and canonical project data stay untouched.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from gbrc.config import load_fund_configs, resolve_fund
from gbrc.etf.migration import migrate_legacy_snapshots
from gbrc.etf.snapshot import (
    SnapshotValidationError,
    canonical_filename,
    normalize_snapshot,
    validate_snapshot,
)
from gbrc.etf.storage import (
    SnapshotConflictError,
    atomic_write_json,
    load_normalized_snapshot,
    save_normalized_snapshot,
)


def koact_legacy_snapshot() -> dict[str, object]:
    """Return a provider-style KoAct record with stock ratios plus cash.

    The two security source ratios total 100%.  Cash is an additional NAV asset,
    which reproduces the denominator mismatch that previously pushed totals over
    100%.
    """

    return {
        "date": "2026-07-05",
        "actual_date": "2026-07-03",
        "etf_ticker": "2ETFQ1",
        "etf_name": "KoAct US Nasdaq Growth Companies Active",
        "source_url": "https://example.invalid/koact/2ETFQ1",
        "fetched_at": "2026-07-05T20:56:39+09:00",
        "holdings_count": 3,
        "holdings": [
            {
                "code": "AAA US EQUITY",
                "name": "Alpha",
                "qty": 6,
                "value_krw": 600,
                "weight": 60.0,
            },
            {
                "code": "BBB US EQUITY",
                "name": "Beta",
                "qty": 4,
                "value_krw": 400,
                "weight": 40.0,
            },
            {
                "code": "",
                "name": "Cash",
                "qty": 100,
                "value_krw": 100,
                "weight": 10.0,
            },
        ],
    }


class SnapshotNormalizationTests(unittest.TestCase):
    def test_koact_cash_is_counted_once_on_a_single_nav_denominator(self) -> None:
        normalized = normalize_snapshot(koact_legacy_snapshot())

        security_source_sum = sum(
            row["source_weight_pct"]
            for row in normalized["holdings"]
            if row["asset_type"] == "security"
        )
        self.assertAlmostEqual(security_source_sum, 100.0, places=8)
        self.assertAlmostEqual(
            normalized["metrics"]["source_non_cash_weight_sum_pct"],
            100.0,
            places=8,
        )

        nav_sum = sum(row["nav_weight_pct"] for row in normalized["holdings"])
        self.assertAlmostEqual(nav_sum, 100.0, places=7)
        self.assertAlmostEqual(
            normalized["metrics"]["nav_weight_sum_pct"], 100.0, places=7
        )

        cash_rows = [
            row for row in normalized["holdings"] if row["asset_type"] == "cash"
        ]
        self.assertEqual(len(cash_rows), 1)
        expected_cash_weight = 100 / (600 + 400 + 100) * 100
        self.assertAlmostEqual(
            cash_rows[0]["nav_weight_pct"], expected_cash_weight, places=7
        )
        self.assertAlmostEqual(
            normalized["metrics"]["cash_weight_pct"],
            cash_rows[0]["nav_weight_pct"],
            places=8,
        )

    def test_source_weights_are_preserved_separately_from_nav_weights(self) -> None:
        normalized = normalize_snapshot(koact_legacy_snapshot())
        rows_by_name = {row["name"]: row for row in normalized["holdings"]}

        self.assertEqual(rows_by_name["Alpha"]["source_weight_pct"], 60.0)
        self.assertEqual(rows_by_name["Beta"]["source_weight_pct"], 40.0)
        self.assertEqual(rows_by_name["Cash"]["source_weight_pct"], 10.0)
        self.assertNotEqual(
            rows_by_name["Alpha"]["source_weight_pct"],
            rows_by_name["Alpha"]["nav_weight_pct"],
        )
        for row in normalized["holdings"]:
            self.assertEqual(row["weight"], row["nav_weight_pct"])

    def test_fallback_uses_actual_date_for_canonical_filename(self) -> None:
        normalized = normalize_snapshot(koact_legacy_snapshot())

        self.assertEqual(normalized["requested_date"], "2026-07-05")
        self.assertEqual(normalized["as_of_date"], "2026-07-03")
        self.assertEqual(normalized["date"], "2026-07-03")
        self.assertEqual(normalized["actual_date"], "2026-07-03")
        self.assertEqual(normalized["quality"]["status"], "fallback")
        self.assertTrue(normalized["quality"]["requested_date_differs"])
        self.assertEqual(canonical_filename(normalized), "2026-07-03_2ETFQ1.json")

    def test_validator_rejects_duplicate_security_codes(self) -> None:
        normalized = normalize_snapshot(koact_legacy_snapshot())
        duplicate = deepcopy(normalized)
        security_rows = [
            row for row in duplicate["holdings"] if row["asset_type"] == "security"
        ]
        security_rows[1]["code"] = security_rows[0]["code"]

        issues = validate_snapshot(duplicate)

        self.assertTrue(issues)
        self.assertIn("duplicate holding code: AAA US EQUITY", issues)

        raw_duplicate = koact_legacy_snapshot()
        raw_duplicate["holdings"][1]["code"] = "AAA US EQUITY"
        with self.assertRaisesRegex(
            SnapshotValidationError, "duplicate holding code: AAA US EQUITY"
        ):
            normalize_snapshot(raw_duplicate)


class SnapshotStorageTests(unittest.TestCase):
    def test_save_is_atomic_and_idempotent(self) -> None:
        snapshot = normalize_snapshot(koact_legacy_snapshot())

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_replace = os.replace
            with mock.patch(
                "gbrc.etf.storage.os.replace", wraps=original_replace
            ) as replace:
                path, status = save_normalized_snapshot(snapshot, root=root)

            self.assertEqual(status, "created")
            self.assertEqual(path, root / "2026-07-03_2ETFQ1.json")
            self.assertEqual(replace.call_count, 1)
            self.assertEqual(load_normalized_snapshot(path), snapshot)
            self.assertEqual(list(root.glob(f".{path.name}.*.tmp")), [])

            first_bytes = path.read_bytes()
            same_path, second_status = save_normalized_snapshot(snapshot, root=root)

            self.assertEqual(same_path, path)
            self.assertEqual(second_status, "unchanged")
            self.assertEqual(path.read_bytes(), first_bytes)
            self.assertEqual(list(root.glob(f".{path.name}.*.tmp")), [])

    def test_atomic_write_failure_preserves_target_and_cleans_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "state.json"
            old_bytes = b'{"state": "old"}\n'
            target.write_bytes(old_bytes)

            with mock.patch(
                "gbrc.etf.storage.os.replace",
                side_effect=OSError("simulated replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated replace failure"):
                    atomic_write_json(target, {"state": "new"})

            self.assertEqual(target.read_bytes(), old_bytes)
            self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_save_rejects_conflicting_content_without_overwrite(self) -> None:
        original = normalize_snapshot(koact_legacy_snapshot())
        changed_raw = koact_legacy_snapshot()
        changed_raw["holdings"][0]["value_krw"] = 700
        changed = normalize_snapshot(changed_raw)
        self.assertNotEqual(original["content_sha256"], changed["content_sha256"])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path, status = save_normalized_snapshot(original, root=root)
            self.assertEqual(status, "created")
            original_bytes = path.read_bytes()

            with self.assertRaisesRegex(
                SnapshotConflictError, "refusing overwrite"
            ):
                save_normalized_snapshot(changed, root=root)

            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(load_normalized_snapshot(path), original)
            self.assertEqual(list(root.glob(f".{path.name}.*.tmp")), [])


class FundConfigurationTests(unittest.TestCase):
    def test_exactly_two_canonical_funds_are_configured(self) -> None:
        funds = load_fund_configs()

        self.assertEqual(len(funds), 2)
        self.assertEqual(set(funds), {"time", "koact"})
        self.assertEqual(funds["time"].snapshot_ticker, "456600")
        self.assertEqual(funds["time"].listing_ticker, "456600.KS")
        self.assertEqual(funds["koact"].snapshot_ticker, "2ETFQ1")
        self.assertEqual(funds["koact"].listing_ticker, "0015B0.KS")
        self.assertEqual(resolve_fund("456600"), funds["time"])
        self.assertEqual(resolve_fund("2ETFQ1"), funds["koact"])


class LegacyMigrationTests(unittest.TestCase):
    def test_migration_is_non_destructive_and_idempotent(self) -> None:
        raw = koact_legacy_snapshot()
        source_json = json.dumps(
            raw, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8") + b"\n"

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "legacy"
            destination = base / "normalized"
            quarantine = base / "quarantine"
            source.mkdir()
            source_path = source / "2026-07-05_2ETFQ1.json"
            source_path.write_bytes(source_json)
            original_mtime_ns = source_path.stat().st_mtime_ns

            first = migrate_legacy_snapshots(
                source=source,
                destination=destination,
                quarantine=quarantine,
            )

            self.assertEqual(first["expected"], 1)
            self.assertEqual(first["created"], 1)
            self.assertEqual(first["unchanged"], 0)
            self.assertEqual(first["failed"], 0)
            self.assertEqual(len(first["fallback_dates"]), 1)
            self.assertEqual(
                first["fallback_dates"][0]["normalized_file"],
                "2026-07-03_2ETFQ1.json",
            )

            canonical_path = destination / "2026-07-03_2ETFQ1.json"
            self.assertTrue(canonical_path.is_file())
            canonical_bytes = canonical_path.read_bytes()
            migrated = load_normalized_snapshot(canonical_path)
            self.assertEqual(migrated["requested_date"], "2026-07-05")
            self.assertEqual(migrated["as_of_date"], "2026-07-03")

            self.assertTrue(source_path.is_file())
            self.assertEqual(source_path.read_bytes(), source_json)
            self.assertEqual(source_path.stat().st_mtime_ns, original_mtime_ns)
            self.assertFalse(quarantine.exists())

            second = migrate_legacy_snapshots(
                source=source,
                destination=destination,
                quarantine=quarantine,
            )

            self.assertEqual(second["expected"], 1)
            self.assertEqual(second["created"], 0)
            self.assertEqual(second["unchanged"], 1)
            self.assertEqual(second["failed"], 0)
            self.assertEqual(source_path.read_bytes(), source_json)
            self.assertEqual(source_path.stat().st_mtime_ns, original_mtime_ns)
            self.assertEqual(canonical_path.read_bytes(), canonical_bytes)
            self.assertFalse(quarantine.exists())


if __name__ == "__main__":
    unittest.main()
