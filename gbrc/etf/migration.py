"""Non-destructive migration from legacy ETF snapshots to schema v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .snapshot import SnapshotValidationError, normalize_snapshot
from .storage import (
    LEGACY_HOLDINGS_DIR,
    NORMALIZED_HOLDINGS_DIR,
    QUARANTINE_DIR,
    SnapshotConflictError,
    atomic_write_json,
    save_normalized_snapshot,
)


def migrate_legacy_snapshots(
    source: Path = LEGACY_HOLDINGS_DIR,
    destination: Path = NORMALIZED_HOLDINGS_DIR,
    quarantine: Path = QUARANTINE_DIR / "etf_migration",
) -> dict[str, Any]:
    files = sorted(source.glob("????-??-??_*.json"))
    summary: dict[str, Any] = {
        "source": str(source),
        "destination": str(destination),
        "expected": len(files),
        "created": 0,
        "unchanged": 0,
        "failed": 0,
        "fallback_dates": [],
        "failures": [],
    }

    for path in files:
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            normalized = normalize_snapshot(raw)
            target, status = save_normalized_snapshot(normalized, destination)
            summary[status] += 1
            if normalized["requested_date"] != normalized["as_of_date"]:
                summary["fallback_dates"].append({
                    "source": path.name,
                    "requested_date": normalized["requested_date"],
                    "as_of_date": normalized["as_of_date"],
                    "normalized_file": target.name,
                })
        except (
            OSError,
            json.JSONDecodeError,
            SnapshotValidationError,
            SnapshotConflictError,
            ValueError,
        ) as exc:
            summary["failed"] += 1
            failure = {
                "source": str(path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            summary["failures"].append(failure)
            atomic_write_json(quarantine / f"{path.stem}.error.json", failure)

    return summary


def validate_normalized_directory(
    root: Path = NORMALIZED_HOLDINGS_DIR,
) -> dict[str, Any]:
    from .storage import load_normalized_snapshot

    files = sorted(root.glob("????-??-??_*.json"))
    summary: dict[str, Any] = {
        "root": str(root),
        "files": len(files),
        "valid": 0,
        "invalid": 0,
        "funds": {},
        "failures": [],
    }
    seen_keys: set[tuple[str, str]] = set()

    for path in files:
        try:
            snapshot = load_normalized_snapshot(path)
            expected_name = f"{snapshot['as_of_date']}_{snapshot['fund_id']}.json"
            if path.name != expected_name:
                raise ValueError(
                    f"filename mismatch: expected {expected_name}, got {path.name}"
                )
            key = (snapshot["fund_id"], snapshot["as_of_date"])
            if key in seen_keys:
                raise ValueError(f"duplicate fund/as_of_date: {key}")
            seen_keys.add(key)
            summary["valid"] += 1
            fund = summary["funds"].setdefault(snapshot["fund_id"], {
                "files": 0,
                "min_date": snapshot["as_of_date"],
                "max_date": snapshot["as_of_date"],
            })
            fund["files"] += 1
            fund["min_date"] = min(fund["min_date"], snapshot["as_of_date"])
            fund["max_date"] = max(fund["max_date"], snapshot["as_of_date"])
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            summary["invalid"] += 1
            summary["failures"].append({
                "file": str(path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    return summary
