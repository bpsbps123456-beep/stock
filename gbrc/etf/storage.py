"""Atomic, idempotent storage for canonical ETF snapshots and run artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .snapshot import canonical_filename, validate_snapshot


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_HOLDINGS_DIR = PROJECT_ROOT / "data" / "etf_holdings"
NORMALIZED_HOLDINGS_DIR = PROJECT_ROOT / "data" / "normalized" / "etf_holdings"
RAW_HOLDINGS_DIR = PROJECT_ROOT / "data" / "raw" / "etf_holdings"
DERIVED_BRIEFINGS_DIR = PROJECT_ROOT / "data" / "derived" / "briefings"
STATE_RUNS_DIR = PROJECT_ROOT / "data" / "state" / "runs"
QUARANTINE_DIR = PROJECT_ROOT / "data" / "quarantine"


class SnapshotConflictError(RuntimeError):
    pass


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def save_normalized_snapshot(
    snapshot: dict[str, Any],
    root: Path = NORMALIZED_HOLDINGS_DIR,
) -> tuple[Path, str]:
    issues = validate_snapshot(snapshot)
    if issues:
        raise ValueError("Invalid ETF snapshot: " + "; ".join(issues))

    path = root / canonical_filename(snapshot)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("content_sha256") == snapshot.get("content_sha256"):
            return path, "unchanged"
        raise SnapshotConflictError(
            f"Different content already exists for {path.name}; refusing overwrite"
        )

    atomic_write_json(path, snapshot)
    return path, "created"


def save_raw_capture(
    raw_snapshot: dict[str, Any],
    root: Path = RAW_HOLDINGS_DIR,
) -> tuple[Path | None, str]:
    payload = raw_snapshot.get("_raw_payload")
    if payload is None:
        return None, "unavailable"

    fund_id = str(raw_snapshot.get("etf_ticker") or "unknown")
    requested_date = str(raw_snapshot.get("date") or "unknown-date")
    if isinstance(payload, str):
        import hashlib

        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        path = root / f"{requested_date}_{fund_id}_{digest[:12]}.html"
        if path.exists():
            return path, "unchanged"
        atomic_write_text(path, payload)
    else:
        import hashlib

        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        path = root / f"{requested_date}_{fund_id}_{digest[:12]}.json"
        if path.exists():
            return path, "unchanged"
        atomic_write_json(path, payload)
    return path, "created"


def load_normalized_snapshot(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    issues = validate_snapshot(snapshot)
    if issues:
        raise ValueError(f"Invalid snapshot {path}: {'; '.join(issues)}")
    return snapshot
