"""Read-only data-quality audits for GBRC's on-demand JSON logs.

The functions in this module deliberately return plain dictionaries, lists,
strings, integers, and booleans so that callers can serialize the result with
``json.dumps`` without a custom encoder.  Audits never write to a log file.
"""

from __future__ import annotations

import json
import re
import runpy
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


_DATE_RE = r"\d{4}-\d{2}-\d{2}"
_OPTIONS_FILENAME_RE = re.compile(rf"^(?P<date>{_DATE_RE})\.json$")
_EARNINGS_FILENAME_RE = re.compile(
    rf"^(?P<date>{_DATE_RE})_(?P<ticker>[A-Za-z0-9.-]+)\.json$"
)


def _log_dir(root: str | Path, name: str) -> Path:
    """Accept either a project root, ``data`` directory, or log directory."""

    path = Path(root).expanduser()
    if path.name == name:
        return path
    if path.name == "data":
        return path / name
    return path / "data" / name


def _project_root(root: str | Path, log_dir: Path) -> Path:
    """Find the nearest project root containing the canonical watchlist."""

    candidates = [Path(root).expanduser(), log_dir, *log_dir.parents]
    for candidate in candidates:
        if (candidate / "watchlist.py").is_file():
            return candidate
    return Path(root).expanduser()


def _failure(code: str, file: str | None, message: str) -> dict[str, str]:
    result = {"code": code, "message": message}
    if file is not None:
        result["file"] = file
    return result


def _read_json(path: Path) -> tuple[Any | None, dict[str, str] | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, {
            "file": path.name,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def _signature(payload: Any) -> tuple[str, ...]:
    if isinstance(payload, dict):
        return tuple(sorted(str(key) for key in payload))
    return (f"<{type(payload).__name__}>",)


def _signature_summary(counter: Counter[tuple[str, ...]]) -> list[dict[str, Any]]:
    return [
        {"keys": list(keys), "count": count}
        for keys, count in sorted(counter.items(), key=lambda item: item[0])
    ]


def _ticker_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        yield from value.keys()
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            if isinstance(item, str):
                yield item


def _expected_watchlist(project_root: Path) -> tuple[list[str], dict[str, str] | None]:
    path = project_root / "watchlist.py"
    if not path.is_file():
        return [], _failure(
            "watchlist_missing", None, f"Canonical watchlist not found: {path}"
        )

    try:
        namespace = runpy.run_path(str(path))
    except Exception as exc:  # The audit must report a broken watchlist, not crash.
        return [], _failure(
            "watchlist_load_error", path.name, f"{type(exc).__name__}: {exc}"
        )

    tickers: set[str] = set()
    for name in ("PROFIT_TICKERS", "LOSS_TICKERS"):
        tickers.update(ticker.strip().upper() for ticker in _ticker_values(namespace.get(name)))

    if not tickers:
        tickers.update(
            ticker.strip().upper()
            for ticker in _ticker_values(namespace.get("TICKER_TO_SECTOR"))
        )

    if not tickers:
        sector_map = namespace.get("SECTOR_MAP")
        if isinstance(sector_map, dict):
            for members in sector_map.values():
                tickers.update(ticker.strip().upper() for ticker in _ticker_values(members))

    tickers.discard("")
    if not tickers:
        return [], _failure(
            "watchlist_empty", path.name, "No ticker symbols were found in the canonical watchlist"
        )
    return sorted(tickers), None


def _status(*, files: int, legacy: bool, needs_migration: bool) -> str:
    if files == 0:
        return "empty" if not needs_migration else "needs_migration"
    return "needs_migration" if needs_migration else ("legacy" if legacy else "current")


def audit_options_logs(root: str | Path) -> dict[str, Any]:
    """Audit option-ranking logs without modifying them.

    ``root`` may point to the GBRC project, its ``data`` directory, or directly
    to ``data/options_log``.
    """

    directory = _log_dir(root, "options_log")
    files = sorted(directory.glob("*.json")) if directory.is_dir() else []
    failures: list[dict[str, str]] = []
    parse_failures: list[dict[str, str]] = []
    signatures: Counter[tuple[str, ...]] = Counter()
    score_versions: Counter[str] = Counter()
    score_version_missing = 0
    entry_count = 0
    valid_entry_count = 0
    schema_version_missing_files: list[str] = []
    filename_mismatches: list[dict[str, Any]] = []
    parsed_logs: list[tuple[str, Path, dict[str, Any], list[dict[str, Any]]]] = []

    if not directory.is_dir():
        failures.append(
            _failure("log_directory_missing", None, f"Options log directory not found: {directory}")
        )

    for path in files:
        payload, parse_error = _read_json(path)
        if parse_error:
            parse_failures.append(parse_error)
            failures.append(_failure("parse_error", path.name, parse_error["message"]))
            continue

        signatures[_signature(payload)] += 1
        if not isinstance(payload, dict):
            failures.append(
                _failure(
                    "top_level_not_object",
                    path.name,
                    "Top-level JSON value must be an object",
                )
            )
            continue

        if payload.get("schema_version") in (None, ""):
            schema_version_missing_files.append(path.name)

        match = _OPTIONS_FILENAME_RE.fullmatch(path.name)
        payload_date = payload.get("date")
        mismatch_fields: list[str] = []
        filename_date: str | None = None
        if match is None:
            mismatch_fields.append("filename_pattern")
        else:
            filename_date = match.group("date")
            if payload_date != filename_date:
                mismatch_fields.append("date")
        if mismatch_fields:
            filename_mismatches.append(
                {
                    "file": path.name,
                    "fields": mismatch_fields,
                    "filename_date": filename_date,
                    "payload_date": payload_date,
                }
            )
            failures.append(
                _failure(
                    "filename_mismatch",
                    path.name,
                    "Filename date does not match the payload date or expected pattern",
                )
            )

        rankings = payload.get("rankings")
        if not isinstance(rankings, list):
            failures.append(
                _failure("rankings_not_list", path.name, "The rankings field must be a list")
            )
            continue

        entry_count += len(rankings)
        valid_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rankings):
            if not isinstance(row, dict):
                score_version_missing += 1
                failures.append(
                    _failure(
                        "entry_not_object",
                        path.name,
                        f"rankings[{index}] must be an object",
                    )
                )
                continue
            valid_entry_count += 1
            valid_rows.append(row)
            version = row.get("score_version")
            if version in (None, ""):
                score_version_missing += 1
            else:
                score_versions[str(version)] += 1

        sort_date = str(payload_date) if isinstance(payload_date, str) else (filename_date or "")
        parsed_logs.append((sort_date, path, payload, valid_rows))

    expected_tickers, watchlist_failure = _expected_watchlist(
        _project_root(root, directory)
    )
    if watchlist_failure:
        failures.append(watchlist_failure)

    latest_log: dict[str, Any] | None = None
    latest_missing_tickers: list[str] = []
    latest_extra_tickers: list[str] = []
    if parsed_logs:
        sort_date, latest_path, latest_payload, latest_rows = max(
            parsed_logs, key=lambda item: (item[0], item[1].name)
        )
        del sort_date
        observed_tickers = sorted(
            {
                str(row["ticker"]).strip().upper()
                for row in latest_rows
                if row.get("ticker") not in (None, "")
            }
        )
        latest_missing_tickers = sorted(set(expected_tickers) - set(observed_tickers))
        latest_extra_tickers = sorted(set(observed_tickers) - set(expected_tickers))
        latest_log = {
            "file": latest_path.name,
            "date": latest_payload.get("date"),
            "entry_count": len(latest_rows),
            "ticker_count": len(observed_tickers),
        }

    legacy = bool(schema_version_missing_files or score_version_missing)
    needs_migration = bool(legacy or failures or filename_mismatches)
    return {
        "domain": "options",
        "status": _status(files=len(files), legacy=legacy, needs_migration=needs_migration),
        "legacy": legacy,
        "needs_migration": needs_migration,
        "log_directory": str(directory),
        "file_count": len(files),
        "parsed_file_count": len(files) - len(parse_failures),
        "entry_count": entry_count,
        "valid_entry_count": valid_entry_count,
        "parse_failures": parse_failures,
        "top_level_schema_signatures": _signature_summary(signatures),
        "schema_version_missing": len(schema_version_missing_files),
        "schema_version_missing_files": schema_version_missing_files,
        "score_versions": dict(sorted(score_versions.items())),
        "score_version_missing": score_version_missing,
        "expected_watchlist_count": len(expected_tickers),
        "latest_log": latest_log,
        "latest_missing_tickers": latest_missing_tickers,
        "latest_extra_tickers": latest_extra_tickers,
        "filename_mismatch_count": len(filename_mismatches),
        "filename_mismatches": filename_mismatches,
        "failures": failures,
    }


def audit_earnings_logs(root: str | Path) -> dict[str, Any]:
    """Audit earnings-analysis logs without modifying them.

    ``root`` may point to the GBRC project, its ``data`` directory, or directly
    to ``data/earnings_log``.
    """

    directory = _log_dir(root, "earnings_log")
    files = sorted(directory.glob("*.json")) if directory.is_dir() else []
    failures: list[dict[str, str]] = []
    parse_failures: list[dict[str, str]] = []
    signatures: Counter[tuple[str, ...]] = Counter()
    entry_count = 0
    schema_version_missing_files: list[str] = []
    filename_mismatches: list[dict[str, Any]] = []
    verification_counts: Counter[str] = Counter()
    verification_files: dict[str, list[str]] = {
        "dict": [],
        "null": [],
        "missing": [],
        "other": [],
    }

    if not directory.is_dir():
        failures.append(
            _failure(
                "log_directory_missing",
                None,
                f"Earnings log directory not found: {directory}",
            )
        )

    for path in files:
        payload, parse_error = _read_json(path)
        if parse_error:
            parse_failures.append(parse_error)
            failures.append(_failure("parse_error", path.name, parse_error["message"]))
            continue

        signatures[_signature(payload)] += 1
        if not isinstance(payload, dict):
            failures.append(
                _failure(
                    "top_level_not_object",
                    path.name,
                    "Top-level JSON value must be an object",
                )
            )
            continue

        entry_count += 1
        if payload.get("schema_version") in (None, ""):
            schema_version_missing_files.append(path.name)

        if "verification" not in payload:
            verification_state = "missing"
        elif isinstance(payload["verification"], dict):
            verification_state = "dict"
        elif payload["verification"] is None:
            verification_state = "null"
        else:
            verification_state = "other"
        verification_counts[verification_state] += 1
        verification_files[verification_state].append(path.name)

        match = _EARNINGS_FILENAME_RE.fullmatch(path.name)
        payload_date = payload.get("analysis_date", payload.get("date"))
        payload_ticker_raw = payload.get("ticker")
        payload_ticker = (
            str(payload_ticker_raw).strip().upper()
            if payload_ticker_raw not in (None, "")
            else None
        )
        mismatch_fields: list[str] = []
        filename_date: str | None = None
        filename_ticker: str | None = None
        if match is None:
            mismatch_fields.append("filename_pattern")
        else:
            filename_date = match.group("date")
            filename_ticker = match.group("ticker").upper()
            if payload_date != filename_date:
                mismatch_fields.append("date")
            if payload_ticker != filename_ticker:
                mismatch_fields.append("ticker")
        if mismatch_fields:
            filename_mismatches.append(
                {
                    "file": path.name,
                    "fields": mismatch_fields,
                    "filename_date": filename_date,
                    "payload_date": payload_date,
                    "filename_ticker": filename_ticker,
                    "payload_ticker": payload_ticker,
                }
            )
            failures.append(
                _failure(
                    "filename_mismatch",
                    path.name,
                    "Filename date/ticker does not match the payload or expected pattern",
                )
            )

    verification_summary = {
        state: {
            "count": verification_counts.get(state, 0),
            "files": verification_files[state],
        }
        for state in ("dict", "null", "missing", "other")
    }
    legacy = bool(schema_version_missing_files)
    needs_migration = bool(
        legacy
        or failures
        or filename_mismatches
        or verification_counts.get("missing", 0)
        or verification_counts.get("other", 0)
    )
    return {
        "domain": "earnings",
        "status": _status(files=len(files), legacy=legacy, needs_migration=needs_migration),
        "legacy": legacy,
        "needs_migration": needs_migration,
        "log_directory": str(directory),
        "file_count": len(files),
        "parsed_file_count": len(files) - len(parse_failures),
        "entry_count": entry_count,
        "parse_failures": parse_failures,
        "top_level_schema_signatures": _signature_summary(signatures),
        "schema_version_missing": len(schema_version_missing_files),
        "schema_version_missing_files": schema_version_missing_files,
        "verification": verification_summary,
        "verification_dict_count": verification_counts.get("dict", 0),
        "verification_not_dict_count": entry_count - verification_counts.get("dict", 0),
        "filename_mismatch_count": len(filename_mismatches),
        "filename_mismatches": filename_mismatches,
        "failures": failures,
    }


def audit_all(project_root: str | Path) -> dict[str, Any]:
    """Return a combined, JSON-serializable audit for both on-demand domains."""

    options = audit_options_logs(project_root)
    earnings = audit_earnings_logs(project_root)
    combined_failures: list[dict[str, str]] = []
    for domain, summary in (("options", options), ("earnings", earnings)):
        for item in summary["failures"]:
            combined_failures.append({"domain": domain, **item})

    legacy = bool(options["legacy"] or earnings["legacy"])
    needs_migration = bool(options["needs_migration"] or earnings["needs_migration"])
    file_count = int(options["file_count"]) + int(earnings["file_count"])
    entry_count = int(options["entry_count"]) + int(earnings["entry_count"])
    return {
        "status": _status(
            files=file_count,
            legacy=legacy,
            needs_migration=needs_migration,
        ),
        "legacy": legacy,
        "needs_migration": needs_migration,
        "project_root": str(Path(project_root).expanduser()),
        "file_count": file_count,
        "entry_count": entry_count,
        "parse_failure_count": len(options["parse_failures"])
        + len(earnings["parse_failures"]),
        "options": options,
        "earnings": earnings,
        "failures": combined_failures,
    }


__all__ = ["audit_options_logs", "audit_earnings_logs", "audit_all"]
