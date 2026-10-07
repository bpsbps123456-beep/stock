"""System-level baseline and reproducibility helpers."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .etf.storage import PROJECT_ROOT, STATE_RUNS_DIR, atomic_write_json


KST = timezone(timedelta(hours=9))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def create_baseline_manifest() -> tuple[Path, dict[str, Any]]:
    now = datetime.now(KST)
    status_lines = _git("status", "--porcelain=v1").splitlines()
    status_counts = {"modified": 0, "deleted": 0, "untracked": 0, "other": 0}
    for line in status_lines:
        code = line[:2]
        if code == "??":
            status_counts["untracked"] += 1
        elif "D" in code:
            status_counts["deleted"] += 1
        elif "M" in code:
            status_counts["modified"] += 1
        else:
            status_counts["other"] += 1

    critical_names = [
        "AGENTS.md",
        "CLAUDE.md",
        "etf_crawler.py",
        "cumulative_briefing.py",
        "config/etfs.json",
        "options_scanner.py",
        "options_tracker.py",
        "earnings_prefill.py",
        "watchlist.py",
    ]
    critical_hashes = {
        name: _sha256(PROJECT_ROOT / name)
        for name in critical_names
        if (PROJECT_ROOT / name).is_file()
    }

    data_roots = [
        PROJECT_ROOT / "data" / "etf_holdings",
        PROJECT_ROOT / "data" / "options_log",
        PROJECT_ROOT / "data" / "earnings_log",
    ]
    data_summary = {}
    for root in data_roots:
        files = sorted(path for path in root.rglob("*") if path.is_file())
        data_summary[str(root.relative_to(PROJECT_ROOT))] = {
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
            "manifest_sha256": hashlib.sha256(
                "\n".join(
                    f"{path.relative_to(root).as_posix()} {_sha256(path)}"
                    for path in files
                ).encode("utf-8")
            ).hexdigest(),
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": now.isoformat(timespec="seconds"),
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "worktree_status_counts": status_counts,
        "worktree_status": status_lines,
        "critical_file_sha256": critical_hashes,
        "data_summary": data_summary,
    }
    path = (
        STATE_RUNS_DIR.parent
        / "baselines"
        / f"{now.strftime('%Y-%m-%d_%H%M%S')}.json"
    )
    atomic_write_json(path, payload)
    return path, payload
