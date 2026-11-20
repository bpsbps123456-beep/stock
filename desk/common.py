"""Shared helpers for every desk module: JSON-safe values, atomic and write-once files, cache reads.

One implementation each (AGENTS.md: no second canonical module for the same job). Calendar and
record locations live in config.py.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def clean(obj):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def replacing(path: Path):
    """Yield a temporary path next to `path`; on success it replaces `path` in one step.

    Readers never see a half-written file, and a crash leaves the old file in place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        yield Path(temp)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _dump(data, indent: int | None, allow_nan: bool) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=indent, allow_nan=allow_nan)
    return text + "\n" if indent is not None else text


def write_atomic(path: Path, data, *, indent: int | None = 2, allow_nan: bool = False) -> None:
    """Write JSON through a temporary file and os.replace (caches, ledgers, the trade journal)."""
    text = _dump(data, indent, allow_nan)
    with replacing(path) as temp:
        with open(temp, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())


def write_once(path: Path, data, *, indent: int | None = 2) -> bool:
    """Publish a complete JSON file only if `path` does not exist yet. Returns True if this call wrote it.

    For records locked once (pre-open observations, published outlooks, month baskets): two desk
    processes racing keep the first file, and nobody ever sees a partial one."""
    path = Path(path)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _dump(data, indent, False)
    with tempfile.TemporaryDirectory(dir=path.parent) as folder:
        draft = Path(folder) / "draft.json"
        with open(draft, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            if os.name == "nt":
                draft.rename(path)  # Windows rename is atomic and refuses an existing target.
            else:
                os.link(draft, path)
        except FileExistsError:
            return False
    return True


def read_cache(path: Path) -> dict | None:
    """A regenerable cache file, or None when it is missing or unreadable (then the caller recomputes).

    Never use this for records that cannot be rebuilt: a broken record must raise, not be overwritten."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None
