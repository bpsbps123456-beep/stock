"""AI 탐방 cutaway pictures (단면도): Blender-rendered panels plus a label layout.

desk/scenes/blender/ models and renders each panel (needs Blender, see build.py); it writes the
image to desk/static/scenes/ and the zone anchors to desk/scenes/anchors/. layout.py places the
labels and writes desk/static/scenes/scenes.json, which /api/tour hands to the page.
Rules: docs/desk_system.md §8.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANCHORS = HERE / "anchors"
STATIC = HERE.parent / "static" / "scenes"
SCENES_JSON = STATIC / "scenes.json"


def load(path: Path = SCENES_JSON) -> dict:
    """The laid-out scenes for the page ({} before the first build)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
