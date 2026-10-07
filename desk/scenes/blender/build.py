"""Render the AI 탐방 cutaway panels with Blender (Cycles), then lay the labels out.

    blender -b --factory-startup -P desk/scenes/blender/build.py -- [panel ...] [--preview] [--out DIR]
    py -m desk.scenes.layout

Blender: "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe". No panel names = all.
Writes desk/static/scenes/<panel>.webp (2x pixels, transparent background) and
desk/scenes/anchors/<panel>.json (zone anchors, hover masks, occupancy). --preview renders a
quick half-size image to --out (default: the system temp folder) and exports nothing.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # repo root
sys.path.insert(0, HERE)

import kit  # noqa: E402

MODULES = ("power", "fab", "server", "dc")


def panels():
    out = {}
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        out.update(mod.PANELS)
    return out


def main(argv):
    preview = "--preview" in argv
    out_dir = None
    if "--out" in argv:
        out_dir = argv[argv.index("--out") + 1]
    names = [a for a in argv if not a.startswith("--") and a != out_dir]
    table = panels()
    for name in names or list(table):
        build = table[name]
        t0 = time.time()
        K = kit.Kit(name)
        spec = build(K) or {}
        width = spec.get("width", 1200)
        K.camera(rot=spec.get("rot", (60, 0, 45)), width=width, margin=spec.get("margin", (0.05, 0.08)),
                 aspect=spec.get("aspect"))
        K.lights(**spec.get("light", {}))
        if preview:
            d = out_dir or tempfile.gettempdir()
            K.sc.render.resolution_percentage = 50
            K.render(os.path.join(d, f"{name}_preview.webp"), samples=spec.get("preview_samples", 24), line=0.8)
            print(f"[scenes] preview {name} -> {d} ({time.time() - t0:.0f}s)")
            continue
        img_dir = out_dir or os.path.join(ROOT, "desk", "static", "scenes")
        anc_dir = out_dir or os.path.join(ROOT, "desk", "scenes", "anchors")
        img = os.path.join(img_dir, f"{name}.webp")
        K.render(img, samples=spec.get("samples", 128), line=spec.get("line", 1.3))
        tmp = os.path.join(tempfile.gettempdir(), f"desk_scene_{name}_id.png")
        data = K.export(os.path.join(anc_dir, f"{name}.json"), tmp, f"{name}.webp")
        hidden = [s["zone"] for s in data["spots"] if s.get("hidden")]
        print(f"[scenes] {name}: {data['w']}x{data['h']} · {len(data['spots'])} zones"
              + (f" · HIDDEN {hidden}" if hidden else "") + f" ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
