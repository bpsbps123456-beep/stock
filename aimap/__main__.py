"""py -m aimap [build|serve|deploy] — see aimap/__init__.py."""

from __future__ import annotations

import functools
import http.server
import mimetypes
import shutil
import subprocess
import sys

from .build import DIST, PROJECT, build

PORT = 8330


def _build() -> dict:
    r = build()
    print(f"[aimap] {r['out']} · v{r['version']} · 파일 {r['files']}개 · {r['bytes'] / 1e6:.1f}MB · "
          f"일터 {r['counts']['stops']} · 분야 {r['counts']['zones']} · 회사 {r['counts']['companies']}")
    for p in r["problems"]:
        print(f"[aimap] 문제: {p}")
    return r


def serve() -> None:
    mimetypes.add_type("application/manifest+json", ".webmanifest")
    mimetypes.add_type("image/webp", ".webp")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DIST))
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
        print(f"[aimap] http://127.0.0.1:{PORT}/ (Ctrl+C로 종료)")
        httpd.serve_forever()


def deploy() -> int:
    r = _build()
    if r["problems"]:
        print("[aimap] 문제를 먼저 고치세요. 배포하지 않았습니다.")
        return 1
    vercel = shutil.which("vercel")
    if not vercel:
        print("[aimap] Vercel CLI가 없습니다: npm i -g vercel 후 vercel login")
        return 1
    if not (DIST / ".vercel" / "project.json").exists():
        subprocess.run([vercel, "link", "--yes", "--project", PROJECT, "--cwd", str(DIST)], check=True)
    return subprocess.run([vercel, "deploy", "--prod", "--yes", "--cwd", str(DIST)]).returncode


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "build"
    if cmd == "build":
        return 1 if _build()["problems"] else 0
    if cmd == "serve":
        if not (DIST / "index.html").exists():
            _build()
        serve()
        return 0
    if cmd == "deploy":
        return deploy()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
