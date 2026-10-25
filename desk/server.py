"""Local web server for the desk (standard library only). Binds to 127.0.0.1."""

from __future__ import annotations

import json
import mimetypes
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import config, dip, giant, journal, news, tour, trend, value
from .state import Desk, clean

DESK = Desk()
APP = "night-desk"


class DeskServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second process bind a port that is already
    # listening, and requests then split between the two servers. Bind exclusively.
    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    server_version = "NightDesk/1.0"

    def log_message(self, fmt, *args):  # keep the console quiet except for errors
        return

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200) -> None:
        self._send(status, json.dumps(clean(payload), ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise journal.JournalError("요청 내용을 읽지 못했습니다 (UTF-8 JSON이어야 합니다)") from exc
        if not isinstance(payload, dict):
            raise journal.JournalError("요청 형식이 올바르지 않습니다")
        return payload

    def _guard(self, fn) -> None:
        try:
            fn()
        except (journal.JournalError, dip.DipError, trend.TrendError, news.NewsError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            path = "/static/index.html"
        if path == "/api/ping":
            return self._json({"app": APP, "journal": str(config.JOURNAL_DIR)})
        if path == "/api/state":
            return self._guard(lambda: self._json(DESK.state()))
        if path == "/api/news":
            q = parse_qs(urlparse(self.path).query)
            return self._guard(lambda: self._json(news.payload(
                as_of=(q.get("as_of") or [None])[0], basis=(q.get("basis") or ["public"])[0],
                force=q.get("refresh") == ["1"])))
        if path == "/api/tour":  # AI 탐방: hand-made route + latest fund PDFs, no prices needed
            return self._guard(lambda: self._json(tour.payload()))
        if path == "/api/value":  # 저평가 표시(§8.4): 6시간 캐시, ?refresh=1이면 다시 계산(20~40초)
            force = parse_qs(urlparse(self.path).query).get("refresh") == ["1"]
            return self._guard(lambda: self._json(value.payload(force=force)))
        if path == "/api/giant":  # 거인의 어깨: 60분 캐시, ?refresh=1이면 다시 계산(30~60초)
            force = parse_qs(urlparse(self.path).query).get("refresh") == ["1"]
            return self._guard(lambda: self._json(giant.payload(force=force)))
        if path.startswith("/api/trend"):  # 추세 등급(§11): 60분 캐시, ?refresh=1이면 다시 계산(10~20초)
            q = parse_qs(urlparse(self.path).query)
            if path == "/api/trend":
                return self._guard(lambda: self._json(trend.payload(force=q.get("refresh") == ["1"])))
            if path == "/api/trend/chart":
                return self._guard(lambda: self._json(trend.chart((q.get("t") or [""])[0])))
        if path.startswith("/api/dip"):  # 급락 후 담기: 목록 2분 캐시, 종목 3분봉 재생, 뉴스
            q = parse_qs(urlparse(self.path).query)
            arg = lambda k: (q.get(k) or [None])[0]
            if path == "/api/dip":
                return self._guard(lambda: self._json(dip.payload(force=arg("refresh") == "1")))
            if path == "/api/dip/detail":
                return self._guard(lambda: self._json(dip.detail(arg("t") or "", arg("day"))))
            if path == "/api/dip/news":
                return self._guard(lambda: self._json({"t": arg("t"), "items": dip.news(arg("t") or "")}))
        if path.startswith("/static/"):
            target = (config.STATIC / path[len("/static/"):]).resolve()
            if config.STATIC.resolve() in target.parents and target.is_file():
                ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                if ctype.startswith("text/") or ctype.endswith("javascript"):
                    ctype += "; charset=utf-8"
                return self._send(200, target.read_bytes(), ctype)
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path

        def run():
            if path == "/api/refresh":
                DESK.ensure(force=bool(self._body().get("prices")))
                return self._json(DESK.state(force_quotes=True))
            if path == "/api/trades":
                entry = journal.add_entry(self._body())
                DESK.ensure()
                return self._json({"entry": entry, "state": DESK.state()})
            if path == "/api/settings":
                journal.save_settings(self._body())
                return self._json(DESK.state())
            if path == "/api/update-data":
                result = DESK.run_briefing()
                return self._json({"briefing": result, "state": DESK.state(force_quotes=True)})
            self._json({"error": "not found"}, 404)

        self._guard(run)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path

        def run():
            if path.startswith("/api/trades/"):
                if not journal.delete_entry(path.rsplit("/", 1)[-1]):
                    return self._json({"error": "기록을 찾지 못했습니다"}, 404)
                return self._json({"state": DESK.state()})
            self._json({"error": "not found"}, 404)

        self._guard(run)


def running_desk(ports=None) -> str | None:
    """URL of a desk already serving this journal, so a second launch reuses it."""
    for port in filter(None, ports or config.PORTS):
        try:  # a closed localhost port on Windows costs the whole HTTP timeout (1.5s); probe first
            socket.create_connection((config.HOST, port), timeout=0.3).close()
        except OSError:
            continue
        url = f"http://{config.HOST}:{port}/"
        try:
            with urllib.request.urlopen(url + "api/ping", timeout=1.5) as resp:
                info = json.loads(resp.read().decode("utf-8"))
        except (OSError, ValueError):
            continue
        if info.get("app") == APP and info.get("journal") == str(config.JOURNAL_DIR):
            return url
    return None


def _pinned(argv: list[str]) -> list[int]:
    """--port=N pins one port (tests keep off the user's 8321)."""
    return [int(a.split("=", 1)[1]) for a in argv if a.startswith("--port=")]


def _reuse(url: str, argv: list[str]) -> int:
    print(f"야간 데스크가 이미 켜져 있습니다: {url}", flush=True)
    if "--no-browser" not in argv:
        webbrowser.open(url)
    return 0


def _warm() -> None:
    """Prices and the 7종목 signal, computed while the page is already open. /api/state waits on
    DESK's lock, so the first view fills in as soon as this is done."""
    started = time.time()
    try:
        DESK.ensure()
    except Exception:  # the page reports the same error when it asks for /api/state
        print("계산 실패 — 화면을 새로 고치면 다시 시도합니다.\n" + traceback.format_exc(limit=3), flush=True)
    else:
        print(f"준비 완료 ({time.time() - started:.0f}초)", flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[2:] if argv is None else argv
    if "--launcher" in argv:
        # Started from 야간데스크.cmd: in a console, Ctrl+C without a selection is "stop", not
        # "copy", and it once shut the desk down mid-evening. Only closing the window stops it.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW("야간 데스크 — 이 창을 닫으면 꺼집니다")
    existing = running_desk()
    if existing:
        return _reuse(existing, argv)
    server = None
    pinned = _pinned(argv)
    for port in pinned or config.PORTS:  # 0 = let Windows pick any free port
        try:
            server = DeskServer((config.HOST, port), Handler)
            break
        except OSError as exc:
            existing = None if pinned else running_desk()
            if existing:  # a double-click a moment earlier won the port for this journal
                return _reuse(existing, argv)
            print(f"포트 {port} 사용 불가 ({exc.strerror}) — 다음 포트를 시도합니다", flush=True)
    if server is None:
        return 1
    url = f"http://{config.HOST}:{server.server_address[1]}/"
    # The port opens before the prices load: a first launch after the US close spends a while
    # downloading, and a bare console window meanwhile looked like a desk that never started.
    print("가격과 7종목 신호를 계산합니다 — 화면은 먼저 열리고, 계산이 끝나면 채워집니다.", flush=True)
    threading.Thread(target=_warm, name="desk-warm", daemon=True).start()
    if "--launcher" in argv:
        print(f"열기: {url}\n이 창은 최소화해 두세요. 끄려면 창을 닫으면 됩니다.", flush=True)
    else:
        print(f"열기: {url}  (종료: Ctrl+C)", flush=True)
    if "--no-browser" not in argv:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


LAUNCH_WAIT_SEC = 60


def launch(argv: list[str] | None = None) -> int:
    """Double-click entry (야간데스크.cmd and the desktop shortcut '야간 데스크'): reuse a running
    desk, or start one in its own minimized window and wait until it answers; then open the page
    and let this window close — the shape of the user's other local-site shortcuts."""
    argv = sys.argv[2:] if argv is None else argv
    ports = _pinned(argv) or None
    url = running_desk(ports)
    if url:
        print("야간 데스크가 이미 켜져 있어 화면만 엽니다.", flush=True)
    else:
        print("야간 데스크를 켭니다 — 서버 창은 작업 표시줄에 최소화됩니다(그 창을 닫으면 꺼짐).", flush=True)
        kwargs = {}
        if os.name == "nt":
            show = subprocess.STARTUPINFO()
            show.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            show.wShowWindow = 7  # SW_SHOWMINNOACTIVE: the page, not the console, ends up in front
            kwargs = {"creationflags": subprocess.CREATE_NEW_CONSOLE, "startupinfo": show}
        proc = subprocess.Popen([sys.executable, "-m", "desk", "serve", "--launcher", "--no-browser", *argv],
                                cwd=config.ROOT, **kwargs)
        deadline = time.time() + LAUNCH_WAIT_SEC
        while not (url := running_desk(ports)):
            if proc.poll() is not None or time.time() > deadline:
                print("야간 데스크가 켜지지 않았습니다. 작업 표시줄의 '야간 데스크' 창에 오류가 있는지 보고,"
                      " 없으면 이 창을 닫고 다시 실행하세요.", flush=True)
                return 1
            time.sleep(0.5)
    if "--no-browser" not in argv:
        webbrowser.open(url)
    print(f"열었습니다: {url}", flush=True)
    return 0
