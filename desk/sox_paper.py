"""Frozen, forward-only SOXL/SOXS paper records; desk_system.md §16.

The research engine and strategy classes are reused, never reimplemented here.
Only the completed opening window is replayed, from unfilled Alpaca SIP bars.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests

from . import config
from .news import market_calendar

RESEARCH = config.ROOT / "data/derived/analysis/sox_open_switch"
STORE = Path(os.environ.get("SOX_PAPER_DIR") or RESEARCH / "paper")
EARLIEST_START = "2026-11-05"
REVIEW_SESSIONS = 60
SOURCES = ("code/lib.py", "code/fam_F1.py", "code/h0.py", "cost_table.json")
F1A = {"N": 1, "entry": "confirm", "tgt": 2, "stop": None, "rev": False, "filt": None}
S2 = {"ref": "low", "thr": .03, "target": .01, "stop": None, "sides": [-1]}
RULES = [
    {"id": "S1", "name": "첫 1분봉 방향 돌파 F1-A", "conditions": [
        "SOXL 첫 1분봉이 양봉이면 SOXL, 음봉이면 SOXS; 같은 방향 범위 밖 종가 확인",
        "다음 1분봉 종가에 모의 진입 · 목표는 첫 봉 반대 끝까지 거리의 2배(2R) · 손절 없음 · 하루 한 번"]},
    {"id": "S2", "name": "SOXL 3% 상승 후 SOXS", "conditions": [
        "SOXL 장중 저점 대비 종가 +3%면 다음 1분봉 종가에 SOXS 모의 진입",
        "목표 +1% · 손절 없음 · 마감 15분 전까지 결정 · 청산 다음 봉부터 저점 재설정"]},
]
_LOCK = threading.RLock()
_MODULES = None


class PaperError(ValueError):
    pass


def now_utc():
    return datetime.now(timezone.utc)


def _iso(now):
    return now.astimezone(timezone.utc).isoformat()


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def _write_lock():
    """OS-owned lock is released even after closing a server window or a reboot."""
    with _LOCK:
        STORE.mkdir(parents=True, exist_ok=True)
        path = STORE / ".write.lock"
        handle = path.open("a+b")
        try:
            if path.stat().st_size == 0:
                handle.write(b" ")
                handle.flush()
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise PaperError("종이 기록 갱신이 이미 진행 중입니다") from exc
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _source_hashes():
    return {p: _file_digest(RESEARCH / p) for p in SOURCES}


def _file_digest(path):
    # Git autocrlf and Windows/Linux checkouts must describe the same source.
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()


def _next_open_session(now):
    cal = market_calendar()
    day = max(date.fromisoformat(EARLIEST_START), now.astimezone(config.NY).date())
    for _ in range(370):
        if cal.is_session(day.isoformat()):
            opening = datetime.combine(day, time(9, 30), config.NY)
            if opening > now:
                return day.isoformat()
        day += timedelta(days=1)
    raise PaperError("다음 미국 거래일을 찾지 못했습니다")


def initialize(now=None):
    now = now or now_utc()
    with _write_lock():
        path = STORE / "manifest.json"
        if path.exists():
            return _check_manifest()
        body = {"schema": 1, "start_session": _next_open_session(now), "frozen_at": _iso(now),
                "scheme": "actual", "cutoff_kst": "24:00", "review_sessions": REVIEW_SESSIONS,
                "settings": {"S1": F1A, "S2": S2, "fill": "close", "delay_minutes": 1,
                             "touch": 1.0002, "commission": 0},
                "sources": _source_hashes(), "costs": _read(RESEARCH / "cost_table.json"),
                "adapter_sha256": _file_digest(Path(__file__)),
                "review": "60 valid sessions; reporting only; never authorizes real trading"}
        body["sha256"] = _digest(body)
        _atomic(path, body)
        return body


def _check_manifest():
    path = STORE / "manifest.json"
    if not path.exists():
        raise PaperError("먼저 py -m desk paper init으로 규칙을 잠가 주세요")
    obj = _read(path)
    body = {k: v for k, v in obj.items() if k != "sha256"}
    if obj.get("sha256") != _digest(body):
        raise PaperError("동결 원장 해시가 바뀌었습니다; 기록을 덮어쓸 수 없습니다")
    if (obj.get("adapter_sha256") != _file_digest(Path(__file__)) or
            obj["sources"] != _source_hashes()) or obj["settings"] != {
            "S1": F1A, "S2": S2, "fill": "close", "delay_minutes": 1, "touch": 1.0002, "commission": 0}:
        raise PaperError("동결 후 연구 코드·비용·규칙이 바뀌었습니다; 기존 실험 갱신을 멈춥니다")
    return obj


def _module(name, path, lib_name=None):
    """Remove research-script startup side effects, preserving all class calculations.

    Namespaced imports prevent the research scripts' generic `lib` module and
    SOS_UNLOCK/sys.path mutations from leaking into the running desk server.
    """
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    keep = []
    for node in tree.body:
        text = ast.unparse(node)
        if text.startswith("sys.path.insert(") or text.startswith("os.environ['SOS_UNLOCK'] ="):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "lib":
            node.module = lib_name
        keep.append(node)
    tree.body = keep
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _research_modules():
    global _MODULES
    with _LOCK:
        if _MODULES is None:
            code = RESEARCH / "code"
            lib = _module("_gbrc_sox_paper_lib", code / "lib.py")
            f1 = _module("_gbrc_sox_paper_f1", code / "fam_F1.py", lib.__name__)
            h0 = _module("_gbrc_sox_paper_h0", code / "h0.py", lib.__name__)
            _MODULES = lib, f1, h0
    return _MODULES


def cutoff(day):
    day = date.fromisoformat(str(day))
    return datetime.combine(day + timedelta(days=1), time(0), config.KST).astimezone(config.NY)


def _eligible_sessions(manifest, now):
    cal = market_calendar()
    start = manifest["start_session"]
    end = now.astimezone(config.NY).date().isoformat()
    if end < start:
        return []
    return [str(s.date()) for s in cal.sessions_in_range(start, end)
            if cutoff(str(s.date())) + timedelta(minutes=15) <= now]


def _credentials():
    env = dict(os.environ)
    path = config.ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    key = env.get("ALPACA_KEY_ID") or env.get("APCA_API_KEY_ID")
    secret = env.get("ALPACA_SECRET_KEY") or env.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise PaperError("Alpaca 키가 없어 종이 기록 가격을 받지 못했습니다")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def fetch_bars(session):
    """Only the two traded ETFs and the finished opening window; never order endpoints."""
    start = datetime.combine(date.fromisoformat(session), time(9, 30), config.NY)
    params = {"symbols": "SOXL,SOXS", "timeframe": "1Min", "start": _iso(start),
              "end": _iso(cutoff(session)), "limit": 10000, "feed": "sip", "adjustment": "split", "sort": "asc"}
    rows = {s: [] for s in ("SOXL", "SOXS")}
    headers = _credentials()
    while True:
        try:
            response = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=headers,
                                    params=params, timeout=25)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise PaperError(f"Alpaca SIP 자료를 받지 못했습니다 ({type(exc).__name__})") from exc
        for sym in rows:
            rows[sym].extend((data.get("bars") or {}).get(sym, []))
        token = data.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    out = {}
    for sym, bars in rows.items():
        frame = pd.DataFrame(bars)
        if frame.empty:
            raise PaperError(f"{sym} 1분봉이 없습니다")
        frame.index = pd.to_datetime(frame.pop("t"), utc=True).dt.tz_convert(config.NY)
        out[sym] = frame.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    return out


def _data(session, bars):
    lib, _, _ = _research_modules()
    start = datetime.combine(date.fromisoformat(session), time(9, 30), config.NY)
    grid = pd.date_range(start, cutoff(session), freq="min", inclusive="left")
    arrays = {}
    hashes = {}
    for sym in ("SOXL", "SOXS"):
        frame = bars[sym].copy()
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None or frame.index.has_duplicates:
            raise PaperError(f"{sym} 1분봉 시각이 잘못되었거나 중복입니다")
        frame.index = frame.index.tz_convert(config.NY)
        frame = frame[(frame.index >= start) & (frame.index < cutoff(session))].sort_index()
        if not frame.index.equals(grid):
            raise PaperError(f"{sym} 1분봉 누락: {len(grid.difference(frame.index))}분; 가격을 채우지 않습니다")
        cols = ["Open", "High", "Low", "Close", "Volume"]
        vals = frame[cols].to_numpy(float)
        if not np.isfinite(vals).all() or (vals[:, :4] <= 0).any() or (vals[:, 4] <= 0).any():
            raise PaperError(f"{sym} 결측·비정상 가격 또는 거래 없는 봉입니다")
        if ((frame.High < frame[["Open", "Close", "Low"]].max(axis=1)) |
                (frame.Low > frame[["Open", "Close", "High"]].min(axis=1))).any():
            raise PaperError(f"{sym} OHLC 범위가 잘못되었습니다")
        hashes[sym] = hashlib.sha256(frame[cols].to_csv(float_format="%.12g").encode()).hexdigest()
        arrays[sym] = {}
        for fld in cols:
            arr = np.full((1, lib.NMIN), np.nan)
            arr[0, lib.RTH0:lib.RTH0 + len(grid)] = frame[fld].to_numpy(float)
            arrays[sym][fld] = arr
    return SimpleNamespace(days=[date.fromisoformat(session)], A=arrays), hashes


def _trade(session, row):
    _, symbol, entry, exit_, entry_price, exit_price, ret, why = row
    day = date.fromisoformat(session)
    def at(label, close=False):
        value = datetime.combine(day, time.fromisoformat(label), config.NY)
        return value + timedelta(minutes=1) if close else value
    # Target touches happen somewhere inside their bar; never claim its exact close.
    exit_is_close = why != "target"
    return {"symbol": symbol, "entry_bar_et": entry, "exit_bar_et": exit_,
            "entry_at_et": at(entry, True).isoformat(), "exit_at_et": at(exit_, exit_is_close).isoformat(),
            "entry_at_kst": at(entry, True).astimezone(config.KST).isoformat(),
            "exit_at_kst": at(exit_, exit_is_close).astimezone(config.KST).isoformat(),
            "exit_time_basis": "within_bar" if why == "target" else "bar_close",
            "entry_price": float(entry_price), "exit_price": float(exit_price),
            "return_pct": float(ret * 100), "why": why}


def replay(session, bars, manifest):
    lib, f1, h0 = _research_modules()
    data, hashes = _data(session, bars)
    rules = {}
    for rid, factory in (("S1", lambda: f1.make(F1A)()), ("S2", lambda: h0.H0(**S2))):
        ret, trades = lib.Engine(data, costs=manifest["costs"], scheme="actual").run_day(factory(), 0)
        ret2, _ = lib.Engine(data, costs=manifest["costs"], scheme="actual", cost_mult=2).run_day(factory(), 0)
        rules[rid] = {"return_pct": float(ret * 100), "return_cost2_pct": float(ret2 * 100),
                      "trades": [_trade(session, t) for t in trades]}
    return {"session": session, "status": "valid", "manifest_sha256": manifest["sha256"],
            "source": "Alpaca SIP 1Min split-adjusted", "bar_sha256": hashes,
            "cutoff_et": cutoff(session).strftime("%H:%M"),
            "cutoff_at_kst": cutoff(session).astimezone(config.KST).isoformat(), "rules": rules}


def update(day=None, now=None):
    now = now or now_utc()
    with _write_lock():
        manifest = _check_manifest()
        eligible = _eligible_sessions(manifest, now)
        if day is not None and day not in eligible:
            raise PaperError("그 날짜는 동결 시작 이후의 완료된 미국 거래일이 아닙니다 (마감 +15분 필요)")
        for session in ([day] if day else eligible):
            path = STORE / "days" / f"{session}.json"
            if path.exists():
                record = _read(path)
                if record.get("manifest_sha256") != manifest["sha256"]:
                    raise PaperError(f"{session} 기록의 동결 해시가 다릅니다")
                continue
            try:
                bars = fetch_bars(session)
                record = replay(session, bars, manifest)
            except (PaperError, KeyError, ValueError) as exc:
                # Failures remain retryable and are never counted as zero-return days.
                _atomic(STORE / "missing" / f"{session}.json", {"session": session, "status": "missing",
                        "error": str(exc), "recorded_at": _iso(now)})
                continue
            # Preserve unfilled source values locally: a later vendor correction
            # cannot silently change an already-scored day. Raw cache is ignored by Git.
            raw = {}
            for symbol, frame in bars.items():
                subset = frame[["Open", "High", "Low", "Close", "Volume"]]
                raw[symbol] = [{"at": stamp.isoformat(), **{key: float(value) for key, value in row.items()}}
                               for stamp, row in subset.iterrows()]
            _atomic(STORE / "cache" / f"{session}.json", raw)
            record["recorded_at"] = _iso(now)
            record["sha256"] = _digest(record)
            _atomic(path, record)
            (STORE / "missing" / f"{session}.json").unlink(missing_ok=True)
    return payload(now=now)


def _summary(rid, days):
    ret = np.array([d["rules"][rid]["return_pct"] for d in days], float)
    n = len(ret)
    ci = None
    if n >= 2:
        rng = np.random.default_rng(20261105)
        values = rng.choice(ret, size=(10000, n), replace=True).mean(axis=1)
        ci = [float(x) for x in np.quantile(values, [.025, .975])]
    months = []
    for month in sorted({d["session"][:7] for d in days}):
        vv = [d["rules"][rid]["return_pct"] for d in days if d["session"].startswith(month)]
        months.append({"month": month, "days": len(vv), "return_pct": float((np.prod(1 + np.array(vv) / 100) - 1) * 100)})
    return {"sessions": n, "trades": sum(len(d["rules"][rid]["trades"]) for d in days),
            "mean_daily_pct": float(ret.mean()) if n else None,
            "total_return_pct": float((np.prod(1 + ret / 100) - 1) * 100) if n else None,
            "ci95_pct": ci,
            "cost2_mean_pct": float(np.mean([d["rules"][rid]["return_cost2_pct"] for d in days])) if n else None,
            "drop_top2_mean_pct": float(np.sort(ret)[:-2].mean()) if n > 2 else None,
            "months": months, "review_ready": n >= REVIEW_SESSIONS}


def payload(force=False, now=None):
    now = now or now_utc()
    if force:
        return update(now=now)
    if not (STORE / "manifest.json").exists():
        return {"initialized": False, "status": "규칙 동결 전", "warnings": [], "rules": RULES, "days": []}
    manifest = _check_manifest()
    days = []
    for path in sorted((STORE / "days").glob("*.json")):
        record = _read(path)
        if record.get("sha256") != _digest({k: v for k, v in record.items() if k != "sha256"}):
            raise PaperError(f"{path.stem} 종이 기록 해시가 바뀌었습니다")
        if record.get("manifest_sha256") != manifest["sha256"] or record["session"] < manifest["start_session"]:
            raise PaperError("동결 실험과 다른 날짜 또는 규칙의 기록입니다")
        days.append(record)
    recorded = {d["session"] for d in days}
    eligible = _eligible_sessions(manifest, now)
    missing = [_read(p) for p in sorted((STORE / "missing").glob("*.json")) if p.stem not in recorded]
    pending = [s for s in eligible if s not in recorded]
    return {"initialized": True, "start_session": manifest["start_session"], "frozen_at": manifest["frozen_at"],
            "manifest_sha256": manifest["sha256"], "cutoff_kst": "24:00", "review_sessions": REVIEW_SESSIONS,
            "valid_sessions": len(days), "missing_sessions": len(missing), "pending_sessions": len(pending),
            "status": "첫 점검 시점" if len(days) >= REVIEW_SESSIONS else "종이 기록 쌓는 중",
            "warnings": [d["error"] for d in missing],
            "rules": [{**r, **_summary(r["id"], days)} for r in RULES],
            "days": sorted(days + missing, key=lambda d: d["session"], reverse=True)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("init", "update", "status"), default="status")
    parser.add_argument("--day")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            initialize()
        result = update(args.day) if args.command == "update" else payload()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except PaperError as exc:
        print(str(exc), file=sys.stderr)
        return 1
