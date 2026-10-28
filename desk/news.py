"""Reviewed news evidence and point-in-time observation (desk_system.md §13).

Article availability, our first recording, and price outcomes are separate clocks.
This module never feeds the portfolio engine. Labels are reviewed interpretations.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from . import config

LEDGER = config.ROOT / "data" / "news_evidence" / "ledger.json"
PRICES = config.CACHE / "news_prices.json"
KINDS = {"pressure": "매도 압력", "relief": "매물 소화·자금 안정", "squeeze": "반등 연료", "fundamental": "사업·신용 위험"}
QUALITIES = {"article": "보도 원문 확인", "syndicated": "통신사 재게재 확인", "chart": "재게재 차트 확인"}
# 전수 표본(§13.2·§13.3)과 같은 부호. 2026-10-28부터 기록하는 기사는 bank·family 칸이 필수다(없으면 null, §13.1).
FAMILIES = {"D": "기록적 매도·디그로싱", "R": "매수 복귀·숏 청산·매물 인수", "S": "극단적 숏", "T": "롱·레버리지 쏠림"}
FIELDS_FROM = "2026-10-28T00:00:00+00:00"
PRIOR_T_SESSIONS = 30          # D를 기록할 때 앞 30거래일 안에 공개된 T 기사가 있었는지 그 시점 값으로 고정한다
FORWARD_LAG_H = 48             # R 전방 채점: 공개 뒤 이 시간 안에 기록한 기사만(결과를 보고 고른 기사를 빼려고)
R_HORIZONS = (1, 3, 5, 10, 20)


class NewsError(ValueError):
    pass


def stamp(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("timezone")
        return result
    except (ValueError, TypeError, AttributeError) as exc:
        raise NewsError("시각은 시간대가 있는 ISO 형식이어야 합니다") from exc


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate(data: dict) -> None:
    if not isinstance(data, dict) or data.get("version") != 1:
        raise NewsError("뉴스 원장은 version 1이어야 합니다")
    for section in ("episodes", "events", "assessments"):
        if not isinstance(data.get(section), list):
            raise NewsError(f"{section}: 목록이 필요합니다")
    episodes, events = {}, {}
    for ep in data["episodes"]:
        if not isinstance(ep, dict) or not all(isinstance(ep.get(k), str) and ep[k].strip() for k in ("id", "label", "benchmark")):
            raise NewsError("국면의 id, label, benchmark가 필요합니다")
        if ep["id"] in episodes or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", ep["benchmark"]):
            raise NewsError("중복 국면 또는 잘못된 종목 코드")
        episodes[ep["id"]] = ep
    required = ("id", "episode", "source_group", "title", "source", "url", "published_date", "available_at",
                "recorded_at", "time_note", "scope", "measurement", "window", "summary", "interpretation", "next_check")
    for row in data["events"]:
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k].strip() for k in required):
            raise NewsError("기사 필수 항목이 빠졌습니다")
        if row["id"] in events or row["episode"] not in episodes:
            raise NewsError("중복 기사 id 또는 없는 국면")
        if row.get("kind") not in KINDS or row.get("quality") not in QUALITIES:
            raise NewsError("기사 종류·확인 수준이 잘못되었습니다")
        if row.get("time_precision") not in ("exact", "unknown"):
            raise NewsError("time_precision은 exact 또는 unknown입니다")
        if row.get("stance") not in ("support", "against", "conditional"):
            raise NewsError("stance는 support, against, conditional 중 하나입니다")
        url = urlparse(row["url"])
        if url.scheme != "https" or not url.netloc or url.username or url.password:
            raise NewsError("출처는 인증 정보 없는 HTTPS 주소여야 합니다")
        try:
            datetime.strptime(row["published_date"], "%Y-%m-%d")
        except ValueError as exc:
            raise NewsError("published_date 형식 오류") from exc
        available, recorded = stamp(row["available_at"]), stamp(row["recorded_at"])
        if available > recorded:
            raise NewsError("기사 공개가 기록 시각보다 늦습니다")
        if row["time_precision"] == "unknown" and available != recorded:
            raise NewsError("공개 시각 미확인 기사는 첫 기록 시각부터만 사용합니다")
        newer = recorded >= stamp(FIELDS_FROM)
        if newer and ("bank" not in row or "family" not in row):
            raise NewsError("2026-10-28부터 기록하는 기사는 bank·family 칸이 필요합니다(해당 없으면 null)")
        bank = row.get("bank")
        if bank is not None and not (isinstance(bank, str) and bank.strip()):
            raise NewsError("bank는 원 노트를 낸 기관 이름 또는 null입니다")
        if row.get("family") not in (None, *FAMILIES):
            raise NewsError("family는 D·R·S·T 또는 null입니다")
        if newer and row.get("family") == "D":
            p = row.get("prior_t")
            if not isinstance(p, dict) or not isinstance(p.get("found"), bool) or not isinstance(p.get("ids"), list):
                raise NewsError("D 기사에는 기록 때 계산한 prior_t(앞 30거래일 T 기사 유무)가 필요합니다")
        events[row["id"]] = row
    for row in events.values():
        for t_id in (row.get("prior_t") or {}).get("ids", []):
            t = events.get(t_id)
            if not t or t.get("family") != "T" or stamp(t["recorded_at"]) > stamp(row["recorded_at"]):
                raise NewsError("prior_t는 그 D보다 먼저 기록된 T 기사만 가리킬 수 있습니다")
    for row in events.values():
        previous = events.get(row.get("supersedes"))
        if row.get("supersedes") and (not previous or previous["episode"] != row["episode"] or
                stamp(previous["recorded_at"]) >= stamp(row["recorded_at"]) or
                stamp(previous["available_at"]) > stamp(row["available_at"])):
            raise NewsError("정정은 같은 국면의 과거 기록을 참조해야 합니다")
    seen = set()
    for row in data["assessments"]:
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k].strip() for k in ("id", "episode", "recorded_at", "conclusion")):
            raise NewsError("판단 기록 필수 항목이 빠졌습니다")
        if row["id"] in seen or row["episode"] not in episodes:
            raise NewsError("중복 판단 또는 없는 국면")
        seen.add(row["id"])
        at = stamp(row["recorded_at"])
        for key in ("supports", "contradicts"):
            if not isinstance(row.get(key), list):
                raise NewsError("판단의 지지·반대 기사 목록이 필요합니다")
            for event_id in row[key]:
                event = events.get(event_id)
                if not event or event["episode"] != row["episode"] or stamp(event["recorded_at"]) > at:
                    raise NewsError("판단 근거가 없거나 당시 기록되지 않았습니다")
        if not isinstance(row.get("next_checks"), list) or not row["next_checks"] or not all(isinstance(s, str) and s.strip() for s in row["next_checks"]):
            raise NewsError("다음 확인 항목이 필요합니다")


def read_ledger(path: Path = LEDGER) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    validate(data)
    return data


def write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def import_records(incoming: dict, path: Path = LEDGER) -> int:
    """Append reviewed records. First recording is stamped here, never backdated."""
    if not isinstance(incoming, dict) or set(incoming) - {"episodes", "events", "assessments", "version"}:
        raise NewsError("가져오기 형식 오류")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise NewsError("다른 뉴스 기록 작업이 진행 중입니다") from exc
    os.close(fd)
    try:
        data = read_ledger(path) if path.exists() else {"version": 1, "episodes": [], "events": [], "assessments": []}
        at, added = now_iso(), 0
        for section in ("episodes", "events", "assessments"):
            rows = incoming.get(section, [])
            if not isinstance(rows, list):
                raise NewsError(f"{section}: 목록이 필요합니다")
            known = {r["id"]: r for r in data[section]}
            for raw in rows:
                row = copy.deepcopy(raw)
                if not isinstance(row, dict) or not row.get("id"):
                    raise NewsError("기록 id가 필요합니다")
                if row["id"] in known:
                    old = known[row["id"]]
                    comparable = {k: v for k, v in row.items() if k not in ("recorded_at", "prior_t")}
                    expected = {k: v for k, v in old.items() if k not in ("recorded_at", "prior_t")}
                    if section == "events" and row.get("time_precision") == "unknown":
                        comparable.pop("available_at", None)
                        expected.pop("available_at", None)
                    if comparable != expected:
                        raise NewsError("기존 기록은 덮어쓸 수 없습니다. 새 id와 supersedes를 사용하세요")
                    continue
                if section != "episodes":
                    row["recorded_at"] = at
                if section == "events" and row.get("time_precision") == "unknown":
                    row["available_at"] = at
                if section == "events" and row.get("family") == "D":
                    row["prior_t"] = prior_t(data["events"], row)   # 가져오는 값은 쓰지 않고 지금 원장으로 계산
                data[section].append(row)
                known[row["id"]] = row
                added += 1
        validate(data)
        write_atomic(path, data)
        return added
    finally:
        lock.unlink()


def visible_events(data: dict, cutoff: datetime, basis: str) -> list[dict]:
    if basis not in ("public", "recorded"):
        raise NewsError("조회 기준은 public 또는 recorded입니다")
    rows = [r for r in data["events"] if stamp(r["available_at"]) <= cutoff and
            (basis == "public" or stamp(r["recorded_at"]) <= cutoff)]
    replaced = {r["supersedes"] for r in rows if r.get("supersedes")}
    return sorted((r for r in rows if r["id"] not in replaced), key=lambda r: stamp(r["available_at"]))


def refresh_prices(path: Path = PRICES) -> dict:
    from . import marketdata
    tickers = {e["benchmark"] for e in read_ledger()["episodes"]}
    meta = config.CACHE / "prices_meta.json"
    if meta.exists():  # Preserve the shared desk cache's ticker coverage.
        tickers.update(json.loads(meta.read_text(encoding="utf-8")).get("tickers", []))
    close = marketdata.history(sorted(tickers), force=True)["Close"]
    wanted = {e["benchmark"] for e in read_ledger()["episodes"]}
    snapshot = {"source": "Yahoo Finance / desk.marketdata.history · 분할 반영 종가", "fetched_at": now_iso(),
                "series": {t: {d.strftime("%Y-%m-%d"): float(v) for d, v in close[t].dropna().items()}
                           for t in wanted if t in close}}
    if wanted - snapshot["series"].keys() or any(not s for s in snapshot["series"].values()):
        raise NewsError("비교 종목 가격이 누락되어 기존 캐시를 유지합니다")
    write_atomic(path, snapshot)
    return snapshot


@lru_cache(maxsize=1)
def market_calendar():
    import exchange_calendars
    return exchange_calendars.get_calendar("XNYS")


def completed_prices(series: dict, cutoff: datetime) -> list[tuple[str, float, datetime]]:
    rows = []
    calendar = market_calendar()
    for day, value in sorted(series.items()):
        if not calendar.is_session(day):
            continue
        closing = calendar.session_close(day).to_pydatetime()
        if closing <= cutoff and isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            rows.append((day, value, closing))
    return rows


def anchor_session(event: dict):
    """기사 공개 뒤 첫 정규장(그 종가가 기준). 공개 시각 미확인이면 None."""
    if event["time_precision"] != "exact":
        return None
    available = stamp(event["available_at"])
    calendar = market_calendar()
    anchor = calendar.date_to_session(available.astimezone(config.NY).date().isoformat(), direction="next")
    if calendar.session_close(anchor).to_pydatetime() <= available:
        anchor = calendar.next_session(anchor)
    return anchor


def prior_t(events: list[dict], row: dict) -> dict:
    """D 기사 공개 전 PRIOR_T_SESSIONS 거래일(공개일 포함) 안에 공개된 T 기사. 그때 원장에 있던 기사만 본다."""
    calendar = market_calendar()
    end = stamp(row["available_at"])
    last = calendar.date_to_session(end.astimezone(config.NY).date().isoformat(), direction="previous")
    first = calendar.sessions_window(last, -PRIOR_T_SESSIONS)[0].strftime("%Y-%m-%d")
    ids = [e["id"] for e in events if e.get("family") == "T" and stamp(e["available_at"]) <= end
           and stamp(e["available_at"]).astimezone(config.NY).date().isoformat() >= first]
    return {"found": bool(ids), "ids": ids, "sessions": PRIOR_T_SESSIONS, "from": first}


def forward_r(events: list[dict], series: dict, cutoff: datetime) -> dict:
    """R 전방 채점: 공개 뒤 FORWARD_LAG_H 안에 기록한 R 기사의 1~20거래일 종가 변화와 SOXX 하락 저점의 20거래일 유지.

    저점·하락 국면은 바닥 감지기(desk/bottom.py, §13.6)와 같은 정의다. 사후 후보(§13.3 GS의 R, §13.5 R 뉴스)를
    새 기사로 확인하는 표이며 매매 규칙이 아니다.
    """
    import pandas as pd

    from .bottom import frame
    rows = completed_prices(series, cutoff)
    prices = {r[0]: r[1] for r in rows}
    fr = frame(pd.Series([r[1] for r in rows], index=pd.DatetimeIndex([r[0] for r in rows]))) if len(rows) > 60 else None
    calendar, items = market_calendar(), []
    for e in events:
        lag_h = (stamp(e["recorded_at"]) - stamp(e["available_at"])).total_seconds() / 3600
        if e.get("family") != "R" or lag_h > FORWARD_LAG_H:
            continue
        item = {"id": e["id"], "title": e["title"], "url": e["url"], "bank": e.get("bank"), "available_at": e["available_at"],
                "anchor": None, "returns": {str(n): None for n in R_HORIZONS}, "in_dd": None, "low": None, "hold20": None}
        anchor = anchor_session(e)
        days = [s.strftime("%Y-%m-%d") for s in calendar.sessions_window(anchor, max(R_HORIZONS) + 1)] if anchor is not None else []
        if days and days[0] in prices:
            base = prices[days[0]]
            item["anchor"] = days[0]
            item["returns"] = {str(n): prices[days[n]] / base - 1 if days[n] in prices else None for n in R_HORIZONS}
            if fr is not None and pd.Timestamp(days[0]) in fr.index:
                r = fr.loc[pd.Timestamp(days[0])]
                item.update(in_dd=bool(r["in_dd"]), low=None if math.isnan(r["low"]) else float(r["low"]),
                            hold20=None if math.isnan(r["hold"]) else int(r["hold"]))
        items.append(item)

    def summary(group: list[dict]) -> dict:
        r10 = [i["returns"]["10"] for i in group if i["returns"]["10"] is not None]
        held = [i["hold20"] for i in group if i["in_dd"] and i["hold20"] is not None]
        return {"n": len(group), "n10": len(r10), "mean10": sum(r10) / len(r10) if r10 else None,
                "dd_n": len(held), "held": sum(held)}

    base10 = None  # 아무 날 대조: 첫 R 기준일이 든 해 1월부터 조회 시각까지 모든 장의 10거래일 변화 평균
    anchors = sorted(i["anchor"] for i in items if i["anchor"])
    if anchors:
        seq = [r[1] for r in rows if r[0] >= anchors[0][:4] + "-01-01"]
        moves = [seq[k + 10] / seq[k] - 1 for k in range(len(seq) - 10)]
        base10 = sum(moves) / len(moves) if moves else None
    return {"items": items, "all": summary(items), "gs": summary([i for i in items if i["bank"] == "GS"]),
            "base10": base10, "lag_h": FORWARD_LAG_H, "horizons": [str(n) for n in R_HORIZONS]}


def reaction(event: dict, rows: list) -> dict:
    empty = {"anchor": None, "returns": {str(n): None for n in (1, 3, 5)}, "worst_close_5": None}
    if event["time_precision"] != "exact":
        return {**empty, "reason": "기사 공개 시각 미확인"}
    calendar = market_calendar()
    anchor = anchor_session(event)
    days = [s.strftime("%Y-%m-%d") for s in calendar.sessions_window(anchor, 6)]
    prices = {r[0]: r[1] for r in rows}
    if days[0] not in prices:
        return {**empty, "reason": "공개 뒤 첫 종가 대기"}
    base = prices[days[0]]
    return {"anchor": days[0], "returns": {str(n): prices[days[n]] / base - 1 if days[n] in prices else None for n in (1, 3, 5)},
            "worst_close_5": min(0, min(prices[d] / base - 1 for d in days[1:])) if all(d in prices for d in days[1:]) else None,
            "reason": "공개 뒤 첫 종가 기준 · 체결 수익률 아님"}


def payload(as_of: str | None = None, basis: str = "public", force: bool = False,
            data: dict | None = None, prices: dict | None = None) -> dict:
    cutoff = stamp(as_of) if as_of else datetime.now(timezone.utc)
    if cutoff > datetime.now(timezone.utc):
        raise NewsError("미래 시각으로 조회할 수 없습니다")
    data = data if data is not None else read_ledger()
    validate(data)
    warning = None
    if force:
        try:
            prices = refresh_prices()
        except Exception as exc:
            warning = f"가격 갱신 실패 · 저장 자료 사용: {type(exc).__name__}: {exc}"
    if prices is None:
        prices = json.loads(PRICES.read_text(encoding="utf-8")) if PRICES.exists() else {"series": {}}
    events = visible_events(data, cutoff, basis)
    episodes = []
    for ep in data["episodes"]:
        rows = completed_prices(prices.get("series", {}).get(ep["benchmark"], {}), cutoff)
        items = []
        groups = set()
        for event in (r for r in events if r["episode"] == ep["id"]):
            duplicate = event["source_group"] in groups
            groups.add(event["source_group"])
            items.append({**event, "duplicate_source": duplicate, "reaction": reaction(event, rows)})
        assessments = [a for a in data["assessments"] if a["episode"] == ep["id"] and stamp(a["recorded_at"]) <= cutoff]
        assessment = max(assessments, key=lambda a: stamp(a["recorded_at"]), default=None)
        quote = None
        if rows:
            quote = {"day": rows[-1][0], "close": rows[-1][1],
                     "day_return": rows[-1][1] / rows[-2][1] - 1 if len(rows) > 1 else None,
                     "above_prior5_low": rows[-1][1] / min(r[1] for r in rows[-6:-1]) - 1 if len(rows) >= 6 else None}
        episodes.append({**ep, "events": items, "assessment": assessment, "quote": quote,
                         "groups": len(groups), "counts": {k: len({r["source_group"] for r in items if r["kind"] == k}) for k in KINDS}})
    soxx_eps = {ep["id"] for ep in data["episodes"] if ep["benchmark"] == "SOXX"}
    forward = forward_r([e for e in visible_events(data, cutoff, "recorded") if e["episode"] in soxx_eps],
                        prices.get("series", {}).get("SOXX", {}), cutoff)
    return {"as_of": cutoff.isoformat(), "basis": basis, "kinds": KINDS, "qualities": QUALITIES, "families": FAMILIES,
            "forward_r": forward,
            "episodes": episodes, "price_source": prices.get("source"), "prices_fetched_at": prices.get("fetched_at"),
            "warning": warning, "last_recorded_at": max((e["recorded_at"] for e in events), key=stamp, default=None),
            "method": "중요 기사를 확인하고 해석을 기록합니다. 같은 원자료의 재보도는 독립 근거로 세지 않습니다.",
            "replay_note": "공개 기준의 과거 표시는 후일 수집·해석한 기사로 복습한 결과입니다. 당시 실시간 판단 기록은 아닙니다.",
            "reaction_note": "공개 뒤 첫 미국 종가에서 1·3·5거래일 변화입니다. 아직 지나지 않은 기간은 빈칸입니다. 5일 내 하락은 종가 기준입니다."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="중요 뉴스 기록·복습")
    parser.add_argument("command", choices=("show", "validate", "import", "refresh-prices"), nargs="?", default="show")
    parser.add_argument("file", nargs="?")
    parser.add_argument("--as-of")
    parser.add_argument("--basis", choices=("public", "recorded"), default="public")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            data = read_ledger()
            print(f"뉴스 원장 OK: 기사 {len(data['events'])}건 / 국면 {len(data['episodes'])}개")
        elif args.command == "import":
            if not args.file:
                parser.error("import에는 JSON 파일이 필요합니다")
            print(f"추가: {import_records(json.loads(Path(args.file).read_text(encoding='utf-8-sig')))}건")
        elif args.command == "refresh-prices":
            print(json.dumps(refresh_prices(), ensure_ascii=False))
        else:
            print(json.dumps(payload(args.as_of, args.basis), ensure_ascii=False, indent=2))
        return 0
    except (NewsError, OSError, json.JSONDecodeError) as exc:
        print(f"뉴스 오류: {exc}")
        return 2
