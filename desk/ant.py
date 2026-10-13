"""개미 지수 1단계: 뽐뿌 증권포럼(id=stock) 글 수집 (docs/desk_system.md §12).

글 번호는 시간 순으로 커진다. 최신 번호부터 거꾸로 view.php를 한 건씩 열어
제목·본문·댓글·조회·추천을 KST 날짜별 JSONL(`data/derived/desk/ant/posts/`)에 붙인다.
끊겨도 state.json의 범위에서 이어 받는다. 작성자 이름은 저장하지 않고 소금 친 해시만 남긴다.
robots.txt는 /zboard/를 허용한다(view.php?id=my·검색만 금지). 사이트 nginx가 분당 약 15건을 넘으면
429를 주므로 5초에 1건(하루 약 1.7만 건)으로 받는다. 목록은 999쪽(약 3만 건)까지만 열려 과거는 글 번호로 연다.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import secrets
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import ROOT

BOARD = "stock"
LIST_URL = "https://www.ppomppu.co.kr/zboard/zboard.php?id=" + BOARD
VIEW_URL = "https://www.ppomppu.co.kr/zboard/view.php?id=" + BOARD + "&no={no}"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/129.0 Safari/537.36")

ANT_DIR = Path(os.environ.get("DESK_ANT_DIR") or ROOT / "data" / "derived" / "desk" / "ant")
SINCE = "2025-01-01"   # 2026 장세가 주 근거, 2025는 보조 확인(AGENTS 연구 보고 규칙)
GAP = 5.0              # 요청 간격(초). 2026-10-13 측정: 0.5~1초 간격은 429, 제한 뒤에는 4초에 1건만 통과
PENALTY = 60.0         # 429를 받으면 이만큼 쉬고 다시 묻는다
CATCH_UP_HOURS = 3     # 며칠 걸리는 거꾸로 받기 중에도 이 간격마다 새 글을 따라잡는다
CHUNK = 40             # 이만큼 받고 파일·상태를 쓴다
EMPTY_STOP = 200       # 이만큼 연속으로 글이 없으면 쪽 구조가 바뀐 것으로 보고 멈춘다(10/13 표본의 지운 글은 드문드문)
SETTLE_HOURS = 24      # 글 쓴 뒤 이 시간이 안 지났으면 댓글이 더 붙으므로 다음 실행에서 다시 받는다
BODY_MAX = 4000
COMMENT_MAX = 1000

_LIST_NO = re.compile(r'class="baseList-space baseList-numb" colspan="2">(\d+)</td>')
_TITLE = re.compile(r'<div id="topTitle">\s*<h1>(.*?)(?:<span id="comment">(\d*)</span>)?\s*</h1>', re.S)
_NAME = re.compile(r'<li class="topTitle-name">(.*?)</li>', re.S)
_DATE = re.compile(r'<li>등록일 (\d{4}-\d{2}-\d{2} \d{2}:\d{2})</li>')
_VIEWS = re.compile(r'<li>조회수 (\d+)</li>')
_UP = re.compile(r'id="vote_list_btn_txt">(\d+)<')
_DOWN = re.compile(r'id="vote_anti_list_btn_txt">(\d+)<')
_BODY = re.compile(r'<!--DCM_BODY-->(.*?)<!--/DCM_BODY-->', re.S)
_CMT = "var initialCommentData = "
_TAG = re.compile(r"<[^>]+>")
_BR = re.compile(r"<\s*(br|/p|/div|/li)\s*/?>", re.I)
_SPACE = re.compile(r"[ \t\r\f\v\xa0]+")


class Blocked(RuntimeError):
    """연속 실패가 많아 멈춘다(차단·사이트 장애 의심)."""


def text_of(fragment: str, limit: int | None = None) -> str:
    s = re.sub(r"<(script|style)\b.*?</\1>", " ", fragment or "", flags=re.S | re.I)
    s = _BR.sub("\n", s)
    s = html.unescape(_TAG.sub(" ", s))
    lines = [_SPACE.sub(" ", x).strip() for x in s.split("\n")]
    s = "\n".join(x for x in lines if x)
    return s[:limit] if limit else s


def author_key(name: str, salt: str) -> str:
    return hashlib.sha1((salt + name).encode("utf-8")).hexdigest()[:12] if name else ""


def _comments(page: str, salt: str) -> tuple[int | None, list[dict]]:
    i = page.find(_CMT)
    if i < 0:
        return None, []
    try:
        data, _ = json.JSONDecoder().raw_decode(page, i + len(_CMT))
    except ValueError:
        return None, []
    out: list[dict] = []

    def walk(node):
        if isinstance(node, list):
            for x in node:
                walk(x)
        elif isinstance(node, dict):
            if "memo" in node and "meta" in node:
                block = node.get("block_status") or {}
                if not (block.get("is_blocked") or block.get("is_spam")):
                    out.append({
                        # 오늘 댓글은 time_detail이 비고 시각만 온다. 24시간 뒤 다시 받을 때 날짜가 찬다
                        "dt": (node.get("meta") or {}).get("time_detail")
                        or (node.get("meta") or {}).get("time_display", ""),
                        "text": text_of(node.get("memo", ""), COMMENT_MAX),
                        "up": int((node.get("vote_btn") or {}).get("vote_count") or 0),
                        "down": int((node.get("anti_vote_btn") or {}).get("anti_vote_count") or 0),
                        "depth": int(node.get("depth") or 0),
                        "a": author_key(text_of(node.get("name", "")), salt),
                    })
            for v in node.values():
                if isinstance(v, (list, dict)):
                    walk(v)

    walk(data.get("comments", []))
    return data.get("total_comment"), out


def parse_view(page: str, no: int, salt: str = "") -> dict | None:
    """view.php 한 쪽 → 글 레코드. 지운 글·없는 번호는 None."""
    m, d = _TITLE.search(page), _DATE.search(page)
    if not m or not d:
        return None
    body = _BODY.search(page)
    name = _NAME.search(page)
    total, comments = _comments(page, salt)
    num = lambda rx: int(rx.search(page).group(1)) if rx.search(page) else 0
    return {
        "no": no,
        "dt": d.group(1),
        "title": text_of(m.group(1)),
        "body": text_of(body.group(1), BODY_MAX) if body else "",
        "views": num(_VIEWS),
        "up": num(_UP),
        "down": num(_DOWN),
        "ncomment": int(total if total is not None else (m.group(2) or 0)),
        "comments": comments,
        "a": author_key(text_of(name.group(1)) if name else "", salt),
    }


def newest_no(page: str) -> int:
    nos = [int(x) for x in _LIST_NO.findall(page)]
    if not nos:
        raise ValueError("목록에서 글 번호를 못 찾았다(쪽 구조 변경?)")
    return max(nos)


# ---------------------------------------------------------------- 저장

def _state_path() -> Path:
    return ANT_DIR / "state.json"


def load_state() -> dict:
    p = _state_path()
    st = json.loads(p.read_text("utf-8")) if p.exists() else {}
    st.setdefault("salt", secrets.token_hex(8))
    st.setdefault("high", None)        # 받아 둔 가장 큰 번호
    st.setdefault("low", None)         # 거꾸로 내려온 가장 작은 번호(그 위로는 빈틈 없음)
    st.setdefault("floor_done", None)  # 이 날짜(since)까지 내려갔으면 기록
    st.setdefault("recheck", {})       # 번호 → [다시 받을 시각, 실패 횟수]: 댓글이 더 붙는 글·실패한 번호
    return st


def save_state(st: dict) -> None:
    ANT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _state_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), "utf-8")
    os.replace(tmp, _state_path())


def append_posts(records: list[dict]) -> None:
    by_day: dict[str, list[dict]] = {}
    for r in records:
        by_day.setdefault(r["dt"][:10], []).append(r)
    folder = ANT_DIR / "posts"
    folder.mkdir(parents=True, exist_ok=True)
    for day, rows in by_day.items():
        with open(folder / f"{day}.jsonl", "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_day(day: str) -> dict[int, dict]:
    """그날 글(번호 → 레코드). 다시 받은 글은 뒤 줄이 앞 줄을 이긴다."""
    p = ANT_DIR / "posts" / f"{day}.jsonl"
    out: dict[int, dict] = {}
    if p.exists():
        for line in p.read_text("utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["no"]] = r
    return out


# ---------------------------------------------------------------- 받기

def kst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9))).replace(tzinfo=None)


def _stamp(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M")


class Fetcher:
    """한 연결로 GAP초마다 한 건. 429는 PENALTY만큼 쉬고, 다른 오류는 10·20·40…초 쉬고 다시 묻는다."""

    def __init__(self, gap: float = GAP, session=None, sleep=time.sleep, clock=time.monotonic):
        if session is None:
            import requests

            session = requests.Session()
            session.headers["User-Agent"] = UA
        self.gap, self.session, self.sleep, self.clock = gap, session, sleep, clock
        self._last = -1e9

    def get(self, url: str) -> str:
        err = ""
        for attempt in range(6):
            wait = self._last + self.gap - self.clock()
            if wait > 0:
                self.sleep(wait)
            self._last = self.clock()
            try:
                r = self.session.get(url, timeout=20)
            except Exception as e:  # requests 예외 전부(연결 끊김·시간 초과)
                err, pause = type(e).__name__, 10 * 2 ** attempt
            else:
                if r.status_code == 200:
                    return r.content.decode("euc-kr", "replace")
                if r.status_code == 404:
                    return ""
                err = f"HTTP {r.status_code}"
                pause = PENALTY if r.status_code == 429 else 10 * 2 ** attempt
            if attempt < 5:
                self.sleep(pause)
        raise IOError(err)


def collect(since: str = SINCE, gap: float = GAP, max_posts: int | None = None, log=print,
            get=None, now=kst_now, clock=time.monotonic) -> dict:
    """새 글 → 때가 된 다시 볼 글 → since까지 거꾸로(CATCH_UP_HOURS마다 앞 두 단계를 다시). 반환: 요약."""
    get = get or Fetcher(gap).get
    st = load_state()
    salt = st["salt"]
    budget = max_posts if max_posts is not None else 10**9
    stats = Counter()
    empty = 0  # 연속으로 글이 없던 번호 수

    def store(res) -> None:
        t = now()
        append_posts([r for _, r, _ in res if r])
        for no, rec, err in res:
            key = str(no)
            tries = st["recheck"].pop(key, [None, 0])[1]
            if err:
                stats["fail"] += 1
                if tries < 2:  # 세 번 실패한 번호는 버린다
                    st["recheck"][key] = [_stamp(t + timedelta(hours=1)), tries + 1]
            elif rec is None:
                stats["missing"] += 1
            else:
                stats["saved"] += 1
                due = datetime.strptime(rec["dt"], "%Y-%m-%d %H:%M") + timedelta(hours=SETTLE_HOURS)
                if due > t:
                    st["recheck"][key] = [_stamp(due), 0]

    def run(nos: list[int]) -> list:
        nonlocal budget, empty
        out, streak = [], 0
        for no in nos:
            try:
                rec = parse_view(get(VIEW_URL.format(no=no)), no, salt)
                out.append((no, rec, None))
                streak = 0
                empty = 0 if rec else empty + 1
                if empty >= EMPTY_STOP:
                    store(out)
                    save_state(st)
                    raise Blocked(f"{EMPTY_STOP}개 번호 연속으로 글이 없다 — 쪽 구조가 바뀌었는지 parse_view를 확인")
            except IOError as e:
                out.append((no, None, str(e)))
                streak += 1
                if streak >= 3:  # 세 번호 연속 실패면 차단·장애로 보고 멈춘다(받은 것은 저장)
                    store(out)
                    save_state(st)
                    raise Blocked(f"연속 실패 3건(마지막 {e}) — 잠시 뒤 다시 실행하면 이어 받는다")
        store(out)
        budget -= len(nos)
        return out

    def catch_up() -> int:
        newest = newest_no(get(LIST_URL))
        if st["high"] is None:  # 처음: 최신 번호부터 거꾸로 내려간다
            st["high"], st["low"] = newest, newest + 1
        forward = list(range(st["high"] + 1, newest + 1))
        for i in range(0, len(forward), CHUNK):
            if budget <= 0:
                break
            part = forward[i:i + CHUNK]
            run(part)
            st["high"] = max(part)  # high가 진행 표시라 중간에 멈춰도 다음 실행이 잇는다
            save_state(st)
        stamp = _stamp(now())
        due = sorted(int(k) for k, (when, _) in st["recheck"].items() if when <= stamp)
        for i in range(0, len(due), CHUNK):
            if budget <= 0:
                break
            run(due[i:i + CHUNK])
            save_state(st)
        return newest

    def finished() -> bool:
        return st.get("floor_done") == since or st["low"] <= 1

    newest = catch_up()
    while budget > 0 and not finished():
        cycle_end = clock() + CATCH_UP_HOURS * 3600
        while budget > 0 and not finished() and clock() < cycle_end:
            part = list(range(st["low"] - 1, max(0, st["low"] - 1 - min(CHUNK, budget)), -1))
            res = run(part)
            st["low"] = min(part)
            dates = [r["dt"][:10] for _, r, _ in res if r]
            if dates and max(dates) < since:
                st["floor_done"] = since
            save_state(st)
            stats["chunks"] += 1
            if stats["chunks"] % 25 == 0:
                log(f"{_stamp(now())}  번호 {st['low']:,} ({min(dates) if dates else '?'})  "
                    f"저장 {stats['saved']:,}  없음 {stats['missing']:,}  실패 {stats['fail']:,}")
        if budget > 0 and not finished():
            newest = catch_up()
    st["last_run"] = _stamp(now())
    save_state(st)
    return {"newest": newest, "low": st["low"], "high": st["high"], "since": since,
            "done": finished(), "recheck": len(st["recheck"]), **stats}


def status() -> dict:
    st = load_state() if _state_path().exists() else {}
    folder = ANT_DIR / "posts"
    days = {}
    for p in sorted(folder.glob("*.jsonl")) if folder.exists() else []:
        last: dict[str, int] = {}  # 다시 받은 글은 뒤 줄만 센다(json.dumps가 "no"를 맨 앞에 쓴다)
        for line in p.read_text("utf-8").splitlines():
            if line.startswith('{"no": '):
                last[line[7:line.index(",")]] = line.count('"depth": ')
        days[p.stem] = (len(last), sum(last.values()))
    return {"state": {k: v for k, v in st.items() if k not in ("salt", "recheck")},
            "recheck": len(st.get("recheck", {})), "days": days}


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="py -m desk ant")
    ap.add_argument("command", choices=["collect", "status"])
    ap.add_argument("--since", default=SINCE)
    ap.add_argument("--gap", type=float, default=GAP, help="요청 간격(초). 사이트 제한 때문에 4초 아래로 내리지 않는다")
    ap.add_argument("--max", type=int, default=None, help="이번 실행에서 열 글 번호 수 상한")
    a = ap.parse_args(argv)
    if a.command == "status":
        s = status()
        print(json.dumps(s["state"], ensure_ascii=False), f"recheck {s['recheck']}")
        for day, (n, c) in s["days"].items():
            print(f"{day}  글 {n:>5,}  댓글 {c:>6,}")
        return 0
    try:
        out = collect(a.since, max(a.gap, 4.0), a.max, log=lambda m: print(m, flush=True))
        print(json.dumps(out, ensure_ascii=False), flush=True)
    except Blocked as e:
        print("멈춤:", e, flush=True)
        return 1
    return 0
