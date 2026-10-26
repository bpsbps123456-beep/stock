"""개미 지수 판정용 로더 (HYPOTHESES.md §2.1·§2.2, 2026-10-15 동결판 그대로).

읽기만 한다: data/derived/desk/ant/posts/YYYY-MM-DD.jsonl(컷오프 이하 파일), state.json(소금·진행 번호).
가격은 거래일 달력(날짜)만 쓴다. 가격 값·창 마감 뒤 값은 읽지도 만들지도 않는다(앞날 주가 방화벽).

산출(interim/results/):
  windows_kr.parquet  W_KR(D) 본판 + 고정 24시간 강건판. 창 하나에 한 줄(창 id, 경계, 길이·종류, 번호 증가·결번율·장애·스팸, 집계 색인)
  windows_us.parquet  W_U(D) 본판(08:29 ET) + 09:00 ET 강건판 + 고정 24시간판
  units.parquet       분석 단위 긴 표. 글 = 첫 줄 제목·본문, 댓글 = 모든 줄의 합집합(날짜 확정 뒤 중복 제거).
                      단위마다 자기 시각이 든 창 id(변형별), 작성자 해시, 종류. 댓글에는 부모 글 창 기준 '마감 전' 표지
  loader_*.{json,csv,parquet}  스냅샷 목록, 달력(날짜만), 댓글 날짜 확정률, 날짜별 결번율, 소금 연속성, 요약

시각은 모두 KST(tz 없는 datetime)다. 창 경계는 [open, close) 반열림이다: 마감 분(07:59)은 포함, 08:00은 다음 창.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[6]
ANT_DIR = ROOT / "data" / "derived" / "desk" / "ant"
POSTS_DIR = ANT_DIR / "posts"
STATE_PATH = ANT_DIR / "state.json"
INTERIM = Path(__file__).resolve().parents[1]
RESULTS = INTERIM / "results"

CUTOFF = "2026-10-15"          # 이 날짜 이하 파일만 쓴다(스냅샷)
CAL_START = date(2024, 11, 1)  # 2025-01-02 창의 직전 거래일을 덮는다

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")

FULL_RX = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
TOD_RX = re.compile(r"^\d{2}:\d{2}:\d{2}$")

# 같은 요일 기준선(장애·스팸): 직전 같은 요일 창 최대 8개, 최소 3개, 연휴 뒤 창은 기준에서 뺀다(§2.0 rz8와 같은 규칙)
BASE_N, BASE_MIN = 8, 3
OUTAGE_RATIO = 0.20            # §2.4: 번호 증가가 같은 요일 중앙값의 20% 미만
SPAM_NO_RATIO = 1.5            # 스팸 파도: 번호 증가 ≥ 중앙값 × 1.5 이면서
SPAM_SAVED_RATIO = 1.2         #            저장 글 수 < 중앙값 × 1.2 (notes_loader.md)


# ---------------------------------------------------------------- 스냅샷 읽기

def read_snapshot(posts_dir: Path = POSTS_DIR, cutoff: str = CUTOFF, manifest: dict | None = None,
                  now: datetime | None = None) -> tuple[list[dict], dict]:
    """컷오프 이하 파일의 완성된 줄만 읽는다. manifest가 있으면 그 바이트 수만 읽고 SHA-256을 맞춘다(재현).

    반환 rows: [{"file": "YYYY-MM-DD", "line": i, "rec": dict}], manifest: {"S": 스냅샷 시각, "files": {...}}.
    S는 모든 줄의 수집 시각 상한이다(읽기를 마친 시각). 수집기가 쓰는 중인 마지막 미완성 줄은 버린다.
    """
    rows: list[dict] = []
    files_out: dict[str, dict] = {}
    paths = sorted(p for p in Path(posts_dir).glob("*.jsonl") if p.stem <= cutoff)
    for p in paths:
        if manifest is not None:
            spec = manifest["files"].get(p.name)
            if spec is None:  # 스냅샷 뒤에 생긴 파일(거꾸로 받기가 더 내려간 날)
                continue
            with open(p, "rb") as fh:
                data = fh.read(spec["bytes"])
            if hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise ValueError(f"{p.name}: 스냅샷 SHA-256 불일치(앞부분이 바뀌었다)")
        else:
            data = p.read_bytes()
            data = data[: data.rfind(b"\n") + 1]
        n = 0
        for i, line in enumerate(data.split(b"\n")):
            if not line.strip():
                continue
            rows.append({"file": p.stem, "line": i, "rec": json.loads(line.decode("utf-8"))})
            n += 1
        files_out[p.name] = {"bytes": len(data), "lines": n, "sha256": hashlib.sha256(data).hexdigest()}
    if manifest is not None:
        S = datetime.fromisoformat(manifest["S"])
    else:
        S = (now or datetime.now(KST).replace(tzinfo=None)).replace(microsecond=0)
    return rows, {"cutoff": cutoff, "S": S.isoformat(), "files": files_out}


# ---------------------------------------------------------------- 글·댓글 정리(§2.2 중복 줄·댓글 시각)

def _pdt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


def _cdt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _tod(s: str) -> time:
    return datetime.strptime(s, "%H:%M:%S").time()


@dataclass
class Resolved:
    posts: list[dict] = field(default_factory=list)
    comments: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def resolve(rows: list[dict], S: datetime) -> Resolved:
    """같은 no의 여러 줄 → 글 하나(첫 줄 제목·본문) + 댓글 합집합(날짜 확정 → 해시+본문+시각 중복 제거).

    댓글 날짜(원천은 수집 당일 댓글을 'HH:MM:SS'로만 준다):
      - 한 줄 안의 시각만 댓글은 모두 그 줄의 수집 날짜(KST 달력일)에 달렸다. 날짜가 붙은 댓글은 수집 날짜보다 앞선다.
        (10/15 스냅샷 확인: 맞춘 줄 1,466개 모두 날짜 1개, 날짜 붙은 댓글이 같은 날인 줄 0개, 24시간 안인데 날짜가 붙은 줄 10개)
      - matched: 다른 줄에서 같은 (해시, 본문, 시각) 댓글이 날짜와 함께 나오면 그 날짜. 그 줄의 수집 날짜도 정해진다.
      - 줄마다 수집 날짜 범위 [lo, hi]를 세운다. lo = max(글 날짜(시각이 글보다 이르면 +1), 날짜 붙은 댓글 최대 + 1),
        hi = 스냅샷 S(시각이 S보다 늦으면 −1). 같은 파일은 덧붙이기만 하므로 줄 순서 = 수집 순서다.
        앞으로 lo를, 뒤로 hi를 전파한다. 시각만 댓글의 날짜는 항상 hi(가능한 가장 늦은 날짜)다.
        lo == hi이면 'row'(유일), 아니면 'latest'. 늦게 놓으면 마감 전 집계에서 빠지는 쪽으로만 틀린다.
    """
    stats = defaultdict(int)
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["file"], rows[i]["line"]))
    by_no: dict[int, list[int]] = defaultdict(list)
    for i in order:
        by_no[rows[i]["rec"]["no"]].append(i)
    stats["files_multi_no"] = sum(1 for idxs in by_no.values() if len({rows[i]["file"] for i in idxs}) > 1)

    lo: dict[int, date] = {}
    hi: dict[int, date] = {}
    exact: dict[int, date] = {}
    fidx_by_no: dict[int, dict] = {}
    for no, idxs in by_no.items():
        fidx: dict[tuple, set] = defaultdict(set)
        for i in idxs:
            for c in rows[i]["rec"]["comments"]:
                if FULL_RX.match(c["dt"]):
                    fidx[(c["a"], c["text"], c["dt"][11:])].add(c["dt"][:10])
        fidx_by_no[no] = fidx
        for i in idxs:
            rec = rows[i]["rec"]
            P = _pdt(rec["dt"])
            tods = [_tod(c["dt"]) for c in rec["comments"] if TOD_RX.match(c["dt"])]
            fulls = [_cdt(c["dt"]) for c in rec["comments"] if FULL_RX.match(c["dt"])]
            l = P.date() + (timedelta(days=1) if tods and min(tods) < P.time() else timedelta(0))
            if fulls:
                l = max(l, max(fulls).date() + timedelta(days=1))
            h = S.date() - (timedelta(days=1) if tods and max(tods) > S.time() else timedelta(0))
            md = {fd for c in rec["comments"] if TOD_RX.match(c["dt"])
                  for fd in fidx.get((c["a"], c["text"], c["dt"]), ())}
            if md:
                if len(md) > 1:
                    stats["row_multi_match_dates"] += 1
                d = date.fromisoformat(max(md))
                if not (l <= d <= h):
                    stats["row_match_outside_bounds"] += 1
                exact[i] = d
                l = h = d
            lo[i], hi[i] = l, h

    # 파일 안 줄 순서 = 수집 순서: lo는 앞으로, hi는 뒤로 전파한다
    by_file: dict[str, list[int]] = defaultdict(list)
    for i in order:
        by_file[rows[i]["file"]].append(i)
    for idxs in by_file.values():
        run = None
        for i in idxs:
            if run is not None and i not in exact:
                lo[i] = max(lo[i], run)
            run = lo[i] if run is None else max(run, lo[i])
        run = None
        for i in reversed(idxs):
            if run is not None:
                hi[i] = min(hi[i], run)
            run = hi[i] if run is None else min(run, hi[i])
    stats["row_lo_gt_hi"] = sum(1 for i in lo if lo[i] > hi[i])

    res = Resolved()
    for no, idxs in by_no.items():
        first, last = rows[idxs[0]]["rec"], rows[idxs[-1]]["rec"]
        P = _pdt(first["dt"])
        fidx = fidx_by_no[no]
        # 1) 시각만 댓글 묶음(같은 키가 여러 줄에 나오면 같은 댓글: 범위의 교집합)
        tgroups: dict[tuple, list[int]] = defaultdict(list)
        for i in idxs:
            for c in rows[i]["rec"]["comments"]:
                if TOD_RX.match(c["dt"]) and (c["a"], c["text"], c["dt"]) not in fidx:
                    tgroups[(c["a"], c["text"], c["dt"])].append(i)
        tassign: dict[tuple, tuple[datetime, str]] = {}
        for k, ii in tgroups.items():
            d = min(hi[i] for i in ii)
            l = max(lo[i] for i in ii)
            tassign[k] = (datetime.combine(d, _tod(k[2])), "row" if l == d else "latest")
        # 2) 줄마다 출현을 고유 댓글로 묶는다. 고유 키 = (해시, 본문, 확정 시각)
        uniq: dict[tuple, dict] = {}
        for i in idxs:
            root_key = None
            for pos, c in enumerate(rows[i]["rec"]["comments"]):
                raw = c["dt"]
                if FULL_RX.match(raw):
                    dt, conf = _cdt(raw), "full"
                elif TOD_RX.match(raw):
                    k = (c["a"], c["text"], raw)
                    if k in fidx:
                        ds = fidx[k]
                        d = hi[i].isoformat() if hi[i].isoformat() in ds else max(ds)
                        dt, conf = datetime.combine(date.fromisoformat(d), _tod(raw)), "matched"
                    else:
                        dt, conf = tassign[k]
                else:
                    stats["comment_dt_unknown"] += 1
                    continue
                key = (c["a"], c["text"], dt)
                if int(c.get("depth") or 0) == 0 or root_key is None:
                    root_key = key
                u = uniq.get(key)
                if u is None:
                    u = uniq[key] = {"conf": conf, "first": (i, pos)}
                elif conf != "full":  # 시각만 출현이 있으면 그 판정을 남긴다(matched/row/latest)
                    u["conf"] = conf
                u.update(src_row=i, src_pos=pos, depth=int(c.get("depth") or 0), root_key=root_key)
        ordered = sorted(uniq.items(), key=lambda kv: (kv[0][2], kv[1]["first"]))
        cid_of = {k: j for j, (k, _) in enumerate(ordered)}
        for j, (k, u) in enumerate(ordered):
            res.comments.append({
                "no": no, "cid": j, "a": k[0], "text": k[1], "dt_kst": k[2], "dt_conf": u["conf"],
                "depth": u["depth"], "root_cid": cid_of.get(u["root_key"], j),
                "src_line": f'{rows[u["src_row"]]["file"]}:{rows[u["src_row"]]["line"]}', "src_pos": u["src_pos"],
                "post_dt_kst": P, "before_post": k[2] < P,
            })
        nsaved = [len(rows[i]["rec"]["comments"]) for i in idxs]
        ncom = [int(rows[i]["rec"].get("ncomment") or 0) for i in idxs]
        res.posts.append({
            "no": no, "dt_kst": P, "a": first["a"], "title": first["title"], "text": first["body"],
            "n_rows": len(idxs), "ncomment_first": ncom[0], "ncomment_last": ncom[-1],
            "nsaved_first": nsaved[0], "nsaved_last": nsaved[-1], "n_union": len(ordered),
            "trunc_first": ncom[0] > nsaved[0], "trunc_last": ncom[-1] > nsaved[-1],
            "trunc_any": any(a > b for a, b in zip(ncom, nsaved)),
            "body_edited": first["body"] != last["body"], "title_edited": first["title"] != last["title"],
        })
    res.stats = dict(stats)
    return res


# ---------------------------------------------------------------- 거래일 달력

def _cal_from_yf(start: date, end: date) -> tuple[list[date], list[date]]:
    import yfinance as yf

    out = []
    for t in ("^KS11", "^GSPC"):
        h = yf.Ticker(t).history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                                 interval="1d", auto_adjust=False)
        out.append(sorted({d.date() for d in pd.DatetimeIndex(h.index) if d.date() <= end}))  # 날짜만 쓴다
    return out[0], out[1]


def _cal_from_pmc(start: date, end: date) -> tuple[list[date], list[date]]:
    import pandas_market_calendars as pmc  # noqa: F401 (설치돼 있을 때만)

    kr = pmc.get_calendar("XKRX").valid_days(start, end)
    us = pmc.get_calendar("XNYS").valid_days(start, end)
    return [d.date() for d in kr], [d.date() for d in us]


def _xcal_check(start: date, end: date, kr: list[date], us: list[date]) -> dict:
    """exchange_calendars가 있으면 거래일 차이와 NYSE 조기 폐장일을 함께 본다(달력 원천은 바꾸지 않는다)."""
    try:
        import exchange_calendars as xc
    except ImportError:
        return {"available": False, "nyse_early": []}
    out = {"available": True}
    for name, code, mine in (("krx", "XKRX", kr), ("nyse", "XNYS", us)):
        cal = xc.get_calendar(code, start=pd.Timestamp(start), end=pd.Timestamp(end))
        theirs = {d.date() for d in cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))}
        out[name + "_only_in_source"] = sorted(str(d) for d in set(mine) - theirs)
        out[name + "_only_in_xcal"] = sorted(str(d) for d in theirs - set(mine))
        if code == "XNYS":
            closes = cal.schedule.loc[pd.Timestamp(start):pd.Timestamp(end), "close"]
            out["nyse_early"] = sorted(str(ix.date()) for ix, c in closes.items()
                                       if c.tz_convert(ET).time() < time(16, 0))
    return out


def trading_calendars(start: date, end: date, cache: Path | None = None) -> dict:
    """KRX·NYSE 거래일. pandas_market_calendars가 있으면 그것, 없으면 yfinance ^KS11·^GSPC 거래일(날짜만).

    cache(parquet)에는 날짜와 거래일 여부만 둔다(가격 열 없음). 원천·대조 결과는 같은 이름의 .json에 둔다.
    """
    meta_path = cache.with_suffix(".json") if cache is not None else None
    if cache is not None and cache.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text("utf-8"))
        c = pd.read_parquet(cache)
        if meta.get("start", "9999") <= start.isoformat() and meta.get("end", "") >= end.isoformat():
            c = c[(c["date"] >= pd.Timestamp(start)) & (c["date"] <= pd.Timestamp(end))]
            return {"krx": [d.date() for d in c.loc[c["krx"], "date"]],
                    "nyse": [d.date() for d in c.loc[c["nyse"], "date"]],
                    "nyse_early": {d.date() for d in c.loc[c["nyse_early"], "date"]},
                    "source": meta["source"], "check": meta.get("check", {})}
    try:
        kr, us = _cal_from_pmc(start, end)
        source = "pandas_market_calendars"
    except ImportError:
        kr, us = _cal_from_yf(start, end)
        source = "yfinance ^KS11/^GSPC"
    check = _xcal_check(start, end, kr, us)
    early = {date.fromisoformat(x) for x in check.get("nyse_early", [])} & set(us)
    if cache is not None:
        days = pd.date_range(start, end, freq="D")
        kset, uset = set(kr), set(us)
        df = pd.DataFrame({"date": days, "krx": [d.date() in kset for d in days],
                           "nyse": [d.date() in uset for d in days],
                           "nyse_early": [d.date() in early for d in days]})
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        meta_path.write_text(json.dumps({"source": source, "start": start.isoformat(), "end": end.isoformat(),
                                         "check": check}, ensure_ascii=False, indent=1), "utf-8")
    return {"krx": kr, "nyse": us, "nyse_early": early, "source": source, "check": check}


# ---------------------------------------------------------------- 창

def et_to_kst(d: date, hh: int, mm: int) -> datetime:
    """미국 동부 시각 → KST(tz 없는 datetime). 서머타임은 tz 변환이 정한다."""
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET).astimezone(KST).replace(tzinfo=None)


def dst_transitions(y0: int, y1: int) -> list[datetime]:
    """America/New_York 서머타임 전환 시각(KST)."""
    out = []
    t = datetime(y0, 1, 1, tzinfo=ET)
    prev = t.utcoffset()
    while t.year <= y1:
        t2 = t + timedelta(hours=1)
        if t2.utcoffset() != prev:
            out.append(t2.astimezone(KST).replace(tzinfo=None))
            prev = t2.utcoffset()
        t = t2
    return out


def _kind(P: date, D: date) -> tuple[str, int]:
    """(P, D) 사이 휴장일로 창 종류: 없음 = weekday, 토·일만 = weekend, 평일 휴장 포함 = holiday."""
    closed = [P + timedelta(days=k) for k in range(1, (D - P).days)]
    wk = [x for x in closed if x.weekday() < 5]
    if not closed:
        return "weekday", 0
    return ("holiday" if wk else "weekend"), len(wk)


def make_windows(cal: dict, market: str) -> pd.DataFrame:
    """market 'kr': main = W_KR, fixed24. market 'us': main = W_U(08:29 ET), close0900(09:00 ET 강건판), fixed24."""
    own = sorted(cal["krx" if market == "kr" else "nyse"])
    other = set(cal["nyse" if market == "kr" else "krx"])
    own_set = set(own)
    early = cal.get("nyse_early", set())
    trans = dst_transitions(own[0].year, own[-1].year + 1)
    recs = []
    for P, D in zip(own, own[1:]):
        kind, n_wk = _kind(P, D)
        if market == "kr":
            # 한국 창의 미국 세션은 미국 날짜 P..D−1에 열린 것: P에 미국이 쉬었거나 (P, D) 평일에 미국이 열렸으면 엇갈림
            span = [P + timedelta(days=k) for k in range(0, (D - P).days)]
            mismatch = any(x.weekday() < 5 and ((x in own_set) != (x in other)) for x in span)
            variants = {"main": (datetime.combine(P, time(8)), datetime.combine(D, time(8))),
                        "fixed24": (datetime.combine(D - timedelta(days=1), time(8)), datetime.combine(D, time(8)))}
        else:
            # 미국 창에 든 한국 세션은 날짜 P+1..D
            span = [P + timedelta(days=k) for k in range(1, (D - P).days + 1)]
            mismatch = any(x.weekday() < 5 and ((x in own_set) != (x in other)) for x in span)
            variants = {"main": (et_to_kst(P, 8, 30), et_to_kst(D, 8, 30)),
                        "close0900": (et_to_kst(P, 9, 0), et_to_kst(D, 9, 0)),
                        "fixed24": (et_to_kst(D - timedelta(days=1), 8, 30), et_to_kst(D, 8, 30))}
        mon = D - timedelta(days=D.weekday())
        # 전환 주: 전환 시각(일요일 KST 15/16시)이 창 안에 있거나, D가 전환 다음 월요일로 시작하는 주에 있다
        o0, c0 = variants["main"]
        dst = any(o0 <= t < c0 or mon == t.date() + timedelta(days=1) for t in trans)
        for v, (o, c) in variants.items():
            prefix = {"main": "K", "fixed24": "K24-"}.get(v, "") if market == "kr" else \
                {"main": "U", "close0900": "U0900-", "fixed24": "U24-"}[v]
            length = (c - o).total_seconds() / 3600
            recs.append({
                "win_id": f"{prefix}{D.isoformat()}", "market": market, "variant": v, "D": pd.Timestamp(D),
                "P": pd.Timestamp(P), "weekday": D.weekday(), "open_kst": o, "close_kst": c,
                "length_h": length, "kind": kind if v != "fixed24" else "fixed24", "main_kind": kind,
                "closed_weekdays": n_wk, "over96": length > 96, "xholiday_mismatch": mismatch,
                "dst_week": dst or any(o <= t < c for t in trans),
                "early_close_D": (D in early) if market == "us" else False,
                "early_close_P": (P in early) if market == "us" else False,
            })
    return pd.DataFrame(recs)


def _ns(x) -> np.ndarray:
    return np.array(pd.to_datetime(pd.Series(x)).to_numpy(dtype="datetime64[s]"))


def assign_window(t: np.ndarray, wins: pd.DataFrame) -> np.ndarray:
    """시각 배열 → 그 시각이 든 창 id([open, close)). 창 밖·결측은 None. 창은 서로 겹치지 않아야 한다."""
    w = wins.sort_values("close_kst")
    o, c, ids = _ns(w["open_kst"]), _ns(w["close_kst"]), w["win_id"].to_numpy()
    j = np.searchsorted(c, t, side="right")
    out = np.full(len(t), None, dtype=object)
    ok = (j < len(c)) & ~np.isnat(t)
    jj = j[ok]
    inside = t[ok] >= o[jj]
    sel = np.flatnonzero(ok)[inside]
    out[sel] = ids[jj[inside]]
    return out


# ---------------------------------------------------------------- 번호 증가(지운 글 포함)·품질 표지

def number_boundary(post_dt: np.ndarray, post_no: np.ndarray, t: np.ndarray) -> np.ndarray:
    """경계 번호 B(t) = dt < t(= dt ≤ 마감 분)인 저장 글 번호의 누적 최대. 앞선 글이 없으면 NaN."""
    o = np.argsort(post_dt, kind="stable")
    dts, cm = post_dt[o], np.maximum.accumulate(post_no[o].astype(float))
    k = np.searchsorted(dts, t, side="left") - 1
    out = np.where(k >= 0, cm[np.clip(k, 0, None)], np.nan)
    return out


def _same_weekday_median(df: pd.DataFrame, col: str) -> np.ndarray:
    """직전 같은 요일 창(연휴 뒤 창·불완전 창 제외) 최대 BASE_N개의 중앙값. BASE_MIN개 미만이면 NaN. 자기 제외."""
    out = np.full(len(df), np.nan)
    hist: dict[int, list[float]] = defaultdict(list)
    for k, (wd, kind, comp, x) in enumerate(zip(df["weekday"], df["main_kind"], df["complete"], df[col])):
        h = hist[wd]
        if len(h) >= BASE_MIN:
            out[k] = float(np.median(h[-BASE_N:]))
        if comp and kind != "holiday" and np.isfinite(x):
            h.append(float(x))
    return out


def add_window_stats(wins: pd.DataFrame, posts: pd.DataFrame, units: pd.DataFrame, t_end: datetime,
                     id_col: str) -> pd.DataFrame:
    """창 한 변형에 번호 증가·결번율·완전성·단위 집계·장애·스팸·색인(u_lo, u_hi)을 붙인다."""
    wins = wins.sort_values(["variant", "close_kst"]).reset_index(drop=True)
    pdt, pno = _ns(posts["dt_kst"]), posts["no"].to_numpy()
    o, c = _ns(wins["open_kst"]), _ns(wins["close_kst"])
    b_open, b_close = number_boundary(pdt, pno, o), number_boundary(pdt, pno, c)
    sorted_no = np.sort(pno)
    n_saved = np.where(np.isfinite(b_open) & np.isfinite(b_close),
                       np.searchsorted(sorted_no, np.nan_to_num(b_close), "right")
                       - np.searchsorted(sorted_no, np.nan_to_num(b_open), "right"), np.nan)
    wins["b_open"], wins["b_close"] = b_open, b_close
    wins["d_no"] = b_close - b_open
    wins["n_saved_in_range"] = n_saved
    wins["missing_rate"] = np.where(wins["d_no"] > 0, 1 - n_saved / wins["d_no"], np.nan)
    wins["complete"] = np.isfinite(b_open) & (wins["close_kst"] <= t_end).to_numpy()
    wins["d_no_per_h"] = wins["d_no"] / wins["length_h"]
    # 단위 집계(창 id로 모은 것): 글 수, 댓글 수, 작성자 수, 댓글 날짜 확정 몫, 잘림 몫
    u = units[units[id_col].notna()]
    g = u.groupby(id_col)
    agg = pd.DataFrame({
        "n_posts": g["utype"].apply(lambda s: int((s == "post").sum())),
        "n_comments": g["utype"].apply(lambda s: int((s == "comment").sum())),
        "n_authors": g["a"].nunique(),
    })
    cm = u[u["utype"] == "comment"].groupby(id_col)["dt_conf"]
    agg["c_conf_share"] = cm.apply(lambda s: float((s != "latest").mean()))
    agg["c_latest"] = cm.apply(lambda s: int((s == "latest").sum()))
    pu = u[u["utype"] == "post"].groupby(id_col)
    # 잘림 비율은 첫 줄 표지로 센다(audit_leak.md): 마지막 줄(다시 받은 줄)의 ncomment는 마감 뒤 붙은 댓글 수라
    # 그 글이 마감 뒤에 받은 주목이 섞인다. 첫 줄이 마감에 가장 가깝다(제목·본문 첫 줄 규칙과 같은 원리)
    agg["trunc_share"] = pu["trunc_first"].mean()
    wins = wins.merge(agg, left_on="win_id", right_index=True, how="left")
    for col in ("n_posts", "n_comments", "n_authors", "c_latest"):
        wins[col] = wins[col].fillna(0).astype(int)
    wins["saved_per_h"] = wins["n_posts"] / wins["length_h"]
    # 장애·스팸(같은 요일 기준선, 직전 자료만)
    parts = []
    for v, wv in wins.groupby("variant", sort=False):
        wv = wv.copy()
        wv["base_no_per_h"] = _same_weekday_median(wv, "d_no_per_h")
        wv["base_saved_per_h"] = _same_weekday_median(wv, "saved_per_h")
        r_no = wv["d_no_per_h"] / wv["base_no_per_h"]
        r_sv = wv["saved_per_h"] / wv["base_saved_per_h"]
        wv["ratio_no"], wv["ratio_saved"] = r_no, r_sv
        known = r_no.notna() & wv["complete"]
        wv["outage"] = np.where(known, r_no < OUTAGE_RATIO, pd.NA)
        wv["spam_wave"] = np.where(known & r_sv.notna(), (r_no >= SPAM_NO_RATIO) & (r_sv < SPAM_SAVED_RATIO), pd.NA)
        parts.append(wv)
    wins = pd.concat(parts, ignore_index=True)
    wins["outage"] = wins["outage"].astype("boolean")
    wins["spam_wave"] = wins["spam_wave"].astype("boolean")
    # 집계용 색인: units(dt 순)에서 이 창의 행 범위 [u_lo, u_hi)
    udt = _ns(units["dt_kst"])
    wins["u_lo"] = np.searchsorted(udt, _ns(wins["open_kst"]), "left")
    wins["u_hi"] = np.searchsorted(udt, _ns(wins["close_kst"]), "left")
    return wins


# ---------------------------------------------------------------- 마감 전 댓글·글별 관심

def comments_before_close(units: pd.DataFrame, close_kst) -> pd.DataFrame:
    """dt < close(= dt ≤ 마감 분)인 댓글만. close는 한 시각 또는 댓글 행마다의 시각 배열."""
    c = units[units["utype"] == "comment"]
    close = close_kst if np.isscalar(close_kst) or isinstance(close_kst, (datetime, pd.Timestamp)) \
        else pd.Series(close_kst, index=units.index)[c.index]
    return c[c["dt_kst"] < close]


def pre_close_for_post_window(units: pd.DataFrame, wins: pd.DataFrame, post_col: str) -> pd.Series:
    """댓글마다 부모 글 창(post_col: 'pwin_kr' 등)의 마감 전인지. 부모 창이 없으면 False."""
    close = units[post_col].map(dict(zip(wins["win_id"], wins["close_kst"])))
    return (units["utype"] == "comment") & close.notna() & (units["dt_kst"] < close)


def salt_continuity(units: pd.DataFrame) -> pd.DataFrame:
    """날짜(KST)별 작성자 해시 집합과 직전 날짜의 겹침률. 소금이 바뀌면 겹침률이 0 가까이 떨어진다.
    시각이 불확실한(latest) 댓글은 뺀다. 표지: 겹침률 < 직전 8일 중앙값의 절반(직전 3일 이상일 때)."""
    u = units[(units["dt_conf"] != "latest") & units["dt_kst"].notna() & (units["a"] != "")]
    sets = u.groupby(u["dt_kst"].dt.date)["a"].apply(set).sort_index()
    recs, prev, hist = [], None, []
    for d, s in sets.items():
        ov = len(s & prev) / len(s) if prev else np.nan
        jac = len(s & prev) / len(s | prev) if prev else np.nan
        base = float(np.median(hist[-8:])) if len(hist) >= 3 else np.nan
        recs.append({"date": str(d), "n_authors": len(s), "overlap_prev": ov, "jaccard_prev": jac,
                     "base_overlap": base, "flag_drop": bool(np.isfinite(base) and ov < 0.5 * base)})
        if np.isfinite(ov):
            hist.append(ov)
        prev = s
    return pd.DataFrame(recs)


def salt_check(state_path: Path = STATE_PATH) -> dict:
    """state.json 소금이 있는지만 본다. 값은 출력하지 않고, 바뀜 확인용 지문(SHA-256 앞 8자)만 남긴다."""
    if not state_path.exists():
        return {"present": False}
    st = json.loads(state_path.read_text("utf-8"))
    s = st.get("salt")
    ok = isinstance(s, str) and bool(re.fullmatch(r"[0-9a-f]{16}", s or ""))
    return {"present": ok, "fingerprint": hashlib.sha256(s.encode()).hexdigest()[:8] if ok else None,
            "high": st.get("high"), "low": st.get("low")}


# ---------------------------------------------------------------- 조립

ID_COL = {("kr", "main"): "win_kr", ("kr", "fixed24"): "win_kr24",
          ("us", "main"): "win_u", ("us", "close0900"): "win_u0900", ("us", "fixed24"): "win_u24"}


def build(rows: list[dict], manifest: dict, cal: dict, state: dict | None = None) -> dict:
    """스냅샷 줄 → posts·comments·units·windows_kr·windows_us. state는 진행 번호(high) 확인용(없어도 된다)."""
    S = datetime.fromisoformat(manifest["S"])
    res = resolve(rows, S)
    posts = pd.DataFrame(res.posts)
    posts["dt_kst"] = pd.to_datetime(posts["dt_kst"]).astype("datetime64[s]")
    ccols = ["no", "cid", "a", "text", "dt_kst", "dt_conf", "depth", "root_cid", "src_line", "src_pos",
             "post_dt_kst", "before_post"]
    comments = pd.DataFrame(res.comments, columns=ccols)
    for col in ("dt_kst", "post_dt_kst"):
        comments[col] = pd.to_datetime(comments[col]).astype("datetime64[s]")

    # 끝 경계: 컷오프 다음 날 0시. 앞으로 받기(high)가 아직 컷오프 안이면 high 글 시각까지만 완전하다
    t_end = datetime.combine(date.fromisoformat(manifest["cutoff"]) + timedelta(days=1), time(0))
    if state and state.get("high") is not None and state["high"] < int(posts["no"].max()):
        t_end = min(t_end, posts.loc[posts["no"] <= state["high"], "dt_kst"].max().to_pydatetime())

    # 단위 표: 글(첫 줄 제목·본문) + 댓글(합집합). 조회·추천·다른의견은 싣지 않는다(§2.2 수치 필드)
    pu = posts.assign(utype="post", cid=-1, dt_conf="post", depth=np.nan, root_cid=np.nan, src_line=None,
                      src_pos=np.nan, post_dt_kst=posts["dt_kst"], before_post=False)
    cu = comments.assign(utype="comment", title=None)
    units = pd.concat([pu, cu], ignore_index=True, sort=False)
    units = units.sort_values(["dt_kst", "utype", "no", "cid"], kind="stable", na_position="last")
    units = units.reset_index(drop=True)
    for col in ("dt_kst", "post_dt_kst"):
        units[col] = units[col].astype("datetime64[s]")
    for col in ("trunc_first", "trunc_last", "trunc_any", "body_edited", "title_edited"):
        units[col] = units[col].astype("boolean")  # 글 행만 값이 있다
    for col in ("n_rows", "ncomment_first", "ncomment_last", "nsaved_first", "nsaved_last", "n_union",
                "cid", "depth", "root_cid", "src_pos"):
        units[col] = units[col].astype("Int64")
    units["before_post"] = units["before_post"].astype(bool)

    pdt, pno = _ns(posts["dt_kst"]), posts["no"].to_numpy()
    udt, updt = _ns(units["dt_kst"]), _ns(units["post_dt_kst"])
    data_lo = posts["dt_kst"].min()
    out = {}
    for market in ("kr", "us"):
        w = make_windows(cal, market)
        w = w[(w["close_kst"] > data_lo) & (w["close_kst"] <= t_end)].reset_index(drop=True)
        comp = np.isfinite(number_boundary(pdt, pno, _ns(w["open_kst"])))
        for v in w["variant"].unique():
            wv = w[(w["variant"] == v) & comp]  # 완전한 창에만 id를 붙인다(앞쪽 거꾸로 받기 중인 날은 None)
            col = ID_COL[(market, v)]
            units[col] = assign_window(udt, wv)
            if v == "main":
                units["p" + col] = assign_window(updt, wv)  # 부모 글의 창(글 행은 자기 창)
        out[market] = w

    is_c = units["utype"] == "comment"
    units["in120"] = is_c & (units["dt_kst"] >= units["post_dt_kst"]) & \
        (units["dt_kst"] < units["post_dt_kst"] + pd.Timedelta(minutes=121))
    for mk, market, col in (("kr", "kr", "pwin_kr"), ("u", "us", "pwin_u")):
        wm = out[market][out[market]["variant"] == "main"]
        close_of = units[col].map(dict(zip(wm["win_id"], wm["close_kst"])))
        units[f"pre_close_{mk}"] = pre_close_for_post_window(units, wm, col)
        # 순위 자격: 등록 분 ≤ 마감 분 − 120분(한국 05:59, 미국 19:29/20:29 KST)
        units[f"rank_ok_{mk}"] = (units["utype"] == "post") & close_of.notna() & \
            (units["dt_kst"] <= close_of - pd.Timedelta(minutes=121))
        cnt = units[units["in120"] & units[f"pre_close_{mk}"]].groupby("no").size()
        # 글별 관심: 등록 뒤 120분 안이면서 부모 글 창 마감 전인 댓글 수(글 창이 없으면 NaN)
        units[f"c120_{mk}"] = np.where((units["utype"] == "post") & close_of.notna(),
                                       units["no"].map(cnt).fillna(0), np.nan)

    for market in ("kr", "us"):
        parts = []
        for v, wv in out[market].groupby("variant", sort=False):
            parts.append(add_window_stats(wv, posts, units, t_end, ID_COL[(market, v)]))
        out[market] = pd.concat(parts, ignore_index=True)
    return {"posts": posts, "comments": comments, "units": units, "windows_kr": out["kr"],
            "windows_us": out["us"], "res_stats": res.stats, "t_end": t_end}


def comment_date_table(units: pd.DataFrame) -> pd.DataFrame:
    """날짜별 댓글 날짜 확정률. by = 'post_date'(부모 글 날짜) / 'comment_date'(놓은 날짜)."""
    c = units[units["utype"] == "comment"]
    recs = []
    for by, key in (("post_date", c["post_dt_kst"].dt.date), ("comment_date", c["dt_kst"].dt.date)):
        for d, s in c.groupby(key)["dt_conf"]:
            vc = s.value_counts()
            lt = int(vc.get("latest", 0))
            to = int(vc.get("matched", 0) + vc.get("row", 0) + lt)
            recs.append({"by": by, "date": str(d), "n": int(len(s)), "full": int(vc.get("full", 0)),
                         "matched": int(vc.get("matched", 0)), "row": int(vc.get("row", 0)),
                         "latest": lt, "time_only": to,
                         "conf_rate_time_only": (to - lt) / to if to else np.nan,
                         "conf_rate_all": 1 - lt / len(s)})
    return pd.DataFrame(recs)


def missing_by_day(posts: pd.DataFrame, t_end: datetime) -> pd.DataFrame:
    """KST 달력일별 번호 증가·저장 수·결번율(경계 = 0시 누적 최대 번호)."""
    pdt, pno = _ns(posts["dt_kst"]), posts["no"].to_numpy()
    days = pd.date_range(posts["dt_kst"].min().normalize(), pd.Timestamp(t_end).normalize(), freq="D")
    edges = _ns(days)
    b = number_boundary(pdt, pno, edges)
    sorted_no = np.sort(pno)
    recs = []
    for k in range(len(days) - 1):
        if not (np.isfinite(b[k]) and np.isfinite(b[k + 1])):
            continue
        d_no = b[k + 1] - b[k]
        saved = np.searchsorted(sorted_no, b[k + 1], "right") - np.searchsorted(sorted_no, b[k], "right")
        recs.append({"date": str(days[k].date()), "b_start": int(b[k]), "b_end": int(b[k + 1]), "d_no": int(d_no),
                     "saved": int(saved), "missing_rate": 1 - saved / d_no if d_no > 0 else np.nan})
    return pd.DataFrame(recs)


def summarize(b: dict, cd: pd.DataFrame, mb: pd.DataFrame, sc: pd.DataFrame, man: dict, cal: dict,
              salt: dict) -> dict:
    u, wk, wu = b["units"], b["windows_kr"], b["windows_us"]
    c = u[u["utype"] == "comment"]
    vc = c["dt_conf"].value_counts()
    lt = int(vc.get("latest", 0))
    to = int(vc.get("matched", 0) + vc.get("row", 0) + lt)

    def wsum(w, v):
        x = w[(w["variant"] == v) & w["complete"]]
        mr = x["missing_rate"].dropna()
        return {"n_windows": int(len(x)), "first": str(x["D"].min().date()) if len(x) else None,
                "last": str(x["D"].max().date()) if len(x) else None,
                "kinds": {k: int(n) for k, n in x["kind"].value_counts().items()},
                "over96": int(x["over96"].sum()), "xholiday_mismatch": int(x["xholiday_mismatch"].sum()),
                "dst_week": int(x["dst_week"].sum()),
                "missing_rate": {"median": round(float(mr.median()), 4), "mean": round(float(mr.mean()), 4),
                                 "p90": round(float(mr.quantile(0.9)), 4), "max": round(float(mr.max()), 4)}
                if len(mr) else None,
                "outage": int(x["outage"].fillna(False).sum()), "outage_unknown": int(x["outage"].isna().sum()),
                "spam_wave": int(x["spam_wave"].fillna(False).sum()),
                "n_posts": int(x["n_posts"].sum()), "n_comments": int(x["n_comments"].sum())}

    ov = sc["overlap_prev"].dropna()
    return {
        "snapshot": {"cutoff": man["cutoff"], "S": man["S"], "files": len(man["files"]),
                     "lines": int(sum(f["lines"] for f in man["files"].values())), "t_end": str(b["t_end"])},
        "calendar": {"source": cal["source"], "krx_days": len(cal["krx"]), "nyse_days": len(cal["nyse"]),
                     "check": cal.get("check", {})},
        "posts": int((u["utype"] == "post").sum()), "posts_multi_row": int((b["posts"]["n_rows"] > 1).sum()),
        "comments_unique": int(len(c)),
        "comment_dates": {"full": int(vc.get("full", 0)), "matched": int(vc.get("matched", 0)),
                          "row": int(vc.get("row", 0)), "latest": lt, "time_only": to,
                          "conf_rate_time_only": round((to - lt) / to, 4) if to else None,
                          "conf_rate_all": round(1 - lt / len(c), 4) if len(c) else None,
                          "before_post": int(c["before_post"].sum())},
        "resolve_stats": b["res_stats"],
        "units": {"total": int(len(u)), "posts": int((u["utype"] == "post").sum()), "comments": int(len(c)),
                  "in_win_kr": int(u["win_kr"].notna().sum()), "in_win_u": int(u["win_u"].notna().sum()),
                  "comments_pre_close_kr": int(u["pre_close_kr"].sum()),
                  "comments_pre_close_u": int(u["pre_close_u"].sum())},
        "truncated_posts": {"trunc_first": int(b["posts"]["trunc_first"].sum()),
                            "trunc_last": int(b["posts"]["trunc_last"].sum()),
                            "trunc_any": int(b["posts"]["trunc_any"].sum())},
        "body_edited_multi_row": int(b["posts"]["body_edited"].sum()),
        "windows_kr": {v: wsum(wk, v) for v in ("main", "fixed24")},
        "windows_us": {v: wsum(wu, v) for v in ("main", "close0900", "fixed24")},
        "missing_by_day": {"days": int(len(mb)),
                           "median": round(float(mb["missing_rate"].median()), 4) if len(mb) else None,
                           "max": round(float(mb["missing_rate"].max()), 4) if len(mb) else None},
        "salt": {"present": salt.get("present"),
                 "min_overlap_prev": round(float(ov.min()), 4) if len(ov) else None,
                 "median_overlap_prev": round(float(ov.median()), 4) if len(ov) else None,
                 "flag_drop_days": sc.loc[sc["flag_drop"], "date"].tolist() if len(sc) else []},
    }


def run(cutoff: str = CUTOFF, posts_dir: Path = POSTS_DIR, results: Path = RESULTS,
        manifest_path: Path | None = None) -> dict:
    manifest = json.loads(Path(manifest_path).read_text("utf-8")) if manifest_path else None
    rows, man = read_snapshot(posts_dir, cutoff, manifest)
    salt = salt_check()
    man["salt_fingerprint"] = salt.get("fingerprint")  # 값이 아니라 바뀜 확인용 지문
    man["state_high"], man["state_low"] = salt.get("high"), salt.get("low")
    cal = trading_calendars(CAL_START, date.fromisoformat(cutoff), results / "loader_calendar.parquet")
    b = build(rows, man, cal, {"high": salt.get("high")})
    results.mkdir(parents=True, exist_ok=True)
    units = b["units"]
    units.to_parquet(results / "units.parquet", index=False)
    b["windows_kr"].to_parquet(results / "windows_kr.parquet", index=False)
    b["windows_us"].to_parquet(results / "windows_us.parquet", index=False)
    cd = comment_date_table(units)
    cd.to_csv(results / "loader_comment_dates.csv", index=False, encoding="utf-8-sig")
    mb = missing_by_day(b["posts"], b["t_end"])
    mb.to_csv(results / "loader_missing_by_day.csv", index=False, encoding="utf-8-sig")
    sc = salt_continuity(units)
    sc.to_csv(results / "loader_salt_continuity.csv", index=False, encoding="utf-8-sig")
    if manifest_path is None:
        (results / "loader_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), "utf-8")
    summary = summarize(b, cd, mb, sc, man, cal, salt)
    (results / "loader_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                                 "utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="py loader.py")
    ap.add_argument("--cutoff", default=CUTOFF, help="이 날짜 이하 파일만 쓴다")
    ap.add_argument("--manifest", default=None,
                    help="이전 스냅샷 목록(loader_manifest.json)으로 같은 바이트만 다시 읽는다")
    a = ap.parse_args(argv)
    s = run(a.cutoff, manifest_path=Path(a.manifest) if a.manifest else None)
    print(json.dumps(s, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
