"""실적 뒤 흐름 — 실적 발표 뒤 조건부 규칙(B1·B4·A2)에 맞는 종목을 자동으로 쌓고 성적을 채점한다(desk_system.md §15).

근거 연구: data/derived/analysis/earnings_trend/rules_report.md (1,629종목 12,441건, 2026 상반기로 고르고 7~9월로 확인,
우연 기준선·보통 날·대형주 점검). 실전 추적은 연구 구간이 끝난 뒤(반응일 2026-10-01부터)만 센다.
관찰 화면이다: 매수 추천·목표 비중·주문 수량을 만들지 않고 7종목 엔진에 들어가지 않는다.

흐름: 나스닥 실적 달력(누가 언제 발표) → SEC 8-K 2.02 접수 시각(장전·장후 판정, 없으면 나스닥 시각)
      → 반응일 종가까지의 가격 특징(연구 events_wide.py와 같은 정의) → 규칙 판정 → 이후 SPY 대비 성적 누적 → 경보.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time as _time
import urllib.error
import urllib.request
import zlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import watchlist

from . import config
from .news import market_calendar, now_iso, write_atomic

STATE = config.CACHE / "erules.json"
LIVE_START = "2026-10-01"          # 연구 확인 구간(2026-07~09) 다음 날부터 실전으로 센다
MAX_AGE_SEC = 3600
MIN_CAP_NOW = 3e9                  # 연구 표본과 같은 문턱(지금 시총), 사건 시점 시총 20억 달러 미만은 뺀다
MIN_CAP_EVENT = 2e9
DEDUP_SESSIONS = 31                # 45일(약 31거래일) 안 재발표는 앞의 것만
NEXT_CAP_SESSIONS = 90             # 다음 실적을 못 찾으면 90거래일에서 닫는다
CAL_REFETCH_DAYS = 3               # 나스닥 달력은 늦게 고쳐지므로 최근 3일은 다시 받는다
SEC_WINDOW_DAYS = 3                # 달력 날짜와 8-K 접수일의 허용 차이
ALARM_MIN_N = 10                   # 닫힌 건수가 이만큼 쌓여야 경보를 판정한다
BENCH = "SPY"
NASDAQ_CAL = "https://api.nasdaq.com/api/calendar/earnings?date={day}"
NASDAQ_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
SEC_HEADERS = {"User-Agent": "GBRC-research personal-noncommercial research@example.com", "Accept-Encoding": "gzip, deflate"}
SEC_GAP_SEC = 0.15

# 규칙 정본. 문턱·연구 성적은 rules_report.md §2와 같다(바꾸려면 연구를 다시 돌리고 이 표와 §15를 함께 고친다).
RULES = {
    "B1": {"side": "buy", "label": "투매형 반등", "entry": "반응일 종가", "horizon": "20거래일", "out": "h20",
           "conditions": ["52주 고점 −30% 이하", "평소 하루 변동 3.5% 이상", "실적날 종가가 그날 범위 아래 25%"],
           "research": {"2026A": [0.586, 58, 0.405], "2026B": [0.604, 53, 0.409], "2025": [0.574, 101, 0.509]}},
    "B4": {"side": "buy", "label": "기술주 거래량 폭발", "entry": "반응일 종가", "horizon": "다음 실적 전날까지", "out": "nx",
           "conditions": ["기술 업종", "평소 하루 변동 3.5% 이상", "실적날 거래량 50일 평균의 3배 이상(방향 무관)"],
           "research": {"2026A": [0.725, 80, 0.461], "2026B": [0.629, 35, 0.344], "2025": [0.385, 78, 0.471]}},
    "A2": {"side": "avoid", "label": "중형주 추격", "entry": "반응일 5거래일 뒤 종가", "horizon": "20거래일", "out": "e5h20",
           "conditions": ["시총 70~300억 달러", "실적날 SPY 대비 +2~8%", "그 뒤 5일 SPY 대비 +3% 이상 더 오름"],
           "research": {"2026A": [0.264, 72, 0.456], "2026B": [0.238, 42, 0.401], "2025": [0.556, 124, 0.513]}},
}


class ErulesError(ValueError):
    """화면에 그대로 보여 줄 실적 뒤 흐름 오류."""


# ----------------------------------------------------------------------------- 판정(순수 함수)
def features(c: np.ndarray, h: np.ndarray, lo: np.ndarray, v: np.ndarray, spy: np.ndarray, k: int,
             cap_now: float | None, px_last: float | None) -> dict | None:
    """반응일 위치 k에서 연구(events_wide.py)와 같은 정의로 계산한다. k+5 종가가 있으면 d5_x도."""
    if k < 61 or k >= len(c) or not (c[k - 1] > 0) or np.isnan(c[k]):
        return None
    p0 = c[k - 1]
    r = np.diff(np.log(c), prepend=np.nan)
    vol60 = float(np.nanstd(r[k - 60:k]) * 100)
    avgv = float(np.nanmean(v[k - 51:k - 1]))
    hi252 = float(np.nanmax(h[max(0, k - 253):k - 1]))
    r0 = (c[k] / p0 - 1) * 100
    f = {
        "r0_x": r0 - (spy[k] / spy[k - 1] - 1) * 100,
        "vol60": vol60,
        "vol_ratio": float(v[k] / avgv) if avgv > 0 else None,
        "dist_hi": (p0 / hi252 - 1) * 100 if hi252 > 0 else None,
        "clv": float((c[k] - lo[k]) / (h[k] - lo[k])) if h[k] > lo[k] else None,
        "mcap_b": cap_now * p0 / px_last / 1e9 if cap_now and px_last and px_last > 0 else None,
    }
    if k + 5 < len(c) and not np.isnan(c[k + 5]):
        f["d5_x"] = (c[k + 5] / c[k] - 1) * 100 - (spy[k + 5] / spy[k] - 1) * 100
    return {key: (round(val, 4) if isinstance(val, float) else val) for key, val in f.items()}


def matches(f: dict, sector: str) -> tuple[list[str], bool]:
    """(맞은 규칙, A2 판정 대기 여부). A2는 반응 뒤 5거래일 종가가 있어야 판정한다."""
    out, wait = [], False
    if f is None:
        return out, wait
    hv = f.get("vol60") is not None and f["vol60"] >= 3.5
    if hv and f.get("dist_hi") is not None and f["dist_hi"] <= -30 and f.get("clv") is not None and f["clv"] <= 0.25:
        out.append("B1")
    if hv and sector == "Technology" and f.get("vol_ratio") is not None and f["vol_ratio"] >= 3:
        out.append("B4")
    cap, r0 = f.get("mcap_b"), f.get("r0_x")
    if cap is not None and 7 <= cap < 30 and r0 is not None and 2 <= r0 < 8:
        if "d5_x" not in f:
            wait = True
        elif f["d5_x"] >= 3:
            out.append("A2")
    return out, wait


def excess(c: np.ndarray, spy: np.ndarray, a: int, b: int) -> float | None:
    if a < 0 or b >= len(c) or b < a or np.isnan(c[a]) or np.isnan(c[b]) or not (c[a] > 0):
        return None
    return round(float((c[b] / c[a] - 1) * 100 - (spy[b] / spy[a] - 1) * 100), 3)


def outcome(c: np.ndarray, spy: np.ndarray, start: int, end: int | None, last: int) -> dict:
    """start 종가 진입 → end 종가. end가 아직 오지 않았으면 마지막 확정 종가까지의 값(닫히지 않음)."""
    if start > last:
        return {"closed": False, "x": None, "days": 0, "end": None}
    stop = min(end, last) if end is not None else last
    return {"closed": end is not None and end <= last, "x": excess(c, spy, start, stop), "days": int(stop - start),
            "end": int(end) if end is not None else None}


def rule_stats(events: list[dict], code: str) -> dict:
    """닫힌 실전 사례의 SPY 대비 승률, 같은 결과 칸의 전체 사건 기준 승률, 경보 상태."""
    rule = RULES[code]
    key = rule["out"]
    done = [e["out"][key]["x"] for e in events
            if code in e.get("matches", []) and e.get("out", {}).get(key, {}).get("closed") and e["out"][key]["x"] is not None]
    base = [e["out"][key]["x"] for e in events
            if e.get("out", {}).get(key, {}).get("closed") and e["out"][key]["x"] is not None]
    n = len(done)
    win = sum(x > 0 for x in done) / n if n else None
    if n < ALARM_MIN_N:
        state = "building"
    elif (rule["side"] == "buy" and win < 0.5) or (rule["side"] == "avoid" and win > 0.5):
        state = "alarm"
    else:
        state = "holding"
    return {"n": n, "win": round(win, 3) if win is not None else None,
            "mean": round(float(np.mean(done)), 2) if n else None,
            "median": round(float(np.median(done)), 2) if n else None,
            "base_n": len(base), "base_win": round(sum(x > 0 for x in base) / len(base), 3) if base else None,
            "state": state}


def reaction_day(cal, accepted: str | None, cal_day: str, nasdaq_time: str,
                 yf_time: str | None = None) -> tuple[str | None, str]:
    """반응 세션(YYYY-MM-DD)과 근거. 시각 출처 순서: SEC 8-K 접수 → 나스닥 장전·장후 → yfinance 시각.
    16:00(미 동부) 이후면 다음 세션, 전이면 그날(휴일이면 다음) 세션. yfinance는 연구처럼 16시 이후=장후,
    04:00~09:30=장전만 믿고 나머지('12:00' 같은 미상 시각)는 버린다."""
    if accepted:
        d, hm = accepted.split(" ")
        after = int(hm[:2]) * 60 + int(hm[3:5]) >= 16 * 60
        src = "sec"
    elif nasdaq_time in ("time-after-hours", "time-pre-market"):
        d, after, src = cal_day, nasdaq_time == "time-after-hours", "nasdaq"
    elif yf_time:
        d, hm = yf_time.split(" ")
        minutes = int(hm[:2]) * 60 + int(hm[3:5])
        if minutes >= 16 * 60:
            after = True
        elif 4 * 60 <= minutes < 9 * 60 + 30:
            after = False
        else:
            return None, "unknown"
        src = "yf"
    else:
        return None, "unknown"
    s = cal.date_to_session(d, direction="next")
    if after and s.strftime("%Y-%m-%d") == d:
        s = cal.next_session(s)
    return s.strftime("%Y-%m-%d"), src


# ----------------------------------------------------------------------------- 자료 받기(네트워크)
def _get(url: str, headers: dict, timeout: int = 30) -> bytes:
    r = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)
    body = r.read()
    enc = (r.headers.get("Content-Encoding") or "").lower()
    if enc == "gzip":
        body = gzip.decompress(body)
    elif enc == "deflate":
        body = zlib.decompress(body)
    return body


def fetch_calendar(day: str) -> list[dict]:
    rows = (json.loads(_get(NASDAQ_CAL.format(day=day), NASDAQ_HEADERS)).get("data") or {}).get("rows") or []
    return [{"symbol": (x.get("symbol") or "").strip().upper(), "name": x.get("name") or "", "time": x.get("time") or ""}
            for x in rows if x.get("symbol")]


class Sec:
    def __init__(self) -> None:
        self.last = 0.0

    def json(self, url: str):
        wait = SEC_GAP_SEC - (_time.time() - self.last)
        if wait > 0:
            _time.sleep(wait)
        self.last = _time.time()
        return json.loads(_get(url, SEC_HEADERS, timeout=60))

    def earnings_8k(self, cik: int) -> list[str]:
        """최근 8-K 중 Item 2.02 접수 시각(미 동부 'YYYY-MM-DD HH:MM')."""
        rec = self.json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")["filings"]["recent"]
        out = []
        for i, form in enumerate(rec["form"]):
            if form == "8-K" and "2.02" in (rec["items"][i] or ""):
                et = pd.Timestamp(rec["acceptanceDateTime"][i]).tz_convert(config.NY)
                out.append(et.strftime("%Y-%m-%d %H:%M"))
        return out


def fetch_prices(tickers: list[str], start: str, fetched: datetime) -> dict[str, pd.DataFrame]:
    """일봉 OHLCV(배당·분할 수정, 연구와 같음). 아직 확정되지 않은 오늘 봉은 뺀다."""
    import yfinance as yf

    from .forecast import final_close
    parts = {k: [] for k in ("Close", "High", "Low", "Volume")}
    allt = sorted(set(tickers) | {BENCH})
    for i in range(0, len(allt), 200):
        chunk = allt[i:i + 200]
        df = yf.download(chunk, start=start, auto_adjust=True, progress=False, threads=True)
        for k in parts:
            parts[k].append(df[k] if isinstance(df.columns, pd.MultiIndex) else df[[k]].rename(columns={k: chunk[0]}))
    out = {k: pd.concat(v, axis=1).sort_index() for k, v in parts.items()}
    keep = [d for d in out["Close"].index if final_close(d.strftime("%Y-%m-%d"), fetched)]
    out = {k: v.loc[keep, ~v.columns.duplicated()] for k, v in out.items()}
    if BENCH not in out["Close"] or out["Close"][BENCH].dropna().empty:
        raise ErulesError("SPY 가격을 받지 못해 기존 원장을 유지합니다")
    return out


# ----------------------------------------------------------------------------- 원장 갱신
def empty_state() -> dict:
    return {"version": 1, "live_start": LIVE_START, "calendar": {}, "sec": {}, "ciks": {}, "ciks_at": None,
            "events": {}, "updated_at": None, "updated_epoch": 0, "latest_close": None, "warnings": []}


def read_state(path: Path = STATE) -> dict:
    if not path.exists():
        return empty_state()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ErulesError("실적 뒤 흐름 원장은 version 1이어야 합니다")
    return data


def _ny_today() -> str:
    return datetime.now(config.NY).date().isoformat()


def refresh_calendar(state: dict, universe: dict, today: str, warnings: list[str]) -> None:
    """LIVE_START부터 오늘까지 평일 달력을 받아 유니버스 종목만 남긴다(이미 받은 날은 최근 3일만 다시)."""
    start = (date.fromisoformat(LIVE_START) - timedelta(days=7)).isoformat()
    redo_from = (date.fromisoformat(today) - timedelta(days=CAL_REFETCH_DAYS)).isoformat()
    for d in pd.bdate_range(start, today):
        day = d.strftime("%Y-%m-%d")
        if day in state["calendar"] and day < redo_from:
            continue
        try:
            rows = fetch_calendar(day)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            warnings.append(f"나스닥 달력 {day} 실패: {type(exc).__name__}")
            continue
        seen, keep = set(), []
        for r in rows:
            u = universe.get(r["symbol"])
            if not u or (u.get("cap") or 0) < MIN_CAP_NOW:
                continue
            name = (u.get("name") or r["name"]).split(" Class ")[0]
            if name in seen:  # GOOG·GOOGL처럼 한 회사의 두 주식은 앞 것만
                continue
            seen.add(name)
            keep.append({"symbol": r["symbol"], "time": r["time"]})
        state["calendar"][day] = keep


def refresh_timing(state: dict, candidates: list[tuple[str, str]], warnings: list[str]) -> None:
    """후보 종목의 SEC 8-K 2.02 접수 시각을 받아 둔다. 403·429면 멈추고 나스닥 시각으로 넘어간다."""
    sec = Sec()
    if not state.get("ciks") or (state.get("ciks_at") or "") < (date.today() - timedelta(days=7)).isoformat():
        try:
            data = sec.json("https://www.sec.gov/files/company_tickers.json")
            state["ciks"] = {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}
            state["ciks_at"] = date.today().isoformat()
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            warnings.append(f"SEC 종목표 실패: {type(exc).__name__}")
    for sym, day in candidates:
        cached = state["sec"].get(sym)
        if cached and cached["fetched"] >= day:
            continue
        cik = state["ciks"].get(sym.replace("-", ".")) or state["ciks"].get(sym)
        if not cik:
            state["sec"][sym] = {"fetched": _ny_today(), "accepts": []}
            continue
        try:
            accepts = sec.earnings_8k(cik)
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                warnings.append(f"SEC가 HTTP {exc.code}로 거절 — 남은 종목은 나스닥 시각으로 판정")
                return
            accepts = []
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
            accepts = []
        floor = (date.fromisoformat(LIVE_START) - timedelta(days=10)).isoformat()  # 9월 말 장후 발표 → 10/1 반응도 잡는다
        state["sec"][sym] = {"fetched": _ny_today(), "accepts": [a for a in accepts if a >= floor]}


def refresh_yf_times(state: dict, pairs: list[tuple[str, str]], warnings: list[str]) -> None:
    """SEC·나스닥 시각이 모두 없는 소수 사건(주로 해외 발행사 ADR)만 yfinance 실적 일정으로 시각을 찾는다."""
    import yfinance as yf
    cache = state.setdefault("yf_times", {})
    for sym, day in pairs:
        if (cache.get(sym) or {}).get("fetched", "") >= day:
            continue
        try:
            f = yf.Ticker(sym).get_earnings_dates(limit=8)
            rows = [] if f is None or f.empty else [ts.tz_convert(config.NY).strftime("%Y-%m-%d %H:%M") for ts in f.index]
        except Exception as exc:  # Yahoo가 막거나 종목이 없으면 그 사건은 시각 미상으로 남는다
            warnings.append(f"yfinance 시각 {sym} 실패: {type(exc).__name__}")
            rows = []
        cache[sym] = {"fetched": _ny_today(), "times": rows}


def _yf_time_near(state: dict, sym: str, day: str) -> str | None:
    times = ((state.get("yf_times") or {}).get(sym) or {}).get("times") or []
    near = [t for t in times if abs((date.fromisoformat(t[:10]) - date.fromisoformat(day)).days) <= SEC_WINDOW_DAYS]
    return min(near, key=lambda t: abs((date.fromisoformat(t[:10]) - date.fromisoformat(day)).days)) if near else None


def _accepted_near(state: dict, sym: str, day: str) -> str | None:
    accepts = (state["sec"].get(sym) or {}).get("accepts") or []
    near = [a for a in accepts if abs((date.fromisoformat(a[:10]) - date.fromisoformat(day)).days) <= SEC_WINDOW_DAYS]
    return min(near, key=lambda a: abs((date.fromisoformat(a[:10]) - date.fromisoformat(day)).days)) if near else None


def build_events(state: dict, universe: dict, cal) -> None:
    """달력 + 접수 시각 → 사건(반응일). 같은 종목 31거래일 안 재발표는 앞의 것만 남긴다."""
    found, unknown = {}, []
    for day in sorted(state["calendar"]):
        for r in state["calendar"][day]:
            acc = _accepted_near(state, r["symbol"], day)
            rd, src = reaction_day(cal, acc, day, r["time"], _yf_time_near(state, r["symbol"], day))
            if rd is None:
                unknown.append(f"{r['symbol']} {day}")  # SEC 접수도 나스닥 시각도 없어 장전·장후를 모른다
                continue
            if rd < LIVE_START:
                continue
            key = f"{r['symbol']}|{rd}"
            u = universe.get(r["symbol"], {})
            found.setdefault(key, {"ticker": r["symbol"], "name": u.get("name") or "", "sector": u.get("sector") or "",
                                   "ai": watchlist.AI_SUBSECTOR.get(r["symbol"], ""), "cal_day": day,
                                   "timing": {"src": src, "accepted": acc, "nasdaq": r["time"],
                                              "yf": _yf_time_near(state, r["symbol"], day) if src == "yf" else None},
                                   "reaction": rd})
    sessions = {}
    by_ticker = {}
    for key, ev in sorted(found.items(), key=lambda kv: kv[1]["reaction"]):
        by_ticker.setdefault(ev["ticker"], []).append((key, ev))
    for t, evs in by_ticker.items():
        last = None
        for key, ev in evs:
            if last is not None:
                n = len(cal.sessions_in_range(last, ev["reaction"])) - 1
                if n < DEDUP_SESSIONS:
                    continue
            last = ev["reaction"]
            old = state["events"].get(key, {})
            state["events"][key] = {**ev, **{k: old[k] for k in ("features", "matches", "wait_a2", "out", "status") if k in old}}
            sessions[key] = ev["reaction"]
    # 달력에서 사라진(날짜가 바뀐) 사건은 지운다 — 같은 종목의 새 날짜 사건이 대신 생긴다
    for key in [k for k in state["events"] if k not in sessions]:
        del state["events"][key]
    state["unknown_timing"] = sorted(set(unknown))


def score_events(state: dict, prices: dict[str, pd.DataFrame], universe: dict) -> None:
    """특징·규칙·결과를 다시 계산한다(가격이 있는 사건만). 닫힌 결과도 같은 가격에서 같은 값이 나온다."""
    C, H, L, V = prices["Close"], prices["High"], prices["Low"], prices["Volume"]
    days = [d.strftime("%Y-%m-%d") for d in C.index]
    pos = {d: i for i, d in enumerate(days)}
    last = len(days) - 1
    spy = C[BENCH].to_numpy(dtype=float)
    by_ticker = {}
    for key, ev in state["events"].items():
        by_ticker.setdefault(ev["ticker"], []).append(key)
    for t, keys in by_ticker.items():
        keys.sort(key=lambda k: state["events"][k]["reaction"])
        if t not in C or C[t].dropna().empty:
            for k in keys:
                state["events"][k]["status"] = "no_price"
            continue
        c, h, lo, v = (X[t].to_numpy(dtype=float) for X in (C, H, L, V))
        valid = np.where(~np.isnan(c))[0]
        px_last = c[valid[-1]]
        cap_now = (universe.get(t) or {}).get("cap")
        for i, key in enumerate(keys):
            ev = state["events"][key]
            k = pos.get(ev["reaction"])
            if k is None:
                ev["status"] = "waiting"  # 반응일 종가가 아직 확정되지 않음
                continue
            f = features(c, h, lo, v, spy, k, cap_now, px_last)
            if f is None:
                ev["status"] = "no_history"
                continue
            if f.get("mcap_b") is not None and f["mcap_b"] * 1e9 < MIN_CAP_EVENT:
                ev["status"] = "small"
                continue
            ev["features"] = f
            ev["matches"], ev["wait_a2"] = matches(f, ev.get("sector", ""))
            nk = pos.get(state["events"][keys[i + 1]]["reaction"]) if i + 1 < len(keys) else None
            nx_end = (nk - 1) if nk is not None else k + NEXT_CAP_SESSIONS
            ev["out"] = {"h20": outcome(c, spy, k, k + 20, last), "nx": outcome(c, spy, k, nx_end, last),
                         "e5h20": outcome(c, spy, k + 5, k + 25, last)}
            ev["out"]["nx"]["capped"] = nk is None
            ev["status"] = "ok"
    state["latest_close"] = days[-1] if days else None


def refresh(state: dict | None = None, path: Path = STATE, universe: dict | None = None) -> dict:
    """달력·접수 시각·가격을 새로 받아 원장을 고친다. 실패한 부분은 경고로 남기고 이전 값을 유지한다."""
    from . import dip
    state = state if state is not None else read_state(path)
    warnings: list[str] = []
    universe = universe if universe is not None else dip.universe()
    today = _ny_today()
    cal = market_calendar()
    refresh_calendar(state, universe, today, warnings)
    cands = sorted({(r["symbol"], d) for d, rows in state["calendar"].items() for r in rows})
    refresh_timing(state, cands, warnings)
    no_time = [(s, d) for s, d in cands if not _accepted_near(state, s, d)
               and next((r["time"] for r in state["calendar"][d] if r["symbol"] == s), "") not in ("time-after-hours", "time-pre-market")]
    refresh_yf_times(state, no_time, warnings)
    build_events(state, universe, cal)
    tickers = sorted({ev["ticker"] for ev in state["events"].values()})
    if tickers:
        start = (date.fromisoformat(LIVE_START) - timedelta(days=420)).isoformat()
        try:
            prices = fetch_prices(tickers, start, datetime.now(timezone.utc))
            score_events(state, prices, universe)
        except Exception as exc:  # 가격 실패는 원장을 지우지 않고 경고만 남긴다
            warnings.append(f"가격 갱신 실패 · 저장 원장 사용: {type(exc).__name__}: {exc}")
    state.update({"updated_at": now_iso(), "updated_epoch": _time.time(), "warnings": warnings})
    write_atomic(path, state)
    return state


# ----------------------------------------------------------------------------- 화면 자료
def _row(ev: dict, code: str | None = None) -> dict:
    f = ev.get("features") or {}
    key = RULES[code]["out"] if code else "h20"
    o = (ev.get("out") or {}).get(key) or {}
    return {"ticker": ev["ticker"], "name": ev.get("name", ""), "ai": ev.get("ai", ""), "sector": ev.get("sector", ""),
            "reaction": ev["reaction"], "timing": ev["timing"]["src"], "mcap_b": f.get("mcap_b"), "r0_x": f.get("r0_x"),
            "d5_x": f.get("d5_x"), "vol_ratio": f.get("vol_ratio"), "dist_hi": f.get("dist_hi"), "clv": f.get("clv"),
            "vol60": f.get("vol60"), "matches": ev.get("matches", []), "wait_a2": ev.get("wait_a2", False),
            "x": o.get("x"), "days": o.get("days"), "closed": o.get("closed", False), "capped": o.get("capped", False)}


def payload(force: bool = False, path: Path = STATE, state: dict | None = None) -> dict:
    warning = None
    if state is None:
        state = read_state(path)
        if force or _time.time() - state.get("updated_epoch", 0) > MAX_AGE_SEC:
            try:
                state = refresh(state, path)
            except Exception as exc:
                warning = f"갱신 실패 · 저장 원장 사용: {type(exc).__name__}: {exc}"
    events = [e for e in state["events"].values() if e.get("status") == "ok"]
    rules = []
    for code, rule in RULES.items():
        hits = sorted((e for e in events if code in e.get("matches", [])), key=lambda e: e["reaction"], reverse=True)
        rules.append({"code": code, **rule, "live": rule_stats(events, code),
                      "open": [_row(e, code) for e in hits if not e["out"][rule["out"]]["closed"]],
                      "closed": [_row(e, code) for e in hits if e["out"][rule["out"]]["closed"]]})
    recent_from = sorted({e["reaction"] for e in events})[-10:]
    recent = sorted((_row(e) for e in events if recent_from and e["reaction"] >= recent_from[0]),
                    key=lambda r: (r["reaction"], -(r["mcap_b"] or 0)), reverse=True)
    status = {}
    for e in state["events"].values():
        status[e.get("status", "pending")] = status.get(e.get("status", "pending"), 0) + 1
    return {"live_start": state.get("live_start", LIVE_START), "updated_at": state.get("updated_at"),
            "latest_close": state.get("latest_close"), "events_n": len(events), "status_counts": status,
            "calendar_days": len(state.get("calendar", {})), "unknown_timing": state.get("unknown_timing", []),
            "rules": rules, "recent": recent,
            "warning": warning or ("; ".join(state.get("warnings") or []) or None),
            "alarm_min_n": ALARM_MIN_N,
            "method": "나스닥 실적 달력으로 발표 종목을 찾고, SEC 8-K 2.02 접수 시각으로 장전·장후를 가려 반응일을 정한다(없으면 나스닥 시각). "
                      "특징·규칙은 연구(rules_report.md)와 같은 정의다. 성적은 진입 종가부터의 SPY 대비 초과수익이며 수정 종가(배당·분할 반영)로 잰다. "
                      "기준 승률은 같은 결과 칸이 닫힌 모든 실적 사건의 승률이다."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="실적 뒤 흐름 규칙 추적(§15)")
    parser.add_argument("command", choices=("show", "refresh", "summary"), nargs="?", default="summary")
    args = parser.parse_args(argv)
    try:
        if args.command == "refresh":
            state = refresh()
            print(f"갱신: 사건 {len(state['events'])}건 · 최근 종가 {state['latest_close']} · 경고 {len(state['warnings'])}건")
            for w in state["warnings"][:10]:
                print(" -", w)
        data = payload(force=False)
        if args.command == "show":
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"실적 뒤 흐름 · 실전 시작 {data['live_start']} · 사건 {data['events_n']}건 · 최근 종가 {data['latest_close']}")
            for r in data["rules"]:
                lv = r["live"]
                win = f"{lv['win'] * 100:.0f}%" if lv["win"] is not None else "—"
                base = f"{lv['base_win'] * 100:.0f}%" if lv["base_win"] is not None else "—"
                print(f"  {r['code']} {r['label']}: 닫힘 {lv['n']}건 승률 {win}(기준 {base}) · 진행 {len(r['open'])}건 · {lv['state']}")
        return 0
    except (ErulesError, OSError, json.JSONDecodeError) as exc:
        print(f"실적 뒤 흐름 오류: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
