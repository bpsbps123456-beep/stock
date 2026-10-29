"""실적 뒤 흐름 — 2026년에 강한 실적 후 조건부 규칙 8개(매수 K1~K5, 회피 X1~X3)를 자동으로 쌓고 채점한다(desk_system.md §15).

근거 연구: data/derived/analysis/earnings_trend/rules_report.md §7(엄선). 1,629종목 12,441건에서 2026 상반기로 고르고 7~9월로 확인한
규칙 325개 중, 실제로 살 수 있는 '다음 날 정규장 시가' 진입으로 다시 재서 2026 세 시즌 모두 강한 것만 남겼다(code/select2026.py).
실전 추적은 연구 구간이 끝난 뒤(반응일 2026-10-01부터)만 센다. 관찰 화면이다: 매수 추천·목표 비중·주문 수량을 만들지 않고
7종목 엔진에 들어가지 않는다.

흐름: 나스닥 실적 달력(누가 언제 발표) → SEC 8-K 2.02 접수 시각(장전·장후, 없으면 나스닥·yfinance 시각)
      → 반응일 종가까지의 가격 특징(연구 events_wide.py와 같은 정의) → 규칙 판정(EPS 조건은 필요한 종목만 yfinance로)
      → 다음 날 시가(E0) 또는 6거래일째 시가(E5) 진입 기준 SPY 대비 성적 누적 → 경보.
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
SEC_WINDOW_DAYS = 3                # 달력 날짜와 8-K 접수일·yfinance 일정의 허용 차이
ALARM_MIN_N = 10                   # 닫힌 건수가 이만큼 쌓여야 경보를 판정한다
BENCH = "SPY"
NASDAQ_CAL = "https://api.nasdaq.com/api/calendar/earnings?date={day}"
NASDAQ_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
SEC_HEADERS = {"User-Agent": "GBRC-research personal-noncommercial research@example.com", "Accept-Encoding": "gzip, deflate"}
SEC_GAP_SEC = 0.15

# 결과 칸 → (진입 거래일 오프셋, 끝 오프셋(None = 다음 실적 반응일 전날)). 진입은 그날 정규장 시가, 끝은 종가.
OUTCOMES = {"o_h20": (1, 20), "o_nx": (1, None), "o5_h20": (6, 25), "o5_nx": (6, None)}
ENTRY_TEXT = {1: "반응 다음 날 정규장 시가", 6: "반응 6거래일째 정규장 시가(5일 뒤 종가를 보고 다음 날)"}


def _rng(v, lo=None, hi=None):
    if v is None:
        return None
    return (lo is None or v >= lo) and (hi is None or v < hi)


def _le(v, hi):
    return None if v is None else v <= hi


# 조건 정의 — 연구 code/rules.py predicates()와 같은 문턱(값이 아직 없으면 None = 판정 대기).
PRED = {
    "vol_hi": ("실적날 거래량이 직전 50거래일 평균의 3배 이상", lambda f, s: _rng(f.get("vol_ratio"), 3)),
    "hivol": ("평소 하루 변동(60거래일 표준편차) 3.5% 이상", lambda f, s: _rng(f.get("vol60"), 3.5)),
    "tech": ("나스닥 업종 Technology", lambda f, s: s == "Technology"),
    "indus": ("나스닥 업종 Industrials", lambda f, s: s == "Industrials"),
    "fin": ("나스닥 업종 Finance", lambda f, s: s == "Finance"),
    "eps_ok": ("EPS 서프라이즈 +3~15%", lambda f, s: _rng(f.get("eps"), 3, 15)),
    "eps_big": ("EPS 서프라이즈 +15% 이상", lambda f, s: _rng(f.get("eps"), 15)),
    "d5_dn": ("반응 뒤 5거래일 SPY 대비 −3% 이하", lambda f, s: _le(f.get("d5_x"), -3)),
    "d5_up": ("반응 뒤 5거래일 SPY 대비 +3% 이상", lambda f, s: _rng(f.get("d5_x"), 3)),
    "sig_down": ("실적날 하락이 평소 하루 변동의 3배 이상", lambda f, s: _le(f.get("r0_sig"), -3)),
    "sig_up": ("실적날 상승이 평소 하루 변동의 3배 이상", lambda f, s: _rng(f.get("r0_sig"), 3)),
    "gapdn_cont": ("−3% 이하 갭 하락 뒤 장중 −1% 더 밀림",
                   lambda f, s: None if f.get("gap") is None or f.get("o2c") is None else f["gap"] <= -3 and f["o2c"] <= -1),
    "gap_go": ("+3% 이상 갭 상승 뒤 장중 +1% 더 오름",
               lambda f, s: None if f.get("gap") is None or f.get("o2c") is None else f["gap"] >= 3 and f["o2c"] >= 1),
    "below_ma": ("발표 전날 종가가 50·200일선 모두 아래", lambda f, s: None if f.get("below_ma") is None else f["below_ma"] == 1),
    "large": ("사건 시점 시총 300~2,000억 달러", lambda f, s: _rng(f.get("mcap_b"), 30, 200)),
    "midcap": ("사건 시점 시총 70~300억 달러", lambda f, s: _rng(f.get("mcap_b"), 7, 30)),
    "pre60_up": ("발표 전 60거래일 SPY 대비 +15% 이상", lambda f, s: _rng(f.get("pre60_x"), 15)),
    "r0_up": ("실적날 SPY 대비 +2~8%", lambda f, s: _rng(f.get("r0_x"), 2, 8)),
}
EPS_PREDS = {"eps_ok", "eps_big"}


def _research(n, win, base, median, seasons, big, y2025):
    return {"n": n, "win": win, "base": base, "median": median, "seasons": seasons, "big": big, "y2025": y2025}


# 규칙 정본(2026 엄선 8개). 연구 수치는 다음 날 시가 진입 기준이다(rules_report.md §7, results/curated_final.json).
# seasons = 2026 1~3월·4~6월·7~9월 [승률, 건수], big = 사건 시점 시총 100억 달러 이상만 [승률, 건수], y2025 = [승률, 기준](참고).
RULES = {
    "K1": {"side": "buy", "label": "실적 괜찮은데 5일 밀림", "preds": ["vol_hi", "eps_ok", "d5_dn"], "out": "o5_h20",
           "horizon": "반응일 +25거래일 종가까지(약 19거래일 보유)",
           "research": _research(102, 0.706, 0.439, 4.01, [[0.651, 43], [0.824, 34], [0.64, 25]], [0.678, 59], [0.552, 0.514])},
    "K2": {"side": "buy", "label": "기술주 거래량 폭발", "preds": ["vol_hi", "tech", "hivol"], "out": "o_nx",
           "horizon": "다음 실적 반응일 전날 종가까지",
           "research": _research(115, 0.696, 0.425, 11.52, [[0.774, 31], [0.694, 49], [0.629, 35]], [0.672, 64], [0.385, 0.473])},
    "K3": {"side": "buy", "label": "투매형 갭 하락", "preds": ["sig_down", "gapdn_cont", "below_ma"], "out": "o_nx",
           "horizon": "다음 실적 반응일 전날 종가까지",
           "research": _research(83, 0.627, 0.425, 4.77, [[0.576, 33], [0.733, 30], [0.55, 20]], [0.56, 50], [0.474, 0.473])},
    "K4": {"side": "buy", "label": "대형 고변동주", "preds": ["large", "hivol"], "out": "o_h20",
           "horizon": "반응일 +20거래일 종가까지(약 19거래일 보유)",
           "research": _research(119, 0.588, 0.405, 3.07, [[0.625, 24], [0.58, 50], [0.578, 45]], [0.588, 119], [0.463, 0.514])},
    "K5": {"side": "buy", "label": "오르던 기술주의 큰 EPS 비트", "preds": ["pre60_up", "eps_big", "tech"], "out": "o5_h20",
           "horizon": "반응일 +25거래일 종가까지(약 19거래일 보유)",
           "research": _research(90, 0.644, 0.439, 4.55, [[0.667, 18], [0.70, 40], [0.562, 32]], [0.69, 42], [0.452, 0.514])},
    "X1": {"side": "avoid", "label": "산업재 실적 급등 추격", "preds": ["sig_up", "gap_go", "indus"], "out": "o5_h20",
           "horizon": "반응일 +25거래일 종가까지",
           "research": _research(80, 0.20, 0.439, -6.68, [[0.231, 26], [0.25, 24], [0.133, 30]], [0.234, 47], [0.561, 0.514])},
    "X2": {"side": "avoid", "label": "중형주 추격", "preds": ["r0_up", "midcap", "d5_up"], "out": "o5_h20",
           "horizon": "반응일 +25거래일 종가까지",
           "research": _research(114, 0.263, 0.439, -5.39, [[0.231, 52], [0.35, 20], [0.262, 42]], [0.253, 75], [0.556, 0.514])},
    "X3": {"side": "avoid", "label": "금융주 실적 뒤 추가 상승", "preds": ["fin", "d5_up"], "out": "o5_nx",
           "horizon": "다음 실적 반응일 전날 종가까지",
           "research": _research(139, 0.201, 0.439, -8.06, [[0.22, 59], [0.409, 22], [0.103, 58]], [0.161, 62], [0.451, 0.474])},
}
# 사용자 결정(2026-10-29): 실적 후 매매는 핵심 규칙 K1·K2로 하고, 피할 규칙은 X2를 쓴다. 나머지는 보조로만 본다.
CORE_BUY = ("K1", "K2")
CORE_AVOID = ("X2",)
for _code, _rule in RULES.items():
    _rule["core"] = _code in CORE_BUY + CORE_AVOID
    _rule["conditions"] = [PRED[p][0] for p in _rule["preds"]]
    _rule["entry"] = ENTRY_TEXT[OUTCOMES[_rule["out"]][0]]
    _rule["needs_eps"] = any(p in EPS_PREDS for p in _rule["preds"])


class ErulesError(ValueError):
    """화면에 그대로 보여 줄 실적 뒤 흐름 오류."""


# ----------------------------------------------------------------------------- 판정(순수 함수)
def features(c: np.ndarray, o: np.ndarray, h: np.ndarray, lo: np.ndarray, v: np.ndarray, spy: np.ndarray, k: int,
             cap_now: float | None, px_last: float | None) -> dict | None:
    """반응일 위치 k(그날 종가까지)에서 연구(events_wide.py)와 같은 정의로 계산한다. k+5 종가가 있으면 d5_x도."""
    if k < 61 or k >= len(c) or not (c[k - 1] > 0) or np.isnan(c[k]):
        return None
    p0 = c[k - 1]
    r = np.diff(np.log(c), prepend=np.nan)
    vol60 = float(np.nanstd(r[k - 60:k]) * 100)
    avgv = float(np.nanmean(v[k - 51:k - 1]))
    hi252 = float(np.nanmax(h[max(0, k - 253):k - 1]))
    ma50, ma200 = float(np.nanmean(c[k - 51:k - 1])), float(np.nanmean(c[max(0, k - 201):k - 1]))
    r0 = float((c[k] / p0 - 1) * 100)
    f = {
        "r0": r0,
        "r0_x": r0 - float((spy[k] / spy[k - 1] - 1) * 100),
        "r0_sig": r0 / vol60 if vol60 > 0 else None,
        "gap": float((o[k] / p0 - 1) * 100) if o[k] > 0 else None,
        "o2c": float((c[k] / o[k] - 1) * 100) if o[k] > 0 else None,
        "vol60": vol60,
        "vol_ratio": float(v[k] / avgv) if avgv > 0 else None,
        "dist_hi": float((p0 / hi252 - 1) * 100) if hi252 > 0 else None,
        "clv": float((c[k] - lo[k]) / (h[k] - lo[k])) if h[k] > lo[k] else None,
        "below_ma": int(p0 < ma50 and p0 < ma200),
        "pre60_x": float((c[k - 1] / c[k - 61] - 1) * 100 - (spy[k - 1] / spy[k - 61] - 1) * 100) if c[k - 61] > 0 else None,
        "mcap_b": float(cap_now * p0 / px_last / 1e9) if cap_now and px_last and px_last > 0 else None,
    }
    if k + 5 < len(c) and not np.isnan(c[k + 5]):
        f["d5_x"] = float((c[k + 5] / c[k] - 1) * 100 - (spy[k + 5] / spy[k] - 1) * 100)
    return {key: (round(val, 4) if isinstance(val, float) else val) for key, val in f.items()}


def judge(code: str, f: dict, sector: str) -> str:
    """'hit'(모두 맞음) / 'wait'(틀린 조건은 없고 아직 모르는 값이 있음: 5일 뒤 값·EPS) / 'no'."""
    vals = [PRED[p][1](f, sector) for p in RULES[code]["preds"]]
    if any(v is False for v in vals):
        return "no"
    return "wait" if any(v is None for v in vals) else "hit"


def eps_needed(f: dict, sector: str) -> bool:
    """EPS 값만 있으면 맞을 수 있는 규칙이 있나(나머지 조건에 틀린 것이 없음) — 그런 종목만 yfinance를 부른다."""
    for rule in RULES.values():
        if not rule["needs_eps"]:
            continue
        others = [PRED[p][1](f, sector) for p in rule["preds"] if p not in EPS_PREDS]
        if not any(v is False for v in others):
            return True
    return False


def excess_open(c: np.ndarray, o: np.ndarray, sc: np.ndarray, so: np.ndarray, a: int, b: int) -> float | None:
    """a일 시가에 사서 b일 종가까지, 종목 수익 − SPY(같은 시가·종가) 수익(%)."""
    if a < 0 or b < a or b >= len(c) or np.isnan(o[a]) or np.isnan(c[b]) or not (o[a] > 0) or not (so[a] > 0):
        return None
    return round(float((c[b] / o[a] - 1) * 100 - (sc[b] / so[a] - 1) * 100), 3)


def outcome(c: np.ndarray, o: np.ndarray, sc: np.ndarray, so: np.ndarray, start: int, end: int, last: int) -> dict:
    """start일 시가 진입 → end일 종가. 진입 전이면 entered=False, 끝 전이면 마지막 확정 종가까지의 값(닫히지 않음)."""
    if start > last:
        return {"entered": False, "closed": False, "x": None, "days": 0}
    if end < start:  # 다음 실적이 진입보다 먼저 와 버린 드문 경우 — 성적에서 뺀다
        return {"entered": False, "closed": True, "x": None, "days": 0}
    stop = min(end, last)
    return {"entered": True, "closed": end <= last, "x": excess_open(c, o, sc, so, start, stop), "days": int(stop - start)}


def rule_stats(events: list[dict], code: str) -> dict:
    """닫힌 실전 사례의 SPY 대비 승률, 같은 결과 칸의 전체 사건 기준 승률, 경보 상태."""
    rule = RULES[code]
    key = rule["out"]

    def closed(e: dict) -> bool:
        o = (e.get("out") or {}).get(key) or {}
        return bool(o.get("closed")) and o.get("x") is not None
    done = [e["out"][key]["x"] for e in events if code in e.get("hits", []) and closed(e)]
    base = [e["out"][key]["x"] for e in events if closed(e)]
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
    parts = {k: [] for k in ("Open", "Close", "High", "Low", "Volume")}
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
            state["events"][key] = {**ev, **{k: old[k] for k in ("features", "hits", "waits", "out", "status", "entry_dates") if k in old}}
            sessions[key] = ev["reaction"]
    # 달력에서 사라진(날짜가 바뀐) 사건은 지운다 — 같은 종목의 새 날짜 사건이 대신 생긴다
    for key in [k for k in state["events"] if k not in sessions]:
        del state["events"][key]
    state["unknown_timing"] = sorted(set(unknown))


def refresh_eps(state: dict, needed: list[tuple[str, str, str]], warnings: list[str]) -> None:
    """EPS 서프라이즈(연구와 같은 yfinance get_earnings_dates 'Surprise(%)'). EPS로 판정이 갈리는 종목만 부른다.
    나스닥 달력의 서프라이즈는 2026-10 발표 223건에서 yfinance와 구간이 19% 달라 쓰지 않는다. 못 받으면 다음 날 다시."""
    import yfinance as yf
    cache = state.setdefault("eps", {})
    today = _ny_today()
    for key, sym, day in needed:
        hit = cache.get(key)
        if hit and (hit.get("value") is not None or hit.get("fetched", "") >= today):
            continue
        value = None
        try:
            f = yf.Ticker(sym).get_earnings_dates(limit=8)
            if f is not None and not f.empty:
                for ts, row in f.iterrows():
                    gap = abs((ts.tz_convert(config.NY).date() - date.fromisoformat(day)).days)
                    v = row.get("Surprise(%)")
                    if gap <= SEC_WINDOW_DAYS and v is not None and not pd.isna(v):
                        value = float(v)
                        break
        except Exception as exc:  # Yahoo가 막으면 그 규칙은 'EPS 대기'로 남고 다음 갱신에 다시 묻는다
            warnings.append(f"EPS {sym} 실패: {type(exc).__name__}")
        cache[key] = {"fetched": today, "value": value}


def score_events(state: dict, prices: dict[str, pd.DataFrame], universe: dict, cal=None) -> list[tuple[str, str, str]]:
    """특징·규칙·결과를 다시 계산한다(가격이 있는 사건만). 반환: EPS가 있어야 판정이 끝나는 (키, 종목, 달력일)."""
    C, O, H, L, V = (prices[k] for k in ("Close", "Open", "High", "Low", "Volume"))
    days = [d.strftime("%Y-%m-%d") for d in C.index]
    pos = {d: i for i, d in enumerate(days)}
    last = len(days) - 1
    sc, so = C[BENCH].to_numpy(dtype=float), O[BENCH].to_numpy(dtype=float)
    eps_cache = state.get("eps", {})
    need_eps: list[tuple[str, str, str]] = []
    by_ticker: dict[str, list[str]] = {}
    for key, ev in state["events"].items():
        by_ticker.setdefault(ev["ticker"], []).append(key)
    for t, keys in by_ticker.items():
        keys.sort(key=lambda k: state["events"][k]["reaction"])
        if t not in C or C[t].dropna().empty:
            for k in keys:
                state["events"][k]["status"] = "no_price"
            continue
        c, o, h, lo, v = (X[t].to_numpy(dtype=float) for X in (C, O, H, L, V))
        valid = np.where(~np.isnan(c))[0]
        px_last = c[valid[-1]]
        cap_now = (universe.get(t) or {}).get("cap")
        for i, key in enumerate(keys):
            ev = state["events"][key]
            if cal is not None:
                win = cal.sessions_window(ev["reaction"], 7)
                ev["entry_dates"] = {"1": win[1].strftime("%Y-%m-%d"), "6": win[6].strftime("%Y-%m-%d")}
            k = pos.get(ev["reaction"])
            if k is None:
                ev["status"] = "waiting"  # 반응일 종가가 아직 확정되지 않음
                continue
            f = features(c, o, h, lo, v, sc, k, cap_now, px_last)
            if f is None:
                ev["status"] = "no_history"
                continue
            if f.get("mcap_b") is not None and f["mcap_b"] * 1e9 < MIN_CAP_EVENT:
                ev["status"] = "small"
                continue
            sector = ev.get("sector", "")
            eps = (eps_cache.get(key) or {}).get("value")
            if eps is not None:
                f["eps"] = eps
            elif eps_needed(f, sector):
                need_eps.append((key, t, ev["cal_day"]))
            ev["features"] = f
            verdict = {code: judge(code, f, sector) for code in RULES}
            ev["hits"] = [c_ for c_, j in verdict.items() if j == "hit"]
            ev["waits"] = [c_ for c_, j in verdict.items() if j == "wait"]
            nk = pos.get(state["events"][keys[i + 1]]["reaction"]) if i + 1 < len(keys) else None
            nx_end = (nk - 1) if nk is not None else k + NEXT_CAP_SESSIONS
            ev["out"] = {}
            for name, (a, b) in OUTCOMES.items():
                ev["out"][name] = outcome(c, o, sc, so, k + a, nx_end if b is None else k + b, last)
                if b is None:
                    ev["out"][name]["capped"] = nk is None
            ev["status"] = "ok"
            ev.pop("matches", None)
            ev.pop("wait_a2", None)
    state["latest_close"] = days[-1] if days else None
    return need_eps


def refresh(state: dict | None = None, path: Path = STATE, universe: dict | None = None) -> dict:
    """달력·접수 시각·가격(·필요한 EPS)을 새로 받아 원장을 고친다. 실패한 부분은 경고로 남기고 이전 값을 유지한다."""
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
            need = score_events(state, prices, universe, cal)
            if need:
                refresh_eps(state, need, warnings)
                score_events(state, prices, universe, cal)
        except Exception as exc:  # 가격 실패는 원장을 지우지 않고 경고만 남긴다
            warnings.append(f"가격 갱신 실패 · 저장 원장 사용: {type(exc).__name__}: {exc}")
    state.update({"updated_at": now_iso(), "updated_epoch": _time.time(), "warnings": warnings})
    write_atomic(path, state)
    return state


# ----------------------------------------------------------------------------- 화면 자료
def _row(ev: dict, code: str | None = None) -> dict:
    f = ev.get("features") or {}
    key = RULES[code]["out"] if code else "o_h20"
    o = (ev.get("out") or {}).get(key) or {}
    offset = str(OUTCOMES[key][0])
    return {"ticker": ev["ticker"], "name": ev.get("name", ""), "ai": ev.get("ai", ""), "sector": ev.get("sector", ""),
            "reaction": ev["reaction"], "entry_date": (ev.get("entry_dates") or {}).get(offset), "timing": ev["timing"]["src"],
            "mcap_b": f.get("mcap_b"), "r0_x": f.get("r0_x"), "d5_x": f.get("d5_x"), "eps": f.get("eps"),
            "vol_ratio": f.get("vol_ratio"), "vol60": f.get("vol60"), "gap": f.get("gap"), "o2c": f.get("o2c"),
            "pre60_x": f.get("pre60_x"), "below_ma": f.get("below_ma"),
            "hits": ev.get("hits", []), "waits": ev.get("waits", []),
            "x": o.get("x"), "days": o.get("days"), "entered": o.get("entered", False), "closed": o.get("closed", False),
            "capped": o.get("capped", False)}


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
    rules, upcoming = [], []
    for code, rule in RULES.items():
        hits = sorted((e for e in events if code in e.get("hits", [])), key=lambda e: e["reaction"], reverse=True)
        rows = [_row(e, code) for e in hits]
        upcoming += [{**r, "code": code, "side": rule["side"], "label": rule["label"], "core": rule["core"],
                      "research_win": rule["research"]["win"]} for r in rows if not r["entered"] and not r["closed"]]
        rules.append({"code": code, **{k: v for k, v in rule.items() if k != "preds"}, "live": rule_stats(events, code),
                      "open": [r for r in rows if r["entered"] and not r["closed"]],
                      "closed": [r for r in rows if r["closed"]]})
    # 핵심 규칙: K1·K2 진입 예정(아직 진입일 시가 전), X2에 걸려 아직 기간이 끝나지 않은 종목(따라 사지 않을 목록)
    core_buy = [u for u in upcoming if u["code"] in CORE_BUY]
    x2 = next(r for r in rules if r["code"] in CORE_AVOID)
    core_avoid = [{**r, "code": x2["code"], "label": x2["label"]} for r in x2["open"]] +                  [u for u in upcoming if u["code"] in CORE_AVOID]
    recent_from = sorted({e["reaction"] for e in events})[-10:]
    recent = sorted((_row(e) for e in events if recent_from and e["reaction"] >= recent_from[0]),
                    key=lambda r: (r["reaction"], -(r["mcap_b"] or 0)), reverse=True)
    status: dict[str, int] = {}
    for e in state["events"].values():
        status[e.get("status", "pending")] = status.get(e.get("status", "pending"), 0) + 1
    return {"live_start": state.get("live_start", LIVE_START), "updated_at": state.get("updated_at"),
            "latest_close": state.get("latest_close"), "events_n": len(events), "status_counts": status,
            "calendar_days": len(state.get("calendar", {})), "unknown_timing": state.get("unknown_timing", []),
            "rules": rules, "upcoming": sorted(upcoming, key=lambda r: (r["entry_date"] or "", r["side"] != "buy")),
            "core": {"buy": list(CORE_BUY), "avoid": list(CORE_AVOID),
                     "entries": sorted(core_buy, key=lambda r: (r["entry_date"] or "", r["code"])),
                     "avoid_now": sorted(core_avoid, key=lambda r: r["reaction"], reverse=True)},
            "recent": recent, "warning": warning or ("; ".join(state.get("warnings") or []) or None),
            "alarm_min_n": ALARM_MIN_N,
            "method": "나스닥 실적 달력으로 발표 종목을 찾고, SEC 8-K 2.02 접수 시각으로 장전·장후를 가려 반응일을 정한다(없으면 나스닥·yfinance 시각). "
                      "특징·조건은 연구(rules_report.md)와 같은 정의이고, EPS 서프라이즈는 연구와 같은 yfinance 값이다. "
                      "성적은 진입일 정규장 시가에 사서 끝날 종가까지의 종목 수익 − 같은 시가·종가의 SPY 수익이며 수정 가격(배당·분할 반영)으로 잰다. "
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
            print(f"실적 뒤 흐름 · 실전 시작 {data['live_start']} · 채점한 발표 {data['events_n']}건 · 최근 종가 {data['latest_close']}")
            for r in data["rules"]:
                lv = r["live"]
                win = f"{lv['win'] * 100:.0f}%" if lv["win"] is not None else "—"
                base = f"{lv['base_win'] * 100:.0f}%" if lv["base_win"] is not None else "—"
                print(f"  {r['code']} {r['label']}: 닫힘 {lv['n']}건 승률 {win}(기준 {base}) · 보유 중 {len(r['open'])}건 · {lv['state']}")
            for u in data["upcoming"]:
                print(f"  진입 예정 {u['entry_date']} 시가 · {u['code']} {u['ticker']}")
        return 0
    except (ErulesError, OSError, json.JSONDecodeError) as exc:
        print(f"실적 뒤 흐름 오류: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
