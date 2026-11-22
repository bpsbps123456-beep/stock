"""§3.2 FINVIZ 30·40·175 모의 관찰의 자동 채점(2026-11-22 사용자: "왜 바로 넣지는 않는 거지? 뭐 자동 채점해요").

두 안의 차이는 SOXX 100일선 이탈 때 쓰는 월초 바스켓뿐이다(주간 전략·시장 전환은 같다). 그래서 매달 월초에 고정된
두 바스켓을 같은 방식으로 산다고 보고 그달 성적을 비교한다.

- 바스켓: 현행 §3.1 `data/normalized/desk_finviz/YYYY-MM.json`, 30·40·175 `.../sma30_40_175/YYYY-MM.json`.
- 성적: 월 첫 거래일 정규장 시가에 원래 7자리를 각 1/7(빈자리·같은 발행사 중복은 달러 현금 0%)로 사서 주식 수를 그대로
  두고, 그달 마지막 정규장 종가(진행 중인 달은 마지막 완료 세션 종가)까지의 수익. 배당·분할 수정 일봉(yfinance).
- 전방 기록: 등록(2026-11-20) 뒤에 고정되는 2026-12 바스켓부터 센다. 11월 바스켓은 연구 표본에서 복원했으므로
  참고로만 보이고 합계·판정에 넣지 않는다.
- 대피 표시: 그달 §3.1 개장 전 관찰 기록(finviz_observations/) 중 mode가 finviz였던 세션 수. 바스켓이 실제 주문에
  쓰인 달인지 알려 줄 뿐 채점은 대피 여부와 무관하게 한다(대피 달만 세면 1년에 몇 번뿐이다).
- 다시 묻는 때: 닫힌 전방 달이 REVIEW_AFTER개 쌓이면 화면에 '교체 재검토'를 띄운다. 기준값 3개월은 이 결정 때
  정한 기본값이다(대피가 1년에 5번 남짓이라 그보다 길면 판단이 너무 늦다).
목표 비중·주문 수량을 만들지 않고 주문 초안(§3.1)을 바꾸지 않는다.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np

from . import config
from .common import clean, read_cache, write_atomic
from .config import market_calendar

FORWARD_FROM = "2026-12"      # 등록(2026-11-20) 뒤 처음 고정되는 월초 바스켓
REVIEW_AFTER = 3              # 닫힌 전방 달이 이만큼 쌓이면 교체를 다시 묻는다
SLOTS = 7
MAX_AGE_SEC = 3600
CACHE = config.CACHE / "finviz_score.json"
VARIANTS = ("current", "sma30_40_175")
LABEL = {"current": "현행 §3.1", "sma30_40_175": "30·40·175"}


def basket_dirs() -> dict[str, Path]:
    from . import finviz_basket, finviz_trend
    return {"current": finviz_basket.DIRECTORY, "sma30_40_175": finviz_trend.DIRECTORY}


def month_sessions(month: str) -> list[str]:
    first = date.fromisoformat(month + "-01")
    nxt = date(first.year + (first.month == 12), first.month % 12 + 1, 1)
    last = date.fromordinal(nxt.toordinal() - 1)
    return [d.date().isoformat() for d in market_calendar().sessions_in_range(first.isoformat(), last.isoformat())]


def basket_return(symbols: list[str], opens: dict, closes: dict) -> float | None:
    """원래 7자리 각 1/7, 빈자리는 현금. 한 종목이라도 시가·종가가 없으면 None(가격 대기)."""
    if len(symbols) > SLOTS:
        raise ValueError("바스켓은 7자리를 넘을 수 없습니다")
    ratios = []
    for tk in symbols:
        o, c = opens.get(tk), closes.get(tk)
        if o is None or c is None or not (np.isfinite(o) and np.isfinite(c) and o > 0 and c > 0):
            return None
        ratios.append(c / o)
    return (SLOTS - len(symbols) + sum(ratios)) / SLOTS - 1


def load_months() -> list[dict]:
    """두 안 모두 바스켓이 고정된 달(오래된 순)."""
    dirs = basket_dirs()
    months = sorted({p.stem for p in dirs["current"].glob("????-??.json")} &
                    {p.stem for p in dirs["sma30_40_175"].glob("????-??.json")})
    out = []
    for m in months:
        rec = {v: json.loads((dirs[v] / f"{m}.json").read_text(encoding="utf-8")) for v in VARIANTS}
        out.append(dict(month=m, forward=m >= FORWARD_FROM,
                        effective=rec["current"]["effective_date"],
                        baskets={v: list(rec[v]["symbols"]) for v in VARIANTS}))
    return out


def evacuation_days(month: str, observations: Path | None = None) -> tuple[int, int]:
    """그달 §3.1 개장 전 관찰 기록 중 (finviz 세션 수, 기록된 세션 수)."""
    from . import finviz_switch
    folder = observations or finviz_switch.observation_dir()
    hit = seen = 0
    for p in folder.glob(f"{month}-??.json"):
        seen += 1
        hit += json.loads(p.read_text(encoding="utf-8")).get("mode") == "finviz"
    return hit, seen


def score(months: list[dict], prices: dict, completed: str, observations: Path | None = None) -> dict:
    """prices: {"Open": DataFrame, "Close": DataFrame}(날짜 문자열 인덱스). completed = 마지막 완료 세션."""
    rows = []
    for m in months:
        sessions = [d for d in month_sessions(m["month"]) if d <= completed]
        row = dict(month=m["month"], forward=m["forward"], baskets=m["baskets"], effective=m["effective"])
        row["evacuation_days"], row["recorded_days"] = evacuation_days(m["month"], observations)
        if not sessions or m["effective"] > completed:
            row.update(status="시작 전", returns=None)
            rows.append(row)
            continue
        end = sessions[-1]
        closed = end == month_sessions(m["month"])[-1]
        rets = {}
        for v in VARIANTS:
            syms = m["baskets"][v]
            opens = {tk: _at(prices, "Open", m["effective"], tk) for tk in syms}
            closes = {tk: _at(prices, "Close", end, tk) for tk in syms}
            rets[v] = basket_return(syms, opens, closes)
        priced = all(r is not None for r in rets.values())
        row.update(end=end, returns=rets if priced else None,
                   status=("닫힘" if closed else "진행 중") if priced else "가격 대기")
        rows.append(row)
    done = [r for r in rows if r["forward"] and r["status"] == "닫힘"]
    total = {v: float(np.prod([1 + r["returns"][v] for r in done]) - 1) if done else None for v in VARIANTS}
    wins = sum(r["returns"]["sma30_40_175"] > r["returns"]["current"] for r in done)
    evac = sum(r["evacuation_days"] > 0 for r in done)
    if not done:
        verdict = f"전방 채점은 {FORWARD_FROM} 월초 바스켓부터(닫힌 달 0/{REVIEW_AFTER})"
    elif len(done) < REVIEW_AFTER:
        verdict = f"쌓는 중 {len(done)}/{REVIEW_AFTER}"
    else:
        verdict = "교체 재검토 시점 — 주문 초안을 30·40·175로 바꿀지 사용자에게 다시 묻는다"
    return dict(months=rows, closed=len(done), wins=wins, evacuation_months=evac, total=total,
                review_after=REVIEW_AFTER, forward_from=FORWARD_FROM, verdict=verdict, completed=completed)


def _at(prices: dict, field: str, day: str, tk: str) -> float | None:
    frame = prices.get(field)
    if frame is None or tk not in frame.columns or day not in frame.index:
        return None
    v = frame.at[day, tk]
    return float(v) if np.isfinite(v) else None


def fetch_prices(symbols: list[str], start: str, now: datetime) -> dict:
    from .erules import fetch_prices as daily
    raw = daily(symbols, start, now)
    return {k: raw[k].set_axis([d.strftime("%Y-%m-%d") for d in raw[k].index]) for k in ("Open", "Close")}


def cached(path: Path = CACHE) -> dict | None:
    """/api/state가 읽는 값. 네트워크를 쓰지 않는다(갱신은 서버의 5분 관찰 루프가 payload로 한다)."""
    return read_cache(path)


def payload(force: bool = False, path: Path = CACHE) -> dict:
    """화면용 채점. 한 시간 캐시. 실패해도 데스크 화면은 그대로(오류 문구만)."""
    from .state import last_completed_session
    now = datetime.now(config.NY)
    completed = last_completed_session(now)
    cached = read_cache(path)
    if not force and cached and cached.get("completed") == completed and time.time() - cached.get("epoch", 0) < MAX_AGE_SEC:
        return cached
    try:
        months = load_months()
        symbols = sorted({tk for m in months for v in VARIANTS for tk in m["baskets"][v]})
        live = [m for m in months if m["effective"] <= completed]
        prices = fetch_prices(symbols, min(m["effective"] for m in live), now) if live and symbols else {}
        out = score(months, prices, completed)
    except Exception as exc:  # 채점 실패가 주문 화면을 막지 않는다
        return dict(error=f"30·40·175 자동 채점을 계산하지 못했습니다: {exc}", completed=completed)
    out["epoch"] = time.time()
    write_atomic(path, clean(out), indent=1)
    return out
