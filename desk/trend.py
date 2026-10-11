"""추세 등급 탭: 두 펀드(TIME·KoAct)의 모든 보유 종목을 지금의 가격·이평선 추세로 매일 S~D로 나눈다.

정본은 docs/desk_system.md §11이다. 거인의 어깨 회복 등급(§9, 급락에서 되찾은 정도)과 달리 지금 가격이
이평선들 위에서 어떤 모양인지(현재 추세 상태)만 본다. 거래량·실적·뉴스는 쓰지 않는다(2026-10-12 사용자
요청: 순수히 가격 라인 중심). 관찰용 화면이라 매수 추천·목표 비중·주문 수량을 만들지 않는다(§1).
가격은 yfinance 일봉(분할 반영 종가)이다. TradingView MCP 값은 넣지 않는다(AGENTS.md).

점수와 등급 사다리의 근거는 data/derived/analysis/trend_grade/report.md다.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import watchlist
from gbrc.etf.market import company_key, is_cash_holding, listing_market, yahoo_ticker
from gbrc.etf.storage import NORMALIZED_HOLDINGS_DIR, load_normalized_snapshot

from . import config, strategies
from .state import clean

# 계산 방식이 바뀌면 올린다. 캐시의 방식이 다르면 60분 안이어도 다시 계산한다(설명란과 데이터가 어긋나지 않게).
SPEC = "2026-10-12 미국 기준 순위·당일 점수·주간 등급·실적 일정"
CACHE_PATH = config.CACHE / "trend.json"
CHART_PATH = config.CACHE / "trend_chart.json"
EARN_PATH = config.CACHE / "trend_earnings.json"
EARN_MAX_AGE_H = 12       # 실적 일정은 하루 한두 번만 바뀐다
NY = ZoneInfo("America/New_York")
MAX_AGE_MIN = 60
FUNDS = (("456600", "TIME"), ("2ETFQ1", "KoAct"))
MARKETS = {"US": "미국", "KR": "한국", "JP": "일본", "HK": "홍콩", "CN": "중국"}
BENCH = "QQQ"
HISTORY_PERIOD = "2y"     # 200일선·52주 고점에 충분한 일봉
HISTORY_DAYS = 30         # 화면의 등급 이력 칸 수(거래일)
MAX_MISSING = 0.2         # 시세를 못 받은 보유 종목이 이 비율을 넘으면 계산을 실패로 둔다
CHART_DAYS = 160          # 종목 차트 길이(거래일)
MAS = (20, 50, 200)       # 화면에 보이는 이평선
_LOCK = threading.Lock()

# ── 점수와 등급 (근거 data/derived/analysis/trend_grade/report.md, 연구 code/final.py FINAL과 같은 식) ──
# 재료마다 그날 등급 대상 안에서 백분위를 내고 같은 가중으로 평균 → 최근 SMOOTH일 평균 → 다시 백분위.
PARTS = {"L": 1.0, "M": 1.0, "H": 1.0, "Q": 1.0}
PART_NOTES = {
    "L": "200일선 위 거리(종가÷200일 이평−1)",
    "M": "50일선 위 거리(종가÷50일 이평−1)",
    "H": "52주 고점 근접(종가÷252일 최고 고가−1)",
    "Q": "126일 가격 추세선의 기울기×매끄러움(Clenow: ln 종가 회귀 기울기의 연율 상승률×R²)",
}
SMOOTH = 10               # 점수의 최근 10일 평균. 등급이 하루하루 덜 흔들린다(연구: 하루 바뀜 16%→7%)
SMA_MIN = 120             # 긴 이평선·52주 고점은 봉 120개부터 가진 봉으로 계산(신규 상장 주도주가 빠지지 않게)
MIN_BARS = 130            # 등급 대상: 봉 130개 이상(연구 채점과 같은 기준)
# 점수 백분위 하한(초과). 상위 10% S / 20% A / 40% B / 20% C / 하위 10% D.
LADDER = (("S", 0.90), ("A", 0.70), ("B", 0.30), ("C", 0.10))
LABELS = {"S": "미국 보유 상위 10% · 가장 강한 추세", "A": "다음 20% · 강한 추세", "B": "가운데 40%",
          "C": "다음 20% · 약한 추세", "D": "하위 10% 또는 하락 배열"}
# 관문: S·A는 종가>50일선>200일선(상승 배열)일 때만, 아니면 B로 내린다. 종가<200일선이고 50일선<200일선
# (하락 배열)이면 순위와 무관하게 D. 순위가 높아도 이평선 위에 있지 않으면 '강한 추세'라 부르지 않는다.


# 2026 검증(연구 code/final.py FINAL, 두 펀드 미국 보유, t일 종가 등급 → 다음 날 시가부터). 화면이 그대로 읽는다.
# 숫자를 바꾸려면 연구를 다시 돌리고 이 값과 docs/desk_system.md §11을 함께 고친다.
CHECK = {
    "subtitle": "2026년 1~5월로 정하고 6~10월로 확인 · 두 펀드 미국 보유 종목 · 2025년은 보조",
    "verdict": ("관찰용 등급입니다. 2026년 1~5월에는 S가 D보다 이후 20일 동안 평균 15.9%p 더 올랐지만, 6~7월 급락·반등 때는 "
                "순서가 거꾸로 섰습니다. 이평선·추세 점수 168개와 장세 지표 24개를 쟀지만 6~10월에도 통한 점수는 없었고, "
                "언제 통할지 미리 알려 주는 지표도 없었습니다. 점수들은 사실상 '지금 추세의 세기' 한 가지를 여러 방식으로 잰 것입니다."),
    "tables": [
        {"title": "등급별 이후 20거래일 초과수익(그날 보유 종목 평균 대비, %p)", "columns": ["S", "A", "B", "C", "D"],
         "rows": [["1~5월(고른 구간)", [7.1, 4.2, 1.7, -6.3, -8.7]],
                  ["6~10월(확인 구간)", [-3.3, -1.5, -1.1, -1.1, 5.9]],
                  ["2025년(보조)", [-1.5, 0.4, -0.3, 2.2, -0.4]]]},
        {"title": "S등급만 매주 같은 비중으로 들었다면(%, 주 첫 거래일 시가 교체, 다 채운 주만, 낙폭은 매일 종가)",
         "columns": ["S 보유", "S 최대 낙폭", "보유 종목 전체", "QQQ"],
         "rows": [["1/5~5/26", [59.5, -21.3, 50.1, 17.2]], ["6/8~10/5", [-2.5, -36.0, 8.5, 4.4]]]},
        {"title": "7월 급락(6/22 종가 등급 → 7/29 종가, %)", "columns": ["S", "A", "B", "C", "D", "전체", "QQQ"],
         "rows": [["6/22→7/29", [-42.3, -31.9, -16.4, -14.3, 9.1, -16.8, -10.3]]]},
    ],
    "points": [
        "1~5월 S−D 차이 +15.9%p는 날짜 묶음을 다시 뽑아도 +8~+23%p로 0보다 컸습니다. 다만 S 혼자의 +7.1%p는 SNDK 한 종목이 "
        "57%를 차지했고, 기여 상위 3종목을 빼면 +2.4%p로 A보다 낮았습니다.",
        "6~10월의 역전은 6·7월 신호에 몰려 있습니다(S −6.8%p, D +6.4~+9.4%p). 8월~9월 초 신호만 보면 S +2.0%p였습니다. "
        "9월 초에는 다시 맞았지만 겹치지 않는 20일 창 1개뿐이라 '다시 통한다'고 할 수 없습니다.",
        "7월 표의 −42%는 시작·끝 날짜를 바꿔 본 483쌍 중 가장 나쁜 경우입니다(중앙값 S −18% 대 전체 −5%). S 종목은 변동성이 커서"
        "(연율 91% 대 65%) 빠질 때 더 빠졌고, 변동성이 비슷한 다른 종목도 −35%였습니다.",
        "S등급의 이후 20일 최대 낙폭은 전체보다 1~5월 2.0%p, 6~10월 3.7%p, 2025년 3.8%p 깊었고, 21개월 중 17개월에서 더 깊었습니다. "
        "A등급은 전체와 비슷했습니다(−0.6~+1.4%p). S를 사면 더 큰 흔들림을 견뎌야 합니다.",
        "언제 통할지 미리 알려 주는 가격 지표(시장 폭, QQQ 20·50일선, SOXX 20일선, 최근 성적 등 24개)는 찾지 못했습니다. "
        "최근 성적이 좋았던 직후(6월)에 오히려 크게 뒤집혔습니다.",
        "S등급 주간 보유는 5~7종목이라 두 종목만 바뀌어도 누적 수익이 13%p쯤 움직였습니다. 위 등급별 초과수익이 더 믿을 만한 근거입니다.",
        "2025년(효과 0)은 2026년 보유 종목으로 잰 보조 확인이라 생존 편향이 있습니다.",
        "점수를 10일 평균해 하루에 등급이 바뀌는 종목은 약 7%입니다. S등급이 이어지는 기간은 중앙값 6거래일, 평균 13거래일이었습니다.",
    ],
    # S 종목 화면에 붙이는 위험 문구(위 네 번째 점의 근거). A는 전체와 비슷해서 붙이지 않는다.
    "risk": "S 종목은 변동성이 커서 검증에서 이후 20일 최대 낙폭이 보유 종목 평균보다 2~4%p 깊었습니다(1~5월 2.0, 6~10월 3.7, 2025년 3.8%p).",
    "report": "data/derived/analysis/trend_grade/report.md",
}


# ----------------------------------------------------------------------------- holdings
def holdings(root: Path = NORMALIZED_HOLDINGS_DIR) -> tuple[list[dict], dict[str, str]]:
    """두 펀드 최신 검증 스냅샷의 주식·ETF 보유를 회사 하나에 한 줄로 합친다(선물·현금 제외).

    GOOGL·GOOG처럼 두 펀드가 다른 클래스주를 들면 한 줄로 묶고 먼저 나온 상장의 시세를 쓴다.
    """
    rows: dict[str, dict] = {}
    dates: dict[str, str] = {}
    for fund_id, label in FUNDS:
        paths = sorted(Path(root).glob(f"*_{fund_id}.json"))
        if not paths:
            continue
        as_of = paths[-1].name[:10]
        dates[label] = as_of
        for h in load_normalized_snapshot(paths[-1])["holdings"]:
            code = h.get("code")
            market = listing_market(code)
            if is_cash_holding(h) or market not in MARKETS:
                continue
            key = company_key(code)
            sym = yahoo_ticker(code, as_of)
            if not sym:
                continue
            row = rows.setdefault(key, {"key": key, "symbol": sym, "name": str(h.get("name") or sym),
                                        "market": MARKETS[market], "funds": {}})
            row["funds"][label] = round(row["funds"].get(label, 0.0) + float(h["weight"]), 2)
    for row in rows.values():
        row["sector"] = strategies.sector_of(row["key"], row["symbol"])
        row["sub"] = strategies.subsector_of(row["key"], row["symbol"])
        row["ko"] = korean_name(row["key"])
        alias = watchlist.AI_UNIVERSE_ALIASES.get(row["symbol"])
        if alias and "." in alias:  # 다른 시장의 같은 회사(SKHY ADR → 000660.KS): 상장이 짧을 때 가격 이력으로 쓴다
            row["proxy"] = alias
    return sorted(rows.values(), key=lambda r: -max(r["funds"].values())), dates


def korean_name(key: str) -> str | None:
    """AI 산업 지도의 한글 이름(짧은 이름 우선). 지도에 없으면 None."""
    from . import tour

    sym = tour.symbol(key)
    try:
        company = tour.load_map()["companies"].get(sym) or {}
    except (OSError, ValueError):
        return None
    return company.get("short") or company.get("ko")


# ----------------------------------------------------------------------------- prices
# 시장마다 정규장 마감(현지 시각). 마감 15분 뒤까지 그날 봉은 아직 움직이는 중이라 뺀다(장중 등급 깜빡임 방지).
SESSION = {"미국": ("America/New_York", "16:15"), "한국": ("Asia/Seoul", "15:45"), "일본": ("Asia/Tokyo", "15:45"),
           "홍콩": ("Asia/Hong_Kong", "16:25"), "중국": ("Asia/Shanghai", "15:15")}


def settled(bars: pd.DataFrame, market: str, now: datetime | None = None) -> pd.DataFrame:
    """그 시장의 오늘 봉이 아직 마감 전이면 뺀다. 지난 날짜 봉은 그대로 둔다."""
    from zoneinfo import ZoneInfo

    if bars is None or bars.empty:
        return bars
    tz, close = SESSION.get(market, SESSION["미국"])
    local = (now or datetime.now(config.KST)).astimezone(ZoneInfo(tz))
    last = bars.index[-1]
    if last.date() == local.date() and local.strftime("%H:%M") < close:
        return bars.iloc[:-1]
    return bars


def fetch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """2년 일봉(분할 반영, 배당 미반영 종가). 받지 못한 종목은 빠진다."""
    import yfinance as yf

    frame = yf.download(sorted(set(symbols)), period=HISTORY_PERIOD, auto_adjust=False, progress=False,
                        threads=True, group_by="column")
    out = {}
    for sym in sorted(set(symbols)):
        try:
            bars = pd.DataFrame({k: frame[k][sym] for k in ("Open", "High", "Low", "Close")})
        except KeyError:
            continue
        bars = bars[bars["Close"].notna() & (bars["High"] > 0)]
        bars.index = pd.DatetimeIndex(bars.index).tz_localize(None).normalize()
        if len(bars):
            out[sym] = bars[~bars.index.duplicated(keep="last")]
    return out


# ----------------------------------------------------------------------------- score (pure)
def sma(close: pd.Series, n: int) -> pd.Series:
    """n일 단순 이평. 상장이 짧은 종목도 봉 SMA_MIN개부터 가진 봉으로 계산한다(연구와 같은 규칙)."""
    return close.rolling(n, min_periods=min(n, SMA_MIN)).mean()


def clenow(close: pd.Series, n: int = 126) -> pd.Series:
    """최근 n일 ln(종가)를 날짜에 회귀한 기울기의 연율 상승률 × R²(Clenow). 가파르고 매끄러울수록 크다."""
    y = np.log(close.to_numpy(float))
    out = np.full(len(y), np.nan)
    if len(y) >= n:
        w = np.lib.stride_tricks.sliding_window_view(y, n)
        x = np.arange(n, dtype=float) - (n - 1) / 2.0
        sxx = float(x @ x)
        yc = w - w.mean(axis=1, keepdims=True)
        b = (yc * x).sum(axis=1) / sxx
        syy = (yc * yc).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = np.where(syy > 0, b * b * sxx / syy, np.nan)
        out[n - 1:] = (np.exp(252.0 * b) - 1.0) * r2
    return pd.Series(out, index=close.index)


def indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """종목 하나의 날짜별 점수 재료(원값)·관문·화면 값. 그날 종가까지의 자료만 쓴다."""
    c, hi = bars["Close"], bars["High"]
    m = {n: sma(c, n) for n in (20, 50, 200)}
    high52 = hi.rolling(252, min_periods=SMA_MIN).max()
    out = pd.DataFrame(index=bars.index)
    out["L"] = c / m[200] - 1.0
    out["M"] = c / m[50] - 1.0
    out["H"] = c / high52 - 1.0
    out["Q"] = clenow(c, 126)
    out["S"] = c / m[20] - 1.0  # 화면 값(20일선 대비)
    out["up"] = ((c > m[50]) & (m[50] > m[200])).where(m[200].notna())
    out["down"] = ((c < m[200]) & (m[50] < m[200])).where(m[200].notna())
    out["order"] = ((c > m[20]) & (m[20] > m[50]) & (m[50] > m[200])).where(m[200].notna())
    out["bars"] = np.arange(1, len(c) + 1)
    out["close"] = c
    for n in MAS:
        out[f"ma{n}"] = m[n]
    return out


def grade_of(pct: float | None) -> str | None:
    """그날 등급 대상 안 점수 백분위(0~1) → 관문 전 등급. 하한보다 크면 그 등급, 가장 낮은 칸은 D."""
    if pct is None or not np.isfinite(pct):
        return None
    return next((g for g, floor in LADDER if pct > floor), "D")


def gate(grade: str | None, up, down) -> str | None:
    """관문: S·A는 상승 배열(종가>50일선>200일선)일 때만, 하락 배열(종가<200일선, 50일선<200일선)은 D."""
    if grade is None:
        return None
    if down is True or down == 1:
        return "D"
    if grade in ("S", "A") and not (up is True or up == 1):
        return "B"
    return grade


def _wide(ind: dict[str, pd.DataFrame], col: str, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """종목별 자기 거래일 값을 기준 달력에 맞춘다. 미국 밖 종목은 그 날짜 또는 직전 현지 거래일 값
    (최대 5거래일까지 이어 씀)."""
    return pd.DataFrame({s: d[col].reindex(cal, method="ffill", limit=5) for s, d in ind.items() if len(d)},
                        index=cal, columns=[s for s, d in ind.items() if len(d)])


def rank_vs(x: pd.DataFrame, ref: set[str] | None = None) -> pd.DataFrame:
    """날짜마다 백분위(0~1). ref 종목끼리는 보통 백분위이고, ref 밖 종목은 ref 분포 안에서 몇 등쯤인지다.
    그래서 ref 밖 종목을 더해도 ref 종목의 순위는 바뀌지 않는다(미국 밖 종목이 미국 종목 등급을 흔들지 않게)."""
    cols = list(x.columns) if ref is None else [c for c in x.columns if c in ref]
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    out[cols] = x[cols].rank(axis=1, pct=True)
    R = x[cols].to_numpy(float)
    n = np.isfinite(R).sum(axis=1)
    for c in x.columns:
        if c in cols:
            continue
        v = x[c].to_numpy(float)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            p = ((R < v).sum(axis=1) + 0.5 * (R == v).sum(axis=1) + 0.5) / n
        p[~np.isfinite(v[:, 0]) | (n == 0)] = np.nan
        out[c] = np.minimum(p, 1.0)
    return out


def score_table(ind: dict[str, pd.DataFrame], cal: pd.DatetimeIndex, ref: set[str] | None = None):
    """날짜 × 종목 점수 백분위(0~1) 두 가지와 재료별 백분위.

    재료마다 그날 등급 대상(봉 MIN_BARS개 이상) 안에서 백분위를 내고 같은 무게로 평균한 뒤(한 재료가 비면
    남은 재료로), SMOOTH일 평균을 내어 다시 백분위로 바꾼다(연구 harness와 같은 순서) → pct.
    평균하지 않은 오늘 하루 점수의 백분위 → raw(화면의 '당일 점수' 등급; 급락 때 먼저 내려간다).
    ref가 있으면 그 종목들(미국 보유)만 순위를 정하고 나머지는 그 분포에 견준다(rank_vs).
    """
    ok = _wide(ind, "bars", cal) >= MIN_BARS
    parts, num, den = {}, None, None
    for part, weight in PARTS.items():
        x = _wide(ind, part, cal).astype(float).replace([np.inf, -np.inf], np.nan).where(ok)
        pr = rank_vs(x, ref)
        parts[part] = pr
        has = pr.notna().astype(float)
        num = pr.fillna(0.0) * weight if num is None else num + pr.fillna(0.0) * weight
        den = has * weight if den is None else den + has * weight
    raw = (num / den).where(den > 0)
    smooth = raw.rolling(SMOOTH, min_periods=1).mean().where(raw.notna()) if SMOOTH > 1 else raw
    return rank_vs(smooth, ref), rank_vs(raw, ref), parts


def grade_table(pct: pd.DataFrame, ind: dict[str, pd.DataFrame], cal: pd.DatetimeIndex) -> pd.DataFrame:
    """날짜 × 종목 등급(관문 적용). 점수가 없으면 None."""
    up, down = _wide(ind, "up", cal), _wide(ind, "down", cal)
    cols = {}
    for sym in pct.columns:
        u, dn = up[sym].to_numpy(object), down[sym].to_numpy(object)
        # object로 고정: pandas 3은 글자·None 목록을 문자열형으로 바꾸며 None을 NaN으로 만든다.
        cols[sym] = pd.Series([gate(grade_of(p), u[i], dn[i]) for i, p in enumerate(pct[sym].to_numpy(float))],
                              index=pct.index, dtype=object)
    return pd.DataFrame(cols, index=pct.index, columns=pct.columns, dtype=object)


def _pct(v) -> float | None:
    return round(float(v) * 100.0, 2) if v is not None and np.isfinite(v) else None


def _streak(letters: list) -> int:
    """오늘 등급이 끊기지 않고 이어진 거래일 수(오늘 포함)."""
    if not letters or letters[-1] is None:
        return 0
    n = 0
    for g in reversed(letters):
        if g != letters[-1]:
            break
        n += 1
    return n


def stock_row(row: dict, d: pd.DataFrame | None, letters: list, pct_now, parts_now: dict, raw: str | None = None) -> dict:
    """화면 한 줄. d는 그 종목 자신의 거래일로 계산한 indicators(). raw는 평균하지 않은 오늘 점수의 등급."""
    letters = [g if isinstance(g, str) else None for g in letters]
    raw = raw if isinstance(raw, str) else None
    out = {**row, "grade": letters[-1] if letters else None, "raw": raw,
           "pct": round(pct_now * 100, 1) if pct_now is not None and np.isfinite(pct_now) else None,
           "prev": letters[-2] if len(letters) > 1 else None, "g5": letters[-6] if len(letters) > 5 else None,
           "streak": _streak(letters), "hist": letters[-HISTORY_DAYS:],
           "parts": {k: (round(v * 100) if v is not None and np.isfinite(v) else None) for k, v in parts_now.items()}}
    if d is None or d.empty:
        return {**out, "why": "시세 없음", "bars": 0}
    last = d.iloc[-1]
    c = d["close"]
    ret = lambda n: _pct(c.iloc[-1] / c.iloc[-1 - n] - 1) if len(c) > n else None  # noqa: E731
    out.update(date=d.index[-1].date().isoformat(), bars=int(last["bars"]), close=float(last["close"]),
               chg1d=ret(1), r1m=ret(21), r3m=ret(63),
               d20=_pct(last["S"]), d50=_pct(last["M"]), d200=_pct(last["L"]), h52=_pct(last["H"]),
               order=None if pd.isna(last["order"]) else bool(last["order"]))
    if out["grade"] is None:
        out["why"] = f"상장 {int(last['bars'])}거래일"
    return out


def chart_of(d: pd.DataFrame, grades: pd.Series, start=None, weekly: pd.Series | None = None) -> dict:
    """종목 자신의 거래일로 그린 종가·이평선과, 그날(또는 직전 미국 거래일) 등급. start가 있으면 그날부터
    (미국 기준 최근 CHART_DAYS거래일), 없으면 최근 CHART_DAYS봉."""
    tail = d.loc[start:] if start is not None else d.iloc[-CHART_DAYS:]
    letters = lambda s: [v if isinstance(v, str) else None for v in s.reindex(tail.index, method="ffill")]  # noqa: E731
    num = lambda s: [None if pd.isna(v) else round(float(v), 4) for v in s]  # noqa: E731
    return {"dates": [t.date().isoformat() for t in tail.index], "close": num(tail["close"]),
            **{f"ma{n}": num(tail[f"ma{n}"]) for n in MAS},
            "grades": letters(grades), "wgrades": letters(weekly) if weekly is not None else None}


# ----------------------------------------------------------------------------- weekly grade
# 판단 단위는 '주'(2026-10-12 연구: S 반감기 3~4주, 일·주·월 갱신 수익은 구별되지 않고 월·분기는 S 구간의 절반 이상을
# 놓친다). 등급은 매일 계산하고, 주간 등급은 직전 주 마지막 거래일 종가의 등급을 그 주 내내 쓴다.
def week_done(as_of: pd.Timestamp, now: datetime | None = None) -> bool:
    """as_of(미국 마지막 확정일)가 그 주의 마지막 거래일인가: 금요일이거나, 그 주 금요일 장 마감 정산(뉴욕 16:15)이
    이미 지났다(성금요일처럼 금요일이 휴장이면 목요일이 그 주 마지막 날)."""
    if as_of.weekday() == 4:
        return True
    friday = (pd.Timestamp(as_of) + pd.Timedelta(days=4 - as_of.weekday())).date()
    ny = (now or datetime.now(config.KST)).astimezone(NY)
    return (ny.date(), ny.strftime("%H:%M")) >= (friday, SESSION["미국"][1])


def weekly_table(letters: pd.DataFrame, done: bool) -> tuple[pd.DataFrame, list]:
    """(날짜 × 종목 그날 쓰던 주간 등급, 주간 결정일 목록). 주간 등급 = 직전 주 마지막 거래일 종가의 등급.
    마지막 주가 끝났으면(done) 그 주 마지막 날도 결정일이 된다(다음 주에 쓸 등급)."""
    idx = letters.index
    weeks = idx.to_period("W-SUN")
    last = pd.Series(idx, index=idx).groupby(weeks).max()
    prev_last = last.shift(1).reindex(weeks).to_numpy()
    vals = letters.to_numpy(object)
    pos = {d: i for i, d in enumerate(idx)}
    rows = [list(vals[pos[pd.Timestamp(d)]]) if not pd.isna(d) else [None] * len(letters.columns) for d in prev_last]
    eff = pd.DataFrame(rows, index=idx, columns=letters.columns, dtype=object)
    decisions = list(last.iloc[:-1]) + ([last.iloc[-1]] if done else [])
    return eff, decisions


def weekly_fields(letters: pd.Series, eff: pd.Series, decisions: list) -> dict:
    """한 종목의 주간 등급(지금 쓸 등급)·지난주 등급·이어진 주 수·결정일·최근 이력."""
    seq = [letters.get(d) if isinstance(letters.get(d), str) else None for d in decisions]
    hist = [g if isinstance(g, str) else None for g in eff.iloc[-HISTORY_DAYS:]]
    return {"wgrade": seq[-1] if seq else None, "wprev": seq[-2] if len(seq) > 1 else None,
            "wstreak": _streak(seq), "wfrom": decisions[-1].date().isoformat() if decisions else None, "whist": hist}


# ----------------------------------------------------------------------------- earnings
# S 등급이 바뀌는 때는 분기 경계가 아니라 종목별 실적 발표 무렵이다(2026-10-12 연구: 실적 창 이탈 1.19배·진입
# 1.37배, 반응일 당일 이탈 3.8배). 그래서 다음·지난 실적 날짜를 함께 보인다. 자료는 yfinance get_earnings_dates.
def _earnings_of(sym: str) -> list[str]:
    import yfinance as yf

    try:
        df = yf.Ticker(sym).get_earnings_dates(limit=8)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    return sorted({ts.date().isoformat() for ts in df.index})


def earnings(symbols: list[str], force: bool = False) -> dict[str, list[str]]:
    """종목별 실적 발표일(지난 것과 예정된 것). EARN_MAX_AGE_H 동안 저장해 둔다. 받지 못하면 빈 목록."""
    cached = {}
    if not force and EARN_PATH.exists():
        try:
            data = json.loads(EARN_PATH.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
        if _time.time() - data.get("_epoch", 0) < EARN_MAX_AGE_H * 3600:
            cached = data.get("dates", {})
    todo = [s for s in symbols if s not in cached]
    if todo:
        log = logging.getLogger("yfinance")
        level = log.level
        log.setLevel(logging.CRITICAL)  # 실적이 없는 ETF·신규 상장마다 찍는 경고를 서버 창에 남기지 않는다
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                cached.update(zip(todo, pool.map(_earnings_of, todo)))
        finally:
            log.setLevel(level)
        config.CACHE.mkdir(parents=True, exist_ok=True)
        _write(EARN_PATH, {"_epoch": _time.time(), "dates": cached})
    return {s: cached.get(s, []) for s in symbols}


def earnings_info(dates: list[str], today: date) -> dict | None:
    """다음 실적(오늘 포함 이후 첫 날)과 지난 실적, 오늘부터의 달력 일수."""
    if not dates:
        return None
    days = [date.fromisoformat(d) for d in dates]
    nxt = [d for d in days if d >= today]
    past = [d for d in days if d < today]
    return {"next": nxt[0].isoformat() if nxt else None, "next_in": (nxt[0] - today).days if nxt else None,
            "last": past[-1].isoformat() if past else None, "last_ago": (today - past[-1]).days if past else None}


REALIZED_DAYS = 20       # '최근 성적': 이 거래일 전 등급별로 그 뒤 종가 수익을 본다(사후 확인용)


def realized(letters: pd.DataFrame, close: pd.DataFrame, bench_close: pd.Series, days: int = REALIZED_DAYS) -> dict | None:
    """days거래일 전 종가의 등급별로 그 뒤 오늘 종가까지의 평균 수익(%). 다음 기간을 맞히는 값이 아니다
    (연구: 최근 성적이 좋았던 직후 오히려 크게 뒤집혔다)."""
    if len(letters) <= days:
        return None
    base = letters.index[-1 - days]
    then = letters.iloc[-1 - days]
    ret = (close.iloc[-1] / close.iloc[-1 - days] - 1.0) * 100.0
    ok = then.notna() & ret.notna()
    by = {}
    for g in (*(x for x, _ in LADDER), "D"):
        sel = ret[ok & (then == g)]
        by[g] = {"n": int(len(sel)), "ret": round(float(sel.mean()), 2) if len(sel) else None}
    b = bench_close.reindex(letters.index, method="ffill")
    return {"days": days, "from": base.date().isoformat(), "by_grade": by,
            "all": round(float(ret[ok].mean()), 2) if ok.any() else None,
            "bench": round(float((b.iloc[-1] / b.iloc[-1 - days] - 1.0) * 100.0), 2)}


def compute(now: datetime | None = None) -> dict:
    rows, fund_dates = holdings()
    if not rows:
        raise RuntimeError("두 펀드의 검증 스냅샷이 없습니다 — 펀드 자료 갱신(py gbrc.py briefing)을 먼저 하세요")
    bars = fetch([r["symbol"] for r in rows] + [r["proxy"] for r in rows if r.get("proxy")] + [BENCH])
    own_bars = {}
    for r in rows:  # 상장이 짧으면 같은 회사의 다른 상장 이력으로 등급을 매긴다(화면에 표시)
        own, alt = bars.get(r["symbol"]), bars.get(r.get("proxy") or "")
        if alt is not None and (own is None or len(own) < MIN_BARS) and len(alt) >= MIN_BARS:
            r["proxy_used"] = {"symbol": r["proxy"], "own_bars": 0 if own is None else len(own)}
            own_bars[r["symbol"]] = own
            bars[r["symbol"]] = alt
    if BENCH not in bars:
        raise RuntimeError(f"{BENCH} 가격을 받지 못했습니다 — 잠시 뒤 다시 계산하세요")
    bench = settled(bars.pop(BENCH), "미국", now)
    market_of = lambda r: "한국" if r.get("proxy_used") and r["proxy"].endswith((".KS", ".KQ")) else r["market"]  # noqa: E731
    missing = [r["symbol"] for r in rows if r["symbol"] not in bars]
    if len(missing) > len(rows) * MAX_MISSING:  # 요청 제한 등으로 많이 빠지면 순위 모집단이 줄어 등급이 틀린다 — 캐시하지 않는다
        raise RuntimeError(f"보유 {len(rows)}종목 중 {len(missing)}종목의 시세를 받지 못했습니다 — 잠시 뒤 다시 계산하세요")
    ind = {r["symbol"]: indicators(settled(bars[r["symbol"]], market_of(r), now))
           for r in rows if r["symbol"] in bars and len(bars[r["symbol"]])}
    # 10일 평균이 화면 첫 칸부터 다 쌓이도록 SMOOTH일 더 앞에서 시작한다.
    cal = bench.index[-(max(CHART_DAYS, HISTORY_DAYS) + 5 + SMOOTH):]
    # 순위는 미국 보유끼리(연구와 같은 모집단). 다른 상장 이력으로 대신 잰 종목(SKHY → 000660.KS)은 한국 거래일·원화라
    # 미국 밖 종목처럼 그 분포에 견준다.
    ref = {r["symbol"] for r in rows if r["market"] == "미국" and not r.get("proxy_used")}
    pct, raw_pct, parts = score_table(ind, cal, ref)
    letters = grade_table(pct, ind, cal)
    raw_today = grade_table(raw_pct.iloc[-1:], ind, cal[-1:]).iloc[-1]
    weekly, decisions = weekly_table(letters, week_done(cal[-1], now))
    today_ny = (now or datetime.now(config.KST)).astimezone(NY).date()
    earn = earnings([r["symbol"] for r in rows])
    stocks, charts = [], {}
    for r in rows:
        sym = r["symbol"]
        seq = list(letters[sym]) if sym in letters else []
        now_parts = {k: v[sym].iloc[-1] if sym in v else None for k, v in parts.items()}
        # 화면 숫자도 등급과 같은 날짜(기준 달력의 마지막 날 또는 그 전 현지 거래일)로 맞춘다. 한국 장이 끝난
        # 저녁에는 한국 종목의 오늘 봉이 있지만, 등급은 아직 어제 미국 마감 기준이기 때문이다.
        d = ind[sym].loc[:cal[-1]] if sym in ind else None
        stocks.append(stock_row(r, d, seq, pct[sym].iloc[-1] if sym in pct else None, now_parts,
                                raw_today.get(sym) if sym in raw_today.index else None))
        if sym in letters:
            stocks[-1].update(weekly_fields(letters[sym], weekly[sym], decisions))
        stocks[-1]["earn"] = earnings_info(earn.get(sym, []), today_ny)
        if own_bars.get(sym) is not None:  # 대체 이력 종목: 종가·수익률은 그 상장 자체 가격(이평선 대비·등급·차트는 대체 이력)
            c = settled(own_bars[sym], r["market"], now)["Close"].loc[:cal[-1]]
            ret = lambda n: _pct(c.iloc[-1] / c.iloc[-1 - n] - 1) if len(c) > n else None  # noqa: E731
            if len(c):
                stocks[-1].update(date=c.index[-1].date().isoformat(), close=float(c.iloc[-1]),
                                  chg1d=ret(1), r1m=ret(21), r3m=ret(63))
        if d is not None and len(d):
            charts[sym] = {"symbol": sym, "proxy": (r.get("proxy_used") or {}).get("symbol"),
                           **chart_of(d, letters[sym] if sym in letters else pd.Series(dtype=object), cal[-CHART_DAYS],
                                      weekly[sym] if sym in weekly else None)}
    recent = realized(letters, _wide(ind, "close", cal), bench["Close"])
    graded = [s for s in stocks if s["grade"]]
    dist = {g: sum(1 for s in graded if s["grade"] == g) for g, _ in (*LADDER, ("D", 0))}
    dist["none"] = len(stocks) - len(graded)
    above = lambda k: round(100 * sum(1 for s in graded if (s.get(k) or 0) > 0) / len(graded)) if graded else None  # noqa: E731
    bi = indicators(bench).iloc[-1]
    warnings = [f"{s['symbol']} 시세를 받지 못했습니다" for s in stocks if s.get("why") == "시세 없음"]
    stale = [k for k, v in fund_dates.items() if (cal[-1].date() - datetime.fromisoformat(v).date()).days > 7]
    if stale:
        warnings.append(f"{', '.join(stale)} 보유 내역이 일주일 넘게 지났습니다 — 펀드 자료 갱신 후 다시 계산하세요")
    data = {
        "computed_at": (now or datetime.now(config.KST)).isoformat(timespec="seconds"),
        "as_of": cal[-1].date().isoformat(), "prev_day": cal[-2].date().isoformat(),
        "funds": fund_dates, "days": [t.date().isoformat() for t in cal[-HISTORY_DAYS:]],
        "week_from": decisions[-1].date().isoformat() if decisions else None,
        "week_prev": decisions[-2].date().isoformat() if len(decisions) > 1 else None,
        "today_ny": today_ny.isoformat(),
        "dist": dist, "breadth": {"above50": above("d50"), "above200": above("d200"), "n": len(graded)},
        "bench": {"symbol": BENCH, "d50": _pct(bi["M"]), "d200": _pct(bi["L"]), "h52": _pct(bi["H"])},
        "realized": recent, "stocks": stocks, "rules": rules(), "check": CHECK, "warnings": warnings,
    }
    return data, charts


def payload(force: bool = False) -> dict:
    """캐시가 MAX_AGE_MIN보다 새로우면 그대로 쓰고, 아니면 다시 계산한다(10~20초)."""
    with _LOCK:
        if not force and CACHE_PATH.exists() and CHART_PATH.exists():
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if cached.get("_spec") == SPEC and _time.time() - cached.get("_epoch", 0) < MAX_AGE_MIN * 60:
                cached.update(rules=rules(), check=CHECK)  # 규칙·검증 문구는 캐시가 아니라 지금 코드에서
                return cached
        data, charts = compute()
        data = clean(data)
        data["_epoch"], data["_spec"] = _time.time(), SPEC
        config.CACHE.mkdir(parents=True, exist_ok=True)
        _write(CHART_PATH, clean(charts))  # 차트를 먼저: 새 등급이 보이면 그 차트도 있다
        _write(CACHE_PATH, data)
        return data


def rules() -> dict:
    return {
        "ladder": [list(x) for x in LADDER], "labels": LABELS, "parts": PARTS, "part_notes": PART_NOTES,
        "smooth": SMOOTH, "min_bars": MIN_BARS, "sma_min": SMA_MIN, "history_days": HISTORY_DAYS,
        "notes": [
            "점수: 그날 등급 대상 종목 안에서 네 재료의 순위(백분위)를 같은 무게로 평균합니다 — 200일선 위 거리, 50일선 위 거리, "
            "52주 고점 근접, 126일 가격 추세선의 기울기×매끄러움(Clenow). 이 점수를 최근 10거래일 평균해 다시 순위를 냅니다.",
            "등급: 순위 상위 10% S, 다음 20% A, 가운데 40% B, 다음 20% C, 하위 10% D. 다만 S·A는 종가>50일선>200일선(상승 배열)일 때만 "
            "주고 아니면 B로 내립니다. 종가가 200일선 아래이고 50일선도 200일선 아래(하락 배열)면 순위와 무관하게 D입니다.",
            "상대 등급이라 시장 전체가 빠져도 S는 대개 있습니다. 관문 때문에 급락장에서는 S·A가 줄고 D가 늘어납니다.",
            "등급 대상: 두 펀드 최신 보유 내역의 주식·ETF(선물·현금 제외) 중 일봉 130개 이상. 200일선·52주 고점은 상장 120거래일부터 가진 봉으로 계산합니다. "
            "과거 등급 이력도 오늘 보유 종목끼리 비교한 값이라, 펀드가 끝까지 들고 간 종목 위주여서 실제보다 좋아 보입니다.",
            "순위는 미국 보유 종목끼리 냅니다(연구 검증과 같은 모집단). 미국 밖 종목(한국·일본·홍콩·중국)은 같은 날 미국 종목들 사이에서 몇 등쯤인지로 "
            "등급을 매겨 미국 종목 등급을 바꾸지 않습니다. 현지 통화 종가로 재고, 그 미국 날짜 또는 직전 현지 거래일 값을 씁니다. 미국 밖 종목 등급의 성적은 검증하지 않았습니다.",
            "‘당일 X’는 10일 평균하지 않은 오늘 점수만으로 매긴 등급입니다. 7월 급락 때 S 종목이 평균 등급에서 내려오기까지 중앙값 6거래일(당일 점수는 2일)이 걸려서 함께 보입니다.",
            "주간 등급: 직전 주 마지막 거래일 종가의 등급을 그 주 내내 씁니다(금요일 마감 뒤에는 그 금요일 등급이 다음 주 등급). 판단 단위로는 주가 근거가 "
            "가장 좋았습니다. S는 절반이 3~4주 안에 바뀌고, 일·주·월로 바꿔도 수익은 구별되지 않았으며, 월·분기 단위는 S 구간의 절반 이상을 놓쳤습니다. "
            "주간 기준에서는 다를 때 더 빠른 일별 등급(‘일별 X’)을 함께 붙입니다.",
            "실적: S 등급은 분기 경계가 아니라 종목별 실적 발표 무렵에 가장 자주 바뀌었습니다(발표 무렵 이탈 1.2배·진입 1.4배, 발표 다음 날 이탈은 평소의 3.8배). "
            "그래서 7일 안 발표와 5일 안에 지난 발표를 꼬리표로 표시합니다. 날짜는 Yahoo 실적 일정(12시간마다 갱신)이고, ETF·신규 상장은 없을 수 있습니다.",
            "장이 열려 있는 시장의 오늘 봉은 마감 15분 뒤까지 빼고 계산합니다. 그래서 장중에는 어제 종가 기준 등급입니다. "
            "표의 숫자도 등급과 같은 날짜 기준입니다(한국 장이 끝난 저녁에도 한국 종목은 어제 미국 마감일 기준).",
            "가격은 Yahoo Finance(yfinance) 2년 일봉, 분할 반영 종가입니다. 거래량·실적·뉴스는 쓰지 않습니다. 계산은 60분 동안 저장해 둡니다.",
            "관찰용 화면입니다. 매수 추천·목표 비중·주문 수량을 만들지 않습니다(데스크 §1).",
        ],
    }


class TrendError(ValueError):
    """화면에 400으로 돌려줄 입력 오류."""


def _write(path: Path, data) -> None:
    """임시 파일에 다 쓴 뒤 바꿔 끼운다. 차트를 읽는 요청이 반쯤 쓰인 파일을 보지 않는다."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def chart(symbol: str) -> dict:
    """한 종목 차트(종가·20·50·200일선·날짜별 등급). 등급 계산 때 함께 저장한 캐시에서 읽는다."""
    if not symbol:
        raise TrendError("종목 코드가 비었습니다")
    if not CHART_PATH.exists():
        payload()
    charts = json.loads(CHART_PATH.read_text(encoding="utf-8"))
    if symbol in charts:
        return charts[symbol]
    held = CACHE_PATH.exists() and any(s.get("symbol") == symbol for s in
                                       json.loads(CACHE_PATH.read_text(encoding="utf-8")).get("stocks", []))
    raise TrendError(f"{symbol}은(는) 시세를 받지 못해 차트가 없습니다" if held
                     else f"{symbol}은(는) 두 펀드 보유 종목이 아닙니다")
