"""급락 후 담기 탭: 오늘 크게 빠진 미국 종목을 찾고, 프리장·정규장 흐름과 과거 비슷한 경우의 결과를 보여 준다.

정본은 docs/desk_system.md §10이다. 관찰용 화면이라 매수 추천·목표 비중·주문 수량을 만들지 않는다(§1).

- 종목 목록: 나스닥 공개 종목표(api.nasdaq.com)의 시총 10억 달러 이상 미국 상장 보통주. 하루 한 번 받는다.
- 시세: Yahoo 시세 묶음 요청(1,000종목씩)으로 장전·장중·장후 가격을 받는다. 급락 후보만 1분봉을 받아
  프리장 저가·반등·시각별 조건을 계산한다. 3분봉은 1분봉을 묶어 만든다(Yahoo에는 3분봉이 없다).
- 판정 규칙(오류 체결 거르기, 시각별 조건, 정규장 매수 규칙)은 연구 스크립트
  data/derived/analysis/premarket_shock/code/의 hours.py·rebound.py·intraday.py와 같다. 데스크에서는 이 파일이 정본이다.
- 근거 수치는 같은 폴더 report.md의 연구 결과를 아래 상수로 옮겼고 화면이 그대로 읽는다.
- TradingView MCP 값은 넣지 않는다(AGENTS.md).
"""

from __future__ import annotations

import json
import re
import threading
import time as _time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import config
from .state import clean

CACHE_PATH = config.CACHE / "dip.json"
UNIVERSE_PATH = config.CACHE / "dip_universe.json"
SCAN_MAX_AGE_SEC = 120          # 급락 목록 캐시
UNIVERSE_MAX_AGE_H = 24
MIN_CAP = 1e9                   # 시총 10억 달러 이상
DROP = 7.0                      # 급락 후보: 기준 종가 대비 -7% 이하
MAX_DETAIL = 40                 # 1분봉으로 프리장 흐름을 계산하는 후보 수(하락 큰 순)
DEEP = 15.0                     # 연구의 급락 사건 문턱: 프리장 -15%
CONFIRM = 10.0                  # 오류 체결 거르기: 1시간봉 종가 -10% 이하가 연속 두 봉
PM_FILTER = 7.0                 # 프리장 저가는 5분봉 종가가 -7% 이하인 봉에서만 잰다
REBOUND_08 = 5.0                # 08:00 조건: 그 시각 프리장 저점 대비 반등
BAND_0929 = (10.0, 20.0)        # 09:29 조건: 프리장 저점 대비 반등 10~20%
AH_NEWS = -5.0                  # 전날 애프터 종가가 이만큼 빠졌으면 '전날 애프터 뉴스'
MARKET = -1.5                   # SPY가 이만큼 빠졌으면 '시장 동반 급락'
TPS = (2.0, 3.0, 5.0)           # 장중 규칙 재생의 익절 폭
HOLD_MIN = 15                   # 바닥 15분: 그날 저가가 15분 동안 새로 나오지 않음
HOLD_NEAR = 2.0                 # 그리고 종가가 그 저가의 +2% 안(아직 바닥 근처)
HOLD_DD = -7.0                  # 그리고 그 순간 전일 종가 대비 -7% 이하
HOLD_START = 9 * 60 + 45        # 09:45 이후 마감한 봉부터(연구와 같다)
HOLD_STOP = 1.0                 # 바닥 15분 매수의 손절: 매수 전 저가 -1%(봉 저가가 닿으면)
WINDOW_END = 11 * 60            # 11:00 ET = 서머타임 24:00 KST, 사용자 주문 가능 시간 끝
OPEN_STOP = 5.0                 # 시가 매수 손절 기준(그 전 장중 저가가 없으므로)
DETAIL_DAYS = 29                # Yahoo 1분봉은 최근 30일, 한 요청 8일
QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
NASDAQ_LIST = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true"
NASDAQ_NEWS = "https://api.nasdaq.com/api/news/topic/articlebysymbol?q={sym}|stocks&offset=0&limit=8"
NOT_COMMON = re.compile(r"Warrant|\bUnits?\b|\bRights?\b|Preferred|\bNotes\b|Debentures", re.I)
HOUR_LABEL = {4: "05:00", 5: "06:00", 6: "07:00", 7: "08:00", 8: "09:00", 9: "09:29"}
_LOCK = threading.Lock()
_DETAIL: dict[tuple[str, str], tuple[float, dict]] = {}

# ---------------------------------------------------------------- 연구 근거(report.md에서 옮긴 값)
STUDY = {
    "report": "data/derived/analysis/premarket_shock/report.md",
    "universe": "나스닥 공개 종목표의 미국 상장 보통주 3,672종목, 사건 시점 시총 10억 달러 이상",
    "period": "2023-11 ~ 2026-10 (1시간봉), 2026-08-10 ~ 10-06 (분봉)",
    "event": "프리장 1시간봉 종가 -10% 이하가 연속 두 봉이고 그 저가가 -15% 이하인 날(오류 체결을 거르기 위한 조건)",
}
# 프리장 매수 시각별: 그 시각 -15% 이하(직전 봉 -10% 이하)에 사서 당일 +5% 지정가, 못 닿으면 다음 날 시가.
# 실적 외·개별 악재. (시각, 2025 닿음%, 2025 평균%, 2025 건, 2026 닿음%, 2026 평균%, 2026 건, 못 닿은 건 평균%)
TIME_STATS = [
    ("05:00", 62, -3.3, 39, 60, -2.4, 20, -15.6),
    ("06:00", 47, -4.9, 60, 57, -1.5, 44, -12.2),
    ("07:00", 47, -4.0, 60, 53, -1.3, 43, -10.6),
    ("08:00", 49, -2.3, 73, 57, -1.3, 54, -9.6),
    ("09:00", 53, -1.2, 112, 53, -1.9, 73, -8.8),
    ("09:29", 60, -0.2, 129, 53, -0.2, 92, -7.0),
    ("시가", 59, -0.0, 129, 54, -0.4, 92, -7.0),
]
# 08:00 매수를 '그 시각 프리장 저점에서 5% 이상 반등 중'으로 나눈 결과(68개 후보 중 두 해 모두 플러스였던 칸, t 0.8·0.1)
TIME_REBOUND_08 = {"반등 중": (76, 1.0, 17, 74, 1.0, 19), "저점 근처": (41, -3.3, 56, 49, -2.6, 35)}
# 프리장 뉴스·실적 외 급락을 09:29 저점 대비 반등 폭으로 나눠 시가에 산 결과: (첫 1시간 종가 %, 다음 날 시가 %, 건)
REBOUND_STATS = [
    ("5% 미만", {"2026상": (0.2, -1.7, 29), "2026하": (1.0, -2.2, 16), "2025": (-0.3, -1.0, 63), "2024": (-0.2, 0.3, 58)}),
    ("5~10%", {"2026상": (0.9, -4.2, 22), "2026하": (-0.8, -1.5, 4), "2025": (0.0, -0.5, 31), "2024": (-0.9, -1.4, 39)}),
    ("10~20%", {"2026상": (6.0, 6.6, 13), "2026하": (6.3, 6.3, 6), "2025": (-0.2, 0.9, 25), "2024": (2.3, 5.5, 23)}),
    ("20% 이상", {"2026상": (2.4, 1.0, 9), "2026하": None, "2025": (1.7, 3.3, 17), "2024": (3.4, 1.7, 9)}),
]
# 정해 둔 하락률에 지정가로 사서 시가에 판 평균(실적 외·개별, 체결가 대비 %, 건)
LIMIT_STATS = [
    ("-15%", {"2026상": (-2.4, 119), "2026하": (-7.2, 43), "2025": (-3.0, 214), "2024": (-1.8, 204)}),
    ("-20%", {"2026상": (-5.6, 52), "2026하": (-8.8, 26), "2025": (-3.2, 124), "2024": (-2.8, 96)}),
    ("-25%", {"2026상": (-4.5, 39), "2026하": (-15.2, 14), "2025": (-4.0, 72), "2024": (-2.0, 64)}),
    ("-30%", {"2026상": (-10.5, 21), "2026하": (-16.8, 11), "2025": (-4.5, 50), "2024": (-3.0, 39)}),
    ("-35%", {"2026상": (-8.0, 17), "2026하": (-16.0, 10), "2025": (-3.3, 35), "2024": (-2.8, 26)}),
]
HOLD_STATS = {"broke": "48~52%", "new_low_5d": "62~72%", "ah_broke": "53~65%"}
# 갭 하락(시가 -7% 이하) 종목의 정규장 분봉. 앞 = 8/10~9/5(5분봉 127건), 뒤 = 9/8~10/6(3분봉 107건)
INTRADAY_LOW = {"앞": (51, 58, 65, 21), "뒤": (50, 51, 61, 30)}   # 첫 15분·첫 30분·11시 전·13시 이후 저점 비율%
INTRADAY_STATS = [  # (규칙, +익절, 앞 닿음%, 앞 평균%, 앞 건, 뒤 닿음%, 뒤 평균%, 뒤 건)
    ("시가", 3, 65, 0.7, 127, 50, 0.1, 107),
    ("시가", 5, 45, 0.6, 127, 21, -0.2, 107),
    ("10:00", 3, 42, 0.1, 123, 23, 0.1, 104),
    ("저점+2% 반등", 3, 48, -0.6, 108, 35, -0.4, 98),
    ("VWAP 회복", 3, 44, -0.3, 78, 33, -0.0, 81),
    ("15분 고가 돌파", 3, 36, 0.5, 47, 14, -0.5, 66),
]
# 그날 바닥일 확률(2026-10-08 연구 lowfind): 15분마다 그 순간 -7% 이하인 종목의 지금까지 저가가 그날 끝까지 버틴 비율.
# (저가 뒤 지난 시간, 앞 8/10~9/5 5분봉 %, 뒤 9/8~10/6 5분봉 %, 뒤 3분봉 %, 깨진 날 더 빠진 폭 중앙, 그 하위 10%)
SURVIVAL = [
    ("0~14분", 13, 12, 12, "−1.3~−1.7%", "−4.4~−5.4%"),
    ("15~29분", 35, 31, 31, "−1.2~−1.5%", "−4.2~−4.4%"),
    ("30~59분", 40, 40, 40, "−1.0~−1.4%", "−3.8~−4.0%"),
    ("60~119분", 50, 48, 50, "−0.9~−1.4%", "−3.3~−3.6%"),
    ("120분 이상", 63, 59, 59, "−0.8~−1.3%", "−2.5~−3.4%"),
]
SURVIVAL_EDGES = [(0, 15), (15, 30), (30, 60), (60, 120), (120, 10_000)]
# 바닥 15분 매수(11:00 ET 전 신호, +2% 지정가, 못 닿으면 다음 날 시가). 60개 조합 중 세 칸 모두 플러스였던 6개의 대표.
# (손절, [(기간·봉, 평균%, 승률%, 익절%, 손절%, 최악%, 건)])
BOTTOM_STATS = [
    ("손절 없음", [("앞 5분봉", 0.41, 66, 52, 0, -21.2, 250), ("뒤 5분봉", 0.14, 63, 49, 0, -11.1, 226),
                ("뒤 3분봉", 0.10, 64, 49, 0, -15.8, 222)]),
    ("11:00까지 저가 −1% 손절", [("앞 5분봉", 0.31, 57, 47, 22, -11.4, 250), ("뒤 5분봉", 0.06, 57, 45, 15, -6.9, 226),
                           ("뒤 3분봉", 0.12, 57, 44, 18, -6.9, 222)]),
]
CASES = [
    {"t": "SMCI", "day": "2026-03-20", "news": "공동창업자 대중국 엔비디아 칩 밀수 기소(법무부)", "when": "전날 18:00 ET(애프터)",
     "pm_low": -28.1, "at_0929": -27.1, "reb": 1.4, "open": -27.0, "close": -33.2,
     "after": "다음 날 프리장 -38.5% 신저가", "lesson": "형사 사건·엔비디아 공급 의존 → 크기를 셀 수 없는 악재. 개장 전 반등이 없었다."},
    {"t": "FUTU", "day": "2026-05-22", "news": "중국 증권감독관리위원회 처벌 사전 통지(벌금 18.5억 위안·본토 영업 정리)", "when": "프리장",
     "pm_low": -41.1, "at_0929": -34.6, "reb": 11.1, "open": -34.3, "close": -27.5,
     "after": "다음 날 종가 -13.0%(시가 대비 +32%)", "lesson": "강제력은 있었지만 크기가 숫자로 나왔다(본토 계좌 13%) → 최악 확정 뒤 반등."},
    {"t": "BULL", "day": "2026-10-07", "news": "하원 중국특위 '중국과 구조적 연결' 보고서", "when": "프리장 07:00 ET",
     "pm_low": -32.7, "at_0929": -19.8, "reb": 19.2, "open": -19.7, "close": None,
     "after": "시가 대비 첫 1시간 -0.5%", "lesson": "강제력 없는 보고서·이미 알려진 이야기 → 개장 전 절반 회복. 후속 규제가 나오면 다른 경우가 된다."},
    {"t": "STX", "day": "2026-10-02", "news": "도시바 AI용 하드디스크 생산 두 배 보도(닛케이)", "when": "프리장 06~07시 ET",
     "pm_low": -14.4, "at_0929": -13.1, "reb": 1.6, "open": -13.2, "close": -10.2,
     "after": "개장 3분 저점 -16.3% → 09:48 -10.8% → 10:54 -15.5% 재확인 → 종가 -10.2%, 다음 날 시가 -7.4%",
     "lesson": "개장 직후 바닥(갭 하락일의 약 절반)과 오후 회복. 같은 종목이 10/6에는 장 내내 밀렸다."},
]


class DipError(ValueError):
    """사용자 입력(종목·날짜)이 잘못됐거나 자료 범위 밖일 때 — 화면에 그대로 보여 줄 문장."""


# ---------------------------------------------------------------- 순수 계산(테스트 대상)
def classify_quote(q: dict, now_epoch: float | None = None) -> dict | None:
    """Yahoo 시세 한 줄 → 세션, 기준 종가, 최신가, 등락(%). 장전은 전일 종가, 장중·장후는 전전일 종가 기준."""
    price, prev = q.get("regularMarketPrice"), q.get("regularMarketPreviousClose")
    if not price or not prev:
        return None
    state = q.get("marketState") or ""
    t_reg = q.get("regularMarketTime") or 0
    pre, t_pre = q.get("preMarketPrice"), q.get("preMarketTime") or 0
    post, t_post = q.get("postMarketPrice"), q.get("postMarketTime") or 0
    row = {"symbol": q.get("symbol"), "name": q.get("shortName") or q.get("longName") or q.get("symbol"),
           "cap": q.get("marketCap"), "avg_volume": q.get("averageDailyVolume3Month"), "day_chg": None, "post_chg": None}
    if state in ("PRE", "PREPRE") and pre and t_pre > t_reg:
        row.update(session="장전", ref=price, last=pre)
    elif state == "REGULAR":
        row.update(session="장중", ref=prev, last=price, day_chg=_pct(price / prev))
    else:  # 장 마감 뒤: 정규장 등락에 장후 등락을 더해 전전일 종가와 비교한다
        last = post if post and t_post > t_reg else price
        # PREPRE(20:00~04:00 ET, 한국 낮)에는 Yahoo에 새 시세가 없어 지난 세션과 장후 가격을 보여 준다
        row.update(session="장후" if state in ("POST", "POSTPOST", "PREPRE") else "휴장", ref=prev, last=last,
                   day_chg=_pct(price / prev), post_chg=_pct(post / price) if post and t_post > t_reg else None)
    row["chg"] = _pct(row["last"] / row["ref"])
    ts = q.get("earningsTimestamp")
    now_epoch = now_epoch or _time.time()
    row["earnings"] = bool(ts and now_epoch - 36 * 3600 <= ts <= now_epoch + 2 * 3600)
    return row


def _pct(ratio) -> float | None:
    return None if ratio is None or not np.isfinite(ratio) else round((float(ratio) - 1) * 100, 2)


def _minutes(idx) -> np.ndarray:
    return np.asarray(idx.hour * 60 + idx.minute)


def split_sessions(bars: pd.DataFrame, day: date) -> dict:
    """1분봉(뉴욕 시각) → 그날 장전·정규장·장후, 전 거래일 정규장·장후, 다음 거래일 정규장 시가."""
    m = _minutes(bars.index)
    dates = np.asarray(bars.index.date)
    reg = (m >= 570) & (m < 960)
    prev_days = sorted({d for d in dates[reg] if d < day})
    prev = prev_days[-1] if prev_days else None
    nxt = bars[reg & (dates > day)]
    return {
        "pm": bars[(dates == day) & (m >= 240) & (m < 570)],
        "reg": bars[(dates == day) & reg],
        "post": bars[(dates == day) & (m >= 960) & (m < 1200)],
        "prev_reg": bars[(dates == prev) & reg] if prev else bars.iloc[:0],
        "prev_ah": bars[(dates == prev) & (m >= 960) & (m < 1200)] if prev else bars.iloc[:0],
        "next_open": float(nxt.Open.iloc[0]) if len(nxt) else None,
    }


def resample(bars: pd.DataFrame, minutes: int, origin: str = "start_day") -> pd.DataFrame:
    if bars.empty:
        return bars
    return bars.resample(f"{minutes}min", origin=origin).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Open"])


def pm_low(pm: pd.DataFrame, c0: float) -> tuple[float | None, str | None]:
    """프리장 저가. 5분봉 종가가 -7% 이하인 봉의 저가만 써서 한 번 찍고 사라지는 오류 체결을 거른다."""
    if pm.empty or not c0:
        return None, None
    b5 = resample(pm, 5)
    ok = b5[b5.Close <= c0 * (1 - PM_FILTER / 100)]
    if ok.empty:
        return None, None
    return float(ok.Low.min()), ok.Low.idxmin().strftime("%H:%M")


def hourly_conditions(pm: pd.DataFrame, prev_ah: pd.DataFrame, c0: float) -> list[dict]:
    """연구(hours.py)의 시각별 매수 조건: 그 시각 1시간봉 종가 -15% 이하, 직전 봉(첫 봉이면 전날 애프터 마지막
    1시간) -10% 이하, 두 봉 모두 폭이 있다(이어 찍힌 봉 거르기). 반등은 그때까지 인정 봉 저가 대비."""
    out = []
    if pm.empty or not c0:
        return out
    hours = pm.resample("60min", origin="start_day", offset="0min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna(subset=["Open"])
    ah = prev_ah.resample("60min", origin="start_day").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna(subset=["Open"]) if len(prev_ah) else prev_ah
    prev = ah.iloc[-1] if len(ah) else None
    ok = lambda b: b is not None and b.Close <= c0 * (1 - CONFIRM / 100) and b.High > b.Low * 1.0005
    lows = []
    for ts, b in hours.iterrows():
        if ok(b):
            lows.append(b.Low)
        cond = bool(b.Close <= c0 * (1 - DEEP / 100) and ok(b) and ok(prev))
        reb = (b.Close / min(lows) - 1) * 100 if lows else None
        if ts.hour in HOUR_LABEL:
            out.append({"at": HOUR_LABEL[ts.hour], "hour": ts.hour, "close_pct": _pct(b.Close / c0), "cond": cond,
                        "reb": None if reb is None else round(reb, 2)})
        prev = b
    return out


def features(bars: pd.DataFrame, day: date, ref: float | None, earnings: bool, now_et: datetime | None = None) -> dict:
    """한 종목의 그날 흐름: 기준 종가, 전날 애프터, 프리장 저가·반등, 08:00·09:29 조건, 정규장 요약."""
    s = split_sessions(bars, day)
    c0 = float(s["prev_reg"].Close.iloc[-1]) if len(s["prev_reg"]) else ref
    if not c0:
        return {}
    ah_ret = _pct(s["prev_ah"].Close.iloc[-1] / c0) if len(s["prev_ah"]) else 0.0
    lo, lo_at = pm_low(s["pm"], c0)
    pm_last = float(s["pm"].Close.iloc[-1]) if len(s["pm"]) else None
    reb = _pct(pm_last / lo) if lo and pm_last else None
    hours = hourly_conditions(s["pm"], s["prev_ah"], c0)
    now_m = (now_et.hour * 60 + now_et.minute) if now_et and now_et.date() == day else 24 * 60
    h8 = next((h for h in hours if h["at"] == "08:00"), None)
    if now_m < 8 * 60:
        flag08 = None                                      # 아직 08:00 전
    else:
        flag08 = bool(h8 and h8["cond"] and h8["reb"] is not None and h8["reb"] >= REBOUND_08)
    if now_m < 9 * 60 + 29:
        flag0929 = None
    else:
        flag0929 = bool(ah_ret > AH_NEWS and not earnings and lo and lo <= c0 * (1 - DEEP / 100)
                        and reb is not None and BAND_0929[0] <= reb < BAND_0929[1])
    out = {"c0": c0, "ah_ret": ah_ret, "news_ah": ah_ret <= AH_NEWS, "pm_low": _pct(lo / c0) if lo else None,
           "pm_low_at": lo_at, "pm_last": _pct(pm_last / c0) if pm_last else None, "reb": reb,
           "hours": hours, "flag08": flag08, "flag0929": flag0929, "next_open": _pct(s["next_open"] / c0) if s["next_open"] else None}
    reg = s["reg"]
    if len(reg):
        o = float(reg.Open.iloc[0])
        low_ts = reg.Low.idxmin()
        out.update(open=_pct(o / c0), reg_low=_pct(reg.Low.min() / c0), reg_low_at=low_ts.strftime("%H:%M"),
                   reg_last=_pct(reg.Close.iloc[-1] / c0), vwap=_pct(_vwap(reg).iloc[-1] / c0),
                   from_low=_pct(reg.Close.iloc[-1] / reg.Low.min()))
        bh = bottom_hold(closed_bars(resample(reg, 3), 3, now_m), c0, 3)
        out["hold"] = {k: bh.get(k) for k in ("signal_at", "since", "near", "dd", "survival")}
        # 바닥 15분: 신호가 났으면 True, 11:00가 지났는데 없으면 False, 그 전이면 None(지켜보는 중)
        out["flag_hold"] = True if bh.get("signal_at") else (False if now_m > WINDOW_END else None)
    return out


def _vwap(reg: pd.DataFrame) -> pd.Series:
    tp = (reg.High + reg.Low + reg.Close) / 3
    vol = reg.Volume.cumsum().replace(0, np.nan)
    return ((tp * reg.Volume).cumsum() / vol).ffill().fillna(reg.Close)


def closed_bars(b: pd.DataFrame, minutes: int, now_m: int) -> pd.DataFrame:
    """실시간에는 아직 끝나지 않은 봉을 뺀다(봉 마감 때 판단하는 규칙이 중간 가격으로 신호를 내지 않게)."""
    if b.empty:
        return b
    return b[_minutes(b.index) + minutes <= now_m]


def survival_of(since: float | None) -> int | None:
    """저가 뒤 지난 분 → 연구(뒤 3분봉)의 그날 바닥일 확률(%)."""
    if since is None:
        return None
    for (lo_, hi_), row in zip(SURVIVAL_EDGES, SURVIVAL):
        if lo_ <= since < hi_:
            return row[3]
    return None


def bottom_hold(b: pd.DataFrame, c0: float, minutes: int) -> dict:
    """바닥 15분(연구 lowfind의 NL15_2): 봉 마감 때 그날 저가가 15분 동안 새로 안 나왔고, 종가가 저가 +2% 안이며,
    전일 종가 대비 -7% 이하면 신호. 신호는 09:45~11:00에 마감한 봉에서 첫 번째만 센다(사용자가 깨어 있는 시간).
    마지막 봉 기준의 지금 상태(저가 뒤 지난 분, 저가 대비, 전일 대비)도 함께 돌려준다."""
    out = {"signal_i": None}
    if b.empty or not c0:
        return out
    mins = _minutes(b.index)
    end = mins + minutes
    lo, cl = b.Low.to_numpy(), b.Close.to_numpy()
    run_low = np.minimum.accumulate(lo)
    low_end = np.zeros(len(b), int)                    # 그때까지 저가가 나온 봉이 끝난 시각
    for i in range(len(b)):
        low_end[i] = end[i] if i == 0 or lo[i] <= run_low[i - 1] else low_end[i - 1]
    for i in range(len(b)):
        if end[i] < HOLD_START or end[i] > WINDOW_END:
            continue
        since = end[i] - low_end[i]
        if (since >= HOLD_MIN and (cl[i] / run_low[i] - 1) * 100 <= HOLD_NEAR
                and (cl[i] / c0 - 1) * 100 <= HOLD_DD):
            out["signal_i"] = i
            out["signal_at"] = f"{end[i] // 60:02d}:{end[i] % 60:02d}"
            break
    last = len(b) - 1
    out.update(since=int(end[last] - low_end[last]), near=_pct(cl[last] / run_low[last]), dd=_pct(cl[last] / c0),
               low=float(run_low[last]), now_end=int(end[last]))
    out["survival"] = survival_of(out["since"])
    return out


def intraday_rules(b: pd.DataFrame, minutes: int, next_open: float | None, c0: float | None = None) -> list[dict]:
    """연구(intraday.py)의 정규장 매수 규칙 재생: 09:30~11:00 ET에 봉 종가로 판단·체결, +3%·+5% 지정가는
    다음 봉부터 그날 마감까지 고가로 판정, 못 닿으면 다음 날 시가. 손절은 11:00까지 봉 종가가 매수 전 저가 아래."""
    if b.empty:
        return []
    o = float(b.Open.iloc[0])
    mins = _minutes(b.index)
    end = mins + minutes
    vw = _vwap(b)
    run_low = b.Low.cummin()
    orb = b[mins < 585].High.max()
    entries = {"open": (-1, o, None)}
    for i in range(len(b)):
        if end[i] > WINDOW_END:
            break
        c = float(b.Close.iloc[i])
        if "t1000" not in entries and end[i] == 600:
            entries["t1000"] = (i, c, float(run_low.iloc[i]))
        if "rev2" not in entries and run_low.iloc[i] < o and c >= run_low.iloc[i] * 1.02:
            entries["rev2"] = (i, c, float(run_low.iloc[i]))
        if "vwap" not in entries and i > 0 and b.Close.iloc[i - 1] < vw.iloc[i - 1] and c > vw.iloc[i]:
            entries["vwap"] = (i, c, float(run_low.iloc[i]))
        if "orb15" not in entries and mins[i] >= 585 and c > orb:
            entries["orb15"] = (i, c, float(run_low.iloc[i]))
    bh = bottom_hold(b, c0, minutes) if c0 else {"signal_i": None}
    if bh["signal_i"] is not None:
        i = bh["signal_i"]
        entries["hold15"] = (i, float(b.Close.iloc[i]), float(run_low.iloc[i]))
    names = {"open": "시가", "t1000": "10:00", "rev2": "저점+2% 반등", "vwap": "VWAP 회복", "orb15": "15분 고가 돌파",
             "hold15": "바닥 15분 유지"}
    out = []
    for key in names:
        if key not in entries:
            out.append({"rule": key, "name": names[key], "at": None})
            continue
        i0, px, low_before = entries[key]
        after = b.iloc[i0 + 1:]
        stop = px * (1 - OPEN_STOP / 100) if low_before is None else low_before
        row = {"rule": key, "name": names[key], "price": px, "entry_vs_open": _pct(px / o),
               "at": "09:30" if i0 < 0 else (b.index[i0] + pd.Timedelta(minutes=minutes)).strftime("%H:%M"),
               "close": _pct(b.Close.iloc[-1] / px), "next_open": _pct(next_open / px) if next_open else None}
        stopped = None
        for j in range(len(after)):
            if end[i0 + 1 + j] > WINDOW_END:
                break
            # 바닥 15분은 연구대로 매수 전 저가 -1%를 봉 저가가 닿으면, 나머지는 봉 종가가 매수 전 저가 아래면
            hit_stop = (after.Low.iloc[j] <= low_before * (1 - HOLD_STOP / 100)) if key == "hold15" else after.Close.iloc[j] < stop
            if hit_stop:
                stopped = (after.index[j] + pd.Timedelta(minutes=minutes)).strftime("%H:%M")
                break
        row["stop_at"] = stopped
        for tp in TPS:
            hit = after[after.High >= px * (1 + tp / 100)]
            # 매수·손절과 같이 봉이 끝나는 시각(그 일이 있었다고 알 수 있는 때)으로 적는다
            row[f"tp{int(tp)}_at"] = (hit.index[0] + pd.Timedelta(minutes=minutes)).strftime("%H:%M") if len(hit) else None
        out.append(row)
    return out


def hour_replay(day_bars: pd.DataFrame, s: dict, f: dict) -> list[dict]:
    """연구(hours.py)의 시각별 매수 재생: 조건이 맞은 시각에 그 1시간봉 종가로 사서 +5% 지정가(장외 봉은 종가,
    정규장 봉은 고가로 판정)를 그날 마감까지, 못 닿으면 다음 날 시가."""
    out = []
    c0 = f.get("c0")
    if not c0:
        return out
    m = _minutes(day_bars.index)
    for h in f.get("hours", []):
        if not h["cond"]:
            continue
        px = c0 * (1 + h["close_pct"] / 100)
        bar_end = (h["hour"] + 1) * 60 if h["hour"] < 9 else 570
        after = day_bars[(m >= bar_end) & (m < 960)]
        am = _minutes(after.index)
        hi = np.where(am < 570, after.Close, after.High)
        hit = np.nonzero(hi >= px * 1.05)[0]
        out.append({"at": h["at"], "price": px, "close_pct": h["close_pct"], "reb": h["reb"],
                    "tp5_at": after.index[hit[0]].strftime("%H:%M") if len(hit) else None,
                    "next_open": _pct(s["next_open"] / px) if s["next_open"] else None})
    return out


def news_rows(payload: dict) -> list[dict]:
    rows = (payload or {}).get("data", {}) or {}
    out = []
    for r in rows.get("rows", []) or []:
        url = r.get("url") or ""
        out.append({"title": r.get("title"), "publisher": r.get("publisher"), "when": r.get("created") or r.get("ago"),
                    "url": url if url.startswith("http") else "https://www.nasdaq.com" + url})
    return out


# ---------------------------------------------------------------- 네트워크
def _headers() -> dict:
    return {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def universe(force: bool = False) -> dict:
    """{기호: {name, cap, sector, industry}} — 시총 10억 달러 이상 미국 상장 보통주. 하루 캐시."""
    if not force and UNIVERSE_PATH.exists():
        cached = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
        if _time.time() - cached.get("_epoch", 0) < UNIVERSE_MAX_AGE_H * 3600:
            return cached["symbols"]
    import requests

    rows = requests.get(NASDAQ_LIST, headers=_headers(), timeout=60).json()["data"]["rows"]
    out = {}
    for r in rows:
        sym = (r.get("symbol") or "").strip()
        try:
            cap = float(r.get("marketCap") or 0)
        except ValueError:
            continue
        if re.fullmatch(r"[A-Z]{1,5}", sym) and cap >= MIN_CAP and not NOT_COMMON.search(r.get("name") or ""):
            out[sym] = {"name": r.get("name"), "cap": cap, "sector": r.get("sector"), "industry": r.get("industry")}
    config.CACHE.mkdir(parents=True, exist_ok=True)
    UNIVERSE_PATH.write_text(json.dumps({"_epoch": _time.time(), "symbols": out}, ensure_ascii=False), encoding="utf-8")
    return out


def quotes(symbols: list[str]) -> list[dict]:
    """Yahoo 시세 묶음 요청(1,000종목씩). yfinance의 인증(crumb) 처리를 그대로 쓴다."""
    from yfinance.data import YfData

    data, out = YfData(), []
    for i in range(0, len(symbols), 1000):
        res = data.get_raw_json(QUOTE_URL, params={"symbols": ",".join(symbols[i:i + 1000])})
        out += (res.get("quoteResponse") or {}).get("result") or []
    return out


def minute_bars(symbols: list[str], **kw) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    if not symbols:
        return {}
    frame = yf.download(symbols, interval="1m", prepost=True, auto_adjust=False, progress=False,
                        group_by="ticker", threads=True, **kw)
    out = {}
    for sym in symbols:
        try:
            b = frame[sym] if isinstance(frame.columns, pd.MultiIndex) else frame
        except KeyError:
            continue
        b = b[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "Close"])
        if len(b):
            b.index = b.index.tz_convert(config.NY)
            out[sym] = b
    return out


def kst_shift(day: date) -> int:
    """그날 뉴욕 시각에 더하면 한국 시각이 되는 시간: 서머타임 13, 표준시 14."""
    noon = datetime(day.year, day.month, day.day, 12, tzinfo=config.NY)
    return int((noon.astimezone(config.KST).utcoffset() - noon.utcoffset()).total_seconds() // 3600)


def _session_day(bars: pd.DataFrame) -> date | None:
    m = _minutes(bars.index)
    days = sorted(set(np.asarray(bars.index.date)[m >= 240]))
    return days[-1] if days else None


# ---------------------------------------------------------------- 화면 자료
def compute() -> dict:
    now = datetime.now(config.NY)
    uni = universe()
    rows, warnings = [], []
    spy = None
    try:
        quoted = quotes(sorted(uni) + ["SPY"])
    except Exception as exc:
        raise DipError(f"Yahoo 시세를 받지 못했습니다({type(exc).__name__}). 잠시 뒤 다시 시도하세요.") from exc
    for q in quoted:
        r = classify_quote(q)
        if not r:
            continue
        if r["symbol"] == "SPY":
            spy = r
            continue
        info = uni.get(r["symbol"], {})
        r["name"] = info.get("name") or r["name"]
        r["sector"], r["industry"] = info.get("sector"), info.get("industry")
        rows.append(r)
    cands = sorted([r for r in rows if r["chg"] is not None and r["chg"] <= -DROP], key=lambda r: r["chg"])
    bars = {}
    try:
        bars = minute_bars([r["symbol"] for r in cands[:MAX_DETAIL]], period="2d")
    except Exception as exc:  # 분봉이 안 와도 목록은 보인다
        warnings.append(f"분봉을 받지 못했습니다({type(exc).__name__})")
    for r in cands:
        b = bars.get(r["symbol"])
        day = _session_day(b) if b is not None else None
        if day:
            r.update(features(b, day, r["ref"], r["earnings"], now))
            r["day"] = day.isoformat()
    market = spy["chg"] if spy else None
    for r in cands:
        r["market"] = market is not None and market <= MARKET
    return {"computed_at": datetime.now(config.KST).isoformat(timespec="seconds"),
            "now_et": now.strftime("%Y-%m-%d %H:%M"), "dst": bool(now.dst()), "kst_shift": kst_shift(now.date()),
            "session": spy["session"] if spy else None, "spy": spy, "scanned": len(rows),
            "rows": cands, "warnings": warnings,
            "rules": {"drop": DROP, "deep": DEEP, "min_cap": MIN_CAP, "max_detail": MAX_DETAIL, "rebound_08": REBOUND_08,
                      "band_0929": BAND_0929, "ah_news": AH_NEWS, "market": MARKET, "tps": TPS, "window_end": "11:00"},
            "research": research()}


def research() -> dict:
    return {"study": STUDY, "time": TIME_STATS, "time_08": TIME_REBOUND_08, "rebound": REBOUND_STATS,
            "limit": LIMIT_STATS, "hold": HOLD_STATS, "intraday_low": INTRADAY_LOW, "intraday": INTRADAY_STATS,
            "survival": SURVIVAL, "bottom": BOTTOM_STATS,
            "cases": CASES}


def payload(force: bool = False) -> dict:
    """급락 목록은 SCAN_MAX_AGE_SEC 동안 캐시한다(장 중에는 2분마다 새로 고쳐도 3~15초)."""
    with _LOCK:
        if not force and CACHE_PATH.exists():
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if _time.time() - cached.get("_epoch", 0) < SCAN_MAX_AGE_SEC:
                cached["research"] = research()
                return cached
        data = clean(compute())
        data["_epoch"] = _time.time()
        config.CACHE.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data


def _check_symbol(t: str) -> str:
    t = (t or "").strip().upper().replace(".", "-")
    if not re.fullmatch(r"[A-Z][A-Z\-]{0,9}", t):
        raise DipError("종목 기호를 영문으로 입력하세요 (예: STX).")
    return t


def detail(t: str, day: str | None = None, minutes: int = 3) -> dict:
    """한 종목·한 날의 3분봉과 기준선, 시각별 조건, 정규장 규칙 재생. 날짜가 없으면 가장 최근 거래일."""
    t = _check_symbol(t)
    now = datetime.now(config.NY)
    target = None
    if day:
        try:
            target = date.fromisoformat(day)
        except ValueError as exc:
            raise DipError("날짜는 2026-10-02처럼 입력하세요.") from exc
        if (now.date() - target).days > DETAIL_DAYS:
            raise DipError(f"Yahoo 1분봉은 최근 {DETAIL_DAYS + 1}일만 있어 {day} 자료는 다시 볼 수 없습니다.")
        if target > now.date():
            raise DipError("앞으로의 날짜는 볼 수 없습니다.")
    key = (t, (target or now.date()).isoformat())
    hit = _DETAIL.get(key)
    live = target is None or target == now.date()
    if hit and _time.time() - hit[0] < (60 if live else 6 * 3600):
        return hit[1]
    end_day = min((target or now.date()) + timedelta(days=4), now.date() + timedelta(days=1))
    start_day = (target or now.date()) - timedelta(days=4)
    bars = minute_bars([t], start=start_day.isoformat(), end=end_day.isoformat()).get(t)
    if bars is None:
        raise DipError(f"{t}의 1분봉을 받지 못했습니다. 기호를 확인하세요.")
    if target is None:
        target = _session_day(bars)
    s = split_sessions(bars, target)
    if s["pm"].empty and s["reg"].empty:
        raise DipError(f"{t}의 {target} 거래 자료가 없습니다(휴장일이거나 상장 전).")
    q = None
    try:
        q = classify_quote(next(iter(quotes([t])), {})) if live else None
    except Exception:
        q = None
    earnings = bool(q and q.get("earnings"))
    f = features(bars, target, q["ref"] if q else None, earnings, now if live else None)
    day_bars = pd.concat([s["pm"], s["reg"], s["post"]]).sort_index()
    b3 = resample(day_bars, minutes)
    m3 = _minutes(b3.index)
    reg3 = b3[(m3 >= 570) & (m3 < 960)]
    if live:                                           # 진행 중인 봉은 규칙 판단에서 뺀다(차트에는 그린다)
        reg3 = closed_bars(reg3, minutes, now.hour * 60 + now.minute)
    vw = _vwap(s["reg"]).resample(f"{minutes}min", origin="start_day").last() if len(s["reg"]) else pd.Series(dtype=float)
    c0 = f.get("c0")
    candles = [{"t": ts.strftime("%H:%M"), "o": float(r.Open), "h": float(r.High), "l": float(r.Low), "c": float(r.Close),
                "v": float(r.Volume), "ses": "pm" if mm < 570 else "reg" if mm < 960 else "post",
                "vwap": float(vw.get(ts)) if ts in vw.index and pd.notna(vw.get(ts)) else None}
               for (ts, r), mm in zip(b3.iterrows(), m3)]
    out = {"t": t, "day": target.isoformat(), "minutes": minutes, "c0": c0, "live": live, "earnings": earnings,
           "kst_shift": kst_shift(target),
           "name": (q or {}).get("name") or t, "candles": candles, "features": f,
           "hour_replay": hour_replay(day_bars, s, f),
           "rules": intraday_rules(reg3, minutes, s["next_open"], c0) if len(reg3) else [],
           "next_open": s["next_open"], "now_et": now.strftime("%H:%M") if live else None}
    out = clean(out)
    if len(_DETAIL) > 200:
        _DETAIL.clear()
    _DETAIL[key] = (_time.time(), out)
    return out


def news(t: str) -> list[dict]:
    import requests

    t = _check_symbol(t)
    try:
        res = requests.get(NASDAQ_NEWS.format(sym=t.replace("-", ".")), headers=_headers(), timeout=20).json()
    except Exception as exc:
        raise DipError(f"뉴스를 받지 못했습니다({type(exc).__name__}).") from exc
    return news_rows(res)
