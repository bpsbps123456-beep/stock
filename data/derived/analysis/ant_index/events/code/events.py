# -*- coding: utf-8 -*-
"""개미 지수 부록 13 — 사건 목록(가격만).

가격으로만 고점·저점·가짜 사건을 정한다. 포럼 특징은 읽지 않는다(창 표에서는 win_id·D·variant 세 칸만 읽는다).
가격은 이 사건 목록을 만드는 데만 쓴다.

정의(HYPOTHESES_addendum13.md '사건'):
- 꺾임점: 일별 종가 지그재그. 직전 극값에서 반대 방향으로 X% 움직이면 그 극값을 고점·저점으로 확정한다.
  주판 X = 8%, 강건판 X = 12%. 고점 확정은 종가 <= 고점 x (1 - X), 저점 확정은 종가 >= 저점 x (1 + X)(경계 포함).
- 대상: KOSPI(^KS11, 국내 창 W_KR), SK하이닉스(000660.KS, W_KR), SOXX(미국 창 W_U).
- 기간: 2025-08-21 ~ 2026-10-23. 지그재그는 2025-01-02부터 돌려 방향을 미리 정한다(예열). 극값 날짜가 기간 안인 것만 쓴다.
- 가짜 바닥: 하루 -5% 이하이거나 5일 낙폭 -8% 이하(종가/5거래일 전 종가 - 1)인 날 중, 확정 저점 ±3거래일 밖인 날.
- 가짜 고점: 종가가 20일(오늘 포함) 최고인 날 중, 확정 고점 ±3거래일 밖인 날.
- 끝 극값(아직 X% 반대 움직임이 없는 것)은 '미확정'으로 따로 둔다. 10/23 종가가 들어오면 rejudge_tail()로 다시 판정한다.

구현 선택(정의에 없는 부분, 근거와 함께 events.json meta에 남긴다):
- 같은 종가가 이어지면 처음 날을 극값으로 둔다(새 극값은 엄격히 넘어야 바뀐다).
- 첫 방향이 정해지지 않은 구간에서는 시작일부터의 최고·최저를 함께 들고 가다가, 먼저 X%에 닿은 쪽을 확정한다.
  종가 하나로는 같은 날 두 쪽이 함께 닿을 수 없다(증명은 zigzag() 주석). 확정한 날이 곧 다음 극값 후보의 첫날이다.
- 미확정 끝 극값 ±3거래일 안의 가짜 후보는 가짜 목록에서 빼서 '보류'로 둔다(끝 극값이 확정되면 정의상 가짜가 아니다).

가격 자료:
- 주 자료는 yfinance 일봉 종가(auto_adjust=False의 Close)다.
- 2026-10-22 관찰: yfinance가 ^KS11·SOXX의 10/21 종가를 비우거나(NaN) 장중 막대를 붙이고, 000660.KS는 2025-09-19 막대가 없고
  네 날(2025-09-18, 2026-03-27, 2026-04-01, 2026-06-24)의 종가가 다음 거래일 기준가와 다르다. 그래서 네이버 시세로 모든 날을 대조한다.
  - KOSPI·SOXX: 네이버 일별 종가와 대조한다(2025-01~2026-10 전 구간 일치 확인).
  - SK하이닉스: 네이버 '다음 거래일 기준가'(다음 날 종가 - 전일 대비)와 대조한다. 기준가는 KRX 공식 전일 종가다.
    네이버의 일별 종가 칸은 시간외(NXT 애프터마켓, 20:00) 가격이 들어간 날이 있어(2026-10-22 basic API의 overMarketPriceInfo로 확인)
    2026-09-14 뒤 여러 날 기준가와 다르다. 그래서 그 칸은 쓰지 않는다.
  - 어긋나거나 빈 날만 네이버 값으로 바꾸고, 바꾼 날과 두 값을 prices_fetch_meta.json·events.json에 적는다.
- 10/22·10/23은 시장 마감 뒤(KRX 18:00 KST, NYSE 17:00 ET) 확정된 종가만 넣는다. 없으면 비워 둔다.

실행(이 폴더에서):
  py events.py fetch   # yfinance + 네이버로 종가 스냅샷(results/prices_close.csv)을 새로 만든다
  py events.py build   # 스냅샷으로 results/events.json을 만든다
  py events.py all     # fetch 다음 build. 10/23 종가 뒤(10/24 06:00 KST 이후) 다시 돌리면 끝 극값을 다시 판정한다
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import snap_paths as SP  # noqa: E402

HERE = Path(__file__).resolve().parent
EVENTS = HERE.parent
RESULTS = EVENTS / "results"
ANT = EVENTS.parent
FINAL_RESULTS = SP.SNAP_RES        # 부록 14: 창 표·달력은 사건 연구 스냅샷(2026-10-23 07:59)에서 읽는다(10/21까지 final과 같음)
ADDENDUM = ANT / "HYPOTHESES_addendum13.md"

PRICES_CSV = RESULTS / "prices_close.csv"
FETCH_META = RESULTS / "prices_fetch_meta.json"
EVENTS_JSON = RESULTS / "events.json"

FETCH_START = "2025-01-01"
CUTOFF = "2026-10-21"                     # 달력 대조 끝(final 스냅샷 마감). 10/22·10/23은 POST_CUTOFF 칸으로 채운다
POST_CUTOFF = ("2026-10-22", "2026-10-23")
EVENT_START = "2025-08-21"
EVENT_END = "2026-10-23"
FOCUS_LOW = "2026-10-15"

THRESHOLDS = (("8", 0.08, "주판"), ("12", 0.12, "강건판"))
FAKE_BOTTOM_1D = -0.05
FAKE_BOTTOM_5D = -0.08
FAKE_BOTTOM_LAG = 5
FAKE_TOP_LOOKBACK = 20
EXCLUDE_TD = 3
TOL = 1e-12          # 경계(정확히 X%)를 포함하려고 두는 부동소수 여유
MATCH_REL = 1e-6     # 두 출처 종가가 같다고 보는 상대 오차

ASSETS: dict[str, dict] = {
    "KOSPI": dict(ticker="^KS11", market="krx", window="W_KR", win_file="windows_kr.parquet",
                  naver_url="https://m.stock.naver.com/api/index/KOSPI/price", verify="close", decimals=2),
    "SKHYNIX": dict(ticker="000660.KS", market="krx", window="W_KR", win_file="windows_kr.parquet",
                    naver_url="https://m.stock.naver.com/api/stock/000660/price", verify="ref_next", decimals=0),
    "SOXX": dict(ticker="SOXX", market="nyse", window="W_U", win_file="windows_us.parquet",
                 naver_url="https://api.stock.naver.com/etf/SOXX.O/price", verify="close", decimals=2),
}
MARKET_TZ = {"krx": "Asia/Seoul", "nyse": "America/New_York"}
FINAL_AFTER = {"krx": (18, 0), "nyse": (17, 0)}   # 이 현지 시각이 지나야 그날 종가를 확정으로 본다
CAL_COL = {"krx": "krx", "nyse": "nyse"}


# ---------------------------------------------------------------------------
# 지그재그
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Pivot:
    kind: str          # "high" | "low"
    idx: int           # 극값 날의 위치
    confirm_idx: int   # 반대 방향 X%에 처음 닿은 날의 위치
    value: float


@dataclass(frozen=True)
class Tail:
    """끝에 남은 미확정 극값. 방향이 아직 정해지지 않았으면 kind=None이고 hi_idx·lo_idx를 함께 둔다."""
    kind: str | None
    idx: int | None
    value: float | None
    hi_idx: int | None = None
    lo_idx: int | None = None


def _rose(v: float, base: float, x: float) -> bool:
    return v / base - 1.0 >= x - TOL


def _fell(v: float, base: float, x: float) -> bool:
    return v / base - 1.0 <= -x + TOL


def zigzag(closes, x: float) -> tuple[list[Pivot], Tail]:
    """일별 종가 지그재그. 확정 꺾임점 목록과 끝의 미확정 극값을 돌려준다.

    - 방향 미정(처음): 시작일부터의 최고 hi·최저 lo를 함께 든다. 오늘 종가가 lo에서 X% 오르면 lo를 저점으로,
      hi에서 X% 내리면 hi를 고점으로 확정한다. 두 쪽이 같은 날 함께 닿는 일은 없다:
      hi가 lo보다 늦게 생겼다면 hi가 생긴 날 이미 lo에서 X% 올라 있어야 하고(그날 확정됐어야 함),
      lo가 늦게 생겼다면 lo가 생긴 날 이미 hi에서 X% 내려 있어야 한다. 그래서 둘 다 닿으면 구현 오류로 본다.
    - 상승 중: 새 최고면 고점 후보를 옮기고, 후보에서 X% 내리면 후보를 고점으로 확정한다. 확정한 날이 새 저점 후보다.
    - 하락 중: 반대로 한다. 한 날에는 꺾임점이 최대 하나 확정된다.
    - 같은 종가가 이어지면 처음 날을 후보로 둔다(엄격히 넘어야 옮긴다).
    """
    c = [float(v) for v in closes]
    for v in c:
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"종가가 양의 유한수가 아니다: {v}")
    n = len(c)
    if n == 0:
        return [], Tail(None, None, None)
    pivots: list[Pivot] = []
    hi = lo = 0
    trend = 0   # 0 미정, +1 상승 중(고점 후보 hi), -1 하락 중(저점 후보 lo)
    for t in range(1, n):
        v = c[t]
        if trend == 0:
            up = _rose(v, c[lo], x)
            down = _fell(v, c[hi], x)
            if up and down:
                raise AssertionError(f"방향 미정 구간에서 같은 날 두 쪽이 함께 닿음(t={t}) — 종가 하나로는 불가능")
            if up:
                pivots.append(Pivot("low", lo, t, c[lo]))
                trend, hi = +1, t
            elif down:
                pivots.append(Pivot("high", hi, t, c[hi]))
                trend, lo = -1, t
            else:
                if v > c[hi]:
                    hi = t
                if v < c[lo]:
                    lo = t
        elif trend == +1:
            if v > c[hi]:
                hi = t
            elif _fell(v, c[hi], x):
                pivots.append(Pivot("high", hi, t, c[hi]))
                trend, lo = -1, t
        else:
            if v < c[lo]:
                lo = t
            elif _rose(v, c[lo], x):
                pivots.append(Pivot("low", lo, t, c[lo]))
                trend, hi = +1, t
    if trend == +1:
        tail = Tail("high", hi, c[hi])
    elif trend == -1:
        tail = Tail("low", lo, c[lo])
    else:
        tail = Tail(None, None, None, hi_idx=hi, lo_idx=lo)
    return pivots, tail


# ---------------------------------------------------------------------------
# 가짜 사건
# ---------------------------------------------------------------------------
def _near(t: int, anchors: list[int], radius: int = EXCLUDE_TD) -> int | None:
    """anchors 중 t에서 가장 가까운 것까지의 거래일 거리(radius 안이면 그 거리, 아니면 None)."""
    best = None
    for a in anchors:
        d = abs(t - a)
        if d <= radius and (best is None or d < best):
            best = d
    return best


def _nearest(t: int, anchors: list[int]) -> int | None:
    return min((abs(t - a) for a in anchors), default=None)


def fake_bottom_candidates(closes) -> list[dict]:
    """하루 -5% 이하이거나 5일 -8% 이하인 날(위치·수익률·걸린 조건)."""
    c = np.asarray(closes, dtype=float)
    out = []
    for t in range(1, len(c)):
        r1 = c[t] / c[t - 1] - 1.0
        r5 = c[t] / c[t - FAKE_BOTTOM_LAG] - 1.0 if t >= FAKE_BOTTOM_LAG else None
        hit1 = r1 <= FAKE_BOTTOM_1D + TOL
        hit5 = r5 is not None and r5 <= FAKE_BOTTOM_5D + TOL
        if hit1 or hit5:
            out.append({"idx": t, "r1": r1, "r5": r5,
                        "trigger": "+".join(k for k, h in (("1d", hit1), ("5d", hit5)) if h)})
    return out


def fake_bottom_candidates_alt(closes) -> list[int]:
    """민감도(사용 안 함): '5일 낙폭'을 직전 5거래일 최고 종가 대비 하락으로 읽은 판."""
    c = np.asarray(closes, dtype=float)
    out = []
    for t in range(1, len(c)):
        r1 = c[t] / c[t - 1] - 1.0
        dd = c[t] / c[max(0, t - FAKE_BOTTOM_LAG):t].max() - 1.0
        if r1 <= FAKE_BOTTOM_1D + TOL or (t >= FAKE_BOTTOM_LAG and dd <= FAKE_BOTTOM_5D + TOL):
            out.append(t)
    return out


def fake_top_candidates(closes) -> list[dict]:
    """종가가 20일(오늘 포함) 최고인 날. 같은 값이면 최고로 본다. 앞 19거래일이 있어야 한다."""
    c = np.asarray(closes, dtype=float)
    out = []
    for t in range(FAKE_TOP_LOOKBACK - 1, len(c)):
        prev_max = c[t - FAKE_TOP_LOOKBACK + 1:t].max()
        if c[t] >= prev_max:
            out.append({"idx": t, "prev19_max": float(prev_max)})
    return out


def split_fakes(cands: list[dict], confirmed: list[int], tail_idx: int | None) -> tuple[list[dict], list[dict], list[dict]]:
    """후보를 (가짜, 보류, 확정 극값 근처라 뺀 것)으로 나눈다.
    가짜 = 확정 극값 ±3거래일 밖이고 미확정 끝 극값 ±3거래일 밖. 보류 = 확정 극값 밖이지만 끝 극값 ±3 안."""
    fakes, held, dropped = [], [], []
    tails = [tail_idx] if tail_idx is not None else []
    for r in cands:
        t = r["idx"]
        dc = _near(t, confirmed)
        if dc is not None:
            dropped.append({**r, "dist_confirmed": dc})
            continue
        dt_ = _near(t, tails)
        if dt_ is not None:
            held.append({**r, "dist_tail": dt_})
            continue
        fakes.append(r)
    return fakes, held, dropped


def label_runs(idxs: list[int], gap: int = EXCLUDE_TD) -> list[int]:
    """참고용 묶음 번호: 앞 후보와 gap 거래일 이내면 같은 묶음."""
    runs, cur, prev = [], 0, None
    for t in idxs:
        if prev is not None and t - prev > gap:
            cur += 1
        runs.append(cur)
        prev = t
    return runs


# ---------------------------------------------------------------------------
# 끝 극값(미확정) 판정 — 10/23 종가가 들어오면 다시 부른다
# ---------------------------------------------------------------------------
def rejudge_tail(dates: list[str], closes, x: float, deadline: str = EVENT_END, focus: str = FOCUS_LOW) -> dict:
    """마감일까지의 종가만으로 끝 극값과 focus(10/15) 저점의 상태를 판정한다.

    - 마감일 뒤 종가는 버린다(마감 뒤 반등으로는 확정하지 않는다).
    - focus 날이 마감일까지 확정 저점이 되었으면 '확정', 아니면 '미확정'.
    - 끝 극값은 정의상 '미확정'이다. 마감일 종가가 아직 없으면 stage='판정 대기'.
    """
    keep = [i for i, d in enumerate(dates) if d <= deadline]
    d = [dates[i] for i in keep]
    c = [float(closes[i]) for i in keep]
    pivots, tail = zigzag(c, x)
    have_deadline = bool(d) and d[-1] == deadline
    out: dict = {
        "x": x, "deadline": deadline, "data_through": d[-1] if d else None,
        "deadline_close_in": have_deadline,
        "stage": "마감 뒤 판정 끝" if have_deadline else f"판정 대기({deadline} 종가 전)",
    }
    if tail.kind is None:
        out["tail"] = {"kind": None, "status": "미확정", "note": "방향 미정"}
    else:
        need = tail.value * (1 + x) if tail.kind == "low" else tail.value * (1 - x)
        last = c[-1]
        out["tail"] = {
            "kind": tail.kind, "date": d[tail.idx], "close": tail.value, "status": "미확정",
            "confirm_level": need, "last_close": last, "move_from_tail": last / tail.value - 1.0,
            "td_since_tail": len(c) - 1 - tail.idx,
            "same_close_later": [d[i] for i in range(tail.idx + 1, len(c)) if c[i] == tail.value],
        }
    if focus in d:
        fi = d.index(focus)
        conf = next((p for p in pivots if p.idx == fi and p.kind == "low"), None)
        undercut = [d[i] for i in range(fi + 1, len(c)) if c[i] < c[fi]]
        f = {"date": focus, "close": c[fi], "status": "확정" if conf else "미확정",
             "confirm_date": d[conf.confirm_idx] if conf else None,
             "is_tail_low": tail.kind == "low" and tail.idx == fi,
             "undercut_on": undercut}
        if conf:
            f["note"] = f"{d[conf.confirm_idx]} 종가로 {x:.0%} 반등 확정"
        elif f["is_tail_low"]:
            f["note"] = f"끝 저점 후보가 {focus}다. {x:.0%} 반등(종가 {c[fi] * (1 + x):,.2f} 이상)이 {deadline}까지 없으면 미확정으로 둔다"
        elif undercut:
            f["note"] = (f"{focus} 종가가 {', '.join(undercut)}에 더 낮은 종가로 깨져 {focus}는 저점 후보가 아니다. "
                         f"끝 극값 후보는 {out['tail'].get('date')}({out['tail'].get('kind')})")
        else:
            f["note"] = f"{focus}는 지금 하락 구간의 최저 종가가 아니다. 끝 극값 후보는 {out['tail'].get('date')}({out['tail'].get('kind')})"
        out["focus"] = f
    else:
        out["focus"] = {"date": focus, "status": "자료 없음"}
    return out


# ---------------------------------------------------------------------------
# 가격 받기·대조
# ---------------------------------------------------------------------------
def is_final(market: str, day: str, now: datetime) -> bool:
    tz = ZoneInfo(MARKET_TZ[market])
    h, m = FINAL_AFTER[market]
    y, mo, d = (int(s) for s in day.split("-"))
    return now >= datetime(y, mo, d, h, m, tzinfo=tz)


def expected_dates(market: str, calendar: pd.DataFrame, extra_present: set[str], now: datetime) -> tuple[list[str], dict]:
    """로더 달력(2025-01-01~CUTOFF)의 거래일 + 마감 뒤 확정된 10/22·10/23(어느 출처에든 있을 때)."""
    col = CAL_COL[market]
    cal = calendar.loc[calendar[col] & (calendar["date"] >= pd.Timestamp(FETCH_START))
                       & (calendar["date"] <= pd.Timestamp(CUTOFF)), "date"]
    days = [pd.Timestamp(v).strftime("%Y-%m-%d") for v in cal]
    post = {}
    for d in POST_CUTOFF:
        if not is_final(market, d, now):
            post[d] = "마감 전(비워 둠)"
        elif d in extra_present:
            days.append(d)
            post[d] = "확정 종가 있음"
        else:
            post[d] = "마감 뒤인데 두 출처 모두 없음(비워 둠)"
    return days, post


def _num(s):
    if s is None:
        return None
    s = str(s).replace(",", "").replace("+", "").strip()
    if s in ("", "-", "N/A"):
        return None
    return float(s)


def naver_history(url: str, start: str, session=None, pause: float = 0.3) -> pd.DataFrame:
    """네이버 일별 시세. ref_next = 다음 거래일의 (종가 - 전일 대비) = 이날 공식 종가(다음 날 기준가)."""
    import requests
    s = session or requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1",
               "Referer": "https://m.stock.naver.com/", "Accept": "application/json"}
    rows, page = [], 1
    while page <= 40:
        js, last = None, None
        for i in range(5):
            try:
                r = s.get(f"{url}?pageSize=60&page={page}", headers=headers, timeout=60)
                if r.status_code == 200 and r.text[:1] in "[{":
                    js = r.json()
                    break
                last = f"status={r.status_code}"
            except Exception as exc:  # noqa: BLE001
                last = str(exc)[:120]
            time.sleep(2 * (i + 1))
        if js is None:
            raise RuntimeError(f"네이버 응답 실패: {url} page={page} {last}")
        if not isinstance(js, list) or not js:
            break
        for r in js:
            rows.append({"date": str(r["localTradedAt"])[:10], "close": _num(r.get("closePrice")),
                         "chg": _num(r.get("compareToPreviousClosePrice"))})
        if rows[-1]["date"] < start or len(js) < 60:
            break
        page += 1
        time.sleep(pause)
    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return add_ref_next(df)


def add_ref_next(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ref"] = df["close"] - df["chg"]
    df["ref_next"] = df["ref"].shift(-1)
    df["next_date"] = df["date"].shift(-1)
    return df


def yf_daily(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    end = (date.fromisoformat(EVENT_END) + timedelta(days=1)).isoformat()
    d = yf.download(ticker, start=FETCH_START, end=end, auto_adjust=False, progress=False,
                    multi_level_index=False, actions=False, threads=False)
    if d is None or d.empty:
        raise RuntimeError(f"yfinance 빈 응답: {ticker}")
    out = pd.DataFrame({"date": pd.to_datetime(d.index).strftime("%Y-%m-%d"),
                        "yf_close": d["Close"].astype(float).to_numpy(),
                        "yf_adj": d["Adj Close"].astype(float).to_numpy()})
    return out.drop_duplicates("date", keep="last").reset_index(drop=True)


def _get(df: pd.DataFrame, d: str, col: str):
    s = df.loc[df["date"] == d, col]
    if s.empty:
        return None
    v = s.iloc[0]
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v) else float(v)


def reconcile(asset: str, cfg: dict, days: list[str], yfd: pd.DataFrame, nv: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """날마다 종가 하나를 정한다. yfinance가 주 자료이고, 네이버 대조값과 어긋나거나 비면 네이버 대조값을 쓴다."""
    rows, repairs = [], []
    for j, d in enumerate(days):
        yc, ya = _get(yfd, d, "yf_close"), _get(yfd, d, "yf_adj")
        nc, nr = _get(nv, d, "close"), _get(nv, d, "ref_next")
        if nr is not None:
            nd = nv.loc[nv["date"] == d, "next_date"].iloc[0]
            # 네이버에 빠진 날이 있으면 다음 행의 기준가는 이날 종가가 아니다 → 대조값으로 쓰지 않는다
            if j + 1 < len(days) and nd != days[j + 1]:
                nr = None
        ref = nc if cfg["verify"] == "close" else nr
        if yc is not None and ref is not None:
            if abs(yc / ref - 1.0) <= MATCH_REL:
                close, src, note = round(yc, cfg["decimals"]), "yfinance", "네이버와 일치"
            else:
                close, src = ref, f"naver_{cfg['verify']}"
                note = f"yfinance {yc:,.4f}가 네이버 대조값 {ref:,.4f}와 다름 → 네이버 값"
                repairs.append({"asset": asset, "date": d, "yf_close": yc, "naver_check": ref, "used": ref, "reason": "불일치"})
        elif yc is not None:
            close, src, note = round(yc, cfg["decimals"]), "yfinance_unverified", "네이버 대조값 없음(마지막 날 등)"
        elif ref is not None:
            close, src = ref, f"naver_{cfg['verify']}"
            note = "yfinance 종가 없음 → 네이버 값"
            repairs.append({"asset": asset, "date": d, "yf_close": None, "naver_check": ref, "used": ref, "reason": "yfinance 없음"})
        elif d in POST_CUTOFF:
            repairs.append({"asset": asset, "date": d, "yf_close": None, "naver_check": None, "used": None,
                            "reason": "마감 뒤 날인데 공식 종가를 정할 수 없음 → 비워 둠"})
            continue
        else:
            raise ValueError(f"{asset} {d}: 두 출처 모두 종가 없음")
        rows.append({"asset": asset, "date": d, "close": close, "source": src, "yf_close": yc, "yf_adj": ya,
                     "naver_close": nc, "naver_ref_next": nr, "note": note})
    return pd.DataFrame(rows), repairs


def fetch(now: datetime | None = None) -> dict:
    """yfinance + 네이버로 종가 스냅샷을 만든다."""
    import requests
    now = now or datetime.now(ZoneInfo("Asia/Seoul"))
    cal = pd.read_parquet(FINAL_RESULTS / "loader_calendar.parquet")
    session = requests.Session()
    frames, meta_assets, all_repairs = [], {}, []
    for asset, cfg in ASSETS.items():
        yfd = yf_daily(cfg["ticker"])
        nv = naver_history(cfg["naver_url"], FETCH_START, session)
        present = set(yfd.loc[yfd["yf_close"].notna(), "date"]) | set(nv.loc[nv["close"].notna(), "date"])
        days, post = expected_dates(cfg["market"], cal, present, now)
        df, repairs = reconcile(asset, cfg, days, yfd, nv)
        frames.append(df)
        all_repairs += repairs
        yf_dates = set(yfd["date"])
        info = {
            "ticker": cfg["ticker"], "naver_url": cfg["naver_url"], "verify": cfg["verify"],
            "rows": len(df), "first": df["date"].iloc[0], "last": df["date"].iloc[-1], "post_cutoff": post,
            "yf_missing_or_nan": sorted(set(days) - set(yfd.loc[yfd["yf_close"].notna(), "date"])),
            "yf_extra_ignored": sorted(d for d in yf_dates - set(days)),
            "source_counts": df["source"].value_counts().to_dict(),
        }
        if cfg["verify"] == "ref_next":
            both = df.dropna(subset=["naver_close"])
            diff = both.loc[(both["naver_close"] / both["close"] - 1).abs() > MATCH_REL, "date"].tolist()
            info["naver_close_col_differs_from_official"] = {
                "n": len(diff), "dates": diff,
                "note": "네이버 일별 '종가' 칸이 다음 날 기준가(KRX 공식 종가)와 다른 날. 그 칸은 쓰지 않는다"}
        meta_assets[asset] = info
    prices = pd.concat(frames, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PRICES_CSV, index=False, encoding="utf-8")
    meta = {
        "made": now.isoformat(timespec="seconds"),
        "purpose": "부록 13 사건 목록(가격만) 재현용 종가 스냅샷. 포럼 특징과 같은 표에 넣지 않는다.",
        "fetch_start": FETCH_START, "requested_end": CUTOFF, "post_cutoff_slots": list(POST_CUTOFF),
        "final_after_local": {k: f"{h:02d}:{m:02d} {MARKET_TZ[k]}" for k, (h, m) in FINAL_AFTER.items()},
        "calendar": "snapshot_1023/results/loader_calendar.parquet (~10/21 KRX=^KS11, NYSE=^GSPC 거래일, 10/22~ 거래소 달력)",
        "assets": meta_assets, "repairs": all_repairs,
        "sha256": hashlib.sha256(PRICES_CSV.read_bytes()).hexdigest(),
    }
    FETCH_META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# 사건 목록 만들기
# ---------------------------------------------------------------------------
def load_windows(win_file: str) -> dict[str, str]:
    """창 표에서 날짜→창 id만 읽는다(포럼 칸은 읽지 않는다). main 변형만."""
    w = pd.read_parquet(FINAL_RESULTS / win_file, columns=["win_id", "variant", "D"])
    w = w[w["variant"] == "main"]
    return {pd.Timestamp(d).strftime("%Y-%m-%d"): wid for d, wid in zip(w["D"], w["win_id"])}


def _r(v, k=6):
    return None if v is None else round(float(v), k)


def windows_before(winmap: dict[str, str], day: str) -> int:
    """사건일 앞에 있는 창 수(달력만 센다). 첫 창은 EVENT_START(2025-08-21)이다 — 그 앞 창은 수집 전이라 비어 있다."""
    return sum(1 for d in winmap if EVENT_START <= d < day)


def asset_events(dates: list[str], closes: list[float], x: float, winmap: dict[str, str]) -> dict:
    """한 자산·한 문턱의 사건 목록."""
    pivots, tail = zigzag(closes, x)
    in_period = [k for k, p in enumerate(pivots) if EVENT_START <= dates[p.idx] <= EVENT_END]
    rows = []
    for k in in_period:
        p = pivots[k]
        prev = pivots[k - 1] if k > 0 else None
        nxt = pivots[k + 1] if k + 1 < len(pivots) else None
        row = {
            "kind": p.kind, "date": dates[p.idx], "close": p.value, "win_id": winmap.get(dates[p.idx]),
            "n_windows_before": windows_before(winmap, dates[p.idx]),
            "confirm_date": dates[p.confirm_idx], "confirm_lag_td": p.confirm_idx - p.idx,
            "confirm_close": closes[p.confirm_idx],
            "prev_pivot_date": dates[prev.idx] if prev else None,
            "amp_in": _r(p.value / prev.value - 1) if prev else None,
            "len_in_td": p.idx - prev.idx if prev else None,
            "next_pivot_date": dates[nxt.idx] if nxt else None,
            "amp_out": _r(nxt.value / p.value - 1) if nxt else None,
            "len_out_td": nxt.idx - p.idx if nxt else None,
        }
        if nxt is None and tail.kind is not None:
            row["amp_out_to_tail_partial"] = _r(tail.value / p.value - 1)
            row["tail_date"] = dates[tail.idx]
        # 지그재그 특성: 저점 확정(+X%) 뒤 고점 후보에서 X%보다 덜 빠져도 저점 아래로 내려갈 수 있다(8%면 7.41~8% 하락).
        # 그런 날은 새 꺾임점이 아니므로 표시만 한다.
        end = nxt.idx if nxt else len(closes) - 1
        row["breached_after_confirm"] = [dates[i] for i in range(p.confirm_idx + 1, end + 1)
                                         if (closes[i] < p.value if p.kind == "low" else closes[i] > p.value)]
        rows.append(row)
    conf_low = [p.idx for p in pivots if p.kind == "low"]
    conf_high = [p.idx for p in pivots if p.kind == "high"]
    tail_low = tail.idx if tail.kind == "low" else None
    tail_high = tail.idx if tail.kind == "high" else None
    leg_start = pivots[-1].idx if pivots else -1   # 마지막 확정 꺾임점 뒤 = 끝이 아직 확정되지 않은 구간

    def in_win(r):
        return EVENT_START <= dates[r["idx"]] <= EVENT_END

    fb_c = [r for r in fake_bottom_candidates(closes) if in_win(r)]
    ft_c = [r for r in fake_top_candidates(closes) if in_win(r)]
    fb, fb_held, fb_drop = split_fakes(fb_c, conf_low, tail_low)
    ft, ft_held, ft_drop = split_fakes(ft_c, conf_high, tail_high)

    def fmt_fb(rs, runs=None):
        out = []
        for j, r in enumerate(rs):
            o = {"date": dates[r["idx"]], "close": closes[r["idx"]], "win_id": winmap.get(dates[r["idx"]]),
                 "n_windows_before": windows_before(winmap, dates[r["idx"]]),
                 "r1": _r(r["r1"]), "r5": _r(r["r5"]), "trigger": r["trigger"],
                 "nearest_confirmed_low_td": _nearest(r["idx"], conf_low),
                 "in_unconfirmed_leg": r["idx"] > leg_start}
            if runs is not None:
                o["run"] = runs[j]
            for key in ("dist_tail", "dist_confirmed"):
                if key in r:
                    o[key] = r[key]
            out.append(o)
        return out

    def fmt_ft(rs, runs=None):
        out = []
        for j, r in enumerate(rs):
            o = {"date": dates[r["idx"]], "close": closes[r["idx"]], "win_id": winmap.get(dates[r["idx"]]),
                 "n_windows_before": windows_before(winmap, dates[r["idx"]]),
                 "over_prev19_max": _r(closes[r["idx"]] / r["prev19_max"] - 1),
                 "nearest_confirmed_high_td": _nearest(r["idx"], conf_high),
                 "in_unconfirmed_leg": r["idx"] > leg_start}
            if runs is not None:
                o["run"] = runs[j]
            for key in ("dist_tail", "dist_confirmed"):
                if key in r:
                    o[key] = r[key]
            out.append(o)
        return out

    alt = [t for t in fake_bottom_candidates_alt(closes) if EVENT_START <= dates[t] <= EVENT_END
           and _near(t, conf_low) is None and (tail_low is None or _near(t, [tail_low]) is None)]
    prim = [r["idx"] for r in fb]
    tail_out = None
    if tail.kind is not None:
        tail_out = {"kind": tail.kind, "date": dates[tail.idx], "close": tail.value, "status": "미확정",
                    "win_id": winmap.get(dates[tail.idx]),
                    "confirm_level": tail.value * (1 + x) if tail.kind == "low" else tail.value * (1 - x),
                    "last_date": dates[-1], "last_close": closes[-1],
                    "move_from_tail": _r(closes[-1] / tail.value - 1),
                    "amp_in": _r(tail.value / pivots[-1].value - 1) if pivots else None,
                    "prev_pivot_date": dates[pivots[-1].idx] if pivots else None,
                    "same_close_later": [dates[i] for i in range(tail.idx + 1, len(closes)) if closes[i] == tail.value],
                    "tie_rule": "같은 종가면 처음 날"}
    return {
        "x": x,
        "tops": [r["date"] for r in rows if r["kind"] == "high"],
        "bottoms": [r["date"] for r in rows if r["kind"] == "low"],
        "pivots": rows,
        "tail_unconfirmed": tail_out,
        "warmup_pivots": [{"kind": p.kind, "date": dates[p.idx], "close": p.value, "confirm_date": dates[p.confirm_idx]}
                          for p in pivots if dates[p.idx] < EVENT_START],
        "fake_bottoms": fmt_fb(fb, label_runs([r["idx"] for r in fb])),
        "fake_tops": fmt_ft(ft, label_runs([r["idx"] for r in ft])),
        "held_near_tail": {"fake_bottoms": fmt_fb(fb_held), "fake_tops": fmt_ft(ft_held)},
        "excluded_near_confirmed": {"fake_bottoms": [dates[r["idx"]] for r in fb_drop],
                                    "fake_tops": [dates[r["idx"]] for r in ft_drop]},
        "sensitivity_fake_bottom_5d_from_max": {
            "rule": "5일 낙폭을 '직전 5거래일 최고 종가 대비'로 읽은 판(사용 안 함)",
            "n": len(alt), "added": [dates[t] for t in alt if t not in prim],
            "removed": [dates[t] for t in prim if t not in alt]},
        "counts": {"tops": sum(r["kind"] == "high" for r in rows), "bottoms": sum(r["kind"] == "low" for r in rows),
                   "fake_bottoms": len(fb), "fake_bottom_runs": len(set(label_runs(prim))),
                   "fake_tops": len(ft), "fake_top_runs": len(set(label_runs([r["idx"] for r in ft]))),
                   "held_fake_bottoms": len(fb_held), "held_fake_tops": len(ft_held)},
    }


def _pivot_dates(dates, closes, x):
    pv, _ = zigzag(closes, x)
    return [(p.kind, dates[p.idx]) for p in pv if EVENT_START <= dates[p.idx] <= EVENT_END]


def robustness(dates: list[str], closes: list[float], adj: list[float], raw_yf: list[tuple[str, float]], x: float) -> dict:
    """사건 날짜가 (a) 배당 조정 종가 (b) 고치기 전 yfinance (c) 예열 시작을 늦춘 경우에 바뀌는지."""
    base = _pivot_dates(dates, closes, x)
    out = {}
    out["adj_close"] = {"same": _pivot_dates(dates, adj, x) == base}
    if not out["adj_close"]["same"]:
        out["adj_close"]["alt"] = _pivot_dates(dates, adj, x)
    rd = [d for d, _ in raw_yf]
    rc = [v for _, v in raw_yf]
    alt = _pivot_dates(rd, rc, x)
    out["raw_yfinance"] = {"same": alt == base, "note": "yfinance 원값(빈 날 뺌, 틀린 날 그대로)"}
    if alt != base:
        out["raw_yfinance"]["only_in_raw"] = [a for a in alt if a not in base]
        out["raw_yfinance"]["only_in_used"] = [b for b in base if b not in alt]
    for start in ("2025-03-04", "2025-05-02", "2025-07-01"):
        i0 = next(i for i, d in enumerate(dates) if d >= start)
        alt = _pivot_dates(dates[i0:], closes[i0:], x)
        out[f"warmup_from_{start}"] = {"same": alt == base}
        if alt != base:
            out[f"warmup_from_{start}"]["diff"] = {"only_alt": [a for a in alt if a not in base],
                                                   "only_base": [b for b in base if b not in alt]}
    return out


def build() -> dict:
    if not PRICES_CSV.exists():
        raise FileNotFoundError(f"{PRICES_CSV} 없음 — 먼저 `py events.py fetch`")
    prices = pd.read_csv(PRICES_CSV, dtype={"date": str})
    fmeta = json.loads(FETCH_META.read_text(encoding="utf-8"))
    sha = hashlib.sha256(PRICES_CSV.read_bytes()).hexdigest()
    if sha != fmeta.get("sha256"):
        raise RuntimeError("prices_close.csv가 fetch 기록과 다르다")
    cal = pd.read_parquet(FINAL_RESULTS / "loader_calendar.parquet")
    out_assets, checks = {}, {}
    krx_dates = {}
    for asset, cfg in ASSETS.items():
        p = prices[prices["asset"] == asset].sort_values("date").reset_index(drop=True)
        p = p[p["date"] <= EVENT_END]
        dates = p["date"].tolist()
        closes = p["close"].astype(float).tolist()
        if len(set(dates)) != len(dates) or dates != sorted(dates):
            raise ValueError(f"{asset}: 날짜 중복·순서 오류")
        col = CAL_COL[cfg["market"]]
        cal_days = [pd.Timestamp(v).strftime("%Y-%m-%d") for v in
                    cal.loc[cal[col] & (cal["date"] >= pd.Timestamp(FETCH_START)) & (cal["date"] <= pd.Timestamp(CUTOFF)), "date"]]
        upto = [d for d in dates if d <= CUTOFF]
        checks[asset] = {"calendar_match_to_cutoff": upto == cal_days, "n_days": len(dates),
                         "first": dates[0], "last": dates[-1]}
        if upto != cal_days:
            raise ValueError(f"{asset}: 로더 달력과 거래일이 다르다")
        if cfg["market"] == "krx":
            krx_dates[asset] = dates
        winmap = load_windows(cfg["win_file"])
        f = p["yf_adj"] / p["yf_close"]
        factor = f.where(np.isfinite(f)).ffill().bfill().to_numpy()
        adj = (p["close"].to_numpy() * factor).tolist()
        raw = [(d, float(v)) for d, v in zip(p["date"], p["yf_close"]) if pd.notna(v)]
        th = {}
        for key, x, label in THRESHOLDS:
            ev = asset_events(dates, closes, x, winmap)
            ev["label"] = label
            ev["robustness"] = robustness(dates, closes, adj, raw, x)
            ev["rejudge"] = rejudge_tail(dates, closes, x)
            th[key] = ev
        post = {d: (float(p.loc[p["date"] == d, "close"].iloc[0]) if d in dates else None) for d in POST_CUTOFF}
        src = p.loc[p["source"] != "yfinance", ["date", "source", "close", "yf_close", "note"]]
        out_assets[asset] = {
            "ticker": cfg["ticker"], "window": cfg["window"], "calendar": cfg["market"].upper(),
            "data_through": dates[-1], "post_cutoff_closes": post,
            "price_rows_not_plain_yfinance": json.loads(src.to_json(orient="records", force_ascii=False)),
            "thresholds": th,
        }
    if len(krx_dates) == 2:
        a, b = krx_dates.values()
        checks["krx_assets_same_days"] = a == b
    snap = json.loads((SP.SNAP / "snapshot_manifest.json").read_text(encoding="utf-8"))
    focus_rows = []
    for asset in ASSETS:
        for key, _, _ in THRESHOLDS:
            rj = out_assets[asset]["thresholds"][key]["rejudge"]
            focus_rows.append({"asset": asset, "threshold": key, "focus_status": rj["focus"]["status"],
                               "focus_note": rj["focus"].get("note"), "tail_kind": rj["tail"].get("kind"),
                               "tail_date": rj["tail"].get("date"), "tail_status": rj["tail"]["status"],
                               "stage": rj["stage"]})
    result = {
        "meta": {
            "made": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
            "addendum": ADDENDUM.name,
            "scope": "사건 목록(가격만). 포럼 특징을 읽지 않고, 가격은 이 목록을 만드는 데만 쓴다.",
            "period": [EVENT_START, EVENT_END], "warmup_from": FETCH_START,
            "forum_snapshot": {"first": snap.get("first"), "cutoff": snap.get("cutoff"), "S": snap.get("S"),
                               "path": "events/snapshot_1023/snapshot_manifest.json"},
            "definitions": {
                "pivot": "일별 종가 지그재그. 고점 확정: 종가 <= 고점x(1-X). 저점 확정: 종가 >= 저점x(1+X). 경계 포함. X=8%(주판), 12%(강건판)",
                "first_direction": "처음엔 시작일부터의 최고·최저를 함께 들고, 먼저 X%에 닿은 쪽을 확정. 종가 하나로는 같은 날 두 쪽이 함께 닿지 않는다(코드에서 단언)",
                "same_day": "한 날에는 꺾임점이 최대 하나 확정된다. 확정한 날은 곧 다음 극값 후보의 첫날이다(그날이 다음 극값이 될 수 있다)",
                "ties": "같은 종가가 이어지면 처음 날을 극값으로 둔다",
                "event_dates": "극값 날짜가 기간 안(2025-08-21~2026-10-23)인 확정 꺾임점만 쓴다. 기간 앞 꺾임점은 진폭 기준과 ±3 제외에만 쓴다",
                "fake_bottom": "하루 수익률 <= -5% 또는 종가/5거래일 전 종가 - 1 <= -8%인 날 중, 확정 저점 ±3거래일 밖(거리 >= 4)",
                "fake_top": "종가 >= 직전 19거래일 최고 종가(오늘 포함 20일 최고)인 날 중, 확정 고점 ±3거래일 밖",
                "held": "미확정 끝 극값 ±3거래일 안의 가짜 후보는 보류(가짜 목록에서 뺌)",
                "run": "참고용 묶음 번호: 앞 후보와 3거래일 이내면 같은 묶음(정의에는 없음, 표본 독립성 확인용)",
                "amp": "amp_in = 이 극값/앞 꺾임점 - 1, amp_out = 다음 꺾임점/이 극값 - 1. len은 거래일 수",
                "confirm_lag": "극값 날부터 X% 반대 움직임에 처음 닿은 날까지 거래일 수(그 사이 포럼 값은 사후에야 사건임을 안다)",
                "win_id": "snapshot_1023/results 창 표(main)에서 D = 사건 거래일인 창. K창은 전 거래일 08:00~그날 08:00 KST, U창은 그날 21:30 KST 마감 — 사건일 장 전 정보다",
                "tail_rejudge": "끝 극값은 '미확정'. 10/23 종가가 들어오면 `py events.py all`로 rejudge_tail()을 다시 돈다. 10/23 뒤 반등은 확정에 쓰지 않는다",
            },
            "price_source": {
                "primary": "yfinance 일봉 Close(auto_adjust=False)",
                "verify": {"KOSPI": "네이버 지수 일별 종가", "SOXX": "네이버 해외 ETF 일별 종가",
                           "SKHYNIX": "네이버 다음 거래일 기준가(= KRX 공식 전일 종가)"},
                "repair_rule": "yfinance가 비거나 대조값과 상대 1e-6 넘게 다르면 네이버 대조값을 쓴다",
                "repairs": fmeta.get("repairs", []),
                "fetch_made": fmeta.get("made"), "prices_csv_sha256": sha,
                "fetch_assets": fmeta.get("assets"),
            },
            "checks": checks,
        },
        "focus_2026_10_15": focus_rows,
        "assets": out_assets,
    }
    EVENTS_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="부록 13 사건 목록(가격만)")
    ap.add_argument("cmd", nargs="?", default="build", choices=["fetch", "build", "all"])
    a = ap.parse_args(argv)
    if a.cmd in ("fetch", "all"):
        m = fetch()
        print("fetch:", {k: (v["rows"], v["last"], v["source_counts"]) for k, v in m["assets"].items()})
        print("repairs:", len(m["repairs"]))
    if a.cmd in ("build", "all"):
        r = build()
        for asset, v in r["assets"].items():
            for key, ev in v["thresholds"].items():
                print(asset, key, ev["counts"], "tail:", (ev["tail_unconfirmed"] or {}).get("date"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
