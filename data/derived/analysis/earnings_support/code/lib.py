"""공용 계산: 1시간봉 → 160분봉, 피벗, 지지·저항 구역(`kalman_zones/dip_scalper.pine` 그대로), 사건 창.

- 160분봉: 정규장만. 근사는 1시간봉 하루 3봉([09:30–12:30)·[12:30–14:30)·[14:30–16:00), `rsi_breakout`과 같음),
  exact=True는 5분봉에서 09:30부터 160분씩(12:10·14:50 시작).
- 피벗: ta.pivotlow/pivothigh(5, 5)와 같은 규칙(`dip_reason/code/sr_zones_ref.py`): 가운데가 창의 최솟값(최댓값)이고
  왼쪽 5봉은 엄격히 더 높다(낮다). 같은 봉이면 저점을 먼저 넣는다(Pine 순서).
- 구역: 확정된 스윙 최근 60개 → 가격순으로 가장 낮은 스윙에서 0.5% 안은 한 구역 → (닿은 횟수, 마지막 봉) 강한 순으로
  1.2% 안에 붙은 구역 제외 → 구역 가운데 < 기준 종가면 지지. Z1 = 가장 가까운 지지.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
CACHE = ROOT / "cache"
H1 = CACHE / "h1"
RES = ROOT / "results"
EVENTS = REPO / "data/derived/analysis/earnings_trend/results/events_wide.csv"
EPS = REPO / "data/derived/analysis/earnings_trend/results/wide_eps.csv"
PIV_N, KEEP, TOL, GAP = 5, 60, 0.005, 0.012
REG0, REG1 = 570, 960          # 09:30, 16:00 (분)
PRE0, POST1 = 240, 1200        # 04:00, 20:00


def calendar():
    import exchange_calendars as xcals
    return xcals.get_calendar("XNYS", start="2023-01-01")


def load_h1(t: str) -> pd.DataFrame | None:
    p = H1 / f"{t}.parquet"
    if not p.exists():
        return None
    x = pd.read_parquet(p)
    if x.index.tz is None:
        x.index = x.index.tz_localize("America/New_York")
    else:
        x.index = x.index.tz_convert("America/New_York")
    x = x[~x.index.duplicated(keep="last")].sort_index()
    return x


def mod(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(idx.hour * 60 + idx.minute)


# 160분봉·피벗·구역은 데스크 정본 desk/zones.py를 그대로 쓴다(같은 계산을 두 곳에 두지 않는다).
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(REPO))
from desk.zones import bars160, pivots, zones  # noqa: E402,F401


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(EVENTS)
    ev = ev[(ev.reaction_date >= "2025-03-01") & (ev.reaction_date <= "2026-10-28")].copy()
    ev["amc"] = ev["reaction_date"] > ev["date"]
    return ev.reset_index(drop=True)


def period_of(d: str) -> str:
    if d < "2026-01-01":
        return "2025"
    return "2026H1" if d < "2026-07-01" else "2026H2"


def abs_move_history() -> pd.DataFrame:
    """종목별 과거 반응일 수익률 r0(%) — events_wide(2024-09~)에 그 앞 실적(2023-06~2024-08, wide_eps 시각 + 일봉 종가)을 더한다.
    시각이 장중(09:30~16:00)이거나 없는 발표는 반응일을 정할 수 없어 뺀다."""
    ev = pd.read_csv(EVENTS)[["ticker", "reaction_date", "r0"]]
    first = ev.groupby("ticker")["reaction_date"].min()
    close = pd.read_parquet(CACHE / "d1_close.parquet")
    cal = calendar()
    sessions = cal.sessions
    eps = pd.read_csv(EPS)
    eps = eps[(eps.date >= "2023-06-05") & (eps.date < "2024-09-04") & eps.ticker.isin(close.columns)]
    rows = []
    for r in eps.itertuples():
        if r.ticker in first and r.date >= first[r.ticker]:
            continue
        t = str(r.time_et) if isinstance(r.time_et, str) else ""
        if not t:
            continue
        hm = int(t[:2]) * 60 + int(t[3:5])
        d = pd.Timestamp(r.date)
        if hm >= 960:
            react = sessions[sessions > d]
        elif hm < 570:
            react = sessions[sessions >= d]
        else:
            continue
        if not len(react):
            continue
        rd = react[0]
        base = sessions[sessions < rd][-1]
        s = close[r.ticker]
        if rd in s.index and base in s.index and np.isfinite(s[rd]) and np.isfinite(s[base]) and s[base] > 0:
            rows.append((r.ticker, rd.strftime("%Y-%m-%d"), (s[rd] / s[base] - 1) * 100))
    old = pd.DataFrame(rows, columns=["ticker", "reaction_date", "r0"])
    out = pd.concat([old, ev], ignore_index=True).dropna()
    return out.drop_duplicates(["ticker", "reaction_date"], keep="last").sort_values(["ticker", "reaction_date"])


def prior_scale(hist: pd.DataFrame, ticker: str, before: str, n: int = 8, need: int = 4) -> float | None:
    """그 종목 직전 n번(최소 need번) 반응일 |r0| 평균(%), 판단일(before) 전 반응일만."""
    h = hist[(hist.ticker == ticker) & (hist.reaction_date < before)].tail(n)
    if len(h) < need:
        return None
    return float(h["r0"].abs().mean())
