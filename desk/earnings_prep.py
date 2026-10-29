"""Draft an earnings-reaction forecast (desk_system.md §14.2) before the reports come out.

For each report it gathers what is known before the release and writes the mechanical baseline:
- s_hist: mean |reaction-day return| of the last 8 reports (Yahoo earnings dates + daily closes).
- implied move: ATM straddle of the first option expiry on/after the reaction day ÷ last close, with the
  normal days inside the option window taken out (event move = 0.8·√((IM/0.8)² − n·σ_d²)).
- scale s = W_IM·event move + (1 − W_IM)·s_hist (only one of them if the other is missing).
- distribution: P(up) = P_UP and quantiles = s × Z (2026 reaction quantiles in units of s_hist).
- 160-minute support zones Z1..Z6 (desk/zones.py) with the chance the reaction day trades through each
  (TOUCH, pre-market + regular session, from the earnings_support backtest).
The judgment (`forecast`) starts equal to the baseline; edit it with reasons, then lock it with
`py -m desk forecast record <file>`. Prices come from yfinance only (no TradingView in scripts).
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, zones
from .news import market_calendar

# 2026 reaction-day returns ÷ s_hist (mean |r| of the previous 8 reports), 4,701 reports of 1,629 US stocks
# (data/derived/analysis/earnings_trend/results/events_wide.csv). Picked from 2026-01~06 the 80% band
# covered 85% in 2026-07~10 and 81% in 2025.
Z = {"0.1": -1.67, "0.25": -0.78, "0.5": 0.01, "0.75": 0.91, "0.9": 1.84}
P_UP = 0.50  # share of up reactions: 2026-01~06 51.0%, 2026-07~10 50.0% (≥$50B: 50.8%, 49.4%)
W_IM = 0.5   # weight of the option-implied move in the scale; refit once enough reports are scored
HIST_N, HIST_MIN = 8, 4
# P(reaction-day pre-market or regular low fills a limit at base × (1 − m·s)) by m: 2026 reports of stocks
# ≥$10B at the time, 2,680 (data/derived/analysis/earnings_support/results/touch_calibration.csv, window B,
# fill rule of desk/forecast.py). 2025, 2026H1 and 2026H2 agree within ±2%p. Below 0.25s it stays at 0.70.
TOUCH = {0.25: 0.699, 0.5: 0.590, 0.75: 0.488, 1.0: 0.390, 1.25: 0.312, 1.5: 0.241,
         1.75: 0.184, 2.0: 0.146, 2.25: 0.115, 2.5: 0.088, 2.75: 0.064, 3.0: 0.050}


def reaction_day(ts: pd.Timestamp, cal) -> str | None:
    """Session that first trades on a report released at `ts` (None when released in the session)."""
    t = ts.tz_convert(config.NY)
    day = t.strftime("%Y-%m-%d")
    minutes = t.hour * 60 + t.minute
    if not cal.is_session(day):  # weekend or holiday release
        return cal.date_to_session(day, direction="next").strftime("%Y-%m-%d")
    if minutes < 570:
        return day
    if minutes >= 960:
        return cal.next_session(day).strftime("%Y-%m-%d")
    return None


def past_reactions(ticker: str, before: str, n: int = HIST_N) -> list[dict]:
    """Reaction-day close-to-close returns of the last `n` reports before `before`."""
    import yfinance as yf
    cal = market_calendar()
    tk = yf.Ticker(ticker)
    try:
        dates = tk.get_earnings_dates(limit=28)
    except Exception:
        return []
    if dates is None or dates.empty:
        return []
    closes = tk.history(period="5y", auto_adjust=False)["Close"]
    closes.index = closes.index.strftime("%Y-%m-%d")
    out = []
    for ts in sorted(dates.index):
        r = reaction_day(pd.Timestamp(ts), cal)
        if r is None or r >= before or r not in closes.index:
            continue
        b = cal.previous_session(r).strftime("%Y-%m-%d")
        if b in closes.index:
            out.append({"reaction_date": r, "r": float(closes[r] / closes[b] - 1)})
    return out[-n:]


def daily_sigma(ticker: str, skip: list[str]) -> float | None:
    """Standard deviation of the last 20 close-to-close returns, report reaction days left out."""
    import yfinance as yf
    c = yf.Ticker(ticker).history(period="3mo", auto_adjust=False)["Close"]
    r = c.pct_change().dropna()
    r = r[[d.strftime("%Y-%m-%d") not in skip for d in r.index]].tail(20)
    return float(r.std()) if len(r) >= 10 else None


def implied_move(ticker: str, reaction: str, spot: float, last_close_day: str, sigma: float | None) -> dict | None:
    """ATM straddle ÷ spot for the first expiry on/after the reaction day, then the event-only move."""
    import yfinance as yf
    cal = market_calendar()
    tk = yf.Ticker(ticker)
    exps = [e for e in tk.options if e >= reaction]
    if not exps:
        return None
    e = exps[0]
    ch = tk.option_chain(e)
    c, p = ch.calls.set_index("strike"), ch.puts.set_index("strike")
    strikes = sorted(set(c.index) & set(p.index))
    below = [k for k in strikes if k <= spot]
    above = [k for k in strikes if k >= spot]
    if not below or not above:
        return None

    def price(row) -> float:
        bid, ask = float(row.bid or 0), float(row.ask or 0)
        return (bid + ask) / 2 if bid > 0 and ask > 0 else float(row.lastPrice)

    k0, k1 = below[-1], above[0]
    s0 = price(c.loc[k0]) + price(p.loc[k0])
    s1 = price(c.loc[k1]) + price(p.loc[k1])
    straddle = s0 if k1 == k0 else s0 + (s1 - s0) * (spot - k0) / (k1 - k0)
    im = straddle / spot
    sessions = cal.sessions_in_range(cal.next_session(last_close_day), e)
    extra = max(0, len(sessions) - 1)  # every session in the window except the reaction day
    event = im
    if sigma and extra:
        var = (im / 0.8) ** 2 - extra * sigma ** 2
        event = 0.8 * math.sqrt(max(var, (0.5 * im / 0.8) ** 2))
    return {"expiry": e, "straddle": round(straddle, 4), "implied_move": round(im, 4),
            "normal_days": extra, "sigma_day": None if sigma is None else round(sigma, 4),
            "implied_move_event": round(event, 4)}


def p_touch(depth: float, scale: float) -> float | None:
    """Chance the reaction day trades `depth` (fraction) below the base close, from TOUCH."""
    if not TOUCH or not scale or depth <= 0:
        return None
    ms, ps = zip(*sorted(TOUCH.items()))
    p = float(np.interp(depth / scale, ms, ps))
    return round(min(max(p, 0.01), 0.99), 3)


def distribution(scale: float, p_up: float = P_UP) -> dict:
    return {"p_up": p_up, "q": {k: round(z * scale, 4) for k, z in Z.items()}}


def draft_event(ticker: str, timing: str, base: str, reaction: str, announce_at: str, now: pd.Timestamp) -> dict:
    import time
    import yfinance as yf
    start = (now - pd.Timedelta(days=728)).strftime("%Y-%m-%d")  # period='730d' asks from the listing day for young stocks
    for attempt in range(3):  # Yahoo sometimes answers an empty frame under load
        h1 = yf.Ticker(ticker).history(start=start, interval="1h", auto_adjust=False)
        if not h1.empty:
            break
        time.sleep(3 * (attempt + 1))
    daily = yf.Ticker(ticker).history(period="1mo", auto_adjust=False)["Close"]
    done = [d for d in daily.index if d.strftime("%Y-%m-%d") < now.tz_convert(config.NY).strftime("%Y-%m-%d")
            or now.tz_convert(config.NY).hour >= 16]
    last_day = done[-1].strftime("%Y-%m-%d")
    spot = float(daily[done[-1]])
    hist = past_reactions(ticker, reaction)
    s_hist = float(np.mean([abs(x["r"]) for x in hist])) if len(hist) >= HIST_MIN else None
    sigma = daily_sigma(ticker, [x["reaction_date"] for x in hist])
    try:
        im = implied_move(ticker, reaction, spot, last_day, sigma)
    except Exception:  # no listed options: fall back to history alone
        im = None
    parts = [(W_IM, im["implied_move_event"])] if im else []
    if s_hist:
        parts.append((1 - W_IM, s_hist))
    scale = sum(w * v for w, v in parts) / sum(w for w, _ in parts) if parts else None
    if scale is None:
        raise ValueError(f"{ticker}: 과거 반응도 옵션도 없어 기계 기준을 만들 수 없습니다")
    zs = zones.current(h1, ref=spot, now=now)
    levels = []
    for z in zs["supports"][:6]:
        depth = 1 - z["lo"] / spot
        levels.append({"name": z["name"], "lo": z["lo"], "hi": z["hi"], "price": z["lo"], "touches": z["touches"],
                       "dist_ref": round(-depth, 4), **({"p_touch": p_touch(depth, scale)} if TOUCH else {})})
    reasons = [f"기준 규모 {scale:.1%}: " + " · ".join(
        ([f"옵션 내재 변동 {im['implied_move']:.1%}({im['expiry']} 만기 ATM, 사건만 {im['implied_move_event']:.1%})"] if im else [])
        + ([f"최근 {len(hist)}번 반응 평균 {s_hist:.1%}"] if s_hist else []))]
    if hist:
        reasons.append("최근 반응: " + ", ".join(f"{x['reaction_date'][2:]} {x['r']:+.1%}" for x in hist[-4:]))
    return {"ticker": ticker, "timing": timing, "announce_at": announce_at, "base_date": base, "reaction_date": reaction,
            "inputs": {"spot_ref": round(spot, 4), "spot_ref_date": last_day, "s_hist": None if s_hist is None else round(s_hist, 4),
                       "hist": hist, **(im or {}), "w_im": W_IM if im and s_hist else (1.0 if im else 0.0),
                       "scale": round(scale, 4), "zones_asof": zs["asof"], "swings": zs["swings"],
                       "resistances": zs["resistances"][:3]},
            "mechanical": distribution(scale), "forecast": distribution(scale), "levels": levels,
            "thesis": "기계 기준 그대로(판단 없음)", "reasons": reasons, "risks": ["방향은 실적 내용이 정한다 — 발표 전 방향 근거는 약하다"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="실적 반응 예측 초안")
    parser.add_argument("--date", required=True, help="발표일(뉴욕 날짜) YYYY-MM-DD")
    parser.add_argument("--amc", nargs="*", default=[], help="장 마감 뒤 발표 종목. TICKER 또는 TICKER@HH:MM(뉴욕)")
    parser.add_argument("--bmo", nargs="*", default=[], help="장 시작 전 발표 종목. TICKER 또는 TICKER@HH:MM(뉴욕)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    cal = market_calendar()
    if not cal.is_session(args.date):
        parser.error("발표일은 미국 거래일이어야 합니다")
    now = pd.Timestamp.now(tz="UTC")
    events = []
    for timing, names, default in (("amc", args.amc, "16:00"), ("bmo", args.bmo, "07:00")):
        for item in names:
            t, _, hm = item.partition("@")
            when = pd.Timestamp(f"{args.date} {hm or default}", tz=config.NY)
            if timing == "amc":
                base, react = args.date, cal.next_session(args.date).strftime("%Y-%m-%d")
            else:
                base, react = cal.previous_session(args.date).strftime("%Y-%m-%d"), args.date
            events.append(draft_event(t.upper(), timing, base, react, when.isoformat(), now))
            print(f"{t.upper()}: 기준 {events[-1]['inputs']['scale']:.1%} · 지지 {len(events[-1]['levels'])}개")
    record = {"id": f"{args.date}-earnings", "type": "earnings", "title": f"{args.date} 실적 반응 예측",
              "author": "Claude", "request": "", "benchmark": "QQQ", "summary": "",
              "model": {"name": "기계 기준 v1", "text": __doc__.strip(), "z": Z, "p_up": P_UP, "w_im": W_IM,
                        "touch": {str(k): v for k, v in TOUCH.items()}},
              "events": events, "drafted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    Path(args.out).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"초안: {args.out} — 판단(forecast)·근거를 고친 뒤 record로 잠근다")
    return 0
