"""Locked forecasts checked against later prices (desk_system.md §14).

A forecast is stamped and hashed when it is recorded and never edited afterwards. Prices start
at the first US regular-session close after the recording, so nothing seen while writing it can
leak into the score. This module never feeds the portfolio engine.

Earnings records (`type: "earnings"`, §14.2) predict the reaction-day close of each report from
the last close before it. They must be recorded before the announcement (and before the base
close for after-close reports) and are scored per event as soon as the reaction close settles.
A scored report is also split into a market part (beta × a proxy fixed from pre-release closes)
and its own part, so a night when every report falls with the market is not read as skill or luck.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import config
from .news import market_calendar, now_iso, stamp, write_atomic

LEDGER = config.ROOT / "data" / "forecast" / "ledger.json"
PRICES = config.CACHE / "forecast_prices.json"
ESTIMATES = config.CACHE / "forecast_estimates.json"
MARKET = config.CACHE / "forecast_market.json"
# Candidates for an earnings report's market part (§14.2). Equal-weight funds keep a reporting
# megacap's own move out of its "market" (AMZN is ~22% of XLY, ~2% of RSPD); SOXX caps names near 8%.
MARKET_PROXIES = ("RSP", "QQQE", "SOXX", "IBIT", "RSPT", "RSPC", "RSPD", "RSPF", "RSPH", "RSPN", "RSPG", "RSPM",
                  "RSPS", "RSPU", "RSPR")
BETA_SESSIONS, BETA_MIN, BETA_CALENDAR_DAYS = 120, 60, 200
PRICE_MAX_AGE_SEC = 3600
ESTIMATE_MAX_AGE_SEC = 24 * 3600
OPS = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b}
KINDS = {"benchmark_return", "benchmark_min_close", "basket_excess", "basket_count_beat", "factor_rho",
         "ticker_excess", "event_excess", "eps_change", "manual"}
REQUIRED = ("id", "title", "author", "request", "benchmark", "horizon_sessions", "summary",
            "universe", "baskets", "primary_basket", "picks", "claims")
EARN_REQUIRED = ("id", "type", "title", "author", "request", "benchmark", "summary", "model", "events")
QUANTILES = ("0.1", "0.25", "0.5", "0.75", "0.9")
TIMINGS = ("amc", "bmo")
EDGE_MIN = 5  # minutes a listed release time may sit inside the session edge (16:00 vs 15:55 listings)
EXT_TOL = 1.005  # extended-hours low counts only if the bar closes or the next bar trades within 0.5% of it
REG_THROUGH = 0.999  # a regular-session limit fills only when the low goes 0.1% through it


class ForecastError(ValueError):
    pass


def is_earnings(fc: dict) -> bool:
    return isinstance(fc, dict) and fc.get("type") == "earnings"


def digest(forecast: dict) -> str:
    body = {k: v for k, v in forecast.items() if k != "sha256"}
    text = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_dist(d: dict, where: str) -> None:
    """A reaction forecast: P(close up) and five rising return quantiles."""
    p = d.get("p_up") if isinstance(d, dict) else None
    if not isinstance(p, (int, float)) or not 0 < p < 1:
        raise ForecastError(f"{where}: 상승 확률 p_up은 0과 1 사이입니다")
    q = d.get("q")
    if not isinstance(q, dict) or set(q) != set(QUANTILES) or not all(
            isinstance(q[k], (int, float)) and math.isfinite(q[k]) for k in QUANTILES):
        raise ForecastError(f"{where}: 분위수 q는 {', '.join(QUANTILES)} 다섯 개 숫자입니다")
    vals = [q[k] for k in QUANTILES]
    if any(b < a for a, b in zip(vals, vals[1:])):
        raise ForecastError(f"{where}: 분위수는 커지는 순서여야 합니다")


def event_times(ev: dict) -> dict:
    """Release, base close and reaction close of one report, with the session rules checked.

    amc: released on the base day at/after its close, reacts the next session.
    bmo: released before the reaction session opens; base = the session before.
    The record must be made before the cutoff = the earlier of the release and the base close,
    so the base close is the first close after recording, as for every other forecast.
    """
    cal = market_calendar()
    t = ev.get("ticker")
    if not isinstance(t, str) or not t or ev.get("timing") not in TIMINGS:
        raise ForecastError(f"{t}: 종목과 발표 시점(amc·bmo)이 필요합니다")
    try:
        at = stamp(ev.get("announce_at", ""))
    except Exception as exc:
        raise ForecastError(f"{t}: 발표 시각은 시간대가 있는 ISO 형식입니다") from exc
    base, react = ev.get("base_date"), ev.get("reaction_date")
    if not (isinstance(base, str) and isinstance(react, str) and cal.is_session(base) and cal.is_session(react)):
        raise ForecastError(f"{t}: 기준일·반응일은 미국 거래일입니다")
    if cal.next_session(base).strftime("%Y-%m-%d") != react:
        raise ForecastError(f"{t}: 반응일은 기준일 다음 거래일입니다")
    base_close = cal.session_close(base).to_pydatetime()
    react_open = cal.session_open(react).to_pydatetime()
    react_close = cal.session_close(react).to_pydatetime()
    edge = timedelta(minutes=EDGE_MIN)
    if ev["timing"] == "amc" and not base_close - edge <= at < react_open:
        raise ForecastError(f"{t}: 장 마감 뒤 발표는 기준일 마감과 반응일 개장 사이여야 합니다")
    if ev["timing"] == "bmo" and not base_close <= at < react_open + edge:
        raise ForecastError(f"{t}: 장 시작 전 발표는 기준일 마감과 반응일 개장 사이여야 합니다")
    return {"announce": at, "base_close": base_close, "react_open": react_open, "react_close": react_close,
            "cutoff": min(at, base_close)}


def validate_earnings(fc: dict, locked: bool = True) -> None:
    if not all(k in fc for k in EARN_REQUIRED) or not isinstance(fc["events"], list) or not fc["events"]:
        raise ForecastError("실적 예측 필수 항목이 빠졌습니다")
    if not isinstance(fc["model"], dict) or not fc["model"].get("text"):
        raise ForecastError("실적 예측: 계산 방법(model.text)이 필요합니다")
    seen, cutoffs = set(), []
    for ev in fc["events"]:
        if not isinstance(ev, dict):
            raise ForecastError("실적 예측: 사건 형식 오류")
        cutoffs.append(event_times(ev)["cutoff"])
        key = (ev["ticker"], ev["reaction_date"])
        if key in seen:
            raise ForecastError(f"{ev['ticker']}: 같은 반응일 사건이 두 번 있습니다")
        seen.add(key)
        for part in ("mechanical", "forecast"):
            check_dist(ev.get(part), f"{ev['ticker']} {part}")
        if not (ev.get("thesis") and isinstance(ev.get("reasons"), list) and ev["reasons"]
                and isinstance(ev.get("risks"), list)):
            raise ForecastError(f"{ev['ticker']}: 판단 근거·위험이 필요합니다")
        levels = ev.get("levels", [])
        if not isinstance(levels, list):
            raise ForecastError(f"{ev['ticker']}: 지정가 단계는 목록입니다")
        for lv in levels:
            price, p = (lv.get("price"), lv.get("p_touch")) if isinstance(lv, dict) else (None, None)
            if not isinstance(lv, dict) or not lv.get("name") or not isinstance(price, (int, float)) or price <= 0 or (
                    p is not None and not (isinstance(p, (int, float)) and 0 < p < 1)):
                raise ForecastError(f"{ev['ticker']}: 지정가 단계는 이름·가격(>0)·닿을 확률(0~1)입니다")
    if locked:
        at = stamp(fc.get("recorded_at", ""))
        if fc.get("sha256") != digest(fc):
            raise ForecastError(f"{fc['id']}: 기록 뒤 내용이 바뀌었습니다(해시 불일치)")
        if any(at >= c for c in cutoffs):
            raise ForecastError(f"{fc['id']}: 발표나 기준일 마감 중 이른 시각 뒤에는 기록할 수 없습니다")


def validate_forecast(fc: dict, locked: bool = True) -> None:
    if is_earnings(fc):
        validate_earnings(fc, locked)
        return
    if not isinstance(fc, dict) or not all(k in fc for k in REQUIRED):
        raise ForecastError("예측 필수 항목이 빠졌습니다")
    if not isinstance(fc["horizon_sessions"], int) or not 1 <= fc["horizon_sessions"] <= 126:
        raise ForecastError("horizon_sessions는 1~126 거래일입니다")
    universe, baskets = fc["universe"], fc["baskets"]
    if not isinstance(universe, dict) or not universe or not isinstance(baskets, dict):
        raise ForecastError("universe·baskets 형식 오류")
    known = set(universe) | {fc["benchmark"]}
    for key, basket in baskets.items():
        tickers = basket.get("tickers") if isinstance(basket, dict) else None
        if not tickers or len(set(tickers)) != len(tickers) or set(tickers) - known or not basket.get("label"):
            raise ForecastError(f"바스켓 {key}: 이름과 중복 없는 유니버스 종목이 필요합니다")
    if fc["primary_basket"] not in baskets:
        raise ForecastError("primary_basket이 바스켓에 없습니다")
    if fc.get("headline_basket", fc["primary_basket"]) not in baskets:
        raise ForecastError("headline_basket이 바스켓에 없습니다")
    picks = [p.get("ticker") for p in fc["picks"]]
    if picks != baskets[fc["primary_basket"]]["tickers"]:
        raise ForecastError("picks와 primary 바스켓의 종목·순서가 같아야 합니다")
    for p in fc["picks"]:
        if not (p.get("thesis") and isinstance(p.get("reasons"), list) and p["reasons"] and isinstance(p.get("risks"), list)):
            raise ForecastError(f"{p.get('ticker')}: 판단 근거·위험이 필요합니다")
    ids = set()
    for c in fc["claims"]:
        if not isinstance(c, dict) or c.get("kind") not in KINDS or not c.get("id") or not c.get("text"):
            raise ForecastError("예측 항목의 id·text·kind가 필요합니다")
        if c["id"] in ids:
            raise ForecastError(f"중복 예측 항목 {c['id']}")
        ids.add(c["id"])
        p = c.get("probability")
        if not isinstance(p, (int, float)) or not 0 < p < 1:
            raise ForecastError(f"{c['id']}: 확률은 0과 1 사이입니다")
        if c["kind"] != "manual" and c.get("op") not in OPS:
            raise ForecastError(f"{c['id']}: 비교 기호 오류")
        for ref in ("basket", "versus"):
            if ref in c and c[ref] != fc["benchmark"] and c[ref] not in baskets:
                raise ForecastError(f"{c['id']}: 없는 바스켓 {c[ref]}")
        if "ticker" in c and c["ticker"] not in universe:
            raise ForecastError(f"{c['id']}: 유니버스 밖 종목")
        if c["kind"] == "eps_change" and not (isinstance(c.get("base"), (int, float)) and c["base"] != 0):
            raise ForecastError(f"{c['id']}: EPS 기록값(base)은 0이 아닌 숫자여야 합니다")
    if locked:
        stamp(fc.get("recorded_at", ""))
        if fc.get("sha256") != digest(fc):
            raise ForecastError(f"{fc['id']}: 기록 뒤 내용이 바뀌었습니다(해시 불일치)")


def validate(data: dict) -> None:
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("forecasts"), list):
        raise ForecastError("예측 원장은 version 1과 forecasts 목록이 필요합니다")
    seen = set()
    for fc in data["forecasts"]:
        validate_forecast(fc)
        if fc["id"] in seen:
            raise ForecastError(f"중복 예측 {fc['id']}")
        seen.add(fc["id"])


def read_ledger(path: Path = LEDGER) -> dict:
    if not path.exists():
        return {"version": 1, "forecasts": []}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    validate(data)
    return data


def record(incoming: dict, path: Path = LEDGER) -> dict:
    """Stamp and hash a new forecast. An existing id can never be rewritten."""
    fc = copy.deepcopy(incoming)
    for key in ("recorded_at", "sha256"):
        fc.pop(key, None)
    validate_forecast(fc, locked=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError as exc:
        raise ForecastError("다른 예측 기록 작업이 진행 중입니다") from exc
    try:
        data = read_ledger(path)
        if any(f["id"] == fc["id"] for f in data["forecasts"]):
            raise ForecastError("이미 기록된 예측은 바꿀 수 없습니다. 새 id로 기록하세요")
        fc["recorded_at"] = now_iso()
        fc["sha256"] = digest(fc)
        data["forecasts"].append(fc)
        validate(data)
        write_atomic(path, data)
        return fc
    finally:
        lock.unlink()


def schedule(fc: dict) -> dict:
    """Baseline = first regular-session close after the recording; target = baseline + horizon."""
    cal = market_calendar()
    at = stamp(fc["recorded_at"])
    day = cal.date_to_session(at.astimezone(config.NY).date().isoformat(), direction="next")
    if cal.session_close(day).to_pydatetime() <= at:
        day = cal.next_session(day)
    window = cal.sessions_window(day, fc["horizon_sessions"] + 1)
    days = [s.strftime("%Y-%m-%d") for s in window]
    closes = {d: cal.session_close(d).to_pydatetime() for d in days}
    return {"baseline": days[0], "target": days[-1], "days": days, "closes": closes}


CLOSE_GRACE_SEC = 20 * 60  # Yahoo's daily bar settles a few minutes after the closing auction


def final_close(day: str, at: datetime) -> bool:
    """True when `day` is a session whose close had settled at `at`; an intraday bar never enters the cache."""
    cal = market_calendar()
    return cal.is_session(day) and (at - cal.session_close(day).to_pydatetime()).total_seconds() >= CLOSE_GRACE_SEC


def tickers_of(data: dict) -> list[str]:
    out = set()
    for fc in data["forecasts"]:
        if is_earnings(fc):
            out.update(ev["ticker"] for ev in fc["events"])
            out.update(MARKET_PROXIES)
        else:
            out.update(fc["universe"])
        out.add(fc["benchmark"])
    return sorted(out)


def first_day(fc: dict) -> str:
    return min(ev["base_date"] for ev in fc["events"]) if is_earnings(fc) else schedule(fc)["baseline"]


def extended_lows(events: list[dict], fetched: datetime) -> dict:
    """Lowest fillable price per window around each settled report, from Yahoo 1-hour bars.

    post = base-day after-hours (amc only), pre = reaction-day pre-market, reg = regular session.
    An extended bar's low counts only if the bar closes, or the next bar trades, within 0.5% of
    it (one stray print is not a fill), so its fillable floor is max(low, min(close, next low)/1.005).
    A regular-session limit needs the low to go 0.1% through it: floor = low / 0.999.
    """
    import yfinance as yf
    out = {}
    by_ticker: dict[str, list[dict]] = {}
    for ev in events:
        if event_times(ev)["react_close"] + timedelta(seconds=CLOSE_GRACE_SEC) <= fetched:
            by_ticker.setdefault(ev["ticker"], []).append(ev)
    for t, evs in by_ticker.items():
        start = min(e["base_date"] for e in evs)
        end = (pd.Timestamp(max(e["reaction_date"] for e in evs)) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            bars = yf.Ticker(t).history(start=start, end=end, interval="1h", prepost=True, auto_adjust=False)
        except Exception:  # one ticker failing leaves the others scored on the daily bar
            continue
        if bars is None or bars.empty:
            continue
        bars = bars.tz_convert(config.NY)
        nxt = bars["Low"].shift(-1)
        floor = pd.concat([bars["Close"], nxt], axis=1).min(axis=1) / EXT_TOL
        ext_floor = pd.concat([bars["Low"], floor], axis=1).max(axis=1)
        for ev in evs:
            tm = event_times(ev)
            day = pd.Timestamp(ev["reaction_date"], tz=config.NY)
            spans = {"pre": (day + pd.Timedelta(hours=4), pd.Timestamp(tm["react_open"]).tz_convert(config.NY)),
                     "reg": (pd.Timestamp(tm["react_open"]).tz_convert(config.NY),
                             pd.Timestamp(tm["react_close"]).tz_convert(config.NY))}
            if ev["timing"] == "amc":
                b = pd.Timestamp(tm["base_close"]).tz_convert(config.NY)
                spans["post"] = (b, b + pd.Timedelta(hours=4))
            row = {}
            for name, (a, z) in spans.items():
                sel = (bars.index >= a) & (bars.index < z)
                if not sel.any():
                    continue
                low = float(bars["Low"][sel].min())
                eff = float(ext_floor[sel].min()) if name != "reg" else low / REG_THROUGH
                row[name] = {"low": low, "floor": eff}
            out[f"{t}|{ev['reaction_date']}"] = row
    return out


def refresh_prices(data: dict | None = None, path: Path = PRICES) -> dict:
    import yfinance as yf
    data = data if data is not None else read_ledger()
    tickers = tickers_of(data)
    if not tickers:
        raise ForecastError("기록된 예측이 없습니다")
    start = min(first_day(fc) for fc in data["forecasts"])
    start = (pd.Timestamp(start) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    frame = yf.download(tickers, start=start, auto_adjust=False, actions=True, progress=False, threads=True)
    fetched = datetime.now(timezone.utc)
    snap = {"source": "Yahoo Finance / yfinance · 종가(분할 반영)와 수정 종가(배당 반영)", "fetched_at": now_iso(),
            "fetched_epoch": time.time(), "close": {}, "adj": {}, "splits": {}}
    final = lambda d: final_close(d, fetched)
    events = [ev for fc in data["forecasts"] if is_earnings(fc) for ev in fc["events"]]
    earn = {ev["ticker"] for ev in events}
    fields = [("Close", "close", tickers), ("Adj Close", "adj", tickers)]
    fields += [(f, f.lower(), sorted(earn)) for f in ("Open", "High", "Low")] if earn else []
    for field, key, names in fields:
        snap.setdefault(key, {})
        for t in names:
            if t in frame[field]:
                s = frame[field][t].dropna()
                snap[key][t] = {d.strftime("%Y-%m-%d"): float(v) for d, v in s.items() if final(d.strftime("%Y-%m-%d"))}
    if events:
        snap["ext"] = extended_lows(events, fetched)
    if "Stock Splits" in frame:
        for t in tickers:
            if t in frame["Stock Splits"]:
                s = frame["Stock Splits"][t].fillna(0)
                if (s != 0).any():
                    snap["splits"][t] = {d.strftime("%Y-%m-%d"): float(v) for d, v in s[s != 0].items()}
    bench = {fc["benchmark"] for fc in data["forecasts"]}
    if any(not snap["adj"].get(b) for b in bench):
        raise ForecastError("비교 지수 가격이 누락되어 기존 캐시를 유지합니다")
    write_atomic(path, snap)
    return snap


def refresh_estimates(data: dict | None = None, path: Path = ESTIMATES) -> dict:
    """Current next-fiscal-year EPS consensus for tickers an eps_change claim names."""
    import yfinance as yf
    data = data if data is not None else read_ledger()
    wanted = sorted({c["ticker"] for fc in data["forecasts"] for c in fc.get("claims", []) if c["kind"] == "eps_change"})
    values = {}
    for t in wanted:
        try:
            trend = yf.Ticker(t).eps_trend
            v = float(trend.loc["+1y", "current"])
            if math.isfinite(v):
                values[t] = v
        except Exception:  # one bad ticker must not drop the others
            continue
    snap = {"source": "yfinance eps_trend · 다음 회계연도(+1y) 컨센서스 EPS", "fetched_at": now_iso(),
            "fetched_epoch": time.time(), "values": values}
    write_atomic(path, snap)
    return snap


def market_fit(adj: pd.DataFrame, ticker: str, base: str, skip: set[str]) -> dict | None:
    """The proxy that moved most like the stock before a report, and the stock's beta to it.

    Only closes up to the base day enter, so the choice was knowable before the release and the
    reaction never picks its own benchmark. The stock's past reaction days are left out so one
    earnings gap does not set the beta. None = no stock prices (try again later).
    """
    upto = adj[adj.index <= base]
    if ticker not in upto or upto[ticker].dropna().empty:
        return None
    rets = upto.pct_change(fill_method=None)
    rets = rets[~rets.index.isin(skip)]
    best = None
    for p in MARKET_PROXIES:
        if p == ticker or p not in rets:
            continue
        pair = pd.concat([rets[ticker], rets[p]], axis=1, keys=["s", "m"]).dropna().iloc[-BETA_SESSIONS:]
        if len(pair) < BETA_MIN or not pair["m"].var() > 0:
            continue
        corr = float(pair["s"].corr(pair["m"]))
        if best is None or corr > best["corr"]:
            best = {"proxy": p, "beta": round(float(pair["s"].cov(pair["m"]) / pair["m"].var()), 4),
                    "corr": round(corr, 4), "n": len(pair), "from": pair.index[0], "to": pair.index[-1]}
    return best or {"proxy": None, "reason": f"발표 전 겹치는 거래일 {BETA_MIN}일 미만"}


def market_key(ev: dict) -> str:
    return f"{ev['ticker']}|{ev['reaction_date']}"


def pending_market(data: dict, market: dict | None, now: datetime) -> list[dict]:
    """Reports whose base close has settled but whose market proxy is not fixed yet."""
    done = (market or {}).get("events", {})
    return [ev for fc in data["forecasts"] if is_earnings(fc) for ev in fc["events"] if market_key(ev) not in done
            and now >= event_times(ev)["base_close"] + timedelta(seconds=CLOSE_GRACE_SEC)]


def refresh_market(data: dict | None = None, path: Path = MARKET, now: datetime | None = None) -> dict:
    """Fix each settled report's market proxy and beta once. Pre-release closes only, so they never change."""
    import yfinance as yf
    data = data if data is not None else read_ledger()
    now = now or datetime.now(timezone.utc)
    snap = _read(path) or {"events": {}}
    todo = pending_market(data, snap, now)
    if not todo:
        return snap
    names = sorted({ev["ticker"] for ev in todo} | set(MARKET_PROXIES))
    start = (pd.Timestamp(min(ev["base_date"] for ev in todo)) - pd.Timedelta(days=BETA_CALENDAR_DAYS)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(max(ev["base_date"] for ev in todo)) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frame = yf.download(names, start=start, end=end, auto_adjust=False, progress=False, threads=True)
    if frame is None or frame.empty:
        raise ForecastError("시장 몫 계산용 가격을 받지 못했습니다")
    adj = frame["Adj Close"]
    adj.index = adj.index.strftime("%Y-%m-%d")
    for ev in todo:
        skip = {h["reaction_date"] for h in (ev.get("inputs") or {}).get("hist", []) if h.get("reaction_date")}
        fit = market_fit(adj, ev["ticker"], ev["base_date"], skip)
        if fit is not None:
            snap["events"][market_key(ev)] = fit
    snap.update(source="yfinance 수정 종가(분할·배당 반영)", fetched_at=now_iso(),
                method=f"발표 전 {BETA_SESSIONS}거래일(기준일 종가까지, 지난 실적 반응일 제외) 하루 수익으로 "
                       f"후보 {len(MARKET_PROXIES)}개 중 상관이 가장 큰 지수를 고르고 그 베타를 쓴다")
    write_atomic(path, snap)
    return snap


SPLIT_TOL = 1.35


def unadjusted(series: dict, splits: dict, after: str) -> bool:
    """A split inside the window whose day still shows the ~1/k drop: Yahoo has not rescaled history yet."""
    days = sorted(series)
    for day, k in splits.items():
        if day <= after or not k or k <= 0 or day not in series:
            continue
        i = days.index(day)
        if i and abs(math.log(series[day] / series[days[i - 1]] * k)) < math.log(SPLIT_TOL):
            return True
    return False


def split_factor(splits: dict, after: str) -> float:
    """Product of split ratios after a date. yfinance rescales every earlier close by it."""
    factor = 1.0
    for day, k in splits.items():
        if day > after and k and k > 0:
            factor *= k
    return factor


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def spearman(xs: dict, ys: dict) -> tuple[float | None, int]:
    frame = pd.DataFrame({"x": pd.Series(xs, dtype=float), "y": pd.Series(ys, dtype=float)}).dropna()
    if len(frame) < 5:
        return None, len(frame)
    return float(frame["x"].rank().corr(frame["y"].rank())), len(frame)


def evaluate(fc: dict, prices: dict | None, estimates: dict | None = None, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    sch = schedule(fc)
    bench = fc["benchmark"]
    adj = (prices or {}).get("adj", {})
    close = (prices or {}).get("close", {})
    days = [d for d in sch["days"] if sch["closes"][d] <= now and d in adj.get(bench, {})]
    base = sch["baseline"]
    out = {"baseline": base, "target": sch["target"], "sessions": len(sch["days"]) - 1,
           "elapsed": max(0, len(days) - 1), "complete": sch["target"] in days, "days": days,
           "latest": days[-1] if days else None}
    if not days or days[0] != base:
        out.update(status="기준 종가 대기", paths={}, returns={}, baskets={}, claims=[
            {**c, "observed": None, "state": "pending"} for c in fc["claims"]], picks=[], brier=None, resolved=0)
        return out

    def path(t: str, field: dict = adj) -> list[float | None]:
        series, last, vals = field.get(t, {}), None, []
        b = series.get(base)
        for d in days:
            if d in series:
                last = series[d]
            vals.append(None if b is None or last is None else last / b - 1)
        return vals

    splits = (prices or {}).get("splits", {})
    recorded_day = stamp(fc["recorded_at"]).astimezone(config.NY).date().isoformat()
    bad = sorted(t for t in set(fc["universe"]) | {bench} if unadjusted(adj.get(t, {}), splits.get(t, {}), base))
    out["price_issues"] = bad
    if bench in bad:
        out.update(status="분할 반영 대기", paths={}, returns={}, baskets={}, picks=[], brier=None, resolved=0,
                   claims=[{**c, "observed": None, "state": "pending"} for c in fc["claims"]])
        return out
    paths = {t: path(t) for t in (set(fc["universe"]) | {bench}) - set(bad)}
    returns = {t: p[-1] for t, p in paths.items() if p and p[-1] is not None}
    r_bench = returns.get(bench)

    def basket_path(tickers):
        cols = [paths[t] for t in tickers if paths.get(t) and paths[t][0] is not None]
        if not cols:
            return [None] * len(days)
        return [sum(c[i] for c in cols if c[i] is not None) / max(1, sum(c[i] is not None for c in cols))
                for i in range(len(days))]

    baskets = {}
    for key, b in fc["baskets"].items():
        bp = basket_path(b["tickers"])
        r = bp[-1]
        beta = b.get("beta")
        baskets[key] = {"label": b["label"], "tickers": b["tickers"], "beta": beta, "path": bp, "return": r,
                        "excess": None if r is None or r_bench is None else r - r_bench,
                        "alpha": None if r is None or r_bench is None or beta is None else r - beta * r_bench,
                        "beat": sum(1 for t in b["tickers"] if returns.get(t) is not None and r_bench is not None
                                    and returns[t] > r_bench)}

    ranked = sorted((t for t in fc["universe"] if t in returns), key=lambda t: -returns[t])
    rank_of = {t: i + 1 for i, t in enumerate(ranked)}
    picks = []
    for p in fc["picks"]:
        t = p["ticker"]
        pp = paths.get(t) or []
        cs = close.get(t, {})
        picks.append({"ticker": t, "base_close": cs.get(base), "last_close": cs.get(days[-1]),
                      "return": returns.get(t), "excess": None if returns.get(t) is None or r_bench is None
                      else returns[t] - r_bench, "rank": rank_of.get(t), "of": len(ranked),
                      "worst": min([v for v in pp if v is not None], default=None)})

    est = (estimates or {}).get("values", {})
    claims = []
    for c in fc["claims"]:
        value, final = None, out["complete"]
        k = c["kind"]
        threshold, note = c.get("value"), None
        if k == "benchmark_return":
            value = r_bench
        elif k == "benchmark_min_close":
            closes = [close.get(bench, {}).get(d) for d in days]
            closes = [v for v in closes if v is not None]
            value = min(closes) if closes else None
            factor = split_factor(splits.get(bench, {}), recorded_day)
            if factor != 1:  # the recorded level is in pre-split units
                threshold = c["value"] / factor
                note = f"기록 뒤 {bench} 분할 {factor:g}:1 반영 — 기준 {c['value']:g} → {threshold:.2f}"
            if value is not None and not OPS[c["op"]](value, threshold):
                final = True  # a broken low cannot be repaired later
        elif k in ("basket_excess", "basket_count_beat"):
            b = baskets[c["basket"]]
            if k == "basket_count_beat":
                value = b["beat"]
            elif c.get("versus", bench) == bench:
                value = b["alpha"] if c.get("adjust") == "beta" else b["excess"]
            else:
                o = baskets[c["versus"]]
                field = "alpha" if c.get("adjust") == "beta" else "return"
                value = None if b[field] is None or o[field] is None else b[field] - o[field]
        elif k == "factor_rho":
            xs = {t: v.get(c["factor"]) for t, v in fc["universe"].items()}
            value, n = spearman(xs, {t: returns.get(t) for t in fc["universe"]})
        elif k == "ticker_excess":
            r = returns.get(c["ticker"])
            value = None if r is None or r_bench is None else r - r_bench
        elif k == "event_excess":
            a, z = c["start"], c["end"]
            s, m = adj.get(c["ticker"], {}), adj.get(bench, {})
            final = z in days
            if a in s and a in m and days:
                last = z if z in days else days[-1]
                if last >= a and last in s and last in m:
                    value = (s[last] / s[a] - 1) - (m[last] / m[a] - 1)
        elif k == "eps_change":
            v = est.get(c["ticker"])
            value = None if v is None else (v - c["base"]) / abs(c["base"])  # 적자 기업도 개선이면 +
        if k == "manual":
            state = "manual"
        elif value is None:
            state = "pending"
        else:
            ok = OPS[c["op"]](value, threshold)
            state = ("hit" if ok else "miss") if final else ("on_track" if ok else "off_track")
        claims.append({**c, "observed": value, "state": state, "threshold": threshold, "note": note})
    resolved = [(c["probability"], 1.0 if c["state"] == "hit" else 0.0) for c in claims if c["state"] in ("hit", "miss")]
    out.update(status="완료" if out["complete"] else "진행 중", paths=paths, returns=returns, baskets=baskets,
               picks=picks, claims=claims,
               brier=None if not resolved else sum((p - o) ** 2 for p, o in resolved) / len(resolved),
               resolved=len(resolved))
    return out


def pinball(q: dict, y: float) -> float:
    """Mean quantile (pinball) loss over the five quantiles; lower is better, in return units."""
    return sum(max(float(k) * (y - v), (float(k) - 1) * (y - v)) for k, v in q.items()) / len(q)


def score_dist(d: dict, y: float) -> dict:
    q, p = d["q"], d["p_up"]
    up = 1.0 if y > 0 else 0.0
    return {"p_up": p, "q": q, "brier": (p - up) ** 2, "dir_hit": None if p == 0.5 else (y > 0) == (p > 0.5),
            "in80": q["0.1"] <= y <= q["0.9"], "in50": q["0.25"] <= y <= q["0.75"],
            "pinball": pinball(q, y), "err50": y - q["0.5"]}


def market_split(fit: dict | None, close: dict, splits: dict, base: str, react: str, y: float) -> dict | None:
    """Market part = beta × the proxy's base→reaction close move; own part = the rest of the reaction."""
    proxy = (fit or {}).get("proxy")
    px = close.get(proxy, {}) if proxy else {}
    if base not in px or react not in px or unadjusted(px, splits.get(proxy, {}), base):
        return None
    move = px[react] / px[base] - 1
    part = fit["beta"] * move
    return {"proxy": proxy, "beta": fit["beta"], "corr": fit.get("corr"), "proxy_return": move, "part": part, "own": y - part}


def evaluate_earnings(fc: dict, prices: dict | None, now: datetime | None = None, market: dict | None = None) -> dict:
    """Score each report once its reaction close has settled; earlier events stay waiting."""
    now = now or datetime.now(timezone.utc)
    p = prices or {}
    close, low, high, opn = (p.get(k, {}) for k in ("close", "low", "high", "open"))
    ext, splits, bench = p.get("ext", {}), p.get("splits", {}), fc["benchmark"]
    fits = (market or {}).get("events", {})
    rows = []
    for ev in fc["events"]:
        tm = event_times(ev)
        t, b, r = ev["ticker"], ev["base_date"], ev["reaction_date"]
        cs = close.get(t, {})
        row = {"ticker": t, "name": ev.get("name"), "timing": ev["timing"], "announce_at": ev["announce_at"],
               "base_date": b, "reaction_date": r, "base_close": cs.get(b), "close": None, "return": None,
               "levels": [], "state": "waiting"}
        settled = now >= tm["react_close"] + timedelta(seconds=CLOSE_GRACE_SEC)
        if settled and b in cs and r in cs:
            row["state"] = "split" if unadjusted(cs, splits.get(t, {}), b) else "scored"
        elif settled:
            row["state"] = "no_price"
        if row["state"] != "scored":
            rows.append(row)
            continue
        y = cs[r] / cs[b] - 1
        mb = close.get(bench, {})
        w = ext.get(f"{t}|{r}", {})
        reg_low = low.get(t, {}).get(r)
        floors = {k: v["floor"] for k, v in w.items()}
        if "reg" not in floors and reg_low is not None:
            floors["reg"] = reg_low / REG_THROUGH
        row.update(close=cs[r], open=opn.get(t, {}).get(r), high=high.get(t, {}).get(r), low=reg_low,
                   ext={k: v["low"] for k, v in w.items()}, floors=floors, **{"return": y},
                   bench_return=mb[r] / mb[b] - 1 if b in mb and r in mb else None,
                   market=market_split(fits.get(market_key(ev)), close, splits, b, r, y),
                   scale=(ev.get("inputs") or {}).get("scale"),
                   forecast=score_dist(ev["forecast"], y), mechanical=score_dist(ev["mechanical"], y))
        for lv in ev.get("levels", []):
            price = lv["price"]
            hit = {k: f <= price for k, f in floors.items()}
            touched = hit.get("pre", False) or hit.get("reg", False)  # 반응일 프리장 + 정규장(p_touch 정의)
            row["levels"].append({**lv, "dist": price / cs[b] - 1, "touched": touched, "touched_post": hit.get("post"),
                                  "to_close": cs[r] / price - 1 if touched else None,
                                  "brier": None if lv.get("p_touch") is None else (lv["p_touch"] - float(touched)) ** 2})
        rows.append(row)
    scored = [x for x in rows if x["state"] == "scored"]
    done = all(x["state"] == "scored" for x in rows)
    return {"status": "완료" if done else ("진행 중" if scored else "발표 대기"), "events": rows,
            "score": earnings_score(scored)}


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def earnings_score(rows: list[dict]) -> dict:
    """Running report card: my calls against the mechanical baseline on the same events."""
    out = {"n": len(rows)}
    for side in ("forecast", "mechanical"):
        s = [x[side] for x in rows]
        calls = [x["dir_hit"] for x in s if x["dir_hit"] is not None]
        out[side] = {"brier": _mean([x["brier"] for x in s]), "pinball": _mean([x["pinball"] for x in s]),
                     "in80": _mean([float(x["in80"]) for x in s]), "in50": _mean([float(x["in50"]) for x in s]),
                     "calls": len(calls), "hits": sum(calls)}
    out["size_ratio"] = _mean([abs(x["return"]) / x["scale"] for x in rows if x.get("scale")])
    split = [x for x in rows if x.get("market")]
    calls = [x for x in split if x["forecast"]["p_up"] != 0.5]
    out["own"] = {"n": len(split), "calls": len(calls),
                  "hits": sum((x["market"]["own"] > 0) == (x["forecast"]["p_up"] > 0.5) for x in calls),
                  "abs_part": _mean([abs(x["market"]["part"]) for x in split]),
                  "abs_own": _mean([abs(x["market"]["own"]) for x in split])}
    lv = [l for x in rows for l in x["levels"]]
    out["levels"] = {"n": len(lv), "touched": sum(l["touched"] for l in lv),
                     "brier": _mean([l["brier"] for l in lv]),
                     "to_close": _mean([l["to_close"] for l in lv if l["touched"]])}
    return out


def _stale(snap: dict | None, max_age: float) -> bool:
    return snap is None or time.time() - snap.get("fetched_epoch", 0) > max_age


def _missed_close(snap: dict | None, now: datetime) -> bool:
    """A session settled after this snapshot was fetched, so its close is not in the cache yet."""
    if snap is None:
        return True
    cal = market_calendar()
    last = cal.date_to_session(now.astimezone(config.NY).date().isoformat(), direction="previous")
    while not final_close(last.strftime("%Y-%m-%d"), now):
        last = cal.previous_session(last)
    settled = cal.session_close(last).to_pydatetime() + timedelta(seconds=CLOSE_GRACE_SEC)
    return datetime.fromtimestamp(snap.get("fetched_epoch", 0), timezone.utc) < settled


def payload(force: bool = False, data: dict | None = None, prices: dict | None = None,
            estimates: dict | None = None, now: datetime | None = None, market: dict | None = None) -> dict:
    data = data if data is not None else read_ledger()
    validate(data)
    warning = None
    if data["forecasts"] and prices is None:
        prices = _read(PRICES)
        market = _read(MARKET) if market is None else market
        if force or _stale(prices, PRICE_MAX_AGE_SEC) or _missed_close(prices, now or datetime.now(timezone.utc)):
            try:
                prices = refresh_prices(data)
            except Exception as exc:
                warning = f"가격 갱신 실패 · 저장 자료 사용: {type(exc).__name__}: {exc}"
            if pending_market(data, market, now or datetime.now(timezone.utc)):  # 가격 갱신 주기에만 다시 시도
                try:
                    market = refresh_market(data)
                except Exception as exc:
                    warning = (warning + " · " if warning else "") + f"시장 몫 계산 실패: {type(exc).__name__}"
        estimates = _read(ESTIMATES) if estimates is None else estimates
        if force or _stale(estimates, ESTIMATE_MAX_AGE_SEC):
            try:
                estimates = refresh_estimates(data)
            except Exception as exc:
                warning = (warning + " · " if warning else "") + f"전망치 갱신 실패: {type(exc).__name__}"
    items = [{**fc, "result": evaluate_earnings(fc, prices, now, market) if is_earnings(fc)
              else evaluate(fc, prices, estimates, now)} for fc in data["forecasts"]]
    history = [{**x, "record": fc["id"]} for fc in items if is_earnings(fc)
               for x in fc["result"]["events"] if x["state"] == "scored"]
    return {"forecasts": items, "warning": warning, "earnings": {"score": earnings_score(history), "events": history},
            "price_source": (prices or {}).get("source"), "prices_fetched_at": (prices or {}).get("fetched_at"),
            "estimates_fetched_at": (estimates or {}).get("fetched_at"),
            "method": "기록 시각을 찍고 해시로 잠근 예측입니다. 기준가는 기록 뒤 첫 미국 정규장 종가이고, "
                      "수익률은 수정 종가(배당 반영), 저점 비교는 종가로 잽니다. 바스켓은 같은 금액으로 사서 들고 있는 수익입니다."}


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else [])
    if argv[:1] == ["prep"]:  # 실적 반응 예측 초안(§14.2): py -m desk forecast prep --date … --amc … --out …
        from .earnings_prep import main as prep
        return prep(argv[1:])
    parser = argparse.ArgumentParser(description="예측 기록·점검")
    parser.add_argument("command", choices=("show", "validate", "record", "refresh"), nargs="?", default="show")
    parser.add_argument("file", nargs="?")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            data = read_ledger()
            print(f"예측 원장 OK: {len(data['forecasts'])}건")
        elif args.command == "record":
            if not args.file:
                parser.error("record에는 JSON 파일이 필요합니다")
            fc = record(json.loads(Path(args.file).read_text(encoding="utf-8-sig")))
            if is_earnings(fc):
                for ev in fc["events"]:
                    print(f"  {ev['ticker']}: 기준 {ev['base_date']} 종가 → 반응 {ev['reaction_date']} 종가 · "
                          f"기록 마감 {event_times(ev)['cutoff'].isoformat()}")
                print(f"기록: {fc['id']} · {fc['recorded_at']} · 실적 {len(fc['events'])}건 · sha256 {fc['sha256']}")
            else:
                sch = schedule(fc)
                print(f"기록: {fc['id']} · {fc['recorded_at']} · 기준 {sch['baseline']} → 마감 {sch['target']} · sha256 {fc['sha256']}")
        elif args.command == "refresh":
            data = read_ledger()
            snap = refresh_prices(data)
            est = refresh_estimates(data)
            mk = refresh_market(data)
            print(f"가격 {len(snap['adj'])}종목 · 전망치 {len(est['values'])}종목 · 시장 몫 {len(mk.get('events', {}))}건 갱신")
        else:
            print(json.dumps(payload(), ensure_ascii=False, indent=2, default=str))
        return 0
    except (ForecastError, OSError, json.JSONDecodeError) as exc:
        print(f"예측 오류: {exc}")
        return 2
