"""Locked forecasts checked against later prices (desk_system.md §14).

A forecast is stamped and hashed when it is recorded and never edited afterwards. Prices start
at the first US regular-session close after the recording, so nothing seen while writing it can
leak into the score. This module never feeds the portfolio engine.
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
PRICE_MAX_AGE_SEC = 3600
ESTIMATE_MAX_AGE_SEC = 24 * 3600
OPS = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b}
KINDS = {"benchmark_return", "benchmark_min_close", "basket_excess", "basket_count_beat", "factor_rho",
         "ticker_excess", "event_excess", "eps_change", "manual"}
REQUIRED = ("id", "title", "author", "request", "benchmark", "horizon_sessions", "summary",
            "universe", "baskets", "primary_basket", "picks", "claims")


class ForecastError(ValueError):
    pass


def digest(forecast: dict) -> str:
    body = {k: v for k, v in forecast.items() if k != "sha256"}
    text = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_forecast(fc: dict, locked: bool = True) -> None:
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
        out.update(fc["universe"])
        out.add(fc["benchmark"])
    return sorted(out)


def refresh_prices(data: dict | None = None, path: Path = PRICES) -> dict:
    import yfinance as yf
    data = data if data is not None else read_ledger()
    tickers = tickers_of(data)
    if not tickers:
        raise ForecastError("기록된 예측이 없습니다")
    start = min(schedule(fc)["baseline"] for fc in data["forecasts"])
    start = (pd.Timestamp(start) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    frame = yf.download(tickers, start=start, auto_adjust=False, actions=True, progress=False, threads=True)
    fetched = datetime.now(timezone.utc)
    snap = {"source": "Yahoo Finance / yfinance · 종가(분할 반영)와 수정 종가(배당 반영)", "fetched_at": now_iso(),
            "fetched_epoch": time.time(), "close": {}, "adj": {}, "splits": {}}
    final = lambda d: final_close(d, fetched)
    for field, key in (("Close", "close"), ("Adj Close", "adj")):
        for t in tickers:
            if t in frame[field]:
                s = frame[field][t].dropna()
                snap[key][t] = {d.strftime("%Y-%m-%d"): float(v) for d, v in s.items() if final(d.strftime("%Y-%m-%d"))}
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
    wanted = sorted({c["ticker"] for fc in data["forecasts"] for c in fc["claims"] if c["kind"] == "eps_change"})
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
            estimates: dict | None = None, now: datetime | None = None) -> dict:
    data = data if data is not None else read_ledger()
    validate(data)
    warning = None
    if data["forecasts"] and prices is None:
        prices = _read(PRICES)
        if force or _stale(prices, PRICE_MAX_AGE_SEC) or _missed_close(prices, now or datetime.now(timezone.utc)):
            try:
                prices = refresh_prices(data)
            except Exception as exc:
                warning = f"가격 갱신 실패 · 저장 자료 사용: {type(exc).__name__}: {exc}"
        estimates = _read(ESTIMATES) if estimates is None else estimates
        if force or _stale(estimates, ESTIMATE_MAX_AGE_SEC):
            try:
                estimates = refresh_estimates(data)
            except Exception as exc:
                warning = (warning + " · " if warning else "") + f"전망치 갱신 실패: {type(exc).__name__}"
    items = [{**fc, "result": evaluate(fc, prices, estimates, now)} for fc in data["forecasts"]]
    return {"forecasts": items, "warning": warning,
            "price_source": (prices or {}).get("source"), "prices_fetched_at": (prices or {}).get("fetched_at"),
            "estimates_fetched_at": (estimates or {}).get("fetched_at"),
            "method": "기록 시각을 찍고 해시로 잠근 예측입니다. 기준가는 기록 뒤 첫 미국 정규장 종가이고, "
                      "수익률은 수정 종가(배당 반영), 저점 비교는 종가로 잽니다. 바스켓은 같은 금액으로 사서 들고 있는 수익입니다."}


def main(argv: list[str] | None = None) -> int:
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
            sch = schedule(fc)
            print(f"기록: {fc['id']} · {fc['recorded_at']} · 기준 {sch['baseline']} → 마감 {sch['target']} · sha256 {fc['sha256']}")
        elif args.command == "refresh":
            data = read_ledger()
            snap = refresh_prices(data)
            est = refresh_estimates(data)
            print(f"가격 {len(snap['adj'])}종목 · 전망치 {len(est['values'])}종목 갱신")
        else:
            print(json.dumps(payload(), ensure_ascii=False, indent=2, default=str))
        return 0
    except (ForecastError, OSError, json.JSONDecodeError) as exc:
        print(f"예측 오류: {exc}")
        return 2
