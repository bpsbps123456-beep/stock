"""Assemble everything the page shows into one JSON-ready dict.

Heavy work (prices, the model path) is cached per Book and rebuilt only when a new TIME
snapshot or a new US close appears; live quotes refresh every 90 seconds.
"""

from __future__ import annotations

import math
import subprocess
import sys
import threading
import time
from datetime import date, datetime, time as dtime

import numpy as np

from gbrc.etf.storage import NORMALIZED_HOLDINGS_DIR

from . import alerts, config, journal, marketdata, strategies as S

WEEKDAY = "월화수목금토일"


def clean(obj):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _label(day: str) -> str:
    d = date.fromisoformat(day)
    return f"{d.month}월 {d.day}일({WEEKDAY[d.weekday()]})"


def last_completed_session(now: datetime | None = None) -> str:
    now_ny = (now or datetime.now(config.NY)).astimezone(config.NY)
    day = now_ny.date()
    if not (config.is_us_session(day) and now_ny.time() >= dtime(16, 5)):
        day = date.fromordinal(day.toordinal() - 1)
        while not config.is_us_session(day):
            day = date.fromordinal(day.toordinal() - 1)
    return day.isoformat()


def _drawdown(values: np.ndarray) -> tuple[float, float]:
    peak = np.maximum.accumulate(np.maximum(values, values[0]))
    dd = values / peak - 1
    return float(dd.min()), float(dd[-1])


def _cap_note(weight: float, raw: float | None, sector: str) -> str | None:
    if raw is None or abs(weight - raw) < 5e-4:
        return None
    if weight > raw:
        return f"다른 종목의 상한 초과분을 배분받아 {raw:.0%}→{weight:.0%}"
    if abs(weight - config.RULE["stock_cap"]) < 1e-6:
        return f"한 종목 25% 상한 적용: {raw:.0%}→{weight:.0%}"
    if S.in_semis_budget(sector):
        return f"반도체 합계 60% 상한 적용: {raw:.0%}→{weight:.0%}"
    return f"같은 섹터 60% 상한 적용: {raw:.0%}→{weight:.0%}"


class Desk:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.book = None
        self.path = None
        self.key = None
        self.quotes: dict = {}
        self.quotes_at = 0.0
        self.busy = None

    # -- cache control ------------------------------------------------------
    def _data_key(self):
        return max((p.stem[:10] for p in NORMALIZED_HOLDINGS_DIR.glob(f"*_{S.TIME_FUND}.json")), default=None)

    def ensure(self, force: bool = False) -> None:
        with self.lock:
            key = self._data_key()
            stale_close = self.book is not None and self.book.days[-1] < last_completed_session()
            if force or self.book is None or key != self.key or stale_close:
                extra = sorted({e["ticker"] for e in journal.load_entries() if e.get("ticker")})
                self.book = S.load_book(force_prices=force or stale_close, extra=extra)
                self.path = S.run_model(self.book)
                self.key = key

    def _get_quotes(self, tickers: set[str], force: bool) -> dict:
        missing = tickers - set(self.quotes)
        if force or missing or time.time() - self.quotes_at > config.QUOTE_MAX_AGE_SEC:
            fresh = marketdata.live_quotes(sorted(tickers))
            if "_error" not in fresh:
                self.quotes.update(fresh)
                self.quotes_at = time.time()
            else:
                self.quotes["_error"] = fresh["_error"]
        return self.quotes

    # -- actions --------------------------------------------------------------
    def run_briefing(self) -> dict:
        """Run the canonical daily collection (py gbrc.py briefing) and reload."""
        with self.lock:
            self.busy = "브리핑 실행 중"
            try:
                proc = subprocess.run([sys.executable, str(config.ROOT / "gbrc.py"), "briefing"],
                                      cwd=config.ROOT, capture_output=True, text=True,
                                      encoding="utf-8", errors="replace", timeout=900)
                tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
                self.ensure(force=True)
                return {"ok": proc.returncode == 0, "log": tail}
            finally:
                self.busy = None

    # -- state ------------------------------------------------------------------
    def state(self, force_quotes: bool = False) -> dict:
        self.ensure()
        book, path = self.book, self.path
        settings = journal.load_settings()
        now = datetime.now(config.KST)
        session, open_kst = config.next_us_session()
        tg = S.next_target(book, path, session)
        target, ranked = tg["target"], tg["ranked"]
        entries = journal.load_entries()
        last_close = {}
        for tk in book.cls.columns:
            col = book.cls[tk].to_numpy(float)
            ok = np.flatnonzero(np.isfinite(col))
            if len(ok):
                last_close[tk] = (book.days[ok[-1]], float(col[ok[-1]]))

        relevant = ({book.ticker[k] for k in target} | {r["ticker"] for r in ranked[:12]}
                    | {e["ticker"] for e in entries if e.get("ticker")})
        quotes = self._get_quotes(relevant, force_quotes)
        price = {}
        for tk in relevant:
            q = quotes.get(tk)
            price[tk] = q["price"] if q else (last_close.get(tk, (None, None))[1])
        acct = journal.compute(entries, price)

        # model and TIME ETF in KRW; the ETF is read at the KRX close that prices each US day
        ms = S.model_status(book, path, alerts.LONG_WINDOW)
        model, etf, j, mdd, dd_now = ms["model"], ms["etf"], ms["etf_last"], ms["mdd"], ms["dd_now"]

        # target rows with the reason each name is there
        rank = {r["key"]: r for r in ranked}
        since = {}
        for ev in sorted(path.events, key=lambda e: e["date"]):
            if ev["side"] == "buy":
                since[ev["key"]] = ev["date"]
        rows = []
        for key, wv in sorted(target.items(), key=lambda x: -x[1]):
            tk, sector, r = book.ticker[key], book.sector.get(key, S.UNKNOWN), rank.get(key)
            reasons = ([f"120거래일 수익률 {r['mom120']:+.0%} · {r['rank']}위 · TIME 비중 {r['time_weight']:.1%}"]
                       if r else ["수익률 순위에 없음 · 다음 재조정일에 다시 판단합니다"])
            note = _cap_note(wv, tg["raw"].get(key), sector)
            if note:
                reasons.append(note)
            q = quotes.get(tk) or {}
            lc = last_close.get(tk)
            rows.append({"key": key, "ticker": tk, "name": book.names.get(key, ""), "weight": wv,
                         "sector": sector, "semis": S.in_semis_budget(sector),
                         "price": price.get(tk), "quote": q, "last_close": lc[1] if lc else None,
                         "last_close_date": lc[0] if lc else None,
                         "change": (price.get(tk) / lc[1] - 1) if lc and price.get(tk) else None,
                         "since": since.get(key), "reasons": reasons})
        cash_target = max(0.0, 1.0 - sum(target.values()))

        # actions: target weights vs the journal account at current prices
        equity = acct["market_value"] + max(acct["cash"], 0.0)
        tol = float(settings["tolerance"])
        held = {p["ticker"]: p for p in acct["positions"]}
        actions = []
        for tk in sorted({r["ticker"] for r in rows} | set(held)):
            tw = next((r["weight"] for r in rows if r["ticker"] == tk), 0.0)
            px = price.get(tk)
            cur = held.get(tk)
            cur_val = cur["value"] if cur and cur["value"] == cur["value"] else 0.0
            if not px or equity <= 0:
                side = "BUY" if tw > 0 and not cur else ("SELL" if tw == 0 and cur else "HOLD")
                actions.append({"ticker": tk, "side": side, "target_weight": tw, "current_weight": 0.0,
                                "qty": None, "amount": None, "price": px})
                continue
            diff = tw * equity - cur_val
            if tw == 0 and cur:
                side, qty = "SELL", cur["qty"]
            elif abs(diff) < tol * equity:
                side, qty = "HOLD", 0.0
            else:
                side, qty = ("BUY" if diff > 0 else "SELL"), abs(diff) / px
            actions.append({"ticker": tk, "side": side, "target_weight": tw,
                            "current_weight": cur_val / equity, "qty": qty,
                            "whole": math.floor(qty + 1e-9) if qty else 0, "amount": qty * px, "price": px})
        order = {"SELL": 0, "BUY": 1, "HOLD": 2}
        actions.sort(key=lambda a: (order[a["side"]], -a["target_weight"]))

        # data status
        warnings = []
        krx = [d.strftime("%Y-%m-%d") for d in book.etf.index]
        need = max((d for d in krx if d < session), default=None)
        if need and book.pdfs[-1] < need:
            warnings.append(f"{_label(need)} 기준 TIME 보유 내역 PDF가 아직 없습니다. 한국시간 21:45 이후 ‘자료 갱신’을 눌러 주세요.")
        if book.days[-1] < last_completed_session():
            warnings.append("최신 미국장 종가가 아직 반영되지 않았습니다.")
        if "_error" in quotes:
            warnings.append("현재 시세를 불러오지 못해 전일 종가로 계산했습니다.")
        if acct["positions"] and not acct["funded"]:
            warnings.append("입금 기록이 없어 계좌의 현금을 확인할 수 없습니다. 달러 예수금을 ‘입금’으로 먼저 기록하세요.")

        ctx = {"account_drawdown": None, "model_drawdown": dd_now,
               "model_vs_etf_20d": ms["vs_etf_20d"], "model_vs_etf_120d": ms["vs_etf_long"]}
        acct_hist = journal.history(entries, book.cls, book.days, book.fx)
        twr = np.array(acct_hist["twr"], float)
        if acct["funded"] and len(twr) and np.all(np.isfinite(twr) & (twr > 0)):
            ctx["account_drawdown"] = _drawdown(twr)[1]
        try:
            risk = alerts.my_pause_rule(ctx)
        except Exception as exc:  # an edited rule must not take the page down
            risk = [{"level": "warning", "title": "점검 기준을 확인하지 못했습니다", "body": str(exc)}]

        # per-ticker price series with user trades and model entries/exits
        chart_tickers = sorted({r["ticker"] for r in rows} | set(held)
                               | {e["ticker"] for e in entries if e.get("ticker")})
        series = {}
        start = max(0, len(book.days) - 190)
        for tk in chart_tickers:
            if tk not in book.cls.columns:
                continue
            col = book.cls[tk].to_numpy(float)
            first = np.flatnonzero(np.isfinite(col))
            begin = max(start, int(first[0])) if len(first) else start  # recent listings start at their first close
            col = col[begin:]
            keys = [k for k, t in book.ticker.items() if t == tk]
            marks = [{"date": e["date"], "kind": "me_" + e["type"], "price": e["price"], "qty": e["qty"]}
                     for e in entries if e.get("ticker") == tk]
            marks += [{"date": ev["date"], "kind": "model_" + ev["side"], "price": ev["price"]}
                      for ev in path.events if ev["key"] in keys and ev["date"] >= book.days[begin]]
            series[tk] = {"dates": book.days[begin:], "close": col, "marks": marks}

        rebalance_on = session if tg["rebalance_day"] else config.next_rebalance(session)
        return clean({
            "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "busy": self.busy,
            "settings": settings,
            "strategy": config.STRATEGY_NAME,
            "session": {"date": session, "label": _label(session),
                        "open_kst": open_kst.strftime("%m월 %d일 %H:%M"),
                        "rebalance_day": tg["rebalance_day"], "next_rebalance": _label(rebalance_on)},
            "status": {"pdf": book.pdfs[-1], "pdf_used": tg["pdf"], "last_close": book.days[-1],
                       "quotes_at": datetime.fromtimestamp(self.quotes_at, config.KST).strftime("%H:%M")
                       if self.quotes_at else None,
                       "warnings": warnings},
            "targets": rows,
            "cash_target": cash_target,
            "actions": actions,
            "rankings": [{"ticker": r["ticker"], "rank": r["rank"], "mom120": r["mom120"],
                          "time_weight": r["time_weight"], "sector": r["sector"],
                          "semis": S.in_semis_budget(r["sector"]), "in_target": r["key"] in target}
                         for r in ranked[:15]],
            "account": {**acct, "history": acct_hist},
            "journal": sorted(entries, key=lambda e: (e["date"], e.get("created_at", "")), reverse=True),
            "model": {"start": path.days[0], "dates": path.days, "series": {"model": model, "etf": etf},
                      "stats": {"total": model[-1] / 100 - 1, "mdd": mdd, "dd_now": dd_now},
                      "etf_total": etf[j] / 100 - 1, "etf_through": path.days[j], "vs_etf_20d": ms["vs_etf_20d"]},
            "risk": risk,
            "risk_rules": alerts.describe(),
            "series": series,
            "fx": float(book.fx[-1]),
        })
