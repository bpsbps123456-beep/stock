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
from datetime import date, datetime, timedelta

import numpy as np

from gbrc.etf.storage import NORMALIZED_HOLDINGS_DIR

from . import alerts, config, finviz_basket, finviz_switch, finviz_trend, journal, marketdata, strategies as S
from .bottom_paper import latest_completed

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
    return latest_completed((now or datetime.now(config.NY)) - timedelta(minutes=5))


def _drawdown(values: np.ndarray) -> tuple[float, float]:
    peak = np.maximum.accumulate(np.maximum(values, values[0]))
    dd = values / peak - 1
    return float(dd.min()), float(dd[-1])


def _cap_note(weight: float, raw: float | None, sector: str) -> str | None:
    if raw is None or abs(weight - raw) < 5e-4:
        return None
    if weight > raw:
        return f"다른 종목의 상한 초과분을 배분받아 {raw:.0%}→{weight:.0%}"
    rule = config.RULE
    if abs(weight - rule["stock_cap"]) < 1e-6:
        return f"한 종목 {rule['stock_cap']:.0%} 상한 적용: {raw:.0%}→{weight:.0%}"
    if S.in_semis_budget(sector):
        return f"반도체 합계 {rule['semis_cap']:.0%} 상한 적용: {raw:.0%}→{weight:.0%}"
    return f"같은 섹터 {rule['sector_cap']:.0%} 상한 적용: {raw:.0%}→{weight:.0%}"


class Desk:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.book = None
        self.path = None
        self.key = None
        self.quotes: dict = {}
        self.quotes_at = 0.0
        self.busy = None
        self.capture_attempt = None
        self.capture_issue = None
        self.basket = None
        self.basket_issue = None
        self.last_bar_repair = None
        self.repaired_bars = {}
        self.trend_paper = finviz_trend.Paper()

    # -- cache control ------------------------------------------------------
    def _data_key(self):
        return max((p.stem[:10] for p in NORMALIZED_HOLDINGS_DIR.glob(f"*_{S.TIME_FUND}.json")), default=None)

    def capture_month(self, now):
        with self.lock:
            stamp = time.monotonic()
            if self.capture_attempt is None or stamp - self.capture_attempt >= 300:
                self.capture_attempt = stamp
                try:
                    finviz_basket.capture_due(now=now)
                    self.capture_issue = None
                except finviz_basket.BasketError as exc:
                    self.capture_issue = str(exc)
                self.trend_paper.capture(now)

    def ensure(self, force: bool = False) -> None:
        with self.lock:
            now = datetime.now(config.NY)
            session, _ = config.next_us_session(now)
            self.capture_month(now)
            self.basket_issue = self.capture_issue
            try:
                self.basket = finviz_basket.load_basket(session)
            except finviz_basket.BasketError as exc:
                self.basket, self.basket_issue = None, str(exc)
            extra = {e["ticker"] for e in journal.load_entries() if e.get("ticker")}
            extra.add("SOXX")
            if self.basket:
                extra.update(self.basket["symbols"])
            key = self._data_key(), tuple(sorted(extra))
            completed = last_completed_session(now)
            stale_close = self.book is not None and self.book.days[-1] < completed
            if force or self.book is None or key != self.key or stale_close:
                book = S.load_book(force_prices=force or stale_close, extra=sorted(extra), through=completed)
                path = S.run_model(book)
                self.book, self.path, self.key = book, path, key
                self.last_bar_repair = None
                self.repaired_bars = {}
            if self.basket and (self.last_bar_repair is None or time.monotonic() - self.last_bar_repair >= 300):
                self.last_bar_repair = time.monotonic()
                self.repaired_bars.update(marketdata.repair_basket_bars(self.book, self.basket, completed))

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

    def _decision(self):
        with self.lock:
            self.ensure()
            book, path, basket, basket_issue = self.book, self.path, self.basket, self.basket_issue
            now = datetime.now(config.KST)
            session, open_kst = config.next_us_session(now)
            if basket and basket["effective_date"][:7] != session[:7]:
                basket = None
            tg = S.next_target(book, path, session)
            completed = last_completed_session(now)
            switch = finviz_switch.decision(book, tg, session, completed, basket, basket_issue)
            if self.repaired_bars:
                switch["bar_sources"] = dict(self.repaired_bars)
            weekly_target = {book.ticker[k]: w for k, w in tg["target"].items()}
            try:
                finviz_switch.record_decision(switch, weekly_target, session, now)
            except OSError:
                switch["record_error"] = "이번 신호를 모의 관찰 기록에 저장하지 못했습니다."
            paper = self.trend_paper.observe(book, tg, session, completed, now, weekly_target)
            return book, path, now, session, open_kst, tg, switch, paper

    def maintain_paper(self):
        now = datetime.now(config.NY)
        self.capture_month(now)
        session, _ = config.next_us_session(now)
        filename = session + ".json"
        if not all((folder / filename).exists() for folder in
                   (config.CACHE / "finviz_observations", finviz_trend.observation_dir())):
            if last_completed_session(now) == finviz_switch.market_calendar().previous_session(session).date().isoformat():
                self._decision()

    # -- state ------------------------------------------------------------------
    def state(self, force_quotes: bool = False) -> dict:
        book, path, now, session, open_kst, tg, switch, paper = self._decision()
        settings = journal.load_settings()
        applied = switch["applied"] = settings["finviz_orders"]
        alt = applied and switch["mode"] == "finviz"
        blocked = applied and switch["mode"] == "blocked"
        target = {} if blocked else tg["target"]
        ranked = tg["ranked"]
        tickers = book.ticker
        if alt:
            target = {r["ticker"]: r["weight"] for r in switch["new_targets"]}
            tickers = {tk: tk for tk in target}
        entries = journal.load_entries()
        last_close = {}
        for tk in book.cls.columns:
            col = book.cls[tk].to_numpy(float)
            ok = np.flatnonzero(np.isfinite(col))
            if len(ok):
                last_close[tk] = (book.days[ok[-1]], float(col[ok[-1]]))

        relevant = ({tickers[k] for k in target} | {r["ticker"] for r in ranked[:12]}
                    | {e["ticker"] for e in entries if e.get("ticker")})
        quotes = self._get_quotes(relevant, force_quotes)
        price = {}
        for tk in relevant:
            q = quotes.get(tk)
            price[tk] = q["price"] if q else (last_close.get(tk, (None, None))[1])
        acct = journal.compute(entries, price)
        if applied and switch["order_due"]:
            needed = {tickers[k] for k in target} | {p["ticker"] for p in acct["positions"]}
            missing = []
            for tk in sorted(needed):
                quoted_at = (quotes.get(tk) or {}).get("epoch")
                current = quoted_at and datetime.fromtimestamp(quoted_at, config.NY).date().isoformat() >= switch["signal_date"]
                close_day, closing = last_close.get(tk, (None, None))
                if (not current or not price.get(tk)) and close_day == switch["signal_date"]:
                    price[tk] = closing
                elif not current or not price.get(tk):
                    missing.append(tk)
            acct = journal.compute(entries, price)
            if missing:
                switch = dict(switch, mode="blocked", order_due=False, new_targets=[], new_cash_target=None,
                              reason="주문 평가 가격 확인 대기 · " + ", ".join(missing))
                blocked, alt, target = True, False, {}

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
            tk = tickers[key]
            sector, r = (S.sector_of(key, tk), None) if alt else (book.sector.get(key, S.UNKNOWN), rank.get(key))
            reasons = ([f"120거래일 수익률 {r['mom120']:+.0%} · {r['rank']}위 · TIME 비중 {r['time_weight']:.1%}"]
                       if r else ["수익률 순위에 없음 · 다음 재조정일에 다시 판단합니다"])
            note = _cap_note(wv, tg["raw"].get(key), sector)
            if note:
                reasons.append(note)
            if alt:
                pos = switch["basket_symbols"].index(tk) + 1
                reasons = [f"월말 재무·추세 조건 통과 · 상대거래량 {pos}위 · 월초 한 자리 1/7"]
            q = quotes.get(tk) or {}
            lc = last_close.get(tk)
            rows.append({"key": key, "ticker": tk, "name": book.names.get(key, ""), "weight": wv,
                         "sector": sector, "semis": S.in_semis_budget(sector),
                         "price": price.get(tk), "quote": q, "last_close": lc[1] if lc else None,
                         "last_close_date": lc[0] if lc else None,
                         "change": (price.get(tk) / lc[1] - 1) if lc and price.get(tk) else None,
                         "since": switch["effective_date"] if alt else since.get(key), "reasons": reasons})
        cash_target = None if blocked else max(0.0, 1.0 - sum(target.values()))

        # actions: target weights vs the journal account at current prices
        equity = acct["market_value"] + max(acct["cash"], 0.0)
        tol = float(settings["tolerance"])
        held = {p["ticker"]: p for p in acct["positions"]}
        actions = []
        order_tickers = {r["ticker"] for r in rows} | set(held)
        if applied and not switch["order_due"]:
            order_tickers = set()
        for tk in sorted(order_tickers):
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

        due = switch["order_due"] if applied else tg["rebalance_day"]
        rebalance_on = session if due else config.next_rebalance(session)
        if alt and not due:
            next_month = (date.fromisoformat(session).replace(day=28) + timedelta(days=4)).replace(day=1)
            rebalance_on = finviz_basket.month_dates(next_month)[1]
        return clean({
            "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "busy": self.busy,
            "settings": settings,
            "strategy": "SOXX 100일선 · 주간 TIME / 월간 FINVIZ" if applied else config.STRATEGY_NAME,
            "finviz_switch": switch,
            "finviz_trend_paper": paper,
            "research_comparison": finviz_switch.research_comparison(),
            "rule": dict(config.RULE),
            "session": {"date": session, "label": _label(session),
                        "open_kst": open_kst.strftime("%m월 %d일 %H:%M"),
                        "rebalance_day": due, "next_rebalance": None if blocked else _label(rebalance_on)},
            "status": {"pdf": book.pdfs[-1], "pdf_used": tg["pdf"], "last_close": book.days[-1],
                       "quotes_at": datetime.fromtimestamp(self.quotes_at, config.KST).strftime("%H:%M")
                       if self.quotes_at else None,
                       "warnings": warnings},
            "targets": rows,
            "cash_target": cash_target,
            "actions": actions,
            "rankings": [{"ticker": r["ticker"], "rank": r["rank"], "mom120": r["mom120"],
                          "time_weight": r["time_weight"], "sector": r["sector"],
                          "detail": S.subsector_of(r["key"], r["ticker"]),
                          "semis": S.in_semis_budget(r["sector"]), "in_target": r["key"] in tg["target"]}
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
