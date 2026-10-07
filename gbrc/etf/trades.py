"""Real-trade timing and holdings-contribution analysis for one fund.

    py gbrc.py etf trades [--fund time] [--end YYYY-MM-DD] [--top 10]

Answers three questions from the validated snapshots alone:
1. When did the fund make money, and did intra-month trading add to it?
2. After the fund really bought or sold (quantity-based), how did the stock do
   — from the manager's valuation close and from the day a follower could act?
3. How would holding the fund's US-listed Top N have done versus the ETF?

Research output only: no target weights, order sizes, or rebalancing tickets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from . import flows as qty_flows
from . import market
from .storage import (  # load_fund_snapshots: desk/ imports it from here too
    PROJECT_ROOT,
    atomic_write_json,
    atomic_write_text,
    load_fund_snapshots,
)

ANALYSIS_DIR = PROJECT_ROOT / "data" / "derived" / "analysis"
BENCHMARK = "QQQ"
FX_USD = "KRW=X"
FORWARD_WINDOWS = (5, 20)
MAJOR_TRADE_PP = 0.25
BUY_KINDS = ("매수", "신규")
SELL_KINDS = ("매도", "편출")

Prices = Mapping[str, Mapping[str, float]]  # ticker -> {YYYY-MM-DD: close}


# --------------------------------------------------------------------------
# 입력
# --------------------------------------------------------------------------

def required_tickers(snapshots: Mapping[str, Mapping[str, Any]], listing_ticker: str) -> set[str]:
    tickers = {BENCHMARK, FX_USD, listing_ticker}
    for as_of, snapshot in snapshots.items():
        for holding in snapshot.get("holdings", []):
            tickers.update(market.yahoo_ticker_candidates(holding.get("code"), as_of))
    return tickers


def download_prices(tickers: Sequence[str], start: str, end: str) -> dict[str, dict[str, float]]:
    """Unadjusted closes, the same basis the NAV uses."""
    import yfinance as yf

    frame = yf.download(sorted(tickers), start=start, end=end, auto_adjust=False,
                        progress=False)["Close"]
    prices: dict[str, dict[str, float]] = {}
    for ticker in sorted(tickers):
        if ticker not in frame.columns:
            continue
        series = frame[ticker].dropna()
        if series.empty:  # 만기 지난 선물 월물 등 Yahoo가 지운 종목
            continue
        prices[ticker] = {idx.strftime("%Y-%m-%d"): float(value) for idx, value in series.items()}
    return prices


# --------------------------------------------------------------------------
# 구간 수익률
# --------------------------------------------------------------------------

def _close(prices: Prices, ticker: str, on_or_before: str) -> float | None:
    point = market.close_on_or_before(prices.get(ticker) or {}, on_or_before)
    return point[1] if point else None


def market_return(code: str, base: str, target: str, prices: Prices,
                  sessions: Sequence[str]) -> float | None:
    """Return between the closes two PDFs reflect, in KRW for USD assets."""
    series = market.series_for(code, target, prices)
    base_point = market.close_on_or_before(series, market.price_reference_date(code, base, sessions))
    target_point = market.close_on_or_before(
        series, market.price_reference_date(code, target, sessions))
    if not base_point or not target_point or base_point[1] <= 0:
        return None
    # 현지 휴장으로 두 PDF가 같은 종가를 가리키면 가격 변화는 0이다.
    ret = 0.0 if target_point[0] <= base_point[0] else target_point[1] / base_point[1] - 1
    if market.listing_market(code) in {"US", "FUT"}:
        fx0 = _close(prices, FX_USD, market.previous_session(base, 1, sessions))
        fx1 = _close(prices, FX_USD, market.previous_session(target, 1, sessions))
        if fx0 and fx1:
            ret = (1 + ret) * fx1 / fx0 - 1
    return ret


def segment_return(prev_row: Mapping[str, Any] | None, next_row: Mapping[str, Any] | None,
                   code: str, base: str, target: str, prices: Prices,
                   sessions: Sequence[str]) -> float | None:
    """One-snapshot-step return: the PDF's own valuation when both days have it."""
    if (prev_row and next_row and prev_row["qty"] > 0 and next_row["qty"] > 0
            and prev_row["value_krw"] > 0):
        split = qty_flows.split_factor(prev_row, next_row)
        before = prev_row["value_krw"] / prev_row["qty"]
        after = next_row["value_krw"] / next_row["qty"] * split
        return after / before - 1
    return market_return(code, base, target, prices, sessions)


@dataclass
class Segment:
    base: str
    target: str
    portfolio_return: float
    contributions: dict[str, float]  # holding key -> contribution (fraction of NAV)
    missing: list[str]


def build_segments(snapshots: Mapping[str, Mapping[str, Any]], prices: Prices,
                   sessions: Sequence[str]) -> list[Segment]:
    """Holdings-based daily return: sum of prior weight x holding return."""
    dates = sorted(snapshots)
    maps = {d: qty_flows.keyed_securities(snapshots[d]) for d in dates}
    segments = []
    for base, target in zip(dates, dates[1:]):
        contributions, missing, total = {}, [], 0.0
        for key, row in maps[base].items():
            ret = segment_return(row, maps[target].get(key), row["code"], base, target,
                                 prices, sessions)
            if ret is None:
                missing.append(key)
                continue
            contribution = row["weight"] / 100.0 * ret
            contributions[key] = contribution
            total += contribution
        segments.append(Segment(base, target, total, contributions, missing))
    return segments


def _compound(values: Sequence[float]) -> float:
    product = 1.0
    for value in values:
        product *= 1 + value
    return product - 1


def monthly_breakdown(snapshots: Mapping[str, Mapping[str, Any]], segments: Sequence[Segment],
                      prices: Prices, sessions: Sequence[str]) -> list[dict[str, Any]]:
    """Month by month: actual (holdings-based) vs month-start basket held unchanged."""
    dates = sorted(snapshots)
    maps = {d: qty_flows.keyed_securities(snapshots[d]) for d in dates}
    by_target = {segment.target: segment for segment in segments}
    months: dict[str, list[str]] = {}
    for d in dates[1:]:
        months.setdefault(d[:7], []).append(d)

    rows = []
    start = dates[0]
    for month, month_dates in months.items():
        end = month_dates[-1]
        path = [d for d in dates if start <= d <= end]
        month_segments = [by_target[d] for d in path[1:]]
        actual = _compound([s.portfolio_return for s in month_segments])

        static = 0.0
        for key, row in maps[start].items():
            code, growth = row["code"], 1.0
            for base, target in zip(path, path[1:]):
                prev_row, next_row = maps[base].get(key), maps[target].get(key)
                if next_row is not None:
                    code = next_row["code"]
                ret = segment_return(prev_row, next_row, code, base, target, prices, sessions)
                growth *= 1 + (ret or 0.0)
            static += row["weight"] / 100.0 * (growth - 1)

        totals: dict[str, float] = {}
        for segment in month_segments:
            for key, value in segment.contributions.items():
                totals[key] = totals.get(key, 0.0) + value
        ranked = sorted(totals.items(), key=lambda item: -item[1])
        rows.append({
            "month": month, "start": start, "end": end,
            "actual_pct": actual * 100, "static_pct": static * 100,
            "trade_effect_pp": (actual - static) * 100,
            "best": [(key, value * 100) for key, value in ranked[:3]],
            "worst": [(key, value * 100) for key, value in ranked[::-1][:3] if value < 0],
        })
        start = end
    return rows


# --------------------------------------------------------------------------
# 실매매 이벤트 성과
# --------------------------------------------------------------------------

def _forward_return(series: Mapping[str, float], start: str | None, sessions_ahead: int,
                    ) -> tuple[str | None, float | None]:
    if not series or not start:
        return None, None
    ordered = sorted(series)
    if start not in series:
        return None, None
    index = ordered.index(start)
    if index + sessions_ahead >= len(ordered):
        return start, None
    return start, series[ordered[index + sessions_ahead]] / series[start] - 1


def _start_on_or_before(series: Mapping[str, float], ref: str) -> str | None:
    point = market.close_on_or_before(series or {}, ref)
    return point[0] if point else None


def _start_on_or_after(series: Mapping[str, float], ref: str) -> str | None:
    later = sorted(d for d in (series or {}) if d >= ref)
    return later[0] if later else None


def collect_events(snapshots: Mapping[str, Mapping[str, Any]], prices: Prices,
                   sessions: Sequence[str]) -> list[dict[str, Any]]:
    """Every day-over-day real trade, with US-listed forward excess returns.

    mgr: from the close the PDF was valued at (about when the manager traded).
    fol: from the first US close on/after the PDF date (when a reader of the
    evening briefing could act).
    """
    dates = sorted(snapshots)
    benchmark = prices.get(BENCHMARK) or {}
    events = []
    for base, target in zip(dates, dates[1:]):
        result = qty_flows.quantity_flows([snapshots[base], snapshots[target]])
        for flow in result.flows.values():
            if flow.kind not in BUY_KINDS + SELL_KINDS:
                continue
            event: dict[str, Any] = {
                "date": target, "key": flow.key, "code": flow.code, "name": flow.name,
                "kind": flow.kind, "selective_pct": flow.selective_pct,
                "trade_pp": flow.trade_pp, "weight": flow.weight,
            }
            series = market.series_for(flow.code, target, prices)
            if market.is_us_listed(flow.code) and series:
                starts = {
                    "mgr": _start_on_or_before(
                        series, market.price_reference_date(flow.code, target, sessions)),
                    "fol": _start_on_or_after(series, target),
                }
                for basis, start in starts.items():
                    for n in FORWARD_WINDOWS:
                        _, stock = _forward_return(series, start, n)
                        bench_start = _start_on_or_before(benchmark, start) if start else None
                        _, bench = _forward_return(benchmark, bench_start, n)
                        event[f"{basis}{n}"] = stock
                        event[f"{basis}{n}_excess"] = (
                            None if stock is None or bench is None else stock - bench
                        )
            events.append(event)
    return events


def event_stats(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for side, kinds in (("매수·신규", BUY_KINDS), ("매도·편출", SELL_KINDS)):
        chosen = [e for e in events if e["kind"] in kinds]
        for n in FORWARD_WINDOWS:
            row: dict[str, Any] = {"side": side, "window": n}
            for basis in ("mgr", "fol"):
                values = [e[f"{basis}{n}_excess"] for e in chosen
                          if e.get(f"{basis}{n}_excess") is not None]
                hits = [v for v in values if (v > 0 if kinds == BUY_KINDS else v < 0)]
                row[f"{basis}_count"] = len(values)
                row[f"{basis}_mean_pct"] = mean(values) * 100 if values else None
                row[f"{basis}_hit_pct"] = len(hits) / len(values) * 100 if values else None
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# 미국 상장 Top N 보유 백테스트
# --------------------------------------------------------------------------

def _max_drawdown(path: Sequence[float]) -> float:
    peak, worst = float("-inf"), 0.0
    for value in path:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return worst


def backtest_top_n(snapshots: Mapping[str, Mapping[str, Any]], prices: Prices, *,
                   top_n: int | None, every: int, weighting: str,
                   end_date: str | None = None) -> dict[str, Any] | None:
    """Hold the US-listed Top N from each rebalance PDF, bought at the first US
    close on/after the PDF date, valued daily in KRW. No costs or taxes."""
    dates = sorted(snapshots)
    rebalances = dates[::every]
    calendar = sorted(d for d in (prices.get(BENCHMARK) or {}) if d >= dates[0])
    if end_date:
        calendar = [d for d in calendar if d <= end_date]
    if not calendar:
        return None
    value, path = 1.0, []
    for index, rebalance in enumerate(rebalances):
        next_rebalance = rebalances[index + 1] if index + 1 < len(rebalances) else None
        window = [d for d in calendar if d >= rebalance
                  and (next_rebalance is None or d <= _first_on_or_after(calendar, next_rebalance))]
        if len(window) < 2:
            continue
        rows = qty_flows.us_listed_top(snapshots[rebalance], top_n or 10_000)
        entry = window[0]
        basket = []
        for row in rows:
            series = market.series_for(row["code"], rebalance, prices)
            if entry in series:
                basket.append((series, row["weight"]))
        if not basket:
            continue
        total = sum(weight for _, weight in basket)
        weights = [(series, (weight / total) if weighting == "fund" else 1 / len(basket))
                   for series, weight in basket]
        fx0 = _close(prices, FX_USD, entry)
        for day in window[1:] if path else window:
            level = 0.0
            for series, weight in weights:
                point = market.close_on_or_before(series, day)
                level += weight * (point[1] / series[entry] if point else 1.0)
            fx1 = _close(prices, FX_USD, day)
            fx_factor = fx1 / fx0 if fx0 and fx1 else 1.0
            path.append((day, value * level * fx_factor))
        value = path[-1][1]
    if not path:
        return None
    levels = [level for _, level in path]
    return {"return_pct": (levels[-1] - 1) * 100, "max_drawdown_pct": _max_drawdown(levels) * 100,
            "start": path[0][0], "end": path[-1][0]}


def _first_on_or_after(calendar: Sequence[str], target: str) -> str:
    later = [d for d in calendar if d >= target]
    return later[0] if later else calendar[-1]


def reference_returns(snapshots: Mapping[str, Mapping[str, Any]], prices: Prices,
                      listing_ticker: str, end_date: str | None = None) -> dict[str, Any]:
    """ETF bought at the first KRX close after the first PDF; QQQ in KRW."""
    first = sorted(snapshots)[0]
    out: dict[str, Any] = {}
    etf = {d: v for d, v in (prices.get(listing_ticker) or {}).items()
           if d > first and (not end_date or d <= end_date)}
    if etf:
        ordered = sorted(etf)
        levels = [etf[d] / etf[ordered[0]] for d in ordered]
        out["etf"] = {"return_pct": (levels[-1] - 1) * 100,
                      "max_drawdown_pct": _max_drawdown(levels) * 100,
                      "start": ordered[0], "end": ordered[-1]}
    bench = {d: v for d, v in (prices.get(BENCHMARK) or {}).items()
             if d >= first and (not end_date or d <= end_date)}
    if bench:
        ordered = sorted(bench)
        fx0 = _close(prices, FX_USD, ordered[0])
        levels = []
        for d in ordered:
            fx1 = _close(prices, FX_USD, d)
            levels.append(bench[d] / bench[ordered[0]] * ((fx1 / fx0) if fx0 and fx1 else 1.0))
        out["qqq_krw"] = {"return_pct": (levels[-1] - 1) * 100,
                          "max_drawdown_pct": _max_drawdown(levels) * 100,
                          "start": ordered[0], "end": ordered[-1]}
    return out


# --------------------------------------------------------------------------
# 보고서
# --------------------------------------------------------------------------

def _pct(value: float | None, digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:+.{digits}f}%"


def _short(key: str) -> str:
    return key.replace(" US EQUITY", "")


def analyze(fund_id: str, display_name: str, listing_ticker: str, end_date: str | None = None,
            top_n: int = 10, *, snapshots: Mapping[str, Mapping[str, Any]] | None = None,
            prices: Prices | None = None,
            price_loader: Callable[[Sequence[str], str, str], Prices] = download_prices,
            ) -> tuple[str, dict[str, Any]]:
    """Return (markdown report, JSON summary)."""
    if snapshots is None:
        snapshots = load_fund_snapshots(fund_id, end_date or "9999-12-31")
    if len(snapshots) < 3:
        raise ValueError("분석하려면 검증 스냅샷이 3개 이상 필요합니다")
    dates = sorted(snapshots)
    end_date = end_date or dates[-1]
    if prices is None:
        start = (date.fromisoformat(dates[0]) - timedelta(days=20)).isoformat()
        stop = (date.today() + timedelta(days=1)).isoformat()
        prices = price_loader(sorted(required_tickers(snapshots, listing_ticker)), start, stop)
    sessions = market.merge_sessions(dates, (prices.get(listing_ticker) or {}).keys())

    segments = build_segments(snapshots, prices, sessions)
    months = monthly_breakdown(snapshots, segments, prices, sessions)
    events = collect_events(snapshots, prices, sessions)
    stats = event_stats(events)
    references = reference_returns(snapshots, prices, listing_ticker, end_date)
    backtests = []
    for every, label in ((20, "월 1회"), (5, "주 1회"), (1, "매일")):
        for n in (5, top_n, None):
            for weighting in ("fund", "equal"):
                result = backtest_top_n(snapshots, prices, top_n=n, every=every,
                                        weighting=weighting, end_date=end_date)
                if result:
                    backtests.append({"rebalance": label, "top_n": n or "미국 전체",
                                      "weighting": weighting, **result})

    totals: dict[str, float] = {}
    for segment in segments:
        for key, value in segment.contributions.items():
            totals[key] = totals.get(key, 0.0) + value * 100
    holding_based = _compound([s.portfolio_return for s in segments]) * 100
    missing = sum(len(s.missing) for s in segments)

    lines = [f"# {display_name} ({fund_id}) 실매매·수익 구간 분석",
             "",
             f"기간 {dates[0]} → {end_date}, 검증 스냅샷 {len(dates)}개. "
             "보유 기반 수익률은 PDF 평가가격(평가액/수량)으로 계산한 운용 수익이며 보수·매매비용 전입니다. "
             "연구용 분석이며 목표 비중이나 주문 수량을 만들지 않습니다.",
             "",
             "## 1. 언제 벌었나 — 월별 보유 기반 수익과 매매 효과",
             "",
             "매매 효과 = 실제(일별 보유 기반) − 월초 보유 종목을 그대로 들고 있었을 때. "
             "양수면 그 달의 사고팔기가 수익을 더했습니다.",
             "",
             "| 월 | 구간 | 실제 | 월초 고정 | 매매 효과 | 기여 상위 | 기여 하위 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in months:
        best = ", ".join(f"{_short(k)} {v:+.1f}%p" for k, v in row["best"]) or "—"
        worst = ", ".join(f"{_short(k)} {v:+.1f}%p" for k, v in row["worst"]) or "—"
        lines.append(f"| {row['month']} | {row['start']}→{row['end']} | {_pct(row['actual_pct'])} | "
                     f"{_pct(row['static_pct'])} | {row['trade_effect_pp']:+.1f}%p | {best} | {worst} |")
    trade_effect_sum = sum(row["trade_effect_pp"] for row in months)
    etf_ref = references.get("etf", {}).get("return_pct")
    lines += ["",
              f"보유 기반 누적 {_pct(holding_based)}, 월별 매매 효과 합계 {trade_effect_sum:+.1f}%p. "
              f"같은 기간 ETF 시장가격 {_pct(etf_ref)}(공개 다음 KRX 거래일 매수 기준). "
              f"가격이 없어 0으로 둔 보유 구간 {missing}건.",
              "",
              "## 2. 실매매 뒤 주가 — 매니저 기준 vs 따라 산 사람 기준",
              "",
              "미국 상장 종목의 선별 수량 ±5% 이상 매매만 셉니다. 초과수익은 같은 기간 QQQ 대비입니다. "
              "매니저 기준은 PDF 평가 종가에서, 추종 기준은 PDF 날짜 이후 첫 미국 종가(저녁 브리핑을 보고 "
              "살 수 있는 시점)에서 셉니다. 적중은 매수 뒤 초과수익 > 0, 매도 뒤 < 0입니다.",
              "",
              "| 구분 | 기간 | 매니저 건수 | 매니저 평균 | 매니저 적중 | 추종 평균 | 추종 적중 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in stats:
        lines.append(f"| {row['side']} | +{row['window']}거래일 | {row['mgr_count']} | "
                     f"{_pct(row['mgr_mean_pct'], 2)} | {_pct(row['mgr_hit_pct'], 0).lstrip('+')} | "
                     f"{_pct(row['fol_mean_pct'], 2)} | {_pct(row['fol_hit_pct'], 0).lstrip('+')} |")

    lines += ["", "## 3. 종목별 누적 기여 (관측 기간, NAV %p)", ""]
    ranked = sorted(totals.items(), key=lambda item: -item[1])
    lines.append("상위: " + ", ".join(f"{_short(k)} {v:+.1f}" for k, v in ranked[:12]))
    lines.append("")
    lines.append("하위: " + ", ".join(f"{_short(k)} {v:+.1f}" for k, v in ranked[::-1][:8]))

    current = snapshots[end_date] if end_date in snapshots else snapshots[dates[-1]]
    top = qty_flows.us_listed_top(current, top_n)
    lines += ["", f"## 4. 현재 미국 상장 Top {top_n}의 주요 실매매 이력", "",
              f"추정 매매 규모 {MAJOR_TRADE_PP:.2f}%p 이상과 신규·편출만 표시합니다. "
              "괄호는 매니저 기준 20거래일 뒤 주가 수익률입니다.", ""]
    for rank, row in enumerate(top, 1):
        history = [e for e in events if e["key"] == row["key"]
                   and (e["kind"] in {"신규", "편출"} or abs(e["trade_pp"] or 0) >= MAJOR_TRADE_PP)]
        rendered = []
        for e in history:
            size = "" if e["selective_pct"] is None else f"{e['selective_pct']:+.0f}%"
            after = "" if e.get("mgr20") is None else f" ({e['mgr20'] * 100:+.0f}%)"
            rendered.append(f"{e['date'][5:]} {e['kind']}{size}{after}")
        lines.append(f"{rank}. **{_short(row['key'])}** {row['weight']:.2f}% — "
                     + ("; ".join(rendered) if rendered else "관측 기간 중 큰 실매매 없음"))

    lines += ["", "## 5. 미국 상장 Top N 보유 백테스트 (원화, 비용·세금 제외)", "",
              "각 교체일 PDF의 미국 상장 상위 N개를 PDF 날짜 이후 첫 미국 종가에 사서 다음 교체일까지 "
              "들고 갑니다. 연구용이며 주문표가 아닙니다.", "",
              "| 교체 | 종목 수 | 비중 | 수익 | 최대 낙폭 |", "| --- | --- | --- | --- | --- |"]
    for bt in backtests:
        weighting = "펀드 비중" if bt["weighting"] == "fund" else "동일 비중"
        lines.append(f"| {bt['rebalance']} | {bt['top_n']} | {weighting} | "
                     f"{_pct(bt['return_pct'])} | {bt['max_drawdown_pct']:.1f}% |")
    for label, key in (("ETF 보유", "etf"), ("QQQ(원화)", "qqq_krw")):
        ref = references.get(key)
        if ref:
            lines.append(f"| {label} | — | — | {_pct(ref['return_pct'])} | "
                         f"{ref['max_drawdown_pct']:.1f}% |")
    lines += ["", "표본은 관측 기간 한 번의 시장 국면입니다. 교체 주기나 종목 수에 따라 결과가 크게 "
              "흔들리면 운의 비중이 큰 것으로 읽습니다."]

    summary = {
        "fund_id": fund_id, "start": dates[0], "end": end_date, "snapshots": len(dates),
        "holding_based_pct": holding_based, "trade_effect_sum_pp": trade_effect_sum,
        "missing_segments": missing, "monthly": months, "event_stats": stats,
        "event_count": len(events), "contributions_pp": dict(ranked),
        "backtests": backtests, "references": references,
    }
    return "\n".join(lines) + "\n", summary


def write_analysis(fund_alias: str, end_date: str | None = None, top_n: int = 10,
                   out_dir: Path = ANALYSIS_DIR) -> tuple[Path, dict[str, Any]]:
    from gbrc.config import resolve_fund

    fund = resolve_fund(fund_alias)
    report, summary = analyze(fund.snapshot_ticker, fund.display_name, fund.listing_ticker,
                              end_date, top_n)
    path = Path(out_dir) / f"{summary['end']}_{fund.snapshot_ticker}_trades.md"
    atomic_write_text(path, report)
    atomic_write_json(path.with_suffix(".json"), summary)
    return path, summary
