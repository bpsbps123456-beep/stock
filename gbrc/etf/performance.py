"""Two-fund performance by ETF market-price closes (docs/etf_system.md §8).

The winner is decided by D-1/D-3/D-5/D-20 close-to-close returns of the listed ETFs.
AUM and creation flows never pick the winner; a missing price stays N/A.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from gbrc.config import performance_funds

from . import market, prices, storage
from .storage import NORMALIZED_HOLDINGS_DIR

HORIZONS = [("D-1", 1), ("D-3", 3), ("D-5", 5), ("D-20", 20)]


def default_target_date(root: Path = NORMALIZED_HOLDINGS_DIR) -> str | None:
    """Latest date both funds have a snapshot for (else the latest of either)."""
    date_sets = []
    for fund in performance_funds():
        dates = storage.snapshot_dates(fund["snapshot_ticker"], root)
        if not dates:
            return None
        date_sets.append(set(dates))
    common_dates = set.intersection(*date_sets) if date_sets else set()
    if common_dates:
        return sorted(common_dates)[-1]
    all_dates = [d for dates in date_sets for d in dates]
    return sorted(all_dates)[-1] if all_dates else None


def comparison_dates(target, horizons, sessions=None, root: Path = NORMALIZED_HOLDINGS_DIR):
    """Use the same KRX-session D-n and snapshot fallback as the holding report."""
    date_sets = [
        set(storage.snapshot_dates(fund["snapshot_ticker"], root))
        for fund in performance_funds()
    ]
    common_dates = set.intersection(*date_sets) if date_sets else set()
    fallback_dates = set.union(*date_sets) if date_sets else set()
    all_snapshot_dates = sorted(common_dates or fallback_dates)
    out = {}
    for label, n in horizons:
        intended, used = market.select_base_date(all_snapshot_dates, target, n, sessions)
        out[label] = used or intended
    return out


def format_price(value):
    if value is None:
        return "N/A"
    return f"{value:,.0f}"


def performance_report(target_date: str | None = None,
                       root: Path = NORMALIZED_HOLDINGS_DIR) -> str:
    """Compare the two tracked funds by ETF market-price close returns."""
    target = target_date or default_target_date(root)
    if not target:
        return "  No ETF snapshots available for performance comparison."

    funds = performance_funds()
    base_dates = comparison_dates(target, HORIZONS, prices.krx_sessions(target, root), root)
    all_dates = [target] + list(base_dates.values())
    parsed_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in all_dates]
    price_start = (min(parsed_dates) - timedelta(days=7)).strftime("%Y-%m-%d")
    price_end = (max(parsed_dates) + timedelta(days=3)).strftime("%Y-%m-%d")
    close_history = prices.load_close_history(
        [fund["price_ticker"] for fund in funds], price_start, price_end,
    )

    results = {}
    for fund in funds:
        ticker = fund["price_ticker"]
        series = close_history.get(ticker)
        target_point = prices.close_point_on_or_before(series, target)
        fund_results = {}
        for label, base_date in base_dates.items():
            fund_results[label] = {
                "base_date": base_date,
                "base_point": prices.close_point_on_or_before(series, base_date),
                "target_point": target_point,
                "return": prices.close_return_pct(close_history, ticker, base_date, target),
            }
        results[fund["label"]] = fund_results

    tickers = ", ".join(f"{fund['label']}={fund['price_ticker']}" for fund in funds)
    out = [
        "",
        "### [두 펀드 성과 비교] — ETF 시장가격 종가 기준",
        f"  기준일: {target}",
        "  성과 판정은 ETF 가격 수익률 기준이며, AUM/설정현금액 변화는 제외합니다.",
        f"  가격 티커: {tickers}",
        f"  {'구간':<5} {'비교일':<12} {'TIME':>22} {'KoAct':>22}"
        f" {'차이':>10} {'우위':>8}",
        f"  {'-'*88}",
    ]

    for label, _ in HORIZONS:
        base_date = base_dates[label]
        cells = {}
        returns = {}
        for fund in funds:
            fund_result = results.get(fund["label"], {}).get(label, {})
            base_point = fund_result.get("base_point")
            target_point = fund_result.get("target_point")
            ret = fund_result.get("return")
            if base_point and target_point and ret is not None:
                base_actual, base_close = base_point
                target_actual, target_close = target_point
                date_note = "" if str(base_actual) == base_date else f"@{base_actual}"
                target_note = "" if str(target_actual) == target else f"@{target_actual}"
                cells[fund["label"]] = (
                    f"{format_price(base_close)}{date_note}->"
                    f"{format_price(target_close)}{target_note} "
                    f"{ret:+.2f}%"
                )
                returns[fund["label"]] = ret
            else:
                cells[fund["label"]] = "N/A"
                returns[fund["label"]] = None

        if returns.get("TIME") is None or returns.get("KoAct") is None:
            diff_txt = "N/A"
            winner = "N/A"
        else:
            diff = returns["TIME"] - returns["KoAct"]
            diff_txt = f"{diff:+.2f}%p"
            winner = "TIME" if diff > 0 else "KoAct" if diff < 0 else "동률"

        out.append(f"  {label:<5} {base_date:<12} {cells['TIME']:>22}"
                   f" {cells['KoAct']:>22} {diff_txt:>10} {winner:>8}")
    out.append("")
    return "\n".join(out)
