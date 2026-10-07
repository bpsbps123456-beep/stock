"""'내 7종목 점검' — the daily briefing's view of this week's 7 stocks.

Names, momentum ranks and the funds' quantity-based trades only: no weights or share
counts, because orders stay in the desk (docs/desk_system.md §1). gbrc's daily workflow
puts this block first under '## 핵심 결론' on today's run; `py -m desk digest` prints it.

Ranks are those the next weekly rebalance would use (latest TIME PDF dated before it,
latest US closes). D-5/D-20 trades reuse the briefing's own session calendar and base
dates, so they match the report's consensus lines.
"""

from __future__ import annotations

from gbrc.etf import market
from gbrc.etf.flows import flows_between, format_flow_label

from . import alerts, config, strategies as S
from .state import _label

KOACT_FUND = "2ETFQ1"
HOLD_LINE = config.RULE["n"] + config.RULE["buffer"]  # 12: last week's names stay inside it


def holding_notes(rank: int | None, time5: str, time20: str, koact20: str) -> list[str]:
    notes = []
    if rank is None:
        notes.append("TIME 목록에서 빠짐 → 다음 재조정에서 매도")
    elif rank > HOLD_LINE:
        notes.append(f"유지선({HOLD_LINE}위) 밖 → 다음 재조정에서 교체")
    elif rank >= HOLD_LINE - 1:
        notes.append(f"유지선({HOLD_LINE}위) 근접")
    if time5.startswith("편출") or time20.startswith("편출"):
        notes.append("TIME 편출")
    if time20.startswith("매도") and koact20.startswith(("매도", "편출")):
        notes.append("두 펀드 모두 D-20 매도")
    return notes


def rebalance_changes(preview: list[str], held: set[str]) -> tuple[list[str], list[str]]:
    """Names that would enter and leave if the weekly rebalance ran on these ranks."""
    return [k for k in preview if k not in held], sorted(k for k in held if k not in preview)


def _flow_labels(engine, fund: str, target: str, sessions, n: int) -> dict[str, str]:
    snaps = engine.fund_snapshots(fund, target)
    if target not in snaps:
        return {}
    _, used = market.select_base_date(sorted(snaps), target, n, sessions)
    result = flows_between(snaps, used, target)
    if result is None:
        return {}
    return {S.research_key(flow.code): format_flow_label(flow) for flow in result.flows.values()}


def briefing_block(engine=None) -> list[str]:
    if engine is None:
        from gbrc.etf.engine import BriefingEngine  # the validated snapshots the briefing uses
        engine = BriefingEngine()
    book = S.load_book()
    path = S.run_model(book)
    session, _ = config.next_us_session()
    tg = S.next_target(book, path, session)
    rebalance_on = session if tg["rebalance_day"] else config.next_rebalance(session)
    pdf, ranked = S.momentum_ranking(book, len(book.days), rebalance_on)
    rank = {r["key"]: r for r in ranked}
    held = set(path.selected)
    preview = S.select(ranked, held)
    entering, leaving = rebalance_changes(preview, held)

    latest = book.pdfs[-1]
    sessions = engine.krx_sessions(latest)
    t5 = _flow_labels(engine, S.TIME_FUND, latest, sessions, 5)
    t20 = _flow_labels(engine, S.TIME_FUND, latest, sessions, 20)
    k20 = _flow_labels(engine, KOACT_FUND, latest, sessions, 20)

    when = "오늘 밤" if tg["rebalance_day"] else _label(rebalance_on)
    lines = [f"### [내 7종목 점검] — 순위 기준 TIME PDF {pdf} · 다음 재조정 {when}",
             "  주문 수량은 야간 데스크가 계산합니다. 운용사 매매는 참고 신호이며, 교체는 규칙대로 재조정일에만 합니다.",
             "",
             "| 종목 | 모멘텀 순위 | TIME 실매매 D-5 | TIME D-20 | KoAct D-20 | 메모 |",
             "|---|---:|---|---|---|---|"]
    for key in sorted(held, key=lambda k: rank[k]["rank"] if k in rank else 10 ** 6):
        r = rank.get(key)
        cells = (t5.get(key, "—"), t20.get(key, "—"), k20.get(key, "보유 안 함"))
        notes = holding_notes(r["rank"] if r else None, *cells)
        lines.append(f"| {book.ticker[key]} | {r['rank'] if r else '—'} | {' | '.join(cells)} | {' · '.join(notes)} |")
    lines.append("")
    waiting = [r for r in ranked[:config.RULE["n"]] if r["key"] not in held]
    if waiting:
        lines.append("  대기(7위 안이지만 자리 없음): "
                     + ", ".join(f"{r['ticker']} {r['rank']}위({r['sector']})" for r in waiting))
    verb = "오늘 밤 재조정" if tg["rebalance_day"] else "지금 순위로 재조정하면"
    if entering or leaving:
        lines.append(f"  {verb}: 편입 {', '.join(book.ticker[k] for k in entering) or '없음'} / "
                     f"편출 {', '.join(book.ticker[k] for k in leaving) or '없음'}")
    else:
        lines.append(f"  {verb}: 변동 없음")
    ms = S.model_status(book, path, alerts.LONG_WINDOW)
    risk = alerts.my_pause_rule({"account_drawdown": None, "model_drawdown": ms["dd_now"],
                                 "model_vs_etf_20d": ms["vs_etf_20d"], "model_vs_etf_120d": ms["vs_etf_long"]})
    gap = "—" if ms["vs_etf_20d"] is None else f"{ms['vs_etf_20d']:+.1%}p"
    lines.append(f"  7종목 모델: 고점 대비 {ms['dd_now']:+.1%} · 최근 20거래일 TIME ETF 대비 {gap} · 멈춤 규칙 "
                 + (" / ".join(a["title"] for a in risk) if risk else "경고 없음"))
    return lines
