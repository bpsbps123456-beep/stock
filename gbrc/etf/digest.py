"""Core-conclusion digest across the two funds (docs/etf_system.md §5).

`[두 펀드 실매매 합의]`: names both funds really bought or sold (quantity-based) over
D-5 and D-20, and where they disagreed. `[<기준 펀드> 미국 상장 Top 10 관찰군]`: the
observation fund's US-listed Top 10 with each name's real trades — a research list,
not an order ticket.
"""

from __future__ import annotations

from pathlib import Path

from gbrc.config import load_fund_configs, observation_primary_alias

from . import flows as qty_flows
from . import market, prices, storage
from .report import build_entry_date_map, format_short_date
from .storage import NORMALIZED_HOLDINGS_DIR


def fund_label(alias: str) -> str:
    return "TIME" if alias == "time" else "KoAct"


def _short_code(code):
    return (code or "").replace(" US EQUITY", "")


def _format_consensus_item(company, flows_by_label):
    parts = [
        f"{label} {qty_flows.format_flow_label(flow)}"
        for label, flow in flows_by_label
        if flow is not None
    ]
    return f"{_short_code(company)} ({', '.join(parts)})"


def core_digest(target_date: str, root: Path = NORMALIZED_HOLDINGS_DIR) -> str:
    """Both funds' real-trade consensus and the US-listed watch list for target_date."""
    out: list[str] = []
    funds = load_fund_configs()
    sessions = prices.krx_sessions(target_date, root)
    primary = observation_primary_alias()
    secondary = next(alias for alias in funds if alias != primary)
    labels = {alias: fund_label(alias) for alias in (primary, secondary)}
    snapshots = {
        alias: storage.load_fund_snapshots(funds[alias].snapshot_ticker, target_date, root)
        for alias in (primary, secondary)
    }
    missing = [labels[a] for a in snapshots if target_date not in snapshots[a]]
    if missing:
        return f"  {', '.join(missing)} {target_date} 스냅샷이 없어 교차 요약을 만들지 않았습니다."

    results = {}
    for alias, snaps in snapshots.items():
        dates = sorted(snaps)
        for n in (5, 20):
            _, used = market.select_base_date(dates, target_date, n, sessions)
            results[(alias, n)] = qty_flows.flows_between(snaps, used, target_date)

    out.append("### [두 펀드 실매매 합의] — 수량 기준(공통 배율·분할 제거)")
    out.append("  두 펀드가 같은 종목을 실제로 함께 사고판 경우와 방향이 엇갈린 경우입니다. "
               "비중 변화가 아니라 수량 변화로 판정합니다.")
    for n in (5, 20):
        a, b = results[(primary, n)], results[(secondary, n)]
        if a is None or b is None:
            out.append(f"  D-{n}: 비교 스냅샷 부족")
            continue
        out.append(f"  D-{n} ({labels[primary]} {a.base_date}→{a.current_date}, "
                   f"{labels[secondary]} {b.base_date}→{b.current_date})")
        buckets = qty_flows.cross_fund_consensus(a, b)
        for key, title in (("common_buy", "공통 실매수"), ("common_sell", "공통 실매도"),
                           ("divergence", "디버전스")):
            items = buckets[key]
            rendered = ", ".join(
                _format_consensus_item(company, [(labels[primary], fa), (labels[secondary], fb)])
                for company, fa, fb in items
            )
            out.append(f"    {title} {len(items)}: {rendered or '없음'}")

    primary_snaps = snapshots[primary]
    primary_dates = sorted(primary_snaps)
    current = primary_snaps[target_date]
    entry_map = build_entry_date_map(primary_snaps, target_date)
    top = qty_flows.us_listed_top(current, 10)
    secondary_rows = {
        market.company_key(row["key"]): row
        for row in qty_flows.keyed_securities(snapshots[secondary][target_date]).values()
    }
    flows5 = results[(primary, 5)]
    flows20 = results[(primary, 20)]
    other20 = results[(secondary, 20)]
    other20_by_company = (
        {market.company_key(k): f for k, f in other20.flows.items()} if other20 else {}
    )

    def flow_cell(result, key):
        if result is None:
            return "N/A"
        return qty_flows.format_flow_label(result.flows.get(key))

    out.append("")
    out.append(f"### [{labels[primary]} 미국 상장 Top 10 관찰군] — 기준일 {target_date}")
    out.append(f"  조사 우선순위 목록이며 주문표가 아닙니다. 비중은 {labels[primary]}의 NAV 비중, "
               "실매매는 수량 기준입니다.")
    out.append(f"  {'순위':>3}  {'종목':<8} {labels[primary] + ' 비중':>10} {'실매매 D-5':<14}"
               f" {'실매매 D-20':<14} {'편입일':<9} {labels[secondary] + ' 비중':>11}"
               f" {labels[secondary] + ' D-20':<14}")
    for rank, row in enumerate(top, 1):
        entry = entry_map.get(row["code"])
        entry_text = format_short_date(entry) if entry else "?"
        if entry and primary_dates and entry == primary_dates[0]:
            entry_text = "≤" + entry_text
        other = secondary_rows.get(market.company_key(row["key"]))
        other_weight = f"{other['weight']:.2f}%" if other else "미보유"
        other_flow = (
            qty_flows.format_flow_label(other20_by_company.get(market.company_key(row["key"])))
            if other else "—"
        )
        out.append(f"  {rank:>3}  {_short_code(row['code']):<8} {row['weight']:>9.2f}%"
                   f" {flow_cell(flows5, row['key']):<14} {flow_cell(flows20, row['key']):<14}"
                   f" {entry_text:<9} {other_weight:>11} {other_flow:<14}")
    out.append(f"  Top 10 비중 합계 {sum(row['weight'] for row in top):.2f}%")

    for n in (5, 20):
        result = results[(primary, n)]
        if result is None:
            continue
        before = {row["key"] for row in qty_flows.us_listed_top(primary_snaps[result.base_date], 10)}
        after = {row["key"] for row in top}
        entered = ", ".join(sorted(_short_code(k) for k in after - before)) or "없음"
        left = ", ".join(sorted(_short_code(k) for k in before - after)) or "없음"
        out.append(f"  Top 10 교체(D-{n}, {result.base_date} 대비): 진입 {entered} / 이탈 {left}")
    return "\n".join(out)
