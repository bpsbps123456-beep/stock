"""One fund's ETF-only daily report (docs/etf_system.md §5–§7).

`fund_report(fund, date)` returns the whole section text the briefing embeds: Top 10,
cash, sectors, quantity-based real trades, the four D-n change lists, the D-20 entry
cluster, the one-glance table of every holding, the interpretation, and the review of
the previous day's reading. The section helpers are pure functions over snapshots;
`cumulative.py` reuses them.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from gbrc.config import FundConfig
from watchlist import resolve_sector

from . import flows as qty_flows
from . import market, prices, storage
from .market import is_cash_holding
from .storage import NORMALIZED_HOLDINGS_DIR

DIFF_THRESHOLD = 0.10  # %p — docs/etf_system.md §5, never changed ad hoc

Snapshots = Mapping[str, Mapping[str, Any]]  # {as_of_date: snapshot}


# --------------------------------------------------------------------------
# Dates and holding identity
# --------------------------------------------------------------------------

def format_short_date(date_str):
    """YYYY-MM-DD → M/D. 출력 표의 신규 편입일 표시에 사용."""
    if not date_str:
        return "?"
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return f"{d.month}/{d.day}"
    except ValueError:
        return date_str


def format_new_marker(entry_date):
    """신규 표시는 반드시 현재 보유 구간의 편입일을 함께 표기한다."""
    return f"신규({format_short_date(entry_date)})"


class EntryDateMap(dict):
    """현재 연속 보유 구간 편입일. 원천 코드 표기가 달라도 같은 보유로 조회한다."""

    def get(self, code, default=None):
        return super().get(market.holding_key(code), default)

    def __getitem__(self, code):
        return super().__getitem__(market.holding_key(code))

    def __contains__(self, code):
        return super().__contains__(market.holding_key(code))


def build_entry_date_map(snapshots: Snapshots, through_date: str | None = None) -> EntryDateMap:
    """보유 종목별 현재 연속 보유 구간의 시작일을 계산.

    선물 롤오버(NQU6→NQZ6)와 원천 대소문자 차이는 같은 보유로 이어 붙인다.
    """
    entry_dates = EntryDateMap()
    prev_keys = set()
    for date_str in sorted(snapshots):
        if through_date and date_str > through_date:
            continue
        snap = snapshots.get(date_str)
        if not snap:
            continue
        curr_keys = {market.holding_key(h.get("code", "")) for h in snap.get("holdings", [])}
        for key in curr_keys:
            if key not in prev_keys:
                dict.__setitem__(entry_dates, key, date_str)
        prev_keys = curr_keys
    return entry_dates


def get_holding_sector(holding):
    """보유 종목을 GBRC 섹터로 분류(정본 watchlist.py)."""
    if is_cash_holding(holding):
        return "현금"
    return resolve_sector(market.listing_key(holding.get("code", "")))


def select_comparison_snapshot(snapshots: Snapshots, target, n, sessions=None):
    """Select the snapshot for D-n, counting actual KRX sessions.

    ``sessions`` 없이 호출하면 스냅샷 날짜를 세션 달력으로 쓴다. 의도한 세션에
    스냅샷이 없으면 그 이전의 가장 가까운 스냅샷으로 물러선다.
    """
    intended, used = market.select_base_date(sorted(snapshots), target, n, sessions)
    if not used:
        return intended, None
    return used, snapshots.get(used)


def _price_reference_date(holding, etf_date, sessions=None):
    """Map an ETF snapshot date to the latest close reflected in that NAV.

    as_of D의 PDF는 직전 KRX 세션(D-1) 마감 기준이다. 국내 종목은 그 세션 종가,
    해외 종목은 그 세션보다 앞선 마지막 현지 종가를 쓴다. 반환값은 on-or-before 조회용이다.
    """
    if not holding or not etf_date:
        return etf_date
    return market.price_reference_date(holding.get("code", ""), etf_date, sessions)


def format_close_return(value):
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def calc_relative_change(prev_weight, curr_weight):
    """이전 비중 대비 상대 변화율(%). 절대 %p 변화와 함께 해석한다."""
    if prev_weight is None or prev_weight == 0:
        return None
    return ((curr_weight - prev_weight) / prev_weight) * 100


def format_relative_change(value):
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def find_cash_holding(snap):
    """스냅샷에서 현금 라인을 찾는다."""
    if not snap:
        return None
    for h in snap.get("holdings", []):
        if is_cash_holding(h):
            return h
    return None


def get_cash_weight(snap):
    cash = find_cash_holding(snap)
    if not cash:
        return None
    return cash.get("weight")


def classify_cash_regime(snapshots: Snapshots, target):
    """최근 3/5개 스냅샷의 현금 연속 증감으로 리스크 상태 판정."""
    cash_series = []
    for date_str in sorted(d for d in snapshots if d <= target):
        weight = get_cash_weight(snapshots[date_str])
        if weight is not None:
            cash_series.append((date_str, weight))

    if len(cash_series) < 4:
        return "중립", "최근 현금 데이터 부족"

    for window in (5, 3):
        if len(cash_series) < window + 1:
            continue
        recent = cash_series[-(window + 1):]
        changes = [
            recent[i + 1][1] - recent[i][1]
            for i in range(len(recent) - 1)
        ]
        start, end = recent[0], recent[-1]
        total = end[1] - start[1]
        if all(change < 0 for change in changes):
            return ("리스크온",
                    f"최근 {window}회 연속 현금 감소 "
                    f"({start[0]} {start[1]:.2f}% → "
                    f"{end[0]} {end[1]:.2f}%, {total:+.2f}%p)")
        if all(change > 0 for change in changes):
            return ("방어 전환",
                    f"최근 {window}회 연속 현금 증가 "
                    f"({start[0]} {start[1]:.2f}% → "
                    f"{end[0]} {end[1]:.2f}%, {total:+.2f}%p)")

    latest = cash_series[-1]
    return "중립/혼조", f"연속 방향성 없음 (최근 현금 {latest[1]:.2f}%)"


def diff_holdings(curr_snap, prev_snap, threshold=DIFF_THRESHOLD,
                  entry_date_map=None, flow_result=None):
    """두 스냅샷 비교 → {increased, decreased, new_in, new_out}

    보유 식별은 holding_key로 한다(선물 월물 롤오버·대소문자·티커 변경을 같은 보유로 본다).
    flow_result를 주면 각 항목에 수량 기준 실매매 판정(flow)을 붙인다.
    """
    curr_map = qty_flows.keyed_securities(curr_snap)
    prev_map = qty_flows.keyed_securities(prev_snap)
    entry_date_map = entry_date_map or {}
    flows = flow_result.flows if flow_result else {}

    increased, decreased, new_in, new_out = [], [], [], []

    for key, curr in curr_map.items():
        if key not in prev_map:
            new_in.append({"code": curr["code"], "name": curr["name"],
                           "weight": curr["weight"],
                           "entry_date": entry_date_map.get(curr["code"]),
                           "flow": flows.get(key)})
            continue
        prev = prev_map[key]
        delta = curr["weight"] - prev["weight"]
        item = {
            "code": curr["code"], "name": curr["name"],
            "prev": prev["weight"], "curr": curr["weight"],
            "delta": delta,
            "relative_pct": calc_relative_change(prev["weight"], curr["weight"]),
            "flow": flows.get(key),
        }
        if delta >= threshold:
            increased.append(item)
        elif delta <= -threshold:
            decreased.append(item)

    for key, prev in prev_map.items():
        if key not in curr_map:
            new_out.append({"code": prev["code"], "name": prev["name"],
                            "weight": prev["weight"], "flow": flows.get(key)})

    increased.sort(key=lambda x: -x["delta"])
    decreased.sort(key=lambda x: x["delta"])
    new_in.sort(key=lambda x: -x["weight"])
    new_out.sort(key=lambda x: -x["weight"])
    return {"increased": increased, "decreased": decreased,
            "new_in": new_in, "new_out": new_out,
            "prev_date": prev_snap["date"], "curr_date": curr_snap["date"],
            "flow_available": flow_result is not None}


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def _change_rows(items, top_n, show_flow):
    header = (f"  {'코드':<18} {'종목명':<35} {'이전':>6} {'현재':>6} {'Δ':>7} {'상대':>8}"
              + (f"  {'실매매(수량)':<14}" if show_flow else ""))
    rows = [header]
    for d in items[:top_n]:
        row = (f"  {d['code']:<18} {d['name'][:33]:<35}"
               f" {d['prev']:>5.2f}% {d['curr']:>5.2f}%"
               f" {d['delta']:>+5.2f}%p"
               f" {format_relative_change(d.get('relative_pct')):>8}")
        if show_flow:
            row += f"  {qty_flows.format_flow_label(d.get('flow'))}"
        rows.append(row)
    return rows


def format_diff_section(diff, title, top_n=15):
    """비교 결과를 사람이 읽는 텍스트로. 매수/매도 각각 top_n개로 제한.

    실매매(수량) 열: 공통 배율·분할을 걷어낸 선별 수량 변화. `가격효과`는 수량이
    그대로인데 가격 때문에 비중만 움직였다는 뜻이다.
    """
    lines = [f"\n### {title}  ({diff['prev_date']} → {diff['curr_date']})"]
    show_flow = bool(diff.get("flow_available"))

    if diff["increased"]:
        total = len(diff["increased"])
        suffix = f" — 상위 {top_n}/{total}개" if total > top_n else f" — 총 {total}개"
        lines.append(f"\n  🔺 비중 증가{suffix}")
        lines.extend(_change_rows(diff["increased"], top_n, show_flow))

    if diff["decreased"]:
        total = len(diff["decreased"])
        suffix = f" — 상위 {top_n}/{total}개" if total > top_n else f" — 총 {total}개"
        lines.append(f"\n  🔻 비중 감소{suffix}")
        lines.extend(_change_rows(diff["decreased"], top_n, show_flow))

    if diff["new_in"]:
        lines.append("\n  🆕 신규 편입")
        lines.append(f"  {'코드':<18} {'종목명':<35} {'편입일':>10} {'비중':>6}")
        for d in diff["new_in"]:
            marker = format_new_marker(d.get("entry_date"))
            lines.append(f"  + {d['code']:<18} {d['name'][:33]:<35}"
                         f" {marker:>10} {d['weight']:>5.2f}%")

    if diff["new_out"]:
        lines.append("\n  ❌ 편출")
        for d in diff["new_out"]:
            lines.append(f"  - {d['code']:<18} {d['name'][:35]:<37}"
                         f" (직전 {d['weight']:>5.2f}%)")

    if not any([diff["increased"], diff["decreased"],
                diff["new_in"], diff["new_out"]]):
        lines.append("\n  (변화 없음 / 임계값 미만)")

    return "\n".join(lines)


def format_quantity_flow_section(curr_snap, flow_results_by_n):
    """수량 기준 실매매: 비중 변화 중 운용사가 실제로 사고판 부분만 분리한다."""
    lines = [f"\n### [수량 기준 실매매] — 기준일 {curr_snap['date']}"]
    lines.append("  PDF 수량은 설정단위(CU)당 주식 수입니다. 전 종목 공통 배율(설정·환매·일괄 조정)과 "
                 "주식분할을 걷어낸 선별 수량 변화가 "
                 f"±{qty_flows.TRADE_THRESHOLD_PCT:.0f}% 이상이면 실매매로 봅니다.")
    lines.append("  추정 매매는 선별 수량 변화가 현재 NAV에서 차지하는 비중(%p)이며, "
                 "신규는 현재 비중, 편출은 직전 비중입니다. 지수선물은 수량이 없어 제외합니다.")
    for n in (1, 3, 5, 20):
        result = flow_results_by_n.get(n)
        if result is None:
            lines.append(f"\n  D-{n}: 비교 스냅샷 없음")
            continue
        common = f"x{result.common_factor:.4f}"
        if not result.common_estimated:
            common += " (일부 구간 추정 불가)"
        lines.append(f"\n  D-{n} ({result.base_date} → {result.current_date}) 공통 배율 {common}")
        groups = [
            ("실매수", result.by_kind("매수")),
            ("실매도", result.by_kind("매도")),
            ("신규", result.by_kind("신규")),
            ("편출", result.by_kind("편출")),
        ]
        if not any(flows for _, flows in groups):
            lines.append(f"  실매매 없음 (선별 수량 변화 ±{qty_flows.TRADE_THRESHOLD_PCT:.0f}% 미만)")
            continue
        for label, flows in groups:
            if not flows:
                continue
            lines.append(f"  {label} {len(flows)}개")
            lines.append(f"    {'코드':<18} {'종목명':<32} {'선별 수량Δ':>10}"
                         f" {'추정 매매':>10} {'현재 비중':>8}")
            for flow in flows:
                selective = ("—" if flow.selective_pct is None
                             else f"{flow.selective_pct:+.1f}%")
                trade = "N/A" if flow.trade_pp is None else f"{flow.trade_pp:+.2f}%p"
                lines.append(f"    {flow.code:<18} {flow.name[:30]:<32} {selective:>10}"
                             f" {trade:>10} {flow.weight:>7.2f}%")
    return "\n".join(lines)


def format_top10_section(curr_snap):
    """최신 증권 비중 Top 10. 현금은 별도 표에서만 분석한다."""
    lines = [f"\n### [현재 비중 Top 10] — 기준일 {curr_snap['date']}"]
    lines.append(f"  {'순위':>3}  {'코드':<18} {'종목명':<38} {'비중':>6}")
    lines.append(f"  {'-'*72}")
    securities = [
        holding for holding in curr_snap["holdings"]
        if not is_cash_holding(holding)
    ]
    top10 = sorted(securities, key=lambda x: -x["weight"])[:10]
    for i, h in enumerate(top10, 1):
        lines.append(f"  {i:>3}  {h['code']:<18} {h['name'][:36]:<38}"
                     f" {h['weight']:>5.2f}%")
    return "\n".join(lines)


def format_cash_section(curr_snap, horizon_snaps, entry_date_map=None,
                        cash_regime=None):
    """현금 보유액은 별도 표로 민감하게 점검."""
    entry_date_map = entry_date_map or {}
    curr_cash = find_cash_holding(curr_snap)
    lines = [f"\n### [현금 보유액 별도 점검] — 기준일 {curr_snap['date']}"]
    lines.append(f"  {'항목':<8} {'현재 평가액':>18} {'비중':>7}"
                 f" {'D-1Δ':>8} {'D-3Δ':>8} {'D-5Δ':>8} {'D-20Δ':>8}")
    lines.append(f"  {'-'*76}")

    if not curr_cash:
        lines.append(f"  {'현금':<8} {'N/A':>18} {'N/A':>7}"
                     f" {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        return "\n".join(lines)

    def fmt_weight_delta(prev_snap):
        if not prev_snap:
            return "N/A"
        prev_cash = find_cash_holding(prev_snap)
        if not prev_cash:
            return format_new_marker(entry_date_map.get(
                curr_cash.get("code", "")))
        delta = curr_cash["weight"] - prev_cash["weight"]
        return f"{delta:+.2f}%p"

    deltas = []
    for n in (1, 3, 5, 20):
        _, prev_snap = horizon_snaps.get(n, (None, None))
        deltas.append(fmt_weight_delta(prev_snap))

    value_krw = curr_cash.get("value_krw")
    value_text = f"{value_krw:,}원" if isinstance(value_krw, int) else "N/A"
    lines.append(f"  {'현금':<8} {value_text:>18} {curr_cash['weight']:>6.2f}%"
                 f" {deltas[0]:>8} {deltas[1]:>8}"
                 f" {deltas[2]:>8} {deltas[3]:>8}")
    if cash_regime:
        label, detail = cash_regime
        lines.append(f"  판정: {label} — {detail}")
    return "\n".join(lines)


def aggregate_sector_weights(snap):
    """스냅샷을 섹터별 비중과 대표 종목으로 집계."""
    totals = {}
    members = {}
    if not snap:
        return totals, members

    for h in snap.get("holdings", []):
        if is_cash_holding(h):
            continue
        sector = get_holding_sector(h)
        totals[sector] = totals.get(sector, 0.0) + h["weight"]
        members.setdefault(sector, []).append(h)

    for holdings in members.values():
        holdings.sort(key=lambda x: -x["weight"])
    return totals, members


def format_sector_section(curr_snap, horizon_snaps, entry_date_map=None):
    """섹터별 현재 비중과 4시계열 변화를 별도 표로 점검."""
    entry_date_map = entry_date_map or {}
    curr_totals, curr_members = aggregate_sector_weights(curr_snap)
    prev_totals_by_horizon = {}
    prev_available = {}

    for n in (1, 3, 5, 20):
        _, prev_snap = horizon_snaps.get(n, (None, None))
        prev_available[n] = prev_snap is not None
        prev_totals_by_horizon[n], _ = aggregate_sector_weights(prev_snap)

    def fmt_delta(sector, n):
        if not prev_available[n]:
            return "N/A"
        prev = prev_totals_by_horizon[n].get(sector)
        if prev is None:
            first_dates = [
                entry_date_map.get(h.get("code", ""))
                for h in curr_members.get(sector, [])
                if entry_date_map.get(h.get("code", ""))
            ]
            return format_new_marker(min(first_dates) if first_dates else None)
        return f"{curr_totals[sector] - prev:+.2f}%p"

    lines = [f"\n### [섹터별 비중/변화 요약] — 기준일 {curr_snap['date']}"]
    lines.append(f"  {'섹터':<18} {'현재':>7}"
                 f" {'D-1Δ':>8} {'D-3Δ':>8} {'D-5Δ':>8} {'D-20Δ':>8}"
                 f"  {'주요 종목':<32}")
    lines.append(f"  {'-'*108}")

    for sector, weight in sorted(curr_totals.items(), key=lambda x: -x[1]):
        top_names = []
        for h in curr_members.get(sector, [])[:3]:
            key = market.listing_key(h.get("code", ""))
            top_names.append(key or h["name"][:10])
        lines.append(f"  {sector:<18} {weight:>6.2f}%"
                     f" {fmt_delta(sector, 1):>8}"
                     f" {fmt_delta(sector, 3):>8}"
                     f" {fmt_delta(sector, 5):>8}"
                     f" {fmt_delta(sector, 20):>8}"
                     f"  {', '.join(top_names)[:32]:<32}")

    return "\n".join(lines)


def format_d20_new_cluster(diff20):
    """D-20 내 신규 편입 종목은 기존 보유 변화와 분리해서 해석한다."""
    lines = ["\n### [D-20 신규 편입 클러스터]"]
    if not diff20 or not diff20.get("new_in"):
        lines.append("  D-20 기준 신규 편입 없음")
        return "\n".join(lines)

    lines.append(f"  {'섹터':<18} {'코드':<18} {'종목명':<32} {'편입일':>10} {'비중':>6}")
    lines.append(f"  {'-'*92}")
    rows = []
    for item in diff20["new_in"]:
        holding = {
            "code": item["code"],
            "name": item["name"],
            "weight": item["weight"],
        }
        rows.append((get_holding_sector(holding), item))

    for sector, item in sorted(rows, key=lambda x: (x[0], -x[1]["weight"])):
        marker = format_new_marker(item.get("entry_date"))
        lines.append(f"  {sector:<18} {item['code']:<18} {item['name'][:30]:<32}"
                     f" {marker:>10} {item['weight']:>5.2f}%")
    return "\n".join(lines)


def top_five_security_weight(snap):
    if not snap:
        return None
    securities = [
        holding for holding in snap.get("holdings", [])
        if not is_cash_holding(holding)
    ]
    return sum(
        holding["weight"]
        for holding in sorted(securities, key=lambda item: -item["weight"])[:5]
    )


def _sector_move_deltas(curr_snap, prev_snap):
    if not curr_snap or not prev_snap:
        return {}
    current, _ = aggregate_sector_weights(curr_snap)
    previous, _ = aggregate_sector_weights(prev_snap)
    return {
        sector: current.get(sector, 0.0) - previous.get(sector, 0.0)
        for sector in set(current) | set(previous)
    }


def format_sector_rotation(curr_snap, prev_snap):
    if not prev_snap:
        return "D-5 비교 스냅샷 없음"
    moves = list(_sector_move_deltas(curr_snap, prev_snap).items())
    increases = sorted(
        (item for item in moves if item[1] >= DIFF_THRESHOLD),
        key=lambda item: -item[1],
    )[:3]
    decreases = sorted(
        (item for item in moves if item[1] <= -DIFF_THRESHOLD),
        key=lambda item: item[1],
    )[:3]

    def render(items):
        return ", ".join(f"{sector} {delta:+.2f}%p" for sector, delta in items)

    increase_text = render(increases) or "유의미한 확대 없음"
    decrease_text = render(decreases) or "유의미한 축소 없음"
    return f"확대: {increase_text}; 축소: {decrease_text}"


def _format_trade_summary(result, limit=5):
    if result is None:
        return "D-5 비교 스냅샷 없음"
    summary = qty_flows.summarize_trades(result)
    parts = []
    for key, label in (("buy", "매수"), ("sell", "매도")):
        names = summary[key]
        shown = ", ".join(names[:limit]) + (" 외" if len(names) > limit else "")
        parts.append(f"{label} {len(names)}" + (f" — {shown}" if names else ""))
    parts.append(f"신규 {len(summary['new'])}, 편출 {len(summary['exited'])}")
    return "; ".join(parts)


def format_portfolio_interpretation(curr_snap, horizon_snaps,
                                    horizon_diffs_by_n, cash_regime,
                                    flow_results_by_n=None):
    """Summarize cash, sector rotation, concentration, entries, and real trades."""
    lines = ["\n### [포트폴리오 해석]"]
    if cash_regime:
        label, detail = cash_regime
        lines.append(f"  현금/위험선호: {label} — {detail}")
    else:
        lines.append("  현금/위험선호: N/A")

    _, prev5 = horizon_snaps.get(5, (None, None))
    lines.append(f"  D-5 섹터 로테이션: {format_sector_rotation(curr_snap, prev5)}")

    current_top5 = top_five_security_weight(curr_snap)
    previous_top5 = top_five_security_weight(prev5)
    if current_top5 is None or previous_top5 is None:
        lines.append("  상위 5개 증권 집중도: N/A")
    else:
        lines.append(
            f"  상위 5개 증권 집중도: {current_top5:.2f}% "
            f"(D-5 {current_top5 - previous_top5:+.2f}%p)"
        )

    diff5 = horizon_diffs_by_n.get(5)
    diff20 = horizon_diffs_by_n.get(20)
    if not diff5:
        lines.append("  신규/편출: D-5 비교 스냅샷 없음")
    else:
        lines.append(
            f"  신규/편출: D-5 신규 {len(diff5['new_in'])}개, "
            f"편출 {len(diff5['new_out'])}개; "
            f"D-20 신규 {len(diff20['new_in']) if diff20 else 'N/A'}개"
        )
    if flow_results_by_n is not None:
        lines.append(
            "  실매매(D-5, 수량 기준): "
            f"{_format_trade_summary(flow_results_by_n.get(5))}"
        )
    return "\n".join(lines)


def format_previous_change_review(curr_snap, snapshots: Snapshots, horizon_snaps, diff1,
                                  sessions=None):
    """Recompute the prior rules-based interpretation and verify the next move."""
    lines = ["\n### [직전 변화 복기]"]
    if not diff1:
        lines.append("  D-1 비교 스냅샷 없음")
        return "\n".join(lines)

    lines.append(
        f"  {diff1['prev_date']} → {diff1['curr_date']}: "
        f"확대 {len(diff1['increased'])}개, 축소 {len(diff1['decreased'])}개, "
        f"신규 {len(diff1['new_in'])}개, 편출 {len(diff1['new_out'])}개"
    )

    def flow_note(item):
        if not diff1.get("flow_available"):
            return ""
        return f" — 실매매: {qty_flows.format_flow_label(item.get('flow'))}"

    if diff1["increased"]:
        item = diff1["increased"][0]
        lines.append(
            f"  최대 확대: {item['code']} {item['delta']:+.2f}%p "
            f"({format_relative_change(item.get('relative_pct'))}){flow_note(item)}"
        )
    else:
        lines.append("  최대 확대: 임계값 이상 없음")
    if diff1["decreased"]:
        item = diff1["decreased"][0]
        lines.append(
            f"  최대 축소: {item['code']} {item['delta']:+.2f}%p "
            f"({format_relative_change(item.get('relative_pct'))}){flow_note(item)}"
        )
    else:
        lines.append("  최대 축소: 임계값 이상 없음")

    prev_date, prev_snap = horizon_snaps.get(1, (None, None))
    if not prev_snap:
        lines.append("  전일 해석 검증: 비교 스냅샷 없음")
        return "\n".join(lines)

    previous_cash_label, _ = classify_cash_regime(snapshots, prev_date)
    current_cash = get_cash_weight(curr_snap)
    previous_cash = get_cash_weight(prev_snap)
    if current_cash is None or previous_cash is None:
        lines.append("  전일 현금 해석 검증: N/A")
    else:
        cash_delta = current_cash - previous_cash
        if previous_cash_label == "리스크온":
            cash_result = "일치" if cash_delta < 0 else "불일치"
        elif previous_cash_label == "방어 전환":
            cash_result = "일치" if cash_delta > 0 else "불일치"
        else:
            cash_result = "판정 유보"
        lines.append(
            f"  전일 현금 해석 검증: {previous_cash_label} → "
            f"다음 날 {cash_delta:+.2f}%p ({cash_result})"
        )

    _, previous_d5 = select_comparison_snapshot(snapshots, prev_date, 5, sessions)
    prior_sector_moves = _sector_move_deltas(prev_snap, previous_d5)
    actual_sector_moves = _sector_move_deltas(curr_snap, prev_snap)
    prior_signals = sorted(
        (
            (sector, delta)
            for sector, delta in prior_sector_moves.items()
            if abs(delta) >= DIFF_THRESHOLD
        ),
        key=lambda item: -abs(item[1]),
    )[:2]
    if not prior_signals:
        lines.append("  전일 섹터 해석 검증: 유의미한 D-5 선행 방향 없음")
    else:
        rendered = []
        for sector, prior_delta in prior_signals:
            actual_delta = actual_sector_moves.get(sector, 0.0)
            result = "일치" if prior_delta * actual_delta > 0 else "불일치"
            rendered.append(
                f"{sector} {prior_delta:+.2f}%p→{actual_delta:+.2f}%p {result}"
            )
        lines.append(f"  전일 섹터 해석 검증: {'; '.join(rendered)}")

    previous_top5 = top_five_security_weight(prev_snap)
    previous_d5_top5 = top_five_security_weight(previous_d5)
    current_top5 = top_five_security_weight(curr_snap)
    if None in (previous_top5, previous_d5_top5, current_top5):
        lines.append("  전일 집중도 해석 검증: N/A")
    else:
        prior_direction = previous_top5 - previous_d5_top5
        actual_direction = current_top5 - previous_top5
        if abs(prior_direction) < DIFF_THRESHOLD:
            concentration_result = "판정 유보"
        else:
            concentration_result = (
                "일치" if prior_direction * actual_direction > 0 else "불일치"
            )
        lines.append(
            f"  전일 집중도 해석 검증: D-5 {prior_direction:+.2f}%p → "
            f"다음 날 {actual_direction:+.2f}%p ({concentration_result})"
        )
    return "\n".join(lines)


def _weights_by_key(snap):
    """All rows including cash (key ''), summed by holding_key."""
    weights = {}
    for holding in snap.get("holdings", []):
        key = market.holding_key(holding.get("code", ""))
        weights[key] = weights.get(key, 0.0) + holding["weight"]
    return weights


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------

def fund_report(fund: FundConfig, target_date: str | None = None,
                root: Path = NORMALIZED_HOLDINGS_DIR) -> str:
    """The fund's ETF-only briefing section for target_date (default: latest snapshot)."""
    fund_id = fund.snapshot_ticker
    dates = storage.snapshot_dates(fund_id, root)
    if not dates:
        return "  No ETF snapshots. Run `py gbrc.py briefing` first."

    out: list[str] = []
    target = target_date or dates[-1]
    curr = storage.load_snapshot(fund_id, target, root)
    if not curr:
        past = [d for d in dates if d <= target]
        if past:
            out.append(f"  ({target} snapshot missing; using {past[-1]})")
            target = past[-1]
            curr = storage.load_snapshot(fund_id, target, root)
    if not curr:
        out.append(f"  No snapshot available on or before {target}.")
        return "\n".join(out)

    out.append(f"\n{'='*100}")
    out.append(f"  {fund.display_name} ({fund_id}) holdings change report - {target}")
    out.append(f"  Holdings {curr['holdings_count']}, total weight "
               f"{sum(h['weight'] for h in curr['holdings']):.2f}%")
    out.append(f"{'='*100}")

    snapshots = storage.load_fund_snapshots(fund_id, target, root)
    sessions = prices.krx_sessions(target, root)
    horizons = [("D-1 compare", 1), ("D-3 compare", 3),
                ("D-5 compare", 5), ("D-20 compare", 20)]
    horizon_snaps = {}
    horizon_diffs = []
    horizon_diffs_by_n = {}
    flow_results_by_n = {}
    entry_date_map = build_entry_date_map(snapshots, target)
    cash_regime = classify_cash_regime(snapshots, target)

    for label, n in horizons:
        prev_date, prev = select_comparison_snapshot(snapshots, target, n, sessions)
        horizon_snaps[n] = (prev_date, prev)
        flow_result = (
            qty_flows.flows_between(snapshots, prev_date, target) if prev else None
        )
        flow_results_by_n[n] = flow_result
        diff = diff_holdings(curr, prev, entry_date_map=entry_date_map,
                             flow_result=flow_result) if prev else None
        horizon_diffs.append((label, diff))
        horizon_diffs_by_n[n] = diff

    out.append(format_top10_section(curr))
    out.append(format_cash_section(curr, horizon_snaps, entry_date_map, cash_regime))
    out.append(format_sector_section(curr, horizon_snaps, entry_date_map))
    out.append(format_quantity_flow_section(curr, flow_results_by_n))

    for label, diff in horizon_diffs:
        if not diff:
            out.append(f"\n### {label}: no comparison snapshot")
            continue
        out.append(format_diff_section(diff, label))

    out.append(format_d20_new_cluster(horizon_diffs_by_n.get(20)))

    out.append(f"\n{'='*146}")
    out.append("  [One-glance table] all current holdings x "
               "D-1/D-3/D-5/D-20 weight change + close return")
    out.append(f"{'='*146}")

    horizon_maps = {}
    horizon_available = {}
    base_dates = []
    for n in (1, 3, 5, 20):
        snap_date, snap = horizon_snaps.get(n, (None, None))
        horizon_available[n] = snap is not None
        horizon_maps[n] = _weights_by_key(snap) if snap else {}
        base_dates.append(f"D-{n}={snap_date or 'N/A'}")
    out.append(f"  (comparison dates: {' / '.join(base_dates)})")
    out.append("  dW is weight change in percentage points; return is "
               "close-to-close from that comparison date to current date.")
    out.append("  Price dates follow the NAV valuation: KR = previous KRX session close, "
               "foreign = last local close before it.")

    all_dates = [target] + [
        snap_date for snap_date, _ in horizon_snaps.values() if snap_date
    ]
    parsed_dates = [datetime.strptime(d, "%Y-%m-%d").date()
                    for d in all_dates]
    price_start = (min(parsed_dates) - timedelta(days=12)).strftime("%Y-%m-%d")
    price_end = (max(parsed_dates) + timedelta(days=3)).strftime("%Y-%m-%d")

    candidates_by_code = {
        h["code"]: market.yahoo_ticker_candidates(h.get("code"), target)
        for h in curr.get("holdings", [])
    }
    current_holding_by_code = {
        h["code"]: h
        for h in curr.get("holdings", [])
    }
    close_history = prices.load_close_history(
        [ticker for tickers in candidates_by_code.values() for ticker in tickers],
        price_start, price_end,
    )
    # 만기 지난 선물 월물은 Yahoo에서 사라지므로 연속물(NQ=F)로 대체한다.
    market_ticker_by_code = {
        code: next((t for t in tickers if t in close_history),
                   tickers[0] if tickers else None)
        for code, tickers in candidates_by_code.items()
    }

    def fmt_delta_return(curr_w, prev_w, available, code=None, base_date=None):
        if not available:
            return "N/A".rjust(18)
        entry_date = entry_date_map.get(code)
        if prev_w is None or (entry_date and base_date and base_date < entry_date):
            return format_new_marker(entry_date).rjust(18)

        delta = curr_w - prev_w
        holding = current_holding_by_code.get(code)
        base_price_date = _price_reference_date(holding, base_date, sessions)
        target_price_date = _price_reference_date(holding, target, sessions)
        ret = prices.close_return_pct(close_history, market_ticker_by_code.get(code),
                                      base_price_date, target_price_date)
        return f"{delta:+.2f}({format_close_return(ret)})".rjust(18)

    out.append(f"  {'Rank':>3}  {'Code':<18} {'Name':<32} {'Weight':>7}"
               f" {'D-1 dW(ret)':>18} {'D-3 dW(ret)':>18}"
               f" {'D-5 dW(ret)':>18} {'D-20 dW(ret)':>18}")
    out.append(f"  {'-'*146}")

    for i, h in enumerate(sorted(curr["holdings"], key=lambda x: -x["weight"]), 1):
        key = market.holding_key(h["code"])
        cells = []
        for n in (1, 3, 5, 20):
            cells.append(fmt_delta_return(
                h["weight"], horizon_maps[n].get(key), horizon_available[n],
                h["code"], horizon_snaps.get(n, (None, None))[0],
            ))
        out.append(f"  {i:>3}  {h['code']:<18} {h['name'][:30]:<32}"
                   f" {h['weight']:>6.2f}% {' '.join(cells)}")
    out.append("")
    out.append(format_portfolio_interpretation(
        curr, horizon_snaps, horizon_diffs_by_n, cash_regime, flow_results_by_n,
    ))
    out.append(format_previous_change_review(
        curr, snapshots, horizon_snaps, horizon_diffs_by_n.get(1), sessions,
    ))
    out.append("")
    return "\n".join(out)
