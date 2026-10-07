"""Quantity-based trade detection for ETF PDF snapshots.

PDF quantities are shares per creation unit (CU).  Creation/redemption flows
and pro-rata funding sales scale every holding by one common factor on the
same day, and a stock split multiplies one holding's quantity while its price
moves inversely.  Neither is a stock-specific decision.  What remains after
removing both is the manager's selective trade.  Weights cannot show this:
a weight rises when the price rises even if not a single share was bought.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .market import company_key, holding_key, is_cash_holding, is_us_listed

TRADE_THRESHOLD_PCT = 5.0  # 선별 수량 변화 ±5% 이상만 실매매로 본다.
MIN_COMMON_SAMPLE = 5  # 공통 배율은 양일 보유 종목이 이 수 이상일 때만 추정한다.
_SPLIT_QTY_TOLERANCE = math.log(1.05)
_SPLIT_PRICE_TOLERANCE = math.log(1.35)

BUY_KINDS = frozenset({"매수", "신규"})
SELL_KINDS = frozenset({"매도", "편출"})


@dataclass(frozen=True)
class QuantityFlow:
    key: str
    code: str
    name: str
    status: str  # held | new | exited
    weight: float  # current NAV weight, 0 when exited
    prev_weight: float  # base NAV weight, 0 when new
    selective_pct: float | None  # held only; None when quantities are unavailable
    split_factor: float
    trade_pp: float | None  # estimated traded NAV weight in %p
    kind: str  # 매수 | 매도 | 유지 | 신규 | 편출 | N/A


def keyed_securities(snapshot: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Non-cash holdings keyed by holding_key; same-key rows are merged."""
    merged: dict[str, dict[str, Any]] = {}
    if not snapshot:
        return merged
    for holding in snapshot.get("holdings", []):
        if is_cash_holding(holding):
            continue
        key = holding_key(holding.get("code"))
        row = merged.get(key)
        if row is None:
            merged[key] = {
                "key": key,
                "code": holding.get("code", ""),
                "name": holding.get("name", ""),
                "weight": float(holding.get("weight") or 0.0),
                "value_krw": float(holding.get("value_krw") or 0.0),
                "qty": float(holding.get("qty") or 0.0),
            }
            continue
        if float(holding.get("weight") or 0.0) > row["weight"]:
            row["code"], row["name"] = holding.get("code", ""), holding.get("name", "")
        row["weight"] += float(holding.get("weight") or 0.0)
        row["value_krw"] += float(holding.get("value_krw") or 0.0)
        row["qty"] += float(holding.get("qty") or 0.0)
    return merged


def split_factor(prev: Mapping[str, Any], curr: Mapping[str, Any]) -> float:
    """Return k for a k-for-1 split (1/k for a reverse split), else 1.0."""
    q0, q1 = float(prev.get("qty") or 0), float(curr.get("qty") or 0)
    v0, v1 = float(prev.get("value_krw") or 0), float(curr.get("value_krw") or 0)
    if min(q0, q1, v0, v1) <= 0:
        return 1.0
    qty_ratio = q1 / q0
    price_ratio = (v1 / q1) / (v0 / q0)
    whole = round(qty_ratio) if qty_ratio >= 1 else round(1 / qty_ratio)
    if whole < 2:
        return 1.0
    factor = float(whole) if qty_ratio >= 1 else 1.0 / whole
    if abs(math.log(qty_ratio / factor)) > _SPLIT_QTY_TOLERANCE:
        return 1.0
    if abs(math.log(price_ratio * factor)) > _SPLIT_PRICE_TOLERANCE:
        return 1.0
    return factor


def daily_common_factor(
    prev_map: Mapping[str, Mapping[str, Any]],
    curr_map: Mapping[str, Mapping[str, Any]],
) -> float | None:
    """Median quantity ratio of holdings held on both days (splits excluded)."""
    ratios = []
    for key, curr in curr_map.items():
        prev = prev_map.get(key)
        if not prev or prev["qty"] <= 0 or curr["qty"] <= 0:
            continue
        if split_factor(prev, curr) != 1.0:
            continue
        ratios.append(curr["qty"] / prev["qty"])
    if len(ratios) < MIN_COMMON_SAMPLE:
        return None
    return median(ratios)


def classify_selective(selective_pct: float | None) -> str:
    if selective_pct is None:
        return "N/A"
    if selective_pct >= TRADE_THRESHOLD_PCT:
        return "매수"
    if selective_pct <= -TRADE_THRESHOLD_PCT:
        return "매도"
    return "유지"


@dataclass(frozen=True)
class FlowResult:
    base_date: str
    current_date: str
    common_factor: float  # chained creation/redemption factor
    common_estimated: bool  # False when some day lacked enough holdings
    flows: dict[str, QuantityFlow]

    def by_kind(self, *kinds: str) -> list[QuantityFlow]:
        selected = [flow for flow in self.flows.values() if flow.kind in kinds]
        return sorted(selected, key=lambda flow: -abs(flow.trade_pp or 0.0))


def quantity_flows(path: Sequence[Mapping[str, Any]]) -> FlowResult:
    """Compare the first and last snapshot of a chronological path.

    Every snapshot between them is used to chain the daily common factor and
    split factors and to detect holdings that left and came back.
    """
    if len(path) < 2:
        raise ValueError("quantity_flows needs at least two snapshots")
    maps = [keyed_securities(snapshot) for snapshot in path]
    base_map, curr_map = maps[0], maps[-1]

    common, estimated = 1.0, True
    for prev_map, next_map in zip(maps, maps[1:]):
        factor = daily_common_factor(prev_map, next_map)
        if factor is None:
            estimated = False
            continue
        common *= factor

    flows: dict[str, QuantityFlow] = {}
    for key in set(base_map) | set(curr_map):
        base, curr = base_map.get(key), curr_map.get(key)
        continuous = all(key in snapshot_map for snapshot_map in maps)
        if curr and (not base or not continuous):
            flows[key] = QuantityFlow(
                key, curr["code"], curr["name"], "new", curr["weight"],
                base["weight"] if base else 0.0, None, 1.0, curr["weight"], "신규",
            )
            continue
        if base and not curr:
            flows[key] = QuantityFlow(
                key, base["code"], base["name"], "exited", 0.0, base["weight"],
                None, 1.0, -base["weight"], "편출",
            )
            continue

        splits = 1.0
        for prev_map, next_map in zip(maps, maps[1:]):
            splits *= split_factor(prev_map[key], next_map[key])
        selective = None
        trade_pp = None
        if base["qty"] > 0 and curr["qty"] > 0:
            ratio = curr["qty"] / base["qty"] / (common * splits)
            selective = (ratio - 1.0) * 100.0
            trade_pp = curr["weight"] * (ratio - 1.0) / ratio
        flows[key] = QuantityFlow(
            key, curr["code"], curr["name"], "held", curr["weight"], base["weight"],
            selective, splits, trade_pp, classify_selective(selective),
        )
    return FlowResult(
        str(path[0].get("as_of_date") or path[0].get("date")),
        str(path[-1].get("as_of_date") or path[-1].get("date")),
        common,
        estimated,
        flows,
    )


def format_flow_label(flow: QuantityFlow | None) -> str:
    """Short label for change tables: 매수 +26%, 매도 -21%, 가격효과, 신규 …"""
    if flow is None or flow.kind == "N/A":
        return "N/A"
    split_note = "·분할" if flow.split_factor != 1.0 else ""
    if flow.kind in {"매수", "매도"}:
        return f"{flow.kind} {flow.selective_pct:+.0f}%{split_note}"
    if flow.kind == "유지":
        return f"가격효과{split_note}"
    return flow.kind


def cross_fund_consensus(
    primary: FlowResult,
    secondary: FlowResult,
) -> dict[str, list[tuple[str, QuantityFlow | None, QuantityFlow | None]]]:
    """Group companies traded by both funds in the same or opposite direction."""
    by_company_a = {company_key(flow.key): flow for flow in primary.flows.values()}
    by_company_b = {company_key(flow.key): flow for flow in secondary.flows.values()}
    buckets: dict[str, list[tuple[str, QuantityFlow | None, QuantityFlow | None]]] = {
        "common_buy": [],
        "common_sell": [],
        "divergence": [],
    }
    for company in sorted(set(by_company_a) | set(by_company_b)):
        a, b = by_company_a.get(company), by_company_b.get(company)
        kind_a = a.kind if a else None
        kind_b = b.kind if b else None
        if kind_a in BUY_KINDS and kind_b in BUY_KINDS:
            buckets["common_buy"].append((company, a, b))
        elif kind_a in SELL_KINDS and kind_b in SELL_KINDS:
            buckets["common_sell"].append((company, a, b))
        elif (kind_a in BUY_KINDS and kind_b in SELL_KINDS) or (
            kind_a in SELL_KINDS and kind_b in BUY_KINDS
        ):
            buckets["divergence"].append((company, a, b))

    def size(item: tuple[str, QuantityFlow | None, QuantityFlow | None]) -> float:
        return -sum(abs(flow.trade_pp or 0.0) for flow in item[1:] if flow)

    for items in buckets.values():
        items.sort(key=size)
    return buckets


def us_listed_top(snapshot: Mapping[str, Any], n: int = 10) -> list[dict[str, Any]]:
    """US-listed securities ranked by NAV weight (the user-facing watch list)."""
    rows = [row for row in keyed_securities(snapshot).values() if is_us_listed(row["code"])]
    rows.sort(key=lambda row: (-row["weight"], row["key"]))
    return rows[:n]


def path_between(
    snapshots: Mapping[str, Mapping[str, Any]],
    base_date: str,
    current_date: str,
) -> list[Mapping[str, Any]]:
    """Chronological snapshots from base_date to current_date inclusive."""
    return [snapshots[d] for d in sorted(snapshots) if base_date <= d <= current_date]


def flows_between(
    snapshots: Mapping[str, Mapping[str, Any]],
    base_date: str | None,
    current_date: str,
) -> FlowResult | None:
    if not base_date or base_date not in snapshots or current_date not in snapshots:
        return None
    path = path_between(snapshots, base_date, current_date)
    if len(path) < 2:
        return None
    return quantity_flows(path)


def summarize_trades(result: FlowResult | None, limit: int | None = None) -> dict[str, list[str]]:
    """Render buy/sell/new/exit lists as short strings for interpretation lines."""
    if result is None:
        return {"buy": [], "sell": [], "new": [], "exited": []}

    def render(flows: Iterable[QuantityFlow]) -> list[str]:
        out = []
        for flow in flows:
            label = flow.key.replace(" US EQUITY", "")
            if flow.kind in {"매수", "매도"}:
                out.append(f"{label} {flow.selective_pct:+.0f}%")
            else:
                out.append(label)
        return out[:limit] if limit else out

    return {
        "buy": render(result.by_kind("매수")),
        "sell": render(result.by_kind("매도")),
        "new": render(result.by_kind("신규")),
        "exited": render(result.by_kind("편출")),
    }
