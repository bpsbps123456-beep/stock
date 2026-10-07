"""Canonical ETF snapshot schema and validation.

Version 1 files under ``data/etf_holdings`` are treated as immutable source
records.  This module converts them into version 2 NAV-basis snapshots.  The
canonical ``weight`` always uses the same denominator as ``value_krw`` and
therefore includes cash exactly once.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any


ETF_SNAPSHOT_SCHEMA_VERSION = 2
PARSER_VERSION = "gbrc-etf-v2"
KST = timezone(timedelta(hours=9))


class SnapshotValidationError(ValueError):
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _raw_hash(payload: Any) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return _stable_hash(payload)


def _iso_kst(value: str | None) -> str:
    if value:
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=KST)
            return parsed.astimezone(KST).isoformat(timespec="seconds")
        except ValueError:
            pass
    return datetime.now(KST).isoformat(timespec="seconds")


def _is_cash(holding: dict[str, Any]) -> bool:
    code = str(holding.get("code") or "").strip()
    name = str(holding.get("name") or "").strip().lower()
    return not code or name in {"cash", "현금"} or "현금" in name


def _number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _content_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": snapshot.get("schema_version"),
        "fund_id": snapshot.get("fund_id"),
        "as_of_date": snapshot.get("as_of_date"),
        "holdings": snapshot.get("holdings", []),
    }


def normalize_snapshot(raw_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy/provider snapshot to the canonical v2 representation."""
    raw = deepcopy(raw_snapshot)
    requested_date = str(
        raw.get("requested_date") or raw.get("date") or ""
    )
    as_of_date = str(
        raw.get("as_of_date") or raw.get("actual_date") or requested_date
    )
    fund_id = str(raw.get("fund_id") or raw.get("etf_ticker") or "")

    source_holdings = list(raw.get("holdings") or [])
    total_value = sum(max(_number(row.get("value_krw")), 0.0)
                      for row in source_holdings)
    if total_value <= 0:
        raise SnapshotValidationError(["holdings total value_krw must be positive"])

    holdings: list[dict[str, Any]] = []
    for row in source_holdings:
        value_krw = max(_number(row.get("value_krw")), 0.0)
        source_weight = row.get("source_weight_pct")
        if source_weight is None:
            source_weight = row.get("weight")
        nav_weight = value_krw / total_value * 100.0
        qty = _number(row.get("qty"))
        if qty.is_integer():
            qty = int(qty)
        holdings.append({
            "code": str(row.get("code") or "").strip(),
            "name": str(row.get("name") or "").strip(),
            "asset_type": "cash" if _is_cash(row) else "security",
            "qty": qty,
            "value_krw": int(round(value_krw)),
            "source_weight_pct": (
                round(_number(source_weight), 8)
                if source_weight is not None else None
            ),
            "nav_weight_pct": round(nav_weight, 8),
            # Compatibility key used by the existing analytics/report code.
            "weight": round(nav_weight, 8),
        })

    holdings.sort(key=lambda item: (-item["weight"], item["code"]))
    cash_weight = sum(item["nav_weight_pct"] for item in holdings
                      if item["asset_type"] == "cash")
    source_non_cash = sum(
        item["source_weight_pct"] or 0.0
        for item in holdings
        if item["asset_type"] != "cash"
    )

    normalized: dict[str, Any] = {
        "schema_version": ETF_SNAPSHOT_SCHEMA_VERSION,
        "fund_id": fund_id,
        "etf_ticker": fund_id,
        "etf_name": raw.get("etf_name"),
        "requested_date": requested_date,
        "as_of_date": as_of_date,
        # Compatibility aliases.  Canonical analysis always uses as_of_date.
        "date": as_of_date,
        "actual_date": as_of_date,
        "retrieved_at": _iso_kst(raw.get("retrieved_at") or raw.get("fetched_at")),
        "source_url": raw.get("source_url"),
        "parser_version": PARSER_VERSION,
        "source_sha256": _stable_hash(raw),
        "raw_sha256": _raw_hash(raw.get("_raw_payload")),
        "holdings_count": len(holdings),
        "holdings": holdings,
        "metrics": {
            "nav_weight_sum_pct": round(sum(
                item["nav_weight_pct"] for item in holdings
            ), 8),
            "cash_weight_pct": round(cash_weight, 8),
            "invested_nav_weight_pct": round(100.0 - cash_weight, 8),
            "gross_exposure_pct": round(sum(
                item["nav_weight_pct"] for item in holdings
                if item["asset_type"] != "cash"
            ), 8),
            "source_non_cash_weight_sum_pct": round(source_non_cash, 8),
        },
        "quality": {
            "status": "fallback" if requested_date != as_of_date else "valid",
            "requested_date_differs": requested_date != as_of_date,
            "checks": [],
        },
    }
    normalized["content_sha256"] = _stable_hash(_content_payload(normalized))
    issues = validate_snapshot(normalized)
    if issues:
        normalized["quality"]["status"] = "invalid"
        normalized["quality"]["checks"] = issues
        raise SnapshotValidationError(issues)
    normalized["quality"]["checks"] = [
        "schema",
        "identity",
        "dates",
        "holdings_count",
        "unique_codes",
        "cash_count",
        "finite_nonnegative_values",
        "nav_weight_sum",
    ]
    return normalized


def validate_snapshot(snapshot: dict[str, Any]) -> list[str]:
    """Return validation issues.  An empty list means the snapshot is valid."""
    issues: list[str] = []
    if snapshot.get("schema_version") != ETF_SNAPSHOT_SCHEMA_VERSION:
        issues.append("schema_version must be 2")

    fund_id = str(snapshot.get("fund_id") or "")
    if fund_id not in {"456600", "2ETFQ1"}:
        issues.append("fund_id must be 456600 or 2ETFQ1")

    for field in ("requested_date", "as_of_date"):
        value = str(snapshot.get(field) or "")
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            issues.append(f"{field} must be YYYY-MM-DD")

    holdings = snapshot.get("holdings")
    if not isinstance(holdings, list) or not holdings:
        issues.append("holdings must be a non-empty list")
        return issues
    if snapshot.get("holdings_count") != len(holdings):
        issues.append("holdings_count does not match holdings length")

    seen: set[str] = set()
    cash_count = 0
    for index, holding in enumerate(holdings):
        code = str(holding.get("code") or "").strip()
        asset_type = holding.get("asset_type")
        key = "__CASH__" if asset_type == "cash" else code
        if not key:
            issues.append(f"holding[{index}] has an empty non-cash code")
        elif key in seen:
            issues.append(f"duplicate holding code: {key}")
        seen.add(key)
        if asset_type == "cash":
            cash_count += 1

        for field in ("qty", "value_krw", "nav_weight_pct", "weight"):
            value = holding.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                issues.append(f"holding[{index}].{field} must be finite numeric")
            elif value < 0:
                issues.append(f"holding[{index}].{field} must be nonnegative")

    if cash_count > 1:
        issues.append("at most one cash holding is allowed")

    weight_sum = sum(_number(item.get("nav_weight_pct")) for item in holdings)
    if not 99.99 <= weight_sum <= 100.01:
        issues.append(f"nav weight sum must be 100%, got {weight_sum:.6f}%")

    expected_hash = _stable_hash(_content_payload(snapshot))
    if snapshot.get("content_sha256") != expected_hash:
        issues.append("content_sha256 does not match canonical content")
    return issues


def canonical_filename(snapshot: dict[str, Any]) -> str:
    return f"{snapshot['as_of_date']}_{snapshot['fund_id']}.json"


def snapshot_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "fund_id": snapshot["fund_id"],
        "requested_date": snapshot["requested_date"],
        "as_of_date": snapshot["as_of_date"],
        "holdings_count": snapshot["holdings_count"],
        "cash_weight_pct": snapshot["metrics"]["cash_weight_pct"],
        "nav_weight_sum_pct": snapshot["metrics"]["nav_weight_sum_pct"],
        "quality_status": snapshot["quality"]["status"],
        "content_sha256": snapshot["content_sha256"],
    }
