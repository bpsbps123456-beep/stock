"""Holding identity, listing market, and price-reference-date rules.

A Korean ETF PDF dated D describes the creation basket after the previous KRX
session (D-1) closed.  Domestic holdings are valued at that session's close;
foreign holdings use the last local close strictly before it.  Measured on the
2026-04-10..09-23 snapshots by comparing value_krw/qty with market closes:
KR median error 0.00%, US/JP/HK/CN 0.25-0.30%.  Shifting either rule by one
session raises the error to 2-6%.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from datetime import date, timedelta
from typing import Any, Iterable, Mapping, Sequence

# 선물 월물 코드. 예: NQZ6 INDEX → 루트 NQ, 12월(Z), 2026년(6)
FUTURES_RE = re.compile(r"^([A-Z]{1,4}?)([FGHJKMNQUVXZ])(\d{1,2}) INDEX$")
KR_CODE_RE = re.compile(r"^[0-9A-Z]{6}$")
FUTURES_YAHOO_EXCHANGE = {"NQ": "CME", "ES": "CME", "RTY": "CME", "YM": "CBT"}
EXCHANGE_MARKETS = {
    "US": "US",
    "JP": "JP",
    "HK": "HK",
    "CH": "CN",
    "C1": "CN",
    "C2": "CN",
    "KS": "KR",
    "KQ": "KR",
}
FOREIGN_MARKETS = frozenset({"US", "JP", "HK", "CN", "FUT"})

# 운용사가 같은 종목의 코드를 바꿔 부른 경우. 연속 보유로 이어 붙인다.
CODE_ALIASES = {
    "SKHYV US EQUITY": "SKHY US EQUITY",  # 2026-07-14 발행일 거래 티커 → 정규 티커
}
# 두 펀드를 교차 비교할 때만 같은 회사로 묶는 클래스주.
SHARE_CLASS_ALIASES = {
    "GOOGL US EQUITY": "GOOG US EQUITY",
}


def canonical_code(code: Any) -> str:
    """Collapse whitespace and case differences such as 'NQM6 Index'."""
    return " ".join(str(code or "").split()).upper()


def holding_key(code: Any) -> str:
    """Identity used to match one holding across snapshots.

    Futures roll every quarter (NQU6 → NQZ6); the exposure is continuous, so
    every contract month of a root shares one key.
    """
    canonical = canonical_code(code)
    canonical = CODE_ALIASES.get(canonical, canonical)
    match = FUTURES_RE.match(canonical)
    if match:
        return f"{match.group(1)} FUTURES"
    return canonical


def company_key(code: Any) -> str:
    """Identity used only when matching the same company across two funds."""
    key = holding_key(code)
    return SHARE_CLASS_ALIASES.get(key, key)


def is_cash_holding(holding: Mapping[str, Any]) -> bool:
    code = str(holding.get("code") or "").strip()
    name = str(holding.get("name") or "").strip()
    return (
        holding.get("asset_type") == "cash"
        or code == ""
        or "현금" in name
        or name.lower() == "cash"
    )


def listing_market(code: Any) -> str | None:
    canonical = canonical_code(code)
    if not canonical:
        return None
    if FUTURES_RE.match(canonical):
        return "FUT"
    parts = canonical.split()
    if len(parts) >= 3 and parts[-1] == "EQUITY":
        return EXCHANGE_MARKETS.get(parts[-2])
    if len(parts) == 1 and KR_CODE_RE.match(canonical) and canonical[:2].isdigit():
        return "KR"
    return None


def is_us_listed(code: Any) -> bool:
    return listing_market(code) == "US"


def _futures_year(digits: str, as_of: str | None) -> int:
    if len(digits) == 2:
        return 2000 + int(digits)
    base = date.fromisoformat(as_of).year if as_of else date.today().year
    decade = base - base % 10
    candidates = [decade - 10 + int(digits), decade + int(digits), decade + 10 + int(digits)]
    return min(candidates, key=lambda year: (abs(year - base), -year))


def yahoo_ticker(code: Any, as_of: str | None = None) -> str | None:
    """Return the Yahoo Finance symbol for a PDF code, or None when unpriceable."""
    canonical = canonical_code(code)
    market = listing_market(canonical)
    if market is None:
        return None
    if market == "FUT":
        root, month, digits = FUTURES_RE.match(canonical).groups()
        exchange = FUTURES_YAHOO_EXCHANGE.get(root)
        if not exchange:
            return None
        return f"{root}{month}{_futures_year(digits, as_of) % 100:02d}.{exchange}"

    raw = canonical.split()[0]
    if market == "US":
        # Bloomberg 클래스주 표기(MOG/A)를 Yahoo 표기(MOG-A)로 바꾼다.
        return raw.replace("/", "-").replace(".", "-")
    if market == "JP":
        return f"{raw}.T"
    if market == "HK":
        return f"{raw.zfill(4)}.HK"
    if market == "CN":
        if raw[:1] in {"6", "9"}:
            return f"{raw}.SS"
        if raw[:1] in {"4", "8"}:
            return f"{raw}.BJ"
        return f"{raw}.SZ"
    # KR: Yahoo는 코스닥 종목도 .KS로 조회된다(440110 확인). 원천이 KQ를 명시할 때만 .KQ.
    suffix = ".KQ" if canonical.endswith(" KQ EQUITY") else ".KS"
    return f"{raw.zfill(6)}{suffix}"


def yahoo_ticker_candidates(code: Any, as_of: str | None = None) -> list[str]:
    """Symbols to try in order. Yahoo drops expired futures months, so a
    contract falls back to the continuous front month (NQ=F)."""
    primary = yahoo_ticker(code, as_of)
    if primary is None:
        return []
    if listing_market(code) == "FUT":
        root = FUTURES_RE.match(canonical_code(code)).group(1)
        return [primary, f"{root}=F"]
    return [primary]


def series_for(
    code: Any,
    as_of: str | None,
    prices: Mapping[str, Mapping[str, float]],
) -> Mapping[str, float]:
    """First non-empty price series among the code's Yahoo candidates."""
    for ticker in yahoo_ticker_candidates(code, as_of):
        series = prices.get(ticker)
        if series:
            return series
    return {}


def previous_session(target: str, n: int = 1, sessions: Sequence[str] | None = None) -> str:
    """Return the n-th KRX session strictly before target.

    ``sessions`` is a sorted list of known KRX session dates.  Without enough
    session history the weekday calendar is used as a fallback.
    """
    if sessions:
        ordered = sorted(set(sessions))
        index = bisect_left(ordered, target)
        if index >= n:
            return ordered[index - n]
    current = date.fromisoformat(target)
    counted = 0
    while counted < n:
        current -= timedelta(days=1)
        if current.weekday() < 5:
            counted += 1
    return current.isoformat()


def price_reference_date(code: Any, as_of: str, sessions: Sequence[str] | None = None) -> str:
    """Latest local close date reflected in the PDF dated ``as_of``.

    Use it with an on-or-before close lookup.  Domestic holdings resolve to
    the previous KRX session; foreign holdings to the day before it, so the
    lookup lands on the last local close strictly before that session.
    """
    valuation_session = previous_session(as_of, 1, sessions)
    if listing_market(code) in FOREIGN_MARKETS:
        return (date.fromisoformat(valuation_session) - timedelta(days=1)).isoformat()
    return valuation_session


def select_base_date(
    snapshot_dates: Sequence[str],
    target: str,
    n: int,
    sessions: Sequence[str] | None = None,
) -> tuple[str, str | None]:
    """Return (intended session, snapshot date used) for a D-n comparison.

    D-n counts actual KRX sessions.  When the intended session has no
    snapshot, the latest earlier snapshot is used; with no earlier snapshot,
    the earliest one before the target is used so short histories still
    compare.  The caller prints the date actually used.
    """
    intended = previous_session(target, n, sessions or snapshot_dates)
    available = sorted(d for d in snapshot_dates if d < target)
    if intended in available:
        return intended, intended
    earlier = [d for d in available if d <= intended]
    if earlier:
        return intended, earlier[-1]
    return intended, (available[0] if available else None)


def close_on_or_before(series: Mapping[str, float], target: str) -> tuple[str, float] | None:
    """Latest (date, close) at or before target from a {YYYY-MM-DD: close} map."""
    eligible = [d for d in series if d <= target]
    if not eligible:
        return None
    chosen = max(eligible)
    value = series[chosen]
    if value is None:
        return None
    return chosen, float(value)


def return_between(
    series: Mapping[str, float] | None,
    base_date: str,
    target_date: str,
) -> float | None:
    """Close-to-close percentage return, or None when a close is missing."""
    if not series:
        return None
    base = close_on_or_before(series, base_date)
    target = close_on_or_before(series, target_date)
    if not base or not target or target[0] <= base[0] or base[1] <= 0:
        return None
    return (target[1] / base[1] - 1.0) * 100.0


def merge_sessions(*groups: Iterable[str]) -> list[str]:
    return sorted({d for group in groups for d in group if d})
