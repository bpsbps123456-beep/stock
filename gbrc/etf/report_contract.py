"""Deterministic checks for the ETF-only report contract."""

from __future__ import annotations

import re


class ReportContractError(ValueError):
    pass


REQUIRED_FUND_MARKERS = (
    "[현재 비중 Top 10]",
    "[현금 보유액 별도 점검]",
    "[섹터별 비중/변화 요약]",
    "[수량 기준 실매매]",
    "[D-20 신규 편입 클러스터]",
    "[One-glance table]",
    "[포트폴리오 해석]",
    "[직전 변화 복기]",
    "D-1 dW(ret)",
    "D-3 dW(ret)",
    "D-5 dW(ret)",
    "D-20 dW(ret)",
)

ONE_GLANCE_HEADER_RE = re.compile(
    r"^\s*Rank\s+Code\s+Name\s+Weight\s+"
    r"D-1 dW\(ret\)\s+D-3 dW\(ret\)\s+"
    r"D-5 dW\(ret\)\s+D-20 dW\(ret\)\s*$"
)
FORBIDDEN_HEADING_RE = re.compile(
    r"^#{1,6}\s+.*(?:옵션|\bPER\b)",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _one_glance_row_count(report: str) -> int:
    lines = report.splitlines()
    header_indexes = [
        index for index, line in enumerate(lines)
        if ONE_GLANCE_HEADER_RE.match(line)
    ]
    if len(header_indexes) != 1:
        raise ReportContractError(
            f"expected one all-holdings table, found {len(header_indexes)}"
        )
    index = header_indexes[0] + 2
    rows = 0
    while index < len(lines) and lines[index].strip():
        if re.match(r"^\s+\d+\s{2}", lines[index]):
            rows += 1
        index += 1
    return rows


def _section_body(report: str, marker: str) -> str:
    lines = report.splitlines()
    indexes = [index for index, line in enumerate(lines) if marker in line]
    if len(indexes) != 1:
        return ""
    body: list[str] = []
    for line in lines[indexes[0] + 1:]:
        if line.startswith("### "):
            break
        body.append(line)
    return "\n".join(body)


def _validate_cash_isolation(report: str) -> None:
    for marker in ("[현재 비중 Top 10]", "[섹터별 비중/변화 요약]", "[수량 기준 실매매]"):
        body = _section_body(report, marker)
        if re.search(r"(?:현금|\bCash\b)", body, flags=re.IGNORECASE):
            raise ReportContractError(
                f"cash must appear only in its dedicated analysis section: {marker}"
            )


def validate_fund_report(report: str, expected_holdings: int) -> None:
    issues = [marker for marker in REQUIRED_FUND_MARKERS if marker not in report]
    if issues:
        raise ReportContractError(
            "missing required report markers: " + ", ".join(issues)
        )
    if FORBIDDEN_HEADING_RE.search(report):
        raise ReportContractError("ETF report contains a forbidden option/PER section")
    rows = _one_glance_row_count(report)
    if rows != expected_holdings:
        raise ReportContractError(
            f"all-holdings row count mismatch: expected {expected_holdings}, got {rows}"
        )
    _validate_cash_isolation(report)


def validate_performance_report(report: str) -> None:
    if "두 펀드 성과 비교" not in report:
        raise ReportContractError("performance heading is missing")
    for horizon in ("D-1", "D-3", "D-5", "D-20"):
        if not re.search(rf"^\s*{re.escape(horizon)}\s", report, flags=re.MULTILINE):
            raise ReportContractError(f"performance horizon is missing: {horizon}")


def validate_daily_report(report: str, fund_headings: list[str]) -> None:
    """Require decision-first ordering and exactly the two configured funds."""
    level_two = [
        line.strip()
        for line in report.splitlines()
        if re.match(r"^## (?!#)", line)
    ]
    expected = [
        "## 핵심 결론",
        "## 두 펀드 가격 성과 비교",
        *fund_headings,
    ]
    if level_two != expected:
        raise ReportContractError(
            f"daily report section order mismatch: expected {expected}, got {level_two}"
        )
    if FORBIDDEN_HEADING_RE.search(report):
        raise ReportContractError("daily report contains a forbidden option/PER heading")
