"""Unit tests for the deterministic ETF report contract."""

from __future__ import annotations

import unittest

from gbrc.etf.report_contract import (
    ReportContractError,
    validate_daily_report,
    validate_fund_report,
    validate_performance_report,
)


ONE_GLANCE_HEADER = (
    "  Rank  Code  Name  Weight  D-1 dW(ret)  D-3 dW(ret)  "
    "D-5 dW(ret)  D-20 dW(ret)"
)


def valid_fund_report(
    row_count: int = 2,
    *,
    top10_body: str = "  1  AAA  Alpha  90.00%",
    sector_body: str = "AI 반도체  90.00%",
    flow_body: str = "  D-5 실매수 1개: CODE1 +10.0% (+0.50%p)",
) -> str:
    lines = [
        "### [현재 비중 Top 10]",
        top10_body,
        "### [현금 보유액 별도 점검]",
        "Cash  10.00%",
        "### [섹터별 비중/변화 요약]",
        sector_body,
        "### [수량 기준 실매매]",
        flow_body,
        "### [D-20 신규 편입 클러스터]",
        "신규 편입 없음",
        "### [One-glance table]",
        ONE_GLANCE_HEADER,
        "  ----  ----  ----  ------  -----------  -----------  -----------  ------------",
    ]
    for rank in range(1, row_count + 1):
        is_cash = rank == row_count
        code = "CASH" if is_cash else f"CODE{rank}"
        name = "Cash" if is_cash else f"Holding {rank}"
        lines.append(
            f"{rank:6d}  {code}  {name}  50.00%  +0.00  +0.00  +0.00  +0.00"
        )
    lines.extend(
        [
            "",
            "### [포트폴리오 해석]",
            "집중도와 현금 수준을 함께 해석합니다.",
            "### [직전 변화 복기]",
            "직전 변화가 현재까지 이어졌는지 확인합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def valid_performance_report() -> str:
    return "\n".join(
        [
            "### [두 펀드 성과 비교] — ETF 시장가격 종가 기준",
            "D-1  TIME +0.10%  KoAct +0.20%  KoAct",
            "D-3  TIME +0.30%  KoAct +0.40%  KoAct",
            "D-5  TIME +0.50%  KoAct +0.60%  KoAct",
            "D-20  TIME +2.00%  KoAct +1.00%  TIME",
        ]
    )


def daily_report(level_two_headings: list[str]) -> str:
    sections = ["# GBRC fixture"]
    sections.extend(f"{heading}\nfixture body" for heading in level_two_headings)
    return "\n\n".join(sections)


class ReportContractTests(unittest.TestCase):
    fund_headings = ["## TIME Fixture (456600)", "## KoAct Fixture (2ETFQ1)"]

    def test_complete_fund_performance_and_daily_reports_pass(self) -> None:
        self.assertIsNone(validate_fund_report(valid_fund_report(), 2))
        self.assertIsNone(validate_performance_report(valid_performance_report()))
        ordered = [
            "## 핵심 결론",
            "## 두 펀드 가격 성과 비교",
            *self.fund_headings,
        ]
        self.assertIsNone(
            validate_daily_report(daily_report(ordered), self.fund_headings)
        )

    def test_all_holdings_row_count_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(
            ReportContractError,
            "all-holdings row count mismatch: expected 3, got 2",
        ):
            validate_fund_report(valid_fund_report(row_count=2), 3)

    def test_missing_quantity_trade_section_fails(self) -> None:
        report = valid_fund_report().replace("### [수량 기준 실매매]", "### [실매매 생략]")
        with self.assertRaisesRegex(
            ReportContractError,
            r"missing required report markers: \[수량 기준 실매매\]",
        ):
            validate_fund_report(report, 2)

    def test_cash_in_top10_sector_or_trade_section_fails(self) -> None:
        cases = [
            {"top10_body": "  1  CASH  Cash  10.00%"},
            {"sector_body": "현금  10.00%"},
            {"flow_body": "  실매수: 현금 +1.00%p"},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(
                    ReportContractError,
                    "cash must appear only in its dedicated analysis section",
                ):
                    validate_fund_report(valid_fund_report(**kwargs), 2)

    def test_level_two_option_and_level_four_per_headings_fail(self) -> None:
        for forbidden_heading in (
            "## 옵션×ETF 등급",
            "#### 종목별 PER 표",
        ):
            with self.subTest(heading=forbidden_heading):
                report = valid_fund_report() + forbidden_heading + "\n"
                with self.assertRaisesRegex(
                    ReportContractError,
                    "forbidden option/PER section",
                ):
                    validate_fund_report(report, 2)

    def test_extra_per_column_breaks_exact_eight_column_header(self) -> None:
        report = valid_fund_report().replace(
            ONE_GLANCE_HEADER,
            ONE_GLANCE_HEADER + "  PER",
        )
        with self.assertRaisesRegex(
            ReportContractError,
            "expected one all-holdings table, found 0",
        ):
            validate_fund_report(report, 2)

    def test_wrong_global_section_order_fails(self) -> None:
        wrong_order = [
            "## 핵심 결론",
            self.fund_headings[0],
            "## 두 펀드 가격 성과 비교",
            self.fund_headings[1],
        ]
        with self.assertRaisesRegex(
            ReportContractError,
            "daily report section order mismatch",
        ):
            validate_daily_report(daily_report(wrong_order), self.fund_headings)


if __name__ == "__main__":
    unittest.main()
