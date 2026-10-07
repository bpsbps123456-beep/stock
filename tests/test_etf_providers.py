"""Network-free tests for the two provider parsers (shapes copied from real captures)."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from gbrc.config import load_fund_configs
from gbrc.etf import providers
from gbrc.etf.snapshot import normalize_snapshot

FUNDS = load_fund_configs()

TIME_HTML = """<form><input type="date" name="pdfDate" value="2026-10-02" min="2023-05-16"></form>
<table><thead><tr><th>종목코드</th><th>종목명</th><th>수량</th><th>평가금액(원)</th><th>비중(%)</th></tr></thead>
<tbody>
<tr>
  <td>INTC US EQUITY</td><td>Intel Corp</td><td>417</td><td>67,584,024</td><td>5.71</td>
</tr>
<tr>
  <td>NQZ6 INDEX</td><td>NASDAQ 100 E-MINI INDEX DEC 2026</td><td>0</td><td>49,854,157</td><td>4.22</td>
</tr>
<tr><td>합계</td><td colspan="4">-</td></tr>
<tr>
  <td>BAD US EQUITY</td><td>Bad Row</td><td>n/a</td><td>1</td><td>0.1</td>
</tr>
<tr>
  <td></td><td>현금</td><td>1,000</td><td>1,000,000</td><td>0.08</td>
</tr>
</tbody></table>"""

KOACT_PAYLOAD = {"pdf": {"gijunYMD": "20261006", "list": [
    {"itmNo": "CASH00000001", "secNm": "설정현금액", "evalA": "979613759", "applyQ": "979613759",
     "ratio": ""},
    {"itmNo": "KRD010010001", "secNm": "원화현금", "evalA": "100", "applyQ": "100", "ratio": ""},
    {"itmNo": "nvda us equity", "secNm": "NVIDIA", "evalA": "600", "applyQ": "3", "ratio": "60.5"},
    {"itmNo": "MSFT US Equity", "secNm": "Microsoft", "evalA": "300", "applyQ": "1.5", "ratio": ""},
]}}


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class ProviderTests(unittest.TestCase):
    def test_source_urls_follow_each_provider(self) -> None:
        self.assertEqual(providers.source_url(FUNDS["time"], "2026-10-02"),
                         "https://timeetf.co.kr/m11_view.php?idx=6&cate=&pdfDate=2026-10-02")
        self.assertEqual(providers.source_url(FUNDS["koact"], "2026-10-06"),
                         "https://www.samsungactive.co.kr/api/v1/product/etf-pdf/2ETFQ1.do"
                         "?gijunYMD=20261006")

    def test_time_table_rows_and_provider_date(self) -> None:
        raw = providers.parse_time_html(TIME_HTML, FUNDS["time"], "2026-10-05", "url")
        self.assertEqual(raw["actual_date"], "2026-10-02")  # 휴장일 요청 → 직전 바스켓
        self.assertEqual(raw["etf_ticker"], "456600")
        self.assertEqual(raw["etf_idx"], 6)
        self.assertEqual([h["code"] for h in raw["holdings"]], ["INTC US EQUITY", "NQZ6 INDEX", ""])
        intc = raw["holdings"][0]
        self.assertEqual((intc["qty"], intc["value_krw"], intc["weight"]), (417, 67584024, 5.71))
        self.assertEqual(raw["holdings_count"], 3)  # 합계·숫자 오류 행은 건너뛴다

    def test_time_page_without_table_is_none(self) -> None:
        self.assertIsNone(providers.parse_time_html("<html></html>", FUNDS["time"], "2026-10-05", "u"))

    def test_koact_cash_rows_and_nav_weights(self) -> None:
        raw = providers.parse_samsung_koact(KOACT_PAYLOAD, FUNDS["koact"], "2026-10-06", "url")
        self.assertEqual(raw["actual_date"], "2026-10-06")
        self.assertIsNone(raw["etf_idx"])
        by_code = {h["code"]: h for h in raw["holdings"]}
        self.assertEqual(set(by_code), {"NVDA US EQUITY", "MSFT US EQUITY", ""})
        # 설정현금액은 합계에서 빠지고, 원화현금은 현금 한 줄이 된다. 총평가액 1,000원.
        self.assertAlmostEqual(by_code["NVDA US EQUITY"]["weight"], 60.0)
        self.assertEqual(by_code["NVDA US EQUITY"]["source_weight_pct"], 60.5)
        self.assertAlmostEqual(by_code["MSFT US EQUITY"]["source_weight_pct"], 30.0)
        self.assertEqual((by_code[""]["name"], by_code[""]["weight"]), ("Cash", 10.0))
        self.assertEqual(normalize_snapshot(raw)["holdings_count"], 3)

    def test_koact_missing_provider_date_falls_back_to_request(self) -> None:
        payload = {"pdf": {"list": KOACT_PAYLOAD["pdf"]["list"]}}
        raw = providers.parse_samsung_koact(payload, FUNDS["koact"], "2026-10-06", "url")
        self.assertEqual(raw["actual_date"], "2026-10-06")

    def test_fetch_holdings_dispatches_by_provider(self) -> None:
        body = json.dumps(KOACT_PAYLOAD).encode("utf-8")
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(body)) as urlopen:
            raw = providers.fetch_holdings(FUNDS["koact"], "2026-10-06")
        request = urlopen.call_args.args[0]
        self.assertIn("2ETFQ1.do?gijunYMD=20261006", request.full_url)
        self.assertEqual(request.get_header("Referer"),
                         "https://www.samsungactive.co.kr/etf/view.do?id=2ETFQ1")
        self.assertEqual(raw["_raw_payload"], KOACT_PAYLOAD)
        with mock.patch("urllib.request.urlopen",
                        return_value=FakeResponse(TIME_HTML.encode("utf-8"))):
            raw = providers.fetch_holdings(FUNDS["time"], "2026-10-05")
        self.assertEqual(raw["actual_date"], "2026-10-02")


if __name__ == "__main__":
    unittest.main()
