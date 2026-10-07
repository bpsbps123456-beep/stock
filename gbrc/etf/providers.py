"""Provider collectors for the two funds' daily constituent files (PDF).

`fetch_holdings(fund, date)` is the only entry point that touches the network. The
parsers are pure, so recorded provider payloads can be replayed in tests. The return
value is the raw snapshot that `snapshot.normalize_snapshot` accepts; a provider that
answers a holiday with the previous basket shows it through `actual_date`.
"""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime
from typing import Any

from gbrc.config import FundConfig

USER_AGENT = "Mozilla/5.0"
TIMEOUT_SEC = 30
SAMSUNG_CASH_ROW = "CASH00000001"      # 설정현금액: 구성종목이 아닌 설정단위 합계라 뺀다
SAMSUNG_KRW_DEPOSIT = "KRD010010001"   # 원화현금 → 현금 한 줄


def source_url(fund: FundConfig, date_str: str) -> str:
    """The provider address for the basket dated `date_str` (YYYY-MM-DD)."""
    if fund.provider == "samsung_koact":
        ymd = date_str.replace("-", "")
        return f"{fund.source_url}/api/v1/product/etf-pdf/{fund.provider_id}.do?gijunYMD={ymd}"
    return f"{fund.source_url}?idx={fund.legacy_idx}&cate=&pdfDate={date_str}"


def fetch_holdings(fund: FundConfig, date_str: str) -> dict[str, Any] | None:
    url = source_url(fund, date_str)
    if fund.provider == "samsung_koact":
        referer = f"{fund.source_url}/etf/view.do?id={fund.provider_id}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Referer": referer})
        with urllib.request.urlopen(request, timeout=TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        return parse_samsung_koact(data, fund, date_str, url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SEC) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    return parse_time_html(html, fund, date_str, url)


def _raw_snapshot(fund: FundConfig, date_str: str, actual_date: str | None, url: str,
                  payload: Any, holdings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "date": date_str,
        "actual_date": actual_date,
        "etf_idx": fund.legacy_idx,
        "etf_ticker": fund.snapshot_ticker,
        "etf_name": fund.display_name,
        "source_url": url,
        "_raw_payload": payload,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "holdings_count": len(holdings),
        "holdings": holdings,
    }


# --------------------------------------------------------------------------
# TIME (timeetf.co.kr HTML)
# --------------------------------------------------------------------------

def parse_time_html(html: str, fund: FundConfig, date_str: str, url: str) -> dict[str, Any] | None:
    """Constituent table of the TIME page. None when the page has no table."""
    # 응답이 요청 날짜 데이터인지 검증
    m_check = re.search(r'name=["\']pdfDate["\'][^>]*value=["\'](\d{4}-\d{2}-\d{2})', html)
    actual_date = m_check.group(1) if m_check else None

    # 종목코드 헤더 → 그 다음 </table>까지가 구성종목 영역
    m = re.search(r"<th[^>]*>종목코드</th>.*?</table>", html, re.DOTALL)
    if not m:
        return None

    holdings = []
    for tr in re.findall(r"<tr>(.*?)</tr>", m.group(0), re.DOTALL):
        cells = [re.sub(r"<.*?>", "", c).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)]
        if len(cells) < 5:
            continue
        try:
            holdings.append({
                "code": cells[0],
                "name": cells[1],
                "qty": int(cells[2].replace(",", "")),
                "value_krw": int(cells[3].replace(",", "")),
                "source_weight_pct": float(cells[4]),
                "weight": float(cells[4]),
            })
        except (ValueError, IndexError):
            continue
    return _raw_snapshot(fund, date_str, actual_date, url, html, holdings)


# --------------------------------------------------------------------------
# KoAct (Samsung Active ETF-PDF API)
# --------------------------------------------------------------------------

def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int_value(value: Any) -> int:
    number = parse_number(value)
    if number is None:
        return 0
    return int(round(number))


def normalize_samsung_code(code: str | None) -> str:
    code = (code or "").strip()
    parts = code.split()
    if len(parts) >= 3 and parts[-1].lower() == "equity":
        return f"{parts[0].upper()} {parts[1].upper()} EQUITY"
    return code.upper()


def format_yyyymmdd(raw: Any, fallback: str) -> str:
    raw = str(raw or "").strip()
    if re.match(r"^\d{8}$", raw):
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return fallback


def parse_samsung_koact(data: dict[str, Any], fund: FundConfig, date_str: str,
                        url: str) -> dict[str, Any]:
    pdf = data.get("pdf") or {}
    rows = pdf.get("list") or []
    actual_date = format_yyyymmdd(pdf.get("gijunYMD"), date_str)

    portfolio_rows = []
    for row in rows:
        raw_code = (row.get("itmNo") or "").strip()
        if raw_code == SAMSUNG_CASH_ROW:
            continue
        portfolio_rows.append((row, parse_int_value(row.get("evalA"))))

    total_value = sum(value for _, value in portfolio_rows)
    holdings = []
    cash_value = 0

    for row, value_krw in portfolio_rows:
        raw_code = (row.get("itmNo") or "").strip()
        name = (row.get("secNm") or "").strip()
        if raw_code == SAMSUNG_KRW_DEPOSIT:
            cash_value += value_krw
            continue

        ratio = parse_number(row.get("ratio"))
        if ratio is None:
            ratio = (value_krw / total_value * 100.0) if total_value else 0.0

        holdings.append({
            "code": normalize_samsung_code(raw_code),
            "name": name,
            "qty": parse_number(row.get("applyQ")) or 0,
            "value_krw": value_krw,
            "source_weight_pct": ratio,
            "weight": ratio,
        })

    if cash_value:
        cash_weight = (cash_value / total_value * 100.0) if total_value else 0.0
        holdings.append({
            "code": "",
            "name": "Cash",
            "qty": cash_value,
            "value_krw": cash_value,
            "source_weight_pct": None,
            "weight": cash_weight,
        })

    # Samsung's disclosed ratio is normalized over non-cash holdings.  Use
    # evaluation value / total NAV for the compatibility weight so cash is
    # included exactly once.  The original ratio remains available above.
    if total_value:
        for holding in holdings:
            holding["weight"] = holding["value_krw"] / total_value * 100.0

    holdings.sort(key=lambda h: -h["weight"])
    return _raw_snapshot(fund, date_str, actual_date, url, data, holdings)
