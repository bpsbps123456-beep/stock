"""GBRC two-fund ETF holdings collector and comparison report.

Validated snapshots are stored under ``data/normalized/etf_holdings``.  The
daily report is ETF-only, keeps cash in its own analytical section, and prints
every current holding in the one-glance evidence table.

사용법:
  py etf_crawler.py --backfill 2026-04-27 2026-05-05   # 범위 백필
  py etf_crawler.py --today                            # 오늘 데이터만
  py etf_crawler.py --compare                          # 최신 스냅샷 비교 출력
"""

import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from gbrc.config import (
    fund_alias_map,
    load_fund_configs,
    observation_primary_alias,
    performance_funds,
)
from gbrc.etf import flows as qty_flows
from gbrc.etf import market
from gbrc.etf.snapshot import normalize_snapshot
from gbrc.etf.storage import (
    NORMALIZED_HOLDINGS_DIR,
    load_normalized_snapshot,
    save_normalized_snapshot,
)

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from watchlist import TICKER_TO_SECTOR, resolve_sector
except ImportError:
    TICKER_TO_SECTOR = {}

    def resolve_sector(listing_key, default="미분류"):
        return TICKER_TO_SECTOR.get(listing_key, default)

_FUND_CONFIGS = load_fund_configs()
_DEFAULT_FUND = _FUND_CONFIGS["time"]
ETF_ALIAS = "time"
ETF_IDX = _DEFAULT_FUND.legacy_idx
ETF_TICKER = _DEFAULT_FUND.snapshot_ticker
ETF_NAME = _DEFAULT_FUND.display_name
BASE_URL = _DEFAULT_FUND.source_url
ETF_FETCHER = _DEFAULT_FUND.provider
ETF_F_ID = None
ETF_CONFIGS = {
    alias: {
        "fetcher": fund.provider,
        "idx": fund.legacy_idx,
        "ticker": fund.snapshot_ticker,
        "name": fund.display_name,
        "base_url": fund.source_url,
        "f_id": fund.provider_id if fund.provider != "time" else None,
        "listing_ticker": fund.listing_ticker,
    }
    for alias, fund in _FUND_CONFIGS.items()
}
ETF_ALIASES = fund_alias_map()
PERFORMANCE_ETFS = performance_funds()
HOLDINGS_DIR = str(NORMALIZED_HOLDINGS_DIR)
DIFF_THRESHOLD = 0.10  # %p — Q2=B
REPORT_RANK_LIMIT = None  # Path-B/F: print all current holdings.
# 워치리스트 매칭 (US 종목코드 → 우리 워치리스트 ticker)
US_CODE_RE = re.compile(r"^([A-Z0-9./-]+)\s+US\s+EQUITY$")


def set_active_etf(alias):
    global ETF_ALIAS, ETF_IDX, ETF_TICKER, ETF_NAME, BASE_URL, ETF_FETCHER, ETF_F_ID

    key = ETF_ALIASES.get((alias or "time").lower())
    if not key or key not in ETF_CONFIGS:
        choices = ", ".join(sorted(ETF_ALIASES))
        raise SystemExit(f"Unknown ETF alias: {alias}. Choices: {choices}")

    config = ETF_CONFIGS[key]
    ETF_ALIAS = key
    ETF_IDX = config["idx"]
    ETF_TICKER = config["ticker"]
    ETF_NAME = config["name"]
    BASE_URL = config["base_url"]
    ETF_FETCHER = config["fetcher"]
    ETF_F_ID = config["f_id"]


def pop_etf_arg(argv):
    if "--etf" not in argv:
        return "time"
    idx = argv.index("--etf")
    if idx + 1 >= len(argv):
        raise SystemExit("--etf requires an ETF alias, e.g. --etf koact")
    alias = argv[idx + 1]
    del argv[idx:idx + 2]
    return alias


def fetch_time_holdings(date_str):
    """특정 날짜의 구성종목 HTML 가져와서 파싱"""
    url = f"{BASE_URL}?idx={ETF_IDX}&cate=&pdfDate={date_str}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    # 응답이 요청 날짜 데이터인지 검증
    m_check = re.search(
        r'name=["\']pdfDate["\'][^>]*value=["\'](\d{4}-\d{2}-\d{2})', html)
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

    return {
        "date": date_str,
        "actual_date": actual_date,
        "etf_idx": ETF_IDX,
        "etf_ticker": ETF_TICKER,
        "etf_name": ETF_NAME,
        "source_url": url,
        "_raw_payload": html,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "holdings_count": len(holdings),
        "holdings": holdings,
    }


def parse_number(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int_value(value):
    number = parse_number(value)
    if number is None:
        return 0
    return int(round(number))


def normalize_samsung_code(code):
    code = (code or "").strip()
    parts = code.split()
    if len(parts) >= 3 and parts[-1].lower() == "equity":
        return f"{parts[0].upper()} {parts[1].upper()} EQUITY"
    return code.upper()


def format_yyyymmdd(raw, fallback):
    raw = str(raw or "").strip()
    if re.match(r"^\d{8}$", raw):
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return fallback


def fetch_samsung_koact_holdings(date_str):
    """Fetch Samsung Active KoAct constituents from the ETF PDF API."""
    ymd = date_str.replace("-", "")
    f_id = ETF_F_ID or ETF_TICKER
    url = f"{BASE_URL}/api/v1/product/etf-pdf/{f_id}.do?gijunYMD={ymd}"
    referer = f"{BASE_URL}/etf/view.do?id={f_id}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Referer": referer},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))

    pdf = data.get("pdf") or {}
    rows = pdf.get("list") or []
    actual_date = format_yyyymmdd(pdf.get("gijunYMD"), date_str)

    portfolio_rows = []
    for row in rows:
        raw_code = (row.get("itmNo") or "").strip()
        if raw_code == "CASH00000001":
            continue
        value_krw = parse_int_value(row.get("evalA"))
        portfolio_rows.append((row, value_krw))

    total_value = sum(value for _, value in portfolio_rows)
    holdings = []
    cash_value = 0

    for row, value_krw in portfolio_rows:
        raw_code = (row.get("itmNo") or "").strip()
        name = (row.get("secNm") or "").strip()
        if raw_code == "KRD010010001":
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
    return {
        "date": date_str,
        "actual_date": actual_date,
        "etf_idx": ETF_IDX,
        "etf_ticker": ETF_TICKER,
        "etf_name": ETF_NAME,
        "source_url": url,
        "_raw_payload": data,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "holdings_count": len(holdings),
        "holdings": holdings,
    }


def fetch_holdings(date_str):
    if ETF_FETCHER == "samsung_koact":
        return fetch_samsung_koact_holdings(date_str)
    return fetch_time_holdings(date_str)


def save_snapshot(snap):
    normalized = normalize_snapshot(snap)
    path, _ = save_normalized_snapshot(normalized, Path(HOLDINGS_DIR))
    return str(path)


def load_snapshot(date_str):
    path = Path(HOLDINGS_DIR) / f"{date_str}_{ETF_TICKER}.json"
    if not path.exists():
        return None
    return load_normalized_snapshot(path)


def list_snapshot_dates():
    if not os.path.exists(HOLDINGS_DIR):
        return []
    files = [f for f in os.listdir(HOLDINGS_DIR)
             if f.endswith(f"_{ETF_TICKER}.json")]
    return sorted(f[:10] for f in files)


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


def build_entry_date_map(dates=None, through_date=None, snapshots=None):
    """보유 종목별 현재 연속 보유 구간의 시작일을 계산.

    선물 롤오버(NQU6→NQZ6)와 원천 대소문자 차이는 같은 보유로 이어 붙인다.
    """
    entry_dates = EntryDateMap()
    prev_keys = set()
    for date_str in sorted(dates or list_snapshot_dates()):
        if through_date and date_str > through_date:
            continue
        snap = snapshots.get(date_str) if snapshots is not None else load_snapshot(date_str)
        if not snap:
            continue
        curr_keys = {market.holding_key(h.get("code", "")) for h in snap.get("holdings", [])}
        for key in curr_keys:
            if key not in prev_keys:
                dict.__setitem__(entry_dates, key, date_str)
        prev_keys = curr_keys
    return entry_dates


# Backward-compatible alias for older ad-hoc imports.
build_first_seen_map = build_entry_date_map


def get_us_ticker(code):
    """'INTC US EQUITY' → 'INTC', 한국 종목은 None"""
    m = US_CODE_RE.match(code)
    return m.group(1) if m else None


def get_listing_key(code):
    """US ticker 또는 거래소 코드 첫 토큰을 섹터 매핑 키로 사용."""
    code = (code or "").strip()
    if not code:
        return ""
    us_ticker = get_us_ticker(code)
    if us_ticker:
        return us_ticker
    return code.split()[0]


def to_yahoo_us_ticker(us_ticker):
    """Bloomberg 클래스주 표기를 Yahoo 표기로 변환. 'MOG/A' → 'MOG-A'

    섹터 매핑 키는 원천 표기를 그대로 쓰므로 가격 조회 경로에서만 변환한다.
    """
    return us_ticker.replace("/", "-").replace(".", "-")


def get_market_ticker_for_holding(holding, as_of=None):
    """Return a yfinance-compatible ticker when the holding is priceable.

    한국 6자리 코드(009150, 0043Y0), 중국 A주(CH), 지수선물 월물(NQZ6 → NQZ26.CME)을
    포함한다. 변환 규칙 정본은 gbrc/etf/market.py다.
    """
    return market.yahoo_ticker((holding or {}).get("code"), as_of)


def price_reference_date_for_holding(holding, etf_date, sessions=None):
    """Map an ETF snapshot date to the latest close reflected in that NAV.

    as_of D의 PDF는 직전 KRX 세션(D-1) 마감 기준이다. 국내 종목은 그 세션 종가,
    해외 종목은 그 세션보다 앞선 마지막 현지 종가를 쓴다. 반환값은 on-or-before 조회용이다.
    """
    if not holding or not etf_date:
        return etf_date
    return market.price_reference_date(holding.get("code", ""), etf_date, sessions)


def load_close_history(tickers, start_date, end_date):
    """Unadjusted closes (docs/etf_system.md: close-to-close; matches NAV valuation)."""
    if yf is None:
        return {}

    tickers = sorted({t for t in tickers if t})
    if not tickers:
        return {}

    try:
        data = yf.download(tickers, start=start_date, end=end_date,
                           auto_adjust=False, progress=False)
    except Exception:
        return {}

    if data is None or data.empty or "Close" not in data:
        return {}

    close = data["Close"]
    history = {}
    if hasattr(close, "columns"):
        for ticker in tickers:
            if ticker not in close.columns:
                continue
            series = close[ticker].dropna()
            if not series.empty:
                history[ticker] = series
    elif len(tickers) == 1:
        series = close.dropna()
        if not series.empty:
            history[tickers[0]] = series
    return history


def close_point_on_or_before(series, date_str):
    if series is None or series.empty or not date_str:
        return None
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None
    eligible = series[series.index.date <= target]
    if eligible.empty:
        return None
    try:
        return eligible.index[-1].date(), float(eligible.iloc[-1])
    except (TypeError, ValueError):
        return None


def close_return_pct(close_history, ticker, base_date, target_date):
    if not ticker or not base_date or not target_date:
        return None
    series = close_history.get(ticker)
    base_point = close_point_on_or_before(series, base_date)
    target_point = close_point_on_or_before(series, target_date)
    if not base_point or not target_point:
        return None
    base_actual_date, base = base_point
    target_actual_date, target = target_point
    if target_actual_date <= base_actual_date:
        return None
    if base is None or target is None or base <= 0:
        return None
    return (target / base - 1.0) * 100.0


def format_close_return(value):
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def list_snapshot_dates_for_ticker(etf_ticker):
    if not os.path.isdir(HOLDINGS_DIR):
        return []
    suffix = f"_{etf_ticker}.json"
    out = []
    for fn in os.listdir(HOLDINGS_DIR):
        if fn.endswith(suffix):
            out.append(fn[:10])
    return sorted(out)


_SESSION_CACHE = {}


def get_krx_sessions(through_date):
    """KRX 거래 세션 목록. D-n과 가격 기준일은 평일이 아니라 실제 세션으로 센다.

    두 펀드의 검증 스냅샷 날짜(모두 KRX 세션)에 두 ETF 시장가격 거래일을 합친다.
    가격 조회가 실패해도 스냅샷 날짜만으로 동작한다.
    """
    cache_key = (HOLDINGS_DIR, through_date)
    if cache_key in _SESSION_CACHE:
        return _SESSION_CACHE[cache_key]
    snapshot_dates = [
        d for etf in PERFORMANCE_ETFS
        for d in list_snapshot_dates_for_ticker(etf["snapshot_ticker"])
    ]
    through = datetime.strptime(through_date, "%Y-%m-%d").date()
    history = load_close_history(
        [etf["price_ticker"] for etf in PERFORMANCE_ETFS],
        (through - timedelta(days=120)).strftime("%Y-%m-%d"),
        (through + timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    price_dates = [
        idx.strftime("%Y-%m-%d")
        for series in history.values()
        for idx in series.index
    ]
    sessions = [
        d for d in market.merge_sessions(snapshot_dates, price_dates)
        if d <= through_date
    ]
    _SESSION_CACHE[cache_key] = sessions
    return sessions


def default_performance_target_date():
    date_sets = []
    for etf in PERFORMANCE_ETFS:
        dates = list_snapshot_dates_for_ticker(etf["snapshot_ticker"])
        if not dates:
            return None
        date_sets.append(set(dates))
    common_dates = set.intersection(*date_sets) if date_sets else set()
    if common_dates:
        return sorted(common_dates)[-1]
    all_dates = [d for dates in date_sets for d in dates]
    return sorted(all_dates)[-1] if all_dates else None


def performance_comparison_dates(target, horizons, sessions=None):
    """Use the same KRX-session D-n and snapshot fallback as the holding report."""
    date_sets = [
        set(list_snapshot_dates_for_ticker(etf["snapshot_ticker"]))
        for etf in PERFORMANCE_ETFS
    ]
    common_dates = set.intersection(*date_sets) if date_sets else set()
    fallback_dates = set.union(*date_sets) if date_sets else set()
    all_snapshot_dates = sorted(common_dates or fallback_dates)
    out = {}
    for label, n in horizons:
        intended, used = market.select_base_date(
            all_snapshot_dates, target, n, sessions,
        )
        out[label] = used or intended
    return out


def format_price(value):
    if value is None:
        return "N/A"
    return f"{value:,.0f}"


def fund_performance_report(target_date=None):
    """Compare the two tracked funds by ETF market-price close returns."""
    target = target_date or default_performance_target_date()
    if not target:
        print("  No ETF snapshots available for performance comparison.")
        return

    horizons = [("D-1", 1), ("D-3", 3), ("D-5", 5), ("D-20", 20)]
    base_dates = performance_comparison_dates(
        target, horizons, get_krx_sessions(target),
    )
    all_dates = [target] + list(base_dates.values())
    parsed_dates = [datetime.strptime(d, "%Y-%m-%d").date()
                    for d in all_dates]
    price_start = (min(parsed_dates) - timedelta(days=7)).strftime("%Y-%m-%d")
    price_end = (max(parsed_dates) + timedelta(days=3)).strftime("%Y-%m-%d")
    price_tickers = [etf["price_ticker"] for etf in PERFORMANCE_ETFS]
    close_history = load_close_history(price_tickers, price_start, price_end)

    results = {}
    for etf in PERFORMANCE_ETFS:
        ticker = etf["price_ticker"]
        series = close_history.get(ticker)
        target_point = close_point_on_or_before(series, target)
        etf_results = {}
        for label, base_date in base_dates.items():
            base_point = close_point_on_or_before(series, base_date)
            ret = close_return_pct(close_history, ticker, base_date, target)
            etf_results[label] = {
                "base_date": base_date,
                "base_point": base_point,
                "target_point": target_point,
                "return": ret,
            }
        results[etf["label"]] = etf_results

    print()
    print("### [두 펀드 성과 비교] — ETF 시장가격 종가 기준")
    print(f"  기준일: {target}")
    print("  성과 판정은 ETF 가격 수익률 기준이며, AUM/설정현금액 변화는 제외합니다.")
    print("  가격 티커: TIME=456600.KS, KoAct=0015B0.KS")
    print(f"  {'구간':<5} {'비교일':<12} {'TIME':>22} {'KoAct':>22}"
          f" {'차이':>10} {'우위':>8}")
    print(f"  {'-'*88}")

    for label, _ in horizons:
        base_date = base_dates[label]
        cells = {}
        returns = {}
        for etf in PERFORMANCE_ETFS:
            etf_result = results.get(etf["label"], {}).get(label, {})
            base_point = etf_result.get("base_point")
            target_point = etf_result.get("target_point")
            ret = etf_result.get("return")
            if base_point and target_point and ret is not None:
                base_actual, base_close = base_point
                target_actual, target_close = target_point
                date_note = "" if str(base_actual) == base_date else f"@{base_actual}"
                target_note = "" if str(target_actual) == target else f"@{target_actual}"
                cells[etf["label"]] = (
                    f"{format_price(base_close)}{date_note}->"
                    f"{format_price(target_close)}{target_note} "
                    f"{ret:+.2f}%"
                )
                returns[etf["label"]] = ret
            else:
                cells[etf["label"]] = "N/A"
                returns[etf["label"]] = None

        if returns.get("TIME") is None or returns.get("KoAct") is None:
            diff_txt = "N/A"
            winner = "N/A"
        else:
            diff = returns["TIME"] - returns["KoAct"]
            diff_txt = f"{diff:+.2f}%p"
            winner = "TIME" if diff > 0 else "KoAct" if diff < 0 else "동률"

        print(f"  {label:<5} {base_date:<12} {cells['TIME']:>22}"
              f" {cells['KoAct']:>22} {diff_txt:>10} {winner:>8}")
    print()


def _fund_label(alias):
    return "TIME" if alias == "time" else "KoAct"


def _load_fund_snapshots(snapshot_ticker, through_date):
    snapshots = {}
    for date_str in list_snapshot_dates_for_ticker(snapshot_ticker):
        if date_str > through_date:
            continue
        path = Path(HOLDINGS_DIR) / f"{date_str}_{snapshot_ticker}.json"
        snapshots[date_str] = load_normalized_snapshot(path)
    return snapshots


def _short_code(code):
    return (code or "").replace(" US EQUITY", "")


def _format_consensus_item(company, flows_by_label):
    parts = [
        f"{label} {qty_flows.format_flow_label(flow)}"
        for label, flow in flows_by_label
        if flow is not None
    ]
    return f"{_short_code(company)} ({', '.join(parts)})"


def core_digest_report(target_date):
    """핵심 결론 요약: 두 펀드의 수량 기준 실매매 합의·디버전스와 미국 상장 관찰군."""
    sessions = get_krx_sessions(target_date)
    primary = observation_primary_alias()
    secondary = next(alias for alias in _FUND_CONFIGS if alias != primary)
    labels = {alias: _fund_label(alias) for alias in (primary, secondary)}
    snapshots = {
        alias: _load_fund_snapshots(_FUND_CONFIGS[alias].snapshot_ticker, target_date)
        for alias in (primary, secondary)
    }
    missing = [labels[a] for a in snapshots if target_date not in snapshots[a]]
    if missing:
        print(f"  {', '.join(missing)} {target_date} 스냅샷이 없어 교차 요약을 만들지 않았습니다.")
        return

    results = {}
    for alias, snaps in snapshots.items():
        dates = sorted(snaps)
        for n in (5, 20):
            _, used = market.select_base_date(dates, target_date, n, sessions)
            results[(alias, n)] = qty_flows.flows_between(snaps, used, target_date)

    print("### [두 펀드 실매매 합의] — 수량 기준(공통 배율·분할 제거)")
    print("  두 펀드가 같은 종목을 실제로 함께 사고판 경우와 방향이 엇갈린 경우입니다. "
          "비중 변화가 아니라 수량 변화로 판정합니다.")
    for n in (5, 20):
        a, b = results[(primary, n)], results[(secondary, n)]
        if a is None or b is None:
            print(f"  D-{n}: 비교 스냅샷 부족")
            continue
        print(f"  D-{n} ({labels[primary]} {a.base_date}→{a.current_date}, "
              f"{labels[secondary]} {b.base_date}→{b.current_date})")
        buckets = qty_flows.cross_fund_consensus(a, b)
        for key, title in (("common_buy", "공통 실매수"), ("common_sell", "공통 실매도"),
                           ("divergence", "디버전스")):
            items = buckets[key]
            rendered = ", ".join(
                _format_consensus_item(company, [(labels[primary], fa), (labels[secondary], fb)])
                for company, fa, fb in items
            )
            print(f"    {title} {len(items)}: {rendered or '없음'}")

    primary_snaps = snapshots[primary]
    primary_dates = sorted(primary_snaps)
    current = primary_snaps[target_date]
    entry_map = build_entry_date_map(primary_dates, target_date, snapshots=primary_snaps)
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

    print()
    print(f"### [{labels[primary]} 미국 상장 Top 10 관찰군] — 기준일 {target_date}")
    print(f"  조사 우선순위 목록이며 주문표가 아닙니다. 비중은 {labels[primary]}의 NAV 비중, "
          "실매매는 수량 기준입니다.")
    print(f"  {'순위':>3}  {'종목':<8} {labels[primary] + ' 비중':>10} {'실매매 D-5':<14}"
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
        print(f"  {rank:>3}  {_short_code(row['code']):<8} {row['weight']:>9.2f}%"
              f" {flow_cell(flows5, row['key']):<14} {flow_cell(flows20, row['key']):<14}"
              f" {entry_text:<9} {other_weight:>11} {other_flow:<14}")
    print(f"  Top 10 비중 합계 {sum(row['weight'] for row in top):.2f}%")

    for n in (5, 20):
        result = results[(primary, n)]
        if result is None:
            continue
        before = {row["key"] for row in qty_flows.us_listed_top(primary_snaps[result.base_date], 10)}
        after = {row["key"] for row in top}
        entered = ", ".join(sorted(_short_code(k) for k in after - before)) or "없음"
        left = ", ".join(sorted(_short_code(k) for k in before - after)) or "없음"
        print(f"  Top 10 교체(D-{n}, {result.base_date} 대비): 진입 {entered} / 이탈 {left}")


def is_cash_holding(holding):
    """Return True for the canonical cash row or a legacy cash equivalent."""
    return market.is_cash_holding(holding)


def get_holding_sector(holding):
    """보유 종목을 GBRC 섹터로 분류."""
    if is_cash_holding(holding):
        return "현금"

    key = get_listing_key(holding.get("code", ""))
    return resolve_sector(key)


def is_business_day(d):
    return d.weekday() < 5


def prev_business_days(target_date_str, n):
    """target 기준 n번째 직전 영업일 (n=1이면 어제, n=5이면 5거래일 전)"""
    d = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    count = 0
    while count < n:
        d -= timedelta(days=1)
        if is_business_day(d):
            count += 1
    return d.strftime("%Y-%m-%d")


def select_comparison_snapshot(dates, target, n, sessions=None, snapshots=None):
    """Select the snapshot for D-n, counting actual KRX sessions.

    ``sessions`` 없이 호출하면 스냅샷 날짜를 세션 달력으로 쓴다. 의도한 세션에
    스냅샷이 없으면 그 이전의 가장 가까운 스냅샷으로 물러선다.
    """
    intended, used = market.select_base_date(dates, target, n, sessions)
    if not used:
        return intended, None
    if snapshots is not None and used in snapshots:
        return used, snapshots[used]
    return used, load_snapshot(used)


def backfill(start_date_str, end_date_str):
    """start ~ end 영업일 모두 크롤링"""
    start = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    cursor = start
    fetched = 0
    while cursor <= end:
        if is_business_day(cursor):
            ds = cursor.strftime("%Y-%m-%d")
            existing = load_snapshot(ds)
            if existing and existing.get("holdings_count", 0) > 0:
                print(f"  [skip] {ds} (이미 존재, {existing['holdings_count']}종목)")
            else:
                snap = fetch_holdings(ds)
                if snap and snap["holdings_count"] > 0:
                    save_snapshot(snap)
                    print(f"  [OK]   {ds} → {snap['holdings_count']}종목 저장")
                    fetched += 1
                else:
                    print(f"  [FAIL] {ds} 파싱 실패")
        cursor += timedelta(days=1)
    print(f"\n  완료: {fetched}건 신규 저장")


# ───────────────────────────────────────────────
# 비교 분석
# ───────────────────────────────────────────────


def calc_relative_change(prev_weight, curr_weight):
    """이전 비중 대비 상대 변화율(%). 절대 %p 변화와 함께 해석한다."""
    if prev_weight is None or prev_weight == 0:
        return None
    return ((curr_weight - prev_weight) / prev_weight) * 100


def format_relative_change(value):
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def get_cash_weight(snap):
    cash = find_cash_holding(snap)
    if not cash:
        return None
    return cash.get("weight")


def classify_cash_regime(dates, target):
    """최근 3/5개 스냅샷의 현금 연속 증감으로 리스크 상태 판정."""
    cash_series = []
    for date_str in sorted(d for d in dates if d <= target):
        snap = load_snapshot(date_str)
        weight = get_cash_weight(snap)
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


def find_cash_holding(snap):
    """스냅샷에서 현금 라인을 찾는다."""
    if not snap:
        return None
    for h in snap.get("holdings", []):
        if is_cash_holding(h):
            return h
    return None


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
            key = get_listing_key(h.get("code", ""))
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


def _top_five_security_weight(snap):
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


def _format_sector_rotation(curr_snap, prev_snap):
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
    lines.append(f"  D-5 섹터 로테이션: {_format_sector_rotation(curr_snap, prev5)}")

    current_top5 = _top_five_security_weight(curr_snap)
    previous_top5 = _top_five_security_weight(prev5)
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


def format_previous_change_review(curr_snap, dates, horizon_snaps, diff1,
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

    previous_cash_label, _ = classify_cash_regime(dates, prev_date)
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

    _, previous_d5 = select_comparison_snapshot(dates, prev_date, 5, sessions)
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

    previous_top5 = _top_five_security_weight(prev_snap)
    previous_d5_top5 = _top_five_security_weight(previous_d5)
    current_top5 = _top_five_security_weight(curr_snap)
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


def compare_report_legacy(target_date=None):
    return compare_report(target_date)


def _load_snapshot_cache(dates, through_date):
    """Load every snapshot up to through_date once per report."""
    cache = {}
    for date_str in dates:
        if date_str > through_date:
            continue
        snap = load_snapshot(date_str)
        if snap:
            cache[date_str] = snap
    return cache


def _weights_by_key(snap):
    """All rows including cash (key ''), summed by holding_key."""
    weights = {}
    for holding in snap.get("holdings", []):
        key = market.holding_key(holding.get("code", ""))
        weights[key] = weights.get(key, 0.0) + holding["weight"]
    return weights


def compare_report(target_date=None):
    """Compare ETF holdings and print the ETF-only briefing report."""
    dates = list_snapshot_dates()
    if not dates:
        print("  No ETF snapshots. Run --today or --backfill first.")
        return

    target = target_date or dates[-1]
    curr = load_snapshot(target)
    if not curr:
        past = [d for d in dates if d <= target]
        if past:
            print(f"  ({target} snapshot missing; using {past[-1]})")
            target = past[-1]
            curr = load_snapshot(target)
    if not curr:
        print(f"  No snapshot available on or before {target}.")
        return

    print(f"\n{'='*100}")
    print(f"  {ETF_NAME} ({ETF_TICKER}) holdings change report - {target}")
    print(f"  Holdings {curr['holdings_count']}, total weight "
          f"{sum(h['weight'] for h in curr['holdings']):.2f}%")
    print(f"{'='*100}")

    snapshots = _load_snapshot_cache(dates, target)
    sessions = get_krx_sessions(target)
    horizons = [("D-1 compare", 1), ("D-3 compare", 3),
                ("D-5 compare", 5), ("D-20 compare", 20)]
    horizon_snaps = {}
    horizon_diffs = []
    horizon_diffs_by_n = {}
    flow_results_by_n = {}
    entry_date_map = build_entry_date_map(dates, target, snapshots=snapshots)
    cash_regime = classify_cash_regime(dates, target)

    for label, n in horizons:
        prev_date, prev = select_comparison_snapshot(
            dates, target, n, sessions, snapshots,
        )
        horizon_snaps[n] = (prev_date, prev)
        flow_result = (
            qty_flows.flows_between(snapshots, prev_date, target) if prev else None
        )
        flow_results_by_n[n] = flow_result
        diff = diff_holdings(curr, prev, entry_date_map=entry_date_map,
                             flow_result=flow_result) if prev else None
        horizon_diffs.append((label, diff))
        horizon_diffs_by_n[n] = diff

    print(format_top10_section(curr))
    print(format_cash_section(curr, horizon_snaps, entry_date_map,
                              cash_regime))
    print(format_sector_section(curr, horizon_snaps, entry_date_map))
    print(format_quantity_flow_section(curr, flow_results_by_n))

    for label, diff in horizon_diffs:
        if not diff:
            print(f"\n### {label}: no comparison snapshot")
            continue
        print(format_diff_section(diff, label))

    print(format_d20_new_cluster(horizon_diffs_by_n.get(20)))

    print(f"\n{'='*146}")
    print("  [One-glance table] all current holdings x "
          "D-1/D-3/D-5/D-20 weight change + close return")
    print(f"{'='*146}")

    horizon_maps = {}
    horizon_available = {}
    base_dates = []
    for n in (1, 3, 5, 20):
        snap_date, snap = horizon_snaps.get(n, (None, None))
        horizon_available[n] = snap is not None
        horizon_maps[n] = _weights_by_key(snap) if snap else {}
        base_dates.append(f"D-{n}={snap_date or 'N/A'}")
    print(f"  (comparison dates: {' / '.join(base_dates)})")
    print("  dW is weight change in percentage points; return is "
          "close-to-close from that comparison date to current date.")
    print("  Price dates follow the NAV valuation: KR = previous KRX session close, "
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
    close_history = load_close_history(
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
        base_price_date = price_reference_date_for_holding(
            holding, base_date, sessions)
        target_price_date = price_reference_date_for_holding(
            holding, target, sessions)
        ret = close_return_pct(close_history, market_ticker_by_code.get(code),
                               base_price_date, target_price_date)
        return f"{delta:+.2f}({format_close_return(ret)})".rjust(18)

    print(f"  {'Rank':>3}  {'Code':<18} {'Name':<32} {'Weight':>7}"
          f" {'D-1 dW(ret)':>18} {'D-3 dW(ret)':>18}"
          f" {'D-5 dW(ret)':>18} {'D-20 dW(ret)':>18}")
    print(f"  {'-'*146}")

    ranked_holdings = sorted(curr["holdings"],
                             key=lambda x: -x["weight"])[:REPORT_RANK_LIMIT]
    for i, h in enumerate(ranked_holdings, 1):
        key = market.holding_key(h["code"])
        cells = []
        for n in (1, 3, 5, 20):
            cells.append(fmt_delta_return(
                h["weight"], horizon_maps[n].get(key), horizon_available[n],
                h["code"], horizon_snaps.get(n, (None, None))[0],
            ))
        print(f"  {i:>3}  {h['code']:<18} {h['name'][:30]:<32}"
              f" {h['weight']:>6.2f}% {' '.join(cells)}")
    print()
    print(format_portfolio_interpretation(
        curr, horizon_snaps, horizon_diffs_by_n, cash_regime,
        flow_results_by_n,
    ))
    print(format_previous_change_review(
        curr, dates, horizon_snaps, horizon_diffs_by_n.get(1), sessions,
    ))
    print()

if __name__ == "__main__":
    set_active_etf(pop_etf_arg(sys.argv))

    if "--backfill" in sys.argv:
        idx = sys.argv.index("--backfill")
        start, end = sys.argv[idx + 1], sys.argv[idx + 2]
        backfill(start, end)
    elif "--today" in sys.argv:
        today = datetime.now().strftime("%Y-%m-%d")
        snap = fetch_holdings(today)
        if snap and snap["holdings_count"] > 0:
            path = save_snapshot(snap)
            print(f"  저장: {path} ({snap['holdings_count']}종목)")
        else:
            print(f"  {today} 데이터 없음 (휴일?)")
    elif "--compare" in sys.argv:
        idx = sys.argv.index("--compare")
        target = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else None
        compare_report(target)
    elif "--performance" in sys.argv:
        idx = sys.argv.index("--performance")
        target = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else None
        fund_performance_report(target)
    else:
        print(__doc__)
