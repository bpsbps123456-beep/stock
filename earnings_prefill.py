"""
GBRC 실적 분석 사전 데이터 확보 스크립트 (하네스 엔지니어링)

실행: py earnings_prefill.py <TICKER>
목적: 황금률 #1 위반 (데이터 가정) 을 구조적으로 차단.
      실적 분석 진입 전 반드시 이 스크립트로 필수 데이터 확보.

출력:
  - 콘솔: 6단계 분석에 필요한 모든 지표 요약
  - data/earnings_prefill/<TICKER>_<DATE>.json: 후속 분석 입력용
"""

import sys
import os
import json
import warnings
from datetime import datetime

import yfinance as yf

warnings.filterwarnings("ignore")


def safe(v, default="N/A"):
    if v is None:
        return default
    try:
        if isinstance(v, float) and (v != v):
            return default
    except Exception:
        pass
    return v


def fmt_money(v):
    if v is None or v == "N/A":
        return "N/A"
    try:
        v = float(v)
    except Exception:
        return str(v)
    if abs(v) >= 1e12:
        return f"${v / 1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v:,.0f}"


def fmt_pct(v):
    if v is None or v == "N/A":
        return "N/A"
    try:
        return f"{float(v) * 100:+.2f}%"
    except Exception:
        return str(v)


def tier_from_mcap(mcap):
    if mcap is None or mcap == "N/A":
        return "N/A"
    mcap = float(mcap)
    if mcap >= 300e9:
        return "메가캡 (>$300B)"
    if mcap >= 50e9:
        return "대형주 ($50~300B)"
    if mcap >= 10e9:
        return "중형주 ($10~50B)"
    if mcap >= 2e9:
        return "소형주 ($2~10B)"
    return "마이크로캡 (<$2B)"


def collect(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info or {}

    # 가격/시총
    price = safe(info.get("currentPrice") or info.get("regularMarketPrice"))
    mcap = safe(info.get("marketCap"))
    shares = safe(info.get("sharesOutstanding"))
    float_shares = safe(info.get("floatShares"))

    # EPS / 컨센서스
    trailing_eps = safe(info.get("trailingEps"))
    forward_eps = safe(info.get("forwardEps"))
    trailing_pe = safe(info.get("trailingPE"))
    forward_pe = safe(info.get("forwardPE"))

    # 마진
    gross_margin = safe(info.get("grossMargins"))
    op_margin = safe(info.get("operatingMargins"))
    net_margin = safe(info.get("profitMargins"))
    fcf_margin = None
    rev = safe(info.get("totalRevenue"))
    fcf = safe(info.get("freeCashflow"))
    if rev not in (None, "N/A") and fcf not in (None, "N/A"):
        try:
            fcf_margin = float(fcf) / float(rev)
        except Exception:
            fcf_margin = None

    # 성장
    rev_growth = safe(info.get("revenueGrowth"))
    earnings_growth = safe(info.get("earningsGrowth"))

    # 부채
    total_debt = safe(info.get("totalDebt"))
    total_cash = safe(info.get("totalCash"))
    ebitda = safe(info.get("ebitda"))
    net_debt = None
    nd_to_ebitda = None
    if total_debt not in (None, "N/A") and total_cash not in (None, "N/A"):
        try:
            net_debt = float(total_debt) - float(total_cash)
        except Exception:
            net_debt = None
    if net_debt is not None and ebitda not in (None, "N/A"):
        try:
            nd_to_ebitda = net_debt / float(ebitda)
        except Exception:
            nd_to_ebitda = None

    # 컨센서스 (이것이 핵심 - 애널리스트 추천)
    target_mean = safe(info.get("targetMeanPrice"))
    target_high = safe(info.get("targetHighPrice"))
    target_low = safe(info.get("targetLowPrice"))
    recommendation = safe(info.get("recommendationKey"))
    num_analysts = safe(info.get("numberOfAnalystOpinions"))

    # 실적 일정
    earnings_date = None
    try:
        cal = t.calendar
        if cal is not None and hasattr(cal, "get"):
            edt = cal.get("Earnings Date")
            if isinstance(edt, (list, tuple)) and edt:
                earnings_date = str(edt[0])
    except Exception:
        pass

    # 최근 분기별 매출/EPS
    quarterly = {}
    try:
        qfin = t.quarterly_financials
        if qfin is not None and not qfin.empty:
            cols = [str(c)[:10] for c in qfin.columns[:5]]
            if "Total Revenue" in qfin.index:
                revs = qfin.loc["Total Revenue"].tolist()[:5]
                quarterly["revenue"] = dict(zip(cols, [float(x) if x == x else None for x in revs]))
            if "Net Income" in qfin.index:
                nis = qfin.loc["Net Income"].tolist()[:5]
                quarterly["net_income"] = dict(zip(cols, [float(x) if x == x else None for x in nis]))
    except Exception:
        pass

    # 연간 실적
    annual = {}
    try:
        afin = t.financials
        if afin is not None and not afin.empty:
            cols = [str(c)[:10] for c in afin.columns[:4]]
            if "Total Revenue" in afin.index:
                revs = afin.loc["Total Revenue"].tolist()[:4]
                annual["revenue"] = dict(zip(cols, [float(x) if x == x else None for x in revs]))
            if "Net Income" in afin.index:
                nis = afin.loc["Net Income"].tolist()[:4]
                annual["net_income"] = dict(zip(cols, [float(x) if x == x else None for x in nis]))
    except Exception:
        pass

    # 섹터
    sector = safe(info.get("sector"))
    industry = safe(info.get("industry"))

    return {
        "ticker": ticker.upper(),
        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sector": sector,
        "industry": industry,
        "price": price,
        "market_cap": mcap,
        "market_cap_tier": tier_from_mcap(mcap),
        "shares_outstanding": shares,
        "float_shares": float_shares,
        "trailing_eps": trailing_eps,
        "forward_eps": forward_eps,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "gross_margin": gross_margin,
        "operating_margin": op_margin,
        "net_margin": net_margin,
        "fcf_margin": fcf_margin,
        "revenue_ttm": rev,
        "fcf_ttm": fcf,
        "revenue_growth_yoy": rev_growth,
        "earnings_growth_yoy": earnings_growth,
        "ebitda_ttm": ebitda,
        "total_debt": total_debt,
        "total_cash": total_cash,
        "net_debt": net_debt,
        "net_debt_to_ebitda": nd_to_ebitda,
        "analyst_target_mean": target_mean,
        "analyst_target_high": target_high,
        "analyst_target_low": target_low,
        "analyst_recommendation": recommendation,
        "num_analysts": num_analysts,
        "earnings_date": earnings_date,
        "quarterly_financials": quarterly,
        "annual_financials": annual,
        "rule_of_40_estimate": (
            (float(rev_growth) * 100 + (fcf_margin or 0) * 100)
            if rev_growth not in (None, "N/A") and fcf_margin is not None
            else None
        ),
    }


def print_report(data: dict) -> None:
    t = data["ticker"]
    print()
    print("=" * 80)
    print(f"  GBRC 실적 분석 사전 데이터 — {t}")
    print(f"  수집 시각: {data['collected_at']}")
    print("=" * 80)

    print(f"\n[기본 정보]")
    print(f"  섹터/산업      : {data['sector']} / {data['industry']}")
    print(f"  현재가         : {fmt_money(data['price'])}")
    print(f"  시가총액       : {fmt_money(data['market_cap'])}  [{data['market_cap_tier']}]")
    print(f"  유통주식수     : {fmt_money(data['shares_outstanding']).replace('$', '')}")
    print(f"  실적 발표일    : {data['earnings_date']}")

    print(f"\n[밸류에이션 — 황금률 #1 필수]")
    print(f"  Trailing EPS   : ${data['trailing_eps']}")
    print(f"  Forward EPS    : ${data['forward_eps']}")
    print(f"  Trailing PER   : {data['trailing_pe']}")
    print(f"  Forward PER    : {data['forward_pe']}")

    print(f"\n[마진 — 황금률 #7 필수]")
    print(f"  Gross Margin   : {fmt_pct(data['gross_margin'])}")
    print(f"  Op Margin      : {fmt_pct(data['operating_margin'])}")
    print(f"  Net Margin     : {fmt_pct(data['net_margin'])}")
    print(f"  FCF Margin     : {fmt_pct(data['fcf_margin'])}")

    print(f"\n[성장 — 황금률 #3 필수]")
    print(f"  Revenue YoY    : {fmt_pct(data['revenue_growth_yoy'])}")
    print(f"  Earnings YoY   : {fmt_pct(data['earnings_growth_yoy'])}")
    print(f"  Rule of 40     : {data['rule_of_40_estimate']:.1f}" if data['rule_of_40_estimate'] is not None else "  Rule of 40     : N/A")

    print(f"\n[재무건전성 — S&N ⑤번 필수]")
    print(f"  Total Debt     : {fmt_money(data['total_debt'])}")
    print(f"  Total Cash     : {fmt_money(data['total_cash'])}")
    print(f"  Net Debt       : {fmt_money(data['net_debt'])}")
    print(f"  ND/EBITDA      : {data['net_debt_to_ebitda']:.2f}x" if data['net_debt_to_ebitda'] is not None else "  ND/EBITDA      : N/A")

    print(f"\n[애널리스트 컨센서스]")
    print(f"  Target Mean    : {fmt_money(data['analyst_target_mean'])}")
    print(f"  Target High    : {fmt_money(data['analyst_target_high'])}")
    print(f"  Target Low     : {fmt_money(data['analyst_target_low'])}")
    print(f"  Recommendation : {data['analyst_recommendation']}")
    print(f"  # Analysts     : {data['num_analysts']}")

    if data['quarterly_financials']:
        print(f"\n[최근 분기별 매출 (Revenue)]")
        for k, v in data['quarterly_financials'].get('revenue', {}).items():
            print(f"  {k} : {fmt_money(v)}")

    if data['annual_financials']:
        print(f"\n[연간 매출]")
        for k, v in data['annual_financials'].get('revenue', {}).items():
            print(f"  {k} : {fmt_money(v)}")

    print()
    print("=" * 80)
    print("  ⚠️  누락된 데이터는 반드시 WebSearch 로 추가 확보 후 분석 진입할 것")
    print("      - Q1 컨센서스 EPS (yfinance 미제공 — WebSearch 필요)")
    print("      - 전년 동기 실제 실적 (분기별 YoY 비교용)")
    print("      - 기존 가이던스 (이번 발표의 raise/cut 판정용)")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("사용법: py earnings_prefill.py <TICKER>")
        print("예시:   py earnings_prefill.py UNH")
        sys.exit(1)

    ticker = sys.argv[1].upper().strip()
    data = collect(ticker)

    out_dir = os.path.join("data", "earnings_prefill")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}_{datetime.now().strftime('%Y-%m-%d')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    print_report(data)
    print(f"\n  저장: {out_path}\n")


if __name__ == "__main__":
    main()
