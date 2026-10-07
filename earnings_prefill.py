"""
GBRC 실적 분석 사전 데이터 확보 스크립트 (하네스 엔지니어링)

실행: py earnings_prefill.py <TICKER>
목적: 황금률 #1 위반 (데이터 가정) 을 구조적으로 차단.
      실적 분석 진입 전 반드시 이 스크립트로 필수 데이터 확보.

출력:
  - 콘솔: 6단계 분석에 필요한 모든 지표 요약
  - data/earnings_prefill/<DATE>_<TICKER>.json: 후속 분석 입력용
"""

import sys
import os
import json
import warnings
from datetime import date, datetime

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


def _num(v):
    v = safe(v, None)
    return float(v) if v is not None else None


def classify_session(ts) -> str:
    """발표 시각(미 동부) → 장전/장중/장후. 장후 발표면 첫 가격 반응은 다음 거래일."""
    minutes = ts.hour * 60 + ts.minute
    if minutes < 9 * 60 + 30:
        return "장전"
    if minutes < 12 * 60:
        return "장중"
    return "장후"


def fetch_earnings_events(t, limit: int = 12) -> list[dict]:
    """yfinance get_earnings_dates → 최신순 실적 이벤트 목록.

    .calendar는 다음 발표일만 주고 장후 발표를 하루 늦게 준다
    (2026-09-27 실측: MU 공식 9/30 → calendar 10/1). 옵션 스캐너와 같은 함수를 쓴다.
    """
    try:
        frame = t.get_earnings_dates(limit=limit)
    except Exception:
        return []
    if frame is None or frame.empty:
        return []
    events = [
        {
            "date": ts.date(),
            "time_et": ts.strftime("%H:%M"),
            "session": classify_session(ts),
            "eps_estimate": _num(row.get("EPS Estimate")),
            "eps_reported": _num(row.get("Reported EPS")),
            "eps_surprise_pct": _num(row.get("Surprise(%)")),
        }
        for ts, row in frame.iterrows()
    ]
    return sorted(events, key=lambda e: e["date"], reverse=True)


def select_earnings_event(events: list[dict], today: date) -> dict | None:
    """이번 분석 대상 실적 이벤트를 고른다.

    events: fetch_earnings_events() 결과(최신순). 과거 이벤트는 eps_reported가 있고
            미래 이벤트는 None이다.
    반환: 고른 이벤트, 고를 것이 없으면 None.

    배경: 기존 .calendar는 '다음' 발표일만 줘서 발표 다음 날 실행하면 다음 분기
          날짜가 저장됐다(2026-04 RMBS·GOOGL·META·STX 4건).

    규칙: 오늘에 가장 가까운 발표. 직전 발표와 다음 발표의 중간 지점까지는 방금 발표한
          분기를 분석하는 중으로 본다(고정 일수 없이 회사별 발표 간격에 맞춰진다).
          거리가 같으면 지난 발표를 고른다. 실제치 유무가 아니라 날짜로 가려서,
          yfinance 반영이 늦어 실제치가 빈 지난 발표도 같은 분기로 잡는다.
    """
    if not events:
        return None
    return min(events, key=lambda e: (abs((e["date"] - today).days), e["date"] > today))


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

    # 실적 일정 + EPS 컨센서스 (발표 시각·추정치·실제치·서프라이즈)
    today = date.today()
    event = select_earnings_event(fetch_earnings_events(t), today)
    earnings_event = None
    if event:
        earnings_event = {
            **event,
            "date": event["date"].isoformat(),
            "days_from_collection": (event["date"] - today).days,
            "source": "yfinance.get_earnings_dates",
        }

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
        "earnings_date": earnings_event["date"] if earnings_event else None,
        "earnings_event": earnings_event,
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
    ev = data.get("earnings_event")
    if ev:
        print(f"  실적 발표일    : {ev['date']} {ev['time_et']} ET ({ev['session']}, "
              f"수집일 기준 {ev['days_from_collection']:+d}일)")
    else:
        print("  실적 발표일    : N/A")

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

    print("\n[EPS 컨센서스 — S&N ① 실적의 강도]")
    if ev:
        surprise = ev["eps_surprise_pct"]
        print(f"  EPS Estimate   : {safe(ev['eps_estimate'])}")
        print(f"  Reported EPS   : {safe(ev['eps_reported'])}")
        print(f"  Surprise       : {f'{surprise:+.2f}%' if surprise is not None else 'N/A'}")
        print(f"  출처           : {ev['source']}")
    else:
        print("  N/A")

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
    print("  ⚠️  누락된 데이터는 반드시 보완 후 분석 진입할 것 (순서: earnings_framework.md 데이터 소스)")
    if not ev or ev["eps_estimate"] is None:
        print("      - EPS 컨센서스 (TradingView earnings_per_share_forecast_fq → WebSearch)")
    print("      - 매출 컨센서스 (yfinance 미제공 — TradingView revenue_forecast_fq → WebSearch)")
    print("      - 전년 동기 실제 실적 (분기별 YoY 비교용)")
    print("      - 기존 가이던스 (raise/cut 판정용 — TradingView 실적콜 녹취록 → WebSearch)")
    print("=" * 80)


def earnings_log_to_md(json_path: str) -> str:
    """실적 분석 JSON → 마크다운 요약"""
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)

    sn = d.get("sn_score", {})
    vr = d.get("verification") or {}
    lines = [
        f"# {d['ticker']} 실적 분석 — {d.get('analysis_date', '')}",
        f"**분석 기간**: {d.get('fiscal_period', '')}  |  **분석 시 주가**: ${d.get('price_at_analysis', 'N/A')}  |  **시총**: {d.get('market_cap_at_analysis', 'N/A')}",
        "",
        "## 핵심 지표",
        "| 항목 | 값 |",
        "|---|---|",
    ]
    kn = d.get("key_numbers", {})
    for k, v in kn.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## S&N 스코어",
        f"**총점: {sn.get('total', 'N/A')}/33**",
        "",
        "| 항목 | 점수 |",
        "|---|---|",
    ]
    for k, v in sn.get("breakdown", {}).items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        f"**퀀트 등급**: {d.get('quant_tier', 'N/A')}",
        "",
        "## 시장 반응 예측",
        f"- 초반: {d.get('market_reaction_prediction', {}).get('initial', 'N/A')}",
        f"- 중장기: {d.get('market_reaction_prediction', {}).get('mid_long_term', 'N/A')}",
        "",
        f"## 투자의견: {d.get('investment_opinion', 'N/A')}",
        "",
        f"**핵심 논리**: {d.get('key_thesis', 'N/A')}",
        "",
        "**리스크**:",
    ]
    for r in d.get("key_risks", []):
        lines.append(f"- {r}")

    if vr:
        lines += ["", "## 검증 결과"]
        for k, v in vr.items():
            lines.append(f"- **{k}**: {v}")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("사용법: py earnings_prefill.py <TICKER>")
        print("예시:   py earnings_prefill.py UNH")
        sys.exit(1)

    ticker = sys.argv[1].upper().strip()
    data = collect(ticker)

    out_dir = os.path.join("data", "earnings_prefill")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{datetime.now().strftime('%Y-%m-%d')}_{ticker}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    print_report(data)
    print(f"\n  저장: {out_path}\n")


if __name__ == "__main__":
    main()
