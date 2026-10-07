"""
성장기반 리서치 센터 (GBRC) - 데일리 브리핑
실행: py daily_analysis.py
옵션: py daily_analysis.py --refresh  (주간 컨센서스 강제 갱신)
"""

import yfinance as yf
import warnings
import json
import math
import os
import re
import sys
import urllib.request
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    from deep_translator import GoogleTranslator
    _tr = GoogleTranslator(source="en", target="ko")
    def translate_ko(text):
        try:
            return _tr.translate(text) or text
        except Exception:
            return text
except ImportError:
    def translate_ko(text):
        return text

from watchlist import PROFIT_SECTORS, LOSS_SECTORS, PROFIT_TICKERS, LOSS_TICKERS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "data", "estimates_cache.json")
SW_URLS_PATH = os.path.join(BASE_DIR, "data", "simplywall_urls.json")


def _load_simplywall_urls():
    try:
        with open(SW_URLS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

SIMPLYWALL_URLS = _load_simplywall_urls()


# ─── 포매터 ───────────────────────────────────────
def fmt_pct(v):
    return f"{v*100:>+6.1f}%" if v is not None else "    N/A"

def fmt_margin(v):
    return f"{v*100:>6.1f}%" if v is not None else "   N/A"

def fmt_per(v):
    return f"{v:>6.1f}x" if v is not None else "   N/A"

def fmt_evs(v):
    return f"{v:>6.1f}x" if v is not None else "   N/A"

def fmt_accel(v):
    return f"{v*100:>+7.1f}%p" if v is not None else "     N/A"

def impl_margin(eps, rev, shares):
    if eps and rev and rev > 0 and shares:
        return (eps * shares) / rev
    return None

def safe_val(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except:
        return None


# ─── Simply Wall St FY28 실제 컨센서스 스크래핑 ────
SW_ROW_PATTERN = re.compile(
    r'data-cy-id="earnings-and-revenue-growth-row-\d+"[^>]*>'
    r'<td>([^<]+)</td>'
    r'<td[^>]*>([\d,\-\.]+)</td>'
    r'<td[^>]*>([\d,\-\.]+)</td>'
)

def fetch_simplywall_fy28(ticker):
    """Simply Wall St 페이지에서 FY28 매출·순이익 추출. 실패 시 빈 dict."""
    url = SIMPLYWALL_URLS.get(ticker)
    if not url:
        return {}
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return {}

    for date_str, rev_s, ni_s in SW_ROW_PATTERN.findall(html):
        if "2028" in date_str:
            try:
                # 단위: 백만 달러 → 절대값으로 변환
                rev = float(rev_s.replace(",", "")) * 1e6
                ni  = float(ni_s.replace(",", "")) * 1e6
                return {"sw_fy28_rev": rev, "sw_fy28_ni": ni}
            except ValueError:
                return {}
    return {}


# ─── 주간 컨센서스 캐시 ───────────────────────────
def current_week():
    return datetime.now().strftime("%G-W%V")

def load_estimates_cache():
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("week") != current_week():
        return None
    return data.get("tickers", {})

def save_estimates_cache(tickers_data):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"week": current_week(), "tickers": tickers_data}, f, indent=2, ensure_ascii=False)

def fetch_estimates_one(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    est = stock.earnings_estimate
    rev = stock.revenue_estimate

    def sg(df, row, col):
        try:
            return safe_val(df.loc[row, col])
        except:
            return None

    return {
        "shares": info.get("sharesOutstanding") or 0,
        "ev":     info.get("enterpriseValue") or 0,
        "target": info.get("targetMeanPrice"),
        "fy26_eps":   sg(est, "0y",  "avg"),
        "fy27_eps":   sg(est, "+1y", "avg"),
        "fy26_eps_g": sg(est, "0y",  "growth"),
        "fy27_eps_g": sg(est, "+1y", "growth"),
        "fy26_rev":   sg(rev, "0y",  "avg"),
        "fy27_rev":   sg(rev, "+1y", "avg"),
        "fy26_rev_g": sg(rev, "0y",  "growth"),
        "fy27_rev_g": sg(rev, "+1y", "growth"),
    }

def refresh_estimates(all_tickers):
    print("  [주간 컨센서스 갱신 중... yfinance FY26/FY27]")
    estimates = {}
    for t in all_tickers:
        try:
            estimates[t] = fetch_estimates_one(t)
        except Exception as e:
            print(f"    {t}: yfinance 오류 ({e})")
            estimates[t] = {}

    print("  [Simply Wall St FY28 컨센서스 스크래핑 중...]")
    sw_ok, sw_miss = 0, 0
    for t in all_tickers:
        sw = fetch_simplywall_fy28(t)
        if sw:
            estimates[t].update(sw)
            sw_ok += 1
        else:
            sw_miss += 1
    print(f"    SimpyWall 수집 완료 — {sw_ok}종목 ✓ / {sw_miss}종목 N/A")

    save_estimates_cache(estimates)
    print(f"  [컨센서스 캐시 저장 완료 — {current_week()}]\n")
    return estimates


# ─── 현재가 일괄 수집 (일간, API 1회) ────────────
def get_live_prices(all_tickers):
    try:
        df = yf.download(all_tickers, period="2d", progress=False, auto_adjust=True)["Close"]
        return {str(k): safe_val(v) for k, v in df.iloc[-1].items()}
    except:
        return {}


# ─── 뉴스 ─────────────────────────────────────────
def get_news(ticker):
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return []
        titles = []
        seen = set()
        for item in news[:8]:
            content = item.get("content", item)
            title = content.get("title", "") if isinstance(content, dict) else item.get("title", "")
            if not title or title in seen:
                continue
            seen.add(title)
            provider = content.get("provider", {}) if isinstance(content, dict) else {}
            source = provider.get("displayName", "") if isinstance(provider, dict) else item.get("publisher", "")
            titles.append(f"[{source}] {title}")
        return titles[:3]
    except:
        return []


# ─── 지표 계산 ────────────────────────────────────
def compute_profit(est, price):
    shares     = est.get("shares", 0)
    fy26_eps   = est.get("fy26_eps")
    fy27_eps   = est.get("fy27_eps")
    fy27_eps_g = est.get("fy27_eps_g")
    fy26_rev   = est.get("fy26_rev")
    fy27_rev   = est.get("fy27_rev")
    fy26_rev_g = est.get("fy26_rev_g")
    fy27_rev_g = est.get("fy27_rev_g")

    sw_fy28_rev = est.get("sw_fy28_rev")
    sw_fy28_ni  = est.get("sw_fy28_ni")
    # FY28 EPS = SimpyWall 순이익 / 주식수 (실제 컨센서스)
    sw_fy28_eps = (sw_fy28_ni / shares) if (sw_fy28_ni and shares) else None

    # SimpyWall 실제 데이터만 사용. 없으면 N/A (가정 외삽 안 함)
    fy28_eps = sw_fy28_eps
    fy28_rev = sw_fy28_rev

    fy28_rev_g = ((fy28_rev - fy27_rev) / fy27_rev) if (fy28_rev and fy27_rev and fy27_rev > 0) else None
    fy28_eps_g = ((fy28_eps - fy27_eps) / abs(fy27_eps)) if (fy28_eps and fy27_eps and fy27_eps != 0) else None
    accel = (fy28_eps_g - fy27_eps_g) if (fy28_eps_g is not None and fy27_eps_g is not None) else None

    return {
        "price":  price,
        "target": fy28_eps * 20 if fy28_eps and fy28_eps > 0 else None,
        "fy26_rev_g": fy26_rev_g, "fy27_rev_g": fy27_rev_g, "fy28_rev_g": fy28_rev_g,
        "fy26_m":  impl_margin(fy26_eps, fy26_rev, shares),
        "fy27_m":  impl_margin(fy27_eps, fy27_rev, shares),
        "fy28_m":  impl_margin(fy28_eps, fy28_rev, shares),
        "fy26_per": price / fy26_eps if price and fy26_eps and fy26_eps > 0 else None,
        "fy27_per": price / fy27_eps if price and fy27_eps and fy27_eps > 0 else None,
        "fy28_per": price / fy28_eps if price and fy28_eps and fy28_eps > 0 else None,
        "accel":   accel,
    }

def compute_loss(est, price):
    shares     = est.get("shares", 0)
    ev         = est.get("ev", 0)
    fy26_eps   = est.get("fy26_eps")
    fy27_eps   = est.get("fy27_eps")
    fy26_rev   = est.get("fy26_rev")
    fy27_rev   = est.get("fy27_rev")
    fy26_rev_g = est.get("fy26_rev_g")
    fy27_rev_g = est.get("fy27_rev_g")

    sw_fy28_rev = est.get("sw_fy28_rev")
    sw_fy28_ni  = est.get("sw_fy28_ni")
    sw_fy28_eps = (sw_fy28_ni / shares) if (sw_fy28_ni and shares) else None

    fy28_eps = sw_fy28_eps
    fy28_rev = sw_fy28_rev

    fy28_rev_g = ((fy28_rev - fy27_rev) / fy27_rev) if (fy28_rev and fy27_rev and fy27_rev > 0) else None
    accel = (fy28_rev_g - fy27_rev_g) if (fy28_rev_g is not None and fy27_rev_g is not None) else None

    return {
        "price":  price,
        "ev_b":   ev / 1e9 if ev else 0,
        "target": est.get("target"),
        "fy26_evs": ev / fy26_rev if ev and fy26_rev and fy26_rev > 0 else None,
        "fy27_evs": ev / fy27_rev if ev and fy27_rev and fy27_rev > 0 else None,
        "fy28_evs": ev / fy28_rev if ev and fy28_rev and fy28_rev > 0 else None,
        "fy26_m":  impl_margin(fy26_eps, fy26_rev, shares),
        "fy27_m":  impl_margin(fy27_eps, fy27_rev, shares),
        "fy28_m":  impl_margin(fy28_eps, fy28_rev, shares),
        "fy26_rev_g": fy26_rev_g, "fy27_rev_g": fy27_rev_g, "fy28_rev_g": fy28_rev_g,
        "accel":   accel,
    }


# ─── 출력 ─────────────────────────────────────────
def print_profit_sectors(sectors, estimates, prices):
    W = 130
    print("=" * W)
    print("  [흑자기업]  Revenue Growth (YoY)    |  Margin (Implied Net)       |  P/E (Forward)              | EPS 가속도")
    print("=" * W)
    print(
        f"{'':>6} | {'현재가 (목표)':>16} |"
        f" {'FY26':>7} {'FY27':>7} {'FY28':>7}"
        f" | {'FY26':>7} {'FY27':>7} {'FY28':>7}"
        f" | {'FY26':>7} {'FY27':>7} {'FY28':>7}"
        f" | {'Δg(27→28)':>9}"
    )
    print("-" * W)
    for sector, tickers in sectors.items():
        print(f"  << {sector} >>")
        for t in tickers:
            try:
                d = compute_profit(estimates.get(t, {}), prices.get(t))
                tgt = f"({d['target']:>.0f})" if d["target"] else "(N/A)"
                print(
                    f"  {t:>5} | ${d['price']:>7.2f} {tgt:>7} |"
                    f" {fmt_pct(d['fy26_rev_g'])} {fmt_pct(d['fy27_rev_g'])} {fmt_pct(d['fy28_rev_g'])}"
                    f" | {fmt_margin(d['fy26_m'])} {fmt_margin(d['fy27_m'])} {fmt_margin(d['fy28_m'])}"
                    f" | {fmt_per(d['fy26_per'])} {fmt_per(d['fy27_per'])} {fmt_per(d['fy28_per'])}"
                    f" | {fmt_accel(d['accel'])}"
                )
            except Exception as e:
                print(f"  {t:>5} | error: {e}")
        print()
    print("=" * W)


def print_loss_sectors(sectors, estimates, prices):
    W = 135
    print()
    print("=" * W)
    print("  [적자기업]  EV/Sales                |  Forward Margin              |  Revenue Growth             | Rev 가속도")
    print("=" * W)
    print(
        f"{'':>6} | {'현재가 (목표)':>16} | {'EV':>6}"
        f" | {'FY26':>7} {'FY27':>7} {'FY28':>7}"
        f" | {'FY26':>8} {'FY27':>8} {'FY28':>8}"
        f" | {'FY26':>7} {'FY27':>7} {'FY28':>7}"
        f" | {'Δg(27→28)':>9}"
    )
    print("-" * W)
    for sector, tickers in sectors.items():
        print(f"  << {sector} >>")
        for t in tickers:
            try:
                d = compute_loss(estimates.get(t, {}), prices.get(t))

                def fm2(v):
                    return f"{v*100:>+7.1f}%" if v is not None else "     N/A"

                tgt = f"({d['target']:>.0f})" if d["target"] else "(N/A)"
                print(
                    f"  {t:>5} | ${d['price']:>7.2f} {tgt:>7} | {d['ev_b']:>5.1f}B"
                    f" | {fmt_evs(d['fy26_evs'])} {fmt_evs(d['fy27_evs'])} {fmt_evs(d['fy28_evs'])}"
                    f" | {fm2(d['fy26_m'])} {fm2(d['fy27_m'])} {fm2(d['fy28_m'])}"
                    f" | {fmt_pct(d['fy26_rev_g'])} {fmt_pct(d['fy27_rev_g'])} {fmt_pct(d['fy28_rev_g'])}"
                    f" | {fmt_accel(d['accel'])}"
                )
            except Exception as e:
                print(f"  {t:>5} | error: {e}")
        print()
    print("=" * W)


def print_news(all_sectors):
    W = 100
    print()
    print("=" * W)
    print("  [뉴스]  yfinance 최신 헤드라인 (한글 번역)")
    print("=" * W)
    all_tickers = [t for tickers in all_sectors.values() for t in tickers]
    has_news = False
    for t in all_tickers:
        news = get_news(t)
        if news:
            has_news = True
            for n in news:
                print(f"  {t:>5} | {translate_ko(n)}")
    if not has_news:
        print("  (뉴스 없음)")
    print("=" * W)


# ─── 메인 ─────────────────────────────────────────
if __name__ == "__main__":
    import io, contextlib

    force_refresh = "--refresh" in sys.argv

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    date_str = datetime.now().strftime("%Y-%m-%d")

    print()
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  성장기반 리서치 센터 (GBRC) - Daily Briefing │")
    print(f"  │  {now}                              │")
    print(f"  └─────────────────────────────────────────────┘")
    print()

    all_tickers = PROFIT_TICKERS + LOSS_TICKERS

    estimates = None if force_refresh else load_estimates_cache()
    if estimates is None:
        estimates = refresh_estimates(all_tickers)
    else:
        print(f"  [컨센서스 캐시 사용 — {current_week()}]\n")

    print("  [현재가 수집 중...]")
    prices = get_live_prices(all_tickers)
    print()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_profit_sectors(PROFIT_SECTORS, estimates, prices)
        print_loss_sectors(LOSS_SECTORS, estimates, prices)
        print_news({**PROFIT_SECTORS, **LOSS_SECTORS})
        print()
        print("  * FY26/FY27 = yfinance 애널리스트 컨센서스")
        print("  * FY28 = Simply Wall St 애널리스트 컨센서스 (없으면 N/A)")
        print("  * Margin = Implied Net Margin (EPS x Shares / Revenue)")
        print("  * 흑자 목표가 = FY28 EPS x 20 | 적자 목표가 = 애널리스트 컨센서스")
        print()

    output = buf.getvalue()
    print(output)

    val_dir = os.path.join(BASE_DIR, "data", "valuation")
    os.makedirs(val_dir, exist_ok=True)
    val_path = os.path.join(val_dir, f"{date_str}.md")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(f"# GBRC 밸류에이션 브리핑 — {date_str}\n\n```\n{output}```\n")
    print(f"  밸류에이션 저장: {val_path}")
