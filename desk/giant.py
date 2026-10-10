"""거인의 어깨 탭: AI 유니버스 안에서 어느 섹터와 어느 회사에 돈이 몰리는지 보여 준다.

정본은 docs/desk_system.md §9다. 유니버스는 watchlist.AI_UNIVERSE이고 전고점은 ai_ath_gap과
같은 기준이다. 관찰용 화면이라 매수 추천·목표 비중·주문 수량을 만들지 않는다(§1).

- 기준 구간: SOXX 최근 1년 최대 하락(고점→저점)과 그 뒤 반등. 조정에서 덜 빠지고 먼저
  회복한 종목이 주도주라는 관찰을 숫자로 본다.
- 고래 신호: 두 펀드의 실매매(gbrc.etf.flows — 설정·환매·분할을 걷어낸 수량 변화), 국내
  외국인·기관 순매수(네이버 증권), 50거래일 상승일/하락일 거래량 비율. 미국은 투자 주체별
  매매를 공개하지 않아 공매도 잔고(FINRA, 월 2회)의 한 달 변화를 대신 보여 준다.
- 규모: 달러 시총과 1억 달러 거래당 가격 변동(Amihud 근사). 같은 돈이 작은 종목을 더 크게
  움직인다는 관찰(data/derived/analysis/recovery_factors/report.md)을 종목마다 본다.
- 가격·실적은 yfinance다. TradingView MCP 값은 넣지 않는다(AGENTS.md).
"""

from __future__ import annotations

import json
import threading
import time as _time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from statistics import median

import numpy as np
import pandas as pd

import ai_ath_gap
import watchlist

from . import config
from .state import clean

CACHE_PATH = config.CACHE / "giant.json"
MAX_AGE_MIN = 60
ANCHOR = "SOXX"
BENCHMARKS = ["SOXX", "QQQ"]
FUNDS = {"TIME": "456600", "KoAct": "2ETFQ1"}

# ── 화면 기준 (사용자 조정 영역) ──────────────────────────────────────
ANCHOR_LOOKBACK = 252   # 이 거래일 안의 최대 하락 구간을 기준 조정으로 삼는다
WINDOWS = {"r1w": 5, "r1m": 21, "r3m": 63}
UD_DAYS = 50            # 상승일·하락일 거래량 비율(매집 지표)을 보는 거래일 수
IMPACT_DAYS = 50        # 1억 달러당 가격 변동(중앙값)을 재는 최근 거래일 수
LEADERS_PER_SUB = 2     # ◆ 조건 2: 소분류 안 저점 이후 수익률 상위 몇 위까지
# 종목 근거 꼬리표. 숫자를 바꾸면 화면이 그대로 따른다(payload["rules"]).
# short_float: 유통 주식 대비 공매도 잔고(%). 7월 급락 코호트 미국 70종목에서 이 비율이 높을수록
# 전고점에서 멀었다(ρ -0.53, recovery_factors/report.md).
TAGS = {"rev_growth": 50.0, "low_pe": 10.0, "flow_pct": 0.5, "ud": 1.2, "short_float": 15.0}
# 시장 폭 = 유니버스 중 전고점 -10% 이내 종목 비율. 2026-02~09 주간 재현 검증에서 이 비율이
# 중앙값(19%) 이상인 날은 ◆·모멘텀·펀드 매수 같은 주도주 신호가 유니버스보다 나았고, 아래인
# 날은 뒤졌다(data/derived/analysis/giant_validation/validation.md).
BREADTH_CHECK = {"split": 19.0, "period": "2026-02~09", "dates": 29,
                 "wide_excess": 6.1, "wide_hit": 80, "narrow_excess": -3.4, "narrow_hit": 36}
# 지수가 막 바닥을 만들면 '저점 이후'가 모두 0에 가까워 ◆가 비거나 흔들린다. 미리 정한 보완안 4개를
# 비교한 결과(variants.py), 규칙은 그대로 두고 ◆가 빌 때 AI 전체를 그대로 드는 '보류'가 가장 단순하고
# 근거가 분명했다: 7/27 바닥에서 AI 전체 +16.7%, 현금 0%, 최근 1개월 강자 대체 +4.3%(20거래일).
# 회복 등급: 기준 하락 직전에 전고점 -15% 안이었고 그 하락에서 15% 넘게 빠진 종목만 회복률(빠진 폭 중
# 되찾은 %)을 잰다. 섹터 등급은 그 종목이 3개 이상일 때 중앙값으로 매긴다(사용자 요청 2026-10-06,
# 근거 data/derived/analysis/sector_grades/report.md). 하한 %: S 90, A 60, B 40, C 20, 그 아래 D.
COHORT_NEAR, COHORT_CRASH = -15.0, -15.0
GRADE_LEAD, GRADE_LAG = 35, 17  # 종목 자신의 고점은 기준 고점 35거래일 전부터, 저점은 기준 저점 17거래일 뒤까지
GRADES = (("S", 90.0), ("A", 60.0), ("B", 40.0), ("C", 20.0))
GRADE_MIN = 3
FRESH_LOW_DAYS = 10     # 기준 저점이 이 거래일 안이면 '저점 이후' 순위가 아직 짧다고 알린다
HOLD_CHECK = {"date": "2026-07-27", "universe": 16.7, "substitute": 4.3,
              "compound_hold": 80.5, "compound_cash": 54.7, "compound_universe": 67.2}
# ────────────────────────────────────────────────────────────────

NAVER_TREND = "https://m.stock.naver.com/api/stock/{code}/trend?pageSize=60"
# 통화 단위 / 1달러. EUR만 EURUSD=X(1유로의 달러 값)를 뒤집는다. 시장 통화는 yfinance info가
# 없을 때만 쓴다.
FX_TICKERS = {"KRW": "KRW=X", "JPY": "JPY=X", "TWD": "TWD=X", "CNY": "CNY=X", "HKD": "HKD=X",
              "EUR": "EURUSD=X"}
MARKET_CURRENCY = {"미국": "USD", "한국": "KRW", "일본": "JPY", "대만": "TWD", "중국": "CNY",
                   "홍콩": "HKD", "오스트리아": "EUR", "독일": "EUR", "프랑스": "EUR"}
FUND_ALIASES = watchlist.AI_UNIVERSE_ALIASES
_FUND_EXCHANGES = {"US": ("",), "JP": (".T",), "HK": (".HK",), "TT": (".TW", ".TWO"), "AV": (".VI",)}
_LOCK = threading.Lock()


def pct(ratio) -> float | None:
    v = float(ratio)
    return round((v - 1.0) * 100.0, 2) if np.isfinite(v) else None


def on_or_before(series: pd.Series, day: str) -> float | None:
    s = series[series.index <= pd.Timestamp(day)]
    return float(s.iloc[-1]) if len(s) else None


def find_anchor(close: pd.Series, lookback: int = ANCHOR_LOOKBACK) -> dict:
    """최근 lookback 거래일에서 가장 깊었던 하락의 고점·저점. 새 고점이 나와도 유지된다."""
    recent = close.iloc[-lookback:]
    drawdown = recent / recent.cummax() - 1.0
    low_at = drawdown.idxmin()
    peak_at = recent[recent.index <= low_at].idxmax()
    last = float(recent.iloc[-1])
    peak, low = float(recent[peak_at]), float(recent[low_at])
    i_peak, i_low = recent.index.get_loc(peak_at), recent.index.get_loc(low_at)
    return {
        "index": ANCHOR,
        "peak_date": peak_at.date().isoformat(), "peak": peak,
        "low_date": low_at.date().isoformat(), "low": low,
        # 회복 등급에서 종목 자신의 고점·저점을 찾는 창
        "grade_from": recent.index[max(0, i_peak - GRADE_LEAD)].date().isoformat(),
        "grade_to": recent.index[min(len(recent) - 1, i_low + GRADE_LAG)].date().isoformat(),
        "last_date": recent.index[-1].date().isoformat(), "last": last,
        "crash": pct(low / peak), "since_low": pct(last / low), "vs_peak": pct(last / peak),
        "age": len(recent) - 1 - recent.index.get_loc(low_at),  # 저점 뒤 지난 거래일
    }


def ud_ratio(bars: pd.DataFrame, days: int = UD_DAYS) -> float | None:
    """상승일 거래량 합 / 하락일 거래량 합. 1보다 크면 오르는 날에 더 많이 거래됐다."""
    tail = bars.iloc[-(days + 1):]
    change = tail["Close"].diff().iloc[1:]
    volume = tail["Volume"].iloc[1:]
    up, down = float(volume[change > 0].sum()), float(volume[change < 0].sum())
    return round(up / down, 2) if down > 0 else None


def own_recovery(bars: pd.DataFrame, anchor: dict) -> float | None:
    """종목 자신의 급락 고점·저점으로 잰 회복률(빠진 폭 중 되찾은 %). 급락을 맞지 않았으면 None.

    고점은 기준 고점 GRADE_LEAD거래일 전부터 기준 저점까지의 최고 종가, 저점은 그 뒤 기준 저점
    GRADE_LAG거래일 뒤까지의 최저 종가다. 고점이 그때까지의 전고점 -15% 안이고 저점까지 15% 넘게
    빠진 종목만 잰다(data/derived/analysis/recovery_factors와 같은 정의).
    """
    close = bars["Close"]
    win = close[(close.index >= pd.Timestamp(anchor["grade_from"])) & (close.index <= pd.Timestamp(anchor["low_date"]))]
    if win.empty:
        return None
    peak_at, peak = win.idxmax(), float(win.max())
    highs = bars["High"][(bars.index >= "2010-01-01") & (bars.index <= peak_at)]
    after = close[(close.index >= peak_at) & (close.index <= pd.Timestamp(anchor["grade_to"]))]
    low = float(after.min())
    if pct(peak / float(highs.max())) < COHORT_NEAR or pct(low / peak) > COHORT_CRASH:
        return None
    return round((float(close.iloc[-1]) - low) / (peak - low) * 100.0, 1)


def stock_metrics(bars: pd.DataFrame, anchor: dict) -> dict:
    close = bars["Close"]
    last = float(close.iloc[-1])
    peak_c = on_or_before(close, anchor["peak_date"])
    low_c = on_or_before(close, anchor["low_date"])
    out = {
        "date": close.index[-1].date().isoformat(),
        "close": last,
        "crash": pct(low_c / peak_c) if peak_c and low_c else None,
        "since_low": pct(last / low_c) if low_c else None,
        # 급락 전 고점일 대비 순수익률. 많이 빠져서 많이 튄 효과(베타)를 걷어 낸 회복도다.
        "cycle": pct(last / peak_c) if peak_c else None,
        "above50": bool(last > float(close.iloc[-50:].mean())) if len(close) >= 50 else None,
        "ud": ud_ratio(bars),
    }
    for key, n in WINDOWS.items():
        out[key] = pct(last / float(close.iloc[-1 - n])) if len(close) > n else None
    out["recovery"] = own_recovery(bars, anchor)
    out["grade"] = grade_of(out["recovery"])
    ath = ai_ath_gap.measure(bars)
    out.update(gap_pct=round(ath["gap_pct"], 2), peak_date=ath["peak_date"].isoformat(),
               bucket=ai_ath_gap.band_of(ath["gap_pct"], ath["days_since_peak"]))
    return out


def price_impact(bars: pd.DataFrame, per_usd: float, days: int = IMPACT_DAYS) -> float | None:
    """1억 달러 거래당 하루 가격 변동(%) — Amihud 비유동성의 근사.

    최근 days거래일의 |일간 수익률| ÷ 달러 거래대금을 재고 중앙값을 쓴다. 같은 돈이 오를 때도
    내릴 때도 작은 종목을 더 크게 움직이는지 비교하는 상대 지표다. 실제로 1억 달러를 사면 작은
    종목은 이보다 크게 움직이거나 그만큼 살 수 없다.
    """
    tail = bars.iloc[-(days + 1):]
    moves = tail["Close"].pct_change().abs().iloc[1:]
    dollars = (tail["Close"] * tail["Volume"]).iloc[1:] / per_usd
    ratio = (moves / dollars.where(dollars > 0)).dropna()
    if len(ratio) < days // 2:
        return None
    return round(float(ratio.median()) * 1e8 * 100.0, 4)


def short_interest(info: dict) -> dict | None:
    """미국 공매도 잔고(FINRA 집계, 월 2회·약 2주 늦게 공개)의 한 달 변화.

    cover_pct는 줄어든 공매도 주식 수(= 되사들인 환매 매수)를 발행 주식 대비 %로 나타낸다. 국내
    순매수의 시총 대비 %와 같은 눈금이고, 마이너스는 새 공매도(매도)다.
    """
    now, prev = info.get("sharesShort"), info.get("sharesShortPriorMonth")
    shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    if not (now and prev and shares):
        return None

    def day(key):
        ts = info.get(key)
        return datetime.fromtimestamp(ts, timezone.utc).date().isoformat() if isinstance(ts, (int, float)) else None

    flt = info.get("shortPercentOfFloat")
    return {
        "float_pct": round(flt * 100.0, 1) if isinstance(flt, (int, float)) and np.isfinite(flt) else None,
        "cover_pct": round((prev - now) / shares * 100.0, 2),
        "chg_pct": round((now / prev - 1.0) * 100.0, 1),
        "from": day("sharesShortPreviousMonthDate"), "to": day("dateShortInterest"),
    }


def fx_per_usd(currencies) -> dict[str, float]:
    """통화 단위 / 1달러(Yahoo 최근 종가). 받지 못한 통화는 빠지고, 그 종목의 달러 값은 비운다."""
    import yfinance as yf

    out = {"USD": 1.0}
    for cur in sorted({c for c in currencies if c in FX_TICKERS}):
        try:
            last = float(yf.Ticker(FX_TICKERS[cur]).history(period="5d")["Close"].dropna().iloc[-1])
        except Exception:
            continue
        out[cur] = 1.0 / last if cur == "EUR" else last
    return out


def next_fy_pe(price, eps) -> float | None:
    """가격 ÷ 다음 회계연도 컨센서스 EPS. 둘 중 하나라도 없거나 이익이 0 이하면 비운다."""
    ok = lambda v: isinstance(v, (int, float)) and np.isfinite(v) and v > 0
    return round(price / eps, 1) if ok(price) and ok(eps) else None


def fundamentals(symbol: str) -> dict:
    """최근 분기 매출 증가율(전년 대비)·영업이익률·선행 PER·시총·통화·공매도 잔고.
    없으면 비운다(가정으로 채우지 않음)."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    try:
        info = ticker.info or {}
    except Exception:
        return {}

    def ratio(key):
        v = info.get(key)
        return round(v * 100.0, 1) if isinstance(v, (int, float)) and np.isfinite(v) else None

    pe = info.get("forwardPE")
    pe = round(pe, 1) if isinstance(pe, (int, float)) and np.isfinite(pe) and pe > 0 else None
    if symbol.endswith(".T"):
        # 도쿄 상장의 Yahoo forwardEps는 컨센서스와 맞지 않는다(2026-10-06 어드반테스트 237.7엔 대
        # 다음 회계연도 1,202.6엔 → PER 169.5배 대 33.5배). 다음 회계연도 평균 EPS로 다시 계산한다.
        try:
            eps = float(ticker.earnings_estimate.loc["+1y", "avg"])
        except Exception:
            eps = None
        pe = next_fy_pe(info.get("currentPrice") or info.get("regularMarketPrice"), eps)
    return {
        "rev_growth": ratio("revenueGrowth"),
        "op_margin": ratio("operatingMargins"),
        "fwd_pe": pe,
        "mcap": info.get("marketCap"),
        "currency": info.get("currency"),
        "short": short_interest(info),
    }


def _num(text) -> float:
    s = str(text or "").replace(",", "").replace("%", "").replace("+", "").strip()
    return float(s) if s else 0.0


def kr_flow_from_rows(rows: list[dict], low_date: str, mcap: float | None) -> dict | None:
    """네이버 일별 투자자 동향으로 저점 다음 날부터의 개인·외국인·기관 순매수(시총 대비 %).

    세 주체의 합이 0이 아니면 나머지(기타법인·자사주·ADR 원주 전환 등)가 반대편에 섰다는 뜻이라
    `other_pct`로 남긴다. 예: 2026-07 미국 ADR을 상장한 SK하이닉스는 세 주체가 모두 팔았다.
    """
    if not rows or not mcap:
        return None
    rows = sorted(rows, key=lambda r: r["bizdate"])
    low = low_date.replace("-", "")
    after = [r for r in rows if r["bizdate"] > low]
    before = [r for r in rows if r["bizdate"] <= low]
    if not after:
        return None
    value = lambda key: sum(_num(r.get(key)) * _num(r.get("closePrice")) for r in after) / mcap * 100.0
    foreign, inst, indiv = value("foreignerPureBuyQuant"), value("organPureBuyQuant"), value("individualPureBuyQuant")
    hold_pp = (_num(after[-1].get("foreignerHoldRatio")) - _num(before[-1].get("foreignerHoldRatio"))
               if before else None)
    return {
        "foreign_pct": round(foreign, 2),
        "inst_pct": round(inst, 2),
        "indiv_pct": round(indiv, 2),
        "other_pct": round(-(foreign + inst + indiv), 2),
        "foreign_hold_pp": round(hold_pp, 2) if hold_pp is not None else None,
        "from": after[0]["bizdate"], "to": after[-1]["bizdate"],
        "complete": bool(before),  # 받은 기간이 저점 이전까지 닿았는가
    }


def kr_flow(symbol: str, low_date: str, mcap: float | None) -> dict | None:
    import requests

    try:
        r = requests.get(NAVER_TREND.format(code=symbol.split(".")[0]), timeout=15,
                         headers={"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"})
        return kr_flow_from_rows(r.json(), low_date, mcap)
    except Exception:
        return None


def universe_symbol(key: str, universe: set[str]) -> str | None:
    """펀드 PDF 코드(gbrc holding_key)를 유니버스 심볼로 바꾼다. 유니버스 밖이면 None."""
    parts = str(key).split()
    code = parts[0]
    if code in FUND_ALIASES:
        alias = FUND_ALIASES[code]
        return alias if alias in universe else None
    exch = parts[1] if len(parts) > 2 else ""
    if exch == "HK":
        candidates = [code.zfill(4) + ".HK"]
    elif exch == "CH":
        candidates = [code + (".SS" if code.startswith("6") else ".SZ")]
    elif exch in _FUND_EXCHANGES:
        candidates = [code + suffix for suffix in _FUND_EXCHANGES[exch]]
    elif code.isdigit() and len(code) == 6:
        candidates = [code + ".KS", code + ".KQ"]
    else:
        candidates = [code]
    return next((c for c in candidates if c in universe), None)


def fund_flows(low_date: str, universe: set[str], through: str | None = None) -> tuple[dict, dict]:
    """두 펀드의 실매매를 '저점 이후'와 '최근 20거래일'로 잰다. PDF D는 D-1 바스켓이다."""
    from gbrc.etf.flows import flows_between
    from gbrc.etf.trades import load_fund_snapshots

    through = through or date.today().isoformat()
    out: dict[str, dict] = {}
    meta: dict[str, list[str]] = {}
    for fund, fund_id in FUNDS.items():
        snaps = load_fund_snapshots(fund_id, through)
        dates = sorted(snaps)
        if len(dates) < 2:
            continue
        windows = {"since_low": next((d for d in dates if d > low_date), None),
                   "d20": dates[-21] if len(dates) > 21 else dates[0]}
        for label, base in windows.items():
            result = flows_between(snaps, base, dates[-1])
            if result is None:
                continue
            meta[f"{fund}_{label}"] = [result.base_date, result.current_date]
            for key, flow in result.flows.items():
                sym = universe_symbol(key, universe)
                if sym is None:
                    continue
                slot = out.setdefault(sym, {}).setdefault(fund, {})
                slot[label] = {"kind": flow.kind,
                               "pct": round(flow.selective_pct, 1) if flow.selective_pct is not None else None,
                               "pp": round(flow.trade_pp, 2) if flow.trade_pp is not None else None}
                slot["weight"] = round(flow.weight, 2)
    return out, meta


def _median(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(median(vals), 2) if vals else None


BUY_KINDS, SELL_KINDS = {"매수", "신규"}, {"매도", "편출"}


def _fund_counts(members: list[dict], window: str) -> tuple[int, int]:
    """(펀드×종목) 실매매 건수. 두 펀드가 같은 종목을 사면 2건이다."""
    kinds = [f[window]["kind"] for m in members for f in m.get("funds", {}).values() if window in f]
    return sum(k in BUY_KINDS for k in kinds), sum(k in SELL_KINDS for k in kinds)


def _cap_weighted(members: list[dict], flow: str, key: str, cap: str) -> float | None:
    """종목별 시총 대비 수급(%)을 시총 가중으로 합친다(작은 종목 한 개가 섹터 값을 흔들지 않게).
    국내는 원화 시총(mcap), 미국 공매도는 달러 시총(mcap_usd)으로 가중한다."""
    pairs = [(m[flow][key], m[cap]) for m in members
             if m.get(flow) and m.get(cap) and m[flow].get(key) is not None]
    total = sum(c for _, c in pairs)
    return round(sum(v * c for v, c in pairs) / total, 2) if total else None


def grade_of(recovery: float | None) -> str | None:
    if recovery is None:
        return None
    return next((g for g, floor in GRADES if recovery >= floor), "D")


def recovery_grade(members: list[dict]) -> dict:
    """급락을 맞은 종목(recovery가 있는 종목)의 회복률 중앙값과 등급. GRADE_MIN개 미만이면 등급을 비운다."""
    vals = [m["recovery"] for m in members if m.get("recovery") is not None]
    rec = round(median(vals), 1) if len(vals) >= GRADE_MIN else None
    return {"recovery": rec, "recovery_n": len(vals), "grade": grade_of(rec)}


def summarize(members: list[dict], uni: dict) -> dict:
    n = len(members)
    near = sum(1 for m in members if m["gap_pct"] >= ai_ath_gap.NEAR_PCT)
    above = [m["above50"] for m in members if m.get("above50") is not None]
    since = _median(m.get("since_low") for m in members)
    leader = max(members, key=lambda m: m["since_low"] if m.get("since_low") is not None else -1e9)
    buy_low, sell_low = _fund_counts(members, "since_low")
    buy_20, sell_20 = _fund_counts(members, "d20")
    return {
        "n": n,
        "since_low": since,
        "crash": _median(m.get("crash") for m in members),
        "cycle": _median(m.get("cycle") for m in members),
        "r1m": _median(m.get("r1m") for m in members),
        "r3m": _median(m.get("r3m") for m in members),
        "rev_growth": _median(m.get("rev_growth") for m in members),
        "near_high_pct": round(near / n * 100) if n else None,
        "above50_pct": round(sum(above) / len(above) * 100) if above else None,
        "rs": round(since - uni["since_low"], 2) if since is not None and uni.get("since_low") is not None else None,
        "leader": leader["symbol"],
        "fund_buy": buy_low, "fund_sell": sell_low, "fund_buy_20": buy_20, "fund_sell_20": sell_20,
        "kr_n": sum(1 for m in members if m.get("kr_flow")),
        "kr_foreign_pct": _cap_weighted(members, "kr_flow", "foreign_pct", "mcap"),
        "kr_inst_pct": _cap_weighted(members, "kr_flow", "inst_pct", "mcap"),
        "kr_indiv_pct": _cap_weighted(members, "kr_flow", "indiv_pct", "mcap"),
        "us_n": sum(1 for m in members if m.get("short")),
        "us_short_cover": _cap_weighted(members, "short", "cover_pct", "mcap_usd"),
        **recovery_grade(members),
    }


def rank_and_check(stocks: dict, uni: dict) -> list[dict]:
    """섹터·소분류 요약을 만들고 종목마다 ◆ 네 조건을 붙인다."""
    groups, sub_since = [], {}
    for group, subs in watchlist.AI_UNIVERSE.items():
        g_members, sub_rows = [], []
        for sub, syms in subs.items():
            members = [stocks[s] for s in syms if s in stocks]
            if not members:
                continue
            ordered = sorted(members, key=lambda m: m["since_low"] if m.get("since_low") is not None else -1e9,
                             reverse=True)
            for i, m in enumerate(ordered, 1):
                m["rank_in_sub"], m["sub_size"] = i, len(ordered)
            row = summarize(members, uni)
            row.update(sub=sub, members=[m["symbol"] for m in ordered])
            sub_rows.append(row)
            sub_since[(group, sub)] = row["since_low"]
            g_members += members
        if g_members:
            row = summarize(g_members, uni)
            row.update(group=group, subs=sub_rows)
            groups.append(row)
    for m in stocks.values():
        sector = sub_since.get((m["group"], m["sub"]))
        m["checks"] = [
            sector is not None and uni.get("since_low") is not None and sector > uni["since_low"],
            m["rank_in_sub"] <= LEADERS_PER_SUB,
            m["gap_pct"] >= ai_ath_gap.NEAR_PCT,
            (m.get("ud") or 0) >= 1.0,
        ]
        m["star"] = all(m["checks"])
    groups.sort(key=lambda g: g["since_low"] if g["since_low"] is not None else -1e9, reverse=True)
    return groups


def add_comparison_views(data: dict, tour_map: dict | None = None) -> dict:
    """기존 가격으로 탐방 분야·종목 등급을 덧붙인다. ◆ 순위·조건에는 영향을 주지 않는다.

    캐시를 읽을 때도 지도 역할을 다시 연결해 탐방 갱신과 기존 캐시에 즉시 대응한다.
    분야끼리 소속이 겹칠 수 있지만 각 분야와 전체 종목 목록에서는 회사당 한 번만 센다.
    """
    if "stocks" not in data:
        return data
    from . import tour

    if tour_map is None:
        tour_map = tour.load_map()
    stocks, uni = data["stocks"], data["universe"]
    for stock in stocks.values():
        stock["grade"] = grade_of(stock.get("recovery"))
        stock["tour_fields"] = []
    rows = []
    for field in watchlist.ai_tour_sectors(tour_map, tour.universe_symbol):
        members = [stocks[s] for s in field["symbols"] if s in stocks]
        if not members:
            continue
        row = summarize(members, uni)
        row.update(key=field["key"], group=field["group"], sub=field["sub"],
                   members=[m["symbol"] for m in members])
        rows.append(row)
        for stock in members:
            stock["tour_fields"].append({"key": field["key"], "label": field["sub"], "parent": field["group"]})
    data["tour_sectors"] = rows
    data["tour_as_of"] = tour_map["as_of"]
    data["tour_unplaced"] = [s for s, m in stocks.items() if not m["tour_fields"]]
    return data


def breadth_series(bars: dict, symbols: list[str], calendar: pd.DatetimeIndex) -> pd.Series:
    """날짜마다 유니버스 중 전고점 -10% 이내 종목 비율(%). 상장 60거래일 미만 종목은 뺀다.

    전고점은 2010년 이후 고가의 누적 최대로, 전고점 연구의 2010년 대체 규칙과 같다.
    아시아 종목은 그 미국 날짜 또는 직전 현지 거래일 값을 쓴다.
    """
    gaps, counts = {}, {}
    for s in symbols:
        b = bars.get(s)
        if b is None or len(b) < 60:
            continue
        b = b[b.index >= "2010-01-01"]
        gaps[s] = (b["Close"] / b["High"].cummax() - 1).reindex(calendar, method="ffill")
        counts[s] = pd.Series(np.arange(1, len(b) + 1), index=b.index).reindex(calendar, method="ffill")
    gap, n = pd.DataFrame(gaps), pd.DataFrame(counts)
    gap = gap.where(n >= 60)
    return (gap >= ai_ath_gap.NEAR_PCT / 100).sum(axis=1) / gap.notna().sum(axis=1) * 100


def compute() -> dict:
    rows = ai_ath_gap.universe_rows()
    symbols = [s for _, _, s in rows]
    universe = set(symbols)
    warnings: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        bars = dict(pool.map(lambda s: ai_ath_gap.fetch(s, None), symbols + BENCHMARKS))
        funda = dict(zip(symbols, pool.map(fundamentals, symbols)))
    if bars.get(ANCHOR) is None:
        raise RuntimeError(f"{ANCHOR} 가격을 받지 못해 기준 구간을 정할 수 없습니다")
    anchor = find_anchor(bars[ANCHOR]["Close"])
    currency = {s: funda[s].get("currency") or MARKET_CURRENCY.get(ai_ath_gap.market_of(s)) for s in symbols}
    fx = fx_per_usd(currency.values())
    no_fx = sorted({str(c) for c in currency.values() if c not in fx})
    if no_fx:
        warnings.append(f"환율을 받지 못해 달러 환산을 비운 통화: {', '.join(no_fx)}")

    kr = [s for s in symbols if s.endswith((".KS", ".KQ"))]
    with ThreadPoolExecutor(max_workers=8) as pool:
        flows = dict(zip(kr, pool.map(lambda s: kr_flow(s, anchor["low_date"], funda[s].get("mcap")), kr)))
    try:
        funds, fund_meta = fund_flows(anchor["low_date"], universe)
    except Exception as exc:  # 펀드 PDF가 없어도 나머지 화면은 보여 준다
        funds, fund_meta = {}, {}
        warnings.append(f"펀드 실매매를 읽지 못했습니다: {type(exc).__name__}")

    stocks = {}
    for group, sub, sym in rows:
        b = bars.get(sym)
        if b is None or len(b) < 2:
            warnings.append(f"{sym} 가격 없음")
            continue
        m = stock_metrics(b, anchor)
        f = funda.get(sym, {})
        per_usd = fx.get(currency[sym])
        m.update(symbol=sym, name=ai_ath_gap.NAMES.get(sym, sym), group=group, sub=sub,
                 market=ai_ath_gap.market_of(sym), rev_growth=f.get("rev_growth"),
                 op_margin=f.get("op_margin"), fwd_pe=f.get("fwd_pe"), mcap=f.get("mcap"),
                 mcap_usd=round(f["mcap"] / per_usd) if f.get("mcap") and per_usd else None,
                 impact=price_impact(b, per_usd) if per_usd else None,
                 funds=funds.get(sym, {}), kr_flow=flows.get(sym), short=f.get("short"))
        stocks[sym] = m
    missing_flow = [s for s in kr if s in stocks and stocks[s]["kr_flow"] is None]
    if missing_flow:
        warnings.append(f"국내 수급을 받지 못한 종목 {len(missing_flow)}개")
    # 공매도 잔고는 FINRA 집계일이 종목마다 같아야 한다. 가장 흔한 기간을 화면 제목에 쓴다.
    windows = Counter((m["short"]["from"], m["short"]["to"]) for m in stocks.values() if m.get("short"))
    short_window = list(windows.most_common(1)[0][0]) if windows else None

    uni = {k: _median(m.get(k) for m in stocks.values())
           for k in ("since_low", "crash", "cycle", "r1m", "r3m", "rev_growth")}
    uni.update(recovery_grade(list(stocks.values())))
    groups = rank_and_check(stocks, uni)
    bench = {}
    for b in BENCHMARKS:
        if bars.get(b) is not None:
            bm = stock_metrics(bars[b], anchor)
            bench[b] = {k: bm[k] for k in ("since_low", "crash", "cycle", "r1m", "r3m", "gap_pct")}
    cal = bars[ANCHOR].index[-ANCHOR_LOOKBACK:]
    wide = breadth_series(bars, symbols, cal).dropna()
    breadth = {"now": round(float(wide.iloc[-1]), 1), "median_1y": round(float(wide.median()), 1),
               "history": [[d.date().isoformat(), round(float(v), 1)] for d, v in wide.iloc[::5].items()],
               "check": BREADTH_CHECK}
    return {
        "computed_at": datetime.now(config.KST).isoformat(timespec="seconds"),
        "anchor": anchor,
        "universe": {"n": len(stocks), **uni},
        "bench": bench,
        "breadth": breadth,
        "groups": groups,
        "stocks": stocks,
        "fund_windows": fund_meta,
        "short_window": short_window,
        "rules": {"near_high_pct": ai_ath_gap.NEAR_PCT, "leaders_per_sub": LEADERS_PER_SUB,
                  "ud_days": UD_DAYS, "impact_days": IMPACT_DAYS, "anchor_lookback": ANCHOR_LOOKBACK,
                  "tags": TAGS, "fresh_low_days": FRESH_LOW_DAYS, "hold_check": HOLD_CHECK,
                  "grades": [list(g) for g in GRADES], "grade_min": GRADE_MIN,
                  "cohort": {"near": COHORT_NEAR, "crash": COHORT_CRASH}},
        "warnings": warnings,
    }


def payload(force: bool = False) -> dict:
    """캐시가 MAX_AGE_MIN보다 새로우면 그대로 쓰고, 아니면 다시 계산한다(첫 계산 30~60초)."""
    with _LOCK:  # 탭 열기와 다시 계산이 겹쳐도 한 번만 계산한다
        if not force and CACHE_PATH.exists():
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if _time.time() - cached.get("_epoch", 0) < MAX_AGE_MIN * 60:
                return clean(add_comparison_views(cached))
        data = clean(add_comparison_views(compute()))
        data["_epoch"] = _time.time()
        config.CACHE.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data
