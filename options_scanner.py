"""
GBRC 옵션 스캐너 v3 - 적자기업 상대 강도 탐지

스코어링 5팩터 (각 20점, 총 100점):
  ① P/C Ratio      — 콜 우위 = 강세 심리
  ② 이상 콜 규모    — 달러 기준, 복권성 필터 적용
  ③ 상대 옵션 강도  — 옵션 거래량 / 주식 평균 거래량 (종목간 공정 비교)
  ④ IV Skew        — 콜 IV > 풋 IV = 구조적 상방 수요
  ⑤ Term Structure — 역전 = 단기 이벤트 기대

yfinance 한계:
  - 과거 옵션 데이터 없음 (일별 OI 추적 불가)
  - 블록/스윕 구분 불가
  - IV Rank/Percentile 불가
"""

import yfinance as yf
import warnings
import math
import json
import os
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

from watchlist import SECTOR_MAP, TICKER_TO_SECTOR, PROFIT_TICKERS, LOSS_TICKERS

try:  # ⑩ ETF 디버전스: TIME 검증 스냅샷(docs/options_system.md)
    from gbrc.config import load_fund_configs
    from gbrc.etf.market import us_ticker as get_us_ticker
    from gbrc.etf.storage import load_snapshot as load_fund_snapshot, snapshot_dates

    ETF_FUND_ID = load_fund_configs()["time"].snapshot_ticker

    def list_snapshot_dates():
        return snapshot_dates(ETF_FUND_ID)

    def load_snapshot(date_str):
        return load_fund_snapshot(ETF_FUND_ID, date_str)

    ETF_AVAILABLE = True
except Exception:
    ETF_AVAILABLE = False

TODAY = datetime.now().date()
SCORE_VERSION = "v3.5"
NEAR_END = TODAY + timedelta(days=7)
MID_END = TODAY + timedelta(days=30)
FAR_END = TODAY + timedelta(days=90)


def safe_div(a, b, default=0):
    return a / b if b and b != 0 else default


def cap_label(mcap):
    if mcap >= 10e9:
        return "대형"
    elif mcap >= 2e9:
        return "중형"
    return "소형"


def classify_strike(strike, price):
    if price == 0:
        return "?"
    pct = (strike - price) / price
    if abs(pct) <= 0.03:
        return "ATM"
    return "OTM" if pct > 0 else "ITM"


def calc_iv_skew(chain, price):
    """OTM 풋 IV - OTM 콜 IV. 음수 = 콜 수요 강함(강세).
    v3.4 검증: |skew|>1.0 = 데이터 이상 (IV 극단값) → None"""
    calls, puts = chain.calls, chain.puts
    if calls.empty or puts.empty or price == 0:
        return None
    otm_calls = calls[(calls["strike"] >= price * 1.05) & (calls["strike"] <= price * 1.20)]
    otm_puts = puts[(puts["strike"] >= price * 0.80) & (puts["strike"] <= price * 0.95)]
    if otm_calls.empty or otm_puts.empty:
        return None
    call_iv = otm_calls["impliedVolatility"].median()
    put_iv = otm_puts["impliedVolatility"].median()
    if call_iv and put_iv and not math.isnan(call_iv) and not math.isnan(put_iv):
        skew = put_iv - call_iv
        if abs(skew) > 1.0:        # 이상치 가드 (정상 범위 -0.3 ~ +0.3)
            return None
        return skew
    return None


def calc_term_structure(near_chain, far_chain, price):
    """장기 ATM IV / 단기 ATM IV. < 1.0 = 역전(단기 이벤트 기대).
    v3.4 검증: near_iv<0.05 일 때 비율 폭주 → None (AEHR 782 같은 케이스)"""
    def get_atm_iv(chain, price):
        calls = chain.calls
        if calls.empty or price == 0:
            return None
        atm = calls.iloc[(calls["strike"] - price).abs().argsort()[:3]]
        iv = atm["impliedVolatility"].median()
        return iv if iv and not math.isnan(iv) else None
    near_iv = get_atm_iv(near_chain, price)
    far_iv = get_atm_iv(far_chain, price)
    if near_iv and far_iv and near_iv > 0.05:   # near_iv 너무 작으면 이상치
        ratio = far_iv / near_iv
        if 0.1 <= ratio <= 5.0:                 # 정상 범위 가드
            return ratio
    return None


def calc_max_pain(chain):
    """옵션 매도자 손실 최소 가격"""
    calls, puts = chain.calls, chain.puts
    if calls.empty or puts.empty:
        return None
    strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
    min_pain = float("inf")
    mp_strike = None
    for s in strikes:
        pain = 0
        for _, row in calls.iterrows():
            oi = row.get("openInterest", 0) or 0
            if row["strike"] < s:
                pain += (s - row["strike"]) * oi * 100
        for _, row in puts.iterrows():
            oi = row.get("openInterest", 0) or 0
            if row["strike"] > s:
                pain += (row["strike"] - s) * oi * 100
        if pain < min_pain:
            min_pain = pain
            mp_strike = s
    return mp_strike


def calc_options_target(chains, price, hot_strikes):
    """
    옵션 시장이 암시하는 목표가와 기한 추출.
    1순위: 현재가 위 가장 큰 콜 OI Wall (감마 자석 = 끌려갈 가격)
    2순위: 달러 규모 최대 OTM 콜 포지션 (확신 있는 베팅)
    ★ 스캔 당일 만기는 제외 (핀 리스크 — 자석 아니라 못)
    """
    if not price or price == 0:
        return None

    today_str = TODAY.strftime("%Y-%m-%d")

    # 현재가 위 콜 OI 중 가장 큰 것 (당일 만기 제외)
    above_oi = []
    for date, chain in chains:
        if date == today_str:
            continue  # 당일 만기 = 핀, 방향성 신호 아님
        for _, row in chain.calls.iterrows():
            oi = row.get("openInterest", 0) or 0
            strike = row["strike"]
            if oi > 100 and strike > price * 1.02:  # 현재가 +2% 이상
                above_oi.append({
                    "date": date, "strike": strike, "oi": int(oi),
                    "upside": (strike - price) / price * 100,
                })

    if above_oi:
        # OI 가장 큰 것
        best = max(above_oi, key=lambda x: x["oi"])
        return {
            "price": best["strike"],
            "date": best["date"],
            "upside": best["upside"],
            "basis": "OI Wall",
            "oi": best["oi"],
        }

    # OI가 없으면 hot strike 중 OTM 콜 최대 달러규모 (당일 만기 제외)
    otm_hots = [h for h in hot_strikes if h["otm_pct"] > 2 and h.get("date") != today_str]
    if otm_hots:
        best = max(otm_hots, key=lambda x: x["dollar_val"])
        return {
            "price": best["strike"],
            "date": best["date"],
            "upside": best["otm_pct"],
            "basis": "Hot Call",
            "oi": best.get("oi", 0),
        }

    return None


def get_top_call_oi(chains, price, top_n=3):
    """전체 체인에서 콜 OI가 가장 큰 행사가 top N"""
    all_rows = []
    for date, chain in chains:
        for _, row in chain.calls.iterrows():
            oi = row.get("openInterest", 0) or 0
            if oi > 0:
                all_rows.append({
                    "date": date,
                    "strike": row["strike"],
                    "oi": int(oi),
                    "otm_pct": (row["strike"] - price) / price * 100 if price else 0,
                })
    all_rows.sort(key=lambda x: x["oi"], reverse=True)
    return all_rows[:top_n]


def get_5d_return(ticker):
    """최근 5거래일 수익률"""
    try:
        h = yf.Ticker(ticker).history(period="7d")
        if h is not None and len(h) >= 2:
            return (h["Close"].iloc[-1] / h["Close"].iloc[0]) - 1
    except:
        pass
    return None


def classify_flow(r):
    """옵션 흐름 성격 분류 (v3.4: 헤지성 풋 추가)"""
    pc_vol = safe_div(r["total_put_vol"], r["total_call_vol"], 1)
    pc_oi = safe_div(r["total_put_oi"], r["total_call_oi"], 1)
    hot = r.get("hot_strikes", [])
    otm_hots = [h for h in hot if h["otm_pct"] > 5]
    far_otm_hots = [h for h in hot if h["otm_pct"] > 30]
    ret_5d = r.get("ret_5d")
    skew = r.get("iv_skew")

    if far_otm_hots and len(far_otm_hots) >= max(len(otm_hots) * 0.7, 1):
        return "복권성 콜"
    # v3.4 Candidate D — 주가 상승 + P/C 높음 + 풋 IV 높음 → 보유분 헤지 (강세 시그널)
    if (pc_vol >= 1.3
            and ret_5d is not None and ret_5d >= 0.10
            and skew is not None and skew >= 0):
        return "헤지성 풋"
    if pc_vol > 1.2 and pc_oi > 1.0:
        return "약세 방향성"
    ts = r.get("term_structure")
    if ts and ts < 0.9 and 0.7 < pc_vol < 1.3:
        return "이벤트 변동성"
    if pc_oi > 1.0 and pc_vol < 0.7:
        return "헤지+콜 유입"
    if pc_vol < 0.6 and otm_hots:
        return "강세 방향성"
    if pc_vol < 0.5:
        return "콜 우위"
    return "혼조"


# ── 메인 스캔 ──

def scan_ticker(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        mcap = info.get("marketCap", 0)
        avg_vol = info.get("averageVolume", 0)

        # 실적 발표일
        try:
            ed = t.get_earnings_dates(limit=4)
            next_earn = None
            if ed is not None and not ed.empty:
                future = [d for d in ed.index if d.date() >= TODAY]
                if future:
                    next_earn = min(future).date()
        except:
            next_earn = None

        # 5일 수익률
        ret_5d = get_5d_return(ticker)

        dates = t.options
        if not dates:
            return None

        near_dates = [d for d in dates if datetime.strptime(d, "%Y-%m-%d").date() <= NEAR_END]
        mid_dates = [d for d in dates if NEAR_END < datetime.strptime(d, "%Y-%m-%d").date() <= MID_END]
        far_dates = [d for d in dates if MID_END < datetime.strptime(d, "%Y-%m-%d").date() <= FAR_END]

        r = {
            "ticker": ticker, "price": price, "mcap": mcap,
            "cap_label": cap_label(mcap), "avg_vol": avg_vol,
            "ret_5d": ret_5d,
            "next_earnings": next_earn,
            "earn_within_30d": next_earn and (next_earn - TODAY).days <= 30 if next_earn else False,
            "near_call_vol": 0, "near_put_vol": 0,
            "near_call_oi": 0, "near_put_oi": 0,
            "mid_call_vol": 0, "mid_put_vol": 0,
            "mid_call_oi": 0, "mid_put_oi": 0,
            "total_call_vol": 0, "total_put_vol": 0,
            "total_call_oi": 0, "total_put_oi": 0,
            "hot_strikes": [],
            "iv_skew": None, "term_structure": None,
            "near_max_pain": None, "top_call_oi": [],
        }

        first_near_chain = None
        first_far_chain = None
        all_chains = []  # (date, chain) for OI analysis

        # ── 단기 만기 ──
        for d in near_dates:
            chain = t.option_chain(d)
            all_chains.append((d, chain))
            if first_near_chain is None:
                first_near_chain = chain
                r["near_max_pain"] = calc_max_pain(chain)

            calls, puts = chain.calls, chain.puts
            c_vol = calls["volume"].sum() if "volume" in calls.columns else 0
            p_vol = puts["volume"].sum() if "volume" in puts.columns else 0
            c_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
            p_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
            r["near_call_vol"] += c_vol
            r["near_put_vol"] += p_vol
            r["near_call_oi"] += c_oi
            r["near_put_oi"] += p_oi

            for _, row in calls.iterrows():
                vol = row.get("volume", 0) or 0
                oi = row.get("openInterest", 0) or 0
                strike = row.get("strike", 0)
                iv = row.get("impliedVolatility", 0) or 0
                if vol > 300 and (oi == 0 or vol / max(oi, 1) > 0.4):
                    r["hot_strikes"].append({
                        "date": d, "strike": strike,
                        "vol": int(vol), "oi": int(oi),
                        "ratio": vol / max(oi, 1),
                        "otm_pct": (strike - price) / price * 100 if price else 0,
                        "iv": iv,
                        "zone": classify_strike(strike, price),
                        "dollar_val": vol * (row.get("lastPrice", 0) or 0) * 100,
                        "period": "단기",
                    })

        # ── 중기 만기 ──
        for d in mid_dates:
            chain = t.option_chain(d)
            all_chains.append((d, chain))
            calls, puts = chain.calls, chain.puts
            c_vol = calls["volume"].sum() if "volume" in calls.columns else 0
            p_vol = puts["volume"].sum() if "volume" in puts.columns else 0
            c_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
            p_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
            r["mid_call_vol"] += c_vol
            r["mid_put_vol"] += p_vol
            r["mid_call_oi"] += c_oi
            r["mid_put_oi"] += p_oi

            for _, row in calls.iterrows():
                vol = row.get("volume", 0) or 0
                oi = row.get("openInterest", 0) or 0
                strike = row.get("strike", 0)
                iv = row.get("impliedVolatility", 0) or 0
                if vol > 300 and (oi == 0 or vol / max(oi, 1) > 0.4):
                    r["hot_strikes"].append({
                        "date": d, "strike": strike,
                        "vol": int(vol), "oi": int(oi),
                        "ratio": vol / max(oi, 1),
                        "otm_pct": (strike - price) / price * 100 if price else 0,
                        "iv": iv,
                        "zone": classify_strike(strike, price),
                        "dollar_val": vol * (row.get("lastPrice", 0) or 0) * 100,
                        "period": "중기",
                    })

        # ── 장기 만기 (IV 분석용) ──
        for d in far_dates[:1]:
            chain = t.option_chain(d)
            if first_far_chain is None:
                first_far_chain = chain

        # IV skew
        skew_chain = first_far_chain or (t.option_chain(mid_dates[-1]) if mid_dates else first_near_chain)
        if skew_chain:
            r["iv_skew"] = calc_iv_skew(skew_chain, price)

        # Term structure
        if first_near_chain and first_far_chain:
            r["term_structure"] = calc_term_structure(first_near_chain, first_far_chain, price)

        # 집계
        r["total_call_vol"] = r["near_call_vol"] + r["mid_call_vol"]
        r["total_put_vol"] = r["near_put_vol"] + r["mid_put_vol"]
        r["total_call_oi"] = r["near_call_oi"] + r["mid_call_oi"]
        r["total_put_oi"] = r["near_put_oi"] + r["mid_put_oi"]

        # 상대 옵션 강도: 총 옵션 거래량 / 주식 평균 거래량
        total_opt_vol = r["total_call_vol"] + r["total_put_vol"]
        r["opt_intensity"] = safe_div(total_opt_vol, avg_vol) if avg_vol else None

        # 달러 가치 순 정렬
        r["hot_strikes"].sort(key=lambda x: x["dollar_val"], reverse=True)

        # Top 콜 OI 행사가 (감마 자석)
        r["top_call_oi"] = get_top_call_oi(all_chains, price, top_n=3)

        # 옵션 목표가/기한
        r["options_target"] = calc_options_target(all_chains, price, r["hot_strikes"])

        # 흐름 분류
        r["flow_type"] = classify_flow(r)

        return r
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ── 스코어링: 5팩터 + 흐름 가감점 ──

def calc_score(r):
    if "error" in r:
        return 0
    score = 0

    # ① P/C Ratio (v3.5 — 콘트라리언 반전 강화)
    # 근거: 17일치 296쌍 — P/C<0.4 평균 -0.74% (소매 FOMO 천장),
    #        P/C≥1.3 평균 +1.29% (펀드 헤지 매수 = 보유 강세)
    # v3.5 재검증: 최근 592건 5D 기준 0.7~1.3 구간이 가장 양호,
    #        P/C<0.4 고득점군은 약해 극단 콜 우위 페널티 강화.
    pc = safe_div(r["total_put_vol"], r["total_call_vol"], 1)
    if pc < 0.4:
        score -= 15        # 소매 FOMO 천장
    elif pc < 0.7:
        score += 8         # 콜 우위 (건강)
    elif pc < 1.3:
        score += 12        # 균형/헤지 혼합 구간
    else:
        score += 15        # 헤지 풋 우위 → 큰손 보유 + 보험 = 강세 시그널

    # ② 이상 콜 규모 (v3.4 — $20-40M 보너스 추가, Candidate C)
    hot = r.get("hot_strikes", [])
    meaningful = [h for h in hot if not (h["otm_pct"] > 30 and h["dollar_val"] < 500_000)]
    if meaningful:
        top_dollar = sum(h["dollar_val"] for h in meaningful[:5])
        if top_dollar > 40_000_000:
            score += 25    # 단일일 폭증 — streak 없어도 큰손 진입
        elif top_dollar > 20_000_000:
            score += 23    # Candidate C 신규 보너스 영역
        elif top_dollar > 5_000_000:
            score += 20
        elif top_dollar > 1_000_000:
            score += 14
        elif top_dollar > 300_000:
            score += 8
        elif top_dollar > 50_000:
            score += 3

    # ③ 상대 옵션 강도 (v3.5 — 노이즈 축소)
    # 최근 누적 로그에서 intensity 단독 상관이 약해 최대 가점을 낮춤.
    intensity = r.get("opt_intensity")
    if intensity is not None:
        if intensity > 0.15:
            score += 8
        elif intensity > 0.08:
            score += 5
        elif intensity > 0.04:
            score += 2
        elif intensity > 0.02:
            score += 0
    else:
        score += 0

    # ④ IV Skew (변경 없음)
    skew = r.get("iv_skew")
    if skew is not None:
        if skew < -0.10:
            score += 20
        elif skew < -0.03:
            score += 12
        elif skew < 0.03:
            score += 6
    else:
        score += 6

    # ⑤ Term Structure (이상치 가드 그대로)
    ts = r.get("term_structure")
    if ts is not None and ts <= 10:
        if ts < 0.85:
            score += 20
        elif ts < 0.95:
            score += 12
        elif ts < 1.05:
            score += 5
    else:
        score += 5

    # ⑥ 흐름 성격 (v3.5 — 방향성 콜 과점수 방어)
    # 최근 로그에서 강세 방향성은 수익률 구분력이 약했고,
    # 헤지+콜/이벤트/약세 방향성은 콘트라리언 의미가 더 강했음.
    flow = r.get("flow_type", "혼조")
    if flow == "강세 방향성":
        score += 3
    elif flow == "헤지성 풋":           # NEW v3.4 (Candidate D)
        score += 8
    elif flow == "헤지+콜 유입":         # v3.4: -5 → +10 (Candidate E)
        score += 12
    elif flow == "이벤트 변동성":
        score += 5
    elif flow == "혼조":
        score += 0
    elif flow == "약세 방향성":
        score -= 5

    # ⑦ 5일 수익률 (v3.4 — 페널티 완화, Candidate B)
    # 근거: SNDK +16.6%, MU +15.5%, AMD +11.4% 같은 추세 지속 종목이
    #        기존 -8/-15 페널티 때문에 점수 깎여 매수 시그널 놓치던 문제 해결
    ret_5d = r.get("ret_5d")
    if ret_5d is not None:
        pct = ret_5d * 100
        if pct >= 50:
            score -= 15        # 진짜 과열만 강한 페널티
        elif pct >= 30:
            score -= 5         # 완화: 기존 -15 → -5
        elif pct >= 20:
            score -= 3         # 완화: 기존 -8 → -3
        elif pct >= 15:
            score += 3         # 추세 지속
        elif pct >= 10:
            score += 5         # 건강한 모멘텀 (구제 핵심 구간)
        elif pct >= 5:
            score += 3
        elif pct >= -5:
            score += 5         # 신선한 베팅
        elif pct < -10:
            score -= 5

    # 어닝 ±3일 신뢰도 반감 (v3.4 신규)
    # 근거: CRWV/UUUU 어닝 미스로 옵션 점수 무력화 — 어닝 임박은 정보 비대칭 구간
    next_earn = r.get("next_earnings")
    if next_earn:
        try:
            days_to_earn = (next_earn - TODAY).days
            if -3 <= days_to_earn <= 3:
                score = int(score * 0.7)
                r["earnings_proximity_flag"] = True
        except Exception:
            pass

    # 소형주 유동성 할인
    if r.get("cap_label") == "소형":
        score = int(score * 0.85)

    return max(min(score, 100), 0)


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "options_log")

HOT_DOLLAR_THRESHOLD = 40_000_000   # $40M
HOT_DOLLAR_STREAK_DAYS = 3
HOT_DOLLAR_BONUS = 10

# ⑩ ETF 시그널 가감점 (v3.3) — 14일치 디버전스 35건 검증, ETF 69% 적중
ETF_LOOKBACK_DAYS = 7
ETF_BUY_THRESHOLD = 0.10
ETF_SELL_THRESHOLD = -0.10
ETF_OPT_BULL_CUTOFF = 50
ETF_OPT_BEAR_CUTOFF = 40
ETF_BONUS = 10


def get_etf_signal_map(target_date_str, lookback_calendar_days=ETF_LOOKBACK_DAYS):
    """target 시점에 가까운 영업일 ETF vs lookback 일 전 ETF 비중 변화 (1주 누적).
    종목별 dict {ticker: weight_diff_pp} 반환. ETF 데이터 없으면 빈 dict.
    현재일 스캔에서는 같은 날짜 ETF 스냅샷이 없으면 stale 보정을 막기 위해 비활성화."""
    if not ETF_AVAILABLE:
        return {}
    snap_dates = list_snapshot_dates()
    if not snap_dates:
        return {}
    td = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    curr_d = next((d for d in sorted(snap_dates, reverse=True)
                   if datetime.strptime(d, "%Y-%m-%d").date() <= td), None)
    if not curr_d:
        return {}
    if td == TODAY and curr_d != target_date_str:
        return {}
    cutoff = td - timedelta(days=lookback_calendar_days)
    prev_d = next((d for d in sorted(snap_dates, reverse=True)
                   if datetime.strptime(d, "%Y-%m-%d").date() <= cutoff), None)
    if not prev_d or prev_d == curr_d:
        return {}
    curr = load_snapshot(curr_d)
    prev = load_snapshot(prev_d)
    if not curr or not prev:
        return {}
    def to_w(snap):
        out = {}
        for h in snap["holdings"]:
            t = get_us_ticker(h.get("code", ""))
            if t:
                out[t] = h.get("weight", 0)
        return out
    curr_w, prev_w = to_w(curr), to_w(prev)
    return {t: curr_w.get(t, 0) - prev_w.get(t, 0)
            for t in set(curr_w) | set(prev_w)}


def load_prev_hot_dollars(ticker, n_days=HOT_DOLLAR_STREAK_DAYS):
    """직전 n 거래일 로그에서 해당 종목 hot_dollar 반환 (오래된 순)"""
    if not os.path.exists(LOG_DIR):
        return []
    today_str = TODAY.strftime("%Y-%m-%d") + ".json"
    files = sorted(f for f in os.listdir(LOG_DIR) if f.endswith(".json") and f != today_str)
    values = []
    for fname in files[-n_days:]:
        try:
            with open(os.path.join(LOG_DIR, fname), encoding="utf-8") as fp:
                data = json.load(fp)
            for entry in data.get("rankings", []):
                if entry["ticker"] == ticker:
                    values.append(entry["factors"].get("hot_dollar", 0) or 0)
                    break
        except:
            pass
    return values


def save_daily_log(profit_results, loss_results):
    """당일 스캔 결과를 JSON으로 저장"""
    os.makedirs(LOG_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = os.path.join(LOG_DIR, f"{date_str}.json")

    def make_entries(results, group):
        entries = []
        for rank, r in enumerate(results, 1):
            if "error" in r:
                continue
            entries.append({
                "rank": rank,
                "group": group,
                "ticker": r["ticker"],
                "price": r["price"],
                "score": r["score"],
                "flow_type": r["flow_type"],
                "cap_label": r["cap_label"],
                "factors": {
                    "pc_ratio": safe_div(r["total_put_vol"], r["total_call_vol"], None),
                    "hot_dollar": sum(h["dollar_val"] for h in r.get("hot_strikes", [])[:5]
                                      if not (h["otm_pct"] > 30 and h["dollar_val"] < 500_000)),
                    "opt_intensity": r.get("opt_intensity"),
                    "iv_skew": r.get("iv_skew"),
                    "term_structure": r.get("term_structure"),
                    "ret_5d": r.get("ret_5d"),
                    "etf_diff_1w": r.get("etf_diff_1w"),
                },
                "forward_returns": {"1d": None, "3d": None, "5d": None},
                "hot_dollar_streak": r.get("hot_dollar_streak", False),
                "etf_adjustment": r.get("etf_adjustment", 0),
                "earnings_proximity_flag": r.get("earnings_proximity_flag", False),
                "score_version": SCORE_VERSION,
            })
        return entries

    log = {
        "date": date_str,
        "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rankings": make_entries(profit_results, "profit") + make_entries(loss_results, "loss"),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n  로그 저장: {filepath}")


# ── 출력 함수 ──

def _ts_fmt(ts):
    if ts is None: return "N/A"
    return f"{ts:.2f}" if ts <= 10 else "(!)"

def _tgt_fmt(r):
    t = r.get("options_target")
    if not t: return "-"
    return f"${t['price']:.0f}({t['upside']:+.0f}%) ~{t['date'][5:]}"

def _earn_fmt(r):
    if not r.get("next_earnings"): return ""
    days = (r["next_earnings"] - TODAY).days
    return f"{days}일후" if days <= 30 else ""


def _etf_fmt(r):
    d = r.get("etf_diff_1w")
    if d is None:
        return "—"
    adj = r.get("etf_adjustment", 0)
    flag = ""
    if adj < 0:
        flag = " 🚨"  # 옵션 강세 + ETF 매도
    elif adj > 0:
        flag = " 💎"  # 옵션 약세 + ETF 매수 (펀드매니저 정보 우위)
    return f"{d:+.2f}p{flag}"


def print_summary_md(results, title):
    """마크다운 요약 테이블 출력"""
    print(f"\n### {title}\n")
    print("| 종목 | 현재가 | 5D | 점수 | 흐름 | ETF Δ1w | P/C | 강도 | Skew | Term | 옵션목표 | 실적 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            continue
        streak = " ★" if r.get("hot_dollar_streak") else ""
        pc   = f"{safe_div(r['total_put_vol'], r['total_call_vol'], 99):.2f}"
        oi   = f"{r['opt_intensity']:.1%}" if r.get("opt_intensity") else "N/A"
        skew = f"{r['iv_skew']:+.2f}" if r.get("iv_skew") is not None else "N/A"
        ts   = _ts_fmt(r.get("term_structure"))
        ret  = f"{r['ret_5d']:+.1%}" if r.get("ret_5d") is not None else "N/A"
        print(f"| **{r['ticker']}{streak}** | ${r['price']:.2f} | {ret} | **{r['score']}** | {r['flow_type']} | {_etf_fmt(r)} | {pc} | {oi} | {skew} | {ts} | {_tgt_fmt(r)} | {_earn_fmt(r)} |")
    print()


def format_detail_md(results, title):
    """전종목 상세 블록 — 마크다운 문자열 반환"""
    lines = [f"# {title} 상세\n", f"_생성: {TODAY}_\n", "---\n"]

    for r in results:
        if "error" in r:
            continue
        ret = r.get("ret_5d")
        ret_s = f"{ret:+.1%}" if ret is not None else "N/A"
        streak = " ★" if r.get("hot_dollar_streak") else ""

        lines.append(f"## {r['ticker']}{streak} — {r['score']}pt | {r['flow_type']} | ${r['price']:.2f} | 5D {ret_s}\n")

        if r.get("hot_dollar_streak"):
            lines.append(f"> **★ hot_dollar 누적 신호 (+{HOT_DOLLAR_BONUS}pt)** — 직전 {HOT_DOLLAR_STREAK_DAYS}거래일 연속 ${HOT_DOLLAR_THRESHOLD//1_000_000}M+ 콜 매수\n")

        adj = r.get("etf_adjustment", 0)
        etf_d = r.get("etf_diff_1w")
        if adj < 0:
            lines.append(f"> **🚨 ETF 디버전스 ({adj:+d}pt)** — 옵션 강세인데 펀드매니저 1주 비중 {etf_d:+.2f}p 매도. 디버전스 A 패턴 (검증 65% ETF 적중)\n")
        elif adj > 0:
            lines.append(f"> **💎 펀드매니저 정보 우위 (+{adj}pt)** — 옵션 약세인데 펀드매니저 1주 비중 {etf_d:+.2f}p 매수. 디버전스 B 패턴 (검증 73% ETF 적중)\n")

        tgt = r.get("options_target")
        if tgt:
            lines.append(f"- **옵션목표**: ${tgt['price']:.0f} ({tgt['upside']:+.1f}%) ~{tgt['date']} [{tgt['basis']}, OI {tgt['oi']:,}]")

        if ret is not None:
            pc = safe_div(r["total_put_vol"], r["total_call_vol"], 1)
            if ret < -0.03 and pc < 0.5:
                lines.append("- **⚡ 주가 하락 vs 콜 우위 → 역발상 강세 신호**")
            elif ret > 0.05 and pc < 0.5:
                lines.append("- 주가 상승 + 콜 우위 → 추세 확인, 모멘텀 지속 기대")
            elif ret > 0.03 and pc > 1.0:
                lines.append("- **⚠ 주가 상승 vs 풋 우위 → 헤지/차익실현 진행 중**")

        mp = r.get("near_max_pain")
        if mp:
            dev = (mp - r["price"]) / r["price"] * 100
            msg = "현재가 MaxPain 크게 상회 → 만기 시 하방 압력" if dev < -10 else \
                  "현재가 MaxPain 크게 하회 → 반등 가능성" if dev > 10 else "MaxPain 근처"
            lines.append(f"- **MaxPain**: ${mp:.0f} ({dev:+.1f}%) — {msg}")

        skew = r.get("iv_skew")
        if skew is not None:
            msg = "콜 IV > 풋 IV → 상방 수요 강함" if skew < -0.05 else \
                  "풋 IV > 콜 IV → 하방 헤지 수요" if skew > 0.05 else "중립적"
            lines.append(f"- **IV Skew**: {skew:+.3f} ({msg})")
        else:
            lines.append("- **IV Skew**: N/A")

        ts = r.get("term_structure")
        if ts is not None:
            msg = "강한 역전 → 단기 급변 기대" if ts < 0.90 else \
                  "소폭 역전" if ts < 1.0 else "정상"
            lines.append(f"- **Term Structure**: {ts:.3f} ({msg})")

        if r.get("earn_within_30d"):
            days = (r["next_earnings"] - TODAY).days
            lines.append(f"- **실적**: {days}일 후")

        lines.append("")

        top_oi = r.get("top_call_oi", [])
        if top_oi:
            lines.append("#### 콜 OI Wall (감마 자석)")
            lines.append("| 만기 | 행사가 | OI | 방향 |")
            lines.append("|---|---|---|---|")
            for w in top_oi:
                otm = w["otm_pct"]
                label = f"OTM +{otm:.0f}%" if otm > 3 else f"ITM {otm:+.0f}%" if otm < -3 else "ATM"
                above = "↑위" if otm > 0 else "↓아래"
                lines.append(f"| {w['date']} | ${w['strike']:.0f} ({label}) | {w['oi']:,} | {above} |")
            lines.append("")

        hot = r.get("hot_strikes", [])
        meaningful = [h for h in hot if not (h["otm_pct"] > 30 and h["dollar_val"] < 500_000)]
        if meaningful:
            lines.append("#### 이상 콜 포지션")
            lines.append("| 기간 | 만기 | 행사가 | Vol | OI | Vol/OI | 규모 |")
            lines.append("|---|---|---|---|---|---|---|")
            for h in meaningful[:5]:
                otm_label = f"OTM +{h['otm_pct']:.0f}%" if h["otm_pct"] > 3 else \
                            f"ITM {h['otm_pct']:+.0f}%" if h["otm_pct"] < -3 else "ATM"
                d = h["dollar_val"]
                d_str = f"${d/1_000_000:.1f}M" if d >= 1_000_000 else f"${d/1_000:.0f}K"
                lines.append(f"| {h['period']} | {h['date']} | ${h['strike']:.0f} ({otm_label}) | {h['vol']:,} | {h['oi']:,} | {h['ratio']:.1f}x | {d_str} |")
        else:
            lines.append("_이상 콜 포지션 없음_")

        lines.append("\n---\n")

    return "\n".join(lines)


def scan_group(tickers, label, etf_signal_map=None):
    """종목 그룹 스캔 → 점수 정렬된 결과 리스트 반환"""
    results = []
    print(f"\n  ── {label} 스캔 ({len(tickers)}종목) ──")
    for ticker in tickers:
        print(f"  {ticker}...", end="", flush=True)
        r = scan_ticker(ticker)
        if r:
            results.append(r)
            if "error" not in r:
                inten = r.get("opt_intensity")
                inten_s = f"{inten:.1%}" if inten else "N/A"
                print(f" OK ({inten_s})", end="")
            else:
                print(f" ERR", end="")
        else:
            print(" SKIP", end="")
    print()

    for r in results:
        r["score"] = calc_score(r)

    # ⑧ 섹터 베타 가점 — 대장주(40+)가 있으면 같은 섹터 나머지에 +5
    sector_leader = {}
    for r in results:
        t = r.get("ticker")
        sec = TICKER_TO_SECTOR.get(t)
        if not sec:
            continue
        s = r.get("score", 0)
        if s >= 40 and s > sector_leader.get(sec, (None, 0))[1]:
            sector_leader[sec] = (t, s)

    for r in results:
        t = r.get("ticker")
        sec = TICKER_TO_SECTOR.get(t)
        if not sec or sec not in sector_leader:
            continue
        leader_ticker, _ = sector_leader[sec]
        if t == leader_ticker:
            continue
        r["sector_beta"] = 5
        r["score"] = min(r["score"] + 5, 100)

    # ⑨ hot_dollar 누적 신호 — 직전 3거래일 연속 $40M+ 이면 +10
    for r in results:
        if "error" in r:
            continue
        prev = load_prev_hot_dollars(r["ticker"])
        if len(prev) >= HOT_DOLLAR_STREAK_DAYS and all(v >= HOT_DOLLAR_THRESHOLD for v in prev):
            r["hot_dollar_streak"] = True
            r["score"] = min(r["score"] + HOT_DOLLAR_BONUS, 100)

    # ⑩ ETF 시그널 가감점 (v3.3) — 펀드매니저 정보 우위 보정
    if etf_signal_map:
        for r in results:
            if "error" in r:
                continue
            etf_diff = etf_signal_map.get(r["ticker"])
            r["etf_diff_1w"] = etf_diff
            r["etf_adjustment"] = 0
            if etf_diff is None:
                continue
            cur = r.get("score", 0)
            if cur >= ETF_OPT_BULL_CUTOFF and etf_diff <= ETF_SELL_THRESHOLD:
                r["score"] = max(cur - ETF_BONUS, 0)
                r["etf_adjustment"] = -ETF_BONUS
            elif cur <= ETF_OPT_BEAR_CUTOFF and etf_diff >= ETF_BUY_THRESHOLD:
                r["score"] = min(cur + ETF_BONUS, 100)
                r["etf_adjustment"] = ETF_BONUS

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results


# ── 메인 ──

if __name__ == "__main__":
    import sys, io, contextlib

    detail_flag = "--detail" in sys.argv

    now = datetime.now()
    print()
    print(f"  ┌──────────────────────────────────────────────────────────┐")
    print(f"  |  GBRC 옵션 스캐너 v3 - 흑자/적자 상대 강도 비교           |")
    print(f"  |  {now.strftime('%Y-%m-%d %H:%M')}                                            |")
    print(f"  |  단기: ~{NEAR_END}  |  중기: ~{MID_END}  |  장기: ~{FAR_END}  |")
    print(f"  └──────────────────────────────────────────────────────────┘")

    # ETF 시그널 (v3.3) — 디버전스 보정용
    etf_sig_map = get_etf_signal_map(TODAY.strftime("%Y-%m-%d"))
    if etf_sig_map:
        print(f"  ETF 시그널 ({len(etf_sig_map)}종목, 1주 비중 변화 매칭)")
    else:
        print(f"  ETF 시그널: 데이터 없음 (⑩ 레이어 비활성)")

    # 스캔
    profit_results = scan_group(PROFIT_TICKERS, "흑자기업", etf_sig_map)
    loss_results = scan_group(LOSS_TICKERS, "적자기업", etf_sig_map)

    # 요약 테이블 (항상 출력 + summary md 저장)
    sum_buf = io.StringIO()
    with contextlib.redirect_stdout(sum_buf):
        print_summary_md(profit_results, "흑자기업 옵션 흐름")
        print_summary_md(loss_results, "적자기업 옵션 흐름")
    summary_md = sum_buf.getvalue()
    print(summary_md)

    os.makedirs(LOG_DIR, exist_ok=True)
    summary_path = os.path.join(LOG_DIR, f"{TODAY.strftime('%Y-%m-%d')}_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# GBRC 옵션 요약 — {TODAY}\n\n{summary_md}")

    # 상세 블록 — 마크다운으로 자동 저장
    detail_md = format_detail_md(profit_results, "흑자기업") + \
                format_detail_md(loss_results, "적자기업")

    os.makedirs(LOG_DIR, exist_ok=True)
    detail_path = os.path.join(LOG_DIR, f"{TODAY.strftime('%Y-%m-%d')}_detail.md")
    with open(detail_path, "w", encoding="utf-8") as f:
        f.write(detail_md)

    # 로그 저장
    save_daily_log(profit_results, loss_results)
    print(f"\n  상세 저장: {detail_path}")

    # --detail 플래그 시 상세 콘솔 출력
    if detail_flag:
        print(detail_md)

    # 범례
    print()
    print("  ── 스코어링 5팩터 ──")
    print("  ① P/C Ratio    <0.4 과열 감점 | 0.7~1.3 균형/헤지 | >1.3 헤지 우위")
    print("  ② 이상 콜 규모  달러 기준 실제 프리미엄 규모 (복권성 제외)")
    print("  ③ 상대 옵션강도  v3.5부터 보조 지표로 축소 반영")
    print("  ④ IV Skew      음수 = 콜 수요 > 풋 수요 (구조적 강세)                (20점)")
    print("  ⑤ Term Str.    < 0.90 역전 = 단기 이벤트 기대 | > 1.0 정상            (20점)")
    print()
    print("  ── 참고 ──")
    print("  흐름 성격   헤지+콜/헤지성 풋 우선, 강세방향성 콜 과열은 제한 반영")
    print("  MaxPain    만기 시 주가가 끌려가기 쉬운 가격 (주가 > MP → 하방 압력)")
    print("  콜 OI Wall  OI 집중 행사가 = 감마 자석, 돌파 시 딜러 헤지 매수로 급등 트리거")
    print("  5D 수익률   주가 흐름과 옵션 흐름의 괴리 여부 판단용")
    print("  소형주(<2B) 유동성 낮아 신호 노이즈 가능 → 점수 15% 할인")
    print()
