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

TODAY = datetime.now().date()
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
    """OTM 풋 IV - OTM 콜 IV. 음수 = 콜 수요 강함(강세)"""
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
        return put_iv - call_iv
    return None


def calc_term_structure(near_chain, far_chain, price):
    """장기 ATM IV / 단기 ATM IV. < 1.0 = 역전(단기 이벤트 기대)"""
    def get_atm_iv(chain, price):
        calls = chain.calls
        if calls.empty or price == 0:
            return None
        atm = calls.iloc[(calls["strike"] - price).abs().argsort()[:3]]
        iv = atm["impliedVolatility"].median()
        return iv if iv and not math.isnan(iv) else None
    near_iv = get_atm_iv(near_chain, price)
    far_iv = get_atm_iv(far_chain, price)
    if near_iv and far_iv and near_iv > 0:
        return far_iv / near_iv
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
    """옵션 흐름 성격 분류"""
    pc_vol = safe_div(r["total_put_vol"], r["total_call_vol"], 1)
    pc_oi = safe_div(r["total_put_oi"], r["total_call_oi"], 1)
    hot = r.get("hot_strikes", [])
    otm_hots = [h for h in hot if h["otm_pct"] > 5]
    far_otm_hots = [h for h in hot if h["otm_pct"] > 30]

    if far_otm_hots and len(far_otm_hots) >= max(len(otm_hots) * 0.7, 1):
        return "복권성 콜"
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


# ── 스코어링: 5팩터 × 20점 + 흐름 가감점 ──

def calc_score(r):
    if "error" in r:
        return 0
    score = 0

    # ① P/C Ratio (20점) — 콜 우위 = 강세
    pc = safe_div(r["total_put_vol"], r["total_call_vol"], 1)
    if pc < 0.3:
        score += 20
    elif pc < 0.5:
        score += 15
    elif pc < 0.7:
        score += 8
    elif pc > 1.3:
        score -= 5

    # ② 이상 콜 규모 (20점) — 달러 기준, 복권성 제외
    hot = r.get("hot_strikes", [])
    meaningful = [h for h in hot if not (h["otm_pct"] > 30 and h["dollar_val"] < 500_000)]
    if meaningful:
        top_dollar = sum(h["dollar_val"] for h in meaningful[:5])
        if top_dollar > 5_000_000:
            score += 20
        elif top_dollar > 1_000_000:
            score += 14
        elif top_dollar > 300_000:
            score += 8
        elif top_dollar > 50_000:
            score += 3

    # ③ 상대 옵션 강도 (20점) — 종목간 공정 비교의 핵심
    intensity = r.get("opt_intensity")
    if intensity is not None:
        if intensity > 0.15:     # 옵션 거래량이 주식 거래량의 15% 이상
            score += 20
        elif intensity > 0.08:
            score += 14
        elif intensity > 0.04:
            score += 8
        elif intensity > 0.02:
            score += 3
    else:
        score += 6  # 데이터 없으면 중립

    # ④ IV Skew (20점) — 구조적 방향성 신호
    skew = r.get("iv_skew")
    if skew is not None:
        if skew < -0.10:
            score += 20
        elif skew < -0.03:
            score += 12
        elif skew < 0.03:
            score += 6
        # skew > 0.03: 풋 수요 높음 → 가산 없음
    else:
        score += 6  # N/A → 중립 (불이익 방지)

    # ⑤ Term Structure (20점) — 역전 = 단기 급변 기대
    # ★ 값이 10 초과 시 이상치로 간주 (계산 왜곡) → N/A 처리
    ts = r.get("term_structure")
    if ts is not None and ts <= 10:
        if ts < 0.85:
            score += 20
        elif ts < 0.95:
            score += 12
        elif ts < 1.05:
            score += 5
    else:
        score += 5  # N/A 또는 이상치 → 중립

    # ⑥ 흐름 성격 가감점 — 돈의 확신도 반영
    flow = r.get("flow_type", "혼조")
    if flow == "강세 방향성":
        score += 10
    elif flow == "이벤트 변동성":
        score -= 3
    elif flow == "헤지+콜 유입":
        score -= 5
    elif flow == "혼조":
        score -= 5
    elif flow == "약세 방향성":
        score -= 15

    # ⑦ 5일 수익률 — 이미 끝난 파티 페널티 / 신선한 신호 가점
    # v3.2: +10~20% 구간 가점 공백 보정 (MRVL/AMD 유형의 건강한 추세 지속 구제)
    ret_5d = r.get("ret_5d")
    if ret_5d is not None:
        pct = ret_5d * 100
        if pct >= 30:
            score -= 15  # 과열 — 단기 고점 리스크
        elif pct >= 20:
            score -= 8
        elif pct >= 15:
            score += 1   # 추세 지속 (완만)
        elif pct >= 10:
            score += 3   # 건강한 추세 초기
        elif pct >= 5:
            score += 3   # 건강한 상승
        elif pct >= -5:
            score += 5   # 신선한 베팅
        elif pct < -10:
            score -= 5   # 하락 추세 중 콜 매수는 역추세

    # 소형주 유동성 할인
    if r.get("cap_label") == "소형":
        score = int(score * 0.85)

    return max(min(score, 100), 0)


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "options_log")


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
                },
                "forward_returns": {"1d": None, "3d": None, "5d": None},
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

def fmt_target(r):
    """옵션 목표가 포맷"""
    t = r.get("options_target")
    if not t:
        return "       -       "
    return f"${t['price']:>6.0f}({t['upside']:+.0f}%) ~{t['date'][5:]}"


def print_summary_table(results, title):
    """서머리 테이블 출력"""
    W = 150
    print()
    print("=" * W)
    print(f"  [{title}]")
    print("-" * W)
    print(f"  {'':>5} | {'현재가':>7} | {'5D':>6} | {'점수':>3}"
          f" | {'흐름':>10}"
          f" | {'P/C':>5} | {'강도':>5} | {'Skew':>6} | {'Term':>5}"
          f" | {'옵션목표 (기한)':>22}"
          f" | {'콜Vol(단/중)':>14} | {'실적':>6}")
    print("-" * W)

    for r in results:
        if "error" in r:
            print(f"  {r['ticker']:>5} | ERROR")
            continue
        pc = safe_div(r["total_put_vol"], r["total_call_vol"], 99)
        inten = r.get("opt_intensity")
        inten_s = f"{inten:>4.1%}" if inten else "  N/A"
        skew_s = f"{r['iv_skew']:>+.2f}" if r.get("iv_skew") is not None else "   N/A"
        ts_s = f"{r['term_structure']:>.2f}" if r.get("term_structure") is not None else "  N/A"
        ret = r.get("ret_5d")
        ret_s = f"{ret:>+.1%}" if ret is not None else "   N/A"
        earn_s = ""
        if r.get("next_earnings"):
            days = (r["next_earnings"] - TODAY).days
            if days <= 30:
                earn_s = f"{days}일후"
        tgt_s = fmt_target(r)
        bar = "#" * (r["score"] // 5)
        print(
            f"  {r['ticker']:>5} | ${r['price']:>6.2f} | {ret_s} |"
            f" {r['score']:>3} |"
            f" {r['flow_type']:>10} |"
            f" {pc:>5.2f} |"
            f" {inten_s} |"
            f" {skew_s} |"
            f" {ts_s} |"
            f" {tgt_s:>22} |"
            f" {r['near_call_vol']:>6,.0f}/{r['mid_call_vol']:>6,.0f} |"
            f" {earn_s:>6}"
            f"  {bar}"
        )
    print("=" * W)


def print_detail(results, title, top_n=5):
    """상위 N개 종목 상세 해석"""
    print()
    print(f"  ── {title} 상위 {top_n}개 상세 ──")

    for r in results[:top_n]:
        if "error" in r:
            continue
        ticker = r["ticker"]
        hot = r.get("hot_strikes", [])
        meaningful = [h for h in hot if not (h["otm_pct"] > 30 and h["dollar_val"] < 500_000)]
        ret = r.get("ret_5d")
        ret_s = f"{ret:+.1%}" if ret is not None else "N/A"

        print(f"\n  ┌─ [{ticker}] ${r['price']:.2f} | 점수 {r['score']}/100 | {r['flow_type']} | {r['cap_label']}주 | 5일 {ret_s}")

        # 옵션 목표가
        tgt = r.get("options_target")
        if tgt:
            print(f"  |  옵션 목표: ${tgt['price']:.0f} ({tgt['upside']:+.1f}%) ~{tgt['date']} [{tgt['basis']}, OI {tgt['oi']:,}]")

        # 주가 vs 옵션 괴리
        if ret is not None:
            pc = safe_div(r["total_put_vol"], r["total_call_vol"], 1)
            if ret < -0.03 and pc < 0.5:
                print(f"  |  ** 주가 하락 vs 콜 우위 → 역발상 강세 신호 **")
            elif ret > 0.05 and pc < 0.5:
                print(f"  |  주가 상승 + 콜 우위 → 추세 확인, 모멘텀 지속 기대")
            elif ret > 0.03 and pc > 1.0:
                print(f"  |  ** 주가 상승 vs 풋 우위 → 헤지/차익실현 진행 중 **")

        # Max Pain
        mp = r.get("near_max_pain")
        if mp:
            dev = (mp - r["price"]) / r["price"] * 100
            if dev < -10:
                mp_msg = "현재가가 MaxPain 크게 상회 → 만기 시 하방 압력 가능"
            elif dev > 10:
                mp_msg = "현재가가 MaxPain 크게 하회 → 반등 가능성"
            else:
                mp_msg = "MaxPain 근처에서 거래 중"
            print(f"  |  MaxPain: ${mp:.0f} ({dev:+.1f}%) — {mp_msg}")

        # IV Skew
        skew = r.get("iv_skew")
        if skew is not None:
            if skew < -0.05:
                skew_msg = "콜 IV > 풋 IV → 상방 수요 강함"
            elif skew > 0.05:
                skew_msg = "풋 IV > 콜 IV → 하방 헤지 수요"
            else:
                skew_msg = "중립적"
            print(f"  |  IV Skew: {skew:+.3f} ({skew_msg})")
        else:
            print(f"  |  IV Skew: N/A (OTM 구간 데이터 부족)")

        # Term Structure
        ts = r.get("term_structure")
        if ts is not None:
            if ts < 0.90:
                ts_msg = "강한 역전 → 단기 급변 기대"
            elif ts < 1.0:
                ts_msg = "소폭 역전 → 단기 변동성 높음"
            else:
                ts_msg = "정상 → 특별한 단기 기대 없음"
            print(f"  |  Term Structure: {ts:.3f} ({ts_msg})")

        # 실적
        if r.get("earn_within_30d"):
            days = (r["next_earnings"] - TODAY).days
            print(f"  |  실적 {days}일 후 (참고)")

        # 콜 OI Wall
        top_oi = r.get("top_call_oi", [])
        if top_oi:
            print(f"  |")
            print(f"  |  콜 OI Wall (감마 자석):")
            for w in top_oi:
                otm = w["otm_pct"]
                label = f"OTM +{otm:.0f}%" if otm > 3 else f"ITM {otm:+.0f}%" if otm < -3 else "ATM"
                above = "↑위" if otm > 0 else "↓아래" if otm < 0 else "=현재"
                print(f"  |    {w['date']} ${w['strike']:.0f} ({label}) — OI {w['oi']:,} {above}")

        # 이상 콜
        if meaningful:
            print(f"  |")
            print(f"  |  이상 콜 포지션:")
            for h in meaningful[:5]:
                otm_label = f"OTM +{h['otm_pct']:.0f}%" if h["otm_pct"] > 3 else \
                            f"ITM {h['otm_pct']:+.0f}%" if h["otm_pct"] < -3 else "ATM"
                dollar = h["dollar_val"]
                d_str = f"${dollar/1_000_000:.1f}M" if dollar >= 1_000_000 else f"${dollar/1_000:.0f}K"
                print(
                    f"  |    {h['period']} {h['date']} | ${h['strike']:.0f} ({otm_label})"
                    f" | Vol {h['vol']:,} vs OI {h['oi']:,}"
                    f" | Vol/OI {h['ratio']:.1f}x"
                    f" | {d_str}"
                )
        else:
            print(f"  |  이상 콜 포지션: 없음")

        print(f"  └─")


def scan_group(tickers, label):
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
    # 대장주 자신에겐 가점 없음 (중복 방지)
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

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results


# ── 메인 ──

if __name__ == "__main__":
    now = datetime.now()
    print()
    print(f"  ┌──────────────────────────────────────────────────────────┐")
    print(f"  |  GBRC 옵션 스캐너 v3 - 흑자/적자 상대 강도 비교           |")
    print(f"  |  {now.strftime('%Y-%m-%d %H:%M')}                                            |")
    print(f"  |  단기: ~{NEAR_END}  |  중기: ~{MID_END}  |  장기: ~{FAR_END}  |")
    print(f"  └──────────────────────────────────────────────────────────┘")

    # 스캔
    profit_results = scan_group(PROFIT_TICKERS, "흑자기업")
    loss_results = scan_group(LOSS_TICKERS, "적자기업")

    # 서머리 테이블
    print_summary_table(profit_results, "흑자기업 옵션 흐름")
    print_summary_table(loss_results, "적자기업 옵션 흐름")

    # 상세 (각 상위 8개)
    print_detail(profit_results, "흑자기업", top_n=8)
    print_detail(loss_results, "적자기업", top_n=8)

    # 로그 저장
    save_daily_log(profit_results, loss_results)

    # 범례
    print()
    print("  ── 스코어링 5팩터 ──")
    print("  ① P/C Ratio    < 0.5 콜 우위 | 0.5~1.0 중립 | > 1.0 풋 우위          (20점)")
    print("  ② 이상 콜 규모  달러 기준 실제 프리미엄 규모 (복권성 제외)             (20점)")
    print("  ③ 상대 옵션강도  옵션Vol / 주식AvgVol → 종목간 공정 비교의 핵심       (20점)")
    print("  ④ IV Skew      음수 = 콜 수요 > 풋 수요 (구조적 강세)                (20점)")
    print("  ⑤ Term Str.    < 0.90 역전 = 단기 이벤트 기대 | > 1.0 정상            (20점)")
    print()
    print("  ── 참고 ──")
    print("  흐름 성격   강세방향성 > 콜우위 > 헤지+콜유입 > 혼조 > 약세방향성")
    print("  MaxPain    만기 시 주가가 끌려가기 쉬운 가격 (주가 > MP → 하방 압력)")
    print("  콜 OI Wall  OI 집중 행사가 = 감마 자석, 돌파 시 딜러 헤지 매수로 급등 트리거")
    print("  5D 수익률   주가 흐름과 옵션 흐름의 괴리 여부 판단용")
    print("  소형주(<2B) 유동성 낮아 신호 노이즈 가능 → 점수 15% 할인")
    print()
