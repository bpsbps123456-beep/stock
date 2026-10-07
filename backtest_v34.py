"""v3.3 vs v3.4 옵션 점수 백테스트 — 기존 로그 in-memory 재계산"""
import json
import os
from glob import glob

LOG_DIR = "data/options_log"
TARGET_TICKERS = ["AMD", "SNDK", "MU", "JOBY", "INTC", "MSFT", "MRVL", "META",
                  "NVDA", "AVGO", "AAPL", "GOOGL", "AMZN", "ORCL", "NBIS"]


def recompute_v34(entry):
    f = entry["factors"]
    pc = f.get("pc_ratio") or 1
    hot = f.get("hot_dollar") or 0
    inten = f.get("opt_intensity")
    skew = f.get("iv_skew")
    term = f.get("term_structure")
    ret5d = f.get("ret_5d")
    flow = entry["flow_type"]

    # v3.4 데이터 검증 가드
    if skew is not None and abs(skew) > 1.0:
        skew = None
    if term is not None and (term < 0.1 or term > 5.0):
        term = None

    # 헤지성 풋 라벨 재분류 (P/C≥1.3 + 5D≥+10% + skew≥0)
    if (pc >= 1.3 and ret5d is not None and ret5d >= 0.10
            and skew is not None and skew >= 0):
        flow = "헤지성 풋"

    score = 0
    # ① P/C 콘트라리언
    if pc < 0.4:   score -= 10
    elif pc < 0.7: score += 10
    elif pc < 1.3: score += 5
    else:          score += 15

    # ② hot_dollar
    if hot > 40_000_000:    score += 25
    elif hot > 20_000_000:  score += 23
    elif hot > 5_000_000:   score += 20
    elif hot > 1_000_000:   score += 14
    elif hot > 300_000:     score += 8
    elif hot > 50_000:      score += 3

    # ③ intensity
    if inten is not None:
        if inten > 0.15:   score += 20
        elif inten > 0.08: score += 14
        elif inten > 0.04: score += 8
        elif inten > 0.02: score += 3
    else: score += 6

    # ④ skew
    if skew is not None:
        if skew < -0.10:   score += 20
        elif skew < -0.03: score += 12
        elif skew < 0.03:  score += 6
    else: score += 6

    # ⑤ term
    if term is not None:
        if term < 0.85:    score += 20
        elif term < 0.95:  score += 12
        elif term < 1.05:  score += 5
    else: score += 5

    # ⑥ flow
    flow_pts = {"강세 방향성": 10, "헤지성 풋": 5, "헤지+콜 유입": 10,
                "이벤트 변동성": -3, "혼조": -5, "약세 방향성": -15,
                "콜 우위": 0, "복권성 콜": 0}
    score += flow_pts.get(flow, 0)

    # ⑦ 5D
    if ret5d is not None:
        pct = ret5d * 100
        if pct >= 50:   score -= 15
        elif pct >= 30: score -= 5
        elif pct >= 20: score -= 3
        elif pct >= 15: score += 3
        elif pct >= 10: score += 5
        elif pct >= 5:  score += 3
        elif pct >= -5: score += 5
        elif pct < -10: score -= 5

    # streak / ETF adjustment 보존
    if entry.get("hot_dollar_streak"):
        score += 10
    score += entry.get("etf_adjustment", 0) or 0

    if entry.get("cap_label") == "소형":
        score = int(score * 0.85)

    return max(min(score, 100), 0), flow


def main():
    files = sorted(glob(os.path.join(LOG_DIR, "2026-*.json")))
    by_ticker = {t: [] for t in TARGET_TICKERS}

    for fp in files:
        date = os.path.basename(fp).replace(".json", "")
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("rankings", []):
            if r["ticker"] in TARGET_TICKERS:
                v33 = r["score"]
                v34, new_flow = recompute_v34(r)
                by_ticker[r["ticker"]].append({
                    "date": date,
                    "v33": v33,
                    "v34": v34,
                    "delta": v34 - v33,
                    "flow_v33": r["flow_type"],
                    "flow_v34": new_flow,
                    "pc": r["factors"].get("pc_ratio"),
                    "ret5d": r["factors"].get("ret_5d"),
                    "hot$": r["factors"].get("hot_dollar"),
                })

    print(f"{'='*100}")
    print(f"{'Ticker':<7} {'Date':<12} {'v3.3':>5} {'v3.4':>5} {'Δ':>5}  {'P/C':>5} {'5D%':>6} {'hot$M':>7}  flow_v3.3 → v3.4")
    print("=" * 100)
    for ticker in TARGET_TICKERS:
        rows = by_ticker[ticker]
        if not rows:
            continue
        for row in rows:
            pc_s = f"{row['pc']:.2f}" if row['pc'] is not None else "N/A"
            ret_s = f"{row['ret5d']*100:+.1f}" if row['ret5d'] is not None else "N/A"
            hot_s = f"{row['hot$']/1e6:.1f}" if row['hot$'] is not None else "N/A"
            flow_change = (f"{row['flow_v33']} → {row['flow_v34']}"
                           if row['flow_v33'] != row['flow_v34']
                           else row['flow_v33'])
            print(f"{ticker:<7} {row['date']:<12} {row['v33']:>5} {row['v34']:>5} {row['delta']:>+5}  "
                  f"{pc_s:>5} {ret_s:>6} {hot_s:>7}  {flow_change}")
        # 평균
        avg33 = sum(r['v33'] for r in rows) / len(rows)
        avg34 = sum(r['v34'] for r in rows) / len(rows)
        print(f"{ticker:<7} {'AVG':<12} {avg33:>5.1f} {avg34:>5.1f} {avg34-avg33:>+5.1f}")
        print("-" * 100)


if __name__ == "__main__":
    main()
