"""
GBRC 옵션 트래커 - 예측 vs 실제 검증 시스템

기능:
  1. 과거 로그에 실제 수익률(1d/3d/5d) 채우기
  2. 스코어 상위 vs 하위 비교 (아웃퍼폼 검증)
  3. 팩터별 상관관계 분석
  4. 누적 성적표 출력

사용법:
  py options_tracker.py           # 전체 리포트
  py options_tracker.py --fill    # 수익률 채우기만
"""

import json
import os
import sys
from datetime import datetime, timedelta
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "options_log")
TODAY = datetime.now().date()


def get_log_files():
    """날짜순 로그 파일 목록"""
    if not os.path.exists(LOG_DIR):
        return []
    files = [f for f in os.listdir(LOG_DIR) if f.endswith(".json")]
    files.sort()
    return files


def load_log(filename):
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_log(filename, data):
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_return(ticker, base_date_str, days_forward):
    """base_date 종가 대비 N 거래일 후 종가 수익률"""
    try:
        base_date = datetime.strptime(base_date_str, "%Y-%m-%d").date()
        # 충분한 범위로 히스토리 가져오기
        start = base_date - timedelta(days=1)
        end = base_date + timedelta(days=days_forward + 5)  # 주말/공휴일 여유

        if end > TODAY:
            return None  # 아직 미래

        t = yf.Ticker(ticker)
        h = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if h.empty or len(h) < 2:
            return None

        # base_date 종가 찾기
        base_prices = h[h.index.date == base_date]
        if base_prices.empty:
            # base_date가 휴장이면 직전 거래일
            before = h[h.index.date <= base_date]
            if before.empty:
                return None
            base_price = before["Close"].iloc[-1]
        else:
            base_price = base_prices["Close"].iloc[-1]

        # N거래일 후 종가
        after = h[h.index.date > base_date]
        if len(after) < days_forward:
            return None

        future_price = after["Close"].iloc[min(days_forward - 1, len(after) - 1)]
        return (future_price / base_price) - 1

    except Exception:
        return None


def fill_returns():
    """모든 로그의 빈 forward_returns를 실제 데이터로 채움"""
    files = get_log_files()
    filled_count = 0

    for fname in files:
        log = load_log(fname)
        date_str = log["date"]
        modified = False

        for entry in log.get("rankings", []):
            fr = entry.get("forward_returns", {})
            ticker = entry["ticker"]

            for period, days in [("1d", 1), ("3d", 3), ("5d", 5)]:
                if fr.get(period) is None:
                    ret = get_return(ticker, date_str, days)
                    if ret is not None:
                        fr[period] = round(ret, 5)
                        modified = True
                        filled_count += 1

            entry["forward_returns"] = fr

        if modified:
            save_log(fname, log)
            print(f"  {fname}: 수익률 업데이트")

    return filled_count


def generate_report():
    """누적 성적표 생성"""
    files = get_log_files()
    if not files:
        print("  로그 파일이 없습니다. 옵션 스캐너를 먼저 실행하세요.")
        return

    print()
    print(f"  ┌──────────────────────────────────────────────┐")
    print(f"  |  GBRC 옵션 트래커 - 누적 성적표               |")
    print(f"  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}                              |")
    print(f"  |  로그: {len(files)}일분 ({files[0][:10]} ~ {files[-1][:10]})  |")
    print(f"  └──────────────────────────────────────────────┘")

    # 데이터 수집
    all_entries = []
    for fname in files:
        log = load_log(fname)
        date_str = log["date"]
        for entry in log.get("rankings", []):
            entry["log_date"] = date_str
            all_entries.append(entry)

    # 수익률이 채워진 것만
    has_1d = [e for e in all_entries if e.get("forward_returns", {}).get("1d") is not None]
    has_3d = [e for e in all_entries if e.get("forward_returns", {}).get("3d") is not None]
    has_5d = [e for e in all_entries if e.get("forward_returns", {}).get("5d") is not None]

    print(f"\n  데이터: 총 {len(all_entries)}건 | 1D 수익률 {len(has_1d)}건 | 3D {len(has_3d)}건 | 5D {len(has_5d)}건")

    if not has_1d:
        print("\n  아직 수익률 데이터가 없습니다. 내일 이후 --fill 옵션으로 실행하세요.")
        print("  (최소 1거래일 경과 필요)")
        return

    # ── 1. 상위 vs 하위 비교 ──
    print(f"\n  ══ 1. 스코어 상위 vs 하위 비교 ══")

    for period, entries in [("1D", has_1d), ("3D", has_3d), ("5D", has_5d)]:
        if not entries:
            continue

        period_key = period.lower()

        # 각 날짜별로 상위 3 / 하위 3 분리
        by_date = {}
        for e in entries:
            d = e["log_date"]
            if d not in by_date:
                by_date[d] = []
            by_date[d].append(e)

        top_returns = []
        bottom_returns = []
        for d, day_entries in by_date.items():
            day_entries.sort(key=lambda x: x.get("score", 0), reverse=True)
            top3 = day_entries[:3]
            bot3 = day_entries[-3:]
            for e in top3:
                r = e.get("forward_returns", {}).get(period_key)
                if r is not None:
                    top_returns.append(r)
            for e in bot3:
                r = e.get("forward_returns", {}).get(period_key)
                if r is not None:
                    bottom_returns.append(r)

        if top_returns and bottom_returns:
            top_avg = sum(top_returns) / len(top_returns)
            bot_avg = sum(bottom_returns) / len(bottom_returns)
            diff = top_avg - bot_avg
            win = "상위 아웃퍼폼" if diff > 0 else "하위 아웃퍼폼"
            emoji = "O" if diff > 0 else "X"
            print(f"\n    [{period}] 상위3 평균: {top_avg:+.2%} | 하위3 평균: {bot_avg:+.2%}"
                  f" | 차이: {diff:+.2%} [{emoji} {win}]"
                  f" (표본: 상위 {len(top_returns)}건, 하위 {len(bottom_returns)}건)")

    # ── 2. 팩터별 상관관계 ──
    print(f"\n  ══ 2. 팩터별 유효성 (5D 수익률 기준) ══")

    if len(has_5d) >= 10:
        factor_names = ["pc_ratio", "hot_dollar", "opt_intensity", "iv_skew", "term_structure"]
        display_names = ["P/C Ratio", "이상콜규모", "상대옵션강도", "IV Skew", "Term Str."]

        for fname, dname in zip(factor_names, display_names):
            pairs = []
            for e in has_5d:
                fval = e.get("factors", {}).get(fname)
                ret = e.get("forward_returns", {}).get("5d")
                if fval is not None and ret is not None:
                    pairs.append((fval, ret))

            if len(pairs) < 5:
                print(f"    {dname:>12}: 데이터 부족 ({len(pairs)}건)")
                continue

            # 단순 상관계수
            n = len(pairs)
            x = [p[0] for p in pairs]
            y = [p[1] for p in pairs]
            mx, my = sum(x)/n, sum(y)/n
            cov = sum((xi-mx)*(yi-my) for xi, yi in pairs) / n
            sx = (sum((xi-mx)**2 for xi in x) / n) ** 0.5
            sy = (sum((yi-my)**2 for yi in y) / n) ** 0.5
            corr = cov / (sx * sy) if sx > 0 and sy > 0 else 0

            # 방향 해석 (P/C ratio와 term structure는 낮을수록 강세이므로 음의 상관이 좋음)
            if fname in ["pc_ratio", "term_structure"]:
                effective = "유효" if corr < -0.1 else "약함" if abs(corr) <= 0.1 else "역행"
            elif fname == "iv_skew":
                effective = "유효" if corr < -0.1 else "약함" if abs(corr) <= 0.1 else "역행"
            else:
                effective = "유효" if corr > 0.1 else "약함" if abs(corr) <= 0.1 else "역행"

            bar_len = int(abs(corr) * 20)
            bar = "#" * bar_len
            sign = "+" if corr > 0 else "-"
            print(f"    {dname:>12}: r={corr:+.3f} [{effective}] {sign}{bar} (n={n})")

        print(f"\n    해석: '유효' = 팩터가 수익률 예측에 기여")
        print(f"          '역행' = 팩터가 반대로 작동 → 가중치 재검토 필요")
        print(f"          '약함' = 상관 약함 → 더 많은 데이터 필요")
    else:
        print(f"    5D 수익률 데이터 {len(has_5d)}건 — 최소 10건 이상 필요")
        days_needed = max(0, 10 - len(has_5d))
        print(f"    약 {days_needed}일 후부터 팩터 분석 가능")

    # ── 3. 종합 스코어 유효성 ──
    print(f"\n  ══ 3. 종합 스코어 유효성 ══")

    for period, entries in [("1D", has_1d), ("3D", has_3d), ("5D", has_5d)]:
        if len(entries) < 5:
            continue
        period_key = period.lower()
        pairs = [(e["score"], e["forward_returns"][period_key]) for e in entries
                 if e["forward_returns"].get(period_key) is not None]
        if len(pairs) < 5:
            continue

        n = len(pairs)
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        mx, my = sum(x)/n, sum(y)/n
        cov = sum((xi-mx)*(yi-my) for xi, yi in pairs) / n
        sx = (sum((xi-mx)**2 for xi in x) / n) ** 0.5
        sy = (sum((yi-my)**2 for yi in y) / n) ** 0.5
        corr = cov / (sx * sy) if sx > 0 and sy > 0 else 0

        verdict = "유효" if corr > 0.1 else "약함" if abs(corr) <= 0.1 else "역행"
        print(f"    [{period}] 스코어↔수익률 상관: r={corr:+.3f} [{verdict}] (n={n})")

    # ── 4. 일별 기록 요약 ──
    print(f"\n  ══ 4. 일별 기록 ══")
    print(f"    {'날짜':>12} | {'1위':>6} {'점수':>4} {'1D':>7} {'3D':>7} {'5D':>7}"
          f" | {'10위':>6} {'점수':>4} {'1D':>7} {'3D':>7} {'5D':>7}")
    print(f"    {'-'*90}")

    for fname in files:
        log = load_log(fname)
        rankings = log.get("rankings", [])
        if not rankings:
            continue
        top = rankings[0]
        bot = rankings[-1]

        def fmt_r(entry, period):
            v = entry.get("forward_returns", {}).get(period)
            return f"{v:+.1%}" if v is not None else "    -"

        print(f"    {log['date']:>12}"
              f" | {top['ticker']:>6} {top['score']:>4}"
              f" {fmt_r(top, '1d'):>7} {fmt_r(top, '3d'):>7} {fmt_r(top, '5d'):>7}"
              f" | {bot['ticker']:>6} {bot['score']:>4}"
              f" {fmt_r(bot, '1d'):>7} {fmt_r(bot, '3d'):>7} {fmt_r(bot, '5d'):>7}")

    # ── 5. 충분한 데이터 후 가중치 제안 ──
    if len(has_5d) >= 30:
        print(f"\n  ══ 5. 가중치 조정 제안 ══")
        print(f"    30일+ 데이터 축적 완료. 팩터별 상관계수 기반 제안:")

        factor_names = ["pc_ratio", "hot_dollar", "opt_intensity", "iv_skew", "term_structure"]
        display_names = ["P/C Ratio", "이상콜규모", "상대옵션강도", "IV Skew", "Term Str."]
        correlations = []

        for fname in factor_names:
            pairs = []
            for e in has_5d:
                fval = e.get("factors", {}).get(fname)
                ret = e.get("forward_returns", {}).get("5d")
                if fval is not None and ret is not None:
                    pairs.append((fval, ret))
            if len(pairs) >= 10:
                n = len(pairs)
                x = [p[0] for p in pairs]
                y = [p[1] for p in pairs]
                mx, my = sum(x)/n, sum(y)/n
                cov = sum((xi-mx)*(yi-my) for xi, yi in pairs) / n
                sx = (sum((xi-mx)**2 for xi in x) / n) ** 0.5
                sy = (sum((yi-my)**2 for yi in y) / n) ** 0.5
                corr = cov / (sx * sy) if sx > 0 and sy > 0 else 0
                # 방향 보정: P/C, skew, term은 낮을수록 강세
                if fname in ["pc_ratio", "iv_skew", "term_structure"]:
                    corr = -corr
                correlations.append(abs(corr))
            else:
                correlations.append(0)

        total_corr = sum(correlations)
        if total_corr > 0:
            suggested = [int(round(c / total_corr * 100)) for c in correlations]
            # 반올림 보정
            diff = 100 - sum(suggested)
            if diff != 0:
                max_idx = correlations.index(max(correlations))
                suggested[max_idx] += diff

            print(f"\n    현재 가중치 vs 데이터 기반 제안:")
            print(f"    {'팩터':>12} | {'현재':>6} | {'제안':>6} | {'상관계수':>8}")
            for dname, curr_w, sug_w, corr in zip(display_names, [20]*5, suggested, correlations):
                change = sug_w - curr_w
                arrow = "↑" if change > 3 else "↓" if change < -3 else "="
                print(f"    {dname:>12} |   {curr_w:>3}점 |   {sug_w:>3}점 {arrow} | r={corr:.3f}")

            print(f"\n    ⚠ 이 제안은 참고용입니다. 자동 적용하지 않습니다.")
            print(f"    적용하려면 직접 options_scanner.py의 calc_score() 가중치를 조정하세요.")
    elif len(has_5d) >= 10:
        remaining = 30 - len(has_5d)
        print(f"\n  ══ 5. 가중치 조정 ══")
        print(f"    현재 {len(has_5d)}일 — {remaining}일 더 쌓이면 가중치 조정 제안이 활성화됩니다.")
        print(f"    (과적합 방지를 위해 최소 30일 데이터 요구)")

    print()


# ───────────────────────────────────────────────
# 돌발 호재 감지 — 분석 실패 vs 불가항력 구분
# ───────────────────────────────────────────────

# 구체적 이벤트 키워드 (계약, M&A, FDA, 대형 주문 등)
CATALYST_KEYWORDS = [
    # 계약/파트너십 — 상대방 명시된 건만
    "signs deal", "signs agreement", "partners with", "partnership with",
    "wins contract", "awarded contract", "selected by", "secures contract",
    "inks deal", "joint venture",
    # M&A
    "acquires", "acquisition of", "to acquire", "buys",
    "merger with", "takeover", "buyout",
    # FDA / 규제 승인
    "fda approves", "fda clears", "fda approval", "fda clearance",
    "receives approval", "cleared by",
    # 대형 주문 / 자금 조달
    "secures $", "raises $", "announces order", "receives order",
    "billion contract", "million contract", "billion deal", "million deal",
    # 실적 서프라이즈
    "beats estimates", "raises guidance", "preannounces",
    "earnings beat", "q1 beat", "q2 beat", "q3 beat", "q4 beat",
]

# 제외 키워드 — 구체적 사건 아닌 것
EXCLUDE_KEYWORDS = [
    "rally", "surge in", "jump on", "rises as", "stocks climb",
    "analyst", "target raised", "target price", "rating", "upgraded by",
    "could", "might", "may ", "predict", "forecast",
    "5 stocks", "3 stocks", "top stocks", "best stocks",
]


def detect_breaking_catalyst(ticker, window_hours=36):
    """
    해당 종목의 최근 돌발 호재 감지.
    window_hours 이내의 뉴스 중 구체적 이벤트만 리턴.
    """
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return []

        cutoff = datetime.now().timestamp() - window_hours * 3600
        catalysts = []
        seen = set()

        for item in news[:15]:
            content = item.get("content", item) if isinstance(item.get("content"), dict) else item
            title = content.get("title", "") if isinstance(content, dict) else ""
            if not title or title in seen:
                continue
            seen.add(title)

            # 발행 시간 체크
            pub = content.get("pubDate") or item.get("providerPublishTime")
            if isinstance(pub, str):
                try:
                    pub_ts = datetime.fromisoformat(pub.replace("Z", "+00:00")).timestamp()
                except Exception:
                    pub_ts = None
            elif isinstance(pub, (int, float)):
                pub_ts = pub
            else:
                pub_ts = None

            if pub_ts is not None and pub_ts < cutoff:
                continue  # 너무 오래된 뉴스

            title_lower = title.lower()

            # 제외 키워드 먼저 체크
            if any(ex in title_lower for ex in EXCLUDE_KEYWORDS):
                continue

            # 이벤트 키워드 매칭
            matched = [kw for kw in CATALYST_KEYWORDS if kw in title_lower]
            if not matched:
                continue

            # 티커 자체가 언급되거나 상대방으로 보이는 고유명사가 있어야 진짜 이벤트
            # 간단 휴리스틱: 대문자로 시작하는 단어가 3개 이상
            cap_words = [w for w in title.split() if w and w[0].isupper() and len(w) > 2]
            if len(cap_words) < 2:
                continue

            provider = content.get("provider", {}) if isinstance(content, dict) else {}
            source = provider.get("displayName", "") if isinstance(provider, dict) else ""

            catalysts.append({
                "title": title,
                "source": source,
                "matched_keywords": matched[:2],
            })

        return catalysts[:3]
    except Exception:
        return []


def classify_miss(ticker, predicted_score, actual_return, threshold_pct=3.0):
    """
    예측 실패 케이스를 분류:
      - "catalyst": 돌발 호재로 상승 (분석 실패 아님)
      - "analysis_miss": 돌발 호재 없는데 예측 빗나감 (진짜 실패)
      - None: 실패 아님
    """
    if actual_return is None:
        return None
    actual_pct = actual_return * 100

    # 낮은 점수인데 크게 상승 = 놓친 기회
    if predicted_score < 40 and actual_pct > threshold_pct:
        catalysts = detect_breaking_catalyst(ticker)
        if catalysts:
            return {"type": "catalyst", "catalysts": catalysts, "actual": actual_pct}
        return {"type": "analysis_miss", "actual": actual_pct}

    return None


def report_catalyst_analysis():
    """최신 로그의 misses를 돌발 호재 여부로 분류"""
    files = get_log_files()
    if len(files) < 2:
        print("  로그 2개 이상 필요 (실제 수익률 비교를 위해)")
        return

    # 가장 오래된 로그와 그 다음날 로그 가격 비교
    prev_log = load_log(files[-2])
    curr_log = load_log(files[-1])
    curr_prices = {r["ticker"]: r.get("price", 0) for r in curr_log.get("rankings", [])}

    print()
    print("=" * 100)
    print(f"  돌발 호재 분석 — {prev_log['date']} 점수 vs {curr_log['date']} 실제")
    print("=" * 100)

    genuine_misses = []
    catalyst_misses = []

    for entry in prev_log.get("rankings", []):
        ticker = entry["ticker"]
        score = entry.get("score", 0)
        prev_price = entry.get("price", 0)
        curr_price = curr_prices.get(ticker, 0)
        if not prev_price or not curr_price:
            continue
        ret = (curr_price - prev_price) / prev_price

        result = classify_miss(ticker, score, ret)
        if result is None:
            continue

        if result["type"] == "catalyst":
            catalyst_misses.append((ticker, score, result))
        else:
            genuine_misses.append((ticker, score, result))

    print(f"\n  [돌발 호재로 상승한 종목 — 분석 실패 아님]")
    if catalyst_misses:
        for ticker, score, r in catalyst_misses:
            print(f"    {ticker:>5} | 점수 {score:>3} | 실제 +{r['actual']:>5.2f}%")
            for c in r["catalysts"]:
                print(f"         └ [{c['source']}] {c['title'][:80]}")
                print(f"             ↳ 매칭: {', '.join(c['matched_keywords'])}")
    else:
        print(f"    (해당 없음)")

    print(f"\n  [분석이 놓친 상승 — 진짜 보완 필요]")
    if genuine_misses:
        for ticker, score, r in genuine_misses:
            print(f"    {ticker:>5} | 점수 {score:>3} | 실제 +{r['actual']:>5.2f}% | 돌발 호재 없음")
    else:
        print(f"    (해당 없음)")

    print()
    print("  ※ 돌발 호재 = 오늘/어제 발생한 구체적 이벤트(계약·M&A·FDA·대형 주문)")
    print("     섹터 랠리, 애널리스트 의견, 추측성 기사는 제외")
    print("=" * 100)


if __name__ == "__main__":
    mode = "--fill" if "--fill" in sys.argv else "--report"

    if "--catalyst" in sys.argv:
        report_catalyst_analysis()
        sys.exit(0)

    if mode == "--fill" or "--report" in sys.argv or len(sys.argv) == 1:
        print("\n  수익률 데이터 채우는 중...")
        filled = fill_returns()
        print(f"  완료: {filled}건 업데이트\n")

    if mode != "--fill":
        generate_report()
