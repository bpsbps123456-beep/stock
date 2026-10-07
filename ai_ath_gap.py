"""AI 밸류체인 종목을 전고점과의 거리로 네 구간에 나눈다 (Path-Q).

유니버스 정본은 watchlist.AI_UNIVERSE다. 가격은 yfinance 일봉(분할 조정, 배당 미조정)이며
전고점은 장중 고가 기준이라 트레이딩뷰 기본 차트의 All Time High와 같은 기준이다.
TradingView MCP 값은 이 스크립트에 넣지 않는다(AGENTS.md). 대조는 대화 중에 한다.

사용법:
  py ai_ath_gap.py                    # 최신 봉 기준 → data/derived/analysis/ai_ath_gap/
  py ai_ath_gap.py --asof 2026-10-02  # 해당일까지의 봉만으로 다시 계산
"""
import argparse
import csv
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import yfinance as yf

from watchlist import AI_UNIVERSE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "derived", "analysis", "ai_ath_gap")

# ── 분류 기준 (사용자 조정 영역) ──────────────────────────────────────
NEAR_PCT, MID_PCT = -10.0, -20.0  # ② 10% 이내, ③ 20% 이내의 경계
# 최근 이 거래일 안에 신고가를 새로 썼으면 '이미 전고점을 넘은' 종목(①)으로 본다.
BREAKOUT_DAYS = 21
# 닷컴(2000)·일본 버블(1989) 고점은 이번 사이클의 비교 대상이 아니다. 역대 고점이
# 이보다 오래됐으면 이 날 이후 최고가를 전고점으로 쓰고 역대 고점은 비고에 남긴다.
LEGACY_CUTOFF = pd.Timestamp("2010-01-01")
YOUNG_DAYS = 504  # 상장 2년(거래일) 미만이면 비고에 표시
# ────────────────────────────────────────────────────────────────

BUCKETS = ["① 전고점 돌파", "② 10% 이내", "③ 20% 이내", "④ 그 외"]
BENCHMARKS = ["SOXX", "QQQ"]
MARKETS = {".KS": "한국", ".KQ": "한국", ".T": "일본", ".TW": "대만", ".TWO": "대만",
           ".SZ": "중국", ".HK": "홍콩", ".VI": "오스트리아", ".DE": "독일", ".PA": "프랑스"}

NAMES = {
    "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "042700.KS": "한미반도체",
    "003160.KS": "디아이", "009150.KS": "삼성전기", "353200.KS": "대덕전자",
    "007810.KS": "코리아써키트", "011070.KS": "LG이노텍", "001820.KS": "삼화콘덴서",
    "005070.KS": "코스모AM&T(구 코스모신소재)", "007660.KS": "이수페타시스",
    "298040.KS": "효성중공업", "010120.KS": "LS ELECTRIC", "267260.KS": "HD현대일렉트릭",
    "034020.KS": "두산에너빌리티", "084370.KQ": "유진테크", "089030.KQ": "테크윙",
    "232140.KQ": "와이씨", "086390.KQ": "유니테스트", "222800.KQ": "심텍",
    "095340.KQ": "ISC", "200710.KQ": "에이디테크놀로지", "399720.KQ": "가온칩스",
    "058470.KQ": "리노공업", "440110.KQ": "파두", "240810.KQ": "원익IPS",
    "036930.KQ": "주성엔지니어링", "095610.KQ": "테스", "357780.KQ": "솔브레인",
    "092870.KQ": "엑시콘", "252990.KQ": "샘씨엔에스", "036540.KQ": "SFA반도체",
    "052710.KQ": "아모텍", "149950.KQ": "아바텍", "078600.KQ": "대주전자재료",
    "320000.KQ": "한울반도체", "306620.KQ": "GIS(구 네온테크)",
    "039030.KQ": "이오테크닉스", "000990.KS": "DB하이텍", "403870.KQ": "HPSP",
    "319660.KQ": "피에스케이", "031980.KQ": "피에스케이홀딩스", "067310.KQ": "하나마이크론",
    "064760.KQ": "티씨케이", "131290.KQ": "티에스이", "080220.KQ": "제주반도체",
    "005290.KQ": "동진쎄미켐", "131970.KQ": "두산테스나", "281820.KS": "케이씨텍",
    "089970.KQ": "브이엠", "183300.KQ": "코미코", "420770.KQ": "기가비스",
    "166090.KQ": "하나머티리얼즈", "356860.KQ": "티엘비", "101490.KQ": "에스앤에스텍",
    "195870.KS": "해성디에스", "083450.KQ": "GST", "388210.KQ": "씨엠티엑스",
    "425420.KQ": "티에프이", "074600.KQ": "원익QnC", "059090.KQ": "미코",
    "036810.KQ": "에프에스티", "053610.KQ": "프로텍", "064290.KQ": "인텍플러스",
    "033640.KQ": "네패스", "122640.KQ": "예스티", "490470.KQ": "세미파이브",
    "102710.KQ": "이엔에프테크놀로지", "253590.KQ": "네오셈", "077360.KQ": "덕산하이메탈",
    "080580.KQ": "오킨스전자", "045100.KQ": "한양이엔지", "089890.KQ": "코세스",
    "219130.KQ": "타이거일렉", "144960.KQ": "뉴파워프라즈마", "094170.KQ": "동운아나텍",
    "078350.KQ": "한양디지텍", "101160.KQ": "월덱스", "098120.KQ": "마이크로컨텍솔",
    "014680.KS": "한솔케미칼", "281740.KQ": "레이크머티리얼즈", "104830.KQ": "원익머트리얼즈",
    "140860.KQ": "파크시스템스",
    "6594.T": "니덱", "6762.T": "TDK", "7741.T": "HOYA", "4004.T": "레조낙",
    "2802.T": "아지노모토", "4062.T": "이비덴", "285A.T": "키옥시아",
    "8035.T": "도쿄일렉트론", "6981.T": "무라타", "6976.T": "태양유전", "5803.T": "후지쿠라",
    "2408.TW": "난야테크", "2344.TW": "윈본드", "3037.TW": "유니마이크론",
    "8046.TW": "난야PCB", "3189.TW": "킨서스", "2327.TW": "야게오", "2492.TW": "화신과기",
    "8299.TWO": "파이슨", "ATS.VI": "AT&S", "300308.SZ": "이노라이트",
    "300502.SZ": "이오프토링크", "2513.HK": "Z.ai(즈푸)", "0100.HK": "미니맥스",
    "2317.TW": "폭스콘(훙하이)", "2382.TW": "콴타", "2308.TW": "델타전자", "2454.TW": "미디어텍",
    "3661.TW": "알칩", "6857.T": "어드반테스트", "3110.T": "닛토보", "6501.T": "히타치",
    "7011.T": "미쓰비시중공업", "ENR.DE": "지멘스에너지", "SU.PA": "슈나이더 일렉트릭",
}


def universe_rows():
    rows = [(group, sub, sym) for group, subs in AI_UNIVERSE.items()
            for sub, syms in subs.items() for sym in syms]
    dupes = [s for s, n in Counter(r[2] for r in rows).items() if n > 1]
    if dupes:
        raise SystemExit(f"AI_UNIVERSE에 중복 종목: {dupes}")
    return rows


def market_of(symbol):
    for suffix, market in MARKETS.items():
        if symbol.endswith(suffix):
            return market
    return "미국"


def fetch(symbol, asof):
    bars = yf.Ticker(symbol).history(period="max", auto_adjust=False)
    if bars.empty:
        return symbol, None
    # 거래소 현지 날짜를 유지한다. UTC로 바꾸면 아시아 봉이 전날로 밀린다.
    bars.index = bars.index.tz_localize(None).normalize()
    bars = bars[~bars.index.duplicated(keep="last")]
    bars = bars[(bars["High"] > 0) & bars["Close"].notna()]
    if asof is not None:
        bars = bars[bars.index <= asof]
    return symbol, (bars if len(bars) else None)


def measure(bars):
    last_at = bars.index[-1]
    close = float(bars["Close"].iloc[-1])
    ath_at = bars["High"].idxmax()
    ref = bars[bars.index >= LEGACY_CUTOFF] if ath_at < LEGACY_CUTOFF else bars
    peak_at, close_peak_at = ref["High"].idxmax(), ref["Close"].idxmax()
    peak = float(ref["High"].max())
    year = bars[bars.index > last_at - pd.Timedelta(days=365)]
    return {
        "date": last_at.date(),
        "close": close,
        "peak": peak,
        "peak_date": peak_at.date(),
        "days_since_peak": len(bars) - 1 - bars.index.get_loc(peak_at),
        "gap_pct": (close / peak - 1) * 100,
        # 장중 꼬리 하나가 전고점이 되면 구간이 바뀐다. 종가 기준은 비고에서만 쓴다.
        "gap_close_pct": (close / float(ref["Close"].max()) - 1) * 100,
        "days_since_close_peak": len(bars) - 1 - bars.index.get_loc(close_peak_at),
        "gap_52w_pct": (close / float(year["High"].max()) - 1) * 100,
        "ath": float(bars["High"].max()),
        "ath_date": ath_at.date(),
        "gap_ath_pct": (close / float(bars["High"].max()) - 1) * 100,
        "bars": len(bars),
    }


def band_of(gap_pct, days_since_peak):
    if gap_pct >= NEAR_PCT:
        return BUCKETS[0] if days_since_peak <= BREAKOUT_DAYS else BUCKETS[1]
    return BUCKETS[2] if gap_pct >= MID_PCT else BUCKETS[3]


def notes_of(m):
    notes = []
    if m["ath_date"] < LEGACY_CUTOFF.date():
        notes.append(f"⚑ {m['ath_date'].year}년 역대 고점 대비 {m['gap_ath_pct']:+.0f}%")
    if m["bars"] < YOUNG_DAYS:
        notes.append(f"상장 {m['bars']}거래일")
    if m["days_since_peak"] <= BREAKOUT_DAYS and m["gap_pct"] < NEAR_PCT:
        notes.append("신고가 직후 급락")
    close_band = band_of(m["gap_close_pct"], m["days_since_close_peak"])
    if close_band != m["bucket"]:
        notes.append(f"종가 기준 {m['gap_close_pct']:+.0f}% → {close_band[0]}")
    return notes


def breakout_window_range(results):
    """① 판정이 그대로인 BREAKOUT_DAYS 범위. 경계 민감도를 숫자로 남긴다."""
    near = [r["days_since_peak"] for r in results if r["gap_pct"] >= NEAR_PCT]
    inside = [d for d in near if d <= BREAKOUT_DAYS]
    outside = [d for d in near if d > BREAKOUT_DAYS]
    return max(inside, default=0), min(outside, default=10**6) - 1


def fmt_price(x):
    return f"{x:,.0f}" if x >= 1000 else f"{x:,.2f}"


def build_report(results, missing, bench, run_label):
    lines = [f"# AI 밸류체인 전고점 거리 분류 — {run_label}", ""]
    basis = {}
    for r in results:
        basis.setdefault(market_of(r["symbol"]), Counter())[r["date"]] += 1
    # 실행 당일 봉은 장중이면 지연 시세다. 장 마감 뒤 다시 돌리면 종가로 바뀐다.
    basis_txt = ", ".join(
        f"{mk} {d:%m/%d}{'(실행 시점)' if d == date.today() else ''}"
        for mk, d in ((mk, c.most_common(1)[0][0]) for mk, c in basis.items()))
    lines += [
        f"- 유니버스 {len(results) + len(missing)}종목(정본 `watchlist.AI_UNIVERSE`), "
        f"계산 {len(results)} · 데이터 없음 {len(missing)}{' ' + str(missing) if missing else ''}",
        f"- 가격: 시장별 마지막 봉 종가 — {basis_txt}",
        "- 전고점: 장중 고가 기준 역대 최고가(분할 조정·배당 미조정). "
        f"역대 고점이 {LEGACY_CUTOFF:%Y}년 이전이면 그 뒤 최고가로 대체(⚑)",
        f"- 구간: ① 최근 {BREAKOUT_DAYS}거래일 안에 신고가 + 그 고점 {NEAR_PCT:.0f}% 이내 / "
        f"② {NEAR_PCT:.0f}% 이내 / ③ {MID_PCT:.0f}% 이내 / ④ 그 외 "
        "(① 결과는 신고가 경과일 기준을 {}~{}거래일 중 무엇으로 해도 같음)".format(
            *breakout_window_range(results)),
        "- 참고 지수: " + ", ".join(
            f"{b} {m['gap_pct']:+.1f}% (전고점 {m['peak_date']})" for b, m in bench.items()),
        "",
        "## 구간별 개수",
        "",
        "| 구간 | 종목 수 |",
        "|---|---:|",
    ]
    counts = Counter(r["bucket"] for r in results)
    lines += [f"| {b} | {counts.get(b, 0)} |" for b in BUCKETS]

    lines += ["", "## 층별 지도 (괴리율 %)", "",
              "| 대분류 | 소분류 | " + " | ".join(BUCKETS) + " |",
              "|---|---|" + "---|" * len(BUCKETS)]
    by_sub = {}
    for r in results:
        by_sub.setdefault((r["group"], r["sub"]), []).append(r)
    for group, subs in AI_UNIVERSE.items():
        for sub in subs:
            members = sorted(by_sub.get((group, sub), []), key=lambda r: -r["gap_pct"])
            cells = [", ".join(f"{r['name']} {r['gap_pct']:+.0f}" for r in members
                               if r["bucket"] == b) for b in BUCKETS]
            lines.append(f"| {group} | {sub} | " + " | ".join(cells) + " |")

    for b in BUCKETS:
        members = sorted((r for r in results if r["bucket"] == b), key=lambda r: -r["gap_pct"])
        lines += ["", f"## {b} ({len(members)})", "",
                  "| 종목 | 층 | 현재가 | 전고점 | 전고점일 | 괴리 | 52주 고점 대비 | 비고 |",
                  "|---|---|---:|---:|---|---:|---:|---|"]
        for r in members:
            lines.append(
                f"| {r['name']} | {r['sub']} | {fmt_price(r['close'])} | {fmt_price(r['peak'])} "
                f"| {r['peak_date']} | {r['gap_pct']:+.1f}% | {r['gap_52w_pct']:+.1f}% "
                f"| {'; '.join(r['notes'])} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--asof", help="YYYY-MM-DD까지의 봉만 사용")
    args = parser.parse_args()
    asof = pd.Timestamp(args.asof) if args.asof else None

    rows = universe_rows()
    symbols = [r[2] for r in rows] + BENCHMARKS
    with ThreadPoolExecutor(max_workers=8) as pool:
        fetched = dict(pool.map(lambda s: fetch(s, asof), symbols))

    results, missing = [], []
    for group, sub, sym in rows:
        bars = fetched.get(sym)
        if bars is None:
            missing.append(sym)
            continue
        m = measure(bars)
        m.update(group=group, sub=sub, symbol=sym, name=NAMES.get(sym, sym))
        m["bucket"] = band_of(m["gap_pct"], m["days_since_peak"])
        m["notes"] = notes_of(m)
        results.append(m)
    bench = {b: measure(fetched[b]) for b in BENCHMARKS if fetched.get(b) is not None}

    run_label = f"{asof:%Y-%m-%d} 기준" if asof is not None else f"{date.today():%Y-%m-%d} 실행"
    report = build_report(results, missing, bench, run_label)
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, f"{(asof or pd.Timestamp(date.today())):%Y-%m-%d}_ai_ath_gap")
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write(report)
    fields = ["group", "sub", "symbol", "name", "bucket", "date", "close", "peak", "peak_date",
              "days_since_peak", "gap_pct", "gap_close_pct", "gap_52w_pct", "ath", "ath_date",
              "gap_ath_pct", "bars", "notes"]
    with open(stem + ".csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: ("; ".join(r[k]) if k == "notes" else r[k]) for k in fields})
    print(report)
    print(f"저장: {stem}.md / .csv")


if __name__ == "__main__":
    main()
