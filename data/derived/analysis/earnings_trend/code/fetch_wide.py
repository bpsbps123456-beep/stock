"""실적 후 흐름 조건 규칙 연구 — 넓은 표본(약 1,700종목) 원자료 받기. 보도자료는 읽지 않는다.

실행: py data/derived/analysis/earnings_trend/code/fetch_wide.py [--step eps|ohlcv|sec|frames|all]

- 종목: 데스크 '급락 후 담기' 종목표(data/derived/desk/dip_universe.json, 나스닥 공개 종목표)에서 지금 시총 30억 달러 이상
  + 핵심 32 + 확대 76. 지금 살아 있는 종목만이라 생존 편향이 있다(사건 시점 시총 20억 달러 미만은 events_wide.py가 뺀다).
- 실적 발표일·EPS 서프라이즈: yfinance get_earnings_dates(8스레드).
- 발표 시각: SEC 8-K 2.02 접수 시각이 1순위(yfinance 시각은 108종목 대조 98.6% 일치, '12:00' 미상·하루 밀림이 있었다).
- 일봉 OHLCV(수정): yfinance. 크기가 커서 cache/wide/*.parquet에만 둔다(Git 제외, 코드로 다시 만든다).
- 분기 매출: SEC frames API(달력 분기마다 전 회사 한 번에). CIK→티커는 company_tickers.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_sec import ROOT, RESULTS, UNIVERSE, Sec, broad_universe  # noqa: E402

warnings.filterwarnings("ignore")
REPO = ROOT.parents[3]
W = ROOT / "cache" / "wide"
BENCH = ["SPY", "QQQ", "SOXX", "IGV", "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
START, END = "2023-06-01", "2026-10-29"  # END 전날(2026-10-28)까지
MIN_CAP_NOW = 3e9
FIELDS = ["Open", "High", "Low", "Close", "Volume"]
FRAME_CONCEPTS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                  "RevenuesNetOfInterestExpense"]


def universe() -> pd.DataFrame:
    d = json.loads((REPO / "data/derived/desk/dip_universe.json").read_text(encoding="utf-8"))["symbols"]
    rows = [{"ticker": s, "name": v.get("name", ""), "cap_now": v.get("cap"), "sector": v.get("sector", ""),
             "industry": v.get("industry", "")} for s, v in d.items() if (v.get("cap") or 0) >= MIN_CAP_NOW]
    df = pd.DataFrame(rows)
    extra = [t for t in UNIVERSE + broad_universe() if t not in set(df["ticker"])]
    df = pd.concat([df, pd.DataFrame({"ticker": extra})], ignore_index=True)
    sys.path.insert(0, str(REPO))
    import watchlist  # noqa: E402
    df["ai_sub"] = df["ticker"].map(lambda t: watchlist.AI_SUBSECTOR.get(t, ""))
    return df.drop_duplicates("ticker").reset_index(drop=True)


def load_ohlcv() -> dict[str, pd.DataFrame]:
    return {k: pd.read_parquet(W / f"{k.lower()}.parquet") for k in FIELDS}


def step_eps(tickers: list[str]) -> None:
    """이미 받은 종목은 건너뛰고 빠진 종목만 받는다. Yahoo가 막으면 '데이터 없음'처럼 보이므로
    스레드 3개로 천천히, 빈 결과는 쉬었다가 최대 3번 다시 묻는다."""
    import time
    path = RESULTS / "wide_eps.csv"
    have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["ticker"])
    done = set(have["ticker"])

    def one(t: str) -> list[dict]:
        time.sleep(0.4)
        try:
            f = yf.Ticker(t).get_earnings_dates(limit=16)
        except Exception:  # yfinance가 종목마다 다른 예외를 던진다 — 빈 결과로 둔다
            return []
        if f is None or f.empty:
            return []
        return [{"ticker": t, "date": ts.strftime("%Y-%m-%d"), "time_et": ts.strftime("%H:%M"),
                 "eps_estimate": r.get("EPS Estimate"), "eps_reported": r.get("Reported EPS"),
                 "eps_surprise_pct": r.get("Surprise(%)")} for ts, r in f.iterrows()]
    frames = [have]
    todo = [t for t in tickers if t not in done]
    for attempt in range(3):
        if not todo:
            break
        print(f"EPS 받기 {attempt + 1}회차: {len(todo)}종목", flush=True)
        with ThreadPoolExecutor(3) as ex:
            got = list(ex.map(one, todo))
        new = [r for rs in got for r in rs]
        frames.append(pd.DataFrame(new))
        todo = [t for t, rs in zip(todo, got) if not rs]
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        if todo:
            time.sleep(60)
    out = pd.concat(frames, ignore_index=True)
    print(f"EPS 이력 {len(out)}행, 종목 {out['ticker'].nunique()}개, 끝내 없음 {len(todo)}개", flush=True)


def step_ohlcv(tickers: list[str]) -> None:
    allt = tickers + BENCH
    parts = {k: [] for k in FIELDS}
    for i in range(0, len(allt), 200):
        df = yf.download(allt[i:i + 200], start=START, end=END, auto_adjust=True, progress=False, threads=True)
        for k in parts:
            parts[k].append(df[k])
        print(f"OHLCV {min(i + 200, len(allt))}/{len(allt)}", flush=True)
    data = {k: pd.concat(v, axis=1).sort_index() for k, v in parts.items()}
    data = {k: v.loc[:, ~v.columns.duplicated()] for k, v in data.items()}
    last = data["Close"].iloc[-1]
    print(f"끝 {data['Close'].index[-1].date()}, 마지막 날 종가 빈 종목 {int(last.isna().sum())}개(상장폐지·거래정지 등)")
    for b in BENCH:
        if pd.isna(last.get(b)):
            raise SystemExit(f"벤치마크 {b} 마지막 종가가 비었다 — 멈춤")
    for k, v in data.items():
        v.to_parquet(W / f"{k.lower()}.parquet")


def step_sec(tickers: list[str]) -> None:
    sec = Sec(gap=0.15)
    ciks = {row["ticker"].upper(): int(row["cik_str"]) for row in sec.json("https://www.sec.gov/files/company_tickers.json").values()}
    (W / "ciks.json").write_text(json.dumps(ciks), encoding="utf-8")
    rows = []
    for n, t in enumerate(tickers):
        cik = ciks.get(t.replace("-", ".")) or ciks.get(t)
        if not cik:
            continue
        try:
            sub = sec.json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        except Exception:  # 제출 목록이 없는 회사(해외 발행사 일부)는 yfinance 시각을 쓴다
            continue
        rec = sub["filings"]["recent"]
        for i, form in enumerate(rec["form"]):
            if form == "8-K" and "2.02" in (rec["items"][i] or "") and rec["filingDate"][i] >= "2024-06-01":
                et = pd.Timestamp(rec["acceptanceDateTime"][i]).tz_convert("America/New_York")
                rows.append({"ticker": t, "cik": cik, "filing_date": rec["filingDate"][i],
                             "acceptance_et": et.strftime("%Y-%m-%d %H:%M")})
        if n % 200 == 0:
            print(f"SEC {n}/{len(tickers)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(W / "sec_202.csv", index=False)
    print(f"SEC 2.02 {len(out)}건, 종목 {out['ticker'].nunique()}개", flush=True)


def step_frames() -> None:
    sec = Sec(gap=0.15)
    rows = []
    for y in [2023, 2024, 2025, 2026]:
        for qn in [1, 2, 3, 4]:
            if (y, qn) > (2026, 3):
                continue
            for c in FRAME_CONCEPTS:
                try:
                    fr = sec.json(f"https://data.sec.gov/api/xbrl/frames/us-gaap/{c}/USD/CY{y}Q{qn}.json")
                except Exception:  # 아직 만들어지지 않은 분기 틀
                    continue
                for d in fr["data"]:
                    rows.append({"cik": d["cik"], "concept": c, "frame": f"{y}Q{qn}", "end": d["end"], "val": d["val"]})
    out = pd.DataFrame(rows)
    # 같은 회사·분기에서 개념이 여럿이면 큰 값(총매출 ≥ 하위 항목)
    out = out.sort_values("val").drop_duplicates(["cik", "frame"], keep="last")
    out.to_csv(W / "frames_revenue.csv", index=False)
    print(f"frames 매출 {len(out)}행, 회사 {out['cik'].nunique()}개", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["eps", "ohlcv", "sec", "frames", "all"])
    a = ap.parse_args()
    W.mkdir(parents=True, exist_ok=True)
    uni = universe()
    uni.to_csv(RESULTS / "wide_universe.csv", index=False)
    tickers = uni["ticker"].tolist()
    print(f"종목 {len(tickers)}개", flush=True)
    if a.step in ("eps", "all"):
        step_eps(tickers)
    if a.step in ("ohlcv", "all"):
        step_ohlcv(tickers)
    if a.step in ("sec", "all"):
        step_sec(tickers)
    if a.step in ("frames", "all"):
        step_frames()


if __name__ == "__main__":
    main()
