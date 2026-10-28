"""실적 추세 연구 — 가격·EPS 컨센서스 받기(yfinance).

실행: py data/derived/analysis/earnings_trend/code/fetch_market.py

- 일봉 수정종가(배당·분할 반영) → results/prices_adj_close.csv. 대량 백테스트라 yfinance를 쓰고,
  표본 몇 개를 트레이딩뷰와 대조한다(메모리 규칙: 가격은 TV 우선, 대량 이력만 yfinance).
- 마지막 날 값이 비면 ffill로 메우지 않고 멈춘다(2026-10-14 yfinance가 최근 종가를 NaN으로 준 적 있음).
- get_earnings_dates의 EPS 추정치·실제·서프라이즈 → results/eps_consensus.csv.
  발표 당시 컨센서스다(발표 뒤 값이 아님). 매출 컨센서스 이력은 무료 출처가 없어 쓰지 않는다
  (트레이딩뷰는 스크립트 입력 금지 — AGENTS.md).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse  # noqa: E402

from fetch_sec import RESULTS, UNIVERSE, suffix  # noqa: E402

warnings.filterwarnings("ignore")
BENCH = ["QQQ", "SPY", "SOXX", "IGV", "XLV", "XLU"]
START, END = "2023-06-01", "2026-10-28"  # END는 yfinance 규칙상 그 전날까지


def prices(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(수정 종가, 수정 시가). 시가는 '발표 뒤 첫 정규장 시가에 들어가는' 경우를 재려고 받는다."""
    df = yf.download(tickers + BENCH, start=START, end=END, auto_adjust=True, progress=False, threads=True)
    close = df["Close"].sort_index().dropna(axis=1, how="all")
    last = close.iloc[-1]
    if last.isna().any():
        raise SystemExit(f"마지막 날({close.index[-1].date()}) 종가가 빈 종목: {list(last[last.isna()].index)} — 멈춤")
    opn = df["Open"].sort_index()[close.columns]
    for f in (close, opn):
        f.index = f.index.strftime("%Y-%m-%d")
    return close.round(4), opn.round(4)


def eps(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            frame = yf.Ticker(t).get_earnings_dates(limit=24)
        except Exception as ex:  # yfinance가 종목별로 KeyError 등을 던질 때가 있다
            print(f"{t}: EPS 이력 실패 {type(ex).__name__}", file=sys.stderr)
            continue
        if frame is None or frame.empty:
            print(f"{t}: EPS 이력 없음", file=sys.stderr)
            continue
        for ts, r in frame.iterrows():
            rows.append({"ticker": t, "date": ts.strftime("%Y-%m-%d"), "time_et": ts.strftime("%H:%M"),
                         "eps_estimate": r.get("EPS Estimate"), "eps_reported": r.get("Reported EPS"),
                         "eps_surprise_pct": r.get("Surprise(%)")})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="core", choices=["core", "broad"])
    ap.add_argument("--prices-only", action="store_true")
    a = ap.parse_args()
    sfx = suffix(a.set)
    tickers = UNIVERSE if a.set == "core" else sorted(pd.read_csv(RESULTS / "sec_events_broad.csv")["ticker"].unique())
    p, o = prices(tickers)
    p.to_csv(RESULTS / f"prices_adj_close{sfx}.csv")
    o.to_csv(RESULTS / f"prices_adj_open{sfx}.csv")
    print(f"가격 {p.shape[1]}종목 × {p.shape[0]}일, 끝 {p.index[-1]}")
    if "--prices-only" in sys.argv:
        return
    e = eps(tickers)
    e.to_csv(RESULTS / f"eps_consensus{sfx}.csv", index=False)
    print(f"EPS 이력 {len(e)}행")


if __name__ == "__main__":
    main()
