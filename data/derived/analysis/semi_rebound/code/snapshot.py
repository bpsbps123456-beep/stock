"""예측 고정 전 반도체 43종목(+SKHY)의 사전 정보 스냅샷 (2026-10-28 밤, 미국 10/28 장 전후).

yfinance에서 실적 전망 수정(eps_trend·eps_revisions), 실적 발표일, 공매도, 시총, 선행 PER를 받는다.
결과: ../results/snapshot_2026-10-28.json. 예측 원장(data/forecast/)에 그대로 넣어 한 달 뒤 비교한다.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

from analyze import TICKERS

OUT = Path(__file__).resolve().parents[1] / "results"
EXTRA = ["SKHY"]
INFO_KEYS = ["marketCap", "shortPercentOfFloat", "sharesShort", "sharesShortPriorMonth", "dateShortInterest",
             "forwardPE", "trailingPE", "forwardEps", "revenueGrowth", "earningsGrowth", "operatingMargins",
             "recommendationMean", "targetMeanPrice", "numberOfAnalystOpinions", "earningsTimestamp",
             "earningsTimestampStart", "earningsTimestampEnd", "isEarningsDateEstimate", "currentPrice",
             "regularMarketPreviousClose", "fiftyTwoWeekHigh", "longName"]


def num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v if isinstance(v, (str, bool)) else None
    return None if math.isnan(f) or math.isinf(f) else f


def frame(df):
    if df is None or getattr(df, "empty", True):
        return None
    return {str(i): {str(c): num(v) for c, v in row.items()} for i, row in df.iterrows()}


def one(t: str) -> dict:
    tk = yf.Ticker(t)
    out = {"ticker": t}
    try:
        info = tk.info or {}
        out["info"] = {k: num(info.get(k)) for k in INFO_KEYS}
    except Exception as exc:
        out["info_error"] = f"{type(exc).__name__}: {exc}"
    for name in ("eps_trend", "eps_revisions", "earnings_estimate", "revenue_estimate"):
        try:
            out[name] = frame(getattr(tk, name))
        except Exception as exc:
            out[name + "_error"] = f"{type(exc).__name__}: {exc}"
    try:
        ed = tk.get_earnings_dates(limit=6)
        out["earnings_dates"] = None if ed is None else [
            {"date": i.isoformat(), **{str(c): num(v) for c, v in r.items()}} for i, r in ed.iterrows()]
    except Exception as exc:
        out["earnings_dates_error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> None:
    rows = {}
    for t in TICKERS + EXTRA:
        rows[t] = one(t)
        time.sleep(0.3)
    snap = {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": "yfinance 1.2.2",
            "tickers": rows}
    (OUT / "snapshot_2026-10-28.json").write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [t for t, r in rows.items() if "info_error" in r or not r.get("eps_trend")]
    print(f"saved {len(rows)} tickers; missing info/eps_trend: {bad}")


if __name__ == "__main__":
    main()
