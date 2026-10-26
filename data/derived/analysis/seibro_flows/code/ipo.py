"""신규 상장(IPO·ADR 첫 거래) 날짜표(BRIEF v2 §3: 미국 첫 거래 뒤 4주 안 ISIN은 분모에서 뺀다).

후보: 보통주·ADR·REIT 중 SEIBRO 첫 등장 ≥ 2023-02-01이고, 첫 등장 뒤 20 SEIBRO 날짜의 합계 거래가 2천만 달러 이상인 것.
yfinance 일봉의 첫 날짜를 첫 거래일로 보고, 첫 등장 45일 전 이후면 신규 상장으로 기록한다. 흐름–가격 연결이 아니다(상장일만 쓴다).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "derived"


def main() -> None:
    m = pd.read_csv(DER / "isin_master.csv")
    d = pd.read_parquet(DER / "settle_daily.parquet")
    d["start"] = pd.to_datetime(d["start"])
    m = m[m["figi_type"].isin(["Common Stock", "ADR", "REIT", "NY Reg Shrs"]) & (m["first_seen"] >= "2023-02-01")]
    out = {}
    for _, r in m.iterrows():
        sub = d[d["isin"] == r["isin"]].sort_values("start")
        if sub.empty:
            continue
        dates = sorted(sub["start"].unique())[:20]
        amt = sub[sub["start"].isin(dates)]["tot"].sum()
        if amt < 2e7 or not isinstance(r["figi_ticker"], str):
            continue
        try:
            h = yf.Ticker(r["figi_ticker"]).history(period="max", auto_adjust=False)
        except Exception:
            continue
        if h.empty:
            continue
        first = h.index[0].tz_localize(None).normalize()
        if first >= pd.Timestamp(r["first_seen"]) - pd.Timedelta(days=45):
            out[r["isin"]] = {"ticker": r["figi_ticker"], "first_trade": first.date().isoformat(),
                              "first_seen": r["first_seen"], "amt_first20": float(amt)}
    (DER / "ipo.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(out), sorted(((v["first_trade"], v["ticker"]) for v in out.values()))[-15:])


if __name__ == "__main__":
    main()
