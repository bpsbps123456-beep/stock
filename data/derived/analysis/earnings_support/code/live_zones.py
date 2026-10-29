"""오늘 밤(10/29 장 마감 뒤) 발표 10종목의 160분봉 지지·저항 구역 — 10/28 정규장 마감 기준.

실행: py data/derived/analysis/earnings_support/code/live_zones.py [YYYY-MM-DD]
출력: results/live_zones.csv (지지 Z1~Z6, 저항 R1~R3, 거리는 그날 공식 종가 대비 %)
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lib import CACHE, RES, bars160, load_h1, pivots, zones  # noqa: E402

LIVE = ["SNDK", "INTC", "RDDT", "AMZN", "AXTI", "TWLO", "MSTR", "ILMN", "COIN", "FSLR"]
SLOT = {0: "09:30", 1: "12:30", 2: "14:30"}


def main(day: str = "2026-10-28") -> None:
    close = pd.read_parquet(CACHE / "d1_close.parquet")
    rows = []
    for t in LIVE:
        h = load_h1(t)
        if h is None:
            print("자료 없음", t)
            continue
        b = bars160(h)
        b = b[b["day"] <= pd.Timestamp(day)].reset_index(drop=True)
        if b.empty or b["day"].iloc[-1] != pd.Timestamp(day):
            print("그날 봉 없음", t)
            continue
        D = len(b) - 1
        ref_bar = float(b["close"].iloc[D])
        ref = float(close.loc[day, t]) if t in close.columns and pd.Timestamp(day) in close.index else ref_bar
        piv = pivots(b)
        sup, res, total = zones(piv, D, ref_bar)
        for side, zs, n in (("지지", sup, 6), ("저항", res, 3)):
            for k, z in enumerate(zs[:n], 1):
                lo, hi, cnt, q = z
                rows.append({"ticker": t, "side": side, "level": f"{'Z' if side == '지지' else 'R'}{k}",
                             "lo": round(lo, 4), "hi": round(hi, 4), "count": cnt,
                             "last_swing": f"{b['day'].iloc[q]:%Y-%m-%d} {SLOT[int(b['g'].iloc[q])]}",
                             "lo_pct": round((lo / ref - 1) * 100, 2), "hi_pct": round((hi / ref - 1) * 100, 2),
                             "ref_close": round(ref, 4), "ref_bar_close": round(ref_bar, 4), "swings_total": total,
                             "short_hist": total < 60, "asof": day})
    out = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    out.to_csv(RES / "live_zones.csv", index=False, encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out[["ticker", "level", "lo", "hi", "count", "lo_pct", "hi_pct", "last_swing", "swings_total"]])


if __name__ == "__main__":
    main(*sys.argv[1:])
