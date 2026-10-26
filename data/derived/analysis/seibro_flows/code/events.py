"""가격만으로 정하는 사건(SEIBRO 미사용).

A 급락 시작: 종가가 직전 5세션 최고 종가보다 8% 이상 낮아진 첫날 R → 그 최고 종가 다음 날 A. 직전 A에서 10세션 넘게 지나야 새 사건.
T 고점: 종가 지그재그 8%(고점 뒤 8% 하락이 확정된 고점).
asof: 그 날까지의 가격만 쓴다(구간 끝 절단). 확정 날짜(R, T의 8% 하락 확인일)를 함께 남긴다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def closes(ticker: str) -> pd.Series:
    px = pd.read_csv(ROOT / "raw" / "prices.csv", parse_dates=["date"])
    s = px[px["ticker"] == ticker].set_index("date")["Close"].dropna()
    return s[~s.index.duplicated()]


def drawdown_starts(c: pd.Series, asof=None, drop=0.08, look=5, gap=10) -> list[dict]:
    c = c[c.index <= pd.Timestamp(asof)] if asof else c
    v, idx = c.values, c.index
    out, last_a = [], -10**9
    for i in range(look, len(v)):
        w = v[i - look:i]
        j = int(w.argmax()) + i - look
        if v[i] <= (1 - drop) * w.max():
            a = j + 1
            if a - last_a > gap:
                out.append({"A": idx[a].date().isoformat(), "R": idx[i].date().isoformat(),
                            "peak": idx[j].date().isoformat()})
                last_a = a
    return out


def zigzag_tops(c: pd.Series, asof=None, pct=0.08) -> tuple[list[dict], list[dict], dict]:
    """확정 고점·저점과 마지막 미확정 극값."""
    c = c[c.index <= pd.Timestamp(asof)] if asof else c
    v, idx = c.values, c.index
    tops, bottoms = [], []
    mode, ext_i = 0, 0  # 0 미정, 1 오름(고점 찾는 중), -1 내림
    lo_i = hi_i = 0
    for i in range(1, len(v)):
        if mode >= 0:
            if v[i] > v[hi_i]:
                hi_i = i
            if v[i] <= v[hi_i] * (1 - pct) and (mode == 1 or hi_i > 0):
                tops.append({"T": idx[hi_i].date().isoformat(), "confirmed": idx[i].date().isoformat()})
                mode, lo_i = -1, i
                continue
        if mode <= 0:
            if v[i] < v[lo_i]:
                lo_i = i
            if v[i] >= v[lo_i] * (1 + pct):
                bottoms.append({"B": idx[lo_i].date().isoformat(), "confirmed": idx[i].date().isoformat()})
                mode, hi_i = 1, i
    pending = {"mode": "up" if mode == 1 else "down" if mode == -1 else "none",
               "extreme": idx[hi_i if mode >= 0 else lo_i].date().isoformat()}
    return tops, bottoms, pending


def main() -> None:
    out = {}
    for t in ["SOXX", "QQQ", "EWY", "^SOX"]:
        c = closes(t)
        rec = {}
        for name, asof in [("full_20261023", "2026-10-23"), ("front_20260529", "2026-05-29")]:
            a = drawdown_starts(c, asof)
            tp, bt, pend = zigzag_tops(c, asof)
            rec[name] = {"A": a, "T": tp, "B": bt, "pending": pend}
        out[t] = rec
        f = rec["full_20261023"]
        a26 = [x["A"] for x in f["A"] if x["A"] >= "2026"]
        t26 = [x["T"] for x in f["T"] if x["T"] >= "2026"]
        print(t, "A2026:", a26, "| T2026:", t26, "| pending:", f["pending"])
        print("   A2025:", [x["A"] for x in f["A"] if "2025" <= x["A"] < "2026"],
              "| T2025:", [x["T"] for x in f["T"] if "2025" <= x["T"] < "2026"])
    (ROOT / "derived").mkdir(exist_ok=True)
    (ROOT / "derived" / "events.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
