"""AI 탐방 두 축 진영별 수익률: OpenAI만 / Anthropic만 / 양쪽 계약 상장사의 YTD·3개월·1개월 수익률.

진영은 desk/tour_map.json의 계약 줄(links, 2026-10-05 기준)로 나눈다. 비상장은 뺀다.
가격은 yfinance 수정주가(현지 통화), 기준 종가는 2026-10-05 이하 마지막 거래일.
py data/derived/analysis/ai_axes_returns/axes_returns.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from desk import tour  # noqa: E402

OUT = Path(__file__).with_name("results.csv")
END = "2026-10-06"           # yfinance end is exclusive → last close 2026-10-05
YAHOO = {"GOOG": "GOOGL", "SKHY": "000660.KS", "064400": "064400.KS", "9984": "9984.T", "028260": "028260.KS",
         "010140": "010140.KS", "035720": "035720.KS", "TCS": "TCS.NS", "005930": "005930.KS",
         "017670": "017670.KS", "018260": "018260.KS"}


def camps(m: dict) -> pd.DataFrame:
    axes: dict[str, set] = {}
    for ln in m["links"]:
        axes.setdefault(ln["t"], set()).add(ln["axis"])
    spots: dict[str, list] = {}
    for s, z in tour.zones(m):
        for mem in z["members"]:
            spots.setdefault(mem["t"], []).append((s["no"], mem.get("main", False)))
    rows = []
    for t, a in axes.items():
        c = m["companies"][t]
        if c["market"] == "private":
            continue
        sp = spots.get(t, [])
        main = next((no for no, mn in sp if mn), sp[0][0] if sp else None)
        rows.append({"t": t, "ko": c["ko"], "camp": "양쪽" if len(a) == 2 else ("OpenAI만" if "openai" in a else "Anthropic만"),
                     "stop": main, "layer": "인프라(1~12)" if main and main <= 12 else "모델·앱(13~14)" if main else "경로 밖",
                     "yahoo": YAHOO.get(t, t)})
    return pd.DataFrame(rows)


def returns(df: pd.DataFrame) -> pd.DataFrame:
    px = yf.download(sorted(df["yahoo"]), start="2025-12-01", end=END, auto_adjust=True, progress=False)["Close"]
    out = []
    for _, r in df.iterrows():
        s = px[r["yahoo"]].dropna() if r["yahoo"] in px else pd.Series(dtype=float)
        rec = r.to_dict()
        if len(s) < 25:
            rec.update(last=None, ytd=None, r3m=None, r1m=None)
        else:
            base = s[s.index < "2026-01-01"]
            rec.update(last=str(s.index[-1].date()),
                       ytd=(s.iloc[-1] / base.iloc[-1] - 1) * 100 if len(base) else None,
                       r3m=(s.iloc[-1] / s.iloc[-64] - 1) * 100 if len(s) >= 64 else None,
                       r1m=(s.iloc[-1] / s.iloc[-22] - 1) * 100)
        out.append(rec)
    return pd.DataFrame(out)


if __name__ == "__main__":
    res = returns(camps(tour.load_map()))
    res.to_csv(OUT, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(res.sort_values(["camp", "ytd"]).to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    for by in (["camp"], ["camp", "layer"]):
        g = res.groupby(by).agg(n=("t", "size"), ytd_n=("ytd", "count"), ytd_med=("ytd", "median"), ytd_mean=("ytd", "mean"),
                                r3m_med=("r3m", "median"), r1m_med=("r1m", "median"))
        print("\n", g.round(1).to_string())
