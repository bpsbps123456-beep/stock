"""AI 탐방 병목 부서 vs 병목 아닌 부서: 상장사 수익률(올해·3개월·1개월)과 선행 PER 비교.

회사는 desk/tour_map.json의 대표 부서(main) 하나로 센다. 병목 여부는 그 부서의 bottleneck 표시(2026-10-05).
가격은 yfinance 수정주가(현지 통화), 기준 종가는 2026-10-05 이하 마지막 거래일. 선행 PER은 거인의 어깨 캐시(giant.json).
py data/derived/analysis/bottleneck_returns/bottleneck_returns.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import watchlist  # noqa: E402
from desk import tour  # noqa: E402

OUT = Path(__file__).with_name("results.csv")
END = "2026-10-06"
SUFFIX = {"KR": [".KS", ".KQ"], "JP": [".T"], "TW": [".TW", ".TWO"], "HK": [".HK"], "CN": [".SZ", ".SS"], "IN": [".NS"],
          "EU": [".DE", ".PA", ".AS", ".SW", ".L", ".MI", ".ST", ".CO", ".VI", ""], "US": [""]}


def companies(m: dict) -> pd.DataFrame:
    alias = {a: t for t, c in m["companies"].items() for a in c.get("aliases", [])}
    ysym = {}
    for subs in watchlist.AI_UNIVERSE.values():
        for syms in subs.values():
            for s in syms:
                k = tour.universe_symbol(s)
                ysym.setdefault(alias.get(k, k), s)
    spots = {}
    for s, z in tour.zones(m):
        for mem in z["members"]:
            spots.setdefault(mem["t"], []).append((s, z, mem))
    rows = []
    for t, lst in spots.items():
        c = m["companies"][t]
        if c["market"] in ("private", "fund"):
            continue
        s, z, _ = next((x for x in lst if x[2].get("main")), lst[0])
        if z.get("special"):
            continue
        cands = [ysym[t]] if t in ysym else [t.zfill(4) + sx if sx == ".HK" else t + sx for sx in SUFFIX.get(c["market"], [""])]
        rows.append({"t": t, "ko": c["ko"], "stop": s["no"], "zone": z["name"], "lead": z.get("lead") == t,
                     "bottleneck": bool(z.get("bottleneck")), "cands": cands})
    return pd.DataFrame(rows)


def returns(df: pd.DataFrame) -> pd.DataFrame:
    allc = sorted({c for cs in df["cands"] for c in cs})
    px = yf.download(allc, start="2025-12-01", end=END, auto_adjust=True, progress=False, threads=True)["Close"]
    giant = json.loads((ROOT / "data/derived/desk/giant.json").read_text(encoding="utf-8"))["stocks"]
    out = []
    for _, r in df.iterrows():
        rec = {k: r[k] for k in ("t", "ko", "stop", "zone", "lead", "bottleneck")}
        sym, s = None, pd.Series(dtype=float)
        for c in r["cands"]:
            if c in px and px[c].dropna().size >= 25:
                sym, s = c, px[c].dropna()
                break
        rec["yahoo"] = sym
        if sym:
            base = s[s.index < "2026-01-01"]
            rec["ytd"] = (s.iloc[-1] / base.iloc[-1] - 1) * 100 if len(base) else None
            rec["r3m"] = (s.iloc[-1] / s.iloc[-64] - 1) * 100 if len(s) >= 64 else None
            rec["r1m"] = (s.iloc[-1] / s.iloc[-22] - 1) * 100
        g = giant.get(sym) or giant.get(r["t"]) or {}
        rec["fwd_pe"] = g.get("fwd_pe")
        out.append(rec)
    return pd.DataFrame(out)


if __name__ == "__main__":
    res = returns(companies(tour.load_map()))
    res.to_csv(OUT, index=False, encoding="utf-8-sig")
    res["layer"] = res["stop"].map(lambda n: "인프라(1~12)" if n <= 12 else "모델·앱(13~14)")
    agg = dict(n=("t", "size"), priced=("ytd", "count"), ytd_med=("ytd", "median"), r3m_med=("r3m", "median"),
               r1m_med=("r1m", "median"), pe_n=("fwd_pe", "count"), pe_med=("fwd_pe", "median"))
    print(res.groupby("bottleneck").agg(**agg).round(1).to_string())
    print(res[res.stop <= 12].groupby("bottleneck").agg(**agg).round(1).to_string())
    print(res[res.lead & (res.stop <= 12)].groupby("bottleneck").agg(**agg).round(1).to_string())
    print("missing price:", res[res.yahoo.isna()].t.tolist())
    print(res[res.t.isin(["GEV", "BE", "ENR", "7011", "034020", "CAT", "CMI"])][["t", "zone", "bottleneck", "ytd", "r3m", "r1m", "fwd_pe"]].round(1).to_string(index=False))
