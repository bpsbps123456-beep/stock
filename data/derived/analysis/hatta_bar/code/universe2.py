"""2차 종목 넓히기: 나스닥 목록(시총 10억 달러 이상, 펀드·SPAC·우선주·워런트 제외)의 Alpaca 일봉을 받고,
2026년 6월(연구 기간 전) 일 거래대금 중앙값 상위 비AI 종목으로 시총 구간마다 100종목을 채운다. AI 115종목은 모두 남긴다.
결과: cache/daily_2026.parquet, cache/daily_2025.parquet(일봉, 분할 반영), results/universe2.json
사용: py universe2.py"""
import os, json, time
import numpy as np
import pandas as pd
import requests
from fetch_alpaca import keys, Pace, ROOT
from build import SYMS as AI_SYMS

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "../cache")
RES = os.path.join(HERE, "../results")
URL = "https://data.alpaca.markets/v2/stocks/bars"
TIERS = [(1, 5, "1-5B"), (5, 20, "5-20B"), (20, 100, "20-100B"), (100, 1e9, "100B+")]
PER_TIER = 100
BAD = "Acquisition|Warrant|Unit|Right|Preferred|Depositary Shares Each|Notes|ETF|Fund|Trust"


def candidates():
    u = json.load(open(os.path.join(ROOT, "data/derived/analysis/premarket_shock/bars/universe.json"), encoding="utf-8"))
    df = pd.DataFrame.from_dict(u, orient="index")
    df["shares"] = df.cap / df.price
    bad = df.name.str.contains(BAD, case=False, na=False)
    keep = (df.cap >= 1e9) & ~bad & df.index.str.fullmatch(r"[A-Z]{1,5}")
    df = df[keep | df.index.isin(AI_SYMS)]
    return df


def daily(syms, start, end, head, pace):
    rows = []
    for i in range(0, len(syms), 150):
        batch = syms[i:i + 150]
        token = None
        while True:
            p = {"symbols": ",".join(batch), "timeframe": "1Day", "start": start, "end": end, "limit": 10000,
                 "adjustment": "split", "feed": "sip"}
            if token:
                p["page_token"] = token
            for attempt in range(6):
                pace.wait()
                r = requests.get(URL, headers=head, params=p, timeout=60)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(2 + 3 * attempt)
                    continue
                r.raise_for_status()
                break
            j = r.json()
            for s, bars in (j.get("bars") or {}).items():
                for b in bars:
                    rows.append((s, b["t"][:10], b["o"], b["h"], b["l"], b["c"], b["v"], b.get("vw", np.nan)))
            token = j.get("next_page_token")
            if not token:
                break
        print(f"daily {start[:4]} {i + len(batch)}/{len(syms)}", flush=True)
    d = pd.DataFrame(rows, columns=["sym", "date", "o", "h", "l", "c", "v", "vw"])
    d["date"] = pd.to_datetime(d.date)
    return d.sort_values(["sym", "date"]).reset_index(drop=True)


def main():
    head, pace = keys(), Pace(3.0)
    cand = candidates()
    syms = sorted(cand.index)
    d26 = daily(syms, "2026-03-01T00:00:00Z", "2026-10-29T00:00:00Z", head, pace)
    d26.to_parquet(os.path.join(CACHE, "daily_2026.parquet"))
    jun = d26[(d26.date >= "2026-06-01") & (d26.date <= "2026-06-30")].copy()
    jun["dv"] = jun.v * jun.vw.fillna(jun.c)
    adv = jun.groupby("sym").dv.median()
    last = jun.sort_values("date").groupby("sym").c.last()
    cand["adv_jun"] = adv
    cand["cap_jun_b"] = cand.shares * last / 1e9
    cand = cand.dropna(subset=["adv_jun", "cap_jun_b"])
    cand["ai"] = cand.index.isin(AI_SYMS)
    pick = set(AI_SYMS)
    out = {}
    for lo, hi, name in TIERS:
        t = cand[(cand.cap_jun_b >= lo) & (cand.cap_jun_b < hi)]
        n_ai = int(t.ai.sum())
        extra = t[~t.ai].sort_values("adv_jun", ascending=False).head(max(0, PER_TIER - n_ai))
        pick |= set(extra.index)
        print(name, "AI", n_ai, "추가", len(extra), "6월 거래대금 하한 $%.0fM" % (extra.adv_jun.min() / 1e6 if len(extra) else 0))
    for s in sorted(pick):
        if s not in cand.index:
            continue
        r = cand.loc[s]
        out[s] = {"shares": float(r.shares), "cap_jun_b": float(r.cap_jun_b), "adv_jun": float(r.adv_jun), "ai": bool(r.ai),
                  "sector": r.sector, "industry": r.industry}
    json.dump(out, open(os.path.join(RES, "universe2.json"), "w"), ensure_ascii=False, indent=1)
    d25 = daily(sorted(out), "2025-03-01T00:00:00Z", "2025-11-04T00:00:00Z", head, pace)
    d25.to_parquet(os.path.join(CACHE, "daily_2025.parquet"))
    print("종목", len(out), "AI", sum(v["ai"] for v in out.values()))


if __name__ == "__main__":
    main()
