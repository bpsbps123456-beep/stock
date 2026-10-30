"""2차 비용 모형: 앞 구간(7~8월) 신호에서 시총 구간 × 시간대(장전·개장 15분·오전·장후)마다 최대 80건을 무작위로 뽑아
신호 마감 + 10초의 실측 호가로 반 호가를 잰다. 칸 표본이 20건 미만이면 그 시간대 전체 중앙값을 쓴다.
사용: py cond_cost.py   → results/cost_model.json, cache/cost_quotes.parquet"""
import os, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import engine as E
from fetch_alpaca import keys, Pace
from fetch_quotes import last_quote

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
TIERS = [(-np.inf, 5, "<5B"), (5, 20, "5-20B"), (20, 100, "20-100B"), (100, np.inf, "100B+")]


def tier_of(cap):
    for lo, hi, nm in TIERS:
        if lo <= cap < hi:
            return nm
    return "100B+"


def tod_of(m):
    return "pre" if m < 570 else ("open15" if m < 585 else ("am" if m < 660 else "post"))


def main():
    R = pd.read_parquet(os.path.join(E.CACHE, "cond_2026.parquet"))
    R = R[(R.arm == "sig") & (R.day <= 20260831)].copy()
    R["tier"] = R.cap_b.map(tier_of)
    R["tod"] = R.mins.map(tod_of)
    rng = np.random.default_rng(20261030)
    pick = []
    for _, g in R.groupby(["tier", "tod"]):
        pick.append(g.iloc[rng.choice(len(g), min(80, len(g)), replace=False)])
    S = pd.concat(pick).drop_duplicates(["sym", "ts"])
    head, pace = keys(), Pace(3.0)

    def one(r):
        t = pd.Timestamp(int(r[1]) * 60, unit="s", tz="UTC") + pd.Timedelta(seconds=70)
        ask, bid = last_quote(r[0], t, head, pace)
        return r[0], int(r[1]), ask, bid

    with ThreadPoolExecutor(3) as ex:
        q = list(ex.map(one, S[["sym", "ts"]].itertuples(index=False, name=None)))
    Q = pd.DataFrame(q, columns=["sym", "ts", "ask", "bid"]).merge(S[["sym", "ts", "tier", "tod"]], on=["sym", "ts"])
    ok = (Q.ask > 0) & (Q.bid > 0) & (Q.ask >= Q.bid)
    Q["half"] = np.where(ok, (Q.ask - Q.bid) / (Q.ask + Q.bid), np.nan)
    Q.to_parquet(os.path.join(E.CACHE, "cost_quotes.parquet"))
    by_tod = Q.groupby("tod").half.median().to_dict()
    table = {}
    for (tier, tod), g in Q.groupby(["tier", "tod"]):
        h = g.half.dropna()
        table[f"{tier}|{tod}"] = {"n": int(len(h)), "half": float(h.median() if len(h) >= 20 else by_tod[tod])}
    json.dump({"table": table, "by_tod": by_tod, "n_quotes": int(ok.sum())}, open(os.path.join(RES, "cost_model.json"), "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps(table, ensure_ascii=False))


if __name__ == "__main__":
    main()
