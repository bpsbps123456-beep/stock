"""실적 추세 연구 — XBRL companyfacts → 분기(3개월) 재무 표.

실행: py data/derived/analysis/earnings_trend/code/fundamentals.py  → results/quarters.csv

- 손익은 3개월 값을 그대로 쓴다. 현금흐름(SBC·OCF·CapEx)은 10-Q가 연초 누계라
  '같은 시작일의 직전 누계'를 빼서 3개월 값을 만든다(4분기 = 연간 − 9개월 누계).
- 같은 기간 값이 여러 번 제출되면 처음 제출된 값을 쓴다(나중 재작성 값이 섞이지 않게).
- 매출 YoY는 보도자료 표(에이전트 추출)가 시장이 본 숫자라 1순위이고, 여기 값은 대조용이다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse  # noqa: E402

from fetch_sec import CACHE, RESULTS, UNIVERSE, suffix  # noqa: E402

FLOW = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                "RevenuesNetOfInterestExpense"],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "op_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense",
            "ShareBasedPaymentArrangementNoncashExpense"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsToAcquireOtherPropertyPlantAndEquipment"],
}
# 가중평균 주식수는 더할 수 없는 값이라 누계끼리 빼지 않는다(4분기는 연간 가중평균을 그대로 쓴다)
SHARES = ["WeightedAverageNumberOfDilutedSharesOutstanding"]
INSTANT = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "st_invest": ["ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
}
MIN_END = "2022-06-01"


def _facts(gaap: dict, concept: str) -> pd.DataFrame:
    if concept not in gaap:
        return pd.DataFrame()
    units = gaap[concept]["units"]
    rows = units.get("USD") or units.get("shares") or next(iter(units.values()))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["form"].isin(["10-Q", "10-K", "10-Q/A", "10-K/A"])]
    df = df.sort_values("filed").drop_duplicates(["start", "end"] if "start" in df else ["end"], keep="first")
    return df[df["end"] >= MIN_END]


def quarterize(df: pd.DataFrame) -> dict[str, float]:
    """{분기말: 3개월 값}. 3개월 값이 없으면 같은 시작일 누계끼리 뺀다."""
    if df.empty or "start" not in df:
        return {}
    df = df.copy()
    df["s"], df["e"] = pd.to_datetime(df["start"]), pd.to_datetime(df["end"])
    df["dur"] = (df["e"] - df["s"]).dt.days
    out = {}
    for _, r in df[(df["dur"] >= 80) & (df["dur"] <= 100)].iterrows():
        out[r["end"]] = float(r["val"])
    by_start = {k: g for k, g in df[df["dur"] > 100].groupby("start")}
    for start, g in by_start.items():
        same = df[df["start"] == start]
        for _, r in g.iterrows():
            if r["end"] in out:
                continue
            prev = same[((r["e"] - same["e"]).dt.days > 60) & ((r["e"] - same["e"]).dt.days < 120)]
            if len(prev):
                out[r["end"]] = float(r["val"]) - float(prev.iloc[-1]["val"])
    return out


def instants(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    return {r["end"]: float(r["val"]) for _, r in df.iterrows()}


def company(ticker: str) -> pd.DataFrame:
    gaap = json.loads((CACHE / "facts" / f"{ticker}.json").read_text(encoding="utf-8"))["facts"].get("us-gaap", {})
    cols: dict[str, dict[str, float]] = {}
    for name, concepts in FLOW.items():
        merged: dict[str, float] = {}
        for c in concepts:
            for end, v in quarterize(_facts(gaap, c)).items():
                # 매출은 같은 분기 후보 중 가장 큰 값(총매출 ≥ 하위 항목). 나머지는 앞 순위 개념 우선
                if name == "revenue":
                    merged[end] = max(merged.get(end, v), v)
                else:
                    merged.setdefault(end, v)
        cols[name] = merged
    shares: dict[str, float] = {}
    df = _facts(gaap, SHARES[0])
    if not df.empty:
        df = df.assign(dur=(pd.to_datetime(df["end"]) - pd.to_datetime(df["start"])).dt.days)
        for _, r in df.sort_values("dur", ascending=False).iterrows():
            shares[r["end"]] = float(r["val"])  # 짧은 기간 값이 나중에 덮어써 3개월 값이 우선
    cols["dil_shares"] = shares
    for name, concepts in INSTANT.items():
        merged = {}
        for c in concepts:
            for end, v in instants(_facts(gaap, c)).items():
                merged.setdefault(end, v)
        cols[name] = merged
    ends = sorted(cols["revenue"])
    df = pd.DataFrame({k: pd.Series(v) for k, v in cols.items()}).reindex(ends)
    df.index.name = "period_end"
    df = df.reset_index()
    # 분기말이 며칠씩 다른 회사(52·53주 회계)는 전년 동기를 '1년 ±10일'로 찾는다
    e = pd.to_datetime(df["period_end"])
    yo = []
    for i, d in enumerate(e):
        j = ((e - (d - pd.Timedelta(days=364))).abs().dt.days <= 10) & (e < d)
        yo.append(df.loc[j, "revenue"].iloc[-1] if j.any() else None)
    df["revenue_py"] = yo
    df["revenue_yoy_xbrl"] = (df["revenue"] / df["revenue_py"] - 1) * 100
    if df["gross_profit"].isna().all() and df["cost_of_revenue"].notna().any():
        df["gross_profit"] = df["revenue"] - df["cost_of_revenue"]
    df.insert(0, "ticker", ticker)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="core", choices=["core", "broad"])
    a = ap.parse_args()
    sfx = suffix(a.set)
    tickers = UNIVERSE if a.set == "core" else sorted(pd.read_csv(RESULTS / "sec_events_broad.csv")["ticker"].unique())
    frames = []
    for t in tickers:
        try:
            frames.append(company(t))
        except (FileNotFoundError, KeyError, ValueError) as ex:
            print(f"{t}: facts 처리 불가 {type(ex).__name__}", file=sys.stderr)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(RESULTS / f"quarters{sfx}.csv", index=False)
    cov = out[out["period_end"] >= "2024-09-01"].groupby("ticker")[["revenue", "sbc", "ocf", "capex", "dil_shares"]].count()
    print(cov.to_string() if a.set == "core" else f"종목 {out['ticker'].nunique()}개")
    print(f"저장: quarters{sfx}.csv ({len(out)}행)")


if __name__ == "__main__":
    main()
