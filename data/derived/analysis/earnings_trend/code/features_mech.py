"""실적 추세 연구 — 기계 계산 요인만(보도자료 정독 없이 XBRL·yfinance) 핵심 32 + 확대 76종목.

실행: py data/derived/analysis/earnings_trend/code/features_mech.py  → results/features_mech.csv

정독 표본(32)의 결론 중 '숫자로만 되는 부분'(성장 수준·가속·수익성·EPS 서프라이즈·기대치)이
더 큰 표본에서도 같은지 보는 확인용이다. 가이던스·내러티브는 없다.
yoy_x = XBRL 첫 제출 분기 매출의 YoY, accel_x = yoy_x − 직전 분기 yoy_x.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import add_xbrl_eps, size_tier  # noqa: E402
from fetch_sec import RESULTS  # noqa: E402


def one(sfx: str, label: str) -> pd.DataFrame:
    ev = pd.read_csv(RESULTS / f"events{sfx}.csv")
    q = pd.read_csv(RESULTS / f"quarters{sfx}.csv")
    eps = pd.read_csv(RESULTS / f"eps_consensus{sfx}.csv")
    eps["d"] = pd.to_datetime(eps["date"])
    q = q.sort_values(["ticker", "period_end"])
    q["yoy_prev_x"] = q.groupby("ticker")["revenue_yoy_xbrl"].shift(1)
    key = q[["ticker", "period_end", "revenue_yoy_xbrl", "yoy_prev_x"]]
    df = ev.merge(key, on=["ticker", "period_end"], how="left")
    df["yoy"] = df["revenue_yoy_xbrl"]
    df["accel"] = df["yoy"] - df["yoy_prev_x"]
    df = add_xbrl_eps(df, q, eps)
    df["size_tier"] = df["mcap_pre_b"].apply(size_tier)
    rd = pd.to_datetime(df["reaction_date"])
    df["season"] = rd.dt.year.astype(str) + "Q" + rd.dt.quarter.astype(str)
    df["set"] = label
    return df


def main() -> None:
    df = pd.concat([one("", "core"), one("_broad", "broad")], ignore_index=True)
    # 매출이 아주 작거나(<$20M/분기) YoY가 ±300%를 넘는 값은 비율이 의미를 잃어 자른다
    q_rev = df["yoy"].abs() > 300
    df.loc[q_rev, ["yoy", "accel", "rule40", "op_leverage"]] = np.nan
    df.to_csv(RESULTS / "features_mech.csv", index=False)
    print(df.groupby(["set"]).agg(n=("ticker", "size"), tickers=("ticker", "nunique"),
                                  yoy_ok=("yoy", "count"), accel_ok=("accel", "count"),
                                  eps_ok=("eps_surprise", "count")).to_string())


if __name__ == "__main__":
    main()
