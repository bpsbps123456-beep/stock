"""실적 전날 옵션 흐름 → 반응일 시험(사전 등록 BRIEF.md 그대로).

실행: py data/derived/analysis/earnings_optflow/code/run.py  → results/main.json, results/cells.csv, results/events.csv
      py data/derived/analysis/earnings_optflow/code/run.py --forward  → results/forward.json
        (보고서 '다음 확인': 반응일 2026-10-29~2026-12-31, 데스크 실적 뒤 흐름 원장 `data/derived/desk/erules.json` 사건)
자료: earnings_trend/results/events_wide.csv(사건·반응일 수익), options_sentiment 캐시의 OCC 일별 거래량
      (먼저 `py data/derived/analysis/options_sentiment/code/fetch.py --since <날짜> --until <날짜>`로 채운다.
       --until 없이 돌리면 그 연구의 Yahoo 캐시를 짧게 덮어쓰니 꼭 붙인다).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ANALYSIS = HERE.parents[1]
RES = HERE.parent / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ANALYSIS / "options_sentiment" / "code"))
from fetch import load_occ  # noqa: E402

from desk.news import market_calendar  # noqa: E402

MIN_SIDES = 1000  # 기준일 전체(양쪽 합) 거래량 하한 ≈ 500계약
NORMAL_DAYS, NORMAL_MIN = 20, 10
PERIODS = {"2026H1": ("2026-01-01", "2026-06-30"), "2026H2": ("2026-07-01", "2026-10-28"),
           "2025": ("2024-11-15", "2025-12-31")}
VARS = {"V1": "전체 풋 비중", "V2": "고객 풋 비중", "V3": "평소 대비 풋 비중"}
OUTS = {"Y1": ("r0_x", "반응일 − SPY"), "Y2": ("r0", "반응일 수익")}
LARGE_B = 10.0
FORWARD = ("2026-10-29", "2026-12-31")  # 결과 보기 전에 정한 확인 구간(반응일)
HYPOTHESIS = {("SNDK", "2026-10-30"), ("INTC", "2026-10-30"), ("RDDT", "2026-10-30"), ("AMZN", "2026-10-30"),
              ("AXTI", "2026-10-30"), ("TWLO", "2026-10-30"), ("MSTR", "2026-10-30"), ("ILMN", "2026-10-30"),
              ("COIN", "2026-10-30"), ("FSLR", "2026-10-30")}  # 가설을 만든 10건은 확인에서 뺀다


def forward_events() -> pd.DataFrame:
    led = json.loads((ROOT / "data" / "derived" / "desk" / "erules.json").read_text(encoding="utf-8"))
    rows = []
    for ev in led["events"].values():
        f = ev.get("features") or {}
        r = ev.get("reaction") or ""
        if ev.get("status") != "ok" or not FORWARD[0] <= r <= FORWARD[1] or f.get("r0_x") is None:
            continue
        if (ev["ticker"], r) not in HYPOTHESIS:
            rows.append({"ticker": ev["ticker"], "reaction_date": r, "season": "forward", "mcap_b": f.get("mcap_b"),
                         "sector": ev.get("sector"), "r0": f.get("r0"), "r0_x": f["r0_x"]})
    out = pd.DataFrame(rows)
    cal = market_calendar()
    out["base_date"] = [cal.previous_session(d).strftime("%Y-%m-%d") for d in out["reaction_date"]]
    return out


def forward() -> None:
    ev = forward_events()
    fl = flows(sorted(ev["ticker"].unique()))
    df = ev.merge(fl[["ticker", "base_date", "tot", "V1"]], on=["ticker", "base_date"], how="left")
    df = df[df["tot"] >= MIN_SIDES]
    ic = season_ic(df, "V1", "r0_x")
    large = season_ic(df[df["mcap_b"] >= LARGE_B], "V1", "r0_x")
    q = quintile_spread(df, "V1", "r0_x")
    checks = {"ic_negative_p_lt_0.05": ic["ic"] is not None and ic["ic"] < 0 and ic["p_neg"] < 0.05,
              "quintile_spread_le_-1pp": q["spread"] <= -1.0, "large_cap_negative": large["ic"] is not None and large["ic"] < 0}
    out = {"window": FORWARD, "events": len(ev), "used": len(df), "last_reaction": df["reaction_date"].max(),
           "occ_to": fl["base_date"].max(), "ic": ic, "large": large, "quintile": q, "checks": checks,
           "pass": all(checks.values())}
    (RES / "forward.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


def events() -> pd.DataFrame:
    ev = pd.read_csv(ANALYSIS / "earnings_trend" / "results" / "events_wide.csv",
                     usecols=["ticker", "date", "reaction_date", "season", "mcap_b", "sector", "r0", "r0_x"])
    cal = market_calendar()
    ev["base_date"] = [cal.previous_session(d).strftime("%Y-%m-%d") for d in ev["reaction_date"]]
    return ev


def flows(tickers: list[str]) -> pd.DataFrame:
    occ = load_occ(tickers)
    occ["P"] = occ["C_P"] + occ["F_P"] + occ["M_P"]
    occ["C"] = occ["C_C"] + occ["F_C"] + occ["M_C"]
    occ["tot"] = occ["P"] + occ["C"]
    occ["V1"] = occ["P"] / occ["tot"].where(occ["tot"] > 0)
    occ["V2"] = occ["C_P"] / (occ["C_P"] + occ["C_C"]).where(occ["C_P"] + occ["C_C"] > 0)
    occ = occ.sort_values(["underlying", "date"])
    ok = occ["V1"].where(occ["tot"] >= MIN_SIDES)
    prior = ok.groupby(occ["underlying"]).transform(lambda s: s.shift(1).rolling(NORMAL_DAYS, min_periods=NORMAL_MIN).median())
    occ["V3"] = occ["V1"] - prior
    return occ.rename(columns={"underlying": "ticker", "date": "base_date"})


def season_ic(df: pd.DataFrame, v: str, y: str) -> dict:
    rows = []
    for _, g in df.dropna(subset=[v, y]).groupby("season"):
        if len(g) < 30:
            continue
        rho = spearmanr(g[v], g[y]).statistic
        rows.append((len(g), float(rho)))
    if not rows:
        return {"n": 0, "ic": None, "p_neg": None, "seasons": 0}
    n = np.array([r[0] for r in rows])
    rho = np.array([r[1] for r in rows])
    z = np.arctanh(np.clip(rho, -0.999, 0.999)) * np.sqrt(n - 3)
    big_z = z.sum() / np.sqrt(len(z))
    return {"n": int(n.sum()), "ic": float((rho * n).sum() / n.sum()), "p_neg": float(norm.cdf(big_z)),
            "seasons": len(rows), "by_season": [round(r, 4) for r in rho]}


def quintile_spread(df: pd.DataFrame, v: str, y: str) -> dict:
    d = df.dropna(subset=[v, y]).copy()
    d["q"] = d.groupby("season")[v].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    hi, lo = d.loc[d["q"] == 4, y], d.loc[d["q"] == 0, y]
    return {"spread": float(hi.mean() - lo.mean()), "put_mean": float(hi.mean()), "call_mean": float(lo.mean()),
            "n_each": int(min(len(hi), len(lo)))}


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    ev = events()
    fl = flows(sorted(ev["ticker"].unique()))
    df = ev.merge(fl[["ticker", "base_date", "tot", "V1", "V2", "V3"]], on=["ticker", "base_date"], how="left")
    cover = {"events": len(ev), "with_occ": int(df["tot"].notna().sum()), "occ_from": fl["base_date"].min(),
             "occ_to": fl["base_date"].max()}
    df = df[df["tot"] >= MIN_SIDES]
    cover["above_min_volume"] = len(df)
    period = pd.Series(None, index=df.index, dtype=object)
    for name, (a, b) in PERIODS.items():
        period[(df["reaction_date"] >= a) & (df["reaction_date"] <= b)] = name
    df["period"] = period
    df = df.dropna(subset=["period"])
    df.to_csv(RES / "events.csv", index=False)
    cells = []
    for pname in PERIODS:
        for v in VARS:
            for yk, (y, _) in OUTS.items():
                sub = df[df["period"] == pname]
                cells.append({"period": pname, "var": v, "out": yk, **season_ic(sub, v, y), **quintile_spread(sub, v, y)})
        sub = df[(df["period"] == pname) & (df["mcap_b"] >= LARGE_B)]
        cells.append({"period": pname + "_large", "var": "V1", "out": "Y1", **season_ic(sub, "V1", "r0_x"),
                      **quintile_spread(sub, "V1", "r0_x")})
    y26 = df[df["period"].isin(["2026H1", "2026H2"])]
    whole = season_ic(y26, "V1", "r0_x")
    large = season_ic(y26[y26["mcap_b"] >= LARGE_B], "V1", "r0_x")
    cell = {c["period"]: c for c in cells if c["var"] == "V1" and c["out"] == "Y1"}
    checks = {"1_both_halves_negative": all(cell[p]["ic"] is not None and cell[p]["ic"] < 0 for p in ("2026H1", "2026H2")),
              "2_pooled_p_lt_0.01": whole["p_neg"] is not None and whole["p_neg"] < 0.01,
              "3_quintile_spread_le_-1pp": all(cell[p]["spread"] <= -1.0 for p in ("2026H1", "2026H2")),
              "4_large_cap_negative": large["ic"] is not None and large["ic"] < 0}
    out = {"coverage": cover, "n_by_period": df["period"].value_counts().to_dict(), "main_2026": whole,
           "large_2026": large, "checks": checks, "pass": all(checks.values()), "cells": cells}
    (RES / "main.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    pd.DataFrame(cells).drop(columns=["by_season"], errors="ignore").to_csv(RES / "cells.csv", index=False)
    print(json.dumps({k: out[k] for k in ("coverage", "n_by_period", "main_2026", "large_2026", "checks", "pass")},
                     ensure_ascii=False, indent=1, default=float))
    print(pd.DataFrame(cells)[["period", "var", "out", "n", "ic", "p_neg", "spread", "put_mean", "call_mean"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    forward() if "--forward" in sys.argv[1:] else main()
