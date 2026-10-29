"""실적 뒤 흐름 — '반응일 종가'가 아니라 며칠 늦게(이미 오른 뒤) 들어가면 어떤가(2026-10-29 사용자 질문).

실행: py data/derived/analysis/earnings_trend/code/late_entry.py → results/late_entry.json
B1은 반응일 k에서 j거래일 뒤(j=1·3·4) 종가 진입, 반응 뒤 j일 동안 이미 SPY 대비 얼마나 올랐는지로 나눈다.
결과 = 진입 → 원래 끝(k+20) 남은 기간, 진입 → 진입+20거래일 두 가지. B4는 진입 → 다음 실적 전날.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import window  # noqa: E402
from fetch_sec import RESULTS  # noqa: E402
from fetch_wide import load_ohlcv  # noqa: E402
from rules import predicates  # noqa: E402


def main() -> None:
    ev = pd.read_csv(RESULTS / "events_wide.csv")
    ev["window"] = ev["reaction_date"].apply(window).fillna("other")
    C = load_ohlcv()["Close"]
    spy = C["SPY"].values
    last = len(C) - 1
    P = predicates(ev, e5=False)
    rules = {"B1": "clv_lo & deep_dd & hivol", "B4": "vol_hi & tech & hivol"}
    out = {}
    for code, rule in rules.items():
        m = np.logical_and.reduce([P[n][2] for n in rule.split(" & ")])
        rows = []
        for _, r in ev[m].iterrows():
            if r["ticker"] not in C:
                continue
            c = C[r["ticker"]].values
            k, end = int(r["k"]), int(r["end_k"])

            def x(a: int, b: int) -> float:
                if b > last or a > b or np.isnan(c[a]) or np.isnan(c[b]):
                    return np.nan
                return (c[b] / c[a] - 1) * 100 - (spy[b] / spy[a] - 1) * 100
            for j in (1, 3, 4):
                rows.append({"ticker": r["ticker"], "window": r["window"], "j": j, "runup": x(k, k + j),
                             "rest": x(k + j, k + 20) if code == "B1" else x(k + j, end),
                             "next20": x(k + j, k + j + 20), "full": x(k, k + 20) if code == "B1" else x(k, end)})
        df = pd.DataFrame(rows)
        df = df[df["window"].isin(["2025", "2026A", "2026B"])]
        df["period"] = np.where(df["window"] == "2025", "2025", "2026")
        df["band"] = pd.cut(df["runup"], [-999, 0, 7, 999], labels=["반응 뒤 하락", "0~+7%", "+7% 이상"])
        res = []
        for (per, j, band), g in df.groupby(["period", "j", "band"], observed=True):
            for col in ["rest", "next20"]:
                y = g[col].dropna()
                if len(y):
                    res.append({"rule": code, "period": per, "j": j, "band": str(band), "outcome": col, "n": len(y),
                                "win": round(float((y > 0).mean()), 3), "median": round(float(y.median()), 2)})
        on_time = df[df["j"] == 1].groupby("period")["full"].agg(lambda s: (s.dropna() > 0).mean()).round(3).to_dict()
        out[code] = {"rows": res, "on_time_win": on_time}
    (RESULTS / "late_entry.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for code, v in out.items():
        print("==", code, "제때(반응일 종가) 승률", v["on_time_win"])
        t = pd.DataFrame(v["rows"])
        print(t.pivot_table(index=["period", "j", "band"], columns="outcome", values=["n", "win", "median"]).to_string())


if __name__ == "__main__":
    main()
