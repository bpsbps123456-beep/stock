"""실적 뒤 흐름 — 2026년에 강한 규칙만 엄선(2026-10-29 사용자 요청 "2026년에만 강하게 쓸 수 있는 것만 남기자").

실행: py data/derived/analysis/earnings_trend/code/select2026.py → results/curated_2026.csv, results/curated_2026.json

대상: 이미 2026 전방 검증(상반기 고르기 → 7~9월 확인)을 통과한 규칙 325개(rules_passed.csv).
진입: 사용자가 실제로 살 수 있는 '다음 날 정규장 시가'로 다시 잰다(미국 종가는 한국 05~06시라 주문 시간 밖).
  E0 규칙 = 반응일 다음 날 시가 진입, E5 규칙 = 반응 6거래일째 시가 진입(5일 뒤 종가를 보고 다음 날 시가).
  끝은 원래 정의와 같다(h20 = 반응일+20 / E5는 +25 거래일 종가, nx = 다음 실적 반응일 전날 종가).
엄선 기준(결과를 보기 전에 정함 — 2025는 보지 않고 적기만 한다):
  ① 2026 건수 ≥ 60, 2026 세 시즌(Q1·Q2·Q3) 각각 ≥ 12건
  ② 2026 전체 승률 − 기준 승률 ≥ +15%p(매수) 또는 ≤ −15%p(회피)
  ③ 세 시즌 각각 같은 쪽으로 ≥ 8%p
  ④ 2026 중앙 초과수익 ≥ +2%(매수) 또는 ≤ −2%(회피)
  ⑤ 생존 편향 점검: 사건 시점 시총 100억 달러 이상 2026 사건 ≥ 15건이고 같은 쪽으로 ≥ 8%p
  ⑥ 보통 날에도 같은 효과가 나는 규칙(framework.csv earnings_specific == False) 제외, 지금 기준 목록인 'ai' 조건 제외
  ⑦ 사건이 40% 넘게 겹치면 세 시즌 최소 차이가 큰 하나만
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

SEASONS = ["2026Q1", "2026Q2", "2026Q3"]
REAL = {("E0", "h20_x"): "o_h20", ("E0", "nx_x"): "o_nx", ("E5", "e5_h20_x"): "o5_h20", ("E5", "e5_nx_x"): "o5_nx"}
CRIT = {"n26": 60, "n_season": 12, "lift": 0.15, "season_lift": 0.08, "median": 2.0, "n_big": 15, "big_lift": 0.08, "jaccard": 0.4}


def realistic_outcomes(ev: pd.DataFrame) -> pd.DataFrame:
    o = load_ohlcv()
    C, O = o["Close"], o["Open"]
    sc, so = C["SPY"].values, O["SPY"].values
    last = len(C) - 1
    cols = {k: [] for k in ["o_h20", "o_nx", "o5_h20", "o5_nx"]}
    for _, r in ev.iterrows():
        t = r["ticker"]
        if t not in C:
            for k in cols:
                cols[k].append(np.nan)
            continue
        c, op = C[t].values, O[t].values
        k, end = int(r["k"]), int(r["end_k"])

        def x(a: int, b: int) -> float:
            if a > last or b > last or b <= a or np.isnan(op[a]) or np.isnan(c[b]) or op[a] <= 0:
                return np.nan
            return (c[b] / op[a] - 1) * 100 - (sc[b] / so[a] - 1) * 100
        cols["o_h20"].append(x(k + 1, k + 20))
        cols["o_nx"].append(x(k + 1, end))
        cols["o5_h20"].append(x(k + 6, k + 25))
        cols["o5_nx"].append(x(k + 6, end))
    return ev.assign(**cols)


def stats(y: np.ndarray, sel: np.ndarray, m: np.ndarray) -> tuple[int, float, float, float]:
    ok = sel & ~np.isnan(y)
    base = float((y[ok] > 0).mean()) if ok.any() else np.nan
    s = ok & m
    return int(s.sum()), (float((y[s] > 0).mean()) if s.any() else np.nan), base, (float(np.median(y[s])) if s.any() else np.nan)


def main() -> None:
    ev = pd.read_csv(RESULTS / "events_wide.csv")
    ev["window"] = ev["reaction_date"].apply(window).fillna("other")
    ev = realistic_outcomes(ev)
    rules = pd.read_csv(RESULTS / "rules_passed.csv")
    fw = pd.read_csv(RESULTS / "framework.csv")
    regime_only = set(fw.loc[fw["earnings_specific"] == False, "rule"])  # noqa: E712 — 보통 날에도 같은 효과
    big = (ev["mcap_b"] >= 10).values
    y26 = ev["window"].str.startswith("2026").values
    P = {e: predicates(ev, e5=(e == "E5")) for e in ("E0", "E5")}
    rows, masks = [], {}
    for _, r in rules.iterrows():
        col = REAL[(r["entry"], r["target"])]
        y = ev[col].to_numpy(dtype=float)
        m = np.logical_and.reduce([P[r["entry"]][n][2] for n in r["rule"].split(" & ")])
        side = 1 if r["side"] == "매수" else -1
        n26, w26, b26, med26 = stats(y, y26, m)
        row = {"entry": r["entry"], "target": r["target"], "real": col, "side": r["side"], "rule": r["rule"],
               "conditions": " + ".join(P[r["entry"]][n][1] for n in r["rule"].split(" & ")),
               "n_2026": n26, "win_2026": w26, "base_2026": b26, "lift_2026": w26 - b26, "median_2026": med26}
        for s in SEASONS:
            n, w, b, _ = stats(y, (ev["season"] == s).values, m)
            row.update({f"n_{s}": n, f"win_{s}": w, f"base_{s}": b, f"lift_{s}": (w - b) if n else np.nan})
        n, w, b, _ = stats(y, y26 & big, m)
        row.update({"n_big": n, "win_big": w, "lift_big": (w - b) if n else np.nan})
        n, w, b, med = stats(y, (ev["window"] == "2025").values, m)
        row.update({"n_2025": n, "win_2025": w, "base_2025": b, "median_2025": med})
        row["min_season_lift"] = min(side * row[f"lift_{s}"] for s in SEASONS) if all(row[f"n_{s}"] for s in SEASONS) else np.nan
        ok = (n26 >= CRIT["n26"] and all(row[f"n_{s}"] >= CRIT["n_season"] for s in SEASONS)
              and side * row["lift_2026"] >= CRIT["lift"] and row["min_season_lift"] >= CRIT["season_lift"]
              and side * med26 >= CRIT["median"]
              and row["n_big"] >= CRIT["n_big"] and side * row["lift_big"] >= CRIT["big_lift"]
              and r["rule"] not in regime_only and "ai" not in r["rule"].split(" & "))
        row["pass"] = bool(ok)
        rows.append(row)
        masks[(r["entry"], r["target"], r["rule"])] = m & y26
    df = pd.DataFrame(rows).drop_duplicates(["entry", "target", "rule"])
    # 겹침 묶기: 통과 규칙끼리, 세 시즌 최소 차이가 큰 순
    keep, picked = [], []
    for _, r in df[df["pass"]].sort_values("min_season_lift", ascending=False).iterrows():
        m = masks[(r["entry"], r["target"], r["rule"])]
        if any(((m & pm).sum() / max((m | pm).sum(), 1)) >= CRIT["jaccard"] for pm in picked):
            continue
        picked.append(m)
        keep.append(r)
    cur = pd.DataFrame(keep)
    df.to_csv(RESULTS / "curated_2026_all.csv", index=False)
    cur.to_csv(RESULTS / "curated_2026.csv", index=False)
    summary = {"criteria": CRIT, "candidates": len(df), "pass": int(df["pass"].sum()), "after_dedup": len(cur),
               "fail_reasons": {
                   "n": int(((df["n_2026"] < CRIT["n26"]) | df[[f"n_{s}" for s in SEASONS]].lt(CRIT["n_season"]).any(axis=1)).sum()),
                   "lift": int((df.apply(lambda r: (1 if r["side"] == "매수" else -1) * r["lift_2026"], axis=1) < CRIT["lift"]).sum()),
                   "season": int((df["min_season_lift"] < CRIT["season_lift"]).sum())}}
    (RESULTS / "curated_2026.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    show = ["entry", "target", "side", "rule", "n_2026", "win_2026", "base_2026", "median_2026"] + [f"win_{s}" for s in SEASONS] + \
           ["n_big", "win_big", "win_2025", "base_2025"]
    print(json.dumps(summary, ensure_ascii=False))
    print(cur[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
