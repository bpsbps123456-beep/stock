"""실적 후 흐름 예측 틀 — 최종 규칙표(시즌별 성적·AI 종목·가짜 사건)와 2026-10 발표 종목의 현재 해당 여부.

실행: py data/derived/analysis/earnings_trend/code/final_rules.py  → results/final_rules.csv, results/live_matches.csv

FINAL은 framework.csv의 통과 묶음 대표 중에서 골랐다. 기준(사람이 고름, 그래서 아래에 적는다):
  ① 2026A·2026B 모두 기준 승률 대비 ±15%p 이상 ② 2026B 건수 20 이상 ③ 가짜 사건(보통 날)에서는 같은 효과가 없거나(실적 고유)
  거래량 3배처럼 보통 날엔 거의 생기지 않는 조건 ④ 같은 뜻의 규칙은 하나만 ⑤ AI·기술주에 쓸 수 있는 것을 먼저.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import window  # noqa: E402
from fetch_sec import RESULTS  # noqa: E402
from fetch_wide import W  # noqa: E402
from rules import predicates  # noqa: E402

FINAL = [
    # (번호, 진입, 기간 열, 규칙, 한 줄 이름)
    ("B1", "E0", "h20_x", "clv_lo & deep_dd & hivol", "바닥 투매형 반등: 고점 −30% 아래 고변동 종목이 실적날 저가 근처로 마감"),
    ("B2", "E0", "h20_x", "large & hivol", "시총 300~2,000억 고변동 종목의 실적 뒤 20일"),
    ("B3", "E0", "h20_x", "r0_up & pre20_up & eps_big", "이미 오르던 종목이 EPS 크게 이기고 적당히(+2~8%) 오름"),
    ("B4", "E0", "nx_x", "vol_hi & tech & hivol", "기술주 고변동 종목의 실적날 거래량 3배 이상(방향 무관)"),
    ("B5", "E0", "nx_x", "eps_big & smallcap & tech", "기술 소형주(20~70억 달러) EPS +15% 이상"),
    ("B6", "E0", "h20_x", "r0_big_down & deep_dd & tech", "고점 −30% 아래 기술주의 실적날 −8% 이상 급락"),
    ("A1", "E5", "e5_nx_x", "fin & d5_up", "금융주가 실적 뒤 5일 SPY 대비 +3% 이상 더 오름"),
    ("A2", "E5", "e5_h20_x", "r0_up & midcap & d5_up", "중형주(70~300억)가 실적날 오르고 5일 더 오름"),
    ("A3", "E0", "h20_x", "clv_lo & vol_lo & lowvol", "저변동 종목이 거래량 없이 실적날 저가 근처 마감"),
    ("A4", "E0", "nx_x", "above_ma & large & consd", "경기소비재 대형주(이평선 위)의 실적 뒤"),
]
SEASONS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3"]


def main() -> None:
    ev = pd.read_csv(RESULTS / "events_wide.csv")
    ev["window"] = ev["reaction_date"].apply(window).fillna("other")
    pl = pd.read_parquet(W / "placebo_wide.parquet")
    pl["window"] = pl["reaction_date"].apply(window).fillna("other")
    fw = pd.read_csv(RESULTS / "framework.csv").set_index(["entry", "target", "rule"])
    rows, live = [], []
    for code, entry, ycol, rule, name in FINAL:
        e5 = entry == "E5"
        P = predicates(ev, e5)
        m = np.logical_and.reduce([P[n][2] for n in rule.split(" & ")])
        y = pd.to_numeric(ev[ycol], errors="coerce")
        row = {"code": code, "name": name, "entry": entry, "target": ycol, "rule": rule,
               "conditions": " + ".join(P[n][1] for n in rule.split(" & "))}
        for s in SEASONS:
            sel = (ev["season"] == s).values & y.notna().values
            base = (y[sel] > 0).mean()
            row[f"{s}_n"] = int((sel & m).sum())
            row[f"{s}_win"] = round(float((y[sel & m] > 0).mean()), 3) if (sel & m).any() else np.nan
            row[f"{s}_base"] = round(float(base), 3)
        for w in ["2025", "2026A", "2026B"]:
            sel = ev["window"].eq(w).values & y.notna().values
            row[f"{w}_n"] = int((sel & m).sum())
            row[f"{w}_win"] = round(float((y[sel & m] > 0).mean()), 3) if (sel & m).any() else np.nan
            row[f"{w}_base"] = round(float((y[sel] > 0).mean()), 3)
            row[f"{w}_median"] = round(float(np.median(y[sel & m])), 2) if (sel & m).any() else np.nan
        sel = ev["window"].str.startswith("2026").values & y.notna().values & (ev["ai_sub"].fillna("") != "").values
        row["ai_2026_n"], row["ai_2026_win"] = int((sel & m).sum()), (round(float((y[sel & m] > 0).mean()), 3) if (sel & m).any() else np.nan)
        f = fw.loc[(entry, ycol, rule)]
        row.update({"placebo_n_2026": f["placebo_n_2026"], "placebo_win_2026": round(f["placebo_win_2026"], 3),
                    "placebo_base_2026": round(f["placebo_base_2026"], 3), "earnings_specific": f["earnings_specific"]})
        rows.append(row)
        # 지금 해당: 2026-10 이후 반응일 사건(E5는 반응 뒤 5거래일이 지난 것만)
        now = (ev["reaction_date"] >= "2026-10-01").values & m
        for _, r in ev[now].iterrows():
            if e5 and pd.isna(r.get("d5_x")):
                continue
            live.append({"code": code, "ticker": r["ticker"], "reaction_date": r["reaction_date"], "ai": r["ai_sub"],
                         "mcap_b": round(r["mcap_b"], 1) if pd.notna(r["mcap_b"]) else np.nan, "r0_x": round(r["r0_x"], 1),
                         "eps_surprise": r["eps_surprise"], "so_far_x": round(r["nx_x"], 1) if pd.notna(r["nx_x"]) else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "final_rules.csv", index=False)
    lv = pd.DataFrame(live)
    lv.to_csv(RESULTS / "live_matches.csv", index=False)
    show = ["code", "rule", "2026A_n", "2026A_win", "2026A_base", "2026B_n", "2026B_win", "2026B_base", "2026B_median",
            "2025_n", "2025_win", "2025_base", "ai_2026_n", "ai_2026_win", "placebo_win_2026", "placebo_base_2026"]
    pd.set_option("display.width", 250)
    print(out[show].to_string(index=False))
    print(out[["code"] + [f"{s}_win" for s in SEASONS]].to_string(index=False))
    print(f"지금 해당 {len(lv)}건, AI 종목 {int((lv['ai'].fillna('') != '').sum()) if len(lv) else 0}건")


if __name__ == "__main__":
    main()
