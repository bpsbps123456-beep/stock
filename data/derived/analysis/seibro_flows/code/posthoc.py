"""사후 설명 계산(BRIEF §9 목록 밖, 판정 아님). 보고서 §5의 같은 주 상관과 재현율 기저율을 다시 만든다."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "describe"


def main() -> None:
    w = pd.read_csv(RES / "weekly_screen.csv")
    w["year"] = w["week"].str[:4]
    out = {}
    for y in ["2024", "2025", "2026"]:
        s = w[w["year"] == y]
        r1 = s["SOXL_rank_net"] == 1
        out[y] = {"corr_S1_sameweek": round(s["S1_w"].corr(s["SOXX_same_week"]), 3),
                  "corr_S1_next1w": round(s["S1_w"].corr(s["SOXX_1w"]), 3), "n": int(len(s)),
                  "rank1_weeks": int(r1.sum()),
                  "mean_sameweek_when_rank1": round(s.loc[r1, "SOXX_same_week"].mean(), 4),
                  "mean_sameweek_when_netsell": round(s.loc[s["SOXL_net_M"] < 0, "SOXX_same_week"].mean(), 4),
                  "mean_next1w_when_rank1": round(s.loc[r1, "SOXX_1w"].mean(), 4),
                  "mean_next1w_all": round(s["SOXX_1w"].mean(), 4)}
    pct = pd.read_parquet(ROOT / "results" / "real" / "pct_daily.parquet")
    pct.index = pd.DatetimeIndex(pct.index)
    p = pct.loc["2024-07-01":"2026-10-23"]
    base = {}
    for s in ["S1", "S3", "S7", "S9n", "C1", "C2"]:
        on = (p[s] >= 80).astype(float).where(p[s].notna())
        base[s] = round(float(on.shift(1).rolling(10, min_periods=1).max().dropna().mean()), 3)
    res = {"weekly_corr": out, "recall_base_rate": base}
    (RES / "sameweek_and_base.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
