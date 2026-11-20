"""조기 실전(판정 아님): 반응일 2026-10-22~11-12 사건을 다음 날 시가 → 11-18 종가(최대 +20)까지 QQQ 대비로 본다.
결과를 본 뒤 자주 나온 단일 조건(관심 적음) 두 개만 본다. → results/live_check.csv
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "code"))
import rules as R  # noqa: E402


def main() -> None:
    d = R.load()
    px = {k: pd.read_parquet(BASE / f"cache/px_{k}.parquet") for k in ("open", "close", "adjclose")}
    idx = px["close"]["QQQ"].dropna().index
    px = {k: v.reindex(idx) for k, v in px.items()}
    ac, ao = px["adjclose"], px["open"] * px["adjclose"] / px["close"]
    n = len(idx)
    rd = d.reaction_date.astype(str)
    live = d[(rd >= "2026-10-22") & (rd <= "2026-11-12")].copy()
    ys = []
    for e in live.itertuples():
        k = int(e.k)
        end = min(k + 20, n - 2)  # yfinance 11-19 행이 72종목 비어 11-18 종가까지
        ys.append(100 * ((ac[e.ticker].iloc[end] / ao[e.ticker].iloc[k + 1] - 1) - (ac["QQQ"].iloc[end] / ao["QQQ"].iloc[k + 1] - 1)))
    live["y_live"] = ys
    rows = []
    for nm, m in (("전체", np.ones(len(live), bool)), ("발표 전 옵션 거래 급증 없음(occ_tot_ab≤0.276)", (live.occ_tot_ab <= 0.276).to_numpy()),
                  ("반응일 거래량 적음(px_vol0≤2.215)", (live.px_vol0 <= 2.215).to_numpy())):
        y = live.y_live[m].dropna()
        rows.append({"조건": nm, "건수": int(len(y)), "QQQ보다 못 간 비율": round(float((y < 0).mean()), 3),
                     "중앙 초과수익": round(float(y.median()), 2)})
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "results/live_check.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
