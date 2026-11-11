"""지금 상태 표: 새로 받은 조건(now/atoms_now.csv)에 강한 조합 겹침 수와 대표 조합을 붙인다.

강한 조합 = 사전 기준 양쪽 통과 62개 + 사후 견고 기준 8개(중복 제외 65개, results/strong_combos.csv).
겹침 수별 과거 벨트 비율은 2024~2026 전체에서 고른 조합이라 설명용이다.
실행: py data/derived/analysis/bottom_zone_20261111/belt/code/now_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BELT = Path(__file__).resolve().parents[1]
RES = BELT / "results"
NOW = BELT / "now"
LOW_DATE, LOW = "2026-11-05", 131.90


def main():
    strong = pd.read_csv(RES / "strong_combos.csv", encoding="utf-8-sig")["조합"].tolist()
    atoms = pd.read_csv(NOW / "atoms_now.csv", parse_dates=["date"], encoding="utf-8-sig").set_index("date")
    panel = pd.read_csv(NOW / "data" / "signal_panel.csv", parse_dates=["date"]).set_index("date")
    px = pd.read_csv(NOW / "data" / "prices.csv", parse_dates=["date"])
    close = px[px.ticker == "SOXX"].set_index("date").sort_index()["close"]
    rows = []
    for d in atoms.loc["2026-11-02":].index:
        r = atoms.loc[d]
        on = [c for c in strong if all(r[p] for p in c.split(" + "))]
        prev_close = close.loc[:d - pd.Timedelta(days=1)].iloc[-1]
        rows.append(dict(
            관찰일=str(d.date()), 직전종가=round(float(prev_close), 2),
            저점대비=round(float(prev_close / LOW - 1), 4) if d > pd.Timestamp(LOW_DATE) else None,
            낙폭60=round(float(panel.loc[d, "dd60"]), 4), SOXL비중백분위=panel.loc[d, "T5_P"],
            공포탐욕=round(float(panel.loc[d, "G"]), 1), 수급원거래일=str(panel.loc[d, "source_date"])[:10],
            대표조합=bool(r["낙폭20"] and r["SOXL비중상위20"]), 강한조합겹침=len(on),
            켜진조건=" ".join(c for c in atoms.columns if r[c])))
    out = pd.DataFrame(rows)
    out.to_csv(NOW / "now_table.csv", index=False, encoding="utf-8-sig")
    stack = pd.read_csv(RES / "strong_stack_daily.csv", parse_dates=["date"]).set_index("date")
    (NOW / "now_table.json").write_text(json.dumps(dict(
        rows=rows, low=dict(date=LOW_DATE, close=LOW),
        last_close=dict(date=str(close.index[-1].date()), close=float(close.iloc[-1])),
        note="강한 조합 65개는 2024~2026 전체로 고른 것. 겹침 수 표는 설명용",
        history_rows=int(len(stack))), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(out.drop(columns="켜진조건").to_string(index=False))
    print(out[["관찰일", "켜진조건"]].to_string(index=False))


if __name__ == "__main__":
    main()
