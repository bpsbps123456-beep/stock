"""dip_gap_limit.csv 위험 쪽 보기(2026-10-11): '날마다 같은 돈'은 거래가 적으면 하루 돈이 한두 종목에 몰린다.
대표 변형마다 날마다 같은 돈 기준의 최악의 날, 누적(단순 합) 최대 낙폭, 매매 있는 날 수를 낸다. 정규장 매수만 따로.
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/trend_zones/code/dip_gap_risk.py"""
import os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402

PICK = ["G 2.0%", "G 2.2%", "G 2.4%", "G 2.5%", "G 3.0%", "G 3.5%", "G 4.0%",
        "L 1.0% 아래 · 취소A", "L 2.0% 아래 · 취소A"]


def main():
    x = pd.read_csv(os.path.join(Zm.OUT, "dip_gap_limit.csv"))
    lines = ["변형 / 날마다 같은 돈: 합 · 매매 있는 날 · 하루 평균 매매 수 · 최악의 날 · 누적 최대 낙폭"]
    for scope, f in (("24시간", lambda z: z), ("정규장 매수만", lambda z: z[z.ses == "정규"])):
        for w in x.win.unique():
            lines += ["", f"== {w} · {scope}"]
            for v in PICK:
                z = f(x[(x.win == w) & (x["var"] == v)])
                g = z.groupby("day").ret
                daily = g.mean()
                cum = daily.cumsum()
                mdd = (cum - cum.cummax()).min()
                lines.append(f"  {v:<18} 합 {daily.sum() * 100:+7.1f}%  {len(daily):3d}일  하루 {g.size().mean():5.1f}건  "
                             f"최악의 날 {daily.min() * 100:+6.1f}%  최대 낙폭 {mdd * 100:+6.1f}%p")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(Zm.OUT, "dip_gap_risk.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)


if __name__ == "__main__":
    main()
