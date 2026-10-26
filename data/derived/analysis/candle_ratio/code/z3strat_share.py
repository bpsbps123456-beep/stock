"""z3strat 보조: 추세 판정 28개의 '추세' 봉 비율(고르기·확인, 종목 중앙). 고른 전략이 사실상 늘 애매(= 늘 절반·늘 현금)인지 확인용.

py code/z3strat_share.py → results/z3strat_share_5m.csv
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd

from engine import HERE
from z3strat_run import SYMS
from zones3 import prepare


def work(sym):
    P = prepare(sym, "5m", "dst")
    seg = P["seg"]
    return [dict(sym=sym, clf=k, trend_sel=float(v[seg == 0].mean()), trend_conf=float(v[seg == 1].mean())) for k, v in P["flags"].items()]


if __name__ == "__main__":
    with Pool(3) as pool:
        rows = [r for rs in pool.map(work, SYMS) for r in rs]
    t = pd.DataFrame(rows).groupby("clf")[["trend_sel", "trend_conf"]].median()
    t.to_csv(HERE / "results" / "z3strat_share_5m.csv", float_format="%.4f")
    print(t.round(3).to_string())
