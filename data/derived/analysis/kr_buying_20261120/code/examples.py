"""설명용: 펀드 161종목의 2026 수익(1/2 시가 → 11/18 종가, QQQ 대비)과 2026 평균 한국인 매수 비중(K1 주별 평균).
결과: results/examples_2026.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PX = BASE.parents[0] / "post_earnings_risers_20261120/cache"


def main() -> None:
    p = pd.read_csv(BASE / "cache/panel.csv", parse_dates=["date"])
    f = p[p.fund & (p.date >= "2026-01-01")]
    k = f.groupby("ticker").agg(K1=("K1", "mean"), weeks_present=("K3", lambda s: int((s > 0).sum())), weeks=("K3", "size"))
    ac = pd.read_parquet(PX / "w_adjclose.parquet")
    ac = ac[ac.index <= "2026-11-18"]
    a = ac.loc["2026-01-02":]
    ret = (a.iloc[-1] / a.iloc[0] - 1) * 100
    k["ret_2026"] = ret.reindex(k.index)
    k["ret_x_qqq"] = k.ret_2026 - ret["QQQ"]
    k = k.dropna(subset=["ret_2026"])
    k["K1_tile"] = pd.qcut(k.K1.rank(method="first"), 3, labels=["한국인 적게", "중간", "한국인 많이"])
    k.round(3).to_csv(BASE / "results/examples_2026.csv")
    print("종목 수", len(k), "QQQ 2026", round(ret["QQQ"], 1))
    print(k.groupby("K1_tile", observed=True).agg(n=("ret_2026", "size"), med=("ret_x_qqq", "median"), mean=("ret_x_qqq", "mean"),
                                                   beat=("ret_x_qqq", lambda s: (s > 0).mean())).round(2))
    print("상관(순위) K1 vs 2026 QQQ 대비:", round(k.K1.rank().corr(k.ret_x_qqq.rank()), 3))
    print(k.sort_values("ret_x_qqq", ascending=False).head(12)[["ret_x_qqq", "K1", "weeks_present", "K1_tile"]].round(3).to_string())
    print(k.sort_values("K1", ascending=False).head(10)[["ret_x_qqq", "K1", "weeks_present"]].round(3).to_string())


if __name__ == "__main__":
    main()
