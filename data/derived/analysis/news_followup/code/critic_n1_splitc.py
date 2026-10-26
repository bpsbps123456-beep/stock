"""반박 검산(N1 설명용): 같은 주가 하락 첫 발생(C) 안에서 금리 하락 − 상승 20일 차이가 금리 자료(^TNX·DGS10)·청산 시각에 버티는가.
-> results/critic_n1_splitc.csv"""
import pathlib, sys
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "code"))
from n1_lib import attach_controls, block_boot, conditions, daily_table, onsets  # noqa
import n1_run as R  # noqa
op, cl = R.load("raw")
dgs = pd.read_csv(HERE / "cache" / "critic_dgs10.csv", parse_dates=["observation_date"], na_values=".").set_index("observation_date").DGS10.reindex(R.SES)
rows = []
for ylab, Y in [("TNX", cl["^TNX"]), ("DGS10", dgs)]:
    for ex in ["close", "open_i+5"]:
        for sym in R.SYMS:
            first = cl[sym].first_valid_index()
            o, c, t = op[sym].loc[first:], cl[sym].loc[first:], Y.loc[first:]
            d = daily_table(o, c, t)
            if ex == "open_i+5":
                for h in (5, 20): d[f"r{h}"] = o.shift(-h) / o - 1
            C = attach_controls(d, onsets(conditions(c, t)["C"]))
            g = R.in_period(C, 2003, 2023)
            gf, gr = g[g.dy5_sig < 0], g[g.dy5_sig > 0]
            for h in (5, 20):
                bt = block_boot({"M": gf, "X": gr}, h, R.months_of(d, 2003, 2023), B=5000)
                rows.append(dict(yld=ylab, exit=ex, sym=sym, h=h, n_fall=len(gf), n_rise=len(gr),
                                 dmean=bt["dmean_X"] * 100, lo=bt["dmean_X_ci"][0] * 100, hi=bt["dmean_X_ci"][1] * 100))
out = pd.DataFrame(rows)
out.to_csv(HERE / "results" / "critic_n1_splitc.csv", index=False, encoding="utf-8-sig")
print(out.round(2).to_string(index=False))
