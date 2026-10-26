"""반박 검산(N1): 기간별 SOXX M−(b) 5일(‘2020년 이후에만 컸다’)이 금리 자료·청산 시각에 버티는가 -> results/critic_n1_periods.csv"""
import pathlib, sys
import pandas as pd
HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "code"))
from n1_lib import attach_controls, conditions, daily_table, onsets, summarize  # noqa
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
                d["r5"] = o.shift(-5) / o - 1
            ev = {k: attach_controls(d, onsets(v)) for k, v in conditions(c, t).items()}
            for p in ["2003-07", "2008-09", "2010-19", "2020-23", "2024-26"]:
                y0, y1 = R.PERIODS[p]
                sM = summarize(R.in_period(ev["M"], y0, y1), 5); sC = summarize(R.in_period(ev["C"], y0, y1), 5)
                rows.append(dict(yld=ylab, exit=ex, sym=sym, period=p, n=sM["n"], M_minus_b=(sM["mean"] - sM["b_mean"]) * 100,
                                 M_minus_C=(sM["mean"] - sC["mean"]) * 100))
out = pd.DataFrame(rows)
out.to_csv(HERE / "results" / "critic_n1_periods.csv", index=False, encoding="utf-8-sig")
print(out.pivot_table(index=["sym", "period"], columns=["yld", "exit"], values="M_minus_b").round(2).to_string())
