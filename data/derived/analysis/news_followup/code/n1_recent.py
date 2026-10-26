"""N1 보조(AGENTS 연구 보고 규칙): 2026 1~5월(P1)·6~10월(P2)·2025(Y25) 구간별 M. 대조 (b)는 같은 구간 + 같은 낙폭 구간 날.
-> results/n1_recent.csv"""
import pathlib, sys
import numpy as np
import pandas as pd
import exchange_calendars as xcals

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "code"))
from n1_lib import attach_controls, block_boot, conditions, daily_table, onsets, summarize  # noqa: E402

cal = xcals.get_calendar("XNYS", start="1994-01-01", end="2026-12-31")
SES = cal.sessions_in_range("1994-01-03", "2026-10-23").tz_localize(None)
raw = pd.read_parquet(HERE / "cache" / "n1_raw.parquet")
OP = raw.pivot(index="date", columns="sym", values="O").reindex(SES)
CL = raw.pivot(index="date", columns="sym", values="C").reindex(SES)
WIN = {"Y25": ("2025-01-01", "2025-12-31"), "P1_2026-01~05": ("2026-01-01", "2026-05-31"), "P2_2026-06~10": ("2026-06-01", "2026-10-23")}
rows = []
for s in ["SOXX", "QQQ", "SPY"]:
    first = CL[s].first_valid_index()
    o, c, t = OP[s].loc[first:], CL[s].loc[first:], CL["^TNX"].loc[first:]
    d = daily_table(o, c, t)
    for w, (a, b) in WIN.items():
        d2 = d.copy()
        d2["year"] = np.where((d2.index >= a) & (d2.index <= b), 1, 0)  # 구간 안 = 1 (대조 묶음 열쇠로 재사용)
        ev = {fam: attach_controls(d2, onsets(cond)) for fam, cond in conditions(c, t).items()}
        g = {k: v[v.year == 1] for k, v in ev.items()}
        for h in [5, 20]:
            sM, sC, sD = (summarize(g[k], h) for k in ["M", "C", "D"])
            bt = block_boot(g, h, np.unique(d2[d2.year == 1].month), B=5000)
            rows.append(dict(sym=s, window=w, h=h, n_M=sM["n"], wins_M=sM["wins"], hit_M=sM["hit"], mean_M=sM["mean"],
                             b_hit=sM["b_hit"], b_mean=sM["b_mean"], n_C=sC["n"], hit_C=sC["hit"], mean_C=sC["mean"],
                             n_D=sD["n"], hit_D=sD["hit"], mean_D=sD["mean"], dmean_b=bt["dmean_b"],
                             dmean_b_lo=bt["dmean_b_ci"][0], dmean_b_hi=bt["dmean_b_ci"][1], dmean_C=bt["dmean_C"]))
df = pd.DataFrame(rows)
df.to_csv(HERE / "results" / "n1_recent.csv", index=False, encoding="utf-8-sig")
f = df.copy()
for col in ["hit_M", "mean_M", "b_hit", "b_mean", "hit_C", "mean_C", "hit_D", "mean_D", "dmean_b", "dmean_b_lo", "dmean_b_hi", "dmean_C"]:
    f[col] = (f[col] * 100).round(2)
print(f.to_string(index=False))
