"""반박 검산(N1): (1) 청산을 한국 주문 시간 안(시가)으로 바꾸면, (2) ^TNX 대신 FRED DGS10을 쓰면 판정이 바뀌는가.
n1_lib·n1_run 그대로 쓰고 r{h} 열만 바꾼다. 결과 -> results/critic_n1_exec.csv"""
import pathlib, sys
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "code"))
from n1_lib import attach_controls, conditions, daily_table, onsets  # noqa
import n1_run as R  # noqa

op, cl = R.load("raw")
dgs = pd.read_csv(HERE / "cache" / "critic_dgs10.csv", parse_dates=["observation_date"], na_values=".")
dgs = dgs.set_index("observation_date").DGS10.reindex(R.SES)
tnx = cl["^TNX"]
# 금리 자료 비교
both = pd.DataFrame(dict(t=tnx, f=dgs)).dropna()
print("수준 차 중앙(%p)", round(float((both.t - both.f).median()), 3), "| 일변화 상관", round(float(both.t.diff().corr(both.f.diff())), 3))
s5 = pd.DataFrame(dict(t=np.sign(tnx.diff(5)), f=np.sign(dgs.diff(5)))).dropna()
s5 = s5[s5.index.year >= 2003]
print("2003~ 5일 변화 부호 일치", round(float((s5.t == s5.f).mean()) * 100, 1), "% (n", len(s5), ")")

rows = []
def run(sym, ylds, exitmode):
    first = cl[sym].first_valid_index()
    o, c, t = op[sym].loc[first:], cl[sym].loc[first:], ylds.loc[first:]
    d = daily_table(o, c, t)
    if exitmode == "open_i+5":     # 진입 시가 -> 6번째 세션 시가(5세션 보유, 한국 22:30/23:30 청산)
        for h in (5, 20): d[f"r{h}"] = o.shift(-h) / o - 1
    elif exitmode == "open_i+4":   # Codex open5: 5번째 세션 시가
        for h in (5, 20): d[f"r{h}"] = o.shift(-(h - 1)) / o - 1
    ev = {fam: attach_controls(d, onsets(cond)) for fam, cond in conditions(c, t).items()}
    return d, ev

for ylab, ylds in [("TNX", tnx), ("DGS10", dgs)]:
    for ex in ["close", "open_i+5", "open_i+4"]:
        if ylab == "DGS10" and ex == "open_i+4": continue
        for sym in R.SYMS:
            d, ev = run(sym, ylds, ex)
            for pname in ["2003-23", "2024-26"]:
                y0, y1 = R.PERIODS[pname]
                for h in (5, 20):
                    r = R.cell(d, ev, y0, y1, h)
                    rows.append(dict(yld=ylab, exit=ex, sym=sym, period=pname, h=h, **r))
        print(ylab, ex, "done")
out = pd.DataFrame(rows)
out.to_csv(HERE / "results" / "critic_n1_exec.csv", index=False, encoding="utf-8-sig")
cols = ["yld", "exit", "sym", "period", "h", "n_M", "hit_M", "mean_M", "dhit_b", "dhit_b_ci_lo", "dmean_b", "dmean_b_ci_lo",
        "dhit_C", "dhit_C_ci_lo", "dmean_C", "dmean_C_ci_lo", "dmean_C_ci_hi"]
pd.set_option("display.width", 250)
x = out[cols].copy()
for c_ in cols[6:]: x[c_] = (x[c_] * 100).round(2)
print(x.to_string(index=False))
# 판정 재현
for (yl, ex, sym), g in out[(out.period == "2003-23") & (out.h == 5)].groupby(["yld", "exit", "sym"]):
    r = g.iloc[0]
    ok = all(r[k] > 0 and r[f"{k}_ci_lo"] > 0 for k in ["dhit_b", "dmean_b", "dhit_C", "dmean_C"])
    n0 = sum(r[f"{k}_ci_lo"] > 0 for k in ["dhit_b", "dmean_b", "dhit_C", "dmean_C"])
    print(yl, ex, sym, "통과" if ok else "실패", f"0 위 {n0}/4")
