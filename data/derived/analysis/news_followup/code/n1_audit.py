"""N1 감사: (1) 순수 반복문 독립 재계산(신호 날짜·5일 수익·(b) 대조) (2) 자른 자료 미리보기 점검 (3) 연도별 M−(b) 부호 수.
-> results/n1_audit.csv, results/n1_by_year.csv"""
import pathlib, sys
import numpy as np
import pandas as pd
import exchange_calendars as xcals

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "code"))
from n1_lib import attach_controls, conditions, daily_table, onsets  # noqa: E402

cal = xcals.get_calendar("XNYS", start="1994-01-01", end="2026-12-31")
SES = cal.sessions_in_range("1994-01-03", "2026-10-23").tz_localize(None)
raw = pd.read_parquet(HERE / "cache" / "n1_raw.parquet")
OP = raw.pivot(index="date", columns="sym", values="O").reindex(SES)
CL = raw.pivot(index="date", columns="sym", values="C").reindex(SES)
rows = []
yrows = []
for s in ["SOXX", "QQQ", "SPY"]:
    first = CL[s].first_valid_index()
    o = OP[s].loc[first:].to_list(); c = CL[s].loc[first:].to_list(); t = CL["^TNX"].loc[first:].to_list()
    dates = list(CL[s].loc[first:].index)
    n = len(c)
    # 순수 반복문
    cond = [False] * n
    for k in range(5, n):
        dy = t[k] - t[k - 5]
        cond[k] = (c[k] / c[k - 5] - 1 < 0) and (dy == dy) and (dy < 0)
    ent, prev = [], -100
    for k in range(1, n - 1):
        if cond[k] and not cond[k - 1] and (k + 1) - prev >= 5:
            prev = k + 1; ent.append(k + 1)
    r5 = {dates[i]: c[i + 4] / o[i] - 1 for i in ent if i + 4 < n}
    # 라이브러리
    oo, cc, tt = OP[s].loc[first:], CL[s].loc[first:], CL["^TNX"].loc[first:]
    d = daily_table(oo, cc, tt)
    ev = attach_controls(d, onsets(conditions(cc, tt)["M"]))
    lib = ev.r5.dropna().to_dict()
    same_dates = set(r5) == set(lib)
    maxdiff = max(abs(r5[k] - lib[k]) for k in r5) if same_dates else np.nan
    # (b) 대조 무작위 25개 독립 재계산
    rng = np.random.default_rng(1)
    pick = rng.choice(len(ev.dropna(subset=["b_mean5"])), 25, replace=False)
    bd = []
    for j in pick:
        e = ev.dropna(subset=["b_mean5"]).iloc[j]
        y, b = e.year, e.bucket
        vals = []
        for i in range(20, n - 4):
            if dates[i].year != y: continue
            dd = c[i - 1] / max(c[i - 20:i]) - 1
            bk = 0 if -dd < .05 else (1 if -dd < .10 else 2)
            if bk == b: vals.append(c[i + 4] / o[i] - 1)
        bd.append(abs(np.mean(vals) - e.b_mean5))
    # 자른 자료 미리보기: 2015-06-30 이후를 지운 자료로 신호를 다시 낸다
    cut = pd.Timestamp("2015-06-30")
    ev2 = onsets(conditions(cc.loc[:cut], tt.loc[:cut])["M"])
    full = [x for x in onsets(conditions(cc, tt)["M"]) if dates[x] <= cut]
    trunc_same = [dates[x] for x in ev2 if x < len(dates)] == [dates[x] for x in full][:len(ev2)]
    rows.append(dict(sym=s, n_signals=len(ent), same_dates=same_dates, max_r5_diff=maxdiff,
                     b_ctrl_maxdiff=float(max(bd)), truncated_same=trunc_same,
                     tnx_missing_sessions=int(pd.Series(t).isna().sum())))
    # 연도별 M−(b) 5일 평균 차이
    e = ev[(ev.year >= 2003) & (ev.year <= 2023)].dropna(subset=["r5", "b_mean5"])
    g = e.groupby("year").apply(lambda x: pd.Series(dict(n=len(x), ex_mean=(x.r5 - x.b_mean5).mean(),
                                                          ex_hit=((x.r5 > 0) - x.b_hit5).mean())))
    g["sym"] = s
    yrows.append(g.reset_index())
a = pd.DataFrame(rows); print(a.to_string(index=False))
a.to_csv(HERE / "results" / "n1_audit.csv", index=False, encoding="utf-8-sig")
y = pd.concat(yrows)
y.to_csv(HERE / "results" / "n1_by_year.csv", index=False, encoding="utf-8-sig")
for s, g in y.groupby("sym"):
    print(s, "2003~2023 21해 중 M−(b) 평균 > 0:", int((g.ex_mean > 0).sum()), "· 상승비율 > 0:", int((g.ex_hit > 0).sum()),
          "| 연도별 평균 차이 중앙", round(g.ex_mean.median() * 100, 2), "%p")
print(y.pivot(index="year", columns="sym", values="ex_mean").mul(100).round(2).to_string())
