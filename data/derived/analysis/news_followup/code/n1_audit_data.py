"""N1 자료 점검: 달력 대비 누락, 시가 이상(0·고저 밖·전일 종가와 같음), 금리 단위 단절·급변."""
import pathlib
import numpy as np
import pandas as pd
import exchange_calendars as xcals

HERE = pathlib.Path(__file__).resolve().parent.parent
raw = pd.read_parquet(HERE / "cache" / "n1_raw.parquet")
cal = xcals.get_calendar("XNYS", start="1994-01-01", end="2026-12-31")
ses = cal.sessions_in_range("1994-01-03", "2026-10-23").tz_localize(None)
print("XNYS sessions", len(ses), ses[0].date(), ses[-1].date())
rows = []
for s, g in raw.groupby("sym"):
    g = g.set_index("date").sort_index()
    first = g.index[0]
    sub = ses[ses >= first]
    miss = sub.difference(g.index)
    extra = g.index.difference(ses)
    o, h, l, c = g.O, g.H, g.L, g.C
    pc = c.shift(1)
    bad0 = (o <= 0) | o.isna()
    outside = (o > h * 1.001) | (o < l * 0.999)
    same_prev = (o - pc).abs() < 1e-9
    flat = (o == h) & (h == l) & (l == c)
    for yr0, yr1 in [(1994, 2002), (2003, 2007), (2008, 2013), (2014, 2023), (2024, 2026)]:
        m = (g.index.year >= yr0) & (g.index.year <= yr1)
        if m.sum() == 0: continue
        rows.append(dict(sym=s, years=f"{yr0}-{yr1}", n=int(m.sum()),
                         miss=int(((miss.year >= yr0) & (miss.year <= yr1)).sum()),
                         extra=int(((extra.year >= yr0) & (extra.year <= yr1)).sum()),
                         open_bad=int(bad0[m].sum()), open_outside_HL=int(outside[m].sum()),
                         open_eq_prevclose=round(float(same_prev[m].mean()), 3), ohlc_flat=int(flat[m].sum())))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(HERE / "results" / "n1_audit_data.csv", index=False, encoding="utf-8-sig")
t = raw[raw.sym == "^TNX"].set_index("date").C.sort_index()
d1 = t.diff()
print("TNX range by year (min,max):")
print(t.groupby(t.index.year).agg(["min", "max"]).T.round(2).to_string())
print("TNX |1d change| > 0.35:", d1[d1.abs() > 0.35].round(3).to_dict())
sox = raw[raw.sym == "^SOX"].set_index("date").sort_index()
soxx = raw[raw.sym == "SOXX"].set_index("date").sort_index()
j = pd.concat([sox.C.pct_change(5), soxx.C.pct_change(5)], axis=1, keys=["SOX", "SOXX"]).dropna()
for y0, y1 in [(2001, 2007), (2008, 2016), (2017, 2026)]:
    jj = j[(j.index.year >= y0) & (j.index.year <= y1)]
    print(y0, y1, "corr 5d ret SOX vs SOXX", round(jj.corr().iloc[0, 1], 4), "sign agree", round(float((np.sign(jj.SOX) == np.sign(jj.SOXX)).mean()), 4))
