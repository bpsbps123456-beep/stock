"""반박(시점·실행): 시가 품질, ^TNX 품질(결측·멈춤·채권 휴장일·단위). 읽기 전용 검산."""
import pathlib
import numpy as np, pandas as pd, exchange_calendars as xcals
HERE = pathlib.Path(__file__).resolve().parent.parent
raw = pd.read_parquet(HERE / "cache" / "n1_raw.parquet")
cal = xcals.get_calendar("XNYS", start="1994-01-01", end="2026-12-31")
SES = cal.sessions_in_range("1994-01-03", "2026-10-23").tz_localize(None)
P = {k: raw.pivot(index="date", columns="sym", values=k).reindex(SES) for k in "OHLC"}
print("== 시가 품질: O==전일C, O==C, O 밖[L,H] (연도구간별 %)")
for s in ["SOXX", "QQQ", "SPY", "^SOX"]:
    O, H, L, C = (P[k][s] for k in "OHLC")
    ok = C.notna()
    df = pd.DataFrame(dict(eqprev=(O == C.shift(1)), eqC=(O == C), out=((O > H + 1e-9) | (O < L - 1e-9)), eqH=((O==H)&(O==L))))[ok]
    for a, b in [(1994, 2000), (2001, 2002), (2003, 2005), (2006, 2007), (2008, 2012), (2013, 2019), (2020, 2026)]:
        x = df[(df.index.year >= a) & (df.index.year <= b)]
        if len(x): print(s, a, b, len(x), (x.mean() * 100).round(1).to_dict())
t = P["C"]["^TNX"]
print("== ^TNX")
print("NYSE 세션 중 결측", int(t.isna().sum()), "/", len(t))
miss = t[t.isna()].index
print("결측 연도분포", pd.Series(miss.year).value_counts().sort_index().to_dict())
d1 = t.diff()
print("전일과 같은 값(멈춤) 비율 연도구간:")
for a, b in [(1994, 2002), (2003, 2007), (2008, 2013), (2014, 2019), (2020, 2026)]:
    x = d1[(d1.index.year >= a) & (d1.index.year <= b)].dropna()
    print(a, b, len(x), round(float((x == 0).mean() * 100), 2), "%")
print("범위", float(t.min()), float(t.max()), "| 하루 변화 |>0.5| 개수", int((d1.abs() > 0.5).sum()))
print(d1[d1.abs() > 0.4].to_string())
# 채권 휴장일(SIFMA 권고: 콜럼버스·재향군인의 날) 근사: 10월 둘째 월요일, 11/11 평일
def bond_hol(idx):
    out = []
    for dt in idx:
        if dt.month == 10 and dt.weekday() == 0 and 8 <= dt.day <= 14: out.append(dt)
        if dt.month == 11 and dt.day == 11: out.append(dt)
    return pd.DatetimeIndex(out)
bh = bond_hol(SES)
print("채권 휴장(근사) NYSE 세션", len(bh), "중 ^TNX 결측", int(t.reindex(bh).isna().sum()), "멈춤(전일과 같음)", int((d1.reindex(bh) == 0).sum()))
print(pd.DataFrame(dict(tnx=t.reindex(bh), d=d1.reindex(bh))).tail(12).to_string())
