"""line5d_depth 보완: 공정한 비교 기준과 계좌 복리(2026-10-09).
기준 = 같은 38종목·같은 기간의 모든 정규장 1시간봉 시가에서 산 경우의 평균(같은 매도 규칙). 날짜를 고르지 않은 '아무 때나 사기' 평균이다.
계좌 = 종목마다 자금을 따로 두고 그 깊이 선의 매매를 복리로 굴린 수익(38개 같은 비중 평균)과 최악 낙폭. 사용: py line5d_fair.py <1시간봉 폴더>"""
import os, sys
import numpy as np, pandas as pd

hdir = sys.argv[1]
h = pd.read_parquet(os.path.join(hdir, "h1.parquet"))
t = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/line5d_depth_trades.csv"), parse_dates=["day"])
t = t[t.N == 5]
# 기준: 모든 정규장 1시간봉 시가 매수(같은 매도 규칙)
rows = []
for s, x in h.groupby("sym"):
    x = x.sort_index(); tt = x.index
    o, hi = x.Open.values, x.High.values
    mod = tt.hour * 60 + tt.minute
    reg = (mod >= 570) & (mod < 960)
    kst = tt.tz_convert("Asia/Seoul"); tk = tt.as_unit("ns").asi8
    t8 = kst.normalize() + pd.Timedelta(hours=8)
    dead = np.where(kst < t8, t8.as_unit("ns").asi8, (t8 + pd.Timedelta(days=1)).as_unit("ns").asi8)
    for i in np.where(reg & (tt.year == 2026))[0]:
        ent = o[i]; tp = ent * 1.02 if mod[i] < 660 else None; px = np.nan
        for k in range(i + 1, len(o)):
            if tk[k] >= dead[i]: px = o[k]; break
            if tp is not None and hi[k] >= tp: px = max(o[k], tp); break
        if not np.isnan(px):
            rows.append((s, tt[i].date(), px / ent - 1))
b = pd.DataFrame(rows, columns=["sym", "day", "ret"]); b["day"] = pd.to_datetime(b.day)
b["반기"] = np.where(b.day < "2026-07-01", "1~6월", "7~10월")
base = b.groupby("반기").ret.mean()
print("기준(아무 날·아무 정규장 시각에 산 평균, 같은 매도):", {k: f"{v:+.2%}" for k, v in base.items()})
out = []
for (hv, d), v in t.groupby(["반기", "d"]):
    comp, mdds = [], []
    for s, w in v.sort_values("day").groupby("sym"):
        eq = np.cumprod(1 + w.ret.values)
        comp.append(eq[-1] - 1); mdds.append((eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min())
    nsym = t.sym.nunique()
    comp += [0.0] * (nsym - len(comp))      # 그 기간 매매가 없던 종목은 0%
    out.append(dict(반기=hv, 깊이=f"-{d:.0%}", 건수=len(v), 평균=v.ret.mean(), 기준대비=v.ret.mean() - base[hv],
                    승률=(v.ret > 0).mean(), 최악=v.ret.min(), 계좌복리=np.mean(comp), 최악낙폭=min(mdds)))
r = pd.DataFrame(out)
for c, f in (("평균", "{:+.2%}"), ("기준대비", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("계좌복리", "{:+.1%}"), ("최악낙폭", "{:+.1%}")):
    r[c] = r[c].map(f.format)
pd.set_option("display.width", 200)
print("5일 저가 기준, 트레이딩뷰 AI 38종목, 2026\n" + r.to_string(index=False))
