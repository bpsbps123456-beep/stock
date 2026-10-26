"""box_low_line.pine을 파이썬으로 그대로 옮겨 3분봉(1분봉 합성, 9/10~10/6)과 1시간봉 결과를 맞춰 본다(2026-10-09).
L = 직전 N거래일 정규장 저가 최소, P = L x 0.97, 정규장 봉에서 처음 low <= P → min(open, P) 매수(하루 1번),
11:00 ET 전 매수면 다음 봉부터 +2% 지정가(max(open, tp)), 아니면 매수 뒤 첫 한국 08시 이후 봉 시가(정리 시각이 지난 봉은 목표보다 정리를 먼저 본다).
저가 출처 두 가지를 같이 본다: 'daily' = yfinance 일봉 저가(첫 판), 'intra' = 3분봉 정규장 저가 최소(지금 Pine, 1분봉이 없는 9/9 전 날만 일봉).
사용: py line5d_port.py <dip raw> <일봉 폴더>"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../dip_reason/code"))
from fetch import TV_AI

raw, ddir = sys.argv[1], sys.argv[2]
rows, gaps = [], []
for s in TV_AI + ["TEM"]:
    p = os.path.join(raw, f"1m_{s}.parquet")
    if not os.path.exists(p):
        continue
    m = pd.read_parquet(p)
    b = m.resample("3min", origin="start_day", label="left", closed="left").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    dl = pd.read_parquet(os.path.join(ddir, f"1d_{s}.parquet")).Low
    dl.index = pd.to_datetime(dl.index).date
    t = b.index; o, h, l = b.Open.values, b.High.values, b.Low.values
    mod = t.hour * 60 + t.minute; reg = (mod >= 570) & (mod < 960); day = np.asarray(t.date)
    kst = t.tz_convert("Asia/Seoul"); tk = t.as_unit("ns").asi8
    t8 = kst.normalize() + pd.Timedelta(hours=8)
    deadv = np.where(kst < t8, t8.as_unit("ns").asi8, (t8 + pd.Timedelta(days=1)).as_unit("ns").asi8)
    il = pd.Series(l[reg]).groupby(day[reg]).min()          # 3분봉 정규장 저가 최소
    for D, v in il.items():
        if D in dl.index:
            gaps.append((s, D, v / dl[D] - 1))
    for src in ("daily", "intra"):
        lows = dl.copy()
        if src == "intra":
            lows = pd.concat([dl[[k for k in dl.index if k < il.index.min()]], il])
        Ls = {}
        for D in sorted(set(day)):
            prev = lows[[k for k in lows.index if k < D]]
            Ls[D] = prev.iloc[-5:].min() if len(prev) >= 5 else np.nan
        P = np.array([Ls[D] * 0.97 for D in day])
        hold, ent, ei, tp, dead, last = False, 0.0, 0, np.nan, 0, None
        for i in range(len(o)):
            if hold:
                if tk[i] >= dead:                     # 정리 시각이 먼저(깊이 연구 line5d_depth.py와 같은 순서)
                    rows.append((src, s, t[ei], ent, o[i] / ent - 1, "8시")); hold = False
                elif not np.isnan(tp) and i > ei and h[i] >= tp:
                    rows.append((src, s, t[ei], ent, max(o[i], tp) / ent - 1, "+2%")); hold = False
            elif reg[i] and day[i] != last and not np.isnan(P[i]) and l[i] <= P[i]:
                ent = min(o[i], P[i]); ei = i; tp = ent * 1.02 if mod[i] < 660 else np.nan
                dead, hold, last = deadv[i], True, day[i]
R = pd.DataFrame(rows, columns=["src", "sym", "t", "ent", "ret", "why"])
R = R[(R.t >= "2026-09-10") & (R.t < "2026-10-07")]
g = pd.DataFrame(gaps, columns=["sym", "day", "gap"])
print(f"3분봉 정규장 저가 vs yfinance 일봉 저가(같은 날 {len(g)}개): 같음 {(g.gap.abs() < 1e-4).mean():.0%}, 0.5% 넘게 다름 {(g.gap.abs() > 0.005).mean():.1%}, 최대 {g.gap.abs().max():.2%}")
h1 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/line5d_depth_trades.csv"), parse_dates=["day"])
h1 = h1[(h1.N == 5) & (h1.d == 0.03) & (h1.day >= "2026-09-10") & (h1.day < "2026-10-07")]
for src, r in R.groupby("src", sort=False):
    tv = r[r.sym != "TEM"]
    print(f"[{src}] 3분봉 9/10~10/6 트레이딩뷰 AI 38종목: {len(tv)}회 평균 {tv.ret.mean():+.2%} 승률 {(tv.ret > 0).mean():.0%} 최악 {tv.ret.min():+.1%} 합 {tv.ret.sum():+.2f} | 매도 이유 {tv.why.value_counts().to_dict()}")
    both = set(zip(tv.sym, tv.t.dt.date)) & set(zip(h1.sym, h1.day.dt.date))
    print(f"[{src}] 매수일 겹침: 3분봉 {len(tv)} / 1시간봉 {len(h1)} / 공통 {len(both)}")
    print(f"[{src}] TEM 매수:", [(f"{x.t:%m/%d %H:%M}", round(x.ent, 2), f"{x.ret:+.1%}", x.why) for x in r[r.sym == "TEM"].itertuples()])
print(f"1시간봉 근사 같은 기간: {len(h1)}회 평균 {h1.ret.mean():+.2%} 승률 {(h1.ret > 0).mean():.0%} 합 {h1.ret.sum():+.2f}")
