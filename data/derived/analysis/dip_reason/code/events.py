"""이격 매수(보라선 EMA 60 -2%, 3분봉) 거래를 재현하고, 매수 순간에 이미 알 수 있는 '이유 있는 하락' 신호를 붙인다.
매수·매도 규칙은 dip_scalper.pine과 같다: 저가가 EMA[1]x0.98에 닿으면 min(시가, 매수선)에 사고,
RSI(14)>70 다음 봉 시가 또는 매수 뒤 첫 아침 8시(한국) 이후 첫 봉 시가에 판다. 24시간(프리·정규·애프터) 매매로 본다.
yfinance에는 주간거래(20~04시 ET) 봉이 없다.

신호(모두 매수 직전 봉까지의 값):
  grp_dev  같은 그룹 다른 종목들의 보라선 대비 위치 중앙값(그룹 = watchlist.AI_UNIVERSE 대분류, TEM은 바이오·유전체)
  qqq_dev  QQQ의 보라선 대비 위치
  rvol     최근 30분 거래량 / 지난 10거래일 같은 시각 30분 평균
  d_prev   매수가 / 가장 최근 정규장 종가 - 1(애프터는 당일 종가)
  d_high   매수가 / 오늘(ET 날짜) 4시 이후 최고가 - 1
  speed    최근 30분 수익률
  run20    전일까지 20거래일 수익률
사용: py events.py <raw 폴더> <3|5> <출력 parquet>"""
import os, sys
import numpy as np, pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../kalman_zones/code"))   # osc.rsi(Pine과 같은 RSI)만 쓴다
from osc import rsi
from fetch import universe

GAP, X, HR = 0.02, 70, 8


def bars(raw, s, mins):
    """1분봉 → 3분봉(시계 기준 정렬), 5분봉은 그대로."""
    if mins == 5:
        b = pd.read_parquet(f"{raw}/5m_{s}.parquet")
    else:
        m = pd.read_parquet(f"{raw}/1m_{s}.parquet")
        b = m.resample(f"{mins}min", label="left", closed="left").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
    return b[b.High > 0]


def prep(b, ema_len, rsi_len):
    et = b.index
    mod = et.hour * 60 + et.minute
    d = pd.DataFrame(index=et)
    d["o"], d["h"], d["l"], d["c"], d["v"] = b.Open.values, b.High.values, b.Low.values, b.Close.values, b.Volume.values
    d["ema"] = b.Close.ewm(span=ema_len, adjust=False).mean().values
    d["dev"] = d.c / d.ema - 1
    d["r"] = rsi(b.Close.values.astype(float), rsi_len)
    d["mod"] = mod
    d["ses"] = np.where((mod >= 570) & (mod < 960), "정규", np.where(mod < 570, "프리", "애프터"))
    d["date"] = et.date
    return d


def sim(d):
    """dip_scalper.pine sim3()과 같은 상태 기계. 반환: (매수 봉, 매도 봉, 매수가, 매도가, 이유)."""
    o, l, r, ema = d.o.values, d.l.values, d.r.values, d.ema.values
    kst = d.index.tz_convert("Asia/Seoul")
    tk = kst.as_unit("ns").asi8
    out, hold, ent, ei, dead = [], False, 0.0, 0, 0
    for i in range(61, len(d)):
        if hold:
            if i - 1 >= ei and r[i - 1] > X:
                out.append((ei, i, ent, o[i], "RSI")); hold = False
            elif tk[i] >= dead:
                out.append((ei, i, ent, o[i], "8시")); hold = False
        else:
            lvl = ema[i - 1] * (1 - GAP)
            if l[i] <= lvl:
                ent, ei, hold = min(o[i], lvl), i, True
                t8 = kst[i].normalize() + pd.Timedelta(hours=HR)
                dead = (t8 if kst[i] < t8 else t8 + pd.Timedelta(days=1)).as_unit("ns").value
    return out


def tod_rvol(d, k=10, days=10):
    """최근 k봉 거래량 합 / 지난 days거래일 같은 시각 k봉 합의 평균(오늘 제외). 매수 봉 직전까지."""
    v = d.v.rolling(k, min_periods=k).sum().shift(1).where(d.ses == "정규")   # yfinance 프리·애프터 거래량은 0이라 정규장만
    tab = v.groupby([d.date, d["mod"]]).last().unstack()          # 날짜 x 시각
    base = tab.shift(1).rolling(days, min_periods=5).mean()        # 지난 날들 같은 시각 평균
    b = base.stack().rename("base")
    key = pd.MultiIndex.from_arrays([d.date, d["mod"]])
    return (v.values / b.reindex(key).values)


def daily_feats(raw, s):
    dd = pd.read_parquet(f"{raw}/1d_{s}.parquet")
    c = dd.Close
    f = pd.DataFrame({"close": c, "prev_close": c.shift(1), "run20": c.shift(1) / c.shift(21) - 1})
    f.index = pd.to_datetime(f.index).date
    return f


def build(raw, mins):
    ema_len, rsi_len = (60, 14) if mins == 3 else (36, 9)        # 5분봉은 같은 180분 EMA, 약 45분 RSI로 근사
    grp = universe()
    D = {}
    for s in list(grp) + ["QQQ"]:
        try:
            D[s] = prep(bars(raw, s, mins), ema_len, rsi_len)
        except FileNotFoundError:
            pass
    # 공통 시각 격자에서 각 종목 보라선 대비 위치(직전 봉 값, 최대 30분 앞당겨 채움)
    grid = pd.DatetimeIndex(sorted(set().union(*[d.index for d in D.values()])))
    lim = 30 // mins
    devs = pd.DataFrame({s: d.dev.reindex(grid).ffill(limit=lim) for s, d in D.items()})
    rows = []
    for s, g in grp.items():
        if s not in D:
            continue
        d = D[s]
        peers = [p for p, q in grp.items() if q == g and p != s and p in devs]
        gdev = devs[peers].median(axis=1).shift(1) if peers else pd.Series(np.nan, index=grid)
        qdev = devs["QQQ"].shift(1)
        rv = tod_rvol(d, 30 // mins)
        hi = d.groupby("date").h.cummax().shift(1)                  # 직전 봉까지 오늘 최고가
        sp = d.c.shift(1) / d.c.shift(1 + 30 // mins) - 1
        df = daily_feats(raw, s)
        for ei, xi, ent, px, why in sim(d):
            t = d.index[ei]
            dt = d.date.iloc[ei]
            pc = (df.close if d.ses.iloc[ei] == "애프터" else df.prev_close).get(dt, np.nan)   # 가장 최근 정규장 종가
            rows.append(dict(
                sym=s, grp=g, t=t, date=dt, ses=d.ses.iloc[ei], ent=ent, px=px, why=why,
                ret=px / ent - 1, mae=d.l.values[ei:xi + 1].min() / ent - 1, hold_min=(xi - ei) * mins,
                grp_dev=gdev.get(t, np.nan), qqq_dev=qdev.get(t, np.nan), rvol=rv[ei],
                d_prev=ent / pc - 1, d_high=ent / hi.iloc[ei] - 1 if d.date.iloc[ei - 1] == dt else np.nan,
                speed=sp.iloc[ei], run20=df.run20.get(dt, np.nan)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raw, mins, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    ev = build(raw, mins)
    ev.to_parquet(out)
    print(f"{mins}분봉 거래 {len(ev)}건 · 종목 {ev.sym.nunique()} · {ev.t.min()} -> {ev.t.max()}")
    print(ev[ev.sym == "TEM"].tail(5).to_string())
