"""큰 봉(60·120·160·240분, 일봉) RSI를 3분·5분 기준봉의 매 시점에 붙인다. Pine ta.rsi와 같은 와일더 RMA다.
  dev   진행 중 값: 마지막 완성 봉까지의 RMA 상태에 '지금 가격'을 현재 봉 종가로 넣은 RSI(트레이딩뷰 실시간 값과 같음)
  conf  완성 값: 마지막으로 완성된 큰 봉의 RSI
분봉 큰 봉은 5분봉(프리·애프터 포함)으로 만들고 미 동부 04:00에 맞춘다(하루 1440분이 60·120·160·240으로 나누어떨어짐).
일봉은 정규장 일봉이다. 트레이딩뷰처럼 프리장에는 어제 일봉, 정규장에는 진행 중 일봉, 애프터에는 오늘 완성 일봉을 본다."""
import numpy as np, pandas as pd


def rma_state(c, n):
    """종가 배열 → (평균 상승, 평균 하락, RSI) 배열. 첫 n개 변화의 단순평균으로 시작(Pine과 같음)."""
    m = len(c)
    g, l, r = np.full(m, np.nan), np.full(m, np.nan), np.full(m, np.nan)
    ch = np.diff(c, prepend=np.nan)
    up, dn = np.maximum(ch, 0), np.maximum(-ch, 0)
    if m <= n:
        return g, l, r
    ag, al = up[1:n + 1].mean(), dn[1:n + 1].mean()
    g[n], l[n] = ag, al
    for k in range(n + 1, m):
        ag = (ag * (n - 1) + up[k]) / n
        al = (al * (n - 1) + dn[k]) / n
        g[k], l[k] = ag, al
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(l == 0, 100.0, 100 - 100 / (1 + g / l))
    r[np.isnan(g)] = np.nan
    return g, l, r


def _rsi_from(ag, al, prev, p, n):
    """완성 봉 상태(ag, al, 직전 종가)에 지금 가격 p를 넣은 진행 중 RSI."""
    ch = p - prev
    g = (ag * (n - 1) + np.maximum(ch, 0)) / n
    l = (al * (n - 1) + np.maximum(-ch, 0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(l == 0, 100.0, 100 - 100 / (1 + g / l))
    out[np.isnan(ag)] = np.nan
    return out


def intraday_bars(m5, tf):
    b = m5.resample(f"{tf}min", origin="start_day", offset="4h", label="left", closed="left").agg(
        {"Close": "last"}).dropna()
    return b.index, b.Close.values


def rsi_intraday(m5, tf, n, t, p):
    """기준봉 시각 t(봉 시작)·가격 p에서 tf분 큰 봉 RSI (dev, conf)."""
    hs, hc = intraday_bars(m5, tf)
    g, l, r = rma_state(hc, n)
    anchor = t.normalize() + pd.Timedelta(hours=4)
    off = ((t - anchor) // pd.Timedelta(minutes=tf)).astype(np.int64)
    cur = anchor + pd.to_timedelta(off * tf, unit="min")         # 지금 속한 큰 봉의 시작
    j = np.searchsorted(hs.as_unit("ns").asi8, cur.as_unit("ns").asi8, side="left") - 1   # 마지막으로 완성된 큰 봉(시간 단위를 ns로 맞춤)
    ok = j >= 0
    jj = np.where(ok, j, 0)
    dev = np.where(ok, _rsi_from(g[jj], l[jj], hc[jj], p, n), np.nan)
    conf = np.where(ok, r[jj], np.nan)
    return dev, conf


def rsi_daily(d1, n, t, p, ses):
    """정규장 일봉 RSI. 프리: 어제 완성값 / 정규: 진행 중 값 / 애프터: 오늘 완성값. conf는 지금 시점에 완성된 마지막 일봉."""
    days = pd.to_datetime(d1.index).date
    c = d1.Close.values
    g, l, r = rma_state(c, n)
    today = np.asarray(t.date)
    jt = np.searchsorted(days, today, side="left")                 # 오늘 일봉 위치(있다면)
    has_today = (jt < len(days)) & (days[np.minimum(jt, len(days) - 1)] == today)
    jy = jt - 1                                                    # 어제(완성) 일봉
    after = ses == "애프터"
    jc = np.where(after & has_today, jt, jy)                       # 완성된 마지막 일봉
    okc = jc >= 0
    jc0 = np.where(okc, jc, 0)
    conf = np.where(okc, r[jc0], np.nan)
    oky = jy >= 0
    jy0 = np.where(oky, jy, 0)
    devr = np.where(oky, _rsi_from(g[jy0], l[jy0], c[jy0], p, n), np.nan)
    dev = np.where(ses == "정규", devr, conf)
    return dev, conf


def ema_state(c, n):
    """Pine ta.ema와 같은 EMA(첫 n개 단순평균으로 시작, 그 전은 nan)."""
    e = np.full(len(c), np.nan)
    if len(c) < n:
        return e
    a = 2 / (n + 1)
    e[n - 1] = c[:n].mean()
    for k in range(n, len(c)):
        e[k] = a * c[k] + (1 - a) * e[k - 1]
    return e


def ema_intraday(m5, tf, lens, t, p):
    """기준봉 시각 t·가격 p에서 tf분 큰 봉 EMA들. {n: (dev, conf)} — dev는 지금 가격을 진행 중 봉 종가로 넣은 값."""
    hs, hc = intraday_bars(m5, tf)
    anchor = t.normalize() + pd.Timedelta(hours=4)
    off = ((t - anchor) // pd.Timedelta(minutes=tf)).astype(np.int64)
    cur = anchor + pd.to_timedelta(off * tf, unit="min")
    j = np.searchsorted(hs.as_unit("ns").asi8, cur.as_unit("ns").asi8, side="left") - 1
    ok, jj = j >= 0, np.maximum(j, 0)
    out = {}
    for n in lens:
        e = ema_state(hc.astype(float), n)
        a = 2 / (n + 1)
        conf = np.where(ok, e[jj], np.nan)
        out[n] = (a * p + (1 - a) * conf, conf)
    return out
