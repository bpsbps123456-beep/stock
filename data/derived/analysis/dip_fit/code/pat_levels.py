"""이격매수매도 — 이전 가격대로 신호 뒤 반등을 미리 알 수 있었나(2026-10-10 사용자 질문).

질문: 10/9 CRCL은 정규장 신호 뒤 빠르게 반등했고 MRVL은 계속 빠졌다. LITE는 테마 급등 뒤 되돌림에서 회복했고 COHR는 못 했다.
      체결가가 이전에 의미 있던 가격대(전날 종가·저가·고가, 오늘 시가·프리장 저가, 5일 저가, 거래 많이 쌓인 가격대, 스윙 저점,
      VWAP, 라운드 넘버) 위에 있었는지로 미리 가를 수 있었나.

결과 변수(시험 전에 정함, 바꾸지 않음)
  Y1 = 거래 수익 ret(지금 매도 규칙: RSI14>70 다음 봉 시가 또는 매수 뒤 첫 한국 08시 이후 봉 시가)
  Y2 = 빠른 반등: 매수 뒤 20봉(60분) 안에 RSI 매도(why == RSI and bars <= 20). 5분봉 보조 기간은 12봉(60분).
사건 연구(테마 급등 뒤 되돌림, LITE·COHR 질문용, 따로 정의)
  급등: 그날 04:00 ET부터 봉 j까지 최고가 / 전날 정규장 종가 - 1 = G >= +5%
  사건 봉 j: 정규장 09:30~14:57 시작 3분봉 중 처음으로 종가 c_j <= 전날 종가 x (1 + G/2)(오른 폭의 절반 이상 되돌림). 종목·날짜당 1건.
  판단 시점 = 봉 j+1 시가, 가격 e = c_j. 특성은 봉 j까지 값만.
  Z1 = 그날 정규장 마지막 3분봉 종가 / e - 1,  Z2 = 봉 j+1~장 마감 사이 종가 최고가 >= e + (그때까지 최고가 - e) / 2(빠진 폭 절반 회복)
  기간 9/10~10/8(완성된 날). 10/9는 10:18 ET까지만 있어 사례 출력만.

특성(매수 봉 i 시가 시점에 알 수 있는 값, 체결가 ent는 써도 됨; 봉 i의 고가·저가·종가·거래량은 안 씀)
  L = 가장 최근 끝난 정규장(애프터 매수는 그날, 나머지는 전날). 일봉(1d)은 L까지.
  pc ph pl pdo   ent / L 종가·고가·저가·시가 - 1
  l5 h5          ent / L까지 5일 최저가·최고가 - 1,  pos20 = 20일 범위 안 위치(0=저점, 1=고점)
  dl             ent / 오늘(04시~) 봉 i-1까지 최저가 - 1(음수 = 신저가)
  pml pmh        ent / 오늘 프리장 저가·고가 - 1(정규장 매수만)
  op             ent / 오늘 정규장 시가 - 1(정규·애프터 매수)
  vw             ent / 오늘 정규장 VWAP(봉 i-1까지, (h+l+c)/3 x 거래량) - 1
  vw5            ent / 5일 VWAP(L까지 5거래일 정규장 5분봉 + 오늘 정규장 봉 i-1까지) - 1
  vp5r vp20r     5·20거래일 정규장 5분봉 거래량 가격 분포(봉 거래량을 고가~저가에 고르게 나눔)에서 ent ±1% 안 몫 /
                 그 기간 가격 범위에 고르게 퍼졌을 때 기대 몫. 1보다 크면 거래가 쌓인 가격대, 0에 가까우면 빈 가격대.
  poc5           ent / 5일 최다 거래 가격(0.5% 칸) - 1
  sw_n sw_b sw_a 스윙 저점(좌우 15분: 5분봉 3봉, 3분봉 5봉, 왼쪽은 엄격히 큼): L까지 5거래일 정규장 5분봉 + 오늘 정규장 봉(i-1까지 확정분).
                 sw_n = ent ±0.5% 안 스윙 저점 수(받친 횟수), sw_b = ent 아래 가장 가까운 스윙 저점까지 거리, sw_a = 위(깨진) 저점까지 거리
  rnd_up rnd_dn  굵은 라운드(간격 = 10^floor(log10 ent)/2: 83→5달러, 310→50달러) 바로 위·바로 아래 0.5% 안
  rnd_f          잔 라운드(간격 = 10^floor(log10 ent)/10) ±0.2% 안
  통제용: atr_d(L까지 일봉 ATR14/종가), vol20(9/9까지 20일 변동성, features.csv), open30(09:30~09:59 매수)

체결 점검: 프리·애프터 매수 중 '직전 1분 종가보다 1% 넘게 아래로 튄 체결'(꼬리)은 뺀다. 체결 분 = 매수 3분봉 안 저가가 ent 이하가 된 첫 1분봉,
  직전 1분 종가 = 그 앞 1분봉 종가. ent < 직전 1분 종가 x 0.99면 꼬리.
고르기·확인: 정규장이 주 범위. 문턱(앞 절반 3분위)·방향은 앞 절반(9/10~9/23)에서 정하고 뒤 절반(9/24~10/8)에 그대로.
  t = 날짜 군집(날짜마다 A평균 - B평균의 평균 / 표준오차). 앞 절반 정규장에서 |t| 상위 서로 다른 특성 3개를 후보로 뒤에서 확인.
  판정(미리 정함): 뒤 t(앞 방향) >= 2이고 시간대·변동성 통제·회귀·상위 1종목/1일 빼기에서 같은 방향(t >= 1)이면 확인됨,
  뒤 t < 1이면 뒤에서 실패, 그 사이 또는 통제에서 꺾이면 약함.
보조: 7/14~9/8 5분봉 근사(stop_lab.load('early'), EMA36·RSI9) 정규장 매수에 같은 문턱·방향.
입력: ../results/trades.csv, ../results/features.csv, scratchpad raw_1009(1m·5m·1d), raw(앞 기간 5m·1d). 다시 받지 않음.
출력: ../results/pat_levels.txt(요약), pat_levels_tests.csv(모든 시험), pat_levels_trades.csv(거래별 특성),
      pat_levels_events.csv(사건 연구), pat_levels_event_tests.csv
사용: py pat_levels.py"""
import os, sys, time
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
from events import bars, prep
import build
import stop_lab

RAW = build.RAW
RAW_E = stop_lab.RAW
RES = os.path.join(HERE, "../results")
ET = "America/New_York"
MID = 20260924
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def dint_of(dates):
    return np.array([x.year * 10000 + x.month * 100 + x.day for x in dates], dtype=np.int64)


def pivots(l, dint, rth, k):
    """스윙 저점: l[j] < min(l[j-k:j]) and l[j] <= min(l[j+1:j+k+1]), 창 전체가 같은 날 정규장."""
    s = pd.Series(l)
    left = s.shift(1).rolling(k, min_periods=k).min().values
    right = s[::-1].shift(1).rolling(k, min_periods=k).min()[::-1].values
    ok = pd.Series(rth.astype(float)).rolling(2 * k + 1, center=True, min_periods=2 * k + 1).sum().values == 2 * k + 1
    same = np.zeros(len(l), bool)
    n = len(l)
    if n > 2 * k:
        same[k:n - k] = dint[:n - 2 * k] == dint[2 * k:]
    return (l < left) & (l <= right) & ok & same


class Struct:
    """일봉 + 지난 정규장 5분봉 구조(거래량 분포·스윙 저점·VWAP)."""

    def __init__(self, raw, s, cut_dint=None, cut_ts=None):
        dd = pd.read_parquet(f"{raw}/1d_{s}.parquet")
        dd.index = pd.to_datetime(dd.index)
        if cut_dint is not None:                       # 미래 참조 점검: 이 날짜까지만 남김
            dd = dd[dint_of(dd.index.date) <= cut_dint]
        self.dint = dint_of(dd.index.date)
        self.O, self.H, self.L, self.C = (dd[c].values.astype(float) for c in ["Open", "High", "Low", "Close"])
        pc = np.r_[np.nan, self.C[:-1]]
        tr = np.nanmax(np.vstack([self.H - self.L, np.abs(self.H - pc), np.abs(self.L - pc)]), axis=0)
        self.atrp = pd.Series(tr / self.C).rolling(14).mean().values
        m = pd.read_parquet(f"{raw}/5m_{s}.parquet")
        if cut_ts is not None:
            m = m[m.index < cut_ts]
        mod = m.index.hour * 60 + m.index.minute
        m = m[(mod >= 570) & (mod < 960)]
        self.md = dint_of(m.index.date)
        self.ml, self.mh, self.mc, self.mv = (m[c].values.astype(float) for c in ["Low", "High", "Close", "Volume"])
        self.mtyp = (self.ml + self.mh + self.mc) / 3
        self.udays, self.ust = np.unique(self.md, return_index=True)
        self.uen = np.r_[self.ust[1:], len(self.md)]
        self.piv = pivots(self.ml, self.md, np.ones(len(self.md), bool), 3)

    def sess(self, Ld, n):
        p = np.searchsorted(self.udays, Ld, side="right")
        if p == 0:
            return slice(0, 0), 0
        a = max(0, p - n)
        return slice(self.ust[a], self.uen[p - 1]), p - a


class Today:
    """매수 판단용 봉(3분봉 또는 5분봉) 배열."""

    def __init__(self, idx, o, h, l, c, v, k):
        self.idx = idx
        self.o, self.h, self.l, self.c, self.v = o, h, l, c, v
        self.mod = (idx.hour * 60 + idx.minute).values
        self.dint = dint_of(idx.date)
        self.rth = (self.mod >= 570) & (self.mod < 960)
        self.typ = (h + l + c) / 3
        self.k = k
        self.piv = pivots(l, self.dint, self.rth, k)


def vprof(l, h, v, ent, band=0.01):
    if len(v) == 0 or v.sum() <= 0:
        return np.nan, np.nan
    blo, bhi = ent * (1 - band), ent * (1 + band)
    rngb = h - l
    w = np.where(rngb > 0, np.clip(np.minimum(h, bhi) - np.maximum(l, blo), 0, None) / np.where(rngb > 0, rngb, 1),
                 ((l >= blo) & (l <= bhi)).astype(float))
    share = (v * w).sum() / v.sum()
    rng = h.max() - l.min()
    exp = min(1.0, (bhi - blo) / rng) if rng > 0 else 1.0
    typ = (l + h) / 2
    b = np.floor(np.log(typ) / np.log(1.005))
    vb = pd.Series(v).groupby(b).sum()
    poc = np.exp((vb.idxmax() + 0.5) * np.log(1.005))
    return share / exp, poc


def level_feats(st, T, i, ent, ses):
    f = {}
    D = T.dint[i] if i < len(T.dint) else T.dint[i - 1]
    aft = ses == "애프터"
    p = np.searchsorted(st.dint, D, side="right" if aft else "left") - 1
    if p < 20:
        return f
    C, H, Lo, O = st.C, st.H, st.L, st.O
    f.update(pc=ent / C[p] - 1, ph=ent / H[p] - 1, pl=ent / Lo[p] - 1, pdo=ent / O[p] - 1,
             l5=ent / Lo[p - 4:p + 1].min() - 1, h5=ent / H[p - 4:p + 1].max() - 1,
             pret=C[p] / C[p - 1] - 1, atr_d=st.atrp[p])
    lo20, hi20 = Lo[p - 19:p + 1].min(), H[p - 19:p + 1].max()
    f["pos20"] = (ent - lo20) / (hi20 - lo20) if hi20 > lo20 else np.nan
    Ld = st.dint[p]
    ds = np.searchsorted(T.dint, D, "left")
    seg = np.arange(ds, i)
    if len(seg):
        f["dl"] = ent / T.l[seg].min() - 1
    pm = seg[T.mod[seg] < 570]
    rth = seg[T.rth[seg]]
    if ses == "정규" and len(pm):
        f["pml"] = ent / T.l[pm].min() - 1
        f["pmh"] = ent / T.h[pm].max() - 1
    if ses != "프리":
        r0 = rth[0] if len(rth) else (i if ses == "정규" and i < len(T.o) else None)
        if r0 is not None:
            f["op"] = ent / T.o[r0] - 1
        if len(rth) and T.v[rth].sum() > 0:
            f["vw"] = ent / ((T.typ[rth] * T.v[rth]).sum() / T.v[rth].sum()) - 1
    s5, n5 = st.sess(Ld, 5)
    s20, n20 = st.sess(Ld, 20)
    if n5 >= 5:
        f["vp5r"], poc = vprof(st.ml[s5], st.mh[s5], st.mv[s5], ent)
        f["poc5"] = ent / poc - 1 if poc == poc else np.nan
        num, den = (st.mtyp[s5] * st.mv[s5]).sum(), st.mv[s5].sum()
        if ses == "정규" and len(rth):
            num, den = num + (T.typ[rth] * T.v[rth]).sum(), den + T.v[rth].sum()
        f["vw5"] = ent / (num / den) - 1 if den > 0 else np.nan
        lows = st.ml[s5][st.piv[s5]]
        if ses == "정규" and len(rth):
            pj = rth[T.piv[rth] & (rth + T.k <= i - 1)]
            lows = np.r_[lows, T.l[pj]]
        f["sw_n"] = int((np.abs(lows / ent - 1) <= 0.005).sum())
        b, a = lows[lows <= ent], lows[lows > ent]
        f["sw_b"] = ent / b.max() - 1 if len(b) else np.nan
        f["sw_a"] = a.min() / ent - 1 if len(a) else np.nan
    if n20 >= 20:
        f["vp20r"], _ = vprof(st.ml[s20], st.mh[s20], st.mv[s20], ent)
    mag = 10 ** np.floor(np.log10(ent))
    stp = mag / 2
    lo = np.floor(ent / stp) * stp
    f["rnd_up"] = float(ent / lo - 1 <= 0.005)
    f["rnd_dn"] = float((lo + stp) / ent - 1 <= 0.005)
    fs = mag / 10
    f["rnd_f"] = float(abs(ent / (np.round(ent / fs) * fs) - 1) <= 0.002)
    return f


def today_from_prep(d, k):
    return Today(d.index, d.o.values.astype(float), d.h.values.astype(float), d.l.values.astype(float),
                 d.c.values.astype(float), d.v.values.astype(float), k)


def tail_flag(m1, t, ent):
    """프리·애프터 꼬리 체결: 체결 분의 직전 1분 종가보다 1% 넘게 아래."""
    w = m1.index.searchsorted(t)
    e = m1.index.searchsorted(t + pd.Timedelta(minutes=3))
    lw = m1.Low.values
    for k in range(w, e):
        if lw[k] <= ent * (1 + 1e-9):
            if k == 0:
                return np.nan, np.nan
            pcl = m1.Close.values[k - 1]
            return float(ent < pcl * 0.99), ent / pcl - 1
    return np.nan, np.nan


# ---------------- 거래 특성 ----------------

def late_trades():
    t = pd.read_csv(os.path.join(RES, "trades.csv"))
    t["ts"] = pd.to_datetime(t.t, utc=True).dt.tz_convert(ET)
    t["dint"] = dint_of(t.ts.dt.date)
    t["mod"] = t.ts.dt.hour * 60 + t.ts.dt.minute
    t["y1"] = t.ret
    t["y2"] = ((t.why == "RSI") & (t.bars <= 20)).astype(float)
    f = pd.read_csv(os.path.join(RES, "features.csv"))
    t = t.merge(f[["sym", "vol20"]], on="sym", how="left")
    t["vol20"] = t.vol20.fillna(f.vol20.median())
    rows = []
    for s, g in t.groupby("sym"):
        d = prep(bars(RAW, s, 3), 60, 14)
        T = today_from_prep(d, 5)
        st = Struct(RAW, s)
        m1 = pd.read_parquet(os.path.join(RAW, f"1m_{s}.parquet"))
        for k, r in g.iterrows():
            i = d.index.get_indexer([r.ts])[0]
            assert i > 0, (s, r.ts)
            x = level_feats(st, T, i, r.ent, r.ses)
            x["tail"], x["jump1m"] = tail_flag(m1, r.ts, r.ent) if r.ses != "정규" else (0.0, np.nan)
            x["_k"] = k
            rows.append(x)
    F = pd.DataFrame(rows).set_index("_k")
    return t.join(F)


def early_trades():
    S = stop_lab.load("early")
    b = stop_lab.run(S, None, "early")
    b = b[b.ses == "정규"].copy()
    f = pd.read_csv(os.path.join(RES, "features.csv"))
    b = b.merge(f[["sym", "vol20"]], on="sym", how="left")
    b["vol20"] = b.vol20.fillna(f.vol20.median())
    rows = []
    for s, g in b.groupby("sym"):
        A = S[s]
        T = Today(A["idx"], A["o"], A["h"], A["l"], A["c"], A["v"], 3)
        try:
            st = Struct(RAW_E, s)
        except FileNotFoundError:
            continue
        for k, r in g.iterrows():
            x = level_feats(st, T, int(r.ei), r.ent, r.ses)
            x["_k"] = k
            rows.append(x)
    F = pd.DataFrame(rows).set_index("_k")
    b = b.join(F)
    b["ts"] = pd.DatetimeIndex(b.t).tz_convert(ET) if hasattr(b.t.iloc[0], "tz") else pd.to_datetime(b.t, utc=True).dt.tz_convert(ET)
    b["dint"] = dint_of(pd.DatetimeIndex(b.ts).date)
    b["mod"] = pd.DatetimeIndex(b.ts).hour * 60 + pd.DatetimeIndex(b.ts).minute
    b["y1"] = b.ret
    b["y2"] = ((b.why == "RSI") & (b.bars <= 12)).astype(float)
    b["tail"] = 0.0
    return b


# ---------------- 통계 ----------------

def tstat(x):
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return (x.mean() if n else np.nan), np.nan, n
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(n)), n


def daydiff(d, a, y, strat=None):
    """날짜마다(층이 있으면 날짜·층마다) A평균 - B평균 → 층 평균 → 날짜 t."""
    keys = [d.dint] + ([strat] if strat is not None else [])
    g = d.groupby(keys + [a])[y].mean().unstack()
    if True not in g.columns or False not in g.columns:
        return np.nan, np.nan, 0
    x = (g[True] - g[False]).dropna()
    if strat is not None:
        x = x.groupby(level=0).mean()
    return tstat(x)


def fe_reg(d, a, y, covs=None):
    """y ~ A + 통제(기본: 개장30분 + log(atr_d) + vol20), 날짜 고정효과, 날짜 군집 표준오차."""
    if covs is None:
        covs = ["open30", "latr", "vol20"]
    d = d.assign(open30=(d["mod"] < 600).astype(float), latr=np.log(d.atr_d),
                 lG=np.log(d["G"]) if "G" in d else np.nan)
    k = d[covs + [y]].notna().all(axis=1)
    d, a = d[k], a[k]
    X = np.column_stack([a.astype(float)] + [d[c].values.astype(float) for c in covs])
    Y = d[y].values.astype(float)
    g = d.dint.values
    Xd = X - pd.DataFrame(X).groupby(g).transform("mean").values
    Yd = Y - pd.Series(Y).groupby(g).transform("mean").values
    try:
        XtX = np.linalg.inv(Xd.T @ Xd)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    b = XtX @ Xd.T @ Yd
    e = Yd - Xd @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    G = 0
    for gg in np.unique(g):
        m = g == gg
        u = Xd[m].T @ e[m]
        meat += np.outer(u, u)
        G += 1
    V = XtX @ meat @ XtX * G / max(G - 1, 1)
    return b[0], b[0] / np.sqrt(V[0, 0]) if V[0, 0] > 0 else np.nan


def drop_worst(d, a, y, col, sgn):
    best = None
    for v in d[col].unique():
        k = d[col] != v
        dd, tt, _ = daydiff(d[k], a[k], y)
        if np.isnan(dd):
            continue
        if best is None or sgn * dd < sgn * best[1]:
            best = (v, dd, tt)
    return best if best else (None, np.nan, np.nan)


FEATS = [("pc", "sign"), ("ph", "sign"), ("pl", "sign"), ("op", "sign"), ("pml", "sign"), ("dl", "sign"), ("l5", "sign"),
         ("vw", "sign"), ("vw5", "sign"), ("poc5", "sign"), ("pos20", "pos"), ("vp5r", "vp"), ("vp20r", "vp"),
         ("sw_n", "count"), ("sw_b", "cont"), ("sw_a", "cont"), ("rnd_up", "bin"), ("rnd_dn", "bin"), ("rnd_f", "bin")]
NAME = dict(pc="전날 종가 대비", ph="전날 고가 대비", pl="전날 저가 대비", op="오늘 시가 대비", pml="오늘 프리장 저가 대비",
            dl="오늘 저가 대비(음수=신저가)", l5="5일 저가 대비", vw="오늘 VWAP 대비", vw5="5일 VWAP 대비", poc5="5일 최다거래가 대비",
            pos20="20일 범위 위치", vp5r="5일 거래 밀도", vp20r="20일 거래 밀도", sw_n="±0.5% 스윙저점 수", sw_b="아래 스윙저점 거리",
            sw_a="위(깨진) 스윙저점 거리", rnd_up="굵은 라운드 바로 위", rnd_dn="굵은 라운드 바로 아래", rnd_f="잔 라운드 근처",
            pdo="전날 시가 대비", pmh="오늘 프리장 고가 대비", h5="5일 고가 대비", G="급등 폭", pret="전날 수익률")


def splits(feat, kind, d1):
    x = d1[feat].dropna()
    out = []
    if len(x) < 20:
        return out
    if kind == "sign":
        out.append(("음수", f"{feat}<0", lambda z, c=feat: z[c] < 0))
    if kind == "pos":
        out.append(("<0.5", f"{feat}<0.5", lambda z, c=feat: z[c] < 0.5))
    if kind == "vp":
        out.append(("<1", f"{feat}<1(빈 쪽)", lambda z, c=feat: z[c] < 1))
    if kind in ("sign", "pos", "vp", "cont"):
        q1, q2 = x.quantile([1 / 3, 2 / 3])
        out.append(("하위1/3", f"{feat}<={q1:+.4f}", lambda z, c=feat, v=q1: z[c] <= v))
        out.append(("상위1/3", f"{feat}>{q2:+.4f}", lambda z, c=feat, v=q2: z[c] > v))
    if kind == "count":
        out.append((">=1", f"{feat}>=1", lambda z, c=feat: z[c] >= 1))
        out.append((">=2", f"{feat}>=2", lambda z, c=feat: z[c] >= 2))
    if kind == "bin":
        out.append(("참", f"{feat}=1", lambda z, c=feat: z[c] == 1))
    return out


def basic(d, a, y):
    A, B = d[a], d[~a]
    dd, tt, nd = daydiff(d, a, y)
    return dict(nA=len(A), nB=len(B), mA=A[y].mean(), mB=B[y].mean(), dd=dd, t=tt, nd=nd)


def run_tests(t):
    rows = []
    for sc, d in {"정규": t[t.ses == "정규"], "연장깨끗": t[(t.ses != "정규") & (t["tail"] == 0)]}.items():
        d1, d2 = d[d.dint < MID], d[d.dint >= MID]
        for feat, kind in FEATS:
            for nm, rule, fn in splits(feat, kind, d1):
                k1, k2 = d1[feat].notna(), d2[feat].notna()
                e1, e2 = d1[k1], d2[k2]
                a1, a2 = fn(e1), fn(e2)
                for y in ("y1", "y2"):
                    r1 = basic(e1, a1, y)
                    sgn = 1 if (r1["dd"] if r1["dd"] == r1["dd"] else 0) >= 0 else -1
                    r2 = basic(e2, a2, y)
                    rows.append(dict(scope=sc, feat=feat, name=NAME[feat], split=nm, rule=rule, y=y, sgn=sgn,
                                     **{f"앞_{k}": v for k, v in r1.items()}, **{f"뒤_{k}": v for k, v in r2.items()},
                                     뒤_t방향=sgn * r2["t"] if r2["t"] == r2["t"] else np.nan, _fn=fn))
    return pd.DataFrame(rows)


def pct(x):
    return "nan" if pd.isna(x) else f"{100 * x:+.2f}%"


def fy(y, x):
    """차이 표시: Y1은 %, Y2는 %p."""
    return pct(x) if y == "y1" else ("nan" if pd.isna(x) else f"{100 * x:+.1f}%p")


def fm(y, x):
    """평균 표시: Y1은 %, Y2(비율)는 %."""
    return pct(x) if y == "y1" else ("nan" if pd.isna(x) else f"{100 * x:.1f}%")


def control_block(d, fn, feat, y, sgn, label, q_atr):
    k = d[feat].notna()
    e = d[k]
    a = fn(e)
    r = basic(e, a, y)
    tod = daydiff(e, a, y, strat=(e["mod"] < 600))
    atr_t = daydiff(e, a, y, strat=np.select([e.atr_d <= q_atr[0], e.atr_d <= q_atr[1]], [0, 1], 2))
    vq = e.vol20.quantile([1 / 3, 2 / 3]).values
    vol_t = daydiff(e, a, y, strat=np.select([e.vol20 <= vq[0], e.vol20 <= vq[1]], [0, 1], 2))
    both = daydiff(e, a, y, strat=(e["mod"] < 600).astype(int) * 3 + np.select([e.atr_d <= q_atr[0], e.atr_d <= q_atr[1]], [0, 1], 2))
    rb, rt = fe_reg(e, a, y)
    xs = drop_worst(e, a, y, "sym", sgn)
    xd = drop_worst(e, a, y, "dint", sgn)
    o30 = (e["mod"] < 600)
    say(f"  [{label}] A {r['nA']}건 {fm(y, r['mA'])} vs B {r['nB']}건 {fm(y, r['mB'])} | 날짜차 {fy(y, r['dd'])} t {r['t']:+.2f} ({r['nd']}일)"
        f" | A 중 개장30분 {100 * o30[a].mean():.0f}% vs B {100 * o30[~a].mean():.0f}% | atr_d A {100 * e.atr_d[a].mean():.1f}% B {100 * e.atr_d[~a].mean():.1f}%")
    say(f"      통제: 시간대 층 {fy(y, tod[0])} t {tod[1]:+.2f} | atr 층 {fy(y, atr_t[0])} t {atr_t[1]:+.2f} | vol20 층 {fy(y, vol_t[0])} t {vol_t[1]:+.2f}"
        f" | 시간대xatr {fy(y, both[0])} t {both[1]:+.2f} | 회귀(날짜FE+개장30분+log atr+vol20) {fy(y, rb)} t {rt:+.2f}"
        f" | -{xs[0]} {fy(y, xs[1])} t {xs[2]:+.2f} | -{xd[0]} {fy(y, xd[1])} t {xd[2]:+.2f}")
    return dict(t=r["t"], dd=r["dd"], tod_t=tod[1], atr_t=atr_t[1], vol_t=vol_t[1], both_t=both[1], reg_t=rt, xs_t=xs[2], xd_t=xd[2],
                xs=xs[0], xd=xd[0], nA=r["nA"], nB=r["nB"], mA=r["mA"], mB=r["mB"])


# ---------------- 사건 연구(테마 급등 뒤 되돌림) ----------------

def spike_events(syms):
    rows = []
    for s in syms:
        d = prep(bars(RAW, s, 3), 60, 14)
        T = today_from_prep(d, 5)
        st = Struct(RAW, s)
        days = np.unique(T.dint)
        for D in days[(days >= 20260910)]:
            p = np.searchsorted(st.dint, D, "left") - 1
            if p < 20:
                continue
            pc = st.C[p]
            ds, de = np.searchsorted(T.dint, D, "left"), np.searchsorted(T.dint, D, "right")
            rm = np.maximum.accumulate(T.h[ds:de])
            for j in range(ds, de):
                if not (570 <= T.mod[j] < 900):
                    continue
                G = rm[j - ds] / pc - 1
                if G >= 0.05 and T.c[j] <= pc * (1 + G / 2):
                    e = T.c[j]
                    rest = np.arange(j + 1, de)
                    rest = rest[T.rth[rest]]
                    full = D <= 20261008
                    z1 = T.c[rest[-1]] / e - 1 if full and len(rest) else np.nan
                    z2 = float(T.c[rest].max() >= e + (rm[j - ds] - e) / 2) if full and len(rest) else np.nan
                    x = level_feats(st, T, j + 1, e, "정규")
                    x.update(sym=s, dint=D, t=T.idx[j], mod=T.mod[j], e=e, G=G, peak=rm[j - ds], pc0=pc, z1=z1, z2=z2,
                             z_max=(T.c[rest].max() / e - 1) if full and len(rest) else np.nan)
                    rows.append(x)
                    break
    return pd.DataFrame(rows)


def cl_diff(d, a, z):
    """평균 차와 날짜 군집 표준오차(사건이 날짜마다 적어 날짜 안 짝 대신 회귀)."""
    k = d[z].notna()
    d, a = d[k], a[k]
    if a.sum() < 3 or (~a).sum() < 3:
        return np.nan, np.nan, int(a.sum()), int((~a).sum())
    X = np.column_stack([np.ones(len(d)), a.astype(float)])
    Y = d[z].values
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    e = Y - X @ b
    XtX = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    g = d.dint.values
    G = 0
    for gg in np.unique(g):
        m = g == gg
        u = X[m].T @ e[m]
        meat += np.outer(u, u)
        G += 1
    V = XtX @ meat @ XtX * G / max(G - 1, 1)
    return b[1], b[1] / np.sqrt(V[1, 1]) if V[1, 1] > 0 else np.nan, int(a.sum()), int((~a).sum())


EV_FEATS = [("G", "med"), ("pret", "sign"), ("ph", "sign"), ("pdo", "sign"), ("h5", "sign"), ("pmh", "sign"), ("pml", "med"),
            ("op", "sign"), ("vw", "sign"), ("vw5", "sign"), ("poc5", "sign"), ("vp5r", "vp"), ("sw_n", "count1")]


def ev_split(feat, kind, e1):
    x = e1[feat].dropna()
    if kind == "sign":
        return (lambda z, c=feat: z[c] < 0), f"{feat}<0"
    if kind == "med":
        m = x.median()
        return (lambda z, c=feat, v=m: z[c] <= v), f"{feat}<={m:+.4f}"
    if kind == "vp":
        return (lambda z, c=feat: z[c] < 1), f"{feat}<1"
    return (lambda z, c=feat: z[c] >= 1), f"{feat}>=1"


def event_tests(ev):
    """주 t = 날짜마다 A평균 - B평균(지시 방식), 보조 = 평균차의 날짜 군집 회귀 t."""
    e1, e2 = ev[ev.dint < MID], ev[(ev.dint >= MID) & ev.z1.notna()]
    rows = []
    for feat, kind in EV_FEATS:
        if e1[feat].notna().sum() < 10:
            continue
        fn, rule = ev_split(feat, kind, e1)
        for z in ("z1", "z2"):
            k1, k2 = e1[feat].notna(), e2[feat].notna()
            f1, f2 = e1[k1], e2[k2]
            a1, a2 = fn(f1), fn(f2)
            d1, t1, nd1 = daydiff(f1, a1, z)
            sg = 1 if (d1 if d1 == d1 else 0) >= 0 else -1
            d2, t2, nd2 = daydiff(f2, a2, z)
            c1, ct1, _, _ = cl_diff(f1, a1, z)
            c2, ct2, _, _ = cl_diff(f2, a2, z)
            rows.append(dict(feat=feat, name=NAME.get(feat, feat), rule=rule, z=z, sgn=sg, kind=kind,
                             앞_nA=int(a1.sum()), 앞_nB=int((~a1).sum()), 앞_mA=f1[a1][z].mean(), 앞_mB=f1[~a1][z].mean(),
                             앞_dd=d1, 앞_t=t1, 앞_nd=nd1, 앞_군집t=ct1,
                             뒤_nA=int(a2.sum()), 뒤_nB=int((~a2).sum()), 뒤_mA=f2[a2][z].mean(), 뒤_mB=f2[~a2][z].mean(),
                             뒤_dd=d2, 뒤_t=t2, 뒤_nd=nd2, 뒤_군집t=ct2, 뒤_t방향=sg * t2 if t2 == t2 else np.nan, _fn=fn))
    return pd.DataFrame(rows)


def event_controls(e, fn, feat, z, sgn, label):
    k = e[feat].notna() & e[z].notna()
    e = e[k]
    a = fn(e)
    dd, tt, nd = daydiff(e, a, z)
    tod = daydiff(e, a, z, strat=(e["mod"] < 600))
    gq = np.select([e.G <= 0.06, e.G <= 0.08], [0, 1], 2)
    gs = daydiff(e, a, z, strat=gq)
    rb, rt = fe_reg(e, a, z, ["open30", "lG", "latr"])
    xs = drop_worst(e, a, z, "sym", sgn)
    xd = drop_worst(e, a, z, "dint", sgn)
    fz = (lambda v: pct(v)) if z == "z1" else (lambda v: "nan" if pd.isna(v) else f"{100 * v:+.1f}%p")
    say(f"  [{label}] A {int(a.sum())}건 vs B {int((~a).sum())}건 | 날짜차 {fz(dd)} t {tt:+.2f} ({nd}일) | 시간대 층 t {tod[1]:+.2f} | 급등폭(5~6·6~8·8%+) 층 t {gs[1]:+.2f}"
        f" | 회귀(날짜FE+개장30분+log G+log atr) {fz(rb)} t {rt:+.2f} | -{xs[0]} t {xs[2]:+.2f} | -{xd[0]} t {xd[2]:+.2f}")
    return dict(t=tt, tod_t=tod[1], g_t=gs[1], reg_t=rt, xs_t=xs[2], xd_t=xd[2], xs=xs[0], xd=xd[0], nA=int(a.sum()), nB=int((~a).sum()),
                mA=e[a][z].mean(), mB=e[~a][z].mean(), dd=dd)


def lookahead_check(t, n=80, seed=7):
    """무작위 거래 n건: 봉 i의 고가·저가·종가·거래량을 지우고 i 뒤 봉·그 시각 뒤 5분봉·그날 이후 일봉(애프터는 그날까지)을 잘라 다시 계산."""
    rng = np.random.default_rng(seed)
    pick = t.iloc[rng.choice(len(t), n, replace=False)]
    cols = [f for f, _ in FEATS] + ["pdo", "pmh", "h5", "pret", "atr_d"]
    bad = 0
    for _, r in pick.iterrows():
        d = prep(bars(RAW, r.sym, 3), 60, 14)
        i = d.index.get_indexer([r.ts])[0]
        full = level_feats(Struct(RAW, r.sym), today_from_prep(d, 5), i, r.ent, r.ses)
        x = d.iloc[:i + 1].copy()
        x.iloc[-1, x.columns.get_indexer(["h", "l", "c", "v"])] = np.nan
        T = Today(x.index, x.o.values.astype(float), x.h.values.astype(float), x.l.values.astype(float),
                  x.c.values.astype(float), x.v.values.astype(float), 5)
        cut = r.dint if r.ses == "애프터" else r.dint - 1
        st = Struct(RAW, r.sym, cut_dint=cut, cut_ts=r.ts)
        part = level_feats(st, T, i, r.ent, r.ses)
        for c in cols:
            a, b = full.get(c, np.nan), part.get(c, np.nan)
            if not ((pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and abs(a - b) < 1e-9)):
                bad += 1
                say(f"   [미래 참조 의심] {r.sym} {r.ts} {c}: 전체 {a} vs 잘라서 {b}")
    return n, len(cols), bad


# ---------------- 10/9 사례 ----------------

def case_rows():
    out = []
    for s in ["CRCL", "MRVL", "LITE", "COHR"]:
        d, dead, hot = build.arrays(s)
        T = today_from_prep(d, 5)
        st = Struct(RAW, s)
        x0 = np.searchsorted(T.dint, 20261009, "left")
        for i in range(x0, len(d)):
            lvl = d.ema.values[i - 1] * 0.98
            if d.l.values[i] <= lvl and not hot[i]:
                ent = min(d.o.values[i], lvl)
                x = level_feats(st, T, i, ent, d.ses.iloc[i])
                after = d.iloc[i:]
                rr = after.r.values
                sold = next((j for j in range(1, len(after)) if rr[j - 1] > 70), None)
                x.update(sym=s, t=d.index[i], ent=ent, last=d.c.values[-1], last_t=d.index[-1],
                         y_now=d.c.values[-1] / ent - 1, rsi_sell=(f"{sold}봉 뒤 RSI 매도 {d.o.values[i + sold] / ent - 1:+.2%}" if sold else "10:18까지 RSI 매도 없음"))
                out.append(x)
                break
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    t = late_trades()
    say("이전 가격대로 신호 뒤 반등을 미리 알 수 있었나 — pat_levels (2026-10-10)")
    say("=" * 100)
    say("결과 변수(시험 전에 정함): Y1 = 거래 수익 ret(지금 매도 규칙), Y2 = 빠른 반등(매수 뒤 20봉=60분 안에 RSI 매도, 5분봉 보조는 12봉)")
    say("사건 연구(테마 급등 뒤 되돌림): 04시부터 봉 j까지 최고가가 전날 종가 +5% 이상(G)이고, 정규장 09:30~14:57 3분봉 종가가 처음으로")
    say("  전날 종가 x (1+G/2) 이하(오른 폭 절반 되돌림)인 봉 j. 판단 = 봉 j+1 시가, 가격 e = c_j, 특성은 봉 j까지.")
    say("  Z1 = 그날 정규장 마지막 종가 / e - 1, Z2 = 이후 장중 종가 최고가가 e + (최고가 - e)/2 이상(빠진 폭 절반 회복). 9/10~10/8, 10/9는 사례만.")
    say("특성은 모두 매수 봉 시가 시점 값(봉 i-1까지, 지난 날, 체결가). 정의는 code/pat_levels.py 머리말.")
    say("")
    n_ext = (t.ses != "정규").sum()
    tl = t[t.ses != "정규"]
    say(f"[데이터] 거래 {len(t)}건(정규 {(t.ses == '정규').sum()}, 프리 {(t.ses == '프리').sum()}, 애프터 {(t.ses == '애프터').sum()}), 종목 {t.sym.nunique()}, "
        f"앞 {(t.dint < MID).sum()}건 / 뒤 {(t.dint >= MID).sum()}건")
    say(f"[체결 점검] 프리·애프터 {n_ext}건 중 꼬리(직전 1분 종가보다 1% 넘게 아래) {int((tl['tail'] == 1).sum())}건, 판정 불가 {int(tl['tail'].isna().sum())}건 → "
        f"깨끗한 체결 {int((tl['tail'] == 0).sum())}건. 꼬리 평균 Y1 {pct(tl[tl['tail'] == 1].ret.mean())}, 깨끗 {pct(tl[tl['tail'] == 0].ret.mean())}")
    for sc, d in {"정규": t[t.ses == "정규"], "연장깨끗": tl[tl["tail"] == 0]}.items():
        for h, dd in (("앞", d[d.dint < MID]), ("뒤", d[d.dint >= MID])):
            say(f"  {sc} {h}: {len(dd)}건 Y1 {pct(dd.y1.mean())} Y2 {100 * dd.y2.mean():.0f}% | 개장 30분 매수 {100 * (dd['mod'] < 600).mean() if sc == '정규' else 0:.0f}%")
    cov = t[t.ses == "정규"][[f for f, _ in FEATS]].notna().mean()
    say("  정규장 특성 채움률: " + " ".join(f"{k} {100 * v:.0f}%" for k, v in cov.items()))
    say("")
    n_ck, n_col, bad = lookahead_check(t)
    say(f"[미래 참조 점검] 무작위 거래 {n_ck}건 x 특성 {n_col}개: 매수 봉의 고가·저가·종가·거래량을 지우고 그 뒤 자료를 잘라 다시 계산 → 다른 값 {bad}개")
    say("")
    say("[이미 반박된 것과 겹치는 부분 — 다시 시험하지 않고 결과만 대조]")
    say("  - dip_reason 1절: d_prev(전일 종가 대비)·d_high(오늘 고점 대비)는 정규장 안에서 t +0.45 / +0.77(3분봉), 5분봉 +0.52 / +0.29. 여기 pc는 같은 값의 부호 분할만 새로 본다.")
    say("  - dip_reason 10절: '직전 정규장 저가를 종가로 이탈하면 손절'은 앞 +0.81 → 뒤 -0.42로 반박. 여기 pl은 매수 순간 그 아래에서 샀는지(매수 필터)라 각도가 다르다.")
    say("  - dip_reason 14절: 40분봉 지지 구역(스윙 피벗 묶음) 근처·버틴 시간·지지 피로는 효과 없음(468규칙, 계열 최선 6개 반박). 여기 sw_n·sw_b·sw_a는")
    say("    15분 스윙 저점으로 다시 정의한 '받친 횟수·거리'이고, 거래량 가격대(vp·poc)·VWAP·라운드·전날/오늘 기준가는 새 각도다.")
    say("  - fam_context: day_ret(i-1 종가의 전날 종가 대비)·prev_ret 3분위는 탈락. pc 3분위는 그와 거의 같은 값이다.")
    say("")

    tab = run_tests(t)
    ntest = len(tab)
    say(f"[시험 범위] 특성 {len(FEATS)}개, 분할 {tab[tab.scope == '정규'][['feat', 'split']].drop_duplicates().shape[0]}개(정규)·"
        f"{tab[tab.scope == '연장깨끗'][['feat', 'split']].drop_duplicates().shape[0]}개(연장깨끗) x 결과 2개 = 시험 {ntest}개"
        f"(정규 {(tab.scope == '정규').sum()}, 연장깨끗 {(tab.scope == '연장깨끗').sum()}). 분할끼리 겹쳐 독립이 아님.")
    for sc in ("정규", "연장깨끗"):
        x = tab[tab.scope == sc]
        say(f"  {sc}: 앞 |t|>=2 {int((x['앞_t'].abs() >= 2).sum())}/{len(x)}(우연 기대 약 {0.05 * len(x):.0f}~{0.1 * len(x):.0f}개, 날짜 약 10일이라 t 꼬리 두꺼움),"
            f" 뒤 |t|>=2 {int((x['뒤_t'].abs() >= 2).sum())}/{len(x)}, 앞 방향 그대로 뒤 t>=2 {int((x['뒤_t방향'] >= 2).sum())}개,"
            f" 앞·뒤 같은 부호 {int((np.sign(x['앞_t']) == np.sign(x['뒤_t'])).sum())}/{int(x[['앞_t', '뒤_t']].notna().all(axis=1).sum())}")
    say("")
    say("[앞 절반 정규장 |t| 상위 12개(고르기용)]")
    reg = tab[(tab.scope == "정규") & (tab["앞_nA"] >= 15) & (tab["앞_nB"] >= 15) & (tab["앞_nd"] >= 6)].copy()
    reg["s"] = reg["앞_t"].abs()
    for _, r in reg.sort_values("s", ascending=False).head(12).iterrows():
        say(f"  {r.feat:6s} {r['name']:16s} {r.split:6s} {r.rule:22s} {r.y} | 앞 A{r['앞_nA']}/B{r['앞_nB']} {fm(r.y, r['앞_mA'])} vs {fm(r.y, r['앞_mB'])}"
            f" 날짜차 {fy(r.y, r['앞_dd'])} t {r['앞_t']:+.2f} | 뒤 A{r['뒤_nA']}/B{r['뒤_nB']} 날짜차 {fy(r.y, r['뒤_dd'])} t {r['뒤_t']:+.2f}")
    picks, seen = [], set()
    for _, r in reg.sort_values("s", ascending=False).iterrows():
        if r.feat in seen:
            continue
        seen.add(r.feat)
        picks.append(r)
        if len(picks) == 3:
            break
    say("")
    say("[후보 3개: 앞에서 정한 문턱·방향을 뒤에 그대로 + 통제]")
    early = early_trades()
    say(f"  (보조 5분봉 7/14~9/8 정규장 {len(early)}건, Y1 {pct(early.y1.mean())}, Y2 {100 * early.y2.mean():.0f}%)")
    regt = t[t.ses == "정규"]
    q_atr = regt[regt.dint < MID].atr_d.quantile([1 / 3, 2 / 3]).values
    ext = t[(t.ses != "정규") & (t["tail"] == 0)]
    cands = []
    for r in picks:
        fn, sgn, y, feat = r["_fn"], int(r.sgn), r.y, r.feat
        say(f"\n 후보 {feat} {r['name']} · {r.split} ({r.rule}) · {y} · 앞 방향 {'A가 좋음' if sgn > 0 else 'A가 나쁨'}")
        c1 = control_block(regt[regt.dint < MID], fn, feat, y, sgn, "앞 정규", q_atr)
        c2 = control_block(regt[regt.dint >= MID], fn, feat, y, sgn, "뒤 정규", q_atr)
        y_other = "y2" if y == "y1" else "y1"
        c2o = basic(regt[(regt.dint >= MID) & regt[feat].notna()], fn(regt[(regt.dint >= MID) & regt[feat].notna()]), y_other)
        c1o = basic(regt[(regt.dint < MID) & regt[feat].notna()], fn(regt[(regt.dint < MID) & regt[feat].notna()]), y_other)
        say(f"  [다른 결과 {y_other}] 앞 날짜차 {fy(y_other, c1o['dd'])} t {c1o['t']:+.2f} | 뒤 {fy(y_other, c2o['dd'])} t {c2o['t']:+.2f}")
        ce = None
        if early[feat].notna().sum() > 30:
            ce = control_block(early, fn, feat, y, sgn, "보조 5분봉 7/14~9/8 정규", early.atr_d.quantile([1 / 3, 2 / 3]).values)
        else:
            say("  [보조 5분봉] 이 특성은 5분봉 기간에 값이 부족")
        cx = None
        if ext[feat].notna().sum() > 30:
            ex2 = ext[ext.dint >= MID]
            cx = basic(ex2[ex2[feat].notna()], fn(ex2[ex2[feat].notna()]), y)
            ex1 = ext[ext.dint < MID]
            cx1 = basic(ex1[ex1[feat].notna()], fn(ex1[ex1[feat].notna()]), y)
            say(f"  [연장 깨끗한 체결] 앞 날짜차 {fy(y, cx1['dd'])} t {cx1['t']:+.2f} ({cx1['nA']}/{cx1['nB']}) | 뒤 {fy(y, cx['dd'])} t {cx['t']:+.2f} ({cx['nA']}/{cx['nB']})")
        bt = sgn * c2["t"]
        ctl = [sgn * c2[k] for k in ("tod_t", "atr_t", "vol_t", "both_t", "reg_t", "xs_t", "xd_t")]
        if not bt == bt or bt < 1:
            verdict = "뒤에서 실패"
        elif bt >= 2 and all(v == v and v >= 1 for v in ctl):
            verdict = "확인됨"
        else:
            verdict = "약함"
        say(f"  → 뒤 t(앞 방향) {bt:+.2f}, 통제 t(앞 방향) {' '.join(f'{v:+.2f}' for v in ctl)} → {verdict}")
        cands.append(dict(feat=feat, name=r["name"], rule=f"{r.rule} ({r.split}), {y}, 앞 방향 {'A가 좋음' if sgn > 0 else 'A가 나쁨'}",
                          first=f"앞 정규 A {c1['nA']}건 {fm(y, c1['mA'])} vs B {c1['nB']}건 {fm(y, c1['mB'])}, 날짜차 {fy(y, c1['dd'])} t {c1['t']:+.2f}",
                          second=f"뒤 정규 A {c2['nA']}건 {fm(y, c2['mA'])} vs B {c2['nB']}건 {fm(y, c2['mB'])}, 날짜차 {fy(y, c2['dd'])} t {c2['t']:+.2f}"
                          + (f"; 5분봉 보조 t {ce['t']:+.2f}" if ce else ""),
                          controls=f"뒤: 시간대 층 t {c2['tod_t']:+.2f}, atr 층 {c2['atr_t']:+.2f}, vol20 층 {c2['vol_t']:+.2f}, 시간대xatr {c2['both_t']:+.2f}, "
                                   f"회귀 {c2['reg_t']:+.2f}, -{c2['xs']} {c2['xs_t']:+.2f}, -{c2['xd']} {c2['xd_t']:+.2f}",
                          verdict=verdict))

    # 뒤 절반에서 따로 강한 것(결과 보고 고른 것이라 관찰만)
    say("\n[참고: 뒤 절반 정규장에서 |t| 상위 5개 — 앞에서 고르지 않은 것, 관찰만]")
    rb = tab[(tab.scope == "정규")].copy()
    rb["s"] = rb["뒤_t"].abs()
    for _, r in rb.sort_values("s", ascending=False).head(5).iterrows():
        ek = early[r.feat].notna()
        eb = basic(early[ek], r["_fn"](early[ek]), r.y) if ek.sum() > 30 else dict(dd=np.nan, t=np.nan)
        say(f"  {r.feat:6s} {r['name']:16s} {r.split:6s} {r.rule:22s} {r.y} | 앞 t {r['앞_t']:+.2f} | 뒤 날짜차 {fy(r.y, r['뒤_dd'])} t {r['뒤_t']:+.2f}"
            f" | 5분봉 7/14~9/8 {fy(r.y, eb['dd'])} t {eb['t']:+.2f}")

    # 연장 깨끗한 체결: 앞 상위 3개(보조)
    say("\n[보조: 프리·애프터 깨끗한 체결 앞 |t| 상위 3개(서로 다른 특성) → 뒤]")
    rx = tab[(tab.scope == "연장깨끗") & (tab["앞_nA"] >= 15) & (tab["앞_nB"] >= 15)].copy()
    rx["s"] = rx["앞_t"].abs()
    seen = set()
    for _, r in rx.sort_values("s", ascending=False).iterrows():
        if r.feat in seen:
            continue
        seen.add(r.feat)
        say(f"  {r.feat:6s} {r['name']:16s} {r.split:6s} {r.rule:22s} {r.y} | 앞 {fy(r.y, r['앞_dd'])} t {r['앞_t']:+.2f} | 뒤(앞 방향) t {r['뒤_t방향']:+.2f}")
        if len(seen) == 3:
            break

    tab.drop(columns="_fn").to_csv(os.path.join(RES, "pat_levels_tests.csv"), index=False, encoding="utf-8-sig")
    keep = ["sym", "day", "t", "ses", "half", "ret", "why", "bars", "ent", "y2", "tail", "jump1m", "atr_d", "vol20"] + \
           [f for f, _ in FEATS] + ["pdo", "pmh", "h5", "pret"]
    t[keep].to_csv(os.path.join(RES, "pat_levels_trades.csv"), index=False, encoding="utf-8-sig")

    # ---------- 사건 연구 ----------
    say("\n" + "=" * 100)
    say("[사건 연구: 테마 급등 뒤 절반 되돌림 — 회복하나(LITE형) 못 하나(COHR형)]")
    ev = spike_events(sorted(t.sym.unique()))
    ev.to_csv(os.path.join(RES, "pat_levels_events.csv"), index=False, encoding="utf-8-sig")
    full = ev[ev.z1.notna()]
    say(f"  사건 {len(full)}건(9/10~10/8, 종목 {full.sym.nunique()}, 날짜 {full.dint.nunique()}) · 앞 {(full.dint < MID).sum()} / 뒤 {(full.dint >= MID).sum()}"
        f" · Z1 평균 {pct(full.z1.mean())} 중앙 {pct(full.z1.median())} · Z2 회복 {100 * full.z2.mean():.0f}% · 개장 30분 안 사건 {100 * (pd.DatetimeIndex(full.t).hour * 60 + pd.DatetimeIndex(full.t).minute < 600).mean():.0f}%")
    et = event_tests(full)
    et.drop(columns="_fn").to_csv(os.path.join(RES, "pat_levels_event_tests.csv"), index=False, encoding="utf-8-sig")
    say(f"  시험 {len(et)}개(특성 {et.feat.nunique()} x Z 2, 문턱은 부호·앞 절반 중앙값·밀도 1·스윙 1개). t = 날짜마다 A평균-B평균(사건이 있는 날 중 두 무리 다 있는 날),"
        f" 괄호는 평균차의 날짜 군집 회귀 t.")
    say(f"  앞 |t|>=2 {int((et['앞_t'].abs() >= 2).sum())}개, 앞 방향 그대로 뒤 t>=2 {int((et['뒤_t방향'] >= 2).sum())}개, 앞·뒤 같은 부호 "
        f"{int((np.sign(et['앞_t']) == np.sign(et['뒤_t'])).sum())}/{int(et[['앞_t', '뒤_t']].notna().all(axis=1).sum())}")
    et["s"] = et["앞_t"].abs()
    for _, r in et.sort_values("s", ascending=False).iterrows():
        fz = (lambda v: pct(v)) if r.z == "z1" else (lambda v: "nan" if pd.isna(v) else f"{100 * v:.0f}%")
        say(f"  {r.feat:5s} {r['name']:14s} {r.rule:18s} {r.z} | 앞 A{r['앞_nA']}/B{r['앞_nB']} {fz(r['앞_mA'])} vs {fz(r['앞_mB'])} t {r['앞_t']:+.2f}({r['앞_군집t']:+.2f}, {r['앞_nd']}일)"
            f" | 뒤 A{r['뒤_nA']}/B{r['뒤_nB']} {fz(r['뒤_mA'])} vs {fz(r['뒤_mB'])} t {r['뒤_t']:+.2f}({r['뒤_군집t']:+.2f}, {r['뒤_nd']}일) (앞 방향 {r['뒤_t방향']:+.2f})")
    say("")
    say("  [사건 후보: 앞 |t| 상위 서로 다른 특성 3개 → 뒤 + 통제]")
    epick, seen = [], set()
    for _, r in et[(et["앞_nA"] >= 10) & (et["앞_nB"] >= 10)].sort_values("s", ascending=False).iterrows():
        if r.feat in seen:
            continue
        seen.add(r.feat)
        epick.append(r)
        if len(epick) == 3:
            break
    for r in epick:
        sgn = int(r.sgn)
        say("")
        say(f"   후보 {r.feat} {r['name']} · {r.rule} · {r.z} · 앞 방향 {'A가 좋음' if sgn > 0 else 'A가 나쁨'}")
        c1 = event_controls(full[full.dint < MID], r["_fn"], r.feat, r.z, sgn, "앞")
        c2 = event_controls(full[full.dint >= MID], r["_fn"], r.feat, r.z, sgn, "뒤")
        bt = sgn * c2["t"]
        ctl = [sgn * c2[k] for k in ("tod_t", "g_t", "reg_t", "xs_t", "xd_t")]
        if not bt == bt or bt < 1:
            verdict = "뒤에서 실패"
        elif bt >= 2 and all(v == v and v >= 1 for v in ctl):
            verdict = "확인됨"
        else:
            verdict = "약함"
        say(f"   → 뒤 t(앞 방향) {bt:+.2f}, 통제 t(앞 방향) {' '.join(f'{v:+.2f}' for v in ctl)} → {verdict}")
        fz = (lambda v: pct(v)) if r.z == "z1" else (lambda v: "nan" if pd.isna(v) else f"{100 * v:.0f}%")
        fd = (lambda v: pct(v)) if r.z == "z1" else (lambda v: "nan" if pd.isna(v) else f"{100 * v:+.0f}%p")
        cands.append(dict(feat=f"[사건] {r.feat}", name=r["name"], rule=f"급등 되돌림 사건 {r.rule}, {r.z}, 앞 방향 {'A가 좋음' if sgn > 0 else 'A가 나쁨'}",
                          first=f"앞 A {c1['nA']}건 {fz(c1['mA'])} vs B {c1['nB']}건 {fz(c1['mB'])}, 날짜차 {fd(c1['dd'])} t {c1['t']:+.2f}",
                          second=f"뒤 A {c2['nA']}건 {fz(c2['mA'])} vs B {c2['nB']}건 {fz(c2['mB'])}, 날짜차 {fd(c2['dd'])} t {c2['t']:+.2f}",
                          controls=f"뒤: 시간대 층 t {c2['tod_t']:+.2f}, 급등폭 층 {c2['g_t']:+.2f}, 회귀 {c2['reg_t']:+.2f}, -{c2['xs']} {c2['xs_t']:+.2f}, -{c2['xd']} {c2['xd_t']:+.2f}",
                          verdict=verdict))
    pd.DataFrame(cands).to_csv(os.path.join(RES, "pat_levels_candidates.csv"), index=False, encoding="utf-8-sig")
    ec = ev[ev.dint == 20261009]
    say("\n  10/9 사건(10:18 ET까지 자료, 결과 미확정):")
    for _, r in ec[ec.sym.isin(["LITE", "COHR", "CRCL", "MRVL"])].iterrows():
        say(f"   {r.sym} {r.t:%H:%M} e {r.e:.2f} G {pct(r.G)} | 전날 수익 {pct(r.pret)} 전날 고가 {pct(r.ph)} 전날 시가 {pct(r.pdo)} 5일 고가 {pct(r.h5)}"
            f" 프리 고가 {pct(r.pmh)} 프리 저가 {pct(r.pml)} 시가 {pct(r.op)} VWAP {pct(r.vw)} 5일VWAP {pct(r.vw5)} 밀도 {r.vp5r:.2f} 최다거래가 {pct(r.poc5)} 스윙 {r.sw_n}")
    say(f"   10/9 급등 되돌림 사건 전체 {len(ec)}건: {', '.join(ec.sym)}")

    # ---------- 10/9 사례 ----------
    say("\n[10/9 첫 이격 매수 사례(정규장, 10:18 ET까지)]")
    cs = case_rows()
    for _, r in cs.iterrows():
        say(f"  {r.sym} {r.t:%H:%M} 체결 {r.ent:.2f} → 10:18 {r['last']:.2f}({pct(r.y_now)}), {r.rsi_sell}")
        say(f"     전날 종가 {pct(r.pc)} 저가 {pct(r.pl)} 고가 {pct(r.ph)} | 오늘 시가 {pct(r.get('op'))} 프리 저가 {pct(r.get('pml'))} 오늘 저가 {pct(r.get('dl'))}"
            f" | 5일 저가 {pct(r.l5)} 20일 위치 {r.pos20:.2f} | VWAP {pct(r.get('vw'))} 5일VWAP {pct(r.vw5)} | 밀도5 {r.vp5r:.2f} 밀도20 {r.vp20r:.2f}"
            f" 최다거래가 {pct(r.poc5)} | 스윙 {r.sw_n}개, 아래 {pct(r.sw_b)} 위 {pct(r.sw_a)} | 라운드 위 {int(r.rnd_up)} 아래 {int(r.rnd_dn)} 잔 {int(r.rnd_f)}")
    cs.to_csv(os.path.join(RES, "pat_levels_cases.csv"), index=False, encoding="utf-8-sig")
    say("")
    say("[요약]")
    say(f"  시험: 거래 {ntest}개 + 사건 {len(et)}개 = {ntest + len(et)}개. 후보는 앞 절반에서만 골랐다(거래 3 + 사건 3).")
    for c in cands:
        say(f"  {c['verdict']:6s} {c['feat']} {c['name']}: {c['rule']} | {c['second']}")
    yb = regt.groupby(regt.dint >= MID).y2.agg(["sum", "count"])
    say(f"  Y2(60분 안 RSI 매도)는 정규장 앞 {int(yb.loc[False, 'sum'])}/{int(yb.loc[False, 'count'])}건, 뒤 {int(yb.loc[True, 'sum'])}/{int(yb.loc[True, 'count'])}건뿐이라"
        " Y2 시험은 검정력이 낮다. RSI 매도 중앙값은 매수 뒤 75봉(약 3.8시간)이다.")
    say(f"\n실행 {time.time() - t0:.0f}초")
    with open(os.path.join(RES, "pat_levels.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
