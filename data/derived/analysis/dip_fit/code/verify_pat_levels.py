"""반박 검증: pat_levels.py 후보(거래 3 + 사건 3)와 관찰 2개를 직접 다시 계산해 무너뜨려 본다. 2026-10-10.

결과 변수(분석가가 시험 전에 정한 그대로, 바꾸지 않음)
  Y1 = 거래 수익 ret, Y2 = 매수 뒤 20봉(60분) 안 RSI 매도(why=RSI & bars<=20).
  사건 연구: 04시부터 봉 j까지 최고가가 전날 정규장 종가 +5% 이상(G), 정규장 09:30~14:57 3분봉 종가가 처음으로 전날 종가x(1+G/2) 이하인 봉 j.
    판단 = 봉 j+1 시가, e = c_j. Z1 = 그날 정규장 마지막 종가/e-1, Z2 = 이후 정규장 종가 최고가 >= e+(최고가-e)/2.
점검
  0) 독립 재계산: pos20·vp20r·sw_b·poc5·vp5r를 1분봉(매수 시각 앞)·5분봉·일봉에서 따로 다시 짜서 분석가 값과 대조(정규장 678건 전부).
     사건 240건도 3분봉에서 따로 다시 찾아 z1·z2 대조.
  1) 재현: 내 날짜 군집 t(날짜마다 A평균-B평균)로 앞·뒤 t 다시 잼.
  2) 꼬리 체결: 정규장 매수에도 '체결 분 직전 1분 종가보다 1% 넘게 아래' 체결이 있는지, 빼고 다시.
  3) 통제: 개장 30분 층, 시각 3층(09:30~38 / 09:39~59 / 10시~), atr_d 3분위 층, vol20 3분위 층, 날짜FE 회귀(개장30분+log atr+vol20),
     A 쪽 기여 최대 1종목·1일 빼기, 가장 많은 종목·날짜 빼기.
  4) 우연: 같은 고르기 절차(정규장 96시험 → 앞 |t| 상위 서로 다른 특성 3개 → 뒤 t)를
     (a) 같은 날짜 안에서 특성 행을 섞어 500회, (b) 종목 바꿔 끼우기(종목 순열, 시각이 가장 가까운 그 종목 거래의 특성) 300회 돌려 실제와 비교.
     사건 26시험도 (a) 500회. 뒤 절반 최대 |t|(관찰 poc5 고르기)도 같은 분포와 비교.
  5) 관찰: poc5<0(뒤에서 고름)·G<=6.7%(사건 Z2) — 통제·쏠림·기계적 설명(Z2 문턱 = (최고가-e)/2, G가 작으면 문턱도 작음).
  6) 10/9 사례: 후보·관찰 규칙을 그대로 대입해 실제(10:18 ET까지)와 맞는지.
저장: ../results/verify_pat_levels.txt
사용: py verify_pat_levels.py"""
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
RAW = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip/raw_1009"
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
from events import bars, prep  # noqa: E402

ET = "America/New_York"
MID = 20260924
LINES = []
rng = np.random.default_rng(20261010)


def say(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def pct(v):
    return "nan" if pd.isna(v) else f"{100 * v:+.2f}%"


def dint_of(dates):
    return np.array([x.year * 10000 + x.month * 100 + x.day for x in dates], dtype=np.int64)


# ---------------- 통계 ----------------

def tst(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 3 or v.std(ddof=1) == 0:
        return (v.mean() if len(v) else np.nan), np.nan, len(v)
    return v.mean(), v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), len(v)


def dd_np(dc, nd, a, y):
    """dc: 날짜 코드(0..nd-1), a: 0/1, y. 날짜마다 A평균-B평균 → 평균, t, 날짜 수."""
    a = a.astype(float)
    sA, nA = np.bincount(dc, y * a, nd), np.bincount(dc, a, nd)
    sB, nB = np.bincount(dc, y * (1 - a), nd), np.bincount(dc, 1 - a, nd)
    ok = (nA > 0) & (nB > 0)
    if ok.sum() < 3:
        return np.nan, np.nan, int(ok.sum())
    x = sA[ok] / nA[ok] - sB[ok] / nB[ok]
    return tst(x)


def dd_strat(d, a, y, strat):
    """날짜x층마다 A-B → 날짜 안 층 평균 → 날짜 t."""
    df = pd.DataFrame({"d": d.dint.values, "s": np.asarray(strat), "a": np.asarray(a, bool), "y": d[y].values})
    g = df.groupby(["d", "s", "a"]).y.mean().unstack()
    if True not in g.columns or False not in g.columns:
        return np.nan, np.nan, 0
    x = (g[True] - g[False]).dropna().groupby(level=0).mean()
    return tst(x.values)


def dd_df(d, a, y):
    dc, u = pd.factorize(d.dint.values)
    return dd_np(dc, len(u), np.asarray(a, float), d[y].values.astype(float))


def fe_reg(d, a, y, covs):
    X = np.column_stack([np.asarray(a, float)] + [d[c].values.astype(float) for c in covs])
    Y = d[y].values.astype(float)
    k = ~np.isnan(X).any(axis=1) & ~np.isnan(Y)
    X, Y, g = X[k], Y[k], d.dint.values[k]
    Xd = X - pd.DataFrame(X).groupby(g).transform("mean").values
    Yd = Y - pd.Series(Y).groupby(g).transform("mean").values
    XtX = np.linalg.pinv(Xd.T @ Xd)
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


# ---------------- 독립 재계산 ----------------

def piv_lows(l, k):
    out = []
    for j in range(k, len(l) - k):
        if l[j] < l[j - k:j].min() and l[j] <= l[j + 1:j + k + 1].min():
            out.append(l[j])
    return out


def vdens(l, h, v, ent):
    blo, bhi = ent * 0.99, ent * 1.01
    w = np.zeros(len(l))
    for q in range(len(l)):
        if h[q] > l[q]:
            w[q] = max(0.0, min(h[q], bhi) - max(l[q], blo)) / (h[q] - l[q])
        else:
            w[q] = float(blo <= l[q] <= bhi)
    share = (v * w).sum() / v.sum()
    span = h.max() - l.min()
    exp = min(1.0, (bhi - blo) / span) if span > 0 else 1.0
    return share / exp


def poc_of(l, h, v):
    typ = (l + h) / 2
    b = np.floor(np.log(typ) / np.log(1.005)).astype(np.int64)
    s = {}
    for bb, vv in zip(b, v):
        s[bb] = s.get(bb, 0.0) + vv
    keys = sorted(s)
    best = max(keys, key=lambda kk: (s[kk], -kk))   # 같으면 낮은 칸(분석가 idxmax와 같은 규칙)
    return np.exp((best + 0.5) * np.log(1.005))


def my_feats(sym, rows):
    """정규장 매수만. 일봉은 매수일 전날까지, 5분봉은 L까지 정규장, 오늘은 1분봉 매수 시각 앞까지 3분봉."""
    dd = pd.read_parquet(f"{RAW}/1d_{sym}.parquet")
    ddi = dint_of(pd.to_datetime(dd.index).date)
    H, L = dd.High.values.astype(float), dd.Low.values.astype(float)
    m5 = pd.read_parquet(f"{RAW}/5m_{sym}.parquet")
    mod5 = m5.index.hour * 60 + m5.index.minute
    m5 = m5[(mod5 >= 570) & (mod5 < 960)]
    d5 = dint_of(m5.index.date)
    m1 = pd.read_parquet(f"{RAW}/1m_{sym}.parquet")
    out = {}
    for k, r in rows.iterrows():
        D = r.dint
        p = np.searchsorted(ddi, D, "left") - 1
        if p < 20:
            continue
        x = {}
        lo20, hi20 = L[p - 19:p + 1].min(), H[p - 19:p + 1].max()
        x["pos20_v"] = (r.ent - lo20) / (hi20 - lo20)
        Ld = ddi[p]
        days = np.unique(d5[d5 <= Ld])
        if len(days) >= 20:
            s20 = np.isin(d5, days[-20:])
            x["vp20r_v"] = vdens(m5.Low.values[s20].astype(float), m5.High.values[s20].astype(float),
                                 m5.Volume.values[s20].astype(float), r.ent)
        if len(days) >= 5:
            s5 = np.isin(d5, days[-5:])
            l5, h5, v5 = (m5[c].values[s5].astype(float) for c in ("Low", "High", "Volume"))
            x["vp5r_v"] = vdens(l5, h5, v5, r.ent)
            x["poc5_v"] = r.ent / poc_of(l5, h5, v5) - 1
            lows = []
            for dd_ in days[-5:]:
                q = d5[s5] == dd_
                lows += piv_lows(l5[q], 3)
            # 오늘 정규장 3분봉, 매수 시각 앞 1분봉만으로
            day0 = pd.Timestamp(f"{str(D)[:4]}-{str(D)[4:6]}-{str(D)[6:]} 09:30", tz=ET)
            mm = m1[(m1.index >= day0) & (m1.index < r.ts)]
            if len(mm):
                b3 = mm.resample("3min", label="left", closed="left", origin=day0).agg(
                    {"Low": "min", "High": "max", "Close": "last"}).dropna(subset=["Close"])
                b3 = b3[b3.High > 0]
                lows += piv_lows(b3.Low.values.astype(float), 5)
            lows = np.array(lows)
            b = lows[lows <= r.ent]
            x["sw_b_v"] = r.ent / b.max() - 1 if len(b) else np.nan
            x["sw_n_v"] = int((np.abs(lows / r.ent - 1) <= 0.005).sum())
        # 정규장 꼬리 체결(분석가가 연장에 쓴 정의 그대로)
        w = m1.index.searchsorted(r.ts)
        e = m1.index.searchsorted(r.ts + pd.Timedelta(minutes=3))
        tail = np.nan
        for q in range(w, e):
            if m1.Low.values[q] <= r.ent * (1 + 1e-9):
                pcl = m1.Close.values[q - 1]
                tail = float(r.ent < pcl * 0.99)
                x["fill_open"] = float(q == w)
                break
        x["tail_v"] = tail
        out[k] = x
    return out


# ---------------- 고르기 절차(분석가와 같은 분할) ----------------

FEATS = [("pc", "sign"), ("ph", "sign"), ("pl", "sign"), ("op", "sign"), ("pml", "sign"), ("dl", "sign"), ("l5", "sign"),
         ("vw", "sign"), ("vw5", "sign"), ("poc5", "sign"), ("pos20", "pos"), ("vp5r", "vp"), ("vp20r", "vp"),
         ("sw_n", "count"), ("sw_b", "cont"), ("sw_a", "cont"), ("rnd_up", "bin"), ("rnd_dn", "bin"), ("rnd_f", "bin")]


def split_defs(x1, kind):
    """x1 = 앞 절반 값(nan 제외). (이름, 함수) 목록."""
    out = []
    if len(x1) < 20:
        return out
    if kind == "sign":
        out.append(("음수", lambda z: z < 0))
    if kind == "pos":
        out.append(("<0.5", lambda z: z < 0.5))
    if kind == "vp":
        out.append(("<1", lambda z: z < 1))
    if kind in ("sign", "pos", "vp", "cont"):
        q1, q2 = np.quantile(x1, [1 / 3, 2 / 3])
        out.append(("하위1/3", lambda z, v=q1: z <= v))
        out.append(("상위1/3", lambda z, v=q2: z > v))
    if kind == "count":
        out.append((">=1", lambda z: z >= 1))
        out.append((">=2", lambda z: z >= 2))
    if kind == "bin":
        out.append(("참", lambda z: z == 1))
    return out


def procedure(F, early, dc1, nd1, dc2, nd2, Y1, Y2, feats=FEATS, minn=15, mind=6, npick=3):
    """F: (n, nfeat) 배열(정규장 거래 순서). 반환: 시험표(list of dict)와 고른 3개."""
    tests = []
    for fi, (feat, kind) in enumerate(feats):
        col = F[:, fi]
        c1, c2 = col[early], col[~early]
        k1, k2 = ~np.isnan(c1), ~np.isnan(c2)
        for nm, fn in split_defs(c1[k1], kind):
            a1, a2 = fn(c1[k1]).astype(float), fn(c2[k2]).astype(float)
            for yn, (y1, y2) in (("y1", Y1), ("y2", Y2)):
                r1 = dd_np(dc1[k1], nd1, a1, y1[k1])
                r2 = dd_np(dc2[k2], nd2, a2, y2[k2])
                sg = 1 if (r1[0] if r1[0] == r1[0] else 0) >= 0 else -1
                tests.append(dict(feat=feat, split=nm, y=yn, t1=r1[1], nd1=r1[2], nA1=int(a1.sum()), nB1=int(len(a1) - a1.sum()),
                                  t2=r2[1], t2d=sg * r2[1] if r2[1] == r2[1] else np.nan, dd1=r1[0], dd2=r2[0]))
    T = pd.DataFrame(tests)
    ok = T[(T.nA1 >= minn) & (T.nB1 >= minn) & (T.nd1 >= mind)].copy()
    ok["s"] = ok.t1.abs()
    picks, seen = [], set()
    for _, r in ok.sort_values("s", ascending=False).iterrows():
        if r.feat in seen:
            continue
        seen.add(r.feat)
        picks.append(r)
        if len(picks) == npick:
            break
    return T, picks


def stats_of(T, picks):
    pt = np.array([p.t2d for p in picks], float)
    return dict(e2=int((T.t1.abs() >= 2).sum()), l2=int((T.t2.abs() >= 2).sum()), ld2=int((T.t2d >= 2).sum()),
                lmax=float(T.t2.abs().max()), pmin=float(np.nanmin(pt)) if len(pt) else np.nan,
                pmean=float(np.nanmean(pt)) if len(pt) else np.nan, p1=int((pt >= 1).sum()), p115=int((pt >= 1.15).sum()),
                emax=float(T.t1.abs().max()))


# ---------------- 사건 ----------------

def my_events(sym):
    d = bars(RAW, sym, 3)
    idx = d.index
    mod = idx.hour * 60 + idx.minute
    di = dint_of(idx.date)
    o, h, l, c = (d[x].values.astype(float) for x in ("Open", "High", "Low", "Close"))
    dd = pd.read_parquet(f"{RAW}/1d_{sym}.parquet")
    ddi = dint_of(pd.to_datetime(dd.index).date)
    C = dd.Close.values.astype(float)
    rows = []
    for D in np.unique(di):
        if D < 20260910 or D > 20261009:
            continue
        p = np.searchsorted(ddi, D, "left") - 1
        if p < 20:
            continue
        pc = C[p]
        q = np.where(di == D)[0]
        runmax = -np.inf
        for j in q:
            runmax = max(runmax, h[j])
            if not (570 <= mod[j] < 900):
                continue
            G = runmax / pc - 1
            if G >= 0.05 and c[j] <= pc * (1 + G / 2):
                rest = q[(q > j) & (mod[q] >= 570) & (mod[q] < 960)]
                z1 = c[rest[-1]] / c[j] - 1 if len(rest) and D <= 20261008 else np.nan
                z2 = float(c[rest].max() >= c[j] + (runmax - c[j]) / 2) if len(rest) and D <= 20261008 else np.nan
                rows.append(dict(sym=sym, dint=D, mod_v=mod[j], e_v=c[j], G_v=G, peak_v=runmax, z1_v=z1, z2_v=z2,
                                 hurdle=(runmax - c[j]) / 2 / c[j],
                                 rest_n=len(rest)))
                break
    return rows


EV_FEATS = [("G", "med"), ("pret", "sign"), ("ph", "sign"), ("pdo", "sign"), ("h5", "sign"), ("pmh", "sign"), ("pml", "med"),
            ("op", "sign"), ("vw", "sign"), ("vw5", "sign"), ("poc5", "sign"), ("vp5r", "vp"), ("sw_n", "count1")]


def ev_fn(x1, kind):
    if kind == "sign":
        return lambda z: z < 0
    if kind == "med":
        m = np.median(x1)
        return lambda z, v=m: z <= v
    if kind == "vp":
        return lambda z: z < 1
    return lambda z: z >= 1


def ev_procedure(F, early, dc1, nd1, dc2, nd2, Z):
    tests = []
    for fi, (feat, kind) in enumerate(EV_FEATS):
        col = F[:, fi]
        c1, c2 = col[early], col[~early]
        k1, k2 = ~np.isnan(c1), ~np.isnan(c2)
        if k1.sum() < 10:
            continue
        fn = ev_fn(c1[k1], kind)
        a1, a2 = fn(c1[k1]).astype(float), fn(c2[k2]).astype(float)
        for zn, (z1, z2) in Z.items():
            r1 = dd_np(dc1[k1], nd1, a1, z1[k1])
            r2 = dd_np(dc2[k2], nd2, a2, z2[k2])
            sg = 1 if (r1[0] if r1[0] == r1[0] else 0) >= 0 else -1
            tests.append(dict(feat=feat, z=zn, t1=r1[1], nA1=int(a1.sum()), nB1=int(len(a1) - a1.sum()), t2=r2[1],
                              t2d=sg * r2[1] if r2[1] == r2[1] else np.nan))
    T = pd.DataFrame(tests)
    ok = T[(T.nA1 >= 10) & (T.nB1 >= 10)].copy()
    ok["s"] = ok.t1.abs()
    picks, seen = [], set()
    for _, r in ok.sort_values("s", ascending=False).iterrows():
        if r.feat in seen:
            continue
        seen.add(r.feat)
        picks.append(r)
        if len(picks) == 3:
            break
    return T, picks


def control_report(e, a, y, label, q_atr):
    """e: 정규장 한 절반, a: bool 배열."""
    a = np.asarray(a, bool)
    r = dd_df(e, a, y)
    tod = dd_strat(e, a, y, (e["mod"] < 600).values)
    tod3 = dd_strat(e, a, y, np.select([e["mod"] < 579, e["mod"] < 600], [0, 1], 2))
    atr = dd_strat(e, a, y, np.select([e.atr_d <= q_atr[0], e.atr_d <= q_atr[1]], [0, 1], 2))
    vq = e.vol20.quantile([1 / 3, 2 / 3]).values
    vol = dd_strat(e, a, y, np.select([e.vol20 <= vq[0], e.vol20 <= vq[1]], [0, 1], 2))
    e2 = e.assign(open30=(e["mod"] < 600).astype(float), o9=(e["mod"] < 579).astype(float), latr=np.log(e.atr_d))
    rb, rt = fe_reg(e2, a, y, ["open30", "o9", "latr", "vol20"])
    # A 쪽 기여 최대 1종목·1일(효과를 가장 줄이는 것) 빼기
    sg = np.sign(r[0]) if r[0] == r[0] else 1
    worst_s, worst_d = (None, np.nan), (None, np.nan)
    for col, store in (("sym", "s"), ("dint", "d")):
        best = None
        for v in e[col].unique():
            k = (e[col] != v).values
            dd_, tt, _ = dd_df(e[k], a[k], y)
            if dd_ != dd_:
                continue
            if best is None or sg * dd_ < sg * best[1]:
                best = (v, dd_, tt)
        if store == "s":
            worst_s = best
        else:
            worst_d = best
    say(f"   [{label}] A {a.sum()}건 {pct(e[y][a].mean()) if y == 'y1' else f'{100 * e[y][a].mean():.1f}%'} vs B {(~a).sum()}건 "
        f"{pct(e[y][~a].mean()) if y == 'y1' else f'{100 * e[y][~a].mean():.1f}%'} | 날짜차 {pct(r[0])} t {r[1]:+.2f} ({r[2]}일)"
        f" | A 개장30분 {100 * (e['mod'][a] < 600).mean():.0f}% B {100 * (e['mod'][~a] < 600).mean():.0f}%")
    say(f"      층: 개장30분 t {tod[1]:+.2f} | 시각3층 t {tod3[1]:+.2f} | atr t {atr[1]:+.2f} | vol20 t {vol[1]:+.2f} | 회귀(날짜FE+개장30분+9:38전+log atr+vol20) {pct(rb)} t {rt:+.2f}"
          f" | -{worst_s[0]} t {worst_s[2]:+.2f} | -{worst_d[0]} t {worst_d[2]:+.2f}")
    return dict(t=r[1], tod=tod[1], tod3=tod3[1], atr=atr[1], vol=vol[1], reg=rt, xs=worst_s[2], xd=worst_d[2])


def main():
    t0 = time.time()
    say("반박 검증 — pat_levels 후보 (2026-10-10)")
    say("=" * 100)
    say("결과 변수(분석가가 시험 전에 정한 그대로): Y1 = ret, Y2 = 20봉(60분) 안 RSI 매도. 사건: 04시~봉 j 최고가 >= 전날 종가 +5%(G), 정규장 09:30~14:57")
    say("  3분봉 종가가 처음 전날 종가x(1+G/2) 이하인 봉 j, 판단 j+1 시가, e=c_j. Z1 = 정규장 마지막 종가/e-1, Z2 = 이후 종가 최고 >= e+(최고가-e)/2.")
    say("")
    t = pd.read_csv(os.path.join(RES, "trades.csv"))
    f = pd.read_csv(os.path.join(RES, "pat_levels_trades.csv"))
    assert len(t) == len(f) and (t.sym.values == f.sym.values).all() and (t.t.values == f.t.values).all()
    t["ts"] = pd.to_datetime(t.t, utc=True).dt.tz_convert(ET)
    t["dint"] = dint_of(t.ts.dt.date)
    t["mod"] = t.ts.dt.hour * 60 + t.ts.dt.minute
    t["y1"] = t.ret
    t["y2"] = ((t.why == "RSI") & (t.bars <= 20)).astype(float)
    for c in [x for x, _ in FEATS] + ["atr_d", "vol20", "tail", "h5", "pret"]:
        t[c] = f[c].values
    R = t[t.ses == "정규"].copy()
    say(f"[데이터] 정규장 {len(R)}건(앞 {(R.dint < MID).sum()}, 뒤 {(R.dint >= MID).sum()}), Y2 앞 {int(R[R.dint < MID].y2.sum())}/{(R.dint < MID).sum()}"
        f" 뒤 {int(R[R.dint >= MID].y2.sum())}/{(R.dint >= MID).sum()}, 개장30분 매수 {100 * (R['mod'] < 600).mean():.0f}%, 09:30~09:38 매수 {100 * (R['mod'] < 579).mean():.0f}%")

    # ---------- 0) 독립 재계산 ----------
    mine = {}
    for s, g in R.groupby("sym"):
        mine.update(my_feats(s, g))
    M = pd.DataFrame.from_dict(mine, orient="index")
    R = R.join(M)
    say("\n[0] 독립 재계산(1분봉은 매수 시각 앞, 일봉은 전날까지, 5분봉은 전날까지 정규장) vs 분석가 값, 정규장 678건")
    for a_, b_ in (("pos20", "pos20_v"), ("vp20r", "vp20r_v"), ("vp5r", "vp5r_v"), ("poc5", "poc5_v"), ("sw_b", "sw_b_v"), ("sw_n", "sw_n_v")):
        x, y = R[a_].values.astype(float), R[b_].values.astype(float)
        both = ~np.isnan(x) & ~np.isnan(y)
        nmis = int((np.isnan(x) != np.isnan(y)).sum())
        dif = np.abs(x[both] - y[both])
        say(f"  {a_:6s}: 둘 다 값 {both.sum()}건, 최대 차 {dif.max():.2e}, 1e-6 넘게 다름 {int((dif > 1e-6).sum())}건, 한쪽만 nan {nmis}건")
    tl = R.tail_v
    say(f"  정규장 꼬리 체결(체결 분 직전 1분 종가보다 1% 넘게 아래): {int((tl == 1).sum())}건 / 판정 {int(tl.notna().sum())}건"
        f" (그중 매수 봉 첫 분 체결 {int(((tl == 1) & (R.fill_open == 1)).sum())}건). 꼬리 Y1 {pct(R[tl == 1].y1.mean())} 깨끗 {pct(R[tl == 0].y1.mean())}")

    # ---------- 1~3) 후보 재현·통제 ----------
    q_atr = R[R.dint < MID].atr_d.quantile([1 / 3, 2 / 3]).values
    E1, E2 = R[R.dint < MID], R[R.dint >= MID]
    cands = [("vp20r", lambda z: z < 1, "vp20r<1", "y1"),
             ("sw_b", lambda z, v=np.quantile(E1.sw_b.dropna(), 2 / 3): z > v, f"sw_b>{np.quantile(E1.sw_b.dropna(), 2 / 3):.4f}", "y1"),
             ("pos20", lambda z, v=np.quantile(E1.pos20.dropna(), 2 / 3): z > v, f"pos20>{np.quantile(E1.pos20.dropna(), 2 / 3):.4f}", "y1")]
    say("\n[1~3] 거래 후보 재현 + 통제 + 꼬리 체결 제외")
    out = {}
    for feat, fn, rule, y in cands:
        say(f"\n  후보 {rule} ({y}, 앞 방향 A가 나쁨)")
        res = {}
        for lab, e in (("앞", E1), ("뒤", E2)):
            e = e[e[feat].notna()]
            res[lab] = control_report(e, fn(e[feat].values), y, lab, q_atr)
        e = E2[E2[feat].notna() & (E2.tail_v == 0)]
        r = dd_df(e, fn(e[feat].values), y)
        say(f"   [뒤, 꼬리 체결 뺌 {len(e)}건] 날짜차 {pct(r[0])} t {r[1]:+.2f}")
        for sk in ("sym", "dint"):
            top = E2[sk].value_counts().index[0]
            e = E2[E2[feat].notna() & (E2[sk] != top)]
            r = dd_df(e, fn(e[feat].values), y)
            say(f"   [뒤, 가장 많은 {sk} {top}({(E2[sk] == top).sum()}건) 뺌] 날짜차 {pct(r[0])} t {r[1]:+.2f}")
        out[feat] = res
    A = pd.DataFrame({c: fn(R[c].values) for c, fn, _, _ in cands}).astype(float)
    say(f"\n  후보끼리 겹침(정규장 A 지표 상관): vp20r-sw_b {A.vp20r.corr(A.sw_b):+.2f}, vp20r-pos20 {A.vp20r.corr(A.pos20):+.2f}, sw_b-pos20 {A.sw_b.corr(A.pos20):+.2f}")
    say(f"  vp20r<1과 20일 범위 끝 근처(pos20<0.1 또는 >0.9): A 중 {100 * ((R.pos20 < 0.1) | (R.pos20 > 0.9))[R.vp20r < 1].mean():.0f}% vs B 중 {100 * ((R.pos20 < 0.1) | (R.pos20 > 0.9))[R.vp20r >= 1].mean():.0f}%")

    # ---------- 4) 우연: 같은 절차 무작위 ----------
    say("\n[4] 우연 점검: 같은 고르기 절차(정규장 96시험 → 앞 |t| 상위 서로 다른 특성 3개 → 뒤 앞-방향 t)")
    R = R.reset_index(drop=True)
    early = (R.dint < MID).values
    dc1, u1 = pd.factorize(R.dint.values[early])
    dc2, u2 = pd.factorize(R.dint.values[~early])
    Y1 = (R.y1.values[early], R.y1.values[~early])
    Y2 = (R.y2.values[early], R.y2.values[~early])
    names = [x for x, _ in FEATS]
    F = R[names].values.astype(float)
    T0, P0 = procedure(F, early, dc1, len(u1), dc2, len(u2), Y1, Y2)
    s0 = stats_of(T0, P0)
    say(f"  실제: 시험 {len(T0)}개, 고른 3개 = " + ", ".join(f"{p.feat} {p.split} {p.y}(앞 t {p.t1:+.2f}, 뒤 앞-방향 t {p.t2d:+.2f})" for p in P0))
    say(f"  실제 통계: 앞 |t|>=2 {s0['e2']}개, 뒤 |t|>=2 {s0['l2']}개, 뒤 앞-방향 t>=2 {s0['ld2']}개, 뒤 최대 |t| {s0['lmax']:.2f},"
        f" 고른 3개 뒤 t 최소 {s0['pmin']:+.2f} 평균 {s0['pmean']:+.2f}, 뒤 t>=1인 것 {s0['p1']}/3")
    dates = R.dint.values
    perm_a = []
    for it in range(500):
        idx = np.arange(len(R))
        for D in np.unique(dates):
            q = np.where(dates == D)[0]
            idx[q] = rng.permutation(q)
        T, P = procedure(F[idx], early, dc1, len(u1), dc2, len(u2), Y1, Y2)
        perm_a.append(stats_of(T, P))
    PA = pd.DataFrame(perm_a)
    # (b) 종목 바꿔 끼우기
    syms = R.sym.unique()
    by = {s: R.index[R.sym == s].values for s in syms}
    tsn = R.ts.values.astype("datetime64[ns]").astype(np.int64)
    perm_b = []
    for it in range(300):
        pi = dict(zip(syms, rng.permutation(syms)))
        idx = np.empty(len(R), int)
        for s in syms:
            src = by[pi[s]]
            for k in by[s]:
                idx[k] = src[np.argmin(np.abs(tsn[src] - tsn[k]))]
        T, P = procedure(F[idx], early, dc1, len(u1), dc2, len(u2), Y1, Y2)
        perm_b.append(stats_of(T, P))
    PB = pd.DataFrame(perm_b)
    for lab, P in (("(a) 날짜 안 섞기 500회", PA), ("(b) 종목 바꿔 끼우기 300회", PB)):
        say(f"  {lab}:")
        say(f"    앞 |t|>=2 개수: 중앙 {P.e2.median():.0f}, 90% {P.e2.quantile(0.9):.0f} → 실제 {s0['e2']}개 이상 비율 {(P.e2 >= s0['e2']).mean():.2f}")
        say(f"    고른 3개 뒤 t 평균: 중앙 {P.pmean.median():+.2f}, 90% {P.pmean.quantile(0.9):+.2f}, 95% {P.pmean.quantile(0.95):+.2f} → 실제 {s0['pmean']:+.2f} 이상 비율 {(P.pmean >= s0['pmean']).mean():.3f}")
        say(f"    고른 3개 모두 뒤 t>=1.15: 비율 {(P.p115 == 3).mean():.3f} | 고른 3개 뒤 t 최소 >= 실제 {s0['pmin']:+.2f}: 비율 {(P.pmin >= s0['pmin']).mean():.3f}")
        say(f"    뒤 |t|>=2 개수: 중앙 {P.l2.median():.0f}, 95% {P.l2.quantile(0.95):.0f} → 실제 {s0['l2']}개 이상 비율 {(P.l2 >= s0['l2']).mean():.3f}")
        say(f"    뒤 앞-방향 t>=2 개수: 중앙 {P.ld2.median():.0f}, 95% {P.ld2.quantile(0.95):.0f} → 실제 {s0['ld2']}개 이상 비율 {(P.ld2 >= s0['ld2']).mean():.3f}")
        say(f"    뒤 최대 |t|(관찰 poc5 고르기와 같은 일): 중앙 {P.lmax.median():.2f}, 95% {P.lmax.quantile(0.95):.2f} → 실제 {s0['lmax']:.2f} 이상 비율 {(P.lmax >= s0['lmax']).mean():.3f}")

    # 뒤 절반 전체가 이상한가: 10/6 하루 빼고 뒤 |t|>=2 개수
    k = R.dint.values != 20261006
    e_ = early[k]
    dc1b, u1b = pd.factorize(R.dint.values[k][e_])
    dc2b, u2b = pd.factorize(R.dint.values[k][~e_])
    Tb, Pb = procedure(F[k], e_, dc1b, len(u1b), dc2b, len(u2b), (R.y1.values[k][e_], R.y1.values[k][~e_]), (R.y2.values[k][e_], R.y2.values[k][~e_]))
    say(f"  10/6 빼면: 뒤 |t|>=2 {int((Tb.t2.abs() >= 2).sum())}개, 고른 3개 뒤 t " + ", ".join(f"{p.feat} {p.t2d:+.2f}" for p in Pb))

    # 부분 표본에서 같은 절차 + 날짜 안 섞기 300회
    auction = (R.tail_v == 1) & (R.fill_open == 1) & (R["mod"] == 570)
    say(f"  정규장 꼬리 89건 내역: 09:30 첫 분(개장 경매 갭) {int(auction.sum())}건, 그 밖 장중 튄 체결 {int(((R.tail_v == 1) & ~auction).sum())}건")
    subsets = {"10/6 뺌": R.dint.values != 20261006,
               "장중 튄 체결 뺌(개장 경매 갭은 둠)": ~(((R.tail_v == 1) & ~auction).values),
               "꼬리 전부 뺌": (R.tail_v == 0).values}
    for lab, k in subsets.items():
        e_ = early[k]
        c1_, uu1 = pd.factorize(R.dint.values[k][e_])
        c2_, uu2 = pd.factorize(R.dint.values[k][~e_])
        Ya = (R.y1.values[k][e_], R.y1.values[k][~e_])
        Yb = (R.y2.values[k][e_], R.y2.values[k][~e_])
        Fk = F[k]
        Ts, Ps = procedure(Fk, e_, c1_, len(uu1), c2_, len(uu2), Ya, Yb)
        ss = stats_of(Ts, Ps)
        dk = R.dint.values[k]
        pm = []
        for it in range(300):
            idx = np.arange(k.sum())
            for D in np.unique(dk):
                q = np.where(dk == D)[0]
                idx[q] = rng.permutation(q)
            T, P = procedure(Fk[idx], e_, c1_, len(uu1), c2_, len(uu2), Ya, Yb)
            pm.append(stats_of(T, P)["pmean"])
        pm = np.array(pm)
        say(f"  [{lab}, {k.sum()}건] 고른 3개 " + ", ".join(f"{p.feat} {p.split} {p.y}(앞 {p.t1:+.2f}, 뒤 {p.t2d:+.2f})" for p in Ps)
            + f" | 뒤 t 평균 {ss['pmean']:+.2f}, 날짜 안 섞기 300회 중 이 이상 {np.mean(pm >= ss['pmean']):.3f}")

    # ---------- 5a) 관찰 poc5<0 ----------
    say("\n[5a] 관찰 poc5<0(뒤 절반을 보고 고름) — 통제·쏠림")
    for lab, e in (("앞", E1), ("뒤", E2)):
        e = e[e.poc5.notna()]
        control_report(e, (e.poc5 < 0).values, "y1", lab, q_atr)
    e = E2[E2.poc5.notna() & (E2.tail_v == 0)]
    r = dd_df(e, (e.poc5 < 0).values, "y1")
    say(f"   [뒤, 꼬리 체결 뺌] 날짜차 {pct(r[0])} t {r[1]:+.2f}")
    # poc5와 다른 후보 상관(같은 신호의 거울인가)
    say(f"   poc5<0 vs pos20 상위1/3 상관 {np.corrcoef((R.poc5 < 0).astype(float), A.pos20)[0, 1]:+.2f}, vs vp20r<1 {np.corrcoef((R.poc5 < 0).astype(float), A.vp20r)[0, 1]:+.2f}")

    # ---------- 사건 ----------
    say("\n" + "=" * 100)
    say("[사건] 독립 재탐지 + 재현 + 우연 + G 관찰")
    ev = pd.read_csv(os.path.join(RES, "pat_levels_events.csv"))
    ev["dint"] = ev.dint.astype(np.int64)
    mine = []
    for s in sorted(t.sym.unique()):
        mine += my_events(s)
    ME = pd.DataFrame(mine)
    J = ev.merge(ME, on=["sym", "dint"], how="outer", indicator=True)
    say(f"  분석가 사건 {len(ev)}건(10/9 포함), 내 재탐지 {len(ME)}건, 양쪽 {int((J._merge == 'both').sum())}건, 분석가만 {int((J._merge == 'left_only').sum())}, 나만 {int((J._merge == 'right_only').sum())}")
    b = J[J._merge == "both"]
    say(f"  e 최대 차 {np.nanmax(np.abs(b.e - b.e_v)):.2e}, G 최대 차 {np.nanmax(np.abs(b.G - b.G_v)):.2e}, z1 최대 차 {np.nanmax(np.abs(b.z1 - b.z1_v)):.2e},"
        f" z2 다른 건 {int(((b.z2 != b.z2_v) & b.z2.notna()).sum())}")
    full = ev[ev.z1.notna()].merge(ME[["sym", "dint", "hurdle", "rest_n"]], on=["sym", "dint"], how="left").reset_index(drop=True)
    full["mod"] = full["mod"].astype(int)
    say(f"  완성 사건 {len(full)}건, 종목 {full.sym.nunique()}, 날짜 {full.dint.nunique()}, 앞 {(full.dint < MID).sum()} / 뒤 {(full.dint >= MID).sum()}, Z2 {100 * full.z2.mean():.0f}%")
    ee = (full.dint < MID).values
    edc1, eu1 = pd.factorize(full.dint.values[ee])
    edc2, eu2 = pd.factorize(full.dint.values[~ee])
    Z = {"z1": (full.z1.values[ee], full.z1.values[~ee]), "z2": (full.z2.values[ee], full.z2.values[~ee])}
    EF = full[[x for x, _ in EV_FEATS]].values.astype(float)
    ET0, EP0 = ev_procedure(EF, ee, edc1, len(eu1), edc2, len(eu2), Z)
    say(f"  실제: 시험 {len(ET0)}개, 앞 |t|>=2 {int((ET0.t1.abs() >= 2).sum())}개, 고른 3개 = "
        + ", ".join(f"{p.feat} {p.z}(앞 t {p.t1:+.2f}, 뒤 앞-방향 t {p.t2d:+.2f})" for p in EP0))
    ev_perm = []
    edates = full.dint.values
    for it in range(500):
        idx = np.arange(len(full))
        for D in np.unique(edates):
            q = np.where(edates == D)[0]
            idx[q] = rng.permutation(q)
        T, P = ev_procedure(EF[idx], ee, edc1, len(eu1), edc2, len(eu2), Z)
        pt = np.array([p.t2d for p in P])
        ev_perm.append(dict(e2=int((T.t1.abs() >= 2).sum()), pmean=np.nanmean(pt), emax=T.t1.abs().max(), l2d=int((T.t2d >= 1.9).sum())))
    EPm = pd.DataFrame(ev_perm)
    say(f"  날짜 안 섞기 500회: 앞 |t|>=2 개수 중앙 {EPm.e2.median():.0f}, 90% {EPm.e2.quantile(0.9):.0f} → 실제 {int((ET0.t1.abs() >= 2).sum())}개 이상 비율 {(EPm.e2 >= (ET0.t1.abs() >= 2).sum()).mean():.2f};"
        f" 고른 3개 뒤 t 평균 중앙 {EPm.pmean.median():+.2f} → 실제 {np.mean([p.t2d for p in EP0]):+.2f} 이상 비율 {(EPm.pmean >= np.mean([p.t2d for p in EP0])).mean():.2f}")
    say(f"  26시험 중 뒤 앞-방향 t>=1.9(관찰 G Z2 수준) 개수: 실제 {int((ET0.t2d >= 1.9).sum())}, 무작위 중앙 {EPm.l2d.median():.0f}, 1개 이상 비율 {(EPm.l2d >= 1).mean():.2f}")
    full["open30"] = (full["mod"] < 600).astype(float)
    full["latr"] = np.log(full.atr_d)
    full["lG"] = np.log(full.G)
    full["lhur"] = np.log(full.hurdle / full.atr_d)
    full["lrest"] = np.log(full.rest_n)
    for feat, fn, z in (("vw", lambda x: x < 0, "z2"), ("vp5r", lambda x: x < 1, "z2"), ("h5", lambda x: x < 0, "z1")):
        for lab, e in (("앞", full[ee]), ("뒤", full[~ee])):
            e = e[e[feat].notna()]
            a = fn(e[feat].values)
            r = dd_df(e, a, z)
            rb, rt = fe_reg(e, a, z, ["open30", "lG", "latr"])
            say(f"  [사건 후보 {feat} {z} {lab}] A {a.sum()} B {(~a).sum()} 날짜차 {r[0]:+.4f} t {r[1]:+.2f} ({r[2]}일) | 회귀 t {rt:+.2f}")

    # ---------- 5b) G 관찰: 기계적인가 ----------
    say("\n[5b] 관찰 G<=6.7% → Z2 높음: Z2 문턱 = (최고가-e)/2 이므로 G가 작으면 문턱이 작다(기계적).")
    thr = np.median(full[ee].G)
    say(f"  앞 절반 G 중앙값 {thr:.4f}. 문턱 hurdle=(최고가-e)/(2e): G 작음 중앙 {pct(full[full.G <= thr].hurdle.median())}, 큼 중앙 {pct(full[full.G > thr].hurdle.median())};"
        f" hurdle/atr_d 작음 {full[full.G <= thr].hurdle.div(full[full.G <= thr].atr_d).median():.2f} 큼 {full[full.G > thr].hurdle.div(full[full.G > thr].atr_d).median():.2f}")
    for lab, e in (("앞", full[ee]), ("뒤", full[~ee])):
        a = (e.G <= thr).values
        r = dd_df(e, a, "z2")
        r1 = dd_df(e, a, "z1")
        rb, rt = fe_reg(e, a, "z2", ["open30", "latr"])
        rb2, rt2 = fe_reg(e, a, "z2", ["open30", "lhur", "lrest"])
        hq = e.lhur.quantile([1 / 3, 2 / 3]).values
        hs = dd_strat(e, a, "z2", np.select([e.lhur <= hq[0], e.lhur <= hq[1]], [0, 1], 2))
        say(f"  [{lab}] Z2 A {100 * e.z2[a].mean():.0f}% B {100 * e.z2[~a].mean():.0f}% 날짜차 t {r[1]:+.2f} | Z1 날짜차 {pct(r1[0])} t {r1[1]:+.2f}"
            f" | 회귀(개장30분+log atr) t {rt:+.2f} | 회귀(개장30분+log(문턱/atr)+log 남은 봉) t {rt2:+.2f} | 문턱/atr 3분위 층 t {hs[1]:+.2f}")
    # 무작위 걷기 기대: 문턱만으로 Z2 맞히기(같은 날 다른 사건 표본에서 문턱/atr 순위로)
    for lab, e in (("앞", full[ee]), ("뒤", full[~ee])):
        hq = e.lhur.quantile([1 / 3, 2 / 3]).values
        grp = np.select([e.lhur <= hq[0], e.lhur <= hq[1]], [0, 1], 2)
        say(f"  [{lab}] 문턱/atr 3분위별 Z2: " + " / ".join(f"{100 * e.z2[grp == g].mean():.0f}%({(grp == g).sum()})" for g in range(3)))

    # ---------- 6) 10/9 사례 ----------
    say("\n[6] 10/9 사례(10:18 ET까지 자료) — 규칙 그대로 대입")
    cs = pd.read_csv(os.path.join(RES, "pat_levels_cases.csv"))
    q_sw = np.quantile(E1.sw_b.dropna(), 2 / 3)
    q_pos = np.quantile(E1.pos20.dropna(), 2 / 3)
    for _, r in cs.iterrows():
        ts = pd.Timestamp(r.t)
        row = pd.DataFrame([dict(sym=r.sym, dint=20261009, ts=ts, ent=r.ent)])
        mf = my_feats(r.sym, row)[0]
        flags = dict(vp20r=("나쁨" if mf["vp20r_v"] < 1 else "보통"), sw_b=("나쁨" if mf["sw_b_v"] > q_sw else "보통"),
                     pos20=("나쁨" if mf["pos20_v"] > q_pos else "보통"), poc5=("좋음" if mf["poc5_v"] < 0 else "보통"))
        say(f"  {r.sym} {ts:%H:%M} 체결 {r.ent:.2f}, 10:18 {pct(r.y_now)} ({r.rsi_sell}) | vp20r {mf['vp20r_v']:.2f}({flags['vp20r']}) sw_b {pct(mf['sw_b_v'])}({flags['sw_b']})"
            f" pos20 {mf['pos20_v']:.2f}({flags['pos20']}) poc5 {pct(mf['poc5_v'])}({flags['poc5']}) | 분석가 값 vp20r {r.vp20r:.2f} sw_b {pct(r.sw_b)} pos20 {r.pos20:.2f} poc5 {pct(r.poc5)}")
    e9 = ev[(ev.dint == 20261009) & ev.sym.isin(["LITE", "COHR"])]
    for _, r in e9.iterrows():
        say(f"  사건 {r.sym} {str(r.t)[11:16]} e {r.e:.2f} G {pct(r.G)} (G<=6.7%? {'예' if r.G <= thr else '아니오'}) | vw {pct(r.vw)}(vw<0? {'예' if r.vw < 0 else '아니오'})"
            f" vp5r {r.vp5r:.2f}(<1? {'예' if r.vp5r < 1 else '아니오'}) h5 {pct(r.h5)}(<0? {'예' if r.h5 < 0 else '아니오'}) 문턱 {pct((r.peak - r.e) / 2 / r.e)}")
    for s in ("LITE", "COHR"):
        m1 = pd.read_parquet(f"{RAW}/1m_{s}.parquet")
        m1 = m1[m1.index >= pd.Timestamp("2026-10-09 09:30", tz=ET)]
        r = e9[e9.sym == s].iloc[0]
        after = m1[m1.index >= pd.Timestamp(r.t) + pd.Timedelta(minutes=3)]
        say(f"   {s} 사건 뒤~10:18 종가 최고 {after.Close.max():.2f}({pct(after.Close.max() / r.e - 1)}), 절반 회복선 {r.e + (r.peak - r.e) / 2:.2f}, 10:18 {after.Close.iloc[-1]:.2f}({pct(after.Close.iloc[-1] / r.e - 1)})")
    say(f"\n시험 수: 거래 {len(T0)}개(정규)+사건 {len(ET0)}개 재현, 무작위 절차 (a) 500회 (b) 300회, 사건 500회. 실행 {time.time() - t0:.0f}초")
    with open(os.path.join(RES, "verify_pat_levels.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
