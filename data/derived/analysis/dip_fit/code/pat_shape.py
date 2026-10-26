"""이격매수매도 — 매수 직전 3분봉 '하락 모양'으로 빠른 반등·회복을 미리 알 수 있나 (2026-10-10).

질문(사용자, 10/9 장중): CRCL은 정규장 신호 뒤 빠르게 반등했고 MRVL은 계속 빠졌다. LITE·COHR은 광자 테마로 프리장부터 오르고
개장 직후 더 쏜 뒤 되돌렸는데 LITE만 회복했다. 3분봉 모양으로 미리 알 수 있었나.
이 파일은 '하락 모양(매수 직전 3분봉 경로)' 갈래다.

결과 변수(시험 전에 고정):
  Y1 = 거래 수익 ret(지금 매도 규칙, trades.csv)
  Y2 = 빠른 반등 = 매수 뒤 20봉(60분) 안에 RSI 매도(why == RSI 이고 bars <= 20)
  보조(고르기에 안 씀) r60 = 매수 봉 포함 20번째 봉 종가 / 매수가 - 1
사건 연구(따로 정의):
  E1 '테마 급등 뒤 되돌림' 부분집단 = 정규장 매수 중 매수 직전까지 그날(04시~) 고점이 전날 정규장 종가 +5% 이상인 매수. 결과는 Y1·Y2 그대로.
  E2 10/9 사례 = 같은 규칙으로 10/9 매수를 재현(1분봉이 10:18 ET에서 끝나 10:15 봉까지만 씀).
     결과 = 10:15 봉까지 RSI 매도 여부와 10:15 봉 종가 기준 평가 수익. 하루치라 설명용.

특성(모두 매수 봉 i 시가 시점에 앎: 봉 i-1까지 값, 지난 날 값, 지표 선, 봉 i 시가, 체결 사실·매수가):
  dd_day 매수가/그날(04시~) 고점 -1      kday 그날 고점 뒤 지난 봉 수       dd20 매수가/직전 20봉 고점 -1
  spd20 dd20/(20봉 고점 뒤 봉 수)(봉당 낙폭)  sl5·sl10 직전 5·10봉 종가 기울기     ndown 연속 음봉 수(i-1부터 거꾸로)
  big1 직전 5봉 중 가장 큰 한 봉 하락       conc big1 / 직전 5봉 하락 합(1이면 한 봉 폭포, 작으면 흘러내림)
  lw1 직전 봉 아래꼬리 비율               o_line 봉 i 시가/매수선 -1(선 위에서 열렸다가 봉 안에서 닿았나)
  rsi1 직전 봉 RSI14                     ema_sl EMA60 직전 20봉 기울기       vol5 직전 5봉 거래량/지난 10일 같은 시각 평균(정규장)
  tmin 개장 뒤 분                        gap 그날 개장 갭(09:30 시가/전날 종가 -1)
  spike 그날 고점/전날 종가 -1             retr 그날 상승분 중 되돌린 비율((고점-매수가)/(고점-전날 종가), 상승 2% 이상일 때)
  d_prev 매수가/전날 정규장 종가 -1
방법: 정규장이 주 결과. 문턱(하위1/3·상위1/3)과 방향은 앞 절반(9/10~9/23)에서 정해 뒤 절반(9/24~10/8)에 그대로 쓴다.
  t = 날짜 군집(날짜마다 A평균-B평균). 통제 = 같은 날·같은 시간대(개장 30분 안/나머지)·같은 atr14 삼분위 칸 안 차이의 날짜 평균,
  ret/atr14. 상위 1종목·1일(규칙 방향 차이를 가장 줄이는 것) 빼고 다시 잼.
  고르기: 정규장 주 시험에서 최대 2개 + E1 부분집단에서 1개(질문이 두 갈래라 시험 전에 2+1로 정함).
  점수 = 앞 절반에서 원 t와 통제 t 부호가 같을 때 min(|t|). Y2는 A의 사건 5건 이상일 때만.
  프리·애프터는 '깨끗한 체결'(닿은 1분봉의 직전 1분 종가가 매수가 +1% 이하)만 보조로 잰다(고르기에 안 씀).
  앞 기간 보조: stop_lab.load("early")(7/14~9/8 5분봉, EMA36·RSI9). 창 길이는 분 단위로 맞춘다(20봉→12봉, 10→6, 5→3). 문턱은 그 기간 삼분위.
입력(다시 받지 않음): ../results/trades.csv, ../results/features.csv, raw_1009 1분봉, stop_lab early 캐시
출력: ../results/pat_shape.txt, pat_shape_tests.csv, pat_shape_feats.csv, pat_shape_1009.csv
사용: py pat_shape.py"""
import os, sys, bisect
from datetime import date
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
from events import bars, prep, tod_rvol
from rules import hot_days
import build
import stop_lab

RAW = build.RAW
RES = os.path.join(HERE, "../results")
CUT = pd.Timestamp("2026-10-09 10:18", tz="America/New_York")      # 1분봉 마지막(미완성) 봉 시작 — 10:15 봉까지 씀
E2DAY = date(2026, 10, 9)
FEATS = ["dd_day", "kday", "dd20", "spd20", "sl5", "sl10", "ndown", "big1", "conc", "lw1", "o_line", "rsi1", "ema_sl",
         "vol5", "tmin", "gap", "spike", "retr", "d_prev"]
EXT_FEATS = [f for f in FEATS if f not in ("vol5", "tmin", "gap")]       # 프리·애프터에 의미 있는 것
E1_FEATS = ["retr", "d_prev", "dd_day", "kday", "rsi1", "vol5", "tmin", "big1"]
SPIKE = 0.05
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    print(s)


def pct(x):
    return "nan" if pd.isna(x) else f"{100 * x:+.2f}%"


# ---------------------------------------------------------------- 특성 계산
def prev_close_map(day, ses, c):
    """날짜 -> 직전 거래일 정규장 종가, 그날 정규장 종가."""
    reg = pd.Series(c[ses == "정규"], index=day[ses == "정규"])
    rc = reg.groupby(level=0).last()
    ds = list(rc.index)
    def prev(dt):
        k = bisect.bisect_left(ds, dt)
        return rc.iloc[k - 1] if k > 0 else np.nan
    return prev, rc.to_dict()


def feat_at(o, h, l, c, r, ema, day, ses, mod, rv5, i, ent, gap_pct, prev, today_close, W=(20, 10, 5)):
    """봉 i 시가 시점 특성. W = (20봉, 10봉, 5봉) 창(5분봉은 12, 6, 3)."""
    w20, w10, w5 = W
    line = ema[i - 1] * (1 - gap_pct)
    dt = day[i]
    f = {}
    j0 = i
    while j0 > 0 and day[j0 - 1] == dt:
        j0 -= 1
    if j0 < i:
        k = j0 + int(np.argmax(h[j0:i]))
        dh = h[k]
        f["dd_day"], f["kday"] = ent / dh - 1, i - k
    else:
        dh = np.nan
        f["dd_day"] = f["kday"] = np.nan
    ww = h[i - w20:i]
    k = int(np.argmax(ww))
    f["dd20"] = ent / ww[k] - 1
    f["spd20"] = f["dd20"] / (w20 - k)
    f["sl5"] = c[i - 1] / c[i - 1 - w5] - 1
    f["sl10"] = c[i - 1] / c[i - 1 - w10] - 1
    n = 0
    while n < 30 and c[i - 1 - n] < o[i - 1 - n]:
        n += 1
    f["ndown"] = n
    rr = c[i - w5:i] / c[i - w5 - 1:i - 1] - 1
    f["big1"] = rr.min()
    neg = np.minimum(rr, 0).sum()
    f["conc"] = rr.min() / neg if neg < 0 else np.nan
    rg = h[i - 1] - l[i - 1]
    f["lw1"] = (min(o[i - 1], c[i - 1]) - l[i - 1]) / rg if rg > 0 else np.nan
    f["o_line"] = o[i] / line - 1
    f["rsi1"] = r[i - 1]
    f["ema_sl"] = ema[i - 1] / ema[i - 1 - w20] - 1
    f["vol5"] = rv5[i] if ses[i] == "정규" and np.isfinite(rv5[i]) else np.nan
    pc = today_close.get(dt, np.nan) if ses[i] == "애프터" else prev(dt)
    if ses[i] == "정규":
        f["tmin"] = mod[i] - 570
        jo = j0
        while jo <= i and ses[jo] != "정규":
            jo += 1
        f["gap"] = o[jo] / pc - 1 if jo <= i else np.nan
    else:
        f["tmin"] = f["gap"] = np.nan
    f["spike"] = dh / pc - 1
    f["retr"] = (dh - ent) / (dh - pc) if (dh - pc) > 0.02 * pc else np.nan
    f["d_prev"] = ent / pc - 1
    return f


def sym_bars(s):
    d = prep(bars(RAW, s, 3), 60, 14)
    return d[d.index < CUT]


def trade_feats(t):
    rows = []
    for s, g in t.groupby("sym"):
        d = sym_bars(s)
        o, h, l, c, r, ema = (d[k].values.astype(float) for k in ("o", "h", "l", "c", "r", "ema"))
        day, ses, mod = np.asarray(d.date), d.ses.values, d["mod"].values
        rv5 = tod_rvol(d, 5, 10)
        prev, tc = prev_close_map(day, ses, c)
        gp = 0.03 if s in build.LEV else 0.02
        idx = d.index.get_indexer(g.ts)
        m1 = pd.read_parquet(os.path.join(RAW, f"1m_{s}.parquet"))
        for (ix, tr), i in zip(g.iterrows(), idx):
            if i < 0:
                continue
            f = feat_at(o, h, l, c, r, ema, day, ses, mod, rv5, i, tr.ent, gp, prev, tc)
            f["ix"] = ix
            f["ent_chk"] = min(o[i], ema[i - 1] * (1 - gp)) / tr.ent - 1
            f["r60"] = c[i + 19] / tr.ent - 1 if i + 19 < len(c) else np.nan
            # 1분 꼬리 체결: 닿은 1분봉의 직전 1분 종가가 매수가보다 1% 넘게 위
            t0 = tr.ts
            w = m1[(m1.index >= t0) & (m1.index < t0 + pd.Timedelta(minutes=3))]
            hit = w.index[w.Low <= tr.ent * (1 + 1e-6)]
            prv = np.nan
            if len(hit):
                p1 = m1.index.get_loc(hit[0])
                prv = m1.Close.iloc[p1 - 1] / tr.ent - 1 if p1 > 0 else np.nan
            f["prv1m"] = prv
            rows.append(f)
    F = pd.DataFrame(rows).set_index("ix")
    return t.join(F)


# ---------------------------------------------------------------- 통계
def tstat(x):
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return (x.mean() if n else np.nan), np.nan, n
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(n)), n


def cell_diff(d, a, y, keys):
    """keys 칸마다 A평균-B평균 → 날짜 안 평균 → 날짜 t."""
    g = d.assign(_a=a.values).groupby(keys + ["_a"])[y].mean().unstack()
    if True not in g.columns or False not in g.columns:
        return np.nan, np.nan, 0
    x = (g[True] - g[False]).dropna()
    if not len(x):
        return np.nan, np.nan, 0
    return tstat(x.groupby(level=0).mean())


def stats(d, a, sgn, ctrl_keys, full=True, ys=("1", "2")):
    A, B = d[a], d[~a]
    r = dict(nA=len(A), nB=len(B), symA=A.sym.nunique(), mA=A.ret.mean(), mB=B.ret.mean(),
             y2A=A.y2.mean(), y2B=B.y2.mean(), y2nA=int(A.y2.sum()), y2nB=int(B.y2.sum()),
             r60A=A.r60.mean(), r60B=B.r60.mean())
    for y, p in (("ret", "1"), ("y2", "2")):
        r[f"dd{p}"], r[f"t{p}"], r["nd"] = cell_diff(d, a, y, ["day"])
        r[f"cdd{p}"], r[f"ct{p}"], _ = cell_diff(d, a, y, ["day"] + ctrl_keys)
    if not full:
        return r
    r["ra_dd"], r["ra_t"], _ = cell_diff(d, a, "ra", ["day"])
    r["r60_dd"], r["r60_t"], _ = cell_diff(d, a, "r60", ["day"])
    for key, col in (("xs", "sym"), ("xd", "day")):
        for y, p in (("ret", "1"), ("y2", "2")):
            if p not in ys:
                r[f"{key}{p}"], r[f"{key}{p}_dd"], r[f"{key}{p}_t"] = None, np.nan, np.nan
                continue
            best = None
            for v in d[col].unique():
                k = d[col] != v
                dd, tt, _ = cell_diff(d[k], a[k], y, ["day"])
                if np.isnan(dd):
                    continue
                if best is None or sgn * dd < sgn * best[1]:
                    best = (v, dd, tt)
            r[f"{key}{p}"], r[f"{key}{p}_dd"], r[f"{key}{p}_t"] = best if best else (None, np.nan, np.nan)
    return r


def fmt(r, p):
    g = lambda k: r[f"{p}_{k}"]
    s = (f"A {g('nA')}건/{g('symA')}종목 vs B {g('nB')}건 | Y1 {pct(g('mA'))} vs {pct(g('mB'))} 날짜차 {pct(g('dd1'))} t={g('t1'):+.2f}"
         f" 통제 {pct(g('cdd1'))} t={g('ct1'):+.2f} | Y2 {100 * g('y2A'):.1f}%({g('y2nA')}) vs {100 * g('y2B'):.1f}%({g('y2nB')})"
         f" 날짜차 {100 * g('dd2'):+.1f}%p t={g('t2'):+.2f} 통제 t={g('ct2'):+.2f} | r60 {pct(g('r60A'))} vs {pct(g('r60B'))} ({g('nd')}일)")
    if f"{p}_ra_t" in r and not pd.isna(r.get(f"{p}_ra_t")):
        s += f"\n      ret/atr t={g('ra_t'):+.2f} | r60 날짜차 {pct(g('r60_dd'))} t={g('r60_t'):+.2f}"
        if not pd.isna(g("xs1_dd")):
            s += f" | 상위 1개 빼기 Y1: -{g('xs1')} {pct(g('xs1_dd'))} t={g('xs1_t'):+.2f}, -{g('xd1')} {pct(g('xd1_dd'))} t={g('xd1_t'):+.2f}"
        if not pd.isna(g("xs2_dd")):
            s += f" | 상위 1개 빼기 Y2: -{g('xs2')} {100 * g('xs2_dd'):+.1f}%p t={g('xs2_t'):+.2f}, -{g('xd2')} {100 * g('xd2_dd'):+.1f}%p t={g('xd2_t'):+.2f}"
    return s


def splits(d1, feats, kinds=("하위1/3", "상위1/3")):
    out = []
    for c in feats:
        x = d1[c].dropna()
        if len(x) < 20:
            continue
        q33, q50, q67 = x.quantile([1 / 3, 0.5, 2 / 3])
        if "하위1/3" in kinds:
            out.append((c, "하위1/3", f"{c}<={q33:.4g}", lambda z, c=c, v=q33: (z[c] <= v) & z[c].notna()))
        if "상위1/3" in kinds:
            out.append((c, "상위1/3", f"{c}>{q67:.4g}", lambda z, c=c, v=q67: (z[c] > v) & z[c].notna()))
        if "하위1/2" in kinds:
            out.append((c, "하위1/2", f"{c}<={q50:.4g}", lambda z, c=c, v=q50: (z[c] <= v) & z[c].notna()))
    return out


def run_pool(d, pool, feats, ctrl_keys, kinds=("하위1/3", "상위1/3"), full=True):
    d1, d2 = d[d.half == "앞"], d[d.half == "뒤"]
    rows, fns = [], {}
    for feat, nm, rule, fn in splits(d1, feats, kinds):
        k1, k2 = d1[feat].notna(), d2[feat].notna()
        e1, e2 = d1[k1], d2[k2]                       # 특성 값이 있는 거래끼리 비교
        a1, a2 = fn(e1), fn(e2)
        for y, p in (("ret", "1"), ("y2", "2")):
            r1 = stats(e1, a1, 1, ctrl_keys, full=False)
            sgn = 1 if (r1[f"dd{p}"] if not np.isnan(r1[f"dd{p}"]) else 0) >= 0 else -1
            ok = np.sign(r1[f"t{p}"]) == np.sign(r1[f"ct{p}"]) and not np.isnan(r1[f"t{p}"]) and not np.isnan(r1[f"ct{p}"])
            if p == "2" and r1["y2nA"] < 5:
                ok = False
            score = min(abs(r1[f"t{p}"]), abs(r1[f"ct{p}"])) if ok else 0.0
            R1 = stats(e1, a1, sgn, ctrl_keys, full, ys=(p,))
            R2 = stats(e2, a2, sgn, ctrl_keys, full, ys=(p,))
            rows.append(dict(pool=pool, feat=feat, split=nm, rule=rule, Y=f"Y{p}", sgn=sgn, score=score,
                             **{f"앞_{k}": v for k, v in R1.items()}, **{f"뒤_{k}": v for k, v in R2.items()}))
            fns[(pool, feat, nm)] = fn
    return pd.DataFrame(rows), fns


# ---------------------------------------------------------------- 무작위 특성 대조(고르기 우연 수준)
def null_max(d, nfeat, ctrl_keys, iters, seed=7):
    rng = np.random.default_rng(seed)
    d1 = d[d.half == "앞"].copy()
    out = []
    for _ in range(iters):
        best = 0.0
        for j in range(nfeat):
            z = pd.Series(rng.standard_normal(len(d1)), index=d1.index)
            q33, q67 = z.quantile([1 / 3, 2 / 3])
            for a in (z <= q33, z > q67):
                for y, p in (("ret", "1"), ("y2", "2")):
                    dd, tt, _ = cell_diff(d1, a, y, ["day"])
                    cd, ct, _ = cell_diff(d1, a, y, ["day"] + ctrl_keys)
                    if p == "2" and d1.y2[a].sum() < 5:
                        continue
                    if not (np.isnan(tt) or np.isnan(ct)) and np.sign(tt) == np.sign(ct):
                        best = max(best, min(abs(tt), abs(ct)))
        out.append(best)
    return np.array(out)


# ---------------------------------------------------------------- 앞 기간(5분봉) 보조
def early_check(cands):
    S = stop_lab.load("early")
    tr = stop_lab.run(S, None, "early")
    tr = tr[tr.ses == "정규"].copy()
    f9 = pd.read_csv(os.path.join(RES, "features.csv")).set_index("sym")
    rows = []
    for s, g in tr.groupby("sym"):
        A = S[s]
        o, h, l, c, r, ema = (A[k].astype(float) for k in ("o", "h", "l", "c", "r", "ema"))
        day, ses, mod = A["day"], A["ses"], A["mod"]
        dd = pd.DataFrame({"v": A["v"], "ses": ses, "date": day, "mod": mod}, index=A["idx"])
        rv = tod_rvol(dd, 3, 10)
        prev, tc = prev_close_map(day, ses, c)
        gp = 0.03 if s in build.LEV else 0.02
        for x in g.itertuples():
            f = feat_at(o, h, l, c, r, ema, day, ses, mod, rv, x.ei, x.ent, gp, prev, tc, W=(12, 6, 3))
            f.update(sym=s, day=x.day, ret=x.ret, y2=float(x.why == "RSI" and x.bars <= 12), mod=mod[x.ei],
                     r60=c[x.ei + 11] / x.ent - 1 if x.ei + 11 < len(c) else np.nan)
            rows.append(f)
    E = pd.DataFrame(rows)
    E["open30"] = E.tmin < 30
    E["atr14"] = E.sym.map(f9.atr14)
    q1, q2 = f9.atr14.quantile([1 / 3, 2 / 3])
    E["atrt"] = np.select([E.atr14 <= q1, E.atr14 <= q2, E.atr14 > q2], [0, 1, 2], -1)
    E["ra"] = E.ret / E.atr14
    say(f"\n== 앞 기간 보조(7/14~9/8 5분봉 EMA36·RSI9, 정규장 {len(E)}건 {E.day.nunique()}일, Y2 = 60분(12봉) 안 RSI 매도 {E.y2.mean() * 100:.1f}%) ==")
    say("창은 분 단위로 맞춤(20봉→12, 10→6, 5→3). 문턱은 이 기간 삼분위(지표 길이가 달라 3분봉 값 그대로 못 씀), 방향은 3분봉 앞 절반에서 정한 것.")
    for pool, feat, split, Y, sgn in cands:
        if pool != "정규":
            x = E[(E.spike >= SPIKE)]
        else:
            x = E
        x = x[x[feat].notna()]
        if split.startswith("하위"):
            v = x[feat].quantile(1 / 3 if split == "하위1/3" else 0.5)
            a = x[feat] <= v
        else:
            v = x[feat].quantile(2 / 3)
            a = x[feat] > v
        a = a & x[feat].notna()
        r = stats(x, a, sgn, ["open30", "atrt"], full=False)
        say(f"[{pool}] {feat} {split}(문턱 {v:.4g}) {Y} 방향 {'A가 좋음' if sgn > 0 else 'A가 나쁨'}: A {r['nA']}건 vs B {r['nB']}건 | "
            f"Y1 {pct(r['mA'])} vs {pct(r['mB'])} 날짜차 {pct(r['dd1'])} t={r['t1']:+.2f} 통제 t={r['ct1']:+.2f} | "
            f"Y2 {100 * r['y2A']:.1f}% vs {100 * r['y2B']:.1f}% t={r['t2']:+.2f} 통제 t={r['ct2']:+.2f} ({r['nd']}일)")
    return E


# ---------------------------------------------------------------- 10/9 사례
def case_1009(cands, thr):
    rows = []
    for s in build.SYMS:
        try:
            d = sym_bars(s)
        except FileNotFoundError:
            continue
        kst = d.index.tz_convert("Asia/Seoul")
        t8 = kst.normalize() + pd.Timedelta(hours=8)
        dead = np.where(kst < t8, t8.as_unit("ns").asi8, (t8 + pd.Timedelta(days=1)).as_unit("ns").asi8)
        hot = hot_days(RAW, s, d, 0.07)
        gp = 0.03 if s in build.LEV else 0.02
        # 마지막 봉까지 들고 있는 거래도 잡도록 sim을 직접 돈다(build.sim과 같은 규칙)
        o, l, r, ema, tk = d.o.values, d.l.values, d.r.values, d.ema.values, d.index.as_unit("ns").asi8
        c, h = d.c.values, d.h.values
        hold, out = False, []
        for i in range(61, len(d)):
            if hold:
                why = "RSI" if (i - 1 >= ei and r[i - 1] > 70) else ("8시" if tk[i] >= dl else None)
                if why:
                    out.append((ei, i, ent, o[i], why))
                    hold = False
            elif not hot[i]:
                lvl = ema[i - 1] * (1 - gp)
                if l[i] <= lvl:
                    ent, ei, hold, dl = min(o[i], lvl), i, True, dead[i]
        if hold:
            out.append((ei, None, ent, np.nan, "보유중"))
        day, ses, mod = np.asarray(d.date), d.ses.values, d["mod"].values
        rv5 = tod_rvol(d, 5, 10)
        prev, tc = prev_close_map(day, ses, c)
        for ei, xi, ent, px, why in out:
            if day[ei] != E2DAY:
                continue
            f = feat_at(d.o.values, h, l, c, r, ema, day, ses, mod, rv5, ei, ent, gp, prev, tc)
            f.update(sym=s, t=d.index[ei], ses=ses[ei], ent=ent, why=why, bars=(xi - ei) if xi is not None else len(d) - ei,
                     ret_exit=px / ent - 1 if xi is not None else np.nan, mtm=c[-1] / ent - 1,
                     low_after=l[ei:].min() / ent - 1, rsi_hit=why == "RSI")
            rows.append(f)
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(RES, "pat_shape_1009.csv"), index=False, encoding="utf-8-sig")
    reg = C[C.ses == "정규"]
    say(f"\n== E2 10/9 사례(10:15 봉까지, 정규장 매수 {len(reg)}건 / 전체 {len(C)}건) ==")
    cols = ["sym", "t", "ent", "why", "bars", "mtm", "low_after", "dd_day", "kday", "dd20", "spd20", "sl5", "ndown", "big1", "conc",
            "lw1", "o_line", "rsi1", "ema_sl", "vol5", "tmin", "gap", "spike", "retr", "d_prev"]
    pick = reg[reg.sym.isin(["CRCL", "MRVL", "LITE", "COHR"])]
    for _, x in pick.iterrows():
        say(f"{x.sym} {x.t.strftime('%H:%M')} 매수 {x.ent:.2f} → {x.why}({x.bars}봉), 10:15 평가 {pct(x.mtm)}, 매수 뒤 최저 {pct(x.low_after)}")
        say("   " + " ".join(f"{k}={x[k]:.4g}" if isinstance(x[k], (float, np.floating)) else f"{k}={x[k]}" for k in cols[7:]))
    say("후보 규칙에 넣으면(A면 규칙 조건 충족):")
    for (pool, feat, split, Y, sgn), v in zip(cands, thr):
        lab = []
        for _, x in pick.iterrows():
            val = x[feat]
            if pd.isna(val):
                lab.append(f"{x.sym} 값없음")
                continue
            inA = val <= v if split.startswith("하위") else val > v
            if pool != "정규" and not (x.spike >= SPIKE):
                lab.append(f"{x.sym} 대상아님(spike {pct(x.spike)})")
                continue
            lab.append(f"{x.sym} {val:.4g}{'(A)' if inA else '(B)'}")
        say(f"  [{pool}] {feat} {split} 문턱 {v:.4g} {Y} {'A가 좋음' if sgn > 0 else 'A가 나쁨'}: " + ", ".join(lab))
    # 그날 정규장 매수 전체의 단면(하루치, 설명용)
    if len(reg) >= 8:
        say(f"10/9 정규장 매수 {len(reg)}건 단면(하루 39~45분 창, 설명용): 특성 vs 10:15 평가 수익 스피어만")
        parts = []
        for f in FEATS:
            k = reg[f].notna()
            if k.sum() >= 8:
                parts.append(f"{f} {reg[f][k].corr(reg.mtm[k], method='spearman'):+.2f}({int(k.sum())})")
        say("   " + " | ".join(parts))
        say(f"   10:15까지 RSI 매도 {int(reg.rsi_hit.sum())}건: " + ", ".join(reg[reg.rsi_hit].sym))
    return C


def main():
    t = pd.read_csv(os.path.join(RES, "trades.csv"))
    t["ts"] = pd.to_datetime(t.t, utc=True).dt.tz_convert("America/New_York")
    t["day"] = pd.to_datetime(t.day).dt.date
    t["y2"] = ((t.why == "RSI") & (t.bars <= 20)).astype(float)
    f9 = pd.read_csv(os.path.join(RES, "features.csv")).set_index("sym")
    t["atr14"] = t.sym.map(f9.atr14)
    q1, q2 = f9.atr14.quantile([1 / 3, 2 / 3])
    t["atrt"] = np.select([t.atr14 <= q1, t.atr14 <= q2, t.atr14 > q2], [0, 1, 2], -1)
    t["ra"] = t.ret / t.atr14
    cache = os.path.join(RES, "pat_shape_feats.csv")
    T = trade_feats(t)
    T.drop(columns=["ts"]).to_csv(cache, index=False, encoding="utf-8-sig")
    T["open30"] = (T.ses == "정규") & (T.tmin < 30)
    T["tailfill"] = T.prv1m > 0.01

    say("# 매수 직전 3분봉 하락 모양 → 빠른 반등·회복 (pat_shape, 2026-10-10)")
    say("결과 변수(시험 전 고정): Y1 = 거래 수익 ret(지금 매도 규칙) | Y2 = 매수 뒤 20봉(60분) 안 RSI 매도 | 보조 r60 = 20번째 봉 종가 수익(고르기에 안 씀)")
    say(f"사건 연구: E1 = 정규장 매수 중 매수 전 그날 고점 >= 전날 정규장 종가 +{SPIKE:.0%}(테마 급등 뒤 되돌림), 결과 Y1·Y2 | "
        "E2 = 10/9 같은 규칙 재현, 10:15 봉까지 RSI 매도 여부·10:15 종가 평가(하루치, 설명용)")
    say(f"매수가 재현 오차 최대 {T.ent_chk.abs().max():.2e}, 특성 매칭 {T.dd20.notna().sum()}/{len(T)}건")
    reg = T[T.ses == "정규"].copy()
    say(f"정규장 {len(reg)}건(앞 {int((reg.half == '앞').sum())} / 뒤 {int((reg.half == '뒤').sum())}), 개장 30분 안 {reg.open30.mean() * 100:.0f}%, "
        f"Y2 {reg.y2.mean() * 100:.1f}%(앞 {int(reg[reg.half == '앞'].y2.sum())}건 / 뒤 {int(reg[reg.half == '뒤'].y2.sum())}건), "
        f"정규장 꼬리 체결 비중 {reg.tailfill.mean() * 100:.0f}%(정규장은 실거래량이라 빼지 않음)")
    ext = T[(T.ses != "정규") & (~T.tailfill) & T.prv1m.notna()].copy()
    say(f"프리·애프터 깨끗한 체결 {len(ext)}건(전체 {int((T.ses != '정규').sum())}건 중, 꼬리 {int(((T.ses != '정규') & T.tailfill).sum())}건 뺌, "
        f"닿은 1분봉 못 찾음 {int(((T.ses != '정규') & T.prv1m.isna()).sum())}건 뺌), Y2 {ext.y2.mean() * 100:.1f}%")
    e1 = reg[reg.spike >= SPIKE].copy()
    say(f"E1 급등 뒤 되돌림 정규장 매수 {len(e1)}건(앞 {int((e1.half == '앞').sum())} / 뒤 {int((e1.half == '뒤').sum())}), "
        f"Y1 평균 {pct(e1.ret.mean())} vs 나머지 정규 {pct(reg[~(reg.spike >= SPIKE)].ret.mean())}")
    say("t = 날짜 군집. 통제 = 같은 날·개장30분 여부·atr14 삼분위 칸 안 차이의 날짜 평균. 프리·애프터 통제 = 같은 날·세션·atr14 칸.")

    # 정규장 기본 사실: 시간대·변동성만으로 Y1·Y2가 갈리나
    say("\n== 통제 변수 자체 (정규장) ==")
    for h in ("앞", "뒤"):
        x = reg[reg.half == h]
        for nm, a in (("개장30분 안", x.open30), ("atr14 상위1/3", x.atrt == 2)):
            r = stats(x, a, 1, [], full=False)
            say(f"[{h}] {nm}: A {r['nA']} vs B {r['nB']} | Y1 {pct(r['mA'])} vs {pct(r['mB'])} 날짜차 t={r['t1']:+.2f} | "
                f"Y2 {100 * r['y2A']:.1f}% vs {100 * r['y2B']:.1f}% t={r['t2']:+.2f}")

    # 주 시험
    tabs = []
    tab_reg, fn_reg = run_pool(reg, "정규", FEATS, ["open30", "atrt"])
    tabs.append(tab_reg)
    tab_e1, fn_e1 = run_pool(e1, "E1급등되돌림", E1_FEATS, ["open30"], kinds=("하위1/2",))   # 표본이 작아 통제는 시간대만
    tabs.append(tab_e1)
    ext["open30"] = False
    tab_ext, _ = run_pool(ext, "프리애프터깨끗", EXT_FEATS, ["ses", "atrt"])
    tabs.append(tab_ext)
    tab = pd.concat(tabs, ignore_index=True)
    n_reg, n_e1, n_ext = len(tab_reg), len(tab_e1), len(tab_ext)
    say(f"\n시험 수: 정규장 주 시험 {n_reg}개({len(FEATS)}특성 x 하위1/3·상위1/3 x Y1·Y2) + E1 부분집단 {n_e1}개({len(E1_FEATS)}특성 x 하위1/2 x Y1·Y2)"
        f" + 프리·애프터 깨끗한 체결 보조 {n_ext}개({len(EXT_FEATS)}특성 x 2분할 x Y1·Y2) = {len(tab)}개. 고르기는 정규장 주 시험·E1에서만.")
    say(f"앞 |t|>=2(원 t) 개수: 정규 {int((tab_reg['앞_t1'].abs().ge(2) & (tab_reg.Y == 'Y1')).sum() + (tab_reg['앞_t2'].abs().ge(2) & (tab_reg.Y == 'Y2')).sum())}/{n_reg}"
        f" (우연 기대 약 {0.05 * n_reg:.0f}~{0.1 * n_reg:.0f}개, 날짜 10개라 t 꼬리가 두꺼움)")

    # 겹침 인용
    say("\n== 이미 반박된 것과 겹침(dip_reason/report.md, dip_fit) ==")
    say("- dd_day(그날 고점 대비) = 1절 d_high, sl10(30분 기울기) = 1절 '30분 하락 속도'(정규 t +0.81/+1.01), vol5 ≈ 1절 거래량 배수(10봉, 정규 t −0.49/−0.50),"
        " d_prev = 1절 전일 대비. 1절은 정규장 |t|<=1.5. 여기서는 문턱 분할·Y2로 다시 잼.")
    say("- tmin(개장 직후) = 3절: 개장 30분 매수가 나빴던 날 반반, 일관성 없음. 여기서는 통제 변수로 쓰고 특성으로도 잼.")
    say("- 10절 칼손절은 매수 '뒤' 경로(폭포·흘러내림)였고 AUC 0.74~0.80이었으나 손절 규칙은 모두 반박. 여기서는 매수 '전' 모양만 씀.")
    say("- 14절 지지 구역 버팀: 효과 없음. fam_context day_ret(그날 이미 빠진 폭)도 탈락.")

    # 고르기
    say("\n== 앞 절반 점수(원 t·통제 t 부호 같을 때 min|t|) 상위 10 ==")
    for pool, k in (("정규", 2), ("E1급등되돌림", 1)):
        x = tab[tab.pool == pool].sort_values("score", ascending=False)
        say(f"[{pool}] " + " | ".join(f"{r.feat} {r.split} {r.Y} {r.score:.2f}" for _, r in x.head(10).iterrows()))
    cands, thr = [], []
    picks = []
    for pool, k in (("정규", 2), ("E1급등되돌림", 1)):
        x = tab[(tab.pool == pool) & (tab.score > 0)].sort_values("score", ascending=False).drop_duplicates("feat").head(k)
        picks.append(x)
    picks = pd.concat(picks)

    # 무작위 특성 대조: 정규장 주 시험과 같은 수의 특성으로 고른 최고 점수의 분포
    nul = null_max(reg, len(FEATS), ["open30", "atrt"], 100)
    best_real = tab_reg.score.max()
    say(f"\n무작위 특성 대조(정규장, {len(FEATS)}개 무작위 특성 x 2분할 x Y1·Y2, 100회): 최고 점수 중앙값 {np.median(nul):.2f}, 90% {np.quantile(nul, .9):.2f}, "
        f"95% {np.quantile(nul, .95):.2f} | 실제 최고 {best_real:.2f} (p≈{(nul >= best_real).mean():.2f})")
    nul1 = null_max(e1, len(E1_FEATS), ["open30"], 100, seed=11) if len(e1) > 30 else np.array([np.nan])
    say(f"무작위 특성 대조(E1, {len(E1_FEATS)}개, 100회, 분할은 삼분위로 근사): 최고 점수 중앙값 {np.nanmedian(nul1):.2f}, 95% {np.nanquantile(nul1, .95):.2f} | "
        f"실제 E1 최고 {tab_e1.score.max():.2f}")

    say("\n== 고른 후보(정규장 2 + E1 1) — 뒤 절반 확인 ==")
    for _, p in picks.iterrows():
        g = tab[(tab.pool == p.pool) & (tab.feat == p.feat) & (tab.split == p.split) & (tab.Y == p.Y)]
        sgn_y = int(p.sgn)
        cands.append((p.pool, p.feat, p.split, p.Y, sgn_y))
        thr.append(float(p.rule.split("<=")[-1].split(">")[-1]))
        say(f"\n[{p.pool}] {p.rule} ({p.feat} {p.split}) — 고른 결과 {p.Y}, 방향 {'A가 좋음' if sgn_y > 0 else 'A가 나쁨'}, 앞 점수 {p.score:.2f}")
        for _, r in g.iterrows():
            say("  앞: " + fmt(r, "앞"))
            say("  뒤: " + fmt(r, "뒤"))
        # 시간대별로 쪼갬
        base = reg if p.pool == "정규" else e1
        fn = fn_reg[(p.pool, p.feat, p.split)] if p.pool == "정규" else fn_e1[(p.pool, p.feat, p.split)]
        for h in ("앞", "뒤"):
            for nm, kk in (("개장30분 안", base.open30), ("나머지", ~base.open30)):
                x = base[(base.half == h) & kk & base[p.feat].notna()]
                if len(x) < 10:
                    continue
                a = fn(x)
                r = stats(x, a, sgn_y, [], full=False)
                say(f"    [{h}·{nm}] A {r['nA']} vs B {r['nB']} | Y1 {pct(r['mA'])} vs {pct(r['mB'])} t={r['t1']:+.2f} | "
                    f"Y2 {100 * r['y2A']:.1f}% vs {100 * r['y2B']:.1f}% t={r['t2']:+.2f}")
        # 프리·애프터 깨끗한 체결(같은 문턱)
        if p.feat in EXT_FEATS and p.pool == "정규":
            for h in ("앞", "뒤"):
                x = ext[(ext.half == h) & ext[p.feat].notna()]
                a = fn(x)
                r = stats(x, a, sgn_y, ["ses", "atrt"], full=False)
                say(f"    [프리·애프터 깨끗, 같은 문턱, {h}] A {r['nA']} vs B {r['nB']} | Y1 {pct(r['mA'])} vs {pct(r['mB'])} t={r['t1']:+.2f} 통제 t={r['ct1']:+.2f} | "
                    f"Y2 {100 * r['y2A']:.1f}% vs {100 * r['y2B']:.1f}% t={r['t2']:+.2f}")

    early_check(cands)
    case_1009(cands, thr)

    # 모든 시험 행
    say("\n== 모든 시험(앞 | 뒤) ==")
    for _, r in tab.iterrows():
        say(f"[{r.pool}] {r.rule} {r.Y} 방향 {'A가 좋음' if r.sgn > 0 else 'A가 나쁨'} 점수 {r.score:.2f}")
        say("  앞: " + fmt(r, "앞").split("\n")[0])
        say("  뒤: " + fmt(r, "뒤").split("\n")[0])
    tab.to_csv(os.path.join(RES, "pat_shape_tests.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(RES, "pat_shape.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
