"""반박 검증(2026-10-09): volsqueeze 연구자의 best_rule.definition(DIV_40_p6_H0)만 보고 따로 만든 재구현.
연구자 코드(volsqueeze.py)는 가져오지 않는다. 기준봉 배열만 공용 시험대 stop_lab.load에서 받는다(o h l c r tk day mod).
구역·피벗·40분봉·RSI·에피소드·다이버전스·매도·무작위 기준은 모두 여기서 새로 계산한다.
사용: py support/verify_volsqueeze.py  (code 폴더에서) → ../results/support/verify_volsqueeze.txt
"""
import os, sys, time, bisect
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, PERIODS, RAW
from fetch import user_universe, universe

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "support", "verify_volsqueeze.txt"))
_fh = open(OUT, "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _fh.write(s + "\n"); _fh.flush()


U78 = set(user_universe()["합집합"])
UALL = U78 | set(universe())
MIN_NS = 60 * 10**9


# ---------------------------------------------------------------- 큰 봉·지표
def htf_bars(sym, tf):
    """5분봉 → tf분봉(미 동부 04:00 기준 격자). 반환 시작 ns, 고, 저, 종."""
    m = pd.read_parquet(os.path.join(RAW, f"5m_{sym}.parquet"))
    m = m[(m.High > 0) & m.Close.notna()]
    idx = m.index
    mod = idx.hour * 60 + idx.minute
    k = np.floor((mod.values - 240) / tf).astype(int)
    start_local = idx.normalize() + pd.to_timedelta(240 + k * tf, unit="min")
    key = start_local.as_unit("ns").asi8
    g = pd.DataFrame({"k": key, "h": m.High.values, "l": m.Low.values, "c": m.Close.values})
    agg = g.groupby("k", sort=True).agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    return agg.index.values.astype(np.int64), agg.h.values.astype(float), agg.l.values.astype(float), agg.c.values.astype(float)


def wilder_rsi(c, n):
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    d = np.diff(c)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = up[:n].mean(), dn[:n].mean()
    for t in range(n, len(c)):
        if t > n:
            ag = (ag * (n - 1) + up[t - 1]) / n
            al = (al * (n - 1) + dn[t - 1]) / n
        out[t] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def swing_points(h, l, N=5):
    """[(봉 j, 가격)] 저점 먼저. 저점: 왼쪽 N봉보다 엄격히 낮고 좌우 2N+1봉 최저."""
    n = len(l)
    s_l, s_h = pd.Series(l), pd.Series(h)
    left_min = s_l.shift(1).rolling(N).min().values      # l[j-N..j-1]
    right_min = s_l[::-1].shift(1).rolling(N).min().values[::-1]   # l[j+1..j+N]
    left_max = s_h.shift(1).rolling(N).max().values
    right_max = s_h[::-1].shift(1).rolling(N).max().values[::-1]
    out = []
    for j in range(N, n - N):
        if l[j] < left_min[j] and l[j] <= right_min[j]:
            out.append((j, l[j]))
        if h[j] > left_max[j] and h[j] >= right_max[j]:
            out.append((j, h[j]))
    return out


def make_zones(pts, keep=60, tol=0.005, gap=0.012):
    pts = pts[-keep:]
    grp = []
    for j, p in sorted(pts, key=lambda t: t[1]):
        if grp and (p - grp[-1][0]) / grp[-1][0] <= tol:
            g = grp[-1]
            g[1] = p; g[2] += 1; g[3] = max(g[3], j)
        else:
            grp.append([p, p, 1, j])
    kept = []
    for g in sorted(grp, key=lambda g: g[2] * 1e7 + g[3], reverse=True):
        m = (g[0] + g[1]) / 2
        if all(abs(m - km) / km >= gap for km in [(k[0] + k[1]) / 2 for k in kept]):
            kept.append(g)
    kept.sort(key=lambda g: g[1])
    return np.array([g[0] for g in kept]), np.array([g[1] for g in kept])


def zone_timeline(h, l, eff_of):
    """확정 시각마다 구역(lo 배열, hi 배열). eff_of(j) = 피벗 j를 쓸 수 있는 시각."""
    pts = swing_points(h, l)
    times, Z = [], []
    k = 0
    while k < len(pts):
        j = pts[k][0]
        while k < len(pts) and pts[k][0] == j:
            k += 1
        times.append(eff_of(j)); Z.append(make_zones(pts[:k]))
    return np.array(times, dtype=np.int64), Z


def find_episodes(o, l, c, tk, drank, zt, Z, band, brk=0.005, away=0.02, maxd=3):
    """[(s, E, top, lo, hi, 첫닿음 체결가)] E = 버팀으로 끝난 마지막 봉(없으면 s-1)."""
    zi = np.searchsorted(zt, tk, side="right") - 1
    n = len(o)
    res = []
    st = None
    cache = {}
    for i in range(1, n):
        if st is None:
            z = zi[i]
            if z < 0:
                continue
            if z not in cache:
                lo_, hi_ = Z[z]
                cache[z] = ((hi_ * (1 + band)).tolist(), lo_, hi_)
            tops, los, his = cache[z]
            k = bisect.bisect_left(tops, c[i - 1]) - 1          # top < 직전 종가 중 가장 높은 구역(구역은 hi 순 정렬)
            if k < 0:
                continue
            if l[i] > tops[k]:
                continue
            st = [i, tops[k], los[k], his[k], min(o[i], tops[k])]
        s, top, lo_, hi_, px = st
        if c[i] < lo_ * (1 - brk) or c[i] > hi_ * (1 + away) or drank[i] - drank[s] >= maxd:
            res.append((s, i - 1, top, lo_, hi_, px)); st = None
    if st is not None:
        s, top, lo_, hi_, px = st
        res.append((s, n - 1, top, lo_, hi_, px))
    return res


# ---------------------------------------------------------------- 매도
def exit_index(r, tk, n):
    """봉 i 시가에 샀을 때 매도 봉: RSI>70 봉 다음 봉 또는 첫 KST 08:00 이후 첫 봉. 08:00이 데이터 밖이면 n(제외)."""
    hit = np.where(np.nan_to_num(r) > 70, np.arange(n), n)
    nxt = np.minimum.accumulate(hit[::-1])[::-1]
    kst = pd.DatetimeIndex(tk, tz="UTC").tz_convert("Asia/Seoul")
    t8 = kst.normalize() + pd.Timedelta(hours=8)
    dead = np.where(kst < t8, t8.as_unit("ns").asi8, (t8 + pd.Timedelta(days=1)).as_unit("ns").asi8)
    j8 = np.searchsorted(tk, dead, side="left")
    x = np.minimum(nxt + 1, j8)
    return np.where(j8 >= n, n, x)


# ---------------------------------------------------------------- 종목별 준비
def prep_sym(A, sym, band_list=(0.003,)):
    o, h, l, c, r, tk = A["o"], A["h"], A["l"], A["c"], A["r"], A["tk"]
    n = len(o)
    days = np.array([d.toordinal() for d in A["day"]])
    _, drank = np.unique(days, return_inverse=True)
    st40, h40, l40, c40 = htf_bars(sym, 40)
    end40 = st40 + 40 * MIN_NS
    zt, Z = zone_timeline(h40, l40, lambda j: end40[j + 5])
    X = exit_index(r, tk, n)
    eps = {b: find_episodes(o, l, c, tk, drank, zt, Z, b) for b in band_list}
    return dict(n=n, days=days, X=X, eps=eps)


def div_signal(A, eps, sym, tf=40, w=2, p=6.0, rlen=14, hmin=0, strict_start=True, cache={}):
    """에피소드마다 첫 다이버전스 신호의 결정 봉 d(매수 = d+1 시가). 반환 [(에피소드 번호, d)]."""
    key = (sym, tf, w, rlen, A["mins"], len(A["tk"]))
    if key not in cache:
        st, hh, ll, cc = htf_bars(sym, tf)
        end = st + tf * MIN_NS
        rs = wilder_rsi(cc, rlen)
        m = len(ll)
        lows = []
        for j in range(w, m - w):
            if ll[j] <= ll[j - w:j + w + 1].min():
                lows.append(j)
        lows = np.array(lows, dtype=int)
        conf = end[lows + w]
        dcf = np.searchsorted(A["tk"], conf, side="left") - 1
        cache[key] = (lows, ll[lows], rs[lows], end[lows], dcf)
    lows, lv, rv, en, dcf = cache[key]
    tk, n = A["tk"], len(A["tk"])
    hb = max(1, int(np.ceil(hmin / A["mins"])))
    out = []
    for e, (s, E, top, lo_, hi_, px) in enumerate(eps):
        a = np.searchsorted(en, tk[s] if strict_start else tk[s - 1], side="right")   # 40분봉 끝 > 기준 시각
        best_l, best_r = None, None
        for q in range(a, len(lows)):
            d = dcf[q]
            if d > E:
                break
            if lv[q] > top or np.isnan(rv[q]):
                continue
            if best_l is not None and rv[q] - best_r >= p and d >= s + hb - 1 and d + 1 < n:
                out.append((e, d)); break
            if best_l is None or lv[q] < best_l:
                best_l, best_r = lv[q], rv[q]
    return out


# ---------------------------------------------------------------- 표 만들기
def trades_period(period, S, P, sig_fn=None, band=0.003, kind="div", H=0):
    lo_d, hi_d = PERIODS[period][3], PERIODS[period][4]
    rows = []
    for sym, A in S.items():
        if sym not in P:
            continue
        pp = P[sym]
        o, n, X = A["o"], pp["n"], pp["X"]
        mod = A["mod"]
        inper = np.array([lo_d <= d <= hi_d for d in A["day"]])
        eps = pp["eps"][band]
        if kind == "first":
            ei = np.array([e[0] for e in eps], dtype=int)
            px = np.array([e[5] for e in eps])
        elif kind == "hold":
            hb = max(1, int(np.ceil(H / A["mins"]))) if H > 0 else 1
            sel = [(e[0] + hb - 1) for e in eps if e[0] + hb - 1 <= e[1] and e[0] + hb < n]
            ei = np.array(sel, dtype=int) + 1
            px = o[ei] if len(ei) else np.zeros(0)
        else:
            sg = sig_fn(A, eps, sym)
            ei = np.array([d for _, d in sg], dtype=int) + 1
            px = o[ei] if len(ei) else np.zeros(0)
        if not len(ei):
            continue
        k = inper[ei]
        ei, px = ei[k], px[k]
        x = X[ei]
        ok = x < n
        ei, px, x = ei[ok], px[ok], x[ok]
        if not len(ei):
            continue
        rows.append(pd.DataFrame(dict(sym=sym, ei=ei, day=pp["days"][ei], b=mod[ei] // 30,
                                      reg=(mod[ei] >= 570) & (mod[ei] < 960), ret=o[x] / px - 1)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["sym", "ei", "day", "b", "reg", "ret"])


def rand_table(period, S, P):
    lo_d, hi_d = PERIODS[period][3], PERIODS[period][4]
    out = {}
    for sym, A in S.items():
        if sym not in P:
            continue
        o, X, n = A["o"], P[sym]["X"], P[sym]["n"]
        inper = np.array([lo_d <= d <= hi_d for d in A["day"]])
        ii = np.nonzero(inper & (np.arange(n) >= 1))[0]
        x = X[ii]
        ok = x < n
        ii, x = ii[ok], x[ok]
        r = o[x] / o[ii] - 1
        g = pd.Series(r).groupby(A["mod"][ii] // 30).mean()
        out[sym] = g
    return out


def add_rand(T, RT):
    if not len(T):
        T["q"] = []
        return T
    T["q"] = [RT[s].get(b, np.nan) for s, b in zip(T.sym, T.b)]
    return T


def tday(v, day):
    g = pd.Series(v).groupby(np.asarray(day)).sum()
    sd = g.std(ddof=1)
    return g.mean() / (sd / np.sqrt(len(g))) if len(g) > 1 and sd > 0 else np.nan


def st(T, univ, reg):
    x = T[T.sym.isin(U78 if univ == "u78" else UALL)]
    if reg:
        x = x[x.reg]
    if not len(x):
        return dict(n=0, avg=np.nan, win=np.nan, worst=np.nan, sum=0, t=np.nan, rand=np.nan, ex=np.nan, tex=np.nan)
    ex = x.ret - x.q
    return dict(n=len(x), avg=x.ret.mean() * 100, win=(x.ret > 0).mean() * 100, worst=x.ret.min() * 100,
                sum=x.ret.sum() * 100, t=tday(x.ret.values, x.day.values), rand=x.q.mean() * 100,
                ex=ex.mean() * 100, tex=tday(ex.fillna(0).values, x.day.values))


def f(d):
    return (f"n {d['n']:>5} 평균 {d['avg']:+.2f}% 승률 {d['win']:.1f}% 최악 {d['worst']:+.1f}% 합 {d['sum']:+.2f}%p "
            f"t {d['t']:+.2f} 무작위 {d['rand']:+.2f}% 초과 {d['ex']:+.2f}%(t {d['tex']:+.2f})") if d["n"] else "n 0"


# ---------------------------------------------------------------- 일봉(버틴 날수 단조성 재확인)
def daily_hold_table():
    SPLIT = date(2026, 4, 7)
    rows = []
    for sym in sorted(UALL):
        p = os.path.join(RAW, f"1d_{sym}.parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p).dropna(subset=["Open", "High", "Low", "Close"])
        d = d[d.High > 0]
        o, h, l, c = (d[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
        n = len(o)
        if n < 60:
            continue
        days = np.array(pd.to_datetime(d.index).date)
        idx = np.arange(n, dtype=np.int64)
        zt, Z = zone_timeline(h, l, lambda j: j + 6)
        eps = find_episodes(o, l, c, idx, idx, zt, Z, 0.01, 0.015, 0.05, 20)
        half = (days >= SPLIT).astype(int)
        r5 = np.full(n, np.nan); r5[:n - 5] = o[5:] / o[:n - 5] - 1
        rnd = [np.nanmean(r5[(half == hh) & (idx >= 30)]) for hh in (0, 1)]
        for (s, E, top, lo_, hi_, px) in eps:
            cand = [("A_first", s, px)]
            for H in (0, 1, 2, 3, 5, 10):
                dd = s + H
                if dd <= E and dd + 1 < n:
                    cand.append((f"T_{H}", dd + 1, o[dd + 1]))
            for name, ei, pr in cand:
                if ei + 5 < n:
                    rows.append((name, sym, days[ei], half[ei], o[ei + 5] / pr - 1, rnd[half[ei]]))
    D = pd.DataFrame(rows, columns=["rule", "sym", "day", "half", "ret", "q"])
    return D


# ---------------------------------------------------------------- 실행
if __name__ == "__main__":
    t0 = time.time()
    say("volsqueeze 반박 검증 — 독립 재구현(best_rule.definition만 보고), 2026-10-09")
    BANDS = (0.002, 0.003, 0.005)
    R = {}
    for period in ("early", "late"):
        S = load(period)
        P = {}
        for sym, A in S.items():
            try:
                P[sym] = prep_sym(A, sym, BANDS)
            except FileNotFoundError:
                pass
        RT = rand_table(period, S, P)
        R[period] = (S, P, RT)
        say(f"[{period}] 종목 {len(P)} · 준비 {time.time() - t0:.0f}초 · 에피소드(띠 0.3%) {sum(len(P[s]['eps'][0.003]) for s in P)}")

    def run(kind="div", band=0.003, H=0, **kw):
        out = {}
        for period in ("early", "late"):
            S, P, RT = R[period]
            fn = (lambda A, eps, sym: div_signal(A, eps, sym, **kw)) if kind == "div" else None
            out[period] = add_rand(trades_period(period, S, P, fn, band, kind, H), RT)
        return out

    # 1. 재현
    say("\n== 1. 재현: DIV_40_p6_H0 (주 매도 RSI70/8시) vs 연구자 보고 ==")
    REP = {("early", "u78", True): (48, 0.18), ("early", "u78", False): (255, 0.07), ("early", "all", True): (73, 0.20),
           ("early", "all", False): (410, 0.06), ("late", "u78", True): (36, 0.16), ("late", "u78", False): (126, 0.31),
           ("late", "all", True): (65, 0.28), ("late", "all", False): (205, 0.32)}
    best = run("div", p=6.0)
    best_loose = run("div", p=6.0, strict_start=False)
    first = run("first")
    for period in ("early", "late"):
        for un in ("u78", "all"):
            for reg in (True, False):
                a, b, c_ = st(best[period], un, reg), st(best_loose[period], un, reg), st(first[period], un, reg)
                rn, ra = REP[(period, un, reg)]
                say(f"-- {period} {un} {'정규' if reg else '24h'}  보고 n {rn} 평균 {ra:+.2f}%")
                say(f"   재구현(시작 뒤 끝난 40분봉)  {f(a)}")
                say(f"   재구현(연구자식 느슨 시작)    {f(b)}")
                say(f"   첫 닿음                      {f(c_)}")

    # 2. 이웃
    say("\n== 2. 이웃 매개변수(사용자78 정규장 / 전체137 정규장): early | late 평균, 무작위 초과, 첫닿음 대비 ==")
    fa = {p: (st(first[p], "u78", True), st(first[p], "all", True)) for p in ("early", "late")}
    grid = [dict(p=x) for x in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)] + \
           [dict(p=6.0, w=x) for x in (1, 3)] + [dict(p=6.0, rlen=x) for x in (10, 20)] + \
           [dict(p=6.0, tf=x) for x in (30, 60)] + [dict(p=6.0, hmin=x) for x in (60, 120, 240)] + \
           [dict(p=6.0, band=x) for x in (0.002, 0.005)]
    NB = []
    for g in grid:
        g = dict(g)
        band = g.pop("band", 0.003)
        res = run("div", band=band, **g)
        lab = ",".join(f"{k}={v}" for k, v in g.items()) + (f",band={band}" if band != 0.003 else "")
        line = []
        for un_i, un in enumerate(("u78", "all")):
            for p in ("early", "late"):
                s_ = st(res[p], un, True)
                fb = fa[p][un_i]
                line.append(f"{s_['n']:>4} {s_['avg']:+.2f} ex{s_['ex']:+.2f} vsA{s_['avg'] - fb['avg']:+.2f}")
                NB.append(dict(lab=lab, un=un, per=p, n=s_["n"], avg=s_["avg"], ex=s_["ex"], vsA=s_["avg"] - fb["avg"]))
        say(f"   {lab:<22} u78: {line[0]} | {line[1]}    all: {line[2]} | {line[3]}")
    NBd = pd.DataFrame(NB)
    for un in ("u78", "all"):
        x = NBd[(NBd.un == un)]
        e, l_ = x[x.per == "early"].set_index("lab"), x[x.per == "late"].set_index("lab")
        both = ((e.ex > 0) & (e.vsA > 0) & (l_.ex > 0) & (l_.vsA > 0))
        say(f"   {un}: 이웃 {len(e)}개 중 late 초과>0 {int((l_.ex > 0).sum())}, late 첫닿음 이김 {int((l_.vsA > 0).sum())}, 두 기간 둘 다 이김 {int(both.sum())}")

    # 3. 쏠림
    say("\n== 3. 쏠림: 최선안 수익 합 중 상위 종목·날짜 몫 ==")
    for period in ("early", "late"):
        for un in ("u78", "all"):
            T = best[period]
            x = T[T.sym.isin(U78 if un == "u78" else UALL) & T.reg]
            if not len(x):
                continue
            tot = x.ret.sum() * 100
            bs = (x.groupby("sym").ret.sum() * 100).sort_values(ascending=False)
            bd = (x.groupby("day").ret.sum() * 100).sort_values(ascending=False)
            wo_s = x[x.sym != bs.index[0]]
            wo_d = x[x.day != bd.index[0]]
            say(f"-- {period} {un} 정규: 합 {tot:+.2f}%p · 종목 {x.sym.nunique()} · 날짜 {x.day.nunique()}")
            say(f"   상위 종목 {bs.index[0]} {bs.iloc[0]:+.2f}%p, 상위3 {', '.join(f'{k} {v:+.2f}' for k, v in bs.head(3).items())}")
            say(f"   상위 날짜 {date.fromordinal(int(bd.index[0]))} {bd.iloc[0]:+.2f}%p, 상위3 {', '.join(f'{date.fromordinal(int(k))} {v:+.2f}' for k, v in bd.head(3).items())}")
            say(f"   상위 종목 빼면 평균 {wo_s.ret.mean() * 100:+.2f}% (n {len(wo_s)}), 상위 날짜 빼면 평균 {wo_d.ret.mean() * 100:+.2f}% (n {len(wo_d)})"
                f" · 초과(무작위) 빼기 전 {(x.ret - x.q).mean() * 100:+.2f}% → 상위 종목 뺀 뒤 {(wo_s.ret - wo_s.q).mean() * 100:+.2f}%")
            say(f"   중앙값 {x.ret.median() * 100:+.2f}%, 최고 {x.ret.max() * 100:+.2f}%, 최고 1건 빼면 평균 {x.ret.drop(x.ret.idxmax()).mean() * 100:+.2f}%")

    # 4. 버틴 시간 단조성(분봉)
    say("\n== 4. 버틴 시간 단조성 재현(H분 버틴 순간 다음 봉 시가 매수, 주 매도) ==")
    HT = {H: run("hold", H=H) for H in (0, 15, 30, 60, 120, 240, 480)}
    for period in ("early", "late"):
        for un in ("u78", "all"):
            for reg in (True, False):
                a = st(first[period], un, reg)
                cells = [f"닿음 {a['avg']:+.2f}({a['ex']:+.2f},n{a['n']})"]
                for H, T in HT.items():
                    s_ = st(T[period], un, reg)
                    cells.append(f"{H}분 {s_['avg']:+.2f}({s_['ex']:+.2f},n{s_['n']})")
                say(f"   {period} {un} {'정규' if reg else '24h'}: " + " / ".join(cells))

    # 5. 일봉 버틴 날수
    say("\n== 5. 일봉 버틴 날수 단조성 재현(5거래일 매도, 평균(무작위 초과)) 앞 | 뒤 ==")
    D = daily_hold_table()
    for un in ("u78", "all"):
        x = D[D.sym.isin(U78 if un == "u78" else UALL)]
        for rule in ["A_first"] + [f"T_{H}" for H in (0, 1, 2, 3, 5, 10)]:
            y = x[x.rule == rule]
            cs = []
            for hh in (0, 1):
                z = y[y.half == hh]
                cs.append(f"n {len(z):>4} {z.ret.mean() * 100:+.2f}% ({(z.ret - z.q).mean() * 100:+.2f})")
            say(f"   {un} {rule:<8} " + " | ".join(cs))
    say(f"\n걸린 시간 {time.time() - t0:.0f}초")
