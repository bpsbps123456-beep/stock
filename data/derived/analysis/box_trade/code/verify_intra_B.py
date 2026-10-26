"""intra_B 최선안 반박 검증 [키 verify_intra_B] 2026-10-09
best_rule.definition 글만 보고 원자료(raw 1m·5m·1d parquet)에서 따로 짠 구현. intra_B.py·stop_lab 배열은 쓰지 않는다.

정의(검증 대상): 지지선 L = 그날 이전 5거래일 정규장 저가 최솟값(그날 안에서는 고정). 지정가 L×0.98.
정규장(9:30~16:00 ET) 봉 저가 < 지정가면 min(봉 시가, 지정가) 체결, 종목당 하루 한 번.
매도 RSI8: RSI(Pine ta.rsi, 3분봉 14 / 5분봉 9, 프리·애프터 봉 포함)가 봉 k에서 > 70이면 봉 k+1 시가,
          아니면 매수 뒤 첫 한국 08:00 이후 첫 봉 시가(둘 중 먼저). 매수 봉 자신의 RSI도 판정에 넣는다.
매도 TP2W: 매수 봉 시각 < 11:00 ET면 +2% 지정가(매수 다음 정규장 봉 고가 ≥ 목표면 max(시가, 목표)), 못 닿으면 08시 봉 시가.
          11:00 뒤 체결이면 08시 봉 시가.
봉: 앞(고르기) 7/14~9/8 = 5분봉(raw 5m), 뒤(확인) 9/10~10/6 = 3분봉(raw 1m을 3분으로 묶음).
L 원천 두 가지: (a) 5분봉 정규장 저가로 만든 일봉(연구자와 같은 방식) (b) yfinance 1d 일봉 Low(Pine request.security "D"와 같은 방식).
비교: 무작위1(같은 종목·같은 30분 칸·기간 전체 정규장 봉 시가 매수, 같은 매도), 무작위3(같은 종목·같은 날·같은 칸),
      무작위2(같은 종목·같은 날 모든 정규장 봉), QQQ 같은 구간(시장 헤지).
사용: py verify_intra_B.py → ../results/verify_intra_B.txt, ../results/verify_intra_B_trades.csv, ../results/verify_intra_B_neighbors.csv
"""
import os, sys, time
from datetime import date
import numpy as np, pandas as pd
from numba import njit

ROOT = r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis"
SCR = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad"
RAW = SCR + "/dip/raw"
DLY = SCR + "/box/daily"
OUTD = os.path.join(ROOT, "box_trade", "results")
OUT = os.path.join(OUTD, "verify_intra_B.txt")
sys.path.insert(0, os.path.join(ROOT, "dip_reason", "code"))
from fetch import user_universe, universe  # noqa: E402

U78 = set(user_universe()["합집합"])
U137 = U78 | set(universe())
PER = {"early": (5, 9, date(2026, 7, 14), date(2026, 9, 8)), "late": (3, 14, date(2026, 9, 10), date(2026, 10, 7))}
PL = {"early": "고르기 7/14~9/8 5분봉", "late": "확인 9/10~10/6 3분봉"}
_fh = None


def say(*a):
    global _fh
    if _fh is None:
        _fh = open(OUT, "w", encoding="utf-8")
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _fh.write(s + "\n")
    _fh.flush()


def pct(v, k=2):
    return "n/a" if v is None or not np.isfinite(v) else f"{v * 100:+.{k}f}%"


# ------------------------------------------------------------ 지표(Pine ta.rsi = Wilder RMA, 첫 값은 SMA)
def rsi_pine(c, n):
    c = np.asarray(c, float)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    ch = np.diff(c)
    up, dn = np.maximum(ch, 0), np.maximum(-ch, 0)
    au, ad = up[:n].mean(), dn[:n].mean()
    a = 1.0 / n
    for i in range(n, len(c)):
        if i > n:
            au = a * up[i - 1] + (1 - a) * au
            ad = a * dn[i - 1] + (1 - a) * ad
        out[i] = 100.0 if ad == 0 else (0.0 if au == 0 else 100 - 100 / (1 + au / ad))
    return out


def get_bars(sym, mins):
    if mins == 5:
        b = pd.read_parquet(f"{RAW}/5m_{sym}.parquet")
    else:
        m = pd.read_parquet(f"{RAW}/1m_{sym}.parquet")
        b = m.resample("3min", label="left", closed="left").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
    b = b[b.High > 0]
    return b


def daily_from_5m(sym):
    b = pd.read_parquet(f"{RAW}/5m_{sym}.parquet")
    b = b[b.High > 0]
    mod = b.index.hour * 60 + b.index.minute
    b = b[(mod >= 570) & (mod < 960)]
    g = b.groupby(b.index.date).agg(Low=("Low", "min"), Close=("Close", "last"))
    return g


def daily_from_1d(sym):
    d = pd.read_parquet(f"{DLY}/1d_{sym}.parquet")
    d.index = pd.to_datetime(d.index).date
    return d[["Low", "Close"]]


def l5_map(dly, ref_close=None):
    """날짜 → 그날 이전 5거래일 Low 최솟값. ref_close(5분봉 일봉 종가)와 5% 넘게 다르면(분할·자료 불일치) 그 창은 버림."""
    days = list(dly.index)
    lo = dly.Low.values
    cl = dly.Close.values
    out = {}
    bad = np.zeros(len(days), bool)
    r = np.r_[np.nan, cl[1:] / cl[:-1] - 1]
    bad |= (r <= -0.40) | (r >= 0.80)
    if ref_close is not None:
        rc = ref_close.reindex(days).values
        with np.errstate(invalid="ignore", divide="ignore"):
            bad |= np.isfinite(rc) & (np.abs(cl / rc - 1) > 0.05)
    for i in range(5, len(days)):
        if bad[i - 5:i + 1].any():
            out[days[i]] = np.nan
        else:
            out[days[i]] = lo[i - 5:i].min()
    return out


# ------------------------------------------------------------ 매도
@njit(cache=False)
def exits(o, h, r, reg, x8, e_arr, fill, tp_on):
    """반환 RSI8, RSI8R, 8시, TP2W 매도가. x8 = 각 봉의 08시 정리 봉 위치(-1 = 자료 없음)."""
    n = len(o)
    m = len(e_arr)
    out = np.full((m, 4), np.nan)
    for q in range(m):
        e = e_arr[q]
        d = x8[e]
        if d < 0:
            continue
        # RSI8: 봉 k(k ≥ e)가 RSI > 70이면 k+1 시가, 08시보다 먼저일 때
        x = d
        k = e
        while k + 1 < d:
            if r[k] > 70.0:
                x = k + 1
                break
            k += 1
        out[q, 0] = o[x]
        x = d
        k = e
        while k + 1 < d:
            if r[k] > 70.0 and reg[k] and reg[k + 1]:
                x = k + 1
                break
            k += 1
        out[q, 1] = o[x]
        out[q, 2] = o[d]
        px = o[d]
        if tp_on[q]:
            tg = fill[q] * 1.02
            k = e + 1
            while k < d:
                if reg[k] and h[k] >= tg:
                    px = max(o[k], tg)
                    break
                k += 1
        out[q, 3] = px
    return out


EXN = ("RSI8", "RSI8R", "8시", "TP2W")


# ------------------------------------------------------------ 종목 준비
def prep_sym(sym, period, dsrc):
    mins, rl, lo, hi = PER[period]
    b = get_bars(sym, mins)
    idx = b.index
    o, h, l, c = (b[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    r = rsi_pine(c, rl)
    mod = (idx.hour * 60 + idx.minute).values
    reg = (mod >= 570) & (mod < 960)
    day = np.array(idx.date)
    # 다음 한국 08:00 (봉 시작 시각 기준, 같은 시각이면 다음 날)
    kst = idx.tz_convert("Asia/Seoul")
    t8 = kst.normalize() + pd.Timedelta(hours=8)
    nxt = t8.where(kst < t8, t8 + pd.Timedelta(days=1))
    tns = idx.as_unit("ns").asi8
    x8 = np.searchsorted(tns, nxt.as_unit("ns").asi8, side="left")
    x8 = np.where(x8 < len(o), x8, -1).astype(np.int64)
    inper = (day >= lo) & (day <= hi)
    L = np.array([dsrc.get(d, np.nan) for d in day])
    return dict(sym=sym, o=o, h=h, l=l, c=c, r=r, mod=mod, reg=reg, day=day, x8=x8, inper=inper, L=L, tns=tns, idx=idx)


def entries(Q, nd_key="L", depth=0.02, thr=0.0):
    L = Q[nd_key]
    lim = L * (1 - depth)
    ok = Q["reg"] & Q["inper"] & np.isfinite(lim) & (Q["l"] < lim * (1 - thr))
    i = np.nonzero(ok)[0]
    if not len(i):
        return i, np.zeros(0)
    d = Q["day"][i]
    keep = np.r_[True, d[1:] != d[:-1]]
    i = i[keep]
    return i, np.minimum(Q["o"][i], lim[i])


def trade_rows(Q, i, fill):
    if not len(i):
        return None
    tp_on = Q["mod"][i] < 660
    px = exits(Q["o"], Q["h"], Q["r"], Q["reg"], Q["x8"], i.astype(np.int64), fill, tp_on)
    df = pd.DataFrame({"sym": Q["sym"], "day": Q["day"][i], "e": i, "mod": Q["mod"][i], "fill": fill,
                       "lo_bar": Q["l"][i], "o_bar": Q["o"][i], "c_bar": Q["c"][i]})
    for k, ex in enumerate(EXN):
        df[ex] = px[:, k] / fill - 1
    # RSI8 매도 시각(QQQ 헤지용)
    return df


def random_base(Q):
    """모든 기간 안 정규장 봉 시가 매수 → 같은 매도. (sym, 칸) 평균, (sym, 날, 칸) 평균, (sym, 날) 평균."""
    i = np.nonzero(Q["reg"] & Q["inper"])[0]
    if not len(i):
        return None
    fill = Q["o"][i]
    px = exits(Q["o"], Q["h"], Q["r"], Q["reg"], Q["x8"], i.astype(np.int64), fill, Q["mod"][i] < 660)
    df = pd.DataFrame({"day": Q["day"][i], "bk": Q["mod"][i] // 30})
    for k, ex in enumerate(EXN):
        df[ex] = px[:, k] / fill - 1
    return dict(r1=df.groupby("bk")[list(EXN)].mean(), r3=df.groupby(["day", "bk"])[list(EXN)].mean(),
                r2=df.groupby("day")[list(EXN)].mean())


def attach_random(df, RB):
    out = []
    for s, g in df.groupby("sym"):
        rb = RB.get(s)
        g = g.copy()
        bk = g["mod"].values // 30
        for ex in EXN:
            if rb is None:
                g["r1_" + ex] = np.nan; g["r3_" + ex] = np.nan; g["r2_" + ex] = np.nan
                continue
            g["r1_" + ex] = rb["r1"][ex].reindex(bk).values
            g["r3_" + ex] = rb["r3"][ex].reindex(pd.MultiIndex.from_arrays([g.day.values, bk])).values
            g["r2_" + ex] = rb["r2"][ex].reindex(g.day.values).values
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else df


def dct(day, x):
    s = pd.Series(x).groupby(np.asarray(day)).sum()
    if len(s) < 3 or s.std(ddof=1) == 0:
        return np.nan
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def stat(t, syms, ex="RSI8", rnd="r1_"):
    t = t[t.sym.isin(syms)]
    t = t[np.isfinite(t[ex])]
    if not len(t):
        return dict(n=0, avg=np.nan, win=np.nan, worst=np.nan, sum=0.0, t=np.nan, exc=np.nan, te=np.nan, days=0, syms=0)
    x = t[ex].values
    r = t[rnd + ex].values
    e = x - r
    k = np.isfinite(e)
    return dict(n=len(t), avg=x.mean(), win=(x > 0).mean(), worst=x.min(), sum=x.sum(), t=dct(t.day.values, x),
                exc=e[k].mean() if k.any() else np.nan, te=dct(t.day.values[k], e[k]) if k.sum() > 2 else np.nan,
                days=t.day.nunique(), syms=t.sym.nunique())


def line(s):
    if s["n"] == 0:
        return "n0"
    return (f"n{s['n']:>4} 평균 {pct(s['avg'])} 승률 {s['win']:.0%} 최악 {pct(s['worst'], 1)} 합 {s['sum']:+.2f} t {s['t']:+.2f}"
            f" 초과 {pct(s['exc'])} te {s['te']:+.2f} ({s['days']}일 {s['syms']}종목)")


def main():
    t0 = time.time()
    say("intra_B 최선안 반박 검증 [verify_intra_B] 2026-10-09")
    say(__doc__.split("사용:")[0].strip())
    say("")
    syms = sorted(U137)
    have5 = [s for s in syms if os.path.exists(f"{RAW}/5m_{s}.parquet")]
    have1 = [s for s in syms if os.path.exists(f"{RAW}/1m_{s}.parquet")]
    say(f"종목: 전체 {len(syms)} (5분봉 있음 {len(have5)}, 1분봉 있음 {len(have1)}), 사용자78 {len(U78)}")
    # 일봉 원천
    DA, DB = {}, {}
    for s in have5:
        ga = daily_from_5m(s)
        DA[s] = l5_map(ga)
        try:
            gb = daily_from_1d(s)
            DB[s] = {k: v for k, v in l5_map(gb, ref_close=ga.Close).items()}
        except FileNotFoundError:
            DB[s] = {}
    Qs = {}
    for period in PER:
        lst = have5 if period == "early" else [s for s in have1 if s in have5]
        for s in lst:
            Q = prep_sym(s, period, DA[s])
            Q["LB"] = np.array([DB[s].get(d, np.nan) for d in Q["day"]])
            Qs[(period, s)] = Q
    say(f"준비 {time.time() - t0:.0f}s")
    RB = {}
    for (p, s), Q in Qs.items():
        RB[(p, s)] = random_base(Q)
    say(f"무작위 기준 {time.time() - t0:.0f}s")

    def run(period, key="L", depth=0.02, thr=0.0, Lmap=None):
        rows = []
        for (p, s), Q in Qs.items():
            if p != period:
                continue
            if Lmap is not None:
                Q = dict(Q)
                Q["LX"] = np.array([Lmap[s].get(d, np.nan) for d in Q["day"]])
                key_ = "LX"
            else:
                key_ = key
            i, f = entries(Q, key_, depth, thr)
            df = trade_rows(Q, i, f)
            if df is not None:
                rows.append(df)
        if not rows:
            return pd.DataFrame()
        T = pd.concat(rows, ignore_index=True)
        return attach_random(T, {s: RB[(period, s)] for (p, s) in Qs if p == period})

    # 1) 재현
    say("== 1) 재현: L5D LIM2%(정규장), 매도 RSI8·TP2W. 초과 = 무작위1(같은 종목·같은 30분 칸) 대비, te = 날짜 군집 t")
    say("   연구자 보고: 앞78 RSI8 n312 +0.73% 초과 +0.71% te 3.61 / 뒤78 n79 +0.87% 초과 +0.75% te 2.40 / 앞137 n603 +0.64% / 뒤137 n162 +0.49% 초과 +0.34% te 1.32")
    say("               TP2W 앞78 +0.38% / 뒤78 +0.78% / 앞137 +0.36% / 뒤137 +0.50%")
    TR = {}
    for src, key in (("(a) 5분봉 정규장 일봉", "L"), ("(b) 1d 일봉 Low", "LB")):
        for p in PER:
            T = run(p, key)
            TR[(src, p)] = T
            for un, us in (("78", U78), ("137", U137)):
                for ex in ("RSI8", "TP2W", "RSI8R", "8시"):
                    s1 = stat(T, us, ex)
                    s3 = stat(T, us, ex, "r3_")
                    s2 = stat(T, us, ex, "r2_")
                    say(f"  {src} {PL[p]} {un:>3} {ex:<5} {line(s1)} | 무작위3 초과 {pct(s3['exc'])} te {s3['te']:+.2f} | 무작위2 초과 {pct(s2['exc'])} te {s2['te']:+.2f}")
    say("")
    TA = {p: TR[("(a) 5분봉 정규장 일봉", p)] for p in PER}
    TB = {p: TR[("(b) 1d 일봉 Low", p)] for p in PER}
    pd.concat([TA[p].assign(period=p, src="a") for p in PER] + [TB[p].assign(period=p, src="b") for p in PER]).to_csv(
        os.path.join(OUTD, "verify_intra_B_trades.csv"), index=False, encoding="utf-8-sig")

    # 2) QQQ 헤지(같은 구간)
    say("== 2) 시장 대비: 같은 매수 봉 시가 → RSI8 매도 봉 시가 구간의 QQQ 수익을 뺀 값((a) 원천)")
    for p in PER:
        mins = PER[p][0]
        qb = get_bars("QQQ", mins)
        qt = qb.index.as_unit("ns").asi8
        qo = qb.Open.values
        T = TA[p]
        hed = []
        for s, g in T.groupby("sym"):
            Q = Qs[(p, s)]
            for _, rw in g.iterrows():
                e = int(rw.e)
                # RSI8 매도 봉 다시 찾기
                d = Q["x8"][e]
                x = d
                for k in range(e, d - 1):
                    if Q["r"][k] > 70:
                        x = k + 1
                        break
                a = np.searchsorted(qt, Q["tns"][e]); b = np.searchsorted(qt, Q["tns"][x])
                if a < len(qo) and b < len(qo):
                    hed.append(qo[b] / qo[a] - 1)
                else:
                    hed.append(np.nan)
        T = T.copy()
        T["qqq"] = np.nan
        T.loc[T.sort_values("sym", kind="stable").index, "qqq"] = hed
        TA[p] = T
        for un, us in (("78", U78), ("137", U137)):
            x = T[T.sym.isin(us)]
            h_ = (x.RSI8 - x.qqq).values
            k = np.isfinite(h_)
            say(f"  {PL[p]} {un:>3}: n{k.sum()} QQQ 같은 구간 평균 {pct(np.nanmean(x.qqq))} → 헤지 뒤 평균 {pct(h_[k].mean())} te {dct(x.day.values[k], h_[k]):+.2f}")
    say("")

    # 3) 이웃 매개변수
    say("== 3) 이웃 매개변수((a) 원천): 박스 길이 N일 × 이탈 깊이. 칸 = 평균/초과1(te). 부호 바뀌면 ✗")
    rows = []
    for nd in (3, 4, 5, 7, 10):
        Lmap = {}
        for s in have5:
            ga = daily_from_5m(s)
            days = list(ga.index); lo_ = ga.Low.values
            Lmap[s] = {days[i]: lo_[i - nd:i].min() for i in range(nd, len(days))}
        for dep in (0.01, 0.015, 0.02, 0.025, 0.03):
            cells = []
            for p in PER:
                T = run(p, depth=dep, Lmap=Lmap)
                for un, us in (("78", U78), ("137", U137)):
                    for ex in ("RSI8", "TP2W"):
                        st = stat(T, us, ex)
                        rows.append(dict(nd=nd, depth=dep, period=p, univ=un, exit=ex, **st))
                        if ex == "RSI8":
                            cells.append(f"{p[0]}{un} n{st['n']} {pct(st['avg'])}/{pct(st['exc'])}({st['te']:+.1f}){'' if st['n'] and st['exc'] > 0 and st['avg'] > 0 else '✗'}")
            say(f"  L{nd}D -{dep * 100:.1f}%: " + " | ".join(cells))
    NB = pd.DataFrame(rows)
    NB.to_csv(os.path.join(OUTD, "verify_intra_B_neighbors.csv"), index=False, encoding="utf-8-sig")
    for ex in ("RSI8", "TP2W"):
        for un in ("78", "137"):
            g = NB[(NB.exit == ex) & (NB.univ == un)]
            piv = g.pivot_table(index=["nd", "depth"], columns="period", values="exc")
            pa = g.pivot_table(index=["nd", "depth"], columns="period", values="avg")
            both = ((piv.early > 0) & (piv.late > 0)).sum()
            bothavg = ((pa.early > 0) & (pa.late > 0)).sum()
            nn = ((piv.index.get_level_values(0).isin([4, 5, 7])) & (piv.index.get_level_values(1).isin([0.015, 0.02, 0.025])))
            say(f"  {ex} {un}: 25칸 중 두 기간 초과>0 {both}, 평균>0 {bothavg} | 바로 이웃 9칸(L4~7D × 1.5~2.5%) 초과>0 {int(((piv.early > 0) & (piv.late > 0))[nn].sum())}/9,"
                f" 뒤 기간 초과 최소 {pct(piv.late[nn].min())}")
    say("")

    # 4) 쏠림
    say("== 4) 쏠림((a) 원천, RSI8)")
    for p in PER:
        T = TA[p]
        for un, us in (("78", U78), ("137", U137)):
            x = T[T.sym.isin(us) & np.isfinite(T.RSI8)].copy()
            x["exc"] = x.RSI8 - x.r1_RSI8
            bs = x.groupby("sym").RSI8.sum().sort_values(ascending=False)
            bd = x.groupby("day").RSI8.sum().sort_values(ascending=False)
            tot = x.RSI8.sum()
            say(f"  {PL[p]} {un}: 합 {tot:+.2f}, 상위1 종목 {bs.index[0]} {bs.iloc[0] / tot:.0%} · 상위5 종목 {bs.head(5).sum() / tot:.0%} ({', '.join(bs.head(5).index)})"
                f" | 상위1 날 {bd.index[0]} {bd.iloc[0] / tot:.0%} · 상위3 날 {bd.head(3).sum() / tot:.0%}")
            for lab, m in (("상위1 종목 뺌", x.sym != bs.index[0]), ("상위3 종목 뺌", ~x.sym.isin(bs.head(3).index)),
                           ("상위1 날 뺌", x.day != bd.index[0]), ("상위2 날 뺌", ~x.day.isin(bd.head(2).index))):
                xx = x[m]
                say(f"      {lab:<10} n{len(xx)} 평균 {pct(xx.RSI8.mean())} 초과 {pct(xx.exc.mean())} te {dct(xx.day.values, xx.exc.values):+.2f}")
            # 날짜별 분포
            dd = x.groupby("day").agg(n=("RSI8", "size"), avg=("RSI8", "mean"), exc=("exc", "mean"))
            say(f"      날 {len(dd)}개 중 초과>0 {int((dd.exc > 0).sum())}, 하루 최다 거래 {int(dd.n.max())}건({dd.n.idxmax()}, 평균 {pct(dd.loc[dd.n.idxmax(), 'avg'])})")
    say("  확인 기간 사용자78 날짜별(RSI8):")
    x = TA["late"][TA["late"].sym.isin(U78)].copy()
    x["exc"] = x.RSI8 - x.r1_RSI8
    for d, g in x.groupby("day"):
        say(f"    {d} n{len(g):>2} 평균 {pct(g.RSI8.mean())} 초과 {pct(g.exc.mean())} 종목 {','.join(g.sym.head(8))}{'…' if len(g) > 8 else ''}")
    say("")

    # 5) 자료 품질: 체결 봉 저가가 공식 일봉 저가보다 낮은가(가짜 꼬리)
    say("== 5) 자료 품질 — 체결 봉 저가가 1d 일봉 Low보다 0.1% 넘게 낮으면 '가짜 꼬리 의심'")
    for p in PER:
        T = TA[p]
        flag = []
        for _, rw in T.iterrows():
            try:
                gb = DLOW[rw.sym]
                dl = gb.get(rw.day, np.nan)
            except KeyError:
                dl = np.nan
            flag.append(np.isfinite(dl) and rw.lo_bar < dl * 0.999)
        T = T.assign(fake=flag)
        TA[p] = T
        for un, us in (("78", U78), ("137", U137)):
            x = T[T.sym.isin(us)]
            f = x[x.fake]
            g = x[~x.fake]
            say(f"  {PL[p]} {un}: 의심 {len(f)}/{len(x)}건 평균 {pct(f.RSI8.mean())} | 의심 뺀 n{len(g)} 평균 {pct(g.RSI8.mean())} 초과 {pct((g.RSI8 - g.r1_RSI8).mean())}"
                  f" te {dct(g.day.values, (g.RSI8 - g.r1_RSI8).values):+.2f} | TP2W {pct(g.TP2W.mean())}")
    # 뒤 기간: 3분봉(1분봉 원천) 꼬리가 5분봉 원천에도 있는가
    T = TA["late"]
    miss = []
    for s, g in T.groupby("sym"):
        b5 = get_bars(s, 5)
        t5 = b5.index.as_unit("ns").asi8
        l5 = b5.Low.values
        Q = Qs[("late", s)]
        for _, rw in g.iterrows():
            t = Q["tns"][int(rw.e)]
            a = max(np.searchsorted(t5, t, side="right") - 1, 0)          # 3분봉 시작을 품은 5분봉
            b = max(np.searchsorted(t5, t + 3 * 60 * 10**9, side="left"), a + 1)   # 3분봉 끝 전에 시작한 5분봉까지
            ok = l5[a:b].min() <= rw.fill * 1.0005
            miss.append((s, rw.day, not ok, rw.RSI8, rw.r1_RSI8))
    M = pd.DataFrame(miss, columns=["sym", "day", "miss", "RSI8", "r1"])
    for un, us in (("78", U78), ("137", U137)):
        m = M[M.sym.isin(us)]
        say(f"  확인 {un}: 3분봉 체결가에 5분봉(별도 원천)이 닿지 않은 거래 {int(m.miss.sum())}/{len(m)}건, 평균 {pct(m[m.miss].RSI8.mean())}"
            f" | 뺀 뒤 평균 {pct(m[~m.miss].RSI8.mean())} 초과 {pct((m[~m.miss].RSI8 - m[~m.miss].r1).mean())}")
    say("")

    # 6) 체결 여유·갭 여부
    say("== 6) 체결 여유와 갭 체결")
    for thr in (0.0, 0.002, 0.005):
        cells = []
        for p in PER:
            T = run(p, thr=thr)
            for un, us in (("78", U78), ("137", U137)):
                st = stat(T, us, "RSI8")
                cells.append(f"{p[0]}{un} n{st['n']} {pct(st['avg'])}/{pct(st['exc'])}({st['te']:+.1f})")
        say(f"  지정가보다 {thr * 100:.1f}% 더 내려가야 체결: " + " | ".join(cells))
    for p in PER:
        T = TA[p]
        for un, us in (("78", U78), ("137", U137)):
            x = T[T.sym.isin(us)]
            gapf = x.o_bar <= x.fill + 1e-9
            for lab, m in (("시가 체결(갭/봉 시작이 지정가 아래)", gapf), ("지정가 체결", ~gapf)):
                xx = x[m]
                say(f"  {PL[p]} {un} {lab}: n{len(xx)} 평균 {pct(xx.RSI8.mean())} 초과 {pct((xx.RSI8 - xx.r1_RSI8).mean())} te {dct(xx.day.values, (xx.RSI8 - xx.r1_RSI8).values):+.2f}")
    say("")

    # 7) 종목 묶음 사후 선정 점검: 사용자78(10/6 보유 = 확인 기간 끝에 정한 묶음) vs 나머지 59(AI 묶음 중 사용자78 밖)
    say("== 7) 종목 묶음 점검 — 사용자78은 10/6 보유 종목(확인 기간 끝에 정해진 묶음). 나머지59 = 전체137 - 사용자78")
    U59 = U137 - U78
    for p in PER:
        T = TA[p]
        for lab, us in (("사용자78", U78), ("나머지59", U59)):
            x = T[T.sym.isin(us)]
            parts = []
            for ex in ("RSI8", "TP2W"):
                s1 = stat(x, us, ex); s3 = stat(x, us, ex, "r3_")
                parts.append(f"{ex} n{s1['n']} 평균 {pct(s1['avg'])} 초과1 {pct(s1['exc'])}(te {s1['te']:+.2f}) 초과3 {pct(s3['exc'])}(te {s3['te']:+.2f})")
            h_ = (x.RSI8 - x.qqq).values
            k = np.isfinite(h_)
            parts.append(f"QQQ 헤지 {pct(h_[k].mean())}(te {dct(x.day.values[k], h_[k]):+.2f})")
            say(f"  {PL[p]} {lab}: " + " | ".join(parts))
    say("  같은 날끼리(확인 기간, RSI8): 날짜별 사용자78 평균 vs 나머지59 평균")
    x = TA["late"]
    a = x[x.sym.isin(U78)].groupby("day").RSI8.agg(["size", "mean"])
    b = x[x.sym.isin(U59)].groupby("day").RSI8.agg(["size", "mean"])
    j = a.join(b, how="outer", lsuffix="_78", rsuffix="_59")
    for d, rw in j.iterrows():
        say(f"    {d} 78 n{0 if not np.isfinite(rw.size_78) else int(rw.size_78):>2} {pct(rw.mean_78)} | 59 n{0 if not np.isfinite(rw.size_59) else int(rw.size_59):>2} {pct(rw.mean_59)}")
    both = j.dropna()
    dd = (both.mean_78 - both.mean_59).values
    say(f"  둘 다 거래한 날 {len(both)}일: 78 - 59 평균 차 {pct(dd.mean())}, t {dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))):+.2f}")
    say("  이웃 25칸 — 나머지59, 확인 기간(RSI8): 평균/초과1(te)")
    cnt = 0
    tot = 0
    for nd in (3, 4, 5, 7, 10):
        Lmap = {}
        for s in have5:
            ga = daily_from_5m(s)
            days = list(ga.index); lo_ = ga.Low.values
            Lmap[s] = {days[i]: lo_[i - nd:i].min() for i in range(nd, len(days))}
        cells = []
        for dep in (0.01, 0.015, 0.02, 0.025, 0.03):
            res = {}
            for p in PER:
                res[p] = stat(run(p, depth=dep, Lmap=Lmap), U59, "RSI8")
            st = res["late"]
            tot += 1
            cnt += int(res["early"]["exc"] > 0 and st["exc"] > 0)
            cells.append(f"-{dep * 100:.1f}% 앞 {pct(res['early']['exc'])} 뒤 n{st['n']} {pct(st['avg'])}/{pct(st['exc'])}({st['te']:+.1f})")
        say(f"    L{nd}D " + " | ".join(cells))
    say(f"  나머지59에서 두 기간 초과>0 칸: {cnt}/{tot}")
    say("")
    # 8) 확인 기간 큰 날의 시장
    say("== 8) 확인 기간 거래가 몰린 날의 QQQ(정규장 시가 갭, 시가→종가)")
    qd = pd.read_parquet(f"{RAW}/1d_QQQ.parquet") if os.path.exists(f"{RAW}/1d_QQQ.parquet") else pd.read_parquet(f"{DLY}/1d_QQQ.parquet")
    qd.index = pd.to_datetime(qd.index).date
    qd["gap"] = qd.Open / qd.Close.shift(1) - 1
    qd["oc"] = qd.Close / qd.Open - 1
    x = TA["late"]
    cnts = x.groupby("day").size().sort_values(ascending=False)
    for d in cnts.index[:6]:
        if d in qd.index:
            say(f"  {d} 거래 {cnts[d]}건(137) QQQ 갭 {pct(qd.loc[d, 'gap'])} 시가→종가 {pct(qd.loc[d, 'oc'])}")
    say(f"소요 {time.time() - t0:.0f}s")


DLOW = {}

if __name__ == "__main__":
    U = sorted(U137)
    for s in U:
        try:
            d = daily_from_1d(s)
            DLOW[s] = d.Low.to_dict()
        except FileNotFoundError:
            pass
    main()
