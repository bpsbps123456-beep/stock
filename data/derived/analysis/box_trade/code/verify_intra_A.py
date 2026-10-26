"""반박 검증 — 분봉 A(장중 박스 바닥 버팀) 독립 재구현 [키 verify_intra_A, 2026-10-09]

연구자 보고의 best_rule.definition 글만 보고 다시 짰다. intra_A.py·stop_lab.py 코드는 쓰지 않는다(종목 묶음 fetch.py만 사용).
규칙 A_OPEN|N120|b0.3|H1:
  봉: 확인 9/10~10/7 = 1분봉→3분봉(시계 정렬), 고르기 7/14~9/8 = 5분봉. 정규장 9:30~16:00 ET 봉만 이어 붙임(창은 전날 정규장으로 이어짐).
  L = 직전 K봉(현재 봉 제외) 저가 최솟값, K = 120분/봉 길이. L을 만든 봉(같은 값이면 먼저 나온 봉)이 현재 봉보다 15분 이상 앞.
  신호(봉 k 종가 확정 뒤): c[k-1] > L(1+b), l[k] <= L(1+b), c[k] >= L.  체결: 같은 날 다음 정규장 봉 k+1 시가.
  매도: 매수 봉 시작 + 60분 이후 첫 정규장 봉 시가(15:00 이후 매수면 다음 날 9:30). 종목당 한 포지션, 판 봉 다음 봉부터 재매수.
비교 기준(모두 같은 매도 = 정규장 봉 시가 매수 → 60분 뒤 첫 정규장 봉 시가):
  칸무작위 = 같은 종목·같은 날·같은 30분 칸의 모든 정규장 봉(연구자 기준과 같은 정의)
  날무작위 = 같은 종목·같은 날 모든 정규장 봉
  시각무작위 = 같은 종목·같은 30분 시각대(기간 안 모든 날) — 날짜 정보 없이 시각대만 맞춘 '보유'
  무조건 = 그 기간 묶음 안 모든 정규장 봉(같은 보유 시간의 '아무 때나 사서 보유')
사용: py verify_intra_A.py → ../results/verify_intra_A.txt
"""
import os, sys, time, math
from datetime import date
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

RAW = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip/raw"
sys.path.insert(0, r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis\dip_reason\code")
from fetch import user_universe, universe  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "results", "verify_intra_A.txt"))
U78 = set(user_universe()["합집합"])
U137 = U78 | set(universe())
SYMS = sorted(U137)
PER = {"early": (5, date(2026, 7, 14), date(2026, 9, 8)),
       "late": (3, date(2026, 9, 10), date(2026, 10, 7)),
       "late5": (5, date(2026, 9, 10), date(2026, 10, 7))}
NS_MIN = 60 * 10**9
_fh = None


def say(*a):
    global _fh
    if _fh is None:
        _fh = open(OUT, "w", encoding="utf-8")
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _fh.write(s + "\n")
    _fh.flush()


def reg_bars(sym, mins):
    if mins == 5:
        p = os.path.join(RAW, f"5m_{sym}.parquet")
        if not os.path.exists(p):
            return None
        b = pd.read_parquet(p)
    else:
        p = os.path.join(RAW, f"1m_{sym}.parquet")
        if not os.path.exists(p):
            return None
        m = pd.read_parquet(p)
        b = m.resample(f"{mins}min", label="left", closed="left").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    b = b[b.Close.notna() & (b.High > 0) & (b.Low > 0)]
    mod = b.index.hour * 60 + b.index.minute
    b = b[(mod >= 570) & (mod < 960)]
    idx = b.index
    return dict(t=idx.as_unit("ns").asi8.astype(np.int64), o=b.Open.values.astype(float), h=b.High.values.astype(float),
                l=b.Low.values.astype(float), c=b.Close.values.astype(float),
                day=np.array([d.toordinal() for d in idx.date], np.int64),
                mod=(idx.hour * 60 + idx.minute).values.astype(np.int64), mins=mins)


def fwd(B, hold_min):
    """봉마다 '시가 매수 → hold_min 뒤 첫 정규장 봉 시가 매도' 수익과 매도 봉."""
    x = np.searchsorted(B["t"], B["t"] + np.int64(hold_min) * NS_MIN, side="left")
    ok = x < len(B["t"])
    r = np.full(len(x), np.nan)
    r[ok] = B["o"][x[ok]] / B["o"][ok] - 1
    return r, np.where(ok, x, -1)


def trades(B, N, b, age_min, tol, hold_min, lo, hi, shift=0.0):
    """규칙 A_OPEN 변형. 반환: (매수 봉 배열, 수익 배열)."""
    mins = B["mins"]
    K = N // mins
    l, c, o, day = B["l"], B["c"], B["o"], B["day"]
    n = len(l)
    if n <= K + 1:
        return np.zeros(0, np.int64), np.zeros(0)
    L = np.full(n, np.nan)
    pos = np.full(n, -10**9, np.int64)
    W = swv(l[:-1], K)                     # W[j] = l[j..j+K-1] → 봉 k=j+K의 직전 K봉
    L[K:] = W.min(axis=1)
    pos[K:] = W.argmin(axis=1) + np.arange(n - K)   # argmin = 먼저 나온 봉
    L = L * (1 + shift)
    age = np.arange(n) - pos
    need = math.ceil(age_min / mins)
    band = L * (1 + b)
    cp = np.r_[np.nan, c[:-1]]
    sig = np.zeros(n, bool)
    with np.errstate(invalid="ignore"):
        sig[K:] = (age[K:] >= need) & (cp[K:] > band[K:]) & (l[K:] <= band[K:]) & (c[K:] >= L[K:] * (1 - tol))
    k = np.nonzero(sig[:-1])[0]
    k = k[day[k + 1] == day[k]]
    e = k + 1
    e = e[(day[e] >= lo) & (day[e] <= hi)]
    r, x = fwd(B, hold_min)
    E, R = [], []
    last = -1
    for ei in e:
        if ei <= last:
            continue
        xi = x[ei]
        if xi < 0:
            break
        E.append(ei)
        R.append(r[ei])
        last = xi
    return np.array(E, np.int64), np.array(R)


def dct(day, x):
    s = pd.Series(np.asarray(x, float)).groupby(np.asarray(day)).sum()
    if len(s) < 3 or s.std(ddof=1) == 0:
        return np.nan
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def fp(v, k=2):
    return "  n/a" if v is None or not np.isfinite(v) else f"{v * 100:+.{k}f}%"


def ft(v):
    return " n/a" if v is None or not np.isfinite(v) else f"{v:+.1f}"


def base_tables(B, hold_min, lo, hi):
    """기간 안 정규장 봉의 기준 수익과 칸·날·시각대 평균."""
    r, _ = fwd(B, hold_min)
    df = pd.DataFrame(dict(r=r, day=B["day"], slot=B["mod"] // 30))
    df = df[(df.day >= lo) & (df.day <= hi)]
    return df


def run(period, variant, cache):
    mins, d0, d1 = PER[period]
    lo, hi = d0.toordinal(), d1.toordinal()
    rows = []
    unc = []
    for s in SYMS:
        key = (s, mins)
        if key not in cache:
            cache[key] = reg_bars(s, mins)
        B = cache[key]
        if B is None:
            continue
        N, b, age, tol, hold, shift = variant
        E, R = trades(B, N, b, age, tol, hold, lo, hi, shift)
        bt = base_tables(B, hold, lo, hi)
        okb = bt[bt.r.notna()]
        unc.append(dict(sym=s, s=okb.r.sum(), k=len(okb)))
        if not len(E):
            continue
        slot_m = okb.groupby(["day", "slot"]).r.mean()
        day_m = okb.groupby("day").r.mean()
        tod_m = okb.groupby("slot").r.mean()
        dd, sl = B["day"][E], B["mod"][E] // 30
        rows.append(pd.DataFrame(dict(sym=s, day=dd, mod=B["mod"][E], e=E, ret=R,
                                      r_slot=slot_m.reindex(pd.MultiIndex.from_arrays([dd, sl])).values,
                                      r_day=day_m.reindex(dd).values, r_tod=tod_m.reindex(sl).values)))
    T = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return T, pd.DataFrame(unc)


def summ(T, U, syms):
    t = T[T.sym.isin(syms)]
    u = U[U.sym.isin(syms)]
    unc = u.s.sum() / u.k.sum()
    exs = t.ret - t.r_slot
    exd = t.ret - t.r_day
    ext = t.ret - t.r_tod
    return dict(n=len(t), avg=t.ret.mean(), win=(t.ret > 0).mean(), worst=t.ret.min(), sum=t.ret.sum(), t=dct(t.day, t.ret),
                slot=t.r_slot.mean(), exs=exs.mean(), tes=dct(t.day[exs.notna()], exs.dropna()),
                exd=exd.mean(), ted=dct(t.day, exd), ext=ext.mean(), tet=dct(t.day, ext), unc=unc, exu=t.ret.mean() - unc)


def line(s):
    return (f"n {s['n']:>5} 평균 {fp(s['avg'])} 승률 {s['win']:.0%} 최악 {fp(s['worst'], 1)} 합 {s['sum']:+6.2f} t {ft(s['t'])} | "
            f"칸무작위 {fp(s['slot'])} 초과 {fp(s['exs'])} te {ft(s['tes'])} | 날무작위 초과 {fp(s['exd'])} te {ft(s['ted'])} | "
            f"시각무작위 초과 {fp(s['ext'])} te {ft(s['tet'])} | 무조건 {fp(s['unc'], 3)} 차 {fp(s['exu'])}")


def main():
    T0 = time.time()
    say("반박 검증 — 분봉 A 독립 재구현 [verify_intra_A] 2026-10-09")
    say(__doc__.split("사용:")[0].strip())
    say("")
    cache = {}
    BEST = (120, 0.003, 15, 0.0, 60, 0.0)
    res = {}
    say("== 1) 재현: A_OPEN|N120|b0.3|H1 (정규장)")
    say("   연구자 보고: early 사용자78 n 7011 평균 +0.02% 칸무작위 -0.08% 초과 +0.11% te 14.4 / 전체137 n 12234 초과 +0.11% te 18.0")
    say("               late  사용자78 n 3478 평균 +0.07% 칸무작위 -0.04% 초과 +0.11% te 9.2 / 전체137 n 6058 초과 +0.13% te 12.8")
    say("               무조건 무작위 H1: early 사용자78 +0.037% / late 사용자78 +0.077%")
    for pk in ("early", "late", "late5"):
        T, U = run(pk, BEST, cache)
        res[pk] = (T, U)
        for un, us in (("사용자78", U78), ("전체137", U137)):
            say(f"  {pk:<5} {un:<6} {line(summ(T, U, us))}")
    say("")

    # ---------------------------------------------------------------- 2) 쏠림
    say("== 2) 쏠림 점검 (정규장·사용자78)")
    for pk in ("early", "late"):
        T, U = res[pk]
        t = T[T.sym.isin(U78)].copy()
        t["exs"] = t.ret - t.r_slot
        t["exu"] = t.ret - summ(T, U, U78)["unc"]
        byd = t.groupby("day").agg(ret=("ret", "sum"), exs=("exs", "sum"), n=("ret", "size"))
        bys = t.groupby("sym").agg(ret=("ret", "sum"), exs=("exs", "sum"), n=("ret", "size"))
        say(f"  {pk}: 날 {len(byd)}일, 종목 {len(bys)}개. 수익 합 {t.ret.sum():+.2f}, 칸초과 합 {t.exs.sum():+.2f}")
        top = byd.sort_values("ret", ascending=False)
        say("    수익 합 상위 날: " + ", ".join(f"{date.fromordinal(int(d)).strftime('%m/%d')} {v.ret:+.2f}(n{int(v.n)})" for d, v in top.head(4).iterrows()))
        say("    수익 합 하위 날: " + ", ".join(f"{date.fromordinal(int(d)).strftime('%m/%d')} {v.ret:+.2f}(n{int(v.n)})" for d, v in top.tail(3).iterrows()))
        for drop in (1, 2, 3):
            keep = ~t.day.isin(top.head(drop).index)
            say(f"    수익 상위 {drop}일 빼면: n {keep.sum()} 평균 {fp(t.ret[keep].mean())} (무조건 {fp(summ(T, U, U78)['unc'], 3)}), 칸초과 {fp(t.exs[keep].mean())}")
        tope = byd.sort_values("exs", ascending=False)
        say(f"    칸초과 합 상위 2일 몫 {tope.exs.head(2).sum() / t.exs.sum():.0%}, 칸초과 플러스인 날 {int((byd.exs > 0).sum())}/{len(byd)}")
        tops = bys.sort_values("exs", ascending=False)
        say("    칸초과 상위 종목: " + ", ".join(f"{s} {v.exs:+.2f}(n{int(v.n)})" for s, v in tops.head(5).iterrows())
            + f" | 상위 5종목 몫 {tops.exs.head(5).sum() / t.exs.sum():.0%}, 칸초과 플러스 종목 {int((bys.exs > 0).sum())}/{len(bys)}")
        topr = bys.sort_values("ret", ascending=False)
        say("    수익 상위 종목: " + ", ".join(f"{s} {v.ret:+.2f}(n{int(v.n)})" for s, v in topr.head(5).iterrows())
            + f" | 상위 1종목 빼면 평균 {fp(t.ret[t.sym != topr.index[0]].mean())}, 상위 3종목 빼면 {fp(t.ret[~t.sym.isin(topr.index[:3])].mean())}")
        late_day = t["mod"] >= 900
        say(f"    15:00 이후 매수(다음 날 시가까지 보유) n {late_day.sum()} 평균 {fp(t.ret[late_day].mean())} 칸초과 {fp(t.exs[late_day].mean())} | "
            f"15:00 전 n {(~late_day).sum()} 평균 {fp(t.ret[~late_day].mean())} 칸초과 {fp(t.exs[~late_day].mean())}")
        awake = t["mod"] < 660
        say(f"    9:30~11:00 ET 매수(한국 깨어 있는 시간) n {awake.sum()} 평균 {fp(t.ret[awake].mean())} 칸초과 {fp(t.exs[awake].mean())} "
            f"시각초과 {fp((t.ret - t.r_tod)[awake].mean())}")
        w = t.ret.idxmin()
        say(f"    최악 거래: {t.sym[w]} {date.fromordinal(int(t.day[w]))} {t['mod'][w] // 60}:{t['mod'][w] % 60:02d} {fp(t.ret[w], 1)}")
    say("")

    # ---------------------------------------------------------------- 3) 이웃
    say("== 3) 이웃 매개변수 (정규장). 한 칸씩만 바꿈. (N분, 밴드, 바닥나이분, 종가이탈허용, 보유분, 가짜바닥)")
    VARS = [("기준 N120 b0.3 age15 tol0 H60", BEST),
            ("N60", (60, 0.003, 15, 0.0, 60, 0.0)), ("N90", (90, 0.003, 15, 0.0, 60, 0.0)),
            ("N180", (180, 0.003, 15, 0.0, 60, 0.0)), ("N240", (240, 0.003, 15, 0.0, 60, 0.0)),
            ("b0.15", (120, 0.0015, 15, 0.0, 60, 0.0)), ("b0.6", (120, 0.006, 15, 0.0, 60, 0.0)),
            ("age0", (120, 0.003, 0, 0.0, 60, 0.0)), ("age30", (120, 0.003, 30, 0.0, 60, 0.0)),
            ("이탈허용0.3%(c>=L*0.997)", (120, 0.003, 15, 0.003, 60, 0.0)),
            ("H30", (120, 0.003, 15, 0.0, 30, 0.0)), ("H120", (120, 0.003, 15, 0.0, 120, 0.0)),
            ("가짜바닥+0.6%", (120, 0.003, 15, 0.0, 60, 0.006)), ("가짜바닥+1.2%", (120, 0.003, 15, 0.0, 60, 0.012)),
            ("가짜바닥+2.0%", (120, 0.003, 15, 0.0, 60, 0.020))]
    NB = []
    for name, v in VARS:
        for pk in ("early", "late"):
            T, U = run(pk, v, cache) if v != BEST else res[pk]
            for un, us in (("사용자78", U78), ("전체137", U137)):
                s = summ(T, U, us)
                NB.append(dict(var=name, pk=pk, un=un, **s))
                say(f"  {name:<26} {pk:<5} {un:<6} n {s['n']:>5} 평균 {fp(s['avg'])} t {ft(s['t'])} | 칸초과 {fp(s['exs'])} te {ft(s['tes'])} | "
                    f"날초과 {fp(s['exd'])} te {ft(s['ted'])} | 시각초과 {fp(s['ext'])} te {ft(s['tet'])} | 무조건차 {fp(s['exu'])}")
    nb = pd.DataFrame(NB)
    nb.to_csv(OUT.replace(".txt", "_neighbors.csv"), index=False, encoding="utf-8-sig")
    real = nb[~nb["var"].str.startswith("가짜")]
    for col, lab in (("exs", "칸초과"), ("exd", "날초과"), ("ext", "시각초과"), ("exu", "무조건차")):
        pos = (real[col] > 0).sum()
        say(f"  {lab} > 0 인 칸: {pos}/{len(real)} (진짜 바닥 변형 12개 × 2기간 × 2묶음)")
    say("")

    # ---------------------------------------------------------------- 4) 칸무작위 분해
    say("== 4) 칸무작위 분해 — 같은 30분 칸의 봉을 매수 봉 앞(신호를 만든 하락 전)·뒤로 나눔. 정규장.")
    say("   칸 앞 봉은 '이 칸에서 박스 바닥까지 떨어진다'는 사후 정보가 든 매수라 기준을 낮춘다.")
    for pk in ("early", "late"):
        mins, d0, d1 = PER[pk]
        lo, hi = d0.toordinal(), d1.toordinal()
        rows = []
        for s in SYMS:
            B = cache.get((s, mins))
            if B is None:
                continue
            E, R = trades(B, *BEST[:5], lo, hi, 0.0)
            r, _ = fwd(B, 60)
            key = B["day"] * 100 + B["mod"] // 30
            for e, ret in zip(E, R):
                idx = np.nonzero((key == key[e]) & np.isfinite(r))[0]
                pre, post = idx[idx < e], idx[idx >= e]
                rows.append(dict(sym=s, day=B["day"][e], ret=ret, pre=r[pre].mean() if len(pre) else np.nan,
                                 post=r[post].mean() if len(post) else np.nan, npre=len(pre)))
        D = pd.DataFrame(rows)
        for un, us in (("사용자78", U78), ("전체137", U137)):
            t = D[D.sym.isin(us)]
            ex = (t.ret - t.post).dropna()
            say(f"  {pk:<5} {un:<6} 규칙 {fp(t.ret.mean(), 3)} | 칸 앞 봉 매수 {fp(t.pre.mean(), 3)} (봉 {t.npre.sum()}) | "
                f"칸 뒤 봉(매수 봉 포함) {fp(t.post.mean(), 3)} | 규칙 - 칸 뒤 = {fp(ex.mean(), 3)} te {ft(dct(t.day[ex.index], ex))}")
    say("")
    say(f"실행 시간 {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
