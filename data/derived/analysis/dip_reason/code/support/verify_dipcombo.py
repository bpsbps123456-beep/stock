"""dipcombo 반박 검증(2026-10-09). 연구자 코드(dipcombo.py)를 쓰지 않고 best_rule.definition 글만 보고 다시 만든다.

정의(연구자 보고 그대로):
  구역: 5분봉(프리·애프터 포함)을 ET 04:00 기준 40분봉으로 묶음. 좌우 5봉 피벗(저점: l[p]가 p-5~p+5 최저, 왼쪽 5봉 모두 l[p]보다 큼;
        고점 대칭). 피벗 p는 진행 중 40분봉 k에서 p+5<k일 때만. 최근 스윙 60개 가격순, 첫 가격 대비 0.5% 안이면 한 구역
        (lo=첫, hi=마지막, cnt), 구역 중심끼리 1.2% 미만이면 (cnt, 마지막 스윙 봉) 큰 쪽만.
  버팀: c[i-1]이 [lo×0.995, hi×1.02] 안인 구역 중 중심이 가장 가까운 것. 완성 40분봉을 거꾸로, 종가가 띠 밖이면 멈춤,
        그 연속 구간에서 저가 ≤ hi×1.01인 가장 이른 봉부터 k-1까지 개수 = T.
  진입: T≥12 & C40[k-1] ≥ lo×0.995 → 지정가 max(EMA[i-1]×0.98, hi), 아니면 EMA[i-1]×0.98. 저가 ≤ 지정가면 min(시가, 지정가).
  매도: RSI[j-1]>70 → j 시가, 아니면 매수 뒤 첫 한국 08시 이후 첫 봉 시가. 전날 +7% 다음 날 정규장 매수 안 함.
기준봉·EMA·RSI·08시·hot7 배열은 공용 시험대 stop_lab.load()에서 가져온다(연구자와 같은 시험대). 그 밖(40분봉, 피벗, 구역,
버팀, 상태 기계, 지표, 무작위 기준)은 모두 새로 짰다.
사용: py support/verify_dipcombo.py  (code 폴더) → ../results/support/verify_dipcombo.txt
"""
import os, sys, time
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, base_trades, UNIV, RAW

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "support", "verify_dipcombo.txt"))
WIN = {"early": (date(2026, 7, 28), date(2026, 9, 8)), "late": (date(2026, 9, 10), date(2026, 10, 6))}
DEF = dict(tf=40, tol=0.005, zgap=0.012, keep=60, brk=0.005, upl=0.02, touch=0.01, near=0.01)
_fh = open(OUT, "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _fh.write(s + "\n"); _fh.flush()


# ---------------------------------------------------------------- 40분봉(날짜 + 04:00부터 분 // tf 로 직접 묶음)
_B = {}


def big_bars(sym, tf):
    if (sym, tf) in _B:
        return _B[(sym, tf)]
    m = pd.read_parquet(os.path.join(RAW, f"5m_{sym}.parquet"))
    m = m[(m.High > 0) & m.Close.notna()]
    et = m.index
    off = np.asarray(et.hour * 60 + et.minute) - 240
    m, et, off = m[off >= 0], et[off >= 0], off[off >= 0]
    start = (et.normalize() + pd.to_timedelta((off // tf) * tf + 240, unit="m")).as_unit("ns").asi8
    df = pd.DataFrame({"s": start, "h": m.High.values, "l": m.Low.values, "c": m.Close.values})
    g = df.groupby("s", sort=True)
    agg = pd.DataFrame({"h": g.h.max(), "l": g.l.min(), "c": g.c.last()})
    r = (agg.index.values.astype(np.int64), agg.h.values.astype(float), agg.l.values.astype(float), agg.c.values.astype(float))
    _B[(sym, tf)] = r
    return r


def find_pivots(H, Lw, n=5):
    out = []                                   # (봉, 종류 0=저점 1=고점, 가격) — 같은 봉이면 저점 먼저
    for p in range(n, len(Lw) - n):
        if Lw[p] <= Lw[p - n:p + n + 1].min() and (Lw[p - n:p] > Lw[p]).all():
            out.append((p, 0, Lw[p]))
        if H[p] >= H[p - n:p + n + 1].max() and (H[p - n:p] < H[p]).all():
            out.append((p, 1, H[p]))
    return out


def build_zones(sw, tol, zgap):
    order = sorted(range(len(sw)), key=lambda q: sw[q][2])   # 안정 정렬
    Z = []
    for q in order:
        b, _, p = sw[q]
        if Z and (p - Z[-1][0]) / Z[-1][0] <= tol:
            Z[-1][1] = p; Z[-1][2] += 1; Z[-1][3] = max(Z[-1][3], b)
        else:
            Z.append([p, p, 1, b])
    kept = []
    for z in sorted(Z, key=lambda z: (z[2], z[3]), reverse=True):
        c = (z[0] + z[1]) / 2
        ok = True
        for y in kept:
            cy = (y[0] + y[1]) / 2
            if abs(c - cy) / cy < zgap:
                ok = False; break
        if ok:
            kept.append(z)
    kept.sort(key=lambda z: (z[0] + z[1]) / 2)
    return (np.array([z[0] for z in kept]), np.array([z[1] for z in kept]), np.array([z[2] for z in kept], float))


def hold_time(k, lo, hi, C, Lw, brk, upl, touch):
    a, b, th = lo * (1 - brk), hi * (1 + upl), hi * (1 + touch)
    j, first = k - 1, None
    while j >= 0 and a <= C[j] <= b:
        if Lw[j] <= th:
            first = j
        j -= 1
    T = (k - first) if first is not None else 0
    broken = k - 1 >= 0 and C[k - 1] < a
    return T, broken


_ZC = {}


def zone_state(A, P, prm, lo_m, hi_m):
    """봉마다 P가 [lo×(1-lo_m), hi×(1+hi_m)] 안인 가장 가까운 구역과 그 버팀 T(띠: brk·upl·touch)."""
    sym, tf = A["sym"], prm["tf"]
    st, H, Lw, C = big_bars(sym, tf)
    zkey = (sym, tf, prm["tol"], prm["zgap"], prm["keep"])
    if zkey not in _ZC:
        piv = find_pivots(H, Lw)
        _ZC[zkey] = (piv, np.array([x[0] for x in piv], np.int64), {})
    piv, pbar, zcache = _ZC[zkey]
    off = A["mod"] - 240
    bstart = (A["idx"].normalize() + pd.to_timedelta((off // tf) * tf + 240, unit="m")).as_unit("ns").asi8
    k = np.searchsorted(st, bstart, side="left")                 # 완성 큰 봉 = 0..k-1
    v = np.searchsorted(pbar, k - 6, side="right")               # p <= k-6 (p+5 < k)인 피벗 수
    n = len(P)
    zlo, zhi, zcnt = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    T, brk_ = np.full(n, -1, np.int64), np.zeros(n, bool)
    good = np.isfinite(P)
    for vv in np.unique(v[good]):
        if vv not in zcache:
            zcache[vv] = build_zones(piv[max(0, vv - prm["keep"]):vv], prm["tol"], prm["zgap"]) if vv > 0 else None
        z = zcache[vv]
        if z is None or not len(z[0]):
            continue
        ii = np.flatnonzero(good & (v == vv))
        p = P[ii][:, None]
        lo, hi, cnt = z
        inb = (p >= lo * (1 - lo_m)) & (p <= hi * (1 + hi_m))
        dist = np.where(inb, np.abs(p - (lo + hi) / 2), np.inf)
        j = dist.argmin(1)
        has = np.isfinite(dist[np.arange(len(ii)), j])
        zlo[ii[has]], zhi[ii[has]], zcnt[ii[has]] = lo[j[has]], hi[j[has]], cnt[j[has]]
    memo = {}
    for i in np.flatnonzero(np.isfinite(zlo)):
        key = (k[i], zlo[i], zhi[i])
        if key not in memo:
            memo[key] = hold_time(k[i], zlo[i], zhi[i], C, Lw, prm["brk"], prm["upl"], prm["touch"])
        T[i], brk_[i] = memo[key]
    return dict(lo=zlo, hi=zhi, cnt=zcnt, T=T, broken=brk_, found=np.isfinite(zlo))


# ---------------------------------------------------------------- 상태 기계(직접 앞으로 훑기)
def simulate(A, lim, reg_only):
    o, l, r, tk, dead = A["o"], A["l"], A["r"], A["tk"], A["dead"]
    n = len(o)
    with np.errstate(invalid="ignore"):
        hit = (l <= lim) & ~A["hot7"]
    if reg_only:
        hit &= A["reg"]
    hit[:61] = False
    cand = np.flatnonzero(hit)
    out, p = [], 0
    while p < len(cand):
        ei = cand[p]
        ent, dl = min(o[ei], lim[ei]), dead[ei]
        j, x = ei + 1, None
        while j < n:
            if r[j - 1] > 70:
                x = (j, "RSI"); break
            if tk[j] >= dl:
                x = (j, "8시"); break
            j += 1
        if x is None:
            break
        out.append((ei, x[0], ent, o[x[0]] / ent - 1, x[1]))
        p = np.searchsorted(cand, x[0] + 1)
    return out


def trades(S, limfn, period, reg_only, extra=None):
    lo, hi = WIN[period]
    rows = []
    for s, A in S.items():
        lim = limfn(s, A)
        for ei, xi, ent, ret, why in simulate(A, lim, reg_only):
            d = A["day"][ei]
            if lo <= d <= hi:
                rec = dict(sym=s, day=d, ses=A["ses"][ei], hr=int(A["mod"][ei] // 60), ret=ret, ei=ei, ent=ent, why=why)
                if extra is not None:
                    rec.update(extra(s, ei))
                rows.append(rec)
    return pd.DataFrame(rows, columns=["sym", "day", "ses", "hr", "ret", "ei", "ent", "why"] + ([] if extra is None else ["T", "z"]))


# ---------------------------------------------------------------- 지표
def cl_t(x, day):
    x = np.asarray(x, float); day = np.asarray(day)
    if len(x) < 3:
        return np.nan
    m = x.mean()
    e = pd.Series(x - m).groupby(day).sum().values
    se = np.sqrt((e ** 2).sum()) / len(x)
    return m / se if se > 0 else np.nan


def st(t):
    if not len(t):
        return dict(n=0, avg=np.nan, win=np.nan, worst=np.nan, sum=0.0, t=np.nan)
    return dict(n=len(t), avg=t.ret.mean(), win=(t.ret > 0).mean(), worst=t.ret.min(), sum=t.ret.sum(), t=cl_t(t.ret, t.day))


def diff(t, b):
    days = sorted(set(t.day) | set(b.day))
    dd = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    sd = dd.std(ddof=1)
    return t.ret.sum() - b.ret.sum(), (dd.mean() / (sd / np.sqrt(len(dd))) if sd > 0 else 0.0), dd


def fmt(d):
    if not d["n"]:
        return "    0건"
    return (f"{d['n']:5d}건 평균 {d['avg']:+.2%} 승률 {d['win']:.0%} 최악 {d['worst']:+.1%} 합 {d['sum']:+6.2f} t {d['t']:+.1f}")


# ---------------------------------------------------------------- 무작위 기준(봉 시가 진입, 같은 매도)
def rand_exp(S, period, reg_only):
    lo, hi = WIN[period]
    parts = []
    for s, A in S.items():
        o, r, tk, dead = A["o"], A["r"], A["tk"], A["dead"]
        n = len(o)
        above = np.flatnonzero(r > 70)
        q = np.searchsorted(above, np.arange(n))                       # 봉 i 이상에서 첫 RSI>70
        jr = np.where(q < len(above), above[np.minimum(q, len(above) - 1)] + 1, n + 1)
        jd = np.searchsorted(tk, dead, side="left")
        j = np.minimum(jr, jd)
        dd = np.array([lo <= x <= hi for x in A["day"]])
        m = dd & ~A["hot7"] & (j < n)
        if reg_only:
            m &= A["reg"]
        ii = np.flatnonzero(m)
        parts.append(pd.DataFrame({"sym": s, "ses": A["ses"][ii], "hr": A["mod"][ii] // 60, "rr": o[j[ii]] / o[ii] - 1}))
    return pd.concat(parts).groupby(["sym", "ses", "hr"]).rr.mean()


def excess(t, RE):
    if not len(t):
        return np.nan, 0.0
    e = t.ret.values - RE.reindex(pd.MultiIndex.from_arrays([t.sym, t.ses, t.hr])).values
    return np.nanmean(e), np.nansum(e)


# ---------------------------------------------------------------- 규칙
def lvl_of(A):
    return np.r_[np.nan, A["ema"][:-1]] * 0.98


def make_R(t0, prm, cache):
    def f(s, A):
        key = (s, A["mins"], tuple(sorted(prm.items())))
        if key not in cache:
            cache[key] = zone_state(A, np.r_[np.nan, A["c"][:-1]], prm, prm["brk"], prm["upl"])
        z = cache[key]
        lv = lvl_of(A)
        held = z["found"] & ~z["broken"] & (z["T"] >= t0)
        return np.where(held, np.fmax(lv, z["hi"]), lv)
    return f


def base_lim(s, A):
    return lvl_of(A)


def main():
    t00 = time.time()
    say("dipcombo 반박 검증(독립 재구현) —", time.strftime("%Y-%m-%d %H:%M"))
    S = {p: load(p) for p in ("early", "late")}
    ZS = {}
    # 0) 기준 거래 재현
    say("\n[0] 지금 이격매수 재현(내 상태 기계) vs stop_lab.base_trades")
    BASE = {}
    for p in ("early", "late"):
        for reg in (False, True):
            b = trades(S[p], base_lim, p, reg)
            BASE[(p, reg)] = b
            ref = base_trades(p, reg)
            ref = ref[(ref.day >= WIN[p][0]) & (ref.day <= WIN[p][1])]
            same = len(b) == len(ref) and np.allclose(np.sort(b.ret.values), np.sort(ref.ret.values))
            say(f"  {p} {'정규' if reg else '24h'}: 내 {len(b)}건 합 {b.ret.sum():+.2f} / stop_lab {len(ref)}건 합 {ref.ret.sum():+.2f} 일치={same}")
    RE = {(p, reg): rand_exp(S[p], p, reg) for p in ("early", "late") for reg in (False, True)}

    # 1) best rule 재현
    say("\n[1] 최선안 R 상단 T>=12 재현 (연구자 보고값과 비교)")
    rep = {("early", False, "사용자78"): (3636, 0.65, 23.66, 0.15), ("early", False, "전체137"): (5883, 0.65, 38.27, 1.63),
           ("early", True, "사용자78"): (1620, 0.27, 4.29, 1.06), ("early", True, "전체137"): (2837, 0.27, 7.60, 1.33),
           ("late", False, "사용자78"): (1825, 0.80, 14.65, -2.39), ("late", False, "전체137"): (2876, 0.79, 22.67, -2.12),
           ("late", True, "사용자78"): (843, 0.38, 3.20, 0.92), ("late", True, "전체137"): (1509, 0.48, 7.25, 1.31)}
    best = make_R(12, DEF, ZS)
    BEST = {}
    for p in ("early", "late"):
        for reg in (False, True):
            t = trades(S[p], best, p, reg)
            BEST[(p, reg)] = t
            for un, us in UNIV.items():
                tt, bb = t[t.sym.isin(us)], BASE[(p, reg)][BASE[(p, reg)].sym.isin(us)]
                d = st(tt); ds, dtt, _ = diff(tt, bb)
                ex, exs = excess(tt, RE[(p, reg)]); bex, bexs = excess(bb, RE[(p, reg)])
                r_ = rep[(p, reg, un)]
                say(f"  {p:5s} {'정규' if reg else '24h ':4s} {un:6s} 내: {fmt(d)} 차이 {ds:+.2f} dt {dtt:+.1f} 무작위초과합 {exs:+.2f}(기준 {bexs:+.2f})"
                    f" | 보고: {r_[0]}건 평균 {r_[1]:+.2f}% 합 {r_[2]:+.2f} 차이 {r_[3]:+.2f}")
    say(f"  ({time.time() - t00:.0f}초)")

    # 2) 쏠림: 최선안 - 기준 날짜별·종목별
    say("\n[2] 쏠림 — 최선안 - 기준 수익 합 차이의 날짜·종목 분해")
    for p in ("early", "late"):
        for reg in (False, True):
            for un in ("사용자78",):
                us = UNIV[un]
                t, b = BEST[(p, reg)], BASE[(p, reg)]
                t, b = t[t.sym.isin(us)], b[b.sym.isin(us)]
                ds, dtt, dd = diff(t, b)
                ds_sym = t.groupby("sym").ret.sum().sub(b.groupby("sym").ret.sum(), fill_value=0).sort_values()
                top_d = dd.sort_values()
                say(f"  {p} {'정규' if reg else '24h'} {un}: 차이 {ds:+.2f}, 나은 날 {(dd > 1e-12).sum()} / 나쁜 날 {(dd < -1e-12).sum()} / 전체 {len(dd)}일")
                say(f"    가장 좋은 날 {top_d.index[-1]} {top_d.iloc[-1]:+.2f}, 2위 {top_d.index[-2]} {top_d.iloc[-2]:+.2f}; 가장 나쁜 날 {top_d.index[0]} {top_d.iloc[0]:+.2f}")
                say(f"    가장 좋은 날 빼면 차이 {ds - top_d.iloc[-1]:+.2f}, 가장 나쁜 날 빼면 {ds - top_d.iloc[0]:+.2f}")
                say(f"    종목 상위 {ds_sym.index[-1]} {ds_sym.iloc[-1]:+.2f}, {ds_sym.index[-2]} {ds_sym.iloc[-2]:+.2f} / 하위 {ds_sym.index[0]} {ds_sym.iloc[0]:+.2f}, {ds_sym.index[1]} {ds_sym.iloc[1]:+.2f};"
                    f" 상위 1종목 빼면 {ds - ds_sym.iloc[-1]:+.2f}, 하위 1종목 빼면 {ds - ds_sym.iloc[0]:+.2f}")

    # 3) 단조성: 기준 거래를 매수 지정가(EMA×0.98)의 ±1% 구역 상태로 나눔
    say("\n[3] 단조성 — 지금 이격매수 거래를 매수 지정가 기준 구역(±1%) 상태로 나눔")
    BK = [("공중", None), ("깨짐", None), ("T0", (0, 0)), ("T1-2", (1, 2)), ("T3-5", (3, 5)), ("T6-11", (6, 11)),
          ("T12-23", (12, 23)), ("T24+", (24, 10 ** 9))]
    ZL = {}
    for p in ("early", "late"):
        for s, A in S[p].items():
            ZL[(p, s)] = zone_state(A, lvl_of(A), DEF, DEF["near"], DEF["near"])
    rng = np.random.default_rng(11)
    for reg in (False, True):
        for un in UNIV:
            for p in ("early", "late"):
                b = BASE[(p, reg)]
                b = b[b.sym.isin(UNIV[un])].copy()
                b["found"] = [ZL[(p, s)]["found"][e] for s, e in zip(b.sym, b.ei)]
                b["brk"] = [ZL[(p, s)]["broken"][e] for s, e in zip(b.sym, b.ei)]
                b["T"] = [ZL[(p, s)]["T"][e] for s, e in zip(b.sym, b.ei)]
                say(f"  [{p} {'정규' if reg else '24h'} {un}] 전체 {fmt(st(b))}")
                for nm, rg in BK:
                    if nm == "공중":
                        m = ~b.found
                    elif nm == "깨짐":
                        m = b.found & b.brk
                    else:
                        m = b.found & ~b.brk & (b["T"] >= rg[0]) & (b["T"] <= rg[1])
                    ex, _ = excess(b[m], RE[(p, reg)])
                    say(f"    {nm:7s} {fmt(st(b[m]))}  무작위초과 {ex:+.2%}" if m.sum() else f"    {nm:7s}     0건")
                m = (b.found & ~b.brk & (b["T"] >= 12)).values
                if m.sum() >= 3:
                    days = np.array(sorted(set(b.day)))
                    g1 = b[m].groupby("day").ret.agg(["sum", "count"]).reindex(days, fill_value=0)
                    g0 = b[~m].groupby("day").ret.agg(["sum", "count"]).reindex(days, fill_value=0)
                    est = g1["sum"].sum() / g1["count"].sum() - g0["sum"].sum() / g0["count"].sum()
                    w = rng.multinomial(len(days), np.ones(len(days)) / len(days), size=4000)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        bd = (w @ g1["sum"].values) / (w @ g1["count"].values) - (w @ g0["sum"].values) / (w @ g0["count"].values)
                    bd = bd[np.isfinite(bd)]
                    say(f"    T>=12 vs 나머지: {m.sum()}건 vs {(~m).sum()}건 평균 차이 {est:+.2%}  날짜 부트스트랩 P(<=0)={(bd <= 0).mean():.2f}")

    # 3b) 구역 상단만 지정가(버팀 무시)로 산 거래를 매수 순간 T로 나눔
    say("\n[3b] 구역 상단 지정가만(Z, 버팀 무시)으로 산 거래 — 매수 순간 T별(무작위 대비 초과)")

    def z_only(s, A):
        z = ZS[(s, A["mins"], tuple(sorted(DEF.items())))]
        return np.where(z["found"] & ~z["broken"], z["hi"], np.nan)
    for reg in (False, True):
        for p in ("early", "late"):
            def ext(s, ei, p=p):
                z = ZS[(s, S[p][s]["mins"], tuple(sorted(DEF.items())))]
                return dict(T=int(z["T"][ei]), z=float(z["hi"][ei]))
            t = trades(S[p], z_only, p, reg, extra=ext)
            t = t[t.sym.isin(UNIV["사용자78"])]
            row = []
            for nm, rg in BK[2:]:
                m = (t["T"] >= rg[0]) & (t["T"] <= rg[1])
                ex, _ = excess(t[m], RE[(p, reg)])
                row.append(f"{nm} {m.sum()}건 {t[m].ret.mean():+.2%}/초과{ex:+.2%}")
            say(f"  {p} {'정규' if reg else '24h'} 사용자78: " + " | ".join(row))
    say(f"  ({time.time() - t00:.0f}초)")

    # 4) 이웃 매개변수
    say("\n[4] 이웃 매개변수 — 사용자78 수익 합 차이(규칙-기준) early | late, 24h / 정규")
    var = [(f"t0={t0}", t0, {}) for t0 in (0, 3, 6, 8, 10, 12, 14, 16, 20, 24)]
    var += [("t0=12 깨짐0.3%", 12, dict(brk=0.003)), ("t0=12 깨짐1%", 12, dict(brk=0.01)),
            ("t0=12 떠남1%", 12, dict(upl=0.01)), ("t0=12 떠남3%", 12, dict(upl=0.03)),
            ("t0=12 닿음0.5%", 12, dict(touch=0.005)), ("t0=12 닿음2%", 12, dict(touch=0.02)),
            ("t0=12 묶음0.4%", 12, dict(tol=0.004)), ("t0=12 묶음0.6%", 12, dict(tol=0.006)),
            ("t0=12 간격1.0%", 12, dict(zgap=0.010)), ("t0=12 간격1.5%", 12, dict(zgap=0.015)),
            ("t0=12 스윙40", 12, dict(keep=40)), ("t0=12 스윙80", 12, dict(keep=80)),
            ("30분봉 t0=16(8h)", 16, dict(tf=30)), ("60분봉 t0=8(8h)", 8, dict(tf=60))]
    cnt = dict(both24=0, both_reg=0, late24_neg=0)
    for nm, t0, chg in var:
        prm = dict(DEF, **chg)
        f = make_R(t0, prm, ZS)
        cells = []
        for reg in (False, True):
            for p in ("early", "late"):
                t = trades(S[p], f, p, reg)
                us = UNIV["사용자78"]
                ds, dtt, _ = diff(t[t.sym.isin(us)], BASE[(p, reg)][BASE[(p, reg)].sym.isin(us)])
                ex, exs = excess(t[t.sym.isin(us)], RE[(p, reg)])
                cells.append((ds, dtt, exs))
        e24, l24, er, lr = cells
        cnt["both24"] += e24[0] > 0 and l24[0] > 0
        cnt["both_reg"] += er[0] > 0 and lr[0] > 0
        cnt["late24_neg"] += l24[0] < 0
        say(f"  {nm:16s} 24h: {e24[0]:+6.2f}(dt{e24[1]:+.1f}) | {l24[0]:+6.2f}(dt{l24[1]:+.1f})   정규: {er[0]:+6.2f}(dt{er[1]:+.1f}, 초과합{er[2]:+.2f}) | "
            f"{lr[0]:+6.2f}(dt{lr[1]:+.1f}, 초과합{lr[2]:+.2f})")
    bex = {k: excess(v[v.sym.isin(UNIV["사용자78"])], RE[k])[1] for k, v in BASE.items()}
    say(f"  기준 무작위초과합(사용자78): early 24h {bex[('early', False)]:+.2f}, late 24h {bex[('late', False)]:+.2f}, "
        f"early 정규 {bex[('early', True)]:+.2f}, late 정규 {bex[('late', True)]:+.2f}")
    say(f"  이웃 {len(var)}개: 24h 두 기간 모두 차이>0 {cnt['both24']}개, late 24h 차이<0 {cnt['late24_neg']}개, 정규 두 기간 모두 >0 {cnt['both_reg']}개")
    say(f"\n끝 {time.time() - t00:.0f}초")


if __name__ == "__main__":
    main()
