"""지지 구역 '버틴 시간' × 지금 이격매수 결합 연구(2026-10-09, 키 dipcombo).

질문: 지금 지표(기준봉 EMA×0.98 지정가, RSI>70 다음 봉 시가 또는 다음 한국 아침 8시 정리, 전날 +7% 다음 날 정규장 매수 안 함)의
거래를 매수 순간의 40분봉 지지 구역 상태로 나누면 성과가 달라지는가, 그리고 '오래 버틴 구역 근처에서만 산다'·'오래 버틴 구역이면
매수선을 구역 상단으로 올린다' 같은 규칙이 지금 지표보다 나은가.

구역(사용자 트레이딩뷰 dip_scalper 지지·저항, code/sr_zones_ref.py와 같은 계산):
  40분봉 = 5분봉(프리·애프터 포함)을 미 동부 04:00 기준 40분으로 묶음(하루 24봉).
  스윙 = 좌우 5봉 피벗 저점(l[p]가 p±5 최저, 왼쪽 5봉은 엄격히 큼)·고점(대칭). 같은 봉이면 저점 먼저.
  피벗 p는 p+5번째 40분봉이 끝난 뒤(진행 중인 40분봉 k에서 p+5 < k)에만 쓴다. 최근 스윙 60개를 가격순 정렬,
  구역 첫 가격 대비 0.5% 안이면 한 구역(lo=첫 가격, hi=마지막 가격, cnt=스윙 수), 구역 중심끼리 1.2% 미만이면
  (cnt, 마지막 스윙 봉) 큰 쪽만 남김. 지지·저항 구분 없이 남은 구역 전부를 쓴다.
상태(기준봉 i에서 판단, 미래 참조 없음 — 40분봉은 i가 속한 40분봉 직전까지 완성된 것만):
  '근처' = 가격 P가 [lo×(1-1%), hi×(1+1%)] 안인 구역 중 중심이 가장 가까운 것. 없으면 '공중'.
  버틴 시간 T(40분봉 수) = 완성 40분봉을 k-1부터 거꾸로 보며 종가가 [lo×0.995, hi×1.02] 안인 연속 구간에서
    저가가 hi×1.01 이하로 처음 닿은 봉부터 k-1까지의 봉 수. 구간이 종가 lo×0.995 미만(깨짐) 또는 hi×1.02 초과(떠남)에서
    끊긴다. k-1 종가가 lo×0.995 미만이면 '깨짐', 그 밖에 T=0이면 '첫 닿음'. 1봉=40분, 24봉=프리·정규·애프터 하루.
  매수 조건 판단에는 지정가 lvl = EMA[i-1]×0.98(필터형) 또는 직전 종가 c[i-1](구역 상단형)을 P로 쓴다.
체결: 봉 i 저가가 지정가 이하면 min(시가, 지정가). 매도: RSI[j-1]>70 → 봉 j 시가, 아니면 매수 뒤 첫 한국 08시 이후 첫 봉 시가.
  고정 보유는 매수 봉 시작 + 1·4·24·72시간 이후 첫 봉 시가(자료 끝이면 제외).
비교 기준: (a) 같은 구역 '버팀 무시'판(t0=0), (b) 같은 종목·같은 세션·같은 ET 시각(시) 모든 봉 시가 진입의 기대값(같은 매도),
  (c) 지금 이격매수(base_trades와 같은 상태 기계, 같은 날짜 창).
기간: early 5분봉(EMA36·RSI9) — 구역 예열 때문에 7/28~9/8만 평가(7/14 시작 자료), late 3분봉(EMA60·RSI14) 9/10~10/6.
고르기: early 사용자78·24시간 '수익 합 차이(규칙-기준)' 최대, 단 early 사용자78 정규장만 차이 ≥ 0. late는 확인만.
검사: 기준 거래가 stop_lab.base_trades와 같은지, 무작위 봉에서 5분봉을 그 40분봉 시작 전까지 잘라 다시 계산해도 상태가 같은지(미래 참조),
  TEM 10/6 09:57 구역이 sr_zones_ref.py 출력과 같은지(별도 확인: 71.47~81.83 지지 7개 일치).
사용: py support/dipcombo.py   (code 폴더에서) → ../results/support/dipcombo.txt
"""
import os, sys, time
from datetime import date
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, base_trades, UNIV, PERIODS, RAW, GAP, X

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "support", "dipcombo.txt"))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
TF, NP, KEEP, TOL, ZGAP = 40, 5, 60, 0.005, 0.012
NEAR, BRK, UPL, BELOW = 0.01, 0.005, 0.02, 0.03
CAP = 240                                            # 버팀 거꾸로 보기 최대 40분봉 수(10거래일)
WIN = {"early": (date(2026, 7, 28), date(2026, 9, 8)), "late": (date(2026, 9, 10), date(2026, 10, 6))}
WNAME = {"early": "고르기 early 7/28~9/8(5분봉)", "late": "확인 late 9/10~10/6(3분봉)"}
HOLD = {"1h": 1, "4h": 4, "1d": 24, "3d": 72}
BUCK = ["공중", "깨짐", "T0 첫닿음", "T1-2(40~80분)", "T3-5(2~3.3h)", "T6-11(4~7.3h)", "T12-23(8~15h)", "T24+(하루+)"]
_fh = open(OUT, "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _fh.write(s + "\n"); _fh.flush()


# ------------------------------------------------------------------ 40분봉 구역
def bars40(sym):
    m5 = pd.read_parquet(os.path.join(RAW, f"5m_{sym}.parquet"))
    b = m5.resample(f"{TF}min", origin="start_day", offset="4h", label="left", closed="left").agg(
        {"High": "max", "Low": "min", "Close": "last"}).dropna(subset=["Close"])
    b = b[b.High > 0]
    return b.index.as_unit("ns").asi8, b.High.values.astype(float), b.Low.values.astype(float), b.Close.values.astype(float)


def swings(h, l):
    n = len(l)
    if n < 2 * NP + 1:
        return np.zeros(0, np.int64), np.zeros(0)
    lmin, hmax = swv(l, 2 * NP + 1).min(1), swv(h, 2 * NP + 1).max(1)
    lleft, hleft = swv(l, NP).min(1), swv(h, NP).max(1)
    p = np.arange(NP, n - NP)
    isl = (l[p] == lmin[p - NP]) & (lleft[p - NP] > l[p])
    ish = (h[p] == hmax[p - NP]) & (hleft[p - NP] < h[p])
    sp = np.r_[p[isl], p[ish]]
    sv = np.r_[l[p[isl]], h[p[ish]]]
    typ = np.r_[np.zeros(isl.sum()), np.ones(ish.sum())]
    o = np.lexsort((typ, sp))
    return sp[o], sv[o]


def make_zones(px, bi):
    Z = []
    for k in np.argsort(px, kind="stable"):
        p, q = px[k], bi[k]
        if Z and (p - Z[-1][0]) / Z[-1][0] <= TOL:
            Z[-1][1] = p; Z[-1][2] += 1; Z[-1][3] = max(Z[-1][3], q)
        else:
            Z.append([p, p, 1, q])
    keep = []
    for z in sorted(Z, key=lambda z: z[2] * 1e7 + z[3], reverse=True):
        mid = (z[0] + z[1]) / 2
        if all(abs(mid - (y[0] + y[1]) / 2) / ((y[0] + y[1]) / 2) >= ZGAP for y in keep):
            keep.append(z)
    keep.sort(key=lambda z: (z[0] + z[1]) / 2)
    a = np.array([z[:3] for z in keep], float).reshape(-1, 3)
    return a[:, 0], a[:, 1], a[:, 2]


def hold_T(k, lo, hi, C, L):
    """반환 (T, k-1 상태: 0 띠 안 / 1 위로 떠남 / 2 아래로 깨짐)."""
    a, b, th = lo * (1 - BRK), hi * (1 + UPL), hi * (1 + NEAR)
    j, first, last = k - 1, -1, 0
    while j >= 0 and k - 1 - j < CAP:
        c = C[j]
        if c < a or c > b:
            if j == k - 1:
                last = 2 if c < a else 1
            break
        if L[j] <= th:
            first = j
        j -= 1
    return (k - first if first >= 0 else 0), last


def zone_state(P, nsb, zc, kk, C, L, dn, up, memo):
    """가격 P(봉마다)의 [lo(1-dn), hi(1+up)] 안 가장 가까운 구역과 버틴 시간."""
    n = len(P)
    zlo, zhi, zcnt = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    T, last = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    ok = np.isfinite(P)
    order = np.flatnonzero(ok)
    order = order[np.argsort(nsb[order], kind="stable")]
    if not len(order):
        return dict(lo=zlo, hi=zhi, cnt=zcnt, T=T, last=last, f=np.zeros(n, bool))
    vals = nsb[order]
    cut = np.flatnonzero(np.r_[True, vals[1:] != vals[:-1], True])
    for s, e in zip(cut[:-1], cut[1:]):
        lo, hi, cnt = zc[vals[s]]
        if not len(lo):
            continue
        ii = order[s:e]
        p = P[ii][:, None]
        inb = (p >= lo * (1 - dn)) & (p <= hi * (1 + up))
        d = np.where(inb, np.abs(p - (lo + hi) / 2), np.inf)
        j = d.argmin(1)
        g = np.isfinite(d[np.arange(len(ii)), j])
        ii, j = ii[g], j[g]
        zlo[ii], zhi[ii], zcnt[ii] = lo[j], hi[j], cnt[j]
    f = np.isfinite(zlo)
    for i in np.flatnonzero(f):
        key = (kk[i], zlo[i], zhi[i])
        v = memo.get(key)
        if v is None:
            v = memo[key] = hold_T(kk[i], zlo[i], zhi[i], C, L)
        T[i], last[i] = v
    return dict(lo=zlo, hi=zhi, cnt=zcnt, T=T, last=last, f=f)


def zone_below(P, nsb, zc, mincnt=1):
    """P가 공중일 때 쓸 '아래 구역': hi가 [P×(1-3%), P/(1+1%)) 안인 구역 중 가장 높은 hi."""
    n = len(P)
    out = np.full(n, np.nan)
    ok = np.flatnonzero(np.isfinite(P))
    order = ok[np.argsort(nsb[ok], kind="stable")]
    if not len(order):
        return out
    vals = nsb[order]
    cut = np.flatnonzero(np.r_[True, vals[1:] != vals[:-1], True])
    for s, e in zip(cut[:-1], cut[1:]):
        lo, hi, cnt = zc[vals[s]]
        m = cnt >= mincnt
        if not m.any():
            continue
        hi = hi[m]
        ii = order[s:e]
        p = P[ii][:, None]
        good = (hi >= p * (1 - BELOW)) & (hi < p / (1 + NEAR))
        v = np.where(good, hi, -np.inf).max(1)
        g = np.isfinite(v)
        out[ii[g]] = v[g]
    return out


# ------------------------------------------------------------------ 종목별 미리 계산
def prep_sym(A):
    o, l, c, r, ema, tk = A["o"], A["l"], A["c"], A["r"], A["ema"], A["tk"]
    n = len(o)
    hs, H40, L40, C40 = bars40(A["sym"])
    sp, sv = swings(H40, L40)
    K = len(hs)
    step = TF * 60 * 10 ** 9
    anchor = (A["idx"].normalize() + pd.Timedelta(hours=4)).as_unit("ns").asi8
    cur = anchor + ((tk - anchor) // step) * step
    kk = np.searchsorted(hs, cur, side="left")                      # 완성 40분봉 = 0..kk-1
    ns_k = np.searchsorted(sp, np.arange(K + 1) - NP - 1, side="right")  # p+5 < k 인 스윙 수
    nsb = ns_k[np.minimum(kk, K)]
    zc = {}
    for v in np.unique(nsb):
        a = max(0, v - KEEP)
        zc[v] = make_zones(sv[a:v], sp[a:v]) if v > 0 else (np.zeros(0), np.zeros(0), np.zeros(0))
    lvl = np.r_[np.nan, ema[:-1]] * (1 - GAP)
    cp = np.r_[np.nan, c[:-1]]
    memo = {}
    Pd = dict(lvl=lvl, nsw=nsb)
    Pd["s1"] = zone_state(lvl, nsb, zc, kk, C40, L40, NEAR, NEAR, memo)
    Pd["s05"] = zone_state(lvl, nsb, zc, kk, C40, L40, 0.005, 0.005, memo)
    Pd["s2"] = zone_state(lvl, nsb, zc, kk, C40, L40, 0.02, 0.02, memo)
    Pd["sc"] = zone_state(cp, nsb, zc, kk, C40, L40, BRK, UPL, memo)
    Pd["zb1"] = zone_below(lvl, nsb, zc, 1)
    Pd["zb2"] = zone_below(lvl, nsb, zc, 2)
    s1 = Pd["s1"]
    b = np.zeros(n, np.int64)
    T = s1["T"]
    b[s1["f"] & (s1["last"] == 2)] = 1
    b[s1["f"] & (s1["last"] != 2) & (T == 0)] = 2
    for code, (a_, z_) in zip(range(3, 8), [(1, 2), (3, 5), (6, 11), (12, 23), (24, 10 ** 9)]):
        b[s1["f"] & (T >= a_) & (T <= z_)] = code
    Pd["buck"] = b
    # 매도 미리 계산
    hi_r = np.flatnonzero(r > X)
    pos = np.searchsorted(hi_r, np.arange(n), side="left")
    nr = np.where(pos < len(hi_r), hi_r[np.minimum(pos, len(hi_r) - 1)], n + 5)
    e1 = nr + 1
    e2 = np.searchsorted(tk, A["dead"], side="left")
    Pd["xi"] = np.minimum(e1, e2)
    Pd["xrsi"] = e1 <= e2
    for hk, hh in HOLD.items():
        Pd["h" + hk] = np.searchsorted(tk, tk + hh * 3600 * 10 ** 9, side="left")
    # 무작위 진입(봉 i 시가에 사서 같은 매도) 수익
    xi = Pd["xi"]
    okx = xi < n
    rr = np.full(n, np.nan)
    rr[okx] = o[xi[okx]] / o[okx] - 1
    Pd["rand"] = rr
    for hk in HOLD:
        j = Pd["h" + hk]
        v = np.full(n, np.nan)
        g = j < n
        v[g] = o[j[g]] / o[g] - 1
        Pd["rand" + hk] = v
    return Pd


# ------------------------------------------------------------------ 상태 기계(Pine과 같음, 매수 지정가만 바꿈)
def sim_fast(A, Pd, lim, reg_only):
    o, l, n = A["o"], A["l"], len(A["o"])
    with np.errstate(invalid="ignore"):
        ok = (l <= lim) & ~A["hot7"]
    if reg_only:
        ok &= A["reg"]
    ok[:61] = False
    cand = np.flatnonzero(ok)
    xi_a, xr = Pd["xi"], Pd["xrsi"]
    out, i = [], 61
    while True:
        p = np.searchsorted(cand, i)
        if p >= len(cand):
            break
        ei = cand[p]
        xi = xi_a[ei]
        if xi >= n:
            break
        out.append((ei, xi, min(o[ei], lim[ei]), "RSI" if xr[ei] else "8시"))
        i = xi + 1
    return out


def run_rule(S, PD, limfn, period, reg_only):
    lo, hi = WIN[period]
    rows = []
    for s, A in S.items():
        Pd = PD[s]
        lim = limfn(Pd)
        o, n = A["o"], len(A["o"])
        for ei, xi, ent, why in sim_fast(A, Pd, lim, reg_only):
            dy = A["day"][ei]
            if not (lo <= dy <= hi):
                continue
            rec = [s, dy, A["ses"][ei], int(A["mod"][ei] // 60), o[xi] / ent - 1, why, ei, ent, int(Pd["buck"][ei]),
                   int(Pd["sc"]["T"][ei]), float(Pd["sc"]["cnt"][ei]), float(Pd["s1"]["cnt"][ei])]
            for hk in HOLD:
                j = Pd["h" + hk][ei]
                rec.append(o[j] / ent - 1 if j < n else np.nan)
            rows.append(rec)
    cols = ["sym", "day", "ses", "hr", "ret", "why", "ei", "ent", "buck", "Tc", "cntc", "cnt1"] + ["r" + h for h in HOLD]
    return pd.DataFrame(rows, columns=cols)


def rand_table(S, PD, period, reg_only):
    """(종목, 세션, ET 시) 별 무작위 진입 기대 수익·승률. 전날 +7% 정규장 봉은 제외(규칙도 못 사는 봉)."""
    lo, hi = WIN[period]
    parts = []
    for s, A in S.items():
        Pd = PD[s]
        day = A["day"]
        m = np.array([lo <= d <= hi for d in day]) & ~A["hot7"]
        if reg_only:
            m &= A["reg"]
        m[:61] = False
        df = pd.DataFrame({"sym": s, "ses": A["ses"][m], "hr": A["mod"][m] // 60, "rr": Pd["rand"][m],
                           **{"rr" + h: Pd["rand" + h][m] for h in HOLD}})
        parts.append(df)
    d = pd.concat(parts)
    g = d.groupby(["sym", "ses", "hr"])
    t = g.mean()
    t["win"] = g.rr.apply(lambda x: (x.dropna() > 0).mean())
    return t


def attach_rand(t, RT):
    if not len(t):
        return t
    k = pd.MultiIndex.from_arrays([t.sym, t.ses, t.hr])
    R = RT.reindex(k)
    t = t.copy()
    t["rnd"] = R.rr.values
    t["rndwin"] = R.win.values
    for h in HOLD:
        t["rnd" + h] = R["rr" + h].values
    return t


# ------------------------------------------------------------------ 지표
def ct(x, day):
    """날짜 군집 t(평균 / 날짜 군집 표준오차)."""
    x = pd.Series(np.asarray(x, float)).reset_index(drop=True)
    day = pd.Series(np.asarray(day)).reset_index(drop=True)
    g = x.notna()
    x, day = x[g], day[g]
    n = len(x)
    if n < 3:
        return np.nan
    e = (x - x.mean()).groupby(day.values).sum()
    se = np.sqrt((e ** 2).sum()) / n
    return x.mean() / se if se > 0 else np.nan


def summ(t, b=None):
    if not len(t):
        return dict(n=0, avg=np.nan, win=np.nan, worst=np.nan, sum=0.0, t=np.nan, rnd=np.nan, ex=np.nan, ext=np.nan, exs=0.0,
                    dsum=(-b.ret.sum() if b is not None else np.nan), dt=np.nan, **{h: np.nan for h in HOLD})
    d = dict(n=len(t), avg=t.ret.mean(), win=(t.ret > 0).mean(), worst=t.ret.min(), sum=t.ret.sum(), t=ct(t.ret, t.day),
             rnd=t.rnd.mean(), ex=(t.ret - t.rnd).mean(), ext=ct(t.ret - t.rnd, t.day), exs=(t.ret - t.rnd).sum(),
             **{h: t["r" + h].mean() for h in HOLD})
    if b is not None:
        days = sorted(set(b.day) | set(t.day))
        dd = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
        sd = dd.std(ddof=1)
        d["dsum"] = t.ret.sum() - b.ret.sum()
        d["dt"] = dd.mean() / (sd / np.sqrt(len(dd))) if sd > 0 else 0.0
    return d


def fmt_row(lbl, d, wide=True):
    f = lambda v, s: "   n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else s.format(v)
    s = (f"{lbl:16s} {d['n']:5d} {f(d['avg'], '{:+.2%}'):>7s} {f(d['win'], '{:.0%}'):>4s} {f(d['worst'], '{:+.1%}'):>6s} "
         f"{d['sum']:+7.2f} {f(d['t'], '{:+.1f}'):>5s}")
    if wide:
        s += (f" | 1h {f(d['1h'], '{:+.2%}'):>7s} 4h {f(d['4h'], '{:+.2%}'):>7s} 1d {f(d['1d'], '{:+.2%}'):>7s} "
              f"3d {f(d['3d'], '{:+.2%}'):>7s} | 무작위 {f(d['rnd'], '{:+.2%}'):>7s} 초과 {f(d['ex'], '{:+.2%}'):>7s} t{f(d['ext'], '{:+.1f}'):>5s}")
    return s


HDR = f"{'상태':16s} {'건수':>5s} {'평균':>7s} {'승률':>4s} {'최악':>6s} {'합':>7s} {'t':>5s} | 고정 보유 평균(1h·4h·1d·3d) | 같은 종목·세션·시각 무작위 기대·초과"


def lookahead_check(S, PD, period, syms=("NVDA", "TEM", "IREN", "SNDK"), m=60, seed=7):
    """봉 i가 속한 40분봉 시작 전까지로 5분봉을 잘라 구역·버틴 시간을 처음부터 다시 계산해 미리 계산값과 비교."""
    rng = np.random.default_rng(seed)
    lo_d, hi_d = WIN[period]
    tot = bad = 0
    step = TF * 60 * 10 ** 9
    for s in syms:
        A, Pd = S[period].get(s), PD[period].get(s)
        if A is None:
            continue
        m5 = pd.read_parquet(os.path.join(RAW, f"5m_{s}.parquet"))
        inwin = np.array([lo_d <= d <= hi_d for d in A["day"]])
        pool_z = np.flatnonzero(inwin & Pd["s1"]["f"])
        pool_n = np.flatnonzero(inwin & ~Pd["s1"]["f"] & np.isfinite(Pd["lvl"]))
        pick = np.r_[rng.choice(pool_z, min(m, len(pool_z)), replace=False), rng.choice(pool_n, min(m // 3, len(pool_n)), replace=False)]
        for i in pick:
            tk = A["tk"][i]
            anchor = (A["idx"][i].normalize() + pd.Timedelta(hours=4)).value
            cur = anchor + ((tk - anchor) // step) * step
            m5t = m5[m5.index.as_unit("ns").asi8 < cur]
            b = m5t.resample(f"{TF}min", origin="start_day", offset="4h", label="left", closed="left").agg(
                {"High": "max", "Low": "min", "Close": "last"}).dropna(subset=["Close"])
            b = b[b.High > 0]
            hh, ll, cc = b.High.values.astype(float), b.Low.values.astype(float), b.Close.values.astype(float)
            sp, sv = swings(hh, ll)
            zlo, zhi, zcnt = make_zones(sv[-KEEP:], sp[-KEEP:]) if len(sp) else (np.zeros(0),) * 3
            P = Pd["lvl"][i]
            inb = (P >= zlo * (1 - NEAR)) & (P <= zhi * (1 + NEAR))
            if inb.any():
                j = np.where(inb, np.abs(P - (zlo + zhi) / 2), np.inf).argmin()
                T, last = hold_T(len(cc), zlo[j], zhi[j], cc, ll)
                same = (Pd["s1"]["f"][i] and np.isclose(Pd["s1"]["lo"][i], zlo[j]) and np.isclose(Pd["s1"]["hi"][i], zhi[j])
                        and Pd["s1"]["T"][i] == T and Pd["s1"]["last"][i] == last)
            else:
                same = not Pd["s1"]["f"][i]
            tot += 1
            bad += 0 if same else 1
    return tot, bad


def boot_diff(t, mask, B=4000, seed=3):
    """날짜 단위 부트스트랩: mean(ret | mask) - mean(ret | ~mask). 반환 (차이, 2.5%, 97.5%, P(차이<=0))."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"day": t.day.values, "r": t.ret.values, "m": np.asarray(mask)})
    g1 = d[d.m].groupby("day").r.agg(["sum", "count"])
    g0 = d[~d.m].groupby("day").r.agg(["sum", "count"])
    days = sorted(set(d.day))
    s1, n1 = g1["sum"].reindex(days, fill_value=0).values, g1["count"].reindex(days, fill_value=0).values
    s0, n0 = g0["sum"].reindex(days, fill_value=0).values, g0["count"].reindex(days, fill_value=0).values
    w = rng.multinomial(len(days), np.ones(len(days)) / len(days), size=B)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = (w @ s1) / (w @ n1) - (w @ s0) / (w @ n0)
    diff = diff[np.isfinite(diff)]
    est = s1.sum() / max(n1.sum(), 1) - s0.sum() / max(n0.sum(), 1)
    return est, np.quantile(diff, 0.025), np.quantile(diff, 0.975), (diff <= 0).mean()


# ------------------------------------------------------------------ 규칙(최대 100개)
def L(Pd):
    return Pd["lvl"]


def held(Pd, t0, mincnt=1):
    s = Pd["sc"]
    base = s["f"] & (s["last"] != 2) & (s["cnt"] >= mincnt)
    return base & (s["T"] >= t0) if t0 > 0 else base


def filt(cond):
    return lambda Pd: np.where(cond(Pd), Pd["lvl"], np.nan)


def reloc(t0, mincnt=1, skip_air=False):
    def f(Pd):
        h = held(Pd, t0, mincnt)
        other = np.where(Pd["s1"]["f"], Pd["lvl"], np.nan) if skip_air else Pd["lvl"]
        return np.where(h, np.fmax(Pd["lvl"], Pd["sc"]["hi"]), other)
    return f


def zone_only(t0):
    return lambda Pd: np.where(held(Pd, t0), Pd["sc"]["hi"], np.nan)


def down(key):
    return lambda Pd: np.where(~Pd["s1"]["f"] & np.isfinite(Pd[key]), Pd[key], Pd["lvl"])


RULES = {
    "기준(지금 지표)": L,
    "F 근처±1% 아무 T": filt(lambda P: P["s1"]["f"]),
    "F 근처±0.5%": filt(lambda P: P["s05"]["f"]),
    "F 근처±2%": filt(lambda P: P["s2"]["f"]),
    "F T0 첫닿음만": filt(lambda P: P["s1"]["f"] & (P["s1"]["T"] == 0) & (P["s1"]["last"] != 2)),
    "F T>=1": filt(lambda P: P["s1"]["f"] & (P["s1"]["T"] >= 1)),
    "F T>=3": filt(lambda P: P["s1"]["f"] & (P["s1"]["T"] >= 3)),
    "F T>=6": filt(lambda P: P["s1"]["f"] & (P["s1"]["T"] >= 6)),
    "F T>=12": filt(lambda P: P["s1"]["f"] & (P["s1"]["T"] >= 12)),
    "F T>=24": filt(lambda P: P["s1"]["f"] & (P["s1"]["T"] >= 24)),
    "F T1-5": filt(lambda P: P["s1"]["f"] & (P["s1"]["T"] >= 1) & (P["s1"]["T"] <= 5)),
    "F T3-11": filt(lambda P: P["s1"]["f"] & (P["s1"]["T"] >= 3) & (P["s1"]["T"] <= 11)),
    "F T6-23": filt(lambda P: P["s1"]["f"] & (P["s1"]["T"] >= 6) & (P["s1"]["T"] <= 23)),
    "F T>=3 (±2%)": filt(lambda P: P["s2"]["f"] & (P["s2"]["T"] >= 3)),
    "F cnt>=2": filt(lambda P: P["s1"]["f"] & (P["s1"]["cnt"] >= 2)),
    "F cnt>=2 & T>=3": filt(lambda P: P["s1"]["f"] & (P["s1"]["cnt"] >= 2) & (P["s1"]["T"] >= 3)),
    "X 공중만": filt(lambda P: ~P["s1"]["f"]),
    "X 깨짐 빼기": filt(lambda P: ~(P["s1"]["f"] & (P["s1"]["last"] == 2))),
    "X T>=12 빼기": filt(lambda P: ~(P["s1"]["f"] & (P["s1"]["T"] >= 12))),
    "X T>=24 빼기": filt(lambda P: ~(P["s1"]["f"] & (P["s1"]["T"] >= 24))),
    "X T0 빼기": filt(lambda P: ~(P["s1"]["f"] & (P["s1"]["T"] == 0) & (P["s1"]["last"] != 2))),
    "R 상단 t0=0(버팀무시)": reloc(0),
    "R 상단 T>=1": reloc(1),
    "R 상단 T>=3": reloc(3),
    "R 상단 T>=6": reloc(6),
    "R 상단 T>=12": reloc(12),
    "R 상단 T>=24": reloc(24),
    "R 상단 T>=3 cnt>=2": reloc(3, 2),
    "R 상단 T>=12 cnt>=2": reloc(12, 2),
    "RF 상단 T>=3 +공중빼기": reloc(3, 1, True),
    "RF 상단 T>=12 +공중빼기": reloc(12, 1, True),
    "D 공중이면 아래구역 상단": down("zb1"),
    "D 아래구역(cnt>=2) 상단": down("zb2"),
    "Z 구역만 t0=0(버팀무시)": zone_only(0),
    "Z 구역만 T>=3": zone_only(3),
    "Z 구역만 T>=6": zone_only(6),
    "Z 구역만 T>=12": zone_only(12),
    "Z 구역만 T>=24": zone_only(24),
}
FAMILY_A = {  # 각 규칙의 '버팀 무시' 짝(비교 기준 a)
    **{k: "F 근처±1% 아무 T" for k in RULES if k.startswith("F T") or k.startswith("F cnt")},
    "F T>=3 (±2%)": "F 근처±2%",
    **{k: "R 상단 t0=0(버팀무시)" for k in RULES if k.startswith("R") or k.startswith("RF")},
    **{k: "Z 구역만 t0=0(버팀무시)" for k in RULES if k.startswith("Z")},
}


def main():
    t0 = time.time()
    say("지지 구역 '버틴 시간' × 지금 이격매수 결합(dipcombo) —", time.strftime("%Y-%m-%d %H:%M"))
    say(__doc__.split("사용:")[0].strip())
    S, PD, BASE, RT = {}, {}, {}, {}
    for period in ("early", "late"):
        S[period] = load(period)
        PD[period] = {s: prep_sym(A) for s, A in S[period].items()}
        for reg in (False, True):
            RT[(period, reg)] = rand_table(S[period], PD[period], period, reg)
        say(f"[준비] {period} 종목 {len(S[period])}  {time.time() - t0:.0f}초")
    # 예열 확인: early 첫날들 스윙 수
    A = S["early"]["NVDA"]; Pd = PD["early"]["NVDA"]
    for dd in (date(2026, 7, 21), date(2026, 7, 28)):
        i = int(np.searchsorted(np.array([x.toordinal() for x in A["day"]]), dd.toordinal()))
        say(f"[예열] NVDA {dd} 시점 확정 스윙 {Pd['nsw'][i]}개(60개 쓰면 꽉 참)")

    # 0) 빠른 상태 기계 = stop_lab.base_trades 확인
    for period in ("early", "late"):
        for reg in (False, True):
            mine = run_rule(S[period], PD[period], L, period, reg)
            ref = base_trades(period, reg)
            lo, hi = WIN[period]
            ref = ref[(ref.day >= lo) & (ref.day <= hi)]
            ok = len(mine) == len(ref) and np.allclose(np.sort(mine.ret.values), np.sort(ref.ret.values))
            say(f"[검산] {period} {'정규장만' if reg else '24시간'} 기준 거래 {len(mine)} vs stop_lab {len(ref)} 일치={ok}")

    for period in ("early", "late"):
        tot, bad = lookahead_check(S, PD, period)
        say(f"[미래참조 검사] {period}: 봉 {tot}개에서 5분봉을 그 40분봉 시작 전까지 잘라 다시 계산 → 다른 값 {bad}개")

    # 1) 모든 규칙 실행
    RES = {}
    for name, fn in RULES.items():
        for period in ("early", "late"):
            for reg in (False, True):
                t = attach_rand(run_rule(S[period], PD[period], fn, period, reg), RT[(period, reg)])
                RES[(name, period, reg)] = t
    say(f"[실행] 규칙 {len(RULES) - 1}개 × 2기간 × 2범위  {time.time() - t0:.0f}초")

    def cell(name, period, reg, un):
        us = UNIV[un]
        t = RES[(name, period, reg)]
        b = RES[("기준(지금 지표)", period, reg)]
        return summ(t[t.sym.isin(us)], b[b.sym.isin(us)])

    # 2) 단조성: 기준 거래를 매수 순간 구역 상태로 나눈 표
    say("\n" + "=" * 100)
    say("1) 지금 이격매수 거래를 매수 지정가(EMA×0.98)의 40분봉 구역 상태로 나눔 — 매도 RSI>70/아침 8시, 합=수익률 합")
    for reg in (False, True):
        for un in UNIV:
            for period in ("early", "late"):
                b = RES[("기준(지금 지표)", period, reg)]
                b = b[b.sym.isin(UNIV[un])]
                say(f"\n[{WNAME[period]} · {'정규장만' if reg else '24시간'} · {un}]")
                say(HDR)
                for code, nm in enumerate(BUCK):
                    say(fmt_row(nm, summ(b[b.buck == code])))
                say(fmt_row("전체", summ(b)))
    say("\n[T>=12(8시간+ 버팀) 근처 매수 vs 나머지 기준 거래] 평균 차이, 날짜 부트스트랩 95% 구간, P(차이<=0)")
    for reg in (False, True):
        for un in UNIV:
            for period in ("early", "late"):
                b = RES[("기준(지금 지표)", period, reg)]
                b = b[b.sym.isin(UNIV[un])]
                m = b.buck.values >= 6
                e, a_, z_, pz = boot_diff(b, m)
                say(f"  {WNAME[period]:22s} {'정규장만' if reg else '24시간':5s} {un:6s} T>=12 {m.sum():4d}건 vs {(~m).sum():4d}건  "
                    f"차이 {e:+.2%} [{a_:+.2%}, {z_:+.2%}] P(<=0)={pz:.2f}")
    # 2b) 구역 강도(cnt)
    say("\n[구역 강도] 근처 구역(±1%)의 스윙 수 cnt별, 24시간 사용자78")
    for period in ("early", "late"):
        b = RES[("기준(지금 지표)", period, False)]
        b = b[b.sym.isin(UNIV["사용자78"]) & (b.buck >= 1)]
        say(f"  {WNAME[period]}")
        for nm, m in (("cnt=1", b.cnt1 == 1), ("cnt=2", b.cnt1 == 2), ("cnt>=3", b.cnt1 >= 3)):
            say("  " + fmt_row(nm, summ(b[m]), wide=False))
    # 2c) 구역 상단 지정가(버팀 무시) 거래를 매수 순간 T로 나눔 — '오래 버틸수록 돌아서나'를 구역 매수 자체로
    say("\n[구역 상단 지정가 매수(Z t0=0)] 매수 순간 버틴 시간 T별 — 구역만으로 산 경우의 단조성")
    TB = [(0, 0, "T0"), (1, 2, "T1-2"), (3, 5, "T3-5"), (6, 11, "T6-11"), (12, 23, "T12-23"), (24, 10 ** 9, "T24+")]
    for reg in (False, True):
        for un in UNIV:
            for period in ("early", "late"):
                z = RES[("Z 구역만 t0=0(버팀무시)", period, reg)]
                z = z[z.sym.isin(UNIV[un])]
                say(f"\n  [{WNAME[period]} · {'정규장만' if reg else '24시간'} · {un}]")
                say("  " + HDR)
                for a_, z_, nm in TB:
                    say("  " + fmt_row(nm, summ(z[(z.Tc >= a_) & (z.Tc <= z_)])))

    # 3) 규칙 비교표
    say("\n" + "=" * 100)
    say("2) 규칙 비교 — 차이 = 규칙 수익 합 - 지금 지표 수익 합(같은 날짜 창·종목), dt = 날짜별 합 차이 t")
    rows = []
    for name in RULES:
        for period in ("early", "late"):
            for reg in (False, True):
                for un in UNIV:
                    d = cell(name, period, reg, un)
                    rows.append(dict(rule=name, period=period, reg=reg, un=un, **d))
    R = pd.DataFrame(rows)
    R.to_csv(OUT.replace(".txt", "_grid.csv"), index=False, encoding="utf-8-sig")
    for un in UNIV:
        for reg in (False, True):
            say(f"\n[{un} · {'정규장만' if reg else '24시간'}]  early(고르기) | late(확인)")
            say(f"{'규칙':26s} {'건수':>5s} {'평균':>7s} {'승률':>4s} {'합':>7s} {'차이':>7s} {'dt':>5s} {'초과평균':>7s} {'초과합':>6s} | "
                f"{'건수':>5s} {'평균':>7s} {'승률':>4s} {'합':>7s} {'차이':>7s} {'dt':>5s} {'초과평균':>7s} {'초과합':>6s}")
            for name in RULES:
                s = f"{name:26s}"
                for period in ("early", "late"):
                    d = R[(R.rule == name) & (R.period == period) & (R.reg == reg) & (R.un == un)].iloc[0]
                    s += (f" {d.n:5d} {d.avg:+7.2%} {d.win:4.0%} {d['sum']:+7.2f} {d.dsum:+7.2f} {d["dt"]:+5.1f} {d.ex:+7.2%} {d.exs:+6.2f}"
                          if d.n else f" {0:5d} {'':>7s} {'':>4s} {0:+7.2f} {d.dsum:+7.2f} {'':>5s} {'':>7s} {'':>6s}")
                    s += " |" if period == "early" else ""
                say(s)

    # 4) 고르기·확인
    cand = [k for k in RULES if k != "기준(지금 지표)"]
    def g(name, period, reg, un, col):
        return R[(R.rule == name) & (R.period == period) & (R.reg == reg) & (R.un == un)].iloc[0][col]
    both = {}
    for un in UNIV:
        for reg in (False, True):
            both[(un, reg)] = [k for k in cand if g(k, "early", reg, un, "dsum") > 0 and g(k, "late", reg, un, "dsum") > 0]
    both_avg = [k for k in cand if g(k, "early", False, "사용자78", "avg") > g("기준(지금 지표)", "early", False, "사용자78", "avg")
                and g(k, "late", False, "사용자78", "avg") > g("기준(지금 지표)", "late", False, "사용자78", "avg")]
    elig = [k for k in cand if g(k, "early", True, "사용자78", "dsum") >= 0]
    best = max(elig, key=lambda k: g(k, "early", False, "사용자78", "dsum")) if elig else None
    best_avg = max([k for k in cand if g(k, "early", False, "사용자78", "n") >= 100],
                   key=lambda k: g(k, "early", False, "사용자78", "avg"))
    say("\n" + "=" * 100)
    say(f"3) 시험 규칙 {len(cand)}개(기준 제외).")
    for (un, reg), v in both.items():
        say(f"  두 기간 모두 수익 합이 지금 지표보다 큰 규칙 [{un}·{'정규장만' if reg else '24시간'}]: {len(v)}개 {v}")
    say(f"  두 기간 모두 평균이 지금 지표보다 큰 규칙 [사용자78·24시간]: {len(both_avg)}개 {both_avg}")
    say(f"  고르기 기준(early 사용자78 24시간 합 차이 최대, 정규장만 차이≥0) 후보 {len(elig)}개 → 선택: {best}")
    say(f"  (참고) early 사용자78 24시간 평균 최고(100건 이상): {best_avg}")
    for nm in [best, best_avg]:
        if nm is None:
            continue
        say(f"\n[{nm}] 두 기간 결과 + 비교 기준")
        a_nm = FAMILY_A.get(nm)
        for period in ("early", "late"):
            for reg in (False, True):
                for un in UNIV:
                    d = cell(nm, period, reg, un)
                    bb = cell("기준(지금 지표)", period, reg, un)
                    s = (f"  {WNAME[period]:22s} {'정규장만' if reg else '24시간':5s} {un:6s} 규칙 {d['n']:5d}건 평균 {d['avg']:+.2%} "
                         f"승률 {d['win']:.0%} 최악 {d['worst']:+.1%} 합 {d['sum']:+.2f} t {d['t']:+.1f} | (c)지금 {bb['n']}건 "
                         f"평균 {bb['avg']:+.2%} 합 {bb['sum']:+.2f} 차이 {d['dsum']:+.2f} dt {d['dt']:+.1f} | (b)무작위 평균 "
                         f"{d['rnd']:+.2%} 초과 {d['ex']:+.2%} t {d['ext']:+.1f}")
                    if a_nm:
                        aa = cell(a_nm, period, reg, un)
                        s += f" | (a){a_nm} {aa['n']}건 평균 {aa['avg']:+.2%} 합 {aa['sum']:+.2f}"
                    say(s)
    say(f"\n끝 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
