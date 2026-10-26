"""지지 구역 '버팀 끝자락'을 시간 대신 상태로 잡는 연구(2026-10-09, 키 volsqueeze).
질문: 지지 구역 위에서 버티는 동안 (1) 변동폭이 줄고(수축) (2) 정규장 거래량이 마른 뒤 처음 늘어나는 봉,
(3) RSI가 같은 구역을 다시 두드릴 때 저점을 높이는 상승 다이버전스 — 이런 상태 신호 뒤에 사면
같은 구역의 첫 닿음 매수·같은 종목 같은 시간대 무작위 매수·지금 이격매수 지표보다 나은가. 오래 두드리면 깨지는지도 본다.

[지지 구역] 사용자 트레이딩뷰 지표(sr_zones_ref.py)와 같다. 40분봉(5m_<티커>.parquet, 미 동부 04:00 기준, 프리·애프터 포함)
  좌우 5봉 피벗 저점·고점(왼쪽 5봉보다 엄격히 낮고/높고 좌우 11봉 최저/최고), 최근 60개, 가격순 0.5% 안은 한 구역(lo~hi, 닿은 수),
  구역 중심끼리 1.2% 미만이면 (닿은 수, 마지막 스윙 봉) 큰 쪽만 남김. 피벗 j는 40분봉 j+5가 끝난 시각부터 쓴다(미래 참조 없음).
[에피소드] 기준봉(early 5분봉 7/14~9/8, late 3분봉 9/10~10/6; stop_lab.load) i에서 보유 구역이 없을 때,
  그 시각까지 확정된 구역 중 띠 위끝 top = hi×1.003 이 직전 종가 c[i-1]보다 낮은 것 중 가장 높은 구역을 '아래 지지'로 보고,
  저가 l[i] ≤ top 이면 닿음(에피소드 시작 s=i). 첫 닿음 매수 = top 지정가, 체결 min(o[s], top).
  봉 종가마다: c < lo×0.995 → 깨짐 / c > hi×1.02 → 이미 오름(끝자락 지남) / 시작부터 3거래일 → 시간 끝. 그 전까지 '버팀'.
  버틴 시간(분) = (d - s + 1) × 봉 길이. 신호는 버팀 중인 봉 d의 종가에서 판단하고 다음 봉 d+1 시가에 산다. 에피소드당 규칙별 1회.
[상태 신호] k = 30분 또는 60분(5분봉 6·12봉, 3분봉 10·20봉), TR = max(h, 직전 c) - min(l, 직전 c)
  수축비 sqz[d] = 최근 k봉 평균 TR / 그 앞 3k봉 평균 TR
  SQZ_now(k,q)  sqz[d] ≤ q
  SQZ_vol(k,q)  d-k..d-1 중 sqz ≤ q 였던 봉이 있고, 봉 d-k..d가 모두 정규장, v[d] ≥ 2×평균 v[d-k..d-1], 양봉(c>o)
  SQZ_box(k,q)  sqz[d-1] ≤ q 이고 c[d] > max h[d-k..d-1](수축 상자 위로 종가 돌파)
  DRY_vol(k,m)  정규장 거래량비 rv = v / 지난 10거래일 같은 시각 평균(최소 5일), 평균 rv[d-k..d-1] ≤ m, 그리고 SQZ_vol과 같은 거래량 증가 양봉
  SQZDRY_vol(k) 수축(q 0.7) + 마름(m 0.8) + 거래량 증가 양봉
  DIV_base(p)   기준봉 국소 저점(좌우 2봉 최저, 봉 j+2 종가에 확정)이 띠 안(l ≤ top)에 두 번 이상: 에피소드 안 앞선 띠 안 저점 중
                가장 낮은 것보다 RSI(기준 RSI: early RSI9, late RSI14)가 p 이상 높은 새 저점 j가 확정되는 봉 d = j+2
  DIV_40(p)     같은 판단을 40분봉 좌우 2봉 저점·40분봉 RSI14(완성 봉 값)로. 확정 = 40분봉 j+2 끝, d = 그 시각 직전 기준봉
  각 신호 × 최소 버팀 Hmin ∈ {0, 120분} (Hmin 전에 난 신호는 버리고 그 뒤 첫 신호).
[매도] 주 매도 = 지금 지표와 같음: 매수 봉 이후 RSI>70 봉 다음 봉 시가 또는 매수 뒤 첫 한국 아침 8시 이후 첫 봉 시가(8시가 데이터 밖이면 제외).
  보조 = 고정 1시간·4시간(벽시계, 다음 봉 시가), 1·3거래일(같은 시각 이후 첫 봉 시가).
[비교 기준] (a) 같은 구역 첫 닿음 지정가 매수(모든 에피소드), (a2) 버틴 시간 H 도달 즉시 매수(시간만), (b) 같은 종목·같은 30분 시간대
  모든 봉 시가 매수의 평균(무작위 기대값; 거래마다 짝지어 초과 = 수익 - 기대값), (c) 지금 이격매수 지표 거래(stop_lab.base_trades).
[고르기] early, 사용자78, 정규장 매수, 주 매도에서 n≥30, 평균 > (a), 무작위 초과 > 0 인 규칙 중 초과 평균 최대. late는 확인만.
[일봉] 1년 정규장 일봉, 같은 구역 계산(일봉 좌우 5봉 피벗, 피벗 j는 j+6일부터), 띠 1%, 깨짐 종가 < lo×0.985, 이미 오름 > hi×1.05, 20거래일 끝.
  수축비 = 5일 평균 TR / 그 앞 15일 평균 TR, 거래량 증가 = v[d] ≥ 1.5×평균 v[d-5..d-1] 양봉, 마름 = 평균 v[d-5..d-1] ≤ 0.7×평균 v[d-25..d-6],
  DIV = 일봉 좌우 2봉 저점·RSI14. Hmin ∈ {0, 3거래일}. 매수 d+1 시가, 매도 N거래일 뒤 시가(N=1·3·5·10, 주 = 5).
  앞 절반(매수일 ~2026-04-06)으로 고르고 뒤 절반(2026-04-07~10-06)으로 확인.
날짜 군집 t = 매수일별 합의 평균 / (표준편차/√일수). 사용: py support/volsqueeze.py (code 폴더에서) → ../results/support/volsqueeze.txt
"""
import os, sys, time, bisect
from datetime import date
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, base_trades, UNIV, PERIODS, RAW
from htf import rma_state

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "support", "volsqueeze.txt"))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
_fh = open(OUT, "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _fh.write(s + "\n"); _fh.flush()


PIV, KEEP, TOL, ZGAP = 5, 60, 0.005, 0.012
BAND, BRK, AWAY, MAXD = 0.003, 0.005, 0.02, 3
XR = 70.0
HMINS = (0, 120)
KMIN = (30, 60)
EXITS = ("p", "1h", "4h", "1d", "3d")
ENAME = {"p": "RSI70/8시", "1h": "1시간", "4h": "4시간", "1d": "1거래일", "3d": "3거래일"}
U78, UALL = UNIV["사용자78"], UNIV["전체137"]


# ------------------------------------------------------------------ 구역(사용자 지표와 같은 계산)
def pivots(h, l, N=PIV):
    """[(봉 j, 가격)] — 같은 봉이면 저점 먼저(지표와 같은 순서)."""
    n = len(l)
    out = []
    if n < 2 * N + 1:
        return out
    lw = sliding_window_view(l, 2 * N + 1).min(axis=1)
    hw = sliding_window_view(h, 2 * N + 1).max(axis=1)
    for j in range(N, n - N):
        if l[j] == lw[j - N] and (l[j - N:j] > l[j]).all():
            out.append((j, l[j]))
        if h[j] == hw[j - N] and (h[j - N:j] < h[j]).all():
            out.append((j, h[j]))
    return out


def zones_from(px):
    order = sorted(range(len(px)), key=lambda k: px[k][1])
    Z = []
    for k in order:
        q, p = px[k]
        if Z and (p - Z[-1][0]) / Z[-1][0] <= TOL:
            Z[-1][1] = p; Z[-1][2] += 1; Z[-1][3] = max(Z[-1][3], q)
        else:
            Z.append([p, p, 1, q])
    keep = []
    for z in sorted(Z, key=lambda z: z[2] * 1e7 + z[3], reverse=True):
        mid = (z[0] + z[1]) / 2
        if all(abs(mid - (k[0] + k[1]) / 2) / ((k[0] + k[1]) / 2) >= ZGAP for k in keep):
            keep.append(z)
    keep.sort(key=lambda z: z[0])
    return keep


def zone_sets(h, l, eff):
    """피벗이 확정될 때마다 구역을 다시 계산. eff(j) = 피벗 봉 j가 쓰일 수 있는 시각(또는 봉 번호). 반환 (시각 배열, [(tops, los, his)])."""
    P = pivots(h, l)
    T, Zs, acc = [], [], []
    k = 0
    while k < len(P):
        j = P[k][0]
        while k < len(P) and P[k][0] == j:
            acc.append(P[k]); k += 1
        Z = zones_from(acc[-KEEP:])
        T.append(eff(j))
        Zs.append(Z)
    return np.asarray(T, dtype=np.int64), Zs


def bars40(sym):
    m5 = pd.read_parquet(os.path.join(RAW, f"5m_{sym}.parquet"))
    b = m5.resample("40min", origin="start_day", offset="4h", label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna(subset=["Close"])
    b = b[b.High > 0]
    st = b.index.as_unit("ns").asi8
    return st, b.High.values.astype(float), b.Low.values.astype(float), b.Close.values.astype(float)


# ------------------------------------------------------------------ 에피소드(버팀) 상태 기계
def episodes(o, l, c, zid, Zs, rank, band, brk, away, maxd):
    """반환 배열: s(닿은 봉), E(버팀 상태로 끝난 마지막 봉), end(끝난 봉, 없으면 n), typ(0 깨짐 1 오름 2 시간 3 데이터 끝), lo, hi, top, px(첫 닿음 체결가)."""
    n = len(o)
    tops_cache = {}
    rows = []
    cur = None
    ol, ll, cl, rk = o.tolist(), l.tolist(), c.tolist(), rank.tolist()
    zl = zid.tolist()
    for i in range(1, n):
        if cur is None:
            z = zl[i]
            if z < 0:
                continue
            T = tops_cache.get(z)
            if T is None:
                Z = Zs[z]
                T = ([zz[1] * (1 + band) for zz in Z], [zz[0] for zz in Z], [zz[1] for zz in Z])
                tops_cache[z] = T
            tops = T[0]
            k = bisect.bisect_left(tops, cl[i - 1]) - 1
            if k < 0:
                continue
            top = tops[k]
            if ll[i] > top:
                continue
            cur = (i, T[1][k], T[2][k], top, min(ol[i], top))
        s, zlo, zhi, top, px = cur
        ci = cl[i]
        if ci < zlo * (1 - brk):
            typ = 0
        elif ci > zhi * (1 + away):
            typ = 1
        elif rk[i] - rk[s] >= maxd:
            typ = 2
        else:
            continue
        rows.append((s, i - 1, i, typ, zlo, zhi, top, px))
        cur = None
    if cur is not None:
        s, zlo, zhi, top, px = cur
        rows.append((s, n - 1, n, 3, zlo, zhi, top, px))
    if not rows:
        return {k: np.zeros(0) for k in ("s", "E", "end", "typ", "lo", "hi", "top", "px")}
    a = np.array(rows, dtype=float)
    return dict(s=a[:, 0].astype(int), E=a[:, 1].astype(int), end=a[:, 2].astype(int), typ=a[:, 3].astype(int),
                lo=a[:, 4], hi=a[:, 5], top=a[:, 6], px=a[:, 7])


# ------------------------------------------------------------------ 공용 계산
def rmean(x, k, lag=0):
    """out[d] = mean x[d-lag-k+1 .. d-lag] (모자라면 nan)."""
    n = len(x)
    cs = np.r_[0.0, np.cumsum(np.nan_to_num(x))]
    hi = np.arange(n) - lag + 1
    lo = hi - k
    out = np.full(n, np.nan)
    ok = lo >= 0
    out[ok] = (cs[hi[ok]] - cs[lo[ok]]) / k
    return out


def next_true(cond):
    n = len(cond)
    idx = np.where(cond, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def first_in_ep(nt, s, E, hb, n):
    """에피소드마다 d ∈ [s+hb-1, E] 중 첫 신호 봉(없으면 -1). 매수 봉 d+1 < n 이어야 함."""
    d0 = np.maximum(s, s + hb - 1)
    out = np.full(len(s), -1)
    ok = d0 <= E
    f = np.where(ok, nt[np.minimum(d0, n - 1)], n)
    good = ok & (f <= E) & (f + 1 < n)
    out[good] = f[good]
    return out


def local_lows(l, w=2):
    n = len(l)
    m = np.zeros(n, bool)
    if n >= 2 * w + 1:
        mn = sliding_window_view(l, 2 * w + 1).min(axis=1)
        m[w:n - w] = l[w:n - w] <= mn
    return m


def div_events(ep, lows_idx, lowv, rsiv, conf_d, s_key):
    """에피소드마다 띠 안 저점 사건 [(d, rsi 차이)] — 앞선 띠 안 저점 중 가장 낮은 것 대비 RSI 차이. conf_d = 확정되는 결정 봉."""
    out = []
    S, E, TOP = ep["s"], ep["E"], ep["top"]
    for e in range(len(S)):
        a = np.searchsorted(s_key, S[e], side="left")
        evs = []
        best = None
        for q in range(a, len(lows_idx)):
            d = conf_d[q]
            if d > E[e]:
                break
            if lowv[q] > TOP[e] or np.isnan(rsiv[q]):
                continue
            if best is not None:
                evs.append((d, rsiv[q] - best[1]))
            if best is None or lowv[q] < best[0]:
                best = (lowv[q], rsiv[q])
        out.append(evs)
    return out


# ------------------------------------------------------------------ 분봉: 종목별 준비
def tod_rv(v, reg, dayi, mod):
    d = pd.DataFrame({"day": dayi, "mod": mod, "v": np.where(reg, v, np.nan)})
    tab = d.groupby(["day", "mod"]).v.last().unstack()
    base = tab.shift(1).rolling(10, min_periods=5).mean().stack()
    key = pd.MultiIndex.from_arrays([dayi, mod])
    b = base.reindex(key).values
    with np.errstate(divide="ignore", invalid="ignore"):
        rv = np.where(reg & (b > 0), v / b, np.nan)
    return rv


def exits_intraday(A):
    o, r, tk, dead, mod = A["o"], A["r"], A["tk"], A["dead"], A["mod"]
    n = len(o)
    nr = next_true(np.nan_to_num(r) > XR)
    j8 = np.searchsorted(tk, dead, side="left")
    xp = np.minimum(nr + 1, j8)
    xp = np.where(j8 < n, xp, n)
    X = {"p": xp}
    for name, hrs in (("1h", 1), ("4h", 4)):
        X[name] = np.searchsorted(tk, tk + hrs * 3600 * 10**9, side="left")
    dayi = np.array([d.toordinal() for d in A["day"]])
    ud, rank = np.unique(dayi, return_inverse=True)
    key = rank.astype(np.int64) * 1440 + mod
    for name, N in (("1d", 1), ("3d", 3)):
        X[name] = np.searchsorted(key, (rank + N).astype(np.int64) * 1440 + mod, side="left")
    return X, rank, dayi


def rets(o, X, i, px):
    out = {}
    n = len(o)
    for e in EXITS:
        x = X[e][i]
        ok = x < n
        r = np.full(len(i), np.nan)
        r[ok] = o[x[ok]] / px[ok] - 1
        out[e] = r
    return out


def prep_period(period):
    S = load(period)
    mins = PERIODS[period][0]
    lo_d, hi_d = PERIODS[period][3], PERIODS[period][4]
    trades = []       # dict 행 모음(규칙·기준)
    rand_rows = []
    nep = 0
    t0 = time.time()
    for sym, A in S.items():
        o, h, l, c, v, reg = A["o"], A["h"], A["l"], A["c"], A["v"], A["reg"]
        n = len(o)
        tk, mod = A["tk"], A["mod"]
        X, rank, dayi = exits_intraday(A)
        inper = np.array([lo_d <= d <= hi_d for d in A["day"]])
        # 40분봉 구역
        st40, h40, l40, c40 = bars40(sym)
        step = 40 * 60 * 10**9
        ZT, Zs = zone_sets(h40, l40, lambda j: st40[min(j + PIV, len(st40) - 1)] + step if j + PIV < len(st40) else 2**62)
        zid = np.searchsorted(ZT, tk, side="right") - 1
        ep = episodes(o, l, c, zid, Zs, rank, BAND, BRK, AWAY, MAXD)
        if len(ep["s"]) == 0:
            continue
        # 매수 봉이 기간 안인 에피소드만 쓰되, 신호 매수 봉도 기간 안이어야 함(아래에서 거름)
        nep += int(inper[ep["s"]].sum())
        ses = A["ses"]
        # ---- 무작위 기대값(같은 종목·30분 시간대, 기간 안 모든 봉 시가 매수)
        ii = np.nonzero(inper & (np.arange(n) >= 1))[0]
        R = rets(o, X, ii, o[ii])
        rdf = pd.DataFrame({"b": mod[ii] // 30, **{e: R[e] for e in EXITS}})
        rtab = rdf.groupby("b").mean()

        def add(rule, d_sig, epi, extra=None, entry_px=None, entry_i=None):
            """d_sig: 결정 봉(매수 = d+1 시가). entry_px/entry_i 가 있으면 그대로(첫 닿음)."""
            if entry_i is None:
                ei = d_sig + 1
                px = o[ei]
            else:
                ei, px = entry_i, entry_px
            keep = inper[ei]
            if not keep.any():
                return
            ei, px, epi2 = ei[keep], px[keep], epi[keep]
            Rr = rets(o, X, ei, px)
            b = mod[ei] // 30
            q = rtab.reindex(b)
            row = dict(rule=rule, per=period, sym=sym, day=dayi[ei], ses=ses[ei], ei=ei,
                       hold=(ei - ep["s"][epi2]) * mins, typ=ep["typ"][epi2])
            for e in EXITS:
                row["r_" + e] = Rr[e]
                row["q_" + e] = q[e].values
            if extra is not None:
                row["x"] = extra[keep]
            trades.append(pd.DataFrame(row))

        E_, S_ = ep["E"], ep["s"]
        allep = np.arange(len(S_))
        # (a) 첫 닿음
        add("A_첫닿음", None, allep, entry_px=ep["px"], entry_i=S_)
        # (a2) 버틴 시간 H 도달 즉시 매수(시간만) + 결과(오름/깨짐) 단조성
        for H in (0, 15, 30, 60, 120, 240, 480, 960):
            hb = max(1, int(np.ceil(H / mins))) if H > 0 else 1
            d = S_ + hb - 1
            ok = (d <= E_) & (d + 1 < n)
            add(f"T_{H}", d[ok], allep[ok])
        # ---- 상태 특징
        cp = np.r_[c[0], c[:-1]]
        tr = np.maximum(h, cp) - np.minimum(l, cp)
        rv = tod_rv(v, reg, dayi, mod)
        green = c > o
        for km in KMIN:
            k = km // mins
            with np.errstate(divide="ignore", invalid="ignore"):
                sqz = rmean(tr, k) / rmean(tr, 3 * k, lag=k)
            sqz_min_prev = pd.Series(sqz).rolling(k, min_periods=k).min().shift(1).values
            vprev = rmean(v, k, lag=1)
            regall = rmean(reg.astype(float), k + 1) == 1.0
            with np.errstate(invalid="ignore"):
                volup = regall & (vprev > 0) & (v >= 2 * vprev) & green
                dry = pd.Series(rv).rolling(k, min_periods=k).mean().shift(1).values   # 정규장 아닌 봉이 끼면 nan → 마름 아님
                hprev = pd.Series(h).rolling(k, min_periods=k).max().shift(1).values
                sq1 = np.r_[np.nan, sqz[:-1]]
            conds = {}
            for q in (0.5, 0.7):
                with np.errstate(invalid="ignore"):
                    conds[f"SQZ_now_k{km}_q{q}"] = sqz <= q
                    conds[f"SQZ_vol_k{km}_q{q}"] = (sqz_min_prev <= q) & volup
                    conds[f"SQZ_box_k{km}_q{q}"] = (sq1 <= q) & (c > hprev)
            for m in (0.5, 0.8):
                with np.errstate(invalid="ignore"):
                    conds[f"DRY_vol_k{km}_m{m}"] = (dry <= m) & volup
            with np.errstate(invalid="ignore"):
                conds[f"SQZDRY_vol_k{km}"] = (sqz_min_prev <= 0.7) & (dry <= 0.8) & volup
            for name, cond in conds.items():
                nt = next_true(np.asarray(cond, bool))
                for H in HMINS:
                    hb = max(1, int(np.ceil(H / mins)))
                    f = first_in_ep(nt, S_, E_, hb, n)
                    ok = f >= 0
                    add(f"{name}_H{H}", f[ok], allep[ok])
            if km == 60:
                # 상태 단조성: 수축비·마름 구간별(에피소드마다 그 구간에 처음 든 봉)
                for nm, val, edges in (("BIN_sqz", sqz, [0, .5, .7, 1, 1.5, np.inf]),
                                       ("BIN_dry", dry, [0, .5, .8, 1.2, 2, np.inf])):
                    b = np.digitize(val, edges) - 1
                    b = np.where(np.isnan(val), -1, b)
                    for bi in range(len(edges) - 1):
                        nt = next_true(b == bi)
                        f = first_in_ep(nt, S_, E_, 1, n)
                        ok = f >= 0
                        add(f"{nm}_{bi}", f[ok], allep[ok])
        # ---- 다이버전스: 기준봉
        ll = local_lows(l, 2)
        jj = np.nonzero(ll)[0]
        conf_d = jj + 2
        evs = div_events(ep, jj, l[jj], A["r"][jj], conf_d, jj)
        # 40분봉
        ll40 = local_lows(l40, 2)
        j40 = np.nonzero(ll40)[0]
        _, _, r40 = rma_state(c40, 14)
        conf_t = np.where(j40 + 2 < len(st40), st40[np.minimum(j40 + 2, len(st40) - 1)] + step, 2**62)
        d40 = np.searchsorted(tk, conf_t, side="left") - 1          # 확정 시각 직전 기준봉(그 종가에서 판단)
        st40_end = st40[j40] + step
        s_key40 = np.searchsorted(tk, st40_end, side="left")         # 40분봉이 끝나는 기준봉 위치(에피소드 시작 이후 봉만)
        evs40 = div_events(ep, j40, l40[j40], r40[j40], d40, s_key40)
        for src, EV in (("DIV_base", evs), ("DIV_40", evs40)):
            for p in (3, 6):
                for H in HMINS:
                    hb = max(1, int(np.ceil(H / mins)))
                    ds, es = [], []
                    for e, lst in enumerate(EV):
                        dmin = S_[e] + hb - 1
                        for d, dif in lst:
                            if d >= dmin and dif >= p and d + 1 < n:
                                ds.append(d); es.append(e); break
                    if ds:
                        add(f"{src}_p{p}_H{H}", np.array(ds), np.array(es))
            # 단조성: RSI 차이 구간(에피소드마다 구간별 첫 사건)
            edges = [-np.inf, -5, 0, 3, 6, 10, np.inf]
            for bi in range(len(edges) - 1):
                ds, es, xs = [], [], []
                for e, lst in enumerate(EV):
                    for d, dif in lst:
                        if edges[bi] <= dif < edges[bi + 1] and d + 1 < n:
                            ds.append(d); es.append(e); xs.append(dif); break
                if ds:
                    add(f"BIN_{src}_{bi}", np.array(ds), np.array(es), extra=np.array(xs))
    T = pd.concat(trades, ignore_index=True)
    T["u78"] = T.sym.isin(U78)
    say(f"[{period}] 종목 {len(S)} · 기간 안 에피소드 {nep} · 계산 {time.time() - t0:.0f}초")
    return T


# ------------------------------------------------------------------ 통계
def stat(x, e="p"):
    x = x[~np.isnan(x["r_" + e].values)]
    n = len(x)
    if n == 0:
        return dict(n=0, avg=np.nan, win=np.nan, worst=np.nan, sum=0.0, t=np.nan, rand=np.nan, ex=np.nan, tex=np.nan)
    r = x["r_" + e]
    ex = r - x["q_" + e]

    def tday(s):
        g = s.groupby(x.day.values).sum()
        sd = g.std(ddof=1)
        return g.mean() / (sd / np.sqrt(len(g))) if len(g) > 1 and sd > 0 else np.nan
    return dict(n=n, avg=r.mean(), win=(r > 0).mean(), worst=r.min(), sum=r.sum(), t=tday(r),
                rand=x["q_" + e].mean(), ex=ex.mean(), tex=tday(ex.fillna(0)))


_G = {}


def cell(T, rule, univ, reg, e="p"):
    if id(T) not in _G:
        _G[id(T)] = dict(tuple(T.groupby("rule")))
    x = _G[id(T)].get(rule)
    if x is None:
        return stat(T.iloc[:0], e)
    if univ == "사용자78":
        x = x[x.u78]
    if reg:
        x = x[x.ses == "정규"]
    return stat(x, e)


def fmt(d, keys=("n", "avg", "win", "worst", "sum", "t", "rand", "ex", "tex")):
    f = {"n": "{:>5.0f}", "avg": "{:+.2%}", "win": "{:.0%}", "worst": "{:+.1%}", "sum": "{:+.2f}", "t": "{:+.2f}",
         "rand": "{:+.2%}", "ex": "{:+.2%}", "tex": "{:+.2f}"}
    return " ".join(("   n/a" if (d[k] != d[k]) else f[k].format(d[k])).rjust(7) for k in keys)


HDR = "      n     평균   승률     최악      합   t(날)  무작위  초과(b)  초과t"


def base_cell(period, univ, reg):
    b = base_trades(period, reg)
    b = b[b.sym.isin(U78 if univ == "사용자78" else UALL)]
    days = b.day
    g = b.ret.groupby(days).sum()
    t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan
    return dict(n=len(b), avg=b.ret.mean(), win=(b.ret > 0).mean(), worst=b.ret.min(), sum=b.ret.sum(), t=t)


# ------------------------------------------------------------------ 일봉
DBAND, DBRK, DAWAY, DMAXD = 0.01, 0.015, 0.05, 20
DEX = (1, 3, 5, 10)
SPLIT = date(2026, 4, 7)


def daily_all():
    rows = []
    syms = sorted(UALL)
    for sym in syms:
        p = os.path.join(RAW, f"1d_{sym}.parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p).dropna(subset=["Open", "High", "Low", "Close"])
        d = d[d.High > 0]
        o, h, l, c, v = (d[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
        days = np.array(pd.to_datetime(d.index).date)
        n = len(o)
        if n < 60:
            continue
        ZT, Zs = zone_sets(h, l, lambda j: j + PIV + 1)       # 피벗 j는 j+6일(오른쪽 5일 종가 확정 뒤)부터
        zid = np.searchsorted(ZT, np.arange(n), side="right") - 1
        ep = episodes(o, l, c, zid, Zs, np.arange(n), DBAND, DBRK, DAWAY, DMAXD)
        if len(ep["s"]) == 0:
            continue
        S_, E_ = ep["s"], ep["E"]
        half = np.where(days < SPLIT, 0, 1)
        # 무작위: 같은 종목·같은 절반 모든 날 시가 매수
        rnd = {}
        for N in DEX:
            r = np.full(n, np.nan)
            r[:n - N] = o[N:] / o[:n - N] - 1
            rnd[N] = [np.nanmean(r[(half == hh) & (np.arange(n) >= 30)]) for hh in (0, 1)]

        def add(rule, ei, px, epi):
            ok = ei < n
            ei, px, epi = ei[ok], px[ok], epi[ok]
            if not len(ei):
                return
            row = dict(rule=rule, sym=sym, day=days[ei], half=half[ei], ei=ei, hold=ei - S_[epi])
            for N in DEX:
                x = ei + N
                r = np.full(len(ei), np.nan)
                g = x < n
                r[g] = o[x[g]] / px[g] - 1
                row[f"r_{N}"] = r
                row[f"q_{N}"] = np.array([rnd[N][hh] for hh in half[ei]])
            rows.append(pd.DataFrame(row))
        allep = np.arange(len(S_))
        add("A_첫닿음", S_, ep["px"], allep)
        for H in (0, 1, 2, 3, 5, 10):
            dd = S_ + H                                     # 닿은 날 + H일 종가까지 버팀 → 다음 날 시가
            ok = dd <= E_
            add(f"T_{H}", dd[ok] + 1, o[np.minimum(dd[ok] + 1, n - 1)], allep[ok])
        cp = np.r_[c[0], c[:-1]]
        tr = np.maximum(h, cp) - np.minimum(l, cp)
        with np.errstate(divide="ignore", invalid="ignore"):
            sqz = rmean(tr, 5) / rmean(tr, 15, lag=5)
            sqmin = pd.Series(sqz).rolling(5, min_periods=5).min().shift(1).values
            vprev = rmean(v, 5, lag=1)
            volup = (vprev > 0) & (v >= 1.5 * vprev) & (c > o)
            dry = rmean(v, 5, lag=1) / rmean(v, 20, lag=6) <= 0.7
        conds = {}
        for q in (0.6, 0.8):
            with np.errstate(invalid="ignore"):
                conds[f"D_SQZ_now_q{q}"] = sqz <= q
                conds[f"D_SQZ_vol_q{q}"] = (sqmin <= q) & volup
        with np.errstate(invalid="ignore"):
            conds["D_DRY_vol"] = dry & volup
            conds["D_SQZDRY_vol"] = (sqmin <= 0.8) & dry & volup
        for name, cond in conds.items():
            nt = next_true(np.asarray(cond, bool))
            for H in (0, 3):
                f = first_in_ep(nt, S_, E_, H + 1, n)
                ok = f >= 0
                add(f"{name}_H{H}", f[ok] + 1, o[f[ok] + 1], allep[ok])
        # 상태 단조성(수축비)
        edges = [0, .5, .7, 1, 1.5, np.inf]
        b = np.where(np.isnan(sqz), -1, np.digitize(sqz, edges) - 1)
        for bi in range(len(edges) - 1):
            f = first_in_ep(next_true(b == bi), S_, E_, 1, n)
            ok = f >= 0
            add(f"DBIN_sqz_{bi}", f[ok] + 1, o[f[ok] + 1], allep[ok])
        _, _, r14 = rma_state(c, 14)
        jj = np.nonzero(local_lows(l, 2))[0]
        EV = div_events(ep, jj, l[jj], r14[jj], jj + 2, jj)
        for p_ in (3, 6):
            for H in (0, 3):
                ds, es = [], []
                for e, lst in enumerate(EV):
                    for dd, dif in lst:
                        if dd >= S_[e] + H and dif >= p_ and dd + 1 < n:
                            ds.append(dd); es.append(e); break
                if ds:
                    ds = np.array(ds)
                    add(f"D_DIV_p{p_}_H{H}", ds + 1, o[ds + 1], np.array(es))
        edges = [-np.inf, -5, 0, 3, 6, 10, np.inf]
        for bi in range(len(edges) - 1):
            ds, es = [], []
            for e, lst in enumerate(EV):
                for dd, dif in lst:
                    if edges[bi] <= dif < edges[bi + 1] and dd + 1 < n:
                        ds.append(dd); es.append(e); break
            if ds:
                ds = np.array(ds)
                add(f"DBIN_div_{bi}", ds + 1, o[ds + 1], np.array(es))
        # 에피소드 결과(일봉)
        rows.append(pd.DataFrame(dict(rule="EP", sym=sym, day=days[S_], half=half[S_], ei=S_, hold=E_ - S_ + 1,
                                      typ=ep["typ"], **{f"r_{N}": np.nan for N in DEX}, **{f"q_{N}": np.nan for N in DEX})))
    D = pd.concat(rows, ignore_index=True)
    D["u78"] = D.sym.isin(U78)
    return D


def dstat(x, N=5):
    x = x[~np.isnan(x[f"r_{N}"].values)]
    n = len(x)
    if n == 0:
        return dict(n=0, avg=np.nan, win=np.nan, worst=np.nan, sum=0.0, t=np.nan, rand=np.nan, ex=np.nan, tex=np.nan)
    r = x[f"r_{N}"]
    ex = r - x[f"q_{N}"]

    def tday(s):
        g = s.groupby(x.day.values).sum()
        sd = g.std(ddof=1)
        return g.mean() / (sd / np.sqrt(len(g))) if len(g) > 1 and sd > 0 else np.nan
    return dict(n=n, avg=r.mean(), win=(r > 0).mean(), worst=r.min(), sum=r.sum(), t=tday(r),
                rand=x[f"q_{N}"].mean(), ex=ex.mean(), tex=tday(ex))


def dcell(D, rule, univ, hh, N=5):
    if id(D) not in _G:
        _G[id(D)] = dict(tuple(D.groupby("rule")))
    x = _G[id(D)].get(rule)
    if x is None:
        return dstat(D.iloc[:0], N)
    x = x[x.half == hh]
    if univ == "사용자78":
        x = x[x.u78]
    return dstat(x, N)


# ------------------------------------------------------------------ 실행
if __name__ == "__main__":
    t0 = time.time()
    say("버팀 끝자락 상태 신호(변동성 수축·거래량·RSI 다이버전스) 연구 — 2026-10-09, 데이터 끝 2026-10-06 19:59 ET")
    say("구역: 40분봉 좌우5 피벗·0.5% 묶음·1.2% 간격·최근 60개 / 닿음 띠 hi×1.003 / 깨짐 종가<lo×0.995 / 이미 오름 종가>hi×1.02 / 3거래일 끝")
    TT = {p: prep_period(p) for p in ("early", "late")}
    PN = {"early": "고르기 early 7/14~9/8 5분봉", "late": "확인 late 9/10~10/6 3분봉"}

    # ---------------- 1. 에피소드 개요·버틴 시간 단조성
    say("\n== 1. 버틴 시간 단조성: 에피소드가 H분 버틴(종가가 구역 아래로 안 깨지고 +2% 위로도 안 감) 순간 다음 봉 시가 매수, 주 매도(RSI70/8시) ==")
    say("   '오름%' = 그 시점까지 버틴 에피소드 중 결국 +2% 위로 끝난 비율, '깨짐%' = 구역 아래로 끝난 비율(반대 가설 확인)")
    for p in ("early", "late"):
        T = TT[p]
        for un in ("사용자78", "전체137"):
            for reg in (False, True):
                say(f"-- {PN[p]} · {un} · {'정규장 매수만' if reg else '24시간'}")
                say("  H분   " + HDR + "   오름%  깨짐%")
                st = cell(T, "A_첫닿음", un, reg)
                x = T[(T.rule == "A_첫닿음") & ((T.u78) if un == "사용자78" else True)]
                if reg:
                    x = x[x.ses == "정규"]
                say(f"  닿음  {fmt(st)}   {(x.typ == 1).mean():5.0%}  {(x.typ == 0).mean():5.0%}")
                for H in (0, 15, 30, 60, 120, 240, 480, 960):
                    st = cell(T, f"T_{H}", un, reg)
                    x = T[(T.rule == f"T_{H}") & ((T.u78) if un == "사용자78" else True)]
                    if reg:
                        x = x[x.ses == "정규"]
                    say(f"  {H:>4}  {fmt(st)}   {(x.typ == 1).mean():5.0%}  {(x.typ == 0).mean():5.0%}")

    # ---------------- 2. 상태 단조성
    say("\n== 2. 상태 단조성(에피소드마다 그 구간에 처음 든 봉 → 다음 봉 시가 매수, 주 매도) ==")
    BINS = {"BIN_sqz": ("60분 수축비(최근 60분 평균 TR / 그 앞 180분)", ["<0.5", "0.5~0.7", "0.7~1", "1~1.5", "≥1.5"]),
            "BIN_dry": ("60분 정규장 거래량비(같은 시각 10일 평균 대비)", ["<0.5", "0.5~0.8", "0.8~1.2", "1.2~2", "≥2"]),
            "BIN_DIV_base": ("기준봉 띠 안 재저점의 RSI 차이(앞선 최저 저점 대비)", ["<-5", "-5~0", "0~3", "3~6", "6~10", "≥10"]),
            "BIN_DIV_40": ("40분봉 띠 안 재저점의 RSI14 차이", ["<-5", "-5~0", "0~3", "3~6", "6~10", "≥10"])}
    for key, (title, labs) in BINS.items():
        say(f"-- {title}")
        for p in ("early", "late"):
            for un in ("사용자78", "전체137"):
                for reg in (False, True):
                    say(f"   {p} {un} {'정규' if reg else '24h'}")
                    for bi, lb in enumerate(labs):
                        st = cell(TT[p], f"{key}_{bi}", un, reg)
                        if st["n"]:
                            say(f"     {lb:>8} {fmt(st)}")

    # ---------------- 3. 규칙 표
    rules = sorted({r for r in TT["early"].rule.unique() if r.split("_")[0] in ("SQZ", "DRY", "SQZDRY", "DIV")})
    say(f"\n== 3. 규칙 {len(rules)}개(분봉) — 주 매도, 정규장 매수만 / 24시간. 기준(a) 첫 닿음 평균, (b) 무작위 짝 초과 ==")
    res = []
    for r in rules:
        for p in ("early", "late"):
            for un in ("사용자78", "전체137"):
                for reg in (True, False):
                    st = cell(TT[p], r, un, reg)
                    a = cell(TT[p], "A_첫닿음", un, reg)
                    res.append(dict(rule=r, per=p, un=un, reg=reg, **st, a=a["avg"],
                                    beat=(st["n"] >= 30) and (st["avg"] > a["avg"]) and (st["ex"] > 0)))
    R = pd.DataFrame(res)
    for un in ("사용자78", "전체137"):
        for reg in (True, False):
            say(f"-- {un} · {'정규장 매수만' if reg else '24시간'}   (early | late)  각: n 평균 초과(b) 첫닿음대비")
            a_e = cell(TT["early"], "A_첫닿음", un, reg)["avg"]
            a_l = cell(TT["late"], "A_첫닿음", un, reg)["avg"]
            say(f"   첫 닿음 평균 early {a_e:+.2%} / late {a_l:+.2%}")
            sub = R[(R.un == un) & (R.reg == reg)]
            for r in rules:
                e = sub[(sub.rule == r) & (sub.per == "early")].iloc[0]
                l_ = sub[(sub.rule == r) & (sub.per == "late")].iloc[0]
                mk = "◎" if (e.beat and l_.beat) else ("○" if e.beat else " ")
                say(f"   {mk} {r:<24} {e.n:>5.0f} {e.avg:+.2%} {e.ex:+.2%} {e.avg - e.a:+.2%} | {l_.n:>5.0f} {l_.avg:+.2%} {l_.ex:+.2%} {l_.avg - l_.a:+.2%}")
    both = {}
    for un in ("사용자78", "전체137"):
        for reg in (True, False):
            sub = R[(R.un == un) & (R.reg == reg)]
            be = set(sub[(sub.per == "early") & sub.beat].rule)
            bl = set(sub[(sub.per == "late") & sub.beat].rule)
            both[(un, reg)] = (len(be), len(bl), len(be & bl), sorted(be & bl))
            say(f"   기준(a·b) 둘 다 이긴 규칙: {un} {'정규' if reg else '24h'} — early {len(be)} / late {len(bl)} / 두 기간 모두 {len(be & bl)} {sorted(be & bl)}")

    # ---------------- 4. 최선안(early 사용자78 정규장)
    sub = R[(R.un == "사용자78") & R.reg & (R.per == "early") & (R.n >= 30)]
    cand = sub[sub.beat]
    pick = (cand if len(cand) else sub).sort_values("ex", ascending=False).iloc[0].rule
    say(f"\n== 4. 최선안(고르기: early·사용자78·정규장·주 매도에서 n≥30, 첫닿음보다 높고 무작위 초과>0 중 초과 최대): {pick}"
        f"{'' if len(cand) else '  ※ 조건 통과 규칙 없음 → 초과 최대를 참고로'} ==")
    best_cells = []
    for p in ("early", "late"):
        for un in ("사용자78", "전체137"):
            for reg in (True, False):
                say(f"-- {PN[p]} · {un} · {'정규장 매수만' if reg else '24시간'}")
                say("   매도      규칙/기준   " + HDR)
                for e in EXITS:
                    for lab, rr in (("최선안", pick), ("(a)첫닿음", "A_첫닿음"), ("(a2)120분", "T_120")):
                        st = cell(TT[p], rr, un, reg, e)
                        say(f"   {ENAME[e]:<8} {lab:<10} {fmt(st)}")
                        if lab == "최선안" and e == "p":
                            best_cells.append(dict(period=p, universe=un, scope="정규장" if reg else "24시간", **st))
                b = base_cell(p, un, reg)
                say(f"   (c) 지금 이격매수 지표: n {b['n']} 평균 {b['avg']:+.2%} 승률 {b['win']:.0%} 최악 {b['worst']:+.1%} 합 {b['sum']:+.2f} t {b['t']:+.2f}")

    # ---------------- 5. 일봉
    say("\n== 5. 일봉 1년(앞 절반 2025-10-07~2026-04-06 고르기 / 뒤 절반 2026-04-07~10-06 확인), 매도 5거래일 뒤 시가 ==")
    D = daily_all()
    ep = D[D.rule == "EP"]
    for hh in (0, 1):
        x = ep[ep.half == hh]
        say(f"   {'앞' if hh == 0 else '뒤'} 절반 에피소드 {len(x)} · 오름 {(x.typ == 1).mean():.0%} 깨짐 {(x.typ == 0).mean():.0%} 시간끝 {(x.typ == 2).mean():.0%}")
    for un in ("사용자78", "전체137"):
        say(f"-- 버틴 날수 단조성 · {un}  (앞 | 뒤)")
        for rr in ["A_첫닿음"] + [f"T_{H}" for H in (0, 1, 2, 3, 5, 10)]:
            a, b = dcell(D, rr, un, 0), dcell(D, rr, un, 1)
            say(f"   {rr:<10} {fmt(a, ('n', 'avg', 'win', 'ex', 'tex'))} | {fmt(b, ('n', 'avg', 'win', 'ex', 'tex'))}")
        say(f"-- 수축비 구간 · {un}")
        for bi, lb in enumerate(["<0.5", "0.5~0.7", "0.7~1", "1~1.5", "≥1.5"]):
            a, b = dcell(D, f"DBIN_sqz_{bi}", un, 0), dcell(D, f"DBIN_sqz_{bi}", un, 1)
            say(f"   {lb:<10} {fmt(a, ('n', 'avg', 'win', 'ex', 'tex'))} | {fmt(b, ('n', 'avg', 'win', 'ex', 'tex'))}")
        say(f"-- RSI 다이버전스 차이 구간 · {un}")
        for bi, lb in enumerate(["<-5", "-5~0", "0~3", "3~6", "6~10", "≥10"]):
            a, b = dcell(D, f"DBIN_div_{bi}", un, 0), dcell(D, f"DBIN_div_{bi}", un, 1)
            say(f"   {lb:<10} {fmt(a, ('n', 'avg', 'win', 'ex', 'tex'))} | {fmt(b, ('n', 'avg', 'win', 'ex', 'tex'))}")
    drules = sorted(r for r in D.rule.unique() if r.startswith("D_"))
    say(f"-- 일봉 규칙 {len(drules)}개 (앞 | 뒤) n 평균 초과(b) 첫닿음대비")
    dres = []
    for un in ("사용자78", "전체137"):
        say(f"   [{un}] 첫 닿음 앞 {dcell(D, 'A_첫닿음', un, 0)['avg']:+.2%} / 뒤 {dcell(D, 'A_첫닿음', un, 1)['avg']:+.2%}")
        for r in drules:
            row = []
            for hh in (0, 1):
                st = dcell(D, r, un, hh)
                a = dcell(D, "A_첫닿음", un, hh)
                bt = st["n"] >= 30 and st["avg"] > a["avg"] and st["ex"] > 0
                dres.append(dict(rule=r, un=un, half=hh, beat=bt, **st))
                row.append((st, a, bt))
            mk = "◎" if (row[0][2] and row[1][2]) else ("○" if row[0][2] else " ")
            say(f"   {mk} {r:<18} " + " | ".join(f"{s['n']:>5.0f} {s['avg']:+.2%} {s['ex']:+.2%} {s['avg'] - a['avg']:+.2%}" for s, a, _ in row))
    DR = pd.DataFrame(dres)
    for un in ("사용자78", "전체137"):
        s0 = set(DR[(DR.un == un) & (DR.half == 0) & DR.beat].rule)
        s1 = set(DR[(DR.un == un) & (DR.half == 1) & DR.beat].rule)
        say(f"   일봉 기준 둘 다 이긴 규칙 {un}: 앞 {len(s0)} / 뒤 {len(s1)} / 둘 다 {len(s0 & s1)} {sorted(s0 & s1)}")
        both[("일봉", un)] = (len(s0), len(s1), len(s0 & s1), sorted(s0 & s1))
    dsub = DR[(DR.un == "사용자78") & (DR.half == 0) & (DR.n >= 30)]
    dc = dsub[dsub.beat]
    dpick = (dc if len(dc) else dsub).sort_values("ex", ascending=False).iloc[0].rule
    say(f"-- 일봉 최선안(앞 절반·사용자78·5일에서 고름): {dpick}")
    for un in ("사용자78", "전체137"):
        for hh in (0, 1):
            for N in DEX:
                a, b, c_ = dcell(D, dpick, un, hh, N), dcell(D, "A_첫닿음", un, hh, N), dcell(D, "T_3", un, hh, N)
                say(f"   {un} {'앞' if hh == 0 else '뒤'} {N:>2}일  최선 {fmt(a)}\n{'':>22}첫닿음 {fmt(b)}\n{'':>22}3일버팀 {fmt(c_)}")
    n_rules = len(rules) + len(drules)
    say(f"\n시험 규칙 수: 분봉 {len(rules)} + 일봉 {len(drules)} = {n_rules}  (구간 단조성 표는 서술용으로 세지 않음)")
    say(f"걸린 시간 {time.time() - t0:.0f}초")
