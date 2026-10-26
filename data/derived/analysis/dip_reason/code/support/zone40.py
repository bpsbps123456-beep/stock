"""40분봉 지지 구역 '버틴 시간' 연구(2026-10-09, 키 zone40).

질문: 가격이 지지 구역에 닿은 뒤 깨지지 않고 오래 버틸수록 위로 돌아서기 쉬운가, 그리고 '버팀의 끝자락'(아직 많이 오르기 전)에
사면 이득인가. 반대로 오래 두드리면 결국 깨지는가.

구역(사용자 트레이딩뷰 지표, sr_zones_ref.py와 같은 계산)
  40분봉: 5m_<티커>.parquet(프리·애프터 포함)를 미 동부 04:00 기준 40분으로 묶음.
  스윙: 좌우 5봉 피벗(저점: 가운데 저가가 11봉 최소이고 왼쪽 5봉보다 엄격히 낮음, 고점은 반대). 피벗 봉 i는 i+5봉이 끝난 시각
        (그 40분봉 시작 + 40분)부터 쓴다. 최근 스윙 60개, 가격순 정렬 뒤 묶음의 첫 가격 대비 0.5% 안은 한 구역,
        구역 중간값끼리 1.2% 미만이면 (닿은 횟수, 마지막 스윙 봉) 큰 쪽만 남김.
버팀(에피소드), 기준봉 = early 5분봉 / late 3분봉
  닿음: 봉 i 시작 때 확정된 구역 중 직전 종가 아래에서 가장 가까운 구역(상단 < c[i-1])의 상단에 저가가 닿음(l[i] <= 상단).
        닿는 순간 구역 경계(하단 lo, 상단 hi)를 고정한다.
  끝:   종가 < lo*(1-x) → '깨짐', 종가 > hi*(1+y) → '위로 벗어남'(돌파). 깨진 봉 종가가 그 아래 다른 구역 안에 있으면
        다음 봉부터 그 구역 에피소드를 바로 시작(버팀 시계는 깨진 봉에서 시작).
  버틴 시간 T: 세션 시계(04:00~20:00 ET만 셈, 밤·주말 제외)로 닿은 봉 시작부터 지금 봉 끝까지.
진입
  (i)  T가 Tth에 처음 도달한 봉이 끝났을 때 아직 안 끝났으면(깨짐·돌파 없음) 다음 봉 시가 매수.
  (ii) 에피소드가 '위로 벗어남'으로 끝났고 그때 T >= Tth이면 다음 봉 시가 매수. 단 시가 > hi*(1+y+1%)면 건너뜀(이미 오름).
  기준 (a) 늘 가장 가까운 지지 구역 상단에 지정가(닿으면 min(시가, 상단)), (b) 같은 종목·같은 30분 시간대 모든 봉 시가 진입의
       평균(같은 매도 규칙), (c) 지금 이격매수 지표(stop_lab.sim).
매도: 지금 지표와 같은 RSI(early 5분봉 RSI9, late 3분봉 RSI14) > 70 다음 봉 시가, 또는 매수 뒤 첫 한국시간 아침 8시 정리.
      참고로 고정 보유 1시간·4시간·1일(960 세션분)·3일(세션 시계).
한 종목에 한 포지션(매도 봉 다음부터 다시 매수). 고르기 = early(7/14~9/8), 확인 = late(9/10~10/6; 원자료가 10/6 19:59 ET에서 끝남).
"""
import os, sys, time
from datetime import date
import numpy as np, pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
from stop_lab import load, sim, RAW, USER78, ALL

OUT = os.path.normpath(os.path.join(CODE, "../results/support/zone40.txt"))
N, KEEP, TOL, ZGAP = 5, 60, 0.005, 0.012
STEP = 40 * 60 * 10**9
RANGE = {"early": (date(2026, 7, 14), date(2026, 9, 8)), "late": (date(2026, 9, 10), date(2026, 10, 6))}
PN = {"early": "고르기 7/14~9/8(5분봉)", "late": "확인 9/10~10/6(3분봉)"}
XS, YS, ZPLUS = (0.0, 0.005, 0.01), (0.005, 0.01, 0.02), 0.01
T_I, T_II = (30, 60, 120, 240, 480), (0, 60, 120, 240, 480)
T_TAB = (0, 30, 60, 120, 240, 480, 960)
HOLDS = {"1h": 60, "4h": 240, "1d": 960, "3d": 2880}
EXITS = ("main",) + tuple(HOLDS)
RSI_X = 70.0
UNIV = {"사용자78": USER78, "전체137": set(ALL)}
RULES = [("i", x, y, t) for x in XS for y in YS for t in T_I] + [("ii", x, y, t) for x in XS for y in YS for t in T_II]


def rname(k):
    f, x, y, t = RULES[k]
    if f == "i":
        return f"(i) 버팀 {t/60:g}h 도달 매수 x{x:.1%} y{y:.1%}"
    return f"(ii) 버팀 {t/60:g}h+ 뒤 +{y:.1%} 돌파 매수(+{y+ZPLUS:.1%} 넘게 오른 시가 제외) x{x:.1%}"


# ---------------------------------------------------------------- 구역
def cluster(px, bi):
    order = np.argsort(px, kind="stable")
    Z = []
    for k in order:
        p, q = px[k], bi[k]
        if Z and (p - Z[-1][0]) / Z[-1][0] <= TOL:
            Z[-1][1] = p; Z[-1][2] += 1; Z[-1][3] = max(Z[-1][3], q)
        else:
            Z.append([p, p, 1, q])
    keep = []
    for z in sorted(Z, key=lambda z: z[2] * 1e7 + z[3], reverse=True):
        mid = (z[0] + z[1]) / 2
        if all(abs(mid - (k[0] + k[1]) / 2) / ((k[0] + k[1]) / 2) >= ZGAP for k in keep):
            keep.append(z)
    keep.sort(key=lambda z: (z[0] + z[1]) / 2)
    return keep


_Z = {}


def zone_sets(s):
    """확정 스윙 수 k마다의 구역 묶음. conf[j] = j번째 스윙을 쓸 수 있게 되는 시각(ns)."""
    if s in _Z:
        return _Z[s]
    m5 = pd.read_parquet(f"{RAW}/5m_{s}.parquet")
    m5 = m5[m5.High > 0]
    b = m5.resample("40min", origin="start_day", offset="4h", label="left", closed="left").agg(
        {"High": "max", "Low": "min", "Close": "last"}).dropna(subset=["Close"])
    hs = b.index.as_unit("ns").asi8
    h, l = b.High.values.astype(float), b.Low.values.astype(float)
    px, bi = [], []
    for i in range(N, len(h) - N):
        if l[i] == l[i - N:i + N + 1].min() and (l[i - N:i] > l[i]).all():
            px.append(l[i]); bi.append(i)
        if h[i] == h[i - N:i + N + 1].max() and (h[i - N:i] < h[i]).all():
            px.append(h[i]); bi.append(i)
    px, bi = np.array(px, float), np.array(bi, np.int64)
    conf = hs[bi + N] + STEP
    sets = [cluster(px[max(0, k - KEEP):k], bi[max(0, k - KEEP):k]) for k in range(len(px) + 1)]
    J = max(1, max(len(z) for z in sets))
    ZLO, ZHI = np.full((len(sets), J), np.inf), np.full((len(sets), J), np.inf)
    ZC = np.zeros((len(sets), J), np.int64)
    for k, zs in enumerate(sets):
        for j, z in enumerate(zs):
            ZLO[k, j], ZHI[k, j], ZC[k, j] = z[0], z[1], z[2]
    _Z[s] = dict(hs=hs, conf=conf, ZLO=ZLO, ZHI=ZHI, ZC=ZC, bi=bi)
    return _Z[s]


# ---------------------------------------------------------------- 에피소드
@njit(cache=True)
def episodes(c, cand, clo, chi, sid, ZLO, ZHI, x, y):
    n = len(c)
    T0 = np.empty(n, np.int64); S = np.empty(n, np.int64); E = np.empty(n, np.int64); K = np.empty(n, np.int64)
    LO = np.empty(n); HI = np.empty(n); TY = np.empty(n, np.int64)
    m = 0
    i = 1
    while i < n:
        if not cand[i]:
            i += 1
            continue
        s, t0, lo, hi, ty = i, i, clo[i], chi[i], 0
        while True:
            lb, ub = lo * (1 - x), hi * (1 + y)
            e, kind = s, 2
            while e < n:
                if c[e] < lb:
                    kind = 0
                    break
                if c[e] > ub:
                    kind = 1
                    break
                e += 1
            T0[m], S[m], E[m], K[m], LO[m], HI[m], TY[m] = t0, s, e, kind, lo, hi, ty
            m += 1
            if kind == 0 and e + 1 < n:
                p, k, best = c[e], sid[e + 1], -1
                for j in range(ZLO.shape[1]):
                    zl, zh = ZLO[k, j], ZHI[k, j]
                    if zl == np.inf:
                        break
                    if zh < lo and zl * (1 - x) <= p and p <= zh:
                        best = j
                if best >= 0:
                    s, t0, lo, hi, ty = e + 1, e, ZLO[k, best], ZHI[k, best], 1
                    continue
            break
        i = e + 1
    return T0[:m], S[:m], E[:m], K[:m], LO[:m], HI[:m], TY[:m]


@njit(cache=True)
def dedup(ei, xi):
    """ei 오름차순. 앞 거래의 매도 봉 이후에만 다시 산다. 매도 봉이 없으면(자료 끝) 이후 전부 버림."""
    keep = np.zeros(len(ei), np.bool_)
    last = -1
    for k in range(len(ei)):
        if ei[k] > last:
            keep[k] = True
            last = xi[k]
    return keep


# ---------------------------------------------------------------- 종목 준비
def prep(A, Z):
    n = len(A["o"])
    tk = A["tk"]
    sid = np.searchsorted(Z["conf"], tk, side="right")             # 봉 시작 때까지 확정된 스윙 수
    # 미래 참조 점검: 쓰는 마지막 스윙의 확정 시각 <= 봉 시작
    used = sid > 0
    assert (Z["conf"][np.maximum(sid - 1, 0)][used] <= tk[used]).all()
    ordn = np.array([d.toordinal() for d in A["day"]])
    _, rank = np.unique(ordn, return_inverse=True)
    clock = rank * 960 + (A["mod"] - 240)
    c, l, o, r = A["c"], A["l"], A["o"], A["r"]
    P = np.r_[np.nan, c[:-1]]
    zh, zl = Z["ZHI"][sid], Z["ZLO"][sid]
    with np.errstate(invalid="ignore"):
        jn = (zh < P[:, None]).sum(1) - 1
    ok = jn >= 0
    jj = np.maximum(jn, 0)
    hi_n = np.where(ok, zh[np.arange(n), jj], np.nan)
    lo_n = np.where(ok, zl[np.arange(n), jj], np.nan)
    cnt_n = np.where(ok, Z["ZC"][sid, jj], 0)
    cand = ok & (l <= hi_n)
    pos = np.flatnonzero(r > RSI_X)
    k = np.searchsorted(pos, np.arange(n), "left")
    nr = np.where(k < len(pos), pos[np.minimum(k, max(len(pos) - 1, 0))] if len(pos) else n, n)
    d8 = np.searchsorted(tk, A["dead"], "left")
    X = {"main": np.minimum(nr + 1, d8)}
    for nm, H in HOLDS.items():
        X[nm] = np.searchsorted(clock, clock + H, "left")
    dayo = ordn
    return dict(n=n, sid=sid, clock=clock, cand=cand, hi_n=hi_n, lo_n=lo_n, cnt_n=cnt_n, X=X, dayo=dayo,
                buck=(A["mod"] // 30).astype(np.int64))


def rets(A, Q, ei, ent):
    """진입 봉 ei·가격 ent → 매도 규칙별 수익(자료 끝이면 NaN)."""
    o, n = A["o"], Q["n"]
    out = {}
    for nm in EXITS:
        xi = Q["X"][nm][ei]
        okx = xi < n
        out[nm] = np.where(okx, o[np.minimum(xi, n - 1)] / ent - 1, np.nan)
    return out


def rand_table(A, Q, lo, hi):
    """같은 종목·같은 30분 시간대·기간 안 모든 봉 시가 진입의 평균 수익(매도 규칙별), 길이 48 배열."""
    o, n = A["o"], Q["n"]
    inp = (Q["dayo"] >= lo.toordinal()) & (Q["dayo"] <= hi.toordinal())
    idx = np.flatnonzero(inp)
    T = {}
    for nm in EXITS:
        xi = Q["X"][nm][idx]
        okx = xi < n
        rr = o[np.minimum(xi, n - 1)] / o[idx] - 1
        b = Q["buck"][idx]
        s = np.bincount(b[okx], weights=rr[okx], minlength=48)
        c = np.bincount(b[okx], minlength=48)
        with np.errstate(invalid="ignore", divide="ignore"):
            T[nm] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return T


# ---------------------------------------------------------------- 실행
def run_period(period):
    lo_d, hi_d = RANGE[period]
    S = load(period)
    trades, eps, brk = [], [], []
    for s in ALL:
        if s not in S:
            continue
        A = S[s]
        Z = zone_sets(s)
        Q = prep(A, Z)
        RT = rand_table(A, Q, lo_d, hi_d)
        n, o, c, mins = Q["n"], A["o"], A["c"], A["mins"]
        clock, reg, dayo = Q["clock"], A["reg"], Q["dayo"]
        inper = lambda ei: (dayo[ei] >= lo_d.toordinal()) & (dayo[ei] <= hi_d.toordinal())

        def emit(rule, ei, ent, extra=None):
            if not len(ei):
                return
            order = np.argsort(ei, kind="stable")
            ei, ent = ei[order], ent[order]
            xm = Q["X"]["main"][ei]
            for scope in ("24h", "reg"):
                m = np.ones(len(ei), bool) if scope == "24h" else reg[ei]
                e2, p2, x2 = ei[m], ent[m], xm[m]
                kp = dedup(e2, x2) & inper(e2)
                e2, p2 = e2[kp], p2[kp]
                if not len(e2):
                    continue
                R = rets(A, Q, e2, p2)
                d = dict(rule=rule, scope=scope, sym=s, ei=e2, day=dayo[e2], reg=reg[e2])
                for nm in EXITS:
                    d[nm] = R[nm]
                    d["rb_" + nm] = RT[nm][Q["buck"][e2]]
                trades.append(pd.DataFrame(d))

        # (a) 가장 가까운 지지 구역 상단 지정가
        ia = np.flatnonzero(Q["cand"])
        emit("A", ia, np.minimum(o[ia], Q["hi_n"][ia]))
        # (c) 지금 이격매수 지표(전날 +7% 다음 날 정규장 제외 포함) — 24h/정규장만 각각 시뮬레이션
        for scope, ro in (("24h", False), ("reg", True)):
            tr = sim(A, None, True, ro)
            if tr:
                ei = np.array([t[0] for t in tr], np.int64)
                ent = np.array([t[2] for t in tr])
                kp = inper(ei)
                ei, ent = ei[kp], ent[kp]
                if len(ei):
                    R = rets(A, Q, ei, ent)
                    d = dict(rule="C", scope=scope, sym=s, ei=ei, day=dayo[ei], reg=reg[ei])
                    for nm in EXITS:
                        d[nm] = R[nm]
                        d["rb_" + nm] = RT[nm][Q["buck"][ei]]
                    trades.append(pd.DataFrame(d))

        for xv in XS:
            for yv in YS:
                T0, Ss, E, K, LO, HI, TY = episodes(c, Q["cand"], Q["lo_n"], Q["hi_n"], Q["sid"],
                                                    Z["ZLO"], Z["ZHI"], xv, yv)
                if not len(T0):
                    continue
                c0 = clock[T0]
                # ---- 단조성 표용(중복 허용): T_TAB마다 '아직 버티는 중'인 에피소드의 다음 봉 시가 진입
                for tt in T_TAB:
                    if tt == 0:
                        ei = Ss.copy()
                        ent = np.where(TY == 0, np.minimum(o[ei], HI), o[ei])
                        alive = np.ones(len(ei), bool)
                    else:
                        b = np.maximum(np.searchsorted(clock, c0 + tt - mins, "left"), Ss)
                        alive = (b < E) & (b + 1 < n)
                        ei = np.minimum(b + 1, n - 1)
                        ent = o[ei]
                    sel = alive & inper(ei)
                    if not sel.any():
                        continue
                    e2, p2 = ei[sel], ent[sel]
                    R = rets(A, Q, e2, p2)
                    eps.append(pd.DataFrame(dict(x=xv, y=yv, T=tt, sym=s, day=dayo[e2], reg=reg[e2], kind=K[sel],
                                                 main=R["main"], h4=R["4h"], d1=R["1d"],
                                                 rb=RT["main"][Q["buck"][e2]], rb4=RT["4h"][Q["buck"][e2]],
                                                 rb1d=RT["1d"][Q["buck"][e2]])))
                # ---- 돌파 표용: '위로 벗어남'으로 끝난 에피소드, 끝날 때 T 구간별
                up = (K == 1) & (E + 1 < n)
                if up.any():
                    ei = E[up] + 1
                    Tend = clock[E[up]] + mins - c0[up]
                    sel = inper(ei)
                    e2 = ei[sel]
                    R = rets(A, Q, e2, o[e2])
                    brk.append(pd.DataFrame(dict(x=xv, y=yv, Tend=Tend[sel], run=o[e2] / HI[up][sel] - 1, sym=s,
                                                 day=dayo[e2], reg=reg[e2], main=R["main"], h4=R["4h"], d1=R["1d"],
                                                 rb=RT["main"][Q["buck"][e2]])))
                # ---- 규칙 거래
                for k, (f, x2, y2, tt) in enumerate(RULES):
                    if x2 != xv or y2 != yv:
                        continue
                    if f == "i":
                        b = np.maximum(np.searchsorted(clock, c0 + tt - mins, "left"), Ss)
                        ok = (b < E) & (b + 1 < n)
                        ei = b[ok] + 1
                        assert (ei > Ss[ok]).all()
                    else:
                        ok = (K == 1) & (E + 1 < n)
                        ok &= (clock[np.minimum(E, n - 1)] + mins - c0) >= tt
                        ei = E[ok] + 1
                        okc = o[ei] <= HI[ok] * (1 + y2 + ZPLUS)
                        ei = ei[okc]
                    emit(k, ei, o[ei])
    T = pd.concat(trades, ignore_index=True)
    T["period"] = period
    EP = pd.concat(eps, ignore_index=True); EP["period"] = period
    BR = pd.concat(brk, ignore_index=True); BR["period"] = period
    return T, EP, BR


def ctt(v, d):
    """날짜 군집 t(평균 0 검정)."""
    v, d = np.asarray(v, float), np.asarray(d)
    m = ~np.isnan(v)
    v, d = v[m], d[m]
    if len(v) < 3:
        return np.nan
    mu = v.mean()
    g = pd.Series(v - mu).groupby(d).sum()
    G = len(g)
    if G < 2:
        return np.nan
    se = np.sqrt((g ** 2).sum()) / len(v) * np.sqrt(G / (G - 1))
    return mu / se if se > 0 else np.nan


def stats(t, ex="main"):
    r = t[ex].values
    m = ~np.isnan(r)
    r, d, rb = r[m], t.day.values[m], t["rb_" + ex].values[m]
    if not len(r):
        return dict(n=0, avg=np.nan, win=np.nan, worst=np.nan, sum=0.0, t=np.nan, rnd=np.nan, ex=np.nan, t_ex=np.nan)
    exv = r - rb
    return dict(n=len(r), avg=r.mean(), win=(r > 0).mean(), worst=r.min(), sum=r.sum(), t=ctt(r, d),
                rnd=np.nanmean(rb), ex=np.nanmean(exv), t_ex=ctt(exv, d))


def fmt_row(st):
    if not st["n"]:
        return "   0건"
    return (f"{st['n']:5d}건 평균 {st['avg']:+.2%} 승률 {st['win']:.0%} 최악 {st['worst']:+.1%} 합 {st['sum']:+.2f} "
            f"t {st['t']:+.2f} | 무작위 {st['rnd']:+.2%} 초과 {st['ex']:+.2%} t초과 {st['t_ex']:+.2f}")


def main():
    t0 = time.time()
    L = []
    P = lambda *a: L.append(" ".join(str(x) for x in a))
    res = {p: run_period(p) for p in ("early", "late")}
    T = pd.concat([res[p][0] for p in res], ignore_index=True)
    EP = pd.concat([res[p][1] for p in res], ignore_index=True)
    BR = pd.concat([res[p][2] for p in res], ignore_index=True)
    sel = lambda df, p, u, sc: df[(df.period == p) & df.sym.isin(UNIV[u]) & ((sc == "24h") | df.reg.astype(bool))]

    P("40분봉 지지 구역 버틴 시간 연구 (zone40, 2026-10-09)")
    P("자료: 원자료 끝 = 10/6 19:59 ET(10/7·10/8 없음). 고르기 early 7/14~9/8 5분봉(RSI9), 확인 late 9/10~10/6 3분봉(RSI14).")
    P(f"시험 규칙 {len(RULES)}개 = (i) x3×y3×T5 + (ii) x3×y3×T5. 선택 = early·전체137·24시간·기본 매도에서 무작위 대비 초과수익의 날짜 군집 t 최대(건수 30 이상).")
    P("기본 매도 = RSI>70 다음 봉 시가 또는 매수 뒤 첫 한국 8시. '무작위' = 같은 종목·같은 30분 시간대 모든 봉 시가 진입 평균(같은 매도).")
    P("")

    # ---- 구역 계산 점검: TEM 10/6 09:57 ET
    Z = zone_sets("TEM")
    tq = pd.Timestamp("2026-10-06 09:57", tz="America/New_York").as_unit("ns").value
    k = np.searchsorted(Z["conf"], tq, side="right")
    zs = [(Z["ZLO"][k, j], Z["ZHI"][k, j], Z["ZC"][k, j]) for j in range(Z["ZLO"].shape[1]) if np.isfinite(Z["ZLO"][k, j])]
    P("[점검] TEM 10/6 09:57 ET 구역(5분봉으로 만든 40분봉, 확정 스윙만):",
      ", ".join(f"{a:.2f}~{b:.2f}×{c}" for a, b, c in zs if 70 < (a + b) / 2 < 90))
    P("       참고 sr_zones_ref.py(1분봉 기반): 지지 81.78~81.83×2 80.25~80.61×4 78.97~79.26×4 77.01~77.13×3 75.80~76.18×7 "
      "74.72×1 71.47~71.48×2 / 저항 83.76~84.00×4 86.25~86.60×3 89.09×1")
    P("")

    # ---- 단조성 표 A
    P("=" * 110)
    P("[표 A] 버틴 시간 T에 아직 버티는 중(깨짐·돌파 없음)일 때 다음 봉 시가에 산 결과 — 에피소드마다(중복 허용), x=0.5% y=1%")
    P("  T=0은 닿는 순간 구역 상단 지정가. 깨짐 = 그 뒤 결국 하단-x 아래 종가, 돌파 = 상단+y 위 종가. 초과 = 수익 - 같은 시간대 무작위.")
    for p in ("early", "late"):
        for u, sc in (("전체137", "24h"), ("전체137", "reg"), ("사용자78", "24h"), ("사용자78", "reg")):
            P(f"-- {PN[p]} · {u} · {'24시간' if sc == '24h' else '정규장 진입만'}")
            P(f"   {'T':>5} {'건수':>6} {'깨짐':>5} {'돌파':>5} {'평균(기본)':>9} {'승률':>4} {'초과':>7} {'t초과':>6} {'4h평균':>7} {'4h초과':>7} {'1d평균':>7} {'1d초과':>7}")
            e = sel(EP, p, u, sc)
            e = e[(e.x == 0.005) & (e.y == 0.01)]
            for tt in T_TAB:
                q = e[e["T"] == tt]
                if not len(q):
                    continue
                exv = q.main - q.rb
                P(f"   {tt/60:>4g}h {len(q):6d} {(q.kind == 0).mean():5.0%} {(q.kind == 1).mean():5.0%} {q.main.mean():+9.2%} "
                  f"{(q.main > 0).mean():4.0%} {exv.mean():+7.2%} {ctt(exv, q.day):+6.2f} {q.h4.mean():+7.2%} "
                  f"{(q.h4 - q.rb4).mean():+7.2%} {q.d1.mean():+7.2%} {(q.d1 - q.rb1d).mean():+7.2%}")
    P("")
    P("[표 A2] 모든 x·y 조합: T별 '결국 깨짐' 비율 / 기본 매도 초과수익(전체137·24시간). 행 = x,y / 열 = T")
    for p in ("early", "late"):
        P(f"-- {PN[p]}")
        P("   x    y    | " + " | ".join(f"{tt/60:>4g}h 깨짐 초과  " for tt in T_TAB))
        e = sel(EP, p, "전체137", "24h")
        for xv in XS:
            for yv in YS:
                q = e[(e.x == xv) & (e.y == yv)]
                cells = []
                for tt in T_TAB:
                    w = q[q["T"] == tt]
                    cells.append(f"{len(w):5d} {(w.kind == 0).mean():3.0%} {(w.main - w.rb).mean():+.2%}" if len(w) else " " * 18)
                P(f"   {xv:.1%} {yv:.1%} | " + " | ".join(cells))
    P("")

    # ---- 돌파 표 B
    P("=" * 110)
    P("[표 B] '위로 벗어남'으로 끝난 에피소드: 끝날 때까지 버틴 시간 구간별, 돌파 다음 봉 시가 매수(중복 허용), x=0.5% y=1%")
    P("  '이미 오름' = 다음 봉 시가가 상단+2% 초과(규칙 (ii)는 이 경우 제외).")
    bins = [0, 30, 60, 120, 240, 480, 1e9]
    labs = ["<0.5h", "0.5~1h", "1~2h", "2~4h", "4~8h", "8h+"]
    for p in ("early", "late"):
        for u, sc in (("전체137", "24h"), ("전체137", "reg"), ("사용자78", "24h")):
            b = sel(BR, p, u, sc)
            b = b[(b.x == 0.005) & (b.y == 0.01)].copy()
            b["bk"] = pd.cut(b.Tend, bins, right=False, labels=labs)
            P(f"-- {PN[p]} · {u} · {'24시간' if sc == '24h' else '정규장 진입만'}")
            P(f"   {'버틴 시간':>8} {'건수':>6} {'이미오름':>7} {'평균':>7} {'승률':>4} {'초과':>7} {'t초과':>6} {'4h':>7} {'1d':>7} | 제외 안 한 경우 평균")
            for lb in labs:
                q = b[b.bk == lb]
                if not len(q):
                    continue
                cap = q.run <= 0.01 + ZPLUS
                w = q[cap]
                exv = w.main - w.rb
                P(f"   {lb:>8} {len(w):6d} {(~cap).mean():7.0%} {w.main.mean():+7.2%} {(w.main > 0).mean():4.0%} "
                  f"{exv.mean():+7.2%} {ctt(exv, w.day):+6.2f} {w.h4.mean():+7.2%} {w.d1.mean():+7.2%} | {q.main.mean():+.2%}")
    P("")

    # ---- 기준
    P("=" * 110)
    P("[기준] (a) 가장 가까운 지지 상단 지정가 / (c) 지금 이격매수 지표. 한 종목 한 포지션. 기본 매도.")
    base = {}
    for p in ("early", "late"):
        for u in UNIV:
            for sc in ("24h", "reg"):
                for nm in ("A", "C"):
                    q = T[(T.rule == nm) & (T.period == p) & (T.scope == sc) & T.sym.isin(UNIV[u])]
                    base[(nm, p, u, sc)] = stats(q)
                    P(f"  {PN[p]:<22} {u:<6} {'24시간' if sc == '24h' else '정규장':<4} {'(a) 지지 닿음' if nm == 'A' else '(c) 이격매수':<9} "
                      + fmt_row(base[(nm, p, u, sc)]))
    P("")

    # ---- 규칙 90개
    rs = {}
    for (rule, p, sc), g in T[T.rule.apply(lambda v: not isinstance(v, str))].groupby(["rule", "period", "scope"]):
        for u in UNIV:
            rs[(rule, p, u, sc)] = stats(g[g.sym.isin(UNIV[u])])
    empty = stats(T.iloc[:0])
    get = lambda k, p, u, sc: rs.get((k, p, u, sc), empty)
    score = []
    for k in range(len(RULES)):
        st = get(k, "early", "전체137", "24h")
        score.append(st["t_ex"] if st["n"] >= 30 and not np.isnan(st["t_ex"]) else -np.inf)
    order = np.argsort(score)[::-1]
    best = int(order[0])

    def beats(k, p, u, sc):
        st, a = get(k, p, u, sc), base[("A", p, u, sc)]
        return st["n"] >= 10 and st["avg"] > a["avg"] and st["ex"] > 0

    def beats_c(k, p, u, sc):
        st, cc = get(k, p, u, sc), base[("C", p, u, sc)]
        return st["n"] >= 10 and st["avg"] > cc["avg"]

    P("=" * 110)
    P("[규칙] 두 기간 모두 기준보다 나은 규칙 수 (나음 = 평균이 (a)보다 크고 무작위 대비 초과 > 0, 건수 10 이상)")
    for u in UNIV:
        for sc in ("24h", "reg"):
            nb = sum(beats(k, "early", u, sc) and beats(k, "late", u, sc) for k in range(len(RULES)))
            ne = sum(beats(k, "early", u, sc) for k in range(len(RULES)))
            nc = sum(beats_c(k, "early", u, sc) and beats_c(k, "late", u, sc) for k in range(len(RULES)))
            P(f"  {u} {'24시간' if sc == '24h' else '정규장'}: 고르기만 {ne}/{len(RULES)}, 두 기간 모두 {nb}/{len(RULES)}  "
              f"(참고: 두 기간 모두 평균이 (c) 이격매수보다 큼 {nc})")
            for k in range(len(RULES)):
                if beats(k, "early", u, sc) and beats(k, "late", u, sc):
                    a, b = get(k, "early", u, sc), get(k, "late", u, sc)
                    P(f"      #{k:02d} {rname(k)} | 고르기 {a['n']}건 {a['avg']:+.2%} 초과 {a['ex']:+.2%} | "
                      f"확인 {b['n']}건 {b['avg']:+.2%} 초과 {b['ex']:+.2%} (선택 기준으로 뽑힌 것 아님)")
    P("")
    P("[선택 기준 민감도] early·전체137·24시간 t초과 최대를 건수 하한별로 다시 고르면(확인 기간은 결과만 봄)")
    for mn in (30, 100, 300, 1000):
        sc_ = [get(k, "early", "전체137", "24h") for k in range(len(RULES))]
        cand_k = [k for k in range(len(RULES)) if sc_[k]["n"] >= mn and not np.isnan(sc_[k]["t_ex"])]
        if not cand_k:
            continue
        kb = max(cand_k, key=lambda k: sc_[k]["t_ex"])
        a, b = sc_[kb], get(kb, "late", "전체137", "24h")
        P(f"  건수≥{mn:<5} #{kb:02d} {rname(kb)} | 고르기 {a['n']}건 {a['avg']:+.2%} 초과 {a['ex']:+.2%} t {a['t_ex']:+.2f} | "
          f"확인 {b['n']}건 {b['avg']:+.2%} 초과 {b['ex']:+.2%} t {b['t_ex']:+.2f}")
    P("")
    P("[규칙 상위 15 — early·전체137·24시간 t초과 순] 고르기 | 확인")
    for k in order[:15]:
        k = int(k)
        a, b = get(k, "early", "전체137", "24h"), get(k, "late", "전체137", "24h")
        P(f"  #{k:02d} {rname(k)}")
        P(f"      고르기 {fmt_row(a)}")
        P(f"      확인   {fmt_row(b)}")
    P("")
    P("[규칙 전체 90개 한 줄 요약: early 건수/평균/초과/t초과 | late 건수/평균/초과/t초과] 전체137·24시간")
    for k in range(len(RULES)):
        a, b = get(k, "early", "전체137", "24h"), get(k, "late", "전체137", "24h")
        P(f"  #{k:02d} {rname(k):<60} | {a['n']:4d} {a['avg']:+.2%} {a['ex']:+.2%} {a['t_ex']:+.2f} | "
          f"{b['n']:4d} {b['avg']:+.2%} {b['ex']:+.2%} {b['t_ex']:+.2f}")
    P("")

    # ---- 최선안
    P("=" * 110)
    P(f"[최선안] #{best} {rname(best)}  (고르기 기간으로만 선택)")
    cells = []
    for p in ("early", "late"):
        for u in UNIV:
            for sc in ("24h", "reg"):
                q = T[(T.rule == best) & (T.period == p) & (T.scope == sc) & T.sym.isin(UNIV[u])]
                for ex in EXITS:
                    st = stats(q, ex)
                    qa = T[(T.rule == "A") & (T.period == p) & (T.scope == sc) & T.sym.isin(UNIV[u])]
                    qc = T[(T.rule == "C") & (T.period == p) & (T.scope == sc) & T.sym.isin(UNIV[u])]
                    sa, sc_ = stats(qa, ex), stats(qc, ex)
                    P(f"  {PN[p]:<22} {u:<6} {'24시간' if sc == '24h' else '정규장':<4} {ex:<4} {fmt_row(st)}"
                      f" || (a) {sa['avg']:+.2%}/{sa['n']} (c) {sc_['avg']:+.2%}/{sc_['n']}")
                    if ex == "main":
                        cells.append(dict(period=PN[p], universe=u, scope="24시간" if sc == "24h" else "정규장 진입만",
                                          n=int(st["n"]), avg=round(float(st["avg"]) * 100, 3) if st["n"] else 0.0,
                                          win=round(float(st["win"]) * 100, 1) if st["n"] else 0.0,
                                          sum=round(float(st["sum"]) * 100, 2), t=round(float(st["t"]), 2) if st["n"] > 2 else 0.0,
                                          vs_baseline=f"무작위 {st['rnd']:+.2%} 초과 {st['ex']:+.2%}(t {st['t_ex']:+.2f}); "
                                                      f"(a)지지닿음 {sa['avg']:+.2%}/{sa['n']}건; (c)이격매수 {sc_['avg']:+.2%}/{sc_['n']}건"))
    P("")
    P(f"걸린 시간 {time.time() - t0:.0f}초")
    txt = "\n".join(L)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    import json
    with open(os.path.join(os.path.dirname(RAW), "zone40_best.json"), "w", encoding="utf-8") as f:
        json.dump(dict(best=best, name=rname(best), cells=cells), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
