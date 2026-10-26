"""zone40 반박 검증(2026-10-09): best_rule.definition 문장만 보고 다시 만든다(zone40.py 코드 안 씀).

공용 시험대 stop_lab.load(기준봉)·base_trades(이격매수 (c))만 빌려 쓴다.
구역: 5분봉 → ET 04:00 기준 40분봉(날짜별 (분-240)//40 칸), 좌우 5봉 피벗, 확정 = 피벗+5봉의 끝,
      최근 스윙 60개, 첫 가격 대비 0.5% 묶음, 중간값 1.2% 미만이면 (스윙 수, 마지막 봉) 큰 쪽.
에피소드: 봉 i 시작 때 구역 중 상단 < c[i-1]인 가장 가까운 구역 상단에 l[i]가 닿으면 시작(lo·hi 고정),
      종가 < lo(1-x) 깨짐, > hi(1+y) 돌파. 깨진 봉 종가가 아래 구역 [zl(1-x), zh] 안이면 다음 봉부터 이어서(시계는 깨진 봉부터).
규칙(i): 세션 시계(04~20시 분, 자료 있는 날만)로 시작 봉 시작~지금 봉 끝이 T에 처음 이른 봉 종가까지 살아 있으면 다음 봉 시가.
매도: r[k]>70(k>=매수 봉) 다음 봉 시가, 또는 매수 봉 기준 첫 한국 8시 이후 첫 봉 시가. 한 종목 한 포지션(매도 봉 다음부터).
"""
import os, sys, time
from datetime import date
import numpy as np, pandas as pd
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view as swv

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
from stop_lab import load, base_trades, RAW, USER78, ALL

OUT = os.path.normpath(os.path.join(CODE, "../results/support/verify_zone40.txt"))
PER = {"early": (date(2026, 7, 14), date(2026, 9, 8)), "late": (date(2026, 9, 10), date(2026, 10, 6))}
UNI = {"전체137": set(ALL), "사용자78": set(USER78)}
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    L.append(s)
    print(s, flush=True)


# ------------------------------------------------------------------ 40분봉·스윙·구역
def swings40(sym):
    m = pd.read_parquet(f"{RAW}/5m_{sym}.parquet")
    m = m[m.High > 0]
    mod = m.index.hour * 60 + m.index.minute
    slot = (mod - 240) // 40
    start = m.index.normalize() + pd.to_timedelta(240 + 40 * slot, unit="min")
    g = m.groupby(start).agg(H=("High", "max"), Lo=("Low", "min")).sort_index()
    st = g.index.as_unit("ns").asi8
    h, l = g.H.to_numpy(float), g.Lo.to_numpy(float)
    nb = len(h)
    rows = []                                                     # (bar, price) 순서: 봉 순, 같은 봉은 저점 먼저
    if nb >= 11:
        wl, wh = swv(l, 11), swv(h, 11)
        lf_l, lf_h = swv(l, 5), swv(h, 5)
        for c in range(5, nb - 5):
            if l[c] == wl[c - 5].min() and np.all(lf_l[c - 5] > l[c]):
                rows.append((c, l[c]))
            if h[c] == wh[c - 5].max() and np.all(lf_h[c - 5] < h[c]):
                rows.append((c, h[c]))
    bar = np.array([r[0] for r in rows], np.int64)
    px = np.array([r[1] for r in rows], float)
    conf = st[bar + 5] + 40 * 60 * 10**9 if len(bar) else np.zeros(0, np.int64)
    return dict(bar=bar, px=px, conf=conf, st=st)


def build_zones(px, bar):
    """px, bar: 최근 60개 스윙. 반환 [(lo, hi, cnt, last)] 중간값 오름차순."""
    cl = []
    for j in np.argsort(px, kind="stable"):
        p = px[j]
        if cl and (p - cl[-1][0]) / cl[-1][0] <= 0.005:
            cl[-1][1] = p
            cl[-1][2] += 1
            cl[-1][3] = max(cl[-1][3], bar[j])
        else:
            cl.append([p, p, 1, bar[j]])
    kept = []
    for z in sorted(cl, key=lambda z: (z[2], z[3]), reverse=True):
        mz = (z[0] + z[1]) / 2
        if all(abs(mz - (k[0] + k[1]) / 2) / ((k[0] + k[1]) / 2) >= 0.012 for k in kept):
            kept.append(z)
    return sorted(kept, key=lambda z: (z[0] + z[1]) / 2)


def zone_table(SW, sids):
    """필요한 확정 스윙 수 k마다 구역 표. 반환 ZL, ZH (k × J, 빈칸 nan), NZ."""
    ks = np.unique(sids)
    tabs = {}
    for k in ks:
        a = max(0, k - 60)
        tabs[k] = build_zones(SW["px"][a:k], SW["bar"][a:k]) if k > 0 else []
    K = int(SW["px"].shape[0]) + 1
    J = max([len(v) for v in tabs.values()] + [1])
    ZL = np.full((K, J), np.nan)
    ZH = np.full((K, J), np.nan)
    NZ = np.zeros(K, np.int64)
    for k, zs in tabs.items():
        NZ[k] = len(zs)
        for j, z in enumerate(zs):
            ZL[k, j], ZH[k, j] = z[0], z[1]
    return ZL, ZH, NZ


# ------------------------------------------------------------------ 에피소드·체결
@njit(cache=True)
def nearest_support(c, l, sid, ZL, ZH, NZ):
    n = len(c)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    touch = np.zeros(n, np.bool_)
    for i in range(1, n):
        k = sid[i]
        best = -1
        bh = -1.0
        for j in range(NZ[k]):
            if ZH[k, j] < c[i - 1] and ZH[k, j] > bh:
                bh = ZH[k, j]
                best = j
        if best >= 0:
            lo[i] = ZL[k, best]
            hi[i] = ZH[k, best]
            touch[i] = l[i] <= hi[i]
    return lo, hi, touch


@njit(cache=True)
def episodes(c, touch, lo_n, hi_n, sid, ZL, ZH, NZ, x, y):
    n = len(c)
    T0 = np.empty(n, np.int64); S = np.empty(n, np.int64); E = np.empty(n, np.int64)
    K = np.empty(n, np.int64); LO = np.empty(n); HI = np.empty(n); CH = np.empty(n, np.int64)
    m = 0
    i = 1
    while i < n:
        if not touch[i]:
            i += 1
            continue
        t0, s, lo, hi, ch = i, i, lo_n[i], hi_n[i], 0
        while True:
            dn, up = lo * (1 - x), hi * (1 + y)
            e = s
            kind = 2
            while e < n:
                if c[e] < dn:
                    kind = 0
                    break
                if c[e] > up:
                    kind = 1
                    break
                e += 1
            T0[m] = t0; S[m] = s; E[m] = e; K[m] = kind; LO[m] = lo; HI[m] = hi; CH[m] = ch
            m += 1
            if kind == 0 and e + 1 < n:
                k = sid[e + 1]
                p = c[e]
                bj = -1
                bh = -1.0
                for j in range(NZ[k]):
                    zl, zh = ZL[k, j], ZH[k, j]
                    if zh < lo and zl * (1 - x) <= p and p <= zh and zh > bh:
                        bj = j
                        bh = zh
                if bj >= 0:
                    t0, s, lo, hi, ch = e, e + 1, ZL[k, bj], ZH[k, bj], 1
                    continue
            break
        i = e + 1
    return T0[:m], S[:m], E[:m], K[:m], LO[:m], HI[:m], CH[:m]


@njit(cache=True)
def exit_bar(r, tk, dead):
    """봉 j에 샀을 때 매도 봉(없으면 n)."""
    n = len(r)
    nr = np.empty(n, np.int64)
    nxt = n
    for j in range(n - 1, -1, -1):
        if r[j] > 70.0:
            nxt = j
        nr[j] = nxt
    out = np.empty(n, np.int64)
    for j in range(n):
        a = nr[j] + 1 if nr[j] < n else n
        b = np.searchsorted(tk, dead[j])
        out[j] = min(a, b)
    return out


@njit(cache=True)
def one_position(ei, xb):
    keep = np.zeros(len(ei), np.bool_)
    busy = -1
    for k in range(len(ei)):
        if ei[k] > busy:
            keep[k] = True
            busy = xb[ei[k]]
    return keep


# ------------------------------------------------------------------ 통계
def cl_t(v, d):
    v = np.asarray(v, float); d = np.asarray(d)
    ok = np.isfinite(v)
    v, d = v[ok], d[ok]
    if len(v) < 3:
        return np.nan
    u = v - v.mean()
    s = pd.Series(u).groupby(d).sum().to_numpy()
    G = len(s)
    if G < 2:
        return np.nan
    se = np.sqrt((s ** 2).sum() * G / (G - 1)) / len(v)
    return v.mean() / se if se > 0 else np.nan


def st(df):
    df = df[np.isfinite(df.ret)]
    if not len(df):
        return dict(n=0, avg=np.nan, win=np.nan, worst=np.nan, sum=0, t=np.nan, rb=np.nan, ex=np.nan, tex=np.nan)
    ex = df.ret - df.rb
    return dict(n=len(df), avg=df.ret.mean(), win=(df.ret > 0).mean(), worst=df.ret.min(), sum=df.ret.sum(),
                t=cl_t(df.ret, df.day), rb=df.rb.mean(), ex=ex.mean(), tex=cl_t(ex, df.day))


def fr(s):
    if not s["n"]:
        return "0건"
    return (f"{s['n']:5d}건 평균 {s['avg']:+.3%} 승률 {s['win']:.1%} 최악 {s['worst']:+.1%} 합 {s['sum']*100:+.2f}%p "
            f"t {s['t']:+.2f} | 무작위 {s['rb']:+.2%} 초과 {s['ex']:+.2%} t초과 {s['tex']:+.2f}")


# ------------------------------------------------------------------ 실행
XY = [(x, y) for x in (0.0, 0.0025, 0.005, 0.01) for y in (0.0025, 0.005, 0.01, 0.02)]
TS = (0, 30, 60, 120, 240, 360, 480, 600, 720, 960)


def run(period):
    lo_d, hi_d = (d.toordinal() for d in PER[period])
    S = load(period)
    rule_rows, ep_rows, a_rows = [], [], []
    for sym in ALL:
        if sym not in S:
            continue
        A = S[sym]
        o, l, c, r, tk = A["o"], A["l"], A["c"], A["r"], A["tk"]
        n, mins = len(o), int(A["mins"])
        SW = swings40(sym)
        sid = np.searchsorted(SW["conf"], tk, side="right")
        if len(SW["conf"]):
            u = sid > 0
            assert np.all(SW["conf"][sid[u] - 1] <= tk[u])        # 쓰는 스윙은 봉 시작 전에 확정
            assert np.all((sid == len(SW["conf"])) | (SW["conf"][np.minimum(sid, len(SW["conf"]) - 1)] > tk))
        ZL, ZH, NZ = zone_table(SW, np.unique(np.r_[sid, np.minimum(sid + 1, len(SW["px"]))]))
        lo_n, hi_n, touch = nearest_support(c, l, sid, ZL, ZH, NZ)
        dord = np.array([d.toordinal() for d in A["day"]])
        drank = np.unique(dord, return_inverse=True)[1]
        clock = drank * 960 + (A["mod"].astype(np.int64) - 240)
        assert np.all(np.diff(clock) > 0)
        xb = exit_bar(r, tk, A["dead"])
        ok_x = xb < n
        ret_at = lambda ei, px: np.where(ok_x[ei], o[np.minimum(xb[ei], n - 1)] / px - 1, np.nan)
        inper = (dord >= lo_d) & (dord <= hi_d)
        # 무작위: 같은 종목·같은 30분 칸, 기간 안 모든 봉 시가 진입
        bk = (A["mod"] // 30).astype(np.int64)
        j = np.flatnonzero(inper & ok_x)
        rr = o[xb[j]] / o[j] - 1
        rb_tab = np.full(48, np.nan)
        cnt = np.bincount(bk[j], minlength=48)
        sm = np.bincount(bk[j], weights=rr, minlength=48)
        rb_tab[cnt > 0] = sm[cnt > 0] / cnt[cnt > 0]
        reg = A["reg"]

        def emit(store, tag, ei, px, **kw):
            o_ = np.argsort(ei, kind="stable")
            ei, px = ei[o_], px[o_]
            for scope in ("24h", "reg"):
                m = np.ones(len(ei), bool) if scope == "24h" else reg[ei]
                e2, p2 = ei[m], px[m]
                if not len(e2):
                    continue
                kp = one_position(e2, xb) & inper[e2]
                e2, p2 = e2[kp], p2[kp]
                if not len(e2):
                    continue
                store.append(pd.DataFrame(dict(tag=[tag] * len(e2), scope=scope, sym=sym, ei=e2, day=dord[e2],
                                               ret=ret_at(e2, p2), rb=rb_tab[bk[e2]], **kw)))

        ia = np.flatnonzero(touch)
        emit(a_rows, "A", ia, np.minimum(o[ia], hi_n[ia]))
        for (x, y) in XY:
            T0, Ss, E, K, LO, HI, CH = episodes(c, touch, lo_n, hi_n, sid, ZL, ZH, NZ, x, y)
            if not len(T0):
                continue
            c0 = clock[T0]
            for T in TS:
                if T == 0:
                    ei = Ss.copy()
                    px = np.where(CH == 0, np.minimum(o[ei], HI), o[ei])
                    alive = np.ones(len(ei), bool)
                else:
                    b = np.maximum(np.searchsorted(clock, c0 + T - mins, side="left"), Ss)
                    alive = (b < E) & (b + 1 < n)
                    # 확인: 판단 봉 b 끝의 경과 시간 >= T, 그 직전 봉까지는 T 미만(처음 이른 봉)
                    bb = np.minimum(b, n - 1)
                    assert np.all((clock[bb] + mins - c0)[alive] >= T)
                    prev_ok = (bb > Ss) & alive
                    assert np.all((clock[bb - 1] + mins - c0)[prev_ok] < T)
                    ei = np.minimum(b + 1, n - 1)
                    px = o[ei]
                    assert np.all(ei[alive] > b[alive])
                # 에피소드 표(중복 허용)
                sel = alive & inper[ei]
                if sel.any():
                    ep_rows.append(pd.DataFrame(dict(x=x, y=y, T=T, sym=sym, day=dord[ei[sel]], reg=reg[ei[sel]],
                                                     kind=K[sel], ret=ret_at(ei[sel], px[sel]), rb=rb_tab[bk[ei[sel]]])))
                # 규칙 (i) 거래(한 포지션)
                if T > 0 and alive.any():
                    emit(rule_rows, f"i|{x}|{y}|{T}", ei[alive], px[alive])
            # 규칙 (ii) 일부 확인: 돌파로 끝남 & 끝 봉까지 경과 >= T, 다음 봉 시가 <= hi(1+y+1%)
            up = (K == 1) & (E + 1 < n)
            Tend = clock[np.minimum(E, n - 1)] + mins - c0
            for T in (0, 60, 120, 240, 480):
                q = up & (Tend >= T)
                ei = E[q] + 1
                okc = o[ei] <= HI[q] * (1 + y + 0.01)
                if okc.any():
                    emit(rule_rows, f"ii|{x}|{y}|{T}", ei[okc], o[ei[okc]])
    R = pd.concat(rule_rows, ignore_index=True)
    EP = pd.concat(ep_rows, ignore_index=True)
    AA = pd.concat(a_rows, ignore_index=True)
    return R, EP, AA


def main():
    t_start = time.time()
    P("zone40 반박 검증 — best_rule.definition으로 독립 재구현 (verify_zone40.py)")
    # 구역 점검: TEM 10/6 09:57 ET
    SW = swings40("TEM")
    tq = pd.Timestamp("2026-10-06 09:57", tz="America/New_York").as_unit("ns").value
    k = int(np.searchsorted(SW["conf"], tq, side="right"))
    zs = build_zones(SW["px"][max(0, k - 60):k], SW["bar"][max(0, k - 60):k])
    P("[구역 점검] TEM 10/6 09:57 ET:", ", ".join(f"{z[0]:.2f}~{z[1]:.2f}x{z[2]}" for z in zs if 70 < (z[0] + z[1]) / 2 < 90))
    P("")
    res = {p: run(p) for p in ("early", "late")}
    for p in res:
        res[p][0]["period"] = p; res[p][1]["period"] = p; res[p][2]["period"] = p
    R = pd.concat([res[p][0] for p in res], ignore_index=True)
    EP = pd.concat([res[p][1] for p in res], ignore_index=True)
    AA = pd.concat([res[p][2] for p in res], ignore_index=True)
    R.to_parquet(os.path.join(os.path.dirname(RAW), "verify_zone40_trades.parquet"))
    pick = lambda df, p, u, sc: df[(df.period == p) & df.sym.isin(UNI[u]) & (df.scope == sc)]

    # 기준 (a), (c)
    P("[기준] (a) 지지 상단 지정가(한 포지션), (c) 이격매수(stop_lab.base_trades)")
    for p in ("early", "late"):
        for u in UNI:
            for sc in ("24h", "reg"):
                a = st(pick(AA, p, u, sc))
                b = base_trades(p, sc == "reg")
                b = b[b.sym.isin(UNI[u])]
                P(f"  {p:5} {u:6} {sc:3} (a) {fr(a)}")
                P(f"  {p:5} {u:6} {sc:3} (c) {len(b):5d}건 평균 {b.ret.mean():+.3%} 합 {b.ret.sum()*100:+.1f}%p")
    P("")

    # 최선안 #04
    best = "i|0.0|0.005|480"
    P("[최선안 #04 재현] (i) 8h 버팀 x0% y0.5%")
    for p in ("early", "late"):
        for u in UNI:
            for sc in ("24h", "reg"):
                q = pick(R, p, u, sc)
                q = q[q.tag == best]
                P(f"  {p:5} {u:6} {sc:3} {fr(st(q))}")
    P("")

    # 쏠림: early·전체·24h
    q = pick(R, "early", "전체137", "24h")
    q = q[(q.tag == best) & np.isfinite(q.ret)]
    P("[쏠림] #04 early 전체137 24h 거래:", len(q), "건")
    by_s = q.groupby("sym").ret.agg(["count", "sum"]).sort_values("sum", ascending=False)
    by_d = q.groupby("day").ret.agg(["count", "sum"]).sort_values("sum", ascending=False)
    P("  종목 수", len(by_s), "| 날짜 수", len(by_d))
    P("  상위 종목:", ", ".join(f"{s}({int(r['count'])}건 {r['sum']*100:+.2f}%p)" for s, r in by_s.head(5).iterrows()))
    P("  상위 날짜:", ", ".join(f"{date.fromordinal(int(d))}({int(r['count'])}건 {r['sum']*100:+.2f}%p)" for d, r in by_d.head(5).iterrows()))
    P("  하위 날짜:", ", ".join(f"{date.fromordinal(int(d))}({int(r['count'])}건 {r['sum']*100:+.2f}%p)" for d, r in by_d.tail(3).iterrows()))
    tot = q.ret.sum()
    for lab, drop in (("상위 1일 제외", [by_d.index[0]]), ("상위 2일 제외", list(by_d.index[:2])), ("상위 3일 제외", list(by_d.index[:3]))):
        w = q[~q.day.isin(drop)]
        P(f"  {lab}: {len(w)}건 평균 {w.ret.mean():+.3%} 초과 {(w.ret - w.rb).mean():+.3%}")
    for lab, drop in (("상위 1종목 제외", [by_s.index[0]]), ("상위 3종목 제외", list(by_s.index[:3]))):
        w = q[~q.sym.isin(drop)]
        P(f"  {lab}: {len(w)}건 평균 {w.ret.mean():+.3%} 초과 {(w.ret - w.rb).mean():+.3%}")
    P(f"  합 {tot*100:+.2f}%p 중 상위 1일 몫 {by_d['sum'].iloc[0]/tot:.0%}, 상위 3일 {by_d['sum'].iloc[:3].sum()/tot:.0%}, "
      f"상위 1종목 {by_s['sum'].iloc[0]/tot:.0%}")
    ql = pick(R, "late", "전체137", "24h")
    ql = ql[(ql.tag == best) & np.isfinite(ql.ret)]
    P("  late 거래 목록(종목 날짜 수익):", ", ".join(f"{s} {date.fromordinal(int(d)):%m/%d} {v:+.2%}" for s, d, v in zip(ql.sym, ql.day, ql.ret)))
    P("")

    # 이웃
    P("[이웃] 규칙 (i), 전체137·24h, 평균 / 초과(무작위 대비) / 건수 — early | late")
    for (x, y) in XY:
        cells = []
        for T in (240, 360, 480, 600, 720, 960):
            tg = f"i|{x}|{y}|{T}"
            s1 = st(pick(R, "early", "전체137", "24h").pipe(lambda d: d[d.tag == tg]))
            s2 = st(pick(R, "late", "전체137", "24h").pipe(lambda d: d[d.tag == tg]))
            cells.append(f"{T/60:g}h {s1['ex']:+.2%}({s1['n']}) | {s2['ex']:+.2%}({s2['n']})")
        P(f"  x{x:.2%} y{y:.2%}: " + " ;  ".join(cells))
    P("")
    P("[(ii) 일부] 전체137·24h early | late (평균, 초과, 건수)")
    for tg in ("ii|0.0|0.005|240", "ii|0.0|0.005|0", "ii|0.005|0.01|120", "ii|0.01|0.02|480"):
        s1 = st(pick(R, "early", "전체137", "24h").pipe(lambda d: d[d.tag == tg]))
        s2 = st(pick(R, "late", "전체137", "24h").pipe(lambda d: d[d.tag == tg]))
        P(f"  {tg}: early {s1['n']}건 {s1['avg']:+.2%} 초과 {s1['ex']:+.2%} | late {s2['n']}건 {s2['avg']:+.2%} 초과 {s2['ex']:+.2%}")
    P("")

    # 단조성 표 A (x=0.5%, y=1%) + 다른 조합
    for (x, y) in ((0.005, 0.01), (0.0, 0.005), (0.01, 0.02)):
        P(f"[표 A 재현] x{x:.1%} y{y:.1%}, 에피소드마다(중복 허용), 기본 매도")
        for p in ("early", "late"):
            for u, sc in (("전체137", "24h"), ("전체137", "reg"), ("사용자78", "24h")):
                e = EP[(EP.period == p) & (EP.x == x) & (EP.y == y) & EP.sym.isin(UNI[u])]
                if sc == "reg":
                    e = e[e.reg.astype(bool)]
                row = []
                for T in TS:
                    w = e[(e["T"] == T) & np.isfinite(e.ret)]
                    if not len(w):
                        continue
                    row.append(f"{T/60:g}h n{len(w)} 깨짐{(w.kind == 0).mean():.0%} 초과{(w.ret - w.rb).mean():+.2%}(t{cl_t(w.ret - w.rb, w.day):+.1f})")
                P(f"  {p:5} {u} {sc}: " + " | ".join(row))
        P("")
    # 단조성 요약: T>=30 구간에서 초과수익과 T의 순위상관(전 조합)
    P("[단조성 요약] 조합별 T(0.5h~16h) 대 초과수익 스피어만 상관, 전체137·24h")
    for p in ("early", "late"):
        rhos = []
        for (x, y) in XY:
            e = EP[(EP.period == p) & (EP.x == x) & (EP.y == y)]
            g = e[(e["T"] > 0) & np.isfinite(e.ret)].assign(ex=lambda d: d.ret - d.rb).groupby("T").ex.agg(["mean", "count"])
            g = g[g["count"] >= 30]
            if len(g) >= 4:
                rhos.append(pd.Series(g.index).corr(pd.Series(g["mean"].values), method="spearman"))
        rhos = np.array(rhos)
        P(f"  {p}: 조합 {len(rhos)}개, 상관 평균 {np.nanmean(rhos):+.2f}, 양(+0.5 이상) {int((rhos >= 0.5).sum())}개, 음(-0.5 이하) {int((rhos <= -0.5).sum())}개")
    P("")
    P(f"걸린 시간 {time.time() - t_start:.0f}초")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
