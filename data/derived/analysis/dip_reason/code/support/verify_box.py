"""반박 검증 — box 연구(장중 박스 버팀) 최선안을 정의 문장만 보고 다시 만든다(2026-10-09, box.py 코드는 쓰지 않음).

최선안(#52): 정규장, 박스 = 같은 날 정규장 안에서 저가 폭(max/min-1)이 w 이하로 유지된 가장 긴 연속 구간,
  봉 i-1까지의 박스가 D ≥ N분이고 거래량 감소(뒤 절반 < 앞 절반, 시각 보정)이면 봉 i에 지정가 F*(1+w/4),
  l[i] ≤ 지정가면 min(시가, 지정가) 체결. 박스당 1회, 종목당 1포지션, 매도 봉 다음 봉부터 재매수.
  매도: r[k] > 70인 봉 k(매수 봉 포함)의 다음 봉 시가 또는 매수 뒤 첫 한국 8시 이후 첫 봉 시가 중 먼저.
같이 보는 것: (a) 버팀 무시(N=15, 거래량 조건 없음), (b) 같은 종목·30분 시각대 무작위, (c) 지금 이격매수 지표,
  이웃 매개변수, 종목·날짜 쏠림, 박스 운명 표(버틴 시간별 위/아래 비율), N 문턱별 단조성, 사후 후보(240분+ LIFT·BRK 고정 보유).
사용: py support/verify_box.py (code 폴더) → ../results/support/verify_box.txt
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from datetime import date
import numpy as np, pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, run as lab_run, UNIV, RAW

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "support", "verify_box.txt"))
RANGE = {"early": (date(2026, 7, 14), date(2026, 9, 8)), "late": (date(2026, 9, 10), date(2026, 10, 6))}
PK = {"early": "앞", "late": "뒤"}
U78, U137 = UNIV["사용자78"], UNIV["전체137"]
RX = 70.0
HOLD = {"H4": 240, "D1": 1440}
_f = None


def out(*a):
    global _f
    if _f is None:
        _f = open(OUT, "w", encoding="utf-8")
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _f.write(s + "\n")
    _f.flush()


def pct(v, d=2):
    return "  n/a" if v is None or not np.isfinite(v) else f"{v * 100:+.{d}f}%"


# ------------------------------------------------------------------ 시각 보정 거래량(5분봉 파일, 직전 10거래일·최소 5일, 오늘 제외)
def rel_volume(A):
    m = pd.read_parquet(os.path.join(RAW, f"5m_{A['sym']}.parquet"))
    m = m[m.High > 0]
    t = m.index
    mm = t.hour * 60 + t.minute
    keep = (mm >= 570) & (mm < 960)
    dord = np.array([x.toordinal() for x in t.date])[keep]
    slot = np.asarray(mm[keep])
    vol = m.Volume.values[keep].astype(float)
    days = np.unique(dord)
    slots = np.arange(570, 960, 5)
    grid = np.full((len(days), len(slots)), np.nan)
    grid[np.searchsorted(days, dord), (slot - 570) // 5] = vol
    base = np.full_like(grid, np.nan)
    for k in range(len(days)):
        win = grid[max(0, k - 10):k]
        if len(win) == 0:
            continue
        cnt = np.isfinite(win).sum(0)
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(win, 0) if len(win) else np.full(len(slots), np.nan)
        base[k] = np.where(cnt >= 5, mu, np.nan)
    dayo = np.array([x.toordinal() for x in A["day"]])
    mod = A["mod"]
    reg = A["reg"]
    rv = np.full(len(mod), np.nan)
    di = np.searchsorted(days, dayo)
    okd = (di < len(days)) & (days[np.minimum(di, len(days) - 1)] == dayo) & reg
    si = (mod // 5) * 5
    sk = (si - 570) // 5
    okd &= (sk >= 0) & (sk < len(slots))
    b = np.full(len(mod), np.nan)
    b[okd] = base[di[okd], sk[okd]] * (A["mins"] / 5.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rv = A["v"] / b
    rv[~np.isfinite(rv)] = np.nan
    rv[~reg] = np.nan
    return rv


# ------------------------------------------------------------------ 상태 기계(박스 + 포지션)
@njit(cache=False)
def simulate(o, h, l, c, r, tk, dead, valid, newseg, inper, rv, mins, w, N, usev, mode, limfrac, rx):
    """mode 0 = LOW(박스 하단 지정가), 1 = LIFT(저가가 띠 위로 떠남, 다음 봉 시가), 2 = BRK(종가 > 박스 고가, 다음 봉 시가)."""
    n = len(o)
    cs = np.zeros(n + 1)
    cn = np.zeros(n + 1)
    for i in range(n):
        x = rv[i]
        if np.isfinite(x):
            cs[i + 1] = cs[i] + x
            cn[i + 1] = cn[i] + 1
        else:
            cs[i + 1] = cs[i]
            cn[i + 1] = cn[i]
    E = np.empty(n, np.int64); Xb = np.empty(n, np.int64); P = np.empty(n); DD = np.empty(n)
    k = 0
    s = -1; F = 0.0; Lm = 0.0; T = 0.0
    hold = False; ei = -1; dd = 0; lastx = -1; lastbox = -1; ent = 0.0; curD = 0.0
    pend = False; pbox = -1; pT = 0.0; pD = 0.0
    for i in range(n):
        if hold and i > ei:
            if r[i - 1] > rx or tk[i] >= dd:
                E[k] = ei; Xb[k] = i; P[k] = ent; DD[k] = curD; k += 1
                hold = False; lastx = i
        if (not hold) and i > lastx and inper[i] and valid[i] and (not newseg[i]):
            if mode == 0:
                if s >= 0 and s != lastbox:
                    D = (i - s) * mins
                    okv = True
                    if usev:
                        a = s; b = i - 1; L = b - a + 1
                        okv = False
                        if L >= 2:
                            mid = a + L // 2
                            k1 = cn[mid] - cn[a]; k2 = cn[b + 1] - cn[mid]
                            if k1 > 0 and k2 > 0:
                                okv = (cs[b + 1] - cs[mid]) / k2 < (cs[mid] - cs[a]) / k1
                    if D >= N and okv:
                        lim = F * (1 + limfrac * w)
                        if l[i] <= lim:
                            ent = min(o[i], lim); hold = True; ei = i; dd = dead[i]; lastbox = s; curD = D
            else:
                if pend and pbox != lastbox and o[i] <= pT * 1.01:
                    ent = o[i]; hold = True; ei = i; dd = dead[i]; lastbox = pbox; curD = pD
        pend = False
        if mode > 0 and valid[i] and (not newseg[i]) and s >= 0:
            D = (i - s) * mins
            if D >= N:
                if mode == 1:
                    sig = (l[i] > F * (1 + w)) and (c[i] <= T * 1.01)
                else:
                    sig = (c[i] > T) and (c[i] <= T * 1.01)
                if sig and usev:
                    a = s; b = i - 1; L = b - a + 1
                    sig = False
                    if L >= 2:
                        mid = a + L // 2
                        k1 = cn[mid] - cn[a]; k2 = cn[b + 1] - cn[mid]
                        if k1 > 0 and k2 > 0:
                            sig = (cs[b + 1] - cs[mid]) / k2 < (cs[mid] - cs[a]) / k1
                if sig and i + 1 < n and valid[i + 1] and (not newseg[i + 1]):
                    pend = True; pbox = s; pT = T; pD = D
        # 박스 갱신(봉 i 종가까지)
        if not valid[i]:
            s = -1
        elif newseg[i] or s < 0:
            s = i; F = l[i]; Lm = l[i]; T = h[i]
        elif l[i] > F * (1 + w) or l[i] < Lm / (1 + w):
            s = i; F = l[i]; Lm = l[i]; T = h[i]
        else:
            if l[i] < F:
                F = l[i]
            if l[i] > Lm:
                Lm = l[i]
            if h[i] > T:
                T = h[i]
    return E[:k], Xb[:k], P[:k], DD[:k]


@njit(cache=False)
def box_list(l, valid, newseg, w):
    """박스 목록: (시작 봉, 봉 수, 끝 방식 1 위 / -1 아래 / 0 세그먼트 끝)."""
    n = len(l)
    S = np.empty(n, np.int64); Lb = np.empty(n, np.int64); How = np.empty(n, np.int64)
    k = 0
    s = -1; F = 0.0; Lm = 0.0
    for i in range(n):
        if not valid[i]:
            if s >= 0:
                S[k] = s; Lb[k] = i - s; How[k] = 0; k += 1
            s = -1
            continue
        if newseg[i] or s < 0:
            if s >= 0:
                S[k] = s; Lb[k] = i - s; How[k] = 0; k += 1
            s = i; F = l[i]; Lm = l[i]
        elif l[i] > F * (1 + w):
            S[k] = s; Lb[k] = i - s; How[k] = 1; k += 1
            s = i; F = l[i]; Lm = l[i]
        elif l[i] < Lm / (1 + w):
            S[k] = s; Lb[k] = i - s; How[k] = -1; k += 1
            s = i; F = l[i]; Lm = l[i]
        else:
            if l[i] < F:
                F = l[i]
            if l[i] > Lm:
                Lm = l[i]
    if s >= 0:
        S[k] = s; Lb[k] = n - s; How[k] = 0; k += 1
    return S[:k], Lb[:k], How[:k]


# ------------------------------------------------------------------ 준비
D = {}


def ready(period):
    lo, hi = RANGE[period]
    S = load(period)
    for sym, A in S.items():
        n = len(A["o"])
        dayo = np.array([x.toordinal() for x in A["day"]])
        inper = (dayo >= lo.toordinal()) & (dayo <= hi.toordinal())
        reg = A["reg"].astype(bool)
        dchg = np.ones(n, bool)
        dchg[1:] = dayo[1:] != dayo[:-1]
        prev_reg = np.zeros(n, bool)
        prev_reg[1:] = reg[:-1]
        seg = {"reg": (reg, reg & ((~prev_reg) | dchg)), "all": (np.ones(n, bool), dchg)}
        o, r, tk = A["o"], A["r"], A["tk"]
        # 다음 RSI>70 봉(자기 포함)을 뒤에서 앞으로
        nxt = np.empty(n, np.int64)
        cur = n + 5
        for i in range(n - 1, -1, -1):
            if r[i] > RX:
                cur = i
            nxt[i] = cur
        ex_rsi = nxt + 1
        ex_dead = np.searchsorted(tk, A["dead"], side="left")
        ex = np.minimum(ex_rsi, ex_dead)
        ex = np.where(ex < n, ex, -1)
        exits = {}
        for sc in ("reg", "all"):
            pool = np.nonzero(seg[sc][0])[0]
            dct_ = {"RSI8": ex}
            for hk, hm in HOLD.items():
                q = np.searchsorted(tk[pool], tk + hm * 60 * 10**9, side="left")
                dct_[hk] = np.where(q < len(pool), pool[np.minimum(q, len(pool) - 1)], -1)
            exits[sc] = dct_
        bk = A["mod"] // 30
        rnd = {}
        for sc in ("reg", "all"):
            m0 = inper & seg[sc][0]
            rnd[sc] = {}
            for k_, xx in exits[sc].items():
                m = m0 & (xx >= 0)
                ret = np.where(xx >= 0, o[np.maximum(xx, 0)] / o - 1, np.nan)
                tab = pd.Series(ret[m]).groupby(bk[m]).mean()
                arr = np.full(48, np.nan)
                arr[tab.index.values] = tab.values
                rnd[sc][k_] = arr
        D[(period, sym)] = dict(A=A, dayo=dayo, inper=inper, seg=seg, exits=exits, rnd=rnd, bk=bk, rv=rel_volume(A))


def trades(period, sc, mode, w, N, usev, limfrac=0.25, syms=None):
    rows = []
    for (p, sym), Q in D.items():
        if p != period or (syms is not None and sym not in syms):
            continue
        A = Q["A"]
        valid, newseg = Q["seg"][sc]
        E, Xb, P, DDv = simulate(A["o"], A["h"], A["l"], A["c"], A["r"], A["tk"], A["dead"], valid, newseg, Q["inper"],
                                 Q["rv"], float(A["mins"]), w, float(N), usev, mode, limfrac, RX)
        if not len(E):
            continue
        o = A["o"]
        d = dict(sym=sym, day=Q["dayo"][E], e=E, D=DDv, RSI8=o[Xb] / P - 1, r_RSI8=Q["rnd"][sc]["RSI8"][Q["bk"][E]])
        for hk in HOLD:
            xx = Q["exits"][sc][hk][E]
            d[hk] = np.where(xx >= 0, o[np.maximum(xx, 0)] / P - 1, np.nan)
            d["r_" + hk] = Q["rnd"][sc][hk][Q["bk"][E]]
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["sym", "day", "e", "D", "RSI8", "r_RSI8"])


def ct(day, x):
    s = pd.Series(np.asarray(x, float)).groupby(np.asarray(day)).sum()
    if len(s) < 3 or s.std(ddof=1) == 0:
        return np.nan
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def st(t, syms, ex="RSI8"):
    t = t[t.sym.isin(syms)]
    t = t[np.isfinite(t[ex].astype(float))]
    if not len(t):
        return dict(n=0, avg=np.nan, win=np.nan, worst=np.nan, sum=0.0, rnd=np.nan, exc=np.nan, te=np.nan, t=np.nan)
    x = t[ex].values.astype(float)
    rr = t["r_" + ex].values.astype(float)
    exc = x - rr
    ok = np.isfinite(exc)
    return dict(n=len(t), avg=x.mean(), win=(x > 0).mean(), worst=x.min(), sum=x.sum(), rnd=np.nanmean(rr),
                exc=np.nanmean(exc), te=ct(t.day.values[ok], exc[ok]), t=ct(t.day.values, x))


def line(s):
    return (f"n{s['n']:>5} 평균 {pct(s['avg'])} 승률 {s['win'] * 100 if s['n'] else float('nan'):.0f}% 최악 {pct(s['worst'], 1)} 합 {s['sum']:+.2f}"
            f" | 무작위 {pct(s['rnd'])} 초과 {pct(s['exc'])} te {s['te']:+.2f}")


def main():
    t0 = time.time()
    for p in RANGE:
        ready(p)
    out(f"반박 검증 — box 최선안 독립 재구현 (준비 {time.time() - t0:.0f}s)")
    out(__doc__.split("사용:")[0].strip())
    out("")

    # ---------------- (c) 지금 지표
    C = {}
    for p in RANGE:
        lo, hi = RANGE[p]
        for reg_only in (True, False):
            b = lab_run(load(p), None, p, reg_only)
            b = b[(b.day >= lo) & (b.day <= hi)]
            C[(p, reg_only)] = b

    # ---------------- 1) 최선안 재현
    out("== 1) 최선안 #52 재현 (LOW w0.5% N≥60 vdec, 지정가 F*(1+w/4), RSI·8시)")
    BEST = {}
    for sc in ("reg", "all"):
        for p in RANGE:
            tb = trades(p, sc, 0, 0.005, 60, True)
            ta = trades(p, sc, 0, 0.005, 15, False)
            BEST[(sc, p)] = tb
            for un, us in (("사용자78", U78), ("전체137", U137)):
                s = st(tb, us); a = st(ta, us)
                cc = C[(p, sc == "reg")]
                cc = cc[cc.sym.isin(us)]
                out(f"  {sc:<3} {PK[p]} {un:<5} 최선 {line(s)} | (a) n{a['n']} 평균 {pct(a['avg'])} | (c) n{len(cc)} 평균 {pct(cc.ret.mean())} 합 {cc.ret.sum():+.2f}")
    out("")

    # ---------------- 2) 이웃 매개변수(정규장, 사용자78·전체137, RSI·8시)
    out("== 2) 이웃 매개변수 — 정규장 RSI·8시. 칸: n 평균 초과(te) / (a) 평균. 부호 = 초과 부호")
    NB = [("기준 w0.5 N60 vdec lim1/4", 0.005, 60, True, 0.25),
          ("w0.4", 0.004, 60, True, 0.25), ("w0.6", 0.006, 60, True, 0.25), ("w0.75", 0.0075, 60, True, 0.25),
          ("N45", 0.005, 45, True, 0.25), ("N90", 0.005, 90, True, 0.25), ("N120", 0.005, 120, True, 0.25),
          ("lim 0(하단 그대로)", 0.005, 60, True, 0.0), ("lim 1/2", 0.005, 60, True, 0.5),
          ("vdec 없음", 0.005, 60, False, 0.25),
          ("w0.4 N45", 0.004, 45, True, 0.25), ("w0.6 N90", 0.006, 90, True, 0.25)]
    A0 = {p: trades(p, "reg", 0, 0.005, 15, False) for p in RANGE}
    nb_rows = []
    for lab, w, N, uv, lf in NB:
        cells = []
        rec = dict(lab=lab)
        for p in RANGE:
            t_ = trades(p, "reg", 0, w, N, uv, lf)
            a_ = A0[p] if w == 0.005 else trades(p, "reg", 0, w, 15, False)
            for un, us in (("78", U78), ("137", U137)):
                s = st(t_, us); a = st(a_, us)
                rec[f"{p}{un}"] = s["exc"]
                rec[f"{p}{un}_a"] = s["avg"] - a["avg"]
                cells.append(f"{p[0]}{un} n{s['n']:>4} {pct(s['avg'])} {pct(s['exc'])}({s['te']:+.1f})/a{pct(a['avg'])}")
        nb_rows.append(rec)
        out(f"  {lab:<22} " + " | ".join(cells))
    NBd = pd.DataFrame(nb_rows)
    both = ((NBd.early78 > 0) & (NBd.late78 > 0)).sum()
    out(f"  이웃 {len(NBd)}개 중 앞·뒤 모두 무작위 대비 초과 > 0 (사용자78): {both}개, 뒤 기간 초과 > 0: {(NBd.late78 > 0).sum()}개, "
        f"뒤 기간 (a)보다 높음: {(NBd.late78_a > 0).sum()}개")
    out("")

    # ---------------- 3) 쏠림(앞 기간 = 고른 기간에서 이득이 어디서 왔나)
    out("== 3) 쏠림 — 최선안 정규장 사용자78, 초과(무작위 대비) 합의 출처")
    for p in RANGE:
        t = BEST[("reg", p)]
        t = t[t.sym.isin(U78)].copy()
        t["exc"] = t.RSI8 - t.r_RSI8
        t = t[np.isfinite(t.exc)]
        bd = t.groupby("day").exc.sum().sort_values()
        bs = t.groupby("sym").exc.sum().sort_values()
        tot = t.exc.sum()
        topd, tops = bd.index[-1], bs.index[-1]
        out(f"  {PK[p]}: 초과 합 {tot:+.3f} (n{len(t)}, 평균 {pct(t.exc.mean())})."
            f" 최대 날 {date.fromordinal(int(topd))} {bd.iloc[-1]:+.3f}, 최대 종목 {tops} {bs.iloc[-1]:+.3f}, 최악 거래 {pct(t.RSI8.min(), 1)}")
        w1 = t[t.day != topd]; w2 = t[t.sym != tops]
        w3 = t[t.exc < t.exc.quantile(0.99)]
        out(f"      최대 날 빼면 평균 초과 {pct(w1.exc.mean())} te {ct(w1.day.values, w1.exc.values):+.2f};"
            f" 최대 종목 빼면 {pct(w2.exc.mean())}; 상위 1% 거래 빼면 {pct(w3.exc.mean())}; 양(+) 날 비율 {(bd > 0).mean():.0%} ({len(bd)}일)")
        # 상위 3일 비중
        out(f"      상위 3일 초과 합 {bd.iloc[-3:].sum():+.3f}, 하위 3일 {bd.iloc[:3].sum():+.3f}; 상위 5종목 {bs.iloc[-5:].sum():+.3f}")
    out("")

    # ---------------- 4) 박스 운명 표(정규장): 오래 버틸수록 위로 끝나나
    out("== 4) 박스 운명 — 정규장, 버틴 시간 ≥ X인 박스의 끝 방식. 위/(위+아래)")
    XS = (15, 30, 60, 120, 240)
    for w in (0.005, 0.010, 0.015):
        for p in RANGE:
            for un, us in (("78", U78), ("137", U137)):
                agg = {x: np.zeros(3, np.int64) for x in XS}
                for (pp, sym), Q in D.items():
                    if pp != p or sym not in us:
                        continue
                    valid, newseg = Q["seg"]["reg"]
                    S_, Lb, How = box_list(Q["A"]["l"], valid, newseg, w)
                    m = Q["inper"][S_]
                    S_, Lb, How = S_[m], Lb[m], How[m]
                    mins = Q["A"]["mins"]
                    for x in XS:
                        mm = Lb * mins >= x
                        agg[x] += [int((How[mm] == 1).sum()), int((How[mm] == -1).sum()), int((How[mm] == 0).sum())]
                cells = []
                for x in XS:
                    u, d_, z = agg[x]
                    tot = u + d_ + z
                    cells.append(f"{x}분 {tot:>5} 위/(위+아래) {u / max(u + d_, 1):.0%} 마감 {z / max(tot, 1):.0%}")
                out(f"  w{w * 100:.1f}% {PK[p]}{un:<3} " + " | ".join(cells))
    out("")

    # ---------------- 5) 단조성 — N 문턱(박스가 N분을 넘긴 뒤 첫 기회)별 거래 결과(정규장, 거래량 조건 없음, RSI·8시)
    out("== 5) 단조성 — 문턱 N(15/30/60/120/240분) 넘긴 뒤 첫 기회 진입. 칸: n 평균 초과. 정규장, 사용자78, RSI·8시")
    MN = []
    for mode, mname in ((0, "LOW"), (1, "LIFT"), (2, "BRK")):
        for w in (0.005, 0.010, 0.015):
            cells = []
            for p in RANGE:
                for N in XS:
                    s = st(trades(p, "reg", mode, w, N, False), U78)
                    MN.append(dict(mode=mname, w=w, p=p, N=N, **s))
                    cells.append(f"{p[0]}N{N} {s['n']} {pct(s['avg'])} {pct(s['exc'])}")
            out(f"  {mname:<4} w{w * 100:.1f}% " + " | ".join(cells))
    MN = pd.DataFrame(MN)
    out("  스피어만(N 순서 vs 평균, 거래 10건 이상 칸):")
    for mname in ("LOW", "LIFT", "BRK"):
        cells = []
        for w in (0.005, 0.010, 0.015):
            for p in RANGE:
                g = MN[(MN["mode"] == mname) & (MN.w == w) & (MN.p == p) & (MN.n >= 10)]
                rho = pd.Series(g.N.values).corr(pd.Series(g.avg.values), method="spearman") if len(g) >= 3 else np.nan
                cells.append(f"w{w * 100:.1f}{p[0]} {rho:+.1f}")
        out(f"    {mname:<4} " + " ".join(cells))
    out("")
    # 연구자 2) 표 일부 재현: (a) 거래(N=15)를 D 구간으로 나눔
    out("  연구자 2) 표 재현 일부 — (a) 거래를 진입 순간 D 구간으로(정규장, 사용자78, RSI·8시). 칸: n 평균 초과")
    BUCK = ((15, 30), (30, 60), (60, 120), (120, 240), (240, 10**9))
    for mode, mname, w in ((1, "LIFT", 0.010), (2, "BRK", 0.015), (0, "LOW", 0.005)):
        for p in RANGE:
            t = trades(p, "reg", mode, w, 15, False)
            cells = []
            for a_, b_ in BUCK:
                s = st(t[(t.D >= a_) & (t.D < b_)], U78)
                cells.append(f"{a_}-{b_ if b_ < 10**9 else '+'} {s['n']} {pct(s['avg'])} {pct(s['exc'])}")
            out(f"    {mname} w{w * 100:.1f}% {PK[p]} " + " | ".join(cells))
    out("")

    # ---------------- 6) 사후 후보(연구자가 '검증 아님'으로 낸 것) 쏠림
    out("== 6) 사후 후보 쏠림 — 240분+ 박스 LIFT w1.0 / BRK w1.5, 고정 보유(정규장, 사용자78, 거래량 조건 없음, N≥240)")
    for mode, mname, w, hk in ((1, "LIFT", 0.010, "H4"), (1, "LIFT", 0.010, "D1"), (2, "BRK", 0.015, "H4"), (2, "BRK", 0.015, "D1")):
        for p in RANGE:
            t = trades(p, "reg", mode, w, 240, False)
            t = t[t.sym.isin(U78)].copy()
            t = t[np.isfinite(t[hk].astype(float))]
            if not len(t):
                continue
            t["exc"] = t[hk].astype(float) - t["r_" + hk]
            bd = t.groupby("day").exc.sum().sort_values()
            bs = t.groupby("sym").exc.sum().sort_values()
            w1 = t[t.day != bd.index[-1]]
            w2 = t[t.sym != bs.index[-1]]
            w3 = t[~t.day.isin(bd.index[-2:])]
            out(f"  {mname} w{w * 100:.1f} {hk} {PK[p]}: n{len(t)} 날{t.day.nunique()} 초과 {pct(t.exc.mean())} te {ct(t.day.values, t.exc.values):+.2f}"
                f" | 최대 날({date.fromordinal(int(bd.index[-1]))}) 빼면 {pct(w1.exc.mean())}, 상위 2일 빼면 {pct(w3.exc.mean())},"
                f" 최대 종목({bs.index[-1]}) 빼면 {pct(w2.exc.mean())}, 중앙값 {pct(t.exc.median())}")
    out(f"\n소요 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
