"""박스권 매매 연구 — 분봉 A: 장중 박스 바닥 버팀 (키 intra_A, 2026-10-09)

질문(사용자 2026-10-09): 최근 N분 최저가(박스 바닥)나 그 위 얇은 밴드를 깨지 않고 버티면 거기서 반등해 수익이 나는가?
반대 가설 B(바닥을 막 이탈한 점이 오히려 바닥)와 일반 박스 상단 돌파 C를 같은 매도 규칙으로 함께 비교한다.

박스(봉 k에서 쓰는 값, 모두 봉 k 이전 봉으로만 계산 — 미래 참조 없음):
  범위(scope) reg = 정규장 9:30~16:00 봉만 이어 붙인 열(창이 전날 정규장으로 이어짐), all = 프리·정규·애프터 모든 봉(4:00~20:00 ET).
  K = N분 / 봉 길이. L = 직전 K봉 저가 최솟값(박스 바닥), U = 직전 K봉 고가 최댓값(박스 상단).
  바닥 나이 = k - (L을 만든 봉) ≥ 15분(이어서 미끄러지는 신저가가 아니라 '만들어진 바닥'의 재시험만 본다). C는 U의 나이 ≥ 15분.
  밴드 = [L, L×(1+b)].
진입(종목당 한 번에 한 포지션, 매도한 봉 다음 봉부터 다시 살 수 있음):
  A_OPEN  봉 k-1 종가 > 밴드 상단, 봉 k 저가 ≤ 밴드 상단(닿음), 봉 k 종가 ≥ L(종가로 안 깨짐) → 봉 k+1 시가 매수(같은 날, 같은 범위의 바로 다음 봉).
  A_PLIM  봉 k-1 종가 > 밴드 상단일 때 봉 k에 밴드 상단 L×(1+b) 지정가를 미리 걸어 둠. 봉 k 저가 ≤ 지정가면 min(봉 k 시가, 지정가)에 체결.
          (확인 없이 '박스 바닥 지정가'. 깨짐은 손절 매도로만 처리)
  A_CLIM  (이웃 점검) A_OPEN과 같은 확인 봉 k 뒤 봉 k+1부터 그날 범위 끝까지 밴드 상단 지정가. 체결 전 어떤 봉 종가가 L 아래면 취소.
  B_OPEN  (비교) 봉 k 종가 < L×(1-b)(바닥을 종가로 이탈) → 봉 k+1 시가 매수.
  B_LIM   (비교) 봉 k에 L×(1-b) 지정가(바닥 아래, 이탈 순간에 체결). min(시가, 지정가).
  C_OPEN  (비교) 봉 k 종가 > U(상단 돌파) → 봉 k+1 시가 매수.
매도(같은 진입에 모두 적용, 매수 봉 다음 봉부터 판단):
  RSI8  지금 이격매수와 같음: 기준 RSI(5분봉 RSI9 / 3분봉 RSI14) > 70인 봉의 다음 봉 시가, 아니면 매수 뒤 첫 한국 아침 8시 이후 첫 봉 시가.
  H1/H4 매수 봉 시작 + 1/4시간 이후 첫 범위 봉 시가(reg는 정규장 봉, all은 모든 봉).
  TGT   목표 지정가 매도 + 종가 손절 + 아침 8시 정리. A: 목표 U(박스 상단), 손절 = 봉 종가 < L×(1-b)면 다음 봉 시가.
        B_OPEN: 목표 U, 손절 = 이탈 봉 저가×(1-b). B_LIM: 목표 U, 손절 = 지정가×(1-b). C: 목표 = 체결가×U/L(박스 높이만큼), 손절 = 박스 중간.
        목표는 봉 고가 ≥ 목표면 max(시가, 목표)에 체결. 같은 봉 판단 순서: 8시 → 손절(직전 봉 종가) → 목표. 매수 봉 안의 목표 체결은 세지 않는다.
        reg 범위에서는 목표·손절을 정규장 봉에서만 본다. 손절로 판 뒤에는 그 거래의 아침 8시까지 그 종목을 다시 사지 않는다.
비교 기준:
  무작위 = 같은 종목·같은 ET 날짜·같은 30분 시각대·같은 범위의 모든 봉 시가에 사서 같은 매도 규칙으로 판 평균(거래마다 짝지음).
          TGT는 목표·손절 거리(%)를 그 거래와 같게 둔다. 무작위 = '같은 시간대에 사서 같은 방식으로 보유'(매수 후 보유 비교).
  이격매수 = stop_lab.sim(3분봉 EMA60 -2% 지정가, RSI>70·아침 8시, 전날 +7% 다음 날 정규장 매수 안 함). 같은 날짜 범위.
  B·C = 같은 박스 길이·밴드·매도로 만든 반대·돌파 방식.
기간: 고르기 early = 7/14~9/8(5분봉; RSI9 근사), 확인 late = 9/10~10/6(3분봉, 원자료가 10/6 19:57 ET에서 끝나 10/7은 없음).
      봉 크기 점검 late5 = 같은 확인 날짜를 5분봉(고르기와 같은 봉)으로.
고르기: early·reg·사용자78·거래 30건 이상인 A 규칙 64개 중 '무작위 대비 초과의 날짜 군집 t' 최대. late는 확인만 한다.
시험 규칙: A 64개(N 4 × b 2 × 진입 2 × 매도 4) + 비교 B·C 21개(선택 규칙의 N·b에서 3방식 × 매도 4 = 12, 선택 매도에서 나머지 N 3 × 3방식 = 9)
          + 이웃 점검 2개(b 0.6%, A_CLIM) + 대조 6개(가짜 바닥: 진짜 바닥보다 0.6%·1.2% 위 수준에 같은 규칙 × A_OPEN·A_PLIM = 4,
          지정가가 0.1% 이상 관통해야 체결 × A_PLIM·B_LIM = 2) = 93개.
무조건 무작위 = 그 기간·범위의 모든 봉 시가에 사서 같은 매도(날짜 짝 없음). '아무 때나 사도 이만큼'의 절대 기준.
사용: py intra_A.py → ../results/intra_A.txt
"""
import os, sys, time
from datetime import date
import numpy as np, pandas as pd
from numba import njit

DIP = r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis\dip_reason\code"
sys.path.insert(0, DIP)
from stop_lab import load, sim as base_sim, UNIV  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "results", "intra_A.txt"))
PER = {"early": ("early", date(2026, 7, 14), date(2026, 9, 8)),
       "late": ("late", date(2026, 9, 10), date(2026, 10, 7)),
       "late5": ("early", date(2026, 9, 10), date(2026, 10, 7))}
PLAB = {"early": "고르기 7/14~9/8 5분봉", "late": "확인 9/10~10/6 3분봉", "late5": "확인 9/10~10/6 5분봉(봉 크기 점검)"}
NS = (60, 120, 240, 480)
BS = (0.003, 0.010)
EXITS = ("RSI8", "H1", "H4", "TGT")
MODE = {"RSI8": 0, "H1": 1, "H4": 2, "TGT": 3}
EXLAB = {"RSI8": "RSI>70·아침8시", "H1": "고정 1시간", "H4": "고정 4시간", "TGT": "목표·손절·8시"}
KIND = {"A_OPEN": 0, "A_PLIM": 1, "A_CLIM": 2, "B_OPEN": 3, "B_LIM": 4, "C_OPEN": 5}
SCOPES = ("reg", "all")
SCLAB = {"reg": "정규장", "all": "24시간"}
MINAGE = 15
RSI_X = 70.0
U78, U137 = UNIV["사용자78"], UNIV["전체137"]
UNS = (("사용자78", U78), ("전체137", U137))
WHY = {1: "RSI", 2: "8시", 3: "시간", 4: "손절", 5: "목표"}
_fh = None


def say(*a):
    global _fh
    if _fh is None:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        _fh = open(OUT, "w", encoding="utf-8")
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _fh.write(s + "\n")
    _fh.flush()


# ------------------------------------------------------------------ numba 핵심
@njit(cache=True)
def roll(lS, hS, K):
    """직전 K봉(k-K..k-1)의 저가 최솟값·그 봉, 고가 최댓값·그 봉."""
    n = len(lS)
    L = np.full(n, np.nan)
    U = np.full(n, np.nan)
    aL = np.full(n, -1, np.int64)
    aU = np.full(n, -1, np.int64)
    for k in range(K, n):
        mn = 1e300
        mx = -1e300
        im = -1
        ix = -1
        for q in range(k - K, k):
            if lS[q] < mn:
                mn = lS[q]
                im = q
            if hS[q] > mx:
                mx = hS[q]
                ix = q
        L[k] = mn
        U[k] = mx
        aL[k] = im
        aU[k] = ix
    return L, aL, U, aU


@njit(cache=True)
def cands(kind, oS, lS, cS, dS, L, aL, U, aU, b, minage, nxt, inp, shift, pen):
    n = len(oS)
    ek = np.empty(n, np.int64)
    fl = np.empty(n)
    Tp = np.empty(n)
    Sp = np.empty(n)
    m = 0
    armed = False
    alim = 0.0
    aT = 0.0
    aS = 0.0
    aref = 0.0
    aday = -1
    for k in range(1, n):
        Lk = L[k] * (1.0 + shift)
        if not (Lk > 0.0):
            armed = False
            continue
        ageL = k - aL[k]
        ageU = k - aU[k]
        if kind == 0:
            band = Lk * (1.0 + b)
            if ageL >= minage and cS[k - 1] > band and lS[k] <= band and cS[k] >= Lk and nxt[k] and inp[k + 1]:
                ek[m] = k + 1; fl[m] = oS[k + 1]; Tp[m] = U[k]; Sp[m] = Lk * (1.0 - b); m += 1
        elif kind == 1:
            band = Lk * (1.0 + b)
            if ageL >= minage and cS[k - 1] > band and lS[k] <= band * (1.0 - pen) and inp[k]:
                ek[m] = k; fl[m] = min(oS[k], band); Tp[m] = U[k]; Sp[m] = Lk * (1.0 - b); m += 1
        elif kind == 2:
            if armed:
                if dS[k] != aday:
                    armed = False
                elif lS[k] <= alim * (1.0 - pen):
                    if inp[k]:
                        ek[m] = k; fl[m] = min(oS[k], alim); Tp[m] = aT; Sp[m] = aS; m += 1
                    armed = False
                elif cS[k] < aref:
                    armed = False
            band = Lk * (1.0 + b)
            if ageL >= minage and cS[k - 1] > band and lS[k] <= band and cS[k] >= Lk:
                armed = True; alim = band; aT = U[k]; aS = Lk * (1.0 - b); aref = Lk; aday = dS[k]
        elif kind == 3:
            if ageL >= minage and cS[k] < Lk * (1.0 - b) and nxt[k] and inp[k + 1]:
                ek[m] = k + 1; fl[m] = oS[k + 1]; Tp[m] = U[k]; Sp[m] = lS[k] * (1.0 - b); m += 1
        elif kind == 4:
            lim = Lk * (1.0 - b)
            if ageL >= minage and cS[k - 1] > lim and lS[k] <= lim * (1.0 - pen) and inp[k]:
                ek[m] = k; fl[m] = min(oS[k], lim); Tp[m] = U[k]; Sp[m] = lim * (1.0 - b); m += 1
        elif kind == 5:
            if ageU >= minage and cS[k] > U[k] and nxt[k] and inp[k + 1]:
                f = oS[k + 1]
                ek[m] = k + 1; fl[m] = f; Tp[m] = f * U[k] / Lk; Sp[m] = 0.5 * (U[k] + Lk); m += 1
    return ek[:m], fl[:m], Tp[:m], Sp[:m]


@njit(cache=True)
def exit_one(e, T, S, mode, o, h, c, r, tk, dead_e, chk, xH, x):
    n = len(o)
    if mode == 1 or mode == 2:
        j = xH[e]
        if j < 0:
            return -1, 0.0, 0
        return j, o[j], 3
    for i in range(e + 1, n):
        if mode == 0:
            if r[i - 1] > x:
                return i, o[i], 1
            if tk[i] >= dead_e:
                return i, o[i], 2
        else:
            if tk[i] >= dead_e:
                return i, o[i], 2
            if chk[i] and chk[i - 1] and c[i - 1] < S:
                return i, o[i], 4
            if chk[i] and h[i] >= T:
                return i, max(o[i], T), 5
    return -1, 0.0, 0


@njit(cache=True)
def simulate(ce, cf, cT, cS, mode, o, h, c, r, tk, dead, chk, xH, x):
    m = len(ce)
    E = np.empty(m, np.int64)
    XI = np.empty(m, np.int64)
    F = np.empty(m)
    P = np.empty(m)
    W = np.empty(m, np.int64)
    TT = np.empty(m)
    SS = np.empty(m)
    q = 0
    busy = -1
    until = -1
    for j in range(m):
        e = ce[j]
        if e <= busy or tk[e] < until:
            continue
        xi, px, why = exit_one(e, cT[j], cS[j], mode, o, h, c, r, tk, dead[e], chk, xH, x)
        if xi < 0:
            busy = len(o) + 1
            continue
        E[q] = e; XI[q] = xi; F[q] = cf[j]; P[q] = px; W[q] = why; TT[q] = cT[j]; SS[q] = cS[j]
        q += 1
        busy = xi
        if why == 4:
            until = dead[e]
    return E[:q], XI[:q], F[:q], P[:q], W[:q], TT[:q], SS[:q]


@njit(cache=True)
def rnd_tgt(E, F, TT, SS, gs, ge, Xs, o, h, c, r, tk, dead, chk, xH, x):
    """TGT 무작위: 같은 (날짜, 30분) 칸의 범위 봉 시가에 사서 목표·손절 거리(%)를 같게 둔 평균."""
    m = len(E)
    out = np.full(m, np.nan)
    for j in range(m):
        e = E[j]
        a = gs[e]
        bb = ge[e]
        if a < 0:
            continue
        tr = TT[j] / F[j]
        sr = SS[j] / F[j]
        s = 0.0
        k = 0
        for p in range(a, bb):
            g = Xs[p]
            xi, px, why = exit_one(g, o[g] * tr, o[g] * sr, 3, o, h, c, r, tk, dead[g], chk, xH, x)
            if xi >= 0:
                s += px / o[g] - 1.0
                k += 1
        if k > 0:
            out[j] = s / k
    return out


@njit(cache=True)
def rnd_bars(Xs, mode, o, h, c, r, tk, dead, chk, xH, x):
    """범위 봉마다 시가 매수 → 매도 규칙(mode 0~2) 수익. 계산 불가면 nan."""
    n = len(o)
    out = np.full(n, np.nan)
    for p in range(len(Xs)):
        g = Xs[p]
        xi, px, why = exit_one(g, 0.0, 0.0, mode, o, h, c, r, tk, dead[g], chk, xH, x)
        if xi >= 0:
            out[g] = px / o[g] - 1.0
    return out


# ------------------------------------------------------------------ 종목·범위 문맥
def _ord(days):
    return np.array([d.toordinal() for d in days], dtype=np.int64)


def make_ctx(A, sc):
    o, h, l, c, r, tk, dead, reg, mod = A["o"], A["h"], A["l"], A["c"], A["r"], A["tk"], A["dead"], A["reg"], A["mod"]
    n, mins = len(o), A["mins"]
    dayi = A.get("_dayi")
    if dayi is None:
        dayi = A["_dayi"] = _ord(A["day"])
    Xs = np.nonzero(reg)[0].astype(np.int64) if sc == "reg" else np.arange(n, dtype=np.int64)
    chk = reg.copy() if sc == "reg" else np.ones(n, np.bool_)
    oS, hS, lS, cS, dS = o[Xs], h[Xs], l[Xs], c[Xs], dayi[Xs]
    nxt = np.zeros(len(Xs), np.bool_)
    nxt[:-1] = (dS[1:] == dS[:-1]) & ((Xs[1:] == Xs[:-1] + 1) | (sc == "reg"))
    R = {N: roll(lS, hS, N // mins) for N in NS}
    xH = {}
    tS = tk[Xs]
    for ex, hm in (("H1", 60), ("H4", 240)):
        kk = np.searchsorted(tS, tk + np.int64(hm) * 60 * 10**9, side="left")
        xH[ex] = np.where(kk < len(Xs), Xs[np.minimum(kk, len(Xs) - 1)], -1).astype(np.int64)
    dummy = np.full(n, -1, np.int64)
    key = dayi * 48 + mod // 30
    kS = key[Xs]
    brk = np.r_[True, kS[1:] != kS[:-1]]
    gid = np.cumsum(brk) - 1
    starts = np.nonzero(brk)[0]
    ends = np.r_[starts[1:], len(Xs)]
    gs = np.full(n, -1, np.int64)
    ge = np.full(n, -1, np.int64)
    gs[Xs] = starts[gid]
    ge[Xs] = ends[gid]
    # 모드 0~2 무작위: 봉별 수익 → (날짜, 30분) 칸 평균
    RB, RM = {}, {}
    for ex in ("RSI8", "H1", "H4"):
        xh = xH.get(ex, dummy)
        rb = rnd_bars(Xs, MODE[ex], o, h, c, r, tk, dead, chk, xh, RSI_X)
        RB[ex] = rb
        v = rb[Xs]
        ok = np.isfinite(v)
        sm = np.bincount(gid[ok], weights=v[ok], minlength=len(starts))
        ct = np.bincount(gid[ok], minlength=len(starts))
        with np.errstate(invalid="ignore", divide="ignore"):
            gm = np.where(ct > 0, sm / np.maximum(ct, 1), np.nan)
        rm = np.full(n, np.nan)
        rm[Xs] = gm[gid]
        RM[ex] = rm
    return dict(sc=sc, Xs=Xs, chk=chk, oS=oS, lS=lS, cS=cS, dS=dS, nxt=nxt, R=R, xH=xH, dummy=dummy, gs=gs, ge=ge,
                RB=RB, RM=RM, dayi=dayi, mins=mins)


def run_rule(A, C, kind, N, b, ex, lo, hi, shift=0.0, pen=0.0):
    """규칙 하나·종목 하나·범위 하나 → 거래 배열 dict(기간 [lo, hi] 매수만)."""
    o, h, c, r, tk, dead = A["o"], A["h"], A["c"], A["r"], A["tk"], A["dead"]
    inp = (C["dS"] >= lo) & (C["dS"] <= hi)
    L, aL, U, aU = C["R"][N]
    minage = int(np.ceil(MINAGE / C["mins"]))
    ek, fl, Tp, Sp = cands(KIND[kind], C["oS"], C["lS"], C["cS"], C["dS"], L, aL, U, aU, b, minage, C["nxt"], inp, shift, pen)
    if not len(ek):
        return None
    ce = C["Xs"][ek]
    xh = C["xH"].get(ex, C["dummy"])
    E, XI, F, P, W, TT, SS = simulate(ce, fl, Tp, Sp, MODE[ex], o, h, c, r, tk, dead, C["chk"], xh, RSI_X)
    if not len(E):
        return None
    if ex == "TGT":
        rnd = rnd_tgt(E, F, TT, SS, C["gs"], C["ge"], C["Xs"], o, h, c, r, tk, dead, C["chk"], C["dummy"], RSI_X)
    else:
        rnd = C["RM"][ex][E]
    return dict(dayi=C["dayi"][E], e=E, mod=A["mod"][E], ret=P / F - 1, rnd=rnd, why=W,
                hold=(tk[XI] - tk[E]) / 6e10, tgt=TT / F - 1, stp=SS / F - 1)


# ------------------------------------------------------------------ 집계
def dct(day, x):
    s = pd.Series(np.asarray(x, float)).groupby(np.asarray(day)).sum()
    if len(s) < 3 or s.std(ddof=1) == 0:
        return np.nan
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def stat(t, syms=None):
    if t is None or not len(t):
        return dict(n=0, avg=np.nan, win=np.nan, worst=np.nan, sum=0.0, t=np.nan, rnd=np.nan, exc=np.nan, te=np.nan, days=0)
    if syms is not None:
        t = t[t.sym.isin(syms)]
    if not len(t):
        return dict(n=0, avg=np.nan, win=np.nan, worst=np.nan, sum=0.0, t=np.nan, rnd=np.nan, exc=np.nan, te=np.nan, days=0)
    ret = t.ret.values
    rnd = t.rnd.values
    ok = np.isfinite(rnd)
    exc = ret[ok] - rnd[ok]
    return dict(n=len(t), avg=ret.mean(), win=(ret > 0).mean(), worst=ret.min(), sum=ret.sum(), t=dct(t.dayi.values, ret),
                rnd=np.nanmean(rnd) if ok.any() else np.nan, exc=exc.mean() if ok.any() else np.nan,
                te=dct(t.dayi.values[ok], exc), days=t.dayi.nunique())


def fp(v, k=2):
    return "  n/a" if v is None or not np.isfinite(v) else f"{v * 100:+.{k}f}%"


def ft(v):
    return " n/a" if v is None or not np.isfinite(v) else f"{v:+.1f}"


def line(s):
    return (f"n {s['n']:>5} 평균 {fp(s['avg'])} 승률 {s['win'] if np.isfinite(s['win']) else 0:4.0%} 최악 {fp(s['worst'], 1)} "
            f"합 {s['sum']:+7.2f} t {ft(s['t'])} | 무작위 {fp(s['rnd'])} 초과 {fp(s['exc'])} te {ft(s['te'])}")


def rid(kind, N, b, ex, shift=0.0, pen=0.0):
    s = f"{kind}|N{N}|b{b * 100:.1f}|{ex}"
    if shift:
        s += f"|가짜바닥+{shift * 100:.1f}%"
    if pen:
        s += f"|관통{pen * 100:.1f}%체결"
    return s


def evaluate(rules):
    """rules: [(kind, N, b, ex)] → {rid: {(pk, sc): DataFrame}}. 이격매수 기준도 같이 만든다."""
    out = {rid(*ru): {} for ru in rules}
    base = {}
    unc = {}
    t0 = time.time()
    for aset, pks in (("early", ("early", "late5")), ("late", ("late",))):
        S = load(aset)
        acc = {rid(*ru): {(pk, sc): [] for pk in pks for sc in SCOPES} for ru in rules}
        bacc = {(pk, sc): [] for pk in pks for sc in SCOPES}
        uacc = {(pk, sc, ex): [] for pk in pks for sc in SCOPES for ex in ("RSI8", "H1", "H4")}
        for sym, A in S.items():
            for sc in SCOPES:
                C = make_ctx(A, sc)
                for pk in pks:
                    lo, hi = PER[pk][1].toordinal(), PER[pk][2].toordinal()
                    for ru in rules:
                        d = run_rule(A, C, *ru[:4], lo, hi, *ru[4:])
                        if d is not None:
                            d["sym"] = sym
                            acc[rid(*ru)][(pk, sc)].append(pd.DataFrame(d))
                    # 이격매수 기준(같은 날짜, 매수 범위 = sc)
                    bt = base_sim(A, None, True, sc == "reg")
                    if bt:
                        ei = np.array([x[0] for x in bt], np.int64)
                        ret = np.array([x[3] / x[2] - 1 for x in bt])
                        dd = C["dayi"][ei]
                        m = (dd >= lo) & (dd <= hi)
                        if m.any():
                            bacc[(pk, sc)].append(pd.DataFrame(dict(sym=sym, dayi=dd[m], e=ei[m], mod=A["mod"][ei[m]], ret=ret[m],
                                                                    rnd=C["RM"]["RSI8"][ei[m]])))
                    # 무조건 무작위(범위 안 모든 봉 시가 매수)
                    Xs = C["Xs"]
                    dX = C["dayi"][Xs]
                    mX = Xs[(dX >= lo) & (dX <= hi)]
                    for ex in ("RSI8", "H1", "H4"):
                        v = C["RB"][ex][mX]
                        v = v[np.isfinite(v)]
                        uacc[(pk, sc, ex)].append((sym, v.sum(), len(v)))
        for k, dd in acc.items():
            for kk, lst in dd.items():
                out[k][kk] = pd.concat(lst, ignore_index=True) if lst else pd.DataFrame(columns=["sym", "dayi", "ret", "rnd"])
        for kk, lst in bacc.items():
            base[kk] = pd.concat(lst, ignore_index=True)
        for kk, lst in uacc.items():
            unc[kk] = lst
        print(f"  {aset} 완료 {time.time() - t0:.0f}s", flush=True)
    return out, base, unc


def unc_mean(lst, syms):
    s = sum(x[1] for x in lst if x[0] in syms)
    k = sum(x[2] for x in lst if x[0] in syms)
    return s / k if k else np.nan


def wk(dayi):
    return np.array([date.fromordinal(int(x)).isocalendar()[1] for x in dayi])


def main():
    T0 = time.time()
    say("박스권 매매 연구 — 분봉 A: 장중 박스 바닥 버팀 [키 intra_A] 2026-10-09")
    say(__doc__.split("사용:")[0].strip())
    say("")
    say("표기: n 거래수, 평균 거래당 수익, 승률, 최악, 합 = 거래 수익 합(1거래 1단위), t = 수익의 날짜 군집 t,")
    say("      무작위 = 같은 종목·날짜·30분 칸 무작위 진입 평균, 초과 = 평균 - 무작위, te = 초과의 날짜 군집 t. 수수료 0.")
    say("")

    # ---------------------------------------------------------------- 1단계: A 64개
    A_RULES = [(k, N, b, ex) for N in NS for b in BS for k in ("A_OPEN", "A_PLIM") for ex in EXITS]
    print("A 규칙 평가 중...", flush=True)
    RA, BASE, UNC = evaluate(A_RULES)

    say("== 0) 비교 기준")
    say("  (c) 지금 이격매수 지표 — 매수 범위별, 매도 RSI>70·아침 8시. 무작위 = 같은 칸 무작위 진입 + 같은 매도")
    for pk in PER:
        for sc in SCOPES:
            for un, us in UNS:
                s = stat(BASE[(pk, sc)], us)
                say(f"  {PLAB[pk]:<26} {SCLAB[sc]:<4} {un:<6} {line(s)}")
    say("")
    say("  무조건 무작위 = 범위 안 모든 봉 시가에 사서 매도 규칙대로 판 평균(그 기간의 '아무 때나 사서 보유')")
    for pk in PER:
        for sc in SCOPES:
            cells = []
            for un, us in UNS:
                cells.append(f"{un} " + " ".join(f"{ex} {fp(unc_mean(UNC[(pk, sc, ex)], us), 3)}" for ex in ("RSI8", "H1", "H4")))
            say(f"  {PLAB[pk]:<26} {SCLAB[sc]:<4} " + " | ".join(cells))
    say("")

    # ---------------------------------------------------------------- 2) A 격자 표
    say("== 1) A 규칙 64개 — 고르기(early)·확인(late) 나란히. 정규장·사용자78. 확인 열은 고르기에 쓰지 않았다.")
    say("   열: 규칙 | early n 평균 초과 te | late n 평균 초과 te | late 전체137 초과 | late 24시간·사용자78 초과 | 평균-무조건무작위 early/late")
    rows = []
    for ru in A_RULES:
        k = rid(*ru)
        e = stat(RA[k][("early", "reg")], U78)
        lt = stat(RA[k][("late", "reg")], U78)
        l137 = stat(RA[k][("late", "reg")], U137)
        e137 = stat(RA[k][("early", "reg")], U137)
        lall = stat(RA[k][("late", "all")], U78)
        eall = stat(RA[k][("early", "all")], U78)
        l5 = stat(RA[k][("late5", "reg")], U78)
        eu = unc_mean(UNC[("early", "reg", ru[3])], U78) if ru[3] != "TGT" else np.nan
        lu = unc_mean(UNC[("late", "reg", ru[3])], U78) if ru[3] != "TGT" else np.nan
        rows.append(dict(rule=k, kind=ru[0], N=ru[1], b=ru[2], ex=ru[3], e_n=e["n"], e_avg=e["avg"], e_exc=e["exc"], e_te=e["te"],
                         e_unc=eu, l_unc=lu,
                         e_sum=e["sum"], l_n=lt["n"], l_avg=lt["avg"], l_exc=lt["exc"], l_te=lt["te"], l_sum=lt["sum"],
                         e137_exc=e137["exc"], l137_exc=l137["exc"], eall_exc=eall["exc"], lall_exc=lall["exc"], l5_exc=l5["exc"]))
    G = pd.DataFrame(rows)
    for _, x in G.iterrows():
        say(f"  {x.rule:<26} | {x.e_n:>5} {fp(x.e_avg)} {fp(x.e_exc)} {ft(x.e_te)} | {x.l_n:>5} {fp(x.l_avg)} {fp(x.l_exc)} {ft(x.l_te)}"
            f" | {fp(x.l137_exc)} | {fp(x.lall_exc)} | {fp(x.e_avg - x.e_unc)} / {fp(x.l_avg - x.l_unc)}")
    say("")
    elig = G[G.e_n >= 30]
    best = elig.sort_values("e_te", ascending=False).iloc[0]
    say("  고르기 상위 5개(early·정규장·사용자78, 초과 te 순):")
    for _, x in elig.sort_values("e_te", ascending=False).head(5).iterrows():
        say(f"    {x.rule:<26} early 초과 {fp(x.e_exc)} te {ft(x.e_te)} → late 초과 {fp(x.l_exc)} te {ft(x.l_te)}")
    both = G[(G.e_exc > 0) & (G.l_exc > 0)]
    both137 = G[(G.e137_exc > 0) & (G.l137_exc > 0)]
    bothall = G[(G.eall_exc > 0) & (G.lall_exc > 0)]
    say(f"  두 기간 모두 무작위보다 나은 A 규칙: 정규장·사용자78 {len(both)}/64, 정규장·전체137 {len(both137)}/64, 24시간·사용자78 {len(bothall)}/64")
    both_t = G[(G.e_te > 1) & (G.l_te > 1)]
    say(f"  두 기간 모두 초과 te > 1: {len(both_t)}/64 " + (", ".join(both_t.rule) if len(both_t) else ""))
    # 이격매수 평균보다 나은 규칙 수
    bE = stat(BASE[("early", "reg")], U78)["avg"]
    bL = stat(BASE[("late", "reg")], U78)["avg"]
    nb = G[(G.e_avg > bE) & (G.l_avg > bL)]
    say(f"  두 기간 모두 거래당 평균이 이격매수(정규장·사용자78 early {fp(bE)}, late {fp(bL)})보다 높은 A 규칙: {len(nb)}/64")
    G2 = G[G.ex != "TGT"]
    nu = G2[(G2.e_avg > G2.e_unc) & (G2.l_avg > G2.l_unc)]
    say(f"  두 기간 모두 거래당 평균이 무조건 무작위(같은 매도, 날짜 짝 없음)보다 높은 A 규칙: {len(nu)}/{len(G2)}(TGT 제외) "
        f"| early만 {(G2.e_avg > G2.e_unc).sum()}, late만 {(G2.l_avg > G2.l_unc).sum()}")
    say(f"  late 초과가 0 이하인 A 규칙: {(G.l_exc <= 0).sum()}/64, late 초과 중앙값 {fp(G.l_exc.median(), 3)}, early 초과 중앙값 {fp(G.e_exc.median(), 3)}")
    say("")
    # 차원별 평균(초과): 박스 길이·밴드·진입·매도
    say("  차원별 평균 초과(정규장·사용자78, 규칙들 단순 평균): early / late")
    for col in ("N", "b", "kind", "ex"):
        g = G.groupby(col)[["e_exc", "l_exc"]].mean()
        say("    " + col + ": " + " | ".join(f"{i}: {fp(v.e_exc, 3)} / {fp(v.l_exc, 3)}" for i, v in g.iterrows()))
    say("")

    kind, N, b, ex = best.kind, int(best.N), float(best.b), best.ex
    BEST = rid(kind, N, b, ex)
    say(f"== 2) 고른 규칙: {BEST}  (early·정규장·사용자78 초과 te 최대)")
    for pk in PER:
        for sc in SCOPES:
            for un, us in UNS:
                s = stat(RA[BEST][(pk, sc)], us)
                say(f"  {PLAB[pk]:<26} {SCLAB[sc]:<4} {un:<6} {line(s)}")
    say("")

    # ---------------------------------------------------------------- 2단계: 비교 B·C + 이웃
    CMP = []
    for k2 in ("B_OPEN", "B_LIM", "C_OPEN"):
        for e2 in EXITS:
            CMP.append((k2, N, b, e2))
        for N2 in NS:
            if N2 != N:
                CMP.append((k2, N2, b, ex))
    NBR = [(kind, N, 0.006, ex), ("A_CLIM", N, b, ex)]
    CTL = [("A_OPEN", N, b, ex, 0.006), ("A_OPEN", N, b, ex, 0.012), ("A_PLIM", N, b, ex, 0.006), ("A_PLIM", N, b, ex, 0.012),
           ("A_PLIM", N, b, ex, 0.0, 0.001), ("B_LIM", N, b, ex, 0.0, 0.001)]
    # A 쪽 같은 N·b의 다른 매도, 다른 N의 같은 매도는 이미 A 64개에 있음
    print("비교·이웃 규칙 평가 중...", flush=True)
    RC, _, _ = evaluate(CMP + NBR + CTL)
    RALL = {**RA, **RC}
    n_tested = len(A_RULES) + len(CMP) + len(NBR) + len(CTL)

    say(f"== 3) 같은 매도 규칙끼리 A·B·C (박스 N={N}분, 밴드 b={b * 100:.1f}%). 칸: n / 평균 / 초과 / te")
    akinds = ("A_OPEN", "A_PLIM", "B_OPEN", "B_LIM", "C_OPEN")
    for e2 in EXITS:
        say(f"  -- 매도 {EXLAB[e2]}")
        for pk in ("early", "late"):
            for sc in SCOPES:
                cells = []
                for k2 in akinds:
                    s = stat(RALL[rid(k2, N, b, e2)][(pk, sc)], U78)
                    cells.append(f"{k2} {s['n']:>4} {fp(s['avg'])} {fp(s['exc'])} {ft(s['te'])}")
                say(f"    {pk:<5} {SCLAB[sc]:<4} 사용자78 | " + " | ".join(cells))
            s137 = []
            for k2 in akinds:
                s = stat(RALL[rid(k2, N, b, e2)][(pk, "reg")], U137)
                s137.append(f"{k2} {s['n']:>4} {fp(s['avg'])} {fp(s['exc'])} {ft(s['te'])}")
            say(f"    {pk:<5} 정규장 전체137 | " + " | ".join(s137))
    say("")
    say(f"  박스 길이별 A·B·C (매도 {EXLAB[ex]}, b={b * 100:.1f}%, 정규장·사용자78). 칸: 평균 / 초과")
    for N2 in NS:
        for pk in ("early", "late"):
            cells = []
            for k2 in akinds:
                s = stat(RALL[rid(k2, N2, b, ex)][(pk, "reg")], U78)
                cells.append(f"{k2} {s['n']:>4} {fp(s['avg'])} {fp(s['exc'])}")
            say(f"    N{N2:<4} {pk:<5} | " + " | ".join(cells))
    say("")

    # ---------------------------------------------------------------- 4) 최근 장세: 주별
    say("== 4) 최근 장세 — 주별(ISO 주, 정규장·사용자78). 칸: n / 평균 / 초과. 마지막 줄은 이격매수(정규장 매수).")
    for pk in ("early", "late"):
        tabs = {k2: RALL[rid(k2, N, b, ex)][(pk, "reg")] for k2 in akinds}
        tabs["이격매수"] = BASE[(pk, "reg")]
        allw = sorted(set(np.concatenate([wk(t[t.sym.isin(U78)].dayi.values) for t in tabs.values() if len(t)])))
        say(f"  {PLAB[pk]}")
        for w in allw:
            cells = []
            for k2, t in tabs.items():
                t = t[t.sym.isin(U78)]
                tw = t[wk(t.dayi.values) == w] if len(t) else t
                s = stat(tw)
                cells.append(f"{k2} {s['n']:>3} {fp(s['avg'])} {fp(s['exc'])}")
            d0 = date.fromisocalendar(2026, int(w), 1)
            say(f"    {d0.month}/{d0.day}주 | " + " | ".join(cells))
    say("")

    # ---------------------------------------------------------------- 5) 고른 규칙 세부
    say(f"== 5) 고른 규칙 {BEST} 세부")
    for pk in ("early", "late"):
        for sc in SCOPES:
            t = RALL[BEST][(pk, sc)]
            t = t[t.sym.isin(U78)]
            if not len(t):
                continue
            why = t.why.map(WHY).value_counts(normalize=True)
            awake = (t["mod"] >= 1140) | (t["mod"] < 660)
            sa, ss = stat(t[awake]), stat(t[~awake])
            day = t.groupby("dayi").ret.sum().sort_values(ascending=False)
            top2 = day.head(2).sum()
            say(f"  {PLAB[pk]} {SCLAB[sc]} 사용자78: 매도 이유 " + ", ".join(f"{k} {v:.0%}" for k, v in why.items())
                + f" | 보유 중앙 {t.hold.median():.0f}분 | 목표 거리 중앙 {fp(t.tgt.median())} 손절 거리 중앙 {fp(t.stp.median())}")
            say(f"      깨어 있는 시간(한국 08~24시 = ET 19:00~11:00) 매수 {line(sa)}")
            say(f"      자는 시간(ET 11:00~19:00) 매수              {line(ss)}")
            say(f"      수익 합 {t.ret.sum():+.2f} 중 상위 2일 {top2:+.2f}, 상위 2일 빼면 평균 {fp(t[~t.dayi.isin(day.head(2).index)].ret.mean())}")
    say("")
    say("  이웃 점검(같은 N·매도):")
    for ru in NBR:
        k = rid(*ru)
        cells = []
        for pk in ("early", "late"):
            s = stat(RALL[k][(pk, "reg")], U78)
            cells.append(f"{pk} n {s['n']} 평균 {fp(s['avg'])} 초과 {fp(s['exc'])} te {ft(s['te'])}")
        say(f"    {k:<26} " + " | ".join(cells))
    for N2 in NS:
        if N2 == N:
            continue
        k = rid(kind, N2, b, ex)
        cells = []
        for pk in ("early", "late"):
            s = stat(RALL[k][(pk, "reg")], U78)
            cells.append(f"{pk} n {s['n']} 평균 {fp(s['avg'])} 초과 {fp(s['exc'])} te {ft(s['te'])}")
        say(f"    {k:<26} " + " | ".join(cells))
    say("")

    say("  대조(같은 N·b·매도). 가짜 바닥 = 진짜 바닥 L 대신 L×(1+δ)를 바닥으로 놓고 같은 규칙. 진짜 바닥이 특별하면 가짜는 나빠야 한다.")
    say("  관통 체결 = 지정가보다 0.1% 이상 내려간 봉에서만 체결(닿기만 한 봉은 체결 안 된 것으로 봄).")
    for ru in [(kind2, N, b, ex) for kind2 in ("A_OPEN", "A_PLIM", "B_LIM")] + CTL:
        k = rid(*ru)
        for sc in SCOPES:
            cells = []
            for pk in ("early", "late"):
                s = stat(RALL[k][(pk, sc)], U78)
                cells.append(f"{pk} n {s['n']:>5} 평균 {fp(s['avg'])} 초과 {fp(s['exc'])} te {ft(s['te'])}")
            say(f"    {k:<40} {SCLAB[sc]:<4} " + " | ".join(cells))
    say("")

    # ---------------------------------------------------------------- 저장용 표
    G.to_csv(os.path.join(os.path.dirname(OUT), "intra_A_grid.csv"), index=False, encoding="utf-8-sig")
    say(f"시험 규칙 수: {n_tested}개 (A 64 + 비교 B·C {len(CMP)} + 이웃 {len(NBR)} + 대조 {len(CTL)}). 고르기 대상은 A 64개뿐.")
    say(f"격자 표: {os.path.join(os.path.dirname(OUT), 'intra_A_grid.csv')}")
    say(f"실행 시간 {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
