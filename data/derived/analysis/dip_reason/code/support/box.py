"""지지 버팀 연구 — 장중 박스(횡보 바닥) 버팀(2026-10-09, 키 box).
질문: 가격이 좁은 바닥 띠 안에서 오래 버틸수록 위로 돌아설 가능성이 커지나? 버팀 끝자락(아직 많이 오르기 전)에 사면 이득인가?
반대 가능성(오래 두드리면 바닥이 약해져 깨짐)도 같이 본다.

박스 정의(봉 단위, 미래 참조 없음 — 봉 j의 상태는 봉 j 종가까지로 정한다):
  구간(세그먼트): 정규장 범위 = 같은 ET 날짜의 9:30~16:00 정규장 봉만(9:30에 새로 시작), 24시간 범위 = 같은 ET 날짜 4:00~20:00 모든 봉.
  박스는 세그먼트 첫 봉에서 시작한다. 이어지는 봉 j마다, 박스 안 저가의 최솟값 F와 최댓값 Lm에 대해
    l[j] > F*(1+w)   → 저가가 띠 위로 떠남 = '위로 끝남'(LIFT 이벤트). 박스는 봉 j에서 새로 시작.
    l[j] < Lm/(1+w)  → 저가가 띠 아래로 깨짐 = '아래로 끝남'. 박스는 봉 j에서 새로 시작.
    그 외            → 박스 유지(F, Lm, 박스 상단 T = 박스 안 고가 최댓값 갱신).
  즉 박스 = 저가들의 폭(max/min-1)이 w 이하로 유지된 가장 긴 연속 구간. 버틴 시간 D = 박스 봉 수 × 봉 길이(분).
진입(모두 '봉 j-1까지의 박스 상태'로 판단, 한 박스에서 한 번만, 종목당 한 번에 한 포지션):
  BRK  봉 j 종가 > T(j-1) 이고 ≤ T*1.01(돌파 직후)이면 봉 j+1 시가 매수, 단 그 시가도 ≤ T*1.01일 때만.
  LIFT 봉 j 저가가 띠 위로 떠남(l[j] > F*(1+w)) 이고 c[j] ≤ T*1.01이면 봉 j+1 시가 매수(시가 ≤ T*1.01). = '버팀 끝자락'
  LOW  봉 j-1 박스가 D ≥ N이면 봉 j에 지정가 F*(1+w/4)(박스 안 하단 근처). l[j] ≤ 지정가면 min(시가, 지정가)에 체결.
  매수 봉(BRK·LIFT는 j+1, LOW는 j)은 박스와 같은 세그먼트여야 한다.
  거래량 조건 vdec(정규장만 의미): 박스 뒤쪽 절반의 시각 보정 거래량 평균 < 앞쪽 절반 평균.
    시각 보정 거래량 = 봉 거래량 / 지난 10거래일(최소 5일) 같은 시각 봉 거래량 평균(오늘 제외).
매도(같은 진입에 모두 적용):
  RSI·8시 = 지금 지표와 같음: 기준 RSI(앞 기간 5분봉 RSI9, 뒤 기간 3분봉 RSI14) > 70인 봉 다음 봉 시가, 또는 매수 뒤 첫 한국 아침 8시 이후 첫 봉 시가.
  고정 1시간·4시간·1일·3일 = 매수 봉 시작 + H 이후 첫 봉 시가(정규장 범위는 정규장 봉만, 24시간 범위는 모든 봉). 데이터 밖이면 제외.
비교 기준:
  (a) 버팀 무시 = 같은 진입 방식·같은 w, D ≥ 15분(박스가 막 생긴 상태)이면 바로 진입, 거래량 조건 없음.
  (b) 무작위 = 같은 종목·같은 기간·같은 범위에서 같은 30분 시각대 모든 봉 시가에 샀을 때 같은 매도 규칙의 평균(거래마다 짝지음).
  (c) 지금 이격매수 지표 거래(stop_lab.sim, 같은 날짜 범위).
고르기: 앞 기간(7/14~9/8, 5분봉)·정규장·사용자78·RSI·8시 매도에서 거래 30건 이상 규칙 중 '무작위 대비 초과수익의 날짜 군집 t' 최대.
뒤 기간(9/10~10/6, 3분봉; 데이터가 10/6 19:59 ET에서 끝남)은 확인만 한다.
사용: py support/box.py   (code 폴더에서) → ../results/support/box.txt
"""
import os, sys, time
from datetime import date
import numpy as np, pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, sim as base_sim, UNIV, RAW, X as RSI_X

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "support", "box.txt"))
PER = {"early": (date(2026, 7, 14), date(2026, 9, 8)), "late": (date(2026, 9, 10), date(2026, 10, 7))}
PLAB = {"early": "앞(고르기) 7/14~9/8 5분봉", "late": "뒤(확인) 9/10~10/6 3분봉"}
PSH = {"early": "앞", "late": "뒤"}
WS = (0.005, 0.010, 0.015)
NS = (30, 60, 120, 240)
N0 = 15
ENT = ("BRK", "LIFT", "LOW")
VOLS = ("any", "vdec")
EXITS = ("RSI8", "H1", "H4", "D1", "D3")
EXLAB = {"RSI8": "RSI·8시", "H1": "고정1시간", "H4": "고정4시간", "D1": "고정1일", "D3": "고정3일"}
HMIN = {"H1": 60, "H4": 240, "D1": 1440, "D3": 4320}
SCOPES = ("reg", "all")
SCLAB = {"reg": "정규장", "all": "24시간"}
BUCK = ((15, 30), (30, 60), (60, 120), (120, 240), (240, 10**9))
BLAB = ["15-30분", "30-60분", "60-120분", "120-240분", "240분+"]
U78, U137 = UNIV["사용자78"], UNIV["전체137"]
_fh = None


def say(*a):
    global _fh
    if _fh is None:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        _fh = open(OUT, "w", encoding="utf-8")
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _fh.write(s + "\n"); _fh.flush()


@njit(cache=False)
def box_state(l, h, valid, newseg, w):
    n = len(l)
    S = np.full(n, -1, np.int64)
    F = np.full(n, np.nan); Lm = np.full(n, np.nan); T = np.full(n, np.nan)
    E = np.zeros(n, np.int8)
    s = -1; f = 0.0; lm = 0.0; t = 0.0
    for j in range(n):
        if not valid[j]:
            s = -1
            continue
        if newseg[j] or s < 0:
            s = j; f = l[j]; lm = l[j]; t = h[j]
        elif l[j] > f * (1 + w):
            E[j] = 1; s = j; f = l[j]; lm = l[j]; t = h[j]
        elif l[j] < lm / (1 + w):
            E[j] = -1; s = j; f = l[j]; lm = l[j]; t = h[j]
        else:
            if l[j] < f:
                f = l[j]
            if l[j] > lm:
                lm = l[j]
            if h[j] > t:
                t = h[j]
        S[j] = s; F[j] = f; Lm[j] = lm; T[j] = t
    return S, F, Lm, T, E


@njit(cache=False)
def greedy(e, box, x1, n):
    acc = np.zeros(len(e), np.bool_)
    busy = -1
    lastbox = -2
    for k in range(len(e)):
        if e[k] <= busy or box[k] == lastbox:
            continue
        acc[k] = True
        lastbox = box[k]
        busy = x1[e[k]] if x1[e[k]] >= 0 else n + 1
    return acc


def _ord(days):
    return np.array([d.toordinal() for d in days], dtype=np.int64)


def norm_vol(A, dayi):
    """정규장 봉의 시각 보정 거래량(지난 10거래일(최소 5일) 같은 시각 평균 대비, 오늘 제외).
    기준 평균은 5분봉 파일(7/14부터)로 만든다: 1분봉은 9/9부터라 뒤 기간 첫 주에 기준이 비기 때문.
    3분봉은 그 봉이 속한 5분 칸 평균 × 3/5를 기준으로 쓴다(앞 기간 5분봉은 자기 자신과 같은 자료)."""
    reg, mins = A["reg"], A["mins"]
    v = np.where(reg, A["v"], np.nan)
    m5 = pd.read_parquet(os.path.join(RAW, f"5m_{A['sym']}.parquet"))
    m5 = m5[m5.High > 0]
    i5 = m5.index
    mod5 = i5.hour * 60 + i5.minute
    r5 = (mod5 >= 570) & (mod5 < 960)
    d5 = np.array([x.toordinal() for x in i5.date])
    tab = pd.DataFrame({"d": d5[r5], "m": mod5[r5], "v": m5.Volume.values[r5].astype(float)}).pivot_table(
        index="d", columns="m", values="v", aggfunc="last")
    base = tab.shift(1).rolling(10, min_periods=5).mean()
    b = base.stack()
    key = pd.MultiIndex.from_arrays([dayi, (A["mod"] // 5) * 5])
    bb = b.reindex(key).values * (mins / 5.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        nv = v / bb
    nv[~np.isfinite(nv)] = np.nan
    return nv


def seg_range_mean(cs, cn, a, b):
    """[a, b] 구간 평균(NaN 제외). cs·cn은 앞에 0을 붙인 누적합."""
    s = cs[b + 1] - cs[a]
    k = cn[b + 1] - cn[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(k > 0, s / np.maximum(k, 1), np.nan)


P = {}


def prep(period, sym, A):
    n, mins = len(A["o"]), A["mins"]
    o, h, l, c, r, tk, dead, reg, mod = A["o"], A["h"], A["l"], A["c"], A["r"], A["tk"], A["dead"], A["reg"], A["mod"]
    dayi = _ord(A["day"])
    lo, hi = PER[period]
    inper = (dayi >= lo.toordinal()) & (dayi <= hi.toordinal())
    dchg = np.r_[True, dayi[1:] != dayi[:-1]]
    seg = {"all": (np.ones(n, bool), dchg), "reg": (reg.copy(), reg & np.r_[True, (~reg[:-1]) | (dayi[1:] != dayi[:-1])])}
    # 매도 위치(진입 봉 e → 매도 봉). -1 = 데이터 밖
    pos = np.nonzero(r > RSI_X)[0]
    ar = np.arange(n)
    if len(pos):
        k = np.searchsorted(pos, ar)
        krsi = np.where(k < len(pos), pos[np.minimum(k, len(pos) - 1)] + 1, n + 10)
    else:
        krsi = np.full(n, n + 10)
    idead = np.searchsorted(tk, dead, side="left")
    x1 = np.minimum(krsi, idead)
    x1 = np.where(x1 < n, x1, -1)
    X = {}
    for sc in SCOPES:
        Xs = np.nonzero(reg)[0] if sc == "reg" else ar
        d = {"RSI8": x1}
        for ex, hm in HMIN.items():
            kk = np.searchsorted(tk[Xs], tk + np.int64(hm) * 60 * 10**9, side="left")
            d[ex] = np.where(kk < len(Xs), Xs[np.minimum(kk, len(Xs) - 1)], -1)
        X[sc] = d
    # 무작위 기준: 범위 안 모든 봉 시가 매수 → 같은 매도. 30분 시각대별 평균
    bucket = mod // 30
    RND = {}
    for sc in SCOPES:
        ok = inper & seg[sc][0]
        RND[sc] = {}
        for ex in EXITS:
            xx = X[sc][ex]
            m = ok & (xx >= 0)
            ret = o[np.maximum(xx, 0)] / o - 1
            sm = np.bincount(bucket[m], weights=ret[m], minlength=48)
            ct = np.bincount(bucket[m], minlength=48)
            with np.errstate(invalid="ignore", divide="ignore"):
                RND[sc][ex] = np.where(ct > 0, sm / np.maximum(ct, 1), np.nan)
    nv = norm_vol(A, dayi)
    fin = np.isfinite(nv)
    cs = np.r_[0.0, np.cumsum(np.where(fin, nv, 0.0))]
    cn = np.r_[0, np.cumsum(fin)]
    BOX, CAND = {}, {}
    for sc in SCOPES:
        valid, newseg = seg[sc]
        for w in WS:
            S, F, Lm, T, E = box_state(l, h, valid, newseg, w)
            BOX[(sc, w)] = (S, E)
            # 봉 j(1..n-1)에서 봉 j-1 박스 상태로 판단
            j = np.arange(1, n)
            ok = valid[j - 1] & valid[j] & ~newseg[j] & (S[j - 1] >= 0)
            Sp, Fp, Tp = S[j - 1], F[j - 1], T[j - 1]
            D = (j - Sp) * mins
            # 거래량 감소: 박스 [Sp, j-1]의 앞 절반 vs 뒤 절반
            a, b = np.maximum(Sp, 0), j - 1
            L = b - a + 1
            mid = a + L // 2
            m1 = seg_range_mean(cs, cn, a, np.maximum(mid - 1, a))
            m2 = seg_range_mean(cs, cn, mid, b)
            vdec = (L >= 2) & np.isfinite(m1) & np.isfinite(m2) & (m2 < m1)
            e2 = np.minimum(j + 1, n - 1)
            nxt_ok = (j + 1 < n) & valid[e2] & ~newseg[e2]
            for ent in ENT:
                if ent == "BRK":
                    sig = ok & (c[j] > Tp) & (c[j] <= Tp * 1.01) & nxt_ok & (o[e2] <= Tp * 1.01)
                    e, fill = e2, o[e2]
                elif ent == "LIFT":
                    sig = ok & (E[j] == 1) & (c[j] <= Tp * 1.01) & nxt_ok & (o[e2] <= Tp * 1.01)
                    e, fill = e2, o[e2]
                else:
                    lim = Fp * (1 + w / 4)
                    sig = ok & (l[j] <= lim)
                    e, fill = j, np.minimum(o[j], lim)
                sig = sig & (D >= N0) & inper[np.minimum(e, n - 1)]
                ii = np.nonzero(sig)[0]
                CAND[(sc, w, ent)] = dict(e=e[ii].astype(np.int64), fill=fill[ii], D=D[ii], box=Sp[ii],
                                          vdec=vdec[ii], F=Fp[ii], T=Tp[ii])
    P[(period, sym)] = dict(n=n, mins=mins, o=o, c=c, dayi=dayi, inper=inper, reg=reg, mod=mod, bucket=bucket,
                           X=X, RND=RND, BOX=BOX, CAND=CAND, seg=seg, day=A["day"])


def trades(period, sc, ent, w, N, vol):
    """규칙 하나의 거래표(모든 137 종목)."""
    rows = []
    for (p, sym), Q in P.items():
        if p != period:
            continue
        C = Q["CAND"][(sc, w, ent)]
        m = C["D"] >= N
        if vol == "vdec":
            m &= C["vdec"]
        if not m.any():
            continue
        e, box = C["e"][m], C["box"][m]
        acc = greedy(e, box, Q["X"][sc]["RSI8"], Q["n"])
        e, fill, D = e[acc], C["fill"][m][acc], C["D"][m][acc]
        Fl, Tp = C["F"][m][acc], C["T"][m][acc]
        o, bk = Q["o"], Q["bucket"][e]
        d = dict(sym=sym, dayi=Q["dayi"][e], e=e, mod=Q["mod"][e], D=D, rise=fill / Fl - 1, fill=fill, T=Tp)
        for ex in EXITS:
            xx = Q["X"][sc][ex][e]
            d[ex] = np.where(xx >= 0, o[np.maximum(xx, 0)] / fill - 1, np.nan)
            d["r_" + ex] = Q["RND"][sc][ex][bk]
        rows.append(pd.DataFrame(d))
    if not rows:
        return pd.DataFrame(columns=["sym", "dayi", "e", "mod", "D", "rise", "fill", "T"] + list(EXITS) + ["r_" + x for x in EXITS])
    return pd.concat(rows, ignore_index=True)


def dct(day, x):
    """날짜 군집 t: 날짜별 합의 평균 / 표준오차."""
    s = pd.Series(x).groupby(np.asarray(day)).sum()
    if len(s) < 3 or s.std(ddof=1) == 0:
        return np.nan
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def stat(t, syms, ex="RSI8"):
    t = t[t.sym.isin(syms)]
    t = t[np.isfinite(t[ex])]
    if not len(t):
        return dict(n=0, avg=np.nan, win=np.nan, worst=np.nan, sum=0.0, t=np.nan, rnd=np.nan, exc=np.nan, te=np.nan)
    rr = t["r_" + ex].values
    exc = t[ex].values - rr
    okr = np.isfinite(rr)
    return dict(n=len(t), avg=t[ex].mean(), win=(t[ex] > 0).mean(), worst=t[ex].min(), sum=t[ex].sum(),
                t=dct(t.dayi.values, t[ex].values), rnd=np.nanmean(rr), exc=np.nanmean(exc),
                te=dct(t.dayi.values[okr], exc[okr]))


def base_c(period, reg_only):
    """(c) 지금 이격매수 지표 거래(같은 날짜 범위), 매도는 그 지표 자체(RSI·8시)."""
    S = load(period)
    lo, hi = PER[period]
    rows = []
    for s, A in S.items():
        for ei, xi, ent, px, why in base_sim(A, None, True, reg_only):
            dy = A["day"][ei]
            if lo <= dy <= hi:
                rows.append((s, dy.toordinal(), px / ent - 1))
    return pd.DataFrame(rows, columns=["sym", "dayi", "RSI8"])


def f_pct(v, k=2):
    return "   n/a" if v is None or not np.isfinite(v) else f"{v * 100:+.{k}f}%"


def main():
    t0 = time.time()
    for period in PER:
        S = load(period)
        for sym, A in S.items():
            prep(period, sym, A)
    print(f"준비 {time.time() - t0:.0f}s", flush=True)

    say("지지 버팀 연구 — 장중 박스(횡보 바닥) 버팀 [키 box] 2026-10-09")
    say(__doc__.split("사용:")[0].strip())
    say("")
    say("표기: n 거래수, 평균 거래당 수익, 승률, 최악, 합 = 거래 수익 합, 무작위 = 같은 종목·시각대 무작위 진입 평균(b),")
    say("      초과 = 평균 - 무작위, te = 초과의 날짜 군집 t, t = 수익 자체의 날짜 군집 t. 기본 매도 RSI·8시.")
    say("")

    # ---------------------------------------------------------------- (c) 지금 지표
    say("== 비교 기준 (c) 지금 이격매수 지표(3분봉 EMA60 -2% 지정가, RSI>70·아침 8시, 전날 +7% 다음 날 정규장 매수 안 함)")
    BC = {}
    for period in PER:
        for reg_only in (True, False):
            b = base_c(period, reg_only)
            BC[(period, "reg" if reg_only else "all")] = b
            for un, us in (("사용자78", U78), ("전체137", U137)):
                x = b[b.sym.isin(us)]
                say(f"  {PLAB[period]:<22} {'정규장 매수' if reg_only else '24시간 매수'} {un}: n {len(x):>4} 평균 {f_pct(x.RSI8.mean())}"
                    f" 승률 {(x.RSI8 > 0).mean():.0%} 최악 {f_pct(x.RSI8.min(), 1)} 합 {x.RSI8.sum():+.2f} t {dct(x.dayi.values, x.RSI8.values):+.2f}")
    say("")

    # ---------------------------------------------------------------- 1) 박스 운명(상태 수준): 오래 버틸수록 위로 끝나나?
    say("== 1) 박스의 운명 — 버틴 시간 X에 도달한 박스가 결국 위로(저가가 띠 위로 떠남) 끝나나, 아래로(띠 아래로 깨짐) 끝나나")
    say("   (정규장 범위, 박스 하나당 한 번 집계. 위% + 아래% + 장마감% = 100. fwd60 = X 도달 봉 종가 → 60분 뒤 종가(장마감이면 마지막 봉) 평균,")
    say("    무조건 = 같은 종목·기간 모든 정규장 봉의 fwd60 평균.  위/(위+아래) = 장중에 끝난 박스 중 위로 끝난 비율)")
    XS = (15, 30, 60, 120, 240)
    for w in WS:
        for period in PER:
            for un, us in (("사용자78", U78), ("전체137", U137)):
                agg = {x: [0, 0, 0, [], 0] for x in XS}
                unc = []
                for (p, sym), Q in P.items():
                    if p != period or sym not in us:
                        continue
                    S_, E_ = Q["BOX"][("reg", w)]
                    n, mins, c = Q["n"], Q["mins"], Q["c"]
                    valid, newseg = Q["seg"]["reg"]
                    K = 60 // mins
                    # 세그먼트 마지막 봉
                    segid = np.cumsum(newseg) * valid
                    last = pd.Series(np.arange(n)).groupby(segid).transform("max").values
                    fidx = np.minimum(np.arange(n) + K, last)
                    fwd = c[fidx] / c - 1
                    okb = valid & Q["inper"]
                    unc.append(fwd[okb & (np.arange(n) < last)])
                    st = np.nonzero(valid & (S_ == np.arange(n)) & Q["inper"])[0]   # 박스 시작 봉
                    if not len(st):
                        continue
                    # 박스 끝 = 다음 박스 시작 직전 또는 세그먼트 끝
                    nxt = np.r_[st[1:], n]
                    endb = np.minimum(nxt - 1, last[st])
                    L = (endb - st + 1) * mins
                    after = endb + 1
                    how = np.where((after < n) & (after == np.r_[st[1:], -1]),
                                   E_[np.minimum(after, n - 1)], 0)
                    for x in XS:
                        mm = L >= x
                        if not mm.any():
                            continue
                        hw = how[mm]
                        A_ = agg[x]
                        A_[0] += int((hw == 1).sum()); A_[1] += int((hw == -1).sum()); A_[2] += int((hw == 0).sum())
                        b = st[mm] + x // mins - 1
                        A_[3].append(fwd[b][b < last[b]])
                uncm = np.concatenate(unc).mean() if unc else np.nan
                cells = []
                for x in XS:
                    up, dn, ss, fl, _ = agg[x]
                    tot = up + dn + ss
                    fw = np.concatenate(fl).mean() if fl else np.nan
                    if tot:
                        cells.append(f"{x:>3}분 {tot:>5}개 위{up / tot:4.0%} 아래{dn / tot:4.0%} 마감{ss / tot:4.0%} 위/(위+아래){up / max(up + dn, 1):4.0%} fwd60 {f_pct(fw, 3)}")
                say(f"  w={w * 100:.1f}% {PSH[period]} {un}: 무조건 fwd60 {f_pct(uncm, 3)}")
                for cl in cells:
                    say("      " + cl)
    say("")

    # ---------------------------------------------------------------- 2) 단조성: 버틴 시간 구간별 거래 결과
    say("== 2) 단조성 표 — 진입 순간의 버틴 시간 D 구간별(규칙 (a) 거래 = D ≥ 15분이면 진입, 거래량 조건 없음)")
    say("   칸: n / 평균 / 초과(무작위 대비) — 매도 RSI·8시")
    MONO = []
    for sc in SCOPES:
        for ent in ENT:
            for w in WS:
                T = {p: trades(p, sc, ent, w, N0, "any") for p in PER}
                for bi, (a, b) in enumerate(BUCK):
                    cells = []
                    for p in PER:
                        tt = T[p][(T[p].D >= a) & (T[p].D < b)]
                        for un, us in (("78", U78), ("137", U137)):
                            s = stat(tt, us)
                            cells.append(f"{s['n']:>4} {f_pct(s['avg'])} {f_pct(s['exc'])}")
                            MONO.append(dict(sc=sc, ent=ent, w=w, bucket=BLAB[bi], period=p, un=un, **s))
                    say(f"  {SCLAB[sc]:<4} {ent:<4} w{w * 100:.1f}% {BLAB[bi]:<9} | 앞78 {cells[0]} | 앞137 {cells[1]} | 뒤78 {cells[2]} | 뒤137 {cells[3]}")
                say("")
    MONO = pd.DataFrame(MONO)
    # 구간 평균의 순위 상관(버틴 시간이 길수록 좋아지나)
    say("  단조성 요약(스피어만: 구간 순서 vs 평균 수익, 사용자78 / 전체137, 거래 10건 미만 구간 제외):")
    for sc in SCOPES:
        for ent in ENT:
            line = []
            for p in PER:
                for un in ("78", "137"):
                    rs = []
                    for w in WS:
                        g = MONO[(MONO.sc == sc) & (MONO.ent == ent) & (MONO.w == w) & (MONO.period == p) & (MONO.un == un) & (MONO.n >= 10)]
                        if len(g) >= 3:
                            rs.append(pd.Series(range(len(g))).corr(pd.Series(g.avg.values), method="spearman"))
                    line.append(f"{p[0]}{un} " + ",".join(f"{x:+.1f}" for x in rs))
            say(f"    {SCLAB[sc]:<4} {ent:<4} (w 0.5/1.0/1.5) " + " | ".join(line))
    say("")
    say("  2b) 같은 표를 고정 보유 매도로(정규장, 초과 = 무작위 대비). 칸: n / 1시간 / 4시간 / 1일 초과")
    for ent in ENT:
        for w in WS:
            T = {p: trades(p, "reg", ent, w, N0, "any") for p in PER}
            for bi, (a, b) in enumerate(BUCK):
                cells = []
                for p in PER:
                    tt = T[p][(T[p].D >= a) & (T[p].D < b)]
                    for un, us in (("78", U78), ("137", U137)):
                        ss = [stat(tt, us, ex) for ex in ("H1", "H4", "D1")]
                        cells.append(f"{ss[0]['n']:>4} {f_pct(ss[0]['exc'])} {f_pct(ss[1]['exc'])} {f_pct(ss[2]['exc'])}")
                say(f"  정규장 {ent:<4} w{w * 100:.1f}% {BLAB[bi]:<9} | 앞78 {cells[0]} | 앞137 {cells[1]} | 뒤78 {cells[2]} | 뒤137 {cells[3]}")
            say("")

    # ---------------------------------------------------------------- 3) 규칙 격자(정규장): N × w × 진입 × 거래량
    say("== 3) 규칙 격자 — 정규장 범위, 매도 RSI·8시. 72개 = N(30/60/120/240분) × w(0.5/1.0/1.5%) × 진입(BRK/LIFT/LOW) × 거래량(any/vdec)")
    say("   칸: n 평균 초과(te) | (a)버팀무시 평균. ✓ = 평균이 무작위·(a) 둘 다보다 높음")
    A0 = {}
    for p in PER:
        for ent in ENT:
            for w in WS:
                t0_ = trades(p, "reg", ent, w, N0, "any")
                A0[(p, ent, w)] = {un: stat(t0_, us) for un, us in (("78", U78), ("137", U137))}
    GRID = []
    k = 0
    for ent in ENT:
        for w in WS:
            for N in NS:
                for vol in VOLS:
                    k += 1
                    rec = dict(no=k, ent=ent, w=w, N=N, vol=vol)
                    cells = []
                    for p in PER:
                        t_ = trades(p, "reg", ent, w, N, vol)
                        for un, us in (("78", U78), ("137", U137)):
                            s = stat(t_, us)
                            a0 = A0[(p, ent, w)][un]
                            beat = s["n"] > 0 and s["avg"] > s["rnd"] and s["avg"] > a0["avg"]
                            for kk, vv in s.items():
                                rec[f"{p}_{un}_{kk}"] = vv
                            rec[f"{p}_{un}_a0"] = a0["avg"]
                            rec[f"{p}_{un}_beat"] = beat
                            if un == "78":
                                cells.append(f"{s['n']:>4} {f_pct(s['avg'])} {f_pct(s['exc'])}({s['te']:+.1f}) a{f_pct(a0['avg'])}{'✓' if beat else ' '}")
                    GRID.append(rec)
                    say(f"  {k:>2} {ent:<4} w{w * 100:.1f}% N{N:>3} {vol:<4} | 앞78 {cells[0]} | 뒤78 {cells[1]}")
    G = pd.DataFrame(GRID)
    G.to_csv(OUT.replace(".txt", "_grid.csv"), index=False, encoding="utf-8-sig")
    say("")
    both78 = int((G.early_78_beat & G.late_78_beat).sum())
    both137 = int((G.early_137_beat & G.late_137_beat).sum())
    br78 = int(((G.early_78_avg > G.early_78_rnd) & (G.late_78_avg > G.late_78_rnd)).sum())
    ba78 = int(((G.early_78_avg > G.early_78_a0) & (G.late_78_avg > G.late_78_a0)).sum())
    say(f"  시험 규칙 72개. 두 기간 모두 무작위·(a) 둘 다 이긴 규칙: 사용자78 {both78}개, 전체137 {both137}개"
        f" (사용자78 무작위만 {br78}개, (a)만 {ba78}개)")
    say(f"  앞 기간에서 둘 다 이긴 규칙 {int(G.early_78_beat.sum())}개 중 뒤에서도 이긴 규칙 {both78}개(사용자78)")
    say("")

    # ---------------------------------------------------------------- 4) 최선안(앞 기간으로만 고름)
    cand = G[G.early_78_n >= 30].sort_values("early_78_te", ascending=False)
    best = cand.iloc[0]
    say("== 4) 최선안(앞 기간·정규장·사용자78·RSI·8시, 거래 30건 이상 중 초과 te 최대) — 뒤 기간은 확인만")
    say("  앞 기간 상위 5개와 뒤 기간 결과:")
    for _, r_ in cand.head(5).iterrows():
        say(f"    #{int(r_.no):>2} {r_.ent} w{r_.w * 100:.1f}% N{int(r_.N)} {r_.vol}: 앞 n{int(r_.early_78_n)} 평균 {f_pct(r_.early_78_avg)} 초과 {f_pct(r_.early_78_exc)}(te {r_.early_78_te:+.2f})"
            f" → 뒤 n{int(r_.late_78_n)} 평균 {f_pct(r_.late_78_avg)} 초과 {f_pct(r_.late_78_exc)}(te {r_.late_78_te:+.2f})")
    ent, w, N, vol = best.ent, float(best.w), int(best.N), best.vol
    say(f"  선택: #{int(best.no)} {ent} w{w * 100:.1f}% N≥{N}분 거래량 {vol}")
    say("")
    BEST = {}
    for sc in SCOPES:
        for p in PER:
            tb = trades(p, sc, ent, w, N, vol)
            ta = trades(p, sc, ent, w, N0, "any")
            BEST[(sc, p)] = tb
            for un, us in (("사용자78", U78), ("전체137", U137)):
                say(f"  [{SCLAB[sc]} {PLAB[p]} {un}]")
                for ex in EXITS:
                    s, a = stat(tb, us, ex), stat(ta, us, ex)
                    say(f"    {EXLAB[ex]:<7} 최선 n{s['n']:>4} 평균 {f_pct(s['avg'])} 승률 {s['win']:.0%} 최악 {f_pct(s['worst'], 1)} 합 {s['sum']:+.2f} t {s['t']:+.2f}"
                        f" | 무작위 {f_pct(s['rnd'])} 초과 {f_pct(s['exc'])} te {s['te']:+.2f} | (a) n{a['n']:>4} 평균 {f_pct(a['avg'])} 합 {a['sum']:+.2f}")
                bc = BC[(p, sc)]
                x = bc[bc.sym.isin(us)]
                say(f"    (c) 이격매수 n{len(x):>4} 평균 {f_pct(x.RSI8.mean())} 합 {x.RSI8.sum():+.2f}")
    say("")
    tb = BEST[("reg", "late")]
    tb78 = tb[tb.sym.isin(U78)]
    if len(tb78):
        say(f"  뒤 기간 정규장 사용자78 최선안 거래: 진입 시 바닥 대비 위치(rise) 중앙값 {tb78.rise.median() * 100:+.2f}%, 버틴 시간 중앙값 {tb78.D.median():.0f}분,"
            f" 종목 {tb78.sym.nunique()}개, 날짜 {tb78.dayi.nunique()}일")
    tb = BEST[("reg", "early")]
    tb78 = tb[tb.sym.isin(U78)]
    if len(tb78):
        say(f"  앞 기간 정규장 사용자78 최선안 거래: rise 중앙값 {tb78.rise.median() * 100:+.2f}%, 버틴 시간 중앙값 {tb78.D.median():.0f}분,"
            f" 종목 {tb78.sym.nunique()}개, 날짜 {tb78.dayi.nunique()}일")
    say("")

    # ---------------------------------------------------------------- 5) 24시간 LOW가 좋아 보이는 이유 점검(세션별)
    say("== 5) 24시간 범위 LOW(오래 버틴 박스 하단 지정가)가 좋아 보이는 이유 — 매수 세션별(D ≥ 120분, 거래량 조건 없음, RSI·8시)")
    for w in (0.010, 0.015):
        for p in PER:
            tt = trades(p, "all", "LOW", w, 120, "any")
            ses = np.where(tt["mod"] < 570, "프리", np.where(tt["mod"] < 960, "정규", "애프터"))
            for sname in ("프리", "정규", "애프터"):
                x = tt[ses == sname]
                cells = []
                for un, us in (("78", U78), ("137", U137)):
                    s_ = stat(x, us)
                    cells.append(f"{un}: n{s_['n']:>4} 평균 {f_pct(s_['avg'])} 무작위 {f_pct(s_['rnd'])} 초과 {f_pct(s_['exc'])}(te {s_['te']:+.2f})")
                say(f"  w{w * 100:.1f}% {PSH[p]} {sname:<3} " + " | ".join(cells))
    say("")

    # ---------------------------------------------------------------- 6) 사후 관찰: 오래 버틴 박스의 돌파·끝자락 + 고정 보유
    say("== 6) 사후 관찰(검증 아님) — 2b 표에서 240분+ 구간의 BRK·LIFT가 고정 보유에서 두 기간 모두 플러스로 보여 추가로 본 24조합.")
    say("   뒤 기간을 본 뒤 고른 것이라 확인 기간 결과가 아니다. 규칙 수(72)에 넣지 않고 '본 조합'으로만 센다.")
    say("   칸: 초과(te, 가장 큰 날 하루를 빼면 초과). 정규장, 거래량 조건 없음")
    for ent in ("BRK", "LIFT"):
        for w in (0.010, 0.015):
            for N in (120, 240):
                for p in PER:
                    t = trades(p, "reg", ent, w, N, "any")
                    for un, us in (("78", U78), ("137", U137)):
                        x = t[t.sym.isin(us)]
                        out = []
                        for ex in ("RSI8", "H1", "H4", "D1"):
                            xx = x[np.isfinite(x[ex])]
                            exc = (xx[ex] - xx["r_" + ex]).values
                            ds = pd.Series(exc).groupby(xx.dayi.values).sum()
                            if len(ds):
                                big = ds.idxmax()
                                keep = xx.dayi.values != big
                                wo = exc[keep].mean() if keep.any() else np.nan
                            else:
                                wo = np.nan
                            out.append(f"{EXLAB[ex]} {f_pct(np.nanmean(exc) if len(exc) else np.nan)}(te{dct(xx.dayi.values, exc):+.1f},{f_pct(wo)})")
                        say(f"  {ent:<4} w{w * 100:.1f}% N≥{N:<3} {PSH[p]}{un:<3} n{len(x):>4} 날{x.dayi.nunique():>2} | " + " ".join(out))
    say(f"\n소요 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
