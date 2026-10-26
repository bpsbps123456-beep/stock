"""박스권 위 돌파 매수(C) 일봉 연구 — 키 daily_C (2026-10-09 사용자 요청: 박스권 매매 A 버팀 / B 이탈이 바닥 / C 위 돌파 비교).

신호(일봉, 미래 참조 없음): 박스 상단 = 직전 N일 최고가 max(High[t-N..t-1]), 하단 = 직전 N일 최저가.
  C 돌파: Close[t] > 상단[t] 이고 Close[t-1] <= 상단[t-1] (처음 넘은 날).
  수축 필터: 박스 폭(상단/하단-1)의 직전 250일 백분위 <= 30%.  거래량 필터: Volume[t] >= 1.5 x 직전 20일 평균.
체결: E1 다음 날 시가 / E2 상단에 지정가 매수(다음 3일, 닿으면 min(시가, 상단)) / E3 돌파일 고가 위 매수 스탑(다음 날, max(시가, 고가)).
매도: H1·H3·H5·H10 = h거래일 뒤 시가 / STOP 종가 < 상단이면 다음 시가(최대 10일) /
      TGT 지정가 = 진입가 + 박스 높이, 손절은 STOP과 같음(최대 10일) / TRAIL 종가 < 진입 뒤 최고 종가 x0.92면 다음 시가(최대 20일).
비교 기준: R1 같은 날 같은 묶음 전 종목 같은 보유(같은 날 무작위), R2 같은 종목 같은 기간 아무 날 같은 보유(매수 후 보유).
참고용 A·B 간이판(같은 매도 H1~H10): A 저가가 하단 +2% 안에 닿고 종가는 하단 위 / B 종가가 하단을 처음 이탈 → 다음 날 시가.
기간(진입일 기준): 고르기 2026-01-02~06-30, 확인 2026-07-01~10-08, 참고 2024-10~2025-12.
분할 의심(종가·시가 하루 -40% 이하 또는 +80% 이상)이 박스·보유 구간에 있으면 거래를 뺀다.
사용: py daily_C.py   → results/daily_C.txt"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", "Mean of empty slice")

ROOT = r"C:\Users\geunt\OneDrive\Desktop\주식"
DD = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/box/daily"
OUT = os.path.join(ROOT, r"data\derived\analysis\box_trade\results\daily_C.txt")
EARN = os.path.join(ROOT, r"data\derived\analysis\box_trade\results\earnings_cache.csv")
sys.path.insert(0, os.path.join(ROOT, r"data\derived\analysis\dip_reason\code"))
from fetch import user_universe  # noqa: E402

ETF = {"SOXL", "SOXS", "TQQQ", "SQQQ", "TECL", "LABU", "NAIL", "DPST", "KORU", "SPXL", "TNA", "QQQ", "SPY", "IWM", "SMH"}
W = 21                                    # 보유 창(0..20일)
PER = {"고르기": ("2026-01-02", "2026-06-30"), "확인": ("2026-07-01", "2026-10-08"), "참고25": ("2024-10-01", "2025-12-31")}
NS = [10, 20, 40, 60]


# ---------------------------------------------------------------- 데이터
def load():
    cal = pd.read_parquet(f"{DD}/1d_SPY.parquet").index
    ticks = sorted(f[3:-8] for f in os.listdir(DD) if f.startswith("1d_") and f.endswith(".parquet"))
    ticks = [t for t in ticks if t not in ETF]
    arr = {k: np.full((len(cal), len(ticks)), np.nan) for k in "OHLCV"}
    for j, t in enumerate(ticks):
        d = pd.read_parquet(f"{DD}/1d_{t}.parquet").reindex(cal)
        for k, col in zip("OHLCV", ["Open", "High", "Low", "Close", "Volume"]):
            arr[k][:, j] = d[col].to_numpy(float)
    return cal, ticks, arr


def bad_days(O, C):
    pc = np.vstack([np.full((1, C.shape[1]), np.nan), C[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        r, g = C / pc - 1, O / pc - 1
    bad = (r <= -0.4) | (r >= 0.8) | (g <= -0.4) | (g >= 0.8)
    cb = np.vstack([np.zeros((1, C.shape[1])), np.cumsum(bad, 0)])       # cb[b+1]-cb[a] = [a,b] 안의 수
    return bad, cb


def earn_days(cal, ticks):
    """실적 반응일(장 마감 뒤 발표면 다음 거래일, 그 밖은 당일) 표시 [T,S]. 캐시에 없는 종목은 False(미상)."""
    m = np.zeros((len(cal), len(ticks)), bool)
    known = np.zeros(len(ticks), bool)
    if not os.path.exists(EARN):
        return m, known
    e = pd.read_csv(EARN)
    pos = {t: j for j, t in enumerate(ticks)}
    cd = pd.DatetimeIndex(cal)
    for t, g in e.groupby("ticker"):
        if t not in pos:
            continue
        known[pos[t]] = True
        for s in g["earnings_dt"]:
            ts = pd.Timestamp(str(s)[:19])
            day = ts.normalize()
            k = cd.searchsorted(day)
            if ts.hour >= 16:                        # 장 마감 뒤 → 다음 거래일 반응
                k = cd.searchsorted(day + pd.Timedelta(days=1))
            if k < len(cd):
                m[k, pos[t]] = True
    return m, known


# ---------------------------------------------------------------- 시뮬레이션
def pad(a, n=W + 3):
    return np.vstack([a, np.full((n, a.shape[1]), np.nan)])


def gather(P, e, s):
    return P[e[:, None] + np.arange(W + 1)[None, :], s[:, None]]


def simulate(A, e, s, px, ex, lv=None, lo=None):
    """A: 패딩된 O H L C. e 진입일 인덱스, px 진입가. ex = ('H',h) | ('STOP',) | ('TGT',) | ('TRAIL',).
    반환: k(진입일부터 매도일까지 거래일), 매도가, kind(0=시가 매도, 1=장중 지정가 매도)."""
    Ow, Cw, Hw = gather(A["O"], e, s), gather(A["C"], e, s), gather(A["H"], e, s)
    n = len(e)
    ii = np.arange(n)
    kind = np.zeros(n, int)
    if ex[0] == "H":
        k = np.full(n, ex[1])
        return k, Ow[ii, k], kind
    if ex[0] == "STOP":
        m = 10
        cond = Cw[:, :m] < lv[:, None]
        has = cond.any(1)
        k = np.where(has, cond.argmax(1) + 1, m)
        return k, Ow[ii, k], kind
    if ex[0] == "TGT":
        m = 10
        tg = px + (lv - lo)
        hit = Hw[:, :m] >= tg[:, None]
        stp = Cw[:, :m] < lv[:, None]
        fh = np.where(hit.any(1), hit.argmax(1), 99)
        fs = np.where(stp.any(1), stp.argmax(1), 99)
        k = np.full(n, m)
        price = Ow[ii, m].copy()
        t_ = (fh <= fs) & (fh < 99)
        s_ = (fs < fh) & (fs < 99)
        k[t_] = fh[t_]
        price[t_] = np.maximum(Ow[ii[t_], fh[t_]], tg[t_])
        kind[t_] = 1
        k[s_] = fs[s_] + 1
        price[s_] = Ow[ii[s_], fs[s_] + 1]
        return k, price, kind
    if ex[0] == "TRAIL":
        m = 20
        mc = np.maximum(np.fmax.accumulate(Cw[:, :m], axis=1), px[:, None])
        cond = Cw[:, :m] < 0.92 * mc
        has = cond.any(1)
        k = np.where(has, cond.argmax(1) + 1, m)
        return k, Ow[ii, k], kind
    raise ValueError(ex)


# ---------------------------------------------------------------- 비교 기준
def base_r1(O, C, cb, umask):
    """R1[e,k] = 같은 날 e 시가 진입, k일 뒤 (시가|종가) 매도 수익의 묶음 평균. [T, W+1, 2]."""
    T = O.shape[0]
    Op, Cp = pad(O), pad(C)
    cbp = np.vstack([cb, np.repeat(cb[-1:], W + 3, 0)])
    out = np.full((T, W + 1, 2), np.nan)
    o0 = O[:, umask]
    for k in range(W + 1):
        nb = (cbp[k + 1:k + 1 + T] - cbp[:T])[:, umask] > 0
        for j, P in enumerate((Op, Cp)):
            r = P[k:k + T][:, umask] / o0 - 1
            r[nb] = np.nan
            with np.errstate(invalid="ignore"):
                out[:, k, j] = np.nanmean(np.where(np.isfinite(r), r, np.nan), 1) if np.isfinite(r).any() else np.nan
    return out


def base_r2(O, C, cb, emask):
    """R2[s,k] = 같은 종목, 기간 안 아무 날 시가 진입 k일 뒤 매도 평균. [S, W+1, 2]."""
    T, S = O.shape
    Op, Cp = pad(O), pad(C)
    cbp = np.vstack([cb, np.repeat(cb[-1:], W + 3, 0)])
    out = np.full((S, W + 1, 2), np.nan)
    o0 = O[emask]
    for k in range(W + 1):
        nb = (cbp[k + 1:k + 1 + T] - cbp[:T])[emask] > 0
        for j, P in enumerate((Op, Cp)):
            r = P[k:k + T][emask] / o0 - 1
            r[nb] = np.nan
            with np.errstate(invalid="ignore"):
                out[:, k, j] = np.nanmean(r, 0)
    return out


# ---------------------------------------------------------------- 통계
def stats(r, x1, x2, e):
    n = len(r)
    if n == 0:
        return dict(n=0)
    def ct(x):                                   # 날짜 군집 t
        if n < 3:
            return np.nan
        d = pd.Series(x - x.mean()).groupby(e).sum().to_numpy()
        se = np.sqrt((d ** 2).sum()) / n
        return x.mean() / se if se > 0 else np.nan
    return dict(n=n, avg=r.mean() * 100, win=(r > 0).mean() * 100, worst=r.min() * 100, sum=r.sum() * 100,
                x1=x1.mean() * 100, x2=x2.mean() * 100, t=ct(r), t1=ct(x1), t2=ct(x2), nd=len(np.unique(e)))


def main():
    cal, ticks, A0 = load()
    O, H, L, C, V = (A0[k] for k in "OHLCV")
    T, S = O.shape
    bad, cb = bad_days(O, C)
    cbp = np.vstack([cb, np.repeat(cb[-1:], W + 3, 0)])
    A = {k: pad(A0[k]) for k in "OHLC"}
    sp = set(pd.read_csv(f"{DD}/sp500_list.csv").iloc[:, 0].astype(str))
    u78 = set(user_universe()["합집합"])
    U = {"S&P500": np.array([t in sp for t in ticks]), "사용자78": np.array([t in u78 for t in ticks])}
    dates = pd.DatetimeIndex(cal)
    pmask = {p: (dates >= a) & (dates <= b) for p, (a, b) in PER.items()}
    R1 = {u: base_r1(O, C, cb, m) for u, m in U.items()}
    R2 = {p: base_r2(O, C, cb, pm) for p, pm in pmask.items()}
    em, eknown = earn_days(cal, ticks)
    em_near = em | np.vstack([np.zeros((1, S), bool), em[:-1]])       # 반응일 또는 그다음 날

    dfH, dfL, dfC, dfV = (pd.DataFrame(x) for x in (H, L, C, V))
    vavg = dfV.rolling(20, min_periods=20).mean().shift(1).to_numpy()
    volx = V >= 1.5 * vavg
    box = {}
    for N in NS:
        hi = dfH.rolling(N, min_periods=N).max().shift(1).to_numpy()
        lo = dfL.rolling(N, min_periods=N).min().shift(1).to_numpy()
        wd = pd.DataFrame(hi / lo - 1)
        pr = wd.rolling(250, min_periods=60).rank(pct=True).to_numpy()
        prevC = np.vstack([np.full((1, S), np.nan), C[:-1]])
        prevhi = np.vstack([np.full((1, S), np.nan), hi[:-1]])
        prevlo = np.vstack([np.full((1, S), np.nan), lo[:-1]])
        with np.errstate(invalid="ignore"):
            sigC = (C > hi) & (prevC <= prevhi)
            touchA = (L <= lo * 1.02) & (C > lo)
            prevA = np.vstack([np.zeros((1, S), bool), touchA[:-1]])
            sigA = touchA & ~prevA
            sigB = (C < lo) & (prevC >= prevlo)
            contr = pr <= 0.30
        box[N] = dict(hi=hi, lo=lo, C=sigC, A=sigA, B=sigB, contr=contr)

    # ---------------- 규칙 목록 (C 100개)
    EXITS = [("H", 1), ("H", 3), ("H", 5), ("H", 10), ("STOP",), ("TGT",), ("TRAIL",)]
    rules = []
    for N in NS:
        for f in ["없음", "수축", "거래량"]:
            for ex in EXITS:
                rules.append(("C", N, f, "E1", ex))
        for ex in [("H", 5), ("H", 10)]:
            rules.append(("C", N, "수축+거래량", "E1", ex))
    for E in ["E2", "E3"]:
        for N in [20, 60]:
            for ex in [("H", 5), ("H", 10)]:
                rules.append(("C", N, "없음", E, ex))
    nC = len(rules)
    ref_rules = [(sig, N, "없음", "E1", ("H", h)) for sig in "AB" for N in NS for h in (1, 3, 5, 10)]

    def exname(ex):
        return f"H{ex[1]}" if ex[0] == "H" else ex[0]

    def rname(r):
        return f"{r[0]}_N{r[1]}_{r[2]}_{r[3]}_{exname(r[4])}"

    def trades(rule):
        sig, N, f, E, ex = rule
        b = box[N]
        m = b[sig].copy()
        if "수축" in f:
            m &= b["contr"]
        if "거래량" in f:
            m &= volx
        m[-1] = False
        t, s = np.nonzero(m)
        hi, lo = b["hi"][t, s], b["lo"][t, s]
        if E == "E1":
            e, px = t + 1, O[t + 1, s]
        elif E == "E2":                                   # 상단 지정가 되돌림 매수, 다음 3일
            e = np.full(len(t), -1)
            px = np.full(len(t), np.nan)
            for d in (1, 2, 3):
                ok = (e < 0) & (t + d < T)
                idx = np.where(ok)[0]
                hitd = L[np.minimum(t[idx] + d, T - 1), s[idx]] <= hi[idx]
                j = idx[hitd]
                e[j] = t[j] + d
                px[j] = np.fmin(O[t[j] + d, s[j]], hi[j])
            keep = e >= 0
            t, s, hi, lo, e, px = t[keep], s[keep], hi[keep], lo[keep], e[keep], px[keep]
        else:                                             # E3 돌파일 고가 위 매수 스탑, 다음 날만
            hh = H[t, s]
            ok = H[np.minimum(t + 1, T - 1), s] >= hh
            ok &= t + 1 < T
            t, s, hi, lo, hh = t[ok], s[ok], hi[ok], lo[ok], hh[ok]
            e, px = t + 1, np.fmax(O[t + 1, s], hh)
        k, xp, kind = simulate(A, e, s, px, ex, hi, lo)
        ret = xp / px - 1
        lo_idx = np.maximum(t - N, 0)
        nbad = cbp[np.minimum(e + k + 1, T), s] - cbp[lo_idx, s]
        ok = np.isfinite(ret) & (nbad == 0) & (e + k < T) & np.isfinite(px)
        t, s, e, k, kind, ret, px = t[ok], s[ok], e[ok], k[ok], kind[ok], ret[ok], px[ok]
        return dict(t=t, s=s, e=e, k=k, kind=kind, ret=ret, px=px)

    def evaluate(tr, uni, per):
        sel = U[uni][tr["s"]] & pmask[per][tr["e"]]
        e, s, k, kind, r = tr["e"][sel], tr["s"][sel], tr["k"][sel], tr["kind"][sel], tr["ret"][sel]
        b1 = R1[uni][e, k, kind]
        b2 = R2[per][s, k, kind]
        ok = np.isfinite(b1) & np.isfinite(b2)
        return stats(r[ok], (r - b1)[ok], (r - b2)[ok], e[ok]), sel

    res = {}
    TR = {}
    for rule in rules + ref_rules:
        tr = trades(rule)
        TR[rname(rule)] = tr
        for uni in U:
            for per in PER:
                res[(rname(rule), uni, per)] = evaluate(tr, uni, per)[0]

    def beats(st):
        return st.get("n", 0) >= 30 and st["x1"] > 0 and st["x2"] > 0

    # ---------------- 고르기: S&P500 고르기 기간 min(t1, t2) 최대, n >= 100
    cands = []
    for rule in rules:
        st = res[(rname(rule), "S&P500", "고르기")]
        if st.get("n", 0) >= 100:
            cands.append((min(st["t1"], st["t2"]), rname(rule)))
    cands.sort(reverse=True)
    best = cands[0][1]
    cands78 = []
    for rule in rules:
        st = res[(rname(rule), "사용자78", "고르기")]
        if st.get("n", 0) >= 30:
            cands78.append((min(st["t1"], st["t2"]), rname(rule)))
    cands78.sort(reverse=True)
    best78 = cands78[0][1]

    nb_sp = sum(beats(res[(rname(r), "S&P500", "고르기")]) and beats(res[(rname(r), "S&P500", "확인")]) for r in rules)
    nb_78 = sum(beats(res[(rname(r), "사용자78", "고르기")]) and beats(res[(rname(r), "사용자78", "확인")]) for r in rules)
    nb_all = sum(all(beats(res[(rname(r), u, p)]) for u in U for p in ("고르기", "확인")) for r in rules)

    L_ = []
    P = L_.append

    def row(name, uni, per):
        st = res[(name, uni, per)]
        if st.get("n", 0) == 0:
            return f"{name:<28} {uni:<7} {per:<5}  n=0"
        return (f"{name:<28} {uni:<7} {per:<5} n={st['n']:>5} 날={st['nd']:>3} 평균={st['avg']:+6.2f}% 승률={st['win']:5.1f}% "
                f"최악={st['worst']:+6.1f}% 합={st['sum']:+8.0f}% t={st['t']:+5.2f} | R1초과={st['x1']:+5.2f}%(t{st['t1']:+5.2f}) "
                f"R2초과={st['x2']:+5.2f}%(t{st['t2']:+5.2f})")

    P("박스 위 돌파 매수(C) 일봉 연구 — daily_C  (작성 2026-10-09, 데이터 yfinance 일봉 2024-10-09~2026-10-08, 조정 안 함)")
    P(__doc__)
    P(f"종목: S&P500 {U['S&P500'].sum()}개, 사용자78 중 일봉 있는 {U['사용자78'].sum()}개 (레버리지·지수 ETF 제외)")
    P(f"시험한 C 규칙 {nC}개 (+ 참고용 A·B 간이판 {len(ref_rules)}개, 고르기 대상 아님)")
    P("R1 = 같은 날 같은 묶음 전 종목을 같은 날 시가에 사서 같은 날짜에 판 평균(같은 날 무작위 진입). "
      "R2 = 같은 종목을 같은 기간 아무 날 시가에 사서 같은 일수 보유한 평균(매수 후 보유). "
      "장중 지정가 매도(TGT)는 그날 종가로 맞춤. t = 진입일 군집 표준오차 t.")
    P("'기준보다 나음' = 건수 30 이상, R1초과>0, R2초과>0.")
    P(f"두 기간(고르기·확인) 모두 기준보다 나은 C 규칙: S&P500 {nb_sp}/{nC}, 사용자78 {nb_78}/{nC}, 두 묶음 모두 {nb_all}/{nC}")
    def x1pos(st):
        return st.get("n", 0) >= 30 and st["x1"] > 0
    for u in U:
        a = sum(x1pos(res[(rname(r), u, "고르기")]) and x1pos(res[(rname(r), u, "확인")]) for r in rules)
        b = sum(x1pos(res[(rname(r), u, "확인")]) for r in rules)
        c = sum(x1pos(res[(rname(r), u, "고르기")]) for r in rules)
        P(f"  (느슨한 기준 R1초과>0만) {u}: 고르기 {c}/{nC}, 확인 {b}/{nC}, 두 기간 모두 {a}/{nC}")
    P("")
    P(f"== 고른 최선안(S&P500 고르기 기간 min(t_R1, t_R2) 최대, 건수 100 이상): {best}")
    for uni in U:
        for per in PER:
            P(row(best, uni, per))
    P("")
    P(f"== 사용자78 고르기 기간 기준 최선안(참고): {best78}")
    for uni in U:
        for per in ("고르기", "확인"):
            P(row(best78, uni, per))
    P("")
    P("== S&P500 고르기 기간 상위 15개와 확인 기간 성적 (선택편향 확인)")
    for sc, nm in cands[:15]:
        a, b = res[(nm, "S&P500", "고르기")], res[(nm, "S&P500", "확인")]
        c = res[(nm, "사용자78", "확인")]
        P(f"{nm:<28} 점수{sc:+5.2f} | 고르기 n={a['n']:>5} 평균{a['avg']:+5.2f} R1초과{a['x1']:+5.2f} R2초과{a['x2']:+5.2f} | "
          f"확인 n={b.get('n',0):>5} 평균{b.get('avg',np.nan):+5.2f} R1초과{b.get('x1',np.nan):+5.2f} R2초과{b.get('x2',np.nan):+5.2f} | "
          f"확인·78 n={c.get('n',0):>4} 평균{c.get('avg',np.nan):+5.2f} R1초과{c.get('x1',np.nan):+5.2f}")
    P("")
    P("== 전 규칙 표 (S&P500·사용자78 × 고르기·확인·참고25)")
    for rule in rules:
        for uni in U:
            for per in PER:
                P(row(rname(rule), uni, per))
    P("")
    P("== 참고 A·B 간이판 전 표 (고르기 대상 아님)")
    for rule in ref_rules:
        for uni in U:
            for per in PER:
                P(row(rname(rule), uni, per))
    P("")
    P("== 같은 매도 규칙(H1·H3·H5·H10, 다음 날 시가 진입, 필터 없음)으로 본 A·B·C 비교 — 평균 / R1초과 / 건수")
    for uni in U:
        for per in PER:
            P(f"-- {uni} {per}")
            for N in NS:
                parts = []
                for sig in "ABC":
                    for h in (1, 3, 5, 10):
                        st = res[(f"{sig}_N{N}_없음_E1_H{h}", uni, per)]
                        parts.append(f"{sig}H{h} {st.get('avg',np.nan):+5.2f}/{st.get('x1',np.nan):+5.2f}/{st.get('n',0)}")
                P(f"N{N:<3} " + " | ".join(parts))
    P("")

    # ---------------- 월별 추이 (최근 장세)
    def monthly(name, uni):
        tr = TR[name]
        sel = U[uni][tr["s"]]
        e, s, k, kind, r = tr["e"][sel], tr["s"][sel], tr["k"][sel], tr["kind"][sel], tr["ret"][sel]
        b1 = R1[uni][e, k, kind]
        mo = dates[e].to_period("M")
        df = pd.DataFrame(dict(m=mo, r=r * 100, x=(r - b1) * 100)).dropna()
        return df.groupby("m").agg(n=("r", "size"), avg=("r", "mean"), x1=("x", "mean"))

    def cell(tb, m):
        if m not in tb.index:
            return f"{'-':>32}"
        v = tb.loc[m]
        return f"{v['n']:>5.0f} {v['avg']:+6.2f} {v['x1']:+6.2f}".rjust(32)

    P("== 월별 추이 (진입월, S&P500): 건수 / 평균 / R1초과")
    names = [best, "C_N20_없음_E1_H5", "A_N20_없음_E1_H5", "B_N20_없음_E1_H5"]
    tabs = [monthly(nm, "S&P500") for nm in names]
    P(f"{'월':<8}" + "".join(f"{nm:>32}" for nm in names))
    for m in tabs[0].index.union(tabs[2].index):
        P(f"{str(m):<8}" + "".join(cell(tb, m) for tb in tabs))
    P("")
    P("== 월별 추이 (사용자78): 건수 / 평균 / R1초과")
    tabs = [monthly(nm, "사용자78") for nm in names]
    for m in tabs[0].index.union(tabs[2].index):
        P(f"{str(m):<8}" + "".join(cell(tb, m) for tb in tabs))
    P("")

    # ---------------- 실적 반응일 돌파 vs 아닌 돌파 (최선안)
    P("== 실적 반응일(발표가 장 마감 뒤면 다음 거래일) 또는 그다음 날의 신호 vs 나머지 — 최선안과 C_N20_없음_E1_H5, 실적 일정 캐시 있는 종목만")
    for nm in [best, "C_N20_없음_E1_H5"]:
        tr = TR[nm]
        for uni in U:
            for per in ("고르기", "확인"):
                sel = U[uni][tr["s"]] & pmask[per][tr["e"]] & eknown[tr["s"]]
                isE = em_near[tr["t"], tr["s"]]
                for lab, mm in (("실적", sel & isE), ("비실적", sel & ~isE)):
                    e, s, k, kind, r = tr["e"][mm], tr["s"][mm], tr["k"][mm], tr["kind"][mm], tr["ret"][mm]
                    b1, b2 = R1[uni][e, k, kind], R2[per][s, k, kind]
                    ok = np.isfinite(b1) & np.isfinite(b2)
                    st = stats(r[ok], (r - b1)[ok], (r - b2)[ok], e[ok])
                    if st.get("n", 0):
                        P(f"{nm:<26} {uni:<7} {per:<4} {lab:<4} n={st['n']:>5} 평균={st['avg']:+6.2f}% 승률={st['win']:5.1f}% "
                          f"R1초과={st['x1']:+5.2f}%(t{st['t1']:+5.2f}) R2초과={st['x2']:+5.2f}%")
    P("")

    # ---------------- 나이키 확인
    P("== NKE 확인 (2026-07-01 이후 신호, 신호일 → 진입일 시가, 수익)")
    j = ticks.index("NKE")
    for N in NS:
        for sig in "ABC":
            ts = np.where(box[N][sig][:, j] & (dates >= "2026-07-01"))[0]
            for t in ts:
                e = t + 1
                txt = f"N{N} {sig} 신호 {dates[t].date()} 종가 {C[t, j]:.2f} (상단 {box[N]['hi'][t, j]:.2f} 하단 {box[N]['lo'][t, j]:.2f})"
                if e < T:
                    txt += f" → {dates[e].date()} 시가 {O[e, j]:.2f}, 10/8 종가 {C[-1, j]:.2f} ({(C[-1, j] / O[e, j] - 1) * 100:+.1f}%)"
                P(txt)
    P("")
    P("한계: 일봉이라 장중 순서(지정가·스탑 체결 순서)를 모른다. E3 매수 스탑은 한국시간 24시 이후 체결분을 사용자가 실행할 수 없다. "
      "보유가 겹치는 거래(같은 종목 재돌파·H10)가 있어 군집 t가 다소 부풀 수 있다. 확인 기간 끝(10/8) 뒤 매도가 필요한 거래는 뺐다. "
      "배당 미조정. S&P500 목록은 현재 구성(생존 편향).")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L_))
    print("\n".join(L_[:40]))
    print("...")
    print(f"best={best} best78={best78} nb_sp={nb_sp} nb_78={nb_78} nb_all={nb_all} nC={nC}")


if __name__ == "__main__":
    main()
