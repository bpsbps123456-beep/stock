"""시가 되찾음 매수 연구(2026-10-10 사용자: 빠른 반등은 드물다 → 빠르게 반등한 종목은 그 뒤 더 오르나. CRCL 10/9·AMD 10/6형).
이격매수 신호와 무관하게 유니버스 전체 정규장 사건을 본다. 정의·격자·결과는 아래로 시험 전에 고정했다.

사건(정규장 09:30~16:00 ET, 3분봉 = 1분봉을 시계 기준으로 묶음):
  O = 그날 09:30 봉 시가. D = 개장 뒤 M분 안(봉 시작 < 09:30+M) 저가 <= O x (1-d)인 첫 봉.
  k = D 이후(D 포함) 3분봉 종가 >= O인 첫 봉(되찾음). L = 개장~k 저가 최저 봉(같으면 늦은 봉, k 시점에 아는 값).
  속도 = k 봉 시작 - L 봉 시작(분). 빠름 = 속도 <= 3N분, 느림 = 그보다 늦게 되찾음. 끝내 못 되찾으면 사지 않는다.
  매수 = k 다음 봉(e) 시가. e가 그날 정규장 봉이 아니면(15:57 봉에서 되찾음) 사지 않는다. 종목·날짜·정의당 1건.
  격자: d {1.5%, 2.5%} x M {30, 60분} x N {10, 20봉 = 30, 60분} = 8개 정의.
보유 규칙(고정):
  a  RSI14 > 70인 봉(매수 봉 포함) 다음 봉 시가, 못 팔면 매수 뒤 첫 한국 08시(= ET 19:00) 이후 첫 봉 시가(지표와 같음)
  b  그날 정규장 종가(15:57 봉 종가)
  c  매수가 +3% 지정가 매도(1분봉 고가로 체결 판정), 못 닿으면 a와 같은 08시 봉 시가
  d  다음 거래일 정규장 종가(10/9 매수는 다음 날 자료가 없어 빠진다)
  프리·애프터 체결 정리(한 분짜리 꼬리 배제): 봉 시가 매도는 그 분 시가가 직전 1분 종가에서 1% 넘게 벗어나면 그 분 종가로 판다.
  지정가 매도는 프리·애프터 분의 고가를 직전 1분 종가 x 1.01로 자른 값으로 판정한다. 정규장은 그대로.
결과(시험 전 고정):
  ret     거래당 수익
  ex_day  주 결과. ret - B_day. B_day = 같은 종목·같은 날 진입 봉부터 30분 안(3분봉 10개) 아무 봉 시가에 사서 같은 규칙으로 판 평균
          (시간대 맞춤 무작위의 기댓값). 진입 전 봉은 쓰지 않는다: 이미 '빠졌다가 되찾았다'는 경로를 알고 고른 봉이라
          그 앞 봉은 더 싸게 산 것이 확정이다(사후 정보). 진입 뒤 봉은 무작위 보행이면 기댓값 차이가 0이다.
  ex_oth  ret - B_oth. B_oth = 같은 종목·같은 시각(분) 봉에 같은 절반의 다른 날 들어갔을 때 같은 규칙 평균(되찾음 날 대 보통 날).
  ex_q    ret - 같은 진입·청산 시각 QQQ 수익(시장 전체 반등일 통제).
  t는 날짜 군집. 승률 = ret > 0 비율.
고르기·확인: 앞 절반 9/10~9/23(10일)에서 빠름 32개(정의 8 x 보유 4) 중 ex_day 날짜 군집 t 상위 최대 3개(평균 > 0, 건수 >= 15)를 고르고
  뒤 절반 9/24~10/9(12일)에서 확인한다. 느림 32개와 빠름-느림 차 32개는 비교용(고르기 대상 아님).
  선택편향: 날짜 부호 뒤집기(1,000번)로 32개 시험 최대 t의 우연 분포를 잰다.
판정(시험 전 고정): 확인됨 = 뒤 ex_day > 0, t >= 2, 상위 1종목·상위 1일 각각 빼도 평균 > 0, ex_q 평균 > 0.
  약함 = 뒤 ex_day > 0이지만 확인됨 아님. 뒤에서 실패 = 뒤 ex_day <= 0.
보조: 7/14~9/8 5분봉(stop_lab.load('early'), EMA36·RSI9). 같은 분 단위(M·3N·30분 창), 지정가·꼬리 정리는 5분봉 단위. 1분봉 없음.
깨어 있음: 매수는 한국 08~24시(= ET 19:00~11:00)에만 낼 수 있다. 정규장 매수는 11:00 ET 전 진입만 실행 가능('깸' 열).
  자는 동안 실행 가능 = 위쪽 지정가와 08시 정리만 쓰는 보유 c. a(RSI 시장가)·b(16시 종가)·d(다음 날 16시)는 깨어 있어야 한다.
종목: build.SYMS 137개에서 GOOG(GOOGL 중복)·SOXX(ETF)를 뺀 135개.
저장: ../results/fr_reclaim.txt(요약), fr_reclaim_events.csv(사건·결과), fr_reclaim_tests.csv(모든 시험)
사용: py fr_reclaim.py"""
import os, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
sys.path.insert(0, HERE)
from events import bars, prep
import stop_lab
from build import SYMS as _SYMS

SP = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip"
RAW, RAW_E = os.path.join(SP, "raw_1010"), os.path.join(SP, "raw")
OUT = os.path.join(HERE, "../results")
SYMS = [s for s in _SYMS if s not in ("GOOG", "SOXX")]
DS, MS, NS, RULES = (0.015, 0.025), (30, 60), (10, 20), ("a", "b", "c", "d")
RNAME = {"a": "a RSI/08시", "b": "b 당일 종가", "c": "c +3%/08시", "d": "d 다음날 종가"}
SLEEP_OK = {"a": False, "b": False, "c": True, "d": False}
LO, MID, HI = date(2026, 9, 10), date(2026, 9, 24), date(2026, 10, 9)
E_LO, E_HI = date(2026, 7, 14), date(2026, 9, 8)
WIN, AWAKE, TGT, TAIL = 30, 660, 0.03, 0.01
NS_MIN = 60 * 10**9
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LINES.append(s)


def pct(x, nd=2):
    return "   nan" if not np.isfinite(x) else f"{x * 100:+.{nd}f}%"


# ---------------- 자료 ----------------

def dead_of(idx):
    kst = idx.tz_convert("Asia/Seoul")
    t8 = kst.normalize() + pd.Timedelta(hours=8)
    return np.where(kst < t8, t8.as_unit("ns").asi8, (t8 + pd.Timedelta(days=1)).as_unit("ns").asi8).astype(np.int64)


def fine_of(tk, o, h, c, reg):
    """체결 판정용 세밀 배열(1분봉 또는 5분봉). 프리·애프터 고가는 직전 종가 x 1.01로 자른다."""
    pc = np.r_[np.nan, c[:-1]]
    he = np.where(reg | ~np.isfinite(pc), h, np.minimum(h, pc * (1 + TAIL)))
    return dict(tk=tk, o=o, h=h, c=c, reg=reg, pc=pc, he=he)


def load_main(s):
    d = prep(bars(RAW, s, 3), 60, 14)
    m = pd.read_parquet(os.path.join(RAW, f"1m_{s}.parquet"))
    m = m[m.High > 0]
    mod1 = m.index.hour * 60 + m.index.minute
    F = fine_of(m.index.as_unit("ns").asi8, m.Open.values.astype(float), m.High.values.astype(float),
                m.Close.values.astype(float), np.asarray((mod1 >= 570) & (mod1 < 960)))
    idx = d.index
    return dict(sym=s, mins=3, idx=idx, tk=idx.as_unit("ns").asi8, o=d.o.values.astype(float), h=d.h.values.astype(float),
                l=d.l.values.astype(float), c=d.c.values.astype(float), r=d.r.values.astype(float), dead=dead_of(idx),
                reg=(d.ses.values == "정규"), mod=d["mod"].values.astype(int), day=np.asarray(d.date), F=F)


def load_early(A):
    A = dict(A)
    A["F"] = fine_of(A["tk"], A["o"], A["h"], A["c"], A["reg"].astype(bool))
    A["reg"] = A["reg"].astype(bool)
    return A


def qqq(per):
    if per == "main":
        m = pd.read_parquet(os.path.join(RAW, "1m_QQQ.parquet"))
    else:
        m = bars(RAW_E, "QQQ", 5)
    m = m[m.High > 0]
    mod = m.index.hour * 60 + m.index.minute
    rg = m[(mod >= 570) & (mod < 960)]
    dc = rg.Close.groupby(rg.index.date).last()
    return dict(tk=m.index.as_unit("ns").asi8, o=m.Open.values.astype(float), c=m.Close.values.astype(float), dc=dc.to_dict(),
                days=sorted(dc.index))


# ---------------- 봉마다 네 보유 규칙 ----------------

def clean_open(A):
    """봉 시가 매도가. 정규장은 시가, 프리·애프터는 꼬리면 그 분(5분봉은 그 봉) 종가."""
    F, tk, o = A["F"], A["tk"], A["o"]
    pos = np.searchsorted(F["tk"], tk)
    pos = np.minimum(pos, len(F["tk"]) - 1)
    o1, pc, c1 = F["o"][pos], F["pc"][pos], F["c"][pos]
    tail = np.isfinite(pc) & (np.abs(o1 / pc - 1) > TAIL)
    return np.where(A["reg"], o, np.where(tail, c1, o1))


def bar_returns(A):
    """정규장 봉 j 시가에 샀을 때 a·b·c·d 수익과 청산 시각. 반환 DataFrame(정규장 봉만)."""
    tk, o, r, reg, day, mod = A["tk"], A["o"], A["r"], A["reg"], A["day"], A["mod"]
    F = A["F"]
    n = len(tk)
    xo = clean_open(A)
    # RSI > 70 다음 첫 위치(뒤에서 채움)
    nxt = np.full(n + 1, n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        nxt[i] = i if r[i] > 70 else nxt[i + 1]
    regi = np.where(reg)[0]
    rday = day[regi]
    # 정규장 종가·다음 거래일 종가
    dc = {}
    for dy in pd.unique(rday):
        lastj = regi[rday == dy][-1]
        dc[dy] = A["c"][lastj]
    days = sorted(dc)
    nd = {days[i]: dc[days[i + 1]] for i in range(len(days) - 1)}
    ndd = {days[i]: days[i + 1] for i in range(len(days) - 1)}
    rows = []
    for j in regi:
        P = o[j]
        di = int(np.searchsorted(tk, A["dead"][j]))
        if di >= n:
            ra = rc = np.nan
            xa_t = xc_t = -1
            xc_fill = False
            why = ""
        else:
            ri = nxt[j] + 1
            if ri <= di:
                ai, why = ri, "RSI"
            else:
                ai, why = di, "8시"
            ra, xa_t = xo[ai] / P - 1, tk[ai]
            s0 = int(np.searchsorted(F["tk"], tk[j]))
            s1 = int(np.searchsorted(F["tk"], tk[di]))
            T = P * (1 + TGT)
            hit = np.flatnonzero(F["he"][s0:s1] >= T)
            if len(hit):
                m = s0 + hit[0]
                px = F["o"][m] if (F["reg"][m] and F["o"][m] >= T) else T
                rc, xc_t, xc_fill = px / P - 1, F["tk"][m], True
            else:
                rc, xc_t, xc_fill = xo[di] / P - 1, tk[di], False
        dy = day[j]
        rows.append((j, dy, mod[j], tk[j], P, ra, why, xa_t, dc[dy] / P - 1, rc, xc_t, xc_fill,
                     nd.get(dy, np.nan) / P - 1, ndd.get(dy)))
    return pd.DataFrame(rows, columns=["j", "day", "mod", "tk", "P", "a", "why_a", "xa_t", "b", "c", "xc_t", "xc_fill", "d", "nday"])


# ---------------- 사건 ----------------

def q_ret(Q, R, rule, row_j):
    """같은 진입·청산 시각 QQQ 수익."""
    t0 = R.at[row_j, "tk"]
    i0 = min(np.searchsorted(Q["tk"], t0), len(Q["tk"]) - 1)
    q0 = Q["o"][i0]
    dy = R.at[row_j, "day"]
    if rule == "a":
        t1 = R.at[row_j, "xa_t"]
        if t1 < 0:
            return np.nan
        q1 = Q["o"][min(np.searchsorted(Q["tk"], t1), len(Q["tk"]) - 1)]
    elif rule == "b":
        q1 = Q["dc"].get(dy, np.nan)
    elif rule == "c":
        t1 = R.at[row_j, "xc_t"]
        if t1 < 0:
            return np.nan
        if R.at[row_j, "xc_fill"]:
            q1 = Q["c"][max(np.searchsorted(Q["tk"], t1, "right") - 1, 0)]
        else:
            q1 = Q["o"][min(np.searchsorted(Q["tk"], t1), len(Q["tk"]) - 1)]
    else:
        nd = R.at[row_j, "nday"]
        q1 = Q["dc"].get(nd, np.nan) if nd is not None else np.nan
    return q1 / q0 - 1


def half_of(dy, per):
    if per == "early":
        return "보조"
    return "앞" if dy < MID else "뒤"


def events(A, Q, per, lo, hi):
    tk, o, l, c, reg, day, mod = A["tk"], A["o"], A["l"], A["c"], A["reg"], A["day"], A["mod"]
    R = bar_returns(A)
    R["half"] = [half_of(x, per) for x in R.day]
    Rj = R.set_index("j", drop=False)
    piv = {x: R.pivot_table(index="day", columns="mod", values=x, aggfunc="first") for x in RULES}
    rows, cnt = [], []
    regi = np.where(reg)[0]
    for dy in pd.unique(day[regi]):
        if not (lo <= dy <= hi):
            continue
        ri = regi[day[regi] == dy]
        if mod[ri[0]] != 570:
            continue
        O = o[ri[0]]
        lows, closes, mods = l[ri], c[ri], mod[ri]
        for dd in DS:
            for M in MS:
                cand = np.flatnonzero((mods < 570 + M) & (lows <= O * (1 - dd)))
                if not len(cand):
                    continue
                Dp = cand[0]
                rc_ = np.flatnonzero(closes[Dp:] >= O)
                if not len(rc_) or Dp + rc_[0] + 1 >= len(ri):
                    cnt.append((A["sym"], dy, dd, M, "못 되찾음"))
                    continue
                kp = Dp + rc_[0]
                lw = lows[:kp + 1]
                Lp = len(lw) - 1 - int(np.argmin(lw[::-1]))
                k, L, e = ri[kp], ri[Lp], ri[kp + 1]
                spd = (tk[k] - tk[L]) / NS_MIN
                cnt.append((A["sym"], dy, dd, M, "되찾음"))
                win = R[(R.day == dy) & (R.tk >= tk[e]) & (R.tk < tk[e] + WIN * NS_MIN)]
                hf = half_of(dy, per)
                oth_days = [x for x in piv["a"].index if x != dy and lo <= x <= hi and half_of(x, per) == hf]
                row = dict(per=per, half=hf, sym=A["sym"], day=dy, dd=dd, M=M, O=O, low=lows[:kp + 1].min(),
                           drop=lows[:kp + 1].min() / O - 1, tD=A["idx"][ri[Dp]], tL=A["idx"][L], tk_=A["idx"][k],
                           te=A["idx"][e], spd=spd, P=o[e], awake=bool(mod[e] < AWAKE), nwin=len(win))
                for x in RULES:
                    row[x] = Rj.at[e, x]
                    row["B_day_" + x] = win[x].mean()
                    col = piv[x][mod[e]] if mod[e] in piv[x].columns else pd.Series(dtype=float)
                    row["B_oth_" + x] = col.reindex(oth_days).mean()
                    row["q_" + x] = q_ret(Q, Rj, x, e)
                row["why_a"] = Rj.at[e, "why_a"]
                row["xa_t"] = pd.Timestamp(Rj.at[e, "xa_t"], tz="UTC").tz_convert("America/New_York") if Rj.at[e, "xa_t"] > 0 else None
                row["xc_fill"] = Rj.at[e, "xc_fill"]
                row["B_day_cfill"] = win.xc_fill.astype(float).mean()
                row["a_bars"] = int(np.searchsorted(tk, Rj.at[e, "xa_t"]) - e) if Rj.at[e, "xa_t"] > 0 else -1
                row["xc_t"] = pd.Timestamp(Rj.at[e, "xc_t"], tz="UTC").tz_convert("America/New_York") if Rj.at[e, "xc_t"] > 0 else None
                rows.append(row)
    return rows, cnt


def build_all():
    Q = qqq("main")
    rows, cnt = [], []
    for s in SYMS:
        try:
            A = load_main(s)
        except FileNotFoundError:
            continue
        r_, c_ = events(A, Q, "main", LO, HI)
        rows += r_
        cnt += c_
    QE = qqq("early")
    SE = stop_lab.load("early")
    for s in SYMS:
        if s not in SE:
            continue
        r_, c_ = events(load_early(SE[s]), QE, "early", E_LO, E_HI)
        rows += r_
        cnt += c_
    ev = pd.DataFrame(rows)
    for x in RULES:
        ev["ex_day_" + x] = ev[x] - ev["B_day_" + x]
        ev["ex_oth_" + x] = ev[x] - ev["B_oth_" + x]
        ev["ex_q_" + x] = ev[x] - ev["q_" + x]
    return ev, pd.DataFrame(cnt, columns=["sym", "day", "dd", "M", "res"])


# ---------------- 통계 ----------------

def tcl(y, g):
    """평균과 날짜 군집 t."""
    y, g = np.asarray(y, float), np.asarray(g)
    ok = np.isfinite(y)
    y, g = y[ok], g[ok]
    n = len(y)
    if n < 3:
        return (y.mean() if n else np.nan), np.nan, n
    m = y.mean()
    gi = pd.factorize(g)[0]
    G = gi.max() + 1
    if G < 2:
        return m, np.nan, n
    S = np.bincount(gi, weights=y - m)
    se = np.sqrt(G / (G - 1) * (S ** 2).sum()) / n
    return m, (m / se if se > 0 else np.nan), n


def ols_cl(y, X, g):
    """y = a + X b, 날짜 군집 표준오차(CR1). 반환 (계수, t)."""
    X = np.column_stack([np.ones(len(y))] + ([X] if X.ndim == 1 else [X]))
    n, k = X.shape
    A = np.linalg.pinv(X.T @ X)
    b = A @ X.T @ y
    e = y - X @ b
    gi = pd.factorize(np.asarray(g))[0]
    G = gi.max() + 1
    S = np.zeros((G, k))
    np.add.at(S, gi, X * e[:, None])
    V = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1)) * A @ (S.T @ S) @ A
    se = np.sqrt(np.maximum(np.diag(V), 1e-300))
    return b, b / se


def sel(ev, half, dd, M, N, grp):
    x = ev[(ev.half == half) & (ev.dd == dd) & (ev.M == M)]
    f = x.spd <= 3 * N
    return x[f] if grp == "빠름" else (x[~f] if grp == "느림" else x)


def stats(x, rule):
    y = x[rule].values.astype(float)
    ok = np.isfinite(y)
    out = dict(n=int(ok.sum()), days=x.day[ok].nunique(), ret=np.nanmean(y) if ok.any() else np.nan,
               win=(y[ok] > 0).mean() if ok.any() else np.nan, awake=x.awake[ok].mean() if ok.any() else np.nan)
    for k in ("ex_day", "ex_oth", "ex_q"):
        m, t, _ = tcl(x[f"{k}_{rule}"].values, x.day.values)
        out[k], out["t_" + k] = m, t
    return out


def fs_diff(x, N, rule):
    """빠름 - 느림 ex_day 차(날짜 군집 t)."""
    y = x[f"ex_day_{rule}"].values.astype(float)
    fl = (x.spd <= 3 * N).values.astype(float)
    ok = np.isfinite(y)
    y, fl, g = y[ok], fl[ok], x.day.values[ok]
    if fl.sum() < 3 or (1 - fl).sum() < 3:
        return np.nan, np.nan
    b, t = ols_cl(y, fl, g)
    return b[1], t[1]


def all_tests(ev):
    rows = []
    for half in ("앞", "뒤", "보조"):
        for dd in DS:
            for M in MS:
                xa = ev[(ev.half == half) & (ev.dd == dd) & (ev.M == M)]
                for N in NS:
                    for rule in RULES:
                        for grp in ("빠름", "느림"):
                            st = stats(sel(ev, half, dd, M, N, grp), rule)
                            row = dict(half=half, dd=dd, M=M, N=N, rule=rule, grp=grp, **st)
                            if grp == "빠름":
                                row["fs_diff"], row["t_fs"] = fs_diff(xa, N, rule)
                            rows.append(row)
    return pd.DataFrame(rows)


def signflip_null(ev, keys, n=1000, seed=7):
    """앞 절반 빠름 32개 시험에 같은 날짜 부호 뒤집기를 적용한 최대 t 분포."""
    rng = np.random.default_rng(seed)
    days = sorted(ev[ev.half == "앞"].day.unique())
    di = {d: i for i, d in enumerate(days)}
    G = len(days)
    pre = []
    for dd, M, N, rule in keys:
        x = sel(ev, "앞", dd, M, N, "빠름")
        y = x[f"ex_day_{rule}"].values.astype(float)
        ok = np.isfinite(y)
        gi = np.array([di[d] for d in x.day.values[ok]], dtype=np.int64)
        pre.append((np.bincount(gi, weights=y[ok], minlength=G), np.bincount(gi, minlength=G).astype(float), ok.sum(),
                    len(np.unique(gi))))
    mx = []
    for _ in range(n):
        s = rng.choice([-1.0, 1.0], G)
        best = -np.inf
        for S, ng, nn, Gu in pre:
            if nn < 3 or Gu < 2:
                continue
            m = (s * S).sum() / nn
            res = s * S - ng * m
            se = np.sqrt(Gu / (Gu - 1) * (res ** 2).sum()) / nn
            if se > 0:
                best = max(best, m / se)
        mx.append(best)
    return np.percentile(mx, 95), np.median(mx)


def drop_top(x, rule, key):
    """ex_day 합에 가장 크게 기여한 key 하나를 빼고 다시 잰다."""
    col = f"ex_day_{rule}"
    s = x.groupby(key)[col].sum()
    if not len(s):
        return None, np.nan, np.nan, 0
    top = s.idxmax()
    y = x[x[key] != top]
    m, t, n = tcl(y[col].values, y.day.values)
    return top, m, t, n


# ---------------- 출력 ----------------

def label(dd, M, N, rule):
    return f"d{dd * 100:.1f}% M{M} N{N} {RNAME[rule]}"


def main():
    os.makedirs(OUT, exist_ok=True)
    say("시가 되찾음 매수 연구 (fr_reclaim.py, 2026-10-10)")
    say("[시험 전 고정 정의]")
    say("  사건: 정규장 시가 O, 개장 뒤 M분 안 저가 <= O(1-d) → 그 뒤 3분봉 종가 >= O인 첫 봉 k → 다음 봉 시가 매수. 종목·날짜·정의당 1건.")
    say("  빠름 = 저점 봉(개장~k 최저, k 시점에 앎) → k가 3N분 이내, 느림 = 그보다 늦게 되찾음(대조). 못 되찾으면 안 삼.")
    say("  격자: d {1.5%, 2.5%} x M {30, 60분} x N {10, 20봉} = 8 정의 x 보유 4 = 빠름 32개(고르기 대상) + 느림 32 + 빠름-느림 차 32(비교용).")
    say("  보유: a RSI14>70 다음 봉/08시(ET 19:00), b 당일 종가, c +3% 지정가/08시, d 다음 거래일 종가. 자는 동안 가능 = c만.")
    say("  결과: ret, ex_day(주) = ret - 같은 종목·같은 날 진입 뒤 30분 안 아무 봉 진입 평균, ex_oth = ret - 같은 종목·같은 시각 다른 날 평균,")
    say("        ex_q = ret - 같은 시각 QQQ 수익. t = 날짜 군집. 프리·애프터 매도는 1분 꼬리(직전 1분 종가 ±1% 초과) 배제.")
    say("  고르기 9/10~9/23(10일) → 확인 9/24~10/9(12일), 보조 7/14~9/8 5분봉. 판정: 확인됨 = 뒤 ex_day>0·t>=2·상위 1종목/1일 빼도 >0·ex_q>0.")
    say(f"  종목 {len(SYMS)}개(GOOG·SOXX 제외). 진입 '깸' = 11:00 ET(한국 24시) 전 진입 비율.")
    ev, cnt = build_all()
    ev.to_csv(os.path.join(OUT, "fr_reclaim_events.csv"), index=False, encoding="utf-8-sig")

    say("")
    say("1. 사건 수(종목·날짜, 정의별: 낙폭 발생 / 되찾음 / 빠름 N10 / 빠름 N20)")
    cnt["half"] = [("보조" if x < LO else ("앞" if x < MID else "뒤")) for x in cnt.day]
    for half in ("앞", "뒤", "보조"):
        for dd in DS:
            for M in MS:
                c = cnt[(cnt.half == half) & (cnt.dd == dd) & (cnt.M == M)]
                x = ev[(ev.half == half) & (ev.dd == dd) & (ev.M == M)]
                say(f"  {half:3s} d{dd * 100:.1f}% M{M:2d}: 낙폭 {len(c):4d}  되찾음 {len(x):4d}({len(x) / max(len(c), 1):.0%})  "
                    f"빠름10 {int((x.spd <= 30).sum()):4d}  빠름20 {int((x.spd <= 60).sum()):4d}  날짜 {x.day.nunique():2d}  깸 {x.awake.mean():.0%}")

    T = all_tests(ev)
    T.to_csv(os.path.join(OUT, "fr_reclaim_tests.csv"), index=False, encoding="utf-8-sig")

    say("")
    say("2. 앞 절반(9/10~9/23) 고르기: 빠름 32개. ex = ret - 기준, ( ) = 날짜 군집 t. 느림 = 같은 정의 늦게 되찾음의 ex_day")
    say(f"  {'규칙':28s} {'n':>4s} {'ret':>7s} {'승률':>4s} {'ex_day':>14s} {'ex_oth':>14s} {'ex_q':>14s} {'느림 ex_day':>14s} {'빠름-느림':>14s}")
    F = T[(T.half == "앞") & (T.grp == "빠름")].reset_index(drop=True)
    Sl = T[(T.half == "앞") & (T.grp == "느림")].reset_index(drop=True)
    for i, r in F.iterrows():
        s = Sl.iloc[i]
        say(f"  {label(r.dd, r.M, r.N, r.rule):28s} {r.n:4d} {pct(r.ret):>7s} {r.win:4.0%} {pct(r.ex_day):>7s}({r.t_ex_day:5.2f}) "
            f"{pct(r.ex_oth):>7s}({r.t_ex_oth:5.2f}) {pct(r.ex_q):>7s}({r.t_ex_q:5.2f}) {pct(s.ex_day):>7s}({s.t_ex_day:5.2f}) "
            f"{pct(r.fs_diff):>7s}({r.t_fs:5.2f})")
    keys = [(r.dd, r.M, r.N, r.rule) for r in F.itertuples()]
    q95, q50 = signflip_null(ev, keys)
    say(f"  선택편향: 날짜 부호 뒤집기 1,000번, 32개 시험 최대 t(ex_day)의 95% 분위 {q95:.2f}, 중앙값 {q50:.2f}.")
    pick = F[(F.ex_day > 0) & (F.n >= 15)].sort_values("t_ex_day", ascending=False).head(3)
    say("  고른 규칙(ex_day t 상위, 평균 > 0, n >= 15): " + (", ".join(f"{label(r.dd, r.M, r.N, r.rule)}(t {r.t_ex_day:.2f})"
                                                           for r in pick.itertuples()) or "없음"))

    say("")
    say("3. 뒤 절반(9/24~10/9) 확인과 보조(7/14~9/8 5분봉)")
    cands = []
    for r in pick.itertuples():
        dd, M, N, rule = r.dd, r.M, r.N, r.rule
        say(f"[{label(dd, M, N, rule)}]  자는 동안 실행 가능: {'예' if SLEEP_OK[rule] else '아니오'}")
        out = {}
        for half in ("앞", "뒤", "보조"):
            x = sel(ev, half, dd, M, N, "빠름")
            st = stats(x, rule)
            xa = ev[(ev.half == half) & (ev.dd == dd) & (ev.M == M)]
            fd, ft = fs_diff(xa, N, rule)
            sl = stats(sel(ev, half, dd, M, N, "느림"), rule)
            out[half] = (st, fd, ft, sl)
            say(f"   {half:3s} n {st['n']:3d}(날짜 {st['days']:2d}) ret {pct(st['ret'])} 승률 {st['win']:.0%}  ex_day {pct(st['ex_day'])}(t {st['t_ex_day']:.2f})"
                f"  ex_oth {pct(st['ex_oth'])}(t {st['t_ex_oth']:.2f})  ex_q {pct(st['ex_q'])}(t {st['t_ex_q']:.2f})"
                f"  | 느림 n {sl['n']} ex_day {pct(sl['ex_day'])}  빠름-느림 {pct(fd)}(t {ft:.2f})")
        xb = sel(ev, "뒤", dd, M, N, "빠름")
        xb = xb[np.isfinite(xb[f"ex_day_{rule}"])]
        ts, ms, tts, ns_ = drop_top(xb, rule, "sym")
        td, md, ttd, nd_ = drop_top(xb, rule, "day")
        aw = xb[xb.awake]
        ma, ta, na = tcl(aw[f"ex_day_{rule}"].values, aw.day.values)
        say(f"   뒤 상위 1종목({ts}) 제외 ex_day {pct(ms)}(t {tts:.2f}, n {ns_}) / 상위 1일({td}) 제외 {pct(md)}(t {ttd:.2f}, n {nd_})")
        say(f"   뒤 11:00 ET 전 진입만(깸) ex_day {pct(ma)}(t {ta:.2f}, n {na})")
        bst = out["뒤"][0]
        if not (bst["ex_day"] > 0):
            vd = "뒤에서 실패"
        elif bst["t_ex_day"] >= 2 and ms > 0 and md > 0 and bst["ex_q"] > 0:
            vd = "확인됨"
        else:
            vd = "약함"
        say(f"   판정: {vd}")
        cands.append(dict(dd=dd, M=M, N=N, rule=rule, out=out, top=(ts, ms, tts, td, md, ttd), awake=(ma, ta, na), verdict=vd))

    say("")
    say("4. 뒤 절반 전체(빠름 32개, 고르기와 무관한 참고): ex_day 평균(t)")
    B = T[(T.half == "뒤") & (T.grp == "빠름")]
    for r in B.itertuples():
        say(f"  {label(r.dd, r.M, r.N, r.rule):28s} n {r.n:3d} ret {pct(r.ret)} ex_day {pct(r.ex_day)}({r.t_ex_day:5.2f}) "
            f"ex_q {pct(r.ex_q)}({r.t_ex_q:5.2f}) 빠름-느림 {pct(r.fs_diff)}({r.t_fs:5.2f})")
    pos_b = int((B.ex_day > 0).sum())
    say(f"  뒤 절반 빠름 32개 중 ex_day > 0: {pos_b}개, t >= 2: {int((B.t_ex_day >= 2).sum())}개, t <= -2: {int((B.t_ex_day <= -2).sum())}개")

    say("")
    say("5. 참고(빠름 N10): +3% 지정가 08시 전 체결률 사건 대 같은 날 30분 안 아무 봉, a 규칙 RSI 매도 비율·보유 봉 중앙값")
    for half in ("앞", "뒤", "보조"):
        for dd in DS:
            x = sel(ev, half, dd, 30, 10, "빠름")
            x = x[np.isfinite(x.c)]
            say(f"  {half:3s} d{dd * 100:.1f}% M30 N10: n {len(x):4d}  +3% 체결 {x.xc_fill.astype(float).mean():.0%} 대 {x.B_day_cfill.mean():.0%}  "
                f"a RSI 매도 {(x.why_a == 'RSI').mean():.0%}, 보유 중앙 {x.a_bars.median():.0f}봉")

    say("")
    say("6. 사례(모든 정의, 사건이면 진입·결과)")
    case_rows = []
    for sym, dy in (("CRCL", date(2026, 10, 9)), ("AMD", date(2026, 10, 6))):
        x = ev[(ev.sym == sym) & (ev.day == dy) & (ev.per == "main")]
        for dd in DS:
            for M in MS:
                y = x[(x.dd == dd) & (x.M == M)]
                if not len(y):
                    say(f"  {sym} {dy} d{dd * 100:.1f}% M{M}: 사건 아님(낙폭 미달 또는 못 되찾음)")
                    continue
                e = y.iloc[0]
                say(f"  {sym} {dy} d{dd * 100:.1f}% M{M}: O {e.O:.2f} 저점 {e.low:.2f}({pct(e["drop"])}) {e.tL:%H:%M} → 되찾음 {e.tk_:%H:%M} "
                    f"속도 {e.spd:.0f}분(N10 {'빠름' if e.spd <= 30 else '느림'}, N20 {'빠름' if e.spd <= 60 else '느림'}) 매수 {e.te:%H:%M} {e.P:.2f}")
                say(f"      a {pct(e.a)}({e.why_a} {e.xa_t:%m/%d %H:%M}) B_day {pct(e.B_day_a)} | b {pct(e.b)} B_day {pct(e.B_day_b)} | "
                    f"c {pct(e.c)}({'+3% 체결' if e.xc_fill else '08시'} {e.xc_t:%m/%d %H:%M}) B_day {pct(e.B_day_c)} | d {pct(e.d)}")
                case_rows.append(e)

    with open(os.path.join(OUT, "fr_reclaim.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    return ev, T, pick, cands


if __name__ == "__main__":
    main()
