"""반박 검증: 시가 되찾음 매수(fr_reclaim.py)에서 앞 절반이 고른 후보 3개.
  1) d1.5% M60 N10 빠름 → b 당일 종가   2) d1.5% M60 N10 빠름 → d 다음 날 종가   3) d1.5% M60 N20 빠름 → b 당일 종가
기본값은 '살아남지 못함'. 정의·격자·판정은 fr_reclaim.py 맨 앞 고정값을 그대로 쓰고 바꾸지 않는다.

독립 재계산(분석가 코드 재사용 안 함):
  사건(O·D·L·k·속도·매수 봉 e)을 1분봉 → 정규장 3분봉(09:30 기준 3분 묶음)으로 직접 다시 찾는다.
  b·d 수익, B_day(진입 봉부터 30분 안 아무 봉 평균), QQQ 같은 시각 수익을 직접 계산해 분석가 CSV와 맞춘다.
  a(RSI)·c(+3% 지정가)만 fr_reclaim.bar_returns를 빌린다(지표 시뮬, 가짜 사건 절차 재현용).
추가 반박 시험:
  ex_x   = ret - 같은 날·같은 3분봉에 다른 134종목이 같은 규칙으로 낸 평균(시장 전체 반등일 통제, 횡단면 시간 맞춤 무작위)
  ex_qb  = ret - beta x QQQ 같은 시각 수익. beta는 9/9 이전 120거래일 일봉(사전 값)
  가짜 사건 절차(300번): 같은 날 안에서 사건 종목 표시를 다른 종목으로 섞고(진입 시각·빠름 표시·정의 구조 유지)
       앞 절반 32개 고르기 → 뒤 절반 판정까지 같은 절차. 최대 t 분포·'확인됨' 거짓 양성 비율.
  빠름 표시 섞기(1,000번): 같은 날·같은 정의 안에서 빠름/느림 표시만 섞어 빠름-느림 차의 우연 분포.
  꼬리: 1분 저가가 시가·종가·직전 종가 중 최저보다 1% 넘게 아래인 분을 몸통으로 바꿔 사건을 다시 찾는다.
        매수 봉 첫 분 시가가 직전 1분 종가에서 1% 넘게 튄 경우를 센다.
  자는 동안: b·d 청산은 16:00 ET(한국 05:00) 종가 = 장 마감 시장가라 위쪽 지정가가 아님. 매수는 시장가(깨어 있어야 함).
결과: ../results/verify_fr_reclaim.txt
사용: py verify_fr_reclaim.py"""
import os, sys, time
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fr_reclaim as FR

RAW, OUT = FR.RAW, FR.OUT
CACHE = os.path.join(FR.SP, "verify_frr_perbar.parquet")
SYMS = FR.SYMS
LO, MID, HI = date(2026, 9, 10), date(2026, 9, 24), date(2026, 10, 9)
DS, MS, NS, RULES = (0.015, 0.025), (30, 60), (10, 20), ("a", "b", "c", "d")
CANDS = [(0.015, 60, 10, "b"), (0.015, 60, 10, "d"), (0.015, 60, 20, "b")]
NSEC = 60 * 10**9
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def pct(x):
    return "  nan" if not np.isfinite(x) else f"{x * 100:+.2f}%"


def tcl(y, g):
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


def diff_cl(y, f, g):
    """y = a + b f, 날짜 군집 t(CR1)."""
    y, f, g = np.asarray(y, float), np.asarray(f, float), np.asarray(g)
    ok = np.isfinite(y)
    y, f, g = y[ok], f[ok], g[ok]
    if f.sum() < 3 or (1 - f).sum() < 3:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(y)), f])
    A = np.linalg.inv(X.T @ X)
    b = A @ X.T @ y
    e = y - X @ b
    gi = pd.factorize(g)[0]
    G = gi.max() + 1
    S = np.zeros((G, 2))
    np.add.at(S, gi, X * e[:, None])
    n = len(y)
    V = G / max(G - 1, 1) * (n - 1) / max(n - 2, 1) * A @ (S.T @ S) @ A
    return b[1], b[1] / np.sqrt(V[1, 1])


# ---------------- 봉별 표(독립 b·d + 분석가 a·c) ----------------

def reg3(m):
    """1분봉 → 정규장 3분봉(09:30 기준). 반환 DataFrame(day, mod, o, h, l, c, o1pc: 첫 분 시가/직전 1분 종가)."""
    mod = m.index.hour * 60 + m.index.minute
    pc = m.Close.shift(1)
    r = m[(mod >= 570) & (mod < 960)].copy()
    r["pc"] = pc[(mod >= 570) & (mod < 960)]
    r["day"] = r.index.date
    r["mod"] = 570 + (r.index.hour * 60 + r.index.minute - 570) // 3 * 3
    # 꼬리 정리 저가: 1분 저가가 min(시가, 종가, 직전 종가)보다 1% 넘게 아래면 몸통 최저로
    body = np.minimum(np.minimum(r.Open, r.Close), r.pc.fillna(r.Open))
    r["lc"] = np.where(r.Low < body * 0.99, np.minimum(r.Open, r.Close), r.Low)
    g = r.groupby(["day", "mod"], sort=True)
    b = pd.DataFrame({"o": g.Open.first(), "h": g.High.max(), "l": g.Low.min(), "lc": g.lc.min(), "cl": g.Close.last(),
                      "pc0": g.pc.first()}).reset_index()
    b["tk"] = [pd.Timestamp(d).tz_localize("America/New_York").value + (md * 60) * 10**9 for d, md in zip(b.day, b["mod"])]
    return b


def qqq_tab():
    m = pd.read_parquet(os.path.join(RAW, "1m_QQQ.parquet"))
    m = m[m.High > 0]
    mod = m.index.hour * 60 + m.index.minute
    rg = m[(mod >= 570) & (mod < 960)]
    dc = rg.Close.groupby(rg.index.date).last()
    days = sorted(dc.index)
    return dict(tk=m.index.as_unit("ns").asi8, o=m.Open.values.astype(float), c=m.Close.values.astype(float),
                dc=dc.to_dict(), nd={days[i]: dc[days[i + 1]] for i in range(len(days) - 1)})


def betas():
    q = pd.read_parquet(os.path.join(RAW, "1d_QQQ.parquet")).Close
    q = q[q.index < "2026-09-10"].pct_change()
    out = {}
    for s in SYMS:
        p = os.path.join(RAW, f"1d_{s}.parquet")
        if not os.path.exists(p):
            continue
        x = pd.read_parquet(p).Close
        x = x[x.index < "2026-09-10"].pct_change()
        j = pd.concat([x, q], axis=1, join="inner").dropna().tail(120)
        out[s] = np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var() if len(j) > 40 else 1.0
    return out


def fwd_mean(tk, day, v, win=30 * NSEC):
    """같은 날 [tk, tk+30분) 안 봉 v 평균(NaN 제외)."""
    out = np.full(len(tk), np.nan)
    vv = np.where(np.isfinite(v), v, 0.0)
    cnt = np.isfinite(v).astype(float)
    cs, cc = np.r_[0, np.cumsum(vv)], np.r_[0, np.cumsum(cnt)]
    hi = np.searchsorted(tk, tk + win)
    # 같은 날로 자르기
    dlast = pd.Series(np.arange(len(tk))).groupby(day).transform("max").values + 1
    hi = np.minimum(hi, dlast)
    lo = np.arange(len(tk))
    nn = cc[hi] - cc[lo]
    out = np.where(nn > 0, (cs[hi] - cs[lo]) / np.maximum(nn, 1), np.nan)
    return out


def build_perbar():
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    Q = qqq_tab()
    frames = []
    t0 = time.time()
    for i, s in enumerate(SYMS):
        p = os.path.join(RAW, f"1m_{s}.parquet")
        if not os.path.exists(p):
            continue
        m = pd.read_parquet(p)
        m = m[m.High > 0]
        b = reg3(m)
        dcl = b.groupby("day").cl.last()
        days = sorted(dcl.index)
        nd = {days[k]: dcl[days[k + 1]] for k in range(len(days) - 1)}
        b["rb"] = b.day.map(dcl) / b.o - 1
        b["rd"] = b.day.map(nd) / b.o - 1
        # 분석가 a·c(지표 시뮬)
        A = FR.load_main(s)
        R = FR.bar_returns(A)
        R = R[["day", "mod", "a", "c", "b", "d", "xa_t", "xc_t", "xc_fill"]].rename(columns={"b": "b_fr", "d": "d_fr"})
        b = b.merge(R, on=["day", "mod"], how="left")
        b["sym"] = s
        frames.append(b)
        if i % 20 == 0:
            print(f"  {i}/{len(SYMS)} {s} {time.time() - t0:.0f}s", flush=True)
    P = pd.concat(frames, ignore_index=True)
    P = P.sort_values(["sym", "tk"]).reset_index(drop=True)
    # QQQ 같은 시각 수익
    qi0 = np.minimum(np.searchsorted(Q["tk"], P.tk.values), len(Q["tk"]) - 1)
    q0 = Q["o"][qi0]
    P["q_b"] = P.day.map(Q["dc"]).values / q0 - 1
    P["q_d"] = P.day.map(Q["nd"]).astype(float).values / q0 - 1
    xa = np.rint(P.xa_t.fillna(-1).values.astype(float) / 1e9).astype(np.int64) * 10**9
    qa = Q["o"][np.minimum(np.searchsorted(Q["tk"], xa), len(Q["tk"]) - 1)]
    P["q_a"] = np.where(xa > 0, qa / q0 - 1, np.nan)
    xc = np.rint(P.xc_t.fillna(-1).values.astype(float) / 1e9).astype(np.int64) * 10**9
    qcf = Q["c"][np.maximum(np.searchsorted(Q["tk"], xc, "right") - 1, 0)]
    qco = Q["o"][np.minimum(np.searchsorted(Q["tk"], xc), len(Q["tk"]) - 1)]
    P["q_c"] = np.where(xc > 0, np.where(P.xc_fill.fillna(False).astype(bool), qcf, qco) / q0 - 1, np.nan)
    P = P.rename(columns={"rb": "b", "rd": "d"})
    # B_day(진입 봉 포함 30분 창) - 종목별
    for x in RULES:
        P["Bd_" + x] = np.nan
    for s, g in P.groupby("sym"):
        ix = g.index.values
        for x in RULES:
            P.loc[ix, "Bd_" + x] = fwd_mean(g.tk.values, g.day.values, g[x].values.astype(float))
    # 횡단면 기준: 같은 날·같은 봉 다른 종목 평균
    for x in RULES:
        v = P[x].astype(float)
        ok = np.isfinite(v)
        key = [P.day, P["mod"]]
        sm = v.where(ok, 0).groupby(key).transform("sum")
        ct = ok.astype(float).groupby(key).transform("sum")
        own_v = v.where(ok, 0)
        P["X_" + x] = (sm - own_v) / (ct - ok.astype(float)).replace(0, np.nan)
    P.to_parquet(CACHE)
    return P


# ---------------- 사건(독립) ----------------

def find_events(P, lowcol="l"):
    rows = []
    for (s, dy), g in P.groupby(["sym", "day"], sort=False):
        if not (LO <= dy <= HI) or g["mod"].iloc[0] != 570:
            continue
        O = g.o.iloc[0]
        lows, closes, mods = g[lowcol].values, g.cl.values, g["mod"].values
        idx = g.index.values
        for dd in DS:
            for M in MS:
                cand = np.flatnonzero((mods < 570 + M) & (lows <= O * (1 - dd)))
                if not len(cand):
                    continue
                Dp = cand[0]
                rc = np.flatnonzero(closes[Dp:] >= O)
                if not len(rc) or Dp + rc[0] + 1 >= len(g):
                    continue
                kp = Dp + rc[0]
                lw = lows[:kp + 1]
                Lp = len(lw) - 1 - int(np.argmin(lw[::-1]))
                spd = (mods[kp] - mods[Lp])
                rows.append(dict(sym=s, day=dy, dd=dd, M=M, O=O, low=lw.min(), spd=spd, mod_k=mods[kp],
                                 mod_e=mods[kp + 1], pi=idx[kp + 1]))
    ev = pd.DataFrame(rows)
    ev["half"] = np.where(ev.day < MID, "앞", "뒤")
    return ev


def attach(ev, P, beta):
    R = P.loc[ev.pi.values]
    out = ev.copy()
    out["P"] = R.o.values
    out["o1pc"] = (R.o / R.pc0 - 1).values
    for x in RULES:
        r = R[x].values.astype(float)
        out[x] = r
        out["ex_day_" + x] = r - R["Bd_" + x].values
        out["ex_q_" + x] = r - R["q_" + x].values
        out["ex_x_" + x] = r - R["X_" + x].values
        out["ex_qb_" + x] = r - out.sym.map(beta).fillna(1.0).values * R["q_" + x].values
    return out


def sel(ev, half, dd, M, N, grp="빠름"):
    x = ev[(ev.half == half) & (ev.dd == dd) & (ev.M == M)]
    f = x.spd <= 3 * N
    return x[f] if grp == "빠름" else x[~f]


def drop_top(x, col, key):
    s = x.groupby(key)[col].sum()
    top = s.idxmax()
    y = x[x[key] != top]
    m, t, n = tcl(y[col], y.day)
    return top, m, t, n


def verdict(xb, rule):
    """fr_reclaim 고정 판정. 반환 (판정, 평균, t)."""
    col = f"ex_day_{rule}"
    xb = xb[np.isfinite(xb[col])]
    m, t, n = tcl(xb[col], xb.day)
    if not (m > 0):
        return "실패", m, t
    _, ms, _, _ = drop_top(xb, col, "sym")
    _, md, _, _ = drop_top(xb, col, "day")
    mq = np.nanmean(xb[f"ex_q_{rule}"])
    return ("확인됨" if (t >= 2 and ms > 0 and md > 0 and mq > 0) else "약함"), m, t


# ---------------- 가짜 사건 절차 ----------------

def procedure(ev):
    """앞 절반 빠름 32개 → t 상위 3(평균>0, n>=15) → 뒤 판정. 반환 (최대 t, [(키, 판정, 뒤 평균, 뒤 t)])."""
    res = []
    for dd in DS:
        for M in MS:
            for N in NS:
                x = sel(ev, "앞", dd, M, N)
                for r in RULES:
                    m, t, n = tcl(x[f"ex_day_{r}"], x.day)
                    res.append(((dd, M, N, r), m, t, n))
    tmax = np.nanmax([z[2] for z in res])
    pk = sorted([z for z in res if z[1] > 0 and z[3] >= 15 and np.isfinite(z[2])], key=lambda z: -z[2])[:3]
    outs = []
    for (dd, M, N, r), m, t, n in pk:
        v, mb, tb = verdict(sel(ev, "뒤", dd, M, N), r)
        outs.append(((dd, M, N, r), v, mb, tb))
    return tmax, outs


def fake_events(ev, P, rng, keymap, daysyms):
    """같은 날 안에서 사건 종목을 다른 종목으로 섞는다(같은 종목·날 사건은 같은 가짜 종목으로)."""
    mp = {}
    for dy, syms in ev.groupby("day").sym.unique().items():
        pool = daysyms[dy]
        pick = rng.choice(pool, size=len(syms), replace=False)
        for s, f in zip(syms, pick):
            mp[(s, dy)] = f
    fs = [mp[(s, d)] for s, d in zip(ev.sym.values, ev.day.values)]
    pi = np.array([keymap.get((f, d, me), -1) for f, d, me in zip(fs, ev.day.values, ev.mod_e.values)])
    ok = pi >= 0
    fe = ev[ok].copy()
    fe["sym"] = np.array(fs)[ok]
    R = P.loc[pi[ok]]
    for x in RULES:
        r = R[x].values.astype(float)
        fe["ex_day_" + x] = r - R["Bd_" + x].values
        fe["ex_q_" + x] = r - R["q_" + x].values
        fe["ex_x_" + x] = r - R["X_" + x].values
    return fe


def main():
    say("반박 검증: 시가 되찾음 매수 후보 3개 (verify_fr_reclaim.py, 2026-10-10)")
    say("[고정값 그대로] d {1.5,2.5%} x M {30,60분} x N {10,20봉}, 보유 a·b·c·d, 앞 9/10~9/23 → 뒤 9/24~10/9, 판정 = 뒤 ex_day>0·t>=2·상위 1종목/1일 빼도 >0·ex_q>0")
    say("[추가 기준] ex_x = 같은 날·같은 봉 다른 종목 평균 대비, ex_qb = 사전 120일 베타 x QQQ 대비. 판단은 k 봉 종가까지의 값만 씀")
    P = build_perbar()
    beta = betas()
    keymap = {(s, d, md): i for i, s, d, md in zip(P.index.values, P.sym.values, P.day.values, P["mod"].values)}
    daysyms = {d: g.sym.unique() for d, g in P[P["mod"] == 570].groupby("day")}

    # ---- 1. 독립 재계산 대조 ----
    ev = attach(find_events(P), P, beta)
    fr = pd.read_csv(os.path.join(OUT, "fr_reclaim_events.csv"), parse_dates=["te"])
    fr = fr[fr.per == "main"].copy()
    fr["day"] = pd.to_datetime(fr.day).dt.date
    fr["mod_e"] = fr.te.apply(lambda t: pd.Timestamp(t).hour * 60 + pd.Timestamp(t).minute)
    j = ev.merge(fr, on=["sym", "day", "dd", "M"], how="outer", suffixes=("", "_fr"), indicator=True)
    say("")
    say("1. 독립 재계산 대조(정규장 1분봉 → 3분봉, 분석가 코드 미사용)")
    say(f"  사건 수: 내 {len(ev)} / 분석가 {len(fr)} / 둘 다 {int((j._merge == 'both').sum())} / 내만 {int((j._merge == 'left_only').sum())} / 분석가만 {int((j._merge == 'right_only').sum())}")
    b = j[j._merge == "both"]
    say(f"  매수 봉 다름 {int((b.mod_e != b.mod_e_fr).sum())}건, 속도 다름 {int((b.spd != b.spd_fr).sum())}건, 매수가 최대 차 {np.nanmax(np.abs(b.P / b.P_fr - 1)):.2e}")
    for x in ("b", "d"):
        say(f"  {x}: 수익 최대 차 {np.nanmax(np.abs(b[x] - b[x + '_fr'])):.2e}, ex_day 최대 차 {np.nanmax(np.abs(b['ex_day_' + x] - b['ex_day_' + x + '_fr'])):.2e}, "
            f"ex_q 최대 차 {np.nanmax(np.abs(b['ex_q_' + x] - b['ex_q_' + x + '_fr'])):.2e}")
    for x in ("a", "c"):
        say(f"  {x}(지표 시뮬 공용): ex_day 최대 차 {np.nanmax(np.abs(b['ex_day_' + x] - b['ex_day_' + x + '_fr'])):.2e}")

    # ---- 2. 후보 재측정 ----
    say("")
    say("2. 후보 재측정: 평균(날짜 군집 t). ex_day 주, ex_q·ex_x·ex_qb 시장 통제")
    real = {}
    for dd, M, N, r in CANDS:
        lab = f"d{dd * 100:.1f}% M{M} N{N} {FR.RNAME[r]}"
        say(f"[{lab}]")
        for half in ("앞", "뒤"):
            x = sel(ev, half, dd, M, N)
            x = x[np.isfinite(x[r])]
            st = {k: tcl(x[f"{k}_{r}"], x.day) for k in ("ex_day", "ex_q", "ex_x", "ex_qb")}
            xa = ev[(ev.half == half) & (ev.dd == dd) & (ev.M == M)]
            xa = xa[np.isfinite(xa[r])]
            fd, ft = diff_cl(xa[f"ex_day_{r}"], (xa.spd <= 3 * N).astype(float), xa.day)
            fx, fxt = diff_cl(xa[f"ex_x_{r}"], (xa.spd <= 3 * N).astype(float), xa.day)
            say(f"   {half} n {len(x)}(날짜 {x.day.nunique()}) ret {pct(x[r].mean())} 승률 {(x[r] > 0).mean():.0%} | "
                + "  ".join(f"{k} {pct(v[0])}({v[1]:.2f})" for k, v in st.items())
                + f" | 빠름-느림 ex_day {pct(fd)}({ft:.2f}) ex_x {pct(fx)}({fxt:.2f})")
            real[(dd, M, N, r, half)] = st
        xb = sel(ev, "뒤", dd, M, N)
        xb = xb[np.isfinite(xb[r])]
        for col in (f"ex_day_{r}", f"ex_x_{r}"):
            ts, ms, tts, ns_ = drop_top(xb, col, "sym")
            td, md, ttd, nd_ = drop_top(xb, col, "day")
            say(f"   뒤 {col[:-2]}: 상위 1종목({ts}) 제외 {pct(ms)}(t {tts:.2f}) / 상위 1일({td}) 제외 {pct(md)}(t {ttd:.2f})")
        aw = xb[xb.mod_e < 660]
        m1, t1, n1 = tcl(aw[f"ex_day_{r}"], aw.day)
        m2, t2, n2 = tcl(aw[f"ex_x_{r}"], aw.day)
        say(f"   뒤 11:00 ET 전 진입만 n {n1}: ex_day {pct(m1)}({t1:.2f}) ex_x {pct(m2)}({t2:.2f}); 11시 뒤 진입 비율 {1 - len(aw) / len(xb):.0%}(실행 불가)")
        v, mb, tb = verdict(xb, r)
        say(f"   고정 판정 재현: {v} (뒤 ex_day {pct(mb)}, t {tb:.2f})")

    # ---- 3. 날짜 쏠림·시장 반등일 ----
    say("")
    say("3. 날짜 쏠림: 후보 1 뒤 절반 날짜별 건수·ex_day 합·QQQ 하루(시가→종가)·QQQ 개장 30분 저점")
    Q = pd.read_parquet(os.path.join(RAW, "1m_QQQ.parquet"))
    qm = Q.index.hour * 60 + Q.index.minute
    Qr = Q[(qm >= 570) & (qm < 960)]
    qd = Qr.groupby(Qr.index.date).agg(o=("Open", "first"), c=("Close", "last"))
    q30 = Qr[(Qr.index.hour * 60 + Qr.index.minute) < 630].groupby(Qr[(Qr.index.hour * 60 + Qr.index.minute) < 630].index.date).Low.min()
    dd, M, N, r = CANDS[0]
    xb = sel(ev, "뒤", dd, M, N)
    tab = xb.groupby("day").agg(n=("sym", "size"), s=(f"ex_day_{r}", "sum"), sx=(f"ex_x_{r}", "sum"))
    for dy, rr in tab.iterrows():
        say(f"   {dy} n {int(rr.n):3d} ex_day 합 {rr.s * 100:+6.2f}%p ex_x 합 {rr.sx * 100:+6.2f}%p | QQQ {pct(qd.c[dy] / qd.o[dy] - 1)} 30분 저점 {pct(q30[dy] / qd.o[dy] - 1)}")
    # QQQ도 되찾은 날 vs 아닌 날
    qrec = {}
    for dy in qd.index:
        g = Qr[Qr.index.date == dy]
        o = g.Open.iloc[0]
        lo = g[(g.index.hour * 60 + g.index.minute) < 630].Low.min()
        qrec[dy] = lo <= o * (1 - 0.005)
    for dd, M, N, r in CANDS:
        xb = sel(ev, "뒤", dd, M, N)
        xb = xb[np.isfinite(xb[r])]
        f = xb.day.map(qrec).astype(bool)
        a1 = tcl(xb[f][f"ex_day_{r}"], xb[f].day)
        a2 = tcl(xb[~f][f"ex_day_{r}"], xb[~f].day)
        say(f"   d{dd * 100:.1f}% M{M} N{N} {r}: QQQ도 개장 30분 -0.5% 이상 빠진 날 n {a1[2]} ex_day {pct(a1[0])}({a1[1]:.2f}) / 아닌 날 n {a2[2]} {pct(a2[0])}({a2[1]:.2f})")

    # ---- 4. 꼬리 ----
    say("")
    say("4. 꼬리 체결")
    ev_c = find_events(P, "lc")
    k1 = set(zip(ev.sym, ev.day, ev.dd, ev.M, ev.mod_e))
    k2 = set(zip(ev_c.sym, ev_c.day, ev_c.dd, ev_c.M, ev_c.mod_e))
    say(f"   1분 꼬리 저가를 몸통으로 바꾸면: 사건 {len(k1)} → {len(k2)}, 사라지거나 매수 봉이 바뀐 사건 {len(k1 - k2)}, 새로 생긴 {len(k2 - k1)}")
    ev_c = attach(ev_c, P, beta)
    for dd, M, N, r in CANDS:
        for half in ("앞", "뒤"):
            x = sel(ev_c, half, dd, M, N)
            m, t, n = tcl(x[f"ex_day_{r}"], x.day)
            say(f"   꼬리 정리 후 d{dd * 100:.1f}% M{M} N{N} {r} {half}: n {n} ex_day {pct(m)}({t:.2f})")
    for dd, M, N, r in CANDS:
        x = sel(ev, "뒤", dd, M, N)
        say(f"   매수 첫 분 시가가 직전 1분 종가에서 1% 넘게 튐: d{dd * 100:.1f}% M{M} N{N} 뒤 {int((x.o1pc.abs() > 0.01).sum())}/{len(x)}건")

    # ---- 5. 가짜 사건 절차 ----
    say("")
    NR = 300
    rng = np.random.default_rng(20261010)
    tmx, conf, weak, pk_b = [], 0, 0, []
    xnull = {c: [] for c in CANDS}
    t0 = time.time()
    for it in range(NR):
        fe = fake_events(ev, P, rng, keymap, daysyms)
        tm, outs = procedure(fe)
        tmx.append(tm)
        if any(o[1] == "확인됨" for o in outs):
            conf += 1
        if any(o[1] in ("확인됨", "약함") for o in outs):
            weak += 1
        pk_b += [o[2] for o in outs]
        for c in CANDS:
            x = sel(fe, "뒤", *c[:3])
            xnull[c].append(np.nanmean(x[f"ex_x_{c[3]}"]))
    tmx = np.array(tmx)
    tm_real, outs_real = procedure(ev)
    say(f"5. 가짜 사건 같은 절차 {NR}번(같은 날 안 사건 종목 섞기, 진입 시각·빠름 표시·정의 구조 유지, {time.time() - t0:.0f}초)")
    say(f"   실제: 앞 32개 최대 t {tm_real:.2f}, 고른 3개 뒤 판정 " + ", ".join(f"{o[1]}({pct(o[2])}, t {o[3]:.2f})" for o in outs_real))
    say(f"   가짜: 앞 최대 t 중앙 {np.median(tmx):.2f}, 95% {np.percentile(tmx, 95):.2f}, 실제 이상 비율 {(tmx >= tm_real).mean():.0%}")
    say(f"   가짜: 고른 규칙 중 '확인됨' 나온 회차 {conf}/{NR}({conf / NR:.1%}), '약함' 이상 {weak}/{NR}({weak / NR:.0%}), 고른 규칙 뒤 ex_day 평균 {pct(np.nanmean(pk_b))}")
    for c in CANDS:
        rv = tcl(sel(ev, "뒤", *c[:3])[f"ex_x_{c[3]}"], sel(ev, "뒤", *c[:3]).day)[0]
        nv = np.array(xnull[c])
        say(f"   d{c[0] * 100:.1f}% M{c[1]} N{c[2]} {c[3]} 뒤 ex_x 실제 {pct(rv)} vs 가짜 95% {pct(np.nanpercentile(nv, 95))}, 가짜가 실제 이상 {np.mean(nv >= rv):.0%}")

    # ---- 6. 빠름 표시 섞기 ----
    say("")
    NP = 1000
    say(f"6. 빠름 표시 섞기 {NP}번(같은 날·같은 정의 안), 뒤 절반 빠름-느림 차(단순 평균차)의 우연 분포")
    for dd, M, N, r in CANDS:
        xa = ev[(ev.half == "뒤") & (ev.dd == dd) & (ev.M == M)]
        xa = xa[np.isfinite(xa[r])].reset_index(drop=True)
        for col in (f"ex_day_{r}", f"ex_x_{r}"):
            y = xa[col].values
            f = (xa.spd <= 3 * N).values
            obs = y[f].mean() - y[~f].mean()
            grp = [np.flatnonzero(xa.day.values == d) for d in xa.day.unique()]
            nd = []
            for _ in range(NP):
                ff = f.copy()
                for gi in grp:
                    ff[gi] = rng.permutation(f[gi])
                nd.append(y[ff].mean() - y[~ff].mean())
            nd = np.array(nd)
            say(f"   d{dd * 100:.1f}% M{M} N{N} {col}: 실제 {pct(obs)}, 섞기 95% {pct(np.percentile(nd, 95))}, p(섞기 >= 실제) {np.mean(nd >= obs):.2f}")

    # ---- 7. 사례 ----
    say("")
    say("7. 사례 재현")
    for s, dy in (("CRCL", date(2026, 10, 9)), ("AMD", date(2026, 10, 6))):
        x = ev[(ev.sym == s) & (ev.day == dy) & (ev.dd == 0.015) & (ev.M == 60)]
        for e in x.itertuples():
            say(f"   {s} {dy}: O {e.O:.2f} 저점 {e.low:.2f}({pct(e.low / e.O - 1)}) 되찾음 봉 {e.mod_k // 60:02d}:{e.mod_k % 60:02d} 속도 {e.spd}분 "
                f"매수 {e.mod_e // 60:02d}:{e.mod_e % 60:02d} {e.P:.2f} | b {pct(e.b)} ex_day {pct(e.ex_day_b)} ex_x {pct(e.ex_x_b)} | d {pct(e.d)}")

    # ---- 8. 자는 동안 ----
    say("")
    say("8. 자는 동안 실행 가능 표시")
    say("   b·d: 청산이 16:00 ET(한국 05:00) 정규장 종가 = 장 마감 시장가(MOC). 위쪽 지정가가 아니고 08시 정리도 아님 → 아니오(분석가 표시와 같음).")
    say("   매수: k 다음 봉 시가 시장가 = 깨어 있어야 함(ET 11:00 = 한국 24시 전). 후보 1 뒤 절반 11시 뒤 진입은 위 2절 비율만큼 실행 불가.")
    with open(os.path.join(OUT, "verify_fr_reclaim.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
