"""반박 검증 — 일봉 A 최선안(A_N10_k1_m0_U지정가 + 중간 목표·바닥−3% 손절)을 정의 문장만 보고 독립 재구현(2026-10-09).

정의(연구자 best_rule.definition):
  F = 직전 N봉 저가 최저(오늘 제외), BH = 직전 N봉 고가 최고, ATR = Pine ta.atr(14)[1](RMA, 첫 값 = TR 14개 단순평균).
  U = F + k×ATR. 조건: not (low[1] <= U[1] and close[1] >= F[1]). 조건이면 U에 그날만 유효한 지정가 매수,
  low <= U면 체결가 min(open, U).
  매도: 다음 날부터 목표 T = max((F+BH)/2, 체결가×1.01) 지정가(시가가 T 위면 시가). 종가 < F×0.97이면 다음 날 시가(매수일 종가 포함).
        둘 다 아니면 매수 10거래일 뒤 시가. 데이터 끝(10/8) 넘으면 10/8 종가 평가(미완료).
기준: (1) 같은 날 무작위 = 같은 묶음 전 종목이 그날 시가에 사서 같은 매도 규칙(자기 F·BH),
      (2) 같은 날 묶음 평균의 '같은 보유 구간'(매수일 시가 → 실제 매도일 시가/10/8 종가) 매수 후 보유,
      (3) 같은 종목의 같은 기간 무작위 날 진입(같은 매도 규칙) 평균.
연구자 코드(daily_A.py)는 읽기만 했고 복사하지 않았다. 루프는 numba 거래 단위.
사용: py verify_daily_A.py → ../results/verify_daily_A.txt
"""
import glob, os, sys
import numpy as np, pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
DD = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/box/daily"
OUT = os.path.join(HERE, "../results/verify_daily_A.txt")
sys.path.insert(0, os.path.join(HERE, "../../dip_reason/code"))
from fetch import user_universe

LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


# ------------------------------------------------------------ 데이터 (날짜 = SPY 거래일)
files = sorted(glob.glob(os.path.join(DD, "1d_*.parquet")))
tick = [os.path.basename(f)[3:-8] for f in files]
spy = pd.read_parquet(os.path.join(DD, "1d_SPY.parquet"))
dates = pd.DatetimeIndex(spy.index[~spy.index.duplicated()]).normalize()
ND, NT = len(dates), len(tick)
Ox, Hx, Lx, Cx = (np.full((ND, NT), np.nan) for _ in range(4))
for j, f in enumerate(files):
    d = pd.read_parquet(f)
    d = d[~d.index.duplicated(keep="last")]
    d.index = pd.DatetimeIndex(d.index).normalize()
    d = d.reindex(dates)
    Ox[:, j], Hx[:, j], Lx[:, j], Cx[:, j] = d.Open.values, d.High.values, d.Low.values, d.Close.values
LAST = ND - 1

# 분할 의심: 시가 또는 종가가 전일 종가 대비 ≤0.6배 또는 ≥1.8배
prevC = np.full_like(Cx, np.nan)
prevC[1:] = Cx[:-1]
with np.errstate(invalid="ignore", divide="ignore"):
    jump = ((Cx / prevC) <= 0.6) | ((Cx / prevC) >= 1.8) | ((Ox / prevC) <= 0.6) | ((Ox / prevC) >= 1.8)
jump_days = [(tick[t], dates[d].date()) for d, t in zip(*np.nonzero(jump))]
# 거래 제외: 매수일 e 기준 [e-80, e+12] 안에 의심일(박스·ATR·보유 구간 전부 덮도록)
cj = np.vstack([np.zeros((1, NT)), np.cumsum(jump, axis=0)])
lo_ = np.clip(np.arange(ND) - 80, 0, ND)
hi_ = np.clip(np.arange(ND) + 13, 0, ND)
BADWIN = (cj[hi_] - cj[lo_]) > 0


def lowest_prev(X, n):
    """X[t-n..t-1]의 최소(오늘 제외). 창 안에 NaN 있으면 NaN."""
    R = np.full_like(X, np.nan)
    for t in range(n, ND):
        w = X[t - n:t]
        R[t] = np.where(np.isnan(w).any(axis=0), np.nan, np.nanmin(np.where(np.isnan(w), np.inf, w), axis=0))
    return R


def highest_prev(X, n):
    return -lowest_prev(-X, n)


def pine_atr_prev(n=14):
    """Pine ta.atr(n): TR(첫 봉은 H-L), RMA 시드 = 처음 n개 TR 평균. 반환값은 [1] 시프트(어제까지)."""
    tr = np.where(np.isnan(prevC), Hx - Lx, np.fmax(Hx - Lx, np.fmax(np.abs(Hx - prevC), np.abs(Lx - prevC))))
    A = np.full_like(tr, np.nan)
    for t in range(NT):
        col = tr[:, t]
        ok = np.nonzero(~np.isnan(col))[0]
        if len(ok) < n:
            continue
        s = ok[0]
        a = np.nan
        cnt = 0
        acc = 0.0
        for i in range(s, ND):
            v = col[i]
            if np.isnan(v):
                A[i, t] = np.nan
                continue
            if cnt < n:
                acc += v
                cnt += 1
                if cnt == n:
                    a = acc / n
                    A[i, t] = a
            else:
                a = (a * (n - 1) + v) / n
                A[i, t] = a
    S = np.full_like(A, np.nan)
    S[1:] = A[:-1]
    return S


ATRp = pine_atr_prev(14)
FCACHE, BCACHE = {}, {}


def box(n):
    if n not in FCACHE:
        FCACHE[n], BCACHE[n] = lowest_prev(Lx, n), highest_prev(Hx, n)
    return FCACHE[n], BCACHE[n]


# ------------------------------------------------------------ 거래 단위 매도 시뮬레이터
@njit(cache=False)
def run_exit(es, ts, Ps, Ts, Ss, tgt_from_same_day, O, H, L, C, last, cap):
    n = len(es)
    ret = np.full(n, np.nan)
    xday = np.zeros(n, np.int64)
    inc = np.zeros(n, np.bool_)
    for i in range(n):
        e, t, P, T, S = es[i], ts[i], Ps[i], Ts[i], Ss[i]
        done = False
        d = e
        while not done:
            if d > last:                      # 데이터 끝
                ret[i] = C[last, t] / P - 1
                xday[i] = last
                inc[i] = True
                break
            if d == e + cap:                  # 10거래일 뒤 시가
                ret[i] = O[d, t] / P - 1
                xday[i] = d
                break
            if d > e or tgt_from_same_day:
                if d > e and O[d, t] >= T:    # 시가가 목표 위
                    ret[i] = O[d, t] / P - 1
                    xday[i] = d
                    break
                if H[d, t] >= T:
                    ret[i] = T / P - 1
                    xday[i] = d
                    break
            if C[d, t] < S:                   # 종가 손절 → 다음 날 시가
                if d + 1 > last:
                    ret[i] = C[last, t] / P - 1
                    xday[i] = last
                    inc[i] = True
                else:
                    ret[i] = O[d + 1, t] / P - 1
                    xday[i] = d + 1
                break
            d += 1
    return ret, xday, inc


def signals(n, k, y, tgt="mid"):
    F, BH = box(n)
    U = F + k * ATRp
    with np.errstate(invalid="ignore"):
        heldprev = np.zeros((ND, NT), bool)
        heldprev[1:] = (Lx[:-1] <= U[:-1]) & (Cx[:-1] >= F[:-1])
        cond = ~heldprev & np.isfinite(U)
        fill = cond & (Lx <= U) & np.isfinite(Ox)
    e, t = np.nonzero(fill)
    P = np.minimum(Ox[e, t], U[e, t])
    ok = ~BADWIN[e, t] & np.isfinite(BH[e, t]) & (P > 0)
    e, t, P = e[ok], t[ok], P[ok]
    base = (F[e, t] + BH[e, t]) / 2 if tgt == "mid" else BH[e, t]
    T = np.maximum(base, P * 1.01)
    S = F[e, t] * (1 - y)
    r, x, inc = run_exit(e.astype(np.int64), t.astype(np.int64), P, T, S, False, Ox, Hx, Lx, Cx, LAST, 10)
    return pd.DataFrame(dict(e=e, t=t, P=P, ret=r, x=x, inc=inc, F=F[e, t], U=U[e, t], BH=BH[e, t]))


RCACHE = {}


def random_same_rule(n, y, tgt="mid", rows=None):
    """모든 종목·모든 날 시가 매수, 같은 매도 규칙(자기 F·BH). 시가 매수라 매수일 장중 목표도 인정."""
    key = (n, y, tgt)
    if key in RCACHE:
        return RCACHE[key]
    F, BH = box(n)
    rows = np.nonzero(dates >= "2025-12-01")[0]
    ee, tt = np.meshgrid(rows, np.arange(NT), indexing="ij")
    ee, tt = ee.ravel(), tt.ravel()
    P = Ox[ee, tt]
    ok = np.isfinite(P) & (P > 0) & np.isfinite(F[ee, tt]) & np.isfinite(BH[ee, tt]) & ~BADWIN[ee, tt]
    ee, tt, P = ee[ok], tt[ok], P[ok]
    base = (F[ee, tt] + BH[ee, tt]) / 2 if tgt == "mid" else BH[ee, tt]
    T = np.maximum(base, P * 1.01)
    S = F[ee, tt] * (1 - y)
    r, _, _ = run_exit(ee.astype(np.int64), tt.astype(np.int64), P, T, S, True, Ox, Hx, Lx, Cx, LAST, 10)
    M = np.full((ND, NT), np.nan)
    M[ee, tt] = r
    RCACHE[key] = M
    return M


# ------------------------------------------------------------ 묶음·기간
sp = set(pd.read_csv(os.path.join(DD, "sp500_list.csv")).iloc[:, 0].astype(str)) & set(tick)
u78 = set(user_universe()["합집합"]) & set(tick)
UNI = {"S&P500": np.array(sorted(tick.index(s) for s in sp)), "사용자78": np.array(sorted(tick.index(s) for s in u78))}
PERIODS = {"고르기": ("2026-01-02", "2026-06-30"), "확인": ("2026-07-01", "2026-10-08")}

# 같은 보유 구간 매수 후 보유(묶음 평균): 매수일 시가 → 매도일(xday) 시가, 미완료·손절 다음날 등은 그 날 시가/10월8일 종가
def univ_hold(e, x, inc, cols):
    oe = Ox[e][:, cols]
    ox = np.where(inc[:, None], Cx[LAST][cols][None, :], Ox[x][:, cols])
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = ox / oe - 1
    return np.nanmean(rr, axis=1)


def daytstat(df, col):
    g = df.groupby("e")[col].mean()
    if len(g) < 3:
        return np.nan
    return g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))


def weekt(df, col):
    wk = dates[df.e.values].to_period("W")
    g = df.groupby(wk)[col].mean()
    if len(g) < 3:
        return np.nan
    return g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))


def evaluate(df, n, y, tgt="mid", label="", show=True):
    R = random_same_rule(n, y, tgt)
    res = {}
    for un, cols in UNI.items():
        rday = np.nanmean(R[:, cols], axis=1)
        sub = df[np.isin(df.t, cols)].copy()
        sub["rday"] = rday[sub.e.values]
        sub["hold_u"] = univ_hold(sub.e.values, sub.x.values, sub.inc.values, cols)
        for p, (a, b) in PERIODS.items():
            m = (dates[sub.e.values] >= a) & (dates[sub.e.values] <= b)
            s = sub[m].copy()
            pm = (dates >= a) & (dates <= b)
            rtk = np.nanmean(np.where(pm[:, None], R, np.nan), axis=0)
            s["rtk"] = rtk[s.t.values]
            s["exd"] = s.ret - s.rday
            s["exh"] = s.ret - s.hold_u
            s["ext"] = s.ret - s.rtk
            uncond = np.nanmean(R[pm][:, cols])
            r = dict(n=len(s), avg=s.ret.mean() * 100, win=(s.ret > 0).mean() * 100, worst=s.ret.min() * 100, sum=s.ret.sum() * 100,
                     t=daytstat(s, "ret"), exd=s.exd.mean() * 100, texd=daytstat(s, "exd"), twk=weekt(s, "exd"),
                     exh=s.exh.mean() * 100, texh=daytstat(s, "exh"), ext=s.ext.mean() * 100, unc=uncond * 100,
                     rday=s.rday.mean() * 100, inc=int(s.inc.sum()), hold=(s.x - s.e).mean())
            res[(un, p)] = (r, s)
            if show:
                say(f"  {label}{un:7} {p}: n {r['n']:>5} 평균 {r['avg']:+.2f}% 승률 {r['win']:.0f}% 최악 {r['worst']:+.1f}% 합 {r['sum']:+.0f}% t {r['t']:+.2f}"
                    f" | 같은날무작위 {r['rday']:+.2f}% 초과 {r['exd']:+.2f}%(일 t {r['texd']:+.2f}, 주 t {r['twk']:+.2f})"
                    f" | 같은구간 묶음보유 대비 {r['exh']:+.2f}%(t {r['texh']:+.2f}) | 같은종목무작위 대비 {r['ext']:+.2f}% | 무조건 {r['unc']:+.2f}%"
                    f" | 미완료 {r['inc']} 보유 {r['hold']:.1f}일")
    return res


if __name__ == "__main__":
    say("=" * 100)
    say(f"verify_daily_A — 독립 재구현. 종목 {NT}, 날짜 {dates[0].date()}~{dates[-1].date()}, S&P500 {len(UNI['S&P500'])}, 사용자78 {len(UNI['사용자78'])}")
    say(f"분할 의심일 {len(jump_days)}: {jump_days}")
    say("=" * 100)

    # ---- 1) 최선안 재현
    say("\n[1] 최선안 재현 N10 k1.0 손절 −3% 중간 목표")
    df0 = signals(10, 1.0, 0.03)
    base = evaluate(df0, 10, 0.03, label="")
    say("  연구자 보고: 고르기 SP n 9669 평균 +0.35 초과 +0.17 / U78 n 1468 +1.00 초과 +0.66; 확인 SP n 5948 0.00 초과 +0.11 / U78 n 800 +0.77 초과 +0.63")

    # NKE
    t_nke = tick.index("NKE")
    nk = df0[(df0.t == t_nke) & (dates[df0.e.values] >= "2026-09-01")]
    say("  NKE 9월 이후: " + "; ".join(f"{dates[e]:%m-%d} 체결 {P:.2f} U {u:.2f} F {f:.2f} → {r*100:+.1f}% (매도 {dates[x]:%m-%d})"
                                    for e, P, u, f, r, x in zip(nk.e, nk.P, nk.U, nk.F, nk.ret, nk.x)))

    # ---- 2) 분해: 전날 종가 이탈(B형) vs 정상
    say("\n[2] 분해 — 전날 종가 < 전날 F(B형) / 시가 체결(갭) / 장중 체결")
    F10, _ = box(10)
    for un in UNI:
        for p in PERIODS:
            r, s = base[(un, p)]
            brk = Cx[s.e.values - 1, s.t.values] < F10[s.e.values - 1, s.t.values]
            gap = Ox[s.e.values, s.t.values] <= s.U.values
            parts = []
            for lab, mk in (("B형(전날 이탈)", brk), ("정상·시가", ~brk & gap), ("정상·장중", ~brk & ~gap)):
                q = s[mk]
                parts.append(f"{lab} n {len(q)} 평균 {q.ret.mean()*100:+.2f} 초과 {q.exd.mean()*100:+.2f}")
            say(f"  {un:7} {p}: " + " | ".join(parts))

    # ---- 3) 이웃
    say("\n[3] 이웃 매개변수 — 한 단계씩(박스 N 5/20, 밴드 k 0.5/1.5, 손절 0%/5%, 목표 상단)")
    neigh = [("N5", 5, 1.0, 0.03, "mid"), ("N20", 20, 1.0, 0.03, "mid"), ("k0.5", 10, 0.5, 0.03, "mid"), ("k1.5", 10, 1.5, 0.03, "mid"),
             ("손절0%", 10, 1.0, 0.0, "mid"), ("손절5%", 10, 1.0, 0.05, "mid"), ("목표상단", 10, 1.0, 0.03, "top"), ("N15", 15, 1.0, 0.03, "mid"),
             ("N7", 7, 1.0, 0.03, "mid"), ("k0.75", 10, 0.75, 0.03, "mid"), ("k1.25", 10, 1.25, 0.03, "mid")]
    NB = {}
    for lab, n, k, y, tg in neigh:
        df = signals(n, k, y, tg)
        NB[lab] = evaluate(df, n, y, tg, show=False)
        say(f"  {lab:8}: " + " | ".join(
            f"{p} {un} n {NB[lab][(un,p)][0]['n']} 평균 {NB[lab][(un,p)][0]['avg']:+.2f} 초과 {NB[lab][(un,p)][0]['exd']:+.2f}(t {NB[lab][(un,p)][0]['texd']:+.1f}) 보유대비 {NB[lab][(un,p)][0]['exh']:+.2f}"
            for p in PERIODS for un in UNI))

    # ---- 4) 쏠림(확인 기간)
    say("\n[4] 쏠림 — 확인 기간")
    for un in UNI:
        r, s = base[(un, "확인")]
        s = s.copy()
        s["tk"] = np.array(tick)[s.t.values]
        s["day"] = dates[s.e.values]
        tot = s.exd.sum()
        bt = s.groupby("tk").exd.sum().sort_values(ascending=False)
        bd = s.groupby("day").exd.sum().sort_values(ascending=False)
        say(f"  {un}: 초과 합 {tot*100:+.1f}%p (n {len(s)}), 종목 {s.tk.nunique()}개, 날짜 {s.day.nunique()}일")
        say(f"    상위 종목 초과 합: " + ", ".join(f"{k} {v*100:+.1f}" for k, v in bt.head(6).items()))
        say(f"    하위 종목 초과 합: " + ", ".join(f"{k} {v*100:+.1f}" for k, v in bt.tail(4).items()))
        say(f"    상위 날짜 초과 합: " + ", ".join(f"{k:%m-%d} {v*100:+.1f}(n {int((s.day==k).sum())})" for k, v in bd.head(6).items()))
        for drop_lab, mask in (("상위 1종목 뺌", ~s.tk.isin(bt.index[:1])), ("상위 3종목 뺌", ~s.tk.isin(bt.index[:3])),
                               ("상위 1일 뺌", ~s.day.isin(bd.index[:1])), ("상위 5일 뺌", ~s.day.isin(bd.index[:5])),
                               ("상위 10거래 뺌", ~s.index.isin(s.exd.sort_values(ascending=False).index[:10])),
                               ("NKE 뺌", s.tk != "NKE")):
            q = s[mask]
            say(f"    {drop_lab:12}: n {len(q)} 평균 {q.ret.mean()*100:+.2f}% 초과 {q.exd.mean()*100:+.2f}% (일 t {daytstat(q,'exd'):+.2f}) 보유대비 {q.exh.mean()*100:+.2f}%")
        m = s.groupby(s.day.dt.to_period("M")).agg(n=("ret", "size"), avg=("ret", "mean"), ex=("exd", "mean"), exh=("exh", "mean"))
        say("    월별 n/평균/초과/보유대비: " + "  ".join(f"{str(i)} {int(v.n)}/{v.avg*100:+.2f}/{v.ex*100:+.2f}/{v.exh*100:+.2f}" for i, v in m.iterrows()))
        # 상위 10% 날짜(초과 합 기여 큰 날) 비중
        share = bd.head(max(1, len(bd) // 10)).sum() / tot if tot else np.nan
        say(f"    상위 10% 날짜({max(1, len(bd)//10)}일)의 초과 합 비중: {share*100:.0f}%")

    # ---- 5) 한 사건(7/29 하락 → 7/30 반등) 제거, 고르기 기간 쏠림, 첫 정의(어제 밴드 밖일 때만)
    say("\n[5] 사건·정의 점검")
    ev = pd.to_datetime(["2026-07-29", "2026-07-30"])
    for un in UNI:
        for p in PERIODS:
            r, s = base[(un, p)]
            s = s.copy()
            s["day"] = dates[s.e.values]
            bd = s.groupby("day").exd.sum().sort_values(ascending=False)
            q = s[~s.day.isin(ev)]
            q5 = s[~s.day.isin(bd.index[:5])]
            ev_share = s[s.day.isin(ev)].exd.sum() / s.exd.sum() if s.exd.sum() else np.nan
            say(f"  {un:7} {p}: 7/29·7/30 매수 n {len(s) - len(q)}, 초과 합 비중 {ev_share*100:.0f}% | 그 두 날 뺌 n {len(q)} 평균 {q.ret.mean()*100:+.2f}% 초과 {q.exd.mean()*100:+.2f}%(t {daytstat(q,'exd'):+.2f}) 보유대비 {q.exh.mean()*100:+.2f}%"
                f" | 상위 5일({', '.join(f'{d:%m-%d}' for d in bd.index[:5])}) 뺌 초과 {q5.exd.mean()*100:+.2f}% 보유대비 {q5.exh.mean()*100:+.2f}% 평균 {q5.ret.mean()*100:+.2f}%")
    i_spy = tick.index("SPY")
    for d in ["2026-07-29", "2026-07-30"]:
        i = dates.get_loc(pd.Timestamp(d))
        say(f"  SPY {d}: 전일 종가 대비 {(Cx[i, i_spy]/Cx[i-1, i_spy]-1)*100:+.2f}%")
    # 첫 정의: 어제 밴드 밖(low[1] > U[1])일 때만
    F10, B10 = box(10)
    U10 = F10 + 1.0 * ATRp
    with np.errstate(invalid="ignore"):
        previn = np.zeros((ND, NT), bool)
        previn[1:] = Lx[:-1] <= U10[:-1]
        fill = ~previn & np.isfinite(U10) & (Lx <= U10) & np.isfinite(Ox)
    e, t = np.nonzero(fill)
    P = np.minimum(Ox[e, t], U10[e, t])
    ok = ~BADWIN[e, t] & np.isfinite(B10[e, t]) & (P > 0)
    e, t, P = e[ok], t[ok], P[ok]
    T = np.maximum((F10[e, t] + B10[e, t]) / 2, P * 1.01)
    r, x, inc = run_exit(e.astype(np.int64), t.astype(np.int64), P, T, F10[e, t] * 0.97, False, Ox, Hx, Lx, Cx, LAST, 10)
    dfb = pd.DataFrame(dict(e=e, t=t, P=P, ret=r, x=x, inc=inc, F=F10[e, t], U=U10[e, t], BH=B10[e, t]))
    say("  첫 정의(어제 밴드 밖) 같은 규칙:")
    evaluate(dfb, 10, 0.03, label="첫정의 ")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\n" + "\n".join(LINES) + "\n")
    say("\n저장:", os.path.abspath(OUT))
