"""박스권 매매 연구 — 일봉 A '박스 바닥 버팀 매수'(2026-10-09, 사용자 요청: 이격매수 말고 박스권 지표 하나 더).

A: 직전 N일 최저가 F(오늘 제외)를 박스 바닥, 밴드 상단 U = F + k×ATR14(어제까지 RMA)로 본다.
   밴드 진입일 s: 저가 ≤ U 이고 전날이 '버틴 밴드 날'(저가 ≤ U·종가 ≥ F)이 아님(전날 종가로 바닥을 깼으면 오늘은 새 바닥 시험).
   m=0  : U에 미리 건 지정가 매수(진입일 s에 체결, 체결가 min(시가, U)). 종가 조건 없음.
   m≥1  : s..s+m-1 종가가 모두 F 이상(종가로 바닥을 안 깸) → s+m 시가 매수(또는 U 지정가 변형).
매도(모두 사용자가 실행 가능한 것만: 시가 주문·미리 건 지정가 매도·종가 확인 뒤 다음 시가 손절):
   FH{H}  : 매수일 e 기준 H거래일 뒤 시가 매도(e+H 시가).
   MID/TOP: 박스 중간 (F+BH)/2 / 상단 BH(직전 N일 최고가)에 지정가 매도(목표가 < 매수가×1.01이면 매수가×1.01).
            시가 매수는 매수일부터, 지정가 매수는 다음 날부터 목표 판정. 시가가 목표 위면 시가에 체결.
   P5     : 매수가 +5% 지정가 매도.
   S0/S3  : 종가 < F(×0.97)이면 다음 날 시가 손절. 목표·손절 규칙은 10거래일 뒤 시가에 정리.
기준: 같은 날 무작위(같은 묶음 모든 종목이 그날 시가에 사서 같은 매도 규칙), 같은 종목 무작위 날(같은 기간),
      무조건 보유(기간·묶음 전체 평균 = 같은 보유 길이의 매수 후 보유), 그리고 B(종가로 바닥 이탈)·C(종가로 상단 돌파).
기간: 고르기 2026-01-02~06-30, 확인 2026-07-01~10-08(매수일 기준), 참고 2024-10~2025-12.
분할 의심(종가·시가가 전일 종가 대비 −40% 이하·+80% 이상)일 앞 75일~뒤 12일은 매수·기준에서 뺀다.
마지막 날(10/8)을 넘는 거래는 10/8 종가로 평가하고 '미완료'로 센다.
사용: py daily_A.py   → ../results/daily_A.txt
"""
import glob, os, sys, warnings
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DAILY = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/box/daily"
RES = os.path.join(HERE, "../results")
sys.path.insert(0, os.path.join(HERE, "../../dip_reason/code"))
from fetch import user_universe

NS, KS, MS = [5, 10, 20, 40, 60], [0.0, 0.5, 1.0], [0, 1, 2, 3]
PER = {"고르기": ("2026-01-02", "2026-06-30"), "확인": ("2026-07-01", "2026-10-08"), "참고": ("2024-10-01", "2025-12-31")}
CAP = 10
EXITS = {"FH1": ("fh", 1), "FH3": ("fh", 3), "FH5": ("fh", 5), "FH10": ("fh", 10),
         "MID_S0": ("tg", "mid", 0.0), "MID_S3": ("tg", "mid", 0.03), "TOP_S3": ("tg", "top", 0.03), "P5_S3": ("tg", "p5", 0.03)}
XNAME = {"FH1": "1일 보유", "FH3": "3일 보유", "FH5": "5일 보유", "FH10": "10일 보유", "MID_S0": "중간 목표·바닥 손절",
         "MID_S3": "중간 목표·바닥−3% 손절", "TOP_S3": "상단 목표·바닥−3% 손절", "P5_S3": "+5% 목표·바닥−3% 손절"}

# ---------------------------------------------------------------- 데이터
fs = sorted(glob.glob(os.path.join(DAILY, "1d_*.parquet")))
RAW = {os.path.basename(f)[3:-8]: pd.read_parquet(f) for f in fs}
IDX = RAW["SPY"].index
TK = sorted(RAW)
TPOS = {t: i for i, t in enumerate(TK)}


def panel(col):
    return np.column_stack([RAW[t][col][~RAW[t].index.duplicated()].reindex(IDX).values for t in TK]).astype(float)


O, H, L, C = panel("Open"), panel("High"), panel("Low"), panel("Close")
ND, NT = C.shape
LAST = ND - 1
PC = np.vstack([np.full(NT, np.nan), C[:-1]])
with np.errstate(invalid="ignore", divide="ignore"):
    BAD = (C / PC < 0.6) | (C / PC > 1.8) | (O / PC < 0.6) | (O / PC > 1.8)
cb = np.vstack([np.zeros(NT), np.cumsum(BAD, axis=0)])
lo = np.clip(np.arange(ND) - 75, 0, ND - 1)
hi = np.clip(np.arange(ND) + 12, 0, ND - 1)
TAINT = (cb[hi + 1] - cb[lo]) > 0                                  # [d-75, d+12] 안에 분할 의심일
TR = np.fmax(H - L, np.fmax(np.abs(H - PC), np.abs(L - PC)))
ATR = pd.DataFrame(TR).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().shift(1).values   # 어제까지 ATR14(RMA)
F = {N: pd.DataFrame(L).rolling(N, min_periods=N).min().shift(1).values for N in NS}           # 직전 N일 최저가
BH = {N: pd.DataFrame(H).rolling(N, min_periods=N).max().shift(1).values for N in NS}          # 직전 N일 최고가

SP = set(pd.read_csv(os.path.join(DAILY, "sp500_list.csv")).iloc[:, 0]) & set(TK)
U78 = set(user_universe()["합집합"]) & set(TK)
UNIV = {"S&P500": np.array(sorted(TPOS[t] for t in SP)), "사용자78": np.array(sorted(TPOS[t] for t in U78))}
DAYS = pd.DatetimeIndex(IDX)
PMASK = {p: (DAYS >= a) & (DAYS <= b) for p, (a, b) in PER.items()}


def down(X, m):
    """down(X, m)[e] = X[e-m]"""
    if m == 0:
        return X.copy()
    Y = np.full_like(X, np.nan if X.dtype.kind == "f" else False)
    Y[m:] = X[:-m]
    return Y


def up(X, j):
    """up(X, j)[s] = X[s+j]"""
    if j == 0:
        return X.copy()
    Y = np.full_like(X, np.nan)
    Y[:-j] = X[j:]
    return Y


# ---------------------------------------------------------------- 매도 시뮬레이션(거래 배열 단위)
def simulate(e, t, P, lev_f, lev_bh, start_off, xkey):
    """e 매수일 인덱스, t 종목 인덱스, P 매수가, lev_f 손절 기준(바닥), lev_bh 박스 상단, start_off 0=시가 매수 1=지정가 매수.
    반환 (수익률, 보유일, 미완료)."""
    spec = EXITS[xkey]
    n = len(e)
    ret = np.full(n, np.nan)
    hold = np.zeros(n, int)
    inc = np.zeros(n, bool)
    if spec[0] == "fh":
        j = e + spec[1]
        ok = j <= LAST
        ret[ok] = O[j[ok], t[ok]] / P[ok] - 1
        ret[~ok] = C[LAST, t[~ok]] / P[~ok] - 1
        inc[~ok] = True
        hold[:] = spec[1]
        return ret, hold, inc
    _, tg, y = spec
    if tg == "mid":
        tgt = (lev_f + lev_bh) / 2
    elif tg == "top":
        tgt = lev_bh.copy()
    else:
        tgt = P * 1.05
    tgt = np.fmax(tgt, P * 1.01)
    stop = lev_f * (1 - y)
    done = np.zeros(n, bool)
    for h in range(CAP + 1):
        j = e + h
        live = ~done
        past = live & (j > LAST)                                   # 데이터 끝: 마지막 종가로 평가
        ret[past] = C[LAST, t[past]] / P[past] - 1
        inc[past] = True
        hold[past] = h
        done |= past
        live = ~done
        if h == CAP:
            ret[live] = O[j[live], t[live]] / P[live] - 1
            hold[live] = h
            done |= live
            break
        jj = np.minimum(j, LAST)
        o, hh, cc = O[jj, t], H[jj, t], C[jj, t]
        if h >= start_off:
            g = live & (h > 0) & (o >= tgt)                      # 시가가 목표 위로 갭
            ret[g] = o[g] / P[g] - 1
            hold[g] = h
            done |= g
            live = ~done
            hit = live & (hh >= tgt)
            ret[hit] = tgt[hit] / P[hit] - 1
            hold[hit] = h
            done |= hit
            live = ~done
        st = live & (cc < stop)                                     # 종가 손절 → 다음 날 시가
        if st.any():
            j1 = j + 1
            ok = st & (j1 <= LAST)
            ret[ok] = O[j1[ok], t[ok]] / P[ok] - 1
            hold[ok] = h + 1
            nok = st & (j1 > LAST)
            ret[nok] = C[LAST, t[nok]] / P[nok] - 1
            inc[nok] = True
            done |= st
    return ret, hold, inc


# ---------------------------------------------------------------- 무작위 기준(모든 종목·날 시가 매수, 같은 매도 규칙)
RAND = {}


def rand_mat(N, xkey):
    key = ("fh", xkey) if EXITS[xkey][0] == "fh" else (N, xkey)
    if key not in RAND:
        ee, tt = np.meshgrid(np.arange(ND), np.arange(NT), indexing="ij")
        ee, tt = ee.ravel(), tt.ravel()
        P = O[ee, tt]
        ok = np.isfinite(P) & np.isfinite(F[N][ee, tt]) & np.isfinite(BH[N][ee, tt]) & ~TAINT[ee, tt] & (P > 0)
        R = np.full(ND * NT, np.nan)
        r, _, _ = simulate(ee[ok], tt[ok], P[ok], F[N][ee[ok], tt[ok]], BH[N][ee[ok], tt[ok]], 0, xkey)
        R[ok] = r
        RAND[key] = R.reshape(ND, NT)
    return RAND[key]


# ---------------------------------------------------------------- 신호
def fresh_band(N, k, mode="hold"):
    """오늘 밴드 진입(저가 ≤ U). 중복 제거: mode="hold"(본안)는 어제가 '버틴 밴드 날'(저가 ≤ U, 종가 ≥ F)이 아닐 때만 새 시험으로 본다
    — 어제 종가로 바닥을 깼으면 오늘은 어제 저가가 들어간 새 바닥의 시험이다(나이키 10/2 → 10/5). mode="band"(첫 정의)는 어제 밴드 밖일 때만."""
    U = F[N] + k * ATR
    with np.errstate(invalid="ignore"):
        inb = (L <= U) & np.isfinite(U)
        held = inb & (C >= F[N])
    y = held if mode == "hold" else inb
    prev = np.vstack([np.zeros((1, NT), bool), y[:-1]])
    return inb & ~prev, U


def trades_A(N, k, m, entry="open", mode="hold"):
    fr, U = fresh_band(N, k, mode)
    if m == 0:
        E = fr
        e, t = np.nonzero(E)
        P = np.fmin(O[e, t], U[e, t])
        return dict(e=e, t=t, s=e, P=P, f=F[N][e, t], bh=BH[N][e, t], off=1)
    ok = fr.copy()
    for j in range(m):
        with np.errstate(invalid="ignore"):
            ok &= up(C, j) >= F[N]
    E = down(ok.astype(float), m) > 0.5
    e, t = np.nonzero(E)
    s = e - m
    f, bh, u = F[N][s, t], BH[N][s, t], U[s, t]
    if entry == "open":
        return dict(e=e, t=t, s=s, P=O[e, t], f=f, bh=bh, off=0)
    fill = L[e, t] <= u                                            # 다음 날 U 지정가
    return dict(e=e[fill], t=t[fill], s=s[fill], P=np.fmin(O[e, t], u)[fill], f=f[fill], bh=bh[fill], off=1)


def trades_B(N):
    """B: 종가가 직전 N일 최저가 아래로 새로 이탈(전날 종가는 그날 바닥 이상) → 다음 날 시가. 손절 기준은 이탈일 저가."""
    with np.errstate(invalid="ignore"):
        br = C < F[N]
    prev = np.vstack([np.zeros((1, NT), bool), br[:-1]])
    s, t = np.nonzero(br & ~prev)
    e = s + 1
    k = e <= LAST
    s, t, e = s[k], t[k], e[k]
    return dict(e=e, t=t, s=s, P=O[e, t], f=L[s, t], bh=BH[N][s, t], off=0)


def trades_C(N):
    """C: 종가가 직전 N일 최고가 위로 새로 돌파 → 다음 날 시가. 손절 기준은 돌파 수준(BH), 중간·상단 목표는 매수가+1%로 대체됨."""
    with np.errstate(invalid="ignore"):
        bo = C > BH[N]
    prev = np.vstack([np.zeros((1, NT), bool), bo[:-1]])
    s, t = np.nonzero(bo & ~prev)
    e = s + 1
    k = e <= LAST
    s, t, e = s[k], t[k], e[k]
    return dict(e=e, t=t, s=s, P=O[e, t], f=BH[N][s, t], bh=BH[N][s, t], off=0)


# ---------------------------------------------------------------- 평가
def evaluate(tr, N, xkey):
    e, t = tr["e"], tr["t"]
    ok = np.isfinite(tr["P"]) & (tr["P"] > 0) & ~TAINT[e, t] & np.isfinite(tr["f"]) & np.isfinite(tr["bh"])
    tr = {k: (v[ok] if isinstance(v, np.ndarray) else v) for k, v in tr.items()}
    r, hold, inc = simulate(tr["e"], tr["t"], tr["P"], tr["f"], tr["bh"], tr["off"], xkey)
    R = rand_mat(N, xkey)
    df = pd.DataFrame(dict(day=DAYS[tr["e"]], sday=DAYS[tr["s"]], e=tr["e"], t=tr["t"], tk=np.array(TK)[tr["t"]], ret=r, hold=hold, inc=inc,
                           P=tr["P"], f=tr["f"], bh=tr["bh"]))
    out = {}
    for un, cols in UNIV.items():
        Ru = R[:, cols]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rday = np.nanmean(Ru, axis=1)
        d = df[np.isin(df.t, cols)].dropna(subset=["ret"]).copy()
        d["rday"] = rday[d.e]
        for p, pm in PMASK.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rtk = np.nanmean(np.where(pm[:, None], R, np.nan), axis=0)
                uncond = np.nanmean(Ru[pm])
            x = d[pm[d.e]].copy()
            x["rtk"] = rtk[x.t]
            out[(un, p)] = (x, uncond)
    return out


def tclust(x, col):
    g = x.groupby("day")[col].mean()
    if len(g) < 3 or g.std(ddof=1) == 0:
        return np.nan
    return g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))


def stats(x, uncond):
    if len(x) == 0:
        return dict(n=0)
    exd = x.ret - x.rday
    ext = x.ret - x.rtk
    xx = x.assign(exd=exd)
    return dict(n=len(x), avg=x.ret.mean() * 100, win=(x.ret > 0).mean() * 100, worst=x.ret.min() * 100, sum=x.ret.sum() * 100,
                t=tclust(x, "ret"), exd=exd.mean() * 100, texd=tclust(xx, "exd"), ext=ext.mean() * 100, unc=uncond * 100,
                rday=x.rday.mean() * 100, inc=int(x.inc.sum()), hold=x.hold.mean(), ndays=x.day.nunique())


def fmt(s):
    if s.get("n", 0) == 0:
        return "n=0"
    return (f"n={s['n']:>5} 평균 {s['avg']:+6.2f}% 승률 {s['win']:4.0f}% 최악 {s['worst']:+6.1f}% 합 {s['sum']:+8.0f}% t {s['t']:+5.2f} | "
            f"같은날무작위 {s['rday']:+5.2f}% 초과 {s['exd']:+5.2f}%(t {s['texd']:+5.2f}) | 같은종목무작위 대비 {s['ext']:+5.2f}% | 무조건보유 {s['unc']:+5.2f}%"
            + (f" | 미완료 {s['inc']}" if s["inc"] else ""))


LOG = []


def P_(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


ROWS = []        # 모든 시험 규칙 결과


def run_rule(name, fam, N, tr, xkey, meta):
    out = evaluate(tr, N, xkey)
    row = dict(rule=name, fam=fam, N=N, exit=xkey, **meta)
    for (un, p), (x, unc) in out.items():
        s = stats(x, unc)
        for kk, v in s.items():
            row[f"{un}|{p}|{kk}"] = v
    ROWS.append(row)
    return out, row


def score(row, p="고르기"):
    a, b = row.get(f"S&P500|{p}|exd", np.nan), row.get(f"사용자78|{p}|exd", np.nan)
    na, nb = row.get(f"S&P500|{p}|n", 0), row.get(f"사용자78|{p}|n", 0)
    if not (na >= 200 and nb >= 30) or not (a > 0 and b > 0):
        return -np.inf
    return (a + b) / 2


def aname(N, k, m, entry="open"):
    ent = "U지정가" if m == 0 else ("다음시가" if entry == "open" else "다음날U지정가")
    return f"A_N{N}_k{k:g}_m{m}_{ent}"


if __name__ == "__main__":
    P_("=" * 110)
    P_("일봉 A '박스 바닥 버팀 매수' 연구 (2026-10-09)  데이터: yfinance 일봉 2024-10-09~2026-10-08, 종목", NT,
       "| S&P500", len(UNIV["S&P500"]), "| 사용자78 중 데이터", len(UNIV["사용자78"]))
    P_("분할 의심일", int(BAD.sum()), "개(앞 75일~뒤 12일 제외):", ", ".join(f"{TK[t]} {DAYS[d]:%Y-%m-%d}" for d, t in zip(*np.nonzero(BAD))))
    P_("=" * 110)

    # ---- 1단계: 신호 60개(5 N × 3 k × 4 m) × 5일 보유
    P_("\n[1단계] 신호 격자 60개, 매도 5일 보유(FH5) — 초과 = 같은 날 무작위(같은 묶음 전 종목 그날 시가 매수·5일 보유) 대비")
    P_(f"{'규칙':28} | {'고르기 SP n':>10} {'평균':>6} {'초과':>6} {'t':>5} | {'U78 n':>5} {'평균':>6} {'초과':>6} | {'확인 SP n':>9} {'평균':>6} {'초과':>6} {'t':>5} | {'U78 n':>5} {'평균':>6} {'초과':>6}")
    TRS = {}
    for N in NS:
        for k in KS:
            for m in MS:
                nm = aname(N, k, m)
                tr = trades_A(N, k, m)
                TRS[nm] = (N, k, m, "open", tr)
                _, r = run_rule(nm, "A", N, tr, "FH5", dict(k=k, m=m, entry="open", stage=1))
                g = lambda u, p, c: r.get(f"{u}|{p}|{c}", np.nan)
                P_(f"{nm:28} | {g('S&P500','고르기','n'):>10.0f} {g('S&P500','고르기','avg'):+6.2f} {g('S&P500','고르기','exd'):+6.2f} {g('S&P500','고르기','texd'):+5.1f} | "
                   f"{g('사용자78','고르기','n'):>5.0f} {g('사용자78','고르기','avg'):+6.2f} {g('사용자78','고르기','exd'):+6.2f} | "
                   f"{g('S&P500','확인','n'):>9.0f} {g('S&P500','확인','avg'):+6.2f} {g('S&P500','확인','exd'):+6.2f} {g('S&P500','확인','texd'):+5.1f} | "
                   f"{g('사용자78','확인','n'):>5.0f} {g('사용자78','확인','avg'):+6.2f} {g('사용자78','확인','exd'):+6.2f}")

    st1 = sorted([r for r in ROWS if r.get("stage") == 1], key=score, reverse=True)
    top3 = [r["rule"] for r in st1[:3] if score(r) > -np.inf]
    P_("\n고르기 점수 = (S&P500 초과 + 사용자78 초과)/2, 조건 S&P500 n≥200·사용자78 n≥30·두 초과 모두 플러스. 상위 3:", top3)

    # ---- 2단계: 상위 3 신호 × 나머지 매도 7개
    P_("\n[2단계] 상위 3 신호 × 매도 7개(1·3·10일 보유, 중간 목표·바닥/−3% 손절, 상단 목표·−3% 손절, +5% 목표·−3% 손절)")
    for nm in top3:
        N, k, m, ent, tr = TRS[nm]
        for xk in ["FH1", "FH3", "FH10", "MID_S0", "MID_S3", "TOP_S3", "P5_S3"]:
            run_rule(nm, "A", N, tr, xk, dict(k=k, m=m, entry=ent, stage=2))
    a_rows = [r for r in ROWS if r["fam"] == "A"]
    best = max(a_rows, key=score)
    bN, bk, bm = best["N"], best["k"], best["m"]
    # ---- 2b: 최선 신호의 '다음 날 U 지정가' 변형
    if bm >= 1:
        trU = trades_A(bN, bk, bm, entry="limit")
        TRS[aname(bN, bk, bm, "limit")] = (bN, bk, bm, "limit", trU)
        for xk in sorted({best["exit"], "FH5"}):
            run_rule(aname(bN, bk, bm, "limit"), "A", bN, trU, xk, dict(k=bk, m=bm, entry="limit", stage=3))
    a_rows = [r for r in ROWS if r["fam"] == "A"]
    best = max(a_rows, key=score)
    for r in sorted([r for r in a_rows if r["stage"] >= 2], key=lambda r: (r["rule"], r["exit"])):
        P_(f"  {r['rule']:28} {XNAME[r['exit']]:22} 고르기 SP {r['S&P500|고르기|avg']:+6.2f}%/초과 {r['S&P500|고르기|exd']:+5.2f} U78 {r['사용자78|고르기|avg']:+6.2f}%/초과 {r['사용자78|고르기|exd']:+5.2f}"
           f" | 확인 SP {r['S&P500|확인|avg']:+6.2f}%/초과 {r['S&P500|확인|exd']:+5.2f} U78 {r['사용자78|확인|avg']:+6.2f}%/초과 {r['사용자78|확인|exd']:+5.2f}")

    P_("\n" + "=" * 110)
    P_(f"[최선안(고르기 기간 점수로 선택)] {best['rule']} + {XNAME[best['exit']]}  (고르기 점수 {score(best):+.2f}%)")
    bnm, bxk = best["rule"], best["exit"]
    N_, k_, m_, ent_, tr_ = TRS[bnm]
    out_best = evaluate(tr_, N_, bxk)
    for un in UNIV:
        for p in ["고르기", "확인", "참고"]:
            x, unc = out_best[(un, p)]
            P_(f"  {un:7} {p:4}: {fmt(stats(x, unc))}")

    # ---- 민감도: 중복 제거를 첫 정의(어제 밴드 밖일 때만)로 바꾼 같은 규칙
    trb = trades_A(bN, bk, bm, entry=ent_, mode="band")
    _, rb = run_rule(bnm + "_첫정의", "A민감도", bN, trb, bxk, dict(k=bk, m=bm, entry=ent_, stage=8))
    P_("  민감도(중복 제거 첫 정의 '어제 밴드 밖'): " + " | ".join(
        f"{un} {p} n {rb[f'{un}|{p}|n']} 평균 {rb[f'{un}|{p}|avg']:+.2f}% 초과 {rb[f'{un}|{p}|exd']:+.2f}%" for un in UNIV for p in ["고르기", "확인"]))

    if bm == 0:                                                     # 저가가 U를 0.2% 넘게 뚫어야 체결(닿기만 한 날 제외)
        Ub = tr_["f"] + bk * ATR[tr_["e"], tr_["t"]]
        keep = (O[tr_["e"], tr_["t"]] <= Ub) | (L[tr_["e"], tr_["t"]] <= Ub * 0.998)
        trm = {kk: (v[keep] if isinstance(v, np.ndarray) else v) for kk, v in tr_.items()}
        _, rm = run_rule(bnm + "_체결보수", "A민감도", bN, trm, bxk, dict(k=bk, m=bm, entry=ent_, stage=8))
        P_("  민감도(지정가 체결 보수: 시가 체결이거나 저가가 U보다 0.2% 아래): " + " | ".join(
            f"{un} {p} n {rm[f'{un}|{p}|n']} 평균 {rm[f'{un}|{p}|avg']:+.2f}% 초과 {rm[f'{un}|{p}|exd']:+.2f}%" for un in UNIV for p in ["고르기", "확인"]))
    # 한 종목 한 포지션(들고 있는 동안 같은 종목 새 매수 안 함 — 하락 추세에서 날마다 다시 사는 것 방지)
    r_all, h_all, _ = simulate(tr_["e"], tr_["t"], tr_["P"], tr_["f"], tr_["bh"], tr_["off"], bxk)
    keep = np.zeros(len(tr_["e"]), bool)
    last_t, free = -1, -1
    for i in np.lexsort((tr_["e"], tr_["t"])):
        if tr_["t"][i] != last_t:
            last_t, free = tr_["t"][i], -1
        if tr_["e"][i] >= free and np.isfinite(r_all[i]):
            keep[i] = True
            free = tr_["e"][i] + h_all[i]
    tr1 = {kk: (v[keep] if isinstance(v, np.ndarray) else v) for kk, v in tr_.items()}
    _, r1 = run_rule(bnm + "_1포지션", "A민감도", bN, tr1, bxk, dict(k=bk, m=bm, entry=ent_, stage=8))
    P_("  민감도(한 종목 한 포지션): " + " | ".join(
        f"{un} {p} n {r1[f'{un}|{p}|n']} 평균 {r1[f'{un}|{p}|avg']:+.2f}% 초과 {r1[f'{un}|{p}|exd']:+.2f}%(t {r1[f'{un}|{p}|texd']:+.1f})" for un in UNIV for p in ["고르기", "확인"]))
    # 어디서 나오나: 체결 방식 × 전날 바닥 이탈 여부
    P_("  분해(최선안): 체결=시가(시가가 이미 U 아래, 갭) vs 장중 U, 전날=종가로 바닥 이탈(B 다음 날과 같은 자리) vs 정상 시험")
    for un in UNIV:
        for p in ["고르기", "확인"]:
            x = out_best[(un, p)][0].copy()
            Ux = x.f.values + bk * ATR[x.e.values, x.t.values]
            x["gap"] = O[x.e.values, x.t.values] <= Ux
            x["brk"] = C[x.e.values - 1, x.t.values] < F[bN][x.e.values - 1, x.t.values]
            x["ex"] = x.ret - x.rday
            parts = []
            for lab, mk in (("시가·이탈다음", x.gap & x.brk), ("시가·정상", x.gap & ~x.brk), ("장중·이탈다음", ~x.gap & x.brk), ("장중·정상", ~x.gap & ~x.brk)):
                y = x[mk]
                parts.append(f"{lab} n {len(y)} 평균 {y.ret.mean()*100:+.2f}% 초과 {y.ex.mean()*100:+.2f}%" if len(y) else f"{lab} n 0")
            P_(f"    {un:7} {p:4}: " + " | ".join(parts) + f" | 평균 보유 {x.hold.mean():.1f}일")

    # ---- 같은 매도 규칙의 A 다른 매도들(참고: 최선 신호의 매도별)
    P_("\n[최선 신호의 매도 규칙별(같은 신호)]")
    for r in [r for r in a_rows if r["rule"] == bnm]:
        P_(f"  {XNAME[r['exit']]:22} 고르기 SP {r['S&P500|고르기|avg']:+6.2f}%(초과 {r['S&P500|고르기|exd']:+5.2f}) U78 {r['사용자78|고르기|avg']:+6.2f}%(초과 {r['사용자78|고르기|exd']:+5.2f})"
           f" | 확인 SP {r['S&P500|확인|avg']:+6.2f}%(초과 {r['S&P500|확인|exd']:+5.2f}, 승률 {r['S&P500|확인|win']:.0f}%, 최악 {r['S&P500|확인|worst']:+.1f}%)"
           f" U78 {r['사용자78|확인|avg']:+6.2f}%(초과 {r['사용자78|확인|exd']:+5.2f}, n {r['사용자78|확인|n']})")

    # ---- 비교: B·C (같은 매도 규칙)
    P_("\n[A·B·C 비교] B = 종가로 직전 N일 최저가 새로 이탈 → 다음 시가, C = 종가로 직전 N일 최고가 새로 돌파 → 다음 시가")
    P_("  (1) 모든 N, 5일 보유")
    for N in NS:
        for fam, fn in (("B", trades_B), ("C", trades_C)):
            run_rule(f"{fam}_N{N}", fam, N, fn(N), "FH5", dict(stage=9))
    for N in NS:
        line = f"  N={N:<3}"
        for fam in ["A", "B", "C"]:
            if fam == "A":
                r = next(r for r in ROWS if r["rule"] == aname(N, bk, bm) and r["exit"] == "FH5")
                lab = f"A(k{bk:g},m{bm})"
            else:
                r = next(r for r in ROWS if r["rule"] == f"{fam}_N{N}" and r["exit"] == "FH5")
                lab = fam
            line += (f" | {lab} 고르기 SP {r['S&P500|고르기|avg']:+5.2f}(초과 {r['S&P500|고르기|exd']:+5.2f}) U78 {r['사용자78|고르기|exd']:+5.2f}"
                     f" 확인 SP {r['S&P500|확인|avg']:+5.2f}(초과 {r['S&P500|확인|exd']:+5.2f}) U78 {r['사용자78|확인|exd']:+5.2f}")
        P_(line)
    P_(f"  (2) N={bN}, 최선 매도와 10일 보유")
    for xk in sorted({bxk, "FH10"}):
        for fam, fn in (("B", trades_B), ("C", trades_C)):
            if not any(r["rule"] == f"{fam}_N{bN}" and r["exit"] == xk for r in ROWS):
                run_rule(f"{fam}_N{bN}", fam, bN, fn(bN), xk, dict(stage=9))
        for fam in ["A", "B", "C"]:
            nm = bnm if fam == "A" else f"{fam}_N{bN}"
            rr = [r for r in ROWS if r["rule"] == nm and r["exit"] == xk]
            if not rr:
                tr = TRS[nm][4]
                run_rule(nm, "A", bN, tr, xk, dict(k=bk, m=bm, entry=ent_, stage=4))
                rr = [ROWS[-1]]
            r = rr[0]
            P_(f"  {XNAME[xk]:22} {fam}: 고르기 SP n {r['S&P500|고르기|n']:>5} {r['S&P500|고르기|avg']:+6.2f}%(초과 {r['S&P500|고르기|exd']:+5.2f}, t {r['S&P500|고르기|texd']:+4.1f}) "
               f"U78 n {r['사용자78|고르기|n']:>4} {r['사용자78|고르기|avg']:+6.2f}%(초과 {r['사용자78|고르기|exd']:+5.2f}) | 확인 SP n {r['S&P500|확인|n']:>5} {r['S&P500|확인|avg']:+6.2f}%"
               f"(초과 {r['S&P500|확인|exd']:+5.2f}, t {r['S&P500|확인|texd']:+4.1f}) U78 n {r['사용자78|확인|n']:>4} {r['사용자78|확인|avg']:+6.2f}%(초과 {r['사용자78|확인|exd']:+5.2f})")

    # ---- 집계
    nS = sum(r["fam"] == "A민감도" for r in ROWS)
    a_rows = [r for r in ROWS if r["fam"] == "A"]
    nA = len(a_rows)
    nAll = len(ROWS)
    both = {un: sum((r.get(f"{un}|고르기|exd", -1) > 0) and (r.get(f"{un}|확인|exd", -1) > 0) for r in a_rows) for un in UNIV}
    both2 = sum(all((r.get(f"{un}|고르기|exd", -1) > 0) and (r.get(f"{un}|확인|exd", -1) > 0) for un in UNIV) for r in a_rows)
    P_(f"\n[시험 수] 전체 {nAll}개(A {nA}개 + 최선안 민감도 {nS}개 + 비교용 B·C {nAll - nA - nS}개). A {nA}개 중 두 기간 모두 같은 날 무작위보다 나은 규칙: "
       f"S&P500 {both['S&P500']}개, 사용자78 {both['사용자78']}개, 두 묶음 모두 {both2}개")
    a1 = [r for r in a_rows if r["stage"] == 1]
    P_("  1단계 60개 중 확인 기간 초과>0: S&P500", sum(r["S&P500|확인|exd"] > 0 for r in a1), "사용자78", sum(r["사용자78|확인|exd"] > 0 for r in a1),
       "| 고르기 초과>0: S&P500", sum(r["S&P500|고르기|exd"] > 0 for r in a1), "사용자78", sum(r["사용자78|고르기|exd"] > 0 for r in a1))
    # 고르기 점수 순위 대비 확인 순위 상관
    sc = pd.DataFrame([dict(sel=(r["S&P500|고르기|exd"] + r["사용자78|고르기|exd"]) / 2, conf=(r["S&P500|확인|exd"] + r["사용자78|확인|exd"]) / 2) for r in a1])
    P_(f"  1단계 60개: 고르기 초과 vs 확인 초과 순위상관 {sc.sel.rank().corr(sc.conf.rank()):+.2f}")

    P_("  (설명용, 선택에 쓰지 않음) 1단계 60개 중 확인 기간 두 묶음 평균 초과 상위 5: " + ", ".join(
        f"{r['rule']}({(r['S&P500|확인|exd'] + r['사용자78|확인|exd']) / 2:+.2f}%, 고르기 {(r['S&P500|고르기|exd'] + r['사용자78|고르기|exd']) / 2:+.2f}%)"
        for r in sorted(a1, key=lambda r: -(r['S&P500|확인|exd'] + r['사용자78|확인|exd']))[:5]))
    for kk in KS:
        for mm in MS:
            g = [r for r in a1 if r["k"] == kk and r["m"] == mm]
            P_(f"  k{kk:g} m{mm}: N 5개 평균 초과 — 고르기 SP {np.mean([r['S&P500|고르기|exd'] for r in g]):+.2f} U78 {np.mean([r['사용자78|고르기|exd'] for r in g]):+.2f}"
               f" | 확인 SP {np.mean([r['S&P500|확인|exd'] for r in g]):+.2f} U78 {np.mean([r['사용자78|확인|exd'] for r in g]):+.2f}")

    # ---- 월별(최근 장세)
    P_(f"\n[월별 — 최선안 {bnm} + {XNAME[bxk]}] 매수월, S&P500 / 사용자78: n 평균 초과(같은 날 무작위 대비)")
    tr_all = out_best
    for un in UNIV:
        x = pd.concat([tr_all[(un, p)][0] for p in ["참고", "고르기", "확인"]])
        x = x.assign(ex=x.ret - x.rday)
        g = x.groupby(x.day.dt.to_period("M")).agg(n=("ret", "size"), avg=("ret", "mean"), ex=("ex", "mean"))
        P_(f"  {un}: " + "  ".join(f"{str(i)[2:]} {int(r.n)}/{r.avg*100:+.1f}/{r.ex*100:+.1f}" for i, r in g.iterrows()))

    # ---- 실적 직후 신호 분리
    ec = os.path.join(RES, "earnings_cache.csv")
    if os.path.exists(ec):
        E = pd.read_csv(ec)
        E["dt"] = pd.to_datetime(E.earnings_dt, utc=True).dt.tz_convert("America/New_York")
        E["d"] = E.dt.dt.tz_localize(None).dt.normalize()
        E["after"] = (E.dt.dt.hour >= 16)
        pos = np.searchsorted(DAYS.values, E.d.values)                          # 그날(또는 다음 거래일)
        pos = np.where(E.after.values & (pos < ND) & (DAYS.values[np.minimum(pos, ND - 1)] == E.d.values), pos + 1, pos)
        E["ri"] = pos
        rx = {t: np.sort(g.ri.values[g.ri.values < ND]) for t, g in E.groupby("ticker")}
        cov = set(rx)
        P_(f"\n[실적 반응일 기준 분리 — 캐시 {len(cov)}종목, 최선안] 신호 바닥일 s가 실적 반응일 0~5거래일 뒤인가")
        for un in UNIV:
            for p in ["고르기", "확인"]:
                x = tr_all[(un, p)][0]
                x = x[x.tk.isin(cov)].copy()
                si = np.searchsorted(DAYS.values, x.sday.values)
                lag = []
                for tk, s in zip(x.tk, si):
                    a = rx[tk]
                    q = a[a <= s]
                    lag.append(s - q[-1] if len(q) else 999)
                x["lag"] = lag
                x["ex"] = x.ret - x.rday
                for lab, mk in (("실적 0~5일 뒤", x.lag <= 5), ("그 외", x.lag > 5)):
                    y = x[mk]
                    if len(y):
                        P_(f"  {un:7} {p:4} {lab:10}: n {len(y):>5} 평균 {y.ret.mean()*100:+6.2f}% 승률 {(y.ret>0).mean()*100:4.0f}% 초과 {y.ex.mean()*100:+5.2f}%")
    else:
        P_("\n[실적 분리] earnings_cache.csv 없음 — 생략")

    # ---- 나이키 점검
    P_("\n[나이키(NKE) 점검] 10/1 장 마감 뒤 실적 → 10/2 갭하락 저가 31.97 → 10/8 종가 34.74")
    t = TPOS["NKE"]
    for d in DAYS[(DAYS >= "2026-09-25")]:
        i = DAYS.get_loc(d)
        P_(f"  {d:%m-%d} 시 {O[i,t]:.2f} 고 {H[i,t]:.2f} 저 {L[i,t]:.2f} 종 {C[i,t]:.2f} | " + " ".join(
            f"F{N} {F[N][i,t]:.2f}" for N in (5, 20, 60)) + f" ATR {ATR[i,t]:.2f}")
    P_("  A 규칙(1단계 60개+변형) 중 9/25 이후 NKE 매수 — (밴드진입일→매수일 체결가, 10/8 종가 평가): 해당 규칙")
    hits = {}
    for nm, (N, k, m, ent, tr) in TRS.items():
        mk = (tr["t"] == t) & (DAYS[tr["e"]] >= "2026-09-25")
        for e, P, s in zip(tr["e"][mk], tr["P"][mk], tr["s"][mk]):
            hits.setdefault(f"{DAYS[s]:%m-%d}→{DAYS[e]:%m-%d} {P:.2f}({(C[LAST, t] / P - 1) * 100:+.1f}%)", []).append(nm.replace("A_", ""))
    for kk, v in sorted(hits.items()):
        P_(f"    {kk}: {len(v)}개 — {', '.join(v[:8])}{' …' if len(v) > 8 else ''}")
    mk = tr_["t"] == t
    sub = {kk: (v[mk] if isinstance(v, np.ndarray) else v) for kk, v in tr_.items()}
    r_, h_, i_ = simulate(sub["e"], sub["t"], sub["P"], sub["f"], sub["bh"], sub["off"], bxk)
    rows = [f"{DAYS[e]:%m-%d} 체결 {P:.2f} → {XNAME[bxk]} {r*100:+.1f}%({h}일{', 미완료' if i else ''})"
            for e, P, r, h, i in zip(sub["e"], sub["P"], r_, h_, i_) if DAYS[e] >= pd.Timestamp("2026-09-01")]
    P_(f"  최선안({bnm}) NKE 9월 이후 거래: " + ("; ".join(rows) or "없음"))

    # ---- 저장
    pd.DataFrame(ROWS).to_csv(os.path.join(RES, "daily_A_rules.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(RES, "daily_A.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\n" + "\n".join(LOG) + "\n")
    P_("\n저장:", os.path.abspath(os.path.join(RES, "daily_A.txt")))
