"""박스권 위 돌파 매수(C) 분봉 연구 — 키 intra_C (2026-10-09 사용자 요청: 박스권 매매 A 버팀 / B 이탈이 바닥 / C 위 돌파 비교).

신호(봉 k 종가로 판단, 봉 k+1 시가에 시장가 매수 — 미래 참조 없음):
  박스 상단 hi[k] = max(High[k-n..k-1]), 하단 lo[k] = min(Low[k-n..k-1]), n = 박스 분 / 봉 분(프리·애프터 봉 포함, 연속 봉).
  C 돌파: Close[k] > hi[k] 이고 Close[k-1] <= hi[k-1]   (Pine: ta.crossover(close, ta.highest(high, n)[1]))
  수축: (hi-lo)/lo <= QN[N] x 일봉 ATR14%(전날 종가까지)        거래량: Volume[k] >= 2 x 같은 5분 칸 지난 10거래일 평균(정규장 봉만)
매도(같은 세트, 매수 봉 다음 봉부터):
  X1 RSI(3분 14·5분 9) > 70 이면 다음 봉 시가, 아니면 다음 한국 8시   X2 1시간 뒤 시가   X3 다음 한국 8시(미 동부 19:00)
  X4 +2% 지정가(닿으면 max(시가, 지정가))   X5 추적: 종가 <= max(매수가, 매수 뒤 최고 종가) x 0.985 → 다음 봉 시가
  X6 박스 높이 목표(매수가 + 상단-하단 지정가) + 종가 < 박스 중간이면 다음 봉 시가
  모든 규칙은 다음 한국 8시에 남은 물량을 판다(이격매수와 같은 정리 시각).
사용자 주문 조건(desk_system §1): 한국 08~24시 = 미 동부 19:00~다음 날 11:00(서머타임). 데이터가 있는 깨어 있는 봉 = 04:00~11:00, 19:00~20:00.
  모드 i(실행 가능): 시장가 매수·매도(신호·시간·손절)는 깨어 있는 봉에서만, 자는 동안 나온 매도 신호는 깨어난 첫 봉 시가. 지정가 매도는 늘 걸려 있다.
  모드 ii(신호 품질 참고): 수면 제약 없음.
범위: 정규 = 매수 봉 09:30~11:00(신호 봉도 정규장) 모드 i / 깨어 = 깨어 있는 봉 전부(프리·애프터 포함, 거래량 0·얇은 가격) 모드 i /
      정규종일 = 매수 봉 09:30~16:00 모드 ii(참고).
비교 기준(같은 시각 이후만 비교 — 미래 참조 없음): RA = 같은 날 같은 시각 봉에 다른 종목 전부(주식 137 / 3배ETF 11 따로)를
          같은 매도로 샀을 때 평균, RB = 같은 종목을 기간 안 다른 날 같은 시각 봉에 같은 매도로 샀을 때 평균(매수 후 보유 대용).
          X6은 신호의 목표·손절 %를 그대로 쓰고 RA·RB 후보에서 10개씩 표본. 이격매수 = stop_lab.base_trades.
          A·B 간이판(같은 박스·같은 매도 X1~X5): A 저가가 하단 +0.3% 안에 처음 닿고 종가는 하단 위 / B 종가가 하단을 처음 이탈.
기간: 고르기 early5 = 7/14~9/8 5분봉, 확인 late3 = 9/10~10/6 3분봉(3배 ETF 10/8까지), late5 = 9/10~10/6 5분봉(봉 크기 같게).
데이터 정리: 프리·애프터 봉 중 고가·저가가 앞뒤 15봉 종가 중앙값에서 8% 넘게 벗어난 봉(단발 오류 체결)은 뺀다.
사용: py intra_C.py   → results/intra_C.txt
"""
import os, sys, time
from datetime import date
import numpy as np, pandas as pd

CODE = r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis\dip_reason\code"
sys.path.insert(0, CODE)
from fetch import universe, user_universe   # noqa: E402
from events import rsi                       # noqa: E402  (osc.rsi, Pine과 같은 RSI)

SCR = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad"
RAW = SCR + "/dip/raw"
DAILY = SCR + "/box/daily"
CACHE = SCR + "/box/intra_C_cache"
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT = os.path.join(RES, "intra_C.txt")
EARN = os.path.join(RES, "earnings_cache.csv")

USER78 = set(user_universe()["합집합"])
ALL137 = USER78 | set(universe())
LEV11 = ["SOXL", "SOXS", "TQQQ", "SQQQ", "TECL", "LABU", "NAIL", "DPST", "KORU", "SPXL", "TNA"]
SYMS = sorted(ALL137) + LEV11
UNIV = {"사용자78": USER78, "전체137": ALL137, "3배ETF11": set(LEV11)}

CFG = {"early5": dict(mins=5, rl=9, lo=date(2026, 7, 14), hi=date(2026, 9, 8), name="고르기 7/14~9/8 5분봉"),
       "late3": dict(mins=3, rl=14, lo=date(2026, 9, 10), hi=date(2026, 10, 8), name="확인 9/10~10/6 3분봉"),
       "late5": dict(mins=5, rl=9, lo=date(2026, 9, 10), hi=date(2026, 10, 6), name="확인 9/10~10/6 5분봉")}
NS = (30, 60, 120, 240)          # 박스 길이(분)
QN = {30: 0.12, 60: 0.18, 120: 0.28, 240: 0.50}   # 수축: 박스 폭 <= QN x 일봉 ATR14% (early5 정규장 전 봉 분포의 하위 33%, 수익 안 봄)
VOLK = 2.0                       # 거래량: 같은 시각 평균의 2배
RSIX, TGT, TRAIL = 70.0, 0.02, 0.015
XS = ("X1_RSI70", "X2_1시간", "X3_8시", "X4_+2%", "X5_추적1.5%", "X6_박스높이")
XSHORT = XS[:5]
SCOPES = {"정규": "i", "깨어": "i", "정규종일": "ii"}
NRAND6 = 10
SEED = 7


def dkey(d):
    return d.year * 10000 + d.month * 100 + d.day


# ---------------------------------------------------------------- 데이터
def clean(b):
    idx = b.index
    mod = idx.hour * 60 + idx.minute
    ext = (mod < 570) | (mod >= 960)
    med = b.Close.rolling(15, center=True, min_periods=5).median()
    dev = np.maximum((b.High / med - 1).abs(), (b.Low / med - 1).abs())
    return b[~(ext & (dev > 0.08))]


def load_bars(s, mins):
    b = pd.read_parquet(f"{RAW}/{'5m' if mins == 5 else '1m'}_{s}.parquet")[["Open", "High", "Low", "Close", "Volume"]]
    b = clean(b[(b.High > 0)].dropna(subset=["Close"]))
    if mins == 3:
        b = b.resample("3min", label="left", closed="left").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
    return b


def daily_atr(s):
    p = f"{DAILY}/1d_{s}.parquet"
    d = pd.read_parquet(p if os.path.exists(p) else f"{RAW}/1d_{s}.parquet")
    pc = d.Close.shift(1)
    tr = pd.concat([d.High - d.Low, (d.High - pc).abs(), (d.Low - pc).abs()], axis=1).max(axis=1)
    j = d.Close / pc - 1
    tr[(j < -0.4) | (j > 0.8)] = np.nan          # 분할 의심 날
    a = (tr.rolling(14, min_periods=10).mean() / d.Close).shift(1)   # D일 값 = D-1 종가까지
    a.index = [dkey(x) for x in pd.to_datetime(d.index)]
    return a


def vol_base(s, days):
    """정규장 5분 칸별 거래량의 지난 10거래일 평균(오늘 제외). 반환 Series[(날짜, 칸)]."""
    f = pd.read_parquet(f"{RAW}/5m_{s}.parquet")
    idx = f.index
    mod = np.asarray(idx.hour * 60 + idx.minute)
    reg = (mod >= 570) & (mod < 960)
    dk = np.asarray(idx.year * 10000 + idx.month * 100 + idx.day)
    x = pd.DataFrame({"d": dk[reg], "m": mod[reg], "v": f.Volume.values[reg]})
    tab = x.pivot_table(index="d", columns="m", values="v", aggfunc="sum")
    tab = tab.reindex(sorted(set(tab.index) | set(days)))
    return tab.rolling(10, min_periods=5).mean().shift(1).stack()


def one(s, mins, rl):
    b = load_bars(s, mins)
    idx = b.index
    tks = idx.as_unit("s").asi8
    mod = np.asarray(idx.hour * 60 + idx.minute, dtype=np.int64)
    dk = np.asarray(idx.year * 10000 + idx.month * 100 + idx.day, dtype=np.int64)
    o, h, l, c, v = (b[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    kst = idx.tz_convert("Asia/Seoul")
    t8 = kst.normalize() + pd.Timedelta(hours=8)
    dead = np.where(kst < t8, t8.as_unit("s").asi8, (t8 + pd.Timedelta(days=1)).as_unit("s").asi8)
    jd = np.searchsorted(tks, dead, "left")
    jd[jd >= len(tks)] = -1
    atr = daily_atr(s)
    datr = pd.Series(dk).map(atr).values.astype(float)
    base = vol_base(s, set(dk.tolist()))
    slot = mod // 5 * 5
    base.index.names = ["d", "m"]
    bv = base.reindex(pd.MultiIndex.from_arrays([dk, slot], names=["d", "m"])).values.astype(float) * (mins / 5)
    reg = (mod >= 570) & (mod < 960)
    rvol = np.where(reg & (bv > 0), v / np.where(bv > 0, bv, 1), np.nan)
    A = dict(o=o, h=h, l=l, c=c, v=v, tk=tks, mod=mod, dk=dk, r=np.asarray(rsi(c, rl), dtype=float), jd=jd,
             datr=datr, rvol=rvol)
    for N in NS:
        nb = N // mins
        A[f"hi{N}"] = pd.Series(h).rolling(nb).max().shift(1).values
        A[f"lo{N}"] = pd.Series(l).rolling(nb).min().shift(1).values
    return A


def build(cfg):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{cfg}.npz")
    if os.path.exists(p):
        z = np.load(p)
        return {k: z[k] for k in z.files}
    mins, rl = CFG[cfg]["mins"], CFG[cfg]["rl"]
    parts, off = [], 0
    for sid, s in enumerate(SYMS):
        A = one(s, mins, rl)
        n = len(A["o"])
        A["jd"] = np.where(A["jd"] >= 0, A["jd"] + off, -1)
        A["sid"] = np.full(n, sid)
        parts.append(A)
        off += n
    G = {k: np.concatenate([P[k] for P in parts]) for k in parts[0]}
    np.savez(p, **G)
    return G


# ---------------------------------------------------------------- 매도 계산(벡터)
def nxt_true(mask):
    n = len(mask)
    ar = np.where(mask, np.arange(n), n)
    return np.r_[np.minimum.accumulate(ar[::-1])[::-1], n]


def tgt_stop(G, E, T, S, nxtA, chunk=20000):
    """매수 봉 E(시가 매수). 지정가 매도 T(다음 봉부터 늘 걸림), 손절: 종가 < S → 다음 깨어 있는 봉 시가, 8시 정리.
    반환 (매도 봉, 매도가)."""
    o, h, c, jd = G["o"], G["h"], G["c"], G["jd"]
    n = len(o)
    XI = np.empty(len(E), np.int64)
    PX = np.empty(len(E))
    for a in range(0, len(E), chunk):
        e = E[a:a + chunk]
        J = jd[e]
        W = int((J - e).max()) + 1
        m = np.arange(W)
        K = e[:, None] + m[None, :]
        vk = K < J[:, None]
        Kc = np.minimum(K, n - 1)
        st = vk & (c[Kc] < S[a:a + chunk, None])
        has = st.any(1)
        t = np.where(has, e + st.argmax(1) + 1, n)
        xs = np.minimum(nxtA[t], J)
        Jj = K + 1
        hit = (Jj < xs[:, None]) & (h[np.minimum(Jj, n - 1)] >= T[a:a + chunk, None])
        hh = hit.any(1)
        jl = e + hit.argmax(1) + 1
        XI[a:a + chunk] = np.where(hh, jl, xs)
        PX[a:a + chunk] = np.where(hh, np.maximum(o[np.minimum(jl, n - 1)], T[a:a + chunk]), o[xs])
    return XI, PX


def trail(G, E, nxtA, z=TRAIL, chunk=20000):
    o, c, jd = G["o"], G["c"], G["jd"]
    n = len(o)
    XI = np.empty(len(E), np.int64)
    for a in range(0, len(E), chunk):
        e = E[a:a + chunk]
        J = jd[e]
        W = int((J - e).max()) + 1
        K = e[:, None] + np.arange(W)[None, :]
        vk = K < J[:, None]
        C = np.where(vk, c[np.minimum(K, n - 1)], -np.inf)
        P = np.maximum(np.maximum.accumulate(C, axis=1), o[e][:, None])
        st = vk & (C <= P * (1 - z))
        t = np.where(st.any(1), e + st.argmax(1) + 1, n)
        XI[a:a + chunk] = np.minimum(nxtA[t], J)
    return XI


def all_exits(G, E, mode):
    """후보 매수 봉 E 전부에 대해 X1~X5(와 참고 매도) 매도 봉·수익. 전체 길이 배열로 돌려준다."""
    n = len(G["o"])
    o, jd = G["o"], G["jd"]
    nxtA = G["nxtA"] if mode == "i" else np.arange(n + 1)
    J = jd[E]
    out = {}

    def put(name, xi, px=None):
        XI = np.full(n, -1, np.int64)
        R = np.full(n, np.nan)
        XI[E] = xi
        R[E] = (o[xi] if px is None else px) / o[E] - 1
        out[name] = (XI, R)

    t = G["nxt_rsi"][E + 1]
    put("X1_RSI70", np.minimum(nxtA[t], J))
    t = np.searchsorted(G["key"], G["key"][E] + 3600, "left")
    put("X2_1시간", np.minimum(nxtA[t], J))
    put("X3_8시", J)
    xi, px = tgt_stop(G, E, o[E] * (1 + TGT), np.full(len(E), -np.inf), nxtA)
    put("X4_+2%", xi, px)
    put("X5_추적1.5%", trail(G, E, nxtA))
    if mode == "ii":
        t = np.searchsorted(G["key"], G["key"][E] + 4 * 3600, "left")
        put("R_4시간", np.minimum(t, J))
    ro = G["nxt_open"][E + 1]
    ok = (ro < n) & (G["sid"][np.minimum(ro, n - 1)] == G["sid"][E])
    xi = np.where(ok, ro, -1)
    XI = np.full(n, -1, np.int64)
    R = np.full(n, np.nan)
    XI[E] = xi
    R[E] = np.where(ok, o[np.minimum(ro, n - 1)] / o[E] - 1, np.nan)
    out["R_다음날시가"] = (XI, R)
    return out


def x6_exits(G, E, N, mode):
    n = len(G["o"])
    o = G["o"]
    nxtA = G["nxtA"] if mode == "i" else np.arange(n + 1)
    H, L = G[f"hi{N}"][E - 1], G[f"lo{N}"][E - 1]
    ok = np.isfinite(H) & np.isfinite(L)
    E2 = E[ok]
    T = o[E2] + (H[ok] - L[ok])
    S = (H[ok] + L[ok]) / 2
    xi, px = tgt_stop(G, E2, T, S, nxtA)
    XI = np.full(n, -1, np.int64)
    R = np.full(n, np.nan)
    TP = np.full(n, np.nan)
    SP = np.full(n, np.nan)
    XI[E2] = xi
    R[E2] = px / o[E2] - 1
    TP[E2] = (H[ok] - L[ok]) / o[E2]
    SP[E2] = (o[E2] - S) / o[E2]
    return XI, R, TP, SP


def chain(sig, XI):
    """한 종목 한 포지션(Pine pyramiding 0): 신호 봉 순서대로, 앞 거래 매도 봉 다음 봉부터 새로 산다."""
    n = len(sig)
    nx = nxt_true(sig & (XI >= 0))
    out = []
    e = nx[0]
    while e < n:
        out.append(e)
        e = nx[XI[e] + 1]
    return np.asarray(out, np.int64)


# ---------------------------------------------------------------- 통계
def tclu(df, col):
    s = df.groupby("day")[col].sum()
    if len(s) < 3 or s.std(ddof=1) == 0:
        return np.nan
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def stats(df):
    if not len(df):
        return dict(n=0)
    return dict(n=len(df), days=df.day.nunique(), avg=df.ret.mean(), win=(df.ret > 0).mean(), worst=df.ret.min(),
                sum=df.ret.sum(), t=tclu(df, "ret"), ra=df.ra.mean(), exa=df.exa.mean(), t_exa=tclu(df, "exa"),
                rb=df.rb.mean(), exb=df.exb.mean(), t_exb=tclu(df, "exb"))


# ---------------------------------------------------------------- 실행
def prepare(cfg):
    G = build(cfg)
    n = len(G["o"])
    sid, mod, dk = G["sid"], G["mod"], G["dk"]
    lo, hi = dkey(CFG[cfg]["lo"]), dkey(CFG[cfg]["hi"])
    awake = ((mod >= 240) & (mod < 660)) | ((mod >= 1140) & (mod < 1200))
    G["nxtA"] = nxt_true(awake)
    trig = np.zeros(n, bool)
    trig[1:] = G["r"][:-1] > RSIX
    G["nxt_rsi"] = nxt_true(trig)
    G["nxt_open"] = nxt_true(mod == 570)
    G["key"] = sid.astype(np.int64) * 10**10 + G["tk"]
    ses = np.where(mod < 570, 0, np.where(mod < 960, 1, 2))
    prev_same = np.r_[False, (sid[1:] == sid[:-1]) & (dk[1:] == dk[:-1])]
    prev_reg = np.r_[False, ses[:-1] == 1] & prev_same
    inper = (dk >= lo) & (dk <= hi) & (G["jd"] >= 0) & prev_same
    G["elig"] = {"정규": inper & (mod >= 570) & (mod < 660) & prev_reg,
                 "깨어": inper & awake,
                 "정규종일": inper & (mod >= 570) & (mod < 960) & prev_reg}
    cls = np.isin(sid, [SYMS.index(t) for t in LEV11]).astype(np.int64)
    G["gA"] = np.unique(cls * 10**11 + G["tk"], return_inverse=True)[1]          # 같은 시각(봉) · 주식/3배ETF
    G["gB"] = np.unique(sid.astype(np.int64) * 10**5 + mod, return_inverse=True)[1]  # 같은 종목 · 같은 시각
    cand = G["elig"]["정규"] | G["elig"]["깨어"] | G["elig"]["정규종일"]
    E = np.flatnonzero(cand)
    t0 = time.time()
    G["EX"] = {m: all_exits(G, E, m) for m in ("i", "ii")}
    E6 = {"i": np.flatnonzero(G["elig"]["정규"] | G["elig"]["깨어"]), "ii": np.flatnonzero(G["elig"]["정규종일"])}
    G["X6"] = {(N, m): x6_exits(G, E6[m], N, m) for N in NS for m in ("i", "ii")}
    print(f"[{cfg}] bars {n:,} 후보 {len(E):,} 매도 계산 {time.time() - t0:.0f}s", flush=True)
    # 신호
    c, h, l = G["c"], G["h"], G["l"]
    same = np.r_[False, sid[1:] == sid[:-1]]
    SIG = {}
    for N in NS:
        H, L = G[f"hi{N}"], G[f"lo{N}"]
        up = (c > H)
        cross = up & np.r_[False, ~up[:-1]] & same
        width = (H - L) / L
        cq = width <= QN[N] * G["datr"]
        cv = G["rvol"] >= VOLK
        dn = c < L
        bdn = dn & np.r_[False, ~dn[:-1]] & same
        tch = (l <= L * 1.003) & (c > L)
        atch = tch & np.r_[False, ~tch[:-1]] & same
        for nm, k in ((f"C_N{N}_없음", cross), (f"C_N{N}_수축", cross & cq), (f"C_N{N}_거래량", cross & cv),
                      (f"C_N{N}_수축+거래량", cross & cq & cv), (f"A_N{N}", atch), (f"B_N{N}", bdn)):
            SIG[nm] = (N, np.r_[False, k[:-1]])     # 봉 k 신호 → 봉 k+1 매수
        G[f"width{N}"] = width
    G["SIG"] = SIG
    return G


def baselines(G, scope, mode):
    """RA = 같은 시각 봉에 산 다른 종목 전부(주식 137 / 3배ETF 11 따로)의 같은 매도 평균,
    RB = 같은 종목을 기간 안 다른 날 같은 시각 봉에 산 같은 매도 평균(매수 후 보유 대용). 둘 다 자기 거래는 뺀다."""
    el = G["elig"][scope]
    out = {}
    for x, (XI, R) in G["EX"][mode].items():
        m = el & np.isfinite(R)
        k = np.flatnonzero(m)
        res = []
        for g in ("gA", "gB"):
            gid = G[g]
            ng = gid.max() + 1
            sm = np.bincount(gid[m], R[m], minlength=ng)
            ct = np.bincount(gid[m], minlength=ng)
            arr = np.full(len(R), np.nan)
            c1 = ct[gid[k]] - 1
            arr[k] = np.where(c1 > 0, (sm[gid[k]] - R[k]) / np.maximum(c1, 1), np.nan)
            res.append(arr)
        out[x] = tuple(res)
    return out


def rand_x6(G, scope, mode, tr, TP, SP, rng):
    """X6 기준: 신호의 목표·손절 %를 그대로 쓰고 RA·RB 후보에서 10개씩 뽑아 같은 매도."""
    if not len(tr):
        return np.array([]), np.array([])
    mem = np.flatnonzero(G["elig"][scope] & (G["jd"] >= 0))
    o = G["o"]
    nxtA = G["nxtA"] if mode == "i" else np.arange(len(o) + 1)
    out = []
    for g in ("gA", "gB"):
        gid = G[g]
        ordr = mem[np.argsort(gid[mem], kind="stable")]
        gm = gid[ordr]
        a, b = np.searchsorted(gm, gid[tr], "left"), np.searchsorted(gm, gid[tr], "right")
        owner, ent = [], []
        for k in range(len(tr)):
            pool = ordr[a[k]:b[k]]
            pool = pool[(pool != tr[k]) & (G["sid"][pool] != G["sid"][tr[k]])] if g == "gA" else pool[pool != tr[k]]
            if len(pool) > NRAND6:
                pool = rng.choice(pool, NRAND6, replace=False)
            owner.append(np.full(len(pool), k))
            ent.append(pool)
        owner, ent = np.concatenate(owner), np.concatenate(ent)
        if not len(ent):
            out.append(np.full(len(tr), np.nan))
            continue
        xi, px = tgt_stop(G, ent, o[ent] * (1 + TP[tr][owner]), o[ent] * (1 - SP[tr][owner]), nxtA)
        r = px / o[ent] - 1
        ct = np.bincount(owner, minlength=len(tr))
        out.append(np.where(ct > 0, np.bincount(owner, r, minlength=len(tr)) / np.maximum(ct, 1), np.nan))
    return out[0], out[1]


def trades_df(G, tr, ret, ra, rb, XI):
    sid = G["sid"][tr]
    df = pd.DataFrame(dict(sym=np.asarray(SYMS)[sid], day=G["dk"][tr], e=tr, xi=XI[tr], ret=ret, ra=ra, rb=rb,
                           exa=ret - ra, exb=ret - rb, mod=G["mod"][tr]))
    return df[np.isfinite(df.ret) & np.isfinite(df.ra) & np.isfinite(df.rb)]


def run_cfg(cfg, rng):
    G = prepare(cfg)
    rows, TR = [], {}
    for scope, mode in SCOPES.items():
        BL = baselines(G, scope, mode)
        G.setdefault("BL", {})[scope] = BL
        el = G["elig"][scope]
        for nm, (N, sig0) in G["SIG"].items():
            sig = sig0 & el
            xs = XS if nm.startswith("C_") else XSHORT
            for x in xs:
                if x == "X6_박스높이":
                    XI, R, TP, SP = G["X6"][(N, mode)]
                    tr = chain(sig, XI)
                    ra, rb = rand_x6(G, scope, mode, tr, TP, SP, rng)
                else:
                    XI, R = G["EX"][mode][x]
                    tr = chain(sig, XI)
                    ra, rb = BL[x][0][tr], BL[x][1][tr]
                df = trades_df(G, tr, R[tr], ra, rb, XI)
                TR[(cfg, scope, nm, x)] = df
                for un, us in UNIV.items():
                    rows.append(dict(cfg=cfg, scope=scope, entry=nm, exit=x, univ=un, **stats(df[df.sym.isin(us)])))
        print(f"  [{cfg}] {scope} 끝", flush=True)
    return G, pd.DataFrame(rows), TR


def fmt_row(r):
    if not r.get("n"):
        return "n=0"
    return (f"n={r['n']:>5} 날={r['days']:>3.0f} 평균={r['avg']:+.2%} 승률={r['win']:.0%} 최악={r['worst']:+.1%} "
            f"합={r['sum']:+.2f} t={r['t']:+.2f} | RA={r['ra']:+.2%} 초과A={r['exa']:+.2%}(t{r['t_exa']:+.2f}) "
            f"| RB={r['rb']:+.2%} 초과B={r['exb']:+.2%}(t{r['t_exb']:+.2f})")


def earn_days():
    if not os.path.exists(EARN):
        return {}
    e = pd.read_csv(EARN)
    out = {}
    for t, s in zip(e.ticker, e.earnings_dt):
        ts = pd.Timestamp(s)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("America/New_York")
        d = ts.normalize()
        if ts.hour * 60 + ts.minute >= 960:
            d = d + pd.offsets.BDay(1)
        out.setdefault(t, set()).update({dkey(d), dkey(d + pd.offsets.BDay(1))})
    return out


def base_summary():
    sys.path.insert(0, CODE)
    from stop_lab import base_trades
    lines = []
    for p, nm in (("early", "7/14~9/8 5분봉 근사"), ("late", "9/10~10/6 3분봉")):
        for reg in (False, True):
            b = base_trades(p, reg)
            for un in ("사용자78", "전체137"):
                x = b[b.sym.isin(UNIV[un])]
                lines.append(f"  이격매수 {nm} {'정규장만' if reg else '24시간'} {un}: n={len(x)} 평균={x.ret.mean():+.2%} "
                             f"승률={(x.ret > 0).mean():.0%} 최악={x.ret.min():+.1%} 합={x.ret.sum():+.2f}")
    return lines


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    res, TRS, GS = [], {}, {}
    for cfg in CFG:
        G, df, TR = run_cfg(cfg, rng)
        res.append(df)
        TRS.update(TR)
        GS[cfg] = G
        print(f"[{cfg}] 끝 {time.time() - t0:.0f}s", flush=True)
    R = pd.concat(res, ignore_index=True)
    R.to_csv(os.path.join(CACHE, "intra_C_rows.csv"), index=False, encoding="utf-8-sig")
    report(R, TRS, GS)
    print(f"완료 {time.time() - t0:.0f}s → {OUT}")


def pick(R, cfg, scope, entry, exit, univ):
    x = R[(R.cfg == cfg) & (R.scope == scope) & (R.entry == entry) & (R.exit == exit) & (R.univ == univ)]
    return x.iloc[0].to_dict() if len(x) else {"n": 0}


def cell(r, short=False):
    if not r.get("n"):
        return "n=0"
    if short:
        return f"n={r['n']:>4} {r['avg']:+.2%} A{r['exa']:+.2%}({r['t_exa']:+.1f}) B{r['exb']:+.2%}({r['t_exb']:+.1f})"
    return fmt_row(r)


def beats(r, nmin=30):
    return r.get("n", 0) >= nmin and r["exa"] > 0 and r["exb"] > 0


def wk_of(day):
    return pd.to_datetime(pd.Series(day).astype(str)).dt.to_period("W-FRI").astype(str).str[-10:].values


def report(R, TRS, GS):
    L = []
    w = L.append
    R = R.assign(score=np.minimum(R.t_exa, R.t_exb))
    w("박스권 위 돌파 매수(C) 분봉 연구 — intra_C (작성 2026-10-09, yfinance 1분·5분봉, 미 동부 시각)")
    w(__doc__.split("사용:")[0].strip())
    w("")
    w(f"종목: 사용자78 / 전체137(사용자78 ∪ AI 127) / 3배ETF11(참고). 박스 {NS}분, 수축 박스 폭 <= {QN} x 일봉 ATR14%, 거래량 {VOLK}배.")
    C_rules = [(e, x) for e in sorted({k for k in GS["early5"]["SIG"] if k.startswith("C_")}) for x in XS]
    w(f"시험한 C 규칙: 진입 16개(박스 4 x 수축 유무 x 거래량 유무) x 매도 6개 = {len(C_rules)}개 "
      f"(+ 최선 진입의 참고 매도 2개: 4시간·다음 날 시가, + 비교용 A·B 간이판 8진입 x 5매도 = 40개 — 고르기 대상 아님)")
    w("고르기 규칙(확인 기간을 보기 전에 정함): early5·정규·사용자78, 건수 30 이상 중 min(초과A t, 초과B t) 최대.")
    w("'기준보다 나음' = 건수 30 이상, 초과A > 0, 초과B > 0. t = 매수일 군집 t(날짜별 합의 평균/표준오차).")
    w("초과A = 수익 - RA, 초과B = 수익 - RB (정의는 위 '비교 기준').")
    w("주의: 처음엔 '같은 종목·같은 날·같은 구간(09:30~11:00) 무작위'를 기준으로 했다. 돌파가 나온 날은 신호 전 봉이 싸서 무작위가 좋아지고(C 불리),")
    w("      이탈이 나온 날은 반대(B 유리)라 신호가 생긴 경로를 미리 아는 편향이 있어 버렸다(그때 C 초과 t -3~-7, B +1~+5로 과장됨).")
    w("")
    w("== 비교: 지금 이격매수 지표(stop_lab.base_trades, 같은 종목 묶음. 24시간 = 지정가 매수가 자는 동안에도 걸림)")
    try:
        L.extend(base_summary())
    except Exception as ex:
        w(f"  (이격매수 기준 계산 실패: {ex})")
    w("")
    w("== 두 기간 모두 기준보다 나은 C 규칙 수 (정규 범위)")
    for un in ("사용자78", "전체137"):
        ce = cl = cb = cb5 = pos = 0
        for e, x in C_rules:
            a, b, c5 = (pick(R, cf, "정규", e, x, un) for cf in ("early5", "late3", "late5"))
            ce += beats(a)
            cl += beats(b)
            cb += beats(a) and beats(b)
            cb5 += beats(a) and beats(b) and beats(c5)
            pos += a.get("n", 0) >= 30 and b.get("n", 0) >= 30 and a["avg"] > 0 and b["avg"] > 0
        w(f"  {un}: 고르기 {ce}/{len(C_rules)}, 확인(3분) {cl}/{len(C_rules)}, 두 기간 모두 {cb}/{len(C_rules)}, "
          f"세 칸(early5·late3·late5) 모두 {cb5}/{len(C_rules)} | 기준 없이 평균 수익만 두 기간 플러스 {pos}/{len(C_rules)}")
    w("")
    sel = R[(R.cfg == "early5") & (R.scope == "정규") & (R.univ == "사용자78") & (R.entry.str.startswith("C_")) & (R.n >= 30)]
    sel = sel.sort_values("score", ascending=False)
    best = sel.iloc[0]
    be, bx = best.entry, best.exit
    w(f"== 고른 최선안: {be} + {bx}")
    for cfg in CFG:
        for scope in SCOPES:
            for un in UNIV:
                w(f"  {CFG[cfg]['name']:<20} {scope:<5} {un:<7} {fmt_row(pick(R, cfg, scope, be, bx, un))}")
    w("")
    w("== 고르기 상위 12개와 확인 기간(선택편향 확인). 칸 = n 평균 초과A(t) 초과B(t)")
    for _, r in sel.head(12).iterrows():
        w(f"  {r.entry + ' ' + r.exit:<30} 고르기78 {cell(r.to_dict(), True)} | 확인3분78 {cell(pick(R, 'late3', '정규', r.entry, r.exit, '사용자78'), True)}"
          f" | 확인5분78 {cell(pick(R, 'late5', '정규', r.entry, r.exit, '사용자78'), True)} | 확인3분137 {cell(pick(R, 'late3', '정규', r.entry, r.exit, '전체137'), True)}")
    w("")
    w(f"== 최선 진입 {be}의 매도별 (정규)")
    for x in XS:
        for un in ("사용자78", "전체137"):
            w(f"  {x:<12} {un:<7} 고르기  {fmt_row(pick(R, 'early5', '정규', be, x, un))}")
            w(f"  {'':<12} {'':<7} 확인3분 {fmt_row(pick(R, 'late3', '정규', be, x, un))}")
    w("  참고 매도(정규종일 = 수면 제약 없음 4시간 / 정규 = 다음 날 정규장 시가, 8시 정리 없이 밤새 보유):")
    for cfg in ("early5", "late3"):
        G = GS[cfg]
        sig = G["SIG"][be][1]
        for scope, x in (("정규종일", "R_4시간"), ("정규", "R_다음날시가")):
            mode = SCOPES[scope]
            XI, Rr = G["EX"][mode][x]
            tr = chain(sig & G["elig"][scope], XI)
            BL = G["BL"][scope]
            df = trades_df(G, tr, Rr[tr], BL[x][0][tr], BL[x][1][tr], XI)
            for un in ("사용자78", "전체137"):
                w(f"    {CFG[cfg]['name']:<20} {scope:<5} {x:<10} {un:<7} {fmt_row(stats(df[df.sym.isin(UNIV[un])]))}")
    w("")
    w("== A(버팀)·B(이탈)·C(돌파) 같은 박스·같은 매도 비교 (정규 범위, 필터 없음). 칸 = n 평균 초과A(t) 초과B(t)")
    for un in ("사용자78", "전체137"):
        for cfg in ("early5", "late3"):
            w(f"  [{un} {CFG[cfg]['name']}]")
            for N in NS:
                for x in XSHORT:
                    cells = [f"{kind[0]} " + cell(pick(R, cfg, "정규", kind, x, un), True)
                             for kind in (f"A_N{N}", f"B_N{N}", f"C_N{N}_없음")]
                    w(f"    N{N:<3} {x:<12} " + " | ".join(cells))
    w("")
    w("== RSI 매도 문제 점검 (정규·사용자78, 필터 없는 C): 매수 봉 종가 RSI>70 비율, X1이 매수 다음 봉에 바로 판 비율")
    for cfg in ("early5", "late3"):
        G = GS[cfg]
        for N in NS:
            df = TRS[(cfg, "정규", f"C_N{N}_없음", "X1_RSI70")]
            df = df[df.sym.isin(USER78)]
            d3 = TRS[(cfg, "정규", f"C_N{N}_없음", "X3_8시")]
            d3 = d3[d3.sym.isin(USER78)]
            if not len(df):
                continue
            rr = G["r"][df.e.values]
            quick = (df.xi.values - df.e.values) == 1
            w(f"  {CFG[cfg]['name']:<20} N{N:<3} n={len(df)} 매수봉 RSI>70 {np.mean(rr > RSIX):.0%} | 다음 봉에 RSI 매도 {quick.mean():.0%} "
              f"(그 거래 평균 {df.ret.values[quick].mean() if quick.any() else np.nan:+.2%}) | X1 평균 {df.ret.mean():+.2%} vs X3 8시 {d3.ret.mean():+.2%}")
    w("")
    w(f"== 최근 장세: 최선안 {be} + {bx} 주별 (정규, 3분봉, 주 = 금요일 끝)")
    df = TRS[("late3", "정규", be, bx)]
    df = df.assign(wk=wk_of(df.day))
    for un in ("사용자78", "전체137", "3배ETF11"):
        x = df[df.sym.isin(UNIV[un])]
        for wk, g in x.groupby("wk"):
            w(f"  {un:<7} ~{wk}: n={len(g):>4} 평균={g.ret.mean():+.2%} 승률={(g.ret > 0).mean():.0%} 초과A={g.exa.mean():+.2%} 초과B={g.exb.mean():+.2%} 합={g.ret.sum():+.2f}")
    w("  고르기 기간 월별(5분봉, 정규·사용자78):")
    d0 = TRS[("early5", "정규", be, bx)]
    d0 = d0[d0.sym.isin(USER78)].assign(mo=lambda z: (z.day // 100) % 100)
    for mo, g in d0.groupby("mo"):
        w(f"    {mo}월: n={len(g):>4} 평균={g.ret.mean():+.2%} 초과A={g.exa.mean():+.2%} 초과B={g.exb.mean():+.2%}")
    w("  같은 진입의 매도별 최근 주별(정규·전체137, 3분봉) 평균/초과A:")
    for x in XS:
        d = TRS[("late3", "정규", be, x)]
        d = d[d.sym.isin(ALL137)]
        d = d.assign(wk=wk_of(d.day))
        w(f"    {x:<12} " + " | ".join(f"~{wk} {g.ret.mean():+.2%}/{g.exa.mean():+.2%}(n{len(g)})" for wk, g in d.groupby("wk")))
    w("")
    ed = earn_days()
    w("== 실적 반응일(발표 뒤 첫 정규장과 그다음 날) 뺀 결과 — 최선안 (정규)")
    for cfg in CFG:
        df = TRS[(cfg, "정규", be, bx)]
        m = np.array([d in ed.get(s, set()) for s, d in zip(df.sym, df.day)], dtype=bool)
        for un in ("사용자78", "전체137"):
            x = df[~m & df.sym.isin(UNIV[un])]
            w(f"  {CFG[cfg]['name']:<20} {un:<7} 실적일 뺌 {fmt_row(stats(x))} (뺀 거래 {int((m & df.sym.isin(UNIV[un])).sum())})")
    w("")
    w("== 전 C 규칙 표 (정규 범위). 칸 = n 평균 초과A(t) 초과B(t)")
    for e, x in C_rules:
        cells = [f"{cfg}·{un[:2]} " + cell(pick(R, cfg, "정규", e, x, un), True)
                 for cfg in ("early5", "late3") for un in ("사용자78", "전체137")]
        w(f"  {e + ' ' + x:<30} " + " | ".join(cells))
    w("")
    w("== 깨어 범위(프리·애프터 포함) 고르기 상위 8개와 확인 — 프리·애프터는 거래량 0·얇은 가격이라 참고만")
    s2 = R[(R.cfg == "early5") & (R.scope == "깨어") & (R.univ == "사용자78") & (R.entry.str.startswith("C_")) & (R.n >= 30)]
    for _, r in s2.sort_values("score", ascending=False).head(8).iterrows():
        w(f"  {r.entry + ' ' + r.exit:<30} 고르기78 {cell(r.to_dict(), True)} | 확인3분78 {cell(pick(R, 'late3', '깨어', r.entry, r.exit, '사용자78'), True)}")
    w("")
    w("== 정규종일(수면 제약 없음, 09:30~16:00 매수) 고르기 상위 8개와 확인 — 신호 품질 참고")
    s3 = R[(R.cfg == "early5") & (R.scope == "정규종일") & (R.univ == "사용자78") & (R.entry.str.startswith("C_")) & (R.n >= 30)]
    for _, r in s3.sort_values("score", ascending=False).head(8).iterrows():
        w(f"  {r.entry + ' ' + r.exit:<30} 고르기78 {cell(r.to_dict(), True)} | 확인3분78 {cell(pick(R, 'late3', '정규종일', r.entry, r.exit, '사용자78'), True)}")
    os.makedirs(RES, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
