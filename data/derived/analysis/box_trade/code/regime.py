"""박스권 매매 장세 지도(2026-10-09, 키 regime): '바닥 이탈 매수(B)가 최근 장세에 더 맞나'를 시간·장세별로 잰다.

사용자 주장: A(버팀) = 특정 기간 최저점(밴드)을 안 깨면 거기서 반등, B(이탈이 바닥) = 지지선을 막 깨는 점이 오히려 바닥(손절 물량),
'B가 최근 트렌드에 더 맞다'(예: NKE 10/1 장 뒤 실적 → 10/2 갭하락 저가 31.97이 바닥). 비교용 C = 박스 상단 돌파.

표준 정의(일봉, N=20):
  Lmin = 직전 N일(t-N..t-1) 최저 저가, Hmax = 직전 N일 최고 고가 — 신호일 t 봉은 안 씀.
  A: Lmin <= 저가[t] <= Lmin*1.01 (안 깨고 1% 안까지 내려와 버팀) → t+1 시가 매수.
  B: 장 시작 전 Lmin*(1-dep) 지정가 매수를 걸어 둠 → 저가[t] < 지정가면 체결, 체결가 = min(시가[t], 지정가).
  C: 종가[t] > Hmax → t+1 시가 매수.
  Bn(진단용): B와 같은 이탈(저가[t] < Lmin)인데 그날 지정가 대신 t+1 시가 매수 — 당일 지정가 체결 이점을 뺀 '이탈 뒤 흐름'만.
매도(같은 매도 규칙끼리 A/B/C 비교): Hh = 매수 봉 e에서 h봉 뒤 시가(e+h 시가, h=1·3·5·10),
  T5S5 = 매수가 +5% 지정가 매도(B는 매수 당일 고가는 안 씀, 갭 위면 시가), 종가 < 매수가 -5%면 다음 날 시가 매도, 최대 10봉(e+10 시가).
  (사용자 조건: 미국 시가 = 한국 22:30/23:30 깨어 있는 시간 → 시가 매수·매도 가능, 자는 동안은 아래 매수·위 매도 지정가만 → 손절은 종가 확인 후 다음 시가)
같은 종목 겹치기 금지: 보유 중 신호는 버림(e >= 직전 매도 봉일 때만 새로 산다) — Pine pyramiding=0과 같음.
무작위(주 기준) = 같은 묶음 모든 종목이 같은 매수 봉 e 시가에 사서 같은 매도 규칙을 쓴 평균(같은 날 시장 효과 제거).
  = '같은 보유 기간 매수 후 보유'. 보조 기준 = 같은 종목·같은 달 모든 날 시가 진입 평균(이전 daily.py 관례).
장세(신호 전날 t-1 종가 기준, 미래 참조 없음): QQQ 종가 > 50일 단순평균이면 '위'; QQQ 20일 실현 변동성(연율) > 기준이면 '높음'.
  변동성 기준 = 2024-11~2025-12 중앙값(2026 구간엔 미리 알 수 있는 값).
실적 직후(나이키형): 실적 발표 반응 첫 거래일 r(16시 이후 발표면 다음 거래일, 그 전이면 당일)부터 r+2까지 사이 신호. 발표일은 미리 정해진 일정.
분할 의심: 종가/전날 종가 또는 시가/전날 종가가 0.6 미만·1.8 초과인 날이 박스~매도 구간에 있으면 그 거래·무작위 칸 제외.
기간(신호일 기준): 참고 2024-11-01~2025-12-31, 고르기 2026-01-02~06-30, 확인 2026-07-01~10-08. 최근 3개월 2026-07-09~, 6개월 2026-04-09~.
t: 날짜 군집(거래일 max(5,h) 묶음) CR1.  묶음: S&P 500(현재 구성 종목, 생존 편향 있음), 사용자78(TV AI ∪ TIME·KoAct 미국 보유).
사용: py regime.py   (box_trade/code 폴더)
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "../results"))
OUT = os.path.join(RES, "regime.txt")
CACHE = os.path.join(RES, "earnings_cache.csv")
DAILY = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/box/daily"
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "../../dip_reason/code")))
from fetch import user_universe  # noqa: E402

ETF = {"SOXL", "SOXS", "TQQQ", "SQQQ", "TECL", "LABU", "NAIL", "DPST", "KORU", "SPXL", "TNA", "QQQ", "SPY", "IWM", "SMH", "SOXX"}
P_REF = ("2024-11-01", "2025-12-31")
P_SEL = ("2026-01-02", "2026-06-30")
P_CONF = ("2026-07-01", "2026-10-08")
P_R3 = ("2026-07-09", "2026-10-08")
P_R6 = ("2026-04-09", "2026-10-08")
P_PRE = ("2024-11-01", "2026-04-08")
EXITS = ("H1", "H3", "H5", "H10", "T5S5")
HZ = {"H1": 1, "H3": 3, "H5": 5, "H10": 10, "T5S5": 10}
LINES = []
THRU = 0.0          # B 체결 조건: 저가 < 지정가*(1-THRU). 기본 0, 보수 점검 때 0.002(0.2% 뚫려야 체결)


def out(s=""):
    print(s)
    LINES.append(str(s))


# ---------------------------------------------------------------- 데이터
def ensure_earnings(syms):
    """실적 발표일 캐시(공유). 없으면 yfinance get_earnings_dates(limit=12)로 만든다."""
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE)
    import yfinance as yf
    rows = []
    for t in syms:
        for k in range(4):
            try:
                d = yf.Ticker(t).get_earnings_dates(limit=12)
                if d is not None and len(d):
                    rows += [(t, ix.isoformat(), r.get("EPS Estimate"), r.get("Reported EPS"), r.get("Surprise(%)"))
                             for ix, r in d.iterrows()]
                break
            except Exception:
                time.sleep(3 * (k + 1))
        time.sleep(0.5)
    df = pd.DataFrame(rows, columns=["ticker", "earnings_dt", "eps_estimate", "reported_eps", "surprise_pct"])
    df.to_csv(CACHE + ".tmp", index=False)
    os.replace(CACHE + ".tmp", CACHE)
    return df


def load():
    dfs = {os.path.basename(f)[3:-8]: pd.read_parquet(f) for f in sorted(glob.glob(DAILY + "/1d_*.parquet"))}
    dates = dfs["SPY"].index
    tick = sorted(dfs)
    mats = {}
    for c in ("Open", "High", "Low", "Close"):
        m = np.column_stack([dfs[t][c].reindex(dates).to_numpy(float) for t in tick])
        m[~(m > 0)] = np.nan
        mats[c] = m
    return dates, tick, mats["Open"], mats["High"], mats["Low"], mats["Close"]


DATES, TICK, O, H, L, C = load()
T, M = O.shape
TI = {t: i for i, t in enumerate(TICK)}
DSTR = np.array([d.strftime("%Y-%m-%d") for d in DATES])
MON = np.array([d[:7] for d in DSTR])
MONTHS = sorted(set(MON))
MI = np.searchsorted(MONTHS, MON)

SP = set(pd.read_csv(os.path.join(DAILY, "sp500_list.csv")).iloc[:, 0]) & set(TICK)
U78 = set(user_universe()["합집합"]) & set(TICK)
UNI = {"SP500": np.array([t in SP for t in TICK]), "U78": np.array([t in U78 for t in TICK])}

# 분할 의심 날(그 날과 전날 사이 점프)
_pc = np.vstack([np.full((1, M), np.nan), C[:-1]])
with np.errstate(invalid="ignore", divide="ignore"):
    _bad = ((C / _pc < 0.6) | (C / _pc > 1.8) | (O / _pc < 0.6) | (O / _pc > 1.8))
CB = np.cumsum(_bad, axis=0)


def badfree(j, a, b):
    """(a, b] 사이 분할 의심 날이 없으면 True."""
    a = np.clip(a, 0, T - 1)
    b = np.clip(b, 0, T - 1)
    return CB[b, j] - CB[a, j] == 0


def roll(x, n, how):
    d = pd.DataFrame(x).rolling(n, min_periods=n)
    return (d.min() if how == "min" else d.max()).shift(1).to_numpy()


LMIN = {n: roll(L, n, "min") for n in (10, 20, 60)}
HMAX = {n: roll(H, n, "max") for n in (10, 20, 60)}


def regime_arrays(close_q):
    q = pd.Series(close_q)
    sma = q.rolling(50, min_periods=50).mean()
    above = (q > sma).astype(float).where(sma.notna()).shift(1).to_numpy()          # t-1 종가 기준
    vol = np.log(q / q.shift(1)).rolling(20, min_periods=20).std().mul(np.sqrt(252)).shift(1).to_numpy()
    return above, vol


ABOVE, VOL = regime_arrays(C[:, TI["QQQ"]])
_ref = (DSTR >= P_REF[0]) & (DSTR <= P_REF[1])
VTHR = float(np.nanmedian(VOL[_ref]))
HIVOL = np.where(np.isnan(VOL), np.nan, (VOL > VTHR).astype(float))

# 실적 반응 창 EW[t, j]
EARN = ensure_earnings([t for t in TICK if t not in ETF])
_ed = pd.to_datetime(EARN.earnings_dt.str[:19])
_day = _ed.dt.normalize().to_numpy().astype("datetime64[D]")
_cal = DATES.to_numpy().astype("datetime64[D]")
_pos = np.searchsorted(_cal, _day)
_ok = _pos < T
_is = np.zeros(len(_pos), bool)
_is[_ok] = _cal[_pos[_ok]] == _day[_ok]
_r = _pos + ((_ed.dt.hour.to_numpy() >= 16) & _is)
EW = np.zeros((T, M), bool)
for tk, r in zip(EARN.ticker, _r):
    if tk in TI and r < T:
        EW[r:r + 3, TI[tk]] = True
EARN_COVER = sorted(set(EARN.ticker) & set(TICK))

# 다음 5일 동안 그날 저가가 안 깨짐(바닥 유지)
_fut = pd.DataFrame(L[::-1]).rolling(5, min_periods=5).min().shift(1).to_numpy()[::-1]   # min L[t+1..t+5]
with np.errstate(invalid="ignore"):
    HOLD5 = np.where(np.isnan(_fut) | np.isnan(L), np.nan, (_fut >= L).astype(float))


# ---------------------------------------------------------------- 매도
def exit_px(j, e, P, intr, kind):
    """매수 봉 e, 매수가 P → (매도가, 매도 봉). 끝까지 데이터 없으면 NaN."""
    hz = HZ[kind]
    ok = e + hz <= T - 1
    xp = np.full(len(j), np.nan)
    xd = e + hz
    if kind[0] == "H":
        d = np.where(ok, e + hz, 0)
        xp = np.where(ok, O[d, j], np.nan)
        return xp, xd
    xd = xd.copy()
    tgt, stp = P * 1.05, P * 0.95
    alive = ok.copy()
    for k in range(10):
        d = np.minimum(e + k, T - 1)
        hi, op, cl = H[d, j], O[d, j], C[d, j]
        with np.errstate(invalid="ignore"):
            hitt = alive & (hi >= tgt) & ~((k == 0) & intr)
        xp[hitt] = np.maximum(op[hitt], tgt[hitt])
        xd[hitt] = d[hitt]
        alive &= ~hitt
        with np.errstate(invalid="ignore"):
            st = alive & (cl < stp)
        dn = np.minimum(d + 1, T - 1)
        xp[st] = O[dn, j][st]
        xd[st] = d[st] + 1
        alive &= ~st
    dn = np.minimum(e + 10, T - 1)
    xp[alive] = O[dn, j][alive]
    xp[~ok] = np.nan
    return xp, xd


# 무작위(시가 진입) 수익 행렬 R[kind][e, j]와 묶음별 같은 날 평균
EE, JJ = np.meshgrid(np.arange(T), np.arange(M), indexing="ij")
EE, JJ = EE.ravel(), JJ.ravel()
RAND, RAND_DAY, RAND_MON = {}, {}, {}
for k in EXITS:
    p0 = O[EE, JJ]
    xp, xd = exit_px(JJ, EE, p0, np.zeros(len(EE), bool), k)
    r = xp / p0 - 1
    r[~badfree(JJ, EE - 1, np.minimum(EE + HZ[k], T - 1))] = np.nan
    R = r.reshape(T, M)
    RAND[k] = R
    RAND_DAY[k] = {u: np.nanmean(np.where(m[None, :], R, np.nan), axis=1) for u, m in UNI.items()}
    mm = np.full((len(MONTHS), M), np.nan)
    for mi in range(len(MONTHS)):
        rows = R[MI == mi]
        if np.isfinite(rows).any():
            mm[mi] = np.nanmean(rows, axis=0)
    RAND_MON[k] = mm
HOLD5_DAY = {u: np.nanmean(np.where(m[None, :], HOLD5, np.nan), axis=1) for u, m in UNI.items()}


# ---------------------------------------------------------------- 규칙
def make_rules():
    rs = []
    for N in (10, 20, 60):
        for ex in EXITS:
            rs.append(("A", N, 0.0, "-", ex))
            rs.append(("B", N, 0.0, "-", ex))
            rs.append(("C", N, 0.0, "-", ex))
    for ex in EXITS:
        rs.append(("Bn", 20, 0.0, "-", ex))          # 이탈 확인 뒤 다음 날 시가(진단)
    for N in (20, 60):
        for ex in EXITS:
            rs.append(("B", N, 0.03, "-", ex))       # 3% 더 깊게 깬 곳
            rs.append(("B", N, 0.0, "earn", ex))     # 실적 직후 이탈(나이키형)
    for ex in EXITS:
        rs.append(("A", 20, 0.0, "earn", ex))
        rs.append(("C", 20, 0.0, "earn", ex))
        rs.append(("B", 20, 0.0, "qbelow", ex))      # QQQ 50일선 아래(t-1)
        rs.append(("A", 20, 0.0, "qbelow", ex))      # 같은 장세 필터의 A·C(비교용)
        rs.append(("C", 20, 0.0, "qbelow", ex))
        rs.append(("B", 20, 0.0, "hivol", ex))       # QQQ 변동성 높음(t-1)
    return rs


def rname(r):
    fam, N, dep, f, ex = r
    s = f"{fam}{N}"
    if dep:
        s += f"-{dep:.0%}"
    if f != "-":
        s += f"[{f}]"
    return f"{s}/{ex}"


def trades(rule):
    fam, N, dep, filt, ex = rule
    Lm, Hm = LMIN[N], HMAX[N]
    with np.errstate(invalid="ignore"):
        if fam == "A":
            sig = (L >= Lm) & (L <= Lm * 1.01)
        elif fam == "Bn":
            sig = L < Lm
        elif fam == "B":
            lim = Lm * (1 - dep)
            sig = L < lim * (1 - THRU)
        else:
            sig = C > Hm
    if filt == "earn":
        sig &= EW
    elif filt == "qbelow":
        sig &= (ABOVE == 0)[:, None]
    elif filt == "hivol":
        sig &= (HIVOL == 1)[:, None]
    t, j = np.nonzero(sig)
    off = 0 if fam == "B" else 1
    e = t + off
    m = e <= T - 1
    t, j, e = t[m], j[m], e[m]
    if fam == "B":
        P = np.minimum(O[t, j], lim[t, j])        # 미리 건 지정가: 갭 아래로 열리면 시가 체결
    else:
        P = O[e, j]
    xp, xd = exit_px(j, e, P, np.full(len(j), fam == "B"), ex)
    ret = xp / P - 1
    ok = np.isfinite(ret) & badfree(j, t - N, np.minimum(xd, T - 1))
    t, j, e, P, xd, ret = t[ok], j[ok], e[ok], P[ok], xd[ok], ret[ok]
    order = np.lexsort((t, j))
    keep = np.zeros(len(t), bool)
    lj, lx = -1, -1
    for i in order:
        if j[i] != lj:
            lj, lx = j[i], -1
        if e[i] >= lx:
            keep[i] = True
            lx = xd[i]
    t, j, e, P, xd, ret = t[keep], j[keep], e[keep], P[keep], xd[keep], ret[keep]
    df = pd.DataFrame({"t": t, "j": j, "e": e, "P": P, "xd": xd, "ret": ret, "date": DSTR[t], "mon": MON[t],
                       "ew": EW[t, j], "above": ABOVE[t], "hivol": HIVOL[t], "hold5": HOLD5[t, j],
                       "rnd2": RAND_MON[ex][MI[e], j], "g": t // max(5, HZ[ex]),
                       "day0": C[t, j] / P - 1 if fam == "B" else np.nan})
    for u, msk in UNI.items():
        df["in_" + u] = msk[j]
        df["rnd_" + u] = RAND_DAY[ex][u][e]
        df["h5b_" + u] = HOLD5_DAY[u][t]
    return df


def ct(x, g):
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return np.nan
    s = pd.Series(x - x.mean()).groupby(np.asarray(g)).sum().values
    G = len(s)
    if G < 5:                                   # 날짜 묶음 5개 미만이면 t를 내지 않음
        return np.nan
    se = np.sqrt(G / (G - 1) * (s ** 2).sum()) / n
    return x.mean() / se if se > 0 else np.nan


def sub(df, u, per=None):
    d = df[df["in_" + u]]
    if per is not None:
        d = d[(d.date >= per[0]) & (d.date <= per[1])]
    return d


def stats(d, u):
    n = len(d)
    if n == 0:
        return dict(n=0, avg=np.nan, win=np.nan, worst=np.nan, sum=0.0, t=np.nan, rnd=np.nan, exc=np.nan, te=np.nan,
                    exc2=np.nan, h5=np.nan, h5b=np.nan)
    ex = d.ret - d["rnd_" + u]
    ex2 = (d.ret - d.rnd2).dropna()
    return dict(n=n, avg=d.ret.mean(), win=(d.ret > 0).mean(), worst=d.ret.min(), sum=d.ret.sum(), t=ct(d.ret, d.g),
                rnd=d["rnd_" + u].mean(), exc=ex.mean(), te=ct(ex, d.g), exc2=ex2.mean() if len(ex2) else np.nan,
                h5=d.hold5.mean(), h5b=d["h5b_" + u].mean())


def fp(x, w=6, nd=2):
    return f"{x * 100:+{w}.{nd}f}" if np.isfinite(x) else " " * (w - 3) + "n/a"


def ft(x):
    return f"{x:+5.1f}" if np.isfinite(x) else "  n/a"


# ---------------------------------------------------------------- 미래 참조 점검
def lookahead_check(rng):
    bad = 0
    tries = 0
    for _ in range(6):
        t0 = int(rng.integers(80, T - 15))
        L2, H2, q2 = L.copy(), H.copy(), C[:, TI["QQQ"]].copy()
        L2[t0 + 1:] = L2[t0 + 1:] * rng.uniform(0.5, 1.5, L2[t0 + 1:].shape)
        H2[t0 + 1:] = H2[t0 + 1:] * rng.uniform(0.5, 1.5, H2[t0 + 1:].shape)
        q2[t0 + 1:] = q2[t0 + 1:] * rng.uniform(0.5, 1.5, q2[t0 + 1:].shape)
        for n in (10, 20, 60):
            a, b = roll(L2, n, "min")[:t0 + 1], LMIN[n][:t0 + 1]
            bad += int(not np.allclose(a, b, equal_nan=True))
            a, b = roll(H2, n, "max")[:t0 + 1], HMAX[n][:t0 + 1]
            bad += int(not np.allclose(a, b, equal_nan=True))
            tries += 2
        ab, vo = regime_arrays(q2)
        bad += int(not np.allclose(ab[:t0 + 2], ABOVE[:t0 + 2], equal_nan=True))
        bad += int(not np.allclose(vo[:t0 + 2], VOL[:t0 + 2], equal_nan=True))
        tries += 2
    return bad, tries


def trade_check(df, fam, N, rng, k=300):
    """표본 거래를 원 배열에서 손으로 다시 계산: 박스는 t-N..t-1만, B 체결가는 [저가, 시가] 안, 매수 봉 = 신호 봉(+1)."""
    bad = 0
    s = df.sample(min(k, len(df)), random_state=int(rng.integers(1e9)))
    for _, r in s.iterrows():
        t, j = int(r.t), int(r.j)
        lm = np.min(L[t - N:t, j])
        hm = np.max(H[t - N:t, j])
        if fam == "A":
            ok = (lm <= L[t, j] <= lm * 1.01) and r.e == t + 1 and np.isclose(r.P, O[t + 1, j])
        elif fam == "B":
            ok = (L[t, j] < lm) and r.e == t and np.isclose(r.P, min(O[t, j], lm)) and L[t, j] - 1e-9 <= r.P <= O[t, j] + 1e-9
        else:
            ok = (C[t, j] > hm) and r.e == t + 1 and np.isclose(r.P, O[t + 1, j])
        bad += int(not ok)
    return bad, len(s)


# ---------------------------------------------------------------- 보고
def main():
    rng = np.random.default_rng(7)
    rules = make_rules()
    TR = {r: trades(r) for r in rules}
    std = {"A": ("A", 20, 0.0, "-", "H5"), "B": ("B", 20, 0.0, "-", "H5"), "C": ("C", 20, 0.0, "-", "H5"),
           "B실적": ("B", 20, 0.0, "earn", "H5"), "Bn": ("Bn", 20, 0.0, "-", "H5")}

    out("박스권 매매 장세 지도 (키 regime, 2026-10-09)  — 스크립트 box_trade/code/regime.py")
    out(f"데이터: 일봉 {DSTR[0]}~{DSTR[-1]} ({T}거래일), 종목 {M}개 | S&P500 {UNI['SP500'].sum()}개, 사용자78 {UNI['U78'].sum()}개")
    out(f"실적 캐시: {len(EARN_COVER)}종목(ETF 제외 {len([t for t in TICK if t not in ETF])}개 중) | 분할 의심 날 {int(_bad.sum())}개 제외")
    out(f"QQQ 20일 변동성 기준(2024-11~2025-12 중앙값) = {VTHR:.1%} 연율")
    out("표준 정의 N=20: A 버팀(저가가 20일 최저가~+1%, 안 깸) → 다음 날 시가 / B 이탈(20일 최저가 지정가, 체결 min(시가, 지정가))"
        " / C 20일 최고가 종가 돌파 → 다음 날 시가. 매도 H5 = 매수 봉+5봉 시가.")
    out("수익 = 단순 수익(수수료 0). 무작위 = 같은 묶음·같은 매수 봉 시가 진입·같은 매도(같은 날 시장 효과 제거). 초과 = 평균 - 무작위.")
    out("초과2 = 평균 - 같은 종목·같은 달 무작위. te = 초과의 날짜 군집 t. 바닥 = 신호일 저가가 다음 5일간 안 깨진 비율(괄호: 같은 날 무작위).")
    out()

    # 1) 월별 표
    for u in ("SP500", "U78"):
        out(f"■ 1. 월별 초과수익 — {u}, 표준 정의 N=20 · H5 (신호 월 기준, 칸: n / 평균 / 초과, %)")
        out("  월      | QQQ월% 위50 고변 | A n   평균  초과  | B n   평균  초과  바닥(무작위) | C n   평균  초과  | B실적 n 평균  초과 | Bn 초과 | 무작위")
        q = C[:, TI["QQQ"]]
        for mi, mo in enumerate(MONTHS):
            if mo < "2024-11":
                continue
            rows = np.where(MI == mi)[0]
            qret = q[rows[-1]] / q[rows[0] - 1] - 1 if rows[0] > 0 else np.nan
            ab = np.nanmean(ABOVE[rows])
            hv = np.nanmean(HIVOL[rows])
            cells = []
            for key in ("A", "B", "C"):
                d = TR[std[key]]
                d = d[d["in_" + u] & (d.mon == mo)]
                if len(d):
                    s = stats(d, u)
                    c = f"{s['n']:>4} {fp(s['avg'],5,1)} {fp(s['exc'],5,1)}"
                    if key == "B":
                        c += f"  {s['h5']:.0%}({s['h5b']:.0%})"
                    cells.append(c)
                else:
                    cells.append("   0    -     -  " + ("          " if key == "B" else ""))
            d = TR[std["B실적"]]
            d = d[d["in_" + u] & (d.mon == mo)]
            s = stats(d, u)
            cells.append(f"{s['n']:>4} {fp(s['avg'],5,1)} {fp(s['exc'],5,1)}" if len(d) else "   0    -     - ")
            d = TR[std["Bn"]]
            d = d[d["in_" + u] & (d.mon == mo)]
            cells.append(f"{fp(stats(d, u)['exc'],5,1)}  " if len(d) else "    -   ")
            rnd = np.nanmean(RAND_DAY["H5"][u][rows])
            abs_ = f"{ab:4.0%}" if np.isfinite(ab) else "   -"
            out(f"  {mo} | {fp(qret,5,1)} {abs_} {hv:4.0%} | " + " | ".join(cells) + f" | {fp(rnd,5,1)}")
        out()

    # 2) 최근 vs 그 전
    out("■ 2. 최근 3·6개월 대 그 전 (N=20 · H5, 칸: n 평균 초과 te 승률 바닥(같은 날 무작위) [B: 당일 = 체결→그날 종가])")
    out("   차이 = 최근3 초과 - 그 전 초과, t = 차이/sqrt(se1²+se2²) (se = 초과/te)")
    pers = [("그전 24-11~26-04-08", P_PRE), ("최근6 26-04-09~", P_R6), ("최근3 26-07-09~", P_R3),
            ("참고 24-11~25-12", P_REF), ("고르기 26-01~06", P_SEL), ("확인 26-07~10-08", P_CONF)]
    for u in ("SP500", "U78"):
        out(f"  [{u}]")
        for key in ("A", "B", "Bn", "C", "B실적"):
            cells = []
            for nm, p in pers:
                dd = sub(TR[std[key]], u, p)
                s = stats(dd, u)
                c = (f"{nm}: n{s['n']:>5} {fp(s['avg'])} 초과{fp(s['exc'])} te{ft(s['te'])} 승{s['win']:.0%} 바닥{s['h5']:.0%}({s['h5b']:.0%})"
                     if s["n"] else f"{nm}: n    0")
                if key in ("B", "B실적") and s["n"]:
                    c += f" 당일{fp(dd.day0.mean())}"
                cells.append(c)
            a = stats(sub(TR[std[key]], u, P_PRE), u)
            b = stats(sub(TR[std[key]], u, P_R3), u)
            se = np.sqrt((a["exc"] / a["te"]) ** 2 + (b["exc"] / b["te"]) ** 2)
            cells.append(f"차이(최근3-그 전) {fp(b['exc'] - a['exc'])} t{ft((b['exc'] - a['exc']) / se)}")
            out(f"   {key:>4} | " + "\n        | ".join(cells))
        out()

    # 3) 장세별
    out("■ 3. 장세별 (QQQ 50일선 위/아래 × QQQ 20일 변동성 높/낮, 신호 전날 기준) — N=20 · H5, 칸: n 평균 초과 te")
    for u in ("SP500", "U78"):
        for pnm, p in (("2년 전체", ("2024-11-01", "2026-10-08")), ("2026년", ("2026-01-02", "2026-10-08")), ("최근3", P_R3)):
            out(f"  [{u} · {pnm}]")
            for ab, hv, nm in ((1, 0, "위·낮음"), (1, 1, "위·높음"), (0, 0, "아래·낮음"), (0, 1, "아래·높음")):
                cells = []
                for key in ("A", "B", "Bn", "C", "B실적"):
                    d = sub(TR[std[key]], u, p)
                    d = d[(d.above == ab) & (d.hivol == hv)]
                    s = stats(d, u)
                    cells.append(f"{key} n{s['n']:>5} {fp(s['avg'])} 초과{fp(s['exc'])} te{ft(s['te'])}")
                ndays = int(((ABOVE == ab) & (HIVOL == hv) & (DSTR >= p[0]) & (DSTR <= p[1])).sum())
                out(f"   {nm:<6}({ndays:>3}일) | " + " | ".join(cells))
        out()

    # 4) 실적 직후 이탈
    out("■ 4. 실적 직후(반응일~+2일) vs 그 밖 — N=20 · H5, S&P500·사용자78 합친 고유 거래가 아니라 묶음별 (칸: n 평균 초과)")
    for u in ("SP500", "U78"):
        out(f"  [{u}] 기간별")
        for key, rr in (("A실적", ("A", 20, 0.0, "earn", "H5")), ("B실적", std["B실적"]), ("C실적", ("C", 20, 0.0, "earn", "H5"))):
            cells = []
            for nm, p in pers:
                s = stats(sub(TR[rr], u, p), u)
                cells.append(f"{nm.split(' ')[0]} n{s['n']:>4} {fp(s['avg'])} 초과{fp(s['exc'])} te{ft(s['te'])}")
            out(f"   {key} | " + " | ".join(cells))
        for key in ("A", "B", "C"):
            d = TR[std[key]]
            cells = []
            for nm, p in pers:
                s = stats(sub(d[~d.ew], u, p), u)
                cells.append(f"{nm.split(' ')[0]} n{s['n']:>4} {fp(s['avg'])} 초과{fp(s['exc'])} te{ft(s['te'])}")
            out(f"   {key}실적밖 | " + " | ".join(cells))
    out("  [S&P500+사용자78 합집합] B실적 월별(무작위 = S&P500 같은 날): 월 n 평균 초과 바닥")
    d = TR[std["B실적"]]
    d = d[d.in_SP500 | d.in_U78].copy()
    for mo, g in d.groupby("mon"):
        s = stats(g.assign(in_SP500=True), "SP500")
        out(f"   {mo}  n{s['n']:>3} {fp(s['avg'])} 초과{fp(s['exc'])} 바닥{s['h5']:.0%}")
    out()

    # 5) 규칙 격자
    out(f"■ 5. 규칙 격자 {len(rules)}개 (박스 N 10·20·60 × A/B/C × 매도 5종 + Bn20(이탈 뒤 다음 날 시가) + B 3% 깊게·실적 직후·QQQ 50일선 아래·고변동 필터, A/C 실적 직후·QQQ 50일선 아래)")
    out("   칸: n 평균 초과 te  (기준 = 같은 날 같은 묶음 무작위). ✓ = 고르기·확인 둘 다 초과 > 0")
    res = []
    for r in rules:
        row = {"rule": rname(r), "r": r}
        for u in ("SP500", "U78"):
            for pn, p in (("REF", P_REF), ("SEL", P_SEL), ("CONF", P_CONF)):
                row[(u, pn)] = stats(sub(TR[r], u, p), u)
        res.append(row)
    both = {u: [x for x in res if x[(u, "SEL")]["exc"] > 0 and x[(u, "CONF")]["exc"] > 0] for u in ("SP500", "U78")}
    both2 = [x for x in both["SP500"] if x in both["U78"]]
    for x in res:
        cells = []
        for u in ("SP500", "U78"):
            for pn in ("SEL", "CONF", "REF"):
                s = x[(u, pn)]
                cells.append(f"{pn[0]} n{s['n']:>5} {fp(s['avg'],5,2)} {fp(s['exc'],5,2)} {ft(s['te'])}")
        mk = ("✓" if x in both["SP500"] else " ") + ("✓" if x in both["U78"] else " ")
        out(f"   {x['rule']:<20}{mk} SP500 " + " ".join(cells[:3]) + " | U78 " + " ".join(cells[3:]))
    out(f"  두 기간(고르기·확인) 모두 무작위 초과 > 0: S&P500 {len(both['SP500'])}/{len(rules)}, 사용자78 {len(both['U78'])}/{len(rules)},"
        f" 둘 다 {len(both2)}/{len(rules)}")
    for fam in ("A", "B", "Bn", "C"):
        fs = [x for x in res if x["r"][0] == fam and x["r"][3] == "-" and x["r"][2] == 0]
        out(f"   기본형 {fam}({len(fs)}개): S&P500 둘 다 + {sum(x in both['SP500'] for x in fs)}, 사용자78 둘 다 + {sum(x in both['U78'] for x in fs)}")
    out()

    # 6) 최선안(고르기 기간 S&P500, n>=30, te 최대)
    cand = [x for x in res if x[("SP500", "SEL")]["n"] >= 30 and np.isfinite(x[("SP500", "SEL")]["te"])]
    best = max(cand, key=lambda x: x[("SP500", "SEL")]["te"])
    out(f"■ 6. 고르기 기간(2026-01~06, S&P500, n>=30) 무작위 초과 te 최대 = {best['rule']}")
    for u in ("SP500", "U78"):
        for pn, p in (("고르기", P_SEL), ("확인", P_CONF), ("참고", P_REF), ("최근3", P_R3)):
            s = stats(sub(TR[best["r"]], u, p), u)
            out(f"   {u:<5} {pn}: n{s['n']:>5} 평균{fp(s['avg'])} 승{s['win']:.0%} 최악{fp(s['worst'],6,1)} 합{s['sum']:+7.2f}"
                f" t{ft(s['t'])} 무작위{fp(s['rnd'])} 초과{fp(s['exc'])} te{ft(s['te'])} 초과2{fp(s['exc2'])}")
    top = sorted(cand, key=lambda x: -x[("SP500", "SEL")]["te"])[:8]
    bf = best["r"]
    out("   같은 박스·필터·매도의 다른 방식(S&P500 | 사용자78), 칸: 고르기 초과(te) → 확인 초과(te) | 참고 초과:")
    for fam in ("A", "B", "Bn", "C"):
        z = next((x for x in res if x["r"] == (fam,) + tuple(bf[1:])), None)
        if z is None:
            continue
        cc = []
        for u in ("SP500", "U78"):
            a, b, c = z[(u, "SEL")], z[(u, "CONF")], z[(u, "REF")]
            cc.append(f"{fp(a['exc'])}({ft(a['te'])})→{fp(b['exc'])}({ft(b['te'])}) | {fp(c['exc'])}")
        out(f"     {z['rule']:<20} " + " || ".join(cc))
    nb = next(x for x in res if x["r"] == (bf[0], bf[1], bf[2], "-", bf[4]))
    out(f"     필터 없는 같은 규칙 {nb['rule']}: S&P500 고르기 {fp(nb[('SP500','SEL')]['exc'])}({ft(nb[('SP500','SEL')]['te'])})"
        f" → 확인 {fp(nb[('SP500','CONF')]['exc'])}({ft(nb[('SP500','CONF')]['te'])})")
    out("   최선안 월별 (칸: n 평균 초과, S&P500 | 사용자78):")
    for mo in MONTHS:
        if mo < "2024-11":
            continue
        cc = []
        for u in ("SP500", "U78"):
            d = TR[bf]
            d = d[d["in_" + u] & (d.mon == mo)]
            s_ = stats(d, u)
            cc.append(f"n{s_['n']:>4} {fp(s_['avg'])} {fp(s_['exc'])}" if s_["n"] else "n   0      -      -")
        out(f"     {mo} " + " | ".join(cc))
    out("   고르기 te 상위 8 (확인은 참고로만): " + "; ".join(
        f"{x['rule']} 고르기te{ft(x[('SP500','SEL')]['te'])}→확인 초과{fp(x[('SP500','CONF')]['exc'])}" for x in top))
    out()

    # 6b) 체결 보수 민감도
    out("■ 6b. B 체결 민감도: 저가가 지정가보다 x% 넘게 뚫린 날만 체결로 칠 때 (칸: n 초과(te)). 고르기에 쓰지 않은 점검.")
    global THRU
    for rule in (("B", 20, 0.0, "-", "H1"), ("B", 20, 0.0, "-", "H5"), ("B", 20, 0.0, "earn", "H5"), bf):
        for th in (0.0, 0.002, 0.005):
            THRU = th
            dd = trades(rule)
            cc = []
            for u in ("SP500", "U78"):
                for pn, p in (("그 전", P_PRE), ("고르기", P_SEL), ("확인", P_CONF), ("최근3", P_R3)):
                    s_ = stats(sub(dd, u, p), u)
                    cc.append(f"{pn} n{s_['n']:>5} {fp(s_['exc'])}({ft(s_['te'])})")
            out(f"   {rname(rule):<18} x={th:.1%} | SP500 " + " ".join(cc[:4]) + " | U78 " + " ".join(cc[4:]))
    THRU = 0.0
    out()

    # 7) 같은 매도끼리 A/B/C (N=20)
    out("■ 7. 같은 매도 규칙끼리 A/B/Bn/C (N=20, S&P500), 칸: 고르기 초과(te) → 확인 초과(te) | 참고 초과")
    for ex in EXITS:
        cells = []
        for fam in ("A", "B", "Bn", "C"):
            x = next(z for z in res if z["r"] == (fam, 20, 0.0, "-", ex))
            a, b, c = x[("SP500", "SEL")], x[("SP500", "CONF")], x[("SP500", "REF")]
            cells.append(f"{fam} {fp(a['exc'])}({ft(a['te'])})→{fp(b['exc'])}({ft(b['te'])}) | {fp(c['exc'])}")
        out(f"   {ex:<5} " + " || ".join(cells))
    out()

    # 8) 나이키
    out("■ 8. NKE 확인 (B20: 10/2 장 전 20일 최저가 지정가)")
    j = TI["NKE"]
    t = int(np.where(DSTR == "2026-10-02")[0][0])
    lm = LMIN[20][t, j]
    P = min(O[t, j], lm)
    out(f"   20일 최저가(9/3~10/1 저가 최저) {lm:.2f}, 10/2 시가 {O[t, j]:.2f} 저가 {L[t, j]:.2f} → 체결 {P:.2f}(갭 아래라 시가),"
        f" 실적창 {bool(EW[t, j])}, 장세 QQQ50 위={ABOVE[t]:.0f} 고변동={HIVOL[t]:.0f}")
    for h in (1, 3, 4):
        if t + h <= T - 1:
            out(f"   {h}봉 뒤 시가({DSTR[t+h]}) {O[t+h, j]:.2f} → {O[t+h, j]/P-1:+.2%}")
    out(f"   10/8 종가 {C[-1, j]:.2f} ({C[-1, j]/P-1:+.2%}), 10/2 이후 최저 {np.nanmin(L[t:, j]):.2f} = 10/2 저가 유지: {np.nanmin(L[t+1:, j]) > L[t, j]}")
    dn = TR[("B", 20, 0.0, "earn", "H3")]
    hit = dn[(dn.j == j)]
    out("   B20[실적]/H3 거래 목록 NKE: " + ", ".join(f"{r.date} {r.ret:+.1%}" for r in hit.itertuples()))
    out()

    # 9) 미래 참조 점검
    b1, n1 = lookahead_check(rng)
    out("■ 9. 미래 참조 점검")
    out(f"   신호일 뒤 저가·고가·QQQ를 무작위로 바꿔도 박스·장세 값이 그대로인지: 불일치 {b1}/{n1}")
    for fam in ("A", "B", "C"):
        b2, n2 = trade_check(TR[(fam, 20, 0.0, "-", "H5")], fam, 20, rng)
        out(f"   {fam}20 표본 {n2}건 손 계산(박스 t-N..t-1, 매수 봉, 체결가 범위): 불일치 {b2}")
    out()
    # 10) 요약(위 표의 값)
    out("■ 10. 요약 (N=20 · H5, 무작위 초과 %/거래, 괄호 te)")
    for u in ("SP500", "U78"):
        cc = []
        for key in ("A", "B", "C", "B실적"):
            a_ = stats(sub(TR[std[key]], u, P_PRE), u)
            b_ = stats(sub(TR[std[key]], u, P_R3), u)
            cc.append(f"{key} 그 전 {fp(a_['exc'],5,2)}({ft(a_['te'])}) → 최근3 {fp(b_['exc'],5,2)}({ft(b_['te'])})")
        out(f"   {u:<5} " + " | ".join(cc))
    sb = best[("SP500", "SEL")]
    cb = best[("SP500", "CONF")]
    out(f"   고르기로 고른 최선안 {best['rule']}: S&P500 고르기 {fp(sb['exc'])}(te{ft(sb['te'])}) → 확인 {fp(cb['exc'])}(te{ft(cb['te'])}),"
        f" 0.2% 보수 체결이면 약 절반(6b). 시험 {len(rules)}개 중 두 기간 모두 무작위 초과>0: S&P500 {len(both['SP500'])}, 사용자78 {len(both['U78'])}, 둘 다 {len(both2)}.")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    return res, best, both, both2, TR


if __name__ == "__main__":
    main()
