"""일봉 B: 박스 바닥 이탈 = 바닥 매수 (box_trade 키 daily_B, 2026-10-09).

사용자 가설(2026-10-09): 지지선(최근 N일 최저가)을 막 이탈하는 점이 손절 물량이 나온 바닥이다(나이키 10/2 실적 갭하락형).
시험 규칙 100개 = 신호 25개 x 매도 4개. 신호·체결·매도 정의는 아래 SIGNALS·EXITS 주석과 결과 파일 머리말에 있다.
미래 참조 금지: 박스 저점(lowN)은 신호일 t 이전 N개 봉의 저가 최소값. 체결은 미리 건 지정가(min(시가, 지정가)) 또는 다음 날 시가.
고르기 2026-01-02~06-30(S&P 500, 무작위 대비 초과의 날짜 군집 t 최대, 30건 이상), 확인 2026-07-01~10-08, 참고 2024-10~2025-12.
사용: py daily_B.py [일봉 폴더]   (기본: 이번 세션 scratchpad/box/daily)"""
import glob, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
DD = sys.argv[1] if len(sys.argv) > 1 else r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/box/daily"
sys.path.insert(0, os.path.join(HERE, "../../dip_reason/code"))
from fetch import user_universe  # noqa: E402

PERIODS = [("참고", "2024-10-09", "2025-12-31"), ("고르기", "2026-01-02", "2026-06-30"), ("확인", "2026-07-01", "2026-10-08")]
NS = (20, 60, 120)
NF = 60                      # 조건(필터) 변형에 쓰는 박스 길이(사전에 고정)
KMAX = 10                    # 최대 보유(거래일)
STOP_BUF = 0.03              # 손절: 종가 < 이탈일 저가 x (1-3%) 이면 다음 날 시가 매도
TGT_MIN = 0.02               # 목표: max(박스 저점, 진입가 x 1.02) 지정가 매도 = 박스 바닥 재진입
DIPB = 30                    # 같은 깊이 대조군의 최대 깊이(%)


# ---------------------------------------------------------------- 데이터
def load_panel():
    cal = pd.read_parquet(os.path.join(DD, "1d_SPY.parquet")).index
    cal = pd.DatetimeIndex(cal).normalize().unique().sort_values()
    syms, arr = [], {k: [] for k in "OHLCV"}
    for f in sorted(glob.glob(os.path.join(DD, "1d_*.parquet"))):
        s = os.path.basename(f)[3:-8]
        d = pd.read_parquet(f)
        d.index = pd.DatetimeIndex(d.index).normalize()
        d = d[~d.index.duplicated(keep="last")].reindex(cal)
        syms.append(s)
        for k, col in zip("OHLCV", ["Open", "High", "Low", "Close", "Volume"]):
            arr[k].append(d[col].values.astype(float))
    P = {k: np.column_stack(v) for k, v in arr.items()}
    for k in "OHL":                                   # 결측·0 가격 정리
        P[k][P[k] <= 0] = np.nan
    return cal, syms, P


def earnings_window(cal, syms):
    """실적 반응일(16:00 이후 발표면 다음 거래일, 그 전이면 당일)부터 +2거래일까지 True. 발표 일정은 사전에 공개된 값."""
    E = pd.read_csv(os.path.join(RES, "earnings_cache.csv"))
    W = np.zeros((len(cal), len(syms)), bool)
    col = {s: j for j, s in enumerate(syms)}
    cd = cal.values.astype("datetime64[D]")
    covered = set()
    for s, g in E.groupby("ticker"):
        if s not in col:
            continue
        covered.add(s)
        for v in g.earnings_dt:
            t = pd.Timestamp(v)
            t = t.tz_convert("America/New_York") if t.tzinfo else t
            D = np.datetime64(t.date(), "D")
            after = t.hour * 60 + t.minute >= 960
            rd = np.searchsorted(cd, D, side="right" if after else "left")
            if rd < len(cal) and (cd[rd] - D) <= np.timedelta64(5, "D"):
                W[rd:rd + 3, col[s]] = True
    return W, covered


def indicators(P):
    O, H, L, C, V = (pd.DataFrame(P[k]) for k in "OHLCV")
    I = {}
    for N in NS:
        I[f"low{N}"] = L.shift(1).rolling(N, min_periods=N).min().values
        I[f"high{N}"] = H.shift(1).rolling(N, min_periods=N).max().values
    I["vavg20"] = V.shift(1).rolling(20, min_periods=20).mean().values
    d = C.diff()
    au = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    ad = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    I["rsi"] = (100 - 100 / (1 + au / ad)).values
    r = C / C.shift(1)
    bad = ((r <= 0.6) | (r >= 1.8)).fillna(False).values          # 분할 의심 날
    I["cbad"] = np.vstack([np.zeros((1, bad.shape[1]), int), np.cumsum(bad, 0)])   # cbad[i] = bad[:i] 합
    return I


# ---------------------------------------------------------------- 신호
def shift_fwd(A, k, fill=np.nan):
    """A[t+k]를 t 위치에."""
    out = np.full_like(A, fill, dtype=A.dtype)
    out[:-k] = A[k:]
    return out


def signal(P, I, EW, fam, N, y=0.0, filt=None):
    """반환: dict(t, s, e, p, bl, lv, intra_entry) — 신호일 t, 종목 s, 진입일 e, 진입가 p, 이탈일 저가 bl, 박스 저점 lv.
    fam L: 신호일 장 전에 lowN x (1-y) 지정가 매수. 저가 < 지정가면 체결, 체결가 min(시가, 지정가). 진입일 = t.
    fam C: 종가 < lowN 이면 다음 날 시가 매수. 진입일 = t+1.
    fam H: 저가 < lowN(이탈일 t), 다음 날 저가 >= 이탈일 저가(버팀) 이면 그다음 날 시가 매수. 진입일 = t+2."""
    O, H, L, C, V = (P[k] for k in "OHLCV")
    lv = I[f"low{N}"]
    with np.errstate(invalid="ignore"):
        if fam == "L":
            lim = lv * (1 - y)
            m = L < lim
            p = np.fmin(O, lim)
            e_off = 0
        elif fam == "C":
            m = C < lv
            p = shift_fwd(O, 1)
            e_off = 1
        else:
            m = (L < lv) & (shift_fwd(L, 1) >= L)
            p = shift_fwd(O, 2)
            e_off = 2
        if filt == "gap":                      # 시가가 박스 저점 아래(갭 이탈) — 시가에 알 수 있음
            m &= O < lv
        elif filt == "earn":                   # 실적 반응일~+2일
            m &= EW
        elif filt == "vol2":                   # 이탈일 거래량 >= 20일 평균 x 2 (종가에 앎: C·H만)
            m &= V >= 2 * I["vavg20"]
        elif filt == "rsi30":                  # L은 전날 종가 RSI, C·H는 이탈일 종가 RSI
            m &= (np.vstack([np.full((1, O.shape[1]), np.nan), I["rsi"][:-1]]) if fam == "L" else I["rsi"]) < 30
        m &= np.isfinite(p) & np.isfinite(lv)
    t, s = np.nonzero(m)
    e = t + e_off
    ok = e < len(O)
    t, s, e = t[ok], s[ok], e[ok]
    cprev = np.vstack([np.full((1, O.shape[1]), np.nan), C[:-1]])
    with np.errstate(invalid="ignore"):
        if fam == "L":                          # 전날 종가 대비 체결 깊이(%)
            dd = 1 - p[t, s] / cprev[t, s]
        else:                                   # 이탈일 종가 등락(%)
            dd = 1 - C[t, s] / cprev[t, s]
    bdip = np.clip(np.nan_to_num(np.round(dd * 100), nan=-1), -1, DIPB).astype(int)
    return dict(t=t, s=s, e=e, p=p[t, s], bl=L[t, s], lv=lv[t, s], N=N, intra_entry=(fam == "L"), fam=fam, bdip=bdip)


def ref_signal(P, I, kind, N):
    """비교용 간이 A·C(정밀판은 다른 담당). 모두 종가 확인 → 다음 날 시가 매수.
    A(버팀): 저가가 박스 저점~+2% 밴드에 닿고 저점은 안 깸, 종가 > 박스 저점.  C(돌파): 종가 > 직전 N일 최고가."""
    O, H, L, C = (P[k] for k in "OHLC")
    lv, hv = I[f"low{N}"], I[f"high{N}"]
    with np.errstate(invalid="ignore"):
        m = ((L <= lv * 1.02) & (L >= lv) & (C > lv)) if kind == "A" else (C > hv)
    p = shift_fwd(O, 1)
    m &= np.isfinite(p) & np.isfinite(lv)
    t, s = np.nonzero(m)
    e = t + 1
    ok = e < len(O)
    t, s, e = t[ok], s[ok], e[ok]
    return dict(t=t, s=s, e=e, p=p[t, s], bl=L[t, s], lv=lv[t, s], N=N, intra_entry=False, fam="ref", bdip=np.full(len(t), -1))


# ---------------------------------------------------------------- 매도
def exits(P, sig, ex):
    """반환: x(청산일), xp(청산가), xin(장중 지정가 청산 여부), done(데이터 안에서 청산 완료).
    X1·X3·X5·X10: 진입일 e에서 k거래일 뒤 시가 매도(한국 22:30 깨어 있을 때 시장가).
    XTS: 목표 지정가 max(박스 저점, 진입가 x 1.02)(장중 체결, 시가가 위면 시가) / 손절: 종가 < 이탈일 저가 x 0.97 → 다음 날 시가 / 최대 10일 뒤 시가."""
    O, H, C = P["O"], P["H"], P["C"]
    T = len(O)
    e, s, p = sig["e"], sig["s"], sig["p"]
    n = len(e)
    if ex.startswith("X") and ex[1:].isdigit():
        k = int(ex[1:])
        x = e + k
        done = x < T
        xp = np.full(n, np.nan)
        xp[done] = O[x[done], s[done]]
        done &= np.isfinite(xp)
        return x, xp, np.zeros(n, bool), done
    tgt = np.fmax(sig["lv"], p * (1 + TGT_MIN))
    stp = sig["bl"] * (1 - STOP_BUF)
    ks = np.arange(KMAX + 1)
    idx = e[:, None] + ks[None, :]
    inb = idx < T
    ic = np.minimum(idx, T - 1)
    Hk, Ok, Ck = H[ic, s[:, None]], O[ic, s[:, None]], C[ic, s[:, None]]
    Hk[~inb] = np.nan; Ok[~inb] = np.nan; Ck[~inb] = np.nan
    with np.errstate(invalid="ignore"):
        th = Hk[:, :KMAX] >= tgt[:, None]
        if sig["intra_entry"]:
            th[:, 0] = False                      # 장중 체결한 날은 목표 체결을 안 셈(순서 불명)
        sh = Ck[:, :KMAX] < stp[:, None]
    BIG = 99
    kt = np.where(th.any(1), th.argmax(1), BIG)
    kss = np.where(sh.any(1), sh.argmax(1), BIG)
    x = np.full(n, -1)
    xp = np.full(n, np.nan)
    xin = np.zeros(n, bool)
    a = (kt <= kss) & (kt < BIG)                  # 목표 먼저(같은 날이면 장중 목표가 종가 손절보다 앞)
    x[a] = e[a] + kt[a]
    og = Ok[a, kt[a]]
    xp[a] = np.where(og >= tgt[a], og, tgt[a])
    xin[a] = og < tgt[a]
    b = (~a) & (kss < BIG)                        # 손절: 다음 날 시가
    x[b] = e[b] + kss[b] + 1
    xp[b] = Ok[b, np.minimum(kss[b] + 1, KMAX)]
    c = ~(a | b)
    x[c] = e[c] + KMAX
    xp[c] = Ok[c, KMAX]
    done = (x < T) & np.isfinite(xp)
    return x, xp, xin, done


def take(sig, x, xp, xin, done, I, T):
    """분할 의심 구간 제외 + 종목당 한 번에 한 포지션(청산 전 새 신호 무시). 미청산(데이터 끝)도 자리를 차지한다."""
    s, t, e = sig["s"], sig["t"], sig["e"]
    N = sig["N"]
    cb = I["cbad"]
    lo = np.maximum(t - N, 0)
    hi = np.minimum(np.where(done, x, T - 1) + 1, T)
    clean = (cb[hi, s] - cb[lo, s]) == 0
    clean &= np.isfinite(sig["p"])
    order = np.lexsort((e, s))
    keep = np.zeros(len(s), bool)
    last_s, last_x, last_open = -1, -1, False
    xx = np.where(done, x, 10 ** 9)
    for i in order:
        if not clean[i]:
            continue
        si = s[i]
        if si != last_s:
            last_s, last_x, last_open = si, -1, False
        if e[i] > last_x or (e[i] == last_x and last_open):
            keep[i] = True
            last_x, last_open = xx[i], not xin[i]
    return keep, clean


# ---------------------------------------------------------------- 기준
def baselines(P, I, mask_u):
    """같은 날 무작위: 진입일 e 시가에 산 종목 묶음 전체 평균(시가→k일 뒤 시가, 장중 청산은 종가). bad 구간 제외."""
    O, C = P["O"], P["C"]
    T = len(O)
    cb = I["cbad"]
    RO, RC = np.full((KMAX + 2, T), np.nan), np.full((KMAX + 2, T), np.nan)
    SO, SC = {}, {}
    rows = np.arange(T)
    for k in range(KMAX + 2):
        if k == 0:
            ro = np.zeros_like(O); rc = C / O - 1
        else:
            ro = np.full_like(O, np.nan); rc = np.full_like(O, np.nan)
            ro[:-k] = O[k:] / O[:-k] - 1
            rc[:-k] = C[k:] / O[:-k] - 1
        hi = np.minimum(rows + k + 1, T)
        badw = (cb[hi] - cb[rows]) > 0
        ro[badw] = np.nan; rc[badw] = np.nan
        with np.errstate(all="ignore"):
            RO[k] = np.nanmean(np.where(mask_u[None, :], ro, np.nan), 1)
            RC[k] = np.nanmean(np.where(mask_u[None, :], rc, np.nan), 1)
        SO[k], SC[k] = ro, rc
    return RO, RC, SO, SC


def dip_control(P, I, mask_u):
    """같은 날·같은 깊이 무작위(박스와 무관한 단순 눌림 매수) 대조군.
    L형: 그날 전날 종가 x (1-b%) 지정가에 체결된 모든 종목(min(시가, 지정가)) → k일 뒤 시가(DL_O)/종가(DL_C). 인덱스 [b, k, t].
    C형: 그날 종가 등락이 -b%(반올림)인 모든 종목을 다음 날 시가 매수 → 진입 k일 뒤 시가(DC_O)/종가(DC_C). 인덱스 [b, k, 신호일 t]."""
    O, L, C = P["O"], P["L"], P["C"]
    T, S = O.shape
    K = KMAX + 2
    cb = I["cbad"]
    rows = np.arange(T)
    cprev = np.vstack([np.full((1, S), np.nan), C[:-1]])
    out = {k + q: np.zeros((DIPB + 1, K, T)) for k in ("LO", "LC", "CO", "CC") for q in ("s", "n")}   # 합·개수(자기 자신 빼기용)
    U = mask_u[None, :]

    def fwd(A, k):
        B = np.full_like(A, np.nan)
        if k == 0:
            return A.copy()
        B[:-k] = A[k:]
        return B
    badw = {k: (cb[np.minimum(rows + k + 2, T)] - cb[rows]) > 0 for k in range(K)}
    with np.errstate(all="ignore"):
        bins_c = np.round((1 - C / cprev) * 100)
        O1 = fwd(O, 1)
        for b in range(DIPB + 1):
            lim = cprev * (1 - b / 100)
            ent = np.fmin(O, lim)
            fill = (L < lim) & U
            mc = (bins_c == b) & U & np.isfinite(O1)
            for k in range(K):
                ro = fwd(O, k) / ent - 1; rc = fwd(C, k) / ent - 1
                m = fill & ~badw[k]
                for nm, rr in (("LO", ro), ("LC", rc)):
                    mm = m & np.isfinite(rr)
                    out[nm + "s"][b, k] = np.where(mm, rr, 0).sum(1); out[nm + "n"][b, k] = mm.sum(1)
                ro2 = fwd(O, k + 1) / O1 - 1; rc2 = fwd(C, k + 1) / O1 - 1
                m2 = mc & ~badw[k]
                for nm, rr in (("CO", ro2), ("CC", rc2)):
                    mm = m2 & np.isfinite(rr)
                    out[nm + "s"][b, k] = np.where(mm, rr, 0).sum(1); out[nm + "n"][b, k] = mm.sum(1)
    return out


def dip_base(P, D, sig, sel, k, ib):
    """같은 날·같은 깊이 대조군 평균(그 종목 자신은 뺀다). sel = 거래 마스크, k = 진입~청산 일수, ib = 장중 청산 여부."""
    n = int(sel.sum())
    if sig["fam"] not in ("L", "C") or n == 0:
        return np.full(n, np.nan)
    O, L, C = P["O"], P["L"], P["C"]
    T = len(O)
    t, s, bd = sig["t"][sel], sig["s"][sel], sig["bdip"][sel]
    bb = np.maximum(bd, 0); kk = np.minimum(k, KMAX + 1)
    pre = "L" if sig["fam"] == "L" else "C"
    ssum = np.where(ib, D[pre + "Cs"][bb, kk, t], D[pre + "Os"][bb, kk, t])
    scnt = np.where(ib, D[pre + "Cn"][bb, kk, t], D[pre + "On"][bb, kk, t])
    with np.errstate(all="ignore"):
        cprev = C[np.maximum(t - 1, 0), s]
        if pre == "L":                                   # 자기 자신이 그 깊이 대조군에 들어갔는지와 그 수익
            lim = cprev * (1 - bb / 100)
            inn = L[t, s] < lim
            ent = np.fmin(O[t, s], lim)
            xi = np.minimum(t + kk, T - 1)
        else:
            inn = np.round((1 - C[t, s] / cprev) * 100) == bb
            ent = O[np.minimum(t + 1, T - 1), s]
            xi = np.minimum(t + 1 + kk, T - 1)
        own = np.where(ib, C[xi, s], O[xi, s]) / ent - 1
        inn &= np.isfinite(own)
        base = (ssum - np.where(inn, own, 0)) / (scnt - inn)
    base[(bd < 0) | ~np.isfinite(base)] = np.nan
    return base


def stock_bh(SO, SC, pmask):
    """같은 종목 매수 후 보유: 기간 안 모든 날 시가에 사서 k일 보유한 평균(종목별). 반환 (시가 청산 [k, s], 종가 청산 [k, s])."""
    with np.errstate(all="ignore"):
        A = np.vstack([np.nanmean(SO[k][pmask], 0) for k in sorted(SO)])
        B = np.vstack([np.nanmean(SC[k][pmask], 0) for k in sorted(SC)])
    return A, B


def cl_t(v, g):
    """평균의 날짜 군집 강건 t."""
    v = np.asarray(v, float)
    n = len(v)
    if n < 3:
        return np.nan
    m = v.mean()
    df = pd.DataFrame({"u": v - m, "g": g}).groupby("g").u.sum()
    G = len(df)
    if G < 2:
        return np.nan
    se = np.sqrt((df.values ** 2).sum() * G / (G - 1)) / n
    return m / se if se > 0 else np.nan


# ---------------------------------------------------------------- 실행
def build(P, I, EW):
    sigs = {}
    for N in NS:
        for y in (0, 2, 4, 6):
            sigs[f"L{y}_N{N}"] = ("L", N, y / 100, None)
    for N in NS:
        sigs[f"C_N{N}"] = ("C", N, 0, None)
    for N in NS:
        sigs[f"H_N{N}"] = ("H", N, 0, None)
    for fam, f in (("L", "gap"), ("L", "earn"), ("C", "vol2"), ("C", "gap"), ("C", "earn"), ("C", "rsi30"), ("H", "earn")):
        sigs[f"{fam}0_N{NF}+{f}" if fam == "L" else f"{fam}_N{NF}+{f}"] = (fam, NF, 0, f)
    return sigs


SIG_DESC = {
    "L": "이탈 당일 지정가: 장 전에 lowN x (1-y%) 매수 지정가, 저가 < 지정가면 체결가 min(시가, 지정가)",
    "C": "종가 이탈: 종가 < lowN 이면 다음 날 시가 매수",
    "H": "버팀 확인: 저가 < lowN 이탈일 다음 날 저가 >= 이탈일 저가면 그다음 날 시가 매수",
}
EXITS = ["X1", "X5", "X10", "XTS"]


def main():
    cal, syms, P = load_panel()
    T, S = P["C"].shape
    I = indicators(P)
    EW, covered = earnings_window(cal, syms)
    sp = set(pd.read_csv(os.path.join(DD, "sp500_list.csv")).iloc[:, 0])
    u78 = set(user_universe()["합집합"])
    UNIV = {"S&P500": np.array([s in sp for s in syms]), "사용자78": np.array([s in u78 for s in syms])}
    pidx = {}
    for name, a, b in PERIODS:
        pidx[name] = (cal >= pd.Timestamp(a)) & (cal <= pd.Timestamp(b))
    BASE, DIP = {}, {}
    for un, mk in UNIV.items():
        RO, RC, SO, SC = baselines(P, I, mk)
        BASE[un] = (RO, RC, {pn: stock_bh(SO, SC, pm) for pn, pm in pidx.items()})
        DIP[un] = dip_control(P, I, mk)
    dates = cal.strftime("%Y-%m-%d").values

    def evaluate(sig, ex, tag):
        x, xp, xin, done = exits(P, sig, ex)
        keep, clean = take(sig, x, xp, xin, done, I, T)
        ret = xp / sig["p"] - 1
        rows = []
        for un, mk in UNIV.items():
            RO, RC, SB = BASE[un]
            inu = mk[sig["s"]]
            for pn, pm in pidx.items():
                sel = keep & done & inu & pm[sig["t"]]
                k = (x[sel] - sig["e"][sel])
                r = ret[sel]
                eidx = sig["e"][sel]
                ib = xin[sel]
                br = np.where(ib, RC[k, eidx], RO[k, eidx])
                ss = sig["s"][sel]
                sbo = np.where(ib, SB[pn][1][k, ss], SB[pn][0][k, ss])
                g = dates[sig["t"][sel]]
                n = int(sel.sum())
                ex_d = r - br
                ex_s = r - sbo
                okd = np.isfinite(ex_d)
                dbase = dip_base(P, DIP[un], sig, sel, k, ib)
                ex_p = r - dbase
                okp = np.isfinite(ex_p)
                rows.append(dict(rule=tag, exit=ex, univ=un, period=pn, n=n,
                                 avg=np.nanmean(r) if n else np.nan, win=np.mean(r > 0) if n else np.nan,
                                 worst=np.nanmin(r) if n else np.nan, sum=np.nansum(r), t=cl_t(r, g) if n else np.nan,
                                 base=np.nanmean(br) if n else np.nan, exc=np.nanmean(ex_d) if n else np.nan,
                                 t_exc=cl_t(ex_d[okd], g[okd]) if okd.sum() > 2 else np.nan,
                                 exc_stock=np.nanmean(ex_s) if n else np.nan, dates=len(set(g)),
                                 hold=np.mean(k) if n else np.nan,
                                 n_dip=int(okp.sum()), exc_dip=np.nanmean(ex_p) if okp.any() else np.nan,
                                 t_dip=cl_t(ex_p[okp], g[okp]) if okp.sum() > 2 else np.nan,
                                 open_pos=int((keep & ~done & inu & pm[sig["t"]]).sum())))
        return rows, dict(x=x, xp=xp, xin=xin, done=done, keep=keep, ret=ret)

    sigdefs = build(P, I, EW)
    allrows, cache = [], {}
    for name, (fam, N, y, f) in sigdefs.items():
        sig = signal(P, I, EW, fam, N, y, f)
        cache[name] = sig
        for ex in EXITS:
            rows, tr = evaluate(sig, ex, name)
            allrows += rows
            cache[(name, ex)] = tr
    R = pd.DataFrame(allrows)
    n_rules = len(sigdefs) * len(EXITS)

    # 비교용 A·C·B(종가) 간이 — 선택에 쓰지 않음
    refrows = []
    refsig = {}
    for N in NS:
        for kind in ("A", "C"):
            sg = ref_signal(P, I, kind, N)
            refsig[(kind, N)] = sg
            for ex in ("X1", "X3", "X5", "X10"):
                rows, _ = evaluate(sg, ex, f"{kind}ref_N{N}")
                refrows += rows
        for ex in ("X3",):
            rows, _ = evaluate(cache[f"C_N{N}"], ex, f"C_N{N}")
            refrows += rows
    RF = pd.DataFrame(refrows)

    # 선택: 고르기 기간 S&P500, 30건·10날짜 이상, 무작위 대비 초과의 날짜 군집 t 최대
    key = ["rule", "exit"]
    sel = R[(R.univ == "S&P500") & (R.period == "고르기") & (R.n >= 30) & (R.dates >= 10)].sort_values("t_exc", ascending=False)
    best = sel.iloc[0]
    bname, bex = best.rule, best.exit
    sel78 = R[(R.univ == "사용자78") & (R.period == "고르기") & (R.n >= 30) & (R.dates >= 10)].sort_values("t_exc", ascending=False)
    bname2, bex2 = sel78.iloc[0].rule, sel78.iloc[0].exit

    def piv(univ):
        a = R[(R.univ == univ) & (R.period == "고르기")].set_index(key)
        b = R[(R.univ == univ) & (R.period == "확인")].set_index(key)
        return a, b
    both = {}
    for un in UNIV:
        a, b = piv(un)
        j = a.join(b, lsuffix="_s", rsuffix="_c")
        both[un] = j
    nboth = {un: int(((j.exc_s > 0) & (j.exc_c > 0) & (j.n_s >= 10) & (j.n_c >= 10)).sum()) for un, j in both.items()}
    nboth_raw = {un: int(((j.avg_s > 0) & (j.avg_c > 0) & (j.n_s >= 10) & (j.n_c >= 10)).sum()) for un, j in both.items()}

    # ------------------------------------------------------------ 출력
    L = []
    w = L.append
    w("일봉 B: 박스 바닥 이탈 = 바닥 매수 (box_trade/daily_B, 2026-10-09)")
    w(f"데이터: yfinance 일봉 {dates[0]}~{dates[-1]} ({T}거래일), 종목 {S}개 중 S&P500 {UNIV['S&P500'].sum()} · 사용자78 {UNIV['사용자78'].sum()}."
      f" 실적 일정 캐시 {len(covered)}종목(S&P500 중 {len(covered & sp)}, 사용자78 중 {len(covered & u78)}).")
    w("주의: S&P500은 현재 구성 종목(생존 편향). 가격은 분할·배당 미조정 → 종가 -40% 이하·+80% 이상 날이 박스~청산 구간에 있으면 그 거래 제외.")
    w("")
    w("[정의]")
    w("  박스 저점 lowN = 신호일 t 직전 N거래일(t-N..t-1) 저가 최소. N = 20·60·120. 조건 변형은 N=60 고정.")
    for k, v in SIG_DESC.items():
        w(f"  {k}: {v}")
    w("  조건: gap = 시가 < lowN(시가에 앎) / earn = 실적 반응일(16시 이후 발표면 다음 거래일)~+2거래일 / vol2 = 이탈일 거래량 >= 직전 20일 평균 x2 / rsi30 = RSI14 < 30 (L은 전날 종가, C·H는 이탈일 종가)")
    w(f"  매도: X1·X5·X10 = 진입일에서 1·5·10거래일 뒤 시가 매도. XTS = 목표 지정가 max(lowN, 진입가 x {1 + TGT_MIN:.2f})(박스 바닥 재진입) / 손절 종가 < 이탈일 저가 x {1 - STOP_BUF:.2f} → 다음 날 시가 / 최대 {KMAX}일 뒤 시가.")
    w("  같은 종목은 한 번에 한 포지션(청산 전 새 신호 무시). 수수료 0. 수익은 단순 수익률(%).")
    w("  기준: '무작위' = 같은 진입일 시가에 같은 묶음 전 종목을 사서 같은 날 청산(장중 목표 청산은 그날 종가)한 평균. '종목 보유' = 같은 종목을 기간 안 아무 날 시가에 사서 같은 일수 보유한 평균.")
    w(f"  시험 규칙 {n_rules}개 = 신호 {len(sigdefs)}개 x 매도 {len(EXITS)}개. 선택: 고르기(2026-01-02~06-30) S&P500, 30건·10날짜 이상 중 무작위 대비 초과의 날짜 군집 t 최대.")
    w("  보조(선택에 안 씀): 최선안의 X3, 간이 A(버팀)·C(돌파) 참고판 X1·X3·X5·X10.")
    w("")

    def fmt(r):
        if r is None or r["n"] == 0:
            return "  건수 0"
        return (f"{r['n']:5d}건({r['dates']:3d}일) 평균 {r['avg']*100:+6.2f}% 승률 {r['win']*100:4.0f}% 최악 {r['worst']*100:+6.1f}% 합 {r['sum']*100:+8.1f}%"
                f" t {r['t']:+5.2f} | 무작위 {r['base']*100:+5.2f}% 초과 {r['exc']*100:+6.2f}% (t {r['t_exc']:+5.2f}) | 같은깊이 눌림 대비 {r['exc_dip']*100:+6.2f}% (t {r['t_dip']:+5.2f}, {r['n_dip']}건)"
                f" | 종목보유 대비 {r['exc_stock']*100:+6.2f}% | 보유 {r['hold']:.1f}일")

    def get(rule, ex, un, pn, D=R):
        q = D[(D.rule == rule) & (D.exit == ex) & (D.univ == un) & (D.period == pn)]
        return q.iloc[0] if len(q) else None

    w("=" * 100)
    w(f"[최선안] {bname} / {bex}  (고르기 기간 S&P500 무작위 대비 초과 t 최대)")
    for un in UNIV:
        for pn, _, _ in PERIODS[::-1]:
            w(f"  {un:6s} {pn}: {fmt(get(bname, bex, un, pn))}")
    w("  보조 X3:")
    srt = cache[bname]
    rows, _ = evaluate(srt, "X3", bname)
    X3 = pd.DataFrame(rows)
    for un in UNIV:
        for pn, _, _ in PERIODS[::-1]:
            w(f"  {un:6s} {pn}: {fmt(get(bname, 'X3', un, pn, X3))}")
    w("")
    w(f"[보조 선택] 사용자78 고르기 기준 최선: {bname2} / {bex2} (같은 기준, 사용자78에서 고름 — 선택 2회째라 선택 편향 더 큼)")
    for un in UNIV:
        for pn, _, _ in PERIODS[::-1]:
            w(f"  {un:6s} {pn}: {fmt(get(bname2, bex2, un, pn))}")
    w("")
    w(f"두 기간(고르기·확인) 모두 무작위 대비 초과 > 0 (양쪽 10건 이상): S&P500 {nboth['S&P500']}/{n_rules}, 사용자78 {nboth['사용자78']}/{n_rules}")
    w(f"두 기간 모두 평균 수익 > 0: S&P500 {nboth_raw['S&P500']}/{n_rules}, 사용자78 {nboth_raw['사용자78']}/{n_rules}")
    for un, j in both.items():
        jd = j[np.isfinite(j.exc_dip_s) & np.isfinite(j.exc_dip_c)]
        w(f"  {un}: 고르기 무작위 초과>0 {int((j.exc_s > 0).sum())}/{n_rules}, 확인 {int((j.exc_c > 0).sum())}/{n_rules} | 같은 깊이 눌림 대비 >0 두 기간 모두 {int(((jd.exc_dip_s > 0) & (jd.exc_dip_c > 0)).sum())}/{len(jd)} (L·C형만), 고르기 {int((jd.exc_dip_s > 0).sum())}, 확인 {int((jd.exc_dip_c > 0).sum())}")
    w("")
    w("[고르기 상위 10 (S&P500) → 확인]")
    for _, r in sel.head(10).iterrows():
        c = get(r.rule, r.exit, "S&P500", "확인")
        w(f"  {r.rule:16s} {r.exit:4s} 고르기 n{r.n:5d} 평균 {r.avg*100:+5.2f}% 초과 {r.exc*100:+5.2f}% t {r.t_exc:+5.2f} → 확인 n{c.n:5d} 평균 {c.avg*100:+5.2f}% 초과 {c.exc*100:+5.2f}% t {c.t_exc:+5.2f}")
    w("")

    # 매도 규칙별 A/B/C 비교 (같은 N, 종가 확인 → 다음 날 시가 매수로 구조를 맞춤)
    w("[같은 매도 규칙끼리 A(버팀)·B(종가 이탈 C_N)·C(돌파) — 모두 종가 확인 다음 날 시가 매수, S&P500 / 사용자78]")
    for N in NS:
        for ex in ("X1", "X3", "X5", "X10"):
            for un in UNIV:
                for pn in ("고르기", "확인"):
                    a = get(f"Aref_N{N}", ex, un, pn, RF)
                    b = get(f"C_N{N}", ex, un, pn, R if ex != "X3" else RF)
                    c = get(f"Cref_N{N}", ex, un, pn, RF)
                    def sh(r):
                        return f"n{r['n']:5d} {r['avg']*100:+5.2f}% 초과{r['exc']*100:+5.2f}%(t{r['t_exc']:+4.1f})" if r is not None and r["n"] else "n0"
                    w(f"  N{N:3d} {ex:3s} {un:6s} {pn}: A {sh(a)} | B {sh(b)} | C {sh(c)}")
    w("")

    # 분기별 추세: 최선안, B 기본형, A·C 참고
    w("[기간별 모습(신호일 기준): 2026-06까지 분기, 2026-07부터 월] 각 칸 = 건수 / 무작위 대비 초과 평균(날짜 군집 t) / 같은 깊이 눌림 대비 초과")
    lab = np.array([d[:7] if d >= "2026-07" else str(pd.Period(d, freq="Q")) for d in dates])

    def bucket(sig, ex, univ):
        x, xp, xin, done = exits(P, sig, ex)
        keep, _ = take(sig, x, xp, xin, done, I, T)
        RO, RC, _ = BASE[univ]
        mk = UNIV[univ][sig["s"]] & keep & done
        e, t = sig["e"][mk], sig["t"][mk]
        k = x[mk] - e
        r = xp[mk] / sig["p"][mk] - 1
        ib = xin[mk]
        br = np.where(ib, RC[k, e], RO[k, e])
        db = dip_base(P, DIP[univ], sig, mk, k, ib)
        df = pd.DataFrame({"q": lab[t], "r": r, "x": r - br, "xd": r - db, "d": dates[t]})
        return {q: (len(g), g.x.mean(), cl_t(g.x.values, g.d.values), g.xd.mean()) for q, g in df.groupby("q")}
    bN = cache[bname]["N"]
    lines = [(f"최선 {bname} {bex}", cache[bname], bex, "S&P500"), (f"최선 {bname} {bex}", cache[bname], bex, "사용자78"),
             (f"보조 {bname2} {bex2}", cache[bname2], bex2, "S&P500"), (f"보조 {bname2} {bex2}", cache[bname2], bex2, "사용자78"),
             (f"B L0_N{NF}+earn X5", cache[f"L0_N{NF}+earn"], "X5", "S&P500"),
             (f"B L0_N{bN} X5", cache[f"L0_N{bN}"], "X5", "S&P500"), (f"B C_N{bN} X5", cache[f"C_N{bN}"], "X5", "S&P500"),
             (f"B H_N{bN} X5", cache[f"H_N{bN}"], "X5", "S&P500"),
             (f"A참고 N{bN} X5", refsig[("A", bN)], "X5", "S&P500"), (f"C참고 N{bN} X5", refsig[("C", bN)], "X5", "S&P500"),
             (f"A참고 N{bN} X5", refsig[("A", bN)], "X5", "사용자78"), (f"C참고 N{bN} X5", refsig[("C", bN)], "X5", "사용자78"),
             (f"B C_N{bN} X5", cache[f"C_N{bN}"], "X5", "사용자78")]
    res = [(nm, un, bucket(sg, ex, un)) for nm, sg, ex, un in lines]
    first = {}
    for i_, l_ in enumerate(lab):
        first.setdefault(l_, i_)
    qs = sorted(set().union(*[set(d) for _, _, d in res]), key=lambda q: first[q])
    for q in qs:
        w(f"  {q}")
        for nm, un, d in res:
            if q in d:
                n_, x_, t_, xd_ = d[q]
                dtxt = f"{xd_*100:+6.2f}%" if np.isfinite(xd_) else "     -"
                w(f"     {nm:22s} {un:6s} {n_:5d}건 초과 {x_*100:+6.2f}% (t {t_:+5.1f}) | 같은깊이 대비 {dtxt}")
    w("")

    # 바닥 유지율: 이탈일 저가가 이후 10거래일 동안 안 깨졌나(박스 이탈 vs 같은 낙폭의 비이탈 날)
    w("[이탈일 저가가 바닥으로 남은 비율 = 이후 10거래일 저가가 모두 그날 저가 이상] 낙폭 = 1 - 그날 저가/전날 종가, 박스 N=60")
    Lm, Cm = P["L"], P["C"]
    with np.errstate(all="ignore"):
        cprev = np.vstack([np.full((1, S), np.nan), Cm[:-1]])
        drop = 1 - Lm / cprev
        fmin = pd.DataFrame(Lm).shift(-1).rolling(10, min_periods=10).min().shift(-9).values
        held = fmin >= Lm
        brk = Lm < I[f"low{NF}"]
        nob = Lm >= I[f"low{NF}"]
    rr = np.arange(T)
    cbad = I["cbad"]
    okw = ((cbad[np.minimum(rr + 12, T)] - cbad[np.maximum(rr - NF, 0)]) == 0)[:, None] if False else         (cbad[np.minimum(rr + 12, T)] - cbad[np.maximum(rr - NF, 0)]) == 0
    valid = np.isfinite(fmin) & okw & np.isfinite(drop)
    edges = [(0, 0.02), (0.02, 0.04), (0.04, 0.07), (0.07, 0.12), (0.12, 1.0)]
    for un in UNIV:
        U = UNIV[un][None, :]
        for pn in ("참고", "고르기", "확인"):
            pm = pidx[pn][:, None]
            cells = []
            for a, b in edges:
                ib_ = (drop >= a) & (drop < b) & valid & U & pm
                g1, g2, g3 = ib_ & brk, ib_ & nob, ib_ & brk & EW
                f = lambda m: f"{held[m].mean()*100:4.0f}%({int(m.sum())})" if m.sum() else "   -"
                cells.append(f"{int(a*100)}~{int(b*100) if b < 1 else ''}%: 이탈 {f(g1)} 비이탈 {f(g2)} 실적이탈 {f(g3)}")
            w(f"  {un:6s} {pn}: " + " | ".join(cells))
    j0 = syms.index("NKE"); t0 = int(np.where(dates == "2026-10-02")[0][0])
    w(f"  NKE 10/2 낙폭 {drop[t0, j0]*100:.1f}% (저가 31.97 / 전날 종가 35.15), 이후 4거래일(10/5~10/8) 저가 최소 {np.nanmin(Lm[t0+1:, j0]):.2f} → 지금까지 유지(10일 미완)")
    w("")

    # 나이키 확인
    w("[나이키 NKE 2026-10-02 (10/1 16:00 실적 → 10/2 갭하락, 저가 31.97)]")
    j = syms.index("NKE")
    ti = int(np.where(dates == "2026-10-02")[0][0])
    for name in sigdefs:
        sig = cache[name]
        hit = np.nonzero((sig["s"] == j) & (sig["t"] == ti))[0]
        if not len(hit):
            w(f"  {name:16s} 신호 없음")
            continue
        i = hit[0]
        parts = []
        for ex in EXITS:
            tr = cache[(name, ex)]
            if not tr["keep"][i]:
                parts.append(f"{ex}: 이전 포지션 보유 중이라 건너뜀")
            elif tr["done"][i]:
                parts.append(f"{ex}: {dates[tr['x'][i]]} {tr['xp'][i]:.2f} {tr['ret'][i]*100:+.1f}%")
            else:
                mtm = P["C"][-1, j] / sig["p"][i] - 1
                parts.append(f"{ex}: 미청산(10/8 종가 {mtm*100:+.1f}%)")
        w(f"  {name:16s} 진입 {dates[sig['e'][i]]} {sig['p'][i]:.2f} (lowN {sig['lv'][i]:.2f}) | " + " | ".join(parts))
    w("")
    w("[전체 결과 표: 규칙 x 매도 x 묶음 x 기간] avg·win·worst·sum·base(무작위)·exc(무작위 대비)·exc_dip(같은 깊이 눌림 대비)·exc_stock은 %, t는 날짜 군집 t, open_pos = 데이터 끝 미청산 건수")
    R2 = R.copy()
    for c in ("avg", "win", "worst", "sum", "base", "exc", "exc_stock", "exc_dip"):
        R2[c] = (R2[c] * 100).round(2)
    R2["t"] = R2.t.round(2); R2["t_exc"] = R2.t_exc.round(2); R2["t_dip"] = R2.t_dip.round(2); R2["hold"] = R2.hold.round(1)
    w(R2.to_string(index=False))
    w("")
    w("[참고 A·C 간이판 + B 종가형 X3]")
    RF2 = RF.copy()
    for c in ("avg", "win", "worst", "sum", "base", "exc", "exc_stock", "exc_dip"):
        RF2[c] = (RF2[c] * 100).round(2)
    RF2["t"] = RF2.t.round(2); RF2["t_exc"] = RF2.t_exc.round(2); RF2["t_dip"] = RF2.t_dip.round(2); RF2["hold"] = RF2.hold.round(1)
    w(RF2.to_string(index=False))
    txt = "\n".join(L)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "daily_B.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n".join(L[:120]))


if __name__ == "__main__":
    main()
