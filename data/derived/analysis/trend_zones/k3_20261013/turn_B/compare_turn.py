"""추세 5단계 — 꺾임에 더 빨리 반응하는 가운데 판정 10개 비교 (turn_B, 연구용). 저장소 파일은 읽기·import만 한다.

사전 등록 후보(모두 RSI14 파인식 ≥70 → +2, ≤30 → -2가 먼저)
  B0 T3(20) ±0.5 | B1 T3(15) ±0.5 | B2 T3(10) ±0.5
  B3 B0 + 최근 다리 덮어쓰기 m=1.0 | B4 B0 + m=1.5 | B5 B1 + m=1.5 (u = sd20·√10)
  B6 2단계 스윙(1.5·sd20·√10 되돌림) | B7 옛 T0(20) ±0.5 | B8 EMA10 거리/sd20 ±0.655 | B9 (T3(10)+T3(20))/2 ±0.5
정답: 로그 종가 지그재그(θ = 0.05, 0.10 로그, 1.0·sd20·√10, 2.0·sd20·√10), 꺾임점 사이 날에 스윙 방향.

실행: PYTHONIOENCODING=utf-8 py compare_turn.py   결과: 같은 폴더 results.json
"""
import json
import os
import sys

sys.dont_write_bytecode = True  # 저장소 code/ 폴더에 .pyc를 만들지 않는다

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.abspath(os.path.join(HERE, "..", "panel_1d_20261012.parquet"))
ZCODE = "C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/trend_zones/code"
sys.path.insert(0, ZCODE)
import zones as ZM  # noqa: E402  (Ind, Fwd, period_rows, evaluate)

MIN_BARS = 130
P_ALL = ("2024-01-02", "2026-10-12")
P_LY = ("2025-10-13", "2026-10-12")
P_26 = ("2026-01-02", "2026-10-12")
PER = {"2024": ("2024-01-02", "2024-12-31"), "2025": ("2025-01-02", "2025-12-31"),
       "2026H1": ("2026-01-02", "2026-05-29"), "2026H2": ("2026-06-01", "2026-10-12"),
       "all": P_ALL}
THETAS = ("Z5", "Z10", "ZV1", "ZV2")
IDS = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]


def kmul(N):
    return np.sqrt((N - 1) * (2 * N - 1) / (6 * N))


# ------------------------------------------------------------------ 자료
def load():
    x = pd.read_parquet(PANEL)
    x["d"] = pd.to_datetime(x["ts"]).dt.strftime("%Y-%m-%d")
    etf = sorted(x.loc[x.is_etf, "sym"].unique())
    f = {k: x.pivot_table(index="d", columns="sym", values=k).sort_index() for k in ("Open", "High", "Low", "Close")}
    return f, etf


def rsi_pine(c, n=14):
    """파인 ta.rsi와 같게: RMA, 첫 14개 변화의 단순평균으로 시작 (impl_B와 같은 식)."""
    out = np.full(len(c), np.nan)
    idx = np.where(np.isfinite(c))[0]
    if len(idx) < n + 1:
        return out
    s = idx[0]
    d = np.diff(c[s:])
    u, dn = np.maximum(d, 0), np.maximum(-d, 0)
    au, ad = u[:n].mean(), dn[:n].mean()

    def val(a, b):
        if b == 0:
            return 100.0 if a > 0 else np.nan
        return 100 - 100 / (1 + a / b)

    out[s + n] = val(au, ad)
    for k in range(n, len(d)):
        au = (au * (n - 1) + u[k]) / n
        ad = (ad * (n - 1) + dn[k]) / n
        out[s + k + 1] = val(au, ad)
    return out


# ------------------------------------------------------------------ 지표 값
def features(C):
    lC = np.log(C)
    lr = lC.diff()
    sd = {N: lr.rolling(N, min_periods=N).std(ddof=1) for N in (10, 15, 20)}
    T3 = {N: (lC - lC.rolling(N, min_periods=N).mean()) / (sd[N] * kmul(N)) for N in (10, 15, 20)}
    rsi = pd.DataFrame({s: rsi_pine(C[s].to_numpy()) for s in C.columns}, index=C.index)
    T0 = (lC - lC.shift(20)) / (sd[20] * np.sqrt(20))
    ema10 = C.ewm(span=10, adjust=False).mean()
    T5 = (lC - np.log(ema10)) / sd[20]
    T9 = (T3[10] + T3[20]) / 2
    mx10 = C.rolling(10, min_periods=10).max()
    mn10 = C.rolling(10, min_periods=10).min()
    dd = np.log(C / mx10)
    ru = np.log(C / mn10)
    u10 = sd[20] * np.sqrt(10)
    nb = C.notna().cumsum()
    V = (nb >= MIN_BARS) & C.notna()
    for a in (rsi, sd[10], sd[15], sd[20], T3[10], T3[15], T3[20], T0, T5, dd, ru):
        V &= np.isfinite(a)
    V &= (sd[10] > 0) & (sd[15] > 0) & (sd[20] > 0)
    # 꾸준한 추세: 60개 로그 종가(t-59..t)의 시간 회귀 R², 60거래일 수익 C_t/C_{t-60}-1
    tt = pd.DataFrame(np.tile(np.arange(len(C), dtype=float)[:, None], (1, C.shape[1])), index=C.index,
                      columns=C.columns)
    r2 = lC.rolling(60, min_periods=60).corr(tt) ** 2
    ret60 = C / C.shift(60) - 1
    return dict(lC=lC, lr=lr, sd=sd, T3=T3, rsi=rsi, T0=T0, T5=T5, T9=T9, dd=dd, ru=ru, u10=u10, V=V,
                mx10=mx10, mn10=mn10, r2=r2, ret60=ret60)


# ------------------------------------------------------------------ 단계
def five(F, T, tau):
    r, t = F["rsi"].to_numpy(), T.to_numpy()
    with np.errstate(invalid="ignore"):
        z = np.select([r >= 70, r <= 30, t >= tau, t <= -tau], [2, -2, 1, -1], 0)
    return np.where(F["V"].to_numpy(), z, -9).astype(np.int8)


def leg_override(F, Zb, m):
    """RSI 극단이 아니면 최근 10일 다리(dd = ln C/max10, ru = ln C/min10, u = sd20·√10)로 덮어쓴다."""
    dd, ru, u = F["dd"].to_numpy(), F["ru"].to_numpy(), F["u10"].to_numpy()
    with np.errstate(invalid="ignore"):
        cdn = dd <= -m * u
        cup = ru >= m * u
    z = Zb.astype(np.int16).copy()
    nonext = (Zb != -9) & (np.abs(Zb) != 2)
    both = nonext & cdn & cup
    z = np.where(nonext & cdn & ~cup, -1, z)
    z = np.where(nonext & cup & ~cdn, 1, z)
    with np.errstate(invalid="ignore"):
        big = np.where(-dd > ru, -1, np.where(ru > -dd, 1, z))  # 같으면 기본 단계 유지(사실상 없음)
    z = np.where(both, big, z)
    return z.astype(np.int8)


def swing2(F, C):
    """B6: 회색 없는 2단계 스윙. 반환: 단계 배열, ln(C/ext)/(sd20·√10) 배열."""
    Cn = C.to_numpy()
    V = F["V"].to_numpy()
    sd = F["sd"][20].to_numpy()
    T = F["T3"][20].to_numpy()
    r = F["rsi"].to_numpy()
    nD, nS = Cn.shape
    Z = np.full((nD, nS), -9, dtype=np.int8)
    R = np.full((nD, nS), np.nan)
    for j in range(nS):
        s, ext = 0, np.nan
        for i in range(nD):
            if not V[i, j]:
                if s != 0:  # 유효 구간 뒤 끊김은 없지만 안전하게 재시작
                    s = 0
                continue
            c = Cn[i, j]
            k = 1.5 * sd[i, j] * np.sqrt(10)
            if s == 0:
                s = 1 if T[i, j] >= 0 else -1
                s = 1 if T[i, j] == 0 else s
                ext = c
            elif s == 1:
                ext = max(ext, c)
                if np.log(c / ext) <= -k:
                    s, ext = -1, c
            else:
                ext = min(ext, c)
                if np.log(c / ext) >= k:
                    s, ext = 1, c
            R[i, j] = np.log(c / ext) / (sd[i, j] * np.sqrt(10))
            Z[i, j] = 2 if r[i, j] >= 70 else (-2 if r[i, j] <= 30 else s)
    return Z, R


def build(F, C):
    Z0 = five(F, F["T3"][20], 0.5)
    Z1 = five(F, F["T3"][15], 0.5)
    Z6, R6 = swing2(F, C)
    Zs = {
        "B0": Z0,
        "B1": Z1,
        "B2": five(F, F["T3"][10], 0.5),
        "B3": leg_override(F, Z0, 1.0),
        "B4": leg_override(F, Z0, 1.5),
        "B5": leg_override(F, Z1, 1.5),
        "B6": Z6,
        "B7": five(F, F["T0"], 0.5),
        "B8": five(F, F["T5"], 0.655),
        "B9": five(F, F["T9"], 0.5),
    }
    Tval = {"B0": F["T3"][20], "B1": F["T3"][15], "B2": F["T3"][10], "B3": F["T3"][20], "B4": F["T3"][20],
            "B5": F["T3"][15], "B6": pd.DataFrame(R6, index=C.index, columns=C.columns), "B7": F["T0"],
            "B8": F["T5"], "B9": F["T9"]}
    return Zs, Tval


# ------------------------------------------------------------------ 정답: 로그 종가 지그재그
def zigzag(y, th, th_up=None):
    """y 로그 종가, th[e] = 극값 e에서 쓰는 되돌림 문턱(로그, 고점에서 내릴 때), th_up = 저점에서 오를 때(없으면 th).
    반환: 꺾임점 [(index, +1 고점 / -1 저점)]."""
    if th_up is None:
        th_up = th
    ok = np.isfinite(y) & np.isfinite(th) & np.isfinite(th_up)
    if not ok.any():
        return []
    s = int(np.argmax(ok))
    piv = []
    hi = lo = s
    d = 0
    ext = s
    for t in range(s + 1, len(y)):
        if not np.isfinite(y[t]):
            continue
        if d == 0:
            if y[t] > y[hi]:
                hi = t
            if y[t] < y[lo]:
                lo = t
            dn_tr = y[t] - y[hi] <= -th[hi]
            up_tr = y[t] - y[lo] >= th_up[lo]
            if dn_tr and up_tr:  # 사실상 없음: 먼저 생긴 극값을 꺾임점으로
                dn_tr = hi < lo
                up_tr = not dn_tr
            if dn_tr:
                piv.append((hi, 1))
                d, ext = -1, t
            elif up_tr:
                piv.append((lo, -1))
                d, ext = 1, t
        elif d == 1:
            if y[t] > y[ext]:
                ext = t
            elif y[t] - y[ext] <= -th[ext]:
                piv.append((ext, 1))
                d, ext = -1, t
        else:
            if y[t] < y[ext]:
                ext = t
            elif y[t] - y[ext] >= th_up[ext]:
                piv.append((ext, -1))
                d, ext = 1, t
    return piv


def labels(F, pct=False):
    """pct=True: Z5·Z10을 백분율로(고점 대비 -θ = ln(1-θ), 저점 대비 +θ = ln(1+θ)). ZV는 그대로."""
    lC = F["lC"].to_numpy()
    sd = F["sd"][20].to_numpy()
    nD, nS = lC.shape
    L, SW = {}, {}
    for th_name in THETAS:
        lab = np.zeros((nD, nS), dtype=np.int8)
        sw = []
        for j in range(nS):
            y = lC[:, j]
            thu = None
            if th_name in ("Z5", "Z10"):
                q = 0.05 if th_name == "Z5" else 0.10
                th = np.full(nD, -np.log(1 - q) if pct else q)
                thu = np.full(nD, np.log(1 + q)) if pct else None
            elif th_name == "ZV1":
                th = 1.0 * sd[:, j] * np.sqrt(10)
            else:
                th = 2.0 * sd[:, j] * np.sqrt(10)
            piv = zigzag(y, th, thu)
            for (a, _), (b, tb) in zip(piv[:-1], piv[1:]):
                dirn = 1 if tb == 1 else -1
                lab[a + 1:b + 1, j] = dirn
                sw.append((j, a, b, dirn))
        L[th_name] = lab
        SW[th_name] = sw
    return L, SW


# ------------------------------------------------------------------ 평가
def pmask(dates, p):
    a, b = p
    return np.array([a <= d <= b for d in dates])


def agree_opp(Z, L, colmask, rm):
    z = Z[rm][:, colmask]
    lab = L[rm][:, colmask]
    sel = (z != -9) & (lab != 0)
    sz = np.sign(z[sel].astype(np.int16))
    ll = lab[sel]
    n = int(sel.sum())
    if not n:
        return {"agree": float("nan"), "opp": float("nan"), "n": 0}
    return {"agree": float(np.mean(sz == ll)), "opp": float(np.mean(sz == -ll)), "n": n}


def flips(Z, colmask, rm):
    z = Z[:, colmask].astype(np.int16)
    zt, zp = z[1:], z[:-1]
    ok = (zt != -9) & (zp != -9) & rm[1:, None]
    ch = ok & (zt != zp)
    vd = ((z != -9) & rm[:, None]).sum()
    return float(ch.sum() / (vd / 252)) if vd else float("nan")


def direct(Z, colmask, rm, gap=3):
    """상방 계열 ↔ 하방 계열 전환 중 직전 반대 계열 마지막 날과 거리가 gap 거래일 이하인 횟수(종목·연)."""
    z = Z[:, colmask].astype(np.int16)
    nD, nS = z.shape
    cnt = 0
    vd = ((z != -9) & rm[:, None]).sum()
    for j in range(nS):
        last_s, last_i = 0, -10 ** 9
        for i in range(nD):
            v = z[i, j]
            if v == -9:
                last_s, last_i = 0, -10 ** 9
                continue
            sg = int(np.sign(v))
            if sg == 0:
                continue
            if last_s != 0 and sg == -last_s and i - last_i <= gap and rm[i]:
                cnt += 1
            last_s, last_i = sg, i
    return float(cnt / (vd / 252)) if vd else float("nan")


def rev_share(Z, F, colmask, rm):
    lr = F["lr"].to_numpy()[:, colmask]
    z = Z[:, colmask].astype(np.int16)
    zt, zp = z[1:], z[:-1]
    ok = (zt != -9) & (zp != -9) & rm[1:, None]
    ch = ok & (zt != zp)
    d = np.sign(zt - zp)
    with np.errstate(invalid="ignore"):
        rv = ch & (d * np.sign(lr[1:]) < 0)
    return float(rv.sum() / ch.sum()) if ch.sum() else float("nan")


def lags(Z, sw, colmask, dates, p):
    a_, b_ = p
    cols = set(np.where(colmask)[0])
    out = []
    for j, a, b, dirn in sw:
        if j not in cols or a + 1 >= len(dates):
            continue
        if not (a_ <= dates[a + 1] <= b_) or Z[a + 1, j] == -9:
            continue
        seg = np.sign(Z[a + 1:b + 1, j].astype(np.int16))
        hit = np.where((seg == dirn) & (Z[a + 1:b + 1, j] != -9))[0]
        L_ = b - a
        out.append((int(hit[0]) + 1 if len(hit) else L_ + 1, bool(len(hit)), dirn, dates[a + 1][:4]))
    return out


def lag_stats(x):
    if not x:
        return {"n": 0}
    v = np.array([t[0] for t in x])
    r = np.array([t[1] for t in x])
    return {"n": int(len(v)), "median": float(np.median(v)), "mean": float(np.mean(v)), "never": float(1 - r.mean())}


def steady(Z, F, L10, colmask, rm, up=True):
    r2 = F["r2"].to_numpy()[:, colmask]
    rt = F["ret60"].to_numpy()[:, colmask]
    z = Z[:, colmask]
    lab = L10[:, colmask]
    with np.errstate(invalid="ignore"):
        cond = (r2 >= 0.8) & ((rt >= 0.15) if up else (rt <= -0.15)) & (lab == (1 if up else -1))
    sel = cond & (z != -9) & rm[:, None]
    zz = z[sel]
    if not len(zz):
        return float("nan"), 0
    return float(np.mean((zz > 0) if up else ((zz < 0)))), int(len(zz))


def gray(Z, colmask, rm):
    z = Z[rm][:, colmask]
    v = z != -9
    return float((z[v] == 0).mean())


# ------------------------------------------------------------------ 실행
def main():
    f, etf = load()
    C = f["Close"]
    dates = list(C.index)
    syms = list(C.columns)
    stock = np.array([s not in etf for s in syms])
    etfm = ~stock
    F = features(C)

    # --- 재현 확인
    jx = syms.index("SOXX")
    i12 = dates.index("2026-10-12")
    repro = {f"T3({N})": float(F["T3"][N].iat[i12, jx]) for N in (20, 15, 10)}
    repro.update({"close": float(C.iat[i12, jx]), "max10": float(F["mx10"].iat[i12, jx]),
                  "dd_pct": float(C.iat[i12, jx] / F["mx10"].iat[i12, jx] - 1), "rsi": float(F["rsi"].iat[i12, jx])})
    print("재현 SOXX 10/12", {k: round(v, 4) for k, v in repro.items()})

    Zs, Tval = build(F, C)
    L, SW = labels(F)
    rms = {k: pmask(dates, v) for k, v in PER.items()}
    rm_ly = pmask(dates, P_LY)
    rm_26 = pmask(dates, P_26)

    # 정답 기본 통계
    gt = {}
    for th in THETAS:
        lab = L[th]
        st = {}
        for pk, rm in rms.items():
            l_ = lab[rm][:, stock]
            v = F["V"].to_numpy()[rm][:, stock]
            st[pk] = {"labeled_share": float((l_[v] != 0).mean()), "up_share": float((l_[v] == 1).sum() / max((l_[v] != 0).sum(), 1))}
        nsw = sum(1 for s in SW[th] if stock[s[0]] and P_ALL[0] <= dates[s[1] + 1] <= P_ALL[1])
        st["n_swings_stock_all"] = nsw
        gt[th] = st

    # 예측(보조): zones.py 엔진
    Os, Hs, Ls, Cs = (f[k].loc[:, stock] for k in ("Open", "High", "Low", "Close"))
    I = ZM.Ind(Os, Hs, Ls, Cs)
    assert list(I.dates) == dates and list(I.syms) == [s for s in syms if s not in etf]
    Fw = ZM.Fwd(I, 10)
    prow = {p: ZM.period_rows(I, p, 10) for p in ("앞", "뒤", "2025")}

    def case_rows(sym, a, b):
        j = syms.index(sym)
        rows = []
        for i, d in enumerate(dates):
            if not (a <= d <= b):
                continue
            u = float(F["u10"].iat[i, j])
            rows.append({"date": d, "close": round(float(C.iat[i, j]), 4), "rsi": round(float(F["rsi"].iat[i, j]), 2),
                         "T3_20": round(float(F["T3"][20].iat[i, j]), 4), "T3_15": round(float(F["T3"][15].iat[i, j]), 4),
                         "T3_10": round(float(F["T3"][10].iat[i, j]), 4), "T0": round(float(F["T0"].iat[i, j]), 4),
                         "T_ema10": round(float(F["T5"].iat[i, j]), 4), "T_mix": round(float(F["T9"].iat[i, j]), 4),
                         "dd_over_u": round(float(F["dd"].iat[i, j]) / u, 4), "ru_over_u": round(float(F["ru"].iat[i, j]) / u, 4),
                         "B6_dist": round(float(Tval["B6"].iat[i, j]), 4),
                         "zz_Z10": int(L["Z10"][i, j]), "zz_Z5": int(L["Z5"][i, j]),
                         "stage": {n: int(Zs[n][i, j]) for n in IDS}})
        return rows

    cases = {"SOXX_0901_1012": case_rows("SOXX", "2026-09-01", "2026-10-12"),
             "EWY_0908_1012": case_rows("EWY", "2026-09-08", "2026-10-12"),
             "SPY_0720_0731": case_rows("SPY", "2026-07-20", "2026-07-31"),
             "MU_0622_0731": case_rows("MU", "2026-06-22", "2026-07-31")}

    out = {}
    for n in IDS:
        Z = Zs[n]
        res = {}
        ao = {th: {pk: agree_opp(Z, L[th], stock, rm) for pk, rm in rms.items()} for th in THETAS}
        res["agree_opp"] = ao
        cells26 = [ao[th][h] for th in THETAS for h in ("2026H1", "2026H2")]
        res["score"] = float(np.mean([c["agree"] - 2 * c["opp"] for c in cells26]))
        res["opp26_mean"] = float(np.mean([c["opp"] for c in cells26]))
        res["agree26_mean"] = float(np.mean([c["agree"] for c in cells26]))
        res["score_2026H1"] = float(np.mean([ao[th]["2026H1"]["agree"] - 2 * ao[th]["2026H1"]["opp"] for th in THETAS]))
        res["score_2026H2"] = float(np.mean([ao[th]["2026H2"]["agree"] - 2 * ao[th]["2026H2"]["opp"] for th in THETAS]))
        for y in ("2024", "2025"):
            res[f"score_{y}"] = float(np.mean([ao[th][y]["agree"] - 2 * ao[th][y]["opp"] for th in THETAS]))
        res["gray_2026"] = gray(Z, stock, rm_26)
        res["gray_all"] = gray(Z, stock, rms["all"])
        lz = lags(Z, SW["Z10"], stock, dates, P_ALL)
        res["lag_Z10"] = {"all": lag_stats(lz), "down_swings": lag_stats([t for t in lz if t[2] == -1]),
                          "up_swings": lag_stats([t for t in lz if t[2] == 1]),
                          "2026": lag_stats([t for t in lz if t[3] == "2026"]),
                          "2026_down": lag_stats([t for t in lz if t[3] == "2026" and t[2] == -1])}
        res["lag_ZV1"] = lag_stats(lags(Z, SW["ZV1"], stock, dates, P_ALL))
        res["flips_ly"] = flips(Z, stock, rm_ly)
        res["flips_all"] = flips(Z, stock, rms["all"])
        res["direct_ly"] = direct(Z, stock, rm_ly)
        res["direct_all"] = direct(Z, stock, rms["all"])
        su, nsu = steady(Z, F, L["Z10"], stock, rms["all"], True)
        sdn, nsd = steady(Z, F, L["Z10"], stock, rms["all"], False)
        res["steady_up"], res["steady_up_n"], res["steady_dn"], res["steady_dn_n"] = su, nsu, sdn, nsd
        res["rev"] = rev_share(Z, F, stock, rms["all"])
        Zf = Z[:, stock]
        m5 = {}
        for p in ("앞", "뒤", "2025"):
            e = ZM.evaluate(Zf, Fw, prow[p], full=True)
            m5[p] = {"S": e["S"] * 100, "terms_pp": {k: v * 100 for k, v in e["terms"].items()},
                     "t": e["t"], "share": {str(k): v for k, v in e["share"].items()}}
        res["pred"] = m5
        # ETF
        aoe = {th: {h: agree_opp(Z, L[th], etfm, rms[h]) for h in ("2026H1", "2026H2")} for th in THETAS}
        res["etf"] = {"score_2026": float(np.mean([aoe[th][h]["agree"] - 2 * aoe[th][h]["opp"]
                                                   for th in THETAS for h in ("2026H1", "2026H2")])),
                      "opp26_mean": float(np.mean([aoe[th][h]["opp"] for th in THETAS for h in ("2026H1", "2026H2")])),
                      "flips_ly": flips(Z, etfm, rm_ly), "gray_2026": gray(Z, etfm, rm_26),
                      "agree_opp_2026": aoe,
                      "flips_ly_by_etf": {s: flips(Z, np.array([x == s for x in syms]), rm_ly) for s in etf}}
        # 사례
        def at(sym, d):
            j, i = syms.index(sym), dates.index(d)
            return int(Z[i, j]), float(Tval[n].iat[i, j])
        res["soxx_level"], res["soxx_T"] = at("SOXX", "2026-10-12")
        res["ewy_level"], res["ewy_T"] = at("EWY", "2026-10-12")
        spy = [r for r in cases["SPY_0720_0731"] if "2026-07-23" <= r["date"] <= "2026-07-28"]
        res["spy_jul_down_days"] = sum(1 for r in spy if r["stage"][n] < 0 and r["stage"][n] != -9)
        res["spy_jul_days"] = len(spy)
        mu = [r for r in cases["MU_0622_0731"] if "2026-06-26" <= r["date"] <= "2026-07-28"]
        res["mu_jul_down_days"] = sum(1 for r in mu if -9 < r["stage"][n] < 0)
        res["mu_jul_up_days"] = sum(1 for r in mu if r["stage"][n] > 0)
        res["mu_jul_gray_days"] = sum(1 for r in mu if r["stage"][n] == 0)
        res["mu_jul_days"] = len(mu)
        soxx = cases["SOXX_0901_1012"]
        res["soxx_first_down_after_1005"] = next((r["date"] for r in soxx if r["date"] > "2026-10-05"
                                                  and -9 < r["stage"][n] < 0), "none")
        ewy = cases["EWY_0908_1012"]
        res["ewy_first_down_after_1002"] = next((r["date"] for r in ewy if r["date"] > "2026-10-02"
                                                 and -9 < r["stage"][n] < 0), "none")
        out[n] = res
        a26 = {f"{th}{h[-2:]}": round(ao[th][h]["agree"], 3) for th in THETAS for h in ("2026H1", "2026H2")}
        print(f"{n} score {res['score']:+.4f} opp26 {res['opp26_mean']:.4f} s24 {res['score_2024']:+.3f} "
              f"s25 {res['score_2025']:+.3f} gray26 {res['gray_2026']:.3f} lagZ10 {res['lag_Z10']['all']['median']} "
              f"never {res['lag_Z10']['all']['never']:.3f} flips {res['flips_ly']:.1f} direct {res['direct_ly']:.2f} "
              f"steady {res['steady_up']:.3f}/{res['steady_dn']:.3f} rev {res['rev']:.4f} "
              f"S {m5['앞']['S']:+.2f}/{m5['뒤']['S']:+.2f} T1 {m5['앞']['terms_pp']['T1 상방-하방']:+.2f}/"
              f"{m5['뒤']['terms_pp']['T1 상방-하방']:+.2f} SOXX {res['soxx_level']} ({res['soxx_T']:+.3f}) "
              f"EWY {res['ewy_level']} SPY↓{res['spy_jul_down_days']} MU↓{res['mu_jul_down_days']}↑{res['mu_jul_up_days']} "
              f"ETF {res['etf']['score_2026']:+.3f}/{res['etf']['flips_ly']:.1f}")
        print("   agree26", a26)

    # --- 견고성(보조): 백분율 지그재그, 종목별·월별 B0 대비
    Lp, _ = labels(F, pct=True)
    rob = {}
    months = sorted({d[:7] for d in dates if "2026-01" <= d[:7] <= "2026-10"})
    rm_m = {m: np.array([d[:7] == m for d in dates]) for m in months}
    halves = ("2026H1", "2026H2")

    def cell_score(Z, Ld, cm, rm):
        a = agree_opp(Z, Ld, cm, rm)
        return a["agree"] - 2 * a["opp"], a["n"]

    def sym_scores(Z):
        vals = []
        for j in np.where(stock)[0]:
            cm = np.zeros(len(syms), bool)
            cm[j] = True
            cs = [cell_score(Z, L[th], cm, rms[h]) for th in THETAS for h in halves]
            vals.append(np.mean([c[0] for c in cs]) if all(c[1] >= 20 for c in cs) else np.nan)
        return np.array(vals)

    base_sym = sym_scores(Zs["B0"])
    base_m = {m: np.mean([cell_score(Zs["B0"], L[th], stock, rm_m[m])[0] for th in THETAS]) for m in months}
    for n in IDS:
        Z = Zs[n]
        sp = float(np.mean([cell_score(Z, Lp[th], stock, rms[h])[0] for th in THETAS for h in halves]))
        ss = sym_scores(Z)
        ok = np.isfinite(ss) & np.isfinite(base_sym)
        dm = {m: float(np.mean([cell_score(Z, L[th], stock, rm_m[m])[0] for th in THETAS]) - base_m[m]) for m in months}
        rob[n] = {"score_pct_zigzag": sp,
                  "sym_n": int(ok.sum()), "sym_share_better_than_B0": float(np.mean(ss[ok] > base_sym[ok])),
                  "sym_share_worse_than_B0": float(np.mean(ss[ok] < base_sym[ok])),
                  "sym_mean_diff_vs_B0": float(np.mean(ss[ok] - base_sym[ok])),
                  "month_diff_vs_B0": dm, "months_better": int(sum(v > 0 for v in dm.values())),
                  "months_worse": int(sum(v < 0 for v in dm.values()))}
        print("견고성", n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rob[n].items() if k != "month_diff_vs_B0"})

    res_all = {"repro_SOXX_1012": repro, "robustness": rob, "n_stocks": int(stock.sum()), "etfs": etf, "ground_truth": gt,
               "spec_choices": {
                   "zigzag": "로그 종가에서 진행 중 극값 대비 반대로 θ(로그) 이상 움직이면 극값을 꺾임점으로 확정. Z5=0.05, Z10=0.10 로그. "
                             "ZV: θ = k·sd20(극값 날, ddof=1)·√10. 시작은 sd20이 생긴 첫날, 초기에는 최고·최저를 함께 추적. 같은 값이면 앞선 날을 극값으로.",
                   "agree": "150종목 종목·일 합산(pooled). 분모 = 정답 라벨이 있고 단계가 유효한 날.",
                   "lag": "Z10 확정 스윙, p_i+1이 2024-01-02~2026-10-12 안. lag = 첫 일치일 - p_i (다음날=1), 미도달 = 스윙 길이+1",
                   "flips/direct": "flips, direct 모두 2025-10-13~2026-10-12. direct = 부호 계열이 바뀔 때 직전 반대 계열 마지막 날과 거리 ≤3거래일. 2024~2026 값은 *_all",
                   "steady": "R² = 60개 로그 종가(t-59..t)의 시간 회귀, 60일 수익 = C_t/C_{t-60}-1, 2024-01-02~2026-10-12",
                   "B6": "유효 첫날 s=sign(T3(20)) (0이면 +1), ext=C; 매일 ext 갱신 뒤 문턱 비교(sd20은 당일)",
                   "B3-5 tie": "dd·ru 둘 다 넘고 |dd|=ru이면 기본 단계(사실상 없음)"},
               "candidates": out, "cases": cases}
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res_all, fh, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("saved", os.path.join(HERE, "results.json"))
    print("정답 통계", json.dumps(gt, ensure_ascii=False))


if __name__ == "__main__":
    main()
