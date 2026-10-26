"""추세 5단계 — '드문 낙폭(연 4번급)이면 하방' 덮어쓰기 후보 6개 비교 (rare_B, 연구용).
저장소 파일은 읽기·import만 한다(zones.py evaluate). turn_B/compare_turn.py의 함수를 재사용한다.

사용자 정의(2026-10-13): "드물다 = 1년에 4번 정도(계절당 1번, 실적발표급)".

정의
  dd20 = ln(C / 최근 20일(오늘 포함) 최고 종가), ru20 = ln(C / 최근 20일 최저 종가)
  에피소드: dd20 <= ln(1-X)로 처음 내려간 날 시작, dd20 > ln(1-X/2)로 회복하면 끝. 연 횟수 = 에피소드 수 / 연수.
  X4(s, 기간): 그 기간 dd20에서 연 횟수 >= 4가 되는 가장 큰 X(0.25%p 간격). 2026 평가 <- 2024-01-02~2025-12-31 이력,
               2025 평가 <- 2023-03-01~2024-12-31 이력. 이력(그 기간 dd20 유효일) 400일 미만 종목 제외.
  rare_day: 평가 기간에 dd20 <= ln(1-X4)인 날.
후보(RSI14 >= 70 -> +2, <= 30 -> -2가 먼저, 덮어쓰기는 극단이 아닐 때만)
  B0 현재 T3(20) ±0.5
  R1 B0 + dd20 <= -m·sd252·√10 -> -1  (m: 2024~25 주식 종목별 덮어쓰기 연 에피소드 중앙 = 4, 0.05 간격)
  R2 R1 + 대칭 상승 ru20 >= m·sd252·√10 -> +1 (둘 다면 큰 쪽)
  R3 B0 + dd20 <= P_p(dd20, 252) -> -1  (p: 같은 방식, 0.5 간격)
  R4 R3 + ru20 >= P_{100-p}(ru20, 252) -> +1
  R5 R1에서 m = 1.25 고정
실행: PYTHONIOENCODING=utf-8 py rare_compare.py   결과: 같은 폴더 results.json
"""
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "turn_B")))
import compare_turn as CT  # noqa: E402  (load, features, five, labels, agree_opp, pmask, ZM)

ZM = CT.ZM
CAL = ("2024-01-02", "2025-12-31")
HIST = {"2026": ("2024-01-02", "2025-12-31"), "2025": ("2023-03-01", "2024-12-31")}
EVAL = {"2026": ("2026-01-02", "2026-10-12"), "2025": ("2025-01-02", "2025-12-31")}
XGRID = np.round(np.arange(0.0025, 0.8000001, 0.0025), 6)
MGRID = np.round(np.arange(0.50, 3.0000001, 0.05), 4)
PGRID = np.round(np.arange(0.5, 30.0000001, 0.5), 4)
IDS = ["B0", "R1", "R2", "R3", "R4", "R5"]
CASES = ["SOXX", "EWY", "SMH", "MU", "SNDK", "NVDA"]
W = 252


# ------------------------------------------------------------------ 에피소드
def count_eps(dd, s_thr, e_thr, inwin):
    """dd (nD,), s_thr·e_thr (nD, G) 로그 문턱(시작 dd<=s, 끝 dd>e), inwin (nD,) bool.
    문턱이나 dd가 없는 날은 건너뛴다(상태 유지). 반환: 창 안에서 시작한 에피소드 수 (G,), 창 안 유효일 (G,)."""
    G = s_thr.shape[1]
    state = np.zeros(G, bool)
    cnt = np.zeros(G, np.int64)
    nd = np.zeros(G, np.int64)
    for t in range(len(dd)):
        d = dd[t]
        if not np.isfinite(d):
            continue
        s, e = s_thr[t], e_thr[t]
        ok = np.isfinite(s) & np.isfinite(e)
        if not ok.any():
            continue
        new = ok & ~state & (d <= s)
        if inwin[t]:
            cnt += new
            nd += ok
        state = (state | new) & ~(ok & (d > e))
    return cnt, nd


def x_to_end(slog):
    """시작 문턱 ln(1-X) -> 끝 문턱 ln(1-X/2)."""
    X = 1 - np.exp(slog)
    return np.log(1 - X / 2)


def x4_for(dd, inwin):
    s = np.broadcast_to(np.log(1 - XGRID), (len(dd), len(XGRID)))
    e = np.broadcast_to(np.log(1 - XGRID / 2), (len(dd), len(XGRID)))
    cnt, nd = count_eps(dd, s, e, inwin)
    ndays = int(nd[0])
    if ndays < 400:
        return np.nan, ndays, None
    rate = cnt / (ndays / W)
    ok = np.where(rate >= 4)[0]
    if not len(ok):
        return np.nan, ndays, rate
    return float(XGRID[ok.max()]), ndays, rate


# ------------------------------------------------------------------ 이동 백분위(선형보간, numpy 기본 = (n-1)·p)
def roll_pct(a, ps):
    """a (nD,), ps (G,) 백분율 -> (nD, G). 최근 252개(오늘 포함)가 모두 있어야 값."""
    out = np.full((len(a), len(ps)), np.nan)
    fin = np.isfinite(a)
    if fin.sum() < W:
        return out
    s0 = int(np.argmax(fin))
    seg = a[s0:]
    if not np.isfinite(seg).all():  # 중간 결측: 안전하게 끝까지 유효한 마지막 연속 구간만
        bad = np.where(~np.isfinite(seg))[0].max()
        s0 = s0 + bad + 1
        seg = a[s0:]
        if len(seg) < W:
            return out
    win = np.sort(sliding_window_view(seg, W), axis=1)  # (nD-s0-W+1, W)
    pos = np.asarray(ps) / 100 * (W - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, W - 1)
    fr = pos - lo
    vals = win[:, lo] + fr * (win[:, hi] - win[:, lo])
    out[s0 + W - 1:] = vals
    return out


# ------------------------------------------------------------------ 단계
def override(Z0, cdn, cup=None, mag_dn=None, mag_up=None):
    z = Z0.astype(np.int16).copy()
    nonext = (Z0 != -9) & (np.abs(Z0) != 2)
    if cup is None:
        z = np.where(nonext & cdn, -1, z)
        return z.astype(np.int8)
    z = np.where(nonext & cdn & ~cup, -1, z)
    z = np.where(nonext & cup & ~cdn, 1, z)
    big = np.where(mag_dn > mag_up, -1, np.where(mag_up > mag_dn, 1, z))
    z = np.where(nonext & cdn & cup, big, z)
    return z.astype(np.int8)


# ------------------------------------------------------------------ 지표(turn_A 정의 재현: 종목별 비율의 평균)
def flips_ps(Z, colmask, rm):
    z = Z[:, colmask].astype(np.int16)
    zt, zp = z[1:], z[:-1]
    ok = (zt != -9) & (zp != -9) & rm[1:, None]
    ch = (ok & (zt != zp)).sum(0)
    vd = ((z != -9) & rm[:, None]).sum(0)
    m = vd > 0
    return float(np.mean(ch[m] / (vd[m] / W)))


def direct_ps(Z, colmask, rm, gap=3):
    z = Z[:, colmask].astype(np.int16)
    nD, nS = z.shape
    out = []
    for j in range(nS):
        cnt, last_s, last_i = 0, 0, -10 ** 9
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
        vd = ((z[:, j] != -9) & rm).sum()
        if vd:
            out.append(cnt / (vd / W))
    return float(np.mean(out))


def steady_up(Z, r2, slope60, L10, colmask, rm):
    with np.errstate(invalid="ignore"):
        cond = (r2 >= 0.8) & (slope60 >= np.log(1.15)) & (L10 == 1)
    sel = cond[:, colmask] & (Z[:, colmask] != -9) & rm[:, None]
    return float(np.mean(Z[:, colmask][sel] > 0)), int(sel.sum())


def cover(Z, rare, colmask):
    sel = rare[:, colmask] & (Z[:, colmask] != -9)
    z = Z[:, colmask][sel]
    n = int(len(z))
    if not n:
        return {"n": 0}
    # 종목별 평균(보조)
    per = []
    for j in np.where(colmask)[0]:
        sj = rare[:, j] & (Z[:, j] != -9)
        if sj.sum() >= 5:
            per.append(np.mean(Z[sj, j] < 0))
    return {"n": n, "down": float(np.mean(z < 0)), "gray": float(np.mean(z == 0)), "up": float(np.mean(z > 0)),
            "extreme_down": float(np.mean(z == -2)), "n_syms": int(len(per)),
            "per_sym_mean_down": float(np.mean(per)) if per else float("nan")}


def main():
    f, etf = CT.load()
    C = f["Close"]
    dates = list(C.index)
    syms = list(C.columns)
    nD, nS = C.shape
    stock = np.array([s not in etf for s in syms])
    etfm = ~stock
    F = CT.features(C)
    Cn = C.to_numpy()
    lr = F["lr"]
    mx20 = C.rolling(20, min_periods=20).max()
    mn20 = C.rolling(20, min_periods=20).min()
    dd20 = np.log(C / mx20).to_numpy()
    ru20 = np.log(C / mn20).to_numpy()
    sd252 = lr.rolling(W, min_periods=W).std(ddof=1).to_numpy()
    u252 = sd252 * np.sqrt(10)
    di = {d: i for i, d in enumerate(dates)}

    # ---- 재현 확인(B0)
    Z0 = CT.five(F, F["T3"][20], 0.5)
    rm_ly = CT.pmask(dates, CT.P_LY)
    rm_all = CT.pmask(dates, CT.P_ALL)
    lC = F["lC"]
    tt = pd.DataFrame(np.tile(np.arange(nD, dtype=float)[:, None], (1, nS)), index=C.index, columns=C.columns)
    slope60 = (lC.rolling(60, min_periods=60).cov(tt) / tt.rolling(60, min_periods=60).var() * 60).to_numpy()
    r2 = F["r2"].to_numpy()
    L, SW = CT.labels(F)
    repro = {"flips_ly_per_sym_mean": flips_ps(Z0, stock, rm_ly), "direct_ly_per_sym_mean": direct_ps(Z0, stock, rm_ly),
             "flips_ly_pooled_turnB": CT.flips(Z0, stock, rm_ly), "direct_ly_pooled_turnB": CT.direct(Z0, stock, rm_ly),
             "steady_up_slope_all": steady_up(Z0, r2, slope60, L["Z10"], stock, rm_all)[0]}
    print("재현 B0", {k: round(v, 4) for k, v in repro.items()})

    # ---- X4 (종목 자기 연 4번 낙폭)
    X4 = {}
    x4_detail = {}
    for ev, (a, b) in HIST.items():
        inwin = np.array([a <= d <= b for d in dates])
        xs = np.full(nS, np.nan)
        for j in range(nS):
            x, nd, rate = x4_for(dd20[:, j], inwin)
            xs[j] = x
            if syms[j] in CASES + ["GLW"]:
                x4_detail.setdefault(syms[j], {})[ev] = {"X4": x, "hist_days": nd}
        X4[ev] = xs
        print(ev, "X4 주식 중앙", np.nanmedian(xs[stock]), "종목 수", int(np.isfinite(xs[stock]).sum()),
              "ETF", {syms[j]: xs[j] for j in np.where(etfm)[0]})
    rare = {}
    for ev, (a, b) in EVAL.items():
        em = np.array([a <= d <= b for d in dates])
        with np.errstate(invalid="ignore"):
            rare[ev] = em[:, None] & np.isfinite(X4[ev])[None, :] & (dd20 <= np.log(1 - X4[ev])[None, :])

    # ---- m 보정 (R1): 2024~25 주식, 종목별 덮어쓰기 에피소드 연 횟수 중앙 = 4
    inC = np.array([CAL[0] <= d <= CAL[1] for d in dates])
    rates_m = []
    for j in np.where(stock)[0]:
        s = -np.outer(u252[:, j], MGRID)
        e = x_to_end(s)
        cnt, nd = count_eps(dd20[:, j], s, e, inC)
        if nd[0] >= W:
            rates_m.append(cnt / (nd[0] / W))
    rates_m = np.array(rates_m)
    med_m = np.median(rates_m, axis=0)
    jm = int(np.argmin(np.abs(med_m - 4)))
    M_CAL = float(MGRID[jm])
    print("m 보정", M_CAL, "중앙", med_m[jm], "이웃", dict(zip(MGRID[max(jm - 3, 0):jm + 4], np.round(med_m[max(jm - 3, 0):jm + 4], 3))),
          "종목", len(rates_m))

    # ---- p 보정 (R3)
    Pdd = {}
    rates_p = []
    for j in range(nS):
        Pdd[j] = roll_pct(dd20[:, j], PGRID)
    for j in np.where(stock)[0]:
        s = Pdd[j].copy()
        s[~(s < 0)] = np.nan  # 0이면 문턱 의미 없음
        e = x_to_end(s)
        cnt, nd = count_eps(dd20[:, j], s, e, inC)
        if nd.max() >= W:
            rates_p.append(np.where(nd > 0, cnt / (np.maximum(nd, 1) / W), np.nan))
    rates_p = np.array(rates_p)
    med_p = np.nanmedian(rates_p, axis=0)
    jp = int(np.argmin(np.abs(med_p - 4)))
    P_CAL = float(PGRID[jp])
    print("p 보정", P_CAL, "중앙", med_p[jp], "이웃", dict(zip(PGRID[max(jp - 3, 0):jp + 4], np.round(med_p[max(jp - 3, 0):jp + 4], 3))),
          "종목", len(rates_p))

    # ---- 백분위 관례 민감도(보조): 위치 = p/100·n - 0.5 (C=1/2)
    def roll_pct_half(a, ps):
        out_ = np.full((len(a), len(ps)), np.nan)
        fin = np.isfinite(a)
        if fin.sum() < W:
            return out_
        s0 = int(np.argmax(fin))
        seg = a[s0:]
        if not np.isfinite(seg).all():
            return out_
        win = np.sort(sliding_window_view(seg, W), axis=1)
        pos = np.clip(np.asarray(ps) / 100 * W - 0.5, 0, W - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, W - 1)
        fr = pos - lo
        out_[s0 + W - 1:] = win[:, lo] + fr * (win[:, hi] - win[:, lo])
        return out_
    rates_h = []
    for j in np.where(stock)[0]:
        s = roll_pct_half(dd20[:, j], PGRID)
        s[~(s < 0)] = np.nan
        cnt, nd = count_eps(dd20[:, j], s, x_to_end(s), inC)
        if nd.max() >= W:
            rates_h.append(np.where(nd > 0, cnt / (np.maximum(nd, 1) / W), np.nan))
    med_h = np.nanmedian(np.array(rates_h), axis=0)
    P_HALF = float(PGRID[int(np.argmin(np.abs(med_h - 4)))])
    print("p 보정(C=1/2 관례)", P_HALF)

    # ---- 후보 단계
    with np.errstate(invalid="ignore"):
        thr_m = M_CAL * u252
        cdn1 = dd20 <= -thr_m
        cup1 = ru20 >= thr_m
        thr_5 = 1.25 * u252
        cdn5 = dd20 <= -thr_5
        qdn = np.column_stack([roll_pct(dd20[:, j], [P_CAL])[:, 0] for j in range(nS)])
        qup = np.column_stack([roll_pct(ru20[:, j], [100 - P_CAL])[:, 0] for j in range(nS)])
        cdn3 = (qdn < 0) & (dd20 <= qdn)
        cup3 = (qup > 0) & (ru20 >= qup)
    Zs = {"B0": Z0,
          "R1": override(Z0, cdn1),
          "R2": override(Z0, cdn1, cup1, -dd20, ru20),
          "R3": override(Z0, cdn3),
          "R4": override(Z0, cdn3, cup3, -dd20, ru20),
          "R5": override(Z0, cdn5)}
    conds = {"B0": (None, None), "R1": (cdn1, None), "R2": (cdn1, cup1), "R3": (cdn3, None), "R4": (cdn3, cup3),
             "R5": (cdn5, None)}
    sthr = {"R1": -thr_m, "R2": -thr_m, "R5": -thr_5, "R3": np.where(qdn < 0, qdn, np.nan), "R4": np.where(qdn < 0, qdn, np.nan)}
    uthr = {"R2": thr_m, "R4": np.where(qup > 0, qup, np.nan)}

    # ---- 예측(보조)
    Os, Hs, Ls, Cs = (f[k].loc[:, stock] for k in ("Open", "High", "Low", "Close"))
    I = ZM.Ind(Os, Hs, Ls, Cs)
    assert list(I.dates) == dates
    Fw = ZM.Fwd(I, 10)
    prow = {p: ZM.period_rows(I, p, 10) for p in ("앞", "뒤", "2025")}

    rms = {k: CT.pmask(dates, v) for k, v in CT.PER.items()}
    rm26 = CT.pmask(dates, EVAL["2026"])
    rm25 = CT.pmask(dates, EVAL["2025"])
    last = nD - 1

    # 새 20일 고점(앞으로 10거래일 안)
    is_hi = (Cn >= mx20.to_numpy() - 1e-12) & np.isfinite(Cn)
    newhi10 = np.zeros((nD, nS), bool)
    rec10 = np.zeros((nD, nS), bool)
    mxv = mx20.to_numpy()
    for k in range(1, 11):
        newhi10[:-k] |= is_hi[k:]
        with np.errstate(invalid="ignore"):
            rec10[:-k] |= Cn[k:] >= mxv[:-k] - 1e-12
    fwd_ok = np.arange(nD) + 10 <= last

    # 사례: 문턱 가격 풀기
    def solve_price(j, i, fn, lo_frac=0.3):
        """fn(Cx) -> True면 하방 조건 성립. Cx를 내리면 성립한다고 보고 가장 큰 성립 가격을 찾는다."""
        hi = float(np.nanmax(Cn[i - 19:i, j].tolist() + [Cn[i, j]])) * 1.2
        grid = np.linspace(hi, Cn[i, j] * lo_frac, 4000)
        okv = np.array([fn(c) for c in grid])
        if not okv.any():
            return float("nan")
        k = int(np.argmax(okv))
        if k == 0:
            return float(grid[0])
        a_, b_ = grid[k], grid[k - 1]  # a_ 성립, b_ 불성립
        for _ in range(60):
            mid = (a_ + b_) / 2
            if fn(mid):
                a_ = mid
            else:
                b_ = mid
        return float(a_)

    def path(j, i, Cx):
        c = Cn[:i + 1, j].copy()
        c[i] = Cx
        return c

    def f_t3(j, i):
        def fn(Cx):
            c = path(j, i, Cx)
            lc = np.log(c[i - 20:i + 1])
            r = np.diff(lc)
            sd = np.std(r, ddof=1)
            t3 = (lc[-1] - lc[-20:].mean()) / (sd * CT.kmul(20))
            return t3 <= -0.5
        return fn

    def f_m(j, i, m):
        def fn(Cx):
            c = path(j, i, Cx)
            r = np.diff(np.log(c[i - W:i + 1]))
            sd = np.std(r, ddof=1)
            dd = np.log(Cx / c[i - 19:i + 1].max())
            return dd <= -m * sd * np.sqrt(10)
        return fn

    def f_p(j, i, p):
        def fn(Cx):
            c = path(j, i, Cx)
            cc = c[i - W - 19 + 1:i + 1]
            ddw = np.log(cc[19:] / np.array([cc[k - 19:k + 1].max() for k in range(19, len(cc))]))
            q = np.percentile(ddw, p)
            return (q < 0) and (ddw[-1] <= q)
        return fn

    def f_rsi(j, i):
        def fn(Cx):
            c = path(j, i, Cx)
            return CT.rsi_pine(c)[i] <= 30
        return fn

    i12 = di["2026-10-12"]
    case_info = {}
    for s in CASES:
        j = syms.index(s)
        info = {"close": float(Cn[i12, j]), "rsi": float(F["rsi"].iat[i12, j]), "T3": float(F["T3"][20].iat[i12, j]),
                "dd20_pct": float(np.exp(dd20[i12, j]) - 1), "max20": float(mxv[i12, j]),
                "X4_2026": float(X4["2026"][j]), "rare_today": bool(rare["2026"][i12, j]),
                "u252": float(u252[i12, j]),
                "override_m_dd_pct": float(np.exp(-M_CAL * u252[i12, j]) - 1),
                "override_m125_dd_pct": float(np.exp(-1.25 * u252[i12, j]) - 1),
                "override_p_dd_pct": float(np.exp(qdn[i12, j]) - 1) if np.isfinite(qdn[i12, j]) else None,
                "stage": {n: int(Zs[n][i12, j]) for n in IDS},
                "price_T3_down": solve_price(j, i12, f_t3(j, i12)),
                "price_rsi30": solve_price(j, i12, f_rsi(j, i12)),
                "price_R1": solve_price(j, i12, f_m(j, i12, M_CAL)),
                "price_R5": solve_price(j, i12, f_m(j, i12, 1.25)),
                "price_R3": solve_price(j, i12, f_p(j, i12, P_CAL)),
                "price_X4_rare": float(mxv[i12, j] * (1 - X4["2026"][j]))}
        case_info[s] = info
        print("사례", s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()})

    def window_count(Z, s, a, b):
        j = syms.index(s)
        idx = [i for i, d in enumerate(dates) if a <= d <= b]
        z = Z[idx, j]
        return {"days": len(idx), "down": int(((z < 0) & (z != -9)).sum()), "gray": int((z == 0).sum()),
                "up": int((z > 0).sum()), "seq": [int(v) for v in z]}

    out = {}
    for n in IDS:
        Z = Zs[n]
        res = {}
        res["param"] = {"B0": None, "R1": M_CAL, "R2": M_CAL, "R3": P_CAL, "R4": P_CAL, "R5": 1.25}[n]
        res["cover26"] = cover(Z, rare["2026"], stock)
        res["cover25"] = cover(Z, rare["2025"], stock)
        res["etf_cover26"] = cover(Z, rare["2026"], etfm)
        for h, (a, b) in (("H1", ("2026-01-02", "2026-05-29")), ("H2", ("2026-06-01", "2026-10-12"))):
            hm = np.array([a <= d <= b for d in dates])
            res[f"cover26_{h}"] = cover(Z, rare["2026"] & hm[:, None], stock)
        res["etf_cover26_by_etf"] = {}
        for j in np.where(etfm)[0]:
            cm = np.zeros(nS, bool)
            cm[j] = True
            c_ = cover(Z, rare["2026"], cm)
            res["etf_cover26_by_etf"][syms[j]] = {"n": c_["n"], "down": c_.get("down")}
        res["etf_cover25"] = cover(Z, rare["2025"], etfm)
        # 덮어쓰기 실제 발동(단계를 바꾼 날)
        ch = (Z != Z0) & (Z0 != -9)
        v26 = (Z != -9) & rm26[:, None]
        per = []
        for j in np.where(stock)[0]:
            vd = v26[:, j].sum()
            if vd:
                per.append(ch[rm26, j].sum() / (vd / W))
        res["fire_days_per_sy_2026"] = float(np.mean(per))
        res["fire_days_pooled_2026"] = float(ch[rm26][:, stock].sum() / (v26[:, stock].sum() / W))
        res["fire_to_down_share"] = float(((Z == -1) & ch)[rm26][:, stock].sum() / max(ch[rm26][:, stock].sum(), 1))
        res["changed_cells_2026_stock"] = int(ch[rm26][:, stock].sum())
        # 덮어쓰기 조건 에피소드(2026, 2024~25)
        if n != "B0":
            for tag, (a, b) in (("2026", EVAL["2026"]), ("cal", CAL)):
                inw = np.array([a <= d <= b for d in dates])
                rl = []
                for j in np.where(stock)[0]:
                    s = sthr[n][:, j][:, None]
                    cnt, nd = count_eps(dd20[:, j], s, x_to_end(s), inw)
                    if nd[0] >= (100 if tag == "2026" else W):
                        rl.append(cnt[0] / (nd[0] / W))
                res[f"fire_eps_median_{tag}"] = float(np.median(rl))
                res[f"fire_eps_mean_{tag}"] = float(np.mean(rl))
                res[f"fire_eps_nsyms_{tag}"] = len(rl)
            if n in uthr:
                inw = rm26
                rl = []
                for j in np.where(stock)[0]:
                    s = -uthr[n][:, j][:, None]  # 상승: -ru20 <= -thr 로 같은 기계 사용
                    cnt, nd = count_eps(-ru20[:, j], s, x_to_end(s), inw)
                    if nd[0] >= 100:
                        rl.append(cnt[0] / (nd[0] / W))
                res["fire_up_eps_median_2026"] = float(np.median(rl))
        # 오경보: 덮어쓰기로 하방이 된 날 -> 10거래일 안 새 20일 최고 종가
        for tag, rmx in (("2026", rm26), ("2025", rm25)):
            sel = (Z == -1) & ch & rmx[:, None] & fwd_ok[:, None]
            sel[:, ~stock] = False
            res[f"false_rare_{tag}"] = float(newhi10[sel].mean()) if sel.sum() else float("nan")
            res[f"false_rare_recover_prev_high_{tag}"] = float(rec10[sel].mean()) if sel.sum() else float("nan")
            res[f"false_rare_n_{tag}"] = int(sel.sum())
            # 기준선: B0가 이미 하방(-1)인 날
            selb = (Z0 == -1) & rmx[:, None] & fwd_ok[:, None]
            selb[:, ~stock] = False
            res[f"base_b0_down_newhi10_{tag}"] = float(newhi10[selb].mean())
            res[f"base_b0_down_recover_prev_high_{tag}"] = float(rec10[selb].mean())
            selr = rare[tag] & fwd_ok[:, None]
            selr[:, ~stock] = False
            res[f"base_rare_newhi10_{tag}"] = float(newhi10[selr].mean())
        # 바뀜·직행·꾸준한 상승·점수
        res["flips"] = flips_ps(Z, stock, rm_ly)
        res["direct"] = direct_ps(Z, stock, rm_ly)
        res["flips_pooled"] = CT.flips(Z, stock, rm_ly)
        res["direct_pooled"] = CT.direct(Z, stock, rm_ly)
        res["steady_up"], res["steady_up_n"] = steady_up(Z, r2, slope60, L["Z10"], stock, rm_all)
        ao = {th: {pk: CT.agree_opp(Z, L[th], stock, rm) for pk, rm in rms.items()} for th in CT.THETAS}
        cells = [ao[th][h] for th in CT.THETAS for h in ("2026H1", "2026H2")]
        res["score"] = float(np.mean([c["agree"] - 2 * c["opp"] for c in cells]))
        res["score_2026H1"] = float(np.mean([ao[th]["2026H1"]["agree"] - 2 * ao[th]["2026H1"]["opp"] for th in CT.THETAS]))
        res["score_2026H2"] = float(np.mean([ao[th]["2026H2"]["agree"] - 2 * ao[th]["2026H2"]["opp"] for th in CT.THETAS]))
        res["score_2025"] = float(np.mean([ao[th]["2025"]["agree"] - 2 * ao[th]["2025"]["opp"] for th in CT.THETAS]))
        res["opp26_mean"] = float(np.mean([c["opp"] for c in cells]))
        res["gray_2026"] = CT.gray(Z, stock, rm26)
        pr = {}
        for p in ("앞", "뒤", "2025"):
            e = ZM.evaluate(Z[:, stock], Fw, prow[p], full=True)
            pr[p] = {"S": e["S"] * 100, "terms_pp": {k: v * 100 for k, v in e["terms"].items()}, "t": e["t"]}
        res["pred"] = pr
        res["etf_flips"] = flips_ps(Z, etfm, rm_ly)
        res["etf_direct"] = direct_ps(Z, etfm, rm_ly)
        aoe = [CT.agree_opp(Z, L[th], etfm, rms[h]) for th in CT.THETAS for h in ("2026H1", "2026H2")]
        res["etf_score"] = float(np.mean([c["agree"] - 2 * c["opp"] for c in aoe]))
        res["cases_1012"] = {s: int(Z[i12, syms.index(s)]) for s in CASES}
        res["mu_0626_0728"] = window_count(Z, "MU", "2026-06-26", "2026-07-28")
        res["glw_0702_0714"] = window_count(Z, "GLW", "2026-07-02", "2026-07-14")
        res["soxx_0901_1012"] = window_count(Z, "SOXX", "2026-09-01", "2026-10-12")
        res["ewy_0908_1012"] = window_count(Z, "EWY", "2026-09-08", "2026-10-12")
        out[n] = res
        c26, c25 = res["cover26"], res["cover25"]
        print(f"{n} p={res['param']} cover26 {c26['down']:.3f}/{c26['gray']:.3f}/{c26['up']:.3f} n{c26['n']} "
              f"cover25 {c25['down']:.3f} etf26 {res['etf_cover26']['down']:.3f} fire {res['fire_days_per_sy_2026']:.2f} "
              f"eps26 {res.get('fire_eps_median_2026', float('nan')):.2f} flips {res['flips']:.2f} direct {res['direct']:.2f} "
              f"steady {res['steady_up']:.4f} score {res['score']:.4f} false {res['false_rare_2026']:.3f} "
              f"S {pr['앞']['S']:+.2f}/{pr['뒤']['S']:+.2f}/{pr['2025']['S']:+.2f} "
              f"T1 {pr['앞']['terms_pp']['T1 상방-하방']:+.2f}/{pr['뒤']['terms_pp']['T1 상방-하방']:+.2f} "
              f"cases {res['cases_1012']} MU↓{res['mu_0626_0728']['down']} GLW↓{res['glw_0702_0714']['down']} "
              f"ETFflips {res['etf_flips']:.1f}")

    # B0 대비 하방 표시가 는 rare_day 칸 수
    for n in IDS[1:]:
        r = rare["2026"][:, stock]
        a_ = Zs["B0"][:, stock][r]
        b_ = Zs[n][:, stock][r]
        out[n]["rare26_newly_down"] = int(((b_ < 0) & (a_ >= 0) & (a_ != -9)).sum())

    res_all = {
        "repro_B0": repro,
        "calibration": {"m": M_CAL, "m_grid_median": dict(zip(map(float, MGRID), map(float, med_m))), "m_nsyms": int(len(rates_m)),
                        "p_half_convention": P_HALF,
                        "p": P_CAL, "p_grid_median": dict(zip(map(float, PGRID), map(float, med_p))), "p_nsyms": int(len(rates_p))},
        "X4": {ev: {"stock_median": float(np.nanmedian(X4[ev][stock])), "stock_q25": float(np.nanpercentile(X4[ev][stock], 25)),
                    "stock_q75": float(np.nanpercentile(X4[ev][stock], 75)), "n_stock": int(np.isfinite(X4[ev][stock]).sum()),
                    "etf": {syms[j]: float(X4[ev][j]) for j in np.where(etfm)[0]}} for ev in X4},
        "X4_cases": x4_detail,
        "rare_days": {ev: {"stock": int(rare[ev][:, stock].sum()), "etf": int(rare[ev][:, etfm].sum())} for ev in rare},
        "cases_1012": case_info,
        "candidates": out,
        "spec_choices": {
            "flips/direct": "turn_A 재현: 종목별 (바뀜 수 / (유효일/252))의 150종목 단순 평균, 2025-10-13~2026-10-12. turn_B 합산식도 *_pooled로 함께 저장",
            "steady_up": "60개 로그 종가 시간 회귀 R² ≥ 0.8, 회귀 기울기×60 ≥ ln1.15, Z10(로그 0.10) 스윙 +1, 2024-01-02~2026-10-12, 상방 계열(+1·+2) 비율",
            "episode": "상태 기계를 자료 시작부터 돌리고, 창 안에서 시작한 에피소드만 센다(창 시작 전 상태는 과거 자료로 이어받음). 연수 = 창 안 유효일/252",
            "override_episode": "문턱 L(로그, 그날 값)을 X = 1-e^-L로 바꿔 시작 dd20 ≤ -L, 끝 dd20 > ln(1-X/2)",
            "calibration_select": "격자에서 종목별 연 에피소드 중앙이 4에 가장 가까운 값. 보정 종목 = 창 안 덮어쓰기 유효일 252일 이상",
            "percentile": "numpy/pandas 선형보간((n-1)·p 위치) — 파인 ta.percentile_linear_interpolation과 같다고 가정(이식 대조 안 함)",
            "R2/R4 tie": "둘 다 성립하면 원값 |dd20| vs ru20 큰 쪽",
            "fire_eps": "덮어쓰기 '조건' 에피소드(RSI 무관). 2026은 유효일 100일 이상 종목 중앙",
            "false_rare": "덮어쓰기로 -1이 된 날(B0와 다름, Z=-1), t+1..t+10 중 C가 그날 20일 최고 종가(C ≥ 20일 최고)인 날이 있는 비율. t+10이 자료 안인 날만. 보조: t일 20일 최고 종가 회복",
            "cover": "rare_day이면서 단계 유효(-9 아님)인 주식 칸 합산 비율",
            "trigger_price": "10/12 종가를 Cx로 바꿔 다시 계산(sd252·20일 최고·백분위 창에 Cx 반영)한 조건 성립 최대 가격",
        },
    }
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res_all, fh, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("saved", os.path.join(HERE, "results.json"))


if __name__ == "__main__":
    main()
