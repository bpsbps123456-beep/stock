"""추세 5단계 — 사용자 기준 '드문 낙폭(종목 자기 이력 연 4번급)이면 하방' 후보 6개 비교 (rare_R = A·B 불일치 해결판, 연구용).

rare_A/rare_compare.py를 바탕으로 두 군데만 고쳤다(근거: diag_calib.py·diag_calib.json).
  1) m·p 보정 선택 규칙: '중앙이 4에 가장 가까운 값' 대신 X4 정의(연 횟수 ≥ 4가 되는 가장 큰 X)와 같은 꼴인
     '종목별 연 에피소드 중앙 ≥ 4 가운데 가장 깊은 문턱(m 최대·p 최소)'. 보정 종목 = 이력 400일 이상(X4와 같은 기준).
  2) B0 false_rare: 덮어쓰기가 없으므로 해당 없음(NaN). B0 −1 날 전체의 비교 기준선은 false_ref_B0_down에만 둔다.

사용자(2026-10-13): "드물다는 기준은 1년에 4번 정도. 계절당 1번이면 큰 거다. 실적발표급으로 4번만 있는 거니까."
저장소 파일은 읽기·import만 한다. turn_A/turn_compare.py(지그재그 score·flips·direct·steady_up·예측)를 import해 재사용.

후보(사전 등록, RSI14 ≥70 → +2, ≤30 → −2가 먼저, 덮어쓰기는 극단이 아닐 때만):
  B0 현재 / R1 dd20 ≤ −m·sd252·√10 → −1 (m 보정) / R2 R1 + 대칭 상승 / R3 dd20 ≤ P_p(dd20, 252) (p 보정)
  R4 R3 + 대칭 상승 ru20 ≥ P_{100−p}(ru20, 252) / R5 R1에서 m = 1.25 고정
실행: PYTHONIOENCODING=utf-8 py rare_compare.py   결과: 같은 폴더 results.json, results.txt
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TURN_A = os.path.join(HERE, "..", "turn_A")
sys.path.insert(0, TURN_A)
import turn_compare as TC  # noqa: E402  (per_symbol, stage5, agree_tab, score_of, flips_direct, ZM)

ZM = TC.ZM
PANEL = TC.PANEL
P_2026 = ("2026-01-02", "2026-10-12")
P_2025 = ("2025-01-02", "2025-12-31")
H_2026 = ("2024-01-02", "2025-12-31")      # 2026 평가용 이력(보정 기간도 같다)
H_2025 = ("2023-03-01", "2024-12-31")      # 2025 평가용 이력
MIN_HIST = 400
XGRID = np.round(np.arange(0.0025, 0.8000001, 0.0025), 4)
MGRID = np.round(np.arange(0.50, 3.0000001, 0.05), 2)
PGRID = np.round(np.arange(0.5, 30.0000001, 0.5), 1)
CANDS = ["B0", "R1", "R2", "R3", "R4", "R5"]
CASE_SYMS = ["SOXX", "EWY", "SMH", "MU", "SNDK", "NVDA"]
SQ10 = np.sqrt(10.0)
K20 = np.sqrt(19 * 39 / 120)


def inper(d, p):
    return (d >= p[0]) & (d <= p[1])


# ---------------------------------------------------------------- 에피소드 (히스테리시스, 벡터화)
def episodes(dd, thr, rec):
    """dd, thr, rec: (..., n). 시작: dd ≤ thr, 끝: dd > rec. 값이 NaN인 날은 상태 유지(건너뜀). 반환: 에피소드 수, 상태."""
    dd = np.asarray(dd, float)
    thr = np.broadcast_to(thr, np.broadcast_shapes(np.shape(thr), dd.shape))
    rec = np.broadcast_to(rec, thr.shape)
    ddb = np.broadcast_to(dd, thr.shape)
    ok = np.isfinite(ddb) & np.isfinite(thr) & np.isfinite(rec)
    ev = np.where(ok & (ddb <= thr), 1.0, np.where(ok & (ddb > rec), 0.0, np.nan))
    two = ev.reshape(-1, ev.shape[-1])
    st = pd.DataFrame(two).ffill(axis=1).fillna(0.0).to_numpy()
    prev = np.concatenate([np.zeros((st.shape[0], 1)), st[:, :-1]], axis=1)
    n = ((st == 1) & (prev == 0)).sum(axis=1)
    start = (st == 1) & (prev == 0)
    return n.reshape(ev.shape[:-1]), st.reshape(ev.shape), start.reshape(ev.shape)


def thr_from_log(a):
    """a: 문턱(로그, 음수). 같은 정의로 X = 1 − e^a, 회복선 ln(1 − X/2)."""
    X = 1.0 - np.exp(a)
    return a, np.log(1.0 - X / 2.0)


def x4_of(dd, years):
    """dd: 이력 기간 dd20(유한값만). 연 횟수 ≥ 4가 되는 가장 큰 X (0.25%p 격자)."""
    thr = np.log(1 - XGRID)[:, None]
    rec = np.log(1 - XGRID / 2)[:, None]
    n, _, _ = episodes(dd[None, :], thr, rec)
    ok = n / years >= 4
    return float(XGRID[ok].max()) if ok.any() else float("nan")


def roll_pct_sorted(x, w=252):
    """끝점 정렬 이동 창의 정렬본. 창 안에 NaN 있으면 그 행 NaN."""
    n = len(x)
    out = np.full((n, w), np.nan)
    if n < w:
        return out
    W = np.lib.stride_tricks.sliding_window_view(x, w)
    S = np.sort(W, axis=1)
    bad = ~np.isfinite(W).all(axis=1)
    S[bad] = np.nan
    out[w - 1:] = S
    return out


def pct_from_sorted(S, q):
    """파인 ta.percentile_linear_interpolation 해석: 정렬 창에서 순위 (q/100)·(N−1) 선형보간(numpy 'linear')."""
    w = S.shape[1]
    pos = q / 100.0 * (w - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, w - 1)
    fr = pos - lo
    return S[:, lo] + fr * (S[:, hi] - S[:, lo])


# ---------------------------------------------------------------- 종목별 계산
def per_symbol(g):
    o, piv = TC.per_symbol(g)
    C = o["C"].to_numpy()
    cs = pd.Series(C)
    H20 = cs.rolling(20, min_periods=20).max().to_numpy()
    L20 = cs.rolling(20, min_periods=20).min().to_numpy()
    o["H20"], o["L20"] = H20, L20
    o["dd20"] = np.log(C / H20)
    o["ru20"] = np.log(C / L20)
    o["sd252"] = pd.Series(o["lr"].to_numpy()).rolling(252, min_periods=252).std(ddof=1).to_numpy()
    o["u252"] = o["sd252"] * SQ10
    Sdd = roll_pct_sorted(o["dd20"].to_numpy())
    Sru = roll_pct_sorted(o["ru20"].to_numpy())
    pdd = np.vstack([pct_from_sorted(Sdd, p) for p in PGRID])            # (len(PGRID), n)
    pru = np.vstack([pct_from_sorted(Sru, 100 - p) for p in PGRID])
    return o, piv, pdd, pru, Sdd, Sru


def main():
    raw = pd.read_parquet(PANEL)
    raw["ts"] = pd.to_datetime(raw["ts"])
    parts, pivs, PD, PU, SD, SU = [], {}, {}, {}, {}, {}
    off = 0
    for sym, g in raw.groupby("sym", sort=True):
        o, p, pdd, pru, sdd, sru = per_symbol(g)
        parts.append(o)
        pivs[sym] = {k: [off + i for i in v] for k, v in p.items()}
        PD[sym], PU[sym] = pdd, pru
        SD[sym], SU[sym] = sdd[-1], sru[-1]      # 마지막 날 정렬 창(사례 문턱 계산용)
        off += len(o)
    df = pd.concat(parts, ignore_index=True)
    sym_rows = {s: np.flatnonzero(df["sym"].to_numpy() == s) for s in df["sym"].unique()}
    is_etf = df.groupby("sym")["is_etf"].first().to_dict()
    stocks = sorted(s for s in sym_rows if not is_etf[s])
    etfs = sorted(s for s in sym_rows if is_etf[s])
    dates = df["date"].to_numpy()
    dd20 = df["dd20"].to_numpy()
    ru20 = df["ru20"].to_numpy()
    u252 = df["u252"].to_numpy()
    R = df["rsi"].to_numpy()

    # ---------------------------------------------------------------- 1. 보정 (2024-01-02~2025-12-31, 주식)
    calib_m, calib_p = {}, {}
    m_rates = {m: [] for m in MGRID}
    p_rates = {p: [] for p in PGRID}
    for s in stocks:
        r = sym_rows[s]
        w = inper(dates[r], H_2026)
        okm = w & np.isfinite(dd20[r]) & np.isfinite(u252[r])
        if okm.sum() >= MIN_HIST:
            yrs = okm.sum() / 252.0
            a = -(MGRID[:, None] * u252[r][okm][None, :])
            thr, rec = thr_from_log(a)
            n, _, _ = episodes(dd20[r][okm][None, :], thr, rec)
            for m, k in zip(MGRID, n):
                m_rates[m].append(k / yrs)
        pdd = PD[s][:, w]
        okp = np.isfinite(dd20[r][w]) & np.isfinite(pdd[0])
        if okp.sum() >= MIN_HIST:
            yrs = okp.sum() / 252.0
            a = np.minimum(pdd[:, okp], -1e-12)        # 문턱이 0 이상이면 사실상 상시 → 아주 작은 음수로(작은 p에선 발생 안 함)
            thr, rec = thr_from_log(a)
            n, _, _ = episodes(dd20[r][w][okp][None, :], thr, rec)
            for p, k in zip(PGRID, n):
                p_rates[p].append(k / yrs)
    m_med = {float(m): float(np.median(v)) for m, v in m_rates.items()}
    p_med = {float(p): float(np.median(v)) for p, v in p_rates.items()}
    # '중앙값이 4가 되게': 4에 가장 가까운 값(동률이면 문턱이 더 깊은 쪽 = m 큰 쪽, p 작은 쪽)
    m_close = min(m_med, key=lambda m: (abs(m_med[m] - 4), -m))     # 참고: 4에 가장 가까운 값(동률은 깊은 쪽)
    p_close = min(p_med, key=lambda p: (abs(p_med[p] - 4), p))
    m_ge4 = max([m for m in m_med if m_med[m] >= 4], default=float("nan"))
    p_ge4 = min([p for p in p_med if p_med[p] >= 4], default=float("nan"))
    m_star, p_star = m_ge4, p_ge4                                   # 채택: 중앙 ≥ 4 가운데 가장 깊은 문턱(X4 정의와 같은 꼴)
    print("m 보정 곡선:", {m: round(v, 2) for m, v in m_med.items() if 0.9 <= m <= 2.2})
    print("p 보정 곡선:", {p: round(v, 2) for p, v in p_med.items() if p <= 15})
    print(f"m* = {m_star} (중앙 {m_med[m_star]:.2f}), ≥4 최대 m = {m_ge4}; p* = {p_star} (중앙 {p_med[p_star]:.2f}), ≥4 최소 p = {p_ge4}")
    calib = {"rule": "median>=4 deepest (X4-consistent)", "m_closest": m_close, "p_closest": p_close, "m_star": m_star, "m_median": m_med[m_star], "m_ge4_max": m_ge4, "m_curve": m_med,
             "p_star": p_star, "p_median": p_med[p_star], "p_ge4_min": p_ge4, "p_curve": p_med,
             "n_stocks_m": len(m_rates[MGRID[0]]), "n_stocks_p": len(p_rates[PGRID[0]]),
             "m_iqr": [float(np.percentile(m_rates[m_star], 25)), float(np.percentile(m_rates[m_star], 75))],
             "p_iqr": [float(np.percentile(p_rates[p_star], 25)), float(np.percentile(p_rates[p_star], 75))]}

    # ---------------------------------------------------------------- 2. 문턱·단계
    pi = int(np.flatnonzero(PGRID == p_star)[0])
    Pd = np.full(len(df), np.nan)
    Pu = np.full(len(df), np.nan)
    for s, r in sym_rows.items():
        Pd[r] = PD[s][pi]
        Pu[r] = PU[s][pi]
    df["Pd"], df["Pu"] = Pd, Pu
    b0 = TC.stage5(R, df["T3_20"].to_numpy(), 0.5)
    ext = (R >= 70) | (R <= 30)

    def ow(base, cd, cu=None, dmag=None, umag=None):
        if cu is None:
            out = np.where(cd, -1, base)
        else:
            both = cd & cu
            big = np.where(dmag > umag, -1, np.where(umag > dmag, 1, base))
            out = np.where(both, big, np.where(cd, -1, np.where(cu, 1, base)))
        return np.where(ext, base, out)

    with np.errstate(invalid="ignore"):
        thr = {"R1": -m_star * u252, "R5": -1.25 * u252, "R3": Pd}
        cdR1 = dd20 <= thr["R1"]
        cuR1 = ru20 >= m_star * u252
        cdR5 = dd20 <= thr["R5"]
        cdR3 = dd20 <= Pd
        cuR3 = ru20 >= Pu
    stg = {"B0": b0, "R1": ow(b0, cdR1), "R2": ow(b0, cdR1, cuR1, -dd20, ru20), "R3": ow(b0, cdR3),
           "R4": ow(b0, cdR3, cuR3, -dd20, ru20), "R5": ow(b0, cdR5)}
    cond = {"R1": (cdR1, None), "R2": (cdR1, cuR1), "R3": (cdR3, None), "R4": (cdR3, cuR3), "R5": (cdR5, None)}
    dthr = {"R1": thr["R1"], "R2": thr["R1"], "R3": Pd, "R4": Pd, "R5": thr["R5"]}
    valid = df["valid"].to_numpy()
    for k, v in stg.items():
        df["z_" + k] = np.where(valid, v, -9).astype(int)

    stock = ~df["is_etf"].to_numpy()
    etf = df["is_etf"].to_numpy()
    V = valid
    S_all = pd.Series(stock & V)
    E_all = pd.Series(etf & V)
    S_ly = S_all & inper(df["date"], TC.P_LASTY)
    E_ly = E_all & inper(df["date"], TC.P_LASTY)
    S_inall = S_all & inper(df["date"], TC.PER["ALL"])
    stU = (df["slope60"] * 60 >= np.log(1.15)) & (df["r2_60"] >= 0.8) & (df["lab_Z10"] == 1)
    stD = (df["slope60"] * 60 <= -np.log(1.15)) & (df["r2_60"] >= 0.8) & (df["lab_Z10"] == -1)

    # ---------------------------------------------------------------- 3. 정답용 자기 연4번 낙폭 X4
    X4 = {"2026": {}, "2025": {}}
    for lab, hist in (("2026", H_2026), ("2025", H_2025)):
        for s, r in sym_rows.items():
            w = inper(dates[r], hist) & np.isfinite(dd20[r])
            if w.sum() < MIN_HIST:
                continue
            X4[lab][s] = x4_of(dd20[r][w], w.sum() / 252.0)
    rare = {}
    rare_start = {}
    for lab, per in (("2026", P_2026), ("2025", P_2025)):
        m = np.zeros(len(df), bool)
        ms = np.zeros(len(df), bool)
        for s, X in X4[lab].items():
            if not np.isfinite(X):
                continue
            r = sym_rows[s]
            inp = inper(dates[r], per)
            m[r] = inp & (dd20[r] <= np.log(1 - X)) & valid[r]
            # 평가 기간 안 에피소드 시작일(같은 히스테리시스 정의)
            rr = r[inp]
            _, _, st0 = episodes(dd20[rr][None, :], np.log(1 - X), np.log(1 - X / 2))
            ms[rr] = st0[0] & valid[rr]
        rare[lab], rare_start[lab] = m, ms

    def cover(z, mask):
        v = z[mask]
        n = len(v)
        if not n:
            return {"n": 0}
        per_sym = df.loc[mask].assign(zz=v).groupby("sym")["zz"].apply(lambda x: np.mean(x < 0))
        return {"n": int(n), "down": float(np.mean(v < 0)), "gray": float(np.mean(v == 0)), "up": float(np.mean(v > 0)),
                "down_ex": float(np.mean(v == -2)), "down_sym_mean": float(per_sym.mean()), "n_sym": int(len(per_sym))}

    # 미래 10일 안 20일 최고 종가 갱신(오경보)
    C = df["C"].to_numpy()
    H20 = df["H20"].to_numpy()
    newhi = np.isfinite(dd20) & (dd20 >= -1e-12)
    fut_newhi = np.zeros(len(df), bool)
    fut_recover = np.zeros(len(df), bool)
    fut_ok = np.zeros(len(df), bool)
    for s, r in sym_rows.items():
        n = len(r)
        nh = newhi[r]
        c = C[r]
        h = H20[r]
        for t in range(n - 10):
            fut_ok[r[t]] = True
            fut_newhi[r[t]] = nh[t + 1:t + 11].any()
            fut_recover[r[t]] = np.isfinite(h[t]) and (c[t + 1:t + 11] >= h[t]).any()

    # 예측 엔진
    st_raw = raw[~raw["is_etf"]].copy()
    st_raw["d"] = st_raw["ts"].dt.strftime("%Y-%m-%d")
    pv = {k: st_raw.pivot_table(index="d", columns="sym", values=k).sort_index() for k in ("Open", "High", "Low", "Close")}
    I = ZM.Ind(pv["Open"], pv["High"], pv["Low"], pv["Close"])
    F = ZM.Fwd(I, 10)
    prow = {p: ZM.period_rows(I, p, 10) for p in ("앞", "뒤", "2025")}

    in26 = inper(df["date"], P_2026).to_numpy()
    res = {"meta": {}, "calibration": calib, "cands": {}, "cases": {}, "x4": {}}
    for k in CANDS:
        st = "z_" + k
        z = df[st].to_numpy()
        r = {"id": k, "param": (m_star if k in ("R1", "R2") else p_star if k in ("R3", "R4") else 1.25 if k == "R5" else None)}
        # 드문 낙폭 덮기
        for lab in ("2026", "2025"):
            r[f"rare_{lab}_stock"] = cover(z, rare[lab] & stock)
            r[f"rare_{lab}_etf"] = cover(z, rare[lab] & etf)
            r[f"rare_{lab}_stock_first"] = cover(z, rare_start[lab] & stock)
            r[f"rare_{lab}_etf_first"] = cover(z, rare_start[lab] & etf)
        r["cover26"] = r["rare_2026_stock"]["down"]
        r["cover26_gray"] = r["rare_2026_stock"]["gray"]
        r["cover26_up"] = r["rare_2026_stock"]["up"]
        r["cover25"] = r["rare_2025_stock"]["down"]
        r["cover25_gray"] = r["rare_2025_stock"]["gray"]
        r["cover26_first"] = r["rare_2026_stock_first"]["down"]
        r["etf_cover26"] = r["rare_2026_etf"]["down"]
        r["etf_cover26_gray"] = r["rare_2026_etf"]["gray"]
        r["etf_cover25"] = r["rare_2025_etf"]["down"]
        for h, per in (("h1", TC.PER["2026H1"]), ("h2", TC.PER["2026H2"])):
            r[f"rare_2026{h}_stock"] = cover(z, rare["2026"] & stock & inper(dates, per))
            r[f"cover26_{h}"] = r[f"rare_2026{h}_stock"]["down"]
        # 덮어쓰기 발동
        if k != "B0":
            ch = (z != df["z_B0"].to_numpy()) & valid
            for grp, gm in (("stock", stock), ("etf", etf)):
                M = gm & valid & in26
                sy = M.sum() / 252.0
                r[f"fire_{grp}"] = {"changed_days_per_sy": float((ch & M).sum() / sy),
                                    "to_down_per_sy": float((ch & M & (z == -1)).sum() / sy),
                                    "to_up_per_sy": float((ch & M & (z == 1)).sum() / sy),
                                    "changed_share": float((ch & M).sum() / M.sum())}
            # 덮어쓰기 조건 에피소드(2026, 종목별 연환산 중앙) — 하락 조건
            eps26, eps_hist = [], []
            cdn = dthr[k]
            for s in stocks:
                rr = sym_rows[s]
                for per, store, minn in ((P_2026, eps26, 150), (H_2026, eps_hist, MIN_HIST)):
                    w = inper(dates[rr], per) & np.isfinite(dd20[rr]) & np.isfinite(cdn[rr])
                    if w.sum() < minn:
                        continue
                    a = np.minimum(cdn[rr][w], -1e-12)
                    t_, rc = thr_from_log(a)
                    n, _, _ = episodes(dd20[rr][w][None, :], t_[None, :], rc[None, :])
                    store.append(n[0] / (w.sum() / 252.0))
            r["fire_eps_median"] = float(np.median(eps26))
            r["fire_eps_iqr"] = [float(np.percentile(eps26, 25)), float(np.percentile(eps26, 75))]
            r["fire_eps_n_stocks"] = len(eps26)
            r["fire_eps_hist_median"] = float(np.median(eps_hist))
            r["fire_days_per_sy"] = r["fire_stock"]["changed_days_per_sy"]
            # 오경보: 덮어쓰기로 하방이 된 날(B0는 0·+1)
            Mf = stock & valid & in26 & (z == -1) & (df["z_B0"].to_numpy() >= 0) & fut_ok
        else:
            r["fire_days_per_sy"] = 0.0
            r["fire_eps_median"] = 0.0
            Mf = np.zeros(len(df), bool)                              # 덮어쓰기 없음 → 해당 없음(NaN)
        r["false_rare"] = float(fut_newhi[Mf].mean()) if Mf.sum() else float("nan")
        r["false_rare_full_recover"] = float(fut_recover[Mf].mean()) if Mf.sum() else float("nan")
        r["false_rare_n"] = int(Mf.sum())
        Mb = stock & valid & in26 & (df["z_B0"].to_numpy() == -1) & fut_ok
        r["false_ref_B0_down"] = float(fut_newhi[Mb].mean())
        # 기존 지표
        tab = TC.agree_tab(df, st, S_all)
        r["score"] = TC.score_of(tab, ("2026H1", "2026H2"))
        r["score_2026H1"] = TC.score_of(tab, ("2026H1",))
        r["score_2026H2"] = TC.score_of(tab, ("2026H2",))
        r["score_2025"] = TC.score_of(tab, ("2025",))
        r["opp26_mean"] = float(np.mean([tab[th][p]["opp"] for th in TC.THETAS for p in ("2026H1", "2026H2")]))
        fd = TC.flips_direct(df, st, S_ly)
        r["flips"], r["direct"] = fd["flips_py"], fd["direct_py"]
        zz = df[st]
        r["steady_up"] = float(np.mean(np.sign(zz[S_inall & stU]) > 0))
        r["steady_dn"] = float(np.mean(np.sign(zz[S_inall & stD]) < 0))
        r["gray_2026"] = float(np.mean(zz[S_all & inper(df["date"], P_2026)] == 0))
        etab = TC.agree_tab(df, st, E_all)
        r["etf_score"] = TC.score_of(etab, ("2026H1", "2026H2"))
        efd = TC.flips_direct(df, st, E_ly)
        r["etf_flips"], r["etf_direct"] = efd["flips_py"], efd["direct_py"]
        # 예측
        tmp = pd.DataFrame({"d": df["date"], "sym": df["sym"], "z": zz})[stock]
        Zp = tmp.pivot(index="d", columns="sym", values="z").reindex(index=I.dates, columns=I.syms).fillna(-9)
        Zm = Zp.to_numpy().astype(np.int8)
        if k == "B0":
            Zb0 = Zm.copy()
        else:
            # 덮어쓰기로 바뀐 칸(B0 0·+1 → −1)의 이후 10일(t+1 시가 기준) 초과·원수익 — 하방 표시가 맞았나
            chm = (Zm == -1) & (Zb0 >= 0)
            fwd = {}
            for p, lab in (("앞", "front"), ("뒤", "back"), ("2025", "2025")):
                rows_ = prow[p]
                M = chm[rows_] & np.isfinite(F.ex[rows_])
                ex_, rw_ = F.ex[rows_][M], F.raw[rows_][M]
                fwd[lab] = {"n": int(M.sum()), "ex_mean_pct": float(ex_.mean() * 100) if M.any() else None,
                            "ex_median_pct": float(np.median(ex_) * 100) if M.any() else None,
                            "up_share": float(np.mean(rw_ > 0)) if M.any() else None}
            r["overwrite_fwd10"] = fwd
        for p, lab in (("앞", "front"), ("뒤", "back"), ("2025", "2025")):
            ev = ZM.evaluate(Zm, F, prow[p], full=True)
            r[f"S_{lab}"] = ev["S"] * 100
            r[f"T1_{lab}"] = ev["terms"]["T1 상방-하방"] * 100
            r[f"pred_{lab}"] = {"terms": {kk: vv * 100 for kk, vv in ev["terms"].items()}, "t": ev["t"],
                                "share": {str(a): b for a, b in ev["share"].items()}}
        # 사례 단계·기간 하방 일수
        for sym in CASE_SYMS:
            row = df[(df.sym == sym) & (df.date == "2026-10-12")].iloc[0]
            r[f"{sym.lower()}_level"] = int(row[st])
        for sym, p in (("MU", ("2026-06-26", "2026-07-28")), ("GLW", ("2026-07-02", "2026-07-14"))):
            t = df[(df.sym == sym) & inper(df["date"], p)]
            r[f"{sym.lower()}_jul_down_days"] = int(((t[st] < 0) & (t[st] != -9)).sum())
            r[f"{sym.lower()}_jul_n_days"] = int(len(t))
            r[f"{sym.lower()}_jul_levels"] = dict(zip(t["date"], t[st].astype(int)))
        res["cands"][k] = r
        print(k, "done", {kk: round(r[kk], 4) if isinstance(r[kk], float) else r[kk]
                          for kk in ("cover26", "cover26_gray", "cover25", "flips", "direct", "score", "steady_up",
                                     "false_rare", "fire_days_per_sy", "fire_eps_median")})

    # ---------------------------------------------------------------- 4. 사례 문턱 가격(10/12)
    def solve_price(fn, lo, hi):
        """fn(c) True인 가장 큰 c (fn은 c가 낮을수록 True 쪽). 격자 후 이분."""
        cs = np.linspace(lo, hi, 1201)
        f = np.array([fn(c) for c in cs])
        if not f.any():
            return float("nan")
        i = int(np.flatnonzero(f).max())
        if i == len(cs) - 1:
            return float(cs[-1])
        a, b = cs[i], cs[i + 1]
        for _ in range(60):
            mid = (a + b) / 2
            if fn(mid):
                a = mid
            else:
                b = mid
        return float(a)

    def rsi_last(cl):
        return TC.rsi_pine(cl)[-1]

    for sym in CASE_SYMS:
        r = sym_rows[sym]
        c_all = C[r].copy()
        last = r[-1]
        prevC = c_all[:-1]
        lr_prev = np.diff(np.log(prevC))
        d20prev = dd20[r][:-1]

        def sd252_of(c):
            x = np.r_[lr_prev[-251:], np.log(c / prevC[-1])]
            return np.std(x, ddof=1) if len(x) == 252 else np.nan

        def dd_of(c, look=19):
            return np.log(c / max(prevC[-look:].max(), c))

        def t3_of(c):
            lc = np.log(np.r_[prevC[-19:], c])
            lrr = np.r_[lr_prev[-19:], np.log(c / prevC[-1])]
            return (lc[-1] - lc.mean()) / (np.std(lrr, ddof=1) * K20)

        def pdd_of(c, q):
            win = np.r_[d20prev[-251:], dd_of(c)]
            return np.percentile(win, q, method="linear")

        hiC = prevC[-19:].max()
        lo, hi = 0.3 * hiC, 1.2 * hiC
        info = {"close": float(C[last]), "H20": float(H20[last]), "dd20_pct": float((np.exp(dd20[last]) - 1) * 100),
                "sd252": float(df.at[last, "sd252"]), "rsi": float(R[last]), "T3_20": float(df.at[last, "T3_20"]),
                "X4_2026": X4["2026"].get(sym), "X4_2026_price": (float(H20[last] * (1 - X4["2026"][sym]))
                                                                 if sym in X4["2026"] and np.isfinite(X4["2026"][sym]) else None),
                "levels": {k: res["cands"][k][f"{sym.lower()}_level"] for k in CANDS}}
        info["B0_down_price"] = solve_price(lambda c: t3_of(c) <= -0.5, lo, hi)
        info["rsi30_price"] = solve_price(lambda c: rsi_last(np.r_[prevC, c]) <= 30, lo, hi)
        for k, mm in (("R1", m_star), ("R5", 1.25)):
            info[f"{k}_trigger_price"] = solve_price(lambda c: dd_of(c) <= -mm * sd252_of(c) * SQ10, lo, hi)
            info[f"{k}_trigger_static"] = float(H20[last] * np.exp(-mm * u252[last])) if np.isfinite(u252[last]) else None
            info[f"{k}_thr_pct"] = float((np.exp(-mm * u252[last]) - 1) * 100) if np.isfinite(u252[last]) else None
        info["R3_trigger_price"] = solve_price(lambda c: dd_of(c) <= pdd_of(c, p_star), lo, hi)
        info["R3_trigger_static"] = float(H20[last] * np.exp(Pd[last])) if np.isfinite(Pd[last]) else None
        info["R3_thr_pct"] = float((np.exp(Pd[last]) - 1) * 100) if np.isfinite(Pd[last]) else None
        # 다음 거래일(10/13) 종가 문턱: 20일 창이 하루 밀린다(오늘 포함 최근 19개 + 내일)
        allC = c_all
        lr_all = np.diff(np.log(allC))

        def sd252_next(c):
            x = np.r_[lr_all[-251:], np.log(c / allC[-1])]
            return np.std(x, ddof=1)

        def dd_next(c):
            return np.log(c / max(allC[-19:].max(), c))

        def pdd_next(c, q):
            win = np.r_[dd20[r][-251:], dd_next(c)]
            return np.percentile(win, q, method="linear")
        hiN = allC[-19:].max()
        for k, mm in (("R1", m_star), ("R5", 1.25)):
            info[f"{k}_trigger_next"] = solve_price(lambda c: dd_next(c) <= -mm * sd252_next(c) * SQ10, 0.3 * hiN, 1.2 * hiN)
        info["R3_trigger_next"] = solve_price(lambda c: dd_next(c) <= pdd_next(c, p_star), 0.3 * hiN, 1.2 * hiN)
        res["cases"][sym] = info
        print(sym, json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in info.items()}, ensure_ascii=False))
    for k in CANDS:
        rr = res["cands"][k]
        rr["soxx_trigger_price"] = (res["cases"]["SOXX"].get(f"{'R1' if k in ('R1', 'R2') else 'R3' if k in ('R3', 'R4') else 'R5'}_trigger_price")
                                    if k != "B0" else res["cases"]["SOXX"]["B0_down_price"])

    # X4 요약
    xs26 = [v for s, v in X4["2026"].items() if s in stocks and np.isfinite(v)]
    xs25 = [v for s, v in X4["2025"].items() if s in stocks and np.isfinite(v)]
    res["x4"] = {"2026_stock_median": float(np.median(xs26)), "2026_stock_iqr": [float(np.percentile(xs26, 25)), float(np.percentile(xs26, 75))],
                 "2026_n_stock": len(xs26), "2025_stock_median": float(np.median(xs25)), "2025_n_stock": len(xs25),
                 "2026_etf": {s: X4["2026"].get(s) for s in etfs}, "2025_etf": {s: X4["2025"].get(s) for s in etfs},
                 "2026_case": {s: X4["2026"].get(s) for s in CASE_SYMS}}
    # 2026 실제 rare 에피소드 연환산(주식 중앙)
    ep26 = []
    for s in stocks:
        X = X4["2026"].get(s)
        if X is None or not np.isfinite(X):
            continue
        rr = sym_rows[s]
        w = inper(dates[rr], P_2026) & np.isfinite(dd20[rr])
        n, _, _ = episodes(dd20[rr][w][None, :], np.log(1 - X), np.log(1 - X / 2))
        ep26.append(n[0] / (w.sum() / 252.0))
    res["x4"]["2026_realized_eps_per_year_median"] = float(np.median(ep26))
    # 참고: 드문 낙폭 날 자체의 10일 안 회복 비율(지표와 무관한 기준선)
    Mr = rare["2026"] & stock & fut_ok
    res["x4"]["rare26_fut_newhi20"] = float(fut_newhi[Mr].mean())
    res["x4"]["rare26_fut_full_recover"] = float(fut_recover[Mr].mean())
    res["x4"]["rare26_fut_n"] = int(Mr.sum())
    # 민감도: 백분위 보간을 순위 (p/100)·N − 0.5 (위키백과 C=1/2 변형)로 바꾸면 R3가 어떻게 되나(p* 고정)
    Pd_alt = np.full(len(df), np.nan)
    for s, rr in sym_rows.items():
        Ssort = roll_pct_sorted(dd20[rr])
        pos = min(max(p_star / 100.0 * 252 - 0.5, 0.0), 251.0)
        lo_ = int(np.floor(pos))
        hi_ = min(lo_ + 1, 251)
        Pd_alt[rr] = Ssort[:, lo_] + (pos - lo_) * (Ssort[:, hi_] - Ssort[:, lo_])
    with np.errstate(invalid="ignore"):
        z_alt = np.where(valid, ow(b0, dd20 <= Pd_alt), -9)
    zr3 = df["z_R3"].to_numpy()
    res["sensitivity_R3_interp_half"] = {
        "cover26": float(np.mean(z_alt[rare["2026"] & stock] < 0)),
        "cells_differ_from_R3_stock_all": int(((z_alt != zr3) & stock & valid).sum()),
        "stock_valid_cells": int((stock & valid).sum()),
        "median_thr_gap_logpt": float(np.nanmedian(np.abs(Pd_alt - Pd)[stock & in26])),
        "flips": TC.flips_direct(df.assign(z_alt=z_alt), "z_alt", S_ly)["flips_py"],
        "direct": TC.flips_direct(df.assign(z_alt=z_alt), "z_alt", S_ly)["direct_py"]}
    print("R3 보간 민감도", res["sensitivity_R3_interp_half"])
    print("드문 낙폭 날 기준선(10일 안 20일 신고 종가 / 완전 회복):", res["x4"]["rare26_fut_newhi20"], res["x4"]["rare26_fut_full_recover"])
    res["meta"] = {"panel": os.path.abspath(PANEL), "n_stock": len(stocks), "n_etf": len(etfs),
                   "rare_cells_2026_stock": int((rare["2026"] & stock).sum()), "rare_cells_2025_stock": int((rare["2025"] & stock).sum()),
                   "rare_eps_2026_stock": int((rare_start["2026"] & stock).sum()),
                   "rare_cells_2026_etf": int((rare["2026"] & etf).sum()),
                   "valid_rows_stock_2026": int((stock & valid & in26).sum()),
                   "grids": {"X": [float(XGRID[0]), float(XGRID[-1]), 0.0025], "m": [float(MGRID[0]), float(MGRID[-1]), 0.05],
                             "p": [float(PGRID[0]), float(PGRID[-1]), 0.5]},
                   "percentile": "numpy linear: 정렬 창 순위 (p/100)(N-1) 선형보간"}

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(clean(res), f, ensure_ascii=False, indent=1)
    keys = ["param", "cover26", "cover26_gray", "cover26_up", "cover26_first", "cover26_h1", "cover26_h2", "cover25", "cover25_gray", "etf_cover26",
            "etf_cover26_gray", "etf_cover25", "fire_days_per_sy", "fire_eps_median", "false_rare", "false_rare_full_recover",
            "false_rare_n", "flips", "direct", "steady_up", "steady_dn", "gray_2026", "score", "score_2026H1", "score_2026H2",
            "score_2025", "opp26_mean", "S_front", "S_back", "S_2025", "T1_front", "T1_back", "T1_2025", "etf_score",
            "etf_flips", "etf_direct", "soxx_level", "ewy_level", "smh_level", "mu_level", "sndk_level", "nvda_level",
            "soxx_trigger_price", "mu_jul_down_days", "glw_jul_down_days"]
    lines = ["key".ljust(26) + "".join(k.rjust(11) for k in CANDS)]
    for key in keys:
        row = []
        for k in CANDS:
            v = res["cands"][k].get(key)
            row.append((f"{v:.4f}" if isinstance(v, float) else str(v)).rjust(11))
        lines.append(key.ljust(26) + "".join(row))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "results.txt"), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
