"""R3 반박 검증 (rare_v, 연구용). R3 = B0 + (RSI 극단 아닐 때) dd20 ≤ P12.5(dd20, 252일, 오늘 포함) → −1.

독립 재구현: 백분위는 pandas rolling.quantile(linear), X4·에피소드는 날짜 순 상태기계(격자 벡터)로 다시 짠다.
turn_A/turn_compare.py에서는 per_symbol(B0 입력·지그재그·60일 회귀), stage5, flips_direct만 가져온다.
실행: PYTHONIOENCODING=utf-8 py refute_r3.py   결과: results.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "turn_A"))
import turn_compare as TC  # noqa: E402

P_STAR = 12.5
PER = {"2024": ("2024-01-02", "2024-12-31"), "2025": ("2025-01-02", "2025-12-31"),
       "2026H1": ("2026-01-02", "2026-05-29"), "2026H2": ("2026-06-01", "2026-10-12")}
HIST = {"2024": ("2023-01-03", "2023-12-29"), "2025": ("2023-03-01", "2024-12-31"), "2026": ("2024-01-02", "2025-12-31")}
XGRID = np.round(np.arange(0.0025, 0.8000001, 0.0025), 4)
H = 10


def inper(d, p):
    return (d >= p[0]) & (d <= p[1])


def ep_count(dd, thr, rec):
    """상태기계. dd (n,), thr/rec (g, n) 또는 (g,1). NaN 날은 상태 유지. 반환 (g,) 시작 횟수, (g, n) 상태."""
    g = thr.shape[0]
    n = len(dd)
    thr = np.broadcast_to(thr, (g, n))
    rec = np.broadcast_to(rec, (g, n))
    st = np.zeros(g, bool)
    cnt = np.zeros(g, int)
    S = np.zeros((g, n), bool)
    for t in range(n):
        x = dd[t]
        if np.isfinite(x):
            th, rc = thr[:, t], rec[:, t]
            ok = np.isfinite(th) & np.isfinite(rc)
            new = ok & ~st & (x <= th)
            cnt += new
            st = np.where(ok & ~st & (x <= th), True, np.where(ok & st & (x > rc), False, st))
        S[:, t] = st
    return cnt, S


def x4(dd):
    dd = dd[np.isfinite(dd)]
    yrs = len(dd) / 252.0
    thr = np.log(1 - XGRID)[:, None]
    rec = np.log(1 - XGRID / 2)[:, None]
    c, _ = ep_count(dd, thr, rec)
    ok = c / yrs >= 4
    return float(XGRID[ok].max()) if ok.any() else float("nan")


def runs(mask):
    """True 구간 (시작, 끝) 목록(끝 포함)."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def main():
    raw = pd.read_parquet(TC.PANEL)
    raw["ts"] = pd.to_datetime(raw["ts"])
    parts, pivs = [], {}
    for sym, g in raw.groupby("sym", sort=True):
        o, p = TC.per_symbol(g)
        C = o["C"].to_numpy()
        cs = pd.Series(C)
        o["H20"] = cs.rolling(20, min_periods=20).max().to_numpy()
        o["L20"] = cs.rolling(20, min_periods=20).min().to_numpy()
        o["dd20"] = np.log(C / o["H20"].to_numpy())
        o["ru20"] = np.log(C / o["L20"].to_numpy())
        # 20일 안 최저 종가가 며칠 전인가(0 = 오늘)
        lowpos = cs.rolling(20, min_periods=20).apply(lambda w: 19 - int(np.argmin(w)), raw=True)
        o["days_since_low"] = lowpos.to_numpy()
        o["Pd"] = pd.Series(o["dd20"].to_numpy()).rolling(252, min_periods=252).quantile(P_STAR / 100, interpolation="linear").to_numpy()
        o["Pd504"] = pd.Series(o["dd20"].to_numpy()).rolling(504, min_periods=504).quantile(P_STAR / 100, interpolation="linear").to_numpy()
        o["sd252"] = pd.Series(o["lr"].to_numpy()).rolling(252, min_periods=252).std(ddof=1).to_numpy()
        lc = np.log(C)
        n = len(C)
        f10 = np.full(n, np.nan)
        f10[:n - H] = lc[H:] - lc[:n - H]
        o["f10"] = f10
        # 앞으로 10일 안 더 낮은 종가 없음 / 20일 최고 종가 새로 씀 / 오늘 H20 회복
        fmin = np.full(n, np.nan)
        newhi = np.zeros(n, bool)
        rec = np.zeros(n, bool)
        dd = o["dd20"].to_numpy()
        h20 = o["H20"].to_numpy()
        for t in range(n - H):
            fmin[t] = C[t + 1:t + H + 1].min()
            newhi[t] = (dd[t + 1:t + H + 1] >= -1e-12).any()
            rec[t] = np.isfinite(h20[t]) and (C[t + 1:t + H + 1] >= h20[t]).any()
        o["no_lower10"] = np.where(np.isfinite(fmin), fmin > C, np.nan)
        o["newhi10"] = newhi
        o["recover10"] = rec
        o["fut_ok"] = np.isfinite(fmin)
        parts.append(o)
        pivs[sym] = p
    df = pd.concat(parts, ignore_index=True)
    off = 0
    sym_rows = {}
    for sym, g in df.groupby("sym", sort=True):
        sym_rows[sym] = g.index.to_numpy()
    piv_rows = {s: {k: [sym_rows[s][i] for i in v] for k, v in pivs[s].items()} for s in sym_rows}
    is_etf = df.groupby("sym")["is_etf"].first().to_dict()
    stocks = sorted(s for s in sym_rows if not is_etf[s])
    etfs = sorted(s for s in sym_rows if is_etf[s])
    dates = df["date"].to_numpy()
    R = df["rsi"].to_numpy()
    dd20 = df["dd20"].to_numpy()
    Pd = df["Pd"].to_numpy()
    T3 = df["T3_20"].to_numpy()
    valid = df["valid"].to_numpy()
    stock = ~df["is_etf"].to_numpy()
    etf = ~stock

    b0 = TC.stage5(R, T3, 0.5)
    ext = (R >= 70) | (R <= 30)
    with np.errstate(invalid="ignore"):
        cd = dd20 <= Pd
    r3 = np.where(ext, b0, np.where(cd, -1, b0))
    df["z_B0"] = np.where(valid, b0, -9)
    df["z_R3"] = np.where(valid, r3, -9)
    ow = valid & (r3 != b0)                       # 덮어쓰기가 단계를 바꾼 칸
    df["ow"] = ow

    # 횡단면 초과수익(주식, 같은 날 평균 대비)
    f10 = df["f10"].to_numpy()
    mkt = df[stock & np.isfinite(f10)].groupby("date")["f10"].mean()
    df["ex10"] = df["f10"] - df["date"].map(mkt)
    ex10 = df["ex10"].to_numpy()

    res = {"meta": {}, "repro": {}, "late": {}, "calib": {}, "period": {}, "noncirc": {}, "cases": {}, "mitig": {}}

    # ------------------------------------------------ X4·rare (재구현)
    X4 = {lab: {} for lab in HIST}
    for lab, hp in HIST.items():
        minn = 200 if lab == "2024" else 400
        for s, r in sym_rows.items():
            w = inper(dates[r], hp) & np.isfinite(dd20[r])
            if w.sum() >= minn:
                X4[lab][s] = x4(dd20[r][w])
    rare = {}
    rare_phase = {}                              # 1 첫날, 2 깊어지는 중(아직 바닥 전·당일 바닥), 3 바닥 뒤
    for per, (a, b) in PER.items():
        lab = per[:4]
        m = np.zeros(len(df), bool)
        ph = np.zeros(len(df), np.int8)
        for s, X in X4[lab].items():
            if not np.isfinite(X):
                continue
            r = sym_rows[s]
            inp = inper(dates[r], (a, b))
            rr = r[inp]
            c, S = ep_count(dd20[rr], np.full((1, len(rr)), np.log(1 - X)), np.full((1, len(rr)), np.log(1 - X / 2)))
            m[rr] = (dd20[rr] <= np.log(1 - X)) & valid[rr]
            # 에피소드별 바닥(에피소드 안 최저 dd20 날)
            for s0, e0 in runs(S[0]):
                seg = dd20[rr][s0:e0 + 1]
                tr = s0 + int(np.nanargmin(seg))
                ph[rr[s0]] = 1
                ph[rr[s0 + 1:tr + 1]] = 2
                ph[rr[tr + 1:e0 + 1]] = 3
        rare[per] = m
        rare_phase[per] = ph
    rare26 = rare["2026H1"] | rare["2026H2"]

    def cov(z, M):
        v = z[M]
        return {"n": int(M.sum()), "down": float(np.mean(v < 0)) if M.any() else None,
                "gray": float(np.mean(v == 0)) if M.any() else None, "up": float(np.mean(v > 0)) if M.any() else None}

    res["repro"] = {"cover26_B0": cov(b0, rare26 & stock), "cover26_R3": cov(r3, rare26 & stock),
                    "x4_2026_median": float(np.nanmedian([X4["2026"][s] for s in stocks if s in X4["2026"]])),
                    "rare26_cells": int((rare26 & stock).sum())}
    print("repro", res["repro"])

    # ------------------------------------------------ (1) 늦게 켜짐: 덮어쓰기 칸의 위치·이후 10일
    dsl = df["days_since_low"].to_numpy()
    H20 = df["H20"].to_numpy()
    L20 = df["L20"].to_numpy()
    C = df["C"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        retr = (np.log(C) - np.log(L20)) / (np.log(H20) - np.log(L20))
    bounced = (dsl >= 3) & (retr >= 1 / 3)
    nolow = df["no_lower10"].to_numpy() == 1
    fut = df["fut_ok"].to_numpy()
    newhi = df["newhi10"].to_numpy()
    recov = df["recover10"].to_numpy()
    lz10 = df["lab_Z10"].to_numpy()
    lzv2 = df["lab_ZV2"].to_numpy()

    def late_block(M):
        Mf = M & fut & np.isfinite(ex10)
        e = ex10[Mf]
        f = f10[Mf]
        return {"n": int(M.sum()), "n_fut": int(Mf.sum()),
                "bounced_share": float(np.mean(bounced[M])) if M.any() else None,
                "days_since_low_med": float(np.nanmedian(dsl[M])) if M.any() else None,
                "retr_med": float(np.nanmedian(retr[M])) if M.any() else None,
                "B0_up_share": float(np.mean(b0[M] > 0)) if M.any() else None,
                "no_lower_close_next10": float(np.mean(nolow[Mf])) if Mf.any() else None,
                "newhi20_in10": float(np.mean(newhi[Mf])) if Mf.any() else None,
                "full_recover_in10": float(np.mean(recov[Mf])) if Mf.any() else None,
                "f10_mean_pct": float(np.mean(f) * 100) if Mf.any() else None,
                "f10_q": [float(np.percentile(f, q) * 100) for q in (10, 25, 50, 75, 90)] if Mf.any() else None,
                "f10_up_share": float(np.mean(f > 0)) if Mf.any() else None,
                "ex10_mean_pct": float(np.mean(e) * 100) if Mf.any() else None,
                "ex10_median_pct": float(np.median(e) * 100) if Mf.any() else None,
                "z10_upswing_share": float(np.mean(lz10[M] == 1)) if M.any() else None,
                "z10_dnswing_share": float(np.mean(lz10[M] == -1)) if M.any() else None,
                "zv2_upswing_share": float(np.mean(lzv2[M] == 1)) if M.any() else None,
                "zv2_dnswing_share": float(np.mean(lzv2[M] == -1)) if M.any() else None}

    grp = {"overwrite(R3−1,B0≥0)": ow, "B0_down(−1)": valid & (b0 == -1), "B0_gray_kept(R3 0)": valid & (r3 == 0),
           "all_valid": valid}
    for per, p in PER.items():
        Mp = inper(dates, p) & stock
        res["late"][per] = {k: late_block(v & Mp) for k, v in grp.items()}
    Mall = (inper(dates, ("2024-01-02", "2026-10-12"))) & stock
    res["late"]["ALL"] = {k: late_block(v & Mall) for k, v in grp.items()}
    # 덮어쓰기 첫날(구간 시작)만
    first_ow = np.zeros(len(df), bool)
    for s, r in sym_rows.items():
        for a, b in runs(ow[r]):
            first_ow[r[a]] = True
    res["late"]["ALL_first_day"] = late_block(first_ow & Mall)
    res["late"]["2026_first_day"] = late_block(first_ow & stock & inper(dates, ("2026-01-02", "2026-10-12")))
    # 덮어쓰기 칸 가운데 이미 반등(bounced) vs 아님의 이후 10일
    M26 = stock & inper(dates, ("2026-01-02", "2026-10-12"))
    res["late"]["2026_ow_bounced"] = late_block(ow & M26 & bounced)
    res["late"]["2026_ow_notbounced"] = late_block(ow & M26 & ~bounced)
    res["late"]["2026_ow_from_B0up"] = late_block(ow & M26 & (b0 == 1))
    # rare 날의 단계(국면별): 첫날·깊어지는 중·바닥 뒤
    ph26 = rare_phase["2026H1"] | rare_phase["2026H2"]
    res["late"]["rare26_phase"] = {}
    for code, name in ((1, "first"), (2, "deepening"), (3, "after_trough")):
        M = rare26 & stock & (ph26 == code)
        res["late"]["rare26_phase"][name] = {"n": int(M.sum()), "B0": cov(b0, M), "R3": cov(r3, M),
                                            "ow_share": float(np.mean(ow[M])) if M.any() else None}
    # 덮어쓰기 칸이 rare 에피소드 어느 국면인가
    Mo = ow & M26
    res["late"]["ow26_phase_share"] = {name: float(np.mean(ph26[Mo] == code)) for code, name in
                                       ((0, "not_rare_episode"), (1, "first"), (2, "deepening"), (3, "after_trough"))}
    print("late ALL ow", res["late"]["ALL"]["overwrite(R3−1,B0≥0)"])
    print("late ALL B0down", res["late"]["ALL"]["B0_down(−1)"])

    # ------------------------------------------------ (2) 보정 편차: 종목별 연 에피소드(조건 기준)·실제 발동
    def cond_eps(s, p, minn):
        r = sym_rows[s]
        w = inper(dates[r], p) & np.isfinite(dd20[r]) & np.isfinite(Pd[r])
        if w.sum() < minn:
            return None
        a = np.minimum(Pd[r][w], -1e-12)
        X = 1 - np.exp(a)
        c, _ = ep_count(dd20[r][w], a[None, :], np.log(1 - X / 2)[None, :])
        return float(c[0] / (w.sum() / 252.0))

    rows = []
    for s in stocks + etfs:
        r = sym_rows[s]
        e_h = cond_eps(s, HIST["2026"], 400)
        e_26 = cond_eps(s, ("2026-01-02", "2026-10-12"), 150)
        w26 = inper(dates[r], ("2026-01-02", "2026-10-12")) & valid[r]
        ow_runs = len(runs(ow[r] & w26))
        yrs26 = w26.sum() / 252.0
        thr26 = np.nanmedian(np.exp(Pd[r][w26]) - 1) if w26.any() else np.nan
        sd_end25 = df.loc[r[dates[r] <= "2025-12-31"], "sd252"].iloc[-1] if (dates[r] <= "2025-12-31").any() else np.nan
        rows.append({"sym": s, "etf": bool(is_etf[s]), "eps_hist": e_h, "eps_2026": e_26,
                     "ow_runs_py_2026": ow_runs / yrs26 if yrs26 else None, "ow_days_py_2026": float(ow[r][w26].sum() / yrs26) if yrs26 else None,
                     "thr26_med_pct": float(thr26 * 100) if np.isfinite(thr26) else None,
                     "X4_2026_pct": float(X4["2026"].get(s, np.nan) * 100) if np.isfinite(X4["2026"].get(s, np.nan)) else None,
                     "sd252_2025end": float(sd_end25) if np.isfinite(sd_end25) else None})
    cal = pd.DataFrame(rows)
    st_cal = cal[~cal.etf].dropna(subset=["eps_hist", "sd252_2025end"])
    st_cal = st_cal.assign(vt=pd.qcut(st_cal.sd252_2025end, 3, labels=["저변동", "중변동", "고변동"]))

    def q(x):
        x = pd.Series(x).dropna()
        return {"n": int(len(x)), "min": float(x.min()), "p10": float(x.quantile(.1)), "p25": float(x.quantile(.25)),
                "med": float(x.median()), "p75": float(x.quantile(.75)), "p90": float(x.quantile(.9)), "max": float(x.max()),
                "lt2": int((x < 2).sum()), "gt8": int((x > 8).sum())}
    res["calib"]["stock_eps_hist"] = q(st_cal.eps_hist)
    res["calib"]["stock_eps_2026"] = q(cal[~cal.etf].eps_2026)
    res["calib"]["stock_ow_runs_py_2026"] = q(cal[~cal.etf].ow_runs_py_2026)
    res["calib"]["stock_ow_days_py_2026"] = q(cal[~cal.etf].ow_days_py_2026)
    res["calib"]["by_vol"] = {str(k): {"n": int(len(g)), "sd252_med": float(g.sd252_2025end.median()),
                                       "eps_hist_med": float(g.eps_hist.median()), "eps_2026_med": float(g.eps_2026.median()),
                                       "ow_runs_py_2026_med": float(g.ow_runs_py_2026.median()),
                                       "thr26_med_pct": float(g.thr26_med_pct.median()), "X4_2026_med_pct": float(g.X4_2026_pct.median())}
                              for k, g in st_cal.groupby("vt", observed=True)}
    res["calib"]["etf"] = cal[cal.etf].round(3).to_dict(orient="records")
    sc = cal[~cal.etf].dropna(subset=["thr26_med_pct", "X4_2026_pct"])
    res["calib"]["thr_vs_x4"] = {"ratio_med": float((sc.thr26_med_pct / -sc.X4_2026_pct).median()),
                                 "ratio_p10": float((sc.thr26_med_pct / -sc.X4_2026_pct).quantile(.1)),
                                 "ratio_p90": float((sc.thr26_med_pct / -sc.X4_2026_pct).quantile(.9)),
                                 "shallower_than_x4_share": float(np.mean(-sc.thr26_med_pct < sc.X4_2026_pct)),
                                 "corr_eps26_vs_sd": float(cal[~cal.etf][["eps_2026", "sd252_2025end"]].corr().iloc[0, 1])}
    ext_cal = cal[~cal.etf].sort_values("eps_2026")
    res["calib"]["rarest_2026"] = ext_cal.head(6).round(3).to_dict(orient="records")
    res["calib"]["most_frequent_2026"] = ext_cal.tail(6).round(3).to_dict(orient="records")
    # 덮어쓰기 문턱이 얕은(> −5%) 날
    M = ow & M26
    thr_pct = np.exp(Pd) - 1
    res["calib"]["ow26_thr_shallow_lt5pct_share"] = float(np.mean(thr_pct[M] > -0.05))
    res["calib"]["ow26_thr_pct_quant"] = [float(np.nanpercentile(thr_pct[M], q) * 100) for q in (10, 25, 50, 75, 90)]
    print("calib", res["calib"]["stock_eps_hist"], res["calib"]["stock_eps_2026"], res["calib"]["by_vol"])

    # ------------------------------------------------ (3) 기간별 flips·direct·cover
    for per, p in PER.items():
        Mk = pd.Series(stock & valid & inper(dates, p))
        out = {}
        for k in ("B0", "R3"):
            fd = TC.flips_direct(df, "z_" + k, Mk)
            zz = df["z_" + k].to_numpy()
            out[k] = {"flips": fd["flips_py"], "direct": fd["direct_py"],
                      "cover": cov(zz, rare[per] & stock)["down"], "cover_gray": cov(zz, rare[per] & stock)["gray"],
                      "gray_all": float(np.mean(zz[Mk.to_numpy()] == 0)), "down_all": float(np.mean(zz[Mk.to_numpy()] < 0))}
        out["ow_days_py"] = float((ow & Mk.to_numpy()).sum() / (Mk.sum() / 252.0))
        out["rare_cells"] = int((rare[per] & stock).sum())
        out["x4_hist"] = HIST[per[:4]]
        res["period"][per] = out
        Ek = pd.Series(etf & valid & inper(dates, p))
        res["period"][per]["etf"] = {k: {"flips": TC.flips_direct(df, "z_" + k, Ek)["flips_py"],
                                         "direct": TC.flips_direct(df, "z_" + k, Ek)["direct_py"],
                                         "cover": cov(df["z_" + k].to_numpy(), rare[per] & etf)["down"]} for k in ("B0", "R3")}
    print("period", json.dumps({k: {kk: v[kk] for kk in ("B0", "R3")} for k, v in res["period"].items()}, ensure_ascii=False)[:1500])

    # 직행(direct) 분해: R3에서 새로 생긴 직행이 어떤 전환인가
    def direct_events(z, M):
        ev = []
        for s in stocks:
            r = sym_rows[s]
            r = r[M[r]]
            if len(r) == 0:
                continue
            zz = np.sign(z[r])
            idx = df["idx"].to_numpy()[r]
            ls, li, lrow = 0, -99, None
            for t in range(len(zz)):
                if zz[t] != 0:
                    if ls != 0 and zz[t] != ls and idx[t] - li <= 3:
                        ev.append((s, int(r[t]), int(ls), int(zz[t])))
                    ls, li, lrow = zz[t], idx[t], r[t]
        return ev
    Mly = stock & valid & inper(dates, TC.P_LASTY)
    evB = set(direct_events(b0, Mly))
    evR = set(direct_events(r3, Mly))
    newR = evR - evB
    sy = Mly.sum() / 252.0
    res["noncirc"]["direct_decomp"] = {
        "B0_py": len(evB) / sy, "R3_py": len(evR) / sy, "new_in_R3_py": len(newR) / sy, "gone_in_R3_py": len(evB - evR) / sy,
        "new_up_to_down_py": sum(1 for e in newR if e[2] == 1) / sy, "new_down_to_up_py": sum(1 for e in newR if e[2] == -1) / sy,
        "new_down_to_up_f10_med_pct": float(np.nanmedian([f10[e[1]] for e in newR if e[2] == -1]) * 100) if any(e[2] == -1 for e in newR) else None,
        "new_up_to_down_f10_med_pct": float(np.nanmedian([f10[e[1]] for e in newR if e[2] == 1]) * 100) if any(e[2] == 1 for e in newR) else None}
    print("direct decomp", res["noncirc"]["direct_decomp"])

    # ------------------------------------------------ (4) 정의와 무관한 지표
    # 4a. 단계별 이후 10일 초과수익: 하방 계열·상방 계열 평균, 스프레드(상−하), 기간별
    nc = {}
    for per, p in PER.items():
        Mp = stock & valid & inper(dates, p) & fut & np.isfinite(ex10)
        o = {}
        for k, z in (("B0", b0), ("R3", r3)):
            dn = Mp & (z < 0)
            up = Mp & (z > 0)
            gy = Mp & (z == 0)
            o[k] = {"down_ex": float(ex10[dn].mean() * 100), "gray_ex": float(ex10[gy].mean() * 100),
                    "up_ex": float(ex10[up].mean() * 100), "spread_up_minus_down": float((ex10[up].mean() - ex10[dn].mean()) * 100),
                    "down_only_m1_ex": float(ex10[Mp & (z == -1)].mean() * 100), "n_down": int(dn.sum())}
        o["ow_cells_ex"] = float(ex10[Mp & ow].mean() * 100) if (Mp & ow).any() else None
        o["ow_cells_ex_median"] = float(np.median(ex10[Mp & ow]) * 100) if (Mp & ow).any() else None
        o["B0_gray_ex_for_ref"] = o["B0"]["gray_ex"]
        o["n_ow"] = int((Mp & ow).sum())
        nc[per] = o
    res["noncirc"]["fwd_ex10"] = nc
    print("fwd", json.dumps(nc, ensure_ascii=False)[:2000])

    # 4b. 스윙 지연: Z10·ZV2 확정 스윙. 하락 스윙(고점 a → 저점 b): 첫 하방 계열까지 일수.
    #     상승 스윙(저점 a → 고점 b): 하방 계열에서 벗어날 때까지(≥0) 일수, 스윙 안 하방 계열 날 비율.
    sw = {}
    for th in ("Z10", "ZV2"):
        for per, p in list(PER.items()) + [("ALL", ("2024-01-02", "2026-10-12"))]:
            o = {"B0": {"dn_lag": [], "up_exit_lag": [], "up_down_days": 0, "up_days": 0, "dn_never": 0},
                 "R3": {"dn_lag": [], "up_exit_lag": [], "up_down_days": 0, "up_days": 0, "dn_never": 0}}
            for s in stocks:
                pv = piv_rows[s][th]
                for a, b in zip(pv[:-1], pv[1:]):
                    if not (p[0] <= dates[a + 1] <= p[1]) or not valid[a + 1:b + 1].all():
                        continue
                    d = 1 if C[b] > C[a] else -1
                    for k, z in (("B0", b0), ("R3", r3)):
                        seg = np.sign(z[a + 1:b + 1])
                        if d == -1:
                            hit = np.flatnonzero(seg < 0)
                            if len(hit):
                                o[k]["dn_lag"].append(int(hit[0]) + 1)
                            else:
                                o[k]["dn_lag"].append(b - a + 1)
                                o[k]["dn_never"] += 1
                        else:
                            hit = np.flatnonzero(seg >= 0)
                            o[k]["up_exit_lag"].append(int(hit[0]) + 1 if len(hit) else b - a + 1)
                            o[k]["up_down_days"] += int((seg < 0).sum())
                            o[k]["up_days"] += len(seg)
            sw[f"{th}_{per}"] = {k: {"n_dn": len(v["dn_lag"]), "dn_lag_mean": float(np.mean(v["dn_lag"])) if v["dn_lag"] else None,
                                     "dn_lag_med": float(np.median(v["dn_lag"])) if v["dn_lag"] else None,
                                     "dn_never": v["dn_never"] / len(v["dn_lag"]) if v["dn_lag"] else None,
                                     "n_up": len(v["up_exit_lag"]),
                                     "up_exit_lag_mean": float(np.mean(v["up_exit_lag"])) if v["up_exit_lag"] else None,
                                     "up_exit_lag_med": float(np.median(v["up_exit_lag"])) if v["up_exit_lag"] else None,
                                     "upswing_down_day_share": v["up_down_days"] / v["up_days"] if v["up_days"] else None}
                                 for k, v in o.items()}
    res["noncirc"]["swing"] = sw
    print("swing", json.dumps({k: v for k, v in sw.items() if k.endswith("ALL")}, ensure_ascii=False))

    # 4c. 꾸준한 상승 오판: 60일 R² ≥ 0.8·기울기 ≥ ln1.15/60 (스윙 라벨 무관)과 Z10 +1 둘 다
    stU_raw = (df["slope60"] * 60 >= np.log(1.15)) & (df["r2_60"] >= 0.8)
    stU_raw = stU_raw.to_numpy()
    stU = stU_raw & (lz10 == 1)
    MA = stock & valid & inper(dates, ("2024-01-02", "2026-10-12"))
    res["noncirc"]["steady"] = {
        k: {"steadyU_Z10_up_share": float(np.mean(z[MA & stU] > 0)), "steadyU_Z10_down_share": float(np.mean(z[MA & stU] < 0)),
            "steadyU_raw_down_share": float(np.mean(z[MA & stU_raw] < 0)), "n_raw": int((MA & stU_raw).sum())}
        for k, z in (("B0", b0), ("R3", r3))}
    res["noncirc"]["steady"]["ow_in_steadyU_raw"] = int((ow & MA & stU_raw).sum())
    res["noncirc"]["steady"]["ow_in_steadyU_raw_f10_med_pct"] = float(np.nanmedian(f10[ow & MA & stU_raw]) * 100) if (ow & MA & stU_raw).any() else None
    print("steady", res["noncirc"]["steady"])

    # ------------------------------------------------ (5) 사례: 2026 덮어쓰기 구간(시작일 기준) 최악·최선
    cases = []
    for s in stocks + etfs:
        r = sym_rows[s]
        w = inper(dates[r], ("2026-01-02", "2026-10-12"))
        for a, b in runs(ow[r] & w):
            i0, i1 = r[a], r[b]
            cases.append({"sym": s, "etf": bool(is_etf[s]), "start": dates[i0], "end": dates[i1], "days": int(b - a + 1),
                          "C0": round(float(C[i0]), 2), "dd20_pct": round(float((np.exp(dd20[i0]) - 1) * 100), 2),
                          "thr_pct": round(float((np.exp(Pd[i0]) - 1) * 100), 2),
                          "X4_pct": round(float(X4["2026"].get(s, np.nan) * 100), 2) if np.isfinite(X4["2026"].get(s, np.nan)) else None,
                          "B0_at_start": int(b0[i0]), "rsi": round(float(R[i0]), 1), "T3": round(float(T3[i0]), 2),
                          "days_since_low": int(dsl[i0]), "retr": round(float(retr[i0]), 2) if np.isfinite(retr[i0]) else None,
                          "f10_pct": round(float((np.exp(f10[i0]) - 1) * 100), 2) if np.isfinite(f10[i0]) else None,
                          "ex10_pct": round(float(ex10[i0] * 100), 2) if np.isfinite(ex10[i0]) else None,
                          "newhi10": bool(newhi[i0]), "B0_path": [int(x) for x in b0[i0:i1 + 1]],
                          "after_end_B0_R3": [int(b0[i1 + 1]), int(r3[i1 + 1])] if i1 + 1 in set(r) else None})
    cdf = pd.DataFrame(cases)
    res["cases"]["n_runs_2026"] = int(len(cdf))
    res["cases"]["n_runs_2026_stock"] = int((~cdf.etf).sum())
    cs_ = cdf.dropna(subset=["f10_pct"])
    res["cases"]["runs_f10_quant"] = [float(cs_.f10_pct.quantile(q)) for q in (.1, .25, .5, .75, .9)]
    res["cases"]["runs_newhi_share"] = float(cs_.newhi10.mean())
    res["cases"]["runs_from_B0up_share"] = float((cdf.B0_at_start == 1).mean())
    res["cases"]["runs_1day_share"] = float((cdf.days == 1).mean())
    res["cases"]["worst"] = cs_.sort_values("f10_pct", ascending=False).head(10).drop(columns=["B0_path"]).to_dict(orient="records")
    res["cases"]["best"] = cs_.sort_values("f10_pct").head(6).drop(columns=["B0_path"]).to_dict(orient="records")
    res["cases"]["from_B0_up"] = cdf[cdf.B0_at_start == 1].drop(columns=["B0_path"]).head(10).to_dict(orient="records")
    focus = cdf[cdf.sym.isin(["SOXX", "EWY", "SMH", "MU", "SNDK", "NVDA", "GLW", "SPY", "QQQ"])]
    res["cases"]["focus"] = focus.drop(columns=["B0_path"]).to_dict(orient="records")
    print("cases worst", json.dumps(res["cases"]["worst"][:6], ensure_ascii=False))

    # ------------------------------------------------ 탐색(사후, 사전 등록 아님): 부작용 줄이는 변형
    X4p = np.full(len(df), np.nan)                 # 그해 X4(직전 2년 이력) 고정 문턱
    for s, r in sym_rows.items():
        for lab, p in (("2025", PER["2025"]), ("2026", ("2026-01-02", "2026-10-12"))):
            X = X4[lab].get(s, np.nan)
            if np.isfinite(X):
                w = inper(dates[r], p)
                X4p[r[w]] = np.log(1 - X)
    with np.errstate(invalid="ignore"):
        var = {
            "R3": cd,
            "R3_T3neg": cd & (T3 < 0),                                  # 20일 평균 아래일 때만
            "R3_notbounced": cd & ~bounced,                             # 이미 반등(저점 3일 이상 전·1/3 되돌림)한 날 제외
            "R3_504": dd20 <= df["Pd504"].to_numpy(),                  # 2년 창 백분위
            "X4fix": dd20 <= X4p,                                       # 사용자 정의 그대로(직전 2년 연 4번 낙폭, 해마다 고정)
            "X4fix_T3neg": (dd20 <= X4p) & (T3 < 0),
        }
    mit = {}
    for k, c in var.items():
        z = np.where(ext, b0, np.where(c, -1, b0))
        df["z_tmp"] = np.where(valid, z, -9)
        o = {}
        for per in ("2025", "2026H1", "2026H2"):
            Mk = pd.Series(stock & valid & inper(dates, PER[per]))
            fd = TC.flips_direct(df, "z_tmp", Mk)
            Mp = Mk.to_numpy() & fut & np.isfinite(ex10)
            owk = valid & (z != b0)
            o[per] = {"cover": cov(z, rare[per] & stock)["down"], "flips": fd["flips_py"], "direct": fd["direct_py"],
                      "ow_ex10_mean": float(ex10[Mp & owk].mean() * 100) if (Mp & owk).any() else None,
                      "ow_newhi10": float(np.mean(newhi[Mp & owk])) if (Mp & owk).any() else None,
                      "ow_n": int((Mp & owk).sum()),
                      "spread_up_minus_down": float((ex10[Mp & (z > 0)].mean() - ex10[Mp & (z < 0)].mean()) * 100)}
        Mly_s = pd.Series(stock & valid & inper(dates, TC.P_LASTY))
        fd = TC.flips_direct(df, "z_tmp", Mly_s)
        o["lastyear"] = {"flips": fd["flips_py"], "direct": fd["direct_py"]}
        o["cover26"] = cov(z, rare26 & stock)["down"]
        mit[k] = o
    res["mitig"] = mit
    print("mitig", json.dumps({k: {"cover26": v["cover26"], "ly": v["lastyear"]} for k, v in mit.items()}, ensure_ascii=False))

    res["meta"] = {"p_star": P_STAR, "n_stock": len(stocks), "n_etf": len(etfs), "H": H,
                   "bounced_def": "20일 최저 종가가 3거래일 이상 전 & (lnC−lnL20)/(lnH20−lnL20) ≥ 1/3",
                   "fwd": "종가→10거래일 뒤 종가 로그수익, 초과 = 같은 날 주식 평균 대비",
                   "X4_hist": HIST, "periods": PER,
                   "mitig_note": "사후 탐색 6개 변형(R3 포함), 사전 등록 아님"}

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 5)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(clean(res), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
