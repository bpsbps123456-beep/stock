"""추세 5단계 가운데 판정 대안 13개 비교 (impl_A, 독립 재현용). 저장소 파일은 읽기·import만 한다.

실행: PYTHONIOENCODING=utf-8 py compare_mid.py
결과: 같은 폴더 results.json, results.txt
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "..", "panel_1d_20261012.parquet")
ZONES_DIR = r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/trend_zones/code"
sys.path.insert(0, ZONES_DIR)
import zones as ZM  # noqa: E402  (Ind, Fwd, period_rows, evaluate만 쓴다)

DDOF = int(os.environ.get("DDOF", "1"))   # 사양: 표본 표준편차(ddof=1) = 파인 ta.stdev(..., false)
SUFFIX = "" if DDOF == 1 else f"_ddof{DDOF}"
MIN_IDX = 129                     # bar_index >= 129 → 일봉 130개째부터
P_ALL = ("2024-01-02", "2026-10-12")
P_CAL = ("2024-01-02", "2025-12-31")
P_2026 = ("2026-01-02", "2026-10-12")
P_LASTY = ("2025-10-13", "2026-10-12")
K3_MULT = np.sqrt((20 - 1) * (2 * 20 - 1) / (6 * 20))     # 2.4850

CANDS = ["K0", "K1a", "K1b", "K2n", "K2c", "K3n", "K3c", "K4c", "K5c", "K6a", "K6b", "K7", "K8"]
SEVEN = {"K1a", "K1b", "K8"}
TCOL = {"K0": "T0", "K1a": "T0", "K1b": "T0", "K2n": "T2", "K2c": "T2", "K3n": "T3", "K3c": "T3",
        "K4c": "T4", "K5c": "T5", "K6a": "T0", "K6b": "T0", "K7": "T7", "K8": "T3"}


# ---------------------------------------------------------------- 지표 계산
def rsi_pine(c, n=14):
    """파인 ta.rsi: RMA(alpha=1/n), 첫 값은 1..n 변화의 단순 평균."""
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    d = np.diff(c, prepend=np.nan)
    u = np.where(d > 0, d, 0.0)
    v = np.where(d < 0, -d, 0.0)
    au, ad = u[1:n + 1].mean(), v[1:n + 1].mean()

    def val(a, b):
        if b == 0:
            return 100.0 if a > 0 else (50.0 if a == 0 else 0.0)
        return 100 - 100 / (1 + a / b)
    out[n] = val(au, ad)
    for i in range(n + 1, len(c)):
        au = (au * (n - 1) + u[i]) / n
        ad = (ad * (n - 1) + v[i]) / n
        out[i] = val(au, ad)
    return out


def per_symbol(g):
    g = g.sort_values("ts").reset_index(drop=True)
    C = g["Close"].to_numpy(float)
    lc = np.log(C)
    s = pd.Series(lc)
    lr = s.diff()
    sd20 = lr.rolling(20, min_periods=20).std(ddof=DDOF)
    sd10 = lr.rolling(10, min_periods=10).std(ddof=DDOF)
    o = pd.DataFrame({"sym": g["sym"], "date": g["ts"].dt.strftime("%Y-%m-%d"), "is_etf": g["is_etf"],
                      "Open": g["Open"].to_numpy(float), "C": C, "idx": np.arange(len(C))})
    o["lr"] = lr.to_numpy()
    o["sd20"] = sd20.to_numpy()
    o["rsi"] = rsi_pine(C)
    o["T0"] = ((s - s.shift(20)) / (sd20 * np.sqrt(20))).to_numpy()
    o["T2"] = (sum((s - s.shift(N)) / (sd20 * np.sqrt(N)) for N in (5, 10, 15, 20)) / 4).to_numpy()
    o["T3"] = ((s - s.rolling(20, min_periods=20).mean()) / (sd20 * K3_MULT)).to_numpy()
    # OLS 기울기: 최근 20개 ln C(오래된 것 x=0 … 오늘 x=19)
    w = (np.arange(20) - 9.5) / 665.0
    b = np.full(len(C), np.nan)
    if len(C) >= 20:
        win = np.lib.stride_tricks.sliding_window_view(lc, 20)
        b[19:] = win @ w
    o["T4"] = b * 19 / sd20.to_numpy()
    ema10 = pd.Series(C).ewm(span=10, adjust=False).mean()
    o["T5"] = ((s - np.log(ema10)) / sd20).to_numpy()
    o["T7"] = ((s - s.shift(10)) / (sd10 * np.sqrt(10))).to_numpy()
    hi10 = pd.Series(C).rolling(10, min_periods=10).max()
    lo10 = pd.Series(C).rolling(10, min_periods=10).min()
    o["dn10"] = np.log(C / hi10.to_numpy())
    o["up10"] = np.log(C / lo10.to_numpy())
    o["u10"] = (sd20 * np.sqrt(10)).to_numpy()
    # 사건 정의용 창
    hi20 = pd.Series(C).rolling(20, min_periods=20).max()
    lo20 = pd.Series(C).rolling(20, min_periods=20).min()
    o["hi10"], o["lo10"], o["hi20"], o["lo20"] = hi10.to_numpy(), lo10.to_numpy(), hi20.to_numpy(), lo20.to_numpy()
    return o


def stage5(rsi, T, tau):
    return np.select([rsi >= 70, rsi <= 30, T >= tau, T <= -tau], [2, -2, 1, -1], 0)


def stage7(rsi, T, a, b):
    return np.select([rsi >= 70, rsi <= 30, T >= b, T >= a, T <= -b, T <= -a], [3, -3, 2, 1, -2, -1], 0)


def stage6(rsi, T0, dn, up, u, m):
    base = stage5(rsi, T0, 0.5)
    ext = (rsi >= 70) | (rsi <= 30)
    cd = dn <= -m * u
    cu = up >= m * u
    both = cd & cu
    over = np.where(both, np.where(np.abs(dn) > up, -1, np.where(up > np.abs(dn), 1, base)),
                    np.where(cd, -1, np.where(cu, 1, base)))
    return np.where(ext, base, over)


def calibrate(df, col, target, mask):
    v = np.abs(df.loc[mask, col].to_numpy())
    v = v[np.isfinite(v)]
    lo, hi = 0.0, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.mean(v < mid) < target:
            lo = mid
        else:
            hi = mid
    tau = (lo + hi) / 2
    return float(tau), float(np.mean(v < tau))


def inper(d, p):
    return (d >= p[0]) & (d <= p[1])


# ---------------------------------------------------------------- 지표 M1~M4
def m1(df, st, mask):
    """mask: 이번 표본(기간·종목군·유효). 연속 유효 이틀만."""
    prev = df.groupby("sym")[st].shift(1)
    pidx = df.groupby("sym")["idx"].shift(1)
    cur = df[st]
    ok = mask & (cur != -9) & (prev != -9) & prev.notna() & (pidx == df["idx"] - 1)
    ch = ok & (cur != prev)
    dirn = np.sign(cur - prev)
    rev = ch & (dirn * np.sign(df["lr"]) < 0)
    strong = rev & (df["lr"].abs() >= df["sd20"])
    sy = (mask & (cur != -9)).sum() / 252.0
    n_ch, n_rev, n_str = int(ch.sum()), int(rev.sum()), int(strong.sum())
    return {"changes": n_ch, "rev": n_rev, "strong_rev": n_str, "sym_years": float(sy),
            "rev_share": n_rev / n_ch if n_ch else float("nan"),
            "changes_per_sy": n_ch / sy, "rev_per_sy": n_rev / sy, "strong_rev_per_sy": n_str / sy}


def fam(z):
    return np.sign(z)


def m2(df, st, mask, ev):
    out = {}
    for name, e in ev.items():
        M = mask & e & (df[st] != -9)
        z = fam(df.loc[M, st].to_numpy())
        n = int(M.sum())
        out[name] = {"n": n, "down": float(np.mean(z < 0)) if n else float("nan"),
                     "gray": float(np.mean(z == 0)) if n else float("nan"),
                     "up": float(np.mean(z > 0)) if n else float("nan")}
    return out


def m3(df, st, mask):
    """2025-10-13~2026-10-12, 연 바뀜 수(종목·연 기준)와 같은 단계 연속 일수 중앙값."""
    changes, days, runs = 0, 0, []
    for sym, g in df[mask & (df[st] != -9)].groupby("sym"):
        z = g[st].to_numpy()
        idx = g["idx"].to_numpy()
        days += len(z)
        brk = np.r_[True, (np.diff(idx) != 1) | (z[1:] != z[:-1])]      # 새 연속 구간 시작
        chg = (np.diff(idx) == 1) & (z[1:] != z[:-1])
        changes += int(chg.sum())
        starts = np.flatnonzero(brk)
        runs += list(np.diff(np.r_[starts, len(z)]))
    sy = days / 252.0
    return {"flips_per_sy": changes / sy if sy else float("nan"), "median_run": float(np.median(runs)) if runs else float("nan"),
            "mean_run": float(np.mean(runs)) if runs else float("nan"), "n_runs": len(runs), "sym_years": sy}


def m4_events(df, common_valid, sample_mask, up=False):
    """급락(급등) 사건 목록: (sym, p_row, q_row, end_row) — 행 번호는 df의 위치."""
    evs = []
    for sym, g in df[sample_mask_syms(df, sample_mask)].groupby("sym", sort=False):
        pos = g.index.to_numpy()
        C = g["C"].to_numpy()
        dates = g["date"].to_numpy()
        cv = common_valid.loc[pos].to_numpy()
        n = len(C)
        cond = np.zeros(n, bool)
        for q in range(15, n):
            w = C[q - 15:q]
            cond[q] = (C[q] >= 1.15 * w.min()) if up else (C[q] <= 0.85 * w.max())
        for q in range(16, n):
            if not cond[q] or cond[q - 1]:
                continue
            if not (P_ALL[0] <= dates[q] <= P_ALL[1]):
                continue
            if q + 20 > n - 1:
                continue
            w = C[q - 15:q]
            k = (np.flatnonzero(w == w.min())[-1] if up else np.flatnonzero(w == w.max())[-1])
            p = q - 15 + k
            if not cv[p:q + 21].all():
                continue
            evs.append((sym, pos[p], pos[q], q - p))
    return evs


def sample_mask_syms(df, sample_mask):
    syms = set(df.loc[sample_mask, "sym"])
    return df["sym"].isin(syms)


def m4(df, st, evs, up=False, sentinel=36):
    lags, early, reached, zero = [], 0, 0, 0
    z = df[st].to_numpy()
    for sym, p, q, qp in evs:
        seg = z[p:q + 21]                     # p..q+20 (같은 종목 연속 행)
        hit = np.flatnonzero(seg > 0) if up else np.flatnonzero(seg < 0)
        if len(hit):
            lag = int(hit[0])
            reached += 1
            zero += lag == 0
            if lag <= qp:
                early += 1
        else:
            lag = sentinel
        lags.append(lag)
    n = len(evs)
    return {"n": n, "median_lag": float(np.median(lags)) if n else float("nan"),
            "early": early / n if n else float("nan"), "lag0_share": zero / n if n else float("nan"), "reached_by_q20": reached / n if n else float("nan"),
            "median_lag_reached": float(np.median([l for l in lags if l != sentinel])) if reached else float("nan")}


# ---------------------------------------------------------------- 실행
def main():
    raw = pd.read_parquet(PANEL)
    raw["ts"] = pd.to_datetime(raw["ts"])
    df = pd.concat([per_symbol(g) for _, g in raw.groupby("sym", sort=True)], ignore_index=True)

    # ---- K0 재현 확인
    e = df[(df.sym == "EWY") & df.date.isin(["2026-10-08", "2026-10-09", "2026-10-12"])]
    chk = {r.date: {"C": r.C, "T0": r.T0, "rsi": r.rsi} for r in e.itertuples()}
    print("EWY 확인:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in chk.items()})

    base_valid = (df["idx"] >= MIN_IDX) & np.isfinite(df["rsi"])
    nonext = (df["rsi"] > 30) & (df["rsi"] < 70)
    stock = ~df["is_etf"]
    cal_mask = stock & inper(df["date"], P_CAL) & base_valid & np.isfinite(df["T0"]) & nonext
    target = float(np.mean(np.abs(df.loc[cal_mask, "T0"]) < 0.5))
    taus = {}
    for k, col in (("K2c", "T2"), ("K3c", "T3"), ("K4c", "T4"), ("K5c", "T5")):
        m = stock & inper(df["date"], P_CAL) & base_valid & np.isfinite(df[col]) & nonext
        taus[k] = calibrate(df, col, target, m)
    print("K0 비극단 애매 비율(2024~25):", round(target, 4), "보정 τ:", taus)

    R, T = df["rsi"].to_numpy(), {c: df[c].to_numpy() for c in ("T0", "T2", "T3", "T4", "T5", "T7")}
    stg = {
        "K0": stage5(R, T["T0"], 0.5), "K1a": stage7(R, T["T0"], 0.25, 0.75), "K1b": stage7(R, T["T0"], 0.5, 1.0),
        "K2n": stage5(R, T["T2"], 0.5), "K2c": stage5(R, T["T2"], taus["K2c"][0]),
        "K3n": stage5(R, T["T3"], 0.5), "K3c": stage5(R, T["T3"], taus["K3c"][0]),
        "K4c": stage5(R, T["T4"], taus["K4c"][0]), "K5c": stage5(R, T["T5"], taus["K5c"][0]),
        "K6a": stage6(R, T["T0"], df["dn10"].to_numpy(), df["up10"].to_numpy(), df["u10"].to_numpy(), 1.0),
        "K6b": stage6(R, T["T0"], df["dn10"].to_numpy(), df["up10"].to_numpy(), df["u10"].to_numpy(), 1.5),
        "K7": stage5(R, T["T7"], 0.5), "K8": stage7(R, T["T3"], 0.25, 0.75),
    }
    need = {k: [TCOL[k]] + (["dn10", "up10", "u10"] if k in ("K6a", "K6b") else []) for k in CANDS}
    for k in CANDS:
        ok = base_valid.to_numpy() & np.all([np.isfinite(df[c].to_numpy()) for c in need[k]], axis=0)
        df["z_" + k] = np.where(ok, stg[k], -9).astype(int)
    common = np.all([df["z_" + k].to_numpy() != -9 for k in CANDS], axis=0)
    common = pd.Series(common, index=df.index) & np.isfinite(df["sd20"]) & np.isfinite(df["hi20"])
    print("유효 행(공통):", int(common.sum()), "개별 유효 차이:",
          {k: int(((df["z_" + k] != -9) & ~common).sum()) for k in CANDS})

    # 사건 정의 (지표와 무관)
    C = df["C"]
    ev = {
        "E_dn_abs": C / df["hi10"] - 1 <= -0.10,
        "E_dn_vol": np.log(C / df["hi10"]) <= -1.5 * df["sd20"] * np.sqrt(10),
        "E_dn20": (C / df["hi20"] - 1 <= -0.15) & (df["hi10"] == df["hi20"]),
        "E_up_abs": C / df["lo10"] - 1 >= 0.10,
        "E_up_vol": np.log(C / df["lo10"]) >= 1.5 * df["sd20"] * np.sqrt(10),
        "E_up20": (C / df["lo20"] - 1 >= 0.15) & (df["lo10"] == df["lo20"]),
    }

    S_all = stock & inper(df["date"], P_ALL) & common
    E_all = df["is_etf"] & inper(df["date"], P_ALL) & common
    S_ly = stock & inper(df["date"], P_LASTY) & common
    E_ly = df["is_etf"] & inper(df["date"], P_LASTY) & common

    dn_ev = m4_events(df, common, stock, up=False)
    up_ev = m4_events(df, common, stock, up=True)
    dn_ev = [e for e in dn_ev if not df.at[e[2], "is_etf"]]
    up_ev = [e for e in up_ev if not df.at[e[2], "is_etf"]]
    print("M4 사건: 급락", len(dn_ev), "급등", len(up_ev))

    # M5 준비: 150종목 피벗 + zones.py 엔진
    st_raw = raw[~raw["is_etf"]].copy()
    st_raw["d"] = st_raw["ts"].dt.strftime("%Y-%m-%d")
    piv = {k: st_raw.pivot_table(index="d", columns="sym", values=k).sort_index() for k in ("Open", "High", "Low", "Close")}
    I = ZM.Ind(piv["Open"], piv["High"], piv["Low"], piv["Close"])
    F = ZM.Fwd(I, 10)
    prow = {p: ZM.period_rows(I, p, 10) for p in ("앞", "뒤", "2025")}

    res = {"meta": {}, "cands": {}, "cases": {}}
    for k in CANDS:
        st = "z_" + k
        r = {"id": k}
        zz = df[st]
        for lab, p in (("gray_2024_25", P_CAL), ("gray_2026", P_2026)):
            M = stock & inper(df["date"], p) & common
            r[lab] = float(np.mean(zz[M] == 0))
        a = m1(df, st, S_all)
        r.update({"rev_share": a["rev_share"], "changes_per_sy": a["changes_per_sy"], "rev_per_sy": a["rev_per_sy"],
                  "strong_rev_per_sy": a["strong_rev_per_sy"], "m1_detail": a})
        b = m2(df, st, S_all, ev)
        r["m2"] = b
        r.update({"dn_abs_down": b["E_dn_abs"]["down"], "dn_abs_gray": b["E_dn_abs"]["gray"], "dn_abs_up": b["E_dn_abs"]["up"],
                  "dn_abs_n": b["E_dn_abs"]["n"], "dn_vol_down": b["E_dn_vol"]["down"], "dn_vol_gray": b["E_dn_vol"]["gray"],
                  "dn20_down": b["E_dn20"]["down"], "dn20_gray": b["E_dn20"]["gray"],
                  "up_abs_up": b["E_up_abs"]["up"], "up_abs_gray": b["E_up_abs"]["gray"], "up_abs_down": b["E_up_abs"]["down"],
                  "up_abs_n": b["E_up_abs"]["n"], "up_vol_up": b["E_up_vol"]["up"], "up20_up": b["E_up20"]["up"]})
        c = m3(df, st, S_ly)
        r.update({"flips_last_year": c["flips_per_sy"], "median_run": c["median_run"], "m3_detail": c})
        d1 = m4(df, st, dn_ev, up=False)
        d2 = m4(df, st, up_ev, up=True)
        r.update({"peak_median_lag": d1["median_lag"], "peak_early": d1["early"], "peak_n": d1["n"],
                  "trough_median_lag": d2["median_lag"], "trough_early": d2["early"], "trough_n": d2["n"],
                  "m4_detail": {"peak": d1, "trough": d2}})
        # M5
        zf = zz.to_numpy()
        if k in SEVEN:
            zf = np.where(zf == -9, -9, np.sign(zf) * np.where(np.abs(zf) == 3, 2, 1))
        tmp = pd.DataFrame({"d": df["date"], "sym": df["sym"], "z": zf})[stock.to_numpy()]
        Zp = tmp.pivot(index="d", columns="sym", values="z").reindex(index=I.dates, columns=I.syms).fillna(-9)
        Z = Zp.to_numpy().astype(np.int8)
        m5 = {}
        for p, lab in (("앞", "front"), ("뒤", "back"), ("2025", "2025")):
            ev5 = ZM.evaluate(Z, F, prow[p], full=True)
            r[f"S_{lab}"] = ev5["S"] * 100
            r[f"T1_{lab}"] = ev5["terms"]["T1 상방-하방"] * 100
            m5[lab] = {"S": ev5["S"] * 100, "terms": {kk: vv * 100 for kk, vv in ev5["terms"].items()},
                       "t": ev5["t"], "share": {str(z): s for z, s in ev5["share"].items()}}
        r["m5_detail"] = m5
        # ETF
        ea = m1(df, st, E_all)
        eb = m2(df, st, E_all, ev)
        ec = m3(df, st, E_ly)
        r.update({"etf_rev_share": ea["rev_share"], "etf_dn_abs_down": eb["E_dn_abs"]["down"], "etf_up_abs_up": eb["E_up_abs"]["up"],
                  "etf_flips": ec["flips_per_sy"],
                  "etf_detail": {"m1": ea, "m2": {kk: eb[kk] for kk in ("E_dn_abs", "E_up_abs")}, "m3": ec}})
        r["tau"] = {"K0": 0.5, "K1a": 0.25, "K1b": 0.5, "K2n": 0.5, "K3n": 0.5, "K6a": 0.5, "K6b": 0.5, "K7": 0.5,
                    "K8": 0.25}.get(k, taus.get(k, (np.nan,))[0])
        # M6 요약
        ew = df[(df.sym == "EWY")].set_index("date")
        r["ewy_1012_level"] = int(ew.at["2026-10-12", st])
        r["ewy_1012_T"] = float(ew.at["2026-10-12", TCOL[k]])
        after = ew[(ew.index > "2026-10-02") & (ew[st] < 0) & (ew[st] != -9)]
        r["ewy_first_down_after_1002"] = after.index[0] if len(after) else "없음(10/12까지)"
        mu = df[(df.sym == "MU") & inper(df["date"], ("2026-06-26", "2026-07-28"))]
        r["mu_jul_down_days"] = int(((mu[st] < 0) & (mu[st] != -9)).sum())
        r["mu_jul_up_days"] = int((mu[st] > 0).sum())
        r["mu_jul_n_days"] = int(len(mu))
        res["cands"][k] = r
        print(k, "done")

    # M6 사례 표
    cols = ["date", "C", "rsi", "T0", "T2", "T3", "T4", "T5", "T7", "dn10", "up10", "u10"] + ["z_" + k for k in CANDS]
    for sym, p in (("EWY", ("2026-09-08", "2026-10-12")), ("MU", ("2026-06-22", "2026-08-14"))):
        t = df[(df.sym == sym) & inper(df["date"], p)][cols]
        res["cases"][sym] = json.loads(t.to_json(orient="records"))

    res["meta"] = {
        "panel": os.path.abspath(PANEL), "n_stock_syms": int(df.loc[stock, "sym"].nunique()),
        "n_etf_syms": int(df.loc[df.is_etf, "sym"].nunique()),
        "k0_gray_nonext_2024_25": target, "calib": {k: {"tau": v[0], "share": v[1]} for k, v in taus.items()},
        "k3_mult": float(K3_MULT), "ewy_check": chk,
        "common_valid_rows_stock_all": int(S_all.sum()), "m4_events": {"peak": len(dn_ev), "trough": len(up_ev)},
        "m4_sentinel_unreached": 36, "ddof": DDOF,
    }

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        return o
    with open(os.path.join(HERE, f"results{SUFFIX}.json"), "w", encoding="utf-8") as f:
        json.dump(clean(res), f, ensure_ascii=False, indent=1)

    # 요약 표
    keys = ["gray_2024_25", "gray_2026", "rev_share", "changes_per_sy", "rev_per_sy", "strong_rev_per_sy", "flips_last_year",
            "median_run", "dn_abs_down", "dn_abs_gray", "dn_vol_down", "dn20_down", "up_abs_up", "up_abs_gray",
            "peak_median_lag", "peak_early", "trough_median_lag", "trough_early", "S_front", "S_back", "S_2025",
            "T1_front", "T1_back", "T1_2025", "etf_rev_share", "etf_dn_abs_down", "etf_up_abs_up", "etf_flips",
            "ewy_1012_level", "ewy_1012_T", "ewy_first_down_after_1002", "mu_jul_down_days", "mu_jul_up_days", "tau"]
    lines = [f"K0 비극단 애매(2024~25) {target:.4f}, 보정 {taus}"]
    lines.append("key".ljust(26) + "".join(k.rjust(11) for k in CANDS))
    for key in keys:
        row = []
        for k in CANDS:
            v = res["cands"][k][key]
            row.append((f"{v:.4f}" if isinstance(v, float) else str(v)).rjust(11))
        lines.append(key.ljust(26) + "".join(row))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, f"results{SUFFIX}.txt"), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
