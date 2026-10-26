"""추세 5단계 — 꺾임에 더 빨리 반응하는 가운데 판정 10개 비교 (turn_A, 연구용). 저장소 파일은 읽기·import만 한다.

사전 등록 후보 B0~B9(모두 RSI14 파인식 ≥70 → +2, ≤30 → −2가 먼저), 정답 = 사후 로그 종가 지그재그(θ 4가지).
실행: PYTHONIOENCODING=utf-8 py turn_compare.py
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

MIN_IDX = 129                                   # bar_index >= 129 → 일봉 130개째부터
PER = {"2024": ("2024-01-02", "2024-12-31"), "2025": ("2025-01-02", "2025-12-31"),
       "2026H1": ("2026-01-02", "2026-05-29"), "2026H2": ("2026-06-01", "2026-10-12"),
       "ALL": ("2024-01-02", "2026-10-12")}
P_2026 = ("2026-01-02", "2026-10-12")
P_LASTY = ("2025-10-13", "2026-10-12")
THETAS = ("Z5", "Z10", "ZV1", "ZV2")
CANDS = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]
TCOL = {"B0": "T3_20", "B1": "T3_15", "B2": "T3_10", "B3": "T3_20", "B4": "T3_20", "B5": "T3_15",
        "B6": "B6_dd", "B7": "T0", "B8": "TE", "B9": "T9"}
TAU = {"B0": 0.5, "B1": 0.5, "B2": 0.5, "B3": 0.5, "B4": 0.5, "B5": 0.5, "B6": None, "B7": 0.5, "B8": 0.655, "B9": 0.5}


def kmult(N):
    return np.sqrt((N - 1) * (2 * N - 1) / (6 * N))


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


def zigzag(lc, th):
    """로그 종가 지그재그. th[i] = i일이 극값일 때의 되돌림 문턱(로그). 반환: 라벨(확정 스윙 안만 ±1, 나머지 0), 확정 꺾임점."""
    n = len(lc)
    lab = np.zeros(n, np.int8)
    i = 0
    while i < n and not np.isfinite(th[i]):
        i += 1
    if i >= n - 1:
        return lab, []
    hi = lo = i
    mode, piv, ext = 0, [], i
    j = i + 1
    while j < n and mode == 0:
        if lc[j] > lc[hi]:
            hi = j
        if lc[j] < lc[lo]:
            lo = j
        if lc[j] - lc[lo] >= th[lo]:
            mode, piv, ext = 1, [lo], j
        elif lc[hi] - lc[j] >= th[hi]:
            mode, piv, ext = -1, [hi], j
        j += 1
    if mode == 0:
        return lab, []
    for k in range(j, n):
        if mode == 1:
            if lc[k] > lc[ext]:
                ext = k
            elif lc[ext] - lc[k] >= th[ext]:
                piv.append(ext)
                mode, ext = -1, k
        else:
            if lc[k] < lc[ext]:
                ext = k
            elif lc[k] - lc[ext] >= th[ext]:
                piv.append(ext)
                mode, ext = 1, k
    # 확정 꺾임점끼리만 라벨(마지막 확정 꺾임점 뒤 진행 중 스윙은 뺀다)
    for a, b in zip(piv[:-1], piv[1:]):
        lab[a + 1:b + 1] = 1 if lc[b] > lc[a] else -1
    return lab, piv


def b6_swing(C, sd20, rsi, T20, valid):
    """회색 없는 2단계 스윙. 첫 유효일 s = sign(T3(20))(0이면 +1), ext = C."""
    n = len(C)
    st = np.full(n, -9, np.int8)
    ddn = np.full(n, np.nan)                       # ln(C/ext) / (sd20·√10) — 상태 기준 되돌림(참고 T)
    s, ext, started = 0, np.nan, False
    for i in range(n):
        if not valid[i]:
            continue
        thr = 1.5 * sd20[i] * np.sqrt(10)
        if not started:
            s = 1 if T20[i] >= 0 else -1
            if T20[i] == 0:
                s = 1
            ext, started = C[i], True
        elif s == 1:
            ext = max(ext, C[i])
            if np.log(C[i] / ext) <= -thr:
                s, ext = -1, C[i]
        else:
            ext = min(ext, C[i])
            if np.log(C[i] / ext) >= thr:
                s, ext = 1, C[i]
        ddn[i] = np.log(C[i] / ext) / (sd20[i] * np.sqrt(10))
        st[i] = 2 if rsi[i] >= 70 else (-2 if rsi[i] <= 30 else s)
    return st, ddn


def per_symbol(g):
    g = g.sort_values("ts").reset_index(drop=True)
    C = g["Close"].to_numpy(float)
    lc = np.log(C)
    s = pd.Series(lc)
    lr = s.diff()
    sd = {N: lr.rolling(N, min_periods=N).std(ddof=1) for N in (10, 15, 20)}
    o = pd.DataFrame({"sym": g["sym"], "date": g["ts"].dt.strftime("%Y-%m-%d"), "is_etf": g["is_etf"],
                      "C": C, "idx": np.arange(len(C))})
    o["lr"] = lr.to_numpy()
    o["sd20"] = sd[20].to_numpy()
    o["rsi"] = rsi_pine(C)
    for N in (10, 15, 20):
        o[f"T3_{N}"] = ((s - s.rolling(N, min_periods=N).mean()) / (sd[N] * kmult(N))).to_numpy()
    o["T0"] = ((s - s.shift(20)) / (sd[20] * np.sqrt(20))).to_numpy()
    ema10 = pd.Series(C).ewm(span=10, adjust=False).mean()
    o["TE"] = ((s - np.log(ema10)) / sd[20]).to_numpy()
    o["T9"] = (o["T3_10"] + o["T3_20"]) / 2
    hi10 = pd.Series(C).rolling(10, min_periods=10).max().to_numpy()
    lo10 = pd.Series(C).rolling(10, min_periods=10).min().to_numpy()
    o["dd10"] = np.log(C / hi10)
    o["ru10"] = np.log(C / lo10)
    o["u10"] = (sd[20] * np.sqrt(10)).to_numpy()
    # 60일 로그 회귀 기울기·R² (꾸준한 추세)
    n = 60
    x = np.arange(n) - (n - 1) / 2
    sxx = (x ** 2).sum()
    slope = np.full(len(lc), np.nan)
    r2 = np.full(len(lc), np.nan)
    if len(lc) >= n:
        win = np.lib.stride_tricks.sliding_window_view(lc, n)
        dm = win - win.mean(1, keepdims=True)
        b = win @ x / sxx
        resid = dm - np.outer(b, x)
        slope[n - 1:] = b
        r2[n - 1:] = 1 - (resid ** 2).sum(1) / (dm ** 2).sum(1)
    o["slope60"], o["r2_60"] = slope, r2
    # 유효: 130개째부터 + 모든 후보 입력이 유한
    need = ["rsi", "sd20", "T3_10", "T3_15", "T3_20", "T0", "TE", "dd10", "ru10"]
    valid = (o["idx"].to_numpy() >= MIN_IDX) & np.all([np.isfinite(o[c].to_numpy()) for c in need], axis=0)
    o["valid"] = valid
    st6, dd6 = b6_swing(C, o["sd20"].to_numpy(), o["rsi"].to_numpy(), o["T3_20"].to_numpy(), valid)
    o["z_B6"], o["B6_dd"] = st6, dd6
    # 지그재그 정답(θ 4가지)
    sdv = o["sd20"].to_numpy()
    ths = {"Z5": np.where(np.isfinite(sdv), 0.05, np.nan), "Z10": np.where(np.isfinite(sdv), 0.10, np.nan),
           "ZV1": 1.0 * sdv * np.sqrt(10), "ZV2": 2.0 * sdv * np.sqrt(10)}
    piv_all = {}
    for name, th in ths.items():
        lab, piv = zigzag(lc, th)
        o["lab_" + name] = lab
        piv_all[name] = piv
    return o, piv_all


def stage5(rsi, T, tau):
    return np.select([rsi >= 70, rsi <= 30, T >= tau, T <= -tau], [2, -2, 1, -1], 0)


def overwrite(base, rsi, dd, ru, u, m):
    ext = (rsi >= 70) | (rsi <= 30)
    cd = dd <= -m * u
    cu = ru >= m * u
    both = cd & cu
    big = np.where(np.abs(dd) > ru, -1, np.where(ru > np.abs(dd), 1, base))
    over = np.where(both, big, np.where(cd, -1, np.where(cu, 1, base)))
    return np.where(ext, base, over)


def inper(d, p):
    return (d >= p[0]) & (d <= p[1])


# ---------------------------------------------------------------- 지표
def agree_tab(df, st, mask):
    out = {}
    z = np.sign(df[st].to_numpy())
    for th in THETAS:
        lab = df["lab_" + th].to_numpy()
        out[th] = {}
        for p, rng in PER.items():
            M = (mask & inper(df["date"], rng)).to_numpy() & (lab != 0)
            n = int(M.sum())
            a = float(np.mean(z[M] == lab[M])) if n else float("nan")
            o = float(np.mean(z[M] == -lab[M])) if n else float("nan")
            out[th][p] = {"n": n, "agree": a, "opp": o, "gray": float(np.mean(z[M] == 0)) if n else float("nan"),
                          "score": a - 2 * o}
    return out


def score_of(tab, periods):
    return float(np.mean([tab[th][p]["score"] for th in THETAS for p in periods]))


def flips_direct(df, st, mask):
    changes, days, whip = 0, 0, 0
    for _, g in df[mask].groupby("sym", sort=False):
        z = g[st].to_numpy()
        idx = g["idx"].to_numpy()
        days += len(z)
        changes += int(((np.diff(idx) == 1) & (z[1:] != z[:-1])).sum())
        sg = np.sign(z)
        ls, li = 0, -99
        for t in range(len(sg)):
            if sg[t] != 0:
                if ls != 0 and sg[t] != ls and idx[t] - li <= 3:
                    whip += 1
                ls, li = sg[t], idx[t]
    sy = days / 252.0
    return {"flips_py": changes / sy if sy else float("nan"), "direct_py": whip / sy if sy else float("nan"),
            "sym_years": sy}


def rev_share(df, st, mask):
    prev = df.groupby("sym")[st].shift(1)
    pidx = df.groupby("sym")["idx"].shift(1)
    pval = df.groupby("sym")["valid"].shift(1).fillna(False).astype(bool)
    cur = df[st]
    ok = mask & pval & (pidx == df["idx"] - 1)
    ch = ok & (cur != prev)
    rev = ch & (np.sign(cur - prev) * np.sign(df["lr"]) < 0)
    sy = mask.sum() / 252.0
    return {"rev_share": float(rev.sum() / ch.sum()) if ch.sum() else float("nan"), "changes": int(ch.sum()),
            "rev": int(rev.sum()), "rev_py": float(rev.sum() / sy)}


def swing_lags(df, st, swings, period):
    """swings: (sym, a_row, b_row) — df 행 번호(같은 종목 연속). lag = 처음 스윙 방향 계열이 된 날 − a (1 = 다음날)."""
    z = np.sign(df[st].to_numpy())
    lc = np.log(df["C"].to_numpy())
    val = df["valid"].to_numpy()
    dates = df["date"].to_numpy()
    lags, never, n_up, n_dn = [], 0, 0, 0
    lag_by_dir = {1: [], -1: []}
    for sym, a, b in swings:
        if not (period[0] <= dates[a + 1] <= period[1]):
            continue
        if not val[a + 1:b + 1].all():
            continue
        d = 1 if lc[b] > lc[a] else -1
        seg = z[a + 1:b + 1]
        hit = np.flatnonzero(seg == d)
        L = b - a
        if len(hit):
            lag = int(hit[0]) + 1
        else:
            lag = L + 1
            never += 1
        lags.append(lag)
        lag_by_dir[d].append(lag)
    n = len(lags)
    return {"n": n, "median": float(np.median(lags)) if n else float("nan"),
            "mean": float(np.mean(lags)) if n else float("nan"), "never": never / n if n else float("nan"),
            "median_up": float(np.median(lag_by_dir[1])) if lag_by_dir[1] else float("nan"),
            "median_dn": float(np.median(lag_by_dir[-1])) if lag_by_dir[-1] else float("nan"),
            "n_up": len(lag_by_dir[1]), "n_dn": len(lag_by_dir[-1])}


def main():
    raw = pd.read_parquet(PANEL)
    raw["ts"] = pd.to_datetime(raw["ts"])
    parts, pivs = [], {}
    off = 0
    for sym, g in raw.groupby("sym", sort=True):
        o, p = per_symbol(g)
        parts.append(o)
        pivs[sym] = {k: [off + i for i in v] for k, v in p.items()}
        off += len(o)
    df = pd.concat(parts, ignore_index=True)

    # ---- 재현 확인
    sx = df[(df.sym == "SOXX")].set_index("date")
    chk = {d: {c: float(sx.at[d, c]) for c in ("C", "rsi", "T3_20", "T3_15", "T3_10", "T0", "sd20")}
           for d in ("2026-10-08", "2026-10-09", "2026-10-12")}
    chk["SOXX_hi10_1012"] = float(np.exp(-sx.at["2026-10-12", "dd10"]) * sx.at["2026-10-12", "C"])
    print("SOXX 확인:", json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 4) for kk, vv in v.items()})
                                   for k, v in chk.items()}, ensure_ascii=False))

    R = df["rsi"].to_numpy()
    T = {c: df[c].to_numpy() for c in ("T3_10", "T3_15", "T3_20", "T0", "TE", "T9")}
    dd, ru, u = df["dd10"].to_numpy(), df["ru10"].to_numpy(), df["u10"].to_numpy()
    b0 = stage5(R, T["T3_20"], 0.5)
    b1 = stage5(R, T["T3_15"], 0.5)
    stg = {"B0": b0, "B1": b1, "B2": stage5(R, T["T3_10"], 0.5),
           "B3": overwrite(b0, R, dd, ru, u, 1.0), "B4": overwrite(b0, R, dd, ru, u, 1.5),
           "B5": overwrite(b1, R, dd, ru, u, 1.5), "B7": stage5(R, T["T0"], 0.5),
           "B8": stage5(R, T["TE"], 0.655), "B9": stage5(R, T["T9"], 0.5)}
    valid = df["valid"].to_numpy()
    for k, v in stg.items():
        df["z_" + k] = np.where(valid, v, -9).astype(int)
    df["z_B6"] = np.where(valid, df["z_B6"], -9).astype(int)
    assert ((df["z_B6"] != -9) == df["valid"]).all()

    stock = ~df["is_etf"]
    V = df["valid"]
    S_all = stock & V
    E_all = df["is_etf"] & V
    S_ly = stock & V & inper(df["date"], P_LASTY)
    E_ly = df["is_etf"] & V & inper(df["date"], P_LASTY)
    S_inall = S_all & inper(df["date"], PER["ALL"])

    # 꾸준한 추세(60일 회귀 기울기×60 ≥ ln 1.15, R² ≥ 0.8) + Z10 스윙 방향
    stU = (df["slope60"] * 60 >= np.log(1.15)) & (df["r2_60"] >= 0.8) & (df["lab_Z10"] == 1)
    stD = (df["slope60"] * 60 <= -np.log(1.15)) & (df["r2_60"] >= 0.8) & (df["lab_Z10"] == -1)

    # Z10 확정 스윙 목록(주식)
    swings = []
    stock_syms = set(df.loc[stock, "sym"])
    for sym in stock_syms:
        p = pivs[sym]["Z10"]
        swings += [(sym, a, b) for a, b in zip(p[:-1], p[1:])]

    # 예측(보조): zones.py 엔진, 150종목
    st_raw = raw[~raw["is_etf"]].copy()
    st_raw["d"] = st_raw["ts"].dt.strftime("%Y-%m-%d")
    piv = {k: st_raw.pivot_table(index="d", columns="sym", values=k).sort_index() for k in ("Open", "High", "Low", "Close")}
    I = ZM.Ind(piv["Open"], piv["High"], piv["Low"], piv["Close"])
    F = ZM.Fwd(I, 10)
    prow = {p: ZM.period_rows(I, p, 10) for p in ("앞", "뒤", "2025")}

    res = {"meta": {}, "cands": {}, "soxx_0901_1012": {}, "cases": {}}
    for k in CANDS:
        st = "z_" + k
        r = {"id": k, "tau": TAU[k], "tcol": TCOL[k]}
        zz = df[st]
        r["gray_2026"] = float(np.mean(zz[S_all & inper(df["date"], P_2026)] == 0))
        r["gray_2024_25"] = float(np.mean(zz[S_all & inper(df["date"], ("2024-01-02", "2025-12-31"))] == 0))
        tab = agree_tab(df, st, S_all)
        r["agree_tab"] = tab
        r["score"] = score_of(tab, ("2026H1", "2026H2"))
        r["score_2026H1"] = score_of(tab, ("2026H1",))
        r["score_2026H2"] = score_of(tab, ("2026H2",))
        r["score_2024"] = score_of(tab, ("2024",))
        r["score_2025"] = score_of(tab, ("2025",))
        r["opp26_mean"] = float(np.mean([tab[th][p]["opp"] for th in THETAS for p in ("2026H1", "2026H2")]))
        r["score26_by_theta"] = {th: float(np.mean([tab[th][p]["score"] for p in ("2026H1", "2026H2")])) for th in THETAS}
        r["agree26_mean"] = float(np.mean([tab[th][p]["agree"] for th in THETAS for p in ("2026H1", "2026H2")]))
        for th in THETAS:
            r[f"agree26h1_{th.lower()}"] = tab[th]["2026H1"]["agree"]
            r[f"agree26h2_{th.lower()}"] = tab[th]["2026H2"]["agree"]
            r[f"opp26h1_{th.lower()}"] = tab[th]["2026H1"]["opp"]
            r[f"opp26h2_{th.lower()}"] = tab[th]["2026H2"]["opp"]
        lg = swing_lags(df, st, swings, PER["ALL"])
        lg26 = swing_lags(df, st, swings, P_2026)
        r["lag_z10_median"], r["lag_z10_never"] = lg["median"], lg["never"]
        r["lag_z10_detail"] = {"ALL": lg, "2026": lg26}
        fd = flips_direct(df, st, S_ly)
        r["flips"], r["direct"] = fd["flips_py"], fd["direct_py"]
        r["flips_detail"] = {"last_year": fd, "ALL": flips_direct(df, st, S_inall)}
        r["steady_up"] = float(np.mean(np.sign(zz[S_inall & stU]) > 0))
        r["steady_dn"] = float(np.mean(np.sign(zz[S_inall & stD]) < 0))
        r["steady_n"] = {"up": int((S_inall & stU).sum()), "dn": int((S_inall & stD).sum())}
        r["steady_2026"] = {"up": float(np.mean(np.sign(zz[S_all & inper(df["date"], P_2026) & stU]) > 0)),
                            "dn": float(np.mean(np.sign(zz[S_all & inper(df["date"], P_2026) & stD]) < 0))}
        rv = rev_share(df, st, S_inall)
        r["rev"], r["rev_detail"] = rv["rev_share"], rv
        # 예측
        tmp = pd.DataFrame({"d": df["date"], "sym": df["sym"], "z": zz})[stock.to_numpy()]
        Zp = tmp.pivot(index="d", columns="sym", values="z").reindex(index=I.dates, columns=I.syms).fillna(-9)
        Z = Zp.to_numpy().astype(np.int8)
        m5 = {}
        for p, lab in (("앞", "front"), ("뒤", "back"), ("2025", "2025")):
            ev5 = ZM.evaluate(Z, F, prow[p], full=True)
            r[f"S_{lab}"] = ev5["S"] * 100
            r[f"T1_{lab}"] = ev5["terms"]["T1 상방-하방"] * 100
            m5[lab] = {"S": ev5["S"] * 100, "terms": {kk: vv * 100 for kk, vv in ev5["terms"].items()},
                       "t": ev5["t"], "share": {str(z): s for z, s in ev5["share"].items()}}
        r["pred_detail"] = m5
        # ETF
        etab = agree_tab(df, st, E_all)
        r["etf_score_2026"] = score_of(etab, ("2026H1", "2026H2"))
        r["etf_flips"] = flips_direct(df, st, E_ly)["flips_py"]
        r["etf_direct"] = flips_direct(df, st, E_ly)["direct_py"]
        r["etf_gray_2026"] = float(np.mean(zz[E_all & inper(df["date"], P_2026)] == 0))
        # 사례
        tc = TCOL[k]
        for sym in ("SOXX", "EWY"):
            row = df[(df.sym == sym) & (df.date == "2026-10-12")].iloc[0]
            r[f"{sym.lower()}_level"] = int(row[st])
            r[f"{sym.lower()}_T"] = float(row[tc])
            r[f"{sym.lower()}_extra"] = {"rsi": float(row["rsi"]), "dd10_over_u": float(row["dd10"] / row["u10"]),
                                         "ru10_over_u": float(row["ru10"] / row["u10"]), "T3_20": float(row["T3_20"])}
        spy = df[(df.sym == "SPY") & inper(df["date"], ("2026-07-23", "2026-07-28"))]
        r["spy_jul_down_days"] = int((spy[st] < 0).sum())
        r["spy_jul_n_days"] = int(len(spy))
        r["spy_jul_levels"] = dict(zip(spy["date"], spy[st].astype(int)))
        mu = df[(df.sym == "MU") & inper(df["date"], ("2026-06-26", "2026-07-28"))]
        r["mu_jul_down_days"] = int(((mu[st] < 0) & (mu[st] != -9)).sum())
        r["mu_jul_up_days"] = int((mu[st] > 0).sum())
        r["mu_jul_n_days"] = int(len(mu))
        ew = df[(df.sym == "EWY")].set_index("date")
        after = ew[(ew.index > "2026-10-02") & (ew[st] < 0)]
        r["ewy_first_down_after_1002"] = after.index[0] if len(after) else "없음"
        so = df[(df.sym == "SOXX")].set_index("date")
        after = so[(so.index > "2026-10-06") & (so[st] < 0)]
        r["soxx_first_down_after_1006"] = after.index[0] if len(after) else "없음"
        res["cands"][k] = r
        print(k, "done", round(r["score"], 4))

    cols = ["date", "C", "rsi", "T3_20", "T3_15", "T3_10", "T0", "TE", "T9", "dd10", "ru10", "u10", "B6_dd",
            "lab_Z5", "lab_Z10", "lab_ZV1", "lab_ZV2"] + ["z_" + k for k in CANDS]
    t = df[(df.sym == "SOXX") & inper(df["date"], ("2026-09-01", "2026-10-12"))][cols]
    res["soxx_0901_1012"] = json.loads(t.to_json(orient="records"))
    for sym, p in (("EWY", ("2026-09-08", "2026-10-12")), ("SPY", ("2026-07-15", "2026-08-05")),
                   ("MU", ("2026-06-22", "2026-08-05"))):
        t = df[(df.sym == sym) & inper(df["date"], p)][cols]
        res["cases"][sym] = json.loads(t.to_json(orient="records"))

    res["meta"] = {
        "panel": os.path.abspath(PANEL), "n_stock_syms": int(df.loc[stock, "sym"].nunique()),
        "n_stock_syms_valid_all": int(df.loc[S_inall, "sym"].nunique()),
        "n_etf_syms": int(df.loc[df.is_etf, "sym"].nunique()), "soxx_check": chk,
        "k_mult": {N: float(kmult(N)) for N in (10, 15, 20)},
        "valid_rows_stock_all": int(S_inall.sum()),
        "labeled_rows_stock_all": {th: int((S_inall & (df["lab_" + th] != 0)).sum()) for th in THETAS},
        "z10_swings_stock_total": len(swings),
        "steady_rows": {"up": int((S_inall & stU).sum()), "dn": int((S_inall & stD).sum())},
        "periods": PER, "flips_window": P_LASTY,
        "overwrite_changed_share_stock_all": {
            "B3_vs_B0": float(np.mean(df.loc[S_inall, "z_B3"] != df.loc[S_inall, "z_B0"])),
            "B4_vs_B0": float(np.mean(df.loc[S_inall, "z_B4"] != df.loc[S_inall, "z_B0"])),
            "B5_vs_B1": float(np.mean(df.loc[S_inall, "z_B5"] != df.loc[S_inall, "z_B1"]))},
        "last_confirmed_pivot": {sym: {th: df.at[pivs[sym][th][-1], "date"] if pivs[sym][th] else None for th in THETAS}
                                 for sym in ("SOXX", "EWY", "SPY", "MU")},
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
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(clean(res), f, ensure_ascii=False, indent=1)

    keys = ["score", "score_2026H1", "score_2026H2", "score_2024", "score_2025", "agree26_mean", "opp26_mean",
            "gray_2026", "gray_2024_25", "lag_z10_median", "lag_z10_never", "flips", "direct", "steady_up", "steady_dn",
            "rev", "S_front", "S_back", "S_2025", "T1_front", "T1_back", "T1_2025", "etf_score_2026", "etf_flips",
            "soxx_level", "soxx_T", "ewy_level", "ewy_T", "spy_jul_down_days", "mu_jul_down_days", "mu_jul_up_days",
            "ewy_first_down_after_1002", "soxx_first_down_after_1006"]
    keys += [f"agree26{h}_{th.lower()}" for h in ("h1", "h2") for th in THETAS]
    keys += [f"opp26{h}_{th.lower()}" for h in ("h1", "h2") for th in THETAS]
    lines = ["key".ljust(28) + "".join(k.rjust(11) for k in CANDS)]
    for key in keys:
        row = []
        for k in CANDS:
            v = res["cands"][k][key]
            row.append((f"{v:.4f}" if isinstance(v, float) else str(v)).rjust(11))
        lines.append(key.ljust(28) + "".join(row))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "results.txt"), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
