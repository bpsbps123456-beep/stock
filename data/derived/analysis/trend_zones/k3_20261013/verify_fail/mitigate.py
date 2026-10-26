"""K3n 약점(깜빡임·꾸준한 추세 끊김) 완화안 빠른 시험. 판단 보조용.
V1 거부권: K3n 단계가 K0 단계와 반대 부호(둘 다 비애매)면 애매로.
V2 히스테리시스: ±1 진입은 |T3|≥0.5, 유지는 T3가 ∓0.25를 넘기 전까지. RSI 극단은 그대로 덮어쓴다.
V3 = V1 + V2.
입력: panel_feat2.parquet, 실행: PYTHONIOENCODING=utf-8 py mitigate.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "panel_feat2.parquet")).sort_values(["sym", "date"]).reset_index(drop=True)
raw = pd.read_parquet(os.path.join(HERE, "..", "panel_1d_20261012.parquet"))
etf = set(raw.loc[raw.is_etf, "sym"].unique())
df["is_etf"] = df.sym.isin(etf)
df["lr"] = df.groupby("sym")["C"].transform(lambda s: np.log(s).diff())
valid = (df.z_K0 != -9) & (df.z_K3n != -9)


def hyst(g, T, enter=0.5, stay=0.25):
    z = np.zeros(len(g), int)
    s = 0
    for i, t in enumerate(T):
        if not np.isfinite(t):
            s = 0
            continue
        if s == 1 and t <= -enter:
            s = -1
        elif s == -1 and t >= enter:
            s = 1
        elif s == 1 and t < stay:
            s = 0 if t > -enter else -1
        elif s == -1 and t > -stay:
            s = 0 if t < enter else 1
        elif s == 0:
            s = 1 if t >= enter else (-1 if t <= -enter else 0)
        z[i] = s
    return z


R = df.rsi.to_numpy()
ext = np.select([R >= 70, R <= 30], [2, -2], 0)
base3 = df.z_K3n.to_numpy()
k0 = df.z_K0.to_numpy()
h = np.concatenate([hyst(g, g.T3.to_numpy()) for _, g in df.groupby("sym", sort=False)])
hz = np.where(ext != 0, ext, h)
veto = lambda z: np.where((np.sign(z) * np.sign(k0) < 0) & (np.abs(z) == 1) & (np.abs(k0) == 1), 0, z)
V = {"K0": k0, "K3n": base3, "V1": veto(base3), "V2": hz, "V3": veto(hz)}
for k, z in V.items():
    df["v_" + k] = np.where(valid, z, -9)

df["hi10"] = df.groupby("sym")["C"].transform(lambda s: s.rolling(10, min_periods=10).max())
df["lo10"] = df.groupby("sym")["C"].transform(lambda s: s.rolling(10, min_periods=10).min())
steady_up = (df.slope60 * 60 >= np.log(1.15)) & (df.r2_60 >= 0.8)
steady_dn = (df.slope60 * 60 <= -np.log(1.15)) & (df.r2_60 >= 0.8)

P = {"2024": ("2024-01-02", "2024-12-31"), "2025": ("2025-01-02", "2025-12-31"), "26H1": ("2026-01-02", "2026-05-29"),
     "26H2": ("2026-06-01", "2026-10-12"), "ALL": ("2024-01-02", "2026-10-12")}


def metrics(col, M):
    z = df[col]
    prev = df.groupby("sym")[col].shift(1)
    same = df.sym.eq(df.sym.shift(1))
    ok = M & same & (prev != -9)
    ch = ok & (z != prev)
    rev = ch & (np.sign(z - prev) * np.sign(df.lr) < 0)
    sy = M.sum() / 252
    # 직행 휩소(비0 부호가 3일 안에 반대로)
    ws = 0
    for _, g in df[M].groupby("sym", sort=False):
        s = np.sign(g[col].to_numpy())
        ls, li = 0, -99
        for t, v in enumerate(s):
            if v != 0:
                if ls != 0 and v != ls and t - li <= 3:
                    ws += 1
                ls, li = v, t
    dn = M & (df.C / df.hi10 - 1 <= -0.10)
    up = M & (df.C / df.lo10 - 1 >= 0.10)
    lab = df.zz10
    mz = M & (lab != 0)
    return {"flips": ch.sum() / sy, "rev": rev.sum() / max(1, ch.sum()), "whip": ws / sy,
            "dn10_down": (z[dn] < 0).mean(), "dn10_up": (z[dn] > 0).mean(), "up10_up": (z[up] > 0).mean(),
            "stUp_up": (z[M & steady_up] > 0).mean(), "stUp_dn": (z[M & steady_up] < 0).mean(),
            "stDn_dn": (z[M & steady_dn] < 0).mean(), "stDn_up": (z[M & steady_dn] > 0).mean(),
            "zz10_opp": (np.sign(z[mz]) == -lab[mz]).mean()}


def crash_lag(col, p):
    lags = []
    for sym, g in df[~df.is_etf].groupby("sym", sort=False):
        C = g.C.to_numpy(); z = g[col].to_numpy(); d = g.date.to_numpy(); n = len(C)
        cond = np.zeros(n, bool)
        for q in range(15, n):
            cond[q] = C[q] <= 0.85 * C[q - 15:q].max()
        for q in range(16, n):
            if not cond[q] or cond[q - 1] or not (p[0] <= d[q] <= p[1]) or q + 20 > n - 1:
                continue
            w = C[q - 15:q]
            pp = q - 15 + np.flatnonzero(w == w.max())[-1]
            seg = z[pp:q + 21]
            if (seg == -9).any():
                continue
            hit = np.flatnonzero(seg < 0)
            lags.append(hit[0] if len(hit) else 36)
    return float(np.median(lags)), len(lags)


rows = []
for pn, p in P.items():
    for grp, gm in (("주식", ~df.is_etf), ("ETF", df.is_etf)):
        M = valid & gm & (df.date >= p[0]) & (df.date <= p[1])
        for k in V:
            r = {"기간": pn, "군": grp, "안": k}
            r.update({kk: round(float(vv), 3) for kk, vv in metrics("v_" + k, M).items()})
            if grp == "주식":
                r["crash_lag_med"] = crash_lag("v_" + k, p)[0]
            rows.append(r)
out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(out.to_string(index=False))
out.to_csv(os.path.join(HERE, "mitigate.csv"), index=False, encoding="utf-8-sig")
for sym, a, b in (("EWY", "2026-10-05", "2026-10-12"), ("SPY", "2026-07-22", "2026-07-30"), ("IBKR", "2024-12-06", "2024-12-17"), ("SMH", "2024-01-02", "2024-01-08")):
    print(sym, df[(df.sym == sym) & (df.date >= a) & (df.date <= b)][["date", "C", "T0", "T3"] + ["v_" + k for k in V]].round(3).to_string(index=False))
