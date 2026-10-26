"""fail_hunt.py 후속: 종목별 분해, 꾸준한 추세(60일 회귀 R²), 완화 조건 사례, 상하 직행 휩소, ETF 사례.
입력: fail_hunt.py가 만든 panel_feat.parquet
실행: PYTHONIOENCODING=utf-8 py fail_hunt2.py
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "panel_feat.parquet"))
PERIODS = {
    "2024": ("2024-01-02", "2024-12-31"),
    "2025": ("2025-01-02", "2025-12-31"),
    "26H1": ("2026-01-02", "2026-05-29"),
    "26H2": ("2026-06-01", "2026-10-12"),
    "ALL": ("2024-01-02", "2026-10-12"),
}


def inper(d, p):
    return (d >= p[0]) & (d <= p[1])


valid = (df["z_K0"] != -9) & (df["z_K3n"] != -9) & np.isfinite(df["hi20"])
ALL = valid & inper(df["date"], PERIODS["ALL"])
stock = ~df["is_etf"]

# ---- 60일 로그가격 회귀 기울기·R² (꾸준한 추세: 눈으로 가장 분명한 장면)
def reg60(g):
    lc = np.log(g["C"].to_numpy())
    n = 60
    x = np.arange(n) - (n - 1) / 2
    sxx = (x ** 2).sum()
    slope = np.full(len(lc), np.nan)
    r2 = np.full(len(lc), np.nan)
    if len(lc) >= n:
        win = np.lib.stride_tricks.sliding_window_view(lc, n)
        b = win @ x / sxx
        resid = win - win.mean(1, keepdims=True) - np.outer(b, x)
        sst = ((win - win.mean(1, keepdims=True)) ** 2).sum(1)
        slope[n - 1:] = b
        r2[n - 1:] = 1 - (resid ** 2).sum(1) / sst
    return pd.DataFrame({"slope60": slope, "r2_60": r2}, index=g.index)


rg = pd.concat([reg60(g) for _, g in df.groupby("sym", sort=False)])
df = df.join(rg)
# 60일 동안 기울기×60이 +15% 이상이고 R² ≥ 0.8 = 꾸준한 상승. 거울상 하락.
steady_up = (df["slope60"] * 60 >= np.log(1.15)) & (df["r2_60"] >= 0.8)
steady_dn = (df["slope60"] * 60 <= -np.log(1.15)) & (df["r2_60"] >= 0.8)
# ETF는 변동성이 작으므로 +8%
steady_up_e = (df["slope60"] * 60 >= np.log(1.08)) & (df["r2_60"] >= 0.8)
steady_dn_e = (df["slope60"] * 60 <= -np.log(1.08)) & (df["r2_60"] >= 0.8)


def share(z):
    z = np.asarray(z)
    if len(z) == 0:
        return {"n": 0}
    return {"n": int(len(z)), "down": round(float(np.mean(z < 0)), 4), "gray": round(float(np.mean(z == 0)), 4),
            "up": round(float(np.mean(z > 0)), 4)}


out = {"steady": {}, "steady_etf": {}}
for p, rng in PERIODS.items():
    M = valid & inper(df["date"], rng)
    out["steady"][p] = {k: {"up_trend": share(df.loc[M & stock & steady_up, "z_" + k]),
                            "dn_trend": share(df.loc[M & stock & steady_dn, "z_" + k])} for k in ("K0", "K3n")}
    out["steady_etf"][p] = {k: {"up_trend": share(df.loc[M & df.is_etf & steady_up_e, "z_" + k]),
                                "dn_trend": share(df.loc[M & df.is_etf & steady_dn_e, "z_" + k])} for k in ("K0", "K3n")}

# ---- 상방↔하방 직행(애매 거치지 않거나 3일 안에 반대편) 휩소: 종목·연당
def whipsaw(mask, k):
    n, days = 0, 0
    for sym, g in df[mask].groupby("sym", sort=False):
        z = np.sign(g["z_" + k].to_numpy())
        idx = g["idx"].to_numpy()
        days += len(z)
        # 0이 아닌 부호가 바뀐 지점: 직전 비0 부호와 3일 안
        last_s, last_i = 0, -99
        for t in range(len(z)):
            if z[t] != 0:
                if last_s != 0 and z[t] != last_s and idx[t] - last_i <= 3:
                    n += 1
                last_s, last_i = z[t], idx[t]
    return n / (days / 252.0)


out["whipsaw_py"] = {}
for p, rng in PERIODS.items():
    M = valid & inper(df["date"], rng)
    out["whipsaw_py"][p] = {grp: {k: round(whipsaw(M & gm, k), 3) for k in ("K0", "K3n")}
                            for grp, gm in (("stock", stock), ("etf", df.is_etf))}

# ---- 종목별: zz10 반대율·바뀜 수에서 K3n이 나쁜 종목 수
rows = []
for sym, g in df[ALL].groupby("sym"):
    lab = g["zz10"].to_numpy()
    m = lab != 0
    r = {"sym": sym, "etf": bool(g["is_etf"].iloc[0])}
    for k in ("K0", "K3n"):
        z = np.sign(g["z_" + k].to_numpy())
        r[k + "_opp"] = float(np.mean(z[m] == -lab[m]))
        zz = g["z_" + k].to_numpy()
        r[k + "_flips"] = float(np.sum(zz[1:] != zz[:-1]) / (len(zz) / 252))
    rows.append(r)
ps = pd.DataFrame(rows)
ps["opp_diff"] = ps["K3n_opp"] - ps["K0_opp"]
ps["flip_ratio"] = ps["K3n_flips"] / ps["K0_flips"]
out["per_symbol"] = {"n": len(ps), "K3n_opp_worse": int((ps.opp_diff > 0).sum()), "K3n_opp_worse_by_2pp": int((ps.opp_diff > 0.02).sum()),
                     "K3n_flips_more": int((ps.flip_ratio > 1).sum()), "flip_ratio_median": float(ps.flip_ratio.median()),
                     "worst_opp": ps.sort_values("opp_diff", ascending=False).head(8)[["sym", "K0_opp", "K3n_opp", "opp_diff"]].round(3).to_dict("records"),
                     "etf": ps[ps.etf][["sym", "K0_opp", "K3n_opp", "K0_flips", "K3n_flips"]].round(3).to_dict("records")}
ps.to_csv(os.path.join(HERE, "per_symbol.csv"), index=False, encoding="utf-8-sig")

# ---- 완화 사례: zz10 상승 스윙 한가운데(피벗 후 5일·다음 피벗 3일 이상 남음)인데 K3n<0, K0≥0. (거울상 포함)
def episodes(cond, label):
    res = []
    for sym, g in df[cond].groupby("sym"):
        ids = g["idx"].to_numpy()
        starts = np.flatnonzero(np.r_[True, np.diff(ids) != 1])
        ends = np.r_[starts[1:], len(ids)]
        for s0, e0 in zip(starts, ends):
            rws = g.iloc[s0:e0]
            r0 = rws.iloc[0]
            res.append({"kind": label, "sym": sym, "etf": bool(r0.is_etf), "start": r0.date, "end": rws.iloc[-1].date, "days": int(e0 - s0),
                        "swing_pct": round(float(np.expm1(r0.zz10_swing) * 100), 1), "C": round(float(r0.C), 2),
                        "T0": round(float(r0.T0), 3), "T3": round(float(r0.T3), 3), "K0": int(r0.z_K0), "K3n": int(r0.z_K3n),
                        "minT0": round(float(rws.T0.min()), 3), "minT3": round(float(rws.T3.min()), 3)})
    return pd.DataFrame(res)


mid = (df["zz10_since"] >= 5) & (df["zz10_until"] >= 3)
k3_wrong_up = ALL & mid & (df.zz10 == 1) & (df.z_K3n < 0) & (df.z_K0 >= 0)
k0_wrong_up = ALL & mid & (df.zz10 == 1) & (df.z_K0 < 0) & (df.z_K3n >= 0)
k3_wrong_dn = ALL & mid & (df.zz10 == -1) & (df.z_K3n > 0) & (df.z_K0 <= 0)
k0_wrong_dn = ALL & mid & (df.zz10 == -1) & (df.z_K0 > 0) & (df.z_K3n <= 0)
out["relaxed_counts_days"] = {"K3n_dn_in_upswing_K0_not": int(k3_wrong_up.sum()), "K0_dn_in_upswing_K3n_not": int(k0_wrong_up.sum()),
                              "K3n_up_in_dnswing_K0_not": int(k3_wrong_dn.sum()), "K0_up_in_dnswing_K3n_not": int(k0_wrong_dn.sum())}
# 꾸준한 상승 속 K3n 하방(K0는 상방) 사례
st_case = ALL & ((stock & steady_up) | (df.is_etf & steady_up_e)) & (df.z_K3n < 0) & (df.z_K0 > 0)
out["steady_up_K3n_dn_K0_up_days"] = int(st_case.sum())
out["steady_up_K0_dn_K3n_up_days"] = int((ALL & ((stock & steady_up) | (df.is_etf & steady_up_e)) & (df.z_K0 < 0) & (df.z_K3n > 0)).sum())
E1 = episodes(k3_wrong_up, "K3n하방_상승스윙")
E2 = episodes(k3_wrong_dn, "K3n상방_하락스윙")
E3 = episodes(st_case, "꾸준상승_K3n하방")
cases = pd.concat([E1, E2, E3]).sort_values(["days", "swing_pct"], ascending=False)
cases.to_csv(os.path.join(HERE, "cases_relaxed.csv"), index=False, encoding="utf-8-sig")
out["relaxed_episode_len"] = {lab: {"n": int(len(E)), "ge3": int((E.days >= 3).sum()) if len(E) else 0, "ge5": int((E.days >= 5).sum()) if len(E) else 0}
                              for lab, E in (("E1", E1), ("E2", E2), ("E3", E3))}
# 같은 기준의 K0 쪽 에피소드 길이(비교)
F1 = episodes(k0_wrong_up, "K0하방_상승스윙")
F2 = episodes(k0_wrong_dn, "K0상방_하락스윙")
out["relaxed_episode_len"].update({lab: {"n": int(len(E)), "ge3": int((E.days >= 3).sum()), "ge5": int((E.days >= 5).sum())}
                                   for lab, E in (("F1_K0", F1), ("F2_K0", F2))})

print(json.dumps(out, ensure_ascii=False, indent=1))
print("\n== 완화 사례 상위 (길이순)")
print(cases.head(45).to_string())
print("\n== ETF 사례")
print(cases[cases.etf].head(30).to_string())
with open(os.path.join(HERE, "fail_results2.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=float)
df[["sym", "date", "C", "rsi", "T0", "T3", "z_K0", "z_K3n", "zz10", "zz10_since", "zz10_until", "zz15", "slope60", "r2_60", "ret60", "sma50", "hi10", "lo10"]].to_parquet(
    os.path.join(HERE, "panel_feat2.parquet"))
