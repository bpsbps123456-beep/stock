"""R3 덮어쓰기가 사용자 정의(X4)와 얼마나 어긋나나: 비드문 날 발동 비율·종목별 2026 실제 횟수(X4 라벨 vs R3 조건)."""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refute_r3 as RV
TC = RV.TC

raw = pd.read_parquet(TC.PANEL); raw["ts"] = pd.to_datetime(raw["ts"])
parts = []
for sym, g in raw.groupby("sym", sort=True):
    o, _ = TC.per_symbol(g)
    C = o["C"].to_numpy(); cs = pd.Series(C)
    o["H20"] = cs.rolling(20, min_periods=20).max().to_numpy()
    o["dd20"] = np.log(C / o["H20"].to_numpy())
    o["Pd"] = pd.Series(o["dd20"].to_numpy()).rolling(252, min_periods=252).quantile(0.125, interpolation="linear").to_numpy()
    lc = np.log(C); n = len(C); f = np.full(n, np.nan); f[:n-10] = lc[10:] - lc[:n-10]; o["f10"] = f
    parts.append(o)
df = pd.concat(parts, ignore_index=True)
dates = df["date"].to_numpy(); dd = df["dd20"].to_numpy(); Pd = df["Pd"].to_numpy(); R = df["rsi"].to_numpy(); T3 = df["T3_20"].to_numpy()
valid = df["valid"].to_numpy(); stock = ~df["is_etf"].to_numpy()
b0 = TC.stage5(R, T3, 0.5); ext = (R >= 70) | (R <= 30)
with np.errstate(invalid="ignore"):
    r3 = np.where(ext, b0, np.where(dd <= Pd, -1, b0))
ow = valid & (r3 != b0)
P26 = ("2026-01-02", "2026-10-12")
in26 = RV.inper(dates, P26)
x4p = np.full(len(df), np.nan)
rows = []
for s, g in df.groupby("sym", sort=True):
    r = g.index.to_numpy()
    w = RV.inper(dates[r], RV.HIST["2026"]) & np.isfinite(dd[r])
    if w.sum() < 400:
        continue
    X = RV.x4(dd[r][w])
    if not np.isfinite(X):
        continue
    w26 = RV.inper(dates[r], P26)
    x4p[r[w26]] = np.log(1 - X)
    rr = r[w26 & np.isfinite(dd[r])]
    yrs = len(rr) / 252.0
    c_lab, _ = RV.ep_count(dd[rr], np.full((1, len(rr)), np.log(1 - X)), np.full((1, len(rr)), np.log(1 - X / 2)))
    ok = np.isfinite(Pd[rr])
    a = np.minimum(Pd[rr][ok], -1e-12); XX = 1 - np.exp(a)
    c_r3, _ = RV.ep_count(dd[rr][ok], a[None, :], np.log(1 - XX / 2)[None, :])
    last = r[-1]
    rows.append({"sym": s, "etf": bool(g["is_etf"].iloc[0]), "X4_pct": X * 100, "lab_eps26": c_lab[0] / yrs,
                 "r3_eps26": c_r3[0] / (ok.sum() / 252.0) if ok.sum() else np.nan,
                 "thr_1012_pct": (np.exp(Pd[last]) - 1) * 100})
t = pd.DataFrame(rows)
t["ratio"] = t.r3_eps26 / t.lab_eps26.replace(0, np.nan)
M = ow & in26 & stock
notrare = M & np.isfinite(x4p) & (dd > x4p)
israre = M & np.isfinite(x4p) & (dd <= x4p)
noX = M & ~np.isfinite(x4p)
rare_cells = in26 & stock & valid & np.isfinite(x4p) & (dd <= x4p)
out = {
 "ow26_stock_cells": int(M.sum()),
 "ow26_on_nonrare_day_share": float(notrare.sum() / (notrare.sum() + israre.sum())),
 "ow26_no_x4_cells": int(noX.sum()),
 "ow26_nonrare_f10_mean_pct": float(np.nanmean(df["f10"].to_numpy()[notrare]) * 100),
 "ow26_rare_f10_mean_pct": float(np.nanmean(df["f10"].to_numpy()[israre]) * 100),
 "ow26_nonrare_f10_median_pct": float(np.nanmedian(df["f10"].to_numpy()[notrare]) * 100),
 "ow26_rare_f10_median_pct": float(np.nanmedian(df["f10"].to_numpy()[israre]) * 100),
 "rare26_cells": int(rare_cells.sum()),
 "rare26_missed_by_R3_gray_or_up": int((rare_cells & (r3 >= 0)).sum()),
 "rare26_missed_share": float(np.mean(r3[rare_cells] >= 0)),
 "rare26_missed_because_thr_deeper_share": float(np.mean((dd[rare_cells & (r3 >= 0)] > Pd[rare_cells & (r3 >= 0)]))),
 "stock_lab_eps26_med": float(t[~t.etf].lab_eps26.median()), "stock_r3_eps26_med": float(t[~t.etf].r3_eps26.median()),
 "stock_ratio_q": [float(t[~t.etf].ratio.quantile(q)) for q in (.1, .25, .5, .75, .9)],
 "etf": t[t.etf].round(2).to_dict(orient="records"),
 "case": t[t.sym.isin(["NVDA", "MU", "GLW", "SOXX", "SMH", "EWY"])].round(2).to_dict(orient="records"),
}
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(os.path.join(HERE, "extra_fidelity.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
