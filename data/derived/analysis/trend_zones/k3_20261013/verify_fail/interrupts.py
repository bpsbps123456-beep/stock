"""꾸준한 추세(60일 회귀 R²≥0.8, ±15%/60일, ETF ±8%) 안에서 추세색이 끊긴 횟수(종목·연), 기간별. 사례 표 보조 출력.
입력 panel_feat2.parquet. 실행: PYTHONIOENCODING=utf-8 py interrupts.py"""
import os
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "panel_feat2.parquet")).sort_values(["sym", "date"]).reset_index(drop=True)
raw = pd.read_parquet(os.path.join(HERE, "..", "panel_1d_20261012.parquet"))
df["is_etf"] = df.sym.isin(set(raw.loc[raw.is_etf, "sym"]))
valid = (df.z_K0 != -9) & (df.z_K3n != -9)
thr = np.where(df.is_etf, np.log(1.08), np.log(1.15))
su = (df.slope60 * 60 >= thr) & (df.r2_60 >= 0.8)
sd = (df.slope60 * 60 <= -thr) & (df.r2_60 >= 0.8)
P = {"2024": ("2024-01-02", "2024-12-31"), "2025": ("2025-01-02", "2025-12-31"), "26H1": ("2026-01-02", "2026-05-29"),
     "26H2": ("2026-06-01", "2026-10-12"), "ALL": ("2024-01-02", "2026-10-12")}
same = df.sym.eq(df.sym.shift(1))
for pn, p in P.items():
    inp = valid & (df.date >= p[0]) & (df.date <= p[1])
    line = [pn]
    for grp, gm in (("주식", ~df.is_etf), ("ETF", df.is_etf)):
        for k in ("K0", "K3n"):
            z = df["z_" + k]; zp = z.shift(1)
            mu = inp & gm & su & su.shift(1, fill_value=False) & same
            md = inp & gm & sd & sd.shift(1, fill_value=False) & same
            cut_u = (mu & (zp > 0) & (z <= 0)).sum() / max(1e-9, (inp & gm & su).sum() / 252)
            cut_d = (md & (zp < 0) & (z >= 0)).sum() / max(1e-9, (inp & gm & sd).sum() / 252)
            line.append(f"{grp}{k} 상승끊김 {cut_u:.1f} 하락끊김 {cut_d:.1f}")
    print(" | ".join(line))
pd.set_option("display.width", 250)
c = df[valid & su & (df.z_K3n < 0) & (df.z_K0 > 0) & (df.date >= "2024-01-02")][["sym", "date", "C", "T0", "T3", "z_K0", "z_K3n", "r2_60", "slope60"]]
print(c.assign(trend60=(np.expm1(c.slope60 * 60) * 100).round(1)).round(3).to_string(index=False))
def show(sym, a, b):
    g = df[(df.sym == sym) & (df.date >= a) & (df.date <= b)][["date", "C", "rsi", "T0", "T3", "z_K0", "z_K3n", "zz10", "r2_60"]]
    print(f"\n--- {sym} {a}~{b}"); print(g.round(3).to_string(index=False))
for s, a, b in (("GILD", "2024-05-08", "2024-06-05"), ("ROP", "2025-10-03", "2025-10-24"), ("AMKR", "2023-12-28", "2024-01-19"), ("EEM", "2024-07-17", "2024-08-09"), ("EWJ", "2025-12-10", "2026-01-06"), ("AKAM", "2023-12-28", "2024-01-24")):
    show(s, a, b)
