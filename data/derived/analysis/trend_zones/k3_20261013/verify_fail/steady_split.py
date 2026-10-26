"""꾸준한 추세 안의 반대색을 사후 스윙(지그재그 10%·15%)으로 나눈다: 스윙이 계속된 날(눈으로 틀림) vs 이미 꺾인 날(이른 경고).
입력 panel_feat2.parquet. 실행: PYTHONIOENCODING=utf-8 py steady_split.py"""
import os
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "panel_feat2.parquet"))
raw = pd.read_parquet(os.path.join(HERE, "..", "panel_1d_20261012.parquet"))
df["is_etf"] = df.sym.isin(set(raw.loc[raw.is_etf, "sym"]))
valid = (df.z_K0 != -9) & (df.z_K3n != -9)
thr = np.where(df.is_etf, np.log(1.08), np.log(1.15))
su = (df.slope60 * 60 >= thr) & (df.r2_60 >= 0.8)
sd = (df.slope60 * 60 <= -thr) & (df.r2_60 >= 0.8)
P = {"2024": ("2024-01-02", "2024-12-31"), "2025": ("2025-01-02", "2025-12-31"), "26H1": ("2026-01-02", "2026-05-29"),
     "26H2": ("2026-06-01", "2026-10-12"), "ALL": ("2024-01-02", "2026-10-12")}
for zc in ("zz10", "zz15"):
    print("==", zc)
    for pn, p in P.items():
        M = valid & (df.date >= p[0]) & (df.date <= p[1])
        out = [pn]
        for nm, mk, sgn in (("상승", su, 1), ("하락", sd, -1)):
            m = M & mk
            n = m.sum()
            for k in ("K0", "K3n"):
                z = np.sign(df.loc[m, "z_" + k]); lab = df.loc[m, zc]
                wrong = ((z == -sgn) & (lab == sgn)).sum() / n * 100   # 스윙 계속인데 반대색
                early = ((z == -sgn) & (lab == -sgn)).sum() / n * 100  # 이미 꺾였고 반대색(맞음)
                miss = ((z != -sgn) & (lab == -sgn)).sum() / n * 100   # 이미 꺾였는데 반대색 못 띄움
                out.append(f"{nm}{k} 틀림{wrong:.1f} 이른경고{early:.1f} 놓침{miss:.1f}")
            out.append(f"n={n}")
        print(" | ".join(out))

print("== 꾸준한 상승이고 사후에도 상승 스윙이 이어진 날(zz10=+1): 상방/애매/하방 %  (하락은 거울상)")
for pn, p in P.items():
    M = valid & (df.date >= p[0]) & (df.date <= p[1])
    out = [pn]
    for nm, mk, sgn in (("상승", su, 1), ("하락", sd, -1)):
        m = M & mk & (df.zz10 == sgn)
        for k in ("K0", "K3n"):
            z = np.sign(df.loc[m, "z_" + k]) * sgn
            out.append(f"{nm}{k} 맞는색{(z > 0).mean()*100:.1f} 애매{(z == 0).mean()*100:.1f} 반대{(z < 0).mean()*100:.1f}")
        out.append(f"n={m.sum()}")
    print(" | ".join(out))
