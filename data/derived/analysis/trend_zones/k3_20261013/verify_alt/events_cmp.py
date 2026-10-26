"""급락 사건별 지연 비교: K3n(20,+-0.5) vs K0(20,+-0.5)·K0(10,+-0.5). K3n이 늦은 사례를 뽑는다. 실행: PYTHONIOENCODING=utf-8 py events_cmp.py"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import frontier as FR
A = FR.A
raw = pd.read_parquet(A.PANEL); raw["ts"] = pd.to_datetime(raw["ts"])
df = pd.concat([FR.per_symbol(g) for _, g in raw.groupby("sym", sort=True)], ignore_index=True)
stock = ~df["is_etf"]
common = pd.Series(np.asarray((df["idx"] >= A.MIN_IDX) & np.isfinite(df["rsi"])), index=df.index)
dn_ev = [e for e in A.m4_events(df, common, stock, up=False) if not df.at[e[2], "is_etf"]]
R = df["rsi"].to_numpy()
Z = {k: np.where(common, A.stage5(R, df[c].to_numpy(), 0.5), -9) for k, c in (("K3n", "T3_20"), ("K0", "T0_20"), ("K0_10", "T0_10"))}
rows = []
for sym, p, q, qp in dn_ev:
    r = {"sym": sym, "p": df.at[p, "date"], "q": df.at[q, "date"], "qp": qp, "drop": df.at[q, "C"] / df.at[p, "C"] - 1}
    for k, z in Z.items():
        h = np.flatnonzero(z[p:q + 21] < 0)
        r[k] = int(h[0]) if len(h) else 36
    rows.append(r)
t = pd.DataFrame(rows)
for b in ("K0", "K0_10"):
    d = t["K3n"] - t[b]
    print(f"K3n vs {b}: 빠름 {np.mean(d<0):.3f}, 같음 {np.mean(d==0):.3f}, 늦음 {np.mean(d>0):.3f}, 차이 중앙 {d.median():+.0f}, 평균 {d.mean():+.2f}")
t["late_vs_K0"] = t["K3n"] - t["K0"]
print("\nK3n이 K0(20)보다 가장 늦은 10건:")
print(t.sort_values("late_vs_K0", ascending=False).head(10).to_string(index=False))
t.to_csv(os.path.join(HERE, "events_lag.csv"), index=False, encoding="utf-8-sig")
# 늦은 사건의 공통점: 20일 전 기준가가 직전 급등 꼭대기였는가 (C[p-20]/mean(C[p-19..p]))
late = t[t["late_vs_K0"] >= 3]
print("\n늦음(>=3일) 건수:", len(late), "/", len(t))
