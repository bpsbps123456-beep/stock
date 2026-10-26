"""critic: improve 귀무 p를 누출 이동(미래 쪽 거리 < G일) 제외로 다시 잰다."""
import numpy as np, pandas as pd, pathlib
R = pathlib.Path(__file__).resolve().parent.parent / "results"
g = pd.read_parquet(R / "improve_grid.parquet"); g = g[g.sym == "MU"].copy()
g["ctag"] = np.where(g.cost == 0, "c0", "sens")
best = g.loc[g.groupby(["var", "ctag"]).ret_sel.idxmax()]
n = pd.read_parquet(R / "improve_null_MU.parquet"); f = pd.read_parquet(R / "improve_null_fixed_MU.parquet")
D = int(n["shift"].max()) + 1
print("D", D, "shifts", n["shift"].nunique())
pct = lambda x: (np.exp(x) - 1) * 100
for G in (0, 5, 10, 15):
    for lo in (0, 15):
        ok = lambda s: (s <= D - G - 0) & (s > lo) if G else (s > lo)
        for _, r in best.iterrows():
            if r["var"] not in ("P", "T1", "S2") : continue
            q = n[(n["var"] == r["var"]) & (n.ctag == r.ctag)]; q = q[(D - q["shift"] >= max(G,1)) & (q["shift"] > lo)]
            ff = f[(f["var"] == r["var"]) & (f.ctag == r.ctag)]; ff = ff[(D - ff["shift"] >= max(G,1)) & (ff["shift"] > lo)]
            pp = (1 + (q.ret_conf >= r.ret_conf).sum()) / (1 + len(q))
            pf = (1 + (ff.ret_conf >= r.ret_conf).sum()) / (1 + len(ff))
            print(f"G{G} lo{lo} {r['var']:3s} {r.ctag:4s} {r.tf}m {r.sess} N{r.N} {r.b}/{r.s} conf {pct(r.ret_conf):+.1f}% "
                  f"proc n={len(q)} q95 {pct(q.ret_conf.quantile(.95)):+.1f}% p={pp:.3f} | fixed n={len(ff)} q95 {pct(ff.ret_conf.quantile(.95)):+.1f}% p={pf:.3f}")
# 이동별 고른 칸 확인 수익 (P, c0): 끝 쪽 이동이 튀는지
q = n[(n["var"] == "P") & (n.ctag == "c0")].sort_values("shift")
print(q[["shift", "tf", "sess", "N", "b", "s"]].assign(sel=pct(q.ret_sel).round(1), conf=pct(q.ret_conf).round(1)).tail(12).to_string())
print(q[["shift"]].assign(conf=pct(q.ret_conf).round(1)).head(8).to_string())
