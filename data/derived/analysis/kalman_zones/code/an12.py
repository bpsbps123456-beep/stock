"""개장 후 봉에도 반응(기본) vs 개장 전 완성봉만 보고 판단 — 시간봉별 비교."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
a = pd.read_parquet("dense.parquet"); a = a[(a.win == "short")]
b = pd.read_parquet("dense_preopen.parquet")
def summ(df):
    z = df[df.kind == "zone5"]; o = df[df.kind == "original"].set_index(["tf", "eth"])
    g = z.groupby(["tf", "eth"])
    fx = z[(z.hk == 10) & (z.norm == "rms") & (z.hn == 10) & (z.a == 0.75) & (z.b == 1.75) & z.tp_half & z.panic_hold].set_index(["tf", "eth"])
    return pd.DataFrame({"med_sh": g.all_sh.median(), "p90_sh": g.all_sh.quantile(0.9), "med_ret": g.all_ret.median(),
                         "fix_ret": fx.all_ret, "orig_ret": o.all_ret})
A, B = summ(a), summ(b)
t = A.join(B, lsuffix="_react", rsuffix="_preopen").reset_index()
t["ses"] = np.where(t.eth, "ETH", "RTH")
t.to_csv("dense_compare.csv", index=False)
bands = pd.cut(t.tf, [0, 30, 60, 85, 120, 180, 240, 300, 360], labels=["10-30", "35-60", "65-85", "90-120", "125-180", "185-240", "245-300", "305-360"])
print("시간봉 묶음 평균: 조합 3,360개 샤프 중앙값 / 상위10% / 수익 중앙값 (개장 후에도 반응 → 개장 전만 판단)")
g = t.assign(band=bands).groupby(["ses", "band"], observed=True)
out = g[["med_sh_react", "med_sh_preopen", "p90_sh_react", "p90_sh_preopen", "med_ret_react", "med_ret_preopen", "fix_ret_react", "fix_ret_preopen"]].mean()
print(out.round(3).to_string())
for ses in ("RTH", "ETH"):
    q = t[t.ses == ses]
    print(f"{ses} 순위상관(시간봉↔샤프 중앙값): 반응 {spearmanr(q.tf, q.med_sh_react)[0]:+.2f} → 개장 전만 {spearmanr(q.tf, q.med_sh_preopen)[0]:+.2f}")
print("\n40분 vs 90분 vs 240분 vs 360분")
print(t[t.tf.isin([40, 90, 240, 360])][["tf", "ses", "med_sh_react", "med_sh_preopen", "p90_sh_react", "p90_sh_preopen", "med_ret_react", "med_ret_preopen", "fix_ret_react", "fix_ret_preopen"]].round(3).to_string(index=False))
