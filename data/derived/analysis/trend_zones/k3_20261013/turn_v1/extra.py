import json, numpy as np, pandas as pd
import importlib.util, sys
spec = importlib.util.spec_from_file_location("v1", "v1.py"); v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
df = pd.read_parquet("frame.parquet")
S = df.valid & ~df.etf; E = df.valid & df.etf
out = {}
# θ별 2026 score 차이
for c in ("b4", "b3"):
    out[c + "_dscore26_by_theta"] = {th: v1.score(df, c, S, ("2026H1", "2026H2"), (th,)) - v1.score(df, "b0", S, ("2026H1", "2026H2"), (th,)) for th in v1.THS}
# 2026 B4 고유 변화 전체 목록(주식+ETF)
m26 = df.valid & (df.date >= "2026-01-02") & (df.b4 != df.b0)
t = df[m26][["sym", "etf", "date", "C", "T", "rsi", "dd", "ru", "u", "b0", "b4", "LZ5", "LZ10", "LZV1", "LZV2", "f10"]].copy()
t["ddpct"] = (np.exp(t.dd) - 1) * 100; t["dd_u"] = t.dd / t.u; t["ru_u"] = t.ru / t.u
out["b4_changed_2026"] = t.round(4).to_dict("records")
out["b4_changed_etf_all"] = df[E & (df.b4 != df.b0)][["sym", "date", "b0", "b4"]].to_dict("records")
# 스윙별 지연 개선: B4가 B0보다 빨라진 Z10 스윙 수
raw_pivs = None
# 지그재그 다시(주식, Z10)
cnt = {"faster": 0, "slower": 0, "same": 0}
for s, g in df[~df.etf].groupby("sym"):
    lc = np.log(g.C.to_numpy()); th = np.where(np.isfinite(g.sd20.to_numpy()), 0.10, np.nan)
    lab, piv = v1.zz(lc, th)
    z0 = np.sign(g.b0.to_numpy()); z4 = np.sign(g.b4.to_numpy()); val = g.valid.to_numpy()
    for a, b in zip(piv[:-1], piv[1:]):
        if not val[a + 1:b + 1].all():
            continue
        d = 1 if lc[b] > lc[a] else -1
        def lag(z):
            h = np.flatnonzero(z[a + 1:b + 1] == d)
            return h[0] + 1 if len(h) else b - a + 1
        l0, l4 = lag(z0), lag(z4)
        cnt["faster" if l4 < l0 else ("slower" if l4 > l0 else "same")] += 1
out["z10_swing_lag_b4_vs_b0"] = cnt
# 10일 최고 대비 −7% 이하(RSI>30)인데 애매·상방: B0과 B4 비율, 주식 2026
m = df.valid & ~df.etf & (df.date >= "2026-01-02") & (df.dd <= np.log(0.93)) & (df.rsi > 30)
out["dd7_2026_stock"] = {"rows": int(m.sum()), "b0_nonneg": float((df.loc[m, "b0"] >= 0).mean()), "b4_nonneg": float((df.loc[m, "b4"] >= 0).mean()),
                         "b3_nonneg": float((df.loc[m, "b3"] >= 0).mean()),
                         "in_Z10_down_rows": int((m & (df.LZ10 == -1)).sum()),
                         "in_Z10_down_b4_nonneg": float((df.loc[m & (df.LZ10 == -1), "b4"] >= 0).mean()),
                         "median_dd_u": float((df.loc[m, "dd"] / df.loc[m, "u"]).median()),
                         "share_dd_u_le_m15": float(((df.loc[m, "dd"] / df.loc[m, "u"]) <= -1.5).mean())}
# 덮어쓰기 문턱 u의 크기: 2026 주식 중앙 u (=1.5u가 몇 %인가)
mm = df.valid & ~df.etf & (df.date >= "2026-01-02")
out["u_2026_stock_quantiles_pct_1p5u"] = [float((1 - np.exp(-1.5 * q)) * 100) for q in df.loc[mm, "u"].quantile([.1, .5, .9])]
for s in ("SPY", "QQQ", "EWJ", "EEM", "SOXX", "EWY"):
    out.setdefault("u_2026_etf_1p5u_pct_median", {})[s] = float((1 - np.exp(-1.5 * df.loc[df.valid & (df.sym == s) & (df.date >= "2026-01-02"), "u"].median())) * 100)
json.dump(out, open("extra.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k not in ("b4_changed_2026",)}, ensure_ascii=False, indent=0, default=float))
for r in out["b4_changed_2026"]:
    print(r)
