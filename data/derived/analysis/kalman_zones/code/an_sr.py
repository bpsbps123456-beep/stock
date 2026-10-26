import pandas as pd, numpy as np
pd.set_option("display.width", 260)
F = lambda x: f"{x:+.1%}"
B = pd.read_parquet("run_sr_base.parquet"); T = pd.read_parquet("run_sr_stress.parquet")
K = ["entry", "filter", "tp", "hold"]
M = B.merge(T[["win"] + K + ["all_ret", "all_sh", "all_mdd", "sel_sh", "val_ret"]], on=["win"] + K, suffixes=("", "_st"))
REF = {"long": "보유 +276% / 현재 지표 +290% / 초단타 400가지 중앙 +47%", "y2025": "보유 +948% / 현재 지표 +833% / 초단타 중앙 +84%",
       "short": "보유 +18% / 현재 지표 +33% / 초단타 중앙 +2.4%"}
for win in ("long", "y2025", "short"):
    z = M[M.win == win]
    print(f"\n===== {win}: {REF[win]} =====")
    print("  매수 자리별 중앙 (기본 / 비용 인상) · 하루 체결 · 승률:")
    g = z.groupby("entry").agg(ret=("all_ret", "median"), ret_st=("all_ret_st", "median"), tpd=("trades_per_day", "median"), wr=("win_rate", "median"))
    print(g.sort_values("ret", ascending=False).to_string(formatters={"ret": F, "ret_st": F, "tpd": "{:.2f}".format, "wr": "{:.0%}".format}))
    print("  목표 방식별 중앙:", {k: F(v) for k, v in z.groupby("tp").all_ret.median().items()},
          "| 비용 인상:", {k: F(v) for k, v in z.groupby("tp").all_ret_st.median().items()})
    key = "sel_sh" if win != "short" else "all_sh"
    top = z.sort_values(key, ascending=False).head(8)
    print(top[K + ["trades_per_day", "win_rate", "avg_trade", "sel_ret", "val_ret", "all_ret", "all_sh", "all_mdd", "all_ret_st"]]
          .to_string(index=False, formatters={c: F for c in ["avg_trade", "sel_ret", "val_ret", "all_ret", "all_mdd", "all_ret_st"]} |
                     {"trades_per_day": "{:.2f}".format, "win_rate": "{:.0%}".format, "all_sh": "{:.2f}".format}))
# 세 창 모두에서 일관된 조합: 각 창 순위의 평균(비용 인상 기준)
M["rank"] = M.groupby("win").all_ret_st.rank(ascending=False)
cons = M.pivot_table(index=K, columns="win", values=["all_ret", "all_ret_st", "trades_per_day", "all_mdd"])
cons["avg_rank"] = M.groupby(K)["rank"].mean()
cons = cons.sort_values("avg_rank").head(10)
print("\n세 창 평균 순위(비용 인상 수익 기준) 상위 10")
flat = pd.DataFrame(index=cons.index)
for w in ("long", "y2025", "short"):
    flat[f"{w} 기본"] = cons[("all_ret", w)].map(F)
    flat[f"{w} 비용↑"] = cons[("all_ret_st", w)].map(F)
flat["낙폭(26)"] = cons[("all_mdd", "long")].map(F)
flat["하루체결(26)"] = cons[("trades_per_day", "long")].map("{:.1f}".format)
flat["평균순위"] = cons["avg_rank"].round(1)
print(flat.to_string())
