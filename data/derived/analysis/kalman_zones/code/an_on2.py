import pandas as pd, numpy as np
pd.set_option("display.width", 260)
F = lambda x: f"{x:+.0%}"
L = pd.read_parquet("run_on2_loose.parquet"); S = pd.read_parquet("run_on2_strict.parquet")
K = ["entry", "filter", "tgt", "hold", "pre_buy", "pre_exit"]
M = L.merge(S[["win"] + K + ["all_ret", "all_mdd"]], on=["win"] + K, suffixes=("", "_strict"))
print("프리장 매수·정리 효과 (조합 중앙값, 느슨 체결):")
print((M.pivot_table(index="win", columns=["pre_buy", "pre_exit"], values="all_ret", aggfunc="median") * 100).round(0).to_string())
print("하루 체결 수 중앙값:")
print(M.pivot_table(index="win", columns=["pre_buy", "pre_exit"], values="trades_per_day", aggfunc="median").round(1).to_string())
M["rank"] = M.groupby("win").all_ret.rank(ascending=False)
g = M.groupby(K)
cons = pd.DataFrame({"평균순위": g["rank"].mean()})
for w, lab in (("long", "2026"), ("y2025", "2025"), ("short", "최근60일")):
    x = M[M.win == w].set_index(K)
    cons[lab] = x.all_ret.map(F)
    cons[lab + " 낙폭"] = x.all_mdd.map(F)
    cons[lab + " 엄격"] = x.all_ret_strict.map(F)
    if w == "long":
        cons["하루체결"] = x.trades_per_day.map("{:.1f}".format)
        cons["승률"] = x.win_rate.map("{:.0%}".format)
print("\n세 기간 평균 순위 상위 12 (수수료 0%, 닿으면 체결)")
print(cons.sort_values("평균순위").head(12).round(1).to_string())
base = M[(M.entry == "종가-1%") & (M["filter"] == "없음") & (M.tgt == 0.005) & (M.hold == 1)]
print("\n종가-1%·+0.5%·1세션의 프리장 사용별:")
print(base[["win", "pre_buy", "pre_exit", "all_ret", "all_mdd", "all_ret_strict", "trades_per_day", "win_rate"]]
      .to_string(index=False, formatters={"all_ret": F, "all_mdd": F, "all_ret_strict": F, "trades_per_day": "{:.1f}".format, "win_rate": "{:.0%}".format}))
