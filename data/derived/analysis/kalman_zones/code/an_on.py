import pandas as pd, numpy as np
pd.set_option("display.width", 260)
F = lambda x: f"{x:+.1%}"
B = pd.read_parquet("run_on_base.parquet"); T = pd.read_parquet("run_on_stress.parquet")
REF = {"long": ("+275.6%", "+289.8%"), "y2025": ("+948.1%", "+833.2%"), "short": ("+18.2%", "+33.4%")}
for win in ("long", "y2025", "short"):
    b = B[B.win == win]; t = T[T.win == win]
    ctrl = b[b.entry.str.startswith("참고")].iloc[0]
    z = b[~b.entry.str.startswith("참고")].copy()
    z = z.merge(t[~t.entry.str.startswith("참고")][["entry", "filter", "tgt", "hold", "all_ret", "all_sh", "all_mdd"]],
                on=["entry", "filter", "tgt", "hold"], suffixes=("", "_st"))
    print(f"\n===== {win}: 보유 {REF[win][0]}, 현재 지표 {REF[win][1]}, 참고(종가→시가) {F(ctrl.all_ret)} =====")
    print(f"  400가지 전체 수익 중앙 {F(z.all_ret.median())}, 상위10% {F(z.all_ret.quantile(0.9))} | 스트레스(비용0.2/0.1%+지정가 0.1% 관통) 중앙 {F(z.all_ret_st.median())}")
    key = "sel_sh" if win != "short" else "all_sh"
    top = z.sort_values(key, ascending=False).head(12)
    print(top[["entry", "filter", "tgt", "hold", "trades_per_day", "win_rate", "avg_trade", "sel_ret", "val_ret", "all_ret", "all_sh", "all_mdd", "all_ret_st"]]
          .to_string(index=False, formatters={c: F for c in ["avg_trade", "sel_ret", "val_ret", "all_ret", "all_mdd", "all_ret_st"]} |
                     {"trades_per_day": "{:.2f}".format, "win_rate": "{:.0%}".format, "all_sh": "{:.2f}".format}))
    print("  매수 자리별 중앙값 (전체 수익 / 스트레스 / 하루 체결수):")
    g = z.groupby("entry")[["all_ret", "all_ret_st", "trades_per_day", "win_rate"]].median().sort_values("all_ret", ascending=False)
    print(g.to_string(formatters={"all_ret": F, "all_ret_st": F, "trades_per_day": "{:.2f}".format, "win_rate": "{:.0%}".format}))
    print("  필터별 중앙값:", {k: F(v) for k, v in z.groupby("filter").all_ret.median().items()},
          "| 목표별:", {f"{k:.1%}": F(v) for k, v in z.groupby("tgt").all_ret.median().items()},
          "| 보유일별:", {k: F(v) for k, v in z.groupby("hold").all_ret.median().items()})
