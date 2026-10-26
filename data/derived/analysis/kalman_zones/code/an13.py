import pandas as pd
pd.set_option("display.width", 250)
o = pd.read_parquet("run_v2.parquet")
F = lambda x: f"{x:+.1%}"
for (win, sw), g in o.groupby(["win", "swing"], sort=False):
    b = g[(g["mode"] == "M1") & (g.variant == "뼈대만")].iloc[0]
    gg = g[(g["mode"] == "M1") & (g.variant != "뼈대만")].sort_values("sel_sh", ascending=False)
    print(f"\n===== {win} / 스윙 {sw} | 뼈대만(M1): 선택 {F(b.sel_ret)} 샤프 {b.sel_sh:.2f} | 확인 {F(b.val_ret)} 샤프 {b.val_sh:.2f} | 전체 {F(b.all_ret)} 낙폭 {F(b.all_mdd)}")
    print(pd.concat([gg.head(6), gg.tail(3)])[["variant", "sel_ret", "sel_sh", "val_ret", "val_sh", "all_ret", "all_mdd", "sw_trades", "sw_win", "sw_avg"]]
          .to_string(index=False, formatters={c: F for c in ["sel_ret", "val_ret", "all_ret", "all_mdd", "sw_avg"]} |
                     {c: "{:.2f}".format for c in ["sel_sh", "val_sh"]} | {"sw_win": "{:.0%}".format}))
    print(f"  32가지 중 뼈대보다: 선택 샤프 높음 {(gg.sel_sh > b.sel_sh).sum()}개, 확인 수익 높음 {(gg.val_ret > b.val_ret).sum()}개, 전체 수익 높음 {(gg.all_ret > b.all_ret).sum()}개"
          f" | 스윙 평균 수익/회 {gg.sw_avg.mean():+.2%}, 승률 {gg.sw_win.mean():.0%}")
    for key in ["combo", "rsi"]:
        q = gg[gg.variant.str.startswith(key)]
        print(f"   {key}: 전체 수익 중앙 {F(q.all_ret.median())} (뼈대 {F(b.all_ret)}), 확인 중앙 {F(q.val_ret.median())}, 스윙 1회 평균 {q.sw_avg.mean():+.2%}")
    mf = g[(g["mode"] == "MF") & (g.variant != "뼈대만")]
    bf = g[(g["mode"] == "MF") & (g.variant == "뼈대만")].iloc[0]
    print(f"  (참고 MF 밤새 대응) 뼈대 {F(bf.all_ret)} vs 스윙 추가 중앙 {F(mf.all_ret.median())}, 최고 {F(mf.all_ret.max())}, 스윙 1회 평균 {mf.sw_avg.mean():+.2%}")
