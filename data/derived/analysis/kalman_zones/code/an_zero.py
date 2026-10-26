"""수수료·스프레드 0% 기준 재평가: 초단타 624가지(400 + 지지·저항 224)의 세 창 일관 순위, 엄격 체결(0.1% 관통) 비교."""
import numpy as np, pandas as pd
pd.set_option("display.width", 260)
F = lambda x: f"{x:+.0%}"
def load(tag):
    a = pd.read_parquet(f"run_on_{tag}.parquet"); a = a[~a.entry.str.startswith("참고")].copy()
    a["tp"] = a.tgt.map(lambda t: "목표없음" if t == 0 else f"{t*100:g}%")
    b = pd.read_parquet(f"run_sr_{tag}.parquet")
    b = b[b.entry.str.startswith("지지") | b.tp.str.startswith("저항")]      # 지지·저항만 추가(겹치는 조합 제외)
    cols = ["win", "entry", "filter", "tp", "hold", "trades_per_day", "win_rate", "avg_trade", "all_ret", "all_sh", "all_mdd", "sel_ret", "val_ret", "sel_sh"]
    return pd.concat([a[cols], b[cols]], ignore_index=True)
Z, ZS = load("zero"), load("zero_strict")
K = ["entry", "filter", "tp", "hold"]
M = Z.merge(ZS[["win"] + K + ["all_ret", "all_mdd"]], on=["win"] + K, suffixes=("", "_strict"))
print("조합 수:", M.groupby("win").size().to_dict())
for win in ("long", "y2025", "short"):
    z = M[M.win == win]
    print(f"[{win}] 0% 중앙 {F(z.all_ret.median())} / 상위10% {F(z.all_ret.quantile(0.9))} | 엄격 체결 중앙 {F(z.all_ret_strict.median())}")
M["rank"] = M.groupby("win").all_ret_strict.rank(ascending=False)
M["rank0"] = M.groupby("win").all_ret.rank(ascending=False)
g = M.groupby(K)
cons = pd.DataFrame({"avg_rank_strict": g["rank"].mean(), "avg_rank0": g["rank0"].mean()})
for w, lab in (("long", "26"), ("y2025", "25"), ("short", "최근")):
    x = M[M.win == w].set_index(K)
    cons[f"{lab} 0%"] = x.all_ret.map(F); cons[f"{lab} 엄격"] = x.all_ret_strict.map(F)
    if w == "long":
        cons["낙폭26"] = x.all_mdd.map(F); cons["하루체결26"] = x.trades_per_day.map("{:.1f}".format); cons["승률26"] = x.win_rate.map("{:.0%}".format)
print("\n세 창 평균 순위 (엄격 체결 기준) 상위 12")
print(cons.sort_values("avg_rank_strict").head(12).drop(columns=["avg_rank0"]).round(1).to_string())
print("\n세 창 평균 순위 (0%, 느슨한 체결) 상위 8")
print(cons.sort_values("avg_rank0").head(8).drop(columns=["avg_rank_strict"]).round(1).to_string())
# 이전 추천 조합의 0% 성적
prev = M[(M.entry == "종가-2%") & (M["filter"] == "없음") & (M.tp == "1%") & (M.hold == 2)]
print("\n이전 추천(종가-2%, 필터 없음, +1%, 2세션) 0% / 엄격:", {r.win: f"{F(r.all_ret)} / {F(r.all_ret_strict)} (낙폭 {F(r.all_mdd)})" for r in prev.itertuples()})
