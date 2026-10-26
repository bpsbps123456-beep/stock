import pandas as pd, numpy as np
pd.set_option("display.width", 260)
F = lambda x: f"{x:+.0%}"
R = pd.read_parquet("run_adapt.parquet")
K = ["trend", "up", "dn", "tgt", "pre_from"]
L = R[~R.close_fill].copy(); C = R[R.close_fill].copy()
M = L.merge(C[["win"] + K + ["all_ret", "all_mdd"]], on=["win"] + K, suffixes=("", "_cf"))
M["rk"] = M.groupby("win").all_ret.rank(ascending=False)
M["rk_cf"] = M.groupby("win").all_ret_cf.rank(ascending=False)
g = M.groupby(K)
T = pd.DataFrame({"평균순위(닿으면)": g.rk.mean(), "평균순위(종가)": g.rk_cf.mean()})
T["종합"] = (T["평균순위(닿으면)"] + T["평균순위(종가)"]) / 2
for w, lab in (("long", "26"), ("y2025", "25"), ("short", "최근")):
    x = M[M.win == w].set_index(K)
    T[f"{lab}"] = x.all_ret.map(F); T[f"{lab} 종가"] = x.all_ret_cf.map(F); T[f"{lab} 낙폭"] = x.all_mdd.map(F)
    if w == "long":
        T["하루체결26"] = x.trades_per_day.map("{:.1f}".format)
def lab(i):
    tr, up, dn, tgt, pf = i
    d = "안 삼" if dn < 0 else f"-{dn*100:g}%"
    return f"{tr} | 추세 -{up*100:g}% / 아니면 {d} | 목표 {tgt*100:g}% | 프리 {pf}시~"
T.index = [lab(i) for i in T.index]
print("종합 순위 상위 12 (세 기간 × 두 체결 방식 평균)")
print(T.sort_values("종합").head(12).round(1).to_string())
print("\n비교 기준: 추세 무관 고정 하락폭")
base = T[T.index.str.startswith("(추세 무관)")].sort_values("종합")
print(base.head(8).round(1).to_string())
print("\n추세 정의별 최고 종합순위:")
for tr in ["구간 매수·과매수", "칼만선 상승", "종가>칼만선"]:
    b = T[T.index.str.startswith(tr)].sort_values("종합").iloc[0]
    print(f"  {tr}: 종합 {b['종합']:.1f} | 26 {b['26']} (종가 {b['26 종가']}) | 25 {b['25']} ({b['25 종가']}) | 최근 {b['최근']} ({b['최근 종가']})")
