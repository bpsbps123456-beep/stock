"""5분 간격 전수 탐색 요약: 시간봉별 분포·고정 설정·선택→확인, 40분 vs 240분."""
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
df = pd.read_parquet("dense.parquet")
z = df[df.kind == "zone5"]
bh = df[df.kind == "buyhold"].set_index("win")
og = df[df.kind == "original"]
HK = [1, 2, 3, 5, 7, 10]
AS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
BS = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
print("총 조합 수:", len(z), "| 창별:", z.groupby("win").size().to_dict())
for w in ("short", "long"):
    r = bh.loc[w]
    print(f"[{w}] 보유: 전체 {r.all_ret:+.1%} 샤프 {r.all_sh:.2f} 낙폭 {r.all_mdd:.1%} | 선택 {r.sel_ret:+.1%} | 확인 {r.val_ret:+.1%} (샤프 {r.val_sh:.2f})")


def robust_pick(g):
    """선택 구간 샤프를 (평활·a·b) ±1칸 이웃 평균으로 매겨 가장 높은 설정."""
    g = g.copy()
    g["hi"] = g.hk.map({v: i for i, v in enumerate(HK)})
    g["ai"] = g.a.map({v: i for i, v in enumerate(AS)})
    g["bi"] = g.b.map({v: i for i, v in enumerate(BS)})
    best = None
    for key, gg in g.groupby(["norm", "hn", "tp_half", "panic_hold"]):
        idx = {(r.hi, r.ai, r.bi): r.sel_sh for r in gg.itertuples()}
        rob = []
        for r in gg.itertuples():
            v = [idx[(r.hi + x, r.ai + y, r.bi + q)] for x in (-1, 0, 1) for y in (-1, 0, 1) for q in (-1, 0, 1)
                 if (r.hi + x, r.ai + y, r.bi + q) in idx]
            rob.append(np.mean(v) if len(v) >= 6 else np.nan)
        gg = gg.assign(rob=rob)
        c = gg.loc[gg.rob.idxmax()]
        if best is None or c.rob > best.rob:
            best = c
    return best


rows = []
for (w, tf, eth), g in z.groupby(["win", "tf", "eth"]):
    fam = g[(g.tp_half) & (g.panic_hold)]
    fx = g[(g.hk == 10) & (g.norm == "rms") & (g.hn == 10) & (g.a == 0.75) & (g.b == 1.75) & g.tp_half & g.panic_hold]
    pk = robust_pick(g)
    o = og[(og.win == w) & (og.tf == tf) & (og.eth == eth)].iloc[0]
    rows.append({"win": w, "tf": int(tf), "ses": "ETH" if eth else "RTH", "n": len(g),
                 "med_sh": g.all_sh.median(), "p90_sh": g.all_sh.quantile(0.9), "fam_med_sh": fam.all_sh.median(),
                 "fam_med_ret": fam.all_ret.median(), "fam_med_mdd": fam.all_mdd.median(),
                 "fix_ret": fx.all_ret.iloc[0] if len(fx) else np.nan, "fix_sh": fx.all_sh.iloc[0] if len(fx) else np.nan,
                 "pick": f"{pk.hk:.0f}일 {pk.norm}{pk.hn:.0f} a{pk.a} b{pk.b} {'절반' if pk.tp_half else '전량'}{'·패닉보유' if pk.panic_hold else ''}",
                 "pick_val": pk.val_ret, "pick_valsh": pk.val_sh, "pick_all": pk.all_ret,
                 "orig_ret": o.all_ret, "orig_sh": o.all_sh})
t = pd.DataFrame(rows)
t.to_csv("dense_summary.csv", index=False)

# 1) 짧은 창: 5분 간격 71개 x 2 — 조합 전체 샤프 중앙값 막대
s = t[t.win == "short"]
print("\n[최근 창 7/22~9/25] 시간봉별 전체 구간 샤프 (조합 3,360개의 중앙값 / 상위10%) — 막대는 중앙값")
piv = s.pivot(index="tf", columns="ses", values=["med_sh", "p90_sh", "fix_ret", "orig_ret"])
mx = max(piv["med_sh"].max().max(), 0.01)
for tf in piv.index:
    r = piv.loc[tf]
    bar = lambda v: "#" * max(0, int(round(v / mx * 20)))
    print(f"{tf:>3}분 RTH {r[('med_sh','RTH')]:5.2f}/{r[('p90_sh','RTH')]:4.2f} {bar(r[('med_sh','RTH')]):<20} | "
          f"ETH {r[('med_sh','ETH')]:5.2f}/{r[('p90_sh','ETH')]:4.2f} {bar(r[('med_sh','ETH')]):<20} | "
          f"고정설정 RTH {r[('fix_ret','RTH')]:+.1%} ETH {r[('fix_ret','ETH')]:+.1%} | 원본 RTH {r[('orig_ret','RTH')]:+.1%} ETH {r[('orig_ret','ETH')]:+.1%}")

# 2) 추세: 시간봉과 성적의 순위 상관
from scipy.stats import spearmanr
for ses in ("RTH", "ETH"):
    q = s[s.ses == ses]
    print(f"{ses}: 시간봉↔중앙값 샤프 순위상관 {spearmanr(q.tf, q.med_sh)[0]:+.2f}, ↔고정설정 수익 {spearmanr(q.tf, q.fix_ret)[0]:+.2f}, ↔원본 수익 {spearmanr(q.tf, q.orig_ret)[0]:+.2f}")

# 3) 구간별 평균 (시간봉 묶음)
s = s.assign(band=pd.cut(s.tf, [0, 30, 60, 120, 180, 240, 300, 360], labels=["10-30", "35-60", "65-120", "125-180", "185-240", "245-300", "305-360"]))
print("\n시간봉 묶음별 평균 (최근 창)")
print(s.groupby(["band", "ses"], observed=True)[["med_sh", "p90_sh", "fam_med_ret", "fam_med_mdd", "fix_ret", "orig_ret", "pick_val"]]
      .mean().round(3).to_string())

# 4) 선택 구간으로 고른 설정의 확인 구간 성적 상위/하위
print("\n선택 구간으로 고른 설정 → 확인 구간 수익 상위 10개 시간봉")
print(s.sort_values("pick_val", ascending=False).head(10)[["tf", "ses", "pick", "pick_val", "pick_valsh", "pick_all"]].to_string(index=False))

# 5) 40분 vs 240분 직접 비교
print("\n40분 vs 240분 (최근 창)")
print(s[s.tf.isin([40, 240])][["tf", "ses", "med_sh", "p90_sh", "fam_med_ret", "fam_med_mdd", "fix_ret", "pick", "pick_val", "pick_all", "orig_ret"]].to_string(index=False))

# 6) 2026년 창 (60분 배수 정규장)
print("\n[2026년 창] 60분 배수 정규장")
print(t[t.win == "long"][["tf", "med_sh", "p90_sh", "fam_med_ret", "fam_med_mdd", "fix_ret", "fix_sh", "pick", "pick_val", "pick_valsh", "pick_all", "orig_ret"]].to_string(index=False))
