import pandas as pd, numpy as np
from evalcfg import evaluate
F = lambda x: f"{x:+.1%}"
P = dict(tf=240, eth=False, L=21, norm="rms", n=50, a=0.75, b=1.75)
for win in ("long", "short"):
    m, per, *_ = evaluate(win, **P)
    bh = pd.read_parquet("grid_long2.parquet" if win == "long" else "grid_short2.parquet").query("kind=='buyhold'").iloc[0]
    print(f"[{win}] 전체 {F(m['all_ret'])} 샤프 {m['all_sh']:.2f} 낙폭 {F(m['all_mdd'])} | 선택 {F(m['sel_ret'])} 확인 {F(m['val_ret'])} (낙폭 {F(m['val_mdd'])}) 체결 {m['trades']} 노출 {m['expo']:.0%}"
          f" || 보유 전체 {F(bh.all_ret)} 샤프 {bh.all_sh:.2f} 낙폭 {F(bh.all_mdd)}")
    print("   종목별:", {k: F(v) for k, v in per.items()})
    for c in (0.002, 0.003):
        mc = evaluate(win, **P, cost=c)[0]
        print(f"   시가 체결 비용 편도 {c:.1%}: 전체 {F(mc['all_ret'])} 확인 {F(mc['val_ret'])}")
# 이웃 안정성: L·a·b 한 칸씩
print("\n이웃 값 (2026년 창 확인 구간 / 최근60일 전체)")
LS = [13, 21, 34]; AS = [0.5, 0.75, 1.0]; BS = [1.5, 1.75, 2.0]
res = []
for L in LS:
    for a in AS:
        for b in BS:
            ml = evaluate("long", 240, False, L, "rms", 50, a, b)[0]
            ms = evaluate("short", 240, False, L, "rms", 50, a, b)[0]
            res.append((L, a, b, ml["val_ret"], ml["all_ret"], ml["all_mdd"], ms["all_ret"]))
r = pd.DataFrame(res, columns=["L", "a", "b", "long_val", "long_all", "long_mdd", "short_all"])
print(r.to_string(index=False, formatters={c: F for c in ["long_val", "long_all", "long_mdd", "short_all"]}))
print("이웃 27개: 2026년 확인 구간 최소 {} 중앙 {} / 최근60일 최소 {} 중앙 {}".format(F(r.long_val.min()), F(r.long_val.median()), F(r.short_all.min()), F(r.short_all.median())))
