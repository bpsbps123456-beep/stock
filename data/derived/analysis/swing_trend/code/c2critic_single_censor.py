"""반박 검산: 시험 1 통과 칸을 (a) 잘린 사건 제외, (b) 이진 단서 0/1 분할로 다시 잴 때 P2 차이 부호가 유지되는가.
3분위 경계는 c2single_cells.csv의 cut_lo/cut_hi(P1)를 그대로 쓴다. 위 = x>cut_hi, 아래 = x<=cut_lo (c2single과 같은 규칙인지 원값 재현으로 확인)."""
import numpy as np, pandas as pd
cells = pd.read_csv("results/c2single_cells.csv")
out = []
for det in ["ZZA3", "ZZP10"]:
    ev = pd.read_parquet(f"results/events_{det}.parquet")
    for _, r in cells[cells.det == det].iterrows():
        e = ev[(ev.type == r.type) & (ev.period == "P2")].dropna(subset=[r.outcome, r.feature])
        def diff(e):
            x = e[r.feature]
            if r.binary:
                top = e[x == 1]; bot = e[x == 0]
            else:
                top = e[x > r.cut_hi]; bot = e[x <= r.cut_lo]
            return top[r.outcome].mean() - bot[r.outcome].mean(), len(top), len(bot), top.date.nunique(), bot.date.nunique()
        d0 = diff(e)
        d1 = diff(e[~e.censored])
        import datetime as _dt; e2 = e[e.date <= _dt.date(2026, 9, 22)]
        d2 = diff(e2)
        out.append(dict(det=det, type=r.type, feat=r.feature, out=r.outcome, pass_shift=r.pass_shift, pass_perm=r.pass_perm,
                        P1=r.P1_diff, P2_orig=r.P2_diff, P2_recalc=d0[0], P2_nocens=d1[0], n_nc=f"{d1[1]}/{d1[2]}", days_nc=f"{d1[3]}/{d1[4]}",
                        P2_to0922=d2[0], Y25=r.Y25_diff))
o = pd.DataFrame(out)
o.to_csv("results/c2critic_single_censor.csv", index=False)
ps = o[o.pass_shift]
pd.set_option("display.width", 250)
print((ps[["det","type","feat","out","P1","P2_orig","P2_recalc","P2_nocens","n_nc","days_nc","P2_to0922","Y25"]].round(3)).to_string())
for det in ["ZZA3", "ZZP10"]:
    q = ps[ps.det == det]
    keep = (np.sign(q.P2_nocens) == np.sign(q.P1)).sum(); keep2 = (np.sign(q.P2_to0922) == np.sign(q.P1)).sum()
    print(det, "통과", len(q), "잘림 제외 후 P1 부호 유지", keep, "| 9/22까지 확정만", keep2, "| 재현 최대 오차", (q.P2_orig - q.P2_recalc).abs().max())
    y1 = q[q.out == "y1"]; print("  y1 통과", len(y1), "잘림 제외 부호 유지", (np.sign(y1.P2_nocens) == np.sign(y1.P1)).sum())
