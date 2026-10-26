"""이격 매수가 '아무 때나 사서 같은 RSI 매도'보다 나은지: 깨어 있는 프리·정규장 모든 봉 시가 매수 → 같은 매도 규칙."""
import numpy as np, pandas as pd
from dev_sim import prep, run, SYMS


def next_exit(d, x):
    """봉 k가 매도 가능(직전 봉 RSI > x, 프리·정규, 깨어 있음)이면 elig[k]. nxt[i] = i 이상에서 첫 매도 가능 봉."""
    ses = d["reg"] | d["pre"]
    r_prev = np.r_[np.nan, d["r"][:-1]]
    elig = (r_prev > x) & ses & d["awake"]
    n = len(elig)
    nxt = np.full(n + 1, -1)
    for k in range(n - 1, -1, -1):
        nxt[k] = k if elig[k] else nxt[k + 1]
    return nxt


D = {s: prep(s, 60) for s in SYMS}
t = run(0.02, 70, 60, "hold", {(s, 60): D[s] for s in SYMS})
base = []
for s in SYMS:
    d = D[s]
    nxt = next_exit(d, 70)
    ok = (d["reg"] | d["pre"]) & d["awake"]
    for i in np.where(ok)[0]:
        if i < 61:
            continue
        k = nxt[i + 1]
        if k < 0:
            continue
        base.append((s, d["o"][k] / d["o"][i] - 1, k - i, d["reg"][i]))
b = pd.DataFrame(base, columns=["sym", "ret", "bars", "reg"])
ex = t.ret.mean() - b.ret.mean()
tv = ex / (t.ret.std(ddof=1) / np.sqrt(len(t)))
print(f"이격 매수 {len(t)}회 평균 {t.ret.mean():+.2%} 승률 {(t.ret > 0).mean():.0%} 보유 중앙 {t.bars.median() * 3:.0f}분")
print(f"아무 봉 매수(같은 매도) {len(b)}개 평균 {b.ret.mean():+.2%} 승률 {(b.ret > 0).mean():.0%} 보유 중앙 {b.bars.median() * 3:.0f}분")
print(f"초과 {ex:+.2%}  t={tv:.1f}")
print("\n종목별 (이격 매수 / 아무 봉):")
for s in SYMS:
    a = t[t.sym == s]; c = b[b.sym == s]
    print(f"  {s:5s} {len(a):3d}회 {a.ret.mean():+.2%} 승률 {(a.ret > 0).mean():.0%} | 아무 봉 {c.ret.mean():+.2%} 승률 {(c.ret > 0).mean():.0%}")
t["ses"] = [("정규" if D[s]["reg"][i] else "프리") for s, i in zip(t.sym, t.ei)]
print("\n세션별:", t.groupby("ses").ret.agg(["count", "mean"]).to_string())
t["xk"] = [D[s]["kst"][i] for s, i in zip(t.sym, t.xi)]
print("\n매수 시각(KST) 분포:", t.ekst.map(lambda z: z.hour).value_counts().sort_index().to_dict())
print("매도 시각(KST) 분포:", t.xk.map(lambda z: z.hour).value_counts().sort_index().to_dict())
pd.set_option("display.width", 200)
w = t.sort_values("ret").head(5)[["sym", "ekst", "xk", "ent", "px", "ret", "mae"]]
print("\n가장 나쁜 5건:\n", w.to_string(index=False))
sn = t[t.sym == "SNDK"][["ekst", "xk", "ent", "px", "ret", "mae", "why"]]
print("\nSNDK 전체:\n", sn.to_string(index=False))
