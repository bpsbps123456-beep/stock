"""K3n 실패 후보 에피소드를 눈 기준 맥락(얕은 눌림·곧 신고가)으로 거르고 일별 표를 뽑는다.
입력: panel_feat2.parquet, cases_relaxed.csv
실행: PYTHONIOENCODING=utf-8 py cases_detail.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "panel_feat2.parquet"))
cs = pd.read_csv(os.path.join(HERE, "cases_relaxed.csv"))
df = df.sort_values(["sym", "date"]).reset_index(drop=True)
df["hi20"] = df.groupby("sym")["C"].transform(lambda s: s.rolling(20, min_periods=20).max())
df["lo20"] = df.groupby("sym")["C"].transform(lambda s: s.rolling(20, min_periods=20).min())

ctx = []
for r in cs.itertuples():
    g = df[df.sym == r.sym].reset_index(drop=True)
    i0 = g.index[g.date == r.start][0]
    i1 = g.index[g.date == r.end][0]
    seg = g.iloc[i0:i1 + 1]
    if r.kind.startswith("K3n상방"):
        dd = float((seg.C.max() / g.loc[i0, "lo20"] - 1) * 100)       # 20일 저점 대비 반등 최대
        ref = g.loc[max(0, i0 - 20):i0, "C"].min()
        fwd = g.iloc[i1 + 1:i1 + 11]
        resumed = bool(len(fwd) and fwd.C.min() < ref)                 # 10일 안에 새 저점
    else:
        dd = float((seg.C.min() / g.loc[i0, "hi20"] - 1) * 100)       # 20일 고점 대비 최저
        ref = g.loc[max(0, i0 - 20):i0, "C"].max()
        fwd = g.iloc[i1 + 1:i1 + 11]
        resumed = bool(len(fwd) and fwd.C.max() > ref)                 # 10일 안에 신고가(20일)
    f10 = float((g.loc[min(len(g) - 1, i1 + 10), "C"] / g.loc[i1, "C"] - 1) * 100)
    ctx.append({"dd_pct": round(dd, 1), "resumed10": resumed, "fwd10_pct": round(f10, 1),
                "ret60_pct": round(float(np.expm1(g.loc[i0, "ret60"]) * 100), 1), "r2_60": round(float(g.loc[i0, "r2_60"]), 2)})
cs = pd.concat([cs.reset_index(drop=True), pd.DataFrame(ctx)], axis=1)
cs.to_csv(os.path.join(HERE, "cases_relaxed_ctx.csv"), index=False, encoding="utf-8-sig")

# 눈으로도 틀린 장면: 상승 스윙에서 K3n 하방인데 낙폭이 20일 고점 대비 6% 이내이고 10일 안에 신고가
clear_dn = cs[cs.kind.str.startswith(("K3n하방", "꾸준")) & (cs.dd_pct > -6) & cs.resumed10 & (cs.days >= 2)]
clear_up = cs[cs.kind.str.startswith("K3n상방") & (cs.dd_pct < 6) & cs.resumed10 & (cs.days >= 2)]
print("상승 스윙 K3n 하방 에피소드:", int(cs.kind.str.startswith("K3n하방").sum()), "→ 얕은 눌림(>-6%)·10일 내 신고가·2일 이상:", len(clear_dn))
print("하락 스윙 K3n 상방 에피소드:", int(cs.kind.str.startswith("K3n상방").sum()), "→ 얕은 반등(<6%)·10일 내 신저가·2일 이상:", len(clear_up))
pd.set_option("display.width", 250)
print(clear_dn.sort_values(["days", "swing_pct"], ascending=False).head(30).to_string())
print(clear_up.sort_values(["days", "swing_pct"], ascending=False).head(15).to_string())

# 같은 거름을 K0 쪽 실패(K0 하방·K3n 아님 / K0 상방·K3n 아님)에도: 공정 비교
mid = (df.zz10_since >= 5) & (df.zz10_until >= 3) & (df.date >= "2024-01-02") & (df.z_K0 != -9) & (df.z_K3n != -9)


def eps(cond):
    out = []
    for sym, g0 in df[cond].groupby("sym"):
        ids = g0.index.to_numpy()
        starts = np.flatnonzero(np.r_[True, np.diff(ids) != 1])
        ends = np.r_[starts[1:], len(ids)]
        for s0, e0 in zip(starts, ends):
            out.append((sym, ids[s0], ids[e0 - 1]))
    return out


def clear_count(eplist, down):
    n = 0
    for sym, a, b in eplist:
        if b - a + 1 < 2:
            continue
        g = df.loc[max(0, a - 20):b + 10]
        g = g[g.sym == sym]
        seg = df.loc[a:b]
        if down:
            dd = seg.C.min() / df.loc[a, "hi20"] - 1
            ref = df.loc[max(a - 20, g.index.min()):a, "C"].max()
            fwd = g.loc[b + 1:b + 10, "C"]
            ok = dd > -0.06 and len(fwd) and fwd.max() > ref
        else:
            dd = seg.C.max() / df.loc[a, "lo20"] - 1
            ref = df.loc[max(a - 20, g.index.min()):a, "C"].min()
            fwd = g.loc[b + 1:b + 10, "C"]
            ok = dd < 0.06 and len(fwd) and fwd.min() < ref
        n += bool(ok)
    return n


k3_dn = mid & (df.zz10 == 1) & (df.z_K3n < 0) & (df.z_K0 >= 0)
k0_dn = mid & (df.zz10 == 1) & (df.z_K0 < 0) & (df.z_K3n >= 0)
k3_up = mid & (df.zz10 == -1) & (df.z_K3n > 0) & (df.z_K0 <= 0)
k0_up = mid & (df.zz10 == -1) & (df.z_K0 > 0) & (df.z_K3n <= 0)
print("눈으로 분명한 실패(얕은 눌림/반등, 10일 내 추세 재개, 2일 이상) 에피소드 수:")
print("  상승 스윙 중 하방 표시: K3n만", clear_count(eps(k3_dn), True), "/ K0만", clear_count(eps(k0_dn), True))
print("  하락 스윙 중 상방 표시: K3n만", clear_count(eps(k3_up), False), "/ K0만", clear_count(eps(k0_up), False))


def show(sym, a, b):
    g = df[(df.sym == sym) & (df.date >= a) & (df.date <= b)][["date", "C", "rsi", "T0", "T3", "z_K0", "z_K3n", "zz10", "r2_60", "ret60"]]
    g = g.assign(ret60=(np.expm1(g.ret60) * 100).round(1))
    print(f"\n--- {sym} {a}~{b}")
    print(g.round(3).to_string(index=False))


for sym, a, b in [("SMH", "2023-12-27", "2024-01-12"), ("SOXX", "2023-12-27", "2024-01-12"), ("QQQ", "2023-12-27", "2024-01-12"),
                  ("IBKR", "2024-12-03", "2024-12-27"), ("COST", "2024-12-13", "2025-01-08"), ("WMT", "2025-08-15", "2025-09-10"),
                  ("EWT", "2024-08-29", "2024-09-20"), ("QQQ", "2025-11-25", "2025-12-19"), ("SPY", "2026-07-17", "2026-08-07"),
                  ("EWY", "2026-09-28", "2026-10-12"), ("KORU", "2026-09-28", "2026-10-12")]:
    show(sym, a, b)
