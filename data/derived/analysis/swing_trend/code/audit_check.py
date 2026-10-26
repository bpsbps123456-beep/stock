"""audit 1: swing.py vs 독립 재계산(audit_ref.py) 대조 + Z5 Pine 정의 대조 + MU 시운전 재현.
→ results/audit_recompute.csv, results/audit_z5_pine.csv, 표준출력 요약"""
from __future__ import annotations

import pathlib
import sys
from datetime import date

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import audit_ref as R  # noqa: E402
import swing as S  # noqa: E402

PER = {"P1": (date(2026, 1, 2), date(2026, 5, 29)), "P2": (date(2026, 6, 1), date(2026, 10, 22)),
       "Y25": (date(2025, 1, 2), date(2025, 12, 31))}
SYMS = ["MU", "BE", "NBIS", "NVDA", "SPY"]
SIX = ["ZZP_th0.1", "ZZA_k3.0", "CR_n10_b0.5_s0.5", "EMA_f10_s30", "DON_n20_m10", "Z5"]
COSTS = [0.0, 0.0005, 0.001]

raw = pd.read_parquet(ROOT / "cache" / "daily.parquet")
G = {s: g.sort_values("date").reset_index(drop=True) for s, g in raw.groupby("sym")}


def pidx(g, per):
    a, b = PER[per]
    d = np.array(g["date"].tolist())
    return [i for i, x in enumerate(d) if a <= x <= b]


rows = []
state_mis = []
for s in SYMS:
    g = G[s]
    sw = S.states(g)
    rf = R.six_states(g)
    for nm in SIX:
        a, b = sw[nm], rf[nm]
        state_mis.append((s, nm, int((a != b).sum()), len(a)))
    O = g.O.to_numpy(float); C = g.C.to_numpy(float)
    for per in PER:
        ix = pidx(g, per)
        if not ix:
            continue
        j0, j1 = ix[0], ix[-1]
        assert list(S.period_index(g, per)) == ix, (s, per)
        for nm in SIX:
            for cost in COSTS:
                r1, m1, h1, e1, _ = S.sim_period(O, C, sw[nm], j0, j1, cost)
                r2, m2, h2, e2, _ = R.ledger_sim(O, C, rf[nm], j0, j1, cost)
                sb1 = S.static_bh(O, C, j0, j1, h1)
                sm2 = R.static_mix(O, C, j0, j1, h2)
                bh2 = R.static_mix(O, C, j0, j1, 1.0)
                rows.append(dict(sym=s, per=per, det=nm, cost=cost, warm=j0, days=j1 - j0 + 1,
                                 ret_sw=r1, ret_ref=r2, mdd_sw=m1, mdd_ref=m2, hold_sw=h1, hold_ref=h2,
                                 ent_sw=e1, ent_ref=e2, eqw_sw=sb1[0], eqw_ref=sm2[0], eqwmdd_sw=sb1[1], eqwmdd_ref=sm2[1],
                                 bh_sw=sb1[2], bh_ref=bh2[0], bhmdd_sw=sb1[3], bhmdd_ref=bh2[1]))
df = pd.DataFrame(rows)
pairs = [("ret", "ret"), ("mdd", "mdd"), ("hold", "hold"), ("ent", "ent"), ("eqw", "eqw"), ("eqwmdd", "eqwmdd"), ("bh", "bh"), ("bhmdd", "bhmdd")]
for a, _ in pairs:
    df[f"d_{a}"] = (df[f"{a}_sw"] - df[f"{a}_ref"]).abs()
df.to_csv(ROOT / "results" / "audit_recompute.csv", index=False)
print("상태 불일치(전 기간 날 수):")
for t in state_mis:
    print("  ", t)
print(f"비교 {len(df)}건(5종목 × 구간 × 판정 6 × 비용 3). 최대 절대 차이:")
print(df[[c for c in df.columns if c.startswith("d_")]].max().to_string())
print("1e-9 넘는 건수:", int((df[[c for c in df.columns if c.startswith("d_")]] > 1e-9).any(axis=1).sum()))
print("예열 부족(j0<60):", df.loc[df.warm < 60, ["sym", "per", "warm"]].drop_duplicates().to_string(index=False))

# --- MU 시운전 재현 (비용 0.05%) ---
g = G["MU"]; O = g.O.to_numpy(float); C = g.C.to_numpy(float)
st = R.six_states(g)["ZZA_k3.0"]
for per in ("P1", "P2"):
    ix = pidx(g, per); j0, j1 = ix[0], ix[-1]
    r, m, h, e, tr = R.ledger_sim(O, C, st, j0, j1, 0.0005)
    eq, _ = R.static_mix(O, C, j0, j1, h)
    bh, _ = R.static_mix(O, C, j0, j1, 1.0)
    print(f"MU {per}: 보유 {bh:+.1%}  ZZA k3 {r:+.1%}  같은 비중 {h:.2f} {eq:+.1%}  진입 {e}")

# --- Z5: swing 방식 vs Pine 그대로 ---
zr = []
for s, g in G.items():
    C = g.C.to_numpy(float)
    a = S.states(g)["Z5"]
    zsw = R.z5_zone(C, pine=False)
    zpn = R.z5_zone(C, pine=True)
    b = R.z5_hold(zpn)
    d = np.array(g["date"].tolist())
    for per, (p0, p1) in PER.items():
        m = (d >= p0) & (d <= p1)
        if not m.any():
            continue
        j0 = int(np.flatnonzero(m)[0])
        both = m & ~np.isnan(zsw) & ~np.isnan(zpn)
        zr.append(dict(sym=s, per=per, warm=j0, days=int(m.sum()),
                       hold_mis=int((a[m] != b[m]).sum()),
                       zone_mis=int((zsw[both] != zpn[both]).sum()),
                       zone_na_pine=int(np.isnan(zpn[m]).sum())))
zr = pd.DataFrame(zr)
zr.to_csv(ROOT / "results" / "audit_z5_pine.csv", index=False)
ok = zr.warm >= 60
print("Z5 swing vs Pine — 예열 60일 이상 종목·구간:", int(ok.sum()), "건, 보유 상태 다른 날 합", int(zr[ok].hold_mis.sum()),
      ", 5단계 값 다른 날 합", int(zr[ok].zone_mis.sum()))
print("Z5 예열 60일 미만(빼야 할 종목·구간):")
print(zr[~ok].to_string(index=False))
print("예열 60~129일 구간에서 상태 차이:")
mid = zr[(zr.warm >= 60) & (zr.warm < 160)]
print(mid.to_string(index=False))
