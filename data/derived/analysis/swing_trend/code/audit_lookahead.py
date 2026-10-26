"""audit 2: 미리보기 감사.
(가) 자료를 날짜 T에서 자른 사본으로 판정 28개(swing) + 독립 6개(audit_ref)를 다시 계산해 T까지 상태가 같은지.
(나) 지그재그 사후 반전점: 자른 사본의 반전점 = 전체에서 확정 날 ≤ T인 반전점.
(다) sim_period 자산: 자른 사본(구간 끝 = T)과 전체의 T 전날까지 자산이 같은지.
(라) 체결 시점·구간 경계 회계: 합성 자료로 '다음 날 시가 체결'과 'j0 전날 상태로 첫날 보유'를 확인.
→ results/audit_lookahead.csv"""
from __future__ import annotations

import pathlib
import random
import sys
from datetime import date

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import audit_ref as R  # noqa: E402
import swing as S  # noqa: E402

raw = pd.read_parquet(ROOT / "cache" / "daily.parquet")
G = {s: g.sort_values("date").reset_index(drop=True) for s, g in raw.groupby("sym")}
SYMS = ["MU", "BE", "NBIS", "NVDA", "SPY", "SNDK", "CRWV"]
cal = sorted(G["SPY"]["date"].tolist())
pool = [d for d in cal if date(2025, 3, 3) <= d <= date(2026, 10, 21)]
random.seed(20261025)
TS = sorted(set(random.sample(pool, 10) + [date(2025, 12, 31), date(2026, 1, 2), date(2026, 5, 29), date(2026, 6, 1),
                                          date(2026, 6, 25), date(2026, 7, 17), date(2026, 10, 1)]))

rows = []
for s in SYMS:
    g = G[s]
    full = S.states(g)
    fref = R.six_states(g)
    C = g.C.to_numpy(float); O = g.O.to_numpy(float)
    a20 = S.atr(g.H.to_numpy(float), g.L.to_numpy(float), C, 20)
    zfull = {"ZZP15": S.zigzag(C, 0.15, np.zeros(0)), "ZZA3": S.zigzag(C, 0.0, 3.0 * a20)}
    dts = g["date"].tolist()
    for T in TS:
        if T not in dts:
            continue
        t = dts.index(T)
        if t < 60:
            continue
        gt = g.iloc[: t + 1].reset_index(drop=True)
        part = S.states(gt)
        pref = R.six_states(gt)
        mis = sum(int((full[k][: t + 1] != part[k]).sum()) for k in full)
        mis_ref = sum(int((fref[k][: t + 1] != pref[k]).sum()) for k in fref)
        Ct = gt.C.to_numpy(float)
        at = S.atr(gt.H.to_numpy(float), gt.L.to_numpy(float), Ct, 20)
        zpart = {"ZZP15": S.zigzag(Ct, 0.15, np.zeros(0)), "ZZA3": S.zigzag(Ct, 0.0, 3.0 * at)}
        piv_bad = 0
        for k in zfull:
            st_f, pi_f, pc_f, pt_f = zfull[k]
            st_p, pi_p, pc_p, pt_p = zpart[k]
            keep = pc_f <= t
            same = (np.array_equal(pi_f[keep], pi_p) and np.array_equal(pc_f[keep], pc_p) and np.array_equal(pt_f[keep], pt_p))
            piv_bad += 0 if same else 1
        # (다) 자산 경로: 구간 시작 = T 기준 80거래일 전
        j0 = t - 80
        eq_bad = 0
        for k in ("ZZA_k3.0", "CR_n10_b0.5_s0.5", "Z5"):
            ef = S.sim_period(O, C, full[k], j0, len(C) - 1, 0.0005)[4]
            ep = S.sim_period(gt.O.to_numpy(float), Ct, part[k], j0, t, 0.0005)[4]
            n = t - j0  # 날 j0..t−1의 끝 자산(O[j+1]까지)
            eq_bad += int(np.max(np.abs(ef[:n] - ep[:n])) > 1e-12)
        rows.append(dict(sym=s, T=T, t=t, state_mis_28=mis, state_mis_ref6=mis_ref, pivot_mismatch=piv_bad, eq_mismatch=eq_bad))
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "audit_lookahead.csv", index=False)
print(f"자르기 시험 {len(df)}건(종목 {df.sym.nunique()} × T {df['T'].nunique()}개 중 상장 뒤 60일 넘은 것)")
print("상태 불일치 합(28개):", int(df.state_mis_28.sum()), " 독립 6개:", int(df.state_mis_ref6.sum()),
      " 반전점 불일치:", int(df.pivot_mismatch.sum()), " 자산 불일치:", int(df.eq_mismatch.sum()))
print(df.groupby("sym").size().to_string())

# ---- (라) 합성 자료 체결 시점 ----
o = np.array([100, 101, 103, 106, 110, 115, 121, 128, 136, 145], float)
c = o * 1.004  # 종가는 시가와 다르게
st = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1], np.int64)  # 2일 종가 매수 판정, 5일 종가 매도 판정, 7일 종가 매수 판정
j0, j1 = 1, 9
r, mdd, hold, ent, eq = S.sim_period(o, c, st, j0, j1, 0.0)
# 기대: 보유 날 j = st[j−1]=1인 날 → j = 3,4,5,8,9. 날 j 수익 O[j+1]/O[j] (끝날은 C/O)
exp = (o[4] / o[3]) * (o[5] / o[4]) * (o[6] / o[5]) * (o[9] / o[8]) * (c[9] / o[9]) - 1
print(f"합성 체결: sim {r:.10f}  기대(다음 날 시가 진입·시가 청산) {exp:.10f}  진입 {ent}(기대 2)")
# 같은 날 종가 체결이었다면
alt = (c[5] / c[2]) * (c[9] / c[7]) - 1
print(f"  비교: 판정 날 종가 체결이었다면 {alt:.10f} (다르면 시점 구분됨)")
# 구간 경계: j0 전날 상태 1이면 비용 없이 O[j0]부터 보유, 진입 수 0
st2 = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0], np.int64)
r2, _, h2, e2, _ = S.sim_period(o, c, st2, 2, 6, 0.01)
exp2 = (o[3] / o[2]) * (o[4] / o[3]) * (o[5] / o[4]) * (1 - 0.01) - 1  # 날 2·3·4 보유, 5일 시가 청산 비용 1%
print(f"경계: sim {r2:.10f}  기대 {exp2:.10f}  진입 {e2}(기대 0, 첫날 보유는 진입으로 안 셈)  평균 보유 {h2:.2f}(기대 0.60)")
# 구간 첫날 판정 변경: j0 전날 0, j0 종가에 1 → j0+1 시가 진입, 비용 1회
st3 = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], np.int64)
r3, _, h3, e3, _ = S.sim_period(o, c, st3, 3, 6, 0.01)
exp3 = (1 - 0.01) * (o[5] / o[4]) * (o[6] / o[5]) * (c[6] / o[6]) - 1
print(f"첫날 종가 매수: sim {r3:.10f}  기대 {exp3:.10f}  진입 {e3}(기대 1)")

# 실제 매매 장부 예: MU ZZA k3 P2
g = G["MU"]; O = g.O.to_numpy(float); C = g.C.to_numpy(float)
stm = S.states(g)["ZZA_k3.0"]
ix = S.period_index(g, "P2"); j0, j1 = int(ix[0]), int(ix[-1])
_, _, _, _, trades = R.ledger_sim(O, C, stm, j0, j1, 0.0005)
dts = g["date"].tolist()
print(f"MU ZZA k3 P2: j0 전날 상태 {stm[j0 - 1]}")
for a, b, pa, pb in trades:
    sig_a = dts[a - 1] if a > j0 else "(구간 전)"
    print(f"  판정 {sig_a} → 시가 진입 {dts[a]} {pa:.2f} | 판정 {dts[b - 1]} → 청산 {dts[b]} {pb:.2f} ({pb / pa - 1:+.1%})")
