"""audit 6: 보강 점검.
(가) 독립 재계산을 사용 가능 144종목 전체 × P1·P2·Y25 × 판정 6 × 비용 0.05%로 넓힌다.
(나) shift_state(귀무): 원형 이동 칸 수가 D+1(j0−1..j1)이라 평균 보유가 원래와 얼마나 다른지.
(다) 사후 진실 지그재그(θ 10·15·20%, k 3·4·5)에서 P1·P2·Y25에 시작하는 구간 중 끝점이 극값이 아닌 비율.
(라) 열린 마지막 구간: 자료 끝(10-22)에 확정되지 않은 극점까지 거래일 수.
→ results/audit_extra_recompute144.csv, results/audit_extra_shift.csv, results/audit_extra_truth_legs.csv"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import audit_ref as R  # noqa: E402
import swing as S  # noqa: E402

raw = pd.read_parquet(ROOT / "cache" / "daily.parquet")
G = {s: g.sort_values("date").reset_index(drop=True) for s, g in raw.groupby("sym")}
el = pd.read_csv(ROOT / "results" / "audit_eligibility.csv")
OK = {(r.per, r.sym) for r in el.itertuples() if r.ok}
PERS = ("P1", "P2", "Y25")

# (가)
rows = []
for s, g in G.items():
    sw = S.states(g); rf = R.six_states(g)
    O = g.O.to_numpy(float); C = g.C.to_numpy(float)
    for per in PERS:
        if (per, s) not in OK:
            continue
        ix = S.period_index(g, per); j0, j1 = int(ix[0]), int(ix[-1])
        for nm in rf:
            a = S.sim_period(O, C, sw[nm], j0, j1, 0.0005)
            b = R.ledger_sim(O, C, rf[nm], j0, j1, 0.0005)
            rows.append(dict(sym=s, per=per, det=nm, state_mis=int((sw[nm][: j1 + 1] != rf[nm][: j1 + 1]).sum()),
                             d_ret=abs(a[0] - b[0]), d_mdd=abs(a[1] - b[1]), d_hold=abs(a[2] - b[2]), d_ent=abs(a[3] - b[3])))
rc = pd.DataFrame(rows)
rc.to_csv(ROOT / "results" / "audit_extra_recompute144.csv", index=False)
print(f"(가) 전체 재계산 {len(rc)}건: 상태 불일치 날 {int(rc.state_mis.sum())}, 최대 |Δ| 수익 {rc.d_ret.max():.1e} 낙폭 {rc.d_mdd.max():.1e} "
      f"보유 {rc.d_hold.max():.1e} 진입 {rc.d_ent.max():.0f}")
print("   불일치 있는 판정:", rc[rc.state_mis > 0].groupby("det").size().to_dict())

# (나)
sh = []
for s, g in G.items():
    st_all = S.states(g)
    for per in ("P1", "P2"):
        if (per, s) not in OK:
            continue
        ix = S.period_index(g, per); j0, j1 = int(ix[0]), int(ix[-1]); D = j1 - j0 + 1
        for nm in (S.DEFAULT, S.DEFAULT2, "Z5", "ZZP_th0.1"):
            st = st_all[nm]
            h0 = st[j0 - 1:j1].mean()
            hs = [S.shift_state(st, j0, j1, k)[j0 - 1:j1].mean() for k in range(15, D - 14)]
            sh.append(dict(sym=s, per=per, det=nm, D=D, hold=h0, max_abs_dhold=float(np.max(np.abs(np.array(hs) - h0))),
                           frac_k_changed=float(np.mean(np.abs(np.array(hs) - h0) > 1e-12))))
sh = pd.DataFrame(sh)
sh.to_csv(ROOT / "results" / "audit_extra_shift.csv", index=False)
print(f"(나) 귀무 원형 이동: 종목·구간·판정 {len(sh)}건, 평균 보유가 바뀌는 k 비율 중앙 {sh.frac_k_changed.median():.2f}, "
      f"최대 |Δ보유| {sh.max_abs_dhold.max():.4f}(= 1/D 수준, D {sh.D.min()}~{sh.D.max()})")

# (다)(라)
tl = []
for s, g in G.items():
    C = g.C.to_numpy(float); H = g.H.to_numpy(float); L = g.L.to_numpy(float)
    a20 = S.atr(H, L, C, 20); d = np.array(g["date"].tolist())
    for nm, (th, av) in {**{f"P{t}": (t, np.zeros(0)) for t in (0.10, 0.15, 0.20)},
                         **{f"A{k}": (0.0, k * a20) for k in (3.0, 4.0, 5.0)}}.items():
        st, pi, pc, pt = S.zigzag(C, th, av)
        for per in PERS:
            if (per, s) not in OK:
                continue
            a, b = S.PERIODS[per]
            for k in range(len(pi) - 1):
                if not (a <= d[pi[k]] <= b):
                    continue
                seg = C[pi[k]:pi[k + 1] + 1]
                lo, hi = (C[pi[k]], C[pi[k + 1]]) if pt[k] == 1 else (C[pi[k + 1]], C[pi[k]])
                over = max(0.0, 1 - seg.min() / lo, seg.max() / hi - 1)
                tl.append(dict(sym=s, cfg=nm, per=per, over=over))
            if per == "P2":
                tl.append(dict(sym=s, cfg=nm, per="open_tail", over=float(len(C) - 1 - pc[-1]) if len(pc) else np.nan))
tl = pd.DataFrame(tl)
tl.to_csv(ROOT / "results" / "audit_extra_truth_legs.csv", index=False)
legs = tl[tl.per != "open_tail"]
t = legs.groupby(["cfg", "per"]).over.agg(legs="size", bad=lambda x: int((x > 0).sum()), bad_gt1pct=lambda x: int((x > 0.01).sum()),
                                           max="max")
t["bad_pct"] = (t.bad / t.legs * 100).round(1)
print("(다) 사후 진실 구간 중 끝점이 구간 극값이 아닌 것:")
print(t.round(4).to_string())
ot = tl[tl.per == "open_tail"]
print("(라) 자료 끝 10-22에 마지막 확정 뒤 지난 거래일(열린 구간) 중앙·90분위:")
print(ot.groupby("cfg").over.describe(percentiles=[0.5, 0.9])[["50%", "90%", "max"]].to_string())

# ARM Y24 Z5 (예열 75일, Pine은 129봉 전 na)
g = G["ARM"]; C = g.C.to_numpy(float); d = np.array(g["date"].tolist())
a = S.states(g)["Z5"]; b = R.z5_hold(R.z5_zone(C, pine=True))
m = (d >= S.PERIODS["Y24"][0]) & (d <= S.PERIODS["Y24"][1])
print(f"ARM Y24 Z5: swing vs Pine 보유 다른 날 {int((a[m] != b[m]).sum())}/{int(m.sum())}")
