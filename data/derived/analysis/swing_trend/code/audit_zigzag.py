"""audit 3: 지그재그 사후 반전점 점검(swing.zigzag, 144종목 전 기간 전수 + 무작위 30구간 표본 출력).
점검: ① piv_idx < piv_conf ≤ 다음 piv_idx, 종류 교대, 상태가 확정 날에만 바뀜
      ② 확정 날이 문턱을 처음 넘은 날(그 전에는 안 넘음), 확정 날 종가가 문턱 이상
      ③ 반전점 종가가 직전 확정 날~이번 확정 날의 최저(저점)/최고(고점)
      ④ 구간(반전점→다음 반전점) 안 종가가 양 끝 사이에 있는지(상승 구간에 시작 저점보다 낮은 종가, 하락 구간에 시작 고점보다 높은 종가)
      ⑤ 구간 크기 ≥ 문턱, ⑥ 첫 반전점이 자료 시작에 붙은 가짜인지, 평가 구간 안에 떨어지는지
      ⑦ 독립 재계산(audit_ref.zz_state) 반전점과 같은지
→ results/audit_zigzag_checks.csv, results/audit_zigzag_sample.csv, results/audit_zigzag_leg_violations.csv"""
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
CFG = [("P", t) for t in (0.03, 0.05, 0.07, 0.10, 0.15, 0.20)] + [("A", k) for k in (1.5, 2.0, 3.0, 4.0, 5.0)]
EVAL0 = date(2025, 1, 2)

chk, legs_bad, legs_all = [], [], []
for s, g in G.items():
    C = g.C.to_numpy(float); H = g.H.to_numpy(float); L = g.L.to_numpy(float)
    a20 = S.atr(H, L, C, 20)
    dts = g["date"].tolist()
    for kind, v in CFG:
        if kind == "P":
            st, pi, pc, pt = S.zigzag(C, v, np.zeros(0))
            thr = lambda i, e, up: e * (1 - v) if up else e * (1 + v)
            _, piv_ref = R.zz_state(list(C), thr)
            thrval = lambda i, e: v * e
        else:
            av = v * a20
            st, pi, pc, pt = S.zigzag(C, 0.0, av)
            _, piv_ref = R.zza(H, L, C, v)
            thrval = lambda i, e, av=av: av[i]
        n = len(pi)
        e = dict(sym=s, cfg=f"{kind}{v}", npiv=n)
        e["ref_same"] = int(list(zip(pi, pc, pt)) == [tuple(x) for x in piv_ref])
        e["idx_lt_conf"] = int(np.all(pi < pc))
        e["conf_le_next"] = int(np.all(pc[:-1] <= pi[1:])) if n > 1 else 1
        e["alternate"] = int(np.all(pt[1:] != pt[:-1])) if n > 1 else 1
        flips = np.flatnonzero(np.diff(st) != 0) + 1
        e["flip_eq_conf"] = int(np.array_equal(flips, pc))
        first_cross_bad = thr_bad = ext_bad = 0
        for k in range(n):
            a, f, ty = pi[k], pc[k], pt[k]
            p0 = pc[k - 1] if k > 0 else 0
            seg = C[p0:f + 1]
            if ty == 1:
                ext_bad += int(C[a] != seg.min())
                thr_bad += int(not (C[f] >= C[a] + thrval(f, C[a]) - 1e-12))
                rm = np.minimum.accumulate(C[p0:f])
                for t in range(p0, f):
                    tv = thrval(t, rm[t - p0])
                    if not np.isnan(tv) and C[t] >= rm[t - p0] + tv:
                        first_cross_bad += 1; break
            else:
                ext_bad += int(C[a] != seg.max())
                thr_bad += int(not (C[f] <= C[a] - thrval(f, C[a]) + 1e-12))
                rm = np.maximum.accumulate(C[p0:f])
                for t in range(p0, f):
                    tv = thrval(t, rm[t - p0])
                    if not np.isnan(tv) and C[t] <= rm[t - p0] - tv:
                        first_cross_bad += 1; break
        e["thr_bad"] = thr_bad; e["first_cross_bad"] = first_cross_bad; e["ext_bad"] = ext_bad
        # ④⑤ 구간 성질
        leg_bad = amp_bad = 0
        for k in range(n - 1):
            a, b = pi[k], pi[k + 1]
            seg = C[a:b + 1]
            ty = pt[k]  # +1: 저점→고점(상승 구간)
            amp = C[b] / C[a] - 1
            if ty == 1:  # 상승 구간: 시작 저점 아래 / 끝 고점 위로 벗어난 크기
                o_start = max(0.0, 1 - seg.min() / C[a]); o_end = max(0.0, seg.max() / C[b] - 1)
                amp_ok = (C[b] - C[a]) >= thrval(pc[k], C[a]) - 1e-12
                wi = int(np.argmin(seg)) if o_start >= o_end else int(np.argmax(seg))
            else:  # 하락 구간: 시작 고점 위 / 끝 저점 아래
                o_start = max(0.0, seg.max() / C[a] - 1); o_end = max(0.0, 1 - seg.min() / C[b])
                amp_ok = (C[a] - C[b]) >= thrval(pc[k], C[a]) - 1e-12
                wi = int(np.argmax(seg)) if o_start >= o_end else int(np.argmin(seg))
            inner_bad = (o_start > 0) or (o_end > 0)
            worst = max(o_start, o_end)
            amp_bad += int(not amp_ok)
            rec = dict(sym=s, cfg=f"{kind}{v}", leg="up" if ty == 1 else "down", start=dts[a], end=dts[b],
                       days=int(b - a), amp=amp, start_conf=dts[pc[k]], end_conf=dts[pc[k + 1]],
                       inner_bad=int(inner_bad), over_start=o_start, over_end=o_end, worst=worst,
                       worst_day=dts[a + wi])
            legs_all.append(rec)
            if inner_bad:
                leg_bad += 1
                legs_bad.append(rec)
        e["legs"] = max(n - 1, 0); e["leg_inner_bad"] = leg_bad; e["amp_bad"] = amp_bad
        e["first_piv_idx0"] = int(n > 0 and pi[0] == 0)
        e["first_piv_date"] = dts[pi[0]] if n else None
        e["first_piv_in_eval"] = int(n > 0 and dts[pi[0]] >= EVAL0)
        e["last_conf"] = dts[pc[-1]] if n else None
        chk.append(e)

ck = pd.DataFrame(chk)
ck.to_csv(ROOT / "results" / "audit_zigzag_checks.csv", index=False)
lb = pd.DataFrame(legs_bad)
lb.to_csv(ROOT / "results" / "audit_zigzag_leg_violations.csv", index=False)
la = pd.DataFrame(legs_all)
random.seed(7)
smp = la[la.start >= EVAL0].sample(30, random_state=7).sort_values(["sym", "cfg", "start"])
smp.to_csv(ROOT / "results" / "audit_zigzag_sample.csv", index=False)

print(f"종목 {ck.sym.nunique()} × 설정 {len(CFG)} = {len(ck)}개, 반전점 {int(ck.npiv.sum())}개, 구간 {int(ck.legs.sum())}개")
for c in ("ref_same", "idx_lt_conf", "conf_le_next", "alternate", "flip_eq_conf"):
    print(f"  {c}: 통과 {int(ck[c].sum())}/{len(ck)}")
for c in ("thr_bad", "first_cross_bad", "ext_bad", "amp_bad"):
    print(f"  {c}: {int(ck[c].sum())}")
print("  첫 반전점이 자료 첫날(가짜 시작점):", int(ck.first_piv_idx0.sum()), " 첫 반전점이 2025년 이후(평가 구간 안):", int(ck.first_piv_in_eval.sum()))
print(ck[ck.first_piv_in_eval == 1].groupby("sym").first_piv_date.min().to_string())
la26 = la[la.start >= EVAL0]
lb26 = lb[lb.start >= EVAL0] if len(lb) else lb
print("\n④ 구간 안 종가가 양 끝을 벗어난 구간(2025-01 이후 시작):")
tab = la26.groupby(["cfg", "leg"]).inner_bad.agg(["sum", "count"])
tab["pct"] = (tab["sum"] / tab["count"] * 100).round(1)
print(tab.to_string())
if len(lb26):
    print("벗어난 크기(끝점 대비 비율) 분포 — 시작 쪽/끝 쪽:")
    print(lb26.groupby("cfg").agg(n=("worst", "size"), start_side=("over_start", lambda x: int((x > 0).sum())),
                                  end_side=("over_end", lambda x: int((x > 0).sum())), med=("worst", "median"),
                                  mx=("worst", "max")).round(4).to_string())

# MU 15% 2026 반전점
g = G["MU"]; C = g.C.to_numpy(float); dts = g["date"].tolist()
st, pi, pc, pt = S.zigzag(C, 0.15, np.zeros(0))
print("\nMU 15% 2026 반전점(극점 날 / 확정 날 / 종류):")
for a, f, ty in zip(pi, pc, pt):
    if dts[f] >= date(2026, 1, 1):
        print(f"  {dts[a]} {'저점' if ty == 1 else '고점'} {C[a]:.2f} → 확정 {dts[f]} ({f - a}거래일 뒤, {C[f] / C[a] - 1:+.1%})")
print(f"  마지막 확정 뒤 미확정 극점: {dts[pc[-1]]} 이후 {'최저' if pt[-1] == -1 else '최고'} "
      f"{(C[pc[-1]:].min() if pt[-1] == -1 else C[pc[-1]:].max()):.2f}")
print("\n무작위 표본 30구간(2025-01 이후):")
print(smp.to_string(index=False))
