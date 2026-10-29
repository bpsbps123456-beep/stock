"""감사 보강: (1) 거울(반대 ETF) 대조로 찾은 한 종목만 튄 봉을 모두 정리한 뒤 32칸 다시 계산
(2) 시각 맞춤 B0: 반대 매수 각 매매와 같은 진입 분의 무작위 도달률·손익(신호가 09:30에 몰린 효과 분리)
(3) 거울 확인 도달(자기 고가와 반대 ETF 저가로 만든 고가가 모두 목표 관통) 민감도
impl_a.py는 고치지 않고 함수만 빌린다. 사용: py code/audit_extra.py → results/audit_extra.json
"""
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import impl_a as A  # noqa: E402

OUT = os.path.join(A.ROOT, "results", "audit_extra.json")
T = A.TICKERS
TOL = 0.006   # 거울 불일치 문턱 0.6%


def r6(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 6)


data = {t: A.load(t) for t in T}
days = sorted(set(data["SOXL"].index[(A.minute_of_day(data["SOXL"].index) >= 570)
                                     & (A.minute_of_day(data["SOXL"].index) <= 959)].date))
ND = len(days)
grids = {t: A.regular_grid(data[t], days) for t in T}

# ---- (1) 거울 불일치 봉 전수(정규장 전체) → 고가·저가를 거울 값으로 깎는다
fix = []
clean = copy.deepcopy(data)
for d in range(ND):
    for t in range(1, A.NBAR):
        for Y in T:
            Z = A.OTHER[Y]
            gy, gz = grids[Y], grids[Z]
            imp_hi = gy["C"][d, t - 1] * (1 - (gz["L"][d, t] / gz["C"][d, t - 1] - 1))
            imp_lo = gy["C"][d, t - 1] * (1 - (gz["H"][d, t] / gz["C"][d, t - 1] - 1))
            ts = pd.Timestamp(f"{days[d]} {A.hhmm(t)}", tz="America/New_York")
            o, c = gy["O"][d, t], gy["C"][d, t]
            if gy["H"][d, t] / imp_hi - 1 > TOL and gy["H"][d, t] > max(o, c) + 1e-9:
                new = max(o, c, imp_hi)
                fix.append({"ts": str(ts), "sym": Y, "field": "High", "old": r6(gy["H"][d, t]), "new": r6(new)})
                clean[Y].loc[ts, "High"] = new
            if 1 - gy["L"][d, t] / imp_lo > TOL and gy["L"][d, t] < min(o, c) - 1e-9:
                new = min(o, c, imp_lo)
                fix.append({"ts": str(ts), "sym": Y, "field": "Low", "old": r6(gy["L"][d, t]), "new": r6(new)})
                clean[Y].loc[ts, "Low"] = new

cg = {t: A.regular_grid(clean[t], days) for t in T}
ctab = {t: A.trade_table(cg[t]) for t in T}
otab = {t: A.trade_table(grids[t]) for t in T}
changed = []
main_clean = None
for b_ in A.BASES:
    rs = {t: A.rsi_series(data[t], b_) for t in T}      # 종가는 안 바뀌므로 RSI·신호 그대로
    for w, nw in A.WINDOWS.items():
        sg = {t: A.signals_for(rs[t], days, nw) for t in T}
        for Tn in A.SIGNALS:
            for e_, k in A.FILLS.items():
                key = f"{Tn}|{b_}|{e_}|{w}"
                sbx = {t: sg[t][Tn] for t in T}
                for mode in ["fade", "follow"]:
                    new, _, _ = A.run_trades(sbx, ctab, cg, k, mode)
                    old, _, _ = A.run_trades(sbx, otab, grids, k, mode)
                    if key == A.MAIN and mode == "fade":
                        main_clean = new
                    if len(new) != len(old) or not np.allclose(new["pnl"].to_numpy(float), old["pnl"].to_numpy(float)):
                        changed.append({"cell": key, "mode": mode, "n_old": len(old), "n_new": len(new),
                                        "old_ev": r6(old["pnl"].mean()), "new_ev": r6(new["pnl"].mean()),
                                        "old_hit": r6(old["hit"].mean()), "new_hit": r6(new["hit"].mean())})
b0c = {}
for e_, k in A.FILLS.items():
    for w, nw in A.WINDOWS.items():
        o = A.b0_trades(otab, k, nw, ND)
        n = A.b0_trades(ctab, k, nw, ND)
        b0c[f"{e_}|{w}"] = {"old_hit": r6(o["hit"].mean()), "new_hit": r6(n["hit"].mean()),
                            "old_ev": r6(o["pnl"].mean()), "new_ev": r6(n["pnl"].mean()),
                            "n_changed": int((o["pnl"].to_numpy() != n["pnl"].to_numpy()).sum())}

# ---- (2) 시각 맞춤 B0 (주 칸)
k = 2
rsi = {t: A.rsi_series(data[t], "R") for t in T}
sig = {t: A.signals_for(rsi[t], days, 60)["T1"] for t in T}
fade, _, _ = A.run_trades(sig, otab, grids, k, "fade")
foll, _, _ = A.run_trades(sig, otab, grids, k, "follow")
ne = 60 + k
Hd = np.zeros((ND, ne))
Pd = np.zeros((ND, ne))
for t in T:
    Hd += otab[t]["hit"][:, :ne]
    Pd += otab[t]["pnl"][:, :ne]
b0_hit_by_e = Hd.sum(0) / (2 * ND)
b0_pnl_by_e = Pd.sum(0) / (2 * ND)
same_y_hit = {t: otab[t]["hit"][:, :ne].mean(0) for t in T}
same_y_pnl = {t: otab[t]["pnl"][:, :ne].mean(0) for t in T}
fe = fade["e"].to_numpy(int)
matched = {
    "fade_hit": r6(fade["hit"].mean()), "fade_ev": r6(fade["pnl"].mean()),
    "b0_unmatched_hit": r6(b0_hit_by_e[k:].mean()), "b0_unmatched_ev": r6(b0_pnl_by_e[k:].mean()),
    "b0_time_matched_hit": r6(b0_hit_by_e[fe].mean()), "b0_time_matched_ev": r6(b0_pnl_by_e[fe].mean()),
    "b0_time_and_ticker_matched_hit": r6(np.mean([same_y_hit[y][e] for y, e in zip(fade["Y"], fe)])),
    "b0_time_and_ticker_matched_ev": r6(np.mean([same_y_pnl[y][e] for y, e in zip(fade["Y"], fe)])),
    "b0_hit_at_0932_entry": r6(b0_hit_by_e[2]), "b0_ev_at_0932_entry": r6(b0_pnl_by_e[2]),
    "share_fade_entries_at_0932": r6((fe == 2).mean()),
}
# 시각 맞춤 차이의 날 묶음 부트스트랩(B0 분별 도달률도 같은 뽑기로 다시 잼)
rng = np.random.default_rng(A.SEED)
idx = rng.integers(0, ND, size=(A.NBOOT, ND))
cnt = np.stack([np.bincount(r, minlength=ND) for r in idx]).astype(float)       # (B, ND)
b0rate = (cnt @ Hd) / (2 * cnt.sum(1, keepdims=True))                              # (B, ne)
b0ev = (cnt @ Pd) / (2 * cnt.sum(1, keepdims=True))
fd = fade["d"].to_numpy(int)
w = cnt[:, fd]                                                                     # (B, ntr)
fh = fade["hit"].to_numpy(float)
fp = fade["pnl"].to_numpy(float)
diff_hit = ((w * fh).sum(1) - (w * b0rate[:, fe]).sum(1)) / w.sum(1)
diff_ev = ((w * fp).sum(1) - (w * b0ev[:, fe]).sum(1)) / w.sum(1)
matched["fade_minus_matched_b0_hit"] = r6(fh.mean() - b0_hit_by_e[fe].mean())
matched["fade_minus_matched_b0_hit_ci"] = [r6(x) for x in np.nanpercentile(diff_hit, [2.5, 97.5])]
matched["fade_minus_matched_b0_ev"] = r6(fp.mean() - b0_pnl_by_e[fe].mean())
matched["fade_minus_matched_b0_ev_ci"] = [r6(x) for x in np.nanpercentile(diff_ev, [2.5, 97.5])]
# 09:30 신호 / 그 뒤 신호로 나눈 시각 맞춤
for nm, mk in [("s_0930", fade["s"].to_numpy() == 0), ("s_after_0930", fade["s"].to_numpy() > 0)]:
    matched[nm] = {"n": int(mk.sum()), "fade_hit": r6(fh[mk].mean()), "fade_ev": r6(fp[mk].mean()),
                   "matched_b0_hit": r6(b0_hit_by_e[fe[mk]].mean()), "matched_b0_ev": r6(b0_pnl_by_e[fe[mk]].mean()),
                   "follow_note": "추종은 별도 표"}

# ---- (3) 거울 확인 도달: 자기 고가 ≥ 관통 문턱 AND 거울 고가 ≥ 관통 문턱 × (1 − 0.1%)
def mirror_table(Y, tol=0.001, hold=30):
    Z = A.OTHER[Y]
    gy, gz = grids[Y], grids[Z]
    prevY = np.c_[np.full((ND, 1), np.nan), gy["C"][:, :-1]]
    prevZ = np.c_[np.full((ND, 1), np.nan), gz["C"][:, :-1]]
    imp_hi = prevY * (1 - (gz["L"] / prevZ - 1))
    Hc = np.minimum(gy["H"], np.where(np.isnan(imp_hi), gy["H"], imp_hi * (1 + tol)))
    g2 = dict(gy)
    g2["H"] = Hc
    return A.trade_table(g2, hold)


mtab = {t: mirror_table(t) for t in T}
mf, _, _ = A.run_trades(sig, mtab, grids, k, "fade")
mb = A.b0_trades(mtab, k, 60, ND)
mirror_variant = {"rule": "고가를 min(자기 고가, 반대 ETF 저가로 만든 고가 × 1.001)로 깎고 BRIEF 규칙 그대로",
                  "fade_n": len(mf), "fade_hit": r6(mf["hit"].mean()), "fade_ev": r6(mf["pnl"].mean()),
                  "b0_hit": r6(mb["hit"].mean()), "b0_ev": r6(mb["pnl"].mean())}

out = {"mirror_fixed_bars": fix, "n_fixed": len(fix), "cells_changed_after_fix": changed, "b0_after_fix": b0c,
       "main_after_fix": {"n": len(main_clean), "hit": r6(main_clean["hit"].mean()), "ev": r6(main_clean["pnl"].mean())},
       "time_matched_b0_main": matched, "mirror_confirmed_hit_variant_main": mirror_variant}
with open(OUT, "w", encoding="utf-8") as fh_:
    json.dump(out, fh_, ensure_ascii=False, indent=1, default=str)
print("fixed bars", len(fix))
for f in fix:
    print("  ", f)
print("changed cells", json.dumps(changed, ensure_ascii=False, indent=0))
print("b0 after fix", json.dumps(b0c, ensure_ascii=False))
print("main after fix", out["main_after_fix"])
print("matched", json.dumps(matched, ensure_ascii=False, indent=1))
print("mirror variant", mirror_variant)
