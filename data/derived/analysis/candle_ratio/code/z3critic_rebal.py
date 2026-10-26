"""z3critic: sim_frac의 절반 보유(0.5) 회계 점검.
sim_frac은 봉마다 log(1+pos*r)라 pos=0.5는 매 봉 50%로 다시 맞춘(재조정) 포트폴리오다(잘 때도, 비용 없이).
같은 비중 static_path는 주식 수 고정(비중 표류). 둘의 차이와 재조정 비용 추정, 그리고 FLIP_K80_m2 A1이 그 차이와 같은지 본다.
"""
from __future__ import annotations
import sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from engine import CACHE, HERE
from zones3 import prepare, sim_frac, static_path, targets

SYMS = sorted(p.stem[:-3] for p in CACHE.glob("*_5m.parquet"))

def work(sym):
    P = prepare(sym, "5m", "dst")
    O, C, seg, aw = P["O"], P["C"], P["seg"], P["awake"]
    n = len(O); po = np.zeros(n)
    out = {}
    for w, k in (("sel", 0), ("conf", 1)):
        pass
    cm = sim_frac(O, C, np.full(n, 0.5), seg, aw, 0.001, po)
    fl = P["flags"]["FLIP_K80_m2"]
    a1 = sim_frac(O, C, targets(P["s"], fl, 1), seg, aw, 0.001, po.copy())
    pos_a1 = po.copy()
    row = dict(sym=sym, trend_share_conf=float(fl[seg == 1].mean()))
    nxt = np.r_[O[1:], C[-1]]; r = nxt / O - 1
    for w, k in (("sel", 0), ("conf", 1)):
        st, st_mdd, _, _ = static_path(O, C, seg, 0.5, k)
        m = seg == k
        # 매 봉 0.5로 되돌리는 재조정 거래량: 봉 끝 비중 = 0.5(1+r)/(1+0.5r) → |그 값 - 0.5|
        drift = np.abs(0.5 * (1 + r[m]) / (1 + 0.5 * r[m]) - 0.5)
        drift_aw = drift[aw[m]]  # 깨어 있는 봉만 재조정할 수 있다면
        row.update({f"cm_{w}": float(np.expm1(cm[k])), f"st_{w}": st, f"cm_mdd_{w}": float(np.expm1(cm[6 + k])), f"st_mdd_{w}": st_mdd,
                    f"a1_{w}": float(np.expm1(a1[k])), f"x_a1_{w}": a1[2 + k] / a1[4 + k],
                    f"rebal_cost_{w}": float(drift.sum() * 0.001), f"rebal_cost_aw_{w}": float(drift_aw.sum() * 0.001),
                    f"asleep_share_{w}": float(1 - aw[m].mean())})
    return row

if __name__ == "__main__":
    with Pool(3) as pool:
        rows = pool.map(work, SYMS, chunksize=6)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / "z3critic_rebal.csv", index=False)
    for w in ("sel", "conf"):
        d = df[f"cm_{w}"] - df[f"st_{w}"]
        dm = df[f"cm_mdd_{w}"] - df[f"st_mdd_{w}"]
        a = df[f"a1_{w}"] - df[f"st_{w}"]
        print(w, "중앙 cm %.2f st %.2f 차 %.2f%%p (평균 %.2f, cm>st %.0f%%) | 낙폭 cm %.2f st %.2f 차 %.2f | A1(FLIP80m2) %.2f, A1-st %.2f, A1-cm %.2f | 재조정비용 중앙 %.2f%% (깨어만 %.2f%%) | 잠 비율 %.2f" % (
            df[f"cm_{w}"].median()*100, df[f"st_{w}"].median()*100, d.median()*100, d.mean()*100, (d > 0).mean()*100,
            df[f"cm_mdd_{w}"].median()*100, df[f"st_mdd_{w}"].median()*100, dm.median()*100,
            df[f"a1_{w}"].median()*100, a.median()*100, (df[f"a1_{w}"]-df[f"cm_{w}"]).median()*100,
            df[f"rebal_cost_{w}"].median()*100, df[f"rebal_cost_aw_{w}"].median()*100, df[f"asleep_share_{w}"].median()))
    print("FLIP80m2 추세 비율 conf 중앙", df.trend_share_conf.median(), "x_a1_conf 중앙", df.x_a1_conf.median())
