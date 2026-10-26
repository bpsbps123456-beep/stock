"""BE·NBIS·MU 종목별 표(갈래 strat): 각 b의 원안, 그 b 안 고르기 1위, 전체 고르기 1위(b40 z320 ER A0), 기본값(b40 z160 AGREE A0).
실제 판(깨어 있음 + 편도 0.1%), 확인 구간과 전체(7/17~10/22). mtf_rows.parquet 값과 다시 계산한 값이 같은지도 확인한다.

py code/mtfstrat_syms.py → results/mtfstrat_syms.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import HERE
from zones3 import sim_frac, static_path, targets
from zones_mtf import BASE_TFS, prepare

RES = HERE / "results"
SYMS = ["BE", "NBIS", "MU"]
TOP1 = (40, "z320_ER", "A0")
DEFAULT = (40, "z160_AGREE", "A0")
ACT = {"A0": 0, "A2": 2, "A3": 3}


def one(P, clf, act, cost=0.001):
    O, C, seg = P["O"], P["C"], P["seg"]
    flag = P["flags"][clf] if clf != "BASE" else np.zeros(len(O), np.bool_)
    pos = np.zeros(len(O))
    out = sim_frac(O, C, targets(P["s"], flag, ACT[act]), seg, P["awake"], cost, pos)
    n0, n1 = out[4], out[5]
    x_all = (out[2] + out[3]) / (n0 + n1)
    st, st_mdd, bh, bh_mdd = static_path(O, C, seg, x_all, "all")
    x1 = out[3] / n1
    st1, st1_mdd, bh1, bh1_mdd = static_path(O, C, seg, x1, 1)
    return dict(ret_conf=np.expm1(out[1]), mdd_conf=np.expm1(out[7]), static_conf=st1, static_mdd_conf=st1_mdd, bh_conf=bh1,
                bh_mdd_conf=bh1_mdd, x_conf=x1, ret_sel=np.expm1(out[0]),
                ret_all=np.expm1(out[0] + out[1]), mdd_all=np.expm1(out[8]), static_all=st, static_mdd_all=st_mdd,
                bh_all=bh, bh_mdd_all=bh_mdd, x_all=x_all, turn_all=out[9] + out[10])


def main():
    cells = pd.read_csv(RES / "mtfstrat_cells.csv")
    cells = cells[cells.board == "real"]
    best_b = {b: x.loc[x.ds.idxmax()] for b, x in cells.groupby("b")}
    ref = pd.read_parquet(RES / "mtf_rows.parquet", filters=[("k", "==", 0), ("board", "==", "real"), ("sym", "in", SYMS)])
    for c in ("clf", "action", "sym"):
        ref[c] = ref[c].astype(str)
    ref = ref.set_index(["sym", "b", "clf", "action"])
    rows = []
    maxdiff = 0.0
    for sym in SYMS:
        for b in BASE_TFS:
            P = prepare(sym, b)
            picks = [("원안", "BASE", "A3"), ("b안 1위", best_b[b].clf, best_b[b].action)]
            if b == 40:
                picks += [("전체 1위", TOP1[1], TOP1[2]), ("기본값", DEFAULT[1], DEFAULT[2])]
            for nm, clf, act in picks:
                r = one(P, clf, act)
                q = ref.loc[(sym, b, clf, act)]
                maxdiff = max(maxdiff, abs(r["ret_conf"] - q.ret_conf), abs(r["ret_sel"] - q.ret_sel), abs(r["mdd_conf"] - q.mdd_conf))
                rows.append(dict(sym=sym, b=b, pick=nm, strategy=f"b{b} {clf} {act}", **r))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "mtfstrat_syms.csv", index=False)
    print("parquet 대조 최대 차이", f"{maxdiff:.2e}")
    p = lambda v: f"{100 * v:+.1f}"
    for sym in SYMS:
        print(f"\n== {sym} (실제 판) 확인: 수익 / 낙폭 / 같은 비중 수익 / 같은 비중 낙폭 | 전체: 수익 / 낙폭 / 같은 비중 / 같은 비중 낙폭 / 보유")
        for _, r in df[df.sym == sym].iterrows():
            print(f" {r.strategy:22s} {r.pick:6s} 확인 {p(r.ret_conf)} {p(r.mdd_conf)} {p(r.static_conf)} {p(r.static_mdd_conf)} (보유 {p(r.bh_conf)})"
                  f" | 전체 {p(r.ret_all)} {p(r.mdd_all)} {p(r.static_all)} {p(r.static_mdd_all)} (보유 {p(r.bh_all)} {p(r.bh_mdd_all)}) x {r.x_all:.2f} 회전 {r.turn_all:.0f}")


if __name__ == "__main__":
    main()
