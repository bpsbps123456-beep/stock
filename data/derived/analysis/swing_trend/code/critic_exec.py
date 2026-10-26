"""반박(실행·장세): ① 체결 시점(다음 날 시가 vs 판단 날 종가 근처) ② 신호 뒤 밤사이 갭 ③ 시장 공통 신호(SOXX·QQQ 상태를 모든 종목에 적용).
→ results/critic_exec_*.csv"""
from __future__ import annotations
import numpy as np, pandas as pd
import swing

OUT = swing.HERE / "results"
NAMES = [swing.dname(k, p) for k, p in swing.DETECTORS]
PERS = ("P1", "P2", "Y25", "Y24", "Y23")
WARM = 60


def sim_close(c, st, j0, j1, cost):
    """판단 날 종가 체결(주간거래·프리장 근사의 상한). 날 j 수익 = st[j-1] × (C[j]/C[j-1]-1), j0..j1."""
    v = 1.0; pk = 1.0; mdd = 0.0; hold = 0; prev = st[j0 - 2]
    for j in range(j0, j1 + 1):
        pos = st[j - 1]
        if pos != prev and j > j0:
            v *= 1 - cost
        v *= 1 + pos * (c[j] / c[j - 1] - 1)
        prev = pos; hold += pos
        pk = max(pk, v); mdd = min(mdd, v / pk - 1)
    n = j1 - j0 + 1
    x = hold / n
    px = c[j0 - 1:j1 + 1]
    eq = 1 + x * (px / px[0] - 1)
    return v - 1, mdd, x, float(eq[-1] - 1), float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    data = swing.load()
    ref = data["SPY"]
    D = {p: len(swing.period_index(ref, p)) for p in PERS}
    mkt = {m: swing.states(data[m]) for m in ("SOXX", "QQQ", "SPY")}
    mdates = {m: data[m].date.values for m in mkt}
    rows, gaps = [], []
    for s, g in data.items():
        o, c = g.O.values.astype(float), g.C.values.astype(float)
        sts = swing.states(g)
        dts = g.date.values
        for p in PERS:
            idx = swing.period_index(g, p)
            if len(idx) != D[p] or idx[0] < WARM or g.date.iloc[-1] < swing.PERIODS[p][1]:
                continue
            j0, j1 = int(idx[0]), int(idx[-1])
            for nm in NAMES:
                st = sts[nm]
                for cost in (0.0, 0.002):
                    r_o = swing.sim_period(o, c, st, j0, j1, cost)
                    sw_o = swing.static_bh(o, c, j0, j1, r_o[2])
                    r_c = sim_close(c, st, j0, j1, cost)
                    rows.append((p, s, nm, "own", cost, r_o[0] - sw_o[0], r_o[1] - sw_o[1], r_c[0] - r_c[3], r_c[1] - r_c[4], r_o[2]))
                # 신호 다음 밤 갭(C[d]→O[d+1]) : 매수·매도 신호별
                if True:
                    for d in range(j0 - 1, j1):
                        if st[d] != st[d - 1]:
                            gaps.append((p, s, nm, int(st[d]), np.log(o[d + 1] / c[d]), np.log(c[d + 1] / o[d + 1])))
            # 시장 공통 신호
            for m in mkt:
                al = pd.Series(np.arange(len(mdates[m])), index=mdates[m]).reindex(dts).values
                if np.isnan(al[j0 - 2:j1 + 1]).any():
                    continue
                al = al.astype(int)
                for nm in NAMES:
                    stm = np.zeros(len(c), np.int64)
                    ok = ~pd.isna(pd.Series(dts).map(pd.Series(np.arange(len(mdates[m])), index=mdates[m]))).values
                    stm[ok] = mkt[m][nm][al[ok]] if False else 0
                    stm = np.zeros(len(c), np.int64)
                    stm[j0 - 2:j1 + 1] = mkt[m][nm][al[j0 - 2:j1 + 1]]
                    r_o = swing.sim_period(o, c, stm, j0, j1, 0.0)
                    sw_o = swing.static_bh(o, c, j0, j1, r_o[2])
                    rows.append((p, s, nm, m, 0.0, r_o[0] - sw_o[0], r_o[1] - sw_o[1], np.nan, np.nan, r_o[2]))
    df = pd.DataFrame(rows, columns=["per", "sym", "det", "src", "cost", "ex_open", "dd_open", "ex_close", "dd_close", "x"])
    df.to_csv(OUT / "critic_exec_rows.csv", index=False)
    gp = pd.DataFrame(gaps, columns=["per", "sym", "det", "dir", "gap", "intraday"])
    gp.to_csv(OUT / "critic_exec_gaps.csv", index=False)


if __name__ == "__main__":
    main()
