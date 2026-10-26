"""반박: 뒤집은 판정의 원형 이동 귀무(trade_run.null_kernel 재사용, 모든 종목 같은 k)."""
import numpy as np, pandas as pd
import swing, trade_run
data = swing.load(); ref = data["SPY"]
DETS = ["ZZA_k3.0", "EMA_f20_s50", "CR_n10_b0.5_s0.5", "ZZP_th0.15", "DON_n20_m10"]
for p in ("P1", "P2", "Y25"):
    D = len(swing.period_index(ref, p)); ks = np.arange(15, D - 15 + 1, dtype=np.int64)
    act = {d: [] for d in DETS}; nul = {d: [] for d in DETS}
    for s, g in data.items():
        idx = swing.period_index(g, p)
        if len(idx) != D or idx[0] < 60 or g.date.iloc[-1] < swing.PERIODS[p][1]:
            continue
        o, c = g.O.values.astype(float), g.C.values.astype(float)
        sts = swing.states(g); j0, j1 = int(idx[0]), int(idx[-1])
        for d in DETS:
            st = (1 - sts[d]).astype(np.int64)
            r = swing.sim_period(o, c, st, j0, j1, 0.0); b = swing.static_bh(o, c, j0, j1, r[2])
            act[d].append(r[0] - b[0])
            rr, dd, xs, en, sr, sd = trade_run.null_kernel(o, c, st, j0, j1, 0.0, ks)
            nul[d].append(rr - sr)
    for d in DETS:
        a = np.median(act[d]); nm = np.median(np.array(nul[d]), axis=0)
        print(p, d, "역 실제 %.1f%%p 귀무 중앙 %.1f 95분위 %.1f p %.3f" % (a * 100, np.median(nm) * 100, np.quantile(nm, .95) * 100, (nm >= a).mean()))
