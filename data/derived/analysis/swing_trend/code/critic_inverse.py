"""반박: 판정을 뒤집은 상태(역추세, 판정이 현금이면 보유)의 같은 비중 대비 차이. 장세(추세·되돌림) 의존 확인."""
import numpy as np, pandas as pd
import swing
NAMES = [swing.dname(k, p) for k, p in swing.DETECTORS]
data = swing.load(); ref = data["SPY"]
rows = []
for p in ("P1", "P2", "Y25", "Y24", "Y23"):
    D = len(swing.period_index(ref, p))
    for s, g in data.items():
        idx = swing.period_index(g, p)
        if len(idx) != D or idx[0] < 60 or g.date.iloc[-1] < swing.PERIODS[p][1]:
            continue
        o, c = g.O.values.astype(float), g.C.values.astype(float)
        sts = swing.states(g); j0, j1 = int(idx[0]), int(idx[-1])
        for nm in NAMES:
            st = 1 - sts[nm]
            r = swing.sim_period(o, c, st, j0, j1, 0.0); b = swing.static_bh(o, c, j0, j1, r[2])
            rows.append((p, s, nm, r[0] - b[0], r[1] - b[1]))
df = pd.DataFrame(rows, columns=["per", "sym", "det", "ex", "dd"])
df.to_csv(swing.HERE / "results" / "critic_inverse_rows.csv", index=False)
m = df.groupby(["per", "det"]).ex.median().unstack(0) * 100
print(m.round(1).to_string())
print("이긴 판정 수", (m > 0).sum().to_dict(), "28개 중앙", m.median().round(1).to_dict())
