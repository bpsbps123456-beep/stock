"""표시 거르기 점검(성과 없음): Pine dedup30(모든 시간 신호로 30분 겹침) 대 백테스트(판정 모집단으로 먼저 거른 뒤 겹침, evaluate.cell_rows 순서).
판정 모집단은 BRIEF 정의 04:00~10:59 와 19:00~19:29 ET 로 근사한다(엔진의 X.sane 안 유효 봉 조건은 여기서 재지 않는다).
① Pine 직역 dedup30=True 가 pandas 탐욕 겹침과 같은지, ② 그 Pine 표시를 판정 모집단으로 자른 집합이 백테스트 순서와 몇 건 다른지 센다.
사용: py dedup_check.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parity_v3 as PV  # noqa: E402
import pine_v3 as P  # noqa: E402


def greedy(df):
    keep, last = [], {}
    for r in df.itertuples():
        k = r.sym
        if k not in last or r.tmin - last[k] >= 30:
            keep.append(True); last[k] = r.tmin
        else:
            keep.append(False)
    return df[np.array(keep, bool)] if len(df) else df


base = pd.read_csv(HERE / "pine_signals.csv")
base["tmin"] = (pd.to_datetime(base.ts_et) - pd.Timestamp("2026-01-01")) // pd.Timedelta(minutes=1)
hm = base.ts_et.str[11:]
base["pop"] = ((hm >= "04:00") & (hm < "11:00")) | ((hm >= "19:00") & (hm < "19:30"))
base["kst_ok"] = (pd.to_datetime(base.ts_et).dt.tz_localize("America/New_York").dt.tz_convert("Asia/Seoul").dt.hour >= 8)

# Pine 직역 dedup30=True 실행
rows = []
for sym in PV.universe():
    o, h, l, c, v, t, day, hr, hk, ts = PV.load(sym)
    for cell, ov in PV.CELLS.items():
        prm = dict(P.DEFAULTS); prm.update(ov); prm["dedup30"] = True
        kA, kB, *_ = P.run(o, h, l, c, v, t, day, hr, hk, *[prm[k] for k in P.ORDER])
        sig = kA if PV.PAT[cell] == "A" else kB
        rows += [(cell, sym, ts[i]) for i in np.flatnonzero(sig)]
pine_dd = pd.DataFrame(rows, columns=["cell", "sym", "ts_et"])

key = ["cell", "sym", "ts_et"]
out = []
for cell, g in base.sort_values(["cell", "sym", "tmin"]).groupby("cell", sort=False):
    pd_all = greedy(g)                                   # Pine 기본(onlyTrd 끔) 표시
    same = len(pd_all.merge(pine_dd[pine_dd.cell == cell], on=key, how="outer", indicator=True).query("_merge != 'both'"))
    bt = greedy(g[g["pop"]])                             # 백테스트: 모집단 먼저, 겹침 나중
    trd = greedy(g[g.kst_ok])                            # Pine onlyTrd 켬
    a = set(map(tuple, pd_all[pd_all["pop"]][key].values)); b = set(map(tuple, bt[key].values))
    c = set(map(tuple, trd[trd["pop"]][key].values))
    out.append(dict(cell=cell, pine_dedup_vs_greedy_diff=same, backtest_pop_dedup=len(b),
                    pine_default_in_pop=len(a), hidden_by_pine_default=len(b - a), extra_in_pine_default=len(a - b),
                    pine_onlyTrd_in_pop=len(c), hidden_by_pine_onlyTrd=len(b - c), extra_in_pine_onlyTrd=len(c - b),
                    hidden_pre_default=sum(1 for x in b - a if x[2][11:] < "09:30")))
r = pd.DataFrame(out)
r.to_csv(HERE / "dedup_check.csv", index=False)
print(r.to_string(index=False))
