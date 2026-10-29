"""사용자 실시간 예시(NOW 2026-10-29 09:37~09:39 ET)를 현재 Pine 직역 기본값으로 다시 본다. cache/now_live.parquet(10/26 04:00~10/29 09:43).
사용: py now_live_check.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pine_v3 as P  # noqa: E402

d = pd.read_parquet(HERE.parents[1] / "cache" / "now_live.parquet")
ny = d.index.tz_convert("America/New_York")
t = d.index.tz_convert("UTC").as_unit("ms").asi8.astype(np.int64)
day = (ny.year * 10000 + ny.month * 100 + ny.day).to_numpy(np.int64)
hk = d.index.tz_convert("Asia/Seoul").hour.to_numpy(np.int64)
args = (d.Open.to_numpy(float), d.High.to_numpy(float), d.Low.to_numpy(float), d.Close.to_numpy(float),
        d.Volume.to_numpy(float), t, day, ny.hour.to_numpy(np.int64), hk)
ts = ny.strftime("%m-%d %H:%M")
for brk_name, brk in (("신저가(기본)", P.B_NEW), ("처음 이탈", P.B_FIRST)):
    for dd in (False, True):
        prm = dict(P.DEFAULTS); prm["brkMd"] = brk; prm["dedup30"] = dd
        kA, kB, S, nT, nL, b = P.run(*args, *[prm[k] for k in P.ORDER])
        sel = lambda x: [ts[i] for i in np.flatnonzero(x) if ts[i] >= "10-29 09:30"]
        print(f"{brk_name} dedup30={dd}: ① {sel(kA)} ② {sel(kB)}")
i38 = list(ts).index("10-29 09:38")
prm = dict(P.DEFAULTS); prm["dedup30"] = False
kA, kB, S, nT, nL, b = P.run(*args, *[prm[k] for k in P.ORDER])
for i in range(i38 - 2, i38 + 2):
    r = d.iloc[i]
    print(ts[i], f"O {r.Open} H {r.High} L {r.Low} C {r.Close}", f"S {S[i]} brk {b[i]} touch {nT[i]} level {nL[i]}")
