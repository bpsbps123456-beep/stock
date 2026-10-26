"""반박 검산: 미국 애프터마켓(16:00~20:00 ET, 서머타임 한국 05~09시) 체결을 막으면(체결 확인 안 된 시간, desk_system §1) 성적이 어떻게 바뀌나.

py code/mtfcritic_noafter.py → results/mtfcritic_noafter.csv
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd

from engine import HERE
from mtf_run import SYMS
from zones_mtf import BASE_TFS, prepare
from mtfcritic_exec import PICK, one

OUT = HERE / "results"


def work(sym):
    rows = []
    for b in BASE_TFS:
        P = prepare(sym, b)
        et = pd.DatetimeIndex(P["tb"].start).tz_convert("America/New_York")
        pre = np.asarray(et.hour < 16)
        for nm, fl, ai in [("BASE", None, 3)] + [(n, n, a) for n, a in PICK.get(b, [])]:
            flag = P["flags"][fl] if fl else np.zeros(len(P["C"]), np.bool_)
            for vn, aw, cost in (("base", P["awake"], 0.001), ("noafter", P["awake"] & pre, 0.001), ("noafter0", P["awake"] & pre, 0.0)):
                ret, st, mdd, st_mdd, turn, _ = one(P["O"], P["C"], P, flag, ai, aw, cost)
                rows.append(dict(sym=sym, b=b, clf=nm, action=ai, var=vn, d=(ret - st) * 100, dm=(mdd - st_mdd) * 100, turn=turn))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with Pool(3) as p:
        g = pd.concat(p.map(work, SYMS), ignore_index=True)
    t = g.groupby(["b", "clf", "action", "var"])[["d", "dm", "turn"]].median().unstack("var").round(2)
    t.to_csv(OUT / "mtfcritic_noafter.csv")
    pd.set_option("display.width", 200)
    print(t.to_string())
    tm = pd.read_csv(OUT / "mtfcritic_exec_times.csv")
    print((tm.et_min >= 960).groupby(tm.b).mean().round(3).to_string())
