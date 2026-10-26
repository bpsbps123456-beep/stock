"""반박 검산(실행 관점): 표준시 깨어 있는 시간표, 체결 지연(+5·+30분), 비용 0.2%, 체결 시각 분포.

py code/mtfcritic_exec.py [프로세스 수] → results/mtfcritic_exec.parquet, results/mtfcritic_exec_times.csv
고치지 않은 zones_mtf.prepare를 그대로 쓰고, O(체결가)와 awake만 바꿔 sim_frac을 다시 돌린다.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from engine import HERE, load
from zones3 import awake_std, sim_frac, static_path, targets
from zones_mtf import BASE_TFS, prepare
from mtf_run import SYMS

OUT = HERE / "results"
PICK = {40: [("z320_ER", 0), ("z160_AGREE", 0), ("z320_ER", 2)]}


def delayed_open(P, d5, delay):
    """평가 봉(5분봉 묶음)의 체결가를 봉 시작 + delay분 이후 첫 5분봉 시가로 바꾼다. 없으면 그 봉 종가."""
    tb = P["tb"]
    O = P["O"].copy()
    idx5 = d5.index.asi8 if hasattr(d5.index, "asi8") else pd.DatetimeIndex(d5.index).asi8
    o5 = d5.Open.values.astype(float)
    st = pd.DatetimeIndex(tb.start).as_unit("ns").asi8
    ev = np.flatnonzero(~tb.warm.values)
    b = P["b"]
    sess_end = pd.DatetimeIndex(pd.to_datetime(tb.day.values)).tz_localize("America/New_York").as_unit("ns").asi8 + 1200 * 60_000_000_000
    nom_end = np.minimum(st + b * 60_000_000_000, sess_end)
    t0 = st[ev] + delay * 60_000_000_000
    i = np.searchsorted(idx5, t0, side="left")
    ok = (i < len(idx5)) & (idx5[np.minimum(i, len(idx5) - 1)] < nom_end[ev])
    O[ev] = np.where(ok, o5[np.minimum(i, len(idx5) - 1)], P["C"][ev])
    return O


def one(O, C, P, flag, ai, aw, cost):
    pos = np.zeros(len(O))
    out = sim_frac(O, C, targets(P["s"], flag, ai), P["seg"], aw, cost, pos)
    x = out[3] / out[5]
    st, st_mdd, _, _ = static_path(O, C, P["seg"], x, 1)
    return float(np.expm1(out[1])), st, float(np.expm1(out[7])), st_mdd, out[10], pos


def work(sym):
    d5 = load(sym, "5m")
    d5.index = pd.DatetimeIndex(d5.index).as_unit("ns")
    rows, times = [], []
    for b in BASE_TFS:
        P = prepare(sym, b)
        C = P["C"]
        variants = {
            "base": (P["O"], P["awake"], 0.001),
            "std": (P["O"], awake_std(P["tb"].start), 0.001),
            "cost2": (P["O"], P["awake"], 0.002),
            "d5": (delayed_open(P, d5, 5), P["awake"], 0.001),
            "d30": (delayed_open(P, d5, 30), P["awake"], 0.001),
            "base0": (P["O"], P["awake"], 0.0),
            "std0": (P["O"], awake_std(P["tb"].start), 0.0),
            "d5_0": (delayed_open(P, d5, 5), P["awake"], 0.0),
        }
        strats = [("BASE", None, 3)] + [(nm, nm, ai) for nm, ai in PICK.get(b, [])]
        for nm, fl, ai in strats:
            flag = P["flags"][fl] if fl else np.zeros(len(C), np.bool_)
            for vn, (O, aw, cost) in variants.items():
                ret, st, mdd, st_mdd, turn, pos = one(O, C, P, flag, ai, aw, cost)
                rows.append(dict(sym=sym, b=b, clf=nm, action={0: "A0", 2: "A2", 3: "A3"}[ai], var=vn,
                                 ret=ret, static=st, mdd=mdd, static_mdd=st_mdd, turn=turn))
                if vn == "base" and nm == "BASE":
                    ch = np.flatnonzero((np.abs(np.diff(np.r_[0.0, pos])) > 0) & (P["seg"] == 1))
                    hh = pd.DatetimeIndex(P["tb"].start.iloc[ch]).tz_convert("America/New_York")
                    for h, m in zip(hh.hour, hh.minute):
                        times.append((sym, b, h * 60 + m))
    return pd.DataFrame(rows), pd.DataFrame(times, columns=["sym", "b", "et_min"])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    with Pool(n) as pool:
        res = pool.map(work, SYMS)
    g = pd.concat([r[0] for r in res], ignore_index=True)
    t = pd.concat([r[1] for r in res], ignore_index=True)
    g.to_parquet(OUT / "mtfcritic_exec.parquet")
    t.to_csv(OUT / "mtfcritic_exec_times.csv", index=False)
    g["d_static"] = (g.ret - g.static) * 100
    g["d_mdd"] = (g.mdd - g.static_mdd) * 100
    pd.set_option("display.width", 250)
    tab = g.groupby(["b", "clf", "action", "var"]).agg(d_static=("d_static", "median"), d_mdd=("d_mdd", "median"),
                                                       turn=("turn", "median")).unstack("var")
    print(tab.round(2).to_string())
    t["b"] = t.b.astype(int)
    print(t[t.b >= 180].groupby("b").et_min.value_counts(normalize=True).round(3).to_string())
