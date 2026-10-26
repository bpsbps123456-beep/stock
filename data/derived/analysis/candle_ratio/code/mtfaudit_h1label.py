"""mtfaudit 6: 야후 1시간봉 프리장 이름표 영향. 7/17 이후 5분봉 대조로 확인한 실제 구간:
04:00 = 04:00~04:30, 05:00 = 04:30~05:30, …, 09:00 = 08:30~09:30 (정규장 09:30~·애프터 16:00~는 이름표 그대로 60분, 15:30은 30분).
zones_mtf는 이름표를 시작으로 보고 04:00~08:00의 끝을 +60분(실제보다 30분 늦음, 보수적)으로 둔다. 예열 봉을 실제 구간으로 고쳐 묶으면 값이 얼마나 바뀌는지 독립 재계산으로 잰다.

py code/mtfaudit_h1label.py → results/mtfaudit_h1label.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtfaudit_core as M  # noqa: E402
from mtfaudit_reimpl import BOARDS, CELLS, SYMS  # noqa: E402

_group = M.group


def fixed_group(df, tf, kind):
    if kind != "1h":
        return _group(df, tf, kind)
    idx = pd.DatetimeIndex(df.index)
    lab = np.asarray(idx.hour * 60 + idx.minute)
    keep = (lab >= 240) & (lab < 1200)
    df, idx, lab = df[keep], idx[keep], lab[keep]
    pre = (lab >= 300) & (lab <= 540)
    st_min = np.where(pre, lab - 30, lab)
    en_min = np.where(lab == 240, 270, np.where(pre, lab + 30, np.where(lab == 930, 960, lab + 60)))
    d0 = pd.DatetimeIndex(pd.to_datetime(np.asarray(idx.date))).tz_localize(M.ET)
    st = (d0 + pd.to_timedelta(st_min, unit="m")).as_unit("ns").asi8
    en = (d0 + pd.to_timedelta(en_min, unit="m")).as_unit("ns").asi8
    f = pd.DataFrame(dict(day=np.asarray(idx.date), k=(st_min - 240) // tf, O=df.Open.values.astype(float),
                          H=df.High.values.astype(float), L=df.Low.values.astype(float), C=df.Close.values.astype(float),
                          V=df.Volume.values.astype(float), end=en, first=st))
    out = f.groupby(["day", "k"], sort=True).agg(O=("O", "first"), H=("H", "max"), L=("L", "min"), C=("C", "last"),
                                                  V=("V", "sum"), end=("end", "max"), first=("first", "min"),
                                                  n=("O", "size")).reset_index()
    dd = pd.DatetimeIndex(pd.to_datetime(out.day.values)).tz_localize(M.ET)
    out["start"] = (dd + pd.to_timedelta(240 + out.k.values * tf, unit="m")).as_unit("ns").asi8
    out["warm"] = True
    return out


def main():
    rows = []
    for sym in SYMS:
        raw = M.read_raw(sym)
        for b, clf, act in CELLS:
            res = {}
            for lab, fn in (("coded", _group), ("fixed", fixed_group)):
                M.group = fn
                P = M.build(raw, b)
                flag = P["flags"][clf] if clf != "BASE" else np.zeros(len(P["O"]), bool)
                res[lab] = (P, flag)
            M.group = _group
            for bd, (cost, ideal) in BOARDS.items():
                a, _ = M.metrics(res["coded"][0], M.target(res["coded"][0]["s"], res["coded"][1], act), cost, ideal)
                c, _ = M.metrics(res["fixed"][0], M.target(res["fixed"][0]["s"], res["fixed"][1], act), cost, ideal)
                rows.append(dict(sym=sym, b=b, clf=clf, action=act, board=bd,
                                 d_ret_sel=c["ret_sel"] - a["ret_sel"], d_ret_conf=c["ret_conf"] - a["ret_conf"],
                                 d_x_sel=c["x_sel"] - a["x_sel"]))
    r = pd.DataFrame(rows)
    r.to_csv(M.HERE / "results" / "mtfaudit_h1label.csv", index=False)
    print(r.groupby("board")[["d_ret_sel", "d_ret_conf", "d_x_sel"]].agg(lambda x: float(np.abs(x).max())).to_string())
    print(r[r.board == "real"].round(4).to_string())


if __name__ == "__main__":
    main()
