"""원안 시간봉 효과를 1차 연구(results/xsec, 이상적 체결·비용 0)와 대조한다(갈래 strat).

- 1차 칸: 연장(04:00), N10, 개수, 따라가기, 밤 보유, 0.5/0.5. 5분봉 창 5~360분 72개, 1시간봉 창 60~360분 6개.
- 같은 비중 = 평균 보유 비율 × 보유 수익(단순). 144종목 중앙 (수익 − 같은 비중).
- mtf 이상적 판 원안(A3)과 b 9개에서 종목별로 맞는지 본다(확인 구간은 예열 차이가 없어 거의 같아야 한다).

py code/mtfstrat_xsec_tf.py → results/mtfstrat_xsec_tf.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from engine import HERE
from zones_mtf import BASE_TFS

RES = HERE / "results"
XS = RES / "xsec"


def curve(base):
    cells = pd.read_parquet(XS / f"cells_{base}.parquet")
    m = (cells.sess == "eth") & (cells.N == 10) & (cells.ratio == "count") & (cells.dir == "follow") & (~cells.dayflat) & (cells.b == 0.5) & (cells.s == 0.5)
    idx = cells.index[m].values
    tfs = cells.tf.values[idx]
    rows = []
    for f in sorted(XS.joinpath(base).glob("*_meta.json")):
        sym = f.name[:-10]
        meta = json.load(open(f))
        a = np.load(XS / base / f"{sym}.npy")[:, idx]
        bh = pd.DataFrame(meta["bh"])
        bh = bh[bh.sess == "eth"].set_index("tf")
        bh[["bh_sel", "bh_conf"]] = np.expm1(bh[["bh_sel", "bh_conf"]])  # meta의 보유 수익은 로그다
        for j, tf in enumerate(tfs):
            rs, rc = np.expm1(a[0, j]), np.expm1(a[1, j])
            xs, xc = a[2, j], a[3, j]
            rows.append(dict(base=base, sym=sym, tf=int(tf), ret_sel=rs, ret_conf=rc, x_sel=xs, x_conf=xc,
                             bh_sel=bh.loc[tf, "bh_sel"], bh_conf=bh.loc[tf, "bh_conf"],
                             ds=rs - xs * bh.loc[tf, "bh_sel"], dc=rc - xc * bh.loc[tf, "bh_conf"]))
    return pd.DataFrame(rows)


def main():
    out = []
    for base in ("5m", "1h"):
        d = curve(base)
        med = d.groupby("tf")[["ds", "dc", "ret_sel", "ret_conf", "x_sel", "x_conf"]].median()
        med["f_dc"] = d.groupby("tf").dc.apply(lambda v: (v >= 0).mean())
        med["f_ds"] = d.groupby("tf").ds.apply(lambda v: (v >= 0).mean())
        med["base"] = base
        out.append(med.reset_index())
        print(f"\n[1차 {base} 창, 원안 꼴 시간봉별 144 중앙 (수익 − 같은 비중), 이상적·비용 0]")
        print(f" 시간봉 {len(med)}개, 순위상관(tf, 중앙): 고르기 {spearmanr(med.index, med.ds).statistic:+.2f}, 확인 {spearmanr(med.index, med.dc).statistic:+.2f}")
        sub = med.loc[[t for t in med.index if t in BASE_TFS or base == "1h"]]
        for tf, r in sub.iterrows():
            print(f"  {tf:4d}분 고르기 {100 * r.ds:+.1f} ({r.f_ds:.0%}≥) 확인 {100 * r.dc:+.1f} ({r.f_dc:.0%}≥) | 수익 {100 * r.ret_sel:+.1f} {100 * r.ret_conf:+.1f}")
        if base == "5m":
            # 5~360분 72개를 구간으로 묶은 모양
            bins = pd.cut(med.index, [0, 30, 60, 120, 240, 360])
            print("  묶음 중앙:", {str(k): (round(100 * v.ds.median(), 1), round(100 * v.dc.median(), 1)) for k, v in med.groupby(bins, observed=True)})
            # mtf 이상적 판 원안과 종목별 대조
            g = pd.read_parquet(RES / "mtf_rows.parquet", filters=[("k", "==", 0), ("board", "==", "ideal"), ("action", "==", "A3")])
            g["sym"] = g.sym.astype(str)
            m = g.merge(d, left_on=["sym", "b"], right_on=["sym", "tf"], suffixes=("", "_x"))
            print("  mtf 이상적 원안 대 1차 칸 (종목별):")
            for b, x in m.groupby("b"):
                print(f"   b{b:<4d} 확인 수익 최대 차이 {np.abs(x.ret_conf - x.ret_conf_x).max():.2e}, 고르기 수익 상관 {np.corrcoef(x.ret_sel, x.ret_sel_x)[0, 1]:.3f}"
                      f" 중앙 차이 {100 * (x.ret_sel - x.ret_sel_x).median():+.2f}%p | mtf 중앙 (수익−같은비중) 고르기 {100 * (x.ret_sel - x.static_sel).median():+.1f} 확인 {100 * (x.ret_conf - x.static_conf).median():+.1f}"
                      f" / 1차 {100 * x.ds.median():+.1f} {100 * x.dc.median():+.1f}")
    pd.concat(out).to_csv(RES / "mtfstrat_xsec_tf.csv", index=False)


if __name__ == "__main__":
    main()
