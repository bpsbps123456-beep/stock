"""capture 3 보조: 예지자 지연 곡선(늦게 사거나 팔 때 잃는 몫)과 갑작스러운 반전 비율을 무작위 가격(봉 섞기 200개, 모든 종목 같은 날 순서)과 비교.

반전점은 정의상 종가 극값이라 바로 다음 날 새 방향으로 움직이는 몫이 무작위에서도 크다(선택 효과).
실제 곡선이 무작위보다 앞쪽이 더 가파르면 '반전 직후 몰림'이 구조이고, 같으면 극점 정의의 산물이다.
출력: results/capture_delay_null.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import swing
from capture_confirm import PERS, WARM, eligible
from capture_lib import curve_block

NREP = 200
MAXD = 10
TR = {"T15": (0.15, 0.0), "TK4": (0.0, 4.0)}
OUT = swing.HERE / "results"


def main():
    t0 = time.time()
    D = swing.load()
    el = eligible(D)
    nd = MAXD + 1
    rows = []
    for pi_, p in enumerate(PERS):
        for tn, (th, k) in TR.items():
            real = None; null = None
            for s in el[p]:
                g = D[s]
                O, H, L, C = (g[c].values.astype(np.float64) for c in ("O", "H", "L", "C"))
                idx = swing.period_index(g, p)
                j0, j1 = int(idx[0]), int(idx[-1])
                w = slice(j0 - WARM, j1 + 1)
                a = curve_block(O[w].copy(), H[w].copy(), L[w].copy(), C[w].copy(), WARM, 1, 0, False, th, k, MAXD)
                b = curve_block(O[w].copy(), H[w].copy(), L[w].copy(), C[w].copy(), WARM, NREP, 1000 * pi_ + 11, True, th, k, MAXD)
                real = a if real is None else real + a
                null = b if null is None else null + b

            def unpack(acc):
                n_, orc = acc[:, 0], acc[:, 1]
                ent = acc[:, 4:4 + nd] / orc[:, None]
                ext = acc[:, 4 + nd:4 + 2 * nd] / orc[:, None]
                o_as = acc[:, 4 + 2 * nd]
                ent_as = acc[:, 4 + 2 * nd + 1:4 + 2 * nd + 1 + nd] / np.where(o_as > 0, o_as, np.nan)[:, None]
                o_ae = acc[:, 4 + 2 * nd + 1 + nd]
                ext_ae = acc[:, 4 + 2 * nd + 2 + nd:4 + 2 * nd + 2 + 2 * nd] / np.where(o_ae > 0, o_ae, np.nan)[:, None]
                return dict(n=n_, orc_mean=orc / n_, ab_start=acc[:, 2] / n_, ab_end=acc[:, 3] / n_,
                            entry=ent, exit=ext, entry_abrupt=ent_as, exit_abrupt=ext_ae)

            R, N = unpack(real), unpack(null)
            for key in ("n", "orc_mean", "ab_start", "ab_end"):
                x = N[key]
                rows.append(dict(per=p, truth=tn, metric=key, d=-1, real=float(R[key][0]), null_mean=float(np.nanmean(x)),
                                 null05=float(np.nanquantile(x, 0.05)), null95=float(np.nanquantile(x, 0.95)),
                                 p_lo=float(np.mean(x <= R[key][0]))))
            for key in ("entry", "exit", "entry_abrupt", "exit_abrupt"):
                for dd in range(1, nd):
                    x = N[key][:, dd]
                    rows.append(dict(per=p, truth=tn, metric=key, d=dd, real=float(R[key][0, dd]), null_mean=float(np.nanmean(x)),
                                     null05=float(np.nanquantile(x, 0.05)), null95=float(np.nanquantile(x, 0.95)),
                                     p_lo=float(np.nanmean(x <= R[key][0, dd]))))
            print(p, tn, round(time.time() - t0, 1), "s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "capture_delay_null.csv", index=False)


if __name__ == "__main__":
    main()
