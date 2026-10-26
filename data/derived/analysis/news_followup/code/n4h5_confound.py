"""N4-H5 사후 점검: 'R 사전 표지 → 매수 확정 실패(반대 방향)'가 swing_trend 시장 단서(M2·M3b·M5·M4a)의 대리인지.

py code/n4h5_confound.py → results/n4h5_confound.csv
- 주식 133종목 매수 확정(events_ZZA3)에서 날짜 단위 R 사전 표지(kb ∈ [c−5, c], n4h5_run.py 와 같은 정의).
- 층화 차이: 같은 구간 사건 날짜를 시장 단서 3분위(그 구간 날짜 분포)로 나눠 층마다 R 있음 − 없음 평균 y, 층 사건 수 조화평균 가중.
- 귀무: R 날짜 표지를 고유 날짜 D개에서 원형 회전 k = 1…D−1 전부(층은 날짜에 고정).
- 점-이연 상관: 날짜 단위 R 표지와 단서.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "code"))
import n4h5_run as H  # noqa: E402

CUES = ["M2", "M3a", "M3b", "M4a", "M5"]


def strat_diff(y, lab, strata):
    num = den = 0.0
    for s in np.unique(strata[~pd.isna(strata)]):
        m = (strata == s) & ~np.isnan(y)
        a, b = m & lab, m & ~lab
        if a.sum() < 5 or b.sum() < 5:
            continue
        w = 2 / (1 / a.sum() + 1 / b.sum())
        num += w * (y[a].mean() - y[b].mean())
        den += w
    return num / den if den else np.nan


def main():
    raw = pd.read_parquet(HERE / "cache" / "n3_raw.parquet")
    z = H.zz_runs(raw, "SOXX")
    dates = z["dates"]
    nt = H.note_table(dates)
    R = H.note_sets(nt)[("R", "all")].kb.to_numpy()
    ev = pd.read_parquet(H.SWINGR / "events_ZZA3.parquet")
    ev["date"] = pd.to_datetime(ev.date)
    ev["sidx"] = dates.searchsorted(ev.date)
    rows = []
    for t in (1, -1):
        for per in ("ALL", "Y2026", "Y25", "Y24", "P2"):
            if per == "ALL":
                sub = ev[ev.type == t]
            elif per == "Y2026":
                sub = ev[(ev.type == t) & ev.period.isin(["P1", "P2"])]
            else:
                sub = ev[(ev.type == t) & (ev.period == per)]
            sub = sub.reset_index(drop=True)
            U = np.unique(sub.sidx)
            D = len(U)
            di = np.searchsorted(U, sub.sidx)
            LU = H.counts(U, R, -5, 0) > 0
            dmean = sub.groupby(di)[CUES].mean()
            for cue in CUES:
                x = dmean[cue].to_numpy(float)
                ok = ~np.isnan(x)
                corr = float(np.corrcoef(LU[ok].astype(float), x[ok])[0, 1])
                if cue == "M3a":
                    sd = x
                else:
                    q = np.nanquantile(x, [1 / 3, 2 / 3])
                    sd = np.where(np.isnan(x), np.nan, np.where(x <= q[0], 0, np.where(x <= q[1], 1, 2)))
                strata = sd[di]
                for o in ("y1", "y10", "y20"):
                    y = sub[o].to_numpy(float)
                    act = strat_diff(y, LU[di], strata)
                    raw_d = H.diff_stats(y, LU[di])["diff"]
                    nd = np.array([strat_diff(y, np.roll(LU, -k)[di], strata) for k in range(1, D)])
                    fin = np.isfinite(nd)
                    rows.append(dict(type=t, period=per, cue=cue, outcome=o, dates=D, R_dates=int(LU.sum()),
                                     corr_R_cue=corr, raw_diff=raw_d, strat_diff=act,
                                     p_low=float(np.mean(nd[fin] <= act)) if fin.any() else np.nan,
                                     p_high=float(np.mean(nd[fin] >= act)) if fin.any() else np.nan,
                                     null_q05=float(np.quantile(nd[fin], 0.05)) if fin.any() else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(H.RES / "n4h5_confound.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
