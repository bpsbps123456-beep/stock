"""c2 audit 5: 결과(y1·y10·y20) 정의와 구간 경계 점검(사건표 전체, 두 판정).

- y1의 끝: exit_date가 같은 종목 다음 사건(반대 방향 확정) 날짜와 같은지, 방향이 번갈아 나오는지.
- 잘림: censored 사건 수·구간별 분포, 잘린 y1이 10/22 종가로 끝나는지.
- 구간 경계: 결과 창(y1 청산 시가, y10·y20 끝 시가)이 사건 구간 끝을 넘는 사건 수와 그 몫. 특히 P1 사건 결과가
  P2(6/1~) 가격을 쓰는 수, Y25 사건 결과가 2026 가격을 쓰는 수(고르기·확인 구간 결과 겹침).
- S9 백분위 창에 ATR 미정(NaN) 값이 섞이는 사건(conf_idx 260~269).
- 260거래일 이력 조건으로 빠진 확정 수(구간별, 신규 상장 종목).

py code/c2audit_y1.py → results/c2audit_y1_checks.csv, results/c2audit_y1_boundary.csv
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from c2audit_recompute import ETF, PERIODS, which_period, wilder, zz  # noqa: E402  (독립 구현)


def main():
    raw = pd.read_parquet(ROOT / "cache" / "daily.parquet")
    G = {s: g.sort_values("date").reset_index(drop=True) for s, g in raw.groupby("sym")}
    checks, bound = [], []
    for det in ("ZZA3", "ZZP10"):
        ev = pd.read_parquet(ROOT / "results" / f"events_{det}.parquet").sort_values(["sym", "date"]).reset_index(drop=True)
        bad_exit = bad_alt = bad_cens = 0
        rows = []
        for s, g in ev.groupby("sym"):
            d = G[s]
            ds = list(d.date)
            O, C = d.O.to_numpy(float), d.C.to_numpy(float)
            n = len(ds)
            g = g.reset_index(drop=True)
            for i, e in g.iterrows():
                c = int(e.conf_idx)
                if i + 1 < len(g):
                    nx = g.iloc[i + 1]
                    bad_exit += int(e.exit_date != nx.date)
                    bad_alt += int(nx.type != -e.type)
                if e.censored:
                    # 잘린 사건: 10/22 종가로 끝
                    bad_cens += int(abs(e.y1 - e.type * math.log(C[-1] / O[c + 1])) > 1e-12)
                    end_j = n - 1
                else:
                    end_j = ds.index(e.exit_date) + 1
                a, b = PERIODS[e.period]
                rows.append(dict(period=e.period, type=e.type, censored=bool(e.censored),
                                 y1_end=ds[end_j], y1_days=end_j - (c + 1),
                                 y1_cross=ds[end_j] > b, y10_cross=(c + 11 < n and ds[c + 11] > b), y10_nan=np.isnan(e.y10),
                                 y20_cross=(c + 21 < n and ds[c + 21] > b), y20_nan=np.isnan(e.y20),
                                 into_P2=(e.period == "P1" and ds[end_j] >= PERIODS["P2"][0]),
                                 into_2026=(e.period == "Y25" and ds[end_j] >= PERIODS["P1"][0]),
                                 s9_nanwin=c <= 269, y1=e.y1))
        R = pd.DataFrame(rows)
        checks.append(dict(det=det, events=len(ev), exit_not_next_event=bad_exit, non_alternating=bad_alt,
                           censored=int(ev.censored.sum()), censored_y1_not_1022close=bad_cens))
        for (p, t), q in R.groupby(["period", "type"]):
            bound.append(dict(det=det, period=p, type=t, n=len(q), censored=int(q.censored.sum()),
                              y1_cross_period_end=int(q.y1_cross.sum()), y1_cross_share=round(q.y1_cross.mean(), 3),
                              P1_into_P2=int(q.into_P2.sum()), Y25_into_2026=int(q.into_2026.sum()),
                              y1_days_median=float(q.y1_days.median()),
                              y10_cross=int(q.y10_cross.sum()), y20_cross=int(q.y20_cross.sum()),
                              y10_nan=int(q.y10_nan.sum()), y20_nan=int(q.y20_nan.sum()),
                              y1_mean_censored=round(float(q[q.censored].y1.mean()), 4) if q.censored.any() else np.nan,
                              y1_mean_uncensored=round(float(q[~q.censored].y1.mean()), 4),
                              s9_nan_window=int(q.s9_nanwin.sum())))
    # 260거래일 이력 조건으로 빠진 확정
    excl = []
    for det in ("ZZA3", "ZZP10"):
        for s in sorted(set(G) - ETF):
            d = G[s]
            O, H, L, C = (d[k].to_numpy(float) for k in "OHLC")
            st, piv = zz(C, thr=3.0 * wilder(H, L, C)) if det == "ZZA3" else zz(C, pct=0.10)
            for p, c, t in piv:
                pr = which_period(d.date[c])
                if pr in ("P1", "P2", "Y25") and c < 260:
                    excl.append(dict(det=det, sym=s, period=pr))
    X = pd.DataFrame(excl)
    C_ = pd.DataFrame(checks)
    B = pd.DataFrame(bound)
    C_.to_csv(ROOT / "results" / "c2audit_y1_checks.csv", index=False)
    B.to_csv(ROOT / "results" / "c2audit_y1_boundary.csv", index=False)
    pd.set_option("display.width", 250)
    print(C_.to_string())
    print(B.to_string())
    if len(X):
        print("260일 조건으로 빠진 확정(P1·P2·Y25):")
        print(X.groupby(["det", "period"]).agg(n=("sym", "size"), syms=("sym", lambda v: sorted(set(v)))).to_string())


if __name__ == "__main__":
    main()
