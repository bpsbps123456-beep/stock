"""c2 audit 6: 구간 경계 결과 겹침(P1 → P2)과 P2 잘림(censored)이 시험 1(단독 3분위)에 주는 영향의 크기.

시험 1의 한 칸 = 단서 × 방향(매수·매도) × 결과. 3분위 경계는 P1 사건(같은 방향)으로 정한다. 아래 3분위 = x ≤ q1/3,
위 3분위 = x > q2/3(이진 단서는 0과 1로 갈린다). 차이 = 위 평균 − 아래 평균(로그).
(a) 겹침: P1 사건 결과가 6/1 이후 가격을 쓰면 y1을 5/29 종가로 자르고(엠바고), y10·y20은 끝이 5/29를 넘는 사건을 뺀다.
    P1 차이의 부호가 몇 칸에서 바뀌는지.
(b) 잘림: P2 y1에서 잘린 사건을 뺄 때 P2 차이 부호가 몇 칸 바뀌는지, 잘림 비율이 3분위마다 얼마나 다른지.

py code/c2audit_embargo.py → results/c2audit_embargo.csv
"""
from __future__ import annotations

import math
import pathlib
from datetime import date

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
FEATS = ["M1", "M2", "M3a", "M3b", "M4a", "M4b", "M4c", "M5",
         "S1", "S2", "S3a", "S3b", "S4", "S5", "S6", "S7", "S8", "S9"]
P1_END = date(2026, 5, 29)


def diff(x, y, q1, q2):
    lo, hi = y[x <= q1], y[x > q2]
    lo, hi = lo[~np.isnan(lo)], hi[~np.isnan(hi)]
    return (hi.mean() - lo.mean()) if len(lo) >= 5 and len(hi) >= 5 else np.nan


def main():
    raw = pd.read_parquet(ROOT / "cache" / "daily.parquet")
    G = {s: g.sort_values("date").reset_index(drop=True) for s, g in raw.groupby("sym")}
    out = []
    for det in ("ZZA3", "ZZP10"):
        ev = pd.read_parquet(ROOT / "results" / f"events_{det}.parquet")
        p1 = ev[ev.period == "P1"].copy()
        emb1, emb10, emb20, share = [], [], [], []
        for _, e in p1.iterrows():
            d = G[e.sym]
            ds = list(d.date)
            O, C = d.O.to_numpy(float), d.C.to_numpy(float)
            c = int(e.conf_idx)
            j_end = len(ds) - 1 if e.censored else ds.index(e.exit_date) + 1
            k = ds.index(P1_END)
            if ds[j_end] > P1_END:
                emb1.append(e.type * math.log(C[k] / O[c + 1]))
                share.append((e.y1 - emb1[-1]))
            else:
                emb1.append(e.y1)
                share.append(0.0)
            emb10.append(e.y10 if ds[c + 11] <= P1_END else np.nan)
            emb20.append(e.y20 if ds[c + 21] <= P1_END else np.nan)
        p1["y1_emb"], p1["y10_emb"], p1["y20_emb"], p1["y1_after"] = emb1, emb10, emb20, share
        p2 = ev[ev.period == "P2"]
        for t in (1, -1):
            a, b = p1[p1.type == t], p2[p2.type == t]
            print(det, "type", t, "P1 y1 합 중 6/1 이후 가격 몫 %.1f%%" % (100 * a.y1_after.sum() / a.y1.sum()) if a.y1.sum() else "",
                  "| 평균 y1 %.4f → 엠바고 %.4f" % (a.y1.mean(), a.y1_emb.mean()))
            for f in FEATS:
                q1, q2 = np.quantile(a[f].astype(float), [1 / 3, 2 / 3])
                xa, xb = a[f].astype(float).values, b[f].astype(float).values
                row = dict(det=det, type=t, feat=f)
                for y in ("y1", "y10", "y20"):
                    row[f"P1_{y}"] = diff(xa, a[y].values, q1, q2)
                    row[f"P1emb_{y}"] = diff(xa, a[y + "_emb"].values, q1, q2)
                    row[f"P2_{y}"] = diff(xb, b[y].values, q1, q2)
                unc = ~b.censored.values
                row["P2_y1_uncens"] = diff(xb[unc], b.y1.values[unc], q1, q2)
                bd = b.date.values
                row["P2_n_lo"], row["P2_n_hi"] = int((xb <= q1).sum()), int((xb > q2).sum())
                row["P2_dates_lo"], row["P2_dates_hi"] = len(set(bd[xb <= q1])), len(set(bd[xb > q2]))
                cz = b.censored.values.astype(float)
                row["P2_cens_lo"] = cz[xb <= q1].mean() if (xb <= q1).any() else np.nan
                row["P2_cens_hi"] = cz[xb > q2].mean() if (xb > q2).any() else np.nan
                out.append(row)
    R = pd.DataFrame(out)
    R.to_csv(ROOT / "results" / "c2audit_embargo.csv", index=False)
    for det, g in R.groupby("det"):
        print(f"[{det}] 칸 {len(g)}개 × 결과 3")
        for y in ("y1", "y10", "y20"):
            ok = g[[f"P1_{y}", f"P1emb_{y}"]].notna().all(axis=1)
            flip = (np.sign(g[f"P1_{y}"]) != np.sign(g[f"P1emb_{y}"])) & ok
            agree_full = ((np.sign(g[f"P1_{y}"]) == np.sign(g[f"P2_{y}"])) & g[f"P2_{y}"].notna()).sum()
            agree_emb = ((np.sign(g[f"P1emb_{y}"]) == np.sign(g[f"P2_{y}"])) & g[f"P2_{y}"].notna()).sum()
            print(f"  {y}: 엠바고로 P1 부호가 바뀐 칸 {int(flip.sum())}/{int(ok.sum())}, P1·P2 같은 부호 {agree_full} → 엠바고 {agree_emb}")
        ok = g[["P2_y1", "P2_y1_uncens"]].notna().all(axis=1)
        flip = (np.sign(g.P2_y1) != np.sign(g.P2_y1_uncens)) & ok
        gapc = (g.P2_cens_hi - g.P2_cens_lo).abs()
        print(f"  P2 y1: 잘린 사건을 빼면 부호가 바뀐 칸 {int(flip.sum())}/{int(ok.sum())}, 3분위 간 잘림 비율 차 중앙 {gapc.median():.2f}, 최대 {gapc.max():.2f}")
        thin = g[(g[["P2_dates_lo", "P2_dates_hi"]].min(axis=1) < 10)]
        print(f"  P2에서 P1 경계 3분위(위·아래) 중 하나가 날짜 10일 미만인 칸 {len(thin)}/{len(g)}:",
              thin.apply(lambda r: f"{r.feat}({'매수' if r.type == 1 else '매도'}) 날짜 {r.P2_dates_lo}/{r.P2_dates_hi}", axis=1).tolist())
        print(g.loc[gapc.sort_values(ascending=False).index[:6], ["type", "feat", "P2_cens_lo", "P2_cens_hi", "P2_y1", "P2_y1_uncens"]].round(3).to_string())


if __name__ == "__main__":
    main()
