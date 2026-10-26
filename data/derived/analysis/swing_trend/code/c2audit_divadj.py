"""c2 audit 7: 배당 수정주가의 미래 의존. cache/daily.parquet은 배당 수정(뒤 배당이 앞 가격을 바꿈)이다.
배당 미반영 가격(cache/audit_raw_daily.parquet, 분할만 반영)으로 지그재그 확정을 다시 내 P1·P2·Y25 확정 사건 집합이
얼마나 달라지는지 센다(독립 구현 zz·wilder 사용). → results/c2audit_divadj.csv"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from c2audit_recompute import ETF, which_period, wilder, zz  # noqa: E402

adj = pd.read_parquet(ROOT / "cache" / "daily.parquet")
raw = pd.read_parquet(ROOT / "cache" / "audit_raw_daily.parquet")
rows = []
for s in sorted(set(adj.sym) - ETF):
    a = adj[adj.sym == s].sort_values("date").reset_index(drop=True)
    r = raw[raw.sym == s].sort_values("date").reset_index(drop=True)
    if len(r) != len(a) or (r.date.values != a.date.values).any():
        rows.append(dict(sym=s, note="날짜 불일치"))
        continue
    has_div = bool((r["div"].fillna(0) > 0).any())
    for det in ("ZZA3", "ZZP10"):
        ev = []
        for g in (a, r):
            H, L, C = (g[k].to_numpy(float) for k in "HLC")
            _, piv = zz(C, thr=3.0 * wilder(H, L, C)) if det == "ZZA3" else zz(C, pct=0.10)
            ev.append({(g.date[c], t) for _, c, t in piv if which_period(g.date[c]) in ("P1", "P2", "Y25") and c >= 260})
        rows.append(dict(sym=s, det=det, has_div=has_div, n_adj=len(ev[0]), n_raw=len(ev[1]), only_adj=len(ev[0] - ev[1]),
                         only_raw=len(ev[1] - ev[0])))
R = pd.DataFrame(rows)
R.to_csv(ROOT / "results" / "c2audit_divadj.csv", index=False)
g = R.dropna(subset=["det"]).groupby(["det", "has_div"])[["n_adj", "only_adj", "only_raw"]].sum()
print(g.to_string())
print("배당 종목 수", int(R.dropna(subset=["det"]).drop_duplicates("sym").has_div.sum()), "/ 날짜 불일치", int(R["note"].notna().sum()) if "note" in R else 0)
print(R[(R.only_adj > 0) | (R.only_raw > 0)].to_string())
