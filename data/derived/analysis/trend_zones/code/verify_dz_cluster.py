"""dip_by_zone 차이의 t를 날짜 대신 종목·주 묶음으로도 잰다 (2026-10-11, 검증용). 결과 results/verify_dz_cluster.txt"""
import os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
T = pd.read_csv(os.path.join(RES, "dip_trades.csv"), encoding="utf-8-sig")
T["day"] = pd.to_datetime(T["day"])
T["wk"] = T.day.dt.isocalendar().week
T["b3"] = (T.mae <= -0.03) * 1.0
L = []


def bt(a, b, col, key, n=4000, seed=1):
    da, db = a.groupby(key)[col].agg(["sum", "count"]), b.groupby(key)[col].agg(["sum", "count"])
    ks = np.array(sorted(set(da.index) | set(db.index)))
    A, B = da.reindex(ks).fillna(0).to_numpy(), db.reindex(ks).fillna(0).to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(ks), (n, len(ks)))
    ca, cb = A[idx, 1].sum(1), B[idx, 1].sum(1)
    ok = (ca > 0) & (cb > 0)
    bs = A[idx, 0].sum(1)[ok] / ca[ok] - B[idx, 0].sum(1)[ok] / cb[ok]
    return (a[col].mean() - b[col].mean()) / bs.std(), len(ks), ok.mean()


for scope in ("24시간", "정규장만"):
    for w in T.win.unique():
        x = T[T.win == w]
        if scope == "정규장만":
            x = x[x.ses == "정규"]
        up, dn = x[x.zone.isin([1, 2])], x[x.zone.isin([-1, -2])]
        s = [f"{w} {scope}:"]
        for col in ("ret", "b3"):
            for key in ("day", "sym", "wk"):
                t, k, ok = bt(up, dn, col, key)
                s.append(f"{col}/{key} t {t:+.2f}")
        nw, wa = x[~x.warn], x[x.warn]
        for key in ("day", "wk"):
            t, k, ok = bt(nw, wa, "ret", key)
            s.append(f"SOXX 평상−경고 ret/{key} t {t:+.2f} (묶음 {k}, 양쪽 다 뽑힌 비율 {ok:.2f})")
        L.append(" | ".join(s))
print("\n".join(L))
open(os.path.join(RES, "verify_dz_cluster.txt"), "w", encoding="utf-8").write("\n".join(L))
