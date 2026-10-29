"""종목별 1분봉 → (N, tol)마다 ①·② 후보 표를 만들어 한데 모은다(+ 같은 분 다른 종목 대조값).
사용: py build.py alp1m 2026-07-01 2026-10-28      → cache/cand_alp1m/{A,B}_{N}_{tol}.parquet, plain_{A,B}_{N}.parquet
      py build.py alp1m_2025 2025-07-01 2025-10-31
      py build.py yf1m 2026-09-30 2026-10-28"""
import os, sys, json, time
import numpy as np
import pandas as pd
import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../../../..")))
import watchlist  # noqa: E402

SYMS = sorted(s for s in watchlist.AI_SUBSECTOR if "." not in s)
MIN_FIELDS = ["r1_e2", "r1net_e2", "race_e2", "hit_e2", "r1_e0", "hit_e0"]


def day_int(s):
    return int(s.replace("-", ""))


def main(src, a_day, b_day):
    out_dir = os.path.join(E.CACHE, f"cand_{src}")
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = day_int(a_day), day_int(b_day)
    parts = {}
    minute_parts, stats, removed = [], {}, {}
    t0 = time.time()
    for sid, sym in enumerate(SYMS):
        d = E.load(sym, src)
        if d is None or len(d) < 500:
            stats[sym] = {"bars": 0}
            continue
        a = E.prep(d)
        out = E.bar_outcomes(a)
        inper = (a["day"] >= lo) & (a["day"] <= hi)
        v = out["valid_e2"] & out["valid_e0"] & inper
        m = pd.DataFrame({"ts": a["ts"][v], **{f: out[f][v] for f in MIN_FIELDS}})
        minute_parts.append(m)
        stats[sym] = {"bars": int(inper.sum()), "valid": int(v.sum()), "days": int(len(np.unique(a["day"][inper]))),
                      "sane": int((a["sane"] & inper).sum())}
        for N in E.NS:
            for tol in E.TOLS:
                A, B = E.candidates(sym, a, out, N, tol)
                for tag, X in (("A", A), ("B", B)):
                    if X is not None and len(X):
                        X = X[(X.day >= lo) & (X.day <= hi)]
                        c = removed.setdefault(f"{tag}_{N}_{tol}", {"rows": 0, "invalid": 0, "invalid_tradable": 0,
                                                                     "insane": 0, "insane_tradable": 0})
                        c["rows"] += len(X)
                        c["invalid"] += int((~X.valid).sum())
                        c["invalid_tradable"] += int((~X.valid & X.tradable).sum())
                        c["insane"] += int((~X.sane).sum())
                        c["insane_tradable"] += int((~X.sane & X.tradable).sum())
                        X = X[X.valid].copy()
                        X.insert(0, "sid", sid)
                        parts.setdefault(f"{tag}_{N}_{tol}", []).append(X)
            A, B = E.candidates(sym, a, out, N, 0.0, plain=True)
            for tag, X in (("A", A), ("B", B)):
                if X is not None and len(X):
                    X = X[X.valid & (X.day >= lo) & (X.day <= hi)].copy()
                    X.insert(0, "sid", sid)
                    parts.setdefault(f"plain_{tag}_{N}", []).append(X)
        print(f"{sym} {stats[sym]} {time.time() - t0:.0f}s", flush=True)
    # 같은 분 다른 종목: 그 분 모든 종목 유효 봉의 합·수(자기 자신은 뒤에서 뺀다)
    mm = pd.concat(minute_parts, ignore_index=True)
    g = mm.groupby("ts")
    agg = pd.DataFrame({"cnt": g.size()})
    for f in MIN_FIELDS:
        agg[f"s_{f}"] = g[f].sum()
        agg[f"n_{f}"] = g[f].count()
    for key, xs in parts.items():
        X = pd.concat(xs, ignore_index=True).sort_values(["sid", "ts"], kind="stable").reset_index(drop=True)
        j = agg.reindex(X.ts.to_numpy())
        for f in MIN_FIELDS:
            own = X[f].to_numpy()
            has = ~np.isnan(own)
            s = j[f"s_{f}"].to_numpy() - np.where(has, own, 0.0)
            n = j[f"n_{f}"].to_numpy() - has
            with np.errstate(invalid="ignore", divide="ignore"):
                X[f"bt_{f}"] = np.where(n > 0, s / n, np.nan)
        X["n_same_min"] = j["cnt"].to_numpy()
        X.to_parquet(os.path.join(out_dir, f"{key}.parquet"))
    json.dump({"src": src, "from": a_day, "to": b_day, "syms": SYMS, "stats": stats, "removed": removed},
              open(os.path.join(out_dir, "meta.json"), "w"), ensure_ascii=False, indent=1)
    print("done", len(parts), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
