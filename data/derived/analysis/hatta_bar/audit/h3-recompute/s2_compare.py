"""Pass 2: build H2 from the independent pattern-(1) list and compare with the engine (h3_signals.csv)."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar/audit/h3-recompute")
import h3r_core as C  # noqa: E402

D0, D1 = 20260102, 20260623


def dedup(df, gap=30):
    df = df.sort_values(["sym", "ts"], kind="mergesort")
    keep = np.zeros(len(df), dtype=bool)
    last_sym, last_ts = None, None
    for i, (s, t) in enumerate(zip(df["sym"].to_numpy(), df["ts"].to_numpy())):
        if s != last_sym or t - last_ts >= gap:
            keep[i] = True
            last_sym, last_ts = s, t
    return df[keep]


def summ(df, col="r1n"):
    return dict(n=int(len(df)), days=int(df["day"].nunique()), syms=int(df["sym"].nunique()),
                hit=float(df["hit"].mean()) if len(df) else np.nan,
                r1=float(df["r1"].mean()) if len(df) else np.nan,
                r1n=float(df[col].mean()) if len(df) else np.nan)


def day_boot(df, col, B=2000, seed=7):
    """Day-cluster bootstrap of the pooled mean; one-sided p = (#{boot<=0}+1)/(B+1)."""
    g = df.groupby("day")[col].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    k = len(s)
    idx = rng.integers(0, k, size=(B, k))
    bm = s[idx].sum(1) / n[idx].sum(1)
    return dict(mean=float(s.sum() / n.sum()), se=float(bm.std(ddof=1)),
                p=float((np.sum(bm <= 0) + 1) / (B + 1)))


def diag(sym, t):
    """Evaluate every pattern-(1) condition at bar t of sym (my implementation)."""
    d = C.load_bars(sym)
    A = C.base_arrays(d)
    sw = C.swings(d["l"])
    S = A["S"][t]
    D, _ = C.drop120(d, t)
    lev, npts = C.level_count(d["l"], sw, t, S)
    out = dict(brk=bool(A["brk"][t]), openok=bool(A["openok"][t]), wick=float(A["wick"][t]),
               wick_ok=bool(A["wick"][t] >= C.R_MIN), san=bool(A["san"][t]), trades=float(d["n"][t]),
               dollar=float(d["v"][t] * d["c"][t]), rng_ratio=float((d["h"][t] - d["l"][t]) / d["c"][t]),
               cnt30=int(A["cnt30"][t]), tr=bool(A["tr"][t]), alsum_b=float(A["alsum"][t - 1]),
               e60_dec=bool(A["e60"][t - 1] < A["e60"][t - 31]), nxt=bool(A["nxt"][t]), e2ok=bool(A["e2ok"][t]),
               D=float(D), lev=int(lev), npts=int(npts), S=float(S), low=float(d["l"][t]), open=float(d["o"][t]),
               lowratio=float(d["l"][t] / (S * (1 - C.TOL))), openratio=float(d["o"][t] / (S * (1 - C.TOL))))
    vw = C.day_vwap(d)
    out["vgap"] = float(d["c"][t] / vw[t] - 1)
    return out


def main():
    sig = pd.read_parquet(C.OUT + "/sig1.parquet")
    sig["inwin"] = C.in_window(sig["mins"].to_numpy())
    per = sig[(sig.day >= D0) & (sig.day <= D1)]
    base = per[per.inwin]
    h2_raw = base[base.vgap < -0.03]
    h2 = dedup(h2_raw)
    res = {}
    res["mine_base_(1)_dedup"] = summ(dedup(base))
    res["mine_H2"] = summ(h2)
    res["mine_H2_boot_r1n"] = day_boot(h2, "r1n")
    # alternative dedup orders (to explain engine differences)
    alt1 = dedup(base)
    alt1 = alt1[alt1.vgap < -0.03]
    res["alt_dedup_before_filter"] = summ(alt1)
    alt2 = dedup(per[per.vgap < -0.03])
    alt2 = alt2[alt2.inwin]
    res["alt_dedup_allhours_then_window"] = summ(alt2)

    eng = pd.read_csv(C.BASE + "/results/h3_signals.csv")
    e2 = eng[eng.hyp == "H2"].copy()
    e2 = e2.rename(columns={"r1_e2": "r1", "hit_e2": "hit"})
    e2["hit"] = e2["hit"].astype(bool)
    res["engine_H2_from_csv"] = summ(e2)

    km = set(zip(h2.sym, h2.ts))
    ke = set(zip(e2.sym, e2.ts))
    both = km & ke
    res["overlap"] = dict(both=len(both), mine_only=len(km - ke), engine_only=len(ke - km))

    # field comparison on overlap
    m = h2.merge(e2, on=["sym", "ts"], suffixes=("_m", "_e"))
    cmpf = {}
    for f in ["t", "day", "mins", "r1", "r1n", "c_in", "c_out"]:
        a, b = m[f + "_m"].to_numpy(float), m[f + "_e"].to_numpy(float)
        cmpf[f] = int(np.sum(~np.isclose(a, b, rtol=0, atol=1e-12)))
    cmpf["hit"] = int(np.sum(m["hit_m"].to_numpy() != m["hit_e"].to_numpy()))
    res["overlap_field_mismatch_counts"] = cmpf
    m.to_csv(C.OUT + "/overlap_merge.csv", index=False)

    # classify engine-only
    sig_key = sig.set_index(["sym", "ts"])
    cand = pd.read_parquet(C.OUT + "/cand.parquet").set_index(["sym", "ts"])
    eo = sorted(ke - km)
    rows = []
    for sym, ts in eo:
        er = e2[(e2.sym == sym) & (e2.ts == ts)].iloc[0]
        r = dict(sym=sym, ts=ts, t=int(er.t), day=int(er.day), mins=int(er.mins), eng_r1=er.r1, eng_hit=er.hit)
        if (sym, ts) in sig_key.index:
            sr = sig_key.loc[(sym, ts)]
            r["cls"] = "in_my_(1)_list"
            r["my_vgap"] = float(sr.vgap)
            r["why"] = "vgap>=-3%" if sr.vgap >= -0.03 else "dedup"
        elif (sym, ts) in cand.index:
            cr = cand.loc[(sym, ts)]
            r["cls"] = "my_candidate_failed"
            r["my_D"] = float(cr.D)
            r["my_lev"] = int(cr.lev)
            r["why"] = ("D<2%" if cr.D < C.D_MIN else "") + (" lev<2" if cr.lev < C.K_MIN else "")
        else:
            dg = diag(sym, int(er.t))
            r["cls"] = "not_my_candidate"
            fails = [k for k in ("brk", "openok", "wick_ok", "san", "tr", "nxt", "e2ok") if not dg[k]]
            r["why"] = ",".join(fails)
            r.update({"dg_" + k: v for k, v in dg.items()})
        rows.append(r)
    eo_df = pd.DataFrame(rows)
    eo_df.to_csv(C.OUT + "/engine_only.csv", index=False)
    res["engine_only_classes"] = eo_df["why"].value_counts().to_dict() if len(eo_df) else {}

    # classify mine-only
    mo = sorted(km - ke)
    rows = []
    for sym, ts in mo:
        sr = h2[(h2.sym == sym) & (h2.ts == ts)].iloc[0]
        r = dict(sym=sym, ts=ts, t=int(sr.t), day=int(sr.day), mins=int(sr.mins), vgap=sr.vgap, D=sr.D,
                 lev=int(sr.lev), r1=sr.r1, hit=sr.hit, r1n=sr.r1n)
        # was there an engine H2 signal for the same symbol within 30 minutes (dedup chain difference)?
        es = e2[e2.sym == sym]
        near = es[(es.ts - ts).abs() < 30]
        r["eng_within30"] = int(len(near))
        rows.append(r)
    mo_df = pd.DataFrame(rows)
    mo_df.to_csv(C.OUT + "/mine_only.csv", index=False)
    res["mine_only_n"] = len(mo_df)
    with open(C.OUT + "/s2_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    h2.to_parquet(C.OUT + "/h2_mine.parquet")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    if len(eo_df):
        print(eo_df.head(40).to_string())
    if len(mo_df):
        print(mo_df.head(40).to_string())


if __name__ == "__main__":
    main()
