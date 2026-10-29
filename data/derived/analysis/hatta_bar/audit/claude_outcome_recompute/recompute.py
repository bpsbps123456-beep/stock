"""Independent outcome re-computation for hatta_bar candidate tables (yf1m).

Written from BRIEF.md v2 '결과 측정' / '비교 기준' and the audit task definitions only;
engine.py outcome functions were NOT read. Reports agreement statistics only (no performance numbers).

usage: py recompute.py [n_sample=400] [seed=20261029]
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
import numba as nb

HERE = os.path.dirname(os.path.abspath(__file__))
HB = os.path.abspath(os.path.join(HERE, "../.."))
REPO = os.path.abspath(os.path.join(HB, "../../../.."))
sys.path.insert(0, REPO)
import watchlist  # noqa: E402

RAW = os.path.join(HB, "cache", "yf1m")
CAND = os.path.join(HB, "cache", "cand_yf1m")
TOL = 1e-9
COST = np.array([0.003, 0.001, 0.0005, 0.003])  # pre / open / mid / post, one-way

SYMS = sorted(s for s in watchlist.AI_SUBSECTOR if "." not in s)
_files = sorted(f[:-8] for f in os.listdir(RAW) if f.endswith(".parquet"))
assert [f for f in _files if f not in ("SPY", "QQQ")] == SYMS, "sid mapping mismatch"


def load_raw(sym):
    d = pd.read_parquet(os.path.join(RAW, f"{sym}.parquet"))
    return d


def arrays(d):
    idx = d.index
    if idx.tz is None:
        raise ValueError("naive index")
    idx_et = idx.tz_convert("America/New_York")
    ts = (idx.tz_convert("UTC").as_unit("s").asi8 // 60).astype(np.int64)
    mins = (idx_et.hour * 60 + idx_et.minute).to_numpy().astype(np.int64)
    day = (idx_et.year * 10000 + idx_et.month * 100 + idx_et.day).to_numpy().astype(np.int64)
    o = d["Open"].to_numpy(float); h = d["High"].to_numpy(float)
    l = d["Low"].to_numpy(float); c = d["Close"].to_numpy(float)
    sess = np.where(mins < 570, 0, np.where(mins < 660, 1, np.where(mins < 960, 2, 3))).astype(np.int64)
    return dict(ts=ts, mins=mins, day=day, o=o, h=h, l=l, c=c, sess=sess)


@nb.njit(cache=True)
def atr14(h, l, c):
    n = len(h); out = np.full(n, np.nan); tr = np.empty(n)
    for i in range(n):
        if i == 0:
            tr[i] = h[i] - l[i]
        else:
            a = h[i] - l[i]; b = abs(h[i] - c[i - 1]); e = abs(l[i] - c[i - 1])
            tr[i] = max(a, max(b, e))
    if n < 14:
        return out
    s = 0.0
    for i in range(14):
        s += tr[i]
    out[13] = s / 14.0
    for i in range(14, n):
        out[i] = (out[i - 1] * 13.0 + tr[i]) / 14.0
    return out


@nb.njit(cache=True)
def scan(e, j0, last, h, l, c, up_mult, dn_mult):
    """returns hit, r1, race over bars j0..last (entry e)."""
    hit = 0.0; race = np.nan; r1 = np.nan
    tgt = e * up_mult; dn = e * dn_mult
    for j in range(j0, last + 1):
        u = h[j] >= tgt; d = l[j] <= dn
        if np.isnan(race):
            if u and d:
                race = 0.5
            elif u:
                race = 1.0
            elif d:
                race = 0.0
        if u and hit == 0.0:
            hit = 1.0
    r1 = 0.01 if hit == 1.0 else c[last] / e - 1.0
    return hit, r1, race


@nb.njit(cache=True)
def bar_out(ts, day, sess, o, h, l, c, cost):
    n = len(ts)
    v0 = np.zeros(n, np.bool_); v2 = np.zeros(n, np.bool_)
    r1_0 = np.full(n, np.nan); hit_0 = np.full(n, np.nan); race_0 = np.full(n, np.nan)
    r1_2 = np.full(n, np.nan); hit_2 = np.full(n, np.nan); race_2 = np.full(n, np.nan)
    hits_2 = np.full(n, np.nan); r1net_2 = np.full(n, np.nan); lastw = np.full(n, -1, np.int64)
    for t in range(n):
        end = ts[t] + 30
        if t + 1 >= n or day[t + 1] != day[t] or ts[t + 1] > end:
            continue
        last = t + 1
        while last + 1 < n and day[last + 1] == day[t] and ts[last + 1] <= end:
            last += 1
        lastw[t] = last
        v0[t] = True
        e0 = o[t + 1]
        hh, rr, ra = scan(e0, t + 1, last, h, l, c, 1.01, 0.99)
        hit_0[t] = hh; r1_0[t] = rr; race_0[t] = ra
        if last >= t + 2:
            v2[t] = True
            e2 = c[t + 1]
            hh, rr, ra = scan(e2, t + 2, last, h, l, c, 1.01, 0.99)
            hit_2[t] = hh; r1_2[t] = rr; race_2[t] = ra
            hs, _, _ = scan(e2, t + 2, last, h, l, c, 1.011, 0.99)
            hits_2[t] = hs
            k = cost[sess[t]]
            r1net_2[t] = rr - k if hh == 1.0 else rr - 2.0 * k
    return v0, v2, r1_0, hit_0, race_0, r1_2, hit_2, race_2, hits_2, r1net_2, lastw


def r2_stop(t, stop, a, lastw):
    last = lastw[t]
    if last < t + 2:
        return np.nan
    o, h, l, c = a["o"], a["h"], a["l"], a["c"]
    e = c[t + 1]
    tgt = e * 1.01
    for j in range(t + 2, last + 1):
        if l[j] < stop:
            fill = o[j] if o[j] < stop else stop
            return fill / e - 1.0
        if h[j] >= tgt:
            return 0.01
    return c[last] / e - 1.0


def terciles(pct, method):
    if method == "floor":          # floor(pct*3) clipped to 2
        return np.minimum(np.floor(pct * 3.0), 2.0)
    if method == "gt":             # (pct>1/3)+(pct>2/3)
        return (pct > 1 / 3).astype(float) + (pct > 2 / 3).astype(float)
    raise ValueError(method)


def main(n_sample=400, seed=20261029):
    rng = np.random.default_rng(seed)
    # ---- sample rows across A_* and B_* tables
    tabs = sorted(glob.glob(os.path.join(CAND, "A_*.parquet")) + glob.glob(os.path.join(CAND, "B_*.parquet")))
    frames = []
    total_rows = {}
    for f in tabs:
        X = pd.read_parquet(f)
        key = os.path.basename(f)[:-8]
        total_rows[key] = len(X)
        X = X.assign(table=key, row=np.arange(len(X)))
        frames.append(X)
    ALL = pd.concat(frames, ignore_index=True)
    pick = rng.choice(len(ALL), size=n_sample, replace=False)
    S = ALL.iloc[np.sort(pick)].reset_index(drop=True)

    # ---- per-symbol bar outcomes (all syms, for bt) + extra arrays for sample syms
    per = {}
    minute_rows = []
    raw_issues = {}
    for sid, sym in enumerate(SYMS):
        d = load_raw(sym)
        iss = {"dup_idx": int(d.index.duplicated().sum()), "hi_lt_lo": int((d.High < d.Low).sum()),
               "low_le0": int((d.Low <= 0).sum()), "unsorted": int(not d.index.is_monotonic_increasing),
               "nan": int(d[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum())}
        if any(iss.values()):
            raw_issues[sym] = iss
        a = arrays(d)
        res = bar_out(a["ts"], a["day"], a["sess"], a["o"], a["h"], a["l"], a["c"], COST)
        names = ["v0", "v2", "r1_e0", "hit_e0", "race_e0", "r1_e2", "hit_e2", "race_e2", "hits_e2", "r1net_e2", "lastw"]
        a.update(dict(zip(names, res)))
        gap1 = np.full(len(a["ts"]), 10**9); gap1[:-1] = a["ts"][1:] - a["ts"][:-1]
        a["vg"] = gap1 <= 3          # next bar starts within 2 min after signal close (BRIEF hygiene)
        a["atr"] = atr14(a["h"], a["l"], a["c"])
        a["atrp"] = a["atr"] / a["c"]
        per[sym] = a
        vb = a["v0"] & a["v2"] & a["vg"]
        minute_rows.append(pd.DataFrame({"sid": sid, "ts": a["ts"][vb],
                                         **{f: a[f][vb] for f in ("r1_e2", "r1net_e2", "race_e2", "hit_e2", "r1_e0", "hit_e0")}}))
    MM = pd.concat(minute_rows, ignore_index=True)

    # ---- recompute sampled rows
    fields_direct = ["r1_e0", "hit_e0", "r1_e2", "r1net_e2", "hit_e2", "hits_e2", "race_e2"]
    rec = []
    for i, r in S.iterrows():
        sym = SYMS[int(r.sid)]
        a = per[sym]
        t = int(r.t)
        row = {"i": i, "table": r.table, "row": int(r.row), "sid": int(r.sid), "sym": sym, "t": t, "ts": int(r.ts),
               "sym_col_ok": (r.sym == sym) if "sym" in S.columns else None,
               "t_ts_ok": bool(t < len(a["ts"]) and a["ts"][t] == int(r.ts)),
               "k_ok": bool(int(r.k) == (t if r.table.startswith("A") else t - 1))}
        for f in fields_direct:
            row[f] = a[f][t]
        # R2 stop
        stop = a["l"][t] if r.table.startswith("A") else min(a["l"][t - 1], a["l"][t])
        row["r2_e2"] = r2_stop(t, stop, a, a["lastw"])
        row["v0"] = bool(a["v0"][t]); row["v2"] = bool(a["v2"][t]); row["vg"] = bool(a["vg"][t])
        row["sess"] = int(a["sess"][t]); row["day"] = int(a["day"][t])
        # bt: other symbols' valid (v0&v2) bars at same ts
        o_ = MM[(MM.ts == int(r.ts)) & (MM.sid != int(r.sid))]
        row["bt_r1_e2"] = o_["r1_e2"].mean() if len(o_) else np.nan
        row["bt_r1net_e2"] = o_["r1net_e2"].mean() if len(o_) else np.nan
        row["bt_n"] = len(o_)
        # bv variants
        g = (a["day"] == a["day"][t]) & (a["sess"] == a["sess"][t])
        for vmode in ("v_and", "v_own", "vg_and", "vg_own"):
            for tmode in ("all", "valid"):
                for meth in ("floor", "gt"):
                    for rk in ("average", "max", "min"):
                        base = g & ~np.isnan(a["atrp"])
                        if tmode == "valid":
                            base = base & a["v0"] & a["v2"] & a["vg"]
                        idxs = np.where(base)[0]
                        if t not in set(idxs.tolist()):
                            continue
                        pct = pd.Series(a["atrp"][idxs]).rank(method=rk, pct=True).to_numpy()
                        terc = terciles(pct, meth)
                        my = terc[np.searchsorted(idxs, t)]
                        sel = idxs[terc == my]
                        tag = f"{vmode}|{tmode}|{meth}|{rk}"
                        for f, vf in (("r1_e2", "v2"), ("r1net_e2", "v2"), ("hit_e2", "v2"), ("race_e2", "v2"),
                                      ("r1_e0", "v0"), ("hit_e0", "v0")):
                            if vmode == "v_and":
                                vm = a["v0"][sel] & a["v2"][sel]
                            elif vmode == "vg_and":
                                vm = a["v0"][sel] & a["v2"][sel] & a["vg"][sel]
                            elif vmode == "vg_own":
                                vm = a[vf][sel] & a["vg"][sel]
                            else:
                                vm = a[vf][sel]
                            vals = a[f][sel][vm]
                            row[f"bv_{f}|{tag}"] = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
        rec.append(row)
    R = pd.DataFrame(rec)

    def cmp(mine, theirs):
        mine = np.asarray(mine, float); theirs = np.asarray(theirs, float)
        both_nan = np.isnan(mine) & np.isnan(theirs)
        close = np.abs(mine - theirs) <= TOL
        return ~(both_nan | close)

    report = {"n_sample": n_sample, "seed": seed, "table_rows": total_rows, "raw_issues": raw_issues,
              "sample_by_table": S.table.value_counts().sort_index().to_dict(),
              "sample_by_kind": S.table.str[0].value_counts().to_dict(),
              "sym_col_ok": int(R.sym_col_ok.sum()), "t_ts_ok": int(R.t_ts_ok.sum()), "k_ok": int(R.k_ok.sum()),
              "own_v0": int(R.v0.sum()), "own_v2": int(R.v2.sum()), "own_vg": int(R.vg.sum()), "fields": {}}
    mism = {}
    for f in fields_direct + ["r2_e2", "bt_r1_e2", "bt_r1net_e2"]:
        m = cmp(R[f], S[f])
        report["fields"][f] = {"mismatch": int(m.sum()), "nan_mine": int(np.isnan(R[f].astype(float)).sum()),
                               "nan_table": int(np.isnan(S[f].astype(float)).sum())}
        mism[f] = m
    # bv: pick best variant per field, report all variants
    bvv = {}
    for f in ("r1_e2", "hit_e2", "race_e2", "r1net_e2", "r1_e0", "hit_e0"):
        cols = [c for c in R.columns if c.startswith(f"bv_{f}|")]
        for ccol in cols:
            bvv[ccol] = int(cmp(R[ccol], S[f"bv_{f}"]).sum())
    report["bv_variants"] = dict(sorted(bvv.items(), key=lambda kv: kv[1]))
    out = os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    R.to_csv(os.path.join(out, "mine.csv"), index=False)
    S.to_csv(os.path.join(out, "sample.csv"), index=False)
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in report.items() if k not in ("bv_variants",)}, indent=1, default=str))
    print("bv variants (best 12):")
    for k, v in list(report["bv_variants"].items())[:12]:
        print(" ", k, v)
    return R, S


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else 20261029
    main(n, sd)
