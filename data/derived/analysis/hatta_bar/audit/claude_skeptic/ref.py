"""Auditor's independent re-implementation of BRIEF v2 (does NOT import code/engine.py).
Written from the BRIEF text with explicit per-bar loops. Produces candidate tables comparable to
cache/cand_<src>/{A,B}_{N}_{tol}.parquet and stores them under audit/claude_skeptic/ref_<src>/.
Usage: py ref.py yf1m 2026-09-30 2026-10-28
BLINDING: this script prints no performance statistics."""
import os, sys, json, time
import numpy as np
import pandas as pd
import numba as nb

BASE = "C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar"
ET = "America/New_York"
NS, TOLS, PS = (60, 120, 240, 390), (0.0, 0.0005, 0.001), (2, 3, 5)
COST = np.array([0.003, 0.001, 0.0005, 0.003])


def load(src, sym):
    path = f"{BASE}/cache/{src}/{sym}.parquet"
    if not os.path.exists(path):
        return None
    d = pd.read_parquet(path)
    cols = ["Open", "High", "Low", "Close", "Volume"] + (["Trades"] if "Trades" in d.columns else [])
    d = d[cols].astype(float)
    d = d[(d.High >= d.Low) & (d.Low > 0)]
    d = d[~d.index.duplicated()].sort_index()
    return d


# ---------------- per-bar indicator loops (Pine definitions) ----------------
@nb.njit
def ema(x, n):
    a = 2.0 / (n + 1)
    out = np.empty(len(x))
    for i in range(len(x)):
        out[i] = x[i] if i == 0 else a * x[i] + (1.0 - a) * out[i - 1]
    return out


@nb.njit
def atr14(h, l, c):
    n = len(h)
    tr = np.empty(n)
    out = np.full(n, np.nan)
    for i in range(n):
        if i == 0:
            tr[i] = h[i] - l[i]
        else:
            tr[i] = max(h[i] - l[i], max(abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    if n >= 14:
        s = 0.0
        for i in range(14):
            s += tr[i]
        out[13] = s / 14.0
        for i in range(14, n):
            out[i] = (1.0 / 14.0) * tr[i] + (13.0 / 14.0) * out[i - 1]     # Pine rma form
    return out


@nb.njit
def time_feats(ts, h):
    """hh120[t] = max high over bars starting in (ts_t-120, ts_t]; dens[t] = #bars starting in (ts_t-30, ts_t]."""
    n = len(ts)
    hh = np.empty(n)
    dens = np.empty(n)
    for t in range(n):
        m = h[t]
        cnt = 0
        j = t
        while j >= 0 and ts[j] > ts[t] - 120:
            if h[j] > m:
                m = h[j]
            if ts[j] > ts[t] - 30:
                cnt += 1
            j -= 1
        hh[t] = m
        dens[t] = cnt
    return hh, dens


@nb.njit
def vol_ratio(v):
    n = len(v)
    out = np.full(n, np.nan)
    for t in range(20, n):
        s = 0.0
        for j in range(t - 20, t):
            s += v[j]
        m = s / 20.0
        if m > 0:
            out[t] = v[t] / m
        elif v[t] > 0:
            out[t] = np.inf
    return out


@nb.njit
def trend(e10, e25, e60):
    n = len(e10)
    stack = np.zeros(n, np.bool_)
    persist = np.zeros(n, np.bool_)
    for b in range(n):
        stack[b] = (e10[b] < e25[b]) and (e25[b] < e60[b])
    for b in range(30, n):
        cnt = 0
        for j in range(b - 29, b + 1):
            if stack[j]:
                cnt += 1
        persist[b] = (cnt >= 24) and (e60[b] < e60[b - 30])        # 24/30 = 80%
    return stack, persist


@nb.njit
def support_break(low, N, tol):
    n = len(low)
    S = np.full(n, np.nan)
    brk = np.zeros(n, np.bool_)
    for t in range(N, n):
        m = np.inf
        for j in range(t - N, t - 1):        # t-N .. t-2 inclusive
            if low[j] < m:
                m = low[j]
        S[t] = m
        thr = m * (1 - tol)
        if low[t - 1] >= thr and low[t] < thr and low[t] < low[t - 1]:
            brk[t] = True
    return S, brk


@nb.njit
def swings(low, p):
    n = len(low)
    sw = np.zeros(n, np.bool_)
    for j in range(p, n - p):
        ok = True
        for q in range(1, p + 1):
            if not (low[j] < low[j - q]) or not (low[j] <= low[j + q]):
                ok = False
                break
        sw[j] = ok
    return sw


@nb.njit
def level_counts(ks, low, S, sw, N, p):
    M = len(ks)
    touch = np.zeros(M, np.int32)
    level = np.zeros(M, np.int32)
    for q in range(M):
        t = ks[q]
        s = S[t]
        pts = np.empty(N + 2)
        n = 0
        covered = False
        for j in range(t - N, t - 1):          # j in [t-N, t-2]
            if j + p <= t - 1 and sw[j]:
                pts[n] = low[j]
                n += 1
                if low[j] == s:
                    covered = True             # S's location is a confirmed swing
        if not covered:
            pts[n] = s
            n += 1
        a = np.sort(pts[:n])
        tc = 0
        for i in range(n):
            if a[i] <= s * 1.003:
                tc += 1
        band = a[a <= s * 1.01]
        lc = 0
        for i in range(len(band)):
            if i == 0 or band[i] > band[i - 1] * 1.0015:
                lc += 1
        touch[q] = tc
        level[q] = lc
    return touch, level


@nb.njit
def outcomes(ts, day, mins, sess, o, h, l, c, cost):
    """BRIEF: window = bars starting in [ts_i+1, ts_i+30], same day; skip if signal start > 19:29;
    next bar must start <= 2 min after signal close (ts_i+1). E0: entry o[e], hits from e.
    E2: entry c[e], hits from bars after e. Fields: r1, r1net, hit, hits, race, mfe, mae."""
    n = len(o)
    valid = np.zeros((2, n), np.bool_)
    f = np.full((7, 2, n), np.nan)
    last = np.full(n, -1, np.int64)
    for i in range(n):
        if mins[i] > 19 * 60 + 29:
            continue
        if i + 1 >= n:
            continue
        e = i + 1
        if day[e] != day[i] or ts[e] - (ts[i] + 1) > 2:
            continue
        # window members
        z = -1
        j = e
        while j < n and day[j] == day[i] and ts[j] <= ts[i] + 30:
            z = j
            j += 1
        if z < 0:
            continue
        last[i] = z
        cs = cost[sess[i]]
        for m in range(2):
            en = o[e] if m == 0 else c[e]
            start = e if m == 0 else e + 1
            if start > z:
                continue
            tgt = en * 1.01
            tgs = en * 1.011
            stp = en * 0.99
            first_up = -1
            first_dn = -1
            strict = False
            mx = -1e300
            mn = 1e300
            for j in range(start, z + 1):
                if first_up == -1 and h[j] >= tgt:
                    first_up = j
                if first_dn == -1 and l[j] <= stp:
                    first_dn = j
                if h[j] >= tgs:
                    strict = True
                mx = max(mx, h[j])
                mn = min(mn, l[j])
            valid[m, i] = True
            hit = first_up != -1
            ret = c[z] / en - 1.0
            f[0, m, i] = 0.01 if hit else ret
            f[1, m, i] = 0.01 - cs if hit else ret - cs - cs
            f[2, m, i] = 1.0 if hit else 0.0
            f[3, m, i] = 1.0 if strict else 0.0
            if first_up != -1 or first_dn != -1:
                if first_dn == -1 or (first_up != -1 and first_up < first_dn):
                    f[4, m, i] = 1.0
                elif first_up == -1 or first_dn < first_up:
                    f[4, m, i] = 0.0
                else:
                    f[4, m, i] = 0.5
            f[5, m, i] = mx / en - 1.0
            f[6, m, i] = mn / en - 1.0
    return valid, f, last


@nb.njit
def r2_stop(sig, stops, o, h, l, c, last, valid2):
    """BRIEF R2 (E2): entry c[sig+1], from the bar after: open below stop -> exit at open; low < stop -> stop;
    +1% -> target; both in one bar -> stop. Time exit at window close. Entry already below stop -> 0."""
    M = len(sig)
    r2 = np.full(M, np.nan)
    amb = np.zeros(M, np.bool_)
    for q in range(M):
        i = sig[q]
        if not valid2[i]:
            continue
        z = last[i]
        en = c[i + 1]
        st = stops[q]
        if en < st:
            r2[q] = 0.0
            continue
        res = np.nan
        for j in range(i + 2, z + 1):
            if o[j] < st:
                res = o[j] / en - 1.0
                break
            down = l[j] < st
            up = h[j] >= en * 1.01
            if down:
                if up:
                    amb[q] = True
                res = st / en - 1.0
                break
            if up:
                res = 0.01
                break
        if res != res:
            res = c[z] / en - 1.0
        r2[q] = res
    return r2, amb


def avg_rank_pct(x):
    """pct rank with average ties among non-NaN, NaN stays NaN."""
    out = np.full(len(x), np.nan)
    ok = ~np.isnan(x)
    v = x[ok]
    if len(v) == 0:
        return out
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    ranks = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    out[np.nonzero(ok)[0]] = ranks / len(v)
    return out


def per_symbol(d):
    idx = d.index.tz_convert(ET)
    o, h, l, c, v = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close", "Volume"))
    ts = idx.tz_convert("UTC").tz_localize(None).values.astype("datetime64[m]").astype(np.int64)
    mins = np.asarray(idx.hour * 60 + idx.minute, dtype=np.int64)
    day = np.asarray(idx.year * 10000 + idx.month * 100 + idx.day, dtype=np.int64)
    sess = np.where(mins < 9 * 60 + 30, 0, np.where(mins < 11 * 60, 1, np.where(mins < 16 * 60, 2, 3))).astype(np.int64)
    tradable = ((mins >= 4 * 60) & (mins < 11 * 60)) | ((mins >= 19 * 60) & (mins < 19 * 60 + 30))
    rng = h - l
    single = (o == h) & (h == l) & (l == c)
    prints_ok = (d["Trades"].to_numpy() >= 5) if "Trades" in d.columns else ~single
    sane = prints_ok & (v * c >= 25000.0) & (rng >= 0.002 * c)
    e10, e25, e60 = ema(c, 10), ema(c, 25), ema(c, 60)
    stack, persist = trend(e10, e25, e60)
    atr = atr14(h, l, c)
    hh, dens = time_feats(ts, h)
    volr = vol_ratio(v)
    valid, f, last = outcomes(ts, day, mins, sess, o, h, l, c, COST)
    # volatility-matched baseline (same day, session, ATR% tercile within that group; mean over valid bars)
    atrp = atr / c
    ter = np.zeros(len(o), np.int64)
    g = pd.Series(np.arange(len(o))).groupby([day, sess]).indices
    for key, ix in g.items():
        pct = avg_rank_pct(atrp[ix])
        tt = np.where(np.isnan(pct), 0, np.minimum(np.floor(pct * 3), 2)).astype(np.int64)
        ter[ix] = tt
    bv = {}
    gkey = day * 100 + sess * 10 + ter
    for m, tag in ((0, "e0"), (1, "e2")):
        for k, name in ((0, "r1"), (1, "r1net"), (2, "hit"), (4, "race")):
            x = np.where(valid[m], f[k, m], np.nan)
            s = pd.Series(x).groupby(gkey)
            num = s.sum(min_count=1)       # sum over non-NaN
            den = s.count()
            mean = (num / den).reindex(gkey).to_numpy()
            bv[f"bv_{name}_{tag}"] = mean
    return dict(o=o, h=h, l=l, c=c, v=v, ts=ts, mins=mins, day=day, sess=sess, tradable=tradable, sane=sane,
                stack=stack, persist=persist, atr=atr, hh=hh, dens=dens, volr=volr, valid=valid, f=f, last=last,
                rng=rng, bv=bv)


FIELDS = ["r1", "r1net", "hit", "hits", "race", "mfe", "mae"]


def outcome_cols(a, tt):
    cols = {}
    for m, tag in ((0, "e0"), (1, "e2")):
        for k, name in enumerate(FIELDS):
            cols[f"{name}_{tag}"] = a["f"][k, m][tt]
    cols["valid"] = a["valid"][0][tt] & a["valid"][1][tt]
    for key, arr in a["bv"].items():
        cols[key] = arr[tt]
    return cols


def tables(sym, sid, a, N, tol, lo, hi, filt=True):
    low = a["l"]
    n = len(low)
    S, brk = support_break(low, N, tol)
    ks = np.nonzero(brk)[0].astype(np.int64)
    if len(ks) == 0:
        return None, None
    b = ks - 1
    hh = a["hh"][ks]
    base = dict(sid=sid, sym=sym, N=N, tol=tol, k=ks, S=S[ks], stack=a["stack"][b], persist=a["persist"][b],
                dd=(hh - low[ks]) / hh, volr=np.nan_to_num(a["volr"][ks], nan=0.0))
    for p in PS:
        sw = swings(low, p)
        tc, lc = level_counts(ks, low, S, sw, N, p)
        base[f"touch{p}"] = tc
        base[f"level{p}"] = lc
    thr = S[ks] * (1 - tol)
    # pattern 1: signal bar = break bar
    A = pd.DataFrame(base)
    atrb = a["atr"][b]
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = (a["h"][ks] - low[ks]) / atrb
    A["openok"] = a["o"][ks] >= thr
    rng = a["rng"][ks]
    A["wick"] = np.where(rng > 0, (np.minimum(a["o"][ks], a["c"][ks]) - low[ks]) / np.where(rng > 0, rng, 1), -1.0)
    A["cpos"] = np.where(rng > 0, (a["c"][ks] - low[ks]) / np.where(rng > 0, rng, 1), -1.0)
    A["rngatr"] = np.where(np.isnan(rr), 0.0, rr)
    A["above"] = a["c"][ks] >= S[ks]
    A["sane"] = a["sane"][ks] & (a["dens"][ks] >= 15)
    A["t"] = ks
    A["ts"] = a["ts"][ks]
    A["day"] = a["day"][ks]
    A["sess"] = a["sess"][ks]
    A["tradable"] = a["tradable"][ks]
    for kk, vv in outcome_cols(a, ks).items():
        A[kk] = vv
    r2, amb = r2_stop(ks, low[ks].astype(float), a["o"], a["h"], a["l"], a["c"], a["last"], a["valid"][1])
    A["r2_e2"] = r2
    A["amb_e2"] = amb
    # pattern 2: break bar k (red, same day as t), signal t = k+1
    okB = ks + 1 < n
    kB = ks[okB]
    tB = kB + 1
    B = pd.DataFrame({kk: (vv[okB] if isinstance(vv, np.ndarray) else vv) for kk, vv in base.items()})
    red = (a["c"][kB] < a["o"][kB]) & (a["day"][tB] == a["day"][kB])
    thrB = thr[okB]
    B["below"] = a["c"][kB] < thrB
    green = a["c"][tB] > a["o"][tB]
    rk = a["h"][kB] - low[kB]
    B["eng_basic"] = green & (a["c"][tB] >= a["o"][kB] - 0.1 * rk)
    B["eng_body"] = green & (a["c"][tB] >= a["o"][kB])
    B["eng_full"] = green & (a["c"][tB] >= a["h"][kB])
    B["eng_reclaim"] = green & (a["c"][tB] >= S[kB])
    B["sane"] = a["sane"][kB] & a["sane"][tB] & (a["dens"][tB] >= 15)
    B["t"] = tB
    B["ts"] = a["ts"][tB]
    B["day"] = a["day"][tB]
    B["sess"] = a["sess"][tB]
    B["tradable"] = a["tradable"][tB]
    for kk, vv in outcome_cols(a, tB).items():
        B[kk] = vv
    r2, amb = r2_stop(tB, np.minimum(low[kB], low[tB]).astype(float), a["o"], a["h"], a["l"], a["c"], a["last"],
                      a["valid"][1])
    B["r2_e2"] = r2
    B["amb_e2"] = amb
    B = B[red]
    if not filt:
        return A.reset_index(drop=True), B.reset_index(drop=True)
    A = A[A.valid & (A.day >= lo) & (A.day <= hi)]
    B = B[B.valid & (B.day >= lo) & (B.day <= hi)]
    return A.reset_index(drop=True), B.reset_index(drop=True)


def main(src, a_day, b_day):
    sys.path.insert(0, os.path.abspath(os.path.join(BASE, "../../../..")))
    import watchlist
    syms = sorted(s for s in watchlist.AI_SUBSECTOR if "." not in s)
    lo, hi = int(a_day.replace("-", "")), int(b_day.replace("-", ""))
    out = f"{BASE}/audit/claude_skeptic/ref_{src}"
    os.makedirs(out, exist_ok=True)
    parts = {}
    mins_rows = []
    t0 = time.time()
    for sid, sym in enumerate(syms):
        d = load(src, sym)
        if d is None or len(d) < 500:
            continue
        a = per_symbol(d)
        inper = (a["day"] >= lo) & (a["day"] <= hi)
        vv = a["valid"][0] & a["valid"][1] & inper
        mins_rows.append(pd.DataFrame({"ts": a["ts"][vv], "sid": sid,
                                       **{f"{nm}_{tg}": a["f"][k, m][vv] for (nm, k) in (("r1", 0), ("r1net", 1), ("hit", 2), ("race", 4))
                                          for (tg, m) in (("e0", 0), ("e2", 1))}}))
        for N in NS:
            for tol in TOLS:
                A, B = tables(sym, sid, a, N, tol, lo, hi)
                for tag, X in (("A", A), ("B", B)):
                    if X is not None and len(X):
                        parts.setdefault(f"{tag}_{N}_{tol}", []).append(X)
    mm = pd.concat(mins_rows, ignore_index=True)
    for key, xs in parts.items():
        X = pd.concat(xs, ignore_index=True).sort_values(["sid", "ts"], kind="stable").reset_index(drop=True)
        # same-minute other-symbol mean, computed by explicit exclusion of own sid
        sub = mm[mm.ts.isin(np.unique(X.ts))]
        for f in ("r1_e2", "r1net_e2", "race_e2", "hit_e2", "r1_e0", "hit_e0"):
            gs = sub.groupby("ts")[f]
            tot_s, tot_n = gs.sum(), gs.count()
            own = sub.set_index(["ts", "sid"])[f]
            key2 = pd.MultiIndex.from_arrays([X.ts.to_numpy(), X.sid.to_numpy()])
            ov = own.reindex(key2).to_numpy()
            s_all = tot_s.reindex(X.ts.to_numpy()).to_numpy()
            n_all = tot_n.reindex(X.ts.to_numpy()).to_numpy()
            has = ~np.isnan(ov)
            num = s_all - np.where(has, ov, 0.0)
            den = n_all - has
            with np.errstate(invalid="ignore", divide="ignore"):
                X[f"bt_{f}"] = np.where(den > 0, num / den, np.nan)
        X.to_parquet(f"{out}/{key}.parquet")
    print("ref done", len(parts), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
