"""Independent recomputation of momentum candidates (written without reading engine.py)."""
import sys, os, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 6))
sys.path.insert(0, ROOT)
import watchlist
from desk.config import SEMIS_SECTORS

C = os.path.join(ROOT, "data/derived/analysis/momentum_variants_20261120/cache/")
END = pd.Timestamp("2026-11-18")
AC = pd.read_parquet(C + "px_adjclose.parquet")
CL = pd.read_parquet(C + "px_close.parquet")
OP = pd.read_parquet(C + "px_open.parquet")
VO = pd.read_parquet(C + "px_volume.parquet")
sess = AC.index[AC["QQQ"].notna() & (AC.index <= END)]
AC, CL, OP, VO = [d.reindex(sess) for d in (AC, CL, OP, VO)]
AO = OP * AC / CL
META = pd.read_csv(C + "meta.csv", index_col=0)
H = pd.read_csv(os.path.join(ROOT, "data/derived/analysis/post_earnings_risers_20261120/results/held_days.csv"))
H["date"] = pd.to_datetime(H["date"])
SNAPDATES = np.array(sorted(H.date.unique()))
SNAP = {(f, d): g.set_index("ticker")["weight"] for (f, d), g in H.groupby(["fund", "date"])}

cols = list(AC.columns)
A = AC.values
AOv = AO.values
DVMIN = int(os.environ.get("DVMIN", "1"))
DV = (CL * VO).rolling(20, min_periods=DVMIN).mean().shift(1).values  # avg over k-20..k-1
LR = np.log(AC / AC.shift(1)).values
N = len(sess)
ci = {c: i for i, c in enumerate(cols)}
meta = META.reindex(cols)
in_wide = meta.in_wide.fillna(False).astype(bool).values
in_fund = meta.in_fund.fillna(False).astype(bool).values
in_ai = meta.in_ai.fillna(False).astype(bool).values
tech = ((meta.sector == "Technology") | (meta.etf == "SOXX")).fillna(False).values


def sector_of(t):
    s = watchlist.resolve_sector(t, default=None)
    if s:
        return s
    if t in META.index:
        return "AI 반도체" if META.loc[t, "etf"] == "SOXX" else META.loc[t, "sector"]
    return t


SECT = [sector_of(c) for c in cols]
ISSEMI = np.array([s in SEMIS_SECTORS for s in SECT])

SNAP_STRICT = os.environ.get("SNAP", "strict") == "strict"


def snapshot(fund, k):
    d = sess[k]
    ds = SNAPDATES[SNAPDATES < d] if SNAP_STRICT else SNAPDATES[SNAPDATES <= d]
    if len(ds) == 0:
        return None
    if fund == "FUNDS":
        a = SNAP.get(("TIME", ds[-1]), pd.Series(dtype=float))
        b = SNAP.get(("KoAct", ds[-1]), pd.Series(dtype=float))
        return a.add(b, fill_value=0)
    return SNAP.get((fund, ds[-1]))


def universe(name, k):
    if name in ("TIME", "KoAct", "FUNDS"):
        s = snapshot(name, k)
        m = np.zeros(len(cols), bool)
        if s is None:
            return m
        for t in s.index:
            if t in ci:
                m[ci[t]] = True
        return m
    if name == "AI":
        return in_ai.copy()
    for pre, base in (("WIDETECH", in_wide & tech), ("WIDE", in_wide), ("ALL", in_wide | in_fund | in_ai)):
        if name.startswith(pre):
            X = float(name[len(pre):])
            return base & (np.nan_to_num(DV[k]) >= X * 1e6)
    raise ValueError(name)


def ret(L, skip=0):
    out = np.full_like(A, np.nan)
    # out[k] = A[k-1-skip] / A[k-1-L] - 1  (window k-1-L .. k-1, last `skip` sessions skipped)
    if os.environ.get("SKIPMODE") == "shift":
        out[L + skip + 1:] = A[L:N - skip - 1] / A[:N - L - skip - 1] - 1
        return out
    out[L + 1:] = A[L - skip:N - 1 - skip] / A[:N - L - 1] - 1
    return out


def residual(L=60, minv=54):
    q = LR[:, ci["QQQ"]]
    out = np.full_like(A, np.nan)
    for k in range(L + 1, N):
        R = LR[k - L:k]
        qq = q[k - L:k]
        for j in range(R.shape[1]):
            r = R[:, j]
            v = ~np.isnan(r) & ~np.isnan(qq)
            if v.sum() < minv:
                continue
            x = qq[v]
            y = r[v]
            vx = np.var(x, ddof=1)
            if vx <= 0:
                continue
            beta = np.cov(y, x, ddof=1)[0, 1] / vx
            out[k, j] = y.sum() - beta * x.sum()
    return out


def ivol(k, idx):
    R = LR[max(0, k - 60):k][:, idx]
    return np.nanstd(R, axis=0, ddof=1)


CAPMODE = os.environ.get("CAPMODE", "fill")


def apply_caps(w, idx):
    if CAPMODE == "fill":
        return fill_caps(w, idx)
    return scale_caps(w, idx)


def fill_caps(raw, idx, stock=0.25, sector_cap=0.60, semis=0.50):
    """Progressive proportional fill: all names grow at rate raw_i until stock/sector/semis caps bind."""
    sect = np.array([SECT[i] for i in idx], dtype=object)
    semi = ISSEMI[idx]
    alloc = np.zeros(len(raw))
    target = raw.sum()
    active = raw > 0
    for _ in range(4 * len(raw) + 8):
        used = alloc.sum()
        if target - used <= 1e-12:
            break
        gs = {g: alloc[sect == g].sum() for g in set(sect)}
        ins = alloc[semi].sum()
        active &= np.array([stock - alloc[i] > 1e-12 and sector_cap - gs[sect[i]] > 1e-12
                            and not (semi[i] and semis - ins <= 1e-12) for i in range(len(raw))])
        if not active.any():
            break
        steps = [(target - used) / raw[active].sum()]
        steps += list((stock - alloc[active]) / raw[active])
        for g in set(sect[active]):
            m = active & (sect == g)
            steps.append((sector_cap - gs[g]) / raw[m].sum())
        sr = raw[active & semi].sum()
        if sr > 0:
            steps.append((semis - ins) / sr)
        step = max(0.0, min(steps))
        alloc[active] += step * raw[active]
    return alloc


def scale_caps(w, idx):
    w = w.copy()
    sect = np.array([SECT[i] for i in idx], dtype=object)
    semi = ISSEMI[idx]
    for _ in range(500):
        w = np.minimum(w, 0.25)
        for s in set(sect):
            m = sect == s
            tot = w[m].sum()
            if tot > 0.60:
                w[m] *= 0.60 / tot
        tot = w[semi].sum()
        if tot > 0.50:
            w[semi] *= 0.50 / tot
        excess = 1 - w.sum()
        if excess <= 1e-12:
            break
        frozen = w >= 0.25 - 1e-12
        for s in set(sect):
            m = sect == s
            if w[m].sum() >= 0.60 - 1e-12:
                frozen |= m
        if w[semi].sum() >= 0.50 - 1e-12:
            frozen |= semi
        free = ~frozen & (w > 0)
        if not free.any():
            break  # rest stays cash
        w[free] += excess * w[free] / w[free].sum()
    return w


def run(score, uni, n, buffer, weight, caps, freq="W", dial=None, start=130):
    shares = np.zeros(len(cols))
    cash = 1.0
    basket = []
    nav_open = np.full(N, np.nan)
    nav_close = np.full(N, np.nan)
    lastA = pd.DataFrame(A).ffill().values
    modes = {}
    picks = {}
    for k in range(start, N):
        d = sess[k]
        po = np.where(np.isnan(AOv[k]), lastA[k - 1], AOv[k])
        po = np.nan_to_num(po)
        nav = cash + (shares * po).sum()
        nav_open[k] = nav
        if freq == "W":
            reb = k == start or d.isocalendar()[:2] != sess[k - 1].isocalendar()[:2]
        else:
            reb = k == start or d.month != sess[k - 1].month
        if reb:
            sc = score[k]
            if dial is not None:
                D, alt = dial
                if not np.isnan(D[k]) and D[k] < 0:
                    sc = alt[k]
                    modes[d] = "alt"
                else:
                    modes[d] = "base"
            u = universe(uni, k) & ~np.isnan(A[k - 1]) & ~np.isnan(AOv[k]) & ~np.isnan(sc)
            if dial is not None and os.environ.get("ALTBASE") == "1":
                u &= ~np.isnan(score[k])  # alt mode ranks only names that also have the base score
            idx = np.where(u)[0]
            order = idx[np.argsort(-sc[idx], kind="stable")]
            rank = {j: r for r, j in enumerate(order)}
            keep = [j for j in basket if j in rank and rank[j] < n + buffer]
            keep = sorted(keep, key=lambda j: rank[j])[:n]
            new = keep + [j for j in order if j not in keep][: n - len(keep)]
            new = np.array(sorted(new, key=lambda j: rank[j]), dtype=int)
            basket = list(new)
            picks[d] = [cols[j] for j in new]
            shares[:] = 0
            cash = nav
            if len(new):
                if weight == "equal":
                    w = np.ones(len(new))
                elif weight == "ivol":
                    s = ivol(k, new)
                    w = np.where(np.isfinite(1 / s), 1 / s, 0)
                elif weight == "time":
                    snap = snapshot("TIME", k)
                    w = np.array([snap.get(cols[j], 0.0) for j in new], float)
                w = w / w.sum()
                if caps:
                    w = apply_caps(w, new)
                shares[new] = nav * w / AOv[k, new]
                cash = nav * (1 - w.sum())
        pc = np.nan_to_num(lastA[k])
        nav_close[k] = cash + (shares * pc).sum()
    return pd.Series(nav_open, sess), pd.Series(nav_close, sess), modes, picks


WINDOWS = {"H1": ("2026-01-02", "2026-06-30"), "H2": ("2026-07-01", "2026-11-18"),
           "Y2026": ("2026-01-02", "2026-11-18"), "RECENT": ("2026-09-01", "2026-11-18"),
           "Y2025H2": ("2025-07-01", "2025-12-31")}


def window(no, nc, a, b):
    s0 = sess[sess >= pd.Timestamp(a)][0]
    s1 = sess[sess <= pd.Timestamp(b)][-1]
    o = no.loc[s0]
    path = np.concatenate([[o], nc.loc[s0:s1].values])
    mdd = (path / np.maximum.accumulate(path) - 1).min()
    return nc.loc[s1] / o - 1, mdd


def summary(no, nc):
    out = {}
    for kname, (a, b) in WINDOWS.items():
        r, mdd = window(no, nc, a, b)
        out[kname] = round(float(r), 4)
        out[kname + "_MDD"] = round(float(mdd), 4)
    return out


def icdial(L=60, h=20, K=10, du="ALL50"):
    base = ret(L)
    fwd = np.full_like(A, np.nan)
    for m in range(1, N - h + 1):
        fwd[m] = A[m - 1 + h] / A[m - 1] - 1
    ic = np.full(N, np.nan)
    for m in range(L + 1, N - h + 1):
        u = universe(du, m) & ~np.isnan(base[m]) & ~np.isnan(fwd[m])
        if u.sum() < 10:
            continue
        ic[m] = np.corrcoef(rankdata(base[m, u]), rankdata(fwd[m, u]))[0, 1]
    D = np.full(N, np.nan)
    for k in range(h + K, N):
        v = ic[k - h - K + 1:k - h + 1]  # ic[k-29..k-20]
        v = v[~np.isnan(v)]
        if len(v) >= 5:
            D[k] = v.mean()
    return D


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    sv = os.environ.get("START", "130")
    start = int(sess.get_loc(pd.Timestamp(sv))) if "-" in sv else int(sv)
    r60 = ret(60)
    r120 = ret(120)
    jobs = {
        "BASE": lambda: run(r120, "TIME", 7, 5, "time", True, start=start),
        "WIDETECH50|lb60|M|n3|b5|ivol|nocap": lambda: run(r60, "WIDETECH50", 3, 5, "ivol", False, "M", start=start),
        "TIME|lb120|W|n3|b5|time|nocap": lambda: run(r120, "TIME", 3, 5, "time", False, start=start),
        "WIDETECH50|lb60s0|n5b0|equal|caps0": lambda: run(r60, "WIDETECH50", 5, 0, "equal", False, start=start),
        "F1:u=AI,n=7,lb=60,h=20,K=10,alt=rev5,du=ALL50": lambda: run(r60, "AI", 7, 5, "equal", True, dial=(icdial(), -ret(5)), start=start),
        "RES60|AI|n5|equal": lambda: run(residual(60), "AI", 5, 5, "equal", True, start=start),
        "FUNDS|lb60|W|n3|b10|equal|nocap": lambda: run(r60, "FUNDS", 3, 10, "equal", False, start=start),
        "TIME|lb120s5|n5b5|time|caps0": lambda: run(ret(120, 5), "TIME", 5, 5, "time", False, start=start),
        "WIDETECH50|r60|n7|nocap|equal": lambda: run(r60, "WIDETECH50", 7, 5, "equal", False, start=start),
    }
    res = {}
    for name, f in jobs.items():
        if mode != "all" and mode != name:
            continue
        no, nc, modes, picks = f()
        s = summary(no, nc)
        if modes:
            m26 = {d: v for d, v in modes.items() if d.year == 2026}
            s["alt_H1"] = f"{sum(v == 'alt' for d, v in m26.items() if d.month <= 6)}/{sum(d.month <= 6 for d in m26)}"
            s["alt_H2"] = f"{sum(v == 'alt' for d, v in m26.items() if d.month > 6)}/{sum(d.month > 6 for d in m26)}"
        res[name] = s
        print(name, json.dumps(s), flush=True)
    tag = f"{'strict' if SNAP_STRICT else 'le'}_s{start}"
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
