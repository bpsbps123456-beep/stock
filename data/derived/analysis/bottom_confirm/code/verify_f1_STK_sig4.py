"""Adversarial verification of bottom_confirm F1 rule 'sig4' on the 82-stock set (STK) (BRIEF.md, 2026-10-26).

Rule under test: up[t] >= 4 * sigma20[t]
  up[t]      = C[t] / Ldd[t] - 1, Ldd = lowest close since the latest 60-session closing high (ties -> later day)
  sigma20[t] = sample std (ddof=1) of daily close-to-close returns over the 20 sessions ending at t
               (= C.pct_change().rolling(20).std())
Signals count only on in_dd days (min dd60 over the last 20 sessions <= -20%), eval window 2025-09-01..2026-10-23,
first signal = more than 20 sessions after the previously adopted signal of the same stock.
Main period for STK = calendar 2026; registered L1 needs 2026H1 and 2026H2 >= 70% (n >= 5, >= 2 blocks),
2026 >= 70%, 2025 (Sep-Dec) >= 60%.

This file does NOT import bc_lib or f1_rebound. Everything is rebuilt from the raw CSVs with numpy.
The session calendar is taken from the data itself (rows where at least one of the 82 US stocks has a close)
and only cross-checked against exchange_calendars XNYS if that package is installed.
Run: cd data/derived/analysis/bottom_confirm && python3 -W ignore code/verify_f1_STK_sig4.py
Output: results/verify_f1_STK_sig4.txt
"""
from __future__ import annotations

import io
import pathlib
import re

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(__file__).resolve().parents[5]
RES = HERE / "results"
OUT = RES / "verify_f1_STK_sig4.txt"

OPEN_CSV = ROOT / "data/derived/desk/prices_open.csv"
CLOSE_CSV = ROOT / "data/derived/desk/prices_close.csv"
FAM_SIG = RES / "f1_STK_signals.csv"
FAM_SUM = RES / "f1_STK_summary.csv"
FAM_PH_SUM = RES / "f1_STK_posthoc_summary.csv"
FAM_CODE = HERE / "code/f1_rebound.py"
LIB_CODE = HERE / "code/bc_lib.py"

RULE = "sig4"
MULT = 4.0
SIGWIN = 20
THR = 0.20
COOL = 20
EVAL_START = pd.Timestamp("2025-09-01")
LAST = pd.Timestamp("2026-10-23")
PERIODS = [("2025-09~2026", "2025-09-01", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
           ("2025", "2025-09-01", "2025-12-31")]
MAIN = "2026"
BUCKET_EDGES = [0.03, 0.08, 0.15]          # 0-3, 3-8, 8-15, 15%+
BUCKET_NAMES = ["0-3%", "3-8%", "8-15%", "15%+"]
N_NULL = 2000
N_BOOT = 5000
SEED = 7_2026_1026

buf = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    buf.write(s + "\n")


def f3(x):
    return "  nan" if x is None or not np.isfinite(x) else f"{x:.3f}"


# ------------------------------------------------------------------ raw data (independent loader)
def load():
    o = pd.read_csv(OPEN_CSV, index_col=0, parse_dates=True)
    c = pd.read_csv(CLOSE_CSV, index_col=0, parse_dates=True)
    assert list(o.columns) == list(c.columns) and o.index.equals(c.index)
    drop = [x for x in c.columns if x.endswith(".KS") or x.endswith("=X") or x == "QQQ"]
    c, o = c.drop(columns=drop), o.drop(columns=drop)
    c = c.sort_index()
    c = c[c.notna().any(axis=1)]                 # US sessions = rows where any US stock has a close
    c = c.loc[:LAST]
    o = o.loc[c.index]
    assert c.shape[1] == 82, c.shape
    assert c.index[0] == pd.Timestamp("2025-06-02") and c.index[-1] == LAST
    cal_note = "exchange_calendars not available"
    try:
        import exchange_calendars as xc  # cross-check only
        s = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range("2025-06-02", "2026-10-23"))
        s = s.tz_localize(None) if s.tz is not None else s
        cal_note = f"data calendar == XNYS sessions: {bool(s.equals(c.index))} ({len(c)} sessions)"
    except Exception:  # pragma: no cover
        pass
    # no interior gaps (checked, so no fill rule is needed)
    for k in c.columns:
        fv = c[k].first_valid_index()
        assert c[k].loc[fv:].notna().all() and o[k].loc[fv:].notna().all(), k
    return c.index, c.to_numpy(float), o.to_numpy(float), list(c.columns), cal_note


# ------------------------------------------------------------------ features (strictly causal)
def features(C: np.ndarray, sigwin: int = SIGWIN):
    """Returns L (Ldd), up, dd60, sigma, lowpos. Values at t use only C[..t]."""
    N, K = C.shape
    L = np.full((N, K), np.nan)
    up = np.full((N, K), np.nan)
    dd60 = np.full((N, K), np.nan)
    sg = np.full((N, K), np.nan)
    lowpos = np.full((N, K), -1, int)
    for k in range(K):
        c = C[:, k]
        for t in range(59, N):
            w = c[t - 59:t + 1]
            if not np.isfinite(w).all():
                continue
            mx = w.max()
            pk = t - 59 + int(np.flatnonzero(w == mx)[-1])          # latest tie
            seg = c[pk:t + 1]
            lo = seg.min()
            L[t, k] = lo
            lowpos[t, k] = pk + int(np.flatnonzero(seg == lo)[-1])
            up[t, k] = c[t] / lo - 1
            dd60[t, k] = c[t] / mx - 1
        r = np.full(N, np.nan)
        r[1:] = c[1:] / c[:-1] - 1
        for t in range(sigwin, N):
            rr = r[t - sigwin + 1:t + 1]
            if np.isfinite(rr).all():
                sg[t, k] = rr.std(ddof=1)
    return L, up, dd60, sg, lowpos


def in_dd_of(dd60: np.ndarray, thr: float) -> np.ndarray:
    N, K = dd60.shape
    out = np.zeros((N, K), bool)
    for t in range(N):
        w = dd60[max(0, t - 19):t + 1]
        fin = np.isfinite(w)
        m = np.where(fin, w, np.inf).min(0)
        out[t] = fin.any(0) & (m <= -thr)
    return out


def outcomes(C: np.ndarray, O: np.ndarray, L: np.ndarray):
    N, K = C.shape
    hold20 = np.full((N, K), np.nan)
    near5 = np.full((N, K), np.nan)
    r20 = np.full((N, K), np.nan)
    for t in range(N):
        if t + 20 <= N - 1:
            blk = C[t + 1:t + 21]
            ok = np.isfinite(blk).all(0) & np.isfinite(L[t])
            hold20[t] = np.where(ok, (blk.min(0) >= L[t] - 1e-9).astype(float), np.nan)
            ok2 = np.isfinite(C[t + 20]) & np.isfinite(O[t + 1])
            r20[t] = np.where(ok2, C[t + 20] / O[t + 1] - 1, np.nan)
        if t + 40 <= N - 1:
            blk = C[t + 1:t + 41]
            ok = np.isfinite(blk).all(0) & np.isfinite(O[t + 1])
            near5[t] = np.where(ok, (blk.min(0) >= 0.95 * O[t + 1]).astype(float), np.nan)
    win20 = np.where(np.isfinite(r20), (r20 > 0).astype(float), np.nan)
    return dict(hold20=hold20, near5=near5, win20=win20, r20=r20)


# ------------------------------------------------------------------ signals and evaluation
def first_signals(sig, indd, up, dates, cool=COOL, start=EVAL_START, end=LAST):
    ok = sig & indd & np.isfinite(up)
    dm = (dates >= start) & (dates <= end)
    ok &= dm[:, None]
    rows = []
    for k in range(ok.shape[1]):
        prev = -10 ** 9
        for p in np.flatnonzero(ok[:, k]):
            if p - prev <= cool:
                continue
            rows.append((k, p))
            prev = p
    return np.array(rows, int).reshape(-1, 2)


def period_mask(dates_of_sig, a, b):
    return (dates_of_sig >= pd.Timestamp(a)) & (dates_of_sig <= pd.Timestamp(b))


def boot_ci(v, lab, rng, B=N_BOOT):
    u, inv = np.unique(lab, return_inverse=True)
    if len(u) < 2:
        return (np.nan, np.nan)
    sums = np.bincount(inv, weights=v)
    cnts = np.bincount(inv).astype(float)
    pick = rng.integers(0, len(u), size=(B, len(u)))
    m = sums[pick].sum(1) / cnts[pick].sum(1)
    return tuple(np.quantile(m, [0.025, 0.975]))


def evaluate(fs, dates, oc, up, eval_shift=0, boot=False, rng=None):
    """fs: (k, p) array. eval_shift: evaluate outcomes at p+eval_shift (entry delay)."""
    out = {}
    if len(fs) == 0:
        return {p[0]: dict(n=0) for p in PERIODS}
    k, p = fs[:, 0], fs[:, 1]
    pe = p + eval_shift
    valid = pe <= len(dates) - 1
    k, p, pe = k[valid], p[valid], pe[valid]
    sd = dates[pe]
    for pname, a, b in PERIODS:
        m = period_mask(sd, a, b)
        r = dict(n=int(m.sum()))
        if m.sum():
            kk, pp = k[m], pe[m]
            r["clusters"] = int(len(np.unique(pp // COOL)))
            r["up_median"] = float(np.median(up[pp, kk]))
            for name in ("hold20", "near5", "win20", "r20"):
                v = oc[name][pp, kk]
                fin = np.isfinite(v)
                r[f"{name}_n"] = int(fin.sum())
                r[name] = float(v[fin].mean()) if fin.any() else np.nan
                if name == "hold20" and boot and fin.sum():
                    r["hold20_lo"], r["hold20_hi"] = boot_ci(v[fin], pp[fin] // COOL, rng)
                    r["hold20_clusters"] = int(len(np.unique(pp[fin] // COOL)))
        out[pname] = r
    return out


def l1_pass(ev) -> bool:
    def g(p, k, d=0.0):
        v = ev.get(p, {}).get(k, d)
        return d if v is None or (isinstance(v, float) and not np.isfinite(v)) else v
    halves = [g(h, "hold20_n") >= 5 and g(h, "clusters") >= 2 and g(h, "hold20") >= 0.70 for h in ("2026H1", "2026H2")]
    c25 = g("2025", "hold20_n") < 5 or g("2025", "hold20") >= 0.60
    return bool(all(halves) and c25 and g("2026", "hold20") >= 0.70)


def row_str(label, ev):
    def cell(p):
        r = ev.get(p, {})
        return f"{f3(r.get('hold20', np.nan))} ({r.get('hold20_n', 0):3d})"
    return (f"{label:34s} {cell('2026')}  {cell('2026H1')}  {cell('2026H2')}  {cell('2025')}  "
            f"upmed26={f3(ev.get('2026', {}).get('up_median', np.nan))}  L1={'PASS' if l1_pass(ev) else 'fail'}")


# ------------------------------------------------------------------ null
def null_draws(pool_k, pool_p, pool_b, need_by_bucket, hold, N, K, rng, cool=COOL, n_draws=N_NULL, strat_k=None,
               relax=False):
    """Random (stock, day) picks from the pool, matching counts per bucket (and per stock x bucket when strat_k is
    given), with the same within-stock cooldown. Returns array of hold20 means and the average number of picks per
    draw that had to be refilled (stock x bucket -> pooled bucket) or drawn with the cooldown relaxed (relax=True)."""
    res = np.empty(n_draws)
    short_tot = 0
    relaxed_tot = 0
    bsum, bcnt = {}, {}

    def acc(b, h):
        bsum[b] = bsum.get(b, 0.0) + h
        bcnt[b] = bcnt.get(b, 0) + 1
    buckets = sorted(need_by_bucket, key=lambda b: -need_by_bucket[b] / max(1, (pool_b == b).sum()))
    idx_by_b = {b: np.flatnonzero(pool_b == b) for b in buckets}
    if strat_k is not None:
        idx_by_kb = {}
        for (kk, b), n in strat_k.items():
            idx_by_kb[(kk, b)] = np.flatnonzero((pool_b == b) & (pool_k == kk))
    for d in range(n_draws):
        blocked = np.zeros((N + 2 * cool + 2, K), bool)
        vals = []
        if strat_k is not None:
            leftover = {b: 0 for b in buckets}
            order = sorted(strat_k, key=lambda kb: -strat_k[kb] / max(1, len(idx_by_kb[kb])))
            for kb in order:
                need = strat_k[kb]
                for i in rng.permutation(idx_by_kb[kb]):
                    if need == 0:
                        break
                    kk, pp = pool_k[i], pool_p[i]
                    if blocked[pp + cool, kk]:
                        continue
                    blocked[pp:pp + 2 * cool + 1, kk] = True     # marks |p - q| <= cool (offset by cool)
                    vals.append(hold[pp, kk])
                    acc(kb[1], hold[pp, kk])
                    need -= 1
                leftover[kb[1]] += need
            todo = {b: n for b, n in leftover.items() if n > 0}
            short_tot += sum(todo.values())
        else:
            todo = dict(need_by_bucket)
        for b in buckets:
            need = todo.get(b, 0)
            if need == 0:
                continue
            for i in rng.permutation(idx_by_b[b]):
                if need == 0:
                    break
                kk, pp = pool_k[i], pool_p[i]
                if blocked[pp + cool, kk]:
                    continue
                blocked[pp:pp + 2 * cool + 1, kk] = True
                vals.append(hold[pp, kk])
                acc(b, hold[pp, kk])
                need -= 1
            if need and not relax:
                raise RuntimeError(f"null draw could not fill bucket {b} ({need} short)")
            if need:                                    # relax the cooldown for the residual picks of this bucket only
                relaxed_tot += need
                for i in rng.choice(idx_by_b[b], size=need, replace=False):
                    vals.append(hold[pool_p[i], pool_k[i]])
                    acc(b, hold[pool_p[i], pool_k[i]])
        res[d] = np.mean(vals)
    bmean = {b: bsum[b] / bcnt[b] for b in bsum}
    return res, (short_tot + relaxed_tot) / n_draws, bmean


def rw_hold_prob(z: np.ndarray, n_paths: int = 200_000, seed: int = SEED + 9) -> np.ndarray:
    """Driftless Gaussian random walk, 20 daily steps of unit sigma: P(min cumulative sum >= -z).
    z = log(1 + up) / sigma20 = distance of today's close above Ldd in trailing daily sigmas (known at t)."""
    g = np.random.default_rng(seed).standard_normal((n_paths, 20)).cumsum(1).min(1)
    g.sort()
    return 1 - np.searchsorted(g, -np.asarray(z), side="left") / n_paths


# ------------------------------------------------------------------ logistic (IRLS, descriptive)
def logit(X, y, iters=50):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        pr = 1 / (1 + np.exp(-eta))
        W = pr * (1 - pr)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1])
        g = X.T @ (y - pr)
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return beta, se


# ------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(SEED)
    dates, C, O, cols, cal_note = load()
    N, K = C.shape
    say("=" * 100)
    say("Verification of bottom_confirm F1 'sig4' on STK (82 US AI stocks): up >= 4 * sigma20")
    say("Independent re-implementation from raw CSVs (no bc_lib / f1_rebound import).")
    say("=" * 100)
    say(f"Data: {CLOSE_CSV.relative_to(ROOT)} + prices_open.csv, 82 US stocks (QQQ, .KS, =X dropped), "
        f"{dates[0].date()}..{dates[-1].date()}, {N} sessions; {cal_note}")
    late = [(cols[k], dates[np.flatnonzero(np.isfinite(C[:, k]))[0]].date()) for k in range(K)
            if not np.isfinite(C[0, k])]
    say(f"Late listings (no fill, features start after 60 closes): {late}")

    L, up, dd60, sg, lowpos = features(C)
    indd = in_dd_of(dd60, THR)
    oc = outcomes(C, O, L)
    sig = np.isfinite(up) & np.isfinite(sg) & (up >= MULT * sg)
    fs = first_signals(sig, indd, up, dates)
    ev = evaluate(fs, dates, oc, up, boot=True, rng=rng)

    # =============================================================== 1. independent match
    say("\n" + "-" * 100)
    say("1. INDEPENDENT RE-IMPLEMENTATION vs FAMILY FILES")
    say("-" * 100)
    fam = pd.read_csv(FAM_SIG, encoding="utf-8-sig", parse_dates=["date"])
    fam = fam[fam.rule == RULE].copy()
    mine = pd.DataFrame(dict(date=dates[fs[:, 1]], series=[cols[k] for k in fs[:, 0]], pos=fs[:, 1],
                             up=up[fs[:, 1], fs[:, 0]], dd60=dd60[fs[:, 1], fs[:, 0]],
                             hold20=oc["hold20"][fs[:, 1], fs[:, 0]], near5=oc["near5"][fs[:, 1], fs[:, 0]],
                             win20=oc["win20"][fs[:, 1], fs[:, 0]], r20=oc["r20"][fs[:, 1], fs[:, 0]]))
    a = set(zip(fam.series, fam.date))
    b = set(zip(mine.series, mine.date))
    say(f"First signals: family {len(fam)}, mine {len(mine)}, common {len(a & b)}, "
        f"family-only {len(a - b)}, mine-only {len(b - a)}")
    for lab, s in (("family-only", sorted(a - b)), ("mine-only", sorted(b - a))):
        if s:
            say(f"  {lab}: " + ", ".join(f"{x}@{d.date()}" for x, d in s[:30]))
    mg = fam.merge(mine, on=["series", "date"], suffixes=("_f", "_m"))
    diffs = {}
    for col in ("pos", "up", "dd60", "hold20", "near5", "win20", "r20"):
        x, y = mg[f"{col}_f"].to_numpy(float), mg[f"{col}_m"].to_numpy(float)
        same_nan = (np.isnan(x) == np.isnan(y)).all()
        fin = np.isfinite(x) & np.isfinite(y)
        diffs[col] = (bool(same_nan), float(np.abs(x[fin] - y[fin]).max()) if fin.any() else 0.0)
    say("Per-signal field check on common signals (NaN pattern equal, max |diff|): "
        + ", ".join(f"{k}: {v[0]}/{v[1]:.2e}" for k, v in diffs.items()))

    fsum = pd.read_csv(FAM_SUM, encoding="utf-8-sig")
    fsum = fsum[fsum.rule == RULE].set_index("period")
    say(f"\n{'period':14s} {'n f/m':>9s} {'hold20_n f/m':>13s} {'hold20 fam':>10s} {'hold20 mine':>11s} "
        f"{'near5 f/m':>13s} {'win20 f/m':>13s} {'r20 f/m':>15s} {'upmed f/m':>13s} {'blocks f/m':>10s}")
    max_d = 0.0
    for pname, _, _ in PERIODS:
        r, fr_ = ev[pname], fsum.loc[pname]
        for kk in ("hold20", "near5", "win20", "r20", "up_median"):
            if np.isfinite(r.get(kk, np.nan)) and np.isfinite(fr_[kk]):
                max_d = max(max_d, abs(r[kk] - fr_[kk]))
        say(f"{pname:14s} {int(fr_.n):4d}/{r['n']:<4d} {int(fr_.hold20_n):6d}/{r['hold20_n']:<6d} "
            f"{fr_.hold20:10.4f} {r['hold20']:11.4f} {fr_.near5:6.3f}/{r['near5']:.3f} {fr_.win20:6.3f}/{r['win20']:.3f} "
            f"{fr_.r20:+7.4f}/{r['r20']:+.4f} {fr_.up_median:6.3f}/{r['up_median']:.3f} "
            f"{int(fr_.clusters):4d}/{r['clusters']:<4d}")
    match = (len(a - b) == 0 and len(b - a) == 0 and max_d < 1e-9
             and all(v[0] and v[1] < 1e-9 for v in diffs.values()))
    say(f"Max |diff| of period statistics: {max_d:.2e}  ->  INDEPENDENT MATCH: {match}")
    say(f"Registered L1 recomputed: {'PASS' if l1_pass(ev) else 'fail'}; reported L1=True, L2=False")
    m = ev[MAIN]
    say(f"Block-bootstrap 95% CI of hold20 ({MAIN}, 20-session calendar blocks, B={N_BOOT}): "
        f"[{m['hold20_lo']:.3f}, {m['hold20_hi']:.3f}] over {m['hold20_clusters']} blocks; "
        f"H1 [{ev['2026H1']['hold20_lo']:.3f}, {ev['2026H1']['hold20_hi']:.3f}] "
        f"({ev['2026H1']['hold20_clusters']} blocks), H2 [{ev['2026H2']['hold20_lo']:.3f}, "
        f"{ev['2026H2']['hold20_hi']:.3f}] ({ev['2026H2']['hold20_clusters']} blocks)")
    # concentration
    m26 = period_mask(mine.date, "2026-01-02", "2026-10-23") & mine.hold20.notna()
    g26 = mine[m26].copy()
    g26["blk"] = g26.pos // COOL
    bl = g26.groupby("blk").agg(n=("hold20", "size"), hold=("hold20", "mean"),
                                first=("date", "min"), last=("date", "max"))
    say("2026 signals by 20-session block (signals are cross-sectionally clustered):")
    for blk, r in bl.iterrows():
        say(f"  block {blk:3d} {r['first'].date()}..{r['last'].date()}  n={int(r.n):3d}  hold20={r.hold:.3f}")
    say(f"  largest 3 blocks hold {bl.n.nlargest(3).sum()}/{int(bl.n.sum())} signals; "
        f"block-mean (equal weight per block) hold20 = {bl.hold.mean():.3f}")

    # =============================================================== 2. look-ahead audit
    say("\n" + "-" * 100)
    say("2. LOOK-AHEAD AUDIT")
    say("-" * 100)
    # 2a: truncation test on my implementation (which matches the family's signals exactly)
    cut_rng = np.random.default_rng(SEED + 1)
    cuts = sorted(set(cut_rng.choice(np.arange(70, N - 1), size=25, replace=False).tolist()
                      + [int(p) for p in fs[:5, 1]]))
    bad = 0
    for T in cuts:
        Lt, upt, ddt, sgt, _ = features(C[:T + 1])
        it = in_dd_of(ddt, THR)
        st = np.isfinite(upt) & np.isfinite(sgt) & (upt >= MULT * sgt)
        same = (np.array_equal(st[T], sig[T]) and np.array_equal(it[T], indd[T])
                and np.allclose(np.nan_to_num(upt[T], nan=-9), np.nan_to_num(up[T], nan=-9), atol=1e-12)
                and np.allclose(np.nan_to_num(Lt[T], nan=-9), np.nan_to_num(L[T], nan=-9), atol=1e-12))
        bad += (not same)
    say(f"2a Truncation test: signal/in_dd/up/Ldd at day T recomputed from data ending at T, {len(cuts)} cut points, "
        f"mismatches = {bad}  (0 -> signal uses no data after T)")
    # 2b: static scan of the family code
    pat = {r"shift\(\s*-": "negative shift", r"center\s*=\s*True": "centered window", r"bfill|backfill": "back fill",
           r"\.iloc\[\s*[a-z_]*\s*\+\s*1": "forward index", r"rolling\([^)]*\)\.(?:max|min|mean|std)\(\)\.shift\(-": "fwd roll"}
    for f in (FAM_CODE, LIB_CODE):
        txt = f.read_text(encoding="utf-8")
        hits = []
        for rx, nm in pat.items():
            for mm in re.finditer(rx, txt):
                line = txt[:mm.start()].count("\n") + 1
                hits.append(f"{nm}@L{line}")
        say(f"2b Static scan {f.name}: {hits or 'no negative shifts / centered windows / back-fills'}")
    say("2c Manual reading: f1_rebound.build_rules uses fr['up'] (Ldd over [60d-peak..t]) and "
        "sigma20 = (C/C.shift(1)-1).rolling(20).std() (window ends at t); bc_lib.in_dd = dd60.rolling(20).min() "
        "(ends at t); first_signals uses only signal/in_dd/date; outcomes start at t+1 (hold20 = C[t+1..t+20], "
        "entry O[t+1]). The matched baseline averages outcomes of other in_dd days but is only a comparison, "
        "not a signal input. F1 uses no flow/news data.")
    say("2d Caveats (not look-ahead in the signal): (i) the desk cache is a download-time snapshot, so prices are "
        "split/dividend-adjusted as of download; ratios inside 60-day windows are unaffected except for dividends "
        "inside the window (tiny for these names). (ii) Data start 2025-06-02 means Ldd and in_dd in Sep 2025 can "
        "only see back to June 2025 (truncated look-back, affects the 2025 slice only). (iii) Universe = the current "
        "desk watchlist (82 AI names) -> possible survivorship/selection bias (names that collapsed and were never "
        "or no longer watched are absent; not verified here); it would lift 'the low held' for rule and baseline alike.")

    # =============================================================== 3. robustness
    say("\n" + "-" * 100)
    say("3. ROBUSTNESS  hold20 (n with outcome) in 2026 | 2026H1 | 2026H2 | 2025(Sep-Dec); registered L1 check")
    say("-" * 100)
    say(f"{'variant':34s} {'2026':>11s}  {'2026H1':>11s}  {'2026H2':>11s}  {'2025':>11s}")
    rows = {}

    def variant(label, sig_, indd_=indd, cool=COOL, eval_shift=0, up_=up, oc_=oc):
        fs_ = first_signals(sig_, indd_, up_, dates, cool=cool)
        e = evaluate(fs_, dates, oc_, up_, eval_shift=eval_shift)
        rows[label] = e
        say(row_str(label, e))
        return e

    variant("BASE sig4 (m=4)", sig)
    for mm in (3.0, 3.5, 4.5, 5.0, 6.0):
        variant(f"m={mm:g}", np.isfinite(up) & np.isfinite(sg) & (up >= mm * sg))
    sig_lag = np.zeros_like(sig)
    sig_lag[1:] = sig[:-1]
    variant("lag1: rule fired at t-1, act at t", sig_lag)
    variant("entry delay: same signals, eval t+1", sig, eval_shift=1)
    sig_p2 = sig.copy()
    sig_p2[1:] &= sig[:-1]
    sig_p2[0] = False
    variant("persist: fires t-1 and t", sig_p2)
    variant("cooldown 10", sig, cool=10)
    variant("cooldown 30", sig, cool=30)
    variant("in_dd thr 15%", sig, indd_=in_dd_of(dd60, 0.15))
    variant("in_dd thr 25%", sig, indd_=in_dd_of(dd60, 0.25))
    for w in (15, 30):
        _, _, _, sgw, _ = features(C, sigwin=w)
        variant(f"sigma window {w} (m=4)", np.isfinite(up) & np.isfinite(sgw) & (up >= MULT * sgw))
    ph = pd.read_csv(FAM_PH_SUM, encoding="utf-8-sig")
    ph = ph[ph.rule == "sig4_lag1"].set_index("period")
    say(f"Family post-hoc sig4_lag1 for comparison: 2026 {ph.loc['2026','hold20']:.3f}, "
        f"H1 {ph.loc['2026H1','hold20']:.3f}, H2 {ph.loc['2026H2','hold20']:.3f}, 2025 {ph.loc['2025','hold20']:.3f}")
    fsall = pd.read_csv(FAM_SUM, encoding="utf-8-sig").set_index(["rule", "period"])
    say("Family registered neighbours: " + ", ".join(
        f"{r} 2026 {fsall.loc[(r, '2026'), 'hold20']:.3f} / H1 {fsall.loc[(r, '2026H1'), 'hold20']:.3f} / "
        f"H2 {fsall.loc[(r, '2026H2'), 'hold20']:.3f}" for r in ("sig3", "sig6")))
    n_var = len(rows) - 1
    n_fail = sum(not l1_pass(e) for lbl, e in rows.items() if lbl != "BASE sig4 (m=4)")
    say(f"Variants failing the registered L1: {n_fail}/{n_var} -> "
        + ", ".join(lbl for lbl, e in rows.items() if not l1_pass(e)))

    # =============================================================== 4. null
    say("\n" + "-" * 100)
    say(f"4. NULL: random in_dd (stock, day) in {MAIN}, same count and up-bucket mix, same 20-session cooldown, "
        f"{N_NULL} draws")
    say("-" * 100)
    d26 = (dates >= pd.Timestamp("2026-01-02")) & (dates <= LAST)
    pool = indd & np.isfinite(up) & np.isfinite(oc["hold20"]) & d26[:, None]
    pool_p, pool_k = np.nonzero(pool)
    pool_up = up[pool_p, pool_k]
    pool_b = np.digitize(pool_up, BUCKET_EDGES)
    obs_sel = m26.to_numpy()
    obs = mine[obs_sel]
    obs_k = np.array([cols.index(s) for s in obs.series])
    obs_p = obs.pos.to_numpy()
    obs_b = np.digitize(obs.up.to_numpy(), BUCKET_EDGES)
    obs_h = float(obs.hold20.mean())
    need = {b: int((obs_b == b).sum()) for b in range(4)}
    say(f"Observed {MAIN}: n={len(obs)}, hold20={obs_h:.4f}; bucket mix "
        + ", ".join(f"{BUCKET_NAMES[b]}={need[b]}" for b in range(4)))
    say(f"Pool: {len(pool_p)} in_dd stock-days with outcome; pool hold20 by bucket "
        + ", ".join(f"{BUCKET_NAMES[b]}={np.mean(oc['hold20'][pool_p[pool_b == b], pool_k[pool_b == b]]):.3f}"
                    f"(n={int((pool_b == b).sum())})" for b in range(4)))
    hold = oc["hold20"]
    nres = {}
    r1, _, bm1 = null_draws(pool_k, pool_p, pool_b, need, hold, N, K, np.random.default_rng(SEED + 2))
    nres["A. 4 up buckets (registered)"] = (r1, np.array([bm1[b] for b in obs_b]))
    strat = {}
    for kk, bb in zip(obs_k, obs_b):
        strat[(kk, bb)] = strat.get((kk, bb), 0) + 1
    r2, short, bm2 = null_draws(pool_k, pool_p, pool_b, need, hold, N, K, np.random.default_rng(SEED + 3),
                                strat_k=strat)
    nres[f"B. stock x bucket ({short:.1f} picks/draw refilled)"] = (r2, np.array([bm2[b] for b in obs_b]))
    qs = np.quantile(pool_up, np.linspace(0, 1, 11)[1:-1])
    pool_d = np.digitize(pool_up, qs)
    obs_d = np.digitize(obs.up.to_numpy(), qs)
    need_d = {b: int((obs_d == b).sum()) for b in range(10) if (obs_d == b).sum()}
    r3, relaxed, bm3 = null_draws(pool_k, pool_p, pool_d, need_d, hold, N, K, np.random.default_rng(SEED + 4),
                                  relax=True)
    say("Pool-decile need (decile: count): " + ", ".join(f"{b}:{n}" for b, n in sorted(need_d.items()))
        + f"; top-decile edge up >= {qs[-1]:.3f}")
    nres[f"C. 10 up deciles ({relaxed:.1f} picks/draw no-cooldown)"] = (r3, np.array([bm3[b] for b in obs_d]))
    for cool0 in (0,):
        r0, _, bm0 = null_draws(pool_k, pool_p, pool_b, need, hold, N, K, np.random.default_rng(SEED + 6),
                                cool=cool0, n_draws=1000)
        nres["D. 4 up buckets, NO cooldown (1000)"] = (r0, np.array([bm0[b] for b in obs_b]))
    obs_lab = obs_p // COOL
    v_obs = obs.hold20.to_numpy(float)
    say("p(null >= obs) treats the 260 picks as independent. 'clustered p' block-bootstraps the per-signal gap "
        "(hold20 - null bucket mean) over 20-session blocks, because the signals sit in ~10 market-wide blocks.")
    clustered = {}
    for lbl, (r, e_i) in nres.items():
        p = float((r >= obs_h - 1e-12).mean())
        gap = v_obs - e_i
        u_, inv_ = np.unique(obs_lab, return_inverse=True)
        sums, cnts = np.bincount(inv_, weights=gap), np.bincount(inv_).astype(float)
        pick = np.random.default_rng(SEED + 7).integers(0, len(u_), size=(N_BOOT, len(u_)))
        bm = sums[pick].sum(1) / cnts[pick].sum(1)
        pc = float((bm <= 0).mean())
        clustered[lbl] = pc
        say(f"{lbl:42s} null mean {r.mean():.4f}, 95% [{np.quantile(r, .025):.4f}, {np.quantile(r, .975):.4f}], "
            f"p = {p:.4f} | gap {gap.mean():+.4f}, block 95% [{np.quantile(bm, .025):+.4f}, "
            f"{np.quantile(bm, .975):+.4f}], clustered p = {pc:.4f}")
    p_main = float((r1 >= obs_h - 1e-12).mean())
    p_main_cl = clustered["A. 4 up buckets (registered)"]
    s15 = pool_b == 3
    b_all = np.where(np.isfinite(up), np.digitize(np.nan_to_num(up, nan=-1.0), BUCKET_EDGES), -1)
    entry15 = np.zeros_like(pool)
    entry15[1:] = pool[1:] & (b_all[1:] == 3) & (b_all[:-1] != 3)
    ep, ek = np.nonzero(entry15)
    dsl = pool_p - lowpos[pool_p, pool_k]
    late15 = s15 & (dsl >= 20)
    say(f"Why the cooldown null sits below the per-day bucket average: in the 15%+ pool, the first day a stock enters "
        f"the bucket holds {np.mean(hold[ep, ek]):.3f} (n={len(ep)}) while 15%+ days >= 20 sessions after Ldd hold "
        f"{np.mean(hold[pool_p[late15], pool_k[late15]]):.3f} (n={int(late15.sum())}). Long persistent runs carry "
        f"many pool days but the cooldown lets them contribute one pick per 21 sessions, so the cooldown null "
        f"under-weights them. With no cooldown the null equals the day-weighted matched baseline (row D, ~= family "
        f"hold20_base {fsum.loc[MAIN, 'hold20_base']:.3f}). Observed 15%+ signals: "
        f"{np.mean(v_obs[obs_b == 3]):.3f} (n={int((obs_b == 3).sum())}) vs day-weighted 15%+ pool "
        f"{np.mean(hold[pool_p[s15], pool_k[s15]]):.3f}; median sessions since Ldd: signals "
        f"{np.median(obs_p - lowpos[obs_p, obs_k]):.0f}, pool {np.median(dsl):.0f}.")
    # halves (supplementary, registered buckets)
    for hn, a_, b_ in (("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23")):
        dm = (dates >= pd.Timestamp(a_)) & (dates <= pd.Timestamp(b_))
        pl = indd & np.isfinite(up) & np.isfinite(hold) & dm[:, None]
        pp_, pk_ = np.nonzero(pl)
        pb_ = np.digitize(up[pp_, pk_], BUCKET_EDGES)
        om = period_mask(obs.date, a_, b_).to_numpy()
        ob = obs_b[om]
        nd = {b: int((ob == b).sum()) for b in range(4)}
        oh = float(obs.hold20[om].mean())
        rr, _, _ = null_draws(pk_, pp_, pb_, nd, hold, N, K, np.random.default_rng(SEED + 5), n_draws=1000)
        say(f"  supplementary {hn}: obs {oh:.4f} (n={int(om.sum())}), null mean {rr.mean():.4f}, "
            f"p = {(rr >= oh - 1e-12).mean():.4f}")

    # =============================================================== 5. mechanical check
    say("\n" + "-" * 100)
    say("5. MECHANICAL CHECK: is hold20 just a function of how far price has already rebounded (up)?")
    say("-" * 100)
    up_med_all = float(np.median(mine[period_mask(mine.date, "2026-01-02", "2026-10-23")].up))
    up_med_fam = 0.1887
    say(f"Candidate's 2026 median up: mine {up_med_all:.4f} (family reported {up_med_fam})")
    say(f"{'variant':34s} {'2026':>11s}  {'2026H1':>11s}  {'2026H2':>11s}  {'2025':>11s}")
    say(row_str("BASE sig4 (m=4)", rows["BASE sig4 (m=4)"]))
    for thr_ in (up_med_all, 0.15, 0.20, 0.25):
        variant(f"plain up >= {thr_:.4f}", np.isfinite(up) & (up >= thr_))
    # expected hold20 given each signal's up, from the pool of all in_dd days in the same half (k-NN in up)
    exp = []
    for hn, a_, b_ in (("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23")):
        dm = (dates >= pd.Timestamp(a_)) & (dates <= pd.Timestamp(b_))
        pl = indd & np.isfinite(up) & np.isfinite(hold) & dm[:, None]
        pp_, pk_ = np.nonzero(pl)
        u_ = up[pp_, pk_]
        h_ = hold[pp_, pk_]
        order = np.argsort(u_)
        us, hs = u_[order], h_[order]
        om = period_mask(obs.date, a_, b_).to_numpy()
        e_h = []
        for u0 in obs.up.to_numpy()[om]:
            i = np.searchsorted(us, u0)
            lo, hi = max(0, i - 150), min(len(us), i + 150)
            e_h.append(hs[lo:hi].mean())
        e_h = np.array(e_h)
        exp.append((hn, int(om.sum()), float(obs.hold20[om].mean()), float(e_h.mean())))
    for hn, n_, o_, e_ in exp:
        say(f"k-NN (300 nearest in up, same half, all in_dd stock-days) expected hold20 {hn}: {e_:.4f}  vs observed "
            f"{o_:.4f} (n={n_}) -> diff {o_ - e_:+.4f}")
    ntot = sum(x[1] for x in exp)
    say(f"  2026 pooled: expected {sum(x[1] * x[3] for x in exp) / ntot:.4f} vs observed "
        f"{sum(x[1] * x[2] for x in exp) / ntot:.4f}")
    # within-up-decile: sig4-true vs sig4-false days (all 2026 in_dd days, not first signals)
    say("Within-up-decile comparison over all 2026 in_dd stock-days (sig4 true vs false):")
    pl = pool
    pp_, pk_ = np.nonzero(pl)
    u_, h_, s_ = up[pp_, pk_], hold[pp_, pk_], sig[pp_, pk_]
    dq = np.digitize(u_, np.quantile(u_, np.linspace(0, 1, 11)[1:-1]))
    wsum = wd = 0.0
    for d in range(10):
        mt, mf = (dq == d) & s_, (dq == d) & ~s_
        if mt.sum() and mf.sum():
            say(f"  decile {d}: up {u_[dq == d].min():.3f}..{u_[dq == d].max():.3f}  sig4 {h_[mt].mean():.3f} (n={mt.sum():5d})"
                f"  not {h_[mf].mean():.3f} (n={mf.sum():5d})")
            wsum += mt.sum() * (h_[mt].mean() - h_[mf].mean())
            wd += mt.sum()
        elif mt.sum():
            say(f"  decile {d}: sig4 {h_[mt].mean():.3f} (n={mt.sum()}), no non-signal days")
    say(f"  sig4-weighted within-decile difference (sig4 - not): {wsum / wd:+.4f}")
    X = np.column_stack([np.ones(len(u_)), np.log1p(u_), np.log1p(u_) ** 2, s_.astype(float)])
    beta, se = logit(X, h_)
    say(f"Logit hold20 ~ log1p(up) + log1p(up)^2 + sig4 on {len(u_)} stock-days: sig4 coef {beta[3]:+.3f} "
        f"(naive SE {se[3]:.3f}; stock-days are heavily cross-correlated, so the true SE is several times larger)")
    X0 = np.column_stack([np.ones(len(u_)), np.log1p(u_), np.log1p(u_) ** 2])
    b0, _ = logit(X0, h_)
    p_sig_days = 1 / (1 + np.exp(-(X0[s_] @ b0)))
    say(f"  up-only logit: predicted hold20 on sig4 days {p_sig_days.mean():.3f} vs actual {h_[s_].mean():.3f}")
    say("  -> at a given up, sig4 days hold more often because sig4 also means LOW trailing volatility: the cushion "
        "above Ldd is larger in sigma units. That is still a price-distance effect, now scaled by volatility:")
    # cushion in sigma units: z = log(1+up)/sigma20 (all known at t); sig4 <=> up >= 4 sigma (z ~ >= 3.6-4)
    z_pool = np.log1p(u_) / sg[pp_, pk_]
    z_obs = np.log1p(obs.up.to_numpy()) / sg[obs_p, obs_k]
    rw_obs = rw_hold_prob(z_obs)
    rw_pool = rw_hold_prob(z_pool)
    h1m = period_mask(obs.date, "2026-01-02", "2026-05-29").to_numpy()
    say(f"  Random-walk benchmark (driftless Gaussian, trailing sigma20, 20 closes must stay above Ldd): first signals "
        f"{MAIN} predicted {rw_obs.mean():.3f} vs actual {obs_h:.3f} (H1 {rw_obs[h1m].mean():.3f} vs "
        f"{v_obs[h1m].mean():.3f}, H2 {rw_obs[~h1m].mean():.3f} vs {v_obs[~h1m].mean():.3f}); "
        f"median z of signals {np.median(z_obs):.2f}")
    zq = np.unique(np.quantile(z_pool, np.linspace(0, 1, 11)))
    say("  z-quantile calibration on all 2026 in_dd stock-days (actual vs random-walk; actual for sig4 / non-sig4):")
    dz = np.clip(np.digitize(z_pool, zq[1:-1]), 0, len(zq) - 2)
    for d in range(len(zq) - 1):
        mz = dz == d
        st_, nt_ = mz & s_, mz & ~s_
        say(f"    z {zq[d]:5.2f}..{zq[d + 1]:6.2f}: actual {h_[mz].mean():.3f} RW {rw_pool[mz].mean():.3f}  "
            f"sig4 {f3(h_[st_].mean() if st_.any() else np.nan)} (n={int(st_.sum()):4d})  "
            f"non {f3(h_[nt_].mean() if nt_.any() else np.nan)} (n={int(nt_.sum()):4d})")
    # z-matched expectation (k-NN in z, same half) for the first signals
    ez = []
    for hn, a_, b_ in (("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23")):
        hm = (dates[pp_] >= pd.Timestamp(a_)) & (dates[pp_] <= pd.Timestamp(b_))
        order = np.argsort(z_pool[hm])
        zs, hs = z_pool[hm][order], h_[hm][order]
        om = period_mask(obs.date, a_, b_).to_numpy()
        e_h = []
        for z0 in z_obs[om]:
            i = np.searchsorted(zs, z0)
            lo, hi = max(0, i - 150), min(len(zs), i + 150)
            e_h.append(hs[lo:hi].mean())
        ez.append((hn, int(om.sum()), float(v_obs[om].mean()), float(np.mean(e_h))))
        say(f"  k-NN in z (300 nearest, same half) expected hold20 {hn}: {np.mean(e_h):.4f} vs observed "
            f"{v_obs[om].mean():.4f} -> diff {v_obs[om].mean() - np.mean(e_h):+.4f}")
    ntz = sum(x[1] for x in ez)
    say(f"  2026 pooled z-matched expectation {sum(x[1] * x[3] for x in ez) / ntz:.4f} vs observed {obs_h:.4f}")
    say(f"Outcome quality of the candidate in {MAIN}: near5 {f3(ev[MAIN]['near5'])} (n={ev[MAIN]['near5_n']}), "
        f"win20 {f3(ev[MAIN]['win20'])} (n={ev[MAIN]['win20_n']}), mean r20 {ev[MAIN]['r20']:+.4f}; "
        f"matched baseline (family) hold20 {fsum.loc[MAIN, 'hold20_base']:.4f}, dlo {fsum.loc[MAIN, 'hold20_dlo']:+.4f}")

    # =============================================================== 6. verdict
    say("\n" + "-" * 100)
    say("6. VERDICT")
    say("-" * 100)
    lag_fail = not l1_pass(rows["lag1: rule fired at t-1, act at t"])
    nb_fail = [lbl for lbl in ("m=3.5", "m=4.5", "m=3", "m=5") if not l1_pass(rows[lbl])]
    say(f"independent match: {match}; look-ahead found: {bad > 0}; null p (registered buckets, independent picks) = "
        f"{p_main:.4f}; clustered p = {p_main_cl:.4f}; lag1 fails L1: {lag_fail}; neighbour multipliers failing L1: "
        f"{nb_fail or 'none'}; variants failing L1: {n_fail}/{n_var}")
    if not match or bad > 0:
        v = "refuted"
    elif p_main < 0.05 and p_main_cl < 0.05 and not lag_fail and not nb_fail and n_fail == 0:
        v = "confirmed"
    else:
        v = "fragile"
    pl_ = rows[f"plain up >= {up_med_all:.4f}"]
    say(f"Key facts: 2026 hold20 {ev[MAIN]['hold20']:.3f} (n={ev[MAIN]['hold20_n']}, {ev[MAIN]['hold20_clusters']} "
        f"blocks, block CI [{ev[MAIN]['hold20_lo']:.3f}, {ev[MAIN]['hold20_hi']:.3f}]); H1 {ev['2026H1']['hold20']:.3f} "
        f"is only {100 * (ev['2026H1']['hold20'] - 0.70):.1f}pp above the 70% line and drops to "
        f"{rows['lag1: rule fired at t-1, act at t']['2026H1']['hold20']:.3f} with a one-session delay; m=3.5 gives "
        f"2026 {rows['m=3.5']['2026']['hold20']:.3f} / H1 {rows['m=3.5']['2026H1']['hold20']:.3f}. Plain up >= "
        f"{up_med_all:.3f} gives 2026 {pl_['2026']['hold20']:.3f} (H1 {pl_['2026H1']['hold20']:.3f}, H2 "
        f"{pl_['2026H2']['hold20']:.3f}). Excess over the family's day-weighted matched baseline is "
        f"{ev[MAIN]['hold20'] - fsum.loc[MAIN, 'hold20_base']:+.3f} with block lower bound "
        f"{fsum.loc[MAIN, 'hold20_dlo']:+.3f}; near5 {ev[MAIN]['near5']:.3f}, win20 {ev[MAIN]['win20']:.3f}: "
        f"the 'confirmation' arrives after a ~{100 * ev[MAIN]['up_median']:.0f}% rebound and is followed by a "
        f"coin-flip 20-day return.")
    say(f"VERDICT: {v}")
    OUT.write_text(buf.getvalue(), encoding="utf-8")
    return v


if __name__ == "__main__":
    main()
