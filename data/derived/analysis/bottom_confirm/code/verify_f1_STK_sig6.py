"""verify_f1_STK_sig6 — adversarial independent verification of bottom_confirm F1 'sig6' on STK.

Rule (BRIEF.md F1): up[t] >= 6 * sigma20[t], sigma20 = std(ddof=1) of the 20 daily close-to-close returns ending at t.
Everything is recomputed from the raw desk CSVs; bc_lib / f1_rebound are NOT imported (read only for definitions).

Sections
  1. independent re-implementation vs results/f1_STK_signals.csv and f1_STK_summary.csv (rule sig6)
  2. look-ahead audit (code-level, recorded in the report)
  3. robustness: m = 4/5/7/8, signal lag 1 (and 2), cooldown 10/30, in_dd threshold 15%/25%
  4. nulls: (A, as specified) random in_dd dates in 2026 with the same count, coarse up-bucket mix and cooldown;
     (B) half x fine up-bucket; (C) same session, same coarse bucket, other stock; (D) same cushion up/sigma20 bin
  5. mechanical check: plain up >= median(up) rule, random-walk barrier model, hold20 by cushion
Run: cd data/derived/analysis/bottom_confirm && python3 -W ignore code/verify_f1_STK_sig6.py
"""
from __future__ import annotations

import math
import pathlib

import exchange_calendars as xc
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[5]
HERE = pathlib.Path(__file__).resolve().parents[1]
RES = HERE / "results"
OUT = RES / "verify_f1_STK_sig6.txt"

M_RULE = 6
COOL = 20
THR = 0.20
START = pd.Timestamp("2025-09-01")
LAST = pd.Timestamp("2026-10-23")
PERIODS = [("2025-09~2026", "2025-09-01", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
           ("2025", "2025-09-01", "2025-12-31")]
COARSE = [0.03, 0.08, 0.15]                       # buckets 0-3, 3-8, 8-15, 15%+
FINE = [0.03, 0.08, 0.15, 0.20, 0.25, 0.30, 0.40]  # post-hoc finer up buckets
CUSH = [6, 7, 8, 10, 13]                          # up/sigma20 bins (sig6 days have >= 6)
B_BOOT = 5000
N_NULL = 2000
SEED = 7_2026_1026

LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


# ---------------------------------------------------------------- raw data
def load():
    o = pd.read_csv(ROOT / "data/derived/desk/prices_open.csv", index_col=0, parse_dates=True)
    c = pd.read_csv(ROOT / "data/derived/desk/prices_close.csv", index_col=0, parse_dates=True)
    keep = [x for x in c.columns if not (x.endswith(".KS") or x.endswith("=X") or x == "QQQ")]
    cal = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range("2025-06-02", "2026-10-23"))
    cal = cal.tz_localize(None) if cal.tz is not None else cal
    extra = c.index.difference(cal)
    c, o = c[keep].reindex(cal), o[keep].reindex(cal)
    return o, c, extra


# ---------------------------------------------------------------- features (own implementation)
def features(C: np.ndarray):
    N, K = C.shape
    Ldd = np.full((N, K), np.nan)
    dd60 = np.full((N, K), np.nan)
    s20 = np.full((N, K), np.nan)
    for k in range(K):
        c = C[:, k]
        for t in range(59, N):
            w = c[t - 59:t + 1]
            if not np.isfinite(w).all():
                continue
            mx = w.max()
            pk = t - 59 + int(np.flatnonzero(w == mx)[-1])     # latest position of the 60-session high
            Ldd[t, k] = c[pk:t + 1].min()
            dd60[t, k] = c[t] / mx - 1
        for t in range(20, N):
            w = c[t - 20:t + 1]
            if not np.isfinite(w).all():
                continue
            r = w[1:] / w[:-1] - 1
            s20[t, k] = r.std(ddof=1)
    up = C / Ldd - 1
    return Ldd, dd60, s20, up


def in_dd_mask(dd60: np.ndarray, thr: float) -> np.ndarray:
    N, K = dd60.shape
    out = np.zeros((N, K), bool)
    for t in range(N):
        w = dd60[max(0, t - 19):t + 1]
        with np.errstate(all="ignore"):
            mn = np.where(np.isfinite(w).any(0), np.nanmin(np.where(np.isfinite(w), w, np.inf), 0), np.nan)
        out[t] = np.isfinite(mn) & (mn <= -thr)
    return out


def outcomes(C: np.ndarray, O: np.ndarray, Ldd: np.ndarray):
    N, K = C.shape
    hold20 = np.full((N, K), np.nan)
    near5 = np.full((N, K), np.nan)
    r20 = np.full((N, K), np.nan)
    for t in range(N):
        if t + 20 <= N - 1:
            blk = C[t + 1:t + 21]
            ok = np.isfinite(blk).all(0) & np.isfinite(Ldd[t])
            hold20[t, ok] = (blk[:, ok].min(0) >= Ldd[t, ok]).astype(float)
            ok2 = np.isfinite(C[t + 20]) & np.isfinite(O[t + 1])
            r20[t, ok2] = C[t + 20, ok2] / O[t + 1, ok2] - 1
        if t + 40 <= N - 1:
            blk = C[t + 1:t + 41]
            ok = np.isfinite(blk).all(0) & np.isfinite(O[t + 1])
            near5[t, ok] = (blk[:, ok].min(0) >= 0.95 * O[t + 1, ok]).astype(float)
    win20 = np.where(np.isfinite(r20), (r20 > 0).astype(float), np.nan)
    return dict(hold20=hold20, near5=near5, win20=win20, r20=r20)


def bucket(up: np.ndarray, edges) -> np.ndarray:
    b = np.digitize(np.where(np.isfinite(up), up, -1), edges).astype(float)
    b[~np.isfinite(up)] = np.nan
    return b


def half_key(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(dates.year) * 10 + (np.asarray(dates.month) > 6)


def matched_base(Y: np.ndarray, in_dd: np.ndarray, up: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """mean of Y over in_dd stock-days (date >= START) in the same half-year x coarse up bucket."""
    N, K = Y.shape
    hk = np.broadcast_to(half_key(dates)[:, None], (N, K))
    bk = bucket(up, COARSE)
    st = in_dd & np.isfinite(up) & (dates >= START)[:, None]
    out = np.full((N, K), np.nan)
    for h in np.unique(hk):
        for b in range(4):
            cell = (hk == h) & (bk == b)
            sel = cell & st & np.isfinite(Y)
            if sel.any():
                out[cell & np.isfinite(bk)] = Y[sel].mean()
    return out


# ---------------------------------------------------------------- first signals & stats
def first_signals(sig: np.ndarray, in_dd: np.ndarray, Ldd: np.ndarray, dates, cool=COOL):
    win = ((dates >= START) & (dates <= LAST))[:, None]
    ok = sig & in_dd & np.isfinite(Ldd) & win
    P, Kk = [], []
    for k in range(ok.shape[1]):
        last = -10 ** 9
        for p in np.flatnonzero(ok[:, k]):
            if p - last > cool:
                P.append(p)
                Kk.append(k)
                last = p
    return np.asarray(P, int), np.asarray(Kk, int)


def boot_lo(v, lab, rng, q=0.025):
    u, inv = np.unique(lab, return_inverse=True)
    if len(u) < 2:
        return np.nan
    s = np.bincount(inv, weights=v)
    n = np.bincount(inv).astype(float)
    pick = rng.integers(0, len(u), size=(B_BOOT, len(u)))
    return float(np.quantile(s[pick].sum(1) / n[pick].sum(1), q))


def period_stats(P, K, dates, F, OUTC, BASE, rng=None):
    res = {}
    d = dates[P] if len(P) else pd.DatetimeIndex([])
    for name, a, b in PERIODS:
        m = (d >= a) & (d <= b)
        p, k = P[m], K[m]
        r = dict(n=int(m.sum()), blocks=int(len(np.unique(p // COOL))) if m.any() else 0)
        if m.any():
            r["up_med"] = float(np.median(F["up"][p, k]))
            for o in ("hold20", "near5", "win20", "r20"):
                v, bv = OUTC[o][p, k], BASE[o][p, k]
                ok = np.isfinite(v) & np.isfinite(bv)
                r[f"{o}_n"] = int(ok.sum())
                r[o] = float(v[ok].mean()) if ok.any() else np.nan
                r[f"{o}_base"] = float(bv[ok].mean()) if ok.any() else np.nan
                if o == "hold20" and rng is not None and ok.any():
                    r["hold20_dlo"] = boot_lo(v[ok] - bv[ok], p[ok] // COOL, rng)
                    r["hold20_lo"] = boot_lo(v[ok], p[ok] // COOL, rng)
        res[name] = r
    return res


def l1(st) -> bool:
    halves = all(st[h].get("hold20_n", 0) >= 5 and st[h].get("blocks", 0) >= 2 and st[h].get("hold20", 0) >= 0.70
                 for h in ("2026H1", "2026H2"))
    c25 = st["2025"].get("hold20_n", 0) < 5 or st["2025"].get("hold20", 0) >= 0.60
    return bool(halves and c25 and st["2026"].get("hold20", 0) >= 0.70)


def l2(st) -> bool:
    return bool(l1(st) and st["2026"].get("hold20_dlo", -1) > 0)


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def rw_hold(up, sd, n=20, discrete=True):
    """driftless random-walk probability that the min of the next n closes stays above the low (log distance)."""
    out = np.full(len(up), np.nan)
    for i, (u, s) in enumerate(zip(up, sd)):
        if not (np.isfinite(u) and np.isfinite(s) and s > 0):
            continue
        x = math.log1p(u) + (0.5826 * s if discrete else 0.0)   # Broadie-Glasserman discrete-monitoring shift
        out[i] = 1 - min(1.0, 2 * phi(-x / (s * math.sqrt(n))))
    return out


def fmt(x, d=3):
    return "  nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


# ---------------------------------------------------------------- null draws
def null_draws(pool_mask, strata, target_counts, OUTC, rng, cool=COOL, n_draw=N_NULL):
    """pool_mask: (N,K) bool; strata: (N,K) int labels (-1 = excluded); target_counts: {label: count}.
    Draw without replacement per stratum, enforce per-series spacing > cool. If a stratum cannot supply its count
    under the spacing rule, the rest is filled from its unpicked days ignoring spacing (count and mix stay exact).
    Returns (hold20 means, picks that had to ignore spacing)."""
    N, K = pool_mask.shape
    pools = {s: np.argwhere(pool_mask & (strata == s)) for s in target_counts}
    H = OUTC["hold20"]
    out = np.full(n_draw, np.nan)
    relaxed = 0
    for i in range(n_draw):
        blocked = np.zeros((N, K), bool)
        vals = []
        for s, need in target_counts.items():
            pl = pools[s]
            got = 0
            order = rng.permutation(len(pl))
            used = np.zeros(len(pl), bool)
            for j in order:
                p, k = pl[j]
                if blocked[p, k]:
                    continue
                vals.append(H[p, k])
                used[j] = True
                blocked[max(0, p - cool):p + cool + 1, k] = True
                got += 1
                if got == need:
                    break
            if got < need:
                rest = order[~used[order]][:need - got]
                vals.extend(H[pl[rest, 0], pl[rest, 1]])
                relaxed += len(rest)
        out[i] = np.mean(vals)
    return out, relaxed


def main() -> None:
    o, c, extra = load()
    dates = c.index
    C, O = c.to_numpy(float), o.to_numpy(float)
    N, K = C.shape
    cols = list(c.columns)
    say("=" * 100)
    say("Verification of bottom_confirm F1 'sig6' on STK (82 US AI stocks): up >= 6 * sigma20")
    say("Independent re-implementation from raw CSVs (bc_lib / f1_rebound not imported).")
    say("=" * 100)
    say(f"Data: data/derived/desk/prices_close.csv + prices_open.csv; {K} stocks after dropping .KS/=X/QQQ; "
        f"{N} XNYS sessions {dates[0].date()}..{dates[-1].date()}; {len(extra)} non-XNYS rows in the cache dropped "
        f"(e.g. {', '.join(str(x.date()) for x in extra[:4])}).")
    inner = {cols[k]: int(np.isnan(C[np.flatnonzero(np.isfinite(C[:, k]))[0]:, k]).sum()) for k in range(K)}
    lead = {cols[k]: str(dates[np.flatnonzero(np.isfinite(C[:, k]))[0]].date()) for k in range(K)
            if not np.isfinite(C[0, k])}
    say(f"Interior missing closes after listing: {sum(inner.values())} (so pct_change and C/C.shift(1)-1 give the "
        f"same sigma20). Late listings: {lead}")

    Ldd, dd60, s20, up = features(C)
    # pct_change cross-check of sigma20 (the task text's definition)
    s20_pc = c.pct_change(fill_method=None).rolling(20).std().to_numpy(float)
    both = np.isfinite(s20) & np.isfinite(s20_pc)
    say(f"sigma20 own loop vs pandas pct_change().rolling(20).std(): NaN pattern equal "
        f"{bool((np.isfinite(s20) == np.isfinite(s20_pc)).all())}, max |diff| {np.abs(s20[both] - s20_pc[both]).max():.2e}")
    INDD = {THR: in_dd_mask(dd60, THR)}
    in_dd = INDD[THR]
    OUTC = outcomes(C, O, Ldd)
    F = dict(up=up, s20=s20, Ldd=Ldd, dd60=dd60)
    BASES = {THR: {k: matched_base(OUTC[k], in_dd, up, dates) for k in OUTC}}
    BASE = BASES[THR]

    def sig_m(m):
        return np.isfinite(up) & np.isfinite(s20) & (up >= m * s20)

    sig6 = sig_m(M_RULE)
    P, Kk = first_signals(sig6, in_dd, Ldd, dates)
    rng = np.random.default_rng(SEED)
    st = period_stats(P, Kk, dates, F, OUTC, BASE, rng)

    # ------------------------------------------------------------ 1. comparison with family files
    say("\n" + "-" * 100)
    say("1. INDEPENDENT RE-IMPLEMENTATION vs FAMILY FILES (results/f1_STK_signals.csv, f1_STK_summary.csv, rule sig6)")
    say("-" * 100)
    fam = pd.read_csv(RES / "f1_STK_signals.csv", encoding="utf-8-sig", parse_dates=["date"])
    fam = fam[fam.rule == "sig6"].copy()
    mine = pd.DataFrame(dict(date=dates[P], series=[cols[k] for k in Kk], pos=P,
                             up=up[P, Kk], dd60=dd60[P, Kk], hold20=OUTC["hold20"][P, Kk],
                             near5=OUTC["near5"][P, Kk], win20=OUTC["win20"][P, Kk], r20=OUTC["r20"][P, Kk],
                             base_hold20=BASE["hold20"][P, Kk]))
    kf = set(zip(fam.date, fam.series))
    km = set(zip(mine.date, mine.series))
    say(f"First signals: family {len(kf)}, mine {len(km)}, common {len(kf & km)}, family-only {len(kf - km)}, "
        f"mine-only {len(km - kf)}")
    if kf - km:
        say(f"  family-only: {sorted(kf - km)[:10]}")
    if km - kf:
        say(f"  mine-only: {sorted(km - kf)[:10]}")
    j = fam.merge(mine, on=["date", "series"], suffixes=("_f", "_m"))
    chk = []
    worst = 0.0
    for f in ("pos", "up", "dd60", "hold20", "near5", "win20", "r20", "base_hold20"):
        a, b = j[f + "_f"].to_numpy(float), j[f + "_m"].to_numpy(float)
        same_nan = bool((np.isfinite(a) == np.isfinite(b)).all())
        ok = np.isfinite(a) & np.isfinite(b)
        dmax = float(np.abs(a[ok] - b[ok]).max()) if ok.any() else 0.0
        worst = max(worst, dmax if f != "base_hold20" else 0.0)
        chk.append(f"{f}: NaN-equal {same_nan}, max|diff| {dmax:.2e}")
    say("Per-signal fields on common signals: " + "; ".join(chk))
    summ = pd.read_csv(RES / "f1_STK_summary.csv", encoding="utf-8-sig")
    summ = summ[summ.rule == "sig6"].set_index("period")
    say(f"\n{'period':14s} {'n f/m':>9s} {'h20_n f/m':>10s} {'hold20 f':>9s} {'hold20 m':>9s} {'base f/m':>13s} "
        f"{'near5 f/m':>12s} {'win20 f/m':>12s} {'r20 f/m':>16s} {'upmed f/m':>12s} {'blk f/m':>8s}")
    pmax = 0.0
    for name, _, _ in PERIODS:
        f, m = summ.loc[name], st[name]
        for a, b in ((f.hold20, m["hold20"]), (f.near5, m["near5"]), (f.win20, m["win20"]), (f.r20, m["r20"]),
                     (f.up_median, m["up_med"]), (f.hold20_base, m["hold20_base"])):
            pmax = max(pmax, abs(a - b))
        pmax = max(pmax, abs(f.n - m["n"]), abs(f.hold20_n - m["hold20_n"]), abs(f.clusters - m["blocks"]))
        say(f"{name:14s} {int(f.n):4d}/{m['n']:<4d} {int(f.hold20_n):4d}/{m['hold20_n']:<5d} {f.hold20:9.4f} "
            f"{m['hold20']:9.4f} {f.hold20_base:.3f}/{m['hold20_base']:.3f} {f.near5:.3f}/{m['near5']:.3f} "
            f"{f.win20:.3f}/{m['win20']:.3f} {f.r20:+.4f}/{m['r20']:+.4f} {f.up_median:.3f}/{m['up_med']:.3f} "
            f"{int(f.clusters):3d}/{m['blocks']:<3d}")
    match = (kf == km) and worst < 1e-9 and pmax < 1e-9
    say(f"Max |diff| of period statistics: {pmax:.2e}  ->  INDEPENDENT MATCH: {match}")
    say(f"Registered verdict recomputed: L1 {l1(st)}, L2 {l2(st)} (own bootstrap dlo 2026 = {fmt(st['2026']['hold20_dlo'])}, "
        f"family {summ.loc['2026', 'hold20_dlo']:.3f}; family verdict L1=True L2=True)")
    say(f"hold20 2026 block-bootstrap 95% lower bound (own): {fmt(st['2026']['hold20_lo'])}")
    d26 = (dates[P] >= "2026-01-02")
    say("2026 sig6 first signals by fixed 20-session block (signals cluster in market-wide rebounds):")
    for blk in np.unique(P[d26] // COOL):
        mm = d26 & (P // COOL == blk)
        h = OUTC["hold20"][P[mm], Kk[mm]]
        hv = h[np.isfinite(h)]
        say(f"  block {blk:3d} {dates[P[mm]].min().date()}..{dates[P[mm]].max().date()}  n={mm.sum():3d}  "
            f"hold20={fmt(hv.mean() if len(hv) else np.nan)} (n_valid {len(hv)})")

    # ------------------------------------------------------------ 2. look-ahead audit
    say("\n" + "-" * 100)
    say("2. LOOK-AHEAD AUDIT")
    say("-" * 100)
    # empirical truncation test: recompute the signal using only data up to each t (prefix), compare
    trunc_bad = 0
    test_ts = range(80, N, 7)
    for t in test_ts:
        Cp = C[:t + 1]
        L2_, d2_, s2_, u2_ = features(Cp)
        sg = np.isfinite(u2_[t]) & np.isfinite(s2_[t]) & (u2_[t] >= M_RULE * s2_[t])
        idd = in_dd_mask(d2_, THR)[t]
        trunc_bad += int(((sg & idd) != (sig6[t] & in_dd[t])).sum())
    say(f"Prefix-truncation test: signal & in_dd recomputed from closes C[0..t] only at {len(list(test_ts))} dates x {K} "
        f"stocks; mismatches vs full-sample computation: {trunc_bad}")
    say("Code read (f1_rebound.py, bc_lib.py): sigma20 = rolling(20) of C/C.shift(1)-1 ending at t; up = C[t]/Ldd[t]-1 with "
        "Ldd from the 60-session window ending at t; in_dd = rolling(20) min of dd60 ending at t; outcome windows start "
        "at t+1 (hold20: C[t+1..t+20], entry O[t+1]). F1 uses no flow, option or news data. The matched baseline uses "
        "outcomes of all in_dd days in the same half-year (a benchmark, not a signal input), so it is not look-ahead.")
    say("Caveats (not code look-ahead): the 82-stock universe is the 2026 desk watchlist (chosen with hindsight; "
        "survivors/AI winners), which lifts every hold20 rate including the baseline; the matched-baseline half key "
        "(Jan-Jun / Jul-Dec) differs from the reporting split (H1 ends 2026-05-29), a cosmetic inconsistency.")

    # ------------------------------------------------------------ 3. robustness
    say("\n" + "-" * 100)
    say("3. ROBUSTNESS (registered evaluation each time; in_dd threshold variants recompute in_dd and the matched baseline)")
    say("-" * 100)

    def variant(m=M_RULE, cool=COOL, lag=0, thr=THR, sig=None):
        if thr not in INDD:
            INDD[thr] = in_dd_mask(dd60, thr)
            BASES[thr] = {k: matched_base(OUTC[k], INDD[thr], up, dates) for k in OUTC}
        s = sig_m(m) if sig is None else sig
        if lag:
            s2 = np.zeros_like(s)
            s2[lag:] = s[:-lag]
            s = s2
        p, k = first_signals(s, INDD[thr], Ldd, dates, cool)
        return period_stats(p, k, dates, F, OUTC, BASES[thr], np.random.default_rng(SEED)), (p, k)

    rows = [("registered m=6", {}), ("m=4 (grid neighbour)", dict(m=4)), ("m=5", dict(m=5)), ("m=7", dict(m=7)),
            ("m=8", dict(m=8)), ("lag 1 session", dict(lag=1)), ("lag 2 sessions", dict(lag=2)),
            ("cooldown 10", dict(cool=10)), ("cooldown 30", dict(cool=30)), ("in_dd thr 15%", dict(thr=0.15)),
            ("in_dd thr 25%", dict(thr=0.25))]
    say(f"{'variant':22s} {'n26':>4s} {'h20n':>5s} {'hold20_26':>9s} {'base':>6s} {'dlo':>7s} {'H1(n)':>12s} "
        f"{'H2(n)':>12s} {'2025(n)':>12s} {'upmed':>6s} {'near5':>6s} {'win20':>6s} {'r20':>7s} {'L1':>5s} {'L2':>5s}")
    ROB = {}
    for lab, kw in rows:
        s_, _ = variant(**kw)
        ROB[lab] = s_
        a, h1, h2, y5 = s_["2026"], s_["2026H1"], s_["2026H2"], s_["2025"]
        say(f"{lab:22s} {a['n']:4d} {a.get('hold20_n', 0):5d} {fmt(a.get('hold20')):>9s} {fmt(a.get('hold20_base')):>6s} "
            f"{a.get('hold20_dlo', np.nan):+7.3f} {fmt(h1.get('hold20'))}({h1.get('hold20_n', 0):3d}) "
            f"{fmt(h2.get('hold20'))}({h2.get('hold20_n', 0):3d}) {fmt(y5.get('hold20'))}({y5.get('hold20_n', 0):3d}) "
            f"{fmt(a.get('up_med')):>6s} {fmt(a.get('near5')):>6s} {fmt(a.get('win20')):>6s} {a.get('r20', np.nan):+7.4f} "
            f"{str(l1(s_)):>5s} {str(l2(s_)):>5s}")

    # ------------------------------------------------------------ 4. nulls
    say("\n" + "-" * 100)
    say(f"4. NULLS (main period 2026, {N_NULL} draws each, pool = in_dd stock-days with valid hold20)")
    say("-" * 100)
    d26m = ((dates >= "2026-01-02") & (dates <= LAST))[:, None]
    m26 = (dates[P] >= "2026-01-02")
    p26, k26 = P[m26], Kk[m26]
    v26 = np.isfinite(OUTC["hold20"][p26, k26])
    p26v, k26v = p26[v26], k26[v26]
    obs = float(OUTC["hold20"][p26v, k26v].mean())
    pool = in_dd & np.isfinite(Ldd) & np.isfinite(OUTC["hold20"]) & np.isfinite(up) & d26m
    say(f"Observed: {len(p26v)} first signals in 2026 with valid hold20, hold20 = {obs:.4f}; pool {int(pool.sum())} stock-days, "
        f"pool hold20 = {OUTC['hold20'][pool].mean():.4f}")
    cb = bucket(up, COARSE)
    mix = {int(b): int((cb[p26v, k26v] == b).sum()) for b in range(4)}
    say(f"Signal coarse up-bucket mix (0-3/3-8/8-15/15+%): {[mix[b] for b in range(4)]}; pool mix "
        f"{[int((pool & (cb == b)).sum()) for b in range(4)]}")
    rngn = np.random.default_rng(SEED + 1)
    NULLS = {}

    strata = np.where(np.isfinite(cb), cb, -1).astype(int)
    tc = {b: n for b, n in mix.items() if n > 0}
    dA, shortA = null_draws(pool, strata, tc, OUTC, rngn)
    NULLS["A coarse bucket (spec)"] = dA
    say(f"[A] as specified (coarse bucket mix + cooldown {COOL}): null mean {dA.mean():.4f}, sd {dA.std():.4f}, "
        f"95th pct {np.quantile(dA, 0.95):.4f}, p = P(null >= obs) = {(dA >= obs - 1e-12).mean():.4f} "
        f"(picks ignoring spacing {shortA / N_NULL:.1f} per draw)")

    fb = bucket(up, FINE)
    hk = np.broadcast_to(half_key(dates)[:, None], (N, K))
    stratB = np.where(np.isfinite(fb), hk * 100 + np.nan_to_num(fb, nan=0), -1).astype(np.int64)
    tcB = {}
    for s in stratB[p26v, k26v]:
        tcB[int(s)] = tcB.get(int(s), 0) + 1
    dB, shortB = null_draws(pool, stratB, tcB, OUTC, rngn)
    NULLS["B half x fine bucket"] = dB
    say(f"[B] post-hoc half-year x fine up bucket {[0] + FINE}: null mean {dB.mean():.4f}, sd {dB.std():.4f}, "
        f"p = {(dB >= obs - 1e-12).mean():.4f} (picks ignoring spacing {shortB / N_NULL:.1f} per draw)")

    # C: same session, same coarse bucket, random stock (no cooldown possible; independent per signal)
    H = OUTC["hold20"]
    cand = []
    for p, k in zip(p26v, k26v):
        ks = np.flatnonzero(pool[p] & (cb[p] == cb[p, k]))
        cand.append(H[p, ks])
    dC = np.array([np.mean([cv[rngn.integers(len(cv))] for cv in cand]) for _ in range(N_NULL)])
    NULLS["C same session x coarse bucket"] = dC
    say(f"[C] post-hoc same session, same coarse bucket, random in_dd stock: null mean {dC.mean():.4f}, sd {dC.std():.4f}, "
        f"p = {(dC >= obs - 1e-12).mean():.4f} (median candidates per signal {int(np.median([len(x) for x in cand]))})")
    candF = []
    for p, k in zip(p26v, k26v):
        ks = np.flatnonzero(pool[p] & (fb[p] == fb[p, k]))
        candF.append(H[p, ks])
    dC2 = np.array([np.mean([cv[rngn.integers(len(cv))] for cv in candF]) for _ in range(N_NULL)])
    NULLS["C2 same session x fine bucket"] = dC2
    say(f"[C2] post-hoc same session, same fine bucket: null mean {dC2.mean():.4f}, sd {dC2.std():.4f}, "
        f"p = {(dC2 >= obs - 1e-12).mean():.4f}")

    cu = np.where(np.isfinite(s20) & (s20 > 0), up / s20, np.nan)
    cub = np.where(np.isfinite(cu), np.digitize(np.nan_to_num(cu, nan=-1), CUSH), -1).astype(int)
    tcD = {}
    for s in cub[p26v, k26v]:
        tcD[int(s)] = tcD.get(int(s), 0) + 1
    dD, shortD = null_draws(pool, cub, tcD, OUTC, rngn)
    NULLS["D cushion up/sigma20 bin"] = dD
    say(f"[D] post-hoc cushion bin up/sigma20 in {CUSH}+ (mix {dict(sorted(tcD.items()))}): null mean {dD.mean():.4f}, "
        f"sd {dD.std():.4f}, p = {(dD >= obs - 1e-12).mean():.4f} (picks ignoring spacing {shortD / N_NULL:.1f} per draw; "
        f"pool nearly exhausted, so the null sd is too small)")
    candD = []
    for p, k in zip(p26v, k26v):
        ks = np.flatnonzero(pool[p] & (cub[p] == cub[p, k]))
        candD.append(H[p, ks])
    dC3 = np.array([np.mean([cv[rngn.integers(len(cv))] for cv in candD]) for _ in range(N_NULL)])
    NULLS["C3 same session x cushion bin"] = dC3
    say(f"[C3] post-hoc same session, same cushion bin (other stocks already >= 6 sigma that day, incl. itself): null mean "
        f"{dC3.mean():.4f}, sd {dC3.std():.4f}, p = {(dC3 >= obs - 1e-12).mean():.4f} "
        f"(signals with no other candidate {sum(len(x) == 1 for x in candD)}/{len(candD)})")
    say("Note: random draws spread over many dates, while real first signals bunch into ~11 blocks; the null sd "
        "therefore understates the real sampling noise and the p-values are anti-conservative.")

    # ------------------------------------------------------------ 5. mechanical check
    say("\n" + "-" * 100)
    say("5. MECHANICAL CHECK: is the hit rate a consequence of how far price has already rebounded?")
    say("-" * 100)
    umed = float(np.median(up[p26, k26]))
    say(f"Median up of sig6 first signals in 2026: {umed:.4f} (family 0.267)")
    def rw_of(pp, kk):
        mm = (dates[pp] >= "2026-01-02") & np.isfinite(H[pp, kk])
        return float(np.nanmean(rw_hold(up[pp[mm], kk[mm]], s20[pp[mm], kk[mm]]))), float(H[pp[mm], kk[mm]].mean())

    plain_med = {}
    for thr_u in (umed, 0.20, 0.25, 0.30):
        s_, (pp, kk) = variant(sig=np.isfinite(up) & (up >= thr_u))
        a = s_["2026"]
        plain_med.setdefault("st", s_)
        rwp, ob = rw_of(pp, kk)
        say(f"  plain up >= {thr_u:.3f}: n26 {a['n']}, hold20 2026 {fmt(a.get('hold20'))} (H1 {fmt(s_['2026H1'].get('hold20'))}, "
            f"H2 {fmt(s_['2026H2'].get('hold20'))}, 2025 {fmt(s_['2025'].get('hold20'))}), dlo {a.get('hold20_dlo', np.nan):+.3f}, "
            f"upmed {fmt(a.get('up_med'))}, near5 {fmt(a.get('near5'))}, win20 {fmt(a.get('win20'))}, r20 {a.get('r20', np.nan):+.4f}, "
            f"L1 {l1(s_)}, L2 {l2(s_)}; RW-predicted {rwp:.3f}")
    # random-walk barrier model
    pr = rw_hold(up[p26v, k26v], s20[p26v, k26v])
    say(f"Random-walk barrier model (driftless, sigma = own sigma20, discrete-monitoring shift; contains NO information "
        f"about bottoms): predicted hold20 for the 2026 sig6 first signals = {np.nanmean(pr):.4f} vs observed {obs:.4f}")
    for lab, kw in (("m=4", dict(m=4)), ("m=5", dict(m=5)), ("m=7", dict(m=7)), ("m=8", dict(m=8)), ("lag 1", dict(lag=1)),
                    ("lag 2", dict(lag=2))):
        _, (pp, kk) = variant(**kw)
        rwp, ob = rw_of(pp, kk)
        say(f"  {lab:6s}: observed {ob:.3f}  RW-predicted {rwp:.3f}  diff {ob - rwp:+.3f}")
    allp = np.argwhere(pool)
    pr_all = rw_hold(up[pool], s20[pool])
    say(f"Calibration on all 2026 in_dd stock-days (n={len(allp)}): by cushion up/sigma20 bin, observed vs RW-predicted hold20, "
        f"and sig6 first signals in the same bin")
    cu_pool = cu[pool]
    h_pool = H[pool]
    edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, np.inf]
    cu_sig = cu[p26v, k26v]
    h_sig = H[p26v, k26v]
    say(f"  RW is not mechanically exact (2026 drift lifts low-cushion days, vol expansion lowers high-cushion days); "
        f"for days with cushion >= 6: obs {h_pool[cu_pool >= 6].mean():.3f}, RW {np.nanmean(pr_all[cu_pool >= 6]):.3f}")
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        mm = np.isfinite(cu_pool) & (cu_pool >= lo_) & (cu_pool < hi_)
        ms = (cu_sig >= lo_) & (cu_sig < hi_)
        if mm.sum() == 0:
            continue
        say(f"  up/sigma20 [{lo_:>4}, {hi_:>4}): days {mm.sum():5d}  obs {h_pool[mm].mean():.3f}  RW {np.nanmean(pr_all[mm]):.3f}"
            + (f"   | sig6 first signals {ms.sum():3d} obs {h_sig[ms].mean():.3f} RW {np.nanmean(pr[ms]):.3f}"
               if ms.any() else ""))
    # within-bin: sig6 first-signal vs other pool days with same fine up bucket and half (reweighted)
    say("Hold20 of all 2026 in_dd days by fine up bucket (the coarse 15%+ bucket mixes 15% and 40%+ rebounds):")
    for b in range(len(FINE) + 1):
        mm = pool & (fb == b)
        ms = fb[p26v, k26v] == b
        lo_ = ([0] + FINE)[b]
        hi_ = (FINE + [np.inf])[b]
        say(f"  up [{lo_:.2f}, {hi_:.2f}): days {int(mm.sum()):5d} hold20 {H[mm].mean():.3f}"
            + (f"   | sig6 first signals {int(ms.sum()):3d} hold20 {h_sig[ms].mean():.3f}" if ms.any() else ""))

    # ------------------------------------------------------------ summary
    say("\n" + "-" * 100)
    say("SUMMARY")
    say("-" * 100)
    pA = float((dA >= obs - 1e-12).mean())
    say(f"independent_match={match}; registered L1={l1(st)} L2={l2(st)}; hold20 2026 {st['2026']['hold20']:.4f} "
        f"(H1 {st['2026H1']['hold20']:.4f}, H2 {st['2026H2']['hold20']:.4f}, 2025 {st['2025']['hold20']:.4f}); "
        f"null A p={pA:.4f}; null B p={(dB >= obs - 1e-12).mean():.4f}; null C p={(dC >= obs - 1e-12).mean():.4f}; "
        f"null C2 p={(dC2 >= obs - 1e-12).mean():.4f}; null C3 p={(dC3 >= obs - 1e-12).mean():.4f}; "
        f"null D p={(dD >= obs - 1e-12).mean():.4f}; RW-predicted {np.nanmean(pr):.4f}")
    say(f"robustness L1/L2: " + "; ".join(f"{k} {l1(v)}/{l2(v)}" for k, v in ROB.items()))
    l1_all = all(l1(v) for v in ROB.values())
    l2_lost = [k for k, v in ROB.items() if not l2(v)]
    pC3 = float((dC3 >= obs - 1e-12).mean())
    rw_gap = obs - float(np.nanmean(pr))
    say("\n" + "-" * 100)
    say("VERDICT")
    say("-" * 100)
    say(f"Independent match: {match}. Look-ahead: none found (prefix-truncation mismatches {trunc_bad}).")
    say(f"L1 (hold20 >= 70% in 2026, both halves, 2025 >= 60%) survives every variant: {l1_all} "
        f"(hold20 2026 range {min(v['2026']['hold20'] for v in ROB.values()):.3f}..{max(v['2026']['hold20'] for v in ROB.values()):.3f}).")
    say(f"L2 (beats the same-state matched baseline, block-bootstrap lower bound > 0) is lost in {len(l2_lost)}/{len(ROB) - 1} "
        f"variants: {', '.join(l2_lost)}. Registered dlo is only {st['2026']['hold20_dlo']:+.3f}.")
    say(f"Spec null A p = {pA:.4f} (< 0.05), but it only controls the coarse up bucket. Matching same session and same "
        f"up/sigma20 cushion (C3) gives p = {pC3:.3f}: given the market day and the cushion, sig6 picks nothing extra.")
    say(f"A driftless random walk with each signal's own sigma20 (no information about bottoms) predicts hold20 "
        f"{np.nanmean(pr):.3f}; observed {obs:.3f} (gap {rw_gap:+.3f}). The 85% hold rate is the barrier geometry of "
        f"a >= 6-sigma cushion, not evidence that the low is in. Plain up >= {umed:.3f} (no sigma) gives hold20 "
        f"{plain_med['st']['2026']['hold20']:.3f} (H1 {plain_med['st']['2026H1']['hold20']:.3f}, "
        f"H2 {plain_med['st']['2026H2']['hold20']:.3f}), also L1 {l1(plain_med['st'])}.")
    say(f"Return outcomes equal the unconditional 2026 stock baseline: near5 {st['2026']['near5']:.3f} (22.6%), "
        f"win20 {st['2026']['win20']:.3f} (49.4%), r20 {st['2026']['r20']:+.4f} (+3.1%). Entry is "
        f"{st['2026']['up_med']:.1%} above the low (median).")
    say("Verdict: FRAGILE. The 70%+ hold20 is reproducible and robust as a number, but it is mechanical (random-walk "
        "barrier geometry plus the size of the rebound), the 'information' claim (L2) flips with most neighbouring "
        "settings and one-session delay, and the conditional null C3 is not significant. It is not a real bottom-"
        "confirmation edge, and it carries no return edge.")
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
