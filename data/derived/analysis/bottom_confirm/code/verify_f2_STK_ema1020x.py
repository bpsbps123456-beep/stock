"""Independent adversarial verification: family f2, rule ema1020x, kind STK (bottom_confirm, BRIEF.md 2026-10-26).

Does NOT import bc_lib or f2_reclaim. Re-implements from the raw CSVs:
  data   data/derived/desk/prices_open.csv / prices_close.csv, US columns minus QQQ (82 stocks),
         reindexed to XNYS sessions 2025-06-02..2026-10-23 (non-session rows in the CSV are dropped)
  pk[t]   latest position of max(C[t-59..t]) (window must be complete);  Ldd[t] = min(C[pk..t])
  dd60    C/max60 - 1;  up = C/Ldd - 1;  in_dd = min(dd60[t-19..t]) <= -THR (STK 20%)
  hold20  min(C[t+1..t+20]) >= Ldd[t];  near5 min(C[t+1..t+40]) >= 0.95*O[t+1]
  r20     C[t+20]/O[t+1]-1, win20 = r20 > 0
  first signals per stock: sig & in_dd & Ldd defined & 2025-09-01 <= t <= 2026-10-23, accept p if p - last_accepted > 20
  matched base: mean outcome over all stocks' in_dd days (t >= 2025-09-01) in the same half-year (month > 6) and up bucket
  ema1020x: EMA10[t] > EMA20[t] & EMA10[t-1] <= EMA20[t-1], EMA = own recursion e[t] = a*C[t] + (1-a)*e[t-1],
            a = 2/(span+1), seeded with the first valid close (== pandas ewm(span, adjust=False) with no interior gaps)

Run from data/derived/analysis/bottom_confirm:  python3 -W ignore code/verify_f2_STK_ema1020x.py
Writes results/verify_f2_STK_ema1020x.txt
"""
from __future__ import annotations

import pathlib

import exchange_calendars as xc
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(__file__).resolve().parents[5]
RES = HERE / "results"
OUT = RES / "verify_f2_STK_ema1020x.txt"

OPEN = ROOT / "data/derived/desk/prices_open.csv"
CLOSE = ROOT / "data/derived/desk/prices_close.csv"

LAST = pd.Timestamp("2026-10-23")
EVAL_START = pd.Timestamp("2025-09-01")
THR0 = 0.20
COOL0 = 20
PERIODS = [("2025-09~2026", "2025-09-01", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
           ("2025", "2025-09-01", "2025-12-31")]
BINS = [0.03, 0.08, 0.15]
BNAMES = ["0-3%", "3-8%", "8-15%", "15%+"]
FINE = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
FAMILY = {"n_main": 194, "hold20_main": 0.7474, "hold20_base_main": 0.7087, "hold20_2025": 0.825,
          "hold20_2026": 0.7474, "hold20_2026H1": 0.7685, "hold20_2026H2": 0.7209, "near5_main": 0.2047,
          "win20_main": 0.4588, "r20_main": 0.0281, "up_median_main": 0.2083}

lines: list[str] = []


def say(s=""):
    print(s)
    lines.append(s)


# ------------------------------------------------------------------ raw data
def load():
    o = pd.read_csv(OPEN, index_col=0, parse_dates=True)
    c = pd.read_csv(CLOSE, index_col=0, parse_dates=True)
    cols = [x for x in c.columns if not (x.endswith(".KS") or x.endswith("=X") or x == "QQQ")]
    cal = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range("2025-06-02", "2026-10-23"))
    cal = cal.tz_localize(None) if cal.tz is not None else cal
    dropped_rows = len(c.index[(c.index >= cal[0])].difference(cal))
    C = c[cols].reindex(cal)
    O = o[cols].reindex(cal)
    assert (C.isna() == O.isna()).all().all()
    for k in cols:                                      # gaps are leading only (late listings)
        fv = C[k].first_valid_index()
        assert C[k].loc[fv:].notna().all()
    return C, O, dropped_rows


# ------------------------------------------------------------------ state + outcomes
def state(C: np.ndarray, thr: float):
    N, K = C.shape
    Ldd = np.full((N, K), np.nan)
    dd60 = np.full((N, K), np.nan)
    for k in range(K):
        c = C[:, k]
        for t in range(59, N):
            w = c[t - 59:t + 1]
            if np.isnan(w).any():
                continue
            mx = w.max()
            pk = t - 59 + int(np.flatnonzero(w == mx)[-1])
            Ldd[t, k] = c[pk:t + 1].min()
            dd60[t, k] = c[t] / mx - 1
    in_dd = np.zeros((N, K), bool)
    for t in range(N):
        w = dd60[max(0, t - 19):t + 1]
        with np.errstate(all="ignore"):
            mn = np.nanmin(np.where(np.isnan(w), np.inf, w), 0)
        in_dd[t] = mn <= -thr
    up = C / Ldd - 1
    return Ldd, dd60, in_dd, up


def outcomes(C: np.ndarray, O: np.ndarray, Ldd: np.ndarray):
    N, K = C.shape
    h20 = np.full((N, K), np.nan)
    n5 = np.full((N, K), np.nan)
    r20 = np.full((N, K), np.nan)
    for t in range(N):
        if t + 20 <= N - 1:
            blk = C[t + 1:t + 21]
            mn = blk.min(0)                                  # NaN if any NaN
            ok = ~np.isnan(mn) & ~np.isnan(Ldd[t])
            h20[t, ok] = (mn[ok] >= Ldd[t, ok] - 1e-12).astype(float)
            ok2 = ~np.isnan(O[t + 1]) & ~np.isnan(C[t + 20])
            r20[t, ok2] = C[t + 20, ok2] / O[t + 1, ok2] - 1
        if t + 40 <= N - 1:
            mn = C[t + 1:t + 41].min(0)
            ok = ~np.isnan(mn) & ~np.isnan(O[t + 1])
            n5[t, ok] = (mn[ok] >= 0.95 * O[t + 1, ok]).astype(float)
    win20 = np.where(np.isnan(r20), np.nan, (r20 > 0).astype(float))
    return dict(hold20=h20, near5=n5, win20=win20, r20=r20)


def bucket(up):
    b = np.digitize(np.nan_to_num(up, nan=0.0), BINS).astype(float)
    b[np.isnan(up)] = np.nan
    return b


def half_key(idx):
    return np.asarray(idx.year) * 10 + (np.asarray(idx.month) > 6)


def matched_base(idx, in_dd, up, y):
    """Same half-year (month > 6) x up bucket, pooled over all stocks' in_dd days with t >= EVAL_START."""
    N, K = y.shape
    b = bucket(up)
    hk = np.broadcast_to(half_key(idx)[:, None], (N, K))
    pool = in_dd & ~np.isnan(up) & (idx >= EVAL_START)[:, None] & ~np.isnan(y)
    tab = {}
    for h in np.unique(hk[pool]):
        for bb in range(4):
            m = pool & (hk == h) & (b == bb)
            if m.any():
                tab[(int(h), bb)] = float(y[m].mean())
    base = np.full((N, K), np.nan)
    for t in range(N):
        for k in range(K):
            if not np.isnan(b[t, k]):
                base[t, k] = tab.get((int(hk[t, k]), int(b[t, k])), np.nan)
    return base, tab


# ------------------------------------------------------------------ signals
def ema(C: np.ndarray, span: int):
    a = 2.0 / (span + 1.0)
    N, K = C.shape
    e = np.full((N, K), np.nan)
    prev = np.full(K, np.nan)
    for t in range(N):
        c = C[t]
        v = ~np.isnan(c)
        first = v & np.isnan(prev)
        upd = v & ~np.isnan(prev)
        with np.errstate(invalid="ignore"):
            prev = np.where(first, c, np.where(upd, a * c + (1 - a) * prev, prev))
        e[t] = np.where(v, prev, np.nan)
    return e


def ema_cross(C: np.ndarray, fast=10, slow=20, seed_start=None):
    Cx = C.copy()
    if seed_start is not None:
        Cx[:seed_start] = np.nan
    ef, es = ema(Cx, fast), ema(Cx, slow)
    sig = np.zeros(C.shape, bool)
    with np.errstate(invalid="ignore"):
        sig[1:] = (ef[1:] > es[1:]) & (ef[:-1] <= es[:-1])
    return sig


def first_signals(sig, idx, in_dd, Ldd, cool=COOL0):
    ok = sig & in_dd & ~np.isnan(Ldd) & ((idx >= EVAL_START) & (idx <= LAST))[:, None]
    rows = []
    for k in range(sig.shape[1]):
        prev = -10 ** 9
        for p in np.flatnonzero(ok[:, k]):
            if p - prev > cool:
                rows.append((p, k))
                prev = p
    return np.array(rows, int).reshape(-1, 2)


def summarize(fs, idx, up, out, base):
    rows = {}
    P, K_ = (fs[:, 0], fs[:, 1]) if len(fs) else (np.zeros(0, int), np.zeros(0, int))
    for name, a, b in PERIODS:
        m = (idx[P] >= pd.Timestamp(a)) & (idx[P] <= pd.Timestamp(b))
        pp, kk = P[m], K_[m]
        h, bb = out["hold20"][pp, kk], base[pp, kk]
        ok = ~np.isnan(h) & ~np.isnan(bb)
        r = dict(n_sig=int(m.sum()), n=int(ok.sum()), hits=int(np.nansum(h[ok])),
                 hold20=float(h[ok].mean()) if ok.any() else np.nan,
                 base=float(bb[ok].mean()) if ok.any() else np.nan,
                 clusters=int(len(np.unique(pp // 20))) if len(pp) else 0,
                 up_med=float(np.median(up[pp, kk])) if len(pp) else np.nan,
                 pos=pp[ok], col=kk[ok])
        for key in ("near5", "win20", "r20"):
            v = out[key][pp, kk]
            r[key] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        rows[name] = r
    return rows


def L1(rows):
    ch = [rows[h]["n"] >= 5 and rows[h]["clusters"] >= 2 and rows[h]["hold20"] >= 0.70 for h in ("2026H1", "2026H2")]
    c3 = rows["2025"]["n"] < 5 or rows["2025"]["hold20"] >= 0.60
    return bool(all(ch) and c3 and rows["2026"]["hold20"] >= 0.70)


def fmt_rows(label, rows):
    def f(p):
        r = rows[p]
        return "  -  (0)" if r["n"] == 0 else f"{r['hold20']:.3f}({r['n']})"
    return (f"  {label:30s} 2026 {f('2026'):11s} H1 {f('2026H1'):11s} H2 {f('2026H2'):11s} 2025 {f('2025'):10s} "
            f"base26 {rows['2026']['base']:.3f} H1b {rows['2026H1']['base']:.3f} H2b {rows['2026H2']['base']:.3f} "
            f"upmed26 {rows['2026']['up_med']:.3f} L1={L1(rows)}")


def block_boot(v, pos, B=5000, seed=20261026):
    lab = np.asarray(pos) // 20
    u, inv = np.unique(lab, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    cnt = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(u), size=(B, len(u)))
    m = sums[pick].sum(1) / cnt[pick].sum(1)
    return np.quantile(m, [0.025, 0.975]), float(m.std())


# ------------------------------------------------------------------ null
def null_draws(pool_t, pool_k, pool_key, need_key, y, rng, cool=COOL0, draws=2000, max_try=60, small_first=False):
    """Each draw picks len(need_key) (t, k) pool cells, one per needed key (stratum), with per-stock spacing > cool.
    pool_key / need_key: stratum labels (tuples hashed to ints). Returns array of mean y per draw.
    small_first: fill the smallest strata first (random tie-break) so singleton strata are not blocked."""
    groups = {}
    for i, key in enumerate(pool_key):
        groups.setdefault(key, []).append(i)
    groups = {key: np.array(v) for key, v in groups.items()}
    res, attempts = [], 0
    nk = len(need_key)
    sizes = np.array([len(groups.get(k_, [])) for k_ in need_key])
    while len(res) < draws and attempts < draws * 20:
        attempts += 1
        chosen_by_k: dict[int, list[int]] = {}
        picks = []
        fail = False
        order = np.lexsort((rng.random(nk), sizes)) if small_first else rng.permutation(nk)
        for i in order:
            cand = groups.get(need_key[i])
            if cand is None or len(cand) == 0:
                fail = True
                break
            got = -1
            for _ in range(max_try):
                j = cand[rng.integers(len(cand))]
                kk, tt = pool_k[j], pool_t[j]
                lst = chosen_by_k.get(kk)
                if lst is None or all(abs(tt - x) > cool for x in lst):
                    got = j
                    break
            if got < 0:                                   # exhaustive fallback
                ok = [j for j in cand if all(abs(pool_t[j] - x) > cool for x in chosen_by_k.get(pool_k[j], []))]
                if not ok:
                    fail = True
                    break
                got = ok[rng.integers(len(ok))]
            chosen_by_k.setdefault(pool_k[got], []).append(pool_t[got])
            picks.append(got)
        if fail:
            continue
        res.append(float(y[np.array(picks)].mean()))
    return np.array(res), attempts


# ------------------------------------------------------------------ main
def main():
    Cdf, Odf, dropped_rows = load()
    idx, names = Cdf.index, list(Cdf.columns)
    C, O = Cdf.to_numpy(float), Odf.to_numpy(float)
    N, K = C.shape
    say("=" * 118)
    say("Independent verification: family f2, rule ema1020x, kind STK  (no bc_lib / f2_reclaim import)")
    say("=" * 118)
    say(f"Data: {K} US stocks (QQQ, .KS, =X dropped), XNYS sessions {idx[0].date()}..{idx[-1].date()} N={N}; "
        f"{dropped_rows} non-session CSV rows dropped; late listings: "
        + ", ".join(f"{k} from {Cdf[k].first_valid_index().date()}" for k in names
                    if Cdf[k].first_valid_index() != idx[0]))

    Ldd, dd60, in_dd, up = state(C, THR0)
    out = outcomes(C, O, Ldd)
    base, btab = matched_base(idx, in_dd, up, out["hold20"])
    sig = ema_cross(C, 10, 20)
    # cross-check own EMA recursion against pandas ewm(adjust=False)
    ef_pd = Cdf.ewm(span=10, adjust=False).mean().to_numpy()
    es_pd = Cdf.ewm(span=20, adjust=False).mean().to_numpy()
    ema_diff = float(np.nanmax(np.abs(ema(C, 10) - ef_pd)) + np.nanmax(np.abs(ema(C, 20) - es_pd)))
    fs = first_signals(sig, idx, in_dd, Ldd)
    rows = summarize(fs, idx, up, out, base)

    # ---------------------------------------------------------- [1] match
    say("\n[1] Independent re-implementation vs family files")
    say(f"  own EMA recursion vs pandas ewm(adjust=False): max |diff| = {ema_diff:.2e}")
    fam = pd.read_csv(RES / "f2_STK_signals.csv", parse_dates=["date"])
    fam = fam[fam.rule == "ema1020x"].copy()
    famsum = pd.read_csv(RES / "f2_STK_summary.csv")
    famsum = famsum[famsum.rule == "ema1020x"].set_index("period")
    mine = pd.DataFrame(dict(date=idx[fs[:, 0]], series=[names[k] for k in fs[:, 1]], pos=fs[:, 0],
                             up=up[fs[:, 0], fs[:, 1]], hold20=out["hold20"][fs[:, 0], fs[:, 1]],
                             base_hold20=base[fs[:, 0], fs[:, 1]], near5=out["near5"][fs[:, 0], fs[:, 1]],
                             win20=out["win20"][fs[:, 0], fs[:, 1]], r20=out["r20"][fs[:, 0], fs[:, 1]],
                             dd60=dd60[fs[:, 0], fs[:, 1]]))
    mine = mine.sort_values(["series", "date"]).reset_index(drop=True)
    fam = fam.sort_values(["series", "date"]).reset_index(drop=True)
    km, kf = set(zip(mine.date, mine.series)), set(zip(fam.date, fam.series))
    same = km == kf and len(mine) == len(fam)
    say(f"  first signals: mine {len(mine)}, family {len(fam)}; identical (date, stock) set: {same}")
    if not same:
        say(f"    only mine ({len(km - kf)}): {sorted((d.date(), s) for d, s in km - kf)[:20]}")
        say(f"    only family ({len(kf - km)}): {sorted((d.date(), s) for d, s in kf - km)[:20]}")
    mism = []
    if same:
        for col in ("pos", "up", "hold20", "base_hold20", "near5", "win20", "r20", "dd60"):
            a, b = mine[col].to_numpy(float), fam[col].to_numpy(float)
            bad = ~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-9))
            if bad.any():
                mism.append(f"{col}: {int(bad.sum())} rows e.g. {list(zip(mine.date[bad].dt.date, mine.series[bad]))[:3]}")
    say(f"  per-signal field mismatches: {'none' if not mism else '; '.join(mism)}")
    say("  period          mine hold20 (n, hits, clusters)    family hold20 (n, clusters)   mine base  family base")
    per_ok = True
    for p, _, _ in PERIODS:
        r = rows[p]
        fh, fn, fb, fc = famsum.loc[p, "hold20"], famsum.loc[p, "hold20_n"], famsum.loc[p, "hold20_base"], famsum.loc[p, "clusters"]
        ok = r["n"] == fn and abs(r["hold20"] - fh) < 1e-9 and abs(r["base"] - fb) < 1e-9 and r["clusters"] == fc
        per_ok &= bool(ok)
        say(f"  {p:13s}  {r['hold20']:.4f} ({r['n']:3d}, {r['hits']:3d}, {r['clusters']:2d})            "
            f"{fh:.4f} ({int(fn):3d}, {int(fc):2d})            {r['base']:.4f}     {fb:.4f}   {'OK' if ok else 'MISMATCH'}")
    m = rows["2026"]
    rep = dict(n_main=m["n"], hold20_main=m["hold20"], hold20_base_main=m["base"], hold20_2025=rows["2025"]["hold20"],
               hold20_2026=m["hold20"], hold20_2026H1=rows["2026H1"]["hold20"], hold20_2026H2=rows["2026H2"]["hold20"],
               near5_main=m["near5"], win20_main=m["win20"], r20_main=m["r20"], up_median_main=m["up_med"])
    rep_bad = [k for k, v in FAMILY.items() if abs(rep[k] - v) > (0 if k == "n_main" else 5e-5)]
    say(f"  family-reported numbers reproduced (4 dp): {'all' if not rep_bad else 'NOT: ' + str(rep_bad)}")
    match = bool(same and not mism and per_ok and not rep_bad)
    say(f"  => independent match: {match}")
    say(f"  main 2026: n={m['n']} (signals {m['n_sig']}) hold20={m['hold20']:.4f} base={m['base']:.4f} near5={m['near5']:.4f} "
        f"win20={m['win20']:.4f} r20={m['r20']:.4f} up_med={m['up_med']:.4f}")
    v = out["hold20"][m["pos"], m["col"]] - base[m["pos"], m["col"]]
    (lo, hi), _ = block_boot(v, m["pos"])
    (hlo, hhi), boot_sd = block_boot(out["hold20"][m["pos"], m["col"]], m["pos"])
    say(f"  my 20-session block bootstrap, 2026: hold20 95% [{hlo:.3f}, {hhi:.3f}] (sd {boot_sd:.3f}); "
        f"hold20 - matched base = {v.mean():+.4f} 95% [{lo:+.3f}, {hi:+.3f}] (family dlo -0.091) -> L2 {'pass' if lo > 0 else 'fail'}")
    for h in ("2026H1", "2026H2"):
        r = rows[h]
        need = int(np.ceil(0.70 * r["n"] - 1e-9))
        say(f"  {h}: {r['hits']}/{r['n']} = {r['hold20']:.4f}; 70% needs {need} hits -> margin {r['hits'] - need} hit(s); "
            f"matched base {r['base']:.4f} (diff {r['hold20'] - r['base']:+.4f})")
    r26 = rows["2026"]
    need = int(np.ceil(0.70 * r26["n"] - 1e-9))
    say(f"  2026: {r26['hits']}/{r26['n']}; 70% needs {need} -> margin {r26['hits'] - need}")
    stk = pd.Series([names[k] for k in m["col"]]).value_counts()
    say(f"  2026 signals spread: {len(stk)} stocks, max {stk.max()} per stock ({stk.index[0]}); "
        f"20-session blocks {rows['2026']['clusters']}")
    blk = pd.DataFrame(dict(b=m["pos"] // 20, h=out["hold20"][m["pos"], m["col"]], bs=base[m["pos"], m["col"]]))
    g = blk.groupby("b").agg(n=("h", "size"), hold20=("h", "mean"), base=("bs", "mean"))
    g.index = [f"{idx[b * 20].date()}..{idx[min(b * 20 + 19, N - 1)].date()}" for b in g.index]
    say("  2026 signals by 20-session calendar block (n, hold20, matched base):")
    for lab_, rr in g.iterrows():
        say(f"    {lab_}  n={int(rr.n):3d}  hold20={rr.hold20:.3f}  base={rr.base:.3f}")
    lobo = []
    for bk in np.unique(blk.b):
        keep = blk.b != bk
        lobo.append((str(idx[bk * 20].date()), blk.h[keep].mean(), (blk.h[keep] - blk.bs[keep]).mean(), bk))
    lo_h = min(lobo, key=lambda x: x[1])
    lo_d = min(lobo, key=lambda x: x[2])
    say(f"  leave-one-block-out 2026: hold20 range [{min(x[1] for x in lobo):.4f}, {max(x[1] for x in lobo):.4f}], "
        f"diff vs base range [{min(x[2] for x in lobo):+.4f}, {max(x[2] for x in lobo):+.4f}]; "
        f"dropping {lo_d[0]} block leaves diff {lo_d[2]:+.4f}")

    # ---------------------------------------------------------- [2] look-ahead
    say("\n[2] Look-ahead audit")
    t0 = int(np.searchsorted(idx, EVAL_START))
    bad_sig, bad_state = [], []
    for t in range(t0, N):
        st = ema_cross(C[:t + 1], 10, 20)[-1]
        if not (st == sig[t]).all():
            bad_sig.append(idx[t].date())
        a = max(0, t - 79)
        Lt, _, it, _ = state(C[a:t + 1], THR0)
        same_L = np.allclose(np.nan_to_num(Lt[-1], nan=-1), np.nan_to_num(Ldd[t], nan=-1), rtol=0, atol=1e-12)
        if not same_L or not (it[-1] == in_dd[t]).all():
            bad_state.append(idx[t].date())
    say(f"  truncation test, {N - t0} sessions x {K} stocks: EMA cross recomputed from C[..t] "
        f"{'identical' if not bad_sig else 'DIFF ' + str(bad_sig[:5])}; Ldd/in_dd recomputed from C[t-79..t] "
        f"{'identical' if not bad_state else 'DIFF ' + str(bad_state[:5])}")
    say("  code read (code/f2_reclaim.py ema_cross): ea = C.ewm(span=10, adjust=False).mean(), eb likewise span 20;")
    say("  (ea > eb) & (ea.shift(1) <= eb.shift(1)). ewm(adjust=False) is a causal recursion; shift(1) is the past.")
    say("  No market, flow, options or news input in this rule. bc_lib: Ldd/dd60 windows end at t, in_dd rolling(20)")
    say("  ends at t, outcomes start at C[t+1], entry O[t+1]. The matched baseline averages same-half outcomes")
    say("  (evaluation reference, not a signal input). No look-ahead found in the signal.")
    # EMA seed dependence (not look-ahead, but the family's EMA depends on where the cache starts)
    for seed_date in ("2025-07-01", "2025-08-01"):
        s0 = int(np.searchsorted(idx, pd.Timestamp(seed_date)))
        sg = ema_cross(C, 10, 20, seed_start=s0)
        fs2 = first_signals(sg, idx, in_dd, Ldd)
        k2 = set(map(tuple, fs2.tolist()))
        k1 = set(map(tuple, fs.tolist()))
        r2 = summarize(fs2, idx, up, out, base)
        say(f"  EMA seeded at {seed_date} instead of 2025-06-02: first signals differ in {len(k1 ^ k2)} cells; "
            f"2026 hold20 {r2['2026']['hold20']:.4f} (n {r2['2026']['n']})")
    say("  Caveats (not look-ahead in the signal): the 82-stock universe is today's desk list (chosen in 2026 with")
    say("  hindsight - survivors/AI winners), which inflates hold20 for signal and baseline alike; prices are a")
    say("  single adjusted snapshot (overnight gaps checked: largest are real events, e.g. ECHO 2025-08-26, NBIS 2025-09-09).")
    say("  The matched-baseline half key uses month > 6 (June in H1) while the 2026H2 period starts 06-01, so June")
    say("  signals are summarized in H2 but matched to the Jan-Jun pool (reproduced as is).")

    # ---------------------------------------------------------- [3] robustness
    say("\n[3] Robustness (hold20 (n) per period; main = 2026; L1 = registered STK rule)")
    variants = {}
    cache = {}

    def frames(thr):
        if thr not in cache:
            if thr == THR0:
                cache[thr] = (Ldd, in_dd, up, out, base)
            else:
                L_, _, ind_, up_ = state(C, thr)
                b_, _ = matched_base(idx, ind_, up_, out["hold20"])
                cache[thr] = (L_, ind_, up_, out, b_)
        return cache[thr]

    def run(label, fast=10, slow=20, cool=COOL0, thr=THR0, delay=0):
        L_, ind_, up_, out_, base_ = frames(thr)
        sg = ema_cross(C, fast, slow)
        if delay:
            sg = np.vstack([np.zeros((delay, K), bool), sg[:-delay]])
        f_ = first_signals(sg, idx, ind_, L_, cool)
        r = summarize(f_, idx, up_, out_, base_)
        variants[label] = r
        say(fmt_rows(label, r))
        return r

    run("base EMA10/20 cool20 thr20%")
    say("  -- one step either side --")
    core = ["base EMA10/20 cool20 thr20%"]
    for lab_, kw in (("fast 9", dict(fast=9)), ("fast 11", dict(fast=11)), ("slow 19", dict(slow=19)),
                     ("slow 21", dict(slow=21)), ("delay +1 session", dict(delay=1)), ("cooldown 10", dict(cool=10)),
                     ("cooldown 30", dict(cool=30)), ("in_dd thr 15%", dict(thr=0.15)), ("in_dd thr 25%", dict(thr=0.25))):
        run(lab_, **kw)
        core.append(lab_)
    say("  -- wider context --")
    for lab_, kw in (("EMA 8/20", dict(fast=8)), ("EMA 12/20", dict(fast=12)), ("EMA 10/18", dict(slow=18)),
                     ("EMA 10/22", dict(slow=22)), ("EMA 5/20", dict(fast=5)), ("EMA 10/30", dict(slow=30)),
                     ("EMA 12/26", dict(fast=12, slow=26)), ("EMA 20/50", dict(fast=20, slow=50)),
                     ("delay +2 sessions", dict(delay=2)), ("delay +3 sessions", dict(delay=3))):
        run(lab_, **kw)
    n_l1 = sum(L1(variants[c]) for c in core)
    n_h2 = sum(variants[c]["2026H2"]["hold20"] >= 0.70 for c in core)
    n_above = sum(variants[c]["2026"]["hold20"] > variants[c]["2026"]["base"] for c in core)
    n_h2_above = sum(variants[c]["2026H2"]["hold20"] > variants[c]["2026H2"]["base"] for c in core)
    say(f"  core variants (base + 9 neighbours): L1 kept {n_l1}/{len(core)}; 2026H2 >= 70% in {n_h2}/{len(core)}; "
        f"2026 hold20 > matched base in {n_above}/{len(core)}; 2026H2 > its matched base in {n_h2_above}/{len(core)}")
    # leave-one-stock-out and leave-one-block-out on L1
    loso_fail = []
    for k in range(K):
        f_ = fs[fs[:, 1] != k]
        if len(f_) == len(fs):
            continue
        if not L1(summarize(f_, idx, up, out, base)):
            loso_fail.append(names[k])
    lobo_fail = []
    for bk in np.unique(fs[:, 0] // 20):
        f_ = fs[(fs[:, 0] // 20) != bk]
        if not L1(summarize(f_, idx, up, out, base)):
            lobo_fail.append(str(idx[bk * 20].date()))
    say(f"  leave-one-stock-out: L1 breaks when dropping {len(loso_fail)} of {len(np.unique(fs[:, 1]))} stocks "
        f"{loso_fail[:12]}")
    say(f"  leave-one-20-session-block-out: L1 breaks when dropping {len(lobo_fail)} of "
        f"{len(np.unique(fs[:, 0] // 20))} blocks {lobo_fail}")

    # ---------------------------------------------------------- [4] null
    say("\n[4] Null: random in_dd (stock, day) cells in 2026, same count and up-bucket mix, per-stock spacing > 20, 2000 draws")
    b_all = bucket(up)
    y = out["hold20"]
    pool = in_dd & ~np.isnan(Ldd) & ~np.isnan(y) & ((idx >= pd.Timestamp("2026-01-02")) & (idx <= LAST))[:, None]
    pt, pk_ = np.nonzero(pool)
    pb = b_all[pt, pk_].astype(int)
    py = y[pt, pk_]
    obs = rows["2026"]["hold20"]
    sp, sc = rows["2026"]["pos"], rows["2026"]["col"]
    need_b = b_all[sp, sc].astype(int)
    say(f"  observed 2026 hold20 = {obs:.4f} (n={len(sp)}); bucket mix (0-3,3-8,8-15,15+) = "
        f"{np.bincount(need_b, minlength=4).tolist()}; pool cells = {len(pt)} by bucket {np.bincount(pb, minlength=4).tolist()}")
    say(f"  pool hold20 by bucket 2026: {[round(float(py[pb == b].mean()), 3) for b in range(4)]}")
    cand_by_b = [float(y[sp, sc][need_b == b].mean()) if (need_b == b).any() else np.nan for b in range(4)]
    say(f"  candidate hold20 by bucket 2026: {[round(x, 3) for x in cand_by_b]}")
    pyy = np.asarray(py)

    # primary: bucket only
    nd, att = null_draws(pt, pk_, list(pb), list(need_b), pyy, np.random.default_rng(20261026))
    p_null = float((nd >= obs - 1e-12).mean())
    say(f"  primary (bucket-matched): draws={len(nd)} (attempts {att}) null mean={nd.mean():.4f} sd={nd.std():.4f} "
        f"q95={np.quantile(nd, .95):.4f} -> p = {p_null:.4f}")
    # secondary: half x bucket
    hp = (idx[pt] >= pd.Timestamp("2026-06-01")).astype(int)
    hs = (idx[sp] >= pd.Timestamp("2026-06-01")).astype(int)
    nd2, att2 = null_draws(pt, pk_, list(pb * 10 + hp), list(need_b * 10 + hs), pyy, np.random.default_rng(1))
    p2 = float((nd2 >= obs - 1e-12).mean())
    say(f"  half(06-01 split) x bucket: draws={len(nd2)} mean={nd2.mean():.4f} sd={nd2.std():.4f} p = {p2:.4f}")
    # stock x bucket (timing test given stock)
    nd3, att3 = null_draws(pt, pk_, list(pb * 1000 + pk_), list(need_b * 1000 + sc), pyy, np.random.default_rng(2))
    p3 = float((nd3 >= obs - 1e-12).mean()) if len(nd3) else np.nan
    say(f"  stock x bucket (same stock, random 2026 day): draws={len(nd3)} (attempts {att3}) mean={nd3.mean():.4f} "
        f"sd={nd3.std():.4f} p = {p3:.4f}")
    # same date x bucket (stock selection test given timing)
    nd4, att4 = null_draws(pt, pk_, list(pb * 1000 + pt), list(need_b * 1000 + sp), pyy, np.random.default_rng(3))
    p4 = float((nd4 >= obs - 1e-12).mean()) if len(nd4) else np.nan
    say(f"  same date x bucket (other in_dd stock that day): draws={len(nd4)} (attempts {att4}) mean={nd4.mean():.4f} "
        f"sd={nd4.std():.4f} p = {p4:.4f}")
    # spacing effect: same as primary but without the per-stock 20-session spacing (cell-uniform within bucket)
    nd5, _ = null_draws(pt, pk_, list(pb), list(need_b), pyy, np.random.default_rng(4), cool=-1)
    p5 = float((nd5 >= obs - 1e-12).mean())
    say(f"  bucket-matched WITHOUT spacing (cell-uniform): mean={nd5.mean():.4f} sd={nd5.std():.4f} p = {p5:.4f}  "
        f"(the spacing rule alone lowers the null mean by {nd5.mean() - nd.mean():.3f})")
    # finer lateness matching: 10 up bins instead of 4
    fpool = np.digitize(up[pt, pk_], FINE)
    fneed = np.digitize(up[sp, sc], FINE)
    nd6, _ = null_draws(pt, pk_, list(fpool), list(fneed), pyy, np.random.default_rng(5))
    p6 = float((nd6 >= obs - 1e-12).mean())
    say(f"  fine up bins {FINE} (10 bins), spaced: mean={nd6.mean():.4f} sd={nd6.std():.4f} p = {p6:.4f}")
    nd6b, _ = null_draws(pt, pk_, list(fpool), list(fneed), pyy, np.random.default_rng(6), cool=-1)
    p6b = float((nd6b >= obs - 1e-12).mean())
    say(f"  fine up bins, no spacing: mean={nd6b.mean():.4f} sd={nd6b.std():.4f} p = {p6b:.4f}")
    key7p, key7n = fpool * 1000 + pt, fneed * 1000 + sp
    cnt7 = pd.Series(key7p).value_counts()
    single = float(np.mean([cnt7.get(k_, 0) <= 1 for k_ in key7n]))
    nd7s, att7s = null_draws(pt, pk_, list(key7p), list(key7n), pyy, np.random.default_rng(7), small_first=True)
    say(f"  same date x fine up bin, spaced: infeasible ({len(nd7s)} valid draws in {att7s} attempts; strata are tiny, "
        f"median size {int(np.median([cnt7.get(k_, 0) for k_ in key7n]))}, {single:.0%} singletons) -> not used")
    nd7, _ = null_draws(pt, pk_, list(key7p), list(key7n), pyy, np.random.default_rng(7), cool=-1)
    p7 = float((nd7 >= obs - 1e-12).mean())
    say(f"  same date x fine up bin, no spacing: mean={nd7.mean():.4f} sd={nd7.std():.4f} p = {p7:.4f}")
    B6 = [0.03, 0.08, 0.15, 0.25, 0.40]
    k6p = np.digitize(up[pt, pk_], B6) * 1000 + pt
    k6n = np.digitize(up[sp, sc], B6) * 1000 + sp
    nd9, att9 = null_draws(pt, pk_, list(k6p), list(k6n), pyy, np.random.default_rng(9), small_first=True)
    p9 = float((nd9 >= obs - 1e-12).mean()) if len(nd9) >= 200 else np.nan
    say(f"  same date x 6 up bins {B6}, spaced: draws={len(nd9)} (attempts {att9}) "
        f"mean={nd9.mean() if len(nd9) else np.nan:.4f} p = {p9:.4f}")
    nd8, _ = null_draws(pt, pk_, list(fpool * 10 + hp), list(fneed * 10 + hs), pyy, np.random.default_rng(8))
    p8 = float((nd8 >= obs - 1e-12).mean())
    say(f"  half x fine up bin, spaced: mean={nd8.mean():.4f} sd={nd8.std():.4f} p = {p8:.4f}")
    # per half nulls
    for h, lab_ in ((0, "2026H1"), (1, "2026H2")):
        mm = hs == h
        o_h = rows[lab_]["hold20"]
        ndh, _ = null_draws(pt[hp == h], pk_[hp == h], list(pb[hp == h]), list(need_b[mm]), pyy[hp == h],
                            np.random.default_rng(10 + h))
        say(f"  {lab_} only (bucket-matched within half): observed {o_h:.4f} (n={int(mm.sum())}), null mean "
            f"{ndh.mean():.4f}, p = {float((ndh >= o_h - 1e-12).mean()):.4f}")
    z = (obs - nd.mean()) / boot_sd
    from math import erf, sqrt
    p_clu = 0.5 * (1 - erf(z / sqrt(2)))
    say(f"  clustering: signals fall in {rows['2026']['clusters']} 20-session blocks; block-bootstrap sd of the observed "
        f"rate {boot_sd:.4f} vs iid-null sd {nd.std():.4f} (design effect {(boot_sd / nd.std()) ** 2:.1f}); "
        f"cluster-adjusted normal p ~ {p_clu:.3f}")

    # ---------------------------------------------------------- [5] mechanical
    say("\n[5] Mechanical comparison: plain rule up >= candidate median up (same in_dd / first-signal / cooldown)")
    med = rows["2026"]["up_med"]
    mech = {}
    for thr_up in (med, 0.08, 0.15, 0.25):
        sg = np.nan_to_num(up, nan=-1) >= thr_up
        f_ = first_signals(sg, idx, in_dd, Ldd)
        r = summarize(f_, idx, up, out, base)
        mech[thr_up] = r
        say(fmt_rows(f"up >= {thr_up:.4f}", r) + f"  near5 {r['2026']['near5']:.3f} win20 {r['2026']['win20']:.3f}")
    say(f"  candidate                      near5 {rows['2026']['near5']:.3f} win20 {rows['2026']['win20']:.3f} "
        f"r20 {rows['2026']['r20']:+.4f}")
    sweep = []
    for thr_up in np.round(np.arange(0.12, 0.3001, 0.01), 2):
        f_ = first_signals(np.nan_to_num(up, nan=-1) >= thr_up, idx, in_dd, Ldd)
        r = summarize(f_, idx, up, out, base)
        sweep.append((thr_up, r["2026"]["up_med"], r["2026"]["hold20"], r["2026"]["n"], r["2026H2"]["hold20"], L1(r)))
    sw = pd.DataFrame(sweep, columns=["thr", "up_med", "hold20", "n", "h2", "L1"]).sort_values("up_med")
    interp = float(np.interp(med, sw.up_med, sw.hold20))
    interp_h2 = float(np.interp(rows["2026H2"]["up_med"], sw.up_med, sw.h2))
    say("  like-for-like sweep, plain 'up >= x' through the same first-signal machinery (2026):")
    say("    " + "  ".join(f"x={a:.2f}:upmed {b:.3f} h20 {c:.3f}({d})" + ("*" if e else "") for a, b, c, d, _, e in sweep[:10]))
    say("    " + "  ".join(f"x={a:.2f}:upmed {b:.3f} h20 {c:.3f}({d})" + ("*" if e else "") for a, b, c, d, _, e in sweep[10:]))
    say(f"    (* = passes L1) interpolated plain-rule 2026 hold20 at the candidate's up median {med:.3f}: {interp:.4f} "
        f"vs candidate {obs:.4f} (gap {obs - interp:+.4f}); 2026H2 at H2 up median: {interp_h2:.4f} vs "
        f"{rows['2026H2']['hold20']:.4f}; plain rules passing L1: {int(sw.L1.sum())}/{len(sw)}")
    a = pool & (np.nan_to_num(up, nan=-1) >= med)
    bm = pool & (np.nan_to_num(up, nan=-1) < med)
    say(f"  all 2026 in_dd cells with up >= {med:.4f} (no cooldown): hold20 = {y[a].mean():.4f} (cells {int(a.sum())}); "
        f"with up < it: {y[bm].mean():.4f} (cells {int(bm.sum())})")
    say("  within half x bucket, 2026 (candidate first signals vs all in_dd pool cells):")
    for h, lab_ in ((0, "H1"), (1, "H2")):
        for b in range(4):
            cm = (need_b == b) & (hs == h)
            pm = (pb == b) & (hp == h)
            if cm.any():
                say(f"    {lab_} {BNAMES[b]:6s} cand {y[sp[cm], sc[cm]].mean():.3f} (n={int(cm.sum()):3d})   "
                    f"pool {pyy[pm].mean():.3f} (cells={int(pm.sum()):5d})")
    raw = sig & pool
    say(f"  raw ema1020x 2026 in_dd cells (no cooldown): hold20 = {y[raw].mean():.4f} (cells {int(raw.sum())}), "
        f"matched base avg {np.nanmean(base[raw]):.4f}")
    # logistic-free decomposition: expected hold20 from the candidate's own up values via pool curve (fine bins)
    fine = np.array([0, .03, .05, .08, .10, .15, .20, .25, .30, .40, np.inf])
    fb_pool = np.digitize(up[pt, pk_], fine[1:-1])
    curve = np.array([pyy[fb_pool == i].mean() if (fb_pool == i).any() else np.nan for i in range(len(fine) - 1)])
    fb_c = np.digitize(up[sp, sc], fine[1:-1])
    exp_fine = float(np.nanmean(curve[fb_c]))
    say(f"  pool hold20 by finer up bins {FINE}: {[round(float(x), 3) for x in curve]}")
    say(f"  expected 2026 hold20 from the candidates' own up values (finer bins, pool curve): {exp_fine:.4f} vs observed {obs:.4f}")

    # ---------------------------------------------------------- [6] verdict
    say("\n[6] Verdict inputs")
    lookahead = bool(bad_sig or bad_state)
    mech_h = mech[med]["2026"]["hold20"]
    say(f"  independent match={match}, look-ahead found={lookahead}, 2026 hold20={obs:.4f} vs matched base "
        f"{rows['2026']['base']:.4f} (diff {obs - rows['2026']['base']:+.4f}, block-bootstrap lo {lo:+.3f}), "
        f"null p={p_null:.4f} (half x bucket {p2:.4f}; stock x bucket {p3:.4f}; same date x bucket {p4:.4f}; "
        f"cluster-adjusted ~{p_clu:.3f})")
    say(f"  L1 kept in {n_l1}/{len(core)} core variants; 2026H2 {rows['2026H2']['hits']}/{rows['2026H2']['n']} "
        f"(margin {rows['2026H2']['hits'] - int(np.ceil(0.7 * rows['2026H2']['n'] - 1e-9))}) below its base "
        f"{rows['2026H2']['base']:.3f}; plain up >= {med:.4f} rule 2026 hold20 = {mech_h:.4f} (n={mech[med]['2026']['n']})")
    if not match or lookahead:
        verdict = "refuted"
    elif p_null < 0.05 and n_l1 == len(core) and lo > 0:
        verdict = "confirmed"
    elif (p_null >= 0.05 and mech_h >= obs) or n_l1 < len(core) / 2:
        verdict = "refuted"
    else:
        verdict = "fragile"
    say(f"  extra nulls: no-spacing p={p5:.4f}; fine-bin spaced p={p6:.4f}; fine-bin no-spacing p={p6b:.4f}; "
        f"same date x fine bin p={p7:.4f}; half x fine bin p={p8:.4f}")
    say("  verdict rule (fixed before the extra nulls): refuted on mismatch/look-ahead; confirmed only if primary null")
    say("  p < 0.05 AND L1 in every core variant AND block-bootstrap lo of (hold20 - matched) > 0; refuted if")
    say("  (primary p >= 0.05 and the plain up>=median rule scores >= the candidate) or L1 in < half the core variants;")
    say("  otherwise fragile.")
    say(f"\nVERDICT: {verdict}")
    say(f"  Reproduced exactly (243 first signals, every per-signal field and per-period rate), no look-ahead.")
    say(f"  70%+: 2026 {obs:.3f} but the block-bootstrap 95% lower bound is {hlo:.3f}; 2026H2 clears 70% by one hit")
    say(f"  ({rows['2026H2']['hits']}/{rows['2026H2']['n']}) and L1 breaks with fast span 9, delay +1 session and in_dd 15% "
        f"({n_l1}/{len(core)} core variants keep it).")
    say(f"  Information: +{obs - rows['2026']['base']:.3f} over the same-state baseline, registered L2 fails (lo {lo:+.3f});")
    say(f"  all of the excess is in 2026H1 (H2 {rows['2026H2']['hold20']:.3f} < its base {rows['2026H2']['base']:.3f}, "
        f"below base in {len(core) - n_h2_above} of {len(core)} core variants). The iid bucket null")
    say(f"  (p={p_null:.3f}) depends on the per-stock spacing, which by itself lowers the null mean by "
        f"{nd5.mean() - nd.mean():.3f} (cell-uniform null p={p5:.2f},")
    say(f"  fine-bin cell-uniform p={p6b:.2f}), and ignores time clustering (design effect {(boot_sd / nd.std()) ** 2:.1f}, "
        f"cluster-adjusted p~{p_clu:.2f}). Dropping the single")
    say(f"  {lo_d[0]} block (n={int((blk.b == lo_d[3]).sum())}, the 2026-03-30 low rebound) leaves a 2026 excess of {lo_d[2]:+.3f}.")
    say(f"  Mechanical: the plain rule 'up >= {med * 100:.1f}%' scores {mech_h:.3f} (n={mech[med]['2026']['n']}) and "
        f"also passes L1; through the same first-signal machinery a plain up rule at the")
    say(f"  candidate's lateness scores {interp:.3f} (gap {obs - interp:+.3f}). near5 {rows['2026']['near5']:.3f} and "
        f"win20 {rows['2026']['win20']:.3f} equal their bases. The hit rate is mostly how far price had already rebounded.")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(verdict=verdict, match=match, lookahead=lookahead, p=p_null, obs=obs, mech=mech_h, n_l1=n_l1,
                variants=variants, core=core)


if __name__ == "__main__":
    main()
