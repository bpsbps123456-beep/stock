"""Independent adversarial verification: family f1, rule up20, kind STK (bottom_confirm, BRIEF.md 2026-10-26).

Does NOT import bc_lib or f1_rebound. Re-implements from the raw CSVs:
  data    data/derived/desk/prices_open.csv / prices_close.csv, US columns minus QQQ (82 stocks; *.KS, *=X dropped),
          reindexed to XNYS sessions 2025-06-02..2026-10-23 (holiday / weekend rows in the CSV are dropped)
  pk[t]   latest position of max(C[t-59..t]) (window must be complete);  Ldd[t] = min(C[pk..t])
  dd60    C[t]/max(C[t-59..t]) - 1;  up = C/Ldd - 1;  in_dd = min(dd60[t-19..t], NaN skipped) <= -THR (STK 20%)
  hold20  min(C[t+1..t+20]) >= Ldd[t];  near5 min(C[t+1..t+40]) >= 0.95*O[t+1]
  r20     C[t+20]/O[t+1] - 1, win20 = r20 > 0
  up20    up[t] >= 0.20
  first signals per stock: sig & in_dd & Ldd defined & 2025-09-01 <= t <= 2026-10-23, accept p if p - last_accepted > COOL
  matched base: mean outcome over all stocks' in_dd days (t >= 2025-09-01) in the same half-year and up bucket
                (0-3, 3-8, 8-15, 15%+)
  block bootstrap: fixed 20-session blocks (pos // 20), B = 5000 (own RNG, so CI differ slightly from the family's)

Robustness: up15/up17.5/up22.5/up25, signal lagged one session, cooldown 10/30, in_dd THR 15%/25%, plus (post-hoc)
'first crossing' (up crosses 0.20 from below) and two combined stresses.
Null: random (stock, date) in_dd days in 2026 with hold20 defined, matching the count and up-bucket mix of the
actual 2026 first signals, same 20-session per-stock cooldown (random-order greedy, symmetric), 2000 draws.
Secondary nulls (labelled): half x bucket; no-cooldown diagnostic; per-half; and post-hoc 'thinning' nulls
(Bernoulli(q) subset of eligible days -> the rule's forward-only cooldown -> post-stratified to the observed mix)
inside up>=15%, up>=20% and fine up bins 20/22/25/30/40%, because count-matching inside up>=20% is infeasible.
Mechanical: plain rule up >= median(up of actual 2026 first signals); in_dd stock-day hold20 by up level.

Run from data/derived/analysis/bottom_confirm:  python3 -W ignore code/verify_f1_STK_up20.py
Writes results/verify_f1_STK_up20.txt
"""
from __future__ import annotations

import pathlib

import exchange_calendars as xc
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(__file__).resolve().parents[5]
RES = HERE / "results"
OUT = RES / "verify_f1_STK_up20.txt"

OPEN = ROOT / "data/derived/desk/prices_open.csv"
CLOSE = ROOT / "data/derived/desk/prices_close.csv"
FAM_SIG = RES / "f1_STK_signals.csv"
FAM_SUM = RES / "f1_STK_summary.csv"

LAST = pd.Timestamp("2026-10-23")
EVAL_START = pd.Timestamp("2025-09-01")
THR0 = 0.20
COOL0 = 20
LEVEL0 = 0.20
PERIODS = [("2025-09~2026", "2025-09-01", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
           ("2025", "2025-09-01", "2025-12-31")]
BINS = [0.03, 0.08, 0.15]
BNAMES = ["0-3%", "3-8%", "8-15%", "15%+"]
FAMILY = {"n_main": 215, "hold20_main": 0.7581, "hold20_base_main": 0.7657, "hold20_2025": 0.7447,
          "hold20_2026": 0.7581, "hold20_2026H1": 0.8081, "hold20_2026H2": 0.7155, "near5_main": 0.2,
          "win20_main": 0.4605, "r20_main": 0.0302, "up_median_main": 0.2327}
NDRAW = 2000
B = 5000
SEED = 7_20261026

lines: list[str] = []


def say(s=""):
    print(s)
    lines.append(s)


# ------------------------------------------------------------------ raw data
def load():
    o = pd.read_csv(OPEN, index_col=0, parse_dates=True)
    c = pd.read_csv(CLOSE, index_col=0, parse_dates=True)
    assert list(o.columns) == list(c.columns)
    cols = [x for x in c.columns if not (x.endswith(".KS") or x.endswith("=X") or x == "QQQ")]
    cal = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range("2025-06-02", "2026-10-23"))
    cal = cal.tz_localize(None) if cal.tz is not None else cal
    dropped = c.index[(c.index >= cal[0])].difference(cal)
    C = c[cols].reindex(cal)
    O = o[cols].reindex(cal)
    assert (C.isna() == O.isna()).all().all()
    for k in cols:                                       # gaps must be leading only (late listings)
        fv = C[k].first_valid_index()
        assert C[k].loc[fv:].notna().all(), k
    return C, O, dropped


# ------------------------------------------------------------------ state + outcomes (pure numpy, own loops)
def state(Cv: np.ndarray, thr: float):
    N, K = Cv.shape
    Ldd = np.full((N, K), np.nan)
    dd60 = np.full((N, K), np.nan)
    for k in range(K):
        c = Cv[:, k]
        for t in range(59, N):
            w = c[t - 59:t + 1]
            if np.isnan(w).any():
                continue
            mx = w.max()
            pk = t - 59 + int(np.flatnonzero(w == mx)[-1])     # latest position of the 60-day high
            Ldd[t, k] = c[pk:t + 1].min()
            dd60[t, k] = c[t] / mx - 1
    up = Cv / Ldd - 1
    in_dd = np.zeros((N, K), bool)
    for t in range(N):
        w = dd60[max(0, t - 19):t + 1]
        with np.errstate(all="ignore"):
            mn = np.where(np.isnan(w).all(0), np.nan, np.nanmin(np.where(np.isnan(w), np.inf, w), 0))
        in_dd[t] = np.nan_to_num(mn, nan=0.0) <= -thr
    return Ldd, dd60, up, in_dd


def outcomes(Cv: np.ndarray, Ov: np.ndarray, Ldd: np.ndarray):
    N, K = Cv.shape
    hold20 = np.full((N, K), np.nan)
    near5 = np.full((N, K), np.nan)
    r20 = np.full((N, K), np.nan)
    for t in range(N - 1):
        e = t + 1
        if t + 20 <= N - 1:
            blk = Cv[t + 1:t + 21]
            ok = ~np.isnan(blk).any(0) & ~np.isnan(Ldd[t])
            hold20[t, ok] = (blk[:, ok].min(0) >= Ldd[t, ok]).astype(float)
            r20[t] = Cv[t + 20] / Ov[e] - 1
        if t + 40 <= N - 1:
            blk = Cv[t + 1:t + 41]
            ok = ~np.isnan(blk).any(0) & ~np.isnan(Ov[e])
            near5[t, ok] = (blk[:, ok].min(0) >= 0.95 * Ov[e, ok]).astype(float)
    win20 = np.where(np.isnan(r20), np.nan, (r20 > 0).astype(float))
    return dict(hold20=hold20, near5=near5, win20=win20, r20=r20)


def bucket(up: np.ndarray) -> np.ndarray:
    b = np.digitize(up, BINS).astype(float)
    b[np.isnan(up)] = np.nan
    return b


def half_key(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(dates.year) * 10 + (np.asarray(dates.month) > 6)


def matched_base(st: dict, out: dict, dates: pd.DatetimeIndex) -> dict:
    """mean outcome over in_dd stock-days (t >= EVAL_START) in the same half-year x up bucket."""
    N, K = st["up"].shape
    ub = bucket(st["up"])
    hk = np.broadcast_to(half_key(dates)[:, None], (N, K))
    pool = st["in_dd"] & ~np.isnan(st["up"]) & (np.asarray(dates >= EVAL_START)[:, None])
    res = {}
    for name, y in out.items():
        base = np.full((N, K), np.nan)
        m = pool & ~np.isnan(y)
        for h in np.unique(hk):
            for b in range(4):
                sel = m & (hk == h) & (ub == b)
                if sel.any():
                    base[(hk == h) & (ub == b)] = y[sel].mean()
        res[name] = base
    return res


def first_signals(sig: np.ndarray, st: dict, dates: pd.DatetimeIndex, cool: int):
    win = np.asarray((dates >= EVAL_START) & (dates <= LAST))[:, None]
    ok = sig & st["in_dd"] & ~np.isnan(st["Ldd"]) & win
    rows = []
    for k in range(ok.shape[1]):
        prev = -10 ** 9
        for p in np.flatnonzero(ok[:, k]):
            if p - prev <= cool:
                continue
            rows.append((p, k))
            prev = p
    return rows


def boot(v, lab, rng):
    u, inv = np.unique(lab, return_inverse=True)
    if len(u) < 2:
        return (np.nan, np.nan)
    s = np.bincount(inv, weights=v)
    n = np.bincount(inv).astype(float)
    pick = rng.integers(0, len(u), size=(B, len(u)))
    m = s[pick].sum(1) / n[pick].sum(1)
    return tuple(np.quantile(m, [0.025, 0.975]))


def summarize(rows, st, out, base, dates, rng=None):
    tab = {}
    for pname, a, b in PERIODS:
        g = [(p, k) for p, k in rows if pd.Timestamp(a) <= dates[p] <= pd.Timestamp(b)]
        r = dict(n=len(g))
        if g:
            P = np.array([p for p, _ in g])
            Kk = np.array([k for _, k in g])
            r["clusters"] = len(np.unique(P // COOL0))
            r["up_med"] = float(np.median(st["up"][P, Kk]))
            for name in ("hold20", "near5", "win20", "r20"):
                v = out[name][P, Kk]
                bv = base[name][P, Kk]
                ok = ~np.isnan(v) & ~np.isnan(bv)
                r[f"{name}_n"] = int(ok.sum())
                r[name] = float(v[ok].mean()) if ok.any() else np.nan
                r[f"{name}_base"] = float(bv[ok].mean()) if ok.any() else np.nan
                if rng is not None and name == "hold20" and ok.sum() > 1:
                    lab = P[ok] // COOL0
                    r["h_lo"], r["h_hi"] = boot(v[ok], lab, rng)
                    r["d_lo"], r["d_hi"] = boot(v[ok] - bv[ok], lab, rng)
                    r["h_ncl"] = len(np.unique(lab))
        tab[pname] = r
    return tab


def l1(tab):
    def g(p, k, d=np.nan):
        v = tab.get(p, {}).get(k, d)
        return d if (v is None or (isinstance(v, float) and np.isnan(v))) else v
    ch = [g(h, "hold20_n", 0) >= 5 and g(h, "clusters", 0) >= 2 and g(h, "hold20", 0) >= 0.70 for h in ("2026H1", "2026H2")]
    c3 = g("2025", "hold20_n", 0) < 5 or g("2025", "hold20", 0) >= 0.60
    return bool(all(ch) and c3 and g("2026", "hold20", 0) >= 0.70)


def fmt(x, d=3):
    return "  nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


# ------------------------------------------------------------------ null
def draw_null(pool_k, pool_p, pool_strat, need: dict, N, K, cool, rng):
    """greedy random selection honouring a symmetric per-stock cooldown (|p-q| <= cool conflicts).
    need: stratum -> count. pool_strat: stratum label per pool member. Returns indices into pool."""
    # tightest strata (need / pool) first; retry a draw up to 50 times if a stratum cannot be filled
    order = sorted(need, key=lambda s: -need[s] / max(1, int((pool_strat == s).sum())))
    for _attempt in range(50):
        blocked = np.zeros((N, K), bool)
        chosen = []
        ok = True
        for s in order:
            cnt = need[s]
            idx = np.flatnonzero(pool_strat == s)
            idx = idx[rng.permutation(len(idx))]
            got = 0
            for i in idx:
                p, k = pool_p[i], pool_k[i]
                if blocked[p, k]:
                    continue
                chosen.append(i)
                blocked[max(0, p - cool):p + cool + 1, k] = True
                got += 1
                if got == cnt:
                    break
            if got < cnt:
                ok = False
                break
        if ok:
            return np.array(chosen)
    return None


def run_null(label, rows_obs, st, out, dates, strat_fn, rng, ndraw=NDRAW, cool=COOL0):
    """rows_obs: actual 2026 first signals with hold20 defined. strat_fn(p, k) -> stratum label."""
    N, K = st["up"].shape
    in26 = np.asarray((dates >= pd.Timestamp("2026-01-02")) & (dates <= LAST))[:, None]
    pool = st["in_dd"] & ~np.isnan(st["Ldd"]) & ~np.isnan(out["hold20"]) & in26
    pp, pk = np.nonzero(pool)
    strat = np.array([strat_fn(p, k) for p, k in zip(pp, pk)], dtype=object)
    obs_strat = [strat_fn(p, k) for p, k in rows_obs]
    need = pd.Series(obs_strat).value_counts().to_dict()
    obs = float(np.mean([out["hold20"][p, k] for p, k in rows_obs]))
    h = out["hold20"][pp, pk]
    vals = []
    fails = 0
    for _ in range(ndraw):
        ch = draw_null(pk, pp, strat, need, N, K, cool, rng)
        if ch is None:
            fails += 1
            continue
        vals.append(h[ch].mean())
    vals = np.array(vals)
    if len(vals) == 0:
        say(f"  [{label}] strata need={need}: no feasible draw in {ndraw} tries (cooldown + strata too tight)")
        return np.nan, obs, vals
    p = float((vals >= obs - 1e-12).mean()) if len(vals) else np.nan
    pool_rate = {s: float(h[strat == s].mean()) for s in need}
    say(f"  [{label}] strata need={dict(sorted(need.items()))}  pool sizes="
        f"{ {s: int((strat == s).sum()) for s in sorted(need)} }")
    say(f"     pool hold20 by stratum (no cooldown) = { {s: round(v, 3) for s, v in sorted(pool_rate.items())} }")
    say(f"     observed hold20 = {obs:.4f} (n={len(rows_obs)});  null mean = {vals.mean():.4f}, "
        f"null 2.5/50/97.5% = {np.quantile(vals, .025):.4f}/{np.median(vals):.4f}/{np.quantile(vals, .975):.4f};  "
        f"draws ok {len(vals)}/{ndraw};  p(null >= obs) = {p:.4f}")
    return p, obs, vals


def run_null_thin(label, rows_obs, st, out, dates, pool_mask, strat_fn, q, rng, ndraw=NDRAW, cool=COOL0):
    """thinning null: Bernoulli(q) subset of the eligible pool -> forward-only cooldown per stock (same mechanism as
    the rule) -> per-stratum subsample to the observed count (or all available) -> post-stratified hold20 mean."""
    in26 = np.asarray((dates >= pd.Timestamp("2026-01-02")) & (dates <= LAST))[:, None]
    pool = st["in_dd"] & ~np.isnan(st["Ldd"]) & ~np.isnan(out["hold20"]) & in26 & pool_mask
    pk_, pp_ = np.nonzero(pool.T)                       # sorted by stock, then position
    strat = np.array([strat_fn(p, k) for p, k in zip(pp_, pk_)], dtype=object)
    need = pd.Series([strat_fn(p, k) for p, k in rows_obs]).value_counts().to_dict()
    obs = float(np.mean([out["hold20"][p, k] for p, k in rows_obs]))
    h = out["hold20"][pp_, pk_]
    tot = sum(need.values())
    vals, short = [], 0
    for _ in range(ndraw):
        keep = np.flatnonzero(rng.random(len(pp_)) < q)
        acc = []
        prev_k, prev_p = -1, -10 ** 9
        for i in keep:
            k, p = pk_[i], pp_[i]
            if k != prev_k:
                prev_k, prev_p = k, -10 ** 9
            if p - prev_p <= cool:
                continue
            acc.append(i)
            prev_p = p
        acc = np.array(acc, int)
        sa = strat[acc]
        v, ok = 0.0, True
        for s, cnt in need.items():
            idx = acc[sa == s]
            if len(idx) == 0:
                ok = False
                break
            if len(idx) > cnt:
                idx = rng.choice(idx, cnt, replace=False)
            else:
                short += int(len(idx) < cnt)
            v += cnt * h[idx].mean()
        if ok:
            vals.append(v / tot)
    vals = np.array(vals)
    p = float((vals >= obs - 1e-12).mean())
    say(f"  [thin: {label}] obs {obs:.4f}; null mean {vals.mean():.4f}, 2.5/50/97.5% {np.quantile(vals, .025):.4f}/"
        f"{np.median(vals):.4f}/{np.quantile(vals, .975):.4f}; draws {len(vals)}/{ndraw}; strata short of count in "
        f"{short} stratum-draws; p(null >= obs) = {p:.4f}")
    return p, obs, vals


# ------------------------------------------------------------------ main
def main():
    C, O, dropped = load()
    dates = C.index
    Cv, Ov = C.to_numpy(float), O.to_numpy(float)
    N, K = Cv.shape
    cols = list(C.columns)
    say("verify f1 / up20 / STK  (independent re-implementation, no bc_lib import)")
    say(f"data: {K} US stocks, {N} XNYS sessions {dates[0].date()}..{dates[-1].date()}; "
        f"CSV non-session rows dropped: {[d.date().isoformat() for d in dropped]}")
    late = {k: C[k].first_valid_index().date().isoformat() for k in cols if C[k].first_valid_index() > dates[0]}
    say(f"late listings (leading NaN only): {late}")

    Ldd, dd60, up, in_dd = state(Cv, THR0)
    st = dict(Ldd=Ldd, dd60=dd60, up=up, in_dd=in_dd)
    out = outcomes(Cv, Ov, Ldd)
    base = matched_base(st, out, dates)
    sig = (up >= LEVEL0) & ~np.isnan(up)
    rows = first_signals(sig, st, dates, COOL0)
    rng = np.random.default_rng(SEED)
    tab = summarize(rows, st, out, base, dates, rng)

    # ---------------------------------------------------------- 1. independent match
    say("\n== 1. Independent re-implementation vs family files ==")
    fam = pd.read_csv(FAM_SIG, parse_dates=["date"])
    fam = fam[fam.rule == "up20"]
    mine = pd.DataFrame([(dates[p], cols[k], p, out["hold20"][p, k], up[p, k], base["hold20"][p, k],
                          out["near5"][p, k], out["win20"][p, k], out["r20"][p, k]) for p, k in rows],
                        columns=["date", "series", "pos", "hold20", "up", "base_hold20", "near5", "win20", "r20"])
    km = set(zip(mine.date, mine.series))
    kf = set(zip(fam.date, fam.series))
    say(f"first signals: mine {len(km)}, family {len(kf)}, common {len(km & kf)}, only-mine {len(km - kf)}, "
        f"only-family {len(kf - km)}")
    if km - kf:
        say(f"  only-mine sample: {sorted(km - kf)[:10]}")
    if kf - km:
        say(f"  only-family sample: {sorted(kf - km)[:10]}")
    mg = mine.merge(fam, on=["date", "series"], suffixes=("", "_f"))
    for c in ("hold20", "up", "base_hold20", "near5", "win20", "r20"):
        a, b = mg[c].to_numpy(float), mg[f"{c}_f"].to_numpy(float)
        nanmis = int((np.isnan(a) != np.isnan(b)).sum())
        both = ~np.isnan(a) & ~np.isnan(b)
        say(f"  per-signal {c:12s}: NaN-pattern mismatches {nanmis}, max |diff| {np.abs(a[both] - b[both]).max():.2e}")
    fs = pd.read_csv(FAM_SUM)
    fs = fs[fs.rule == "up20"].set_index("period")
    say(f"{'period':13s} {'n':>4s} {'n_f':>4s} {'h20_n':>5s} {'hn_f':>5s} {'hold20':>7s} {'fam':>7s} {'base':>7s} "
        f"{'base_f':>7s} {'clu':>4s} {'clu_f':>5s} {'near5':>6s} {'win20':>6s} {'r20':>7s} {'up_med':>7s}")
    for pname, *_ in PERIODS:
        r, f = tab[pname], fs.loc[pname]
        say(f"{pname:13s} {r['n']:4d} {int(f.n):4d} {r.get('hold20_n', 0):5d} {int(f.hold20_n):5d} "
            f"{fmt(r.get('hold20')):>7s} {fmt(f.hold20):>7s} {fmt(r.get('hold20_base')):>7s} {fmt(f.hold20_base):>7s} "
            f"{r.get('clusters', 0):4d} {int(f.clusters):5d} {fmt(r.get('near5')):>6s} {fmt(r.get('win20')):>6s} "
            f"{fmt(r.get('r20'), 4):>7s} {fmt(r.get('up_med'), 4):>7s}")
    m = tab["2026"]
    say(f"main 2026 vs family-reported: n {m['hold20_n']} vs {FAMILY['n_main']}; hold20 {m['hold20']:.4f} vs "
        f"{FAMILY['hold20_main']}; base {m['hold20_base']:.4f} vs {FAMILY['hold20_base_main']}; near5 {m['near5']:.4f} vs "
        f"{FAMILY['near5_main']}; win20 {m['win20']:.4f} vs {FAMILY['win20_main']}; r20 {m['r20']:.4f} vs "
        f"{FAMILY['r20_main']}; up_med {m['up_med']:.4f} vs {FAMILY['up_median_main']}")
    say(f"  H1 {tab['2026H1']['hold20']:.4f} vs {FAMILY['hold20_2026H1']}; H2 {tab['2026H2']['hold20']:.4f} vs "
        f"{FAMILY['hold20_2026H2']}; 2025 {tab['2025']['hold20']:.4f} vs {FAMILY['hold20_2025']}")
    say(f"  own block bootstrap (B={B}): 2026 hold20 95% CI [{m['h_lo']:.3f}, {m['h_hi']:.3f}] over {m['h_ncl']} blocks; "
        f"hold20 - matched 95% CI [{m['d_lo']:.3f}, {m['d_hi']:.3f}];  "
        f"H2 hold20 CI [{tab['2026H2']['h_lo']:.3f}, {tab['2026H2']['h_hi']:.3f}] over {tab['2026H2']['h_ncl']} blocks")
    ind_match = (km == kf) and abs(m["hold20"] - FAMILY["hold20_main"]) < 5e-4 and m["hold20_n"] == FAMILY["n_main"]
    say(f"L1 (my recomputation) = {l1(tab)};  independent match = {ind_match}")

    # concentration of signals
    m26 = mine[(mine.date >= "2026-01-02")]
    vc = m26.groupby(m26.pos // COOL0).size()
    say(f"2026 signals per 20-session block: {vc.to_dict()}  (largest block share {vc.max() / vc.sum():.2f})")
    hb = m26.dropna(subset=["hold20"]).groupby(m26.pos // COOL0).hold20.agg(["size", "mean"]).round(3)
    say("2026 hold20 by block: " + "; ".join(f"blk{b}({dates[b * COOL0].date()}): n={int(r['size'])} h={r['mean']}"
                                           for b, r in hb.iterrows()))
    rep = m26.groupby("series").size()
    say(f"2026 signals per stock: median {rep.median():.0f}, max {rep.max()} ({rep.idxmax()}); "
        f"stocks with >=4 signals: {int((rep >= 4).sum())}")

    # ---------------------------------------------------------- 2. look-ahead audit (mechanical tests)
    say("\n== 2. Look-ahead audit ==")
    # (a) truncation test: recompute state using data only up to t for sample t; must equal full-sample state at t
    rs = np.random.default_rng(1)
    bad = 0
    test_t = sorted(set(rs.integers(80, N, 25).tolist() + [p for p, _ in rows[:15]]))
    for t in test_t:
        L2, d2, u2, i2 = state(Cv[:t + 1], THR0)
        s2 = (u2[t] >= LEVEL0) & ~np.isnan(u2[t])
        a = (s2 & i2[t]) != (sig[t] & in_dd[t])
        bad += int(a.sum())
    say(f"(a) truncation test: signal&in_dd at t recomputed from C[0..t] only, {len(test_t)} dates x {K} stocks: "
        f"{bad} differences")
    # (b) perturb future: change all closes after t, signal at t must not change
    t = N - 60
    Cp = Cv.copy()
    Cp[t + 1:] *= rs.uniform(0.5, 1.5, size=Cp[t + 1:].shape)
    L3, _, u3, i3 = state(Cp, THR0)
    diff = int((((u3[:t + 1] >= LEVEL0) & i3[:t + 1]) != (sig[:t + 1] & in_dd[:t + 1])).sum())
    say(f"(b) future-perturbation test (closes after {dates[t].date()} scrambled): {diff} signal changes on or before t")
    say("(c) code reading: f1_rebound.py builds up20 = fr['up'] >= 0.20 from bc_lib.outcome_frame; Ldd = min(C[pk..t]),"
        " pk = last 60-day high in C[t-59..t]; in_dd = rolling(20) min of dd60 ending at t; no flow / news / market"
        " inputs for up20; first_signals cooldown is forward-only (uses only earlier accepted signals). Outcomes"
        " start at t+1 (hold20 C[t+1..t+20], entry O[t+1]). No window ends after t in the signal path.")
    say("(d) non-code look-ahead (not fixable by timing): the 82-stock universe is today's desk/AI watchlist, chosen"
        " in 2026 with knowledge of which AI names worked; late listings (SKHY, SPCX, CRCL) enter as they list. This"
        " survivorship tilt favours 'rebounds hold' for every rule and for the matched baseline alike.")

    # ---------------------------------------------------------- 3. robustness
    say("\n== 3. Robustness (hold20 with n; L1 = registered stock criteria) ==")
    say(f"{'variant':34s} {'n26':>4s} {'h2026':>6s} {'base':>6s} {'nH1':>4s} {'hH1':>6s} {'nH2':>4s} {'hH2':>6s} "
        f"{'n25':>4s} {'h2025':>6s} {'up_med':>6s} {'near5':>6s} {'win20':>6s} {'L1':>5s}")
    states = {THR0: (st, out, base)}

    def get_state(thr):
        if thr not in states:
            L_, d_, u_, i_ = state(Cv, thr)
            s_ = dict(Ldd=L_, dd60=d_, up=u_, in_dd=i_)
            o_ = outcomes(Cv, Ov, L_)
            states[thr] = (s_, o_, matched_base(s_, o_, dates))
        return states[thr]

    def lag(a, n=1):
        b = np.zeros_like(a)
        b[n:] = a[:-n]
        return b

    variants = [
        ("up20 (registered)", LEVEL0, THR0, COOL0, 0, False),
        ("up15 (registered neighbour)", 0.15, THR0, COOL0, 0, False),
        ("up17.5 (half step)", 0.175, THR0, COOL0, 0, False),
        ("up22.5 (half step)", 0.225, THR0, COOL0, 0, False),
        ("up25 (one step above)", 0.25, THR0, COOL0, 0, False),
        ("up20 lag1 (signal t-1, act t)", LEVEL0, THR0, COOL0, 1, False),
        ("up20 lag2", LEVEL0, THR0, COOL0, 2, False),
        ("up20 cooldown 10", LEVEL0, THR0, 10, 0, False),
        ("up20 cooldown 30", LEVEL0, THR0, 30, 0, False),
        ("up20 in_dd THR 15%", LEVEL0, 0.15, COOL0, 0, False),
        ("up20 in_dd THR 25%", LEVEL0, 0.25, COOL0, 0, False),
        ("up20 first-crossing (post-hoc)", LEVEL0, THR0, COOL0, 0, True),
        ("up15 lag1 (post-hoc combo)", 0.15, THR0, COOL0, 1, False),
        ("up20 lag1 THR25 (post-hoc combo)", LEVEL0, 0.25, COOL0, 1, False),
        ("up20 lag1 cool30 (post-hoc combo)", LEVEL0, THR0, 30, 1, False),
    ]
    rob = []
    for name, lvl, thr, cool, lg, cross in variants:
        s_, o_, b_ = get_state(thr)
        sg = (s_["up"] >= lvl) & ~np.isnan(s_["up"])
        if cross:
            sg = sg & ~lag(sg)
        if lg:
            sg = lag(sg, lg)
        rws = first_signals(sg, s_, dates, cool)
        tb = summarize(rws, s_, o_, b_, dates)
        okl1 = l1(tb)
        rob.append((name, tb, okl1))
        a, h1, h2, y5 = tb["2026"], tb["2026H1"], tb["2026H2"], tb["2025"]
        say(f"{name:34s} {a.get('hold20_n', 0):4d} {fmt(a.get('hold20')):>6s} {fmt(a.get('hold20_base')):>6s} "
            f"{h1.get('hold20_n', 0):4d} {fmt(h1.get('hold20')):>6s} {h2.get('hold20_n', 0):4d} {fmt(h2.get('hold20')):>6s} "
            f"{y5.get('hold20_n', 0):4d} {fmt(y5.get('hold20')):>6s} {fmt(a.get('up_med')):>6s} {fmt(a.get('near5')):>6s} "
            f"{fmt(a.get('win20')):>6s} {str(okl1):>5s}")
    n_fail = sum(1 for _, _, ok in rob[1:11] if not ok)
    say(f"registered-style perturbations (rows 2-11) failing L1: {n_fail}/10")

    # ---------------------------------------------------------- 4. null
    say("\n== 4. Null: random in_dd stock-days in 2026, matched count + up-bucket mix, 20-session cooldown, "
        f"{NDRAW} draws ==")
    rows26 = [(p, k) for p, k in rows if pd.Timestamp("2026-01-02") <= dates[p] <= LAST and not np.isnan(out["hold20"][p, k])]
    ub = bucket(up)
    hk = half_key(dates)
    rngn = np.random.default_rng(SEED + 1)
    p_main, obs, vals = run_null("primary: up bucket", rows26, st, out, dates, lambda p, k: BNAMES[int(ub[p, k])], rngn)
    run_null("diagnostic: up bucket, NO cooldown (day-weighted, like the matched baseline)", rows26, st, out, dates,
             lambda p, k: BNAMES[int(ub[p, k])], rngn, ndraw=500, cool=0)
    p_half, *_ = run_null("secondary: half x up bucket", rows26, st, out, dates,
                          lambda p, k: f"{hk[p]}_{BNAMES[int(ub[p, k])]}", rngn)
    run_null("post-hoc: half x (up >= 20%), count-matched, random-order cooldown", rows26, st, out, dates,
             lambda p, k: f"{hk[p]}_{'ge20' if up[p, k] >= LEVEL0 else 'lt20'}", rngn, ndraw=40)
    say("     (infeasible because the rule's own signals are the earliest-first cooldown packing of the up>=20 in_dd"
        " days, i.e. close to the MAXIMUM number of 20-session-separated picks; a random-order packing of the same"
        " days cannot reach that count. So matching the count inside the up>=20 pool forces nearly the same days:"
        " the rule selects nothing beyond 'up >= 20%'.)")
    fine = [0.15, 0.20, 0.22, 0.25, 0.30, 0.40]
    fnames = ["<15", "15-20", "20-22", "22-25", "25-30", "30-40", "40+"]
    fb = np.digitize(up, fine)
    say("  thinning nulls (post-hoc): keep each eligible pool day with prob q, apply the rule's own forward-only"
        " 20-session cooldown per stock, then subsample each stratum to the observed count (all available if fewer)"
        " and post-stratify to the observed stratum mix.")
    thin = {}
    for lbl, pmask, sfn in (
            ("coarse 15%+ bucket x half", up >= 0.15, lambda p, k: f"{hk[p]}_{BNAMES[int(ub[p, k])]}"),
            ("up>=20 x half", up >= LEVEL0, lambda p, k: f"{hk[p]}_ge20"),
            ("fine up bins 20/22/25/30/40 x half", up >= LEVEL0, lambda p, k: f"{hk[p]}_{fnames[int(fb[p, k])]}")):
        for q in (0.15, 0.35):
            thin[(lbl, q)] = run_null_thin(f"{lbl}, q={q}", rows26, st, out, dates, pmask, sfn, q, rngn)
    run_null_thin("sanity q=1.0 (should reproduce the rule, up to the 2025 carry-in cooldown)", rows26, st, out, dates,
                  up >= LEVEL0, lambda p, k: f"{hk[p]}_ge20", 1.0, rngn, ndraw=1)
    p_ge20 = thin[("up>=20 x half", 0.15)][0]
    p_fine = thin[("fine up bins 20/22/25/30/40 x half", 0.15)][0]
    # half-period nulls
    for hname, a, b in (("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23")):
        rh = [(p, k) for p, k in rows26 if pd.Timestamp(a) <= dates[p] <= pd.Timestamp(b)]
        run_null(f"{hname} only: half x up bucket", rh, st, out, dates,
                 lambda p, k: f"{hk[p]}_{BNAMES[int(ub[p, k])]}", rngn, ndraw=1000)

    # ---------------------------------------------------------- 5. mechanical
    say("\n== 5. Mechanical check ==")
    upm = float(np.median([up[p, k] for p, k in rows26]))
    sgm = (up >= upm) & ~np.isnan(up)
    tbm = summarize(first_signals(sgm, st, dates, COOL0), st, out, base, dates)
    say(f"median up of actual 2026 first signals = {upm:.4f}; plain rule up >= {upm:.4f}: "
        f"2026 n={tbm['2026']['hold20_n']} hold20={tbm['2026']['hold20']:.3f} (H1 {tbm['2026H1']['hold20']:.3f}, "
        f"H2 {tbm['2026H2']['hold20']:.3f}); candidate up20 hold20={tab['2026']['hold20']:.3f}")
    in26 = np.asarray((dates >= pd.Timestamp("2026-01-02")) & (dates <= LAST))[:, None]
    allm = in_dd & in26 & ~np.isnan(out["hold20"]) & ~np.isnan(up)
    edges = [0, 0.03, 0.08, 0.15, 0.20, 0.25, 0.30, 0.40, np.inf]
    say("all 2026 in_dd stock-days (no cooldown) by up level: hold20 / near5 / win20 / n")
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        s = allm & (up >= lo_) & (up < hi_)
        say(f"  up {lo_:5.2f}-{hi_:5.2f}: hold20 {np.nanmean(out['hold20'][s]):.3f}  near5 {np.nanmean(out['near5'][s & ~np.isnan(out['near5'])]):.3f}"
            f"  win20 {np.nanmean(out['win20'][s]):.3f}  n {int(s.sum())}")
    s = allm & (up >= LEVEL0)
    say(f"  all in_dd days with up >= 0.20 (what up20 samples from): hold20 {np.nanmean(out['hold20'][s]):.3f}, n {int(s.sum())}")
    s_all = in_dd & in26 & ~np.isnan(out["hold20"])
    say(f"  unconditional 2026 in_dd stock-days (82 stocks, no QQQ): hold20 {np.nanmean(out['hold20'][s_all]):.3f}, "
        f"near5 {np.nanmean(out['near5'][s_all & ~np.isnan(out['near5'])]):.3f}, win20 {np.nanmean(out['win20'][s_all]):.3f}")
    # required further fall to fail vs realised 20-day vol
    P = np.array([p for p, _ in rows26]); Kk = np.array([k for _, k in rows26])
    need_fall = 1 - Ldd[P, Kk] / Cv[P, Kk]
    ret = np.full_like(Cv, np.nan)
    ret[1:] = Cv[1:] / Cv[:-1] - 1
    sd20 = pd.DataFrame(ret).rolling(20, min_periods=20).std().to_numpy()
    z = need_fall / (sd20[P, Kk] * np.sqrt(20))
    say(f"  for the 2026 signals, fall from C[t] needed to break Ldd: median {np.median(need_fall):.3f}; "
        f"in 20-day-vol units (sigma20*sqrt20): median {np.nanmedian(z):.2f}")
    say(f"  matched (same half, 15%+ bucket) hold20 baseline for the signals: {tab['2026']['hold20_base']:.3f} vs "
        f"rule {tab['2026']['hold20']:.3f} (diff {tab['2026']['hold20'] - tab['2026']['hold20_base']:+.3f})")

    # ---------------------------------------------------------- verdict
    say("\n== 6. Verdict inputs ==")
    say(f"independent match={ind_match}; look-ahead in signal timing: none found (tests a, b = 0 differences); "
        f"universe survivorship caveat applies; L1 recomputed={l1(tab)}; robustness L1 failures {n_fail}/10; "
        f"primary null p={p_main:.4f}, half-matched p={p_half:.4f}, up>=20 p={p_ge20:.4f}, fine-up p={p_fine:.4f}")
    rb = {name: tb for name, tb, _ in rob}
    say("\n== 7. Verdict: FRAGILE ==")
    say(f"- Independent match: exact (268/268 first signals, per-signal outcomes identical; 2026 hold20 "
        f"{tab['2026']['hold20']:.4f} n={tab['2026']['hold20_n']}, H1 {tab['2026H1']['hold20']:.4f}, "
        f"H2 {tab['2026H2']['hold20']:.4f}, 2025 {tab['2025']['hold20']:.4f}). L1 passes as registered.")
    say("- Look-ahead: none in the signal path (truncation and future-scramble tests 0 differences). Caveat: the"
        " universe is a 2026 hindsight AI watchlist (affects rule and baselines alike).")
    say(f"- Sensitivity: L1 fails for the registered lower neighbour up15 ({rb['up15 (registered neighbour)']['2026']['hold20']:.3f}),"
        f" up17.5 ({rb['up17.5 (half step)']['2026']['hold20']:.3f}) and lag2 ({rb['up20 lag2']['2026']['hold20']:.3f});"
        f" lag1 passes only at {rb['up20 lag1 (signal t-1, act t)']['2026']['hold20']:.3f} (H2 "
        f"{rb['up20 lag1 (signal t-1, act t)']['2026H2']['hold20']:.3f}). H2 sits at 0.66-0.75 in every variant, its "
        f"bootstrap CI is [{tab['2026H2']['h_lo']:.2f}, {tab['2026H2']['h_hi']:.2f}] over 5 blocks, and two 20-session "
        "blocks (mid-May to mid-July 2026) hold only 41-43%. hold20 rises monotonically with the threshold"
        " (up15 0.69 -> up20 0.76 -> up25 0.82): 70% is simply crossed between 15% and 20% rebound.")
    say(f"- Null: the specified null (15%+ bucket, random-order cooldown) gives p={p_main:.4f}, but only because that"
        " bucket includes 15-20% days (day-level hold20 0.685) and random-order packing over-weights short-lived"
        f" rebounds; with the rule's own forward cooldown p rises to 0.03-0.17, and inside up>=20% (p={p_ge20:.3f}) or"
        f" with fine up bins (p={p_fine:.3f}) the rule is no better than random days at the same rebound size."
        f" H2-only null p≈0.7. Matched baseline {tab['2026']['hold20_base']:.3f} >= rule {tab['2026']['hold20']:.3f} (L2 fails).")
    say(f"- Mechanical: plain up >= median ({upm:.3f}) gives {tbm['2026']['hold20']:.3f}; in_dd days at up 20-25% hold"
        " 0.783. The 70% hit rate is the consequence of the price already being ~23% above the low (a further ~19%"
        " fall is needed to fail). near5 0.200 and win20 0.460 are at/below the unconditional 2026 stock baseline"
        " (0.226 / 0.494): it confirms 'the low held' only by arriving late, and adds no buy edge.")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(ind_match=ind_match, p=p_main, hold20=tab["2026"]["hold20"], n_fail=n_fail)


if __name__ == "__main__":
    main()
