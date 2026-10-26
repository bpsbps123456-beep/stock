"""Independent adversarial verification: family f2, rule ma20x2, kind SOXX (bottom_confirm, BRIEF.md 2026-10-26).

Does NOT import bc_lib. Re-implements from the raw CSVs:
  pk[t]   latest position of max(C[t-59..t]);  Ldd[t] = min(C[pk..t]);  dd60 = C/max60 - 1;  up = C/Ldd - 1
  in_dd   min(dd60[t-19..t]) <= -THR (SOXX 10%)
  hold20  min(C[t+1..t+20]) >= Ldd[t];  near5 min(C[t+1..t+40]) >= 0.95*O[t+1]
  r20     C[t+20]/O[t+1]-1, win20 = r20 > 0;  r60 = C[t+60]/O[t+1]-1
  first signals: sig & in_dd & Ldd defined & 2024-01-02 <= t <= 2026-10-23, then accept p if p - last_accepted > 20
  matched base: mean outcome over in_dd days (t >= 2024-01-02) in the same calendar year and same up bucket
  ma20x2: C[t] > SMA20[t] & C[t-1] > SMA20[t-1] & C[t-2] <= SMA20[t-2]

Run from data/derived/analysis/bottom_confirm:  python3 -W ignore code/verify_f2_SOXX_ma20x2.py
Writes results/verify_f2_SOXX_ma20x2.txt
"""
from __future__ import annotations

import pathlib

import exchange_calendars as xc
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(__file__).resolve().parents[5]
RES = HERE / "results"
OUT = RES / "verify_f2_SOXX_ma20x2.txt"

PRICES = ROOT / "research/news_turning_points_20261025/prices.csv"
CLOSE23 = ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv"

LAST = pd.Timestamp("2026-10-23")
EVAL_START = pd.Timestamp("2024-01-02")
PERIODS = [("2024~2026", "2024-01-02", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
           ("2025", "2025-01-02", "2025-12-31"), ("2024", "2024-01-02", "2024-12-31")]
BINS = [0.03, 0.08, 0.15]
FAMILY = {"rule": "ma20x2", "n_main": 12, "hold20_main": 0.8333, "hold20_base_main": 0.8462, "hold20_2025": 0.75,
          "hold20_2026": 0.75, "hold20_2026H1": 0.0, "hold20_2026H2": 1.0, "near5_main": 0.2727,
          "win20_main": 0.5833, "r20_main": 0.0363, "up_median_main": 0.1042}

lines: list[str] = []


def say(s=""):
    print(s)
    lines.append(s)


# ------------------------------------------------------------------ raw data
def load():
    p = pd.read_csv(PRICES, header=[0, 1], index_col=0)
    p.index = pd.to_datetime(p.index, errors="coerce")
    p = p[p.index.notna()].sort_index()
    s = pd.DataFrame({k: p[(k, "SOXX")] for k in ("open", "close")}).dropna()
    u = pd.read_csv(CLOSE23, encoding="utf-8-sig", index_col=0, parse_dates=True)["SOXX"].dropna()
    cal = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range("2023-01-01", "2026-10-23"))
    cal = cal.tz_localize(None) if cal.tz is not None else cal
    C = pd.Series(np.nan, index=cal)
    C.loc[u.index.intersection(cal)] = u.loc[u.index.intersection(cal)]
    ov = s.index.intersection(u.index)
    diff = float(np.max(np.abs(s.close.loc[ov].to_numpy() - u.loc[ov].to_numpy())))
    C.loc[s.index] = s.close
    O = s.open.reindex(cal)
    union = s.index.union(u.index)
    miss = len(cal.difference(union))
    extra = len(union.difference(cal))
    assert C.notna().all()
    assert O.loc["2023-12-01":].notna().all()
    return C, O, diff, miss, extra


# ------------------------------------------------------------------ state + outcomes
def state(C: np.ndarray, thr: float):
    N = len(C)
    Ldd = np.full(N, np.nan)
    LL = np.full(N, -1)
    dd60 = np.full(N, np.nan)
    for t in range(59, N):
        w = C[t - 59:t + 1]
        mx = w.max()
        pk = t - 59 + int(np.flatnonzero(w == mx)[-1])        # latest position of the max
        seg = C[pk:t + 1]
        Ldd[t] = seg.min()
        LL[t] = pk + int(np.flatnonzero(seg == seg.min())[-1])
        dd60[t] = C[t] / mx - 1
    in_dd = np.zeros(N, bool)
    for t in range(N):
        w = dd60[max(0, t - 19):t + 1]
        w = w[~np.isnan(w)]
        in_dd[t] = len(w) > 0 and w.min() <= -thr
    up = C / Ldd - 1
    return Ldd, LL, dd60, in_dd, up


def outcomes(C: np.ndarray, O: np.ndarray, Ldd: np.ndarray):
    N = len(C)
    h20 = np.full(N, np.nan)
    h40 = np.full(N, np.nan)
    n5 = np.full(N, np.nan)
    r20 = np.full(N, np.nan)
    r60 = np.full(N, np.nan)
    for t in range(N):
        if t + 20 < N and not np.isnan(Ldd[t]):
            h20[t] = float(C[t + 1:t + 21].min() >= Ldd[t] - 1e-12)
        if t + 40 < N and not np.isnan(Ldd[t]):
            h40[t] = float(C[t + 1:t + 41].min() >= Ldd[t] - 1e-12)
        if t + 1 < N and not np.isnan(O[t + 1]):
            if t + 40 < N:
                n5[t] = float(C[t + 1:t + 41].min() >= 0.95 * O[t + 1])
            if t + 20 < N:
                r20[t] = C[t + 20] / O[t + 1] - 1
            if t + 60 < N:
                r60[t] = C[t + 60] / O[t + 1] - 1
    win20 = np.where(np.isnan(r20), np.nan, (r20 > 0).astype(float))
    return dict(hold20=h20, hold40=h40, near5=n5, win20=win20, r20=r20, r60=r60)


def bucket(up):
    b = np.digitize(up, BINS).astype(float)
    b[np.isnan(up)] = np.nan
    return b


def matched_base(idx, in_dd, up, out, key="hold20"):
    """Same calendar year x same up bucket mean over in_dd days with t >= EVAL_START."""
    b = bucket(up)
    yr = np.asarray(idx.year)
    y = out[key]
    pool = in_dd & ~np.isnan(up) & (idx >= EVAL_START) & ~np.isnan(y)
    tab = {}
    for yy in np.unique(yr[pool]):
        for bb in range(4):
            m = pool & (yr == yy) & (b == bb)
            if m.any():
                tab[(yy, bb)] = float(y[m].mean())
    base = np.array([tab.get((yr[t], b[t]), np.nan) if not np.isnan(b[t]) else np.nan for t in range(len(idx))])
    return base


# ------------------------------------------------------------------ signals
def sma(C: np.ndarray, n: int):
    s = pd.Series(C).rolling(n, min_periods=n).mean().to_numpy()
    return s


def ma_reclaim_k(C: np.ndarray, n: int, k: int):
    """k consecutive closes above SMA_n (t-k+1..t) and the close before that at or below. k=1 is ma20x, k=2 ma20x2."""
    s = sma(C, n)
    N = len(C)
    out = np.zeros(N, bool)
    for t in range(k, N):
        if np.isnan(s[t - k]):
            continue
        above = all(C[t - j] > s[t - j] for j in range(k))
        out[t] = above and C[t - k] <= s[t - k]
    return out


def first_signals(sig, idx, in_dd, Ldd, cool=20):
    ok = sig & in_dd & ~np.isnan(Ldd) & (idx >= EVAL_START) & (idx <= LAST)
    acc, prev = [], -10 ** 9
    for p in np.flatnonzero(ok):
        if p - prev > cool:
            acc.append(p)
            prev = p
    return np.array(acc, int)


def summarize(pos, idx, up, out, base):
    rows = {}
    for name, a, b in PERIODS:
        m = (idx[pos] >= pd.Timestamp(a)) & (idx[pos] <= pd.Timestamp(b)) if len(pos) else np.zeros(0, bool)
        pp = pos[m]
        h = out["hold20"][pp]
        bb = base[pp]
        ok = ~np.isnan(h) & ~np.isnan(bb)
        rows[name] = dict(n=int(ok.sum()), hold20=float(h[ok].mean()) if ok.any() else np.nan,
                          base=float(bb[ok].mean()) if ok.any() else np.nan,
                          up_med=float(np.median(up[pp])) if len(pp) else np.nan)
        for k in ("near5", "win20", "r20", "hold40"):
            v = out[k][pp]
            rows[name][k] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    return rows


def L1(rows):
    m, y26, y25 = rows["2024~2026"], rows["2026"], rows["2025"]
    c1 = m["n"] >= 8 and m["hold20"] >= 0.70
    c2 = y26["n"] >= 3 and y26["hold20"] >= 0.70
    c3 = y25["n"] < 3 or y25["hold20"] >= 0.60
    return bool(c1 and c2 and c3)


def fmt_rows(label, rows):
    def f(p):
        r = rows[p]
        return "  -  (0)" if r["n"] == 0 else f"{r['hold20']:.3f}({r['n']})"
    return (f"  {label:34s} main {f('2024~2026'):11s} 2026 {f('2026'):10s} H1 {f('2026H1'):10s} H2 {f('2026H2'):10s} "
            f"2025 {f('2025'):10s} 2024 {f('2024'):10s} upmed {rows['2024~2026']['up_med']:.3f} "
            f"base {rows['2024~2026']['base']:.3f}")


def block_boot(v, pos, B=5000, seed=20261026):
    lab = np.asarray(pos) // 20
    u, inv = np.unique(lab, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    cnt = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(u), size=(B, len(u)))
    m = sums[pick].sum(1) / cnt[pick].sum(1)
    return np.quantile(m, [0.025, 0.975])


# ------------------------------------------------------------------ null
def null_draws(pool_pos, pool_b, need_b, y, rng, cool=20, draws=2000, strata=None, need_strata=None):
    """Random dates from pool, matched on bucket (and optional strata), pairwise spacing > cool."""
    res, attempts = [], 0
    keys = list(zip(need_b, need_strata)) if strata is not None else [(b, None) for b in need_b]
    pools = {}
    for kk in set(keys):
        m = pool_b == kk[0]
        if strata is not None:
            m &= strata == kk[1]
        pools[kk] = pool_pos[m]
    while len(res) < draws and attempts < draws * 200:
        attempts += 1
        order = rng.permutation(len(keys))
        chosen = []
        fail = False
        for i in order:
            cand = pools[keys[i]]
            if chosen:
                ch = np.array(chosen)
                cand = cand[np.all(np.abs(cand[:, None] - ch[None, :]) > cool, axis=1)]
            if len(cand) == 0:
                fail = True
                break
            chosen.append(int(rng.choice(cand)))
        if fail:
            continue
        res.append(float(np.mean(y[chosen])))
    return np.array(res), attempts


# ------------------------------------------------------------------ main
def main():
    Cs, Os, cdiff, miss, extra = load()
    idx = Cs.index
    C, O = Cs.to_numpy(float), Os.to_numpy(float)
    N = len(C)
    say("=" * 110)
    say("Independent verification: family f2, rule ma20x2, kind SOXX  (no bc_lib import)")
    say("=" * 110)
    say(f"Data: SOXX close {idx[0].date()}..{idx[-1].date()}  N={N}; open from 2023-12-01")
    say(f"  overlap max |us_prices_close - prices.csv| = {cdiff:.2e}; XNYS sessions missing={miss} extra={extra}")

    Ldd, LL, dd60, in_dd, up = state(C, 0.10)
    out = outcomes(C, O, Ldd)
    base = matched_base(idx, in_dd, up, out)
    sig = ma_reclaim_k(C, 20, 2)
    # cross-check with a pandas one-liner of the brief's definition
    s20 = pd.Series(C).rolling(20, min_periods=20).mean()
    Cp = pd.Series(C)
    sig2 = ((Cp > s20) & (Cp.shift(1) > s20.shift(1)) & (Cp.shift(2) <= s20.shift(2))).to_numpy()
    assert (sig == sig2).all()
    pos = first_signals(sig, idx, in_dd, Ldd)
    rows = summarize(pos, idx, up, out, base)

    # ---------------------------------------------------------- [1] match
    say("\n[1] Independent re-implementation vs family files")
    fam = pd.read_csv(RES / "f2_SOXX_signals.csv", parse_dates=["date"])
    fam = fam[fam.rule == "ma20x2"].reset_index(drop=True)
    famsum = pd.read_csv(RES / "f2_SOXX_summary.csv")
    famsum = famsum[famsum.rule == "ma20x2"].set_index("period")
    mine = pd.DataFrame(dict(date=idx[pos], pos=pos, up=up[pos], hold20=out["hold20"][pos], base_hold20=base[pos],
                             near5=out["near5"][pos], win20=out["win20"][pos], r20=out["r20"][pos],
                             Ldd=Ldd[pos], dd60=dd60[pos],
                             m_t=C[pos] / s20.to_numpy()[pos] - 1, m_t2=C[pos - 2] / s20.to_numpy()[pos - 2] - 1))
    say(mine.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    same_dates = list(mine.date) == list(fam.date)
    say(f"  first-signal dates identical to family: {same_dates}  (mine {len(mine)}, family {len(fam)})")
    if not same_dates:
        say(f"    only mine: {sorted(set(mine.date) - set(fam.date))}")
        say(f"    only family: {sorted(set(fam.date) - set(mine.date))}")
    mism = []
    if same_dates:
        for col in ("pos", "up", "hold20", "base_hold20", "near5", "win20", "r20", "dd60"):
            a, b = mine[col].to_numpy(float), fam[col].to_numpy(float)
            bad = ~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-9))
            if bad.any():
                mism.append(f"{col}@{list(mine.date[bad].dt.date)}")
    say(f"  per-signal field mismatches: {'none' if not mism else '; '.join(mism)}")
    say("  period      mine hold20 (n)   family hold20 (n)   mine base   family base")
    per_ok = True
    for p, _, _ in PERIODS:
        r = rows[p]
        fh, fn, fb = famsum.loc[p, "hold20"], famsum.loc[p, "hold20_n"], famsum.loc[p, "hold20_base"]
        ok = r["n"] == fn and abs(r["hold20"] - fh) < 1e-9 and abs(r["base"] - fb) < 1e-9
        per_ok &= bool(ok)
        say(f"  {p:10s}  {r['hold20']:.4f} ({r['n']:2d})      {fh:.4f} ({int(fn):2d})          {r['base']:.4f}      "
            f"{fb:.4f}   {'OK' if ok else 'MISMATCH'}")
    m = rows["2024~2026"]
    rep_ok = (abs(m["hold20"] - FAMILY["hold20_main"]) < 5e-5 and m["n"] == FAMILY["n_main"]
              and abs(m["near5"] - FAMILY["near5_main"]) < 5e-5 and abs(m["win20"] - FAMILY["win20_main"]) < 5e-5
              and abs(m["r20"] - FAMILY["r20_main"]) < 5e-5 and abs(m["up_med"] - FAMILY["up_median_main"]) < 5e-5)
    match = bool(same_dates and not mism and per_ok and rep_ok)
    say(f"  family-reported numbers (n, hold20, near5, win20, r20, up_median) reproduced: {rep_ok}")
    say(f"  => independent match: {match}")
    say(f"  main: n={m['n']} hold20={m['hold20']:.4f} base={m['base']:.4f} near5={m['near5']:.4f} "
        f"win20={m['win20']:.4f} r20={m['r20']:.4f} up_med={m['up_med']:.4f}")
    v = out["hold20"][pos] - base[pos]
    ok = ~np.isnan(v)
    lo, hi = block_boot(v[ok], pos[ok])
    say(f"  my 20-session block bootstrap of (hold20 - matched base), main: diff={v[ok].mean():+.4f} "
        f"95% [{lo:+.3f}, {hi:+.3f}]  (family dlo -0.171) -> L2 {'pass' if lo > 0 else 'fail'}")
    pool_main = in_dd & (idx >= EVAL_START) & (idx <= LAST) & ~np.isnan(out["hold20"])
    p0 = float(out["hold20"][pool_main].mean())
    from math import comb
    k_hit, n_ = int(out["hold20"][pos].sum()), len(pos)
    def binom_tail(k, n, p):
        return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))
    say(f"  unconditional in_dd hold20 2024~2026 = {p0:.4f} (days={int(pool_main.sum())}); "
        f"binomial P(X>={k_hit}/{n_} | p={p0:.3f}) = {binom_tail(k_hit, n_, p0):.3f}; "
        f"| p=matched {m['base']:.3f}: {binom_tail(k_hit, n_, m['base']):.3f}")
    # relation to ma20x: ma20x2[t] == ma20x[t-1] & C[t] > SMA20[t]
    s1 = ma_reclaim_k(C, 20, 1)
    rel = np.zeros(N, bool)
    rel[1:] = s1[:-1] & (C[1:] > s20.to_numpy()[1:])
    say(f"  identity check ma20x2[t] == ma20x[t-1] & C[t] > SMA20[t]: {bool((rel == sig).all())} "
        f"(ma20x2 is ma20x delayed one session, keeping only reclaims that survive the next close)")

    # ---------------------------------------------------------- [2] look-ahead
    say("\n[2] Look-ahead audit")
    trunc_bad = []
    check_pos = np.flatnonzero(in_dd & (idx >= EVAL_START))
    for t in check_pos:
        Ct = C[:t + 1]
        st = ma_reclaim_k(Ct, 20, 2)[-1]
        Lt, _, _, it, ut = state(Ct[max(0, t - 120):], 0.10)
        if st != sig[t] or it[-1] != in_dd[t] or abs(Lt[-1] - Ldd[t]) > 1e-12:
            trunc_bad.append(idx[t].date())
    say(f"  truncation test on {len(check_pos)} in_dd days (signal, in_dd, Ldd recomputed from C[..t] only): "
        f"{'all identical' if not trunc_bad else 'DIFF at ' + str(trunc_bad[:10])}")
    say("  code read (code/f2_reclaim.py): ma20x2() = (C > s) & (C.shift(1) > s.shift(1)) & (C.shift(2) <= s.shift(2)),")
    say("  s = C.rolling(20, min_periods=20).mean() -> trailing window ending at t; shifts are positive (past). No")
    say("  market/flow/news input. bc_lib Ldd/dd60/in_dd use windows ending at t; outcomes start at t+1, entry O[t+1].")
    say("  The matched baseline averages same-year outcomes (evaluation reference only, not a signal input).")
    mg = np.minimum(np.abs(mine.m_t.to_numpy()), np.abs(mine.m_t2.to_numpy()))
    say(f"  margins |C/SMA20-1| at t and t-2 across signals: min {mg.min()*100:.2f}% "
        f"(dividend back-adjustment moves the ratio by <~0.2% only inside ex-date windows)")
    tight = mine.loc[mg < 0.002, "date"]
    say(f"  signals with a margin < 0.2% (adjustment-sensitive): {[d.date() for d in tight] if len(tight) else 'none'}")

    # ---------------------------------------------------------- [3] robustness
    say("\n[3] Robustness (hold20 (n) per period; main = 2024~2026)")
    variants = {}

    def run(label, n=20, k=2, cool=20, thr=0.10, delay=0):
        if thr != 0.10:
            L_, _, _, ind_, up_ = state(C, thr)
            out_ = outcomes(C, O, L_)
            base_ = matched_base(idx, ind_, up_, out_)
        else:
            L_, ind_, up_, out_, base_ = Ldd, in_dd, up, out, base
        sg = ma_reclaim_k(C, n, k)
        if delay:
            sg = np.r_[np.zeros(delay, bool), sg[:-delay]]
        ps = first_signals(sg, idx, ind_, L_, cool)
        r = summarize(ps, idx, up_, out_, base_)
        variants[label] = (r, ps)
        say(fmt_rows(label, r))
        return r

    run("base n=20 k=2 cool=20 thr=10% d=0")
    say("  -- one step either side --")
    run("SMA n=19", n=19)
    run("SMA n=21", n=21)
    run("days-above k=1 (= ma20x)", k=1)
    run("days-above k=3", k=3)
    run("delay +1 session", delay=1)
    run("cooldown 10", cool=10)
    run("cooldown 30", cool=30)
    run("in_dd thr 8%", thr=0.08)
    run("in_dd thr 12%", thr=0.12)
    say("  -- wider context --")
    for n in (18, 22, 15, 25, 30, 50):
        run(f"SMA n={n}", n=n)
    run("SMA n=10 k=2", n=10)
    run("delay +2 sessions", delay=2)
    core = ["base n=20 k=2 cool=20 thr=10% d=0", "SMA n=19", "SMA n=21", "days-above k=1 (= ma20x)", "days-above k=3",
            "delay +1 session", "cooldown 10", "cooldown 30", "in_dd thr 8%", "in_dd thr 12%"]
    say("  L1 (registered SOXX rule) and main hold20 - matched base, per variant:")
    for lab, (r, _) in variants.items():
        h1 = r["2026H1"]
        h1s = "-" if h1["n"] == 0 else f"{h1['hold20']:.2f}"
        say(f"    {lab:34s} L1={str(L1(r)):5s}  diff={r['2024~2026']['hold20'] - r['2024~2026']['base']:+.3f}  "
            f"2026H1 {h1s}({h1['n']})")
    n_l1 = sum(L1(variants[c][0]) for c in core)
    n_above_base = sum(variants[c][0]["2024~2026"]["hold20"] > variants[c][0]["2024~2026"]["base"] for c in core)
    say(f"  core variants keeping L1: {n_l1}/{len(core)}; core variants with main hold20 > matched base: "
        f"{n_above_base}/{len(core)}")

    # ---------------------------------------------------------- [4] null
    say("\n[4] Null: random in_dd dates in 2024~2026, same count and up-bucket mix, pairwise spacing > 20 sessions")
    obs = rows["2024~2026"]["hold20"]
    b_all = bucket(up)
    need = b_all[pos].astype(int)
    pp = np.flatnonzero(pool_main)
    pb = b_all[pp]
    say(f"  observed hold20 = {obs:.4f}; bucket mix (0-3,3-8,8-15,15+) = {np.bincount(need, minlength=4).tolist()}; "
        f"pool days = {len(pp)} by bucket {np.bincount(pb.astype(int), minlength=4).tolist()}")
    say(f"  pool hold20 by bucket: {[round(float(out['hold20'][pp][pb == b].mean()), 3) for b in range(4)]}")
    rng = np.random.default_rng(20261026)
    nd, att = null_draws(pp, pb, need, out["hold20"], rng)
    p_null = float((nd >= obs - 1e-12).mean())
    say(f"  draws={len(nd)} (attempts {att}); null mean={nd.mean():.4f} sd={nd.std():.4f} "
        f"q05={np.quantile(nd, .05):.3f} q95={np.quantile(nd, .95):.3f}")
    say(f"  p = P(null hold20 >= observed) = {p_null:.4f}")
    yr = np.asarray(idx.year)
    nd2, _ = null_draws(pp, pb, need, out["hold20"], np.random.default_rng(1), strata=yr[pp], need_strata=yr[pos])
    p2 = float((nd2 >= obs - 1e-12).mean())
    say(f"  secondary (year x bucket matched): draws={len(nd2)} mean={nd2.mean():.4f} p={p2:.4f}")
    m26 = idx[pos] >= pd.Timestamp("2026-01-02")
    pool26 = pool_main & (idx >= pd.Timestamp("2026-01-02"))
    pp26 = np.flatnonzero(pool26)
    obs26 = float(out["hold20"][pos[m26]].mean())
    nd3, _ = null_draws(pp26, b_all[pp26], need[m26], out["hold20"], np.random.default_rng(2))
    p3 = float((nd3 >= obs26 - 1e-12).mean())
    say(f"  2026-only null: observed {obs26:.3f} (n={int(m26.sum())}, buckets "
        f"{np.bincount(need[m26], minlength=4).tolist()}), null mean {nd3.mean():.3f}, p={p3:.4f}")

    # ---------------------------------------------------------- [5] mechanical
    say("\n[5] Mechanical comparison: plain rule up >= candidate median up (same in_dd / first-signal / cooldown)")
    med = rows["2024~2026"]["up_med"]
    mech = {}
    for thr_up in (med, 0.05, 0.08, 0.10, 0.15):
        sg = np.nan_to_num(up, nan=-1) >= thr_up
        ps = first_signals(sg, idx, in_dd, Ldd)
        r = summarize(ps, idx, up, out, base)
        mech[thr_up] = r
        say(fmt_rows(f"up >= {thr_up:.4f}", r))
    a = pool_main & (np.nan_to_num(up, nan=-1) >= med)
    bmask = pool_main & (np.nan_to_num(up, nan=-1) < med)
    say(f"  all in_dd days 2024~2026 with up >= {med:.4f} (no cooldown): hold20 = {out['hold20'][a].mean():.4f} "
        f"(days {int(a.sum())}); with up < it: {out['hold20'][bmask].mean():.4f} (days {int(bmask.sum())})")
    say("  within-bucket comparison, 2024~2026 (candidate first signals vs all in_dd pool days):")
    names = ["0-3%", "3-8%", "8-15%", "15%+"]
    for b in range(4):
        cm = need == b
        if cm.any():
            say(f"    bucket {names[b]:6s} cand {out['hold20'][pos[cm]].mean():.3f} (n={int(cm.sum())})   "
                f"pool {out['hold20'][pp][pb == b].mean():.3f} (days={int((pb == b).sum())})   "
                f"cand dates {[str(idx[q].date()) for q in pos[cm]]}")
    raw = sig & pool_main
    say(f"  raw ma20x2 in_dd days (no cooldown): hold20 = {out['hold20'][raw].mean():.4f} (days {int(raw.sum())}), "
        f"matched base avg {np.nanmean(base[raw]):.4f}")
    fails = pos[out["hold20"][pos] == 0]
    for q in fails:
        brk = q + 1 + int(np.argmax(C[q + 1:q + 21] < Ldd[q] - 1e-12))
        say(f"  failure {idx[q].date()}: up={up[q]*100:.1f}% Ldd={Ldd[q]:.2f}, broken {idx[brk].date()} "
            f"(C={C[brk]:.2f}), new low {C[q + 1:q + 21].min():.2f} on {idx[q + 1 + int(np.argmin(C[q + 1:q + 21]))].date()}")

    # ---------------------------------------------------------- [6] verdict
    say("\n[6] Verdict inputs")
    rb = variants
    say(f"  independent match={match}, look-ahead found={bool(trunc_bad)}, main hold20={obs:.4f} vs matched base "
        f"{rows['2024~2026']['base']:.4f} (diff {obs - rows['2024~2026']['base']:+.4f}), null p={p_null:.4f} "
        f"(year x bucket p={p2:.4f}; 2026-only p={p3:.4f})")
    say("  L1 per core variant: " + ", ".join(f"{c}={L1(rb[c][0])}" for c in core))
    say(f"  plain up>={med:.4f} rule hold20 = {mech[med]['2024~2026']['hold20']:.4f} "
        f"(n={mech[med]['2024~2026']['n']}); 2026H1 candidate hold20 = {rows['2026H1']['hold20']:.2f} "
        f"(n={rows['2026H1']['n']})")

    lookahead = bool(trunc_bad)
    if not match or lookahead:
        verdict = "refuted"
    elif p_null < 0.05 and n_l1 == len(core) and obs > rows["2024~2026"]["base"]:
        verdict = "confirmed"
    elif obs <= rows["2024~2026"]["base"] or p_null >= 0.5:
        verdict = "refuted"
    else:
        verdict = "fragile"
    say(f"\nVERDICT: {verdict}")
    say("  Reproduced exactly (12 first signals, all per-signal fields and per-period rates), no look-ahead.")
    say(f"  But 83.3% is not above the same-state baseline ({rows['2024~2026']['base']*100:.1f}%; L2 fails), the")
    say(f"  random matched-date null gives mean {nd.mean()*100:.1f}% with p={p_null:.2f}, and the plain rule")
    fb = sorted({names[int(b_all[q])] for q in fails})
    pool_hi = [float(out["hold20"][pp][pb == b].mean()) for b in (2, 3)]
    c38 = need == 1
    say(f"  'up >= {med*100:.1f}%' scores {mech[med]['2024~2026']['hold20']*100:.0f}%. Failures sit in bucket(s) {fb}; in the")
    say(f"  3-8% bucket ma20x2 holds {out['hold20'][pos[c38]].mean()*100:.0f}% (n={int(c38.sum())}) vs the pool's "
        f"{out['hold20'][pp][pb == 1].mean()*100:.1f}%, while the 8-15% / 15%+ pool")
    say(f"  days already hold {pool_hi[0]*100:.1f}% / {pool_hi[1]*100:.1f}% with no signal. The only 2026H1 signal "
        f"(2026-03-25) fired before the")
    say(f"  03-30 low and failed (2026H1 0/1). L1 holds in {n_l1}/{len(core)} core variants (breaks with cooldown 10, "
        f"cooldown 30, in_dd 8%);")
    say(f"  main hold20 exceeds the matched base in only {n_above_base}/{len(core)} core variants. The hit rate is "
        f"explained by how far price")
    say("  had already rebounded (up), not by the two-day SMA20 reclaim.")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(verdict=verdict, match=match, lookahead=lookahead, p=p_null, obs=obs)


if __name__ == "__main__":
    main()
