"""Independent adversarial verification of bottom_confirm family f1, rule sig2, kind SOXX.

Rule (BRIEF.md F1): up[t] >= 2 * sigma20[t], sigma20 = std(ddof=1) of daily close-to-close returns over the
20 sessions ending at t. First signals inside in_dd (SOXX THR 10%), evaluation window 2024-01-02~2026-10-23,
20-session cooldown. Main outcome hold20 = min(C[t+1..t+20]) >= Ldd[t].

This file does NOT import bc_lib. It re-reads the raw CSVs and recomputes everything with plain numpy loops,
then compares with results/f1_SOXX_signals.csv and results/f1_SOXX_summary.csv (rule == 'sig2').

Run: cd data/derived/analysis/bottom_confirm && python3 -W ignore code/verify_f1_SOXX_sig2.py
Output: results/verify_f1_SOXX_sig2.txt
"""
from __future__ import annotations

import io
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(__file__).resolve().parents[5]
RES = HERE / "results"
OUT = RES / "verify_f1_SOXX_sig2.txt"

RULE, M_MAIN = "sig2", 2.0
EVAL_A, EVAL_B = pd.Timestamp("2024-01-02"), pd.Timestamp("2026-10-23")
PERIODS = [("2024~2026", "2024-01-02", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
           ("2025", "2025-01-02", "2025-12-31"), ("2024", "2024-01-02", "2024-12-31")]
BUCKET_EDGES = [0.03, 0.08, 0.15]          # 0-3, 3-8, 8-15, 15+
BUCKET_NAMES = ["0-3%", "3-8%", "8-15%", "15%+"]
NDRAW = 2000
SEED = 7_2026_10_26

buf = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    buf.write(s + "\n")


# ------------------------------------------------------------------ raw data (no bc_lib)
def load_raw():
    p = pd.read_csv(ROOT / "research/news_turning_points_20261025/prices.csv", header=[0, 1], index_col=0)
    p.index = pd.to_datetime(p.index, errors="coerce")
    p = p[p.index.notna()].sort_index()
    pc = p[("close", "SOXX")].astype(float).dropna()
    po = p[("open", "SOXX")].astype(float).dropna()
    u = pd.read_csv(ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv",
                    encoding="utf-8-sig", index_col=0, parse_dates=True)["SOXX"].astype(float).dropna()
    idx = u.index.union(pc.index).sort_values()
    idx = idx[idx <= EVAL_B]
    C = u.reindex(idx)
    C.loc[pc.index] = pc
    O = po.reindex(idx)
    ov = u.index.intersection(pc.index)
    info = dict(n=len(idx), first=idx[0].date(), last=idx[-1].date(), overlap=len(ov),
                overlap_maxdiff=float(np.abs(u[ov] - pc[ov]).max()), c_nan=int(C.isna().sum()),
                o_nan_after_2023_12=int(O.loc["2023-12-01":].isna().sum()))
    try:  # calendar cross-check (independent package, not bc_lib)
        import exchange_calendars as xc
        cal = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range("2023-01-01", "2026-10-23"))
        cal = cal.tz_localize(None) if cal.tz is not None else cal
        info["xnys_equal"] = bool(cal.equals(idx))
    except Exception as e:  # pragma: no cover
        info["xnys_equal"] = f"n/a ({e})"
    return C, O, info


# ------------------------------------------------------------------ core state arrays
def state(c: np.ndarray, thr: float, sig_win: int = 20):
    N = len(c)
    Ldd = np.full(N, np.nan)
    dd60 = np.full(N, np.nan)
    pkpos = np.full(N, -1)
    lowpos = np.full(N, -1)
    for t in range(59, N):
        w = c[t - 59:t + 1]
        mx = w.max()
        pk = t - 59 + int(np.flatnonzero(w == mx)[-1])        # latest position of the 60-day max
        seg = c[pk:t + 1]
        Ldd[t] = seg.min()
        lowpos[t] = pk + int(np.flatnonzero(seg == seg.min())[-1])
        pkpos[t] = pk
        dd60[t] = c[t] / mx - 1
    up = c / Ldd - 1
    in_dd = np.zeros(N, bool)
    for t in range(N):
        w = dd60[max(0, t - 19):t + 1]
        w = w[np.isfinite(w)]
        in_dd[t] = len(w) > 0 and w.min() <= -thr
    sig = np.full(N, np.nan)
    for t in range(sig_win, N):
        w = c[t - sig_win:t + 1]
        r = w[1:] / w[:-1] - 1                               # returns of the sig_win sessions ending at t
        sig[t] = np.std(r, ddof=1)
    return dict(Ldd=Ldd, dd60=dd60, up=up, in_dd=in_dd, s20=sig, pk=pkpos, lowpos=lowpos)


def outcomes(c: np.ndarray, o: np.ndarray, Ldd: np.ndarray):
    N = len(c)
    hold20 = np.full(N, np.nan)
    near5 = np.full(N, np.nan)
    r20 = np.full(N, np.nan)
    for t in range(N):
        if t + 20 <= N - 1 and np.isfinite(Ldd[t]):
            hold20[t] = float(c[t + 1:t + 21].min() >= Ldd[t])
        if t + 1 <= N - 1 and np.isfinite(o[t + 1]):
            if t + 40 <= N - 1:
                near5[t] = float(c[t + 1:t + 41].min() >= 0.95 * o[t + 1])
            if t + 20 <= N - 1:
                r20[t] = c[t + 20] / o[t + 1] - 1
    win20 = np.where(np.isfinite(r20), (r20 > 0).astype(float), np.nan)
    return dict(hold20=hold20, near5=near5, r20=r20, win20=win20)


def first_signals(mask: np.ndarray, st: dict, dates: pd.DatetimeIndex, cool: int = 20) -> np.ndarray:
    ok = mask & st["in_dd"] & np.isfinite(st["Ldd"]) & (dates >= EVAL_A) & (dates <= EVAL_B)
    out, prev = [], -10 ** 9
    for p in np.flatnonzero(ok):
        if p - prev <= cool:
            continue
        out.append(p)
        prev = p
    return np.asarray(out, int)


def period_stats(pos: np.ndarray, dates, oc: dict, key="hold20"):
    res = {}
    for name, a, b in PERIODS:
        sel = pos[(dates[pos] >= pd.Timestamp(a)) & (dates[pos] <= pd.Timestamp(b))] if len(pos) else pos
        v = oc[key][sel]
        v = v[np.isfinite(v)]
        res[name] = (int(v.sum()), int(len(v)), float(v.mean()) if len(v) else np.nan, int(len(sel)))
    return res


def l1_soxx(ps: dict) -> bool:
    k, n, r, _ = ps["2024~2026"]
    c1 = n >= 8 and r >= 0.70
    k, n, r, _ = ps["2026"]
    c2 = n >= 3 and r >= 0.70
    k, n, r, _ = ps["2025"]
    c3 = n < 3 or r >= 0.60
    return bool(c1 and c2 and c3)


def fmt(t):
    k, n, r, _ = t
    return f"{k}/{n}={r:.3f}" if n else "0/0=  n/a"


def bucket(x):
    return np.digitize(x, BUCKET_EDGES)


def main():
    C, O, info = load_raw()
    dates = C.index
    c, o = C.to_numpy(float), O.to_numpy(float)
    say("=" * 100)
    say("VERIFY f1 / sig2 / SOXX  (independent re-implementation, bc_lib not imported)")
    say("=" * 100)
    say(f"raw: {info}")

    st = state(c, 0.10)
    oc = outcomes(c, o, st["Ldd"])

    # ---------------------------------------------------------------- 1. independent match
    say("\n[1] Independent re-implementation vs family files")
    sig2 = np.isfinite(st["up"]) & np.isfinite(st["s20"]) & (st["up"] >= M_MAIN * st["s20"])
    # pandas pct_change().rolling(20).std() cross-check of sigma20
    s_pd = C.pct_change().rolling(20).std().to_numpy()
    ok = np.isfinite(s_pd) & np.isfinite(st["s20"])
    say(f"  sigma20 loop vs pandas pct_change().rolling(20).std(): max abs diff {np.abs(s_pd[ok] - st['s20'][ok]).max():.2e}, "
        f"valid-position mismatch {int((np.isfinite(s_pd) != np.isfinite(st['s20'])).sum())}")
    fs = first_signals(sig2, st, dates)
    fam = pd.read_csv(RES / "f1_SOXX_signals.csv", parse_dates=["date"])
    fam = fam[fam.rule == RULE].reset_index(drop=True)
    mine = pd.DataFrame(dict(date=dates[fs], pos=fs, up=st["up"][fs], s20=st["s20"][fs], dd60=st["dd60"][fs],
                             Ldd=st["Ldd"][fs], low_date=dates[st["lowpos"][fs]], hold20=oc["hold20"][fs],
                             near5=oc["near5"][fs], win20=oc["win20"][fs], r20=oc["r20"][fs]))
    say(f"  my first signals: {len(mine)}   family first signals: {len(fam)}")
    sd_m, sd_f = set(mine.date), set(fam.date)
    say(f"  dates only mine: {sorted(d.date() for d in sd_m - sd_f)}   only family: {sorted(d.date() for d in sd_f - sd_m)}")
    j = mine.merge(fam[["date", "pos", "up", "hold20", "near5", "win20", "r20"]], on="date", how="outer", suffixes=("", "_fam"))
    mism = []
    for col in ("pos", "up", "hold20", "near5", "win20", "r20"):
        a, b = j[col].to_numpy(float), j[col + "_fam"].to_numpy(float)
        bad = ~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-9))
        if bad.any():
            mism.append(f"{col}: {int(bad.sum())} rows")
    say(f"  per-signal field mismatches: {mism or 'none'}")
    say("  signals (date, Ldd date, up, 2*s20, dd60, hold20, near5, win20, r20):")
    for r in mine.itertuples():
        say(f"    {r.date.date()}  low@{r.low_date.date()}  up={r.up:+.4f}  2s20={2 * r.s20:.4f}  dd60={r.dd60:+.3f}  "
            f"hold20={r.hold20}  near5={r.near5}  win20={r.win20}  r20={r.r20:+.4f}" if np.isfinite(r.r20) else
            f"    {r.date.date()}  low@{r.low_date.date()}  up={r.up:+.4f}  2s20={2 * r.s20:.4f}  dd60={r.dd60:+.3f}  "
            f"hold20={r.hold20}  near5={r.near5}  win20={r.win20}  r20=nan")
    ps = period_stats(fs, dates, oc)
    summ = pd.read_csv(RES / "f1_SOXX_summary.csv")
    summ = summ[summ.rule == RULE].set_index("period")
    say("  per-period hold20 (mine | family):")
    pmis = []
    for name, *_ in PERIODS:
        k, n, r, nsig = ps[name]
        fr_, fn = summ.loc[name, "hold20"], summ.loc[name, "hold20_n"]
        same = (n == fn) and ((np.isnan(r) and np.isnan(fr_)) or abs(r - fr_) < 1e-9) and nsig == summ.loc[name, "n"]
        if not same:
            pmis.append(name)
        say(f"    {name:10s} n_sig={nsig:2d}  mine {fmt(ps[name]):14s} | family n_sig={int(summ.loc[name, 'n']):2d} "
            f"hold20={fr_:.3f} (n={int(fn)})  {'OK' if same else 'MISMATCH'}")
    for key in ("near5", "win20"):
        k, n, r, _ = period_stats(fs, dates, oc, key)["2024~2026"]
        say(f"    main {key}: mine {k}/{n}={r:.4f} | family {summ.loc['2024~2026', key]:.4f}")
    r20m = np.nanmean(oc["r20"][fs])
    say(f"    main r20 mean: mine {r20m:+.4f} | family {summ.loc['2024~2026', 'r20']:+.4f};  up median mine "
        f"{np.median(st['up'][fs]):.4f} | family {summ.loc['2024~2026', 'up_median']:.4f}")
    say(f"  registered L1 (my numbers): {l1_soxx(ps)}")
    independent_match = (len(mism) == 0 and not (sd_m ^ sd_f) and not pmis)
    say(f"  INDEPENDENT MATCH: {independent_match}")

    # matched baseline (same year x up bucket, in_dd days in eval window, outcome known)
    elig = st["in_dd"] & np.isfinite(st["Ldd"]) & (dates >= EVAL_A) & (dates <= EVAL_B)
    yr = np.asarray(dates.year)
    bk = bucket(st["up"])
    base_h = []
    for p in fs:
        m = elig & (yr == yr[p]) & (bk == bk[p]) & np.isfinite(oc["hold20"])
        base_h.append(oc["hold20"][m].mean())
    base_h = np.asarray(base_h)
    ev = np.isfinite(oc["hold20"][fs])
    say(f"  matched baseline (same year x up bucket) for evaluable signals: mine {base_h[ev].mean():.4f} | "
        f"family {summ.loc['2024~2026', 'hold20_base']:.4f}; signal - baseline = {oc['hold20'][fs][ev].mean() - base_h[ev].mean():+.4f}")

    # ---------------------------------------------------------------- 2. look-ahead audit (data side)
    say("\n[2] Look-ahead audit (mechanical checks)")
    # recompute state on truncated history: values at t must not change if data after t is removed
    rng = np.random.default_rng(1)
    test_t = np.unique(np.r_[fs, rng.choice(np.flatnonzero(elig), 40, replace=False)])
    worst = 0.0
    flips = 0
    for t in test_t:
        st_t = state(c[:t + 1], 0.10)
        for kname in ("up", "s20", "Ldd", "dd60"):
            a, b = st_t[kname][t], st[kname][t]
            if np.isfinite(a) or np.isfinite(b):
                worst = max(worst, abs(a - b))
        flips += int(st_t["in_dd"][t] != st["in_dd"][t])
        s_t = np.isfinite(st_t["up"][t]) and np.isfinite(st_t["s20"][t]) and st_t["up"][t] >= 2 * st_t["s20"][t]
        flips += int(s_t != sig2[t])
    say(f"  truncation test on {len(test_t)} dates (all first signals + 40 random in_dd days): "
        f"max |state(t | data<=t) - state(t | full)| = {worst:.2e}, signal/in_dd flips = {flips}")

    # ---------------------------------------------------------------- 3. robustness
    say("\n[3] Robustness (hold20 k/n; L1 = registered SOXX L1 test on that variant)")
    say(f"    {'variant':32s} {'n_sig':>5s} {'main 2024~2026':>15s} {'2026':>12s} {'2026H1':>11s} {'2026H2':>11s} {'2025':>11s} {'2024':>11s}  L1")
    rows = []

    def run(name, mask, stv=st, ocv=oc, cool=20):
        f = first_signals(mask, stv, dates, cool)
        p = period_stats(f, dates, ocv)
        l1 = l1_soxx(p)
        rows.append((name, len(f), p, l1))
        say(f"    {name:32s} {len(f):5d} {fmt(p['2024~2026']):>15s} {fmt(p['2026']):>12s} {fmt(p['2026H1']):>11s} "
            f"{fmt(p['2026H2']):>11s} {fmt(p['2025']):>11s} {fmt(p['2024']):>11s}  {l1}")
        return f, p

    def sigmask(stv, m):
        return np.isfinite(stv["up"]) & np.isfinite(stv["s20"]) & (stv["up"] >= m * stv["s20"])

    run("sig2 (registered)", sig2)
    for m in (1.0, 1.5, 2.5, 3.0):
        run(f"m={m}", sigmask(st, m))
    for w in (15, 25):
        stw = state(c, 0.10, sig_win=w)
        run(f"sigma window {w}, m=2", sigmask(stw, 2.0), stv=stw)
    for lag in (1, 2):
        sh = np.zeros_like(sig2)
        sh[lag:] = sig2[:-lag]
        run(f"signal delayed {lag} session(s)", sh)
    for cool in (10, 30):
        run(f"cooldown {cool}", sig2, cool=cool)
    for thr in (0.08, 0.12):
        stt = state(c, thr)
        run(f"in_dd threshold {int(thr * 100)}%", sigmask(stt, 2.0), stv=stt, ocv=outcomes(c, o, stt["Ldd"]))
    # which outcomes change under a one-session delay
    sh1 = np.zeros_like(sig2)
    sh1[1:] = sig2[:-1]
    f1 = first_signals(sh1, st, dates)
    say("  lag-1 first signals vs registered (date: hold20 registered -> lag1, up registered -> lag1):")
    for a, b in zip(fs, f1):
        flag = "  <-- flips" if (np.isfinite(oc['hold20'][a]) and np.isfinite(oc['hold20'][b])
                                 and oc['hold20'][a] != oc['hold20'][b]) else ""
        say(f"    {dates[a].date()} -> {dates[b].date()}: hold20 {oc['hold20'][a]} -> {oc['hold20'][b]}, "
            f"up {st['up'][a]:+.4f} -> {st['up'][b]:+.4f}, Ldd {st['Ldd'][a]:.2f} -> {st['Ldd'][b]:.2f}{flag}")
    # sig2 days dropped by the cooldown although a NEW decline low had formed after the accepted signal failed
    say("  cooldown side effect: in_dd sig2 days suppressed by the 20-session cooldown after a NEW lower Ldd formed:")
    raw = np.flatnonzero(sig2 & st["in_dd"] & np.isfinite(st["Ldd"]) & (dates >= EVAL_A) & (dates <= EVAL_B))
    for p in fs:
        q = raw[(raw > p) & (raw <= p + 20) & (st["Ldd"][raw] < st["Ldd"][p] - 1e-9)]
        if len(q):
            q0 = q[0]
            say(f"    after {dates[p].date()} (hold20={oc['hold20'][p]}): first suppressed new-low rebound {dates[q0].date()} "
                f"up={st['up'][q0]:+.4f} low@{dates[st['lowpos'][q0]].date()} hold20={oc['hold20'][q0]} ({len(q)} such days)")
    n_rob = len(rows) - 1
    n_rob_pass = sum(r[3] for r in rows[1:])
    say(f"  variants passing registered L1: {n_rob_pass}/{n_rob}")
    main_rates = [r[2]["2024~2026"][2] for r in rows[1:]]
    say(f"  main hold20 across variants: min {np.nanmin(main_rates):.3f}, median {np.nanmedian(main_rates):.3f}, max {np.nanmax(main_rates):.3f}")
    k, n, _, _ = ps["2024~2026"]
    say(f"  margin: registered main = {k}/{n}; one more failure -> {k - 1}/{n} = {(k - 1) / n:.3f} (< 0.70). "
        f"2026 = {fmt(ps['2026'])}; one more failure -> fails 2026 >= 70%.")

    # ---------------------------------------------------------------- 4. null
    say("\n[4] Null: random in_dd dates in main period, matched count + up-bucket mix, same 20-session cooldown")
    pool = elig & np.isfinite(oc["hold20"])
    ev_pos = fs[np.isfinite(oc["hold20"][fs])]
    obs = float(oc["hold20"][ev_pos].mean())
    need = np.bincount(bk[ev_pos], minlength=4)
    say(f"  pool: {int(pool.sum())} in_dd days with known hold20; pool by bucket "
        f"{dict(zip(BUCKET_NAMES, np.bincount(bk[pool], minlength=4).tolist()))}")
    say(f"  pool hold20 by bucket: " + ", ".join(
        f"{BUCKET_NAMES[b]} {oc['hold20'][pool & (bk == b)].mean():.3f}" if (pool & (bk == b)).any() else f"{BUCKET_NAMES[b]} n/a"
        for b in range(4)))
    say(f"  observed evaluable signals {len(ev_pos)}, bucket mix {dict(zip(BUCKET_NAMES, need.tolist()))}, hold20 {obs:.4f}")
    say("  observed hold20 by bucket: " + ", ".join(
        f"{BUCKET_NAMES[b]} {int(oc['hold20'][ev_pos][bk[ev_pos] == b].sum())}/{int((bk[ev_pos] == b).sum())}" for b in range(4)))
    rng = np.random.default_rng(SEED)
    pool_by_b = [np.flatnonzero(pool & (bk == b)) for b in range(4)]

    def draw(matched=True, cooldown=True, max_try=500):
        for attempt in range(max_try):
            order = np.concatenate([np.full(need[b], b) for b in range(4)]) if matched else np.full(need.sum(), -1)
            rng.shuffle(order)
            chosen = []
            okd = True
            for b in order:
                cand = pool_by_b[b] if b >= 0 else np.flatnonzero(pool)
                if cooldown and chosen:
                    ch = np.asarray(chosen)
                    cand = cand[np.min(np.abs(cand[:, None] - ch[None, :]), axis=1) > 20]
                else:
                    cand = np.setdiff1d(cand, chosen)
                if len(cand) == 0:
                    okd = False
                    break
                chosen.append(int(rng.choice(cand)))
            if okd:
                return np.asarray(chosen), attempt + 1
        return None, max_try

    res = {}
    for label, mt, cd in (("matched mix + cooldown (primary)", True, True), ("matched mix, no cooldown", True, False),
                          ("count only + cooldown", False, True)):
        vals, tries, fails = [], [], 0
        for _ in range(NDRAW):
            ch, a = draw(mt, cd)
            if ch is None:
                fails += 1
                continue
            tries.append(a)
            vals.append(oc["hold20"][ch].mean())
        vals = np.asarray(vals)
        p = float((vals >= obs - 1e-12).mean()) if len(vals) else np.nan
        res[label] = p
        say(f"  {label:34s}: draws {len(vals)} (failed {fails}, mean attempts {np.mean(tries):.1f}), null hold20 mean "
            f"{vals.mean():.3f} [5%,50%,95%] = {np.quantile(vals, [0.05, 0.5, 0.95]).round(3).tolist()}, "
            f"p(null >= {obs:.3f}) = {p:.4f}")
    null_p = res["matched mix + cooldown (primary)"]
    # binomial vs unconditional SOXX in_dd base rate
    from math import comb
    base_all = float(oc["hold20"][pool].mean())
    k, n = int(oc["hold20"][ev_pos].sum()), len(ev_pos)
    pb = sum(comb(n, i) * base_all ** i * (1 - base_all) ** (n - i) for i in range(k, n + 1))
    say(f"  binomial tail vs unconditional in_dd base {base_all:.3f}: P(X >= {k}/{n}) = {pb:.4f}")

    # ---------------------------------------------------------------- 5. mechanical comparison
    say("\n[5] Mechanical check: plain up-threshold rules")
    med = float(np.median(st["up"][fs]))
    med_ev = float(np.median(st["up"][ev_pos]))
    say(f"  candidate up median (all {len(fs)} first signals) = {med:.4f}; evaluable {len(ev_pos)} = {med_ev:.4f}")
    mech = {}
    for lab, x in ((f"up >= {med:.4f} (candidate median)", med), ("up >= 0.05", 0.05), ("up >= 0.08", 0.08),
                   ("up >= 0.03", 0.03)):
        mask = np.isfinite(st["up"]) & (st["up"] >= x)
        f = first_signals(mask, st, dates)
        p = period_stats(f, dates, oc)
        mech[lab] = p
        say(f"    {lab:34s} n_sig={len(f):2d}  main {fmt(p['2024~2026']):14s} 2026 {fmt(p['2026']):11s} "
            f"H1 {fmt(p['2026H1']):10s} H2 {fmt(p['2026H2']):10s} 2025 {fmt(p['2025']):10s}  L1 {l1_soxx(p)}")
    # logistic-free view: within sig2 signals, does hold20 track up?
    h = oc["hold20"][ev_pos]
    u_ = st["up"][ev_pos]
    say(f"  within sig2 evaluable signals: up of hits median {np.median(u_[h == 1]):.4f}, up of misses median "
        f"{np.median(u_[h == 0]):.4f}; misses' up = {np.round(np.sort(u_[h == 0]), 4).tolist()}")
    say(f"  sig2 2*sigma20 at signals: median {np.median(2 * st['s20'][fs]):.4f} (so sig2 ~ 'up >= ~{np.median(2 * st['s20'][fs]) * 100:.1f}%' "
        f"with a volatility-scaled threshold)")
    # sig2 days vs non-sig2 in_dd days within the same bucket
    say("  in_dd day-level hold20 by bucket, sig2-true vs sig2-false (main period, all days not just first signals):")
    for b in range(4):
        m1 = pool & (bk == b) & sig2
        m0 = pool & (bk == b) & ~sig2
        r1 = f"{oc['hold20'][m1].mean():.3f} (n={int(m1.sum())})" if m1.any() else "n/a (n=0)"
        r0 = f"{oc['hold20'][m0].mean():.3f} (n={int(m0.sum())})" if m0.any() else "n/a (n=0)"
        say(f"    {BUCKET_NAMES[b]:6s} sig2 {r1:18s} not sig2 {r0}")

    # ---------------------------------------------------------------- 6. verdict
    say("\n[6] Verdict inputs")
    lag1 = [r for r in rows if r[0] == "signal delayed 1 session(s)"][0]
    say(f"  independent match: {independent_match}; registered L1 (mine): {l1_soxx(ps)}; look-ahead flips: {flips}")
    say(f"  robustness: {n_rob_pass}/{n_rob} variants keep L1; lag1 main {fmt(lag1[2]['2024~2026'])} L1={lag1[3]}")
    say(f"  null p (primary) = {null_p:.4f}; signal sits {oc['hold20'][fs][ev].mean() - base_h[ev].mean():+.3f} vs matched baseline")
    if not independent_match or flips:
        verdict = "refuted"
    elif l1_soxx(ps) and null_p < 0.05 and n_rob_pass == n_rob:
        verdict = "confirmed"
    elif l1_soxx(ps):
        verdict = "fragile"
    else:
        verdict = "refuted"
    say(f"  VERDICT: {verdict}")
    say("\n[7] Reading")
    say("  - Replication is exact (17 dates, every field, every period, matched baseline 0.8607). Family code windows all end at t;")
    say("    no flow/news inputs; truncation test shows no dependence on data after t. Only data caveat: yfinance adjusted closes")
    say("    (split/dividend factors applied retroactively), which shifts ratios by dividend size only.")
    say("  - The pass is a knife edge: 12/16 main and 3/4 in 2026; one more miss fails both. 2026H1 is 0/1.")
    say("  - Neighbours m=1.5 and m=2.5 both fail L1, lag-1 fails (11/16; the flip is 2024-05-24, a new 60-day high where Ldd")
    say("    resets), cooldown 30 fails, in_dd 8% fails. 5/12 variants keep L1.")
    say("  - Null: random in_dd dates with the same up-bucket mix and cooldown reach 0.757 on average; p = 0.75. The rule is 11pp")
    say("    BELOW its matched baseline. All 4 misses are in the 3-8% up bucket (4/8 there vs 0.646 for random 3-8% days); every")
    say("    signal with up >= 8% held. Plain 'up >= 8.5%' (the candidate's median) scores 12/12. sig2 is ~ 'up >= ~5%'")
    say("    (median 2*sigma20 = 4.95%), and 'up >= 5%' gives the same 12/16 main.")
    say("  - Cooldown side effect: three failed pre-low signals (2024-07-31, 2025-03-17, 2026-03-10) block the real post-low")
    say("    rebound signals (2024-08-13, 2025-04-09, 2026-03-31), all of which held. The 75% is a property of rebound size")
    say("    plus cooldown placement, not of the volatility scaling.")
    OUT.write_text(buf.getvalue(), encoding="utf-8")
    return dict(verdict=verdict, independent_match=independent_match, null_p=null_p, hold20_main=ps["2024~2026"][2])


if __name__ == "__main__":
    main()
