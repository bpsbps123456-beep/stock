"""Adversarial verification: bottom_confirm family f1, rule sig3, kind SOXX.

Rule (BRIEF.md F1): fire on session t when up[t] >= 3 * sigma20[t], where
  up[t]      = C[t] / Ldd[t] - 1, Ldd[t] = lowest close since the latest 60-session closing high,
  sigma20[t] = std(ddof=1) of the 20 daily close-to-close returns ending at t (pct_change().rolling(20).std()).
Kept only on in_dd days (min dd60 over the last 20 sessions <= -10%) inside 2024-01-02~2026-10-23, then a
20-session cooldown between accepted signals. Main outcome hold20 = min(C[t+1..t+20]) >= Ldd[t].

bc_lib is NOT imported. Raw CSVs are re-read and every quantity is rebuilt with explicit numpy loops,
then compared with results/f1_SOXX_signals.csv and results/f1_SOXX_summary.csv (rule == 'sig3').

Run:    cd data/derived/analysis/bottom_confirm && python3 -W ignore code/verify_f1_SOXX_sig3.py
Output: results/verify_f1_SOXX_sig3.txt
"""
from __future__ import annotations

import io
import json
import pathlib
import re
from math import erf, sqrt

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(__file__).resolve().parents[5]
RES = HERE / "results"
OUT = RES / "verify_f1_SOXX_sig3.txt"

RULE, M = "sig3", 3.0
THR = 0.10
COOL = 20
EVAL_A, EVAL_B = pd.Timestamp("2024-01-02"), pd.Timestamp("2026-10-23")
PERIODS = [("2024~2026", "2024-01-02", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
           ("2025", "2025-01-02", "2025-12-31"), ("2024", "2024-01-02", "2024-12-31")]
EDGES = np.array([0.03, 0.08, 0.15])
BNAMES = ["0-3%", "3-8%", "8-15%", "15%+"]
NDRAW = 2000
BOOT = 5000
SEED = 3_2026_10_26

_buf = io.StringIO()


def say(*a) -> None:
    s = " ".join(str(x) for x in a)
    print(s)
    _buf.write(s + "\n")


# =========================================================================== raw data
def load_raw():
    p = pd.read_csv(ROOT / "research/news_turning_points_20261025/prices.csv", header=[0, 1], index_col=0)
    p.index = pd.to_datetime(p.index, errors="coerce")
    p = p[p.index.notna()].sort_index()
    sx = pd.DataFrame({k: pd.to_numeric(p[(k, "SOXX")], errors="coerce") for k in ("open", "high", "low", "close")})
    sx = sx.dropna()
    u = pd.read_csv(ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv",
                    encoding="utf-8-sig", index_col=0, parse_dates=True)["SOXX"].astype(float).dropna()
    import exchange_calendars as xc  # independent calendar, not bc_lib
    cal = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range("2023-01-01", "2026-10-23"))
    cal = cal.tz_localize(None) if cal.tz is not None else cal
    C = u.reindex(cal)
    C.loc[sx.index.intersection(cal)] = sx["close"]
    O = sx["open"].reindex(cal)
    ov = u.index.intersection(sx.index)
    un = u.index.union(sx.index)
    un = un[un <= EVAL_B]
    info = dict(sessions=len(cal), first=str(cal[0].date()), last=str(cal[-1].date()),
                union_equals_cal=bool(un.equals(cal)),
                close_overlap_days=len(ov), close_overlap_maxabs=float(np.abs(u[ov] - sx["close"][ov]).max()),
                close_nan=int(C.isna().sum()), open_nan_from_2023_12_01=int(O.loc["2023-12-01":].isna().sum()),
                open_over_close_range=[round(float((sx["open"] / sx["close"]).min()), 3),
                                       round(float((sx["open"] / sx["close"]).max()), 3)])
    return C, O, info


# =========================================================================== state + outcomes
def build_state(c: np.ndarray, thr: float, win: int = 20) -> dict:
    N = len(c)
    dd60 = np.full(N, np.nan)
    Ldd = np.full(N, np.nan)
    pkpos = np.full(N, -1)
    lopos = np.full(N, -1)
    for t in range(59, N):
        w = c[t - 59:t + 1]
        mx = w.max()
        pk = t - 59 + int(np.flatnonzero(w == mx)[-1])      # latest position of the 60-session high
        seg = c[pk:t + 1]
        lo = seg.min()
        Ldd[t], dd60[t], pkpos[t] = lo, c[t] / mx - 1, pk
        lopos[t] = pk + int(np.flatnonzero(seg == lo)[-1])
    up = c / Ldd - 1
    in_dd = np.zeros(N, bool)
    for t in range(N):
        w = dd60[max(0, t - 19):t + 1]
        w = w[np.isfinite(w)]
        in_dd[t] = bool(len(w)) and w.min() <= -thr
    sig = np.full(N, np.nan)
    for t in range(win, N):
        r = c[t - win + 1:t + 1] / c[t - win:t] - 1             # the `win` daily returns ending at t
        sig[t] = r.std(ddof=1)
    return dict(c=c, dd60=dd60, Ldd=Ldd, up=up, in_dd=in_dd, s=sig, pk=pkpos, lo=lopos)


def build_outcomes(c: np.ndarray, o: np.ndarray, Ldd: np.ndarray) -> dict:
    N = len(c)
    h20 = np.full(N, np.nan)
    h40 = np.full(N, np.nan)
    n5 = np.full(N, np.nan)
    r20 = np.full(N, np.nan)
    for t in range(N):
        if np.isfinite(Ldd[t]) and t + 20 <= N - 1:
            h20[t] = float(c[t + 1:t + 21].min() >= Ldd[t])
        if np.isfinite(Ldd[t]) and t + 40 <= N - 1:
            h40[t] = float(c[t + 1:t + 41].min() >= Ldd[t])
        if t + 1 <= N - 1 and np.isfinite(o[t + 1]):
            if t + 40 <= N - 1:
                n5[t] = float(c[t + 1:t + 41].min() >= 0.95 * o[t + 1])   # C[e..e+39]
            if t + 20 <= N - 1:
                r20[t] = c[t + 20] / o[t + 1] - 1                           # C[e+19] / O[e]
    w20 = np.where(np.isfinite(r20), (r20 > 0).astype(float), np.nan)
    return dict(hold20=h20, hold40=h40, near5=n5, r20=r20, win20=w20)


def accept(mask: np.ndarray, st: dict, dates: pd.DatetimeIndex, cool: int = COOL) -> np.ndarray:
    ok = mask & st["in_dd"] & np.isfinite(st["Ldd"]) & (dates >= EVAL_A) & (dates <= EVAL_B)
    keep, last = [], -10 ** 9
    for p in np.flatnonzero(ok):
        if p - last > cool:
            keep.append(p)
            last = p
    return np.asarray(keep, int)


def rate(pos: np.ndarray, dates, y: np.ndarray, a, b):
    sel = pos[(dates[pos] >= pd.Timestamp(a)) & (dates[pos] <= pd.Timestamp(b))] if len(pos) else pos
    v = y[sel]
    v = v[np.isfinite(v)]
    return int(v.sum()), len(v), (float(v.mean()) if len(v) else np.nan), len(sel)


def table(pos, dates, oc, key="hold20"):
    return {n: rate(pos, dates, oc[key], a, b) for n, a, b in PERIODS}


def l1(tb) -> bool:
    _, n, r, _ = tb["2024~2026"]
    _, n6, r6, _ = tb["2026"]
    _, n5, r5, _ = tb["2025"]
    return bool(n >= 8 and r >= 0.70 and n6 >= 3 and r6 >= 0.70 and (n5 < 3 or r5 >= 0.60))


def f(t) -> str:
    k, n, r, _ = t
    return f"{k}/{n}={r:.3f}" if n else "0/0=n/a"


def bucket(x):
    return np.digitize(x, EDGES)


def ncdf(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


# =========================================================================== main
def main() -> dict:
    C, O, info = load_raw()
    dates = C.index
    c, o = C.to_numpy(float), O.to_numpy(float)
    say("=" * 108)
    say("VERIFY f1 / sig3 / SOXX - independent re-implementation (bc_lib NOT imported)")
    say("=" * 108)
    say(f"raw data: {json.dumps(info)}")
    st = build_state(c, THR)
    oc = build_outcomes(c, o, st["Ldd"])
    s_pd = C.pct_change().rolling(20).std().to_numpy()
    okm = np.isfinite(s_pd) & np.isfinite(st["s"])
    say(f"sigma20 loop vs pandas pct_change().rolling(20).std(): max|diff| {np.abs(s_pd[okm] - st['s'][okm]).max():.1e}, "
        f"finite-position mismatches {int((np.isfinite(s_pd) != np.isfinite(st['s'])).sum())}")

    def sigrule(stv, m):
        return np.isfinite(stv["up"]) & np.isfinite(stv["s"]) & (stv["up"] >= m * stv["s"])

    sig = sigrule(st, M)
    fs = accept(sig, st, dates)

    # ------------------------------------------------------------------ [1] independent match
    say("\n[1] Independent match against family files (rule == 'sig3')")
    fam = pd.read_csv(RES / "f1_SOXX_signals.csv", parse_dates=["date"])
    fam = fam[fam.rule == RULE].reset_index(drop=True)
    summ = pd.read_csv(RES / "f1_SOXX_summary.csv")
    summ = summ[summ.rule == RULE].set_index("period")
    elig = st["in_dd"] & np.isfinite(st["Ldd"]) & (dates >= EVAL_A) & (dates <= EVAL_B)
    yr, bk = np.asarray(dates.year), bucket(st["up"])
    mbase = {}
    for key in ("hold20", "near5", "win20"):
        mb = np.full(len(fs), np.nan)
        for i, p in enumerate(fs):
            m = elig & (yr == yr[p]) & (bk == bk[p]) & np.isfinite(oc[key])
            mb[i] = oc[key][m].mean() if m.any() else np.nan
        mbase[key] = mb
    mine = pd.DataFrame(dict(date=dates[fs], pos=fs, up=st["up"][fs], s3=M * st["s"][fs], dd60=st["dd60"][fs],
                             Ldd=st["Ldd"][fs], low=dates[st["lo"][fs]], peak=dates[st["pk"][fs]],
                             hold20=oc["hold20"][fs], hold40=oc["hold40"][fs], near5=oc["near5"][fs],
                             win20=oc["win20"][fs], r20=oc["r20"][fs], base_hold20=mbase["hold20"]))
    say(f"  first signals: mine {len(mine)} | family {len(fam)}")
    only_m = sorted(set(mine.date) - set(fam.date))
    only_f = sorted(set(fam.date) - set(mine.date))
    say(f"  dates only mine: {[str(d.date()) for d in only_m]} | only family: {[str(d.date()) for d in only_f]}")
    j = mine.merge(fam, on="date", how="outer", suffixes=("", "_fam"))
    fld_bad = []
    for col in ("pos", "up", "dd60", "hold20", "hold40", "near5", "win20", "r20", "base_hold20"):
        a, b = j[col].to_numpy(float), j[col + "_fam"].to_numpy(float)
        bad = ~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-9))
        if bad.any():
            fld_bad.append(f"{col}:{int(bad.sum())}")
    say(f"  per-signal field mismatches (pos, up, dd60, hold20, hold40, near5, win20, r20, base_hold20): {fld_bad or 'none'}")
    say("  my first signals:")
    for r in mine.itertuples():
        r20s = f"{r.r20:+.4f}" if np.isfinite(r.r20) else "  nan"
        say(f"    {r.date.date()} peak {r.peak.date()} low {r.low.date()} up={r.up:+.4f} 3s20={r.s3:.4f} dd60={r.dd60:+.3f} "
            f"bucket={BNAMES[bucket(r.up)]:5s} hold20={r.hold20:.0f} base={r.base_hold20:.3f} near5={r.near5} "
            f"win20={r.win20} r20={r20s}")
    tb = table(fs, dates, oc)
    say("  per-period (mine | family):")
    pbad = []
    for name, *_ in PERIODS:
        k, n, r, nsig = tb[name]
        fn, fh, fsig = summ.loc[name, "hold20_n"], summ.loc[name, "hold20"], summ.loc[name, "n"]
        same = (n == fn) and (nsig == fsig) and ((np.isnan(r) and np.isnan(fh)) or abs(r - fh) < 1e-9)
        pbad += [] if same else [name]
        say(f"    {name:10s} n_sig {nsig:2d} hold20 {f(tb[name]):13s} | family n_sig {int(fsig):2d} hold20 {fh:.4f} "
            f"(n={int(fn)}) {'OK' if same else 'MISMATCH'}")
    ok_main = np.isfinite(oc["hold20"][fs])
    extra = {}
    for key in ("near5", "win20"):
        k, n, r, _ = table(fs, dates, oc, key)["2024~2026"]
        extra[key] = (r, summ.loc["2024~2026", key])
        say(f"    main {key}: mine {k}/{n}={r:.4f} | family {summ.loc['2024~2026', key]:.4f}")
    say(f"    main r20: mine {np.nanmean(oc['r20'][fs]):+.4f} | family {summ.loc['2024~2026', 'r20']:+.4f}; "
        f"up median mine {np.median(st['up'][fs]):.4f} | family {summ.loc['2024~2026', 'up_median']:.4f}")
    ok_n5 = np.isfinite(oc["near5"][fs])
    ok_w2 = np.isfinite(oc["win20"][fs])
    say(f"    main matched base hold20: mine {np.nanmean(mbase['hold20'][ok_main]):.4f} | family "
        f"{summ.loc['2024~2026', 'hold20_base']:.4f}; near5 base (signals with known near5, n={int(ok_n5.sum())}) mine "
        f"{np.nanmean(mbase['near5'][ok_n5]):.4f} | family {summ.loc['2024~2026', 'near5_base']:.4f}; win20 base mine "
        f"{np.nanmean(mbase['win20'][ok_w2]):.4f} | family {summ.loc['2024~2026', 'win20_base']:.4f}")
    for name, *_ in PERIODS[1:]:
        a, b = [x for x in PERIODS if x[0] == name][0][1:]
        sel = ok_main & (dates[fs] >= pd.Timestamp(a)) & (dates[fs] <= pd.Timestamp(b))
        fb = summ.loc[name, "hold20_base"]
        say(f"    {name:10s} matched base hold20 mine {np.mean(mbase['hold20'][sel]) if sel.any() else np.nan:.4f} | family {fb:.4f}")
    # own block bootstrap of (hold20 - matched base), fixed 20-session calendar blocks
    v = oc["hold20"][fs][ok_main] - mbase["hold20"][ok_main]
    lab = fs[ok_main] // COOL
    u_, inv = np.unique(lab, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    rng = np.random.default_rng(SEED)
    pick = rng.integers(0, len(u_), size=(BOOT, len(u_)))
    bm = sums[pick].sum(1) / cnts[pick].sum(1)
    dlo = float(np.quantile(bm, 0.025))
    nzero = int((np.abs(v) < 1e-12).sum())
    say(f"    diff (hold20 - matched base) per signal: {np.round(v, 4).tolist()}")
    say(f"    my block bootstrap dlo = {dlo:+.4f} (family {summ.loc['2024~2026', 'hold20_dlo']:+.4f}); "
        f"{nzero}/{len(v)} signals have diff exactly 0, so P(resample mean = 0) = ({nzero}/{len(v)})^{len(v)} = "
        f"{(nzero / len(v)) ** len(v):.4f} < 0.025 -> dlo > 0 is guaranteed by construction once every signal held.")
    big = np.argsort(-v)[:3]
    say(f"    share of the total positive diff from the largest single signal ({mine.date.iloc[np.flatnonzero(ok_main)[big[0]]].date()}): "
        f"{v[big[0]] / v.sum():.2f}; drop it and mean diff falls {v.mean():+.4f} -> {np.delete(v, big[0]).mean():+.4f}")
    say(f"  registered L1 on my numbers: {l1(tb)}")
    independent_match = not fld_bad and not only_m and not only_f and not pbad
    say(f"  INDEPENDENT MATCH: {independent_match}")

    # ------------------------------------------------------------------ [2] look-ahead audit
    say("\n[2] Look-ahead audit")
    src = (HERE / "code/f1_rebound.py").read_text(encoding="utf-8")
    pats = {"shift(-": r"shift\(\s*-", "center=True": r"center\s*=\s*True", "bfill/backfill": r"bfill|backfill",
            "iloc/iat with t+": r"(iat|iloc)\[[^\]]*\+\s*\d", "fillna(method": r"fillna\(\s*method"}
    rule_part = src.split("def decline_low_pos")[0]
    hits = {k: len(re.findall(p, rule_part)) for k, p in pats.items()}
    say(f"  f1_rebound.py rule-construction section (sigma20 + build_rules) pattern hits: {hits}")
    say("  f1_rebound.py signal = (fr['up'] >= 3*sigma20) & notna; sigma20 = (C/C.shift(1)-1).rolling(20).std() "
        "-> trailing window ending at t. fr['up'] comes from bc_lib.outcome_frame: Ldd = min(C[pk..t]), pk = latest "
        "60-session max in C[t-59..t] -> trailing. in_dd = dd60.rolling(20, min_periods=1).min() -> trailing.")
    say("  F1 uses prices only: no flow, options or news inputs, so no flow-lag / note-timing issue applies.")
    say("  posthoc lag1 rows in the family file use rules.shift(1) (past-shift, correct direction).")
    rng2 = np.random.default_rng(11)
    test = np.unique(np.r_[fs, np.flatnonzero(sig & elig), rng2.choice(np.flatnonzero(elig), 60, replace=False)])
    worst, flips = 0.0, 0
    for t in test:
        tt = build_state(c[:t + 1], THR)
        for k in ("up", "s", "Ldd", "dd60"):
            a, b = tt[k][t], st[k][t]
            if np.isfinite(a) or np.isfinite(b):
                worst = max(worst, float(abs(a - b)))
        flips += int(tt["in_dd"][t] != st["in_dd"][t])
        flips += int(bool(sigrule(tt, M)[t]) != bool(sig[t]))
    say(f"  truncation test: {len(test)} dates (all first signals, all sig3-true in_dd days, 60 random in_dd days): "
        f"max |state(t | data<=t) - state(t | full data)| = {worst:.1e}; in_dd/signal flips = {flips}")
    # adjusted-price caveat: ratio of us_prices_close vs prices.csv is 0 diff, both are yfinance adjusted closes
    say("  data caveat: both sources are yfinance adjusted prices (SOXX 3:1 split 2024-03 is back-adjusted); dividend "
        "back-adjustment rescales history by <1%/yr, immaterial to ratios such as up and sigma20 (not a tradable look-ahead).")
    lookahead = bool(flips) or worst > 1e-9 or any(v_ for k, v_ in hits.items())
    say(f"  LOOK-AHEAD FOUND: {lookahead}")

    # ------------------------------------------------------------------ [3] robustness
    say("\n[3] Robustness - hold20 k/n by period; L1 = registered SOXX L1 test applied to the variant")
    say(f"    {'variant':38s} {'nsig':>4s} {'main 24~26':>12s} {'2026':>11s} {'2026H1':>10s} {'2026H2':>10s} "
        f"{'2025':>10s} {'2024':>10s} {'near5':>6s} {'win20':>6s} {'upmed':>6s}  L1")
    rows = {}

    def run(name, mask=None, stv=None, ocv=None, cool=COOL, pos=None):
        stv = st if stv is None else stv
        ocv = oc if ocv is None else ocv
        p_ = accept(mask, stv, dates, cool) if pos is None else pos
        t_ = table(p_, dates, ocv)
        n5 = table(p_, dates, ocv, "near5")["2024~2026"][2]
        w2 = table(p_, dates, ocv, "win20")["2024~2026"][2]
        um = float(np.median(stv["up"][p_])) if len(p_) else np.nan
        rows[name] = (p_, t_, l1(t_))
        say(f"    {name:38s} {len(p_):4d} {f(t_['2024~2026']):>12s} {f(t_['2026']):>11s} {f(t_['2026H1']):>10s} "
            f"{f(t_['2026H2']):>10s} {f(t_['2025']):>10s} {f(t_['2024']):>10s} {n5:6.3f} {w2:6.3f} {um:6.3f}  {l1(t_)}")
        return p_, t_

    run("sig3 (registered)", sig)
    say("    -- multiplier m (registered grid neighbours 2 and 4, then finer half-steps)")
    for m in (2.0, 4.0, 2.5, 3.5):
        run(f"m={m}", sigrule(st, m))
    say("    -- sigma window (m=3)")
    for w in (15, 25):
        stw = build_state(c, THR, win=w)
        run(f"sigma window {w}", sigrule(stw, M), stv=stw)
    say("    -- delay")
    for lag in (1, 2):
        sh = np.zeros_like(sig)
        sh[lag:] = sig[:-lag]
        run(f"mask delayed {lag} (family-style shift)", sh)
    for lag in (1, 2):
        p_ = fs + lag
        p_ = p_[(p_ < len(c))]
        p_ = p_[st["in_dd"][p_]]
        run(f"same events evaluated at t+{lag}", pos=p_)
    say("    -- cooldown")
    for cl in (10, 30):
        run(f"cooldown {cl}", sig, cool=cl)
    say("    -- in_dd threshold")
    for th in (0.08, 0.12):
        stt = build_state(c, th)
        run(f"in_dd threshold {int(th * 100)}%", sigrule(stt, M), stv=stt, ocv=build_outcomes(c, o, stt["Ldd"]))
    var = {k: v_ for k, v_ in rows.items() if k != "sig3 (registered)"}
    core = ["m=2.0", "m=4.0", "mask delayed 1 (family-style shift)", "same events evaluated at t+1",
            "cooldown 10", "cooldown 30", "in_dd threshold 8%", "in_dd threshold 12%"]
    core_pass = sum(var[k][2] for k in core)
    all_pass = sum(v_[2] for v_ in var.values())
    say(f"  requested variants keeping L1: {core_pass}/{len(core)}; all variants: {all_pass}/{len(var)}")
    mains = np.array([v_[1]["2024~2026"][2] for v_ in var.values()])
    say(f"  main hold20 across variants: min {np.nanmin(mains):.3f} median {np.nanmedian(mains):.3f} max {np.nanmax(mains):.3f}")
    k, n, _, _ = tb["2024~2026"]
    k6, n6, _, _ = tb["2026"]
    say(f"  margin: main {k}/{n}; 2026 {k6}/{n6} -> one failure in 2026 gives {(k6 - 1)}/{n6} = {(k6 - 1) / n6:.3f} < 0.70 "
        f"(L1 lost); the 2026 leg rests on {n6} signals, 2026H1 on {tb['2026H1'][1]}.")
    # which signals fail in each variant (and how the variant's set differs from the registered one)
    say("  failing signals per variant (date, up, bucket, low date) and dates added/removed vs registered:")
    reg = set(fs.tolist())
    for name, (p_, t_, ok_) in var.items():
        thr_st = st
        if name.startswith("in_dd threshold"):
            thr_st = build_state(c, int(name.split()[-1][:-1]) / 100)
        elif name.startswith("sigma window"):
            thr_st = build_state(c, THR, win=int(name.split()[-1]))
        oc_v = oc if thr_st is st or name.startswith("sigma window") else build_outcomes(c, o, thr_st["Ldd"])
        bad = [q for q in p_ if oc_v["hold20"][q] == 0]
        add = sorted(set(p_.tolist()) - reg)
        rem = sorted(reg - set(p_.tolist()))
        desc = "; ".join(f"{dates[q].date()} up={thr_st['up'][q]:+.3f} {BNAMES[bucket(thr_st['up'][q])]} low@{dates[thr_st['lo'][q]].date()}"
                         for q in bad) or "none"
        say(f"    {name:38s} fails: {desc}")
        if not name.startswith("same events") and not name.startswith("mask delayed"):
            say(f"    {'':38s} added {[str(dates[q].date()) for q in add]} removed {[str(dates[q].date()) for q in rem]}")
    lp = rows["mask delayed 1 (family-style shift)"][0]
    for p_ in [q for q in lp if oc["hold20"][q] == 0]:
        say(f"  lag1 miss detail: on {dates[p_].date()} the 60-session high rolls to {dates[st['pk'][p_]].date()}, so the "
            f"decline low resets to that day's own close (Ldd={st['Ldd'][p_]:.2f}, up=0); the next 20 closes reach "
            f"{c[p_ + 1:p_ + 21].min():.2f}. The day before, the same signal was graded against the 2024-08-07 low and held.")

    # ------------------------------------------------------------------ [4] null
    say("\n[4] Null: random in_dd dates in the main period, same count + up-bucket mix, >20-session spacing, "
        f"{NDRAW} draws")
    pool = elig & np.isfinite(oc["hold20"]) & np.isfinite(st["up"])
    ev = fs[np.isfinite(oc["hold20"][fs])]
    obs = float(oc["hold20"][ev].mean())
    need = np.bincount(bk[ev], minlength=4)
    say(f"  pool {int(pool.sum())} in_dd days with known hold20; by bucket "
        f"{dict(zip(BNAMES, np.bincount(bk[pool], minlength=4).tolist()))}")
    say("  pool hold20 by bucket: " + ", ".join(
        f"{BNAMES[b]} {oc['hold20'][pool & (bk == b)].mean():.3f}" for b in range(4) if (pool & (bk == b)).any()))
    say("  pool hold20 by year x bucket: " + "; ".join(
        f"{y}: " + ",".join(f"{BNAMES[b]} {oc['hold20'][pool & (bk == b) & (yr == y)].mean():.2f}(n={int((pool & (bk == b) & (yr == y)).sum())})"
                            for b in range(4) if (pool & (bk == b) & (yr == y)).any()) for y in (2024, 2025, 2026)))
    say(f"  observed: {len(ev)} evaluable signals, bucket mix {dict(zip(BNAMES, need.tolist()))}, hold20 {obs:.4f}")
    rng = np.random.default_rng(SEED)
    pools = {b: np.flatnonzero(pool & (bk == b)) for b in range(4)}
    pools_y = {(y, b): np.flatnonzero(pool & (bk == b) & (yr == y)) for y in (2024, 2025, 2026) for b in range(4)}
    need_y = [(yr[p_], bk[p_]) for p_ in ev]

    def draw(labels, getpool, spacing=True, tries=1000):
        for a in range(tries):
            order = list(labels)
            rng.shuffle(order)
            ch = []
            for lb in order:
                cand = getpool(lb)
                if ch:
                    arr = np.asarray(ch)
                    cand = cand[np.abs(cand[:, None] - arr[None, :]).min(1) > COOL] if spacing else np.setdiff1d(cand, arr)
                if not len(cand):
                    break
                ch.append(int(rng.choice(cand)))
            else:
                return np.asarray(ch)
        return None

    nulls = {}
    specs = [("bucket mix + spacing (PRIMARY)", [int(b) for b in bk[ev]], lambda lb: pools[lb], True),
             ("year x bucket mix + spacing", need_y, lambda lb: pools_y[lb], True),
             ("bucket mix, no spacing", [int(b) for b in bk[ev]], lambda lb: pools[lb], False),
             ("count only + spacing", [0] * len(ev), lambda lb: np.flatnonzero(pool), True)]
    for lab_, labels, gp, sp in specs:
        vals, fails = [], 0
        for _ in range(NDRAW):
            ch = draw(labels, gp, sp)
            if ch is None:
                fails += 1
                continue
            vals.append(oc["hold20"][ch].mean())
        vals = np.asarray(vals)
        pv = float((vals >= obs - 1e-12).mean())
        nulls[lab_] = pv
        say(f"  {lab_:32s} draws {len(vals)} (infeasible {fails}) null mean {vals.mean():.3f} "
            f"q05/50/95 {np.quantile(vals, [0.05, 0.5, 0.95]).round(3).tolist()}  p(null >= {obs:.3f}) = {pv:.4f} "
            f"(MC s.e. {np.sqrt(pv * (1 - pv) / max(len(vals), 1)):.4f})")
    null_p = nulls["bucket mix + spacing (PRIMARY)"]
    approx = np.prod([oc["hold20"][pools[b]].mean() ** need[b] for b in range(4) if need[b]])
    say(f"  independence approximation prod_b (pool rate_b)^n_b = {approx:.4f}")
    bh = float(oc["hold20"][pool].mean())
    say(f"  binomial vs unconditional in_dd hold20 {bh:.3f}: P(12/12) = {bh ** len(ev):.4f} "
        "(ignores the up mix, shown only as the naive bar)")

    # ------------------------------------------------------------------ [5] mechanical check
    say("\n[5] Mechanical check: is sig3 just 'price has already rebounded enough'?")
    med = float(np.median(st["up"][fs]))
    say(f"  candidate up median (12 first signals) = {med:.4f}; 3*sigma20 at signals median {np.median(M * st['s'][fs]):.4f} "
        f"range [{np.min(M * st['s'][fs]):.4f}, {np.max(M * st['s'][fs]):.4f}]")
    mech = {}
    for lab_, x in ((f"up >= {med:.4f} (candidate median)", med), ("up >= 0.08 (registered up8)", 0.08),
                    ("up >= 0.10 (registered up10)", 0.10), ("up >= 0.05", 0.05)):
        p_, t_ = run(lab_, np.isfinite(st["up"]) & (st["up"] >= x))
        mech[lab_] = (p_, t_)
    pm = mech[f"up >= {med:.4f} (candidate median)"][0]
    close = [int(np.min(np.abs(pm - p_))) for p_ in fs]
    say(f"  sig3 first signals with a plain-median-rule first signal within 0/<=5/<=10 sessions: "
        f"{sum(d == 0 for d in close)}/{sum(d <= 5 for d in close)}/{sum(d <= 10 for d in close)} of {len(fs)}")
    say("  day-level hold20 in pool by up bucket, sig3-true vs sig3-false (main period, every in_dd day):")
    for b in range(4):
        m1, m0 = pool & (bk == b) & sig, pool & (bk == b) & ~sig
        r1 = f"{oc['hold20'][m1].mean():.3f} (n={int(m1.sum())})" if m1.any() else "n/a (n=0)"
        r0 = f"{oc['hold20'][m0].mean():.3f} (n={int(m0.sum())})" if m0.any() else "n/a (n=0)"
        say(f"    {BNAMES[b]:6s} sig3 {r1:18s} not sig3 {r0}")
    d38 = np.flatnonzero(pool & (bk == 1) & sig)
    eps = np.split(d38, np.flatnonzero(np.diff(d38) > 5) + 1) if len(d38) else []
    say(f"  the only bucket where sig3 differs from 'up' alone is 3-8%: its {len(d38)} sig3-true days form {len(eps)} episodes: "
        + "; ".join(f"{dates[e[0]].date()}~{dates[e[-1]].date()} ({len(e)}d, low@{dates[st['lo'][e[0]]].date()})" for e in eps))
    nm = pool & (bk == 1) & ~sig
    say(f"  3-8% bucket by year, sig3 vs not: " + "; ".join(
        f"{y}: sig3 {int(oc['hold20'][pool & (bk == 1) & sig & (yr == y)].sum())}/{int((pool & (bk == 1) & sig & (yr == y)).sum())}, "
        f"not {int(oc['hold20'][nm & (yr == y)].sum())}/{int((nm & (yr == y)).sum())}" for y in (2024, 2025, 2026)))
    say(f"  P(3 independent 3-8% first signals all hold | pool rate {oc['hold20'][pools[1]].mean():.3f}) = "
        f"{oc['hold20'][pools[1]].mean() ** 3:.3f}; with 2026's 3-8% rate {oc['hold20'][pools_y[(2026, 1)]].mean():.3f} for one of them "
        f"= {oc['hold20'][pools_y[(2024, 1)]].mean() ** 2 * oc['hold20'][pools_y[(2026, 1)]].mean():.3f}")
    say("  pool hold20 when up >= x (every in_dd day, main period / 2026):")
    for x in (0.0, 0.03, 0.05, 0.07, 0.08, 0.098, 0.12, 0.15):
        m = pool & (st["up"] >= x)
        m6 = m & (yr == 2026)
        say(f"    up >= {x:5.3f}: {oc['hold20'][m].mean():.3f} (n={int(m.sum())}) | 2026 "
            f"{oc['hold20'][m6].mean() if m6.any() else np.nan:.3f} (n={int(m6.sum())})")
    # driftless random-walk floor for a 3-sigma cushion over 20 sessions (reflection principle)
    rw = 1 - 2 * (1 - ncdf(3 / sqrt(20)))
    say(f"  context: driftless Gaussian walk floor P(no new low in 20 sessions | cushion = 3 daily sigma) = {rw:.3f}; "
        f"empirical in_dd days with up >= 3*sigma20: {oc['hold20'][pool & sig].mean():.3f} (n={int((pool & sig).sum())}).")

    # ------------------------------------------------------------------ [6] verdict
    say("\n[6] Verdict")
    robust = core_pass == len(core)
    say(f"  independent match {independent_match}; look-ahead {lookahead}; registered L1 (mine) {l1(tb)}; "
        f"requested robustness variants keeping L1 {core_pass}/{len(core)}; primary null p {null_p:.4f}")
    if not independent_match or lookahead or not l1(tb):
        verdict = "refuted"
    elif robust and null_p < 0.05:
        verdict = "confirmed"
    else:
        verdict = "fragile"
    say(f"  VERDICT: {verdict}")
    m6 = pool & (st["up"] >= med)
    say("  reasons:")
    say(f"   - numbers reproduce exactly and the rule uses trailing data only; L1 holds on the registered spec "
        f"(main {f(tb['2024~2026'])}, 2026 {f(tb['2026'])}, 2026H1 {f(tb['2026H1'])}).")
    say(f"   - null p {null_p:.3f} (year x bucket {nulls['year x bucket mix + spacing']:.3f}) >= 0.05: given how far price had "
        f"already rebounded, 12/12 is not unusual. Only the count-only null (ignores up) rejects.")
    say(f"   - mechanical: plain up >= {med:.4f} gives {f(mech[f'up >= {med:.4f} (candidate median)'][1]['2024~2026'])} "
        f"(2026 {f(mech[f'up >= {med:.4f} (candidate median)'][1]['2026'])}); every in_dd day with up >= {med:.4f} held "
        f"({int(oc['hold20'][m6].sum())}/{int(m6.sum())}). sig3 adds nothing in the 8%+ buckets; its only extra content is "
        f"{len(d38)} days in the 3-8% bucket from {len(eps)} short episodes (about 3 distinct lows).")
    say(f"   - L2 dlo > 0 is mechanical once all signals hold; 2026-09-02 (2026 3-8% base 0.32) supplies "
        f"{v[big[0]] / v.sum():.0%} of the positive gap.")
    say(f"   - knife-edge 2026 leg: n=3, one miss drops it below 70%. m=2.5 and sigma windows 15/25 each add a failing "
        f"March-2026 signal (vs the 03-06 low, later undercut on 03-30) and lose L1. A one-session delay lands the 2024-10-15 "
        f"event on a day when the 60-session high rolls and the reference low resets to that day's close, and it then fails.")
    say(f"   - near5 {extra['near5'][0]:.3f} and win20 {extra['win20'][0]:.3f} sit at the matched levels (0.517, 0.658): "
        "no tradable edge after entry.")
    OUT.write_text(_buf.getvalue(), encoding="utf-8")
    res = dict(verdict=verdict, independent_match=independent_match, lookahead=lookahead, null_p=null_p,
               hold20_main=tb["2024~2026"][2], core_pass=core_pass, n_core=len(core))
    say(json.dumps(res))
    OUT.write_text(_buf.getvalue(), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
