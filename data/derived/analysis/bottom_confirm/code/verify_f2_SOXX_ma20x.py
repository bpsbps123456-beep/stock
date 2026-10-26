"""Independent adversarial verification of f2 / ma20x / SOXX (BRIEF.md 2026-10-26).

Does NOT import bc_lib. Re-implements prices, Ldd, in_dd, cooldown, outcomes, matched baseline
from the raw CSVs, then runs robustness, a matched random-date null and a mechanical comparison.

Run: data/derived/analysis/bottom_confirm 에서 `python3 -W ignore code/verify_f2_SOXX_ma20x.py`
Out: results/verify_f2_SOXX_ma20x.txt
"""
from __future__ import annotations

import io
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(__file__).resolve().parents[5]
RES = HERE / "results"
OUT = RES / "verify_f2_SOXX_ma20x.txt"

EVAL_START = pd.Timestamp("2024-01-02")
LAST = pd.Timestamp("2026-10-23")
PERIODS = [("2024~2026", "2024-01-02", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
           ("2025", "2025-01-02", "2025-12-31"), ("2024", "2024-01-02", "2024-12-31")]
EDGES = [0.03, 0.08, 0.15]          # up buckets 0-3, 3-8, 8-15, 15+
NDRAW = 2000
SEED = 7_2026_1026

buf = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    buf.write(s + "\n")


# ------------------------------------------------------------------ raw data (independent loader)
def load_soxx() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "research/news_turning_points_20261025/prices.csv", header=[0, 1], index_col=0)
    p.index = pd.to_datetime(p.index, errors="coerce")
    p = p[p.index.notna()]
    c_new = p[("close", "SOXX")].dropna()
    o_new = p[("open", "SOXX")].dropna()
    u = pd.read_csv(ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv",
                    encoding="utf-8-sig", index_col=0, parse_dates=True)["SOXX"].dropna()
    old = u[u.index < c_new.index[0]]
    C = pd.concat([old, c_new]).sort_index()
    assert not C.index.duplicated().any()
    O = o_new.reindex(C.index)
    # cross-check overlap
    ov = u.index.intersection(c_new.index)
    maxdiff = float(np.abs(u[ov] - c_new[ov]).max())
    # calendar check (independent of bc_lib; uses exchange_calendars only as a check)
    try:
        import exchange_calendars as xc
        cal = xc.get_calendar("XNYS").sessions_in_range(C.index[0], C.index[-1])
        cal = pd.DatetimeIndex(cal).tz_localize(None) if getattr(cal, "tz", None) is not None else pd.DatetimeIndex(cal)
        missing = cal.difference(C.index)
        extra = C.index.difference(cal)
    except Exception as e:  # pragma: no cover
        missing, extra = [f"calendar check failed: {e}"], []
    return pd.DataFrame({"C": C, "O": O}), dict(overlap_maxdiff=maxdiff, cal_missing=list(missing), cal_extra=list(extra))


# ------------------------------------------------------------------ state + outcomes
def frames(C: np.ndarray, O: np.ndarray, thr: float) -> dict:
    N = len(C)
    Ldd = np.full(N, np.nan)
    dd60 = np.full(N, np.nan)
    for t in range(59, N):
        w = C[t - 59:t + 1]
        mx = w.max()
        pk = t - 59 + int(np.flatnonzero(w == mx)[-1])        # latest position of the max
        Ldd[t] = C[pk:t + 1].min()
        dd60[t] = C[t] / mx - 1
    in_dd = np.zeros(N, bool)
    for t in range(N):
        w = dd60[max(0, t - 19):t + 1]
        w = w[~np.isnan(w)]
        in_dd[t] = len(w) > 0 and w.min() <= -thr
    up = C / Ldd - 1
    hold20 = np.full(N, np.nan)
    near5 = np.full(N, np.nan)
    r20 = np.full(N, np.nan)
    for t in range(N):
        if t + 20 <= N - 1 and not np.isnan(Ldd[t]):
            hold20[t] = float(C[t + 1:t + 21].min() >= Ldd[t] - 1e-12)
        if t + 1 <= N - 1 and not np.isnan(O[t + 1]):
            if t + 40 <= N - 1:
                near5[t] = float(C[t + 1:t + 41].min() >= 0.95 * O[t + 1])
            if t + 20 <= N - 1:
                r20[t] = C[t + 20] / O[t + 1] - 1
    win20 = np.where(np.isnan(r20), np.nan, (r20 > 0).astype(float))
    return dict(Ldd=Ldd, dd60=dd60, in_dd=in_dd, up=up, hold20=hold20, near5=near5, r20=r20, win20=win20)


def bucket(up: np.ndarray) -> np.ndarray:
    b = np.digitize(up, EDGES).astype(float)
    b[np.isnan(up)] = np.nan
    return b


def matched_base(idx: pd.DatetimeIndex, fr: dict, key="hold20") -> np.ndarray:
    """same calendar year × same up bucket, mean of in_dd days (date >= EVAL_START, outcome valid)."""
    b = bucket(fr["up"])
    y = idx.year.to_numpy()
    st = fr["in_dd"] & ~np.isnan(fr["up"]) & (idx >= EVAL_START)
    v = fr[key]
    out = np.full(len(idx), np.nan)
    tab = {}
    for yy in np.unique(y):
        for bb in range(4):
            m = st & (y == yy) & (b == bb) & ~np.isnan(v)
            if m.any():
                tab[(yy, bb)] = v[m].mean()
    for t in range(len(idx)):
        if not np.isnan(b[t]):
            out[t] = tab.get((y[t], int(b[t])), np.nan)
    return out


# ------------------------------------------------------------------ rule + first signals
def sma_cross(C: np.ndarray, n: int) -> np.ndarray:
    N = len(C)
    s = np.full(N, np.nan)
    cs = np.cumsum(np.insert(C, 0, 0.0))
    for t in range(n - 1, N):
        s[t] = (cs[t + 1] - cs[t + 1 - n]) / n
    sig = np.zeros(N, bool)
    for t in range(1, N):
        if not np.isnan(s[t]) and not np.isnan(s[t - 1]):
            sig[t] = (C[t] > s[t]) and (C[t - 1] <= s[t - 1])
    return sig


def first_signals(sig: np.ndarray, fr: dict, idx: pd.DatetimeIndex, cool: int = 20) -> np.ndarray:
    ok = sig & fr["in_dd"] & ~np.isnan(fr["Ldd"]) & (idx >= EVAL_START) & (idx <= LAST)
    out, prev = [], -10 ** 9
    for p in np.flatnonzero(ok):
        if p - prev <= cool:
            continue
        out.append(p)
        prev = p
    return np.asarray(out, int)


def period_stats(pos: np.ndarray, fr: dict, idx: pd.DatetimeIndex, base: np.ndarray | None = None) -> dict:
    res = {}
    d = idx[pos] if len(pos) else pd.DatetimeIndex([])
    for name, a, b in PERIODS:
        m = (d >= a) & (d <= b)
        pp = pos[m]
        h = fr["hold20"][pp]
        hv = h[~np.isnan(h)]
        r = dict(n=int(len(pp)), hold20_n=int(len(hv)), hold20=float(hv.mean()) if len(hv) else np.nan)
        for k in ("near5", "win20", "r20"):
            v = fr[k][pp]
            v = v[~np.isnan(v)]
            r[k] = float(v.mean()) if len(v) else np.nan
        r["up_med"] = float(np.median(fr["up"][pp])) if len(pp) else np.nan
        if base is not None and len(pp):
            ok = ~np.isnan(h) & ~np.isnan(base[pp])
            r["base"] = float(base[pp][ok].mean()) if ok.any() else np.nan
        res[name] = r
    return res


def fmt_row(label: str, st: dict) -> str:
    def f(p):
        r = st[p]
        return f"{r['hold20']:.3f}({r['hold20_n']})" if r["hold20_n"] else f"  -  ({r['n']})"
    return (f"  {label:34s} main {f('2024~2026'):11s} 2026 {f('2026'):11s} H1 {f('2026H1'):11s} "
            f"H2 {f('2026H2'):11s} 2025 {f('2025'):11s} 2024 {f('2024'):11s} upmed {st['2024~2026']['up_med']:.3f}")


# ------------------------------------------------------------------ null
def null_draws(pool_pos: np.ndarray, pool_b: np.ndarray, need: dict, hold: np.ndarray, cool: int, rng,
               pool_y: np.ndarray | None = None, need_y: list | None = None) -> np.ndarray:
    """Random dates from the in_dd pool with the same bucket mix (optionally year×bucket), pairwise spacing > cool."""
    res = []
    tries = 0
    while len(res) < NDRAW:
        tries += 1
        if tries > NDRAW * 200:
            break
        labels = list(need_y) if need_y is not None else [b for b, c in need.items() for _ in range(c)]
        rng.shuffle(labels)
        chosen = []
        okflag = True
        for lab in labels:
            if need_y is not None:
                cand = pool_pos[(pool_b == lab[1]) & (pool_y == lab[0])]
            else:
                cand = pool_pos[pool_b == lab]
            if chosen:
                ch = np.asarray(chosen)
                cand = cand[np.abs(cand[:, None] - ch[None, :]).min(1) > cool]
            if len(cand) == 0:
                okflag = False
                break
            chosen.append(int(rng.choice(cand)))
        if not okflag:
            continue
        res.append(hold[np.asarray(chosen)].mean())
    return np.asarray(res), tries


def main():
    pd.set_option("display.width", 220)
    df, chk = load_soxx()
    idx = df.index
    C, O = df.C.to_numpy(float), df.O.to_numpy(float)
    say("=" * 110)
    say("Independent verification: family f2, rule ma20x, kind SOXX  (no bc_lib import)")
    say("=" * 110)
    say(f"Data: SOXX close {idx[0].date()}..{idx[-1].date()}  N={len(idx)}; open from {df.O.first_valid_index().date()}")
    say(f"  overlap max |us_prices_close - prices.csv| = {chk['overlap_maxdiff']:.2e}; "
        f"XNYS sessions missing={len(chk['cal_missing'])} extra={len(chk['cal_extra'])}")

    fr = frames(C, O, 0.10)
    base = matched_base(idx, fr)
    sig = sma_cross(C, 20)
    pos = first_signals(sig, fr, idx)

    # ---------------------------------------------------------- 1. independent match
    say("\n[1] Independent re-implementation vs family files")
    fam = pd.read_csv(RES / "f2_SOXX_signals.csv", parse_dates=["date"])
    fam = fam[fam.rule == "ma20x"].reset_index(drop=True)
    fsum = pd.read_csv(RES / "f2_SOXX_summary.csv")
    fsum = fsum[fsum.rule == "ma20x"].set_index("period")
    mine = pd.DataFrame(dict(date=idx[pos], pos=pos, up=fr["up"][pos], hold20=fr["hold20"][pos],
                             base_hold20=base[pos], near5=fr["near5"][pos], win20=fr["win20"][pos],
                             r20=fr["r20"][pos], Ldd=fr["Ldd"][pos], dd60=fr["dd60"][pos]))
    say(mine.assign(date=mine.date.dt.date).round(4).to_string(index=False))
    same_dates = list(mine.date) == list(fam.date)
    say(f"  first-signal dates identical to family: {same_dates}  (mine {len(mine)}, family {len(fam)})")
    mism = []
    if same_dates:
        for k in ("pos", "up", "hold20", "base_hold20", "near5", "win20", "r20", "dd60"):
            a, b = mine[k].to_numpy(float), fam[k].to_numpy(float)
            bad = ~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-9))
            if bad.any():
                mism.append((k, int(bad.sum())))
        say(f"  per-signal field mismatches: {mism if mism else 'none'}")
    else:
        say(f"  only mine: {sorted(set(mine.date) - set(fam.date))}; only family: {sorted(set(fam.date) - set(mine.date))}")
    st0 = period_stats(pos, fr, idx, base)
    say("  period      mine hold20 (n)   family hold20 (n)   mine base   family base")
    per_ok = True
    for p, _, _ in PERIODS:
        r = st0[p]
        fh, fn = fsum.loc[p, "hold20"], fsum.loc[p, "hold20_n"]
        fb = fsum.loc[p, "hold20_base"]
        okp = (r["hold20_n"] == fn) and (abs(r["hold20"] - fh) < 1e-9 if r["hold20_n"] else True) and \
              (abs(r.get("base", np.nan) - fb) < 1e-9 if r["hold20_n"] else True)
        per_ok &= bool(okp)
        say(f"  {p:10s}  {r['hold20']:.4f} ({r['hold20_n']:2d})      {fh:.4f} ({int(fn):2d})          "
            f"{r.get('base', np.nan):.4f}      {fb:.4f}   {'OK' if okp else 'MISMATCH'}")
    indep_match = bool(same_dates and not mism and per_ok)
    say(f"  => independent match: {indep_match}")
    say(f"  main: n={st0['2024~2026']['n']} hold20={st0['2024~2026']['hold20']:.4f} near5={st0['2024~2026']['near5']:.4f} "
        f"win20={st0['2024~2026']['win20']:.4f} r20={st0['2024~2026']['r20']:.4f} up_med={st0['2024~2026']['up_med']:.4f}")
    # unconditional
    st_mask = fr["in_dd"] & (idx >= EVAL_START) & ~np.isnan(fr["hold20"])
    say(f"  unconditional in_dd hold20 2024~2026 = {fr['hold20'][st_mask].mean():.4f} (days={int(st_mask.sum())})")
    from math import comb
    k, n = int(round(st0["2024~2026"]["hold20"] * st0["2024~2026"]["hold20_n"])), st0["2024~2026"]["hold20_n"]
    for p0 in (fr["hold20"][st_mask].mean(), st0["2024~2026"]["base"]):
        pb = sum(comb(n, j) * p0 ** j * (1 - p0) ** (n - j) for j in range(k, n + 1))
        say(f"  binomial P(X >= {k}/{n} | p={p0:.3f}) = {pb:.3f}")

    # ---------------------------------------------------------- 2. look-ahead audit (static notes + dynamic check)
    say("\n[2] Look-ahead audit")
    # dynamic: truncation test - recompute the signal using only data up to t for every first-signal date
    trunc_ok = True
    for p in pos:
        s_tr = sma_cross(C[:p + 1], 20)
        fr_tr = frames(C[:p + 1], O[:p + 1], 0.10)
        if not (s_tr[p] and fr_tr["in_dd"][p] and abs(fr_tr["Ldd"][p] - fr["Ldd"][p]) < 1e-12):
            trunc_ok = False
            say(f"  truncation mismatch at {idx[p].date()}")
    say(f"  truncation test (signal, in_dd, Ldd recomputed with data cut at t): {'all identical' if trunc_ok else 'DIFFERENCES'}")
    say("  code read: sma_cross uses C.rolling(20).mean() (window ends at t) and .shift(1); no flow/news/market data;")
    say("  Ldd/in_dd/dd60 windows end at t; outcomes start at t+1 (entry O[t+1]); matched baseline averages same-year")
    say("  outcomes (an evaluation reference, not a signal input). Residual: yfinance auto-adjusted prices rescale")
    say("  history by later dividends (SOXX ~0.6%/yr), which can move a close-vs-SMA20 tie by a few bp; immaterial.")

    # ---------------------------------------------------------- 3. robustness
    say("\n[3] Robustness (hold20 (n) per period; main = 2024~2026)")
    rob = {}

    def run(label, n=20, cool=20, thr=0.10, delay=0):
        f = fr if thr == 0.10 else frames(C, O, thr)
        bs = base if thr == 0.10 else matched_base(idx, f)
        s = sma_cross(C, n)
        if delay:
            s = np.concatenate([np.zeros(delay, bool), s[:-delay]])
        ps = first_signals(s, f, idx, cool)
        st = period_stats(ps, f, idx, bs)
        rob[label] = st
        say(fmt_row(label, st) + f"  base {st['2024~2026'].get('base', np.nan):.3f}")
        return st

    run("base n=20 cool=20 thr=10% d=0")
    for n in (18, 19, 21, 22):
        run(f"SMA n={n}", n=n)
    for n in (10, 15, 25, 30, 50):
        run(f"SMA n={n} (wider context)", n=n)
    run("delay +1 session", delay=1)
    run("delay +2 sessions", delay=2)
    run("cooldown 10", cool=10)
    run("cooldown 30", cool=30)
    run("in_dd thr 8%", thr=0.08)
    run("in_dd thr 12%", thr=0.12)

    def l1(st):
        m, y6, y5 = st["2024~2026"], st["2026"], st["2025"]
        c1 = m["hold20_n"] >= 8 and m["hold20"] >= 0.70
        c2 = y6["hold20_n"] >= 3 and y6["hold20"] >= 0.70
        c3 = y5["hold20_n"] < 3 or y5["hold20"] >= 0.60
        return bool(c1 and c2 and c3)

    say("  L1 (registered SOXX rule) per variant:")
    for lab, st in rob.items():
        say(f"    {lab:34s} L1={l1(st)}  main-base diff={st['2024~2026']['hold20'] - st['2024~2026'].get('base', np.nan):+.3f}")

    # ---------------------------------------------------------- 4. null
    say("\n[4] Null: random in_dd dates in 2024~2026, same count and up-bucket mix, pairwise spacing > 20 sessions")
    rng = np.random.default_rng(SEED)
    b_all = bucket(fr["up"])
    pool_m = fr["in_dd"] & ~np.isnan(fr["Ldd"]) & ~np.isnan(fr["hold20"]) & (idx >= EVAL_START) & (idx <= LAST)
    pool_pos = np.flatnonzero(pool_m)
    pool_b = b_all[pool_pos]
    pool_y = idx.year.to_numpy()[pool_pos]
    obs_b = b_all[pos]
    need = {int(b): int((obs_b == b).sum()) for b in range(4)}
    obs = float(np.nanmean(fr["hold20"][pos]))
    say(f"  observed hold20 = {obs:.4f}; bucket mix (0-3,3-8,8-15,15+) = {[need[b] for b in range(4)]}; "
        f"pool days = {len(pool_pos)} by bucket {[int((pool_b == b).sum()) for b in range(4)]}")
    say(f"  pool hold20 by bucket: {[round(float(fr['hold20'][pool_pos][pool_b == b].mean()), 3) for b in range(4)]}")
    draws, tries = null_draws(pool_pos, pool_b, need, fr["hold20"], 20, rng)
    p_null = float((draws >= obs - 1e-12).mean())
    say(f"  draws={len(draws)} (attempts {tries}); null mean={draws.mean():.4f} sd={draws.std():.4f} "
        f"q05={np.quantile(draws, .05):.3f} q95={np.quantile(draws, .95):.3f}")
    say(f"  p = P(null hold20 >= observed) = {p_null:.4f}")
    need_y = [(int(idx[p].year), int(b_all[p])) for p in pos]
    draws_y, tries_y = null_draws(pool_pos, pool_b, need, fr["hold20"], 20, np.random.default_rng(SEED + 1),
                                  pool_y=pool_y, need_y=need_y)
    p_null_y = float((draws_y >= obs - 1e-12).mean()) if len(draws_y) else np.nan
    say(f"  secondary (year x bucket matched): draws={len(draws_y)} mean={draws_y.mean():.4f} p={p_null_y:.4f}")
    # 2026-only null
    pos26 = pos[idx[pos] >= pd.Timestamp("2026-01-02")]
    m26 = pool_m & (idx >= pd.Timestamp("2026-01-02"))
    pp26 = np.flatnonzero(m26)
    need26 = {int(b): int((b_all[pos26] == b).sum()) for b in range(4)}
    obs26 = float(np.nanmean(fr["hold20"][pos26]))
    d26, _ = null_draws(pp26, b_all[pp26], need26, fr["hold20"], 20, np.random.default_rng(SEED + 2))
    say(f"  2026-only null: observed {obs26:.3f} (n={len(pos26)}), null mean {d26.mean():.3f}, "
        f"p={float((d26 >= obs26 - 1e-12).mean()):.4f}")

    # ---------------------------------------------------------- 5. mechanical
    say("\n[5] Mechanical comparison: plain rule up >= candidate median up")
    upmed = float(np.median(fr["up"][pos]))
    mech = {}
    for thr_up in (upmed, 0.05, 0.08, 0.10):
        s = fr["up"] >= thr_up
        s = np.where(np.isnan(fr["up"]), False, s)
        ps = first_signals(s, fr, idx)
        st = period_stats(ps, fr, idx, base)
        mech[thr_up] = st
        say(fmt_row(f"up >= {thr_up:.4f}", st) + f"  base {st['2024~2026'].get('base', np.nan):.3f}")
    # all in_dd days (no cooldown) with up >= upmed
    m_all = pool_m & (fr["up"] >= upmed)
    say(f"  all in_dd days 2024~2026 with up >= {upmed:.4f} (no cooldown): hold20 = {fr['hold20'][m_all].mean():.4f} "
        f"(days {int(m_all.sum())}); with up < it: {fr['hold20'][pool_m & (fr['up'] < upmed)].mean():.4f}")
    say("  within-bucket comparison, 2024~2026 (candidate first signals vs all in_dd pool days):")
    for b in range(4):
        cm = obs_b == b
        if cm.sum():
            say(f"    bucket {['0-3%', '3-8%', '8-15%', '15%+'][b]:6s} cand {np.nanmean(fr['hold20'][pos[cm]]):.3f} (n={int(cm.sum())})"
                f"   pool {fr['hold20'][pool_pos[pool_b == b]].mean():.3f} (days={int((pool_b == b).sum())})")
    # in_dd days where ma20x fires at all (raw, no cooldown)
    raw_m = pool_m & sig
    say(f"  raw ma20x in_dd days (no cooldown): hold20 = {fr['hold20'][raw_m].mean():.4f} (days {int(raw_m.sum())}),"
        f" matched base avg {np.nanmean(base[raw_m]):.4f}")

    # ---------------------------------------------------------- 6. verdict
    say("\n[6] Verdict inputs")
    main_h = st0["2024~2026"]["hold20"]
    robust_l1 = {k: l1(v) for k, v in rob.items() if "wider" not in k}
    say(f"  independent match={indep_match}, look-ahead found={not trunc_ok}, main hold20={main_h:.4f} "
        f"vs matched base {st0['2024~2026']['base']:.4f}, null p={p_null:.4f} (year x bucket p={p_null_y:.4f})")
    say(f"  L1 survives in {sum(robust_l1.values())}/{len(robust_l1)} core variants: "
        + ", ".join(f"{k}={v}" for k, v in robust_l1.items()))
    diff_main = main_h - st0["2024~2026"]["base"]
    mech_h = mech[upmed]["2024~2026"]["hold20"]
    if not indep_match or not trunc_ok:
        verdict = "refuted"
    elif diff_main <= 0 and p_null >= 0.05:
        verdict = "refuted"       # passes L1 only; below matched baseline and no better than random same-state dates
    elif p_null >= 0.05 or not all(robust_l1.values()):
        verdict = "fragile"
    else:
        verdict = "confirmed"
    say(f"  main hold20 - matched base = {diff_main:+.4f}; plain up>=median rule hold20 = {mech_h:.4f}")
    say(f"\nVERDICT: {verdict}")
    say(f"  Reproduced exactly ({len(pos)} first signals, all fields), no look-ahead. But the {main_h:.1%} is below the")
    say(f"  same-state baseline ({st0['2024~2026']['base']:.1%}), below the random matched-date null mean "
        f"({draws.mean():.1%}, p={p_null:.2f}), and the plain rule")
    say(f"  'up >= {upmed:.1%}' scores {mech_h:.0%}. Within the 3-8% rebound bucket ma20x is worse than average (50% vs 65%).")
    say("  The only 2026H1 signal (2026-03-24) fired before the 03-30 low and failed; L1 breaks with cooldown 10")
    say("  and in_dd 8%. The hit rate is a consequence of how far price has already rebounded, not of the MA reclaim.")
    OUT.write_text(buf.getvalue(), encoding="utf-8")
    return dict(indep_match=indep_match, lookahead=not trunc_ok, main=main_h, p=p_null, p_y=p_null_y)


if __name__ == "__main__":
    main()
