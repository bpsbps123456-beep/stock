"""Independent adversarial verification of f1 / up8 / SOXX (BRIEF.md 2026-10-26).

Does NOT import bc_lib. Re-implements the SOXX price series, decline low Ldd, dd60, in_dd, the
20-session cooldown, the outcomes hold20 / near5 / win20 / r20 and the same-state (year x up-bucket)
baseline directly from the raw CSVs, then runs a look-ahead audit, robustness grid, a matched
random-date null and a mechanical comparison.

Rule under test: up[t] = C[t]/Ldd[t] - 1 >= 0.08, on in_dd days, first signals with 20-session cooldown.

Run: cd data/derived/analysis/bottom_confirm && python3 -W ignore code/verify_f1_SOXX_up8.py
Out: results/verify_f1_SOXX_up8.txt
"""
from __future__ import annotations

import io
import pathlib
from math import comb

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(__file__).resolve().parents[5]
RES = HERE / "results"
OUT = RES / "verify_f1_SOXX_up8.txt"

RULE, X = "up8", 0.08
EVAL_START = pd.Timestamp("2024-01-02")
LAST = pd.Timestamp("2026-10-23")
PERIODS = [("2024~2026", "2024-01-02", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
           ("2025", "2025-01-02", "2025-12-31"), ("2024", "2024-01-02", "2024-12-31")]
EDGES = [0.03, 0.08, 0.15]          # up buckets 0-3, 3-8, 8-15, 15%+
BNAMES = ["0-3%", "3-8%", "8-15%", "15%+"]
NDRAW = 2000
SEED = 8_2026_1026

buf = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    buf.write(s + "\n")


# ------------------------------------------------------------------ raw data (own loader)
def load_soxx():
    p = pd.read_csv(ROOT / "research/news_turning_points_20261025/prices.csv", header=[0, 1], index_col=0)
    p.index = pd.to_datetime(p.index, errors="coerce")
    p = p[p.index.notna()]
    c_new = p[("close", "SOXX")].dropna()
    o_new = p[("open", "SOXX")].dropna()
    u = pd.read_csv(ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv",
                    encoding="utf-8-sig", index_col=0, parse_dates=True)["SOXX"].dropna()
    C = pd.concat([u[u.index < c_new.index[0]], c_new]).sort_index()
    assert not C.index.duplicated().any()
    C = C[C.index <= LAST]
    O = o_new.reindex(C.index)
    ov = u.index.intersection(c_new.index)
    chk = dict(overlap_n=len(ov), overlap_maxdiff=float(np.abs(u[ov] - c_new[ov]).max()))
    try:  # calendar used only as a cross-check
        import exchange_calendars as xc
        cal = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range(C.index[0], C.index[-1]))
        cal = cal.tz_localize(None) if cal.tz is not None else cal
        chk["cal_missing"], chk["cal_extra"] = len(cal.difference(C.index)), len(C.index.difference(cal))
    except Exception as e:  # pragma: no cover
        chk["cal_missing"], chk["cal_extra"] = f"n/a ({e})", "n/a"
    return pd.DataFrame({"C": C, "O": O}), chk


# ------------------------------------------------------------------ state and outcomes
def frames(C: np.ndarray, O: np.ndarray, thr: float) -> dict:
    N = len(C)
    Ldd = np.full(N, np.nan)
    Lpos = np.full(N, -1)
    dd60 = np.full(N, np.nan)
    for t in range(59, N):
        w = C[t - 59:t + 1]
        mx = w.max()
        pk = t - 59 + int(np.flatnonzero(w == mx)[-1])          # latest position of the 60d max
        seg = C[pk:t + 1]
        Ldd[t] = seg.min()
        Lpos[t] = pk + int(np.flatnonzero(seg == seg.min())[-1])
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
    return dict(Ldd=Ldd, Lpos=Lpos, dd60=dd60, in_dd=in_dd, up=up, hold20=hold20, near5=near5, r20=r20, win20=win20)


def bucket(up: np.ndarray) -> np.ndarray:
    b = np.digitize(up, EDGES).astype(float)
    b[np.isnan(up)] = np.nan
    return b


def matched_base(idx: pd.DatetimeIndex, fr: dict, key: str = "hold20") -> np.ndarray:
    """Same calendar year x same up bucket: mean outcome of in_dd days (date >= EVAL_START, outcome known)."""
    b = bucket(fr["up"])
    y = idx.year.to_numpy()
    st = fr["in_dd"] & ~np.isnan(fr["up"]) & (idx >= EVAL_START)
    v = fr[key]
    tab = {}
    for yy in np.unique(y):
        for bb in range(4):
            m = st & (y == yy) & (b == bb) & ~np.isnan(v)
            if m.any():
                tab[(yy, bb)] = v[m].mean()
    out = np.full(len(idx), np.nan)
    for t in range(len(idx)):
        if not np.isnan(b[t]):
            out[t] = tab.get((y[t], int(b[t])), np.nan)
    return out


def up_rule(up: np.ndarray, x: float) -> np.ndarray:
    return np.where(np.isnan(up), False, up >= x)


def first_signals(sig: np.ndarray, fr: dict, idx: pd.DatetimeIndex, cool: int = 20,
                  start: pd.Timestamp = EVAL_START) -> np.ndarray:
    ok = sig & fr["in_dd"] & ~np.isnan(fr["Ldd"]) & (idx >= start) & (idx <= LAST)
    out, prev = [], -10 ** 9
    for p in np.flatnonzero(ok):
        if p - prev <= cool:
            continue
        out.append(p)
        prev = p
    return np.asarray(out, int)


def shift(sig: np.ndarray, d: int) -> np.ndarray:
    return sig if d == 0 else np.concatenate([np.zeros(d, bool), sig[:-d]])


def period_stats(pos: np.ndarray, fr: dict, idx: pd.DatetimeIndex, bases: dict | None = None) -> dict:
    res = {}
    d = idx[pos] if len(pos) else pd.DatetimeIndex([])
    for name, a, b in PERIODS:
        pp = pos[(d >= a) & (d <= b)]
        h = fr["hold20"][pp]
        r = dict(n=int(len(pp)), hold20_n=int((~np.isnan(h)).sum()),
                 hold20=float(np.nanmean(h)) if (~np.isnan(h)).any() else np.nan)
        for k in ("near5", "win20", "r20"):
            v = fr[k][pp]
            r[k] = float(np.nanmean(v)) if (~np.isnan(v)).any() else np.nan
        r["up_med"] = float(np.median(fr["up"][pp])) if len(pp) else np.nan
        if bases is not None:
            for k, bs in bases.items():
                ok = ~np.isnan(fr[k][pp]) & ~np.isnan(bs[pp])
                r[f"base_{k}"] = float(bs[pp][ok].mean()) if ok.any() else np.nan
        res[name] = r
    return res


def l1(st: dict) -> bool:
    m, y6, y5 = st["2024~2026"], st["2026"], st["2025"]
    c1 = m["hold20_n"] >= 8 and m["hold20"] >= 0.70
    c2 = y6["hold20_n"] >= 3 and y6["hold20"] >= 0.70
    c3 = y5["hold20_n"] < 3 or y5["hold20"] >= 0.60
    return bool(c1 and c2 and c3)


def fmt_row(label: str, st: dict) -> str:
    def f(p):
        r = st[p]
        return f"{r['hold20']:.3f}({r['hold20_n']:2d})" if r["hold20_n"] else f"  -  ({r['n']:2d})"
    return (f"  {label:30s} main {f('2024~2026'):10s} 2026 {f('2026'):10s} H1 {f('2026H1'):10s} "
            f"H2 {f('2026H2'):10s} 2025 {f('2025'):10s} 2024 {f('2024'):10s} upmed {st['2024~2026']['up_med']:.3f}")


# ------------------------------------------------------------------ null sampler
def null_draws(pool_pos, pool_lab, labels, hold, cool, rng):
    """Random date sets from pool with exactly the given label multiset, pairwise spacing > cool."""
    res, tries = [], 0
    labels = list(labels)
    while len(res) < NDRAW and tries < NDRAW * 500:
        tries += 1
        rng.shuffle(labels)
        chosen = []
        for lab in labels:
            cand = pool_pos[pool_lab == lab]
            if chosen:
                ch = np.asarray(chosen)
                cand = cand[np.abs(cand[:, None] - ch[None, :]).min(1) > cool]
            if len(cand) == 0:
                chosen = None
                break
            chosen.append(int(rng.choice(cand)))
        if chosen is None:
            continue
        res.append(float(hold[np.asarray(chosen)].mean()))
    return np.asarray(res), tries


def block_boot_lo(v: np.ndarray, lab: np.ndarray, rng, B: int = 5000) -> tuple[float, float]:
    u, inv = np.unique(lab, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    pick = rng.integers(0, len(u), size=(B, len(u)))
    m = sums[pick].sum(1) / cnts[pick].sum(1)
    return tuple(np.quantile(m, [0.025, 0.975]))


# ------------------------------------------------------------------ main
def main():
    pd.set_option("display.width", 240)
    df, chk = load_soxx()
    idx = df.index
    C, O = df.C.to_numpy(float), df.O.to_numpy(float)
    N = len(C)
    say("=" * 118)
    say(f"Independent verification: family f1, rule {RULE} (up >= {X:.0%}), kind SOXX  (bc_lib NOT imported)")
    say("=" * 118)
    say(f"Data: SOXX close {idx[0].date()}..{idx[-1].date()} N={N}; open from {df.O.first_valid_index().date()}; "
        f"overlap rows {chk['overlap_n']} max|diff|={chk['overlap_maxdiff']:.1e}; "
        f"XNYS missing={chk['cal_missing']} extra={chk['cal_extra']}")

    fr = frames(C, O, 0.10)
    bases = {k: matched_base(idx, fr, k) for k in ("hold20", "near5", "win20", "r20")}
    sig = up_rule(fr["up"], X)
    pos = first_signals(sig, fr, idx)

    # ============================================================ [1] independent match
    say("\n[1] Independent re-implementation vs family files (results/f1_SOXX_signals.csv, f1_SOXX_summary.csv)")
    fam = pd.read_csv(RES / "f1_SOXX_signals.csv", parse_dates=["date"])
    fam = fam[fam.rule == RULE].reset_index(drop=True)
    fsum = pd.read_csv(RES / "f1_SOXX_summary.csv")
    fsum = fsum[fsum.rule == RULE].set_index("period")
    mine = pd.DataFrame(dict(date=idx[pos], pos=pos, Ldd_date=idx[fr["Lpos"][pos]], Ldd=fr["Ldd"][pos],
                             C=C[pos], up=fr["up"][pos], dd60=fr["dd60"][pos], hold20=fr["hold20"][pos],
                             base_hold20=bases["hold20"][pos], near5=fr["near5"][pos],
                             base_near5=bases["near5"][pos], win20=fr["win20"][pos], base_win20=bases["win20"][pos],
                             r20=fr["r20"][pos], base_r20=bases["r20"][pos]))
    mine["sess_after_low"] = pos - fr["Lpos"][pos]
    say(mine.assign(date=mine.date.dt.date, Ldd_date=mine.Ldd_date.dt.date).round(4).to_string(index=False))
    same_dates = list(mine.date) == list(fam.date)
    say(f"  first-signal dates identical: {same_dates} (mine {len(mine)}, family {len(fam)})")
    mism = []
    if same_dates:
        for k in ("pos", "up", "dd60", "hold20", "base_hold20", "near5", "base_near5", "win20", "base_win20",
                  "r20", "base_r20"):
            a, b = mine[k].to_numpy(float), fam[k].to_numpy(float)
            bad = ~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-9))
            if bad.any():
                mism.append((k, int(bad.sum())))
        say(f"  per-signal field mismatches: {mism if mism else 'none'}")
    else:
        say(f"  only mine: {sorted(set(mine.date) - set(fam.date))}; only family: {sorted(set(fam.date) - set(mine.date))}")
    st0 = period_stats(pos, fr, idx, bases)
    say("  period       mine hold20(n)  family hold20(n)  mine base  family base  mine near5/win20/r20       family")
    per_ok = True
    for p, _, _ in PERIODS:
        r = st0[p]
        fh, fn, fb = fsum.loc[p, "hold20"], int(fsum.loc[p, "hold20_n"]), fsum.loc[p, "hold20_base"]
        okp = r["hold20_n"] == fn and abs(r["hold20"] - fh) < 1e-9 and abs(r["base_hold20"] - fb) < 1e-9
        for k in ("near5", "win20", "r20"):
            okp &= (np.isnan(r[k]) and np.isnan(fsum.loc[p, k])) or abs(r[k] - fsum.loc[p, k]) < 1e-9
        per_ok &= bool(okp)
        say(f"  {p:10s}   {r['hold20']:.4f} ({r['hold20_n']:2d})   {fh:.4f} ({fn:2d})       {r['base_hold20']:.4f}"
            f"     {fb:.4f}     {r['near5']:.3f}/{r['win20']:.3f}/{r['r20']:+.4f}   "
            f"{fsum.loc[p, 'near5']:.3f}/{fsum.loc[p, 'win20']:.3f}/{fsum.loc[p, 'r20']:+.4f}  {'OK' if okp else 'MISMATCH'}")
    indep_match = bool(same_dates and not mism and per_ok)
    say(f"  => independent match: {indep_match}")
    m0 = st0["2024~2026"]
    say(f"  main: n={m0['n']} resolved={m0['hold20_n']} hold20={m0['hold20']:.4f} (matched {m0['base_hold20']:.4f}) "
        f"near5={m0['near5']:.4f} (matched {m0['base_near5']:.4f}) win20={m0['win20']:.4f} (matched {m0['base_win20']:.4f}) "
        f"r20={m0['r20']:+.4f} (matched {m0['base_r20']:+.4f}) up_med={m0['up_med']:.4f}")
    # unresolved last signal, partial path
    for p in pos[np.isnan(fr["hold20"][pos])]:
        seen = C[p + 1:]
        say(f"  unresolved {idx[p].date()}: {len(seen)} of 20 sessions seen, min close so far {seen.min():.2f} vs Ldd "
            f"{fr['Ldd'][p]:.2f} -> {'still holding' if seen.min() >= fr['Ldd'][p] else 'ALREADY BROKEN'}")
    pool_m = fr["in_dd"] & ~np.isnan(fr["Ldd"]) & ~np.isnan(fr["hold20"]) & (idx >= EVAL_START) & (idx <= LAST)
    say(f"  unconditional in_dd hold20 2024~2026 = {fr['hold20'][pool_m].mean():.4f} (days {int(pool_m.sum())})")
    k_, n_ = int(round(m0["hold20"] * m0["hold20_n"])), m0["hold20_n"]
    for lab, p0 in (("unconditional in_dd", fr["hold20"][pool_m].mean()), ("matched", m0["base_hold20"])):
        pb = sum(comb(n_, j) * p0 ** j * (1 - p0) ** (n_ - j) for j in range(k_, n_ + 1))
        say(f"  binomial P(X >= {k_}/{n_} | p={p0:.4f}, {lab}) = {pb:.4f}")
    # L2 degeneracy: reproduce the registered block bootstrap on (hold20 - matched)
    ok = ~np.isnan(fr["hold20"][pos])
    diff = fr["hold20"][pos][ok] - bases["hold20"][pos][ok]
    lab = pos[ok] // 20
    lo, hi = block_boot_lo(diff, lab, np.random.default_rng(SEED))
    n26 = int((idx[pos][ok] >= pd.Timestamp("2026-01-02")).sum())
    say(f"  L2 check: hold20 - matched per signal = {np.round(diff, 4).tolist()}")
    say(f"    my block bootstrap 95% of mean diff = [{lo:+.4f}, {hi:+.4f}] (family dlo +0.0026). Only the {n26} 2026 signals "
        f"have diff > 0 (2026 8-15% bucket base 0.969 < 1); every diff >= 0 because all outcomes are 1.")
    say(f"    P(a resample of {len(diff)} blocks contains no 2026 signal) = ({len(diff) - n26}/{len(diff)})^{len(diff)} = "
        f"{((len(diff) - n26) / len(diff)) ** len(diff):.4f} < 0.025, so dlo > 0 is guaranteed by construction: the L2 pass "
        f"is degenerate (a rule that cannot lose vs a baseline slightly below 1), not evidence of information.")

    # ============================================================ [2] look-ahead audit
    say("\n[2] Look-ahead audit")
    trunc_ok = True
    allfire = np.flatnonzero(sig & fr["in_dd"] & (idx >= EVAL_START))
    for p in allfire:
        f_tr = frames(C[:p + 1], O[:p + 1], 0.10)
        if not (up_rule(f_tr["up"], X)[p] and f_tr["in_dd"][p] and abs(f_tr["Ldd"][p] - fr["Ldd"][p]) < 1e-12):
            trunc_ok = False
            say(f"  truncation mismatch at {idx[p].date()}")
    # also: no day that does NOT fire becomes a firing day with truncated data (checked on all eval days)
    ev = np.flatnonzero((idx >= EVAL_START) & ~np.isnan(fr["up"]))
    for p in ev[::7]:
        f_tr = frames(C[:p + 1], O[:p + 1], 0.10)
        if bool(up_rule(f_tr["up"], X)[p] and f_tr["in_dd"][p]) != bool(sig[p] and fr["in_dd"][p]):
            trunc_ok = False
            say(f"  truncation state mismatch at {idx[p].date()}")
    say(f"  truncation test: signal/in_dd/Ldd recomputed with data cut at t on all {len(allfire)} raw firing days and "
        f"every 7th eval day ({len(ev[::7])}): {'all identical' if trunc_ok else 'DIFFERENCES'}")
    say("  code read (f1_rebound.py build_rules): up8 = (fr['up'] >= 0.08) & fr['up'].notna(); fr['up'] = C/Ldd - 1 with")
    say("  Ldd = min(C[pk..t]), pk = latest 60d max position (window C[t-59..t]); in_dd = rolling(20).min of dd60 ending at t.")
    say("  No flow, news, VIX or other market data enter up8. sigma20 (ret.rolling(20).std, window ending at t) is only")
    say("  used by sig* rules. Outcomes start at t+1 (entry O[t+1]). The matched baseline averages same-year outcomes of")
    say("  other days (an evaluation benchmark, never a signal input). Residual: yfinance dividend-adjusted closes")
    say("  rescale history by later dividends (~0.6%/yr); can move a borderline first-crossing day by one session at most")
    say("  (covered by the delay test below). Posthoc lag1 in the family file is labelled 'posthoc' and not used for L1.")

    # ============================================================ [3] robustness
    say("\n[3] Robustness: hold20 (resolved n) per period; main = 2024~2026; L1 = registered SOXX criterion")
    rob = {}
    frc = {0.10: (fr, bases)}

    def run(label, x=X, cool=20, thr=0.10, delay=0, core=True):
        if thr not in frc:
            f = frames(C, O, thr)
            frc[thr] = (f, {k: matched_base(idx, f, k) for k in ("hold20", "near5", "win20", "r20")})
        f, bs = frc[thr]
        ps = first_signals(shift(up_rule(f["up"], x), delay), f, idx, cool)
        st = period_stats(ps, f, idx, bs)
        rob[label] = (st, core)
        say(fmt_row(label, st) + f"  matched {st['2024~2026']['base_hold20']:.3f} win20 {st['2024~2026']['win20']:.3f}"
            f"/{st['2024~2026']['base_win20']:.3f}  L1={l1(st)}")
        bad = [p for p in ps if f["hold20"][p] == 0]
        if bad:
            say("      failed hold20: " + ", ".join(f"{idx[p].date()} (up {f['up'][p]:.3f}, Ldd {idx[f['Lpos'][p]].date()})" for p in bad))
        return st

    run("registered up8 (cool20 thr10)")
    say("  -- registered grid neighbours (one step either side)")
    run("up5 (grid step below)", x=0.05)
    run("up10 (grid step above)", x=0.10)
    say("  -- finer neighbours (1pp)")
    for x in (0.06, 0.07, 0.09, 0.11, 0.12):
        run(f"up{x * 100:.0f}", x=x)
    say("  -- timing / cooldown / state threshold")
    run("delay +1 session", delay=1)
    run("delay +2 sessions", delay=2, core=False)
    run("cooldown 10", cool=10)
    run("cooldown 30", cool=30)
    run("in_dd thr 8%", thr=0.08)
    run("in_dd thr 12%", thr=0.12)
    core_l1 = {k: l1(v[0]) for k, v in rob.items() if v[1]}
    say(f"  L1 holds in {sum(core_l1.values())}/{len(core_l1)} core variants; fails: "
        f"{[k for k, v in core_l1.items() if not v] or 'none'}")
    say("  2026 halves for each variant (resolved n): " + "; ".join(
        f"{k.split(' (')[0]} H1 {v[0]['2026H1']['hold20']:.2f}({v[0]['2026H1']['hold20_n']}) H2 "
        f"{v[0]['2026H2']['hold20']:.2f}({v[0]['2026H2']['hold20_n']})" for k, v in rob.items()))
    say("  note: 2026H1 has exactly ONE resolved signal in every up>=7% variant, so the H1 '100%' is one observation.")

    # ============================================================ [4] null
    say("\n[4] Null: random in_dd dates in 2024~2026 (resolved hold20), same count and up-bucket mix, "
        "pairwise spacing > 20 sessions, 2000 draws")
    b_all = bucket(fr["up"])
    pool_pos = np.flatnonzero(pool_m)
    pool_b = b_all[pool_pos]
    pos_r = pos[~np.isnan(fr["hold20"][pos])]
    obs = float(fr["hold20"][pos_r].mean())
    labels = [int(b) for b in b_all[pos_r]]
    say(f"  observed hold20 = {obs:.4f} (n={len(pos_r)} resolved; the 13th signal 2026-09-29 is unresolved and excluded)")
    say(f"  bucket mix (0-3,3-8,8-15,15+) = {[labels.count(b) for b in range(4)]}; pool days by bucket = "
        f"{[int((pool_b == b).sum()) for b in range(4)]}; pool hold20 by bucket = "
        f"{[round(float(fr['hold20'][pool_pos][pool_b == b].mean()), 4) for b in range(4)]}")
    fails = pool_pos[(pool_b >= 2) & (fr["hold20"][pool_pos] == 0)]
    say(f"  pool days with up >= 8% that FAILED hold20: {[str(idx[p].date()) + f' (up {fr['up'][p]:.3f})' for p in fails]}")
    draws, tries = null_draws(pool_pos, pool_b, labels, fr["hold20"], 20, np.random.default_rng(SEED))
    p_null = float((draws >= obs - 1e-12).mean())
    say(f"  bucket-matched null: draws={len(draws)} (attempts {tries}); mean={draws.mean():.4f} sd={draws.std():.4f} "
        f"share==1.0: {(draws >= 1 - 1e-12).mean():.4f}")
    say(f"  p = P(null hold20 >= observed) = {p_null:.4f}")
    yb = [int(idx[p].year) * 10 + int(b_all[p]) for p in pos_r]
    pool_yb = idx.year.to_numpy()[pool_pos] * 10 + pool_b
    draws_y, _ = null_draws(pool_pos, pool_yb, yb, fr["hold20"], 20, np.random.default_rng(SEED + 1))
    p_null_y = float((draws_y >= obs - 1e-12).mean())
    say(f"  secondary year x bucket null: draws={len(draws_y)} mean={draws_y.mean():.4f} p={p_null_y:.4f}")
    # unmatched null: shows the hit rate is entirely the up-level effect
    draws_u, _ = null_draws(pool_pos, np.zeros(len(pool_pos), int), [0] * len(pos_r), fr["hold20"], 20,
                            np.random.default_rng(SEED + 2))
    p_null_u = float((draws_u >= obs - 1e-12).mean())
    say(f"  context, UNmatched null (any in_dd day): mean={draws_u.mean():.4f} p={p_null_u:.4f} "
        f"-> beating random in_dd days is entirely explained by the up bucket (the matched p above).")
    # 2026-only
    pos26 = pos_r[idx[pos_r] >= pd.Timestamp("2026-01-02")]
    pp26 = pool_pos[idx[pool_pos] >= pd.Timestamp("2026-01-02")]
    lab26 = [int(b) for b in b_all[pos26]]
    d26, _ = null_draws(pp26, b_all[pp26], lab26, fr["hold20"], 20, np.random.default_rng(SEED + 3))
    obs26 = float(fr["hold20"][pos26].mean())
    say(f"  2026-only bucket-matched null: observed {obs26:.3f} (n={len(pos26)}), null mean {d26.mean():.4f}, "
        f"p={float((d26 >= obs26 - 1e-12).mean()):.4f}")

    # ============================================================ [5] mechanical
    say("\n[5] Mechanical comparison")
    upmed = float(np.median(fr["up"][pos]))
    say(f"  candidate median up = {upmed:.4f} (all 13 first signals). up8 IS a plain 'up >= x' rule, so the comparison")
    say("  is with itself at a slightly higher threshold:")
    mech = {}
    for x in (upmed, 0.08, 0.10, 0.15):
        ps = first_signals(up_rule(fr["up"], x), fr, idx)
        mech[x] = period_stats(ps, fr, idx, bases)
        say(fmt_row(f"up >= {x:.4f}", mech[x]) + f"  matched {mech[x]['2024~2026']['base_hold20']:.3f}")
    say("  hold20 on ALL in_dd days 2024~2026 (no cooldown) by up level:")
    up_p = fr["up"][pool_pos]
    h_p = fr["hold20"][pool_pos]
    yr = idx.year.to_numpy()[pool_pos]
    for a, b in ((0, .03), (.03, .05), (.05, .08), (.08, .10), (.10, .15), (.15, 9)):
        m = (up_p >= a) & (up_p < b)
        per_y = " ".join(f"{y}:{h_p[m & (yr == y)].mean():.2f}({int((m & (yr == y)).sum())})" for y in (2024, 2025, 2026)
                         if (m & (yr == y)).any())
        say(f"    up {a:4.0%}-{b if b < 9 else np.inf:>5.0%}: hold20 {h_p[m].mean():.3f} (days {int(m.sum()):3d})   {per_y}")
    m_hi = up_p >= upmed
    say(f"  all in_dd days with up >= {upmed:.4f}: hold20 {h_p[m_hi].mean():.4f} (days {int(m_hi.sum())}); "
        f"with up < it: {h_p[~m_hi].mean():.4f}")
    # buffer test: is it just 'price must fall >= up/(1+up) within 20 sessions'?
    say("  buffer test: hold20 <=> 20-session forward min close stays above Ldd, i.e. no fall of more than up/(1+up)")
    say("  from the signal close. Share of ALL 2024~2026 sessions (no in_dd, no rebound condition) whose 20d forward")
    say("  min close stays within that same required fall:")
    allp = np.flatnonzero((idx >= EVAL_START) & (np.arange(N) + 20 <= N - 1))
    fmin = np.array([C[t + 1:t + 21].min() / C[t] - 1 for t in allp])
    rates = []
    for p in pos_r:
        req = fr["Ldd"][p] / C[p] - 1
        rates.append(float((fmin >= req - 1e-12).mean()))
    say(f"    per signal required fall: {[round(float(fr['Ldd'][p] / C[p] - 1), 3) for p in pos_r]}")
    say(f"    random-session survival for the same buffers: {[round(r, 3) for r in rates]}; mean {np.mean(rates):.4f} "
        f"vs observed {obs:.4f}")
    say(f"    -> any SOXX day of 2024~2026 with the same cushion survives 20 sessions {np.mean(rates):.0%} of the time; the rule"
        f" adds the in_dd context, worth about {obs - np.mean(rates):+.3f}.")
    # entry quality
    say("  entry quality (the thing a buyer gets): main near5 "
        f"{m0['near5']:.3f} vs matched {m0['base_near5']:.3f}; win20 {m0['win20']:.3f} vs matched {m0['base_win20']:.3f}; "
        f"r20 {m0['r20']:+.4f} vs matched {m0['base_r20']:+.4f}; 2026 win20 {st0['2026']['win20']:.3f} vs "
        f"{st0['2026']['base_win20']:.3f}, r20 {st0['2026']['r20']:+.4f} vs {st0['2026']['base_r20']:+.4f}")
    say(f"  lateness: median up at confirmation {upmed:.1%}; median sessions after the low "
        f"{int(np.median(mine.sess_after_low))} (range {mine.sess_after_low.min()}-{mine.sess_after_low.max()}).")

    # ============================================================ context (post-hoc, outside registered window)
    say("\n[context, post-hoc, not a pass criterion] 2023 (Ldd valid from late Mar 2023) with the same rule")
    pos23 = first_signals(sig, fr, idx, 20, start=pd.Timestamp("2023-01-03"))
    pos23 = pos23[idx[pos23] < EVAL_START]
    m23 = fr["in_dd"] & ~np.isnan(fr["hold20"]) & (idx < EVAL_START) & up_rule(fr["up"], X)
    say(f"  2023 first signals: {[(str(idx[p].date()), round(float(fr['up'][p]), 3), float(fr['hold20'][p])) for p in pos23]}; "
        f"all 2023 in_dd days with up>=8%: hold20 {np.nanmean(fr['hold20'][m23]) if m23.any() else np.nan:.3f} "
        f"(days {int(m23.sum())})")
    say("  regime risk: no 2022-type bear market in the data (no network); bear-market rallies of 8-15% that fail")
    say("  within a month are the obvious failure mode and are untested here.")

    # ============================================================ [6] verdict
    say("\n[6] Verdict inputs")
    say(f"  independent match={indep_match}; look-ahead found={not trunc_ok}; main hold20={m0['hold20']:.4f} "
        f"(n={m0['hold20_n']}) vs matched {m0['base_hold20']:.4f}; bucket-matched null p={p_null:.4f} "
        f"(year x bucket p={p_null_y:.4f}, 2026-only p={float((d26 >= obs26 - 1e-12).mean()):.4f})")
    say(f"  L1 robust: {sum(core_l1.values())}/{len(core_l1)} core variants; lower grid neighbour up5 L1={core_l1['up5 (grid step below)']}")
    verdict = "fragile"
    if not indep_match or not trunc_ok:
        verdict = "refuted"
    elif p_null < 0.05 and all(core_l1.values()):
        verdict = "confirmed"
    say(f"\nVERDICT: {verdict}")
    say("  Reproduced exactly (13 first signals, all fields, all period rates) and no look-ahead. The 12/12 hold20 is")
    say(f"  real in-sample but mechanical: up8 is the 'already rebounded' variable itself. In-dd days with up 8-15% held")
    say(f"  {h_p[(up_p >= .08) & (up_p < .15)].mean():.1%} and 15%+ held {h_p[up_p >= .15].mean():.1%} regardless of any rule, so random dates with the same")
    say(f"  rebound mix reach 100% in {(draws >= 1 - 1e-12).mean():.0%} of draws (p={p_null:.2f}). L2 dlo>0 is guaranteed arithmetically")
    say("  (outcomes all 1, 2026 baseline 0.969). Sensitivity: the lower neighbours up5/up6 fail L1 (75%/73%, 2026 60%);")
    say("  2026 in_dd days with up 5-8% held only 41%, so 8% sits on the 2026 knee. The 100% is not stable either:")
    say("  cooldown 10 adds 2026-06-30 (up 8.6%, failed), in_dd 8% adds 2025-02-19 and 2026-02-11 (up 8.5%/9.0%, failed;")
    say("  2026H1 0.67), delay +2 gives 0.917 (2026H2 0.67). The 2026H1 cell is one signal. Entry quality is no better")
    say("  than matched (near5 0.455 vs 0.519, win20 0.583 vs 0.726, r20 equal). Usable only as a descriptive statement")
    say("  'after +8~9% off the low inside a >=10% drawdown, the low was rarely retested within a month (2024~2026)'; not a")
    say("  bottom-timing signal with information beyond the rebound already paid for. No pre-2023 bear market is tested.")
    OUT.write_text(buf.getvalue(), encoding="utf-8")
    return dict(indep_match=indep_match, lookahead=not trunc_ok, hold20=m0["hold20"], p=p_null, verdict=verdict)


if __name__ == "__main__":
    main()
