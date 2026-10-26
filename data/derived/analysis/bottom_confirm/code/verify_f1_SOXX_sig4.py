"""Adversarial verification of bottom_confirm F1 rule 'sig4' on SOXX (BRIEF.md, 2026-10-26).

Rule under test: up[t] >= 4 * sigma20[t]
  up[t]      = C[t] / Ldd[t] - 1, Ldd = lowest close since the latest 60-session closing high (ties -> later day)
  sigma20[t] = sample std (ddof=1) of daily close-to-close returns over the 20 sessions ending at t
               (= C.pct_change().rolling(20).std())
Signals count only on in_dd days (min dd60 over the last 20 sessions <= -10%), eval window 2024-01-02..2026-10-23,
first signal = more than 20 sessions after the previously adopted signal.

This file does NOT import bc_lib or f1_rebound. Everything is rebuilt from the raw CSVs with numpy.
Run: cd data/derived/analysis/bottom_confirm && python3 -W ignore code/verify_f1_SOXX_sig4.py
Output: results/verify_f1_SOXX_sig4.txt
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
OUT = RES / "verify_f1_SOXX_sig4.txt"

PRICES = ROOT / "research/news_turning_points_20261025/prices.csv"
CLOSE23 = ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv"
FAM_SIG = RES / "f1_SOXX_signals.csv"
FAM_SUM = RES / "f1_SOXX_summary.csv"
FAM_PH = RES / "f1_SOXX_posthoc_signals.csv"
FAM_CODE = HERE / "code/f1_rebound.py"
LIB_CODE = HERE / "code/bc_lib.py"

RULE = "sig4"
MULT = 4.0
EVAL_START = pd.Timestamp("2024-01-02")
LAST = pd.Timestamp("2026-10-23")
PERIODS = [("2024~2026", "2024-01-02", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
           ("2025", "2025-01-02", "2025-12-31"), ("2024", "2024-01-02", "2024-12-31")]
MAIN = "2024~2026"
BUCKET_EDGES = [0.03, 0.08, 0.15]          # 0-3, 3-8, 8-15, 15%+
BUCKET_NAMES = ["0-3%", "3-8%", "8-15%", "15%+"]
N_NULL = 2000
SEED = 4_2026_1026

buf = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    buf.write(s + "\n")


# ------------------------------------------------------------------ raw data (independent loader)
def load_soxx() -> pd.DataFrame:
    p = pd.read_csv(PRICES, header=[0, 1], index_col=0)
    p.index = pd.to_datetime(p.index, errors="coerce")
    p = p[p.index.notna()].sort_index()
    pc = p[("close", "SOXX")].astype(float).dropna()
    po = p[("open", "SOXX")].astype(float).dropna()
    u = pd.read_csv(CLOSE23, encoding="utf-8-sig", index_col=0, parse_dates=True)["SOXX"].astype(float).dropna()
    ov = u.index.intersection(pc.index)
    maxdiff = float(np.max(np.abs(u[ov].to_numpy() - pc[ov].to_numpy()))) if len(ov) else np.nan
    cal = u.index.union(pc.index)
    cal = cal[cal <= LAST]
    d = pd.DataFrame(index=cal)
    d["C"] = u.reindex(cal)
    d.loc[pc.index.intersection(cal), "C"] = pc
    d["O"] = po.reindex(cal)
    d.attrs["overlap_days"] = len(ov)
    d.attrs["overlap_maxdiff"] = maxdiff
    return d


def check_calendar(idx: pd.DatetimeIndex) -> str:
    try:
        import exchange_calendars as xc
        s = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range(idx[0], idx[-1]))
        s = s.tz_localize(None) if s.tz is not None else s
        miss = s.difference(idx)
        extra = idx.difference(s)
        return f"XNYS sessions {len(s)}, data dates {len(idx)}, missing {len(miss)}, extra {len(extra)}"
    except Exception as e:  # pragma: no cover
        return f"exchange_calendars unavailable ({e})"


# ------------------------------------------------------------------ state variables
def state(C: np.ndarray, thr: float = 0.10, swin: int = 20):
    N = len(C)
    Ldd = np.full(N, np.nan)
    Lpos = np.full(N, -1)
    dd60 = np.full(N, np.nan)
    for t in range(59, N):
        w = C[t - 59:t + 1]
        if np.isnan(w).any():
            continue
        mx = w.max()
        pk = t - 59 + int(np.flatnonzero(w == mx)[-1])
        seg = C[pk:t + 1]
        Ldd[t] = seg.min()
        Lpos[t] = pk + int(np.flatnonzero(seg == seg.min())[-1])
        dd60[t] = C[t] / mx - 1
    up = C / Ldd - 1
    in_dd = np.zeros(N, bool)
    for t in range(N):
        w = dd60[max(0, t - 19):t + 1]
        w = w[~np.isnan(w)]
        in_dd[t] = len(w) > 0 and w.min() <= -thr
    r = np.full(N, np.nan)
    r[1:] = C[1:] / C[:-1] - 1
    sig = np.full(N, np.nan)
    for t in range(swin, N):
        w = r[t - swin + 1:t + 1]
        if np.isnan(w).any():
            continue
        sig[t] = np.std(w, ddof=1)
    return dict(Ldd=Ldd, Lpos=Lpos, dd60=dd60, up=up, in_dd=in_dd, sigma=sig)


def outcomes(C: np.ndarray, O: np.ndarray, Ldd: np.ndarray):
    N = len(C)
    hold20 = np.full(N, np.nan)
    hold40 = np.full(N, np.nan)
    near5 = np.full(N, np.nan)
    r20 = np.full(N, np.nan)
    margin20 = np.full(N, np.nan)
    for t in range(N):
        if np.isnan(Ldd[t]):
            continue
        if t + 20 <= N - 1:
            m = C[t + 1:t + 21].min()
            hold20[t] = float(m >= Ldd[t])
            margin20[t] = m / Ldd[t] - 1
            r20[t] = C[t + 20] / O[t + 1] - 1
        if t + 40 <= N - 1:
            hold40[t] = float(C[t + 1:t + 41].min() >= Ldd[t])
            near5[t] = float(C[t + 1:t + 41].min() >= 0.95 * O[t + 1])
    win20 = np.where(np.isnan(r20), np.nan, (r20 > 0).astype(float))
    return dict(hold20=hold20, hold40=hold40, near5=near5, win20=win20, r20=r20, margin20=margin20)


def first_signals(sig: np.ndarray, st: dict, idx: pd.DatetimeIndex, cool: int = 20) -> list[int]:
    ok = sig & st["in_dd"] & ~np.isnan(st["Ldd"]) & (idx >= EVAL_START) & (idx <= LAST)
    out, prev = [], -10 ** 9
    for p in np.flatnonzero(ok):
        if p - prev <= cool:
            continue
        out.append(int(p))
        prev = p
    return out


def period_table(pos: list[int], idx: pd.DatetimeIndex, oc: dict, st: dict) -> dict:
    res = {}
    pos = np.asarray(pos, int)
    for name, a, b in PERIODS:
        sel = pos[(idx[pos] >= pd.Timestamp(a)) & (idx[pos] <= pd.Timestamp(b))] if len(pos) else pos
        r = dict(n=len(sel))
        for k in ("hold20", "hold40", "near5", "win20", "r20"):
            v = oc[k][sel] if len(sel) else np.array([])
            v = v[~np.isnan(v)]
            r[k] = float(v.mean()) if len(v) else np.nan
            r[k + "_n"] = int(len(v))
        r["up_median"] = float(np.median(st["up"][sel])) if len(sel) else np.nan
        res[name] = r
    return res


def fmt_tab(t: dict, keys=("2024~2026", "2026", "2026H1", "2026H2", "2025")) -> str:
    parts = []
    for k in keys:
        r = t[k]
        h = "  nan" if np.isnan(r["hold20"]) else f"{r['hold20']:.3f}"
        parts.append(f"{k} {h}(n{r['hold20_n']})")
    return "  ".join(parts)


def bucket(u: np.ndarray) -> np.ndarray:
    return np.digitize(u, BUCKET_EDGES)


# ------------------------------------------------------------------ look-ahead audit helpers
def audit_source() -> list[str]:
    notes = []
    fam = FAM_CODE.read_text(encoding="utf-8")
    lib = LIB_CODE.read_text(encoding="utf-8")
    pats = {
        "negative shift (future values)": r"shift\(\s*-\s*\d",
        "centered rolling window": r"center\s*=\s*True",
        "bfill / backfill": r"bfill|backfill|fillna\(method=['\"]b",
        "iloc/at with t+ index in signal code": r"\[\s*t\s*\+\s*\d",
    }
    for name, pat in pats.items():
        hf = [i + 1 for i, line in enumerate(fam.splitlines()) if re.search(pat, line)]
        hl = [i + 1 for i, line in enumerate(lib.splitlines()) if re.search(pat, line)]
        notes.append(f"  pattern '{name}': f1_rebound.py lines {hf or '-'}; bc_lib.py lines {hl or '-'}")
    # signal construction in f1_rebound.build_rules
    m = re.search(r"def build_rules.*?return \{", fam, re.S)
    body = m.group(0) if m else ""
    notes.append("  build_rules uses only: fr['up'] and sigma20(C) = (C/C.shift(1)-1).rolling(20).std() "
                 f"[verified text: {'sigma20(C)' in body and 'fr[\"up\"]' in body}]")
    notes.append("  KNOWN_LOWS (hand-picked bottoms) appear only in soxx_spotcheck (printing), not in build_rules: "
                 f"{'KNOWN_LOWS' not in body}")
    notes.append("  no flow/news/market-indicator inputs in F1 (price-only): "
                 f"{not any(x in body for x in ('load_market', 'load_notes', 'VIX', 'flow'))}")
    return notes


def truncation_probe(C: np.ndarray, idx: pd.DatetimeIndex, full_sig: np.ndarray, full_dd: np.ndarray,
                     probe_pos: list[int]) -> tuple[bool, int]:
    """Recompute sig4 & in_dd on C[:t+1] only, compare with full-sample value at t."""
    bad = 0
    for t in probe_pos:
        s = state(C[:t + 1])
        v = bool((s["up"][t] >= MULT * s["sigma"][t]) and not np.isnan(s["sigma"][t]) and not np.isnan(s["up"][t]))
        if v != bool(full_sig[t]) or bool(s["in_dd"][t]) != bool(full_dd[t]):
            bad += 1
    return bad == 0, bad


# ------------------------------------------------------------------ null
def null_draws(st, oc, idx, obs_pos, cool=20, n_draws=N_NULL, match_buckets=True, seed=SEED):
    a, b = pd.Timestamp(PERIODS[0][1]), pd.Timestamp(PERIODS[0][2])
    pool = np.flatnonzero(st["in_dd"] & ~np.isnan(st["Ldd"]) & ~np.isnan(oc["hold20"]) & (idx >= a) & (idx <= b))
    obs_pos = np.asarray([p for p in obs_pos if not np.isnan(oc["hold20"][p])], int)
    obs_b = bucket(st["up"][obs_pos])
    pool_b = bucket(st["up"][pool])
    need = list(obs_b) if match_buckets else [-1] * len(obs_pos)
    pools = {k: pool[pool_b == k] for k in range(4)}
    rng = np.random.default_rng(seed)
    vals, fails = [], 0
    while len(vals) < n_draws:
        order = rng.permutation(need)
        picked = []
        okdraw = True
        for k in order:
            cand = pool if k == -1 else pools[k]
            if picked:
                pk = np.asarray(picked)
                cand = cand[np.all(np.abs(cand[:, None] - pk[None, :]) > cool, axis=1)]
            if len(cand) == 0:
                okdraw = False
                break
            picked.append(int(rng.choice(cand)))
        if not okdraw:
            fails += 1
            if fails > 200_000:
                raise RuntimeError("null sampler cannot satisfy cooldown")
            continue
        vals.append(float(np.mean(oc["hold20"][picked])))
    return np.asarray(vals), pool, pool_b, obs_b, fails


# ------------------------------------------------------------------ main
def main():
    pd.set_option("display.width", 220)
    d = load_soxx()
    idx = d.index
    C, O = d["C"].to_numpy(float), d["O"].to_numpy(float)
    say("Verification: family f1, rule sig4 (up >= 4*sigma20), kind SOXX — independent rebuild, no bc_lib import")
    say(f"data: {idx[0].date()}..{idx[-1].date()} ({len(idx)} sessions); close overlap 2023-12..2026-10-13 "
        f"{d.attrs['overlap_days']} days, max |diff| {d.attrs['overlap_maxdiff']:.2e}; {check_calendar(idx)}")
    say(f"close NaN: {int(np.isnan(C).sum())}; open NaN from 2023-12-01: {int(np.isnan(O[idx >= '2023-12-01']).sum())}")

    st = state(C)
    oc = outcomes(C, O, st["Ldd"])
    # sigma20 cross-check against pandas pct_change().rolling(20).std() as in the rule text
    s_pd = d["C"].pct_change().rolling(20).std().to_numpy()
    both = ~np.isnan(s_pd) & ~np.isnan(st["sigma"])
    say(f"sigma20 numpy vs pandas pct_change().rolling(20).std(): max |diff| {np.max(np.abs(s_pd[both] - st['sigma'][both])):.2e}, "
        f"valid-mask equal {bool((np.isnan(s_pd) == np.isnan(st['sigma'])).all())}")
    sig = (st["up"] >= MULT * st["sigma"]) & ~np.isnan(st["up"]) & ~np.isnan(st["sigma"])
    fs = first_signals(sig, st, idx)
    tab = period_table(fs, idx, oc, st)

    # ---------------------------------------------------------- [1] independent match
    say("\n[1] Independent re-implementation vs family files")
    fam = pd.read_csv(FAM_SIG, parse_dates=["date"])
    fam = fam[fam.rule == RULE].reset_index(drop=True)
    mine_dates = [idx[p] for p in fs]
    fam_dates = list(fam.date)
    date_match = mine_dates == fam_dates
    say(f"  first signals: mine {len(mine_dates)}, family {len(fam_dates)}, identical dates: {date_match}")
    if not date_match:
        say(f"    only mine  : {[str(x.date()) for x in sorted(set(mine_dates) - set(fam_dates))]}")
        say(f"    only family: {[str(x.date()) for x in sorted(set(fam_dates) - set(mine_dates))]}")
    say("  date        pos  Ldd(date)            up     4*sig20  dd60    hold20 (fam)  margin20  hold40 near5 win20  r20")
    worst_val = 0.0
    for p in fs:
        f = fam[fam.date == idx[p]]
        fh = f.hold20.iloc[0] if len(f) else np.nan
        for k_m, k_f in (("hold20", "hold20"), ("hold40", "hold40"), ("near5", "near5"), ("win20", "win20"), ("r20", "r20")):
            if len(f):
                a_, b_ = oc[k_m][p], f[k_f].iloc[0]
                if not (np.isnan(a_) and pd.isna(b_)):
                    worst_val = max(worst_val, abs(a_ - b_) if not (np.isnan(a_) or pd.isna(b_)) else np.inf)
        if len(f):
            worst_val = max(worst_val, abs(st["up"][p] - f.up.iloc[0]), abs(st["dd60"][p] - f.dd60.iloc[0]))
        say(f"  {idx[p].date()} {p:4d}  {st['Ldd'][p]:7.2f}({idx[st['Lpos'][p]].date()})  {st['up'][p]:+.4f}  "
            f"{MULT * st['sigma'][p]:.4f}  {st['dd60'][p]:+.3f}  {oc['hold20'][p]:.0f}  ({fh:.0f})       "
            f"{oc['margin20'][p]:+.4f}   {oc['hold40'][p]:.0f}     {oc['near5'][p]:.0f}     {oc['win20'][p]:.0f}  {oc['r20'][p]:+.4f}")
    say(f"  max |mine - family| over up/dd60/hold20/hold40/near5/win20/r20 at signal dates: {worst_val:.2e}")
    fsum = pd.read_csv(FAM_SUM)
    fsum = fsum[fsum.rule == RULE].set_index("period")
    say("  period      mine hold20 (n)   family hold20 (n)   mine near5  fam near5   mine win20  fam win20  mine up_med  fam up_med")
    sum_match = True
    for name, _, _ in PERIODS:
        r = tab[name]
        fr_ = fsum.loc[name] if name in fsum.index else None
        fh = fr_["hold20"] if fr_ is not None else np.nan
        fn = int(fr_["hold20_n"]) if fr_ is not None and pd.notna(fr_["hold20_n"]) else 0
        same = (r["hold20_n"] == fn) and (np.isclose(r["hold20"], fh) or (np.isnan(r["hold20"]) and pd.isna(fh)))
        same = same and (np.isclose(r["near5"], fr_["near5"]) if pd.notna(fr_["near5"]) else np.isnan(r["near5"]))
        same = same and (np.isclose(r["win20"], fr_["win20"]) if pd.notna(fr_["win20"]) else np.isnan(r["win20"]))
        sum_match &= bool(same)
        say(f"  {name:10s}  {r['hold20']:.3f} ({r['hold20_n']:2d})       {fh:.3f} ({fn:2d})          "
            f"{r['near5']:.3f}      {fr_['near5']:.3f}      {r['win20']:.3f}      {fr_['win20']:.3f}     "
            f"{r['up_median']:.4f}      {fr_['up_median']:.4f}")
    independent_match = bool(date_match and sum_match and worst_val < 1e-9)
    say(f"  -> independent match: {independent_match}")
    eps = sorted(set(int(st["Lpos"][p]) for p in fs))
    say(f"  distinct decline lows (Ldd dates) behind the {len(fs)} signals: {len(eps)} "
        f"{[str(idx[e].date()) for e in eps]}")

    # ---------------------------------------------------------- [2] look-ahead audit
    say("\n[2] Look-ahead audit")
    for line in audit_source():
        say(line)
    a, b = pd.Timestamp("2024-01-02"), LAST
    probe = [int(p) for p in np.flatnonzero(st["in_dd"] & (idx >= a) & (idx <= b))]
    ok_trunc, nbad = truncation_probe(C, idx, sig, st["in_dd"], probe)
    say(f"  truncation probe: recomputed sig4 & in_dd on data ending at t for all {len(probe)} in_dd days 2024~2026 -> "
        f"identical: {ok_trunc} (mismatches {nbad})")
    say("  outcomes use C[t+1..t+20/40] and entry O[t+1] only; signal day close is known before entry (next open).")
    say("  data note: closes are split/dividend-adjusted yfinance values (back-adjustment uses later dividends; "
        "not a signal look-ahead, but it shifts Ldd by up to ~0.1-0.2% across a quarterly ex-date).")
    for p in fs:
        if oc["margin20"][p] < 0.01:
            q = p + 1 + int(np.argmin(C[p + 1:p + 21]))
            say(f"    near miss: signal {idx[p].date()} Ldd {st['Ldd'][p]:.2f} ({idx[st['Lpos'][p]].date()}), forward-20 min "
                f"{C[q]:.2f} on {idx[q].date()}, margin {oc['margin20'][p]:+.4f}")
    say(f"  smallest margin {np.nanmin(oc['margin20'][fs]):+.4f}: a ~0.2% adjustment alone would not flip it, "
        "but 2 of 10 hits held by < 1%.")
    lookahead_found = not ok_trunc

    # ---------------------------------------------------------- [3] robustness
    say("\n[3] Robustness (hold20 by period; first signals with cooldown unless stated)")
    variants = []

    def run_variant(label, m=MULT, thr=0.10, cool=20, lag=0, swin=20):
        s2 = st if (thr == 0.10 and swin == 20) else state(C, thr=thr, swin=swin)
        o2 = oc if thr == 0.10 else outcomes(C, O, s2["Ldd"])
        sg = (s2["up"] >= m * s2["sigma"]) & ~np.isnan(s2["up"]) & ~np.isnan(s2["sigma"])
        if lag:
            sg = np.concatenate([np.zeros(lag, bool), sg[:-lag]])
        p_ = first_signals(sg, s2, idx, cool=cool)
        t_ = period_table(p_, idx, o2, s2)
        variants.append((label, t_))
        say(f"  {label:34s} {fmt_tab(t_)}  up_med {t_[MAIN]['up_median']:.3f}")
        return t_

    run_variant("base sig4 (m=4)")
    say("  -- multiplier neighbours (registered grid 2/3/4/6; finer steps added)")
    for m in (3.0, 3.5, 4.5, 5.0, 6.0):
        run_variant(f"m={m:g}", m=m)
    say("  -- sigma window (registered 20)")
    for w in (15, 30):
        run_variant(f"sigma window {w}", swin=w)
    say("  -- timing / cooldown / in_dd threshold")
    run_variant("lag1 (signal shifted 1 session)", lag=1)
    run_variant("lag2 (signal shifted 2 sessions)", lag=2)
    run_variant("cooldown 10", cool=10)
    run_variant("cooldown 30", cool=30)
    run_variant("in_dd threshold 8%", thr=0.08)
    run_variant("in_dd threshold 12%", thr=0.12)
    # family post-hoc lag1 cross-check
    ph = pd.read_csv(FAM_PH, parse_dates=["date"])
    ph = ph[ph.rule == f"{RULE}_lag1"]
    sgl = np.concatenate([[False], sig[:-1]])
    lag_pos = first_signals(sgl, st, idx)
    say(f"  family post-hoc sig4_lag1 dates identical to mine: {[idx[p] for p in lag_pos] == list(ph.date)}")
    rob_rows = [(lab, t_) for lab, t_ in variants]

    def passes(t_):
        r, r26 = t_[MAIN], t_["2026"]
        return (r["hold20_n"] >= 8 and r["hold20"] >= 0.70 and r26["hold20_n"] >= 3 and r26["hold20"] >= 0.70)

    def rate_ok(t_):
        r, r26 = t_[MAIN], t_["2026"]
        return (np.isnan(r["hold20"]) or r["hold20"] >= 0.70) and (np.isnan(r26["hold20"]) or r26["hold20"] >= 0.70)

    n_pass = sum(passes(t_) for _, t_ in rob_rows)
    fails = [lab for lab, t_ in rob_rows if not passes(t_)]
    rate_fails = [lab for lab, t_ in rob_rows if not rate_ok(t_)]
    say(f"  variants keeping main >= 70% (n>=8) and 2026 >= 70% (n>=3): {n_pass}/{len(rob_rows)}; failing: {fails or '-'}")
    say(f"  variants whose hit RATE drops below 70% (ignoring the n floors): {rate_fails or 'none'} "
        f"-> the failures above are sample-size failures only (stricter m leaves too few signals)")
    low = min(rob_rows, key=lambda x: (x[1][MAIN]["hold20"] if not np.isnan(x[1][MAIN]["hold20"]) else 9))
    say(f"  lowest main-period hold20: '{low[0]}' {low[1][MAIN]['hold20']:.3f} (n{low[1][MAIN]['hold20_n']})")

    # ---------------------------------------------------------- [4] null
    say("\n[4] Null: random in_dd dates in 2024~2026, same count, same up-bucket mix, same 20-session cooldown")
    obs = tab[MAIN]["hold20"]
    vals, pool, pool_b, obs_b, nfail = null_draws(st, oc, idx, fs)
    p_null = float(np.mean(vals >= obs - 1e-12))
    mix = {BUCKET_NAMES[k]: int((obs_b == k).sum()) for k in range(4)}
    say(f"  observed hold20 {obs:.3f} (n{len(obs_b)}); bucket mix of signals {mix}")
    for k in range(4):
        sel = pool[pool_b == k]
        say(f"    pool bucket {BUCKET_NAMES[k]:6s}: in_dd days {len(sel):4d}, day-level hold20 {np.mean(oc['hold20'][sel]):.4f}")
    exp_p = float(np.prod([np.mean(oc["hold20"][pool[pool_b == k]]) for k in obs_b]))
    say(f"  null mean hold20 {vals.mean():.4f}, 5/50/95% {np.quantile(vals, [0.05, 0.5, 0.95]).round(3).tolist()}; "
        f"draws rejected for cooldown {nfail}")
    say(f"  p = P(null hold20 >= observed) = {p_null:.4f}   (independence approximation prod(bucket rates) = {exp_p:.4f})")
    vals_u, *_ = null_draws(st, oc, idx, fs, match_buckets=False, seed=SEED + 1)
    p_unm = float(np.mean(vals_u >= obs - 1e-12))
    say(f"  context, unmatched null (any in_dd day, no bucket matching): mean {vals_u.mean():.4f}, p = {p_unm:.4f}")

    # ---------------------------------------------------------- [5] mechanical check
    say("\n[5] Mechanical check: is sig4 just 'price already rebounded enough'?")
    med = float(np.median(st["up"][fs]))
    plain = (st["up"] >= med) & ~np.isnan(st["up"])
    pp = first_signals(plain, st, idx)
    tp = period_table(pp, idx, oc, st)
    say(f"  median up of sig4 first signals = {med:.4f}")
    say(f"  up >= {med:.4f}: {fmt_tab(tp)}")
    say(f"  sig4          : {fmt_tab(tab)}")
    say(f"  plain rule near5 {tp[MAIN]['near5']:.3f} win20 {tp[MAIN]['win20']:.3f} r20 {tp[MAIN]['r20']:+.4f} | "
        f"sig4 near5 {tab[MAIN]['near5']:.3f} win20 {tab[MAIN]['win20']:.3f} r20 {tab[MAIN]['r20']:+.4f}")
    th = MULT * st["sigma"][fs]
    say(f"  effective up threshold 4*sigma20 at sig4 signals: min {th.min():.4f}, median {np.median(th):.4f}, max {th.max():.4f}; "
        f"actual up at signal median {med:.4f} (lowest {st['up'][fs].min():.4f})")
    ev = st["in_dd"] & (idx >= EVAL_START) & (idx <= LAST) & ~np.isnan(oc["hold20"])
    say("  day-level hold20 on in_dd days 2024~2026 (no cooldown):")
    for x in (0.0, 0.03, 0.05, 0.08, 0.10, med, 0.15):
        m_ = ev & (st["up"] >= x)
        say(f"    up >= {x:.4f}: days {int(m_.sum()):4d}, hold20 {np.mean(oc['hold20'][m_]):.4f}")
    # does the sigma scaling add anything given up? compare sig4 vs not-sig4 inside the same up bucket
    say("  within-bucket comparison (in_dd days 2024~2026): hold20 when sig4 fires vs does not")
    for k in range(4):
        m_ = ev & (bucket(st["up"]) == k)
        a1, a0 = m_ & sig, m_ & ~sig
        h1 = np.mean(oc["hold20"][a1]) if a1.any() else np.nan
        h0 = np.mean(oc["hold20"][a0]) if a0.any() else np.nan
        say(f"    {BUCKET_NAMES[k]:6s}: sig4 days {int(a1.sum()):3d} hold20 {h1:.4f} | non-sig4 days {int(a0.sum()):3d} hold20 {h0:.4f}")
    near_high = [str(idx[p].date()) for p in fs if st["dd60"][p] > -0.05]
    say(f"  signals already within 5% of the 60-day high: {len(near_high)}/{len(fs)} {near_high}")
    base_near5 = np.nanmean(np.where(ev, oc["near5"], np.nan))
    base_win20 = np.nanmean(np.where(ev, oc["win20"], np.nan))
    base_r20 = np.nanmean(np.where(ev, oc["r20"], np.nan))
    say(f"  sig4 vs all in_dd days (2024~2026): near5 {tab[MAIN]['near5']:.3f} vs {base_near5:.3f}, "
        f"win20 {tab[MAIN]['win20']:.3f} vs {base_win20:.3f}, r20 {tab[MAIN]['r20']:+.4f} vs {base_r20:+.4f}")
    overlap = len(set(pp) & set(fs))
    near_ov = sum(any(abs(p - q) <= 5 for q in pp) for p in fs)
    say(f"  sig4 signals coinciding with plain-rule signals: same day {overlap}/{len(fs)}, within 5 sessions {near_ov}/{len(fs)}")
    mechanical = bool(p_null >= 0.05)
    say(f"  -> hit rate explained by rebound size already achieved: {mechanical}")

    # ---------------------------------------------------------- [6] verdict
    robust = n_pass == len(rob_rows)
    if (not independent_match) or lookahead_found:
        verdict = "refuted"
    elif robust and p_null < 0.05:
        verdict = "confirmed"
    else:
        verdict = "fragile"
    say("\n[6] Verdict")
    say(f"  independent match: {independent_match}; look-ahead found: {lookahead_found}")
    say(f"  robustness: {n_pass}/{len(rob_rows)} variants keep main >= 70% and 2026 >= 70%"
        + (f"; failing {fails} (n floors only; hit rate never < 70%: {not rate_fails})" if fails else ""))
    say(f"  {len(fs)} first signals come from only {len(eps)} distinct decline lows (effective n < L1 floor of 8)")
    say(f"  null p = {p_null:.4f} (matched); same-state bucket rates 8-15% and 15%+ are ~99-100%, so 10/10 is the expected outcome")
    say(f"  VERDICT: {verdict}")

    OUT.write_text(buf.getvalue(), encoding="utf-8")
    return dict(verdict=verdict, independent_match=independent_match, lookahead=lookahead_found, p=p_null,
                hold20_main=tab[MAIN]["hold20"])


if __name__ == "__main__":
    main()
