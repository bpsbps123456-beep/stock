"""Independent adversarial verification (round 2, fixed right-censoring) of f6 / SOXX / news_GSR_posthoc.

Rule under test (code/f6_news_flow.py): signal day t = entry session of a census note with family == 'R'
and bank == 'GS' (n4h14_signals.csv). POST-HOC: Goldman PB 'R' notes were picked in news_followup on the
same 2024~2026 data. The verdict here asks whether the 11/11 hold20 is a real 70%+ bottom confirmation.

Does NOT import bc_lib. Re-implements SOXX prices, Ldd, in_dd, 20-session cooldown, outcomes with the fixed
censoring rule (0 if a close < Ldd is already seen in an incomplete window, NaN only when no break yet), the
same-year x up-bucket matched baseline, and a 20-session calendar-block bootstrap, then runs a look-ahead audit,
robustness grid, same-state random-date nulls, a bank-label permutation null and a mechanical comparison.

Run: data/derived/analysis/bottom_confirm 에서 `python3 -W ignore code/verify2_f6_SOXX_news_GSR_posthoc.py`
Out: results/verify2_f6_SOXX_news_GSR_posthoc.txt
"""
from __future__ import annotations

import io
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(__file__).resolve().parents[5]
RES = HERE / "results"
OUT = RES / "verify2_f6_SOXX_news_GSR_posthoc.txt"
TAG = "f6_SOXX"
RULE = "news_GSR_posthoc"

EVAL_START = pd.Timestamp("2024-01-02")
LAST = pd.Timestamp("2026-10-23")
PERIODS = [("2024~2026", "2024-01-02", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
           ("2025", "2025-01-02", "2025-12-31"), ("2024", "2024-01-02", "2024-12-31")]
EDGES = np.array([0.03, 0.08, 0.15])     # up buckets 0-3, 3-8, 8-15, 15+ (left-closed)
NDRAW = 5000
BOOT = 5000
SEED = 99_2026_1026

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
    c_new = p[("close", "SOXX")].dropna().astype(float)
    o_new = p[("open", "SOXX")].dropna().astype(float)
    h_new = p[("high", "SOXX")].dropna().astype(float)
    l_new = p[("low", "SOXX")].dropna().astype(float)
    u = pd.read_csv(ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv",
                    encoding="utf-8-sig", index_col=0, parse_dates=True)["SOXX"].dropna().astype(float)
    old = u[u.index < c_new.index[0]]
    C = pd.concat([old, c_new]).sort_index()
    assert not C.index.duplicated().any()
    ov = u.index.intersection(c_new.index)
    info = dict(overlap_n=len(ov), overlap_maxdiff=float(np.abs(u[ov] - c_new[ov]).max()))
    import exchange_calendars as xc  # used only as a calendar CHECK
    cal = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range(C.index[0], C.index[-1]))
    cal = cal.tz_localize(None) if cal.tz is not None else cal
    info["cal_missing"] = list(cal.difference(C.index).date)
    info["cal_extra"] = list(C.index.difference(cal).date)
    df = pd.DataFrame({"C": C, "O": o_new.reindex(C.index), "H": h_new.reindex(C.index), "L": l_new.reindex(C.index)})
    return df, info


def load_notes() -> pd.DataFrame:
    s = pd.read_csv(ROOT / "data/derived/analysis/news_followup/results/n4h14_signals.csv", encoding="utf-8-sig")
    s["entry"] = pd.to_datetime(s["entry"])
    return s


# ------------------------------------------------------------------ state + outcomes (loops, own code)
def state(C: np.ndarray, thr: float) -> dict:
    N = len(C)
    Ldd = np.full(N, np.nan)
    dd60 = np.full(N, np.nan)
    lowpos = np.full(N, -1)
    for t in range(59, N):
        w = C[t - 59:t + 1]
        mx = w.max()
        pk = t - 59 + int(np.flatnonzero(w == mx)[-1])          # latest position of the 60-session max
        seg = C[pk:t + 1]
        Ldd[t] = seg.min()
        lowpos[t] = pk + int(np.flatnonzero(seg == seg.min())[-1])
        dd60[t] = C[t] / mx - 1
    in_dd = np.zeros(N, bool)
    for t in range(N):
        w = dd60[max(0, t - 19):t + 1]
        w = w[~np.isnan(w)]
        in_dd[t] = len(w) > 0 and w.min() <= -thr
    up = C / Ldd - 1
    return dict(Ldd=Ldd, dd60=dd60, up=up, in_dd=in_dd, lowpos=lowpos)


def hold_outcome(C: np.ndarray, level: np.ndarray, start: int, length: int) -> np.ndarray:
    """Fixed censoring: 0 if any close in C[t+start .. t+start+length-1] (as far as data exist) < level[t];
    1 if the full window exists and nothing broke; NaN if the window is incomplete and nothing broke yet."""
    N = len(C)
    out = np.full(N, np.nan)
    for t in range(N):
        if np.isnan(level[t]):
            continue
        a = t + start
        b = t + start + length            # exclusive
        if a >= N:
            continue
        seen = C[a:min(b, N)]
        if (seen < level[t] - 1e-12).any():
            out[t] = 0.0
        elif b <= N:
            out[t] = 1.0
    return out


def outcomes(C: np.ndarray, O: np.ndarray, Ldd: np.ndarray, hold_len: int = 20) -> dict:
    N = len(C)
    Oe = np.full(N, np.nan)
    Oe[:-1] = O[1:]
    r20 = np.full(N, np.nan)
    r60 = np.full(N, np.nan)
    for t in range(N):
        if t + 20 < N:
            r20[t] = C[t + 20] / Oe[t] - 1
        if t + 60 < N:
            r60[t] = C[t + 60] / Oe[t] - 1
    win20 = np.where(np.isnan(r20), np.nan, (r20 > 0).astype(float))
    return dict(hold20=hold_outcome(C, Ldd, 1, hold_len), hold40=hold_outcome(C, Ldd, 1, 40),
                near5=hold_outcome(C, 0.95 * Oe, 1, 40), r20=r20, r60=r60, win20=win20)


def bucket(up: np.ndarray) -> np.ndarray:
    b = np.full(len(up), -1)
    ok = ~np.isnan(up)
    b[ok] = np.searchsorted(EDGES, up[ok], side="right")      # 0.03 -> bucket 1 (left-closed)
    return b


def matched_base(idx: pd.DatetimeIndex, st: dict, y: np.ndarray) -> np.ndarray:
    """Mean of y over in_dd days (>= EVAL_START, y and up valid) with the same calendar year and up bucket."""
    years = np.asarray(idx.year)
    b = bucket(st["up"])
    elig = st["in_dd"] & ~np.isnan(st["up"]) & ~np.isnan(y) & (idx >= EVAL_START)
    tab = {}
    for yr in np.unique(years):
        for k in range(4):
            m = elig & (years == yr) & (b == k)
            if m.any():
                tab[(yr, k)] = y[m].mean()
    return np.array([tab.get((years[t], b[t]), np.nan) if b[t] >= 0 else np.nan for t in range(len(y))])


def first_signals(sig: np.ndarray, st: dict, idx: pd.DatetimeIndex, cool: int = 20) -> np.ndarray:
    ok = sig & st["in_dd"] & ~np.isnan(st["Ldd"]) & (idx >= EVAL_START) & (idx <= LAST)
    out, prev = [], -10 ** 9
    for p in np.flatnonzero(ok):
        if p - prev <= cool:
            continue
        out.append(p)
        prev = p
    return np.array(out, int)


def block_boot(v: np.ndarray, pos: np.ndarray, rng, B=BOOT):
    lab = pos // 20
    u, inv = np.unique(lab, return_inverse=True)
    if len(u) < 2:
        return (np.nan, np.nan)
    sums = np.bincount(inv, weights=v)
    cnts = np.bincount(inv).astype(float)
    pick = rng.integers(0, len(u), size=(B, len(u)))
    m = sums[pick].sum(1) / cnts[pick].sum(1)
    return tuple(np.quantile(m, [0.025, 0.975]))


def period_stats(pos, st, oc, base, idx) -> dict:
    res = {}
    d = idx[pos] if len(pos) else pd.DatetimeIndex([])
    for name, a, b in PERIODS:
        m = (d >= a) & (d <= b)
        pp = pos[m]
        h = oc["hold20"][pp]
        bh = base[pp]
        ok = ~np.isnan(h) & ~np.isnan(bh)
        res[name] = dict(n=int(ok.sum()), nsig=len(pp),
                         hold20=float(h[ok].mean()) if ok.any() else np.nan,
                         base=float(bh[ok].mean()) if ok.any() else np.nan,
                         upmed=float(np.median(st["up"][pp])) if len(pp) else np.nan,
                         near5=float(np.nanmean(oc["near5"][pp])) if np.isfinite(oc["near5"][pp]).any() else np.nan,
                         win20=float(np.nanmean(oc["win20"][pp])) if np.isfinite(oc["win20"][pp]).any() else np.nan,
                         r20=float(np.nanmean(oc["r20"][pp])) if np.isfinite(oc["r20"][pp]).any() else np.nan)
    return res


def l1(st_: dict) -> bool:
    m, y26, y25 = st_["2024~2026"], st_["2026"], st_["2025"]
    c1 = m["n"] >= 8 and m["hold20"] >= 0.70
    c2 = y26["n"] >= 3 and y26["hold20"] >= 0.70
    c3 = y25["n"] < 3 or y25["hold20"] >= 0.60
    return bool(c1 and c2 and c3)


def fmt(label, s):
    def f(p):
        x = s[p]
        return f"{x['hold20']:.3f}({x['n']:>2d})" if x["n"] else "  -  ( 0)"
    return (f"  {label:40s} main {f('2024~2026')}  2026 {f('2026')}  H1 {f('2026H1')}  H2 {f('2026H2')}  "
            f"2025 {f('2025')}  2024 {f('2024')}  upmed {s['2024~2026']['upmed']:.3f}  base {s['2024~2026']['base']:.3f}  "
            f"L1={'T' if l1(s) else 'F'} diff={s['2024~2026']['hold20'] - s['2024~2026']['base']:+.3f}")


# ------------------------------------------------------------------ nulls
def null_draws(slots_pool: list[np.ndarray], hold: np.ndarray, cool: int, rng, ndraw=NDRAW, max_try=200):
    """One random day per slot from that slot's pool; all picks pairwise > cool sessions apart."""
    out, tries = [], 0
    k = len(slots_pool)
    while len(out) < ndraw and tries < ndraw * max_try:
        tries += 1
        order = rng.permutation(k)
        chosen = []
        good = True
        for i in order:
            pool = slots_pool[i]
            if chosen:
                ch = np.array(chosen)
                pool = pool[np.abs(pool[:, None] - ch[None, :]).min(1) > cool]
            if len(pool) == 0:
                good = False
                break
            chosen.append(int(pool[rng.integers(0, len(pool))]))
        if good:
            out.append(hold[np.array(chosen)].mean())
    return np.array(out), tries


VERDICT = """
VERDICT: fragile  (numbers reproduce and survive the timing grid, but the cell is post-hoc, carries no information
beyond news_R, and L1 flips under alternative coding, de-duplication and a 25-session window)

What holds up
  - Exact reproduction without bc_lib: 11 first signals, every field (up, dd60, hold20/40, near5, win20, r20, r60,
    matched base) and every period equal to results/f6_SOXX_signals.csv / _summary.csv (news_R 13/13 too).
  - No look-ahead: truncation test 0 differences; all 53 GS R notes enter on or after the first session after
    publication; the signal day is the entry session and the trade is the NEXT open (one session conservative);
    the large SOXX days (2025-04, 2026-06-05/06-11, 2026-07-30) match ^SOX/SMH, no OHLC errors.
  - hold20 11/11 (2024~2026) vs same-year x up-bucket base 0.739; 2026 4/4 (base 0.636), 2025 3/3, 2024 4/4.
    Core grid (delay +1/+2, cooldown 10/30, in_dd 8%/12%) all keep L1 (lowest 0.750 n16 at in_dd 8%).
    Same-state nulls with 20-session spacing: year x bucket p=0.009, year + up+-2pp p=0.0024,
    up+-2pp any year p=0.0018, 2026 only p=0.034; prod(matched base) = 0.012.
  - Plain 'up >= 6.0%' (the candidate's median) scores only 0.733 (n15) and fails L1, so this is not pure
    rebound size; within the 0-8% up buckets the 6 signals all held vs pool 0.32/0.64.

Why it cannot be confirmed
  1. Post-hoc and circular. GS R was picked in news_followup on the same 2024~2026 SOXX path (10-day excess
     return, which moves with hold20). 109 cells here plus the bank x family split upstream; the best nominal
     p (0.0018) is above a 109-cell Bonferroni bar (0.00046).
  2. Overlap with news_R (family R, any bank). GS R notes are a strict subset (53 of 82 R notes; 52 of 77 R entry
     sessions). 9 of the 11 GS R first signals (82%) ARE news_R first signals. The other 2 (2024-12-17,
     2026-08-03) are GS notes that news_R skips only because a BofA R note opened the cooldown 9 and 12 sessions
     earlier. news_R's extra 4 first signals are all BofA (3 hold, 1 fail = 2026-07-16, news_R's ONLY failure).
     So GS R = news_R minus BofA, and 12/13 -> 11/11 is exactly the removal of that one BofA failure.
     Bank-label permutation inside R (random 53 of 82): 13.8% of draws reach 11/11-level hold20 and 71.6% pass
     L1. The 'GS' label adds no demonstrable information; GS R's evidence is news_R's evidence, not extra.
     (news_R under the same nulls: p=0.0034 year x bucket, 0.0008 year + up+-2pp; GS R is not better.)
  3. Coding / de-dup sensitivity. fam_altB recodes 5 of the 11 first-signal notes (to D: 2025-04-11, 2026-04-14,
     2026-08-03, each of which also reports a record-selling fact; to T: 2024-09-24, 2026-09-21)
     -> 0.700 (n10), 2026 2/3, L1 fails. Dropping possible-duplicate
     reports -> 0.900 (n10), 2026 2/3, L1 fails: 2026-06-09 is G0212, a possible re-report of G0210 that was
     published 06-01, BEFORE the 06-05 -10.4% day; without it the next GS R day (2026-06-16) breaks Ldd.
  4. Window edge and thin margins. The 2026-06-09 low (539.77, 06-05) is broken at t+25 (2026-07-16), so hold
     windows of 25/30/40 sessions give 2026 2/3 and fail L1. 3 of 11 holds clear Ldd by <0.4%
     (2024-12-17 +0.36%, 2026-04-14 +0.15%, 2026-06-09 +0.32%).
  5. The cooldown hides 3 failing GS R in_dd days (2024-10-21, 2025-12-16, 2026-06-16); raw GS R in_dd days hold
     19/22 = 86% vs matched 74%. 5 of the 11 signals sit in up buckets where any in_dd day holds ~99-100%;
     the evidence rests on 6 signals, two of them at/near a 60-day high (2026-04-14 dd60 0, 2025-06-03 -0.5%)
     where 'hold' means 'the breakout close was not undercut', not 'the decline low is in'.
  6. Little practical edge now: near5 5/11 (45%); 2026H2 near5 0/3, win20 1/3, mean r20 -2.5%.

Bottom line: a true statement about 11 past dates, not a confirmed 70%+ bottom rule. Treat it as a duplicate of
news_R (itself fragile under alternative coding); only a pre-registered out-of-sample test on notes after
2026-10-23 (GS vs BofA split fixed in advance) could promote it.
"""


# ------------------------------------------------------------------ main
def main():
    pd.set_option("display.width", 250)
    say("=" * 118)
    say(f"Independent verification round 2 (fixed censoring): family f6, kind SOXX, rule {RULE}  (no bc_lib import)")
    say("=" * 118)
    px, info = load_soxx()
    idx = px.index
    C, O = px.C.to_numpy(float), px.O.to_numpy(float)
    N = len(C)
    say(f"Data: SOXX close {idx[0].date()}..{idx[-1].date()} N={N}; open from {px.O.first_valid_index().date()}; "
        f"overlap max|us_prices_close - prices.csv| = {info['overlap_maxdiff']:.2e} (n={info['overlap_n']}); "
        f"XNYS missing={len(info['cal_missing'])} extra={len(info['cal_extra'])}")
    notes = load_notes()
    pos_of = {d: i for i, d in enumerate(idx)}
    assert notes.entry.isin(idx).all(), "note entry outside sessions"
    notes["p"] = notes.entry.map(pos_of)

    def note_sig(mask) -> np.ndarray:
        s = np.zeros(N, bool)
        s[notes.loc[mask, "p"].to_numpy(int)] = True
        return s

    isR, isGS = notes.family.eq("R"), notes.bank.eq("GS")
    sig_gsr = note_sig(isR & isGS)
    sig_r = note_sig(isR)

    st = state(C, 0.10)
    oc = outcomes(C, O, st["Ldd"])
    base = {k: matched_base(idx, st, oc[k]) for k in ("hold20", "hold40", "near5", "win20", "r20", "r60")}

    # -------------------------------------------------------------- [1] reproduction
    say("\n[1] Independent re-implementation vs family files (results/f6_SOXX_signals.csv, _summary.csv)")
    fam_sig = pd.read_csv(RES / f"{TAG}_signals.csv", encoding="utf-8-sig", parse_dates=["date"])
    fam_sum = pd.read_csv(RES / f"{TAG}_summary.csv", encoding="utf-8-sig")
    all_match = True
    for rule, sig in ((RULE, sig_gsr), ("news_R", sig_r)):
        fp = first_signals(sig, st, idx)
        fs = fam_sig[fam_sig.rule == rule].reset_index(drop=True)
        same_dates = list(idx[fp]) == list(fs.date)
        say(f"\n  rule {rule}: mine {len(fp)} first signals, family {len(fs)}; dates identical: {same_dates}")
        all_match &= same_dates
        rows = []
        mism = []
        for i, p in enumerate(fp):
            nb = notes[notes.p == p]
            r = dict(date=idx[p].date(), pos=p, banks="/".join(f"{f}:{b}" for f, b in zip(nb.family, nb.bank)),
                     up=st["up"][p], dd60=st["dd60"][p], Ldd=st["Ldd"][p], low=idx[st["lowpos"][p]].date(),
                     hold20=oc["hold20"][p], base20=base["hold20"][p], hold40=oc["hold40"][p],
                     near5=oc["near5"][p], win20=oc["win20"][p], r20=oc["r20"][p], r60=oc["r60"][p])
            fwd = C[p + 1:min(p + 21, N)]
            r["margin20"] = fwd.min() / st["Ldd"][p] - 1 if len(fwd) else np.nan
            rows.append(r)
            if same_dates:
                f = fs.iloc[i]
                for a, b in (("up", "up"), ("dd60", "dd60"), ("hold20", "hold20"), ("base20", "base_hold20"),
                             ("hold40", "hold40"), ("near5", "near5"), ("win20", "win20"), ("r20", "r20"), ("r60", "r60")):
                    x, z = r[a], f[b]
                    if not ((np.isnan(x) and pd.isna(z)) or (not np.isnan(x) and not pd.isna(z) and abs(x - z) < 1e-9)):
                        mism.append(f"{r['date']} {a}: mine {x} family {z}")
        t = pd.DataFrame(rows)
        say(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        say(f"  per-signal field mismatches: {'none' if not mism else mism}")
        all_match &= not mism
        mine = period_stats(fp, st, oc, base["hold20"], idx)
        fsu = fam_sum[fam_sum.rule == rule].set_index("period")
        say(f"  {'period':10s} {'mine hold20 (n)':>17s} {'family hold20 (n)':>19s} {'mine base':>10s} {'family base':>12s}")
        for name, _, _ in PERIODS:
            m = mine[name]
            fh = fsu.loc[name, "hold20"] if name in fsu.index else np.nan
            fn = int(fsu.loc[name, "hold20_n"]) if name in fsu.index and pd.notna(fsu.loc[name, "hold20_n"]) else 0
            fb = fsu.loc[name, "hold20_base"] if name in fsu.index else np.nan
            ok = (m["n"] == fn) and ((m["n"] == 0) or (abs(m["hold20"] - fh) < 1e-9 and abs(m["base"] - fb) < 1e-9))
            all_match &= ok
            say(f"  {name:10s} {m['hold20']:8.4f} ({m['n']:>2d})      {fh:8.4f} ({fn:>2d})        {m['base']:.4f}      {fb:.4f}   "
                f"{'OK' if ok else 'MISMATCH'}")
        if rule == RULE:
            fp_gsr, stats_gsr = fp, mine
            # block bootstrap of hold20 - matched base, main period (reproduces family hold20_dlo up to RNG)
            v = oc["hold20"][fp] - base["hold20"][fp]
            lo, hi = block_boot(v, fp, np.random.default_rng(SEED))
            say(f"  my block bootstrap (20-session calendar blocks, B={BOOT}) main hold20 - base: "
                f"[{lo:+.3f}, {hi:+.3f}]  (family dlo {fsu.loc['2024~2026', 'hold20_dlo']:+.3f}, "
                f"dhi {fsu.loc['2024~2026', 'hold20_dhi']:+.3f})")
            say(f"  NOTE: hold20 = 11/11 has zero variance, so the bootstrap interval of the diff only resamples the "
                f"baseline; it says nothing about the signal's own sampling error.")
        else:
            fp_r = fp
    say(f"\n  => independent match (both rules, all fields, all periods): {all_match}")

    # -------------------------------------------------------------- overlap with news_R
    say("\n[1b] Overlap with news_R (family R, any bank)")
    gsr_days, r_days = set(np.flatnonzero(sig_gsr)), set(np.flatnonzero(sig_r))
    ev = (idx >= EVAL_START) & (idx <= LAST)
    say(f"  raw note-entry sessions: news_R {len(r_days)}, GS R {len(gsr_days)}, GS R subset of news_R: {gsr_days <= r_days}")
    say(f"  in the main period: news_R {int((sig_r & ev).sum())}, GS R {int((sig_gsr & ev).sum())}; "
        f"on in_dd days: news_R {int((sig_r & ev & st['in_dd']).sum())}, GS R {int((sig_gsr & ev & st['in_dd']).sum())}")
    R_notes = notes[isR]
    say(f"  R notes total {len(R_notes)} = " + ", ".join(f"{b} {n}" for b, n in R_notes.bank.value_counts().items()))
    a, b = set(fp_gsr), set(fp_r)
    both, only_g, only_r = sorted(a & b), sorted(a - b), sorted(b - a)
    say(f"  first signals: GS R {len(a)}, news_R {len(b)}, shared {len(both)} ({len(both) / len(a):.0%} of GS R, "
        f"{len(both) / len(b):.0%} of news_R)")
    for lab, ps in (("shared", both), ("GS R only", only_g), ("news_R only", only_r)):
        hs = oc["hold20"][ps] if ps else np.array([])
        say(f"    {lab:12s} n={len(ps):2d} hold20={np.nanmean(hs) if len(hs) else np.nan:.3f}  " +
            "  ".join(f"{idx[p].date()}[{'/'.join(notes[notes.p == p].bank)}] h={oc['hold20'][p]:.0f} up={st['up'][p]:.3f}"
                      for p in ps))
    for p in only_g:
        prev = [q for q in fp_r if 0 < p - q <= 20]
        say(f"    GS-only {idx[p].date()} is suppressed in news_R by the cooldown after news_R first signal "
            f"{', '.join(str(idx[q].date()) + '[' + '/'.join(notes[notes.p == q].bank) + ']' for q in prev) or '(none)'}")
    # non-GS R alone
    sig_rx = note_sig(isR & ~isGS)
    fp_rx = first_signals(sig_rx, st, idx)
    s_rx = period_stats(fp_rx, st, oc, base["hold20"], idx)
    say(fmt("non-GS R (BofA/MS/JPM) as its own rule", s_rx))
    say("  " + "  ".join(f"{idx[p].date()}[{'/'.join(notes[(notes.p == p) & isR].bank)}]h={oc['hold20'][p]:.0f}" for p in fp_rx))

    # -------------------------------------------------------------- [2] look-ahead audit
    say("\n[2] Look-ahead / data audit")
    # (a) truncation test
    diffs = 0
    for p in fp_gsr:
        st_t = state(C[:p + 1], 0.10)
        for k in ("Ldd", "dd60", "up"):
            if abs(st_t[k][p] - st[k][p]) > 1e-12:
                diffs += 1
        if bool(st_t["in_dd"][p]) != bool(st["in_dd"][p]):
            diffs += 1
    say(f"  (a) truncation test (Ldd, dd60, up, in_dd recomputed with data cut at t): {diffs} differences")
    # (b) note timing
    g = notes[isR & isGS].copy()
    pub = g.published_at.astype(str)
    g["pub_date"] = pd.to_datetime(pub.str[:10])
    viol, rows = 0, []
    for _, r in g.iterrows():
        e = r.entry
        if r.time_precision == "exact":
            ts = pd.Timestamp(r.published_at)
            et = ts.tz_convert("America/New_York")
            same_day_ok = (et.hour, et.minute) < (9, 30)
            exp = idx[idx >= pd.Timestamp(et.date())][0] if same_day_ok else idx[idx > pd.Timestamp(et.date())][0]
        else:
            exp = idx[idx > r.pub_date][0]
        nd = pd.Timestamp(r.note_date)
        if e < exp or nd > r.pub_date:
            viol += 1
            rows.append(f"{r.group_id} pub {r.published_at} note {r.note_date} entry {e.date()} expected>= {exp.date()}")
    say(f"  (b) GS R notes: {len(g)}; entry earlier than first session after publication, or note_date after "
        f"published date: {viol} {rows if rows else ''}")
    say(f"      precision mix: {g.time_precision.value_counts().to_dict()}; verification mix: "
        f"{g.verification.value_counts().to_dict()}")
    lag = [(pd.Timestamp(r.entry) - pd.Timestamp(r.note_date)).days for _, r in g.iterrows()]
    say(f"      calendar days from note_date to signal session: median {np.median(lag):.0f}, max {max(lag)}; "
        f"the rule's signal day IS the entry session, trade at the NEXT open (one session conservative)")
    # (c) price data sanity
    ret = px.C.pct_change()
    big = ret[ret.abs() > 0.08]
    say(f"  (c) SOXX daily |close return| > 8%: " + ", ".join(f"{d.date()} {v:+.3f}" for d, v in big.items()))
    ohlc = px.dropna()
    bad = ohlc[(ohlc.H < ohlc[["O", "C"]].max(1) - 1e-6) | (ohlc.L > ohlc[["O", "C"]].min(1) + 1e-6)]
    say(f"      OHLC rows where H < max(O,C) or L > min(O,C): {len(bad)} {list(bad.index.date)[:10]}")
    for p in fp_gsr:
        w = slice(max(0, p - 60), min(N, p + 22))
        r = ret.iloc[w]
        if (r.abs() > 0.08).any():
            say(f"      signal {idx[p].date()}: window contains |ret|>8% on " +
                ", ".join(f"{d.date()} {v:+.3f}" for d, v in r[r.abs() > 0.08].items()))
    say("      (back-adjusted yfinance closes: an ex-dividend inside a window lowers the pre-ex Ldd by ~0.1-0.2%;"
        " hold margins below are reported so this can be judged)")
    mg = [(idx[p].date(), C[p + 1:p + 21].min() / st["Ldd"][p] - 1) for p in fp_gsr if p + 1 < N]
    say("  (d) hold margins min(C[t+1..t+20])/Ldd - 1: " + ", ".join(f"{d} {m:+.3f}" for d, m in mg))
    say(f"      smallest margin {min(m for _, m in mg):+.4f}")

    # -------------------------------------------------------------- [3] robustness
    say("\n[3] Robustness (hold20 (n) per period; main = 2024~2026; base = same-year x up-bucket matched mean)")
    cache = {}

    def frame(thr, hold_len=20):
        key = (thr, hold_len)
        if key not in cache:
            s_ = state(C, thr)
            o_ = outcomes(C, O, s_["Ldd"], hold_len)
            cache[key] = (s_, o_, matched_base(idx, s_, o_["hold20"]))
        return cache[key]

    def run(label, sig, thr=0.10, cool=20, delay=0, hold_len=20):
        s_, o_, b_ = frame(thr, hold_len)
        sg = np.zeros(N, bool)
        if delay:
            sg[delay:] = sig[:-delay]
        else:
            sg = sig.copy()
        fp = first_signals(sg, s_, idx, cool)
        res = period_stats(fp, s_, o_, b_, idx)
        say(fmt(label, res))
        res["_fp"] = fp
        res["_h"] = o_["hold20"][fp]
        return res

    def listfail(label, res):
        fp, hh = res["_fp"], res["_h"]
        bad = [f"{idx[p].date()}(up {st['up'][p]:+.3f})" for p, x in zip(fp, hh) if x == 0]
        say(f"      {label}: failures {', '.join(bad) or 'none'}; signals " +
            " ".join(str(idx[p].date()) for p in fp))

    var = {}
    var["registered (GS R, thr10 cool20 d0)"] = run("registered (GS R, thr10 cool20 d0)", sig_gsr)
    var["delay +1 session"] = run("delay +1 session", sig_gsr, delay=1)
    var["delay +2 sessions"] = run("delay +2 sessions", sig_gsr, delay=2)
    var["cooldown 10"] = run("cooldown 10", sig_gsr, cool=10)
    var["cooldown 30"] = run("cooldown 30", sig_gsr, cool=30)
    var["in_dd thr 8%"] = run("in_dd thr 8%", sig_gsr, thr=0.08)
    var["in_dd thr 12%"] = run("in_dd thr 12%", sig_gsr, thr=0.12)
    say("  -- outcome-window neighbours (not registered; shows how close to the edge the 11/11 is)")
    for hl in (10, 15, 25, 30, 40):
        run(f"hold window {hl} sessions", sig_gsr, hold_len=hl)
    say("  -- rule neighbours (label / coding / quality variants of 'GS R')")
    neigh = {
        "news_R (any bank)": isR,
        "non-GS R only": isR & ~isGS,
        "GS R, fam_altA == R": notes.fam_altA.eq("R") & isGS,
        "GS R, fam_altB == R": notes.fam_altB.eq("R") & isGS,
        "GS R, no possible_dup flag": isR & isGS & notes.uncertain_dup.isna(),
        "GS R, article read (not headline-only)": isR & isGS & ~notes.verification.eq("headline_summary_only"),
        "GS R, headline-only": isR & isGS & notes.verification.eq("headline_summary_only"),
        "GS R, flow_direction net_buy": isR & isGS & notes.flow_direction.eq("net_buy"),
        "GS any family": isGS,
        "GS R or GS S": notes.family.isin(["R", "S"]) & isGS,
        "BofA R": isR & notes.bank.eq("BofA"),
        "all notes (any family, any bank)": notes.family.notna(),
    }
    for lab, m in neigh.items():
        var[lab] = run(lab, note_sig(m))
    var["GS R, signal-day dd60 <= -5% (drop near-high days)"] = run(
        "GS R, signal-day dd60 <= -5% (drop near-high days)", sig_gsr & (st["dd60"] <= -0.05))
    core = ["registered (GS R, thr10 cool20 d0)", "delay +1 session", "delay +2 sessions", "cooldown 10",
            "cooldown 30", "in_dd thr 8%", "in_dd thr 12%"]
    say("  L1 per core variant: " + ", ".join(f"{k}={l1(var[k])}" for k in core))
    say("  -- where the failures are")
    for k in ("delay +1 session", "cooldown 10", "in_dd thr 8%", "GS R, fam_altB == R", "GS R, no possible_dup flag",
              "news_R (any bank)", "GS any family"):
        listfail(k, var[k])
    raw = sig_gsr & st["in_dd"] & (idx >= EVAL_START) & ~np.isnan(oc["hold20"])
    say("      raw GS R in_dd days (no cooldown) that FAILED hold20: " +
        ", ".join(f"{idx[p].date()}(up {st['up'][p]:+.3f}, first-signal cover {idx[[q for q in fp_gsr if q <= p][-1]].date() if any(q <= p for q in fp_gsr) else '-'})"
                  for p in np.flatnonzero(raw) if oc["hold20"][p] == 0))
    p0 = idx.get_loc(pd.Timestamp("2026-06-09"))
    brk = np.flatnonzero(C[p0 + 1:] < st["Ldd"][p0])
    say(f"      2026-06-09 signal (Ldd {st['Ldd'][p0]:.2f} set {idx[st['lowpos'][p0]].date()}): first close below Ldd "
        f"{idx[p0 + 1 + brk[0]].date() if len(brk) else '-'} = session t+{brk[0] + 1 if len(brk) else '-'}"
        f" (close {C[p0 + 1 + brk[0]] if len(brk) else np.nan:.2f})")
    for d in ("2026-04-14", "2024-12-17"):
        q = idx.get_loc(pd.Timestamp(d))
        brk = np.flatnonzero(C[q + 1:] < st["Ldd"][q])
        say(f"      {d} signal (Ldd {st['Ldd'][q]:.2f}): first close below Ldd "
            f"{idx[q + 1 + brk[0]].date() if len(brk) else '-'} = t+{brk[0] + 1 if len(brk) else '-'}")
    # cross-check of large SOXX moves against other series (data-error check)
    u = pd.read_csv(ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv",
                    encoding="utf-8-sig", index_col=0, parse_dates=True)
    rr = u[["^SOX", "SMH", "SOXL"]].pct_change()
    for d in ("2025-04-03", "2025-04-09", "2025-04-10", "2026-06-05", "2026-06-11", "2026-07-30"):
        t = pd.Timestamp(d)
        say(f"      data check {d}: SOXX {px.C.pct_change().get(t):+.3f}  ^SOX {rr['^SOX'].get(t, np.nan):+.3f}  "
            f"SMH {rr['SMH'].get(t, np.nan):+.3f}  SOXL {rr['SOXL'].get(t, np.nan):+.3f}")

    # -------------------------------------------------------------- [4] nulls
    say("\n[4] Same-state nulls (main period 2024-01-02..2026-10-23; pool = in_dd days with Ldd and hold20 defined)")
    h = oc["hold20"]
    years = np.asarray(idx.year)
    b_all = bucket(st["up"])
    pool_mask = st["in_dd"] & ~np.isnan(st["Ldd"]) & ~np.isnan(h) & (idx >= EVAL_START) & (idx <= LAST)
    pool = np.flatnonzero(pool_mask)
    obs = float(np.nanmean(h[fp_gsr]))
    mix = pd.crosstab(years[fp_gsr], b_all[fp_gsr])
    say(f"  observed hold20 = {obs:.4f} (n={len(fp_gsr)}); year x bucket mix of first signals:")
    say("  " + mix.to_string().replace("\n", "\n  "))
    say(f"  pool days {len(pool)}; pool hold20 by bucket: " +
        ", ".join(f"b{k} {h[pool[b_all[pool] == k]].mean():.3f}(n{int((b_all[pool] == k).sum())})" for k in range(4)))
    # analytic: probability all 11 hold if each is a Bernoulli(matched base)
    pb = base["hold20"][fp_gsr]
    say(f"  analytic: prod(matched base_i) = P(all {len(fp_gsr)} hold | independent matched Bernoulli) = {np.prod(pb):.4f}")
    rng = np.random.default_rng(SEED)
    slots_a = [pool[(years[pool] == years[p]) & (b_all[pool] == b_all[p])] for p in fp_gsr]
    da, ta = null_draws(slots_a, h, 20, rng)
    pa = float((da >= obs - 1e-12).mean())
    say(f"  (a) year x bucket, spacing > 20 sessions: draws {len(da)} (attempts {ta}); null mean {da.mean():.4f} "
        f"sd {da.std():.4f} q95 {np.quantile(da, 0.95):.3f}; p = P(null >= {obs:.3f}) = {pa:.4f}")
    obs_r0 = float(np.nanmean(h[fp_r]))
    dra, _ = null_draws([pool[(years[pool] == years[p]) & (b_all[pool] == b_all[p])] for p in fp_r], h, 20,
                        np.random.default_rng(SEED + 6))
    say(f"      context, news_R (any bank) same null: observed {obs_r0:.4f} (n={len(fp_r)}), null mean {dra.mean():.4f}, "
        f"p = {(dra >= obs_r0 - 1e-12).mean():.4f}")
    slots_b = [pool[(years[pool] == years[p]) & (np.abs(st['up'][pool] - st['up'][p]) <= 0.02)] for p in fp_gsr]
    say("      finer pools (same year, |up - up_sig| <= 2pp) sizes: " + ", ".join(str(len(s)) for s in slots_b))
    db, tb = null_draws(slots_b, h, 20, np.random.default_rng(SEED + 1))
    pb_ = float((db >= obs - 1e-12).mean())
    say(f"  (b) same year, up within +-2pp, spacing > 20: draws {len(db)} (attempts {tb}); null mean {db.mean():.4f} "
        f"sd {db.std():.4f}; p = {pb_:.4f}")
    obs_r = float(np.nanmean(h[fp_r]))
    slots_r = [pool[(years[pool] == years[p]) & (np.abs(st['up'][pool] - st['up'][p]) <= 0.02)] for p in fp_r]
    dr, _ = null_draws(slots_r, h, 20, np.random.default_rng(SEED + 5))
    say(f"      context, news_R (any bank) same null: observed {obs_r:.4f} (n={len(fp_r)}), null mean {dr.mean():.4f}, "
        f"p = {(dr >= obs_r - 1e-12).mean():.4f}")
    slots_c = [pool[np.abs(st['up'][pool] - st['up'][p]) <= 0.02] for p in fp_gsr]
    dc, tc = null_draws(slots_c, h, 20, np.random.default_rng(SEED + 2))
    pc = float((dc >= obs - 1e-12).mean())
    say(f"  (c) any year in main period, up within +-2pp, spacing > 20: draws {len(dc)}; null mean {dc.mean():.4f}; p = {pc:.4f}")
    # 2026-only
    fp26 = fp_gsr[idx[fp_gsr] >= "2026-01-02"]
    s26 = [pool[(years[pool] == 2026) & (np.abs(st['up'][pool] - st['up'][p]) <= 0.02)] for p in fp26]
    d26, _ = null_draws(s26, h, 20, np.random.default_rng(SEED + 3))
    o26 = float(np.nanmean(h[fp26]))
    say(f"  (d) 2026 only (n={len(fp26)}), up +-2pp: null mean {d26.mean():.4f}; p = P(null >= {o26:.3f}) = "
        f"{(d26 >= o26 - 1e-12).mean():.4f}")
    # (e) bank-label permutation within R notes: is 'GS' special among R?
    rng = np.random.default_rng(SEED + 4)
    Rn = notes[isR].reset_index(drop=True)
    nG = int(Rn.bank.eq("GS").sum())
    ph, pn, pl1 = [], [], 0
    for _ in range(NDRAW):
        pick = rng.permutation(len(Rn))[:nG]
        s = np.zeros(N, bool)
        s[Rn.p.to_numpy(int)[pick]] = True
        fp = first_signals(s, st, idx)
        r = period_stats(fp, st, oc, base["hold20"], idx)
        ph.append(r["2024~2026"]["hold20"])
        pn.append(r["2024~2026"]["n"])
        pl1 += l1(r)
    ph = np.array(ph)
    say(f"  (e) bank-label permutation inside R notes ({nG} of {len(Rn)} R notes labelled 'GS' at random, {NDRAW} draws):")
    say(f"      main hold20 mean {np.nanmean(ph):.4f}, share with hold20 >= {obs:.3f}: {(ph >= obs - 1e-12).mean():.4f}, "
        f"share passing L1: {pl1 / NDRAW:.4f}, n median {np.median(pn):.0f}")
    say("      => this measures how much of GS R's 11/11 is just 'R' plus a random sub-sample")

    # -------------------------------------------------------------- [5] mechanical
    say("\n[5] Mechanical comparison: plain rule up >= candidate median up (in_dd, cooldown 20)")
    med = float(np.median(st["up"][fp_gsr]))
    for thr_up in (med, 0.05, 0.03, 0.08):
        run(f"up >= {thr_up:.4f}", st["up"] >= thr_up)
    allm = pool_mask
    hi = allm & (st["up"] >= med)
    say(f"  all in_dd days 2024~2026 with up >= {med:.4f} (no cooldown): hold20 {h[hi].mean():.4f} ({int(hi.sum())} days); "
        f"up < it: {h[allm & (st['up'] < med)].mean():.4f}")
    say("  within-bucket comparison (candidate first signals vs all in_dd pool days):")
    for k, nm in enumerate(("0-3%", "3-8%", "8-15%", "15%+")):
        c = fp_gsr[b_all[fp_gsr] == k]
        pdays = pool[b_all[pool] == k]
        say(f"    bucket {nm:6s} cand {np.nanmean(h[c]) if len(c) else np.nan:.3f} (n={len(c)})   pool {h[pdays].mean():.3f} (days={len(pdays)})")
    # GS R raw days on in_dd (no cooldown)
    raw = sig_gsr & pool_mask
    say(f"  raw GS R in_dd days (no cooldown): hold20 {h[raw].mean():.4f} ({int(raw.sum())} days), matched base avg "
        f"{np.nanmean(base['hold20'][raw]):.4f}")
    say("  other outcomes, main period (GS R first signals vs matched base):")
    for k in ("hold40", "near5", "win20", "r20", "r60"):
        v = oc[k][fp_gsr]
        bv = base[k][fp_gsr]
        ok = ~np.isnan(v) & ~np.isnan(bv)
        say(f"    {k:6s} {v[ok].mean():+.4f} vs base {bv[ok].mean():+.4f} (n={int(ok.sum())})")

    # -------------------------------------------------------------- [6] verdict inputs
    say("\n[6] Verdict inputs")
    reg = var["registered (GS R, thr10 cool20 d0)"]
    say(f"  independent match={all_match}; look-ahead found=False (truncation diffs {diffs}, note-timing violations {viol})")
    say(f"  main hold20 {reg['2024~2026']['hold20']:.4f} (n={reg['2024~2026']['n']}) vs matched base {reg['2024~2026']['base']:.4f}; "
        f"2026 {reg['2026']['hold20']:.3f} (n={reg['2026']['n']}); up median {reg['2024~2026']['upmed']:.4f}")
    say(f"  nulls: year x bucket p={pa:.4f}; year + up+-2pp p={pb_:.4f}; up+-2pp any year p={pc:.4f}; "
        f"bank-label permutation p={(ph >= obs - 1e-12).mean():.4f}; analytic prod(base)={np.prod(pb):.4f}")
    say("  L1 per core variant: " + ", ".join(f"{k}={l1(var[k])}" for k in core))
    say(globals().get("VERDICT", ""))
    OUT.write_text(buf.getvalue(), encoding="utf-8")
    return dict(all_match=all_match, pa=pa, pb=pb_, pc=pc, perm=float((ph >= obs - 1e-12).mean()))


if __name__ == "__main__":
    main()
