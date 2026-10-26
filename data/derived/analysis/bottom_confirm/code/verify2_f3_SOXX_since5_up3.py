"""Independent adversarial verification (round 2, fixed censoring) of f3 / since5_up3 / SOXX (BRIEF.md 2026-10-26).

Does NOT import bc_lib (or f3_time, which imports it). Re-implements from the raw CSVs: the SOXX series, the decline
low Ldd and its LAST position since the latest 60-session closing high, dd60, in_dd (10%), the rule
(sessions since that low >= 5 and up = C/Ldd - 1 >= 3%), the 20-session cooldown, the outcomes hold20 / hold40 /
near5 / win20 / r20 with the FIXED right-censoring rule (0 when a close below the level is already observed inside an
incomplete forward window, NaN only when nothing has broken yet), the same-state (year x up-bucket) matched
baseline and the 20-session block bootstrap.

Sections: [1] independent match, [2] look-ahead / data audit, [3] robustness (neighbours, delay 1/2, cooldown 10/30,
in_dd 8%/12%), [4] same-state random-date nulls (year x up-bucket, year x up +-2pp, and a stratified on/off test of
whether 'time since the low' adds anything at equal rebound size), [5] mechanical comparison with plain
up >= candidate median up, [d] per-signal detail, [e] stocks replication (independent recompute), [6] verdict inputs.

Run: cd data/derived/analysis/bottom_confirm && python3 -W ignore code/verify2_f3_SOXX_since5_up3.py
Out: results/verify2_f3_SOXX_since5_up3.txt
Everything here is post hoc (computed after the registered results were seen).
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
OUT = RES / "verify2_f3_SOXX_since5_up3.txt"

EVAL_START = pd.Timestamp("2024-01-02")
LAST = pd.Timestamp("2026-10-23")
PERIODS = [("2024~2026", "2024-01-02", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
           ("2025", "2025-01-02", "2025-12-31"), ("2024", "2024-01-02", "2024-12-31")]
EDGES = [0.03, 0.08, 0.15]          # up buckets 0-3, 3-8, 8-15, 15%+
BNAMES = ["0-3%", "3-8%", "8-15%", "15%+"]
KEYS = ("hold20", "hold40", "near5", "win20", "r20")
RULE_N, RULE_UP = 5, 0.03
NDRAW = 5000
NBOOT = 5000
SEED = 3_2026_1026
N_CELLS = 109

buf = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    buf.write(s + "\n")


# ------------------------------------------------------------------ raw data (own loaders)
def xnys(a, b) -> pd.DatetimeIndex:
    import exchange_calendars as xc
    s = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range(a, b))
    return s.tz_localize(None) if s.tz is not None else s


def load_soxx():
    p = pd.read_csv(ROOT / "research/news_turning_points_20261025/prices.csv", header=[0, 1], index_col=0)
    p.index = pd.to_datetime(p.index, errors="coerce")
    p = p[p.index.notna()]
    c_new = p[("close", "SOXX")].dropna()
    o_new = p[("open", "SOXX")].dropna()
    u = pd.read_csv(ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv",
                    encoding="utf-8-sig", index_col=0, parse_dates=True)
    us = u["SOXX"].dropna()
    C = pd.concat([us[us.index < c_new.index[0]], c_new]).sort_index()
    assert not C.index.duplicated().any()
    C = C[C.index <= LAST]
    O = o_new.reindex(C.index)
    ov = us.index.intersection(c_new.index)
    cal = xnys(C.index[0], C.index[-1])
    chk = dict(overlap_n=len(ov), overlap_maxdiff=float(np.abs(us[ov] - c_new[ov]).max()),
               cal_missing=len(cal.difference(C.index)), cal_extra=len(C.index.difference(cal)))
    other = u[["^SOX", "SMH"]].reindex(C.index)
    return pd.DataFrame({"C": C, "O": O}), chk, other


# ------------------------------------------------------------------ state and outcomes (fixed censoring)
def state(C: np.ndarray, thr: float, look: int = 60) -> dict:
    """Ldd = min close since the latest `look`-session closing high (window ends at t); Lpos = LAST position of that min."""
    N = len(C)
    Ldd, Lpos, dd = np.full(N, np.nan), np.full(N, -1), np.full(N, np.nan)
    for t in range(look - 1, N):
        w = C[t - look + 1:t + 1]
        if np.isnan(w).any():
            continue
        mx = w.max()
        pk = t - look + 1 + int(np.flatnonzero(w == mx)[-1])
        seg = C[pk:t + 1]
        mn = seg.min()
        Ldd[t] = mn
        Lpos[t] = pk + int(np.flatnonzero(seg == mn)[-1])
        dd[t] = C[t] / mx - 1
    in_dd = np.zeros(N, bool)
    for t in range(N):
        w = dd[max(0, t - 19):t + 1]
        w = w[~np.isnan(w)]
        in_dd[t] = len(w) > 0 and w.min() <= -thr
    since = np.where(Lpos >= 0, np.arange(N) - Lpos, -1)
    return dict(Ldd=Ldd, Lpos=Lpos, dd60=dd, in_dd=in_dd, up=C / Ldd - 1, since=since)


def frames(C: np.ndarray, O: np.ndarray, thr: float) -> dict:
    N = len(C)
    fr = state(C, thr)
    Ldd = fr["Ldd"]

    def hold(level: np.ndarray, length: int) -> np.ndarray:
        out = np.full(N, np.nan)
        for t in range(N - 1):
            if np.isnan(level[t]):
                continue
            w = C[t + 1:min(t + length, N - 1) + 1]
            if w.min() < level[t] - 1e-12:
                out[t] = 0.0                     # already broken (complete or incomplete window)
            elif t + length <= N - 1:
                out[t] = 1.0                     # full window, never broken
        return out

    Oe = np.full(N, np.nan)
    Oe[:-1] = O[1:]
    fr["hold20"], fr["hold40"] = hold(Ldd, 20), hold(Ldd, 40)
    fr["near5"] = hold(0.95 * Oe, 40)
    r20 = np.full(N, np.nan)
    r20[:N - 20] = C[20:] / Oe[:N - 20] - 1
    fr["r20"] = r20
    fr["win20"] = np.where(np.isnan(r20), np.nan, (r20 > 0).astype(float))
    brk = np.full(N, -1)
    for t in range(N - 1):
        if np.isnan(Ldd[t]):
            continue
        b = np.flatnonzero(C[t + 1:min(t + 20, N - 1) + 1] < Ldd[t] - 1e-12)
        if len(b):
            brk[t] = t + 1 + int(b[0])
    fr["brk"] = brk
    return fr


def rule(fr: dict, n: int = RULE_N, u: float = RULE_UP) -> np.ndarray:
    up = fr["up"]
    return (fr["since"] >= n) & np.where(np.isnan(up), False, up >= u - 0.0)


def bucket(up: np.ndarray) -> np.ndarray:
    b = np.digitize(up, EDGES).astype(float)
    b[np.isnan(up)] = np.nan
    return b


def matched_base(idx: pd.DatetimeIndex, fr: dict, key: str) -> np.ndarray:
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


def first_signals(sig: np.ndarray, fr: dict, idx: pd.DatetimeIndex, cool: int = 20) -> np.ndarray:
    ok = sig & fr["in_dd"] & ~np.isnan(fr["Ldd"]) & (idx >= EVAL_START) & (idx <= LAST)
    out, prev = [], -10 ** 9
    for p in np.flatnonzero(ok):
        if p - prev <= cool:
            continue
        out.append(p)
        prev = p
    return np.asarray(out, int)


def shift(sig: np.ndarray, d: int) -> np.ndarray:
    return sig if d == 0 else np.concatenate([np.zeros(d, bool), sig[:-d]])


def block_ci(v: np.ndarray, lab: np.ndarray, rng, B: int = NBOOT):
    u, inv = np.unique(lab, return_inverse=True)
    if len(u) < 2:
        return (np.nan, np.nan)
    sums, cnts = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    pick = rng.integers(0, len(u), size=(B, len(u)))
    m = sums[pick].sum(1) / cnts[pick].sum(1)
    return tuple(np.quantile(m, [0.025, 0.975]))


def period_stats(pos: np.ndarray, fr: dict, idx: pd.DatetimeIndex, bases: dict) -> dict:
    res = {}
    d = idx[pos] if len(pos) else pd.DatetimeIndex([])
    rng = np.random.default_rng(SEED)
    for name, a, b in PERIODS:
        pp = pos[(d >= a) & (d <= b)]
        r = dict(n=int(len(pp)), up_med=float(np.median(fr["up"][pp])) if len(pp) else np.nan)
        for k in KEYS:
            v, bs = fr[k][pp], bases[k][pp]
            ok = ~np.isnan(v) & ~np.isnan(bs)
            r[f"{k}_n"] = int(ok.sum())
            r[k] = float(v[ok].mean()) if ok.any() else np.nan
            r[f"{k}_base"] = float(bs[ok].mean()) if ok.any() else np.nan
            if k == "hold20":
                if ok.sum() >= 2:
                    lab = pp[ok] // 20
                    r["hold20_lo"], r["hold20_hi"] = block_ci(v[ok], lab, rng)
                    r["hold20_dlo"], r["hold20_dhi"] = block_ci(v[ok] - bs[ok], lab, rng)
                else:
                    r["hold20_lo"] = r["hold20_hi"] = r["hold20_dlo"] = r["hold20_dhi"] = np.nan
        res[name] = r
    return res


def l1(st: dict) -> bool:
    m, y6, y5 = st["2024~2026"], st["2026"], st["2025"]
    c1 = m["hold20_n"] >= 8 and m["hold20"] >= 0.70
    c2 = y6["hold20_n"] >= 3 and y6["hold20"] >= 0.70
    c3 = y5["hold20_n"] < 3 or y5["hold20"] >= 0.60
    return bool(c1 and c2 and c3)


def l2(st: dict) -> bool:
    return bool(l1(st) and st["2024~2026"].get("hold20_dlo", np.nan) > 0)


def fmt_row(label: str, st: dict) -> str:
    def f(p):
        r = st[p]
        return f"{r['hold20']:.3f}({r['hold20_n']:2d})" if r["hold20_n"] else f"  -  ({r['n']:2d})"
    m = st["2024~2026"]
    return (f"  {label:36s} main {f('2024~2026'):10s} matched {m['hold20_base']:.3f} dlo {m.get('hold20_dlo', np.nan):+.3f} | "
            f"2026 {f('2026'):10s} H1 {f('2026H1'):10s} H2 {f('2026H2'):10s} 2025 {f('2025'):10s} 2024 {f('2024'):10s} "
            f"upmed {m['up_med']:.3f} near5 {m['near5']:.2f} win20 {m['win20']:.2f} L1={l1(st)} L2={l2(st)}")


# ------------------------------------------------------------------ null samplers
def null_draws(cands: list, hold: np.ndarray, cool: int, rng, ndraw: int = NDRAW):
    """Each draw picks one day from each candidate set (random order), pairwise spacing > cool; returns mean hold."""
    res, tries = [], 0
    order = np.arange(len(cands))
    while len(res) < ndraw and tries < ndraw * 200:
        tries += 1
        rng.shuffle(order)
        chosen = []
        for i in order:
            c = cands[i]
            if chosen:
                ch = np.asarray(chosen)
                c = c[np.abs(c[:, None] - ch[None, :]).min(1) > cool]
            if len(c) == 0:
                chosen = None
                break
            chosen.append(int(rng.choice(c)))
        if chosen is None:
            continue
        res.append(float(hold[np.asarray(chosen)].mean()))
    return np.asarray(res), tries


def rejection_draws(cands: list, hold: np.ndarray, cool: int, rng, want: int = 2000, cap: int = 5_000_000,
                    batch: int = 500_000):
    """Exact uniform sampling over spaced configurations: draw one day per set independently, keep the draw only when
    every pair is > cool sessions apart (vectorised). Usually infeasible here (acceptance ~1e-8); reported for honesty."""
    acc, tries = [], 0
    while sum(len(a) for a in acc) < want and tries < cap:
        ch = np.stack([c[rng.integers(0, len(c), size=batch)] for c in cands], 1)
        tries += batch
        s = np.sort(ch, 1)
        ok = (np.diff(s, axis=1) > cool).all(1)
        if ok.any():
            acc.append(hold[ch[ok]].mean(1))
    acc = np.concatenate(acc) if acc else np.array([])
    return acc, tries


def gibbs_draws(cands: list, hold: np.ndarray, cool: int, rng, start: np.ndarray, nkeep: int = NDRAW,
                burn: int = 2000, thin: int = 5):
    """Single-site Gibbs sampler whose stationary law is UNIFORM over configurations (one day per set, all pairs
    > cool apart): each sweep resamples every slot uniformly among its candidates compatible with the others.
    Unlike the sequential sampler it does not over-weight slots drawn first."""
    ch = np.asarray(start, int).copy()
    K = len(cands)
    out = []
    for sweep in range(burn + nkeep * thin):
        for i in rng.permutation(K):
            others = np.delete(ch, i)
            c = cands[i]
            ok = np.abs(c[:, None] - others[None, :]).min(1) > cool
            ch[i] = int(rng.choice(c[ok]))
        if sweep >= burn and (sweep - burn) % thin == 0:
            out.append(float(hold[ch].mean()))
    return np.asarray(out)


def one_seq_draw(cands: list, cool: int, rng):
    for _ in range(100000):
        order = rng.permutation(len(cands))
        ch = np.full(len(cands), -1)
        okall = True
        for i in order:
            c = cands[i]
            got = ch[ch >= 0]
            if len(got):
                c = c[np.abs(c[:, None] - got[None, :]).min(1) > cool]
            if len(c) == 0:
                okall = False
                break
            ch[i] = int(rng.choice(c))
        if okall:
            return ch
    raise RuntimeError("no spaced configuration found")


# ------------------------------------------------------------------ main
def main():
    pd.set_option("display.width", 260)
    df, chk, other = load_soxx()
    idx = df.index
    C, O = df.C.to_numpy(float), df.O.to_numpy(float)
    N = len(C)
    say("=" * 128)
    say("Independent verification round 2 (fixed right-censoring): family f3, rule since5_up3, kind SOXX (bc_lib NOT imported)")
    say("Rule: sessions since the decline low (LAST position of the min close since the latest 60-session closing high) >= 5")
    say("      and up = C[t]/Ldd[t] - 1 >= 3%; first signals = in_dd(10%) days 2024-01-02..2026-10-23, 20-session cooldown.")
    say("=" * 128)
    say(f"Data: SOXX close {idx[0].date()}..{idx[-1].date()} N={N}; open from {df.O.first_valid_index().date()}; "
        f"overlap rows {chk['overlap_n']} max|diff|={chk['overlap_maxdiff']:.1e}; XNYS missing={chk['cal_missing']} "
        f"extra={chk['cal_extra']}")

    fr = frames(C, O, 0.10)
    bases = {k: matched_base(idx, fr, k) for k in KEYS}
    sig = rule(fr)
    pos = first_signals(sig, fr, idx)

    # ============================================================ [1] independent match
    say("\n[1] Independent re-implementation vs family files (results/f3_SOXX_signals.csv, f3_SOXX_summary.csv)")
    fam = pd.read_csv(RES / "f3_SOXX_signals.csv", parse_dates=["date"])
    fam = fam[fam.rule == "since5_up3"].reset_index(drop=True)
    fsum = pd.read_csv(RES / "f3_SOXX_summary.csv")
    fsum = fsum[fsum.rule == "since5_up3"].set_index("period")
    mine = pd.DataFrame(dict(date=idx[pos], pos=pos, Ldd_date=idx[fr["Lpos"][pos]], Ldd=fr["Ldd"][pos], C=C[pos],
                             since=fr["since"][pos], up=fr["up"][pos], dd60=fr["dd60"][pos],
                             hold20=fr["hold20"][pos], base_hold20=bases["hold20"][pos], hold40=fr["hold40"][pos],
                             base_hold40=bases["hold40"][pos], near5=fr["near5"][pos], base_near5=bases["near5"][pos],
                             win20=fr["win20"][pos], base_win20=bases["win20"][pos], r20=fr["r20"][pos],
                             base_r20=bases["r20"][pos]))
    show = mine.copy()
    show["Ldd_date"] = [d.date() for d in show.Ldd_date]
    show["date"] = [d.date() for d in show.date]
    show["broke"] = [str(idx[fr["brk"][p]].date()) if fr["brk"][p] >= 0 else "-" for p in pos]
    say(show[["date", "pos", "Ldd_date", "Ldd", "C", "since", "up", "dd60", "hold20", "base_hold20", "broke", "hold40",
              "near5", "win20", "r20"]].round(4).to_string(index=False))
    same_dates = list(mine.date) == list(fam.date)
    say(f"  first-signal dates identical: {same_dates} (mine {len(mine)}, family {len(fam)})")
    mism = []
    if same_dates:
        for k in ("pos", "up", "dd60", "hold20", "base_hold20", "hold40", "base_hold40", "near5", "base_near5",
                  "win20", "base_win20", "r20", "base_r20"):
            a, b = mine[k].to_numpy(float), fam[k].to_numpy(float)
            bad = ~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-9))
            if bad.any():
                mism.append((k, int(bad.sum())))
        say(f"  per-signal field mismatches: {mism if mism else 'none'}")
    else:
        say(f"  only mine: {sorted(set(mine.date) - set(fam.date))}; only family: {sorted(set(fam.date) - set(mine.date))}")
    st0 = period_stats(pos, fr, idx, bases)
    say("  period      mine hold20(n)  fam hold20(n)  mine base fam base | mine dlo fam dlo | mine near5/win20/r20/hold40 vs family")
    per_ok = True
    for p, _, _ in PERIODS:
        r = st0[p]
        fh, fn, fb = fsum.loc[p, "hold20"], int(fsum.loc[p, "hold20_n"]), fsum.loc[p, "hold20_base"]
        okp = r["hold20_n"] == fn and abs(r["hold20"] - fh) < 1e-9 and abs(r["hold20_base"] - fb) < 1e-9
        for k in ("near5", "win20", "r20", "hold40"):
            okp &= bool((np.isnan(r[k]) and np.isnan(fsum.loc[p, k])) or abs(r[k] - fsum.loc[p, k]) < 1e-9)
        per_ok &= bool(okp)
        say(f"  {p:10s}  {r['hold20']:.4f} ({r['hold20_n']:2d})  {fh:.4f} ({fn:2d})   {r['hold20_base']:.4f}  {fb:.4f} | "
            f"{r['hold20_dlo']:+.3f}  {fsum.loc[p, 'hold20_dlo']:+.3f} | {r['near5']:.3f}/{r['win20']:.3f}/{r['r20']:+.4f}/"
            f"{r['hold40']:.3f}  {'OK' if okp else 'MISMATCH'}")
    indep_match = bool(same_dates and not mism and per_ok)
    say(f"  => independent match: {indep_match}  (bootstrap bounds use a different RNG stream; compared for sign only)")
    m0 = st0["2024~2026"]
    pool_m = fr["in_dd"] & ~np.isnan(fr["Ldd"]) & ~np.isnan(fr["hold20"]) & (idx >= EVAL_START) & (idx <= LAST)
    unc = float(fr["hold20"][pool_m].mean())
    say(f"  main: n={m0['n']} hold20={m0['hold20']:.4f} vs matched {m0['hold20_base']:.4f} (diff {m0['hold20'] - m0['hold20_base']:+.4f}, "
        f"dlo {m0['hold20_dlo']:+.4f}); near5 {m0['near5']:.3f} (matched {m0['near5_base']:.3f}); win20 {m0['win20']:.3f} "
        f"(matched {m0['win20_base']:.3f}); r20 {m0['r20']:+.4f} (matched {m0['r20_base']:+.4f}); up median {m0['up_med']:.4f}")
    say(f"  unconditional in_dd hold20 2024~2026 = {unc:.4f} (resolved days {int(pool_m.sum())})")
    k_, n_ = int(round(m0["hold20"] * m0["hold20_n"])), m0["hold20_n"]
    for lab, p0 in (("unconditional in_dd", unc), ("matched mean", m0["hold20_base"])):
        pb = sum(comb(n_, j) * p0 ** j * (1 - p0) ** (n_ - j) for j in range(k_, n_ + 1))
        say(f"  binomial P(X >= {k_}/{n_} | p={p0:.4f}, {lab}) = {pb:.4f}")
    late = [p for p in np.flatnonzero(sig & fr["in_dd"] & (idx > idx[pos[-1]] + pd.Timedelta(days=0)))]
    say(f"  raw rule days in in_dd after the last first signal {idx[pos[-1]].date()}: "
        f"{[str(idx[p].date()) for p in late] or 'none'}")
    oct_dd = np.flatnonzero(fr["in_dd"] & (idx >= pd.Timestamp("2026-09-24")))
    say("  in_dd days after the 2026-09-03 cooldown window (since / up): " +
        ", ".join(f"{idx[p].date()}({fr['since'][p]},{fr['up'][p] * 100:.1f}%)" for p in oct_dd))
    b26 = [p for p in pos if abs(fr["up"][p] - 0.08) < 1e-3]
    for p in b26:
        say(f"  bucket-edge check: {idx[p].date()} up = {fr['up'][p]:.10f} (C {C[p]:.6f} / Ldd {fr['Ldd'][p]:.6f}); "
            f"bucket {BNAMES[int(bucket(fr['up'])[p])]} matched {bases['hold20'][p]:.4f}; if it fell in 3-8% the matched value "
            f"would be {np.nanmean(fr['hold20'][pool_m & (idx.year == idx[p].year) & (bucket(fr['up']) == 1)]):.4f}")

    # ============================================================ [2] look-ahead audit
    say("\n[2] Look-ahead / timing / data audit")
    raw_eval = np.flatnonzero(sig & (idx >= EVAL_START))
    test_pos = np.unique(np.concatenate([raw_eval, np.arange(np.searchsorted(idx, EVAL_START), N, 5)]))
    bad = 0
    for p in test_pos:
        s_tr = state(C[:p + 1], 0.10)
        same = (s_tr["in_dd"][p] == fr["in_dd"][p] and s_tr["Lpos"][p] == fr["Lpos"][p]
                and abs(s_tr["Ldd"][p] - fr["Ldd"][p]) < 1e-12 and s_tr["since"][p] == fr["since"][p])
        bad += int(not same)
    say(f"  truncation test (state recomputed with prices cut at t) on {len(raw_eval)} raw rule days + every 5th eval day "
        f"({len(test_pos)} days): {'all identical' if bad == 0 else f'{bad} DIFFERENCES'}")
    ties = int(sum(int((C[fr['Lpos'][t] - 0:t + 1] == fr['Ldd'][t]).sum() > 1) for t in range(N) if fr['Lpos'][t] >= 0))
    say(f"  ties in the decline-low close (more than one day equal to Ldd inside [low, t]): {ties} days -> "
        f"'last occurrence' vs 'first occurrence' is immaterial")
    say("  code read (f3_time.py): decline_low_pos uses w = c[t-59:t+1] (ends at t), pk = latest argmax, LL = latest argmin of")
    say("  c[pk:t+1]; since = t - LL; since5_up3 = (since >= 5) & (bc_lib up >= 0.03); up = C/Ldd - 1 at t. No market,")
    say("  flow or news input; KNOWN_LOWS (hindsight lows) is used only in the printed soxx_lows_check, never in a rule.")
    say("  bc_lib.first_signals: in_dd/Ldd at t, eval window, cooldown (p - prev <= 20 dropped). Outcomes start at C[t+1];")
    say("  entry is O[t+1]; near5 uses C[t+1..t+40] vs 0.95*O[t+1]. No forward window enters the signal.")
    r1 = np.diff(np.log(C))
    big = np.flatnonzero(np.abs(r1) > 0.07) + 1
    say(f"  SOXX |daily log return| > 7%: {[f'{idx[p].date()} {r1[p - 1]:+.3f}' for p in big]}")
    gap = np.log(O[1:] / C[:-1])
    bg = [p + 1 for p in np.flatnonzero(np.abs(np.nan_to_num(gap)) > 0.06)]
    say(f"  SOXX |open / prev close| gaps > 6%: {[f'{idx[p].date()} {gap[p - 1]:+.3f}' for p in bg]}")
    say(f"  O[t+1] present for every first signal: {bool(np.isfinite(O[pos + 1]).all())}")
    # cross-check vs ^SOX / SMH (data error days): daily-return disagreement and the Ldd days of the signals
    sox, smh = other["^SOX"].to_numpy(float), other["SMH"].to_numpy(float)
    rs, rx, rm = np.diff(np.log(C)), np.diff(np.log(sox)), np.diff(np.log(smh))
    dis = np.flatnonzero((np.abs(rs - rx) > 0.02) & ~np.isnan(rx)) + 1
    say(f"  days where SOXX and ^SOX daily returns differ by > 2pp (^SOX to {other['^SOX'].last_valid_index().date()}): "
        f"{[f'{idx[p].date()} SOXX {rs[p - 1]:+.3f} SOX {rx[p - 1]:+.3f} SMH {rm[p - 1]:+.3f}' for p in dis] or 'none'}")
    fl = []
    for p in pos:
        lp = fr["Lpos"][p]
        fl.append(f"{idx[p].date()}: low {idx[lp].date()} ret_in {rs[lp - 1]:+.3f} ret_out {rs[lp]:+.3f} "
                  f"(SOX {rx[lp - 1]:+.3f}/{rx[lp]:+.3f})" if lp + 1 < N and lp - 1 >= 0 else f"{idx[p].date()}")
    say("  decline-low days of the first signals, return into / out of the low (SOXX vs ^SOX):")
    for s in fl:
        say("    " + s)
    # same rule on ^SOX (independent price source) where available
    ok_sox = ~np.isnan(sox)
    if ok_sox.all() or True:
        last_sox = int(np.flatnonzero(ok_sox)[-1])
        frx = frames(sox[:last_sox + 1], sox[:last_sox + 1], 0.10)   # open not available: near5/r20 meaningless here
        idx_x = idx[:last_sox + 1]
        bx = {k: matched_base(idx_x, frx, k) for k in ("hold20",)}
        px = first_signals(rule(frx), frx, idx_x)
        say(f"  same rule on ^SOX closes (price index, not the ETF; to {idx_x[-1].date()}): hold20 {np.nanmean(frx['hold20'][px]):.3f} "
            f"(n={int((~np.isnan(frx['hold20'][px])).sum())}) vs matched {np.nanmean(bx['hold20'][px]):.3f}; "
            f"2026 {np.nanmean(frx['hold20'][px[idx_x[px] >= pd.Timestamp('2026-01-02')]]):.3f}; per signal (date, since, up, hold20):")
        say("    " + ", ".join(f"{idx_x[p].date()}({frx['since'][p]},{frx['up'][p] * 100:.1f}%,{frx['hold20'][p]:.0f})" for p in px))
        q = int(np.searchsorted(idx_x, pd.Timestamp("2026-06-12")))
        say(f"    ^SOX on 2026-06-12: in_dd {frx['in_dd'][q]}, since {frx['since'][q]}, up {frx['up'][q] * 100:.2f}%, low "
            f"{idx_x[frx['Lpos'][q]].date()} (SOXX: since {fr['since'][q]}, up {fr['up'][q] * 100:.2f}%, low {idx[fr['Lpos'][q]].date()})")

    # ============================================================ [3] robustness
    say("\n[3] Robustness (hold20 with resolved n per period; matched = year x up-bucket; dlo = block-bootstrap lower bound of "
        "hold20 - matched; L1/L2 = registered SOXX criteria)")
    rob = {}
    frc = {0.10: (fr, bases)}

    def run(label, n=RULE_N, u=RULE_UP, cool=20, thr=0.10, delay=0, sigv=None, core=True, show_fail=True):
        if thr not in frc:
            f = frames(C, O, thr)
            frc[thr] = (f, {k: matched_base(idx, f, k) for k in KEYS})
        f, bs = frc[thr]
        s = rule(f, n, u) if sigv is None else sigv
        ps = first_signals(shift(s, delay), f, idx, cool)
        st = period_stats(ps, f, idx, bs)
        rob[label] = (st, core, ps)
        say(fmt_row(label, st))
        badp = [p for p in ps if f["hold20"][p] == 0]
        if badp and show_fail:
            say("      failed hold20: " + ", ".join(f"{idx[p].date()} (up {f['up'][p]:.3f}, since {f['since'][p]}, "
                                                   f"broke {idx[f['brk'][p]].date()})" for p in badp))
        return st

    run("registered since5_up3")
    say("  -- timing")
    run("delay +1 session", delay=1)
    run("delay +2 sessions", delay=2)
    say("  -- cooldown / in_dd threshold")
    run("cooldown 10", cool=10)
    run("cooldown 30", cool=30)
    run("in_dd thr 8%", thr=0.08)
    run("in_dd thr 12%", thr=0.12)
    say("  -- registered-grid neighbours (F3 cells) and one step either side")
    for n_, u_ in ((5, 0.0), (10, 0.03), (4, 0.03), (6, 0.03), (5, 0.02), (5, 0.04)):
        run(f"since{n_}_up{int(round(u_ * 100))}", n=n_, u=u_)
    say("  -- neighbour grid (post hoc): since n = 3..10 x up = 2, 2.5, 3, 3.5, 4, 5% -> main hold20(n) / 2026 hold20(n) / "
        "matched / L1")
    grid_rows = []
    for n_ in range(3, 11):
        cells = []
        for u_ in (0.02, 0.025, 0.03, 0.035, 0.04, 0.05):
            ps = first_signals(rule(fr, n_, u_), fr, idx)
            st = period_stats(ps, fr, idx, bases)
            m, y6 = st["2024~2026"], st["2026"]
            grid_rows.append(dict(n=n_, u=u_, main=m["hold20"], nm=m["hold20_n"], base=m["hold20_base"], y26=y6["hold20"],
                                  n26=y6["hold20_n"], L1=l1(st), L2=l2(st)))
            cells.append(f"{m['hold20']:.2f}({m['hold20_n']})/{y6['hold20']:.2f}({y6['hold20_n']})/{m['hold20_base']:.2f}"
                         f"{'*' if l1(st) else ' '}")
        say(f"    since{n_:<2d} " + "  ".join(cells))
    G = pd.DataFrame(grid_rows)
    say(f"    (* = L1). L1 in {int(G.L1.sum())}/{len(G)} grid cells; L2 in {int(G.L2.sum())}/{len(G)}; cells with main hold20 > matched: "
        f"{int((G.main > G.base + 1e-12).sum())}/{len(G)}; mean(main - matched) {np.mean(G.main - G.base):+.3f}")
    say(f"    L1 cells: {[f'since{int(r.n)}_up{r.u * 100:g}' for r in G[G.L1].itertuples()]}")
    core = {k: v for k, v in rob.items() if v[1]}
    say(f"  core variants (timing/cooldown/threshold/neighbours): L1 in {sum(l1(v[0]) for v in core.values())}/{len(core)}; "
        f"L2 in {sum(l2(v[0]) for v in core.values())}/{len(core)}; L1 fails: {[k for k, v in core.items() if not l1(v[0])]}")

    # ============================================================ [4] same-state null
    say(f"\n[4] Same-state null: random in_dd days 2024-01-02..2026-10-23 (resolved hold20), one per first signal matched on the "
        f"signal's year and up bucket, pairwise spacing > 20 sessions, {NDRAW} draws; p = P(null mean >= observed)")
    b_all = bucket(fr["up"])
    yr = idx.year.to_numpy()
    pool_pos = np.flatnonzero(pool_m)
    obs = float(fr["hold20"][pos].mean())
    lab_yb = [(int(yr[p]), BNAMES[int(b_all[p])]) for p in pos]
    say(f"  observed hold20 = {obs:.4f} (n={len(pos)}); year x bucket = {lab_yb}")
    tab = pd.DataFrame(dict(y=yr[pool_pos], b=b_all[pool_pos].astype(int), h=fr["hold20"][pool_pos]))
    say("  pool hold20 by year x bucket (days): " + "; ".join(
        f"{y}/{BNAMES[b]} {g.h.mean():.3f}({len(g)})" for (y, b), g in tab.groupby(["y", "b"])))
    rng = np.random.default_rng(SEED)
    c_yb = [pool_pos[(yr[pool_pos] == yr[p]) & (b_all[pool_pos] == b_all[p])] for p in pos]
    d_yb, tr = null_draws(c_yb, fr["hold20"], 20, rng)
    p_yb = float((d_yb >= obs - 1e-12).mean())
    say(f"  [N1] year x up-bucket (registered matching): null mean {d_yb.mean():.4f} sd {d_yb.std():.4f} "
        f"[2.5%, 97.5%] [{np.quantile(d_yb, .025):.3f}, {np.quantile(d_yb, .975):.3f}]  p = {p_yb:.4f}  (attempts {tr})")
    acc, trr = rejection_draws(c_yb, fr["hold20"], 20, np.random.default_rng(SEED + 1))
    say(f"  [N1r] exact rejection sampling: accepted {len(acc)}/{trr} -> infeasible (14 spaced picks from narrow year x bucket "
        f"pools), replaced by a Gibbs sampler with the same uniform target:")
    rng_g = np.random.default_rng(SEED + 3)
    d_g = np.concatenate([gibbs_draws(c_yb, fr["hold20"], 20, rng_g, one_seq_draw(c_yb, 20, rng_g), nkeep=NDRAW // 2)
                          for _ in range(2)])
    p_rej = float((d_g >= obs - 1e-12).mean())
    say(f"  [N1g] year x up-bucket, Gibbs (2 chains x {NDRAW // 2}, burn 2000, thin 5): mean {d_g.mean():.4f} sd {d_g.std():.4f} "
        f"p = {p_rej:.4f}")
    rng_i = np.random.default_rng(SEED + 2)
    ind = np.array([[float(fr["hold20"][int(rng_i.choice(c))]) for c in c_yb] for _ in range(NDRAW)]).mean(1)
    p_ind = float((ind >= obs - 1e-12).mean())
    say(f"  [N1i] same pools, independent draws (no spacing): mean {ind.mean():.4f} sd {ind.std():.4f} p = {p_ind:.4f}")
    c_f = [pool_pos[(yr[pool_pos] == yr[p]) & (np.abs(fr["up"][pool_pos] - fr["up"][p]) <= 0.02 + 1e-12)] for p in pos]
    say("  fine pools (same year, up within +-2pp): " +
        ", ".join(f"{idx[p].date()}:{len(c)} (h {fr['hold20'][c].mean():.2f})" for p, c in zip(pos, c_f)))
    d_f, tr = null_draws(c_f, fr["hold20"], 20, rng)
    p_f = float((d_f >= obs - 1e-12).mean())
    say(f"  [N2] same year, up +-2pp: null mean {d_f.mean():.4f} sd {d_f.std():.4f}  p = {p_f:.4f}  (attempts {tr})")
    d_fg = gibbs_draws(c_f, fr["hold20"], 20, rng_g, one_seq_draw(c_f, 20, rng_g))
    p_fg = float((d_fg >= obs - 1e-12).mean())
    say(f"  [N2g] same, Gibbs uniform sampler: mean {d_fg.mean():.4f} sd {d_fg.std():.4f} p = {p_fg:.4f}")
    # null restricted to days already in the rule's own state (since >= 5 and up >= 3%): is the FIRST such day special?
    st_on = sig[pool_pos]
    c_on = [pool_pos[st_on & (yr[pool_pos] == yr[p]) & (b_all[pool_pos] == b_all[p])] for p in pos]
    d_on, tr = null_draws(c_on, fr["hold20"], 20, rng)
    p_on = float((d_on >= obs - 1e-12).mean())
    say(f"  [N2s] year x up-bucket, pool restricted to days where the rule state is on (since>=5 & up>=3%): null mean "
        f"{d_on.mean():.4f} p = {p_on:.4f}  pool sizes {[len(c) for c in c_on]}")
    m26 = idx[pos] >= pd.Timestamp("2026-01-02")
    obs26 = float(fr["hold20"][pos[m26]].mean())
    d26, _ = null_draws([c for c, k in zip(c_yb, m26) if k], fr["hold20"], 20, rng)
    d26f, _ = null_draws([c for c, k in zip(c_f, m26) if k], fr["hold20"], 20, rng)
    p26, p26f = float((d26 >= obs26 - 1e-12).mean()), float((d26f >= obs26 - 1e-12).mean())
    say(f"  [N3] 2026 only (n={int(m26.sum())}, obs {obs26:.3f}): year x bucket null mean {d26.mean():.3f} p = {p26:.4f}; "
        f"+-2pp null mean {d26f.mean():.3f} p = {p26f:.4f}")
    # does 'time since the low' add anything at equal rebound size? all in_dd days, no cooldown, stratified
    say("  [N4] stratified on/off test (all resolved in_dd days 2024~2026, no cooldown): within year x up-bucket, hold20 of days")
    say("       with since >= 5 vs since < 5 (the rule's own extra condition), weighted by on-days")
    on = fr["since"] >= RULE_N
    rows = []
    for (y, b), g in pd.DataFrame(dict(p=pool_pos, y=yr[pool_pos], b=b_all[pool_pos].astype(int))).groupby(["y", "b"]):
        pp = g.p.to_numpy()
        a, z = pp[on[pp]], pp[~on[pp]]
        rows.append((y, BNAMES[b], len(a), fr["hold20"][a].mean() if len(a) else np.nan, len(z),
                     fr["hold20"][z].mean() if len(z) else np.nan))
    T = pd.DataFrame(rows, columns=["year", "bucket", "n_on", "h_on", "n_off", "h_off"])
    say(T.round(3).to_string(index=False))
    w = T.dropna()
    wd = float(np.average(w.h_on - w.h_off, weights=w.n_on))
    say(f"       weighted (on - off) hold20 difference, strata with both: {wd:+.3f}  (positive = since>=5 adds information)")

    def onoff(sel: np.ndarray) -> float:
        g = pd.DataFrame(dict(y=yr[sel], b=b_all[sel].astype(int), on=on[sel], h=fr["hold20"][sel]))
        num = den = 0.0
        for _, gg in g.groupby(["y", "b"]):
            a, z = gg[gg.on], gg[~gg.on]
            if len(a) and len(z):
                num += len(a) * (a.h.mean() - z.h.mean())
                den += len(a)
        return num / den if den else np.nan
    blk = pool_pos // 20
    ub = np.unique(blk)
    rngb = np.random.default_rng(SEED + 5)
    bs_od = []
    for _ in range(1000):
        pick = rngb.choice(ub, size=len(ub))
        sel = np.concatenate([pool_pos[blk == b_] for b_ in pick])
        bs_od.append(onoff(sel))
    bs_od = np.asarray(bs_od)
    say(f"       20-session block bootstrap (1000) of that difference: 95% [{np.nanquantile(bs_od, .025):+.3f}, "
        f"{np.nanquantile(bs_od, .975):+.3f}]; by year: " +
        ", ".join(f"{y} {onoff(pool_pos[yr[pool_pos] == y]):+.3f}" for y in (2024, 2025, 2026)))
    say("       (day-level association; a since<5 day inside a bucket is often a day at/near a fresh low still falling.")
    say("        It does not transfer to the rule's FIRST signals, which sit at the low edge of their up bucket - see N1/N2s.)")
    up_only = pool_pos[np.where(np.isnan(fr["up"][pool_pos]), False, fr["up"][pool_pos] >= RULE_UP)]
    say(f"       among in_dd days with up >= 3%: since>=5 hold20 {fr['hold20'][up_only[on[up_only]]].mean():.3f} "
        f"(n={int(on[up_only].sum())}) vs since<5 {fr['hold20'][up_only[~on[up_only]]].mean():.3f} (n={int((~on[up_only]).sum())})")
    # informative signals
    inf = pos[bases["hold20"][pos] < 0.95]
    say(f"  informative signals (matched base < 0.95; the others cannot fall short of their bucket by much): {len(inf)}; held "
        f"{int(np.nansum(fr['hold20'][inf]))}/{len(inf)}, expected {np.nansum(bases['hold20'][inf]):.2f}; "
        f"{[(str(idx[p].date()), BNAMES[int(b_all[p])], round(bases['hold20'][p], 3), fr['hold20'][p]) for p in inf]}")
    say(f"  multiple testing: {N_CELLS} cells evaluated; Bonferroni 0.05/{N_CELLS} = {0.05 / N_CELLS:.5f}. The registered "
        f"main hold20 is below its matched value, so no correction is needed to reject 'information beyond rebound size'.")

    # ============================================================ [5] mechanical comparison
    say("\n[5] Mechanical comparison: plain up >= candidate median up (same in_dd, window, cooldown, outcomes)")
    upmed = float(np.median(fr["up"][pos]))
    pl = {}
    for x in (upmed, 0.08, 0.05, 0.03):
        s = np.where(np.isnan(fr["up"]), False, fr["up"] >= x)
        st = run(f"plain up >= {x:.4f}", sigv=s, core=False, show_fail=False)
        pl[x] = (st, rob[f"plain up >= {x:.4f}"][2])
    st_pm, pos_pm = pl[upmed]
    say(f"  candidate main {m0['hold20']:.3f} (n={m0['hold20_n']}, up median {m0['up_med']:.3f}) vs plain up >= {upmed:.3f} "
        f"{st_pm['2024~2026']['hold20']:.3f} (n={st_pm['2024~2026']['hold20_n']}); 2026 {st0['2026']['hold20']:.3f} vs "
        f"{st_pm['2026']['hold20']:.3f}")
    say("  episode pairing: each candidate signal vs the nearest plain-up>=median first signal (sessions apart, hold20 each)")
    for p in pos:
        q = pos_pm[np.argmin(np.abs(pos_pm - p))] if len(pos_pm) else -1
        say(f"    {idx[p].date()} up {fr['up'][p]:.3f} h {fr['hold20'][p]:.0f}  <->  {idx[q].date()} (+{q - p:3d}) up "
            f"{fr['up'][q]:.3f} h {fr['hold20'][q]:.0f}")
    say("  candidate signals by up bucket (held/n, matched mean):")
    for b in range(4):
        m = b_all[pos] == b
        if m.any():
            say(f"    {BNAMES[b]:6s}: {int(np.nansum(fr['hold20'][pos[m]]))}/{int(m.sum())}  matched "
                f"{np.mean(bases['hold20'][pos[m]]):.3f}  dates {[str(idx[p].date()) for p in pos[m]]}")

    # ============================================================ [d] 2026 detail
    say("\n[d] Period detail vs matched")
    for p_ in ("2026", "2026H1", "2026H2", "2025", "2024"):
        r = st0[p_]
        say(f"  {p_:7s} n={r['n']} hold20 {r['hold20']:.3f} matched {r['hold20_base']:.3f} diff {r['hold20'] - r['hold20_base']:+.3f} "
            f"dlo {r['hold20_dlo']:+.3f} | near5 {r['near5']:.3f}/{r['near5_base']:.3f} win20 {r['win20']:.3f}/{r['win20_base']:.3f} "
            f"r20 {r['r20']:+.3f}/{r['r20_base']:+.3f} hold40 {r['hold40']:.3f}({r['hold40_n']})/{r['hold40_base']:.3f}")
    say("  2026 decline episodes (in_dd runs) and what the rule did:")
    d26 = np.flatnonzero(fr["in_dd"] & (idx >= pd.Timestamp("2026-01-02")))
    runs = np.split(d26, np.flatnonzero(np.diff(d26) > 1) + 1)
    for rr in runs:
        lows = sorted(set(int(fr["Lpos"][p]) for p in rr))
        deep = int(rr[np.argmin(fr["dd60"][rr])])
        fs = [p for p in pos if rr[0] <= p <= rr[-1]]
        fs_txt = [str(idx[p].date()) + " h=" + format(fr["hold20"][p], ".0f") for p in fs]
        lows_txt = [str(idx[q].date()) for q in lows][:6]
        say(f"    in_dd {idx[rr[0]].date()}..{idx[rr[-1]].date()} ({len(rr)} d): deepest dd60 {fr['dd60'][deep]:+.3f} on "
            f"{idx[deep].date()}; decline-low dates seen {lows_txt}{'...' if len(lows) > 6 else ''}; first signals {fs_txt}")

    # ============================================================ [e] STK replication (independent)
    say("\n[e] Stocks (82 US names, in_dd 20%, eval 2025-09-01..2026-10-23): independent recompute of since5_up3")
    say(stk_check())

    # ============================================================ verdict inputs
    say("\n[6] Verdict inputs")
    vin = dict(indep_match=indep_match, main_hold20=m0["hold20"], main_matched=m0["hold20_base"], p_null_yb=p_yb,
               p_null_yb_gibbs=p_rej, p_null_fine=p_f, p_null_fine_gibbs=p_fg, p_null_rule_state=p_on, p_2026=p26,
               p_2026_fine=p26f, onoff_diff=wd, plain_upmed_main=st_pm["2024~2026"]["hold20"])
    say("  " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in vin.items()))
    say(f"  core variants losing L1: {[k for k, v in core.items() if not l1(v[0])]}")
    edge = [p for p in pos if abs(fr["up"][p] - 0.08) < 1e-6]
    if edge:
        alt = bases["hold20"][pos].copy()
        for p in edge:
            alt[pos == p] = np.nanmean(fr["hold20"][pool_m & (idx.year == idx[p].year) & (bucket(fr["up"]) == 1)])
        say(f"  bucket-edge counterfactual: if {[str(idx[p].date()) for p in edge]} (up = 8.0000%) were binned 3-8%, the main "
            f"matched mean would be {alt.mean():.4f} (diff {m0['hold20'] - alt.mean():+.4f}); still no excess.")

    say("\n[7] Conclusion")
    say(f"  - Reproduced exactly: 14 first signals, 12/14 = {m0['hold20']:.3f} held (2026 3/4, 2025 3/4, 2024 6/6); L1 passes as "
        f"registered, L2 fails.")
    say(f"  - No information beyond rebound size: matched {m0['hold20_base']:.3f} > observed; same-state null p = {p_yb:.2f} "
        f"(Gibbs {p_rej:.2f}), +-2pp p = {p_f:.2f} (Gibbs {p_fg:.2f}); 2026 p = {p26:.2f}. Informative signals (3-8% bucket) held 2/4 "
        f"vs 2.76 expected; all 10 signals with up >= 8% sat in buckets that hold ~99% anyway.")
    say(f"  - Dominated by the mechanical rule: plain up >= {upmed:.3f} (candidate median) gives "
        f"{st_pm['2024~2026']['hold20']:.3f} ({st_pm['2024~2026']['hold20_n']}) and 2026 {st_pm['2026']['hold20']:.3f}. "
        f"The since>=5 filter only adds earlier, lower-up entries, which are the ones that fail.")
    say("  - Fragile: L1 lost with a 1- or 2-session delay (2026 -> 2/4), with since 4 or 6 or 10 (2026 3/5, 2/3, 2/3), with in_dd 8%,")
    say("    and on ^SOX closes (2026 2/3). L1 survives only along the up axis at since = 5 (up is non-binding there) and cooldown/12%.")
    say(f"  - Stocks: the same rule on 82 AI names is ~0.53 in 2026 (matched ~0.55): no replication.")
    say(f"  - Cost of lateness: up median {m0['up_med']:.3f}; win20 {m0['win20']:.3f} vs matched {m0['win20_base']:.3f}; r20 "
        f"{m0['r20']:+.4f} vs {m0['r20_base']:+.4f}; 2026 hold40 {st0['2026']['hold40']:.3f} ({st0['2026']['hold40_n']}).")
    say("  - Verdict: REFUTED as a real 70%+ bottom confirmation. The 12/14 is what a ~10% rebound off the decline low delivers")
    say("    by itself in this sample (same-state null centred at ~0.89), and the registered L1 pass sits on a knife edge")
    say("    (2026 n = 4, one signal from failing; delay/neighbour/threshold/index variants fail).")
    OUT.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nwrote {OUT}")


# ------------------------------------------------------------------ STK independent recompute
def stk_check() -> str:
    out = io.StringIO()
    o = pd.read_csv(ROOT / "data/derived/desk/prices_open.csv", index_col=0, parse_dates=True)
    c = pd.read_csv(ROOT / "data/derived/desk/prices_close.csv", index_col=0, parse_dates=True)
    drop = [x for x in c.columns if x.endswith(".KS") or x.endswith("=X") or x == "QQQ"]
    cal = xnys("2025-06-02", "2026-10-23")
    c = c.drop(columns=drop).reindex(cal)
    Cv = c.to_numpy(float)
    N, K = Cv.shape
    start, thr = pd.Timestamp("2025-09-01"), 0.20
    hk = cal.year.to_numpy() * 10 + (cal.month.to_numpy() > 6)
    rows, pool = [], []
    for k in range(K):
        x = Cv[:, k]
        st = state(x, thr)
        Ldd = st["Ldd"]
        h = np.full(N, np.nan)
        for t in range(N - 1):
            if np.isnan(Ldd[t]):
                continue
            w = x[t + 1:min(t + 20, N - 1) + 1]
            if np.isnan(w).all():
                continue
            if np.nanmin(w) < Ldd[t] - 1e-12:
                h[t] = 0.0
            elif t + 20 <= N - 1 and not np.isnan(w).any():
                h[t] = 1.0
        up = st["up"]
        b = np.where(np.isnan(up), np.nan, np.digitize(up, EDGES))
        okst = st["in_dd"] & ~np.isnan(Ldd) & (cal >= start)
        for t in np.flatnonzero(okst & ~np.isnan(h) & ~np.isnan(up)):
            pool.append((hk[t], b[t], h[t]))
        s = (st["since"] >= RULE_N) & np.where(np.isnan(up), False, up >= RULE_UP)
        prev = -10 ** 9
        for t in np.flatnonzero(s & okst):
            if t - prev <= 20:
                continue
            rows.append((cal[t], c.columns[k], t, h[t], b[t], hk[t], up[t]))
            prev = t
    P = pd.DataFrame(pool, columns=["s", "b", "h"])
    base = P.groupby(["s", "b"]).h.mean()
    S = pd.DataFrame(rows, columns=["date", "series", "pos", "hold20", "b", "s", "up"])
    S["base"] = [base.get((s, b), np.nan) for s, b in zip(S.s, S.b)]
    fam = pd.read_csv(RES / "f3_STK_signals.csv", parse_dates=["date"])
    fam = fam[fam.rule == "since5_up3"]
    a = S.set_index(["date", "series"]).sort_index()
    f = fam.set_index(["date", "series"]).sort_index()
    same = a.index.equals(f.index)
    hm = same and bool(np.allclose(a.hold20.to_numpy(float), f.hold20.to_numpy(float), equal_nan=True))
    bm = same and bool(np.allclose(a.base.to_numpy(float), f.base_hold20.to_numpy(float), equal_nan=True))
    out.write(f"  first signals {len(S)} vs family {len(fam)}; index identical {same}; hold20 identical {hm}; "
              f"matched base identical {bm}\n")
    for nm, lo, hi in (("2025-09~2026", "2025-09-01", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
                       ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
                       ("2025", "2025-09-01", "2025-12-31")):
        g = S[(S.date >= lo) & (S.date <= hi) & S.hold20.notna() & S.base.notna()]
        out.write(f"    {nm:12s} n={len(g):4d} dates {g.date.nunique():3d} hold20 {g.hold20.mean():.3f} matched {g.base.mean():.3f} "
                  f"diff {g.hold20.mean() - g.base.mean():+.3f} up median {g.up.median():.3f}\n")
    out.write("  -> the same rule on 82 stocks is at or below its matched rebound-size baseline and far below 70%.")
    return out.getvalue()


if __name__ == "__main__":
    main()
