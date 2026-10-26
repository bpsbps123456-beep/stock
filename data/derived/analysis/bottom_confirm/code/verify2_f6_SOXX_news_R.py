"""Independent adversarial verification (round 2, fixed censoring) of f6 / news_R / SOXX (BRIEF.md 2026-10-26).

Does NOT import bc_lib. Re-implements from the raw CSVs: the SOXX series, decline low Ldd, dd60, in_dd,
20-session cooldown, the outcomes hold20 / hold40 / near5 / win20 / r20 with the FIXED right-censoring rule
(0 when a break is already observed inside an incomplete forward window, NaN only when nothing has broken yet),
the same-state (year x up-bucket) matched baseline and the block bootstrap; reads the PB-flow note census
(n4h14_signals.csv) directly and re-derives each note's entry session from published_at.

Rule under test (news_R): signal day t = entry session of a family 'R' note (buy-back / short covering /
block purchase); first signals = in_dd days, 2024-01-02..2026-10-23, 20-session cooldown; entry O[t+1].

Run: cd data/derived/analysis/bottom_confirm && python3 -W ignore code/verify2_f6_SOXX_news_R.py
Out: results/verify2_f6_SOXX_news_R.txt
Everything here is post hoc (computed after the registered results were seen).
"""
from __future__ import annotations

import ast
import io
import pathlib
from math import comb

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(__file__).resolve().parents[5]
RES = HERE / "results"
OUT = RES / "verify2_f6_SOXX_news_R.txt"
NOTES_CSV = ROOT / "data/derived/analysis/news_followup/results/n4h14_signals.csv"
N4H14_PY = ROOT / "data/derived/analysis/news_followup/code/n4h14_run.py"

EVAL_START = pd.Timestamp("2024-01-02")
LAST = pd.Timestamp("2026-10-23")
PERIODS = [("2024~2026", "2024-01-02", "2026-10-23"), ("2026", "2026-01-02", "2026-10-23"),
           ("2026H1", "2026-01-02", "2026-05-29"), ("2026H2", "2026-06-01", "2026-10-23"),
           ("2025", "2025-01-02", "2025-12-31"), ("2024", "2024-01-02", "2024-12-31")]
EDGES = [0.03, 0.08, 0.15]          # up buckets 0-3, 3-8, 8-15, 15%+
BNAMES = ["0-3%", "3-8%", "8-15%", "15%+"]
KEYS = ("hold20", "hold40", "near5", "win20", "r20")
NDRAW = 5000
NBOOT = 5000
SEED = 6_2026_1026
N_CELLS = 109

buf = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    buf.write(s + "\n")


# ------------------------------------------------------------------ raw data (own loaders)
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
    try:
        import exchange_calendars as xc
        cal = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range(C.index[0], C.index[-1]))
        cal = cal.tz_localize(None) if cal.tz is not None else cal
        chk["cal_missing"], chk["cal_extra"] = len(cal.difference(C.index)), len(C.index.difference(cal))
    except Exception as e:  # pragma: no cover
        chk["cal_missing"], chk["cal_extra"] = f"n/a ({e})", "n/a"
    return pd.DataFrame({"C": C, "O": O}), chk


def parse_alt_dicts():
    """ALT_A / ALT_B_EXTRA literal dicts from n4h14_run.py via ast (the file is NOT executed)."""
    tree = ast.parse(N4H14_PY.read_text(encoding="utf-8"))
    got = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
            if nm in ("ALT_A", "ALT_B_EXTRA"):
                got[nm] = ast.literal_eval(node.value)
    a = got["ALT_A"]
    b = {**a, **got["ALT_B_EXTRA"]}
    return a, b


def load_notes(sess: pd.DatetimeIndex):
    n = pd.read_csv(NOTES_CSV, encoding="utf-8-sig")
    n["entry"] = pd.to_datetime(n["entry"])
    # re-derive the entry session from published_at (exact: NY time < 09:30 on a session -> same day; else next session;
    # date-only precision -> next session strictly after the date)
    re_entry = []
    for p, prec in zip(n.published_at, n.time_precision):
        if pd.isna(p):
            re_entry.append(pd.NaT)
            continue
        if prec == "exact":
            ts = pd.Timestamp(p).tz_convert("America/New_York")
            d = pd.Timestamp(ts.date())
            k = int(sess.searchsorted(d, side="left"))
            if k < len(sess) and sess[k] == d and (ts.hour, ts.minute) < (9, 30):
                re_entry.append(sess[k])
            else:
                k = int(sess.searchsorted(d, side="right"))
                re_entry.append(sess[k] if k < len(sess) else pd.NaT)
        else:
            d = pd.Timestamp(str(p)[:10])
            k = int(sess.searchsorted(d, side="right"))
            re_entry.append(sess[k] if k < len(sess) else pd.NaT)
    n["re_entry"] = pd.to_datetime(re_entry)
    alt_a, alt_b = parse_alt_dicts()
    n["my_altA"] = [alt_a[g][1] if g in alt_a else f for g, f in zip(n.group_id, n.family)]
    n["my_altB"] = [alt_b[g][1] if g in alt_b else f for g, f in zip(n.group_id, n.family)]
    for g, (f0, _f1, _) in alt_b.items():
        row = n[n.group_id == g]
        if len(row):
            assert row.family.iloc[0] == f0, (g, f0)
    return n, alt_a, alt_b


# ------------------------------------------------------------------ state and outcomes (fixed censoring)
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
    since = np.where(Lpos >= 0, np.arange(N) - Lpos, -1)

    def hold(level: np.ndarray, length: int) -> np.ndarray:
        out = np.full(N, np.nan)
        for t in range(N):
            if t + 1 > N - 1 or np.isnan(level[t]):
                continue
            w = C[t + 1:min(t + length, N - 1) + 1]
            if w.min() < level[t] - 1e-12:
                out[t] = 0.0                       # already broken (even in an incomplete window)
            elif t + length <= N - 1:
                out[t] = 1.0                       # full window, never broken
        return out

    Oe = np.full(N, np.nan)
    Oe[:-1] = O[1:]
    hold20, hold40 = hold(Ldd, 20), hold(Ldd, 40)
    near5 = hold(0.95 * Oe, 40)
    r20 = np.full(N, np.nan)
    r20[:N - 20] = C[20:] / Oe[:N - 20] - 1
    win20 = np.where(np.isnan(r20), np.nan, (r20 > 0).astype(float))
    # break date (first close below Ldd within 20 sessions) for reporting
    brk = np.full(N, -1)
    for t in range(N - 1):
        if np.isnan(Ldd[t]):
            continue
        w = C[t + 1:min(t + 20, N - 1) + 1]
        b = np.flatnonzero(w < Ldd[t] - 1e-12)
        if len(b):
            brk[t] = t + 1 + int(b[0])
    return dict(Ldd=Ldd, Lpos=Lpos, dd60=dd60, in_dd=in_dd, up=up, since=since, hold20=hold20, hold40=hold40,
                near5=near5, r20=r20, win20=win20, brk=brk)


def bucket(up: np.ndarray) -> np.ndarray:
    b = np.digitize(up, EDGES).astype(float)
    b[np.isnan(up)] = np.nan
    return b


def matched_base(idx: pd.DatetimeIndex, fr: dict, key: str, extra_mask: np.ndarray | None = None) -> np.ndarray:
    """Same calendar year x same up bucket: mean outcome of in_dd days (date >= EVAL_START, outcome known)."""
    b = bucket(fr["up"])
    y = idx.year.to_numpy()
    st = fr["in_dd"] & ~np.isnan(fr["up"]) & (idx >= EVAL_START)
    if extra_mask is not None:
        st = st & extra_mask
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


def first_signals(sig: np.ndarray, fr: dict, idx: pd.DatetimeIndex, cool: int = 20,
                  start: pd.Timestamp = EVAL_START, extra: np.ndarray | None = None) -> np.ndarray:
    ok = sig & fr["in_dd"] & ~np.isnan(fr["Ldd"]) & (idx >= start) & (idx <= LAST)
    if extra is not None:
        ok = ok & extra
    out, prev = [], -10 ** 9
    for p in np.flatnonzero(ok):
        if p - prev <= cool:
            continue
        out.append(p)
        prev = p
    return np.asarray(out, int)


def shift(sig: np.ndarray, d: int) -> np.ndarray:
    return sig if d == 0 else np.concatenate([np.zeros(d, bool), sig[:-d]])


def block_ci(v: np.ndarray, lab: np.ndarray, rng, B: int = NBOOT) -> tuple[float, float]:
    u, inv = np.unique(lab, return_inverse=True)
    if len(u) < 2:
        return (np.nan, np.nan)
    sums, cnts = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    pick = rng.integers(0, len(u), size=(B, len(u)))
    m = sums[pick].sum(1) / cnts[pick].sum(1)
    return tuple(np.quantile(m, [0.025, 0.975]))


def period_stats(pos: np.ndarray, fr: dict, idx: pd.DatetimeIndex, bases: dict, boot: bool = True) -> dict:
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
            if k == "hold20" and boot and ok.sum() >= 2:
                lab = pp[ok] // 20
                r["hold20_lo"], r["hold20_hi"] = block_ci(v[ok], lab, rng)
                r["hold20_dlo"], r["hold20_dhi"] = block_ci(v[ok] - bs[ok], lab, rng)
            elif k == "hold20":
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
    return (f"  {label:34s} main {f('2024~2026'):10s} base {m['hold20_base']:.3f} dlo {m.get('hold20_dlo', np.nan):+.3f} | "
            f"2026 {f('2026'):10s} H1 {f('2026H1'):10s} H2 {f('2026H2'):10s} 2025 {f('2025'):10s} 2024 {f('2024'):10s} "
            f"upmed {m['up_med']:.3f} L1={l1(st)} L2={l2(st)}")


# ------------------------------------------------------------------ null sampler
def null_draws(cands: list[np.ndarray], hold: np.ndarray, cool: int, rng, ndraw: int = NDRAW,
               extra: dict | None = None):
    """Each draw picks one position from each candidate set (order shuffled), pairwise spacing > cool.
    Returns array of mean hold over the picks (and of each `extra` array if given)."""
    res, res_x, tries = [], {k: [] for k in (extra or {})}, 0
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
        ch = np.asarray(chosen)
        res.append(float(hold[ch].mean()))
        for k, arr in (extra or {}).items():
            res_x[k].append(float(np.nanmean(arr[ch])))
    return np.asarray(res), {k: np.asarray(v) for k, v in res_x.items()}, tries


# ------------------------------------------------------------------ main
def main():
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 46)
    df, chk = load_soxx()
    idx = df.index
    C, O = df.C.to_numpy(float), df.O.to_numpy(float)
    N = len(C)
    pos_of = {d: i for i, d in enumerate(idx)}
    notes, alt_a, alt_b = load_notes(idx)
    say("=" * 128)
    say("Independent verification round 2 (fixed right-censoring): family f6, rule news_R, kind SOXX (bc_lib NOT imported)")
    say("=" * 128)
    say(f"Data: SOXX close {idx[0].date()}..{idx[-1].date()} N={N}; open from {df.O.first_valid_index().date()}; "
        f"overlap rows {chk['overlap_n']} max|diff|={chk['overlap_maxdiff']:.1e}; XNYS missing={chk['cal_missing']} "
        f"extra={chk['cal_extra']}")
    say(f"Notes: {len(notes)} rows from {NOTES_CSV.relative_to(ROOT)}; family counts {notes.family.value_counts().to_dict()}")

    fr = frames(C, O, 0.10)
    bases = {k: matched_base(idx, fr, k) for k in KEYS}

    def note_sig(mask: pd.Series, col: str = "entry", lag: int = 0) -> np.ndarray:
        s = np.zeros(N, bool)
        for e in notes.loc[mask, col].dropna():
            p = pos_of[pd.Timestamp(e)] + lag
            if 0 <= p < N:
                s[p] = True
        return s

    isR = notes.family == "R"
    sigR = note_sig(isR)
    pos = first_signals(sigR, fr, idx)

    # ============================================================ [1] independent match
    say("\n[1] Independent re-implementation vs family files (results/f6_SOXX_signals.csv, f6_SOXX_summary.csv)")
    bad_entry = notes[(notes.re_entry != notes.entry) & ~(notes.re_entry.isna() & notes.entry.isna())]
    say(f"  entry session re-derived from published_at/time_precision: {len(notes) - len(bad_entry)}/{len(notes)} identical"
        + ("" if bad_entry.empty else f"; differ: {bad_entry[['group_id', 'published_at', 'entry', 're_entry']].to_string()}"))
    aa = (notes.my_altA == notes.fam_altA).all()
    bb = (notes.my_altB == notes.fam_altB).all()
    say(f"  ALT_A ({len(alt_a)} groups) / ALT_B ({len(alt_b)} groups) parsed from n4h14_run.py match CSV fam_altA/fam_altB: {aa}/{bb}")
    fam = pd.read_csv(RES / "f6_SOXX_signals.csv", parse_dates=["date"])
    fam = fam[fam.rule == "news_R"].reset_index(drop=True)
    fsum = pd.read_csv(RES / "f6_SOXX_summary.csv")
    fsum = fsum[fsum.rule == "news_R"].set_index("period")
    mine = pd.DataFrame(dict(date=idx[pos], pos=pos, Ldd_date=idx[fr["Lpos"][pos]], Ldd=fr["Ldd"][pos], C=C[pos],
                             up=fr["up"][pos], dd60=fr["dd60"][pos], since=fr["since"][pos],
                             hold20=fr["hold20"][pos], base_hold20=bases["hold20"][pos], hold40=fr["hold40"][pos],
                             base_hold40=bases["hold40"][pos], near5=fr["near5"][pos], base_near5=bases["near5"][pos],
                             win20=fr["win20"][pos], base_win20=bases["win20"][pos], r20=fr["r20"][pos],
                             base_r20=bases["r20"][pos]))
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
    say("  period      mine hold20(n) fam hold20(n)  mine base fam base | mine dlo  fam dlo | mine near5/win20/r20/hold40")
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
    say(f"  => independent match: {indep_match}  (bootstrap bounds are resampling output with a different RNG; "
        f"compared for sign only)")
    m0 = st0["2024~2026"]
    pool_m = fr["in_dd"] & ~np.isnan(fr["Ldd"]) & ~np.isnan(fr["hold20"]) & (idx >= EVAL_START) & (idx <= LAST)
    say(f"  main: n={m0['n']} hold20={m0['hold20']:.4f} matched {m0['hold20_base']:.4f}; dlo {m0['hold20_dlo']:+.4f}; "
        f"near5 {m0['near5']:.3f} (matched {m0['near5_base']:.3f}); win20 {m0['win20']:.3f} (matched {m0['win20_base']:.3f}); "
        f"r20 {m0['r20']:+.4f} (matched {m0['r20_base']:+.4f}); up median {m0['up_med']:.4f}")
    say(f"  unconditional in_dd hold20 2024~2026 = {fr['hold20'][pool_m].mean():.4f} (resolved days {int(pool_m.sum())})")
    k_, n_ = int(round(m0["hold20"] * m0["hold20_n"])), m0["hold20_n"]
    for lab, p0 in (("unconditional in_dd", fr["hold20"][pool_m].mean()), ("matched mean", m0["hold20_base"])):
        pb = sum(comb(n_, j) * p0 ** j * (1 - p0) ** (n_ - j) for j in range(k_, n_ + 1))
        say(f"  binomial P(X >= {k_}/{n_} | p={p0:.4f}, {lab}) = {pb:.4f}")
    # Poisson-binomial with each signal's own matched probability
    probs = bases["hold20"][pos]
    dist = np.array([1.0])
    for q in probs:
        dist = np.convolve(dist, [1 - q, q])
    say(f"  Poisson-binomial P(X >= {k_} | per-signal matched p) = {dist[k_:].sum():.4f}")
    diff = fr["hold20"][pos] - bases["hold20"][pos]
    say("  per-signal hold20 - matched: " + ", ".join(f"{idx[p].date()} {d:+.3f}" for p, d in zip(pos, diff)))

    # ============================================================ [a] signal list
    say("\n[a] Every first signal: notes on the signal day, state, outcomes")
    rows = []
    for p in pos:
        d = idx[p]
        nn = notes[notes.entry == d]
        r_n = nn[nn.family == "R"]
        rows.append(dict(
            date=str(d.date()), banks="/".join(r_n.bank), groups="/".join(r_n.group_id),
            scope=" | ".join(str(s)[:40] for s in r_n.scope), verif="/".join(v.replace("_", "")[:12] for v in r_n.verification),
            altA="/".join(r_n.fam_altA), altB="/".join(r_n.fam_altB),
            other_notes="/".join(f"{f}:{b}" for f, b in zip(nn[nn.family != "R"].family, nn[nn.family != "R"].bank)) or "-",
            up=round(fr["up"][p], 4), dd60=round(fr["dd60"][p], 4), within5_of_high=bool(fr["dd60"][p] > -0.05),
            at_new_high=bool(fr["dd60"][p] >= -1e-12), Ldd_date=str(idx[fr["Lpos"][p]].date()), since_low=int(fr["since"][p]),
            hold20=fr["hold20"][p], base=round(bases["hold20"][p], 3), broke=(str(idx[fr["brk"][p]].date()) if fr["brk"][p] >= 0 else "-"),
            hold40=fr["hold40"][p], near5=fr["near5"][p], r20=round(fr["r20"][p], 4) if not np.isnan(fr["r20"][p]) else np.nan))
    sl = pd.DataFrame(rows)
    say(sl[["date", "banks", "groups", "verif", "altA", "altB", "other_notes", "up", "dd60", "within5_of_high",
            "at_new_high", "Ldd_date", "since_low", "hold20", "base", "broke", "hold40", "near5", "r20"]].to_string(index=False))
    say("  scopes: " + "; ".join(f"{r['date']}: {r['scope']}" for r in rows))
    n_hl = int(sum(("headline" in v) for v in notes.loc[notes.entry.isin(idx[pos]) & isR, "verification"]))
    say(f"  R notes on first-signal days: {int((notes.entry.isin(idx[pos]) & isR).sum())}; headline_summary_only among them: {n_hl}")
    say(f"  signals within 5% of the 60d high: {int(sl.within5_of_high.sum())}/{len(sl)}; at a new 60d high (Ldd reset, up=0): "
        f"{int(sl.at_new_high.sum())}")

    # ============================================================ [2] look-ahead audit
    say("\n[2] Look-ahead / timing / data audit")
    trunc_ok = True
    for p in pos:
        f_tr = frames(C[:p + 1], O[:p + 1], 0.10)
        if not (f_tr["in_dd"][p] and abs(f_tr["Ldd"][p] - fr["Ldd"][p]) < 1e-12 and abs(f_tr["up"][p] - fr["up"][p]) < 1e-12):
            trunc_ok = False
            say(f"  truncation mismatch at {idx[p].date()}")
    say(f"  truncation test (state recomputed with prices cut at t) on the {len(pos)} first signals: "
        f"{'identical' if trunc_ok else 'DIFFERENCES'}")
    pa = pd.to_datetime([str(x)[:10] for x in notes.published_at])
    nd = pd.to_datetime(notes.note_date)
    say(f"  entry > published date for all date-only notes: "
        f"{bool((notes.entry[notes.time_precision != 'exact'] > pa[notes.time_precision != 'exact']).all())}; "
        f"note_date <= published date: {bool((nd <= pa).all())}; exact-time notes: {int((notes.time_precision == 'exact').sum())}")
    rr = notes[isR & notes.entry.isin(idx[pos])]
    say("  first-signal R notes: published_at / precision / note_date -> entry (= signal day t; trade at O[t+1]):")
    for _, r in rr.iterrows():
        say(f"    {r.group_id} {r.bank:4s} pub {r.published_at} ({r.time_precision}) note {r.note_date} -> entry {r.entry.date()}")
    say("  f6_news_flow.py: note_days() marks entry sessions only; news_R uses no price, flow or market input. bc_lib")
    say("  first_signals uses in_dd/Ldd at t; outcomes start at C[t+1], entry O[t+1]. So the trade is one full session after")
    say("  the note became public (one session conservative). No forward window ends after t in the signal.")
    say("  Non-timing risks that the code cannot audit: the census was collected and LLM-coded in 2026-10 (family labels")
    say("  assigned with hindsight available), 'date_feed' dates come from aggregator feeds; coverage is uneven by bank/year.")
    r1 = np.diff(np.log(C))
    big = np.flatnonzero(np.abs(r1) > 0.08) + 1
    say(f"  SOXX |daily log return| > 8%: {[f'{idx[p].date()} {r1[p - 1]:+.3f}' for p in big]}")
    gap = np.abs(np.log(O[1:] / C[:-1]))
    bg = np.flatnonzero(gap > 0.06) + 1
    say(f"  SOXX |open/prev close| gaps > 6%: {[f'{idx[p].date()} {np.log(O[p] / C[p - 1]):+.3f}' for p in bg if not np.isnan(gap[p - 1])]}")
    say(f"  O[t+1] present for every first signal: {bool(np.isfinite(O[pos + 1]).all())}")
    say("  Known data error elsewhere in the study (^VIX 2026-02-06 = ^VIX3M) does not enter news_R.")

    # ============================================================ [3] robustness
    say("\n[3] Robustness / neighbours (hold20 resolved n per period; base = matched; dlo = block-bootstrap lower bound of "
        "hold20 - matched; L1/L2 = registered SOXX criteria)")
    rob = {}
    frc = {0.10: (fr, bases)}

    def run(label, sig=None, cool=20, thr=0.10, delay=0, extra=None, core=True, show_fail=True):
        if thr not in frc:
            f = frames(C, O, thr)
            frc[thr] = (f, {k: matched_base(idx, f, k) for k in KEYS})
        f, bs = frc[thr]
        s = sigR if sig is None else sig
        ps = first_signals(shift(s, delay), f, idx, cool, extra=extra)
        st = period_stats(ps, f, idx, bs)
        rob[label] = (st, core, ps)
        say(fmt_row(label, st))
        bad = [p for p in ps if f["hold20"][p] == 0]
        if bad and show_fail:
            say("      failed hold20: " + ", ".join(f"{idx[p].date()} (up {f['up'][p]:.3f})" for p in bad))
        return st

    run("registered news_R")
    say("  -- timing")
    run("delay +1 session", delay=1)
    run("delay +2 sessions", delay=2)
    for k in (2, 4):
        w = sigR.copy()
        for j in range(1, k + 1):
            w |= shift(sigR, j)
        run(f"R note in [t-{k}, t] (event window)", sig=w, core=False)
    say("  -- cooldown / in_dd threshold")
    run("cooldown 10", cool=10)
    run("cooldown 30", cool=30)
    run("in_dd thr 8%", thr=0.08)
    run("in_dd thr 12%", thr=0.12)
    say("  -- coding (b): alternate family codings and verification level")
    run("fam_altA == R", sig=note_sig(notes.fam_altA == "R"))
    run("fam_altB == R", sig=note_sig(notes.fam_altB == "R"))
    run("R excl. headline_summary_only", sig=note_sig(isR & (notes.verification != "headline_summary_only")))
    run("altA R excl. headline_only", sig=note_sig((notes.fam_altA == "R") & (notes.verification != "headline_summary_only")), core=False)
    run("R and scope_ok (SOXX-relevant scope)", sig=note_sig(isR & notes.scope_ok.astype(bool)), core=False)
    run("R, bank GS (posthoc twin)", sig=note_sig(isR & (notes.bank == "GS")), core=False)
    run("R, bank != GS", sig=note_sig(isR & (notes.bank != "GS")), core=False)
    run("R, flow_direction net_buy", sig=note_sig(isR & (notes.flow_direction == "net_buy")), core=False)
    say("  -- placebo note families (same machinery, other labels)")
    for fm in ("N", "D", "T", "S"):
        run(f"placebo family {fm}", sig=note_sig(notes.family == fm), core=False, show_fail=False)
    run("placebo any note", sig=note_sig(notes.family.notna()), core=False, show_fail=False)
    run("placebo any non-R note", sig=note_sig(notes.family != "R"), core=False, show_fail=False)
    say("  -- (c) excluding days within 5% of the 60d high (dd60 > -5%)")
    far = fr["dd60"] <= -0.05
    run("R, dd60 <= -5% before cooldown", extra=far)
    post = pos[fr["dd60"][pos] <= -0.05]
    bases_far = {k: matched_base(idx, fr, k, extra_mask=far) for k in KEYS}
    st_post = period_stats(post, fr, idx, bases)
    st_post_far = period_stats(post, fr, idx, bases_far)
    st_pre_far = period_stats(first_signals(sigR, fr, idx, extra=far), fr, idx, bases_far)
    say(fmt_row("registered signals, near-high dropped", st_post))
    say(fmt_row("  same, baseline also excl. near-high", st_post_far))
    say(fmt_row("dd60<=-5% pre-cooldown, base excl.", st_pre_far))
    core = {k: v for k, v in rob.items() if v[1]}
    say(f"  L1 in {sum(l1(v[0]) for v in core.values())}/{len(core)} core variants; L2 in {sum(l2(v[0]) for v in core.values())}/{len(core)}; "
        f"L2 fails: {[k for k, v in core.items() if not l2(v[0])]}")

    # ============================================================ [4] same-state null
    say(f"\n[4] Same-state null: random in_dd days in 2024~2026 (resolved hold20), pairwise spacing > 20 sessions, {NDRAW} draws")
    b_all = bucket(fr["up"])
    yr = idx.year.to_numpy()
    pool_pos = np.flatnonzero(pool_m)
    obs = float(fr["hold20"][pos].mean())
    lab_yb = [(int(yr[p]), BNAMES[int(b_all[p])]) for p in pos]
    say(f"  observed hold20 = {obs:.4f} (n={len(pos)}); year x bucket labels = {lab_yb}")
    tab = pd.DataFrame(dict(y=yr[pool_pos], b=b_all[pool_pos].astype(int), h=fr["hold20"][pool_pos]))
    say("  pool hold20 by year x bucket (days): " + "; ".join(
        f"{y}/{BNAMES[b]} {g.h.mean():.3f}({len(g)})" for (y, b), g in tab.groupby(["y", "b"])))
    rng = np.random.default_rng(SEED)

    def cands_by(keyfun):
        return [pool_pos[np.array([keyfun(q) == keyfun(p) for q in pool_pos])] for p in pos]

    c_yb = cands_by(lambda q: (yr[q], int(b_all[q])))
    d_yb, _, tr = null_draws(c_yb, fr["hold20"], 20, rng)
    p_yb = float((d_yb >= obs - 1e-12).mean())
    say(f"  [N1] year x up-bucket (registered matching): mean {d_yb.mean():.4f} sd {d_yb.std():.4f}  p = P(null >= obs) = {p_yb:.4f}"
        f"  (attempts {tr})")
    # finer: same year, |up - up_sig| <= 2pp
    c_f = [pool_pos[(yr[pool_pos] == yr[p]) & (np.abs(fr["up"][pool_pos] - fr["up"][p]) <= 0.02 + 1e-12)] for p in pos]
    say("  fine pool sizes (same year, up +-2pp): " + ", ".join(f"{idx[p].date()}:{len(c)} (h {fr['hold20'][c].mean():.2f})" for p, c in zip(pos, c_f)))
    d_f, _, tr = null_draws(c_f, fr["hold20"], 20, rng)
    p_f = float((d_f >= obs - 1e-12).mean())
    say(f"  [N2] same year, up within +-2pp: mean {d_f.mean():.4f} sd {d_f.std():.4f}  p = {p_f:.4f}  (attempts {tr})")
    # (f) also match days since the low
    sb_edges = [1, 6, 16]
    sbin = np.digitize(fr["since"], sb_edges)
    c_us = [pool_pos[(b_all[pool_pos] == b_all[p]) & (sbin[pool_pos] == sbin[p])] for p in pos]
    d_us, _, tr = null_draws(c_us, fr["hold20"], 20, rng)
    p_us = float((d_us >= obs - 1e-12).mean())
    say(f"  [N3] up-bucket x since-low bucket (0 / 1-5 / 6-15 / 16+ sessions), years pooled: mean {d_us.mean():.4f} "
        f"p = {p_us:.4f}  pool sizes {[len(c) for c in c_us]}")
    c_yus = [pool_pos[(yr[pool_pos] == yr[p]) & (b_all[pool_pos] == b_all[p]) & (sbin[pool_pos] == sbin[p])] for p in pos]
    if min(len(c) for c in c_yus) >= 2:
        d_yus, _, tr = null_draws(c_yus, fr["hold20"], 20, rng)
        p_yus = float((d_yus >= obs - 1e-12).mean())
        say(f"  [N4] year x up-bucket x since-low bucket: mean {d_yus.mean():.4f} p = {p_yus:.4f} pool sizes {[len(c) for c in c_yus]}")
    else:
        p_yus = np.nan
        say(f"  [N4] year x up x since: some cells < 2 days ({[len(c) for c in c_yus]}), skipped")
    # 2026-only nulls
    m26 = idx[pos] >= pd.Timestamp("2026-01-02")
    obs26 = float(fr["hold20"][pos[m26]].mean())
    d26, _, _ = null_draws([c for c, k in zip(c_yb, m26) if k], fr["hold20"], 20, rng)
    d26f, _, _ = null_draws([c for c, k in zip(c_f, m26) if k], fr["hold20"], 20, rng)
    p26, p26f = float((d26 >= obs26 - 1e-12).mean()), float((d26f >= obs26 - 1e-12).mean())
    say(f"  [N5] 2026 only (n={int(m26.sum())}, obs {obs26:.3f}): year x bucket null mean {d26.mean():.3f} p = {p26:.4f}; "
        f"+-2pp null mean {d26f.mean():.3f} p = {p26f:.4f}")
    # label permutation among notes (keeps note days and their clustering; tests whether R notes differ from other notes)
    say("  [N6] family-label permutation among the 245 notes (82 R labels reshuffled), statistic = mean(hold20 - matched)")
    ent_pos = np.array([pos_of[e] for e in notes.entry])
    obs_diff = float(np.nanmean(diff))
    perm_d, perm_h, perm_n = [], [], []
    labs = isR.to_numpy().copy()
    rngp = np.random.default_rng(SEED + 7)
    for _ in range(NDRAW):
        rngp.shuffle(labs)
        s = np.zeros(N, bool)
        s[ent_pos[labs]] = True
        ps = first_signals(s, fr, idx)
        ps = ps[~np.isnan(fr["hold20"][ps])]
        if len(ps) == 0:
            continue
        perm_d.append(float(np.mean(fr["hold20"][ps] - bases["hold20"][ps])))
        perm_h.append(float(fr["hold20"][ps].mean()))
        perm_n.append(len(ps))
    perm_d, perm_h = np.asarray(perm_d), np.asarray(perm_h)
    p_perm_d = float((perm_d >= obs_diff - 1e-12).mean())
    p_perm_h = float((perm_h >= obs - 1e-12).mean())
    say(f"    observed mean diff {obs_diff:+.4f}; perm mean diff {perm_d.mean():+.4f} sd {perm_d.std():.4f}; p(diff) = {p_perm_d:.4f}; "
        f"p(hold20 >= {obs:.3f}) = {p_perm_h:.4f}; perm n first signals median {int(np.median(perm_n))}")
    # circular shift of R dates (keeps R-note clustering)
    say("  [N7] circular shift of all R entry sessions by k (15 <= |k|, wrap inside the note span), statistic as N6")
    rpos = np.array([pos_of[e] for e in notes.entry[isR]])
    lo_, hi_ = int(ent_pos.min()), N - 1
    span = hi_ - lo_ + 1
    cs_d, cs_h = [], []
    for k in range(15, span - 15):
        s = np.zeros(N, bool)
        s[lo_ + (rpos - lo_ + k) % span] = True
        ps = first_signals(s, fr, idx)
        ps = ps[~np.isnan(fr["hold20"][ps])]
        if len(ps) == 0:
            continue
        cs_d.append(float(np.mean(fr["hold20"][ps] - bases["hold20"][ps])))
        cs_h.append(float(fr["hold20"][ps].mean()))
    cs_d, cs_h = np.asarray(cs_d), np.asarray(cs_h)
    p_cs_d, p_cs_h = float((cs_d >= obs_diff - 1e-12).mean()), float((cs_h >= obs - 1e-12).mean())
    say(f"    shifts {len(cs_d)}; mean diff {cs_d.mean():+.4f}; p(diff) = {p_cs_d:.4f}; p(hold20) = {p_cs_h:.4f}")
    # N9: no cooldown -- every R-note in_dd day (resolved) in 2024~2026; same circular shift and permutation
    def all_days_stat(s: np.ndarray, lo_d="2024-01-02", hi_d="2026-10-23"):
        q = np.flatnonzero(s & pool_m & (idx >= pd.Timestamp(lo_d)) & (idx <= pd.Timestamp(hi_d)))
        return (float(np.mean(fr["hold20"][q] - bases["hold20"][q])), len(q)) if len(q) else (np.nan, 0)
    obs_all, n_all = all_days_stat(sigR)
    sh_all = []
    for k in range(15, span - 15):
        s = np.zeros(N, bool)
        s[lo_ + (rpos - lo_ + k) % span] = True
        sh_all.append(all_days_stat(s)[0])
    sh_all = np.asarray(sh_all)
    pm_all = []
    labs2 = isR.to_numpy().copy()
    for _ in range(NDRAW):
        rngp.shuffle(labs2)
        s = np.zeros(N, bool)
        s[ent_pos[labs2]] = True
        pm_all.append(all_days_stat(s)[0])
    pm_all = np.asarray(pm_all)
    q_all = np.flatnonzero(sigR & pool_m)
    lo9, hi9 = block_ci(fr["hold20"][q_all] - bases["hold20"][q_all], q_all // 20, np.random.default_rng(SEED))
    say(f"  [N9] no cooldown (all {n_all} R-note in_dd days, resolved): mean(hold20 - matched) {obs_all:+.4f}; block-bootstrap 95% "
        f"[{lo9:+.3f}, {hi9:+.3f}]; circular-shift p = {float(np.nanmean(sh_all >= obs_all - 1e-12)):.4f}; "
        f"label-permutation p = {float(np.nanmean(pm_all >= obs_all - 1e-12)):.4f}")
    p9s, p9p = float(np.nanmean(sh_all >= obs_all - 1e-12)), float(np.nanmean(pm_all >= obs_all - 1e-12))
    for nm, a_, b_ in (("2026", "2026-01-02", "2026-10-23"), ("2026H2", "2026-06-01", "2026-10-23"), ("2024~2025", "2024-01-02", "2025-12-31")):
        v_, n_v = all_days_stat(sigR, a_, b_)
        say(f"    {nm:9s} all R-note in_dd days n={n_v:2d} mean(hold20 - matched) {v_:+.4f}")
    say("  NOTE: 6 R-note in_dd days failed hold20 (all with up < 6%). Five of them (2024-10-21, 2025-12-16, 2026-06-16,")
    say("  2026-07-09, 2026-10-15) fell 5-20 sessions after a kept first signal that held, so the cooldown suppressed them")
    say("  (2026-07-09 by exactly one session: gap 20 <= 20); the sixth, 2026-07-16, is the one failed first signal.")
    say("  The first-signal excess (+0.228) is about three times the all-days excess.")
    # rejection-sampling check of N1 (exactly uniform over spaced configurations)
    rngr = np.random.default_rng(SEED + 11)
    acc, tries_r = [], 0
    while len(acc) < 2000 and tries_r < 2_000_000:
        tries_r += 1
        ch = np.array([int(rngr.choice(c)) for c in c_yb])
        dd_ = np.abs(ch[:, None] - ch[None, :])
        np.fill_diagonal(dd_, 10 ** 6)
        if dd_.min() > 20:
            acc.append(float(fr["hold20"][ch].mean()))
    acc = np.asarray(acc)
    p_rej = float((acc >= obs - 1e-12).mean()) if len(acc) else np.nan
    say(f"  [N1r] same as N1 by rejection sampling (uniform over spaced sets): accepted {len(acc)}/{tries_r}, "
        f"mean {acc.mean():.4f} sd {acc.std():.4f} p = {p_rej:.4f}")
    rng_i = np.random.default_rng(SEED + 12)
    ind = np.array([[float(fr["hold20"][int(rng_i.choice(c))]) for c in c_yb] for _ in range(NDRAW)]).mean(1)
    say(f"  [N1i] same pools, independent draws (no spacing): mean {ind.mean():.4f} sd {ind.std():.4f} "
        f"p = {float((ind >= obs - 1e-12).mean()):.4f}  (spacing lowers the null variance because hold20 is clustered by episode)")
    # null for the near-high-excluded variant (pool also excludes dd60 > -5%)
    pos_far = pos[fr["dd60"][pos] <= -0.05]
    pool_far = pool_pos[fr["dd60"][pool_pos] <= -0.05]
    c_far = [pool_far[(yr[pool_far] == yr[p]) & (b_all[pool_far] == b_all[p])] for p in pos_far]
    d_far, _, _ = null_draws(c_far, fr["hold20"], 20, rng)
    obs_far = float(fr["hold20"][pos_far].mean())
    say(f"  [N8] near-high signals dropped (n={len(pos_far)}, obs {obs_far:.3f}), pool also dd60 <= -5%: mean {d_far.mean():.4f} "
        f"p = {float((d_far >= obs_far - 1e-12).mean()):.4f}")
    # informative signals only: matched base < 0.95 (the rest cannot lose vs their bucket)
    inf = pos[bases["hold20"][pos] < 0.95]
    say(f"  informative signals (matched base < 0.95): {len(inf)}; held {int(fr['hold20'][inf].sum())}/{len(inf)}, expected "
        f"{bases['hold20'][inf].sum():.2f}; dates {[str(idx[p].date()) for p in inf]}")
    inf26 = inf[idx[inf] >= pd.Timestamp("2026-01-02")]
    say(f"    of which 2026: {int(fr['hold20'][inf26].sum())}/{len(inf26)} held at 20d, {int(np.nansum(fr['hold40'][inf26]))}/"
        f"{int((~np.isnan(fr['hold40'][inf26])).sum())} at 40d; 2026H2 informative: "
        f"{[(str(idx[p].date()), fr['hold20'][p], fr['hold40'][p]) for p in inf26 if idx[p] >= pd.Timestamp('2026-06-01')]}")
    # coverage sensitivity: leave-one-signal-note-out and random 30% deletion of R notes
    say("  coverage sensitivity (post hoc): drop the R note(s) of one first-signal day, recompute (next R note may take over)")
    rid = notes.index[isR].to_numpy()
    loo = []
    for p in pos:
        keep = isR & (notes.entry != idx[p])
        ps = first_signals(note_sig(keep), fr, idx)
        st = period_stats(ps, fr, idx, bases)
        loo.append((str(idx[p].date()), st["2024~2026"]["hold20"], st["2024~2026"]["hold20_n"], st["2024~2026"]["hold20_base"],
                    st["2024~2026"]["hold20_dlo"], l1(st), l2(st)))
    for r in loo:
        say(f"    drop {r[0]}: main {r[1]:.3f}({r[2]}) base {r[3]:.3f} dlo {r[4]:+.3f} L1={r[5]} L2={r[6]}")
    rngc = np.random.default_rng(SEED + 13)
    cov = []
    for _ in range(500):
        drop = rngc.random(len(rid)) < 0.30
        keep = pd.Series(False, index=notes.index)
        keep.loc[rid[~drop]] = True
        ps = first_signals(note_sig(keep), fr, idx)
        st = period_stats(ps, fr, idx, bases, boot=True)
        cov.append((st["2024~2026"]["hold20"], st["2024~2026"]["hold20"] - st["2024~2026"]["hold20_base"], l1(st), l2(st),
                    st["2026"]["hold20"]))
    cov = np.array(cov, float)
    say(f"  random 30% deletion of R notes (500 runs): main hold20 median {np.nanmedian(cov[:, 0]):.3f} "
        f"[5%,95%] [{np.nanquantile(cov[:, 0], .05):.3f}, {np.nanquantile(cov[:, 0], .95):.3f}]; diff vs matched median "
        f"{np.nanmedian(cov[:, 1]):+.3f}; L1 share {np.nanmean(cov[:, 2]):.2f}; L2 share {np.nanmean(cov[:, 3]):.2f}; "
        f"2026 hold20 median {np.nanmedian(cov[:, 4]):.3f}")
    say(f"  multiple testing: {N_CELLS} cells evaluated (BRIEF grid + GS twin). Bonferroni alpha 0.05/{N_CELLS} = {0.05 / N_CELLS:.5f}. "
        f"Sidak family-wise chance of >=1 cell with p <= {p_yb:.4f}: 1-(1-p)^{N_CELLS} = {1 - (1 - p_yb) ** N_CELLS:.3f}; "
        f"with p <= {p_f:.4f}: {1 - (1 - p_f) ** N_CELLS:.3f}")

    # ============================================================ [5] mechanical comparison
    say("\n[5] Mechanical comparison: plain up >= candidate median up")
    upmed = float(np.median(fr["up"][pos]))
    for x in (upmed, 0.05, 0.03):
        run(f"plain up >= {x:.4f}", sig=np.where(np.isnan(fr["up"]), False, fr["up"] >= x), core=False, show_fail=False)
    say("  R signals by up bucket (hold20 held/n, matched mean):")
    for b in range(4):
        m = b_all[pos] == b
        if m.any():
            say(f"    {BNAMES[b]:6s}: {int(np.nansum(fr['hold20'][pos[m]]))}/{int(m.sum())}  matched {np.mean(bases['hold20'][pos[m]]):.3f}  "
                f"dates {[str(idx[p].date()) for p in pos[m]]}")
    allR = np.flatnonzero(sigR & pool_m)
    say(f"  all R-note days in in_dd 2024~2026 (no cooldown, resolved): n={len(allR)} hold20 {fr['hold20'][allR].mean():.3f} "
        f"matched {np.nanmean(bases['hold20'][allR]):.3f} diff {np.nanmean(fr['hold20'][allR] - bases['hold20'][allR]):+.3f}")
    for nm, a, b in PERIODS[:4]:
        m = (idx[allR] >= a) & (idx[allR] <= b)
        if m.any():
            say(f"    {nm:10s} n={int(m.sum()):3d} hold20 {fr['hold20'][allR[m]].mean():.3f} matched {np.nanmean(bases['hold20'][allR[m]]):.3f}")
    say("  every R-note in_dd day 2024~2026 (F = kept as first signal; '-' = suppressed by the 20-session cooldown):")
    allRi_ = np.flatnonzero(sigR & fr["in_dd"] & ~np.isnan(fr["Ldd"]) & (idx >= EVAL_START))
    pset = set(pos.tolist())
    for p in allRi_:
        gid = "/".join(f"{g}:{b}" for g, b in zip(notes.group_id[isR & (notes.entry == idx[p])], notes.bank[isR & (notes.entry == idx[p])]))
        say(f"    {idx[p].date()} {'F' if p in pset else '-'} {gid:22s} up {fr['up'][p]:.3f} dd60 {fr['dd60'][p]:+.3f} "
            f"hold20 {fr['hold20'][p]} base {bases['hold20'][p]:.3f} hold40 {fr['hold40'][p]}")

    # ============================================================ [d] 2026 detail
    say("\n[d] 2026 and halves vs matched")
    for p_ in ("2026", "2026H1", "2026H2", "2025", "2024"):
        r = st0[p_]
        say(f"  {p_:7s} n={r['n']} hold20 {r['hold20']:.3f} matched {r['hold20_base']:.3f} diff {r['hold20'] - r['hold20_base']:+.3f} "
            f"dlo {r['hold20_dlo']:+.3f} | near5 {r['near5']:.3f}/{r['near5_base']:.3f} win20 {r['win20']:.3f}/{r['win20_base']:.3f} "
            f"r20 {r['r20']:+.3f}/{r['r20_base']:+.3f} hold40 {r['hold40']:.3f}/{r['hold40_base']:.3f}")
    say("  2026H1 has one signal (2026-04-14) taken at a new 60-day high (Ldd = that day's close, up = 0); it is a breakout")
    say("  hold, not a bottom confirmation. Without it 2026 is 3/4 vs matched "
        f"{np.mean(bases['hold20'][pos[m26 & (fr['dd60'][pos] < -1e-12)]]):.3f}.")

    # ============================================================ [e] STK replication
    say("\n[e] Stocks universe (results/f6_STK_summary.csv / verdict, news_R; registered family output)")
    ss = pd.read_csv(RES / "f6_STK_summary.csv")
    ss = ss[ss.rule == "news_R"]
    for _, r in ss.iterrows():
        say(f"  {r.period:12s} n={int(r.n):4d} clusters {int(r.clusters):2d} hold20 {r.hold20:.3f} matched {r.hold20_base:.3f} "
            f"dlo {r.hold20_dlo:+.3f} near5 {r.near5:.3f}/{r.near5_base:.3f} win20 {r.win20:.3f}/{r.win20_base:.3f} up_med {r.up_median:.3f}")
    sv = pd.read_csv(RES / "f6_STK_verdict.csv")
    sv = sv[sv.rule == "news_R"].iloc[0]
    say(f"  STK verdict news_R: L1={sv.L1} L2={sv.L2}  -> no replication (2026 hold20 ~= matched; 2026H2 below matched)")
    stk = stk_check(notes)
    say(stk)

    # ============================================================ [f] lateness
    say("\n[f] Are R notes published after the rebound already started?")
    pool_all = np.flatnonzero(fr["in_dd"] & ~np.isnan(fr["Ldd"]) & (idx >= EVAL_START))
    up_pool, s_pool = fr["up"][pool_all], fr["since"][pool_all]
    say(f"  all in_dd days 2024~2026 (n={len(pool_all)}): up median {np.median(up_pool):.4f}, IQR "
        f"[{np.quantile(up_pool, .25):.3f}, {np.quantile(up_pool, .75):.3f}]; sessions since low median {np.median(s_pool):.0f}, "
        f"IQR [{np.quantile(s_pool, .25):.0f}, {np.quantile(s_pool, .75):.0f}]")
    allRi = np.flatnonzero(sigR & fr["in_dd"] & ~np.isnan(fr["Ldd"]) & (idx >= EVAL_START))
    say(f"  all R-note days in in_dd (n={len(allRi)}): up median {np.median(fr['up'][allRi]):.4f}, since median "
        f"{np.median(fr['since'][allRi]):.0f}; share up>=3% {np.mean(fr['up'][allRi] >= 0.03):.2f} vs pool {np.mean(up_pool >= 0.03):.2f}")
    say(f"  first signals (n={len(pos)}): up median {np.median(fr['up'][pos]):.4f}, since median {np.median(fr['since'][pos]):.0f}; "
        f"up percentile in pool: {[int(round((up_pool < u).mean() * 100)) for u in fr['up'][pos]]}; "
        f"since percentile: {[int(round((s_pool < s).mean() * 100)) for s in fr['since'][pos]]}")
    say(f"  share of first signals with up >= 3%: {np.mean(fr['up'][pos] >= 0.03):.2f} (pool {np.mean(up_pool >= 0.03):.2f}); "
        f"since >= 1: {np.mean(fr['since'][pos] >= 1):.2f} (pool {np.mean(s_pool >= 1):.2f})")
    # decomposition: unconditional -> bucket matched -> fine matched -> observed
    unc = fr["hold20"][pool_m].mean()
    say(f"  decomposition of the hit rate: unconditional in_dd {unc:.3f} -> year x up-bucket matched {m0['hold20_base']:.3f} "
        f"(null N1 mean {d_yb.mean():.3f}) -> year x up+-2pp {d_f.mean():.3f} -> + since-low bucket {d_us.mean():.3f} -> observed {obs:.3f}")
    share = (m0["hold20_base"] - unc) / (obs - unc) if obs != unc else np.nan
    share_f = (d_f.mean() - unc) / (obs - unc) if obs != unc else np.nan
    say(f"  share of (observed - unconditional) explained by rebound state: bucket {share:.0%}, +-2pp {share_f:.0%}")

    # ============================================================ verdict
    say("\n[6] Verdict inputs")
    vin = dict(indep_match=indep_match, p_null_yb=p_yb, p_null_yb_rejection=p_rej, p_null_fine=p_f, p_null_since=p_us,
               p_2026=p26, p_2026_fine=p26f, p_perm=p_perm_d, p_shift=p_cs_d)
    say("  " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in vin.items()))
    lost = [k for k, v in core.items() if not l2(v[0])]
    say(f"  core variants losing L2: {lost}")
    gs_hl = np.array([bool(((notes.entry == idx[q]) & isR & (notes.bank == "GS")
                            & (notes.verification == "headline_summary_only")).any()) for q in pos])
    ex_gs_hl, ex_tot = float(np.nansum(diff[gs_hl])), float(np.nansum(diff))
    say(f"  summed hold20 - matched over first signals {ex_tot:+.3f}; from the {int(gs_hl.sum())} signal days carrying a GS "
        f"headline_summary_only R note {ex_gs_hl:+.3f}; from the other {int((~gs_hl).sum())} {ex_tot - ex_gs_hl:+.3f}")
    say("\n[7] VERDICT: fragile")
    say(f"""  For (what survives):
  - Exact independent reproduction (13 dates, all per-signal outcomes and matched bases, all periods).
  - No timing look-ahead: entry sessions re-derived from published_at for all 245 notes; signal day = entry session,
    trade at O[t+1]; state uses closes <= t only (truncation test identical).
  - Main period 12/13 vs matched {m0['hold20_base']:.3f}; same-state nulls with the 20-session spacing give p = {p_yb:.4f}
    (year x up-bucket), {p_rej:.4f} (rejection sampler), {p_f:.4f} (year, up +-2pp), {p_us:.4f} (up x since-low); label
    permutation p = {p_perm_d:.4f}; circular date shift p = {p_cs_d:.4f}. Placebo note families (N, D, T, S, any non-R)
    do not do this. The excess is not lateness: R first signals sit at the in_dd median for up and days since the low.
  Against (why it is not confirmed):
  - Multiple testing: {N_CELLS} cells; no p clears Bonferroni {0.05 / N_CELLS:.5f}; P(some cell reaches p <= {p_yb:.4f}
    by chance) ~ {1 - (1 - p_yb) ** N_CELLS:.2f} if independent. Without spacing the same null gives p ~ 0.025.
  - 2026 (the acceptance period): 4/5 vs matched {st0['2026']['hold20_base']:.3f}, null p = {p26:.3f}/{p26f:.3f}, dlo < 0.
    2026H1 is one signal taken at a new 60-day high (breakout, not a bottom). 2026H2: 3/4 vs {st0['2026H2']['hold20_base']:.3f}
    (dlo {st0['2026H2']['hold20_dlo']:+.3f}); its two informative signals (low up) went 1/2 at 20 sessions, 0/2 at 40;
    near5 0/4. Averaged over all 13 R-note in_dd days in 2026H2 (no cooldown) hold20 is BELOW matched (0.692 vs 0.712).
  - Registered robustness: delay +1 loses L2 (0.846, dlo < 0); R note in [t-2, t] loses L1 (0.80); in_dd 8% loses L1
    (0.706); fam_altA / fam_altB lose L2; without headline_summary_only notes n=7 (L1 n-floor fails) and matched 0.944,
    i.e. no excess; non-GS banks 6/7 vs 0.836 (no excess). {ex_gs_hl:+.2f} of the {ex_tot:+.2f} summed excess comes from
    signal days with a GS headline_summary_only note (title/summary coding only), the GS subgroup news_followup already
    picked post hoc on the same 2024~2026 data.
  - Cooldown selection: 5 of the 6 R-note in_dd days that failed hold20 were suppressed by the cooldown after a held
    first signal (2026-07-09 by one session).
    Using all {n_all} R-note in_dd days, the excess is {obs_all:+.3f} (block CI [{lo9:+.3f}, {hi9:+.3f}]; shift p = {p9s:.3f},
    permutation p = {p9p:.3f}) instead of +0.228; cooldown 10 gives 0.833 with dlo +0.002.
  - Near-high days removed: 9/10, null p = 0.063. Leave-one-note-out: dropping the 2026-06-09 or 2026-08-20 note
    (coverage of a single weekly note) drops L1; random 30% R-note deletion keeps L2 in only ~2/3 of runs.
  - Stocks: no replication (2026 0.454 vs 0.452; 2026H2 0.417 vs 0.463), independently recomputed.
  - news_followup H2 (n4h14_verdict.json) had already found R notes do not lift 5/10/20-day SOXX returns over control
    (all, first-only, 2026, altA/altB, no-headline: '불통과'; only GS h10 passed).
  Reading: a genuine-looking but small, coding- and timing-sensitive association in 2024-2025 SOXX data that the
  current regime (2026, esp. H2) does not confirm. Not usable as a '70% bottom confirmation' on its own.""")
    OUT.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nwrote {OUT}")


# ------------------------------------------------------------------ STK independent recompute (news_R only)
def stk_check(notes: pd.DataFrame) -> str:
    out = io.StringIO()
    o = pd.read_csv(ROOT / "data/derived/desk/prices_open.csv", index_col=0, parse_dates=True)
    c = pd.read_csv(ROOT / "data/derived/desk/prices_close.csv", index_col=0, parse_dates=True)
    drop = [x for x in c.columns if x.endswith(".KS") or x.endswith("=X") or x == "QQQ"]
    import exchange_calendars as xc
    cal = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range("2025-06-02", "2026-10-23"))
    cal = cal.tz_localize(None) if cal.tz is not None else cal
    o, c = o.drop(columns=drop).reindex(cal), c.drop(columns=drop).reindex(cal)
    Cv, Ov = c.to_numpy(float), o.to_numpy(float)
    N, K = Cv.shape
    start, thr = pd.Timestamp("2025-09-01"), 0.20
    sig = np.zeros(N, bool)
    pos_of = {d: i for i, d in enumerate(cal)}
    for e in notes.loc[notes.family == "R", "entry"]:
        if e in pos_of:
            sig[pos_of[e]] = True
    rows = []
    pool = []
    hk = (cal.year.to_numpy() * 10 + (cal.month.to_numpy() > 6))
    for k in range(K):
        x = Cv[:, k]
        Ldd = np.full(N, np.nan)
        dd = np.full(N, np.nan)
        for t in range(59, N):
            w = x[t - 59:t + 1]
            if np.isnan(w).any():
                continue
            pk = t - 59 + int(np.flatnonzero(w == w.max())[-1])
            Ldd[t] = x[pk:t + 1].min()
            dd[t] = x[t] / w.max() - 1
        indd = np.array([np.nanmin(dd[max(0, t - 19):t + 1]) <= -thr if np.isfinite(dd[max(0, t - 19):t + 1]).any() else False
                         for t in range(N)])
        up = x / Ldd - 1
        h = np.full(N, np.nan)
        for t in range(N - 1):
            if np.isnan(Ldd[t]):
                continue
            w = x[t + 1:min(t + 20, N - 1) + 1]
            if np.isnan(w).all():
                continue
            mn = np.nanmin(w)
            if mn < Ldd[t] - 1e-12:
                h[t] = 0.0
            elif t + 20 <= N - 1 and not np.isnan(w).any():
                h[t] = 1.0
        b = np.where(np.isnan(up), np.nan, np.digitize(up, EDGES))
        ok_state = indd & ~np.isnan(Ldd) & (cal >= start)
        for t in np.flatnonzero(ok_state & ~np.isnan(h)):
            pool.append((hk[t], b[t], h[t]))
        prev = -10 ** 9
        for t in np.flatnonzero(sig & ok_state):
            if t - prev <= 20:
                continue
            rows.append((cal[t], c.columns[k], t, h[t], b[t], hk[t], up[t]))
            prev = t
    P = pd.DataFrame(pool, columns=["s", "b", "h"])
    base = P.groupby(["s", "b"]).h.mean()
    S = pd.DataFrame(rows, columns=["date", "series", "pos", "hold20", "b", "s", "up"])
    S["base"] = [base.get((s, b), np.nan) for s, b in zip(S.s, S.b)]
    fam = pd.read_csv(RES / "f6_STK_signals.csv", parse_dates=["date"])
    fam = fam[fam.rule == "news_R"]
    a = S.set_index(["date", "series"]).sort_index()
    f = fam.set_index(["date", "series"]).sort_index()
    same = a.index.equals(f.index)
    hm = same and bool(np.allclose(a.hold20.to_numpy(float), f.hold20.to_numpy(float), equal_nan=True))
    bm = same and bool(np.allclose(a.base.to_numpy(float), f.base_hold20.to_numpy(float), equal_nan=True))
    out.write(f"  independent STK recompute (82 stocks, thr 20%, fixed censoring): first signals {len(S)} vs family {len(fam)}; "
              f"index identical {same}; hold20 identical {hm}; matched base identical {bm}\n")
    for nm, lo, hi in (("2026", "2026-01-02", "2026-10-23"), ("2026H1", "2026-01-02", "2026-05-29"),
                       ("2026H2", "2026-06-01", "2026-10-23"), ("2025", "2025-09-01", "2025-12-31")):
        g = S[(S.date >= lo) & (S.date <= hi) & S.hold20.notna()]
        out.write(f"    {nm:7s} n={len(g):4d} dates {g.date.nunique():2d} hold20 {g.hold20.mean():.3f} matched {g.base.mean():.3f}\n")
    return out.getvalue().rstrip()


if __name__ == "__main__":
    main()
