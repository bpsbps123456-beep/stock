"""Family "crowd": Korean SOXL-crowding (BB-1 calendar) as exit / re-entry / stay-out on the weekly 7-stock plan.

Run from the repo root:  py data/derived/analysis/desk_defense_check_20261119/ext_crowd.py

Series (read only, snapshotted once under cache/crowd/ so a later now_update.py run cannot change a rerun):
  bb1_calendar.json  one row per US session: dd60 (SOXX previous close vs 60-session highest close),
                     percentile (250-session percentile of the 5-session Korean SOXL buy / top-50 buy ratio,
                     aligned to the session on which SEIBRO had published it), on (BB-1 = dd60<=-20% & pct>=80).
  Timing: row j is knowable before the open of session j (pit_check() recomputes both columns from inputs
  truncated at the previous close / at the source date and checks known_date <= session date). So the state
  computed from rows <= j decides the exposure held from the open of j (common.replay_frac contract).

Grid, declared before any result was looked at (205 candidates + base):
  C1 shallow-crowded exit   trigger pct>={90,95,98} & -20% < dd60 <= -{5,8}%                               (6)
                            x re-entry {A50: pct<50, A80: pct<80, B: BB-1 on, C10: SOXX10r1 state,
                                        C20: SOXX20r1 state, D5/D10/D20: N sessions after the last trigger
                                        day, E: BB-1 on or SOXX20r1 state}                                  (9)
                            x off-state exposure {cash 0, half 0.5}                                         = 108
  C2 crowding alone         pct>={90,95,98} on {1,3} consecutive known sessions; re-entry pct<{50,80};
                            x {cash, half}                                                                  = 24
  C3 capitulation           cap = pct<={5,10,20} & dd60<=-{15,20,25}%                                      (9)
     C3r  re-entry          plain gate SOXX{10,20,50}r1; back in at cap or at the gate's own recovery       = 27
     C3s  stay-out          in iff gate & not cap, gate in {none, SOXX10r1, SOXX20r1, SOXX50r1}             = 36
  C4 BB-1 on plain gates    C4r re-entry (same latch as C3r with BB-1) x 3 gates; C4o override
                            (in iff gate or BB-1) x 3 gates                                                 = 6
  C5 section 12.3 candidate SOXL/(SOXL+SOXS) dollar volume, 5-day sums, 250-session percentile >= 0.8 at
                            the previous close -> {cash, half} until {10,21} sessions after the last
                            trigger day (written down 2026-10-14, commit 750d1955)                          = 4
State conventions fixed in advance:
  * rows with unknown percentile fire nothing: every percentile-dependent boolean (trigger, pct<x, BB-1, cap)
    keeps its last known value; price-only (SOXX gate) and timer conditions are still evaluated;
  * C1: while the trigger is active it wins over a price re-entry (stay out); timers restart on each trigger day;
  * C3r/C4r latch: exit only on a fresh gate-off (gate true the session before); an early re-entry is held
    until the gate has recovered and failed again; a buy state already on at the gate exit means no exit;
  * the calendar starts 2026-01-02, so every rule starts invested.
Outputs: cache/crowd_daily.parquet, out/crowd_candidates.csv (common.save), cache/crowd_summary.json.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import study  # noqa: E402

ANALYSIS = HERE.parent
BELT_NOW = ANALYSIS / "bottom_zone_20261111" / "belt" / "now"
HF = ANALYSIS / "hf_options" / "cache"
FROZEN = common.CACHE / "crowd"
SOURCES = {"bb1_calendar.json": BELT_NOW / "bb1_calendar.json",
           "source_flows.csv": BELT_NOW / "data" / "source_flows.csv",
           "prices.csv": BELT_NOW / "data" / "prices.csv"}
WRITTEN = "2026-10-14"          # section 12.3 candidate committed 2026-10-14 13:39 KST (before that US open)
BAD = {"jun_jul": ("2026-06-22", "2026-07-29"), "oct_nov": ("2026-10-13", "2026-11-17")}
N_PLACEBO, SEED = 1000, 20261119


# ---------------------------------------------------------------- inputs
def frozen(name):
    """Snapshot once; later runs read the snapshot."""
    FROZEN.mkdir(parents=True, exist_ok=True)
    target = FROZEN / name
    if not target.exists():
        if name == "prices.csv":            # keep only what C5 needs
            px = pd.read_csv(SOURCES[name])
            px[px.ticker.isin(["SOXL", "SOXS"])][["date", "ticker", "close", "volume"]].to_csv(target, index=False)
        else:
            shutil.copy2(SOURCES[name], target)
    return target


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_signal(days):
    rows = json.loads(frozen("bb1_calendar.json").read_text(encoding="utf-8"))
    cal = pd.DataFrame(rows).set_index("day").reindex(days)
    if cal.dd60.isna().any():
        raise ValueError("calendar does not cover every session")
    pct = cal.percentile.astype(float).to_numpy()
    known = np.isfinite(pct)
    on = cal["on"].to_numpy(dtype=object)
    if not all((v is None or (isinstance(v, float) and np.isnan(v))) == (not k) for v, k in zip(on, known)):
        raise ValueError("BB-1 unknown rows differ from percentile unknown rows")
    bb1 = np.array([bool(v) if k else False for v, k in zip(on, known)])
    return dict(pct=pct, dd=cal.dd60.astype(float).to_numpy(), bb1=bb1, known=known)


def lev_crowding():
    """Section 12.3 series rebuilt as hf_options did: bull5 = sum5(SOXL $vol)/sum5(SOXL+SOXS $vol),
    percentile = share of the last 250 values (min 125) <= today's, measured at each close."""
    px = pd.read_csv(frozen("prices.csv"))
    dv = (px.close * px.volume).groupby([px.date, px.ticker]).sum().unstack()
    dv = dv.dropna()
    bull, bear = dv["SOXL"].rolling(5, min_periods=5).sum(), dv["SOXS"].rolling(5, min_periods=5).sum()
    ratio = (bull / (bull + bear)).dropna()
    pct = ratio.rolling(250, min_periods=125).apply(lambda w: (w <= w[-1]).mean(), raw=True)
    return pd.DataFrame(dict(ratio=ratio, pct=pct))


# ---------------------------------------------------------------- state machines (True = invested)
def carry(flag, known):
    out, last = np.zeros(len(flag), bool), False
    for j in range(len(flag)):
        if known[j]:
            last = bool(flag[j])
        out[j] = last
    return out


def within(events, n):
    """An event on any of the last n sessions including today."""
    return pd.Series(events.astype(float)).rolling(n, min_periods=1).max().to_numpy() > 0


def exit_reenter(trig, reenter):
    state, out = True, np.zeros(len(trig), bool)
    for j in range(len(trig)):
        if state:
            state = not trig[j]
        elif reenter[j] and not trig[j]:
            state = True
        out[j] = state
    return out


def latch(gate, buy):
    pos, out = bool(gate[0]), np.zeros(len(gate), bool)
    for j in range(len(gate)):
        if gate[j]:
            pos = True
        else:
            if j == 0 or gate[j - 1]:
                pos = False
            if buy[j]:
                pos = True
        out[j] = pos
    return out


def consecutive(flag, known, k):
    run, out = 0, np.zeros(len(flag), bool)
    for j in range(len(flag)):
        if known[j]:
            run = run + 1 if flag[j] else 0
            out[j] = run >= k
    return out


def build_rules(sig, gates, lev_trigger):
    """key -> (invested bool array, off-state exposure, config). lev_trigger: bool per session (C5)."""
    pct, dd, known = sig["pct"], sig["dd"], sig["known"]
    safe = np.where(known, pct, -1.0)
    bb1 = carry(sig["bb1"], known)
    g = {n: np.asarray(gates[f"SOXX{n}r1"], bool) for n in (10, 20, 50)}
    rules = {}

    def add(key, inv, sub, definition, **cfg):
        for size, off in (("cash", 0.0), ("half", 0.5)) if cfg.pop("sizes", True) else (("cash", 0.0),):
            rules[f"{key}_{size}"] = (inv, off, dict(subfamily=sub, size=size, definition=definition, **cfg))

    for p in (90, 95, 98):
        for d in (5, 8):
            raw = known & (safe >= p) & (dd <= -d / 100) & (dd > -.20)
            trig = carry(raw, known)
            reentries = {"A50": carry(safe < 50, known), "A80": carry(safe < 80, known), "B": bb1,
                         "C10": g[10], "C20": g[20], "E": bb1 | g[20]}
            text = {"A50": "pct<50", "A80": "pct<80", "B": "BB-1 on", "C10": "SOXX>=SMA10", "C20": "SOXX>=SMA20",
                    "E": "BB-1 on or SOXX>=SMA20"}
            for name, cond in reentries.items():
                add(f"C1_p{p}_d{d}_{name}", exit_reenter(trig, cond), "C1",
                    f"exit when pct>={p} and dd60 in (-20%,-{d}%]; back when {text[name]}",
                    trigger=f"p{p}_d{d}", reentry=name)
            for n in (5, 10, 20):
                add(f"C1_p{p}_d{d}_D{n}", ~within(raw, n), "C1",
                    f"exit when pct>={p} and dd60 in (-20%,-{d}%]; back {n} sessions after the last trigger day",
                    trigger=f"p{p}_d{d}", reentry=f"D{n}")
    for p in (90, 95, 98):
        for k in (1, 3):
            for y in (50, 80):
                trig = consecutive(safe >= p, known, k)
                add(f"C2_p{p}_k{k}_A{y}", exit_reenter(trig, carry(safe < y, known)), "C2",
                    f"exit when pct>={p} on {k} consecutive known sessions; back when pct<{y}",
                    trigger=f"p{p}_k{k}", reentry=f"A{y}")
    for p in (5, 10, 20):
        for d in (15, 20, 25):
            cap = carry(known & (safe <= p) & (safe >= 0) & (dd <= -d / 100), known)
            for n in (10, 20, 50):
                add(f"C3r_p{p}_d{d}_SOXX{n}", latch(g[n], cap), "C3r",
                    f"SOXX{n}r1 gate; back in at pct<={p} & dd60<=-{d}% or at the gate's recovery",
                    trigger=f"p{p}_d{d}", reentry=f"SOXX{n}r1", sizes=False)
            for name, gate in (("none", np.ones(len(dd), bool)), ("SOXX10", g[10]), ("SOXX20", g[20]), ("SOXX50", g[50])):
                add(f"C3s_p{p}_d{d}_{name}", gate & ~cap, "C3s",
                    f"{name} gate and stay out while pct<={p} & dd60<=-{d}%",
                    trigger=f"p{p}_d{d}", reentry=name, sizes=False)
    for n in (10, 20, 50):
        add(f"C4r_SOXX{n}", latch(g[n], bb1), "C4r", f"SOXX{n}r1 gate; back in at BB-1 or at the gate's recovery",
            trigger="bb1", reentry=f"SOXX{n}r1", sizes=False)
        add(f"C4o_SOXX{n}", g[n] | bb1, "C4o", f"SOXX{n}r1 gate ignored while BB-1 is on",
            trigger="bb1", reentry=f"SOXX{n}r1", sizes=False)
    for n in (10, 21):
        add(f"C5_lev80_D{n}", ~within(lev_trigger, n), "C5",
            f"SOXL/(SOXL+SOXS) 5-day dollar-volume share in the top 20% of 250 sessions at the previous close; "
            f"out until {n} sessions after the last such day", trigger="lev80", reentry=f"D{n}")
    return rules


def to_frac(inv, off):
    return np.where(inv, 1.0, off)


def spells(frac):
    """(start, end_exclusive, level) for each run below full exposure."""
    out, j, n = [], 0, len(frac)
    while j < n:
        if frac[j] < 1:
            k = j
            while k < n and frac[k] < 1:
                k += 1
            out.append((j, k, float(frac[j])))
            j = k
        else:
            j += 1
    return out


def runs(mask, days):
    text = []
    for a, b, _ in spells(np.where(mask, 0.0, 1.0)):
        text.append(days[a][5:] if b - a == 1 else f"{days[a][5:]}..{days[b - 1][5:]}")
    return text


# ---------------------------------------------------------------- checks
def pit_check(ctx, sig):
    """Recompute both calendar columns from inputs cut at the decision time."""
    days = ctx.days
    close = ctx.book.cls["SOXX"]
    dd = (close / close.rolling(60, min_periods=60).max() - 1).shift().reindex(days).to_numpy()
    dd_err = float(np.nanmax(np.abs(dd - sig["dd"])))
    src = pd.read_csv(frozen("source_flows.csv"), parse_dates=["date", "known_date", "entry"]).set_index("date")
    late, diffs, lags, checked = 0, [], [], 0
    for j, day in enumerate(days):
        cand = src[src.entry == pd.Timestamp(day)]
        if cand.empty or not sig["known"][j]:
            continue
        s = cand.index.max()
        late += int(cand.known_date.max() > pd.Timestamp(day))
        past = src.loc[:s]                                   # nothing after the source trade date
        buy = past.SOXL_buy.where(past.published).rolling(5, min_periods=5).sum()
        tot = past.total_buy.where(past.published).rolling(5, min_periods=5).sum().replace(0, np.nan)
        w = (buy / tot).to_numpy()[-250:]
        w = w[np.isfinite(w)]
        x = (buy / tot).iloc[-1]
        p = 100.0 * ((w < x).sum() + .5 * (w == x).sum()) / len(w)
        diffs.append(abs(p - sig["pct"][j]))
        lags.append(len(pd.bdate_range(s, day)) - 1)
        checked += 1
    return dict(dd60_max_abs_diff_vs_prev_close_recalc=dd_err, percentile_rows_checked=checked,
                percentile_max_abs_diff_truncated_recalc=float(max(diffs)),
                rows_known_after_session_date=late, source_trade_date_lag_weekdays_min_max=[int(min(lags)), int(max(lags))],
                unknown_rows=int((~sig["known"]).sum()))


def lookahead_test(ctx, sig, lev, rules):
    """Scramble everything after session k; exposures through k must not move."""
    rng = np.random.default_rng(7)
    n, bad = len(ctx.days), 0
    cuts = list(range(15, n - 1, 17))
    for k in cuts:
        s = {key: v.copy() for key, v in sig.items()}
        s["pct"][k + 1:] = rng.uniform(0, 100, n - k - 1)
        s["dd"][k + 1:] = -rng.uniform(0, .4, n - k - 1)
        s["bb1"][k + 1:] = rng.random(n - k - 1) < .5
        s["known"][k + 1:] = rng.random(n - k - 1) < .9
        gates = {name: np.r_[g[:k + 1], rng.random(n - k - 1) < .5] for name, g in ctx.gates.items()}
        l2 = lev.copy()
        l2[k + 1:] = rng.random(n - k - 1) < .5
        for key, (inv, _, _) in build_rules(s, gates, l2).items():
            bad += int(not np.array_equal(inv[:k + 1], rules[key][0][:k + 1]))
    return dict(cutoffs=len(cuts), rules=len(rules), violations=bad)


# ---------------------------------------------------------------- fast NAV for placebos
class Fast:
    """USD NAV of a 0/1 exposure path from the base's open-to-open and open-to-close returns (exact for 0/1)."""

    def __init__(self, ctx):
        value, close = np.asarray(ctx.baseline.value), np.asarray(ctx.baseline.close)
        self.oo = value[1:] / value[:-1] - 1
        self.oc = close / value - 1
        self.n = len(value)
        self.h2 = int(np.searchsorted(np.array(ctx.days), "2026-07-01"))

    def usd(self, frac):
        frac = np.atleast_2d(frac)
        nav = np.c_[np.ones(len(frac)), np.cumprod(1 + frac[:, :-1] * self.oo, axis=1)]
        return nav * (1 + frac * self.oc)

    def h2_return(self, frac):
        usd = self.usd(frac)
        return usd[:, -1] / usd[:, self.h2 - 1] - 1


def layouts(pieces, n, count, rng):
    """count x n exposure paths: the same spells (length, level) in random order at random non-overlapping places."""
    out = np.ones((count, n))
    k, total = len(pieces), sum(p[0] for p in pieces)
    sep = 1 if n - total >= k - 1 else 0
    rem = n - total - sep * (k - 1)
    for m in range(count):
        bars = np.sort(rng.choice(rem + k, size=k, replace=False))
        gaps = np.diff(np.r_[-1, bars]) - 1
        pos = 0
        for i, q in enumerate(rng.permutation(k)):
            pos += gaps[i] + (sep if i else 0)
            out[m, pos:pos + pieces[q][0]] = pieces[q][1]
            pos += pieces[q][0]
    return out


def placebo(fast, frac, rng):
    """Percentile of the rule's H2 USD return among random placements: over all of 2026, and inside H2 only."""
    actual = float(fast.h2_return(frac)[0])
    out = dict(fast_h2_usd=actual)
    full = [(b - a, lvl) for a, b, lvl in spells(frac)]
    part = [(b - max(a, fast.h2), lvl) for a, b, lvl in spells(frac) if b > fast.h2]
    for name, pieces, lo in (("year", full, 0), ("h2", part, fast.h2)):
        if not part or not pieces:
            out[name] = np.nan
            continue
        paths = np.ones((N_PLACEBO, fast.n))
        paths[:, lo:] = layouts(pieces, fast.n - lo, N_PLACEBO, rng)
        r = fast.h2_return(paths)
        out[name] = float(100 * ((r < actual).mean() + .5 * (r == actual).mean()))
        out[name + "_median"] = float(np.median(r))
    return out


# ---------------------------------------------------------------- reports
def forward_table(ctx, sig):
    days, n = ctx.days, len(ctx.days)
    value = np.asarray(ctx.baseline.value)
    soxx = ctx.book.opn["SOXX"].reindex(days).to_numpy()
    pb = pd.cut(sig["pct"], [-1, 20, 50, 80, 95, 100], labels=["p<=20", "p20-50", "p50-80", "p80-95", "p>95"])
    db = pd.cut(sig["dd"], [-1, -.20, -.10, -.05, 1], labels=["dd<=-20", "dd-20..-10", "dd-10..-5", "dd>-5"])
    rows = []
    for p in pb.categories:
        for d in db.categories:
            m = (pb == p) & (db == d) & sig["known"]
            row = dict(pct=p, dd60=d, n=int(m.sum()), episodes=len(spells(np.where(m, 0.0, 1.0))))
            for h in (5, 10, 20):
                ix = np.flatnonzero(m)
                ix = ix[ix + h < n]
                row[f"n{h}"] = len(ix)
                row[f"soxx{h}"] = float(np.mean(soxx[ix + h] / soxx[ix] - 1)) if len(ix) else np.nan
                row[f"base{h}"] = float(np.mean(value[ix + h] / value[ix] - 1)) if len(ix) else np.nan
            rows.append(row)
    ix = np.arange(n)
    every = dict(pct="all", dd60="all", n=n, episodes=1)
    for h in (5, 10, 20):
        k = ix[ix + h < n]
        every.update({f"n{h}": len(k), f"soxx{h}": float(np.mean(soxx[k + h] / soxx[k] - 1)),
                      f"base{h}": float(np.mean(value[k + h] / value[k] - 1))})
    return pd.DataFrame(rows + [every])


def spell_list(ctx, sig, frac):
    days, value = ctx.days, np.asarray(ctx.baseline.value)
    close = np.asarray(ctx.baseline.close)
    soxx_o, soxx_c = (ctx.book[k]["SOXX"].reindex(days).to_numpy() for k in ("opn", "cls")) \
        if isinstance(ctx.book, dict) else (ctx.book.opn["SOXX"].reindex(days).to_numpy(),
                                            ctx.book.cls["SOXX"].reindex(days).to_numpy())
    rows = []
    for a, b, _ in spells(frac):
        end_base = value[b] if b < len(days) else close[-1]
        end_soxx = soxx_o[b] if b < len(days) else soxx_c[-1]
        rows.append(dict(exit_open=days[a], back_open=days[b] if b < len(days) else "(still out at 11-17 close)",
                         sessions=b - a, pct_at_exit=None if not sig["known"][a] else float(sig["pct"][a]),
                         dd60_at_exit=float(sig["dd"][a]), base_nav_at_exit=float(value[a]),
                         base_nav_at_back=float(end_base), base_avoided=float(end_base / value[a] - 1),
                         soxx_move=float(end_soxx / soxx_o[a] - 1)))
    return rows


def main():
    started = time.time()
    ctx = common.context()
    days, n = ctx.days, len(ctx.days)
    args = (days, ctx.opn, ctx.close, ctx.fx)
    sig = load_signal(days)
    lev = lev_crowding()
    lev_on = (lev.pct >= .8).shift(1, fill_value=False)            # measured at the previous close
    lev_trigger = lev_on.reindex(days)
    if lev_trigger.isna().any():
        raise ValueError("leverage series does not cover every session")
    lev_trigger = lev_trigger.to_numpy(bool)
    # C5 timers need trigger days before 2026-01-02 as well
    lev_full = {m: (~within(lev_on.to_numpy(bool), m))[[lev_on.index.get_loc(d) for d in days]] for m in (10, 21)}

    rules = build_rules(sig, ctx.gates, lev_trigger)
    for m in (10, 21):
        for size in ("cash", "half"):
            key = f"C5_lev80_D{m}_{size}"
            rules[key] = (lev_full[m], rules[key][1], rules[key][2])
    assert len(rules) == 205, len(rules)

    base = study.replay(*args, ctx.weekly, ctx.gates["none"])
    np.testing.assert_allclose(base.usd, ctx.baseline.close, rtol=1e-10)
    np.testing.assert_allclose(common.replay_frac(*args, ctx.weekly, np.ones(n)).to_numpy(), base.to_numpy(),
                               rtol=1e-10, atol=1e-10)
    curves, configs, fracs = {"base": base}, {"base": dict(subfamily="base", size="full", definition="현행 weekly 7",
                                                         trigger="", reentry="")}, {"base": np.ones(n)}
    for key, (inv, off, cfg) in rules.items():
        fracs[key] = to_frac(inv, off)
        curves[key] = common.replay_frac(*args, ctx.weekly, fracs[key])
        configs[key] = cfg
    np.testing.assert_allclose(curves["base"].to_numpy(),
                               study.replay(*args, ctx.weekly, ctx.gates["none"]).to_numpy(), rtol=1e-10, atol=1e-10)
    common.save("crowd", curves, configs)

    st = common.stats_table(curves)
    cfg = pd.DataFrame.from_dict(configs, orient="index")
    fast = Fast(ctx)
    cash_keys = [k for k in rules if rules[k][1] == 0]
    for key in cash_keys:                                           # the fast path is exact for 0/1 exposure
        np.testing.assert_allclose(fast.usd(fracs[key])[0], curves[key].usd.to_numpy(), rtol=1e-9)
    half_err = max(float(np.max(np.abs(fast.usd(fracs[k])[0] / curves[k].usd.to_numpy() - 1)))
                   for k in rules if rules[k][1] > 0)

    # references: the plain gates C3/C4 modify
    ref = {f"SOXX{m}r1": study.replay(*args, ctx.weekly, ctx.gates[f"SOXX{m}r1"]) for m in (10, 20, 50)}
    ref_st = common.stats_table(ref)

    # matched control and placebo for every rule
    rng = np.random.default_rng(SEED)
    extra = {}
    for key in rules:
        r = st.loc[key]
        ctl = common.matched_control(base, r.h2_expo, "h2")
        pl = placebo(fast, fracs[key], rng)
        extra[key] = dict(matched_h2_usd=ctl[0], matched_h2_usd_mdd=ctl[1], placebo_year=pl["year"], placebo_h2=pl["h2"],
                          placebo_year_median=pl.get("year_median", np.nan), placebo_h2_median=pl.get("h2_median", np.nan),
                          round_trips=len(spells(fracs[key])),
                          out_sessions_h2=int((fracs[key][fast.h2:] < 1).sum()))
    extra = pd.DataFrame.from_dict(extra, orient="index")
    full = st.join(extra).join(cfg)

    # lag sensitivity: every calendar row and the leverage trigger used one session later
    late = {k: np.r_[v[:1] * 0 if k != "dd" else v[:1], v[:-1]] for k, v in sig.items()}
    late["known"][0], late["bb1"][0] = False, False
    late_rules = build_rules(late, ctx.gates, np.r_[False, lev_trigger[:-1]])
    late_on = lev_on.shift(1, fill_value=False).to_numpy(bool)
    for m in (10, 21):
        for size in ("cash", "half"):
            key = f"C5_lev80_D{m}_{size}"
            late_rules[key] = ((~within(late_on, m))[[lev_on.index.get_loc(d) for d in days]],) + late_rules[key][1:]
    late_st = common.stats_table({k: common.replay_frac(*args, ctx.weekly, to_frac(v[0], v[1]))
                                  for k, v in late_rules.items()})
    full["h2_usd_lag1"] = late_st.h2_usd
    full["sep_nov_usd_lag1"] = late_st.sep_nov_usd

    b = st.loc["base"]
    # ---- selections
    selections = []
    for train, test in (("h1", "h2"), ("jan_aug", "sep_nov")):
        for currency in ("krw", "usd"):
            for rule, key in common.select(st, train, currency).items():
                r = st.loc[key]
                selections.append(dict(protocol=f"{train}->{test}", currency=currency, rule=rule, pick=key,
                                       **{f"{w}_{m}": float(r[f"{w}_{m}"]) for w in (train, test)
                                          for m in ("krw", "usd", "krw_mdd", "usd_mdd", "expo")}))

    # ---- subfamilies
    def summary(group):
        rows = dict(n=len(group))
        for w in ("h2", "jul_aug", "sep_nov", "h1", "full"):
            rows[f"{w}_usd_median"] = float(group[f"{w}_usd"].median())
            rows[f"{w}_usd_best"] = float(group[f"{w}_usd"].max())
        rows.update(h2_expo_median=float(group.h2_expo.median()), h2_usd_mdd_median=float(group.h2_usd_mdd.median()),
                    beat_base_h2=int((group.h2_usd > b.h2_usd + 1e-12).sum()),
                    beat_matched_h2=int((group.h2_usd > group.matched_h2_usd + 1e-12).sum()),
                    beat_base_full=int((group.full_usd > b.full_usd + 1e-12).sum()),
                    placebo_year_median=float(group.placebo_year.median()),
                    placebo_h2_median=float(group.placebo_h2.median()),
                    placebo_h2_ge95=int((group.placebo_h2 >= 95).sum()),
                    h2_usd_lag1_median=float(group.h2_usd_lag1.median()))
        return rows

    cand = full.drop(index="base")
    cand = cand.assign(sub=np.where(cand.subfamily.isin(["C1", "C2", "C5"]), cand.subfamily + "_" + cand["size"], cand.subfamily))
    subfamily = pd.DataFrame({name: summary(g) for name, g in cand.groupby("sub")}).T
    c1 = cand[(cand.subfamily == "C1") & (cand["size"] == "cash")]
    c1_by_reentry = pd.DataFrame({name: summary(g) for name, g in c1.groupby("reentry")}).T
    c1_by_trigger = pd.DataFrame({name: summary(g) for name, g in c1.groupby("trigger")}).T

    # ---- C3/C4 against the gate they modify
    gate_delta = []
    for key, r in cand[cand.subfamily.isin(["C3r", "C3s", "C4r", "C4o"])].iterrows():
        gate = r.reentry if r.reentry.endswith("r1") else (r.reentry + "r1" if r.reentry != "none" else None)
        g = ref_st.loc[gate] if gate else b
        gate_delta.append(dict(strategy=key, subfamily=r.subfamily, gate=gate or "none",
                               **{f"d_{w}_usd": float(r[f"{w}_usd"] - g[f"{w}_usd"]) for w in ("h1", "h2", "jul_aug", "sep_nov", "full")},
                               differs=bool(not np.array_equal(fracs[key], (ctx.gates[gate] if gate else np.ones(n, bool)).astype(float)))))
    gate_delta = pd.DataFrame(gate_delta).set_index("strategy")

    # ---- headline rules (chosen by fixed criteria, each labelled)
    def best(mask, column="h2_usd"):
        return cand[mask].sort_values([column], ascending=False).index[0]

    picks = {"C1 best H2 (hindsight)": best((cand.subfamily == "C1") & (cand["size"] == "cash")),
             "C1 best H1-train (max H1 USD among C1 cash with H1 MDD <= base)":
                 cand[(cand.subfamily == "C1") & (cand["size"] == "cash") & (cand.h1_usd_mdd >= b.h1_usd_mdd - 1e-7)]
                 .sort_values("h1_usd", ascending=False).index[0]
                 if ((cand.subfamily == "C1") & (cand["size"] == "cash") & (cand.h1_usd_mdd >= b.h1_usd_mdd - 1e-7)).any() else None,
             "C1 central (p95, dd<=-5%, back at BB-1 or SMA20)": "C1_p95_d5_E_cash",
             "C2 best H2 (hindsight)": best((cand.subfamily == "C2") & (cand["size"] == "cash")),
             "C3r best H2 (hindsight)": best(cand.subfamily == "C3r"),
             "C3s best H2 (hindsight)": best(cand.subfamily == "C3s"),
             "C4r best H2 (hindsight)": best(cand.subfamily == "C4r"),
             "C4o best H2 (hindsight)": best(cand.subfamily == "C4o"),
             "C5 as written (cash, 21 sessions)": "C5_lev80_D21_cash"}
    for s in selections:
        if s["pick"] != "base":
            picks[f"selected {s['protocol']} {s['currency']} {s['rule']}"] = s["pick"]
    headline = []
    for label, key in picks.items():
        if key is None:
            continue
        r = full.loc[key]
        headline.append(dict(label=label, strategy=key, definition=r.definition,
                             **{c: float(r[c]) for c in ("full_usd", "full_usd_mdd", "h1_usd", "h2_usd", "h2_usd_mdd", "jul_aug_usd",
                                                         "sep_nov_usd", "sep_nov_usd_mdd", "h2_krw", "h2_expo", "matched_h2_usd",
                                                         "placebo_year", "placebo_h2", "placebo_h2_median", "h2_usd_lag1")},
                             round_trips=int(r.round_trips), out_sessions_h2=int(r.out_sessions_h2)))

    # ---- trigger days and the bad stretches
    trig_days = {}
    for p in (90, 95, 98):
        for d in (5, 8):
            raw = sig["known"] & (np.where(sig["known"], sig["pct"], -1) >= p) & (sig["dd"] <= -d / 100) & (sig["dd"] > -.20)
            day_arr = np.array(days)
            windows = {"jan_may": ("2026-01-01", "2026-05-31"), "pre_june_peak_0601_0622": ("2026-06-01", "2026-06-22"),
                       "decline_0623_0729": ("2026-06-23", "2026-07-29"), "0730_0921": ("2026-07-30", "2026-09-21"),
                       "pre_oct_0922_1012": ("2026-09-22", "2026-10-12"), "decline_1013_1117": ("2026-10-13", "2026-11-17")}
            trig_days[f"p{p}_d{d}"] = dict(n=int(raw.sum()), runs=runs(raw, days),
                                           **{w: int(raw[(day_arr >= a) & (day_arr <= z)].sum()) for w, (a, z) in windows.items()})
    value = pd.Series(np.asarray(ctx.baseline.value), index=days)
    close = pd.Series(np.asarray(ctx.baseline.close), index=days)
    base_marks = {d: float(close[d]) for d in ("2026-06-22", "2026-07-29", "2026-09-22", "2026-10-12", "2026-11-05", "2026-11-17")}
    base_marks["h2_peak_close"] = [str(close.loc["2026-07-01":].idxmax()), float(close.loc["2026-07-01":].max())]
    base_marks["h1_peak_close"] = [str(close.loc[:"2026-06-30"].idxmax()), float(close.loc[:"2026-06-30"].max())]

    best_c1 = picks["C1 best H2 (hindsight)"]
    spell_tables = {k: spell_list(ctx, sig, fracs[k]) for k in dict.fromkeys(
        [best_c1, "C1_p95_d5_E_cash", picks["C1 best H1-train (max H1 USD among C1 cash with H1 MDD <= base)"] or best_c1])}

    # ---- C5: definition check, trigger days, and the part after it was written down
    hf = pd.read_parquet(HF / "panel.parquet")["lev_soxx_bull5"].dropna()
    hf.index = pd.Index([str(pd.Timestamp(d).date()) for d in hf.index])
    mine = lev.pct.shift(1).reindex(hf.index).loc["2026-01-01":]
    theirs = hf.loc["2026-01-01":]
    both = pd.concat([mine, theirs], axis=1, keys=["mine", "hf"]).dropna()
    after = np.array(days) >= WRITTEN
    c5 = dict(written=WRITTEN, source="belt/now/data/prices.csv (Yahoo close x volume), snapshot cache/crowd/prices.csv",
              rebuilt_vs_hf_options_2026=dict(rows=len(both), max_abs_diff=float((both.mine - both.hf).abs().max()),
                                              flag_agree=float(((both.mine >= .8) == (both.hf >= .8)).mean())),
              trigger_runs_2026=runs(lev_trigger, days), trigger_days_2026=int(lev_trigger.sum()),
              trigger_days_after_written=int(lev_trigger[after].sum()),
              pct_after_written_min_max=[float(lev.pct.shift(1).reindex(days)[after].min()),
                                         float(lev.pct.shift(1).reindex(days)[after].max())],
              out_sessions_after_written={k: int((fracs[k][after] < 1).sum()) for k in rules if k.startswith("C5")},
              window_after_written={k: common.window_stats(curves[k], WRITTEN, "2026-12-31")
                                    for k in ["base"] + [k for k in rules if k.startswith("C5")]},
              in_sample_before_written={k: common.window_stats(curves[k], "2026-01-01", "2026-10-13")
                                        for k in ["base"] + [k for k in rules if k.startswith("C5")]})
    soxx_o = ctx.book.opn["SOXX"].reindex(days).to_numpy()
    c5["soxx_fwd21_open_to_open_2026"] = {}
    ix = np.arange(n)
    ok = ix + 21 < n
    for name, m in (("trigger", lev_trigger), ("rest", ~lev_trigger)):
        for part, pm in (("jan_may", np.array(days) <= "2026-05-31"), ("jun_to_written", (np.array(days) >= "2026-06-01") & ~after)):
            k = ix[m & pm & ok]
            c5["soxx_fwd21_open_to_open_2026"][f"{name}_{part}"] = dict(
                n=len(k), mean=float(np.mean(soxx_o[k + 21] / soxx_o[k] - 1)) if len(k) else None)

    out = dict(
        family="crowd", n_ledgers=len(curves), n_candidates=len(rules), seconds=round(time.time() - started, 1),
        inputs={name: sha(frozen(name)) for name in SOURCES},
        base={k: float(b[k]) for k in ("full_usd", "h1_usd", "h2_usd", "h2_usd_mdd", "jan_aug_usd", "jul_aug_usd", "sep_nov_usd",
                                       "sep_nov_usd_mdd", "h1_krw", "h2_krw", "h2_expo", "full_usd_mdd")},
        base_marks=base_marks, pit_check=pit_check(ctx, sig), lookahead_selftest=lookahead_test(ctx, sig, lev_trigger, rules),
        fast_nav=dict(cash_rules_exact=len(cash_keys), half_rules_max_rel_error=half_err),
        subfamily_counts=cand.groupby("sub").size().to_dict(),
        plain_gates={k: {c: float(ref_st.loc[k, c]) for c in ("h1_usd", "h2_usd", "h2_usd_mdd", "jul_aug_usd", "sep_nov_usd", "full_usd", "h2_expo")}
                     for k in ref_st.index},
        selections=selections, subfamily=subfamily.to_dict("index"), c1_by_reentry=c1_by_reentry.to_dict("index"),
        c1_by_trigger=c1_by_trigger.to_dict("index"), gate_delta=gate_delta.reset_index().to_dict("records"),
        headline=headline, trigger_days=trig_days, spells=spell_tables, c5=c5,
        bb1_runs=runs(carry(sig["bb1"], sig["known"]) & sig["known"], days),
        forward_table=forward_table(ctx, sig).to_dict("records"))
    full.to_csv(FROZEN / "crowd_full_stats.csv")
    (common.CACHE / "crowd_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    return out, full


if __name__ == "__main__":
    result, table = main()
    brief = {k: result[k] for k in ("n_ledgers", "n_candidates", "seconds", "base", "pit_check", "lookahead_selftest",
                                    "fast_nav", "subfamily_counts", "plain_gates")}
    print(json.dumps(brief, indent=1, ensure_ascii=False, default=float))
