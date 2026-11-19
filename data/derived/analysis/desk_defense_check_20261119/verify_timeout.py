"""verify_timeout: independent attack on "leave when SOXX breaks a short SMA, come back after N sessions".

Run from the repo root:  py data/derived/analysis/desk_defense_check_20261119/verify_timeout.py

Targets
  (a) reentry:E1_R1_wait5_orgate  (ext_reentry.py)   exit SOXX close < SMA10, back after 5 sessions or gate recovery
  (b) oos:sr_soxx50_cash20        (oos_prior.py)     SOXX crosses below SMA50 -> cash for 20 sessions

Everything here is re-implemented: own share-level replay (targets from ctx.weekly), own SOXX/QQQ signal from
book.cls, own state machines, own window statistics. Reads only; writes cache/verify_timeout/.
Fill model: signal on a completed close, trade at the next regular-session open, zero commission unless stated.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402  context() only (plus matched_control / select as the lead's own yardsticks)

OUT = common.CACHE / "verify_timeout"
WIN = {"full": ("2026-01-01", "2026-12-31"), "h1": ("2026-01-01", "2026-06-30"), "h2": ("2026-07-01", "2026-12-31"),
       "jan_aug": ("2026-01-01", "2026-08-31"), "sep_nov": ("2026-09-01", "2026-12-31"),
       "jul": ("2026-06-23", "2026-07-29"), "rally": ("2026-07-30", "2026-09-22"),
       "octnov": ("2026-10-13", "2026-11-17"), "post": ("2026-10-12", "2026-11-17")}
NS = (5, 8, 10, 12, 15, 20, 30, 50)
WAITS = (2, 3, 5, 7, 10, 15, 20)
MODES = ("pure", "orgate")
SYMS = ("SOXX", "QQQ")
SEED = 20261119


# ----------------------------------------------------------------------------- own engine
def own_replay(ctx, inv):
    """Share-level replay written from the definition: hold the weekly basket from the open of j iff inv[j]."""
    opn, close, fx, targets = ctx.opn, ctx.close, ctx.fx, ctx.weekly
    n, m = opn.shape
    book_sh, book_cash = np.zeros(m), 1.0          # the always-invested weekly book (reference weights)
    sh, cash, drag = np.zeros(m), 1.0, 1.0
    rows = []
    for j in range(n):
        o, c = np.nan_to_num(opn[j]), np.nan_to_num(close[j])
        safe = np.where(o > 0, o, 1.0)
        ref_nav, nav = book_cash + book_sh @ o, cash + sh @ o
        if j in targets:
            w = np.asarray(targets[j], float)
            book_sh = np.where(w > 0, ref_nav * w / safe, 0.0)
            book_cash = ref_nav * (1 - w.sum())
        turn = 0.0
        if (inv[j] and j in targets) or (j > 0 and inv[j] != inv[j - 1]):
            want = book_sh * o / ref_nav if inv[j] else np.zeros(m)
            turn = float(np.abs(want - sh * o / nav).sum())
            sh = np.where(want > 0, nav * want / safe, 0.0)
            cash = nav * (1 - want.sum())
            drag *= 1 - .001 * turn
        usd = cash + sh @ c
        rows.append((usd * fx[j], usd, turn, 1 - cash / nav, usd * fx[j] * drag))
    return pd.DataFrame(rows, index=ctx.days, columns=["krw", "usd", "turn", "exposure", "krw10"])


def fast_nav(inv, bo, bc):
    """USD close NAV of an in/out schedule from the base open/close NAV only (rows = schedules).

    Session j = overnight leg (close j-1 -> open j, earned iff held at close j-1) x day leg (open j -> close j,
    earned iff inv[j]). Exact because an invested ledger always holds the weekly book's weights.
    """
    inv = np.atleast_2d(np.asarray(inv, bool))
    prev_c = np.r_[1.0, bc[:-1]]
    held = np.c_[np.ones(len(inv), bool), inv[:, :-1]]
    f = np.where(inv, bc / bo, 1.0) * np.where(held, bo / prev_c, 1.0)
    return np.cumprod(f, axis=1)


class Windows:
    def __init__(self, days):
        d = np.asarray(days)
        self.ix = {k: np.flatnonzero((d >= a) & (d <= b)) for k, (a, b) in WIN.items()}

    def path(self, nav, k):
        nav = np.atleast_2d(nav)
        ix = self.ix[k]
        before = nav[:, ix[0] - 1] if ix[0] else np.ones(len(nav))
        return np.c_[before, nav[:, ix]]

    def ret(self, nav, k):
        p = self.path(nav, k)
        return p[:, -1] / p[:, 0] - 1

    def mdd(self, nav, k):
        p = self.path(nav, k)
        return (p / np.maximum.accumulate(p, axis=1) - 1).min(axis=1)


# ----------------------------------------------------------------------------- own signals and state machines
def below_prev(ctx, sym, n):
    """cond[j]: close of session j-1 below its SMA n (known before the open of j). dist[j] = close/SMA-1 there."""
    c = ctx.book.cls[sym].astype(float)
    sma = c.rolling(n, min_periods=n).mean()
    pos = np.arange(ctx.start - 1, ctx.start - 1 + len(ctx.days))
    if c.iloc[pos].isna().any() or sma.iloc[pos].isna().any():
        raise ValueError(f"{sym} SMA{n} warm-up missing")
    return (c < sma).to_numpy()[pos], (c / sma - 1).to_numpy()[pos]


def timeout_schedule(cond, wait, mode):
    """Exit at the open of j when cond[j] (armed); back after `wait` sessions, or (orgate) once cond is off.

    After a timed re-entry with cond still on, the exit re-arms only after cond has been off once.
    wait=None: plain gate (back only when cond is off).
    """
    n = len(cond)
    inv, spells = np.ones(n, bool), []
    invested, armed, a = True, True, None
    for j in range(n):
        if invested:
            armed = armed or not cond[j]
            if armed and cond[j]:
                invested, a = False, j
                inv[j] = False
            continue
        timed = wait is not None and j - a >= wait
        gate = (mode == "orgate" or wait is None) and not cond[j]
        if timed or gate:
            invested, armed = True, not cond[j]
            spells.append((a, j, "gate" if gate else "timeout"))
        else:
            inv[j] = False
    if not invested:
        spells.append((a, None, "open"))
    return inv, spells


def rule_b(ctx, n=50, hold=20):
    """(b) from its text: SOXX close crosses below SMA50 (previous close at/above) -> cash from the next open;
    in again at the open after the 20th session from the signal day; signals ignored while out; starts 'in'."""
    c = ctx.book.cls["SOXX"].astype(float)
    sma = c.rolling(n, min_periods=n).mean()
    N, i0 = len(ctx.days), ctx.start
    inv, state, sig = np.ones(N, bool), "in", None
    for i in range(i0, i0 + N - 1):                     # close i decides session j = i - i0 + 1
        if state == "in":
            if c.iloc[i] < sma.iloc[i] and c.iloc[i - 1] >= sma.iloc[i - 1]:
                state, sig = "out", i
        elif i - sig >= hold:
            state = "in"
        inv[i - i0 + 1] = state == "in"
    return inv


def spells_of(inv):
    out, a = [], None
    for j, v in enumerate(inv):
        if not v and a is None:
            a = j
        if v and a is not None:
            out.append((a, j))
            a = None
    if a is not None:
        out.append((a, None))
    return out


# ----------------------------------------------------------------------------- analysis pieces
def metrics(nav, fx, W):
    nav = np.atleast_2d(nav)
    krw = nav * fx
    return {"full_usd": W.ret(nav, "full"), "full_mdd": W.mdd(nav, "full"), "jan_aug_krw": W.ret(krw, "jan_aug"),
            "jan_aug_krw_mdd": W.mdd(krw, "jan_aug"), "sep_nov_usd": W.ret(nav, "sep_nov"),
            "h1_usd": W.ret(nav, "h1"), "h2_usd": W.ret(nav, "h2"), "h2_mdd": W.mdd(nav, "h2"),
            "jul": W.ret(nav, "jul"), "rally": W.ret(nav, "rally"), "octnov": W.ret(nav, "octnov"),
            "post": W.ret(nav, "post")}


def base_declines(bc, days, depth=.08):
    """High-water-mark decline episodes of the base close NAV at least `depth` deep."""
    eps, peak, pj, tj, low = [], 1.0, -1, -1, 1.0
    series = np.r_[1.0, bc]
    for i in range(1, len(series)):
        v, j = series[i], i - 1
        if v >= peak:
            if low / peak - 1 <= -depth:
                eps.append(dict(peak_j=pj, trough_j=tj, recover_j=j, depth=low / peak - 1))
            peak, pj, low, tj = v, j, v, j
        elif v < low:
            low, tj = v, j
    if low / peak - 1 <= -depth:
        eps.append(dict(peak_j=pj, trough_j=tj, recover_j=None, depth=low / peak - 1))
    for e in eps:
        e["peak_day"] = days[e["peak_j"]] if e["peak_j"] >= 0 else "start"
        e["trough_day"] = days[e["trough_j"]]
        e["recover_day"] = days[e["recover_j"]] if e["recover_j"] is not None else ""
        e["sessions_to_trough"] = e["trough_j"] - e["peak_j"]
        e["sessions_to_recover"] = (e["recover_j"] - e["peak_j"]) if e["recover_j"] is not None else None
    return eps


def episode_ledger(name, inv, nav, bo, bc, days, dist, declines):
    """One row per cash spell. factor = strategy/base ratio after the spell / before it (= sold / re-bought)."""
    ratio = nav / bc
    rows = []
    for a, s in spells_of(inv):
        end = s if s is not None else len(days) - 1
        factor = ratio[end] / (ratio[a - 1] if a else 1.0)
        bought = bo[s] if s is not None else bc[-1]
        np.testing.assert_allclose(factor, bo[a] / bought, rtol=1e-9)
        cluster = "outside_decline"
        for k, e in enumerate(declines):
            last = e["recover_j"] if e["recover_j"] is not None else len(days)
            if e["peak_j"] < a <= last:
                cluster = f"D{k + 1}_{e['peak_day']}"
        rows.append(dict(rule=name, exit_fill=days[a], back_fill=days[s] if s is not None else "(open at end)",
                         sessions_out=(s if s is not None else len(days)) - a,
                         base_while_out=bought / bo[a] - 1, factor=factor, log_edge=float(np.log(factor)),
                         sig_dist_at_exit=dist[a], sig_dist_at_back=dist[s] if s is not None else np.nan,
                         cluster=cluster, in_sep_nov=days[a] >= "2026-09-01"))
    return rows


def timing_variants(inv, bo, bc):
    """NAVs under shifted fills. Keys describe what moved."""
    sp = spells_of(inv)
    late_all = np.r_[inv[0], inv[:-1]]
    exit_late, back_late = inv.copy(), inv.copy()
    for a, s in sp:
        exit_late[a] = True
        if s is not None:
            back_late[s] = False
    exit_late2, back_late2 = exit_late.copy(), back_late.copy()
    for a, s in sp:
        if a + 1 < len(inv) and (s is None or a + 1 < s):
            exit_late2[a + 1] = True
        if s is not None and s + 1 < len(inv):
            back_late2[s + 1] = False
    prev_c = np.r_[1.0, bc[:-1]]
    out = {"as_claimed": fast_nav(inv, bo, bc)[0], "all_fills_+1": fast_nav(late_all, bo, bc)[0],
           "all_fills_+2": fast_nav(np.r_[inv[:2], inv[:-2]], bo, bc)[0],
           "exit_+1_back_on_time": fast_nav(exit_late, bo, bc)[0],
           "exit_on_time_back_+1": fast_nav(back_late, bo, bc)[0],
           "exit_+2_back_on_time": fast_nav(exit_late2, bo, bc)[0],
           "exit_on_time_back_+2": fast_nav(back_late2, bo, bc)[0],
           # decision known before open j executed at the close of j (half a session late)
           "fill_at_next_close": np.cumprod(np.where(np.r_[True, inv[:-1]], bc / prev_c, 1.0)),
           # not executable for this user (05:00-06:00 KST): traded at the signal close itself
           "fill_at_signal_close(not executable)": np.cumprod(np.where(inv, bc / prev_c, 1.0))}
    return out


def random_schedules(durations, n, draws, rng, gap=1):
    """Uniform random placement of spells with the given lengths (order shuffled), >= gap sessions apart."""
    d = np.asarray(durations, int)
    k = len(d)
    free = n - d.sum() - (k - 1) * gap
    if k == 0 or free < 0:
        return np.ones((draws, n), bool)
    inv = np.ones((draws, n), bool)
    for r in range(draws):
        order = rng.permutation(d)
        cuts = np.sort(rng.choice(free + k, size=k, replace=False)) - np.arange(k)   # leading free sessions
        start = cuts + np.r_[0, np.cumsum(order[:-1] + gap)]
        for s, length in zip(start, order):
            inv[r, s:s + length] = False
    return inv


def pct(sample, value):
    sample = np.asarray(sample, float)
    return float((sample < value).mean() + .5 * (sample == value).mean())


def durations_of(inv):
    n = len(inv)
    return [(s if s is not None else n) - a for a, s in spells_of(inv)]


def losing_runs(factors):
    """Longest run of consecutive round trips that bought back higher, and the worst cumulative run product."""
    best_len, worst_prod, run, prod = 0, 1.0, 0, 1.0
    for f in factors:
        if f < 1:
            run, prod = run + 1, prod * f
            best_len, worst_prod = max(best_len, run), min(worst_prod, prod)
        else:
            run, prod = 0, 1.0
    return best_len, worst_prod


def worst_grind(inv, bo, bc):
    """Worst sequence of consecutive invested stints that each lost money (re-entered, fell again, left again)."""
    sp = spells_of(inv)
    stints = []                                      # invested stint between a re-entry and the next exit
    for (a, s), nxt in zip(sp, sp[1:] + [(None, None)]):
        if s is None:
            continue
        stop = bo[nxt[0]] if nxt[0] is not None else bc[-1]
        stints.append((s, nxt[0], stop / bo[s] - 1))
    worst, run, prod, first, span = (0, 1.0, None, None), 0, 1.0, None, None
    for s, e, r in stints:
        if r < 0:
            run, prod = run + 1, prod * (1 + r)
            first = s if run == 1 else first
            if prod < worst[1]:
                worst = (run, prod, first, e)
        else:
            run, prod = 0, 1.0
    return worst, stints


# ----------------------------------------------------------------------------- auxiliary SOXX/QQQ standalone
def frozen_prices():
    OUT.mkdir(parents=True, exist_ok=True)
    frames, meta = {}, OUT / "yf_meta.json"
    need = [s for s in SYMS if not (OUT / f"yf_{s}.csv").exists()]
    if need:
        import yfinance as yf
        for s in need:
            df = yf.download(s, start="2024-06-01", end="2026-11-19", auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "Close", "Adj Close"]].dropna()
            df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
            df.index.name = "date"
            df.to_csv(OUT / f"yf_{s}.csv")
        meta.write_text(json.dumps({"source": "yfinance daily, auto_adjust=False", "range": "2024-06-01..2026-11-18",
                                    "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
    for s in SYMS:
        frames[s] = pd.read_csv(OUT / f"yf_{s}.csv", index_col=0)
    return frames


def standalone(px, year, n, wait, mode, cross=False):
    """Hold the ETF itself; same exit/timeout rule on its own SMA; next-open fills; total-return prices."""
    adj = px["Adj Close"] / px["Close"]
    o, c = (px["Open"] * adj), px["Adj Close"]
    sig_c = px["Close"]
    sma = sig_c.rolling(n, min_periods=n).mean()
    below = (sig_c < sma).shift(1)
    idx = [d for d in px.index if d.startswith(str(year))]
    cond = below.loc[idx].to_numpy().astype(bool)
    if cross:                                        # (b): a fresh cross is needed even at the start
        prior = (sig_c < sma).shift(2).loc[idx].to_numpy().astype(bool)
        first = np.flatnonzero(cond & ~prior)
        start_cond = cond.copy()
        if first.size:
            start_cond[:first[0]] = False
        else:
            start_cond[:] = False
        cond = start_cond
    inv, sp = timeout_schedule(cond, wait, mode)
    prev_close = c.shift(1).loc[idx[0]]
    bo, bc = (o.loc[idx] / prev_close).to_numpy(), (c.loc[idx] / prev_close).to_numpy()
    nav = fast_nav(inv, bo, bc)[0]

    def stat(v):
        p = np.r_[1.0, v]
        return p[-1] - 1, (p / np.maximum.accumulate(p) - 1).min()
    r, d = stat(nav)
    br, bd = stat(bc)
    factors = [bo[a] / (bo[s] if s is not None else bc[-1]) for a, s, _ in sp]
    return dict(ret=r, mdd=d, bh_ret=br, bh_mdd=bd, exits=len(sp), out_share=float(1 - inv.mean()),
                won=int(sum(f > 1 for f in factors)), losing_run=losing_runs(factors)[0],
                losing_run_prod=losing_runs(factors)[1])


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = common.context()
    days, N = list(ctx.days), len(ctx.days)
    W = Windows(days)
    fx = np.asarray(ctx.fx, float)
    R = {"sessions": N, "first": days[0], "last": days[-1]}

    # ---- 0. base with the own loop
    base = own_replay(ctx, np.ones(N, bool))
    bc = base.usd.to_numpy()
    bo = np.asarray(ctx.baseline.value, float)                     # desk engine open NAV, checked below
    np.testing.assert_allclose(bc, np.asarray(ctx.baseline.close, float), rtol=1e-10)
    # own open NAV: close NAV of j-1 moved by the overnight leg is what the fast formula needs; verify it by
    # replaying "out for exactly one session" schedules against the share-level loop on 12 sessions.
    for j in rng.choice(np.arange(1, N - 1), 12, replace=False):
        probe = np.ones(N, bool)
        probe[j] = False
        np.testing.assert_allclose(own_replay(ctx, probe).usd.to_numpy(), fast_nav(probe, bo, bc)[0], rtol=1e-10)
    bm = {k: float(v[0]) for k, v in metrics(bc, fx, W).items()}
    R["base"] = bm
    print("BASE (own loop):", {k: round(v, 4) for k, v in bm.items()})

    # ---- 1. re-implementation vs saved ledgers
    cond10, dist10 = below_prev(ctx, "SOXX", 10)
    cond50, dist50 = below_prev(ctx, "SOXX", 50)
    assert np.array_equal(~cond10, ctx.gates["SOXX10r1"]) and np.array_equal(~cond50, ctx.gates["SOXX50r1"])
    inv_a, sp_a = timeout_schedule(cond10, 5, "orgate")
    inv_b = rule_b(ctx)
    inv_b_generic, _ = timeout_schedule(cond50, 20, "pure")
    led = {"a": own_replay(ctx, inv_a), "b": own_replay(ctx, inv_b)}
    saved = {"a": ("reentry", "E1_R1_wait5_orgate"), "b": ("oos", "sr_soxx50_cash20")}
    R["reproduction"] = {}
    for k, (fam, key) in saved.items():
        ref = pd.read_parquet(common.CACHE / f"{fam}_daily.parquet")
        ref = ref[ref.strategy == key].set_index("date")
        assert list(ref.index) == days, f"{key}: sessions differ"
        diff = {c: float(np.abs(led[k][c].to_numpy() - ref[c].to_numpy()).max()) for c in led[k].columns}
        fast = float(np.abs(fast_nav(inv_a if k == "a" else inv_b, bo, bc)[0] - led[k].usd.to_numpy()).max())
        R["reproduction"][key] = {"max_abs_diff_by_column": diff, "fast_formula_vs_own_loop": fast,
                                  "turn_total": float(led[k].turn.sum())}
        print(f"REPRO {key}: max|diff| {max(diff.values()):.2e} (cols {diff}); fast-vs-loop {fast:.2e}")
    R["reproduction"]["b_equals_generic_n50_wait20_pure"] = bool(np.array_equal(inv_b, inv_b_generic))
    print("(b) own loop == generic timeout(SMA50, 20, pure, start armed):", np.array_equal(inv_b, inv_b_generic),
          "| sessions differing:", int((inv_b != inv_b_generic).sum()))
    navs = {"a": led["a"].usd.to_numpy(), "b": led["b"].usd.to_numpy()}
    for k in navs:
        m = {x: float(v[0]) for x, v in metrics(navs[k], fx, W).items()}
        m["expo_full"] = float(led[k].exposure.mean())
        m["turn_full"] = float(led[k].turn.sum())
        R[f"claim_{k}"] = m
        print(f"CLAIM ({k}) own numbers:", {x: round(v, 4) for x, v in m.items()})
    log = subprocess.run(["git", "-C", str(common.ROOT), "log", "--date=iso", "--format=%h %ad %s", "--diff-filter=A",
                          "--", "data/derived/analysis/stop_research/code/v2_soxx50_20d.py"],
                         capture_output=True, text=True, encoding="utf-8").stdout.strip()
    log2 = subprocess.run(["git", "-C", str(common.ROOT), "log", "--date=iso", "--format=%h %ad", "-1", "fa9c4d89"],
                          capture_output=True, text=True, encoding="utf-8").stdout.strip()
    R["freeze"] = {"docs_commit": log2, "rule_code_first_in_git": log.split(" (")[0][:60]}
    print("FREEZE docs commit:", log2, "| v2_soxx50_20d.py first in git:", log[:40])

    # where (a) sits in the lead's 6,643-ledger table under the stated selection rule
    ms = pd.read_csv(common.OUT / "merged_stats.csv", index_col=0)
    b_row = ms.loc["base"]
    pool = ms[(ms.jan_aug_krw_mdd >= b_row.jan_aug_krw_mdd - 1e-7)].sort_values("jan_aug_krw", ascending=False)
    top = pool.head(20)
    R["selection_pool"] = {"ledgers": len(ms), "pool": len(pool), "top1": pool.index[0],
                           "top20_sep_nov_usd_median": float(top.sep_nov_usd.median()),
                           "top20_sep_nov_usd_min": float(top.sep_nov_usd.min()),
                           "top20_sep_nov_usd_max": float(top.sep_nov_usd.max()),
                           "top20_beat_base_sep_nov": int((top.sep_nov_usd > b_row.sep_nov_usd).sum()),
                           "pool_beat_base_sep_nov_share": float((pool.sep_nov_usd > b_row.sep_nov_usd).mean()),
                           "all_ledgers_beat_base_sep_nov_share": float((ms.sep_nov_usd > b_row.sep_nov_usd).mean()),
                           "all_ledgers_sep_nov_median": float(ms.sep_nov_usd.median())}
    print("\nSELECTION POOL (Jan-Aug KRW MDD no deeper than base), top 12 by Jan-Aug KRW:")
    print(top.head(12)[["jan_aug_krw", "jan_aug_krw_mdd", "sep_nov_usd", "full_usd", "full_usd_mdd", "full_turn"]].round(3))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in R["selection_pool"].items()})

    # ---- 2. episode ledgers
    declines = base_declines(bc, days)
    print("\nBASE DECLINES >= 8% (own HWM):")
    for e in declines:
        row = {k: e[k] for k in ("peak_day", "trough_day", "recover_day", "sessions_to_trough", "sessions_to_recover")}
        row["depth"] = round(e["depth"], 4)
        for k, inv in (("a", inv_a), ("b", inv_b)):
            nav = navs[k]
            p = nav[e["peak_j"]] if e["peak_j"] >= 0 else 1.0
            row[f"{k}_peak_to_trough"] = round(nav[e["trough_j"]] / p - 1, 4)
            exits = [a for a, _ in spells_of(inv) if e["peak_j"] < a <= e["trough_j"]]
            row[f"{k}_exits"] = len(exits)
            if exits:
                row[f"{k}_first_exit_lag"] = exits[0] - e["peak_j"]
                row[f"{k}_base_lost_before_exit"] = round(bo[exits[0]] / (bc[e["peak_j"]] if e["peak_j"] >= 0 else 1) - 1, 4)
        e["rules"] = row
        print(row)
    R["base_declines"] = [e["rules"] for e in declines]
    ledgers = {}
    for k, inv, dist in (("a", inv_a, dist10), ("b", inv_b, dist50)):
        rows = pd.DataFrame(episode_ledger(k, inv, navs[k], bo, bc, days, dist, declines))
        ledgers[k] = rows
        total = float(rows.log_edge.sum())
        np.testing.assert_allclose(np.exp(total), navs[k][-1] / bc[-1], rtol=1e-9)
        rows["share_of_full_log_edge"] = rows.log_edge / total
        sn = rows[rows.in_sep_nov]
        sn_edge = float(np.log((1 + R[f"claim_{k}"]["sep_nov_usd"]) / (1 + bm["sep_nov_usd"])))
        srt = rows.log_edge.sort_values(ascending=False)
        by_cluster = rows.groupby("cluster").agg(trips=("factor", "size"), factor=("factor", "prod"),
                                                 log_edge=("log_edge", "sum"), sessions_out=("sessions_out", "sum"))
        R[f"episodes_{k}"] = {
            "trips": len(rows), "won": int((rows.factor > 1).sum()), "total_log_edge": total,
            "vs_base_final": float(np.exp(total)), "top1_share": float(srt.iloc[0] / total),
            "top2_share": float(srt.iloc[:2].sum() / total), "top3_share": float(srt.iloc[:3].sum() / total),
            "sum_of_losing_log_edge": float(rows.log_edge[rows.log_edge < 0].sum()),
            "sep_nov_log_edge_total": sn_edge, "sep_nov_trips": len(sn),
            "sep_nov_log_edge_from_trips": float(sn.log_edge.sum()),
            "sep_nov_top1_share": float(sn.log_edge.max() / sn_edge) if len(sn) else None,
            "edge_without_best_trip_vs_base_final": float(np.exp(total - srt.iloc[0])),
            "edge_without_best2_vs_base_final": float(np.exp(total - srt.iloc[:2].sum())),
            "clusters": by_cluster.round(4).reset_index().to_dict("records"),
            "rows": rows.round(4).to_dict("records")}
        print(f"\nEPISODES ({k}):")
        print(rows.drop(columns=["rule"]).round(4).to_string(index=False))
        print(by_cluster.round(4))
        print({x: (round(v, 4) if isinstance(v, float) else v) for x, v in R[f"episodes_{k}"].items()
               if x not in ("rows", "clusters")})

    # ---- 3. timing fragility
    R["timing"] = {}
    for k, inv in (("a", inv_a), ("b", inv_b)):
        table = {}
        for name, nav in timing_variants(inv, bo, bc).items():
            m = metrics(nav, fx, W)
            table[name] = {x: float(m[x][0]) for x in ("full_usd", "full_mdd", "jan_aug_krw", "h2_usd", "sep_nov_usd",
                                                       "jul", "octnov", "post")}
        R["timing"][k] = table
        print(f"\nTIMING ({k}):")
        print(pd.DataFrame(table).T.round(4))

    # ---- 4. neighbourhood heat map (own implementation, every cell replayed share-level once)
    cells, invs = [], {}
    for sym in SYMS:
        for n in NS:
            cond, _ = below_prev(ctx, sym, n)
            for wait in WAITS + (None,):
                for mode in (MODES if wait is not None else ("gate",)):
                    inv, sp = timeout_schedule(cond, wait, mode)
                    key = f"{sym}{n}_w{wait if wait is not None else 'inf'}_{mode}"
                    nav = fast_nav(inv, bo, bc)[0]
                    full = own_replay(ctx, inv)
                    np.testing.assert_allclose(full.usd.to_numpy(), nav, rtol=1e-9)
                    m = {x: float(v[0]) for x, v in metrics(nav, fx, W).items()}
                    late = metrics(fast_nav(np.r_[inv[0], inv[:-1]], bo, bc)[0], fx, W)
                    cost = np.cumprod(1 - .001 * full.turn.to_numpy())
                    m.update(sym=sym, n=n, wait=wait if wait is not None else 999, mode=mode, exits=len(sp),
                             out_share=float(1 - inv.mean()), turn=float(full.turn.sum()),
                             h1_keep=m["h1_usd"] / bm["h1_usd"],
                             full_usd_delay1=float(late["full_usd"][0]), sep_nov_usd_delay1=float(late["sep_nov_usd"][0]),
                             jul_delay1=float(late["jul"][0]), octnov_delay1=float(late["octnov"][0]),
                             h1_keep_delay1=float(late["h1_usd"][0]) / bm["h1_usd"],
                             full_usd_10bp=float(nav[-1] * cost[-1] - 1),
                             won=int(sum(bo[a] / (bo[s] if s is not None else bc[-1]) > 1 for a, s, _ in sp)))
                    ctl = common.matched_control(base, float(full.exposure.mean()), "full")
                    m["full_control_usd"], m["full_control_mdd"] = float(ctl[0]), float(ctl[1])
                    cells.append({"key": key, **m})
                    invs[key] = inv
    H = pd.DataFrame(cells).set_index("key")
    grid = H[H["mode"] != "gate"].copy()
    assert len(grid) == len(SYMS) * len(NS) * len(WAITS) * len(MODES)
    for frame in (H, grid):
        frame["both_declines"] = (frame.jul > bm["jul"]) & (frame.octnov > bm["octnov"])
        frame["pass"] = frame.both_declines & (frame.h1_keep >= .9)
        frame["pass_delay1"] = ((frame.jul_delay1 > bm["jul"]) & (frame.octnov_delay1 > bm["octnov"])
                                & (frame.h1_keep_delay1 >= .9))
        frame["pool"] = frame.jan_aug_krw_mdd >= bm["jan_aug_krw_mdd"] - 1e-7
    H.to_csv(OUT / "heatmap.csv")
    np.testing.assert_allclose(H.at["SOXX10_w5_orgate", "full_usd"], R["claim_a"]["full_usd"], rtol=1e-9)
    print("\nHEAT MAP (percent; rows SMA n, columns wait; 'inf' = plain gate reference, not in the 224)")
    for sym in SYMS:
        for mode in MODES:
            part = H[(H.sym == sym) & ((H["mode"] == mode) | (H["mode"] == "gate"))]
            for col in ("full_usd", "full_mdd", "jan_aug_krw", "sep_nov_usd", "jul", "octnov", "h1_keep", "pass",
                        "full_usd_delay1"):
                piv = part.pivot(index="n", columns="wait", values=col)
                piv = piv.astype(float).mul(100).round(0).astype(int) if col != "pass" else piv.astype(int)
                print(f"-- {sym} {mode} {col}  (base {bm.get(col, float('nan')) * 100:.0f})"
                      if col in bm else f"-- {sym} {mode} {col}")
                print(piv.rename(columns={999: "inf"}).to_string())
    counts = {"cells": len(grid), "distinct_schedules": len({invs[k].tobytes() for k in grid.index})}
    for label, part in (("all", grid), ("SOXX", grid[grid.sym == "SOXX"]), ("QQQ", grid[grid.sym == "QQQ"]),
                        ("SOXX_pure", grid[(grid.sym == "SOXX") & (grid["mode"] == "pure")]),
                        ("SOXX_orgate", grid[(grid.sym == "SOXX") & (grid["mode"] == "orgate")])):
        counts[label] = {
            "cells": len(part), "beat_base_jul": int((part.jul > bm["jul"]).sum()),
            "beat_base_octnov": int((part.octnov > bm["octnov"]).sum()),
            "both_declines": int(part.both_declines.sum()), "h1_keep_90": int((part.h1_keep >= .9).sum()),
            "pass(both+h1keep)": int(part["pass"].sum()), "pass_delay1": int(part.pass_delay1.sum()),
            "pass_and_delay1": int((part["pass"] & part.pass_delay1).sum()),
            "full_usd_above_base": int((part.full_usd > bm["full_usd"]).sum()),
            "full_usd_above_base_delay1": int((part.full_usd_delay1 > bm["full_usd"]).sum()),
            "full_above_matched_control": int((part.full_usd > part.full_control_usd).sum()),
            "sep_nov_above_base": int((part.sep_nov_usd > bm["sep_nov_usd"]).sum()),
            "sep_nov_positive": int((part.sep_nov_usd > 0).sum()),
            "mdd_shallower_than_base": int((part.full_mdd > bm["full_mdd"]).sum()),
            "median_full_usd": float(part.full_usd.median()), "median_sep_nov_usd": float(part.sep_nov_usd.median()),
            "median_full_mdd": float(part.full_mdd.median())}
    for claim, key in (("a", "SOXX10_w5_orgate"), ("b_generic", "SOXX50_w20_pure")):
        r = grid.loc[key]
        counts[f"rank_{claim}"] = {
            "key": key, "pass": bool(r["pass"]), "pass_delay1": bool(r.pass_delay1),
            "rank_full_usd_of_224": int((grid.full_usd > r.full_usd).sum() + 1),
            "rank_sep_nov_usd": int((grid.sep_nov_usd > r.sep_nov_usd).sum() + 1),
            "rank_jan_aug_krw": int((grid.jan_aug_krw > r.jan_aug_krw).sum() + 1),
            "rank_full_mdd": int((grid.full_mdd > r.full_mdd).sum() + 1),
            "rank_full_usd_delay1": int((grid.full_usd_delay1 > r.full_usd_delay1).sum() + 1)}
    # 8 immediate neighbours of the claimed cell on the SOXX orgate sheet
    def neighbours(sym, mode, n, wait):
        i, j = NS.index(n), WAITS.index(wait)
        keys = [f"{sym}{NS[a]}_w{WAITS[b]}_{mode}" for a in range(max(0, i - 1), min(len(NS), i + 2))
                for b in range(max(0, j - 1), min(len(WAITS), j + 2)) if (a, b) != (i, j)]
        return grid.loc[keys]
    nb = neighbours("SOXX", "orgate", 10, 5)
    counts["neighbours_a"] = {"n": len(nb), "full_usd_min": float(nb.full_usd.min()),
                              "full_usd_median": float(nb.full_usd.median()), "full_usd_max": float(nb.full_usd.max()),
                              "sep_nov_min": float(nb.sep_nov_usd.min()), "sep_nov_median": float(nb.sep_nov_usd.median()),
                              "mdd_worst": float(nb.full_mdd.min()), "pass": int(nb["pass"].sum()),
                              "above_base_full": int((nb.full_usd > bm["full_usd"]).sum())}
    counts["pass_cells"] = sorted(grid.index[grid["pass"]])
    counts["pass_and_delay1_cells"] = sorted(grid.index[grid["pass"] & grid.pass_delay1])
    sel = grid[grid.pool].sort_values("jan_aug_krw", ascending=False)
    counts["protocol_pick_in_grid"] = {"key": sel.index[0] if len(sel) and sel.jan_aug_krw.iloc[0] > bm["jan_aug_krw"] else "base",
                                       "pool": int(grid.pool.sum()),
                                       "top5": sel.head(5)[["jan_aug_krw", "sep_nov_usd", "full_usd"]].round(4).reset_index().to_dict("records")}
    R["heatmap"] = counts
    print("\nHEAT MAP COUNTS:")
    for k, v in counts.items():
        print(" ", k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items()} if isinstance(v, dict) else v)

    # ---- 5. placebos
    R["placebo"] = {}
    for k, inv in (("a", inv_a), ("b", inv_b)):
        dur = durations_of(inv)
        rnd = fast_nav(random_schedules(dur, N, 2000, rng), bo, bc)
        pm = metrics(rnd, fx, W)
        act = {x: float(v[0]) for x, v in metrics(navs[k], fx, W).items()}
        roll = np.array([np.roll(inv, s) for s in range(1, N)])
        rm = metrics(fast_nav(roll, bo, bc), fx, W)
        far = np.array([min(s, N - s) > 5 for s in range(1, N)])
        res = {"exits": len(dur), "durations": dur, "sessions_out": int(sum(dur))}
        for x in ("full_usd", "h2_usd", "sep_nov_usd", "jan_aug_krw", "full_mdd", "jul", "octnov", "post"):
            res[x] = {"actual": act[x], "random_pct": pct(pm[x], act[x]), "random_median": float(np.median(pm[x])),
                      "random_p95": float(np.quantile(pm[x], .95)), "random_p05": float(np.quantile(pm[x], .05)),
                      "shift_pct": pct(rm[x], act[x]), "shift_pct_excl_5": pct(rm[x][far], act[x])}
        res["random_beat_base_full_share"] = float((pm["full_usd"] > bm["full_usd"]).mean())
        res["random_both_declines_share"] = float(((pm["jul"] > bm["jul"]) & (pm["octnov"] > bm["octnov"])).mean())
        R["placebo"][k] = res
        print(f"\nPLACEBO ({k}) exits={len(dur)} durations={dur}")
        print(pd.DataFrame({x: v for x, v in res.items() if isinstance(v, dict)}).T.round(4))
        print({x: round(v, 4) for x, v in res.items() if isinstance(v, float)})

    # family-wise: best cell of the 224 vs best of 224 random schedules with the same spell lengths
    def family_stats(M):
        """M: dict metric -> array (draws, cells)."""
        ok = (M["jan_aug_krw_mdd"] >= bm["jan_aug_krw_mdd"] - 1e-7) & (M["jan_aug_krw"] > bm["jan_aug_krw"])
        score = np.where(ok, M["jan_aug_krw"], -np.inf)
        pick = score.argmax(axis=1)
        none = ~ok.any(axis=1)
        take = lambda x, default: np.where(none, default, np.take_along_axis(M[x], pick[:, None], 1)[:, 0])
        passed = (M["jul"] > bm["jul"]) & (M["octnov"] > bm["octnov"]) & (M["h1_usd"] / bm["h1_usd"] >= .9)
        return {"best_full_usd": M["full_usd"].max(1), "best_sep_nov_usd": M["sep_nov_usd"].max(1),
                "best_h2_usd": M["h2_usd"].max(1), "best_full_mdd": M["full_mdd"].max(1),
                "best_jan_aug_krw_in_pool": np.where(none, bm["jan_aug_krw"], score.max(1)),
                "pass_cells": passed.sum(1).astype(float),
                "cells_full_above_base": (M["full_usd"] > bm["full_usd"]).sum(1).astype(float),
                "cells_sep_nov_positive": (M["sep_nov_usd"] > 0).sum(1).astype(float),
                "picked_sep_nov_usd": take("sep_nov_usd", bm["sep_nov_usd"]),
                "picked_full_usd": take("full_usd", bm["full_usd"]),
                "picked_full_mdd": take("full_mdd", bm["full_mdd"])}
    keys = list(grid.index)
    stack = lambda fn: {x: np.stack([fn(k)[x] for k in keys], axis=1) for x in
                        ("full_usd", "full_mdd", "jan_aug_krw", "jan_aug_krw_mdd", "sep_nov_usd", "h1_usd", "h2_usd",
                         "jul", "octnov")}
    actual = family_stats(stack(lambda k: metrics(fast_nav(invs[k], bo, bc), fx, W)))
    DR = 300
    cache = {k: metrics(fast_nav(random_schedules(durations_of(invs[k]), N, DR, rng), bo, bc), fx, W) for k in keys}
    rand = family_stats(stack(lambda k: cache[k]))
    cache2 = {k: metrics(fast_nav(np.array([np.roll(invs[k], s) for s in range(1, N)]), bo, bc), fx, W) for k in keys}
    shift = family_stats(stack(lambda k: cache2[k]))
    far = np.array([min(s, N - s) > 5 for s in range(1, N)])
    fam = {}
    for x in actual:
        a = float(actual[x][0])
        fam[x] = {"actual": a, "random_pct": pct(rand[x], a), "random_median": float(np.median(rand[x])),
                  "random_p95": float(np.quantile(rand[x], .95)),
                  "joint_shift_pct_excl_5": pct(shift[x][far], a), "joint_shift_median": float(np.median(shift[x][far])),
                  "joint_shift_p95": float(np.quantile(shift[x][far], .95))}
    R["placebo"]["family"] = {"cells": len(keys), "random_draws": DR, "joint_shifts": int(far.sum()), **fam}
    print("\nFAMILY-WISE PLACEBO (224 cells; random = independent placements with each cell's own spell lengths, "
          f"{DR} draws; joint shift = whole family rolled by the same k, |k|>5, {int(far.sum())} shifts)")
    print(pd.DataFrame(fam).T.round(4))

    # ---- 6. costs
    R["costs"] = {}
    rows = {}
    for k, frame in (("base", base), ("a", led["a"]), ("b", led["b"])):
        for bp in (0, 10, 20):
            nav = frame.usd.to_numpy() * np.cumprod(1 - bp / 1e4 * frame.turn.to_numpy())
            m = metrics(nav, fx, W)
            rows[f"{k}_{bp}bp"] = {"turn_full": float(frame.turn.sum()),
                                   **{x: float(m[x][0]) for x in ("full_usd", "full_mdd", "jan_aug_krw", "h2_usd",
                                                                  "sep_nov_usd", "jul", "octnov")}}
    R["costs"] = rows
    print("\nCOSTS (one-way):")
    print(pd.DataFrame(rows).T.round(4))
    for k in ("a", "b"):
        e = float(led[k].exposure.mean())
        R[f"matched_control_{k}"] = {w: [float(v) for v in common.matched_control(base, float(
            led[k].exposure[(led[k].index >= common.WINDOWS[w][0]) & (led[k].index <= common.WINDOWS[w][1])].mean()), w)]
            for w in ("full", "h2", "sep_nov")}
        R[f"matched_control_{k}"]["exposure_full"] = e
        print(f"MATCHED CONTROL ({k}) [ret, mdd] same average exposure of base:",
              {w: [round(x, 4) for x in v] if isinstance(v, list) else round(v, 4)
               for w, v in R[f"matched_control_{k}"].items()})

    # ---- 7. auxiliary 2025 / 2026 standalone on SOXX and QQQ (yfinance, frozen)
    px = frozen_prices()
    chk = {}
    for s in SYMS:
        bk = ctx.book.cls[s].reindex(days).astype(float)
        yf_c = px[s]["Close"].reindex(days)
        yf_a = px[s]["Adj Close"].reindex(days)
        chk[s] = {"rows": len(px[s]), "first": px[s].index[0], "last": px[s].index[-1],
                  "missing_2026_sessions": int(yf_c.isna().sum()),
                  "max_rel_diff_book_vs_close": float((bk / yf_c - 1).abs().max()),
                  "max_rel_diff_book_vs_adj": float((bk / yf_a - 1).abs().max())}
    R["yf_check"] = chk
    print("\nYF CHECK vs book closes:", chk)
    aux_rows, aux_grid = {}, []
    named = {"a_n10_w5_orgate": (10, 5, "orgate", False), "n10_w3_orgate": (10, 3, "orgate", False),
             "n10_w10_orgate": (10, 10, "orgate", False), "n10_w5_pure": (10, 5, "pure", False),
             "n10_plain_gate": (10, None, "gate", False), "n20_w5_orgate": (20, 5, "orgate", False),
             "b_n50_w20_cross": (50, 20, "pure", True), "n50_plain_gate": (50, None, "gate", False)}
    for s in SYMS:
        for year in (2025, 2026):
            for name, (n, wait, mode, cross) in named.items():
                aux_rows[f"{s}_{year}_{name}"] = standalone(px[s], year, n, wait, mode, cross)
            for n in NS:
                for wait in WAITS:
                    for mode in MODES:
                        r = standalone(px[s], year, n, wait, mode)
                        aux_grid.append(dict(sym=s, year=year, n=n, wait=wait, mode=mode, **r))
    A = pd.DataFrame(aux_rows).T
    G = pd.DataFrame(aux_grid)
    G["beat_ret"], G["beat_mdd"] = G.ret > G.bh_ret, G.mdd > G.bh_mdd
    G["beat_both"] = G.beat_ret & G.beat_mdd
    summary = G.groupby(["sym", "year"]).agg(cells=("ret", "size"), bh_ret=("bh_ret", "first"), bh_mdd=("bh_mdd", "first"),
                                             beat_ret=("beat_ret", "sum"), beat_mdd=("beat_mdd", "sum"),
                                             beat_both=("beat_both", "sum"), median_ret=("ret", "median"),
                                             median_mdd=("mdd", "median"), best_ret=("ret", "max"))
    both_years = G.pivot_table(index=["sym", "n", "wait", "mode"], columns="year", values="beat_both").astype(bool)
    R["aux"] = {"named": A.round(4).to_dict("index"), "grid_summary": summary.round(4).reset_index().to_dict("records"),
                "cells_beating_bh_both_metrics_in_2025_and_2026": {s: int((both_years.loc[s][2025] & both_years.loc[s][2026]).sum())
                                                                    for s in SYMS},
                "cells_beating_bh_return_in_2025_and_2026": {
                    s: int((G[(G.sym == s) & (G.year == 2025)].set_index(["n", "wait", "mode"]).beat_ret
                            & G[(G.sym == s) & (G.year == 2026)].set_index(["n", "wait", "mode"]).beat_ret).sum())
                    for s in SYMS}}
    G.to_csv(OUT / "aux_grid.csv", index=False)
    print("\nAUX standalone (hold the ETF itself, own SMA, next-open fills, total return):")
    print(A.astype(float).round(4).to_string())
    print(summary.round(4).to_string())
    print({k: v for k, v in R["aux"].items() if k.startswith("cells_")})
    for s in SYMS:
        for year in (2025, 2026):
            part = G[(G.sym == s) & (G.year == year) & (G["mode"] == "orgate")]
            print(f"-- {s} {year} orgate: excess return vs buy&hold (pct points)")
            print(part.assign(x=(part.ret - part.bh_ret) * 100).pivot(index="n", columns="wait", values="x").round(0).astype(int).to_string())

    # ---- 8. mechanism: whipsaw and slow-grind sequences
    R["mechanism"] = {}
    for k, inv in (("a", inv_a), ("b", inv_b)):
        f = ledgers[k].factor.to_numpy()
        run, prod = losing_runs(f)
        worst, stints = worst_grind(inv, bo, bc)
        R["mechanism"][k] = {
            "trips": len(f), "bought_back_lower": int((f > 1).sum()), "bought_back_higher": int((f < 1).sum()),
            "product_of_winning_trips": float(np.prod(f[f > 1])), "product_of_losing_trips": float(np.prod(f[f < 1])),
            "longest_losing_trip_run": run, "worst_losing_run_product": prod,
            "worst_consecutive_losing_stints": {"count": worst[0], "cumulative": worst[1] - 1,
                                                "from": days[worst[2]] if worst[2] is not None else "",
                                                "to": days[worst[3]] if worst[3] is not None else "(end)"},
            "invested_stints": [(days[s], days[e] if e is not None else "(end)", round(r, 4)) for s, e, r in stints]}
        print(f"\nMECHANISM ({k}):", {x: (round(v, 4) if isinstance(v, float) else v)
                                     for x, v in R["mechanism"][k].items() if x != "invested_stints"})
        print("  invested stints (re-entry fill, next exit fill, base open-to-open):", R["mechanism"][k]["invested_stints"])
    # same statistic across the grid: how bad can the buy-back-higher sequence get in 2026 data
    grind = []
    for key in grid.index:
        inv = invs[key]
        f = [bo[a] / (bo[s] if s is not None else bc[-1]) for a, s in spells_of(inv)]
        run, prod = losing_runs(f)
        worst, _ = worst_grind(inv, bo, bc)
        grind.append(dict(key=key, losing_run=run, losing_run_prod=prod, grind_count=worst[0], grind_cum=worst[1] - 1,
                          whipsaw_cost=float(np.prod([x for x in f if x < 1])) - 1))
    Gr = pd.DataFrame(grind).set_index("key")
    Gr.to_csv(OUT / "grind.csv")
    worst_key = Gr.grind_cum.idxmin()
    R["mechanism"]["grid"] = {"worst_grind_cell": worst_key, **Gr.loc[worst_key].round(4).to_dict(),
                              "median_whipsaw_cost": float(Gr.whipsaw_cost.median()),
                              "worst_whipsaw_cost_cell": Gr.whipsaw_cost.idxmin(),
                              "worst_whipsaw_cost": float(Gr.whipsaw_cost.min()),
                              "soxx_orgate_n10_row": Gr.loc[[f"SOXX10_w{w}_orgate" for w in WAITS]].round(4).reset_index().to_dict("records")}
    print("GRID grind:", {k: v for k, v in R["mechanism"]["grid"].items() if k != "soxx_orgate_n10_row"})
    print(Gr.loc[[f"SOXX10_w{w}_orgate" for w in WAITS] + ["SOXX50_w20_pure"]].round(4))

    # ---- 9. where the edge sits inside a round trip: the first session out and the re-entry session itself
    oo = np.r_[bo[1:], bc[-1]] / bo                                # base open j -> open j+1 (last: to the close)
    extra = {}
    for k, inv in (("a", inv_a), ("b", inv_b)):
        sp = spells_of(inv)
        first_out = np.array([a for a, _ in sp])
        back = np.array([s for _, s in sp if s is not None])
        draw = np.array([np.prod(oo[rng.choice(N, len(back), replace=False)]) for _ in range(5000)])
        res = {"first_out_sessions": len(first_out), "base_product_on_first_out_sessions": float(np.prod(oo[first_out]) - 1),
               "reentry_sessions": len(back), "base_product_on_reentry_sessions": float(np.prod(oo[back]) - 1),
               "base_mean_on_reentry_sessions": float(oo[back].mean() - 1), "base_mean_all_sessions": float(oo.mean() - 1),
               "reentry_sessions_up": int((oo[back] > 1).sum()),
               "pct_vs_random_sessions": pct(draw, float(np.prod(oo[back]))),
               "reentry_session_returns": [(days[s], round(float(oo[s] - 1), 4)) for s in back]}
        # the delayed variants against the same random-schedule placebo
        rnd = metrics(fast_nav(random_schedules(durations_of(inv), N, 2000, rng), bo, bc), fx, W)
        for name in ("all_fills_+1", "exit_+1_back_on_time", "exit_on_time_back_+1", "fill_at_next_close"):
            t = R["timing"][k][name]
            res[f"random_pct_{name}"] = {x: pct(rnd[x], t[x]) for x in ("full_usd", "h2_usd", "sep_nov_usd", "full_mdd")}
        # longest run of consecutive losing invested stints (re-entered, lost, left again)
        _, stints = worst_grind(inv, bo, bc)
        best, run, prod, start = (0, 1.0, None), 0, 1.0, None
        for s, e, r in stints:
            if r < 0:
                run, prod = run + 1, prod * (1 + r)
                start = s if run == 1 else start
                if run > best[0]:
                    best = (run, prod, start)
            else:
                run, prod = 0, 1.0
        res["longest_losing_stint_run"] = {"count": best[0], "cumulative": best[1] - 1,
                                           "from": days[best[2]] if best[2] is not None else ""}
        extra[k] = res
        print(f"\nEXTRA ({k}):", {x: (round(v, 4) if isinstance(v, float) else v) for x, v in res.items()})
    region = H[(H.sym == "SOXX") & H.n.isin((8, 10, 12)) & (H["mode"] == "orgate")]
    extra["soxx_orgate_n8_12_region"] = {
        "cells": len(region), "full_above_base": int((region.full_usd > bm["full_usd"]).sum()),
        "full_above_base_delay1": int((region.full_usd_delay1 > bm["full_usd"]).sum()),
        "full_usd_min": float(region.full_usd.min()), "full_usd_max": float(region.full_usd.max()),
        "full_usd_delay1_min": float(region.full_usd_delay1.min()), "full_usd_delay1_max": float(region.full_usd_delay1.max()),
        "full_usd_delay1_median": float(region.full_usd_delay1.median()),
        "sep_nov_positive": int((region.sep_nov_usd > 0).sum()),
        "sep_nov_positive_delay1": int((region.sep_nov_usd_delay1 > 0).sum()),
        "sep_nov_above_base_delay1": int((region.sep_nov_usd_delay1 > bm["sep_nov_usd"]).sum()),
        "h1_keep_90_delay1": int((region.h1_keep_delay1 >= .9).sum()),
        "distinct_schedules": len({invs[k].tobytes() for k in region.index}),
        "distinct_exit_date_sets": len({tuple(a for a, _ in spells_of(invs[k])) for k in region.index})}
    saved_detail = pd.read_csv(common.CACHE / "reentry_detail.csv", index_col=0)
    extra["saved_delay1_full_usd_a"] = float(saved_detail.at["E1_R1_wait5_orgate", "full_usd_delay1"])
    print("REGION SOXX orgate n 8-12:", {x: (round(v, 4) if isinstance(v, float) else v)
                                         for x, v in extra["soxx_orgate_n8_12_region"].items()})
    print("saved reentry_detail full_usd_delay1 for (a):", round(extra["saved_delay1_full_usd_a"], 4),
          "| mine:", round(R["timing"]["a"]["all_fills_+1"]["full_usd"], 4))
    R["extra"] = extra

    R["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT / "result.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("\nwrote", OUT / "result.json", "runtime", R["runtime_seconds"])


if __name__ == "__main__":
    main()
