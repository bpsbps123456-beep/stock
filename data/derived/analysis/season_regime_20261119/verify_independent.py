"""Independent re-implementation of the season-regime basket switch (Rule A, Rule B) on frozen data.

Own share-level loop. Inputs come only from the read-only harness
(../desk_defense_check_20261119/common.py) and desk.strategies (project_caps, iso_week,
momentum_ranking); nothing in this folder is imported.

  step 1  MOM, REV, Rule A, Rule B, Rule A late: USD returns, drawdown, exposure, switch dates
  step 2  point-in-time attack: book prices from row start+150 on are scrambled in memory;
          Known-IC and REV targets for sessions <= 150 must not move
  step 3  (--compare) read-only comparison with out/meta_candidates.csv, meta_robustness.csv,
          meta_focus_choices.csv and cache/meta_daily.parquet (no code from this folder is run)

  py data/derived/analysis/season_regime_20261119/verify_independent.py [--compare]
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]  # never import this folder's modules
sys.path.insert(0, str(HERE.parent / "desk_defense_check_20261119"))
import common  # noqa: E402  read-only harness: frozen book, desk targets, gates
from desk import strategies as S  # noqa: E402

WINDOWS = {"full": ("2026-01-01", "2026-11-17"), "jul_nov": ("2026-07-01", "2026-11-17"),
           "0623_0729": ("2026-06-23", "2026-07-29"), "0730_0922": ("2026-07-30", "2026-09-22"),
           "1013_1117": ("2026-10-13", "2026-11-17")}
N_REV, MIN_NAMES, PIT_CUT, SEED = 7, 8, 150, 20261119


# --- share-level books ------------------------------------------------------------------

def mark(shares, px):
    held = shares != 0
    if not np.isfinite(px[held]).all():
        raise ValueError("held name without a market price")
    return float(shares[held] @ px[held])


def buy(nav, w, o):
    """Shares for weights w of nav at open prices o; weight below 1 stays USD cash."""
    need = w > 0
    if not (np.isfinite(o[need]) & (o[need] > 0)).all():
        raise ValueError("target name without an executable open")
    shares = np.zeros_like(w)
    shares[need] = nav * w[need] / o[need]
    return shares, nav * (1 - w.sum())


def basket(opn, close, targets):
    """Shadow portfolio from session 0: rebalance to targets[j] at the open of j, drift otherwise."""
    n, m = opn.shape
    shares, cash = np.zeros(m), 1.0
    w_open, nav_close = np.zeros((n, m)), np.zeros(n)
    for j in range(n):
        nav = cash + mark(shares, opn[j])
        if j in targets:
            shares, cash = buy(nav, targets[j], opn[j])
        held = shares != 0
        w_open[j, held] = shares[held] * opn[j, held] / nav   # weights after this open's trades
        nav_close[j] = cash + mark(shares, close[j])
    return {"targets": targets, "w_open": w_open, "nav_close": nav_close}


def account(opn, close, choice, baskets):
    """Real account: copies the chosen basket at a switch or when that basket rebalances."""
    n, m = opn.shape
    shares, cash = np.zeros(m), 1.0
    nav_close, expo = np.zeros(n), np.zeros(n)
    for t in range(n):
        nav = cash + mark(shares, opn[t])
        pick = choice[t]
        if pick == "CASH":
            shares, cash = np.zeros(m), nav
        elif t == 0 or pick != choice[t - 1] or t in baskets[pick]["targets"]:
            shares, cash = buy(nav, baskets[pick]["w_open"][t], opn[t])
        expo[t] = mark(shares, opn[t]) / nav
        nav_close[t] = cash + mark(shares, close[t])
    return nav_close, expo


# --- signals --------------------------------------------------------------------------------

def rev_targets(book, start, days, rankings, col):
    """7 lowest 5-session returns (adjusted closes to the close before j), 1/7 each, desk caps.

    Returns (targets, short): short counts rebalances with fewer than 7 finite candidates.
    """
    adj, ci = book.adj.to_numpy(float), {tk: i for i, tk in enumerate(book.adj.columns)}
    out, short = {}, 0
    for j, rows in enumerate(rankings):
        if not rows or (out and S.iso_week(days[j]) == S.iso_week(book.days[start + j - 1])):
            continue
        r5 = []
        for pos, r in enumerate(rows):
            x = adj[:, ci[r["ticker"]]]
            with np.errstate(all="ignore"):
                v = x[start + j - 1] / x[start + j - 6] - 1
            if np.isfinite(v):
                r5.append((v, pos, r["key"]))
        raw = {key: 1 / N_REV for _, _, key in sorted(r5)[:N_REV]}
        short += len(raw) < N_REV
        w = np.zeros(len(col))
        for key, v in S.project_caps(raw, book.sector).items():
            w[col[key]] = v
        out[j] = w
    return out, short


def spearman(x, y):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def momentum_ic(book, start, rankings, h):
    """IC[j]: Spearman(mom120, adj return from row start+j-1 to start+j-1+h) over ranked names."""
    adj, ci = book.adj.to_numpy(float), {tk: i for i, tk in enumerate(book.adj.columns)}
    ic = np.full(len(rankings), np.nan)
    for j, rows in enumerate(rankings):
        a, b = start + j - 1, start + j - 1 + h
        if len(rows) < MIN_NAMES or b >= len(adj):
            continue
        with np.errstate(all="ignore"):
            pairs = np.array([(r["mom120"], adj[b, ci[r["ticker"]]] / adj[a, ci[r["ticker"]]] - 1)
                              for r in rows], float)
        pairs = pairs[np.isfinite(pairs).all(axis=1)]
        if len(pairs) >= MIN_NAMES:
            ic[j] = spearman(pairs[:, 0], pairs[:, 1])
    return ic


def known_ic(ic, h, k):
    """Mean IC[j] for j in [t-h-k+1, t-h] (sessions >= 0), at least max(3, k//2) finite, else NaN."""
    out, need = np.full(len(ic), np.nan), max(3, k // 2)
    for t in range(len(ic)):
        if t - h < 0:
            continue
        vals = ic[max(0, t - h - k + 1):t - h + 1]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= need:
            out[t] = vals.mean()
    return out


def rule_choices(known20, known10, gate):
    pick = lambda v: "REV" if np.isfinite(v) and v < 0 else "MOM"  # noqa: E731  undefined -> MOM
    a = [pick(v) for v in known20]
    b = [pick(v) if g else "CASH" for v, g in zip(known10, gate)]
    return a, b, ["MOM"] + a[:-1]          # late: choice[t] = choiceA[t-1], MOM before the start


# --- report ---------------------------------------------------------------------------------

def window(days, nav_close, expo, a, b):
    ix = np.flatnonzero((days >= a) & (days <= b))
    before = nav_close[ix[0] - 1] if ix[0] else 1.0
    path = np.r_[before, nav_close[ix]]
    return path[-1] / before - 1, float(np.min(path / np.maximum.accumulate(path) - 1)), expo[ix].mean()


def switches(days, choice):
    return [(days[t], choice[t]) for t in range(len(choice)) if t == 0 or choice[t] != choice[t - 1]]


def step1(ctx):
    days = np.array(ctx.days)
    mom = basket(ctx.opn, ctx.close, ctx.weekly)
    rev_plan, short = rev_targets(ctx.book, ctx.start, ctx.days, ctx.rankings, ctx.col)
    rev = basket(ctx.opn, ctx.close, rev_plan)
    baskets = {"MOM": mom, "REV": rev}
    ic = {h: momentum_ic(ctx.book, ctx.start, ctx.rankings, h) for h in (10, 20)}
    known20, known10 = known_ic(ic[20], 20, 20), known_ic(ic[10], 10, 10)
    gate = np.asarray(ctx.gates["SOXX20r1"], bool)
    a, b, late = rule_choices(known20, known10, gate)
    n = len(days)
    rules = {"MOM": ["MOM"] * n, "REV": ["REV"] * n, "RuleA": a, "RuleB": b, "RuleA_late": late}
    runs = {name: account(ctx.opn, ctx.close, ch, baskets) for name, ch in rules.items()}

    print("== STEP 1 (USD, zero commission, NAV 1.0 before the first open) ==")
    rel = np.max(np.abs(mom["nav_close"] / ctx.baseline.close - 1))
    print(f"sanity MOM shadow vs ctx.baseline.close: max rel diff {rel:.2e}")
    for name in ("MOM", "REV"):
        d = np.max(np.abs(runs[name][0] / baskets[name]["nav_close"] - 1))
        print(f"sanity account(always {name}) vs {name} shadow: max rel diff {d:.2e}")
    print(f"REV rebalances {len(rev['targets'])} (first {ctx.days[min(rev['targets'])]}, "
          f"{short} with < {N_REV} candidates), MOM rebalances {len(ctx.weekly)}; REV target sums "
          f"{sorted({round(float(w.sum()), 4) for w in rev['targets'].values()})}")
    print(f"Known-IC defined: h20K20 {np.isfinite(known20).sum()}/{n} (from {ctx.days[np.flatnonzero(np.isfinite(known20))[0]]}), "
          f"h10K10 {np.isfinite(known10).sum()}/{n}; IC finite h10 {np.isfinite(ic[10]).sum()} h20 {np.isfinite(ic[20]).sum()}")
    cols = ["full", "full_mdd", "jul_nov", "0623_0729", "0730_0922", "1013_1117", "expo_jul_nov"]
    table = {}
    for name, (nav, expo) in runs.items():
        st = {w: window(days, nav, expo, *WINDOWS[w]) for w in WINDOWS}
        table[name] = [st["full"][0], st["full"][1], st["jul_nov"][0], st["0623_0729"][0],
                       st["0730_0922"][0], st["1013_1117"][0], st["jul_nov"][2]]
    table = pd.DataFrame.from_dict(table, orient="index", columns=cols)
    print(table.to_string(float_format=lambda v: f"{v:+.4f}"))
    h2 = days >= "2026-07-01"
    for name in ("RuleA", "RuleB", "RuleA_late"):
        ch = np.array(rules[name])
        mix = {k: int((ch[h2] == k).sum()) for k in ("MOM", "REV", "CASH")}
        sw = " ".join(f"{d[5:]}>{k}" for d, k in switches(ctx.days, rules[name]))
        print(f"{name} switches ({len(switches(ctx.days, rules[name])) - 1}); jul-nov sessions {mix}:\n  {sw}")
    return {"rev": rev, "ic": ic, "known20": known20, "known10": known10, "rules": rules,
            "runs": runs, "table": table}


def step2(ctx, ref):
    """Scramble adj/cls/opn from book row start+150 on; recompute rankings, IC, Known-IC, REV."""
    rng = np.random.default_rng(SEED)
    cut = ctx.start + PIT_CUT

    def scramble(frame):
        out = frame.copy()
        out.iloc[cut:] = out.iloc[cut:].to_numpy() * rng.uniform(.5, 1.5, size=out.iloc[cut:].shape)
        return out

    pb = replace(ctx.book, adj=scramble(ctx.book.adj), cls=scramble(ctx.book.cls), opn=scramble(ctx.book.opn))
    prank = [S.momentum_ranking(pb, ctx.start + j, day)[1] for j, day in enumerate(ctx.days)]
    sig = lambda rows: [(r["key"], r["mom120"]) for r in rows]  # noqa: E731
    rank_same = all(sig(prank[j]) == sig(ctx.rankings[j]) for j in range(PIT_CUT + 1))
    rank_moved = sum(sig(prank[j]) != sig(ctx.rankings[j]) for j in range(PIT_CUT + 1, len(ctx.days)))
    pic = {h: momentum_ic(pb, ctx.start, prank, h) for h in (10, 20)}
    pk = {"h20K20": (known_ic(pic[20], 20, 20), ref["known20"]), "h10K10": (known_ic(pic[10], 10, 10), ref["known10"])}
    prev = rev_targets(pb, ctx.start, ctx.days, prank, ctx.col)[0]
    old = ref["rev"]["targets"]
    early = [j for j in old if j <= PIT_CUT]
    rev_same = ([j for j in prev if j <= PIT_CUT] == early
                and all(np.array_equal(prev[j], old[j]) for j in early))
    rev_moved = sum(not np.array_equal(prev[j], old.get(j)) for j in prev if j > PIT_CUT)
    print(f"\n== STEP 2 (PIT attack: book rows >= start+{PIT_CUT} = {ctx.book.days[cut]} x U[0.5,1.5], seed {SEED}) ==")
    print(f"rankings sessions <= {PIT_CUT} unchanged: {rank_same}; control: {rank_moved} later sessions changed")
    for name, (new, base) in pk.items():
        same = np.array_equal(new[:PIT_CUT + 1], base[:PIT_CUT + 1], equal_nan=True)
        moved = int(np.sum(~np.isclose(new[PIT_CUT + 1:], base[PIT_CUT + 1:], rtol=0, atol=1e-12, equal_nan=True)))
        print(f"Known-IC {name} sessions <= {PIT_CUT} unchanged: {same}; control: {moved} later sessions changed")
    print(f"REV targets sessions <= {PIT_CUT} unchanged ({len(early)} rebalances): {rev_same}; "
          f"control: {rev_moved}/{sum(j > PIT_CUT for j in prev)} later rebalances changed")
    gate, pgate = ctx.gates["SOXX20r1"], common.study.market_gates(pb, ctx.days)["SOXX20r1"]
    print(f"extra: gate SOXX20r1 recomputed on the scrambled book, sessions <= {PIT_CUT} unchanged: "
          f"{np.array_equal(gate[:PIT_CUT + 1], pgate[:PIT_CUT + 1])}; control: "
          f"{int((gate[PIT_CUT + 1:] != pgate[PIT_CUT + 1:]).sum())} later sessions changed")


def step3(ctx, ref):
    """Compare with this folder's published outputs (run only after steps 1-2 and reading its code)."""
    out = HERE / "out"
    cand = pd.read_csv(out / "meta_candidates.csv", index_col=0)
    rows = {"MOM": "ref:base", "REV": "ref:species_REV", "RuleA": "M2_h20_K20_REV",
            "RuleB": "M5_SOXX20r1_h10_K10_CASH"}
    cols = {"full_usd": "full", "full_usd_mdd": "full_mdd", "h2_usd": "jul_nov", "hell1_usd": "0623_0729",
            "rebound_usd": "0730_0922", "hell2_usd": "1013_1117", "h2_expo": "expo_jul_nov"}
    daily_path = HERE / "cache" / "meta_daily.parquet"
    daily = pd.read_parquet(daily_path) if daily_path.exists() else None
    print("\n== STEP 3 (vs out/meta_candidates.csv; max abs diff over "
          f"{', '.join(cols)}; daily = max rel diff of cache/meta_daily.parquet usd path) ==")
    for name, key in rows.items():
        diff = max(abs(ref["table"].loc[name, mine] - cand.loc[key, theirs]) for theirs, mine in cols.items())
        n_sw = len(switches(ctx.days, ref["rules"][name])) - 1
        sw = "" if np.isnan(cand.loc[key, "switches"]) else f"; switches mine {n_sw} theirs {int(cand.loc[key, 'switches'])}"
        line = f"{name:10s} vs {key:26s} max abs diff {diff:.1e}{sw}"
        if daily is not None:
            led = daily[daily.strategy == key].set_index("date").reindex(ctx.days)
            nav, expo = ref["runs"][name]
            line += (f"; daily usd {np.max(np.abs(nav / led.usd.to_numpy() - 1)):.1e}"
                     f", exposure {np.max(np.abs(expo - led.exposure.to_numpy())):.1e}")
        print(line)
    rob = pd.read_csv(out / "meta_robustness.csv", index_col=0).loc["M2_h20_K20_REV"]
    late = ref["table"].loc["RuleA_late"]
    d = max(abs(late["full"] - rob.delay1_full_usd), abs(late["jul_nov"] - rob.delay1_h2_usd),
            abs(late["full_mdd"] - rob.delay1_full_usd_mdd))
    print(f"RuleA_late vs meta_robustness M2_h20_K20_REV delay1 (full, h2, full_mdd): max abs diff {d:.1e}")
    focus = pd.read_csv(out / "meta_focus_choices.csv", index_col=0)["M2_h20_K20_REV"]
    same = list(focus.reindex(ctx.days)) == ref["rules"]["RuleA"]
    print(f"RuleA daily choices == meta_focus_choices.csv M2_h20_K20_REV: {same}")


if __name__ == "__main__":
    context = common.context()
    result = step1(context)
    step2(context, result)
    if "--compare" in sys.argv:
        step3(context, result)
