"""Shared context for the independent check of desk_defense_20261119.

Reuses Codex's engine (study.plan / replay / market_gates / eligibility) so the 7-stock
calculation keeps a single source. Adds a fractional-exposure replay and one ledger writer.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CODEX = HERE.parent / "desk_defense_20261119"
for _p in (str(ROOT), str(CODEX)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import study  # noqa: E402  Codex engine
from evaluate import summarize  # noqa: E402
from desk import strategies as S  # noqa: E402

OUT = HERE / "out"          # summaries (tracked)
CACHE = HERE / "cache"      # daily ledgers (git-ignored by data/**/cache/, rebuilt by code)
WINDOWS = {"full": ("2026-01-01", "2026-12-31"), "h1": ("2026-01-01", "2026-06-30"),
           "h2": ("2026-07-01", "2026-12-31"), "jan_aug": ("2026-01-01", "2026-08-31"),
           "jul_aug": ("2026-07-01", "2026-08-31"), "sep_nov": ("2026-09-01", "2026-12-31"),
           "oct_nov": ("2026-10-01", "2026-12-31")}
LEDGER = ["krw", "usd", "turn", "exposure", "krw10"]


def context():
    """Everything study.main() builds before it starts looping over candidates."""
    book = study.load_book()
    baseline = S.run_model(book)
    days = baseline.days
    start = book.days.index(days[0])
    col = {key: i for i, key in enumerate(book.keys)}
    tickers = [book.ticker[key] for key in col]
    rankings = [S.momentum_ranking(book, start + j, day)[1] for j, day in enumerate(days)]
    ok = study.eligibility(book, days, tickers)
    gates = study.market_gates(book, days)
    opn, close = [frame.loc[days, tickers].to_numpy() for frame in (book.opn, book.cls)]
    fx = book.fx[start:] / book.fx[start - 1]
    weekly = study.plan(book, days, rankings, col, ok["any"], "week", "rotate")
    return SimpleNamespace(book=book, baseline=baseline, days=days, start=start, col=col,
                           tickers=tickers, rankings=rankings, ok=ok, gates=gates,
                           opn=opn, close=close, fx=fx, weekly=weekly)


def replay_frac(days, opn, close, fx, targets, frac):
    """study.replay generalised: hold frac[j] (0..1) of the shadow basket from the open of day j.

    frac[j] must be decided from information through the close of day j-1. Real holdings are
    rebalanced when the shadow publishes a target while frac>0, or when frac changes. With a
    0/1 frac this reproduces study.replay(gate) exactly (asserted in __main__).
    """
    frac = np.asarray(frac, dtype=float)
    if len(frac) != len(days) or (frac < 0).any() or (frac > 1 + 1e-12).any():
        raise ValueError("frac must be one value in [0, 1] per session")
    shares, shadow = np.zeros(opn.shape[1]), np.zeros(opn.shape[1])
    cash, reserve, drag = 1., 1., 1.
    rows = []
    for j, day in enumerate(days):
        o, c = opn[j], close[j]
        active = (shares != 0) | (shadow != 0)
        if not np.isfinite(o[active]).all() or not np.isfinite(c[active]).all():
            raise ValueError(f"Held security missing market price on {day}")
        o_mark, c_mark = np.nan_to_num(o), np.nan_to_num(c)
        nav = cash + shares @ o_mark
        shadow_nav = reserve + shadow @ o_mark
        trade = j in targets
        if trade:
            w = targets[j]
            if not (o[w > 0] > 0).all():
                raise ValueError(f"Missing executable open on {day}")
            shadow = np.divide(shadow_nav * w, o, out=np.zeros_like(w), where=w > 0)
            reserve = shadow_nav * (1 - w.sum())
        turn = 0.
        if (frac[j] > 0 and trade) or (j > 0 and abs(frac[j] - frac[j - 1]) > 1e-12):
            weights = frac[j] * (shadow * o_mark / shadow_nav)
            turn = np.abs(weights - shares * o_mark / nav).sum()
            shares = np.divide(nav * weights, o, out=np.zeros_like(weights), where=weights > 0)
            cash = nav * (1 - weights.sum())
            drag *= 1 - .001 * turn
        usd = cash + shares @ c_mark
        krw = usd * fx[j]
        rows.append((krw, usd, turn, 1 - cash / nav, krw * drag))
    return pd.DataFrame(rows, index=days, columns=LEDGER)


def save(family, curves, configs):
    """Write cache/<family>_daily.parquet (long ledgers) and out/<family>_candidates.csv (Codex's summary)."""
    if set(curves) != set(configs):
        raise ValueError("Curve and configuration keys differ")
    OUT.mkdir(exist_ok=True)
    for key, frame in curves.items():
        if list(frame.columns) != LEDGER or not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"Bad ledger for {key}")
    daily = pd.concat(curves, names=["strategy", "date"]).reset_index()
    CACHE.mkdir(exist_ok=True)
    daily.to_parquet(CACHE / f"{family}_daily.parquet", index=False)
    table = pd.DataFrame.from_dict({k: {**configs[k], **summarize(curves[k])} for k in sorted(curves)},
                                   orient="index")
    table.index.name = "strategy"
    table.insert(0, "family", family)
    table.to_csv(OUT / f"{family}_candidates.csv")
    return table


def codex_curves(ctx):
    """Codex's 303 ledgers rebuilt in memory: study.main() without its file-writing steps."""
    curves, configs = {}, {}
    args = (ctx.days, ctx.opn, ctx.close, ctx.fx)
    for mode, freq, method, condition in study.selectors():
        targets = study.plan(ctx.book, ctx.days, ctx.rankings, ctx.col, ctx.ok[condition], freq, method)
        for market, gate in ctx.gates.items():
            key = "base" if (mode == "weekly" and market == "none") else f"{mode}_{market}"
            configs[key] = dict(mode=mode, market=market, condition=condition, freq=freq, method=method)
            curves[key] = study.replay(*args, targets, gate)
        if freq == "change" and method == "rotate" and condition in ("trendA", "giantA", "bothA", "bothB"):
            for market in ("QQQ20r1", "QQQ50r1", "SOXX20r1", "SOXX50r1"):
                key = f"switch_{condition}_{market}"
                configs[key] = dict(mode=mode, market=market, condition=condition, freq=freq, method="switch")
                switched = study.switched_targets(ctx.days, ctx.opn, ctx.weekly, targets, ctx.gates[market])
                curves[key] = study.replay(*args, switched, ctx.gates["none"])
        if mode == "weekly":
            configs["half"] = dict(mode="weekly50", market="none", condition="any", freq=freq, method="half")
            curves["half"] = study.replay(*args, targets, ctx.gates["none"], .5)
    return curves, configs


def window_stats(curve, start, end):
    """Return and max drawdown inside [start, end], anchored on the close before the window."""
    ix = np.flatnonzero((curve.index >= start) & (curve.index <= end))
    out = {}
    for column in ("krw", "usd"):
        before = curve[column].iloc[ix[0] - 1] if ix[0] else 1.0
        values = np.r_[before, curve[column].to_numpy()[ix]]
        out[column] = values[-1] / before - 1
        out[f"{column}_mdd"] = np.min(values / np.maximum.accumulate(values) - 1)
    out["expo"] = curve.exposure.to_numpy()[ix].mean()
    out["turn"] = curve.turn.to_numpy()[ix].sum()
    return out


def stats_table(curves):
    """One row per strategy, columns <window>_<krw|krw_mdd|usd|usd_mdd|expo|turn> for WINDOWS."""
    rows = {}
    for key, curve in curves.items():
        rows[key] = {f"{w}_{m}": v for w, (a, b) in WINDOWS.items()
                     for m, v in window_stats(curve, a, b).items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def select(table, train, currency="krw", base="base"):
    """Codex's two selection rules on a training window; base is always a candidate.

    return: train drawdown no deeper than base, highest train return.
    stability: also keeps >=75% of base's train gain, then the shallowest train drawdown
    (ties: higher return, then name).
    """
    ret, mdd = table[f"{train}_{currency}"], table[f"{train}_{currency}_mdd"]
    pool = table.index[(mdd >= mdd[base] - 1e-7) | (table.index == base)]
    best = sorted(pool, key=lambda k: (-ret[k], k))[0]
    keep = [k for k in pool if ret[k] >= .75 * ret[base] - 1e-12 or k == base]
    steady = sorted(keep, key=lambda k: (-round(float(mdd[k]), 6), -ret[k], k))[0]
    return {"return": best, "stability": steady}


def matched_control(base_curve, exposure, window):
    """Hindsight control: hold exposure/base_exposure of the base every day of the window, rest USD cash."""
    a, b = WINDOWS[window]
    ix = np.flatnonzero((base_curve.index >= a) & (base_curve.index <= b))
    normal = base_curve.exposure.to_numpy()[ix].mean()
    if normal <= 0 or exposure > normal + 1e-9:
        return np.nan, np.nan
    usd = base_curve.usd.to_numpy()
    path = np.r_[usd[ix[0] - 1] if ix[0] else 1.0, usd[ix]]
    daily = path[1:] / path[:-1] - 1
    nav = np.r_[1., np.cumprod(1 + exposure / normal * daily)]
    return nav[-1] - 1, np.min(nav / np.maximum.accumulate(nav) - 1)


if __name__ == "__main__":
    ctx = context()
    base = study.replay(ctx.days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, ctx.gates["none"])
    np.testing.assert_allclose(base.usd, ctx.baseline.close, rtol=1e-10)
    for name in ("none", "SOXX100r1", "QQQ20r3"):
        gate = ctx.gates[name]
        a = study.replay(ctx.days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, gate)
        b = replay_frac(ctx.days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, gate.astype(float))
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12, atol=1e-12)
    half = study.replay(ctx.days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, ctx.gates["none"], .5)
    s = summarize(base)
    print(f"sessions={len(ctx.days)} tickers={len(ctx.tickers)} gates={len(ctx.gates)}")
    print(f"base h1 {s['h1_return']:+.4f} h2 {s['h2_return']:+.4f} h2_usd {s['h2_return_usd']:+.4f} "
          f"h2_mdd {s['h2_mdd']:+.4f}; half h2 {summarize(half)['h2_return']:+.4f}")
    print("replay_frac == study.replay for 0/1 gates: OK")
    curves, configs = codex_curves(ctx)
    table = save("codex", curves, configs)
    ref = pd.read_csv(CODEX / "candidates.csv", index_col=0)
    assert set(ref.index) == set(table.index), "candidate keys differ from Codex"
    for column in ("h1_return", "h2_return", "h2_mdd", "h2_return_usd", "full_return", "h2_exposure", "h2_turn"):
        np.testing.assert_allclose(table.loc[ref.index, column], ref[column], rtol=1e-9, atol=1e-9)
    print(f"codex ledgers rebuilt: {len(curves)} strategies match candidates.csv")
    st = stats_table(curves)
    for train, test in (("h1", "h2"), ("jan_aug", "sep_nov")):
        for rule, key in select(st, train).items():
            r = st.loc[key]
            print(f"{train}->{test} {rule:9s} {key:28s} train {r[train + '_krw']:+.3f} "
                  f"test krw {r[test + '_krw']:+.3f} usd {r[test + '_usd']:+.3f} mdd_usd {r[test + '_usd_mdd']:+.3f}")
    b = st.loc["base"]
    print("base", {k: round(float(b[k]), 4) for k in ("jan_aug_krw", "sep_nov_krw", "sep_nov_usd", "sep_nov_usd_mdd", "oct_nov_usd")})

