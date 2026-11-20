"""Claude's independent check of the trend-grid inputs, SOXX gates and two standard-engine paths.

1. Re-derive every stock's SMA(10..200), close, prior-63 volume, relative volume and RSI on every
   signal close from the cached bars (numpy cumulative sums, all symbols, not only the prefiltered
   rows), then compare with features_h1/h2.csv.
2. Re-derive the 13 SOXX gates with an explicit state machine and compare with gate_signals.csv.
3. Rebuild the current screen (balance recipe, 20/50/200) and the 25/50/200 variant from those
   arrays plus recipes.fundamentals(), check the first against recipes()['balance'] and the
   second against the features table, and price both with the standard TimingEngine under the
   current SOXX100 gate, monthly first-session reviews.
4. Diagnostics: one-session-late execution, off-spell attribution, one-step neighbours, and the
   first-day procedure of a weekly/monthly forward split.
Writes only claude_check.csv next to this file. No orders.
"""
import sys

sys.dont_write_bytecode = True

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "finviz_recipes_20261120"
ROOT = HERE.parents[3]
HISTORY = HERE.parent / "bottom_zone_20261111/news/v2/prices/SOXX_full.csv"
assert (ROOT / "AGENTS.md").exists()
sys.path[:0] = [str(PRIOR), str(ROOT), str(HERE)]

import recipes  # noqa: E402
from research import load_inputs, market_gate  # noqa: E402
from timing import TimingEngine  # noqa: E402
from compare import close_terminated, switched_targets  # noqa: E402
from features import load_features  # noqa: E402

market = recipes.market
PERIODS = (10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 100, 125, 150, 175, 200)
WINDOWS = {"ytd": ("2026-01-01", "2026-11-18"), "h1": ("2026-01-01", "2026-06-30"),
           "h2": ("2026-07-01", "2026-11-18"), "q3": ("2026-07-01", "2026-09-30"),
           "q4": ("2026-10-01", "2026-11-18")}
CURRENT, SHORT25 = (20, 50, 200), (25, 50, 200)
ROWS = []


def note(section, path, metric, value, detail=""):
    ROWS.append(dict(section=section, path=path, metric=metric, value=value, detail=detail))


# ------------------------------------------------------------------ 1. stock technicals
def sma_at(close, positions, n):
    """Mean of the n closes ending at each position (inclusive); NaN without a full window."""
    csum = np.r_[0., np.cumsum(close)]
    out = np.full(len(positions), np.nan)
    ok = positions >= n - 1
    p = positions[ok]
    out[ok] = (csum[p + 1] - csum[p + 1 - n]) / n
    return out


def stock_arrays(src):
    sig = pd.DatetimeIndex(src.signals)
    ends = {s: sig[sig < pd.Timestamp(t[0])][-1].date().isoformat() for s, t in market.mid.DEALS.items()}
    shape = (len(sig), len(src.symbols))
    names = ["close", "avg", "rv", "rsi", *[f"sma{n}" for n in PERIODS]]
    out = {k: np.full(shape, np.nan) for k in names}
    missing = []
    with patch.dict(market.tech.SIP_END, ends):
        for i, s in enumerate(src.symbols):
            bars = market.tech.read_bars(s)
            if bars is None:
                missing.append(s)
                continue
            c = bars.close.to_numpy(float)
            v = bars.volume.to_numpy(float)
            unit = bars.split_mult.to_numpy(float)
            pos = bars.index.get_indexer(sig)
            have = pos >= 0
            p = pos[have]
            u = unit[p]
            out["close"][have, i] = c[p] * u
            # volume of the 63 sessions strictly before the signal session
            vsum = np.r_[0., np.cumsum(v)]
            ok = p >= 63
            prior = np.full(len(p), np.nan)
            prior[ok] = (vsum[p[ok]] - vsum[p[ok] - 63]) / 63
            out["avg"][have, i] = prior / u
            out["rv"][have, i] = np.where(prior > 0, v[p] / prior, np.nan)
            out["rsi"][have, i] = market.tech.rsi(bars.close).to_numpy(float)[p]
            for n in PERIODS:
                out[f"sma{n}"][have, i] = sma_at(c, p, n) * u
    return out, missing


def audit_features(src, tech, funds):
    f = load_features()
    r = pd.Index(src.days).get_indexer(f.trade_open)
    c = f.col.to_numpy()
    assert (r >= 0).all()
    worst = {}
    for k in ["close", "avg", "rv", "rsi", *[f"sma{n}" for n in PERIODS]]:
        mine, theirs = tech[k][r, c], f[k].to_numpy()
        both_nan = np.isnan(mine) & np.isnan(theirs)
        if not (both_nan | (np.isfinite(mine) & np.isfinite(theirs))).all():
            raise AssertionError(f"Missing-value pattern differs for {k}")
        rel = np.abs(mine - theirs)[~both_nan] / np.maximum(1e-12, np.abs(theirs[~both_nan]))
        worst[k] = float(rel.max()) if rel.size else 0.
        assert worst[k] < 1e-9, (k, worst[k])
    for k, value in worst.items():
        note("restore", "features", f"max_rel_diff_{k}", value)
    tight = (f.current_ratio > 1.5) & (f.debt_equity >= 0) & (f.debt_equity < .5)
    mismatched = int((tight.to_numpy() != (funds["tight"][r, c] == 1)).sum())
    note("restore", "features", "balance_flag_mismatch_vs_recipes", mismatched, f"{len(f)} rows")
    assert mismatched == 0
    note("restore", "features", "rows", len(f), f"{f.trade_open.nunique()} review days, {f.symbol.nunique()} securities")
    for n in (25, 50, 75, 150, 200):
        note("restore", "features", f"rows_without_full_sma{n}", int(f[f"sma{n}"].isna().sum()))
    first_days = [d for d in ("2026-04-01", "2026-07-01", "2026-10-01") if d in set(f.trade_open)]
    note("forward", "features", "test_first_days_present", "|".join(first_days))
    return f


# ------------------------------------------------------------------ 2. SOXX gates
def soxx_gates(src, lag):
    desk = src.book.cls.SOXX.loc[:src.signals[-1]].dropna()
    warm = pd.read_csv(HISTORY, usecols=["date", "Close"], index_col="date").Close
    close = pd.concat([warm[warm.index < desk.index[0]].loc["2025-01-01":], desk])
    common = desk.index.intersection(warm.index)
    assert len(common) >= 200 and np.array_equal(desk.loc[common].to_numpy(), warm.loc[common].to_numpy())
    values = close.to_numpy(float)
    pos = close.index.get_indexer(src.signals) - lag
    assert (pos >= 199).all() and (close.index[pos + lag] == np.array(src.signals)).all()
    gates = {}
    for n in (25, 50, 75, 100, 125, 150, 200):
        line = sma_at(values, np.arange(len(values)), n)
        gates[f"sma{n}"] = (values >= line)[pos]
        if n in (75, 100, 125):
            for label, up, down in [("confirm2", None, None), ("band1", 1.01, .99)]:
                state, states = np.nan, []
                for t in range(len(values)):
                    if np.isfinite(line[t]):
                        if label == "band1":
                            if values[t] >= up * line[t]:
                                state = 1.
                            elif values[t] < down * line[t]:
                                state = 0.
                        elif t and np.isfinite(line[t - 1]):
                            a = values[t] >= line[t], values[t - 1] >= line[t - 1]
                            if a[0] and a[1]:
                                state = 1.
                            elif not a[0] and not a[1]:
                                state = 0.
                    states.append(state)
                picked = np.array(states)[pos]
                assert np.isfinite(picked).all()
                gates[f"{label}_{n}"] = picked == 1.
    return gates


def audit_gates(src):
    gates = soxx_gates(src, 0)
    saved = pd.read_csv(HERE / "gate_signals.csv")
    assert saved.trade_open.tolist() == list(src.days) and saved.signal_close.tolist() == list(src.signals)
    for key, value in gates.items():
        differs = int((saved[key].astype(bool).to_numpy() != value).sum())
        note("gate", key, "days_differing_from_gate_signals", differs)
        assert differs == 0, key
        off = ~value
        starts = np.flatnonzero(off & np.r_[True, value[:-1]])
        note("gate", key, "off_sessions", int(off.sum()))
        note("gate", key, "off_spells", len(starts), "|".join(src.days[s] for s in starts))
    assert np.array_equal(gates["sma100"], market_gate(src))
    note("gate", "sma100", "equals_market_gate", True)
    return gates, soxx_gates(src, 1)


# ------------------------------------------------------------------ 3. ranks & standard engine
def rank_rule(src, tech, funds, rule):
    s, m, l = rule
    t = tech
    common = (src.membership & (funds["quality"] == 1) & (funds["own"] > .4) & (funds["tight"] == 1)
              & (t["close"] > 10) & (t["avg"] > 500_000) & (t["rsi"] > 40) & (t["rv"] > 1) & np.isfinite(t["rv"]))
    trend = (t[f"sma{s}"] > t[f"sma{m}"]) & (t["close"] > t[f"sma{m}"]) & (t["close"] > t[f"sma{l}"])
    mask = common & trend
    ranks = []
    for j in range(len(src.days)):
        idx = np.flatnonzero(mask[j])
        ranks.append(idx[np.argsort(-t["rv"][j, idx], kind="stable")].tolist())
    return ranks


def features_rank(f, src, rule):
    s, m, l = rule
    ok = ((f[f"sma{s}"] > f[f"sma{m}"]) & (f.close > f[f"sma{m}"]) & (f.close > f[f"sma{l}"])
          & (f.current_ratio > 1.5) & (f.debt_equity >= 0) & (f.debt_equity < .5) & (f.rv > 1) & (f.rsi > 40))
    kept = f[ok].sort_values(["trade_open", "rv", "col"], ascending=[True, False, True])
    return kept.groupby("trade_open").col.agg(list).to_dict()


def window(curve, column, first, last):
    ix = np.flatnonzero((curve.index >= first) & (curve.index <= last))
    a = curve[column].iat[ix[0] - 1] if ix[0] else 1.
    w = np.r_[a, curve[column].to_numpy()[ix]]
    return w[-1] / w[0] - 1, float(np.min(w / np.maximum.accumulate(w) - 1))


def record(label, curve, gate=None):
    for col, cost in [("krw", "0bp"), ("krw10", "10bp"), ("krw25", "25bp")]:
        for name, (first, last) in WINDOWS.items():
            ret, dd = window(curve, col, first, last)
            note("path", label, f"{name}_ret_{cost}", ret)
            note("path", label, f"{name}_mdd_{cost}", dd)
    note("path", label, "turn_ytd", float(curve.turn.sum()))
    if gate is not None:
        note("path", label, "off_mean_exposure", float(curve.exposure.to_numpy()[~gate].mean()))


def spells(days, gate, paths, ref):
    starts = np.flatnonzero(~gate & np.r_[True, gate[:-1]])
    ends = np.flatnonzero(~gate & np.r_[gate[1:], True])
    for key, curve in paths.items():
        product = 1.
        for a, b in zip(starts, ends):
            mark = min(b + 1, len(days) - 1)
            mine = curve.krw10.iat[mark] / curve.krw10.iat[a - 1]
            base = ref.krw10.iat[mark] / ref.krw10.iat[a - 1]
            product *= mine / base
            note("spell", key, f"rel_10bp_{days[a]}_to_{days[b]}", mine / base - 1, f"{b - a + 1} off sessions")
        np.testing.assert_allclose(product, curve.krw10.iat[-1] / ref.krw10.iat[-1], rtol=1e-9)


def main():
    src = load_inputs()
    gate = market_gate(src)
    funds = recipes.fundamentals(src)
    tech, missing = stock_arrays(src)
    note("restore", "bars", "symbols_without_bars", len(missing), "|".join(missing))
    f = audit_features(src, tech, funds)
    gates, late_gates = audit_gates(src)

    reference = recipes.recipes(src)["balance"]
    ranks = {rule: rank_rule(src, tech, funds, rule) for rule in
             [(s, 50, 200) for s in (10, 15, 20, 25, 30, 35)]
             + [(20, 40, 200), (20, 60, 200), (20, 50, 175), (25, 40, 200), (25, 60, 200), (25, 50, 175)]}
    bad = [src.days[j] for j in range(len(src.days)) if ranks[CURRENT][j] != reference[j]]
    note("rank", "20-50-200", "days_differing_from_recipes_balance", len(bad), "|".join(bad[:5]))
    assert not bad
    by_features = features_rank(f, src, SHORT25)
    review_rows = sorted(pd.Index(src.days).get_indexer(f.trade_open.unique()))
    diff25 = [src.days[j] for j in review_rows if ranks[SHORT25][j] != by_features.get(src.days[j], [])]
    note("rank", "25-50-200", "review_days_differing_from_features_table", len(diff25), f"of {len(review_rows)}")
    assert not diff25
    monthly_days = [j for j in review_rows if j == 0 or src.days[j][:7] != src.days[j - 1][:7]]
    for j in monthly_days:
        a, b = set(ranks[CURRENT][j][:7]), set(ranks[SHORT25][j][:7])
        note("rank", "monthly_top7_overlap", src.days[j], len(a & b),
             f"only25={'|'.join(src.symbols[i] for i in sorted(b - a))}; only20={'|'.join(src.symbols[i] for i in sorted(a - b))}")

    engine = TimingEngine(src, gate)
    curves = {}
    for rule, rk in ranks.items():
        key = "-".join(map(str, rule))
        paths = engine.evaluate(rk, "monthly1", detail=False, stress=True)
        curves[key] = paths["switch"]
        curves[key + "/always"] = paths["always"]
    archive = pd.read_csv(PRIOR / "selected_daily.csv", index_col=["key", "date"]).loc["balance/monthly1"]
    np.testing.assert_allclose(curves["20-50-200"][["krw", "krw10", "krw25", "turn"]],
                               archive[["krw", "krw10", "krw25", "turn"]], rtol=2e-10, atol=1e-12)
    note("rank", "20-50-200", "switch_equals_finviz_recipes_balance_monthly1", True)
    for key in ("20-50-200", "25-50-200"):
        record(f"{key}/sma100/monthly1", curves[key], gate)
        record(f"{key}/always/monthly1", curves[key + "/always"])
    for key, value in engine.controls(stress=True).items():
        record(f"control_{key}", value, gate)

    # one-session-late execution: gate state and the monthly basket both one auction late
    for key in ("20-50-200", "25-50-200"):
        monthly = engine.targets(ranks[tuple(map(int, key.split("-")))], "monthly1")
        shifted = {i + 1: monthly[i] for i in engine.reviews["monthly1"] if i + 1 < len(src.days)}
        delayed = close_terminated(shifted, src.opn, src.tradable)
        late = engine._price(switched_targets(src.days, src.opn, engine.desk, delayed, late_gates["sma100"]),
                             detail=False, stress=True)
        record(f"{key}/sma100/monthly1/late1", late, late_gates["sma100"])
    spells(src.days, gate, {"25-50-200": curves["25-50-200"]}, curves["20-50-200"])

    # neighbours: switch YTD/H2/Q4 and always-on training windows (H1 for H2, Q3 for Q4)
    for key, curve in curves.items():
        if key.endswith("/always"):
            for name in ("h1", "q3"):
                ret, dd = window(curve, "krw10", *WINDOWS[name])
                note("neighbour", key, f"{name}_ret_10bp", ret)
                note("neighbour", key, f"{name}_mdd_10bp", dd)
            ix = np.flatnonzero(curve.index <= "2026-06-30")
            note("neighbour", key, "h1_exposure", float(curve.exposure.to_numpy()[ix].mean()))
        else:
            for name in ("ytd", "h2", "q4"):
                ret, dd = window(curve, "krw10", *WINDOWS[name])
                note("neighbour", key, f"{name}_ret_10bp", ret)
                note("neighbour", key, f"{name}_mdd_10bp", dd)
    for centre, family in [("20-50-200", ["20-50-200", "15-50-200", "25-50-200", "20-40-200", "20-60-200", "20-50-175"]),
                           ("25-50-200", ["25-50-200", "20-50-200", "30-50-200", "25-40-200", "25-60-200", "25-50-175"])]:
        for name in ("h1", "q3"):
            values = [window(curves[k + "/always"], "krw10", *WINDOWS[name])[0] for k in family]
            note("neighbour", centre, f"family_median_always_{name}_ret_10bp", float(np.median(values)), "|".join(family))

    # forward first-day procedure: current rule until the test's first open, then 25/50/200
    for schedule in ("weekly1", "monthly1"):
        for first in ("2026-07-01", "2026-10-01"):
            cut = src.days.index(first)
            spliced = [ranks[CURRENT][j] if j < cut else ranks[SHORT25][j] for j in range(len(src.days))]
            variants = {"refresh_first_open": sorted(set(engine.reviews[schedule]) | {cut}),
                        "wait_next_review": engine.reviews[schedule]}
            base_always = engine._price(engine.targets(ranks[CURRENT], schedule), detail=False, stress=False)
            new_always = engine._price(engine.targets(ranks[SHORT25], schedule), detail=False, stress=False)
            for name, reviews in variants.items():
                tag = f"fw_{schedule}_{first}_{name}"
                engine.reviews[tag] = reviews
                plan = engine.targets(spliced, tag)
                always = engine._price(plan, detail=False, stress=False)
                switch = engine._price(switched_targets(src.days, src.opn, engine.desk, plan, gate), detail=False, stress=False)
                np.testing.assert_allclose(always.krw10.iloc[:cut], base_always.krw10.iloc[:cut], rtol=1e-12)
                for label, curve in (("always", always), ("switch", switch)):
                    ret, dd = window(curve, "krw10", first, "2026-11-18")
                    note("forward", tag, f"{label}_test_ret_10bp", ret, f"gate_on_first_day={bool(gate[cut])}")
                    note("forward", tag, f"{label}_turn_first_day", float(curve.turn.iat[cut]))
            ret, _ = window(new_always, "krw10", first, "2026-11-18")
            note("forward", f"fw_{schedule}_{first}_splice_full_year_25", "always_test_ret_10bp", ret,
                 "uses the 25 rule's own pre-cut basket (hindsight splice)")

    out = pd.DataFrame(ROWS)
    out.to_csv(HERE / "claude_check.csv", index=False, float_format="%.10g")
    show = out[out.section.eq("path") & out.metric.str.match(r"(ytd|h2|q4)_(ret|mdd)_10bp|ytd_ret_0bp")]
    print(show.pivot(index="path", columns="metric", values="value").round(4).to_string())
    print(out[out.section.isin(["spell", "forward", "rank"]) & ~out.path.eq("monthly_top7_overlap")].to_string())


if __name__ == "__main__":
    main()
