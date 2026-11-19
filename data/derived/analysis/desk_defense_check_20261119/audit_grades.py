"""Audit of desk_defense_20261119: (1) are the replayed trend / Giant recovery grades the grades the
desk tabs show, (2) do the grades (and SMA / breadth states) carry forward information in 2026.

Read-only analysis. Writes only under cache/audit_grades/. Run from the repo root:
    py data/derived/analysis/desk_defense_check_20261119/audit_grades.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import signals  # noqa: E402  Codex module (pure functions only)
import study  # noqa: E402
import watchlist  # noqa: E402
from desk import giant, trend  # noqa: E402
from desk import strategies as S  # noqa: E402
from gbrc.etf.market import is_cash_holding, listing_market, yahoo_ticker  # noqa: E402
from gbrc.etf.storage import NORMALIZED_HOLDINGS_DIR  # noqa: E402

ROOT, CODEX = common.ROOT, common.CODEX
OUT = common.CACHE / "audit_grades"
BARS = CODEX / "cache" / "grade_bars"
LETTER = {4: "S", 3: "A", 2: "B", 1: "C", 0: "D"}
PERIODS = {"H1": ("2026-01-01", "2026-06-30"), "JulAug": ("2026-07-01", "2026-08-31"),
           "SepNov": ("2026-09-01", "2026-12-31"), "H2": ("2026-07-01", "2026-12-31")}
HORIZONS = (5, 10, 20)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)


def show(title, frame=None):
    print(f"\n=== {title} ===")
    if frame is not None:
        print(frame.to_string())


def letter(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else LETTER[int(v)]


def load_bars(symbols=None):
    paths = sorted(BARS.glob("*.parquet")) if symbols is None else [BARS / f"{s}.parquet" for s in symbols]
    return {p.stem: pd.read_parquet(p) for p in paths if p.exists()}


def git_json(rev, path):
    raw = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
    return json.loads(raw.decode("utf-8"))


# --------------------------------------------------------------------------- 1a trend
def us_roster(fund, day):
    """US-listed rows of one normalized snapshot, same row logic as signals.holdings_history."""
    path = NORMALIZED_HOLDINGS_DIR / f"{day}_{fund}.json"
    rows = {}
    for row in json.loads(path.read_text(encoding="utf-8"))["holdings"]:
        code = row.get("code")
        if listing_market(code) != "US" or is_cash_holding(row):
            continue
        key = S.research_key(code)
        symbol = yahoo_ticker(code, day)
        if code in ("SATS US EQUITY", "SKHYV US EQUITY"):
            symbol = yahoo_ticker(key)
        if symbol:
            rows.setdefault(key, symbol)
    return rows


def trend_compare(label, data, codex, book, bars):
    """Production tab JSON vs Codex's per-date replay, plus an emulation of the tab with its own roster."""
    as_of, days30 = data["as_of"], data["days"]
    funds = {"456600": data["funds"]["TIME"], "2ETFQ1": data["funds"]["KoAct"]}
    prod = {}
    for s in data["stocks"]:
        if s["market"] != "미국":
            continue
        col = book.ticker.get(S.CORPORATE_ALIASES.get(s["key"], s["key"]))
        if col is not None and col in codex.columns:
            prod[col] = s
    us_all = sum(1 for s in data["stocks"] if s["market"] == "미국")
    # the tab ranks the whole 30-day history with today's roster: emulate that with Codex's grader
    histories = {fund: (["0000-00-00"], {"0000-00-00": us_roster(fund, day)}) for fund, day in funds.items()}
    emu, _ = signals.grade_trend(book, bars, histories, [d for d in days30 if d <= signals.END])
    rows = []
    for col, s in prod.items():
        for d, g in zip(days30, s["hist"]):
            if d not in codex.index:
                continue
            rows.append(dict(ticker=col, date=d, prod=g, codex=letter(codex.at[d, col]),
                             emu=letter(emu.at[d, col]) if d in emu.index else None))
    cells = pd.DataFrame(rows)

    def agree(frame, other):
        both = frame[frame["prod"].notna() & frame[other].notna()]
        a_p, a_o = both["prod"].isin(["S", "A"]), both[other].isin(["S", "A"])
        return dict(n=len(both), exact=float((both["prod"] == both[other]).mean()) if len(both) else np.nan,
                    a_or_better=float((a_p == a_o).mean()) if len(both) else np.nan,
                    prod_A=int(a_p.sum()), other_A=int(a_o.sum()), A_both=int((a_p & a_o).sum()),
                    prod_only_graded=int((frame["prod"].notna() & frame[other].isna()).sum()),
                    other_only_graded=int((frame["prod"].isna() & frame[other].notna()).sum()))
    last = cells[cells.date == as_of]
    summary = pd.DataFrame({
        "asof_codex_vs_tab": agree(last, "codex"), "asof_emulation_vs_tab": agree(last, "emu"),
        "hist30_codex_vs_tab": agree(cells, "codex"), "hist30_emulation_vs_tab": agree(cells, "emu")}).T
    show(f"1a TREND {label}: tab as_of {as_of}, fund snapshots {data['funds']}, tab US rows {us_all}, "
         f"mapped to Codex columns {len(prod)}", summary)
    codex_roster = signals.tickers_as_of(signals.holdings_history()[0], as_of)
    tab_roster = {}
    for fund, day in funds.items():
        for k, v in us_roster(fund, day).items():
            tab_roster.setdefault(k, v)
    print(f"roster on {as_of}: tab US {len(tab_roster)} names, Codex US {len(codex_roster)} names, "
          f"tab-only {sorted(set(tab_roster.values()) - set(codex_roster.values()))}, "
          f"codex-only {sorted(set(codex_roster.values()) - set(tab_roster.values()))}")
    bad = last[(last["prod"] != last["codex"]) & last["prod"].notna() & last["codex"].notna()].copy()
    bad["pct_tab"] = [prod[t].get("pct") for t in bad.ticker]
    bad["emu_matches_tab"] = bad.emu == bad["prod"]
    show(f"1a TREND {label}: as_of mismatches (tab vs Codex)", bad)
    miss = last[last["prod"].isna() != last["codex"].isna()]
    if len(miss):
        show(f"1a TREND {label}: graded on one side only", miss)
    emu_bad = last[(last["prod"] != last["emu"]) & last["prod"].notna() & last["emu"].notna()]
    if len(emu_bad):
        show(f"1a TREND {label}: emulation (tab roster, cached bars) still differs from tab", emu_bad)
    # weekly grade (the tab's selectable judgment unit) vs daily
    wk = [(c, s.get("wgrade"), s.get("grade")) for c, s in prod.items()]
    diff = [w for w in wk if w[1] != w[2]]
    print(f"tab weekly grade != daily grade on {as_of}: {len(diff)}/{len(wk)} mapped names; "
          f"A-or-better differs for {sum((w[1] in ('S', 'A')) != (w[2] in ('S', 'A')) for w in wk)}")
    cells.to_csv(OUT / f"trend_cells_{label}.csv", index=False)
    return summary.assign(file=label)


# --------------------------------------------------------------------------- 1a giant
def giant_compare(label, data, codex, close_date, bars):
    stocks, anchor = data["stocks"], data["anchor"]
    t = pd.Timestamp(close_date)
    mine = giant.find_anchor(bars[giant.ANCHOR].loc[:t, "Close"])
    keys = ("peak_date", "low_date", "grade_from", "grade_to", "last_date", "crash")
    show(f"1a GIANT {label}: anchor tab vs cache-recomputed at {close_date}",
         pd.DataFrame({"tab": {k: anchor[k] for k in keys}, "recomputed": {k: mine[k] for k in keys}}))
    rows = []
    for col in codex.columns:
        sym = watchlist.AI_UNIVERSE_ALIASES.get(col, col)
        p = stocks.get(sym)
        rows.append(dict(ticker=col, universe_symbol=sym, in_universe=p is not None,
                         prod=p["grade"] if p else None, prod_recovery=p.get("recovery") if p else None,
                         prod_date=p.get("date") if p else None, codex=letter(codex.at[close_date, col])))
    f = pd.DataFrame(rows)
    u = f[f.in_universe]
    both = u[u["prod"].notna() & u.codex.notna()]
    a_p, a_c = u["prod"].isin(["S", "A"]), u.codex.isin(["S", "A"])
    summary = dict(file=label, close_date=close_date, codex_columns=len(f), in_ai_universe=len(u),
                   not_in_universe=int((~f.in_universe).sum()),
                   both_graded=len(both), exact_both_graded=float((both["prod"] == both.codex).mean()),
                   exact_incl_ungraded=float((u["prod"].fillna("-") == u.codex.fillna("-")).mean()),
                   a_or_better_agree=float((a_p == a_c).mean()), tab_A=int(a_p.sum()), codex_A=int(a_c.sum()),
                   tab_ungraded=int(u["prod"].isna().sum()), codex_ungraded=int(u.codex.isna().sum()),
                   graded_one_side=int((u["prod"].isna() != u.codex.isna()).sum()))
    show(f"1a GIANT {label}: tab vs Codex at close {close_date}", pd.Series(summary).to_frame("value"))
    bad = u[u["prod"].fillna("-") != u.codex.fillna("-")]
    show(f"1a GIANT {label}: mismatches", bad)
    # whole 233-name universe: cache bars + desk formula vs the tab (data / formula equivalence)
    rec = []
    for sym, p in stocks.items():
        b = bars.get(sym)
        if b is None:
            continue
        h = b.loc[:t]
        g = giant.grade_of(giant.own_recovery(h, mine)) if len(h) and (t - h.index[-1]).days <= 7 else None
        rec.append(dict(symbol=sym, tab=p["grade"], recomputed=g, tab_date=p.get("date"), market=p.get("market")))
    r = pd.DataFrame(rec)
    same = r.tab.fillna("-") == r.recomputed.fillna("-")
    us = r.market == "미국"
    print(f"universe-wide recompute at {close_date}: {len(r)} symbols with cached bars of {len(stocks)}; "
          f"grade equal (incl. ungraded) {same.mean():.3f}; US only {same[us].mean():.3f} (n={int(us.sum())}); "
          f"non-US {same[~us].mean():.3f} (n={int((~us).sum())})")
    f.to_csv(OUT / f"giant_cells_{label}.csv", index=False)
    return summary


def giant_intraday_check(data, bars):
    """Working-tree giant.json holds an in-progress bar. Rebuild its recovery from cached peak/low + tab close."""
    anchor, out = data["anchor"], []
    for sym, p in data["stocks"].items():
        b = bars.get(sym)
        if b is None or p.get("recovery") is None:
            continue
        close = b["Close"]
        win = close[(close.index >= pd.Timestamp(anchor["grade_from"])) & (close.index <= pd.Timestamp(anchor["low_date"]))]
        if win.empty:
            continue
        peak_at, peak = win.idxmax(), float(win.max())
        after = close[(close.index >= peak_at) & (close.index <= pd.Timestamp(anchor["grade_to"]))]
        low = min(float(after.min()), float(p["close"]))
        out.append(abs(round((float(p["close"]) - low) / (peak - low) * 100.0, 1) - p["recovery"]))
    out = np.array(out)
    print(f"working-tree giant.json recovery rebuilt from cached peak/low + the tab's own last price: "
          f"n={len(out)}, |diff|<=0.5pt {np.mean(out <= .5):.3f}, max {out.max():.1f}")


# --------------------------------------------------------------------------- shared arrays
def grade_arrays(ctx):
    """Numeric grades as USED at each session (close of the previous session), shape [221, 86]."""
    out = []
    for name in signals.SAVED:
        frame = pd.read_csv(CODEX / name, index_col=0)
        out.append(frame.reindex(index=ctx.book.days, columns=ctx.tickers).shift().loc[ctx.days].to_numpy(float))
    return out


def held_mask(ctx):
    """[session, ticker] True for the names of the latest published weekly base target, and its weights."""
    mask = np.zeros((len(ctx.days), len(ctx.tickers)), bool)
    weight = np.zeros(mask.shape)
    last = None
    for j in range(len(ctx.days)):
        last = ctx.weekly.get(j, last)
        if last is not None:
            mask[j], weight[j] = last > 0, last
    return mask, weight


def period_mask(days, name):
    a, b = PERIODS[name]
    return np.array([a <= d <= b for d in days])


# --------------------------------------------------------------------------- 1b coverage
def coverage(ctx, T, G, held, weight):
    rows = []
    for per in ("H1", "JulAug", "SepNov", "H2"):
        m = period_mask(ctx.days, per)[:, None] & held
        w = np.where(m, weight, 0.)
        for name, X in (("trend", T), ("giant", G)):
            miss, low, ok = m & np.isnan(X), m & (X < 3), m & (X >= 3)
            rows.append(dict(period=per, grade=name, held_ticker_days=int(m.sum()), missing=int(miss.sum()),
                             below_A=int(low.sum()), A_or_better=int(ok.sum()),
                             missing_share=miss.sum() / m.sum(), below_A_share=low.sum() / m.sum(),
                             w_missing=w[miss].sum() / w.sum(), w_below_A=w[low].sum() / w.sum(),
                             w_A_or_better=w[ok].sum() / w.sum()))
        bt = m & (T >= 3) & (G >= 3)
        anymiss = m & (np.isnan(T) | np.isnan(G)) & ~((T < 3) | (G < 3))
        rows.append(dict(period=per, grade="bothA", held_ticker_days=int(m.sum()), missing=int(anymiss.sum()),
                         below_A=int((m & ((T < 3) | (G < 3))).sum()), A_or_better=int(bt.sum()),
                         missing_share=anymiss.sum() / m.sum(), below_A_share=(m & ((T < 3) | (G < 3))).sum() / m.sum(),
                         w_missing=w[anymiss].sum() / w.sum(), w_below_A=w[m & ((T < 3) | (G < 3))].sum() / w.sum(),
                         w_A_or_better=w[bt].sum() / w.sum()))
    cov = pd.DataFrame(rows)
    show("1b coverage on held (7-name weekly basket) ticker-days; for bothA 'missing' = fails only because a grade "
         "is missing, 'below_A' = at least one existing grade below A; w_* = share of base target weight", cov.round(3))
    universe = set(watchlist.AI_SUBSECTOR)
    in_uni = np.array([watchlist.AI_UNIVERSE_ALIASES.get(t, t) in universe for t in ctx.tickers])
    h2 = period_mask(ctx.days, "H2")[:, None] & held
    h1 = period_mask(ctx.days, "H1")[:, None] & held
    for per, m in (("H1", h1), ("H2", h2)):
        gm = m & np.isnan(G)
        print(f"{per}: giant grade missing on {int(gm.sum())} held ticker-days; ticker outside AI universe "
              f"{int((gm & ~in_uni[None, :]).sum())}, in universe but no qualifying crash "
              f"{int((gm & in_uni[None, :]).sum())}")
        names = pd.Series(gm.sum(axis=0), index=ctx.tickers)
        hd = pd.Series(m.sum(axis=0), index=ctx.tickers)
        tab = pd.DataFrame({"held_days": hd, "giant_missing": names, "trend_missing": (m & np.isnan(T)).sum(axis=0),
                            "giant_A": (m & (G >= 3)).sum(axis=0), "trend_A": (m & (T >= 3)).sum(axis=0),
                            "in_ai_universe": in_uni})
        show(f"1b {per} held names", tab[tab.held_days > 0].sort_values("held_days", ascending=False))
    print(f"TIME-book tickers outside the AI universe (never get a Giant grade): "
          f"{[t for t, u in zip(ctx.tickers, in_uni) if not u]}")
    return cov


def missing_variants(ctx, T, G, codex_table):
    """Re-run the grade rules with missing grades treated as pass, and with only missingness failing."""
    args = (ctx.days, ctx.opn, ctx.close, ctx.fx)
    nt, ng = np.isnan(T), np.isnan(G)
    elig = {
        "trendA": {"orig": T >= 3, "missing_pass": (T >= 3) | nt, "only_missing_fails": ~nt},
        "giantA": {"orig": G >= 3, "missing_pass": (G >= 3) | ng, "only_missing_fails": ~ng},
        "bothA": {"orig": (T >= 3) & (G >= 3), "missing_pass": ((T >= 3) | nt) & ((G >= 3) | ng),
                  "only_missing_fails": ~nt & ~ng},
        "bothB": {"orig": (T >= 2) & (G >= 2), "missing_pass": ((T >= 2) | nt) & ((G >= 2) | ng),
                  "only_missing_fails": ~nt & ~ng},
    }
    for cond, variants in elig.items():
        assert np.array_equal(variants["orig"], ctx.ok[cond]), cond
    curves, meta = {}, {}
    for cond, variants in elig.items():
        for variant, eligible in variants.items():
            for freq, method in (("change", "cash"), ("change", "rotate"), ("week", "rotate")):
                targets = study.plan(ctx.book, ctx.days, ctx.rankings, ctx.col, eligible, freq, method)
                for market in ("none", "SOXX100r1", "SOXX10r1"):
                    key = f"{cond}_{freq}_{method}_{market}|{variant}"
                    curves[key] = study.replay(*args, targets, ctx.gates[market])
                    meta[key] = dict(cond=cond, rule=f"{freq}_{method}", market=market, variant=variant)
    curves["base"] = study.replay(*args, ctx.weekly, ctx.gates["none"])
    st = common.stats_table(curves)
    for key in curves:  # the 'orig' variant must reproduce Codex's ledger summary
        if key.endswith("|orig"):
            ref = codex_table.loc[key.split("|")[0]]
            np.testing.assert_allclose(st.loc[key, ["h2_krw", "h2_usd", "h2_expo"]].to_numpy(float),
                                       ref[["h2_krw", "h2_usd", "h2_expo"]].to_numpy(float), rtol=1e-5, atol=5e-6)
    base = curves["base"]
    rows = []
    for key, m in meta.items():
        r = st.loc[key]
        ctl = common.matched_control(base, r.h2_expo, "h2")
        rows.append({**m, "h1_krw": r.h1_krw, "h2_krw": r.h2_krw, "h2_krw_mdd": r.h2_krw_mdd, "h2_usd": r.h2_usd,
                     "h2_usd_mdd": r.h2_usd_mdd, "h2_expo": r.h2_expo, "jul_aug_usd": r.jul_aug_usd,
                     "sep_nov_usd": r.sep_nov_usd, "ctl_h2_usd": ctl[0], "h2_usd_minus_ctl": r.h2_usd - ctl[0]})
    out = pd.DataFrame(rows)
    b = st.loc["base"]
    print(f"\nbase: h1_krw {b.h1_krw:+.4f} h2_krw {b.h2_krw:+.4f} h2_usd {b.h2_usd:+.4f} h2_usd_mdd {b.h2_usd_mdd:+.4f} "
          f"h2_expo {b.h2_expo:.3f}")
    for rule in ("change_cash", "change_rotate", "week_rotate"):
        show(f"1b missing-grade variants, rule {rule} (returns as fractions; ctl = exposure-matched base control, USD)",
             out[out.rule == rule].drop(columns="rule").set_index(["cond", "market", "variant"]).round(4))
    out.to_csv(OUT / "missing_variants.csv", index=False)
    return out


# --------------------------------------------------------------------------- 2 forward information
def block_ci(x, block, reps=4000, seed=7):
    x = np.asarray(x, float)
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    block = int(max(1, min(block, n)))
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(reps, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    means = x[idx.reshape(reps, -1)[:, :n]].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def nw_t(x, lag):
    x = np.asarray(x, float)
    n = len(x)
    if n < 5:
        return np.nan
    e = x - x.mean()
    lag = min(lag, n - 1)
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else np.nan


def forward(opn, close, h):
    """Enter at the open of session j, exit at the close of session j+h-1 (h sessions held)."""
    out = np.full(opn.shape, np.nan)
    out[:len(opn) - h + 1] = close[h - 1:] / opn[:len(opn) - h + 1] - 1
    return out


def information(ctx, T, G, held):
    n_days, n_tk = len(ctx.days), len(ctx.tickers)
    time_u = np.zeros((n_days, n_tk), bool)
    top12 = np.zeros((n_days, n_tk), bool)
    for j, ranked in enumerate(ctx.rankings):
        for r in ranked:
            time_u[j, ctx.col[r["key"]]] = True
            top12[j, ctx.col[r["key"]]] = r["rank"] <= 12
    universes = {"TIME_all": time_u, "top12": top12, "held7": held}
    tA, gA = T >= 3, G >= 3
    signals_ = {
        "trend": {"S": T == 4, "A": T == 3, "B": T == 2, "C": T == 1, "D": T == 0, "missing": np.isnan(T),
                  "SA": T >= 3, "CD": T <= 1, "BCD": T <= 2},
        "giant": {"S": G == 4, "A": G == 3, "B": G == 2, "C": G == 1, "D": G == 0, "missing": np.isnan(G),
                  "SA": G >= 3, "CD": G <= 1, "BCD": G <= 2, "graded": ~np.isnan(G)},
        "both": {"bothA": tA & gA, "trendA_only": tA & ~gA, "giantA_only": gA & ~tA, "neither": ~tA & ~gA,
                 "not_bothA": ~(tA & gA), "trendA": tA, "giantA": gA},
    }
    for n in (10, 20, 50, 100):
        signals_[f"sma{n}"] = {"above": ctx.ok[f"sma{n}"], "below": ~ctx.ok[f"sma{n}"]}
    spreads = {"trend": [("SA-CD", "SA", "CD"), ("S-D", "S", "D"), ("SA-BCD", "SA", "BCD")],
               "giant": [("SA-CD", "SA", "CD"), ("SA-BCD", "SA", "BCD"), ("SA-missing", "SA", "missing")],
               "both": [("bothA-rest", "bothA", "not_bothA"), ("bothA-trendA_only", "bothA", "trendA_only"),
                        ("bothA-giantA_only", "bothA", "giantA_only"), ("bothA-neither", "bothA", "neither")]}
    for n in (10, 20, 50, 100):
        spreads[f"sma{n}"] = [("above-below", "above", "below")]
    buckets, spread_rows, ic_rows, count_rows = [], [], [], []
    for h in HORIZONS:
        F = forward(ctx.opn, ctx.close, h)
        for uname, U in universes.items():
            valid = U & np.isfinite(F)
            with np.errstate(invalid="ignore"):
                mean_u = np.array([F[j, valid[j]].mean() if valid[j].any() else np.nan for j in range(n_days)])
            X = F - mean_u[:, None]
            for per in PERIODS:
                pm = period_mask(ctx.days, per)
                for sname, sets in signals_.items():
                    daily = {}
                    for bname, mask in sets.items():
                        m = valid & mask & pm[:, None]
                        raw, ex = F[m], X[m]
                        with np.errstate(invalid="ignore"):
                            daily[bname] = np.array([F[j, m[j]].mean() if m[j].any() else np.nan for j in range(n_days)])
                        buckets.append(dict(universe=uname, signal=sname, bucket=bname, period=per, h=h, n=int(m.sum()),
                                            days=int(m.any(axis=1).sum()),
                                            mean=raw.mean() if len(raw) else np.nan,
                                            median=np.median(raw) if len(raw) else np.nan,
                                            hit=(raw > 0).mean() if len(raw) else np.nan,
                                            mean_ex=ex.mean() if len(ex) else np.nan,
                                            median_ex=np.median(ex) if len(ex) else np.nan,
                                            hit_ex=(ex > 0).mean() if len(ex) else np.nan))
                    for label, top, bottom in spreads[sname]:
                        d = daily[top] - daily[bottom]
                        d = d[np.isfinite(d)]
                        lo, hi = block_ci(d, h)
                        mt, mb = valid & sets[top] & pm[:, None], valid & sets[bottom] & pm[:, None]
                        pooled = (F[mt].mean() - F[mb].mean()) if mt.any() and mb.any() else np.nan
                        spread_rows.append(dict(universe=uname, signal=sname, spread=label, period=per, h=h,
                                                days=len(d), indep_windows=len(d) / h,
                                                mean_daily=d.mean() if len(d) else np.nan, ci_lo=lo, ci_hi=hi,
                                                nw_t=nw_t(d, h - 1), pooled=pooled,
                                                share_days_pos=(d > 0).mean() if len(d) else np.nan))
                for gname, Xg in (("trend", T), ("giant", G)):
                    ics = []
                    for j in np.flatnonzero(pm):
                        m = valid[j] & np.isfinite(Xg[j])
                        if m.sum() >= 5 and np.unique(Xg[j, m]).size > 1:
                            ics.append(pd.Series(Xg[j, m]).corr(pd.Series(F[j, m]), method="spearman"))
                    ics = np.array(ics)
                    lo, hi = block_ci(ics, h)
                    ic_rows.append(dict(universe=uname, grade=gname, period=per, h=h, days=len(ics),
                                        mean_ic=ics.mean() if len(ics) else np.nan, ci_lo=lo, ci_hi=hi,
                                        nw_t=nw_t(ics, h - 1)))
    for uname, U in universes.items():
        for per in PERIODS:
            pm = period_mask(ctx.days, per) & U.any(axis=1)
            row = dict(universe=uname, period=per, sessions=int(pm.sum()), names=U[pm].sum(axis=1).mean())
            for label, mask in (("trendA", tA), ("giantA", gA), ("bothA", tA & gA), ("bothB", (T >= 2) & (G >= 2)),
                                ("giant_graded", ~np.isnan(G)), ("trend_graded", ~np.isnan(T)),
                                ("sma10", ctx.ok["sma10"]), ("sma20", ctx.ok["sma20"]), ("sma50", ctx.ok["sma50"]),
                                ("sma100", ctx.ok["sma100"])):
                row[label] = (U & mask)[pm].sum(axis=1).mean()
            row["days_zero_giantA"] = int(((U & gA)[pm].sum(axis=1) == 0).sum())
            row["days_zero_bothA"] = int(((U & tA & gA)[pm].sum(axis=1) == 0).sum())
            row["days_zero_trendA"] = int(((U & tA)[pm].sum(axis=1) == 0).sum())
            count_rows.append(row)
    B, Sp, IC, CT = (pd.DataFrame(r) for r in (buckets, spread_rows, ic_rows, count_rows))
    B.to_csv(OUT / "forward_buckets.csv", index=False)
    Sp.to_csv(OUT / "forward_spreads.csv", index=False)
    IC.to_csv(OUT / "forward_ic.csv", index=False)
    show("2 mean number of names per session passing each condition (by universe and period)", CT.round(2))
    pct = lambda f, cols: f.assign(**{c: (f[c] * 100).round(2) for c in cols})  # noqa: E731
    main = B[(B.universe == "TIME_all") & B.period.isin(["H1", "JulAug", "SepNov"])]
    for sname, order in (("trend", ["S", "A", "B", "C", "D", "missing", "SA", "CD"]),
                         ("giant", ["S", "A", "B", "C", "D", "missing", "SA", "CD"]),
                         ("both", ["bothA", "trendA_only", "giantA_only", "neither"]),
                         ("sma10", ["above", "below"]), ("sma20", ["above", "below"]),
                         ("sma50", ["above", "below"]), ("sma100", ["above", "below"])):
        t = pct(main[(main.signal == sname) & main.bucket.isin(order)], ["mean", "median", "hit", "mean_ex", "median_ex", "hit_ex"])
        t["bucket"] = pd.Categorical(t.bucket, order)
        t["cell"] = (t["mean"].map("{:+.1f}".format) + "/" + t["median"].map("{:+.1f}".format) + "/"
                     + t.hit.map("{:.0f}".format) + "/" + t.mean_ex.map("{:+.1f}".format) + "/n" + t.n.astype(str))
        show(f"2 TIME_all {sname}: mean% / median% / hit% / mean excess vs universe %p / n ticker-days  "
             f"(entry next open, hold h sessions)",
             t.pivot_table(index=["bucket"], columns=["period", "h"], values="cell", aggfunc="first", observed=True))
    key = Sp[Sp.period.isin(["H1", "JulAug", "SepNov", "H2"])]
    for uname in universes:
        t = pct(key[key.universe == uname], ["mean_daily", "ci_lo", "ci_hi", "pooled", "share_days_pos"])
        t["cell"] = (t.mean_daily.map("{:+.1f}".format) + " [" + t.ci_lo.map("{:+.1f}".format) + "," + t.ci_hi.map("{:+.1f}".format)
                     + "] t" + t.nw_t.map("{:+.1f}".format) + " d" + t.days.astype(str))
        show(f"2 spreads, universe {uname}: mean daily cross-sectional spread %p [block-bootstrap 95% CI, block=h] "
             f"NW t (lag h-1), d = days with both groups",
             t.pivot_table(index=["signal", "spread"], columns=["period", "h"], values="cell", aggfunc="first"))
    t = IC[IC.universe == "TIME_all"].copy()
    t["cell"] = (t.mean_ic.map("{:+.3f}".format) + " [" + t.ci_lo.map("{:+.2f}".format) + "," + t.ci_hi.map("{:+.2f}".format)
                 + "] t" + t.nw_t.map("{:+.1f}".format))
    show("2 daily Spearman IC (numeric grade vs forward return), TIME_all, graded names",
         t.pivot_table(index="grade", columns=["period", "h"], values="cell", aggfunc="first"))
    return B, Sp, IC, CT


def breadth_information(ctx):
    br = pd.read_csv(CODEX / "breadth.csv", index_col=0).breadth
    level = br.reindex(ctx.book.days).shift().loc[ctx.days].to_numpy(float)   # known at the previous close
    assert np.isfinite(level).all()
    soxx_o, soxx_c = (f.loc[ctx.days, "SOXX"].to_numpy(float) for f in (ctx.book.opn, ctx.book.cls))
    qqq_o, qqq_c = (f.loc[ctx.days, "QQQ"].to_numpy(float) for f in (ctx.book.opn, ctx.book.cls))
    base_open, base_close = np.asarray(ctx.baseline.value, float), np.asarray(ctx.baseline.close, float)
    q1, q2 = np.quantile(level, [1 / 3, 2 / 3])
    print(f"\nbreadth used-at-session: full-2026 tercile cuts {q1:.1f}% / {q2:.1f}% (hindsight cuts); tab threshold 19%")
    groups = {"low": level < q1, "mid": (level >= q1) & (level < q2), "high": level >= q2,
              "lt19": level < 19, "ge19": level >= 19}
    rows, diffs = [], []
    for h in HORIZONS:
        fwd = {}
        for name, o, c in (("SOXX", soxx_o, soxx_c), ("QQQ", qqq_o, qqq_c), ("base_usd", base_open, base_close)):
            f = np.full(len(o), np.nan)
            f[:len(o) - h + 1] = c[h - 1:] / o[:len(o) - h + 1] - 1
            fwd[name] = f
        for per in PERIODS:
            pm = period_mask(ctx.days, per)
            for gname, g in groups.items():
                for asset, f in fwd.items():
                    m = pm & g & np.isfinite(f)
                    x = f[m]
                    rows.append(dict(period=per, h=h, group=gname, asset=asset, n=int(m.sum()),
                                     mean=x.mean() if len(x) else np.nan, median=np.median(x) if len(x) else np.nan,
                                     hit=(x > 0).mean() if len(x) else np.nan))
            for asset, f in fwd.items():
                for label, hi_g, lo_g in (("high-low_tercile", "high", "low"), ("ge19-lt19", "ge19", "lt19")):
                    ok = pm & np.isfinite(f)
                    idx = np.flatnonzero(ok)
                    a, b = groups[hi_g][idx], groups[lo_g][idx]
                    if a.sum() < 3 or b.sum() < 3:
                        diffs.append(dict(period=per, h=h, asset=asset, diff=label, n_hi=int(a.sum()), n_lo=int(b.sum())))
                        continue
                    x = f[idx]
                    rng = np.random.default_rng(11)
                    n, blk = len(x), min(h, len(x))
                    k = int(np.ceil(n / blk))
                    sims = []
                    for _ in range(3000):
                        pos = ((rng.integers(0, n, k)[:, None] + np.arange(blk)[None, :]) % n).ravel()[:n]
                        aa, bb = a[pos], b[pos]
                        if aa.any() and bb.any():
                            sims.append(x[pos][aa].mean() - x[pos][bb].mean())
                    lo, hi = np.percentile(sims, [2.5, 97.5])
                    diffs.append(dict(period=per, h=h, asset=asset, diff=label, n_hi=int(a.sum()), n_lo=int(b.sum()),
                                      mean_diff=x[a].mean() - x[b].mean(), ci_lo=lo, ci_hi=hi))
            lv = level[pm]
            for asset, f in fwd.items():
                ok = np.isfinite(f[pm])
                if ok.sum() >= 10:
                    rho = pd.Series(lv[ok]).corr(pd.Series(f[pm][ok]), method="spearman")
                    diffs.append(dict(period=per, h=h, asset=asset, diff="spearman_level_vs_fwd", n_hi=int(ok.sum()),
                                      mean_diff=rho))
    R, D = pd.DataFrame(rows), pd.DataFrame(diffs)
    R.to_csv(OUT / "breadth_buckets.csv", index=False)
    D.to_csv(OUT / "breadth_diffs.csv", index=False)
    t = R.assign(cell=(R["mean"] * 100).map("{:+.1f}".format) + "/" + (R["median"] * 100).map("{:+.1f}".format) + "/"
                 + (R.hit * 100).map("{:.0f}".format) + "/n" + R.n.astype(str))
    for asset in ("SOXX", "base_usd", "QQQ"):
        show(f"2 breadth level (known at prior close) vs forward {asset}: mean% / median% / hit% / n days",
             t[t.asset == asset].pivot_table(index="group", columns=["period", "h"], values="cell", aggfunc="first")
             .reindex(["low", "mid", "high", "lt19", "ge19"]))
    d = D.copy()
    for c in ("mean_diff", "ci_lo", "ci_hi"):
        d[c] = d[c].where(d["diff"] == "spearman_level_vs_fwd", d[c] * 100).round(2) if c in d else np.nan
    show("2 breadth differences (%p; spearman rows are rho), block bootstrap 95% CI (block=h)",
         d[d.asset.isin(["SOXX", "base_usd"])].set_index(["diff", "asset", "period", "h"]).sort_index())
    days_lt = {per: (int((level[period_mask(ctx.days, per)] < 19).sum()), int(period_mask(ctx.days, per).sum()))
               for per in PERIODS}
    print("sessions with breadth<19 used / sessions:", days_lt)
    return R, D


# --------------------------------------------------------------------------- portfolio-level decomposition
def portfolio(ctx):
    daily = pd.read_parquet(common.CACHE / "codex_daily.parquet")
    curves = {k: g.set_index("date")[common.LEDGER].astype(float) for k, g in daily.groupby("strategy")}
    st = common.stats_table(curves)
    base = curves["base"]
    keys = ["base", "half", "fast_none"] + [f"{c}_{r}_none" for c in ("trendA", "giantA", "bothA", "bothB")
                                             for r in ("week_rotate", "change_rotate", "change_cash")]
    keys += [f"stock{n}_{m}_none" for n in (10, 20, 50, 100) for m in ("cash", "rotate")]
    rows = []
    for k in keys:
        r = st.loc[k]
        row = dict(strategy=k, h1_usd=r.h1_usd, h1_expo=r.h1_expo)
        for w in ("h1", "h2", "jul_aug", "sep_nov"):
            ctl = common.matched_control(base, r[f"{w}_expo"], w)
            row.update({f"{w}_usd": r[f"{w}_usd"], f"{w}_expo": r[f"{w}_expo"], f"{w}_ctl": ctl[0],
                        f"{w}_sel": r[f"{w}_usd"] - ctl[0]})
        row["h2_usd_mdd"] = r.h2_usd_mdd
        rows.append(row)
    P = pd.DataFrame(rows).set_index("strategy")
    show("3 portfolio level (USD, float32 cache): return, mean exposure, exposure-matched base control, "
         "sel = strategy minus control (NaN when exposure exceeds base)", P.round(4))
    P.to_csv(OUT / "portfolio_selection.csv")
    return st, P


def anchor_timeline(ctx, bars, G, time_u):
    soxx = bars[giant.ANCHOR]["Close"]
    rows, prev = [], None
    for j, day in enumerate(ctx.days):
        a = giant.find_anchor(soxx.loc[:pd.Timestamp(day)])
        key = (a["peak_date"], a["low_date"])
        if key != prev:
            rows.append(dict(close_date=day, peak_date=a["peak_date"], low_date=a["low_date"], crash=a["crash"]))
            prev = key
    A = pd.DataFrame(rows)
    show(f"Giant anchor (SOXX deepest drawdown in 252 sessions) changes during 2026: {len(A)} distinct anchors", A)
    return A


def decomposition(ctx, weight):
    """Split each daily cash-overlay rule into exposure timing and name selection.

    timing control = the whole base basket held at the rule's daily pass fraction (share of base target
    weight that passes, known at the prior close); constant control = base at the rule's mean exposure.
    selection = rule - timing control; timing = timing control - constant control.
    """
    args = (ctx.days, ctx.opn, ctx.close, ctx.fx)
    base = study.replay(*args, ctx.weekly, ctx.gates["none"])
    total = weight.sum(axis=1)
    rows = []
    for cond in ("trendA", "giantA", "bothA", "bothB", "sma10", "sma20", "sma50", "sma100"):
        targets = study.plan(ctx.book, ctx.days, ctx.rankings, ctx.col, ctx.ok[cond], "change", "cash")
        rule = study.replay(*args, targets, ctx.gates["none"])
        frac = np.divide((weight * ctx.ok[cond]).sum(axis=1), total, out=np.zeros(len(total)), where=total > 0)
        timing = common.replay_frac(*args, ctx.weekly, np.clip(frac, 0, 1))
        for w in ("h1", "h2", "jul_aug", "sep_nov"):
            a, b = common.WINDOWS[w]
            r, t = common.window_stats(rule, a, b), common.window_stats(timing, a, b)
            ctl = common.matched_control(base, r["expo"], w)[0]
            rows.append(dict(cond=cond, window=w, rule_usd=r["usd"], rule_expo=r["expo"], timing_ctl_usd=t["usd"],
                             timing_ctl_expo=t["expo"], const_ctl_usd=ctl, selection=r["usd"] - t["usd"],
                             timing=t["usd"] - ctl, total=r["usd"] - ctl, rule_usd_mdd=r["usd_mdd"],
                             timing_ctl_mdd=t["usd_mdd"]))
    D = pd.DataFrame(rows)
    show("3 change_cash rules (no market gate), USD: rule vs same-daily-exposure base (timing control) vs "
         "constant-exposure base", D.set_index(["cond", "window"]).round(4))
    D.to_csv(OUT / "decomposition.csv", index=False)
    return D


def held_name_detail(ctx, T, G, held):
    """Which held names carry the within-basket H2 spreads; leave-one-name-out; dates of missing Giant grades."""
    h2 = period_mask(ctx.days, "H2")
    rows = []
    for h in (10, 20):
        F = forward(ctx.opn, ctx.close, h)
        valid = held & np.isfinite(F) & h2[:, None]
        for label, top, bottom in (("trend SA-BCD", T >= 3, T <= 2), ("giant SA-BCD", G >= 3, G <= 2),
                                   ("giant SA-CD", G >= 3, G <= 1), ("sma50 above-below", ctx.ok["sma50"], ~ctx.ok["sma50"])):
            def spread(drop=None):
                v = valid.copy()
                if drop is not None:
                    v[:, drop] = False
                d = []
                for j in np.flatnonzero(v.any(axis=1)):
                    a, b = v[j] & top[j], v[j] & bottom[j]
                    if a.any() and b.any():
                        d.append(F[j, a].mean() - F[j, b].mean())
                d = np.array(d)
                return (d.mean() if len(d) else np.nan), len(d), block_ci(d, h)
            full, n, ci = spread()
            loo = {ctx.tickers[i]: spread(i)[0] for i in np.flatnonzero(valid.any(axis=0))}
            lo_name, hi_name = min(loo, key=loo.get), max(loo, key=loo.get)
            rows.append(dict(h=h, spread=label, full=full, days=n, ci_lo=ci[0], ci_hi=ci[1],
                             loo_min=loo[lo_name], loo_min_drop=lo_name, loo_max=loo[hi_name], loo_max_drop=hi_name,
                             loo_drop_DELL=loo.get("DELL"), loo_drop_AMD=loo.get("AMD")))
    L = pd.DataFrame(rows)
    show("2 held7, H2: within-basket spread and leave-one-name-out (fractions)", L.round(4))
    F = forward(ctx.opn, ctx.close, 20)
    names = []
    for i in np.flatnonzero((held & h2[:, None]).any(axis=0)):
        m = held[:, i] & h2 & np.isfinite(F[:, i])
        up, down = m & (T[:, i] >= 3), m & (T[:, i] <= 2)
        names.append(dict(ticker=ctx.tickers[i], held_days_with_fwd20=int(m.sum()), trendA_days=int(up.sum()),
                          giantA_days=int((m & (G[:, i] >= 3)).sum()),
                          fwd20_all=F[m, i].mean() if m.any() else np.nan,
                          fwd20_trendA=F[up, i].mean() if up.any() else np.nan,
                          fwd20_trend_below=F[down, i].mean() if down.any() else np.nan))
    show("2 held7, H2: per-name forward 20-session return by trend state (fractions)",
         pd.DataFrame(names).set_index("ticker").sort_values("held_days_with_fwd20", ascending=False).round(3))
    miss = held & np.isnan(G) & h2[:, None]
    spans = []
    for i in np.flatnonzero(miss.any(axis=0)):
        d = [ctx.days[j] for j in np.flatnonzero(miss[:, i])]
        spans.append(dict(ticker=ctx.tickers[i], days=len(d), first=d[0], last=d[-1]))
    show("1b H2 held ticker-days with no Giant grade", pd.DataFrame(spans))
    per_day = pd.Series(miss.sum(axis=1), index=ctx.days)
    print("H2 sessions by number of held names without a Giant grade:", per_day[h2].value_counts().sort_index().to_dict())
    print("H2 sessions with >=3 held names missing a Giant grade:", list(per_day[h2][per_day[h2] >= 3].index))
    L.to_csv(OUT / "held_leave_one_out.csv", index=False)
    return L


def breadth_gate(ctx):
    """Codex's post-selection breadth diagnostic rebuilt (gate = breadth at prior close >= 19%)."""
    br = pd.read_csv(CODEX / "breadth.csv", index_col=0).breadth
    level = br.reindex(ctx.book.days).shift().loc[ctx.days].to_numpy(float)
    args = (ctx.days, ctx.opn, ctx.close, ctx.fx)
    curves = {"base": study.replay(*args, ctx.weekly, ctx.gates["none"]),
              "usd_cash": study.replay(*args, {}, ctx.gates["none"])}
    for cut in (10, 15, 19, 25):
        curves[f"breadth>={cut}"] = study.replay(*args, ctx.weekly, level >= cut)
    st = common.stats_table(curves)
    cols = [f"{w}_{m}" for w in ("h1", "h2", "jul_aug", "sep_nov", "oct_nov") for m in ("krw", "usd", "expo")]
    cols += ["h2_krw_mdd", "full_krw", "full_krw_mdd"]
    show("2 breadth gate on the weekly base (19 is the tab threshold calibrated on 2026-02~09; other cuts are "
         "sensitivity only, hindsight)", st[cols].round(4).T)
    gate = level >= 19
    flips = [(ctx.days[j], "in" if gate[j] else "out") for j in range(1, len(gate)) if gate[j] != gate[j - 1]]
    print("breadth>=19 gate flips (session, state):", flips)
    st[cols].to_csv(OUT / "breadth_gate.csv")
    return st


def control_census(ctx):
    """How many of Codex's candidates beat the base held at their own mean H2 exposure (USD)."""
    daily = pd.read_parquet(common.CACHE / "codex_daily.parquet")
    curves = {k: g.set_index("date")[common.LEDGER].astype(float) for k, g in daily.groupby("strategy")}
    st = common.stats_table(curves)
    base = curves["base"]
    rows = []
    for key in st.index:
        if key in ("base", "half"):
            continue
        family = ("switch" if key.startswith("switch_") else "trendA" if key.startswith("trendA") else
                  "giantA" if key.startswith("giantA") else "bothA" if key.startswith("bothA") else
                  "bothB" if key.startswith("bothB") else "stockSMA" if key.startswith("stock") else
                  "fast" if key.startswith("fast") else "weekly_gate")
        for w in ("h2", "jul_aug", "sep_nov"):
            ctl, ctl_mdd = common.matched_control(base, st.at[key, f"{w}_expo"], w)
            rows.append(dict(strategy=key, family=family, window=w, usd=st.at[key, f"{w}_usd"],
                             expo=st.at[key, f"{w}_expo"], ctl=ctl, diff=st.at[key, f"{w}_usd"] - ctl,
                             mdd=st.at[key, f"{w}_usd_mdd"], ctl_mdd=ctl_mdd,
                             beats_base=st.at[key, f"{w}_usd"] > st.at["base", f"{w}_usd"]))
    C = pd.DataFrame(rows)
    ok = C[C.ctl.notna()]
    summary = ok.groupby(["window", "family"]).agg(n=("diff", "size"), beat_ctl=("diff", lambda s: int((s > 0).sum())),
                                                   median_diff=("diff", "median"), best_diff=("diff", "max"),
                                                   worst_diff=("diff", "min"), mean_expo=("expo", "mean"))
    show("3 Codex candidates vs base held at the candidate's own mean exposure (USD return difference); "
         "candidates with exposure above base are excluded", summary.round(4))
    tot = ok.groupby("window").agg(n=("diff", "size"), beat_ctl=("diff", lambda s: int((s > 0).sum())),
                                   beat_base=("beats_base", "sum"), median_diff=("diff", "median"))
    excluded = C[C.ctl.isna()].groupby("window").size().to_dict()
    show(f"3 all candidates with a defined control (excluded for exposure above base: {excluded})", tot.round(4))
    top = ok[ok.window == "h2"].sort_values("diff", ascending=False).head(8)
    show("3 H2: candidates with the largest margin over their exposure-matched control", top.set_index("strategy").round(4))
    C.to_csv(OUT / "control_census.csv", index=False)
    return C


def post_design(ctx, T, G):
    """Signals from 2026-10-12 on: the only sessions after both tabs' formulas were fixed (Giant grade 10-06,
    trend grade 10-12). TIME_all universe."""
    n_days = len(ctx.days)
    U = np.zeros((n_days, len(ctx.tickers)), bool)
    for j, ranked in enumerate(ctx.rankings):
        for r in ranked:
            U[j, ctx.col[r["key"]]] = True
    pm = np.array([d >= "2026-10-12" for d in ctx.days])
    rows = []
    for h in (5, 10):
        F = forward(ctx.opn, ctx.close, h)
        valid = U & np.isfinite(F) & pm[:, None]
        for label, top, bottom in (("trend SA-CD", T >= 3, T <= 1), ("trend SA-BCD", T >= 3, T <= 2),
                                   ("giant SA-CD", G >= 3, G <= 1), ("giant SA-missing", G >= 3, np.isnan(G)),
                                   ("bothA-rest", (T >= 3) & (G >= 3), ~((T >= 3) & (G >= 3))),
                                   ("sma50 above-below", ctx.ok["sma50"], ~ctx.ok["sma50"]),
                                   ("sma100 above-below", ctx.ok["sma100"], ~ctx.ok["sma100"])):
            d = []
            for j in np.flatnonzero(valid.any(axis=1)):
                a, b = valid[j] & top[j], valid[j] & bottom[j]
                if a.any() and b.any():
                    d.append(F[j, a].mean() - F[j, b].mean())
            d = np.array(d)
            lo, hi = block_ci(d, h)
            rows.append(dict(h=h, spread=label, days=len(d), indep_windows=len(d) / h,
                             mean=d.mean() if len(d) else np.nan, ci_lo=lo, ci_hi=hi, nw_t=nw_t(d, h - 1)))
    P = pd.DataFrame(rows)
    show("2 post-design sessions only (signal sessions 2026-10-12..), TIME_all, fractions", P.round(4))
    P.to_csv(OUT / "post_design.csv", index=False)
    return P


def extra(ctx, T, G, held, weight):
    decomposition(ctx, weight)
    held_name_detail(ctx, T, G, held)
    breadth_gate(ctx)
    control_census(ctx)
    post_design(ctx, T, G)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = common.context()
    codex_trend = pd.read_csv(CODEX / "trend_grades.csv", index_col=0)
    codex_giant = pd.read_csv(CODEX / "giant_grades.csv", index_col=0)
    assert codex_trend.index.tolist() == ctx.days and codex_trend.columns.tolist() == sorted(ctx.tickers)
    bars = load_bars()
    print(f"cached grade bars: {len(bars)} symbols; last SOXX bar {bars['SOXX'].index[-1].date()}")

    # ---- 1a
    work_trend = json.loads((ROOT / "data/derived/desk/trend.json").read_text(encoding="utf-8"))
    work_giant = json.loads((ROOT / "data/derived/desk/giant.json").read_text(encoding="utf-8"))
    old_trend = git_json("7c28f39b", "data/derived/desk/trend.json")
    old_giant = git_json("7c28f39b", "data/derived/desk/giant.json")
    print(f"working tree: trend computed {work_trend['computed_at']} as_of {work_trend['as_of']}; giant computed "
          f"{work_giant['computed_at']} anchor last_date {work_giant['anchor']['last_date']}")
    print(f"7c28f39b: trend computed {old_trend['computed_at']} as_of {old_trend['as_of']}; giant computed "
          f"{old_giant['computed_at']} anchor last_date {old_giant['anchor']['last_date']}")
    ts = [trend_compare("7c28f39b", old_trend, codex_trend, ctx.book, bars),
          trend_compare("worktree", work_trend, codex_trend, ctx.book, bars)]
    gs = [giant_compare("7c28f39b", old_giant, codex_giant, old_giant["anchor"]["last_date"], bars),
          giant_compare("worktree_vs_1117", work_giant, codex_giant, "2026-11-17", bars)]
    giant_intraday_check(work_giant, bars)
    pd.concat(ts).to_csv(OUT / "trend_agreement.csv")
    pd.DataFrame(gs).to_csv(OUT / "giant_agreement.csv", index=False)

    # ---- 1b
    T, G = grade_arrays(ctx)
    assert np.array_equal(T >= 3, ctx.ok["trendA"]) and np.array_equal(G >= 3, ctx.ok["giantA"])
    held, weight = held_mask(ctx)
    coverage(ctx, T, G, held, weight)
    st, P = portfolio(ctx)
    missing_variants(ctx, T, G, st)

    # ---- 2
    time_u = np.zeros(held.shape, bool)
    for j, ranked in enumerate(ctx.rankings):
        for r in ranked:
            time_u[j, ctx.col[r["key"]]] = True
    anchor_timeline(ctx, bars, G, time_u)
    information(ctx, T, G, held)
    breadth_information(ctx)
    extra(ctx, T, G, held, weight)


if __name__ == "__main__":
    if "--extra" in sys.argv:   # only the decomposition / leave-one-out / breadth-gate blocks
        OUT.mkdir(parents=True, exist_ok=True)
        _ctx = common.context()
        _T, _G = grade_arrays(_ctx)
        _held, _weight = held_mask(_ctx)
        extra(_ctx, _T, _G, _held, _weight)
    else:
        main()
