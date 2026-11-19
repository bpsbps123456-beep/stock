"""ext_universe: switching WHAT is held when the market turns (family "universe").

Run from the repository root:  py data/derived/analysis/desk_defense_check_20261119/ext_universe.py

Blocks (every decision executed at the open of session j uses closes through session j-1 and
TIME/KoAct PDFs dated before session j; fills at that open through study.replay / common.replay_frac):

U1  sector switch inside the TIME list while a market gate is off
    gates: SOXX10r1 SOXX20r1 SOXX20r3 SOXX50r1 SOXX100r1 QQQ10r1 breadth19 (7)
    a) top-N non-semiconductor TIME names, L in {120,20} x N in {3,5,7} x weight in {time,equal}
       = 12 baskets x (7 gates + "always" reference) = 96
    b) base basket with semiconductors removed: {rest kept at weight (cash remainder), rest re-projected
       to 100%} x (7 gates + always) = 16
    c) base basket scaled to frac in {0, .25, .5, .75} while the gate is off: 4 x 7 = 28
U2  permanent caps: semis cap {0,10,25,35,40,45,50,55,60,100}% x stock cap {15,20,25}% x gate
    {none, SOXX20r3, SOXX100r1, breadth19} = 120 (50/25/none is the base, asserted identical)
U3  AI-map universe (US-listed tickers of watchlist.AI_SUBSECTOR, today's map = hindsight membership)
    weekly top-N equal weight, 25% stock / 60% sector / 50% semis caps, no keep-buffer
    filter {all, near (within 10% of the all-time high), giantA (recovery grade >= A)} x N {5,7,10}
      x L {20,60,120} x gate {none, SOXX20r3, SOXX100r1, breadth19} = 108
    secstr: top K {2,3} subsectors by share of members within 10% of the high, then top-N by L
      = 2 x 3 x 3 x 4 = 72
    control on the TIME list with the same equal-weight engine: N {5,7,10} x L {20,60,120} x 4 gates = 36
U4  TIME + KoAct US rosters (PIT): filter {all, trendA} x N {5,7,10} x L {20,60,120} x 4 gates = 72
Total 140 + 120 + 216 + 72 = 548 ledgers including the base.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from bisect import bisect_left
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402  shared harness (adds the repo root and Codex's folder to sys.path)
import signals  # noqa: E402  Codex: holdings_history / grade_trend (pure, no file writes)
import watchlist  # noqa: E402
from desk import config, giant  # noqa: E402
from desk import strategies as S  # noqa: E402

study = common.study
FAMILY = "universe"
FROZEN = HERE / "frozen_ai"
GRADE_BARS = common.CODEX / "cache" / "grade_bars"
FIELDS = {"Open": "prices_open.csv", "Close": "prices_close.csv", "Adj Close": "prices_adj_close.csv"}
T_TEST = 150

GATES_U1 = ("SOXX10r1", "SOXX20r1", "SOXX20r3", "SOXX50r1", "SOXX100r1", "QQQ10r1", "breadth19")
GATES_X = ("none", "SOXX20r3", "SOXX100r1", "breadth19")
U1A_L, U1A_N, U1A_W = (120, 20), (3, 5, 7), ("time", "equal")
U1C_FRAC = (0.0, 0.25, 0.5, 0.75)
U2_SEMIS = (0.0, 0.10, 0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 1.00)
U2_STOCK = (0.15, 0.20, 0.25)
X_N, X_L = (5, 7, 10), (20, 60, 120)
U3_FILTERS = ("all", "near", "giantA")
U3_K = (2, 3)
U4_FILTERS = ("all", "trendA")

NEAR = giant.ai_ath_gap.NEAR_PCT / 100          # -0.10: the Giant tab's "near the high" line
BREADTH_SPLIT = giant.BREADTH_CHECK["split"]    # 19%, calibrated by the desk on 2026-02~09 (hindsight for H1)
SEMI_GROUPS = {"칩 설계", "파운드리·후공정", "메모리·스토리지", "장비·소재·테스트"}
AI_TOP = {s: top for top, subs in watchlist.AI_UNIVERSE.items() for syms in subs.values() for s in syms}
AI_ALL = sorted(watchlist.AI_SUBSECTOR)
AI_US = sorted(s for s in watchlist.AI_SUBSECTOR if "." not in s)
OTHER = "기타(비반도체)"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def _digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_downloaded(symbols):
    """Open / Close / Adj Close for the wider universes, frozen under frozen_ai/ on first run."""
    meta_path = FROZEN / "sources.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for name, digest in meta["files"].items():
            if _digest(FROZEN / name) != digest:
                raise ValueError(f"Frozen AI price input changed: {name}")
        if not set(symbols) <= set(meta["requested"]):
            raise ValueError("Frozen AI prices do not cover the requested symbols; delete frozen_ai/ to rebuild")
        return {field: pd.read_csv(FROZEN / name, index_col=0) for field, name in FIELDS.items()}, meta
    import yfinance as yf
    symbols = sorted(symbols)
    parts = []
    for i in range(0, len(symbols), 40):
        batch = symbols[i:i + 40]
        frame = yf.download(batch, start="2025-06-01", end="2026-11-18", auto_adjust=False,
                            threads=6, progress=False, group_by="column")
        parts.append(frame[list(FIELDS)])
    raw = pd.concat(parts, axis=1)
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).strftime("%Y-%m-%d")
    raw = raw[raw.index <= "2026-11-17"]
    FROZEN.mkdir(exist_ok=True)
    frames = {}
    for field, name in FIELDS.items():
        frame = raw[field].dropna(axis=1, how="all").sort_index(axis=1)
        frame.index.name = "date"
        frame.to_csv(FROZEN / name, float_format="%.6f")
        frames[field] = pd.read_csv(FROZEN / name, index_col=0)
    meta = {"source": "yfinance download, auto_adjust=False, 2025-06-01..2026-11-18 (last row 2026-11-17)",
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "requested": symbols,
            "missing": sorted(set(symbols) - set(frames["Close"].columns)),
            "files": {name: _digest(FROZEN / name) for name in FIELDS.values()}}
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return frames, meta


def load_bars(symbols):
    """Codex's cached daily High/Close since 2010 (read only)."""
    return {s: pd.read_parquet(GRADE_BARS / f"{s}.parquet") for s in sorted(symbols)
            if (GRADE_BARS / f"{s}.parquet").exists()}


def gap_frame(bars, calendar):
    """Close / running all-time high - 1 for the AI map, as desk.giant.breadth_series computes it."""
    gaps, counts = {}, {}
    for s in AI_ALL:
        b = bars.get(s)
        if b is None or len(b) < 60:
            continue
        b = b[b.index >= "2010-01-01"]
        gaps[s] = (b["Close"] / b["High"].cummax() - 1).reindex(calendar, method="ffill")
        counts[s] = pd.Series(np.arange(1, len(b) + 1), index=b.index).reindex(calendar, method="ffill")
    gap, n = pd.DataFrame(gaps), pd.DataFrame(counts)
    return gap.where(n >= 60)


def label_of(symbol, book_label=None):
    """Cap sector: the desk label when it has one, else the AI-map group (chip groups share the semis budget)."""
    if book_label is not None:
        return book_label
    label = watchlist.resolve_sector(symbol, default=S.UNKNOWN)
    if label != S.UNKNOWN:
        return label
    top = AI_TOP.get(watchlist.AI_UNIVERSE_ALIASES.get(symbol, symbol))
    if top is None:
        return OTHER
    return "AI 반도체" if top in SEMI_GROUPS else top


def build_inputs(ctx, book, ext, bars, histories):
    """Everything a decision may read. `book`, `ext` and `bars` may be perturbed copies."""
    days, start, col, tickers = ctx.days, ctx.start, ctx.col, ctx.tickers
    rankings = [S.momentum_ranking(book, start + j, day)[1] for j, day in enumerate(days)]
    gates = study.market_gates(book, days)
    calendar = pd.DatetimeIndex(book.days)
    gap = gap_frame(bars, calendar[calendar >= "2025-12-01"])
    breadth = (gap >= NEAR).sum(axis=1) / gap.notna().sum(axis=1) * 100
    gates["breadth19"] = (breadth.shift().reindex(pd.DatetimeIndex(days)) >= BREADTH_SPLIT).to_numpy()

    extras = sorted(set(ext["Close"].columns) - set(tickers))
    symbols = list(tickers) + extras
    xcol = {s: i for i, s in enumerate(symbols)}

    def wide(book_frame, field):
        left = book_frame.reindex(index=book.days, columns=tickers)
        right = ext[field].reindex(index=book.days, columns=extras)
        return pd.concat([left, right], axis=1).to_numpy(float)

    adj, opn, cls = wide(book.adj, "Adj Close"), wide(book.opn, "Open"), wide(book.cls, "Close")
    labels = {s: label_of(s) for s in extras}
    labels.update({book.ticker[k]: book.sector[k] for k in col})
    keymap = {}
    for dates, snaps in histories.values():
        for day in dates:
            for key, sym in snaps[day].items():
                keymap.setdefault(key, sym)
    keymap.update({k: book.ticker[k] for k in col})
    return SimpleNamespace(
        ctx=ctx, book=book, days=days, start=start, col=col, tickers=tickers, rankings=rankings,
        gates=gates, gap=gap, breadth=breadth, bars=bars, histories=histories, keymap=keymap,
        symbols=symbols, xcol=xcol, labels=labels, adj=adj,
        opn=opn[start:], cls=cls[start:], opn_book=opn[start:, :len(tickers)], cls_book=cls[start:, :len(tickers)],
        semis_col=np.array([S.in_semis_budget(book.sector[k]) for k in col]))


def perturbed(ctx, book, ext, bars, histories, t, rng=None):
    """Copy of the inputs with every price after session index t changed (x1.5, or random x0.5..2)."""
    cut = ctx.days[t]

    def bump(frame, after):
        out = frame.copy()
        rows = np.flatnonzero(after)
        block = out.iloc[rows].to_numpy(float)
        factor = 1.5 if rng is None else rng.uniform(0.5, 2.0, size=block.shape)
        out.iloc[rows] = block * factor
        return out

    book2 = dataclasses.replace(book, **{name: bump(getattr(book, name), np.array(getattr(book, name).index) > cut)
                                         for name in ("adj", "opn", "cls")})
    ext2 = {field: bump(frame, np.array(frame.index) > cut) for field, frame in ext.items()}
    stamp = pd.Timestamp(cut)
    bars2 = {s: bump(b, np.asarray(b.index > stamp)) for s, b in bars.items()}
    return build_inputs(ctx, book2, ext2, bars2, histories)


# ---------------------------------------------------------------------------
# decision builders
# ---------------------------------------------------------------------------

def schedule_days(inp):
    """Sessions on which the weekly rule publishes: the first session with a ranking, then ISO week starts."""
    out, started = [], False
    for j, (day, ranked) in enumerate(zip(inp.days, inp.rankings)):
        week = j == 0 or S.iso_week(day) != S.iso_week(inp.days[j - 1])
        if not ranked or (started and not week):
            continue
        out.append(j)
        started = True
    return out


def momentum(inp, j, length):
    """Adjusted-close return over `length` sessions ending at the close before session j (all ext columns)."""
    k = inp.start + j
    with np.errstate(invalid="ignore", divide="ignore"):
        m = inp.adj[k - 1] / inp.adj[k - 1 - length] - 1
    return np.where(np.isfinite(m) & (inp.adj[k - 1 - length] > 0), m, np.nan)


def capped(inp, raw, width):
    target = S.project_caps(raw, inp.labels)
    w = np.zeros(width)
    for sym, v in target.items():
        w[inp.xcol[sym]] = v
    return w


def nonsemi_plan(inp, length, n, weight):
    """U1a: weekly top-n non-semiconductor TIME names by `length`-session momentum."""
    targets = {}
    for j in schedule_days(inp):
        m = momentum(inp, j, length)
        pool = [r for r in inp.rankings[j] if not S.in_semis_budget(r["sector"]) and np.isfinite(m[inp.col[r["key"]]])]
        pool.sort(key=lambda r: (-m[inp.col[r["key"]]], r["key"]))
        top = pool[:n]
        if weight == "time":
            total = sum(r["time_weight"] for r in top)
            raw = {r["ticker"]: r["time_weight"] / total for r in top}
        else:
            raw = {r["ticker"]: 1 / len(top) for r in top}
        targets[j] = capped(inp, raw, len(inp.tickers))
    return targets


def drop_semis_plans(inp, weekly):
    """U1b: the base 7 names without the semiconductor budget names (cash remainder / re-projected)."""
    cash, renorm, prev = {}, {}, set()
    for j in schedule_days(inp):
        ranked = inp.rankings[j]
        prev = set(S.select(ranked, prev))
        rows = [r for r in ranked if r["key"] in prev]
        total = sum(r["time_weight"] for r in rows)
        base = capped(inp, {r["ticker"]: r["time_weight"] / total for r in rows}, len(inp.tickers))
        np.testing.assert_allclose(base, weekly[j], rtol=0, atol=1e-12)
        cash[j] = np.where(inp.semis_col, 0.0, weekly[j])
        rest = [r for r in rows if not S.in_semis_budget(r["sector"])]
        total = sum(r["time_weight"] for r in rest)
        renorm[j] = capped(inp, {r["ticker"]: r["time_weight"] / total for r in rest}, len(inp.tickers))
    return cash, renorm


def topn_plan(inp, candidates_at, length, n, allowed_at=None):
    """Weekly top-n by momentum, equal weight under the desk caps; fewer passing names leave cash."""
    targets = {}
    for j in schedule_days(inp):
        m = momentum(inp, j, length)
        names = [s for s in candidates_at(j) if s in inp.xcol and np.isfinite(m[inp.xcol[s]])]
        if allowed_at is not None:
            allowed = allowed_at(j)
            names = [s for s in names if s in allowed]
        names.sort(key=lambda s: (-m[inp.xcol[s]], s))
        top = names[:n]
        targets[j] = capped(inp, {s: 1 / len(top) for s in top}, len(inp.symbols))
    return targets


def near_sets(inp):
    """j -> AI-map symbols within 10% of their all-time high at the close before session j."""
    out = {}
    for j in schedule_days(inp):
        row = inp.gap.loc[pd.Timestamp(inp.days[j - 1])]
        out[j] = set(row.index[row >= NEAR])
    return out


def giant_sets(inp):
    """j -> US AI-map symbols with Giant recovery grade S/A at the close before session j (signals.grade_giant)."""
    out, grades = {}, {}
    for j in schedule_days(inp):
        t = pd.Timestamp(inp.days[j - 1])
        anchor = giant.find_anchor(inp.bars[giant.ANCHOR].loc[:t, "Close"])
        passed = set()
        for s in AI_US:
            history = inp.bars[s].loc[:t]
            if len(history) and (t - history.index[-1]).days <= 7:
                grade = signals.GRADE.get(giant.grade_of(giant.own_recovery(history, anchor)), np.nan)
                grades[inp.days[j - 1], s] = grade
                if grade >= 3:
                    passed.add(s)
        out[j] = passed
    return out, grades


def strong_subsector_sets(inp, k):
    """j -> US members of the k subsectors with the largest share of members within 10% of the high."""
    members = {}
    for sym, sub in watchlist.AI_SUBSECTOR.items():
        members.setdefault(sub, []).append(sym)
    out = {}
    for j in schedule_days(inp):
        row = inp.gap.loc[pd.Timestamp(inp.days[j - 1])]
        scored = []
        for sub, syms in members.items():
            values = row.reindex(syms).dropna()
            us = [s for s in syms if "." not in s]
            if len(values) < giant.GRADE_MIN or not us:
                continue
            scored.append((-(values >= NEAR).mean(), -values.median(), sub, us))
        scored.sort(key=lambda x: x[:3])
        out[j] = {s for item in scored[:k] for s in item[3]}
    return out


def roster_sets(inp):
    """j -> traded symbols of the TIME + KoAct US holdings published before session j."""
    out = {}
    for j in schedule_days(inp):
        names = set()
        for dates, snaps in inp.histories.values():
            i = bisect_left(dates, inp.days[j]) - 1
            if i >= 0:
                names.update(inp.keymap[key] for key in snaps[dates[i]])
        out[j] = names
    return out


def trend_sets(inp):
    """j -> symbols graded S/A by the trend tab at the close before session j (signals.grade_trend, wider columns)."""
    fake = SimpleNamespace(ticker=inp.keymap)
    grades, _ = signals.grade_trend(fake, inp.bars_trend, inp.histories, inp.days)
    out = {}
    for j in schedule_days(inp):
        row = grades.loc[inp.days[j - 1]]
        out[j] = set(row.index[row >= 3])
    return out, grades


def decisions(inp):
    """key -> dict(space, targets, frac, cfg). frac is the per-session exposure fraction (1 = follow targets)."""
    out = {}
    days, gates = inp.days, inp.gates
    ones = np.ones(len(days))
    any_ok = np.ones((len(days), len(inp.tickers)), dtype=bool)

    def add(key, space, targets, frac, **cfg):
        if key in out:
            raise ValueError(f"Duplicate key {key}")
        base = dict(subfamily="", block="", gate="none", universe="TIME", filter="", lookback="", n="",
                    weight="", k="", semis_cap="", stock_cap="", frac_off="")
        base.update(cfg)
        out[key] = dict(space=space, targets=targets, frac=np.asarray(frac, dtype=float), cfg=base)

    weekly = study.plan(inp.book, days, inp.rankings, inp.col, any_ok, "week", "rotate")
    add("base", "book", weekly, ones, subfamily="base", block="base", semis_cap=50, stock_cap=25,
        lookback=120, n=7, weight="time")

    # U1a: top-N non-semiconductor TIME names while the gate is off
    for length in U1A_L:
        for n in U1A_N:
            for weight in U1A_W:
                alt = nonsemi_plan(inp, length, n, weight)
                cfg = dict(block="U1", lookback=length, n=n, weight=weight, filter="non_semis")
                add(f"u1a_L{length}_N{n}_{weight}_always", "book", alt, ones,
                    subfamily="U1a_nonsemi_always", gate="always", **cfg)
                for g in GATES_U1:
                    switched = study.switched_targets(days, inp.opn_book, weekly, alt, gates[g])
                    add(f"u1a_L{length}_N{n}_{weight}_{g}", "book", switched, ones,
                        subfamily=f"U1a_nonsemi_L{length}_switch", gate=g, **cfg)

    # U1b: base basket without semiconductors while the gate is off
    cash, renorm = drop_semis_plans(inp, weekly)
    for name, alt in (("cash", cash), ("renorm", renorm)):
        cfg = dict(block="U1", lookback=120, n=7, weight="time", filter=f"drop_semis_{name}")
        add(f"u1b_{name}_always", "book", alt, ones, subfamily=f"U1b_drop_semis_{name}_always", gate="always", **cfg)
        for g in GATES_U1:
            switched = study.switched_targets(days, inp.opn_book, weekly, alt, gates[g])
            add(f"u1b_{name}_{g}", "book", switched, ones, subfamily=f"U1b_drop_semis_{name}_switch", gate=g, **cfg)

    # U1c: base basket scaled while the gate is off (0 = Codex's weekly_<gate> cash exit)
    for scale in U1C_FRAC:
        for g in GATES_U1:
            add(f"u1c_f{int(round(100 * scale)):02d}_{g}", "book", weekly, np.where(gates[g], 1.0, scale),
                subfamily=f"U1c_scale_{int(round(100 * scale)):02d}", block="U1", gate=g, frac_off=scale,
                lookback=120, n=7, weight="time")

    # U2: permanent caps
    for semis in U2_SEMIS:
        for stock in U2_STOCK:
            with patch.dict(config.RULE, {"semis_cap": semis, "stock_cap": stock}):
                targets = study.plan(inp.book, days, inp.rankings, inp.col, any_ok, "week", "rotate")
            is_base = abs(semis - 0.5) < 1e-12 and abs(stock - 0.25) < 1e-12
            if is_base:
                assert sorted(targets) == sorted(weekly)
                for j in weekly:
                    np.testing.assert_array_equal(targets[j], weekly[j])
            for g in GATES_X:
                if is_base and g == "none":
                    continue
                add(f"u2_s{int(round(100 * semis)):03d}_k{int(round(100 * stock))}_{g}", "book", targets,
                    gates[g].astype(float), subfamily="U2_caps_" + ("nogate" if g == "none" else "gated"),
                    block="U2", gate=g, semis_cap=round(100 * semis), stock_cap=round(100 * stock),
                    lookback=120, n=7, weight="time")

    # U3: AI-map universe (today's membership) and the equal-weight TIME control
    near = near_sets(inp)
    giant_a, inp.giant_grades = giant_sets(inp)
    filters = {"all": None, "near": near.__getitem__, "giantA": giant_a.__getitem__}
    time_names = {j: [r["ticker"] for r in inp.rankings[j]] for j in schedule_days(inp)}
    strong = {k: strong_subsector_sets(inp, k) for k in U3_K}
    for length in X_L:
        for n in X_N:
            plans = {f"u3_{name}_L{length}_N{n}": (topn_plan(inp, lambda j: AI_US, length, n, allowed),
                                                    dict(subfamily=f"U3_ai_{name}", universe="AI_US", filter=name))
                     for name, allowed in filters.items()}
            for k in U3_K:
                plans[f"u3_secstr{k}_L{length}_N{n}"] = (
                    topn_plan(inp, lambda j: AI_US, length, n, strong[k].__getitem__),
                    dict(subfamily="U3_ai_secstr", universe="AI_US", filter="sector_strength", k=k))
            plans[f"u3ctl_time_L{length}_N{n}"] = (
                topn_plan(inp, time_names.__getitem__, length, n),
                dict(subfamily="U3ctl_time_equal", universe="TIME", filter="all"))
            for stem, (targets, cfg) in plans.items():
                for g in GATES_X:
                    add(f"{stem}_{g}", "ext", targets, gates[g].astype(float), block="U3", gate=g,
                        lookback=length, n=n, weight="equal", semis_cap=50, stock_cap=25, **cfg)

    # U4: TIME + KoAct US rosters
    rosters = roster_sets(inp)
    trend_a, inp.trend_grades = trend_sets(inp)
    for length in X_L:
        for n in X_N:
            for name, allowed in (("all", None), ("trendA", trend_a.__getitem__)):
                targets = topn_plan(inp, lambda j: sorted(rosters[j]), length, n, allowed)
                for g in GATES_X:
                    add(f"u4_{name}_L{length}_N{n}_{g}", "ext", targets, gates[g].astype(float),
                        subfamily=f"U4_roster_{name}", block="U4", gate=g, universe="TIME+KoAct", filter=name,
                        lookback=length, n=n, weight="equal", semis_cap=50, stock_cap=25)
    return out


def ledgers(inp, plan):
    ctx = inp.ctx
    curves = {}
    for key, d in plan.items():
        opn, cls = (inp.opn_book, inp.cls_book) if d["space"] == "book" else (inp.opn, inp.cls)
        frac = d["frac"]
        if np.isin(frac, (0.0, 1.0)).all():
            curves[key] = study.replay(ctx.days, opn, cls, ctx.fx, d["targets"], frac.astype(bool))
        else:
            curves[key] = common.replay_frac(ctx.days, opn, cls, ctx.fx, d["targets"], frac)
    return curves


def same_until(a, b, t):
    """Count decisions executed at sessions <= t and assert both runs made the same ones."""
    assert set(a) == set(b)
    checked = 0
    for key in a:
        ta, tb = a[key]["targets"], b[key]["targets"]
        ja, jb = sorted(j for j in ta if j <= t), sorted(j for j in tb if j <= t)
        assert ja == jb, f"{key}: decision dates differ before T"
        for j in ja:
            np.testing.assert_allclose(ta[j], tb[j], rtol=0, atol=1e-12, err_msg=f"{key} session {j}")
        np.testing.assert_array_equal(a[key]["frac"][:t + 1], b[key]["frac"][:t + 1], err_msg=key)
        checked += len(ja) + t + 1
    return checked


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def attribution(inp):
    """Base ledger split by holding: dollar P&L of semis-budget names vs the rest, per window (exact, additive)."""
    ctx = inp.ctx
    opn, cls, weekly = inp.opn_book, inp.cls_book, ctx.weekly
    width = opn.shape[1]
    shares, cash, prev = np.zeros(width), 1.0, np.zeros(width)
    pnl, nav_close = np.zeros(opn.shape), np.zeros(len(ctx.days))
    semis_weight, other_weight = np.zeros(len(ctx.days)), np.zeros(len(ctx.days))
    for j in range(len(ctx.days)):
        o, c = np.nan_to_num(opn[j]), np.nan_to_num(cls[j])
        pnl[j] = shares * (o - prev)                      # overnight, on yesterday's holdings
        nav = cash + shares @ o
        if j in weekly:
            w = weekly[j]
            shares = np.divide(nav * w, o, out=np.zeros_like(w), where=w > 0)
            cash = nav * (1 - w.sum())
        pnl[j] += shares * (c - o)                        # intraday, on today's holdings
        nav_close[j] = cash + shares @ c
        semis_weight[j] = (shares * c)[inp.semis_col].sum() / nav_close[j]
        other_weight[j] = (shares * c)[~inp.semis_col].sum() / nav_close[j]
        prev = c
    base = study.replay(ctx.days, ctx.opn, ctx.close, ctx.fx, weekly, ctx.gates["none"])
    np.testing.assert_allclose(nav_close, base.usd, rtol=1e-12)
    days = np.array(ctx.days)
    index = inp.book.cls
    out = {}
    spans = dict(common.WINDOWS, **{f"2026-{m:02d}": (f"2026-{m:02d}-01", f"2026-{m:02d}-31") for m in range(1, 12)})
    for name, (a, z) in spans.items():
        ix = np.flatnonzero((days >= a) & (days <= z))
        start = nav_close[ix[0] - 1] if ix[0] else 1.0
        before = days[ix[0] - 1] if ix[0] else inp.book.days[inp.start - 1]
        by_name = pnl[ix].sum(axis=0) / start
        total = nav_close[ix[-1]] / start - 1
        assert abs(by_name.sum() - total) < 1e-9
        worst = np.argsort(by_name)[:6]
        out[name] = dict(usd_return=float(total),
                         semis_contribution=float(by_name[inp.semis_col].sum()),
                         other_contribution=float(by_name[~inp.semis_col].sum()),
                         semis_weight_mean=float(semis_weight[ix].mean()),
                         other_weight_mean=float(other_weight[ix].mean()),
                         worst_names={inp.tickers[i]: round(float(by_name[i]), 4) for i in worst},
                         SOXX=float(index["SOXX"].loc[days[ix[-1]]] / index["SOXX"].loc[before] - 1),
                         QQQ=float(index["QQQ"].loc[days[ix[-1]]] / index["QQQ"].loc[before] - 1))
    return out


def same_gate_comparison(table, cfg):
    """Does changing WHAT is held beat the plain base under the same gate (cash exit, or no gate)?"""
    wide = table.join(cfg)
    rows = []
    for key, r in wide.iterrows():
        if key == "base" or r.subfamily.startswith(("U1c", "U2")):
            continue
        if r.gate in ("none", "always"):
            ref = "base"
        elif r.block == "U1":
            ref = f"u1c_f00_{r.gate}"
        else:
            ref = f"u2_s050_k25_{r.gate}"
        q = table.loc[ref]
        rows.append(dict(strategy=key, subfamily=r.subfamily, gate=r.gate, reference=ref,
                         h1=r.h1_usd > q.h1_usd, h2=r.h2_usd > q.h2_usd, h2_mdd=r.h2_usd_mdd > q.h2_usd_mdd,
                         jul_aug=r.jul_aug_usd > q.jul_aug_usd, sep_nov=r.sep_nov_usd > q.sep_nov_usd,
                         sep_nov_mdd=r.sep_nov_usd_mdd > q.sep_nov_usd_mdd,
                         d_h1=r.h1_usd - q.h1_usd, d_h2=r.h2_usd - q.h2_usd, d_h2_mdd=r.h2_usd_mdd - q.h2_usd_mdd,
                         d_jul_aug=r.jul_aug_usd - q.jul_aug_usd, d_sep_nov=r.sep_nov_usd - q.sep_nov_usd))
    frame = pd.DataFrame(rows)
    frame["gated"] = np.where(frame.gate.isin(["none", "always"]), "ungated", "gated")
    frame["h2_both"] = frame.h2 & frame.h2_mdd
    frame["all_three_windows"] = frame.h1 & frame.jul_aug & frame.sep_nov
    out = frame.groupby(["subfamily", "gated"]).agg(
        n=("h2", "size"), beat_h1=("h1", "sum"), beat_h2=("h2", "sum"), beat_h2_mdd=("h2_mdd", "sum"),
        beat_h2_both=("h2_both", "sum"), beat_jul_aug=("jul_aug", "sum"), beat_sep_nov=("sep_nov", "sum"),
        beat_h1_julaug_sepnov=("all_three_windows", "sum"),
        d_h1_med=("d_h1", "median"), d_h2_med=("d_h2", "median"), d_h2_mdd_med=("d_h2_mdd", "median"),
        d_jul_aug_med=("d_jul_aug", "median"), d_sep_nov_med=("d_sep_nov", "median")).reset_index()
    frame.to_csv(common.CACHE / f"{FAMILY}_same_gate.csv", index=False)
    return out


def spell_split(curves, key, gate, days, window="h2"):
    """Log excess of `key` over the base inside the window, split into the gate-off spells and the rest."""
    a, z = common.WINDOWS[window]
    cand, base = np.log(curves[key].usd), np.log(curves["base"].usd)
    excess = (cand.diff() - base.diff()).to_numpy()
    days = np.array(days)
    inside = (days >= a) & (days <= z)
    spells, start = [], None
    for j in range(len(days)):
        off = inside[j] and not gate[j]
        if off and start is None:
            start = j
        if start is not None and (not off or j == len(days) - 1):
            end = j if off else j - 1
            # the session after a spell carries the switch-back open, so it is counted with the spell
            last = min(end + 1, len(days) - 1) if inside[min(end + 1, len(days) - 1)] else end
            spells.append(dict(first=days[start], last=days[end], sessions=end - start + 1,
                               log_excess=float(np.nansum(excess[start:last + 1])),
                               base_usd=float(np.exp(base.iloc[last] - base.iloc[start - 1]) - 1),
                               cand_usd=float(np.exp(cand.iloc[last] - cand.iloc[start - 1]) - 1)))
            start = None
    total = float(np.nansum(excess[inside]))
    ranked = sorted((s["log_excess"] for s in spells), reverse=True)
    return dict(strategy=key, window=window, total_log_excess=total, spells=len(spells),
                in_spells=float(sum(ranked)), top1=ranked[0] if ranked else 0.0, top3=float(sum(ranked[:3])),
                positive_spells=int(sum(x > 0 for x in ranked)), detail=spells)


def report(curves, configs):
    table = common.stats_table(curves)
    cfg = pd.DataFrame.from_dict(configs, orient="index")
    base_curve = curves["base"]
    b = table.loc["base"]
    out = {"n_candidates": len(curves), "base": {c: float(b[c]) for c in table.columns}}
    control = {}
    for key in table.index:
        for window in ("h2", "sep_nov"):
            control[key, window] = common.matched_control(base_curve, table.at[key, f"{window}_expo"], window)
    table["h2_ctl_usd"] = [control[k, "h2"][0] for k in table.index]
    table["h2_ctl_usd_mdd"] = [control[k, "h2"][1] for k in table.index]
    table["sep_nov_ctl_usd"] = [control[k, "sep_nov"][0] for k in table.index]
    table["sep_nov_ctl_usd_mdd"] = [control[k, "sep_nov"][1] for k in table.index]
    for window in ("h2", "sep_nov"):
        a, z = common.WINDOWS[window]
        vals = []
        for key in table.index:
            c10 = curves[key].krw10
            ix = np.flatnonzero((c10.index >= a) & (c10.index <= z))
            vals.append(c10.iloc[ix[-1]] / c10.iloc[ix[0] - 1] - 1)
        table[f"{window}_krw_cost10"] = vals

    def row(key, train, test):
        r = table.loc[key]
        return dict(strategy=key, train_krw=float(r[f"{train}_krw"]), train_krw_mdd=float(r[f"{train}_krw_mdd"]),
                    train_usd=float(r[f"{train}_usd"]), train_usd_mdd=float(r[f"{train}_usd_mdd"]),
                    test_krw=float(r[f"{test}_krw"]), test_krw_mdd=float(r[f"{test}_krw_mdd"]),
                    test_usd=float(r[f"{test}_usd"]), test_usd_mdd=float(r[f"{test}_usd_mdd"]),
                    test_expo=float(r[f"{test}_expo"]), test_turn=float(r[f"{test}_turn"]),
                    test_ctl_usd=float(r[f"{test}_ctl_usd"]), test_ctl_usd_mdd=float(r[f"{test}_ctl_usd_mdd"]),
                    test_krw_cost10=float(r[f"{test}_krw_cost10"]),
                    base_test_krw=float(b[f"{test}_krw"]), base_test_usd=float(b[f"{test}_usd"]),
                    base_test_usd_mdd=float(b[f"{test}_usd_mdd"]))

    not_breadth = (cfg.loc[table.index, "gate"] != "breadth19").to_numpy()
    block = cfg.loc[table.index, "block"]
    pools = {"all": table.index, "ex_breadth19": table.index[not_breadth],
             "U1U2_ex_breadth19": table.index[not_breadth & block.isin(["base", "U1", "U2"]).to_numpy()]}
    for name in ("U1", "U2", "U3", "U4"):
        pools[f"{name}_only_ex_breadth19"] = table.index[not_breadth & block.isin(["base", name]).to_numpy()]
    out["selections"] = []
    for pool, keys in pools.items():
        for name, train, test in (("P1", "h1", "h2"), ("P2", "jan_aug", "sep_nov")):
            for rule, key in common.select(table.loc[keys], train).items():
                out["selections"].append(dict(pool=pool, protocol=f"{name} {train}->{test}", rule=rule,
                                              **row(key, train, test)))
    out["hindsight"] = {"best_h2_usd": row(table.h2_usd.idxmax(), "h1", "h2"),
                        "best_sep_nov_usd": row(table.sep_nov_usd.idxmax(), "jan_aug", "sep_nov")}
    for pool in ("ex_breadth19", "U1U2_ex_breadth19"):
        sub = table.loc[pools[pool]]
        out["hindsight"][f"best_h2_usd_{pool}"] = row(sub.h2_usd.idxmax(), "h1", "h2")
        out["hindsight"][f"best_sep_nov_usd_{pool}"] = row(sub.sep_nov_usd.idxmax(), "jan_aug", "sep_nov")

    cols = ["h1_usd", "h1_usd_mdd", "h2_usd", "h2_usd_mdd", "h2_expo", "h2_ctl_usd", "jan_aug_usd", "sep_nov_usd",
            "sep_nov_usd_mdd", "sep_nov_ctl_usd", "full_usd", "full_usd_mdd", "full_krw", "h1_krw", "h2_krw", "sep_nov_krw"]
    wide = table.join(cfg)
    u2 = wide[wide.block.isin(["U2", "base"]) & (wide.gate == "none")]
    u2 = u2.sort_values(["stock_cap", "semis_cap"])[["semis_cap", "stock_cap", *cols]]
    out["u2_ungated"] = u2.reset_index(names="strategy").to_dict(orient="records")
    u1 = wide[wide.block == "U1"]
    u1 = u1.assign(mechanism=u1.subfamily.where(~u1.subfamily.str.startswith("U1a"),
                                                  "U1a_L" + u1.lookback.astype(str) + "_" + u1.weight.astype(str)))
    pivot = u1.groupby(["gate", "mechanism"])[["h1_usd", "h2_usd", "h2_usd_mdd", "sep_nov_usd", "h2_expo"]].median()
    out["u1_gate_by_mechanism_median"] = pivot.reset_index().to_dict(orient="records")
    pivot.to_csv(common.CACHE / f"{FAMILY}_u1_pivot.csv")
    u2.to_csv(common.CACHE / f"{FAMILY}_u2_ungated.csv")

    cand = table.drop(index="base")
    beat = (cand.h2_usd > b.h2_usd) & (cand.h2_usd_mdd > b.h2_usd_mdd)
    beat_ctl = beat & (cand.h2_usd > cand.h2_ctl_usd)
    over = cand.h2_ctl_usd.isna()
    out["counts"] = dict(total=int(len(cand)), beat_base_both_h2_usd=int(beat.sum()),
                         also_beat_matched_control=int(beat_ctl.sum()),
                         beat_both_but_no_control_defined=int((beat & over).sum()),
                         control_undefined_total=int(over.sum()),
                         beat_control_and_base_mdd_too=int((beat_ctl & (cand.h2_usd_mdd > cand.h2_ctl_usd_mdd)).sum()))
    by_block = {}
    for block, keys in cfg.drop(index="base").groupby("block").groups.items():
        k = list(keys)
        by_block[block] = dict(total=len(k), beat_base_both=int(beat[k].sum()), also_beat_control=int(beat_ctl[k].sum()),
                               sep_nov_beat_base_usd=int((cand.loc[k, "sep_nov_usd"] > b.sep_nov_usd).sum()),
                               sep_nov_beat_control=int((cand.loc[k, "sep_nov_usd"] > cand.loc[k, "sep_nov_ctl_usd"]).sum()))
    out["counts_by_block"] = by_block

    joined = cand.join(cfg)
    joined["gated"] = np.where(joined.gate.isin(["none", "always"]), "ungated", "gated")
    joined["beat"], joined["beat_ctl"] = beat, beat_ctl
    joined["h2_excess_vs_ctl"] = joined.h2_usd - joined.h2_ctl_usd
    joined["sep_nov_excess_vs_ctl"] = joined.sep_nov_usd - joined.sep_nov_ctl_usd
    grouped = joined.groupby(["subfamily", "gated"]).agg(
        n=("h2_usd", "size"), h1_usd_med=("h1_usd", "median"), h2_usd_med=("h2_usd", "median"),
        h2_usd_best=("h2_usd", "max"), h2_usd_mdd_med=("h2_usd_mdd", "median"),
        sep_nov_usd_med=("sep_nov_usd", "median"), h2_expo_med=("h2_expo", "median"),
        h2_excess_vs_ctl_med=("h2_excess_vs_ctl", "median"),
        sep_nov_excess_vs_ctl_med=("sep_nov_excess_vs_ctl", "median"),
        beat_base_both=("beat", "sum"), beat_ctl=("beat_ctl", "sum")).reset_index()
    by_gate = joined.groupby(["block", "gate"]).agg(
        n=("h2_usd", "size"), h1_usd_med=("h1_usd", "median"), h2_usd_med=("h2_usd", "median"),
        sep_nov_usd_med=("sep_nov_usd", "median"), h2_expo_med=("h2_expo", "median"),
        h2_excess_vs_ctl_med=("h2_excess_vs_ctl", "median"), beat_ctl=("beat_ctl", "sum")).reset_index()
    full = table.join(cfg)
    full.index.name = "strategy"
    full.to_csv(common.CACHE / f"{FAMILY}_stats.csv")
    grouped.to_csv(common.CACHE / f"{FAMILY}_subfamily.csv", index=False)
    by_gate.to_csv(common.CACHE / f"{FAMILY}_by_gate.csv", index=False)
    same = same_gate_comparison(table, cfg)
    out["same_gate_reference"] = same.to_dict(orient="records")
    out["subfamily"] = grouped.to_dict(orient="records")
    out["by_gate"] = by_gate.to_dict(orient="records")
    return out, table, grouped, by_gate


def main():
    clock = time.time()
    ctx = common.context()
    book = ctx.book
    histories, held = signals.holdings_history()
    wanted = (set(AI_US) | held) - {""}
    ext, meta = load_downloaded(wanted)
    bars = load_bars(set(AI_ALL) | held | set(ctx.tickers) | set(watchlist.AI_UNIVERSE_ALIASES.values())
                     | {"QQQ", giant.ANCHOR})
    inp = build_inputs(ctx, book, ext, bars, histories)
    inp.bars_trend = {s: bars[s] for s in bars if s in (held | set(ctx.tickers) | {"QQQ"}
                                                         | set(watchlist.AI_UNIVERSE_ALIASES.values()))}
    info = {"download_meta": {k: meta[k] for k in ("source", "downloaded_at", "missing")}}

    # --- input checks -----------------------------------------------------
    assert schedule_days(inp) == sorted(ctx.weekly)
    for name, gate in ctx.gates.items():
        np.testing.assert_array_equal(inp.gates[name], gate)
    ref = pd.read_csv(common.CODEX / "breadth.csv", index_col=0).breadth
    mine = inp.breadth.copy()
    mine.index = mine.index.strftime("%Y-%m-%d")
    np.testing.assert_allclose(mine.reindex(ref.index), ref, rtol=1e-12)
    info["breadth_gate_off_sessions"] = {w: int((~inp.gates["breadth19"])[[i for i, d in enumerate(ctx.days) if a <= d <= z]].sum())
                                         for w, (a, z) in common.WINDOWS.items()}
    overlap = sorted(set(ext["Close"].columns) & set(ctx.tickers))
    recon = {}
    for field, frame in (("Open", book.opn), ("Close", book.cls), ("Adj Close", book.adj)):
        mineframe = ext[field].reindex(index=book.days, columns=overlap)
        rel = (mineframe / frame[overlap] - 1).abs()
        both = mineframe.notna() & frame[overlap].notna()
        only_one = int((mineframe.notna() ^ frame[overlap].notna()).sum().sum())
        worst = rel.where(both).max()
        recon[field] = dict(max_rel_diff=float(worst.max()), worst_symbol=str(worst.idxmax()),
                            cells=int(both.sum().sum()), cells_present_in_one_only=only_one,
                            symbols_over_1e_4=sorted(worst.index[worst > 1e-4]))
    info["reconcile_overlap"] = dict(symbols=len(overlap), **recon)
    first_bar = {s: bars[s].index[0] for s in AI_US}
    info["ai_us"] = dict(
        n=len(AI_US), in_time_book=len(set(AI_US) & set(ctx.tickers)),
        with_prices=int(sum(s in inp.xcol for s in AI_US)),
        semis_budget=int(sum(S.in_semis_budget(inp.labels[s]) for s in AI_US if s in inp.xcol)),
        first_bar_in_2026=sorted(s for s, d in first_bar.items() if d >= pd.Timestamp("2026-01-01")),
        first_bar_since_2025_06=sorted(s for s, d in first_bar.items() if d >= pd.Timestamp("2025-06-01")),
        no_120_session_momentum_on_first_trade=sorted(
            s for s in AI_US if s in inp.xcol and not np.isfinite(momentum(inp, 1, 120)[inp.xcol[s]])),
        ever_in_time_or_koact_roster_2026=len(set(AI_US) & set(inp.keymap.values())))

    # --- decisions and ledgers ---------------------------------------------
    plan = decisions(inp)
    info["seconds_decisions"] = round(time.time() - clock, 1)
    curves = ledgers(inp, plan)
    configs = {key: d["cfg"] for key, d in plan.items()}
    args = (ctx.days, ctx.opn, ctx.close, ctx.fx)
    base = study.replay(*args, ctx.weekly, ctx.gates["none"])
    np.testing.assert_allclose(curves["base"].to_numpy(), base.to_numpy(), rtol=1e-10, atol=1e-10)
    stats = common.window_stats(curves["base"], *common.WINDOWS["h1"]), common.window_stats(curves["base"], *common.WINDOWS["h2"])
    assert abs(stats[0]["krw"] - 2.4505) < 5e-5 and abs(stats[1]["krw"] + 0.3945) < 5e-5, stats
    # the extended (book + downloaded) price matrix reproduces the base when fed the base schedule
    padded = {j: np.r_[w, np.zeros(len(inp.symbols) - len(w))] for j, w in ctx.weekly.items()}
    wide = study.replay(ctx.days, inp.opn, inp.cls, ctx.fx, padded, ctx.gates["none"])
    np.testing.assert_allclose(wide.to_numpy(), base.to_numpy(), rtol=1e-12, atol=1e-12)
    # scale 0 is Codex's cash exit; semis cap 50 / stock cap 25 gated equals the same thing
    for g in ("SOXX20r3", "SOXX100r1"):
        cash_exit = study.replay(*args, ctx.weekly, ctx.gates[g])
        np.testing.assert_allclose(curves[f"u1c_f00_{g}"].to_numpy(), cash_exit.to_numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(curves[f"u2_s050_k25_{g}"].to_numpy(), cash_exit.to_numpy(), rtol=1e-12, atol=1e-12)
    # Giant and trend grades agree with Codex's frozen tables wherever both exist
    codex_giant = pd.read_csv(common.CODEX / "giant_grades.csv", index_col=0)
    pairs = [(v, codex_giant.at[d, s]) for (d, s), v in inp.giant_grades.items() if s in codex_giant.columns]
    assert all((np.isnan(a) and np.isnan(b)) or a == b for a, b in pairs)
    codex_trend = pd.read_csv(common.CODEX / "trend_grades.csv", index_col=0)
    mine_trend = inp.trend_grades[codex_trend.columns].to_numpy(float)
    assert np.array_equal(np.nan_to_num(mine_trend, nan=-9), np.nan_to_num(codex_trend.to_numpy(float), nan=-9))
    info["grade_checks"] = dict(giant_cells_equal_to_codex=len(pairs), trend_cells_equal_to_codex=int(mine_trend.size))
    for key, frame in curves.items():
        assert frame.exposure.between(-1e-9, 1 + 1e-9).all(), key
    info["base_checks"] = dict(base_h1_krw=stats[0]["krw"], base_h2_krw=stats[1]["krw"], base_h2_usd=stats[1]["usd"],
                               extended_matrix_reproduces_base=True, cap50_stock25_targets_identical=True)

    # --- look-ahead self-test ------------------------------------------------
    tested = {}
    for label, rng in (("x1.5", None), ("random_0.5_2.0", np.random.default_rng(20261119))):
        shadow = perturbed(ctx, book, ext, bars, histories, T_TEST, rng)
        shadow.bars_trend = {s: shadow.bars[s] for s in inp.bars_trend}
        changed = float(np.nanmax(np.abs(shadow.cls[T_TEST + 1:] / inp.cls[T_TEST + 1:] - 1)))
        tested[label] = dict(values_checked=same_until(plan, decisions(shadow), T_TEST), max_rel_price_change_after_T=changed)
        later = sum(1 for key in plan for j in plan[key]["targets"] if j > T_TEST)
        tested[label]["candidates"] = len(plan)
        tested[label]["decisions_after_T_not_compared"] = later
    info["lookahead"] = dict(T=T_TEST, T_date=ctx.days[T_TEST], result="identical", runs=tested)

    table = common.save(FAMILY, curves, configs)
    result, stats_table, grouped, by_gate = report(curves, configs)
    info["base_attribution"] = attribution(inp)
    picked = sorted({x["strategy"] for x in result["selections"] if x["strategy"].startswith(("u1a", "u1b"))})
    result["selected_switch_spells"] = [
        spell_split(curves, key, inp.gates[configs[key]["gate"]], ctx.days, window)
        for key in picked for window in ("h1", "h2", "sep_nov")]
    result["info"] = info
    result["grid"] = pd.DataFrame.from_dict(configs, orient="index").groupby(["block", "subfamily"]).size().rename("n").reset_index().to_dict(orient="records")
    result["runtime_seconds"] = round(time.time() - clock, 1)
    (common.CACHE / f"{FAMILY}_summary.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float),
                                                         encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    print(f"candidates {len(curves)} (incl. base) saved={len(table)} runtime {result['runtime_seconds']}s")
    print(json.dumps(info, indent=1, ensure_ascii=False, default=float))
    print(pd.DataFrame(result["u2_ungated"]).round(4).to_string(index=False))
    print(pd.DataFrame(result["u1_gate_by_mechanism_median"]).round(4).to_string(index=False))
    for s in result["selections"]:
        print(f"{s['pool']:18s} {s['protocol']:20s} {s['rule']:9s} {s['strategy']:34s} train krw {s['train_krw']:+.4f} "
              f"mdd {s['train_krw_mdd']:+.4f} | test krw {s['test_krw']:+.4f} usd {s['test_usd']:+.4f} "
              f"usd_mdd {s['test_usd_mdd']:+.4f} expo {s['test_expo']:.3f} turn {s['test_turn']:.2f} "
              f"ctl_usd {s['test_ctl_usd']:+.4f} cost10_krw {s['test_krw_cost10']:+.4f}")
    for name, s in result["hindsight"].items():
        print(f"HINDSIGHT {name:36s} {s['strategy']:34s} test krw {s['test_krw']:+.4f} usd {s['test_usd']:+.4f} "
              f"usd_mdd {s['test_usd_mdd']:+.4f} expo {s['test_expo']:.3f} train krw {s['train_krw']:+.4f}")
    print("counts", result["counts"])
    print(pd.DataFrame(result["same_gate_reference"]).round(4).to_string(index=False))
    for x in result["selected_switch_spells"]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items() if k != "detail"})
        if x["window"] == "h2":
            print(pd.DataFrame(x["detail"]).round(4).to_string(index=False))
    print("counts_by_block", result["counts_by_block"])
    print(grouped.round(4).to_string(index=False))
    print(by_gate.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
