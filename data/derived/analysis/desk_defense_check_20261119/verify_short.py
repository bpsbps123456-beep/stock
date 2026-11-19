"""verify_short: independent check of the hedge family ("boldly take short positions").

Run from the repo root:  py data/derived/analysis/desk_defense_check_20261119/verify_short.py

Reads (never writes) common.py, ext_hedge.py, cache/hedge_daily.parquet, cache/hedge_stats.csv, frozen_etf/*.
Writes only cache/verify_short.json.

Own code here: a sleeve-accounting simulator (base sleeve / ETF sleeve / USD cash) that does not use the
builder's share-level engine, own base replay, own signal loops for four rules (H1, H3, two H4) on the
separately frozen frozen_etf/SOXX.csv + SOXS.csv, own window statistics, own placebo sampler.
The builder's rule states (a = stock fraction, e = ETF weight per session) are reused only for the bulk
checks, after the sleeve simulator reproduced every saved ledger from them.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import ext_hedge as H  # noqa: E402

SHORT_SUBS = ("H1_gate_inverse", "H2_overlay", "H3_symmetric", "H4_crash_short", "H5_exhaustion_short",
              "H7_frozen", "H7_neighbor")
WIN = {"full": ("2026-01-01", "2026-12-31"), "h1": ("2026-01-01", "2026-06-30"), "h2": ("2026-07-01", "2026-12-31"),
       "jan_aug": ("2026-01-01", "2026-08-31"), "sep_nov": ("2026-09-01", "2026-12-31"),
       "julA": ("2026-06-23", "2026-07-29"), "rally": ("2026-07-30", "2026-09-22"),
       "octB": ("2026-10-13", "2026-11-17"), "oos": ("2026-10-12", "2026-11-17")}
N_DRAW = 1000
SEED = 771119
R = {}                                                   # results written to cache/verify_short.json


def say(title):
    print(f"\n=== {title} ===", flush=True)


def show(frame, digits=3):
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200,
                           "display.float_format", lambda v: f"{v:.{digits}f}"):
        print(frame.to_string(), flush=True)


# ---------------------------------------------------------------- own primitives
def my_base(ctx):
    """Base 7-stock plan replayed with my own loop: NAV at each open and close."""
    n_days = len(ctx.days)
    shares, cash = np.zeros(ctx.opn.shape[1]), 1.0
    bo, bc = np.empty(n_days), np.empty(n_days)
    for j in range(n_days):
        nav = cash + shares @ np.nan_to_num(ctx.opn[j])
        if j in ctx.weekly:
            w = ctx.weekly[j]
            px = np.where(w > 0, ctx.opn[j], 1.0)
            if not np.isfinite(px).all():
                raise ValueError("missing open")
            shares, cash = np.where(w > 0, nav * w / px, 0.0), nav * (1 - w.sum())
        bo[j], bc[j] = nav, cash + shares @ np.nan_to_num(ctx.close[j])
    return bo, bc


def sleeve(P, a, x, po, pc, mode="hold", detail=False):
    """Three sleeves: B follows the base NAV, E one ETF, C USD cash. a, x: (draws, sessions) or (sessions,).

    Sleeves are reset to (a, x, 1-a-x) of NAV at an open when the state changes, on the base's weekly trade
    days while a > 0, every session of a special state in "daily" mode; in "split" mode only when the special
    state starts or ends. Everything else drifts.
    """
    a, x = np.atleast_2d(np.asarray(a, float)), np.atleast_2d(np.asarray(x, float))
    n_draw, n_days = a.shape
    B, E, C = np.ones(n_draw), np.zeros(n_draw), np.zeros(n_draw)
    pa, px, pon = np.ones(n_draw), np.zeros(n_draw), np.zeros(n_draw, bool)
    usd = np.empty((n_draw, n_days))
    if detail:
        navo, ew, ov, intra, bw = (np.zeros((n_draw, n_days)) for _ in range(5))
    for j in range(n_days):
        if j:
            B = B * (P.bo[j] / P.bc[j - 1])
            e_open = E * (po[j] / pc[j - 1])
            if detail:
                ov[:, j] = e_open - E
            E = e_open
        nav = B + E + C
        aj, xj = a[:, j], x[:, j]
        special = (aj < 1 - 1e-12) | (xj > 0)
        if mode == "split":
            reb = special != pon
        else:
            reb = (np.abs(aj - pa) > 1e-12) | (np.abs(xj - px) > 1e-12) | (P.wk[j] & (aj > 0))
            if mode == "daily":
                reb = reb | special
        nb, ne = np.where(reb, aj * nav, B), np.where(reb, xj * nav, E)
        C = np.where(reb, nav - nb - ne, C)
        B, E = nb, ne
        if detail:
            navo[:, j], ew[:, j], bw[:, j] = nav, E / nav, B / nav
        e_close = E * (pc[j] / po[j])
        if detail:
            intra[:, j] = e_close - E
        B, E = B * (P.bc[j] / P.bo[j]), e_close
        usd[:, j] = B + E + C
        pa, px, pon = aj, xj, special
    if detail:
        return SimpleNamespace(usd=usd, navo=navo, ew=ew, bw=bw, ov=ov, intra=intra, cash_min=None)
    return usd


def wret(U, idx, lo, hi):
    """U: (sessions, N). Return and max drawdown inside [lo, hi], anchored on the close before."""
    ix = np.flatnonzero((idx >= lo) & (idx <= hi))
    before = U[ix[0] - 1] if ix[0] else np.ones(U.shape[1])
    path = np.vstack([before, U[ix]])
    return path[-1] / before - 1, (path / np.maximum.accumulate(path, axis=0) - 1).min(0)


def runs_of(mask):
    out, begin = [], None
    for j, v in enumerate(mask):
        if v and begin is None:
            begin = j
        if begin is not None and not v:
            out.append((begin, j - 1))
            begin = None
    if begin is not None:
        out.append((begin, len(mask) - 1))
    return out


def my_control(base_usd, base_expo, idx, lo, hi, expo):
    """Fixed fraction expo/normal of the base every day of the window, rest USD cash (hindsight size)."""
    ix = np.flatnonzero((idx >= lo) & (idx <= hi))
    normal = base_expo[ix].mean()
    path = np.r_[base_usd[ix[0] - 1] if ix[0] else 1.0, base_usd[ix]]
    nav = np.r_[1.0, np.cumprod(1 + expo / normal * (path[1:] / path[:-1] - 1))]
    return nav[-1] - 1, np.min(nav / np.maximum.accumulate(nav) - 1)


# ---------------------------------------------------------------- setup
def setup():
    ctx = common.context()
    days = list(ctx.days)
    if not (H.FROZEN / "hedge_ohlc.csv").exists():
        raise FileNotFoundError("frozen ETF bars missing; this check never downloads")
    data, _ = H.load_etf()
    etf_o = np.column_stack([H.field(data, s, "Open").reindex(days).to_numpy() for s in H.ETFS])
    etf_c = np.column_stack([H.field(data, s, "Close").reindex(days).to_numpy() for s in H.ETFS])
    opn_x, close_x = np.hstack([ctx.opn, etf_o]), np.hstack([ctx.close, etf_c])
    breadth = pd.read_csv(common.CODEX / "breadth.csv", index_col=0).breadth.shift().reindex(days)
    cls = ctx.book.cls[["SOXX", "QQQ"]].astype(float)
    env = H.make_env(cls, opn_x, close_x, ctx.fx, ctx.weekly, days, ctx.start, breadth.to_numpy(), len(ctx.tickers))
    specs = H.build_specs(env)
    bo, bc = my_base(ctx)
    P = SimpleNamespace(bo=bo, bc=bc, wk=np.array([j in ctx.weekly for j in range(len(days))]),
                        eo=etf_o, ec=etf_c, days=days, idx=pd.Index(days), n=len(days))
    daily = pd.read_parquet(common.CACHE / "hedge_daily.parquet")
    return ctx, data, env, specs, P, daily


def k_of(s):
    if not s.e.any():
        return 0
    k = int(np.argmax(s.e.max(0)))
    other = np.delete(s.e, k, axis=1)
    if other.any():
        raise ValueError("rule holds two ETFs")
    return k


def sim(P, s, detail=False, a=None, x=None):
    k = k_of(s)
    a = s.a if a is None else a
    x = s.e[:, k] if x is None else x
    return sleeve(P, a, x, P.eo[:, k], P.ec[:, k], s.mode, detail)


# ---------------------------------------------------------------- 1. integrity
def integrity(ctx, data, env, specs, P, L):
    say("1. INTEGRITY")
    days, idx = P.days, P.idx
    out = {}
    # 1a base
    out["my_base_vs_desk_model_max_rel"] = float(np.abs(P.bc / np.asarray(ctx.baseline.close) - 1).max())
    out["my_base_vs_saved_base_ledger_max_rel"] = float(np.abs(P.bc / L["base"].to_numpy() - 1).max())
    U = L.to_numpy()
    base_w = {k: tuple(round(float(v), 4) for v in wret(P.bc[:, None], idx, *WIN[k])[0]) + tuple(
        round(float(v), 4) for v in wret(P.bc[:, None], idx, *WIN[k])[1]) for k in WIN}
    out["base_usd_ret_mdd_by_window"] = base_w
    print("base (ret, mdd):", base_w)
    # 1b weights and builder engine vs saved ledgers; my sleeve simulator vs saved ledgers
    wmin, wmax, tmax, tmin, gap_engine, gap_sleeve, worst_key = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ""
    cash_min = 1.0
    entry_bad = 0
    held_at_end = 0
    for key, (_, s) in specs.items():
        targets = H.targets_of(env, s)
        for w in targets.values():
            tmin, tmax = min(tmin, float(w.min())), max(tmax, float(w.sum()))
        run = H.engine(env, targets)
        wmin, wmax = min(wmin, float(run.w.min())), max(wmax, float(run.w.sum(1).max()))
        saved = L[key].to_numpy()
        gap_engine = max(gap_engine, float(np.abs(run.usd / saved - 1).max()))
        mine = sim(P, s, detail=True)
        g = float(np.abs(mine.usd[0] / saved - 1).max())
        if g > gap_sleeve:
            gap_sleeve, worst_key = g, key
        cash_min = min(cash_min, float((1 - mine.ew[0] - mine.bw[0]).min()))
        k = k_of(s)
        on = mine.ew[0] > 1e-12
        for first, last in runs_of(on):
            if not (np.isfinite(P.eo[first, k]) and P.eo[first, k] > 0):
                entry_bad += 1
            if last + 1 < P.n:
                if not (np.isfinite(P.eo[last + 1, k]) and P.eo[last + 1, k] > 0):
                    entry_bad += 1
            else:
                held_at_end += 1
    out.update(rules=len(specs), target_weight_min=tmin, target_weight_sum_max=tmax, open_weight_min=wmin,
               open_weight_sum_max=wmax, sleeve_cash_fraction_min=cash_min,
               builder_engine_vs_saved_max_rel=gap_engine, my_sleeve_vs_saved_max_rel=gap_sleeve,
               my_sleeve_worst_key=worst_key, etf_entries_or_exits_without_open=entry_bad,
               etf_positions_still_open_on_last_session=held_at_end,
               etf_bars_finite_all_sessions=bool(np.isfinite(P.eo).all() and np.isfinite(P.ec).all() and (P.eo > 0).all()))
    print({k: v for k, v in out.items() if k != "base_usd_ret_mdd_by_window"})

    # 1c prices: book vs frozen, frozen vs the separately frozen single files, leverage behaviour
    price = {}
    for sym in ("SOXX", "QQQ"):
        for name, frame in (("Open", ctx.book.opn), ("Close", ctx.book.cls)):
            mine = H.field(data, sym, name).reindex(days)
            ref = frame[sym].reindex(days).astype(float)
            price[f"{sym}_{name}_vs_book_max_rel_2026"] = float((mine / ref - 1).abs().max())
    alt = {}
    for sym in ("SOXS", "SOXX", "SOXL", "QQQ"):
        f = pd.read_csv(H.FROZEN / f"{sym}.csv", dtype={"date": str}).set_index("date")
        alt[sym] = f
        for name in ("Open", "Close"):
            price[f"{sym}_{name}_ohlc_vs_single_file_max_rel_2026"] = float(
                (H.field(data, sym, name).reindex(days) / f[name].reindex(days) - 1).abs().max())
    f = alt["SOXS"].reindex(days)
    price["SOXS_open_outside_day_low_high_sessions"] = int(((f.Open < f.Low * (1 - 1e-9)) | (f.Open > f.High * (1 + 1e-9))).sum())
    lev = []
    for sym, (mult, und) in H.LEV.items():
        c, o = H.field(data, sym, "Close"), H.field(data, sym, "Open")
        uc, uo = H.field(data, und, "Close"), H.field(data, und, "Open")
        parts = {"close_close": (c / c.shift() - 1, uc / uc.shift() - 1),
                 "overnight": (o / c.shift() - 1, uo / uc.shift() - 1),
                 "intraday": (c / o - 1, uc / uo - 1)}
        row = {"etf": sym, "underlying": und, "nominal": mult}
        for name, (r, u) in parts.items():
            r, u = r.reindex(days).to_numpy(), u.reindex(days).to_numpy()
            slope = float(np.cov(r, u)[0, 1] / np.var(u, ddof=1))
            row[f"slope_{name}"] = slope
            row[f"corr_{name}"] = float(np.corrcoef(r, u)[0, 1])
            if name == "close_close":
                res = r - mult * u
                row["max_abs_resid_vs_nominal"] = float(np.abs(res).max())
                row["resid_day"] = days[int(np.abs(res).argmax())]
        for d in [d for d, v in H.field(data, sym, "Stock Splits").items() if v and d >= "2026-01-01"]:
            j = days.index(d)
            row[f"resid_on_split_{d}"] = float((c.reindex(days).to_numpy()[j] / c.reindex(days).to_numpy()[j - 1] - 1)
                                               - mult * (uc.reindex(days).to_numpy()[j] / uc.reindex(days).to_numpy()[j - 1] - 1))
        div = H.field(data, sym, "Dividends")
        y = div.index >= "2026-01-01"
        row["distributions_2026_pct_of_price"] = float((div[y] / c.reindex(div.index)[y]).sum())
        lev.append(row)
    levf = pd.DataFrame(lev).set_index("etf")
    show(levf, 4)
    price["leverage"] = lev
    sx = H.field(data, "SOXX", "Close").reindex(days)
    r = (sx / sx.shift() - 1)
    ss = H.field(data, "SOXS", "Close").reindex(days)
    rs = ss / ss.shift() - 1
    big = pd.DataFrame({"SOXX_cc": r, "SOXS_cc": rs,
                        "SOXS_open_to_close": ss / H.field(data, "SOXS", "Open").reindex(days) - 1,
                        "base_cc": pd.Series(P.bc, index=days).pct_change()})
    big = big[(big.SOXX_cc.abs() >= .06)]
    print("SOXX sessions with |close-to-close| >= 6%:")
    show(big, 4)
    price["soxx_big_days"] = big.reset_index().rename(columns={"index": "date"}).to_dict("records")
    wtab = {}
    for name, (lo, hi) in WIN.items():
        ix = np.flatnonzero((idx >= lo) & (idx <= hi))
        row = {"base": float(P.bc[ix[-1]] / (P.bc[ix[0] - 1] if ix[0] else 1.) - 1)}
        for sym in ("SOXX", "QQQ", "SOXS", "SQQQ", "PSQ"):
            c = H.field(data, sym, "Close")
            pos = c.index.get_loc(days[ix[0]])
            row[sym] = float(c.reindex(days).to_numpy()[ix[-1]] / c.to_numpy()[pos - 1] - 1)
        wtab[name] = row
    wt = pd.DataFrame(wtab).T
    print("close-to-close returns by window (anchored on the close before):")
    show(wt, 3)
    price["window_returns"] = wtab
    print({k: v for k, v in price.items() if k not in ("leverage", "soxx_big_days", "window_returns")})
    out["prices"] = price

    # 1d four rules with my own signal loops on the separately frozen single files
    sx_f, so_f = alt["SOXX"], alt["SOXS"]
    cal = list(sx_f.index)
    c = sx_f.Close.to_numpy(float)
    pos = {d: i for i, d in enumerate(cal)}
    po, pc = so_f.Open.reindex(days).to_numpy(float), so_f.Close.reindex(days).to_numpy(float)
    prev = [pos[d] - 1 for d in days]                    # calendar index of the close before each session

    def cmp(key, a, x):
        mine = sleeve(P, a, x, po, pc, "hold")[0]
        saved = L[key].to_numpy()
        s = specs[key][1]
        return {"rule": key, "max_rel_diff_usd": float(np.abs(mine / saved - 1).max()),
                "state_days_differ": int((np.abs(a - s.a) > 1e-12).sum() + (np.abs(x - s.e[:, k_of(s)]) > 1e-12).sum()),
                "my_full_usd": float(mine[-1] - 1), "saved_full_usd": float(saved[-1] - 1),
                "short_sessions": int((x > 0).sum())}

    re = []
    # H1: SOXX close < SMA50 -> stocks 0 + SOXS 20% bought once and left alone
    a, x = np.ones(P.n), np.zeros(P.n)
    for j, p in enumerate(prev):
        if c[p] < c[p - 49:p + 1].mean():
            a[j], x[j] = 0.0, 0.20
    re.append(cmp("H1_SOXX50r1_SOXS_20", a, x))
    # H3: n=20, T=5, 30%
    n, hold = 20, 5
    below_c, short_c = np.zeros(len(c), bool), np.zeros(len(c), bool)
    last_on = -1                                         # last close index after which the short is still held
    for i in range(n, len(c)):
        line, line_prev = c[i - n + 1:i + 1].mean(), c[i - n:i].mean()
        below_c[i] = c[i] < line
        if not below_c[i]:
            last_on = -1
        elif not (c[i - 1] < line_prev):
            last_on = i + hold - 1
        elif i > last_on and c[i] < c[i - n:i].min():
            last_on = i + hold - 1
        short_c[i] = below_c[i] and i <= last_on
    a = np.array([0.0 if below_c[p] else 1.0 for p in prev])
    x = np.array([0.30 if short_c[p] else 0.0 for p in prev])
    re.append(cmp("H3_n20_T5_x30", a, x))
    # H4 (SOXX): close more than 7.5% under its 20-session high close, fresh crossing, 5 sessions, flat 50%
    cond = np.zeros(len(c), bool)
    for i in range(19, len(c)):
        cond[i] = c[i] / c[i - 19:i + 1].max() - 1 < -0.075
    on_c, last_on = np.zeros(len(c), bool), -1
    for i in range(1, len(c)):
        if cond[i] and not cond[i - 1] and i > last_on:
            last_on = i + 5 - 1
        on_c[i] = i <= last_on
    on = np.array([on_c[p] for p in prev])
    re.append(cmp("H4_sdd7.5_h5_flat_50", np.where(on, 0.0, 1.0), np.where(on, 0.5, 0.0)))
    # H4 (base NAV): close 8% or more under its 2026 running peak, fresh crossing, 3 sessions, flat 30%
    peak, dd = 1.0, np.zeros(P.n)
    for j in range(P.n):
        peak = max(peak, P.bc[j])
        dd[j] = P.bc[j] / peak - 1
    on, last_on = np.zeros(P.n, bool), -1                # state after the close of session j
    for j in range(P.n):
        was = dd[j - 1] <= -0.08 if j else False
        if dd[j] <= -0.08 and not was and j > last_on:
            last_on = j + 3 - 1
        on[j] = j <= last_on
    on = np.r_[False, on[:-1]]
    re.append(cmp("H4_nav8_h3_flat_30", np.where(on, 0.0, 1.0), np.where(on, 0.3, 0.0)))
    show(pd.DataFrame(re).set_index("rule"), 10)
    out["reimplemented"] = re
    R["integrity"] = out


# ---------------------------------------------------------------- 2. pairs
def pairs(P, L, cfg):
    say("2. SHORT vs CASH TWIN (pure effect of the ETF leg; same stock leg)")
    idx = P.idx
    short = cfg.index[cfg.subfamily.isin(SHORT_SUBS)]
    tw = cfg.twin.reindex(short)
    assert tw.notna().all() and (tw != "").all()
    T = pd.DataFrame(index=short)
    T["subfamily"], T["instrument"], T["etf_pct"], T["twin"] = cfg.subfamily[short], cfg.instrument[short], cfg.etf_pct[short], tw
    US, UT = L[short].to_numpy(), L[tw.to_numpy()].to_numpy()
    for name, (lo, hi) in WIN.items():
        (rs, ms), (rt, mt) = wret(US, idx, lo, hi), wret(UT, idx, lo, hi)
        T[f"{name}_s"], T[f"{name}_t"], T[f"{name}_d"] = rs, rt, rs - rt
        T[f"{name}_mdd_s"], T[f"{name}_mdd_t"], T[f"{name}_mdd_d"] = ms, mt, ms - mt

    def block(g):
        d = {"n": len(g)}
        for name, col in (("full", "full_d"), ("h2", "h2_d"), ("mddF", "full_mdd_d"), ("mddH2", "h2_mdd_d"),
                          ("julA", "julA_d"), ("octB", "octB_d")):
            d[f"{name}_win%"] = 100 * float((g[col] > 1e-9).mean())
        d["julA_tie%"] = 100 * float((g.julA_d.abs() <= 1e-9).mean())
        d["octB_tie%"] = 100 * float((g.octB_d.abs() <= 1e-9).mean())
        for name in ("full", "h2", "julA", "octB", "h1"):
            d[f"{name}_med_d"] = 100 * float(g[f"{name}_d"].median())
        d["mddF_med_d"] = 100 * float(g.full_mdd_d.median())
        d["full_med_s"], d["full_med_t"] = 100 * float(g.full_s.median()), 100 * float(g.full_t.median())
        d["h2_med_s"], d["h2_med_t"] = 100 * float(g.h2_s.median()), 100 * float(g.h2_t.median())
        d["all3_win%"] = 100 * float(((g.full_d > 1e-9) & (g.h2_d > 1e-9) & (g.full_mdd_d > 1e-9)).mean())
        return d

    tabs = {}
    for label, by in (("subfamily", T.subfamily), ("instrument", T.instrument), ("etf_pct", T.etf_pct)):
        rows = {str(k): block(g) for k, g in T.groupby(by)}
        if label == "subfamily":
            rows["ALL"] = block(T)
        tab = pd.DataFrame(rows).T
        print(f"-- by {label} (win% = share of pairs where the short beats its cash twin; med_d = median short-twin, points)")
        show(tab, 1)
        tabs[label] = tab.round(3).to_dict("index")
    # instrument x weight for H1 only (the only subfamily with all three instruments and 10..100)
    h1 = T[T.subfamily == "H1_gate_inverse"]
    grid = h1.groupby(["instrument", "etf_pct"]).apply(lambda g: pd.Series({
        "n": len(g), "full_win%": 100 * (g.full_d > 1e-9).mean(), "h2_win%": 100 * (g.h2_d > 1e-9).mean(),
        "mddF_win%": 100 * (g.full_mdd_d > 1e-9).mean(), "full_med_d": 100 * g.full_d.median(),
        "h2_med_d": 100 * g.h2_d.median(), "julA_med_d": 100 * g.julA_d.median(), "octB_med_d": 100 * g.octB_d.median()}))
    print("-- H1 gate->inverse by instrument x weight")
    show(grid, 1)
    tabs["H1_instrument_x_weight"] = {f"{a}_{b}": v for (a, b), v in grid.round(3).to_dict("index").items()}
    allwin = T[(T.full_d > 1e-9) & (T.h2_d > 1e-9) & (T.full_mdd_d > 1e-9) & (T.h2_mdd_d > 1e-9)]
    tabs["pairs_short_better_on_full_h2_and_both_mdd"] = {"n": int(len(allwin)), "of": int(len(T)),
                                                          "by_subfamily": allwin.subfamily.value_counts().to_dict(),
                                                          "by_signal": cfg.signal[allwin.index].value_counts().head(12).to_dict()}
    print("pairs where the short wins full, H2, full MDD and H2 MDD:", tabs["pairs_short_better_on_full_h2_and_both_mdd"])
    R["pairs"] = tabs
    return T


# ---------------------------------------------------------------- 3. forward selection
def selection(P, daily, cfg, T):
    say("3. FORWARD SELECTION inside the family")
    curves = {k: g.set_index("date")[common.LEDGER] for k, g in daily.groupby("strategy", sort=False)}
    st = common.stats_table(curves)
    ref = pd.read_csv(common.CACHE / "hedge_stats.csv", index_col=0)
    R["stats_table_vs_hedge_stats_csv_max_abs"] = float((st[["full_usd", "h2_usd", "h2_usd_mdd", "sep_nov_usd", "jan_aug_krw"]]
                                                         - ref.loc[st.index, ["full_usd", "h2_usd", "h2_usd_mdd", "sep_nov_usd", "jan_aug_krw"]]).abs().max().max())
    short = list(T.index)
    pools = {"family_1297": list(st.index), "short_rules_968+base": short + ["base"]}
    rows = []
    for pool_name, keys in pools.items():
        table = st.loc[keys]
        for train, test in (("jan_aug", "sep_nov"), ("h1", "h2")):
            for cur in ("krw", "usd"):
                for rule, key in common.select(table, train, cur).items():
                    r = st.loc[key]
                    twin = cfg.twin.get(key, "")
                    twin = twin if isinstance(twin, str) and twin and twin != "base" else ""
                    cand = table.drop(index="base")
                    rows.append({"pool": pool_name, "train": train, "cur": cur, "rule": rule, "chosen": key,
                                 "train_usd": r[f"{train}_usd"], "test_usd": r[f"{test}_usd"], "test_mdd": r[f"{test}_usd_mdd"],
                                 "base_test": st.at["base", f"{test}_usd"], "base_test_mdd": st.at["base", f"{test}_usd_mdd"],
                                 "twin_test": st.at[twin, f"{test}_usd"] if twin else np.nan,
                                 "ctrl_test": my_control(P.bc, curves["base"].exposure.to_numpy(), P.idx, *common.WINDOWS[test], r[f"{test}_expo"])[0],
                                 "rank_in_test": int((cand[f"{test}_usd"] > r[f"{test}_usd"]).sum() + 1), "pool_n": len(cand),
                                 "pool_best_test": cand[f"{test}_usd"].max(), "pool_median_test": cand[f"{test}_usd"].median()})
    sel = pd.DataFrame(rows)
    show(sel.drop(columns=["pool_n"]), 3)
    R["selection"] = sel.round(5).to_dict("records")
    # is the hindsight best reachable; does the train rank carry over
    sh = st.loc[short]
    extra = {}
    for train, test in (("jan_aug", "sep_nov"), ("h1", "h2")):
        best = sh[f"{test}_usd"].idxmax()
        top = sh[f"{train}_usd"].nlargest(len(sh) // 10).index
        bot = sh[f"{train}_usd"].nsmallest(len(sh) // 10).index
        extra[f"{train}->{test}"] = {
            "hindsight_best_test": best, "its_test_usd": float(sh.at[best, f"{test}_usd"]),
            "its_train_usd": float(sh.at[best, f"{train}_usd"]),
            "its_train_rank_of_968": int((sh[f"{train}_usd"] > sh.at[best, f"{train}_usd"]).sum() + 1),
            "its_train_mdd_vs_base": [float(sh.at[best, f"{train}_usd_mdd"]), float(st.at["base", f"{train}_usd_mdd"])],
            "its_full_usd": float(sh.at[best, "full_usd"]), "its_full_mdd": float(sh.at[best, "full_usd_mdd"]),
            "rank_corr_train_test_usd": float(sh[f"{train}_usd"].rank().corr(sh[f"{test}_usd"].rank())),
            "top_decile_by_train_median_test": float(sh.loc[top, f"{test}_usd"].median()),
            "bottom_decile_by_train_median_test": float(sh.loc[bot, f"{test}_usd"].median()),
            "all_short_median_test": float(sh[f"{test}_usd"].median()), "base_test": float(st.at["base", f"{test}_usd"]),
            "short_rules_beating_base_train_usd": int((sh[f"{train}_usd"] > st.at["base", f"{train}_usd"]).sum())}
    print(json.dumps(extra, indent=1, default=float))
    R["selection_extra"] = extra
    return st, curves


# ---------------------------------------------------------------- 4. tail risk
def tail(P, specs, L, cfg, T, data):
    say("4. TAIL RISK")
    days, idx = P.days, P.idx
    month = idx.str[:7]
    sx = H.field(data, "SOXX", "Close").reindex(days).to_numpy()

    def low_day(lo, hi):
        ix = np.flatnonzero((idx >= lo) & (idx <= hi))
        return int(ix[np.argmin(sx[ix])])

    lows = {"jul": low_day("2026-07-01", "2026-08-15"), "nov": low_day("2026-10-13", "2026-11-17")}
    base_low = {"jul": int(np.flatnonzero((idx >= "2026-07-01") & (idx <= "2026-08-15"))[np.argmin(P.bc[(idx >= "2026-07-01") & (idx <= "2026-08-15")])]),
                "nov": int(np.flatnonzero(idx >= "2026-10-13")[np.argmin(P.bc[idx >= "2026-10-13"])])}
    print("SOXX close lows:", {k: days[v] for k, v in lows.items()}, "| base NAV close lows:", {k: days[v] for k, v in base_low.items()})
    trades, dayrows = [], []
    r_sx = np.r_[0.0, sx[1:] / sx[:-1] - 1]
    up_days = [int(j) for j in np.argsort(-r_sx)[:3]]
    for key in T.index:
        s = specs[key][1]
        k = k_of(s)
        run = sim(P, s, detail=True)
        ew, navo = run.ew[0], run.navo[0]
        pnl_in, pnl_ov = run.intra[0], run.ov[0]
        for first, last in runs_of(ew > 1e-12):
            exit_px = P.eo[last + 1, k] if last + 1 < P.n else P.ec[last, k]
            pnl = pnl_in[first:last + 1].sum() + pnl_ov[first + 1:last + 2].sum()
            row = {"rule": key, "etf": H.ETFS[k], "entry": days[first], "last": days[last], "sessions": last - first + 1,
                   "w_entry": float(ew[first]), "etf_ret": float(exit_px / P.eo[first, k] - 1),
                   "contrib": float(pnl / navo[first]), "first": first, "last_j": last}
            for tag, jl in lows.items():                 # P&L of the leg after the low close until it is sold
                if first <= jl <= last:
                    row[f"after_{tag}_low"] = float((pnl_in[jl + 1:last + 1].sum() + pnl_ov[jl + 1:last + 2].sum()) / run.usd[0, jl])
            trades.append(row)
        for j in up_days:
            if ew[j] > 1e-12 or (j and ew[j - 1] > 1e-12):
                u, t = L[key].to_numpy(), L[T.at[key, "twin"]].to_numpy()
                dayrows.append({"rule": key, "day": days[j], "etf": H.ETFS[k], "etf_pct": int(T.at[key, "etf_pct"]),
                                "w_open": float(ew[j]), "held_overnight": bool(j and ew[j - 1] > 1e-12),
                                "day_ret": float(u[j] / u[j - 1] - 1), "twin_day_ret": float(t[j] / t[j - 1] - 1)})
    tr = pd.DataFrame(trades)
    out = {"n_trades": int(len(tr)), "n_rules": int(tr.rule.nunique()),
           "trades_won_share": float((tr.contrib > 0).mean()),
           "median_etf_ret_per_trade": float(tr.etf_ret.median()),
           "mean_etf_ret_per_trade": float(tr.etf_ret.mean())}
    uniq = tr.drop_duplicates(["etf", "entry", "last"])
    out["unique_etf_trades"] = int(len(uniq))
    out["unique_trades_won_share"] = float((uniq.etf_ret > 0).mean())
    out["unique_by_etf"] = uniq.groupby("etf").etf_ret.agg(["count", "median", "mean", "min", "max", lambda v: (v > 0).mean()]).rename(
        columns={"<lambda_0>": "won"}).round(4).to_dict("index")
    print("unique ETF trades (distinct instrument, entry, exit):")
    show(pd.DataFrame(out["unique_by_etf"]).T, 3)
    cols = ["rule", "etf", "entry", "last", "sessions", "w_entry", "etf_ret", "contrib"]
    worst_ret = uniq.nsmallest(8, "etf_ret")[cols]
    worst_con = tr.nsmallest(8, "contrib")[cols]
    print("worst single trades by ETF return (unique):")
    show(worst_ret, 3)
    print("worst single trades by account contribution:")
    show(worst_con, 3)
    out["worst_by_etf_return"], out["worst_by_contribution"] = worst_ret.to_dict("records"), worst_con.to_dict("records")
    # shorts opened right before a rebound
    reb = []
    for tag, jl in lows.items():
        for off in (-2, -1, 0, 1):
            j = jl + off
            g = tr[tr["first"] == j]
            if len(g):
                w = g.loc[g.contrib.idxmin()]
                reb.append({"low": f"{tag} {days[jl]}", "entry_open": days[j], "rules": int(len(g)),
                            "median_etf_ret": float(g.etf_ret.median()), "worst_etf_ret": float(g.etf_ret.min()),
                            "median_contrib": float(g.contrib.median()), "worst_contrib": float(w.contrib),
                            "worst_rule": w.rule, "worst_sessions": int(w.sessions)})
        col = f"after_{tag}_low"
        if col in tr:
            g = tr[tr[col].notna()]
            w = g.loc[g[col].idxmin()]
            reb.append({"low": f"{tag} {days[jl]}", "entry_open": "held over the low close", "rules": int(len(g)),
                        "median_etf_ret": np.nan, "worst_etf_ret": np.nan, "median_contrib": float(g[col].median()),
                        "worst_contrib": float(w[col]), "worst_rule": w.rule, "worst_sessions": int(w.last_j - jl)})
    rebf = pd.DataFrame(reb)
    print("shorts entered around the lows (contrib = ETF-leg P&L / account; 'held over' = P&L after the low close until sold):")
    show(rebf, 3)
    out["around_lows"] = rebf.to_dict("records")
    # biggest SOXX up days
    dr = pd.DataFrame(dayrows)
    upt = []
    for j in up_days:
        g = dr[(dr.day == days[j]) & (dr.etf == "SOXS")]
        base_ret = float(P.bc[j] / P.bc[j - 1] - 1)
        head = {"day": days[j], "SOXX_cc": float(r_sx[j]), "SOXS_cc": float(P.ec[j, 0] / P.ec[j - 1, 0] - 1),
                "SOXS_open_to_close": float(P.ec[j, 0] / P.eo[j, 0] - 1), "base_cc": base_ret, "soxs_rules_exposed": int(len(g))}
        print(head)
        if len(g):
            g = g.assign(gap=g.day_ret - g.twin_day_ret)
            tab = g.groupby(["held_overnight", "etf_pct"]).agg(n=("gap", "size"), med_day_ret=("day_ret", "median"),
                                                               worst_day_ret=("day_ret", "min"), med_vs_twin=("gap", "median"),
                                                               worst_vs_twin=("gap", "min"))
            show(tab, 3)
            head["by_overnight_and_weight"] = {f"{a}_{b}": v for (a, b), v in tab.round(4).to_dict("index").items()}
            w = g.loc[g.day_ret.idxmin()]
            head["worst"] = {"rule": w.rule, "day_ret": float(w.day_ret), "twin_day_ret": float(w.twin_day_ret)}
        upt.append(head)
    out["biggest_soxx_up_days"] = upt
    # worst month
    U = L[T.index].to_numpy()
    UT = L[T.twin.to_numpy()].to_numpy()
    ends = pd.DataFrame(U, index=month).groupby(level=0).last()
    ends_t = pd.DataFrame(UT, index=month).groupby(level=0).last()
    ends_b = pd.Series(P.bc, index=month).groupby(level=0).last()
    m = ends / ends.shift().fillna(1.0) - 1
    mt = ends_t / ends_t.shift().fillna(1.0) - 1
    mb = ends_b / ends_b.shift().fillna(1.0) - 1
    m.columns = mt.columns = T.index
    wm = m.min()
    gap = m - mt
    out["base_monthly"] = mb.round(4).to_dict()
    out["worst_month_any_short_rule"] = {"rule": wm.idxmin(), "month": m[wm.idxmin()].idxmin(), "ret": float(wm.min()),
                                         "twin_same_month": float(mt.at[m[wm.idxmin()].idxmin(), wm.idxmin()]),
                                         "base_same_month": float(mb[m[wm.idxmin()].idxmin()])}
    g = gap.min()
    out["worst_month_vs_twin"] = {"rule": g.idxmin(), "month": gap[g.idxmin()].idxmin(), "short_minus_twin": float(g.min()),
                                  "short": float(m.at[gap[g.idxmin()].idxmin(), g.idxmin()]),
                                  "twin": float(mt.at[gap[g.idxmin()].idxmin(), g.idxmin()])}
    out["worst_month_median_over_rules"] = {"short": float(wm.median()), "twin": float(mt.min().median()), "base": float(mb.min())}
    out["rules_with_a_month_worse_than_base_worst_month"] = int((wm < mb.min() - 1e-9).sum())
    bym = pd.DataFrame({"median_short_minus_twin": gap.median(axis=1), "worst_short_minus_twin": gap.min(axis=1),
                        "best_short_minus_twin": gap.max(axis=1), "share_short_better": (gap > 1e-9).mean(axis=1),
                        "share_tie": (gap.abs() <= 1e-9).mean(axis=1), "base": mb})
    print("monthly short-minus-twin over the 968 pairs:")
    show(bym, 3)
    out["monthly_short_minus_twin"] = bym.round(4).to_dict("index")
    print({k: out[k] for k in ("worst_month_any_short_rule", "worst_month_vs_twin", "worst_month_median_over_rules",
                               "rules_with_a_month_worse_than_base_worst_month", "n_trades", "unique_etf_trades",
                               "trades_won_share", "unique_trades_won_share")})
    R["tail"] = out
    return tr


# ---------------------------------------------------------------- 5. placebo
def placebo(P, specs, L, cfg, T, tr):
    say("5. PLACEBO (same special spells, random other times of 2026)")
    idx = P.idx
    i_h2 = int(np.flatnonzero(idx >= "2026-07-01")[0])
    rng = np.random.default_rng(SEED)
    short = list(T.index)
    groups = {}
    for key in short:
        s = specs[key][1]
        mask = (s.a < 1 - 1e-12) | (s.e.sum(1) > 0)
        groups.setdefault(mask.tobytes(), (mask, []))[1].append(key)
    full_p = np.full((len(short), N_DRAW), np.nan)
    h2_p, dfull_p, dh2_p = full_p.copy(), full_p.copy(), full_p.copy()
    twin_full, twin_h2 = {}, {}
    row_of = {k: i for i, k in enumerate(short)}
    ar = np.arange(N_DRAW)[:, None]
    for mask, keys in groups.values():
        runs = runs_of(mask)
        if not runs:
            continue
        lens = np.array([b - a + 1 for a, b in runs])
        m, free = len(runs), P.n - int(lens.sum())
        starts = np.empty((N_DRAW, m), dtype=int)
        for t in range(N_DRAW):
            gaps = np.sort(rng.choice(free + 1, size=m, replace=False))
            shift = 0
            for g, r in zip(gaps, rng.permutation(m)):
                starts[t, r] = g + shift
                shift += lens[r]

        def moved(v, fill):
            out = np.full((N_DRAW, P.n), fill)
            for r, (a, b) in enumerate(runs):
                out[ar, starts[:, r][:, None] + np.arange(lens[r])[None, :]] = v[a:b + 1][None, :]
            return out

        cache = {}
        for key in keys:
            s = specs[key][1]
            k = k_of(s)
            A, X = moved(s.a, 1.0), moved(s.e[:, k], 0.0)
            u = sleeve(P, A, X, P.eo[:, k], P.ec[:, k], s.mode)
            tk = (s.a.tobytes(), s.mode)
            if tk not in cache:
                cache[tk] = sleeve(P, A, np.zeros_like(X), P.eo[:, k], P.ec[:, k], s.mode)
            v = cache[tk]
            i = row_of[key]
            full_p[i], h2_p[i] = u[:, -1] - 1, u[:, -1] / u[:, i_h2 - 1] - 1
            dfull_p[i] = full_p[i] - (v[:, -1] - 1)
            dh2_p[i] = h2_p[i] - (v[:, -1] / v[:, i_h2 - 1] - 1)
            twin_full[T.at[key, "twin"]] = v[:, -1] - 1
            twin_h2[T.at[key, "twin"]] = v[:, -1] / v[:, i_h2 - 1] - 1
    act_full, act_h2 = T.full_s.to_numpy(), T.h2_s.to_numpy()
    act_dfull, act_dh2 = T.full_d.to_numpy(), T.h2_d.to_numpy()
    valid = ~np.isnan(full_p[:, 0])
    fw = {name: np.nanmax(arr[valid], axis=0) for name, arr in (("full", full_p), ("h2", h2_p), ("dfull", dfull_p), ("dh2", dh2_p))}
    pct = lambda arr, act: np.array([np.nan if np.isnan(arr[i, 0]) else 100 * (arr[i] < act[i]).mean() for i in range(len(act))])
    T["pl_full_pct"], T["pl_h2_pct"] = pct(full_p, act_full), pct(h2_p, act_h2)
    T["pl_dfull_pct"], T["pl_dh2_pct"] = pct(dfull_p, act_dfull), pct(dh2_p, act_dh2)
    T["pl_full_med"] = np.nanmedian(full_p, axis=1)
    T["pl_dfull_med"] = np.nanmedian(dfull_p, axis=1)
    T["fw_full_p"] = [(fw["full"] >= v).mean() for v in act_full]
    T["fw_h2_p"] = [(fw["h2"] >= v).mean() for v in act_h2]
    T["fw_dfull_p"] = [(fw["dfull"] >= v).mean() for v in act_dfull]
    T["fw_dh2_p"] = [(fw["dh2"] >= v).mean() for v in act_dh2]
    top = T.full_s.nlargest(10).index
    best_share = {}
    for key in top:
        g = tr[tr.rule == key]
        tot = T.at[key, "full_d"]
        best_share[key] = (len(g), float(g.contrib.max()) if len(g) else np.nan, int((g.contrib > 0).sum()) if len(g) else 0,
                           g.loc[g.contrib.idxmax(), "entry"] if len(g) else "")
    tab = T.loc[top, ["full_s", "full_t", "full_d", "h2_s", "h2_t", "full_mdd_s", "full_mdd_t", "pl_full_pct", "pl_full_med",
                      "pl_h2_pct", "pl_dfull_pct", "pl_dfull_med", "fw_full_p", "fw_dfull_p"]].copy()
    tab["trades"] = [best_share[k][0] for k in top]
    tab["won"] = [best_share[k][2] for k in top]
    tab["best_trade"] = [best_share[k][1] for k in top]
    tab["best_entry"] = [best_share[k][3] for k in top]
    print("10 best full-year short rules (s = short, t = cash twin, d = s - t; pct = percentile among 1,000 random placements; fw_p = family-wise p):")
    show(tab, 3)
    out = {"n_draw": N_DRAW, "seed": SEED, "placement_groups": len(groups), "rules_with_spells": int(valid.sum()),
           "top10": tab.round(4).reset_index().rename(columns={"index": "rule", "strategy": "rule"}).to_dict("records"),
           "family_max_placebo": {k: {"median": float(np.median(v)), "p95": float(np.quantile(v, .95)), "max": float(v.max())}
                                  for k, v in fw.items()},
           "actual_family_best": {"full": [T.full_s.idxmax(), float(T.full_s.max())], "h2": [T.h2_s.idxmax(), float(T.h2_s.max())],
                                  "dfull": [T.full_d.idxmax(), float(T.full_d.max())], "dh2": [T.h2_d.idxmax(), float(T.h2_d.max())]},
           "family_wise_p_of_actual_best": {"full": float((fw["full"] >= T.full_s.max()).mean()), "h2": float((fw["h2"] >= T.h2_s.max()).mean()),
                                            "dfull": float((fw["dfull"] >= T.full_d.max()).mean()), "dh2": float((fw["dh2"] >= T.h2_d.max()).mean())},
           "rules_above_own_p95": {"full": int((T.pl_full_pct >= 95).sum()), "h2": int((T.pl_h2_pct >= 95).sum()),
                                   "dfull": int((T.pl_dfull_pct >= 95).sum()), "dh2": int((T.pl_dh2_pct >= 95).sum())},
           "rules_family_wise_p_below_0.05": {"full": int((T.fw_full_p < .05).sum()), "h2": int((T.fw_h2_p < .05).sum()),
                                              "dfull": int((T.fw_dfull_p < .05).sum()), "dh2": int((T.fw_dh2_p < .05).sum())},
           "median_percentile_by_subfamily": T.groupby("subfamily")[["pl_full_pct", "pl_h2_pct", "pl_dfull_pct", "pl_dh2_pct"]].median().round(2).to_dict("index")}
    # cash twins: is "cash on the signal" better than cash at random times of the same lengths
    tw_rows = []
    for twin, arr in twin_full.items():
        u = L[twin].to_numpy()
        af, ah = u[-1] - 1, u[-1] / u[i_h2 - 1] - 1
        tw_rows.append({"twin": twin, "subfamily": cfg.subfamily[twin], "full": af, "full_pct": 100 * (arr < af).mean(),
                        "h2": ah, "h2_pct": 100 * (twin_h2[twin] < ah).mean()})
    tw = pd.DataFrame(tw_rows).set_index("twin")
    out["cash_twin_placebo"] = {"n": int(len(tw)), "full_pct_median": float(tw.full_pct.median()), "h2_pct_median": float(tw.h2_pct.median()),
                                "above_p95_full": int((tw.full_pct >= 95).sum()), "above_p95_h2": int((tw.h2_pct >= 95).sum()),
                                "by_subfamily": tw.groupby("subfamily")[["full_pct", "h2_pct"]].median().round(2).to_dict("index"),
                                "best_h2_pct": tw.nlargest(8, "h2_pct").round(4).reset_index().to_dict("records")}
    ti = [row_of[k] for k in top]
    fw10 = {"full": np.max(full_p[ti], axis=0), "dfull": np.max(dfull_p[ti], axis=0), "h2": np.max(h2_p[ti], axis=0)}
    out["top10_only_family_wise_p"] = {"full": float((fw10["full"] >= T.loc[top, "full_s"].max()).mean()),
                                       "dfull": float((fw10["dfull"] >= T.loc[top, "full_d"].max()).mean()),
                                       "h2": float((fw10["h2"] >= T.loc[top, "h2_s"].max()).mean())}
    out["max_own_percentile_in_family"] = {c: [T[c].idxmax(), float(T[c].max())] for c in ("pl_full_pct", "pl_h2_pct", "pl_dfull_pct", "pl_dh2_pct")}
    out["expected_rules_above_p95_if_no_skill"] = round(.05 * int(valid.sum()), 1)
    out["by_instrument"] = T.groupby("instrument").apply(lambda g: pd.Series({
        "n": len(g), "pl_dfull_pct_med": g.pl_dfull_pct.median(), "pl_dh2_pct_med": g.pl_dh2_pct.median(),
        "dfull>=p95": (g.pl_dfull_pct >= 95).sum(), "dh2>=p95": (g.pl_dh2_pct >= 95).sum(),
        "full>=p95": (g.pl_full_pct >= 95).sum(), "h2>=p95": (g.pl_h2_pct >= 95).sum()})).round(2).to_dict("index")
    try:
        ref = json.loads((common.CACHE / "hedge_summary.json").read_text(encoding="utf-8"))["headline"]
        out["vs_builder_placebo"] = [{"rule": h["strategy"], "builder_full_pct": h["placebo"]["full_usd_percentile"],
                                      "mine_full_pct": float(T.at[h["strategy"], "pl_full_pct"]),
                                      "builder_h2_pct": h["placebo"]["h2_usd_percentile"], "mine_h2_pct": float(T.at[h["strategy"], "pl_h2_pct"]),
                                      "builder_full_median": h["placebo"]["full_usd_placebo_median"], "mine_full_median": float(T.at[h["strategy"], "pl_full_med"])}
                                     for h in ref if h["strategy"] in T.index and "placebo" in h]
    except (OSError, KeyError) as err:
        out["vs_builder_placebo"] = [{"rule": "unavailable", "error": str(err)}]
    for k in ("top10_only_family_wise_p", "max_own_percentile_in_family", "expected_rules_above_p95_if_no_skill", "by_instrument"):
        print(k, json.dumps(out[k], default=float))
    print("my placebo vs the builder's own placebo (different sampler seeds):")
    show(pd.DataFrame(out["vs_builder_placebo"]).set_index("rule"), 3)
    for k in ("family_max_placebo", "actual_family_best", "family_wise_p_of_actual_best", "rules_above_own_p95",
              "rules_family_wise_p_below_0.05", "median_percentile_by_subfamily"):
        print(k, json.dumps(out[k], default=float))
    print("cash twins vs random placement:", json.dumps({k: v for k, v in out["cash_twin_placebo"].items() if k != "best_h2_pct"}, default=float))
    show(tw.nlargest(8, "h2_pct"), 3)
    R["placebo"] = out
    return tw


# ---------------------------------------------------------------- 6. delay
def delay(P, specs, cfg, T):
    say("6. ONE-SESSION DELAY")
    idx = P.idx
    i_h2 = int(np.flatnonzero(idx >= "2026-07-01")[0])

    def stats(u):
        peak = np.maximum.accumulate(np.r_[1.0, u])
        return u[-1] - 1, u[-1] / u[i_h2 - 1] - 1, float((np.r_[1.0, u] / peak - 1).min())

    rows = {}
    twin_late = {}
    for key in T.index:
        s = specs[key][1]
        k = k_of(s)
        x = s.e[:, k]
        la, lx = np.r_[1.0, s.a[:-1]], np.r_[0.0, x[:-1]]
        f_all, h_all, m_all = stats(sim(P, s, a=la, x=lx)[0])
        twin = T.at[key, "twin"]
        if twin not in twin_late:
            ts = specs[twin][1]
            twin_late[twin] = stats(sim(P, ts, a=np.r_[1.0, ts.a[:-1]], x=np.zeros(P.n))[0])
        if s.mode == "split":
            f_e = h_e = m_e = np.nan
        else:
            f_e, h_e, m_e = stats(sim(P, s, x=np.where(lx > 0, x, 0.0))[0])
        rows[key] = {"full_all": f_all, "h2_all": h_all, "mdd_all": m_all, "full_entry": f_e, "h2_entry": h_e, "mdd_entry": m_e,
                     "twin_full_all": twin_late[twin][0], "twin_h2_all": twin_late[twin][1], "twin_mdd_all": twin_late[twin][2]}
    D = pd.DataFrame.from_dict(rows, orient="index")
    for c in D.columns:
        T[f"dl_{c}"] = D[c]

    def block(g):
        e = g[g.dl_full_entry.notna()]
        return {"n": len(g),
                "full_med": 100 * g.full_s.median(), "full_med_entry": 100 * e.dl_full_entry.median(), "full_med_all": 100 * g.dl_full_all.median(),
                "h2_med": 100 * g.h2_s.median(), "h2_med_entry": 100 * e.dl_h2_entry.median(), "h2_med_all": 100 * g.dl_h2_all.median(),
                "chg_full_entry_med": 100 * (e.dl_full_entry - e.full_s).median(), "chg_full_all_med": 100 * (g.dl_full_all - g.full_s).median(),
                "chg_h2_entry_med": 100 * (e.dl_h2_entry - e.h2_s).median(), "chg_h2_all_med": 100 * (g.dl_h2_all - g.h2_s).median(),
                "win_full%": 100 * (g.full_d > 1e-9).mean(), "win_full_entry%": 100 * (e.dl_full_entry > e.full_t + 1e-9).mean(),
                "win_full_all%": 100 * (g.dl_full_all > g.dl_twin_full_all + 1e-9).mean(),
                "win_h2%": 100 * (g.h2_d > 1e-9).mean(), "win_h2_entry%": 100 * (e.dl_h2_entry > e.h2_t + 1e-9).mean(),
                "win_h2_all%": 100 * (g.dl_h2_all > g.dl_twin_h2_all + 1e-9).mean()}

    rows = {k: block(g) for k, g in T.groupby("subfamily")}
    rows["ALL"] = block(T)
    tab = pd.DataFrame(rows).T
    print("medians in %, chg = delayed minus on-time (points), win% = share beating the cash twin (entry: on-time twin; all: twin delayed too)")
    show(tab, 1)
    top = T.full_s.nlargest(10).index
    t10 = T.loc[top, ["full_s", "dl_full_entry", "dl_full_all", "full_t", "dl_twin_full_all", "h2_s", "dl_h2_entry", "dl_h2_all",
                      "h2_t", "dl_twin_h2_all", "full_mdd_s", "dl_mdd_entry", "dl_mdd_all"]]
    print("10 best full-year short rules under delay:")
    show(t10, 3)
    best_h2 = T.h2_s.nlargest(5).index
    b5 = T.loc[best_h2, ["h2_s", "dl_h2_entry", "dl_h2_all", "h2_t", "dl_twin_h2_all", "full_s", "dl_full_all"]]
    print("5 best H2 short rules under delay:")
    show(b5, 3)
    R["delay"] = {"by_subfamily": tab.round(3).to_dict("index"), "top10_full": t10.round(4).to_dict("index"),
                  "top5_h2": b5.round(4).to_dict("index"),
                  "abs_change_full_all_points_quantiles": {q: float(100 * (T.dl_full_all - T.full_s).quantile(q)) for q in (.05, .25, .5, .75, .95)},
                  "abs_change_h2_all_points_quantiles": {q: float(100 * (T.dl_h2_all - T.h2_s).quantile(q)) for q in (.05, .25, .5, .75, .95)}}


# ---------------------------------------------------------------- 7. controls and bottom line
def bottom(P, L, cfg, T, st, curves, tw, daily):
    say("7. CONTROLS, LEAD'S NUMBERS, BOTTOM LINE")
    idx = P.idx
    base_expo = curves["base"].exposure.to_numpy()
    out = {}
    # lead's medians: excess over the exposure-matched control (ledger exposure = net beta-equivalent)
    rows = {}
    T["h2_expo"], T["full_expo"] = st.loc[T.index, "h2_expo"], st.loc[T.index, "full_expo"]
    for w in ("h2", "full"):
        T[f"{w}_ctrl"] = [my_control(P.bc, base_expo, idx, *WIN[w], e)[0] for e in T[f"{w}_expo"]]
        T[f"{w}_x"] = T[f"{w}_s"] - T[f"{w}_ctrl"]
    for name, g in T.groupby("subfamily"):
        pos = g[g.h2_expo >= 0]
        rows[name] = {"n": len(g), "h2_excess_med": 100 * g.h2_x.median(), "h2_beat_ctrl%": 100 * (g.h2_x > 1e-9).mean(),
                      "full_excess_med": 100 * g.full_x.median(), "full_beat_ctrl%": 100 * (g.full_x > 1e-9).mean(),
                      "h2_net_expo_med": g.h2_expo.median(), "neg_net_expo_n": int((g.h2_expo < 0).sum()),
                      "h2_excess_med_expo>=0": 100 * pos.h2_x.median() if len(pos) else np.nan,
                      "h2_d_twin_med": 100 * g.h2_d.median(), "full_d_twin_med": 100 * g.full_d.median(),
                      "full_med_s": 100 * g.full_s.median(), "full_med_t": 100 * g.full_t.median(),
                      "mddF_med_s": 100 * g.full_mdd_s.median(), "mddF_med_t": 100 * g.full_mdd_t.median()}
    tab = pd.DataFrame(rows).T
    print("short subfamilies vs exposure-matched control (net beta-equivalent exposure) and vs cash twin, points:")
    show(tab, 1)
    out["short_vs_matched_control"] = tab.round(3).to_dict("index")
    # cash twins vs a fixed smaller position of the same average size
    twins = cfg.index[cfg.subfamily.str.contains("cash_twin") | cfg.subfamily.eq("H5_trim50")]
    C = pd.DataFrame(index=twins)
    C["subfamily"] = cfg.subfamily[twins]
    U = L[twins].to_numpy()
    for w in ("full", "h1", "h2", "jan_aug", "sep_nov", "julA", "octB"):
        ret, mdd = wret(U, idx, *WIN[w])
        expo = np.array([common.window_stats(curves[k], *WIN[w])["expo"] for k in twins])
        ctrl = np.array([my_control(P.bc, base_expo, idx, *WIN[w], e) for e in expo])
        C[f"{w}_ret"], C[f"{w}_mdd"], C[f"{w}_expo"] = ret, mdd, expo
        C[f"{w}_x"], C[f"{w}_mx"] = ret - ctrl[:, 0], mdd - ctrl[:, 1]
    C = C[(C.full_expo < base_expo.mean() - 1e-6)]          # twins that were ever out of the market
    rows = {}
    for name, g in list(C.groupby("subfamily")) + [("ALL", C)]:
        rows[name] = {"n": len(g)}
        for w in ("full", "h1", "h2", "sep_nov", "julA", "octB"):
            rows[name][f"{w}_beat%"] = 100 * (g[f"{w}_x"] > 1e-9).mean()
            rows[name][f"{w}_x_med"] = 100 * g[f"{w}_x"].median()
        rows[name]["h2_mdd_beat%"] = 100 * (g.h2_mx > 1e-9).mean()
        rows[name]["full_mdd_beat%"] = 100 * (g.full_mx > 1e-9).mean()
        rows[name]["full&h2_beat%"] = 100 * ((g.full_x > 1e-9) & (g.h2_x > 1e-9)).mean()
        rows[name]["h1&h2_beat%"] = 100 * ((g.h1_x > 1e-9) & (g.h2_x > 1e-9)).mean()
    tabc = pd.DataFrame(rows).T
    print("cash twins vs a fixed position of the same average size in each window (x = twin - control, points):")
    show(tabc, 1)
    out["cash_twin_vs_fixed_size"] = tabc.round(3).to_dict("index")
    both = C[(C.h1_x > 1e-9) & (C.h2_x > 1e-9) & (C.julA_x > 1e-9) & (C.octB_x > 1e-9)]
    out["cash_twins_beating_fixed_size_in_h1_h2_julA_octB"] = {"n": int(len(both)), "of": int(len(C)),
                                                               "by_subfamily": both.subfamily.value_counts().to_dict(),
                                                               "examples": both.nlargest(6, "h2_x")[["full_ret", "h2_ret", "h2_x", "h1_x", "julA_x", "octB_x", "full_expo"]].round(4).to_dict("index")}
    print("cash twins beating the fixed size in H1, H2, 6/23-7/29 and 10/13-11/17:", json.dumps(out["cash_twins_beating_fixed_size_in_h1_h2_julA_octB"], default=float))
    # frozen October rule
    oos = {}
    for key in ("base", "H7_frozen", "H7c_n50_q20_daily", "H1c_SOXX50r1", "H1_SOXX50r1_SOXS_20", "H2c_SOXX50r1_a80", "H2_SOXX50r1_SOXS_s100_20"):
        u = L[key].to_numpy()[:, None]
        oos[key] = {w: [float(wret(u, idx, *WIN[w])[0][0]), float(wret(u, idx, *WIN[w])[1][0])] for w in ("oos", "octB", "h2", "full", "julA")}
    oos["fixed_80pct_of_base"] = {w: list(map(float, my_control(P.bc, base_expo, idx, *WIN[w], .8 * base_expo[(idx >= WIN[w][0]) & (idx <= WIN[w][1])].mean()))) for w in ("oos", "octB", "h2", "full", "julA")}
    fr = pd.DataFrame({k: {w: v[w][0] for w in v} for k, v in oos.items()}).T
    print("frozen October rule and neighbours, USD return by window (oos = 10/12-11/17):")
    show(fr, 3)
    out["frozen"] = oos
    h7 = T[T.subfamily.isin(("H7_frozen", "H7_neighbor"))]
    out["H7_oos"] = {"n": int(len(h7)), "beat_twin_oos": int((h7.oos_d > 1e-9).sum()), "median_oos_short": float(h7.oos_s.median()),
                     "median_oos_twin": float(h7.oos_t.median()), "beat_twin_full": int((h7.full_d > 1e-9).sum()),
                     "beat_twin_julA": int((h7.julA_d > 1e-9).sum())}
    print("H7 (frozen + 47 neighbours):", out["H7_oos"])
    # survivors: short beats twin on full, H2, both MDD; still after delaying everything one session; own placebo >= 95; family-wise
    a = (T.full_d > 1e-9) & (T.h2_d > 1e-9) & (T.full_mdd_d > 1e-9) & (T.h2_mdd_d > 1e-9)
    b = a & (T.dl_full_all > T.dl_twin_full_all + 1e-9) & (T.dl_h2_all > T.dl_twin_h2_all + 1e-9) & (T.dl_mdd_all > T.dl_twin_mdd_all + 1e-9)
    c = b & (T.pl_dfull_pct >= 95) & (T.pl_dh2_pct >= 95)
    d = c & (T.fw_dfull_p < .05)
    e = b & (T.h1_d > 1e-9) & (T.julA_d > 1e-9) & (T.octB_d > 1e-9) & (T.rally_d > -1e-9)
    f = (T.h1_d > 1e-9) & (T.h2_d > 1e-9)
    out["funnel"] = {"pairs": int(len(T)), "A_short_beats_twin_full_h2_mddF_mddH2": int(a.sum()),
                     "B_and_still_after_one_session_delay_of_everything": int(b.sum()),
                     "C_and_short_minus_twin_above_own_placebo_p95_full_and_h2": int(c.sum()),
                     "D_and_family_wise_p<0.05_on_full_short_minus_twin": int(d.sum()),
                     "E_B_and_beats_twin_in_h1_julA_octB_and_not_worse_in_rally": int(e.sum()),
                     "beats_twin_in_both_h1_and_h2": int(f.sum()),
                     "B_by_subfamily": T[b].subfamily.value_counts().to_dict(),
                     "B_by_signal": cfg.signal[T.index[b]].value_counts().head(10).to_dict(),
                     "C_rules": T[c].sort_values("full_d", ascending=False).head(15)[["full_s", "full_t", "h2_s", "h2_t", "h1_d", "julA_d", "rally_d", "octB_d", "full_mdd_s", "full_mdd_t", "pl_dfull_pct", "pl_dh2_pct", "fw_dfull_p"]].round(4).to_dict("index")}
    print("funnel:", json.dumps({k: v for k, v in out["funnel"].items() if k != "C_rules"}, default=float))
    if c.any():
        show(T[c].sort_values("full_d", ascending=False).head(15)[["full_s", "full_t", "h2_s", "h2_t", "h1_d", "julA_d", "rally_d", "octB_d", "full_mdd_s", "full_mdd_t", "pl_dfull_pct", "pl_dh2_pct", "fw_dfull_p"]], 3)
    # reconcile the lead's twin medians (unique twin ledgers) with the pair-matched ones
    full_all = pd.Series(wret(L.to_numpy(), idx, *WIN["full"])[0], index=L.columns)
    h4, h5 = T[T.subfamily == "H4_crash_short"], T[T.subfamily == "H5_exhaustion_short"]
    out["twin_median_full"] = {
        "H4_unique_twins_120": float(full_all[cfg.index[cfg.subfamily == "H4_crash_cash_twin"]].median()),
        "H4_pair_matched_216": float(h4.full_t.median()), "H4_short_216": float(h4.full_s.median()),
        "H4_paired_median_difference": float(h4.full_d.median()),
        "H5_unique_twins_72": float(full_all[cfg.index[cfg.subfamily == "H5_exhaustion_cash_twin"]].median()),
        "H5_unique_twins_with_trim50_90": float(full_all[cfg.index[cfg.subfamily.isin(("H5_exhaustion_cash_twin", "H5_trim50"))]].median()),
        "H5_pair_matched_162": float(h5.full_t.median()), "H5_short_162": float(h5.full_s.median()),
        "H5_paired_median_difference": float(h5.full_d.median())}
    print("twin medians, full year:", json.dumps(out["twin_median_full"], default=float))
    fe = common.window_stats(curves["H7_frozen"], *WIN["oos"])["expo"]
    te = common.window_stats(curves["H7c_n50_q20_daily"], *WIN["oos"])["expo"]
    out["frozen_oos_controls"] = {"frozen_net_beta_expo_oos": float(fe),
                                  "control_at_that_net_expo": float(my_control(P.bc, base_expo, idx, *WIN["oos"], fe)[0]),
                                  "cash_twin_expo_oos": float(te),
                                  "control_at_twin_expo": float(my_control(P.bc, base_expo, idx, *WIN["oos"], te)[0]),
                                  "base_expo_oos": float(base_expo[(idx >= WIN["oos"][0]) & (idx <= WIN["oos"][1])].mean())}
    print("frozen rule post-freeze controls:", json.dumps(out["frozen_oos_controls"], default=float))
    g = T[f]
    out["beat_twin_in_h1_and_h2"] = {"by_instrument": g.instrument.value_counts().to_dict(), "by_subfamily": g.subfamily.value_counts().to_dict(),
                                     "by_signal": cfg.signal[g.index].value_counts().head(8).to_dict(),
                                     "median_full_d": float(g.full_d.median()), "max_full_d": float(g.full_d.max()),
                                     "median_pl_dfull_pct": float(g.pl_dfull_pct.median()),
                                     "also_after_delay_all": int(((g.dl_full_all > g.dl_twin_full_all + 1e-9) & (g.dl_h2_all > g.dl_twin_h2_all + 1e-9)).sum())}
    print("pairs beating the twin in both H1 and H2:", json.dumps(out["beat_twin_in_h1_and_h2"], default=float))
    inst = T.groupby("instrument").apply(lambda q: pd.Series({
        "n": len(q), "full_win%": 100 * (q.full_d > 1e-9).mean(), "h1_win%": 100 * (q.h1_d > 1e-9).mean(), "h2_win%": 100 * (q.h2_d > 1e-9).mean(),
        "rally_win%": 100 * (q.rally_d > 1e-9).mean(), "full_win_after_delay%": 100 * (q.dl_full_all > q.dl_twin_full_all + 1e-9).mean(),
        "h2_win_after_delay%": 100 * (q.dl_h2_all > q.dl_twin_h2_all + 1e-9).mean(), "full_d_med": 100 * q.full_d.median(),
        "full_d_p5": 100 * q.full_d.quantile(.05), "full_d_p95": 100 * q.full_d.quantile(.95), "mddF_d_med": 100 * q.full_mdd_d.median(),
        "h1_d_med": 100 * q.h1_d.median(), "h2_d_med": 100 * q.h2_d.median(), "funnel_B": int(b[q.index].sum())}))
    print("short minus twin by instrument incl. delay:")
    show(inst, 1)
    out["by_instrument"] = inst.round(3).to_dict("index")
    k5 = [k for k in tw.index if k.startswith("H4c_sdd5_h3_")]
    one = T[T.twin == "H4c_sdd5_h3_a0"].iloc[0]
    out["only_cash_signal_beating_fixed_size_everywhere"] = {
        "signal": "SOXX close more than 5% under its 20-session high close, fresh crossing, out for 3 sessions",
        "placebo_pct": tw.loc[k5, ["full_pct", "h2_pct"]].round(2).to_dict("index"),
        "a0_full_on_time": float(one.full_t), "a0_full_delayed_one_session": float(one.dl_twin_full_all),
        "a0_h2_on_time": float(one.h2_t), "a0_h2_delayed_one_session": float(one.dl_twin_h2_all),
        "base_full": float(P.bc[-1] - 1), "base_h2": float(wret(P.bc[:, None], idx, *WIN["h2"])[0][0])}
    print("the one cash signal that beat the fixed size in all four windows:",
          json.dumps(out["only_cash_signal_beating_fixed_size_everywhere"], default=float))
    # where does the short leg earn: per window sum over pairs
    wsum = {w: {"median_d": float(100 * T[f"{w}_d"].median()), "share_better": float(100 * (T[f"{w}_d"] > 1e-9).mean()),
                "share_worse": float(100 * (T[f"{w}_d"] < -1e-9).mean())} for w in ("h1", "julA", "rally", "octB", "h2", "full")}
    print("short minus twin by window, all 968 pairs:", json.dumps(wsum))
    out["short_minus_twin_by_window"] = wsum
    R["bottom"] = out


def main():
    t0 = time.time()
    ctx, data, env, specs, P, daily = setup()
    L = daily.pivot(index="date", columns="strategy", values="usd")
    assert list(L.index) == P.days and L.shape[1] == 1297
    cfg = pd.read_csv(common.CACHE / "hedge_stats.csv", index_col=0)[["subfamily", "signal", "instrument", "etf_pct", "stock_off_pct", "twin"]]
    for key, (config, s) in specs.items():               # every short rule shares its stock leg with its twin
        if config["subfamily"] in SHORT_SUBS:
            ts = specs[config["twin"]][1]
            assert np.array_equal(s.a, ts.a) and ts.mode == s.mode and not ts.e.any(), key
    integrity(ctx, data, env, specs, P, L)
    T = pairs(P, L, cfg)
    st, curves = selection(P, daily, cfg, T)
    tr = tail(P, specs, L, cfg, T, data)
    tw = placebo(P, specs, L, cfg, T, tr)
    delay(P, specs, cfg, T)
    bottom(P, L, cfg, T, st, curves, tw, daily)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    (common.CACHE / "verify_short.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nwrote {common.CACHE / 'verify_short.json'} in {R['runtime_seconds']}s")


if __name__ == "__main__":
    main()
