"""Family "hedge": inverse / leveraged ETF positions on top of the weekly 7-stock plan ("boldly short").

Run from the repo root:  py data/derived/analysis/desk_defense_check_20261119/ext_hedge.py

Timing: every signal is a state after a completed US close; the state after the close of session j-1 sets
the weights traded at the open of session j. ETF prices are real yfinance daily Open/Close (auto_adjust=False,
split-adjusted, distributions NOT added), frozen in frozen_etf/hedge_ohlc.csv; leveraged decay is in the prices.
ETF columns are appended to copies of ctx.opn / ctx.close and every ledger is study.replay(..., all-True gate)
on explicit target vectors (a_j * base basket weights at the open of j, plus ETF weights).

Ledger column `exposure` is the NET beta-equivalent exposure in units of the base basket:
    stock weight + sum_i w_i * L_i / beta(base on underlying_i),   L = -3 SOXS/SQQQ, -1 PSQ, +3 SOXL/TQQQ
(beta = OLS of base close-to-close USD returns on SOXX / QQQ over all 2026 sessions: hindsight constant, used
only so that common.matched_control compares against "the same average net fraction of base"). For rows without
an ETF it equals study.replay's own exposure. Gross invested fraction is kept in cache/hedge_extras.csv.

Grid (1,297 ledgers = 1,296 candidates + base):
  gates (14, True = stocks allowed): ctx.gates QQQ/SOXX x {10,20,50,100}r1 + {20,50}r3 (12), breadth>=19, NOT(SOXX<SMA20 and SOXX<SMA50)
  H1 gate off -> stocks 0, inverse ETF x%, rest USD cash; held without rebalancing until the gate turns on
       14 gates x {SOXS,SQQQ,PSQ} x {10,20,30,50,100}%                                   = 210  (+14 cash twins, x=0)
  H2 gate off -> stocks kept and inverse ETF x% added: s=100 -> stocks 1-x; s=50 -> stocks 50%, cash 50%-x
       14 gates x {SOXS,SQQQ,PSQ} x s{100,50} x {10,20,30}%                              = 252  (+56 cash twins: stocks 90/80/70/50%)
  H3 SOXX >= SMA n -> basket 100%; below -> stocks 0 and SOXS x% for at most T sessions from the cross-down;
       after the time stop cash; re-short for T sessions on a close below the lowest close of the prior n sessions
       n{10,20,50,100} x T{3,5,10,20} x x{10,20,30,50,100}%                              = 80   (cash twin = H1 twin of SOXX n r1)
  H4 crash short on a fresh crossing of: SOXX close more than {5,7.5,10}% under its 20-session high close, or
       base NAV close {8,12,15}% under its 2026 running peak; exit after {3,5,10} sessions or at SOXX close >= SMA10;
       overlay (stocks 1-x, x{10,20,30,50}) or flat (stocks 0, x{10,20,30,50,100})
       6 x 4 x 9                                                                          = 216  (+120 cash twins: stocks 90/80/70/50/0%)
  H5 exhaustion short on a fresh crossing of: SOXX > {10,15,20}% above SMA50, or Wilder RSI14 >= {70,75,80};
       held {5,10,20} sessions; overlay / flat as in H4                                   = 162  (+18 "trim to 50%" +72 other cash twins)
  H6 leveraged long: {SOXL while SOXX>SMA20>SMA50, TQQQ while QQQ>SMA20>SMA50} and breadth >= {none,19,30};
       x{10,20,30,50}% financed from the stocks pro rata                                  = 24
  H7 stop_research's frozen candidate (code/v2_soxs_hedge.py, 2026-10-10): SOXX close < SMA50 -> stocks 80% + SOXS 20%,
       rebalanced to 80/20 every session (1) + its 47 pre-declared neighbours (SMA{40,50,60} x q{10,15,20,25}% x
       {SOXS,SQQQ} x {daily, buy-once-hold}) + its 24 cash twins                          = 72
Writes cache/hedge_daily.parquet, out/hedge_candidates.csv (common.save), cache/hedge_extras.csv, cache/hedge_summary.json.
"""
from __future__ import annotations

import hashlib
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
from common import study  # noqa: E402

FROZEN = HERE / "frozen_etf"
DOWNLOAD = ("SOXS", "SOXL", "SQQQ", "TQQQ", "PSQ", "SH", "SSG", "SOXX", "QQQ")
ETFS = ("SOXS", "SQQQ", "PSQ", "SOXL", "TQQQ")          # appended columns, in this order
LEV = {"SOXS": (-3, "SOXX"), "SQQQ": (-3, "QQQ"), "PSQ": (-1, "QQQ"), "SOXL": (3, "SOXX"), "TQQQ": (3, "QQQ")}
EX = {name: i for i, name in enumerate(ETFS)}
T_TEST = 150
N_PLACEBO = 1000
SEED = 20261119
# Measured in data/derived/analysis/sox_open_switch: median half-spread at 09:30-09:34 ET (cost_table.json) SOXS
# 5.8bp, SOXL 2.3bp; worst day half-spread SOXS 28.8bp (report.md), widest full spread 60.8bp (report8.md).
ETF_STRESS = 0.0030
EXTRA_WINDOWS = {"hell1": ("2026-06-22", "2026-07-29"), "hell2": ("2026-10-13", "2026-11-17"),
                 "oos": ("2026-10-12", "2026-11-17")}


# ---------------------------------------------------------------- data
def load_etf():
    """Frozen yfinance daily bars for the ETFs (downloaded once, then reused)."""
    path, meta_path = FROZEN / "hedge_ohlc.csv", FROZEN / "hedge_meta.json"
    if not path.exists():
        import yfinance as yf
        FROZEN.mkdir(exist_ok=True)
        parts, missing = [], []
        for sym in DOWNLOAD:
            raw = yf.download(sym, start="2025-06-01", end="2026-11-18", auto_adjust=False, actions=True,
                              progress=False, threads=False)
            if raw is None or raw.empty:
                missing.append(sym)
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            frame = raw[["Open", "Close", "Adj Close", "Dividends", "Stock Splits"]].copy()
            frame.insert(0, "symbol", sym)
            frame.insert(0, "date", pd.DatetimeIndex(raw.index).strftime("%Y-%m-%d"))
            parts.append(frame.reset_index(drop=True))
        data = pd.concat(parts, ignore_index=True)
        data.to_csv(path, index=False, float_format="%.10g")
        meta = {"source": "yfinance", "version": yf.__version__, "auto_adjust": False, "start": "2025-06-01",
                "end_exclusive": "2026-11-18", "symbols": sorted(set(data.symbol)), "missing": missing,
                "rows": int(len(data)), "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    data = pd.read_csv(path, dtype={"date": str, "symbol": str})
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return data, meta


def field(data, sym, name):
    part = data[data.symbol == sym].set_index("date")[name].astype(float)
    if part.index.duplicated().any():
        raise ValueError(f"duplicate dates for {sym}")
    return part.sort_index()


def audit_data(data, ctx):
    """Reconcile SOXX/QQQ with the book and sanity-check the leveraged ETFs against their underlyings."""
    out = {}
    for sym in ("SOXX", "QQQ"):
        for name, frame in (("Open", ctx.book.opn), ("Close", ctx.book.cls)):
            mine = field(data, sym, name).reindex(ctx.book.days)
            ref = frame[sym].reindex(ctx.book.days).astype(float)
            both = mine.notna() & ref.notna()
            out[f"{sym}_{name}_max_rel_diff_vs_book"] = float((mine[both] / ref[both] - 1).abs().max())
            out[f"{sym}_{name}_rows_compared"] = int(both.sum())
    for sym, (lev, und) in LEV.items():
        c, u = field(data, sym, "Close"), field(data, und, "Close")
        r = c.pct_change().dropna()
        res = (c.pct_change() - lev * u.reindex(c.index).pct_change()).dropna()
        splits = field(data, sym, "Stock Splits")
        div = field(data, sym, "Dividends")
        y26 = div.index >= "2026-01-01"
        out[sym] = {"rows": int(len(c)), "missing_2026_sessions": int(c.reindex(ctx.days).isna().sum()
                                                                         + field(data, sym, "Open").reindex(ctx.days).isna().sum()),
                    "max_daily_close_move": [float(r.max()), str(r.idxmax()), float(r.min()), str(r.idxmin())],
                    "max_abs_residual_vs_lev_x_underlying": [float(res.abs().max()), str(res.abs().idxmax())],
                    "split_dates": [d for d, v in splits.items() if v],
                    "distributions_2026_sum_div_over_close": float((div[y26] / c.reindex(div.index)[y26]).sum())}
    return out


# ---------------------------------------------------------------- engine
def engine(env, targets):
    """study.replay with an all-True gate, instrumented: weights at each open, per-leg turnover, ETF P&L."""
    om, cm, o = env.om, env.cm, env.o
    n_days, n = om.shape
    n_s = env.n_s
    shares, cash = np.zeros(n), 1.
    usd, navo = np.empty(n_days), np.empty(n_days)
    ts, te, ov, intra = (np.zeros(n_days) for _ in range(4))
    weights = np.zeros((n_days, n))
    for j in range(n_days):
        nav = cash + shares @ om[j]
        if j:
            ov[j] = shares[n_s:] @ (om[j, n_s:] - cm[j - 1, n_s:])
        navo[j] = nav
        w = targets.get(j)
        if w is not None:
            d = np.abs(w - shares * om[j] / nav)
            ts[j], te[j] = d[:n_s].sum(), d[n_s:].sum()
            shares = np.divide(nav * w, o[j], out=np.zeros(n), where=w > 0)
            cash = nav * (1 - w.sum())
        weights[j] = shares * om[j] / nav
        intra[j] = shares[n_s:] @ (cm[j, n_s:] - om[j, n_s:])
        usd[j] = cash + shares @ cm[j]
    return SimpleNamespace(usd=usd, navo=navo, ts=ts, te=te, ov=ov, intra=intra, w=weights)


def make_env(cls, opn_x, close_x, fx, weekly, days, start, breadth_known, n_s):
    """Everything the rules read. cls: DataFrame of book closes (SOXX, QQQ) on the book calendar."""
    env = SimpleNamespace(o=opn_x, om=np.nan_to_num(opn_x), cm=np.nan_to_num(close_x), fx=fx, n_s=n_s,
                          days=days, start=start, weekly=weekly, n_days=len(days), breadth=breadth_known)
    env.gates = study.market_gates(SimpleNamespace(cls=cls), days)
    env.sx, env.qq = cls["SOXX"].to_numpy(float), cls["QQQ"].to_numpy(float)
    pad = {j: np.r_[w, np.zeros(len(ETFS))] for j, w in weekly.items()}
    base = engine(env, pad)
    env.base_targets, env.base_usd, env.base_navo, env.wb = pad, base.usd, base.navo, base.w
    return env


# ---------------------------------------------------------------- signal primitives (book calendar, state after each close)
def sma(x, n):
    return pd.Series(x).rolling(n).mean().to_numpy()


def rsi_wilder(close, n=14):
    d = np.diff(close)
    up, dn = np.maximum(d, 0), np.maximum(-d, 0)
    out = np.full(len(close), np.nan)
    au, ad = up[:n].mean(), dn[:n].mean()
    out[n] = 100. if ad == 0 else 100 - 100 / (1 + au / ad)
    for i in range(n + 1, len(close)):
        au, ad = (au * (n - 1) + up[i - 1]) / n, (ad * (n - 1) + dn[i - 1]) / n
        out[i] = 100. if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def fresh(cond):
    """First close on which cond is true after a close on which it was false."""
    cond = np.asarray(cond, dtype=bool)
    return cond & ~np.r_[True, cond[:-1]]


def timer_state(trigger, n):
    """On for the n sessions after a trigger close; triggers while on are ignored (hard time stop)."""
    out, rem = np.zeros(len(trigger), dtype=bool), 0
    for i in range(len(trigger)):
        if rem:
            rem -= 1
        if trigger[i] and rem == 0:
            rem = n
        out[i] = rem > 0
    return out


def until_state(trigger, release):
    """On from a trigger close (if release is not already true) until a later close with release true."""
    out, on = np.zeros(len(trigger), dtype=bool), False
    for i in range(len(trigger)):
        if on:
            on = not release[i]
        elif trigger[i] and not release[i]:
            on = True
        out[i] = on
    return out


def symmetric_state(close, n, hold):
    """H3. Returns (below SMA n, short active) after each close."""
    line = sma(close, n)
    below = np.zeros(len(close), dtype=bool)
    short = np.zeros(len(close), dtype=bool)
    rem = 0
    for i in range(n, len(close)):
        below[i] = close[i] < line[i]
        if rem:
            rem -= 1
        if not below[i]:
            rem = 0
        elif not (close[i - 1] < line[i - 1]):
            rem = hold                                   # cross-down: first short
        elif rem == 0 and close[i] < close[i - n:i].min():
            rem = hold                                   # re-short on a new n-session low close
        short[i] = rem > 0
    return below, short


# ---------------------------------------------------------------- rule registry
def spec(a, etf, mode="hold"):
    """a[j] stock fraction of the base basket, etf[j, k] ETF weights; mode hold | daily | split."""
    return SimpleNamespace(a=np.asarray(a, dtype=float), e=np.asarray(etf, dtype=float), mode=mode)


def build_specs(env):
    """key -> (config dict, spec). Every array is per session and uses closes through the previous session."""
    n_days, start = env.n_days, env.start

    def known(x):                                        # state after close of calendar i -> session i + 1
        out = np.asarray(x)[start - 1:start - 1 + n_days]
        if len(out) != n_days:
            raise ValueError("calendar too short")
        return out

    def etf_w(on, sym, x):
        e = np.zeros((n_days, len(ETFS)))
        e[:, EX[sym]] = np.where(on, x, 0.)
        return e

    none = np.zeros((n_days, len(ETFS)))
    sx, qq = env.sx, env.qq
    s10, s20, s50 = sma(sx, 10), sma(sx, 20), sma(sx, 50)
    out = {}

    def add(key, s, **config):
        if key in out:
            raise ValueError(key)
        if (s.a < -1e-12).any() or (s.a + s.e.sum(1) > 1 + 1e-12).any() or (s.e < 0).any():
            raise ValueError(f"bad weights {key}")
        out[key] = (config, s)

    gates = {k: v for k, v in env.gates.items() if k != "none"}
    gates["breadth19"] = env.breadth >= 19
    gates["SOXX20and50"] = ~known((sx < s20) & (sx < s50))
    assert len(gates) == 14

    # H1 / H2 and their cash twins
    for g, on in gates.items():
        off = ~on
        add(f"H1c_{g}", spec(on.astype(float), none), subfamily="H1_cash_twin", signal=g, instrument="cash",
            etf_pct=0, stock_off_pct=0, twin="", dup_codex=int(g in env.gates))
        for a_off in (90, 80, 70, 50):
            add(f"H2c_{g}_a{a_off}", spec(np.where(on, 1., a_off / 100), none), subfamily="H2_cash_twin", signal=g,
                instrument="cash", etf_pct=0, stock_off_pct=a_off, twin="", dup_codex=0)
        for sym in ("SOXS", "SQQQ", "PSQ"):
            for x in (10, 20, 30, 50, 100):
                add(f"H1_{g}_{sym}_{x}", spec(on.astype(float), etf_w(off, sym, x / 100)), subfamily="H1_gate_inverse",
                    signal=g, instrument=sym, etf_pct=x, stock_off_pct=0, twin=f"H1c_{g}", dup_codex=0)
            for s in (100, 50):
                for x in (10, 20, 30):
                    a_off = 100 - x if s == 100 else 50
                    add(f"H2_{g}_{sym}_s{s}_{x}", spec(np.where(on, 1., a_off / 100), etf_w(off, sym, x / 100)),
                        subfamily="H2_overlay", signal=g, instrument=sym, etf_pct=x, stock_off_pct=a_off,
                        twin=f"H2c_{g}_a{a_off}", dup_codex=0)

    # H3 symmetric trend following on SOXX
    for n in (10, 20, 50, 100):
        for hold in (3, 5, 10, 20):
            below, short = symmetric_state(sx, n, hold)
            below, short = known(below), known(short)
            for x in (10, 20, 30, 50, 100):
                add(f"H3_n{n}_T{hold}_x{x}", spec((~below).astype(float), etf_w(short, "SOXS", x / 100)),
                    subfamily="H3_symmetric", signal=f"SOXX_sma{n}", instrument="SOXS", etf_pct=x, stock_off_pct=0,
                    n=n, hold=hold, twin=f"H1c_SOXX{n}r1", dup_codex=0)

    # H4 crash shorts, H5 exhaustion shorts (fresh crossings)
    high20 = pd.Series(sx).rolling(20).max().to_numpy()
    nav_dd = np.zeros(len(sx))                           # base NAV close vs its 2026 running peak, on the calendar
    peak = np.maximum.accumulate(np.r_[1., env.base_usd])[1:]
    nav_dd[start:start + n_days] = env.base_usd / peak - 1
    crash = {f"sdd{d:g}": np.nan_to_num(sx / high20 - 1, nan=0.) < -d / 100 for d in (5, 7.5, 10)}
    crash.update({f"nav{d}": nav_dd <= -d / 100 for d in (8, 12, 15)})
    rsi = rsi_wilder(sx)
    stretch = np.nan_to_num(sx / s50 - 1, nan=0.)
    exhaust = {f"str{d}": stretch > d / 100 for d in (10, 15, 20)}
    exhaust.update({f"rsi{d}": np.nan_to_num(rsi, nan=50.) >= d for d in (70, 75, 80)})
    back = np.nan_to_num(sx - s10, nan=-1.) >= 0         # SOXX close back at or above SMA10

    def short_block(tag, sub, trig, label, on, extra):
        for a_off in (90, 80, 70, 50, 0):
            name = "H5_trim50" if (tag == "H5" and a_off == 50) else f"{sub}_cash_twin"
            add(f"{tag}c_{trig}_{label}_a{a_off}", spec(np.where(on, a_off / 100, 1.), none), subfamily=name,
                signal=trig, instrument="cash", etf_pct=0, stock_off_pct=a_off, twin="", dup_codex=0, **extra)
        for x in (10, 20, 30, 50):
            add(f"{tag}_{trig}_{label}_ov_{x}", spec(np.where(on, 1 - x / 100, 1.), etf_w(on, "SOXS", x / 100)),
                subfamily=f"{sub}_short", signal=trig, instrument="SOXS", etf_pct=x, stock_off_pct=100 - x,
                twin=f"{tag}c_{trig}_{label}_a{100 - x}", dup_codex=0, **extra)
        for x in (10, 20, 30, 50, 100):
            add(f"{tag}_{trig}_{label}_flat_{x}", spec(np.where(on, 0., 1.), etf_w(on, "SOXS", x / 100)),
                subfamily=f"{sub}_short", signal=trig, instrument="SOXS", etf_pct=x, stock_off_pct=0,
                twin=f"{tag}c_{trig}_{label}_a0", dup_codex=0, **extra)

    for trig, cond in crash.items():
        hit = fresh(cond)
        for hold in (3, 5, 10):
            short_block("H4", "H4_crash", trig, f"h{hold}", known(timer_state(hit, hold)), {"hold": hold})
        short_block("H4", "H4_crash", trig, "sma10", known(until_state(hit, back)), {"hold": 0})
    for trig, cond in exhaust.items():
        hit = fresh(cond)
        for hold in (5, 10, 20):
            short_block("H5", "H5_exhaustion", trig, f"h{hold}", known(timer_state(hit, hold)), {"hold": hold})

    # H6 leveraged long in confirmed up-trends
    for sym, px in (("SOXL", sx), ("TQQQ", qq)):
        a20, a50 = sma(px, 20), sma(px, 50)
        trend = known(np.nan_to_num(px - a20, nan=-1.) > 0) & known(np.nan_to_num(a20 - a50, nan=-1.) > 0)
        for b in (0, 19, 30):
            on = trend & (env.breadth >= b)
            for x in (10, 20, 30, 50):
                add(f"H6_{sym}_b{b}_x{x}", spec(np.where(on, 1 - x / 100, 1.), etf_w(on, sym, x / 100)),
                    subfamily="H6_lever_long", signal=f"{LEV[sym][1]}>sma20>sma50&breadth>={b}", instrument=sym,
                    etf_pct=x, stock_off_pct=100 - x, twin="base", dup_codex=0)

    # H7 frozen stop_research candidate and its pre-declared neighbourhood
    for n in (40, 50, 60):
        on = known(np.nan_to_num(sma(sx, n) - sx, nan=-1.) > 0)       # strict: close < SMA n
        for q in (10, 15, 20, 25):
            for mode in ("daily", "split"):
                label = "daily" if mode == "daily" else "hold"
                add(f"H7c_n{n}_q{q}_{label}", spec(np.where(on, 1 - q / 100, 1.), none, mode),
                    subfamily="H7_cash_twin", signal=f"SOXX<sma{n}", instrument="cash", etf_pct=0,
                    stock_off_pct=100 - q, n=n, rebalance=label, twin="", dup_codex=0)
                for sym in ("SOXS", "SQQQ"):
                    frozen = (n, q, mode, sym) == (50, 20, "daily", "SOXS")
                    add("H7_frozen" if frozen else f"H7n_n{n}_q{q}_{sym}_{label}",
                        spec(np.where(on, 1 - q / 100, 1.), etf_w(on, sym, q / 100), mode),
                        subfamily="H7_frozen" if frozen else "H7_neighbor", signal=f"SOXX<sma{n}", instrument=sym,
                        etf_pct=q, stock_off_pct=100 - q, n=n, rebalance=label, twin=f"H7c_n{n}_q{q}_{label}",
                        dup_codex=0)
    return out


def targets_of(env, s):
    """Target vectors (stocks + ETFs) for one rule; the default state is the base plan itself."""
    n_days, wb, weekly = env.n_days, env.wb, env.weekly
    a, e = s.a, s.e
    targets = {}
    if s.mode == "split":
        # stop_research path_hold: split once when the signal turns on, never rebalance the split until it turns off.
        on = (a < 1 - 1e-12) | (e.sum(1) > 0)
        k = int(np.argmax(e.max(0))) if e.any() else None
        sv, iv, prev = 1., 0., False
        for j in range(n_days):
            if j:
                sv *= env.base_navo[j] / env.base_navo[j - 1]
                if k is not None:
                    iv *= env.om[j, env.n_s + k] / env.om[j - 1, env.n_s + k]
            tot = sv + iv
            change = on[j] != prev
            if on[j] and not prev:
                sv, iv = a[j] * tot, (1 - a[j]) * tot
            elif prev and not on[j]:
                sv, iv = tot, 0.
            if change or j in weekly:
                w = np.r_[sv / tot * wb[j, :env.n_s], np.zeros(len(ETFS))]
                if k is not None:
                    w[env.n_s + k] = iv / tot
                targets[j] = w
            prev = on[j]
        return targets
    prev_a, prev_e = 1., np.zeros(len(ETFS))
    for j in range(n_days):
        changed = abs(a[j] - prev_a) > 1e-12 or np.abs(e[j] - prev_e).max() > 1e-12
        special = a[j] < 1 - 1e-12 or e[j].any()
        if changed or (j in weekly and a[j] > 0) or (s.mode == "daily" and special):
            targets[j] = np.r_[a[j] * wb[j, :env.n_s], e[j]]
        prev_a, prev_e = a[j], e[j]
    return targets


# ---------------------------------------------------------------- self-tests
def lookahead_test(ctx, cls, opn_x, close_x, breadth_known, env, specs, mode):
    """Prices after session T scaled (uniform x1.5, or scrambled x U(0.5,2) per cell): decisions through T must not move."""
    rng = np.random.default_rng(7)
    cut = ctx.days[T_TEST]

    def factor(shape):
        return 1.5 if mode == "uniform" else rng.uniform(.5, 2., shape)

    cls2 = cls.copy()
    late = cls2.index > cut
    cls2.loc[late] = cls2.loc[late].to_numpy() * factor((int(late.sum()), cls2.shape[1]))
    opn2, close2 = opn_x.copy(), close_x.copy()
    opn2[T_TEST + 1:] *= factor(opn2[T_TEST + 1:].shape)
    close2[T_TEST + 1:] *= factor(close2[T_TEST + 1:].shape)
    env2 = make_env(cls2, opn2, close2, ctx.fx, ctx.weekly, ctx.days, ctx.start, breadth_known, env.n_s)
    specs2 = build_specs(env2)
    assert set(specs) == set(specs2)
    states_changed_after = targets_checked = 0
    for key, (_, s) in specs.items():
        s2 = specs2[key][1]
        # states for sessions <= T+1 depend on closes <= T only
        if not (np.array_equal(s.a[:T_TEST + 2], s2.a[:T_TEST + 2]) and np.array_equal(s.e[:T_TEST + 2], s2.e[:T_TEST + 2])):
            raise AssertionError(f"look-ahead in state of {key}")
        states_changed_after += not (np.array_equal(s.a, s2.a) and np.array_equal(s.e, s2.e))
        t1 = {j: w for j, w in targets_of(env, s).items() if j <= T_TEST}
        t2 = {j: w for j, w in targets_of(env2, s2).items() if j <= T_TEST}
        if set(t1) != set(t2) or any(not np.allclose(t1[j], t2[j], rtol=0, atol=1e-12) for j in t1):
            raise AssertionError(f"look-ahead in targets of {key}")
        targets_checked += len(t1)
    return dict(mode=mode, T=T_TEST, cut_date=cut, rules_tested=len(specs), states_identical_through_T_plus_1=len(specs),
                targets_identical_through_T=len(specs), target_vectors_compared=targets_checked,
                rules_whose_later_states_changed=int(states_changed_after))


# ---------------------------------------------------------------- analysis helpers
def win(series, index, a, b):
    """Return and max drawdown of a NAV array inside [a, b], anchored on the close before the window."""
    ix = np.flatnonzero((index >= a) & (index <= b))
    before = series[ix[0] - 1] if ix[0] else 1.0
    path = np.r_[before, series[ix]]
    return float(path[-1] / before - 1), float(np.min(path / np.maximum.accumulate(path) - 1))


def control(base_curve, expo, window):
    """common.matched_control without its exposure <= base restriction (same formula)."""
    a, b = common.WINDOWS[window]
    ix = np.flatnonzero((base_curve.index >= a) & (base_curve.index <= b))
    normal = base_curve.exposure.to_numpy()[ix].mean()
    usd = base_curve.usd.to_numpy()
    path = np.r_[usd[ix[0] - 1] if ix[0] else 1.0, usd[ix]]
    nav = np.r_[1., np.cumprod(1 + expo / normal * (path[1:] / path[:-1] - 1))]
    return float(nav[-1] - 1), float(np.min(nav / np.maximum.accumulate(nav) - 1))


def spells(on):
    """[(first, last)] inclusive runs of True."""
    out, begin = [], None
    for j, v in enumerate(on):
        if v and begin is None:
            begin = j
        if begin is not None and (not v or j == len(on) - 1):
            out.append((begin, j if v else j - 1))
            begin = None
    return out


def headline(env, ctx, key, s, base_run, days):
    """Trade list, worst month, where the excess comes from, delays, costs and the random-timing placebo."""
    idx = pd.Index(days)
    run = engine(env, targets_of(env, s))
    etf_on = run.w[:, env.n_s:].sum(1) > 1e-12
    k = int(np.argmax(s.e.max(0))) if s.e.any() else None
    trades = []
    for first, last in spells(etf_on):
        entry = env.om[first, env.n_s + k]
        exit_ = env.om[last + 1, env.n_s + k] if last + 1 < env.n_days else env.cm[last, env.n_s + k]
        pnl = run.intra[first:last + 1].sum() + run.ov[first + 1:last + 2].sum()
        trades.append({"entry": days[first], "last_session": days[last], "sessions": last - first + 1,
                       "etf_return": float(exit_ / entry - 1), "account_contribution": float(pnl / run.navo[first])})
    out = {"strategy": key, "etf": ETFS[k] if k is not None else "none", "n_etf_trades": len(trades),
           "etf_sessions": int(etf_on.sum())}
    if trades:
        worst = min(trades, key=lambda t: t["account_contribution"])
        best = max(trades, key=lambda t: t["account_contribution"])
        out.update(worst_trade=worst, best_trade=best,
                   trades_won=int(sum(t["account_contribution"] > 0 for t in trades)),
                   sum_trade_contribution=float(sum(t["account_contribution"] for t in trades)),
                   trades=trades if len(trades) <= 40 else trades[:40])
    month = idx.str[:7]
    usd = pd.Series(run.usd, index=idx)
    ends = usd.groupby(month).last()
    monthly = ends / np.r_[1., ends.to_numpy()[:-1]] - 1
    base_ends = pd.Series(base_run.usd, index=idx).groupby(month).last()
    base_monthly = base_ends / np.r_[1., base_ends.to_numpy()[:-1]] - 1
    out["monthly_usd"] = {m: round(float(v), 4) for m, v in monthly.items()}
    out["worst_month"] = [str(monthly.idxmin()), float(monthly.min())]
    gap = monthly - base_monthly
    out["worst_month_vs_base"] = [str(gap.idxmin()), float(gap.min())]
    etf_pnl = pd.Series(run.intra + run.ov, index=idx)
    out["etf_leg_pnl_by_month_nav_units"] = {m: round(float(v), 4) for m, v in etf_pnl.groupby(month).sum().items()}

    def log_excess(a, b):
        ix = np.flatnonzero((idx >= a) & (idx <= b))
        i0 = ix[0] - 1
        r = run.usd[ix[-1]] / (run.usd[i0] if ix[0] else 1.)
        r0 = base_run.usd[ix[-1]] / (base_run.usd[i0] if ix[0] else 1.)
        return float(np.log(r / r0))

    total = log_excess("2026-01-01", "2026-12-31")
    parts = {name: log_excess(a, b) for name, (a, b) in (("hell1_0622_0729", EXTRA_WINDOWS["hell1"]),
                                                          ("hell2_1013_1117", EXTRA_WINDOWS["hell2"]))}
    out["log_excess_vs_base_full"] = total
    out["log_excess_vs_base_by_window"] = parts
    out["share_of_excess_from_1013_1117"] = float(parts["hell2_1013_1117"] / total) if abs(total) > 1e-12 else None
    out["share_of_excess_from_0622_0729"] = float(parts["hell1_0622_0729"] / total) if abs(total) > 1e-12 else None
    out["excess_outside_both_declines"] = float(total - sum(parts.values()))
    ix2 = np.flatnonzero((idx >= EXTRA_WINDOWS["hell2"][0]) & (idx <= EXTRA_WINDOWS["hell2"][1]))
    out["etf_leg_pnl_full_nav_units"] = float(etf_pnl.sum())
    out["etf_leg_pnl_1013_1117_nav_units"] = float(etf_pnl.to_numpy()[ix2].sum())

    def rets(r, cost_s=0., cost_e=0.):
        drag = np.cumprod(1 - cost_s * r.ts - cost_e * r.te)
        nav = r.usd * drag
        d = {}
        for name, (a, b) in {**common.WINDOWS, **EXTRA_WINDOWS}.items():
            if name in ("full", "h1", "h2", "jul_aug", "sep_nov", "oct_nov", "oos"):
                d[f"{name}_usd"], d[f"{name}_usd_mdd"] = (round(v, 6) for v in win(nav, idx, a, b))
        return d

    out["cost0"] = rets(run)
    out["cost_10bp_all"] = rets(run, .001, .001)
    out["cost_10bp_stock_30bp_etf"] = rets(run, .001, ETF_STRESS)
    out["turnover_full"] = {"stock": float(run.ts.sum()), "etf": float(run.te.sum())}

    # one-session delays
    late_all = spec(np.r_[1., s.a[:-1]], np.vstack([np.zeros((1, len(ETFS))), s.e[:-1]]), s.mode)
    out["delay_all_one_session"] = rets(engine(env, targets_of(env, late_all)))
    was_on = np.vstack([np.zeros((1, len(ETFS))), s.e[:-1]]) > 0
    late_entry = spec(s.a, np.where(was_on, s.e, 0.), s.mode)
    out["delay_etf_entry_only"] = rets(engine(env, targets_of(env, late_entry)))

    # placebo: the same special-state spells (same rows, same lengths) at random other times in 2026
    special = (s.a < 1 - 1e-12) | (s.e.sum(1) > 0)
    runs = spells(special)
    out["special_spells"] = len(runs)
    out["special_sessions"] = int(special.sum())
    if runs:
        rng = np.random.default_rng(SEED)
        free = env.n_days - int(special.sum())
        full, h2 = np.empty(N_PLACEBO), np.empty(N_PLACEBO)
        for t in range(N_PLACEBO):
            gaps = np.sort(rng.choice(free + 1, size=len(runs), replace=False))   # >= 1 free session between spells
            order = rng.permutation(len(runs))
            a2, e2 = np.ones(env.n_days), np.zeros_like(s.e)
            shift = 0
            for g, r in zip(gaps, order):
                first, last = runs[r]
                length = last - first + 1
                a2[g + shift:g + shift + length] = s.a[first:last + 1]
                e2[g + shift:g + shift + length] = s.e[first:last + 1]
                shift += length
            p = engine(env, targets_of(env, spec(a2, e2, s.mode))).usd
            full[t], h2[t] = p[-1] - 1, win(p, idx, *common.WINDOWS["h2"])[0]
        actual_full, actual_h2 = run.usd[-1] - 1, win(run.usd, idx, *common.WINDOWS["h2"])[0]
        out["placebo"] = {"n": N_PLACEBO, "seed": SEED,
                          "full_usd_actual": float(actual_full), "full_usd_percentile": float((full < actual_full).mean() * 100),
                          "full_usd_placebo_median": float(np.median(full)), "full_usd_placebo_p95": float(np.quantile(full, .95)),
                          "h2_usd_actual": float(actual_h2), "h2_usd_percentile": float((h2 < actual_h2).mean() * 100),
                          "h2_usd_placebo_median": float(np.median(h2)), "h2_usd_placebo_p95": float(np.quantile(h2, .95))}
    return out


# ---------------------------------------------------------------- build
def build():
    ctx = common.context()
    days = ctx.days
    n_s = len(ctx.tickers)
    data, meta = load_etf()
    audit = audit_data(data, ctx)
    etf_o = np.column_stack([field(data, s, "Open").reindex(days).to_numpy() for s in ETFS])
    etf_c = np.column_stack([field(data, s, "Close").reindex(days).to_numpy() for s in ETFS])
    if not (np.isfinite(etf_o).all() and np.isfinite(etf_c).all() and (etf_o > 0).all()):
        raise ValueError("ETF bars incomplete on the 2026 sessions")
    opn_x, close_x = np.hstack([ctx.opn, etf_o]), np.hstack([ctx.close, etf_c])
    breadth = pd.read_csv(common.CODEX / "breadth.csv", index_col=0).breadth.shift().reindex(days)
    assert breadth.notna().all()
    breadth_known = breadth.to_numpy()
    cls = ctx.book.cls[["SOXX", "QQQ"]].astype(float)
    assert list(cls.index) == list(ctx.book.days)
    env = make_env(cls, opn_x, close_x, ctx.fx, ctx.weekly, days, ctx.start, breadth_known, n_s)
    for name, gate in ctx.gates.items():
        assert np.array_equal(env.gates[name], gate), name
    args = (days, opn_x, close_x, ctx.fx)
    all_on = np.ones(len(days), dtype=bool)

    # base: extended arrays reproduce study.replay on the original arrays, and the desk model itself
    base_ref = study.replay(days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, ctx.gates["none"])
    base = study.replay(*args, env.base_targets, all_on)
    np.testing.assert_allclose(base.to_numpy(), base_ref.to_numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(base.usd, ctx.baseline.close, rtol=1e-10)
    np.testing.assert_allclose(env.base_usd, base.usd.to_numpy(), rtol=1e-12)
    base_err = float(np.abs(base.to_numpy() / np.where(base_ref.to_numpy() == 0, 1, base_ref.to_numpy()) - 1).max())

    # hindsight betas for the net-exposure column
    r_base = base.usd.to_numpy()[1:] / base.usd.to_numpy()[:-1] - 1
    beta = {}
    for sym, px in (("SOXX", env.sx), ("QQQ", env.qq)):
        c = px[ctx.start:ctx.start + len(days)]
        r = c[1:] / c[:-1] - 1
        beta[sym] = float(np.cov(r_base, r)[0, 1] / np.var(r, ddof=1))
        for name, (a, b) in (("h1", common.WINDOWS["h1"]), ("h2", common.WINDOWS["h2"])):
            m = (np.array(days[1:]) >= a) & (np.array(days[1:]) <= b)
            beta[f"{sym}_{name}"] = float(np.cov(r_base[m], r[m])[0, 1] / np.var(r[m], ddof=1))
    unit = np.array([LEV[s][0] / beta[LEV[s][1]] for s in ETFS])

    specs = build_specs(env)
    selftest = [lookahead_test(ctx, cls, opn_x, close_x, breadth_known, env, specs, mode)
                for mode in ("uniform", "scrambled")]

    curves, configs, extras, runs = {"base": base}, {"base": dict(subfamily="base", signal="none", instrument="none",
                                                                   etf_pct=0, stock_off_pct=100, twin="", dup_codex=0)}, {}, {}
    worst_engine_gap = 0.
    for key, (config, s) in specs.items():
        targets = targets_of(env, s)
        ledger = study.replay(*args, targets, all_on)
        run = engine(env, targets)
        gross = run.w.sum(1)
        worst_engine_gap = max(worst_engine_gap, float(np.abs(run.usd / ledger.usd.to_numpy() - 1).max()),
                               float(np.abs(gross - ledger.exposure.to_numpy()).max()),
                               float(np.abs(run.ts + run.te - ledger.turn.to_numpy()).max()))
        stock_w, etf_w = run.w[:, :n_s].sum(1), run.w[:, n_s:]
        ledger["exposure"] = stock_w + etf_w @ unit
        curves[key], configs[key] = ledger, config
        h2 = np.array(days) >= "2026-07-01"
        extras[key] = {"gross_h2": float(gross[h2].mean()), "stock_w_h2": float(stock_w[h2].mean()),
                       "etf_w_h2": float(etf_w.sum(1)[h2].mean()), "etf_sessions_full": int((etf_w.sum(1) > 1e-12).sum()),
                       "etf_sessions_h2": int((etf_w.sum(1)[h2] > 1e-12).sum()),
                       "special_spells_full": len(spells((s.a < 1 - 1e-12) | (s.e.sum(1) > 0))),
                       "etf_turn_full": float(run.te.sum()), "stock_turn_full": float(run.ts.sum())}
    assert worst_engine_gap < 1e-10, worst_engine_gap
    # cash twins of Codex's 12 gates must be Codex's own gate ledgers
    for name, gate in ctx.gates.items():
        if name != "none":
            ref = study.replay(days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, gate)
            np.testing.assert_allclose(curves[f"H1c_{name}"].to_numpy(), ref.to_numpy(), rtol=1e-9, atol=1e-10)
    ref = study.replay(days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, breadth_known >= 19)
    np.testing.assert_allclose(curves["H1c_breadth19"].usd, ref.usd, rtol=1e-9)
    assert len(curves) == 1297, len(curves)
    for key, config in configs.items():
        config.setdefault("n", 0), config.setdefault("hold", 0), config.setdefault("rebalance", "")
        config["expo_def"] = "net_beta_equiv"
    common.save("hedge", curves, configs)
    pd.DataFrame.from_dict(extras, orient="index").rename_axis("strategy").to_csv(common.CACHE / "hedge_extras.csv")
    return SimpleNamespace(ctx=ctx, env=env, specs=specs, curves=curves, configs=configs, extras=extras, meta=meta,
                           audit=audit, beta=beta, selftest=selftest, base_err=base_err, engine_gap=worst_engine_gap)


# ---------------------------------------------------------------- report
def report(b, elapsed):
    ctx, env, curves, configs = b.ctx, b.env, b.curves, b.configs
    days = ctx.days
    idx = pd.Index(days)
    st = common.stats_table(curves)
    for name, (lo, hi) in EXTRA_WINDOWS.items():
        vals = {k: common.window_stats(c, lo, hi) for k, c in curves.items()}
        for m in ("usd", "usd_mdd", "krw", "expo"):
            st[f"{name}_{m}"] = pd.Series({k: v[m] for k, v in vals.items()})
    for w, (lo, hi) in common.WINDOWS.items():
        st[f"{w}_krw10"] = pd.Series({k: win(c.krw10.to_numpy(), idx, lo, hi)[0] for k, c in curves.items()})
        st[f"{w}_usd10"] = pd.Series({k: win((c.krw10 / ctx.fx).to_numpy(), idx, lo, hi)[0] for k, c in curves.items()})
    cfg = pd.DataFrame.from_dict(configs, orient="index")
    base_curve, base = curves["base"], st.loc["base"]
    twin_of = cfg.twin.to_dict()
    is_twin = cfg.subfamily.str.contains("cash_twin") | cfg.subfamily.eq("H5_trim50")
    out = {"n_ledgers": len(curves), "subfamily_counts": cfg.subfamily.value_counts().sort_index().to_dict(),
           "etf_meta": b.meta, "data_audit": b.audit, "beta_base_on_index": b.beta, "base_reconciliation_max_rel": b.base_err,
           "engine_vs_study_replay_max_gap": b.engine_gap, "lookahead_selftest": b.selftest,
           "base": {k: round(float(base[k]), 6) for k in st.columns}}

    def row(key, train=None, test=None):
        r = st.loc[key]
        d = {"strategy": key, "subfamily": configs[key]["subfamily"]}
        for w in ([train, test] if train else ["full", "h1", "h2", "jul_aug", "sep_nov", "oct_nov"]):
            for m in ("krw", "krw_mdd", "usd", "usd_mdd", "expo", "turn", "krw10", "usd10"):
                d[f"{w}_{m}"] = round(float(r[f"{w}_{m}"]), 6)
        for w in EXTRA_WINDOWS:
            d[f"{w}_usd"], d[f"{w}_usd_mdd"] = round(float(r[f"{w}_usd"]), 6), round(float(r[f"{w}_usd_mdd"]), 6)
        t = test or "h2"
        c = control(base_curve, r[f"{t}_expo"], t)
        d[f"{t}_matched_usd"], d[f"{t}_matched_usd_mdd"] = round(c[0], 6), round(c[1], 6)
        twin = twin_of.get(key, "")
        if twin:
            d["twin"] = twin
            d[f"{t}_twin_usd"], d[f"{t}_twin_usd_mdd"] = (round(float(st.at[twin, f"{t}_usd"]), 6),
                                                          round(float(st.at[twin, f"{t}_usd_mdd"]), 6))
        if key in b.extras:
            d.update({k: b.extras[key][k] for k in ("gross_h2", "stock_w_h2", "etf_w_h2", "etf_sessions_h2",
                                                    "special_spells_full")})
        return d

    out["selections"] = []
    for protocol, train, test in (("P1", "h1", "h2"), ("P2", "jan_aug", "sep_nov")):
        for scope, table in (("family", st), ("family_without_cash_twins", st[~is_twin.reindex(st.index)])):
            for currency in ("krw", "usd"):
                for rule, key in common.select(table, train, currency).items():
                    out["selections"].append({"protocol": protocol, "scope": scope, "select_in": currency, "rule": rule,
                                              **row(key, train, test)})
    cand = st.drop(index="base")
    out["hindsight"] = {"best_h2_usd": row(cand.h2_usd.idxmax()), "best_sep_nov_usd": row(cand.sep_nov_usd.idxmax()),
                        "best_h2_usd_mdd": row(cand.h2_usd_mdd.idxmax()), "best_full_usd": row(cand.full_usd.idxmax()),
                        "best_h2_krw": row(cand.h2_krw.idxmax())}

    # counts
    counts = []
    for test in ("h2", "sep_nov", "full"):
        ctrl = np.array([control(base_curve, e, test) for e in cand[f"{test}_expo"]])
        merge_style = np.array([common.matched_control(base_curve, e, test) for e in cand[f"{test}_expo"]])
        both = (cand[f"{test}_usd"] > base[f"{test}_usd"] + 1e-9) & (cand[f"{test}_usd_mdd"] > base[f"{test}_usd_mdd"] + 1e-9)
        skill = both & (cand[f"{test}_usd"] > ctrl[:, 0] + 1e-9)
        skill2 = skill & (cand[f"{test}_usd_mdd"] > ctrl[:, 1] + 1e-9)
        merge_skill = both & (cand[f"{test}_usd"] > merge_style[:, 0] + 1e-9)       # NaN control -> not counted
        twin_keys = pd.Series({k: twin_of.get(k, "") for k in cand.index})
        has = twin_keys.ne("")
        tw_ret = pd.Series({k: st.at[t, f"{test}_usd"] if t else np.nan for k, t in twin_keys.items()})
        tw_mdd = pd.Series({k: st.at[t, f"{test}_usd_mdd"] if t else np.nan for k, t in twin_keys.items()})
        beat_twin = has & (cand[f"{test}_usd"] > tw_ret + 1e-9)
        beat_twin2 = beat_twin & (cand[f"{test}_usd_mdd"] > tw_mdd + 1e-9)
        groups = [("ALL", cand.index), ("ALL_without_cash_twins", cand.index[~is_twin.reindex(cand.index)])]
        groups += [(name, part.index) for name, part in cand.groupby(cfg.subfamily.reindex(cand.index))]
        for name, keys in groups:
            counts.append({"test": test, "group": name, "n": len(keys), "beat_base_ret_and_mdd": int(both[keys].sum()),
                           "also_beat_matched_ret": int(skill[keys].sum()),
                           "also_beat_matched_ret_and_mdd": int(skill2[keys].sum()),
                           "also_beat_matched_ret_merge_style": int(merge_skill[keys].sum()),
                           "with_twin": int(has[keys].sum()), "beat_twin_ret": int(beat_twin[keys].sum()),
                           "beat_twin_ret_and_mdd": int(beat_twin2[keys].sum()),
                           "beat_base_and_twin_ret_and_mdd": int((both & beat_twin2)[keys].sum())})
    out["counts"] = counts

    # per-subfamily table
    table = []
    for name, part in st.groupby(cfg.subfamily.reindex(st.index)):
        table.append({"subfamily": name, "n": len(part), "h2_usd_median": float(part.h2_usd.median()),
                      "h2_usd_best": float(part.h2_usd.max()), "h2_usd_worst": float(part.h2_usd.min()),
                      "h2_usd_mdd_median": float(part.h2_usd_mdd.median()),
                      "jul_aug_usd_median": float(part.jul_aug_usd.median()), "sep_nov_usd_median": float(part.sep_nov_usd.median()),
                      "h1_usd_median": float(part.h1_usd.median()), "full_usd_median": float(part.full_usd.median()),
                      "h2_expo_median": float(part.h2_expo.median()), "h2_turn_median": float(part.h2_turn.median()),
                      "oos_1012_1117_usd_median": float(part.oos_usd.median()),
                      "beat_base_h2_ret_and_mdd": int(((part.h2_usd > base.h2_usd + 1e-9) & (part.h2_usd_mdd > base.h2_usd_mdd + 1e-9)).sum())})
    out["subfamily_table"] = table

    # rank persistence inside the family
    out["rank_corr"] = {f"{a}->{t}_{cur}": float(cand[f"{a}_{cur}"].rank().corr(cand[f"{t}_{cur}"].rank()))
                        for a, t in (("h1", "h2"), ("jan_aug", "sep_nov")) for cur in ("krw", "usd")}

    # H7: the frozen candidate out of sample (10-12..11-17) with its neighbourhood
    h7 = [k for k in st.index if configs[k]["subfamily"] in ("H7_frozen", "H7_neighbor")]
    oos_ctrl80 = engine(env, {j: .8 * w for j, w in env.base_targets.items()})
    fixed80 = win(oos_ctrl80.usd, idx, *EXTRA_WINDOWS["oos"])
    out["H7"] = {"frozen": row("H7_frozen"), "frozen_cash_twin": row("H7c_n50_q20_daily"),
                 "base_oos_usd": float(base.oos_usd), "base_oos_usd_mdd": float(base.oos_usd_mdd),
                 "fixed_80pct_stock_oos_usd": fixed80[0], "fixed_80pct_stock_oos_usd_mdd": fixed80[1],
                 "frozen_pre_freeze_0102_1009_usd": win(curves["H7_frozen"].usd.to_numpy(), idx, "2026-01-02", "2026-10-09")[0],
                 "base_pre_freeze_0102_1009_usd": win(base_curve.usd.to_numpy(), idx, "2026-01-02", "2026-10-09")[0],
                 "neighbors_n": len(h7),
                 "neighbors_beat_base_oos_ret": int((st.loc[h7, "oos_usd"] > base.oos_usd + 1e-9).sum()),
                 "neighbors_beat_base_oos_ret_and_mdd": int(((st.loc[h7, "oos_usd"] > base.oos_usd + 1e-9)
                                                             & (st.loc[h7, "oos_usd_mdd"] > base.oos_usd_mdd + 1e-9)).sum()),
                 "neighbors_beat_cash_twin_oos_ret": int(sum(st.at[k, "oos_usd"] > st.at[twin_of[k], "oos_usd"] + 1e-9 for k in h7)),
                 "neighbors_oos_usd_median": float(st.loc[h7, "oos_usd"].median()),
                 "neighbors_oos_usd_min_max": [float(st.loc[h7, "oos_usd"].min()), float(st.loc[h7, "oos_usd"].max())],
                 "cash_twins_oos_usd_median": float(st.loc[[k for k in st.index if configs[k]["subfamily"] == "H7_cash_twin"], "oos_usd"].median()),
                 "soxx_below_sma50_sessions": {"0622_0729": int((curves["H7_frozen"].loc[EXTRA_WINDOWS["hell1"][0]:EXTRA_WINDOWS["hell1"][1], "exposure"] < .99).sum()),
                                               "1012_1117": int((curves["H7_frozen"].loc[EXTRA_WINDOWS["oos"][0]:EXTRA_WINDOWS["oos"][1], "exposure"] < .99).sum()),
                                               "oos_sessions": int(((idx >= EXTRA_WINDOWS["oos"][0]) & (idx <= EXTRA_WINDOWS["oos"][1])).sum())}}

    # headline rules: selections (family scope and twin-free scope), hindsight bests, frozen, best non-twin per short subfamily
    heads = []
    for s in out["selections"]:
        heads.append(s["strategy"])
    heads += [out["hindsight"][k]["strategy"] for k in ("best_h2_usd", "best_sep_nov_usd", "best_h2_usd_mdd")]
    heads.append("H7_frozen")
    for name in ("H1_gate_inverse", "H2_overlay", "H3_symmetric", "H4_crash_short", "H5_exhaustion_short", "H6_lever_long"):
        part = st[cfg.subfamily.reindex(st.index) == name]
        heads += [part.h2_usd.idxmax(), part.jan_aug_krw.idxmax()]
    heads = [k for k in dict.fromkeys(heads) if k != "base"]
    base_run = engine(env, env.base_targets)
    out["headline"] = []
    for key in heads:
        h = headline(env, ctx, key, b.specs[key][1], base_run, days)
        h["stats"] = row(key)
        out["headline"].append(h)
    out["headline_keys"] = heads
    out["runtime_seconds"] = round(time.time() - elapsed, 1)
    st.join(cfg[["subfamily", "signal", "instrument", "etf_pct", "stock_off_pct", "twin"]]).to_csv(
        common.CACHE / "hedge_stats.csv", index_label="strategy")
    (common.CACHE / "hedge_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float),
                                                    encoding="utf-8")
    brief = {k: out[k] for k in ("n_ledgers", "subfamily_counts", "base_reconciliation_max_rel",
                                 "engine_vs_study_replay_max_gap", "lookahead_selftest", "beta_base_on_index", "rank_corr",
                                 "headline_keys", "runtime_seconds")}
    print(json.dumps(brief, indent=1, ensure_ascii=False, default=float))
    return out


if __name__ == "__main__":
    t0 = time.time()
    report(build(), t0)
