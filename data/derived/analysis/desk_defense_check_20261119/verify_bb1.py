"""verify_bb1: attack the BB-1 override claims (crowd C4o/C4r, reentry R4) and the C1 "crowded while shallow" exit.

Run from the repo root:  py data/derived/analysis/desk_defense_check_20261119/verify_bb1.py
Reads only (common.py, ext_reentry.simulate, saved ledgers, belt/now inputs). Writes cache/verify_bb1.json and,
if the optional Yahoo cross-check succeeds, cache/verify_bb1_soxx_yf.csv.

Sections
  1 point-in-time: own rebuild of dd60 / percentile from source_flows.csv, publication timing per row,
    fetch-time evidence, revision check, BB-1 input delayed 1 and 2 sessions
  2 own re-implementation of C4o / C4r (own gate, own state machine, own NAV) against the saved ledgers
  3 episode accounting, price-only half, random crowding half (500 draws), threshold neighbours
  4 the only out-of-sample days (2026-11-11 .. 11-17)
  5 C1 triggers, forward returns, H1 false alarms, delay, circular-shift placebo (1,000 draws)
  6 auxiliary year 2025 on SOXX itself
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import study  # noqa: E402

CROWD = common.CACHE / "crowd"
ZONE = HERE.parent / "bottom_zone_20261111"
NOW = ZONE / "belt" / "now"
RESULT = common.CACHE / "verify_bb1.json"
SEED = 20261119
EPISODES = ["2026-07-20", "2026-07-28", "2026-08-21", "2026-09-04", "2026-09-08", "2026-09-17", "2026-11-05"]
OUT = {}


def show(title, obj=None):
    print("\n=== " + title)
    if obj is not None:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 400,
                                   "display.float_format", lambda v: f"{v:.4f}"):
                print(obj.to_string())
        else:
            print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


# ------------------------------------------------------------------ own signal rebuild
def pct_rank(v, win=250, minn=120):
    out = np.full(len(v), np.nan)
    for i, x in enumerate(v):
        if np.isfinite(x):
            w = v[max(0, i - win + 1): i + 1]
            w = w[np.isfinite(w)]
            if len(w) >= minn:
                out[i] = 100.0 * ((w < x).sum() + .5 * (w == x).sum()) / len(w)
    return out


def rebuild_series():
    """Entry-session aligned dd60 and percentile for every SOXX session since 2022-06, from the raw flow table."""
    src = pd.read_csv(CROWD / "source_flows.csv", parse_dates=["date", "settle_date", "known_date", "entry"]).set_index("date")
    px = pd.read_csv(NOW / "data" / "prices.csv", parse_dates=["date"])
    soxx = px[px.ticker == "SOXX"].set_index("date").sort_index()
    s = src.reindex(soxx.index)
    t5 = s.SOXL_buy.rolling(5, min_periods=5).sum() / s.total_buy.rolling(5, min_periods=5).sum().replace(0, np.nan)
    p = pd.Series(pct_rank(t5.to_numpy()), index=soxx.index)
    sel = src.reset_index().sort_values(["entry", "date"]).drop_duplicates("entry", keep="last").set_index("entry")
    ser = pd.DataFrame(index=soxx.index)
    ser["open"], ser["close"] = soxx.open, soxx.close
    ser["dd"] = (soxx.close / soxx.close.rolling(60, min_periods=60).max() - 1).shift(1)
    ser["source"] = sel.date.reindex(ser.index)
    ser["seibro"] = sel.settle_date.reindex(ser.index)
    ser["known_date"] = sel.known_date.reindex(ser.index)
    pub = sel.published.reindex(ser.index)
    ser["pct"] = [p.get(d, np.nan) if (isinstance(d, pd.Timestamp) and ok is True) else np.nan
                  for d, ok in zip(ser.source, pub)]
    ser["known"] = ser.pct.notna()
    return src, ser


# ------------------------------------------------------------------ state machines (own)
def carry(flag, known):
    out, last = np.zeros(len(flag), bool), False
    for j in range(len(flag)):
        if known[j]:
            last = bool(flag[j])
        out[j] = last
    return out


def c4o(gate, bb):
    return gate | bb


def c4r(gate, bb):
    inv, state = np.zeros(len(gate), bool), True
    for j in range(len(gate)):
        if gate[j] or bb[j]:
            state = True
        elif j == 0 or gate[j - 1]:
            state = False
        inv[j] = state
    return inv


def exit_reenter(trig, back):
    inv, state = np.zeros(len(trig), bool), True
    for j in range(len(trig)):
        if trig[j]:
            state = False
        elif not state and back[j]:
            state = True
        inv[j] = state
    return inv


def recent(events, m):
    out = np.zeros(len(events), bool)
    for j in range(len(events)):
        out[j] = events[max(0, j - m + 1): j + 1].any()
    return out


def lag(a, d, fill):
    return a.copy() if d == 0 else np.r_[np.full(d, fill, dtype=a.dtype), a[:-d]]


def c1_rules(pct, known, dd, g10, g20):
    """The 54 C1 exposure paths (True = invested); cash and half share them."""
    safe = np.where(known, pct, -1.0)
    bb = carry(known & (safe >= 80) & (dd <= -.20), known)
    out = {}
    for p in (90, 95, 98):
        for d in (5, 8):
            raw = known & (safe >= p) & (dd <= -d / 100) & (dd > -.20)
            trig = carry(raw, known)
            backs = {"A50": carry(safe < 50, known), "A80": carry(safe < 80, known), "B": bb,
                     "C10": g10, "C20": g20, "E": bb | g20}
            for name, back in backs.items():
                out[f"C1_p{p}_d{d}_{name}"] = exit_reenter(trig, back)
            for m in (5, 10, 20):
                out[f"C1_p{p}_d{d}_D{m}"] = ~recent(raw, m)
    return out


def runs(mask):
    out, j, n = [], 0, len(mask)
    while j < n:
        if mask[j]:
            k = j
            while k < n and mask[k]:
                k += 1
            out.append((j, k))
            j = k
        else:
            j += 1
    return out


# ------------------------------------------------------------------ NAV on a return stream (own)
class Nav:
    """Close NAV of an exposure path on one asset given its open and close marks.

    frac[j] is held from the open of session j to the open of session j+1 (sold at that open if frac drops).
    For the 7-stock book the marks are the base basket's own open/close NAV, which is exact for 0/1 paths
    with zero commission because the basket rebalances inside itself.
    """

    def __init__(self, days, v_open, v_close, expo=None):
        self.days = np.array(days)
        self.n = len(days)
        self.oo = v_open[1:] / v_open[:-1] - 1
        self.oc = v_close / v_open - 1
        self.log = np.r_[np.log1p(self.oo), np.log1p(self.oc[-1])]
        self.base_close = v_close / v_open[0]
        self.expo = np.ones(self.n) if expo is None else expo

    def close(self, frac):
        f = np.atleast_2d(np.asarray(frac, dtype=float))
        nav_open = np.c_[np.ones(len(f)), np.cumprod(1 + f[:, :-1] * self.oo, axis=1)]
        return nav_open * (1 + f * self.oc)

    def ix(self, a, b):
        return np.flatnonzero((self.days >= a) & (self.days <= b))

    def win(self, frac, a="0000", b="9999"):
        """(return, max drawdown) inside [a, b], anchored on the close before a."""
        c = self.close(frac)
        ix = self.ix(a, b)
        before = c[:, ix[0] - 1] if ix[0] else np.ones(len(c))
        path = np.c_[before, c[:, ix]]
        ret = path[:, -1] / before - 1
        mdd = (path / np.maximum.accumulate(path, axis=1) - 1).min(axis=1)
        return ret, mdd

    def matched(self, frac, a, b):
        """Exposure-matched control: hold the window's mean exposure share of the base every day."""
        f = np.atleast_2d(np.asarray(frac, dtype=float))
        ix = self.ix(a, b)
        normal = self.expo[ix].mean()
        e = (f[:, ix] * self.expo[ix]).mean(axis=1)
        path = np.r_[self.base_close[ix[0] - 1] if ix[0] else 1.0, self.base_close[ix]]
        daily = path[1:] / path[:-1] - 1
        return np.prod(1 + (e / normal)[:, None] * daily, axis=1) - 1


H1 = ("2026-01-01", "2026-06-30")
H2 = ("2026-07-01", "2026-12-31")
JUL = ("2026-06-23", "2026-07-29")
RALLY = ("2026-07-30", "2026-09-22")
OCT = ("2026-10-13", "2026-11-17")


def row(nav, frac):
    out = {}
    for name, (a, b) in (("full", ("0", "9")), ("h1", H1), ("h2", H2), ("jul", JUL), ("rally", RALLY), ("octnov", OCT)):
        r, m = nav.win(frac, a, b)
        out[name], out[name + "_mdd"] = float(r[0]), float(m[0])
    out["expo"] = float(np.mean(frac))
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    ctx = common.context()
    days, n = list(ctx.days), len(ctx.days)
    d_arr = np.array(days)

    # ---------- own shadow basket loop (independent of study.replay)
    shares, cash = np.zeros(ctx.opn.shape[1]), 1.0
    v_open, v_close, expo = np.zeros(n), np.zeros(n), np.zeros(n)
    for j in range(n):
        o, c = np.nan_to_num(ctx.opn[j]), np.nan_to_num(ctx.close[j])
        v_open[j] = cash + shares @ o
        if j in ctx.weekly:
            w = ctx.weekly[j]
            shares = np.divide(v_open[j] * w, ctx.opn[j], out=np.zeros_like(w), where=w > 0)
            cash = v_open[j] * (1 - w.sum())
        v_close[j] = cash + shares @ c
        expo[j] = 1 - cash / v_open[j]
    base_ledger = study.replay(days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, ctx.gates["none"])
    nav = Nav(days, v_open, v_close, expo)
    ones = np.ones(n)
    OUT["engine"] = dict(
        own_basket_vs_codex_base_usd_max_rel=float(np.max(np.abs(v_close / base_ledger.usd.to_numpy() - 1))),
        own_basket_vs_baseline_open_max_rel=float(np.max(np.abs(v_open / np.asarray(ctx.baseline.value) - 1))),
        base=row(nav, ones), sessions=n, first=days[0], last=days[-1])
    show("engine / base (own basket loop)", OUT["engine"])

    # ---------- own gates
    soxx_c = ctx.book.cls["SOXX"]
    pos = [list(soxx_c.index).index(d) for d in days]
    gate = {}
    for m in (10, 20, 50):
        above = (soxx_c >= soxx_c.rolling(m).mean()).to_numpy()
        gate[m] = np.array([bool(above[p - 1]) for p in pos])
        assert np.array_equal(gate[m], ctx.gates[f"SOXX{m}r1"]), m
    soxx_o = ctx.book.opn["SOXX"].reindex(days).to_numpy()
    soxx_cl = soxx_c.reindex(days).to_numpy()
    snav = Nav(days, soxx_o, soxx_cl)

    # ================================================================= 1 point in time
    src, ser = rebuild_series()
    cal = pd.DataFrame(json.loads((CROWD / "bb1_calendar.json").read_text(encoding="utf-8"))).set_index("day")
    mine = ser.copy()
    mine.index = [str(d.date()) for d in mine.index]
    m26 = mine.reindex(days)
    calp = cal.percentile.astype(float).reindex(days).to_numpy()
    pct, known, dd = m26.pct.to_numpy(), m26.known.to_numpy(bool), m26.dd.to_numpy()
    cal_known = np.isfinite(calp)
    book_dd = (soxx_c / soxx_c.rolling(60, min_periods=60).max() - 1).shift().reindex(days).to_numpy()
    cal_on = np.array([v is True for v in cal["on"].reindex(days)])
    own_on = known & (pct >= 80) & (dd <= -.20)
    kst, ny = ZoneInfo("Asia/Seoul"), ZoneInfo("America/New_York")
    margins, lags, same_day = [], [], 0
    all_sessions = list(ser.index)
    for d in days:
        r = ser.loc[pd.Timestamp(d)]
        if not r.known:
            continue
        k = r.known_date
        pub = datetime(k.year, k.month, k.day, 9, 0, tzinfo=kst)
        opn = datetime(int(d[:4]), int(d[5:7]), int(d[8:]), 9, 30, tzinfo=ny)
        margins.append((opn - pub).total_seconds() / 3600)
        lags.append(all_sessions.index(pd.Timestamp(d)) - all_sessions.index(r.source))
        same_day += int(k == pd.Timestamp(d))
    # own recomputation of known_date / entry from the SEIBRO table date with exchange_calendars
    import exchange_calendars as xc
    ksd = pd.DatetimeIndex(xc.get_calendar("XKRX", start="2022-12-01", end="2027-03-01").sessions)
    ksd = ksd[ksd != pd.Timestamp("2026-06-03")]
    nyse = pd.DatetimeIndex(xc.get_calendar("XNYS", start="2022-06-01", end="2027-03-01").sessions)
    s26 = src.loc["2025-12-20":]
    bad_known = bad_entry = 0
    for t, r in s26.iterrows():
        k = ksd[ksd.searchsorted(r.settle_date, side="right")]
        e = nyse[nyse.searchsorted(k, side="left")]
        bad_known += int(k != r.known_date)
        bad_entry += int(e != r.entry)
    gap = [int(np.busday_count(t.date(), r.settle_date.date())) for t, r in s26.iterrows()]
    # fetch-time evidence: when did a non-empty SEIBRO answer for table date d exist on disk
    fetch = []
    for f in sorted((NOW / "data" / "cache").glob("daily_1_202611*.json")):
        d = f.stem.split("_")[-1]
        if d >= "20261109":
            rows_n = json.loads(f.read_text(encoding="utf-8"))["n"]
            stamp = datetime.fromtimestamp(f.stat().st_mtime, tz=kst)
            fetch.append(dict(seibro_table_date=d, rows=rows_n, fetched_kst=stamp.strftime("%Y-%m-%d %H:%M"),
                              fetched_et=stamp.astimezone(ny).strftime("%Y-%m-%d %H:%M")))
    old_cache = sorted((NOW / "data" / "cache").glob("daily_1_2026*.json"))
    stamps = [datetime.fromtimestamp(f.stat().st_mtime, tz=kst).strftime("%Y-%m-%d") for f in old_cache]
    frozen = pd.read_csv(ZONE / "data" / "source_flows.csv", parse_dates=["date"]).set_index("date")
    both = frozen.index.intersection(src.index)
    cols = ["SOXL_buy", "total_buy"]
    rev = (frozen.loc[both, cols] - src.loc[both, cols]).abs()
    near = [(d, round(float(a), 4), float(b)) for d, a, b, o in zip(days, dd, pct, own_on)
            if o and (a > -.21 or b < 85)]
    basis = cal.basis.reindex(days).value_counts().to_dict()
    OUT["pit"] = dict(
        calendar_rows_2026=n, basis=basis,
        own_percentile_vs_calendar_max_abs=float(np.nanmax(np.abs(pct[cal_known] - calp[cal_known]))),
        known_flags_identical=bool(np.array_equal(known, cal_known)),
        own_dd_vs_calendar_max_abs=float(np.max(np.abs(dd - cal.dd60.astype(float).reindex(days).to_numpy()))),
        book_dd_vs_calendar_max_abs=float(np.max(np.abs(book_dd - dd))),
        bb1_on_identical=bool(np.array_equal(own_on, cal_on)), bb1_on_days=int(own_on.sum()),
        unknown_sessions=[d for d, k in zip(days, known) if not k],
        source_to_entry_session_lag=pd.Series(lags).value_counts().sort_index().to_dict(),
        hours_from_assumed_0900KST_publication_to_US_open=dict(min=float(min(margins)), median=float(np.median(margins)),
                                                              max=float(max(margins))),
        rows_where_known_date_equals_session_date=same_day, rows_with_percentile=len(margins),
        known_date_recomputed_mismatch=bad_known, entry_recomputed_mismatch=bad_entry,
        weekdays_trade_to_seibro_table=pd.Series(gap).value_counts().sort_index().to_dict(),
        known_date_monotone=bool(src.known_date.dropna().is_monotonic_increasing),
        fetch_evidence=fetch,
        raw_2026_answers_on_disk=len(old_cache), raw_2026_answers_fetched_on=pd.Series(stamps).value_counts().to_dict(),
        frozen_vs_now_rows=len(both), frozen_vs_now_max_abs_usd=float(rev.max().max()),
        frozen_vs_now_published_flag_changes=int((frozen.loc[both, "published"] != src.loc[both, "published"]).sum()),
        bb1_days_close_to_a_threshold=near)
    show("1 point in time", OUT["pit"])

    # BB-1 states used by the two builders
    bb_carry = carry(own_on, known)            # ext_crowd convention: unknown row keeps the last known BB-1 state
    bb_strict = own_on.copy()                  # ext_reentry convention: unknown row = off
    OUT["pit"]["carry_adds_days"] = [d for d, a, b in zip(days, bb_carry, bb_strict) if a and not b]
    OUT["pit"]["carry_added_day_dd60"] = [round(float(dd[days.index(d)]), 4) for d in OUT["pit"]["carry_adds_days"]]

    def bb_variant(delay=0, pct_only=False, strict=False, p_thr=80, d_thr=.20):
        if pct_only:
            p, k = lag(pct, delay, np.nan), lag(known, delay, False)
            on = k & (np.where(k, p, -1) >= p_thr) & (dd <= -d_thr)
            return on if strict else carry(on, k)
        on = known & (np.where(known, pct, -1) >= p_thr) & (dd <= -d_thr)
        state = on if strict else carry(on, known)
        return lag(state, delay, False)

    def rules_for(bb):
        return {f"{kind}_SOXX{m}": fn(gate[m], bb) for m in (10, 20, 50) for kind, fn in (("C4o", c4o), ("C4r", c4r))}

    table = {"base": row(nav, ones)}
    for m in (10, 20, 50):
        table[f"gate_SOXX{m}"] = row(nav, gate[m])
    variants = {"real": bb_variant(), "delay1_whole_signal": bb_variant(1), "delay2_whole_signal": bb_variant(2),
                "delay1_percentile_only": bb_variant(1, True), "delay2_percentile_only": bb_variant(2, True),
                "unknown_rows_off": bb_variant(strict=True), "price_half_only": dd <= -.20}
    for name, bb in variants.items():
        for key, inv in rules_for(bb).items():
            table[f"{key}|{name}"] = row(nav, inv)
    for d_ in (1, 2):                      # every decision (gate and override) filled d_ sessions late
        for m in (10, 20, 50):
            table[f"gate_SOXX{m}|all_fills_delayed{d_}"] = row(nav, lag(gate[m], d_, True))
        for key, inv in rules_for(bb_carry).items():
            table[f"{key}|all_fills_delayed{d_}"] = row(nav, lag(inv, d_, True))
    tab = pd.DataFrame(table).T
    OUT["rules"] = tab.round(5).to_dict("index")
    late = pd.DataFrame({f"{c}_{w}": {k: table[f"{k}|{c}"][w] for k in [f"gate_SOXX{m}" for m in (10, 20, 50)] + list(rules_for(bb_carry))}
                         for c in ("all_fills_delayed1", "all_fills_delayed2") for w in ("full", "h2", "full_mdd")})
    late.insert(0, "undelayed_full", [table[k]["full"] if k.startswith("gate") else table[k + "|real"]["full"] for k in late.index])
    late.insert(1, "undelayed_h2", [table[k]["h2"] if k.startswith("gate") else table[k + "|real"]["h2"] for k in late.index])
    piv = pd.DataFrame({name: {key: table[f"{key}|{name}"]["full"] for key in rules_for(bb_carry)} for name in variants})
    piv_h2 = pd.DataFrame({name: {key: table[f"{key}|{name}"]["h2"] for key in rules_for(bb_carry)} for name in variants})
    piv_mdd = pd.DataFrame({name: {key: table[f"{key}|{name}"]["full_mdd"] for key in rules_for(bb_carry)} for name in variants})
    show("base and plain gates", tab.loc[["base", "gate_SOXX10", "gate_SOXX20", "gate_SOXX50"]])
    show("1/3 full-year USD by BB-1 input variant", piv)
    show("1/3 H2 USD by BB-1 input variant", piv_h2)
    show("1/3 full-year USD MDD by BB-1 input variant", piv_mdd)
    show("1 gate and override both filled one / two sessions late", late)

    # ================================================================= 2 own loop vs saved ledgers
    saved = pd.read_parquet(common.CACHE / "crowd_daily.parquet")
    saved = saved[saved.strategy.str.startswith("C4")].pivot(index="date", columns="strategy", values="usd").reindex(days)
    saved_x = pd.read_parquet(common.CACHE / "crowd_daily.parquet")
    saved_x = saved_x[saved_x.strategy.str.startswith("C4")].pivot(index="date", columns="strategy", values="exposure").reindex(days)
    cmp = {}
    for key, inv in rules_for(bb_carry).items():
        mine_c = nav.close(inv)[0]
        ref = saved[key + "_cash"].to_numpy()
        cmp[key] = dict(max_rel_diff_usd=float(np.max(np.abs(mine_c / ref - 1))),
                        in_out_days_differ=int(((saved_x[key + "_cash"].to_numpy() > 0) != (inv & (expo > 0))).sum()),
                        own_full=float(mine_c[-1] - 1), saved_full=float(ref[-1] - 1),
                        own_mdd=float(np.min(mine_c / np.maximum.accumulate(np.r_[1., mine_c])[1:] - 1)))
    OUT["reimplementation"] = cmp
    show("2 own C4 vs saved crowd ledgers", pd.DataFrame(cmp).T)

    # ================================================================= 3 in-sample accounting
    ep_start = [days.index(d) for d in EPISODES]
    bb_runs = [(days[a], days[b - 1], b - a) for a, b in runs(bb_carry)]
    OUT["bb1_runs_carried"] = bb_runs
    slog = np.r_[np.log(soxx_o[1:] / soxx_o[:-1]), np.log(soxx_cl[-1] / soxx_o[-1])]
    acct = []
    for key, inv in rules_for(bb_carry).items():
        g = gate[int(key.split("SOXX")[1])]
        diff = inv & ~g
        assert not (g & ~inv).any()
        total = float(nav.log[diff].sum())
        rule_full, gate_full = nav.close(inv)[0, -1], nav.close(g)[0, -1]
        assert abs(np.log(rule_full / gate_full) - total) < 1e-9
        for e, a in enumerate(ep_start):
            b = ep_start[e + 1] if e + 1 < len(ep_start) else n
            ix = np.flatnonzero(diff[a:b]) + a
            gain = float(nav.log[ix].sum())
            acct.append(dict(rule=key, episode=EPISODES[e][5:], bb1_on_sessions=int(bb_carry[a:b].sum()),
                             extra_sessions_held=len(ix),
                             first=days[ix[0]][5:] if len(ix) else "", last=days[ix[-1]][5:] if len(ix) else "",
                             base_move_while_held=float(np.expm1(gain)), soxx_move_same_sessions=float(np.expm1(slog[ix].sum())),
                             share_of_log_gain=gain / total if total else np.nan,
                             full_usd_without_this_episode=float(rule_full / np.exp(gain) - 1)))
        before = np.flatnonzero(diff[:ep_start[0]])
        acct.append(dict(rule=key, episode="total", bb1_on_sessions=int(bb_carry.sum()), extra_sessions_held=int(diff.sum()),
                         first="", last="", base_move_while_held=float(np.expm1(total)),
                         soxx_move_same_sessions=float(np.expm1(slog[diff].sum())), share_of_log_gain=1.0,
                         full_usd_without_this_episode=float(gate_full - 1)))
        assert len(before) == 0
    acct = pd.DataFrame(acct)
    OUT["episodes"] = acct.round(5).to_dict("records")
    show("3 episode accounting (gain over the plain gate)", acct[acct.rule.str.contains("SOXX10")])
    show("3 episode accounting SOXX20 / SOXX50", acct[~acct.rule.str.contains("SOXX10")])

    held = c4o(gate[10], bb_carry) & ~gate[10]
    one = pd.DataFrame({"open_to_next_open_base": np.expm1(nav.log[held]), "soxx": np.expm1(slog[held]),
                        "dd60": dd[held], "pct": pct[held]}, index=d_arr[held])
    order = np.sort(nav.log[held])[::-1]
    OUT["c4o_soxx10_extra_sessions"] = dict(
        rows=one.round(4).reset_index().to_dict("records"), total_log=float(order.sum()),
        top3_share_of_log_gain=float(order[:3].sum() / order.sum()), top5_share=float(order[:5].sum() / order.sum()),
        positive_sessions=int((order > 0).sum()), sessions=int(len(order)),
        gain_without_best_session=float(np.expm1(order[1:].sum())), gain_without_best3=float(np.expm1(order[3:].sum())))
    show("3 C4o_SOXX10: every extra session held (open to next open)", one)
    show("3 C4o_SOXX10: concentration", {k: v for k, v in OUT["c4o_soxx10_extra_sessions"].items() if k != "rows"})
    drop = (c4o(gate[10], dd <= -.20) & ~c4o(gate[10], bb_carry))
    cut = pd.DataFrame({"open_to_next_open_base": np.expm1(nav.log[drop]), "soxx": np.expm1(slog[drop]),
                        "dd60": dd[drop], "pct": pct[drop]}, index=d_arr[drop])
    OUT["crowding_half_removed_sessions"] = dict(rows=cut.round(4).reset_index().to_dict("records"),
                                                 base_move_avoided=float(np.expm1(nav.log[drop].sum())),
                                                 after_registration_2026_11_12=int((d_arr[drop] >= "2026-11-12").sum()),
                                                 bb1_on_days_after_registration=int(bb_carry[d_arr >= "2026-11-12"].sum()))
    show("3 sessions the crowding half removes from the price-only override (SOXX10)", cut)
    show("3 crowding half: what it avoided", {k: v for k, v in OUT["crowding_half_removed_sessions"].items() if k != "rows"})

    # what BB-1 looked like inside each episode
    ep_rows = []
    for e, a in enumerate(ep_start):
        b = ep_start[e + 1] if e + 1 < len(ep_start) else n
        on = np.flatnonzero(bb_carry[a:b]) + a
        j0, j1 = on[0], on[-1]
        ep_rows.append(dict(episode=EPISODES[e][5:], on_days=len(on), last_on=days[j1][5:],
                            dd60_first=float(dd[j0]), pct_first=float(pct[j0]) if known[j0] else np.nan,
                            gate10_on_days_in=int(gate[10][on].sum()), gate20_on_days_in=int(gate[20][on].sum()),
                            gate50_on_days_in=int(gate[50][on].sum()),
                            base_fwd5=float(v_open[min(j0 + 5, n - 1)] / v_open[j0] - 1),
                            base_fwd10=float(v_open[min(j0 + 10, n - 1)] / v_open[j0] - 1),
                            soxx_fwd5=float(soxx_o[min(j0 + 5, n - 1)] / soxx_o[j0] - 1),
                            soxx_fwd10=float(soxx_o[min(j0 + 10, n - 1)] / soxx_o[j0] - 1)))
    OUT["episode_context"] = ep_rows
    show("3 BB-1 episodes: context (forward from the first on-session open)", pd.DataFrame(ep_rows))

    # threshold neighbours of the real definition
    grid = {}
    for d_thr in (.15, .20, .25):
        for p_thr in (0, 60, 70, 80, 90):
            bb = bb_variant(p_thr=p_thr, d_thr=d_thr) if p_thr else dd <= -d_thr
            r = rules_for(bb)
            grid[f"dd{int(d_thr * 100)}_p{p_thr or 'none'}"] = dict(
                on_days=int(bb.sum()),
                **{f"{k}_{w}": float(nav.win(r[f"{k}_SOXX10"], *win)[0][0])
                   for k in ("C4o", "C4r") for w, win in (("full", ("0", "9")), ("h2", H2))},
                C4o_SOXX20_full=float(nav.win(r["C4o_SOXX20"])[0][0]), C4o_SOXX50_full=float(nav.win(r["C4o_SOXX50"])[0][0]))
    OUT["threshold_grid"] = grid
    show("3 threshold neighbours (p none = price half only)", pd.DataFrame(grid).T)

    # random crowding half
    z = pd.Series(ser.pct.dropna().to_numpy())
    from scipy.stats import norm
    score = norm.ppf(np.clip(z / 100, .002, .998))
    rho = float(np.corrcoef(score[1:], score[:-1])[0, 1])
    long = ser.loc["2024-01-02":]
    long_flag = (long.pct >= 80).fillna(False).to_numpy()
    real_freq = dict(pct_ge80_share_2026=float((pct[known] >= 80).mean()),
                     pct_ge80_share_on_dd20_days_2026=float((pct[known & (dd <= -.20)] >= 80).mean()),
                     dd20_days_2026=int((dd <= -.20).sum()), dd20_known_days=int((known & (dd <= -.20)).sum()),
                     pct_ge80_share_2024_2026=float(long_flag.mean()), lag1_autocorr_normal_scores=rho)
    draws = 500
    ar = np.zeros((draws, n + 60))
    ar[:, 0] = rng.standard_normal(draws)
    eps = rng.standard_normal((draws, n + 60))
    for t in range(1, n + 60):
        ar[:, t] = rho * ar[:, t - 1] + np.sqrt(1 - rho ** 2) * eps[:, t]
    rand_ar = norm.cdf(ar[:, 60:]) >= .80
    offs = rng.integers(1, len(long_flag), draws)
    rand_shift = np.array([np.roll(long_flag, -k)[-n:] for k in offs])
    price = dd <= -.20
    freq26 = real_freq["pct_ge80_share_2026"]
    rand_ar_matched = norm.cdf(ar[:, 60:]) >= 1 - freq26
    dd_ix = np.flatnonzero(price & known)
    k_on = int(bb_strict[dd_ix].sum())
    rand_subset = np.zeros((draws, n), bool)
    for i in range(draws):
        rand_subset[i, rng.choice(dd_ix, size=k_on, replace=False)] = True
    all_ix = np.flatnonzero(price)
    pattern = bb_strict[all_ix]
    rotations = np.zeros((len(all_ix) - 1, n), bool)
    for k in range(1, len(all_ix)):
        rotations[k - 1, all_ix] = np.roll(pattern, k)
    rnd = {}
    for label, flags in (("AR1_20pct", rand_ar), ("AR1_matched_2026_freq", rand_ar_matched),
                         ("random_22_of_34_dd20_days", rand_subset), ("rotate_real_pattern_over_dd20_days", rotations),
                         ("circular_shift_real_flag_2024_2026", rand_shift)):
        draws = len(flags)
        for m in (10, 20, 50):
            for kind, fn in (("C4o", c4o), ("C4r", c4r)):
                paths = np.array([fn(gate[m], price & flags[i]) for i in range(draws)])
                full = nav.win(paths)[0]
                h2r, _ = nav.win(paths, *H2)
                mdd = nav.win(paths)[1]
                key = f"{kind}_SOXX{m}"
                real_full, real_h2 = table[f"{key}|unknown_rows_off"]["full"], table[f"{key}|unknown_rows_off"]["h2"]
                rnd[f"{key}|{label}"] = dict(
                    real_full=real_full, plain_gate_full=table[f"gate_SOXX{m}"]["full"],
                    price_only_full=table[f"{key}|price_half_only"]["full"],
                    draws_full_p05=float(np.quantile(full, .05)), draws_full_median=float(np.median(full)),
                    draws_full_p95=float(np.quantile(full, .95)), share_draws_ge_real_full=float((full >= real_full - 1e-12).mean()),
                    real_h2=real_h2, draws_h2_median=float(np.median(h2r)), share_draws_ge_real_h2=float((h2r >= real_h2 - 1e-12).mean()),
                    draws_mdd_median=float(np.median(mdd)), random_on_days_mean=float((price & flags).sum(axis=1).mean()),
                    n_draws=int(draws))
    draws = 500
    OUT["random_crowding"] = dict(freq=real_freq, draws=draws, rotations=len(rotations), results=rnd)
    show("3 random crowding half: frequencies", real_freq)
    show("3 random crowding half (real = unknown rows off, like the random series)", pd.DataFrame(rnd).T)

    # ================================================================= 4 out of sample 11-11 .. 11-17
    oos = {}
    paths = {"base": ones.astype(bool), **{f"gate_SOXX{m}": gate[m] for m in (10, 20, 50)}, **rules_for(bb_carry)}
    j0 = days.index("2026-11-05")
    daily = pd.DataFrame({"dd60": dd[j0:], "pct": pct[j0:], "bb1": bb_carry[j0:].astype(int),
                          **{f"g{m}": gate[m][j0:].astype(int) for m in (10, 20, 50)},
                          **{k: v[j0:].astype(int) for k, v in rules_for(bb_carry).items() if "SOXX10" in k or "SOXX20" in k},
                          "base_day": v_close[j0:] / np.r_[v_close[j0 - 1], v_close[j0:-1]] - 1,
                          "soxx_day": soxx_cl[j0:] / np.r_[soxx_cl[j0 - 1], soxx_cl[j0:-1]] - 1}, index=days[j0:])
    show("4 day by day from 11-05 (1 = invested at that open)", daily)
    for name, inv in paths.items():
        oos[name] = {f"{a[5:]}..{b[5:]}": float(nav.win(inv, a, b)[0][0])
                     for a, b in (("2026-11-05", "2026-11-10"), ("2026-11-11", "2026-11-17"), ("2026-11-12", "2026-11-17"),
                                  ("2026-11-05", "2026-11-17"))}
        oos[name]["invested_sessions_11-11..11-17"] = int(inv[nav.ix("2026-11-11", "2026-11-17")].sum())
    oos["SOXX_itself"] = {f"{a[5:]}..{b[5:]}": float(snav.win(ones, a, b)[0][0])
                          for a, b in (("2026-11-05", "2026-11-10"), ("2026-11-11", "2026-11-17"), ("2026-11-12", "2026-11-17"),
                                       ("2026-11-05", "2026-11-17"))}
    OUT["oos_nov"] = dict(daily=daily.round(4).reset_index().to_dict("records"), windows=oos)
    show("4 window returns USD", pd.DataFrame(oos).T)

    # ================================================================= reentry R4 (same BB-1 input)
    import ext_reentry as er
    soxx_s, breadth, er_bb1, _ = er.inputs(ctx)
    sig = er.build_signals(soxx_s, ctx.start, n, base_ledger.usd.to_numpy(), breadth, er_bb1)
    assert np.array_equal(er_bb1, bb_strict)
    det = pd.read_csv(common.CACHE / "reentry_detail.csv").set_index("strategy")
    by_sub = det.groupby("subfamily").h2_excess_vs_control.median().sort_values(ascending=False)
    r4 = {}

    def r4_paths(bb):
        s2 = dict(sig)
        s2["bb1"] = np.asarray(bb, bool)
        return {f"{e}_R4_{mode}": er.simulate(s2, e, "R4", {}, mode)[0] for e in er.EXITS for mode in ("pure", "orgate")}

    def r4_stats(bb):
        p = r4_paths(bb)
        f = np.array(list(p.values()))
        h2r = nav.win(f, *H2)[0]
        return dict(zip(p, h2r)), dict(zip(p, h2r - nav.matched(f, *H2))), dict(zip(p, nav.win(f)[0]))

    real_h2, real_ex, real_full = r4_stats(bb_strict)
    check = max(abs(real_h2[k] - det.loc[k, "h2_usd"]) for k in real_h2)
    check_ex = max(abs(real_ex[k] - det.loc[k, "h2_excess_vs_control"]) for k in real_ex)
    for name, bb in (("real", bb_strict), ("delay1", lag(bb_strict, 1, False)), ("delay2", lag(bb_strict, 2, False)),
                     ("price_half_only", price)):
        h, x, f = r4_stats(bb)
        for mode in ("pure", "orgate"):
            ks = [k for k in h if k.endswith(mode)]
            r4[f"{name}|{mode}"] = dict(median_h2=float(np.median([h[k] for k in ks])),
                                        median_h2_excess=float(np.median([x[k] for k in ks])),
                                        median_full=float(np.median([f[k] for k in ks])),
                                        E1_h2=float(h[f"E1_R4_{mode}"]), E1_full=float(f[f"E1_R4_{mode}"]))
    med = {"pure": [], "orgate": []}
    for i in range(draws):
        h, x, f = r4_stats(price & rand_ar[i])
        for mode in med:
            med[mode].append(np.median([x[k] for k in x if k.endswith(mode)]))
    for mode in med:
        a = np.array(med[mode])
        r4[f"random_AR1_500|{mode}"] = dict(median_h2_excess_p05=float(np.quantile(a, .05)), median_h2_excess_median=float(np.median(a)),
                                            median_h2_excess_p95=float(np.quantile(a, .95)),
                                            share_draws_ge_real=float((a >= r4[f"real|{mode}"]["median_h2_excess"] - 1e-12).mean()))
    OUT["reentry_R4"] = dict(own_vs_saved_h2_max_abs=float(check), own_vs_saved_excess_max_abs=float(check_ex),
                             subfamily_median_h2_excess_top=by_sub.head(6).round(4).to_dict(),
                             r4_rows={k: {c: float(det.loc[k, c]) for c in ("full_usd", "h1_usd", "h2_usd", "h2_expo", "h1_expo",
                                                                          "h2_excess_vs_control", "h2_usd_delay1", "random_pct_all")}
                                      for k in det.index if "_R4_" in k},
                             variants=r4, base_full=float(table["base"]["full"]), base_h1=float(table["base"]["h1"]))
    show("reentry R4: saved rows", pd.DataFrame(OUT["reentry_R4"]["r4_rows"]).T)
    show("reentry R4: subfamily median H2 excess (top 6), own-vs-saved check",
         dict(top=OUT["reentry_R4"]["subfamily_median_h2_excess_top"], h2_diff=check, excess_diff=check_ex))
    show("reentry R4 variants (6 exits each)", pd.DataFrame(r4).T)

    # ================================================================= 5 C1
    c1 = c1_rules(pct, known, dd, gate[10], gate[20])
    keys = list(c1)
    inv_m = np.array([c1[k] for k in keys])
    sv = pd.read_parquet(common.CACHE / "crowd_daily.parquet")
    sv_usd = sv.pivot(index="date", columns="strategy", values="usd").reindex(days)
    full = pd.read_csv(CROWD / "crowd_full_stats.csv", index_col=0)
    cash_c = nav.close(inv_m)
    half_c = nav.close(np.where(inv_m, 1.0, .5))
    c1chk = dict(cash_max_rel_diff_vs_saved=float(max(np.max(np.abs(cash_c[i] / sv_usd[k + "_cash"].to_numpy() - 1)) for i, k in enumerate(keys))),
                 half_max_rel_diff_vs_saved_fast_approx=float(max(np.max(np.abs(half_c[i] / sv_usd[k + "_half"].to_numpy() - 1)) for i, k in enumerate(keys))))
    saved_c1 = full[full.subfamily == "C1"]
    claim = {}
    for name, g in (("cash", saved_c1[saved_c1["size"] == "cash"]), ("half", saved_c1[saved_c1["size"] == "half"]), ("all108", saved_c1)):
        claim[name] = dict(n=len(g), median_h2_usd=float(g.h2_usd.median()),
                           median_h2_excess=float((g.h2_usd - g.matched_h2_usd).median()),
                           median_h1_usd=float(g.h1_usd.median()), min_h1_usd=float(g.h1_usd.min()), max_h1_usd=float(g.h1_usd.max()),
                           median_full_usd=float(g.full_usd.median()), beat_base_full=int((g.full_usd > table["base"]["full"]).sum()),
                           beat_base_h1=int((g.h1_usd > table["base"]["h1"]).sum()),
                           beat_matched_h2=int((g.h2_usd > g.matched_h2_usd).sum()),
                           median_h2_usd_lag1_saved=float(g.h2_usd_lag1.median()))
    OUT["c1_saved"] = dict(check=c1chk, groups=claim)
    show("5 C1 saved ledgers by group (claim: median H2 -16.8%, excess +1.0pt)", pd.DataFrame(claim).T)
    show("5 own C1 vs saved", c1chk)

    def c1_family(p_, k_, d_, tag):
        r = c1_rules(p_, k_, d_, gate[10], gate[20])
        m_ = np.array([r[k] for k in keys])
        hf = np.where(m_, 1.0, .5)
        out = {}
        for size, f in (("cash", m_.astype(float)), ("half", hf)):
            h2r = nav.win(f, *H2)[0]
            ex = h2r - nav.matched(f, *H2)
            out[size] = dict(h2=h2r, ex=ex, h1=nav.win(f, *H1)[0], full=nav.win(f)[0], jul=nav.win(f, *JUL)[0], oct=nav.win(f, *OCT)[0])
        allh2 = np.r_[out["cash"]["h2"], out["half"]["h2"]]
        allex = np.r_[out["cash"]["ex"], out["half"]["ex"]]
        return dict(tag=tag, cash_median_h2=float(np.median(out["cash"]["h2"])), cash_median_excess=float(np.median(out["cash"]["ex"])),
                    cash_median_h1=float(np.median(out["cash"]["h1"])), cash_median_full=float(np.median(out["cash"]["full"])),
                    cash_median_jul=float(np.median(out["cash"]["jul"])), cash_median_oct=float(np.median(out["cash"]["oct"])),
                    all_median_h2=float(np.median(allh2)), all_median_excess=float(np.median(allex)),
                    cash_beat_matched=int((out["cash"]["ex"] > 0).sum()))

    fam = [c1_family(pct, known, dd, "real")]
    for d_ in (1, 2):
        fam.append(c1_family(lag(pct, d_, np.nan), lag(known, d_, False), lag(dd, d_, dd[0]), f"calendar delayed {d_}"))
        fam.append(c1_family(lag(pct, d_, np.nan), lag(known, d_, False), dd, f"percentile delayed {d_}, price current"))
    fam_t = pd.DataFrame(fam).set_index("tag")
    OUT["c1_delay"] = fam_t.round(5).to_dict("index")
    show("5 C1 family, real and delayed (base H2 %.4f, jul %.4f, oct %.4f)" % (table["base"]["h2"], table["base"]["jul"], table["base"]["octnov"]), fam_t)

    # trigger days
    safe = np.where(known, pct, -1.0)
    shallow = (dd <= -.05) & (dd > -.20)
    raw = known & (safe >= 90) & shallow
    trig_rows = []

    def fwd(v, j, h):
        return float(v[j + h] / v[j] - 1) if j + h < n else np.nan

    for a, b in runs(carry(raw, known)):
        fire = np.flatnonzero(raw[a:b]) + a
        trig_rows.append(dict(first=days[a], last=days[b - 1], sessions=b - a, max_pct=float(np.nanmax(pct[fire])),
                              min_dd=float(dd[fire].min()), p95=bool((safe[fire] >= 95).any()), p98=bool((safe[fire] >= 98).any()),
                              d8=bool((dd[fire] <= -.08).any()),
                              base5=fwd(v_open, a, 5), base10=fwd(v_open, a, 10), base20=fwd(v_open, a, 20),
                              soxx5=fwd(soxx_o, a, 5), soxx10=fwd(soxx_o, a, 10), soxx20=fwd(soxx_o, a, 20)))
    trig_t = pd.DataFrame(trig_rows)
    OUT["c1_trigger_runs_p90_d5"] = trig_t.round(4).to_dict("records")
    show("5 C1 trigger runs (widest: pct>=90, -20%<dd60<=-5%), forward open-to-open from the first trigger open", trig_t)

    def mean_fwd(mask, lo="0", hi="9"):
        out = {"n": int((mask & (d_arr >= lo) & (d_arr <= hi)).sum())}
        for h in (5, 10, 20):
            ix = np.flatnonzero(mask & (d_arr >= lo) & (d_arr <= hi))
            ix = ix[ix + h < n]
            out[f"base{h}"] = float(np.mean(v_open[ix + h] / v_open[ix] - 1)) if len(ix) else np.nan
            out[f"soxx{h}"] = float(np.mean(soxx_o[ix + h] / soxx_o[ix] - 1)) if len(ix) else np.nan
            out[f"n{h}"] = len(ix)
        return out

    cond = {}
    for part, (lo, hi) in (("2026", ("0", "9")), ("H1", H1), ("H2", H2)):
        cond[f"{part} all sessions"] = mean_fwd(np.ones(n, bool), lo, hi)
        cond[f"{part} shallow dd, pct>=90 (trigger days)"] = mean_fwd(raw, lo, hi)
        cond[f"{part} shallow dd, pct>=95"] = mean_fwd(known & (safe >= 95) & shallow, lo, hi)
        cond[f"{part} shallow dd, pct<90 (price half only)"] = mean_fwd(known & (safe < 90) & shallow, lo, hi)
        cond[f"{part} dd>-5%, pct>=90 (crowded at highs)"] = mean_fwd(known & (safe >= 90) & (dd > -.05), lo, hi)
    cond_t = pd.DataFrame(cond).T
    OUT["c1_forward_by_condition"] = cond_t.round(4).to_dict("index")
    show("5 mean forward returns per trigger DAY (overlapping)", cond_t)

    wins = {"jan_may": ("2026-01-01", "2026-05-31"), "pre_july 06-01..06-22": ("2026-06-01", "2026-06-22"),
            "july decline 06-23..07-29": JUL, "rally 07-30..09-22": RALLY, "pre_oct 09-23..10-12": ("2026-09-23", "2026-10-12"),
            "oct_nov decline 10-13..11-17": OCT}
    where = {}
    for p in (90, 95, 98):
        for d_ in (5, 8):
            r_ = known & (safe >= p) & (dd <= -d_ / 100) & (dd > -.20)
            where[f"p{p}_d{d_}"] = {"total": int(r_.sum()), **{w: int(r_[(d_arr >= a) & (d_arr <= b)].sum()) for w, (a, b) in wins.items()},
                                    "first_in_h2": next((d for d, x in zip(days, r_) if x and d >= "2026-06-23"), "")}
    OUT["c1_trigger_days_by_window"] = where
    show("5 C1 trigger days by window", pd.DataFrame(where).T)
    dd_ctx = {d: round(float(dd[days.index(d)]), 4) for d in ("2026-06-22", "2026-06-23", "2026-07-01", "2026-09-22", "2026-10-12", "2026-10-13")}
    OUT["c1_soxx_dd60_at_peaks"] = dd_ctx
    show("5 SOXX dd60 known at the base peaks", dd_ctx)

    # per-rule view of the two declines and H1
    per = pd.DataFrame({k: dict(h1=float(nav.win(c1[k], *H1)[0][0]), jul=float(nav.win(c1[k], *JUL)[0][0]),
                                oct=float(nav.win(c1[k], *OCT)[0][0]), h2=float(nav.win(c1[k], *H2)[0][0]),
                                full=float(nav.win(c1[k])[0][0]), out_h1=int((~c1[k])[nav.ix(*H1)].sum()),
                                out_jul=int((~c1[k])[nav.ix(*JUL)].sum()), out_oct=int((~c1[k])[nav.ix(*OCT)].sum()))
                        for k in keys}).T
    per["reentry"] = [k.split("_")[-1] for k in per.index]
    per["trigger"] = ["_".join(k.split("_")[1:3]) for k in per.index]
    by_re = per.groupby("reentry")[["h1", "jul", "oct", "h2", "full", "out_h1", "out_jul", "out_oct"]].median()
    by_tr = per.groupby("trigger")[["h1", "jul", "oct", "h2", "full", "out_h1", "out_jul", "out_oct"]].median()
    OUT["c1_cash_by_reentry"] = by_re.round(4).to_dict("index")
    OUT["c1_cash_by_trigger"] = by_tr.round(4).to_dict("index")
    OUT["c1_cash_summary"] = dict(
        base_h1=table["base"]["h1"], base_jul=table["base"]["jul"], base_oct=table["base"]["octnov"],
        h1_median=float(per.h1.median()), h1_worst=float(per.h1.min()), h1_best=float(per.h1.max()),
        h1_rules_at_or_above_base=int((per.h1 >= table["base"]["h1"] - 1e-9).sum()),
        jul_median=float(per.jul.median()), jul_rules_better_than_base=int((per.jul > table["base"]["jul"]).sum()),
        oct_median=float(per.oct.median()), oct_rules_better_than_base=int((per.oct > table["base"]["octnov"]).sum()),
        full_rules_at_or_above_base=int((per.full >= table["base"]["full"] - 1e-9).sum()), rules=len(per))
    show("5 C1 cash rules by re-entry (medians of 6 triggers)", by_re)
    show("5 C1 cash rules by trigger (medians of 9 re-entries)", by_tr)
    show("5 C1 cash summary", OUT["c1_cash_summary"])

    # placebo: circular shifts of the percentile series
    def placebo_stat(p_, k_):
        r = c1_rules(p_, k_, dd, gate[10], gate[20])
        m_ = np.array([r[k] for k in keys], dtype=float)
        hf = np.where(m_ > 0, 1.0, .5)
        h_c, h_h = nav.win(m_, *H2)[0], nav.win(hf, *H2)[0]
        e_c, e_h = h_c - nav.matched(m_, *H2), h_h - nav.matched(hf, *H2)
        return (np.median(h_c), np.median(e_c), np.median(np.r_[h_c, h_h]), np.median(np.r_[e_c, e_h]),
                np.median(nav.win(m_, *H1)[0]), np.median(nav.win(m_)[0]), np.median(nav.win(m_, *JUL)[0]), np.median(nav.win(m_, *OCT)[0]))

    names = ["cash_median_h2", "cash_median_excess", "all_median_h2", "all_median_excess", "cash_median_h1", "cash_median_full",
             "cash_median_jul", "cash_median_oct"]
    real_stat = np.array(placebo_stat(pct, known))
    within = np.array([placebo_stat(np.roll(pct, k), np.roll(known, k)) for k in range(1, n)])
    lp, lk = long.pct.to_numpy(), long.known.to_numpy(bool)
    offs2 = rng.integers(1, len(lp), 1000)
    longs = np.array([placebo_stat(np.roll(lp, -k)[-n:], np.roll(lk, -k)[-n:]) for k in offs2])
    plc = {}
    for i, name in enumerate(names):
        plc[name] = dict(real=float(real_stat[i]),
                         within2026_220shifts_median=float(np.median(within[:, i])),
                         within2026_share_ge_real=float((within[:, i] >= real_stat[i] - 1e-12).mean()),
                         long_1000draws_median=float(np.median(longs[:, i])),
                         long_1000draws_p05=float(np.quantile(longs[:, i], .05)), long_1000draws_p95=float(np.quantile(longs[:, i], .95)),
                         long_share_ge_real=float((longs[:, i] >= real_stat[i] - 1e-12).mean()))
    OUT["c1_placebo"] = dict(distinct_long_offsets=int(len(np.unique(offs2))), long_series_sessions=int(len(lp)), stats=plc)
    show("5 C1 placebo by circular shift of the percentile series (price half and gates real)", pd.DataFrame(plc).T)

    # ================================================================= 6 auxiliary year 2025 on SOXX itself
    aux = {}
    for year in ("2025", "2026"):
        y = ser.loc[year]
        if year == "2026":
            y = y.loc[:days[-1]]
        yd = [str(d.date()) for d in y.index]
        yn = len(yd)
        ynav = Nav(yd, y.open.to_numpy(), y.close.to_numpy())
        yp, yk, ydd = y.pct.to_numpy(), y.known.to_numpy(bool), y.dd.to_numpy()
        ysafe = np.where(yk, yp, -1.0)
        ybb = carry(yk & (ysafe >= 80) & (ydd <= -.20), yk)
        yg = {}
        full_close = ser.close
        for m in (10, 20, 50):
            yg[m] = (full_close >= full_close.rolling(m).mean()).shift(1).reindex(y.index).to_numpy(bool)
        res = {"buy_hold": dict(ret=float(ynav.win(np.ones(yn))[0][0]), mdd=float(ynav.win(np.ones(yn))[1][0]), expo=1.0)}
        for m in (10, 20, 50):
            for label, f in ((f"gate_SOXX{m}", yg[m]), (f"C4o_SOXX{m}", c4o(yg[m], ybb)), (f"C4r_SOXX{m}", c4r(yg[m], ybb)),
                             (f"C4o_SOXX{m}_price_only", c4o(yg[m], ydd <= -.20)), (f"C4r_SOXX{m}_price_only", c4r(yg[m], ydd <= -.20))):
                r_, m_ = ynav.win(f)
                res[label] = dict(ret=float(r_[0]), mdd=float(m_[0]), expo=float(np.mean(f)))
        yr = c1_rules(yp, yk, ydd, yg[10], yg[20])
        ym = np.array([yr[k] for k in keys], dtype=float)
        ret = ynav.win(ym)[0]
        ex = ret - ynav.matched(ym, "0", "9")
        yraw = yk & (ysafe >= 90) & (ydd <= -.05) & (ydd > -.20)
        yo = y.open.to_numpy()
        tr = []
        for a, b in runs(carry(yraw, yk)):
            fire = np.flatnonzero(yraw[a:b]) + a
            tr.append(dict(first=yd[a], last=yd[b - 1], sessions=b - a, max_pct=float(np.nanmax(yp[fire])), min_dd=float(ydd[fire].min()),
                           **{f"soxx{h}": (float(yo[a + h] / yo[a] - 1) if a + h < yn else np.nan) for h in (5, 10, 20)}))

        def ymean(mask):
            out = {"n": int(mask.sum())}
            for h in (5, 10, 20):
                ix = np.flatnonzero(mask)
                ix = ix[ix + h < yn]
                out[f"soxx{h}"] = float(np.mean(yo[ix + h] / yo[ix] - 1)) if len(ix) else np.nan
            return out

        yshallow = (ydd <= -.05) & (ydd > -.20)
        ep = []
        for a, b in runs(ybb):
            ep.append(dict(first=yd[a], last=yd[b - 1], sessions=b - a, dd_first=float(ydd[a]), pct_first=float(yp[a]) if yk[a] else np.nan,
                           **{f"soxx{h}": (float(yo[a + h] / yo[a] - 1) if a + h < yn else np.nan) for h in (5, 10, 20)},
                           **{f"extra_held_vs_gate{m}": int((ybb[a:b] & ~yg[m][a:b]).sum()) for m in (10, 20, 50)}))
        aux[year] = dict(sessions=yn, first=yd[0], last=yd[-1], unknown=int((~yk).sum()), rules=res,
                         bb1_on_days=int(ybb.sum()), bb1_episodes=ep, dd20_days=int((ydd <= -.20).sum()),
                         c1_cash=dict(median_ret=float(np.median(ret)), median_excess_vs_matched=float(np.median(ex)),
                                      beat_buy_hold=int((ret > res["buy_hold"]["ret"]).sum()), beat_matched=int((ex > 0).sum()),
                                      worst=float(ret.min()), best=float(ret.max()), rules=len(keys)),
                         c1_trigger_runs=tr,
                         c1_forward=dict(all=ymean(np.ones(yn, bool)), trigger_days=ymean(yraw),
                                         shallow_pct_lt90=ymean(yk & (ysafe < 90) & yshallow),
                                         shallow_pct_ge95=ymean(yk & (ysafe >= 95) & yshallow)))
        show(f"6 SOXX itself {year}: override rules", pd.DataFrame(res).T)
        show(f"6 SOXX itself {year}: BB-1 episodes", pd.DataFrame(ep) if ep else "none")
        show(f"6 SOXX itself {year}: C1 cash family", aux[year]["c1_cash"])
        show(f"6 SOXX itself {year}: C1 trigger runs", pd.DataFrame(tr) if tr else "none")
        show(f"6 SOXX itself {year}: mean forward SOXX by condition", pd.DataFrame(aux[year]["c1_forward"]).T)
    OUT["soxx_itself"] = aux

    # optional Yahoo cross-check of the SOXX closes behind dd60 (adjusted vs unadjusted)
    try:
        import yfinance as yf
        raw_yf = yf.download("SOXX", start="2024-06-01", end="2026-11-19", auto_adjust=False, progress=False)
        if isinstance(raw_yf.columns, pd.MultiIndex):
            raw_yf.columns = raw_yf.columns.get_level_values(0)
        raw_yf.index = pd.DatetimeIndex(raw_yf.index).tz_localize(None)
        raw_yf[["Open", "Close", "Adj Close"]].to_csv(common.CACHE / "verify_bb1_soxx_yf.csv")
        both_ix = raw_yf.index.intersection(ser.loc["2025":].index)
        chk = dict(rows=len(both_ix), last=str(raw_yf.index.max().date()),
                   close_max_rel_diff=float((raw_yf.Close.reindex(both_ix) / ser.close.reindex(both_ix) - 1).abs().max()),
                   adjclose_max_rel_diff=float((raw_yf["Adj Close"].reindex(both_ix) / ser.close.reindex(both_ix) - 1).abs().max()))
        for col in ("Close", "Adj Close"):
            c_ = raw_yf[col]
            dd_y = (c_ / c_.rolling(60, min_periods=60).max() - 1).shift(1)
            a_ = dd_y.reindex(ser.loc["2025":].index)
            b_ = ser.dd.loc["2025":]
            ok_ = a_.notna() & b_.notna()
            chk[f"dd60_{col.replace(' ', '')}_max_abs_diff"] = float((a_ - b_)[ok_].abs().max())
            chk[f"dd20_flag_flips_{col.replace(' ', '')}"] = [str(d.date()) for d in b_.index[ok_ & ((a_ <= -.20) != (b_ <= -.20))]]
        OUT["yahoo_check"] = chk
    except Exception as exc:  # network or schema trouble: report, do not fail the audit
        OUT["yahoo_check"] = dict(error=f"{type(exc).__name__}: {exc}")
    show("6 Yahoo cross-check of SOXX closes", OUT["yahoo_check"])

    OUT["seconds"] = round(time.time() - t0, 1)
    RESULT.write_text(json.dumps(OUT, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nwrote {RESULT} in {OUT['seconds']}s")


if __name__ == "__main__":
    main()
