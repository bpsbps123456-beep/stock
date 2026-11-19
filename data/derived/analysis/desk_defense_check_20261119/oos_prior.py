"""Out-of-sample scoring of rules the repo froze BEFORE the October-November 2026 decline.

Every rule is implemented with its frozen definition (source file quoted per rule), replayed on the
lead's harness (common.context + study.replay, close-marked USD and KRW), and scored from the first
US session after its freeze through 2026-11-17 against the base over the same sessions.

Engines
  overlay(e, hedge)   daily-rebalanced fraction e[j] of the base basket (+ hedge ETF columns appended to
                      copies of ctx.opn/ctx.close), i.e. stop_research's `e * ret` overlay, via study.replay.
  run_weekly(...)     the desk weekly schedule with research overrides (caps, selector, weekly/daily cash
                      gates, decision weekday, entry/exit MA) -> target vectors -> study.replay.
Point in time: a decision at the open of session j uses closes through j-1 (signals are shifted one session).

Run:  PYTHONIOENCODING=utf-8 py data/derived/analysis/desk_defense_check_20261119/oos_prior.py
Writes cache/oos_daily.parquet, out/oos_candidates.csv (common.save) and cache/oos_prior_result.json.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import common
import study
from desk import config, strategies as S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FROZEN = HERE / "frozen_etf"          # yfinance SOXS/HYG/IEF/SOXX/QQQ/SOXL daily bars, frozen 2026-11-19
STOP = ROOT / "data/derived/analysis/stop_research"
END = "2026-11-17"
JUL = ("2026-06-23", "2026-07-29")    # 6/22 close (base peak) -> 7/29 close
OCTNOV = ("2026-10-13", END)          # 10/12 close -> 11/17 close
FULL = ("2026-01-02", END)


# ----------------------------------------------------------------------------- inputs
def etf(sym: str) -> pd.DataFrame:
    return pd.read_csv(FROZEN / f"{sym}.csv", index_col=0)


class Lab:
    def __init__(self):
        self.ctx = ctx = common.context()
        self.days = ctx.days
        self.n = len(ctx.days)
        self.keys = list(ctx.col)
        self.ncol = len(self.keys)
        B = ctx.book
        self.semis = np.array([S.in_semis_budget(B.sector[k]) for k in self.keys])
        self.base = study.replay(ctx.days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, ctx.gates["none"])
        np.testing.assert_allclose(self.base.usd, ctx.baseline.close, rtol=1e-10)
        self.base_w = np.zeros((self.n, self.ncol))
        for j, w in enumerate(ctx.baseline.weights):
            for k, v in w.items():
                self.base_w[j, ctx.col[k]] = v
        self.curves, self.cfg, self.aux = {"base": self.base}, {"base": dict(source="desk", rule="base")}, {}

    # -- signals known at the previous close ------------------------------------------------
    def prev(self, s: pd.Series, fill=False) -> np.ndarray:
        """Series on book.days (value known at that close) -> value usable at each session's open."""
        out = s.reindex(self.ctx.book.days).shift(1).reindex(self.days)
        return out.fillna(fill).to_numpy()

    def below(self, sym: str, n: int) -> pd.Series:
        c = self.ctx.book.cls[sym]
        m = c.rolling(n, min_periods=n).mean()
        return ((c < m) & m.notna()).astype(bool)

    # -- engines ----------------------------------------------------------------------------
    def overlay(self, e, hedge: dict | None = None) -> pd.DataFrame:
        ctx, hedge = self.ctx, hedge or {}
        e = np.asarray(e, float)
        opn, close = ctx.opn, ctx.close
        for sym in hedge:
            px = etf(sym).reindex(self.days)
            assert px[["Open", "Close"]].notna().all().all(), sym
            opn, close = np.c_[opn, px.Open.to_numpy()], np.c_[close, px.Close.to_numpy()]
        targets = {}
        for j in range(self.n):
            w = np.zeros(self.ncol + len(hedge))
            w[:self.ncol] = e[j] * self.base_w[j]
            for i, h in enumerate(hedge.values()):
                w[self.ncol + i] = h[j]
            assert w.sum() <= 1 + 1e-9
            targets[j] = w
        return study.replay(self.days, opn, close, ctx.fx, targets, np.ones(self.n, bool))

    def gate(self, flags) -> pd.DataFrame:
        ctx = self.ctx
        return study.replay(self.days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, np.asarray(flags, bool))

    def project(self, raw, caps):
        if not caps:
            return S.project_caps(raw, self.ctx.book.sector)
        saved = dict(config.RULE)
        try:
            config.RULE.update({k: v for k, v in caps.items() if k in config.RULE})
            out = S.project_caps(raw, self.ctx.book.sector)
        finally:
            config.RULE.clear()
            config.RULE.update(saved)
        if caps.get("raw_ceiling"):
            out = {k: min(v, raw[k]) for k, v in out.items()}
        return out

    def run_weekly(self, *, caps=None, select=None, eq_n=None, wd=0, gate=None, daily=None,
                   entry_ok=None, exit_sig=None, active_from=None):
        """Desk weekly schedule with research overrides. Returns (ledger, info).

        gate[j]  None/'' = normal, 'semis' = semis-budget names to cash, 'all' = everything to cash; read on
                 rebalance days only (signal already shifted to the open of j).
        daily[j] bool: gate_main 'soxx_d' — sell semis at the open the signal turns on, re-buy this week's
                 target semis weights (inside remaining cash) the open it turns off.
        entry_ok[j, col] / exit_sig[j, col]: ma_entry_filter new-entry gate / per-stock full exit.
        active_from: overrides apply from this session; before it the path is the base (common start).
        """
        ctx, col = self.ctx, self.ctx.col
        w, base_t = np.zeros(self.ncol), np.zeros(self.ncol)
        selected, started, last_wk, on = set(), False, None, False
        value, targets, events = [1.0], {}, []
        for j, day in enumerate(self.days):
            if j:
                held = w > 0
                r = np.zeros(self.ncol)
                r[held] = ctx.opn[j, held] / ctx.opn[j - 1, held] - 1
                assert np.isfinite(r).all(), day
                gross, total = float(w @ r), float(w @ (1 + r)) + 1 - w.sum()
                w = w * (1 + r) / total
                value.append(value[-1] * (1 + gross))
            act = active_from is None or day >= active_from
            changed = False
            if act and exit_sig is not None:
                ex = (w > 0) & exit_sig[j]
                if ex.any():
                    w = np.where(ex, 0.0, w)
                    selected -= {self.keys[i] for i in np.flatnonzero(ex)}
                    events += [dict(day=day, type="exit", ticker=ctx.tickers[i]) for i in np.flatnonzero(ex)]
                    changed = True
            wk = S.iso_week(day)
            reb = (not started) or (wk != last_wk and date.fromisoformat(day).weekday() >= wd)
            ranked = ctx.rankings[j]
            sig = (gate[j] if (gate is not None and act) else None) or None
            if reb and ranked:
                sel = (select if (select and act) else S.select)(ranked, selected)
                if eq_n and act:
                    raw = {k: 1.0 / eq_n for k in sel}
                else:
                    tw = {r_["key"]: r_["time_weight"] for r_ in ranked}
                    tot = sum(tw[k] for k in sel)
                    raw = {k: tw[k] / tot for k in sel}
                tgt = self.project(raw, caps if act else None)
                if act and entry_ok is not None:
                    blocked = {k for k in sel if k not in selected and not entry_ok[j, col[k]]}
                    events += [dict(day=day, type="blocked", ticker=ctx.book.ticker[k]) for k in sorted(blocked)]
                    sel = [k for k in sel if k not in blocked]
                    tgt = {k: v for k, v in tgt.items() if k not in blocked}
                started, last_wk, selected = True, wk, set(sel)
                base_t = np.zeros(self.ncol)
                for k, v in tgt.items():
                    base_t[col[k]] = v
                if bool(sig) != on:
                    events.append(dict(day=day, type="gate_on" if sig else "gate_off"))
                on = bool(sig)
                w = np.zeros(self.ncol) if sig == "all" else np.where(self.semis, 0.0, base_t) if sig == "semis" \
                    else base_t.copy()
                changed = True
            elif act and daily is not None and started and bool(daily[j]) != on:
                if daily[j]:
                    w = np.where(self.semis, 0.0, w)
                else:
                    add = np.where(self.semis, base_t, 0.0)
                    room, need = 1 - w.sum(), add.sum()
                    w = w + add * (min(1.0, room / need) if need > 0 else 0.0)
                on = bool(daily[j])
                events.append(dict(day=day, type="gate_on" if on else "gate_off"))
                changed = True
            if changed:
                targets[j] = w.copy()
        ledger = study.replay(self.days, ctx.opn, ctx.close, ctx.fx, targets, ctx.gates["none"])
        return ledger, dict(value=pd.Series(value, index=self.days), events=events)

    def add(self, key, ledger, **cfg):
        assert key not in self.curves, key
        self.curves[key], self.cfg[key] = ledger, cfg
        return ledger


# ----------------------------------------------------------------------------- statistics
def trades(curve, a, b):
    t = curve.turn[(curve.index >= a) & (curve.index <= b)]
    return int((t > 1e-6).sum())


def control(base, expo, a, b):
    """Hindsight control: hold expo/base_expo of the base every day of [a, b], rest USD cash."""
    ix = np.flatnonzero((base.index >= a) & (base.index <= b))
    normal = base.exposure.to_numpy()[ix].mean()
    if normal <= 0 or expo > normal + 1e-9:
        return np.nan, np.nan
    usd = base.usd.to_numpy()
    path = np.r_[usd[ix[0] - 1] if ix[0] else 1.0, usd[ix]]
    nav = np.r_[1.0, np.cumprod(1 + expo / normal * (path[1:] / path[:-1] - 1))]
    return float(nav[-1] - 1), float(np.min(nav / np.maximum.accumulate(nav) - 1))


def score(curve, base, a, b):
    s, o = common.window_stats(curve, a, b), common.window_stats(base, a, b)
    c_ret, c_mdd = control(base, s["expo"], a, b)
    return dict(start=a, end=b, sessions=int(((curve.index >= a) & (curve.index <= b)).sum()),
                usd=s["usd"], usd_mdd=s["usd_mdd"], krw=s["krw"], krw_mdd=s["krw_mdd"], expo=s["expo"],
                turn=s["turn"], trade_days=trades(curve, a, b),
                base_usd=o["usd"], base_usd_mdd=o["usd_mdd"], base_krw=o["krw"], base_krw_mdd=o["krw_mdd"],
                base_expo=o["expo"], base_trade_days=trades(base, a, b), ctrl_usd=c_ret, ctrl_usd_mdd=c_mdd,
                d_usd=s["usd"] - o["usd"], d_mdd=s["usd_mdd"] - o["usd_mdd"],
                d_vs_ctrl=(s["usd"] - c_ret) if np.isfinite(c_ret) else np.nan)


def helped(x, tol=1e-4):
    """+1 better return and shallower drawdown than base, -1 worse on both, 0 otherwise."""
    up, shallow = x["d_usd"] > tol, x["d_mdd"] > tol
    down, deep = x["d_usd"] < -tol, x["d_mdd"] < -tol
    return 1 if (up and not deep) or (shallow and not down) else -1 if (down and not shallow) or (deep and not up) \
        else 0


def verdict(post, min_sessions=10):
    """Mechanical verdict on the post-freeze window.

    held_up: USD return above base, drawdown not deeper, and return above the exposure-matched control.
    failed : USD return below base and drawdown not shallower (or below base while the matched control,
             i.e. simply holding less, did better on return).
    mixed  : anything else.  not_testable: fewer than min_sessions or the rule never departed from base.
    """
    if post["sessions"] < min_sessions:
        return "not_testable"
    if abs(post["d_usd"]) < 1e-9 and abs(post["d_mdd"]) < 1e-9:
        return "not_testable"
    beat_ctrl = (not np.isfinite(post["d_vs_ctrl"])) or post["d_vs_ctrl"] > 1e-4
    if post["d_usd"] > 1e-4 and post["d_mdd"] > -1e-4 and beat_ctrl:
        return "held_up"
    if post["d_usd"] < -1e-4 and post["d_mdd"] < 1e-4:
        return "failed"
    if post["d_usd"] < -1e-4 and not beat_ctrl:
        return "failed"
    return "mixed"


def oo_stats(value: pd.Series, lo, hi, end="2026-10-09", lead_one=True):
    """Open-to-open return/drawdown as the frozen studies measured them (data ended 2026-10-09)."""
    v = value[value.index <= end]
    r = (v.shift(-1) / v - 1).dropna()
    x = r[(r.index >= lo) & (r.index <= hi)].to_numpy()
    c = np.cumprod(1 + x)
    if lead_one:
        c = np.r_[1.0, c]
    return float(c[-1] / (1.0 if lead_one else 1.0) - 1), float((c / np.maximum.accumulate(c) - 1).min())


def overlay_value(lab, e, hedge=None):
    """stop_research harness.overlay: r[k] = e[k]*base_ret[k] + sum h[k]*hedge_ret[k], open to open."""
    v = pd.Series(lab.ctx.baseline.value, index=lab.days)
    r = np.asarray(e, float)[:-1] * (v.shift(-1) / v - 1).to_numpy()[:-1]
    for sym, h in (hedge or {}).items():
        o = etf(sym).Open.reindex(lab.days).to_numpy()
        r = r + np.asarray(h, float)[:-1] * (o[1:] / o[:-1] - 1)
    return pd.Series(np.r_[1.0, np.cumprod(1 + r)], index=lab.days)


# ----------------------------------------------------------------------------- frozen rule definitions
def rule_soxx50_20d(lab, n=50, hold=20):
    """stop_research/code/v2_soxx50_20d.py run_rule(kind='cross', fresh start at the first 2026 session)."""
    c = lab.ctx.book.cls["SOXX"]
    sma = c.rolling(n, min_periods=n).mean()
    below, ok, idx = (c < sma).to_numpy(), sma.notna().to_numpy(), list(c.index)
    i0 = idx.index(lab.days[0])
    e = np.full(len(c), np.nan)
    state, xi, eps = "in", None, []
    for i in range(i0, len(c)):
        if state == "in":
            if below[i] and ok[i - 1] and not below[i - 1]:
                state, xi = "out", i
        elif i - xi >= hold:
            state = "in"
            eps.append((idx[xi], idx[i]))
        e[i] = 1.0 if state == "in" else 0.0
    if state == "out":
        eps.append((idx[xi], None))
    return lab.prev(pd.Series(e, index=idx), fill=1.0), eps


def rule_credit(lab, n=50, low=0.5, m=5):
    """stop_research/code/v2_credit_trend.py: HYG/IEF (Adj Close) < SMA50 -> exposure 0.5, min 5 sessions."""
    r = (etf("HYG")["Adj Close"] / etf("IEF")["Adj Close"]).dropna()
    sma = r.rolling(n).mean()
    f = (r < sma)[sma.notna()]
    fd, fv = np.asarray(f.index), f.to_numpy()
    pos = np.searchsorted(fd, np.asarray(lab.days), side="left") - 1
    on = np.where(pos >= 0, fv[np.clip(pos, 0, None)], False)
    e, red, k0 = np.ones(lab.n), False, -1
    for k in range(lab.n):
        if red and k - k0 >= max(m, 1) and not on[k]:
            red = False
        if not red and on[k]:
            red, k0 = True, k
        e[k] = low if red else 1.0
    return e


def rule_dd10_buy15(lab, X=0.10, D=0.15):
    """stop_research/code/v2_dd10_buy15.py simulate(release='nh252'), signals(warm=True, chain_peak=False)."""
    c = pd.Series(lab.ctx.baseline.close, index=lab.days)
    hold_path = STOP / "cache" / "base_hold.json"
    ext = c
    if hold_path.exists():                       # SMA20 warm-up with the scaled 2023-25 path, as frozen
        h = json.loads(hold_path.read_text(encoding="utf-8"))
        pre = pd.Series(h["close"], index=h["days"]).iloc[-80:]
        ext = pd.concat([pre * (c.iloc[0] / pre.iloc[-1]), c])
    sma = ext.rolling(20, min_periods=1).mean().reindex(lab.days).to_numpy()
    dd = (c / c.rolling(252, min_periods=1).max() - 1).to_numpy()
    cv = c.to_numpy()
    mode, peak, expo, eps, xi = "in", cv[0], np.ones(lab.n), [], None
    for i in range(lab.n):
        if mode == "in":
            peak = max(peak, cv[i])
            if cv[i] / peak - 1 <= -X:
                mode, xi = "out", i
        elif mode == "out":
            if dd[i] <= -D:
                eps.append(dict(why="deep", exit_signal=lab.days[xi], back_signal=lab.days[i]))
                mode = "keep"
            elif cv[i] > sma[i]:
                eps.append(dict(why="sma", exit_signal=lab.days[xi], back_signal=lab.days[i]))
                mode, peak = "in", cv[i]
        elif dd[i] >= -1e-9:
            mode, peak = "in", cv[i]
        expo[i] = 0.0 if mode == "out" else 1.0
    if mode == "out":
        eps.append(dict(why="end", exit_signal=lab.days[xi], back_signal=None))
    e = np.ones(lab.n)
    e[1:] = expo[:-1]
    return e, eps, mode


def rule_earnings_eve(lab, syms=("NVDA", "AVGO", "MU", "TSM"), lvl=0.5):
    """stop_research/code/fam_H_events_funds.py event_exposure(win='gap'): half exposure from the open of the
    session before the first reaction open. TSM = before the open, others after the close (frozen convention).
    Dates: data/derived/desk/trend_earnings.json (read-only; the frozen cache stopped at 2026-10-09)."""
    dates = json.loads((ROOT / "data/derived/desk/trend_earnings.json").read_text(encoding="utf-8"))["dates"]
    arr, e, used = np.asarray(lab.days), np.ones(lab.n), []
    for s in syms:
        for d in dates[s]:
            if not ("2026-01-01" <= d <= "2026-12-31"):
                continue
            i = int(np.searchsorted(arr, d, side="left" if s == "TSM" else "right"))
            if 1 <= i <= lab.n:                   # i == n: reaction open is after the data, eve is the last day
                e[i - 1] = min(e[i - 1], lvl)
                used.append((s, d, lab.days[i - 1]))
    return e, sorted(used, key=lambda x: x[2])


def selector_sub1(lab, m=1):
    """rsi_breakout/code/deskx_subcount.py limited_select: at most m names per sub-sector group."""
    B = lab.ctx.book
    group = {k: S.subsector_of(k, B.ticker[k]) for k in lab.keys}
    group = {k: (f"미분류:{B.ticker[k]}" if g == S.UNKNOWN else g) for k, g in group.items()}
    n, buf = config.RULE["n"], config.RULE["buffer"]

    def select(ranked, current):
        order = [r["key"] for r in ranked]
        cnt, keep = Counter(), []
        for x in order[:n + buf]:
            if x in current and len(keep) < n and cnt[group[x]] < m:
                keep.append(x)
                cnt[group[x]] += 1
        for x in order:
            if len(keep) >= n:
                break
            if x not in keep and cnt[group[x]] < m:
                keep.append(x)
                cnt[group[x]] += 1
        return keep
    return select


def selector_bucket(lab, n=5, buf=5):
    """rsi_breakout/code/bucket_main.py BucketEngine.select(gran='sub'): one name per sub-sector bucket."""
    B = lab.ctx.book
    bucket = {k: S.subsector_of(k, B.ticker[k]) for k in lab.keys}
    bucket = {k: (f"미분류:{B.ticker[k]}" if g == S.UNKNOWN else g) for k, g in bucket.items()}

    def select(ranked, current):
        order = [r["key"] for r in ranked]
        members = {}
        for x in order:
            members.setdefault(bucket[x], []).append(x)
        leaders = list(members)
        lrank = {b: i + 1 for i, b in enumerate(leaders)}
        keep, held = [], set()
        for x in order:
            if x not in current:
                continue
            b = bucket[x]
            if lrank[b] <= n + buf and members[b].index(x) < 2 and b not in held and len(keep) < n:
                keep.append(x)
                held.add(b)
        for b in leaders:
            if len(keep) >= n:
                break
            if b not in held:
                keep.append(members[b][0])
                held.add(b)
        return keep
    return select


def ma_flags(lab, kind, length):
    """ma_entry_filter review.py: previous close >= MA (entry_ok) / previous close < MA (exit)."""
    close = lab.ctx.book.cls[lab.ctx.tickers]
    avg = close.rolling(length, min_periods=length).mean() if kind == "SMA" else \
        close.ewm(span=length, adjust=False, min_periods=length, ignore_na=True).mean()
    ok = ((close >= avg) & close.notna() & avg.notna())
    ex = ((close < avg) & close.notna() & avg.notna())
    sh = lambda f: f.shift(1).reindex(lab.days).fillna(False).to_numpy(bool)
    return sh(ok), sh(ex)


# ----------------------------------------------------------------------------- section 5 alerts
def alerts(lab):
    """docs/desk_system.md §5 (desk/alerts.py thresholds), replayed session by session from 2026-09-28.

    Model = the live rule (ctx.baseline), KRW close index; TIME ETF = first KRX close after each US session
    (S.etf_index). A value for US session j is complete once that KRX close exists, i.e. before session j+1.
    """
    ctx = lab.ctx
    model = S.model_index(ctx.book, ctx.baseline)
    etf_ix = S.etf_index(ctx.book, lab.days)
    curve = np.r_[100.0, model]
    dd = (curve / np.maximum.accumulate(curve) - 1)[1:]
    gap = lambda n: np.array([np.nan if j < n else (model[j] / model[j - n] - 1) - (etf_ix[j] / etf_ix[j - n] - 1)
                              for j in range(lab.n)])
    g20, g120 = gap(20), gap(120)
    frame = pd.DataFrame(dict(model_krw=model, etf_krw=etf_ix, dd=dd, gap20=g20, gap120=g120), index=lab.days)
    frame["a_gap20"] = frame.gap20 <= -0.10
    frame["a_dd"] = frame.dd <= -0.353
    frame["a_gap120"] = frame.gap120 < 0
    return frame


def spells(flags: pd.Series):
    out, start, idx = [], None, list(flags.index)
    for i, on in enumerate(flags.to_numpy()):
        if on and start is None:
            start = i
        if start is not None and (not on or i == len(idx) - 1):
            last = i if on else i - 1
            out.append((idx[start], idx[last], last - start + 1))
            start = None
    return out


# ----------------------------------------------------------------------------- main
def main():
    lab = Lab()
    ctx, days, base = lab.ctx, lab.days, lab.base
    ones = np.ones(lab.n)
    rows, repro, notes = [], {}, {}

    # engine self-checks
    chk, info0 = lab.run_weekly()
    np.testing.assert_allclose(chk.to_numpy(), base.to_numpy(), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(info0["value"], ctx.baseline.value, rtol=1e-10)
    np.testing.assert_allclose(lab.overlay(ones)[["krw", "usd", "exposure"]].to_numpy(),
                               base[["krw", "usd", "exposure"]].to_numpy(), rtol=1e-9, atol=1e-9)
    base_val = pd.Series(ctx.baseline.value, index=days)
    repro["base(live50) open-to-open 1/2~5/29, 6/1~10/9, full to 10/9, mdd"] = dict(
        mine=[oo_stats(base_val, "2026-01-02", "2026-05-29")[0], oo_stats(base_val, "2026-06-01", "2026-10-09")[0],
              *oo_stats(base_val, "2026-01-02", "2026-10-09")], frozen=[2.0619, 0.0464, 2.2038, -0.2027])

    def register(key, ledger, *, source, definition, frozen_on, seen_to, post_start, base_curve=None, note="",
                 min_sessions=10, force=None, save=True, **extra):
        b = base if base_curve is None else base_curve
        if save:
            lab.add(key, ledger, source=source, frozen_on=frozen_on, post_start=post_start)
        w = dict(post=score(ledger, b, post_start, END), full=score(ledger, b, *FULL), jul=score(ledger, b, *JUL),
                 octnov=score(ledger, b, *OCTNOV),
                 pre=score(ledger, b, "2026-01-02", seen_to))
        v = force or verdict(w["post"], min_sessions)
        rows.append(dict(name=key, source=source, definition=definition, frozen_on=frozen_on, seen_to=seen_to,
                         windows=w, verdict=v, note=note, jul_helped=helped(w["jul"]),
                         octnov_helped=helped(w["octnov"]), **extra))
        return w

    SR = "stop_research/report.md (docs commit fa9c4d89, 2026-10-10 21:00 KST)"
    P1 = "2026-10-12"                 # first US session after the 2026-10-10 freezes (data seen to 10-09)

    # ---- stop_research survivor 1: SOXX below SMA50 (cross-down) -> 20 sessions cash
    e, eps = rule_soxx50_20d(lab)
    led = lab.gate(e > 0.5)
    np.testing.assert_allclose(lab.overlay(e)[["usd"]].to_numpy(), led[["usd"]].to_numpy(), rtol=1e-9)
    v = overlay_value(lab, e)
    repro["soxx50_20d full/back/AIshock"] = dict(
        mine=[*oo_stats(v, "2026-01-02", "2026-10-09"), oo_stats(v, "2026-06-01", "2026-10-09")[0],
              oo_stats(v, "2026-06-22", "2026-07-29")[0]], frozen=[2.302, -0.142, 0.142, -0.073])
    register("sr_soxx50_cash20", led, source=SR, frozen_on="2026-10-10", seen_to="2026-10-09", post_start=P1,
             definition="SOXX close crosses below SMA50 (prev close at/above) -> all cash from next open; back in at "
                        "the open after the 20th session from the signal day; signals ignored while out; state "
                        "starts 'in' on 2026-01-02 (v2_soxx50_20d.py, kind=cross, fresh)",
             note=f"episodes(signal day, return signal day)={eps}")

    # ---- survivor 2: SOXX below SMA50 -> 80% stock + 20% SOXS, rebalanced daily
    on50 = lab.prev(lab.below("SOXX", 50)).astype(float)
    led = lab.overlay(1 - 0.2 * on50, {"SOXS": 0.2 * on50})
    v = overlay_value(lab, 1 - 0.2 * on50, {"SOXS": 0.2 * on50})
    repro["soxs_hedge full/back/AIshock"] = dict(
        mine=[*oo_stats(v, "2026-01-02", "2026-10-09"), oo_stats(v, "2026-06-01", "2026-10-09")[0],
              oo_stats(v, "2026-06-22", "2026-07-29")[0]], frozen=[2.032, -0.157, 0.030, -0.126])
    register("sr_soxx50_soxs20", led, source=SR, frozen_on="2026-10-10", seen_to="2026-10-09", post_start=P1,
             definition="while SOXX close < SMA50 (decided at close t, applied from open t+1): 80% of the 7-stock "
                        "basket + 20% SOXS, rebalanced to those weights every open; otherwise 100% basket "
                        "(v2_soxs_hedge.py path_daily)",
             note=f"on-days post-freeze={int(on50[days.index(P1):].sum())}; exposure column counts SOXS as invested")
    # same trigger, 20% cash instead of SOXS (the study's own control for the hedge leg)
    led_c = lab.overlay(1 - 0.2 * on50)
    register("sr_soxx50_cash20pct_control", led_c, source=SR + " (control inside v2_soxs_hedge.py)",
             frozen_on="2026-10-10", seen_to="2026-10-09", post_start=P1,
             definition="same SOXX<SMA50 trigger, 80% basket + 20% USD cash (no inverse ETF)", note="control")

    # ---- survivor 3: HYG/IEF below SMA50 -> exposure 0.5 (min 5 sessions)
    e = rule_credit(lab)
    led = lab.overlay(e)
    v = overlay_value(lab, e)
    repro["credit_trend full/back/AIshock"] = dict(
        mine=[*oo_stats(v, "2026-01-02", "2026-10-09"), oo_stats(v, "2026-06-01", "2026-10-09")[0],
              oo_stats(v, "2026-06-22", "2026-07-29")[0]], frozen=[2.264, -0.180, 0.076, -0.164])
    sp = spells(pd.Series(e < 0.999, index=days))
    register("sr_hyg_ief50_half", led, source=SR, frozen_on="2026-10-10", seen_to="2026-10-09", post_start=P1,
             definition="HYG/IEF ratio (Adj Close) < its SMA50 at close t -> 50% of the basket from open t+1, held "
                        "at least 5 sessions, back to 100% when the signal is off (v2_credit_trend.py)",
             note=f"reduced spells (first open, last open, sessions)={sp}")

    # ---- survivor 4: -10% from trailing peak -> cash; buy back at -15% and keep to a new high
    e, eps, mode = rule_dd10_buy15(lab)
    led = lab.gate(e > 0.5)
    v = overlay_value(lab, e)
    repro["dd10_buy15 full/back/AIshock"] = dict(
        mine=[*oo_stats(v, "2026-01-02", "2026-10-09"), oo_stats(v, "2026-06-01", "2026-10-09")[0],
              oo_stats(v, "2026-06-22", "2026-07-29")[0]], frozen=[2.593, -0.166, 0.161, -0.084])
    register("sr_dd10_buy15_keep", led, source=SR, frozen_on="2026-10-10", seen_to="2026-10-09", post_start=P1,
             definition="shadow = base USD close; close <= trailing peak x0.90 -> cash next open; while out: shadow "
                        "252-session drawdown <= -15% -> buy back next open and keep (no stop) until a new 252-session "
                        "high, else shadow > its SMA20 -> back in (v2_dd10_buy15.py, release=nh252)",
             note=f"episodes={[x for x in eps if x['exit_signal'] >= '2026-09-01']}; state at end={mode}")

    # ---- survivor 5 / observation candidate: weekly SOXX<SMA20 -> semis budget 0% that week (cash)
    below20 = lab.prev(lab.below("SOXX", 20))
    gate_sx = np.where(below20, "semis", "")
    wk_base, wk_gate = {0: (base, info0)}, {}
    for wd in range(5):
        if wd:
            wk_base[wd] = lab.run_weekly(wd=wd)
        wk_gate[wd] = lab.run_weekly(wd=wd, gate=gate_sx)
    led, info = wk_gate[0]
    repro["live50·soxx_w front/back/full/mdd (deskx evaluate)"] = dict(
        mine=[oo_stats(info["value"], "2026-01-02", "2026-05-29", lead_one=False)[0],
              oo_stats(info["value"], "2026-06-01", "2026-10-09", lead_one=False)[0],
              *oo_stats(info["value"], "2026-01-02", "2026-10-09", lead_one=False)],
        frozen=[1.8754, 0.0925, 2.1414, -0.1415])
    GM = "rsi_breakout/report.md 추가 5 (docs commit f4741f7e, 2026-10-10 22:24 KST); code gate_main.py"
    ev = [x for x in info["events"] if x["day"] >= "2026-09-28"]
    register("sr_soxx20_week_semis0_mon", led, source=SR + " '관찰 후보' = " + GM + " soxx_w",
             frozen_on="2026-10-10", seen_to="2026-10-09", post_start=P1,
             definition="on the weekly rebalance open (first US session of the ISO week): if the previous close of "
                        "SOXX < SMA20, the semis+unclassified names of that week's target are held as cash, other "
                        "weights unchanged; no mid-week action (v2_I_semicap S2/cap0/현금/주간 == gate_main soxx_w)",
             note=f"gate events since 9/28={ev}")
    for wd in range(1, 5):
        l2, i2 = wk_gate[wd]
        register(f"sr_soxx20_week_semis0_wd{wd}", l2, source=GM + " robustness (decision weekday)",
                 frozen_on="2026-10-10", seen_to="2026-10-09", post_start=P1, base_curve=wk_base[wd][0],
                 definition=f"same rule, weekly decision on the first session with weekday >= {wd} (Mon=0); compared "
                            f"with the base rebalanced on the same weekday",
                 note="vs same-weekday base; vs Monday base see extra table",
                 vs_monday_base=score(l2, base, P1, END))
        lab.add(f"base_wd{wd}", wk_base[wd][0], source="weekday base", frozen_on="2026-10-10", post_start=P1)
    for nsma in (10, 50):
        g = np.where(lab.prev(lab.below("SOXX", nsma)), "semis", "")
        l2, _ = lab.run_weekly(gate=g)
        register(f"sr_soxx{nsma}_week_semis0_mon", l2, source=GM + " robustness (SMA length)",
                 frozen_on="2026-10-10", seen_to="2026-10-09", post_start=P1,
                 definition=f"same weekly semis-to-cash rule with SMA{nsma}", note="robustness variant, not the pick")

    # ---- rsi_breakout 추가 5 grid: bases x cash switches
    below20q = lab.prev(lab.below("QQQ", 20))
    gate_q = np.where(below20q, "all", "")
    grid_bases = {"live50": {}, "c774": dict(caps=dict(semis_cap=0.60)), "nocap": dict(caps=dict(semis_cap=1.0)),
                  "sub5eq": dict(caps=dict(semis_cap=1.0, stock_cap=0.25), select=selector_bucket(lab), eq_n=5)}
    grid_gates = {"none": {}, "soxx_w": dict(gate=gate_sx), "soxx_d": dict(gate=gate_sx, daily=below20),
                  "qqq_w": dict(gate=gate_q)}
    frozen_grid = {"live50·none": [2.0619, 0.0464, 2.2038, -0.2027], "live50·soxx_w": [1.8754, 0.0925, 2.1414, -0.1415],
                   "live50·soxx_d": [1.8119, -0.002, 1.8064, -0.1655], "live50·qqq_w": [0.9583, -0.2311, 0.5058, -0.254],
                   "c774·none": [2.0496, 0.0492, 2.1995, -0.2373], "c774·soxx_w": [1.8642, 0.107, 2.1708, -0.1407],
                   "c774·soxx_d": [1.8031, -0.0069, 1.7837, -0.1655], "c774·qqq_w": [0.9518, -0.2547, 0.4546, -0.2748],
                   "nocap·none": [1.9489, -0.0253, 1.8745, -0.3408], "nocap·soxx_w": [1.7697, 0.0656, 1.9514, -0.1407],
                   "nocap·soxx_d": [1.7106, -0.1022, 1.4334, -0.2686], "nocap·qqq_w": [0.8874, -0.312, 0.2985, -0.3409],
                   "sub5eq·none": [1.9296, 0.0542, 2.0886, -0.3099], "sub5eq·soxx_w": [1.5191, 0.1486, 1.8934, -0.2123],
                   "sub5eq·soxx_d": [1.5207, 0.006, 1.5359, -0.256], "sub5eq·qqq_w": [0.9623, -0.2825, 0.408, -0.2985]}
    grid = {}
    for bname, bkw in grid_bases.items():
        for gname, gkw in grid_gates.items():
            led, info = lab.run_weekly(**bkw, **gkw)
            grid[f"{bname}·{gname}"] = (led, info)
            repro[f"gate grid {bname}·{gname} front/back/full/mdd"] = dict(
                mine=[oo_stats(info["value"], "2026-01-02", "2026-05-29", lead_one=False)[0],
                      oo_stats(info["value"], "2026-06-01", "2026-10-09", lead_one=False)[0],
                      *oo_stats(info["value"], "2026-01-02", "2026-10-09", lead_one=False)],
                frozen=frozen_grid[f"{bname}·{gname}"])
    np.testing.assert_allclose(grid["live50·none"][0].to_numpy(), base.to_numpy(), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(grid["live50·soxx_w"][0].usd, wk_gate[0][0].usd, rtol=1e-12)
    register("gm_live50_soxx_daily", grid["live50·soxx_d"][0], source=GM + " soxx_d", frozen_on="2026-10-10",
             seen_to="2026-10-09", post_start=P1,
             definition="every session: previous SOXX close < SMA20 -> sell the semis-budget names at that open; "
                        "back above -> re-buy this week's target semis weights (inside remaining cash); rebalance "
                        "days apply the same gate",
             note=f"switches since 10/12={sum(1 for x in grid['live50·soxx_d'][1]['events'] if x['day'] >= P1)}")
    register("gm_live50_qqq_week_allcash", grid["live50·qqq_w"][0], source=GM + " qqq_w", frozen_on="2026-10-10",
             seen_to="2026-10-09", post_start=P1,
             definition="weekly rebalance open: previous QQQ close < SMA20 -> hold nothing that week (all USD cash)",
             note=f"gate events since 9/28={[x for x in grid['live50·qqq_w'][1]['events'] if x['day'] >= '2026-09-28']}")
    register("cap50_adopted_vs_cap60", base, source="docs/desk_system.md §3 (commit 7640010a, 2026-10-10 19:23 KST); "
             "rsi_breakout/report.md 추가 2", frozen_on="2026-10-10", seen_to="2026-10-09", post_start=P1,
             base_curve=grid["c774·none"][0], save=False,
             definition="ADOPTED: semis+unclassified budget 50% (was #774's 60%). Scored as live 50% rule versus the "
                        "60% rule over the same sessions (base columns = 60% rule here)",
             note="adopted rule; 'base' in this row is the 60% cap")
    register("gm_nocap", grid["nocap·none"][0], source=GM + " nocap", frozen_on="2026-10-10", seen_to="2026-10-09",
             post_start=P1, definition="no semis budget (stock 25%, sector 60% only) — the 'concentration is fine' "
             "alternative of 추가 5", note="rejected alternative, shown as contrast")
    register("gm_sub5eq", grid["sub5eq·none"][0], source="rsi_breakout/report.md 추가 4·5 (docs commit 243ae675, "
             "2026-10-10 21:32 KST): '세부·5종목은 동결해 전방 모의로만'", frozen_on="2026-10-10", seen_to="2026-10-09",
             post_start=P1,
             definition="T-sub-N5-eq: each week the 120d-momentum leader of each AI sub-sector bucket; top 5 buckets, "
                        "one name each (keep if bucket rank <= 10 and within-bucket rank <= 2), equal weight, stock "
                        "cap 25%, sector cap 60%, no semis budget",
             note="sub-sector labels are today's watchlist (post-hoc classification, as in the original)")
    for key in ("c774·soxx_w", "c774·soxx_d", "c774·qqq_w", "nocap·soxx_w", "nocap·soxx_d", "nocap·qqq_w",
                "sub5eq·soxx_w", "sub5eq·soxx_d", "sub5eq·qqq_w", "c774·none"):
        lab.add("gm_" + key.replace("·", "_"), grid[key][0], source=GM, frozen_on="2026-10-10", post_start=P1)
    notes["gate_grid"] = {k: dict(post=score(v[0], base, P1, END), jul=score(v[0], base, *JUL),
                                  full=score(v[0], base, *FULL),
                                  post_vs_own_base=score(v[0], grid[k.split("·")[0] + "·none"][0], P1, END))
                          for k, v in grid.items()}

    # ---- rsi_breakout 추가 3: one name per sub-sector on the 50% rule
    led, info = lab.run_weekly(select=selector_sub1(lab))
    repro["sub1 B=0.50 front/back/full/mdd (frozen pre-classification: 2.320,-0.007,2.297,-0.189)"] = dict(
        mine=[oo_stats(info["value"], "2026-01-02", "2026-05-29", lead_one=False)[0],
              oo_stats(info["value"], "2026-06-01", "2026-10-09", lead_one=False)[0],
              *oo_stats(info["value"], "2026-01-02", "2026-10-09", lead_one=False)],
        frozen=[2.320, -0.007, 2.297, -0.189])
    register("sub1_per_subsector_cap50", led, source="rsi_breakout/report.md 추가 3 '관찰' (docs commit b64b2c8a, "
             "2026-10-10 19:48 KST); code deskx_subcount.py", frozen_on="2026-10-10", seen_to="2026-10-09",
             post_start=P1,
             definition="weekly selection walks the 120d ranking keeping at most 1 name per AI sub-sector group "
                        "(buffer: last week's names inside top 12 first), then the usual caps incl. semis 50%",
             note="sub-sector labels are today's watchlist; frozen numbers predate the 56-name reclassification")

    # ---- controls: always 90% / 50%
    register("fixed90_control", lab.overlay(0.9 * ones), source=SR + " ('확실한 보험은 덜 들고 있기')",
             frozen_on="2026-10-10", seen_to="2026-10-09", post_start=P1,
             definition="always 90% of the basket, 10% USD cash (control, not a timing rule)", note="control")

    # ---- observation: semis earnings eve half exposure
    e, used = rule_earnings_eve(lab)
    led = lab.overlay(e)
    v = overlay_value(lab, e)
    repro["earnings eve semis4 gap 0.5 full/back/AIshock"] = dict(
        mine=[*oo_stats(v, "2026-01-02", "2026-10-09"), oo_stats(v, "2026-06-01", "2026-10-09")[0],
              oo_stats(v, "2026-06-22", "2026-07-29")[0]], frozen=[2.481, -0.187, 0.083, -0.162])
    post_ev = [u for u in used if u[2] >= P1]
    register("sr_semis_earnings_eve_half", led, source="stop_research/results/H_events_funds.md '관찰' (표본 12일, 사후)",
             frozen_on="2026-10-10", seen_to="2026-10-09", post_start=P1, force="not_testable",
             definition="50% exposure from the open of the session before the first reaction open of NVDA/AVGO/MU/TSM "
                        "earnings to the reaction open (TSM pre-market, others after close)",
             note=f"post-freeze events (symbol, date, half-exposure session)={post_ev}; NVDA 11/17 reaction open "
                  f"(11/18) is outside the data, so only one complete event")

    # ---- concentration_review primary cap20 (+ its two mechanism controls)
    CR = "concentration_review_20261105/protocol.json (commit 0470f90b, 2026-11-05 22:58 KST)"
    for cid, caps in (("cap20", dict(stock_cap=0.20)), ("cap15", dict(stock_cap=0.15)),
                      ("raw_ceiling25", dict(stock_cap=0.25, raw_ceiling=True))):
        fwd, _ = lab.run_weekly(caps=caps, active_from="2026-11-09")
        hist, _ = lab.run_weekly(caps=caps)
        j0 = days.index("2026-11-09")
        np.testing.assert_allclose(fwd.usd[:j0], base.usd[:j0], rtol=1e-12)
        w = register(f"cr_{cid}_forward", fwd, source=CR, frozen_on="2026-11-05", seen_to="2026-11-04",
                     post_start="2026-11-09", force="not_testable",
                     definition=f"{'PRIMARY ' if cid == 'cap20' else 'mechanism control '}{cid}: {caps}; identical to "
                                f"the live rule before 2026-11-09, applied from the 11/9 open (protocol forward test; "
                                f"decision only at 60 sessions, 20 = interim look)",
                     note="7 sessions since activation; protocol status observation_only")
        lab.add(f"cr_{cid}_whole_year", hist, source=CR, frozen_on="2026-11-05", post_start="2026-11-05")
        notes[f"cr_{cid}_whole_year"] = dict(full=score(hist, base, *FULL), jul=score(hist, base, *JUL),
                                             octnov=score(hist, base, *OCTNOV),
                                             after_data_end=score(hist, base, "2026-11-05", END))

    # ---- ma_entry_filter: front-selected SMA20 new-entry hold, front-selected EMA65 full exit, fixed SMA126 exit
    MA = "ma_entry_filter_20261108"
    for key, kind, length, exits, src, fz, seen, start, tag in (
            ("ma_sma20_entry_hold", "SMA", 20, False, f"{MA}/protocol.json (commit 18ab7888, 2026-11-08 18:33 KST)",
             "2026-11-08", "2026-11-04", "2026-11-09", "front-period pick of 6 pre-registered"),
            ("ma_ema65_stock_exit", "EMA", 65, True, f"{MA}/exit_1_200/protocol.json (commit 74a8ab81, 2026-11-09 "
             "00:35 KST)", "2026-11-09", "2026-11-04", "2026-11-09", "front-period pick of 400"),
            ("ma_sma126_stock_exit", "SMA", 126, True, f"{MA}/cross_stock_20261109 ('고정126일선', post-hoc best of "
             "400 then fixed)", "2026-11-09", "2026-11-04", "2026-11-10", "post-hoc pick fixed on 11/9")):
        ok, ex = ma_flags(lab, kind, length)
        fwd, inf = lab.run_weekly(entry_ok=ok, exit_sig=ex if exits else None, active_from=start)
        hist, hinf = lab.run_weekly(entry_ok=ok, exit_sig=ex if exits else None)
        kv = pd.Series(hist.krw.to_numpy(), index=days)
        fr = kv["2026-05-29"] - 1
        bk = kv["2026-11-04"] / kv["2026-05-29"] - 1
        repro[f"{key} whole-year replay KRW front(1/2~5/29), back(6/1~11/4)"] = dict(
            mine=[float(fr), float(bk)],
            frozen={"ma_sma20_entry_hold": [2.2183, -0.2846], "ma_ema65_stock_exit": [2.2815, -0.3707],
                    "ma_sma126_stock_exit": [2.1602, -0.2426]}[key])
        register(f"{key}_forward", fwd, source=src, frozen_on=fz, seen_to=seen, post_start=start,
                 force="not_testable",
                 definition=f"{tag}: per stock, previous close vs {kind}{length}: new entries only if close >= MA "
                            f"(blocked weight stays cash)" + ("; a held name whose previous close < MA is sold in "
                            "full at the open and re-enters only on a weekly rebalance" if exits else "") +
                            f"; base path before {start}",
                 note=f"events since activation={[x for x in inf['events'] if x['day'] >= start]}")
        lab.add(f"{key}_whole_year", hist, source=src, frozen_on=fz, post_start=start)
        notes[f"{key}_whole_year"] = dict(full=score(hist, base, *FULL), jul=score(hist, base, *JUL),
                                          octnov=score(hist, base, *OCTNOV),
                                          after_data_end=score(hist, base, "2026-11-05", END))

    # ---- the adopted rule itself (fixed 2026-09-27) against the TIME ETF and QQQ
    al = alerts(lab)
    q = ctx.book.cls["QQQ"].reindex(days)
    fxs = pd.Series(ctx.fx, index=days)
    def seg(s, a, b):
        x = s[(s.index >= a) & (s.index <= b)].dropna()
        anchor = s[s.index < a].dropna()
        p0 = anchor.iloc[-1] if len(anchor) else x.iloc[0]
        path = np.r_[p0, x.to_numpy()]
        return float(path[-1] / p0 - 1), float(np.min(path / np.maximum.accumulate(path) - 1)), x.index[-1]
    adopted = {}
    for label, a in (("since_2026-09-28", "2026-09-28"), ("since_2026-10-12", P1), ("octnov", OCTNOV[0])):
        adopted[label] = dict(model_krw=seg(al.model_krw, a, END), time_etf_krw=seg(al.etf_krw, a, END),
                              qqq_krw=seg(q * fxs, a, END), qqq_usd=seg(q, a, END),
                              model_usd=seg(base.usd, a, END), usd_cash_krw=seg(fxs, a, END))
    notes["adopted_rule_vs_benchmarks"] = adopted
    notes["adopted_rule_h1"] = dict(model_krw=seg(al.model_krw, "2026-01-02", "2026-09-25"),
                                    time_etf_krw=seg(al.etf_krw, "2026-01-02", "2026-09-25"))

    # ---- §5 alerts from 2026-09-28
    a2 = al[al.index >= "2026-09-28"]
    alert = {}
    for name in ("a_gap20", "a_dd", "a_gap120"):
        alert[name] = dict(spells=spells(a2[name]), days=int(a2[name].sum()),
                           first=a2.index[a2[name]][0] if a2[name].any() else None)
    alert["min_gap20"] = (float(a2.gap20.min()), a2.gap20.idxmin())
    alert["min_gap120"] = (float(a2.gap120.min()), a2.gap120.idxmin())
    alert["min_dd"] = (float(a2.dd.min()), a2.dd.idxmin())
    alert["usd_dd_min"] = float((base.usd / base.usd.cummax() - 1).min())
    alert["etf_missing_sessions"] = [d for d in a2.index if not np.isfinite(a2.etf_krw[d])]
    what_if = {}
    for name in ("a_gap20", "a_dd", "a_gap120"):
        f = alert[name]["first"]
        if f is None:
            continue
        j = days.index(f)
        if j + 1 >= lab.n:
            continue
        nxt = days[j + 1]                         # first open at which the alert could be acted on
        m_open = pd.Series(ctx.baseline.value, index=days)
        m_ret_usd = float(base.usd.iloc[-1] / m_open[nxt] - 1)
        fx_open = float(ctx.fx[j])                # FX mark of the alert session (KRW value at the previous close)
        what_if[name] = dict(alert_session=f, act_open=nxt,
                             model_usd_from_open_to_end=m_ret_usd,
                             model_krw_from_alert_close=seg(al.model_krw, nxt, END)[0],
                             time_etf_krw_from_alert_close=seg(al.etf_krw, nxt, END)[0],
                             usd_cash_krw_from_alert_close=float(ctx.fx[-1] / ctx.fx[j] - 1),
                             sessions_left=lab.n - j - 1)
    alert["what_if"] = what_if
    notes["alerts"] = alert
    notes["alerts_daily"] = a2.round(4).reset_index().to_dict("records")

    # ---- observation '-15% 아래 매도 금지' (claim: after -15% the model never fell 10% more inside 60 sessions)
    dd_usd = base.usd / base.usd.cummax() - 1
    post = dd_usd[dd_usd.index >= P1]
    hit = post.index[post <= -0.15]
    if len(hit):
        h = hit[0]
        after = base.usd[base.usd.index >= h]
        notes["minus15_no_sell"] = dict(first_touch=h, dd_at_touch=float(dd_usd[h]),
                                        worst_after=float(after.min() / after.iloc[0] - 1), worst_day=after.idxmin(),
                                        to_end=float(after.iloc[-1] / after.iloc[0] - 1), sessions=len(after) - 1,
                                        peak_day=base.usd.idxmax(), peak_to_end=float(dd_usd.iloc[-1]))

    # ---- output
    table = common.save("oos", lab.curves, lab.cfg)
    out = dict(rows=rows, repro=repro, notes=notes)
    (common.CACHE / "oos_prior_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                                        encoding="utf-8")
    pd.set_option("display.width", 250)
    print("== reproduction of frozen claims (open-to-open USD to 2026-10-09 unless noted)")
    for k, v in repro.items():
        print(f"{k:72s} mine {np.round(v['mine'], 4).tolist()}  frozen {v['frozen']}")
    print("\n== rows (USD): post-freeze | Jul 6/23-7/29 | Oct-Nov 10/13-11/17 | full year")
    for r in rows:
        w = r["windows"]
        f = lambda x: f"{x['usd']:+.3f}/{x['usd_mdd']:+.3f} vs {x['base_usd']:+.3f}/{x['base_usd_mdd']:+.3f}"
        p = w["post"]
        print(f"{r['name']:34s} {r['verdict']:12s} post[{p['start']} n={p['sessions']}] {f(p)} ctrl {p['ctrl_usd']:+.3f} "
              f"expo {p['expo']:.2f} td {p['trade_days']} krw {p['krw']:+.3f} vs {p['base_krw']:+.3f} | jul {f(w['jul'])} "
              f"| on {f(w['octnov'])} | full {f(w['full'])}")
    print("\n== gate grid (post-freeze USD ret/mdd, vs own base)")
    for k, v in notes["gate_grid"].items():
        p, o = v["post"], v["post_vs_own_base"]
        print(f"{k:16s} post {p['usd']:+.3f}/{p['usd_mdd']:+.3f} expo {p['expo']:.2f} | own base {o['base_usd']:+.3f}/"
              f"{o['base_usd_mdd']:+.3f} | jul {v['jul']['usd']:+.3f}/{v['jul']['usd_mdd']:+.3f} | full "
              f"{v['full']['usd']:+.3f}/{v['full']['usd_mdd']:+.3f}")
    print("\n== alerts", json.dumps(alert, ensure_ascii=False, default=str, indent=1))
    print("\n== adopted", json.dumps(adopted, ensure_ascii=False, default=str, indent=1))
    print("\n== notes", json.dumps({k: v for k, v in notes.items() if k in ("minus15_no_sell", "adopted_rule_h1")},
                                   ensure_ascii=False, default=str, indent=1))
    print(f"\nsaved family oos: {len(table)} ledgers")
    return out


if __name__ == "__main__":
    main()
