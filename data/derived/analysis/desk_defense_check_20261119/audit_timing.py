"""Look-ahead audit of desk_defense_20261119 (read-only; writes only cache/audit_timing.json).

Run from the repo root:  py data/derived/analysis/desk_defense_check_20261119/audit_timing.py [a b c d e f g]
Each section recomputes Codex's inputs with an independent, explicitly point-in-time
implementation or re-runs the pure functions on truncated / perturbed data.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from bisect import bisect_right
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import CODEX, ROOT, study, S  # noqa: E402
import signals  # noqa: E402  Codex module (pure functions only)
import watchlist  # noqa: E402
from desk import config, giant, trend, validate  # noqa: E402

OUT = {}
RNG = np.random.default_rng(20261119)
NY, KST = ZoneInfo("America/New_York"), ZoneInfo("Asia/Seoul")


def say(*parts):
    print(*parts, flush=True)


def win(curve, a, b, column="krw"):
    """Return and max drawdown of a ledger column inside [a, b], anchored on the prior close."""
    ix = np.flatnonzero((curve.index >= a) & (curve.index <= b))
    before = curve[column].iloc[ix[0] - 1] if ix[0] else 1.0
    v = np.r_[before, curve[column].to_numpy()[ix]]
    return float(v[-1] / before - 1), float(np.min(v / np.maximum.accumulate(v) - 1))


# --------------------------------------------------------------------------- a) market gates
def independent_gates(cls: pd.DataFrame, days: list[str]) -> dict:
    """Explicit state machine: for session j walk only the closes dated before days[j]."""
    out = {"none": np.ones(len(days), dtype=bool)}
    all_days = list(cls.index)
    for symbol in ("QQQ", "SOXX"):
        px = cls[symbol].to_numpy(float)
        for n in (10, 20, 50, 100):
            for conf in ((1, 3) if n in (20, 50) else (1,)):
                state, run, states = None, 0, []
                for i in range(len(px)):
                    if i >= n - 1 and np.isfinite(px[i - n + 1:i + 1]).all():
                        above = px[i] >= px[i - n + 1:i + 1].mean()
                        run = run + 1 if above else 0
                        if not above:
                            state = False
                        elif run >= conf:
                            state = True
                    else:
                        run = 0
                    states.append(state)
                first = all_days.index(days[0])
                # the state known before the open of session j is the state after close j-1
                out[f"{symbol}{n}r{conf}"] = np.array([states[first + j - 1] for j in range(len(days))])
    return out


def section_a(ctx):
    say("\n== a) market gates ==")
    book, days = ctx.book, ctx.days
    ind = independent_gates(book.cls, days)
    mism = {k: int((np.asarray(ind[k], dtype=object) != ctx.gates[k].astype(object)).sum()) for k in ctx.gates}
    undefined = {k: int(sum(v is None for v in ind[k])) for k in ind if k != "none"}
    say("gates vs independent j-1 state machine, mismatching sessions:", sum(mism.values()), mism)
    say("sessions with undefined warm-up state:", undefined)
    # perturbation: scramble every close dated >= days[j]; gate[j] must not move
    first = book.days.index(days[0])
    bad = 0
    for j in range(len(days)):
        cls = book.cls[["QQQ", "SOXX"]].copy()
        cls.iloc[first + j:] = cls.iloc[first + j:].to_numpy() * RNG.uniform(.2, 5., size=(len(cls) - first - j, 2))
        g = study.market_gates(SimpleNamespace(cls=cls), days)
        bad += sum(bool(g[k][j] != ctx.gates[k][j]) for k in g)
        # everything up to j must also be unchanged
        bad += sum(int((g[k][:j + 1] != ctx.gates[k][:j + 1]).sum()) for k in g)
    say(f"future-scramble test over {len(days)} cut sessions x {len(ctx.gates)} gates: changed entries =", bad)
    # control: the same scramble one day earlier (close j-1 scrambled) MUST be able to change gate[j]
    moved = 0
    for j in range(1, len(days)):
        cls = book.cls[["QQQ", "SOXX"]].copy()
        cls.iloc[first + j - 1] = cls.iloc[first + j - 1].to_numpy() * np.where(ctx.gates["SOXX10r1"][j], .2, 5.)
        g = study.market_gates(SimpleNamespace(cls=cls), days)
        moved += int(g["SOXX10r1"][j] != ctx.gates["SOXX10r1"][j])
    say(f"control (scramble close j-1): SOXX10r1 gate[j] flipped on {moved}/{len(days) - 1} sessions")
    warm = {s: int(book.cls[s].loc[:"2025-12-31"].notna().sum()) for s in ("QQQ", "SOXX")}
    say("closes available before 2026-01-02:", warm, "first book day", book.days[0])
    d0 = {k: bool(v[0]) for k, v in ctx.gates.items()}
    say("gate state at the first session 2026-01-02:", d0)
    # how often is each gate 'out'
    share = {k: round(float(1 - v.mean()), 3) for k, v in ctx.gates.items()}
    say("share of sessions out of the market:", share)
    # r3 semantics: exit after 1 close below, re-entry after 3 consecutive closes at/above
    flips = {k: int(np.abs(np.diff(v.astype(int))).sum()) for k, v in ctx.gates.items()}
    say("gate flips over the year:", flips)
    # Independent accounting: a gated ledger is the desk engine's own open/close NAV (S.run_model, a different
    # code path from study.replay) chained over the sessions the gate is on: overnight if on at j-1, intraday if on at j.
    v_open, v_close = ctx.baseline.value, ctx.baseline.close
    chain_err = {}
    for name, gate in ctx.gates.items():
        usd, nav = [], 1.0
        for j in range(len(days)):
            if j and gate[j - 1]:
                nav *= v_open[j] / v_close[j - 1]
            if gate[j]:
                nav *= v_close[j] / v_open[j]
            usd.append(nav)
        led = study.replay(days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, gate)
        chain_err[name] = float(np.max(np.abs(np.array(usd) / led.usd.to_numpy() - 1)))
    say("weekly_<gate> ledgers vs independent NAV chain on the desk engine: max relative error",
        max(chain_err.values()), "(gate with the largest:", max(chain_err, key=chain_err.get) + ")")
    # deliberately leaky variant: apply the gate one session early (state after the close of j used at the open of j)
    leak = {}
    for name in ("SOXX100r1", "SOXX10r1", "QQQ20r1", "SOXX50r1"):
        gate = ctx.gates[name]
        early = np.r_[gate[1:], gate[-1]]
        a = study.replay(days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, gate)
        b = study.replay(days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, early)
        leak[name] = dict(h2_krw_as_coded=win(a, "2026-07-01", "2026-12-31")[0], h2_krw_one_day_lookahead=win(b, "2026-07-01", "2026-12-31")[0],
                          h1_krw_as_coded=win(a, "2026-01-01", "2026-06-30")[0], h1_krw_one_day_lookahead=win(b, "2026-01-01", "2026-06-30")[0])
    say("what one session of look-ahead in the gate WOULD have been worth (not present in Codex's code):")
    say(pd.DataFrame(leak).T.round(4).to_string())
    OUT["a_chain_error"] = chain_err
    OUT["a_leak_value"] = leak
    OUT["a"] = dict(mismatch=mism, undefined_warmup=undefined, future_scramble_changed=bad,
                    control_flips=moved, warm_closes=warm, first_state=d0, out_share=share, flips=flips)


# --------------------------------------------------------------------------- b) eligibility / plan / replay
def section_b(ctx):
    say("\n== b) eligibility, plan, replay ==")
    book, days, tickers = ctx.book, ctx.days, ctx.tickers
    first = book.days.index(days[0])
    missing_cols = [t for t in tickers if t not in book.cls.columns]
    say("tickers missing from book.cls columns (would become NaN -> True):", missing_cols)
    tg = pd.read_csv(CODEX / "trend_grades.csv", index_col=0)
    gg = pd.read_csv(CODEX / "giant_grades.csv", index_col=0)
    say("grade files: index", tg.index[0], "..", tg.index[-1], len(tg), "cols", tg.shape[1],
        "| tickers not in grade columns:", [t for t in tickers if t not in tg.columns])
    mism = {}
    cls = book.cls[tickers].to_numpy(float)
    for name in ctx.ok:
        want = np.zeros((len(days), len(tickers)), dtype=bool)
        for j in range(len(days)):
            p = first + j - 1                      # the last completed session before the open of j
            prev_day = book.days[p]
            if name == "any":
                want[j] = True
            elif name.startswith("sma"):
                n = int(name[3:])
                w = cls[p - n + 1:p + 1]
                with np.errstate(invalid="ignore"):
                    want[j] = np.isfinite(w).all(axis=0) & (cls[p] >= w.mean(axis=0))
            else:
                t = tg.loc[prev_day, tickers].to_numpy(float) if prev_day in tg.index else np.full(len(tickers), np.nan)
                g = gg.loc[prev_day, tickers].to_numpy(float) if prev_day in gg.index else np.full(len(tickers), np.nan)
                with np.errstate(invalid="ignore"):
                    want[j] = {"trendA": t >= 3, "giantA": g >= 3, "bothA": (t >= 3) & (g >= 3),
                               "bothB": (t >= 2) & (g >= 2)}[name]
        mism[name] = int((want != ctx.ok[name]).sum())
    say("eligibility vs independent j-1 computation, mismatching cells:", mism)
    # rolling().mean() with NaN inside the window: pandas gives NaN -> False; same as np.isfinite(w).all()
    say("first-session eligibility counts (2026-01-02, 01-05, 01-06):",
        {k: [int(v[j].sum()) for j in (0, 1, 2)] for k, v in ctx.ok.items()})
    say("rankings: sessions with empty ranking:", [days[j] for j, r in enumerate(ctx.rankings) if not r])
    say("first weekly target session:", days[min(ctx.weekly)], "weekly targets:", len(ctx.weekly))

    # momentum_ranking: scramble adj closes dated >= session and drop PDFs dated >= session
    bad_rank = 0
    for j, day in enumerate(days):
        adj = book.adj.copy()
        block = adj.iloc[first + j:]
        adj.iloc[first + j:] = block.to_numpy() * RNG.uniform(.2, 5., size=block.shape)
        cut = [p for p in book.pdfs if p < day]
        b2 = dataclasses.replace(book, adj=adj, pdfs=cut, snaps={p: book.snaps[p] for p in cut}, _w={})
        pdf, rows = S.momentum_ranking(b2, first + j, day)
        same = [(r["key"], round(r["mom120"], 12), r["time_weight"]) for r in rows] == \
               [(r["key"], round(r["mom120"], 12), r["time_weight"]) for r in ctx.rankings[j]]
        bad_rank += not same
    say(f"momentum_ranking future-scramble (adj closes >= session, PDFs >= session removed): changed sessions = {bad_rank}/{len(days)}")

    moved = 0
    for j in range(1, len(days)):
        adj = book.adj.copy()
        row = adj.iloc[first + j - 1].to_numpy(float)
        adj.iloc[first + j - 1] = row * RNG.uniform(.2, 5., size=row.shape)
        got = S.momentum_ranking(dataclasses.replace(book, adj=adj, _w={}), first + j, days[j])[1]
        moved += [r["key"] for r in got] != [r["key"] for r in ctx.rankings[j]]
    say(f"control (scramble adj close j-1): ranking order changed on {moved}/{len(days) - 1} sessions")
    firsts = {}
    for mode, freq, method, cond in study.selectors():
        t = study.plan(book, days, ctx.rankings, ctx.col, ctx.ok[cond], freq, method)
        n = [int((w > 0).sum()) for w in t.values()]
        firsts[mode] = dict(first=days[min(t)], publications=len(t), names_first=n[0], names_median=float(np.median(n)),
                            empty_publications=int(sum(x == 0 for x in n)), mean_weight_sum=float(np.mean([w.sum() for w in t.values()])))
    say(pd.DataFrame(firsts).T.to_string())
    OUT["b_selectors"] = firsts

    # plan: scramble eligibility rows and rankings after cut J; targets at <= J must be identical
    bad_plan = checked = 0
    cuts = list(range(5, len(days) - 1, 9))
    for mode, freq, method, cond in study.selectors():
        ref = study.plan(book, days, ctx.rankings, ctx.col, ctx.ok[cond], freq, method)
        for J in cuts:
            elig = ctx.ok[cond].copy()
            elig[J + 1:] = RNG.random(elig[J + 1:].shape) < .5
            ranks = list(ctx.rankings[:J + 1]) + [list(reversed(r)) for r in ctx.rankings[J + 1:]]
            got = study.plan(book, days, ranks, ctx.col, elig, freq, method)
            keys_ref = sorted(k for k in ref if k <= J)
            keys_got = sorted(k for k in got if k <= J)
            checked += 1
            if keys_ref != keys_got or any(not np.array_equal(ref[k], got[k]) for k in keys_ref):
                bad_plan += 1
    say(f"plan future-scramble: {checked} (selector x cut) cases, changed = {bad_plan}")

    # replay: scramble opens/closes/fx/gate/targets after cut J; ledger rows <= J must be identical
    bad_replay = checked = 0
    for gate_name in ("none", "SOXX100r1", "QQQ20r3", "SOXX10r1"):
        gate = ctx.gates[gate_name]
        ref = study.replay(days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, gate)
        for J in cuts:
            opn, close, fx, g2 = ctx.opn.copy(), ctx.close.copy(), ctx.fx.copy(), gate.copy()
            opn[J + 1:] *= RNG.uniform(.5, 2., size=opn[J + 1:].shape)
            close[J + 1:] *= RNG.uniform(.5, 2., size=close[J + 1:].shape)
            fx[J + 1:] *= 1.3
            g2[J + 1:] = ~g2[J + 1:]
            targets = {k: (v if k <= J else v[::-1].copy()) for k, v in ctx.weekly.items()}
            try:
                got = study.replay(days, opn, close, fx, targets, g2)
            except ValueError:
                got = None
            checked += 1
            if got is None:
                # reversed targets can hit a missing open later; rerun without target scramble
                got = study.replay(days, opn, close, fx, ctx.weekly, g2)
            if not np.allclose(ref.iloc[:J + 1].to_numpy(), got.iloc[:J + 1].to_numpy(), rtol=0, atol=0):
                bad_replay += 1
    say(f"replay future-scramble: {checked} (gate x cut) cases, changed = {bad_replay}")

    # Same-session information: does any fill use the close of its own session? Reprice the base
    # with trades at the open (as coded) versus a deliberately leaky fill at the previous close.
    base = study.replay(days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, ctx.gates["none"])
    # overnight gap paid by the model on rebalance days (open j vs close j-1), basket-weighted
    gaps = []
    for j, w in ctx.weekly.items():
        if j == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            r = ctx.opn[j] / ctx.close[j - 1] - 1
        gaps.append(float(np.nansum(w * np.nan_to_num(r))))
    say(f"weekly rebalances: {len(gaps)}; mean basket overnight gap into the fill {np.mean(gaps):+.4%}, "
        f"sum {np.sum(gaps):+.4%} (the model pays this, i.e. it does NOT fill at the signal close)")
    OUT["b"] = dict(missing_cols=missing_cols, eligibility_mismatch=mism, ranking_changed=bad_rank,
                    plan_changed=bad_plan, replay_changed=bad_replay,
                    first_target=days[min(ctx.weekly)], overnight_gap_sum=float(np.sum(gaps)),
                    base_h1=win(base, "2026-01-01", "2026-06-30"), base_h2=win(base, "2026-07-01", "2026-12-31"))


# --------------------------------------------------------------------------- e) TIME PDF availability
def us_open_kst(day: str) -> datetime:
    y, m, d = map(int, day.split("-"))
    return datetime(y, m, d, 9, 30, tzinfo=NY).astimezone(KST)


def section_e(ctx):
    say("\n== e) TIME PDF availability ==")
    book, days = ctx.book, ctx.days
    first = book.days.index(days[0])
    rows = []
    for j, day in enumerate(days):
        pdf = S.latest_before(book.pdfs, day)
        if pdf is None:
            rows.append((day, None, None, None, None))
            continue
        snap = book.snaps[pdf]
        got = datetime.fromisoformat(snap["retrieved_at"]) if snap.get("retrieved_at") else None
        y, m, d = map(int, pdf.split("-"))
        published = datetime(y, m, d, 21, 45, tzinfo=KST)          # docs/desk_system.md section 3
        rows.append((day, pdf, (pd.Timestamp(day) - pd.Timestamp(pdf)).days,
                     (us_open_kst(day) - published).total_seconds() / 3600,
                     None if got is None else (got - us_open_kst(day)).total_seconds() / 86400))
    df = pd.DataFrame(rows, columns=["session", "pdf", "lag_days", "hours_public_before_open", "retrieved_after_open_days"])
    used = df.dropna(subset=["pdf"])
    say("sessions:", len(df), "| with a PDF:", len(used), "| PDF dated >= session:", int((used.pdf >= used.session).sum()))
    say("calendar-day lag PDF date -> session:", used.lag_days.value_counts().sort_index().to_dict())
    say("hours between the documented publication (21:45 KST on the PDF date) and the US open that uses it: min %.1f median %.1f"
        % (used.hours_public_before_open.min(), used.hours_public_before_open.median()))
    # the freshest PDF that was public before the open (same-date PDF, 21:45 KST < 22:30/23:30 KST open)
    same_day = sum(1 for d in used.session if d in book.snaps)
    say(f"sessions where a same-date PDF exists and was public ~45-105 min before the open but is NOT used: {same_day}")
    back = used.retrieved_after_open_days
    say("PDF retrieved after the open of the session that uses it (back-filled):", int((back > 0).sum()), "of", len(used),
        "| retrieved before that open (live):", int((back <= 0).sum()))
    weekly_sessions = [days[j] for j in ctx.weekly]
    wk = used[used.session.isin(weekly_sessions)]
    say("weekly rebalance sessions:", len(wk), "| using a back-filled PDF:", int((wk.retrieved_after_open_days > 0).sum()))
    live = used[back <= 0]
    say("first session whose PDF was captured live:", None if live.empty else live.session.iloc[0],
        "| live sessions in H2:", int((live.session >= "2026-07-01").sum()), "of", int((used.session >= "2026-07-01").sum()))
    stamp = pd.Series({p: book.snaps[p].get("retrieved_at", "")[:10] for p in book.pdfs})
    say("retrieval-date histogram of the TIME snapshots (top 8):", stamp.value_counts().head(8).to_dict())
    # git: when did each normalized TIME file first enter the repo, and was it rewritten later?
    try:
        log = subprocess.run(["git", "-C", str(ROOT), "log", "--format=@%cs", "--name-only", "--", "data/normalized/etf_holdings/"],
                             capture_output=True, text=True, encoding="utf-8", check=True).stdout
        commits, current = {}, None
        for line in log.splitlines():
            if line.startswith("@"):
                current = line[1:]
            elif line.endswith("_456600.json"):
                commits.setdefault(line.rsplit("/", 1)[-1][:10], []).append(current)
        touched = {p: sorted(set(v)) for p, v in commits.items() if p <= "2026-11-17"}
        multi = {p: v for p, v in touched.items() if len(v) > 1}
        say("TIME snapshot files in git:", len(touched), "| committed on more than one date:", len(multi),
            "| earliest commit date:", min(v[0] for v in touched.values()))
        rewrite_dates = pd.Series([v[-1] for v in multi.values()]).value_counts().head(5).to_dict()
        say("last-rewrite dates of files with several commits:", rewrite_dates)
        OUT["e_git"] = dict(files=len(touched), multi_commit=len(multi), rewrite_dates=rewrite_dates)
    except Exception as exc:  # git is optional evidence
        say("git log unavailable:", exc)
    # survivorship: held names dropped from the ranking because today's price file has no usable history
    dropped = {}
    for j, day in enumerate(days):
        pdf = S.latest_before(book.pdfs, day)
        if pdf is None:
            continue
        ranked = {r["key"] for r in ctx.rankings[j]}
        for key, w in book.weights(pdf).items():
            if w > 0 and key not in ranked:
                dropped.setdefault(key, []).append((day, w))
    summary = {k: dict(ticker=book.ticker[k], sessions=len(v), first=v[0][0], last=v[-1][0],
                       max_weight=round(max(w for _, w in v), 4)) for k, v in dropped.items()}
    say("TIME US holdings excluded from the ranking for lack of a 120-session price history (count of names):", len(summary))
    for k, v in sorted(summary.items(), key=lambda kv: -kv[1]["sessions"])[:25]:
        px = book.adj[v["ticker"]] if v["ticker"] in book.adj.columns else None
        first_px = None if px is None or px.dropna().empty else px.dropna().index[0]
        say("   ", k, v, "| first price in book:", first_px)
    # Is a PDF dated D really public before the US open of D? Evidence: live captures stamped on D itself.
    early = []
    for p in book.pdfs:
        got = book.snaps[p].get("retrieved_at")
        if got and got[:10] == p and datetime.fromisoformat(got) < us_open_kst(p):
            early.append((p, got[11:16]))
    say(f"PDFs captured on their own date before that date's US open: {len(early)}; earliest clock times (KST):",
        sorted(t for _, t in early)[:5])
    # Sensitivity to the publication lag: 0 = same-date PDF (public before the open per the captures above),
    # 1 = as coded (strictly earlier date), 2 = PDF dated before the previous US session.
    lag_rows = {}
    for lag, label in ((0, "lag 0: PDF dated <= session"), (1, "lag 1: as coded"), (2, "lag 2: PDF dated < previous session")):
        ranks = []
        for j, day in enumerate(days):
            session = day + "~" if lag == 0 else day if lag == 1 else book.days[first + j - 1]
            ranks.append(S.momentum_ranking(book, first + j, session)[1])
        targets = study.plan(book, days, ranks, ctx.col, ctx.ok["any"], "week", "rotate")
        led = study.replay(days, ctx.opn, ctx.close, ctx.fx, targets, ctx.gates["none"])
        lag_rows[label] = dict(h1_krw=win(led, "2026-01-01", "2026-06-30")[0], h2_krw=win(led, "2026-07-01", "2026-12-31")[0],
                               h2_usd=win(led, "2026-07-01", "2026-12-31", "usd")[0],
                               h2_usd_mdd=win(led, "2026-07-01", "2026-12-31", "usd")[1], first_trade=days[min(targets)])
    say(pd.DataFrame(lag_rows).T.to_string())
    OUT["e_lag"] = lag_rows
    OUT["e_same_date_captures"] = len(early)
    OUT["e"] = dict(pdf_ge_session=int((used.pdf >= used.session).sum()), lag_hist=used.lag_days.value_counts().sort_index().to_dict(),
                    min_hours_public=float(used.hours_public_before_open.min()), same_day_unused=same_day,
                    backfilled_sessions=int((back > 0).sum()), live_sessions=int((back <= 0).sum()),
                    weekly_backfilled=int((wk.retrieved_after_open_days > 0).sum()), weekly_total=len(wk),
                    dropped=summary)


# --------------------------------------------------------------------------- f) FX
def section_f(ctx):
    say("\n== f) FX timing ==")
    book, days = ctx.book, ctx.days
    first = book.days.index(days[0])
    raw = pd.read_csv(CODEX / "frozen" / "prices_adj_close.csv", index_col=0, parse_dates=True)["KRW=X"]
    idx = raw.dropna().index.strftime("%Y-%m-%d")
    missing = [d for d in book.days[first - 1:] if d not in set(idx)]
    say("US sessions without a same-date KRW=X print (forward-filled):", missing)
    say("KRW=X prints on non-US-session dates in 2026 (ignored):", int(sum((d >= '2026-01-01') and d not in set(book.days) for d in idx)))
    fx_abs = book.fx[first - 1:]
    say(f"fx anchor (close before first session {book.days[first - 1]}): {fx_abs[0]:.2f}; 06-30: {fx_abs[1 + days.index('2026-06-30')]:.2f}; "
        f"last: {fx_abs[-1]:.2f}")
    base = study.replay(days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, ctx.gates["none"])
    res = {}
    for label, shift in (("same-date KRW=X (as coded)", 0), ("previous-date KRW=X (strictly known at the US close)", -1),
                         ("next-date KRW=X", 1)):
        pos = np.clip(np.arange(first, first + len(days)) + shift, 0, len(book.fx) - 1)
        fx = book.fx[pos] / book.fx[first - 1]
        led = base.assign(krw=base.usd * fx)
        res[label] = dict(h1=win(led, "2026-01-01", "2026-06-30"), h2=win(led, "2026-07-01", "2026-12-31"),
                          full=win(led, "2026-01-01", "2026-12-31"))
        say(f"  {label}: H1 {res[label]['h1'][0]:+.4%}  H2 {res[label]['h2'][0]:+.4%} (mdd {res[label]['h2'][1]:+.4%})  full {res[label]['full'][0]:+.4%}")
    daily = pd.Series(book.fx[first:] / book.fx[first - 1:-1] - 1, index=days)
    say(f"KRW=X daily move: mean abs {daily.abs().mean():.3%}, max abs {daily.abs().max():.3%}")
    usd = {w: win(base, a, b, "usd")[0] for w, (a, b) in common.WINDOWS.items()}
    krw = {w: win(base, a, b, "krw")[0] for w, (a, b) in common.WINDOWS.items()}
    say("base USD by window:", {k: round(v, 4) for k, v in usd.items()})
    say("base KRW by window:", {k: round(v, 4) for k, v in krw.items()})
    cash = study.replay(days, ctx.opn, ctx.close, ctx.fx, {}, ctx.gates["none"])
    say("USD cash in KRW: H1 %+.4f H2 %+.4f" % (win(cash, "2026-01-01", "2026-06-30")[0], win(cash, "2026-07-01", "2026-12-31")[0]))
    flat = study.replay(days, ctx.opn, ctx.close, np.ones(len(days)), ctx.weekly, ctx.gates["SOXX100r1"])
    real = study.replay(days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, ctx.gates["SOXX100r1"])
    say("replay with fx == 1: usd/turn/exposure identical to the real-fx ledger:",
        bool(np.array_equal(flat[["usd", "turn", "exposure"]].to_numpy(), real[["usd", "turn", "exposure"]].to_numpy())))
    OUT["f"] = dict(missing_fx_days=missing, sensitivity=res, base_usd=usd, base_krw=krw)


# --------------------------------------------------------------------------- c) grade causality
def load_bars(symbols):
    """Read Codex's bar cache directly (never signals.bars_for, which downloads when a file is missing)."""
    cache = CODEX / "cache" / "grade_bars"
    missing = sorted(s for s in symbols if not (cache / f"{s}.parquet").exists())
    return {s: pd.read_parquet(cache / f"{s}.parquet") for s in sorted(symbols) if s not in missing}, missing


def cut_bars(bars, t):
    return {s: b.loc[:t] for s, b in bars.items() if len(b.loc[:t])}


def scale_after(bars, t, factor=None):
    """Bars dated after t multiplied by 3 (factor=None -> an independent random level per bar)."""
    out = {}
    for s, b in bars.items():
        b = b.copy()
        later = b.index > t
        if later.any():
            f = 3.0 if factor is None else RNG.uniform(.2, 5., size=int(later.sum()))[:, None]
            b.loc[later, ["High", "Close"]] = b.loc[later, ["High", "Close"]].to_numpy() * f
        out[s] = b
    return out


def cut_histories(histories, day):
    return {fund: ([d for d in ds if d <= day], {d: snaps[d] for d in ds if d <= day})
            for fund, (ds, snaps) in histories.items()}


def same(a: pd.Series, b: pd.Series) -> int:
    a, b = a.to_numpy(float), b.to_numpy(float)
    return int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())


def section_c(ctx):
    say("\n== c) trend / giant grade causality ==")
    book, days = ctx.book, ctx.days
    symbols = sorted(set(book.ticker.values()))
    histories, held = signals.holdings_history()
    need = held | set(symbols) | set(watchlist.AI_UNIVERSE_ALIASES.values()) | {"QQQ", giant.ANCHOR}
    bars, missing = load_bars(need)
    say("symbols needed:", len(need), "| missing from Codex's bar cache:", missing)
    saved_t = pd.read_csv(CODEX / "trend_grades.csv", index_col=0)
    saved_g = pd.read_csv(CODEX / "giant_grades.csv", index_col=0)
    # 1. full-data reproduction of every saved row
    full_t, stats = signals.grade_trend(book, bars, histories, days)
    full_g = signals.grade_giant(symbols, bars, days)
    rep_t = sum(same(full_t.loc[d, saved_t.columns], saved_t.loc[d]) for d in days)
    rep_g = sum(same(full_g.loc[d, saved_g.columns], saved_g.loc[d]) for d in days)
    say(f"full recompute vs saved CSV: trend mismatching cells {rep_t}/{saved_t.size}, giant {rep_g}/{saved_g.size}",
        "| trend non-empty", int(saved_t.count().sum()), "giant non-empty", int(saved_g.count().sum()))
    # 2. truncation and future-scaling on dates spread over the year
    picks = sorted(set(days[::9]) | {days[0], days[-1], "2026-06-30", "2026-07-01", "2026-07-27", "2026-09-22", "2026-10-09"})
    res = dict(trend_cut=0, trend_x3=0, trend_rand=0, giant_cut=0, giant_x3=0, giant_rand=0,
               trend_control=0, giant_control=0, cells_t=0, cells_g=0, raised=[])
    for day in picks:
        t = pd.Timestamp(day)
        variants = {"cut": (cut_bars(bars, t), cut_histories(histories, day)),
                    "x3": (scale_after(bars, t), histories),
                    "rand": (scale_after(bars, t, "random"), histories)}
        for label, (b, h) in variants.items():
            try:
                got_t, _ = signals.grade_trend(book, b, h, [day])
                res[f"trend_{label}"] += same(got_t.loc[day, saved_t.columns], saved_t.loc[day])
            except Exception as exc:
                res["raised"].append((day, "trend", label, repr(exc)))
            got_g = signals.grade_giant(symbols, b, [day])
            res[f"giant_{label}"] += same(got_g.loc[day, saved_g.columns], saved_g.loc[day])
        res["cells_t"] += saved_t.shape[1]
        res["cells_g"] += saved_g.shape[1]
        # positive control: move the bar OF the signal date itself (-30% on a random half of the symbols)
        ctrl = {}
        for s, b in bars.items():
            b = b.copy()
            if t in b.index and RNG.random() < .5:
                b.loc[t, ["High", "Close"]] = b.loc[t, ["High", "Close"]].to_numpy() * np.array([1., .7])
            ctrl[s] = b
        got_t, _ = signals.grade_trend(book, ctrl, histories, [day])
        res["trend_control"] += same(got_t.loc[day, saved_t.columns], saved_t.loc[day])
        got_g = signals.grade_giant(symbols, ctrl, [day])
        res["giant_control"] += same(got_g.loc[day, saved_g.columns], saved_g.loc[day])
    say(f"dates tested: {len(picks)} ({picks[0]} .. {picks[-1]}); cells per test: trend {res['cells_t']}, giant {res['cells_g']}")
    say("mismatching cells after removing every later bar and every later PDF: trend", res["trend_cut"], "giant", res["giant_cut"])
    say("mismatching cells after multiplying later bars by 3: trend", res["trend_x3"], "giant", res["giant_x3"])
    say("mismatching cells after random rescaling of later bars: trend", res["trend_rand"], "giant", res["giant_rand"])
    say("positive control (signal-date close -30% on half the symbols) changed cells: trend", res["trend_control"],
        "giant", res["giant_control"])
    say("exceptions:", res["raised"])
    # 3. which information defines the comparison set
    ever_t = int(saved_t.notna().any().sum())
    ever_g = int(saved_g.notna().any().sum())
    in_map = [s for s in symbols if watchlist.AI_UNIVERSE_ALIASES.get(s, s) in set(watchlist.AI_SUBSECTOR)]
    say(f"traded symbols {len(symbols)}: ever trend-graded {ever_t}, ever giant-graded {ever_g}, in today's AI map {len(in_map)}")
    per_day_t = saved_t.notna().sum(axis=1)
    per_day_g = saved_g.notna().sum(axis=1)
    say(f"graded names per day: trend min {per_day_t.min()} median {per_day_t.median():.0f} max {per_day_t.max()}; "
        f"giant min {per_day_g.min()} median {per_day_g.median():.0f} max {per_day_g.max()}")
    a_t, a_g = (saved_t >= 3).sum(axis=1), (saved_g >= 3).sum(axis=1)
    say(f"names graded A or better per day: trend median {a_t.median():.0f} (min {a_t.min()}), giant median {a_g.median():.0f} "
        f"(min {a_g.min()}, days with zero: {int((a_g == 0).sum())})")
    # anchor used by the giant grade through the year (changes only with information up to the date)
    anchors = pd.DataFrame([giant.find_anchor(bars[giant.ANCHOR].loc[:pd.Timestamp(d), "Close"]) for d in days], index=days)
    spells = anchors[["peak_date", "low_date"]].drop_duplicates()
    say("SOXX anchor (peak, low) spells as seen live:", len(spells))
    say(anchors.groupby(["peak_date", "low_date"]).apply(lambda g: f"{g.index[0]}..{g.index[-1]} n={len(g)}").to_string())
    say("anchor low dated after the signal date:", int((anchors.low_date > anchors.index).sum()))
    OUT["c"] = {k: v for k, v in res.items()} | dict(dates=picks, full_repro_trend=rep_t, full_repro_giant=rep_g,
                                                    ever_trend=ever_t, ever_giant=ever_g, in_ai_map=len(in_map),
                                                    coverage=stats)


# --------------------------------------------------------------------------- d) breadth
def section_d(ctx):
    say("\n== d) breadth ==")
    days = ctx.days
    saved = pd.read_csv(CODEX / "breadth.csv", index_col=0).breadth
    universe = sorted(watchlist.AI_SUBSECTOR)
    bars, missing = load_bars(set(universe) | {"QQQ"})
    say("AI universe:", len(universe), "| missing bars:", missing, "| breadth rows:", len(saved), saved.index[0], "..", saved.index[-1])
    picks = sorted(set(saved.index[::9]) | {saved.index[0], saved.index[-1], "2026-06-30", "2026-07-27", "2026-10-09"})
    worst = dict(cut=0., x3=0., rand=0.)
    for day in picks:
        t = pd.Timestamp(day)
        for label, b in (("cut", cut_bars(bars, t)), ("x3", scale_after(bars, t)), ("rand", scale_after(bars, t, "random"))):
            got = float(giant.breadth_series(b, universe, pd.DatetimeIndex([t])).iloc[0])
            worst[label] = max(worst[label], abs(got - float(saved.loc[day])))
    say(f"breadth truncation test on {len(picks)} dates: max abs difference (pct points)", worst)
    # the gate exactly as study.diagnostics builds it, and an explicit j-1 version
    coded = (saved.shift().reindex(days) >= 19).to_numpy()
    first = list(saved.index).index(days[0])
    explicit = np.array([saved.iloc[first + j - 1] >= 19 for j in range(len(days))])
    say("diagnostics gate vs explicit previous-close lookup: mismatches", int((coded != explicit).sum()),
        "| date feeding the first session:", saved.index[first - 1])
    args = (days, ctx.opn, ctx.close, ctx.fx)
    base = study.replay(*args, ctx.weekly, ctx.gates["none"])
    led = study.replay(*args, ctx.weekly, coded)
    windows = {"h1": ("2026-01-01", "2026-06-30"), "h2": ("2026-07-01", "2026-12-31"),
               "jan (before calibration)": ("2026-01-01", "2026-01-31"),
               "feb-sep (calibration period)": ("2026-02-01", "2026-09-30"),
               "jul-sep (H2 part inside calibration)": ("2026-07-01", "2026-09-30"),
               "oct-nov (after calibration)": ("2026-10-06", "2026-12-31")}
    rows = {}
    for w, (a, b) in windows.items():
        ix = (led.index >= a) & (led.index <= b)
        rows[w] = dict(gate_krw=win(led, a, b)[0], gate_usd=win(led, a, b, "usd")[0], gate_usd_mdd=win(led, a, b, "usd")[1],
                       base_krw=win(base, a, b)[0], base_usd=win(base, a, b, "usd")[0],
                       sessions=int(ix.sum()), in_market=int(coded[ix].sum()))
    say(pd.DataFrame(rows).T.round(4).to_string())
    spells = []
    j = 0
    while j < len(days):
        k = j
        while k + 1 < len(days) and coded[k + 1] == coded[j]:
            k += 1
        spells.append((days[j], days[k], "in" if coded[j] else "out", k - j + 1))
        j = k + 1
    say("breadth gate spells:", spells)
    say("breadth at selected closes:", {d: round(float(saved.loc[d]), 1) for d in
                                        ("2025-12-31", "2026-03-31", "2026-06-30", "2026-07-27", "2026-09-22", "2026-10-09", "2026-11-17")})
    sens = {}
    for thr in (10, 13, 15, 17, 18, 19, 20, 21, 23, 25, 30):
        g = (saved.shift().reindex(days) >= thr).to_numpy()
        x = study.replay(*args, ctx.weekly, g)
        sens[thr] = dict(h1=win(x, *windows["h1"])[0], h2=win(x, *windows["h2"])[0], h2_usd=win(x, *windows["h2"], "usd")[0],
                         h2_in_market=int(g[np.array(days) >= "2026-07-01"].sum()),
                         post=win(x, *windows["oct-nov (after calibration)"], "usd")[0])
    say("threshold sensitivity (KRW h1, KRW h2, USD h2, H2 sessions in market, USD after 10-06):")
    say(pd.DataFrame(sens).T.round(4).to_string())
    g18 = (saved.shift().reindex(days) >= 18).to_numpy()
    extra = [j for j in range(len(days)) if g18[j] and not coded[j] and days[j] >= "2026-07-01"]
    usd = base.usd.to_numpy()
    say("H2 sessions in the market at 18% but not at 19% (session, breadth at prior close, base USD move of that session incl. gap):",
        [(days[j], round(float(saved.shift().reindex(days).iloc[j]), 2), round(float(usd[j] / usd[j - 1] - 1), 4)) for j in extra])
    b_prev = saved.shift().reindex(days)
    h2 = b_prev[b_prev.index >= "2026-07-01"]
    say("H2 prior-close breadth distribution: min %.1f median %.1f max %.1f; sessions within 1 point of 19%%: %d"
        % (h2.min(), h2.median(), h2.max(), int(((h2 - 19).abs() <= 1).sum())))
    say("BREADTH_CHECK:", giant.BREADTH_CHECK)
    OUT["d"] = dict(truncation_max_diff=worst, gate_mismatch=int((coded != explicit).sum()), windows=rows,
                    spells=spells, sensitivity=sens)


# --------------------------------------------------------------------------- g) design hindsight
class rule:
    """Temporarily replace config.RULE entries (in memory only)."""

    def __init__(self, **changes):
        self.changes = changes

    def __enter__(self):
        self.saved = dict(config.RULE)
        config.RULE.update(self.changes)

    def __exit__(self, *exc):
        config.RULE.clear()
        config.RULE.update(self.saved)


G_WINDOWS = {"h1": ("2026-01-01", "2026-06-30"), "h2": ("2026-07-01", "2026-12-31"),
             "jul01-sep23 (inside #774 sample)": ("2026-07-01", "2026-09-23"),
             "sep24-nov17 (after #774 sample)": ("2026-09-24", "2026-12-31"),
             "oct12-nov17 (after 50% cap)": ("2026-10-12", "2026-12-31")}


def row_of(led):
    out = {}
    for w, (a, b) in G_WINDOWS.items():
        out[f"{w} krw"] = win(led, a, b)[0]
        out[f"{w} usd"] = win(led, a, b, "usd")[0]
        out[f"{w} usd_mdd"] = win(led, a, b, "usd")[1]
    return out


def section_g(ctx):
    say("\n== g) design hindsight ==")
    book, days = ctx.book, ctx.days
    first = book.days.index(days[0])
    args = (days, ctx.opn, ctx.close, ctx.fx)
    none = ctx.gates["none"]

    def weekly_plan():
        return study.plan(book, days, ctx.rankings, ctx.col, ctx.ok["any"], "week", "rotate")

    variants = {"current rule: semis 50%, today's sector labels": ctx.weekly}
    with rule(semis_cap=.60):
        variants["semis 60%, today's labels"] = weekly_plan()
    old_labels = validate.codex_sectors(book)
    changed = {k: (old_labels[k], book.sector[k]) for k in book.sector if old_labels[k] != book.sector[k]}
    unknown_then = sum(1 for v in old_labels.values() if v == S.UNKNOWN)
    unknown_now = sum(1 for v in book.sector.values() if v == S.UNKNOWN)
    say(f"sector labels: {len(changed)} of {len(book.sector)} TIME US names differ between the 2026-09-27 map and today; "
        f"unclassified then {unknown_then}, now {unknown_now}")
    saved_sector = book.sector
    try:
        book.sector = old_labels
        with rule(semis_cap=.60):
            variants["#774 as chosen 09-27: semis 60%, 09-27 labels"] = weekly_plan()
        variants["semis 50%, 09-27 labels"] = weekly_plan()
    finally:
        book.sector = saved_sector
    cut = days.index("2026-10-12")
    lived = {j: w for j, w in variants["#774 as chosen 09-27: semis 60%, 09-27 labels"].items() if j < cut}
    lived.update({j: w for j, w in ctx.weekly.items() if j >= cut})
    variants["as documented live: 60%/09-27 labels until 10-09, 50%/today's labels from 10-12"] = lived
    table = {name: row_of(study.replay(*args, targets, none)) for name, targets in variants.items()}
    say(pd.DataFrame(table).T.round(4).to_string())
    held_changed = {}
    for j, w in ctx.weekly.items():
        for key, i in ctx.col.items():
            if w[i] > 0 and key in changed:
                held_changed.setdefault(key, 0)
                held_changed[key] += 1
    say("weekly baskets containing a name whose label changed after 09-27 (name: weeks held):",
        {book.ticker[k]: (n, changed[k]) for k, n in sorted(held_changed.items(), key=lambda kv: -kv[1])})

    # neighbours of the chosen rule: how special is n=7 / 120 sessions in each half?
    grid = {}
    for look in (60, 90, 120):
        for n in (5, 7, 10):
            with rule(lookback=look, n=n):
                ranks = [S.momentum_ranking(book, first + j, day)[1] for j, day in enumerate(days)]
                targets = study.plan(book, days, ranks, ctx.col, ctx.ok["any"], "week", "rotate")
            led = study.replay(*args, targets, none)
            grid[f"lookback {look}, n {n}"] = dict(h1_krw=win(led, *G_WINDOWS["h1"])[0], h2_krw=win(led, *G_WINDOWS["h2"])[0],
                                                   h2_usd=win(led, *G_WINDOWS["h2"], "usd")[0],
                                                   h2_usd_mdd=win(led, *G_WINDOWS["h2"], "usd")[1],
                                                   post_usd=win(led, *G_WINDOWS["sep24-nov17 (after #774 sample)"], "usd")[0])
    assert config.RULE == {"lookback": 120, "n": 7, "buffer": 5, "stock_cap": .25, "sector_cap": .60, "semis_cap": .50}
    say("neighbour rules (weekly, TIME weights, current caps):")
    say(pd.DataFrame(grid).T.round(4).to_string())
    # benchmarks over the same windows
    bench = {}
    for sym in ("QQQ", "SOXX"):
        o, c = book.opn[sym], book.cls[sym]
        bench[sym] = {w: float(c.loc[:b].iloc[-1] / c.loc[:a].iloc[-2 if a in c.index else -1] - 1) if False else
                      float(c[(c.index <= b)].iloc[-1] / c[c.index < a].iloc[-1] - 1) for w, (a, b) in G_WINDOWS.items()}
    say("close-to-close USD benchmarks:", {k: {w: round(v, 4) for w, v in d.items()} for k, d in bench.items()})

    # Codex's own candidates in the windows that postdate each design decision
    curves, configs = common.codex_curves(ctx)
    post = {"sep24+": ("2026-09-24", "2026-12-31"), "oct06+": ("2026-10-06", "2026-12-31"),
            "oct12+": ("2026-10-12", "2026-12-31"), "h2": ("2026-07-01", "2026-12-31")}
    keys = ["base", "half", "stock100_cash_SOXX100r1", "weekly_SOXX100r1", "weekly_SOXX10r1", "switch_bothA_QQQ50r1",
            "trendA_week_rotate_none", "trendA_change_cash_none", "giantA_change_cash_none", "giantA_week_rotate_none",
            "bothA_change_cash_SOXX10r1"]
    rows = {k: {f"{w} {c}": win(curves[k], a, b, c)[0] for w, (a, b) in post.items() for c in ("krw", "usd")} |
               {"h2 expo": float(curves[k].loc["2026-07-01":, "exposure"].mean()),
                "oct12+ expo": float(curves[k].loc["2026-10-12":, "exposure"].mean())} for k in keys}
    say("Codex candidates in post-design windows:")
    say(pd.DataFrame(rows).T.round(4).to_string())
    n_sessions = {w: int(((np.array(days) >= a) & (np.array(days) <= b)).sum()) for w, (a, b) in post.items()}
    say("sessions per window:", n_sessions, "| H2 sessions inside the #774 selection sample (<= 09-23):",
        int(((np.array(days) >= "2026-07-01") & (np.array(days) <= "2026-09-23")).sum()))
    # how many of Codex's 301 candidates beat the base in the only window after every design decision
    st = pd.DataFrame({k: {"h2_krw": win(v, *post["h2"])[0], "post_krw": win(v, *post["oct12+"])[0],
                           "post_usd": win(v, *post["oct12+"], "usd")[0],
                           "post_usd_mdd": win(v, *post["oct12+"], "usd")[1],
                           "post_expo": float(v.loc["2026-10-12":, "exposure"].mean())} for k, v in curves.items()}).T
    cand = st.drop(index=["base", "half"])
    say(f"candidates beating base USD return after 10-12: {int((cand.post_usd > st.loc['base', 'post_usd']).sum())}/{len(cand)}; "
        f"with exposure >= 50% in that window: {int(((cand.post_usd > st.loc['base', 'post_usd']) & (cand.post_expo >= .5)).sum())}"
        f"/{int((cand.post_expo >= .5).sum())}")
    say("post-10-12 exposure vs USD return correlation across candidates:",
        round(float(np.corrcoef(cand.post_expo, cand.post_usd)[0, 1]), 3))
    OUT["g"] = dict(cap_variants=table, labels_changed=len(changed), unknown_then=unknown_then, unknown_now=unknown_now,
                    neighbours=grid, bench=bench, post_design=rows, sessions=n_sessions)


# --------------------------------------------------------------------------- h) independent reprice
def section_h(ctx):
    """Reprice stock100_cash_SOXX100r1 (Codex's H1 pick) with a dict-based engine that shares no code with
    study.plan / study.replay: desk weekly decisions, explicit j-1 SMA test, explicit j-1 SOXX state."""
    say("
== h) independent reprice of the H1-selected rule ==")
    book, days = ctx.book, ctx.days
    first = book.days.index(days[0])
    cls, opn = book.cls, book.opn
    soxx = cls["SOXX"].to_numpy(float)

    def stock_ok(ticker, p, n=100):
        w = cls[ticker].to_numpy(float)[p - n + 1:p + 1]
        return bool(np.isfinite(w).all() and w[-1] >= w.mean())

    def market_on(p, n=100):
        # r1: the state is simply the last completed close versus its own average
        w = soxx[p - n + 1:p + 1]
        return bool(w[-1] >= w.mean())

    selected, target, last_pass = set(), {}, None
    shadow, reserve = {}, 1.0          # the ungated stock-filter account
    nav_real, on_prev = 1.0, False
    usd, expo_days = [], 0
    prev_close_shadow = 1.0
    for j, day in enumerate(days):
        k = first + j
        o = {t: float(opn[t].iloc[k]) for t in shadow}
        shadow_open = reserve + sum(q * o[t] for t, q in shadow.items())
        week = j == 0 or S.iso_week(day) != S.iso_week(days[j - 1])
        trade = False
        if week or not target:
            decision = S.weekly_target(book, k, day, selected)
            if decision["ranked"]:
                selected, target = set(decision["selected"]), decision["target"]
                trade = True
        if target:
            passed = frozenset(key for key in selected if stock_ok(book.ticker[key], k - 1))
            if trade or passed != last_pass:
                weights = {book.ticker[key]: target.get(key, 0.0) for key in passed if target.get(key, 0.0) > 0}
                shadow = {t: shadow_open * w / float(opn[t].iloc[k]) for t, w in weights.items()}
                reserve = shadow_open * (1 - sum(weights.values()))
                last_pass = passed
        shadow_close = reserve + sum(q * float(cls[t].iloc[k]) for t, q in shadow.items())
        on = market_on(k - 1)
        if j and on_prev:
            nav_real *= shadow_open / prev_close_shadow
        if on:
            nav_real *= shadow_close / shadow_open
        usd.append(nav_real)
        on_prev, prev_close_shadow = on, shadow_close
    usd = np.array(usd)
    krw = usd * ctx.fx
    curves, _ = common.codex_curves(ctx)
    ref = curves["stock100_cash_SOXX100r1"]
    err = float(np.max(np.abs(usd / ref.usd.to_numpy() - 1)))
    led = pd.DataFrame({"krw": krw, "usd": usd}, index=days)
    say(f"independent engine vs Codex ledger stock100_cash_SOXX100r1: max relative USD error {err:.3e}")
    say("independent H1 KRW %+.4f | H2 KRW %+.4f, H2 KRW mdd %+.4f | H2 USD %+.4f"
        % (win(led, "2026-01-01", "2026-06-30")[0], *win(led, "2026-07-01", "2026-12-31"), win(led, "2026-07-01", "2026-12-31", "usd")[0]))
    say("Codex report: H1 +254.50%, H2 -33.93%, H2 mdd -34.62%, H2 USD -23.74%")
    OUT["h"] = dict(max_rel_error=err, h1_krw=win(led, "2026-01-01", "2026-06-30")[0], h2_krw=win(led, "2026-07-01", "2026-12-31")[0],
                    h2_krw_mdd=win(led, "2026-07-01", "2026-12-31")[1], h2_usd=win(led, "2026-07-01", "2026-12-31", "usd")[0])


SECTIONS = {"h": section_h, "a": section_a, "b": section_b, "c": section_c, "d": section_d, "e": section_e, "f": section_f,
            "g": section_g}


def main(argv):
    ctx = common.context()
    wanted = argv or sorted(SECTIONS)
    for name in wanted:
        SECTIONS[name](ctx)
    path = HERE / "cache" / "audit_timing.json"
    path.parent.mkdir(exist_ok=True)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    previous.update(OUT)
    path.write_text(json.dumps(previous, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    say("\nsaved", path)


if __name__ == "__main__":
    main(sys.argv[1:])
