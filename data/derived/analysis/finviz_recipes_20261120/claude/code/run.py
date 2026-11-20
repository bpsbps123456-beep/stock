"""Claude's preregistered SOXX-exit FINVIZ variants (see ../HYPOTHESIS.md).

Paths, all held only while the prior SOXX close is below its 100-session mean:
  R0  monthly top-7 by relative volume (reproduces seven_sleeves 'switch')
  R1  semi-monthly (first session of month, first session on/after the 16th) top-7
  R2  R1 dates, keep holdings still ranked <= 14 among passers, refill to 7
  RF  monthly top-7 of passers that also show quarterly sales YoY > 0 and operating income > 0
Each is also run with every decision executed one session late, and against
random-7 placebos drawn from the same pass lists. Outputs stay in ../.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True  # never write caches into the shared research folders

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ANALYSIS = OUT.parents[1]
ROOT = ANALYSIS.parents[2]
SEVEN = ANALYSIS / "seven_sleeves_20261120"
assert (ROOT / "AGENTS.md").exists() and SEVEN.exists()
sys.path[:0] = [str(SEVEN), str(ROOT)]

import market as inputs  # noqa: E402  (adds finviz_rotation and desk_defense folders)
from baselines import _ledger, baselines, desk_targets, market_gate  # noqa: E402
from compare import close_terminated, scheduled  # noqa: E402
from earnings import earnings_scores  # noqa: E402
from evaluate import summarize  # noqa: E402
from switch import switched_targets  # noqa: E402
import technicals as tech  # noqa: E402
import add_midcaps as mid  # noqa: E402

SEED = 20261120
DRAWS = 300
BUFFER = 14
SLOTS = 7
LONG_SPELLS = ("2026-08-31", "2026-10-13")


# ---------------------------------------------------------------- schedules
def schedule_flags(days, kind):
    """Monthly: first session of each month. Semi: also the first session on/after the 16th."""
    flags = []
    for j, d in enumerate(days):
        if j == 0:
            flags.append(True)
            continue
        new_month = d[:7] != days[j - 1][:7]
        mid_month = (not new_month) and int(d[8:10]) >= 16 and int(days[j - 1][8:10]) < 16
        flags.append(new_month or (kind == "semi" and mid_month))
    return flags


def build_plan(src, ranks, flags, buffer=None, lag=0, rng=None, log=None, tag=""):
    """Publish equal-weight targets. Decision on session j uses ranks[j] (signal close j-1)
    and executes at the open of session j + lag."""
    n, last = len(src.symbols), len(src.days) - 1
    plan, held = {}, []
    for j, due in enumerate(flags):
        if not due:
            continue
        tj = j + lag
        if tj > last:
            break
        assert src.signals[j] < src.days[tj], (tag, j)
        order = ranks[j]
        if rng is not None:
            chosen = list(rng.choice(order, size=min(SLOTS, len(order)), replace=False)) if order else []
        elif buffer:
            position = {s: r for r, s in enumerate(order)}
            keep = sorted((h for h in held if position.get(h, buffer) < buffer), key=position.get)
            fill = [s for s in order if s not in keep][: SLOTS - len(keep)]
            chosen = keep + fill
        else:
            chosen = list(order[:SLOTS])
        w = np.zeros(n)
        w[chosen] = 1 / SLOTS
        plan[tj] = w
        held = [s for s in chosen if src.tradable[tj, s]]
        if log is not None:
            log.append((tag, src.days[j], src.signals[j], src.days[tj], len(order),
                        "|".join(src.symbols[s] for s in chosen)))
    return close_terminated(plan, src.opn, src.tradable)


# ---------------------------------------------------------------- spells & metrics
def spells(gate):
    off = ~np.asarray(gate)
    runs, j = [], 0
    while j < len(off):
        if off[j]:
            k = j
            while k + 1 < len(off) and off[k + 1]:
                k += 1
            runs.append((j, k))
            j = k + 1
        else:
            j += 1
    return runs


def spell_windows(gate, n):
    return [(a, b, a - 1, min(b + 1, n - 1)) for a, b in spells(gate)]


def spell_relative(frame, ref, windows, column):
    out = []
    for a, b, pre, post in windows:
        mine = frame[column].iloc[post] / frame[column].iloc[pre]
        theirs = ref[column].iloc[post] / ref[column].iloc[pre]
        out.append(mine / theirs - 1)
    return np.array(out)


def describe(src, frame, gate, windows):
    stats = summarize(frame)
    row = {k: stats[k] for k in [
        "full_return", "full_return_10bp", "h1_return", "h1_return_10bp", "h2_return", "h2_return_10bp",
        "full_mdd", "full_mdd_10bp", "h2_mdd", "h2_mdd_10bp", "full_turn", "h1_turn", "h2_turn"]}
    in_window = np.zeros(len(frame), bool)
    for a, b, pre, post in windows:
        in_window[a:post + 1] = True
    off = ~np.asarray(gate)
    row["spell_turn"] = frame.turn.to_numpy()[in_window].sum()
    row["off_mean_active"] = frame.active.to_numpy()[off].mean()
    row["off_mean_sectors"] = frame.sectors.to_numpy()[off].mean()
    row["off_mean_exposure"] = frame.exposure.to_numpy()[off].mean()
    return row


# ---------------------------------------------------------------- audits
def audit_timing(src, gate, f_scores):
    report = {}
    # 1. every execution session uses the immediately preceding completed close
    assert len(src.signals) == len(src.days)
    assert all(s < d for s, d in zip(src.signals, src.days))
    assert all(src.signals[j] == src.days[j - 1] for j in range(1, len(src.days)))
    report["signal_is_previous_session"] = True
    # 2. SOXX gate from raw closes: allowed on j iff close(signal j) >= mean of last 100 closes
    soxx = src.book.cls["SOXX"].dropna()
    sma = soxx.rolling(100).mean()
    above = (soxx >= sma)
    recomputed = np.array([bool(above.loc[s]) for s in src.signals])
    assert (recomputed == np.asarray(gate)).all()
    report["soxx_gate_recomputed_sessions"] = int(len(recomputed))
    # 3. FINVIZ score cells equal relvol on the signal close of the panel that supplied them
    large_rv = inputs.frozen_field(tech.FROZEN, "relvol")
    mid_rv = inputs.frozen_field(mid.FROZEN, "relvol")
    checked = mismatched = 0
    for j, sig in enumerate(src.signals):
        row = f_scores[j]
        for i in np.flatnonzero(np.isfinite(row)):
            raw = src.raw_symbols[j, i]
            cands = [p.at[sig, c] for p in (large_rv, mid_rv) for c in {raw, src.symbols[i]} if c in p.columns]
            checked += 1
            if not any(np.isfinite(v) and abs(v - row[i]) <= 1e-12 * max(1, abs(v)) for v in cands):
                mismatched += 1
    report["finviz_cells_checked"] = checked
    report["finviz_cells_not_equal_to_signal_close_relvol"] = mismatched
    assert mismatched == 0
    # 4. archived decision log: signal close strictly before the trade open, previous session
    log = pd.read_csv(inputs.PRIOR / "midcaps/decisions.csv")
    assert (log.signal_close < log.trade_open).all()
    pos = {d: k for k, d in enumerate(src.days)}
    prev = log[log.trade_open.isin(pos) & (log.trade_open != src.days[0])]
    assert all(src.days[pos[t] - 1] == s for t, s in zip(prev.trade_open, prev.signal_close))
    report["archived_decisions_checked"] = int(len(log))
    return report


def audit_earnings(src, values):
    """Re-derive each usable quarterly score from the filing table with a strict filed < signal rule."""
    table = pd.read_csv(SEVEN / "earnings_filings.csv")
    issuers = pd.read_csv(SEVEN / "earnings_issuers.csv")
    rosters = pd.read_csv(inputs.PRIOR / "broad_membership.csv")
    changed = {t for t, g in rosters.groupby("symbol") if g.cik.nunique() > 1}
    by_ticker = issuers.groupby("ticker").cik.agg(list)
    filings = {cik: g.sort_values("filed") for cik, g in table.groupby("cik")}
    checked = later_filing_used = stale = 0
    for j, sig in enumerate(src.signals):
        sig_ts = pd.Timestamp(sig)
        for i in np.flatnonzero(np.isfinite(values[j])):
            raw = src.raw_symbols[j, i]
            if raw in changed or raw not in by_ticker or len(by_ticker[raw]) != 1:
                continue
            g = filings.get(by_ticker[raw][0])
            if g is None:
                checked += 1
                later_filing_used += 1
                continue
            known = g[pd.to_datetime(g.filed) < sig_ts]
            checked += 1
            if known.empty or not np.isclose(known.score.iloc[-1], values[j, i], rtol=1e-12, atol=0):
                later_filing_used += 1
            elif not 0 <= (sig_ts - pd.Timestamp(known.period.iloc[-1])).days <= 180:
                stale += 1
    assert later_filing_used == 0 and stale == 0
    return {"earnings_cells_checked": checked, "earnings_cells_inconsistent": later_filing_used + stale}


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    src = inputs.load_inputs()
    gate = market_gate(src)
    days, n = src.days, len(src.days)
    base = desk_targets(src)
    controls = baselines(src, gate)  # re-verifies the archived study internally
    print(f"inputs loaded {time.time() - t0:.0f}s, sessions {n}, off sessions {(~gate).sum()}")

    f_scores = inputs.frozen_field(SEVEN, "finviz_scores").loc[days, src.symbols].to_numpy()
    F = src.ranks["F"]
    raw = np.unique(src.raw_symbols)
    fiscal = earnings_scores(src.signals, raw)
    values = np.take_along_axis(fiscal.to_numpy(), np.searchsorted(raw, src.raw_symbols), axis=1)
    values = np.where(src.membership, values, np.nan)
    quality = np.isfinite(values)
    FQ = [[i for i in order if quality[j, i]] for j, order in enumerate(F)]

    audit = audit_timing(src, gate, f_scores)
    audit.update(audit_earnings(src, values))

    monthly = schedule_flags(days, "monthly")
    semi = schedule_flags(days, "semi")
    assert monthly == [scheduled(days, j, "monthly") for j in range(n)]
    audit["semi_monthly_dates"] = [d for d, f in zip(days, semi) if f and d[8:10] >= "16"]

    windows = spell_windows(gate, n)
    audit["spells"] = [(days[a], days[b], b - a + 1) for a, b, _, _ in windows]
    assert len(windows) == 5 and sum(b - a + 1 for a, b, _, _ in windows) == int((~gate).sum())
    assert set(LONG_SPELLS) <= {days[a] for a, _, _, _ in windows}
    late_gate = np.r_[gate[0], gate[:-1]]
    assert gate[0] and gate[1]
    late_windows = spell_windows(late_gate, n)

    specs = {"R0": (F, monthly, None), "R1": (F, semi, None), "R2": (F, semi, BUFFER), "RF": (FQ, monthly, None)}
    decisions, curves, rows = [], {}, {}
    for key, (ranks, flags, buf) in specs.items():
        for lag, g, win in [(0, gate, windows), (1, late_gate, late_windows)]:
            name = key if lag == 0 else f"{key}_late1"
            alt = build_plan(src, ranks, flags, buffer=buf, lag=lag, log=decisions, tag=name)
            curves[name] = _ledger(src, switched_targets(days, src.opn, base, alt, g))
            rows[name] = describe(src, curves[name], g, win)

    # R0 must equal the archived seven_sleeves 'switch' path exactly
    for col in ("krw", "krw10", "turn", "exposure"):
        np.testing.assert_allclose(curves["R0"][col], controls["switch"][col], rtol=1e-10, atol=1e-12)
    audit["R0_reproduces_seven_sleeves_switch"] = True
    for name, frame in controls.items():
        rows[f"ref_{name}"] = describe(src, frame, gate, windows)

    # spell-level relative performance vs R0 (and late variants vs R0_late1)
    spell_rows = []
    for name in curves:
        ref, win = (curves["R0"], windows) if not name.endswith("late1") else (curves["R0_late1"], late_windows)
        for col in ("krw", "krw10"):
            rel = spell_relative(curves[name], ref, win, col)
            total = curves[name][col].iloc[-1] / ref[col].iloc[-1]
            np.testing.assert_allclose(np.prod(1 + rel), total, rtol=1e-10)
            for (a, b, pre, post), r in zip(win, rel):
                spell_rows.append(dict(path=name, cost=col, start=days[a], end=days[b], sessions=b - a + 1,
                                       window_from_close=days[pre], window_to_close=days[post],
                                       relative_to_R0=r))
    spell_frame = pd.DataFrame(spell_rows)
    if (~gate)[-1]:
        audit["last_spell_ongoing_at"] = days[-1]

    # preregistered verdicts
    def verdict(key):
        r, r0 = rows[key], rows["R0"]
        rel = spell_frame.query("path == @key and cost == 'krw10'")
        better = rel.relative_to_R0.to_numpy() > 0
        long_ok = bool(rel[rel.start.isin(LONG_SPELLS)].relative_to_R0.gt(0).any())
        loo = [np.prod(np.delete(1 + rel.relative_to_R0.to_numpy(), k)) - 1 for k in range(len(rel))]
        late = rows[f"{key}_late1"]["full_return_10bp"] - rows["R0_late1"]["full_return_10bp"]
        checks = {
            "1_full_10bp_beats_R0": r["full_return_10bp"] > r0["full_return_10bp"],
            "2_mdd_within_1pp_of_R0": r["full_mdd_10bp"] >= r0["full_mdd_10bp"] - 0.01,
            "3_spells_won_3of5_incl_long": bool(better.sum() >= 3 and long_ok),
            "4_leave_one_spell_out_positive": bool(min(loo) > 0),
            "5_late_execution_improvement_positive": bool(late > 0),
        }
        if key == "R2":
            checks["1b_full_10bp_beats_R1"] = r["full_return_10bp"] > rows["R1"]["full_return_10bp"]
        return dict(checks=checks, passed=all(checks.values()), spells_won=int(better.sum()),
                    leave_one_out_min=float(min(loo)), late_improvement=float(late))
    verdicts = {k: verdict(k) for k in ("R2", "RF")}

    # random-7 placebos from the same pass lists (diagnostic only)
    placebo = []
    for sched_name, flags in (("monthly", monthly), ("semi", semi)):
        for draw in range(DRAWS):
            rng = np.random.default_rng([SEED, 0 if sched_name == "monthly" else 1, draw])
            alt = build_plan(src, F, flags, rng=rng)
            frame = _ledger(src, switched_targets(days, src.opn, base, alt, gate))
            placebo.append(dict(schedule=sched_name, draw=draw, full_return=frame.krw.iloc[-1] - 1,
                                full_return_10bp=frame.krw10.iloc[-1] - 1,
                                full_mdd_10bp=summarize(frame)["full_mdd_10bp"]))
    placebo = pd.DataFrame(placebo)
    pct = {}
    for key, sched_name in (("R0", "monthly"), ("R1", "semi"), ("R2", "semi"), ("RF", "monthly")):
        dist = placebo.query("schedule == @sched_name").full_return_10bp.to_numpy()
        pct[key] = dict(schedule=sched_name, percentile=float((dist < rows[key]["full_return_10bp"]).mean() * 100),
                        placebo_median=float(np.median(dist)), placebo_p10=float(np.percentile(dist, 10)),
                        placebo_p90=float(np.percentile(dist, 90)))

    results = pd.DataFrame.from_dict(rows, orient="index").rename_axis("path")
    results.to_csv(OUT / "results.csv", float_format="%.10g")
    spell_frame.to_csv(OUT / "spells.csv", index=False, float_format="%.10g")
    placebo.to_csv(OUT / "placebo.csv", index=False, float_format="%.10g")
    pd.DataFrame(decisions, columns=["path", "decision_session", "signal_close", "execution_open",
                                     "pass_count", "chosen"]).to_csv(OUT / "decisions.csv", index=False)
    flag = pd.Series(gate, index=days, name="desk_regime")
    daily = pd.concat(curves, names=["path", "date"])
    daily["desk_regime"] = flag.reindex(daily.index.get_level_values("date")).to_numpy()
    daily.to_csv(OUT / "daily.csv", float_format="%.11g")
    summary = dict(audit=audit, verdicts=verdicts, placebo_percentiles=pct,
                   fixed=dict(buffer=BUFFER, slots=SLOTS, seed=SEED, draws=DRAWS, long_spells=LONG_SPELLS),
                   runtime_seconds=round(time.time() - t0, 1))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
                                      encoding="utf-8")
    show = ["full_return", "full_return_10bp", "h1_return_10bp", "h2_return_10bp", "full_mdd_10bp",
            "h2_mdd_10bp", "full_turn", "spell_turn", "off_mean_active", "off_mean_sectors"]
    print(results[show].round(4).to_string())
    print(spell_frame.query("cost == 'krw10'").pivot(index="path", columns="start", values="relative_to_R0").round(4))
    print(json.dumps(dict(verdicts=verdicts, placebo=pct), indent=1, default=str))


if __name__ == "__main__":
    main()
