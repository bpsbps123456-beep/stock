"""BRIEF4: fixed SOXS flow signals and publication-aware 2024--2026 reactions.

py -B data/derived/analysis/seibro_bottoms/code/soxs_multiyear.py
The 2023 observations warm up trailing ranks; they are never performance samples.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
SPEC = importlib.util.spec_from_file_location("seibro_soxs_multiyear_common", HERE / "soxs_focus.py")
COMMON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMON)

START, END = pd.Timestamp("2024-01-02"), pd.Timestamp("2026-11-03")
RANK_START = pd.Timestamp("2023-01-03")
HORIZONS = (5, 10, 20)
SEED, DRAWS = 20261105, 10000
PERIODS = {"2024_2026": (START, END),
           "2024_2025": (START, pd.Timestamp("2025-12-31")),
           "2024": (START, pd.Timestamp("2024-12-31")),
           "2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
           "2026": (pd.Timestamp("2026-01-01"), END)}


def context_panel(base, context):
    """Apply price context after the authoritative publication/rank calculation."""
    if context not in ("allprice", "dd10"):
        raise ValueError("unknown price context")
    x = base.copy()
    x["definition_id"] = x.path_id
    x["context"] = context
    x["path_id"] = context + "__" + x.definition_id
    x["primary"] = ((context == "allprice") & x.measure.eq("gross_usd")
                    & x.delay_sessions.eq(0))
    x["context_eligible"] = True if context == "allprice" else x.state
    x["eligible"] = x.value_valid & x.context_eligible
    x["on"] = x.eligible & x.rank_high
    x["evaluated_entry"] = x.entry.between(START, END)
    x["pending_entry"] = x.entry.gt(END)
    for h in HORIZONS:
        accepted = COMMON.cooldown_positions(x.loc[x.on, "entry_index"].to_numpy(), h)
        x[f"event{h}"] = x.on & x.entry_index.isin(accepted)
    return x


def sample_statistics(x, h):
    """Completed SOXX close reactions, with separate close-based excursions."""
    r = x[f"R{h}_pct"].to_numpy(dtype=float)
    if not np.isfinite(r).all():
        raise ValueError("sample_statistics requires completed finite responses")
    positive, negative = r[r > 0], r[r < 0]
    gain = float(positive.mean()) if len(positive) else None
    loss = float(negative.mean()) if len(negative) else None
    result = {"n": len(r), "n_positive": len(positive), "n_negative": len(negative),
              "n_zero": int((r == 0).sum()),
              "positive_rate": float((r > 0).mean()) if len(r) else None,
              "mean_pct": float(r.mean()) if len(r) else None,
              "median_pct": float(np.median(r)) if len(r) else None,
              "average_gain_pct": gain, "average_loss_pct": loss,
              "payoff_ratio": gain / abs(loss) if gain is not None and loss is not None else None}
    for label, col, extreme in (("MAE", f"MAE{h}_pct", "min"),
                                ("MFE", f"MFE{h}_pct", "max")):
        values = x[col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("completed response has missing close excursion")
        result[f"mean_{label}_pct"] = float(values.mean()) if len(values) else None
        result[f"median_{label}_pct"] = float(np.median(values)) if len(values) else None
        result[f"extreme_{label}_pct"] = float(getattr(values, extreme)()) if len(values) else None
    return result


def year_matched_comparison(completed, h):
    """OFF is weighted by the completed ON sample counts within each year."""
    included, excluded = [], []
    for year, y in completed.groupby(completed.entry.dt.year, sort=True):
        on, off = y[y.on], y[~y.on]
        counts = {"year": int(year), "n_on": len(on), "n_off": len(off)}
        if not len(on) or not len(off):
            excluded.append({**counts, "reason": "no_on" if not len(on) else "no_off"})
            continue
        included.append({**counts,
                         "on_mean_pct": float(on[f"R{h}_pct"].mean()),
                         "off_mean_pct": float(off[f"R{h}_pct"].mean()),
                         "on_positive_rate": float(on[f"R{h}_pct"].gt(0).mean()),
                         "off_positive_rate": float(off[f"R{h}_pct"].gt(0).mean())})
    total = sum(y["n_on"] for y in included)
    result = {"weight": "completed ON count per year", "included_years": included,
              "excluded_years": excluded, "n_on_included": total,
              "n_off_included": sum(y["n_off"] for y in included),
              "n_on_excluded": sum(y["n_on"] for y in excluded),
              "n_off_excluded": sum(y["n_off"] for y in excluded)}
    for group in ("on", "off"):
        for metric in ("mean_pct", "positive_rate"):
            key = f"{group}_{metric}"
            result[key] = sum(y["n_on"] * y[key] for y in included) / total if total else None
    result["on_minus_off_mean_pct"] = (result["on_mean_pct"] - result["off_mean_pct"]
                                          if total else None)
    result["on_minus_off_positive_rate_pp"] = (
        (result["on_positive_rate"] - result["off_positive_rate"]) * 100 if total else None)
    return result


def period_statistics(panel):
    result = {}
    for name, (lo, hi) in PERIODS.items():
        grid = panel[panel.entry.between(lo, hi)]
        x = grid[grid.eligible]
        enabled = x[x.on]
        row = {"entry_sessions": len(grid),
               "latest_source_sessions": int(grid.source_date.notna().sum()),
               "valid_rank_sessions": int(grid.value_valid.sum()),
               "context_sessions_before_rank_validity": int(grid.context_eligible.sum()),
               "eligible_sessions": len(x), "on_days": len(enabled),
               "off_days": int((~x.on).sum()), "reactions": {}}
        for h in HORIZONS:
            y = x[x[f"R{h}_complete"]]
            on, off = y[y.on], y[~y.on]
            groups = {"ON": sample_statistics(on, h), "OFF": sample_statistics(off, h),
                      "ALL": sample_statistics(y, h)}
            a, b = groups["ON"], groups["OFF"]
            delta = {"mean_pct": (a["mean_pct"] - b["mean_pct"]
                                   if len(on) and len(off) else None),
                     "positive_rate_pp": ((a["positive_rate"] - b["positive_rate"]) * 100
                                            if len(on) and len(off) else None)}
            events = x[x[f"event{h}"]]
            completed_events = events[events[f"R{h}_complete"]]
            row["reactions"][f"R{h}"] = {
                **groups, "pooled_on_minus_off": delta,
                "year_matched": year_matched_comparison(y, h),
                "n_incomplete_on": int((~enabled[f"R{h}_complete"]).sum()),
                "n_incomplete_off": int((~x.loc[~x.on, f"R{h}_complete"]).sum()),
                "events": {"cooldown_sessions": h, "accepted": len(events),
                           "incomplete": len(events) - len(completed_events),
                           **sample_statistics(completed_events, h)}}
        result[name] = row
    return result


def primary_shift_tests(panel, *, draws=DRAWS, seed=SEED, rng=None):
    """Four-test family component: R10 mean/win for one gross definition.

    Years rotate independently on completed eligible observations. Rotation keeps
    each year's ON count and the observed ON-count weights fixed. Both statistics
    use the same draws and the same included/excluded years.
    """
    if draws < 1:
        raise ValueError("draws must be positive")
    if rng is None:
        rng = np.random.default_rng(seed)
    x = panel[panel.evaluated_entry & panel.eligible & panel.R10_complete]
    matched = year_matched_comparison(x, 10)
    total = matched["n_on_included"]
    observed = np.zeros(2)
    simulated = np.zeros((draws, 2))
    years = []
    if total:
        for year, y in x.groupby(x.entry.dt.year, sort=True):
            on = y.on.to_numpy(dtype=bool)
            if not on.any() or on.all():
                continue
            response = y.R10_pct.to_numpy(dtype=float)
            values = np.column_stack((response, (response > 0).astype(float) * 100))
            n_on = int(on.sum())
            weight = n_on / total
            rotation = np.empty((len(y), 2))
            for shift in range(len(y)):
                mask = np.roll(on, shift)
                rotation[shift] = values[mask].mean(axis=0) - values[~mask].mean(axis=0)
            observed += rotation[0] * weight
            offsets = rng.integers(0, len(y), size=draws)
            simulated += rotation[offsets] * weight
            years.append({"year": int(year), "n": len(y), "n_on": n_on,
                          "n_off": int((~on).sum()), "weight": weight})
    output = []
    for i, metric in enumerate(("mean_difference_pct", "positive_rate_difference_pp")):
        exceed = int((simulated[:, i] >= observed[i] - 1e-12).sum()) if total else draws
        output.append({"metric": metric, "observed": float(observed[i]) if total else None,
                       "upper_tail": (1 + exceed) / (1 + draws), "exceedances": exceed,
                       "draws": draws, "seed": seed, "included_years": years,
                       "excluded_years": matched["excluded_years"],
                       "n_on_included": total, "n_off_included": matched["n_off_included"],
                       "n_on_excluded": matched["n_on_excluded"],
                       "n_off_excluded": matched["n_off_excluded"]})
    return output


def event_table(daily):
    parts = []
    for h in HORIZONS:
        x = daily[daily[f"event{h}"]].copy()
        x["event_horizon_sessions"] = h
        x["event_window_complete"] = x[f"R{h}_complete"]
        x["event_reaction_pct"] = x[f"R{h}_pct"]
        x["event_MAE_pct"] = x[f"MAE{h}_pct"]
        x["event_MFE_pct"] = x[f"MFE{h}_pct"]
        parts.append(x)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["path_id", "event_horizon_sessions", "entry"], kind="stable")


def main():
    derived = STUDY / "derived"
    inputs = [derived / "measures_daily.parquet", derived / "price_feats.parquet"]
    fingerprints = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    md = pd.read_parquet(inputs[0]).set_index("date").sort_index().loc[RANK_START:]
    pf = pd.read_parquet(inputs[1]).set_index("date").sort_index().loc[:END]
    if md.index.max() != pd.Timestamp("2026-10-30") or pf.index.max() != END:
        raise ValueError("frozen BRIEF4 input coverage changed")
    if md.index.has_duplicates or pf.index.has_duplicates:
        raise ValueError("duplicate source or price dates")
    import exchange_calendars as xc
    sessions = pd.DatetimeIndex(xc.get_calendar("XNYS", start="2022-06-01", end="2027-01-01").sessions)
    sessions = sessions.tz_localize(None)
    price_grid = sessions[(sessions >= pf.index.min()) & (sessions <= END)]
    missing_price_sessions = price_grid.difference(pf.index)
    # Reindex rather than silently compressing a missing US price session.
    pf = pf.reindex(price_grid)
    feat = COMMON.features(md, sessions, rank_start=RANK_START)
    configurations = [(m, length, 0) for m in COMMON.MEASURES for length in (1, 5)]
    configurations += [("gross_usd", length, 1) for length in (1, 5)]
    panels, rows, primary = [], [], []
    rng = np.random.default_rng(SEED)
    for measure, length, delay in configurations:
        base = COMMON.path_panel(md, feat, pf, sessions, measure, length, delay,
                                 start=START, end=END)
        for context in ("allprice", "dd10"):
            panel = context_panel(base, context)
            panels.append(panel)
            row = {"path_id": panel.path_id.iloc[0], "definition_id": panel.definition_id.iloc[0],
                   "measure": measure, "L": length, "delay_sessions": delay,
                   "context": context, "primary": bool(panel.primary.iloc[0]),
                   "periods": period_statistics(panel)}
            if row["primary"]:
                tests = primary_shift_tests(panel, rng=rng)
                for test in tests:
                    test.update(path_id=row["path_id"], horizon_sessions=10)
                primary.extend(tests)
            rows.append(row)
    if len(rows) != 16 or len(primary) != 4:
        raise ValueError("BRIEF4 requires sixteen settings and four primary tests")
    for test, corrected in zip(primary, COMMON.HOLM(np.array([p["upper_tail"] for p in primary]))):
        test["holm4_reference"] = float(corrected)
    summary = {"brief": "BRIEF4.md", "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "common_code_sha256": hashlib.sha256((HERE / "soxs_focus.py").read_bytes()).hexdigest(),
               "input_sha256": fingerprints, "rank_input_start": str(RANK_START.date()),
               "evaluation_start": str(START.date()), "price_end": str(END.date()),
               "last_source": str(md.index.max().date()), "base_definitions": 6,
               "delay_sensitivity_definitions": 2, "price_contexts": 2,
               "settings": 16, "reaction_cells_per_period": 48,
               "threshold_percentile": 80, "rank_window_sessions": 250, "rank_min_valid": 120,
               "missing_US_price_sessions": [str(d.date()) for d in missing_price_sessions],
               "timing": "known_settle 09:00 KST convention, not a historical publication log; latest source before validity; delayed entry recomputes prior-close state",
               "outcomes": "SOXX entry-open to entry-inclusive session 5/10/20 close in percent; every close inside that horizon must be finite and positive",
               "excursions": "MAE=min(0, minimum close/open-1), MFE=max(0, maximum close/open-1), in percent; close-based, not intraday or account returns",
               "positive_rate_units": "fraction 0..1; differences in percentage points",
               "payoff": "mean positive response / absolute mean negative response; missing if either side has no observations",
               "periods": "entry-year groups; completed outcomes may cross a year boundary; pending entries and windows preserved in daily/events",
               "events": "global 5/10/20 US-session cooldown, no year reset; event table has one row per accepted entry and horizon, including pending rows; decline episodes can remain dependent",
               "primary_tests": primary,
               "null": "10000 independent within-year circular shifts, including possible zero shifts; observed ON-count year weights; shared mean/win draws; (1+exceedances)/(10000+1); Holm over exactly four primary tests",
               "random_order": "one generator seeded 20261105, gross L1 then L5, years ascending; no tests for other cells",
               "limitations": "expanded after earlier results were known; historical data reuse and overlapping or clustered episodes; not new independent confirmation; 2023 performance not evaluated",
               "rows": rows}
    daily = pd.concat(panels, ignore_index=True)
    events = event_table(daily)
    if fingerprints != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}:
        raise ValueError("frozen inputs changed during analysis")
    output = STUDY / "results" / "soxs_multiyear"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                   allow_nan=False) + "\n", encoding="utf-8")
    daily.to_csv(output / "daily.csv", index=False, encoding="utf-8")
    events.to_csv(output / "events.csv", index=False, encoding="utf-8")
    print(json.dumps({"output": str(output), "daily_rows": len(daily), "event_rows": len(events),
                      "settings": len(rows), "primary_tests": len(primary)}))


if __name__ == "__main__":
    main()
