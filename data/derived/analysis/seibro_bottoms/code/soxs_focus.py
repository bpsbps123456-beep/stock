"""BRIEF3: six fixed SOXS-only signals, publication-aware 2026 SOXX reactions.

py -B data/derived/analysis/seibro_bottoms/code/soxs_focus.py
Read-only inputs: derived/measures_daily.parquet, price_feats.parquet, events.json.
2025 is only a trailing-rank input. All evaluated entry sessions are in 2026.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PRIOR = STUDY.parent / "seibro_flows" / "code"
START, END = pd.Timestamp("2026-01-02"), pd.Timestamp("2026-11-03")
PERIODS = {"all": (START, END), "H1": (START, pd.Timestamp("2026-06-30")),
           "H2": (pd.Timestamp("2026-07-01"), END)}
MEASURES = ("gross_usd", "net_usd", "buy_share")


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def shared_functions():
    """Load the authoritative helpers without polluting generic module names."""
    old_path, old_bytecode = sys.path[:], sys.dont_write_bytecode
    names = ("events", "calendar_map")
    saved = {name: sys.modules.get(name) for name in names}
    try:
        sys.modules["events"] = module_at("soxs_focus_prior_events", PRIOR / "events.py")
        sys.modules["calendar_map"] = module_at("soxs_focus_calendar_map", PRIOR / "calendar_map.py")
        build = module_at("soxs_focus_build", HERE / "build.py")
        timing = module_at("soxs_focus_timing", HERE / "timing.py")
        harness = module_at("soxs_focus_harness", PRIOR / "harness.py")
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
        for name in names:
            if saved[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved[name]
    return build.pct_rank, timing.shift_null, harness.holm


PCT_RANK, SHIFT_NULL, HOLM = shared_functions()


def features(md, price_sessions):
    """Strict windows on source US sessions; censored days cannot become zeros."""
    md = md.copy().sort_index()
    dates = pd.DatetimeIndex(price_sessions)
    dates = dates[(dates >= "2025-01-01") & (dates <= md.index.max())]
    x = md.loc[md.index >= "2025-01-01"].reindex(dates).copy()
    complete = (~x["cens_SOXS"].fillna(True).astype(bool)
                & ~x["part_SOXS"].fillna(True).astype(bool))
    amount_cols = ["SOXS_buy", "SOXS_sell", "SOXS_net"]
    complete &= np.isfinite(x[amount_cols]).all(axis=1) & x.SOXS_buy.ge(0) & x.SOXS_sell.ge(0)
    buy, sell, net = (x[col].where(complete) for col in amount_cols)
    out = pd.DataFrame(index=dates)
    out["source_observed"] = complete
    for length in (1, 5):
        b = buy.rolling(length, min_periods=length).sum()
        s = sell.rolling(length, min_periods=length).sum()
        n = net.rolling(length, min_periods=length).sum()
        vals = {"gross_usd": b, "net_usd": n, "buy_share": b / (b + s).where(b + s > 0)}
        for measure, values in vals.items():
            key = f"{measure}_L{length}"
            percentile, sample = PCT_RANK(values.to_numpy(dtype=float), win=250, minn=120)
            out[key] = values
            out[key + "_P"] = percentile
            out[key + "_rank_n"] = sample
    return out


def latest_sources(md):
    """Choose the latest source before inspecting measurement validity or sign."""
    x = md.reset_index().rename(columns={md.index.name or "index": "source_date"})
    x = x[x.entry_settle.notna()].sort_values(["entry_settle", "source_date"], kind="stable")
    return x.drop_duplicates("entry_settle", keep="last").set_index("entry_settle")


def reaction(pf, entry):
    """Entry day counts as session 1; reference support is the previous close."""
    empty = {"previous_close": np.nan, "previous_dd60_pct": np.nan, "state": False,
             "R5_pct": np.nan, "R10_pct": np.nan, "complete20": False,
             "min_close20": np.nan, "support20": np.nan, "breach20": np.nan,
             "R5_end": None, "R10_end": None, "support20_end": None}
    if entry not in pf.index:
        # A future entry can still have an already-observable preceding close.
        prior = pf[pf.index < entry]
        if len(prior):
            empty.update(previous_close=float(prior.SOXX_Close.iloc[-1]),
                         previous_dd60_pct=float(prior.dd60.iloc[-1]) * 100,
                         state=bool(prior.dd60.iloc[-1] <= -0.10))
        return empty
    i = pf.index.get_loc(entry)
    if i == 0:
        return empty
    previous = pf.iloc[i - 1]
    prevclose = float(previous.SOXX_Close)
    empty.update(previous_close=prevclose, previous_dd60_pct=float(previous.dd60) * 100,
                 state=bool(previous.dd60 <= -0.10))
    op = pf.SOXX_Open.iloc[i]
    for h in (5, 10):
        if i + h <= len(pf) and np.isfinite(op) and op > 0:
            c = pf.SOXX_Close.iloc[i + h - 1]
            if np.isfinite(c) and c > 0:
                empty[f"R{h}_pct"] = float((c / op - 1) * 100)
                empty[f"R{h}_end"] = str(pf.index[i + h - 1].date())
    tail = pf.SOXX_Close.iloc[i:i + 20]
    if len(tail) == 20 and np.isfinite(tail).all() and (tail > 0).all() and np.isfinite(prevclose) and prevclose > 0:
        breached = bool(tail.min() < 0.97 * prevclose)
        empty.update(complete20=True, min_close20=float(tail.min()), support20=not breached,
                     breach20=breached, support20_end=str(pf.index[i + 19].date()))
    return empty


def cooldown_positions(positions, cooldown=10):
    """Entry-inclusive ten-session outcome windows; accept index >= last + 10."""
    accepted, last = [], -10**9
    for p in positions:
        if p - last >= cooldown:
            accepted.append(int(p))
            last = p
    return accepted


def shift_reference(on, response):
    on, response = np.asarray(on, dtype=bool), np.asarray(response, dtype=float)
    if len(on) < 2 or not on.any() or on.all():
        return {"upper_tail": 1.0, "two_sided": 1.0, "nonidentity_shifts": max(0, len(on) - 1)}
    p, p2, nshifts = SHIFT_NULL(on, response, float(response.mean()))
    return {"upper_tail": float((1 + p * nshifts) / (1 + nshifts)),
            "two_sided": float((1 + p2 * nshifts) / (1 + nshifts)),
            "nonidentity_shifts": nshifts}


def path_panel(md, feat, pf, sessions, measure, length, delay=0):
    key = f"{measure}_L{length}"
    mapping = latest_sources(md)
    sessions = pd.DatetimeIndex(sessions)
    rows = []
    # Keep future published-entry rows as pending, but never score beyond END.
    last = max(END, mapping.index.max())
    if delay:
        last = sessions[sessions.searchsorted(last) + delay]
    grid = sessions[(sessions >= START) & (sessions <= last)]
    shifted = mapping.copy()
    shifted["entry_original"] = shifted.index
    shifted.index = pd.DatetimeIndex([sessions[sessions.get_loc(e) + delay] for e in mapping.index])
    for entry in grid:
        row = {"path_id": key + ("_delay1" if delay else ""), "measure": measure, "L": length,
               "delay_sessions": delay, "primary": measure == "gross_usd" and length == 5 and not delay,
               "entry": entry, "entry_index": int(sessions.get_loc(entry)), "source_date": pd.NaT,
               "entry_original": pd.NaT, "known_settle_kst": None, "value": np.nan, "P": np.nan,
               "rank_n": 0, "value_valid": False, "rank_high": False, "on": False}
        row.update(reaction(pf, entry))
        if entry in shifted.index:
            source = shifted.loc[entry]
            date = source.source_date
            val = feat.loc[date] if date in feat.index else None
            known = pd.Timestamp(source.known_settle).tz_localize("Asia/Seoul") + pd.Timedelta(hours=9)
            entry_time = entry.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
            if pd.isna(known) or known > entry_time:
                raise ValueError("unpublished source at entry")
            row.update(source_date=date, entry_original=source.entry_original,
                       known_settle_kst=known.isoformat(), cens_SOXS=bool(source.cens_SOXS),
                       part_SOXS=bool(source.part_SOXS), unit_id=int(source.unit_id))
            if val is not None:
                p = float(val[key + "_P"])
                row.update(value=float(val[key]), P=p, rank_n=int(val[key + "_rank_n"]),
                           value_valid=bool(np.isfinite(p)), rank_high=bool(np.isfinite(p) and p >= 80))
                row["on"] = row["value_valid"] and row["rank_high"] and row["state"]
        rows.append(row)
    return pd.DataFrame(rows)


def statistics(panel):
    out = {}
    for name, (lo, hi) in PERIODS.items():
        all_rows = panel[panel.entry.between(lo, hi)]
        x = all_rows[all_rows.value_valid & all_rows.state]
        enabled = x[x.on]
        off = x[~x.on]
        item = {"entry_sessions": len(all_rows), "latest_source_sessions": int(all_rows.source_date.notna().sum()),
                "valid_rank_sessions": int(all_rows.value_valid.sum()), "state_sessions_before_validity": int(all_rows.state.sum()),
                "valid_state_sessions": len(x), "on_days": len(enabled), "off_days": len(off),
                "on_cooldown10_events": int(enabled.event10.sum()), "reaction_pct": {}}
        for h in (5, 10):
            col = f"R{h}_pct"
            y = x[x[col].notna()]
            on, off_y = y[y.on], y[~y.on]
            mean = lambda f: float(f[col].mean()) if len(f) else None
            a, b, c = mean(on), mean(y), mean(off_y)
            item["reaction_pct"][f"R{h}"] = {"n_on": len(on), "n_state": len(y), "n_off": len(off_y),
                "mean_on": a, "mean_same_state_all": b, "mean_off": c,
                "on_minus_same_state_all": a - b if a is not None and b is not None else None,
                "on_minus_off": a - c if a is not None and c is not None else None,
                "median_on": float(on[col].median()) if len(on) else None,
                "win_on": float((on[col] > 0).mean()) if len(on) else None,
                "shift_null": shift_reference(y.on.to_numpy(), y[col].to_numpy())}
        y = x[x.complete20]
        on, off_y = y[y.on], y[~y.on]
        rate = lambda f: float(f.support20.astype(float).mean()) if len(f) else None
        a, b, c = rate(on), rate(y), rate(off_y)
        item["support20"] = {"n_complete_on": len(on), "n_complete_state": len(y), "n_complete_off": len(off_y),
                "n_pending_on": int((~enabled.complete20).sum()), "rate_on": a, "rate_same_state_all": b,
                "rate_off": c, "on_minus_state": a - b if a is not None and b is not None else None,
                "on_minus_off": a - c if a is not None and c is not None else None}
        events = x[x.event10]
        item["events10"] = {"n": len(events), "n_R10": int(events.R10_pct.notna().sum()),
                             "mean_R10_pct": float(events.R10_pct.mean()) if events.R10_pct.notna().any() else None,
                             "n_complete20": int(events.complete20.sum()),
                             "support20_rate": rate(events[events.complete20])}
        out[name] = item
    return out


def attach_events(panel):
    panel = panel.copy()
    selected = cooldown_positions(panel.loc[panel.on, "entry_index"].to_numpy())
    panel["event10"] = panel.on & panel.entry_index.isin(selected)
    return panel


def bottom_table(md, feat, pf, events):
    rows = []
    for event in events["B10_events"]:
        date = pd.Timestamp(event["bottom"])
        if date.year != 2026:
            continue
        source, f = md.loc[date], feat.loc[date]
        row = {"bottom": str(date.date()), "SOXX_close": float(pf.at[date, "SOXX_Close"]),
               "SOXS_buy_usd": float(source.SOXS_buy), "SOXS_net_usd": float(source.SOXS_net),
               "cens_SOXS": bool(source.cens_SOXS), "part_SOXS": bool(source.part_SOXS),
               "SOXS_buy_L1_P": float(f.gross_usd_L1_P), "SOXS_net_L1_P": float(f.net_usd_L1_P),
               "SOXS_buy_L5_P": float(f.gross_usd_L5_P), "known_settle_kst": str(source.known_settle.date()) + " 09:00 KST",
               "entry_settle": str(source.entry_settle.date()), "confirmed10": event["confirmed10"],
               "meaning": "ex-post bottom description only; not a real-time feature"}
        rows.append(row)
    if len(rows) != 4:
        raise ValueError("expected four 2026 confirmed 10% reversal bottoms")
    return pd.DataFrame(rows)


def main():
    derived = STUDY / "derived"
    names = ["measures_daily.parquet", "price_feats.parquet", "events.json"]
    fingerprints = {name: hashlib.sha256((derived / name).read_bytes()).hexdigest() for name in names}
    md = pd.read_parquet(derived / names[0]).set_index("date").sort_index()
    md = md.loc[md.index >= "2025-01-01"]
    pf = pd.read_parquet(derived / names[1]).set_index("date").sort_index()
    pf = pf.loc[:END]
    if pf.index.max() != END or md.index.max() != pd.Timestamp("2026-10-30"):
        raise ValueError("frozen input date coverage changed; do not silently mix newer flows")
    events = json.loads((derived / names[2]).read_text(encoding="utf-8"))
    import exchange_calendars as xc
    sessions = pd.DatetimeIndex(xc.get_calendar("XNYS", start="2024-12-01", end="2027-01-01").sessions)
    sessions = sessions.tz_localize(None)
    feat = features(md, sessions[sessions <= md.index.max()])
    panels, results = [], []
    configs = [(measure, length, 0) for measure in MEASURES for length in (1, 5)] + [("gross_usd", 5, 1)]
    for measure, length, delay in configs:
        panel = attach_events(path_panel(md, feat, pf, sessions, measure, length, delay))
        panels.append(panel)
        results.append({"path_id": panel.path_id.iloc[0], "primary": bool(panel.primary.iloc[0]),
                        "delay_sessions": delay, "periods": statistics(panel)})
    original = results[:6]
    p = [r["periods"]["H2"]["reaction_pct"]["R10"]["shift_null"]["upper_tail"] for r in original]
    for row, adjusted in zip(original, HOLM(np.array(p))):
        row["periods"]["H2"]["reaction_pct"]["R10"]["holm6_reference"] = float(adjusted)
    summary = {"brief": "BRIEF3.md", "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "input_sha256": fingerprints, "price_end": str(END.date()), "last_source": str(md.index.max().date()),
               "rank_input_start": "2025-01-02", "signal_paths": 6, "extra_delay_paths": 1,
               "threshold_percentile": 80, "rank_window_sessions": 250, "rank_min_valid": 120,
               "outcomes": "SOXX price reactions in %, support rate 0..1; not user account returns or actual bottom hit rate",
               "timing": "known_settle 09:00 KST convention, latest source per original entry selected before validity filtering",
               "cooldown": "entry-inclusive 10 US sessions, next accepted index >= previous + 10; event20 windows may still overlap",
               "periods": "entry-date groups; outcome windows may cross the half-year boundary; completed outcomes use prices <=2026-11-03",
               "null": "reuse timing.shift_null, include identity using (1 + p_old*n_shifts)/(1+n_shifts); untestable p=1 retained in Holm6",
               "limitations": "historical data reused; clustered decline episodes; no 2025 performance or pre-2025 performance evaluated",
               "rows": results}
    daily = pd.concat(panels, ignore_index=True)
    output = STUDY / "results" / "soxs_focus"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    daily.to_csv(output / "daily.csv", index=False, encoding="utf-8")
    daily[daily.event10].to_csv(output / "events.csv", index=False, encoding="utf-8")
    bottom_table(md, feat, pf, events).to_csv(output / "bottoms2026.csv", index=False, encoding="utf-8")
    after = {name: hashlib.sha256((derived / name).read_bytes()).hexdigest() for name in names}
    if fingerprints != after:
        raise ValueError("input changed during experiment")
    print(json.dumps({"output": str(output), "daily_rows": len(daily), "events": int(daily.event10.sum())}))


if __name__ == "__main__":
    main()
