"""Every-source and every-entry ledgers for the six existing SOXS definitions.

No new signal, threshold search, price download or frozen-panel mutation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


current = module("soxs_ledger_current", HERE / "soxs_current.py")
focus, build = current.focus, current.build
multi = module("soxs_ledger_multiyear", HERE / "soxs_multiyear.py")
KEYS = [f"{measure}_L{length}" for measure in focus.MEASURES for length in (1, 5)]
START = pd.Timestamp("2024-01-02")


def statuses(observed, percentile, *, absent):
    observed = np.asarray(observed, dtype=bool)
    percentile = np.asarray(percentile, dtype=float)
    return np.where(~observed, absent, np.where(~np.isfinite(percentile), "UNKNOWN",
                                               np.where(percentile >= 80, "ON", "OFF")))


def nullable_on(values):
    return pd.array([True if s == "ON" else False if s == "OFF" else pd.NA
                     for s in values], dtype="boolean")


def load_price_panel(path, sessions):
    raw = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if raw.index.has_duplicates:
        raise ValueError("Duplicate dates in the single fresh price snapshot")
    raw = raw.loc[sessions.min():]
    px = raw.reset_index().rename(columns={"open": "Open", "close": "Close"})
    px["ticker"] = "SOXX"
    qa = {"prices": {}}
    pf = build.price_feats(px, sessions, raw.index.max(), qa)
    # A missing session remains a gap, so reaction() cannot quietly skip it.
    grid = sessions[(sessions >= pf.index.min()) & (sessions <= pf.index.max())]
    pf = pf.reindex(grid)
    pf["SOXX_daily_return_pct"] = pf.SOXX_Close.pct_change(fill_method=None) * 100
    return pf, qa


def source_ledger(md, feat, pf, sessions, *, start="2023-01-03"):
    grid = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pf.index.max())]
    observed = grid.isin(md.index)
    cols = ["SOXS_buy", "SOXS_sell", "SOXS_net", "SOXS_tot", "cens_SOXS", "part_SOXS",
            "unit_id", "unit_n_trades", "unit_first", "unit_last", "seibro_date",
            "known_settle", "entry_settle"]
    x = md.reindex(grid)[cols].copy()
    x.index.name = "source_date"
    x["source_observed"] = observed
    incomplete = x.cens_SOXS.fillna(True).astype(bool) | x.part_SOXS.fillna(True).astype(bool)
    for col in ("SOXS_buy", "SOXS_sell", "SOXS_net", "SOXS_tot"):
        x["reported_partial_" + col] = x[col].where(x.part_SOXS.fillna(False).astype(bool))
        x[col] = x[col].where(observed & ~incomplete)
    x["timing_basis"] = np.where(observed, "canonical_published_unit", "calendar_expected_unpublished")
    _, settlement = build.cm.calendars()
    ksd = build.ksd_sessions()
    for date in grid[~observed]:
        sdate = pd.Timestamp(build.cm.seibro_of(date.date(), settlement))
        known = build.next_after(ksd, sdate)
        x.loc[date, ["seibro_date", "known_settle", "entry_settle"]] = (
            sdate, known, build.on_or_after(sessions, known))
    x["known_settle_kst"] = pd.to_datetime(x.known_settle).dt.tz_localize("Asia/Seoul") + pd.Timedelta(hours=9)
    mapping = focus.latest_sources(md)
    selected = mapping.source_date.to_dict()
    x["selected_source_for_same_entry"] = x.entry_settle.map(selected)
    x["selected_for_entry"] = observed & (x.selected_source_for_same_entry == x.index)
    x["selection_reason"] = np.where(~observed, "unpublished_source_no_flow_values",
        np.where(x.selected_for_entry, "latest_source_selected_by_research_model",
                 "published_source_shadowed_by_later_source_in_research_model"))
    x["unit_shared_source"] = x.unit_n_trades.gt(1) & observed
    for col in ("cens_SOXS", "part_SOXS"):
        x[col] = x[col].astype("boolean")
    for col in ("unit_id", "unit_n_trades"):
        x[col] = x[col].astype("Int64")
    for col in ("SOXX_Open", "SOXX_Close", "SOXX_daily_return_pct"):
        x[col] = pf[col].reindex(grid)
    x["dd60_pct"] = pf.dd60.reindex(grid) * 100
    x["source_to_entry_sessions"] = [sessions.get_loc(entry) - sessions.get_loc(date)
                                      for date, entry in x.entry_settle.items()]
    reactions = {entry: focus.reaction(pf, entry) for entry in pd.DatetimeIndex(x.entry_settle.unique())}
    preceding = {entry: sessions[sessions.get_loc(entry) - 1] for entry in reactions}
    previous_available = {entry: date in pf.index for entry, date in preceding.items()}
    x["entry_previous_price_date"] = x.entry_settle.map(
        lambda d: preceding[d] if previous_available[d] else pd.NaT)
    x["entry_previous_dd60_pct"] = x.entry_settle.map(
        lambda d: reactions[d]["previous_dd60_pct"] if previous_available[d] else np.nan)
    x["entry_previous_close"] = x.entry_settle.map(
        lambda d: reactions[d]["previous_close"] if previous_available[d] else np.nan)
    x["source_close_to_entry_open_pct"] = [
        (pf.loc[entry, "SOXX_Open"] / close - 1) * 100 if entry in pf.index and np.isfinite(close) else np.nan
        for entry, close in zip(x.entry_settle, x.SOXX_Close)]
    for key in KEYS:
        for col in (key, key + "_P", key + "_rank_n"):
            x[col] = feat[col].reindex(grid)
        x[key + "_status"] = statuses(observed, x[key + "_P"], absent="UNPUBLISHED")
        x[key + "_on"] = nullable_on(x[key + "_status"])
    paths = []
    for _, row in x.iterrows():
        if not row.source_observed:
            paths.append("")
            continue
        dates = pd.bdate_range(row.unit_first, row.unit_last)
        paths.append(";".join((build.PRIOR / "raw/daily" / f"d{dtype}" /
                              f"{date:%Y%m%d}_{date:%Y%m%d}.json").as_posix()
                             for date in dates for dtype in "1234"))
    x["raw_paths_d1_4"] = paths
    x["raw_source_url"] = "https://seibro.or.kr/websquare/control.jsp?w2xPath=/IPORTAL/user/ovsSec/BIP_CNTS10013V.xml&menuNo=921"
    return x.drop(columns="known_settle")


def entry_ledger(md, feat, pf, sessions, *, start=START, end=None):
    end = pd.Timestamp(end) if end is not None else build.next_after(sessions, pf.index.max())
    out = None
    for measure in focus.MEASURES:
        for length in (1, 5):
            key = f"{measure}_L{length}"
            base = focus.path_panel(md, feat, pf, sessions, measure, length, start=start, end=end)
            base = base[base.entry.le(end)].reset_index(drop=True)
            allprice, dd10 = multi.context_panel(base, "allprice"), multi.context_panel(base, "dd10")
            if out is None:
                common = ["entry", "source_date", "known_settle_kst", "previous_close",
                          "previous_dd60_pct", "state", "cens_SOXS", "part_SOXS", "unit_id"]
                common += [f"{label}{h}{suffix}" for h in (5, 10, 20)
                           for label, suffix in (("R", "_pct"), ("R", "_end"), ("R", "_complete"),
                                                 ("MAE", "_pct"), ("MFE", "_pct"))]
                out = base.reindex(columns=common).copy()
                out["new_source_available"] = out.source_date.notna()
                out["price_session_complete"] = out.entry.isin(pf.index)
                # context_panel's frozen study END is not the ledger's price cutoff.
                out["evaluated_entry"] = out.price_session_complete
                out["pending_entry"] = ~out.price_session_complete
            for old, suffix in (("value", ""), ("P", "_P"), ("rank_n", "_rank_n")):
                out[key + suffix] = base[old]
            state = statuses(base.source_date.notna(), base.P, absent="NO_NEW_SOURCE")
            out[key + "_status"] = state
            out[key + "_on"] = nullable_on(state)
            out[key + "_dd10_on"] = dd10.on.to_numpy(dtype=bool)
            out[key + "_dd10_eligible"] = dd10.eligible.to_numpy(dtype=bool)
            out[key + "_dd10_status"] = np.where(~base.state, "OUTSIDE_DD10", state)
            for h in (5, 10, 20):
                out[key + f"_event{h}"] = allprice[f"event{h}"].to_numpy(dtype=bool)
            transitions, runs, onruns = [], [], []
            previous, run, onrun = None, 0, 0
            for value in state:
                run = run + 1 if value == previous else 1
                onrun = onrun + 1 if value == "ON" else 0
                transitions.append("START" if previous is None else (f"{previous}->{value}" if previous != value else "UNCHANGED"))
                runs.append(run)
                onruns.append(onrun)
                previous = value
            out[key + "_transition"] = transitions
            out[key + "_status_run_sessions"] = runs
            out[key + "_on_streak_sessions"] = onruns
    return out


def monthly_ledger(source, entry):
    rows = []
    months = sorted(set(source.index.strftime("%Y-%m")) | set(entry.entry.dt.strftime("%Y-%m")))
    for month in months:
        s = source[source.index.strftime("%Y-%m") == month]
        e = entry[entry.entry.dt.strftime("%Y-%m") == month]
        for key in KEYS:
            rows.append({"month": month, "key": key, "source_sessions": len(s), "entry_sessions": len(e),
                         "source_on": int(s[key + "_status"].eq("ON").sum()),
                         "source_off": int(s[key + "_status"].eq("OFF").sum()),
                         "source_unknown": int(s[key + "_status"].eq("UNKNOWN").sum()),
                         "source_unpublished": int(s[key + "_status"].eq("UNPUBLISHED").sum()),
                         "source_shadowed_on": int((s[key + "_status"].eq("ON") & ~s.selected_for_entry).sum()),
                         "entry_latest_on": int(e[key + "_status"].eq("ON").sum()),
                         "entry_latest_off": int(e[key + "_status"].eq("OFF").sum()),
                         "entry_unknown": int(e[key + "_status"].eq("UNKNOWN").sum()),
                         "entry_no_new_source": int(e[key + "_status"].eq("NO_NEW_SOURCE").sum()),
                         "dd10_on": int(e[key + "_dd10_on"].sum()),
                         "entry_on_price_pending": int((e[key + "_status"].eq("ON") & e.pending_entry).sum())})
    return pd.DataFrame(rows)


def dictionary(frames):
    common = {
        "source_date": ("원거래일", "US trade date represented by settlement money; not its publication date."),
        "entry": ("공개 후 관찰 시작일", "Every NYSE session; latest source is selected before checking validity."),
        "selected_for_entry": ("연구상 최신자료 채택", "Research latest-source model only; False never means the original published flow was absent or unobservable."),
        "selection_reason": ("연구상 채택·가려짐 이유", "Both published source rows remain in the source ledger even if the model selects only the later one."),
        "source_observed": ("공개 흐름 행 존재", "True includes Top50-censored rows; False is an unpublished source, whose money remains NA."),
        "source_sessions": ("원거래일 수", "Count by US source month; differs from entry month counts."),
        "entry_sessions": ("공개 후 관찰일 수", "Count by US entry month, including days without newly mapped source."),
        "known_settle_kst": ("공개 연구 규약", "Next KSD businessday09:00KST after the last settlement unit date; historical actual release logs unavailable."),
        "timing_basis": ("날짜 근거", "Published units use canonical mapping; unpublished dates show expected calendar mapping only."),
        "seibro_date": ("SEIBRO 결제표 날짜", "Display date equals US settlement+1calendar day, weekends shifted; not US source date."),
        "state": ("공개 직전 10% 하락 상태", "Previous entry-session close is at least10% below its trailing60-session close maximum."),
        "dd60_pct": ("원거래일 고점 대비", "Source-day close versus its trailing60-session close maximum; distinct from entry_previous_dd60_pct."),
        "SOXX_daily_return_pct": ("원거래일 SOXX 등락률", "Same-source-day close/previous close-1, descriptive concurrent price action, not a new signal."),
        "source_close_to_entry_open_pct": ("공개까지 SOXX 변화", "Entry open/source close-1; descriptive and only available after entry starts."),
        "entry_previous_dd60_pct": ("공개 직전 고점 대비", "Drawdown at the actual preceding US session close, reused from canonical reaction()."),
        "previous_dd60_pct": ("공개 직전 고점 대비", "Entry context uses preceding session close; never source-day drawdown."),
    }
    files = {}
    for name, frame in frames.items():
        cols = list(frame.columns)
        if frame.index.name and frame.index.name not in cols:
            cols.insert(0, frame.index.name)
        descriptions = {}
        for col in cols:
            ko, detail = common.get(col, (col, "Recorded daily audit field; see signal and status definitions."))
            if col.endswith("_status"):
                detail = "ON=P>=80, OFF=valid P<80, UNKNOWN=published but invalid/missing; UNPUBLISHED=source flow unavailable; NO_NEW_SOURCE=entry without new mapped source. OUTSIDE_DD10 is price-context ineligibility."
            elif col.endswith("_rank_n"):
                detail = "Number of valid window observations in the trailing250-source-session rank; minimum120."
            elif col.endswith("_P"):
                detail = "Current-inclusive trailing250-source-session canonical midrank percentile; not probability of a market rise."
            elif col in KEYS:
                detail = "USD window sum for gross/net; buy_share is sum(buy)/sum(buy+sell). L1/L5 are source US sessions. Missing source days invalidate the window."
            elif col.startswith(("MAE", "MFE")):
                detail = "Close-based minimum/maximum excursion versus entry open, clipped to include0; not intraday execution risk. NA until full horizon completes."
            elif col.startswith("R") and col.endswith(("_pct", "_complete", "_end")):
                detail = "Canonical entry-inclusive5/10/20-session SOXX close response; incomplete or missing-price windows remain NA."
            elif col.endswith("_on_streak_sessions"):
                detail = "Consecutive NYSE entry sessions with newly mapped valid ON. OFF/UNKNOWN/NO_NEW_SOURCE reset to0."
            descriptions[col] = {"ko": ko, "description": detail}
        files[name] = {"columns": descriptions}
    return {"files": files, "basis": "Six fixed definitions; P80 unchanged. 2023 is warmup only. No new signal or threshold search.",
            "shadowing_note": "Published earlier source rows remain real available data. Shadowing describes only the research model choosing the latest source for one entry session."}


def clean(value):
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    return current.clean(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=STUDY / "results/soxs_current_20261105/raw_snapshot.json")
    parser.add_argument("--prices", type=Path, default=STUDY / "results/soxs_current_20261105/market.prices.csv")
    parser.add_argument("--output", type=Path, default=STUDY / "results/soxs_daily_ledger_20261105")
    args = parser.parse_args(argv)
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    cutoff, available = current.complete_daily_cutoff(snapshot)
    frozen_paths = list((STUDY / "derived").glob("*"))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_paths if p.is_file()}
    md, feat, qa = current.rebuild_current(cutoff)
    sessions = build.nyse_sessions()
    pf, price_qa = load_price_panel(args.prices, sessions)
    source_all = source_ledger(md, feat, pf, sessions)
    source, warmup = source_all.loc[START:].copy(), source_all.loc[:"2023-12-31"].copy()
    interior_missing = source.index[~source.source_observed & (source.index <= md.index.max())]
    if len(interior_missing):
        raise ValueError("Historical source gap must be reviewed, not mislabeled as unpublished")
    entry = entry_ledger(md, feat, pf, sessions)
    monthly = monthly_ledger(source, entry)
    frozen = pd.read_parquet(STUDY / "derived/measures_daily.parquet").set_index("date")
    common = md.index.intersection(frozen.index)
    money = {c: float((md.loc[common, c] - frozen.loc[common, c]).abs().max())
             for c in ("SOXS_buy", "SOXS_sell", "SOXS_net", "SOXS_tot")}
    mapping = {c: int((md.loc[common, c] != frozen.loc[common, c]).sum())
               for c in ("cens_SOXS", "part_SOXS", "seibro_date", "known_settle", "entry_settle")}
    if any(money.values()) or any(mapping.values()):
        raise ValueError("Frozen source overlap changed")
    coverage = {"source_rows": len(source), "entry_rows": len(entry), "warmup_rows": len(warmup),
        "source_first": source.index.min(), "source_last": source.index.max(),
        "entry_first": entry.entry.min(), "entry_last": entry.entry.max(),
        "latest_completed_price_session": pf.index.max(), "latest_published_source": md.index.max(),
        "interior_missing_source_dates": interior_missing.tolist(),
        "unpublished_source_dates": source.index[~source.source_observed].tolist(),
        "entry_no_new_source_dates": entry.loc[~entry.new_source_available, "entry"].tolist(),
        "censored_source_dates": source.index[source.cens_SOXS.fillna(False)].tolist(),
        "partial_source_dates": source.index[source.part_SOXS.fillna(False)].tolist(),
        "shadowed_source_rows": int((source.source_observed & ~source.selected_for_entry).sum()),
        "source_year_counts": source.groupby(source.index.year).size().to_dict(),
        "entry_year_counts": entry.groupby(entry.entry.dt.year).size().to_dict(),
        "snapshot_availability": available, "frozen_overlap_rows": len(common),
        "frozen_money_max_error_usd": money, "frozen_mapping_mismatches": mapping,
        "price_coverage": price_qa, "new_signal_definitions": 0, "thresholds_tested": [80],
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (args.snapshot, args.prices, Path(__file__), HERE / "soxs_current.py", HERE / "soxs_focus.py",
             HERE / "soxs_multiyear.py", HERE / "build.py")}}
    for path, digest in before.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen derived artifact changed")
    counts = monthly.groupby("key").sum(numeric_only=True).to_dict("index")
    september = monthly[(monthly.month == "2026-09") & (monthly.key == "gross_usd_L1")].to_dict("records")
    same_day = {}
    year = source[source.index.year == 2026]
    for status in ("ON", "OFF"):
        returns = year.loc[year.gross_usd_L1_status.eq(status), "SOXX_daily_return_pct"].dropna()
        same_day[status] = {"n": len(returns), "rising_days": int(returns.gt(0).sum()),
                            "falling_days": int(returns.lt(0).sum()), "flat_days": int(returns.eq(0).sum()),
                            "mean_pct": returns.mean(), "median_pct": returns.median()}
    summary = {"scope": "Every US source and entry session; existing six SOXS signals only.",
               "source_vs_entry": "Source-date ON and latest-source entry ON are different counts. Earlier source rows remain published real observations even when the model selects a later source.",
               "all_period_counts": counts, "september2026_gross1_counts": september,
               "same_source_day_soxx_2026_gross1_descriptive": same_day,
               "latest_source_rows": source.tail(8).reset_index().to_dict("records"),
               "latest_entry_rows": entry.tail(5).to_dict("records"),
               "no_new_source_note": "A US session without newly mapped source is neither OFF nor zero flow. No forward fill is used.",
               "reaction_note": "Fresh single price snapshot only; full windows required. 11/5 pending prices are never filled from a different split basis."}
    frames = {"source_daily.csv": source, "warmup_daily.csv": warmup,
              "entry_daily.csv": entry, "monthly.csv": monthly}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(args.output / name, index=name in ("source_daily.csv", "warmup_daily.csv"))
    for name, obj in (("coverage.json", coverage), ("summary.json", summary),
                      ("column_dictionary.json", dictionary(frames))):
        (args.output / name).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2,
                                                  allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(clean({"output": str(args.output), "coverage": coverage,
                                   "september2026_gross1": september}), ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
