"""Current SEIBRO SOXS levels using the frozen study's canonical helpers.

Only summary.json, recent.csv and qa.json are written beside the input snapshot.
No price download, custody rebuild, research-panel overwrite or trading action.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
spec = importlib.util.spec_from_file_location("soxs_current_focus", HERE / "soxs_focus.py")
focus = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = focus
spec.loader.exec_module(focus)
build = sys.modules["soxs_focus_build"]


def complete_daily_cutoff(snapshot):
    grouped = {}
    for row in snapshot["requests"]:
        if row["kind"] != "daily":
            continue
        if row["start"] != row["end"]:
            raise ValueError("Current snapshot must contain single-date requests")
        grouped.setdefault(row["start"], {})[str(row["dtype"])] = row
    complete, excluded = [], []
    for date, rows in sorted(grouped.items()):
        valid = set(rows) == set("1234") and all(
            row.get("n", 0) == len(row.get("rows", [])) > 0 for row in rows.values())
        if valid:
            complete.append(date)
        else:
            excluded.append({"date": date, "list_counts": {
                key: row.get("n", 0) for key, row in rows.items()},
                "reason": "unpublished, empty or incomplete; never a zero-flow observation"})
    if not complete:
        raise ValueError("No complete, nonempty settlement date in snapshot")
    return pd.Timestamp(max(complete)), {"complete_dates": complete, "excluded": excluded}


def required_usd_for_p80(values, *, lower_bound=None):
    """Minimum whole-USD replacement for today's value, using canonical rank.

    Other window observations stay fixed; today's value remains in the rank
    sample. Integer bisection calls the existing rank implementation directly.
    """
    values = np.asarray(values, dtype=float)[-250:].copy()
    finite = values[np.isfinite(values)]
    if not len(values) or not np.isfinite(values[-1]) or len(finite) < 120:
        return None
    low = int(np.floor(finite.min())) - 1
    if lower_bound is not None:
        low = max(low, int(np.ceil(lower_bound)))
    high = max(low, int(np.ceil(finite.max())) + 1)

    def passes(candidate):
        trial = values.copy()
        trial[-1] = candidate
        percentile, _ = focus.PCT_RANK(trial, win=250, minn=120)
        return bool(percentile[-1] >= 80)

    while low < high:
        mid = (low + high) // 2
        if passes(mid):
            high = mid
        else:
            low = mid + 1
    return float(low)


def rebuild_current(cutoff):
    qa = {}
    sd, dates, total, names = build.load_settle(cutoff, qa)
    nyse, ksd = build.nyse_sessions(), build.ksd_sessions()
    cal, units, tmap = build.build_units(dates, total, ksd, nyse, qa, merge_rule="cal")
    seen = sd.groupby("isin").agg(first=("sdate", "min"), last=("sdate", "max"),
                                  n_dates=("sdate", "nunique"), tot_sum=("tot", "sum"))
    labels = sd["in_d1"].map({True: "1", False: ""})
    for dtype in "234":
        labels = labels + sd[f"in_d{dtype}"].map({True: dtype, False: ""})
    seen = seen.join(labels.groupby(sd["isin"]).agg(lambda x: ",".join(sorted(set(x)))).rename("lists"))
    seen = seen.reset_index()
    for col in ("first", "last"):
        seen[col] = seen[col].dt.date.astype(str)
    cls = build.load_classes(names, seen, qa)
    prod_of = dict(zip(cls["isin"], cls["product"]))
    pc = build.product_table(cls, qa)
    semis, _ = build.semis_list(pc)
    pc["R1"] = pc.index.isin(semis)
    pc["R2"] = pc["utype"].eq("EQ_INDEX") & pc["L"].eq(1) & pc["sign"].eq(0) & ~pc["semi"]
    pc["R3"] = pc["utype"].isin(["BOND", "CASH"]) & pc["sign"].ge(0)
    pc["R4"] = pc["utype"].eq("EQ_SINGLE") & pc["L"].eq(1) & pc["sign"].eq(0) & ~pc["R1"]
    pc["R5"] = pc["utype"].eq("INCOME")
    flows = build.build_flows(sd, cal, units, tmap, pc, prod_of,
                              build.load_ipo(prod_of, qa), qa)
    md = build.daily_helpers(flows, tmap)
    return md, focus.features(md, nyse, rank_start="2023-01-03"), qa


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path,
                        default=STUDY / "results/soxs_current_20261105/raw_snapshot.json")
    args = parser.parse_args(argv)
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    cutoff, availability = complete_daily_cutoff(snapshot)
    snapshot_checks = []
    for row in snapshot["requests"]:
        if row["kind"] != "daily" or row["start"] not in availability["complete_dates"]:
            continue
        date = row["start"].replace("-", "")
        path = build.PRIOR / "raw/daily" / ("d" + str(row["dtype"])) / f"{date}_{date}.json"
        cached = json.loads(path.read_text(encoding="utf-8"))
        keys = ("isin", "rank", "buy", "sell", "net", "tot")
        signature = lambda rows: sorted(tuple(r[k] for k in keys) for r in rows)
        same = signature(cached["rows"]) == signature(row["rows"])
        snapshot_checks.append({"date": row["start"], "dtype": row["dtype"], "same": same})
        if not same:
            raise ValueError("Snapshot differs from canonical raw; preserve and review revision before rebuilding")
    frozen_path = STUDY / "derived/measures_daily.parquet"
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    frozen = pd.read_parquet(frozen_path).set_index("date")
    md, feat, qa = rebuild_current(cutoff)
    overlap = md.index.intersection(frozen.index)
    money_errors = {col: float((md.loc[overlap, col] - frozen.loc[overlap, col]).abs().max())
                    for col in ("SOXS_buy", "SOXS_sell", "SOXS_net", "SOXS_tot")}
    metadata_errors = {col: int((md.loc[overlap, col] != frozen.loc[overlap, col]).sum())
                       for col in ("cens_SOXS", "part_SOXS", "seibro_date", "unit_last",
                                   "known_settle", "entry_settle")}
    if any(money_errors.values()) or any(metadata_errors.values()) or len(overlap) != len(frozen):
        raise ValueError("Current rebuild changes frozen overlapping observations")
    latest = md.index.max()
    row = md.loc[latest]
    measures = {}
    for length in (1, 5):
        for measure in focus.MEASURES:
            key = f"{measure}_L{length}"
            history = feat[key].loc[:latest].iloc[-250:]
            current = float(history.iloc[-1])
            median = float(history.median())
            detail = {"value": current, "P": float(feat.loc[latest, key + "_P"]),
                      "rank_n": int(feat.loc[latest, key + "_rank_n"]),
                      "trailing250_valid_median": median,
                      "difference_from_median": current - median,
                      "multiple_of_median": current / median if median else None}
            if measure != "buy_share":
                target = required_usd_for_p80(history, lower_bound=0 if measure == "gross_usd" else None)
                detail["P80_required_window_usd"] = target
                detail["P80_gap_window_usd"] = target - current if target is not None else None
                detail["P80_required_latest_day_usd_prior_days_fixed"] = (
                    target - current + row["SOXS_buy" if measure == "gross_usd" else "SOXS_net"]
                    if target is not None else None)
            measures[key] = detail
    columns = ["SOXS_buy", "SOXS_sell", "SOXS_net", "SOXS_tot", "cens_SOXS", "part_SOXS",
               "seibro_date", "unit_last", "known_settle", "entry_settle"]
    recent = md[columns].join(feat).loc["2026-10-26":].tail(10)
    summary = {"source": snapshot["source"], "snapshot_fetched_at": max(
        r["fetched_at"] for r in snapshot["requests"]), "latest_complete_seibro_date": cutoff,
        "latest_us_source_date": latest, "known_kst_convention": row.known_settle.tz_localize(
            "Asia/Seoul") + pd.Timedelta(hours=9), "entry_us_session": row.entry_settle,
        "not_current_day_flow": "Settlement-based delayed data; empty later dates are unpublished, not zero buying.",
        "latest_source_censored": bool(row.cens_SOXS), "latest_source_partial": bool(row.part_SOXS),
        "latest_raw_usd": {k: float(row[k]) for k in columns[:4]}, "measures": measures,
        "rank_definition": "2023-01-03 onward, trailing250 US sessions including current, min120 valid, midrank ties",
        "P80_definition": "Minimum whole USD replacing current window sum; canonical rank, other samples fixed. This is hypothetical threshold, not a forecast or order.",
        "median_note": "Same current-inclusive250-session valid sample; signed net median multiples are arithmetic, not buying strength.",
        "availability": availability}
    qa["frozen_overlap"] = {"rows": len(overlap), "max_money_errors_usd": money_errors,
                             "metadata_mismatches": metadata_errors, "passed": True}
    qa["snapshot_matches_canonical"] = snapshot_checks
    qa["availability"] = availability
    qa["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        args.snapshot, frozen_path, HERE / "build.py", HERE / "soxs_focus.py", Path(__file__))}
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == frozen_hash
    folder = args.snapshot.parent
    recent.to_csv(folder / "recent.csv", index_label="source_date")
    for name, obj in (("summary.json", summary), ("qa.json", qa)):
        (folder / name).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2,
                                            allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(clean(summary), ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
