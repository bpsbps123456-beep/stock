"""Describe the existing latest SOXS ledger; no new trading signal or threshold."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent.parent
LEDGER = STUDY / "results/soxs_daily_ledger_20261105/source_daily.csv"
OUT = STUDY / "results/soxs_distribution_20261105"
NORMAL = NormalDist()


def describe(frame, key):
    values = frame[key].dropna()
    latest = frame.iloc[-1]
    value = float(latest[key])
    mean, std = float(values.mean()), float(values.std(ddof=1))
    percentile = float(latest[key + "_P"])
    lower, equal, upper = (int((values < value).sum()), int((values == value).sum()),
                           int((values > value).sum()))
    checked_p = (lower + equal / 2) / len(values) * 100
    if not np.isclose(checked_p, percentile, atol=1e-10):
        raise ValueError("Window disagrees with the canonical ledger percentile")
    z = (value - mean) / std
    standardized = (values - mean) / std
    left = min(-3.5, float(np.floor(standardized.min() * 2) / 2))
    right = max(3.5, float(np.ceil(standardized.max() * 2) / 2))
    edges = np.arange(left, right + 0.25, 0.5)
    counts, _ = np.histogram(standardized, bins=edges)
    if int(counts.sum()) != len(values):
        raise ValueError("Histogram must retain every valid observation")
    return {"key": key, "label": "하루 매수" if key.endswith("L1") else "5일 누적 매수",
            "n": len(values), "missing": len(frame) - len(values), "value_usd": value,
            "mean_usd": mean, "sample_std_usd": std, "raw_amount_z": z,
            "normal_model_percentile": NORMAL.cdf(z) * 100,
            "empirical_percentile": percentile, "empirical_upper_midrank_pct": 100 - percentile,
            "percentile_equivalent_normal_z": NORMAL.inv_cdf(percentile / 100),
            "larger_count": upper, "equal_count": equal, "smaller_count": lower,
            "descending_rank": upper + 1, "sample_skew": float(values.skew()),
            "histogram": [{"left": float(a), "right": float(b), "count": int(c),
                           "density": float(c / len(values) / (b - a))}
                          for a, b, c in zip(edges[:-1], edges[1:], counts)]}


def main():
    data = pd.read_csv(LEDGER, parse_dates=["source_date"]).set_index("source_date")
    latest = data.loc[data.source_observed].index.max()
    if not np.isfinite(data.loc[latest, "gross_usd_L1"]):
        raise ValueError("Latest published source is unknown; never fall back to an older valid value")
    # The ledger has every NYSE source session, including unknown/unpublished rows.
    window = data.loc[:latest].tail(250)
    if len(window) != 250:
        raise ValueError("The fixed current-inclusive 250-session window is required")
    result = {"latest_us_source_date": str(latest.date()),
              "retrieval_date_kst": "2026-11-05", "window_start": str(window.index[0].date()),
              "window_end": str(window.index[-1].date()), "window_sessions": len(window),
              "shared_settlement_source_rows": int(window.unit_shared_source.sum()),
              "rank_basis": "canonical current-inclusive250; midpoint ties; missing never zero",
              "normal_model": "Normal(mean, sample SD ddof1), descriptive overlay only",
              "not_a_probability": "Neither tail area nor buying percentile is a bottom/rebound probability.",
              "source_sha256": hashlib.sha256(LEDGER.read_bytes()).hexdigest(),
              "new_signal_definitions": 0, "optimized_thresholds": 0,
              "measures": [describe(window, key) for key in ("gross_usd_L1", "gross_usd_L5")]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "distribution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    template = (Path(__file__).parent / "soxs_distribution_template.html").read_text(encoding="utf-8")
    html = template.replace("__SOXS_DISTRIBUTION_JSON__", json.dumps(result, ensure_ascii=False))
    (OUT / "current_normal.html").write_text(html, encoding="utf-8")
    scratch = os.environ.get("ORCA_CHAT_SCRATCH_DIR")
    if scratch:
        (Path(scratch) / "soxs-current-normal-20261105-a1c4.html").write_text(html, encoding="utf-8")
    print(json.dumps({"source_date": str(latest.date()), "measures": [
        {k: m[k] for k in ("key", "n", "value_usd", "mean_usd", "sample_std_usd",
                            "raw_amount_z", "empirical_percentile", "normal_model_percentile")}
        for m in result["measures"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
