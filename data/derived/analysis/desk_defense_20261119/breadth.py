"""Giant-tab market breadth, frozen separately as a post-selection diagnostic."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import signals
from desk import giant
from watchlist import AI_SUBSECTOR


def rebuild():
    universe = sorted(AI_SUBSECTOR)
    bars = signals.bars_for(set(universe) | {"QQQ"})
    missing = sorted(set(universe) - set(bars))
    if missing:
        raise ValueError(f"Incomplete AI universe: {missing}")
    dates = bars["QQQ"].loc["2025-12-31":signals.END].index
    breadth = giant.breadth_series(bars, universe, dates)
    # Independent date alignment, numerator and denominator for the full period.
    numerator = np.zeros(len(dates), dtype=int)
    denominator = np.zeros(len(dates), dtype=int)
    for symbol in universe:
        local = bars[symbol].loc["2010-01-01":]
        last = local.index.searchsorted(dates, side="right") - 1
        valid = last >= 59
        pos = last[valid]
        prices = local.Close.to_numpy()[pos]
        high = np.maximum.accumulate(local.High.to_numpy())[pos]
        numerator[valid] += (prices / high - 1 >= giant.ai_ath_gap.NEAR_PCT / 100)
        denominator[valid] += 1
    np.testing.assert_allclose(breadth, 100 * numerator / denominator, rtol=1e-12)
    breadth.rename("breadth").to_csv(signals.HERE / "breadth.csv", index_label="close_date")
    initially_short = [s for s in universe if len(bars[s].loc[:"2025-12-31"]) < 60]
    meta = {"first_close": str(dates[0].date()), "last_close": str(dates[-1].date()),
            "sessions": len(dates), "tickers": len(universe),
            "eligible_min": int(denominator.min()), "eligible_max": int(denominator.max()),
            "not_60_bars_on_initial_date": initially_short,
            "threshold_pct": giant.BREADTH_CHECK["split"], "signal_calibration": giant.BREADTH_CHECK["period"],
            "timing": "Latest local close on/before US close date; holidays carry last local close. Trade next US open.",
            "universe": "Current static AI_SUBSECTOR map. Includes later listings only after 60 bars.",
            "checks": "Canonical function matches independent searchsorted as-of denominator and numerator at every close."}
    (signals.HERE / "breadth_coverage.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    rebuild()
