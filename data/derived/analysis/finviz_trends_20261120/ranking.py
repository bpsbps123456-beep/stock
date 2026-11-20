"""Rank the predeclared trend grid on prior-session inputs before each available review."""
from itertools import product

import numpy as np
import pandas as pd

from features import PERIODS, load_features, recipes, review_indices
from timing import review_days

SHORT = (10, 15, 20, 25, 30, 35)
MEDIUM = (40, 50, 60, 75, 100)
LONG = (100, 125, 150, 175, 200)
PROFILES = {
    # current ratio, debt/equity, relative volume, RSI; all four boundaries are strict.
    "balance": (1.5, .5, 1., 40.),
    "middle": (1.25, .75, 1., 40.),
    "loose": (1.5, 1., 1., 40.),
    "active": (1.5, .5, 1.25, 45.),
}
SCHEDULES = ("monthly1", "monthly1_11", "weekly1")
FIELDS = ("short", "medium", "long", "alignment", "profile", "order", "schedule")


def screen_key(short, medium, long, alignment, profile, order, schedule):
    return f"{short}-{medium}-{long}/{alignment}/{profile}/{order}/{schedule}"


def top_seven_days(passed, order, sorted_cols, dates, sessions):
    # rank ties were stably sorted from symbol order. Keep the first seven positions,
    # including any issuer duplication or blocked security for the ledger to handle.
    qualifies = np.take_along_axis(passed, order, axis=1)
    counts = np.cumsum(qualifies, axis=1, dtype=np.int16)
    day, which = np.nonzero(qualifies & (counts <= 7))
    result = np.full((sessions, 7), -1, dtype=np.int16)
    result[dates[day], counts[day, which] - 1] = sorted_cols[day, which]
    return result


class ScreenGrid:
    def __init__(self, src):
        dates = np.asarray(review_indices(src), dtype=int)
        self.reviews = {s: review_days(src.days, np.ones(len(src.days), dtype=bool), s)
                        for s in SCHEDULES}
        assert dates.tolist() == sorted(set.union(*(set(v) for v in self.reviews.values())))
        f = load_features().sort_values(["trade_open", "symbol"])
        assert not f.duplicated(["trade_open", "symbol"]).any()
        np.testing.assert_array_equal(f.symbol, np.asarray(src.symbols)[f.col])
        np.testing.assert_array_equal(np.unique(f.trade_open), np.asarray(src.days)[dates])
        np.testing.assert_array_equal(f.signal_close, np.asarray(src.signals)[pd.Index(src.days).get_indexer(f.trade_open)])
        # Densify within each date's prefiltered base population; omitted columns
        # remain NaN and cannot pass a threshold. The input slot is ticker-sorted.
        day = pd.Index(np.asarray(src.days)[dates]).get_indexer(f.trade_open)
        slot = f.groupby("trade_open").cumcount().to_numpy()
        shape = (len(dates), max(slot) + 1)
        def panel(field):
            values = np.full(shape, np.nan)
            values[day, slot] = f[field]
            return values
        cols = np.full(shape, -1, dtype=np.int16)
        assert max(f.col) < np.iinfo(cols.dtype).max
        cols[day, slot] = f.col
        close, rv, rsi, current, debt = [panel(name) for name in ("close", "rv", "rsi", "current_ratio", "debt_equity")]
        sma = {n: panel(f"sma{n}") for n in PERIODS}
        profiles = {key: (current > c) & (debt >= 0) & (debt < d) & (rv > v) & (rsi > r)
                    for key, (c, d, v, r) in PROFILES.items()}
        orders = {"rv": np.argsort(-rv, axis=1, kind="stable")}
        for n in LONG:
            orders[n] = np.argsort(-close / sma[n], axis=1, kind="stable")
        sorted_cols = {key: np.take_along_axis(cols, order, axis=1) for key, order in orders.items()}
        self._rankings = {}
        self._values = {}
        for short, medium, long in product(SHORT, MEDIUM, LONG):
            if medium >= long:
                continue
            base = (sma[short] > sma[medium]) & (close > sma[medium]) & (close > sma[long])
            for alignment, trend in (("base", base), ("stack", base & (sma[medium] > sma[long]))):
                for profile, mask in profiles.items():
                    passed = trend & mask
                    for order, sort_index in (("rv", "rv"), ("trend", long)):
                        values = (short, medium, long, alignment, profile, order)
                        ranking = top_seven_days(passed, orders[sort_index], sorted_cols[sort_index],
                                                 dates, len(src.days))
                        ranking.flags.writeable = False
                        self._rankings[values] = ranking
                        for schedule in SCHEDULES:
                            dims = (*values, schedule)
                            self._values[screen_key(*dims)] = dims
        self.screens = pd.DataFrame.from_dict(self._values, orient="index", columns=FIELDS).rename_axis("key")
        self.baseline = screen_key(20, 50, 200, "base", "balance", "rv", "monthly1")
        assert len(self.screens) == 6912 and len(self._rankings) == 2304
        reference = recipes.recipes(src)["balance"]
        for j in dates:
            actual = self.ranks(self.baseline)[j]
            assert actual[actual >= 0].tolist() == reference[j][:7], src.days[j]

    def ranks(self, key):
        return self._rankings[self._values[key][:-1]]

    def neighbors(self, key):
        """Self and all valid one-grid-step neighbors with every other choice held fixed."""
        values = self._values[key]
        keys = [key]
        for dim, sequence in enumerate((SHORT, MEDIUM, LONG)):
            position = sequence.index(values[dim])
            for i in (position - 1, position + 1):
                if 0 <= i < len(sequence):
                    variant = (*values[:dim], sequence[i], *values[dim + 1:])
                    candidate = screen_key(*variant)
                    if candidate in self._values:
                        keys.append(candidate)
        return keys


if __name__ == "__main__":
    grid = ScreenGrid(recipes.market.load_inputs())
    print(f"{len(grid.screens)} rules and schedules, {len(grid._rankings)} stock rules")
    print(grid.baseline, grid.neighbors(grid.baseline))
