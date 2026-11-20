"""Execution calendars and the standalone/desk-switch FINVIZ ledgers."""
from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SLEEVES = Path(__file__).resolve().parents[1] / "seven_sleeves_20261120"
sys.path.insert(0, str(SLEEVES))
from baselines import _ledger, desk_targets
from compare import close_terminated, switched_targets


CALENDAR = ("daily", "weekly1", "weekly3", "biweekly", "monthly1", "monthly6",
            "monthly11", "monthly1_11")
SCHEDULES = CALENDAR + ("exit", "exit5")
ANCHOR = date(2026, 1, 5)


def review_days(days, gate, schedule):
    """Use only elapsed sessions and the gate known before today's auction.

    Week/month offsets begin with the first observed session. A truncated week
    cannot trigger a missing third session. Biweekly starts in the Jan 5, 2026
    week and uses the first traded day of every other calendar week thereafter.
    Exit-5 reviews the 1st, 6th, 11th, ... day of each risk-off spell.
    """
    if schedule not in SCHEDULES or len(days) != len(gate):
        raise ValueError("Unknown review schedule or misaligned gate")
    previous = None
    week_count = month_count = off_count = 0
    selected = []
    for j, (stamp, risk_on) in enumerate(zip(days, gate)):
        day = date.fromisoformat(str(stamp)[:10])
        if previous is not None and day <= previous:
            raise ValueError("Study days must be strictly increasing")
        new_week = previous is None or previous.isocalendar()[:2] != day.isocalendar()[:2]
        new_month = previous is None or (previous.year, previous.month) != (day.year, day.month)
        week_count = 1 if new_week else week_count + 1
        month_count = 1 if new_month else month_count + 1
        off_count = 0 if risk_on else off_count + 1
        monday = day - timedelta(days=day.weekday())
        due = {
            "daily": True, "weekly1": week_count == 1, "weekly3": week_count == 3,
            "biweekly": new_week and monday >= ANCHOR and (monday - ANCHOR).days % 14 == 0,
            "monthly1": month_count == 1, "monthly6": month_count == 6,
            "monthly11": month_count == 11, "monthly1_11": month_count in (1, 11),
            "exit": off_count == 1, "exit5": off_count > 0 and (off_count - 1) % 5 == 0,
        }
        if due[schedule]:
            selected.append(j)
        previous = day
    return selected


def _weight_checks(src, plan, stress):
    """Rebalance at cost-adjusted NAV. Target fractions and subsequent drift are
    scale-invariant, so disclosed *weights* are identical in both fee accounts.
    Solve each cost-level financing equation against the true pre-auction holdings.
    """
    names = np.empty(0, dtype=int)
    starting_quotes = allocated = np.empty(0)
    reserves = 1.
    rates = np.array([.001, .0025] if stress else [.001])
    factors = np.ones((len(src.days), len(rates)))
    for j in sorted(plan):
        w = plan[j]
        marked = allocated * src.opn[j, names] / starting_quotes
        nav = reserves + marked.sum()
        coming = np.flatnonzero(w > 0)
        union = np.union1d(names, coming)
        before = np.zeros(len(union))
        before[np.searchsorted(union, names)] = marked / nav
        chargeable = ~src.settlements[j, union]
        before = before[chargeable]
        target = w[union[chargeable]]
        post = np.ones(len(rates))
        for _ in range(10):
            updated = 1. - rates * np.abs(post[:, None] * target - before).sum(axis=1)
            if np.max(np.abs(updated - post)) < 1e-15:
                break
            post = updated
        factors[j] = updated
        names = coming
        allocated = w[names]
        starting_quotes = src.opn[j, names]
        reserves = 1. - allocated.sum()
    closing = allocated * src.close[-1, names] / starting_quotes
    nav = closing.sum() + reserves
    weights = {src.symbols[i]: float(value / nav) for i, value in zip(names, closing)}
    weights["CASH"] = float(max(0., reserves) / nav)
    if not np.isclose(sum(weights.values()), 1., atol=1e-12):
        raise ValueError("End-of-study holdings do not reconcile")
    return weights, factors.cumprod(axis=0)


class TimingEngine:
    def __init__(self, src, gate, desk_plan=None):
        self.src = src
        gate = np.asarray(gate)
        if gate.dtype != bool or gate.shape != (len(src.days),):
            raise ValueError("Supply the point-in-time boolean SOXX gate by execution day")
        self.gate = gate
        self.desk = desk_targets(src) if desk_plan is None else desk_plan
        self.reviews = {name: review_days(src.days, gate, name) for name in SCHEDULES}

    def targets(self, ordered_ranks, schedule):
        """Select the published top seven, then leave blocked or duplicate slots in USD.

        A missing executable security or a second share class does not promote rank
        eight. This preserves the previous study's top-seven-before-termination rule.
        """
        if len(ordered_ranks) != len(self.src.days):
            raise ValueError("Rankings must be aligned to the execution calendar")
        plan = {}
        for j in self.reviews[schedule]:
            weights = np.zeros(len(self.src.symbols))
            used = set()
            for i in ordered_ranks[j][:7]:
                if not isinstance(i, (int, np.integer)) or not 0 <= i < len(weights):
                    raise ValueError(f"Invalid ranked symbol index on {self.src.days[j]}")
                issuer = self.src.issuers[j, i]
                if issuer not in used:
                    weights[i] = 1 / 7
                    used.add(issuer)
            plan[j] = weights
        return close_terminated(plan, self.src.opn, self.src.tradable)

    def _price(self, plan, detail, stress):
        frame = _ledger(self.src, plan)
        if detail or stress:
            weights, drag = _weight_checks(self.src, plan, stress)
            np.testing.assert_allclose(frame.krw * drag[:, 0], frame.krw10, rtol=1e-12)
            if stress:
                frame["krw25"] = frame.krw * drag[:, 1]
            if detail:
                frame.attrs.update(final_weights=weights.copy(), final_weights_10bp=weights.copy(),
                                   last_target_date=self.src.days[max(plan)] if plan else None)
        return frame if detail else frame.drop(columns=["holdings"])

    def controls(self, detail=False, stress=False):
        src = self.src
        cash = switched_targets(src.days, src.opn, self.desk, {}, self.gate)
        return {k: self._price(plan, detail, stress) for k, plan in [("base", self.desk), ("exit", cash)]}

    def evaluate(self, ordered_ranks, schedule, detail=False, stress=False):
        src = self.src
        screen = self.targets(ordered_ranks, schedule)
        routed = switched_targets(src.days, src.opn, self.desk, screen, self.gate)
        paths = {"switch": self._price(routed, detail, stress)}
        if schedule in CALENDAR:
            paths["always"] = self._price(screen, detail, stress)
        return paths

    def verify_original(self, ordered_ranks):
        """Cross-check both historical versions including every daily 0bp/10bp mark."""
        result = self.evaluate(ordered_ranks, "monthly1", detail=True, stress=True)["switch"]
        archive = pd.read_csv(SLEEVES / "selected_daily.csv", index_col=["strategy", "date"]).loc["switch"]
        fields = ["krw", "usd", "turn", "exposure", "krw10", "active"]
        assert list(result.index) == list(archive.index)
        np.testing.assert_allclose(result[fields], archive[fields], rtol=2e-10, atol=1e-12)
        assert result.holdings.fillna("").tolist() == archive.holdings.fillna("").tolist()
        return {"sessions": len(result), "last_date": str(result.index[-1]),
                "end_krw10": float(result.krw10.iloc[-1]),
                "maximum_abs_krw10_difference": float((result.krw10 - archive.krw10).abs().max())}
