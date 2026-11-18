"""At the same executable open, route to the appropriate independently drifted basket."""
import numpy as np


def switched_targets(days, opn, base_targets, alternative_targets, gate):
    """On exit switch to the alternative basket; on recovery return to the base.

    A plan specifies its weights only on its own decision dates. Even when a plan
    is inactive, track its last published basket's drift so a midweek switch can
    use today's executable holdings. A plan with no published basket holds cash.
    """
    if len(days) != len(opn) or len(days) != len(gate):
        raise ValueError("Switch inputs must cover the same trading dates")
    plans = (base_targets, alternative_targets)
    shares = np.zeros((2, opn.shape[1]))
    cash = np.ones(2)
    schedule = {}
    previous = None
    for j, day in enumerate(days):
        prices = opn[j]
        held = (shares != 0).any(axis=0)
        if not (np.isfinite(prices[held]) & (prices[held] > 0)).all():
            raise ValueError(f"Cannot mark shadow basket on {day}")
        safe_prices = np.where(np.isfinite(prices), prices, 0)
        values = shares * safe_prices
        nav = values.sum(axis=1) + cash
        active = 0 if gate[j] else 1
        for i, plan in enumerate(plans):
            if j not in plan:
                continue
            weights = plan[j]
            invested = weights > 0
            if not (np.isfinite(prices[invested]) & (prices[invested] > 0)).all():
                raise ValueError(f"Cannot enter shadow basket on {day}")
            values[i] = weights * nav[i]
            shares[i] = np.divide(values[i], prices, out=np.zeros_like(values[i]), where=invested)
            cash[i] = nav[i] - values[i].sum()
        if j in plans[active]:
            schedule[j] = plans[active][j].copy()
        elif active != previous and (previous is not None or np.any(values[active])):
            schedule[j] = values[active] / nav[active]
        previous = active
    return schedule
