"""Original desk, SOXX cash exit, and desk/monthly FINVIZ switching comparators."""
import numpy as np
import pandas as pd

from market import PRIOR, canon
from compare import close_terminated, market_gates, scheduled, switched_targets
from evaluate import summarize


def market_gate(src):
    """The 11/18 auction can use the 11/17 close; no 11/18 close enters this gate."""
    original = market_gates(src.book, src.desk_path.days)["SOXX100r1"]
    close = src.book.cls["SOXX"].loc[:src.desk_path.days[-1]].dropna()
    assert len(src.days) == len(original) + 1 and len(close) >= 100
    return np.r_[original, close.iloc[-1] >= close.iloc[-100:].mean()]


def _ledger(src, plan):
    """Price a published target plan. Both 0bp and 10bp have independent share/cash books."""
    shares = np.zeros((2, len(src.symbols)))
    cash = np.ones(2)
    legacy_drag = 1.
    records = []
    for j, day in enumerate(src.days):
        o, c = src.opn[j], src.close[j]
        held = (shares != 0).any(axis=0)
        if not (np.isfinite(o[held]) & (o[held] > 0)).all():
            raise ValueError(f"Missing comparator open on {day}")
        values = shares * np.nan_to_num(o)
        opening = values.sum(axis=1) + cash
        turn = 0.
        if j in plan:
            w = plan[j]
            owned = w > 0
            if not (np.isfinite(o[owned]) & (o[owned] > 0)).all():
                raise ValueError(f"Missing execution on {day}")
            turn = np.abs(w - values[0] / opening[0]).sum()
            legacy_drag *= 1 - .001 * turn
            # Mandatory cash settlements carry no brokerage cost. Normal sales
            # and new purchases are charged on the actual post-cost target dollars.
            chargeable = ~src.settlements[j]
            post = opening.copy()
            for _ in range(8):
                orders = post[1] * w[chargeable] - values[1, chargeable]
                after_fees = opening[1] - .001 * np.abs(orders).sum()
                if abs(after_fees - post[1]) < 1e-15 * opening[1]:
                    break
                post[1] = after_fees
            dollars = post[:, None] * w
            shares = np.divide(dollars, o, out=np.zeros_like(shares), where=owned)
            cash = post - dollars.sum(axis=1)
        owned = shares[0] > 0
        if not (np.isfinite(c[owned]) & (c[owned] > 0)).all():
            raise ValueError(f"Missing comparator close on {day}")
        close_values = shares[:, owned] * c[owned]
        gross, net = close_values.sum(axis=1) + cash
        sectors = {}
        for sector, weight in zip(src.sectors[j, owned], close_values[0] / gross):
            sectors[sector] = sectors.get(sector, 0.) + weight
        records.append(dict(krw=gross * src.fx[j], usd=gross, turn=turn,
                            exposure=1 - cash[0] / opening[0], krw10=net * src.fx[j],
                            krw10_legacy=gross * src.fx[j] * legacy_drag, active=int(owned.sum()),
                            holdings="|".join(np.asarray(src.symbols)[owned]), sectors=len(sectors),
                            max_sector_weight=max(sectors.values(), default=0.)))
    return pd.DataFrame(records, index=pd.Index(src.days, name="date"))


def desk_targets(src):
    """Map the canonical weekly rebalance targets onto this research panel."""
    base = {}
    ref = src.desk_path
    for day, weights in ref.rebalances.items():
        allocation = np.zeros(len(src.symbols))
        for key, weight in weights.items():
            allocation[src.loc[canon(src.book.ticker[key])]] = weight
        base[src.days.index(day)] = allocation
    # The only extended session is a Wednesday (no new weekly desk decision).
    if scheduled(src.days, len(src.days) - 1, "weekly"):
        raise ValueError("Need a new canonical weekly decision at the extension date")
    return base


def baselines(src, gate):
    """Rebuild all comparator schedules, verify the saved study and extend one day."""
    ref = src.desk_path
    base = desk_targets(src)
    history = pd.read_csv(PRIOR / "midcaps/decisions.csv").query(
        "frequency == 'monthly' and filter == 'ownership'").set_index("trade_open")
    screen = {}
    for j, day in enumerate(src.days):
        if scheduled(src.days, j, "monthly"):
            chosen = src.ranks["F"][j][:7]
            w = np.zeros(len(src.symbols))
            w[chosen] = 1 / 7
            screen[j] = w
            if day <= ref.days[-1]:
                logged = history.at[day, "selected_before_terminations"]
                expected = set(logged.split("|")) if pd.notna(logged) else set()
                assert {src.symbols[i] for i in chosen} == expected, day
    screen = close_terminated(screen, src.opn, src.tradable)
    exit_plan = switched_targets(src.days, src.opn, base, {}, gate)
    switch = switched_targets(src.days, src.opn, base, screen, gate)
    output = {key: _ledger(src, plan) for key, plan in [
        ("base", base), ("exit", exit_plan), ("switch", switch)]}

    prior = pd.read_csv(PRIOR / "daily.csv").set_index(["strategy", "date"])
    fields = ["krw", "usd", "turn", "exposure", "krw10"]
    for key, archived in [("base", "base"), ("exit", "base_SOXX100")]:
        observed = output[key].loc[ref.days, ["krw", "usd", "turn", "exposure", "krw10_legacy"]]
        np.testing.assert_allclose(observed, prior.loc[archived, fields], rtol=1e-10, atol=1e-12)
    legacy = output["switch"].loc[ref.days].copy()
    legacy["krw10"] = legacy.krw10_legacy
    stats = summarize(legacy)
    archived_stats = pd.read_csv(PRIOR / "midcaps/results.csv", index_col=0).loc["switch_ownership_monthly"]
    np.testing.assert_allclose([stats[k] for k in archived_stats.index], archived_stats, rtol=1e-10, atol=1e-12)
    return output


if __name__ == "__main__":
    from market import load_inputs
    src = load_inputs()
    curves = baselines(src, market_gate(src))
    for key, frame in curves.items():
        print(key, "last session", frame.iloc[-1][["krw", "krw10", "krw10_legacy"]].to_dict())
