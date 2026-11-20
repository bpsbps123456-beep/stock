"""Post-selection execution and cash diagnostics of the original and balance screens.

No rule is selected by this script. Both fixed month-first baskets are delayed
one auction using the already determined weights, then separately rescored from
the prior close at the second and third monthly opens. A final diagnostic sizes
the original screen to exactly the number of executable balance-screen names at
the first open. It fills from the highest executable unique original issuers
in rank order at 1/7 each, including rank eight or lower if necessary.
"""
import numpy as np
import pandas as pd

from research import HERE, describe
from timing import TimingEngine
from baselines import market_gate, _ledger
from compare import close_terminated, switched_targets
from market import load_inputs
from recipes import recipes


def exposure_match(src, ranks, month_first, balance):
    out = {}
    for j in month_first:
        needed = np.count_nonzero(balance[j])
        w = np.zeros(len(src.symbols))
        issuers = set()
        for s in ranks[j]:
            if len(issuers) == needed:
                break
            issuer = src.issuers[j, s]
            if src.tradable[j, s] and issuer not in issuers:
                w[s] = 1 / 7
                issuers.add(issuer)
        assert len(issuers) == needed, src.days[j]
        out[j] = w
    return close_terminated(out, src.opn, src.tradable)


def risk_off_diagnostics(src, gate, routed, path):
    """Compound actual USD results from exit open to recovery open including switch
    fees on both boundaries. An unfinished spell ends at the last recorded close.
    """
    cash = 1.
    shares = np.zeros(len(src.symbols))
    previous_drag = 1.
    entry_value = None
    episodes = []
    for j, on in enumerate(gate):
        marks = np.nan_to_num(src.opn[j])
        nav = cash + shares @ marks
        drag = path.krw10.iat[j] / path.krw.iat[j]
        if not on and entry_value is None:
            entry_value = nav * previous_drag
        elif on and entry_value is not None:
            episodes.append(nav * drag / entry_value)
            entry_value = None
        if j in routed:
            w = routed[j]
            shares = np.divide(nav*w, marks, where=w > 0, out=np.zeros_like(w))
            cash = nav * (1 - w.sum())
        previous_drag = drag
    if entry_value is not None:
        episodes.append(path.krw10.iat[-1] / src.fx[-1] / entry_value)
    return dict(off_sessions=int((~gate).sum()), off_spells=len(episodes),
                off_exposure_mean=path.exposure.loc[~gate].mean(),
                off_names_mean=path.active.loc[~gate].mean(),
                off_spells_net_usd_return=np.prod(episodes) - 1)


def run():
    src = load_inputs()
    gate = market_gate(src)
    engine = TimingEngine(src, gate)
    ranks = recipes(src)
    first = engine.reviews["monthly1"]
    for name, offset in [("fresh2", 1), ("fresh3", 2)]:
        engine.reviews[name] = [j+offset for j in first if j+offset < len(src.days)
                                and src.days[j][:7] == src.days[j+offset][:7]]
    plans = {}
    for key in ("original", "balance"):
        monthly = engine.targets(ranks[key], "monthly1")
        plans[(key, "fresh1")] = monthly
        # Shift already executable targets one session; acquisitions still settle
        # on their real date independently of the delayed discretionary trades.
        shifted = {j+1: monthly[j].copy() for j in first if j+1 < len(src.days)}
        plans[(key, "stale2")] = close_terminated(shifted, src.opn, src.tradable)
        for schedule in ("fresh2", "fresh3"):
            plans[(key, schedule)] = engine.targets(ranks[key], schedule)
    plans[("original_matched_to_balance", "fresh1")] = exposure_match(
        src, ranks["original"], first, plans[("balance", "fresh1")])
    rows = []
    for (recipe, schedule), alt in plans.items():
        routed = switched_targets(src.days, src.opn, engine.desk, alt, gate)
        frame = _ledger(src, routed)
        if schedule == "fresh1" and recipe in ("original", "balance"):
            prior = pd.read_csv(HERE / "switch_results.csv", index_col="key").loc[recipe + "/monthly1"]
            check = describe(frame)
            for field in ("year_ret", "h2_ret", "h2_dd", "year_dd"):
                np.testing.assert_allclose(check[field], prior[field], atol=1e-10)
        stats = describe(frame)
        rows.append(dict(recipe=recipe, timing=schedule, **stats,
                         **risk_off_diagnostics(src, gate, routed, frame)))
    results = pd.DataFrame(rows)
    # All accounts follow the same desk between spells. The five relative
    # risk-off returns including boundary fees must explain their whole-year gap.
    wealth = 1 + results.year_ret
    spell_wealth = 1 + results.off_spells_net_usd_return
    np.testing.assert_allclose(wealth / wealth.iat[0], spell_wealth / spell_wealth.iat[0], atol=1e-11)
    results.to_csv(HERE / "sensitivity.csv", index=False, float_format="%.11g")
    return results


if __name__ == "__main__":
    data = run()
    print(data[["recipe", "timing", "year_ret", "h2_ret", "year_dd", "h2_dd",
                "off_exposure_mean", "off_spells_net_usd_return"]].round(4).to_string(index=False))
