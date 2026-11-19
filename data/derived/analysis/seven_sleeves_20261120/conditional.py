"""Exploratory follow-up specified after the primary grid, before running these paths.

Keep the canonical weighted weekly desk while the prior SOXX close is at/above
its 100-session mean. Below that mean route the entire account into a separately
tracked, publicly knowable monthly shadow account. On the return crossing resume
the currently drifted desk basket. Crossings execute at the next US auction,
netting orders in common names. The six fixed defensive baskets are 5F+2E,
2M+3F+2E, 5M+2E, 7F, 7M, and 7E. Evaluate each with no sector restriction and
with cap two; the cap applies only to defensive holdings. Monthly slots start
equal, earn independently and reset equal at the first US open of each month.
Allocate by alternating M, F, E, enforcing one class per economic issuer. Cash
remains in ineligible slots. Compare all twelve, with no parameter selection.
"""
import numpy as np
import pandas as pd

from experiment import HERE, configuration, prepare
from baselines import _ledger, baselines, desk_targets
from evaluate import summarize
from switch import switched_targets
from slots import replay


BASKETS = {
    "5F2E": (0, 0, 5, 2),
    "3F2M2E": (0, 2, 3, 2),
    "5M2E": (0, 5, 0, 2),
    "7F": (0, 0, 7, 0),
    "7M": (0, 7, 0, 0),
    "7E": (0, 0, 0, 7),
}


def run():
    src, market, eligible, gate = prepare()
    main = desk_targets(src)
    curves = baselines(src, gate)
    caps = {"base": None, "switch": None, "exit": None}
    for name, counts in BASKETS.items():
        for cap in (None, 2):
            _, (layout, _) = configuration(counts, "monthly", "monthly", cap)
            alt = {}
            shadow = replay(market, eligible[cap], layout, cap, open_targets=alt)
            np.testing.assert_allclose(_ledger(src, alt).krw, shadow.krw, atol=1e-11, rtol=1e-11)
            routed = switched_targets(src.days, src.opn, main, alt, gate)
            key = f"{name}_cap{cap or 0}"
            curves[key] = _ledger(src, routed)
            caps[key] = cap
    rows = {}
    for key, frame in curves.items():
        if key not in ("base", "switch", "exit"):
            for j, names in enumerate(frame.holdings):
                owned = [src.loc[x] for x in names.split("|") if x]
                assert len(owned) <= 7 and len(set(src.issuers[j, owned])) == len(owned), (key, src.days[j])
                if not gate[j] and caps[key]:
                    _, counts = np.unique(src.sectors[j, owned], return_counts=True)
                    assert not (counts > caps[key]).any(), (key, src.days[j])
        stats = summarize(frame)
        for period, start in [("all", "2026-01-01"), ("h2", "2026-07-01")]:
            off = frame.loc[(frame.index >= start) & ~gate]
            stats.update({f"{period}_off_sessions": len(off),
                          f"{period}_off_mean_active": off.active.mean(),
                          f"{period}_off_mean_exposure": off.exposure.mean(),
                          f"{period}_off_full_seven": int(off.active.eq(7).sum())})
        rows[key] = stats
    out = pd.DataFrame.from_dict(rows, orient="index").rename_axis("strategy")
    out.to_csv(HERE / "conditional_results.csv", float_format="%.11g")
    all_days = pd.concat(curves, names=["strategy", "date"])
    flag = pd.Series(gate, index=src.days)
    all_days["desk_regime"] = flag.reindex(all_days.index.get_level_values("date")).to_numpy()
    all_days.to_csv(HERE / "conditional_daily.csv", float_format="%.11g")
    return out


if __name__ == "__main__":
    results = run()
    print(results[["full_return_10bp", "h1_return_10bp", "h2_return_10bp", "h2_mdd_10bp",
                   "full_mdd_10bp", "all_off_mean_active"]].round(4))
