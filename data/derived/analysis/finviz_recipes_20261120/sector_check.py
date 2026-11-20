"""Post-selection diagnostic, monthly FINVIZ only. No parameter or recipe selection.

In the SOXX risk-off sleeve, skip a candidate whose sector already holds two
executable issuers and continue down the same relative-volume ranking. A blocked
auction or duplicate issuer still consumes its ordinary 1/7 cash slot. Missing
sector classifications cannot enter the capped sleeve. The risk-on desk keeps
its original holdings/weights. Counts are positions; reported maximum sector
weights are closing value fractions of the entire USD account, including cash.
"""
from collections import Counter

import numpy as np
import pandas as pd

from research import HERE, describe
from timing import TimingEngine
from baselines import market_gate, _ledger
from compare import close_terminated, switched_targets
from market import load_inputs
from recipes import recipes


def capped_targets(src, ranks, reviews):
    plan = {}
    for j in reviews:
        w = np.zeros(len(src.symbols))
        count, issuers = Counter(), set()
        slots = 0
        for s in ranks[j]:
            if slots == 7:
                break
            issuer = src.issuers[j, s]
            sector = src.sectors[j, s]
            if issuer in issuers or not src.tradable[j, s]:
                slots += 1
                issuers.add(issuer)
                continue
            if not src.sector_known[j, s] or count[sector] >= 2:
                continue
            w[s] = 1 / 7
            count[sector] += 1
            issuers.add(issuer)
            slots += 1
        plan[j] = w
    return close_terminated(plan, src.opn, src.tradable)


def run():
    src = load_inputs()
    gate = market_gate(src)
    engine = TimingEngine(src, gate)
    orders = recipes(src)
    previous = pd.read_csv(HERE / "selected_daily.csv", index_col=["key", "date"])
    result = []
    for name in ("original", "balance"):
        rank = orders[name]
        for cap, targets in [(0, engine.targets(rank, "monthly1")),
                             (2, capped_targets(src, rank, engine.reviews["monthly1"]))]:
            routed = switched_targets(src.days, src.opn, engine.desk, targets, gate)
            path = _ledger(src, routed)
            if cap == 0:
                prior = previous.loc[name + "/monthly1"]
                assert prior.index.equals(path.index)
                np.testing.assert_allclose(path[["krw10", "krw", "turn", "exposure"]],
                                           prior[["krw10", "krw", "turn", "exposure"]], rtol=2e-10, atol=1e-12)
            for j in np.flatnonzero(~gate):
                names = path.holdings.iat[j].split("|") if path.holdings.iat[j] else []
                pos = [src.loc[s] for s in names]
                assert len(set(src.issuers[j, pos])) == len(pos) <= 7
                if cap:
                    assert max(Counter(src.sectors[j, pos]).values(), default=0) <= cap
            stats = describe(path)
            row = dict(recipe=name, riskoff_sector_cap=cap, **stats)
            for sample, data in [("all", path), ("off", path.loc[~gate])]:
                row.update({f"{sample}_sessions": len(data), f"{sample}_mean_names": data.active.mean(),
                            f"{sample}_mean_sectors": data.sectors.mean(),
                            f"{sample}_max_sector_weight": data.max_sector_weight.max(),
                            f"{sample}_mean_exposure": data.exposure.mean()})
            result.append(row)
    table = pd.DataFrame(result)
    table.to_csv(HERE / "sector_check.csv", index=False, float_format="%.11g")
    return table


if __name__ == "__main__":
    out = run()
    print(out[["recipe", "riskoff_sector_cap", "year_ret", "h2_ret", "year_dd", "h2_dd",
               "off_mean_names", "off_mean_sectors", "off_max_sector_weight"]].round(4).to_string(index=False))
