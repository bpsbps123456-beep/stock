"""Does the IC dial depend on the exact 'losers' basket? REV built from 5/10/20-session losers, 5/7/10 names.

Sensitivity only (nothing is selected here): every variant is run with the two IC settings that the
forward selection or the family table pointed at, M2 h20/K20 and h10/K10.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from species import Lab, HERE, weekly_plan, trailing_return
from season import momentum_ic, known_ic
from meta import windows


def main():
    lab = Lab()
    ctx = lab.ctx
    ic = {h: momentum_ic(lab, h) for h in (10, 20)}
    rows = []
    for look in (5, 10, 20):
        for n in (5, 7, 10):
            def pick(j, ranked, prev, look=look, n=n):
                scored = [(trailing_return(ctx, r["key"], j, look), r["key"]) for r in ranked]
                scored = sorted((s, k) for s, k in scored if np.isfinite(s))
                return {k: 1.0 for _, k in scored[:n]}
            rev = weekly_plan(ctx, lab.width, pick)
            alone = windows(lab.replay(rev))
            for h, K in ((20, 20), (10, 10)):
                works = ~(known_ic(ic[h], h, K) < 0)
                led = lab.replay(__import__("species").multi_switch(lab.opn, [lab.plans["MOM"], rev],
                                                                     np.where(works, 0, 1)))
                w = windows(led)
                rows.append({"losers_over": look, "names": n, "ic": f"h{h}_K{K}", "rev_full_usd": alone["full_usd"],
                             "rev_season_usd": alone["reversal_season_usd"], "full_usd": w["full_usd"],
                             "full_usd_mdd": w["full_usd_mdd"], "h2_usd": w["h2_usd"], "hell1_usd": w["hell1_usd"],
                             "rebound_usd": w["rebound_usd"], "hell2_usd": w["hell2_usd"]})
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "out" / "sensitivity_rev.csv", index=False)
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:+.3f}")
    print(table.to_string(index=False))
    base = windows(lab.ledgers["MOM"])
    for ic_name, part in table.groupby("ic"):
        print(f"{ic_name}: {int((part.full_usd > base['full_usd']).sum())}/{len(part)} beat base full-year "
              f"(median {part.full_usd.median():+.3f}, base {base['full_usd']:+.3f}); "
              f"{int((part.h2_usd > base['h2_usd']).sum())}/{len(part)} beat base H2")


if __name__ == "__main__":
    main()
