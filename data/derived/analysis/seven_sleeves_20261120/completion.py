"""Five follow-up accounts funding vacancies from the other allowed non-desk rankings.

Cash rows retain the original empty-slot policy. Fill rows keep the same scheduled
reviews and replace the symbol list for each sleeve. The first session is a normal
scheduled review, so a desk slot may start with a non-desk candidate on January 2.
Both rows use the existing engine's turnover: sum of absolute net purchases and
sales divided by opening portfolio value, summed over all US trading sessions.

The only vacancies in the five fill paths followed cash settlements. EA settled
on August 5 and its F slot waited for the next daily, weekly or monthly review
(1, 3 or 19 sessions). HOLX settled on April 7 in a D slot of the D5/F1/E1
monthly technical portfolio, four sessions before the next weekly D review.
"""
from collections import Counter

import numpy as np
import pandas as pd

from experiment import HERE, configuration, prepare, summarize
from slots import replay


# The original study picked the participants in this secondary diagnostic.
# Counts are D, M, F, E. Earnings are monthly and the sector cap is two in every run.
CASES = [((3, 2, 0, 2), 'weekly'), ((3, 0, 2, 2), 'weekly'),
         ((5, 0, 1, 1), 'monthly'), ((3, 3, 0, 1), 'monthly'),
         ((5, 0, 1, 1), 'daily')]
SLOTS = [f'slot{i}' for i in range(1, 8)]


def extend_ranks(ranks, layout):
    """Keep each primary order, then append other participating M/F/E lists in that order."""
    labels = dict.fromkeys(label for label, _ in layout)
    result = {}
    for label in labels:
        others = [ranks[alt] for alt in 'MFE' if alt in labels and alt != label]
        result[label] = [list(dict.fromkeys(item for sequence in lists for item in sequence))
                         for lists in zip(ranks[label], *others)]
    return result


def verify_holdings(curve, market, source):
    indices = {s: i for i, s in enumerate(market.symbols)}
    for j, row in enumerate(curve[SLOTS].itertuples(index=False, name=None)):
        held = [indices[name] for name in row if name]
        assert len(held) == curve.active.iat[j]
        assert len(set(market.issuers[j, held])) == len(held)
        assert source.sector_known[j, held].all()
        assert max(Counter(market.sectors[j, held]).values(), default=0) <= 2


def run():
    src, market, eligible, _ = prepare()
    ranks = eligible[2]
    main = pd.read_csv(HERE / 'results.csv', index_col='strategy')
    metrics = ['full_return_10bp', 'full_mdd_10bp', 'h1_return_10bp', 'h1_mdd_10bp',
               'h2_return_10bp', 'h2_mdd_10bp', 'full_exposure', 'full_turn']
    rows = []
    daily = {}
    for counts, freq in CASES:
        key, (layout, cap) = configuration(counts, freq, 'monthly', 2)
        for policy, orders in [('cash', ranks), ('fill', extend_ranks(ranks, layout))]:
            curve = replay(market, orders, layout, cap, detail=True)
            verify_holdings(curve, market, src)
            stats = summarize(curve)
            if policy == 'cash':
                np.testing.assert_allclose([stats[m] for m in metrics], main.loc[key, metrics].astype(float),
                                           rtol=1e-8, atol=1e-10)
            rows.append(dict(strategy=key, vacancies=policy, sessions=len(curve),
                             min_active=curve.active.min(), mean_active=curve.active.mean(),
                             full_seven_days=curve.active.eq(7).sum(),
                             **{m: stats[m] for m in metrics}))
            daily[key, policy] = curve[['active', 'krw10', *SLOTS]]
    result = pd.DataFrame(rows)
    result.to_csv(HERE / 'completion_results.csv', index=False, float_format='%.11g')
    pd.concat(daily, names=['strategy', 'vacancies', 'date']).to_csv(
        HERE / 'completion_daily.csv', float_format='%.11g')
    print(result.to_string(index=False))


if __name__ == '__main__':
    run()
