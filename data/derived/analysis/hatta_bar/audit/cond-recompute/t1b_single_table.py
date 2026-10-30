"""Extra: whole single-condition table (54 bins x 3 groups x 3 periods) vs results/cond_single.csv.
Also tests the vol_tod=inf policy (inf counted as '>5배' vs excluded)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

eng = pd.read_csv(os.path.join(C.BASE, 'results', 'cond_single.csv'), encoding='utf-8-sig')
F = ['n', 'days', 'syms', 'share', 'r1n', 't', 'hit', 'race', 'r1', 'vs_bt']
for pol in ('last', 'nan'):
    d26, d25, cost, edges, G = C.load_all(pol)
    bad = []
    tot = 0
    for _, r in eng.iterrows():
        P = G[r['group']]
        cell = C.parse_label(r['cell'])
        for A, per, pre in ((P['y26'], 'front', 'f_'), (P['y26'], 'back', 'b_'), (P['y25'], 'all', 'y25_')):
            s, _ = C.eval_cell(A, cell, per)
            tot += 1
            d = [f for f in F if not np.isclose(s.get(f, np.nan), r[pre + f], rtol=1e-9, atol=1e-12, equal_nan=True)]
            if d:
                bad.append((r['group'], r['cell'], per, s.get('n'), r[pre + 'n'], d))
        # eligibility flag
        sf, _ = C.eval_cell(P['y26'], cell, 'front')
        if C.eligible(sf) != bool(r['elig']):
            bad.append((r['group'], r['cell'], 'elig', C.eligible(sf), r['elig'], []))
    print(f'vol_tod inf policy={pol}: {tot - len([b for b in bad if b[2] != "elig"])}/{tot} identical; mismatches={len(bad)}')
    for b in bad[:12]:
        print('   ', b)
