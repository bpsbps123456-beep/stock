"""Task 1 + 2: base_front/back/2025 and top10[0] cells, side by side with the engine."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

inf_policy = sys.argv[1] if len(sys.argv) > 1 else 'last'
d26, d25, cost, edges, G = C.load_all(inf_policy)
E = C.engine()
print('atr edges mine  :', edges)
print('atr edges engine:', E['atr_edges'])

FIELDS = ['n', 'days', 'syms', 'share', 'hit', 'r1n', 't', 'race', 'r1', 'vs_bt']


def row(tag, mine, eng):
    out = {}
    for f in FIELDS:
        a, b = mine.get(f, np.nan), eng.get(f, np.nan) if eng else np.nan
        out[f] = (a, b)
    diffs = [f for f in FIELDS if not (np.isnan(out[f][0]) and np.isnan(out[f][1]))
             and not np.isclose(out[f][0], out[f][1], rtol=1e-9, atol=1e-12)]
    s = ' '.join(f'{f}={out[f][0]:.6g}/{out[f][1]:.6g}' for f in ['n', 'hit', 'r1n', 't', 'race'])
    print(f'  {tag:10s} {s}  {"OK" if not diffs else "DIFF:" + ",".join(diffs)}')
    return out, diffs


report = {'atr_edges': {'mine': edges, 'engine': E['atr_edges']}, 'groups': {}}
for g, P in G.items():
    eg = E['groups'][g]
    print('=====', g)
    rep = {}
    # Task 1: base
    for tag, A, per in (('base_front', P['y26'], 'front'), ('base_back', P['y26'], 'back'), ('base_2025', P['y25'], 'all')):
        s, _ = C.eval_cell(A, (), per)
        rep[tag] = row(tag, s, eg[tag])
    # Task 2: top-1 cell
    lbl = eg['top10'][0]['cell']
    cell = C.parse_label(lbl)
    print('  top1 cell:', lbl, '->', cell, C.cell_label(cell))
    for tag, A, per, ek in (('top1_front', P['y26'], 'front', 'front'), ('top1_back', P['y26'], 'back', 'back'),
                            ('top1_2025', P['y25'], 'all', 'y2025')):
        s, _ = C.eval_cell(A, cell, per)
        rep[tag] = row(tag, s, eg['top10'][0][ek])
    # all top10 front/back/2025 (n, r1n, t) quick parity
    nd = 0
    for i, c in enumerate(eg['top10']):
        cell = C.parse_label(c['cell'])
        for A, per, ek in ((P['y26'], 'front', 'front'), (P['y26'], 'back', 'back'), (P['y25'], 'all', 'y2025')):
            s, _ = C.eval_cell(A, cell, per)
            e = c[ek]
            bad = [f for f in FIELDS if not np.isclose(s.get(f, np.nan), e.get(f, np.nan), rtol=1e-9, atol=1e-12, equal_nan=True)]
            if bad:
                nd += 1
                print(f'   top10[{i}] {c["cell"]} {ek}: DIFF {bad} mine n={s.get("n")} r1n={s.get("r1n")} | eng n={e["n"]} r1n={e["r1n"]}')
    print(f'  top10 x 3 periods: {30 - nd}/30 identical on all 10 fields')
    rep['top10_identical'] = 30 - nd
    report['groups'][g] = {k: (v[1] if isinstance(v, tuple) else v) for k, v in rep.items()}
    report['groups'][g]['tables'] = {k: v[0] for k, v in rep.items() if isinstance(v, tuple)}

with open(os.path.join(C.OUT, f't12_result_{inf_policy}.json'), 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=1, default=float)
