"""4(c) cost sensitivity for the 3 top-1 primary cells: net R1 with cost x {1, 0.5, 0} -> cluster t, bootstrap p, Holm-first bar."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
d26, d25, cost, edges, G = C.load_all('last')
E = C.engine()
for g, P in G.items():
    cell = C.parse_label(E['groups'][g]['top10'][0]['cell'])
    chain = any(vi == C.VAR_KEYS.index('chain') for vi, _ in cell)
    for tag, A, per in (('back', P['y26'], 'back'), ('2025', P['y25'], 'all')):
        s, idx = C.eval_cell(A, cell, per)
        r1 = A['r1'][idx]; r1n = A['r1n'][idx]; c = r1 - r1n
        out = []
        for k in (1.0, 0.5, 0.0):
            AA = dict(A); AA['r1n'] = A['r1'] - k * (A['r1'] - A['r1n'])
            st = C.stats(AA, idx); p = C.cluster_boot_p(AA, idx, 'r1n', B=4000, seed=1)
            out.append(f'cost x{k:.1f}: mean={st["r1n"]:+.5f} t={st["t"]:+.2f} p={p:.3f}')
        flag = ' <- primary window' if (tag == '2025') == chain else ''
        print(f'{g:5s} {tag}: n={s["n"]:3d} days={s["days"]:2d} avg cost={c.mean():.5f} | ' + ' | '.join(out) + flag)
print('Holm first-step bar = 0.05/6 = 0.0083')
