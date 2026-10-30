"""Task 3: independent selection for group ② over all 1-var bins and 2-var AND cells.
Eligibility (front, after dedup): n>=60, days>=15, syms>=10, max day share<=20%. Score = day-clustered t of net R1."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def select(A, cells, codes=None, period='front'):
    rows = []
    for ci, cell in enumerate(cells):
        s, _ = C.eval_cell(A, cell, period, codes)
        if s.get('n', 0) == 0:
            rows.append((ci, 0, False, np.nan))
            continue
        rows.append((ci, s['n'], C.eligible(s), s['t']))
    df = pd.DataFrame(rows, columns=['ci', 'n', 'elig', 't'])
    return df


if __name__ == '__main__':
    group = sys.argv[1] if len(sys.argv) > 1 else '②'
    d26, d25, cost, edges, G = C.load_all('last')
    A = G[group]['y26']
    A25 = G[group]['y25']
    cells = C.all_cells()
    t0 = time.time()
    df = select(A, cells)
    print(f'group {group}: cells={len(cells)} eligible={int(df.elig.sum())}  ({time.time() - t0:.1f}s)')
    E = C.engine()['groups'][group]
    print('engine eligible =', E['eligible'])
    el = df[df.elig].sort_values('t', ascending=False, kind='mergesort').reset_index(drop=True)
    out = []
    for r in range(10):
        cell = cells[int(el.ci[r])]
        lbl = C.cell_label(cell)
        sf, _ = C.eval_cell(A, cell, 'front')
        sb, _ = C.eval_cell(A, cell, 'back')
        sy, _ = C.eval_cell(A25, cell, 'all')
        eng = E['top10'][r]
        same = C.parse_label(eng['cell']) == tuple(sorted(cell))
        print(f'rank{r + 1:2d} mine: {lbl:34s} t={sf["t"]:.4f} n={sf["n"]:4d} | back n={sb.get("n", 0):3d} r1n={sb.get("r1n", np.nan):+.5f}'
              f' | 2025 n={sy.get("n", 0):3d} r1n={sy.get("r1n", np.nan):+.5f} || engine: {eng["cell"]} t={eng["front"]["t"]:.4f} {"SAME" if same else "DIFFERENT"}')
        out.append(dict(rank=r + 1, cell=lbl, front=sf, back=sb, y2025=sy, engine_cell=eng['cell'], same=same))
    # t-gap and closeness: how many eligible cells are within 0.1 / 0.3 of the top t
    tt = el.t.values
    print('top t =', tt[0], ' #eligible within 0.1 of top:', int((tt >= tt[0] - 0.1).sum()), ' within 0.3:', int((tt >= tt[0] - 0.3).sum()))
    # distribution of eligible front t
    print('eligible front t quantiles (50/90/99/max):', np.round(np.quantile(tt, [0.5, 0.9, 0.99, 1.0]), 3))
    json.dump(dict(group=group, eligible=int(df.elig.sum()), engine_eligible=E['eligible'], top10=out),
              open(os.path.join(C.OUT, f't3_result_{"g2" if group == "②" else group}.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
    df.to_csv(os.path.join(C.OUT, f't3_cells_{"g2" if group == "②" else group}.csv'), index=False)
