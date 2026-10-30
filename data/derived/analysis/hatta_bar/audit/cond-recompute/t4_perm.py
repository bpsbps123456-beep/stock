"""Task 4(b): permutation null for group ② (50 perms per scheme).

Shuffle the 18-variable condition rows across ② 2026 signals while (sym, ts, day, outcomes, costs)
stay fixed; re-run the same front selection (1,428 cells, eligibility, cluster t); record the
back net R1 of the top-1 cell, #top-10 cells with back net R1 > 0, and the top-1 front t.

Schemes
  pool   : one permutation over all 2026 ② signals (front+back pooled)
  within : permute front rows among front, back rows among back
  day    : permute only among signals of the same trading day (keeps day composition of every cell;
           sensitivity check — it also keeps day-level condition/outcome links, so it is a stricter null)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

B = int(sys.argv[1]) if len(sys.argv) > 1 else 50
schemes = sys.argv[2].split(',') if len(sys.argv) > 2 else ['pool', 'within', 'day']
d26, d25, cost, edges, G = C.load_all('last')
A = G['②']['y26']
cells = C.all_cells()
codes0 = A['codes']
n = codes0.shape[0]


def run_selection(codes):
    res = []
    for ci, cell in enumerate(cells):
        s, _ = C.eval_cell(A, cell, 'front', codes)
        if s.get('n', 0) and C.eligible(s):
            res.append((s['t'], ci))
    res.sort(key=lambda x: -x[0])
    top = res[:10]
    backs = []
    for t, ci in top:
        sb, _ = C.eval_cell(A, cells[ci], 'back', codes)
        backs.append((sb.get('n', 0), sb.get('r1n', np.nan)))
    return dict(top1_f_t=top[0][0], top1_cell=C.cell_label(cells[top[0][1]]), top1_b_n=backs[0][0],
                top1_b_r1n=backs[0][1], top10_pos=int(sum(1 for b in backs if b[0] > 0 and b[1] > 0)),
                n_elig=len(res))


obs = run_selection(codes0)
print('observed:', obs)
E = C.engine()['groups']['②']['perm']
print('engine perm:', E)

out = {'observed': obs, 'engine': E, 'schemes': {}}
for sch in schemes:
    rng = np.random.default_rng({'pool': 11, 'within': 22, 'day': 33}[sch])
    null = []
    t0 = time.time()
    for b in range(B):
        if sch == 'pool':
            perm = rng.permutation(n)
        elif sch == 'within':
            perm = np.arange(n)
            for m in (A['front'], A['back']):
                ix = np.flatnonzero(m)
                perm[ix] = ix[rng.permutation(ix.size)]
        else:  # day
            perm = np.arange(n)
            for dcode in np.unique(A['dcode']):
                ix = np.flatnonzero(A['dcode'] == dcode)
                perm[ix] = ix[rng.permutation(ix.size)]
        null.append(run_selection(codes0[perm]))
    bn = np.array([x['top1_b_r1n'] for x in null], float)
    pos = np.array([x['top10_pos'] for x in null], float)
    ft = np.array([x['top1_f_t'] for x in null], float)
    bnn = np.array([x['top1_b_n'] for x in null], float)
    k = int(np.sum(np.nan_to_num(bn, nan=-np.inf) >= obs['top1_b_r1n']))
    summ = dict(
        B=B, secs=round(time.time() - t0, 1),
        top1_b_r1n_null_mean=float(np.nanmean(bn)), top1_b_r1n_null_sd=float(np.nanstd(bn, ddof=1)),
        top1_b_r1n_null_p=(1 + k) / (1 + B), n_null_ge_obs=k, n_nan=int(np.isnan(bn).sum()),
        top1_b_n_null_median=float(np.median(bnn)),
        top10_pos_null_mean=float(pos.mean()), top10_pos_null_p=(1 + int(np.sum(pos >= obs['top10_pos']))) / (1 + B),
        top1_f_t_null_mean=float(ft.mean()), top1_f_t_null_p=(1 + int(np.sum(ft >= obs['top1_f_t']))) / (1 + B),
        top1_f_t_null_q90=float(np.quantile(ft, 0.9)),
    )
    print(sch, json.dumps(summ, ensure_ascii=False))
    out['schemes'][sch] = dict(summary=summ, draws=null)

json.dump(out, open(os.path.join(C.OUT, 't4_perm_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
