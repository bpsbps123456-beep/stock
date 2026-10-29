import json, os
import numpy as np, pandas as pd
BASE = r'C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
prim = json.load(open(os.path.join(BASE, 'results', 'primary.json')))
mine = json.load(open(os.path.join(OUT, 'summary.json')))
keys = ['n', 'days', 'syms', 'hit_e2', 'r1_e2', 'r1net_e2', 'race_e2', 'race_resolved']
rep = {}
for arm, name in (('A', 'orig_A'), ('B', 'orig_B')):
    for part in ('front', 'back', 'all', 'ctx_back'):
        for k in keys:
            a, b = mine[arm][part][k], prim[name][part][k]
            d = abs(a - b)
            rep[f'{name}.{part}.{k}'] = (a, b, d)
            if d > 1e-12:
                print('DIFF', name, part, k, a, b)
print('max abs diff over', len(rep), 'fields:', max(v[2] for v in rep.values()))
for arm in ('A', 'B'):
    e = pd.read_csv(os.path.join(BASE, 'results', f'signals_orig_{arm}.csv'))
    m = pd.read_csv(os.path.join(OUT, f'mine_{arm}.csv'))
    ek = set(zip(e.sid, e.ts)); mk = set(zip(m.sid, m.ts))
    print(arm, 'engine', len(ek), 'mine', len(mk), 'only_engine', len(ek - mk), 'only_mine', len(mk - ek))
    j = e.merge(m, on=['sid', 'ts'], suffixes=('_e', '_m'))
    for a, b in (('k_e', 'k_m'), ('t_e', 't_m'), ('S_e', 'S_m'), ('dd_e', 'dd_m'), ('hit_e2', 'hit'), ('r1_e2', 'r1'),
                 ('r1net_e2', 'r1net'), ('race_e2', 'race'), ('day_e', 'day_m'), ('sess_e', 'sess_m')):
        if a not in j.columns:
            a2 = a.replace('race_e2', 'race_e2')
        x = j[a].to_numpy(float); y = j[b].to_numpy(float)
        both_nan = np.isnan(x) & np.isnan(y)
        diff = np.where(both_nan, 0, np.abs(x - y))
        print(f'  {a} vs {b}: mismatches {(np.nan_to_num(diff, nan=1) > 1e-12).sum()}')
    lv = 'level2' if arm == 'A' else None
    if lv:
        print('  level2 vs my level:', (j['level2'] != j['level']).sum(), ' touch2 vs my touch:', (j['touch2'] != j['touch']).sum())
