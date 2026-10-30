"""Task 4(a)/(c) diagnostics.
(a) look-ahead: ATR edges source, cost-quote timestamps vs front window, cost table rebuild from quotes.
(c1) vol_tod warm-up artifact (no minute history before 07-01): size, effect on cells/trees, selection sensitivity.
(c2) clustering: iid vs day vs symbol-day SE for the ② top-1 back mean (is the iid permutation null too narrow?).
(c3) power: minimum detectable effect for the 6 primary hypotheses at the pre-registered Holm level.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from t3_select import select  # noqa: E402

d26, d25, cost, edges, G = C.load_all('last')
E = C.engine()
res = {}

# ---------------------------------------------------------------- (a)
q = pd.read_parquet(os.path.join(C.BASE, 'cache', 'cost_quotes.parquet'))
et = pd.to_datetime(q['ts'] * 60, unit='s', utc=True).dt.tz_convert('America/New_York')
qd = et.dt.strftime('%Y%m%d').astype(int)
print(f'(a) cost quotes: n={len(q)}  ET date range {qd.min()}..{qd.max()}  all within front 20260701..20260831: '
      f'{bool(((qd >= 20260701) & (qd <= 20260831)).all())}')
tab = q.groupby(['tier', 'tod'])['half'].agg(['size', 'median'])
diffs = []
for (t, o), r in tab.iterrows():
    k = f'{t}|{o}'
    e = cost['table'].get(k)
    if e is None or e['n'] != r['size'] or not np.isclose(e['half'], r['median']):
        diffs.append((k, int(r['size']), float(r['median']), e))
byt = q.groupby('tod')['half'].median()
print('    rebuilt per-cell medians that differ from cost_model.json:', diffs if diffs else 'none')
print('    by_tod rebuilt:', {k: round(v, 7) for k, v in byt.items()}, ' engine:', {k: round(v, 7) for k, v in cost['by_tod'].items()})
# do quoted signals exist as front sig rows? (sanity: quotes sampled from front signals)
key26 = set(zip(d26['sym'], d26['ts']))
print('    quotes whose (sym, ts) is a 2026 sig row:', int(sum((s, t) in key26 for s, t in zip(q['sym'], q['ts']))), 'of', len(q),
      '(ts may be signal+offset, so a low count is not by itself a problem)')
f26 = d26[d26['day'] <= C.FRONT_END]
print(f'    ATR edges = exact terciles of {len(f26)} raw 2026 front sig rows (A+B); no back/2025 rows used.')
res['a'] = dict(quote_dates=[int(qd.min()), int(qd.max())], cost_cells_differ=len(diffs))

# ---------------------------------------------------------------- (c1) vol_tod warm-up
print()
for tag, d in (('2026', d26), ('2025', d25)):
    days = np.sort(d['day'].unique())
    first5 = set(days[:5])
    first10 = set(days[:10])
    m5 = d['day'].isin(first5)
    print(f'(c1) {tag}: first 5 trading days {days[0]}..{days[4]} -> {int(m5.sum())} sig rows, vol_tod=inf share {np.isinf(d.loc[m5, "vol_tod"]).mean():.3f};'
          f' rest inf share {np.isinf(d.loc[~m5, "vol_tod"]).mean():.4f}')
for g, P in G.items():
    A = P['y26']
    days = np.unique(A['day'])
    w5 = np.isin(A['day'], days[:5])
    _, ib = C.eval_cell(A, (), 'front')
    vi = C.VAR_KEYS.index('vol_tod')
    s_hi, i_hi = C.eval_cell(A, ((vi, 2),), 'front')
    s5, _ = C.stats(A, ib[w5[ib]]), None
    srest = C.stats(A, ib[~w5[ib]])
    print(f'     {g}: front base dedup n={ib.size}, of which Jul 1-8 = {int(w5[ib].sum())} (net R1 {s5["r1n"]:+.5f}, t {s5["t"]:+.2f})'
          f' vs rest {srest["r1n"]:+.5f}; "시간대거래량 >5배" front n={s_hi["n"]}, Jul 1-8 share {w5[i_hi].mean():.2f}')
    res.setdefault('c1', {})[g] = dict(first5_n=int(w5[ib].sum()), first5_r1n=s5['r1n'], rest_r1n=srest['r1n'],
                                         vol_hi_n=s_hi['n'], vol_hi_first5_share=float(w5[i_hi].mean()))

# share of cells in each group's front top-10 / tree that touch vol_tod
for g in G:
    t10 = [c['cell'] for c in E['groups'][g]['top10']]
    nv = sum('시간대거래량' in c for c in t10)
    print(f'     {g}: top-10 cells using 시간대거래량 = {nv}; tree rule uses vol_tod = {"vol_tod" in E["groups"][g]["tree"]["rule"]}')

# selection sensitivity: vol_tod undefined (NaN) on each window's first 10 trading days
print('     selection sensitivity (vol_tod -> NaN on first 10 trading days of 2026 and 2025):')
cells = C.all_cells()
vi = C.VAR_KEYS.index('vol_tod')
for g, P in G.items():
    A = P['y26']
    codes = A['codes'].copy()
    days = np.unique(A['day'])
    codes[np.isin(A['day'], days[:10]), vi] = -1
    df = select(A, cells, codes)
    el = df[df.elig].sort_values('t', ascending=False, kind='mergesort').reset_index(drop=True)
    top = [C.cell_label(cells[int(c)]) for c in el.ci[:5]]
    tt = el.t.values[:5]
    A25 = P['y25']
    codes25 = A25['codes'].copy()
    d25u = np.unique(A25['day'])
    codes25[np.isin(A25['day'], d25u[:10]), vi] = -1
    c1 = cells[int(el.ci[0])]
    sb, _ = C.eval_cell(A, c1, 'back', codes)
    sy, _ = C.eval_cell(A25, c1, 'all', codes25)
    print(f'       {g}: eligible {int(df.elig.sum())}; top-5 = {[f"{c} ({t:.2f})" for c, t in zip(top, tt)]}')
    print(f'          top-1 back n={sb.get("n", 0)} r1n={sb.get("r1n", np.nan):+.5f}; 2025 n={sy.get("n", 0)} r1n={sy.get("r1n", np.nan):+.5f}'
          f'  (engine top-1: {E["groups"][g]["top10"][0]["cell"]})')
    res['c1'][g]['sens_top5'] = list(zip(top, map(float, tt)))

# ---------------------------------------------------------------- (c2) clustering of the ② top-1 back mean
print()
A = G['②']['y26']
for lbl in ['실적일 실적일']:
    s, idx = C.eval_cell(A, C.parse_label(lbl), 'back')
    v = A['r1n'][idx]
    n = v.size
    se_iid = v.std(ddof=1) / np.sqrt(n)

    def cl_se(keys):
        uk, inv = np.unique(keys, return_inverse=True)
        Gk = len(uk)
        S = np.bincount(inv, weights=v - v.mean())
        return np.sqrt(Gk / (Gk - 1) * np.sum(S ** 2)) / n, Gk
    se_day, gd = cl_se(A['day'][idx])
    se_sd, gsd = cl_se(A['day'][idx].astype(np.int64) * 1000 + A['sid'][idx])
    print(f'(c2) ② top-1 back: n={n}, mean={v.mean():+.5f}; SE iid={se_iid:.5f}, day-cluster({gd})={se_day:.5f}, sym-day-cluster({gsd})={se_sd:.5f}')
    _, ib = C.eval_cell(A, (), 'back')
    sd_back = A['r1n'][ib].std(ddof=1)
    print(f'     ② base back per-signal SD={sd_back:.5f}; iid SE at n=52 (median null top-1 back n) = {sd_back / np.sqrt(52):.5f}'
          f' vs pool-perm null SD 0.00143 -> the pool null behaves like iid draws')
    for nm, se in (('iid', se_iid), ('day', se_day), ('sym-day', se_sd)):
        from math import erf, sqrt
        z = (v.mean() - (-0.000895)) / se
        p = 0.5 * (1 - erf(z / sqrt(2)))
        print(f'     z vs engine null mean (-0.000895) with {nm:7s} SE: z={z:.2f}, normal one-sided p={p:.3f}')
    res['c2'] = dict(n=int(n), se_iid=se_iid, se_day=se_day, se_symday=se_sd)

# ---------------------------------------------------------------- (c3) power / MDE
print()
from statistics import NormalDist
za_holm = NormalDist().inv_cdf(1 - 0.05 / 6)
za = NormalDist().inv_cdf(0.95)
zb = NormalDist().inv_cdf(0.80)
print(f'(c3) MDE (80% power) = (z_alpha + z_0.8) x day-cluster SE; z_alpha Holm-first = {za_holm:.2f}, unadjusted = {za:.2f}')
rows = []
for g, P in G.items():
    eg = E['groups'][g]
    cell = C.parse_label(eg['top10'][0]['cell'])
    chain = any(vi_ == C.VAR_KEYS.index('chain') for vi_, _ in cell)
    A, A25 = P['y26'], P['y25']
    s, idx = (C.eval_cell(A25, cell, 'all') if chain else C.eval_cell(A, cell, 'back'))
    se = abs(s['r1n'] / s['t'])
    mde_h = (za_holm + zb) * se
    mde_u = (za + zb) * se
    win = '2025' if chain else 'back'
    print(f'     {g} 1위칸 [{win}] n={s["n"]} days={s["days"]}: SE={se:.5f}  MDE(Holm)={mde_h:+.4f}  MDE(unadj)={mde_u:+.4f}  observed={s["r1n"]:+.5f}'
          f'  | base back net R1={eg["base_back"]["r1n"]:+.5f}')
    rows.append(dict(group=g, window=win, n=s['n'], se=se, mde_holm=mde_h, mde_unadj=mde_u, obs=s['r1n']))
res['c3'] = rows

json.dump(res, open(os.path.join(C.OUT, 't6_checks_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
