"""Independent re-implementation of the BRIEF2 condition evaluation (audit only).

Written from BRIEF2.md text + auditor notes, without reading code/cond_eval.py.
Reads cache/cond_*.parquet and results/cost_model.json; writes nothing outside
audit/cond-recompute/.
"""
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT = os.path.dirname(os.path.abspath(__file__))
FRONT_END = 20260831
SEP = 30  # minutes, same-symbol overlap window

# ---------------------------------------------------------------- variables
# Each variable -> (Korean label used by the engine, list of bin labels in code order)
VARS = [
    ('cap', '시총', ['<50억$', '50억~200억$', '200억~1000억$', '1000억$+']),
    ('adv', '거래대금', ['<1억$', '1억~10억$', '10억$+']),
    ('px', '주가', ['<20$', '20~100$', '100$+']),
    ('ai', 'AI종목', ['아님', 'AI']),
    ('tod', '시간대', ['장전', '개장15분', '오전', '장후']),
    ('dayret', '당일낙폭', ['<-10%', '-10~-5%', '-5~-2%', '>-2%']),
    ('qqq_day', 'QQQ당일', ['<-1%', '-1~0%', '>0%']),
    ('qqq_30', 'QQQ30분', ['<-0.3%', '-0.3~0%', '>0%']),
    ('rel', '상대낙폭', ['<-5%', '-5~-2%', '>-2%']),
    ('vol_tod', '시간대거래량', ['<2배', '2~5배', '>5배']),
    ('vwap', 'VWAP이격', ['<-3%', '-3~-1%', '>-1%']),
    ('dd', '120분하락', ['2~3%', '3~5%', '>5%']),
    ('ret5d', '5일수익률', ['<-5%', '-5~5%', '>5%']),
    ('atrd', '일봉ATR', ['낮음', '중간', '높음']),
    ('earn', '실적일', ['아님', '실적일', '모름']),
    ('chain', '연쇄', ['첫 신호', '두번째+']),
    ('kcnt', '지지선수', ['적음', '많음']),
    ('recsize', '회복크기', ['<0.3%', '0.3~0.6%', '>0.6%']),
]
VAR_KEYS = [v[0] for v in VARS]
LABEL = {}
for key, kl, bins in VARS:
    for b, bl in enumerate(bins):
        LABEL[(key, b)] = f'{kl} {bl}'
LABEL_TO_BIN = {v: k for k, v in LABEL.items()}


def _digitize(x, edges):
    """Right-open bins [e_i, e_{i+1}); NaN -> -1. inf -> last bin (inf > every edge)."""
    x = np.asarray(x, float)
    code = np.searchsorted(np.asarray(edges, float), x, side='right').astype(np.int64)
    code[np.isnan(x)] = -1
    return code


def load(year):
    d = pd.read_parquet(os.path.join(BASE, 'cache', f'cond_{year}.parquet'))
    d = d[d['arm'] == 'sig'].copy()
    return d


ATR_Q = (100 / 3, 200 / 3)  # exact terciles; auditor note said 33.3/66.7 (edges differ by <1e-4)


def atr_edges_from_front(d26, q=None):
    """Terciles of daily ATR% over raw (pre-dedup) 2026 front sig rows, A and B together."""
    f = d26[d26['day'] <= FRONT_END]
    q = ATR_Q if q is None else q
    v = f['atrd'].dropna().values
    return [float(np.percentile(v, q[0])), float(np.percentile(v, q[1]))]


def add_codes(d, atr_edges, inf_policy='last'):
    """Bin codes per BRIEF2 variable table. -1 = excluded (NaN)."""
    c = {}
    c['cap'] = _digitize(d['cap_b'], [5, 20, 100])            # <5 (incl. <1B) / 5-20 / 20-100 / 100+
    c['adv'] = _digitize(d['adv20'], [1e8, 1e9])
    c['px'] = _digitize(d['prevclose'], [20, 100])
    c['ai'] = d['ai'].astype(bool).values.astype(np.int64)
    m = d['mins'].values
    tod = np.full(len(d), -1, np.int64)
    tod[(m >= 240) & (m < 570)] = 0
    tod[(m >= 570) & (m < 585)] = 1
    tod[(m >= 585) & (m < 660)] = 2
    tod[(m >= 1140) & (m < 1170)] = 3
    c['tod'] = tod
    c['dayret'] = _digitize(d['dayret'], [-0.10, -0.05, -0.02])
    c['qqq_day'] = _digitize(d['qqq_day'], [-0.01, 0.0])
    c['qqq_30'] = _digitize(d['qqq_30'], [-0.003, 0.0])
    c['rel'] = _digitize(d['rel'], [-0.05, -0.02])
    vt = d['vol_tod'].values.astype(float).copy()
    if inf_policy == 'nan':
        vt[np.isinf(vt)] = np.nan
    c['vol_tod'] = _digitize(vt, [2, 5])
    c['vwap'] = _digitize(d['vwap_gap'], [-0.03, -0.01])
    dd = _digitize(d['dd'], [0.03, 0.05])
    c['dd'] = dd
    c['ret5d'] = _digitize(d['ret5d'], [-0.05, 0.05])
    c['atrd'] = _digitize(d['atrd'], atr_edges)
    e = d['earn'].values
    c['earn'] = np.select([e == 0, e == 1, e == -1], [0, 1, 2], -1).astype(np.int64)
    c['chain'] = (d['chain'].values >= 1).astype(np.int64)
    k = d['kcnt'].values
    isA = (d['pat'].values == 'A')
    many = np.where(isA, k >= 3, k >= 2)
    c['kcnt'] = many.astype(np.int64)
    c['recsize'] = _digitize(d['recsize'], [0.003, 0.006])
    return np.column_stack([c[k] for k in VAR_KEYS])  # (n, 18)


def cost_arrays(d, cost):
    table, by_tod = cost['table'], cost['by_tod']
    cap = d['cap_b'].values
    tier = np.select([cap < 5, cap < 20, cap < 100], ['<5B', '5-20B', '20-100B'], '100B+')
    m = d['mins'].values
    tod = np.select([m < 570, m < 585, m < 660], ['pre', 'open15', 'am'], 'post')
    xtod = pd.Series(d['sess_out'].values).map({0: 'pre', 1: 'am', 2: 'am', 3: 'post'}).values

    def half(t, o):
        r = table.get(f'{t}|{o}')
        return r['half'] if r is not None else by_tod[o]
    h_in = np.array([half(t, o) for t, o in zip(tier, tod)])
    h_out = np.array([half(t, o) for t, o in zip(tier, xtod)])
    return h_in, h_out


def prepare(group, d26, d25, cost, atr_edges, inf_policy='last'):
    """Return dict of numpy arrays for a group, sorted by (sid, ts, pat)."""
    pats = {'①': ['A'], '②': ['B'], '아무거나': ['A', 'B']}[group]
    out = {}
    for tag, d in (('y26', d26), ('y25', d25)):
        x = d[d['pat'].isin(pats)].sort_values(['sid', 'ts', 'pat'], kind='mergesort').reset_index(drop=True)
        h_in, h_out = cost_arrays(x, cost)
        hit = x['hit_e2'].values.astype(float)
        r1 = x['r1_e2'].values.astype(float)
        r1n = r1 - h_in - np.where(hit == 1, 0.0, h_out)
        day = x['day'].values
        uday, dcode = np.unique(day, return_inverse=True)
        out[tag] = dict(
            df=x, codes=add_codes(x, atr_edges, inf_policy), sid=x['sid'].values, ts=x['ts'].values,
            day=day, dcode=dcode, ndays=len(uday), r1=r1, r1n=r1n, hit=hit,
            race=x['race_e2'].values.astype(float), bt=x['bt2_r1'].values.astype(float),
            front=(day <= FRONT_END), back=(day > FRONT_END), pat=x['pat'].values,
        )
    return out


# ---------------------------------------------------------------- core ops
def dedup(idx, sid, ts):
    """Greedy: in (sid, ts, pat) order, drop a row if the same symbol has a kept row < SEP min earlier."""
    if idx.size == 0:
        return idx
    s = sid[idx]
    t = ts[idx]
    n = idx.size
    amb = np.zeros(n, bool)
    amb[1:] = (s[1:] == s[:-1]) & ((t[1:] - t[:-1]) < SEP)
    if not amb.any():
        return idx
    keep = np.ones(n, bool)
    lk = t.copy()
    for j in np.flatnonzero(amb):
        p = lk[j - 1]
        if t[j] - p < SEP:
            keep[j] = False
            lk[j] = p
    return idx[keep]


def stats(A, idx):
    n = idx.size
    if n == 0:
        return dict(n=0)
    r1n = A['r1n'][idx]
    dc = A['dcode'][idx]
    nd = np.bincount(dc, minlength=A['ndays'])
    S = np.bincount(dc, weights=r1n, minlength=A['ndays'])
    G = int((nd > 0).sum())
    mean = float(r1n.mean())
    if G > 1:
        se = np.sqrt(G / (G - 1) * np.sum((S - mean * nd) ** 2)) / n
        t = mean / se if se > 0 else np.nan
    else:
        t = np.nan
    race = A['race'][idx]
    return dict(
        n=int(n), days=G, syms=int(np.unique(A['sid'][idx]).size), share=float(nd.max() / n),
        r1n=mean, t=float(t), hit=float(A['hit'][idx].mean()),
        race=float(np.nanmean(race)) if np.isfinite(race).any() else np.nan,
        r1=float(A['r1'][idx].mean()), vs_bt=float((A['r1'][idx] - A['bt'][idx]).mean()),
    )


def eligible(s):
    return s.get('n', 0) >= 60 and s['days'] >= 15 and s['syms'] >= 10 and s['share'] <= 0.20


def cell_mask(codes, cell):
    m = np.ones(codes.shape[0], bool)
    for (vi, b) in cell:
        m &= codes[:, vi] == b
    return m


def eval_cell(A, cell, period, codes=None):
    codes = A['codes'] if codes is None else codes
    pm = {'front': A['front'], 'back': A['back'], 'all': np.ones_like(A['front'])}[period]
    idx = np.flatnonzero(pm & cell_mask(codes, cell))
    idx = dedup(idx, A['sid'], A['ts'])
    return stats(A, idx), idx


def all_cells():
    nb = [len(v[2]) for v in VARS]
    cells = []
    for i in range(len(VARS)):
        for b in range(nb[i]):
            cells.append(((i, b),))
    for i in range(len(VARS)):
        for j in range(i + 1, len(VARS)):
            for a in range(nb[i]):
                for b in range(nb[j]):
                    cells.append(((i, a), (j, b)))
    return cells


def cell_label(cell):
    return ' & '.join(LABEL[(VAR_KEYS[vi], b)] for vi, b in cell)


def parse_label(lbl):
    parts = [p.strip() for p in lbl.split('&')]
    cell = []
    for p in parts:
        key, b = LABEL_TO_BIN[p]
        cell.append((VAR_KEYS.index(key), b))
    return tuple(sorted(cell))


def cluster_boot_p(A, idx, field='r1n', B=2000, seed=0):
    """One-sided day-cluster bootstrap p for mean > 0: (1 + #{boot mean <= 0}) / (1 + B)."""
    if idx.size == 0:
        return np.nan
    v = A['r1n'][idx] if field == 'r1n' else (A['r1'][idx] - A['bt'][idx])
    dc = A['dcode'][idx]
    ud, inv = np.unique(dc, return_inverse=True)
    S = np.bincount(inv, weights=v)
    N = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    G = len(ud)
    pick = rng.integers(0, G, size=(B, G))
    m = S[pick].sum(1) / N[pick].sum(1)
    return float((1 + np.sum(m <= 0)) / (1 + B))


def load_all(inf_policy='last'):
    d26, d25 = load(2026), load(2025)
    cost = json.load(open(os.path.join(BASE, 'results', 'cost_model.json'), encoding='utf-8'))
    edges = atr_edges_from_front(d26)
    groups = {g: prepare(g, d26, d25, cost, edges, inf_policy) for g in ('①', '②', '아무거나')}
    return d26, d25, cost, edges, groups


def engine():
    return json.load(open(os.path.join(BASE, 'results', 'cond_primary.json'), encoding='utf-8'))
