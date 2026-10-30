"""Tree-rule leaves and the 6 primary hypotheses (n, r1n, vs_bt, own cluster-bootstrap p, Holm),
plus an independent sklearn refit of the depth-3 tree (leaf min 80) to see whether the same rule comes out."""
import json
import os
import re
import sys

import numpy as np
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

d26, d25, cost, edges, G = C.load_all('last')
E = C.engine()
FIELDS = ['n', 'days', 'syms', 'share', 'hit', 'r1n', 't', 'race', 'r1', 'vs_bt']


def rule_mask(codes, rule):
    m = np.ones(codes.shape[0], bool)
    for part in rule.split('&'):
        v, op, thr = re.match(r'\s*(\w+)\s*(<=|>)\s*([-\d.]+)\s*', part).groups()
        col = codes[:, C.VAR_KEYS.index(v)]
        m &= (col <= float(thr)) if op == '<=' else (col > float(thr))
    return m


def eval_mask(A, mask, period):
    pm = {'front': A['front'], 'back': A['back'], 'all': np.ones_like(A['front'])}[period]
    idx = C.dedup(np.flatnonzero(pm & mask), A['sid'], A['ts'])
    return C.stats(A, idx), idx


def same(a, b):
    return all(np.isclose(a.get(f, np.nan), b.get(f, np.nan), rtol=1e-9, atol=1e-12, equal_nan=True) for f in FIELDS if f in b)


out = {'tree': {}, 'hyps': {}, 'refit': {}}
hyp_rows = []
for g, P in G.items():
    eg = E['groups'][g]
    rule = eg['tree']['rule']
    A, A25 = P['y26'], P['y25']
    sf, _ = eval_mask(A, rule_mask(A['codes'], rule), 'front')
    sb, ib = eval_mask(A, rule_mask(A['codes'], rule), 'back')
    sy, iy = eval_mask(A25, rule_mask(A25['codes'], rule), 'all')
    ok = [same(sf, eg['tree']['front']), same(sb, eg['tree']['back']), same(sy, eg['tree']['y2025'])]
    print(f'{g} tree rule "{rule}": front n={sf["n"]}/{eg["tree"]["front"]["n"]:.0f} back n={sb["n"]}/{eg["tree"]["back"]["n"]:.0f}'
          f' 2025 n={sy["n"]}/{eg["tree"]["y2025"]["n"]:.0f}  identical(front,back,2025)={ok}  front share={sf["share"]:.3f}')
    # which days dominate the tree leaf on front
    _, i_f = eval_mask(A, rule_mask(A['codes'], rule), 'front')
    days, cnt = np.unique(A['day'][i_f], return_counts=True)
    top = sorted(zip(cnt, days), reverse=True)[:3]
    print('     front leaf top days:', [(int(d), int(c)) for c, d in top])
    out['tree'][g] = dict(rule=rule, front=sf, back=sb, y2025=sy, identical=ok, top_days=[(int(d), int(c)) for c, d in top])

    # primary hypotheses: 1위칸 and 나무
    top1 = C.parse_label(eg['top10'][0]['cell'])
    chain_cell = any(vi == C.VAR_KEYS.index('chain') for vi, _ in top1)
    for kind, mk26, mk25, label, is_chain in (
            ('1위칸', C.cell_mask(A['codes'], top1), C.cell_mask(A25['codes'], top1), eg['top10'][0]['cell'], chain_cell),
            ('나무', rule_mask(A['codes'], rule), rule_mask(A25['codes'], rule), rule, 'chain' in rule)):
        rb, ib = eval_mask(A, mk26, 'back')
        ry, iy = eval_mask(A25, mk25, 'all')
        pb = dict(p_r1n=C.cluster_boot_p(A, ib, 'r1n') if rb['n'] >= 10 else 1.0,
                  p_bt=C.cluster_boot_p(A, ib, 'bt') if rb['n'] >= 10 else 1.0)
        py = dict(p_r1n=C.cluster_boot_p(A25, iy, 'r1n') if ry['n'] >= 10 else 1.0,
                  p_bt=C.cluster_boot_p(A25, iy, 'bt') if ry['n'] >= 10 else 1.0)
        p_hyp = py['p_r1n'] if is_chain else pb['p_r1n']
        key = f'{g} {kind}'
        eh = E['hyps'][key]
        hyp_rows.append(dict(key=key, back_n=rb['n'], back_r1n=rb.get('r1n'), back_vs_bt=rb.get('vs_bt'), y25_n=ry['n'],
                             y25_r1n=ry.get('r1n'), **{f'b_{k}': v for k, v in pb.items()}, **{f'y_{k}': v for k, v in py.items()},
                             p_hyp=p_hyp, eng=eh))

# Holm on my p_hyp
ps = np.array([h['p_hyp'] for h in hyp_rows])
order = np.argsort(ps)
holm = np.empty_like(ps)
run = 0
for r, i in enumerate(order):
    run = max(run, min(1.0, (len(ps) - r) * ps[i]))
    holm[i] = run
print()
print(f'{"hyp":14s} {"back n":>10s} {"back r1n (mine/eng)":>24s} {"p_r1n back (mine/eng)":>22s} {"p_bt back":>16s} {"2025 r1n":>22s} {"p_hyp":>14s} {"Holm":>14s}')
for h, hb in zip(hyp_rows, holm):
    e = h['eng']
    print(f'{h["key"]:14s} {h["back_n"]:4d}/{e["back"]["n"]:4d} {h["back_r1n"] if h["back_n"] else float("nan"):+.5f}/{e["back"].get("r1n", float("nan")):+.5f}'
          f'   {h["b_p_r1n"]:.3f}/{e["back"]["p_r1n"]:.3f}   {h["b_p_bt"]:.3f}/{e["back"]["p_bt"]:.3f}'
          f'  {h["y25_r1n"] if h["y25_n"] else float("nan"):+.5f}/{e["y2025"].get("r1n", float("nan")):+.5f}'
          f'  {h["p_hyp"]:.3f}/{e["p_hyp"]:.3f}  {hb:.3f}/{e["p_holm"]:.3f}')
    h['holm'] = float(hb)
out['hyps'] = hyp_rows

# independent refit of the tree on front, features = bin codes (NaN -> -1), target = net R1
print()
for g, P in G.items():
    A = P['y26']
    res = {}
    for train in ('dedup_base', 'raw'):
        if train == 'dedup_base':
            _, idx = C.eval_cell(A, (), 'front')
        else:
            idx = np.flatnonzero(A['front'])
        X = A['codes'][idx].astype(float)
        y = A['r1n'][idx]
        rules = set()
        for seed in range(5):
            tr = DecisionTreeRegressor(max_depth=3, min_samples_leaf=80, random_state=seed).fit(X, y)
            T = tr.tree_
            leaves = np.flatnonzero(T.children_left == -1)
            best = leaves[np.argmax(T.value[leaves, 0, 0])]
            # path to best leaf
            parent = {}
            for nd in range(T.node_count):
                for ch, side in ((T.children_left[nd], '<='), (T.children_right[nd], '>')):
                    if ch != -1:
                        parent[ch] = (nd, side)
            conds = []
            nd = best
            while nd in parent:
                p, side = parent[nd]
                conds.append(f'{C.VAR_KEYS[T.feature[p]]} {side} {T.threshold[p]:.2f}')
                nd = p
            rules.add(' & '.join(reversed(conds)))
        res[train] = sorted(rules)
        print(f'{g} refit on {train:10s} (n={idx.size}): best-leaf rule(s) over 5 seeds = {sorted(rules)}  | engine: {E["groups"][g]["tree"]["rule"]}')
    out['refit'][g] = res

json.dump(out, open(os.path.join(C.OUT, 't5_tree_hyps_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
