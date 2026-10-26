import json, glob, os, math, collections
E = r'C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/ant_index/emotion'
items = json.load(open(E + '/review_label_200.json', encoding='utf-8'))
pos = {it['id']: i + 1 for i, it in enumerate(items)}
ai = {r['id']: r for r in json.load(open(E + '/labels4/review200_ai.json', encoding='utf-8'))}
user = {}
for f in glob.glob('userlabels100/labels3/*.json'):
    d = json.load(open(f, encoding='utf-8')); user[os.path.basename(f)[:-5]] = d.get('data', d)
EMOS = ['HYPE','JOY','MOCK','DESP','ANGER','FEAR','HOPE','REGRET','ENVY']
TIE = ['MOCK','ANGER','DESP','FEAR','REGRET','HOPE','ENVY','HYPE','JOY']
def prim(e):
    e = {k: (e or {}).get(k, 0) or 0 for k in EMOS}
    m = max(e.values())
    if m == 0: return 'NEUT'
    return [k for k in TIE if e[k] == m][0]
def dirn(p):
    return 'none' if p == 0 else ('down' if p in (1,2) else ('mid' if p == 3 else 'up'))
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); r = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (c - r)/d, (c + r)/d
def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k]*cb[k] for k in set(a)|set(b)) / (n*n)
    return (po - pe)/(1 - pe) if pe < 1 else float('nan')
ids = [i for i in user if i in ai]
print('user saved', len(user), 'matched with AI', len(ids))
missing = sorted(p for p in range(1, 101) if items[p-1]['id'] not in user)
print('missing positions in 1..100:', missing)
print('user positions beyond 100:', sorted(pos[i] for i in user if pos[i] > 100))
# user labels saved before scheme 9 have no HOPE/REGRET/ENVY keys
old = [i for i in ids if 'HOPE' not in (user[i].get('emo') or {})]
print('saved under 6-emotion scheme:', len(old))
pa = [prim(user[i]['emo']) for i in ids]; pb = [prim(ai[i]['emo']) for i in ids]
k = sum(x == y for x, y in zip(pa, pb)); lo, hi = wilson(k, len(ids))
print(f'주된 감정 일치 {k}/{len(ids)} = {k/len(ids):.1%}  윌슨 95% [{lo:.1%}, {hi:.1%}]  κ {kappa(pa,pb):.3f}')
da = [dirn(user[i]['pred']) for i in ids]; db = [dirn(ai[i]['pred']) for i in ids]
k2 = sum(x == y for x, y in zip(da, db)); lo2, hi2 = wilson(k2, len(ids))
print(f'예측 방향 일치 {k2}/{len(ids)} = {k2/len(ids):.1%}  윌슨 95% [{lo2:.1%}, {hi2:.1%}]  κ {kappa(da,db):.3f}')
print('주된 감정 혼동(사용자→AI) 상위:', collections.Counter((x, y) for x, y in zip(pa, pb) if x != y).most_common(8))
print('예측 방향 혼동(사용자→AI):', collections.Counter((x, y) for x, y in zip(da, db) if x != y).most_common(8))
print('감정별 있음/없음 (사용자 n, AI n, 일치, κ):')
for e in EMOS:
    ua = [1 if (user[i]['emo'] or {}).get(e, 0) else 0 for i in ids]; ub = [1 if ai[i]['emo'].get(e, 0) else 0 for i in ids]
    agree = sum(x == y for x, y in zip(ua, ub)) / len(ids)
    print(f'  {e:7s} 사용자 {sum(ua):3d} AI {sum(ub):3d} 일치 {agree:.0%} κ {kappa(ua,ub):.2f}' + ('  (10건 미만: 판정 보류)' if max(sum(ua), sum(ub)) < 10 else ''))
# neutral analysis
un = sum(1 for x in pa if x == 'NEUT'); an = sum(1 for x in pb if x == 'NEUT')
print('중립: 사용자', un, 'AI', an, '둘 다', sum(1 for x, y in zip(pa, pb) if x == y == 'NEUT'))
# by AI confidence
for c in (1, 2, 3):
    sub = [j for j, i in enumerate(ids) if ai[i].get('conf') == c]
    if sub: print(f'AI 확신 {c}: {len(sub)}건, 주된 감정 일치 {sum(pa[j]==pb[j] for j in sub)/len(sub):.0%}, 방향 일치 {sum(da[j]==db[j] for j in sub)/len(sub):.0%}')
oth = [user[i].get('other_emo') for i in ids if user[i].get('other_emo')]
print('직접 쓴 감정:', oth)
json.dump({'n': len(ids), 'prim': [k, len(ids), lo, hi], 'dir': [k2, len(ids), lo2, hi2], 'missing_1_100': missing}, open(E + '/review100_compare.json', 'w', encoding='utf-8'), ensure_ascii=False)
