import json, glob, os, math, collections
E = r'C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/ant_index/emotion'
test = json.load(open(E + '/review_test117.json', encoding='utf-8'))
tid = [t['id'] for t in test]
ai = {r['id']: r for r in json.load(open(E + '/labels5/test117_final.json', encoding='utf-8'))}
user = {}
for f in glob.glob('userlabels_final/labels3/*.json'):
    d = json.load(open(f, encoding='utf-8')); user[os.path.basename(f)[:-5]] = d.get('data', d)
EMOS = ['HYPE','JOY','MOCK','DESP','ANGER','FEAR','HOPE','REGRET','ENVY']
TIE = ['MOCK','ANGER','DESP','FEAR','REGRET','HOPE','ENVY','HYPE','JOY']
def prim(e):
    e = {k: (e or {}).get(k, 0) or 0 for k in EMOS}; m = max(e.values())
    return 'NEUT' if m == 0 else [k for k in TIE if e[k] == m][0]
def dirn(p): return 'none' if p == 0 else ('down' if p in (1,2) else ('mid' if p == 3 else 'up'))
def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n; c = p+z*z/(2*n); r = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (c-r)/d, (c+r)/d
def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b))/n; ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k]*cb[k] for k in set(a)|set(b))/(n*n); return (po-pe)/(1-pe) if pe < 1 else float('nan')
pre = {'L3b6b7daf', 'L60cb9f67'}  # 사용자가 AI 고정(15:24 KST) 전에 매긴 2건
def run(ids, tag):
    pa = [prim(user[i]['emo']) for i in ids]; pb = [prim(ai[i]['emo']) for i in ids]
    da = [dirn(user[i]['pred']) for i in ids]; db = [dirn(ai[i]['pred']) for i in ids]
    k1 = sum(x == y for x, y in zip(pa, pb)); k2 = sum(x == y for x, y in zip(da, db)); n = len(ids)
    l1, h1 = wilson(k1, n); l2, h2 = wilson(k2, n)
    print(f'[{tag}] n={n}')
    print(f'  주된 감정 일치 {k1}/{n} = {k1/n:.1%}  윌슨 [{l1:.1%}, {h1:.1%}]  κ {kappa(pa,pb):.3f}')
    print(f'  예측 방향 일치 {k2}/{n} = {k2/n:.1%}  윌슨 [{l2:.1%}, {h2:.1%}]  κ {kappa(da,db):.3f}')
    return pa, pb, da, db
missing = [i for i in tid if i not in user]
print('시험 117 중 사용자 판정 없음:', len(missing))
ids = [i for i in tid if i in user]
pa, pb, da, db = run(ids, '시험 전체')
run([i for i in ids if i not in pre], '고정 전 2건 제외')
print('주된 감정 혼동(사용자→AI) 상위:', collections.Counter((x, y) for x, y in zip(pa, pb) if x != y).most_common(10))
print('예측 방향 혼동(사용자→AI):', collections.Counter((x, y) for x, y in zip(da, db) if x != y).most_common(8))
print('감정별 있음/없음:')
res = {}
for e in EMOS:
    ua = [1 if user[i]['emo'].get(e, 0) else 0 for i in ids]; ub = [1 if ai[i]['emo'].get(e, 0) else 0 for i in ids]
    kk = kappa(ua, ub); judge = '보류(<10)' if max(sum(ua), sum(ub)) < 10 else ('통과' if kk >= 0.6 else '미달')
    res[e] = [sum(ua), sum(ub), kk, judge]
    print(f'  {e:7s} 사용자 {sum(ua):3d} AI {sum(ub):3d} 일치 {sum(x==y for x,y in zip(ua,ub))/len(ids):.0%} κ {kk:.2f}  {judge}')
sp = [i for i in ids if ai[i].get('split_fields')]
if sp:
    print('AI 두 명이 갈린 단위', len(sp), '주된 감정 일치', f"{sum(prim(user[i]['emo'])==prim(ai[i]['emo']) for i in sp)/len(sp):.0%}",
          '/ 안 갈린 단위', f"{sum(prim(user[i]['emo'])==prim(ai[i]['emo']) for i in ids if i not in sp)/(len(ids)-len(sp)):.0%}")
print('사용자 직접 쓴 감정:', [user[i].get('other_emo') for i in user if user[i].get('other_emo')])
json.dump({'n': len(ids), 'prim': [sum(x==y for x,y in zip(pa,pb)), len(ids)], 'dir': [sum(x==y for x,y in zip(da,db)), len(ids)], 'per_emotion': res},
          open(E + '/review_test117_compare.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# --- 설명용(사후, 판정 아님) ---
calib = json.load(open(E + '/review_calib83_user.json', encoding='utf-8'))
cid = [c['id'] for c in calib]
def share_neut(idl, src): return sum(prim(src[i]['emo']) == 'NEUT' for i in idl)/len(idl)
def mean_emos(idl, src): return sum(sum(1 for e in EMOS if src[i]['emo'].get(e, 0)) for i in idl)/len(idl)
print('\n[설명용] 사용자 중립 몫: 보정', f'{share_neut(cid, user):.0%}', '시험', f'{share_neut(ids, user):.0%}', '| AI v2 시험', f'{share_neut(ids, ai):.0%}')
print('[설명용] 단위당 감정 수: 사용자 보정', f'{mean_emos(cid, user):.2f}', '시험', f'{mean_emos(ids, user):.2f}', '| AI 시험', f'{mean_emos(ids, ai):.2f}')
NEG = {'MOCK','DESP','ANGER','FEAR','REGRET'}; POS = {'HYPE','JOY','HOPE','ENVY'}
def val(e):
    n = sum(e.get(k, 0) for k in NEG); p = sum(e.get(k, 0) for k in POS)
    return 'neu' if n == 0 and p == 0 else ('neg' if n >= p else 'pos')
va = [val(user[i]['emo']) for i in ids]; vb = [val(ai[i]['emo']) for i in ids]
k = sum(x == y for x, y in zip(va, vb)); print(f'[설명용·사후] 3분류(부정/긍정/중립) 일치 {k}/{len(ids)} = {k/len(ids):.0%}, κ {kappa(va, vb):.2f}')
h1 = [i for i in ids if test[tid.index(i)]['pos'] <= 100]; h2 = [i for i in ids if test[tid.index(i)]['pos'] > 100]
for nm, sub in (('건너뛴 19건(1~100번)', h1), ('103~200번', h2)):
    print(f'[설명용] {nm}: n={len(sub)} 주된 감정 일치 {sum(prim(user[i]["emo"])==prim(ai[i]["emo"]) for i in sub)/len(sub):.0%}, 사용자 중립 {share_neut(sub, user):.0%}')
