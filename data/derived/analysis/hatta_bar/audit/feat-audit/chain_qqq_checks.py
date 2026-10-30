"""전체 cond_2026 행: chain 일관성, QQQ 전일 종가 역산 일관성. 성과 열은 읽지 않는다."""
import numpy as np
import pandas as pd

from feat_audit import BASE, load_1m, load_cond

df = load_cond()

# ---------------- chain ----------------
print('=== chain ===')
df = df.sort_values(['arm', 'sym', 'pat', 'day', 'ts']).copy()
g = df.groupby(['arm', 'sym', 'pat', 'day'], sort=False)
df['file_cnt'] = g.cumcount()
df['prev_chain'] = g['chain'].shift()
for arm in ['sig', 'ctx']:
    a = df[df.arm == arm]
    nonmono = (a.prev_chain.notna() & (a.chain <= a.prev_chain)).sum()
    weak_nonmono = (a.prev_chain.notna() & (a.chain < a.prev_chain)).sum()
    z_notfirst = ((a.chain == 0) & (a.file_cnt > 0)).sum()
    below = (a.chain < a.file_cnt).sum()
    dis = (a.chain != a.file_cnt).sum()
    print(f'{arm}: rows {len(a)} | chain not strictly increasing {nonmono} (decreasing {weak_nonmono}) | '
          f'chain==0 but not first file row {z_notfirst} | chain < file count {below} | '
          f'chain != file count {dis} ({dis / len(a):.1%})')
    for pat in ['A', 'B']:
        b = a[a.pat == pat]
        print(f'    {pat}: chain != file count {(b.chain != b.file_cnt).sum()}/{len(b)} '
              f'({(b.chain != b.file_cnt).mean():.1%}); by sess '
              + str(b.assign(x=b.chain != b.file_cnt).groupby('sess').x.mean().round(3).to_dict()))

# ctx 행의 chain이 무엇을 세는가: 같은 종목·묶음·날의 앞선 sig 행 수와 비교
sig = df[df.arm == 'sig'][['sym', 'pat', 'day', 'ts']]
ctx = df[df.arm == 'ctx'][['sym', 'pat', 'day', 'ts', 'chain', 'file_cnt']].copy()
sk = sig.groupby(['sym', 'pat', 'day']).ts.apply(np.sort).to_dict()
ctx['sig_before'] = [int(np.searchsorted(sk.get((s, p, d), np.array([], dtype='int64')), t, side='left'))
                     for s, p, d, t in zip(ctx.sym, ctx.pat, ctx.day, ctx.ts)]
print('ctx: chain == earlier ctx file rows', (ctx.chain == ctx.file_cnt).mean().round(4),
      '| chain == earlier sig file rows', (ctx.chain == ctx.sig_before).mean().round(4),
      '| chain < earlier sig rows', int((ctx.chain < ctx.sig_before).sum()),
      '| chain < earlier ctx rows', int((ctx.chain < ctx.file_cnt).sum()))

# AI 종목: 1차 원안 신호 목록(필터 전, 키 열만)으로 '앞선 신호 수' 재계산
so = {}
for pat in ['A', 'B']:
    s = pd.read_csv(f'{BASE}/results/signals_orig_{pat}.csv', usecols=['sym', 'ts', 'day', 'tradable', 'valid'])
    so[pat] = s
    a = df[(df.arm == 'sig') & (df.pat == pat)]
    a = a[a.sym.isin(set(s.sym))]
    keys = s.groupby(['sym', 'day']).ts.apply(np.sort).to_dict()
    cnt = np.array([int(np.searchsorted(keys.get((sy, d), np.array([], dtype='int64')), t, side='left'))
                    for sy, d, t in zip(a.sym, a.day, a.ts)])
    member = a.set_index(['sym', 'ts']).index.isin(s.set_index(['sym', 'ts']).index)
    tv = s[s.tradable & s.valid].set_index(['sym', 'ts']).index
    member_tv = a.set_index(['sym', 'ts']).index.isin(tv)
    print(f'AI sig {pat}: rows {len(a)} | in signals_orig {member.sum()} (tradable&valid {member_tv.sum()}) | '
          f'chain == earlier signals_orig(all, same day) {(a.chain.to_numpy() == cnt).sum()}')
    tv_rows = s[s.tradable & s.valid & (s.day >= 20260701)]
    in_cond = tv_rows.set_index(['sym', 'ts']).index.isin(a.set_index(['sym', 'ts']).index)
    print(f'    signals_orig tradable&valid (AI, 2026-07-01~) {len(tv_rows)} | in cond sig {in_cond.sum()}')
    bad = a[a.chain.to_numpy() != cnt]
    if len(bad):
        print(bad[['sym', 'day', 'ts', 'mins', 'chain']].assign(cnt=cnt[a.chain.to_numpy() != cnt]).head(10).to_string())

# ---------------- QQQ 전일 종가(엔진) 역산 ----------------
print('\n=== QQQ prevclose implied by engine qqq_day ===')
q = load_1m('QQQ')
pos = np.searchsorted(q['ts'], df.ts.to_numpy(), side='right') - 1
qnow = q['Close'][pos]
df['implied'] = qnow / (1 + df.qqq_day.to_numpy())
df['q_now_same_day'] = q['day'][pos] == df.day.to_numpy()
per = df.groupby('day').implied.agg(['median', 'min', 'max', 'size'])
per['spread'] = per['max'] / per['min'] - 1
print('days', len(per), '| max within-day relative spread of implied prevclose: %.2e' % per.spread.max(),
      '| q_now from another day', int((~df.q_now_same_day).sum()))
q_days = np.unique(q['day'])
rows = []
for d, r in per.iterrows():
    pdv = q_days[q_days < d][-1]
    sel = np.flatnonzero((q['day'] == pdv) & (q['mins'] <= 959))
    s16 = np.flatnonzero((q['day'] == pdv) & (q['mins'] == 960))
    rows.append((d, pdv, r['median'], q['Close'][sel[-1]], q['Open'][s16[0]] if len(s16) else np.nan))
cmp = pd.DataFrame(rows, columns=['day', 'prev_day', 'implied', 'c1559', 'o1600'])
cmp['bp_1559'] = (cmp.implied / cmp.c1559 - 1) * 1e4
cmp['bp_1600'] = (cmp.implied / cmp.o1600 - 1) * 1e4
print('implied vs 15:59 close (bp): median |.| %.2f p90 %.2f max %.2f' % (
    cmp.bp_1559.abs().median(), cmp.bp_1559.abs().quantile(.9), cmp.bp_1559.abs().max()))
print('implied vs 16:00 open  (bp): median |.| %.2f p90 %.2f max %.2f | exact (<0.01bp) %d/%d' % (
    cmp.bp_1600.abs().median(), cmp.bp_1600.abs().quantile(.9), cmp.bp_1600.abs().max(),
    int((cmp.bp_1600.abs() < 0.01).sum()), len(cmp)))
cmp.to_csv('qqq_prevclose_implied.csv', index=False)
print(cmp.head(8).to_string(index=False))
