"""해석이 갈릴 수 있는 정의를 하나씩 바꿔 엔진 결과와 갈리는지(=그 정의가 실제로 검증됐는지) 본다."""
import json, os
import pandas as pd
import recompute as R

VARIANTS = [
    ('base', {}, 'final'),
    ('D창 121분(ts-120..ts)', dict(dd_win=121), 'final'),
    ('D 최고가에서 이탈 봉 제외', dict(dd_excl_break=True), 'final'),
    ('밀도 창 31분', dict(dens_win=31), 'final'),
    ('밀도 창 29분', dict(dens_win=29), 'final'),
    ('다음 봉 간격 <2분(엄격)', dict(gap_strict=True), 'final'),
    ('다음 봉 간격 <=3분', dict(gap_max=3), 'final'),
    ("S' 같은 날 봉만", dict(s_sameday=True), 'final'),
    ('② 이탈 봉 위생 안 봄', dict(b_sane_k=False), 'final'),
    ('level 사슬 묶음(v2)', dict(chain=True), 'final'),
    ('S 점 추가 안 함', dict(add_s=False), 'final'),
    ('스윙 오른쪽도 엄격(<)', dict(right_strict=True), 'final'),
    ('추세 지속 25/30 이상', dict(persist_min=25), 'final'),
    ('창 29분', dict(win=29), 'final'),
    ('창 31분', dict(win=31), 'final'),
    ('시간 청산 비용=진입 시간대', dict(exit_cost_entry=True), 'final'),
    ('경주 같은 봉=1', dict(same_bar=1.0), 'final'),
    ('② 맥락 같은 날만', dict(ctx_b_sameday=True), 'final'),
    ('N창 119', dict(nwin=119), 'final'),
    ('N창 121', dict(nwin=121), 'final'),
    ('거래대금 = 거래량×VWAP', dict(dollar_vwap=True), 'final'),
    ('① 아랫꼬리 > 0.5(엄격)', dict(wick_strict=True), 'final'),
    ('① 아랫꼬리 >= 0.5-1e-9(부동소수 허용)', dict(wick_eps=1e-9), 'final'),
    ('체결 건수 > 5', dict(trades_min=6), 'final'),
    ('② 장악 > (엄격)', dict(eng_strict=True), 'final'),
    ('겹침 제거를 시간대 거르기 전에', {}, 'pre_tradable'),
    ('겹침 제거를 위생·유효 거르기 전에', {}, 'pre_fixed'),
]

base_keys = {}
rows = []
cache = {}
for name, cfg, stage in VARIANTS:
    key = json.dumps(cfg, sort_keys=True)
    if key not in cache:
        cache[key] = R.candidates(cfg)
    allc = cache[key]
    for arm in ('A', 'B'):
        sig, ctx = R.select(allc, arm, stage)
        for part, d in (('sig', sig), ('ctx', ctx)):
            keys = set(zip(d.sid, d.ts))
            if name == 'base':
                base_keys[(arm, part)] = keys
            bk = base_keys[(arm, part)]
            b = d[d.day >= 20260901]
            st = R.stats(b)
            rows.append(dict(variant=name, arm=arm, part=part, n_all=len(d), only_var=len(keys - bk),
                             only_base=len(bk - keys), n_back=st['n'], hit=round(st['hit_e2'], 4),
                             r1=round(st['r1_e2'] * 1e4, 2), r1net=round(st['r1net_e2'] * 1e4, 2),
                             race=round(st['race_e2'], 4)))
    print(name, 'done', flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(R.OUT, 'sensitivity.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
print(df.to_string(index=False))
