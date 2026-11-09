"""Frozen 2026 cross-stock SMA/EMA 1..120 sweep; canonical desk wealth engine."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE/'cross_stock_grid_1_120_20261109'
SPEC = importlib.util.spec_from_file_location('cross_stock_for_grid', HERE/'cross_stock_review.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
COMPARISON_RTOL = 1e-12
NUMERIC_POLICY = {
    'relative_equality_tolerance': COMPARISON_RTOL,
    'predicate': 'abs(Close-MA)<=1e-12*max(abs(Close),abs(MA))이면 수치상 동등한 값으로 처리',
    'correction': '독립검산에서 ARM/SMA3의2026-02-11 수학적 동등값이1.42e-14USD 차이로 매도 처리된 것을 발견했다. 잠정값을 폐기하고 진입·매도 공통 반올림오차 처리로 전 후보를 다시 계산했다. 후보·선택기준·기간·입력은 유지한다.',
}


def candidate_specs():
    return [('buy_hold', None, 0)] + [(f'exit_{kind}{n}', kind, n)
                                     for kind in ('SMA', 'EMA') for n in range(1, 121)]


def select_front(rows, *, baseline=None, require_mdd=False):
    """Read only front rows; tolerance ties prefer SMA, then shorter length."""
    candidates = [r for r in rows if r['strategy'] != 'buy_hold']
    if require_mdd:
        if baseline is None:
            raise ValueError('Front buy-hold row required for MDD constraint.')
        candidates = [r for r in candidates if r['mdd_pct'] >= baseline['mdd_pct']-1e-9]
    if not candidates:
        raise ValueError('No feasible front candidate.')
    best = max(r['return_pct'] for r in candidates)
    ties = [r for r in candidates if abs(r['return_pct']-best) <= 1e-9]
    return min(ties, key=lambda r: (r['kind'] != 'SMA', int(r['length'])))


def study_groups(universe, tickers):
    groups = C.group_members(universe, tickers)
    groups['additional_external'] = sorted(set(groups['broad_external'])-set(groups['jan2_external']))
    return groups


def register_protocol():
    OUT.mkdir(exist_ok=True)
    target = OUT/'protocol.json'
    if target.exists():
        protocol = json.loads(target.read_text(encoding='utf-8'))
        protocol['numerical_equality'] = NUMERIC_POLICY
        target.write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        return protocol
    old, universe, meta, tickers, book, signals = C.load_inputs()
    protocol = {
        'registered_kst': '2026-11-09', 'request': '동일한 여러 종목에서 SMA·EMA 1~120일, 1일 간격 전수 비교',
        'frozen_input_directory': str(C.OUT.relative_to(C.M.ROOT)/'inputs'),
        'frozen_input_meta_sha256': hashlib.sha256((C.OUT/'inputs/input_meta.json').read_bytes()).hexdigest(),
        'universe': '이전 고정126 연구의 공통138종목 그대로. 이전7종목400설정 편입과 겹치지 않는 연초47을 주 선택군, 확대108 및 추가61을 별도 보고. 현재 AI목록은 사후선택 조건.',
        'grid': {'kinds': ['SMA','EMA'], 'minimum': 1, 'maximum': 120, 'step': 1,
                 'candidates': 240, 'stocks': len(tickers), 'candidate_paths': len(tickers)*240,
                 'control_paths': len(tickers)},
        'periods': old['periods'], 'numerical_equality': NUMERIC_POLICY,
        'primary_selector': 'jan2_external 앞 구간 바스켓 원화수익 최대. 하나의 종류·기간을 모든 종목에 공통 적용. 동률1e-9%p내 SMA 우선, 짧은 기간 우선.',
        'secondary_selector': '같은 앞 구간 최대낙폭이 매수보유보다 나쁘지 않은 후보 중 앞 구간 원화수익 최대. 동률은 주 선택과 동일.',
        'support_selector': 'broad_external 앞 구간 수익만으로 같은 방법 선택. 주 선택군을 포함하므로 독립검증으로 부르지 않음.',
        'execution': '전일 확정 정규장 Close<MA면 다음 정규장 Open에 전량매도. 신규/재편입은 주간 첫 미국세션에만 전일Close>=MA. equality유지. USD현금. 시가갭 먼저 반영. SMA1/EMA1은 매수보유와 동일한 무필터 대조.',
        'ma': 'SMA는 N개 유효 관측. EMA는2025첫 유효Close를초기값으로 adjust=False,ignore_na=True,min_periods=N. 2025는예열만.',
        'wealth': '종목별 동일초기자금, 종목간 자금이전 없음. desk.strategies.run_model과 기존 entry_gate/make_exit_callback 재사용. 종목별평균NAV이며 주간동일비중리밸런싱 아님. 6월초 재시작하지 않음.',
        'fees': '사용자수수료0. 가격수익, 배당현금 미반영. 환전/세금/현금이자 제외. 5/10bp×편도합계회전율 비례차감 민감도도 보고.',
        'exposure_control': '각 기간 실제 평균투자비중으로 매수보유 일별USD수익을 고정노출한 사후대조. 기간시작자금은 실제전략 직전종가NAV 가중. 미래평균비중을 아는 실행규칙으로 해석하지 않음.',
        'report': '모든240개 front/back/full 수익·MDD·투자비중, 선택값과 동종류±5이웃, 뒤/전체 사후최고와 개선개수 별도표시. 성과 보고 다른선택조건 추가하지 않음.',
        'limits': '이미관찰한2026 재탐색. 240개 선택편향, 종목간상관, 2026관측기간의선택, 현재목록/생존/가격공급자조건이 남음. 기존7종목모델 성과가 아님. 연구만, 실매매·Pine기본값 자동변경없음.'}
    target.write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return protocol


def run_batch(batch):
    _, _, _, tickers, book, signals = C.load_inputs()
    days = [d for d in book.days if '2026-01-05' <= d <= C.M.END]
    shape = (len(batch), len(days), len(tickers))
    nav, opens, exposure, turnover = [np.zeros(shape) for _ in range(4)]
    buys, sells, exits = [np.zeros(shape, dtype=np.int8) for _ in range(3)]
    indices = {day: i for i, day in enumerate(days)}
    with C.cached_columns(book):
        for j, (name, kind, n) in enumerate(batch):
            averages = None if kind is None else C.M.moving_average(signals[tickers], kind, n).reindex(book.days)
            for k, ticker in enumerate(tickers):
                path, entries, xx, unknown = C.run_sleeve(book, ticker, 'buy_hold' if kind is None else 'exit',
                    max(n, 1), averages, kind=kind or 'SMA', cache_columns=True, comparison_rtol=COMPARISON_RTOL)
                if unknown:
                    raise AssertionError((ticker, name, unknown))
                nav[j,:,k], opens[j,:,k] = path.close, path.value
                exposure[j,:,k] = [sum(w.values()) for w in path.weights]
                turnover[j,:,k] = [r['one_way_turnover'] for r in C.E.turnover(book, path)]
                for event in path.events:
                    (buys if event['side'] == 'buy' else sells)[j,indices[event['date']],k] += 1
                    if event.get('reason') == 'exit_rule':
                        exits[j,indices[event['date']],k] += 1
            if (j+1) % 20 == 0:
                print(f"{batch[0][0]}: {j+1}/{len(batch)} settings", flush=True)
    cache = OUT/f'cache_{batch[0][0]}.npz'
    np.savez_compressed(cache, nav=nav, opens=opens, exposure=exposure, turnover=turnover,
                        buys=buys, sells=sells, exits=exits)
    return str(cache)


def stat_matrix(curves, lo, hi):
    before = curves[lo-1] if lo else np.ones(curves.shape[1:])
    part = np.concatenate([before[None], curves[lo:hi+1]], axis=0)
    return (part[-1]/part[0]-1)*100, np.min(part/np.maximum.accumulate(part, axis=0)-1, axis=0)*100


def collect(protocol, specs, arrays, tickers, book, universe):
    days = [d for d in book.days if '2026-01-05' <= d <= C.M.END]
    k0 = book.days.index(days[0])
    fx = book.fx[k0:k0+len(days)]/book.fx[k0-1]
    groups = study_groups(universe, tickers)
    positions = {t:i for i,t in enumerate(tickers)}
    group_indices = {g: [positions[t] for t in members] for g,members in groups.items()}
    # Cached arrays are strategy x session x stock. Statistics operate session first.
    nav, opens, exp, turn, buys, sells, exits = [arrays[key].transpose(1,2,0) for key in
                                             ('nav','opens','exposure','turnover','buys','sells','exits')]
    costs = {bp: nav*np.cumprod(1-turn*bp/10000., axis=0) for bp in (5,10)}
    per_stock, aggregate, baskets = [], [], {}
    for label, (begin, end) in protocol['periods'].items():
        lo, hi = C.interval(days, begin, end)
        ur, ud = stat_matrix(nav, lo, hi)
        kr, kd = stat_matrix(nav*fx[:,None,None], lo, hi)
        cr = {bp: stat_matrix(costs[bp]*fx[:,None,None], lo, hi)[0] for bp in costs}
        averages = exp[lo:hi+1].mean(axis=0)
        # Each stock's buyhold daily return at its strategy's ex-post mean exposure.
        base = np.concatenate([nav[lo-1,:,0][None] if lo else np.ones((1,len(tickers))), nav[lo:hi+1,:,0]])
        base_ret = base[1:]/base[:-1]-1
        matched = (np.prod(1+base_ret[:,:,None]*averages[None], axis=0)-1)*100
        sums = {key: value[lo:hi+1].sum(axis=0) for key,value in
                [('buys',buys),('sells',sells),('ma_exits',exits),('one_way_turnover',turn)]}
        for j, (name, kind, n) in enumerate(specs):
            for k, ticker in enumerate(tickers):
                per_stock.append({'ticker': ticker, 'strategy': name, 'kind': kind or 'control', 'length': n,
                    'period': label, 'return_pct': kr[k,j], 'mdd_pct': kd[k,j],
                    'usd_return_pct': ur[k,j], 'usd_mdd_pct': ud[k,j],
                    'mean_open_exposure_pct': averages[k,j]*100,
                    **{key: float(value[k,j]) for key,value in sums.items()},
                    **{f'return_{bp}bp_pct': cr[bp][k,j] for bp in costs},
                    'usd_matched_exposure_return_pct': matched[k,j]})
            for group, idx in group_indices.items():
                basket = nav[:,idx,j].mean(axis=1)
                if label == 'full':
                    baskets[group+'::'+name] = basket*fx
                u = C.curve_stats(basket, days, begin, end)
                r = C.curve_stats(basket*fx, days, begin, end)
                fractions = (opens[:,idx,j]*exp[:,idx,j]).sum(axis=1)/opens[:,idx,j].sum(axis=1)
                initial = nav[lo-1,idx,j] if lo else np.ones(len(idx))
                control = float(np.average(matched[idx,j], weights=initial))
                aggregate.append({'group': group, 'strategy': name, 'kind': kind or 'control', 'length': n,
                    'period': label, 'n': len(idx), **r, 'usd_return_pct': u['return_pct'],
                    'usd_mdd_pct': u['mdd_pct'], 'mean_open_exposure_pct': fractions[lo:hi+1].mean()*100,
                    **{f'return_{bp}bp_pct': C.curve_stats(costs[bp][:,idx,j].mean(axis=1)*fx,
                                days, begin, end)['return_pct'] for bp in costs},
                    'usd_matched_exposure_return_pct': control,
                    'usd_vs_matched_exposure_return_pp': u['return_pct']-control,
                    'return_better_count': int(np.sum(kr[idx,j] > kr[idx,0]+1e-9)),
                    'mdd_better_count': int(np.sum(kd[idx,j] > kd[idx,0]+1e-9)),
                    'both_better_count': int(np.sum((kr[idx,j] > kr[idx,0]+1e-9)&(kd[idx,j] > kd[idx,0]+1e-9)))})
    ps, ag = pd.DataFrame(per_stock), pd.DataFrame(aggregate)
    ps.loc[ps.strategy == 'buy_hold'].to_csv(OUT/'per_stock_baseline.csv', index=False)
    for kind in ('SMA','EMA'):
        for label in protocol['periods']:
            ps.loc[(ps.kind == kind)&(ps.period == label)].to_csv(OUT/f'per_stock_{kind}_{label}.csv', index=False)
        selected_curves = {k:v for k,v in baskets.items() if '::exit_'+kind in k or k.endswith('::buy_hold')}
        pd.DataFrame(selected_curves, index=days).to_csv(OUT/f'basket_curves_{kind}.csv', index_label='Date')
    ag.to_csv(OUT/'aggregate.csv', index=False)
    primary = ag.loc[(ag.group == 'jan2_external')&(ag.period == 'front')].to_dict('records')
    base = next(r for r in primary if r['strategy'] == 'buy_hold')
    broad = ag.loc[(ag.group == 'broad_external')&(ag.period == 'front')].to_dict('records')
    selected = {'primary': select_front(primary)['strategy'],
                'secondary': select_front(primary, baseline=base, require_mdd=True)['strategy'],
                'broad': select_front(broad)['strategy']}
    (OUT/'selectors.json').write_text(json.dumps(selected, indent=2)+'\n', encoding='utf-8')
    diagnostics = {}
    for group in groups:
        diagnostics[group] = {}
        for label in ('back','full'):
            part = ag.loc[(ag.group == group)&(ag.period == label)]
            baseline = part.loc[part.strategy == 'buy_hold'].iloc[0]
            rules = part.loc[part.strategy != 'buy_hold']
            best = select_front(rules.to_dict('records'))
            diagnostics[group][label] = {'hindsight_best': best['strategy'],
                'return_better': int((rules.return_pct > baseline.return_pct+1e-9).sum()),
                'mdd_better': int((rules.mdd_pct > baseline.mdd_pct+1e-9).sum()),
                'both_better': int(((rules.return_pct > baseline.return_pct+1e-9)&(rules.mdd_pct > baseline.mdd_pct+1e-9)).sum())}
    neighbor_names = set()
    for name in selected.values():
        _, kind, n = next(s for s in specs if s[0] == name)
        neighbor_names.update(f'exit_{kind}{m}' for m in range(max(1,n-5),min(120,n+5)+1))
    ag.loc[ag.strategy.isin(neighbor_names)].to_csv(OUT/'neighbors.csv', index=False)
    result = {'stocks':len(tickers), 'candidate_count':240, 'path_count':len(tickers)*len(specs),
        'groups':groups, 'periods':protocol['periods'], 'selectors':selected, 'descriptive':diagnostics,
        'input_sha256': {str(p.relative_to(C.M.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((C.OUT/'inputs').iterdir()) if p.is_file()},
        'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
        'comparison_source_sha256': comparison_sources()}
    (OUT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    write_report(result, ag)
    plot(ag, baskets, days, selected)
    return result


def comparison_sources():
    return {str(p.relative_to(C.M.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (C.OUT/'aggregate.csv', HERE/'exit_1_200/summary.csv')}


def write_report(result, ag):
    old = pd.read_csv(C.OUT/'aggregate.csv')
    fixed = old.loc[(old.group == 'broad_external')&(old.strategy == 'exit_SMA126')].set_index('period')
    selected = result['selectors']['primary']
    main = ag.loc[(ag.group == 'broad_external')&(ag.strategy == selected)].set_index('period')
    best = result['descriptive']['broad_external']['full']['hindsight_best']
    lines = ['# SMA·EMA 1~120일: 여러 종목2026 재탐색', '',
             f"**이번 범위에서는 단기3~6일선이 고정126일선보다 나은 방어 후보였다.** 앞 구간만으로 고른 {selected}는 확대외부108종목에서 "
             f"전체 수익 {main.at['full','return_pct']:+.2f}%, 뒤 수익 {main.at['back','return_pct']:+.2f}%, 뒤 낙폭 {main.at['back','mdd_pct']:.2f}%였다. "
             f"고정126의 같은 값은 {fixed.at['full','krw_return_pct']:+.2f}%, {fixed.at['back','krw_return_pct']:+.2f}%, {fixed.at['back','krw_mdd_pct']:.2f}%다. "
             f"전체를 본 사후1위는 {best}이며 앞 구간 선택값과 구분한다.", '',
             '6일이라는 정확한 값의 우위를 확정한 것은 아니다. 연초외부47에서 EMA2·3·6의 앞 수익은20.69~20.82%로 비슷했고 매수보유20.48%와도 가깝다. '
             '확대외부108의 앞 구간 수익으로 고르면 무필터인SMA1이 선택된다. 종목군에 따라 상승장에서 짧은 이평선의 기회비용이 커진다. '
             '다만47에서 고른EMA6를 추가61에 그대로 적용했을 때도 뒤 손실·낙폭과 전체 수익이 개선됐다.', '',
             '138종목 × 240설정 = 33,120후보경로 + 매수보유138경로. 기존과 같은 동결입력, 동일초기자금 종목별독립계좌. '
             '전일종가 신호→다음 정규장시가 전량매도, 주간 첫 미국세션만 재진입. 수수료0, 배당현금 미반영 가격수익. '
             '아래 표는 원화환산이며 앞1/5~5/29, 뒤6/1~11/4, 전체1/5~11/4. 2025는 예열만.', '',
             f"앞 구간만으로 선택: 주 선택 **{result['selectors']['primary']}**, 낙폭제약 보조 **{result['selectors']['secondary']}**, 확대군 선택 **{result['selectors']['broad']}**.", '',
             '|집단|설정|선택 구분|앞 수익|뒤 수익|뒤 최대낙폭|전체 수익|뒤 평균투자비중|뒤10bp수익|',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for group in ('jan2_external','additional_external','broad_external','broad_all'):
        hindsight = result['descriptive'][group]['full']['hindsight_best']
        names = list(dict.fromkeys(['buy_hold',*result['selectors'].values(), hindsight,'exit_SMA120','exit_EMA120']))
        part = ag.loc[ag.group == group].set_index(['strategy','period'])
        for name in names:
            f,b,a = [part.loc[(name,p)] for p in ('front','back','full')]
            role = ('매수보유 대조' if name == 'buy_hold' else '연초47 앞선택' if name == selected
                    else '확대108 앞선택·무필터' if name == result['selectors']['broad']
                    else '각 집단 전체 사후1위' if name == hindsight else '장기선 대조')
            lines.append(f'|{group}|{name}|{role}|{f.return_pct:+.2f}%|{b.return_pct:+.2f}%|{b.mdd_pct:.2f}%|{a.return_pct:+.2f}%|{b.mean_open_exposure_pct:.1f}%|{b.return_10bp_pct:+.2f}%|')
    lines += ['', '주 표의 각 집단에서 주·보조·확대 선택 외에 추가한 최고값은 각 집단의 전체 구간 사후최고다. 전체/뒤 구간 최고값은 주 선택에 사용하지 않았다. '
              '전체240후보의 수익·낙폭 동시개선 개수는 results.json에, 전 기간·집단·종목 결과는 CSV에 모두 저장했다.', '',
              '외부108의 뒤 구간에서는238개 비자명 후보 모두 매수보유보다 바스켓 수익·낙폭이 개선됐다(나머지2개는1일 무필터 대조). '
              '종목별로는 EMA6가81/108종목에서 뒤 수익·낙폭을 함께 개선했고, 전체 구간 동시개선은71/108이다. 바스켓 개선과 모든 개별종목 개선을 구분한다.', '',
              '## 기존7종목 모델과의 구분', '',
              '**여러 종목의 단기 방어 결과를 우리7종목의 전량매도 규칙으로 그대로 옮기면 안 된다.** 아래는 이미검산한 이전7종목 연구의 기록이다. '
              '이번138종목 실험과 달리 실제 순위·상한·주간교체를 유지하며1/2~11/4를 평가했다.', '',
              '|기존7종목 설정|앞 수익|뒤 수익|뒤 낙폭|전체 수익|', '|---|---:|---:|---:|---:|']
    seven = pd.read_csv(HERE/'exit_1_200/summary.csv').set_index('name')
    for name in ('baseline','EMA6','EMA90','SMA126'):
        r = seven.loc[name]
        lines.append(f'|{name}|{r.front_return_pct:+.2f}%|{r.back_return_pct:+.2f}%|{r.back_mdd_pct:.2f}%|{r.full_return_pct:+.2f}%|')
    restricted = seven.loc[(seven.length >= 2)&(seven.length <= 120)]
    count = int((restricted.full_return_pct > seven.loc['baseline','full_return_pct']+1e-9).sum())
    lines += ['', f'기존7종목의2~120일238개 중 전체 수익이 현행을 넘은 것은{count}개였다. 1일선2개는 현행과 같다. '
              '이전400격자에서 범위를1~120으로 제한한 기록을 확인한 것이며 이번138종목 결과로 재선택한 모델이 아니다. '
              'EMA6는 상승 구간 수익을 크게 줄여 전체 결과가 나빠졌다. 따라서 EMA6는 개별종목 방어지표의 연구 후보이며 기존7종목 모델의 개선안으로 채택할 근거는 없다.', '',
              '## 사후 동일투자비중 USD대조', '',
              '|집단|설정|구간|실제USD수익|같은 평균비중 대조|차이|', '|---|---|---|---:|---:|---:|']
    for group in ('jan2_external','additional_external','broad_external'):
        for name in dict.fromkeys(result['selectors'].values()):
            for label in ('front','back','full'):
                r = ag.loc[(ag.group == group)&(ag.strategy == name)&(ag.period == label)].iloc[0]
                lines.append(f'|{group}|{name}|{label}|{r.usd_return_pct:+.2f}%|{r.usd_matched_exposure_return_pct:+.2f}%|{r.usd_vs_matched_exposure_return_pp:+.2f}%p|')
    lines += ['', '## 조건·해석 한계', '',
              '- 연초외부47은 과거7종목400설정 편입과 겹치지 않는 TIME·KoAct 연초보유군. 추가외부61은 그47을 제외한 확대군이다. 확대108은47을 포함하며 독립검증군이 아니다. 전체138에는 과거발견중복30이 포함된다.',
              '- 1일선은 전일Close와 동일하므로 이탈이 없고 매수보유 대조와 같다. 240개에는 이 무필터 대조2개도 포함된다.',
              '- 독립검산에서 수학적으로 같은 평균이1.42e-14USD 차이로 매도되던 ARM/SMA3 사례를 발견해 잠정값을 폐기하고 전 후보를 다시 계산했다. 상대차이1e-12 이하는 동등값으로 보아 진입·매도에 공통 적용한다. 실제가격 이탈과 수치상 반올림오차를 구분하는 회귀검증을 추가했으며 후보·기간·선택기준은 유지했다.',
              '- 현재 AI목록·생존·완전한2026가격 조건에 따른 선택편향과 종목간 시장상관이 남는다. 이미관찰한2026을 추가탐색하므로 새로운 독립표본으로 부르지 않는다. 개별종목별 최적기간을 고르지 않았다.',
              '- 동일평균비중 대조의 비중은 기간이 끝나야 아는 사후값이다. 현금축소 효과를 살피는 설명용이며 실행규칙이 아니다. 앞수익·뒤수익·최대낙폭·비중을 함께 판단한다.',
              '- 5/10bp는 전체편도회전율에 대한 비례차감 민감도이며 실제슬리피지 추정이 아니다. 환전·세금·현금이자 제외. 시가갭은 먼저 부담한다.',
              '- 기존7종목 순위·배분을 제거한 조건이식 실험이다. 기존7종목 모델 성과로 부르지 않으며 운영·주문·Pine기본값은 바꾸지 않는다.', '',
              '![1~120일 비교](grid.png)', '',
              '재현: `py data/derived/analysis/ma_entry_filter_20261108/cross_stock_grid.py` 후 `py data/derived/analysis/ma_entry_filter_20261108/audit_grid_cross_stock.py`. 입력은 이전연구의 동결파일을 재사용한다. 캐시npz는Git제외, 생성CSV는 종류·평가구간별10MB미만으로 나눈다.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def plot(ag, baskets, days, selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, axes = plt.subplots(2,2, figsize=(13,8))
    for col,group in enumerate(('jan2_external','broad_external')):
        for kind,color in [('SMA','#148b8b'),('EMA','#b77329')]:
            part = ag.loc[(ag.group == group)&(ag.kind == kind)&(ag.period == 'full')].sort_values('length')
            axes[0,col].plot(part.length, part.return_pct, label=kind,color=color)
        baseline = ag.loc[(ag.group == group)&(ag.strategy == 'buy_hold')&(ag.period == 'full')].iloc[0]
        axes[0,col].axhline(baseline.return_pct,color='#888',ls=':', label='Buy and hold')
        axes[0,col].set_title(f'{group}: full-period return (descriptive)')
        axes[0,col].set_xlabel('MA length, trading sessions')
        axes[0,col].set_ylabel('KRW return %')
        for name in dict.fromkeys(['buy_hold',selected['primary'],selected['broad']]):
            axes[1,col].plot(pd.to_datetime(days), baskets[group+'::'+name]*100,label=name)
        axes[1,col].axvline(pd.Timestamp('2026-06-01'),color='#888',ls=':')
        axes[1,col].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axes[1,col].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        axes[1,col].set_ylabel('KRW wealth, initial=100')
        for row in range(2):
            axes[row,col].grid(alpha=.2)
            axes[row,col].legend(fontsize=8)
    fig.suptitle('SMA / EMA 1..120 | daily next-open exits, weekly reentry | frozen 2026')
    fig.tight_layout()
    fig.savefig(OUT/'grid.png',dpi=150)
    plt.close(fig)


def main():
    start = time.perf_counter()
    protocol = register_protocol()
    specs = candidate_specs()
    batches = [specs[i:i+81] for i in range(0,len(specs),81)]
    paths = [OUT/f'cache_{batch[0][0]}.npz' for batch in batches]
    # Always rebuild: local ignored caches must never silently bypass changed inputs/code.
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(run_batch,batches))
    arrays = {}
    for path in paths:
        with np.load(path) as data:
            for key in data.files:
                arrays.setdefault(key,[]).append(data[key])
    arrays = {key:np.concatenate(value) for key,value in arrays.items()}
    _,universe,_,tickers,book,_ = C.load_inputs()
    result = collect(protocol,specs,arrays,tickers,book,universe)
    print(json.dumps({'selectors':result['selectors'],'elapsed_seconds':time.perf_counter()-start}))


if __name__ == '__main__':
    main()
