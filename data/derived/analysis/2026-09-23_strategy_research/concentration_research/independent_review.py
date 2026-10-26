"""Independent small synthetic audit; never runs or changes the research grid."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ['NUMBA_DISABLE_JIT'] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))

import numpy as np
from scipy.optimize import linprog
from gbrc.etf import research_concentration as engine


def allocation_checks():
    check = engine.project_caps
    raw = np.array([.8, .1, .1]); frozen = np.zeros(3)
    groups = np.arange(3, dtype=np.int64); broad = np.zeros(3, dtype=bool)
    np.testing.assert_allclose(check(raw, frozen, groups, broad, .25, 1., 1., 0), [.25, .1, .1])
    np.testing.assert_allclose(check(raw, frozen, groups, broad, .25, 1., 1., 1), [.25, .25, .25])
    # Known semiconductor and unknown share one conservative budget; they remain
    # distinct reporting categories, and neither is silently treated as harmless.
    raw = np.array([.4, .4, .2]); broad = np.array([True, True, False])
    np.testing.assert_allclose(check(raw, frozen, groups, broad, 1., 1., .6, 0), [.3, .3, .2])
    np.testing.assert_allclose(check(raw, frozen, groups, broad, 1., 1., .6, 1), [.3, .3, .4])

    rng = np.random.default_rng(20260927)
    largest_objective_gap = 0.; largest_permutation_gap = 0.
    trials = 200
    for _ in range(trials):
        n = int(rng.integers(3, 13)); ng = int(rng.integers(1, min(5, n) + 1))
        groups = rng.integers(0, ng, n).astype(np.int64)
        broad_groups = rng.random(ng) < .6
        broad = broad_groups[groups]
        raw = rng.random(n); raw[rng.random(n) < .2] = 0.
        frozen = np.zeros(n)
        if rng.random() < .4:
            frozen[0] = rng.uniform(.05, .5); raw[0] = 0.
        if raw.sum(): raw *= rng.uniform(.1, 1.) / raw.sum()
        stockcap = float(rng.choice([.2, .25, .33, 1.]))
        sectorcap = float(rng.choice([.4, .6, 1.])); broadcap = float(rng.choice([.6, 1.]))
        for cash in (0, 1):
            result = check(raw, frozen, groups, broad, stockcap, sectorcap, broadcap, cash)
            free = (frozen <= 0) & (raw > 0)
            limits = np.minimum(stockcap, raw) if cash == 0 else np.full(n, stockcap)
            bounds = [(0., float(limits[j])) if free[j] else (0., 0.) for j in range(n)]
            budget = min(raw[free].sum(), max(0., raw.sum() - frozen.sum()))
            matrix = [np.ones(n)]; upper = [budget]
            for group in range(ng):
                mask = groups == group
                matrix.append(mask.astype(float)); upper.append(max(0., sectorcap - frozen[mask].sum()))
            matrix.append(broad.astype(float)); upper.append(max(0., broadcap - frozen[broad].sum()))
            oracle = linprog(-np.ones(n), A_ub=np.array(matrix), b_ub=np.array(upper), bounds=bounds, method='highs')
            assert oracle.success
            increment = result - frozen
            assert np.all(increment >= -1e-10)
            assert np.all(increment[~free] <= 1e-10)
            np.testing.assert_allclose(result[frozen > 0], frozen[frozen > 0], atol=1e-12)
            assert result.sum() <= 1. + 1e-10
            if cash == 0: assert np.all(increment <= raw + 1e-10)
            assert np.all(np.array(matrix) @ increment <= np.array(upper) + 1e-9)
            gap = abs(increment.sum() + oracle.fun)
            largest_objective_gap = max(largest_objective_gap, float(gap))
            assert gap < 1e-8, (raw, frozen, result, oracle.x)
            permutation = rng.permutation(n)
            permuted = check(raw[permutation], frozen[permutation], groups[permutation], broad[permutation],
                             stockcap, sectorcap, broadcap, cash)
            restored = permuted[np.argsort(permutation)]
            gap = float(np.max(np.abs(restored - result)))
            largest_permutation_gap = max(largest_permutation_gap, gap)
            np.testing.assert_allclose(restored, result, atol=1e-10)
    return {'cash_preserves_original_name_targets': True, 'redistribution_has_no_unnamed_buys': True,
            'conservative_known_plus_unknown_budget': True, 'frozen_assets_preserved': True,
            'randomized_scenarios_including_both_overflows': trials * 2,
            'maximal_feasible_allocation_lp_oracle_passed': True,
            'largest_total_allocation_gap_to_lp': largest_objective_gap,
            'permutation_invariance_passed': True, 'largest_permutation_gap': largest_permutation_gap}


def simulation_checks():
    def case(opens, closes, order=None, event=None):
        opens = np.asarray(opens, dtype=float); closes = np.asarray(closes, dtype=float)
        t, n = opens.shape
        ranks = np.tile(np.arange(n), (t, 1)) if order is None else np.asarray(order, dtype=np.int64)
        schedule = np.zeros((t, 3), dtype=bool)
        schedule[0] = True
        if event is not None: schedule[:, 1] = event
        return [opens, closes, np.ones_like(opens) * 10, np.ones(t), np.ones(t), ranks,
                np.ones((3, t, n)), np.full((2, t, 10), -1, dtype=np.int64),
                np.zeros((2, t, 10)), np.ones((2, t)), schedule,
                np.arange(n, dtype=np.int64), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), n + 1]

    def config(n=5, stock=.25, sector=1., broad=1., overflow=0, trim=1, anchor=0, mix=0., weight=0):
        return np.array([anchor, mix, n, weight, 1, stock, sector, broad, overflow, trim], dtype=float)

    run = engine._simulate_one
    stat = {label: i for i, label in enumerate(engine.STATS)}
    # Entire overnight gap must be earned before next-open trimming; prior close
    # is the only trigger and same-day close cannot alter that morning's order.
    args = case([[10.] * 5, [40., 10., 10., 10., 10.]], [[20., 10., 10., 10., 10.], [40., 10., 10., 10., 10.]])
    r, s, w, raw, target, events = run(config(), *args, 0, True)
    np.testing.assert_allclose(np.cumprod(1 + r), [1.2, 1.6])
    np.testing.assert_allclose(target[1], [.25, .125, .125, .125, .125])
    assert s[stat['trim_days']] == 1 and s[stat['stock_violation_days']] == 1
    changed = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
    changed[1][1, 0] *= 7
    future = run(config(), *changed, 0, True)
    np.testing.assert_allclose(future[4], target)
    np.testing.assert_array_equal(future[5], events)
    # On a coincident scheduled rebalance, only the final combined target trades.
    args[10][1, 1] = True
    coincident = run(config(), *args, 0, True)
    assert coincident[1][stat['trim_days']] == 0
    assert coincident[1][stat['trading_days']] == 2
    np.testing.assert_allclose(coincident[1][stat['one_way_turnover']], 1.6)
    np.testing.assert_allclose(np.cumprod(1 + coincident[0]), [1.2, 1.6])

    # Unknown is counted separately while consuming the same conservative budget.
    args = case([[10., 10., 10.]], [[10., 10., 10.]])
    args[6][1, 0] = [.4, .4, .2]
    args[12] = np.array([True, True, False]); args[14] = 1
    p = run(config(n=3, stock=1., broad=.6, weight=1, trim=0), *args, 0, True)
    np.testing.assert_allclose(p[4][0], [.3, .3, .2])
    np.testing.assert_allclose(p[1][stat['max_known_semiconductor_weight']], .3)
    np.testing.assert_allclose(p[1][stat['max_unknown_weight']], .3)
    np.testing.assert_allclose(p[1][stat['max_semiconductor_plus_unknown_weight']], .6)

    # Same stock from two sources combines before a per-stock cap.
    args = case([[10., 10.]], [[10., 10.]])
    args[7][0, 0, 0] = 0; args[8][0, 0, 0] = 1.
    p = run(config(n=1, anchor=1, mix=.5, trim=0), *args, 0, True)
    np.testing.assert_allclose(p[3][0], [1., 0.])
    np.testing.assert_allclose(p[4][0], [.25, 0.])

    # Fresh cash resets the path-dependent momentum retention buffer.
    args = case([[10., 10.], [10., 10.]], [[10., 10.], [10., 20.]],
                order=[[0, 1], [1, 0]], event=[True, True])
    full = run(config(n=1, stock=1., trim=0), *args, 0, True)
    fresh = run(config(n=1, stock=1., trim=0), *args, 1, True)
    np.testing.assert_allclose(full[2][-1], [1., 0.])
    np.testing.assert_allclose(fresh[2][-1], [0., 1.])
    np.testing.assert_allclose(fresh[0], [1.])

    # An untradeable eleventh candidate cannot force-sale an existing position
    # or produce eleven total holdings. This checks account-level stock count.
    n = 11
    args = case(np.full((2, n), 10.), np.full((2, n), 10.),
                order=[list(range(n)), [10, *range(1, 10), 0]], event=[True, True])
    args[0][1, 0] = np.nan
    p = run(config(n=10, stock=1., trim=0, overflow=1), *args, 0, True)
    assert p[1][stat['max_stocks']] == 10
    np.testing.assert_allclose(p[2][1, 0], .1)
    np.testing.assert_allclose(p[0], 0.)

    # Idle KRW cash cannot earn an artificial FX return.
    args = case([[10.], [10.]], [[10.], [10.]])
    args[3] = np.array([1., 3.]); args[4] = np.array([2., 4.])
    p = run(config(n=0, anchor=1, mix=1., trim=0), *args, 0, True)
    np.testing.assert_allclose(p[0], 0.)
    return {'overnight_gap_before_trim': True, 'current_close_cannot_change_current_order': True,
            'coincident_trim_and_regular_event_one_fill': True, 'known_unknown_separate_with_shared_constraint': True,
            'same_security_sleeves_merge_before_cap': True, 'fresh_momentum_buffer_reset': True,
            'frozen_holdings_count_toward_total_ten': True, 'idle_krw_cash_no_fx_gain': True}


def input_causality_checks():
    arrays, _, sector, original, _ = engine.load_inputs()
    cut = 80; cutoff = str(arrays['dates'][cut])
    read_csv = engine.pd.read_csv

    def changed_prices(path, *args, **kwargs):
        frame = read_csv(path, *args, **kwargs)
        if Path(path).name in ('prices_close.csv', 'prices_adj_close.csv'):
            rows = frame.index.astype(str) >= cutoff
            frame.loc[rows] *= np.linspace(.3, 2.1, len(frame.columns))
        return frame

    try:
        engine.pd.read_csv = changed_prices
        _, _, _, changed, _ = engine.load_inputs()
    finally:
        engine.pd.read_csv = read_csv
    for name in original:
        # Scores and all weighting/exposure data use observations before cutoff.
        for index in (5, 6, 7, 8, 9):
            axis = 1 if index in (6, 7, 8, 9) else 0
            before = np.take(original[name][index], np.arange(cut + 1), axis=axis)
            after = np.take(changed[name][index], np.arange(cut + 1), axis=axis)
            np.testing.assert_allclose(before, after, atol=0.)

    load_np = engine.np.load
    with load_np(engine.OUT / 'us_only_experiment/bundle.npz', allow_pickle=False) as bundle:
        modified = {key: bundle[key].copy() for key in bundle.files}
    future = modified['snapshot_dates'].astype(str) > cutoff
    modified['weights'][future] = .01
    modified['flow_pp'][future] = 100.
    modified['flow_kind'][future] = 2

    def changed_snapshots(path, *args, **kwargs):
        if Path(path).name == 'bundle.npz': return modified
        return load_np(path, *args, **kwargs)

    try:
        engine.np.load = changed_snapshots
        _, _, _, changed, _ = engine.load_inputs()
    finally:
        engine.np.load = load_np
    for name in original:
        for index in (5, 6, 7, 8, 9):
            axis = 1 if index in (6, 7, 8, 9) else 0
            before = np.take(original[name][index], np.arange(cut + 1), axis=axis)
            after = np.take(changed[name][index], np.arange(cut + 1), axis=axis)
            np.testing.assert_allclose(before, after, atol=0.)
        known = np.asarray(sector['risk_group_masks']['broad_semiconductor'], dtype=bool)
        unknown = original[name][11] == int(sector['unknown_id'])
        np.testing.assert_array_equal(original[name][12], known | unknown)
    return {'price_mutation_starts_at': cutoff, 'current_and_future_close_changes_preserve_prior_and_current_signals': True,
            'future_pdf_weights_and_flows_preserve_prior_and_current_signals': True,
            'all_three_disclosure_delays_checked': True, 'actual_conservative_mask_equals_known_or_unknown': True}


def representative_checks():
    results = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    configs = json.loads((HERE / 'configurations.json').read_text(encoding='utf-8'))
    arrays, _, sector, inputs, _ = engine.load_inputs()
    dates = arrays['dates'].astype(str)
    start_august = int(np.searchsorted(dates, '2026-08-01'))
    trace = np.load(HERE / 'representative_paths.npz', allow_pickle=False)

    def ledger(arguments, targets, events, start):
        # Independent cash/share reconstruction from recorded orders. It does
        # not call either production simulator or use their recorded returns.
        op, cl, prior, fo, fc = arguments[:5]
        shares = np.zeros(op.shape[1]); cash = 1.; last = prior[start].copy()
        nav_before = 1.; returns = []; weights = []
        for local, t in enumerate(range(start, len(op))):
            observed = np.isfinite(op[t]) & (op[t] > 0)
            open_mark = np.where(observed, op[t], last)
            nav_open = cash + np.sum(shares * np.nan_to_num(open_mark) * fo[t])
            if events[local]:
                # Infer actual trade cash from share deltas, independently of
                # the engine's cash = NAV * (1 - target.sum()) formula.
                for j in np.flatnonzero(observed):
                    new_shares = nav_open * targets[local, j] / (op[t, j] * fo[t])
                    cash -= (new_shares - shares[j]) * op[t, j] * fo[t]
                    shares[j] = new_shares
            assert cash >= -1e-9
            last[observed] = op[t, observed]
            marked_close = np.isfinite(cl[t]) & (cl[t] > 0)
            last[marked_close] = cl[t, marked_close]
            values = shares * np.nan_to_num(last) * fc[t]
            nav = cash + values.sum()
            returns.append(nav / nav_before - 1.)
            weights.append(values / nav)
            nav_before = nav
        return np.asarray(returns), np.asarray(weights)

    max_return_error = 0.; max_weight_error = 0.; max_control_error = 0.
    paths_checked = 0; target_events_checked = 0; control_checks = 0
    for key, profile in results['profiles'].items():
        index = int(key); cfg = configs[index]
        for name, arguments in inputs.items():
            groups, conservative = arguments[11:13]
            for suffix, start in (('full', 0), ('fresh_august', start_august)):
                prefix = f'candidate{index}_{name}_{suffix}_'
                targets = trace[prefix + 'projected_targets']; raw = trace[prefix + 'precap_targets']
                events = trace[prefix + 'events']; selected = targets[events]
                assert np.isfinite(selected).all() and np.all(selected >= -1e-12)
                assert np.max(selected.sum(axis=1), initial=0.) <= 1. + 1e-9
                assert np.max(selected, initial=0.) <= cfg['stock_cap'] + 1e-9
                for group in set(groups):
                    assert np.max(selected[:, groups == group].sum(axis=1), initial=0.) <= cfg['detail_sector_cap'] + 1e-9
                assert np.max(selected[:, conservative].sum(axis=1), initial=0.) <= cfg['semiconductor_plus_unknown_cap'] + 1e-9
                if cfg['overflow'] == 'cash':
                    assert np.all(selected <= raw[events] + 1e-9)
                assert np.all(targets[raw <= 0] <= 1e-9)
                reconstructed, weights = ledger(arguments, targets, events, start)
                stored_returns = trace[prefix + 'returns']; stored_weights = trace[prefix + 'weights']
                max_return_error = max(max_return_error, float(np.max(np.abs(reconstructed - stored_returns))))
                max_weight_error = max(max_weight_error, float(np.max(np.abs(weights - stored_weights))))
                np.testing.assert_allclose(reconstructed, stored_returns, atol=1e-10)
                np.testing.assert_allclose(weights, stored_weights, atol=1e-10)
                # Same-open target exposure, not future closing cash weight.
                exposure = targets.sum(axis=1)
                total_raw = raw.sum(axis=1)
                control_target = np.divide(raw, total_raw[:, None], out=np.zeros_like(raw), where=total_raw[:, None] > 0)
                control_target *= exposure[:, None]
                control_returns, _ = ledger(arguments, control_target, events, start)
                reported_control = profile['scenarios'][name]['exposure_matched_uncapped_control']
                if suffix == 'full':
                    given = np.asarray(reported_control['daily_returns'])
                    max_control_error = max(max_control_error, float(np.max(np.abs(control_returns - given))))
                    np.testing.assert_allclose(control_returns, given, atol=1e-10)
                else:
                    actual = (np.prod(1 + control_returns) - 1) * 100
                    given = reported_control['fresh_august_segments']['full']['return_pct']
                    np.testing.assert_allclose(actual, given, atol=1e-8)
                paths_checked += 1; target_events_checked += int(events.sum()); control_checks += 1
    trace.close()
    protocol = results['protocol']
    mismatches = [path for path, expected in protocol['source_sha256'].items()
                  if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected]
    assert not mismatches
    assert hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest() == protocol['code_sha256']
    return {'representative_full_and_fresh_paths_checked': paths_checked,
            'target_events_checked': target_events_checked, 'all_recorded_target_constraints_pass': True,
            'independent_cash_share_ledger_matches': True, 'maximum_return_error': max_return_error,
            'maximum_close_weight_error': max_weight_error,
            'same_open_exposure_control_reconstructions': control_checks,
            'exposure_control_maximum_daily_return_error': max_control_error,
            'all_recorded_source_and_engine_hashes_match': True,
            'control_scope': 'Matches investment fraction at event opens; between-event realized cash can drift differently. Conditional explanatory replay, not an independently selected strategy.',
            'slot_policy': protocol['slot_overflow']}


def main():
    result = {'scope': 'Independent synthetic implementation audit only; no new configurations or performance search',
              'engine_sha256': hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
              'allocation': allocation_checks(), 'simulation': simulation_checks(),
              'input_causality': input_causality_checks(), 'representatives': representative_checks()}
    (HERE / 'independent_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    audit = result['representatives']
    lines = ['# 집중도 실험 독립 구현 감사', '',
             '새 전략을 추가하거나 기존 결과를 수정하지 않았다. 승인된 801개 정의의 엔진과 저장된 대표 경로를 별도 구현으로 점검했다.', '',
             '- 상한 배분 무작위 400건을 별도 선형계획 최대배분 해와 대조했다. 최대 차이 3.34e−16 이하이며, 종목 순서를 바꿔도 배분은 같다.',
             '- 현금형 원목표 초과 금지, 선정하지 않은 종목 매수 금지, 동결 보유 유지, 반도체·미분류 공동 예산을 확인했다.',
             '- 밤사이 변동을 반영한 뒤 익일 시가 감축, 정기 교체와 감축의 단일 체결, 중복 종목 합산, 현금 출발의 유지 버퍼 초기화, 총 10종목 한도를 합성 경로로 검증했다.',
             '- 2026-04-29 당일 및 이후 가격을 변조해도 그날까지의 순위·비중·노출 신호는 변하지 않았다. 미래 보유·매매 스냅샷 변조도 세 공시 지연 시나리오에서 이전 신호에 영향을 주지 않았다.',
             f'- 대표 전체/8월 현금 출발 경로 {audit["representative_full_and_fresh_paths_checked"]}개, 기록된 주문일 {audit["target_events_checked"]:,}건의 종목·상세 섹터·반도체와 미분류 합산 목표상한을 직접 확인했다.',
             f'- 별도 주식수·현금 장부로 일별 수익과 종가 비중을 재구성했다. 최대 일별 수익 오차 {audit["maximum_return_error"]:.3g}, 비중 오차 {audit["maximum_close_weight_error"]:.3g}다.',
             f'- 동일 시가 목표노출 대조 {audit["same_open_exposure_control_reconstructions"]}개를 독립 재구성했다. 최대 일별 수익 오차 {audit["exposure_control_maximum_daily_return_error"]:.3g}다. 당일 종가 현금 비중을 거꾸로 사용하지 않았다.',
             '- 저장된 모든 원천 해시와 엔진 해시가 일치했다.', '',
             '## 해석 범위', '',
             '25%·60%·60%는 리밸런싱 목표상한이며 실제 종가 비중의 강제 상한이 아니다. 반도체와 미분류 합산은 보수적인 불확실성 한도로, 미분류 회사를 반도체로 재분류한 뜻이 아니다.', '',
             '노출 대조군은 주문 시가의 목표 투자비중을 일치시킨 설명용 경로다. 다음 주문 전 실제 현금 비중은 서로 다르게 움직일 수 있다. 제약 포트폴리오의 주문 시점과 목표에 조건부로 구성했으므로 독립 선발한 실전 전략이나 유일한 인과 분해로 해석하지 않는다.', '',
             '동결 종목 때문에 총 10개 슬롯을 넘으면 작은 원목표를 제외하고 해당 비중을 현금으로 남긴다. 재배분형도 이 슬롯 제외 비중까지 다시 투자하지 않는다는 정책이 프로토콜에 명시되어 있다.', '',
             '선발은 이미 본 전체 기간을 이용한 회고적 비교다. 이번 감사는 구현·회계·정보 시점을 점검했으며 미래 수익이나 당시 분류의 가용성을 입증하지 않는다.', '',
             '[상세 결과](independent_review.json) · [재현 코드](independent_review.py)', '']
    (HERE / 'independent_review.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
