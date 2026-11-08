"""Independent arithmetic, daily-exit signal, turnover and selected-path audit.

Reuses the required canonical wealth/weekly-allocation engine, but never the
exit runner's signal predicate, moving-average, period or turnover functions.
Writes only exit_1_200/audit.json; does not fetch or modify operating inputs.
"""
from __future__ import annotations

from collections import Counter
import copy
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE/'exit_1_200'
SPEC = importlib.util.spec_from_file_location('independent_grid_helpers', HERE/'audit_grid.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
M, S = A.M, A.M.S
TOL = 1e-9


class SignalCalculator:
    """Direct window sums and scalar recursive EMA; no pandas MA implementation."""
    def __init__(self, signals):
        self.days = {day:i for i,day in enumerate(signals.index)}
        self.columns = {ticker:signals[ticker].to_numpy(float) for ticker in signals}
        self.ema = {}

    def value(self, name, ticker, day):
        length, k = int(name[3:]), self.days[day]
        raw = self.columns[ticker]
        if name.startswith('SMA'):
            sample = raw[max(0,k-length+1):k+1]
            return (math.fsum(sample)/length if len(sample) == length and
                    all(math.isfinite(x) for x in sample) else math.nan)
        cache_key = (length,ticker)
        if cache_key not in self.ema:
            state, count, alpha = None, 0, 2./(length+1)
            result = np.full(len(raw),np.nan)
            for i,price in enumerate(raw):
                if math.isfinite(price):
                    state = price if state is None else (1-alpha)*state+alpha*price
                    count += 1
                if count >= length:
                    result[i] = state
            self.ema[cache_key] = result
        return self.ema[cache_key][k]


def own_run(book, averages, kind, length):
    """Independent exit rule around the canonical entry gate and wealth engine."""
    entries, exits, unknown = [], [], []
    def callback(b,k,day,weights):
        sold = []
        for key,weight in weights.items():
            ticker = b.ticker[key]
            close, ma = float(b.cls[ticker].iloc[k-1]), float(averages[ticker].iloc[k-1])
            event = {'session':day,'signal_date':b.days[k-1],'key':key,'ticker':ticker,
                     'close':close,'ma':ma,'weight':weight}
            if not (math.isfinite(close) and math.isfinite(ma)):
                unknown.append(event)
            elif close < ma:
                sold.append(key)
                exits.append({**event,'open':float(b.opn[ticker].iloc[k])})
        return sold
    with M.entry_gate(book,kind,length,entries,averages=averages):
        path = S.run_model(book,exit_at_open=callback)
    return path, entries, exits, unknown


def own_turnover(book,path):
    """Reconstruct pre-trade drift directly, without the engine's drift helper."""
    previous, output = {}, []
    indices = {day:k for k,day in enumerate(book.days)}
    for day,final in zip(path.days,path.weights):
        k = indices[day]
        gross = {key:float(book.opn[book.ticker[key]].iloc[k]/
                          book.opn[book.ticker[key]].iloc[k-1]) for key in previous}
        assert all(math.isfinite(x) and x>0 for x in gross.values())
        total = 1.+math.fsum(weight*(gross[key]-1.) for key,weight in previous.items())
        before = {key:weight*gross[key]/total for key,weight in previous.items()}
        changes = [final.get(key,0.)-before.get(key,0.) for key in set(final)|set(before)]
        buys, sells = math.fsum(max(x,0.) for x in changes), math.fsum(max(-x,0.) for x in changes)
        output.append([buys,sells,buys+sells])
        previous = final
    return np.array(output)


def compare_events(actual,expected,fields):
    assert len(actual) == len(expected), (len(actual),len(expected))
    error = 0.
    # Exit callback uses portfolio insertion order, which must remain stable.
    for left,right in zip(actual,expected):
        for field in fields:
            if field in ['session','signal_date','key','ticker']:
                assert left[field] == right[field], (field,left,right)
            elif field == 'allowed':
                assert bool(left[field]) == A.bool_value(right[field])
            elif left[field] is None or not np.isfinite(left[field]):
                assert pd.isna(right[field])
            else:
                delta = abs(float(left[field])-float(right[field]))
                error = max(error,delta)
                assert delta <= 1e-8, (field,delta,left,right)
    return error


def main():
    started = time.perf_counter()
    result = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    assert A.hash_file(OUT/'protocol.json') == result['protocol_sha256']
    prior = json.loads((A.OUT/'results.json').read_text(encoding='utf-8'))
    assert result['input_sha256'] == prior['input_sha256']
    for name,sha in result['input_sha256'].items():
        assert A.hash_file(M.ROOT/name) == sha, name
    warm_meta = json.loads((A.OUT/'inputs/warmup_meta.json').read_text(encoding='utf-8'))
    assert A.hash_file(A.OUT/'inputs/close_warmup.csv') == warm_meta['output_sha256']
    assert not warm_meta['unresolved_active_tickers']
    prior_audit = json.loads((A.OUT/'audit.json').read_text(encoding='utf-8'))
    assert prior_audit['status'] == 'passed'

    frame = pd.read_csv(OUT/'summary.csv')
    indexed = frame.set_index('name')
    curves = pd.read_csv(OUT/'curves_krw.csv',index_col=0)
    entries = pd.read_csv(OUT/'entry_decisions.csv')
    exits = pd.read_csv(OUT/'exits.csv')
    unknown = pd.read_csv(OUT/'unknown_held_signals.csv')
    turns = pd.read_csv(OUT/'turnover.csv')
    monthly = pd.read_csv(OUT/'monthly.csv')
    names = {f'{kind}{length}' for kind in ['SMA','EMA'] for length in range(1,201)}
    assert len(frame) == 401 and not frame.name.duplicated().any()
    assert set(frame.name) == set(curves) == names|{'baseline'}
    assert set(entries.name) == names
    assert set(exits.name) <= names-{'SMA1','EMA1'}
    assert np.array_equal(curves.SMA1,curves.baseline)
    assert np.array_equal(curves.EMA1,curves.baseline)
    old_curves = pd.read_csv(A.OUT/'curves_krw.csv',index_col=0)
    np.testing.assert_allclose(curves.baseline,old_curves.baseline,rtol=0,atol=1e-9)

    metric_error, monthly_error, cost_error = 0., 0., 0.
    for label,(start,finish) in protocol['periods'].items():
        returns,mdd = A.direct_metrics(curves,start,finish)
        for name in curves:
            for suffix,values in [('return_pct',returns),('mdd_pct',mdd)]:
                error = abs(float(indexed.at[name,f'{label}_{suffix}'])-values[name])
                metric_error = max(metric_error,error)
                assert error <= 1e-8,(name,label,suffix,error)
                assert abs(result['periods'][name][label][suffix]-values[name]) <= 1e-8
    for month in range(1,12):
        label = f'2026-{month:02d}'
        returns,mdd = A.direct_metrics(curves,label+'-01',label+'-31')
        part = monthly.loc[monthly.month == label].set_index('name')
        assert set(part.index) == set(curves)
        for name in curves:
            for suffix,values in [('return_pct',returns),('mdd_pct',mdd)]:
                error = abs(float(part.at[name,suffix])-values[name])
                monthly_error = max(monthly_error,error)
                assert error <= 1e-8

    rows = frame.to_dict('records')
    selected = A.own_winner(rows,'front_return_pct')
    by_type = {kind:A.own_winner(rows,'front_return_pct',kind) for kind in ['SMA','EMA']}
    assert selected == result['selected_by_front_only']
    assert by_type == result['selected_by_type_front_only']
    for field,key in [('back_return_pct','best_back_hindsight'),
                      ('full_return_pct','best_full_hindsight'),
                      ('back_mdd_pct','best_back_mdd_hindsight')]:
        assert A.own_winner(rows,field) == result[key]
    changed = copy.deepcopy(rows)
    for i,row in enumerate(changed):
        row.update(back_return_pct=(-1)**i*1e8,back_mdd_pct=-i*1e5,
                   full_return_pct=i*1e6,recent_return_pct=-i*1e6)
    assert A.G.select_front(changed) == selected
    assert all(A.G.select_front(changed,kind) == by_type[kind] for kind in by_type)
    candidate,base = frame.loc[frame.name!='baseline'],indexed.loc['baseline']
    rb = candidate.back_return_pct > base.back_return_pct+TOL
    mb = candidate.back_mdd_pct > base.back_mdd_pct+TOL
    counts = {'candidates':400,'back_return_better':int(rb.sum()),'back_mdd_better':int(mb.sum()),
              'back_return_and_mdd_better':int((rb&mb).sum()),
              'front_and_back_return_better':int(((candidate.front_return_pct>base.front_return_pct+TOL)&rb).sum()),
              'back_both_better_than_entry_same_n':int(((candidate.vs_entry_back_return_pct_pp>TOL)&
                                                       (candidate.vs_entry_back_mdd_pct_pp>TOL)).sum())}
    assert counts == result['counts']
    assert int((candidate.front_return_pct.max()-candidate.front_return_pct<=TOL).sum()) == result['front_tie_count']
    old_summary = pd.read_csv(A.OUT/'summary.csv').set_index('name')
    for field in ['back_return_pct','back_mdd_pct','full_return_pct']:
        np.testing.assert_allclose(indexed['vs_entry_'+field+'_pp'],
                                   indexed[field]-old_summary.loc[indexed.index,field],rtol=0,atol=1e-8)

    assert not turns.duplicated(['name','session']).any()
    assert len(turns) == len(curves)*len(curves.columns)
    assert set(turns.name) == set(curves.columns)
    np.testing.assert_allclose(turns.buy_weight+turns.sell_weight,turns.one_way_turnover,rtol=0,atol=1e-12)
    assert (turns[['buy_weight','sell_weight','one_way_turnover']]>=0).all().all()
    turn_matrix = turns.pivot(index='session',columns='name',values='one_way_turnover').reindex(index=curves.index,columns=curves.columns)
    assert not turn_matrix.isna().any().any()
    for label,(start,finish) in protocol['periods'].items():
        total = turn_matrix.loc[start:finish].sum()
        np.testing.assert_allclose(total,indexed.loc[total.index,label+'_turnover'],rtol=0,atol=1e-8)
    for bps in [5,10]:
        adjusted = curves*np.cumprod(1.-turn_matrix*bps/10000.,axis=0)
        for label,(start,finish) in protocol['periods'].items():
            returns,_ = A.direct_metrics(adjusted,start,finish)
            for name,value in returns.items():
                error = abs(indexed.at[name,f'{label}_return_{bps}bp_pct']-value)
                cost_error = max(cost_error,error)
                assert error <= 1e-8

    book = M.R.load_book(M.BASE/'inputs',M.END)
    warmup = pd.read_csv(A.OUT/'inputs/close_warmup.csv',index_col=0)
    signals = pd.concat([warmup.reindex(columns=book.cls.columns),book.cls])
    assert not signals.index.duplicated().any() and signals.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(signals.loc[book.days],book.cls,check_names=False)
    calc = SignalCalculator(signals)
    prior_day = {day:book.days[k-1] for k,day in enumerate(book.days) if k}
    ma_error, reasons = 0., Counter()
    for kind,ledger in [('exit',exits),('entry',entries),('unknown',unknown)]:
        for event in ledger.to_dict('records'):
            name,ticker,day = event['name'],event['ticker'],event['signal_date']
            assert day == prior_day[event['session']]
            assert book.ticker[event['key']] == ticker
            close = float(signals.at[day,ticker])
            mean = calc.value(name,ticker,day)
            assert abs(close-event['close']) <= 1e-8 or (pd.isna(close) and pd.isna(event['close']))
            if math.isfinite(mean):
                error = abs(mean-event['ma'])
                ma_error = max(ma_error,error)
                assert error <= 1e-8,(kind,name,ticker,day,error)
            else:
                assert pd.isna(event['ma'])
            finite = math.isfinite(close) and math.isfinite(mean)
            if kind == 'exit':
                assert finite and close < mean and event['weight']>0
                price = float(book.opn.at[event['session'],ticker])
                assert math.isfinite(price) and price>0 and abs(price-event['open'])<=1e-8
            elif kind == 'unknown':
                assert not finite and event['weight']>0
            else:
                allowed = finite and close >= mean
                count = int(np.isfinite(calc.columns[ticker][:calc.days[day]+1]).sum())
                assert count == event['history_count']
                assert allowed == A.bool_value(event['allowed'])
                reason = ('allowed' if allowed else 'missing_close' if not math.isfinite(close) else
                          'below_ma' if math.isfinite(mean) else
                          'insufficient_history' if count<int(name[3:]) else 'missing_recent_history')
                assert reason == event['reason']
                reasons[reason] += 1
    assert not exits.duplicated(['name','session','key']).any()
    assert len(unknown) == result['unknown_held_signals'] == int(frame.unknown_held_signals.sum())
    for name,row in indexed.iterrows():
        part = exits.loc[exits.name==name]
        assert len(part) == row.ma_exits
        assert int((part.session>='2026-06-01').sum()) == row.back_ma_exits
        assert len(unknown.loc[unknown.name==name]) == row.unknown_held_signals
        blocked = entries.loc[(entries.name==name)&~entries.allowed]
        assert len(blocked) == row.blocked_attempts
        assert int((blocked.reason=='below_ma').sum()) == row.below_ma_attempts
        assert len(blocked.loc[blocked.reason!='below_ma']) == row.missing_history_attempts
        assert blocked.ticker.nunique() == row.unique_blocked_tickers
        assert int((blocked.session<='2026-05-29').sum()) == row.front_blocked
        assert int((blocked.session>='2026-06-01').sum()) == row.back_blocked
    print(f'All-grid arithmetic and {len(exits)} exits / {len(entries)} entries verified; replaying paths.',flush=True)

    representatives = list(dict.fromkeys([selected,*by_type.values(),result['best_back_hindsight'],
        result['best_full_hindsight'],result['best_back_mdd_hindsight'],
        'SMA125','SMA132','SMA18','EMA51','SMA200','EMA200','SMA1','EMA1']))
    replay, paths = {}, {}
    baseline = S.run_model(book)
    np.testing.assert_allclose(S.model_index(book,baseline),curves.baseline,rtol=0,atol=1e-8)
    for name in ['baseline']+representatives:
        if name == 'baseline':
            path,ev,ex,un = baseline,[],[],[]
        else:
            kind,length = name[:3],int(name[3:])
            averages = A.independent_ma(signals,kind,length).reindex(book.cls.index)
            path,ev,ex,un = own_run(book,averages,kind,length)
        curve_error = float(np.max(np.abs(S.model_index(book,path)-curves[name].to_numpy())))
        assert curve_error <= 1e-8,(name,curve_error)
        ee = compare_events(ev,entries.loc[entries.name==name].to_dict('records'),
                            ['session','signal_date','key','ticker','close','ma','allowed','proposed_weight'])
        xe = compare_events(ex,exits.loc[exits.name==name].to_dict('records'),
                            ['session','signal_date','key','ticker','close','ma','weight','open'])
        ue = compare_events(un,unknown.loc[unknown.name==name].to_dict('records'),
                            ['session','signal_date','key','ticker','close','ma','weight'])
        actual_turns = own_turnover(book,path)
        stored_turns = turns.loc[turns.name==name].set_index('session').loc[path.days,
                         ['buy_weight','sell_weight','one_way_turnover']].to_numpy()
        turn_error = float(np.max(np.abs(actual_turns-stored_turns)))
        assert turn_error <= 1e-10
        assert int(indexed.at[name,'new_entries']) == sum(e['side']=='buy' for e in path.events)
        assert int(indexed.at[name,'full_exits']) == sum(e['side']=='sell' for e in path.events)
        executed = [e for e in path.events if e.get('reason')=='exit_rule']
        assert len(executed)==len(ex)
        held = dict(zip(path.days,path.weights))
        for event in ex:
            assert event['key'] not in held[event['session']]
        assert all(e['date'] in path.rebalances for e in path.events if e['side']=='buy')
        for weights in path.rebalances.values():
            assert sum(weights.values())<=1+1e-10 and max(weights.values(),default=0)<=.25+1e-10
            sectors = Counter()
            for key,weight in weights.items():
                sectors[book.sector[key]] += weight
            assert max(sectors.values(),default=0)<=.6+1e-10
            assert sum(w for key,w in weights.items() if S.in_semis_budget(book.sector[key]))<=.5+1e-10
        for label,(start,finish) in protocol['periods'].items():
            exposure = np.mean([sum(weights.values()) for day,weights in zip(path.days,path.weights)
                                if start<=day<=finish])*100
            assert abs(exposure-indexed.at[name,label+'_mean_open_exposure_pct'])<=1e-8
        replay[name] = {'full_curve_max_error':curve_error,'entry_values_max_error':ee,
                        'exit_values_max_error':xe,'unknown_values_max_error':ue,
                        'turnover_max_error':turn_error,'ma_exits':len(ex),
                        'new_entries':int(indexed.at[name,'new_entries']),
                        'full_exits':int(indexed.at[name,'full_exits']),
                        'weekly_target_caps_verified':True,'weekly_only_buys_verified':True}
        paths[name] = path
        print(f'{name}: curve and {len(path.days)} turnover rows verified.',flush=True)

    cliff = []
    left,right = paths['SMA125'],paths['SMA126']
    for day,lw,rw in zip(left.days,left.weights,right.weights):
        if lw != rw:
            delta = {book.ticker[key]:{'SMA125':lw.get(key,0.),'SMA126':rw.get(key,0.)}
                     for key in sorted(set(lw)|set(rw)) if abs(lw.get(key,0.)-rw.get(key,0.))>1e-10}
            if delta:
                cliff.append({'session':day,'differing_post_open_weights':delta})
        if len(cliff)>=3:
            break
    output = {'status':'passed','grid_candidates':400,'stored_curve_count':401,
              'metric_cells_checked':401*len(protocol['periods'])*2,
              'monthly_metric_cells_checked':401*11*2,'cost_return_cells_checked':401*4*2,
              'max_metric_error':metric_error,'max_monthly_error':monthly_error,'max_cost_error':cost_error,
              'all_exit_rows_independent_ma_checked':len(exits),
              'all_entry_rows_independent_ma_checked':len(entries),'max_ledger_ma_error':ma_error,
              'entry_reason_counts':dict(reasons),'unknown_held_signals_checked':len(unknown),
              'all_stored_turnover_rows_algebra_checked':len(turns),
              'source_sha_and_previous_warmup_unit_audit_verified':True,
              'one_day_noop_and_frozen_baseline_verified':True,
              'front_selection_verified':selected,'front_selection_unchanged_after_back_score_mutation':True,
              'same_length_entry_only_deltas_verified':True,'summary_counts_verified':counts,
              'manual_replays':replay,'SMA125_vs_SMA126_first_weight_differences':cliff,
              'limitations':[
                  'All-grid stored metrics, signals, row counts and turnover/cost algebra are checked; actual holdings, turnover and complete daily wealth are independently replayed only for the listed paths.',
                  'Reuses the canonical wealth and weekly allocation engine by design; separate execution-order tests cover overnight loss, cash, buffer removal, missing execution price and weekly reentry.',
                  'The 2026 period was already observed. Front selection is chronological arithmetic, not a new untouched holdout. All 800 entry/exit candidates share the model and highly overlapping trades.',
                  'SMA2 minimizes exposure and does not establish superior stock timing; cash and KRW translation can materially affect the comparison.',
                  'Five/ten-basis-point costs are multiplicative sensitivity on all changes in portfolio weight, including partial rebalances; they are not simulated broker cash flows or measured fills.',
                  'Signal is prior completed daily close below MA; execution is next regular open, after any overnight gap. Missing MA/close means hold existing and block entry, not a sale.',
                  'SMA125/SMA126 and nearby winners are ex-post sensitivity comparisons and cannot validate selecting the best full-year length.'
              ],'elapsed_seconds':time.perf_counter()-started}
    (OUT/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ['manual_replays','limitations',
                     'SMA125_vs_SMA126_first_weight_differences']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
