"""Post-selection diagnostics for existing candidates793 and798 only.

Does not modify the frozen engine, selection, protocol or grid artifacts.
"""
from pathlib import Path
import sys
import json
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
sys.path.insert(0,str(REPO))
from gbrc.etf.research_concentration import (
    candidate_grid, load_inputs, _simulate_one, exposure_matched, segment_metrics, STATS,
)
from gbrc.etf.research_validation import _one_metrics
from gbrc.etf.research_us_only import _hash


def load(path): return json.loads(path.read_text(encoding='utf-8'))


def pack(detail,dates):
    r,stats,weights,raw,target,events=detail
    days=np.asarray(dates,dtype=str)
    return {'segments':segment_metrics(r,days),
        'statistics':{name:float(value) for name,value in zip(STATS,stats)},
        'monthly':{month:_one_metrics(r[np.char.startswith(days,month)]) for month in sorted(set(d[:7] for d in days))},
        'daily_returns':r.tolist(),'daily_nav':np.cumprod(1+r).tolist(),
        'daily_holdings':[[[int(j),float(row[j])] for j in np.flatnonzero(row>1e-12)] for row in weights],
        'trace_diagnostics':{'decision_events_including_cash':int(events.sum()),
            'maximum_target_investment_fraction':float(target.sum(axis=1).max()),
            'all_daily_returns_finite':bool(np.isfinite(r).all()),
            'target_overallocation_days':int(stats[18]),
            'target_cap_violation_without_frozen_exception_days':int(stats[19])}}


def run():
    engine=REPO/'gbrc/etf/research_concentration.py'
    protected=[engine,HERE/'protocol.json',HERE/'selection.json',HERE/'configurations.json',
               HERE/'results.json',HERE/'summary.json',HERE/'validation.json',HERE/'candidate_metrics.json',
               *[HERE/f'publication_lag{lag}_{kind}.npy' for lag in range(3) for kind in ('returns','stats')]]
    before={str(path):_hash(path) for path in protected}
    protocol=load(HERE/'protocol.json');selection=load(HERE/'selection.json')
    assert selection['selected_index']==796
    assert _hash(engine)==protocol['code_sha256']
    params,configs=candidate_grid()
    assert configs==load(HERE/'configurations.json')
    arrays,tickers,sector,inputs,source_paths=load_inputs()
    source_before={str(path):_hash(path) for path in source_paths}
    assert source_before==protocol['source_sha256']
    dates=arrays['dates'].astype(str);hstart=int(np.searchsorted(dates,'2026-08-01'))
    result={'research_status':'Post-selection diagnostic of two existing grid neighbors; no new rules, thresholds or ratio search.',
        'selected_index_unchanged':796,'diagnostic_indices':[798,793],
        'original_protocol_path':str(HERE/'protocol.json'),'original_protocol_sha256':_hash(HERE/'protocol.json'),
        'diagnostic_script_sha256':_hash(Path(__file__)),'dates':dates.tolist(),'tickers':tickers,
        'profiles':{},'protected_artifact_sha256':before}
    archives={}
    for index in (798,793):
        profile={'index':index,'config':configs[index],
            'worst_return_pct':float(selection['worst_return'][index]*100),
            'worst_max_drawdown_pct':float(selection['worst_max_drawdown'][index]*100),'scenarios':{}}
        for name,arguments in inputs.items():
            detail=_simulate_one(params[index],*arguments,0,True)
            grid=np.load(HERE/f'{name}_returns.npy',mmap_mode='r')
            gridstats=np.load(HERE/f'{name}_stats.npy',mmap_mode='r')
            np.testing.assert_allclose(detail[0],grid[index],atol=1e-7)
            np.testing.assert_allclose(detail[1],gridstats[index],rtol=1e-10,atol=1e-10)
            fresh=_simulate_one(params[index],*arguments,hstart,True)
            control=exposure_matched(detail,arguments,0)
            fresh_control=exposure_matched(fresh,arguments,hstart)
            row=pack(detail,dates)
            row['fresh_august']={'start':str(dates[hstart]),**pack(fresh,dates[hstart:])}
            row['exposure_matched_uncapped_control']={
                'interpretation':protocol['exposure_matched_control'],
                'segments':segment_metrics(control['returns'],dates),
                'fresh_august_segments':segment_metrics(fresh_control['returns'],dates[hstart:]),
                'daily_returns':control['returns'].tolist(),'daily_nav':control['nav']}
            profile['scenarios'][name]=row
            for suffix,path in [('full',detail),('fresh_august',fresh)]:
                prefix=f'candidate{index}_{name}_{suffix}'
                for key,value in zip(('returns','stats','weights','precap_targets','projected_targets','events'),path):
                    archives[prefix+'_'+key]=value
        result['profiles'][str(index)]=profile
    result['validation']={
        'frozen_artifacts_unchanged':before=={str(path):_hash(path) for path in protected},
        'source_inputs_unchanged':source_before=={str(path):_hash(path) for path in source_paths},
        'original_selected_index_still796':load(HERE/'selection.json')['selected_index']==796,
        'both_neighbors_match_all_three_saved_grids':True,
        'all_full_and_fresh_paths_have_at_most10_stocks':all(
            row['statistics']['max_stocks']<=10 and row['fresh_august']['statistics']['max_stocks']<=10
            for p in result['profiles'].values() for row in p['scenarios'].values())}
    assert all(result['validation'].values())
    np.savez_compressed(HERE/'neighbor_paths.npz',**archives)
    (HERE/'neighbor_profiles.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for index,p in result['profiles'].items():
        print(index,json.dumps(p['config'],ensure_ascii=False))
        for scenario,row in p['scenarios'].items():
            print(scenario,json.dumps({'full':row['segments']['full'],'fresh_august':row['fresh_august']['segments']['full'],
                'control':row['exposure_matched_uncapped_control']['segments']['full'],
                'trading_days':row['statistics']['trading_days'],'turnover':row['statistics']['one_way_turnover'],
                'mean_cash':row['statistics']['mean_cash'],'max_stock':row['statistics']['max_stock_weight'],
                'max_uncertain_semiconductor':row['statistics']['max_semiconductor_plus_unknown_weight']},ensure_ascii=False))
    print(json.dumps(result['validation'],ensure_ascii=False))


if __name__=='__main__': run()
