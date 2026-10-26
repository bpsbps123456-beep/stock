"""Audit two fixed initial 50/50 allocations using saved common-engine paths.

No new strategy selection, ratio optimization or cross-account rebalancing.
Run from any directory: py <this file>
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'results.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def returns_from_nav(nav):
    nav = np.asarray(nav, dtype=float)
    return nav / np.r_[1., nav[:-1]] - 1


def metrics(returns):
    nav = np.r_[1., np.cumprod(1 + np.asarray(returns, dtype=float))]
    return {'return_pct': float((nav[-1]-1)*100),
            'max_drawdown_pct': float(np.min(nav/np.maximum.accumulate(nav)-1)*100)}


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def compound_allocation(nav_a, nav_b):
    """Both component NAVs start at one; allocate initial capital only."""
    a, b = np.asarray(nav_a, dtype=float), np.asarray(nav_b, dtype=float)
    if a.shape != b.shape or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError('Mismatched or nonpositive component NAV')
    nav = .5*a + .5*b
    return nav, a/(a+b)


def analyze(a, b, dates, tickers):
    dates = np.asarray(dates, dtype=str)
    nav, sleeve_a = compound_allocation(a['daily_nav'], b['daily_nav'])
    ra, rb = np.asarray(a['daily_returns']), np.asarray(b['daily_returns'])
    np.testing.assert_allclose(a['daily_nav'], np.cumprod(1+ra), rtol=1e-10)
    np.testing.assert_allclose(b['daily_nav'], np.cumprod(1+rb), rtol=1e-10)
    rm = returns_from_nav(nav)
    # This is an accounting identity for the no-transfer mixture, NOT constant50%.
    prior_sleeve = np.r_[.5, sleeve_a[:-1]]
    np.testing.assert_allclose(rm, prior_sleeve*ra+(1-prior_sleeve)*rb, atol=1e-12)
    masks = {'full':np.ones(len(dates), dtype=bool),
             'train':dates < '2026-06-01',
             'from_2026_03_06':dates >= '2026-03-06',
             'june_july':(dates >= '2026-06-01') & (dates < '2026-08-01'),
             'august_september':dates >= '2026-08-01'}
    segments = {key:{**metrics(rm[mask]), 'component_daily_return_correlation':correlation(ra[mask],rb[mask])}
                for key,mask in masks.items() if mask.any()}
    monthly = {month:metrics(rm[np.char.startswith(dates,month)])
               for month in sorted(set(day[:7] for day in dates))}
    counts=[]; cash=[]; max_weights=[]; joint_holdings=[]; max_stock=None
    for t, (ha, hb) in enumerate(zip(a['daily_holdings'], b['daily_holdings'])):
        weights = {}
        for holdings, share in ((ha,sleeve_a[t]),(hb,1-sleeve_a[t])):
            for index, weight in holdings:
                weights[int(index)] = weights.get(int(index),0.) + share*weight
        weights = {index:weight for index,weight in weights.items() if weight > 1e-12}
        if sum(weights.values()) > 1+1e-9: raise ValueError('Borrowed cash in combined portfolio')
        counts.append(len(weights)); cash.append(max(0.,1-sum(weights.values()))*100)
        top = max(weights, key=weights.get) if weights else None
        largest = weights[top]*100 if top is not None else 0.
        max_weights.append(largest)
        if max_stock is None or largest > max_stock['weight_pct']:
            max_stock={'ticker':tickers[top] if top is not None else None,'date':str(dates[t]),'weight_pct':largest}
        joint_holdings.append([[index,weight] for index,weight in sorted(weights.items())])
    daily_reset = .5*ra + .5*rb
    return {
        'segments':segments,'monthly':monthly,
        'component_239120_cash_days':int(sum(len(h)==0 for h in a['daily_holdings'])),
        'component_239120_first_stock_date':next((str(day) for day,h in zip(dates,a['daily_holdings']) if h),None),
        'actual_max_combined_stocks':int(max(counts)), 'mean_combined_stocks':float(np.mean(counts)),
        'maximum_stock_concentration':max_stock, 'mean_cash_pct':float(np.mean(cash)),
        'sleeve_239120_weight_pct':{'initial':50.,'end':float(sleeve_a[-1]*100),
            'minimum_close':float(sleeve_a.min()*100),'minimum_close_date':str(dates[int(np.argmin(sleeve_a))]),
            'maximum_close':float(sleeve_a.max()*100),'maximum_close_date':str(dates[int(np.argmax(sleeve_a))]),
            'mean_close':float(sleeve_a.mean()*100)},
        'constant_daily_50_50_reset_diagnostic':{
            'interpretation':'Arithmetic mean of component daily returns; implies resetting account capitals to50/50 every preceding Close. Not the requested no-transfer mixture, and not replayed US-open execution.',
            'segments':{key:metrics(daily_reset[mask]) for key,mask in masks.items() if mask.any()}},
        'daily_nav':nav.tolist(),'daily_returns':rm.tolist(),
        'daily_239120_sleeve_weight':sleeve_a.tolist(),'daily_combined_stocks':counts,
        'daily_max_stock_weight_pct':max_weights,'daily_cash_pct':cash,'daily_holdings':joint_holdings,
    }


def synthetic_tests():
    # The two paths reverse their first-day gains. No-transfer NAV returns to1;
    # daily resetting would compound two +25% days to1.5625.
    nav,w = compound_allocation([2.,1.],[.5,1.])
    np.testing.assert_allclose(nav,[1.25,1.])
    np.testing.assert_allclose(w,[.8,.5])
    np.testing.assert_allclose(returns_from_nav(nav),[.25,-.2])
    ra=np.array([1.,-.5]);rb=np.array([-.5,1.])
    np.testing.assert_allclose(np.prod(1+.5*ra+.5*rb),1.5625)
    # A cash account remains one; the invested account doubles then halves.
    nav,w=compound_allocation([1.,1.],[2.,1.])
    np.testing.assert_allclose(nav,[1.5,1.])
    np.testing.assert_allclose(w,[1/3,1/2])
    np.testing.assert_allclose(metrics(returns_from_nav(nav))['max_drawdown_pct'],-100/3)
    return {'two_account_nav_identity':True,'drifting_sleeves_not_daily_reset':True,
            'cash_account_and_compounding':True,'drawdown_includes_initial_nav':True}


def run():
    before=sha(SOURCE)
    source=json.loads(SOURCE.read_text(encoding='utf-8'))
    dates=source['dates']; result={
        'protocol':{
            'created_utc':datetime.now(timezone.utc).isoformat(), 'source':str(SOURCE),
            'source_sha256':before,'audit_code_sha256':sha(Path(__file__)),
            'pairs':[['239120','H6e'],['239120','H6']], 'initial_allocations':[.5,.5],
            'rule':'Initial50/50 capital in two independent strategy accounts; no subsequent transfers between accounts. Combined NAV=.5*NAV239120+.5*NAVClaude.',
            'scenarios':['publication_lag0','publication_lag1','publication_lag2'],
            'dates':[dates[0],dates[-1]],'no_optimization':True,
            'inherited_model':'Saved common US Open execution and same-day Close NAV; zero explicit trading costs; KRWcash; previous-US-session FX Close as open FX proxy; dividends/taxes excluded.',
            'selection_status':'Fixed post-research diagnostics, not independent validation or a new winner search.',
            'segment_policy':'June-July/August-September metrics on continuous accounts inherit earlier holdings and drifting sleeve weights. FreshAugust starts both accounts at cashNAV1 and then splits initial capital50/50.',
            'drawdown_policy':'End-of-US-day KRW NAV including initialNAV1; not intraday drawdown.',
            'daily_reset_policy':'The .5*rA+.5*rB path is retained only as a separate diagnostic; it resets sleeve capital every preceding Close and is not the no-transfer rule.',
            'correlation_policy':'Report Pearson daily simple-return correlations over full history, fixed March6 onward, and June-July. Full-history correlation includes early all-cash zero-return days of239120; March6 onward is a fixed comparison window, not a guarantee of full exposure every day.',
            'claim_scope':'The original +232%/-16%/.52 mixture calculation was not located in compare_codex.py, so its exact combination method is unconfirmed. Daily resetting is an explicitly separate hypothetical diagnostic, not an assertion about the original method. Common-engine paths do not reproduce the original construction.'},
        'pairs':{},'validation':synthetic_tests()}
    for other in ('H6e','H6'):
        pair={}
        for lag in range(3):
            scenario=f'publication_lag{lag}'
            a=source['strategies']['239120']['scenarios'][scenario]
            b=source['strategies'][other]['scenarios'][scenario]
            row=analyze(a,b,dates,source['tickers'])
            af=a['fresh_starts']['august'];bf=b['fresh_starts']['august']
            assert af['date']==bf['date']
            fresh_dates=[day for day in dates if day>=af['date']]
            row['fresh_august']={'start':af['date'],**analyze(af,bf,fresh_dates,source['tickers'])}
            pair[scenario]=row
        result['pairs']['239120+'+other]=pair
    result['validation']['source_results_unchanged']=sha(SOURCE)==before
    result['validation']['all_combined_holdings_within_10']=all(row['actual_max_combined_stocks']<=10 for pair in result['pairs'].values() for row in pair.values())
    assert result['validation']['source_results_unchanged'] and result['validation']['all_combined_holdings_within_10']
    (HERE/'mixture_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# §16 고정 50:50 혼합 감사','','기존 공통 시가 결과만 사용했다. #239120과H6e/H6에 시작자금의절반씩 배정하고, 이후 두계좌 사이 자금이동은하지않는다. `혼합NAV=0.5×계좌A NAV+0.5×계좌B NAV`. 비율탐색·최적화없음.','',
           '| 조합 | 공시지연(US일) | 전체 | 종가MDD | 6~7월 | 8~9월 | 8월현금출발 | 상관 전체/3월6일이후/6~7월 | 합산최대종목 | 최대종목비중 | 최종#239120비중 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,pair in result['pairs'].items():
        for scenario,row in pair.items():
            m=row['segments'];fresh=row['fresh_august']['segments']['full']; concentration=row['maximum_stock_concentration']
            nums=[m['full']['return_pct'],m['full']['max_drawdown_pct'],m['june_july']['return_pct'],m['august_september']['return_pct'],fresh['return_pct']]
            cors='/'.join(f'{m[k]["component_daily_return_correlation"]:.3f}' for k in ('full','from_2026_03_06','june_july'))
            lines.append(f'| {name} | {scenario[-1]} | '+' | '.join(f'{v:+.2f}%' for v in nums)+f' | {cors} | {row["actual_max_combined_stocks"]} | {concentration["weight_pct"]:.2f}% ({concentration["ticker"]}) | {row["sleeve_239120_weight_pct"]["end"]:.2f}% |')
    lines+=['','6~7월·8~9월은기존보유와변한계좌비중을이어받은성과다. 8월현금출발은두계좌를각각현금으로재시작한별도경로다. 상관은일별원화단순수익률의Pearson상관이며낮은상관만으로미래위험감소를보장하지않는다.','',
            '전체상관에는#239120의초기현금·0수익일이포함된다. 이를구분하려고고정된3월6일이후와6~7월상관을함께표시했다. 3월6일이후도매일100%투자됨을뜻하지않는다.','',
            '## 매일 50:50 재설정의 별도 진단','',
            '매일수익률을단순히반씩더하면전일종가마다두계좌의자금을50:50으로다시맞춘경로가된다. 실제두계좌유지와다르며, 그종가자금이동을미국시가에체결한경로로간주할수없다.','',
            '| 조합 | 공시지연 | 초기50:50·이동없음 전체/MDD | 일별50:50재설정 진단 전체/MDD |','| --- | ---: | ---: | ---: |']
    for name,pair in result['pairs'].items():
        for scenario,row in pair.items():
            fixed=row['segments']['full'];reset=row['constant_daily_50_50_reset_diagnostic']['segments']['full']
            lines.append(f'| {name} | {scenario[-1]} | {fixed["return_pct"]:+.2f}% / {fixed["max_drawdown_pct"]:.2f}% | {reset["return_pct"]:+.2f}% / {reset["max_drawdown_pct"]:.2f}% |')
    lines+=['','원래§16의+232%·MDD−16%·상관0.52의정확한혼합산식은compare_codex.py에서확인되지않았다. 위일별재설정은별도가능성을비교하는진단이며원래산식이었다고단정하지않는다. 이감사는원래수치를그대로재현한것이아니다. 공통엔진의시가소수수량·환율근사·세금배당제외등한계가그대로적용된다. 평균상관·과거혼합개선으로최고미래전략을선정하지않는다.','',
            '합산종목비중은각종가의계좌NAV비중으로가중하며, 같은종목은합산했다. 계좌별비중의최소·최대·최종값, 월별성과와일별보유는JSON에있다. 원결과해시불변및현금·복리합성검증을통과했다.','']
    (HERE/'mixture_audit.md').write_text('\n'.join(lines),encoding='utf-8')
    for name,pair in result['pairs'].items():
        for scenario,row in pair.items():
            print(name,scenario,json.dumps({'full':row['segments']['full'],'fresh_august':row['fresh_august']['segments']['full'],'stocks':row['actual_max_combined_stocks'],'max_stock':row['maximum_stock_concentration'],'sleeve':row['sleeve_239120_weight_pct']},ensure_ascii=False))
    print(json.dumps(result['validation'],ensure_ascii=False))


if __name__=='__main__':
    run()
