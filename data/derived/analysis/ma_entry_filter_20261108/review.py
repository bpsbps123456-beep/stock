"""Research-only entry gate. Reuse the operating portfolio engine unchanged."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE.parent / 'concentration_review_20261105'
spec = importlib.util.spec_from_file_location('concentration_input', BASE / 'review.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
S = R.S
END = '2026-11-04'


def moving_average(close, kind, length):
    if kind == 'SMA':
        return close.rolling(length, min_periods=length).mean()
    if kind == 'EMA':
        # Pine seeds EMA at the first valid price. Require N valid bars here too.
        return close.ewm(span=length, adjust=False, min_periods=length, ignore_na=True).mean()
    raise ValueError(kind)


def eligible(close, average, *, comparison_rtol=0.):
    """Roundoff-sized differences may be treated as equality when explicitly set."""
    return bool(np.isfinite(close) and np.isfinite(average) and
                close-average >= -comparison_rtol*max(abs(close), abs(average)))


def apply_gate(decision, current, values, *, comparison_rtol=0.):
    """No replacement or redistribution. Held names are exempt, not sold by the gate."""
    blocked = {key for key in decision['selected'] if key not in current and
               not eligible(*values[key], comparison_rtol=comparison_rtol)}
    return {**decision,
            'selected': [key for key in decision['selected'] if key not in blocked],
            'target': {key:w for key,w in decision['target'].items() if key not in blocked}}, blocked


@contextmanager
def entry_gate(book, kind, length, events, averages=None, *, comparison_rtol=0.):
    original = S.weekly_target
    if averages is None:
        averages = moving_average(book.cls, kind, length)
    closes = {ticker: book.cls[ticker].to_numpy(float) for ticker in book.ticker.values()}
    means = {ticker: averages[ticker].to_numpy(float) for ticker in book.ticker.values()}
    def weekly(b, k, session, current):
        decision = original(b,k,session,current)
        if not decision['ranked']:
            return decision
        values = {}
        for key in decision['selected']:
            ticker = b.ticker[key]
            values[key] = (float(closes[ticker][k-1]), float(means[ticker][k-1]))
        gated, blocked = apply_gate(decision, current, values, comparison_rtol=comparison_rtol)
        for key in decision['selected']:
            if key not in current:
                close, average = values[key]
                events.append({'session':session, 'signal_date':b.days[k-1], 'ticker':b.ticker[key],
                               'key':key, 'close':close, 'ma':average if np.isfinite(average) else None,
                               'allowed':key not in blocked, 'proposed_weight':decision['target'].get(key,0.)})
        return gated
    with patch.object(S,'weekly_target',weekly):
        yield


def period(book, path, begin, end, base=None):
    indices = [i for i,d in enumerate(path.days) if begin <= d <= end]
    if not indices:
        return None
    lo,hi=indices[0],indices[-1]
    krw=S.model_index(book,path)
    segment=np.r_[krw[lo-1] if lo else 100., krw[lo:hi+1]]
    out=R.metrics(segment)
    out['mean_open_exposure_pct']=float(R.exposure(path)[lo:hi+1].mean()*100)
    usd=np.r_[path.close[lo-1] if lo else 1.,path.close[lo:hi+1]]
    out['usd']=R.metrics(usd)
    if base is not None:
        # Same mean exposure, ex-post control, NOT a prospective trading rule.
        meanbase=R.exposure(base)[lo:hi+1].mean()
        scale=R.exposure(path)[lo:hi+1].mean()/meanbase if meanbase else 0.
        bc=np.r_[base.close[lo-1] if lo else 1.,base.close[lo:hi+1]]
        control=np.r_[1.,np.cumprod(1.+(bc[1:]/bc[:-1]-1.)*scale)]
        out['usd_mean_exposure_control']=R.metrics(control)
    return out


def validate(book, path, events):
    for event in events:
        assert event['signal_date'] < event['session']
        assert event['allowed'] == eligible(event['close'],event['ma'] if event['ma'] is not None else np.nan)
        if not event['allowed']:
            assert event['key'] not in path.rebalances[event['session']]
    for weights in path.rebalances.values():
        assert sum(weights.values()) <= 1.+1e-10
        assert max(weights.values(),default=0.) <= .25+1e-10
    return 'passed'


def main():
    protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    book=R.load_book(BASE/'inputs',END)
    baseline=S.run_model(book)
    paths={'baseline':baseline}; decisions={}; validation={}
    for kind in protocol['grid']['type']:
        for length in protocol['grid']['length']:
            name=f'{kind}{length}'; events=[]
            with entry_gate(book,kind,length,events):
                paths[name]=S.run_model(book)
            validation[name]=validate(book,paths[name],events)
            decisions[name]=events
    periods=[('front','2026-01-02','2026-05-29'),('back','2026-06-01',END),
             ('october','2026-10-12',END),('full','2026-01-02',END)]
    rows={label:{name:period(book,path,a,z,baseline) for name,path in paths.items()} for label,a,z in periods}
    candidates=list(paths)[1:]
    selected=max(candidates,key=lambda name:rows['front'][name]['return_pct'])
    monthly={f'2026-{m:02d}':{name:period(book,path,f'2026-{m:02d}-01',f'2026-{m:02d}-31',baseline)
             for name,path in paths.items()} for m in range(1,12)}
    blocked={name:{'blocked_attempts':sum(not e['allowed'] for e in ev),
                   'allowed_new_entries':sum(e['allowed'] for e in ev),
                   'unique_blocked_tickers':sorted({e['ticker'] for e in ev if not e['allowed']})}
             for name,ev in decisions.items()}
    inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((BASE/'inputs').glob('*.csv'))}
    inputs.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((ROOT/'data/normalized/etf_holdings').glob('*_456600.json')) if p.name[:10]<=END})
    result={'end_us':END,'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
            'periods':rows,'selected_by_front_only':selected,'monthly':monthly,
            'counts':blocked,'validation':validation,'input_sha256':inputs}
    (HERE/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'decisions.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame({name:S.model_index(book,path) for name,path in paths.items()},index=baseline.days).to_csv(HERE/'curves_krw.csv',index_label='us_session')
    lines=['# 이평선 아래 신규 편입 보류 — 2026-11-08','',
           '현행1+SMA/EMA × 20/60/120일의 6후보. 현재 운영 모델은 변경하지 않았다.',
           '미국 2026-11-04 종가까지 기존 고정 입력만 재사용했다. 이전 연구에서 이미 본 자료이며 새 독립 검증은 아니다.',
           '원래 7종목 선택·상한 배분 뒤, 기존 보유가 아닌 종목만 전일 종가 >= 이평선일 때 편입한다. 보류 배분은 현금, 대체·재배분 없음. 주중 복귀해도 다음 재조정까지 기다린다.',
           '기존 보유는 이평선 아래여도 이 필터로 팔지 않는다. 따라서 고점에서 산 뒤 하락하는 손실은 막지 못할 수 있다.',
           '', '| 설정 | 앞1~5월 수익 | 뒤6~11/4 수익 | 뒤 최대낙폭 | 뒤 평균투자비중 | 10/9 이후 수익 | 신규보류 시도 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name in paths:
        f,b,o=rows['front'][name],rows['back'][name],rows['october'][name]
        lines.append(f"| {name} | {f['return_pct']:+.2f}% | {b['return_pct']:+.2f}% | {b['mdd_pct']:.2f}% | {b['mean_open_exposure_pct']:.1f}% | {o['return_pct']:+.2f}% | {blocked[name]['blocked_attempts'] if name in blocked else '—'} |")
    b=rows['back'][selected]; original=rows['back']['baseline']
    lines+=['',f"앞 구간 수익만으로 고른 후보는 **{selected}**. 뒤 구간 현행 대비 수익 차이는 {b['return_pct']-original['return_pct']:+.2f}%p, 최대낙폭 차이는 {b['mdd_pct']-original['mdd_pct']:+.2f}%p다. 뒤 구간 1위로 바꾸지 않았다.",
            '뒤 구간 수익을 개선한 후보는 0/6이다. SMA120·EMA120은 보류가 없어 현행과 같다. SMA20은 최대낙폭만 0.15%p 얕아졌고 수익은 더 낮아, 이 표본에서 개선 근거가 없다. 기존 보유 하락을 막는 효과와 신규 편입 보류 효과를 혼동하지 않는다.',
            '', '## 지표 사용', '',
            '`ma_entry_gate.pine`을 Pine Editor에 붙여 넣고 차트에 추가한다. 기본은 정규장 일봉 SMA20이며 최적값이 아니다. 기준봉·길이·SMA/EMA를 변경할 수 있다.',
            '초록은 마지막 완성 기준봉에서 필터 통과, 빨강은 신규매수 보류, 회색은 이력 부족이다. 상태 전환 표시는 조건의 해제·진입이며 매수·매도 주문 신호가 아니다.',
            '일봉·분봉에서 일봉 기준을 사용할 수 있다. 기준봉보다 큰 차트 주기는 오류로 막고, 확정 기준봉만 반영한다. 지표는 7종목 순위·주간 교체·기존 보유 여부를 알지 못한다.',
            '지표는 정규장·분할조정·배당비조정 종가를 요청한다. Python의 같은 Close 정의와 맞췄지만 Yahoo와 TradingView 공급자 차이로 경계값은 달라질 수 있다.',
            'EMA는 입력 첫 종가에서 시작하므로 2025-06부터 예열한 연구와 더 긴 이력을 가진 차트의 초기값 영향도 다를 수 있다. 기본 SMA20은 최소 20개 완성 봉을 요구한다.',
            '', '## 검증·한계', '',
            '차단 기록·실제 배분·신호시점·상한 검증을 통과했다. Python 단위검증은 별도 tests/test_ma_entry_filter.py에 둔다.',
            '2026-11-08 TradingView 웹 Pine Editor에 실제 파일을 붙여 넣었으나 차트 추가 시 로그인 요구로 컴파일·실차트 표시는 확인하지 못했다. Pine 구문·화면 검증 통과로 표현하지 않는다. 상세 상태는 pine_validation.json에 기록한다.',
            '예열만 2025-06부터 쓰고 평가는 2026만 한다. 수수료0·가정 시가 체결, 호가·환전·세금·현금이자 제외. 현재 분류·기존 모델 선택편향은 남는다.',
            'results.json의 USD 동일 평균노출 대조는 현금을 더 들고 있어 덜 잃은 효과를 구분하는 사후 근사다. 미래 평균투자비중을 미리 아는 실행 전략이 아니다.',
            '신규보류 시도는 같은 종목의 다음 주 재심사를 중복 포함한다. 고유 표본 수로 해석하지 않는다. 일별 원화 곡선·월별 표·각 보류 원장·입력 SHA-256을 함께 보존했다.',
            '', 'Pine 구현 기준: [확정 봉·다른 시간대](https://www.tradingview.com/pine-script-docs/concepts/other-timeframes-and-data/) · [알림](https://www.tradingview.com/pine-script-docs/concepts/alerts/)',
            '', '재현: `py data/derived/analysis/ma_entry_filter_20261108/review.py`']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'periods':rows,'selected':selected,'counts':blocked,'validation':validation},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
