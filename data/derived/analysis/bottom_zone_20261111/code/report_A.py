"""Render all fixed result cells and machine-readable audit links in Korean."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def pct(v):return '—' if v is None or pd.isna(v) else f'{v*100:.1f}%'
def num(v):return '—' if v is None or pd.isna(v) else f'{v:,.2f}'
def table(cols,rows):return '\n'.join(['| '+' | '.join(cols)+' |','|'+'---|'*len(cols)]+['| '+' | '.join(map(str,r))+' |' for r in rows])
def rate(v):return f'{pct(v["rate"])} ({v["scored"]}건; 미정{v["pending"]})'


def main():
    s=json.loads((ROOT/'results/summary_A.json').read_text(encoding='utf-8'))
    bs=json.loads((ROOT/'results/bottom_summary_A.json').read_text(encoding='utf-8'))
    current=json.loads((ROOT/'results/current_A.json').read_text(encoding='utf-8'))
    qa=json.loads((ROOT/'data/input_qa.json').read_text(encoding='utf-8'))
    verify=json.loads((ROOT/'results/verify_A.json').read_text(encoding='utf-8'))
    frozen=json.loads((ROOT/'freeze_A.json').read_text(encoding='utf-8'))
    primary=s['primary'];data=s['summaries']
    rows=[]
    for key in primary:
        r=data[key]['daily']['2026']
        rows.append([key,rate(r['on']),rate(r['match']),rate(r['off']),
                     rate(r['matched_on']),pct(r['delta_match']),num(s['rawp'][key]),num(s['holm'][key])])
    valid=[key for key in primary if data[key]['daily']['2026']['delta_match'] is not None]
    positive=[key for key in valid if data[key]['daily']['2026']['delta_match']>0]
    best=max(valid,key=lambda key:data[key]['daily']['2026']['delta_match']) if valid else None
    lead=f'2026년 가격 짝보다 바닥 근처 진입(Z) 비율이 높은 주 칸은 {len(positive)}/7이다.'
    if best:lead+=f' 가장 큰 차이는 {best}의 {pct(data[best]["daily"]["2026"]["delta_match"])}p다.'
    lead+=' 소수 하락 구간의 과거 비교이며 매매 규칙으로 확정하지 않는다.'
    lines=['# 수급·필터 결합으로 바닥 근처 고르기 — 과제 A',
        '', '**'+lead+'**','',
        table(['시험 범위','개수'],[['하락 단일','19'],['하락 결합','8'],['고점 보조','7'],['합계','34'],['판정 밖 X1 민감도','30%·70% 두 칸']]),'',
        '## 2026 주 검정 7칸','',
        'Z는 진입 시가 대비 진입 포함20세션 최저 종가가 −5% 이상인 비율이다. 이미5% 넘게 하락한 미완성 창은 실패에 포함한다. 나머지 미완성 창은 분모에서 뺀다. 가격 짝과 같은 수로 제한한 ON을 별도 열에 둔다.','',
        table(['신호','ON Z','가격 짝 Z','OFF Z','짝 있는 ON Z','짝 대비 차이','이동 p','Holm7'],rows),'',
        '원형 이동 p는 같은 해 전체 세션 신호를10,000회 이동한 단측 ON−OFF 차이의 참고값이다. 가격 짝의 p가 아니다. Holm은 이7칸만 보정한다. p 하나로 합격·탈락을 정하지 않는다.','',
        '## 근거와 해석','',
        '- 계열 대표는2024~2025 결과만으로 골랐다. 선택 때 가격 결과도2025-12-31에 잘랐다. 선택된 대표는 '+', '.join(f'{k}={v}' for k,v in s['representatives'].items())+'다.',
        '- ON과 OFF의 차이가 가격 짝에서 줄면, 직전 낙폭·5일 수익·1일 수익으로 설명되는 몫이 크다. 짝이 부족한 칸은 남은 ON 전체와 짝을 직접 비교하지 않는다.',
        '- 일별 창은 서로 겹친다. 아래10세션 간격 사건에서도 방향과 건수를 확인한다. 서로 다른 사건이 같은 하락 국면일 수 있다.',
        '- 2026은 앞선 연구에서 본 자료다. 이번 단계 분리와 freeze 커밋은 대표 선택의 추가 누출을 막지만 새 홀드아웃을 만들지는 않는다.','',
        '## 시기별 방향 — 대표와 K3','']
    rows=[]
    for key in primary:
        for period in ['2024','2025','2026H1','2026H2','2026','all']:
            r=data[key]['daily'][period]
            rows.append([key,period,rate(r['on']),rate(r['match']),rate(r['off']),pct(r['delta_match']),pct(r['on']['R20'])])
    lines += [table(['신호','기간','ON Z','가격 짝 Z','OFF Z','짝 차이','ON20일 수익'],rows),'',
        '2024·2025의 최종 표는 전체 새 가격판으로 계산했다. 선택 단계 말미의 미완성 창은2025 말에 잘랐으므로 freeze의 선택 통계와 일부 다를 수 있다.','',
        '## 반복 신호를 줄인 사건','']
    rows=[]
    for key in primary:
        r=data[key]['event']['2026']
        rows.append([key,rate(r['on']),rate(r['match']),rate(r['off']),pct(r['on']['R10']),pct(r['on']['R20']),pct(r['on']['min20'])])
    lines += [table(['신호','2026 ON Z','가격 짝 Z','OFF Z','10일 평균','20일 평균','20일 최저 평균'],rows),'',
        '## 바닥10개를 얼마나 잡았나','',
        '구역은 기존 B10 바닥 −10~+5세션, 시가≤바닥 종가×1.05이며 하락 상태에 속하는 날이다. 일별 재현율과10세션 사건 재현율을 나눈다. 정밀도는 구역 안 사건/모든 하락 사건이다.','']
    rows=[]
    for key in primary:
        r=bs[key]
        rows.append([key,f'{r["daily_caught"]}/10',f'{r["caught"]}/10',
                     f'{r["zone_events"]}/{r["event_n"]} ({pct(r["precision"])})',f'{r["match_caught"]}/10',
                     f'{r["match_zone_events"]}/{r["match_event_n"]} ({pct(r["match_precision"])})'])
    lines += [table(['신호','일별 재현','사건 재현','사건 정밀도','짝 재현','짝 정밀도'],rows),'',
        '첫 켜짐 날짜·바닥 대비 세션·가격 차이와27칸 전체는 [바닥별 적중표](results/bottom_hits_A.csv)와 [요약](results/bottom_summary_A.json)에 있다. 사건 간격 때문에 구역 안 일별 신호가 있어도 별도 사건을 못 잡을 수 있다.','']
    lag=pd.read_csv(ROOT/'results/publication_lag.csv')
    lines += ['바닥 당일 흐름을 알 수 있는 규약상 시점과 그때 이미 오른 폭이다. 실제 게시 로그가 아니다.','',
        table(['바닥','공개 후 관찰일','늦음(세션)','바닥 종가 대비 시가'],[[r.bottom,r.entry,r.delay,f'{r.gain_pct:+.2f}%'] for r in lag.itertuples()]),'',
        '## 매수가 말랐을 때인가, 많이 샀을 때인가','']
    e=pd.read_csv(ROOT/'results/exhaustion_bottoms.csv')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="470" viewBox="0 0 800 470" role="img" aria-label="바닥 날 5세션 SOXL 매수 합과 하락 중 최대 비교">',
         '<rect width="800" height="470" fill="#fff"/><g font-family="sans-serif" font-size="14" fill="#18242c">',
         '<text x="24" y="30" font-size="20">바닥 날 SOXL 5일 매수 / 하락 중 최대</text>',
         '<text x="24" y="54">50%는 고정한 매수 소진 기준 · 100%는 하락 중 최대</text>',
         '<line x1="450" y1="75" x2="450" y2="430" stroke="#a44" stroke-dasharray="4 4"/>']
    for i,r in enumerate(e.itertuples()):
        y=85+i*34
        svg += [f'<text x="24" y="{y+18}">{r.bottom}</text>',
                f'<rect x="170" y="{y}" width="{560*r.bottom_buy5_ratio:.1f}" height="23" fill="#286782"/>',
                f'<text x="{180+560*r.bottom_buy5_ratio:.1f}" y="{y+17}">{pct(r.bottom_buy5_ratio)}</text>']
    svg+=['</g></svg>']
    (ROOT/'results/exhaustion_A.svg').write_text('\n'.join(svg),encoding='utf-8')
    dry=int((e.bottom_buy5_ratio<=.5).sum());maximum=int(np.isclose(e.bottom_buy5_ratio,1).sum())
    pos=int((e.bottom_net>0).sum())
    lines += [f'바닥 당일5세션 매수 합이 그 하락 중 최대의 절반 이하인 경우는 **{dry}/10**, 최대였던 경우는 **{maximum}/10**이다. 바닥 당일 SOXL 순매수는 **{pos}/10**에서 양수였다. 바닥의5일 매수/최대 비율 중앙값은 **{pct(e.bottom_buy5_ratio.median())}**다. 이 표는 사후 설명이며 바닥 날의 금액을 그날 알았다는 뜻이 아니다.','',
        '![바닥 날 SOXL 5일 매수와 최대 비교](results/exhaustion_A.svg)','',
        table(['바닥','5일 매수 정점','정점→바닥 세션','바닥5일/최대','바닥하루/최대','바닥 순매수 USD'],
            [[r.bottom,r.buy5_peak,r.peak_to_bottom,pct(r.bottom_buy5_ratio),pct(r.bottom_day_buy_vs_max),f'{r.bottom_net:,.0f}'] for r in e.itertuples()]),'',
        '고점~바닥+10세션의 일별 SOXL 매수·순매수·누적 순매수·보관 좌수·전체 순매수·SOXS 몫은 [설명 원장](results/exhaustion_daily.csv)에 있다. 진행 중인10/05 하락도 포함한다. 하루 SOXX−3% 이하 날만 순서대로 비교한 순매수 변화는 [급락 반응 표](results/crash_response.csv)에 있다.','',
        '## 지금 상태','',
        f'최신 관찰 설정: **{current["date"]}**, 직전 종가 DD60 **{pct(current["dd60"])}**, 하락 상태 **{current["down"]}**. 원거래일 **{current["source_date"]}**, 보관 날짜 **{current["custody_date"]}**. 다음 정규장 시가가 아직 없으면 결과는 미채점이다. K는 **{current["K"]}**이고, 결측을 고려한 켜짐 수 범위는 **{current["K_min"]}~{current["K_max"]}**다.','']
    val=current['values'];rows=[]
    for key,obj in current['signals'].items():
        if key.startswith('S'):value=val.get(key[:2]);rank=val.get(key[:2]+'_P')
        elif key.startswith('T'):value=val.get('T5');rank=val.get('T5_P')
        elif key in ['CH','CL']:value=val.get('C');rank=val.get('C_P')
        elif key.startswith('X'):value=val.get(key+'_value');rank=None
        elif key=='G25':value=val.get('G');rank=None
        elif key.startswith('M'):value=val.get(key+'_value');rank=None
        else:value=obj['on'];rank=None
        state=obj['status']
        rows.append([key,num(value),num(rank),state])
    lines += [table(['신호','값','백분위','상태'],rows),'',
        f'뉴스 기존 전수 표본의 마지막 공개 날짜는 **{current["news_coverage_end"]}**다. 이후는 기사 없음과 구별해 UNKNOWN으로 둔다. 과제 B의 새 계열은 이번 고정판에 끼우지 않았으며 --news-series로 별도 실행할 수 있다.','',
        '## 언제 알 수 있었나','',
        table(['변수','사용 시점'],[
            ['SOXS·SOXL·Top50 결제','원결제표 다음 KSD 영업일09:00 KST 뒤 첫 NYSE 시가; 같은 공개일 최신 원거래일 우선'],
            ['보관·보유 좌수','보관 날짜 둘째 KSD 영업일09:00 KST 뒤 첫 NYSE; 마지막 공개 관측 유지'],
            ['CNN·이평선·DD60·직전수익','관찰 시작 직전 미국 세션 종가까지'],
            ['뉴스','published_at 다음 정규장부터5세션; 날짜만 있는 경우 그날 장중 사용 안 함'],
            ['바닥 날짜·바닥 대비 상승','기존 B10 사후 사건표, 설명·재현율 평가에만 사용']]),'',
        '## 전체 고정 칸과 보조 결과','']
    rows=[]
    for key in data:
        r=data[key]['daily']['2026']
        rows.append([key,rate(r['on']),rate(r['match']),pct(r['on']['Z3']),pct(r['on']['Z8']),pct(r['on']['R10']),pct(r['on']['R20']),pct(r['on']['min20'])])
    lines += [table(['19단일+8결합','2026 ON Z','짝 Z','−3% 지지','−8% 지지','10일 평균','20일 평균','최저종가 평균'],rows),'',
        '고점권7칸의 결과(+5% 넘게 더 오르지 않음·20일 수익)는 [고점 보조](results/high_aux_A.json), X1의30%·70%는 [민감도](results/sensitivity_A.json)다. 둘 다 대표 선택·주 판정에 쓰지 않는다. 6기간의 모든 일별·사건 통계는 [전체 요약](results/summary_A.json), 각 날짜는 [일별](results/daily_A.csv), [사건](results/events_A.csv), [가격 짝](results/matches_A.csv)에 있다.','',
        '## 자료·검증·한계','',
        f'- 새 SEIBRO 원응답 **{qa["requests"]}개**(빈 응답{qa["empty"]}개), 결제 날짜 {qa["first"]}~{qa["last"]}. 날짜별XML gzip을 보존했다. 새 단일 전체 가격 응답3개와 CNN 원응답을 보존했다. SOXX3:1 분할 기준은7/29·10/05·11/05 앵커로 검사했고 섞임이 없었다.',
        '- 빈 응답·Top50 미관측·새 공개 없는 날은0달러·OFF가 아니다. 분모는 Top50 근사이며 전체 한국인 자산/현금을 뜻하지 않는다. 매수 감소가 자금 고갈을 직접 측정하지는 않는다.',
        '- 같은 결제 묶음의 여러 미국 거래일은 균등 배분값이다. 당일 실제 거래액을 식별한 것이 아니다. 보관 좌수는보관금액/평가종가 추정이다. 보관 공시는 늦고 공개 없는 날은 마지막 잔고다.',
        '- 과거 CNN 값이 당시 게시값과 같은지는 확인할 수 없다. 결제·보관 게시 시각도 실측 로그가 아닌 규약이다. 뉴스 표본은 인용·분류와 수집 범위에 의존한다.',
        '- 가격 짝은 연도·상태를 맞춘 최근접3변수 대조다. 다른 교란 요인이나 특정 국면의 차이를 모두 없애지 못한다. greedy 순서와 OFF 부족이 결과에 영향을 줄 수 있다.',
        '- 시가→종가의 SOXX 관찰값이다. 수수료0원·주문 가능 KST08~24시 조건과 별개로 종가 청산, 스프레드·환전·배당·실제 체결을 적용한 전략 수익이 아니다.',
        f'- 독립 재계산: **{verify["comparisons"]:,}개 비교**, 불일치 **{len(verify["mismatches"])}개**, 통과 **{verify["passed"]}**. 주7칸의 Z·가격 짝·원형 이동·Holm과27칸×10바닥 적중표를 다른 코드 경로로 재계산했다. [verify_A.json](results/verify_A.json).',
        '- 단위 테스트는 미래 수급·가격·CNN 변조에도 과거19신호 불변,5일 창 결측,미완성 실패,삼값 결합,짝 중복·근접 제외,연도 경계 사건 간격을 확인한다. 전체 결과는 [unittest_A.log](unittest_A.log)에 있다.',
        '', '## 다음 후보','', '새 문턱·조합은 이번에 탐색하지 않았다. 과제 B의 공개시점이 확인된 뉴스 계열을 연결하는 별도 재실행만 남긴다. 기존 결과에 맞춰 대표를 다시 고르지 않는다.','']
    (ROOT/'report_A.md').write_text('\n'.join(lines),encoding='utf-8')
    # Compact CSV of the exact seven primary comparisons for the coordinator.
    out=[]
    for key in primary:
        r=data[key]['daily']['2026'];br=bs[key]
        out.append(dict(key=key,on_z=r['on']['rate'],on_n=r['on']['scored'],match_z=r['match']['rate'],match_n=r['match']['scored'],
            off_z=r['off']['rate'],off_n=r['off']['scored'],matched_on_z=r['matched_on']['rate'],holm=s['holm'][key],
            daily_recall=br['daily_caught']/10,event_recall=br['recall'],precision=br['precision']))
    pd.DataFrame(out).to_csv(ROOT/'results/primary_2026_A.csv',index=False)
    print('report written',len(lines))


if __name__=='__main__':main()
