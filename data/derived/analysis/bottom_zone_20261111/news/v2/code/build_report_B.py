"""Render the completed report directly from frozen evidence and analyses."""
import csv,json,subprocess
from collections import Counter
from pathlib import Path
from statistics import mean,median
from collect_v2 import NEWS,V2,save,csvwrite

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def pc(v):return '—' if v is None else f'{v*100:.1f}%'
def near(r):return f"{r['near_count']}/{r['complete20']} ({pc(r['near_fraction'])})"
def rt(r):return ' / '.join(pc(r['returns'][str(h)]['mean'])+f" (n={r['returns'][str(h)]['n']})" for h in [5,10,20])
def main():
 b1=read(V2/'results/analysis_B1.json');b2=read(V2/'results/analysis_B2.json');census=read(V2/'census_summary_B1.json')
 calls=rows(V2/'calls_census.csv');joined=rows(V2/'results/calls_with_prices.csv');turns=rows(V2/'B2/turn_catalysts_B2.csv')
 oldweb=list((NEWS/'raw').glob('web_B1_*.json'));newweb=[p for p in (V2/'raw').glob('web_B1_v2_*.json') if read(p).get('queries')]
 b1rss=list((NEWS/'raw/B1').glob('*.json'))+list((V2/'raw/B1').glob('*.json'))
 b2rss=list((V2/'raw/B2').glob('*.json'));b2web=[p for pat in ['web_raw_*.json','gap_raw_*.json','close_raw_*.json','final_raw_*.json'] for p in (V2/'B2').glob(pat)]
 counts=dict(B1_RSS_attempts=sum(1+len(read(p).get('prior_attempts',[])) for p in b1rss),B1_web_queries=sum(len(read(p)['queries']) for p in oldweb+newweb),B1_web_requests=len(oldweb+newweb),B2_RSS_attempts=sum(1+len(read(p).get('prior_attempts',[])) for p in b2rss),B2_web_queries=sum(len(read(p)['requests']) for p in b2web)+4,B2_web_requests=len(b2web)+1)
 counts['B1_total_queries']=counts['B1_RSS_attempts']+counts['B1_web_queries'];counts['B2_total_queries']=counts['B2_RSS_attempts']+counts['B2_web_queries']
 counts['total_queries']=counts['B1_total_queries']+counts['B2_total_queries'];counts['total_requests']=counts['B1_RSS_attempts']+counts['B1_web_requests']+counts['B2_RSS_attempts']+counts['B2_web_requests']
 save(V2/'results/search_counts_B.json',counts)
 L=[]
 def add(s=''):L.append(s)
 def table(headers,data):
  add('| '+' | '.join(headers)+' |');add('| '+' | '.join(['---']*len(headers))+' |')
  for r in data:add('| '+' | '.join(str(x).replace('|','/') for x in r)+' |')
  add()
 add('2026 표본에서 하락 중 상향은 바닥 근처를 더 잘 가리키지 못했고, 하향 집중 뒤의 개선도 표본 3건이라 확인되지 않았다. 진짜 바닥 뒤에도 악재가 남았지만 부정 기사 비율은 가짜 바닥보다 낮았다.');add()
 table(['2026 주 결과','사건 / 완전 관측','바닥 근처','같은 하락 상태 아무 날','가격 짝','Holm p'],[
 ['H-U: 하락 중 상향','4 / 2','0/2','27/66 (40.9%)','0/2','1.000'],
 ['H-D: 하향·목표 하향 집중','5 / 3','2/3','27/66 (40.9%)','1/3','0.442'],
 ['진짜 바닥 +1~+5 부정 기사','4개 바닥','51/160 (31.9%)','가짜 47/133 (35.3%)','기사 비율; 예측 적중률 아님','해당 없음']]);
 add('H-U의 대조 대비 차이는 −40.9%p, H-D는 +25.8%p다. H-D의 가격 짝 대비 차이는 +33.3%p다. 20세션이 끝나지 않은 사건을 성공으로 세지 않았다. 진짜 바닥 10개 중 9개는 이후 5세션에도 부정 기사가 한 건 이상 있었다. 그러나 “가짜보다 부정 비율이 더 높다”는 추측은 이 자료에서 지지되지 않는다.');add()
 add('**자료와 완료 범위**');add()
 table(['항목','완료 값'],[
 ['공개일','2024-01-01~2026-11-10; 2026 주 판정, 2024·2025 병렬 확인'],
 ['B1 최종','24기관 181행; 공개일 확인 178행, 일자 미확인 3행은 가격 분석 제외'],
 ['B1 검색','912개 고정 RSS 질의 + 재시도 2 + 웹 검색어 59 = '+str(counts['B1_total_queries'])+'회; HTTP 검색 요청 938회'],
 ['B1 후보·검토','결과 출현 7,221 / 고유 6,265; 별도 제목 판정 4,200+1,713=5,913; 본문 검토 큐 954'],
 ['B1 본문 확인','원문 80/181=44.2%; 통신사 재보도 50/181=27.6%; 합계 130/181=71.8%; 제목·요약 51'],
 ['B2 검색',f"고정 RSS 2,172 + 재시도 1 + 웹 검색어 {counts['B2_web_queries']} = {counts['B2_total_queries']}회; HTTP 검색 요청 {counts['B2_RSS_attempts']+counts['B2_web_requests']}회"],
 ['B2 범위','진짜 10·완료 국면 첫 가짜 6·현재 1; 187세션 조합 중 185 관측 가능, 미래 2 미관측'],
 ['B2 후보·채택',f"검색 후보 10,138; 수동 판정 1,458; 채택 URL 후보 1,211; 기사군 1,119; 바닥 창 1,138행"],
 ['B2 본문 확인','원문 5/1,211=0.41%; 재보도 9/1,211=0.74%; 합계 14/1,211=1.16%; 나머지는 제목·검색 요약'],
 ['전체 검색',f"검색어 실행 {counts['total_queries']}회, HTTP 검색 요청 {counts['total_requests']}회. 기사 열기·본문 요청은 별도이며 검색 횟수에 넣지 않음."]]);
 add('“전수”는 사전 등록한 공개 검색에서 나온 결과를 판정했다는 뜻이다. 비공개 기관 노트나 검색엔진에 잡히지 않은 모든 의견을 확보했다는 뜻은 아니다. B2는 같은 매체 틀로 수집한 뉴스 목록이며 세계 전체 뉴스의 모집단이 아니다. B2 원문 열람률이 매우 낮아 세부 논조 해석에는 제한이 크다.');add()
 table(['기관','행'],sorted(census['by_house'].items(),key=lambda x:(-x[1],x[0])))
 table(['종류','행'],[['상향 U',41],['하향 DN',44],['목표 상향 TU',81],['목표 하향 TC',15]])
 add('최종 B1 정본은 [v2/calls_census.csv](news/v2/calls_census.csv), [검색 기록](news/v2/search_log_B1_complete.csv), [기관×분기 coverage](news/v2/coverage_B1.csv), [과제 A 전달 계열](news/v2/news_series_B.csv)이다. 루트의 동명 B1 파일 55행과 freeze_B.json은 첫 부분 동결의 감사 이력으로 보존했다. 공개일 미확인 3행은 전달 계열에서도 제외했다. B2 정본은 [bottom_news.csv](news/bottom_news.csv), [검색 기록](news/search_log_B2.csv)이다.');add()
 add('**B1 정의·시점·시험 범위**');add()
 add('공개일만 알면 다음 정규장 시가 진입이다. 시각이 확인된 경우만 뉴욕 09:30 이전 공개에 같은 날 시가를 허용한다. 실제 최종 표본은 날짜 단위다. 상태·DD60·직전 5세션 수익은 모두 진입 직전 종가까지로 계산한다. 하락 상태는 직전 종가/직전 60세션 최고 종가−1≤−10%다. 주 결과는 진입일부터 20개 종가의 최저값/진입 시가−1≥−5%다.');add()
 add('4종류×2상태=연도별 8칸, 3년=24칸을 SOXX와 보조 QQQ에 각각 냈다. 주 검정은 SOXX의 2026 H-U·H-D 두 개뿐이다. H-D는 서로 다른 2기관 이상이 5세션 안 하향·목표 하향을 내면 첫 인지일에 진입하고 다음 4세션은 중복 신호를 막는다. 같은 기관의 서로 다른 범위 변경을 2기관으로 세지 않는다.');add()
 add('가격 짝은 같은 해·상태·20세션 관측 가능한 날 중 표준화 DD60와 직전 5세션 수익의 제곱 거리 최소 한 날이다. 본인 날짜는 제외하고 짝 재사용은 허용한다. 대조는 사후 성과 비교이며 실시간 거래 규칙 검증은 아니다. 원형 이동은 2026 관측 216세션 벡터를 10,000번 이동했다(시드 20261111). H-U 유효 10,000, H-D 유효 9,533; 비어 있는 이동 표본은 제외했다. 고유 이동은 최대 216개뿐이다.');add()
 table(['연도','가설','하락 상태 사건','바닥 근처','가격 짝','5 / 10 / 20세션 평균 수익'],[[r['year'],r['hypothesis'],r['signals']['n'],near(r['signals']),near(r['matched']),rt(r['signals'])] for r in b1['SOXX']['hypotheses']])
 add('2024 H-U 2행은 같은 날 나온 Citi의 서로 다른 범위 변경이다. 독립된 두 번의 바닥 예측으로 해석하지 않는다. 2026 H-U 진입일은 7/14·7/22·11/2·11/9이며 앞 두 건만 20세션 완료다. H-D는 3/13·4/1·9/17·10/20·11/2이며 앞 세 건만 완료다. 11/10 공개 의견은 다음 정규장 시가도 아직 없어 진입 결과를 계산하지 않았다.');add()
 table(['연도','종류','하락 상태','사건 수','바닥 근처','아무 날','가격 짝','5 / 10 / 20 수익'],[[r['year'],{'upgrade':'U','downgrade':'DN','target_raise':'TU','target_cut':'TC'}[r['action']],'안' if r['decline_state'] else '밖',r['signals']['n'],near(r['signals']),near(r['all_days']),near(r['matched']),rt(r['signals'])] for r in b1['SOXX']['grid']])
 add('QQQ는 SOXX로 정한 같은 하락 상태에서 결과만 보조로 비교했다. 2026 H-U 0/2, H-D 2/3; QQQ 아무 날 52/66(78.8%), 가격 짝은 각각 0/2·2/3이다. Holm p는 둘 다 1.000이다. 종목·업종·세계 주식 의견을 SOXX 하나로 재는 노출 차이가 남는다. QQQ 전체 24칸은 analysis_B1.json에 저장했다.');add()
 add('**실제 바닥과의 사후 거리**');add()
 dist=[]
 for y in [2024,2025,2026]:
  for action in ['upgrade','downgrade','target_raise','target_cut']:
   rr=[r for r in joined if r['published_date'].startswith(str(y)) and r['action']==action and r.get('bottom_signed_sessions')!='']
   ds=[int(r['bottom_signed_sessions']) for r in rr];window=[x for x in ds if abs(x)<=10];after=[x for x in ds if x>0]
   dist.append([y,{'upgrade':'U','downgrade':'DN','target_raise':'TU','target_cut':'TC'}[action],len(ds),sum(x<0 for x in window),sum(x==0 for x in window),sum(x>0 for x in window),median(after) if after else '—'])
 table(['연도','종류','날짜 있는 행','바닥 −10~−1','바닥 당일','바닥 +1~+10','바닥 후 전체 행의 중앙 세션'],dist)
 add('각 의견을 가장 가까운 B10 날짜에 사후 대응한 거리다. 실제 당시 바닥을 알았다는 뜻이 아니다. ±10세션 밖도 마지막 중앙값 계산에는 포함했다. 현재 진행 중 11/5 저점은 B10으로 소급 추가하지 않았다. [의견별 거리·가격 결과](news/v2/results/calls_with_prices.csv)에서 모든 행을 확인할 수 있다.');add()
 add('**B2 부정 비율·유형 분포**');add()
 table(['연도','바닥','구간','부정/전체','부정 비율','바닥별 비율 동일 가중'],[[r['year'],r['anchor_type'],{'pre':'−5~−1','zero':'0','post':'+1~+5'}[r['period']],f"{r['neg']}/{r['n']}",pc(r['negative_fraction']),pc(r['episode_equal_negative_fraction'])] for r in b2['grid']])
 add('전체 진짜 +1~+5는 72/274=26.3%, 가짜는 72/184=39.1%다. 2026은 31.9% 대 35.3%다. 바닥별 기사량을 똑같이 가중해도 2026 진짜 29.4%, 가짜 39.6%로 같은 방향이다. AP·Reuters 마감 기사만 남긴 보조 집계도 전체 진짜 10/77=13.0%, 가짜 16/37=43.2%, 2026 진짜 6/29=20.7%, 가짜 9/18=50.0%다. 이는 기사 구성의 차이이지 시장 상승 확률이 아니다.');add()
 table(['바닥','진짜/가짜','이후 부정/전체','전환 재료','0/+1 보도 요약'],[[r['anchor_date'],r['anchor_type'],next(f"{e['neg']}/{e['n']}" for e in b2['episodes'] if e['anchor_date']==r['anchor_date'] and e['period']=='post'),r['catalyst'],r['note']] for r in turns])
 table(['유형','진짜 전체 전/당일/후','가짜 전체 전/당일/후'],[[t,' / '.join(str(next(r for r in b2['grid'] if r['year']=='all' and r['anchor_type']=='real' and r['period']==p)['types'].get(t,0)) for p in ['pre','zero','post']),' / '.join(str(next(r for r in b2['grid'] if r['year']=='all' and r['anchor_type']=='fake' and r['period']==p)['types'].get(t,0)) for p in ['pre','zero','post'])] for t in ['macro_policy','earnings_fundamental','flows_positioning','strategist_call','regulation_geopolitics','capitulation_language','technical','other']])
 add('전환 재료는 바닥 당일 또는 +1 기사에서 확인한 주된 긍정 재료를 국면마다 하나만 정했다. 진짜 10개는 정책 4·실적 1·수급 1·전략가 0·확인 못함 4개다. 가짜 6개는 정책 2·실적 0·수급 1·전략가 0·확인 못함 3개다. 정책에는 거시 지표 안도와 전쟁 해결 기대를 포함한다. 기대 보도가 실제 정책 시행·종전 확인을 뜻하지 않는다. “확인 못함”은 실제 재료가 전혀 없었다는 뜻이 아니다.');add()
 add('근거 예: [Citi 2024-08-07 자산배분 원문](https://www.citigroup.com/rcs/citigpa/storage/public/GIC-AA-August-07-2024.pdf)은 조정 뒤 미국 주식 비중을 높인 범위 변경이다. [BlackRock 2026-04-13 원문](https://www.blackrock.com/us/individual/literature/market-commentary/weekly-investment-commentary-en-us-20260413-back-to-overweight-us-stocks.pdf)도 가치·이익 전망을 근거로 미국 주식 상향을 보도했다. B2는 [2025-04-09 AP 관세 유예 보도](https://apnews.com/article/ca404c694f8c2421d5433813f1da8921), [2026-07-30 연합뉴스 실적 반등](https://www.yna.co.kr/view/AKR20260730195200009), [2026-11-06 뉴시스 저가 매수 보도](https://www.newsis.com/view/NISX20261106_0003858701)처럼 서로 다른 재료가 있었다. AP 링크의 원문 접속 실패는 검색 요약 판정으로 남겼다.');add()
 add('**지금: 판정에서 제외**');add()
 add('새로 받은 SOXX에서 2026-10-05 이후 최저 종가일은 11/5다. 잠정 11/2를 대체했다. Yahoo의 11/5 3:1 분할을 확인했고 조정 종가는 11/4 150.1367 → 11/5 131.9000 → 11/6 136.5300이다. 이 구간 최대 일변동 12.15%로 기계적인 −66.7% 분할 오류가 아니다. 현재 저점은 진짜·가짜 비교에 넣지 않았다.');add()
 recentcalls=[r for r in calls if r['published_date']>='2026-10-05']
 table(['공개일','기관','범위','종류','변경','검증'],[[r['published_date'],r['house'],r['asset_scope'],r['action'],(r['from_rating'] or '종전 등급 미확인')+' → '+(r['to_rating'] or '새 등급 명칭 미확인'),r['verification']] for r in recentcalls])
 table(['최근 세션','기사','부정','긍정','중립','주요 유형'],[[r['date'],r['n'],r['neg'],r['pos'],r['neutral'],', '.join(f'{k}:{v}' for k,v in sorted(r['types'].items(),key=lambda x:-x[1])[:3])] for r in b2['current10']])
 add('최근 10세션은 10/28~11/10이다. 11/5 AI 관련 급락 뒤 11/6 저가 매수·반도체 반등이 보도됐다. 이후 부정 보도도 남아 있다. 현재 바닥 창의 이후 구간은 +1~+3만 관측됐고 +4~+5는 미래다. 이 프로필을 바닥 확률이나 매수 추천으로 바꾸지 않는다.');add()
 add('**검증·동결·한계**');add()
 add('B1 v2 동결 커밋은 `'+b1['freeze_commit']+'`, B2 동결 커밋은 `'+b2['freeze_commit']+'`이다. 각각 해당 가격·논조 비교 전에 SHA-256 명세를 커밋했고 분석 시작과 테스트에서 해시를 다시 확인했다. 첫 부분 동결 `dd65093c`도 보존했다. [B1 동결 명세](news/freeze_B_v2.json), [B2 동결 명세](news/freeze_B2_v2.json), [가격 원본·분할 확인](news/v2/prices/fetch_metadata.json)을 참조한다.');add()
 add('검색·부호화 규칙은 먼저 등록했다. B2는 시드 20261111로 날짜와 후보 순서를 섞었으나 진짜·가짜 라벨을 가리지 못했다. 수집 중 통신사 단신 날짜 누락·매체 식별·마감 기사 필터를 수정했고 비교 전 evidence_QA_B2.json에 기록했다. 고정 검색 외 누락일 보완 질의도 모두 남겼다. 첫 4질의 응답 한 번은 저장을 빠뜨려 같은 질의를 재실행했다. 최초 응답의 완전한 로컬 보관은 없고 두 시도 모두 검색 횟수에 포함했다.');add()
 add('검색 결과·본문 차단, RSS 날짜 대용, 업데이트 기사 날짜, 본문을 읽은 일부만의 선택, 같은 사건의 다른 제목 재보도 의존성이 남는다. 날짜만 있는 뉴스는 세션 내 실제 선후관계를 증명하지 못한다. 부정 비율은 판정 가능한 제목·요약의 기사 수를 분모로 썼다. 내용 없는 마감 단신을 중립으로 간주하지 않았다. 검색이 잡아낸 부정 보도의 존재를 실제 악재 발생 건수와 같다고 볼 수 없다.');add()
 add('신호 수가 작고 뒤쪽 20세션 검열이 크다. H-U·H-D 수치는 연도 내 전방 검증으로 선택된 예측 모델이 아니며 이번 전수 표본의 사후 대조다. 분기·기관·범위를 넓게 탐색한 선택편향이 있어 새 확률 규칙으로 채택하지 않는다. 2026 자료를 모두 사용했으므로 추가 확인은 앞으로 쌓이는 종이 기록이 필요하다.');add()
 test=V2/'results/unittest_full_B.txt'
 text=test.read_text(encoding='utf-8') if test.exists() else ''
 import re
 match=re.search(r'Ran (\d+) tests in ([\d.]+)s',text)
 add('전체 검증: `py -m unittest` → '+(f"{match[1]} tests / {match[2]}s / OK (exit 0)" if match and 'exit_code: 0' in text else '실행 결과 확인 중')+'. [전체 출력](news/v2/results/unittest_full_B.txt). 새 테스트는 정보 시점·20세션 검열·다른 기관 조건·분할·동결 불변·뉴스 창 완전성·전환 재료 조건을 검사한다.');add()
 add('최종 분석·보고서 커밋은 이 보고서를 포함한 `research: 바닥 뉴스 전략가 의견 전수 표본과 바닥 창 대조 (codex)`이며 정확한 해시는 최종 전달 메시지에 기록한다. 허용 경로만 커밋했고 과제 A 및 다른 작업자의 변경은 건드리지 않았다.');
 (NEWS.parent/'report_B.md').write_text('\n'.join(L)+'\n',encoding='utf-8')
 print(json.dumps(counts));print('report lines',len(L))
if __name__=='__main__':main()
