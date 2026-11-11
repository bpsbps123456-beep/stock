"""Assemble explicitly coded public changes, preserving original frozen inputs."""
import csv,hashlib,json,re
from collections import Counter
from datetime import datetime,timezone
from urllib.parse import urlparse
from collect_v2 import NEWS,V2,save,csvwrite
from event_coding_B1 import EVENTS
from web_supplements import inventory

def readcsv(p):
 return list(csv.DictReader(p.open(encoding='utf-8-sig')))

def main():
 if (NEWS/'freeze_B_v2.json').exists(): raise SystemExit('Frozen')
 queue=json.loads((V2/'body_queue_B1.json').read_text(encoding='utf-8'))
 byid={r['candidate_id']:i for i,r in enumerate(queue,1)}
 old=readcsv(NEWS/'calls_census.csv')
 replaced={e['replace_v1'] for e in EVENTS if e['replace_v1']}
 events=[dict(r,queue_ref=byid[r['candidate_id']],department='Research',horizon=(r['published_date'][:4]+' year-end') if r['asset_scope']=='index_target' else '',coding_notes=r['coding_notes']+'; v2 본문 검토 완료, 명시되지 않은 값은 빈칸 유지') for r in old if r['id'] not in replaced]
 events.extend(e for e in EVENTS if 'QUEUE'+str(e['queue_ref']) not in replaced)
 rows=[];seen=set();refs={}
 for e in events:
  q=queue[e['queue_ref']-1]; a=json.loads((V2/'raw/access_B1'/(q['candidate_id']+'.json')).read_text(encoding='utf-8'))
  key='|'.join(str(e.get(k,'')) for k in ['house','department','published_date','asset_scope','action','from_rating','to_rating','horizon'])
  if key in seen: continue
  seen.add(key); group='B1v2-'+hashlib.sha256(key.encode()).hexdigest()[:14]
  url=a.get('resolved_url',q['url']);host=urlparse(url).netloc
  readable=a['access'].startswith('retrieved_') and len(a.get('text',''))>150 and e['queue_ref'] not in {186,805,809,953,954}
  verification='제목·요약만'
  if readable:
   verification='재보도' if any(x in host for x in ['yahoo','itiger','futunn','kedglobal','etnews','medirect','investordaily','fortune','chosun','newsis','hankyung','yna','news1','mk.co','sedaily','finimize','webull','economictimes','interfax','sherwood']) else '원문 열람'
  if e['queue_ref'] in {940,944}:verification='원문 열람' # ordinary web tool rendered public article, archived web011/012
  oldread=next((r for r in old if r['candidate_id']==q['candidate_id'] and r['verification']=='원문 열람'),None)
  if oldread:verification='원문 열람'
  date=e['published_date']; notes=e.get('coding_notes','')
  if verification=='제목·요약만':notes+='; 본문 미열람: '+a['access']+' '+a.get('error','')
  row=dict(id='',house=e['house'],strategist=e.get('strategist',''),published_date=date,published_at='',time_precision='date',asset_scope=e['asset_scope'],action=e['action'],from_rating=e.get('from_rating',''),to_rating=e.get('to_rating',''),direction='+1' if e['action'] in {'upgrade','target_raise'} else '-1',after_selloff=e.get('after_selloff','N'),rationale_tag=e.get('rationale_tag','other'),quote='[요약] '+e['house']+' '+e['asset_scope']+' '+e['action']+': '+str(e.get('from_rating',''))+' → '+str(e.get('to_rating','')),url=url,source=host,verification=verification,source_group=group,candidate_id=q['candidate_id'],queue_ref=e['queue_ref'],department=e.get('department','Research'),horizon=e.get('horizon',''),coding_notes=notes,status='public_search_census_v2',headline=q['title'])
  rows.append(row);refs.setdefault(q['candidate_id'],[]).append(group)
 rows.sort(key=lambda r:(r['published_date'] or '9999',r['house'],r['asset_scope'],r['horizon']))
 for i,r in enumerate(rows,1):r['id']=f'B2C{i:04}'
 csvwrite(V2/'calls_census.csv',rows)
 family={'upgrade':'U','downgrade':'DN','target_raise':'TU','target_cut':'TC'}
 csvwrite(V2/'news_series_B.csv',[dict(date=r['published_date'],family=family[r['action']],asset_scope=r['asset_scope'],id=r['id'],house=r['house'],source_group=r['source_group']) for r in rows if r['published_date']])
 # All title and body dispositions stay distinct: queue inclusion is not final inclusion.
 review=readcsv(NEWS/'review_B1_v2.csv')+readcsv(V2/'review_extra_B1.csv')
 decisions={r['candidate_id']:r for r in review}
 body=[]
 for i,q in enumerate(queue,1):
  a=json.loads((V2/'raw/access_B1'/(q['candidate_id']+'.json')).read_text(encoding='utf-8'))
  selected=q['candidate_id'] in refs
  body.append(dict(queue_ref=i,candidate_id=q['candidate_id'],decision='include' if selected else 'exclude',reason_code='' if selected else 'not_a_change',note='직접 읽은 제목/열람 가능한 본문의 범위·기간·실제 변경 판정; 채택 사건은 source_groups 참조' if selected else '제목 후보였으나 열람 가능한 근거에서 별도의 신규 범위내 변경으로 확정하지 못함(재보도·유지·첫 목표·범위 밖 포함); 미열람은 access에 표시',access=a['access'],source_groups=';'.join(refs.get(q['candidate_id'],[])),title=q['title']))
  decisions[q['candidate_id']]=body[-1]
 csvwrite(V2/'review_B1_bodies.csv',body)
 web=inventory(); logs=[]
 for p in [NEWS/'search_log_B1.csv',V2/'search_log_B1.csv']:
  for r in readcsv(p):
   if r.get('candidate_id'):
    d=decisions.get(r['candidate_id']);r.update(decision='include' if r['candidate_id'] in refs else 'exclude',reason='' if r['candidate_id'] in refs else d.get('reason_code','not_a_change') if d else 'duplicate_or_prior_review',source_groups=';'.join(refs.get(r['candidate_id'],[])))
   if p.parent==NEWS:r['raw_file']='../'+r.get('raw_file','')
   logs.append(r)
 for r in web:
  selected=r['candidate_id'] in refs
  logs.append(dict(r,engine='web search/open',request_id=r['query_id'],decision='include' if selected else 'exclude',reason='' if selected else 'duplicate_or_no_verified_new_change',source_groups=';'.join(refs.get(r['candidate_id'],[]))))
 csvwrite(V2/'search_log_B1_complete.csv',logs)
 protocol=json.loads((NEWS/'protocol_B_v2.json').read_text(encoding='utf-8'))
 coverage=[]
 for h in protocol['houses']:
  for y in (2024,2025,2026):
   for k in range(1,5):
    calls=[r for r in rows if r['house']==h and r['published_date'] and r['published_date'][:4]==str(y) and (int(r['published_date'][5:7])-1)//3+1==k]
    coverage.append(dict(house=h,quarter=f'{y}Q{k}',calls=len(calls),read_full=sum(r['verification']!='제목·요약만' for r in calls),undated_not_assigned=sum(r['house']==h and not r['published_date'] for r in rows)))
 csvwrite(V2/'coverage_B1.csv',coverage)
 meta=dict(calls=len(rows),dated=sum(bool(r['published_date']) for r in rows),by_house=dict(Counter(r['house'] for r in rows)),by_action=dict(Counter(r['action'] for r in rows)),verification=dict(Counter(r['verification'] for r in rows)),title_reviews=len(review),body_reviews=len(queue),result_appearances=sum(bool(r.get('url')) for r in logs),unique_results=len({r.get('candidate_id') for r in logs if r.get('candidate_id')}),rss_planned=504+len(json.loads((V2/'queries_B1.json').read_text(encoding='utf-8'))),web_supplement_queries=sum(len(json.loads(p.read_text(encoding='utf-8')).get('queries',[])) for p in (V2/'raw').glob('web_B1_v2_*.json')),limitations=['전수는 고정 공개 검색의 모든 결과에 대한 판정; 비공개 기관 노트 전수가 아님','본문 차단 및 날짜 미확인 행 존재','최초 목표/유지와 신규 변경을 구별; 별도 신규 변경이 확정되지 않은 제목 후보는 미채택'])
 save(V2/'census_summary_B1.json',meta);print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
