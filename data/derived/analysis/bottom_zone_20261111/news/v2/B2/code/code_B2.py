"""Auditable headline coding. Deterministic rules, followed by manual overrides.

This is explicitly a headline/search-summary study: no inaccessible body is
represented as read. Every search hit retains its rule and disposition.
"""
import argparse,csv,hashlib,json,random,re,sys
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import exchange_calendars as xcals
sys.path.insert(0,str(Path(__file__).resolve().parent))
from collect_B2 import NEWS,V2,B2,RAW,save,csvwrite,inventory

RULES={
 'primary_sources':r'reuters|ap news|apnews\.com|associated press|bloomberg|cnbc|financial times|wall street journal|marketwatch|연합뉴스|연합인포맥스|뉴시스|federal reserve|federalreserve\.gov',
 'exclude_advice':r'\b(?:stocks? to buy|best stocks|top \d+ stocks|buy this|should you buy|millionaire|millionaires|millionaire-maker|is .* a buy|stock forecast|price prediction|technical analysis|stock price today|share price today|historical data|live chart|how to invest|dividend stocks|stock picks|options trading)\b|매수할 종목|추천주|주가 전망|종목 추천|주식 추천|고수익|종목픽|주가 예상|주가예측',
 'exclude_non_equity':r'bitcoin|cryptocurrency|dogecoin|ethereum|\bcrypto\b|비트코인|가상화폐|가상자산|암호화폐',
 'relevance':r'wall street|nasdaq|s&p|dow jones|dow |u\.s\. stocks|us stocks|u\.s\. equit|us equit|stock market|stocks (?:rise|fall|gain|slide|rally|tumble|drop|close|end|rebound|sink|slip|surge|jump)|equities|semiconductor|chip stocks|chip shares|chipmakers|nvidia|뉴욕증시|뉴욕 증시|나스닥|엔비디아|반도체|미국증시|미국 증시|미 증시|연준|federal reserve|\bfed\b',
 'neg':r'\b(?:fall|falls|fell|falling|slide|slides|slid|sliding|tumble|tumbles|tumbled|drop|drops|dropped|sink|sinks|sank|slip|slips|slipped|plunge|plunges|plunged|selloff|sell-off|loss|losses|losing|lower|slump|slumps|crash|crashes|bear market|panic|recession|downgrade|downgrades|restrict|restriction|curbs|ban|bans|tariff|tariffs|hike|hikes|inflation|war|miss|misses|weak|weakness|disappoint|warning|fears|worry|worries|concerns)\b|급락|폭락|하락|약세|투매|공포|침체|우려|경고|충격|매도|불안|부진|관세|규제|악재|하향|인상|긴축|하회|전쟁|붕괴',
 'pos':r'\b(?:rise|rises|rose|rising|rally|rallies|rallied|gain|gains|gained|higher|rebound|rebounds|bounce|bounces|surge|surges|jump|jumps|jumped|advance|advances|record high|records|relief|truce|pause|easing|eases|ease|upgrade|upgrades|beat|beats|strong|strength|buying|optimism|optimistic|soar|soars|recover|recovery|hopes)\b|급등|폭등|상승|강세|반등|회복|완화|휴전|유예|상향|호재|훈풍|안도|기대|낙관|최고치|사상 최고|매수세|상회',
 'types':{
  'regulation_geopolitics':r'export|restriction|curbs|ban\b|bans\b|regulat|china|chinese|iran|israel|war\b|mideast|middle east|수출|규제|중국|이란|이스라엘|전쟁|중동',
  'macro_policy':r'\bfed\b|federal reserve|interest|rates|yield|inflation|cpi|ppi|payroll|jobs|employment|economy|economic|gdp|tariff|trade (?:deal|war|truce)|trump|treasury|연준|금리|물가|고용|실업|관세|무역|트럼프|경기|경제|국채',
  'earnings_fundamental':r'earning|profit|revenue|sales|forecast|outlook|guidance|ai |artificial intelligence|demand|nvidia|chip|semiconductor|실적|이익|매출|엔비디아|반도체|수요|인공지능',
  'flows_positioning':r'flows|buying|hedge funds|retail|investors buy|positioning|자금|수급|개인|헤지펀드|매수세',
  'strategist_call':r'strategist|upgrade|downgrade|rating|target|goldman|morgan stanley|citigroup|\bciti\b|jpmorgan|\bubs\b|bofa|전략가|투자의견|목표치|목표주가|상향|하향|골드만|모건스탠리|씨티',
  'capitulation_language':r'capitulation|panic|bear market|correction|crash|투매|패닉|공포|약세장|붕괴|폭락',
  'technical':r'wall street|nasdaq|s&p|dow |stock|equities|뉴욕|나스닥|증시|주가|주식'
 },
 'price_direction':{
  'neg':r'\b(?:fall|falls|fell|slide|slides|slid|tumble|tumbles|tumbled|drop|drops|dropped|sink|sinks|sank|slip|slips|slipped|plunge|plunges|selloff|sell-off|losses|lower|slump|crash)\b|급락|폭락|하락|약세|투매',
  'pos':r'\b(?:rise|rises|rose|rally|rallies|rallied|gain|gains|gained|higher|rebound|rebounds|bounce|bounces|surge|surges|jump|jumps|jumped|advance|advances|record high|soar|soars)\b|급등|폭등|상승|강세|반등|최고치|사상 최고'
 },
 'catalyst_link':r'\b(?:as|on|after|amid|due to|driven by|thanks to|fueled by)\b|힘입|덕분|영향|따라|소식|안도|기대|발표|유예|완화|휴전',
 'pb_exclusion':r'prime broker|prime brokerage|PB 자금|PB 노트',
 'irrelevant_domestic':r'호남 건설|건설주|부동산|코스피|코스닥|국내 증시|오늘의 증시|오늘의 투자전략|서울 증시|달러.?원|원.?달러|환율|증권가|한국 증시|국내증시|서학픽|중학개미',
 'non_event_commentary':r'\b(?:could double|could triple|could soar|could be|here.s why|here are .* stocks|things to know|stocks to watch|top .* analyst calls|top wall street analyst|how .* stock|better buy|buy the .*dip|chartstopper|generated .* returns|how well|what .* means for|what to know|is .* worth|is .* time to|can .* stock|will .* stock)\b|마감브리핑|프리미엄콘텐츠|데이터랩',
 'general_individual_calls':r'\b(?:upgraded|downgraded|price target)\b|목표주가',
 'mixed':r'mixed|혼조|혼재',
 'note':'Rules fixed before B2 classification. Lead price direction overrides subordinate concern/hope language; simultaneous opposite price directions -> neutral. Otherwise explicit opposing beneficial/adverse material -> neutral. Manual overrides logged separately. Pure semiconductor/Nvidia price reporting without cause -> technical, not an earnings catalyst.'
}

def prepare():
 if (NEWS/'freeze_B2_v2.json').exists():raise SystemExit('Frozen')
 save(B2/'headline_rules_B2.json',dict(registered_at=datetime.now(timezone.utc).isoformat(),rules=RULES))
 inventory();rss=json.loads((B2/'candidates_B2.json').read_text(encoding='utf-8'));byid={r['candidate_id']:r for r in rss};web=[]
 for p in sorted(list(B2.glob('web_raw_*.json'))+list(B2.glob('gap_raw_*.json'))+list(B2.glob('close_raw_*.json'))+list(B2.glob('final_raw_*.json'))):
  d=json.loads(p.read_text(encoding='utf-8'));s=d['result'];matches=list(re.finditer(r'(?m)^([^\n]+) \((https?://[^\s]+)\)\s*$',s))
  for rank,m in enumerate(matches,1):
   detail=s[m.end():matches[rank].start() if rank<len(matches) else len(s)]
   row=dict(candidate_id=hashlib.sha256(m[2].encode()).hexdigest()[:16],headline=m[1],url=m[2],source=m[2].split('/')[2],query_dates=[r['session_date'] for r in d['requests']],queries=[r['id'] for r in d['requests']],published_date='',feed_timestamp='',snippet=detail,raw_file=p.relative_to(NEWS).as_posix(),query_batch=json.dumps(d['requests'],ensure_ascii=False),rank=rank)
   if re.search(r'By Reuters|\(Reuters\)|Thomson Reuters|ByImage: Reuters',m[1]+' '+detail,re.I):row['source']='Reuters (재보도: '+row['source']+')'
   elif re.search(r'(?:By The |The | - )Associated Press',detail,re.I):row['source']='Associated Press (재보도: '+row['source']+')'
   # Exact ISO dates in URL or explicit article timestamp only, never relative "last year".
   iso=re.search(r'(20(?:24|25|26))[-/](\d{2})[-/](\d{2})',m[2])
   if iso:row['published_date']='-'.join(iso.groups())
   # Index-close briefs explicitly identify the session in their title.
   headline_date=re.search(r'\b(\d{1,2})/(\d{1,2})/(20(?:24|25|26))\b',row['headline'])
   if not row['published_date'] and headline_date:
    month,day,year=headline_date.groups()
    row['published_date']=f'{year}-{int(month):02}-{int(day):02}'
    row['date_basis']='통신사 지수 마감 기사 제목의 명시 세션 날짜'
   dated=re.search(r'(?:Published|Updated)\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}),?\s+(20(?:24|25|26))',detail,re.I)
   if not row['published_date'] and dated:
    month,day,year=dated.groups();monthnum=datetime.strptime(month[:3].title(),'%b').month
    row['published_date']=f'{year}-{monthnum:02}-{int(day):02}'
    row['date_basis']='검색 결과 본문에 표시된 Published/Updated 날짜; 최초 공개시각 미확인'
   if not row['published_date']:
    date=re.search(r'Date: (?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) (\w{3})\s+(\d{1,2}) .*? (20\d{2}) UTC',detail)
    if date:
     try:row['published_date']=datetime.strptime(' '.join(date.groups()),'%b %d %Y').date().isoformat()
     except ValueError:pass
   if not row['published_date']:
    md=re.search(r'\b(\d{1,2})/(\d{1,2})/(20(?:24|25|26))\b',detail)
    if md:
     mo,dy,yr=md.groups();row['published_date']=f'{yr}-{int(mo):02}-{int(dy):02}';row['date_basis']='검색 결과에 표시된 명시 기사 날짜; 시각 미확인'
   # Retain the richest retrieved description, not an earlier generic RSS title.
   old=byid.get(row['candidate_id'])
   if old is None or (len(detail)>len(old.get('snippet','')) and row['published_date']):byid[row['candidate_id']]=row
   web.append(row)
 save(B2/'web_inventory_B2.json',web)
 vals=list(byid.values());random.Random(20261111).shuffle(vals)
 save(B2/'all_candidates_B2.json',vals);print('RSS',len(rss),'web appearances',len(web),'unique total',len(vals))

def infer(r):
 title=r['headline'];note=[]
 if not r.get('published_date'):return dict(decision='exclude',reason='publication_day_unverified')
 if r['published_date']<'2024-01-01' or r['published_date']>'2026-11-10':return dict(decision='exclude',reason='out_of_public_window')
 if re.search(r'How major US stock indexes fared',title,re.I):return dict(decision='include',reason='dated_wire_close_brief',type='technical',tone='neutral',possible_turn=False)
 if re.search(RULES['exclude_advice'],title,re.I):return dict(decision='exclude',reason='advice_or_evergreen_not_event_news')
 if re.search(RULES['non_event_commentary'],title,re.I):return dict(decision='exclude',reason='commentary_preview_or_generic_advice')
 if re.search(RULES['irrelevant_domestic'],title,re.I) and not re.search(r'뉴욕증시|뉴욕 증시|나스닥.*마감|미국 증시.*마감|미 증시.*마감',title):return dict(decision='exclude',reason='domestic_prices_or_forex_main_subject')
 if re.search(RULES['general_individual_calls'],title,re.I) and not re.search(r'nvidia|semiconductor|chip|s&p|nasdaq (?:index|composite|100)|u\.s\. (?:stocks|equities)|us (?:stocks|equities)|global equities|엔비디아|반도체|미국 주식|세계 주식',title,re.I):return dict(decision='exclude',reason='individual_non_chip_call')
 if re.search(RULES['exclude_non_equity'],title,re.I) and not re.search(r'nasdaq|s&p|wall street|뉴욕|나스닥|엔비디아|nvidia|주식|stocks',title,re.I):return dict(decision='exclude',reason='other_asset')
 if re.search(RULES['pb_exclusion'],title,re.I):return dict(decision='exclude',reason='pb_flow_note')
 scope_title=re.sub(r'\bNASDAQ:[A-Z]+\b','',title,flags=re.I)
 if not re.search(RULES['relevance'],scope_title,re.I):return dict(decision='exclude',reason='no_explicit_US_equity_chip_or_macro_link')
 # Broad geopolitics alone is insufficient unless its headline links equity/chips.
 haspos=bool(re.search(RULES['pos'],title,re.I));hasneg=bool(re.search(RULES['neg'],title,re.I))
 lead=re.split(r'\b(?:as|after|amid|on|despite|ahead of)\b',title,maxsplit=1,flags=re.I)[0]
 lead=re.sub(r'(?:국채금리|국채 금리|금리|유가|수익률|물가|달러)[^,·…]{0,12}?(?:상승|하락|급등|급락)','',lead)
 dirpos=bool(re.search(RULES['price_direction']['pos'],lead,re.I));dirneg=bool(re.search(RULES['price_direction']['neg'],lead,re.I))
 tone='neutral'
 if re.search(RULES['mixed'],title,re.I) or (dirpos and dirneg):tone='neutral'
 elif dirpos:tone='pos'
 elif dirneg:tone='neg'
 elif haspos!=hasneg:tone='pos' if haspos else 'neg'
 if re.search(r'(?:하락|급락|폭락|약세).{0,12}(?:후|하루 만에|뒤).*?(?:상승|반등)',title):tone='pos'
 typ=next((t for t,pat in RULES['types'].items() if re.search(pat,title,re.I)),'other')
 # Attribution must name causal fundamental material, not just the chip ticker.
 if typ=='earnings_fundamental' and not re.search(r'earning|profit|revenue|sales|forecast|guidance|demand|outlook|실적|이익|매출|수요|가이던스',title,re.I):typ='technical'
 return dict(decision='include',reason='explicit_headline_relevance',type=typ,tone=tone,possible_turn=bool(tone=='pos' and re.search(RULES['catalyst_link'],title,re.I) and typ not in {'technical','other','capitulation_language'}))

def classify():
 vals=json.loads((B2/'all_candidates_B2.json').read_text(encoding='utf-8'))
 overridepath=B2/'manual_overrides_B2.json';overrides=json.loads(overridepath.read_text(encoding='utf-8')) if overridepath.exists() else {}
 calendar=xcals.get_calendar('XNYS',start='2024-01-01',end='2026-12-01');sessions=calendar.sessions.tz_localize(None)
 recent=json.loads((B2/'protocol_B2_amendment.json').read_text(encoding='utf-8'))['recent10']
 windows=json.loads((B2/'windows.json').read_text(encoding='utf-8'));windowlookup={}
 for w in windows:windowlookup.setdefault(w['session_date'],[]).append(w)
 rows=[];reviews=[];seen=set();adopted={}
 for i,r in enumerate(vals,1):
  d=infer(r)
  if d['decision']=='include' and not re.search(RULES['primary_sources'],r['source'],re.I):d.update(decision='exclude',reason='outside_uniform_primary_source_frame')
  d.update(overrides.get(r['candidate_id'],{}));out=dict(r,**d,coding_order=i,coding_method='fixed_headline_rules+explicit_manual_override' if r['candidate_id'] in overrides else 'fixed_headline_rules')
  date=d.get('published_date',r.get('published_date',''));out['published_date']=date
  if d['decision']=='include' and (not date or date<'2024-01-01' or date>'2026-11-10'):
   d.update(decision='exclude',reason='out_of_public_window_after_date_verification');out.update(d)
  if d['decision']=='include':
   j=int(sessions.searchsorted(pd.Timestamp(date)));session=str(sessions[j].date())
   out['mapped_session']=session
   if session not in windowlookup and session not in recent:
    out.update(decision='exclude',reason='outside_all_anchor_windows_and_recent10')
   else:
    h=re.sub(r'\s+-\s+'+re.escape(r['source'])+r'$','',r['headline'],flags=re.I)
    h=re.sub(r'\s*(?:\| Kitco News|By Reuters|– ABC NEWS10)\s*$','',h,flags=re.I)
    h=re.sub(r'[^\w]+',' ',h.lower()).strip()
    if re.search(r'how major us stock indexes fared',h):h='ap major us stock indexes fared '+date
    group='N-'+hashlib.sha256((date+'|'+h).encode()).hexdigest()[:16]
    out['source_group']=group
    for w in windowlookup.get(session,[]):
     key=(w['anchor_date'],group)
     if key in seen:continue
     seen.add(key)
     turn='Y' if w['k'] in {0,1} and d.get('possible_turn') and d['tone']=='pos' and d['type'] not in {'technical','other','capitulation_language'} else 'N'
     rows.append(dict(id='',anchor_date=w['anchor_date'],anchor_type=w['anchor_type'],published_date=date,published_at='',time_precision='date',k=w['k'],headline=r['headline'],type=d['type'],tone=d['tone'],is_turn_catalyst=turn,url=r['url'],source=r['source'],verification=d.get('verification','제목·요약만'),source_group=group,candidate_id=r['candidate_id'],date_basis=d.get('date_basis',r.get('date_basis','RSS 뉴욕 변환 날짜 대용; 원문 공개일 미확인' if r.get('feed_timestamp') else '공개 URL/검색결과의 명시 날짜')),coding_method=out['coding_method'],coding_notes=d.get('note','')))
     adopted.setdefault(r['candidate_id'],[]).append(group)
    # Keep recent profile in review even when not in the anchor window.
  reviews.append(out)
 rows.sort(key=lambda r:(next(w['coding_order'] for w in windows if w['anchor_date']==r['anchor_date']),r['published_date'],r['source_group']))
 for i,r in enumerate(rows,1):r['id']=f'BN{i:05}'
 csvwrite(NEWS/'bottom_news.csv',rows)
 (B2/'coded_candidates_B2.json').write_text(json.dumps([{k:v for k,v in r.items() if k not in {'query_batch','source_home'}} for r in reviews],ensure_ascii=False,separators=(',',':')),encoding='utf-8')
 csvwrite(B2/'review_B2.csv',[{k:v for k,v in r.items() if k not in {'snippet','query_batch'}} for r in reviews])
 logs=[r for r in csv.DictReader((NEWS/'search_log_B2.csv').open(encoding='utf-8-sig')) if r.get('engine')!='web search']
 decisions={r['candidate_id']:r for r in reviews}
 for r in logs:
  if r.get('candidate_id') in decisions:r.update({k:decisions[r['candidate_id']].get(k,'') for k in ['decision','reason','type','tone','source_group','coding_method']})
 for r in json.loads((B2/'web_inventory_B2.json').read_text(encoding='utf-8')):
  d=decisions[r['candidate_id']];logs.append(dict(r,query=r['query_batch'],query_id=';'.join(r['queries']),engine='web search',decision=d['decision'],reason=d['reason'],type=d.get('type',''),tone=d.get('tone',''),source_group=d.get('source_group','')))
 # Full snippets and request URLs remain in raw archives. Avoid duplicating them
 # in every audit row; every query, headline, date, URL and disposition is kept.
 logs=[{k:v for k,v in r.items() if k not in {'snippet','query_batch','request_url','source_home','query_dates','queries','query_session_date','fetched_at'}} for r in logs]
 csvwrite(NEWS/'search_log_B2.csv',logs)
 save(B2/'coding_summary_B2.json',dict(candidates=len(vals),manual_overrides=len(overrides),included_candidates=sum(r['decision']=='include' for r in reviews),rows=len(rows),unique_groups=len({r['source_group'] for r in rows}),coding_method='All 1056 initial eligible primary-source-frame headlines manually reviewed in fixed seeded shuffled order; other results excluded by registered relevance/source rules; bodies read only in the separately logged subset'))
 print('candidates',len(vals),'included',sum(r['decision']=='include' for r in reviews),'rows',len(rows),'manual',len(overrides))

def display(start,end,mode):
 vals=json.loads((B2/'coded_candidates_B2.json').read_text(encoding='utf-8'))
 if mode=='anchor':
  windows=json.loads((B2/'windows.json').read_text(encoding='utf-8'));days={w['session_date'] for w in windows if w['k'] in {0,1}}
  vals=[r for r in vals if r.get('mapped_session') in days and r['decision']=='include']
 elif mode=='wire':vals=[r for r in vals if r['decision']=='include' and re.search(r'reuters|\bap\b|associated press',r['source'],re.I)]
 elif mode=='mixed':vals=[r for r in vals if r['decision']=='include' and r.get('tone')=='neutral']
 elif mode=='included':vals=[r for r in vals if r['decision']=='include']
 for i,r in enumerate(vals[start-1:end],start):print(i,r['candidate_id'],r['published_date'],r['decision'],r.get('type',''),r.get('tone',''),r['headline'])
 print('TOTAL',mode,len(vals))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','classify','display']);p.add_argument('start',nargs='?',type=int,default=1);p.add_argument('end',nargs='?',type=int,default=100);p.add_argument('--mode',default='all');a=p.parse_args()
 prepare() if a.command=='prepare' else classify() if a.command=='classify' else display(a.start,a.end,a.mode)
