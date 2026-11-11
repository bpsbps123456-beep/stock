"""Public article access only; never bypass a login or permission prompt."""
import argparse,json,re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import requests
from bs4 import BeautifulSoup
from googlenewsdecoder import gnewsdecoder
from collect_B2 import B2,NEWS,save,csvwrite

def get(r):
 p=B2/'body'/f"{r['candidate_id']}.json"
 if p.exists():return json.loads(p.read_text(encoding='utf-8'))
 out=dict(candidate_id=r['candidate_id'],headline=r['headline'],url=r['url'],retrieved_at=datetime.now(timezone.utc).isoformat(),access='unread',text='')
 try:
  url=r['url']
  if 'news.google.com/' in url:
   decoded=gnewsdecoder(url,timeout=15);out['resolver']=decoded
   if not decoded.get('success'):raise ValueError(decoded.get('message','unresolved public redirect'))
   url=decoded['decoded_url']
  out['resolved_url']=url
  response=requests.get(url,timeout=25,headers={'User-Agent':'Mozilla/5.0 (compatible; ResearchEvidence/1.0)'})
  out['http_status']=response.status_code
  if response.status_code!=200:out['access']=f'unread_http_{response.status_code}'
  else:
   soup=BeautifulSoup(response.content,'html.parser')
   out['metadata']=[t.get('content','') for t in soup.select('meta[property="article:published_time"],meta[name="date"],meta[itemprop="datePublished"]')]
   out['jsonld']=[t.get_text()[:15000] for t in soup.select('script[type="application/ld+json"]')]
   for t in soup(['script','style','noscript','nav','header','footer','form']):t.decompose()
   body=soup.find('article') or soup.find('main') or soup.body or soup
   out['text']=body.get_text(' ',strip=True)[:60000]
   out['access']='retrieved_pending_read' if len(out['text'])>150 and not re.search(r'Access Denied|Checking your browser|Just a moment|Enable JavaScript|Please enable JS',out['text'][:500],re.I) else 'unread_block_or_empty'
 except Exception as e:out['error']=f'{type(e).__name__}: {e}'
 save(p,out);return out

def collect():
 if (NEWS/'freeze_B2_v2.json').exists():raise SystemExit('Frozen')
 supplemental=json.loads((B2/'supplement_review_queue_B2.json').read_text(encoding='utf-8'))
 turns=json.loads((B2/'turn_review_queue_B2.json').read_text(encoding='utf-8'))
 candidates=json.loads((B2/'coded_candidates_B2.json').read_text(encoding='utf-8'))
 windows=json.loads((B2/'windows.json').read_text(encoding='utf-8'))
 days={w['session_date'] for w in windows if w['k'] in {0,1}}
 chosen={r['candidate_id']:r for r in supplemental+turns}
 for d in sorted(days):
  wires=[r for r in candidates if r['decision']=='include' and r.get('mapped_session')==d and re.search(r'reuters|ap news|associated press|연합뉴스',r['source'],re.I) and re.search(r'close|end |ends|마감|indexes fared',r['headline'],re.I)]
  for r in wires[:2]:chosen[r['candidate_id']]=r
 save(B2/'body_queue_B2.json',list(chosen.values()))
 with ThreadPoolExecutor(max_workers=8) as pool:
  for i,r in enumerate(pool.map(get,chosen.values()),1):
   if i%25==0:print(i,len(chosen),r['access'],flush=True)
 csvwrite(B2/'article_access_B2.csv',[{k:v for k,v in json.loads(p.read_text(encoding='utf-8')).items() if k not in {'text','jsonld','resolver'}} for p in sorted((B2/'body').glob('*.json'))])

def display(a,b):
 rows=json.loads((B2/'supplement_review_queue_B2.json').read_text(encoding='utf-8'))
 for i,r in enumerate(rows[a-1:b],a):
  p=B2/'body'/f"{r['candidate_id']}.json";d=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
  text=d.get('text','');pos=text.find('NEW YORK');text=text[max(0,pos):] if pos>=0 else text
  print(i,r['candidate_id'],r['published_date'],r['type'],r['tone'],r['headline'],'\n',d.get('access'),d.get('metadata'),d.get('resolved_url'),'\n',(text[:2100] if d.get('access')=='retrieved_pending_read' else r.get('snippet','')[:900]).replace('\n',' '))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['collect','display']);p.add_argument('start',nargs='?',type=int,default=1);p.add_argument('end',nargs='?',type=int,default=30);a=p.parse_args();collect() if a.command=='collect' else display(a.start,a.end)
