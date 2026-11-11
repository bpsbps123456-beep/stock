"""Fixed date-template collection. No subsequent price outcomes are read."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
from email.utils import parsedate_to_datetime
import argparse,hashlib,json,random,sys
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import exchange_calendars as xcals
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
from collect_v2 import NEWS,V2,save,csvwrite
B2=V2/'B2';RAW=V2/'raw/B2'

def register():
 protocol=json.loads((NEWS/'protocol_B_v2.json').read_text(encoding='utf-8'))
 dates=protocol['search_B2']['anchors'];anchors=[]
 for typ,key in [('real','real'),('fake','fake_completed')]:
  anchors.extend(dict(anchor_date=d,anchor_type=typ) for d in dates[key])
 # B1 price audit, after its committed freeze, established this current low.
 anchors.append(dict(anchor_date='2026-11-05',anchor_type='current'))
 random.Random(20261111).shuffle(anchors)
 calendar=xcals.get_calendar('XNYS',start='2024-01-01',end='2026-12-01')
 sessions=calendar.sessions.tz_localize(None);windows=[]
 for order,a in enumerate(anchors,1):
  i=sessions.get_loc(pd.Timestamp(a['anchor_date']))
  for k in range(-5,6):
   d=sessions[i+k].date().isoformat()
   windows.append(dict(a,k=k,session_date=d,coding_order=order,status='within_public_window' if d<='2026-11-10' else 'outside_public_window'))
 recent=['2026-10-28','2026-10-29','2026-10-30','2026-11-02','2026-11-03','2026-11-04','2026-11-05','2026-11-06','2026-11-09','2026-11-10']
 needed=sorted({r['session_date'] for r in windows if r['status']=='within_public_window'}|set(recent))
 amendment=dict(version='B2 evidence 2',registered_at=datetime.now(timezone.utc).isoformat(),parent='../../protocol_B_v2.json',current_anchor='2026-11-05',current_anchor_reason='B1 freeze c5edc8af followed by fresh SOXX price audit; provisional 11/02 superseded; no B2 tone/eligibility based on returns',recent10=recent,templates=protocol['search_B2']['templates'],coding=protocol['coding_B2'],date_note='RSS UTC timestamp converted to America/New_York for candidate date, flagged feed-only; source publication day supersedes if verified; non-session maps next XNYS session; current future dates >11/10 unobservable',wire_requirement='Reuters/AP close per window session; missing explicitly reported, never treated as no adverse news')
 save(B2/'protocol_B2_amendment.json',amendment);save(B2/'windows.json',windows)
 queries=[]
 for d in needed:
  end=(datetime.fromisoformat(d)+timedelta(days=1)).date().isoformat()
  for n,t in enumerate(amendment['templates'],1):
   queries.append(dict(id=f'B2-{len(queries)+1:04}',session_date=d,template=n,query=t.format(date=d)+f' after:{d} before:{end}',ko=n>=8))
 save(B2/'queries_B2.json',queries);print('windows',len(windows),'unique days',len(needed),'queries',len(queries))

def fetch(q):
 path=RAW/(q['id']+'.json')
 if path.exists():
  old=json.loads(path.read_text(encoding='utf-8'))
  if not old.get('error'):return old
 else:old=None
 out=dict(q,fetched_at=datetime.now(timezone.utc).isoformat(),results=[])
 if old:out['prior_attempts']=old.get('prior_attempts',[])+[{k:v for k,v in old.items() if k!='prior_attempts'}]
 try:
  ko=q['ko'];r=requests.get('https://news.google.com/rss/search',params=dict(q=q['query'],hl='ko' if ko else 'en-US',gl='KR' if ko else 'US',ceid='KR:ko' if ko else 'US:en'),timeout=40)
  out['request_url']=r.url;r.raise_for_status()
  for item in ET.fromstring(r.content).findall('./channel/item'):
   stamp=item.findtext('pubDate','');url=item.findtext('link','');source=item.find('source')
   dt=parsedate_to_datetime(stamp).astimezone(__import__('zoneinfo').ZoneInfo('America/New_York')) if stamp else None
   out['results'].append(dict(candidate_id=hashlib.sha256(url.encode()).hexdigest()[:16],headline=item.findtext('title',''),published_date=dt.date().isoformat() if dt else '',feed_timestamp=stamp,url=url,source=source.text if source is not None else '',source_home=source.get('url','') if source is not None else '',query_session_date=q['session_date']))
  out['truncated']=len(out['results'])>=100
 except Exception as e:out['error']=f'{type(e).__name__}: {e}'
 save(path,out);return out

def collect(wire_only=False):
 if (NEWS/'freeze_B2_v2.json').exists():raise SystemExit('B2 frozen')
 q=json.loads((B2/'queries_B2.json').read_text(encoding='utf-8'))
 if wire_only:q=[r for r in q if r['template']>=11]
 with ThreadPoolExecutor(max_workers=8) as pool:
  for i,r in enumerate(pool.map(fetch,q),1):
   if i%50==0 or r.get('error') or r.get('truncated'):print(i,len(q),len(r['results']),r.get('error',''),flush=True)
 inventory()

def inventory():
 rows=[];candidates={}
 for path in sorted(RAW.glob('B2-*.json')):
  q=json.loads(path.read_text(encoding='utf-8'));common={k:v for k,v in q.items() if k not in {'results','prior_attempts'}};common['query_id']=common.pop('id')
  for rank,r in enumerate(q['results'],1):
   rows.append(dict(common,rank=rank,**r,decision='pending_review'))
   candidate=candidates.setdefault(r['candidate_id'],dict(r,query_dates=[],queries=[]))
   candidate['query_dates']=sorted(set(candidate['query_dates']+[q['session_date']]));candidate['queries'].append(q['id'])
  if not q['results']:rows.append(dict(common,decision='search_error' if q.get('error') else 'zero_results'))
 csvwrite(NEWS/'search_log_B2.csv',rows)
 vals=list(candidates.values());random.Random(20261111).shuffle(vals)
 save(B2/'candidates_B2.json',vals);print('appearances',sum(bool(r.get('url')) for r in rows),'unique',len(vals),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['register','collect','inventory','wire']);a=p.parse_args();collect(True) if a.command=='wire' else globals()[a.command]()
