"""Stable shuffled review queue and explicit human batch decisions."""
import argparse,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from collect_B2 import B2,save

def init():
 p=B2/'manual_review_queue_B2.json'
 if p.exists():raise SystemExit('Already initialized')
 vals=json.loads((B2/'coded_candidates_B2.json').read_text(encoding='utf-8'))
 save(p,[r for r in vals if r['decision']=='include'])

def display(a,b):
 vals=json.loads((B2/'manual_review_queue_B2.json').read_text(encoding='utf-8'))
 for i,r in enumerate(vals[a-1:b],a):print(i,r['candidate_id'],r['published_date'],r.get('type',''),r.get('tone',''),r['headline'])
 print('TOTAL',len(vals))

def confirm(a,b,spec):
 vals=json.loads((B2/'manual_review_queue_B2.json').read_text(encoding='utf-8'));p=B2/'manual_overrides_B2.json'
 overrides=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
 changes={}
 for x in spec.split(';'):
  if not x:continue
  i,t,*other=x.split(':');changes[int(i)]=(t,other)
 for i in range(a,b+1):
  r=vals[i-1];d={k:r[k] for k in ['decision','reason','type','tone','possible_turn'] if k in r}
  d.update(note=f'제목 직접 검토: seeded shuffled queue {i}; 본문 미열람은 제목·요약만 표시',manual_review_index=i)
  if i in changes:
   t,other=changes[i]
   if t=='exclude':d.update(decision='exclude',reason=other[0] if other else 'not_event_or_scope')
   else:
    d.update(tone=t)
    if other and other[0]:d['type']=other[0]
    if len(other)>1:d['possible_turn']=other[1]=='Y'
  overrides[r['candidate_id']]=d
 save(p,overrides);print('reviewed',len(overrides))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['init','display','confirm']);p.add_argument('start',nargs='?',type=int,default=1);p.add_argument('end',nargs='?',type=int,default=100);p.add_argument('--changes',default='');a=p.parse_args()
 init() if a.command=='init' else display(a.start,a.end) if a.command=='display' else confirm(a.start,a.end,a.changes)
