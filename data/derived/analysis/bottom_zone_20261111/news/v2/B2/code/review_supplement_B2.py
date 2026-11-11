"""Manual second queue, including source/date parser repairs and wire briefs."""
import argparse,json,re
from collect_B2 import B2,save
def confirm(a,b,spec,queue='supplement_review_queue_B2.json'):
 vals=json.loads((B2/queue).read_text(encoding='utf-8'));p=B2/'manual_overrides_B2.json';over=json.loads(p.read_text(encoding='utf-8'));changes={}
 for item in spec.split(';'):
  if item:
   i,t,*other=item.split(':');changes[int(i)]=(t,other)
 for i in range(a,b+1):
  r=vals[i-1];d={k:r[k] for k in ['decision','reason','type','tone','possible_turn'] if k in r}
  d.update(note=f'직접 검토: {queue} {i}; 읽은 범위만 verification 표시',supplemental_review_index=i)
  # An uninformative index-brief title is not evidence of neutral sentiment.
  if 'indexes fared' in r['headline'] and not r.get('snippet'):d.update(decision='exclude',reason='tone_unavailable_generic_headline')
  if i in changes:
   tone,other=changes[i]
   if tone=='exclude':d.update(decision='exclude',reason=other[0] if other else 'out_of_scope')
   else:
    d.update(decision='include',tone=tone)
    if other and other[0]:d['type']=other[0]
    if len(other)>1:d['possible_turn']=other[1]=='Y'
  d['possible_turn']=bool(d.get('possible_turn') and d.get('tone')=='pos')
  over[r['candidate_id']]=d
 save(p,over);print('manual decisions',len(over))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('a',type=int);p.add_argument('b',type=int);p.add_argument('--changes',default='');p.add_argument('--queue',default='supplement_review_queue_B2.json');x=p.parse_args();confirm(x.a,x.b,x.changes,x.queue)
