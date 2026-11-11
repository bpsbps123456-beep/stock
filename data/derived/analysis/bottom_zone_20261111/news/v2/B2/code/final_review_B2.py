"""Final source/date evidence QA before the B2 freeze; no outcome comparison."""
import json,re,sys
from collect_B2 import B2,save
from code_B2 import infer,RULES

def queue():
 vals=json.loads((B2/'all_candidates_B2.json').read_text(encoding='utf-8'))
 over=json.loads((B2/'manual_overrides_B2.json').read_text(encoding='utf-8'))
 rows=[]
 for r in vals:
  d=infer(r)
  if r['candidate_id'] not in over and d['decision']=='include' and re.search(RULES['primary_sources'],r['source'],re.I):rows.append(dict(r,**d))
 save(B2/'final_review_queue_B2.json',rows)
 for i,r in enumerate(rows,1):print(i,r['candidate_id'],r['published_date'],r['source'],r['headline'],r.get('snippet','')[:1000])
 print('TOTAL',len(rows))
if __name__=='__main__':queue()
