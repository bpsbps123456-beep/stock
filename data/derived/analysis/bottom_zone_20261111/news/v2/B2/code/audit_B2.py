"""Evidence completeness checks, without comparing real/fake tone outcomes."""
import csv,json,re
from collect_B2 import NEWS,B2,save,csvwrite

def audit():
 vals=json.loads((B2/'coded_candidates_B2.json').read_text(encoding='utf-8'))
 windows=json.loads((B2/'windows.json').read_text(encoding='utf-8'))
 close=r'clos|finish|ends|end |ending|wrap|markets wrap|indexes fared|마감|3분증시'
 wire=r'reuters|ap news|apnews|associated press|연합뉴스|연합인포맥스|뉴시스'
 byday={}
 for r in vals:
  is_close=bool(r.get('wire_close_verified') or re.search(close,r['headline'],re.I) or (not re.search(r'\bopen\b|\bopens\b|futures|preview',r['headline'],re.I) and re.search(r'closed|closing|ended|finished|마감',r.get('snippet',''),re.I)))
  if r['decision']=='include' and re.search(wire,r['source'],re.I) and is_close:byday.setdefault(r['mapped_session'],[]).append(r)
 rows=[]
 for w in windows:
  hits=byday.get(w['session_date'],[])
  rows.append(dict(w,wire_close_count=len(hits),candidate_ids=';'.join(r['candidate_id'] for r in hits),status='future_unobservable' if w['status']=='outside_public_window' else 'found_headline' if hits else 'searched_not_found'))
 csvwrite(B2/'wire_coverage_B2.csv',rows)
 missing=sorted({r['session_date'] for r in rows if r['status']=='searched_not_found'})
 print('Missing close dates:',missing)
 turns=[r for r in vals if r['decision']=='include' and r.get('possible_turn')]
 anchor_days={w['session_date'] for w in windows if w['k'] in {0,1}}
 turns=[r for r in turns if r['mapped_session'] in anchor_days]
 save(B2/'turn_review_queue_B2.json',turns)
 for i,r in enumerate(turns,1):print('TURN',i,r['candidate_id'],r['published_date'],r['type'],r['tone'],r['headline'],r['url'])
 print('turn candidates',len(turns),'wire dates',len(byday))

if __name__=='__main__':audit()
