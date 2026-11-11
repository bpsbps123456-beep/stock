"""Keep every web result; explicitly selected sources join the body queue."""
import argparse,hashlib,json,re
from collect_v2 import V2,save,csvwrite

def inventory():
 rows=[]
 for p in sorted((V2/'raw').glob('web_B1_v2_*.json')):
  d=json.loads(p.read_text(encoding='utf-8'));s=d['result']
  matches=list(re.finditer(r'(?m)^([^\n]+) \((https?://[^\s]+)\)\s*$',s))
  for rank,m in enumerate(matches,1):
   detail=s[m.end():matches[rank].start() if rank<len(matches) else len(s)]
   rows.append(dict(web_ref=f'{p.stem[-3:]}-{rank:02}',candidate_id=hashlib.sha256(m[2].encode()).hexdigest()[:16],title=m[1],url=m[2],source=m[2].split('/')[2],published_date='',publication_label=(re.search(r'Published: ([^;\n]+)',detail) or ['',''])[1],query_id=p.stem,query=json.dumps(d['queries'],ensure_ascii=False),raw_file=p.relative_to(V2).as_posix(),snippet=detail))
 save(V2/'web_inventory_B1.json',rows)
 return rows

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['display','select']);p.add_argument('refs',nargs='*');a=p.parse_args();rows=inventory()
 if a.command=='display':
  for r in rows:print(r['web_ref'],r['title'],r['url'])
 else:
  selected=set(a.refs);q=json.loads((V2/'body_queue_B1.json').read_text(encoding='utf-8'));ids={r['candidate_id'] for r in q}
  for r in rows:
   if r['web_ref'] in selected and r['candidate_id'] not in ids:q.append(r);ids.add(r['candidate_id'])
  save(V2/'body_queue_B1.json',q);save(V2/'web_selected_B1.json',list(selected));print('queue',len(q))
