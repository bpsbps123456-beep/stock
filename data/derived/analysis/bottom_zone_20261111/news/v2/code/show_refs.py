import json,sys,re
from collect_v2 import V2
q=json.loads((V2/'body_queue_B1.json').read_text(encoding='utf-8'))
for n in map(int,sys.argv[1:]):
 r=json.loads((V2/'raw/access_B1'/(q[n-1]['candidate_id']+'.json')).read_text(encoding='utf-8'))
 print('\n',n,q[n-1]['title'],r.get('metadata'),r.get('resolved_url'))
 text=r.get('text','')
 matches=list(re.finditer(r'upgrad|downgrad|overweight|underweight|target|neutral|상향|하향|중립|비중',text,re.I))
 intervals=[]
 for m in matches:
  a,b=max(0,m.start()-130),min(len(text),m.end()+260)
  if intervals and a<=intervals[-1][1]:intervals[-1]=(intervals[-1][0],max(b,intervals[-1][1]))
  else:intervals.append((a,b))
 print((' ... '.join(text[a:b] for a,b in intervals) or text)[:5000])
