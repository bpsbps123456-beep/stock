"""Print bounded relevant body passages for manual evidence review; no price access."""
import json,re,sys
from collect_v2 import V2
def show(a,b):
 q=json.loads((V2/'body_queue_B1.json').read_text(encoding='utf-8'))
 for i in range(a,min(b,len(q))+1):
  x=q[i-1];p=V2/'raw/access_B1'/(x['candidate_id']+'.json')
  r=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
  s=r.get('text',''); patterns=r'(?:upgrad|downgrad|rais\w*|lift\w*|cut\w*|lower\w*|상향|하향|낮췄|높였|조정)[^.!?]{0,180}(?:target|S&P|equiti|stocks|tech|반도체|목표|의견)|(?:target|overweight|underweight|neutral|비중|목표)[^.!?]{0,180}(?:from|to |에서|으로|에서)'
  matches=list(re.finditer(patterns,s,re.I))
  spans=[]
  for m in matches:
   lo,hi=max(0,m.start()-90),min(len(s),m.end()+180)
   if spans and lo<=spans[-1][1]:spans[-1]=(spans[-1][0],max(hi,spans[-1][1]))
   else:spans.append((lo,hi))
  passage=' ... '.join(s[lo:hi] for lo,hi in spans)
  if not passage:passage=s[:1100]
  print(f'\n{i} {x.get("published_date","")} {x["title"]}\n{r.get("access","not_fetched")} {r.get("resolved_url",x["url"])}\n{passage[:950]}')
if __name__=='__main__':show(int(sys.argv[1]),int(sys.argv[2]))
