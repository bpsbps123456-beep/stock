"""Extend the registered grid for a repeatedly encountered institution."""
import json,sys
from collect_v2 import NEWS,V2,save
h,ko,reason=sys.argv[1:4]
p=json.loads((NEWS/'protocol_B_v2.json').read_text(encoding='utf-8'))
if (NEWS/'freeze_B_v2.json').exists():raise SystemExit('frozen')
p['houses'][h]=ko
if not any(x['house']==h for x in p['additional_houses']):p['additional_houses'].append({'house':h,'reason':reason})
save(NEWS/'protocol_B_v2.json',p)
qs=json.loads((V2/'queries_B1.json').read_text(encoding='utf-8'))
if not any(x['house']==h for x in qs):
 for y in (2024,2025,2026):
  for q in range(1,5):
   a=f'{y}-{q*3-2:02d}-01';b=min(f'{y+1}-01-01' if q==4 else f'{y}-{q*3+1:02d}-01','2026-11-11')
   for template,t in p['search_B1']['templates'].items():
    qs.append(dict(id=f'B1V2-{len(qs)+1:04d}',house=h,alias=h,quarter=f'{y}Q{q}',template=template,start=a,end=b,query=t.format(house=h,house_ko=ko)+f' after:{a} before:{b}'))
save(V2/'queries_B1.json',qs)
print(len(qs))
