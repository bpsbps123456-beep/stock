"""Add repeatedly encountered BNP Paribas, before any price linkage."""
from collect_v2 import NEWS,V2,save
import json
p=json.loads((NEWS/'protocol_B_v2.json').read_text(encoding='utf-8'))
p['houses']['BNP Paribas']='BNP파리바'
if not any(x['house']=='BNP Paribas' for x in p['additional_houses']):
 p['additional_houses'].append({'house':'BNP Paribas','reason':'본문 검토에서 미국·세계 주식의 복수 의견 변경이 반복 발견됨; 가격 연결 전 동일 12분기×3틀 추가'})
save(NEWS/'protocol_B_v2.json',p)
qs=json.loads((V2/'queries_B1.json').read_text(encoding='utf-8'))
if not any(x['house']=='BNP Paribas' for x in qs):
 for y in (2024,2025,2026):
  for q in range(1,5):
   start=f'{y}-{q*3-2:02d}-01';end=min(f'{y+1}-01-01' if q==4 else f'{y}-{q*3+1:02d}-01','2026-11-11')
   for template,t in p['search_B1']['templates'].items():
    qs.append(dict(id=f'B1V2-{len(qs)+1:04d}',house='BNP Paribas',alias='BNP Paribas',quarter=f'{y}Q{q}',template=template,start=start,end=end,query=t.format(house='BNP Paribas',house_ko='BNP파리바')+f' after:{start} before:{end}'))
save(V2/'queries_B1.json',qs)
print(len(qs))
