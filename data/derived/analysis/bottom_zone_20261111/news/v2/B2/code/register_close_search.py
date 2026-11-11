import csv,json
from datetime import datetime,timezone
from collect_B2 import B2,save
rows=list(csv.DictReader((B2/'wire_coverage_B2.csv').open(encoding='utf-8-sig')))
days=sorted({r['session_date'] for r in rows if r['status']=='searched_not_found'})
queries=[dict(id=f'B2CLOSE-{i:03}',session_date=d,query='"How major US stock indexes fared" '+datetime.fromisoformat(d).strftime('%A')+' '+str(int(d[5:7]))+'/'+str(int(d[8:10]))+'/'+d[:4]+' Associated Press') for i,d in enumerate(days,1)]
save(B2/'close_queries_B2.json',dict(registered_at=datetime.now(timezone.utc).isoformat(),queries=queries,reason='Exact dated AP closing brief for every remaining wire gap; registered before calls and before comparing tones'))
print(json.dumps(queries))
