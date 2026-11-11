"""Uniform close-wire coverage supplement, registered before B2 coding."""
import json,sys
from datetime import datetime,timedelta,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from collect_B2 import B2,save
p=B2/'queries_B2.json';rows=json.loads(p.read_text(encoding='utf-8'))
days=sorted({r['session_date'] for r in rows})
existing={r['query'] for r in rows}
templates={11:'Wall Street stocks Nasdaq close (Reuters OR AP)',12:'뉴욕증시 마감'}
for d in days:
 end=(datetime.fromisoformat(d)+timedelta(days=1)).date().isoformat()
 for n,t in templates.items():
  query=t+f' after:{d} before:{end}'
  if query not in existing:
   rows.append(dict(id=f'B2-{len(rows)+1:04}',session_date=d,template=n,query=query,ko=n==12));existing.add(query)
save(p,rows)
a=B2/'protocol_B2_amendment.json';obj=json.loads(a.read_text(encoding='utf-8'));obj['uniform_wire_supplement']=dict(registered_at=datetime.now(timezone.utc).isoformat(),templates=templates,days=len(days),reason='ISO date literal produced sparse fixed-template results; uniform bounds-only wire/close supplement for EVERY date, before tone coding; no selective adverse/positive query')
save(a,obj);print('queries',len(rows))
