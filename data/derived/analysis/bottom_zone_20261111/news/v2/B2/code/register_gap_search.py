"""Uniform extra search for each unfilled close-session, before outcome counts."""
import csv,json
from datetime import datetime,timezone
from collect_B2 import B2,save
rows=list(csv.DictReader((B2/'wire_coverage_B2.csv').open(encoding='utf-8-sig')))
days=sorted({r['session_date'] for r in rows if r['status']=='searched_not_found'})
queries=[dict(id=f'B2GAP-{i:03}',session_date=d,query=datetime.fromisoformat(d).strftime('%B %d %Y')+' Wall Street stocks close Reuters Associated Press') for i,d in enumerate(days,1)]
save(B2/'gap_queries_B2.json',dict(registered_at=datetime.now(timezone.utc).isoformat(),reason='Same close search for every wire-coverage gap, before comparing tones; ordinary syndications permitted if wire byline verified',queries=queries,prior_unarchived_request=dict(queries=queries[:4],reason='First request response accidentally not persisted; exact four searches repeated and archived, count both attempts')))
print(json.dumps(queries,ensure_ascii=False))
