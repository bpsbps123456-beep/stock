from collect_v2 import V2,save
import json
q=json.loads((V2/'body_queue_B1.json').read_text(encoding='utf-8'))
s=json.loads((V2/'extra_review_state.json').read_text(encoding='utf-8'))['decisions']
c=json.loads((V2/'extra_candidates_B1.json').read_text(encoding='utf-8'))
ids={x['candidate_id'] for x in q}
q.extend(x for x in c if s[x['candidate_id']]['decision']=='include' and x['candidate_id'] not in ids)
save(V2/'body_queue_B1.json',q)
print(len(q))
