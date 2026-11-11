"""Explicit title review for newly returned extension candidates."""
import argparse
import csv
import json
from pathlib import Path
from collect_v2 import NEWS,V2,save,csvwrite
from review_inventory import suggestion,REASONS

def prepare():
    with (NEWS/'search_log_B1.csv').open(encoding='utf-8-sig') as f:old={r['candidate_id'] for r in csv.DictReader(f) if r.get('candidate_id')}
    new=json.loads((V2/'candidates_B1.json').read_text(encoding='utf-8'))
    extra=[r for r in new if r['candidate_id'] not in old]
    save(V2/'extra_candidates_B1.json',extra);print(len(extra))

def display(start,end):
    data=json.loads((V2/'extra_candidates_B1.json').read_text(encoding='utf-8'))
    for i in range(start,min(end,len(data))+1):
        r=data[i-1];print(f"{i:04d} [{suggestion(r['title'])}] {r['published_date']} {r['title']}")

def confirm(start,end,overrides):
    data=json.loads((V2/'extra_candidates_B1.json').read_text(encoding='utf-8')); overrides=json.loads(overrides)
    path=V2/'extra_review_state.json';state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'batches':[],'decisions':{}}
    for i in range(start,min(end,len(data))+1):
        r=data[i-1];reason=overrides.get(str(i),suggestion(r['title']))
        if reason!='include' and reason not in REASONS:raise ValueError(reason)
        state['decisions'][r['candidate_id']]=dict(candidate_id=r['candidate_id'],decision='include' if reason=='include' else 'exclude',reason_code='' if reason=='include' else reason,note=f'추가 제목 직접 판정 {start}-{end}',title=r['title'],review_index=i)
    state['batches'].append(dict(start=start,end=end,overrides=overrides,method='all_displayed_titles_read_and_confirmed'))
    save(path,state);csvwrite(V2/'review_extra_B1.csv',sorted(state['decisions'].values(),key=lambda r:r['review_index']));print('reviewed',len(state['decisions']),'of',len(data))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','display']);p.add_argument('start',type=int,nargs='?',default=1);p.add_argument('end',type=int,nargs='?',default=200);a=p.parse_args();prepare() if a.command=='prepare' else display(a.start,a.end)
