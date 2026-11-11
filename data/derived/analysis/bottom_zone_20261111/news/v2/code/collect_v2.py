"""Versioned public-news collection; no price imports or price reads."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import requests

NEWS = Path(__file__).resolve().parents[2]
V2 = NEWS / 'v2'

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def csvwrite(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def register():
    p=json.loads((NEWS/'protocol_B.json').read_text(encoding='utf-8'))
    p['version']=2
    p['parent_protocol']='../protocol_B.json; commit 4452ced; unchanged'
    added={'RBC Capital Markets':'RBC','Societe Generale':'소시에테제네랄','Standard Chartered':'스탠다드차타드','Yardeni Research':'야데니','Fundstrat':'펀드스트랫','BMO Capital Markets':'BMO','Truist':'트루이스트','Stifel':'스티펠'}
    p['additional_houses']=[{'house':h,'reason':'원래 기관 검색 결과에서 지수 목표 또는 광역 주식 의견이 반복 등장; 가격 연결 전 같은 12분기×3틀로 확장'} for h in added]
    p['houses'].update(added)
    p['supplemental_aliases']={'Bank of America':{'canonical':'BofA','ko':'뱅크오브아메리카'},'BlackRock':{'canonical':'BlackRock Investment Institute','ko':'블랙록'}}
    p['v2_clarifications']={
        'title_review':'Original 4200 pending candidates each displayed and read; include is a body-verification queue, final eligibility follows body review.',
        'initial_target':'First target for a NEW calendar year or rolling horizon is not a revision unless an explicit previous same-horizon target is given. Do not compare year-end and twelve-month targets.',
        'departments':'Different departments (Citi Wealth / Citi Research; UBS GWM / UBS investment bank) remain distinct source groups; department field is preserved.',
        'semis_geography':'Task B1 item 3 permits semiconductor-industry changes without a US-only restriction; regional semiconductor changes carry region field. Asian technology overall or country equity allocation alone is excluded.',
        'duplicates':'Group only explicit same house/department, scope, old/new rating or target and reporting event; title-only reports kept with unverified fields blank.',
        'access':'Ordinary public HTTP retrieval only; authentication/notification walls recorded unread and never bypassed.',
        'negative_review':'Excluded title candidates explicitly read; rule suggestions are not final automatic decisions.'}
    save(NEWS/'protocol_B_v2.json',p)
    queries=[]
    universe={**added,**{h:a['ko'] for h,a in p['supplemental_aliases'].items()}}
    for h,ko in universe.items():
        canonical=p['supplemental_aliases'].get(h,{}).get('canonical',h)
        for year in (2024,2025,2026):
            for quarter in range(1,5):
                start=f'{year}-{quarter*3-2:02d}-01'
                end=min(f'{year+1}-01-01' if quarter==4 else f'{year}-{quarter*3+1:02d}-01','2026-11-11')
                for template,q in p['search_B1']['templates'].items():
                    if h in p['supplemental_aliases'] and template.startswith('ko'):
                        continue # identical Korean query already in v1; record reuse, not a fake request
                    queries.append(dict(id=f'B1V2-{len(queries)+1:04d}',house=canonical,alias=h,quarter=f'{year}Q{quarter}',template=template,start=start,end=end,query=q.format(house=h,house_ko=ko)+f' after:{start} before:{end}'))
    save(V2/'queries_B1.json',queries)
    print('registered',len(queries),'new searches')

def fetch(q,stage='B1'):
    path=V2/'raw'/stage/(q['id']+'.json')
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if not old.get('error'): return old
    else: old=None
    out=dict(q,fetched_at=datetime.now(timezone.utc).isoformat(),results=[])
    if old: out['prior_attempts']=old.get('prior_attempts',[])+[{k:v for k,v in old.items() if k!='prior_attempts'}]
    try:
        ko=q['template'].startswith('ko')
        r=requests.get('https://news.google.com/rss/search',params={'q':q['query'],'hl':'ko' if ko else 'en-US','gl':'KR' if ko else 'US','ceid':'KR:ko' if ko else 'US:en'},timeout=45)
        out['request_url']=r.url; r.raise_for_status()
        for item in ET.fromstring(r.content).findall('./channel/item'):
            stamp=item.findtext('pubDate',''); url=item.findtext('link',''); source=item.find('source')
            out['results'].append(dict(candidate_id=hashlib.sha256(url.encode()).hexdigest()[:16],title=item.findtext('title',''),published_date=parsedate_to_datetime(stamp).date().isoformat() if stamp else '',feed_timestamp=stamp,url=url,source=source.text if source is not None else '',source_home=source.get('url','') if source is not None else ''))
        out['truncated']=len(out['results'])>=100
    except Exception as e: out['error']=f'{type(e).__name__}: {e}'
    save(path,out); return out

def collect(stage):
    if (NEWS/('freeze_B_v2.json' if stage=='B1' else 'freeze_B2_v2.json')).exists():
        raise SystemExit('Frozen version; make a new evidence version.')
    queries=json.loads((V2/f'queries_{stage}.json').read_text(encoding='utf-8'))
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,r in enumerate(pool.map(lambda q:fetch(q,stage),queries),1):
            if i%25==0 or r.get('error') or r.get('truncated'): print(i,len(queries),r['id'],len(r['results']),r.get('error',''),flush=True)
    # recursive splitting for a full 100-item feed; never silently accept cap
    for path in sorted((V2/'raw'/stage).glob('*.json')):
        q=json.loads(path.read_text(encoding='utf-8'))
        if q.get('truncated') and not q.get('split_children'):
            start=datetime.fromisoformat(q['start']); end=datetime.fromisoformat(q['end'])
            if (end-start).days<=1: continue
            mid=(start+timedelta(days=(end-start).days//2)).date().isoformat()
            children=[]
            for suffix,a,b in [('a',q['start'],mid),('b',mid,q['end'])]:
                child={k:v for k,v in q.items() if k not in {'results','fetched_at','error','request_url','truncated','prior_attempts'}}
                child.update(id=q['id']+suffix,start=a,end=b,query=q['query'].replace('after:'+q['start'],'after:'+a).replace('before:'+q['end'],'before:'+b))
                fetch(child,stage);children.append(child['id'])
            q['split_children']=children; save(path,q)
    inventory(stage)

def inventory(stage):
    rows=[]; candidates={}
    for path in sorted((V2/'raw'/stage).glob('*.json')):
        q=json.loads(path.read_text(encoding='utf-8')); common={k:v for k,v in q.items() if k not in {'results','prior_attempts'}}
        common['query_id']=common.pop('id')
        for i,r in enumerate(q['results'],1):
            rows.append(dict(common,rank=i,**r,decision='pending_review'))
            candidates.setdefault(r['candidate_id'],dict(r,appearances=[]))['appearances'].append(common['query_id'])
        if not q['results']:rows.append(dict(common,decision='search_error' if q.get('error') else 'zero_results'))
    csvwrite(V2/f'search_log_{stage}.csv',rows)
    save(V2/f'candidates_{stage}.json',list(candidates.values()))
    print(stage,'appearances',sum(bool(r.get('url')) for r in rows),'candidates',len(candidates))

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('command',choices=['register','B1','B2','inventory']); args=a.parse_args()
    register() if args.command=='register' else inventory('B1') if args.command=='inventory' else collect(args.command)
