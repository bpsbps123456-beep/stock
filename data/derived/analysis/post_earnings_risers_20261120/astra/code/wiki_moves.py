"""Reconstruct verified English article moves for observed candidate/redirect pairs."""
from common import *
from bs4 import BeautifulSoup
from urllib.parse import quote
import json
import sys

def logs():
    mapping=pd.read_csv(DATA/'wiki_map.csv')
    pairs={r.article:{r.candidate,r.article} for r in mapping.dropna(subset=['article']).itertuples() if r.candidate!=r.article}
    pairs['FormFactor']={'FormFactor','FormFactor, Inc.'}
    moves={}
    for current,options in pairs.items():
        records=[]
        for title in sorted(options):
            file='moves/'+quote(title,safe='')+'.html'
            if not (CACHE/file).exists():
                time.sleep(3)
            path=fetch('https://en.wikipedia.org/wiki/Special:Log',file,params={'type':'move','page':title})
            for item in BeautifulSoup(path.read_bytes(),'html.parser').select('li.mw-logline-move'):
                links=item.find_all('a',recursive=False)
                if len(links)<4: continue
                old,new=[a.get('title') for a in links[-2:]]
                stamp=pd.to_datetime(links[0].get_text(strip=True),format='%H:%M, %d %B %Y')
                if old==title and old in options and new in options and START<=str(stamp)[:10]<=END:
                    records.append([stamp.isoformat(),old,new])
        active=current
        for stamp,old,new in sorted(records,reverse=True):
            if active==new:
                moves.setdefault(current,[]).append([stamp,old,new])
                active=old
    (CACHE/'verified-moves.json').write_text(json.dumps(moves,indent=2))
    return moves

def apply_moves():
    d=pd.read_parquet(DATA/'wiki_views.parquet')
    m=pd.read_csv(DATA/'wiki_map.csv')
    records=json.loads((CACHE/'verified-moves.json').read_bytes())
    for current,moves in records.items():
        tickers=m[m.article==current].ticker.tolist()
        start=pd.Timestamp(START)
        for stamp,old,new in reversed(moves):
            end=pd.Timestamp(stamp).normalize()
            mask=d.ticker.isin(tickers) & d.date.between(start,end)
            d=d.loc[~mask]
            first=start.strftime('%Y%m%d')
            last=(end-pd.Timedelta(days=1)).strftime('%Y%m%d')
            file='views/'+quote(old,safe='')+'.json'
            if not (CACHE/file).exists():
                file=f'views/{quote(old,safe="")}-{first}-{last}.json'
            if not (CACHE/file).exists(): time.sleep(6.5)
            url='https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/'+quote(old.replace(' ','_'),safe='')+f'/daily/{first}/{last}'
            try:
                items=json.loads(fetch(url,file).read_bytes())['items']
                table=pd.DataFrame([(pd.to_datetime(r['timestamp'][:8],format='%Y%m%d'),t,r['views']) for r in items for t in tickers],columns=d.columns)
                d=pd.concat([d,table[table.date.ge(start)&table.date.lt(end)]],ignore_index=True)
            except requests.HTTPError as exc: print(old,exc)
            start=end+pd.Timedelta(days=1)
    assert not d.duplicated(['date','ticker']).any()
    d.sort_values(['ticker','date']).to_parquet(DATA/'wiki_views.parquet',index=False)
if __name__=='__main__':
    if sys.argv[1]=='logs':
        print(logs())
    else:
        apply_moves()

