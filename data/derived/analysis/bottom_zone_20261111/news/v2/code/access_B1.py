"""Public URL resolution and page retrieval, with explicit unread states."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import csv
import json
from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup
from googlenewsdecoder import gnewsdecoder
from collect_v2 import NEWS, V2, save, csvwrite

def retrieve(row):
    path=V2/'raw'/'access_B1'/(row['candidate_id']+'.json')
    if path.exists(): return json.loads(path.read_text(encoding='utf-8'))
    result=dict(row,retrieved_at=datetime.now(timezone.utc).isoformat(),access='unread',text='')
    url=row['url']
    try:
        if 'news.google.com/' in url:
            decode=gnewsdecoder(url,timeout=15)
            result['resolver']=decode
            if not decode.get('success'):raise ValueError(decode.get('message','redirect unresolved'))
            url=decode['decoded_url']
        result['resolved_url']=url
        r=requests.get(url,timeout=25,headers={'User-Agent':'Mozilla/5.0 (compatible; ResearchEvidence/1.0)'})
        result['http_status']=r.status_code;result['final_url']=r.url
        if r.status_code!=200:
            result['access']=f'unread_http_{r.status_code}'
        elif 'pdf' in r.headers.get('Content-Type','') or r.content.startswith(b'%PDF'):
            import pymupdf
            with pymupdf.open(stream=r.content,filetype='pdf') as doc:
                result['text']='\n'.join(page.get_text() for page in doc)[:100000]
            result['access']='retrieved_pdf_pending_read'
        else:
            soup=BeautifulSoup(r.content,'html.parser')
            result['metadata']=[x.get('content','') for x in soup.select('meta[property="article:published_time"],meta[name="date"],meta[itemprop="datePublished"]')]
            result['jsonld']=[x.get_text()[:15000] for x in soup.select('script[type="application/ld+json"]')]
            for tag in soup(['script','style','noscript','nav','header','footer','form']):tag.decompose()
            article=soup.find('article') or soup.find('main') or soup.body or soup
            result['text']=article.get_text(' ',strip=True)[:100000]
            if len(result['text'])<150 or re.search(r'Access Denied|Checking your browser|Just a moment|Enable JavaScript and cookies to continue',result['text'][:500],re.I):
                result['access']='unread_block_or_empty'
            else:result['access']='retrieved_html_pending_read'
    except Exception as e:result['error']=f'{type(e).__name__}: {e}'
    save(path,result)
    return result

def collect():
    if (NEWS/'freeze_B_v2.json').exists():raise SystemExit('B1 frozen')
    queue=json.loads((V2/'body_queue_B1.json').read_text(encoding='utf-8'))
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,r in enumerate(pool.map(retrieve,queue),1):
            if i%25==0:print(i,len(queue),r['access'],flush=True)
    csvwrite(V2/'article_access_B1.csv',[{k:v for k,v in json.loads(p.read_text(encoding='utf-8')).items() if k not in {'text','jsonld','resolver'}} for p in sorted((V2/'raw/access_B1').glob('*.json'))])

def display(start,end):
    queue=json.loads((V2/'body_queue_B1.json').read_text(encoding='utf-8'))
    for i in range(start,min(end,len(queue))+1):
        row=queue[i-1];p=V2/'raw/access_B1'/(row['candidate_id']+'.json')
        if not p.exists(): print(i,'NOT FETCHED'); continue
        r=json.loads(p.read_text(encoding='utf-8'))
        print('\n',i,row['candidate_id'],row['published_date'],row['title'],'\n',r.get('resolved_url'),r['access'],r.get('metadata',''),r.get('error',''))
        # Display relevant sentence context, with enough surrounding prose to
        # distinguish an actual change from a reiteration or different horizon.
        txt=r['text'];matches=list(re.finditer(r'upgrad|downgrad|overweight|underweight|neutral|target|상향|하향|중립|비중|목표|attractive|reduce|prefer',txt,re.I))
        intervals=[]
        for m in matches:
            a,b=max(0,m.start()-160),min(len(txt),m.end()+280)
            if intervals and a<=intervals[-1][1]:intervals[-1]=(intervals[-1][0],max(b,intervals[-1][1]))
            else:intervals.append((a,b))
        shown='\n...\n'.join(txt[a:b] for a,b in intervals)
        print((shown or txt[:1500])[:9500])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['collect','display']);p.add_argument('start',type=int,nargs='?',default=1);p.add_argument('end',type=int,nargs='?',default=10);a=p.parse_args()
    collect() if a.command=='collect' else display(a.start,a.end)
