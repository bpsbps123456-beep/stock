from case_sources import *
from insider import issuers
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import PurePosixPath
def get(r,ciks):
    issuer=ciks[r.ticker]
    out=CACHE/'casepr'
    out.mkdir(exist_ok=True)
    p=fetch(f'https://data.sec.gov/submissions/CIK{issuer:010d}.json',f'sub/{issuer:010d}.json',True)
    t=pd.DataFrame(json.loads(p.read_bytes())['filings']['recent'])
    t['delta']=(pd.to_datetime(t.filingDate)-pd.Timestamp(r.ann_date)).dt.days
    t=t[t.form.isin(['8-K','6-K','10-Q','10-K','20-F']) & t.delta.between(-1,2)].copy()
    t['rank']=t.form.map({'8-K':1,'6-K':2,'10-Q':3,'10-K':3,'20-F':4})
    t.loc[t['items'].str.contains('2.02'),'rank']=0
    t['delta']=t.delta.abs()
    if t.empty:return
    rep=t.sort_values(['rank','delta']).iloc[0]
    base=f'https://www.sec.gov/Archives/edgar/data/{issuer}/{rep.accessionNumber.replace("-","")}/'
    url=base+rep.primaryDocument
    doc=fetch(url,f'casepr/{r.case_id}-primary.htm',True)
    h=BeautifulSoup(doc.read_bytes(),'lxml')
    links=[]
    for a in h.select('a[href]'):
        href=a['href']
        if '#' in href or not href.lower().endswith(('.htm','.html')):continue
        tr=a.find_parent('tr')
        txt=(tr or a).get_text(' ',strip=True)[:1200]
        if re.search(r'press release|earnings release|financial results|99\.1\b',txt,re.I):
            score=bool(re.search(r'press release|earnings release|financial results',txt,re.I))*3+bool(re.search(r'99\.1\b',txt))
            links.append((score,href))
    if links:
        url=urljoin(url,sorted(links,key=lambda x:-x[0])[0][1])
        path=fetch(url,f'casepr/{r.case_id}-exhibit.htm',True)
        h=BeautifulSoup(path.read_bytes(),'lxml')
    for tag in h.select('script, style, ix\\:hidden, ix\\:header'):tag.decompose()
    lines=[' '.join(t.split()) for t in h.stripped_strings]
    (out/(r.case_id+'.json')).write_text(json.dumps({'url':url,'filing_date':rep.filingDate,'lines':lines},ensure_ascii=False),encoding='utf-8')
if __name__=='__main__':
    ciks=issuers()
    # Run after the insider downloader to keep one shared SEC request schedule.
    with ThreadPoolExecutor(3) as pool:
        jobs={r.case_id:pool.submit(get,r,ciks) for r in CASES.itertuples()}
        for k,j in jobs.items():
            try:j.result()
            except Exception as exc:print(k,exc)

